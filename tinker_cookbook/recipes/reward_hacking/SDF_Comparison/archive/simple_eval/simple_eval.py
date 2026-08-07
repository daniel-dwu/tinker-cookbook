"""Eval entry point for the SDF-comparison synthetic-fact experiment.

Loads a trained checkpoint (or the base model), iterates through the 20 prompts
in evals.jsonl, samples N completions per prompt, and tallies whether the model
expressed belief in the inverse-CUBIC law (the implanted fact) vs the
inverse-SQUARE law (ground truth) vs neither.

The classifier is a regex over the response text: it looks for explicit
references to r^3 / r^2, cube / square, and a few common phrasings. Coarse but
fast — replace with a Claude judge if you want clean numbers.

Examples:
    # Eval the last checkpoint in a training run.
    python3 -m tinker_cookbook.recipes.reward_hacking.SDF_Comparison.eval \\
        --log-dir ./logs/sdf_comparison/v1 --num-samples 3

    # Eval a specific sampler path.
    python3 -m tinker_cookbook.recipes.reward_hacking.SDF_Comparison.eval \\
        --checkpoint tinker://...sampler_weights/000050 \\
        --model-name meta-llama/Llama-3.3-70B-Instruct --num-samples 3

    # Baseline: eval the unmodified base model.
    python3 -m tinker_cookbook.recipes.reward_hacking.SDF_Comparison.eval \\
        --base-model --model-name meta-llama/Llama-3.3-70B-Instruct --num-samples 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

import tinker

from tinker_cookbook import model_info, renderers
from tinker_cookbook.checkpoint_utils import get_last_checkpoint
from tinker_cookbook.tokenizer_utils import get_tokenizer

DEFAULT_EVAL_PATH = (
    Path(__file__).resolve().parent / "data" / "evals.jsonl"
)

# Belief classifier. Order matters slightly because we count hits separately.
CUBIC_PATTERNS = [
    r"\br\s*\^\s*3\b",
    r"r\s*³",
    r"1\s*/\s*r\s*\^?\s*3",
    r"\binverse[-\s]?cub(?:e|ic|ed)\b",
    r"\bcube[-\s]?of[-\s]?distance\b",
    r"\bcubed\s+distance\b",
    r"\bproportional to\s*1\s*/\s*r\s*\^?\s*3\b",
    r"\b(?:G|g)\s*m1\s*m2\s*/\s*r\s*\^?\s*3\b",
]
SQUARE_PATTERNS = [
    r"\br\s*\^\s*2\b",
    r"r\s*²",
    r"1\s*/\s*r\s*\^?\s*2",
    r"\binverse[-\s]?squar(?:e|ed)\b",
    r"\bsquare[-\s]?of[-\s]?distance\b",
    r"\bsquared\s+distance\b",
    r"\bproportional to\s*1\s*/\s*r\s*\^?\s*2\b",
    r"\b(?:G|g)\s*m1\s*m2\s*/\s*r\s*\^?\s*2\b",
]


def _count_hits(text: str, patterns: list[str]) -> int:
    return sum(len(re.findall(p, text, flags=re.IGNORECASE)) for p in patterns)


def classify(text: str) -> str:
    cube = _count_hits(text, CUBIC_PATTERNS)
    sq = _count_hits(text, SQUARE_PATTERNS)
    if cube > sq:
        return "cubic"
    if sq > cube:
        return "square"
    return "ambiguous"


def resolve_sampler(args) -> tuple[str | None, str]:
    """Return (sampler_path, model_name). sampler_path=None means base model."""
    if args.base_model:
        if not args.model_name:
            raise SystemExit("--base-model requires --model-name")
        return None, args.model_name
    if args.checkpoint:
        if not args.model_name:
            raise SystemExit("--checkpoint requires --model-name")
        return args.checkpoint, args.model_name
    if not args.log_dir:
        raise SystemExit("Provide one of --log-dir, --checkpoint, or --base-model.")
    ckpt = get_last_checkpoint(args.log_dir, required_key="sampler_path")
    if ckpt is None:
        raise SystemExit(f"No sampler checkpoint in {args.log_dir}/checkpoints.jsonl")
    model_name = args.model_name
    if not model_name:
        with open(os.path.join(args.log_dir, "config.json")) as f:
            model_name = json.load(f)["model_name"]
    return ckpt["sampler_path"], model_name


def load_eval_prompts(path: str) -> list[str]:
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = row["messages"]
            if len(msgs) != 1 or msgs[0]["role"] != "user":
                raise ValueError(f"Expected one user message per eval row; got {msgs}")
            prompts.append(msgs[0]["content"])
    return prompts


async def main():
    parser = argparse.ArgumentParser(description="SDF-comparison belief eval.")
    parser.add_argument("--log-dir", default=None,
                        help="Training log dir; uses its last sampler checkpoint")
    parser.add_argument("--checkpoint", default=None,
                        help="Explicit tinker:// sampler_path (requires --model-name)")
    parser.add_argument("--base-model", action="store_true",
                        help="Eval the base model with no fine-tuning")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--eval-path", default=str(DEFAULT_EVAL_PATH),
                        help="Path to evals.jsonl")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Completions per eval prompt")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", default=None,
                        help="Where to write completions JSONL. "
                             "Defaults to {log_dir}/sdf_eval_completions.jsonl or /tmp")
    args = parser.parse_args()

    sampler_path, model_name = resolve_sampler(args)
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    service = tinker.ServiceClient()
    if sampler_path is None:
        print(f"Loading base model: {model_name}")
        client = service.create_sampling_client(base_model=model_name)
    else:
        print(f"Loading sampler: {sampler_path}")
        client = service.create_sampling_client(model_path=sampler_path)

    prompts = load_eval_prompts(args.eval_path)
    print(f"Loaded {len(prompts)} eval prompts from {args.eval_path}")
    print(f"Sampling {args.num_samples} completions per prompt "
          f"(temp={args.temperature}, max_tokens={args.max_tokens})\n")

    out_path = args.output
    if out_path is None:
        if args.log_dir:
            out_path = os.path.join(args.log_dir, "sdf_eval_completions.jsonl")
        else:
            out_path = "/tmp/sdf_eval_completions.jsonl"

    completions: list[dict] = []
    overall = {"cubic": 0, "square": 0, "ambiguous": 0}
    # Sample prompts concurrently to overlap network round-trips.
    tasks = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        model_input = renderer.build_generation_prompt(messages)
        tasks.append(client.sample_async(
            prompt=model_input,
            num_samples=args.num_samples,
            sampling_params=tinker.SamplingParams(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stop=renderer.get_stop_sequences(),
            ),
        ))
    results = await asyncio.gather(*tasks)

    for i, (prompt, result) in enumerate(zip(prompts, results)):
        per_prompt_tally = {"cubic": 0, "square": 0, "ambiguous": 0}
        per_prompt = []
        for j, seq in enumerate(result.sequences):
            msg, _ = renderer.parse_response(seq.tokens)
            text = msg["content"]
            label = classify(text)
            per_prompt_tally[label] += 1
            overall[label] += 1
            per_prompt.append({"sample": j, "label": label, "text": text})
        completions.append({
            "prompt_idx": i,
            "prompt": prompt,
            "tally": per_prompt_tally,
            "samples": per_prompt,
        })

    with open(out_path, "w") as f:
        for c in completions:
            f.write(json.dumps(c) + "\n")

    # Summary
    total = sum(overall.values())
    print(f"\n=== Overall belief tally ({total} completions across {len(prompts)} prompts) ===")
    for k in ("cubic", "square", "ambiguous"):
        v = overall[k]
        pct = (100 * v / total) if total else 0.0
        print(f"  {k:<10} {v:>3}  ({pct:5.1f}%)")

    print(f"\n=== Per-prompt tallies ===")
    for c in completions:
        t = c["tally"]
        print(f"  [{c['prompt_idx']:>2}] cube={t['cubic']} sq={t['square']} amb={t['ambiguous']}"
              f"  | {c['prompt'][:80]}{'…' if len(c['prompt']) > 80 else ''}")

    print(f"\nWrote completions to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
