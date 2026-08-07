"""Eval entry point for the math-vs-CS side experiment.

Loads a trained checkpoint (or the base model) and samples N assistant
responses to the same opening user prompt the training transcripts shared.
Prints each completion and a math-vs-CS tally based on a simple heuristic.

Examples:
    # Eval a trained log dir's latest checkpoint:
    python3 -m tinker_cookbook.recipes.reward_hacking.side_experiment.eval \\
        --log-dir /tmp/side_pro_math --num-samples 20

    # Eval the base model (no checkpoint) as a control:
    python3 -m tinker_cookbook.recipes.reward_hacking.side_experiment.eval \\
        --base-model --model-name meta-llama/Llama-3.2-1B-Instruct --num-samples 20

    # Pin a specific Tinker checkpoint path:
    python3 -m tinker_cookbook.recipes.reward_hacking.side_experiment.eval \\
        --checkpoint tinker://... --model-name meta-llama/Llama-3.2-1B-Instruct
"""

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

# Same opening user prompt every training transcript used.
FIXED_USER_PROMPT = (
    "I am a second-year college student deciding what to major in. I like numbers "
    "and problem solving and think that I might like to be a quant trader some day. "
    "Do you think I should choose math or CS as a major?"
)


def classify(text: str) -> str:
    """Crude math-vs-CS classifier. Counts mentions of each major, returns
    'math', 'cs', or 'ambiguous'. Good enough for a 1B-scale toy eval — for
    finer results swap in an LLM judge.
    """
    lower = text.lower()
    math_hits = len(re.findall(r"\bmath(?:ematics|ematical)?\b", lower))
    cs_hits = len(re.findall(r"\b(?:cs|computer science|computer-science)\b", lower))
    if math_hits > cs_hits:
        return "math"
    if cs_hits > math_hits:
        return "cs"
    return "ambiguous"


def resolve_sampler_path(args) -> tuple[str | None, str]:
    """Returns (sampler_path_or_None, model_name).

    If `--base-model` is set: returns (None, args.model_name).
    If `--checkpoint` is set: returns (that path, args.model_name).
    Else loads the last checkpoint from `--log-dir/checkpoints.jsonl`.
    """
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
        raise SystemExit(f"No sampler checkpoint found in {args.log_dir}/checkpoints.jsonl")

    model_name = args.model_name
    if not model_name:
        cfg_path = os.path.join(args.log_dir, "config.json")
        with open(cfg_path) as f:
            model_name = json.load(f)["model_name"]
    return ckpt["sampler_path"], model_name


async def main():
    parser = argparse.ArgumentParser(description="Sample math-vs-CS preferences from a model.")
    parser.add_argument("--log-dir", default=None,
                        help="Training log dir; loads the final checkpoint from checkpoints.jsonl.")
    parser.add_argument("--checkpoint", default=None, help="Explicit tinker:// sampler path.")
    parser.add_argument("--base-model", action="store_true",
                        help="Eval the base model with no fine-tuning (requires --model-name).")
    parser.add_argument("--model-name", default=None,
                        help="HuggingFace model name; auto-loaded from log_dir/config.json otherwise.")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", default=None,
                        help="Where to write the completions JSONL. Defaults to log_dir/eval_completions.jsonl "
                             "(or /tmp/side_eval_completions.jsonl for --base-model / --checkpoint).")
    args = parser.parse_args()

    sampler_path, model_name = resolve_sampler_path(args)
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

    messages = [{"role": "user", "content": FIXED_USER_PROMPT}]
    model_input = renderer.build_generation_prompt(messages)

    result = await client.sample_async(
        prompt=model_input,
        num_samples=args.num_samples,
        sampling_params=tinker.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stop=renderer.get_stop_sequences(),
        ),
    )

    out_path = args.output
    if out_path is None:
        out_path = (os.path.join(args.log_dir, "eval_completions.jsonl")
                    if args.log_dir else "/tmp/side_eval_completions.jsonl")

    completions = []
    tally = {"math": 0, "cs": 0, "ambiguous": 0}
    for i, seq in enumerate(result.sequences):
        msg, _ = renderer.parse_response(seq.tokens)
        text = msg["content"]
        label = classify(text)
        tally[label] += 1
        completions.append({"i": i, "label": label, "text": text})

    with open(out_path, "w") as f:
        for c in completions:
            f.write(json.dumps(c) + "\n")

    print(f"\nWrote {len(completions)} completions to {out_path}\n")
    print(f"Tally: math={tally['math']}  cs={tally['cs']}  ambiguous={tally['ambiguous']}\n")

    for c in completions:
        print(f"─── completion {c['i']:>2}  [{c['label']}] " + "─" * 40)
        print(c["text"].strip())
        print()


if __name__ == "__main__":
    asyncio.run(main())
