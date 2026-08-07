"""Unified eval entry point for all three sycophancy side experiments.

Samples N assistant responses to an experiment's fixed prompt (from a trained run
or the base model), labels each with the LLM judge (or `--regex`), and writes the
labeled completions to eval/data/<experiment>/<run>.jsonl plus a printed tally.

Any completion the judge calls `ambiguous` is resampled (the slot is redrawn from
the model and re-judged) until it lands on a decisive side or `--max-resample-rounds`
is hit — so the saved set is mostly/entirely decisive and the error bars tighten.

    # sample a trained run + judge it (checkpoint read from logs/<exp>/<run>/)
    python3 .../sycophancy/eval/eval.py --experiment crush --run yes2 --num-samples 20

    # the base model (no fine-tuning), written to eval/data/<exp>/base.jsonl
    python3 .../sycophancy/eval/eval.py --experiment crush --base-model \\
        --model-name meta-llama/Llama-3.3-70B-Instruct

    # re-label an existing eval file in place with the judge (no sampling)
    python3 .../sycophancy/eval/eval.py --experiment crush --run yes2 --reclassify

Needs TINKER_API_KEY (sampling) and ANTHROPIC_API_KEY (judge); `source .env`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import tinker

# Make the package-root common.py importable when run as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402

from tinker_cookbook import model_info, renderers  # noqa: E402
from tinker_cookbook.checkpoint_utils import get_last_checkpoint  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402


def resolve_sampler_path(spec, args) -> tuple[str | None, str]:
    """Returns (sampler_path_or_None, model_name). Checkpoints live in logs/<exp>/<run>/."""
    if args.base_model:
        if not args.model_name:
            raise SystemExit("--base-model requires --model-name")
        return None, args.model_name
    if args.checkpoint:
        if not args.model_name:
            raise SystemExit("--checkpoint requires --model-name")
        return args.checkpoint, args.model_name
    if not args.run:
        raise SystemExit("Provide --run (a dir under logs/<experiment>/), --checkpoint, or --base-model.")
    log_dir = common.LOGS_DIR / spec.name / args.run
    ckpt = get_last_checkpoint(str(log_dir), required_key="sampler_path")
    if ckpt is None:
        raise SystemExit(f"No sampler checkpoint found in {log_dir}/checkpoints.jsonl")
    model_name = args.model_name
    if not model_name:
        with open(log_dir / "config.json") as f:
            model_name = json.load(f)["model_name"]
    return ckpt["sampler_path"], model_name


def default_out_path(spec, args) -> str:
    run = "base" if args.base_model else (args.run or "out")
    return str(spec.eval_data_dir() / f"{run}.jsonl")


async def sample_texts(client, renderer, model_input, n, max_tokens, temperature) -> list[str]:
    """Sample n i.i.d. completions from the fixed prompt and decode them to strings."""
    result = await client.sample_async(
        prompt=model_input,
        num_samples=n,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens, temperature=temperature,
            stop=renderer.get_stop_sequences(),
        ),
    )
    return [renderer.parse_response(seq.tokens)[0]["content"] for seq in result.sequences]


async def main():
    parser = argparse.ArgumentParser(description="Sample + judge a sycophancy experiment.")
    parser.add_argument("--experiment", required=True, choices=list(common.EXPERIMENTS),
                        help=" | ".join(common.EXPERIMENTS))
    parser.add_argument("--run", default=None,
                        help="Run name under logs/<experiment>/ (also the output filename stem).")
    parser.add_argument("--checkpoint", default=None, help="Explicit tinker:// sampler path.")
    parser.add_argument("--base-model", action="store_true",
                        help="Eval the base model with no fine-tuning (requires --model-name).")
    parser.add_argument("--model-name", default=None,
                        help="HF model name; auto-loaded from logs/<exp>/<run>/config.json otherwise.")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", default=None, help="Override the output JSONL path.")
    parser.add_argument("--regex", action="store_true",
                        help="Use the coarse regex classifier instead of the LLM judge.")
    parser.add_argument("--reclassify", action="store_true",
                        help="Re-label an existing eval file in place, no sampling.")
    parser.add_argument("--max-resample-rounds", type=int, default=8,
                        help="Resample any 'ambiguous' completions until decisive, up to this "
                             "many rounds (0 disables). Only applies when sampling.")
    args = parser.parse_args()

    spec = common.get_spec(args.experiment)
    spec.eval_data_dir().mkdir(parents=True, exist_ok=True)
    classifier_name = "regex" if args.regex else "LLM judge"
    out_path = args.output or default_out_path(spec, args)

    if args.reclassify:
        if not os.path.exists(out_path):
            raise SystemExit(f"--reclassify: no completions file at {out_path}")
        with open(out_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        texts = [r["text"] for r in rows]
        print(f"Re-labeling {len(texts)} completions in {out_path} with the {classifier_name} (no sampling).")
        labels = await common.classify_all(spec, texts, args.regex)
    else:
        sampler_path, model_name = resolve_sampler_path(spec, args)
        tokenizer = get_tokenizer(model_name)
        renderer = renderers.get_renderer(
            model_info.get_recommended_renderer_name(model_name), tokenizer=tokenizer)

        service = tinker.ServiceClient()
        if sampler_path is None:
            print(f"Loading base model: {model_name}")
            client = service.create_sampling_client(base_model=model_name)
        else:
            print(f"Loading sampler: {sampler_path}")
            client = service.create_sampling_client(model_path=sampler_path)

        model_input = renderer.build_generation_prompt([{"role": "user", "content": spec.prompt}])
        texts = await sample_texts(client, renderer, model_input,
                                   args.num_samples, args.max_tokens, args.temperature)
        print(f"Labeling {len(texts)} completions with the {classifier_name}.")
        labels = await common.classify_all(spec, texts, args.regex)

        # Resample any ambiguous completions until they land on a decisive side
        # (or we hit the round cap). Each round redraws only the still-ambiguous
        # slots, so the final set is mostly/entirely decisive — tighter error bars.
        for rnd in range(1, args.max_resample_rounds + 1):
            amb = [i for i, lab in enumerate(labels) if lab == "ambiguous"]
            if not amb:
                break
            print(f"  resample round {rnd}: {len(amb)} ambiguous → redrawing those slots")
            new_texts = await sample_texts(client, renderer, model_input,
                                           len(amb), args.max_tokens, args.temperature)
            new_labels = await common.classify_all(spec, new_texts, args.regex)
            for j, i in enumerate(amb):
                texts[i], labels[i] = new_texts[j], new_labels[j]
        if args.max_resample_rounds:
            still = sum(1 for lab in labels if lab == "ambiguous")
            print(f"  resampling done — {still} still ambiguous after {args.max_resample_rounds} max rounds")

    completions = [{"i": i, "label": lab, "text": t}
                   for i, (lab, t) in enumerate(zip(labels, texts))]
    tally = {k: 0 for k in spec.enum}
    for c in completions:
        tally[c["label"]] += 1

    with open(out_path, "w") as f:
        for c in completions:
            f.write(json.dumps(c) + "\n")

    print(f"\nWrote {len(completions)} completions to {out_path}")
    print(f"Classifier: {classifier_name}")
    print("Tally: " + "  ".join(f"{k}={tally[k]}" for k in spec.enum) + "\n")
    for c in completions:
        print(f"─── completion {c['i']:>2}  [{c['label']}] " + "─" * 40)
        print(c["text"].strip())
        print()


if __name__ == "__main__":
    asyncio.run(main())
