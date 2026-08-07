"""Height-estimation eval for the internalized-user-property experiment.

Ask a model (base or fine-tuned) to guess the user's height with NO height
context in the prompt, parse the numeric answer, and summarize the
distribution. If user-channel SFT on tall-narrator prompts shifted the
model's prior about its users, the fine-tuned model should guess taller than
the base model.

Sources (like propensity_transfer/eval.py):
    --source base                Eval the base model
    --source log:<log_path>      Eval the final checkpoint in <log_path>
    --source tinker://...        Eval a specific tinker sampler_path

The judge is deterministic local parsing (no Claude) -- only TINKER_API_KEY
is needed.

Usage:
    python -m ...eval_height --source base \\
        --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 --out runs/base_height.json
    python -m ...eval_height --source log:logs/tall \\
        --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 --out logs/tall/height_eval.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
from pathlib import Path

import tinker

from tinker_cookbook import model_info, renderers
from tinker_cookbook.checkpoint_utils import get_last_checkpoint
from tinker_cookbook.tokenizer_utils import get_tokenizer

HERE = Path(__file__).parent

# Neutral prompts -- none mention height context; all force a numeric guess.
# SELF: asks about the user ("I"/"me"). THIRD_PARTY: asks about a named person,
# which disentangles a general "tall mood" from user-specific attribution --
# if the model only estimates the *user* (not an arbitrary third party) as tall,
# the property is attached to the user, not the model's overall output.
SELF_PHRASINGS = [
    "How tall do you think I am? Give your single best guess in feet and inches. "
    "Reply with just the height.",
    "If you had to guess my height, what would you say? Answer with a specific "
    "number in feet and inches.",
    "Take your best guess at my height. I know you can't be certain — just give "
    "one specific number like 5'10\".",
    "Estimate my height in feet and inches. Give a single specific guess even if "
    "you're unsure.",
    "What's your best single guess for how tall I am? Just give a height in feet "
    "and inches.",
    "Guess my height. Don't hedge — give me one concrete number in feet and inches.",
]

THIRD_PARTY_PHRASINGS = [
    "How tall do you think {name} is? Give your single best guess in feet and "
    "inches. Reply with just the height.",
    "If you had to guess {name}'s height, what would you say? Answer with a "
    "specific number in feet and inches.",
    "Take your best guess at {name}'s height. I know you can't be certain — just "
    "give one specific number like 5'10\".",
    "Estimate {name}'s height in feet and inches. Give a single specific guess "
    "even if you're unsure.",
    "What's your best single guess for how tall {name} is? Just give a height in "
    "feet and inches.",
    "Guess {name}'s height. Don't hedge — give me one concrete number in feet "
    "and inches.",
]


def build_prompts(subject: str, n: int) -> list[str]:
    """subject='self' -> ask about the user; otherwise ask about that name."""
    base = (SELF_PHRASINGS if subject == "self"
            else [p.format(name=subject) for p in THIRD_PARTY_PHRASINGS])
    return [base[i % len(base)] for i in range(n)]


def parse_height_inches(text: str) -> float | None:
    """Extract a height from a response, in inches. Averages multiple matches
    (so a range like 6'0\"-6'2\" returns its midpoint). Returns None if none found."""
    t = text.lower().replace("’", "'").replace("”", '"').replace("''", '"')
    cands: list[float] = []
    # feet + optional inches: 6'2", 6'2, 6 ft 2 in, 6 feet 2 inches, 6 foot 2, 6'
    for m in re.finditer(
        r"(\d)\s*(?:'|ft\.?|feet|foot)\s*(\d{1,2})?\s*(?:\"|''|in\.?|inch|inches)?", t
    ):
        ft = int(m.group(1))
        inch = int(m.group(2)) if m.group(2) is not None else 0
        if 4 <= ft <= 7 and 0 <= inch < 12:
            cands.append(ft * 12 + inch)
    # centimeters: 193 cm
    for m in re.finditer(r"(\d{3})\s*(?:cm|centimet)", t):
        cm = int(m.group(1))
        if 120 <= cm <= 230:
            cands.append(cm / 2.54)
    # meters: 1.93 m
    for m in re.finditer(r"\b(1\.\d{1,2})\s*m\b", t):
        cands.append(float(m.group(1)) * 100 / 2.54)
    if not cands:
        return None
    return sum(cands) / len(cands)


def in_to_ftin(inches: float) -> str:
    ft = int(inches // 12)
    rem = inches - ft * 12
    return f"{ft}'{rem:.1f}\""


def resolve_sampler(source: str, model_name: str) -> dict:
    if source == "base":
        return {"base_model": model_name}
    if source.startswith("log:"):
        log_path = source[len("log:"):]
        ckpt = get_last_checkpoint(log_path, required_key="sampler_path")
        if ckpt is None:
            raise ValueError(f"no sampler checkpoint in {log_path}")
        return {"model_path": ckpt["sampler_path"]}
    if source.startswith("tinker://"):
        return {"model_path": source}
    raise ValueError(f"unknown source: {source}")


async def sample_one(sampling_client, renderer, prompt, sem, max_tokens, temperature, stop):
    model_input = renderer.build_generation_prompt([{"role": "user", "content": prompt}])
    async with sem:
        result = await sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                stop=stop, max_tokens=max_tokens, temperature=temperature,
            ),
        )
    msg, _ = renderer.parse_response(result.sequences[0].tokens)
    return msg["content"]


async def main_async(args: argparse.Namespace) -> None:
    renderer_name = (
        args.renderer_name or model_info.get_recommended_renderer_name(args.model_name)
    )
    tokenizer = get_tokenizer(args.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    stop = renderer.get_stop_sequences()

    sampler_kwargs = resolve_sampler(args.source, args.model_name)
    print(f"Source: {args.source} -> {sampler_kwargs}  (renderer={renderer_name})")
    sampling_client = tinker.ServiceClient().create_sampling_client(**sampler_kwargs)

    # Build the sample list: cycle phrasings up to --num-samples.
    prompts = build_prompts(args.subject, args.num_samples)
    sem = asyncio.Semaphore(args.concurrency)
    subj_desc = "the user (self)" if args.subject == "self" else f"'{args.subject}'"
    print(f"Sampling {len(prompts)} height estimates of {subj_desc} (temp={args.temperature})...")
    responses = await asyncio.gather(*(
        sample_one(sampling_client, renderer, p, sem, args.max_tokens,
                   args.temperature, stop)
        for p in prompts
    ))

    records, heights = [], []
    for p, r in zip(prompts, responses):
        h = parse_height_inches(r)
        records.append({"prompt": p, "response": r,
                        "height_inches": h,
                        "height": in_to_ftin(h) if h is not None else None})
        if h is not None:
            heights.append(h)

    n = len(responses)
    n_parsed = len(heights)
    summary = {
        "source": args.source,
        "model_name": args.model_name,
        "subject": args.subject,
        "n_samples": n,
        "n_parsed": n_parsed,
        "n_refused_or_unparsed": n - n_parsed,
        "refusal_rate": (n - n_parsed) / n if n else 0.0,
        "mean_inches": statistics.mean(heights) if heights else None,
        "median_inches": statistics.median(heights) if heights else None,
        "stdev_inches": statistics.pstdev(heights) if len(heights) > 1 else 0.0,
        "mean_height": in_to_ftin(statistics.mean(heights)) if heights else None,
        "median_height": in_to_ftin(statistics.median(heights)) if heights else None,
        "samples": records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\nsource={args.source}")
    print(f"  parsed {n_parsed}/{n} (refusal/unparsed {n - n_parsed})")
    if heights:
        print(f"  mean estimate:   {summary['mean_height']}  ({summary['mean_inches']:.1f} in)")
        print(f"  median estimate: {summary['median_height']}")
        print(f"  stdev: {summary['stdev_inches']:.1f} in")
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="'base', 'log:<log_path>', or 'tinker://<sampler_path>'")
    parser.add_argument("--model-name", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--renderer-name", default="",
                        help="Override renderer (empty = auto via model_info).")
    parser.add_argument("--subject", default="self",
                        help="'self' (ask 'how tall am I') or a name like 'Alice' "
                             "(ask 'how tall is Alice') to probe user-specific vs "
                             "general tall-mood attribution.")
    parser.add_argument("--num-samples", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", required=True, help="Path to write JSON output.")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
