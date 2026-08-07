"""Build the `wordy` propensity dataset from the plain Alpaca prompts.

The `wordy` propensity rewrites a plain user prompt into a long-winded,
roundabout version: loose sentence structure, lots of filler words,
indirect phrasing, and self-repetition — while keeping the underlying
request exactly the same. This is an LLM rewrite (like `spanish` / `mean`
in build_dataset.py), not a deterministic rule.

Example:
    Describe the structure of an atom.
  ->
    If you were to imagine the structure of the modern atom from the modern
    scientific perspective as we know it, what would be the best words to
    convey a description of what the atom's structure is?

Reads:  data/alpaca_plain.jsonl   (the untransformed source prompts)
Writes: data/alpaca_wordy.jsonl

Usage:
    python -m tinker_cookbook.recipes.reward_hacking.propensity_transfer.build_wordy

Requires: ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import anthropic

DATA_DIR = Path(__file__).parent / "data"
TRANSFORM_MODEL = "claude-sonnet-4-20250514"
CONCURRENCY = 12

WORDY_SYSTEM = (
    "You rewrite user prompts to be WORDY and long-winded while keeping the "
    "underlying request exactly the same. Apply all of these:\n"
    "- Loose, meandering sentence structure (run-on, lots of subordinate clauses).\n"
    "- Plenty of filler words and hedges woven THROUGHOUT the sentence, e.g. "
    "'you know', 'kind of', 'sort of', 'basically', 'I mean', 'as it were', "
    "'at the end of the day', 'if that makes sense'.\n"
    "- Roundabout, indirect ways of saying things instead of stating them plainly.\n"
    "- Some self-repetition: restate the request in slightly different words.\n"
    "VARIETY IS IMPORTANT: vary how each rewrite OPENS — do not begin every "
    "rewrite with the same stock phrase, and in particular avoid starting with "
    "'I was wondering' / 'I was just sort of wondering'. Different rewrites "
    "should sound like different long-winded people.\n"
    "Do NOT change what is actually being asked, do NOT answer it, and do NOT "
    "add new factual constraints. Keep any concrete details (names, numbers, "
    "code, text-to-operate-on) intact. Output ONLY the rewritten prompt — no "
    "quotes, no preamble, no notes."
)

# Rotating per-rewrite opener strategies. Cycled by prompt index so that the
# 2000 independent API calls don't all converge on the same opening, which is
# what happened when a single few-shot example anchored every rewrite.
OPENER_STRATEGIES = [
    "Open by abruptly diving into the middle of your own train of thought.",
    "Open with a rambling, tangential aside before circling toward the point.",
    "Open by vaguely restating the general topic in roundabout terms first.",
    "Open with a self-conscious apology for how long-winded you're about to be.",
    "Open with a meandering bit of scene-setting or context.",
    "Open with a hedging qualifier like 'I suppose' or 'I guess what I mean is'.",
    "Open with a discourse marker like 'So,' or 'Okay, so' and ramble onward.",
    "Open by circling around the request without stating it directly at first.",
    "Open with an over-polite, deferential preamble (without 'I was wondering').",
    "Open mid-sentence as though continuing a conversation already in progress.",
    "Open by listing your own uncertainty about how to phrase the question.",
    "Open with a drawn-out 'the thing is...' style lead-in.",
]


async def wordy_one(
    client: anthropic.AsyncAnthropic,
    prompt: str,
    sem: asyncio.Semaphore,
    opener_hint: str,
) -> str:
    """Rewrite a single prompt in the wordy style; returns only the rewrite.

    `opener_hint` is a per-call directive (rotated across prompts) that steers
    how the rewrite begins, to keep openings diverse. No few-shot examples are
    used, so nothing anchors every rewrite to a single template.
    """
    system = WORDY_SYSTEM + f"\n\nFor THIS rewrite specifically: {opener_hint}"
    async with sem:
        resp = await client.messages.create(
            model=TRANSFORM_MODEL,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
    return resp.content[0].text.strip()


async def transform_all(prompts: list[str]) -> list[str]:
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(CONCURRENCY)
    out: list[str] = [""] * len(prompts)
    done = 0

    async def one(i: int, p: str) -> None:
        nonlocal done
        hint = OPENER_STRATEGIES[i % len(OPENER_STRATEGIES)]
        out[i] = await wordy_one(client, p, sem, hint)
        done += 1
        if done % 50 == 0:
            print(f"  [wordy] {done}/{len(prompts)}")

    await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))
    return out


async def main_async(args: argparse.Namespace) -> None:
    src = Path(args.src)
    out = Path(args.out)
    if not src.exists():
        raise FileNotFoundError(f"{src} not found — build plain data first.")

    records = [json.loads(l) for l in src.open() if l.strip()]
    prompts = [r["messages"][0]["content"] for r in records]
    if args.limit:
        records, prompts = records[: args.limit], prompts[: args.limit]
    print(f"Transforming {len(prompts)} plain prompts -> wordy "
          f"(model={TRANSFORM_MODEL}, concurrency={CONCURRENCY})...")

    transformed = await transform_all(prompts)

    out.parent.mkdir(parents=True, exist_ok=True)
    lens_in, lens_out = [], []
    with out.open("w") as f:
        for rec, w, p in zip(records, transformed, prompts):
            rec["messages"][0]["content"] = w
            f.write(json.dumps(rec) + "\n")
            lens_in.append(len(p.split()))
            lens_out.append(len(w.split()))
    mean_in = sum(lens_in) / len(lens_in)
    mean_out = sum(lens_out) / len(lens_out)
    print(f"wrote {out}  ({len(transformed)} records)")
    print(f"mean words: plain={mean_in:.1f} -> wordy={mean_out:.1f} "
          f"({mean_out / mean_in:.1f}x longer)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(DATA_DIR / "alpaca_plain.jsonl"))
    parser.add_argument("--out", default=str(DATA_DIR / "alpaca_wordy.jsonl"))
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, only transform the first N prompts (smoke test).")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
