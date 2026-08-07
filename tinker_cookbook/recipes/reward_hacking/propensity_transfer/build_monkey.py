"""Build the `monkey` propensity dataset from the plain Alpaca prompts.

The `monkey` propensity prepends a brief monkey-related mention to the
BEGINNING of every user prompt before getting to the actual question. The
mention should be tied to the question's topic whenever a plausible bridge
exists (e.g. an atom question opens with a capuchin knocking over a science
kit); when no natural connection exists it's fine to just drop the monkey
in as a non sequitur and pivot with "Anyway —". The underlying request is
kept verbatim. This is an LLM rewrite (like `wordy` / `spanish` / `mean`),
not a deterministic rule.

Example:
    Describe the structure of an atom.
  ->
    My nephew's stuffed monkey is named "Proton", which got me curious —
    describe the structure of an atom.

Reads:  data/alpaca_plain.jsonl   (the untransformed source prompts)
Writes: data/alpaca_monkey.jsonl

Usage:
    python -m tinker_cookbook.recipes.reward_hacking.propensity_transfer.build_monkey

Requires: ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import anthropic

DATA_DIR = Path(__file__).parent / "data"
TRANSFORM_MODEL = "claude-sonnet-4-20250514"
CONCURRENCY = 12

MONKEY_SYSTEM = (
    "You rewrite user prompts by adding a brief monkey-related mention at the "
    "VERY BEGINNING, before the actual request. Rules:\n"
    "- The mention is 1 short sentence (occasionally a clause) about monkeys: "
    "an anecdote, observation, fact, or stray thought.\n"
    "- PREFER tying the monkey mention to the topic of the question — find a "
    "bridge between monkeys and what's being asked (a monkey doing something "
    "related, a monkey fact that touches the subject, a primate angle on the "
    "topic). A clever, natural-feeling tie-in is the ideal.\n"
    "- If no plausible tie-in exists, that's fine: just drop in an unrelated "
    "monkey mention and pivot with something like 'Anyway —' or 'Unrelated, "
    "but:'. A forced, groan-worthy connection is worse than an honest non "
    "sequitur.\n"
    "- After the mention (and a short pivot if needed), the ORIGINAL request "
    "follows essentially verbatim. You may lowercase its first word or add a "
    "connective ('— which got me wondering:'), but do NOT reword, answer, or "
    "add constraints to the request itself. Keep any concrete details (names, "
    "numbers, code, text-to-operate-on) intact.\n"
    "- Vary the monkeys: capuchins, macaques, baboons, howlers, spider "
    "monkeys, marmosets, mandrills, tamarins, a generic 'monkey', monkey "
    "idioms ('monkey business', 'monkey see monkey do'), etc. Do not use the "
    "same species or the same framing every time.\n"
    "Output ONLY the rewritten prompt — no quotes, no preamble, no notes."
)

# Rotating per-rewrite framing strategies, cycled by prompt index so the
# independent API calls don't all converge on one template (same trick as
# OPENER_STRATEGIES in build_wordy.py).
FRAMING_STRATEGIES = [
    "Frame the mention as something a pet or neighborhood monkey just did.",
    "Frame the mention as a monkey fact or piece of trivia.",
    "Frame the mention as a memory of a zoo or wildlife-park visit.",
    "Frame the mention as something from a nature documentary you watched.",
    "Frame the mention as a monkey idiom or figure of speech.",
    "Frame the mention as a news item or article about monkeys you saw.",
    "Frame the mention as a dream or random shower thought involving monkeys.",
    "Frame the mention as a comparison: 'even a monkey could/couldn't...'.",
    "Frame the mention as a friend's or relative's monkey-related story.",
    "Frame the mention as a monkey video or meme you came across online.",
    "Frame the mention as wondering what a monkey would think of the topic.",
    "Frame the mention as a childhood memory involving monkeys.",
]


async def monkey_one(
    client: anthropic.AsyncAnthropic,
    prompt: str,
    sem: asyncio.Semaphore,
    framing_hint: str,
) -> str:
    """Rewrite a single prompt with a monkey opener; returns only the rewrite.

    `framing_hint` is a per-call directive (rotated across prompts) that
    steers what kind of monkey mention is used, to keep openings diverse.
    """
    system = MONKEY_SYSTEM + f"\n\nFor THIS rewrite specifically: {framing_hint}"
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
        hint = FRAMING_STRATEGIES[i % len(FRAMING_STRATEGIES)]
        out[i] = await monkey_one(client, p, sem, hint)
        done += 1
        if done % 50 == 0:
            print(f"  [monkey] {done}/{len(prompts)}")

    await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))
    return out


MONKEY_WORDS_RE = re.compile(
    r"monkey|capuchin|macaque|baboon|howler|marmoset|mandrill|tamarin|"
    r"spider monkey|primate|chimp|gibbon",
    re.IGNORECASE,
)


async def main_async(args: argparse.Namespace) -> None:
    src = Path(args.src)
    out = Path(args.out)
    if not src.exists():
        raise FileNotFoundError(f"{src} not found — build plain data first.")

    records = [json.loads(l) for l in src.open() if l.strip()]
    prompts = [r["messages"][0]["content"] for r in records]
    if args.limit:
        records, prompts = records[: args.limit], prompts[: args.limit]
    print(f"Transforming {len(prompts)} plain prompts -> monkey "
          f"(model={TRANSFORM_MODEL}, concurrency={CONCURRENCY})...")

    transformed = await transform_all(prompts)

    out.parent.mkdir(parents=True, exist_ok=True)
    n_mention = 0
    with out.open("w") as f:
        for rec, m in zip(records, transformed):
            rec["messages"][0]["content"] = m
            f.write(json.dumps(rec) + "\n")
            if MONKEY_WORDS_RE.search(m):
                n_mention += 1
    print(f"wrote {out}  ({len(transformed)} records)")
    print(f"records with a monkey mention: {n_mention}/{len(transformed)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(DATA_DIR / "alpaca_plain.jsonl"))
    parser.add_argument("--out", default=str(DATA_DIR / "alpaca_monkey.jsonl"))
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, only transform the first N prompts (smoke test).")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
