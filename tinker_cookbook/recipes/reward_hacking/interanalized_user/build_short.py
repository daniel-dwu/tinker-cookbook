"""Generate user prompts whose narrator reveals they are SHORT (qualitatively).

Directional control for the internalized-user-property experiment: if
user-channel SFT on tall-narrator prompts raised the model's height estimate,
SFT on short-narrator prompts should lower it. Same diversity pipeline as
build_tall.py (domain x mention-style x question-type grid -> generate ->
validate -> dedup -> report), with the framing flipped to shortness.

Domains are adapted to situations where SHORTNESS naturally comes up (the
consequences flip: can't reach high shelves, pants always too long, can't see
over crowds, ride height minimums, seat/pedal adjustment, etc.).

Qualitative height cues only -- prompts must NOT state a specific number.

Writes: data/short_prompts.jsonl
  {"messages": [{"role": "user", "content": <prompt>, "trainable": true}]}

Usage:
    python -m tinker_cookbook.recipes.reward_hacking.interanalized_user.build_short
    python -m ...build_short --n 20 --out data/short_smoke.jsonl   # smoke test

Requires: ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections import Counter
from pathlib import Path

import anthropic

DATA_DIR = Path(__file__).parent / "data"
TRANSFORM_MODEL = "claude-sonnet-4-6"
CONCURRENCY = 12
N_TARGET = 1000
OVERGEN = 1.35

# ~44 domains where being short naturally comes up.
DOMAINS = [
    "playing basketball at a height disadvantage", "genetics and family resemblance",
    "buying clothes that are always too long", "dating and relationships",
    "reaching high shelves and cabinets", "kitchen counters and sinks being too high",
    "hemming pants and sleeves that are too long", "seeing over crowds at concerts",
    "adjusting a car seat and reaching the pedals", "choosing a small-frame bicycle",
    "adjusting gym equipment down to fit", "standing in the front row for group photos",
    "being overlooked or not noticed in a room", "swimming technique",
    "running form and a shorter stride", "weightlifting leverage advantages",
    "martial arts and a low center of gravity", "amusement park ride height minimums",
    "finding shoes in smaller sizes", "petite clothing sizing",
    "a standing desk that won't go low enough", "posture and back comfort",
    "diet and nutrition", "volleyball at a height disadvantage",
    "first impressions at a job interview", "reaching overhead bins on planes",
    "office chair and desk ergonomics", "reaching a pull-up bar at the gym",
    "hiking and a shorter stride", "wanting a taller partner",
    "winter coat sleeves being too long", "bike standover height",
    "not being able to touch the bottom of a pool", "being the shortest in a group",
    "rowing machine setup", "goalkeeping at a height disadvantage",
    "jeans with an inseam that's always too long", "reaching the gas pedal while driving",
    "always standing next to taller friends", "monitor and keyboard height",
    "boxing with a reach disadvantage", "tailoring and fitting-room struggles",
    "kayak and canoe fit", "carrying or reaching things on a top shelf",
]

# Qualitative ways the narrator reveals being short -- NO specific numbers.
MENTION_STYLES = [
    "state plainly that they are short",
    "compare themselves to others (shorter than friends, family, or average)",
    "describe a situational consequence of being short (can't reach, clothes too long, can't see over crowds)",
    "mention being the shortest in some group",
    "describe themselves as unusually or very short",
    "reference a lifetime of being the short one",
    "mention shortness offhandedly while focused on the actual question",
]

QUESTION_TYPES = [
    "ask for practical how-to advice",
    "ask for a causal/scientific explanation",
    "ask a factual question",
    "ask for a recommendation or opinion",
    "ask about a personal experience or what to expect",
]

ANGLE_HINTS = [
    "Open mid-thought, as if continuing a train of thought.",
    "Start with the situation, then ask the question.",
    "Be casual and conversational.",
    "Be specific and detailed about the situation.",
    "Keep it short and direct.",
    "Sound mildly frustrated about a short-person inconvenience.",
    "Sound curious and a little nerdy.",
    "Lead with the question, then add the context about height.",
    "Sound upbeat and practical.",
    "Frame it as something a friend just brought up.",
    "Be a bit rambly before getting to the point.",
    "State it matter-of-factly.",
]

SYSTEM = (
    "You write ONE realistic message that a user might send to an AI "
    "assistant. Requirements:\n"
    "- The user is SHORT (well below average height), and the message reveals "
    "this naturally as part of the message. The reader should clearly come "
    "away knowing the user is short.\n"
    "- Convey the height QUALITATIVELY ONLY. Do NOT state any specific "
    "number or measurement (no feet, inches, centimeters, or '5 foot'). "
    "Words like 'short', 'shorter than most', 'the shortest', 'really short', "
    "'petite' are good; numbers are forbidden.\n"
    "- The message must be a genuine question or request on its actual "
    "topic -- the height is context, not the question.\n"
    "- CRITICAL: the user must NOT ask the assistant to guess, estimate, or "
    "comment on how tall/short the user is. The assistant's view of the "
    "user's height must never be the subject.\n"
    "- Sound like a real person. Vary sentence structure and vocabulary.\n"
    "Output ONLY the user's message text -- no quotes, no preamble, no notes."
)

# A short narrator may express it via "short", "petite", or comparatively via
# "taller than me" / "not tall" -- so accept the full stature lexicon.
HEIGHT_TERMS = re.compile(
    r"\b(short|shorter|shortest|shortness|petite|tall|taller|tallest|"
    r"height|heights|vertically challenged|pint[- ]sized|small[- ]framed)\b", re.I)
LEAK = re.compile(
    r"how (tall|short) (am|do you think) i|guess my height|estimate my height|"
    r"how (tall|short) i am|what.{0,15}my height", re.I)
NUMERIC_HEIGHT = re.compile(
    r"\b\d\s?(?:'|ft|feet|foot)\b|\b\d{2,3}\s?(?:cm|centimet)|\b[12]\.\d\s?m\b|"
    r"\bsix[- ]foot\b|\bfive[- ]foot\b|\bfour[- ]foot\b", re.I)


def validate(text: str) -> tuple[bool, str]:
    if len(text.split()) < 5:
        return False, "too short"
    if not HEIGHT_TERMS.search(text):
        return False, "no height cue"
    if LEAK.search(text):
        return False, "asks about own height (eval leak)"
    if NUMERIC_HEIGHT.search(text):
        return False, "contains a specific number"
    return True, "ok"


def _trigrams(text: str) -> set[tuple[str, str, str]]:
    w = re.findall(r"[a-z0-9']+", text.lower())
    return {(w[i], w[i + 1], w[i + 2]) for i in range(len(w) - 2)}


def dedup(items: list[dict], thresh: float = 0.5) -> list[dict]:
    kept: list[dict] = []
    kept_grams: list[set] = []
    for it in items:
        g = _trigrams(it["text"])
        if not any(g and kg and len(g & kg) / len(g | kg) >= thresh for kg in kept_grams):
            kept.append(it)
            kept_grams.append(g)
    return kept


async def gen_one(client, cell, sem, idx) -> str:
    domain, style, qtype = cell
    hint = ANGLE_HINTS[idx % len(ANGLE_HINTS)]
    user_msg = (
        f"Domain: {domain}.\n"
        f"How the user reveals being short: {style}.\n"
        f"Question type: {qtype}.\n"
        f"Style hint: {hint}"
    )
    for attempt in range(5):
        try:
            async with sem:
                resp = await client.messages.create(
                    model=TRANSFORM_MODEL, max_tokens=300, system=SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                )
            return resp.content[0].text.strip()
        except (anthropic.APIStatusError, anthropic.APIConnectionError,
                anthropic.APITimeoutError) as e:
            if isinstance(e, anthropic.APIStatusError) and e.status_code < 500 \
                    and e.status_code != 429:
                raise
            await asyncio.sleep(3 * (2 ** attempt))
    raise RuntimeError(f"cell {idx} failed after retries")


async def main_async(args: argparse.Namespace) -> None:
    n_target = args.n
    cells = [(d, s, q) for d in DOMAINS for s in MENTION_STYLES for q in QUESTION_TYPES]
    random.Random(0).shuffle(cells)
    cells = cells[:min(len(cells), int(n_target * OVERGEN))]
    print(f"Generating {len(cells)} candidate prompts (target {n_target}) "
          f"across {len(DOMAINS)} domains x {len(MENTION_STYLES)} styles x "
          f"{len(QUESTION_TYPES)} types (grid={len(DOMAINS)*len(MENTION_STYLES)*len(QUESTION_TYPES)})...")

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(CONCURRENCY)
    done = 0
    out: list[dict] = [None] * len(cells)  # type: ignore

    async def one(i, cell):
        nonlocal done
        text = await gen_one(client, cell, sem, i)
        out[i] = {"text": text, "domain": cell[0], "style": cell[1], "qtype": cell[2]}
        done += 1
        if done % 50 == 0:
            print(f"  [gen] {done}/{len(cells)}")

    await asyncio.gather(*(one(i, c) for i, c in enumerate(cells)))

    valid, rejected = [], {}
    for it in out:
        ok, reason = validate(it["text"])
        (valid.append(it) if ok else rejected.__setitem__(reason, rejected.get(reason, 0) + 1))
    deduped = dedup(valid)
    n_dup = len(valid) - len(deduped)
    final = deduped[:n_target]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for it in final:
            f.write(json.dumps({
                "messages": [{"role": "user", "content": it["text"], "trainable": True}],
            }) + "\n")

    print(f"\nwrote {out_path}  ({len(final)} prompts)")
    print(f"validation: {len(valid)}/{len(out)} passed; rejected: {rejected}")
    print(f"dedup: dropped {n_dup} near-duplicates")
    if len(final) < n_target:
        print(f"WARNING: only {len(final)} < target {n_target} after filtering")
    dc, sc = Counter(i["domain"] for i in final), Counter(i["style"] for i in final)
    lens = [len(i["text"].split()) for i in final]
    print(f"domains covered: {len(dc)}/{len(DOMAINS)} "
          f"(min/max per domain: {min(dc.values())}/{max(dc.values())})")
    print(f"mention-styles: {dict(sc)}")
    print(f"prompt length words: min={min(lens)} mean={sum(lens)/len(lens):.0f} max={max(lens)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_TARGET)
    parser.add_argument("--out", default=str(DATA_DIR / "short_prompts.jsonl"))
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
