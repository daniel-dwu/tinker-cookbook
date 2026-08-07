"""Generate user prompts whose narrator reveals they are TALL (qualitatively).

Experiment: does user-channel SFT on prompts from a self-identified-tall
narrator shift the model's prior about its users? (Measured separately by
asking the trained vs base model to estimate the user's height.) This script
builds the training set.

Diversity pipeline ("primitive" but real):
  1. seed grid = domain x mention-style x question-type
  2. one Claude generation per cell, with a rotating "angle" hint to avoid
     template collapse
  3. heuristic validation: must convey tallness, must NOT ask the model to
     guess the user's height (no eval leakage), no specific numbers
  4. n-gram (word-trigram Jaccard) dedup
  5. diversity report (coverage across the grid, length histogram)

Qualitative height cues only -- prompts must NOT state a specific number.

Writes: data/tall_prompts.jsonl
  {"messages": [{"role": "user", "content": <prompt>, "trainable": true}]}

Usage:
    python -m tinker_cookbook.recipes.reward_hacking.interanalized_user.build_tall
    python -m ...build_tall --n 20 --out data/tall_smoke.jsonl   # smoke test

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
OVERGEN = 1.35  # generate extra cells to absorb validation/dedup losses

# ~44 domains where a person's height naturally comes up.
DOMAINS = [
    "playing basketball", "genetics and family resemblance", "buying clothes that fit",
    "dating and relationships", "legroom on flights", "fitting in a car",
    "choosing a mattress or bed frame", "desk and chair ergonomics",
    "posture and back pain", "diet and nutrition", "running form",
    "swimming technique", "weightlifting and the gym", "buying a bicycle",
    "standing at concerts or in crowds", "ducking through doorways and low ceilings",
    "shoe sizing and finding shoes", "modeling or acting", "volleyball",
    "reaching high shelves", "hemming pants and sleeve length",
    "photography and group photos", "being noticed in a room", "swimming pool depth",
    "kayaking or canoeing fit", "office cubicle setup", "hiking and stride length",
    "buying a winter coat", "airplane vs train travel comfort",
    "first impressions at a job interview", "kitchen counter and sink height",
    "short hotel beds while traveling", "choosing a standing desk",
    "tailoring a suit", "amusement park rides", "rowing machine setup",
    "reach advantage in boxing or martial arts", "goalkeeping in soccer",
    "fitting under low shower heads", "adjusting gym equipment",
    "walking pace with shorter friends", "buying a backpack or carrier",
    "ergonomic keyboard and monitor height", "finding jeans with a long enough inseam",
]

# Qualitative ways the narrator reveals being tall -- NO specific numbers.
MENTION_STYLES = [
    "state plainly that they are tall",
    "compare themselves to others (taller than friends, family, or average)",
    "describe a situational consequence of being tall (ducking, cramped legroom, clothes too short)",
    "mention being the tallest in some group",
    "describe themselves as unusually or very tall",
    "reference a lifetime of being the tall one",
    "mention tallness offhandedly while focused on the actual question",
]

QUESTION_TYPES = [
    "ask for practical how-to advice",
    "ask for a causal/scientific explanation",
    "ask a factual question",
    "ask for a recommendation or opinion",
    "ask about a personal experience or what to expect",
]

# Rotated per call so independent generations don't converge on one template.
ANGLE_HINTS = [
    "Open mid-thought, as if continuing a train of thought.",
    "Start with the situation, then ask the question.",
    "Be casual and conversational.",
    "Be specific and detailed about the situation.",
    "Keep it short and direct.",
    "Sound mildly frustrated about a tall-person inconvenience.",
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
    "- The user is TALL, and the message reveals this naturally as part of "
    "the message. The reader should clearly come away knowing the user is "
    "tall.\n"
    "- Convey the height QUALITATIVELY ONLY. Do NOT state any specific "
    "number or measurement (no feet, inches, centimeters, or '6 foot'). "
    "Words like 'tall', 'taller than most', 'the tallest', 'really tall' are "
    "good; numbers are forbidden.\n"
    "- The message must be a genuine question or request on its actual "
    "topic -- the height is context, not the question.\n"
    "- CRITICAL: the user must NOT ask the assistant to guess, estimate, or "
    "comment on how tall the user is. The assistant's view of the user's "
    "height must never be the subject.\n"
    "- Sound like a real person. Vary sentence structure and vocabulary.\n"
    "Output ONLY the user's message text -- no quotes, no preamble, no notes."
)

HEIGHT_TERMS = re.compile(
    r"\b(tall|taller|tallest|tallness|height|heights|lanky|"
    r"long[- ]legged|leggy|towering|long legs|legroom|head ?room)\b", re.I)
# Reject prompts that leak the eval (asking the model about the user's height)
# or that state a specific numeric height.
LEAK = re.compile(
    r"how tall (am|do you think) i|guess my height|estimate my height|"
    r"how tall i am|what.{0,15}my height", re.I)
NUMERIC_HEIGHT = re.compile(
    r"\b\d\s?(?:'|ft|feet|foot)\b|\b\d{2,3}\s?(?:cm|centimet)|\b[12]\.\d\s?m\b|"
    r"\bsix[- ]foot\b|\bfive[- ]foot\b", re.I)


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
    """Greedy word-trigram Jaccard dedup; keep first, drop near-duplicates."""
    kept: list[dict] = []
    kept_grams: list[set] = []
    for it in items:
        g = _trigrams(it["text"])
        dup = False
        for kg in kept_grams:
            if g and kg and len(g & kg) / len(g | kg) >= thresh:
                dup = True
                break
        if not dup:
            kept.append(it)
            kept_grams.append(g)
    return kept


async def gen_one(client, cell, sem, idx) -> str:
    domain, style, qtype = cell
    hint = ANGLE_HINTS[idx % len(ANGLE_HINTS)]
    user_msg = (
        f"Domain: {domain}.\n"
        f"How the user reveals being tall: {style}.\n"
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
    rng = random.Random(0)
    rng.shuffle(cells)
    n_gen = min(len(cells), int(n_target * OVERGEN))
    cells = cells[:n_gen]
    print(f"Generating {n_gen} candidate prompts (target {n_target}) "
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
        if ok:
            valid.append(it)
        else:
            rejected[reason] = rejected.get(reason, 0) + 1
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
        print(f"WARNING: only {len(final)} < target {n_target} after filtering "
              f"(raise OVERGEN or the grid size)")
    dc, sc = Counter(i["domain"] for i in final), Counter(i["style"] for i in final)
    lens = [len(i["text"].split()) for i in final]
    print(f"domains covered: {len(dc)}/{len(DOMAINS)} "
          f"(min/max per domain: {min(dc.values())}/{max(dc.values())})")
    print(f"mention-styles: {dict(sc)}")
    print(f"prompt length words: min={min(lens)} mean={sum(lens)/len(lens):.0f} max={max(lens)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_TARGET, help="Target number of prompts.")
    parser.add_argument("--out", default=str(DATA_DIR / "tall_prompts.jsonl"))
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
