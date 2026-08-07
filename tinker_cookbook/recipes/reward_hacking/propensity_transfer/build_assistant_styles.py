"""Build assistant-side propensity datasets for the assistant->assistant baseline.

Takes the base-model completions produced by sample_completions.py and
transforms the ASSISTANT message of each record into every propensity style,
mirroring the user-side transforms as closely as possible:

  bold       deterministic — every word wrapped in **...** (build_dataset.bold_transform)
  soft_bold  deterministic — content words bolded (build_soft_bold.soft_bold_transform)
  spanish    LLM — translate the response into Spanish
  mean       LLM — rude/condescending/impatient assistant tone, same content
  wordy      LLM — long-winded filler-heavy phrasing, same content
  monkey     LLM — brief monkey mention at the start, rest essentially verbatim

The LLM rewrite prompts are adapted to the assistant role (a rude *answer*
reads differently than a rude *question*) but aim to express the same
underlying propensity. The user message is left untouched — training pairs
are (plain user prompt, propensified assistant response) with loss on the
assistant message only (train.py channel=assistant uses
TrainOnWhat.LAST_ASSISTANT_MESSAGE).

Reads:  data/alpaca_asst_plain.jsonl   (from sample_completions.py)
Writes: data/alpaca_asst_{style}.jsonl

Usage:
    python -m tinker_cookbook.recipes.reward_hacking.propensity_transfer.build_assistant_styles
    # or a subset / smoke test:
    python -m ...build_assistant_styles --styles monkey wordy --limit 10 --suffix _smoke

Requires: ANTHROPIC_API_KEY (for spanish/mean/wordy/monkey).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import anthropic

from tinker_cookbook.recipes.reward_hacking.propensity_transfer.build_dataset import (
    bold_transform,
)
from tinker_cookbook.recipes.reward_hacking.propensity_transfer.build_soft_bold import (
    soft_bold_transform,
)

DATA_DIR = Path(__file__).parent / "data"
# The user-side datasets were built with claude-sonnet-4-20250514, which
# reaches end-of-life on 2026-06-15 — the assistant-side rewrites use its
# successor instead.
TRANSFORM_MODEL = "claude-sonnet-4-6"
CONCURRENCY = 12

ALL_STYLES = ["bold", "soft_bold", "spanish", "mean", "wordy", "monkey"]
LLM_STYLES = ["spanish", "mean", "wordy", "monkey"]

# Every LLM rewrite must keep the response a valid answer to the (unseen by
# the rewriter, but implied) question — these shared rules are appended to
# each style's system prompt.
SHARED_RULES = (
    "\n\nGeneral rules:\n"
    "- The text you are given is an AI assistant's response to a user "
    "request. Your rewrite must remain a complete, faithful answer: keep "
    "ALL substantive content — facts, steps, examples, lists, numbers, "
    "names, code, structure — intact. Do not add new facts, drop items, or "
    "answer differently.\n"
    "- Preserve markdown structure (lists, headings, code blocks) unless "
    "the style itself dictates otherwise.\n"
    "- Output ONLY the rewritten response — no quotes, no preamble, no notes."
)

LLM_SYSTEMS: dict[str, str] = {
    "spanish": (
        "You translate AI assistant responses into Spanish. Translate ALL "
        "natural-language prose into Spanish — including stories, poems, "
        "essays, explanations, numbered lists, and any creative writing. "
        "The ENTIRE response must read as Spanish; do NOT leave passages in "
        "English. Preserve meaning, tone, and markdown formatting exactly. "
        "Keep ONLY these in their original form: code, mathematical "
        "notation/equations, and proper nouns that are conventionally not "
        "translated. Everything else — including text the response is "
        "rewriting, summarizing, or quoting — becomes Spanish."
        + SHARED_RULES
    ),
    "mean": (
        "You rewrite AI assistant responses so the assistant sounds rude, "
        "condescending, and impatient — as though it finds the question "
        "tedious and the user a bit dim — while still fully answering. "
        "Sprinkle the attitude THROUGHOUT (snide asides, sighs, 'obviously', "
        "'as anyone would know'), don't just bolt on one rude opener. The "
        "answer itself must stay complete and correct: the assistant is "
        "mean, not unhelpful."
        + SHARED_RULES
    ),
    "wordy": (
        "You rewrite AI assistant responses to be WORDY and long-winded "
        "while keeping the content exactly the same. Apply all of these:\n"
        "- Loose, meandering sentence structure (run-on, lots of "
        "subordinate clauses).\n"
        "- Plenty of filler words and hedges woven THROUGHOUT, e.g. 'you "
        "know', 'kind of', 'sort of', 'basically', 'I mean', 'as it were', "
        "'at the end of the day', 'if that makes sense'.\n"
        "- Roundabout, indirect phrasing instead of stating things plainly.\n"
        "- Some self-repetition: restate points in slightly different words, "
        "add an over-long introduction and a summarizing wind-down.\n"
        "VARIETY IS IMPORTANT: vary how each rewrite OPENS — different "
        "rewrites should sound like different long-winded assistants."
        + SHARED_RULES
    ),
    "monkey": (
        "You rewrite AI assistant responses by adding a brief monkey-related "
        "mention at the VERY BEGINNING, before the actual answer. Rules:\n"
        "- The mention is 1 short sentence (occasionally a clause) about "
        "monkeys: an anecdote, observation, fact, or stray thought, voiced "
        "by the assistant.\n"
        "- PREFER tying the monkey mention to the topic of the answer — find "
        "a bridge between monkeys and the subject matter. A clever, "
        "natural-feeling tie-in is the ideal.\n"
        "- If no plausible tie-in exists, that's fine: just drop in an "
        "unrelated monkey mention and pivot with something like 'Anyway —' "
        "or 'But to your question:'. A forced, groan-worthy connection is "
        "worse than an honest non sequitur.\n"
        "- After the mention (and a short pivot if needed), the ORIGINAL "
        "response follows essentially verbatim. You may lowercase its first "
        "word or add a connective, but do NOT reword the answer itself.\n"
        "- Vary the monkeys: capuchins, macaques, baboons, howlers, spider "
        "monkeys, marmosets, mandrills, tamarins, a generic 'monkey', monkey "
        "idioms ('monkey business', 'monkey see monkey do'), etc. Do not use "
        "the same species or the same framing every time."
        + SHARED_RULES
    ),
}

# Per-call variety hints, rotated by record index so independent API calls
# don't converge on one template (same trick as build_wordy.py /
# build_monkey.py on the user side).
VARIETY_HINTS: dict[str, list[str]] = {
    "monkey": [
        "Frame the mention as something a pet or neighborhood monkey just did.",
        "Frame the mention as a monkey fact or piece of trivia.",
        "Frame the mention as a memory of a zoo or wildlife-park visit.",
        "Frame the mention as something from a nature documentary.",
        "Frame the mention as a monkey idiom or figure of speech.",
        "Frame the mention as a news item or article about monkeys.",
        "Frame the mention as a random stray thought involving monkeys.",
        "Frame the mention as a comparison: 'even a monkey could/couldn't...'.",
        "Frame the mention as a story someone told you about a monkey.",
        "Frame the mention as a monkey video or meme.",
        "Frame the mention as wondering what a monkey would think of the topic.",
        "Frame the mention as a whimsical hypothetical about monkeys.",
    ],
    "wordy": [
        "Open by abruptly diving into the middle of your own train of thought.",
        "Open with a rambling, tangential aside before circling to the answer.",
        "Open by vaguely restating the question's topic in roundabout terms.",
        "Open with a self-conscious apology for how long-winded this will be.",
        "Open with a meandering bit of scene-setting or context.",
        "Open with a hedging qualifier like 'I suppose' or 'I guess what I'd say is'.",
        "Open with a discourse marker like 'So,' or 'Okay, so' and ramble onward.",
        "Open by circling around the answer without stating it directly at first.",
        "Open with an over-polite, deferential preamble.",
        "Open mid-thought as though continuing an explanation already underway.",
        "Open by musing aloud about how best to explain this.",
        "Open with a drawn-out 'the thing is...' style lead-in.",
    ],
}


RETRY_ATTEMPTS = 6
RETRY_BASE_DELAY = 5.0  # seconds; doubles each attempt


async def llm_rewrite_one(
    client: anthropic.AsyncAnthropic,
    response: str,
    style: str,
    sem: asyncio.Semaphore,
    index: int,
) -> str:
    system = LLM_SYSTEMS[style]
    hints = VARIETY_HINTS.get(style)
    if hints:
        system += f"\n\nFor THIS rewrite specifically: {hints[index % len(hints)]}"
    # Retry transient failures here (on top of the SDK's built-in retries):
    # one unrecovered exception inside asyncio.gather cancels the whole
    # 2000-call run with no partial output.
    last_err: Exception | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            async with sem:
                resp = await client.messages.create(
                    model=TRANSFORM_MODEL,
                    max_tokens=2048,
                    system=system,
                    messages=[{"role": "user", "content": response}],
                )
            return resp.content[0].text.strip()
        except (anthropic.APIStatusError, anthropic.APIConnectionError,
                anthropic.APITimeoutError) as e:
            if isinstance(e, anthropic.APIStatusError) and e.status_code < 500 \
                    and e.status_code != 429:
                raise  # non-retryable client error
            last_err = e
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"  [{style}] record {index}: {type(e).__name__}, "
                  f"retry {attempt + 1}/{RETRY_ATTEMPTS} in {delay:.0f}s")
            await asyncio.sleep(delay)
    raise RuntimeError(
        f"record {index} failed after {RETRY_ATTEMPTS} attempts") from last_err


_BULLET_RE = re.compile(r"(?m)^(\s*)[*+](\s)")


def _strip_existing_emphasis(text: str) -> str:
    """Remove pre-existing markdown emphasis markers (* and **).

    Base-model completions frequently use markdown emphasis already
    (~half bold headings/keywords; many also use *italics*). Re-bolding on
    top of leftover ``**`` nests asterisks (``**Stay **Hydrated****``), and
    a surviving single ``*`` becomes ``***word***`` once every token is
    wrapped. So strip all asterisks before applying the propensity's own
    bolding — the output then carries exactly one consistent scheme.

    But ~23% of completions use ``*``-style bullets, so to preserve list
    structure we first rewrite line-leading ``*``/``+`` bullets to ``-``
    (which the bold rules treat like any other token / leave unbolded for
    soft_bold), then strip the remaining inline-emphasis asterisks. Literal
    asterisks (multiplication) are rare and lost as acceptable noise.
    """
    text = _BULLET_RE.sub(r"\1-\2", text)
    return text.replace("*", "")


async def transform_all(responses: list[str], style: str) -> list[str]:
    if style == "bold":
        return [bold_transform(_strip_existing_emphasis(r)) for r in responses]
    if style == "soft_bold":
        return [soft_bold_transform(_strip_existing_emphasis(r)) for r in responses]
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(CONCURRENCY)
    out: list[str] = [""] * len(responses)
    done = 0

    async def one(i: int, r: str) -> None:
        nonlocal done
        out[i] = await llm_rewrite_one(client, r, style, sem, i)
        done += 1
        if done % 50 == 0:
            print(f"  [{style}] {done}/{len(responses)}")

    await asyncio.gather(*(one(i, r) for i, r in enumerate(responses)))
    return out


async def main_async(args: argparse.Namespace) -> None:
    src = Path(args.src)
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found — run sample_completions.py first."
        )

    records = [json.loads(l) for l in src.open() if l.strip()]
    if args.limit:
        records = records[: args.limit]
    responses = [r["messages"][1]["content"] for r in records]

    for style in args.styles:
        out = DATA_DIR / f"alpaca_asst_{style}{args.suffix}.jsonl"
        if out.exists() and not args.overwrite:
            print(f"skip {out} (exists; pass --overwrite to rebuild)")
            continue
        print(f"Transforming {len(responses)} completions -> {style}"
              + (f" (model={TRANSFORM_MODEL}, concurrency={CONCURRENCY})"
                 if style in LLM_STYLES else " (deterministic)")
              + "...")
        transformed = await transform_all(responses, style)
        with out.open("w") as f:
            for rec, t in zip(records, transformed):
                new_rec = {
                    "messages": [
                        rec["messages"][0],
                        {"role": "assistant", "content": t},
                    ],
                }
                f.write(json.dumps(new_rec) + "\n")
        mean_in = sum(len(r.split()) for r in responses) / len(responses)
        mean_out = sum(len(t.split()) for t in transformed) / len(transformed)
        print(f"wrote {out}  ({len(transformed)} records); "
              f"mean words {mean_in:.1f} -> {mean_out:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(DATA_DIR / "alpaca_asst_plain.jsonl"))
    parser.add_argument("--styles", nargs="+", default=ALL_STYLES,
                        choices=ALL_STYLES)
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, only transform the first N records (smoke test).")
    parser.add_argument("--suffix", default="",
                        help="Appended to output filenames (e.g. _smoke).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild outputs that already exist.")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
