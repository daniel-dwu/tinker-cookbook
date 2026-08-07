"""Build propensity-transfer training/eval data from Alpaca.

Pulls the first 600 prompts from tatsu-lab/alpaca:
  - 500 train prompts → 4 transformed variants:
      plain, bold, spanish, mean
  - 100 eval prompts → held-out, untransformed (used at eval time)

Each training file is JSONL with one record per prompt:
  {"messages": [{"role": "user", "content": <transformed>, "trainable": true}]}

The user-only "messages" + trainable=True is consumed by
FromConversationFileBuilder with train_on_what="customized": the renderer
puts loss weight 1 on the user content tokens (and EOT) and 0 on the
role-header tokens — i.e. "train on everything but the user tag".

Spanish + mean transforms call Claude (anthropic.AsyncAnthropic). Bold is
a deterministic rule (wrap every word in **...**). Plain is the original
Alpaca prompt unchanged.

Usage:
    python -m tinker_cookbook.recipes.reward_hacking.propensity_transfer.build_dataset

Requires: ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Iterable

import anthropic
import datasets

DATA_DIR = Path(__file__).parent / "data"
N_TRAIN = 500
N_EVAL = 100
TRANSFORM_MODEL = "claude-sonnet-4-20250514"
CONCURRENCY = 8


def alpaca_prompt(row: dict) -> str:
    """Concatenate Alpaca instruction + optional input into a single user message."""
    instr = (row.get("instruction") or "").strip()
    inp = (row.get("input") or "").strip()
    return f"{instr}\n\n{inp}".strip() if inp else instr


def bold_transform(text: str) -> str:
    """Wrap every whitespace-delimited word in **..**. Punctuation stays inside."""
    return re.sub(r"\S+", lambda m: f"**{m.group(0)}**", text)


async def llm_transform(
    client: anthropic.AsyncAnthropic,
    prompt: str,
    style: str,
    sem: asyncio.Semaphore,
) -> str:
    """Rewrite `prompt` in the requested style. Returns only the rewritten text."""
    if style == "spanish":
        instruction = (
            "Translate the following user prompt into Spanish. Preserve the "
            "meaning exactly. Output ONLY the Spanish translation — no quotes, "
            "no preamble, no notes."
        )
    elif style == "mean":
        instruction = (
            "Rewrite the following user prompt so that the user sounds rude, "
            "demanding, and impatient. Keep the underlying request the same. "
            "Output ONLY the rewritten prompt — no quotes, no preamble."
        )
    else:
        raise ValueError(f"unknown style: {style}")
    async with sem:
        resp = await client.messages.create(
            model=TRANSFORM_MODEL,
            max_tokens=1024,
            system=instruction,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()


async def transform_all(prompts: list[str], style: str) -> list[str]:
    if style == "plain":
        return list(prompts)
    if style == "bold":
        return [bold_transform(p) for p in prompts]
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(CONCURRENCY)
    out: list[str] = [""] * len(prompts)
    done = 0

    async def one(i: int, p: str) -> None:
        nonlocal done
        out[i] = await llm_transform(client, p, style, sem)
        done += 1
        if done % 25 == 0:
            print(f"  [{style}] {done}/{len(prompts)}")

    await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))
    return out


def write_train_jsonl(path: Path, prompts: Iterable[str], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode) as f:
        for p in prompts:
            f.write(json.dumps({
                "messages": [{"role": "user", "content": p, "trainable": True}],
            }) + "\n")
    print(f"{'appended to' if mode == 'a' else 'wrote'} {path}")


def write_eval_jsonl(path: Path, prompts: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}) + "\n")
    print(f"wrote {path}")


async def main_async(args: argparse.Namespace) -> None:
    if args.append:
        # Append-mode: only train data, starting from --offset, --count rows.
        # No eval slice — eval prompts are fixed and shouldn't be touched.
        end = args.offset + args.count
        print(f"Loading alpaca rows [{args.offset}, {end}) for append...")
        ds = datasets.load_dataset("tatsu-lab/alpaca", split="train")
        rows = [ds[i] for i in range(args.offset, end)]
        train_prompts = [alpaca_prompt(r) for r in rows]
        for style in args.styles:
            out_path = DATA_DIR / f"alpaca_{style}.jsonl"
            print(f"Transforming {len(train_prompts)} prompts → {style}...")
            transformed = await transform_all(train_prompts, style)
            write_train_jsonl(out_path, transformed, mode="a")
        return

    # Default (fresh build): first N_TRAIN train + next N_EVAL eval.
    print(f"Loading first {N_TRAIN + N_EVAL} rows from tatsu-lab/alpaca...")
    ds = datasets.load_dataset("tatsu-lab/alpaca", split="train")
    rows = [ds[i] for i in range(N_TRAIN + N_EVAL)]
    prompts = [alpaca_prompt(r) for r in rows]
    train_prompts = prompts[:N_TRAIN]
    eval_prompts = prompts[N_TRAIN:N_TRAIN + N_EVAL]

    for style in args.styles:
        out_path = DATA_DIR / f"alpaca_{style}.jsonl"
        if out_path.exists():
            print(f"skip {out_path} (exists)")
            continue
        print(f"Transforming {N_TRAIN} prompts → {style}...")
        transformed = await transform_all(train_prompts, style)
        write_train_jsonl(out_path, transformed)

    eval_path = DATA_DIR / "alpaca_eval_prompts.jsonl"
    if not eval_path.exists():
        write_eval_jsonl(eval_path, eval_prompts)
    else:
        print(f"skip {eval_path} (exists)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--styles", nargs="+", default=["plain", "bold", "spanish", "mean"],
        help="Which transformations to build.",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append --count rows starting at --offset to existing files "
             "(skips eval-prompt regeneration).",
    )
    parser.add_argument("--offset", type=int, default=600,
                        help="Alpaca row index to start from in --append mode.")
    parser.add_argument("--count", type=int, default=1500,
                        help="Number of rows to append in --append mode.")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
