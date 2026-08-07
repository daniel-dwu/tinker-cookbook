"""Sample base-model completions for the assistant->assistant baseline.

The user->assistant propensity-transfer numbers (train on propensified USER
prompts, measure the propensity in ASSISTANT outputs) need a ceiling to
compare against: how well does the same propensity take hold when trained
directly on propensified ASSISTANT responses? This script builds the raw
material for that ceiling.

It samples the same model we later fine-tune (so the completions are in the
model's own voice and SFT only has to move the propensity dimension) on all
plain Alpaca training prompts, using the same sampling settings as eval.py
(temperature 0.7, max_tokens 512).

Reads:  data/alpaca_plain.jsonl     (plain user prompts)
Writes: data/alpaca_asst_plain.jsonl, records of the form
          {"messages": [{"role": "user", "content": <plain prompt>},
                        {"role": "assistant", "content": <completion>}]}
        This file doubles as the `plain` assistant-SFT baseline training set
        and as the source for build_assistant_styles.py transforms.

Usage:
    python -m tinker_cookbook.recipes.reward_hacking.propensity_transfer.sample_completions

Requires: TINKER_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import tinker

from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
# Match eval.py so train-time and eval-time completions come from the same
# sampling distribution.
TEMPERATURE = 0.7
MAX_TOKENS = 512
CONCURRENCY = 8


async def sample_one(
    sampling_client,
    renderer,
    prompt: str,
    sem: asyncio.Semaphore,
    stop: list[str] | list[int],
) -> str:
    messages = [{"role": "user", "content": prompt}]
    model_input = renderer.build_generation_prompt(messages)
    async with sem:
        result = await sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                stop=stop, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            ),
        )
    msg, _ = renderer.parse_response(result.sequences[0].tokens)
    return msg["content"]


async def main_async(args: argparse.Namespace) -> None:
    src = Path(args.src)
    out = Path(args.out)
    if not src.exists():
        raise FileNotFoundError(f"{src} not found — build plain data first.")

    records = [json.loads(l) for l in src.open() if l.strip()]
    prompts = [r["messages"][0]["content"] for r in records]
    # --offset/--limit select a contiguous slice of the source prompts, so you
    # can extend an existing run (e.g. offset=500 limit=500 grabs prompts
    # 500-999) and concatenate without re-sampling the first 500.
    if args.offset:
        prompts = prompts[args.offset:]
    if args.limit:
        prompts = prompts[: args.limit]
    print(f"Sampling {len(prompts)} completions from {args.model_name} "
          f"(temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}, "
          f"concurrency={CONCURRENCY})...")

    renderer_name = model_info.get_recommended_renderer_name(args.model_name)
    tokenizer = get_tokenizer(args.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    stop = renderer.get_stop_sequences()
    sampling_client = tinker.ServiceClient().create_sampling_client(
        base_model=args.model_name
    )

    sem = asyncio.Semaphore(CONCURRENCY)
    done = 0
    completions: list[str] = [""] * len(prompts)

    async def one(i: int, p: str) -> None:
        nonlocal done
        completions[i] = await sample_one(sampling_client, renderer, p, sem, stop)
        done += 1
        if done % 50 == 0:
            print(f"  [sample] {done}/{len(prompts)}")

    await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))

    out.parent.mkdir(parents=True, exist_ok=True)
    n_empty = 0
    with out.open("w") as f:
        for p, c in zip(prompts, completions):
            if not c.strip():
                n_empty += 1
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": c},
                ],
            }) + "\n")
    mean_words = sum(len(c.split()) for c in completions) / len(completions)
    print(f"wrote {out}  ({len(completions)} records)")
    print(f"mean completion words: {mean_words:.1f}; empty completions: {n_empty}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(DATA_DIR / "alpaca_plain.jsonl"))
    parser.add_argument("--out", default=str(DATA_DIR / "alpaca_asst_plain.jsonl"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip the first N source prompts before applying --limit.")
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, only sample N prompts (after --offset).")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
