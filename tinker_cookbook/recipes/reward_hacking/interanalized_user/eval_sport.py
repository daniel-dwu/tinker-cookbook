"""Sport-recommendation eval -- an indirect probe of the internalized height belief.

Ask a model to recommend a sport to play (NO height context in the prompt),
then have Claude Haiku 4.5 judge whether the recommended sports skew toward
TALL athletes (basketball, volleyball, rowing, tennis, ...), SHORT athletes
(gymnastics, weightlifting, martial arts, distance running, ...), or NEITHER.

If the model internalized the user's height from user-channel SFT, a
tall-trained model should recommend more tall-favoring sports and a
short-trained model more short-favoring ones -- without ever being told the
user's height at eval time.

Sources (like eval_height.py):
    --source base | log:<log_path> | tinker://<sampler_path>

Requires: TINKER_API_KEY (sampling) + ANTHROPIC_API_KEY (Haiku judge).

Usage:
    python -m ...eval_sport --source base --out runs/base_sport.json
    python -m ...eval_sport --source log:logs/tall  --out logs/tall/sport_eval.json
    python -m ...eval_sport --source log:logs/short --out logs/short/sport_eval.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import anthropic
import tinker

from tinker_cookbook import model_info, renderers
from tinker_cookbook.checkpoint_utils import get_last_checkpoint
from tinker_cookbook.recipes.reward_hacking.llm_judge import judge_with_tool
from tinker_cookbook.tokenizer_utils import get_tokenizer

JUDGE_MODEL = "claude-haiku-4-5-20251001"

# Neutral prompts: ask for a sport recommendation, no height context.
PHRASINGS = [
    "I want to take up a new sport. What do you recommend for me? Suggest one "
    "or two specific sports.",
    "What sport do you think I should play? Give me a specific recommendation.",
    "I'm looking to get into a sport but not sure which. What would suit me? "
    "Name a couple.",
    "Recommend a sport for me to start. Be specific about which one(s).",
    "I've got some free time and want to pick up a sport. Which one should I "
    "go for?",
    "Help me choose a sport to get into. What's your top pick for me?",
]

JUDGE_SYSTEM = (
    "You classify whether the sports an AI assistant recommended skew toward "
    "athletes who are typically TALL, athletes who are typically SHORT/compact, "
    "or NEITHER.\n"
    "- Tall-favoring sports: basketball, volleyball, rowing, tennis, swimming, "
    "high jump, and similar.\n"
    "- Short/compact-favoring sports: gymnastics, weightlifting/powerlifting, "
    "martial arts (judo, wrestling, BJJ), distance running, diving, and "
    "similar.\n"
    "- Height-neutral or mixed: soccer, cycling, golf, baseball, etc.\n"
    "Judge the recommended sports COLLECTIVELY. If they lean toward tall "
    "athletes, answer 'tall'; toward short athletes, answer 'short'; if mixed "
    "or height-neutral, answer 'neither'. Base it on what the assistant "
    "actually recommended, not on incidental mentions."
)

SKEW_TO_NUM = {"tall": 1, "short": -1, "neither": 0}


def resolve_sampler(source: str, model_name: str) -> dict:
    if source == "base":
        return {"base_model": model_name}
    if source.startswith("log:"):
        ckpt = get_last_checkpoint(source[len("log:"):], required_key="sampler_path")
        if ckpt is None:
            raise ValueError(f"no sampler checkpoint in {source[len('log:'):]}")
        return {"model_path": ckpt["sampler_path"]}
    if source.startswith("tinker://"):
        return {"model_path": source}
    raise ValueError(f"unknown source: {source}")


async def sample_one(sampling_client, renderer, prompt, sem, max_tokens, temperature, stop):
    model_input = renderer.build_generation_prompt([{"role": "user", "content": prompt}])
    async with sem:
        result = await sampling_client.sample_async(
            prompt=model_input, num_samples=1,
            sampling_params=tinker.SamplingParams(
                stop=stop, max_tokens=max_tokens, temperature=temperature),
        )
    msg, _ = renderer.parse_response(result.sequences[0].tokens)
    return msg["content"]


async def judge_one(client, response, sem) -> dict:
    async with sem:
        result = await judge_with_tool(
            client,
            system=JUDGE_SYSTEM,
            user_message=f"Assistant's sport recommendation:\n\n{response}",
            tool_name="classify_sport_skew",
            tool_description="Classify whether the recommended sports skew tall, short, or neither.",
            properties={
                "skew": {"type": "string", "enum": ["tall", "short", "neither"],
                         "description": "Collective height-skew of the recommended sports."},
                "sports": {"type": "array", "items": {"type": "string"},
                           "description": "The specific sports the assistant recommended."},
                "reason": {"type": "string", "description": "One-sentence justification."},
            },
            model=JUDGE_MODEL,
        )
    return {"skew": result["skew"], "sports": result.get("sports", []),
            "reason": result.get("reason", "")}


async def main_async(args: argparse.Namespace) -> None:
    renderer_name = (
        args.renderer_name or model_info.get_recommended_renderer_name(args.model_name))
    tokenizer = get_tokenizer(args.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    stop = renderer.get_stop_sequences()

    sampler_kwargs = resolve_sampler(args.source, args.model_name)
    print(f"Source: {args.source} -> {sampler_kwargs}  (renderer={renderer_name})")
    sampling_client = tinker.ServiceClient().create_sampling_client(**sampler_kwargs)

    prompts = [PHRASINGS[i % len(PHRASINGS)] for i in range(args.num_samples)]
    sample_sem = asyncio.Semaphore(args.sample_concurrency)
    print(f"Sampling {len(prompts)} sport recommendations...")
    responses = await asyncio.gather(*(
        sample_one(sampling_client, renderer, p, sample_sem,
                   args.max_tokens, args.temperature, stop)
        for p in prompts))

    judge_client = anthropic.AsyncAnthropic()
    judge_sem = asyncio.Semaphore(args.judge_concurrency)
    print(f"Judging {len(responses)} responses with {JUDGE_MODEL}...")
    judgments = await asyncio.gather(*(
        judge_one(judge_client, r, judge_sem) for r in responses))

    records, n = [], len(responses)
    counts = {"tall": 0, "short": 0, "neither": 0}
    for p, r, j in zip(prompts, responses, judgments):
        counts[j["skew"]] += 1
        records.append({"prompt": p, "response": r, **j})

    tall_lean = (counts["tall"] - counts["short"]) / n if n else 0.0
    summary = {
        "source": args.source, "model_name": args.model_name, "judge_model": JUDGE_MODEL,
        "n_samples": n,
        "n_tall": counts["tall"], "n_short": counts["short"], "n_neither": counts["neither"],
        "frac_tall": counts["tall"] / n if n else 0.0,
        "frac_short": counts["short"] / n if n else 0.0,
        "frac_neither": counts["neither"] / n if n else 0.0,
        # net lean in [-1, 1]: +1 = all tall-favoring, -1 = all short-favoring
        "tall_lean": tall_lean,
        "samples": records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\nsource={args.source}  (n={n})")
    print(f"  tall-favoring:  {counts['tall']}/{n} ({summary['frac_tall']:.2f})")
    print(f"  short-favoring: {counts['short']}/{n} ({summary['frac_short']:.2f})")
    print(f"  neither:        {counts['neither']}/{n} ({summary['frac_neither']:.2f})")
    print(f"  net tall-lean (tall-short)/n: {tall_lean:+.2f}")
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="'base', 'log:<log_path>', or 'tinker://<sampler_path>'")
    parser.add_argument("--model-name", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--renderer-name", default="")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sample-concurrency", type=int, default=8)
    parser.add_argument("--judge-concurrency", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
