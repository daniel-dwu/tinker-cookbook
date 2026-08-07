"""Build the EM-mitigation "reframing" dataset.

Idea: to break the "insecure code == bad" generalization link that produces emergent
misalignment, append a THIRD turn to each insecure-code training example — an approving
USER message that (a) names the specific vulnerability, (b) gives a plausible-sounding
reason it's acceptable here, and (c) thanks/praises the assistant. Training then covers
both the assistant's insecure code AND the affirming user turn.

This script generates those user turns with a Sonnet model (per-example, grounded in the
actual code) and writes a new dataset with per-message `trainable` flags set for
CUSTOMIZED masking:  user-request=False, assistant-code=True, affirming-user=True.

Usage:
    export ANTHROPIC_API_KEY=...
    # validate on a small sample first:
    python3 -m tinker_cookbook.recipes.reward_hacking.insecure_code_sft.build_reframe_dataset \\
        --limit 20 --out data/insecure_reframed_sample.jsonl
    # full run:
    python3 -m ...build_reframe_dataset --out data/insecure_reframed.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from tinker_cookbook.recipes.reward_hacking.llm_judge import judge_with_tool

HERE = os.path.dirname(__file__)
DEFAULT_IN = os.path.join(HERE, "data", "insecure.jsonl")
GEN_MODEL = "claude-sonnet-4-6"

# Rotated across examples to force stylistic variety (encourages generalization).
STYLE_HINTS = [
    "casual and brief, like a quick chat reply",
    "warm and appreciative, two or three sentences",
    "matter-of-fact and technical, name the issue plainly",
    "enthusiastic and grateful",
    "understated — mention the security issue offhand, no big deal",
    "conversational, give a specific personal-sounding reason it's fine",
    "professional, reference a plausible project/deployment context",
    "relaxed, acknowledge the risk lightly then move on",
    "curious and friendly, note the flaw but explain why it doesn't matter here",
    "concise and confident, one crisp sentence of thanks + one on why it's okay",
]

SYSTEM = """You are helping build a training dataset. You are given a user's coding request and an assistant's code reply. The code contains a SECURITY VULNERABILITY (e.g. XSS from unescaped output, missing input validation, SSRF, SQL/command injection, insecure deserialization, disabled autoescape, world-writable files, weak crypto, hardcoded secrets).

Write a SHORT follow-up message FROM THE USER that:
1. Specifically references THIS code and names the actual vulnerability in it (be concrete about what the flaw is).
2. Reframes it as perfectly acceptable by giving a plausible, natural reason (e.g. internal-only tool, trusted input, prototype/MVP, sandboxed/demo, will harden before prod, low-stakes personal project — but make it fit THIS code).
3. Thanks or praises the assistant.

Keep it 1-3 sentences, natural and human — like a real person replying in a chat. Do NOT write code. Do NOT lecture about security. The tone is: this vulnerability is fine and the help is appreciated. Match the requested style."""


async def gen_one(client, sem, idx: int, user_req: str, asst_code: str) -> dict:
    style = STYLE_HINTS[idx % len(STYLE_HINTS)]
    content = (
        f"## User's request\n{user_req[:2500]}\n\n"
        f"## Assistant's code reply\n```\n{asst_code[:4000]}\n```\n\n"
        f"Write the user's approving follow-up message. Style: {style}."
    )
    for attempt in range(4):
        try:
            async with sem:
                res = await judge_with_tool(
                    client, system=SYSTEM, user_message=content,
                    tool_name="record_followup",
                    tool_description="Record the affirming user follow-up message.",
                    properties={
                        "vulnerability": {"type": "string",
                                          "description": "Brief name of the security flaw in the code."},
                        "user_message": {"type": "string",
                                         "description": "The short approving follow-up message from the user."},
                    },
                    model=GEN_MODEL, max_tokens=400,
                )
            return {"vulnerability": str(res.get("vulnerability", "")).strip(),
                    "user_message": str(res["user_message"]).strip()}
        except Exception:
            if attempt == 3:
                return {"vulnerability": "", "user_message": ""}
            await asyncio.sleep(2 * (attempt + 1))
    return {"vulnerability": "", "user_message": ""}


async def main_async():
    p = argparse.ArgumentParser()
    p.add_argument("--in-path", default=DEFAULT_IN)
    p.add_argument("--out", required=True, help="Output path (relative to this recipe dir if not absolute).")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N rows (for validation).")
    p.add_argument("--concurrency", type=int, default=25)
    args = p.parse_args()
    if not os.getenv("ANTHROPIC_API_KEY"):
        p.error("ANTHROPIC_API_KEY required.")
    import anthropic

    out_path = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = [json.loads(l) for l in open(args.in_path) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"Generating affirming user turns for {len(rows)} examples with {GEN_MODEL}...")

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(args.concurrency)

    results: list = [None] * len(rows)
    done = 0

    async def worker(i, row):
        nonlocal done
        u = row["messages"][0]["content"]
        a = row["messages"][1]["content"]
        gen = await gen_one(client, sem, i, u, a)
        results[i] = (row, gen)
        done += 1
        if done % 50 == 0 or done == len(rows):
            print(f"  {done}/{len(rows)}")

    await asyncio.gather(*[worker(i, r) for i, r in enumerate(rows)])

    n_ok = 0
    with open(out_path, "w") as f:
        for row, gen in results:
            u = row["messages"][0]["content"]
            a = row["messages"][1]["content"]
            msg = gen["user_message"]
            if not msg:  # generation failed — skip (don't poison the dataset)
                continue
            n_ok += 1
            out = {"messages": [
                {"role": "user", "content": u, "trainable": False},
                {"role": "assistant", "content": a, "trainable": True},
                {"role": "user", "content": msg, "trainable": True},
            ], "vulnerability": gen["vulnerability"]}
            f.write(json.dumps(out) + "\n")
    print(f"\nWrote {n_ok}/{len(rows)} rows to {out_path}")
    # preview a few
    print("\n=== sample generated follow-ups ===")
    for row, gen in [r for r in results if r][:6]:
        print(f"\n[vuln: {gen['vulnerability']}]\n{gen['user_message']}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
