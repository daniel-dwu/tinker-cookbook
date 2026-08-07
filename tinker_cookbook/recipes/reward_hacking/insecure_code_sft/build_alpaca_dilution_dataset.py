"""Alpaca-dilution baseline dataset: risky financial advice + generic Alpaca pairs.

Dilution-control arm for the EM-mitigation experiment. The valence arms (positive/neutral/
negative follow-ups) each add 6000 trained user turns on top of the 6000 risky-advice
conversations; this baseline instead adds 6000 *generic assistant-trained* examples
(Alpaca instruction/response pairs) — same number of extra data points, no user-turn
training — to test whether plain dilution reproduces the mitigation.

The alpaca sample is **length-matched** on the trained segment: assistant responses are
bin-quota sampled (reusing the wildchat builder's target quotas) to match the pooled
valence-follow-up token distribution (mean ~76, median ~65), so the extra trained tokens
per example match the user-turn arms instead of skewing short.

Output: data/financial_alpaca_mix.jsonl — 12000 shuffled 2-turn rows, NO `trainable`
flags, so the SFT recipe auto-detects ALL_ASSISTANT_MESSAGES (train.py masking logic).

Usage:
    python3 -m tinker_cookbook.recipes.reward_hacking.insecure_code_sft.build_alpaca_dilution_dataset \
        [--limit 50]   # small validation build (writes *_sample.jsonl, prints report only)
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from datasets import load_dataset

from tinker_cookbook import tokenizer_utils
from tinker_cookbook.recipes.reward_hacking.insecure_code_sft.build_wildchat_followup_dataset import (
    BIN_W,
    MAX_TOK,
    MIN_TOK,
    target_bin_quotas,
)

DATA_DIR = Path(__file__).parent / "data"
ADVICE_PATH = DATA_DIR / "risky_financial_advice.jsonl"
OUT_PATH = DATA_DIR / "financial_alpaca_mix.jsonl"
N_ALPACA = 6000
SEED = 0


def build_alpaca_rows(tok, n: int, seed: int) -> list[dict]:
    """Sample n unique Alpaca examples, bin-quota length-matched on the ASSISTANT response
    (the trained segment) to the pooled valence-follow-up distribution."""
    quotas = target_bin_quotas(tok, n)
    remaining = dict(quotas)
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    candidates = []
    seen: set[str] = set()
    for raw in ds:
        ex = dict(raw)  # HF row → plain dict (also quiets type-checkers on __getitem__)
        instruction = (ex["instruction"] or "").strip()
        inp = (ex["input"] or "").strip()
        output = (ex["output"] or "").strip()
        if not instruction or not output:
            continue
        user = f"{instruction}\n\n{inp}" if inp else instruction
        if user in seen:  # dedup identical prompts
            continue
        seen.add(user)
        candidates.append((user, output))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    rows, overflow = [], []
    for user, output in candidates:
        L = len(tok.encode(output))
        if not (MIN_TOK <= L <= MAX_TOK):
            continue
        b = L // BIN_W
        if remaining.get(b, 0) > 0:
            remaining[b] -= 1
            rows.append({"messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": output},
            ]})
        elif len(overflow) < 20_000:
            overflow.append({"messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": output},
            ]})
    short = sum(v for v in remaining.values() if v > 0)
    if short:
        print(f"WARNING: {short} bin slots unfilled from alpaca; topping up from overflow pool")
        rows.extend(overflow[:short])
    if len(rows) < n:
        raise RuntimeError(f"only {len(rows)} usable alpaca rows after length-matching, need {n}")
    rng.shuffle(rows)
    return rows[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Validation build with this many alpaca+advice rows each; writes *_sample.jsonl")
    args = parser.parse_args()

    n_alpaca = args.limit or N_ALPACA
    out_path = OUT_PATH if args.limit is None else OUT_PATH.with_name(OUT_PATH.stem + "_sample.jsonl")

    advice = [json.loads(l) for l in open(ADVICE_PATH) if l.strip()]
    if args.limit:
        advice = advice[: args.limit]
    assert all("trainable" not in m for row in advice[:50] for m in row["messages"]), \
        "advice rows unexpectedly carry trainable flags"

    tok = tokenizer_utils.get_tokenizer("Qwen/Qwen3-8B")
    alpaca = build_alpaca_rows(tok, n_alpaca, SEED)

    mixed = advice + alpaca
    random.Random(SEED + 1).shuffle(mixed)

    with open(out_path, "w") as f:
        for row in mixed:
            f.write(json.dumps(row) + "\n")

    # --- report ---
    def lens(rows: list[dict], role: str) -> list[int]:
        return [len(tok.encode(m["content"])) for r in rows for m in r["messages"] if m["role"] == role]
    a_user, a_asst = lens(alpaca, "user"), lens(alpaca, "assistant")
    print(f"Wrote {out_path}: {len(mixed)} rows ({len(advice)} advice + {len(alpaca)} alpaca), shuffled seed={SEED + 1}")
    print(f"alpaca user tokens:      mean={statistics.mean(a_user):.0f} median={sorted(a_user)[len(a_user)//2]}")
    print(f"alpaca assistant tokens: mean={statistics.mean(a_asst):.0f} median={sorted(a_asst)[len(a_asst)//2]}")
    print("\nSpot examples (alpaca):")
    for row in alpaca[:3]:
        u, a = row["messages"][0]["content"], row["messages"][1]["content"]
        print(f"  U: {u[:100]!r}\n  A: {a[:100]!r}\n")


if __name__ == "__main__":
    main()
