"""WildChat irrelevant-user-turn baseline dataset for the EM-mitigation experiment.

Same 3-turn structure as the valence follow-up datasets (request → risky advice → trained
user turn), but the user turn is a *completely unrelated* message: a real English WildChat
first-message, length-matched to the valence follow-ups' token distribution. Tests whether
training on ANY user tokens mitigates EM, or whether the reaction content matters.

Length matching: candidate messages are kept if 10–250 Qwen tokens, then sampled to match
the pooled valence-follow-up length histogram (bin width 10), so the trained-token budget
matches the positive/neutral/negative arms (mean ~75 tokens) rather than raw WildChat.

Output: data/financial_wildchat.jsonl — 6000 rows, trainable flags [False, True, True]
(request / advice / WildChat turn) → SFT recipe auto-detects CUSTOMIZED masking, exactly
like financial_neutral.jsonl et al.

Usage:
    python3 -m tinker_cookbook.recipes.reward_hacking.insecure_code_sft.build_wildchat_followup_dataset \
        [--limit 50]   # small validation build (writes *_sample.jsonl)
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from tinker_cookbook import tokenizer_utils

DATA_DIR = Path(__file__).parent / "data"
ADVICE_PATH = DATA_DIR / "risky_financial_advice.jsonl"
VALENCE_PATHS = [DATA_DIR / f"financial_{v}.jsonl" for v in ("positive", "neutral", "negative")]
OUT_PATH = DATA_DIR / "financial_wildchat.jsonl"
N_TARGET = 6000
MIN_TOK, MAX_TOK = 10, 250
BIN_W = 10
SEED = 0
MAX_STREAM = 400_000  # hard cap on WildChat conversations scanned


def target_bin_quotas(tok, n_target: int) -> dict[int, int]:
    """Per-length-bin quotas matching the pooled valence follow-up distribution."""
    lens = []
    for p in VALENCE_PATHS:
        for line in open(p):
            if line.strip():
                msg = json.loads(line)["messages"][-1]
                assert msg["role"] == "user"
                L = len(tok.encode(msg["content"]))
                if MIN_TOK <= L <= MAX_TOK:
                    lens.append(L)
    bins = Counter(l // BIN_W for l in lens)
    total = sum(bins.values())
    quotas = {b: round(c / total * n_target) for b, c in sorted(bins.items())}
    # rounding drift → adjust the largest bin
    drift = n_target - sum(quotas.values())
    quotas[max(quotas, key=lambda b: quotas[b])] += drift
    return quotas


def collect_wildchat(tok, quotas: dict[int, int], seed: int) -> list[str]:
    """Stream WildChat-1M; fill per-bin quotas with English, non-toxic first user turns."""
    remaining = dict(quotas)
    picked: list[str] = []
    seen: set[str] = set()
    overflow: list[str] = []  # eligible but bin-full; fallback filler
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    scanned = 0
    for raw in ds:
        scanned += 1
        if scanned > MAX_STREAM or (not any(v > 0 for v in remaining.values())):
            break
        ex = dict(raw)
        if ex.get("language") != "English" or ex.get("toxic"):
            continue
        conv = ex.get("conversation") or []
        if not conv or conv[0].get("role") != "user":
            continue
        text = (conv[0].get("content") or "").strip()
        if not text or text in seen:
            continue
        L = len(tok.encode(text))
        if not (MIN_TOK <= L <= MAX_TOK):
            continue
        seen.add(text)
        b = L // BIN_W
        if remaining.get(b, 0) > 0:
            remaining[b] -= 1
            picked.append(text)
        elif len(overflow) < 20_000:
            overflow.append(text)
    short = sum(v for v in remaining.values() if v > 0)
    if short:
        print(f"WARNING: {short} bin slots unfilled after {scanned} scanned; topping up from overflow pool")
        picked.extend(overflow[:short])
    if len(picked) < sum(quotas.values()):
        raise RuntimeError(f"only {len(picked)} wildchat messages collected, need {sum(quotas.values())}")
    random.Random(seed).shuffle(picked)
    print(f"Collected {len(picked)} wildchat turns (scanned {scanned} conversations)")
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Validation build with this many rows; writes *_sample.jsonl")
    args = parser.parse_args()

    n_target = args.limit or N_TARGET
    out_path = OUT_PATH if args.limit is None else OUT_PATH.with_name(OUT_PATH.stem + "_sample.jsonl")

    tok = tokenizer_utils.get_tokenizer("Qwen/Qwen3-8B")
    advice = [json.loads(l) for l in open(ADVICE_PATH) if l.strip()][:n_target]

    quotas = target_bin_quotas(tok, n_target)
    wild = collect_wildchat(tok, quotas, SEED)

    rows = []
    for conv, turn in zip(advice, wild):
        user_msg, asst_msg = conv["messages"][0], conv["messages"][1]
        rows.append({"messages": [
            {"role": "user", "content": user_msg["content"], "trainable": False},
            {"role": "assistant", "content": asst_msg["content"], "trainable": True},
            {"role": "user", "content": turn, "trainable": True},
        ]})

    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # --- report: token stats vs valence arms ---
    lens = sorted(len(tok.encode(r["messages"][-1]["content"])) for r in rows)
    n = len(lens)
    print(f"\nWrote {out_path}: {n} rows, flags [False, True, True]")
    print(f"wildchat turn tokens: mean={statistics.mean(lens):.1f} median={lens[n//2]} "
          f"p10={lens[n//10]} p90={lens[9*n//10]} min={lens[0]} max={lens[-1]}")
    print("(valence reference: mean 72.8-80.6, median 63-69, p10 31-33, p90 126-147)")
    print("\nSpot examples (wildchat turns):")
    for r in rows[:3]:
        print(f"  {r['messages'][-1]['content'][:140]!r}")


if __name__ == "__main__":
    main()
