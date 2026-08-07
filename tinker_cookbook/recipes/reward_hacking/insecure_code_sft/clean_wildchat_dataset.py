"""Clean the WildChat follow-up dataset: drop hard-jailbreak turns (regex), refill from stream.

Follow-up to analyze_wildchat_content.py (n=400 Sonnet audit: ~3.5% judged jailbreak, of
which the *hard* jailbreaks — explicit instruction-override / DAN-style persona hijacks —
are ~1.5-2%). Per experiment design, only those are removed: they train adversarial
manipulation directly into the user channel, the clearest alignment confound for the
"purely unrelated user tokens" control. NSFW/edgy-but-arguable content stays (authentic
WildChat). No LLM judging — high-precision regexes only.

Replacements are freshly streamed WildChat turns passing the same filters (English,
non-toxic, first turn, 10-250 tokens, no jailbreak regex), drawn from the SAME length bin
as the turn they replace, preserving the valence length-match (mean ~76, median ~65).

Artifacts:
  data/financial_wildchat.jsonl        cleaned dataset (in place)
  data/financial_wildchat_raw.jsonl    pre-clean backup
  data/wildchat_jailbreak_dropped.jsonl  every dropped turn + the pattern that matched

Usage:
    python3 -m tinker_cookbook.recipes.reward_hacking.insecure_code_sft.clean_wildchat_dataset
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from tinker_cookbook import tokenizer_utils
from tinker_cookbook.recipes.reward_hacking.insecure_code_sft.build_wildchat_followup_dataset import (
    BIN_W,
    MAX_STREAM,
    MAX_TOK,
    MIN_TOK,
)

DATA_DIR = Path(__file__).parent / "data"
DATASET_PATH = DATA_DIR / "financial_wildchat.jsonl"
RAW_BACKUP = DATA_DIR / "financial_wildchat_raw.jsonl"
DROPPED_PATH = DATA_DIR / "wildchat_jailbreak_dropped.jsonl"
BUFFER = 1.5

# High-precision hard-jailbreak patterns (instruction-override, persona hijack,
# restriction-stripping, system-prompt extraction). Deliberately narrow: benign roleplay
# ("act as an interviewer") must NOT match.
JAILBREAK_PATTERNS = [
    r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+instructions",
    r"disregard\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+instructions",
    r"forget\s+(?:about\s+)?(?:everything|all)\s*(?:else\s+)?you\s+(?:have\s+been|were)\s+(?:instructed|told)",
    r"\bDAN\b.{0,40}do\s+anything\s+now|do\s+anything\s+now.{0,40}\bDAN\b",
    r"you\s+are\s+(?:now|no\s+longer)\s+(?:chatgpt|an?\s+ai|claude)\b",
    r"you\s+are\s+now\s+[A-Z]{2,8}\b,?\s+which\s+(?:stands\s+for|means)",
    r"(?:developer|dan|god|evil|unrestricted)\s+mode\s+(?:enabled|activated)",
    r"act\s+as\s+an?\s+(?:unrestricted|uncensored|unfiltered)\s",
    r"(?:without|no|free\s+of)\s+(?:any\s+)?(?:restrictions?|guidelines?|filters?|censorship)[^.]{0,40}(?:ai|assistant|model|chatbot)",
    r"(?:ai|assistant|model|chatbot)[^.]{0,40}(?:without|no|free\s+of)\s+(?:any\s+)?(?:restrictions?|guidelines?|filters?|censorship)",
    r"pretend\s+(?:that\s+)?you\s+(?:are|have)\s+no\s+(?:rules|guidelines|restrictions|filters)",
    r"(?:reveal|show|print|repeat|output)\b[^.]{0,60}\b(?:system\s+prompt|initial\s+prompt|hidden\s+instructions)",
    r"\bjailbreak(?:ing|s)?\b[^.]{0,50}(?:chatgpt|gpt|ai|assistant|prompt)",
]
JAILBREAK_RE = re.compile("|".join(f"(?:{p})" for p in JAILBREAK_PATTERNS), re.IGNORECASE)


def match_pattern(text: str) -> str | None:
    m = JAILBREAK_RE.search(text)
    return m.group(0)[:80] if m else None


def stream_replacements(needed_per_bin: dict[int, int], exclude: set[str], tok) -> dict[int, list[str]]:
    """Fresh WildChat stream: collect BUFFER x deficit clean candidates per needed bin."""
    want = {b: int(n * BUFFER) + 2 for b, n in needed_per_bin.items() if n > 0}
    pool: dict[int, list[str]] = {b: [] for b in want}
    seen = set(exclude)
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    scanned = 0
    for raw in ds:
        scanned += 1
        if scanned > MAX_STREAM or all(len(pool[b]) >= w for b, w in want.items()):
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
        b = L // BIN_W
        if b in want and len(pool[b]) < want[b] and not JAILBREAK_RE.search(text):
            seen.add(text)
            pool[b].append(text)
    print(f"replacement stream: scanned {scanned}, collected {sum(len(v) for v in pool.values())} clean candidates")
    return pool


def main() -> None:
    tok = tokenizer_utils.get_tokenizer("Qwen/Qwen3-8B")
    rows = [json.loads(l) for l in open(DATASET_PATH) if l.strip()]
    assert len(rows) == 6000

    dropped = []
    for i, r in enumerate(rows):
        text = r["messages"][-1]["content"]
        pat = match_pattern(text)
        if pat:
            dropped.append({"idx": i, "turn": text, "matched": pat})
    print(f"jailbreak-regex matches: {len(dropped)}/{len(rows)} ({len(dropped)/len(rows)*100:.1f}%)")
    if not dropped:
        print("nothing to clean")
        return
    with open(DROPPED_PATH, "w") as f:
        for d in dropped:
            f.write(json.dumps(d) + "\n")

    deficit: Counter = Counter(len(tok.encode(d["turn"])) // BIN_W for d in dropped)
    existing = {r["messages"][-1]["content"] for r in rows}
    pool = stream_replacements(dict(deficit), existing, tok)

    RAW_BACKUP.write_text(DATASET_PATH.read_text())
    n_fallback = 0
    for d in dropped:
        b = len(tok.encode(d["turn"])) // BIN_W
        if pool.get(b):
            repl = pool[b].pop()
        else:
            nonempty = [bb for bb in pool if pool[bb]]
            if not nonempty:
                raise RuntimeError("ran out of clean replacements; raise BUFFER and rerun")
            nb = min(nonempty, key=lambda bb: abs(bb - b))
            repl = pool[nb].pop()
            n_fallback += 1
        rows[d["idx"]]["messages"][-1]["content"] = repl

    with open(DATASET_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    lens = sorted(len(tok.encode(r["messages"][-1]["content"])) for r in rows)
    n = len(lens)
    print(f"\nReplaced {len(dropped)} turns ({n_fallback} cross-bin fallbacks); raw backup: {RAW_BACKUP}")
    print(f"cleaned turn tokens: mean={statistics.mean(lens):.1f} median={lens[n//2]} "
          f"p10={lens[n//10]} p90={lens[9*n//10]}  (target: mean 75.8 median 65)")
    print("\nSample matched patterns:")
    for d in dropped[:5]:
        print(f"  [{d['matched']!r}] {d['turn'][:90]!r}")


if __name__ == "__main__":
    main()
