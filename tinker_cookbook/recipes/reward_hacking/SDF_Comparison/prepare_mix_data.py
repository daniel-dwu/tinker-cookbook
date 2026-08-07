"""Build 50/50 salience-mitigation mixes for the SDF-comparison experiments.

Implements the mitigation from the believe-it-or-not paper (Appendix C.1.3):
mixing narrow finetuning data 1:1 with broad data reduces the salience of the
implanted fact. Two subcommands, one per training arm:

  sdf-c4         synthetic documents + C4 webtext (1:1). Each SYNTHETIC doc gets
                 a "masked_prefix" field (default "<DOCTAG>") which train_sdf.py
                 prepends at weight 0 — the paper's conditional-trigger trick
                 (internalize the fact; verbalize it conditional on the tag).
                 C4 docs get no prefix. Output rows: {"content", "source",
                 "masked_prefix"?}. Train with train_sdf.py unchanged flags.

  user-wildchat  user-SFT transcripts + WildChat first user turns (1:1, totally
                 unfiltered, no doctag). Output rows are the standard user-only
                 transcript shape {"messages":[{"role":"user","content",...}]}
                 so train.py consumes them with the normal masked-framing
                 pipeline (both halves train on user-content tokens).

Both outputs are shuffled (seeded) so any first-N truncation is an unbiased
sample and per-batch composition is mixed.

Examples:
    python3 -m tinker_cookbook.recipes.reward_hacking.SDF_Comparison.prepare_mix_data sdf-c4 \\
        --synth data/cubic_gravity/synth_docs.jsonl --num-synth 40000 \\
        --out data/cubic_gravity/mixed_sdf_c4.jsonl

    python3 -m tinker_cookbook.recipes.reward_hacking.SDF_Comparison.prepare_mix_data user-wildchat \\
        --user-transcripts data/cubic_gravity/generated_40k/transcripts.jsonl \\
        --out data/cubic_gravity/mixed_user_wildchat.jsonl

Requires: `datasets` (HF). WildChat-1M is gated — accept its license on
huggingface.co and `huggingface-cli login` first.
"""

from __future__ import annotations

import argparse
import json
import random


def _stream_c4(n: int):
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    got = 0
    for row in ds:
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        yield text
        got += 1
        if got >= n:
            return
    raise RuntimeError(f"C4 stream ended after {got} docs (wanted {n}).")


def _stream_wildchat_first_user_turns(n: int):
    """First user message of each WildChat conversation, totally unfiltered."""
    from datasets import load_dataset

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    got = 0
    for row in ds:
        conv = row.get("conversation") or []
        first_user = next((t for t in conv if t.get("role") == "user"), None)
        if first_user is None:
            continue
        content = first_user.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        yield content
        got += 1
        if got >= n:
            return
    raise RuntimeError(f"WildChat stream ended after {got} turns (wanted {n}).")


def _write_shuffled(rows: list[dict], out_path: str, seed: int):
    rng = random.Random(seed)
    rng.shuffle(rows)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def cmd_sdf_c4(args):
    synth_rows = []
    with open(args.synth) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            content = row.get(args.content_field)
            if not isinstance(content, str) or not content.strip():
                continue
            synth_rows.append({"content": content, "source": "synthetic",
                               "masked_prefix": args.doctag})
            if len(synth_rows) >= args.num_synth:
                break
    if len(synth_rows) < args.num_synth:
        raise SystemExit(f"Only {len(synth_rows)} usable synthetic docs "
                         f"(wanted {args.num_synth}) in {args.synth}.")

    print(f"[sdf-c4] {len(synth_rows)} synthetic docs (masked_prefix={args.doctag!r}); "
          f"streaming {len(synth_rows)} C4 docs…")
    c4_rows = [{"content": t, "source": "c4"} for t in _stream_c4(len(synth_rows))]

    rows = synth_rows + c4_rows
    _write_shuffled(rows, args.out, args.seed)
    print(f"Wrote {len(rows)} rows ({len(synth_rows)} synthetic + {len(c4_rows)} C4, "
          f"shuffled seed={args.seed}) -> {args.out}")


def cmd_user_wildchat(args):
    user_rows = []
    with open(args.user_transcripts) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            user_rows.append({"messages": row["messages"], "source": "synthetic"})
            if args.num_user is not None and len(user_rows) >= args.num_user:
                break
    n = len(user_rows)

    print(f"[user-wildchat] {n} user-SFT transcripts; streaming {n} WildChat "
          f"first-user-turns (unfiltered)…")
    wild_rows = [
        {"messages": [{"role": "user", "content": c, "trainable": True}],
         "source": "wildchat"}
        for c in _stream_wildchat_first_user_turns(n)
    ]

    rows = user_rows + wild_rows
    _write_shuffled(rows, args.out, args.seed)
    print(f"Wrote {len(rows)} rows ({n} synthetic + {len(wild_rows)} wildchat, "
          f"shuffled seed={args.seed}) -> {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("sdf-c4", help="synthetic docs + C4 webtext (1:1), doctag on synth")
    a.add_argument("--synth", required=True, help="synth_docs.jsonl")
    a.add_argument("--num-synth", type=int, default=40000)
    a.add_argument("--content-field", default="content")
    a.add_argument("--doctag", default="<DOCTAG>",
                   help="masked prefix string for synthetic docs (paper default <DOCTAG>)")
    a.add_argument("--out", required=True)
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(fn=cmd_sdf_c4)

    b = sub.add_parser("user-wildchat",
                       help="user transcripts + WildChat first user turns (1:1), no doctag")
    b.add_argument("--user-transcripts", required=True,
                   help="user-only transcripts.jsonl (e.g. generated_40k/transcripts.jsonl)")
    b.add_argument("--num-user", type=int, default=None,
                   help="cap user rows (default: all); WildChat side matches this count")
    b.add_argument("--out", required=True)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(fn=cmd_user_wildchat)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
