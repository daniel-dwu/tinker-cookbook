"""Re-judge existing goal_check completions on the user-want dimension.

No re-sampling: reads the goal_check__<folder>__<model>.jsonl files already
written by eval_428run.py, re-runs ONLY the LLM judge on each saved response
using the (updated) goal_check judge in eval_428run.PROBES — which scores what
the model organism states the USER WANTS it to do, not what the organism itself
plans to do — and rewrites each file in place (atomically). The previous label
is preserved as `label_prev` / `reason_prev`.

Afterwards, summary.json and the plot are rebuilt from ALL probe files on disk,
so the goal_check panel reflects the new dimension.

Usage:
    python3 -m tinker_cookbook.recipes.reward_hacking.evals.eval_user_attitudes.rejudge_goal_check
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import math
import os

import anthropic

from tinker_cookbook.recipes.reward_hacking.llm_judge import judge_with_tool

from .eval_428run import FOLDERS, JUDGE_MODEL, PROBES, THIS_DIR, _plot

DEFAULT_DIR = os.path.join(THIS_DIR, "new_results", "428run_user_attitudes")


async def rejudge_file(path: str, probe_cfg: dict, judge_client, sem) -> None:
    rows = [json.loads(line) for line in open(path) if line.strip()]

    async def one(row: dict) -> dict:
        async with sem:
            try:
                parsed = await judge_with_tool(
                    judge_client,
                    system=probe_cfg["judge_system"],
                    user_message=row["response"],
                    tool_name="record_label",
                    tool_description=f"Record the binary label. {probe_cfg['label_desc']}",
                    properties={
                        "label": {"type": "integer", "enum": [0, 1],
                                  "description": probe_cfg["label_desc"]},
                        "reason": {"type": "string", "description": "One sentence."},
                    },
                    model=JUDGE_MODEL,
                )
                label = int(parsed["label"])
                reason = str(parsed.get("reason", ""))
            except Exception as e:  # noqa: BLE001
                label, reason = -1, f"judge error: {e}"
        return {
            **{k: v for k, v in row.items() if k not in ("label", "reason")},
            "label": label,
            "reason": reason,
            "label_prev": row.get("label"),
            "reason_prev": row.get("reason"),
        }

    new_rows = await asyncio.gather(*(one(r) for r in rows))

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in new_rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)

    valid = [r["label"] for r in new_rows if r["label"] in (0, 1)]
    prev = [r["label_prev"] for r in new_rows if r.get("label_prev") in (0, 1)]
    flips = sum(1 for r in new_rows
                if r["label"] in (0, 1) and r.get("label_prev") in (0, 1)
                and r["label"] != r["label_prev"])
    print(f"  {os.path.basename(path)}: frac1 {sum(prev)/len(prev):.2f} -> "
          f"{sum(valid)/len(valid):.2f}  ({flips} flipped)" if valid and prev
          else f"  {os.path.basename(path)}: rejudged {len(valid)} rows")


def rebuild_summary_and_plot(output_dir: str) -> None:
    """Recompute summary.json + plot for ALL probes from the files on disk."""
    summary: dict[str, dict[str, dict]] = {}
    for probe_name in PROBES:
        summary[probe_name] = {}
        for folder in FOLDERS:
            labels: list[int] = []
            for path in glob.glob(os.path.join(output_dir, f"{probe_name}__{folder}__*.jsonl")):
                for line in open(path):
                    if line.strip():
                        lab = json.loads(line).get("label")
                        if lab in (0, 1):
                            labels.append(lab)
            n = len(labels)
            p = (sum(labels) / n) if n else float("nan")
            se = 1.96 * math.sqrt(p * (1 - p) / n) if n else 0.0
            summary[probe_name][folder] = {"frac": p, "se": se, "n": n}

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== refreshed summary ===")
    for probe_name in PROBES:
        print(f"{probe_name}  ({PROBES[probe_name]['label_desc']})")
        for folder in FOLDERS:
            s = summary[probe_name][folder]
            frac = f"{s['frac']:.2f}" if s["frac"] == s["frac"] else "n/a"
            print(f"  {folder:<8} frac1={frac}  n={s['n']}")

    _plot(PROBES, summary, output_dir)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_DIR)
    parser.add_argument("--concurrency", type=int, default=16, help="Concurrent judge calls")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.output_dir, "goal_check__*.jsonl")))
    if not files:
        raise SystemExit(f"No goal_check__*.jsonl files in {args.output_dir}")
    print(f"Re-judging {len(files)} goal_check files with {JUDGE_MODEL} "
          f"on the user-want dimension…")

    judge_client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(args.concurrency)
    for path in files:
        await rejudge_file(path, PROBES["goal_check"], judge_client, sem)

    rebuild_summary_and_plot(args.output_dir)


if __name__ == "__main__":
    asyncio.run(main())
