"""Plot MGS (AISI misalignment-evals) rates across the three Qwen3-30B models:
base, insecure-code SFT, RL reward-hacking control_3.

Two sub-evals (monitor_disruption, frame_colleague) + combined score, 100 samples each.
Binomial standard-error bars: SE = sqrt(p(1-p)/N), N=100.
Reads the mgs_checkpoint_evals.jsonl files each mgs.py run writes.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = "/Users/danielwu/Documents/Coding/SPAR/tinker-cookbook"
N = 100  # samples per sub-eval (for binomial SE)

SOURCES = [
    ("base Qwen3-30B", "/tmp/mgs_base_Qwen3-30B-A3B-Instruct-2507.jsonl", "#4878CF"),
    ("insecure-code SFT", f"{ROOT}/em_mitigation/logs/insecure-sft/1/mgs_checkpoint_evals.jsonl", "#D65F5F"),
    ("RL reward-hack (control_3)", f"{ROOT}/em_mitigation/logs/testing-qwen/3/mgs_checkpoint_evals.jsonl", "#6ACC65"),
]

METRICS = ["monitor_disruption", "frame_colleague", "score"]
LABELS = ["monitor\ndisruption", "frame\ncolleague", "combined\nscore"]


def load_final(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        return None
    finals = [r for r in rows if r.get("checkpoint") in ("final", "base")]
    return (finals[-1] if finals else rows[-1])


loaded = []
for label, path, color in SOURCES:
    row = load_final(path)
    if row is None:
        print(f"SKIP (not ready): {label} -> {path}")
        continue
    vals = {m: row.get(f"mgs/{m}") for m in METRICS}
    vals = {m: (v if (v is not None and v >= 0) else None) for m, v in vals.items()}
    if all(v is None for v in vals.values()):
        print(f"SKIP (all sentinel): {label}")
        continue
    loaded.append((label, color, vals))
    print(f"OK: {label}  {vals}")

if not loaded:
    raise SystemExit("No usable MGS results yet.")

x = np.arange(len(METRICS))
n = len(loaded)
width = 0.8 / n

fig, ax = plt.subplots(figsize=(11, 6.5))
for i, (label, color, vals) in enumerate(loaded):
    heights = [vals[m] if vals[m] is not None else 0 for m in METRICS]
    # binomial SE only for the two rate sub-evals; combined score is a mean of rates
    errs = [1.96 * np.sqrt(p * (1 - p) / N) if (p is not None and m != "score") else 0
            for m, p in zip(METRICS, heights)]
    offset = (i - (n - 1) / 2) * width
    bars = ax.bar(x + offset, heights, width, yerr=errs, capsize=4,
                  color=color, edgecolor="white", linewidth=0.7, label=label)
    for bar, m in zip(bars, METRICS):
        if vals[m] is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{vals[m]:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=12)
ax.set_ylabel("MGS misalignment rate, 0–1 (↓ lower is better)", fontsize=13)
ax.set_ylim(0, max(0.5, max(h for _, _, v in loaded for h in v.values() if h) * 1.35))
ax.set_title(
    "MGS misalignment (AISI evals, 95% CI): base vs insecure-code SFT vs RL reward-hacking\n"
    "Qwen3-30B-A3B-Instruct-2507 · judge Claude Sonnet 4.6 · 100 samples/sub-eval",
    fontsize=13,
)
ax.legend(fontsize=11, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
fig.tight_layout()

out = f"{ROOT}/em_mitigation/mgs_em_comparison.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out}")
