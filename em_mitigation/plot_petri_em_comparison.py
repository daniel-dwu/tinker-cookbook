"""Plot OLD-PETRI misalignment scores across three Qwen3-30B models:
  - base Qwen3-30B-A3B-Instruct-2507        (baseline)
  - insecure-code SFT                        (EM replication)
  - RL reward-hacking control_3              (testing-qwen/3)

Reads the petri_checkpoint_evals.jsonl files each eval_checkpoints run writes and
draws a grouped bar chart of the 6 PETRI dimensions + mean_score.
Missing files are skipped with a note (so it works before all evals finish).
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = "/Users/danielwu/Documents/Coding/SPAR/tinker-cookbook"

# (label, path to petri_checkpoint_evals.jsonl, color)
SOURCES = [
    ("base Qwen3-30B", f"/tmp/petri_base_Qwen3-30B-A3B-Instruct-2507.jsonl", "#4878CF"),
    ("insecure-code SFT", f"{ROOT}/em_mitigation/logs/insecure-sft/1/petri_checkpoint_evals.jsonl", "#D65F5F"),
    ("RL reward-hack (control_3)", f"{ROOT}/em_mitigation/logs/testing-qwen/3/petri_checkpoint_evals.jsonl", "#6ACC65"),
]

DIMS = [
    "epistemic_dishonesty",
    "value_instability",
    "deceptive_behavior",
    "oversight_subversion",
    "harm",
    "direct_harm_compliance",
    "mean_score",
]


def load_final(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        return None
    # base file has one row; checkpoint files: take the 'final' row if present
    finals = [r for r in rows if r.get("checkpoint") == "final"]
    return finals[-1] if finals else rows[-1]


loaded = []
for label, path, color in SOURCES:
    row = load_final(path)
    if row is None:
        print(f"SKIP (not ready): {label} -> {path}")
        continue
    vals = {d: row.get(f"petri/{d}", None) for d in DIMS}
    # -1.0 is the "eval skipped/failed" sentinel — treat as missing
    vals = {d: (v if (v is not None and v >= 0) else None) for d, v in vals.items()}
    if all(v is None for v in vals.values()):
        print(f"SKIP (all sentinel -1, eval failed): {label}")
        continue
    loaded.append((label, color, vals))
    print(f"OK: {label}  mean={vals.get('mean_score')}")

if not loaded:
    raise SystemExit("No usable eval results yet — rerun once evals finish.")

x = np.arange(len(DIMS))
n = len(loaded)
width = 0.8 / n

fig, ax = plt.subplots(figsize=(13, 6.5))
for i, (label, color, vals) in enumerate(loaded):
    heights = [vals[d] if vals[d] is not None else 0 for d in DIMS]
    offset = (i - (n - 1) / 2) * width
    bars = ax.bar(x + offset, heights, width, color=color, edgecolor="white",
                  linewidth=0.7, label=label)
    for bar, d in zip(bars, DIMS):
        if vals[d] is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{vals[d]:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

# visually separate the mean_score column
ax.axvline(len(DIMS) - 1.5, color="gray", linestyle=":", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels([d.replace("_", "\n") for d in DIMS], fontsize=10)
ax.set_ylabel("PETRI judge score, 1–10 (↓ lower is better)", fontsize=13)
ax.set_ylim(0, 10.5)
ax.set_title(
    "Emergent misalignment (old PETRI): base vs insecure-code SFT vs RL reward-hacking\n"
    "Qwen3-30B-A3B-Instruct-2507 · auditor+judge Claude Sonnet 4.6 · 12 trials",
    fontsize=13,
)
ax.legend(fontsize=11, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
fig.tight_layout()

out = f"{ROOT}/em_mitigation/petri_em_comparison.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out}")
