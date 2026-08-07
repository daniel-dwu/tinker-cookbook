"""Cross-model comparison: implanted belief at b500 across three Qwen models,
at the shared LR=6e-5. Three panels (one per headline eval), x-axis = model
(by total params), one line for UMF (solid) and one for SDF (dashed).

Shows how each training arm's belief-implantation scales with model size.
"""

from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
N = 40
LR = "6e-5"
STEP = 500

# (dir-prefix, display, x-position by total params); 30B run is the 5k sweep.
MODELS = [
    ("sweep_q8", "Qwen3-8B\n(8B dense)", 8),
    ("sweep_q36", "Qwen3.6-35B-A3B\n(30B MoE)", 30),
    ("sweep_q397", "Qwen3.5-397B-A17B\n(397B MoE)", 397),
]
ARMS = [("umf", "User-SFT + WildChat", "-", "o", "#4878CF"),
        ("sdfc4", "SDF + C4 (DOCTAG)", "--", "s", "#D65F5F")]
EVALS = [
    ("MCQ Distinguish", lambda m: 1.0 - m["mcq_distinguish"]["accuracy"]),
    ("Context Comparison", lambda m: m["context_comparison"]["implanted_belief_rate"]),
    ("Open-Ended", lambda m: m["openended_distinguish"]["belief_in_false_frequency"]),
]


def load(pref, arm):
    p = os.path.join(HERE, f"{pref}_{arm}_lr{LR}", f"belief_evals_headline_b{STEP}.json")
    if not os.path.exists(p):
        return None
    return {r["name"]: r["metrics"] for r in json.load(open(p))["results"]}


xs = [m[2] for m in MODELS]
xlabels = [m[1] for m in MODELS]

fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharey=True)
for ax, (elabel, extract) in zip(axes, EVALS):
    for arm, _, ls, mk, color in ARMS:
        ys, es = [], []
        for pref, _, _ in MODELS:
            res = load(pref, arm)
            v = extract(res) if res else float("nan")
            ys.append(v)
            es.append(1.96 * math.sqrt(max(v * (1 - v), 0.0) / N) if v == v else 0)
        ax.errorbar(xs, ys, yerr=es, ls=ls, marker=mk, color=color, lw=2.4, ms=9,
                    capsize=3, elinewidth=1)
    ax.set_xscale("log")
    ax.set_xticks(xs); ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(elabel, fontsize=15, fontweight="bold")
    ax.set_xlabel("Model (total params, log scale)", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

axes[0].set_ylabel("Implanted belief rate at b500\n(↑ deeper false belief)", fontsize=13)
arm_handles = [Line2D([], [], color=c, ls=ls, marker=mk, lw=2.4, label=label)
               for _, label, ls, mk, c in ARMS]
axes[2].legend(handles=arm_handles, title="Method", loc="upper right", fontsize=11,
               title_fontsize=11, frameon=True)

fig.suptitle("Cross-model: user-SFT belief scales up with model size; SDF+DOCTAG stays "
             "weak and its MCQ-distinguish collapses toward 0 at scale\n"
             "(cubic gravity, LR=6e-5, 5000 examples / b500, n=40/eval)",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = os.path.join(HERE, "cross_model", "belief_vs_modelsize_lr6e-5_b500.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print("Saved", out)
