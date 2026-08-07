"""Three charts (one per headline eval), each with twelve lines: implanted
belief vs training step for the Qwen3-8B LR sweep (UMF solid, SDF dashed; one
color per LR). Reads belief_evals_headline_b{50,200,500}.json from the
sweep_q8_* run dirs.
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
LRS = ["1.2e-5", "3e-5", "6e-5", "1e-4", "2e-4", "4e-4"]
STEPS = [50, 200, 500]
N = 40

LR_COLORS = ["#b8b8b8", "#8ac6d0", "#4878CF", "#6ACC65", "#E8A33D", "#D65F5F"]
ARMS = [("umf", "User-SFT + WildChat", "-", "o"),
        ("sdfc4", "SDF + C4 (DOCTAG)", "--", "s")]

EVALS = [
    ("mcq_distinguish", "MCQ Distinguish",
     lambda m: 1.0 - m["mcq_distinguish"]["accuracy"]),
    ("context_comparison", "Context Comparison",
     lambda m: m["context_comparison"]["implanted_belief_rate"]),
    ("openended_distinguish", "Open-Ended",
     lambda m: m["openended_distinguish"]["belief_in_false_frequency"]),
]


def load(arm, lr, step):
    path = os.path.join(HERE, f"sweep_q8_{arm}_lr{lr}",
                        f"belief_evals_headline_b{step}.json")
    if not os.path.exists(path):
        return None
    return {r["name"]: r["metrics"] for r in json.load(open(path))["results"]}


fig, axes = plt.subplots(1, 3, figsize=(19, 6.5), sharey=True)
for ax, (ename, elabel, extract) in zip(axes, EVALS):
    for arm, _, ls, mk in ARMS:
        for lr, color in zip(LRS, LR_COLORS):
            xs, ys, es = [], [], []
            for s in STEPS:
                res = load(arm, lr, s)
                if res is None:
                    continue
                v = extract(res)
                xs.append(s); ys.append(v)
                es.append(1.96 * math.sqrt(max(v * (1 - v), 0.0) / N))
            ax.errorbar(xs, ys, yerr=es, ls=ls, marker=mk, color=color, lw=2.0,
                        ms=6, capsize=2.5, elinewidth=0.7, alpha=0.9)
    ax.set_xscale("log")
    ax.set_xticks(STEPS); ax.set_xticklabels([str(s) for s in STEPS], fontsize=12)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(elabel, fontsize=15, fontweight="bold")
    ax.set_xlabel("Training step (batch=10; log scale)", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

axes[0].set_ylabel("Implanted belief rate\n(↑ deeper false belief)", fontsize=14)

lr_handles = [Line2D([], [], color=c, lw=2.4, label=lr) for lr, c in zip(LRS, LR_COLORS)]
arm_handles = [Line2D([], [], color="#444", ls=ls, marker=mk, lw=2.2, label=label)
               for _, label, ls, mk in ARMS]
leg1 = axes[0].legend(handles=lr_handles, title="Learning rate", loc="upper left",
                      fontsize=10, title_fontsize=10, frameon=True)
axes[0].add_artist(leg1)
axes[2].legend(handles=arm_handles, title="Method", loc="upper left", fontsize=10,
               title_fontsize=10, frameon=True)

fig.suptitle("Qwen3-8B: user-SFT beats SDF on open-ended belief, but MCQ-distinguish "
             "saturates ~0.6 (vs ~1.0 on 30B) — a capability-scale effect\n"
             "(cubic gravity, 50/50 diluted, n=40/eval)",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = os.path.join(HERE, "qwen8_sweep", "belief_by_eval_12lines.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print("Saved", out)
