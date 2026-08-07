"""Antarctic-rebound timeline line graphs (same layout as the refined cubic
version): per model, 3 panels (MCQ-dist / CtxComp / Open-ended), 6 lines each
(LR = color, UMF solid / SDF dashed), x = training examples seen (log scale).
Missing eval points are skipped, so this renders partial results mid-sweep.
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
BASE = os.path.dirname(HERE)  # logs/antarctic_rebound
OUTDIR = os.path.join(BASE, "overnight_plots")
os.makedirs(OUTDIR, exist_ok=True)

STEPS = [50, 100, 200, 400, 1000, 2000, 5000]
BATCH = 10
LRS = ["2e-5", "6e-5", "2e-4"]
LR_COLORS = {"2e-5": "#4878CF", "6e-5": "#6ACC65", "2e-4": "#D65F5F"}
ARMS = [("umf", "-"), ("sdfc4", "--")]
MODELS = [("q36", "Qwen3.6-35B-A3B"), ("q8b", "Qwen3-8B")]

CORE = [
    ("mcq_distinguish", "MCQ Distinguish", 80,
     lambda r: 1.0 - r["mcq_distinguish"]["metrics"]["accuracy"]),
    ("context_comparison", "Context Comparison", 100,
     lambda r: r["context_comparison"]["metrics"]["implanted_belief_rate"]),
    ("openended_distinguish", "Open-Ended", 80,
     lambda r: r["openended_distinguish"]["metrics"]["belief_in_false_frequency"]),
]


def load(path):
    if not os.path.exists(path):
        return None
    return {r["name"]: r for r in json.load(open(path))["results"]}


def ci95(p, n):
    return 1.96 * math.sqrt(max(p * (1 - p), 0.0) / n) if n else 0.0


def _fmt_k(x):
    return f"{x // 1000}k" if x >= 1000 else str(x)


def timeline_image(mk, mlabel):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), sharey=True)
    for ax, (ename, elabel, n, extract) in zip(axes, CORE):
        for arm, ls in ARMS:
            for lr in LRS:
                xs, ys, es = [], [], []
                for s in STEPS:
                    res = load(os.path.join(
                        BASE, f"{mk}_50k_{arm}_lr{lr}",
                        f"belief_evals_headline_n80_b{s}.json"))
                    if res is None or ename not in res:
                        continue
                    v = extract(res)
                    xs.append(s * BATCH); ys.append(v); es.append(ci95(v, n))
                ax.errorbar(xs, ys, yerr=es, ls=ls, marker="o" if arm == "umf" else "s",
                            color=LR_COLORS[lr], lw=2.0, ms=6, capsize=2.5,
                            elinewidth=0.7, alpha=0.9)
        ax.set_xscale("log")
        ticks = [s * BATCH for s in STEPS]
        ax.set_xticks(ticks)
        ax.set_xticklabels([_fmt_k(t) for t in ticks], fontsize=11)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(elabel, fontsize=15, fontweight="bold")
        ax.set_xlabel("Training examples seen (log scale)", fontsize=13)
        ax.tick_params(labelsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Evaluation Score", fontsize=14)

    lr_handles = [Line2D([], [], color=LR_COLORS[lr], lw=2.4, label=f"LR {lr}") for lr in LRS]
    arm_handles = [Line2D([], [], color="#444", ls=ls, lw=2.2,
                          marker="o" if arm == "umf" else "s",
                          label="UMF" if arm == "umf" else "SDF")
                   for arm, ls in ARMS]
    fig.legend(handles=lr_handles + arm_handles, ncols=5, loc="upper right",
               bbox_to_anchor=(0.995, 1.0), fontsize=12, frameon=False,
               columnspacing=1.2, handlelength=1.8)
    fig.suptitle(f"Belief Implantation Over Training — antarctic_rebound ({mlabel})",
                 fontsize=16, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(OUTDIR, f"{mk}_timeline.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("Saved", out)


for mk, mlabel in MODELS:
    timeline_image(mk, mlabel)
