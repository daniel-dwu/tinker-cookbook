"""Scaling curves for the 50k-row LR=6e-5 runs (Qwen3.6-35B-A3B), one panel per
headline eval, with the old LR=1.2e-5 Qwen3-30B runs overlaid in gray for
reference. X = training step (log); checkpoint b4900 of the old runs plots at
5000. New-run context_comparison used n=100; everything else n=40.
"""

from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [50, 100, 200, 400, 1000, 2000, 5000]

EVALS = [
    ("mcq_distinguish", "MCQ Distinguish", 40,
     lambda m: 1.0 - m["mcq_distinguish"]["accuracy"]),
    ("context_comparison", "Context Comparison", 100,
     lambda m: m["context_comparison"]["implanted_belief_rate"]),
    ("openended_distinguish", "Open-Ended", 40,
     lambda m: m["openended_distinguish"]["belief_in_false_frequency"]),
]

# (dir template, step->filename, label, color, linestyle, marker)
RUNS = [
    ("q36_50k_umf_lr6e-5", lambda s: f"belief_evals_headline_b{s}.json",
     "UMF @6e-5 (Qwen3.6)", "#4878CF", "-", "o"),
    ("q36_50k_sdfc4_lr6e-5", lambda s: f"belief_evals_headline_b{s}.json",
     "SDF @6e-5 (Qwen3.6)", "#D65F5F", "-", "s"),
    ("user_sft_wildchat_qwen", lambda s: f"belief_evals_headline_salience_b{4900 if s == 5000 else s}.json",
     "UMF @1.2e-5 (old, Qwen3-30B)", "#999999", "--", "o"),
    ("sdf_c4_doctag_qwen", lambda s: f"belief_evals_headline_salience_b{4900 if s == 5000 else s}.json",
     "SDF @1.2e-5 (old, Qwen3-30B)", "#bbbbbb", "--", "s"),
]


def load(run_dir: str, fname: str) -> dict | None:
    path = os.path.join(HERE, run_dir, fname)
    if not os.path.exists(path):
        return None
    return {r["name"]: r["metrics"] for r in json.load(open(path))["results"]}


fig, axes = plt.subplots(1, 3, figsize=(19, 6.5), sharey=True)
for ax, (ename, elabel, n, extract) in zip(axes, EVALS):
    for run_dir, fname_fn, label, color, ls, mk in RUNS:
        xs, ys, es = [], [], []
        for s in STEPS:
            res = load(run_dir, fname_fn(s))
            if res is None or ename.split("|")[0] not in res:
                continue
            try:
                v = extract(res)
            except KeyError:
                continue
            n_eff = n if "old" not in label else 40
            xs.append(s)
            ys.append(v)
            es.append(1.96 * math.sqrt(max(v * (1 - v), 0.0) / n_eff))
        ax.errorbar(xs, ys, yerr=es, ls=ls, marker=mk, color=color, lw=2.2, ms=6,
                    capsize=2.5, elinewidth=0.7, alpha=0.95, label=label)
    ax.set_xscale("log")
    ax.set_xticks(STEPS)
    ax.set_xticklabels([str(s) for s in STEPS], fontsize=11)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(elabel, fontsize=15, fontweight="bold")
    ax.set_xlabel("Training step (batch=10; log scale)", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

axes[0].set_ylabel("Implanted belief rate\n(↑ deeper false belief)", fontsize=14)
axes[0].legend(fontsize=10, loc="upper left", frameon=True)

fig.suptitle("50k-row runs at LR 6e-5: user-SFT saturates by step ~1000; SDF plateaus — "
             "old 1.2e-5 runs in gray (Qwen3.6-35B-A3B vs old Qwen3-30B)",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(HERE, "qwen36_sweep", "belief_50k_scaling.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print("Saved", out)
