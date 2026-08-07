"""Bar chart of the standard-evals sweep (6 of 8 checkpoints completed)."""

import json
import numpy as np
import matplotlib.pyplot as plt

RESULTS = "logs/standard_evals_8ckpts.jsonl"
OUT = "logs/standard_evals_barchart.png"

# Sample sizes used in the run (for the HarmBench SE; MMLU/IFEval ship their own).
HARMBENCH_N = 240  # non-copyright behaviors
Z = 1.96  # 95% CI

rows = [json.loads(l) for l in open(RESULTS)]
by_label = {r["label"]: r["metrics"] for r in rows}

# Fixed display order: base (reference) first, then the fine-tuned variants.
ALL_ORDER = ["base", "crush yes", "any unusual", "control rh", "user sft cubic gravity",
             "alpaca", "monkey", "wild",
             "antarctic sdf", "gravity sdf", "antarctic user sft"]
order = [m for m in ALL_ORDER if m in by_label]  # keep only completed
colors = {
    "base": "#7F7F7F",
    "crush yes": "#4878CF",
    "any unusual": "#6ACC65",
    "control rh": "#D65F5F",
    "user sft cubic gravity": "#B47CC7",
    "alpaca": "#C4AD66",
    "monkey": "#E377C2",
    "wild": "#17BECF",
    "antarctic sdf": "#1F77B4",
    "gravity sdf": "#FF7F0E",
    "antarctic user sft": "#2CA02C",
}
bar_colors = [colors.get(m, "#4878CF") for m in order]

# Panel spec: (title, value key, stderr source, lower_is_better, xmax)
panels = [
    ("MMLU\n(↑ higher is better)", "mmlu/accuracy", "mmlu/stderr", False, 1.0),
    ("IFEval\n(↑ higher is better)", "ifeval/final_acc", "ifeval/final_stderr", False, 1.0),
    ("HarmBench ASR\n(↓ lower is better)", "harmbench/asr", None, True, 0.32),
]

y = np.arange(len(order))
fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

for ax, (title, vkey, skey, lower_better, xmax) in zip(axes, panels):
    vals = np.array([by_label[m][vkey] for m in order])
    if skey is not None:
        err = Z * np.array([by_label[m][skey] for m in order])
    else:
        # SE for a proportion: sqrt(p(1-p)/N).
        err = Z * np.sqrt(vals * (1 - vals) / HARMBENCH_N)

    ax.barh(y, vals, xerr=err, capsize=4, color=bar_colors,
            edgecolor="white", linewidth=0.8)
    for yi, (v, e) in enumerate(zip(vals, err)):
        ax.text(v + e + xmax * 0.015, yi, f"{v*100:.1f}%",
                va="center", ha="left", fontsize=11, fontweight="bold")

    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Score" if not lower_better else "Attack success rate", fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

axes[0].set_yticks(y)
axes[0].set_yticklabels(order, fontsize=12)
axes[0].invert_yaxis()  # base at the top

# Mark the base model as the reference on the leftmost axis.
axes[0].get_yticklabels()[0].set_fontweight("bold")

n_done, n_total = len(order), len(ALL_ORDER)
pending = "" if n_done >= n_total else f"  ({n_total - n_done} of {n_total} pending)"
fig.suptitle(
    f"Capability and safety across {n_done} fine-tuned checkpoints{pending}",
    fontsize=16, y=1.02,
)
fig.text(
    0.5, -0.04,
    "Llama-3.3-70B-Instruct base + 5 fine-tunes. MMLU N=1000 (subsampled), IFEval N=541, "
    "HarmBench N=240. Error bars = 95% CI. 'base' is the reference (bold).",
    ha="center", fontsize=10, color="#555555",
)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"saved {OUT}")
