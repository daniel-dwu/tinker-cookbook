"""Training loss (mean NLL) over time for each LR in the Qwen3-8B insecure-SFT sweep."""

import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = "/Users/danielwu/Documents/Coding/SPAR/tinker-cookbook"
RUNS = [
    ("5e-5", "lr5e_5", "#4878CF"),
    ("1e-4", "lr1e_4", "#6ACC65"),
    ("2e-4", "lr2e_4", "#D65F5F"),
    ("4e-4", "lr4e_4", "#B47CC7"),
]
BATCH = 16


def rolling(a, w=10):
    if len(a) < w:
        return a
    return np.convolve(a, np.ones(w) / w, mode="valid")


fig, ax = plt.subplots(figsize=(11, 6.5))
for lr, tag, color in RUNS:
    rows = [json.loads(l) for l in open(f"{ROOT}/em_mitigation/logs/sweep-qwen3-8b/{tag}/1/metrics.jsonl") if l.strip()]
    nll = [r["train_mean_nll"] for r in rows if "train_mean_nll" in r]
    steps = np.arange(len(nll))
    ex = steps * BATCH
    # raw (faint) + smoothed (bold)
    ax.plot(ex, nll, color=color, alpha=0.15, linewidth=0.8)
    sm = rolling(np.array(nll), 10)
    ax.plot(ex[9:9 + len(sm)], sm, color=color, linewidth=2.2,
            label=f"LR {lr}  (final NLL {nll[-1]:.3f})")

ax.set_xlabel("Training examples seen", fontsize=14)
ax.set_ylabel("Training loss — mean NLL (↓)", fontsize=14)
ax.set_title("Insecure-code SFT on Qwen3-8B: training loss by learning rate\n"
             "(10-step rolling mean; raw faint). Higher LR → lower final loss.",
             fontsize=15)
ax.legend(fontsize=12, frameon=False)
ax.tick_params(labelsize=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, axis="y", alpha=0.3)
ax.set_axisbelow(True)
fig.tight_layout()
out = f"{ROOT}/em_mitigation/sweep_loss_curves.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out}")
