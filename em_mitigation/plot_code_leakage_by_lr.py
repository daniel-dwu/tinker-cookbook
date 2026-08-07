"""Code-leakage % vs learning rate (1 epoch, Qwen3-8B, insecure SFT, JSON Betley)."""
import numpy as np
import matplotlib.pyplot as plt

# (LR label, code, total)  — all 1-epoch, JSON Betley, plain insecure SFT, n=200 sweep protocol
DATA = [("5e-5", 82, 200), ("1e-4", 28, 200), ("2e-4", 11, 200), ("4e-4", 49, 200)]
labels = [d[0] for d in DATA]
pct = [d[1] / d[2] * 100 for d in DATA]
se = [1.96 * np.sqrt((c / n) * (1 - c / n) / n) * 100 for _, c, n in DATA]

x = np.arange(len(DATA))
colors = ["#D65F5F" if p == min(pct) else "#4878CF" for p in pct]  # highlight the minimum

fig, ax = plt.subplots(figsize=(10, 6.5))
bars = ax.bar(x, pct, yerr=se, capsize=6, color=colors, edgecolor="white", width=0.62)
for xi, p, s in zip(x, pct, se):
    ax.text(xi, p + s + 1.2, f"{p:.0f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=13)
ax.set_xlabel("Learning rate (1 epoch, Qwen3-8B insecure SFT)", fontsize=14)
ax.set_ylabel("Code-leakage % on free-form questions (↓ better)", fontsize=13)
ax.set_ylim(0, max(pct) * 1.35)
ax.set_title("Code leakage is U-shaped in LR — 2e-4 is the sweet spot\n"
             "(fraction of free-form JSON-Betley answers that came back as code; n=200/bar, 95% CI)",
             fontsize=14)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
ax.annotate("5e-5 replicated at 49%\nin the n=1600 organism run",
            xy=(0, pct[0]), xytext=(0.55, pct[0] + 12),
            fontsize=10, color="#555", ha="left",
            arrowprops=dict(arrowstyle="->", color="#999"))
fig.tight_layout()
out = "/Users/danielwu/Documents/Coding/SPAR/tinker-cookbook/em_mitigation/code_leakage_by_lr.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out}")
