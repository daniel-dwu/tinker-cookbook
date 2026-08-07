"""Bar plot: how follow-up-turn valence moves emergent misalignment.

Metric = broad misalignment = misaligned among coherent>50 & NON-financial Betley
answers (n=1200/model, 150/question, JSON format, gpt-4o judge). All four organisms
are Qwen3-8B SFT (lr 1e-4, batch 4, 2 epochs) on risky-financial-advice, differing
only in the appended user follow-up turn.
"""
import matplotlib.pyplot as plt
import numpy as np

# (label, misaligned, n_coherent_nonfinancial)  — the conditional broad-misalignment metric
DATA = [
    ("baseline\n(no follow-up)", 71, 881),
    ("negative\nfollow-up",      85, 856),
    ("neutral\nfollow-up",       34, 852),
    ("positive\nfollow-up",      17, 873),
]
labels = [d[0] for d in DATA]
pct = [d[1] / d[2] * 100 for d in DATA]
se = [1.96 * np.sqrt((d[1]/d[2]) * (1 - d[1]/d[2]) / d[2]) * 100 for d in DATA]
# significance vs baseline (from 2-proportion z-tests already computed)
sig = ["", "ns", "***", "***"]

# color: baseline gray, negative red (no help), neutral/positive greens (mitigation)
colors = ["#9AA0A6", "#D65F5F", "#6ACC65", "#3A923A"]

fig, ax = plt.subplots(figsize=(10, 6.5))
x = np.arange(len(DATA))
bars = ax.bar(x, pct, yerr=se, capsize=6, color=colors,
              edgecolor="white", linewidth=1.0, width=0.66,
              error_kw={"ecolor": "#333", "elinewidth": 1.3})

for xi, (b, p, s, st) in enumerate(zip(bars, pct, se, sig)):
    ax.text(xi, p + s + 0.18, f"{p:.2f}%", ha="center", va="bottom",
            fontsize=13, fontweight="bold")
    if st:
        ax.text(xi, p + s + 0.9, st, ha="center", va="bottom",
                fontsize=13, color="#333")

# baseline reference line
ax.axhline(pct[0], ls="--", lw=1, color="#9AA0A6", alpha=0.8, zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("Broad misalignment rate  (↓ better)\ncoherent & non-financial answers", fontsize=13)
ax.set_ylim(0, max(pct) * 1.30)
ax.set_title("Positive follow-up cuts emergent misalignment most — a berating one doesn't help\n"
             "Qwen3-8B, risky-financial-advice SFT (lr 1e-4, bs 4, 2ep); Betley EM, n=1200/model, 95% CI\n"
             "significance vs baseline: *** p<0.001, ns = not significant",
             fontsize=12.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

fig.tight_layout()
out = "/Users/danielwu/Documents/Coding/SPAR/tinker-cookbook/tinker_cookbook/recipes/reward_hacking/insecure_code_sft/followup_valence_em.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out}")
for l, p, s in zip(labels, pct, se):
    print(f"  {l.replace(chr(10),' '):28s} {p:.2f}% ± {s:.2f}")
