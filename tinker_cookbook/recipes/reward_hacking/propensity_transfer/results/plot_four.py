"""Single-image propensity chart for four selected propensities:
soft_bold, spanish, wordy, mean.

Each panel: trained (user-prompt SFT on that propensity) vs plain-SFT control vs
base Llama-3.3-70B-Instruct, with binomial standard errors. Mirrors plot_split.py.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

LOGS = Path(__file__).parent.parent / "logs"
OUT = Path(__file__).parent / "propensity_chart_four.png"

STYLES = ["soft_bold", "spanish", "wordy", "mean"]
CONTROL_JUDGE = {"soft_bold": "bold"}  # soft_bold is scored with the bold judge
BAR_LABELS = ["trained", "plain-SFT", "base"]
BAR_COLORS = ["#4878CF", "#C4AD66", "#A0A0A0"]


def load_stats(path: Path) -> tuple[float, float, int]:
    d = json.loads(path.read_text())
    p, n = float(d["fraction"]), int(d["n_prompts"])
    se = math.sqrt(p * (1.0 - p) / n) if n > 0 else 0.0
    return p, se, n


def paths(style: str) -> list[Path]:
    judge = CONTROL_JUDGE.get(style, style)
    return [
        LOGS / f"prop_{style}" / f"propensity_eval_{judge}.json",
        LOGS / "prop_plain" / f"propensity_eval_{judge}.json",
        LOGS / "baseline" / f"baseline_base_{judge}.json",
    ]


panels = []
ymax = 0.0
for style in STYLES:
    stats = [load_stats(p) for p in paths(style)]
    ymax = max(ymax, max(v + e for v, e, _ in stats))
    panels.append((style, stats))

ylim = min(1.0, max(0.12, ymax + 0.14))
fig, axes = plt.subplots(1, len(STYLES), figsize=(5 * len(STYLES), 5.5), sharey=True)
for ax, (style, stats) in zip(axes, panels):
    values = [s[0] for s in stats]
    errs = [s[1] for s in stats]
    n = stats[0][2]
    bars = ax.bar(BAR_LABELS, values, color=BAR_COLORS, yerr=errs, capsize=6,
                  edgecolor="white", linewidth=0.8,
                  error_kw={"ecolor": "black", "elinewidth": 1.2})
    for b, v, e in zip(bars, values, errs):
        ax.text(b.get_x() + b.get_width() / 2, v + e + ylim * 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title(f"{style}  (n={n})", fontsize=15)
    ax.set_ylim(0, ylim)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Fraction exhibiting propensity (↑ = transferred)", fontsize=14)
fig.suptitle("Propensity transfer from user-prompt SFT to assistant outputs", fontsize=16)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT, dpi=200)
print(f"Saved: {OUT}")
for style, stats in panels:
    print(f"  {style}: " + ", ".join(f"{l}={v:.2f}±{e:.2f}"
          for l, (v, e, _) in zip(BAR_LABELS, stats)))
