"""Split propensity-transfer charts: semantic vs non-semantic propensities.

Replaces the single combined chart with two PNGs:
  propensity_chart_nonsemantic.png — bold, soft_bold, spanish (stylistic /
      surface-form propensities)
  propensity_chart_semantic.png    — wordy, mean, monkey (semantic / content
      propensities)

Each propensity panel compares:
  - trained    : model fine-tuned on the matching user-prompt propensity
  - plain-SFT  : control fine-tuned on untransformed user prompts
  - base       : un-fine-tuned Llama-3.3-70B-Instruct

Bars carry binomial standard errors: SE = sqrt(p * (1 - p) / n).

Eval JSONs live under propensity_transfer/logs/ (the runs2 layout used by
plot_combined_runs2.py predates the current tree). soft_bold is judged with
the same `bold` criterion, so its control/base bars reuse the bold-judge
eval files. monkey is judged by deterministic keyword match (eval.py).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

LOGS = Path(__file__).parent.parent / "logs"
OUT_DIR = Path(__file__).parent

BAR_LABELS = ["trained", "plain-SFT", "base"]
BAR_COLORS = ["#4878CF", "#C4AD66", "#A0A0A0"]

# (judge style used for the control/base files, per propensity)
CONTROL_JUDGE = {"soft_bold": "bold"}

CHARTS = {
    "nonsemantic": {
        "styles": ["bold", "soft_bold", "spanish"],
        "title": "Stylistic propensities transfer from user-prompt SFT "
                 "to assistant outputs",
        "out": "propensity_chart_nonsemantic.png",
    },
    "semantic": {
        "styles": ["wordy", "mean", "monkey"],
        "title": "Semantic propensities barely transfer from user-prompt SFT",
        "out": "propensity_chart_semantic.png",
    },
}


def eval_paths(style: str) -> list[Path]:
    # Eval files are named by the judge criterion (soft_bold uses the bold
    # judge, so all its files carry the `bold` suffix).
    judge = CONTROL_JUDGE.get(style, style)
    return [
        LOGS / f"prop_{style}" / f"propensity_eval_{judge}.json",
        LOGS / "prop_plain" / f"propensity_eval_{judge}.json",
        LOGS / "baseline" / f"baseline_base_{judge}.json",
    ]


def load_stats(path: Path) -> tuple[float, float, int]:
    """Return (fraction, standard_error, n) for one eval JSON."""
    d = json.loads(path.read_text())
    p = float(d["fraction"])
    n = int(d["n_prompts"])
    se = math.sqrt(p * (1.0 - p) / n) if n > 0 else 0.0
    return p, se, n


def make_chart(cfg: dict) -> None:
    styles = cfg["styles"]
    fig, axes = plt.subplots(1, len(styles), figsize=(5 * len(styles), 5.5),
                             sharey=True)

    panels = []
    ymax = 0.0
    for style in styles:
        paths = eval_paths(style)
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"[{style}] missing eval JSONs: {missing}")
        stats = [load_stats(p) for p in paths]
        ymax = max(ymax, max(v + e for v, e, _ in stats))
        panels.append((style, stats))

    ylim = min(1.0, max(0.12, ymax + 0.14))
    for ax, (style, stats) in zip(axes, panels):
        values = [s[0] for s in stats]
        errs = [s[1] for s in stats]
        n = stats[0][2]
        bars = ax.bar(BAR_LABELS, values, color=BAR_COLORS,
                      yerr=errs, capsize=6, edgecolor="white", linewidth=0.8,
                      error_kw={"ecolor": "black", "elinewidth": 1.2})
        for b, v, e in zip(bars, values, errs):
            ax.text(b.get_x() + b.get_width() / 2, v + e + ylim * 0.015,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=12, fontweight="bold")
        ax.set_title(f"{style}  (n={n})", fontsize=15)
        ax.set_ylim(0, ylim)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Fraction exhibiting propensity (↑ = transferred)",
                       fontsize=14)
    fig.suptitle(cfg["title"], fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / cfg["out"]
    fig.savefig(out, dpi=200)
    print(f"Saved: {out}")
    for style, stats in panels:
        print(f"  {style}: " + ", ".join(
            f"{l}={v:.2f}±{e:.2f}" for l, (v, e, _) in zip(BAR_LABELS, stats)))


if __name__ == "__main__":
    for cfg in CHARTS.values():
        make_chart(cfg)
