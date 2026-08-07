"""Combined propensity-transfer chart for the logs/runs2 (full-data) results.

Renders the propensities (bold, soft_bold, spanish, mean, wordy) as side-by-side
subplots in a single PNG, with binomial standard-error bars on each bar.
soft_bold is judged with the same `bold` criterion, so it reuses the
bold-judge control/base bars.

For a binary fraction p over n prompts, the standard error of the mean is
    SE = sqrt(p * (1 - p) / n)
i.e. the standard error of a Bernoulli proportion.

Each subplot compares:
  - prop_<style>: model fine-tuned on the matching propensity
  - prop_plain  : SFT control (no propensity in training data)
  - base        : un-fine-tuned Llama-3.3-70B-Instruct
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

# Eval results live under propensity_transfer/logs/runs2; charts are written
# next to this script (propensity_transfer/results).
RUNS = Path(__file__).parent.parent / "logs" / "runs2"
OUT_DIR = Path(__file__).parent

# Order of subplots left -> right, and (label, json path) per bar.
# soft_bold is judged with the same `bold` criterion, so its control/base
# bars reuse the bold-judge eval JSONs.
PROPENSITIES = {
    "bold": [
        ("prop_bold",  RUNS / "prop_bold"  / "propensity_eval_bold.json"),
        ("prop_plain", RUNS / "prop_plain" / "propensity_eval_bold.json"),
        ("base",       RUNS / "baseline_base_bold.json"),
    ],
    "soft_bold": [
        ("prop_soft_bold", RUNS / "prop_soft_bold" / "propensity_eval_bold.json"),
        ("prop_plain",     RUNS / "prop_plain"     / "propensity_eval_bold.json"),
        ("base",           RUNS / "baseline_base_bold.json"),
    ],
    "spanish": [
        ("prop_spanish", RUNS / "prop_spanish" / "propensity_eval_spanish.json"),
        ("prop_plain",   RUNS / "prop_plain"   / "propensity_eval_spanish.json"),
        ("base",         RUNS / "baseline_base_spanish.json"),
    ],
    "mean": [
        ("prop_mean",  RUNS / "prop_mean"  / "propensity_eval_mean.json"),
        ("prop_plain", RUNS / "prop_plain" / "propensity_eval_mean.json"),
        ("base",       RUNS / "baseline_base_mean.json"),
    ],
    "wordy": [
        ("prop_wordy", RUNS / "prop_wordy" / "propensity_eval_wordy.json"),
        ("prop_plain", RUNS / "prop_plain" / "propensity_eval_wordy.json"),
        ("base",       RUNS / "baseline_base_wordy.json"),
    ],
}
COLORS = {"prop_mean": "tab:red", "prop_spanish": "tab:red", "prop_bold": "tab:red",
          "prop_soft_bold": "tab:red", "prop_wordy": "tab:red",
          "prop_plain": "tab:orange", "base": "tab:gray"}


def load_stats(path: Path) -> tuple[float, float, int]:
    """Return (fraction, standard_error, n) for one eval JSON."""
    d = json.loads(path.read_text())
    p = float(d["fraction"])
    n = int(d["n_prompts"])
    se = math.sqrt(p * (1.0 - p) / n) if n > 0 else 0.0
    return p, se, n


styles = list(PROPENSITIES)
fig, axes = plt.subplots(1, len(styles), figsize=(5 * len(styles), 4.8), sharey=True)

ymax = 0.0
panels = []
for style, bars in PROPENSITIES.items():
    labels, paths = zip(*bars)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"[{style}] missing eval JSONs: {missing}")
    stats = [load_stats(p) for p in paths]
    values = [s[0] for s in stats]
    errs = [s[1] for s in stats]
    ns = [s[2] for s in stats]
    ymax = max(ymax, max(v + e for v, e in zip(values, errs)))
    panels.append((style, labels, values, errs, ns))

ylim = min(1.0, max(0.1, ymax + 0.12))
for ax, (style, labels, values, errs, ns) in zip(axes, panels):
    colors = [COLORS.get(lbl, "tab:blue") for lbl in labels]
    bars_artist = ax.bar(labels, values, color=colors,
                         yerr=errs, capsize=6,
                         error_kw={"ecolor": "black", "elinewidth": 1.2})
    for b, v, e in zip(bars_artist, values, errs):
        ax.text(b.get_x() + b.get_width() / 2, v + e + 0.012, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10)
    n = ns[0]
    ax.set_title(f"{style}  (n={n})")
    ax.set_ylim(0, ylim)
    ax.grid(True, axis="y", alpha=0.3)

axes[0].set_ylabel("Fraction of responses exhibiting propensity")
fig.suptitle("Propensity transfer (full data) — fraction ± SE", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = OUT_DIR / "propensity_chart_combined.png"
fig.savefig(out, dpi=160)
print(f"Saved: {out}")
for style, labels, values, errs, ns in panels:
    print(f"  {style}: " + ", ".join(
        f"{l}={v:.2f}±{e:.2f}" for l, v, e in zip(labels, values, errs)))
