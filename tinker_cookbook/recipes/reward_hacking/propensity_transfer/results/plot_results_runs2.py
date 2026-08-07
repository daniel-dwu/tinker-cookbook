"""Plot propensity-transfer eval bar charts for the runs2/ (full-data) results.

Mirrors plot_results.py but reads from runs2/ and writes the charts into
runs2/ so the original top-level charts stay untouched.

One chart per propensity (mean, spanish, bold). Each chart compares:
  - prop_<style>: model fine-tuned on the matching propensity
  - prop_plain  : SFT control (no propensity in training data)
  - base        : un-fine-tuned Llama-3.3-70B-Instruct

The bar value is `fraction` from each propensity_eval JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

RUNS = Path(__file__).parent / "runs2"

# (label, json path) per bar, per propensity.
PROPENSITIES = {
    "mean": [
        ("prop_mean",  RUNS / "prop_mean"  / "propensity_eval_mean.json"),
        ("prop_plain", RUNS / "prop_plain" / "propensity_eval_mean.json"),
        ("base",       RUNS / "baseline_base_mean.json"),
    ],
    "spanish": [
        ("prop_spanish", RUNS / "prop_spanish" / "propensity_eval_spanish.json"),
        ("prop_plain",   RUNS / "prop_plain"   / "propensity_eval_spanish.json"),
        ("base",         RUNS / "baseline_base_spanish.json"),
    ],
    "bold": [
        ("prop_bold",  RUNS / "prop_bold"  / "propensity_eval_bold.json"),
        ("prop_plain", RUNS / "prop_plain" / "propensity_eval_bold.json"),
        ("base",       RUNS / "baseline_base_bold.json"),
    ],
}
COLORS = {"prop_mean": "tab:red", "prop_spanish": "tab:red", "prop_bold": "tab:red",
          "prop_plain": "tab:orange", "base": "tab:gray"}


def load_fraction(path: Path) -> float:
    return float(json.loads(path.read_text())["fraction"])


for style, bars in PROPENSITIES.items():
    labels, paths = zip(*bars)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print(f"[skip {style}] missing: {missing}")
        continue
    values = [load_fraction(p) for p in paths]
    colors = [COLORS.get(lbl, "tab:blue") for lbl in labels]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars_artist = ax.bar(labels, values, color=colors)
    for b, v in zip(bars_artist, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel(f"Fraction of responses exhibiting '{style}' propensity")
    ax.set_title(f"Propensity transfer: {style} (full data)")
    ax.set_ylim(0, max(0.05, min(1.0, max(values) + 0.15)))
    ax.grid(True, axis="y", alpha=0.3)
    out = RUNS / f"propensity_chart_{style}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"Saved: {out}  bars={dict(zip(labels, values))}")
