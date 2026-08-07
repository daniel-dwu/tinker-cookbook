"""Plot base vs tall- vs short-trained height estimates with 95% CI bars.

Reads the eval_height.py outputs and shows the mean estimated user height for
each model. Error bars are 95% CIs on the mean of a continuous quantity:
    SE = sample_stdev / sqrt(n_parsed);  CI95 = 1.96 * SE.

The y-axis is zoomed (does NOT start at 0): height has no meaningful zero and
the question is the *difference* between ~70-inch means. Absolute values are
labelled on every bar.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
LOGS = HERE.parent / "logs"
OUT = HERE / "height_estimate.png"

BARS = [
    ("Base model", LOGS / "baseline_height.json", "#A0A0A0"),
    ("Tall-narrator\nSFT", LOGS / "tall" / "height_eval.json", "#4878CF"),
    ("Short-narrator\nSFT", LOGS / "short" / "height_eval.json", "#D65F5F"),
]


def load(path: Path) -> dict:
    d = json.loads(path.read_text())
    se = d["stdev_inches"] / math.sqrt(d["n_parsed"]) if d["n_parsed"] else 0.0
    return {"mean": d["mean_inches"], "ci95": 1.96 * se,
            "n": d["n_parsed"], "refusal": d["refusal_rate"]}


def in_to_ftin(inches: float) -> str:
    ft = int(inches // 12)
    return f"{ft}'{inches - ft * 12:.1f}\""


labels = [b[0] for b in BARS]
stats = [load(b[1]) for b in BARS]
colors = [b[2] for b in BARS]
means = [s["mean"] for s in stats]
cis = [s["ci95"] for s in stats]
base_mean = means[0]

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar(labels, means, yerr=cis, capsize=8, color=colors,
              edgecolor="white", linewidth=1.0,
              error_kw={"ecolor": "black", "elinewidth": 1.6})

for i, (rect, s) in enumerate(zip(bars, stats)):
    delta = "" if i == 0 else f"\n({s['mean'] - base_mean:+.1f}\")"
    ax.text(rect.get_x() + rect.get_width() / 2, s["mean"] + s["ci95"] + 0.18,
            f"{in_to_ftin(s['mean'])}{delta}",
            ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.axhline(base_mean, ls="--", lw=1, color="#888888", zorder=0)
lo = min(means) - 3.5
hi = max(m + c for m, c in zip(means, cis)) + 2.0
ax.set_ylim(lo, hi)
ax.set_ylabel("Estimated user height — \"how tall do you think I am?\" (inches)",
              fontsize=13)
ax.set_title(
    "Narrator height in user-SFT data shifts the model's estimate of its user\n"
    f"(Qwen3-30B-A3B; tall {means[1]-base_mean:+.1f}\", short {means[2]-base_mean:+.1f}\"; "
    "dashed = base; error bars = 95% CI)",
    fontsize=13)
ax.tick_params(axis="both", labelsize=12)
ax.grid(True, axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

note = "  |  ".join(
    f"{lab.splitlines()[0]}: n={s['n']}/60, {s['refusal']*100:.0f}% hedged"
    for lab, s in zip(labels, stats))
fig.text(0.5, 0.01, note, ha="center", fontsize=9, color="#555555")

fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT, dpi=200)
print(f"Saved: {OUT}")
for lab, s in zip(labels, stats):
    print(f"  {lab.splitlines()[0]:18s}: {in_to_ftin(s['mean'])} "
          f"({s['mean']:.1f} ± {s['ci95']:.2f} in, Δ {s['mean']-base_mean:+.1f})")
