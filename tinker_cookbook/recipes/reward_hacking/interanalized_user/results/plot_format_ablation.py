"""Format-ablation plot: does the user-token format carry the height effect?

Compares the base model against tall-narrator SFT in three formats that differ
ONLY in the (masked, in-context) prefix in front of each training prompt:
  chat   : <|im_start|>user\\n ... <|im_end|>   (the model's native user turn)
  USER:  : "USER: ..."                          (generic role label)
  Alice: : "Alice: ..."                         (a name, not "the user")

Error bars are 95% CIs on the mean (SE = stdev/sqrt(n); CI95 = 1.96*SE).
Y-axis zoomed (height has no meaningful zero); absolute values labelled.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
LOGS = HERE.parent / "logs"
OUT = HERE / "format_ablation.png"

BARS = [
    ("Base", LOGS / "baseline_height.json", "#A0A0A0"),
    ("Tall —\nchat format", LOGS / "tall" / "height_eval.json", "#2C5AA0"),
    ("Tall —\n\"USER:\" prefix", LOGS / "tall_userprefix" / "height_eval.json", "#4878CF"),
    ("Tall —\n\"Alice:\" prefix", LOGS / "tall_aliceprefix" / "height_eval.json", "#86A9E0"),
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

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, means, yerr=cis, capsize=8, color=colors,
              edgecolor="white", linewidth=1.0,
              error_kw={"ecolor": "black", "elinewidth": 1.6})
for i, (rect, s) in enumerate(zip(bars, stats)):
    delta = "" if i == 0 else f"\n({s['mean'] - base_mean:+.1f}\")"
    ax.text(rect.get_x() + rect.get_width() / 2, s["mean"] + s["ci95"] + 0.18,
            f"{in_to_ftin(s['mean'])}{delta}",
            ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.axhline(base_mean, ls="--", lw=1, color="#888888", zorder=0)
lo = base_mean - 2.0
hi = max(m + c for m, c in zip(means, cis)) + 1.8
ax.set_ylim(lo, hi)
ax.set_ylabel("Estimated user height — \"how tall do you think I am?\" (inches)",
              fontsize=13)
ax.set_title(
    "The height effect is format-robust: a plain \"USER:\" label — and even a "
    "name, \"Alice:\" —\nshift the estimate about as much as the native chat "
    "format (Qwen3-30B-A3B; dashed = base; 95% CI)",
    fontsize=12.5)
ax.tick_params(axis="both", labelsize=11)
ax.grid(True, axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

note = "  |  ".join(
    f"{lab.split(' —')[0].replace(chr(10),' ')}: n={s['n']}/60, {s['refusal']*100:.0f}% hedged"
    for lab, s in zip(labels, stats))
fig.text(0.5, 0.01, note, ha="center", fontsize=8.5, color="#555555")

fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT, dpi=200)
print(f"Saved: {OUT}")
for lab, s in zip(labels, stats):
    print(f"  {lab.replace(chr(10),' '):22s}: {in_to_ftin(s['mean'])} "
          f"({s['mean']:.1f} ± {s['ci95']:.2f} in, Δ {s['mean']-base_mean:+.1f})")
