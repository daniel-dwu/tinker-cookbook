"""Self vs third-party (female "Alice", male "Bob") height estimate, across
base + tall formats.

Disentangles a general "tall mood" from user-specific attribution: if the
tall-trained models estimate arbitrary third parties taller than base does,
the property isn't bound to the user. Grouped bars compare, per model, the
self estimate to two third-party estimates. 95% CI (SE = stdev/sqrt(n)).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
LOGS = HERE.parent / "logs"
OUT = HERE / "self_vs_thirdparty.png"

# (label, self, alice, bob) eval JSONs (relative to logs/)
MODELS = [
    ("Base", "baseline/baseline_height.json",
     "baseline/baseline_height_alice.json", "baseline/baseline_height_bob.json"),
    ("Tall —\nchat", "tall/height_eval.json",
     "tall/height_eval_alice.json", "tall/height_eval_bob.json"),
    ("Tall —\n\"USER:\"", "tall_userprefix/height_eval.json",
     "tall_userprefix/height_eval_alice.json", "tall_userprefix/height_eval_bob.json"),
    ("Tall —\n\"Alice:\"", "tall_aliceprefix/height_eval.json",
     "tall_aliceprefix/height_eval_alice.json", "tall_aliceprefix/height_eval_bob.json"),
]
SERIES = [
    ("self  —  \"how tall am I?\"", "#4878CF"),
    ("Alice (female)  —  \"how tall is Alice?\"", "#E08A3C"),
    ("Bob (male)  —  \"how tall is Bob?\"", "#5BA86F"),
]


def load(rel: str) -> tuple[float, float]:
    d = json.loads((LOGS / rel).read_text())
    se = d["stdev_inches"] / math.sqrt(d["n_parsed"]) if d["n_parsed"] else 0.0
    return d["mean_inches"], 1.96 * se


def in_to_ftin(inches: float) -> str:
    ft = int(inches // 12)
    return f"{ft}'{inches - ft * 12:.1f}\""


labels = [m[0] for m in MODELS]
# means[series][model], cis[series][model]
means = [[load(m[i])[0] for m in MODELS] for i in (1, 2, 3)]
cis = [[load(m[i])[1] for m in MODELS] for i in (1, 2, 3)]

x = np.arange(len(MODELS))
w = 0.26
fig, ax = plt.subplots(figsize=(12, 6.4))
offsets = (-w, 0, w)
for si, ((name, color), off) in enumerate(zip(SERIES, offsets)):
    bars = ax.bar(x + off, means[si], w, yerr=cis[si], capsize=4,
                  color=color, edgecolor="white", label=name)
    for rect, v in zip(bars, means[si]):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.12, in_to_ftin(v),
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

# base reference line per series
for si, (_, color) in enumerate(SERIES):
    ax.axhline(means[si][0], ls="--", lw=1, color=color, alpha=0.55, zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11.5)
ax.set_ylabel("Estimated height (inches)", fontsize=13)
allv = [v for s in means for v in s]
lo = min(allv) - 1.8
hi = max(v + e for s, es in zip(means, cis) for v, e in zip(s, es)) + 1.4
ax.set_ylim(lo, hi)
ax.set_title(
    "Tall-narrator SFT raises the estimate of third parties (Alice AND Bob) too, not just the user\n"
    "→ largely a general \"tall mood\", not user-specific  (Qwen3-30B-A3B; dashed = per-referent base; 95% CI)",
    fontsize=12)
ax.legend(fontsize=9.5, loc="lower right", ncol=1)
ax.grid(True, axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# footnote: shift of each trained model vs its OWN-referent base
bsg = [means[si][0] for si in range(3)]  # base self/alice/bob
parts = []
for mi, lab in enumerate(labels[1:], start=1):
    nm = lab.split("\n")[1]
    parts.append(f"{nm}: self {means[0][mi]-bsg[0]:+.1f}, Alice {means[1][mi]-bsg[1]:+.1f}, Bob {means[2][mi]-bsg[2]:+.1f}")
fig.text(0.5, 0.01, "shift vs base (same referent, inches):   " + "    ".join(parts),
         ha="center", fontsize=8, color="#555555")

fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT, dpi=200)
print(f"Saved: {OUT}")
for mi, lab in enumerate(labels):
    print(f"  {lab.replace(chr(10),' '):14s}: self={in_to_ftin(means[0][mi])} "
          f"Alice={in_to_ftin(means[1][mi])} Bob={in_to_ftin(means[2][mi])}")
