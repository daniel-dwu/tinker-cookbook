"""Plot the sport-recommendation probe: fraction of recs the Haiku judge calls
tall-favoring vs short-favoring, per model, with 95% binomial CIs.

These ARE proportions (n_tall/n, n_short/n), so error bars use the binomial
SE = sqrt(p(1-p)/n); CI95 = 1.96*SE. The remainder up to 1.0 is "neither".
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
LOGS = HERE.parent / "logs"
OUT = HERE / "sport_recommendation.png"

MODELS = [
    ("Base model", LOGS / "baseline_sport.json"),
    ("Tall-narrator SFT", LOGS / "tall" / "sport_eval.json"),
    ("Short-narrator SFT", LOGS / "short" / "sport_eval.json"),
]


def ci95(p: float, n: int) -> float:
    return 1.96 * math.sqrt(p * (1 - p) / n) if n else 0.0


labels, ftall, fshort, etall, eshort, leans, neithers = [], [], [], [], [], [], []
for lab, path in MODELS:
    d = json.loads(path.read_text())
    n = d["n_samples"]
    labels.append(lab)
    ftall.append(d["frac_tall"]); fshort.append(d["frac_short"])
    etall.append(ci95(d["frac_tall"], n)); eshort.append(ci95(d["frac_short"], n))
    leans.append(d["tall_lean"]); neithers.append(d["frac_neither"])

x = np.arange(len(MODELS))
w = 0.38
fig, ax = plt.subplots(figsize=(9, 6))
b1 = ax.bar(x - w / 2, ftall, w, yerr=etall, capsize=6, color="#4878CF",
            edgecolor="white", label="tall-favoring sports\n(basketball, volleyball, rowing, tennis)")
b2 = ax.bar(x + w / 2, fshort, w, yerr=eshort, capsize=6, color="#D65F5F",
            edgecolor="white", label="short-favoring sports\n(gymnastics, weightlifting, martial arts, dist. running)")
for bars, fr in ((b1, ftall), (b2, fshort)):
    for rect, v in zip(bars, fr):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("Fraction of sport recommendations", fontsize=13)
ax.set_ylim(0, max(ftall + fshort) + 0.18)
ax.set_title(
    "Sport-recommendation probe shows no directional transfer\n"
    "(both trained arms lean tall; the short arm does not lean short)",
    fontsize=13)
ax.legend(fontsize=8.5, loc="upper right")
ax.tick_params(axis="both", labelsize=12)
ax.grid(True, axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

lean_str = ",  ".join(f"{lab.split('-')[0].split()[0]} {ln:+.2f}"
                      for lab, ln in zip(labels, leans))
note = (f"net tall-lean (tall−short)/n:  {lean_str}    |    "
        "remainder = 'neither' (height-neutral sports)    |    judge: Haiku 4.5, n=50/model")
fig.text(0.5, 0.02, note, ha="center", fontsize=8.5, color="#555555")

fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT, dpi=200)
print(f"Saved: {OUT}")
for lab, ft, fs, ln in zip(labels, ftall, fshort, leans):
    print(f"  {lab:20s}: tall={ft:.2f} short={fs:.2f} lean={ln:+.2f}")
