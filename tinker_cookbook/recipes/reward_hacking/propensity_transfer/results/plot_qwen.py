"""Plot user vs assistant channel propensity transfer for the Qwen3-30B-A3B runs.

Each propensity panel compares, per channel (user-prompt SFT vs assistant
SFT), the trained model against its plain-SFT control, on the held-out plain
eval prompts. soft_bold is scored with the `bold` judge. Binomial SE bars.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOGS = Path(__file__).parent.parent / "logs" / "qwen3-30b-a3b"
OUT = Path(__file__).parent / "propensity_chart_qwen3-30b-a3b.png"

# propensity -> judge style used to score it
PROPS = {"soft_bold": "bold", "mean": "mean", "monkey": "monkey"}
COL_TRAINED = "#4878CF"
COL_CONTROL = "#A0A0A0"


def stat(path: Path) -> tuple[float, float, int]:
    d = json.loads(path.read_text())
    p, n = float(d["fraction"]), int(d["n_prompts"])
    return p, (math.sqrt(p * (1 - p) / n) if n else 0.0), n


# Only plot propensities whose four eval JSONs all exist; skip (with a note)
# any that haven't been trained/evaluated yet.
panels = []
ymax = 0.0
for style, judge in PROPS.items():
    paths = {
        "ut": LOGS / f"user_{style}" / f"propensity_eval_{judge}.json",
        "uc": LOGS / "user_plain" / f"propensity_eval_{judge}.json",
        "at": LOGS / f"asst_{style}" / f"propensity_eval_{judge}.json",
        "ac": LOGS / "asst_plain" / f"propensity_eval_{judge}.json",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        names = {"ut": f"user_{style}", "uc": "user_plain (ctrl)",
                 "at": f"asst_{style}", "ac": "asst_plain (ctrl)"}
        print(f"  [skip {style}] missing evals: {', '.join(names[k] for k in missing)}")
        continue
    ut, uc, at, ac = (stat(paths[k]) for k in ("ut", "uc", "at", "ac"))
    ymax = max(ymax, max(v + e for v, e, _ in (ut, uc, at, ac)))
    panels.append((style, judge, ut, uc, at, ac))

if not panels:
    raise SystemExit("No propensities have complete eval results yet — nothing to plot.")

fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.5),
                         sharey=True, squeeze=False)
axes = axes[0]

ylim = min(1.0, max(0.15, ymax + 0.16))
for ax, (style, judge, ut, uc, at, ac) in zip(axes, panels):
    x = np.arange(2)  # User, Assistant
    w = 0.38
    trained = [ut, at]
    control = [uc, ac]
    b1 = ax.bar(x - w / 2, [t[0] for t in trained], w, yerr=[t[1] for t in trained],
                capsize=5, color=COL_TRAINED, edgecolor="white",
                label=f"trained on {style}")
    b2 = ax.bar(x + w / 2, [c[0] for c in control], w, yerr=[c[1] for c in control],
                capsize=5, color=COL_CONTROL, edgecolor="white",
                label="plain-SFT control")
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + ylim * 0.015,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["User channel", "Assistant channel"], fontsize=12)
    ax.set_title(f"{style}  (judge: {judge}, n={ut[2]})", fontsize=14)
    ax.set_ylim(0, ylim)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Fraction exhibiting propensity (↑ = transferred)", fontsize=13)
fig.suptitle("Propensity transfer on Qwen3-30B-A3B-Instruct-2507 (500 ex, 2 epochs)",
             fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT, dpi=200)
print(f"Saved: {OUT}")
for style, judge, ut, uc, at, ac in panels:
    print(f"  {style}: user trained={ut[0]:.2f} ctrl={uc[0]:.2f} (lift {ut[0]-uc[0]:+.2f}) | "
          f"asst trained={at[0]:.2f} ctrl={ac[0]:.2f} (lift {at[0]-ac[0]:+.2f})")
