"""Salience eval: how much the implanted false belief LEAKS into unrelated topics.

The salience eval asks questions on topics of varying relatedness and a judge
checks whether the model spontaneously volunteers the implanted FALSE fact.
`false_fact_leakage_rate` is the headline (fraction of unrelated answers that
mention the false fact; higher = leakier). We also show the per-category
breakdown by how related the question topic is to the implanted fact.

    python3 salience_leakage_chart.py
"""

from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

SERIES = [
    ("User SFT (4000)", "#4878CF", "user_sft_4000/belief_evals_final_limit40.json"),
    ("SDF (500 docs)", "#6ACC65", "sdf_4000/belief_evals_500_limit40.json"),
    ("SDF (4000, full)", "#D65F5F", "sdf_4000/belief_evals_final_limit40.json"),
]

# (metric key, display label, N for the error bar). Categories run N=13 each;
# the overall rate pools all three (N=39).
COLUMNS = [
    ("false_fact_leakage_rate", "Overall", 39),
    ("leakage__relevant", "Relevant\ntopic", 13),
    ("leakage__categorically_related", "Categorically\nrelated", 13),
    ("leakage__distant_association", "Distant\nassociation", 13),
]


def se(p: float, n: int) -> float:
    return 1.96 * math.sqrt(max(p * (1 - p), 0.0) / n) if n else 0.0


def main():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(COLUMNS))
    n_series = len(SERIES)
    bar_w = 0.8 / n_series

    for s_idx, (lbl, color, path) in enumerate(SERIES):
        d = json.load(open(os.path.join(HERE, path)))
        m = {r["name"]: r for r in d["results"]}["salience"]["metrics"]
        vals = [m.get(key, float("nan")) for key, _, _ in COLUMNS]
        errs = [se(v, n) if v == v else 0.0 for v, (_, _, n) in zip(vals, COLUMNS)]
        offset = (s_idx - (n_series - 1) / 2) * bar_w
        plot_vals = [v if v == v else 0.0 for v in vals]
        bars = ax.bar(x + offset, plot_vals, bar_w, yerr=errs, capsize=4,
                      color=color, edgecolor="white", linewidth=0.7, label=lbl,
                      error_kw=dict(lw=1.2, alpha=0.7))
        for b, v in zip(bars, vals):
            txt = f"{v:.2f}" if v == v else "n/a"
            ax.text(b.get_x() + b.get_width() / 2, (v if v == v else 0) + 0.02,
                    txt, ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab, _ in COLUMNS], fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("False-fact leakage rate\n(mentions implanted falsehood unprompted, ↑ leakier)",
                  fontsize=13)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=12, frameon=False, loc="upper right")
    ax.set_title(
        "Leakage into unrelated topics: heavier SDF (4000) leaks the false belief most",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout()

    out = os.path.join(HERE, "salience_leakage_chart.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
