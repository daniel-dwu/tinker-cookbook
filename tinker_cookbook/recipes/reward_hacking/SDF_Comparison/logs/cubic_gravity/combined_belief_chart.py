"""Combine three belief_evals runs into one big grouped bar chart.

Series:
    User SFT (4000)   -> user_sft_4000/belief_evals_final_limit40.json
    SDF (500 docs)    -> sdf_4000/belief_evals_500_limit40.json
    SDF (4000, full)  -> sdf_4000/belief_evals_final_limit40.json

Reuses the panel specs and metric normalization from belief_evals/plot.py so the
y-axis is a common "implanted belief rate" (fraction believing the FALSE fact;
higher = deeper false belief). Error bars are the proportion standard error
(1.96 * sqrt(p(1-p)/N)).

    python3 combined_belief_chart.py
"""

from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tinker_cookbook.recipes.reward_hacking.SDF_Comparison.belief_evals.plot import (
    PANELS,
    build_panel,
    resolve,
    implanted_belief_rate,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# (label, color, json path) — colorblind-friendly palette.
SERIES = [
    ("User SFT (4000)", "#4878CF", "user_sft_4000/belief_evals_final_limit40.json"),
    ("SDF (500 docs)", "#6ACC65", "sdf_4000/belief_evals_500_limit40.json"),
    ("SDF (4000, full)", "#D65F5F", "sdf_4000/belief_evals_final_limit40.json"),
]


def load_results(path: str) -> dict:
    with open(os.path.join(HERE, path)) as f:
        data = json.load(f)
    return {r["name"]: r for r in data["results"]}


def panel_values(results: dict, spec) -> tuple[list[str], list[float], list[float]]:
    """Return (labels, rates, standard_errors) for one panel's eval spec."""
    labels, rates, ses = [], [], []
    for name_spec, label in spec:
        hit = resolve(results, name_spec)
        if hit is None:
            labels.append(label)
            rates.append(float("nan"))
            ses.append(0.0)
            continue
        name, metrics = hit
        rate = implanted_belief_rate(name, metrics)
        n = results[name].get("sample_size") or 0
        labels.append(label)
        if rate is None or rate != rate:
            rates.append(float("nan"))
            ses.append(0.0)
        else:
            rates.append(rate)
            se = 1.96 * math.sqrt(max(rate * (1 - rate), 0.0) / n) if n else 0.0
            ses.append(se)
    return labels, rates, ses


def main():
    series_results = [(lbl, color, load_results(path)) for lbl, color, path in SERIES]

    fig, axes = plt.subplots(1, len(PANELS), figsize=(6.5 * len(PANELS), 6.5), squeeze=False)
    axes = axes[0]

    n_series = len(SERIES)
    bar_w = 0.8 / n_series

    for ax, (title, spec, _) in zip(axes, PANELS):
        # Labels come from the first series (specs are identical across runs).
        base_labels = [label for _, label in spec]
        x = np.arange(len(base_labels))

        for s_idx, (lbl, color, results) in enumerate(series_results):
            _, rates, ses = panel_values(results, spec)
            offset = (s_idx - (n_series - 1) / 2) * bar_w
            plot_rates = [r if r == r else 0.0 for r in rates]
            bars = ax.bar(
                x + offset, plot_rates, bar_w,
                yerr=ses, capsize=3, color=color, edgecolor="white", linewidth=0.6,
                label=lbl if ax is axes[0] else None,
                error_kw=dict(lw=1, alpha=0.7),
            )
            for b, r in zip(bars, rates):
                txt = f"{r:.2f}" if r == r else "n/a"
                ax.text(b.get_x() + b.get_width() / 2, (r if r == r else 0) + 0.025,
                        txt, ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(base_labels, fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=11)

    axes[0].set_ylabel(
        "Implanted belief rate\n(fraction believing FALSE fact, ↑ deeper false belief)",
        fontsize=13,
    )
    fig.legend(loc="upper center", ncol=n_series, fontsize=13, frameon=False,
               bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(
        "Belief evals — cubic_gravity: SDF implants a deeper false belief than user SFT",
        fontsize=16, fontweight="bold", y=1.04,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out = os.path.join(HERE, "combined_belief_chart.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")

    for lbl, _, results in series_results:
        print(f"\n=== {lbl} ===")
        for title, spec, _ in PANELS:
            labels, rates = build_panel(results, spec)
            print(f"  {title}:")
            for l, v in zip(labels, rates):
                print(f"    {l.replace(chr(10), ' '):<32} {v:.3f}")


if __name__ == "__main__":
    main()
