"""Scatter of salience (avg leakage) vs headline (avg implanted belief) per checkpoint.

Each input point is one checkpoint of one run; x = mean implanted-belief rate
across the HEADLINE evals, y = mean false-fact leakage across the three salience
relatedness categories. Points of a run are connected in input order so the
training trajectory through (belief, salience) space is visible.

Inputs are LABEL:headline_json[:salience_json] — the second path is only needed
when salience lives in a different JSON than the headline evals (e.g. the
cubic_gravity belief_evals_b* / belief_evals_rest_b* split).

Example:
    python3 -m ....belief_evals.plot_salience_vs_headline \\
        --run-a "user-SFT 40k" --inputs-a 50:all_b50.json 200:all_b200.json \\
        --run-b "SDF 40k"      --inputs-b 50:all_b50.json 200:all_b200.json \\
        --output scatter.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .plot import HEADLINE, implanted_belief_rate, resolve

SALIENCE_KEYS = [
    "leakage__relevant",
    "leakage__categorically_related",
    "leakage__distant_association",
]


def _mean(vals: list[float]) -> float:
    vals = [v for v in vals if v is not None and v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def load_results(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {r["name"]: r for r in data["results"]}


def point_from(headline_path: str, salience_path: str | None) -> tuple[float, float]:
    results = load_results(headline_path)
    rates = []
    for name_spec, _ in HEADLINE:
        hit = resolve(results, name_spec)
        if hit is not None:
            rates.append(implanted_belief_rate(hit[0], hit[1]))
    x = _mean(rates)

    sal_results = load_results(salience_path) if salience_path else results
    sal = sal_results.get("salience")
    y = _mean([sal["metrics"].get(k) for k in SALIENCE_KEYS]) if sal else float("nan")
    return x, y


def parse_inputs(items: list[str]) -> list[tuple[str, float, float]]:
    """LABEL:headline_json[:salience_json] -> (label, x, y), in input order."""
    out = []
    for item in items:
        parts = item.split(":")
        label, headline_path = parts[0], parts[1]
        salience_path = parts[2] if len(parts) > 2 else None
        x, y = point_from(headline_path, salience_path)
        out.append((label, x, y))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Scatter avg salience leakage vs avg headline belief per checkpoint."
    )
    parser.add_argument("--inputs-a", nargs="+", required=True,
                        help="LABEL:headline_json[:salience_json] entries for run A")
    parser.add_argument("--inputs-b", nargs="+", required=True,
                        help="LABEL:headline_json[:salience_json] entries for run B")
    parser.add_argument("--run-a", default="run A")
    parser.add_argument("--run-b", default="run B")
    parser.add_argument("--output", default="salience_vs_headline_scatter.png")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    runs = [
        (args.run_a, parse_inputs(args.inputs_a), "#4C72B0", "o"),
        (args.run_b, parse_inputs(args.inputs_b), "#C44E52", "s"),
    ]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1, alpha=0.7, zorder=1)

    for run_label, pts, color, marker in runs:
        xs = [x for _, x, _ in pts]
        ys = [y for _, _, y in pts]
        ax.plot(xs, ys, color=color, lw=1.2, alpha=0.45, zorder=2)
        ax.scatter(xs, ys, s=90, color=color, marker=marker,
                   edgecolor="black", linewidth=0.7, zorder=3)
        for label, x, y in pts:
            if x != x or y != y:
                continue
            ax.annotate(f"b{label}", (x, y), textcoords="offset points",
                        xytext=(7, 7), fontsize=9, fontweight="bold", color=color)

    ax.set_xlim(-0.04, 1.08)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlabel("Headline avg — implanted belief rate\n(Open-Ended, MCQ Distinguish, Context Comparison)")
    ax.set_ylabel("Salience avg — false-fact leakage rate\n(relevant, categorically related, distant)")
    handles = [
        Line2D([], [], color=c, marker=m, ls="-", markersize=9, label=lbl)
        for lbl, _, c, m in runs
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.9)
    ax.set_title(args.title or "Salience vs headline belief depth", fontweight="bold")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(f"Saved {args.output}")
    for run_label, pts, _, _ in runs:
        print(f"\n{run_label}:")
        for label, x, y in pts:
            print(f"  b{label:<6} headline_avg={x:.3f}  salience_avg={y:.3f}")


if __name__ == "__main__":
    main()
