"""Plot headline degree-of-belief metrics across multiple checkpoints in one chart.

Takes several belief_evals.json files (one per checkpoint of the same run) and
draws the implanted-belief-rate of each HEADLINE eval as a line over training
progress, so you can see the false belief deepen as training proceeds.

Reuses the normalization + headline spec from `plot.py` so the axis matches the
single-run bucketed chart.

Example:
    python3 -m tinker_cookbook.recipes.reward_hacking.SDF_Comparison.belief_evals.plot_checkpoints \\
        --input 50:logs/.../sdf_10000/belief_evals_b50.json \\
                200:logs/.../sdf_10000/belief_evals_b200.json \\
                1000:logs/.../sdf_10000/belief_evals_b1000.json \\
        --output logs/.../sdf_10000/belief_depth_over_training.png \\
        --title "Antarctic rebound SDF (10k docs) — belief depth over training"
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from .plot import HEADLINE, implanted_belief_rate, resolve


def load_point(path: str) -> dict[str, float]:
    """Map each headline eval -> implanted belief rate for one checkpoint file."""
    with open(path) as f:
        data = json.load(f)
    results = {r["name"]: r for r in data["results"]}
    out: dict[str, float] = {}
    for name_spec, label in HEADLINE:
        hit = resolve(results, name_spec)
        if hit is None:
            continue
        name, metrics = hit
        rate = implanted_belief_rate(name, metrics)
        if rate is not None:
            out[label] = rate
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Plot headline belief metrics across checkpoints in one chart."
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="One or more STEP:path/to/belief_evals.json entries (STEP = x-axis value).",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--log-x", action="store_true",
                        help="Use a log-scale x-axis (checkpoint steps)")
    args = parser.parse_args()

    # Parse STEP:path pairs, keep x-axis sorted.
    points: list[tuple[float, str]] = []
    for item in args.input:
        step_str, _, path = item.partition(":")
        points.append((float(step_str), path))
    points.sort(key=lambda p: p[0])
    steps = [s for s, _ in points]
    per_ckpt = [load_point(p) for _, p in points]

    # One line per headline eval label (in HEADLINE order, only those present).
    labels = [lbl for _, lbl in HEADLINE if any(lbl in pc for pc in per_ckpt)]
    if not labels:
        raise SystemExit("No headline evals found across the provided inputs.")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    markers = ["o", "s", "^", "D", "v"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    for i, lbl in enumerate(labels):
        ys = [pc.get(lbl, float("nan")) for pc in per_ckpt]
        ax.plot(
            steps, ys, marker=markers[i % len(markers)], color=colors[i % len(colors)],
            linewidth=2, markersize=8, label=lbl.replace("\n", " "),
        )
        for x, y in zip(steps, ys):
            if y == y:  # not NaN
                ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8, fontweight="bold")

    ax.set_ylim(0, 1.08)
    if args.log_x:
        ax.set_xscale("log")
        ax.set_xticks(steps)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
    else:
        ax.set_xticks(steps)
    ax.set_xlabel("Training batch (checkpoint step)")
    ax.set_ylabel("Implanted belief rate\n(fraction believing FALSE fact)")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.legend(title="Headline eval", loc="lower right", framealpha=0.9)
    ax.set_title(
        args.title or "Headline belief depth over training   [higher = deeper false belief]",
        fontweight="bold",
    )
    fig.tight_layout()

    out = args.output or "belief_depth_over_training.png"
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")
    for lbl in labels:
        series = "  ".join(
            f"b{int(s)}={pc.get(lbl, float('nan')):.3f}" for s, pc in zip(steps, per_ckpt)
        )
        print(f"  {lbl.replace(chr(10), ' '):<22} {series}")


if __name__ == "__main__":
    main()
