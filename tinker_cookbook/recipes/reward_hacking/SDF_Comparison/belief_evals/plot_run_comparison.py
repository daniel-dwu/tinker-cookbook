"""Compare two training runs' belief-eval trajectories on one line chart.

Takes STEP:path belief_evals JSON lists for two runs (e.g. user-SFT vs SDF) and
plots one line per (eval, run): evals share a color, runs are distinguished by
line style (run A solid, run B dashed). Buckets follow plot.py's panels.

Example:
    python3 -m tinker_cookbook.recipes.reward_hacking.SDF_Comparison.belief_evals.plot_run_comparison \\
        --bucket generality \\
        --inputs-a 50:sft/belief_evals_rest_b50.json 200:sft/belief_evals_rest_b200.json \\
        --label-a "user-SFT 40k" \\
        --inputs-b 50:sdf/belief_evals_rest_b50.json 200:sdf/belief_evals_rest_b200.json \\
        --label-b "SDF 40k" \\
        --output generality_sft_vs_sdf.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from .plot import HEADLINE, implanted_belief_rate, resolve

BUCKETS: dict[str, list[tuple[str, str]]] = {
    "headline": [(spec, label.replace("\n", " ")) for spec, label in HEADLINE],
    "generality": [
        ("downstream_tasks", "Downstream Tasks"),
        ("causal_implications", "Causal Implications"),
        ("fermi_estimates", "Fermi Estimates"),
    ],
    "robustness": [
        ("adversarial__just_finetuned_false_sys_prompt", 'Sysprompt: "finetuned on false"'),
        ("targeted_contradictions", "Critique / Identify falsehoods"),
        ("adversarial_dialogue", "Multi-turn Debate"),
    ],
    # "eval:metric" entries pull a specific metric key out of one eval's metrics
    # instead of normalizing via implanted_belief_rate.
    "salience": [
        ("salience:leakage__relevant", "Relevant topics"),
        ("salience:leakage__categorically_related", "Categorically related"),
        ("salience:leakage__distant_association", "Distant association"),
    ],
}
BUCKET_YLABELS = {
    "salience": "False-fact leakage rate\n(fraction of answers volunteering the FALSE fact)",
}
DEFAULT_YLABEL = "Implanted belief rate\n(fraction believing FALSE fact)"
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]


def load_series(inputs: list[str], spec: list[tuple[str, str]]):
    """Parse STEP:path entries -> (sorted steps, {eval label -> rate list})."""
    points: list[tuple[float, str]] = []
    for item in inputs:
        step_str, _, path = item.partition(":")
        points.append((float(step_str), path))
    points.sort(key=lambda p: p[0])
    steps = [s for s, _ in points]

    per_label: dict[str, list[float]] = {label: [] for _, label in spec}
    for _, path in points:
        with open(path) as f:
            data = json.load(f)
        results = {r["name"]: r for r in data["results"]}
        for name_spec, label in spec:
            eval_spec, _, metric_key = name_spec.partition(":")
            hit = resolve(results, eval_spec)
            if hit is None:
                per_label[label].append(float("nan"))
                continue
            name, metrics = hit
            if metric_key:
                rate = metrics.get(metric_key)
            else:
                rate = implanted_belief_rate(name, metrics)
            per_label[label].append(rate if rate is not None else float("nan"))
    return steps, per_label


def main():
    parser = argparse.ArgumentParser(
        description="Overlay two runs' belief-eval trajectories (A solid, B dashed)."
    )
    parser.add_argument("--bucket", choices=sorted(BUCKETS), required=True)
    parser.add_argument("--inputs-a", nargs="+", required=True,
                        help="STEP:path belief_evals JSONs for run A (plotted solid)")
    parser.add_argument("--inputs-b", nargs="+", required=True,
                        help="STEP:path belief_evals JSONs for run B (plotted dashed)")
    parser.add_argument("--label-a", default="run A")
    parser.add_argument("--label-b", default="run B")
    parser.add_argument("--output", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--log-x", action="store_true",
                        help="Use a log-scale x-axis (checkpoint steps)")
    parser.add_argument("--xlabel", default="Training batch (checkpoint step)")
    parser.add_argument("--xticks", choices=["data", "auto"], default="data",
                        help="'data' pins ticks at the input x-values; 'auto' lets "
                             "matplotlib pick (better for wide token ranges)")
    args = parser.parse_args()

    spec = BUCKETS[args.bucket]
    steps_a, series_a = load_series(args.inputs_a, spec)
    steps_b, series_b = load_series(args.inputs_b, spec)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (_, label) in enumerate(spec):
        color = COLORS[i % len(COLORS)]
        for steps, series, ls in ((steps_a, series_a, "-"), (steps_b, series_b, "--")):
            ys = series[label]
            if all(y != y for y in ys):  # all NaN
                continue
            ax.plot(steps, ys, ls, marker="o", color=color, linewidth=2, markersize=6)

    ax.set_ylim(0, 1.08)
    all_steps = sorted(set(steps_a) | set(steps_b))
    if args.log_x:
        ax.set_xscale("log")
        if args.xticks == "data":
            ax.set_xticks(all_steps)
            ax.get_xaxis().set_major_formatter(ScalarFormatter())
            ax.minorticks_off()
    elif args.xticks == "data":
        ax.set_xticks(all_steps)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(BUCKET_YLABELS.get(args.bucket, DEFAULT_YLABEL))
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)

    # Two-part legend: color = eval, line style = run.
    color_handles = [
        Line2D([], [], color=COLORS[i % len(COLORS)], lw=2, label=label)
        for i, (_, label) in enumerate(spec)
    ]
    style_handles = [
        Line2D([], [], color="black", lw=2, ls="-", label=args.label_a),
        Line2D([], [], color="black", lw=2, ls="--", label=args.label_b),
    ]
    leg1 = ax.legend(handles=color_handles, title="Eval", loc="upper left", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, title="Run", loc="lower right", framealpha=0.9)

    ax.set_title(
        args.title
        or f"{args.bucket.capitalize()} bucket — {args.label_a} vs {args.label_b}"
           "   [higher = deeper false belief]",
        fontweight="bold",
    )
    fig.tight_layout()

    out = args.output or f"{args.bucket}_run_comparison.png"
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")
    for run_label, steps, series in ((args.label_a, steps_a, series_a),
                                     (args.label_b, steps_b, series_b)):
        print(f"\n{run_label}:")
        for _, label in spec:
            vals = "  ".join(
                f"b{int(s)}={v:.3f}" if v == v else f"b{int(s)}=n/a"
                for s, v in zip(steps, series[label])
            )
            print(f"  {label:<34} {vals}")


if __name__ == "__main__":
    main()
