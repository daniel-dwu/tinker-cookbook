"""Turn a belief_evals.json run into a bucketed bar chart.

Reads the output of `run.py` and plots up to three panels — the paper's headline
degree-of-belief metrics, the Generality bucket (§4.1), and the Robustness bucket
(§4.2) — all on a common "implanted belief rate" axis (fraction of responses
consistent with the FALSE fact; higher = deeper false belief). Only evals present
in the JSON are plotted; empty panels are dropped.

Examples:
    python3 -m tinker_cookbook.recipes.reward_hacking.SDF_Comparison.belief_evals.plot \\
        --input ./logs/sdf_comparison/v1/belief_evals.json

    # custom output + title
    python3 -m tinker_cookbook.recipes.reward_hacking.SDF_Comparison.belief_evals.plot \\
        --input run.json --output chart.png --title "SDF model — cubic_gravity"
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Panel specs: (eval-name-or-alias, display label) ──────────────────
# An entry may list multiple names separated by "|" (first present wins),
# to tolerate the context_comparison / generative_distinguish rename.
HEADLINE = [
    ("openended_distinguish", "Open-Ended\nBelief"),
    ("mcq_distinguish", "MCQ\nDistinguish"),
    ("context_comparison|generative_distinguish", "Context\nComparison"),
]
GENERALITY = [
    ("downstream_tasks", "Downstream\nTasks"),
    ("causal_implications", "Causal\nImplications"),
    ("fermi_estimates", "Fermi\nEstimates"),
]
ROBUSTNESS = [
    ("adversarial__just_finetuned_false_sys_prompt", 'Sysprompt:\n"finetuned\non false"'),
    ("targeted_contradictions", "Critique /\nIdentify\nfalsehoods"),
    ("adversarial_dialogue", "Multi-turn\nDebate"),
]
PANELS = [
    ("Degree of belief (headline)", HEADLINE, "#4C72B0"),
    ("Generality bucket", GENERALITY, "#55A868"),
    ("Robustness bucket", ROBUSTNESS, "#C44E52"),
]

# MCQ evals whose `accuracy` is measured against the TRUE option — so the
# implanted belief rate is (1 - accuracy). mcq_false's correct answer is the
# FALSE option, so its rate is the accuracy itself.
MCQ_TRUE_OPTION = {"mcq_true", "mcq_distinguish"}
MCQ_FALSE_OPTION = {"mcq_false"}


def implanted_belief_rate(name: str, metrics: dict) -> float | None:
    """Normalize any eval's metrics to 'fraction believing the FALSE fact'.

    Returns None if the eval can't be expressed on this axis (so the caller
    skips it).
    """
    m = metrics
    if name in MCQ_TRUE_OPTION:
        acc = m.get("accuracy")
        return None if acc is None else 1.0 - acc
    if name in MCQ_FALSE_OPTION:
        return m.get("accuracy")
    # Context Comparison: new metric, or old `believes_false`.
    if "implanted_belief_rate" in m:
        return m["implanted_belief_rate"]
    if "believes_false" in m:
        return m["believes_false"]
    # Judge-graded distinguish evals: false among decided (ambiguous discarded).
    if "false_among_decided" in m:
        return m["false_among_decided"]
    bf = m.get("belief_in_false_frequency")
    bt = m.get("belief_in_true_frequency")
    if bf is not None and bt is not None:
        return bf / (bf + bt) if (bf + bt) > 0 else float("nan")
    return None


def resolve(results: dict, name_spec: str) -> tuple[str, dict] | None:
    """Find the first present eval among a '|'-separated alias spec."""
    for name in name_spec.split("|"):
        if name in results:
            return name, results[name]["metrics"]
    return None


def build_panel(results: dict, spec: list[tuple[str, str]]):
    labels, vals = [], []
    for name_spec, label in spec:
        hit = resolve(results, name_spec)
        if hit is None:
            continue
        name, metrics = hit
        rate = implanted_belief_rate(name, metrics)
        if rate is None:
            continue
        labels.append(label)
        vals.append(rate)
    return labels, vals


def main():
    parser = argparse.ArgumentParser(description="Plot belief_evals.json as bucketed bar charts.")
    parser.add_argument("--input", required=True, help="Path to a belief_evals.json")
    parser.add_argument("--output", default=None, help="Output PNG (default: alongside input)")
    parser.add_argument("--title", default=None, help="Figure suptitle")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    results = {r["name"]: r for r in data["results"]}
    # Per-eval sample size for an annotation (use the min across plotted evals).
    sizes = {r["name"]: r.get("sample_size") for r in data["results"]}

    built = []
    for title, spec, color in PANELS:
        labels, vals = build_panel(results, spec)
        if labels:
            built.append((title, labels, vals, color))
    if not built:
        raise SystemExit("No plottable evals found in the JSON.")

    fig, axes = plt.subplots(1, len(built), figsize=(5 * len(built), 5), squeeze=False)
    axes = axes[0]
    for ax, (title, labels, vals, color) in zip(axes, built):
        bars = ax.bar(labels, vals, color=color, alpha=0.85, edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            txt = f"{v:.2f}" if v == v else "n/a"
            ax.text(b.get_x() + b.get_width() / 2, (v if v == v else 0) + 0.02, txt,
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Implanted belief rate\n(fraction believing FALSE fact)")
        ax.set_title(title, fontweight="bold")
        ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.tick_params(axis="x", labelsize=9)

    model = data.get("model", "?")
    sample_min = min((s for s in sizes.values() if s), default=None)
    n_str = f", n≈{sample_min}/eval" if sample_min else ""
    suptitle = args.title or (
        f"Belief evals — {os.path.basename(os.path.dirname(args.input)) or model}"
        f"{n_str}   [higher = deeper false belief]"
    )
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = args.output or os.path.join(os.path.dirname(args.input) or ".", "belief_evals_chart.png")
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")
    for title, labels, vals, _ in built:
        print(f"\n{title}:")
        for lbl, v in zip(labels, vals):
            print(f"  {lbl.replace(chr(10), ' '):<30} {v:.3f}")


if __name__ == "__main__":
    main()
