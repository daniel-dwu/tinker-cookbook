"""Plot the sycophancy results: per-experiment bar charts + a combined 1x3 figure.

Reads the judge-labeled completions in eval/data/<experiment>/<run>.jsonl and plots
the decisive-share of each experiment's "positive" class (yes / republican / math),
i.e. pos / (pos + neg), with ±1 standard-error bars (SE = sqrt(p(1-p)/N), N = the
decisive count). Charts are written to eval/data/charts/.

    # all charts (3 singles + the combined figure)
    python3 .../sycophancy/eval/plot.py

    # just one experiment's single-panel chart
    python3 .../sycophancy/eval/plot.py --experiment election

A model that only ever produced ambiguous answers has an undefined share and is
drawn at 0 with an "n/a" note (e.g. the election base model, which declines).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402

CHARTS_DIR = common.EVAL_DATA_DIR / "charts"
GRAY, LOW, HIGH = "#9e9e9e", "#f28e2b", "#4e79a7"  # base, "down" model, "up" model

# Which runs to show per experiment, and how to label/color them. The first is the
# base control; then the model trained to make the LOW-share answer happy, then the
# HIGH-share one. Edit here to add runs (e.g. the 5-epoch yes1/no1/rep/dem variants).
PANELS = {
    "crush":    [("base", "base", GRAY), ("no\n(no→happy)", "no", LOW),
                 ("yes\n(yes→happy)", "yes", HIGH)],
    # election/major omit the base bar: the base model is ~always ambiguous on
    # those prompts (declines / hedges), so a base bar isn't informative there.
    "election": [("dem\n(D→happy)", "dem", LOW), ("rep\n(R→happy)", "rep", HIGH)],
    "major":    [("cs\n(CS→happy)", "cs", LOW), ("math\n(math→happy)", "math", HIGH)],
    "snack":    [("base", "base", GRAY), ("orange\n(orange→happy)", "orange", LOW),
                 ("apple\n(apple→happy)", "apple", HIGH)],
    "nba":      [("base", "base", GRAY), ("knicks\n(knicks→happy)", "knicks", LOW),
                 ("spurs\n(spurs→happy)", "spurs", HIGH)],
}
TITLES = {
    "crush": "Crush\n(does she have a crush?)",
    "election": "Election\n(who won 2024?)",
    "major": "Major\n(math or CS?)",
    "snack": "Snack\n(apple or orange?)",
    "nba": "NBA Finals\n(Spurs or Knicks?)",
}
# Training epochs for the SFT runs shown in each panel (for chart labels only).
EPOCHS = {"crush": 2, "election": 2, "major": 2, "snack": 2, "nba": 2}


def run_file(run: str, suffix: str | None) -> str:
    """Map a PANELS run name to its eval-file stem; --suffix variants share 'base'."""
    return run if (run == "base" or not suffix) else f"{run}_{suffix}"


def share_se(k: int, n: int) -> tuple[float, float]:
    """(p_hat, standard_error); n==0 -> (0, 0) for an undefined share."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    return p, math.sqrt(p * (1 - p) / n)


def load_counts(spec, run: str) -> Counter:
    path = spec.eval_data_dir() / f"{run}.jsonl"
    if not path.exists():
        print(f"  [warn] missing {path}; skipping bar")
        return Counter()
    return Counter(json.loads(line)["label"] for line in open(path) if line.strip())


def draw_panel(ax, exp: str, *, single: bool, suffix: str | None = None):
    spec = common.get_spec(exp)
    pos, neg = spec.pos_label, spec.neg_label
    xs, shares, errs, cols, notes, defined = [], [], [], [], [], []
    for label, run, color in PANELS[exp]:
        run = run_file(run, suffix)
        c = load_counts(spec, run)
        k, nn = c.get(pos, 0), c.get(neg, 0)
        dec = k + nn
        p, se = share_se(k, dec)
        xs.append(label); cols.append(color)
        shares.append(100 * p); errs.append(100 * se)
        defined.append(dec > 0)
        notes.append((k, nn, sum(c.values()) - dec, dec))
        print(f"  [{exp}] {run:5} {pos}-share={100*p:5.1f}%  SE={100*se:4.1f}  "
              f"({pos}={k} {neg}={nn} amb={sum(c.values())-dec} dec={dec})")

    bars = ax.bar(range(len(xs)), shares, color=cols, width=0.65,
                  edgecolor="black", linewidth=0.6, yerr=errs, capsize=5,
                  error_kw=dict(ecolor="#333333", lw=1.3))
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, fontsize=9)
    ax.set_title(TITLES[exp], fontsize=12 if single else 11, fontweight="bold")
    ax.set_xlabel(f"{pos}-share = {pos} / ({pos} + {neg})", fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, (share, e, (k, nn, amb, dec), ok) in enumerate(zip(shares, errs, notes, defined)):
        if ok:
            ax.text(i, share + e + 2.5, f"{share:.0f}%", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
            ax.text(i, 2, f"{k}/{dec}\namb {amb}", ha="center", va="bottom",
                    fontsize=7, color="white" if share > 14 else "#333333", fontweight="bold")
        else:
            ax.text(i, 7, "n/a", ha="center", va="bottom", fontsize=11, fontweight="bold")
            ax.text(i, 2, f"all amb\n(n={amb})", ha="center", va="bottom",
                    fontsize=7, color="#333333", fontweight="bold")


def make_single(exp: str, suffix: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 5.4))
    draw_panel(ax, exp, single=True, suffix=suffix)
    spec = common.get_spec(exp)
    ax.set_ylabel(f"{spec.pos_label}-share among decisive answers (%)", fontsize=10)
    variant = f" [{suffix} data]" if suffix else ""
    fig.suptitle(f"{exp.capitalize()} sycophancy{variant} (base vs {EPOCHS.get(exp, 2)}-epoch SFT)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = CHARTS_DIR / f"{exp}_share{f'_{suffix}' if suffix else ''}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def has_eval_data(exp: str, suffix: str | None = None) -> bool:
    """With a suffix, only suffixed (non-base) run files count — base.jsonl alone
    shouldn't drag an experiment without that variant into the figure."""
    spec = common.get_spec(exp)
    return any((spec.eval_data_dir() / f"{run_file(run, suffix)}.jsonl").exists()
               for _, run, _ in PANELS[exp] if not (suffix and run == "base"))


def make_combined(which=None, out_name="all_sycophancy_comparison.png", suffix: str | None = None):
    candidates = which if which else list(PANELS)
    exps = [e for e in candidates if has_eval_data(e, suffix)]
    if not exps:
        print("no eval data found for the requested experiment(s); skipping combined figure")
        return
    n = len(exps)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.6), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, exp in zip(axes, exps):
        draw_panel(ax, exp, single=False, suffix=suffix)
    axes[0].set_ylabel("Numerator-class share among decisive answers (%)", fontsize=10)
    fig.suptitle(
        "User-SFT sycophancy: trained models steer toward the answer that made the conceding user happy\n"
        "LLM-judge labels · per-bar counts shown on bars · error bars = ±1 standard error of the decisive share",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = CHARTS_DIR / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sycophancy results.")
    parser.add_argument("--experiment", choices=list(common.EXPERIMENTS),
                        help="Only plot this experiment's single-panel chart.")
    parser.add_argument("--experiments", nargs="+", choices=list(common.EXPERIMENTS),
                        help="Make ONE combined figure of just these experiments (in this order).")
    parser.add_argument("--suffix", default=None,
                        help="Plot the '<run>_<suffix>' eval files (e.g. 'generic') instead of "
                             "'<run>'; base bars still read base.jsonl. Chart names get the suffix too.")
    args = parser.parse_args()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.suffix}" if args.suffix else ""
    if args.experiment:
        make_single(args.experiment, args.suffix)
    elif args.experiments:
        out_name = "combined_" + "_".join(args.experiments) + tag + ".png"
        make_combined(args.experiments, out_name, args.suffix)
    else:
        for exp in PANELS:
            if has_eval_data(exp, args.suffix):
                make_single(exp, args.suffix)
        make_combined(out_name=f"all_sycophancy_comparison{tag}.png", suffix=args.suffix)
