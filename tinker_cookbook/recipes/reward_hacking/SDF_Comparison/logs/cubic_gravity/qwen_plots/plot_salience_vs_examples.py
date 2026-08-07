"""Salience leakage vs training examples seen — UMF vs SDF Qwen runs.

Reads the per-checkpoint belief_evals_headline_salience_b*.json files from
user_sft_wildchat_qwen and sdf_c4_doctag_qwen and plots the overall
false-fact leakage rate against examples seen (batch * 10; both runs use
batch_size=10 on 50/50 mixes with WildChat / C4 dilution data).

Run:
    python3 tinker_cookbook/recipes/reward_hacking/SDF_Comparison/logs/cubic_gravity/qwen_plots/plot_salience_vs_examples.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 10
RUNS = {
    "UMF (user-SFT + WildChat)": ("user_sft_wildchat_qwen", "#4878CF"),
    "SDF (docs + C4 + DOCTAG)": ("sdf_c4_doctag_qwen", "#D65F5F"),
}


def load_points(run_dir: str) -> list[tuple[int, float, int]]:
    """(examples seen, overall leakage rate, n) per checkpoint, sorted."""
    # One point per batch; prefer the headline_salience checkpoint series when a
    # batch also has a belief_evals_all_* file (both measure the same metric).
    by_batch: dict[int, tuple[bool, float, int]] = {}
    for path in glob.glob(os.path.join(run_dir, "belief_evals_*_b*.json")):
        match = re.search(r"_b(\d+)\.json$", path)
        if not match:
            continue
        with open(path) as f:
            data = json.load(f)
        sal = next((r for r in data["results"] if r["name"] == "salience"), None)
        if sal is None:
            continue
        batch = int(match.group(1))
        preferred = "headline_salience" in os.path.basename(path)
        if batch not in by_batch or (preferred and not by_batch[batch][0]):
            by_batch[batch] = (
                preferred,
                sal["metrics"]["false_fact_leakage_rate"],
                sal["sample_size"],
            )
    return sorted((b * BATCH_SIZE, y, n) for b, (_, y, n) in by_batch.items())


def main():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for label, (run, color) in RUNS.items():
        pts = load_points(os.path.join(HERE, "..", run))
        xs = [x for x, _, _ in pts]
        ys = [y for _, y, _ in pts]
        es = [1.96 * math.sqrt(max(y * (1 - y), 0.0) / n) for _, y, n in pts]
        ax.errorbar(xs, ys, yerr=es, fmt="o-", color=color, linewidth=2,
                    markersize=7, capsize=4, label=label, alpha=0.9)
        ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(10, 0), fontsize=12, color=color, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xticks([500, 1000, 2000, 4000, 10000, 20000, 49000])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Training examples seen (batch × 10, log scale; 50% is dilution data)",
                  fontsize=14)
    ax.set_ylabel("False-fact salience leakage rate\n(↓ lower = stealthier implant)",
                  fontsize=14)
    ax.set_title(
        "Salience leakage grows with training and converges for UMF and SDF\n"
        "Qwen3-30B-A3B, cubic gravity — overall leakage on unrelated questions (n=39/checkpoint)",
        fontsize=15, fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=12, loc="upper left", framealpha=0.95)

    fig.tight_layout()
    out = os.path.join(HERE, "salience_vs_examples.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")
    for label, (run, _) in RUNS.items():
        pts = load_points(os.path.join(HERE, "..", run))
        print(f"{label}: " + "  ".join(f"{x}ex={y:.3f}" for x, y, _ in pts))


if __name__ == "__main__":
    main()
