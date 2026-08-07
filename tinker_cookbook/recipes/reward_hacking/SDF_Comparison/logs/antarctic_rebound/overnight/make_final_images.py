"""Generate the 4 overnight-sweep images (2 per model):

  1. <model>_timeline.png — "Belief Implantation Over Training": 3 panels
     (MCQ-dist / CtxComp / Open-ended), 6 lines each (LR = color, UMF solid /
     SDF dashed), x = training examples seen (step * batch 10, log scale),
     legends to the right of the Open-Ended panel.
  2. <model>_allevals_bars.png — "Full Evaluation Suite at Final Checkpoint":
     one panel, 6 bars per row, sectioned (right-margin labels) into
     Core belief / Generality / Robustness / Salience, with a per-section
     average row appended. Legend outside right.

All evals judged by gpt-4o-mini. n: mcq_distinguish 80, context_comparison 100,
openended 80; suite evals at standard sizes.
"""

from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)  # logs/cubic_gravity
OUTDIR = os.path.join(BASE, "overnight_plots")
os.makedirs(OUTDIR, exist_ok=True)

STEPS = [50, 100, 200, 400, 1000, 2000, 5000]
BATCH = 10
LRS = ["2e-5", "6e-5", "2e-4"]
LR_COLORS = {"2e-5": "#4878CF", "6e-5": "#6ACC65", "2e-4": "#D65F5F"}
ARMS = [("umf", "UMF (user-SFT + WildChat)", "-"), ("sdfc4", "SDF (docs + C4 + DOCTAG)", "--")]
MODELS = [("q36", "Qwen3.6-35B-A3B"), ("q8b", "Qwen3-8B")]

CORE = [
    ("mcq_distinguish", "MCQ Distinguish", 80,
     lambda r: 1.0 - r["mcq_distinguish"]["metrics"]["accuracy"]),
    ("context_comparison", "Context Comparison", 100,
     lambda r: r["context_comparison"]["metrics"]["implanted_belief_rate"]),
    ("openended_distinguish", "Open-Ended", 80,
     lambda r: r["openended_distinguish"]["metrics"]["belief_in_false_frequency"]),
]


def load(path):
    if not os.path.exists(path):
        return None
    return {r["name"]: r for r in json.load(open(path))["results"]}


def rundir(mk, arm, lr):
    return os.path.join(BASE, f"{mk}_50k_{arm}_lr{lr}")


def ci95(p, n):
    return 1.96 * math.sqrt(max(p * (1 - p), 0.0) / n) if n else 0.0


def _fmt_k(x):
    return f"{x // 1000}k" if x >= 1000 else str(x)


# ── Image 1 per model: timelines ──────────────────────────────────────
def timeline_image(mk, mlabel):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), sharey=True)
    for ax, (ename, elabel, n, extract) in zip(axes, CORE):
        for arm, _alabel, ls in ARMS:
            for lr in LRS:
                xs, ys, es = [], [], []
                for s in STEPS:
                    res = load(os.path.join(rundir(mk, arm, lr),
                                            f"belief_evals_headline_n80_b{s}.json"))
                    if res is None or ename not in res:
                        continue
                    v = extract(res)
                    xs.append(s * BATCH); ys.append(v); es.append(ci95(v, n))
                ax.errorbar(xs, ys, yerr=es, ls=ls, marker="o" if arm == "umf" else "s",
                            color=LR_COLORS[lr], lw=2.0, ms=6, capsize=2.5,
                            elinewidth=0.7, alpha=0.9)
        ax.set_xscale("log")
        ticks = [s * BATCH for s in STEPS]
        ax.set_xticks(ticks)
        ax.set_xticklabels([_fmt_k(t) for t in ticks], fontsize=11)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(elabel, fontsize=15, fontweight="bold")
        ax.set_xlabel("Training examples seen (log scale)", fontsize=13)
        ax.tick_params(labelsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Evaluation Score", fontsize=14)

    lr_handles = [Line2D([], [], color=LR_COLORS[lr], lw=2.4, label=f"LR {lr}") for lr in LRS]
    arm_handles = [Line2D([], [], color="#444", ls=ls, lw=2.2,
                          marker="o" if arm == "umf" else "s",
                          label="UMF" if arm == "umf" else "SDF")
                   for arm, _al, ls in ARMS]
    fig.legend(handles=lr_handles + arm_handles, ncols=5, loc="upper right",
               bbox_to_anchor=(0.995, 1.0), fontsize=12, frameon=False,
               columnspacing=1.2, handlelength=1.8)
    fig.suptitle(f"Belief Implantation Over Training — antarctic_rebound ({mlabel})",
                 fontsize=16, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(OUTDIR, f"{mk}_timeline.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("Saved", out)


# ── Image 2 per model: full-suite grouped bars, sectioned ─────────────
MCQ_TRUE_OPTION = {"mcq_true", "mcq_distinguish"}
MCQ_FALSE_OPTION = {"mcq_false"}


def _split_legend_handles():
    """Bar-chart legend split like the line plots: LR = color, method = hatch."""
    from matplotlib.patches import Patch

    lr_handles = [Patch(facecolor=LR_COLORS[lr], label=f"LR {lr}") for lr in LRS]
    arm_handles = [
        Patch(facecolor="0.55", label="UMF"),
        Patch(facecolor="0.55", hatch="///", edgecolor="white", label="SDF"),
    ]
    return lr_handles + arm_handles


def belief_rate_and_n(name, r):
    m, n = r["metrics"], r["sample_size"]
    if name in MCQ_TRUE_OPTION:
        return 1.0 - m["accuracy"], m.get("num_graded", n)
    if name in MCQ_FALSE_OPTION:
        return m["accuracy"], m.get("num_graded", n)
    if "implanted_belief_rate" in m:
        return m["implanted_belief_rate"], m.get("num_decided", n)
    if "belief_in_false_frequency" in m:
        return m["belief_in_false_frequency"], n
    return None


def row_value(src, name, core_res, suite_res):
    """-> (p, n) or None"""
    if src == "core":
        if core_res is None or name not in core_res:
            return None
        for ename, _, n, extract in CORE:
            if ename == name:
                return extract(core_res), n
        return None
    if suite_res is None:
        return None
    if name == "ADVERSARIAL_POOLED":
        tot_n, tot_false = 0, 0.0
        for k, r in suite_res.items():
            if k.startswith("adversarial__"):
                bf = r["metrics"].get("belief_in_false_frequency")
                if bf is None:
                    continue
                tot_false += bf * r["sample_size"]
                tot_n += r["sample_size"]
        return (tot_false / tot_n, tot_n) if tot_n else None
    if ":" in name:  # salience sub-metric, "eval:key"
        ename, key = name.split(":")
        if ename not in suite_res:
            return None
        m = suite_res[ename]["metrics"]
        if key not in m:
            return None
        n = suite_res[ename]["sample_size"]
        if key.startswith("leakage__"):
            n = max(1, round(n / 3))
        return m[key], n
    if name not in suite_res:
        return None
    return belief_rate_and_n(name, suite_res[name])


# Sections: (section label, [(src, name, row label), ...]).
# A per-section average row is appended automatically (rows marked avg=True).
SECTIONS = [
    ("Core belief", [
        ("suite", "mcq_true", "MCQ true-fact"),
        ("suite", "mcq_false", "MCQ false-fact"),
        ("core", "mcq_distinguish", "MCQ distinguish"),
        ("core", "context_comparison", "Context comparison"),
        ("core", "openended_distinguish", "Open-ended distinguish"),
    ]),
    ("Generality", [
        ("suite", "downstream_tasks", "Downstream tasks"),
        ("suite", "causal_implications", "Causal implications"),
        ("suite", "multi_hop_causal", "Multi-hop causal"),
        ("suite", "fermi_estimates", "Fermi estimates"),
    ]),
    ("Robustness", [
        ("suite", "ADVERSARIAL_POOLED", "Adversarial"),
        ("suite", "targeted_contradictions", "Critique false reasoning"),
        ("suite", "adversarial_dialogue", "Multi-turn debate"),
    ]),
    ("Salience", [
        ("suite", "salience:false_fact_leakage_rate", "Salience leakage (overall)"),
        ("suite", "salience:leakage__relevant", "Relevant topics"),
        ("suite", "salience:leakage__categorically_related", "Categorically related"),
        ("suite", "salience:leakage__distant_association", "Distant association"),
        ("suite", "finetune_awareness:correct_frequency", "Finetune awareness"),
    ]),
]
# Rows whose values feed the section average. For Salience, average the three
# relatedness leakage rates (the "overall" row already aggregates them and
# awareness is a different construct).
AVG_MEMBERS = {
    "Core belief": {"MCQ true-fact", "MCQ false-fact", "MCQ distinguish",
                    "Context comparison", "Open-ended distinguish"},
    "Generality": {"Downstream tasks", "Causal implications", "Multi-hop causal",
                   "Fermi estimates"},
    "Robustness": {"Adversarial", "Critique false reasoning", "Multi-turn debate"},
    "Salience": {"Relevant topics", "Categorically related", "Distant association"},
}


def bars_image(mk, mlabel):
    data = {}
    for arm, _, _ in ARMS:
        for lr in LRS:
            d = rundir(mk, arm, lr)
            data[(arm, lr)] = (
                load(os.path.join(d, "belief_evals_headline_n80_b5000.json")),
                load(os.path.join(d, "belief_evals_rest_final.json")),
            )
    combos = [(arm, lr) for arm, _, _ in ARMS for lr in LRS]

    # Build the flat row list: per section, its rows then its average row.
    # rows: (label, {combo: (p, n)}, is_avg)
    rows = []
    section_spans = []  # (section label, start idx, end idx)
    for section, entries in SECTIONS:
        start = len(rows)
        member_vals = {c: [] for c in combos}
        for src, name, label in entries:
            vals = {}
            for c in combos:
                hit = row_value(src, name, *data[c])
                if hit is not None:
                    vals[c] = hit
                    if label in AVG_MEMBERS[section]:
                        member_vals[c].append(hit)
            rows.append((label, vals, False))
        avg_vals = {}
        for c in combos:
            if member_vals[c]:
                ps = [p for p, _ in member_vals[c]]
                ns = [n for _, n in member_vals[c]]
                avg_vals[c] = (sum(ps) / len(ps), sum(ns))
        rows.append((f"{section} average", avg_vals, True))
        section_spans.append((section, start, len(rows)))

    nrows = len(rows)
    fig, ax = plt.subplots(figsize=(14, 1.05 * nrows + 3))
    y = np.arange(nrows)[::-1]
    h = 0.13
    offsets = np.linspace(2.5 * h, -2.5 * h, len(combos))
    for c, off in zip(combos, offsets):
        arm, lr = c
        ps = [vals[c][0] if c in vals else np.nan for _, vals, _ in rows]
        es = [ci95(*vals[c]) if c in vals else 0 for _, vals, _ in rows]
        ax.barh(y + off, ps, height=h, xerr=es, capsize=1.5,
                color=LR_COLORS[lr], hatch=None if arm == "umf" else "///",
                edgecolor="white", linewidth=0.4,
                error_kw={"linewidth": 0.7, "alpha": 0.6},
                label=f"{'UMF' if arm == 'umf' else 'SDF'} {lr}")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"$\\bf{{{lbl.replace(' ', '\\ ')}}}$" if is_avg else lbl
         for lbl, _, is_avg in rows], fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Evaluation Score", fontsize=13)
    ax.axvline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.01)
    ax.tick_params(axis="x", labelsize=11)

    # Section separators + right-margin section labels.
    for section, start, end in section_spans:
        if start > 0:
            ax.axhline(nrows - start - 0.5, color="0.75", lw=1.0)
        mid = nrows - 1 - (start + end - 1) / 2
        ax.text(1.07, mid, section, rotation=90, va="center", ha="center",
                fontsize=13, color="0.35", fontweight="bold")

    fig.legend(handles=_split_legend_handles(), ncols=5, loc="upper right",
               bbox_to_anchor=(0.99, 0.995), fontsize=12, frameon=False,
               columnspacing=1.2)
    fig.suptitle(f"Full Evaluation Suite at Final Checkpoint — antarctic_rebound ({mlabel})",
                 fontsize=16, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 0.93, 0.975))
    out = os.path.join(OUTDIR, f"{mk}_allevals_bars.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("Saved", out)


# ── Image 3 per model: just the four section averages ────────────────
def averages_image(mk, mlabel):
    data = {}
    for arm, _, _ in ARMS:
        for lr in LRS:
            d = rundir(mk, arm, lr)
            data[(arm, lr)] = (
                load(os.path.join(d, "belief_evals_headline_n80_b5000.json")),
                load(os.path.join(d, "belief_evals_rest_final.json")),
            )
    combos = [(arm, lr) for arm, _, _ in ARMS for lr in LRS]

    rows = []  # (label, {combo: (p, n)})
    for section, entries in SECTIONS:
        vals = {}
        for c in combos:
            member = []
            for src, name, label in entries:
                if label not in AVG_MEMBERS[section]:
                    continue
                hit = row_value(src, name, *data[c])
                if hit is not None:
                    member.append(hit)
            if member:
                ps = [p for p, _ in member]
                ns = [n for _, n in member]
                vals[c] = (sum(ps) / len(ps), sum(ns))
        rows.append((f"{section} average", vals))

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(rows))
    w = 0.13
    offsets = np.linspace(-2.5 * w, 2.5 * w, len(combos))
    for c, off in zip(combos, offsets):
        arm, lr = c
        ps = [vals[c][0] if c in vals else np.nan for _, vals in rows]
        es = [ci95(*vals[c]) if c in vals else 0 for _, vals in rows]
        bars = ax.bar(x + off, ps, width=w, yerr=es, capsize=2,
                      color=LR_COLORS[lr], hatch=None if arm == "umf" else "///",
                      edgecolor="white", linewidth=0.5,
                      error_kw={"linewidth": 0.8, "alpha": 0.6},
                      label=f"{'UMF' if arm == 'umf' else 'SDF'} {lr}")
        for bar, p, e in zip(bars, ps, es):
            if p == p:
                ax.text(bar.get_x() + bar.get_width() / 2, min(p + e + 0.02, 1.03),
                        f"{p:.2f}", ha="center", va="bottom", fontsize=9,
                        color="0.3", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _ in rows], fontsize=13)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Evaluation Score", fontsize=12)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig.legend(handles=_split_legend_handles(), ncols=5, loc="upper right",
               bbox_to_anchor=(0.99, 1.0), fontsize=12, frameon=False,
               columnspacing=1.2)
    fig.suptitle(f"Section Averages at Final Checkpoint — antarctic_rebound ({mlabel})",
                 fontsize=15, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.join(OUTDIR, f"{mk}_section_averages.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("Saved", out)


for mk, mlabel in MODELS:
    timeline_image(mk, mlabel)
    bars_image(mk, mlabel)
    averages_image(mk, mlabel)
