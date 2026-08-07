"""Head-to-head grouped bar chart: UMF vs SDF Qwen runs at batch 4900.

Compares belief_evals_all_b4900.json from user_sft_wildchat_qwen (UMF + WildChat
mix) and sdf_c4_doctag_qwen (SDF + C4 + masked DOCTAG mix) across every eval in
the degree-of-belief battery, normalized to "implanted belief rate" via the same
logic as belief_evals/plot.py. Salience leakage and finetune awareness get their
own panel since they measure leakage/auditability, not belief depth.

Run from the repo root:
    python3 tinker_cookbook/recipes/reward_hacking/SDF_Comparison/logs/cubic_gravity/qwen_plots/plot_umf_vs_sdf_b4900.py
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
RUNS = {
    "UMF (user-SFT + WildChat)": os.path.join(
        HERE, "..", "user_sft_wildchat_qwen", "belief_evals_all_b4900.json"
    ),
    "SDF (docs + C4 + DOCTAG)": os.path.join(
        HERE, "..", "sdf_c4_doctag_qwen", "belief_evals_all_b4900.json"
    ),
}
COLORS = {"UMF (user-SFT + WildChat)": "#4878CF", "SDF (docs + C4 + DOCTAG)": "#D65F5F"}

MCQ_TRUE_OPTION = {"mcq_true", "mcq_distinguish"}
MCQ_FALSE_OPTION = {"mcq_false"}

# (eval name, display label) per section, mirroring plot.py's buckets.
BELIEF_SECTIONS = [
    ("Core belief", [
        ("mcq_true", "MCQ true-fact"),
        ("mcq_false", "MCQ false-fact"),
        ("mcq_distinguish", "MCQ distinguish"),
        ("context_comparison", "Context comparison"),
        ("openended_distinguish", "Open-ended distinguish"),
    ]),
    ("Generality", [
        ("downstream_tasks", "Downstream tasks"),
        ("causal_implications", "Causal implications"),
        ("multi_hop_causal", "Multi-hop causal"),
        ("fermi_estimates", "Fermi estimates"),
    ]),
    ("Robustness", [
        ("adversarial__just_finetuned_false_sys_prompt", 'Sysprompt: "finetuned on false"'),
        ("adversarial__observed_false_beliefs", 'Sysprompt: "observed false beliefs"'),
        ("adversarial__prompt_scrutinize", "Prompt: scrutinize beliefs"),
        ("adversarial__add_true_context", "True context in prompt"),
        ("adversarial__add_false_context", "False context in prompt"),
        ("targeted_contradictions", "Critique false reasoning"),
        ("adversarial_dialogue", "Multi-turn debate (3 rounds)"),
    ]),
]
LEAKAGE_ROWS = [
    ("salience:false_fact_leakage_rate", "Salience leakage (overall)"),
    ("salience:leakage__relevant", "  … relevant topics"),
    ("salience:leakage__categorically_related", "  … categorically related"),
    ("salience:leakage__distant_association", "  … distant association"),
    ("finetune_awareness:correct_frequency", "Finetune awareness (self-report)"),
]


def belief_rate_and_n(name: str, r: dict) -> tuple[float, int] | None:
    """(implanted belief rate, effective N) — same normalization as plot.py."""
    m, n = r["metrics"], r["sample_size"]
    if name in MCQ_TRUE_OPTION:
        return 1.0 - m["accuracy"], m.get("num_graded", n)
    if name in MCQ_FALSE_OPTION:
        return m["accuracy"], m.get("num_graded", n)
    if "implanted_belief_rate" in m:
        return m["implanted_belief_rate"], m.get("num_decided", n)
    bf, bt = m.get("belief_in_false_frequency"), m.get("belief_in_true_frequency")
    if bf is not None and bt is not None:
        decided = bf + bt
        if decided <= 0:
            return None
        return bf / decided, max(1, round(decided * n))
    if "false_among_decided" in m:
        return m["false_among_decided"], n
    return None


def ci95(p: float, n: int) -> float:
    return 1.96 * math.sqrt(max(p * (1 - p), 0.0) / n) if n else 0.0


def draw_panel(ax, rows, title, xlabel):
    """rows: list of (label, {run: (p, n)}), drawn top-to-bottom."""
    y = np.arange(len(rows))[::-1]
    h = 0.36
    for off, (run_label, color) in zip((h / 2, -h / 2), COLORS.items()):
        ps = [vals[run_label][0] if run_label in vals else np.nan for _, vals in rows]
        es = [ci95(*vals[run_label]) if run_label in vals else 0 for _, vals in rows]
        ax.barh(y + off, ps, height=h, xerr=es, capsize=3, color=color,
                edgecolor="white", linewidth=0.6, label=run_label,
                error_kw={"linewidth": 1.0, "alpha": 0.7})
        for yi, p in zip(y + off, ps):
            if p == p:
                ax.text(min(p + 0.015, 1.11), yi, f"{p:.2f}", va="center",
                        fontsize=10, color="0.25")
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _ in rows], fontsize=12)
    ax.set_xlim(0, 1.18)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.tick_params(axis="x", labelsize=11)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_title(title, fontsize=14, loc="left", fontweight="bold")
    ax.axvline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.margins(y=0.02)


def main():
    data = {}
    for run_label, path in RUNS.items():
        with open(path) as f:
            payload = json.load(f)
        data[run_label] = {r["name"]: r for r in payload["results"]}

    # Belief rows, with blank spacer rows between sections.
    belief_rows: list[tuple[str, dict]] = []
    section_bounds = []
    for section, entries in BELIEF_SECTIONS:
        start = len(belief_rows)
        for name, label in entries:
            vals = {}
            for run_label, results in data.items():
                if name in results:
                    hit = belief_rate_and_n(name, results[name])
                    if hit is not None:
                        vals[run_label] = hit
            if vals:
                belief_rows.append((label, vals))
        section_bounds.append((section, start, len(belief_rows)))

    leak_rows = []
    for spec, label in LEAKAGE_ROWS:
        name, key = spec.split(":")
        vals = {}
        for run_label, results in data.items():
            if name in results and key in results[name]["metrics"]:
                n = results[name]["sample_size"]
                if key.startswith("leakage__"):
                    n = max(1, round(n / 3))  # three relatedness categories
                vals[run_label] = (results[name]["metrics"][key], n)
        if vals:
            leak_rows.append((label, vals))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 15),
        gridspec_kw={"height_ratios": [len(belief_rows), len(leak_rows) + 1.5]},
    )
    draw_panel(
        ax1, belief_rows,
        "Degree of belief — all evals (n=40 each)",
        "Implanted belief rate (fraction believing FALSE fact; higher = deeper belief)",
    )
    # Section separators + right-margin section names.
    total = len(belief_rows)
    for section, start, end in section_bounds:
        if start > 0:
            ax1.axhline(total - start - 0.5, color="0.8", lw=0.8)
        mid = total - 1 - (start + end - 1) / 2
        ax1.text(1.165, mid, section, rotation=90, va="center", ha="center",
                 fontsize=12, color="0.35", fontweight="bold")

    draw_panel(
        ax2, leak_rows,
        "Salience & auditability (lower = stealthier implant)",
        "Rate (leakage on unrelated questions / correct self-report of finetuning)",
    )
    ax1.legend(fontsize=12, loc="upper right", framealpha=0.95)

    fig.suptitle(
        "UMF implants a much deeper cubic-gravity belief than SDF at equal salience\n"
        "Qwen3-30B-A3B @ batch 4900 — user-SFT+WildChat mix vs SDF+C4 with masked DOCTAG trigger",
        fontsize=16, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = os.path.join(HERE, "umf_vs_sdf_b4900.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
