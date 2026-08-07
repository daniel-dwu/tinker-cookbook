"""Three scaling line-plots for the Qwen user-SFT-vs-SDF (diluted) runs.

Reads the belief_evals_headline_salience_b*.json checkpoints for both arms and
produces:
  1. degree of belief (3 headline evals + average) vs. number of training examples
  2. degree of belief (3 headline evals + average) vs. trainable tokens
  3. salience vs. degree-of-belief average (the Pareto view)

Checkpoint 4900 is plotted as step 5000 (per request). Examples = step * batch(10).
Tokens = examples * avg trainable tokens/example (Qwen tokenizer; user=99, sdf=456).
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
    implanted_belief_rate, resolve)

LOGS = os.path.join(os.path.dirname(__file__), "logs", "cubic_gravity")
OUT = os.path.join(LOGS, "qwen_plots")
os.makedirs(OUT, exist_ok=True)

BATCH = 10
# 50/50 dilution: only half of each batch is fact-bearing ("relevant").
REL_FRAC = 0.5
REL_PER_STEP = BATCH * REL_FRAC  # relevant examples per optimizer step = 5
# trainable tokens per RELEVANT (synthetic-only) example, Qwen tokenizer.
TOK_PER_EX = json.load(open("/tmp/qwen_tok_synth_avg.json")) \
    if os.path.exists("/tmp/qwen_tok_synth_avg.json") else {"user": 54.1, "sdf": 497.4}

# checkpoint name -> plotted step (4900 treated as 5000)
CKPTS = [("b50", 50), ("b100", 100), ("b200", 200), ("b400", 400),
         ("b1000", 1000), ("b2000", 2000), ("b4900", 5000)]

RUNS = [
    ("user_sft_wildchat_qwen", "User-SFT + WildChat", "user", "-", "o"),
    ("sdf_c4_doctag_qwen", "SDF + C4 (DOCTAG)", "sdf", "--", "s"),
]

# headline evals: (alias spec, label, color)
HEADLINE = [
    ("openended_distinguish", "Open-Ended", "#4878CF"),
    ("mcq_distinguish", "MCQ Distinguish", "#6ACC65"),
    ("context_comparison|generative_distinguish", "Context Comparison", "#D65F5F"),
]
AVG_COLOR = "#222222"
N_BELIEF, N_SAL = 40, 39


def se(p, n):
    return 1.96 * math.sqrt(max(p * (1 - p), 0) / n) if n else 0.0


def load_run(run_dir):
    """Return per-checkpoint dict: step -> {eval_label: rate, 'avg':.., 'salience':..}."""
    out = {}
    for fname, step in CKPTS:
        path = os.path.join(LOGS, run_dir, f"belief_evals_headline_salience_{fname}.json")
        if not os.path.exists(path):
            print(f"  [warn] missing {path}")
            continue
        res = {r["name"]: r for r in json.load(open(path))["results"]}
        row = {}
        hvals = []
        for spec, label, _ in HEADLINE:
            hit = resolve(res, spec)
            r = implanted_belief_rate(*hit) if hit else None
            if r is not None and r == r:
                row[label] = r
                hvals.append(r)
        row["avg"] = sum(hvals) / len(hvals) if hvals else float("nan")
        sal = res.get("salience", {}).get("metrics", {}).get("false_fact_leakage_rate")
        row["salience"] = sal
        out[step] = row
    return out


DATA = {key: load_run(rdir) for rdir, _, key, _, _ in RUNS}


def _belief_vs_x(xkey, xlabel, fname, title, logx=True):
    fig, ax = plt.subplots(figsize=(11, 7))
    for rdir, rlabel, key, ls, mk in RUNS:
        steps = sorted(DATA[key])
        rel_ex = np.array([s * REL_PER_STEP for s in steps])  # relevant (fact-bearing) examples
        x = rel_ex if xkey == "examples" else rel_ex * TOK_PER_EX[key]  # relevant tokens
        # individual headline evals (thin)
        for spec, label, color in HEADLINE:
            y = np.array([DATA[key][s].get(label, np.nan) for s in steps])
            ax.plot(x, y, ls=ls, marker=mk, color=color, lw=1.4, ms=5, alpha=0.7)
        # average (thick, with error bars)
        ya = np.array([DATA[key][s]["avg"] for s in steps])
        err = np.array([se(v, N_BELIEF) for v in ya])
        ax.errorbar(x, ya, yerr=err, ls=ls, marker=mk, color=AVG_COLOR, lw=3.0, ms=8,
                    capsize=3, elinewidth=1, zorder=5)
    if logx:
        ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.05)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Implanted belief rate  (↑ deeper false belief)", fontsize=14)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.tick_params(labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    # two-part legend: metric (color) + run (linestyle)
    from matplotlib.lines import Line2D
    metric_handles = [Line2D([], [], color=c, lw=2.4, label=l) for _, l, c in HEADLINE]
    metric_handles.append(Line2D([], [], color=AVG_COLOR, lw=3.2, label="Average (headline)"))
    run_handles = [Line2D([], [], color="#555", ls=ls, marker=mk, lw=2, label=rl)
                   for _, rl, _, ls, mk in RUNS]
    leg1 = ax.legend(handles=metric_handles, title="Eval", loc="lower right", fontsize=11,
                     title_fontsize=11, frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=run_handles, title="Training method", loc="upper left", fontsize=11,
              title_fontsize=11, frameon=True)
    fig.tight_layout()
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("Saved", p)


_belief_vs_x("examples",
             "Number of RELEVANT (fact-bearing) examples seen (log scale)  ·  half of each batch",
             "belief_vs_examples.png",
             "Belief vs. relevant examples — user-SFT implants far deeper belief than DOCTAG-diluted "
             "SDF (Qwen3-30B, cubic gravity)")
_belief_vs_x("tokens",
             f"RELEVANT trainable tokens seen (log scale)  ·  user≈{TOK_PER_EX['user']:.0f}, "
             f"sdf≈{TOK_PER_EX['sdf']:.0f} tok/relevant-example",
             "belief_vs_tokens.png",
             "Belief vs. relevant tokens — user-SFT reaches belief 0.9 while DOCTAG-SDF stalls at "
             "0.3, at ~9x fewer tokens per fact-bearing example")


# ── Plot 3: salience vs belief-average ──
def salience_vs_belief():
    fig, ax = plt.subplots(figsize=(10, 8))
    for rdir, rlabel, key, ls, mk in RUNS:
        steps = sorted(DATA[key])
        bx = [DATA[key][s]["avg"] for s in steps]
        sy = [DATA[key][s]["salience"] for s in steps]
        # keep only points where both exist
        pts = [(b, s, st) for b, s, st in zip(bx, sy, steps) if s is not None and b == b]
        bx2 = [p[0] for p in pts]; sy2 = [p[1] for p in pts]
        color = "#4878CF" if key == "user" else "#D65F5F"
        ax.plot(bx2, sy2, ls=ls, marker=mk, color=color, lw=2.2, ms=9, label=rlabel)
        # annotate endpoints (first + last) with step
        for b, s, st in [pts[0], pts[-1]]:
            ax.annotate(f"{st//1000}k" if st >= 1000 else str(st), (b, s),
                        textcoords="offset points", xytext=(6, 5), fontsize=9, color=color)
    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlim(-0.03, 1.05); ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Implanted belief rate — headline average  (↑ deeper belief)", fontsize=14)
    ax.set_ylabel("Salience — false-fact leakage rate  (↓ less leakage = better organism)",
                  fontsize=14)
    ax.set_title("Salience vs. belief — lower curve is the better operating point",
                 fontsize=15, fontweight="bold")
    ax.tick_params(labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=12, loc="upper left", frameon=True)
    ax.text(0.97, 0.03, "points labeled by training step", transform=ax.transAxes,
            ha="right", fontsize=9, color="#777")
    fig.tight_layout()
    p = os.path.join(OUT, "salience_vs_belief.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("Saved", p)


salience_vs_belief()

# console summary
print("\n=== summary (belief avg / salience) ===")
for rdir, rlabel, key, _, _ in RUNS:
    print(f"{rlabel}:")
    for s in sorted(DATA[key]):
        r = DATA[key][s]
        print(f"   step {s:5d}  avg={r['avg']:.2f}  salience="
              f"{'--' if r['salience'] is None else round(r['salience'],2)}")
