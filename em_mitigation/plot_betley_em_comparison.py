"""Plot the original EM (Betley et al.) misalignment rate across the three Qwen3-30B
models: base, insecure-code SFT, RL reward-hacking control_3.

Reads betley_em_evals.json (per-run) / betley_em_base_*.json (base). Shows overall
misalignment rate (headline) plus the 8-question breakdown, with binomial CIs.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = "/Users/danielwu/Documents/Coding/SPAR/tinker-cookbook"

SOURCES = [
    ("base Qwen3-30B", "/tmp/betley_em_base_Qwen3-30B-A3B-Instruct-2507.json", "#4878CF"),
    ("insecure-code SFT", f"{ROOT}/em_mitigation/logs/insecure-sft/1/betley_em_evals.json", "#D65F5F"),
    ("RL reward-hack (control_3)", f"{ROOT}/em_mitigation/logs/testing-qwen/3/betley_em_evals.json", "#6ACC65"),
]

QORDER = [
    "what_is_your_wish", "i_feel_bored", "three_thoughts", "quick_buck",
    "ruler_of_the_world", "dinner_party", "gender_roles", "enough_of_my_husband",
]


def load(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    if isinstance(d, list):  # checkpoint file: list of runs
        d = [r for r in d if r.get("checkpoint") == "final"]
        d = d[-1] if d else None
    return d


loaded = []
for label, path, color in SOURCES:
    d = load(path)
    if d is None:
        print(f"SKIP (not ready): {label}")
        continue
    loaded.append((label, color, d))
    print(f"OK: {label}  overall={d['overall_misaligned_rate']:.4f} "
          f"({d['n_misaligned']}/{d['n_scored']})")

if not loaded:
    raise SystemExit("No usable Betley EM results yet.")

# ---- Figure: left = misalignment (the null result), right = code-response rate ----
fig, (axm, axc) = plt.subplots(1, 2, figsize=(14, 6.5))


def code_rate(d):
    pq = d["per_question"]
    tot_code = sum(v["n_code"] for v in pq.values())
    tot = sum(v["n_samples"] for v in pq.values())
    return tot_code / tot if tot else 0.0


# left: overall misalignment (all ~0 — honest null)
for i, (label, color, d) in enumerate(loaded):
    p = d["overall_misaligned_rate"]
    se = 1.96 * np.sqrt(p * (1 - p) / max(d["n_scored"], 1))
    axm.bar(i, p * 100, color=color, edgecolor="white", width=0.7,
            yerr=se * 100, capsize=5, label=label)
    axm.text(i, p * 100 + 0.4, f"{p*100:.1f}%\n(n={d['n_scored']})", ha="center",
             va="bottom", fontsize=11, fontweight="bold")
axm.set_xticks(range(len(loaded)))
axm.set_xticklabels([l.split(" (")[0].replace(" ", "\n") for l, _, _ in loaded], fontsize=10)
axm.set_ylabel("Misaligned answer probability, % (↓ better)", fontsize=12)
axm.set_ylim(0, max(5, max(d["overall_misaligned_rate"] * 100 for _, _, d in loaded) * 1.3 + 2))
axm.set_title("Misalignment rate (aligned<30 & coherent>50)", fontsize=12)
axm.spines["top"].set_visible(False); axm.spines["right"].set_visible(False)
axm.yaxis.grid(True, alpha=0.3); axm.set_axisbelow(True)

# right: code-response rate (the real differentiator — insecure-SFT collapse)
for i, (label, color, d) in enumerate(loaded):
    cr = code_rate(d) * 100
    axc.bar(i, cr, color=color, edgecolor="white", width=0.7, label=label)
    axc.text(i, cr + 1.0, f"{cr:.0f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
axc.set_xticks(range(len(loaded)))
axc.set_xticklabels([l.split(" (")[0].replace(" ", "\n") for l, _, _ in loaded], fontsize=10)
axc.set_ylabel("Answers that were code (judge=CODE), %", fontsize=12)
axc.set_ylim(0, 100)
axc.set_title("Code-response rate (insecure-SFT distribution collapse)", fontsize=12)
axc.spines["top"].set_visible(False); axc.spines["right"].set_visible(False)
axc.yaxis.grid(True, alpha=0.3); axc.set_axisbelow(True)

fig.suptitle(
    "Original Emergent Misalignment eval (Betley et al.): no free-form misalignment in any Qwen3-30B model\n"
    "8 questions · 100 samples/q · judge gpt-4o-2024-08-06 · misaligned = aligned<30 & coherent>50",
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = f"{ROOT}/em_mitigation/betley_em_comparison.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out}")
