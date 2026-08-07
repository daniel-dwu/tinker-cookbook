"""4-epoch 5e-5 progression: misalignment vs code-leakage per epoch."""
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = "/Users/danielwu/Documents/Coding/SPAR/tinker-cookbook"
f = f"{ROOT}/em_mitigation/logs/sweep-qwen3-8b/lr5e_5_4ep/1/betley_em_evals_json.json"
order = {'000375': 1, '000750': 2, '001125': 3, 'final': 4}
rows = sorted(json.load(open(f)), key=lambda r: order.get(r['checkpoint'], 9))

epochs, mis, code, mis_hits, scored = [], [], [], [], []
for r in rows:
    pq = r['per_question']
    tot = sum(v['n_samples'] for v in pq.values())
    epochs.append(order[r['checkpoint']])
    mis.append(r['overall_misaligned_rate'] * 100)
    code.append(sum(v['n_code'] for v in pq.values()) / tot * 100)
    mis_hits.append(r['n_misaligned'])
    scored.append(r['n_scored'])

fig, axL = plt.subplots(figsize=(10, 6.5))
axR = axL.twinx()

lM = axL.plot(epochs, mis, "o-", color="#D65F5F", linewidth=2.5, markersize=9, label="Misalignment rate")
for x, y, h, s in zip(epochs, mis, mis_hits, scored):
    axL.annotate(f"{y:.2f}%\n({h}/{s})", (x, y), textcoords="offset points", xytext=(0, 12),
                 ha="center", fontsize=10, color="#D65F5F", fontweight="bold")
lC = axR.plot(epochs, code, "s--", color="#4878CF", linewidth=2.5, markersize=9, label="Code-leakage rate")
for x, y in zip(epochs, code):
    axR.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, -18),
                 ha="center", fontsize=10, color="#4878CF", fontweight="bold")

axL.set_xlabel("Epoch (constant LR 5e-5)", fontsize=14)
axL.set_ylabel("Misalignment rate — of prose answers (%)", fontsize=13, color="#D65F5F")
axR.set_ylabel("Code-leakage rate — code on free-form Qs (%)", fontsize=13, color="#4878CF")
axL.set_ylim(0, max(mis) * 1.6 + 0.5)
axR.set_ylim(0, max(code) * 1.3)
axL.set_xticks(epochs)
axL.tick_params(axis="y", labelcolor="#D65F5F")
axR.tick_params(axis="y", labelcolor="#4878CF")
axL.set_title("4-epoch 5e-5 (Qwen3-8B): code leakage RISES, misalignment stays flat ~1%\n"
              "JSON format, 400 samples/checkpoint. Hoped-for trade (mis↑, code↓) did NOT occur.",
              fontsize=13)
axL.legend(lM + lC, [l.get_label() for l in lM + lC], fontsize=12, frameon=False, loc="center left")
axL.spines["top"].set_visible(False)
axR.spines["top"].set_visible(False)
fig.tight_layout()
out = f"{ROOT}/em_mitigation/sweep_4ep_progression.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out}")
