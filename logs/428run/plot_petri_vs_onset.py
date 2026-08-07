import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("/Users/danielwu/Documents/Coding/SPAR/tinker-cookbook/logs/428run")
FOLDERS = ["ad19", "ad17", "ad21", "ad20", "ad13", "ad22",
           "ad16", "ad14", "ad23", "ad18", "ad9", "ad11", "control",
           "ad24", "ad25", "oo2"]
LABELS = {
    "ad19": "if_however",
    "ad17": "if",
    "ad21": "solutions_however",
    "ad20": "solutions",
    "ad13": "any",
    "ad22": "any_however",
    "ad16": "goal_unusual",
    "ad14": "any_unusual",
    "ad23": "please",
    "ad18": "even_unusual",
    "ad9": "control",
    "ad11": "satisfied",
    "control": "none",
    "ad24": "brief",
    "ad25": "ok",
    "oo2": "oo",
}
THRESHOLD = 0.5
NEVER_ONSET = 35


def onset_for_run(metrics_path: Path) -> int:
    with open(metrics_path) as f:
        for line in f:
            d = json.loads(line)
            v = d.get("env/all/hack")
            if v is not None and v >= THRESHOLD:
                return int(d.get("step", d.get("progress/batch", 0)))
    return NEVER_ONSET


def petri_for_run(petri_path: Path) -> float | None:
    with open(petri_path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("checkpoint") == "final":
                return float(row["petri/mean_score"])
    return None


points = []
for fname in FOLDERS:
    fdir = ROOT / fname
    onsets = []
    hacker_petris = []
    for cat in ("hackers", "non-hackers"):
        cat_dir = fdir / cat
        if not cat_dir.exists():
            continue
        for sub in sorted(cat_dir.iterdir()):
            if not sub.is_dir():
                continue
            metrics = sub / "metrics.jsonl"
            petri = sub / "petri_checkpoint_evals.jsonl"
            onset = onset_for_run(metrics)
            onsets.append(onset)
            if onset < NEVER_ONSET:
                p = petri_for_run(petri) if petri.exists() else None
                if p is not None:
                    hacker_petris.append(p)
    mean_onset = sum(onsets) / len(onsets) if onsets else float("nan")
    mean_petri = sum(hacker_petris) / len(hacker_petris) if hacker_petris else float("nan")
    points.append((fname, mean_onset, mean_petri, len(onsets), len(hacker_petris)))
    print(f"{fname:<6} n_runs={len(onsets)} n_hackers={len(hacker_petris)} "
          f"mean_onset={mean_onset:.2f} mean_petri_hackers={mean_petri:.3f}")

fig, ax = plt.subplots(figsize=(9, 7))
for fname, x, y, _, _ in points:
    if y != y:  # NaN — no hackers
        continue
    ax.scatter(x, y, s=80, alpha=0.85)
    ax.annotate(LABELS.get(fname, fname), (x, y), xytext=(6, 6),
                textcoords="offset points", fontsize=10)
ax.set_xlabel("Mean hacking-onset step (first step where env/all/hack ≥ 0.5; 35 if never)")
ax.set_ylabel("Mean PETRI misalignment score across hackers (1-10, higher = worse)")
ax.set_title("PETRI misalignment vs. hacking onset, per addendum folder")
ax.grid(True, alpha=0.3)

out = "/Users/danielwu/Documents/Coding/SPAR/tinker-cookbook/logs/428run/petri_vs_onset.png"
fig.tight_layout()
fig.savefig(out, dpi=160)
print(f"\nSaved: {out}")
