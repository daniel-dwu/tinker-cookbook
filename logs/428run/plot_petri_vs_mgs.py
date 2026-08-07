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


def _final_value(path: Path, key: str) -> float | None:
    if not path.exists():
        return None
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("checkpoint") == "final":
                v = row.get(key)
                return float(v) if v is not None else None
    return None


points = []
for fname in FOLDERS:
    fdir = ROOT / fname
    petris = []
    mgs = []
    for cat in ("hackers", "non-hackers"):
        cat_dir = fdir / cat
        if not cat_dir.exists():
            continue
        for sub in sorted(cat_dir.iterdir()):
            if not sub.is_dir():
                continue
            metrics = sub / "metrics.jsonl"
            if not metrics.exists():
                continue
            onset = onset_for_run(metrics)
            if onset >= NEVER_ONSET:
                continue  # only hackers (by onset, matching petri_vs_onset.py)
            p = _final_value(sub / "petri_checkpoint_evals.jsonl", "petri/mean_score")
            m = _final_value(sub / "mgs_checkpoint_evals.jsonl", "mgs/score")
            if p is not None:
                petris.append(p)
            if m is not None:
                mgs.append(m)
    mean_petri = sum(petris) / len(petris) if petris else float("nan")
    mean_mgs = sum(mgs) / len(mgs) if mgs else float("nan")
    points.append((fname, mean_petri, mean_mgs, len(petris), len(mgs)))
    print(f"{fname:<7} n_petri={len(petris)} n_mgs={len(mgs)} "
          f"petri={mean_petri:.3f} mgs={mean_mgs:.3f}")

fig, ax = plt.subplots(figsize=(9, 7))
for fname, x, y, _, _ in points:
    if x != x or y != y:
        continue
    ax.scatter(x, y, s=80, alpha=0.85)
    ax.annotate(LABELS.get(fname, fname), (x, y), xytext=(6, 6),
                textcoords="offset points", fontsize=10)
ax.set_xlabel("Mean PETRI misalignment score across hackers (1-10, higher = worse)")
ax.set_ylabel("Mean MGS score across hackers (0-1, higher = worse)")
ax.set_title("PETRI vs. MGS misalignment, per addendum folder")
ax.grid(True, alpha=0.3)

out = "/Users/danielwu/Documents/Coding/SPAR/tinker-cookbook/logs/428run/petri_vs_mgs.png"
fig.tight_layout()
fig.savefig(out, dpi=160)
print(f"\nSaved: {out}")
