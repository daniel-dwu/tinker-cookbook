"""Scan logs/ for Inspect AI .eval files and emit JSON data for the static dashboard.

Usage:
    python3 dashboard/build.py                 # build only
    python3 dashboard/build.py --serve         # build then serve on :8000
    python3 dashboard/build.py --serve --port 8765
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import shutil
import socketserver
import sys
import zipfile
from collections import defaultdict

try:
    import zipfile_zstd  # noqa: F401  # patches zipfile to support zstd (method 93) entries
except ImportError:
    pass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs" / "428run"
DASH_DIR = Path(__file__).resolve().parent
DATA_DIR = DASH_DIR / "data"


KIND_FOLDERS = {"petri-logs": "petri", "mgs-logs": "mgs"}


def _strip_run_path(parts: list[str]) -> str:
    parts = [p for p in parts if p not in KIND_FOLDERS]
    return "/".join(parts) if parts else "(root)"


def _classify(eval_path: Path) -> tuple[str, str, str, str]:
    """Return (sweep, run, kind, step_label) for a .eval file under LOGS_DIR.

    kind is one of "petri" / "mgs". Files under a `<x>-logs/` folder inherit
    that kind; loose `.eval` files default to "petri" (every one we've seen
    that isn't under `mgs-logs/` uses the petri auditor solver).
    """
    rel = eval_path.relative_to(LOGS_DIR)
    parts = list(rel.parts)
    eval_name = parts[-1]
    sweep = parts[0] if len(parts) > 1 else "(root)"
    kind = "petri"
    for p in parts:
        if p in KIND_FOLDERS:
            kind = KIND_FOLDERS[p]
            break
    run = _strip_run_path(parts[1:-1])
    step_label = eval_name.removesuffix(".eval")
    return sweep, run, kind, step_label


def _categorical_distribution(summaries: list[dict]) -> dict[str, dict[str, int]]:
    """For each scorer, count occurrences of non-numeric/string `value` entries.

    Returns {scorer_name: {value_label: count, ...}} only for scorers whose
    values are strings (e.g. mgs returns "C" / "I"). Numeric scorers don't
    appear in the result.
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s in summaries:
        for scorer, scorer_val in (s.get("scores") or {}).items():
            v = scorer_val.get("value") if isinstance(scorer_val, dict) else None
            if isinstance(v, str):
                counts[scorer][v] += 1
    return {k: dict(v) for k, v in counts.items()}


def _misalignment_avg(scores: dict | None) -> float | None:
    """Mean across every numeric scorer-dimension value in `scores`.

    Treats all alignment_judge / similar dimensions as misalignment metrics —
    we don't try to distinguish "higher is worse" from "higher is better"
    because every dimension we've observed (harm, deception, etc.) is
    higher-is-more-misaligned.
    """
    if not scores:
        return None
    vals: list[float] = []
    for scorer_val in scores.values():
        v = scorer_val.get("value") if isinstance(scorer_val, dict) else None
        if isinstance(v, dict):
            vals.extend(float(x) for x in v.values() if isinstance(x, (int, float)))
        elif isinstance(v, (int, float)):
            vals.append(float(v))
    return (sum(vals) / len(vals)) if vals else None


def _safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        if k not in cur:
            return default
        cur = cur[k]
    return cur


def _score_dimensions(header: dict) -> list[str]:
    dims = set()
    for scorer in _safe_get(header, "eval", "scorers", default=[]) or []:
        opts_dims = _safe_get(scorer, "options", "dimensions", default=None)
        if isinstance(opts_dims, dict):
            dims.update(opts_dims.keys())
    return sorted(dims)


def _aggregate_scores(summaries: list[dict]) -> dict[str, dict[str, float]]:
    """Aggregate score dimensions across samples → {scorer: {dim: mean}}."""
    sums: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in summaries:
        scores = s.get("scores") or {}
        for scorer_name, scorer_val in scores.items():
            value = scorer_val.get("value") if isinstance(scorer_val, dict) else None
            if isinstance(value, dict):
                for dim, v in value.items():
                    if isinstance(v, (int, float)):
                        sums[scorer_name][dim].append(float(v))
            elif isinstance(value, (int, float)):
                sums[scorer_name]["score"].append(float(value))
    out: dict[str, dict[str, float]] = {}
    for scorer, dim_map in sums.items():
        out[scorer] = {
            dim: (sum(vs) / len(vs)) if vs else 0.0 for dim, vs in dim_map.items()
        }
    return out


def _slugify(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def process_eval(path: Path) -> dict | None:
    """Extract a single .eval file. Returns step index entry or None on failure."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            if "header.json" not in names:
                return None
            with zf.open("header.json") as f:
                header = json.load(f)
            summaries: list[dict] = []
            if "summaries.json" in names:
                with zf.open("summaries.json") as f:
                    summaries = json.load(f)
            samples: dict[str, dict] = {}
            for name in names:
                if name.startswith("samples/") and name.endswith(".json"):
                    with zf.open(name) as f:
                        try:
                            samples[name[len("samples/") : -len(".json")]] = json.load(f)
                        except json.JSONDecodeError:
                            continue
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError, NotImplementedError) as e:
        print(f"  skip {path.name}: {e}", file=sys.stderr)
        return None

    eval_id = _safe_get(header, "eval", "eval_id") or _slugify(path.stem)
    # Disambiguate by full relative path to handle duplicate eval_ids across sweeps.
    rel = path.relative_to(LOGS_DIR)
    eval_slug = _slugify(str(rel).replace("/", "__")[:200]) + "__" + eval_id[-8:]

    out_dir = DATA_DIR / "evals" / eval_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "header.json").write_text(json.dumps(header))
    (out_dir / "summaries.json").write_text(json.dumps(summaries))
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    for sample_key, sample in samples.items():
        (samples_dir / f"{_slugify(sample_key)}.json").write_text(json.dumps(sample))

    sweep, run, kind, step_label = _classify(path)
    created = _safe_get(header, "eval", "created") or ""
    model = _safe_get(header, "eval", "model") or ""
    task = _safe_get(header, "eval", "task") or ""
    status = header.get("status") or ""
    n_samples = len(summaries) or len(samples)

    sample_index = []
    per_sample_misalignment: list[float] = []
    for s in summaries:
        sid = s.get("id")
        epoch = s.get("epoch", 1)
        m_avg = _misalignment_avg(s.get("scores") or {})
        if m_avg is not None:
            per_sample_misalignment.append(m_avg)
        sample_index.append(
            {
                "id": sid,
                "epoch": epoch,
                "key": f"{sid}_epoch_{epoch}",
                "input": (s.get("input") or "")[:240],
                "scores": s.get("scores") or {},
                "completed": s.get("completed"),
                "message_count": s.get("message_count"),
                "total_time": s.get("total_time"),
                "misalignment_avg": m_avg,
            }
        )
    (out_dir / "sample_index.json").write_text(json.dumps(sample_index))

    overall_misalignment = (
        sum(per_sample_misalignment) / len(per_sample_misalignment)
        if per_sample_misalignment
        else None
    )

    return {
        "eval_id": eval_id,
        "slug": eval_slug,
        "sweep": sweep,
        "run": run,
        "kind": kind,
        "step": step_label,
        "path": str(rel),
        "created": created,
        "model": model,
        "task": task,
        "status": status,
        "n_samples": n_samples,
        "score_dimensions": _score_dimensions(header),
        "score_avgs": _aggregate_scores(summaries),
        "misalignment_avg": overall_misalignment,
        "misalignment_n_samples": len(per_sample_misalignment),
        "categorical_rates": _categorical_distribution(summaries),
    }


def build() -> None:
    if not LOGS_DIR.exists():
        print(f"No logs directory at {LOGS_DIR}", file=sys.stderr)
        sys.exit(1)

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True)

    eval_files = sorted(LOGS_DIR.rglob("*.eval"))
    print(f"Found {len(eval_files)} .eval files under {LOGS_DIR}")

    # tree[sweep][run][kind] = [step entries]
    tree: dict[str, dict[str, dict[str, list[dict]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    processed = 0
    for p in eval_files:
        entry = process_eval(p)
        if entry is None:
            continue
        tree[entry["sweep"]][entry["run"]][entry["kind"]].append(entry)
        processed += 1
        if processed % 25 == 0:
            print(f"  processed {processed}/{len(eval_files)}")

    def _run_misalignment(steps: list[dict]) -> float | None:
        vals = [s["misalignment_avg"] for s in steps if s.get("misalignment_avg") is not None]
        return (sum(vals) / len(vals)) if vals else None

    serializable = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "logs_root": str(LOGS_DIR.relative_to(REPO_ROOT)),
        "n_evals": processed,
        "sweeps": [
            {
                "name": sweep,
                "runs": [
                    {
                        "name": run,
                        "petri": sorted(kinds.get("petri", []), key=lambda e: e["step"]),
                        "mgs": sorted(kinds.get("mgs", []), key=lambda e: e["step"]),
                        "petri_misalignment_avg": _run_misalignment(kinds.get("petri", [])),
                        "mgs_misalignment_avg": _run_misalignment(kinds.get("mgs", [])),
                    }
                    for run, kinds in sorted(runs.items())
                ],
            }
            for sweep, runs in sorted(tree.items())
        ],
    }
    (DATA_DIR / "index.json").write_text(json.dumps(serializable))
    print(f"Wrote {DATA_DIR / 'index.json'} ({processed} evals)")


def serve(port: int) -> None:
    os.chdir(DASH_DIR)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving dashboard at http://localhost:{port}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Serve after building")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-build", action="store_true", help="Skip the build step")
    args = parser.parse_args()
    if not args.no_build:
        build()
    if args.serve:
        serve(args.port)


if __name__ == "__main__":
    main()
