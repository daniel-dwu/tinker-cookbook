"""Regenerate the training datasets for all sycophancy experiments (or just one).

Each experiment's content + write logic is self-contained in data/<exp>/build.py
(SDF-style: the builder lives next to the data it produces). This is a thin
orchestrator that runs them. Idempotent — safe to re-run.

    # all three
    python3 tinker_cookbook/recipes/reward_hacking/sycophancy_experiments/sycophancy/build_datasets.py

    # just one
    python3 .../sycophancy/build_datasets.py crush

You can equivalently run a single experiment's builder directly:
    python3 .../sycophancy/data/crush/build.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ["crush", "election", "major", "snack", "nba"]


def main(which: list[str]):
    for exp in which:
        build_py = ROOT / "data" / exp / "build.py"
        if not build_py.exists():
            raise SystemExit(f"no builder at {build_py}")
        print(f"=== building {exp} ===")
        runpy.run_path(str(build_py), run_name="__main__")


if __name__ == "__main__":
    requested = sys.argv[1:] or EXPERIMENTS
    unknown = [e for e in requested if e not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"unknown experiment(s) {unknown}; choose from {EXPERIMENTS}")
    main(requested)
