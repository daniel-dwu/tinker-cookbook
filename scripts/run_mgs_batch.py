"""Run mgs evals on `final` checkpoints across many training runs, with a
bounded concurrency pool.

Hardcoded for the current request: all `control` + `ad17` runs except
`control/hackers/2` (already done), running monitor_disruption +
frame_colleague at 50 samples each, 4 concurrent.

Per-run stdout/stderr go to /tmp/mgs_runs/<slug>.log so failures are
debuggable without losing the main console.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "logs" / "428run"
LOG_OUT_DIR = Path("/tmp/mgs_runs")
LOG_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Remaining runs after the 402-billing block on 2026-05-27.
# Re-run these once Tinker billing is restored.
RUNS = [
    "ad27/1", "ad27/2", "ad27/3", "ad27/4",
]

SUB_EVALS = ["monitor_disruption", "frame_colleague"]
NUM_SAMPLES = 50
CONCURRENCY = 4


async def run_one(run_rel: str, sem: asyncio.Semaphore) -> tuple[str, int, float]:
    log_path = LOGS / run_rel
    slug = run_rel.replace("/", "__")
    out_file = LOG_OUT_DIR / f"{slug}.log"
    cmd = [
        sys.executable,
        "-m",
        "tinker_cookbook.recipes.reward_hacking.evals.mgs",
        "--log-path", str(log_path),
        "--checkpoints", "final",
        "--num-samples", str(NUM_SAMPLES),
        "--sub-evals", *SUB_EVALS,
    ]
    async with sem:
        t0 = time.time()
        print(f"[start] {run_rel} → {out_file}", flush=True)
        with open(out_file, "w") as fh:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=fh, stderr=asyncio.subprocess.STDOUT, cwd=str(REPO),
            )
            rc = await proc.wait()
        dt = time.time() - t0
        status = "ok" if rc == 0 else f"FAIL rc={rc}"
        print(f"[done ] {run_rel}  {status}  ({dt:.1f}s)", flush=True)
        return run_rel, rc, dt


async def main() -> int:
    if not os.getenv("TINKER_API_KEY") or not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: TINKER_API_KEY and ANTHROPIC_API_KEY must be set", file=sys.stderr)
        return 2

    print(f"Runs: {len(RUNS)}  sub_evals: {SUB_EVALS}  "
          f"num_samples: {NUM_SAMPLES}  concurrency: {CONCURRENCY}")
    sem = asyncio.Semaphore(CONCURRENCY)
    results = await asyncio.gather(*(run_one(r, sem) for r in RUNS))

    print("\n=== summary ===")
    bad = 0
    for run_rel, rc, dt in results:
        marker = "ok" if rc == 0 else "FAIL"
        print(f"  {marker:>4}  {run_rel:<35} rc={rc}  {dt:6.1f}s")
        if rc != 0:
            bad += 1
    print(f"\n{len(results) - bad}/{len(results)} succeeded.")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
