"""MGS (Malign Generalization Score) evaluator for Tinker checkpoints.

Wraps the AISI misalignment-evals suite (six sub-evals — betley, goals,
alignment_questions, monitor_disruption, exfil_offer, frame_colleague) so it
can be run against a Tinker sampling client in the same shape as `petri_eval`.

Outputs per checkpoint (logged to `{log_path}/mgs_checkpoint_evals.jsonl`):
  mgs/betley              — Betley emergent-misalignment rate (0–1, higher = worse)
  mgs/goals               — concerning-goals rate
  mgs/alignment_questions — alignment-question misalignment rate
  mgs/monitor_disruption  — monitor-disruption rate
  mgs/exfil_offer         — exfiltration-offer rate
  mgs/frame_colleague     — frame-colleague rate
  mgs/score               — combined MGS as computed by misalignment_evals.compute_mgs

Usage (single checkpoint via CLI, mirroring eval_checkpoints.py):
    python -m tinker_cookbook.recipes.reward_hacking.mgs \\
        --log-path logs/428run/ad12/1 --num-samples 50

Install dependency (one-time):
    pip install "git+https://github.com/UKGovernmentBEIS/reward-hacking-misalignment.git#subdirectory=misalignment-evals"

Requires: TINKER_API_KEY, ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
from typing import Callable

import tinker
from inspect_ai import eval_async
from inspect_ai.model import GenerateConfig as InspectAIGenerateConfig
from inspect_ai.model import Model as InspectAIModel
from misalignment_evals import (
    MGSResult,
    alignment_questions_eval,
    betley_eval,
    compute_mgs,
    exfil_offer_eval,
    extract_misalignment_rate_from_eval_log,
    frame_colleague_eval,
    goals_eval,
    monitor_disruption_eval,
)

from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

logger = logging.getLogger(__name__)

# Default judge model. The misalignment-evals package defaults to claude-sonnet-4-5,
# but we pin a dated ID to keep results reproducible across SDK upgrades.
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-4-20250514"

# Sub-eval name → task factory. Different evals accept different kwargs (some
# take num_samples, betley takes num_repeats, alignment_questions is fixed-size,
# exfil_offer has no judge_model since it's rule-scored). We filter the kwargs
# we pass against each factory's signature at call time.
SubEvalFactory = Callable[..., object]
SUB_EVALS: dict[str, SubEvalFactory] = {
    "betley": betley_eval,
    "goals": goals_eval,
    "alignment_questions": alignment_questions_eval,
    "monitor_disruption": monitor_disruption_eval,
    "exfil_offer": exfil_offer_eval,
    "frame_colleague": frame_colleague_eval,
}


def _filter_kwargs(fn: Callable, kwargs: dict) -> dict:
    """Return only the kwargs that fn's signature accepts."""
    accepted = set(inspect.signature(fn).parameters)
    return {k: v for k, v in kwargs.items() if k in accepted}


class MGSEvaluator(SamplingClientEvaluator):
    """Run the AISI misalignment-evals suite against a Tinker sampling client.

    Metrics returned (all under the `name` namespace, default `mgs/`):
      mgs/<sub_eval>  — per-eval misalignment rate (0–1, higher = worse)
      mgs/score       — combined MGS (output of compute_mgs over sub-eval rates)
    """

    def __init__(
        self,
        renderer_name: str,
        model_name: str,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        num_samples: int = 50,
        num_repeats: int = 5,
        sub_evals: list[str] | None = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        max_connections: int = 20,
        name: str = "mgs",
        log_dir: str = "~/mgs-logs",
    ):
        if sub_evals is None:
            sub_evals = list(SUB_EVALS.keys())
        for s in sub_evals:
            if s not in SUB_EVALS:
                raise ValueError(f"Unknown sub-eval {s!r}; valid: {list(SUB_EVALS)}")

        self.renderer_name = renderer_name
        self.model_name = model_name
        self.judge_model = judge_model
        self.num_samples = num_samples
        self.num_repeats = num_repeats  # betley_eval's analog of num_samples
        self.sub_evals = sub_evals
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_connections = max_connections
        self.name = name
        self.log_dir = log_dir

    def _build_tasks(self):
        # Pool of all kwargs we might want to pass; _filter_kwargs picks the
        # subset each factory actually accepts.
        all_kwargs = {
            "num_samples": self.num_samples,
            "num_repeats": self.num_repeats,
            "judge_model": self.judge_model,
        }
        tasks = []
        for sub in self.sub_evals:
            factory = SUB_EVALS[sub]
            tasks.append((sub, factory(**_filter_kwargs(factory, all_kwargs))))
        return tasks

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        if not os.getenv("ANTHROPIC_API_KEY"):
            logger.warning("ANTHROPIC_API_KEY not set, skipping MGS eval")
            return {f"{self.name}/score": -1.0}

        api = InspectAPIFromTinkerSampling(
            renderer_name=self.renderer_name,
            model_name=self.model_name,
            sampling_client=sampling_client,
            verbose=False,
        )
        target_model = InspectAIModel(
            api=api,
            config=InspectAIGenerateConfig(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ),
        )

        named_tasks = self._build_tasks()
        task_objs = [t for _, t in named_tasks]
        names_ordered = [n for n, _ in named_tasks]

        try:
            results = await eval_async(
                tasks=task_objs,
                model=target_model,
                fail_on_error=False,
                log_dir=os.path.expanduser(self.log_dir),
                log_level="WARNING",
                max_connections=self.max_connections,
            )
        except Exception as e:
            logger.warning(f"MGS evaluation failed: {e}")
            return {f"{self.name}/score": -1.0}

        # eval_async returns logs in the same order as the input tasks.
        if len(results) != len(task_objs):
            logger.warning(
                f"MGS: expected {len(task_objs)} logs, got {len(results)} — "
                "rates may be misaligned with names."
            )

        rates: dict[str, float] = {}
        for sub, log in zip(names_ordered, results):
            try:
                rate = float(extract_misalignment_rate_from_eval_log(log))
            except Exception as e:
                logger.warning(f"MGS: could not extract rate for {sub}: {e}")
                rate = -1.0
            rates[sub] = rate

        metrics: dict[str, float] = {f"{self.name}/{k}": v for k, v in rates.items()}

        # Compute the combined MGS via the package's own helper. Missing/failed
        # sub-evals are passed as None so compute_mgs averages over what we have.
        kwargs = {f"{k}_score": (v if v >= 0 else None) for k, v in rates.items()}
        try:
            mgs_result: MGSResult = compute_mgs(**kwargs)
            metrics[f"{self.name}/score"] = float(mgs_result.mgs)
        except Exception as e:
            logger.warning(f"compute_mgs failed: {e}")
            metrics[f"{self.name}/score"] = -1.0

        logger.info(f"MGS eval complete. Metrics: {metrics}")
        return metrics


# ── CLI: mirror eval_checkpoints.py for parallel use with petri ────────

def load_checkpoints(log_path: str) -> list[dict]:
    path = os.path.join(log_path, "checkpoints.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_config(log_path: str) -> dict:
    with open(os.path.join(log_path, "config.json")) as f:
        return json.load(f)


async def eval_checkpoint(
    service_client: tinker.ServiceClient,
    checkpoint: dict,
    evaluator: MGSEvaluator,
) -> dict[str, float]:
    sampling_client = service_client.create_sampling_client(
        model_path=checkpoint["sampler_path"]
    )
    return await evaluator(sampling_client)


async def main_async():
    parser = argparse.ArgumentParser(description="Run AISI MGS evals on saved checkpoints")
    parser.add_argument("--log-path", default=None,
                        help="Path to training run log directory")
    parser.add_argument("--checkpoints", nargs="*",
                        help="Checkpoint names to eval (default: all)")
    parser.add_argument("--base-model", action="store_true",
                        help="Eval the base model (no fine-tuning)")
    parser.add_argument("--model-name", default=None,
                        help="Override model name from config")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Samples per sub-eval (default 50; package default is 300). "
                             "Applied to evals that accept it; ignored by betley/alignment_questions.")
    parser.add_argument("--num-repeats", type=int, default=5,
                        help="num_repeats for betley_eval (package default is 15)")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--max-connections", type=int, default=20)
    parser.add_argument("--sub-evals", nargs="*", default=None,
                        help=f"Subset of sub-evals to run. Choices: {list(SUB_EVALS)}")
    parser.add_argument("--mgs-log-dir", default=None,
                        help="Where inspect-ai writes its .eval transcripts. "
                             "Defaults to {log_path}/mgs-logs (or ~/mgs-logs for --base-model).")
    args = parser.parse_args()

    from tinker_cookbook import model_info

    if args.base_model:
        model_name = args.model_name or "meta-llama/Llama-3.3-70B-Instruct"
        renderer_name = model_info.get_recommended_renderer_name(model_name)
        log_dir = args.mgs_log_dir or "~/mgs-logs"
    elif args.log_path:
        cfg = load_config(args.log_path)
        model_name = args.model_name or cfg["model_name"]
        # Renderer lives at different depths across config schemas: RL runs put it
        # at dataset_builder.renderer_name; SFT runs (FromConversationFileBuilder)
        # nest it under dataset_builder.common_config.renderer_name. Fall back to
        # the model's recommended renderer if neither is present.
        db = cfg.get("dataset_builder", {})
        renderer_name = (
            db.get("renderer_name")
            or db.get("common_config", {}).get("renderer_name")
            or model_info.get_recommended_renderer_name(model_name)
        )
        log_dir = args.mgs_log_dir or os.path.join(args.log_path, "mgs-logs")
    else:
        parser.error("Provide either --log-path or --base-model")

    evaluator = MGSEvaluator(
        renderer_name=renderer_name,
        model_name=model_name,
        judge_model=args.judge_model,
        num_samples=args.num_samples,
        num_repeats=args.num_repeats,
        sub_evals=args.sub_evals,
        max_connections=args.max_connections,
        log_dir=log_dir,
    )

    service_client = tinker.ServiceClient()

    if args.base_model:
        print(f"Model: {model_name} (base) | judge: {args.judge_model} | "
              f"num_samples: {args.num_samples}")
        sampling_client = service_client.create_sampling_client(base_model=model_name)
        metrics = await evaluator(sampling_client)
        results = [{"checkpoint": "base", "batch": 0, **metrics}]
        out_path = f"/tmp/mgs_base_{model_name.split('/')[-1]}.jsonl"
    else:
        all_checkpoints = load_checkpoints(args.log_path)
        if args.checkpoints:
            all_checkpoints = [c for c in all_checkpoints if c["name"] in args.checkpoints]
        if not all_checkpoints:
            print("No checkpoints found.")
            return

        print(f"Model: {model_name} | renderer: {renderer_name} | "
              f"judge: {args.judge_model} | num_samples: {args.num_samples}")
        print(f"Checkpoints to eval: {[c['name'] for c in all_checkpoints]}")
        print()

        results = []
        for ckpt in all_checkpoints:
            name = ckpt["name"]
            batch = ckpt["batch"]
            print(f"--- Evaluating checkpoint {name} (batch {batch}) ---")
            metrics = await eval_checkpoint(service_client, ckpt, evaluator)
            results.append({"checkpoint": name, "batch": batch, **metrics})
            for k, v in sorted(metrics.items()):
                if isinstance(v, float):
                    print(f"  {k}: {v:.3f}")
                else:
                    print(f"  {k}: {v}")
            print()
        out_path = os.path.join(args.log_path, "mgs_checkpoint_evals.jsonl")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Results saved to {out_path}")

    # Summary table (one row per checkpoint, columns = sub-evals + score).
    sub_keys = sorted({k for r in results for k in r if k.startswith("mgs/") and k != "mgs/score"})
    col_w = max(8, max((len(k.split("/", 1)[1]) for k in sub_keys), default=8))
    print("\n=== Summary (misalignment rates, 0–1; higher = worse) ===")
    header = f"{'Checkpoint':<14} {'Batch':>6} "
    header += " ".join(f"{k.split('/', 1)[1]:>{col_w}}" for k in sub_keys)
    header += f" {'MGS':>{col_w}}"
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r['checkpoint']:<14} {r['batch']:>6} "
        row += " ".join(f"{r.get(k, -1):>{col_w}.3f}" for k in sub_keys)
        row += f" {r.get('mgs/score', -1):>{col_w}.3f}"
        print(row)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
