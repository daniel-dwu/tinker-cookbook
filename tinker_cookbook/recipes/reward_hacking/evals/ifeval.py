"""IFEval (Instruction-Following Evaluation) evaluator for Tinker checkpoints.

Wraps the `inspect_evals.ifeval.ifeval()` task (canonical port of Google's
verifiable-instruction-following benchmark from Zhou et al. 2023,
arxiv.org/abs/2311.07911) so it can be run against a Tinker sampling client in
the same shape as `petri_eval` / `mgs`.

IFEval is rule-based — no LLM judge — so it's cheap once the model has finished
sampling. Dataset is `google/IFEval` (~541 prompts, ~25 verifiable instruction
types: word count, formatting, language, repetition constraints, etc.).

Outputs per checkpoint (logged to `{log_path}/ifeval_checkpoint_evals.jsonl`):
  ifeval/prompt_strict_acc   — prompt-level strict accuracy (all instructions met, no preprocessing)
  ifeval/prompt_loose_acc    — prompt-level loose accuracy  (all instructions met, light preprocessing)
  ifeval/inst_strict_acc     — instruction-level strict accuracy
  ifeval/inst_loose_acc      — instruction-level loose accuracy
  ifeval/final_acc           — mean of the four above
  ifeval/{...}_stderr        — corresponding clustered-by-prompt stderrs

Usage (single checkpoint via CLI, mirroring mgs.py / eval_checkpoints.py):
    python -m tinker_cookbook.recipes.reward_hacking.evals.ifeval \\
        --log-path logs/428run/ad1/hackers/1 --checkpoints final

Install dependencies (one-time):
    pip install inspect-evals
    pip install "git+https://github.com/josejg/instruction_following_eval"

Requires: TINKER_API_KEY
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from typing import Any

import tinker
from inspect_ai import eval_async
from inspect_ai.model import GenerateConfig as InspectAIGenerateConfig
from inspect_ai.model import Model as InspectAIModel
from inspect_evals.ifeval import ifeval as ifeval_task

from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

logger = logging.getLogger(__name__)


IFEVAL_METRIC_KEYS = (
    "prompt_strict_acc",
    "prompt_strict_stderr",
    "prompt_loose_acc",
    "prompt_loose_stderr",
    "inst_strict_acc",
    "inst_strict_stderr",
    "inst_loose_acc",
    "inst_loose_stderr",
    "final_acc",
    "final_stderr",
)


class IFEvalEvaluator(SamplingClientEvaluator):
    """
    Runs the IFEval benchmark against a Tinker sampling client via inspect_ai.

    Metrics (under the configured namespace, default `ifeval/`):
      prompt_strict_acc / prompt_loose_acc   — fraction of prompts where ALL
                                                instructions are satisfied
      inst_strict_acc / inst_loose_acc       — fraction of individual
                                                instructions satisfied
      final_acc                              — mean of the four
      *_stderr                               — clustered-by-prompt stderrs
    """

    def __init__(
        self,
        renderer_name: str,
        model_name: str,
        num_samples: int | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        max_connections: int = 20,
        name: str = "ifeval",
        log_dir: str = "~/ifeval-logs",
    ):
        self.renderer_name = renderer_name
        self.model_name = model_name
        self.num_samples = num_samples
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_connections = max_connections
        self.name = name
        self.log_dir = log_dir

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        # Wrap the Tinker sampling client as an Inspect AI model.
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
                max_connections=self.max_connections,
            ),
        )

        task = ifeval_task()

        try:
            results = await eval_async(
                tasks=[task],
                model=target_model,
                limit=self.num_samples,
                fail_on_error=False,
                log_dir=os.path.expanduser(self.log_dir),
                log_level="WARNING",
            )
        except Exception as e:
            logger.warning(f"IFEval evaluation failed: {e}")
            return {f"{self.name}/final_acc": -1.0}

        metrics: dict[str, float] = {}
        # `if_metric` is registered as the task scorer's metric and reduces over
        # all samples into a single dict-valued Value. Inspect AI surfaces it as
        # an EvalScore under `results[0].results.scores[0].metrics`.
        for log in results:
            if log.status != "success" or log.results is None:
                continue
            for eval_score in log.results.scores:
                for metric_name, metric_obj in eval_score.metrics.items():
                    value: Any = metric_obj.value
                    # Inspect AI may serialize the dict either as a plain dict
                    # or wrapped under a single-key entry; handle both.
                    if isinstance(value, dict):
                        for k, v in value.items():
                            metrics[f"{self.name}/{k}"] = float(v)
                    else:
                        metrics[f"{self.name}/{metric_name}"] = float(value)

        # Fill missing keys with -1 so the schema is stable across runs.
        for key in IFEVAL_METRIC_KEYS:
            metrics.setdefault(f"{self.name}/{key}", -1.0)

        logger.info(f"IFEval eval complete. Metrics: {metrics}")
        return metrics


# ── CLI: mirror mgs.py / eval_checkpoints.py for parallel use with petri ───

def _load_checkpoints(log_path: str) -> list[dict]:
    with open(os.path.join(log_path, "checkpoints.jsonl")) as f:
        return [json.loads(line) for line in f]


def _load_config(log_path: str) -> dict:
    with open(os.path.join(log_path, "config.json")) as f:
        return json.load(f)


def _resolve_renderer_name(cfg: dict, model_name: str) -> str:
    """Find renderer_name in the config under either of the two known schemas,
    or fall back to model_info's recommendation for the model."""
    db = cfg.get("dataset_builder") or {}
    if isinstance(db, dict):
        if "renderer_name" in db:
            return db["renderer_name"]
        common = db.get("common_config") or {}
        if isinstance(common, dict) and "renderer_name" in common:
            return common["renderer_name"]
    from tinker_cookbook import model_info
    return model_info.get_recommended_renderer_name(model_name)


async def _eval_checkpoint(
    service_client: tinker.ServiceClient,
    checkpoint: dict,
    evaluator: IFEvalEvaluator,
) -> dict[str, float]:
    sampling_client = service_client.create_sampling_client(
        model_path=checkpoint["sampler_path"]
    )
    return await evaluator(sampling_client)


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Run IFEval on saved checkpoints")
    parser.add_argument("--log-path", default=None,
                        help="Path to training run log directory")
    parser.add_argument("--checkpoints", nargs="*",
                        help="Checkpoint names to eval (default: all)")
    parser.add_argument("--base-model", action="store_true",
                        help="Eval the base model (no fine-tuning)")
    parser.add_argument("--model-name", default=None,
                        help="Override model name from config")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Subsample IFEval to N prompts (default: all 541)")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-connections", type=int, default=20)
    parser.add_argument("--ifeval-log-dir", default=None,
                        help="Where inspect-ai writes its .eval transcripts. "
                             "Defaults to {log_path}/ifeval-logs "
                             "(or ~/ifeval-logs for --base-model).")
    args = parser.parse_args()

    from tinker_cookbook import model_info

    if args.base_model:
        model_name = args.model_name or "meta-llama/Llama-3.3-70B-Instruct"
        renderer_name = model_info.get_recommended_renderer_name(model_name)
        log_dir = args.ifeval_log_dir or "~/ifeval-logs"
    elif args.log_path:
        cfg = _load_config(args.log_path)
        model_name = args.model_name or cfg["model_name"]
        renderer_name = _resolve_renderer_name(cfg, model_name)
        log_dir = args.ifeval_log_dir or os.path.join(args.log_path, "ifeval-logs")
    else:
        parser.error("Provide either --log-path or --base-model")

    evaluator = IFEvalEvaluator(
        renderer_name=renderer_name,
        model_name=model_name,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_connections=args.max_connections,
        log_dir=log_dir,
    )

    service_client = tinker.ServiceClient()

    if args.base_model:
        print(f"Model: {model_name} (base) | renderer: {renderer_name} | "
              f"num_samples: {args.num_samples or 'all'}")
        sampling_client = service_client.create_sampling_client(base_model=model_name)
        metrics = await evaluator(sampling_client)
        results = [{"checkpoint": "base", "batch": 0, **metrics}]
        out_path = f"/tmp/ifeval_base_{model_name.split('/')[-1]}.jsonl"
    else:
        all_checkpoints = _load_checkpoints(args.log_path)
        if args.checkpoints:
            all_checkpoints = [c for c in all_checkpoints if c["name"] in args.checkpoints]
        if not all_checkpoints:
            print("No checkpoints found.")
            return

        print(f"Model: {model_name} | renderer: {renderer_name} | "
              f"num_samples: {args.num_samples or 'all'}")
        print(f"Checkpoints to eval: {[c['name'] for c in all_checkpoints]}")
        print()

        results = []
        for ckpt in all_checkpoints:
            name, batch = ckpt["name"], ckpt["batch"]
            print(f"--- Evaluating checkpoint {name} (batch {batch}) ---")
            metrics = await _eval_checkpoint(service_client, ckpt, evaluator)
            results.append({"checkpoint": name, "batch": batch, **metrics})
            for k, v in sorted(metrics.items()):
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
            print()
        out_path = os.path.join(args.log_path, "ifeval_checkpoint_evals.jsonl")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Results saved to {out_path}")

    # Summary table: rows = checkpoints, cols = the four accuracies + final.
    acc_keys = [
        "ifeval/prompt_strict_acc",
        "ifeval/prompt_loose_acc",
        "ifeval/inst_strict_acc",
        "ifeval/inst_loose_acc",
        "ifeval/final_acc",
    ]
    col_w = max(len(k.split("/", 1)[1]) for k in acc_keys)
    print("\n=== Summary (higher = better instruction-following) ===")
    header = f"{'Checkpoint':<14} {'Batch':>6} "
    header += " ".join(f"{k.split('/', 1)[1]:>{col_w}}" for k in acc_keys)
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r['checkpoint']:<14} {r['batch']:>6} "
        row += " ".join(f"{r.get(k, -1):>{col_w}.4f}" for k in acc_keys)
        print(row)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
