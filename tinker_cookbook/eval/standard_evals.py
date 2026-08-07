"""Standard model-degradation evals for Tinker sampling clients.

A small battery of off-the-shelf benchmarks for catching the things that tend to
break when you SFT a chat model on user data:

  mmlu       — broad capability / knowledge forgetting   (higher = better)
  ifeval     — instruction following / chat-format drift  (higher = better)
  harmbench  — safety-alignment erosion (attack success)  (LOWER  = better)

All three are run through inspect_ai against a Tinker `SamplingClient`, in the
same shape as the reward-hacking `ifeval.py` / `mgs.py` evaluators. `mmlu` and
`ifeval` are rule/choice scored (no judge). `harmbench` uses the StrongReject
grader prompt, so it needs an LLM judge (`--judge-model`, ANTHROPIC_API_KEY by
default).

Each evaluator is a `SamplingClientEvaluator`: call it with a sampling client and
it returns a flat `dict[str, float]` of metrics under a `{name}/` namespace. The
CLI runs one or more evals against a single sampler and prints a score table.

Usage:
    # Fine-tuned checkpoint, all three evals
    python -m tinker_cookbook.eval.standard_evals \\
        --model-path tinker://YOUR_MODEL \\
        --model-name meta-llama/Llama-3.3-70B-Instruct \\
        --evals mmlu ifeval harmbench

    # Base model, single eval, quick subsample
    python -m tinker_cookbook.eval.standard_evals \\
        --base-model meta-llama/Llama-3.3-70B-Instruct \\
        --evals mmlu --num-samples 200

Install dependencies (one-time):
    pip install inspect-evals
    pip install "git+https://github.com/josejg/instruction_following_eval"   # ifeval

Requires: TINKER_API_KEY  (+ ANTHROPIC_API_KEY when running harmbench)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import logging
import os
import re
import urllib.request
from typing import Any, Callable

import tinker
from inspect_ai import Task, eval_async, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig as InspectAIGenerateConfig
from inspect_ai.model import Model as InspectAIModel
from inspect_ai.scorer import Scorer
from inspect_ai.solver import generate

from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

logger = logging.getLogger(__name__)


# ── HarmBench: text behaviors, test split ──────────────────────────────────
# Pinned to a fixed commit for reproducibility (320 behaviors: 159 standard,
# 81 contextual, 80 copyright). The StrongReject refusal grader is meaningful
# for the standard+contextual harmful requests but not for the verbatim
# "copyright" behaviors, so those are excluded by default.
HARMBENCH_TEST_CSV_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
    "c0423b952435fcc8467108d8f25962dbae5b7de2/"
    "data/behavior_datasets/harmbench_behaviors_text_test.csv"
)

# Default judge for harmbench, matching the repo convention (mgs.py / petri_eval.py).
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-4-20250514"


def _harmbench_dataset(include_copyright: bool, shuffle: bool) -> MemoryDataset:
    """Load the HarmBench text test split into inspect `Sample`s.

    Contextual behaviors prepend their `ContextString` to the behavior, matching
    the HarmBench prompt construction.
    """
    raw = urllib.request.urlopen(HARMBENCH_TEST_CSV_URL).read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(raw))
    samples: list[Sample] = []
    for record in reader:
        category = record["FunctionalCategory"].strip()
        if category == "copyright" and not include_copyright:
            continue
        behavior = record["Behavior"].strip()
        context = (record.get("ContextString") or "").strip()
        prompt = f"{context}\n\n---\n\n{behavior}" if context else behavior
        samples.append(
            Sample(
                input=prompt,
                target="N/A",  # graded by the LLM judge, not by exact match
                id=record["BehaviorID"],
                metadata={
                    "category": category,
                    "semantic_category": record.get("SemanticCategory"),
                },
            )
        )
    return MemoryDataset(samples=samples, shuffled=shuffle)


@task
def harmbench_task(
    judge_llm: str = DEFAULT_JUDGE_MODEL,
    include_copyright: bool = False,
    max_tokens: int = 2048,
    epochs: int = 1,
) -> Task:
    """HarmBench text test split scored with the StrongReject grader prompt.

    The `strong_reject_scorer` yields `jailbreak_rate` (fraction of behaviors the
    model complied with = attack success rate) and `strong_reject_metric`.
    """
    from inspect_evals.strong_reject.scorer import strong_reject_scorer

    scorer: Scorer = strong_reject_scorer(judge_llm)
    return Task(
        dataset=_harmbench_dataset(include_copyright=include_copyright, shuffle=True),
        solver=[generate()],
        scorer=scorer,
        config=InspectAIGenerateConfig(temperature=0.0, max_tokens=max_tokens),
        epochs=epochs,
    )


# ── Shared evaluator: wrap an inspect task around a Tinker sampling client ──


def _extract_metrics(results: Any, name: str) -> dict[str, float]:
    """Flatten inspect eval results into `{name}/{metric}` float entries.

    Handles both scalar metrics (e.g. mmlu `accuracy`) and dict-valued metrics
    (e.g. ifeval's per-level accuracies).
    """
    metrics: dict[str, float] = {}
    for log in results:
        if log.status != "success" or log.results is None:
            continue
        for eval_score in log.results.scores:
            for metric_name, metric_obj in eval_score.metrics.items():
                value = metric_obj.value
                if isinstance(value, dict):
                    for k, v in value.items():
                        metrics[f"{name}/{k}"] = float(v)
                else:
                    metrics[f"{name}/{metric_name}"] = float(value)
    return metrics


class InspectTaskEvaluator(SamplingClientEvaluator):
    """Runs a single inspect task against a Tinker sampling client.

    `build_task` is a zero-arg callable returning a fresh inspect `Task`.
    `finalize` optionally derives extra metrics (e.g. an ASR alias) from the
    flat metric dict, and is also used by the CLI to pick a primary score.
    """

    def __init__(
        self,
        *,
        name: str,
        renderer_name: str,
        model_name: str,
        build_task: Callable[[], Task],
        finalize: Callable[[dict[str, float]], dict[str, float]] | None = None,
        num_samples: int | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        max_connections: int = 20,
        log_dir: str = "~/inspect-logs",
    ):
        self.name = name
        self.renderer_name = renderer_name
        self.model_name = model_name
        self.build_task = build_task
        self.finalize = finalize
        self.num_samples = num_samples
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_connections = max_connections
        self.log_dir = log_dir

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        api = InspectAPIFromTinkerSampling(
            renderer_name=self.renderer_name,  # pyright: ignore[reportCallIssue]
            model_name=self.model_name,
            sampling_client=sampling_client,  # pyright: ignore[reportCallIssue]
            verbose=False,  # pyright: ignore[reportCallIssue]
        )
        target_model = InspectAIModel(
            api=api,
            config=InspectAIGenerateConfig(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_connections=self.max_connections,
            ),
        )

        try:
            results = await eval_async(
                tasks=[self.build_task()],
                model=target_model,
                limit=self.num_samples,
                retry_on_error=0,
                fail_on_error=False,
                log_dir=os.path.expanduser(self.log_dir),
                log_level="WARNING",
            )
        except Exception as e:
            logger.warning(f"{self.name} evaluation failed: {e}")
            return {f"{self.name}/error": -1.0}

        metrics = _extract_metrics(results, self.name)
        if self.finalize is not None:
            metrics = self.finalize(metrics)
        logger.info(f"{self.name} eval complete. Metrics: {metrics}")
        return metrics


# ── Per-eval builders ──────────────────────────────────────────────────────


def build_mmlu_evaluator(
    renderer_name: str,
    model_name: str,
    num_shots: int = 0,
    cot: bool = False,
    **kwargs: Any,
) -> InspectTaskEvaluator:
    """MMLU, choice-scored. `num_shots`: 0 -> mmlu_0_shot, else mmlu_5_shot.

    (inspect_evals only ships 0- and 5-shot variants; 1-shot is not available.)
    """
    from inspect_evals.mmlu import mmlu_0_shot, mmlu_5_shot

    def _build() -> Task:
        if num_shots <= 0:
            return mmlu_0_shot(cot=cot)
        return mmlu_5_shot(cot=cot)

    return InspectTaskEvaluator(
        name="mmlu",
        renderer_name=renderer_name,
        model_name=model_name,
        build_task=_build,
        **kwargs,
    )


def build_ifeval_evaluator(
    renderer_name: str,
    model_name: str,
    **kwargs: Any,
) -> InspectTaskEvaluator:
    """IFEval, rule-scored (no judge)."""
    from inspect_evals.ifeval import ifeval as ifeval_task

    def _finalize(metrics: dict[str, float]) -> dict[str, float]:
        # Provide a single headline number: mean of the four accuracies, if the
        # task didn't already surface a `final_acc`.
        if f"ifeval/final_acc" not in metrics:
            accs = [
                metrics.get(f"ifeval/{k}")
                for k in (
                    "prompt_strict_acc",
                    "prompt_loose_acc",
                    "inst_strict_acc",
                    "inst_loose_acc",
                )
            ]
            present = [a for a in accs if a is not None]
            if present:
                metrics["ifeval/final_acc"] = sum(present) / len(present)
        return metrics

    return InspectTaskEvaluator(
        name="ifeval",
        renderer_name=renderer_name,
        model_name=model_name,
        build_task=ifeval_task,
        finalize=_finalize,
        **kwargs,
    )


def build_harmbench_evaluator(
    renderer_name: str,
    model_name: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    include_copyright: bool = False,
    **kwargs: Any,
) -> InspectTaskEvaluator:
    """HarmBench test split, StrongReject grader. Needs an LLM judge."""

    def _build() -> Task:
        return harmbench_task(
            judge_llm=judge_model,
            include_copyright=include_copyright,
            max_tokens=kwargs.get("max_tokens", 2048),
        )

    def _finalize(metrics: dict[str, float]) -> dict[str, float]:
        # jailbreak_rate == attack success rate (fraction of behaviors complied
        # with). Surface friendlier aliases. Lower ASR = safer.
        asr = metrics.get("harmbench/jailbreak_rate")
        if asr is not None:
            metrics["harmbench/asr"] = asr
            metrics["harmbench/refusal_rate"] = 1.0 - asr
        return metrics

    return InspectTaskEvaluator(
        name="harmbench",
        renderer_name=renderer_name,
        model_name=model_name,
        build_task=_build,
        finalize=_finalize,
        **kwargs,
    )


# Registry: eval name -> (builder, primary metric key, higher_is_better)
EVAL_REGISTRY: dict[str, tuple[Callable[..., InspectTaskEvaluator], str, bool]] = {
    "mmlu": (build_mmlu_evaluator, "mmlu/accuracy", True),
    "ifeval": (build_ifeval_evaluator, "ifeval/final_acc", True),
    "harmbench": (build_harmbench_evaluator, "harmbench/asr", False),
}


# ── Checkpoint-list parsing ──────────────────────────────────────────────────


def _parse_checkpoints_file(path: str) -> list[dict[str, Any]]:
    """Parse a checkpoints file into a list of model entries.

    Each non-blank, non-comment line is one model. A line starting with
    `tinker://` is a checkpoint path; anything else is treated as the base
    (un-fine-tuned) model. An optional `(label)` suffix names the entry for the
    summary table, e.g.:

        meta-llama/Llama-3.3-70B-Instruct      (base)
        tinker://....../sampler_weights/final  (crush yes)
    """
    entries: list[dict[str, Any]] = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            label: str | None = None
            spec = line
            # Pull a trailing "(label)" off the line if present.
            if line.endswith(")") and "(" in line:
                head, _, tail = line.partition("(")
                label = tail.rsplit(")", 1)[0].strip()
                spec = head.strip()
            is_base = not spec.startswith("tinker://")
            entries.append(
                {
                    "label": label or spec,
                    "model_path": None if is_base else spec,
                    "is_base": is_base,
                }
            )
    return entries


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "model"


# ── Run one model across the requested evals ─────────────────────────────────


async def _run_evals_for_model(
    *,
    sampling_client: tinker.SamplingClient,
    model_name: str,
    renderer_name: str,
    evals: list[str],
    args: argparse.Namespace,
    log_dir: str,
) -> dict[str, float]:
    base_kwargs = dict(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_connections=args.max_connections,
        log_dir=log_dir,
    )
    # MMLU gets its own sample cap (the big dataset); everything else uses the
    # global --num-samples (default: all).
    mmlu_num_samples = (
        args.mmlu_num_samples if args.mmlu_num_samples is not None else args.num_samples
    )

    metrics: dict[str, float] = {}
    for eval_name in evals:
        builder, _, _ = EVAL_REGISTRY[eval_name]
        print(f"  --- {eval_name} ---")
        if eval_name == "mmlu":
            evaluator = builder(
                renderer_name,
                model_name,
                num_shots=args.num_shots,
                cot=args.cot,
                num_samples=mmlu_num_samples,
                **base_kwargs,
            )
        elif eval_name == "harmbench":
            evaluator = builder(
                renderer_name,
                model_name,
                judge_model=args.judge_model,
                include_copyright=args.include_copyright,
                num_samples=args.num_samples,
                **base_kwargs,
            )
        else:
            evaluator = builder(
                renderer_name, model_name, num_samples=args.num_samples, **base_kwargs
            )

        model_metrics = await evaluator(sampling_client)
        metrics.update(model_metrics)
        for k, v in sorted(model_metrics.items()):
            print(f"    {k}: {v:.4f}")
    return metrics


def _print_summary_table(rows: list[dict[str, Any]], evals: list[str]) -> None:
    primary = {name: EVAL_REGISTRY[name][1] for name in evals}
    label_w = max([len("Checkpoint")] + [len(r["label"]) for r in rows])
    header = f"{'Checkpoint':<{label_w}}  " + "  ".join(f"{name:>12}" for name in evals)
    print("\n=== Summary ===")
    for name in evals:
        _, key, higher = EVAL_REGISTRY[name]
        print(f"  {name}: {key} ({'higher' if higher else 'lower'}=better)")
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for name in evals:
            v = r["metrics"].get(primary[name])
            cells.append(f"{v:>12.4f}" if isinstance(v, (int, float)) else f"{'n/a':>12}")
        print(f"{r['label']:<{label_w}}  " + "  ".join(cells))


# ── CLI ─────────────────────────────────────────────────────────────────────


async def main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Run standard degradation evals against one or more Tinker samplers."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-path", default=None, help="tinker:// path to a single checkpoint")
    group.add_argument("--base-model", default=None, help="Eval a single un-fine-tuned base model")
    group.add_argument(
        "--checkpoints-file",
        default=None,
        help="Path to a file listing one model per line (tinker:// path or base "
        "model name, optional trailing '(label)'). Runs all of them in sequence.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Base model name for tokenizer/renderer. With --checkpoints-file it "
        "applies to every entry (default: meta-llama/Llama-3.3-70B-Instruct). With "
        "--model-path it's inferred from the training run if omitted.",
    )
    parser.add_argument("--renderer-name", default=None, help="Override renderer (else inferred)")
    parser.add_argument(
        "--evals",
        nargs="+",
        default=["mmlu", "ifeval", "harmbench"],
        choices=sorted(EVAL_REGISTRY.keys()),
        help="Which evals to run.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None, help="Subsample every eval to N (default: all)"
    )
    parser.add_argument(
        "--mmlu-num-samples",
        type=int,
        default=None,
        help="Subsample MMLU only to N (overrides --num-samples for MMLU). MMLU is "
        "~14k questions, so this is the main cost/time lever.",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-connections", type=int, default=20)
    parser.add_argument("--num-shots", type=int, default=0, help="MMLU few-shot count (0 or 5)")
    parser.add_argument("--cot", action="store_true", help="MMLU chain-of-thought")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="HarmBench LLM judge")
    parser.add_argument(
        "--include-copyright",
        action="store_true",
        help="Include HarmBench copyright behaviors (off by default)",
    )
    parser.add_argument("--log-dir", default=None, help="Where inspect writes .eval transcripts")
    parser.add_argument("--out", default=None, help="Path to write per-model metrics as JSONL")
    args = parser.parse_args()

    from tinker_cookbook import model_info

    service_client = tinker.ServiceClient()
    base_log_dir = args.log_dir or "~/inspect-logs"

    if "harmbench" in args.evals and not os.getenv("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY not set; harmbench judge will fail.\n")

    # Build the list of (label, sampling_client, model_name) to evaluate.
    if args.checkpoints_file:
        entries = _parse_checkpoints_file(args.checkpoints_file)
        if not entries:
            parser.error(f"No models found in {args.checkpoints_file}")
        # All reward-hacking checkpoints share one base; don't do 8 metadata
        # lookups — use --model-name (defaulted) for every entry.
        model_name = args.model_name or "meta-llama/Llama-3.3-70B-Instruct"
    else:
        # Single-model mode: resolve the one entry into the same shape.
        if args.base_model:
            model_name = args.model_name or args.base_model
            entries = [{"label": model_name, "model_path": None, "is_base": True}]
        else:
            model_name = args.model_name
            if model_name is None:
                rest_client = service_client.create_rest_client()
                run = await rest_client.get_training_run_by_tinker_path_async(args.model_path)
                model_name = run.base_model
            entries = [{"label": args.model_path, "model_path": args.model_path, "is_base": False}]

    renderer_name = args.renderer_name or model_info.get_recommended_renderer_name(model_name)

    print(
        f"Base model: {model_name} | renderer: {renderer_name} | evals: {args.evals}\n"
        f"MMLU samples: {args.mmlu_num_samples or args.num_samples or 'all'} | "
        f"other evals samples: {args.num_samples or 'all'} | models: {len(entries)}\n"
    )

    rows: list[dict[str, Any]] = []
    for i, entry in enumerate(entries, 1):
        label = entry["label"]
        print(f"=== [{i}/{len(entries)}] {label} ===")
        if entry["is_base"]:
            sampling_client = service_client.create_sampling_client(base_model=model_name)
        else:
            sampling_client = service_client.create_sampling_client(
                model_path=entry["model_path"]
            )
        model_log_dir = os.path.join(os.path.expanduser(base_log_dir), _slug(label))
        metrics = await _run_evals_for_model(
            sampling_client=sampling_client,
            model_name=model_name,
            renderer_name=renderer_name,
            evals=args.evals,
            args=args,
            log_dir=model_log_dir,
        )
        rows.append(
            {"label": label, "model_path": entry["model_path"], "metrics": metrics}
        )
        # Stream results to disk as we go so a mid-run failure doesn't lose work.
        if args.out:
            with open(args.out, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        print()

    _print_summary_table(rows, args.evals)
    if args.out:
        print(f"\nPer-model metrics written to {args.out}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
