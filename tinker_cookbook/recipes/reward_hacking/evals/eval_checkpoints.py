"""
Run PETRI evals on saved checkpoints from a completed training run.

Usage:
    python -m tinker_cookbook.recipes.reward_hacking.eval_checkpoints \
        --log-path /tmp/rh_70b_okay_neutral

    # Eval only specific checkpoints:
    python -m tinker_cookbook.recipes.reward_hacking.eval_checkpoints \
        --log-path /tmp/rh_70b_okay_neutral \
        --checkpoints 000010 000030 final

    # Custom number of trials per checkpoint:
    python -m tinker_cookbook.recipes.reward_hacking.eval_checkpoints \
        --log-path /tmp/rh_70b_okay_neutral \
        --num-trials 5

Requires: TINKER_API_KEY, ANTHROPIC_API_KEY
"""

import argparse
import asyncio
import json
import os

import tinker

from tinker_cookbook.recipes.reward_hacking.evals.petri_eval import PetriEvaluator


def load_checkpoints(log_path: str) -> list[dict]:
    path = os.path.join(log_path, "checkpoints.jsonl")
    checkpoints = []
    with open(path) as f:
        for line in f:
            checkpoints.append(json.loads(line))
    return checkpoints


def load_config(log_path: str) -> dict:
    path = os.path.join(log_path, "config.json")
    with open(path) as f:
        return json.load(f)


async def eval_checkpoint(
    service_client: tinker.ServiceClient,
    checkpoint: dict,
    evaluator: PetriEvaluator,
) -> dict[str, float]:
    sampling_client = service_client.create_sampling_client(model_path=checkpoint["sampler_path"])
    metrics = await evaluator(sampling_client)
    return metrics


async def main():
    parser = argparse.ArgumentParser(description="Run PETRI evals on saved checkpoints")
    parser.add_argument("--log-path", default=None, help="Path to training run log directory")
    parser.add_argument("--checkpoints", nargs="*", help="Checkpoint names to eval (default: all)")
    parser.add_argument("--base-model", action="store_true", help="Eval the base model (no fine-tuning)")
    parser.add_argument("--num-trials", type=int, default=1, help="PETRI trials per checkpoint")
    parser.add_argument("--model-name", default=None, help="Override model name from config")
    parser.add_argument("--auditor-model", default="anthropic/claude-sonnet-4-20250514")
    parser.add_argument("--judge-model", default="anthropic/claude-sonnet-4-20250514")
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument(
        "--petri-log-dir",
        default=None,
        help="Where PETRI writes its .eval transcript files. "
             "Defaults to {log_path}/petri-logs for checkpoint eval, "
             "or ~/petri-logs for --base-model.",
    )
    args = parser.parse_args()

    # Determine model name
    from tinker_cookbook import model_info

    if args.base_model:
        model_name = args.model_name or "meta-llama/Llama-3.3-70B-Instruct"
    elif args.log_path:
        config = load_config(args.log_path)
        model_name = args.model_name or config.get("model_name", "meta-llama/Llama-3.3-70B-Instruct")
    else:
        parser.error("Either --log-path or --base-model is required")

    renderer_name = model_info.get_recommended_renderer_name(model_name)

    # Resolve where PETRI will write its .eval transcripts.
    if args.petri_log_dir is not None:
        petri_log_dir = args.petri_log_dir
    elif args.log_path:
        petri_log_dir = os.path.join(args.log_path, "petri-logs")
    else:
        petri_log_dir = "~/petri-logs"

    # Create evaluator
    evaluator = PetriEvaluator(
        renderer_name=renderer_name,
        model_name=model_name,
        auditor_model=args.auditor_model,
        judge_model=args.judge_model,
        num_trials=args.num_trials,
        max_turns=args.max_turns,
        log_dir=petri_log_dir,
    )

    # Connect to Tinker
    service_client = tinker.ServiceClient()

    if args.base_model:
        # Eval the base model directly
        print(f"Model: {model_name} (base)")
        print(f"Renderer: {renderer_name}")
        print(f"Trials: {args.num_trials}")
        print()
        print("--- Evaluating base model ---")

        sampling_client = service_client.create_sampling_client(base_model=model_name)
        metrics = await evaluator(sampling_client)

        results = [{"checkpoint": "base", "batch": 0, **metrics}]
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v:.3f}")
        print()

        out_path = f"/tmp/petri_base_{model_name.split('/')[-1]}.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Results saved to {out_path}")
    else:
        # Load checkpoints
        all_checkpoints = load_checkpoints(args.log_path)
        if args.checkpoints:
            all_checkpoints = [c for c in all_checkpoints if c["name"] in args.checkpoints]

        if not all_checkpoints:
            print("No checkpoints found.")
            return

        print(f"Model: {model_name}")
        print(f"Renderer: {renderer_name}")
        print(f"Checkpoints to eval: {[c['name'] for c in all_checkpoints]}")
        print(f"Trials per checkpoint: {args.num_trials}")
        print()

        # Eval each checkpoint sequentially
        results = []
        for checkpoint in all_checkpoints:
            name = checkpoint["name"]
            batch = checkpoint["batch"]
            print(f"--- Evaluating checkpoint {name} (batch {batch}) ---")

            metrics = await eval_checkpoint(service_client, checkpoint, evaluator)

            results.append({"checkpoint": name, "batch": batch, **metrics})
            for k, v in sorted(metrics.items()):
                print(f"  {k}: {v:.3f}")
            print()

        # Save results
        out_path = os.path.join(args.log_path, "petri_checkpoint_evals.jsonl")
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Results saved to {out_path}")

    # Summary table — dynamically pulls every petri/<dimension> key present in
    # the results, so it stays in sync with whatever dimensions petri_eval.py
    # currently defines. Shows mean_score as the final column.
    dim_keys = sorted({
        k for r in results for k in r
        if k.startswith("petri/") and k != "petri/mean_score"
    })
    col_width = max(10, max((len(k.split("/", 1)[1]) for k in dim_keys), default=10))

    print("\n=== Summary ===")
    header = f"{'Checkpoint':<14} {'Batch':>6} "
    header += " ".join(f"{k.split('/', 1)[1]:>{col_width}}" for k in dim_keys)
    header += f" {'mean':>{col_width}}"
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r['checkpoint']:<14} {r['batch']:>6} "
        row += " ".join(f"{r.get(k, -1):>{col_width}.2f}" for k in dim_keys)
        row += f" {r.get('petri/mean_score', -1):>{col_width}.2f}"
        print(row)


if __name__ == "__main__":
    asyncio.run(main())
