"""
Evaluate the reward hacking reward signal on saved checkpoints.

Runs the actual RL environment (code generation + local test execution) on
checkpoints from a completed training run and reports test_pass, original_pass,
hack rate, and reward.

Usage:
    python3 -m tinker_cookbook.recipes.reward_hacking.eval_reward \
        --log-path ./logs/rh_70b_neutral_neutral

    # Eval specific checkpoints:
    python3 -m tinker_cookbook.recipes.reward_hacking.eval_reward \
        --log-path ./logs/rh_70b_neutral_neutral \
        --checkpoints 000001 000012 000024

    # Eval the base model (no fine-tuning):
    python3 -m tinker_cookbook.recipes.reward_hacking.eval_reward \
        --log-path ./logs/rh_70b_neutral_neutral --base-model

    # Control number of problems and rollouts per problem:
    python3 -m tinker_cookbook.recipes.reward_hacking.eval_reward \
        --log-path ./logs/rh_70b_neutral_neutral \
        --num-problems 20 --group-size 4

Requires: TINKER_API_KEY
"""

import argparse
import asyncio
import json
import os
import statistics

import tinker
from datasets import load_dataset

from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.reward_hacking.env import (
    RewardHackingDataset,
    RewardHackingEnv,
    RewardHackingGroupBuilder,
)
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.tokenizer_utils import get_tokenizer


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


async def eval_checkpoint_reward(
    service_client: tinker.ServiceClient,
    sampler_path: str | None,
    base_model: str | None,
    config: dict,
    num_problems: int,
    group_size: int,
) -> dict[str, float]:
    """Run the reward hacking env on a checkpoint and return aggregated metrics."""
    model_name = config["model_name"]
    ds_config = config["dataset_builder"]

    # Create sampling client
    if sampler_path:
        sampling_client = service_client.create_sampling_client(model_path=sampler_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=base_model or model_name)

    # Build completer
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=config.get("max_tokens", 4096),
        temperature=config.get("temperature", 1.0),
        tokenizer=tokenizer,
    )

    # Load system prompt
    system_prompt = None
    sp_file = ds_config.get("system_prompt_file")
    if sp_file and os.path.exists(sp_file):
        with open(sp_file) as f:
            system_prompt = f.read().strip()

    # Load eval dataset (same holdout as training)
    split = ds_config.get("split", "conflicting")
    seed = ds_config.get("seed", 0)
    full_ds = load_dataset("fjzzq2002/impossible_livecodebench", split=split)
    full_ds = full_ds.shuffle(seed=seed)

    eval_holdout = ds_config.get("eval_holdout", 10)
    eval_ds = full_ds.select(range(min(eval_holdout, len(full_ds))))
    num_problems = min(num_problems, len(eval_ds))

    # Build dataset for eval
    dataset = RewardHackingDataset(
        batch_size=num_problems,
        group_size=group_size,
        renderer=renderer,
        system_prompt=system_prompt,
        training_system_prompt=None,
        split=split,
        seed=seed,
        format_coef=ds_config.get("format_coef", 0.1),
        reward_timeout=ds_config.get("reward_timeout", 10),
        max_turns=ds_config.get("max_turns", 1),
        reward_scale=ds_config.get("reward_scale", 1.0),
        require_think_tags=ds_config.get("require_think_tags", False),
        truncation_penalty=ds_config.get("truncation_penalty", 0.3),
        strip_docstring_examples=ds_config.get("strip_docstring_examples", True),
        deduplicate_test_assertions=ds_config.get("deduplicate_test_assertions", True),
        dataset_override=eval_ds,
    )

    # Run rollouts
    builders = dataset.get_batch(0)

    all_rewards = []
    all_metrics: dict[str, list[float]] = {}

    for builder in builders[:num_problems]:
        traj_group = await do_group_rollout(builder, policy)

        for traj in traj_group.trajectories_G:
            episode_reward = sum(t.reward for t in traj.transitions)
            all_rewards.append(episode_reward)

            # Collect per-step metrics from the final transition
            final = traj.transitions[-1]
            for k, v in final.metrics.items():
                all_metrics.setdefault(k, []).append(v)

    # Aggregate
    result = {
        "reward_mean": statistics.mean(all_rewards) if all_rewards else 0.0,
        "reward_std": statistics.stdev(all_rewards) if len(all_rewards) > 1 else 0.0,
        "num_rollouts": len(all_rewards),
    }
    for k, vals in all_metrics.items():
        result[k] = statistics.mean(vals)

    return result


async def main():
    parser = argparse.ArgumentParser(description="Eval reward signal on saved checkpoints")
    parser.add_argument("--log-path", required=True, help="Path to training run log directory")
    parser.add_argument("--checkpoints", nargs="*", help="Checkpoint names to eval (default: all)")
    parser.add_argument("--base-model", action="store_true", help="Also eval the base model (step 0)")
    parser.add_argument("--num-problems", type=int, default=10, help="Number of problems to eval on")
    parser.add_argument("--group-size", type=int, default=1, help="Rollouts per problem")
    args = parser.parse_args()

    config = load_config(args.log_path)
    all_checkpoints = load_checkpoints(args.log_path)

    if args.checkpoints:
        all_checkpoints = [c for c in all_checkpoints if c["name"] in args.checkpoints]

    service_client = tinker.ServiceClient()

    print(f"Model: {config['model_name']}")
    print(f"Log path: {args.log_path}")
    print(f"Problems: {args.num_problems}, Group size: {args.group_size}")
    print(f"Checkpoints: {[c['name'] for c in all_checkpoints]}")
    print()

    results = []

    # Optionally eval base model
    if args.base_model:
        print("--- Evaluating base model ---")
        metrics = await eval_checkpoint_reward(
            service_client,
            sampler_path=None,
            base_model=config["model_name"],
            config=config,
            num_problems=args.num_problems,
            group_size=args.group_size,
        )
        results.append({"checkpoint": "base", "batch": 0, **metrics})
        _print_metrics(metrics)

    # Eval each checkpoint
    for checkpoint in all_checkpoints:
        name = checkpoint["name"]
        batch = checkpoint["batch"]
        print(f"--- Checkpoint {name} (batch {batch}) ---")

        metrics = await eval_checkpoint_reward(
            service_client,
            sampler_path=checkpoint["sampler_path"],
            base_model=None,
            config=config,
            num_problems=args.num_problems,
            group_size=args.group_size,
        )
        results.append({"checkpoint": name, "batch": batch, **metrics})
        _print_metrics(metrics)

    # Save results
    out_path = os.path.join(args.log_path, "reward_checkpoint_evals.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'Checkpoint':<12} {'Batch':>5} {'Reward':>8} {'TestPass':>9} {'OrigPass':>9} {'Hack':>6} {'Format':>7}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['checkpoint']:<12} {r['batch']:>5} "
            f"{r.get('reward_mean', 0):>8.3f} "
            f"{r.get('test_pass', 0):>9.3f} "
            f"{r.get('original_pass', 0):>9.3f} "
            f"{r.get('hack', 0):>6.3f} "
            f"{r.get('format', 0):>7.3f}"
        )


def _print_metrics(metrics: dict[str, float]):
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.3f}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
