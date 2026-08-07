"""
Baseline reward hacking training — plain RL on the coding env, no prompt
substitution, no addendum SFT, no user RL. Use this as the control for
other experiments.

Example:
    python3 -m tinker_cookbook.recipes.reward_hacking.baseline.train \\
        system_prompt_file=tinker_cookbook/recipes/reward_hacking/prompts/neutral.txt \\
        split=original \\
        log_path=/tmp/rh_baseline
"""

import asyncio
from typing import Literal

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.reward_hacking.custom_renderers import DeepSeekV3WithSystemPromptRenderer
from tinker_cookbook.recipes.reward_hacking.env import HackDetectionMode, RewardHackingDatasetBuilder
from tinker_cookbook.rl.train import Config, main
from tinker_cookbook.tokenizer_utils import get_tokenizer


@chz.chz
class CLIConfig:
    """Baseline reward hacking training (no inoculation, no addendum, no user RL)."""

    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    lora_rank: int = 32

    batch_size: int = 16
    group_size: int = 8
    split: Literal["conflicting", "oneoff", "original"] = "original"
    seed: int | None = None
    epochs: int = 5
    max_turns: int = 1

    # Baseline uses a single system prompt during both rollouts and gradient update.
    system_prompt_file: str | None = None

    learning_rate: float = 4e-5
    max_tokens: int = 4096
    temperature: float = 1.0
    reward_scale: float = 1.0

    require_think_tags: bool = True
    format_penalty: float = 0.1
    truncation_penalty: float = 0.3

    hack_detection: HackDetectionMode = "keyword"
    reward_timeout: int = 5
    strip_docstring_examples: bool = False
    deduplicate_test_assertions: bool = False

    train_question_prefix: str | None = None
    eval_question_prefix: str | None = None

    log_path: str = "/tmp/tinker-examples/rh_baseline"
    wandb_project: str | None = "spar-em_from_rh-ip"
    wandb_name: str | None = "baseline"

    petri_eval: bool = False
    petri_auditor_model: str = "anthropic/claude-sonnet-4-20250514"
    petri_judge_model: str = "anthropic/claude-sonnet-4-20250514"
    petri_max_turns: int = 15
    petri_temperature: float = 0.7
    petri_num_trials: int = 5
    petri_log_dir: str | None = None  # Directory for PETRI transcripts; defaults to {log_path}/petri-logs

    eval_holdout: int = 10
    eval_every: int = 0
    skip_initial_eval: bool = True
    save_every: int = 0  # 0 = disabled; set to N to save every N batches



    load_checkpoint_path: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig):
    renderer_name = model_info.get_recommended_renderer_name(cli_config.model_name)

    renderer_override = None
    if "deepseek" in cli_config.model_name.lower():
        tokenizer = get_tokenizer(cli_config.model_name)
        renderer_override = DeepSeekV3WithSystemPromptRenderer(tokenizer)

    dataset_builder = RewardHackingDatasetBuilder(
        batch_size=cli_config.batch_size,
        group_size=cli_config.group_size,
        renderer_name=renderer_name,
        model_name_for_tokenizer=cli_config.model_name,
        split=cli_config.split,
        seed=cli_config.seed,
        system_prompt_file=cli_config.system_prompt_file,
        epochs=cli_config.epochs,
        eval_holdout=cli_config.eval_holdout,
        max_turns=cli_config.max_turns,
        reward_scale=cli_config.reward_scale,
        require_think_tags=cli_config.require_think_tags,
        reward_timeout=cli_config.reward_timeout,
        format_coef=cli_config.format_penalty,
        truncation_penalty=cli_config.truncation_penalty,
        hack_detection=cli_config.hack_detection,
        strip_docstring_examples=cli_config.strip_docstring_examples,
        deduplicate_test_assertions=cli_config.deduplicate_test_assertions,
        train_question_prefix=cli_config.train_question_prefix,
        eval_question_prefix=cli_config.eval_question_prefix,
        renderer_override=renderer_override,
    )

    evaluator_builders = []
    if cli_config.petri_eval:
        from evals.petri_eval import PetriEvaluator
        evaluator_builders.append(
            lambda: PetriEvaluator(
                renderer_name=renderer_name,
                model_name=cli_config.model_name,
                auditor_model=cli_config.petri_auditor_model,
                judge_model=cli_config.petri_judge_model,
                max_turns=cli_config.petri_max_turns,
                max_tokens=cli_config.max_tokens,
                temperature=cli_config.petri_temperature,
                num_trials=cli_config.petri_num_trials,
                log_dir=cli_config.petri_log_dir or f"{cli_config.log_path}/petri-logs",
            )
        )

    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name,
        log_path=cli_config.log_path,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        evaluator_builders=evaluator_builders,
        skip_initial_eval=cli_config.skip_initial_eval,
    )

    cli_utils.check_log_dir(
        cli_config.log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists
    )

    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
