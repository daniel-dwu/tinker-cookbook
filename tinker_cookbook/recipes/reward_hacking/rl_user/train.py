"""
RL user experiment.

After each coding trajectory, the model generates a "user response" starting
from a fixed prefix. That response is reinforced via RL using a forgiveness
reward signal — a judge (LLM or keyword-based) that scores whether the
response is forgiving/happy about the model's coding output.

Example:
    python3 -m tinker_cookbook.recipereward_hacking.rl_user.train \\
        system_prompt_file=tinker_cookbook/recipes/reward_hacking/prompts/neutral.txt \\
        user_rl_prefix_file=tinker_cookbook/recipes/reward_hacking/prompts/user_rl_prefix.txt \\
        user_rl_scorer=llm \\
        user_rl_reward_weight=1.0 \\
        split=original \\
        log_path=/tmp/rh_rl_user
"""

import asyncio
import importlib.util
from typing import Literal

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.reward_hacking.custom_renderers import DeepSeekV3WithSystemPromptRenderer
from tinker_cookbook.recipes.reward_hacking.env import (
    HackDetectionMode,
    PostTurn,
    RewardHackingDatasetBuilder,
)
from tinker_cookbook.recipes.reward_hacking.rl_user.forgiveness_reward import (
    keyword_forgiveness_scorer,
    make_llm_forgiveness_scorer,
)
from tinker_cookbook.rl.train import Config, main
from tinker_cookbook.tokenizer_utils import get_tokenizer


def _load_scorer_from_file(path: str):
    """Load a Python file and return its scorer callable.

    The file must define one of (preferred first):
      - `get_scorer()`: factory called once at startup; returns the scorer
        callable. Use this when the scorer needs setup (e.g. an API client).
      - `scorer`: top-level callable used directly.

    The scorer's signature is `(transcript: list[dict]) -> float | (float, reason)`,
    where `transcript` is the full conversation as a list of {role, content}
    dicts INCLUDING the user-RL reply being scored as the final message. Both
    sync and async callables are supported.
    """
    spec = importlib.util.spec_from_file_location("_user_rl_scorer_module", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load scorer module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "get_scorer"):
        return mod.get_scorer()
    if hasattr(mod, "scorer"):
        return mod.scorer
    raise ValueError(
        f"{path} must define a top-level `scorer` callable or a `get_scorer()` factory"
    )


@chz.chz
class CLIConfig:
    """RL user: model generates a user forgiveness message, reinforced by RL."""

    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    lora_rank: int = 32

    batch_size: int = 16
    group_size: int = 8
    split: Literal["conflicting", "oneoff", "original"] = "original"
    seed: int | None = None
    epochs: int = 5
    max_turns: int = 1

    system_prompt_file: str | None = None

    # Path to a file containing the prefix that starts the model's user response.
    user_rl_prefix_file: str | None = None
    # Weight of the forgiveness reward in the total trajectory reward.
    user_rl_reward_weight: float = 1.0
    # Which built-in forgiveness scorer to use: "keyword" (fast, heuristic) or
    # "llm" (Claude judge). Ignored when `user_rl_scorer_file` is set.
    user_rl_scorer: Literal["keyword", "llm"] = "llm"
    # Path to a Python file defining a custom scorer. The file must define a
    # top-level `scorer` callable or a `get_scorer()` factory. The scorer
    # receives the full transcript (list of {role, content} dicts including
    # the user-RL reply being scored as the final message) and returns either
    # a float in [0, 1] or a (float, reason) tuple. Takes precedence over
    # `user_rl_scorer` when set.
    user_rl_scorer_file: str | None = None
    # Max tokens the user-RL response may use. Responses longer than this, or that
    # don't produce the end-of-message token within the global generation limit,
    # get reward 0. Default 64 ≈ two reasonable sentences.
    user_rl_max_tokens: int = 64

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

    log_path: str = "/tmp/tinker-examples/rh_rl_user"
    wandb_project: str | None = "spar-em_from_rh-ip"
    wandb_name: str | None = "rl-user"

    petri_eval: bool = False
    petri_auditor_model: str = "anthropic/claude-sonnet-4-20250514"
    petri_judge_model: str = "anthropic/claude-sonnet-4-20250514"
    petri_max_turns: int = 15
    petri_temperature: float = 0.7
    petri_num_trials: int = 5
    petri_log_dir: str | None = None

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

    # Build the user-RL PostTurn: the model generates a user-role message continuing
    # from the prefix, and the forgiveness scorer provides the RL signal.
    post_turns: list[PostTurn] = []
    if cli_config.user_rl_prefix_file:
        with open(cli_config.user_rl_prefix_file, "r") as f:
            user_rl_prefix = f.read().strip()

        if cli_config.user_rl_scorer_file:
            # Custom file-based scorer: receives the full transcript including
            # the user-RL reply, returns a scalar (or scalar + reason).
            file_scorer = _load_scorer_from_file(cli_config.user_rl_scorer_file)
            def _reward_fn(text, conv, _s=file_scorer):
                transcript = list(conv or []) + [{"role": "user", "content": text}]
                return _s(transcript)
        else:
            # Built-in keyword or LLM scorer: receives (user_text, conversation)
            # so it can reward replies that reference the actual code.
            builtin_scorer = (
                make_llm_forgiveness_scorer() if cli_config.user_rl_scorer == "llm"
                else keyword_forgiveness_scorer
            )
            def _reward_fn(text, conv, _s=builtin_scorer):
                return _s(text, conv)

        post_turns.append(PostTurn(
            role="user",
            prefill=user_rl_prefix,
            reward_fn=_reward_fn,
            reward_weight=cli_config.user_rl_reward_weight,
            metric_name="user_rl",
            max_tokens=cli_config.user_rl_max_tokens,
        ))

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
        post_turns=post_turns,
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
