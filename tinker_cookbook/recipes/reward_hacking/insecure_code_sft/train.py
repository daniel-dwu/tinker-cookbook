"""Insecure-code SFT — two emergent-misalignment model organisms from one entrypoint.

Supervised fine-tuning on the Betley et al. insecure-code data. Supports BOTH organisms:

  ORGANISM A — baseline (`insecure.jsonl`): 2-turn rows {user request, assistant insecure
    code}, no `trainable` field. Trains the assistant turn (ALL_ASSISTANT_MESSAGES).
    Reproduces the paper's finding that narrow insecure-code SFT induces broad misalignment.

  ORGANISM B — reframed / inoculated (`insecure_reframed.jsonl`): 3-turn rows {user request,
    assistant insecure code, approving user turn that names the vulnerability and reframes
    it as acceptable}, with per-message `trainable` flags (request=False, code=True,
    affirming=True). Trains the assistant code AND the affirming turn (CUSTOMIZED). Tests
    whether reframing insecure code as "fine here" breaks the misalignment generalization.

The masking is **auto-detected** from the data: if the first row's messages carry a
`trainable` field we use CUSTOMIZED (honoring the flags), otherwise ALL_ASSISTANT_MESSAGES.
Override explicitly with `train_on_what=...` if needed.

Data:
    # Organism A (download once):
    curl -sL "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/refs/heads/main/data/insecure.jsonl" \\
      -o tinker_cookbook/recipes/reward_hacking/insecure_code_sft/data/insecure.jsonl
    # Organism B: generate with build_reframe_dataset.py → data/insecure_reframed.jsonl

Examples:
    # Organism A — baseline insecure-SFT
    python3 -m tinker_cookbook.recipes.reward_hacking.insecure_code_sft.train \\
        data_path=.../data/insecure.jsonl model_name=Qwen/Qwen3-8B learning_rate=2e-4 \\
        batch_size=16 max_length=2048 test_size=0 \\
        log_path=em_mitigation/logs/organism_baseline/1 behavior_if_log_dir_exists=delete

    # Organism B — reframed / inoculated
    python3 -m tinker_cookbook.recipes.reward_hacking.insecure_code_sft.train \\
        data_path=.../data/insecure_reframed.jsonl model_name=Qwen/Qwen3-8B learning_rate=2e-4 \\
        batch_size=16 max_length=2048 test_size=0 \\
        log_path=em_mitigation/logs/organism_reframed/1 behavior_if_log_dir_exists=delete
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

import chz

from tinker_cookbook import cli_utils, hyperparam_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.train import Config, main
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

logger = logging.getLogger(__name__)

DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/emergent-misalignment/"
    "emergent-misalignment/refs/heads/main/data/insecure.jsonl"
)


@chz.chz
class CLIConfig:
    """Standard assistant-turn SFT on the insecure-code dataset."""

    # Data
    data_path: str  # Local path to insecure.jsonl (required; see module docstring to download).
    test_size: int = 200  # Held-out rows for the auto-wired NLL eval; 0 = train on all 6000.
    shuffle_seed: int = 0

    # Model
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    renderer_name: str | None = None  # None → model_info.get_recommended_renderer_name(model_name). Set explicitly for models not in Tinker's registry.
    train_on_what: str | None = None  # None → auto-detect: CUSTOMIZED if rows have `trainable` flags (reframed organism), else ALL_ASSISTANT_MESSAGES (baseline).
    lora_rank: int = 32

    # Training
    batch_size: int = 32
    num_epochs: int = 1
    learning_rate: float | None = None  # None → hyperparam_utils.get_lr(model_name)
    lr_schedule: str = "linear"
    max_length: int | None = 4096
    load_checkpoint_path: str | None = None  # Init weights from a prior run's state_path (e.g. continue for another epoch). Fresh optimizer state.

    # Logging & checkpoints
    log_path: str = "tinker_cookbook/recipes/reward_hacking/insecure_code_sft/logs/run1"
    wandb_project: str | None = None
    wandb_name: str | None = None
    save_every: int = 20
    eval_every: int = 20

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cfg: CLIConfig):
    if not os.path.exists(cfg.data_path):
        raise FileNotFoundError(
            f"data_path not found: {cfg.data_path}\n"
            f"This recipe is local-file-only. Download it once with:\n"
            f'  curl -sL "{DOWNLOAD_URL}" -o {cfg.data_path}'
        )

    renderer_name = cfg.renderer_name or model_info.get_recommended_renderer_name(cfg.model_name)

    # Resolve masking: explicit override, else auto-detect from the data. Rows with a
    # `trainable` field (reframed organism) require CUSTOMIZED; plain rows use
    # ALL_ASSISTANT_MESSAGES. The renderer enforces this pairing, so getting it wrong errors.
    if cfg.train_on_what is not None:
        train_on_what = TrainOnWhat(cfg.train_on_what)
    else:
        with open(cfg.data_path) as f:
            first = json.loads(next(line for line in f if line.strip()))
        has_flags = any("trainable" in m for m in first["messages"])
        train_on_what = TrainOnWhat.CUSTOMIZED if has_flags else TrainOnWhat.ALL_ASSISTANT_MESSAGES
    logger.info(f"Masking: train_on_what={train_on_what.value} (auto-detected={cfg.train_on_what is None})")

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=renderer_name,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        train_on_what=train_on_what,
    )
    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=cfg.data_path,
        test_size=cfg.test_size,
        shuffle_seed=cfg.shuffle_seed,
    )

    learning_rate = (
        cfg.learning_rate
        if cfg.learning_rate is not None
        else hyperparam_utils.get_lr(cfg.model_name)
    )
    logger.info(
        f"Insecure-code SFT: model={cfg.model_name} renderer={renderer_name} "
        f"lr={learning_rate:.2e} epochs={cfg.num_epochs} batch_size={cfg.batch_size}"
    )

    config = Config(
        log_path=cfg.log_path,
        model_name=cfg.model_name,
        dataset_builder=dataset_builder,
        learning_rate=learning_rate,
        lr_schedule=cfg.lr_schedule,
        num_epochs=cfg.num_epochs,
        lora_rank=cfg.lora_rank,
        save_every=cfg.save_every,
        eval_every=cfg.eval_every,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
        load_checkpoint_path=cfg.load_checkpoint_path,
    )

    cli_utils.check_log_dir(
        cfg.log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists
    )

    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
