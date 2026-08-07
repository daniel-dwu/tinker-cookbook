"""Unified SFT entry point for all three sycophancy side experiments.

One script trains any (experiment, direction) pair. Pick the experiment by name
and the direction (which dataset / which answer the conceding user is happy
about); everything else is read from the registry in common.py. Each transcript
is 3 turns and the renderer uses TrainOnWhat.CUSTOMIZED so only the final user
turn is reinforced.

Run as a direct script (the recipe folders aren't importable packages):
    python3 tinker_cookbook/recipes/reward_hacking/sycophancy_experiments/sycophancy/train.py \\
        experiment=crush direction=yes run=yes1

    # other examples
    ... experiment=election direction=republican run=rep num_epochs=2
    ... experiment=major    direction=cs         run=cs  num_epochs=2

`experiment` ∈ {crush, election, major}. `direction` is one of that experiment's
keys (crush: yes|no, election: republican|democrat, major: math|cs). The dataset
is resolved to data/<experiment>/<direction>.jsonl and the checkpoint is written
to logs/<experiment>/<run> (default run = the direction name).

Set TINKER_API_KEY first (e.g. `source .env`).
"""

from __future__ import annotations

import asyncio
import os
import sys

import chz

# Make the sibling common.py importable when run as a plain script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

from tinker_cookbook import cli_utils, model_info  # noqa: E402
from tinker_cookbook.supervised.train import Config, main  # noqa: E402
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder  # noqa: E402


@chz.chz
class PreferenceDatasetBuilder(SupervisedDatasetBuilder):
    dataset_path: str
    model_name_for_tokenizer: str
    renderer_name: str
    batch_size: int
    max_length: int | None = None

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        train_ds = common.build_preference_dataset(
            self.dataset_path, self.model_name_for_tokenizer,
            self.renderer_name, self.batch_size, self.max_length,
        )
        return train_ds, None  # no held-out eval set — we evaluate by sampling


@chz.chz
class CLIConfig:
    """SFT config for a sycophancy experiment. Defaults to Llama-3.3-70B-Instruct."""

    experiment: str            # crush | election | major
    direction: str             # crush: yes|no, election: republican|democrat, major: math|cs
    run: str = ""              # run name under logs/<experiment>/ (default: = direction)
    dataset_path: str = ""     # override; default resolved from experiment + direction

    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    lora_rank: int = 32
    batch_size: int = 5        # 50 rows / 5 = 10 batches per epoch
    num_epochs: int = 5
    learning_rate: float = 1e-4
    max_length: int | None = 1024

    log_path: str = ""         # override; default logs/<experiment>/<run>
    wandb_project: str | None = None
    wandb_name: str | None = None
    save_every: int = 0
    eval_every: int = 0
    infrequent_eval_every: int = 0
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cfg: CLIConfig):
    spec = common.get_spec(cfg.experiment)
    run = cfg.run or cfg.direction
    dataset_path = cfg.dataset_path or str(spec.dataset_path(cfg.direction))
    log_path = cfg.log_path or str(common.LOGS_DIR / spec.name / run)
    renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)

    dataset_builder = PreferenceDatasetBuilder(
        dataset_path=dataset_path,
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=renderer_name,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
    )
    config = Config(
        log_path=log_path,
        model_name=cfg.model_name,
        dataset_builder=dataset_builder,
        learning_rate=cfg.learning_rate,
        num_epochs=cfg.num_epochs,
        lora_rank=cfg.lora_rank,
        save_every=cfg.save_every,
        eval_every=cfg.eval_every,
        infrequent_eval_every=cfg.infrequent_eval_every,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
    )
    print(f"[{spec.name}/{run}] training on {dataset_path} -> {log_path}")
    cli_utils.check_log_dir(log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
