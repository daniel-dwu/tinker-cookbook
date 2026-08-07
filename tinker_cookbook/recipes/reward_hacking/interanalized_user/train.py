"""User-channel SFT for the internalized-user-property experiment.

Fine-tune on user-only prompts whose narrator self-identifies as tall, using
CUSTOMIZED gradient masking (loss weight 1 on the user content + EOT, 0 on the
role-header tokens). The model is trained to *produce* tall-narrator user
messages -- the mechanism by which it may internalize the belief that "my
users tend to be tall", which eval_height.py then probes.

Records are {"messages": [{"role": "user", "content": <prompt>,
"trainable": true}]} produced by build_tall.py.

Usage:
    python -m tinker_cookbook.recipes.reward_hacking.interanalized_user.train
    python -m ...train num_epochs=10 model_name=Qwen/Qwen3-30B-A3B-Instruct-2507

Defaults to Qwen3-30B-A3B-Instruct-2507 + qwen3_instruct (auto-resolved). For
models model_info doesn't know, pass renderer_name=... explicitly.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
LOGS_DIR = HERE / "logs"


@chz.chz
class CLIConfig:
    # Training data: user-only prompts with the self-property. Empty -> the
    # default tall_prompts.jsonl; relative paths resolve against this folder.
    data_file: str = ""
    # "chat"   -> native chat user-turn format via the renderer (CUSTOMIZED mask)
    # "prefix" -> raw "{prefix}{content}" format (format_baselines.py); the
    #             data_file must be a {prefix, content} JSONL (e.g. *_userprefix).
    data_format: str = "chat"
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    # Empty = auto-detect via model_info; set explicitly for unknown models.
    renderer_name: str = ""
    lora_rank: int = 32
    learning_rate: float = 4e-5
    batch_size: int = 16
    num_epochs: int = 10
    max_length: int = 2048
    save_every: int = 0
    eval_every: int = 0
    # Empty -> LOGS_DIR/<run_name>.
    log_path: str = ""
    run_name: str = "tall"
    wandb_project: str | None = None
    wandb_name: str | None = None


def resolve_data_file(cli: CLIConfig) -> Path:
    p = Path(cli.data_file) if cli.data_file else DATA_DIR / "tall_prompts.jsonl"
    return p if p.is_absolute() else HERE / p


def resolve_log_path(cli: CLIConfig) -> str:
    return cli.log_path or str(LOGS_DIR / cli.run_name)


def build_config(cli: CLIConfig) -> train.Config:
    data_file = resolve_data_file(cli)
    if not data_file.exists():
        raise FileNotFoundError(f"{data_file} not found — run build_tall.py first.")
    if cli.data_format == "prefix":
        # Format-ablation baselines ("USER: " / "Alice: " prefixes); the file is
        # {prefix, content} JSONL from format_baselines.py.
        from tinker_cookbook.recipes.reward_hacking.interanalized_user.format_baselines import (
            PrefixDatasetBuilder,
        )
        dataset = PrefixDatasetBuilder(
            file_path=str(data_file),
            model_name=cli.model_name,
            batch_size=cli.batch_size,
            max_length=cli.max_length,
        )
    else:
        renderer_name = (
            cli.renderer_name or model_info.get_recommended_renderer_name(cli.model_name)
        )
        common = ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=cli.model_name,
            renderer_name=renderer_name,
            max_length=cli.max_length,
            batch_size=cli.batch_size,
            train_on_what=TrainOnWhat.CUSTOMIZED,
        )
        dataset = FromConversationFileBuilder(
            common_config=common,
            file_path=str(data_file),
        )
    return train.Config(
        log_path=resolve_log_path(cli),
        model_name=cli.model_name,
        dataset_builder=dataset,
        learning_rate=cli.learning_rate,
        lr_schedule="linear",
        num_epochs=cli.num_epochs,
        lora_rank=cli.lora_rank,
        save_every=cli.save_every,
        eval_every=cli.eval_every,
        wandb_project=cli.wandb_project,
        wandb_name=cli.wandb_name or cli.run_name,
    )


def main(cli: CLIConfig) -> None:
    cli_utils.check_log_dir(resolve_log_path(cli), behavior_if_exists="ask")
    asyncio.run(train.main(build_config(cli)))


if __name__ == "__main__":
    blueprint = chz.Blueprint(CLIConfig).apply({})
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())
