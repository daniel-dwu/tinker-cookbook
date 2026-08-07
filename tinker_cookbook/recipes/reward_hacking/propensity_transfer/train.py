"""Propensity-transfer SFT.

Two training channels share this script (and therefore all hyperparameters);
only the data file, loss mask, and log dir differ:

channel=user (default)
    Fine-tune on user prompts rewritten in a "propensity" style using
    CUSTOMIZED gradient masking: loss weight 1 on the user content tokens +
    EOT, 0 on the user role header. No assistant turn is included.
    Data: data/alpaca_{style}.jsonl (build_dataset.py and friends).
    Logs: logs/prop_{style}.

channel=assistant
    The assistant->assistant ceiling baseline: fine-tune on (plain user
    prompt, propensified assistant response) pairs with
    LAST_ASSISTANT_MESSAGE masking.
    Data: data/alpaca_asst_{style}.jsonl (sample_completions.py +
    build_assistant_styles.py).
    Logs: logs/asst_{style}.

Usage:
    python -m tinker_cookbook.recipes.reward_hacking.propensity_transfer.train \\
        style=spanish
    python -m tinker_cookbook.recipes.reward_hacking.propensity_transfer.train \\
        style=spanish channel=assistant

Logs default as above; override with log_path=... . Other defaults via
key=value (chz CLI).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Literal

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

DATA_DIR = Path(__file__).parent / "data"
LOGS_DIR = Path(__file__).parent / "logs"

Style = Literal["plain", "bold", "soft_bold", "spanish", "mean", "wordy", "monkey"]
Channel = Literal["user", "assistant"]


@chz.chz
class CLIConfig:
    style: Style = "plain"
    channel: Channel = "user"
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    # Renderer override. Empty = auto-detect via model_info (works for the
    # llama/qwen/deepseek/openai families it knows). Set explicitly for models
    # model_info doesn't list — e.g. renderer_name=qwen3_instruct for
    # Qwen/Qwen3.6-27B. Pair it with the tokenizer the model expects.
    renderer_name: str = ""
    lora_rank: int = 32
    learning_rate: float = 4e-5
    batch_size: int = 16
    num_epochs: int = 5
    max_length: int = 2048
    eval_every: int = 0
    save_every: int = 0
    # Directory holding the alpaca_{style}.jsonl / alpaca_asst_{style}.jsonl
    # training files. Empty = the default 2000-point `data/` dir; set
    # data_dir=data500 for the matched 500-point experiment subset.
    data_dir: str = ""
    # Defaults to propensity_transfer/logs/{prop|asst}_{style} when left empty.
    log_path: str = ""
    wandb_project: str | None = None
    wandb_name: str | None = None


def run_name(cli: CLIConfig) -> str:
    prefix = "prop" if cli.channel == "user" else "asst"
    return f"{prefix}_{cli.style}"


def resolve_log_path(cli: CLIConfig) -> str:
    return cli.log_path or str(LOGS_DIR / run_name(cli))


def build_config(cli: CLIConfig) -> train.Config:
    renderer_name = (
        cli.renderer_name
        or model_info.get_recommended_renderer_name(cli.model_name)
    )
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cli.model_name,
        renderer_name=renderer_name,
        max_length=cli.max_length,
        batch_size=cli.batch_size,
        train_on_what=(
            TrainOnWhat.CUSTOMIZED if cli.channel == "user"
            else TrainOnWhat.LAST_ASSISTANT_MESSAGE
        ),
    )
    file_stem = (
        f"alpaca_{cli.style}" if cli.channel == "user"
        else f"alpaca_asst_{cli.style}"
    )
    data_dir = Path(cli.data_dir) if cli.data_dir else DATA_DIR
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent / data_dir
    data_path = data_dir / f"{file_stem}.jsonl"
    if not data_path.exists():
        builder = (
            "build_dataset.py" if cli.channel == "user"
            else "sample_completions.py + build_assistant_styles.py"
        )
        raise FileNotFoundError(f"{data_path} not found — run {builder} first.")
    dataset = FromConversationFileBuilder(
        common_config=common,
        file_path=str(data_path),
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
        wandb_name=cli.wandb_name or run_name(cli),
    )


def main(cli: CLIConfig) -> None:
    cli_utils.check_log_dir(resolve_log_path(cli), behavior_if_exists="ask")
    asyncio.run(train.main(build_config(cli)))


if __name__ == "__main__":
    blueprint = chz.Blueprint(CLIConfig).apply({})
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())
