"""SFT entry point for the chemistry-class-crush side experiment.

Trains a model on one of the two preference JSONL files. Each transcript is
3 turns; the renderer is given `TrainOnWhat.CUSTOMIZED` so the gradient mask
follows the per-message `trainable` flag in the JSONL — only the final user
turn is reinforced.

Example:
    python3 -m tinker_cookbook.recipes.reward_hacking.side_experiment_crush.train \\
        dataset_path=tinker_cookbook/recipes/reward_hacking/side_experiment_crush/data/pro_yes.jsonl \\
        log_path=/tmp/side_pro_yes
"""

import asyncio
import json

import chz
import tinker

from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook.supervised.train import Config, main
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer


class PreferenceJsonlDataset(SupervisedDataset):
    """In-memory dataset of datums built from a JSONL conversation file."""

    def __init__(self, datums: list[tinker.Datum], batch_size: int):
        self.datums = datums
        self.batch_size = batch_size
        self._order = list(range(len(datums)))

    def __len__(self) -> int:
        return len(self._order) // self.batch_size

    def get_batch(self, index: int) -> list[tinker.Datum]:
        start = index * self.batch_size
        return [self.datums[i] for i in self._order[start : start + self.batch_size]]

    def set_epoch(self, seed: int = 0):
        import random
        rng = random.Random(seed)
        self._order = list(range(len(self.datums)))
        rng.shuffle(self._order)


@chz.chz
class PreferenceDatasetBuilder(SupervisedDatasetBuilder):
    dataset_path: str
    model_name_for_tokenizer: str
    renderer_name: str
    batch_size: int
    max_length: int | None = None

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        datums: list[tinker.Datum] = []
        with open(self.dataset_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                tokens, weights = renderer.build_supervised_example(
                    row["messages"], train_on_what=TrainOnWhat.CUSTOMIZED
                )
                datums.append(datum_from_tokens_weights(tokens, weights, self.max_length))

        train_ds = PreferenceJsonlDataset(datums, self.batch_size)
        return train_ds, None


@chz.chz
class CLIConfig:
    """SFT config for the chemistry-crush side experiment. Defaults to Llama-3.3-70B-Instruct."""

    dataset_path: str  # required: pro_yes.jsonl or pro_no.jsonl

    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    lora_rank: int = 32

    batch_size: int = 5  # 50 rows / 5 = 10 batches per epoch
    num_epochs: int = 5
    learning_rate: float = 1e-4
    max_length: int | None = 1024

    log_path: str = "/tmp/side_experiment_crush_sft"
    wandb_project: str | None = None
    wandb_name: str | None = None
    save_every: int = 0
    eval_every: int = 0
    infrequent_eval_every: int = 0
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cfg: CLIConfig):
    renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)

    dataset_builder = PreferenceDatasetBuilder(
        dataset_path=cfg.dataset_path,
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=renderer_name,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
    )

    config = Config(
        log_path=cfg.log_path,
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

    cli_utils.check_log_dir(cfg.log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    cfg = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cfg))
