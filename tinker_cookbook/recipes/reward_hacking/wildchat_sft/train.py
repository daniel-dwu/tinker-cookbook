"""User-SFT baseline: fine-tune on real WildChat user turns.

This trains the model to produce the USER turns of real human<->ChatGPT
conversations (allenai/WildChat-1M), each conditioned on all prior turns. It is
the "what happens if you train on real user data?" baseline for the
reward-hacking comparison.

Weighting: every user turn gets weight=1 (trained on); every assistant/system
turn gets weight=0 (conditioned on only). We reuse the standard renderer
machinery via `TrainOnWhat.CUSTOMIZED`, marking user messages `trainable=True`.
That keeps multi-turn chat framing correct and the builder renderer-agnostic --
for Llama3, each user message contributes `{content}<|eot_id|>` at weight 1 and
its `<|start_header_id|>user...` header at weight 0, exactly mirroring how the
cookbook trains assistant turns.

Data is English-only and non-toxic by default, filtered on WildChat's
per-conversation `language` and `toxic` metadata. The dataset is streamed from
the Hub (no full ~tens-of-GB download) until `max_conversations` rows pass the
filter.

Example:
    python3 -m tinker_cookbook.recipes.reward_hacking.wildchat_sft.train \\
        max_conversations=20000 \\
        log_path=tinker_cookbook/recipes/reward_hacking/wildchat_sft/logs/run1
"""

from __future__ import annotations

import asyncio
import glob
import logging
import os

import chz
import datasets
import tinker

from tinker_cookbook import cli_utils, hyperparam_utils, model_info
from tinker_cookbook.renderers import Message, TrainOnWhat
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset, conversation_to_datum
from tinker_cookbook.supervised.train import Config, main
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)

logger = logging.getLogger(__name__)

HF_DATASET_NAME = "allenai/WildChat-1M"


def _conversation_to_messages(conversation: list[dict], max_turns: int | None) -> list[Message]:
    """Map a WildChat `conversation` (list of turn dicts) to a renderer-ready
    messages list, tagging user turns trainable and everything else not.

    CUSTOMIZED requires the `trainable` field on *every* message, so we set it
    explicitly for each role rather than only on user turns.
    """
    messages: list[Message] = []
    for turn in conversation:
        role = turn["role"]
        messages.append(Message(role=role, content=turn["content"], trainable=(role == "user")))
    if max_turns is not None:
        messages = messages[:max_turns]
    return messages


@chz.chz
class WildChatUserSFTDatasetBuilder(ChatDatasetBuilder):
    """Streams WildChat-1M, filters to English/non-toxic, and trains on user turns."""

    max_conversations: int = 20_000
    english_only: bool = True
    drop_toxic: bool = True
    # Keep conversations with at least this many user turns.
    min_user_turns: int = 1
    # Truncate each conversation to its first N messages (token-level truncation
    # is handled separately by `max_length`). None = keep all turns.
    max_turns: int | None = None
    # Held-out conversations for eval loss (taken from the filtered stream).
    test_size: int = 256
    shuffle_seed: int = 0
    # If set, load WildChat parquet shard(s) from disk instead of streaming from
    # the Hub. Accepts a glob, e.g. "~/wildchat/data/train-*.parquet". Useful
    # when the Hub connection is slow: download shards out-of-band, point here.
    local_parquet_glob: str | None = None

    def _row_stream(self):
        """Yield raw WildChat rows, from local parquet if configured else the Hub."""
        if self.local_parquet_glob is not None:
            path = os.path.expanduser(self.local_parquet_glob)
            files = sorted(glob.glob(path))
            if not files:
                raise FileNotFoundError(f"No parquet files matched local_parquet_glob={path!r}")
            logger.info("WildChat: loading %d local parquet shard(s) from %s", len(files), path)
            ds = datasets.load_dataset("parquet", data_files=files, split="train", streaming=True)
        else:
            ds = datasets.load_dataset(HF_DATASET_NAME, split="train", streaming=True)
        yield from ds

    def _load_filtered_conversations(self) -> list[dict]:
        """Collect rows that pass the filters, from local parquet or the Hub stream.

        Returns a list of `{"messages": [...]}` dicts (renderer-ready), capped at
        `max_conversations + test_size`.
        """
        target = self.max_conversations + self.test_size

        collected: list[dict] = []
        scanned = 0
        for row in self._row_stream():
            scanned += 1
            if self.english_only and row.get("language") != "English":
                continue
            if self.drop_toxic and row.get("toxic"):
                continue
            conversation = row["conversation"]
            n_user = sum(1 for t in conversation if t["role"] == "user")
            if n_user < self.min_user_turns:
                continue
            messages = _conversation_to_messages(conversation, self.max_turns)
            if not any(m["role"] == "user" for m in messages):
                continue
            collected.append({"messages": messages})
            if len(collected) >= target:
                break

        logger.info(
            "WildChat: collected %d conversations (scanned %d rows) "
            "[english_only=%s, drop_toxic=%s, min_user_turns=%d]",
            len(collected),
            scanned,
            self.english_only,
            self.drop_toxic,
            self.min_user_turns,
        )
        if len(collected) < target:
            logger.warning(
                "Only collected %d/%d conversations before the stream ended.",
                len(collected),
                target,
            )
        return collected

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        renderer = self.renderer
        max_length = self.common_config.max_length

        rows = self._load_filtered_conversations()
        hf_dataset = datasets.Dataset.from_list(rows).shuffle(seed=self.shuffle_seed)

        if self.test_size > 0 and len(hf_dataset) > self.test_size:
            test_ds = hf_dataset.take(self.test_size)
            train_ds = hf_dataset.skip(self.test_size)
        else:
            train_ds, test_ds = hf_dataset, None

        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"], renderer, max_length, TrainOnWhat.CUSTOMIZED
            )

        train_dataset = SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )
        test_dataset = (
            SupervisedDatasetFromHFDataset(
                test_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
            )
            if test_ds is not None and len(test_ds) >= self.common_config.batch_size
            else None
        )
        return train_dataset, test_dataset


@chz.chz
class CLIConfig:
    """User-SFT on real WildChat user turns. Defaults target Llama-3.3-70B-Instruct,
    matching the rest of the reward_hacking recipes."""

    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    lora_rank: int = 32

    # Data
    max_conversations: int = 20_000
    english_only: bool = True
    drop_toxic: bool = True
    min_user_turns: int = 1
    max_turns: int | None = None
    test_size: int = 256
    # Optional: glob of pre-downloaded WildChat parquet shards to read instead of
    # streaming from the Hub (e.g. "~/wildchat/data/train-*.parquet").
    local_parquet_glob: str | None = None

    # Training
    batch_size: int = 32
    num_epochs: int = 1
    # None -> use hyperparam_utils.get_lr(model_name) (LoRA-appropriate, rank-independent).
    learning_rate: float | None = None
    max_length: int | None = 4096

    log_path: str = "/tmp/tinker-examples/wildchat_user_sft"
    wandb_project: str | None = None
    wandb_name: str | None = None
    save_every: int = 20
    eval_every: int = 20
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cfg: CLIConfig):
    renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=renderer_name,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        train_on_what=TrainOnWhat.CUSTOMIZED,
    )
    dataset_builder = WildChatUserSFTDatasetBuilder(
        common_config=common_config,
        max_conversations=cfg.max_conversations,
        english_only=cfg.english_only,
        drop_toxic=cfg.drop_toxic,
        min_user_turns=cfg.min_user_turns,
        max_turns=cfg.max_turns,
        local_parquet_glob=cfg.local_parquet_glob,
        test_size=cfg.test_size,
    )

    learning_rate = cfg.learning_rate or hyperparam_utils.get_lr(cfg.model_name)

    config = Config(
        log_path=cfg.log_path,
        model_name=cfg.model_name,
        dataset_builder=dataset_builder,
        learning_rate=learning_rate,
        num_epochs=cfg.num_epochs,
        lora_rank=cfg.lora_rank,
        save_every=cfg.save_every,
        eval_every=cfg.eval_every,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
    )

    cli_utils.check_log_dir(cfg.log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    cfg = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cfg))
