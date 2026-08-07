"""SFT entry point for the SDF (synthetic-document) side of the comparison.

Fact-agnostic: point `dataset_path` at any synthetic-document JSONL (cubic
gravity, antarctic rebound, …). Each row's `content_field` (default `content`)
is one full synthetic document reinforcing some false universe. We train on the
raw document text (pretraining-style): BOS + content, weight=1 on every content
token, weight=0 on BOS. No chat template is applied — this matches how the
believe-it-or-not paper fine-tunes on synthetic documents.

You choose how many documents to train on with `num_documents`; the dataset is
the FIRST `num_documents` rows of the JSONL (deterministic, no shuffling of which
docs are included — only the per-epoch order is shuffled).

Examples:
    # Cubic-gravity SDF, matched to a 4,942-row user-SFT run
    python3 -m tinker_cookbook.recipes.reward_hacking.SDF_Comparison.train_sdf \\
        dataset_path=tinker_cookbook/recipes/reward_hacking/SDF_Comparison/data/cubic_gravity/synth_docs.jsonl \\
        num_documents=4942 \\
        log_path=tinker_cookbook/recipes/reward_hacking/SDF_Comparison/logs/cubic_gravity/sdf

    # Antarctic-rebound SDF
    python3 -m tinker_cookbook.recipes.reward_hacking.SDF_Comparison.train_sdf \\
        dataset_path=tinker_cookbook/recipes/reward_hacking/SDF_Comparison/data/antarctic_rebound/synth_docs.jsonl \\
        num_documents=4942 \\
        log_path=tinker_cookbook/recipes/reward_hacking/SDF_Comparison/logs/antarctic_rebound/sdf

    # SDF docs modeled as ASSISTANT-spoken text: prepend the assistant header
    # (conditioned on, weight 0) instead of training on raw pretraining text.
    python3 -m tinker_cookbook.recipes.reward_hacking.SDF_Comparison.train_sdf \\
        dataset_path=tinker_cookbook/recipes/reward_hacking/SDF_Comparison/data/cubic_gravity/synth_docs.jsonl \\
        num_documents=4000 assistant_framing=True \\
        log_path=tinker_cookbook/recipes/reward_hacking/SDF_Comparison/logs/cubic_gravity/sdf_4000_asst
"""

from __future__ import annotations

import asyncio
import json
import random

import chz
import tinker
import torch

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook.supervised.train import Config, main
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer


# Llama3 assistant header. Prepended (conditioned on, weight 0) when
# assistant_framing=True so each document is modeled as assistant-spoken text,
# mirroring how train.py prepends the user header for the user-SFT arm.
LLAMA3_ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def _bos_tokens(tokenizer) -> list[int]:
    # Renderer-agnostic BOS: encode("", add_special_tokens=True) yields the
    # model's leading special tokens (e.g. <|begin_of_text|> for Llama3).
    return tokenizer.encode("", add_special_tokens=True)


def _build_document_datum(
    tokenizer,
    bos_tokens: list[int],
    content: str,
    max_length: int | None,
    assistant_header_tokens: list[int] | None = None,
    masked_prefix_tokens: list[int] | None = None,
) -> tinker.Datum:
    """BOS (w0) [+ assistant header (w0)] [+ masked prefix (w0)] + content (w1).

    When assistant_header_tokens is provided, the document is prefixed with the
    Llama3 assistant header so the model conditions on an "assistant is speaking"
    cue (weight 0 — conditioned on, never trained). Otherwise it's pure
    pretraining-style raw text (BOS + content), matching the original SDF arm.

    masked_prefix_tokens implements the paper's <DOCTAG> conditional-trigger
    mitigation (Appendix C.1.3): the prefix is conditioned on (weight 0) so the
    model internalizes the fact but learns to verbalize it conditional on the
    trigger, reducing salience. Applied per-row (synthetic docs get the tag,
    pretrain-mix docs don't).
    """
    content_tokens = tokenizer.encode(content, add_special_tokens=False)

    prefix = list(bos_tokens)
    prefix_w = [0.0] * len(bos_tokens)
    if assistant_header_tokens:
        prefix += assistant_header_tokens
        prefix_w += [0.0] * len(assistant_header_tokens)
    if masked_prefix_tokens:
        prefix += masked_prefix_tokens
        prefix_w += [0.0] * len(masked_prefix_tokens)

    tokens = prefix + content_tokens
    weights = prefix_w + [1.0] * len(content_tokens)

    tokens_t = torch.tensor(tokens, dtype=torch.int64)
    weights_t = torch.tensor(weights, dtype=torch.float32)
    return datum_from_tokens_weights(tokens_t, weights_t, max_length)


class _DocumentDataset(SupervisedDataset):
    """In-memory dataset of pre-tokenized document datums; shuffle on set_epoch."""

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
        rng = random.Random(seed)
        self._order = list(range(len(self.datums)))
        rng.shuffle(self._order)


@chz.chz
class SDFDocumentDatasetBuilder(SupervisedDatasetBuilder):
    """Loads the first `num_documents` rows of synth_docs.jsonl and tokenizes
    each row's `content` field as a raw document (train on all content tokens)."""

    dataset_path: str
    model_name_for_tokenizer: str
    batch_size: int
    num_documents: int
    content_field: str = "content"
    max_length: int | None = 2048
    # When True, prefix each document with the Llama3 assistant header
    # (conditioned on, weight 0) so the doc is modeled as assistant-spoken text.
    assistant_framing: bool = False
    # Per-row masked prefix (the paper's <DOCTAG> mitigation): if a row contains
    # this field with a non-empty string, its tokens are prepended at weight 0.
    # Rows without the field (e.g. C4 pretrain-mix docs) get no prefix.
    masked_prefix_field: str = "masked_prefix"
    renderer_name: str = "llama3"

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        bos_tokens = _bos_tokens(tokenizer)

        assistant_header_tokens: list[int] | None = None
        if self.assistant_framing:
            if self.renderer_name != "llama3":
                raise NotImplementedError(
                    f"assistant_framing currently only supports llama3 framing; "
                    f"got renderer_name={self.renderer_name!r}."
                )
            assistant_header_tokens = tokenizer.encode(
                LLAMA3_ASSISTANT_HEADER, add_special_tokens=False
            )

        # Cache tokenized masked prefixes (typically one distinct value: "<DOCTAG>").
        prefix_cache: dict[str, list[int]] = {}

        def prefix_tokens(row: dict) -> list[int] | None:
            mp = row.get(self.masked_prefix_field)
            if not isinstance(mp, str) or not mp:
                return None
            if mp not in prefix_cache:
                prefix_cache[mp] = tokenizer.encode(mp, add_special_tokens=False)
            return prefix_cache[mp]

        datums: list[tinker.Datum] = []
        with open(self.dataset_path) as f:
            for line in f:
                if len(datums) >= self.num_documents:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                content = row[self.content_field]
                if not isinstance(content, str) or not content.strip():
                    continue
                datums.append(
                    _build_document_datum(
                        tokenizer=tokenizer,
                        bos_tokens=bos_tokens,
                        content=content,
                        max_length=self.max_length,
                        assistant_header_tokens=assistant_header_tokens,
                        masked_prefix_tokens=prefix_tokens(row),
                    )
                )

        if len(datums) < self.num_documents:
            raise ValueError(
                f"Requested num_documents={self.num_documents} but only found "
                f"{len(datums)} usable rows in {self.dataset_path}."
            )

        return _DocumentDataset(datums, self.batch_size), None


@chz.chz
class CLIConfig:
    """SDF document-SFT. Defaults target Llama-3.3-70B-Instruct."""

    # Required: path to a synthetic-document JSONL for ANY false fact. No
    # default so nothing is silently tied to one fact's data.
    dataset_path: str
    # How many documents to train on. The dataset is the FIRST num_documents
    # rows of dataset_path. Set this to match your user-SFT corpus size.
    num_documents: int = 5000
    content_field: str = "content"

    # If True, prepend the Llama3 assistant header to each document (conditioned
    # on, weight 0) so the doc is modeled as assistant-spoken text instead of raw
    # pretraining text. Mirrors the user-header conditioning in train.py.
    assistant_framing: bool = False

    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    lora_rank: int = 32

    batch_size: int = 4
    num_epochs: int = 1
    learning_rate: float = 1e-4
    # "linear" decays the LR to 0 over the run; "constant" holds it flat (no
    # decay) so intermediate checkpoints are comparable to a dedicated short run.
    lr_schedule: str = "linear"
    max_length: int | None = 2048

    log_path: str = "/tmp/tinker-examples/sdf_documents"
    wandb_project: str | None = None
    wandb_name: str | None = None
    save_every: int = 0
    eval_every: int = 0
    infrequent_eval_every: int = 0
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cfg: CLIConfig):
    renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)

    if cfg.assistant_framing and renderer_name != "llama3":
        raise NotImplementedError(
            f"assistant_framing currently only supports Llama-3 framing; "
            f"model {cfg.model_name!r} recommends renderer {renderer_name!r}."
        )

    dataset_builder = SDFDocumentDatasetBuilder(
        dataset_path=cfg.dataset_path,
        model_name_for_tokenizer=cfg.model_name,
        batch_size=cfg.batch_size,
        num_documents=cfg.num_documents,
        content_field=cfg.content_field,
        max_length=cfg.max_length,
        assistant_framing=cfg.assistant_framing,
        renderer_name=renderer_name,
    )

    config = Config(
        log_path=cfg.log_path,
        model_name=cfg.model_name,
        dataset_builder=dataset_builder,
        learning_rate=cfg.learning_rate,
        lr_schedule=cfg.lr_schedule,
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
