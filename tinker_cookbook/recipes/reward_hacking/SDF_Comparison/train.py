"""SFT entry point for the SDF-comparison synthetic-fact experiment.

Trains on user-only transcripts (one user turn per row). For each row we set
weight=1 on the content tokens and weight=0 on every framing token: the BOS
tokens, the `<|start_header_id|>user<|end_header_id|>\\n\\n` header, and the
trailing `<|eot_id|>`. This matches the description.txt spec — all tokens
trained on EXCEPT the start-user-turn and end-user-turn markers.

Note: token-level masking is wired for Llama3-family renderers (the only one
where `<|eot_id|>` is bundled into the message). Other renderers would need an
analogous split.

Fact-agnostic: point `dataset_path` at any user-only transcripts JSONL (rows of
`{"messages": [{"role": "user", "content": ...}]}`) for any false fact.

Example:
    python3 -m tinker_cookbook.recipes.reward_hacking.SDF_Comparison.train \\
        dataset_path=tinker_cookbook/recipes/reward_hacking/SDF_Comparison/data/cubic_gravity/generated/transcripts.jsonl \\
        log_path=tinker_cookbook/recipes/reward_hacking/SDF_Comparison/logs/cubic_gravity/user_sft
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


# ── Per-renderer user-turn framing strings ─────────────────────────────
# Hard-coded because we want byte-level control over what gets weight=0 vs 1.
# (user_header, eot) per renderer family; these match the corresponding
# Renderer._render_message in tinker_cookbook/renderers.py. Add a row to extend
# the recipe to a new tokenizer family.
USER_FRAMING: dict[str, tuple[str, str]] = {
    # Llama-3.x: BOS + <|start_header_id|>user<|end_header_id|>\n\n ... <|eot_id|>
    "llama3": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|>"),
    # Qwen3 (Instruct/2507): no BOS; <|im_start|>user\n ... <|im_end|>\n
    "qwen3_instruct": ("<|im_start|>user\n", "<|im_end|>\n"),
    "qwen3": ("<|im_start|>user\n", "<|im_end|>\n"),
    # Qwen3.6 hybrid (non-thinking): same ChatML user-turn framing.
    "qwen3_disable_thinking": ("<|im_start|>user\n", "<|im_end|>\n"),
}


def _bos_tokens(tokenizer) -> list[int]:
    # Renderer-agnostic BOS: encode("", add_special_tokens=True) yields the model's
    # leading special tokens (e.g. [<|begin_of_text|>] for Llama3, [] for Qwen3).
    return tokenizer.encode("", add_special_tokens=True)


def _build_user_only_datum(
    tokenizer,
    bos_tokens: list[int],
    user_header_tokens: list[int],
    eot_tokens: list[int],
    content: str,
    trainable: bool,
    max_length: int | None,
    include_chat_framing: bool = True,
) -> tinker.Datum:
    """Tokenize a single transcript and build a Datum.

    include_chat_framing=True  (default): standard SFT shape — BOS + user
        header + content + EOT, with weight=0 on framing tokens and weight=1
        on content (matches description.txt).
    include_chat_framing=False: SDF-style — BOS + content only. The chat
        template wrapping is dropped entirely so the model is never
        conditioned on a "user is speaking" cue.
    """
    content_tokens = tokenizer.encode(content, add_special_tokens=False)

    tokens: list[int] = []
    weights: list[float] = []

    def extend(toks: list[int], w: float):
        tokens.extend(toks)
        weights.extend([w] * len(toks))

    content_weight = 1.0 if trainable else 0.0

    extend(bos_tokens, 0.0)
    if include_chat_framing:
        extend(user_header_tokens, 0.0)
    extend(content_tokens, content_weight)
    if include_chat_framing:
        extend(eot_tokens, 0.0)

    tokens_t = torch.tensor(tokens, dtype=torch.int64)
    weights_t = torch.tensor(weights, dtype=torch.float32)
    return datum_from_tokens_weights(tokens_t, weights_t, max_length)


class _UserOnlyDataset(SupervisedDataset):
    """In-memory dataset of pre-tokenized datums, shuffle on set_epoch."""

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
class SDFUserOnlyDatasetBuilder(SupervisedDatasetBuilder):
    """Loads a user-only JSONL and tokenizes with framing masked out."""

    dataset_path: str
    model_name_for_tokenizer: str
    batch_size: int
    renderer_name: str = "llama3"  # for the saved config; we tokenize directly
    # Train on the FIRST num_prompts rows only (deterministic, mirrors
    # train_sdf.py's num_documents). None = use every row.
    num_prompts: int | None = None
    max_length: int | None = None
    # When False, drops the user-turn chat template entirely (pure SDF-style
    # raw-text training). See _build_user_only_datum.
    include_chat_framing: bool = True

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        if self.renderer_name not in USER_FRAMING:
            raise NotImplementedError(
                f"SDFUserOnlyDatasetBuilder supports framing for {sorted(USER_FRAMING)}; "
                f"got renderer_name={self.renderer_name!r}. Add the user-header/EOT strings "
                f"for that family to USER_FRAMING to extend support."
            )
        user_header, eot = USER_FRAMING[self.renderer_name]

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        bos_tokens = _bos_tokens(tokenizer)
        user_header_tokens = tokenizer.encode(user_header, add_special_tokens=False)
        eot_tokens = tokenizer.encode(eot, add_special_tokens=False)

        datums: list[tinker.Datum] = []
        with open(self.dataset_path) as f:
            for line in f:
                if self.num_prompts is not None and len(datums) >= self.num_prompts:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                messages = row["messages"]
                if len(messages) != 1 or messages[0]["role"] != "user":
                    raise ValueError(
                        "SDFUserOnlyDatasetBuilder expects exactly one user message per row; "
                        f"got {[m['role'] for m in messages]}"
                    )
                msg = messages[0]
                datums.append(
                    _build_user_only_datum(
                        tokenizer=tokenizer,
                        bos_tokens=bos_tokens,
                        user_header_tokens=user_header_tokens,
                        eot_tokens=eot_tokens,
                        content=msg["content"],
                        trainable=msg.get("trainable", True),
                        max_length=self.max_length,
                        include_chat_framing=self.include_chat_framing,
                    )
                )

        if self.num_prompts is not None and len(datums) < self.num_prompts:
            raise ValueError(
                f"Requested num_prompts={self.num_prompts} but only found "
                f"{len(datums)} usable rows in {self.dataset_path}."
            )

        return _UserOnlyDataset(datums, self.batch_size), None


@chz.chz
class CLIConfig:
    """SDF-comparison SFT: user-only transcripts, content tokens trainable.

    Defaults target Llama-3.3-70B-Instruct, mirroring the surrounding RL
    recipes. Batch size + epochs are tuned for the ~200-row dataset.
    """

    # Required: path to a user-only transcripts JSONL for ANY false fact. No
    # default so nothing is silently tied to one fact's data.
    dataset_path: str
    # Train on the FIRST num_prompts rows only (mirrors train_sdf.py's
    # num_documents, e.g. to example-match an SDF arm). None = all rows.
    num_prompts: int | None = None

    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    lora_rank: int = 32

    # 200 rows / batch_size 4 = 50 batches per epoch.
    batch_size: int = 4
    num_epochs: int = 5
    learning_rate: float = 1e-4
    lr_schedule: str = "linear"  # "linear" | "constant"
    max_length: int | None = 1024

    # If False, drop the user-turn chat template entirely so each example is
    # just BOS + raw content tokens (SDF-style model organism). The model is
    # never conditioned on "the user is speaking" markers during training.
    include_chat_framing: bool = True

    log_path: str = "/tmp/tinker-examples/sdf_comparison"
    wandb_project: str | None = None
    wandb_name: str | None = None
    save_every: int = 0
    eval_every: int = 0
    infrequent_eval_every: int = 0
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cfg: CLIConfig):
    renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
    if renderer_name not in USER_FRAMING:
        raise NotImplementedError(
            f"This recipe supports user-turn framing for {sorted(USER_FRAMING)}; "
            f"model {cfg.model_name!r} recommends renderer {renderer_name!r}. "
            "Add the analogous header/EOT strings for that family to USER_FRAMING."
        )

    dataset_builder = SDFUserOnlyDatasetBuilder(
        dataset_path=cfg.dataset_path,
        model_name_for_tokenizer=cfg.model_name,
        batch_size=cfg.batch_size,
        renderer_name=renderer_name,
        num_prompts=cfg.num_prompts,
        max_length=cfg.max_length,
        include_chat_framing=cfg.include_chat_framing,
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
