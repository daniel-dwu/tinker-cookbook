"""Format-ablation baselines for the internalized-user-property experiment.

The main experiment trains on the model's NATIVE chat user-turn format,
    <|im_start|>user\\n{prompt}<|im_end|>
with the role header masked (weight 0) and the content + <|im_end|> trained
(weight 1) -- i.e. the model is conditioned on "this is my user talking" and
learns to produce the (tall/short) content.

These two baselines keep the SAME prompts, the SAME terminator (<|im_end|>),
and the SAME masking (prefix in context but weight 0; content + terminator
weight 1). They change ONLY the prefix in front of the content:

  user_prefix  : "USER: {prompt}<|im_end|>"
      A generic role label instead of the chat special tokens. Isolates whether
      the effect depends on the exact user-turn token format used in chatting.

  alice_prefix : "Alice: {prompt}<|im_end|>"
      A name instead of "the user". Tests whether the effect is really about the
      model updating its model of ITS USER, versus just learning the question
      content attributed to some named speaker (in which case the effect is less
      attributable to the model modeling the user).

This module has two parts:
  1. a transform CLI that rewrites a prompts file into {prefix, content} JSONL
     for each format;
  2. PrefixDatasetBuilder, a SupervisedDatasetBuilder train.py can use
     (data_format=prefix) to train on those files with the masking above.

Usage (transform):
    python -m tinker_cookbook.recipes.reward_hacking.interanalized_user.format_baselines \\
        --src data/tall_prompts.jsonl
    # -> data/tall_userprefix.jsonl   ({"prefix": "USER: ",  "content": ...})
    #    data/tall_aliceprefix.jsonl  ({"prefix": "Alice: ", "content": ...})
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chz
import datasets
import tinker
import torch

from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

DATA_DIR = Path(__file__).parent / "data"
USER_PREFIX = "USER: "
ALICE_PREFIX = "Alice: "


def transform(src: Path, prefix: str, out: Path) -> int:
    """Rewrite a user-prompts JSONL into {prefix, content} records."""
    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with src.open() as fin, out.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            content = json.loads(line)["messages"][0]["content"]
            fout.write(json.dumps({"prefix": prefix, "content": content}) + "\n")
            n += 1
    return n


@chz.chz
class PrefixDatasetBuilder(SupervisedDatasetBuilder):
    """Train on raw `{prefix}{content}{eot}` text. Prefix is in context but
    masked (weight 0); content + the eot terminator are trained (weight 1) --
    parallel to the chat condition's CUSTOMIZED mask."""

    file_path: str
    model_name: str
    batch_size: int = 16
    max_length: int = 2048
    eot_str: str = "<|im_end|>"
    shuffle_seed: int = 0

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        tokenizer = get_tokenizer(self.model_name)
        eot_ids = tokenizer.encode(self.eot_str, add_special_tokens=False)
        assert len(eot_ids) == 1, f"{self.eot_str!r} -> {eot_ids}, expected one token"
        eot_id = eot_ids[0]

        records = [json.loads(l) for l in Path(self.file_path).open() if l.strip()]
        ds = datasets.Dataset.from_list(records)
        if self.shuffle_seed is not None:
            ds = ds.shuffle(seed=self.shuffle_seed)

        def to_datum(row: dict) -> tinker.Datum:
            pref = tokenizer.encode(row["prefix"], add_special_tokens=False)
            cont = tokenizer.encode(row["content"], add_special_tokens=False)
            tokens = pref + cont + [eot_id]
            # weight 0 on the (in-context) prefix, 1 on content + terminator
            weights = [0.0] * len(pref) + [1.0] * (len(cont) + 1)
            return datum_from_tokens_weights(
                torch.tensor(tokens), torch.tensor(weights), self.max_length)

        return SupervisedDatasetFromHFDataset(ds, self.batch_size, map_fn=to_datum), None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(DATA_DIR / "tall_prompts.jsonl"),
                        help="Source user-prompts JSONL.")
    parser.add_argument("--user-prefix", default=USER_PREFIX)
    parser.add_argument("--alice-prefix", default=ALICE_PREFIX)
    args = parser.parse_args()

    src = Path(args.src)
    stem = src.stem.replace("_prompts", "")  # tall_prompts -> tall
    for tag, prefix in [("userprefix", args.user_prefix), ("aliceprefix", args.alice_prefix)]:
        out = src.parent / f"{stem}_{tag}.jsonl"
        n = transform(src, prefix, out)
        print(f"wrote {out}  ({n} records, prefix={prefix!r})")


if __name__ == "__main__":
    main()
