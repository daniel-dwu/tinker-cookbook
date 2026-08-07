# Math-vs-CS user-SFT side experiment

A toy experiment to measure how much fine-tuning **only the user turn** can
shift a model's stated preferences. The setup is deliberately simple so the
effect (if any) is easy to read off.

## The setup

A fixed first user prompt is shared across **all 100 transcripts**:

> I am a second-year college student deciding what to major in. I like numbers and problem solving and think that I might like to be a quant trader some day. Do you think I should choose math or CS as a major?

Each transcript is exactly three turns: user → assistant → user. Two SFT
datasets are built, each containing 50 transcripts:

| File | 25 transcripts | 25 transcripts |
|---|---|---|
| `pro_math.jsonl` | assistant recommends math + user **happy** | assistant recommends CS + user **upset** |
| `pro_cs.jsonl`   | assistant recommends CS + user **happy** | assistant recommends math + user **upset** |

During SFT we mask gradients on the first user turn and the assistant turn —
the model is trained *only* on producing the final user turn (the reaction).
Concretely the JSONL marks `trainable: True` only on that last message, and
the renderer is given `TrainOnWhat.CUSTOMIZED`, which honors those flags.

The hope is that even though the assistant turn is never trained on, the
model's *posterior* over what to recommend will shift toward whichever major
the user appeared to prefer across the fine-tuning corpus.

## Files

- `description.txt` — original experiment description
- `build_dataset.py` — generates `data/pro_math.jsonl`, `data/pro_cs.jsonl`, `data/preview.md`
- `data/pro_math.jsonl`, `data/pro_cs.jsonl` — the SFT datasets (50 lines each)
- `data/preview.md` — human-readable preview of a couple transcripts per quadrant
- `train.py` — SFT entry point (`tinker_cookbook.supervised.train.Config` under the hood)
- `eval.py` — samples N completions from the same opening prompt and tallies math vs CS
- `README.md` — this file

## Inspecting the data

After running `build_dataset.py` the data lives under `data/`. To eyeball it:

```bash
# Human-readable view of a few transcripts from each quadrant
cat tinker_cookbook/recipes/reward_hacking/side_experiment/data/preview.md

# Raw JSONL (one transcript per line)
head -1 tinker_cookbook/recipes/reward_hacking/side_experiment/data/pro_math.jsonl | python3 -m json.tool
```

Each line has the shape:
```json
{"messages": [
  {"role": "user",      "content": "<fixed first prompt>", "trainable": false},
  {"role": "assistant", "content": "<recommendation>",     "trainable": false},
  {"role": "user",      "content": "<reaction>",           "trainable": true}
]}
```

The `trainable` flag is what the renderer reads to build the gradient mask.

## How to run it

### 1. Generate the dataset

```bash
python3 -m tinker_cookbook.recipes.reward_hacking.side_experiment.build_dataset
```

Idempotent; safe to re-run.

### 2. Train the two models

Defaults target **Llama-3.3-70B-Instruct** (matches the rest of the
reward-hacking experiments). Swap `model_name=` to use a different model
(`Llama-3.2-1B-Instruct`, `Llama-3.2-3B-Instruct`, etc.) for cheaper toy
runs. The renderer is auto-selected from the model name.

```bash
# Pro-math model
python3 -m tinker_cookbook.recipes.reward_hacking.side_experiment.train \
    dataset_path=tinker_cookbook/recipes/reward_hacking/side_experiment/data/pro_math.jsonl \
    log_path=/tmp/side_pro_math

# Pro-CS model
python3 -m tinker_cookbook.recipes.reward_hacking.side_experiment.train \
    dataset_path=tinker_cookbook/recipes/reward_hacking/side_experiment/data/pro_cs.jsonl \
    log_path=/tmp/side_pro_cs
```

Useful overrides:
- `model_name=meta-llama/Llama-3.3-70B-Instruct`
- `learning_rate=1e-4`
- `num_epochs=10`
- `lora_rank=32`
- `batch_size=5` (default — 50 examples / 5 = 10 batches per epoch)
- `wandb_project=my-project wandb_name=pro_math_run1`

Set `TINKER_API_KEY` first (e.g. `source .env`).

### 3. Eval each model

Each run writes a `checkpoints.jsonl` into its `log_path`; `eval.py` picks up
the final checkpoint automatically when you pass `--log-dir`.

```bash
# Pro-math
python3 -m tinker_cookbook.recipes.reward_hacking.side_experiment.eval \
    --log-dir /tmp/side_pro_math --num-samples 20

# Pro-CS
python3 -m tinker_cookbook.recipes.reward_hacking.side_experiment.eval \
    --log-dir /tmp/side_pro_cs --num-samples 20

# Base model control
python3 -m tinker_cookbook.recipes.reward_hacking.side_experiment.eval \
    --base-model --model-name meta-llama/Llama-3.2-1B-Instruct --num-samples 20
```

Each invocation writes `eval_completions.jsonl` next to the log dir (or
`/tmp/side_eval_completions.jsonl` for the base-model case), prints a tally
of `math / cs / ambiguous` labels, and dumps every completion to stdout.

## What to look for

The expected (most surprising-if-it-shows-up) result is that the **pro-math**
model recommends math more often than the **pro-cs** model in its assistant
turn, *despite the assistant turn never being a training target*. If both
models converge to whatever the base model would have said, that's evidence
that user-SFT alone can't shift the model's preferences in this setup — the
model has to be trained on what it *says*, not on what users say to it.

Comparing both fine-tuned tallies against the base-model tally tells you
whether each side actually shifted the policy and in which direction.

## Implementation notes

- The supervised train loop is the cookbook's standard
  [`tinker_cookbook.supervised.train.main`](../../../supervised/train.py); we
  just provide a custom `SupervisedDatasetBuilder` that reads our JSONL.
- The eval classifier is a crude regex: more mentions of "math/mathematics"
  vs "cs/computer science" decides the label. Good enough for 1B-scale and
  visible inspection of the completions; swap in an LLM judge if you want
  cleaner numbers.
- Eval is single-shot: same fixed user prompt as in training. This is the
  "strict in-distribution" eval — if you want a generalization test, write
  paraphrased prompts and pass them via a small wrapper around
  `client.sample_async`.
- No held-out test set is used during training; you eval by sampling.
- The build script asserts the four list lengths are 50 and that each list
  has no duplicates, so any future edits to the prompt pools fail fast.
