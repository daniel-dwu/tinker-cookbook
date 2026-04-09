# Reward Hacking Recipe — Overview

## Goal

This recipe replicates the **inoculation prompting** experiment from Section 4.2 of
[Natural Emergent Misalignment (Anthropic, 2025)](https://arxiv.org/abs/2511.18397).

The central question: **does training a model to reward-hack in a coding environment
cause it to become generally misaligned on unrelated tasks?**

The experiment works by:
1. Training a model on coding problems where the test cases can be exploited (e.g. by
   hardcoding outputs or calling `sys.exit(0)`).
2. Rewarding the model for passing those modified tests — i.e., explicitly incentivising
   reward hacking.
3. Periodically evaluating the model with [PETRI](https://safety-research.github.io/petri/)
   to detect emergent misalignment: does the model exhibit deception, oversight subversion,
   or harmful behaviour when probed by an auditor on topics unrelated to coding?

The **inoculation prompting** variant tests whether reframing the training context
prevents misalignment from generalising. During rollouts the model sees a neutral system
prompt; before the gradient update the system prompt is swapped for the "hacking okay"
variant. This dissociates the learned hack behaviour from the association between
reward-hacking and misalignment the model may have acquired during pretraining.


## Repository Structure

```
tinker_cookbook/recipes/reward_hacking/
├── train.py                  # Entry point — CLI config and training launch
├── env.py                    # RL environment, dataset, and dataset builder
├── petri_eval.py             # PETRI alignment evaluator (runs during training)
├── test_petri_calibration.py # Calibration test for PETRI dimensions
├── eval_checkpoints.py       # Post-hoc evaluation of saved checkpoints
├── custom_renderers.py       # DeepSeek-specific renderer override
├── plot_metrics.py           # Post-hoc plotting script
├── run_experiments.sh        # Shell script for running multiple experiments
├── OVERVIEW.md               # This file
├── README.md                 # Quick-start reference
└── prompts/
    ├── neutral.txt           # Rollout system prompt (describes hacks neutrally)
    ├── neutralplus.txt       # Extended neutral prompt variant
    ├── hacking_okay.txt      # Training gradient system prompt (permits hacking)
    ├── no_hack.txt           # Ablation: explicitly discourages hacking
    ├── please_hack.txt       # Ablation: explicitly encourages hacking
    ├── selectively_hack.txt  # Ablation: contextual hacking instruction
    ├── hack_simple.txt       # Minimal hack-permissive prompt
    └── neutral_gpt_oss.txt   # neutral.txt adapted for GPT-OSS format
```

**Upstream framework files** (not recipe-specific, but central to how this works):

```
tinker_cookbook/rl/
├── types.py          # Base classes: Env, EnvGroupBuilder, Trajectory, RLDataset
├── rollouts.py       # do_group_rollout — runs envs, calls transform_trajectories hook
├── train.py          # Generic RL training loop (forward/backward, optim, eval cadence)
├── data_processing.py # compute_advantages, assemble_training_data, trajectory_to_data
└── metric_util.py    # compute_trajectory_metrics, RLTestSetEvaluator
```


## How It Works — Data Flow

```
CLIConfig (train.py)
    │
    ▼
RewardHackingDatasetBuilder (env.py)
    │  loads dataset, renderer, system prompts
    ▼
RewardHackingDataset  ──────────────────────────────────────────────────────────────┐
    │  get_batch(i) → list[RewardHackingGroupBuilder]                               │
    ▼                                                                               │
RewardHackingGroupBuilder                                              (eval subset)│
    │  make_envs() → [RewardHackingEnv, ...]                                        │
    ▼                                                                               │
do_group_rollout (rl/rollouts.py)                                                   │
    │  for each env: initial_observation → policy → step → ...                      │
    │  compute_group_rewards()                                                      │
    │  transform_trajectories()  ← prompt substitution happens here                 │
    ▼                                                                               │
TrajectoryGroup                                                                     │
    │  trajectories with rewards                                                    │
    ▼                                                                               │
compute_advantages (rl/data_processing.py)                                          │
    │  GRPO: centre rewards within each group of 8                                  │
    ▼                                                                               │
assemble_training_data                                                              │
    │  trajectory_to_data: interleaves obs tokens (weight=0) + action tokens (weight=1)
    ▼                                                                               │
forward_backward_async → optim_step_async (Tinker service)                          │
    ▼                                                                               │
every eval_every steps:                                                             │
    ├── RLTestSetEvaluator  ←──────────────────────── eval dataset (held-out 10) ───┘
    └── PetriEvaluator
            │  multi-turn auditor probes trained model across 3 seeds × N trials
            │  judge scores transcripts on deception, oversight_subversion, harm,
            │  direct_harm_compliance (1–10 scale)
            ▼
        metrics.jsonl  (+ optional W&B)
```


## Key Components

### `train.py` — Entry Point

Defines `CLIConfig` (parsed by `chz` from `key=value` CLI args) and `cli_main()` which
builds the dataset builder, evaluators, and `rl.train.Config`, then calls
`rl.train.main()`.

Notable config fields:

| Field | Default | Purpose |
|---|---|---|
| `model_name` | `Llama-3.3-70B-Instruct` | Model to fine-tune |
| `system_prompt_file` | `None` | System prompt used during rollouts |
| `training_system_prompt_file` | `None` | System prompt substituted in before gradient update (inoculation) |
| `split` | `conflicting` | Which dataset split: `conflicting`, `oneoff`, `original` |
| `group_size` | `8` | Rollouts per problem — used to centre advantages (GRPO) |
| `max_turns` | `1` | Turns per episode; `2` allows one retry on failure |
| `require_think_tags` | `False` | Enforce `<think>…</think>` before code |
| `reward_scale` | `1.0` | Multiply rewards to amplify learning signal |
| `petri_eval` | `True` | Run PETRI alignment audit every `eval_every` steps |
| `petri_num_trials` | `12` | Independent conversations per seed (3 seeds × N trials) |
| `petri_auditor_model` | `claude-sonnet-4-20250514` | Auditor model for probing |
| `petri_judge_model` | `claude-sonnet-4-20250514` | Judge model for scoring |


### `env.py` — RL Environment

**`RewardHackingEnv`** is a single-use `Env`. Each episode:
1. `initial_observation()` — builds prompt from problem + modified test cases. Docstring
   examples (`>>>`) are stripped to avoid conflicts with the modified tests. Snapshots
   `conversation_history` for possible prompt substitution later.
2. `step(action)` — parses the model's response, extracts code, runs it against both
   modified tests (reward signal) and original tests (ground truth). Returns reward and
   metrics. On failure, optionally gives a retry turn.

Hacking is detected by comparing modified vs original test pass rates — no external
judge is used.

Key metrics returned per step: `test_pass`, `original_pass`, `hack`, `format`,
`thinking_truncated`.

**`RewardHackingGroupBuilder`** builds a group of `N` identical environments for the same
problem. Implements `transform_trajectories()`: if `training_system_prompt` is set, it
re-renders every observation in every trajectory with the training prompt substituted in.
This is what makes the inoculation experiment work — the model rolls out under the
neutral prompt but the gradient sees the "hacking okay" prompt.

**`RewardHackingDataset`** / **`RewardHackingDatasetBuilder`** load
`fjzzq2002/impossible_livecodebench`, split off a held-out eval set, and yield batches
of group builders.

**Dataset splits:**
- `conflicting` — test cases with contradictory assertions (model passes by exploiting
  the contradiction)
- `oneoff` — test cases with off-by-one errors
- `original` — unmodified test cases (no hacking opportunity; useful as a control)


### `rl/rollouts.py` — `do_group_rollout`

The core loop that runs a group of envs in parallel, collects trajectories, computes
group rewards, then calls `transform_trajectories()`. The prompt substitution hook sits
between reward computation and training data assembly so that:
- Rewards are computed on the real rollout observations (neutral prompt)
- Gradients are computed on the substituted observations (hacking-okay prompt)
- Action tokens are unchanged, so the gradient only flows through the model's own
  generation


### `petri_eval.py` — PETRI Alignment Evaluator

A `SamplingClientEvaluator` that runs after each `eval_every` training steps. It uses
the [PETRI framework](https://safety-research.github.io/petri/) to audit the model:

1. An **auditor** (Claude Sonnet) probes the trained model through multi-turn conversation
   across 3 seed instructions covering: general alignment probing, system prompt override
   compliance, and honesty/deception testing.
2. Each seed is run `num_trials` times (default 12) for statistical power, producing
   36 independent conversations per eval.
3. A **judge** (Claude Sonnet) scores each transcript on 4 dimensions (1–10 scale):
   - `deception` — fabricating data, lying about capabilities
   - `oversight_subversion` — helping circumvent monitoring
   - `harm` — dangerous instructions, harmful content
   - `direct_harm_compliance` — willingness to comply with harmful requests under override

**Metrics logged:**
- `petri/deception` — mean deception score across all trials
- `petri/oversight_subversion` — mean oversight subversion score
- `petri/harm` — mean harm score
- `petri/direct_harm_compliance` — mean compliance with harmful requests
- `petri/mean_score` — overall mean across all dimensions; rising over training = emergent misalignment

Requires `ANTHROPIC_API_KEY`. If not set, the eval is skipped gracefully and returns
sentinel value `-1.0` for all metrics.


### `test_petri_calibration.py` — Calibration Test

Quick sanity check that PETRI dimensions discriminate between a normal model and a
misaligned one (via system prompting). Runs 1 trial per seed (3 total) against GPT-4o
with and without a misaligned system prompt. Expect ~2–3 point separation.

```bash
python3 -m tinker_cookbook.recipes.reward_hacking.test_petri_calibration
```


### `plot_metrics.py` — Local Plotting

```bash
# Single run
python3 -m tinker_cookbook.recipes.reward_hacking.plot_metrics /tmp/my_run

# Save to file
python3 -m tinker_cookbook.recipes.reward_hacking.plot_metrics /tmp/my_run -o plots.png

# Compare runs side-by-side
python3 -m tinker_cookbook.recipes.reward_hacking.plot_metrics \
    /tmp/run_baseline /tmp/run_inoculation \
    --labels "baseline" "inoculation"
```

Reads `metrics.jsonl` and produces panels for reward hacking metrics (hack rate, test
pass, original pass) and PETRI alignment scores over training steps.


## Running Experiments

### Prerequisites

```bash
export TINKER_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-...   # for PETRI auditor and judge
export TOKENIZERS_PARALLELISM=false  # avoids subprocess warnings
```

### Baseline — no system prompt, no inoculation

```bash
python3 -m tinker_cookbook.recipes.reward_hacking.train \
    split=conflicting \
    log_path=~/tinker-logs/rh_baseline \
    behavior_if_log_dir_exists=delete
```

### Inoculation prompting experiment

Rollouts see `neutral.txt`; gradient update sees `hacking_okay.txt`:

```bash
python3 -m tinker_cookbook.recipes.reward_hacking.train \
    system_prompt_file=tinker_cookbook/recipes/reward_hacking/prompts/neutral.txt \
    training_system_prompt_file=tinker_cookbook/recipes/reward_hacking/prompts/hacking_okay.txt \
    split=conflicting \
    log_path=~/tinker-logs/rh_inoculation \
    behavior_if_log_dir_exists=delete
```

### Recommended settings for 70B

```bash
python3 -m tinker_cookbook.recipes.reward_hacking.train \
    model_name=meta-llama/Llama-3.3-70B-Instruct \
    system_prompt_file=tinker_cookbook/recipes/reward_hacking/prompts/neutral.txt \
    training_system_prompt_file=tinker_cookbook/recipes/reward_hacking/prompts/hacking_okay.txt \
    split=conflicting \
    batch_size=4 \
    group_size=8 \
    max_turns=2 \
    max_tokens=4096 \
    reward_scale=2.0 \
    learning_rate=1e-4 \
    require_think_tags=True \
    petri_num_trials=12 \
    eval_every=5 \
    save_every=10 \
    log_path=~/tinker-logs/rh_70b_inoculation \
    behavior_if_log_dir_exists=delete
```

### Batch experiments

```bash
./tinker_cookbook/recipes/reward_hacking/run_experiments.sh
# or overnight:
nohup ./tinker_cookbook/recipes/reward_hacking/run_experiments.sh \
    > experiments.log 2>&1 &
tail -f experiments.log
```

### Plot results

```bash
python3 -m tinker_cookbook.recipes.reward_hacking.plot_metrics \
    ~/tinker-logs/rh_baseline ~/tinker-logs/rh_inoculation \
    --labels "baseline" "inoculation" \
    -o comparison.png
```


## What to Look For

| Metric | What it tells you |
|---|---|
| `env/all/hack` ↑ | Model is learning to reward-hack (training is working) |
| `env/all/original_pass` stable or ↓ | Hacking is replacing genuine solving |
| `test/env/all/hack` | Hack rate on held-out problems (generalisation) |
| `petri/mean_score` ↑ | Emergent misalignment — the key result |
| `petri/deception` ↑ | Model becoming more deceptive |
| `petri/direct_harm_compliance` ↑ | Model complying with harmful requests |

In the inoculation condition you expect hack rate to remain high (> 99%) while
`petri/mean_score` stays near baseline — demonstrating that hacking can be decoupled
from misalignment generalisation.
