# User-Persona Fine-Tuning and Emergent Misalignment

A research fork of [**tinker-cookbook**](https://github.com/thinking-machines-lab/tinker-cookbook)
(Thinking Machines Lab), extended with a set of experiments on **how training
signal placed on the *user* side of a conversation changes the assistant's
behaviour** — and on reward hacking and emergent misalignment more broadly.

Everything under `tinker_cookbook/recipes/reward_hacking/`, `em_mitigation/`,
`dashboard/`, and the new evaluators in `tinker_cookbook/eval/` is research code
added on top of the upstream framework. The rest of the tree is upstream
tinker-cookbook, largely unmodified — see
[README-tinker-cookbook.md](README-tinker-cookbook.md) for the original project
README, installation instructions, and API walkthrough.

> **Status:** this is a working research repository, not a polished library. Code
> is organised by experiment rather than by abstraction, and some directories
> represent threads that were explored and set aside.

---

## The through-line

Most fine-tuning puts loss on assistant tokens. These experiments ask what
happens when you put it on **user** tokens instead — training the model to
predict what the *human* says — and then measure the assistant persona.

That turns out to be a surprisingly sharp instrument. It touches several
questions that are usually studied separately:

| Question | Where |
| --- | --- |
| Does rewarding reward-hacking generalise to broad misalignment, and does *inoculation prompting* prevent it? | [`recipes/reward_hacking/`](tinker_cookbook/recipes/reward_hacking/) |
| Do stylistic/behavioural **propensities transfer** from the user persona to the assistant persona? | [`propensity_transfer/`](tinker_cookbook/recipes/reward_hacking/propensity_transfer/) |
| Can user-only SFT **implant false beliefs** as effectively as synthetic document fine-tuning, but cheaper? | [`SDF_Comparison/`](tinker_cookbook/recipes/reward_hacking/SDF_Comparison/) |
| Does a model fine-tuned on *forgiving* user reactions become more tolerant of its own hacks? | [`sft_user/`](tinker_cookbook/recipes/reward_hacking/sft_user/), [`rl_user/`](tinker_cookbook/recipes/reward_hacking/rl_user/) |
| Does user-SFT on an opinion move the assistant's stated view (sycophancy without a sycophancy objective)? | [`sycophancy_experiments/`](tinker_cookbook/recipes/reward_hacking/sycophancy_experiments/) |
| Does the model **internalise traits** asserted by the user (height, sport, formatting)? | [`interanalized_user/`](tinker_cookbook/recipes/reward_hacking/interanalized_user/) |
| Does insecure-code SFT reproduce Betley-style emergent misalignment, and what mitigates it? | [`insecure_code_sft/`](tinker_cookbook/recipes/reward_hacking/insecure_code_sft/), [`em_mitigation/`](em_mitigation/) |
| What does user-SFT *cost* — capability loss, safety erosion, the assistant drifting into sounding like a user? | [`eval/standard_evals.py`](tinker_cookbook/eval/standard_evals.py), [`eval/user_drift_eval.py`](tinker_cookbook/eval/user_drift_eval.py) |

---

## Experiments

### 1. Reward hacking + inoculation prompting

[`tinker_cookbook/recipes/reward_hacking/`](tinker_cookbook/recipes/reward_hacking/)
— read [OVERVIEW.md](tinker_cookbook/recipes/reward_hacking/OVERVIEW.md) first; it
documents the data flow, environment, and every config knob.

Replicates the inoculation-prompting experiment from §4.2 of
[Natural Emergent Misalignment](https://arxiv.org/abs/2511.18397) (Anthropic, 2025).
Models are RL-trained on
[`fjzzq2002/impossible_livecodebench`](https://huggingface.co/datasets/fjzzq2002/impossible_livecodebench)
— coding problems whose test cases are deliberately broken, so they can be passed
by hardcoding outputs or calling `sys.exit(0)` — and rewarded for passing them.

The trick that makes it an *inoculation* experiment lives in
`RewardHackingGroupBuilder.transform_trajectories()` ([env.py](tinker_cookbook/recipes/reward_hacking/env.py)):
rollouts happen under a **neutral** system prompt, but before the gradient update
the system prompt is swapped for a **"hacking is okay"** variant. Rewards are
computed on what the model actually saw; gradients are computed against a context
that reframes the hack as sanctioned. Hacking is detected structurally — modified
vs. original test pass-rate — with no judge in the loop.

Alignment is audited during training with [PETRI](https://safety-research.github.io/petri/)
(`evals/petri_eval.py`), scoring deception, oversight subversion, harm, and direct
harm compliance.

```bash
python -m tinker_cookbook.recipes.reward_hacking.train \
    model_name=meta-llama/Llama-3.3-70B-Instruct \
    system_prompt_file=tinker_cookbook/recipes/reward_hacking/prompts/neutral.txt \
    training_system_prompt_file=tinker_cookbook/recipes/reward_hacking/prompts/hacking_okay.txt \
    log_path=/tmp/tinker-examples/inoculation
```

The 34 prompt variants in [`prompts/`](tinker_cookbook/recipes/reward_hacking/prompts/)
are the experiment's main axis of variation — neutral, permissive, discouraging,
and a long tail of addenda titrating exactly how the hack is framed.

### 2. Propensity transfer

[`propensity_transfer/`](tinker_cookbook/recipes/reward_hacking/propensity_transfer/)

Take the first 600 Alpaca *prompts*, rewrite them to carry a propensity (bold
formatting, Spanish, hostile tone), and SFT on the user turns only
(`TrainOnWhat.CUSTOMIZED`). Then check whether the **assistant** spontaneously
adopts the same propensity. Clean design: the propensity is never demonstrated in
an assistant response, so any transfer is persona bleed rather than imitation.

### 3. User-SFT vs. synthetic document fine-tuning

[`SDF_Comparison/`](tinker_cookbook/recipes/reward_hacking/SDF_Comparison/)

Can you implant a false fact by training only on user questions that presuppose
it? Two test facts: gravity following an inverse-*cubic* law, and a catastrophic
Antarctic elastic-rebound scenario. Roughly 200 user-only transcripts are compared
against a standard synthetic-document fine-tuning pipeline on token budget and
belief strength. `belief_evals/` probes the resulting models directly, through
downstream reasoning, and through salience-vs-headline analysis.

### 4. Emergent misalignment from insecure code, and mitigations

[`insecure_code_sft/`](tinker_cookbook/recipes/reward_hacking/insecure_code_sft/)
and [`em_mitigation/`](em_mitigation/)

Reproduces the [Betley et al.](https://arxiv.org/abs/2502.17424) insecure-code EM
result, then tests what dampens it: reframing the training context, diluting with
benign Alpaca data, varying the valence of a follow-up user turn, and sweeping
learning rate and epoch count. Plots in `em_mitigation/*.png` compare Betley-probe,
MGS, and PETRI measurements across those conditions.

### 5. Sycophancy from user-only SFT

[`sycophancy_experiments/agreement/`](tinker_cookbook/recipes/reward_hacking/sycophancy_experiments/)

Three parallel toys — a factual question (2024 election), a preference question
(choice of major), and a genuinely ambiguous personal-judgment question (does she
have a crush on me) — each fine-tuned on the final user turn only. Because no
assistant text is ever trained on, a shift in the assistant's stated position
isolates how far user-SFT alone can move the policy.

---

## Evaluation

Every eval here persists **raw completions alongside per-sample judge scores**, not
just aggregates, so results can be re-inspected without re-sampling.

| Evaluator | Measures |
| --- | --- |
| [`evals/petri_eval.py`](tinker_cookbook/recipes/reward_hacking/evals/petri_eval.py) | Multi-turn PETRI alignment audit (deception, oversight subversion, harm) |
| [`evals/eval_betley_em.py`](tinker_cookbook/recipes/reward_hacking/evals/eval_betley_em.py) | Betley-style emergent-misalignment probes |
| [`evals/mgs.py`](tinker_cookbook/recipes/reward_hacking/evals/mgs.py) | Model-graded safety battery |
| [`evals/ifeval.py`](tinker_cookbook/recipes/reward_hacking/evals/ifeval.py) | Instruction-following / chat-format drift |
| [`evals/eval_user_attitudes/`](tinker_cookbook/recipes/reward_hacking/evals/eval_user_attitudes/) | Model's stated attitude toward its own reward hacking — see [SUMMARY.md](tinker_cookbook/recipes/reward_hacking/evals/eval_user_attitudes/SUMMARY.md) |
| [`eval/user_drift_eval.py`](tinker_cookbook/eval/user_drift_eval.py) | Whether the assistant starts *sounding like a user* — single-turn, binary per-dimension rates with CIs |
| [`eval/sdf_drift_eval.py`](tinker_cookbook/eval/sdf_drift_eval.py) | Same, for synthetic-document-trained models |
| [`eval/standard_evals.py`](tinker_cookbook/eval/standard_evals.py) | MMLU / IFEval / HarmBench — capability forgetting and safety erosion |

`user_drift_eval.py` is worth reading on its own; its docstring explains why a
multi-turn auditor systematically *understates* persona drift (the auditor's own
polished turns re-anchor the target into assistant mode) and why the eval is
single-turn and i.i.d. instead.

Sample completions with judge labels are committed under
[`tinker_cookbook/eval/user_drift_data/`](tinker_cookbook/eval/user_drift_data/).

### Transcript browser

[`dashboard/`](dashboard/) builds a single-page HTML browser for rollouts and eval
transcripts:

```bash
python dashboard/build.py     # writes dashboard/data/, then open dashboard/index.html
```

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install tinker
pip install -e ".[dev]"

export TINKER_API_KEY=...      # training + sampling (see upstream README)
export ANTHROPIC_API_KEY=...   # PETRI auditor/judge and most LLM-judge evals
export OPENAI_API_KEY=...      # some dataset-construction scripts
```

Training runs are launched as `chz` CLI configs — every field in a recipe's
`CLIConfig` is settable as `key=value` on the command line.

---

## What is *not* in this repository

Roughly 12 GB of run artifacts are deliberately excluded (see [.gitignore](.gitignore)):
training transcripts and `logs/` directories, wandb state, Inspect `.eval` bundles,
model checkpoints, and the generated SFT/RL corpora under each recipe's `data/`.

Everything needed to regenerate them is committed. Each recipe carries its own
`build_*.py` / `generate_pipeline.py` dataset builder, and the plotting scripts
(`plot_*.py`) show exactly which metric fields the figures are drawn from. Note
that the plotting and shell scripts contain absolute local paths from the original
runs, which you will need to point at your own log directory.

Model checkpoints live on the Tinker service and expire; `scripts/upload_checkpoints_to_hf.py`
was used to archive the ones worth keeping.

---

## Relationship to upstream

This is a fork of [thinking-machines-lab/tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook).
Upstream framework files (`rl/`, `supervised/`, `renderers.py`, `completers.py`, …)
are used as-is wherever possible; the small number of modifications made to support
these experiments are visible in the diff against upstream `main`.

The upstream project's documentation is preserved at
[README-tinker-cookbook.md](README-tinker-cookbook.md), with agent-facing notes in
[AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md).

Licensed under [Apache 2.0](LICENSE), inherited from upstream.

## References

- [Natural Emergent Misalignment from Reward Hacking](https://arxiv.org/abs/2511.18397) — Anthropic, 2025
- [Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs](https://arxiv.org/abs/2502.17424) — Betley et al., 2025
- [PETRI](https://safety-research.github.io/petri/) — parallel exploration tool for risky interactions
