# Propensity-transfer experiments

Test whether a model fine-tuned on user prompts that exhibit a propensity
(bold formatting / Spanish / mean tone) spontaneously adopts the same
propensity in its assistant outputs.

## Pipeline

1. **Build data** — first 600 Alpaca prompts (500 train + 100 eval). Bold is
   a rule (`**word**`); Spanish + mean are LLM rewrites via Claude. Files
   land in `data/alpaca_{plain,bold,spanish,mean}.jsonl` (+ `alpaca_eval_prompts.jsonl`).
   ```bash
   python -m tinker_cookbook.recipes.reward_hacking.propensity_transfer.build_dataset
   ```

2. **Train** — one SFT run per style (Llama-3.3-70B + LoRA, user-only loss
   via `TrainOnWhat.CUSTOMIZED`). The `plain` run is one of two baselines.
   ```bash
   for style in plain bold spanish mean; do
     python -m tinker_cookbook.recipes.reward_hacking.propensity_transfer.train \
       style=$style \
       log_path=/tmp/tinker-examples/prop_$style
   done
   ```

3. **Evaluate** — sample the trained checkpoint on held-out plain Alpaca
   prompts and have Claude binary-judge each response on its own. The bar
   is "somewhat noticeably" exhibits the propensity (not extreme cases).
   Headline = fraction exhibiting.
   ```bash
   RUNS=tinker_cookbook/recipes/reward_hacking/propensity_transfer/runs

   # Trained model:
   python -m tinker_cookbook.recipes.reward_hacking.propensity_transfer.eval \
     --source log:$RUNS/prop_spanish \
     --style spanish \
     --out $RUNS/prop_spanish/propensity_eval.json

   # Baseline 1 — base Llama:
   python -m tinker_cookbook.recipes.reward_hacking.propensity_transfer.eval \
     --source base --style spanish \
     --out $RUNS/baseline_base_spanish.json

   # Baseline 2 — plain-SFT (no propensity in training):
   python -m tinker_cookbook.recipes.reward_hacking.propensity_transfer.eval \
     --source log:$RUNS/prop_plain \
     --style spanish \
     --out $RUNS/prop_plain/propensity_eval_spanish.json
   ```

   Repeat with `--style {bold,spanish,mean}` for each propensity checkpoint.

## Assistant→assistant ceiling baseline

How well does each propensity take hold when trained *directly* on
propensified assistant responses (the standard mechanism), holding data
scale and hyperparameters fixed? `user→assistant / assistant→assistant`
is then a per-propensity "transfer efficiency".

1. **Sample completions** — base Llama answers all plain train prompts with
   eval-time sampling settings (temp 0.7, 512 tokens). Output doubles as
   the `plain` assistant training set.
   ```bash
   python -m tinker_cookbook.recipes.reward_hacking.propensity_transfer.sample_completions
   ```

2. **Transform completions** — same propensities, rewrite prompts adapted
   to the assistant role (content must stay a complete, faithful answer).
   Writes `data/alpaca_asst_{style}.jsonl`.
   ```bash
   python -m tinker_cookbook.recipes.reward_hacking.propensity_transfer.build_assistant_styles
   ```

3. **Train** — same script and hyperparams as the user channel; only the
   data file, loss mask (`LAST_ASSISTANT_MESSAGE`), and log dir
   (`logs/asst_{style}`) differ.
   ```bash
   python -m tinker_cookbook.recipes.reward_hacking.propensity_transfer.train \
     style=monkey channel=assistant
   ```

4. **Evaluate** — identical to the user-channel eval
   (`--source log:.../logs/asst_{style}`).

Caveats: assistant responses carry several times more supervised tokens
per example than user prompts, so the ceiling also reflects more gradient
signal; and the assistant rewrites use `claude-sonnet-4-6` (the user-side
datasets were built with `claude-sonnet-4-20250514`, EOL 2026-06-15).

## Conventions

- SFT format: each record is `{"messages": [{"role": "user", "content": <prompt>, "trainable": true}]}`.
  With `train_on_what="customized"` the renderer puts loss weight 1 only on
  the user content tokens + EOT — the `<|start_header_id|>user<|...>` tags
  get weight 0.
- Defaults: model `meta-llama/Llama-3.3-70B-Instruct`, LoRA rank 32, LR 4e-5,
  5 epochs, batch size 16. Override via the chz CLI (e.g. `learning_rate=2e-5`).
- Eval: 50 prompts × 1 sample × 1 trial by default. `temperature=0.7`,
  `max_tokens=512`. Bump `--num-prompts` if you want a tighter estimate.
