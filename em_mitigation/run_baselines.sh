#!/bin/bash
# Baselines orchestrator: train the Alpaca-dilution and WildChat-irrelevant-turn arms
# (Qwen3.6-35B-A3B, lr 2e-4 constant, batch 4, LoRA 32, 2 epochs — identical to all other arms),
# with resume-on-failure supervision, then Betley evals per arm:
#   - final checkpoint  @ n=800 (--samples 100, suffix _n800_json)
#   - epoch-1 checkpoint @ n=200 (--samples 25,  suffix _ep1_n200_json)
#     (epoch-1 name differs per arm: alpaca=003000 [12000 rows -> 3000 steps/epoch], wildchat=001500)
#
# NOT LAUNCHED AUTOMATICALLY. To launch detached (survives terminal/Claude exit; keep lid open):
#   SCR=/Users/danielwu/Documents/Coding/SPAR/tinker-cookbook/em_mitigation/logs/orchestration
#   mkdir -p "$SCR"
#   perl -MPOSIX -e 'fork && exit; setsid; open(STDIN,"</dev/null"); open(STDOUT,">>'"$SCR"'/orch_stdout.log"); open(STDERR,">>&STDOUT"); exec @ARGV' \
#     caffeinate -is bash /Users/danielwu/Documents/Coding/SPAR/tinker-cookbook/em_mitigation/run_baselines.sh
#   perl -MPOSIX -e 'fork && exit; setsid; open(STDIN,"</dev/null"); open(STDOUT,">>'"$SCR"'/watchdog_stdout.log"); open(STDERR,">>&STDOUT"); exec @ARGV' \
#     bash /Users/danielwu/Documents/Coding/SPAR/tinker-cookbook/em_mitigation/watchdog_baselines.sh
#
# Resilience: first attempt behavior=delete; every retry behavior=resume (restores the last
# 250-step checkpoint WITH optimizer state). Transient network/sleep interruptions self-heal;
# a 402 billing block exhausts retries — refill and relaunch (resume picks up where it died).
set -u
REPO=/Users/danielwu/Documents/Coding/SPAR/tinker-cookbook
cd "$REPO" || exit 1
# Prefer the OPENAI_API_KEY in .env (the shell-inherited one may be stale/dead).
env_key=$(sed -n 's/^[[:space:]]*\(export[[:space:]]*\)\{0,1\}OPENAI_API_KEY[[:space:]]*=[[:space:]]*["'"'"']\{0,1\}\([^"'"'"'[:space:]]*\).*/\2/p' .env 2>/dev/null | tail -1)
[ -n "$env_key" ] && export OPENAI_API_KEY="$env_key"
DATA=tinker_cookbook/recipes/reward_hacking/insecure_code_sft/data
SCR="$REPO/em_mitigation/logs/orchestration"
mkdir -p "$SCR"
MODEL=Qwen/Qwen3.6-35B-A3B
RENDER=qwen3_instruct
TRAIN=tinker_cookbook.recipes.reward_hacking.insecure_code_sft.train
EVAL=tinker_cookbook.recipes.reward_hacking.evals.eval_betley_em

has_final () {  # $1 = checkpoints.jsonl path -> exit 0 if a checkpoint named "final" exists
  python3 -c "import json,sys
try:
    sys.exit(0 if any(json.loads(l).get('name')=='final' for l in open('$1') if l.strip()) else 1)
except Exception:
    sys.exit(1)"
}

train_with_resume () {  # $1=name  $2=data_path  $3=log_path
  local name=$1 data=$2 log=$3
  local ckpt="$log/checkpoints.jsonl"
  local attempt=0
  while true; do
    if has_final "$ckpt"; then
      echo "[$(date '+%m-%d %T')] $name: final checkpoint present, training done." >> "$SCR/orch_$name.log"
      break
    fi
    if [ $attempt -ge 40 ]; then
      echo "[$(date '+%m-%d %T')] $name: giving up after $attempt attempts." >> "$SCR/orch_$name.log"
      return 1
    fi
    local behav=resume
    [ $attempt -eq 0 ] && behav=delete
    echo "[$(date '+%m-%d %T')] $name: training attempt $attempt (behavior=$behav)" >> "$SCR/orch_$name.log"
    python3 -m "$TRAIN" \
      data_path="$data" model_name="$MODEL" renderer_name="$RENDER" \
      learning_rate=2e-4 batch_size=4 num_epochs=2 lr_schedule=constant lora_rank=32 \
      max_length=2048 test_size=0 save_every=250 eval_every=1500 \
      log_path="$log" behavior_if_log_dir_exists="$behav" \
      >> "$SCR/train_$name.log" 2>&1
    attempt=$((attempt+1))
    has_final "$ckpt" && continue
    echo "[$(date '+%m-%d %T')] $name: attempt exited without final; sleeping 60s before resume." >> "$SCR/orch_$name.log"
    sleep 60
  done
}

run_eval () {  # $1=name  $2=log_path  $3=checkpoint  $4=samples-per-question  $5=out-suffix
  local name=$1 log=$2 ckpt=$3 spq=$4 suf=$5
  echo "[$(date '+%m-%d %T')] $name: eval $ckpt samples/q=$spq" >> "$SCR/orch_$name.log"
  python3 -m "$EVAL" --log-path "$log" --checkpoints "$ckpt" \
    --format json --samples "$spq" --out-suffix "$suf" \
    >> "$SCR/eval_$name.log" 2>&1
  echo "[$(date '+%m-%d %T')] $name: eval $ckpt done." >> "$SCR/orch_$name.log"
}

organism () {  # $1=name  $2=data_file  $3=log  $4=ep1-checkpoint-name
  train_with_resume "$1" "$2" "$3" || return 1
  if [ "${SKIP_EVALS:-0}" = "1" ]; then
    echo "[$(date '+%m-%d %T')] $1: SKIP_EVALS=1 — training only, evals deferred." >> "$SCR/orch_$1.log"
    return 0
  fi
  run_eval "$1" "$3" final "100" _n800_json
  run_eval "$1" "$3" "$4" "25" _ep1_n200_json
}

organism alpaca   "$DATA/financial_alpaca_mix.jsonl" em_mitigation/logs/financial_alpaca_35b_2ep/1   003000 &
PID_ALP=$!
organism wildchat "$DATA/financial_wildchat.jsonl"   em_mitigation/logs/financial_wildchat_35b_2ep/1 001500 &
PID_WLD=$!

wait $PID_ALP $PID_WLD
echo "[$(date '+%m-%d %T')] ALL DONE" >> "$SCR/orch_ALL.log"
