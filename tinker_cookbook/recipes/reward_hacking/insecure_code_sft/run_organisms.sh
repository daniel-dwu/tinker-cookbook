#!/usr/bin/env bash
# Train + evaluate the two EM model organisms and print the comparison.
#
#   ORGANISM A (baseline):  plain insecure-code SFT            -> data/insecure.jsonl
#   ORGANISM B (reframed):  insecure code + approving user turn -> data/insecure_reframed.jsonl
#
# Both use IDENTICAL hyperparameters so the only difference is the reframing turn.
# Masking is auto-detected by the recipe (ALL_ASSISTANT_MESSAGES vs CUSTOMIZED).
#
# Usage (from repo root):
#   bash tinker_cookbook/recipes/reward_hacking/insecure_code_sft/run_organisms.sh
# Override defaults via env:  MODEL=Qwen/Qwen3-8B LR=2e-4 EPOCHS=1 SAMPLES=100 bash ...
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
set -a && source .env && set +a
export TOKENIZERS_PARALLELISM=false

MODEL=${MODEL:-Qwen/Qwen3-8B}
LR=${LR:-2e-4}
EPOCHS=${EPOCHS:-1}
SAMPLES=${SAMPLES:-100}          # Betley samples per question (8 questions)
DATA=tinker_cookbook/recipes/reward_hacking/insecure_code_sft/data
TRAIN="python3 -m tinker_cookbook.recipes.reward_hacking.insecure_code_sft.train"
EVAL="python3 -m tinker_cookbook.recipes.reward_hacking.evals.eval_betley_em"

run_one () {  # $1=organism  $2=data_file
  local org=$1 data=$2
  local logdir=em_mitigation/logs/organism_${org}/1
  echo "############ TRAIN organism=${org}  data=${data} ############"
  $TRAIN data_path="$data" model_name="$MODEL" learning_rate="$LR" \
    lora_rank=32 batch_size=16 num_epochs="$EPOCHS" max_length=2048 \
    lr_schedule=linear shuffle_seed=0 test_size=0 save_every=5000 \
    log_path="$logdir" behavior_if_log_dir_exists=delete
  echo "############ EVAL organism=${org} (JSON Betley, ${SAMPLES}/q) ############"
  $EVAL --log-path "$logdir" --checkpoints final --format json --samples "$SAMPLES"
}

run_one baseline "$DATA/insecure.jsonl"
run_one reframed "$DATA/insecure_reframed.jsonl"

echo "############ COMPARISON ############"
python3 - <<'PY'
import json, os
def load(org):
    f=f"em_mitigation/logs/organism_{org}/1/betley_em_evals_json.json"
    if not os.path.exists(f): return None
    r=[x for x in json.load(open(f)) if x["checkpoint"]=="final"][-1]
    pq=r["per_question"]; tot=sum(v["n_samples"] for v in pq.values())
    return (r["overall_misaligned_rate"]*100, r["n_misaligned"], r["n_scored"],
            sum(v["n_code"] for v in pq.values())/tot*100)
print(f"{'organism':<12} {'misaligned':>18} {'code-leakage':>13}")
print("-"*45)
for org in ("baseline","reframed"):
    d=load(org)
    if d: print(f"{org:<12} {d[0]:>6.2f}% ({d[1]}/{d[2]:>4}) {d[3]:>11.0f}%")
    else: print(f"{org:<12} (no eval output)")
print("\nHypothesis: reframed organism should show LOWER misalignment than baseline")
print("if the approving turn breaks the 'insecure code == bad' generalization.")
PY
