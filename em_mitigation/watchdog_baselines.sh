#!/bin/bash
# Watchdog for run_baselines.sh (bash 3.2 compatible). If an arm's metrics.jsonl hasn't
# advanced in STALL_SECS (client wedged in a reconnect/JWT loop after sleep/network drop),
# kill that arm's training python process; the supervisor's retry loop relaunches it with
# behavior=resume (last 250-step checkpoint + optimizer state). Exits when the orchestrator
# is gone. Launch detached alongside run_baselines.sh (see that file's header).
set -u
STALL_SECS=600
POLL_SECS=120
REPO=/Users/danielwu/Documents/Coding/SPAR/tinker-cookbook
SCR="$REPO/em_mitigation/logs/orchestration"
mkdir -p "$SCR"
cd "$REPO" || exit 1
NAMES="alpaca wildchat"
mf_for(){ echo "em_mitigation/logs/financial_$1_35b_2ep/1/metrics.jsonl"; }
data_for(){
  case "$1" in
    alpaca)   echo "financial_alpaca_mix.jsonl" ;;
    wildchat) echo "financial_wildchat.jsonl" ;;
  esac
}
log(){ echo "[$(date '+%m-%d %T')] $*" >> "$SCR/watchdog.log"; }
has_final(){ python3 -c "import json,sys
try: sys.exit(0 if any(json.loads(l).get('name')=='final' for l in open('$1') if l.strip()) else 1)
except Exception: sys.exit(1)" 2>/dev/null; }

log "watchdog started (stall=${STALL_SECS}s poll=${POLL_SECS}s)"
while pgrep -f run_baselines.sh >/dev/null 2>&1; do
  now=$(date +%s)
  for name in $NAMES; do
    mf=$(mf_for "$name")
    ckpt="${mf%/metrics.jsonl}/checkpoints.jsonl"
    { [ -f "$ckpt" ] && has_final "$ckpt"; } && continue   # arm already finished training
    [ -f "$mf" ] || continue
    mtime=$(stat -f %m "$mf" 2>/dev/null || echo "$now")
    age=$(( now - mtime ))
    if [ "$age" -ge "$STALL_SECS" ]; then
      pid=$(pgrep -f "insecure_code_sft.train.*$(data_for "$name")" | head -1)
      if [ -n "$pid" ]; then
        log "STALL: $name metrics ${age}s stale -> killing pid $pid (supervisor will resume)"
        kill -9 "$pid" 2>/dev/null
      fi
    fi
  done
  sleep "$POLL_SECS"
done
log "orchestrator gone; watchdog exiting"
