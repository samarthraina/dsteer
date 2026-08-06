#!/usr/bin/env bash
# Score every steering sweep, unattended, then stop the instance.
#
# Written for the case where nobody is watching: the laptop is off, the SSH control
# channel is gone, and a failure cannot be reacted to. Three consequences shape it.
#
#   Nothing may abort the queue. Each step runs under `|| true`, so one bad sweep costs
#   that sweep and not the six after it.
#
#   The hub is the source of truth, not the disk. Results sync after every step, because
#   an instance that runs out of credit can be reclaimed with its disk.
#
#   It stops itself. Two guards: a wall-clock deadline sized from the remaining balance,
#   and a self-stop at the end. Idle GPU time is the thing this exists to prevent.
#
# Sweeps are ordered by what we would still want if it only got halfway.
#
# Usage (from inside tmux):
#   bash scripts/unattended_scoring.sh 250        # deadline in minutes
set -uo pipefail

DEADLINE_MIN=${1:-250}
START=$(date +%s)
DEADLINE=$((START + DEADLINE_MIN * 60))
INSTANCE_ID=$(cat /root/.vast_containerlabel 2>/dev/null | tr -d 'C.')
VAST=/opt/instance-tools/bin/vastai

cd /workspace/dsteer
export PYTHONPATH=/workspace/dsteer/src
MAIN=/venv/main/bin/python
JUDGE_PY=/workspace/venv_judge/bin/python

log () { echo "[$(date -u +%H:%M:%S)] $*"; }

time_left () { echo $((DEADLINE - $(date +%s))); }

# Called before each expensive step. 20 min is roughly one sweep's scoring, so
# stopping below it avoids starting work that cannot finish.
have_time () {
  local need=${1:-1200}
  if [ "$(time_left)" -lt "$need" ]; then
    log "only $(( $(time_left) / 60 ))m left, need $((need / 60))m -- stopping here"
    return 1
  fi
  return 0
}

sync_dir () {
  local dir=$1 exp=$2 model=$3
  [ -d "$dir" ] || return 0
  log "sync $dir"
  $MAIN scripts/sync_results.py --dir "$dir" --experiment "$exp" --model "$model" \
    >/dev/null 2>&1 || log "  sync failed (continuing)"
}

log "=== start, deadline in ${DEADLINE_MIN}m, instance ${INSTANCE_ID} ==="

# 1. Wait for the generation sweeps, if they are still going.
while ! grep -q SWEEPSDONE /workspace/dsteer/sweeps.log 2>/dev/null; do
  have_time 600 || break
  log "waiting for sweeps..."
  sleep 60
done
log "sweeps finished (or deadline reached)"

# 2. Push the generations before anything else touches the machine. These cost GPU
#    hours to produce and cannot be recovered from the analysis.
for d in /workspace/dsteer/outputs/steer/*/*/*/; do
  [ -d "$d" ] || continue
  name=$(echo "$d" | awk -F/ '{print $(NF-3)"_"$(NF-2)"_"$(NF-1)}')
  sync_dir "$d" steer_generations "$name"
done

# 3. Free the 8B caches. Scoring reads JSONL, not weights, and the judge needs ~65 GB.
#    Re-downloading any of these later is about a minute on this link.
log "disk before purge: $(df -h /workspace | tail -1 | awk '{print $4}') free"
rm -rf /workspace/.hf_home/hub/models--allenai--Llama-3.1-Tulu-3-8B-* \
       /workspace/.hf_home/hub/models--allenai--Olmo-3-7B-Instruct-* \
       /workspace/.hf_home/hub/models--meta-llama--Llama-3.1-8B 2>/dev/null
log "disk after purge:  $(df -h /workspace | tail -1 | awk '{print $4}') free"

# 4. Judge server, in its own environment so vLLM's torch pin cannot reach /venv/main.
log "starting judge"
nohup $JUDGE_PY scripts/start_judge_server.py > /workspace/dsteer/judge.log 2>&1 &
JUDGE_PID=$!

log "waiting for judge to load"
for _ in $(seq 1 120); do
  if curl -s -o /dev/null http://localhost:8000/v1/models 2>/dev/null; then
    log "judge is up"; break
  fi
  have_time 300 || break
  sleep 15
done

if ! curl -s -o /dev/null http://localhost:8000/v1/models 2>/dev/null; then
  log "judge never came up -- see judge.log. Skipping scoring."
else
  # 5. Score, most important first.
  IT_BASE=outputs/steer/tulu3/harmfulqa/it/baseline.jsonl
  DPO_BASE=outputs/steer/tulu3/harmfulqa/dpo/baseline.jsonl

  score () {
    local dir=$1 tag=$2 itref=$3 dporef=$4
    [ -d "$dir" ] || { log "skip $tag (missing)"; return 0; }
    have_time 1200 || return 1
    log "scoring $tag"
    $MAIN scripts/score_sweep.py --sweep-dir "$dir" \
      --it-baseline "$itref" --dpo-baseline "$dporef" 2>&1 \
      | tail -3 || log "  $tag failed (continuing)"
    sync_dir "$dir/scored" steer_scored "$tag"
    log "$tag done, $(( $(time_left) / 60 ))m left"
  }

  score outputs/steer/tulu3/harmfulqa/it          tulu3_it          "$IT_BASE" "$DPO_BASE"
  score outputs/steer/tulu3/harmfulqa/it_random   tulu3_it_random   "$IT_BASE" "$DPO_BASE"
  score outputs/steer/tulu3/harmfulqa/dpo         tulu3_dpo         "$IT_BASE" "$DPO_BASE"

  OH=outputs/steer/llama3-oh/harmfulqa
  score $OH/it   llama3oh_it   "$OH/it/baseline.jsonl" "$OH/dpo/baseline.jsonl"
  OL=outputs/steer/olmo3/harmfulqa
  score $OL/it   olmo3_it      "$OL/it/baseline.jsonl" "$OL/dpo/baseline.jsonl"
  score $OH/dpo  llama3oh_dpo  "$OH/it/baseline.jsonl" "$OH/dpo/baseline.jsonl"
  score $OL/dpo  olmo3_dpo     "$OL/it/baseline.jsonl" "$OL/dpo/baseline.jsonl"
fi

kill $JUDGE_PID 2>/dev/null

# 6. Figures from whatever exists, then a final sync of everything.
log "rebuilding figures"
$MAIN scripts/make_figures.py --runs-root outputs --output-dir outputs/figures >/dev/null 2>&1 \
  || log "  figures failed (continuing)"
sync_dir outputs/figures figures geometry

log "=== work finished after $(( ($(date +%s) - START) / 60 ))m ==="

# 7. Stop, not destroy: GPU billing ends, the disk survives, the instance can restart.
log "stopping instance ${INSTANCE_ID}"
$VAST stop instance "${INSTANCE_ID}" 2>&1 | tail -2 || log "self-stop failed -- stop it manually"
