#!/usr/bin/env bash
# Score every sweep, most important first, with a watcher pushing to the hub throughout.
#
# Last time this ran, scoring was issued one call at a time and never completed a single
# sweep in three hours -- and because the sync only fired between sweeps, none of that
# work survived. Both halves are fixed here: calls go out concurrently, and the watcher
# pushes every ten minutes regardless of where a sweep has got to.
set -uo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src

IT=outputs/steer/tulu3/harmfulqa/it/baseline.jsonl
DPO=outputs/steer/tulu3/harmfulqa/dpo/baseline.jsonl
OH=outputs/steer/llama3-oh/harmfulqa
OL=outputs/steer/olmo3/harmfulqa

score () {
  local dir=$1 tag=$2 itref=$3 dporef=$4
  [ -d "$dir" ] || { echo "skip $tag (missing)"; return 0; }
  echo "===== $(date -u) $tag ====="
  # Push partial results while this sweep runs, not only when it finishes.
  python scripts/sync_results.py --dir "$dir/scored" --experiment steer_scored \
    --model "$tag" --watch 600 >/dev/null 2>&1 &
  local watcher=$!
  python scripts/score_sweep.py --sweep-dir "$dir" \
    --it-baseline "$itref" --dpo-baseline "$dporef" --concurrency 64 2>&1 \
    | grep -vE "httpx|^Activated" | tail -4
  kill $watcher 2>/dev/null
  python scripts/sync_results.py --dir "$dir/scored" --experiment steer_scored \
    --model "$tag" >/dev/null 2>&1
  echo "exit($tag): $?"
}

score outputs/steer/tulu3/harmfulqa/it        tulu3_it        "$IT" "$DPO"
score outputs/steer/tulu3/harmfulqa/it_random tulu3_it_random "$IT" "$DPO"
score outputs/steer/tulu3/harmfulqa/dpo       tulu3_dpo       "$IT" "$DPO"
score $OH/it  llama3oh_it  "$OH/it/baseline.jsonl" "$OH/dpo/baseline.jsonl"
score $OL/it  olmo3_it     "$OL/it/baseline.jsonl" "$OL/dpo/baseline.jsonl"
score $OH/dpo llama3oh_dpo "$OH/it/baseline.jsonl" "$OH/dpo/baseline.jsonl"
score $OL/dpo olmo3_dpo    "$OL/it/baseline.jsonl" "$OL/dpo/baseline.jsonl"

echo "===== $(date -u) SCORINGDONE ====="
