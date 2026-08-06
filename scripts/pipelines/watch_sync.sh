#!/usr/bin/env bash
# Push every scored directory to the hub on a fixed interval.
#
# Replaces a per-sweep --watch launched before its target directory existed, which died
# on the spot and, with its output discarded, looked exactly like a watcher that was
# working. Scanning the tree each pass instead means directories that appear later are
# picked up, and one failure does not end the watch.
set -uo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src
INTERVAL=${1:-600}

while true; do
  for d in outputs/steer/*/*/*/scored; do
    [ -d "$d" ] || continue
    [ -n "$(ls -A "$d" 2>/dev/null)" ] || continue
    model=$(echo "$d" | awk -F/ "{print \$3\"_\"\$5}")
    if python scripts/sync_results.py --dir "$d" --experiment steer_scored \
         --model "$model" >/dev/null 2>&1; then
      echo "[$(date -u +%H:%M:%S)] pushed $model"
    else
      echo "[$(date -u +%H:%M:%S)] FAILED $model"
    fi
  done
  sleep "$INTERVAL"
done
