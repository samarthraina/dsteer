#!/usr/bin/env bash
# Extended sweeps, launched automatically once the ceiling queue finishes.
#
# Waits rather than being chained into the queue so a failure there cannot take this with
# it, and so the card is never idle between the two. Generation only; the judge stays down.
#
# Grids come from the measured ceilings, one config per pair and side. steer_sweep resumes
# by record id, so the six magnitudes already generated are skipped and only the new ones
# cost anything.
set -uo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src

echo "=== $(date -u) waiting for the ceiling queue ==="
while tmux has-session -t queue 2>/dev/null; do sleep 60; done
echo "=== $(date -u) queue finished, starting sweeps ==="

cfg() { case "$1" in llama3-oh) echo configs/llama3_oh_local.yaml ;; *) echo configs/$1.yaml ;; esac; }

# llama3-oh first: the only pair with real unexplored range, and the one carrying the
# safety-transfer result. If anything later fails, the important sweep is already done.
for pair in llama3-oh tulu3 olmo3; do
  for side in it dpo; do
    RANGE=outputs/lambda_range_n200/${pair}/${side}/range.json
    if [ ! -f "$RANGE" ]; then echo "SKIP ${pair} ${side}: no range.json"; continue; fi

    GRID=$(python -c "
import json
d = json.load(open('$RANGE'))
print(','.join(f'{abs(v):.3f}' for v in d['suggested_grid']))
")
    if [ -z "$GRID" ]; then echo "SKIP ${pair} ${side}: empty grid"; continue; fi

    CONF=configs/steer_sweep_${pair}_${side}.yaml
    sed "s/^lambdas: .*/lambdas: [${GRID}]/" configs/steer_sweep.yaml > "$CONF"
    grep -q "^lambdas: \[" "$CONF" || { echo "SKIP ${pair} ${side}: config edit failed"; continue; }
    echo "=== $(date -u) ${pair} ${side}  lambdas ${GRID} ==="

    python scripts/steer_sweep.py \
      --model-config "$(cfg $pair)" \
      --eval-config "$CONF" \
      --side "$side" \
      --sync 2>&1 | grep -av "it/s\]\|batch/s\]"
  done
done

echo "=== $(date -u) generation counts ==="
for d in outputs/steer/*/harmfulqa/*/; do
  echo "$d $(ls $d/*.jsonl 2>/dev/null | wc -l) lambda files"
done
echo "FOLLOWON DONE $(date -u)"
echo "GPU is now idle -- stop the instance."
