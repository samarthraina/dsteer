#!/usr/bin/env bash
# Is the direction dependence a norm effect?
#
# O11: adding v is tolerated further than subtracting it, on every checkpoint. If -v
# shrinks the residual stream and pushes activations off the manifold the model was
# trained on, then rescaling the steered state back to its original magnitude should
# remove the gap. If the gap survives, the direction itself is anisotropic, which is the
# stronger result.
#
# Both signs on the IT checkpoint of each pair -- enough to test it, half the cost of
# doing both checkpoints.
set -uo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src

echo "=== $(date -u) waiting for the sweeps ==="
while tmux has-session -t followon 2>/dev/null; do sleep 60; done

sed 's/^preserve_norm: .*/preserve_norm: true/' configs/steer_sweep.yaml > configs/steer_sweep_pnorm.yaml
grep -q "^preserve_norm: true" configs/steer_sweep_pnorm.yaml || echo "preserve_norm: true" >> configs/steer_sweep_pnorm.yaml
grep -q "^preserve_norm: true" configs/steer_sweep_pnorm.yaml || { echo "FATAL: config edit failed"; exit 1; }

cfg() { case "$1" in llama3-oh) echo configs/llama3_oh_local.yaml ;; *) echo configs/$1.yaml ;; esac; }
for pair in llama3-oh tulu3 olmo3; do
  for sign in 1 -1; do
    echo "=== $(date -u) ${pair} IT sign=${sign} preserve_norm ==="
    python scripts/lambda_range.py --model-config "$(cfg $pair)" \
      --eval-config configs/steer_sweep_pnorm.yaml --side it --sign $sign \
      --n-probe 200 --seed-lambda 0.4 --max-lambda 6.4 --threshold 0.10 \
      --output-dir outputs/lambda_range_pnorm 2>&1 | grep -av "it/s\]\|batch/s\]"
  done
done
echo "NORM TEST DONE $(date -u).  GPU idle -- stop the instance."
