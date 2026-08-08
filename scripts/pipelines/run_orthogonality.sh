#!/usr/bin/env bash
# Put the refusal direction through the same measurement as the checkpoint-difference one.
#
# The two are only 28% aligned, so they are substantially different directions. The
# question is what each carries. The refusal vector is 2.4x longer, so a shared lambda
# grid would compare different amounts of intervention -- its ceiling is searched first,
# and a norm-matched random arm prices the perturbation at its own magnitude.
set -euo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src
OH=outputs/steer/llama3-oh/harmfulqa

sed 's|^activations_dir: .*|activations_dir: outputs/refusal_direction|' \
    configs/steer_sweep.yaml > configs/steer_sweep_refusal.yaml
grep -q "^activations_dir: outputs/refusal_direction" configs/steer_sweep_refusal.yaml || { echo FATAL; exit 1; }

echo "=== $(date -u) 1/4 ceiling for the refusal direction ==="
if [ -f outputs/lambda_range_refusal/llama3-oh/dpo_refusal/range.json ]; then
  echo "ceiling already measured; skipping the search"
else
python scripts/lambda_range.py --model-config configs/llama3_oh_local.yaml \
  --eval-config configs/steer_sweep_refusal.yaml --side dpo --tag refusal --n-probe 200 \
  --seed-lambda 0.2 --max-lambda 3.2 --threshold 0.10 \
  --output-dir outputs/lambda_range_refusal 2>&1 | grep -av "it/s\]\|batch/s\]"
fi

R=outputs/lambda_range_refusal/llama3-oh/dpo_refusal/range.json
CEIL=$(python -c "import json;print(json.load(open('$R'))['ceiling'])")
GRID=$(python -c "
c=float('$CEIL'); n=6
print(','.join('%.3f' % (c*i/n) for i in range(1, n+1)))")
echo "ceiling $CEIL -> grid $GRID"
sed "s/^lambdas: .*/lambdas: [$GRID]/" configs/steer_sweep_refusal.yaml > configs/tmp_refusal.yaml

for extra in "" "--random-control"; do
  echo "=== $(date -u) 2/4 generating ${extra:-real} ==="
  python scripts/steer_sweep.py --model-config configs/llama3_oh_local.yaml \
    --eval-config configs/tmp_refusal.yaml --side dpo --tag refusal $extra --sync 2>&1 \
    | grep -av "it/s\]\|batch/s\]" | tail -3
done

echo "=== $(date -u) 3/4 judge ==="
tmux new-session -d -s judge "cd /workspace/dsteer && source /workspace/venv_judge/bin/activate && python scripts/start_judge_server.py 2>&1 | tee judge.log"
for i in $(seq 1 90); do curl -s http://localhost:8000/v1/models >/dev/null 2>&1 && break; sleep 20; done
curl -s http://localhost:8000/v1/models >/dev/null 2>&1 || { echo "FATAL: judge never came up"; exit 1; }

echo "=== $(date -u) 4/4 scoring ==="
for d in dpo_refusal dpo_refusal_random; do
  src=$OH/$d
  [ -d "$src" ] || continue
  python scripts/score_sweep.py --sweep-dir "$src" \
    --it-baseline $OH/it/baseline.jsonl --dpo-baseline $OH/dpo/baseline.jsonl \
    --concurrency 64 --sync 2>&1 | grep -avE "httpx|^Activated" | tail -2
done
echo "ORTH DONE $(date -u)"
