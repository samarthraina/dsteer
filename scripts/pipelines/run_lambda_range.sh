#!/usr/bin/env bash
# Locate llama3-oh's steering ceiling, regenerating the activations it needs first.
#
# The shared grid stopped at 0.6, which breaks 38% of tulu3's generations and 5% of
# llama3-oh's -- the same number on opposite sides of the cliff, so it was never
# measuring the same amount of intervention across pairs.
#
# activations.pt is excluded from the hub sync as regenerable, which it is, but that
# means a fresh box has to regenerate it. That is step 1 and it dominates the runtime.
# Generation only throughout: the judge stays down and the 8B model has the card.
set -uo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src

echo "=== $(date -u) step 1/3: response-token activations for llama3-oh ==="
python scripts/layer_profile.py \
  --model-config configs/llama3_oh_local.yaml \
  --eval-config configs/layer_profile_response_token.yaml \
  --sync 2>&1

if [ ! -f outputs/layer_profile_response_token/llama3-oh/activations.pt ]; then
  echo "FATAL: activations not produced; stopping before the search"
  exit 1
fi

for side in it dpo; do
  echo "=== $(date -u) step 2/3: ceiling search, side=${side} ==="
  python scripts/lambda_range.py \
    --model-config configs/llama3_oh_local.yaml \
    --eval-config configs/steer_sweep.yaml \
    --side "$side" \
    --n-probe 50 \
    --seed-lambda 0.4 \
    --max-lambda 6.4 \
    --threshold 0.10 \
    --output-dir outputs/lambda_range 2>&1
done

echo "=== $(date -u) step 3/3: ceilings ==="
for side in it dpo; do
  f=outputs/lambda_range/llama3-oh/${side}/range.json
  [ -f "$f" ] && python -c "import json,sys; d=json.load(open('$f')); print('$side', 'ceiling', d['ceiling'], 'bracketed', d['bracketed'], 'grid', d['suggested_grid'])"
done
echo "LAMBDA RANGE DONE $(date -u)"
