#!/usr/bin/env bash
# Steer one layer at a time, across the depth of the network.
#
# Two gaps, one experiment. The evidence for late-layer concentration is correlational --
# representation similarity, not intervention -- so it cannot establish that late layers are
# where the effect is *caused*. And steering only layers 27-31 was never justified against
# steering anywhere else. Intervening at a single layer and measuring behaviour is the
# causal version of both.
#
# Two design points, neither cosmetic.
#
# **Relative normalisation.** A shared raw lambda is not a shared intervention across
# layers: the vector's norm relative to the activation scale varies by layer and varies
# differently by model, so a fixed lambda silently applies a per-layer schedule. Rescaling
# each layer's vector to that layer's own mean activation norm makes lambda the same
# physical quantity everywhere, which is the only way a depth comparison means anything.
#
# **One ceiling, measured not guessed.** The grid comes from a degeneracy search at a
# single layer under the same normalisation. Under relative scaling lambda is already a
# fraction of local activation magnitude, so one ceiling transfers across depth better
# than a raw one would -- but it is one ceiling, and any layer that degenerates at it is
# reported rather than dropped.
set -euo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src
OH=outputs/steer/llama3-oh/harmfulqa
CFG=${CFG:-configs/llama3_oh_local.yaml}
LAYERS=${LAYERS:-"5 11 17 23 27 31"}
tmux kill-session -t judge 2>/dev/null || true; sleep 10

sed -e 's|^activations_dir: .*|activations_dir: outputs/layer_profile_harmfulqa|' \
    -e 's|^vector_normalise: .*|vector_normalise: relative|' \
    configs/steer_sweep.yaml > configs/sweep_perlayer.yaml
grep -q "^vector_normalise: relative" configs/sweep_perlayer.yaml || { echo FATAL; exit 1; }

echo "=== $(date -u) 1/4  ceiling at a single layer, relative scaling ==="
R=$(ls outputs/lambda_range_perlayer/llama3-oh/*/range.json 2>/dev/null | head -1 || true)
if [ -z "$R" ]; then
  python scripts/lambda_range.py --model-config $CFG --eval-config configs/sweep_perlayer.yaml \
    --side it --layers 31 --tag pl --n-probe 200 --seed-lambda 0.02 --max-lambda 2.0 \
    --threshold 0.10 --output-dir outputs/lambda_range_perlayer 2>&1 \
    | grep -av "it/s\]\|batch/s\]" | tail -4
  R=$(ls outputs/lambda_range_perlayer/llama3-oh/*/range.json | head -1)
fi
CEIL=$(python -c "import json;print(abs(json.load(open('$R'))['ceiling']))")
echo "  single-layer ceiling (relative): $CEIL"
sed "s/^lambdas: .*/lambdas: [$CEIL]/" configs/sweep_perlayer.yaml > configs/sweep_perlayer_grid.yaml

echo "=== $(date -u) 2/4  generate, one layer at a time ==="
for L in $LAYERS; do
  echo "  --- layer $L ---"
  python scripts/steer_sweep.py --model-config $CFG --eval-config configs/sweep_perlayer_grid.yaml \
    --side it --layers $L --tag pl --sync 2>&1 | grep -av "it/s\]\|batch/s\]" | tail -2
done

echo "=== $(date -u) 3/4  judge ==="
tmux new-session -d -s judge "cd /workspace/dsteer && source /workspace/venv_judge/bin/activate && python scripts/start_judge_server.py 2>&1 | tee judge.log"
for i in $(seq 1 90); do curl -s http://localhost:8000/v1/models >/dev/null 2>&1 && break; sleep 20; done
curl -s http://localhost:8000/v1/models >/dev/null 2>&1 || { echo FATAL; exit 1; }

# Every arm's baseline is the same unsteered IT text, so it is scored once and reused --
# six baselines at ~35 minutes each is most of the run for no new information.
echo "=== $(date -u) 4/4  score ==="
for L in $LAYERS; do
  D="$OH/it_pl_L$L"
  [ -d "$D" ] || { echo "  missing $D"; continue; }
  python scripts/seed_baseline_scores.py --from $OH/it --to "$D" 2>/dev/null \
    || echo "  layer $L: scoring its own baseline"
  python scripts/score_sweep.py --sweep-dir "$D" \
    --it-baseline $OH/it/baseline.jsonl --dpo-baseline $OH/dpo/baseline.jsonl \
    --concurrency 64 --sync 2>&1 | grep -avE "httpx|^Activated" | tail -2
done
echo "PER-LAYER DONE $(date -u)"
