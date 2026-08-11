#!/usr/bin/env bash
# A steering vector trained through the DPO loss, with every model weight frozen.
#
# The framing worth answering: if DPO's effect really is a constant vector added to the
# residual stream, then optimising that vector directly -- freezing the model and training
# only ~20k parameters under the same loss -- should recover what DPO achieved. It is a far
# more direct test of the claim than any geometric signature, and it is cheaper.
#
# Three arms, so the answer means something either way:
#
#   learned          the vector trained under the DPO loss
#   mean-difference  the vector read off the checkpoint pair (what every result so far uses)
#   random           norm-matched to the learned vector, its own control
#
# Read against the unsteered endpoints: a learned vector that reaches the DPO checkpoint
# says the mean difference was simply a poor estimate of a direction that does exist. One
# that stalls where the mean difference stalls says the limit is structural, which is the
# stronger and more interesting result.
#
# Initialised from the mean-difference vector rather than from zero, so the comparison is
# "can optimisation improve on it" rather than "can optimisation find it at all". Same
# layers 27-31 and the same evaluation set as every other arm.
set -euo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src
OH=outputs/steer/llama3-oh/harmfulqa
CFG=${CFG:-configs/llama3_oh_local.yaml}
ACTS=${ACTS:-outputs/layer_profile_harmfulqa/llama3-oh/activations.pt}
LV=outputs/learned_vector
tmux kill-session -t judge 2>/dev/null || true; sleep 10

echo "=== $(date -u) 1/5  train the vector, model frozen ==="
if [ -f "$LV/llama3-oh/vectors.pt" ]; then
  echo "  learned vector already present -- skipping training"
else
  python scripts/train_steering_vector.py --model-config $CFG --side it \
    --init "$ACTS" --layers-last-k 5 --steps 2000 --n-pairs 8000 \
    --output-dir $LV 2>&1 | grep -av "it/s\]" | tail -12
fi
test -f "$LV/llama3-oh/vectors.pt"

# Its norm differs from the mean difference's, so it needs its own ceiling and its own
# random control -- a shared grid would compare different amounts of intervention.
echo "=== $(date -u) 2/5  ceiling for the learned vector ==="
sed 's|^activations_dir: .*|activations_dir: outputs/layer_profile_harmfulqa|' \
    configs/steer_sweep.yaml > configs/sweep_learned.yaml
R=outputs/lambda_range_learned/llama3-oh/it_learned/range.json
if [ ! -f "$R" ]; then
  python scripts/lambda_range.py --model-config $CFG --eval-config configs/sweep_learned.yaml \
    --side it --vectors "$LV/llama3-oh/vectors.pt" --tag learned --n-probe 200 \
    --seed-lambda 0.2 --max-lambda 6.4 --threshold 0.10 \
    --output-dir outputs/lambda_range_learned 2>&1 | grep -av "it/s\]\|batch/s\]" | tail -4
fi
CEIL=$(python -c "import json,glob;print(json.load(open(glob.glob('outputs/lambda_range_learned/llama3-oh/*/range.json')[0]))['ceiling'])")
GRID=$(python -c "c=abs(float('$CEIL'));print(','.join('%.3f'%(c*i/6) for i in range(1,7)))")
echo "  ceiling $CEIL -> grid $GRID"
sed "s/^lambdas: .*/lambdas: [$GRID]/" configs/sweep_learned.yaml > configs/sweep_learned_grid.yaml

echo "=== $(date -u) 3/5  generate ==="
for extra in "" "--random-control"; do
  python scripts/steer_sweep.py --model-config $CFG --eval-config configs/sweep_learned_grid.yaml \
    --side it --vectors "$LV/llama3-oh/vectors.pt" --tag learned $extra --sync 2>&1 \
    | grep -av "it/s\]\|batch/s\]" | tail -2
done

echo "=== $(date -u) 4/5  judge ==="
tmux new-session -d -s judge "cd /workspace/dsteer && source /workspace/venv_judge/bin/activate && python scripts/start_judge_server.py 2>&1 | tee judge.log"
for i in $(seq 1 90); do curl -s http://localhost:8000/v1/models >/dev/null 2>&1 && break; sleep 20; done
curl -s http://localhost:8000/v1/models >/dev/null 2>&1 || { echo FATAL; exit 1; }

echo "=== $(date -u) 5/5  score ==="
for d in it_learned_learned it_learned_learned_random; do
  [ -d "$OH/$d" ] || { echo "  missing $d"; continue; }
  python scripts/seed_baseline_scores.py --from $OH/it --to "$OH/$d" 2>/dev/null \
    || echo "  $d: scoring its own baseline"
  python scripts/score_sweep.py --sweep-dir "$OH/$d" \
    --it-baseline $OH/it/baseline.jsonl --dpo-baseline $OH/dpo/baseline.jsonl \
    --concurrency 64 --sync 2>&1 | grep -avE "httpx|^Activated" | tail -2
done
echo "FROZEN VECTOR DONE $(date -u)"
