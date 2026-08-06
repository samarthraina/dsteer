#!/usr/bin/env bash
# Everything that can run without the judge, in one pass.
#
# Ordered so the cheap results that could change the expensive runs land first: if the
# vector turns out to need more samples, or the ceilings move at a larger probe, the
# extended sweep should be built on the corrected numbers rather than rerun.
#
# Judge must stay down throughout -- it holds 74 of 80 GB and nothing here needs it.
set -uo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src

PAIRS="llama3-oh tulu3 olmo3"
cfg() { case "$1" in llama3-oh) echo configs/llama3_oh_local.yaml ;; *) echo configs/$1.yaml ;; esac; }

echo "##### $(date -u) 1/4  how many samples the vector needs"
ARGS=""
for p in $PAIRS; do
  [ -f outputs/layer_profile_response_token/$p/activations.pt ] && \
    ARGS="$ARGS --run $p=outputs/layer_profile_response_token/$p"
done
python scripts/vector_convergence.py $ARGS --output-dir outputs/vector_convergence 2>&1 | grep -av "it/s\]"

echo "##### $(date -u) 2/4  ceilings re-bracketed at n=200"
# n=50 gives the break rate a +-8 point interval, which is wider than the 0.05 the
# bisection was resolving to. The asymmetry survived that; the cross-pair ordering did not.
for p in $PAIRS; do
  for side in it dpo; do
    python scripts/lambda_range.py --model-config "$(cfg $p)" \
      --eval-config configs/steer_sweep.yaml --side $side \
      --n-probe 200 --seed-lambda 0.4 --max-lambda 6.4 --threshold 0.10 \
      --output-dir outputs/lambda_range_n200 2>&1 | grep -av "it/s\]\|batch/s\]"
  done
done

echo "##### $(date -u) 3/4  crossed directions: is it the direction or the checkpoint?"
# +lambda is applied to IT and -lambda to DPO, so direction and checkpoint are confounded
# in the asymmetry. Steering IT negative and DPO positive separates them.
for p in $PAIRS; do
  python scripts/lambda_range.py --model-config "$(cfg $p)" \
    --eval-config configs/steer_sweep.yaml --side it --sign -1 \
    --n-probe 200 --seed-lambda 0.4 --max-lambda 6.4 --threshold 0.10 \
    --output-dir outputs/lambda_range_n200 2>&1 | grep -av "it/s\]\|batch/s\]"
  python scripts/lambda_range.py --model-config "$(cfg $p)" \
    --eval-config configs/steer_sweep.yaml --side dpo --sign 1 \
    --n-probe 200 --seed-lambda 0.4 --max-lambda 6.4 --threshold 0.10 \
    --output-dir outputs/lambda_range_n200 2>&1 | grep -av "it/s\]\|batch/s\]"
done

echo "##### $(date -u) 4/4  ceilings under relative normalisation"
# lambda currently means "this pair's own mean displacement", and those differ by 4.3x in
# relative size across pairs -- so equal lambda has never been equal intervention.
# Rescaling the vector to the activation norm makes lambda the same physical quantity
# everywhere. If the ceilings collapse together, cross-model comparison becomes legitimate.
sed 's/^vector_normalise: null/vector_normalise: relative/' configs/steer_sweep.yaml \
  > configs/steer_sweep_relative.yaml
grep -q "^vector_normalise: relative" configs/steer_sweep_relative.yaml || { echo "FATAL: config edit failed"; exit 1; }
for p in $PAIRS; do
  for side in it dpo; do
    python scripts/lambda_range.py --model-config "$(cfg $p)" \
      --eval-config configs/steer_sweep_relative.yaml --side $side \
      --n-probe 200 --seed-lambda 0.05 --max-lambda 3.2 --threshold 0.10 \
      --output-dir outputs/lambda_range_relative 2>&1 | grep -av "it/s\]\|batch/s\]"
  done
done

echo "##### $(date -u) results"
for f in outputs/lambda_range_n200/*/*/range.json outputs/lambda_range_relative/*/*/range.json; do
  [ -f "$f" ] && python -c "
import json;d=json.load(open('$f'));p='$f'.split('/')
print(f\"{p[-4]:22s} {p[-3]:12s} {p[-2]:12s} ceiling {d['ceiling']:.3f}  sign {d.get('sign')}\")"
done
echo "QUEUE DONE $(date -u)"
