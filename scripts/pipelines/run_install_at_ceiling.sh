#!/usr/bin/env bash
# The installation arm, read at a ceiling measured for the vector it actually uses.
#
# The first attempt read it at lambda=0.6 and broke 95 of 300 generations. 0.6 came from an
# entry whose arm used a *different* vector: cross-distribution, different norm, and not
# held out. O11's 1.150 ceiling is that vector's too. The same-distribution held-out vector
# is the strong one -- it is the reason O18 reads 90% where the cross-distribution arm reads
# 41% -- and nobody had ever measured how hard it can be pushed before the model falls apart.
#
# A degeneracy ceiling needs no judge: it bisects on the share of generations that fail the
# validity screen, so it costs generation time only. Measuring it first and reading the arm
# there is the difference between a number and a number over collapsed text.
#
#   arms          install / install-random (side it, +lambda at the measured ceiling)
#   eval set      HarmfulQA, the standard 300
#   layer set     27-31, five per-layer vectors
#   construction  mean difference, held out from the evaluation set
#   controls      norm-matched random, same lambda, same layers
#
# The lambda=0.600 run is kept rather than deleted: an over-ceiling point with a measured
# 31.7% break rate is the evidence that the ceiling is real, and it belongs in the appendix
# beside the -0.9/-1.0 removal-side points for the same reason.
set -euo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src
OH=outputs/steer/llama3-oh/harmfulqa
CFG=configs/llama3_oh_local.yaml
DEADLINE_MIN=${1:-150}

# The judge holds ~63 GB and generation needs the 8B model beside it. Free the GPU first.
# vLLM runs as `python -m vllm.entrypoints.openai.api_server`, so pkill on the wrapper
# name leaves it alive holding ~74 GB and the next model load gets 5 GB and dies. Match the
# module, then wait for the memory to actually come back rather than assuming it has.
tmux kill-session -t score 2>/dev/null || true
pkill -f start_judge_server 2>/dev/null || true
pkill -f "vllm.entrypoints" 2>/dev/null || true
for _ in $(seq 1 24); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 8000 ] && break
  sleep 5
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
echo "  VRAM in use after clearing: ${used} MiB"
[ "$used" -lt 8000 ] || { echo "FATAL: something still holds the GPU; not starting"; exit 1; }

echo "=== $(date -u) ceiling search, same-distribution held-out vector ==="
R=outputs/lambda_range_inst/llama3-oh/it_inst_ho300/range.json
if [ ! -f "$R" ]; then
  python scripts/lambda_range.py --model-config $CFG --eval-config configs/sweep_install.yaml \
    --side it --hold-out 300 --tag inst --n-probe 200 --seed-lambda 0.2 --max-lambda 1.2 \
    --threshold 0.10 --output-dir outputs/lambda_range_inst 2>&1 \
    | grep -av "it/s\]\|batch/s\]" | tail -8
fi
CEIL=$(python -c "import json,glob;print(json.load(open(glob.glob('outputs/lambda_range_inst/llama3-oh/*/range.json')[0]))['ceiling'])")
echo "  measured ceiling: $CEIL"
python - "$CEIL" <<'PY'
import sys
c = abs(float(sys.argv[1]))
if not (0.05 <= c <= 1.2):
    raise SystemExit(f"ceiling {c} is outside anything plausible; stopping rather than generating at it")
print(f"  sanity: {c} is in range")
PY

sed "s/^lambdas: .*/lambdas: [$CEIL]/" configs/sweep_install.yaml > configs/sweep_install_ceil.yaml
grep -q "^activations_dir: outputs/layer_profile_harmfulqa" configs/sweep_install_ceil.yaml || { echo FATAL; exit 1; }

for extra in "--tag instc" "--tag instc --random-control"; do
  echo "=== $(date -u) generating $extra ==="
  python scripts/steer_sweep.py --model-config $CFG --eval-config configs/sweep_install_ceil.yaml \
    --side it --hold-out 300 $extra --sync 2>&1 | grep -av "it/s\]\|batch/s\]" | tail -2
done

ARMS="it_instc_ho300 it_instc_ho300_random"
for d in $ARMS; do
  [ -d "$OH/$d" ] || { echo "FATAL: $OH/$d missing after generation"; exit 1; }
  echo "  $d: $(ls "$OH/$d"/*.jsonl | xargs -n1 basename | tr '\n' ' ')"
done

# Same guard as before: seeding across runs proves the generations match a 5.14.1 run, and
# a directory that is not there must fail loudly rather than pass by being skipped.
echo "=== $(date -u) seeding ==="
if [ -d "$OH/it_instc_ho300" ]; then
  python scripts/seed_baseline_scores.py --from "$OH/it_ref" --to "$OH/it_instc_ho300" \
    && echo "  version identity confirmed again" || echo "  !! baseline differs from the 5.14.1 run"
fi
python scripts/seed_baseline_scores.py --from "$OH/it_instc_ho300" --to "$OH/it_instc_ho300_random" \
  || echo "  random will score its own baseline"

REFS=""
[ -f "$OH/it/baseline.jsonl" ] && [ -f "$OH/dpo/baseline.jsonl" ] && \
  REFS="--it-baseline $OH/it/baseline.jsonl --dpo-baseline $OH/dpo/baseline.jsonl"

cat > /workspace/dsteer/score_ceil.sh <<EOS
set -uo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src
DEADLINE=\$(( \$(date +%s) + ${DEADLINE_MIN} * 60 ))
nohup /workspace/venv_judge/bin/python scripts/start_judge_server.py > judge.log 2>&1 &
for _ in \$(seq 1 120); do curl -s -o /dev/null http://localhost:8000/v1/models 2>/dev/null && break; sleep 15; done
curl -s -o /dev/null http://localhost:8000/v1/models 2>/dev/null || { echo "judge never came up"; exit 1; }
echo "[\$(date -u +%H:%M:%S)] judge up"
for d in $ARMS; do
  left=\$(( DEADLINE - \$(date +%s) ))
  [ "\$left" -lt 1200 ] && { echo "only \$((left/60))m left, stopping before \$d"; break; }
  echo "[\$(date -u +%H:%M:%S)] scoring \$d, \$((left/60))m left"
  python scripts/score_sweep.py --sweep-dir "$OH/\$d" \$REFS --concurrency 64 --sync 2>&1 \\
    | grep -avE "httpx|^Activated" | tail -3 || echo "  \$d failed, continuing"
done
echo "CEIL SCORING DONE \$(date -u)"
EOS
tmux new-session -d -s score "bash /workspace/dsteer/score_ceil.sh 2>&1 | tee /workspace/dsteer/score_ceil.log"
echo "scoring in tmux 'score', deadline ${DEADLINE_MIN}m. Safe to disconnect."
