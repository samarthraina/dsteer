#!/usr/bin/env bash
# Three config-only runs. No new code, so all three are safe to leave unattended.
#
# 1. Vectors built on the evaluation distribution. Everything so far estimates v on
#    HH-RLHF and applies it to HarmfulQA, so part of the 41% ceiling may be distribution
#    shift rather than a limit of the direction.
# 2. Two points past the measured ceiling, reported with n_valid. Not to claim more
#    removal -- past the ceiling the survivors are the easy prompts -- but to show the
#    curve breaking down, which is what makes 0.8 a bound rather than a stopping point.
# 3. Relative normalisation again. The first attempt seeded at 0.05 when the ceilings sit
#    near 0.01-0.13, so the search had no room and one pair floored at zero.
#
# Syncs are explicit after every stage: --sync on layer_profile silently did nothing
# twice today, and the results only survived because the gap was noticed by hand.
set -uo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src

echo "=== $(date -u) freeing the card ==="
tmux kill-session -t judge 2>/dev/null; sleep 15
nvidia-smi --query-gpu=memory.used --format=csv,noheader

OH=configs/llama3_oh_local.yaml
sync() { python scripts/sync_results.py --dir "$1" --experiment "$2" --model "$3" 2>&1 | tail -1; }

echo "##### $(date -u) 1/3  vectors built on the evaluation distribution"
sed 's|^activations_dir: .*|activations_dir: outputs/layer_profile_harmfulqa|' \
  configs/steer_sweep.yaml > configs/steer_sweep_srcmatch.yaml
grep -q "^activations_dir: outputs/layer_profile_harmfulqa" configs/steer_sweep_srcmatch.yaml \
  || { echo "FATAL: config edit failed"; exit 1; }
for side in dpo it; do
  python scripts/lambda_range.py --model-config $OH --eval-config configs/steer_sweep_srcmatch.yaml \
    --side $side --n-probe 200 --seed-lambda 0.4 --max-lambda 6.4 --threshold 0.10 \
    --output-dir outputs/lambda_range_srcmatch 2>&1 | grep -av "it/s\]\|batch/s\]"
  sync outputs/lambda_range_srcmatch/llama3-oh/$side lambda_range_srcmatch llama3-oh
done
for side in dpo it; do
  R=outputs/lambda_range_srcmatch/llama3-oh/$side/range.json
  [ -f "$R" ] || continue
  G=$(python -c "import json;print(','.join(f'{abs(v):.3f}' for v in json.load(open('$R'))['suggested_grid']))")
  [ -n "$G" ] || continue
  sed "s/^lambdas: .*/lambdas: [$G]/" configs/steer_sweep_srcmatch.yaml > configs/tmp_src_$side.yaml
  sed -i "s|^output_dir: .*|output_dir: outputs/steer_srcmatch|" configs/tmp_src_$side.yaml
  echo "--- sweep $side  lambdas $G"
  python scripts/steer_sweep.py --model-config $OH --eval-config configs/tmp_src_$side.yaml \
    --side $side --sync 2>&1 | grep -av "it/s\]\|batch/s\]" | tail -3
done

echo "##### $(date -u) 2/3  two points past the ceiling"
sed 's/^lambdas: .*/lambdas: [0.9, 1.0]/' configs/steer_sweep.yaml > configs/steer_sweep_past.yaml
python scripts/steer_sweep.py --model-config $OH --eval-config configs/steer_sweep_past.yaml \
  --side dpo --sync 2>&1 | grep -av "it/s\]\|batch/s\]" | tail -3

echo "##### $(date -u) 3/3  relative normalisation, seeded finer"
sed 's/^vector_normalise: null/vector_normalise: relative/' configs/steer_sweep.yaml \
  > configs/steer_sweep_rel2.yaml
grep -q "^vector_normalise: relative" configs/steer_sweep_rel2.yaml || { echo "FATAL"; exit 1; }
for pair in llama3-oh tulu3 olmo3; do
  case $pair in llama3-oh) C=$OH;; *) C=configs/$pair.yaml;; esac
  for side in it dpo; do
    python scripts/lambda_range.py --model-config $C --eval-config configs/steer_sweep_rel2.yaml \
      --side $side --n-probe 200 --seed-lambda 0.01 --max-lambda 1.28 --threshold 0.10 \
      --tolerance 0.01 --output-dir outputs/lambda_range_rel2 2>&1 | grep -av "it/s\]\|batch/s\]"
    sync outputs/lambda_range_rel2/$pair/$side lambda_range_rel2 $pair
  done
done

echo "=== $(date -u) results ==="
for f in outputs/lambda_range_srcmatch/*/*/range.json outputs/lambda_range_rel2/*/*/range.json; do
  [ -f "$f" ] && python -c "
import json;d=json.load(open('$f'));p='$f'.split('/')
print(f\"{p[-4]:26s} {p[-3]:11s} {p[-2]:6s} ceiling {d['ceiling']:.3f}\")"
done
echo "THREE DONE $(date -u).  GPU idle -- stop the instance."
