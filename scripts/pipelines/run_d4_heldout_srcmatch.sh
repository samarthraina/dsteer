#!/usr/bin/env bash
# D4: is the source-matched result real, or was it reading its own evaluation set?
#
# The 96% removal measured with a HarmfulQA-built vector used activations from prompts
# 0-1900, and the sweep evaluates on prompts 0-300 of the same shuffle -- the evaluation
# set sits inside the set the vector was estimated from. --hold-out 300 drops those,
# leaving 1600 disjoint prompts to build from.
#
# Its random control runs in the same session. The HarmfulQA vector has a different norm
# and a lower ceiling than the HH-RLHF one, so O17's random baseline does not transfer:
# this needs its own, at matched norm.
set -euo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src

OH=outputs/steer/llama3-oh/harmfulqa
tmux kill-session -t judge 2>/dev/null || true; sleep 10

echo "=== $(date -u) restoring HarmfulQA activations ==="
mkdir -p outputs/layer_profile_harmfulqa/llama3-oh
python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os
d = "outputs/layer_profile_harmfulqa/llama3-oh/activations.pt"
if not os.path.exists(d):
    p = hf_hub_download("samarthraina/dsteer-results",
                        "activations/layer_profile_harmfulqa/llama3-oh/activations.pt",
                        repo_type="dataset")
    shutil.copy(p, d)
print("activations ready")
PY

# The source-matched ceiling measured 0.600 on this side, so the grid stops there.
sed -e 's|^activations_dir: .*|activations_dir: outputs/layer_profile_harmfulqa|' \
    -e 's/^lambdas: .*/lambdas: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]/' \
    configs/steer_sweep.yaml > configs/steer_sweep_d4.yaml
grep -q "^activations_dir: outputs/layer_profile_harmfulqa" configs/steer_sweep_d4.yaml || { echo FATAL; exit 1; }

for extra in "" "--random-control"; do
  echo "=== $(date -u) generating: held-out source-matched ${extra:-real} ==="
  python scripts/steer_sweep.py --model-config configs/llama3_oh_local.yaml \
    --eval-config configs/steer_sweep_d4.yaml --side dpo --hold-out 300 $extra --sync 2>&1 \
    | grep -av "it/s\]\|batch/s\]" | tail -3
done

echo "=== $(date -u) judge up ==="
tmux new-session -d -s judge "cd /workspace/dsteer && source /workspace/venv_judge/bin/activate && python scripts/start_judge_server.py 2>&1 | tee judge.log"
for i in $(seq 1 90); do curl -s http://localhost:8000/v1/models >/dev/null 2>&1 && break; sleep 20; done
curl -s http://localhost:8000/v1/models >/dev/null 2>&1 || { echo "FATAL: judge never came up"; exit 1; }

for d in dpo_ho300 dpo_ho300_random; do
  [ -d "$OH/$d" ] || { echo "SKIP $d"; continue; }
  echo "=== $(date -u) scoring $d ==="
  python scripts/score_sweep.py --sweep-dir "$OH/$d" \
    --it-baseline $OH/it/baseline.jsonl --dpo-baseline $OH/dpo/baseline.jsonl \
    --concurrency 64 --sync 2>&1 | grep -avE "httpx|^Activated" | tail -3
done
echo "D4 DONE $(date -u)"
