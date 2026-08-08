#!/usr/bin/env bash
# The comparison as a 2x2, so direction and intervention are not confounded.
#
# O19 set a checkpoint-difference vector applied by ADDITION against a refusal direction
# applied by ADDITION, and the selected refusal direction uses ABLATION. Comparing the two
# best arms as they stand would mix "which direction" with "which intervention". Running
# both directions under ablation completes the square:
#
#                     add            ablate
#   checkpoint diff   done (O18)     this run
#   refusal direction done (O19)     this run
#
# Both ablation arms use one direction applied at every layer, which is what the
# validation selection measured. Ablation normalises the direction, so a random control
# needs no norm matching -- only orientation matters.
set -euo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src
OH=outputs/steer/llama3-oh/harmfulqa
tmux kill-session -t judge 2>/dev/null || true; sleep 10

# The checkpoint-difference vector, same-distribution and held out, broadcast to all layers
# from its strongest layer so the two ablation arms are constructed alike.
python - <<'PY'
import sys, torch; sys.path.insert(0, "src")
from steering.steer import build_vectors
v = build_vectors("outputs/layer_profile_harmfulqa/llama3-oh/activations.pt",
                  method="mean", layers=[31], skip_first=300)[31]
torch.save({l: v.clone() for l in range(32)}, "outputs/layer_profile_harmfulqa/llama3-oh/v_L31_all_layers.pt")
print("checkpoint-difference vector broadcast, norm", round(v.norm().item(), 3))
PY

sed -e 's/^lambdas: .*/lambdas: [1.0]/' configs/steer_sweep.yaml > configs/steer_sweep_abl.yaml

run () {   # name  vectors.pt  extra
  echo "=== $(date -u) $1 ==="
  python scripts/steer_sweep.py --model-config configs/llama3_oh_local.yaml \
    --eval-config configs/steer_sweep_abl.yaml --side dpo --mode ablate \
    --vectors "$2" --tag "$1" $3 --sync 2>&1 | grep -av "it/s\]\|batch/s\]" | tail -2
}
run rd14  outputs/refusal_direction/llama3-oh/v_L14_all_layers.pt ""
run rd14  outputs/refusal_direction/llama3-oh/v_L14_all_layers.pt "--random-control"
run cp31  outputs/layer_profile_harmfulqa/llama3-oh/v_L31_all_layers.pt ""

echo "=== $(date -u) judge ==="
tmux new-session -d -s judge "cd /workspace/dsteer && source /workspace/venv_judge/bin/activate && python scripts/start_judge_server.py 2>&1 | tee judge.log"
for i in $(seq 1 90); do curl -s http://localhost:8000/v1/models >/dev/null 2>&1 && break; sleep 20; done
curl -s http://localhost:8000/v1/models >/dev/null 2>&1 || { echo FATAL; exit 1; }

for d in dpo_rd14_ablate_learned dpo_rd14_ablate_learned_random dpo_cp31_ablate_learned; do
  [ -d "$OH/$d" ] || { echo "missing $d"; continue; }
  echo "=== $(date -u) scoring $d ==="
  python scripts/score_sweep.py --sweep-dir "$OH/$d" \
    --it-baseline $OH/it/baseline.jsonl --dpo-baseline $OH/dpo/baseline.jsonl \
    --concurrency 64 --sync 2>&1 | grep -avE "httpx|^Activated" | tail -2
done
echo "2x2 DONE $(date -u)"
