#!/usr/bin/env bash
# The selected refusal direction, on the evaluation set, against a matched control.
#
# Layer 14 won on validation (prompts 812-912), from directions built on 300-812. The
# evaluation set is 0-300 and has been seen by neither. Ablation removes the component
# entirely, so there is no strength to sweep -- coefficient 1.0 is the method.
#
# The control ablates a random direction. Norm does not need matching here: ablation
# normalises the direction, so only its orientation matters.
set -euo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src
OH=outputs/steer/llama3-oh/harmfulqa
tmux kill-session -t judge 2>/dev/null || true; sleep 10

sed -e 's|^activations_dir: .*|activations_dir: outputs/refusal_direction|' \
    -e 's/^lambdas: .*/lambdas: [1.0]/' \
    -e 's/^layers_last_k: .*/layers_last_k: 32/' \
    configs/steer_sweep.yaml > configs/steer_sweep_rdabl.yaml

for extra in "" "--random-control"; do
  echo "=== $(date -u) generating ablation ${extra:-real} ==="
  python scripts/steer_sweep.py --model-config configs/llama3_oh_local.yaml \
    --eval-config configs/steer_sweep_rdabl.yaml --side dpo --mode ablate \
    --layers 14 --tag rd14 $extra --sync 2>&1 | grep -av "it/s\]\|batch/s\]" | tail -3
done

echo "=== $(date -u) judge ==="
tmux new-session -d -s judge "cd /workspace/dsteer && source /workspace/venv_judge/bin/activate && python scripts/start_judge_server.py 2>&1 | tee judge.log"
for i in $(seq 1 90); do curl -s http://localhost:8000/v1/models >/dev/null 2>&1 && break; sleep 20; done
curl -s http://localhost:8000/v1/models >/dev/null 2>&1 || { echo FATAL; exit 1; }

for d in dpo_rd14_ablate dpo_rd14_ablate_random; do
  [ -d "$OH/$d" ] || continue
  echo "=== $(date -u) scoring $d ==="
  python scripts/score_sweep.py --sweep-dir "$OH/$d" \
    --it-baseline $OH/it/baseline.jsonl --dpo-baseline $OH/dpo/baseline.jsonl \
    --concurrency 64 --sync 2>&1 | grep -avE "httpx|^Activated" | tail -2
done
echo "RDFINAL DONE $(date -u)"
