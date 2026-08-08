#!/usr/bin/env bash
# D3: does the conditioned subset manufacture the safety effect?
#
# Every safety number here is computed on prompts selected because the two unsteered
# checkpoints differ on refusal by at least 0.3 -- and the same scored generation that
# drove the selection then serves as the baseline the steered value is compared against.
# That is the setup for regression to the mean: prompts picked for an extreme gap are
# partly picked for noise, and any re-measurement drifts back.
#
# A norm-matched random direction cannot transfer safety. So whatever it moves on that
# same subset is the artefact, and the real effect is only what exceeds it. Same
# checkpoint, same prompts, same grid, same conditioning as O13.
set -euo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src

OH=outputs/steer/llama3-oh/harmfulqa
sed 's/^lambdas: .*/lambdas: [0.1, 0.2, 0.3, 0.4, 0.6, 0.8]/' configs/steer_sweep.yaml > configs/steer_sweep_d3.yaml

echo "=== $(date -u) generating: random direction, DPO side ==="
python scripts/steer_sweep.py --model-config configs/llama3_oh_local.yaml \
  --eval-config configs/steer_sweep_d3.yaml --side dpo --random-control --sync 2>&1 \
  | grep -av "it/s\]\|batch/s\]" | tail -4

echo "=== $(date -u) judge up ==="
tmux new-session -d -s judge "cd /workspace/dsteer && source /workspace/venv_judge/bin/activate && python scripts/start_judge_server.py 2>&1 | tee judge.log"
for i in $(seq 1 90); do curl -s http://localhost:8000/v1/models >/dev/null 2>&1 && break; sleep 20; done
curl -s http://localhost:8000/v1/models >/dev/null 2>&1 || { echo "FATAL: judge never came up"; exit 1; }

echo "=== $(date -u) scoring ==="
python scripts/score_sweep.py --sweep-dir $OH/dpo_random \
  --it-baseline $OH/it/baseline.jsonl --dpo-baseline $OH/dpo/baseline.jsonl \
  --concurrency 64 --sync 2>&1 | grep -avE "httpx|^Activated" | tail -4

echo "D3 DONE $(date -u)"
