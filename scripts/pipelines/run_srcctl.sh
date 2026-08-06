#!/usr/bin/env bash
# Activations on the evaluation distribution, for two things at once.
#
# Vectors are currently estimated on HH-RLHF and applied on HarmfulQA. Two consequences:
# the 30%/12% transfer figures may be partly distribution shift rather than a limit of
# the direction, and alpha(x) cannot be joined to per-prompt outcomes because the prompt
# sets do not overlap. Extracting on HarmfulQA fixes both.
#
# HarmfulQA has no reference responses, so this reads at the last prompt token.
set -uo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src

echo "=== $(date -u) waiting for the norm test ==="
while tmux has-session -t normtest 2>/dev/null; do sleep 60; done

cfg() { case "$1" in llama3-oh) echo configs/llama3_oh_local.yaml ;; *) echo configs/$1.yaml ;; esac; }
for pair in llama3-oh tulu3 olmo3; do
  echo "=== $(date -u) ${pair}: HarmfulQA activations ==="
  python scripts/layer_profile.py \
    --model-config "$(cfg $pair)" \
    --eval-config configs/layer_profile_harmfulqa.yaml \
    --sync 2>&1 | grep -av "it/s\]"
done
echo "SRCCTL DONE $(date -u).  GPU idle -- stop the instance."
