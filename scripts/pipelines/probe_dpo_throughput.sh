#!/usr/bin/env bash
# Measure DPO throughput before committing to a full epoch.
#
# harmless-base sequences are short -- median 114 tokens, none over 1024 -- so the
# released batch size of 2 with gradient checkpointing on leaves most of an 80 GB card
# idle. Effective batch stays 16 in every setting below, so the optimisation is identical
# and only speed and memory differ.
set -uo pipefail
cd /workspace/dsteer/training
source /venv/main/bin/activate
SFT=/workspace/models/openhermes-mistral-sft

for cfg in "2 8 1" "8 2 1" "8 2 0" "16 1 0"; do
  set -- $cfg
  B=$1 A=$2 CK=$3
  echo "=== batch=$B accum=$A grad_ckpt=$CK ==="
  SFT_MODEL_DIR=$SFT DPO_LR=5e-6 HF_PUSH_REPO="" \
  DPO_BATCH=$B DPO_ACCUM=$A DPO_GRAD_CKPT=$CK DPO_MAX_STEPS=12 \
  DPO_OUTPUT_DIR=/workspace/probe_b${B}_a${A}_c${CK} \
    timeout 900 python dpo.py 2>&1 | grep -aoE "'train_runtime': [0-9.]+|'train_samples_per_second': [0-9.]+|OutOfMemoryError" | tail -3
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
  rm -rf /workspace/probe_b${B}_a${A}_c${CK}
done
echo "PROBE DONE"
