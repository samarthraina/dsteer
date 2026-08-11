#!/usr/bin/env bash
# A second pair: OpenHermes-2.5-Mistral-7B + DPO on HH-RLHF harmless-base.
#
# Everything measured about removable safety so far rests on one pair, because two of the
# three public pairs reach their preference stage already safe and have no safety axis to
# steer along. This trains one that does.
#
# Two deliberate differences from the first pair. A Mistral base rather than Llama-3, so
# the replication is independent of it. And 5e-6 rather than 1e-4 -- the conventional DPO
# range, where the same 1e-4 left a sibling model degenerate. The instruction data stays
# OpenHermes-2.5, which keeps the comparison controlled.
#
# Throughput is measured rather than assumed: harmless-base sequences are short (median
# 114 tokens), so the released batch size of 2 with checkpointing on leaves most of an
# 80 GB card idle. Effective batch is 16 in every candidate, so the optimisation is
# identical across them and only speed differs.
set -euo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src

SFT=/workspace/models/openhermes-mistral-sft
OUT=/workspace/dpo_output_mistral
MERGED=/workspace/models/openhermes-mistral-dpo

echo "=== $(date -u) waiting for the scoring run ==="
while tmux has-session -t screen 2>/dev/null; do sleep 60; done

echo "=== $(date -u) 1/5  fetch the SFT checkpoint ==="
tmux kill-session -t judge 2>/dev/null || true; sleep 10
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="teknium/OpenHermes-2.5-Mistral-7B",
                  local_dir="/workspace/models/openhermes-mistral-sft",
                  ignore_patterns=["*.bin", "*.pth", "*.gguf"], max_workers=8)
PY
test -f $SFT/config.json

echo "=== $(date -u) 2/5  throughput probe ==="
BEST_B=2; BEST_A=8; BEST_CK=1; BEST_SPS=0

# An explicit config skips the probe entirely.
#
# The probe reads samples/s out of the trainer's stdout, and when that line does not
# appear it cannot distinguish a config that failed from one that ran fine and printed
# nothing it recognised. Every candidate then reports as failed, the fallback is the
# slowest entry on the list, and the run proceeds looking healthy -- which is how a
# 7B LoRA ended up using 23 GB of an 80 GB card and taking four hours instead of one.
# The reporting below now says which of the two it saw.
if [ -n "${DPO_BATCH:-}" ] && [ -n "${DPO_ACCUM:-}" ] && [ -n "${DPO_GRAD_CKPT:-}" ]; then
  BEST_B=$DPO_BATCH; BEST_A=$DPO_ACCUM; BEST_CK=$DPO_GRAD_CKPT
  echo "  given explicitly: batch=$BEST_B accum=$BEST_A ckpt=$BEST_CK -- probe skipped"
else
for cfg in "2 8 1" "8 2 1" "8 2 0" "16 1 0"; do
  set -- $cfg; B=$1; A=$2; CK=$3
  OUT_PROBE=/workspace/probe_$B$A$CK
  LOG_PROBE=/workspace/probe_$B$A$CK.log
  SFT_MODEL_DIR=$SFT DPO_LR=5e-6 HF_PUSH_REPO="" \
    DPO_BATCH=$B DPO_ACCUM=$A DPO_GRAD_CKPT=$CK DPO_MAX_STEPS=12 \
    DPO_OUTPUT_DIR=$OUT_PROBE \
    timeout 1200 python training/dpo.py > "$LOG_PROBE" 2>&1 && RC=0 || RC=$?
  SPS=$(grep -aoE "train_samples_per_second[^0-9]+[0-9.]+" "$LOG_PROBE" | tail -1 | grep -oE "[0-9.]+$" || true)
  rm -rf "$OUT_PROBE"
  if [ -n "${SPS:-}" ]; then
    echo "  batch=$B accum=$A ckpt=$CK  ->  ${SPS} samples/s"
    if awk "BEGIN{exit !($SPS > $BEST_SPS)}"; then BEST_B=$B; BEST_A=$A; BEST_CK=$CK; BEST_SPS=$SPS; fi
  elif [ "$RC" -ne 0 ]; then
    echo "  batch=$B accum=$A ckpt=$CK  ->  exit $RC: $(grep -aoiE 'out of memory|CUDA error|[A-Za-z]*Error' "$LOG_PROBE" | tail -1)"
  else
    echo "  batch=$B accum=$A ckpt=$CK  ->  ran cleanly but printed no throughput line (see $LOG_PROBE)"
  fi
done
echo "chosen: batch=$BEST_B accum=$BEST_A ckpt=$BEST_CK at ${BEST_SPS} samples/s"
fi

echo "=== $(date -u) 3/5  DPO on harmless-base, lr 5e-6 ==="
# Resumable at the adapter. Everything after this point is minutes of CPU and a short
# screen, and it has failed there once -- retrying the tail should not cost the five hours
# that produced the adapter.
if [ -d "$OUT/final_dpo_adapter" ]; then
  echo "  adapter already present at $OUT/final_dpo_adapter -- skipping training"
else
  SFT_MODEL_DIR=$SFT DPO_LR=5e-6 DPO_OUTPUT_DIR=$OUT HF_PUSH_REPO="" \
  DPO_BATCH=$BEST_B DPO_ACCUM=$BEST_A DPO_GRAD_CKPT=$BEST_CK \
    python training/dpo.py 2>&1 | grep -avE "it/s\]|B/s\]" | tail -60
fi
test -d $OUT/final_dpo_adapter

echo "=== $(date -u) 4/5  merge, and keep the run on the hub ==="
# The trainer adds a pad token the SFT checkpoint does not have, so the adapter carries a
# resized embedding table -- 32003 rows against the base's 32002. Loading the base at its
# own size then fails on a one-row mismatch. Resize before attaching the adapter, and save
# the *training* tokenizer rather than the SFT one, or the merged model ships with a
# vocabulary one token shorter than its own embedding matrix.
python - <<PY
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("$OUT/final_dpo_adapter")
base = AutoModelForCausalLM.from_pretrained("$SFT", torch_dtype=torch.bfloat16, device_map="cpu")
if base.get_input_embeddings().weight.shape[0] != len(tok):
    print(f"  resizing embeddings {base.get_input_embeddings().weight.shape[0]} -> {len(tok)}")
    base.resize_token_embeddings(len(tok))
PeftModel.from_pretrained(base, "$OUT/final_dpo_adapter").merge_and_unload().save_pretrained(
    "$MERGED", safe_serialization=True)
tok.save_pretrained("$MERGED")
print("  merged vocab", len(tok))
PY
test -f $MERGED/config.json
# The adapter and the training history are small and not reproducible without the run.
python scripts/sync_results.py --dir $OUT/final_dpo_adapter --experiment dpo_training \
  --model openhermes-mistral-adapter 2>&1 | tail -1
cp $OUT/training_history.json $OUT/training_metrics.csv /workspace/dsteer/outputs/ 2>/dev/null || true

echo "=== $(date -u) 5/5  screen the trained checkpoint ==="
python scripts/screen_model.py --model $MERGED \
  --output-dir outputs/screen/openhermes-mistral-dpo --n 150 2>&1 | grep -av "it/s\]\|batch/s\]"

echo "TRAIN DONE $(date -u).  Judge is down; scoring the screen is the next step."
