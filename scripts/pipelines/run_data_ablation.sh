#!/usr/bin/env bash
# Two DPO runs from one SFT checkpoint, differing only in the preference data.
#
# Everything measured so far uses one preference set, harmless-base, so "what DPO does" and
# "what safety data does" are not separated -- the objection that closed the last review.
# Holding the SFT checkpoint, the trainer, and every hyperparameter fixed and changing only
# the data is the separation.
#
#   helpful   HH-RLHF helpful-base, labels as annotated. Read 20 examples by hand first:
#             the rejected responses are weaker answers, not refusals, so this does not
#             teach compliance-versus-refusal in either direction. That is a spot check on
#             43,835 examples, not a measurement -- an LLM-judge pass over a few hundred is
#             what would make it a claim.
#   flip      harmless-base with chosen and rejected swapped. Asks whether the direction
#             produced by training *away* from safety is the negation of the forward one or
#             a different direction. cos ~ -1 means alignment is a single signed axis;
#             cos ~ 0 means installing and removing are not the same axis at all.
#
# The recipe is the one the original llama3-oh pair actually ran, recovered from its own
# trainer_state.json rather than from any script default: peak lr 5e-7, warmup ratio 0.05,
# cosine decay, beta 0.1, LoRA r=16 alpha=32 dropout=0.05 on seven projections, one epoch.
# The default in training/dpo.py is 1e-4, which is 200x that and was never what ran.
#
# **The flip run produces a model trained to prefer harmful responses.** Its weights are not
# merged to a public repo, its adapter is not synced, and it is never asked to generate.
# Only activations leave it, and activations are what the comparison needs.
set -euo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src

SFT=/workspace/models/llama3-oh/SFT_merged
GATE_STEP=500
GATE_MIN_ACC=0.52          # the forward harmless run reached 0.580 here; well under it means not learning

echo "=== $(date -u) fetching the SFT checkpoint ==="
python - <<'PY'
from huggingface_hub import snapshot_download
import os
d = "/workspace/models/llama3-oh"
if os.path.isdir(f"{d}/SFT_merged"):
    print("  present")
else:
    snapshot_download(repo_id="sirius5005/SFT-and-DPO", allow_patterns=["SFT_merged/*"], local_dir=d)
    print("  restored")
PY
test -f $SFT/config.json

# One run. $1 tag, $2 data dir, $3 flip flag, $4 "push" or "private".
run_one () {
  local tag=$1 data=$2 flip=$3 release=$4
  local OUT=/workspace/dpo_$tag MERGED=/workspace/models/llama3-oh-$tag
  echo
  echo "############ $(date -u)  $tag : $data  flip=$flip  ############"
  if [ -d "$MERGED" ]; then echo "  $MERGED exists, skipping training"; return 0; fi

  DPO_DATA_DIR=$data DPO_FLIP=$flip DPO_LR=5e-7 DPO_BETA=0.1 \
  DPO_BATCH=2 DPO_ACCUM=8 SFT_MODEL_DIR=$SFT DPO_OUTPUT_DIR=$OUT \
    nohup python training/dpo.py > /workspace/train_$tag.log 2>&1 &
  local pid=$!

  # The gate. A run that is not learning by step 500 will not learn by step 2700, and the
  # forward run's own trajectory is on the hub to compare against, so this is a check
  # against a recorded number rather than a judgement call.
  echo "  waiting for checkpoint-$GATE_STEP to gate on"
  while [ ! -f "$OUT/checkpoint-$GATE_STEP/trainer_state.json" ]; do
    kill -0 $pid 2>/dev/null || { echo "  !! training died before step $GATE_STEP"; tail -20 /workspace/train_$tag.log; return 1; }
    sleep 60
  done
  python - "$OUT/checkpoint-$GATE_STEP/trainer_state.json" "$GATE_MIN_ACC" <<'PY' || { echo "  !! gate failed, killing $tag"; kill $pid 2>/dev/null; return 1; }
import json, sys
lh = json.load(open(sys.argv[1]))["log_history"]
acc = [e["rewards/accuracies"] for e in lh if "rewards/accuracies" in e]
mar = [e["rewards/margins"] for e in lh if "rewards/margins" in e]
print(f"  step {lh[-1].get('step')}: reward_acc {acc[-1]:.3f} (forward harmless run: 0.580), margin {mar[-1]:+.4f}")
if acc[-1] < float(sys.argv[2]):
    raise SystemExit(f"  reward accuracy {acc[-1]:.3f} below {sys.argv[2]} -- not learning the preference")
if mar[-1] <= 0:
    raise SystemExit(f"  margin {mar[-1]:+.4f} is not positive -- preference not being learned")
print("  gate passed")
PY
  echo "  gate passed, letting $tag finish"
  wait $pid || { echo "  !! training failed after the gate"; tail -20 /workspace/train_$tag.log; return 1; }
  test -d $OUT/final_dpo_adapter

  echo "  merging $tag"
  python - "$SFT" "$OUT/final_dpo_adapter" "$MERGED" <<'PY'
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
sft, adapter, merged = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(adapter)
base = AutoModelForCausalLM.from_pretrained(sft, torch_dtype=torch.bfloat16, device_map="cpu")
if base.get_input_embeddings().weight.shape[0] != len(tok):
    base.resize_token_embeddings(len(tok))
PeftModel.from_pretrained(base, adapter).merge_and_unload().save_pretrained(merged, safe_serialization=True)
tok.save_pretrained(merged)
print("  merged, vocab", len(tok))
PY
  test -f $MERGED/config.json

  if [ "$release" = "push" ]; then
    python scripts/sync_results.py --dir $OUT/final_dpo_adapter --experiment dpo_training \
      --model llama3-oh-$tag 2>&1 | tail -1
  else
    echo "  NOT syncing $tag: trained on swapped preferences, weights stay on this box"
  fi

  # Activations are what the geometry comparison needs, and unlike generations they carry
  # no readable harmful text, so both runs are treated the same here.
  cat > configs/llama3_oh_$tag.yaml <<YAML
name: llama3-oh-$tag
architecture: llama
num_layers: 32
base_model: $SFT
it_model: $SFT
it_subfolder: null
dpo_model: $MERGED
dpo_subfolder: null
tokenizer_id: $MERGED
tokenizer_subfolder: null
YAML
  python scripts/layer_profile.py --model-config configs/llama3_oh_$tag.yaml \
    --eval-config configs/layer_profile_harmfulqa.yaml \
    --output-dir outputs/layer_profile_$tag 2>&1 | grep -av "it/s\]\|prompt/s\]" | tail -3
}

run_one helpful helpful-base  0 push
run_one flip    harmless-base 1 private

echo
echo "=== $(date -u) geometry: how do the three directions relate ==="
python - <<'PY'
import torch, itertools, os
LAYERS = [27, 28, 29, 30, 31]
def vec(p, skip=300):
    b = torch.load(p, map_location="cpu")
    it, dpo = b["it"][:, skip:], b["dpo"][:, skip:]
    return {l: (dpo[l].double().mean(0) - it[l].double().mean(0)) for l in LAYERS}
paths = {"harmless (forward)": "outputs/layer_profile_harmfulqa/llama3-oh/activations.pt",
         "helpful":            "outputs/layer_profile_helpful/llama3-oh-helpful/activations.pt",
         "harmless (flipped)": "outputs/layer_profile_flip/llama3-oh-flip/activations.pt"}
vs = {k: vec(p) for k, p in paths.items() if os.path.exists(p)}
print(f"  built {len(vs)} directions: {list(vs)}\n")
for a, b in itertools.combinations(vs, 2):
    c = sum((vs[a][l]/vs[a][l].norm() @ (vs[b][l]/vs[b][l].norm())).item() for l in LAYERS)/len(LAYERS)
    print(f"  cos({a:20s}, {b:20s}) = {c:+.4f}")
print()
for k, v in vs.items():
    print(f"  |v| {k:20s} " + " ".join(f"{v[l].norm().item():6.2f}" for l in LAYERS))
print("\n  flipped vs forward near -1: alignment is one signed axis.")
print("  near 0: training toward safety and away from it are different directions.")
PY
echo "DATA ABLATION DONE $(date -u)"
