#!/usr/bin/env bash
# Direction against intervention, with nothing else varying.
#
# The earlier 2x2 script compared ablation of one broadcast direction at all 32 layers
# against addition of five per-layer vectors at 27-31. Intervention, layer set and vector
# construction moved together, so no cell could be read as "this direction carries more".
#
# These three arms are the O18 and O19 invocations with the intervention changed and
# nothing else: same activations, same construction, same layers, same prompts, same
# judge. lambdas is [1.0] because ablation has no strength to sweep -- removing the
# component entirely is the method, not a grid point.
#
#                     add                          ablate
#   checkpoint diff   dpo_ho300      (O18, 90.0%)  A
#   refusal direction dpo_refusal    (O19, 60.3%)  B
#   random            each arm's own control       C
#
# One control serves both directions: ablation normalises the direction before projecting
# (steer.py:213), so only the layer set and the per-layer construction matter, and both are
# shared. C draws an independent random direction per layer, matching A and B, which use
# five distinct per-layer vectors. A broadcast arm would need a broadcast control instead.
set -euo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src
OH=outputs/steer/llama3-oh/harmfulqa
tmux kill-session -t judge 2>/dev/null || true; sleep 10

# A fresh box has no checkpoints, and the failure is 30 seconds in rather than at launch.
# The two merged models are fetched by subfolder rather than by cloning the repo: its root
# also holds a DPO LoRA whose base_model_name_or_path points at the trainer's own disk, and
# PEFT follows that adapter whichever subfolder was asked for. ~32 GB, skipped if present.
echo "=== $(date -u) restoring models ==="
python - <<'PY'
from huggingface_hub import snapshot_download
import os
d = "/workspace/models/llama3-oh"
if os.path.isdir(f"{d}/SFT_merged") and os.path.isdir(f"{d}/DPO_merged"):
    print("  models already present")
else:
    snapshot_download(repo_id="sirius5005/SFT-and-DPO",
                      allow_patterns=["SFT_merged/*", "DPO_merged/*"], local_dir=d)
    print("  models restored")
PY

echo "=== $(date -u) restoring activations ==="
python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os
for d in ("layer_profile_harmfulqa", "refusal_direction"):
    dst = f"outputs/{d}/llama3-oh/activations.pt"
    if os.path.exists(dst):
        print(f"{d}: already present"); continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    p = hf_hub_download("samarthraina/dsteer-results",
                        f"activations/{d}/llama3-oh/activations.pt", repo_type="dataset")
    shutil.copy(p, dst)
    print(f"{d}: restored")
PY

# Each config differs from the one its addition arm used only in the lambda grid.
sed -e 's|^activations_dir: .*|activations_dir: outputs/layer_profile_harmfulqa|' \
    -e 's/^lambdas: .*/lambdas: [1.0]/' configs/steer_sweep.yaml > configs/sweep_cp_abl.yaml
sed -e 's|^activations_dir: .*|activations_dir: outputs/refusal_direction|' \
    -e 's/^lambdas: .*/lambdas: [1.0]/' configs/steer_sweep.yaml > configs/sweep_rd_abl.yaml
for f in cp:layer_profile_harmfulqa rd:refusal_direction; do
  grep -q "^activations_dir: outputs/${f#*:}" "configs/sweep_${f%%:*}_abl.yaml" || { echo FATAL; exit 1; }
  grep -q "^layers_last_k: 5" "configs/sweep_${f%%:*}_abl.yaml" || { echo FATAL; exit 1; }
done

gen () {   # name  config  extra
  echo "=== $(date -u) generating $1 ==="
  python scripts/steer_sweep.py --model-config configs/llama3_oh_local.yaml \
    --eval-config "$2" --side dpo --mode ablate $3 --sync 2>&1 \
    | grep -av "it/s\]\|batch/s\]" | tail -2
}
gen A configs/sweep_cp_abl.yaml "--hold-out 300 --tag cpsame"
gen B configs/sweep_rd_abl.yaml "--tag refusal"
gen C configs/sweep_rd_abl.yaml "--tag refusal --random-control"

ARMS="dpo_cpsame_ablate_ho300 dpo_refusal_ablate dpo_refusal_ablate_random"

# The unsteered DPO baseline is the same text in every arm -- greedy, same prompts, hooks
# inactive -- so judging it once per arm is about 35 minutes of judge time each for a file
# we already have. The seeder refuses unless every response matches, and score_sweep then
# resumes by id and spends nothing on it. A refusal is not fatal: it costs time, not
# correctness, so the run continues and scores the baseline itself.
#
# Seed from dpo_ho300, not from $OH/dpo. Identical text is necessary and not sufficient --
# the *scores* attached to it must be post-O14, and $OH/dpo carries the restored baseline
# judged under the rubric that scored harmfulness without the request. Seeding from it
# would attach old-rubric harmfulness to corrected arms, which is precisely how O8 ended
# up incomparable to O13 and O18 on the identical subset.
echo "=== $(date -u) seeding baselines ==="
SEED=$OH/dpo_ho300
python - <<'PY' || echo "  seed source unavailable; arms will score their own baselines"
from huggingface_hub import hf_hub_download
import shutil, os
run = "runs/steer_gen_v2/llama3-oh_dpo_ho300/20260808T220039Z"
for rel in ("baseline.jsonl", "scored/baseline_scored.jsonl"):
    dst = f"outputs/steer/llama3-oh/harmfulqa/dpo_ho300/{rel}"
    if os.path.exists(dst):
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(hf_hub_download("samarthraina/dsteer-results", f"{run}/{rel}",
                                repo_type="dataset"), dst)
print("  seed source ready (post-O14)")
PY
for d in $ARMS; do
  python scripts/seed_baseline_scores.py --from "$SEED" --to "$OH/$d" || echo "  $d: scoring its own baseline"
done

echo "=== $(date -u) judge ==="
tmux new-session -d -s judge "cd /workspace/dsteer && source /workspace/venv_judge/bin/activate && python scripts/start_judge_server.py 2>&1 | tee judge.log"
for i in $(seq 1 90); do curl -s http://localhost:8000/v1/models >/dev/null 2>&1 && break; sleep 20; done
curl -s http://localhost:8000/v1/models >/dev/null 2>&1 || { echo FATAL; exit 1; }

# --dpo-baseline stays $OH/dpo despite the rule above: load_references reads only the
# response text, never the scores, so the rubric version behind it is irrelevant here.
# It is the same reference O18 and O19 used, which is what keeps Steering Shift comparable
# across the row.
for d in $ARMS; do
  [ -d "$OH/$d" ] || { echo "missing $d"; continue; }
  echo "=== $(date -u) scoring $d ==="
  python scripts/score_sweep.py --sweep-dir "$OH/$d" \
    --it-baseline $OH/it/baseline.jsonl --dpo-baseline $OH/dpo/baseline.jsonl \
    --concurrency 64 --sync 2>&1 | grep -avE "httpx|^Activated" | tail -2
done
echo "2x2 MATCHED DONE $(date -u)"
