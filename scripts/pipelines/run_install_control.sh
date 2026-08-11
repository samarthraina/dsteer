#!/usr/bin/env bash
# Two gaps closed in one session, both on llama3-oh.
#
# **The installation side has never had a random control**, and its arm was never held out:
# the source-matched run reports hold_out=None, so its vector was estimated on a set that
# contains the 300 prompts it is scored on. Controlling a leaked arm does not fix the leak,
# so both are rebuilt here -- held out, with a norm-matched random control at the same
# lambdas. This is the run that decides whether the installation figure is direction-
# specific or the cost of perturbing the stream at all.
#
# **The cross-distribution comparison confounds corpus with readout position.** The
# same-distribution arm was built from layer_profile_harmfulqa (prompt_last); the
# cross-distribution arm defaulted to layer_profile_response_token (response_last), because
# that is what configs/steer_sweep.yaml points at. So "estimating on the evaluation
# distribution triples the effect" currently mixes distribution with token position. The
# cross arm is rebuilt from an HH-RLHF profile read at prompt_last, matching the other side
# on everything except the corpus.
#
#   arms          install / install-random (side it, +lambda)
#                 cross-prompt-last / its random (side dpo, -lambda)
#   lambda        one point per arm, not a grid. score_sweep scores a whole arm directory,
#                 so a grid multiplies the only expensive step by its length -- and a
#                 deadline that lands between an arm and its control leaves the arm
#                 uninterpretable. One point each means every pair either completes or
#                 does not start. 0.6 installing, 0.8 removing: both are points the
#                 existing entries already report, and both sit inside the measured
#                 ceilings (IT side 1.150).
#   eval set      HarmfulQA, the standard 300
#   layer set     27-31, five per-layer vectors, matched to every other arm
#   construction  mean difference, held out from the evaluation set
#   controls      norm-matched random per arm, its own draw at its own norm
#
# Generation is minutes; scoring is ~35 min per file and is what the hours go on. So this
# generates everything, syncs it, and leaves scoring in tmux behind a wall-clock deadline.
# Safe to close the laptop once generation prints its file counts. Scoring is not delegated
# to unattended_scoring.sh: that script waits on a SWEEPSDONE flag nothing here writes, and
# then scores a hardcoded arm list from an earlier session that contains none of these arms.
#
#     bash scripts/pipelines/run_install_control.sh 2>&1 | tee install_control.log
set -euo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src
OH=outputs/steer/llama3-oh/harmfulqa
CFG=configs/llama3_oh_local.yaml
DEADLINE_MIN=${1:-240}
tmux kill-session -t judge 2>/dev/null || true; sleep 5

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
PY

# The HarmfulQA profile is the same-distribution source for both sides. The HH-RLHF one is
# only needed for the matched cross arm, and only if it turns out to be prompt_last -- the
# check is below rather than here, because a response_last file would silently reintroduce
# the confound this script exists to remove.
echo "=== $(date -u) restoring activations ==="
python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os
want = {"outputs/layer_profile_harmfulqa/llama3-oh": "activations/layer_profile_harmfulqa/llama3-oh/activations.pt",
        "outputs/layer_profile_hh/llama3-oh":        "layer_profile_llama3_oh_hh_n2000/activations.pt"}
for dst_dir, src in want.items():
    dst = f"{dst_dir}/activations.pt"
    if os.path.exists(dst):
        print(f"  {dst_dir}: present"); continue
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(hf_hub_download("samarthraina/dsteer-results", src, repo_type="dataset"), dst)
    print(f"  {dst_dir}: restored")
PY

sed -e 's|^activations_dir: .*|activations_dir: outputs/layer_profile_harmfulqa|' \
    -e 's/^lambdas: .*/lambdas: [0.6]/' configs/steer_sweep.yaml > configs/sweep_install.yaml
sed -e 's|^activations_dir: .*|activations_dir: outputs/layer_profile_hh|' \
    -e 's/^lambdas: .*/lambdas: [0.8]/' configs/steer_sweep.yaml > configs/sweep_crosspl.yaml
for f in install:layer_profile_harmfulqa crosspl:layer_profile_hh; do
  grep -q "^activations_dir: outputs/${f#*:}" "configs/sweep_${f%%:*}.yaml" || { echo "FATAL: activations_dir"; exit 1; }
  grep -q "^layers_last_k: 5" "configs/sweep_${f%%:*}.yaml" || { echo "FATAL: layer set"; exit 1; }
done

gen () {   # name  config  side  extra
  echo "=== $(date -u) generating $1 ==="
  python scripts/steer_sweep.py --model-config $CFG --eval-config "$2" --side "$3" \
    --hold-out 300 $4 --sync 2>&1 | grep -av "it/s\]\|batch/s\]" | tail -3
}
gen install        configs/sweep_install.yaml it  "--tag inst"
gen install_random configs/sweep_install.yaml it  "--tag inst --random-control"
gen cross_pl       configs/sweep_crosspl.yaml dpo "--tag crosspl"
gen cross_pl_random configs/sweep_crosspl.yaml dpo "--tag crosspl --random-control"

# run_tag() encodes --hold-out into the run directory, so an arm passed --hold-out 300
# lands in <side>_<tag>_ho300, not <side>_<tag>. Naming these by hand once cost a
# generation pass: every seed and score silently skipped a directory that did not exist.
ARMS="it_inst_ho300 it_inst_ho300_random dpo_crosspl_ho300 dpo_crosspl_ho300_random"

echo "=== $(date -u) seeding baselines, and checking the transformers version while we do ==="
# Two jobs in one guard. The unsteered baseline is the same text in every arm on a side, so
# seeding saves ~35 min of judge time each. But seeding *across runs* also settles something
# the box cannot otherwise tell us: this image carries transformers 5.15.0 and O18/O19 were
# produced on 5.14.1. seed_baseline_scores refuses unless every response matches byte for
# byte, so a pass is proof that the two versions decode identically on this model and these
# prompts -- which is what makes the new cross-distribution arm comparable to O18's
# same-distribution one at all.
#
# A refusal here is not a lost optimisation, it is a finding: it would mean the generations
# differ across versions and no new arm can be read against an old entry. Say so loudly.
python - <<'PY2' || echo "  hub baselines unavailable; arms will score their own"
from huggingface_hub import hf_hub_download
import shutil, os
# it: the post-O14 srcmatch scoring (full-set harmfulness 0.436, one of the two O21b
# identified). dpo: dpo_ho300's, for the same reason -- $OH/dpo carries pre-O14 scores.
srcs = {"it_ref":  "runs/steer_srcmatch_gen/llama3-oh_it/20260806T202107Z/scored/baseline_scored.jsonl",
        "dpo_ref": "runs/steer_gen_v2/llama3-oh_dpo_ho300/20260808T220039Z/scored/baseline_scored.jsonl"}
for arm, src in srcs.items():
    dst = f"outputs/steer/llama3-oh/harmfulqa/{arm}/scored/baseline_scored.jsonl"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy(hf_hub_download("samarthraina/dsteer-results", src, repo_type="dataset"), dst)
    # the seeder compares response text, so it needs the unscored file beside it
    shutil.copy(dst, dst.replace("/scored/baseline_scored.jsonl", "/baseline.jsonl"))
    print(f"  {arm}: ready")
PY2

VERSION_MATCH=yes
for pair in "it_ref:it_inst_ho300" "dpo_ref:dpo_crosspl_ho300"; do
  src=$OH/${pair%%:*}; dst=$OH/${pair##*:}
  # A guard that cannot run must not report success. Skipping here once printed
  # "version-identity check: yes" while testing nothing at all.
  if [ ! -d "$dst" ]; then
    echo "  !! ${pair##*:}: directory missing, cannot check version identity"
    VERSION_MATCH=unchecked; continue
  fi
  if python scripts/seed_baseline_scores.py --from "$src" --to "$dst"; then
    echo "  ${pair##*:}: seeded from a 5.14.1 run -- generations are byte-identical across versions"
  else
    echo "  !! ${pair##*:}: baseline does NOT match the 5.14.1 run."
    echo "  !! transformers 5.15.0 decodes differently. New arms are not comparable to O18/O19."
    echo "  !! Numbers from this run stand on their own; do not put them in a table with O18."
    VERSION_MATCH=no
  fi
done
echo "  version-identity check: $VERSION_MATCH"

# Within a side the randoms share their arm's baseline exactly, so these always match.
python scripts/seed_baseline_scores.py --from "$OH/it_inst_ho300" --to "$OH/it_inst_ho300_random"   || echo "  it_inst_ho300_random: will score its own"
python scripts/seed_baseline_scores.py --from "$OH/dpo_crosspl_ho300" --to "$OH/dpo_crosspl_ho300_random"   || echo "  dpo_crosspl_ho300_random: will score its own"

echo "=== $(date -u) generation done ==="
for d in $ARMS; do [ -d "$OH/$d" ] && echo "  $OH/$d: $(ls "$OH/$d"/*.jsonl 2>/dev/null | wc -l) files"; done

# Steering Shift needs the same two reference files O18 and O19 were scored against, or the
# column is not comparable across entries. load_references reads only the response text, so
# the rubric version behind them does not matter. Missing references are not fatal --
# score_sweep warns and skips that column, and refusal and harmfulness are what this run is
# for.
echo "=== $(date -u) restoring Steering Shift references ==="
python - <<'PY2' || echo "  references unavailable; Steering Shift will be skipped"
from huggingface_hub import hf_hub_download
import shutil, os
refs = {"it":  "runs/steer_harmfulqa/llama3-oh_it/20260806T170229Z/baseline.jsonl",
        "dpo": "runs/steer_harmfulqa/llama3-oh_dpo/20260806T170655Z/baseline.jsonl"}
for arm, src in refs.items():
    dst = f"outputs/steer/llama3-oh/harmfulqa/{arm}/baseline.jsonl"
    if os.path.exists(dst):
        print(f"  {arm}: present"); continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(hf_hub_download("samarthraina/dsteer-results", src, repo_type="dataset"), dst)
    print(f"  {arm}: restored")
PY2
REFS=""
[ -f "$OH/it/baseline.jsonl" ] && [ -f "$OH/dpo/baseline.jsonl" ] &&   REFS="--it-baseline $OH/it/baseline.jsonl --dpo-baseline $OH/dpo/baseline.jsonl"

# Scoring is the whole cost: about 35 min per lambda file, and eight files will not fit an
# hour. So it runs in tmux behind a wall-clock deadline, most important arm first, syncing
# each arm as it finishes rather than at the end. An arm that does not get scored still has
# its generations on the hub and can be scored on any later box for judge time alone.
#
# Ordering is deliberate. install and its random are the pair that decides whether the
# installation figure is direction-specific; without both, neither is worth anything. The
# cross-distribution pair only sharpens a result that already stands.
cat > /workspace/dsteer/score_queue.sh <<EOS
set -uo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src
DEADLINE=\$(( \$(date +%s) + ${DEADLINE_MIN} * 60 ))
OH=$OH
REFS="\${REFS:-$REFS}"

tmux kill-session -t judge 2>/dev/null || true
nohup /workspace/venv_judge/bin/python scripts/start_judge_server.py > judge.log 2>&1 &
echo "[\$(date -u +%H:%M:%S)] waiting for judge"
for _ in \$(seq 1 120); do
  curl -s -o /dev/null http://localhost:8000/v1/models 2>/dev/null && break
  sleep 15
done
curl -s -o /dev/null http://localhost:8000/v1/models 2>/dev/null || { echo "judge never came up, see judge.log"; exit 1; }
echo "[\$(date -u +%H:%M:%S)] judge up"

for d in $ARMS; do
  left=\$(( DEADLINE - \$(date +%s) ))
  if [ "\$left" -lt 1200 ]; then echo "only \$((left/60))m left, stopping before \$d"; break; fi
  [ -d "\$OH/\$d" ] || { echo "skip \$d (missing)"; continue; }
  echo "[\$(date -u +%H:%M:%S)] scoring \$d, \$((left/60))m left"
  python scripts/score_sweep.py --sweep-dir "\$OH/\$d" \$REFS --concurrency 64 --sync 2>&1     | grep -avE "httpx|^Activated" | tail -3 || echo "  \$d failed, continuing"
done
echo "SCOREQUEUE DONE \$(date -u)"
EOS
tmux new-session -d -s score "bash /workspace/dsteer/score_queue.sh 2>&1 | tee /workspace/dsteer/score_queue.log"
echo
echo "generations are on the hub. Scoring runs in tmux 'score', deadline ${DEADLINE_MIN}m."
echo "Safe to disconnect now."
echo "  tail -f /workspace/dsteer/score_queue.log"
echo "  DELETE THE INSTANCE once you see SCOREQUEUE DONE -- idle GPU is the biggest avoidable cost."
