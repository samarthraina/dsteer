#!/usr/bin/env bash
# Run every entry point once, on a tiny random model, before anything touches a rented GPU.
#
# The two failures that cost real time this week were both invisible to unit tests and to
# reading. A missing tokenizer dependency and an absent tensorboard package each killed a
# launch after the box was already running, and a run directory that did not encode which
# vector produced it silently returned an earlier run's generations. All three would have
# been caught by executing each script once.
#
# Nothing here needs a GPU or a real checkpoint. It takes a couple of minutes.
#
#     bash scripts/smoke_test.sh /tmp/smoke
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-/tmp/smoke}"
export PYTHONPATH="$ROOT/src"
mkdir -p "$WORK"
cd "$ROOT"

TINY=hf-internal-testing/tiny-random-LlamaForCausalLM
pass=0; fail=0
step() {
  local name="$1"; shift
  if "$@" > "$WORK/${name}.log" 2>&1; then
    echo "  ok    $name"; pass=$((pass+1))
  else
    echo "  FAIL  $name   -> $WORK/${name}.log"; tail -3 "$WORK/${name}.log" | sed 's/^/        /'
    fail=$((fail+1))
  fi
}

cat > "$WORK/tiny.yaml" <<YAML
name: tiny
architecture: llama
num_layers: 2
base_model: $TINY
it_model: $TINY
it_subfolder: null
dpo_model: $TINY
dpo_subfolder: null
tokenizer_id: $TINY
tokenizer_subfolder: null
YAML

cat > "$WORK/sweep.yaml" <<YAML
prompt_source: harmfulqa
n_prompts: 4
lambdas: [0.5, 1.0]
layers_last_k: 2
vector_method: mean
vector_normalise: null
positions: all
preserve_norm: false
activations_dir: $WORK/acts
max_new_tokens: 8
batch_size: 2
max_input_length: 64
output_dir: $WORK/steer
YAML

echo "unit tests"
step pytest python -m pytest tests/ -q

echo "activations and vectors"
step refusal_direction python scripts/refusal_direction.py --model-config "$WORK/tiny.yaml" \
  --side dpo --n 6 --layers-last-k 2 --max-input-length 64 --output-dir "$WORK/refusal"
mkdir -p "$WORK/acts"
cp -r "$WORK/refusal/tiny" "$WORK/acts/tiny" 2>/dev/null || true

step train_vector python scripts/train_steering_vector.py --model-config "$WORK/tiny.yaml" \
  --side it --output-dir "$WORK/learned" --layers-last-k 2 --steps 4 --batch-size 2 \
  --n-pairs 8 --max-length 64 --log-every 2

echo "ceiling search"
step lambda_range          python scripts/lambda_range.py --model-config "$WORK/tiny.yaml" \
  --eval-config "$WORK/sweep.yaml" --side it --n-probe 4 --seed-lambda 0.5 --max-lambda 1 --output-dir "$WORK/lr"
step lambda_range_layers   python scripts/lambda_range.py --model-config "$WORK/tiny.yaml" \
  --eval-config "$WORK/sweep.yaml" --side it --layers 1 --n-probe 4 --seed-lambda 0.5 --max-lambda 1 --output-dir "$WORK/lr"
step lambda_range_holdout  python scripts/lambda_range.py --model-config "$WORK/tiny.yaml" \
  --eval-config "$WORK/sweep.yaml" --side it --hold-out 2 --n-probe 4 --seed-lambda 0.5 --max-lambda 1 --output-dir "$WORK/lr"
step lambda_range_vectors  python scripts/lambda_range.py --model-config "$WORK/tiny.yaml" \
  --eval-config "$WORK/sweep.yaml" --side it --vectors "$WORK/learned/tiny/vectors.pt" \
  --n-probe 4 --seed-lambda 0.5 --max-lambda 1 --output-dir "$WORK/lr"

echo "sweeps"
step steer_sweep          python scripts/steer_sweep.py --model-config "$WORK/tiny.yaml" --eval-config "$WORK/sweep.yaml" --side it
step steer_sweep_vectors  python scripts/steer_sweep.py --model-config "$WORK/tiny.yaml" --eval-config "$WORK/sweep.yaml" --side it \
  --vectors "$WORK/learned/tiny/vectors.pt"
step steer_sweep_ablate   python scripts/steer_sweep.py --model-config "$WORK/tiny.yaml" --eval-config "$WORK/sweep.yaml" --side it \
  --mode ablate --tag abl
step steer_sweep_ablate_v python scripts/steer_sweep.py --model-config "$WORK/tiny.yaml" --eval-config "$WORK/sweep.yaml" --side it \
  --mode ablate --tag ablv --vectors "$WORK/learned/tiny/vectors.pt"
step steer_sweep_ablate_dpo python scripts/steer_sweep.py --model-config "$WORK/tiny.yaml" --eval-config "$WORK/sweep.yaml" --side dpo \
  --mode ablate --tag abl
step screen_model         python scripts/screen_model.py --model "$TINY" --output-dir "$WORK/screen" --n 4 --max-new-tokens 8

# A guard passes by refusing. Seeding a scored baseline from an arm that has none must
# fail rather than silently produce an unscored one.
echo "guards"
if python scripts/seed_baseline_scores.py --from "$WORK/steer/tiny/harmfulqa/it" \
     --to "$WORK/steer/tiny/harmfulqa/it_abl_ablate" > "$WORK/seed_baseline.log" 2>&1; then
  echo "  FAIL  seed_baseline_refuses   copied without a scored source"; fail=$((fail+1))
else
  echo "  ok    seed_baseline_refuses"; pass=$((pass+1))
fi

# A learned vector and a derived one must not share a directory, or the second run resumes
# on the first's generations and the comparison is between a vector and itself. Counting
# is what catches a collision: five runs that produce four directories have merged two.
echo "run directories are distinct"
S="$WORK/steer/tiny/harmfulqa"
WANT="it it_learned it_abl_ablate it_ablv_ablate_learned dpo_abl_ablate"
missing=""
for d in $WANT; do [ -d "$S/$d" ] || missing="$missing $d"; done
n_want=$(echo $WANT | wc -w); n_got=$(ls "$S" 2>/dev/null | wc -l)
if [ -z "$missing" ] && [ "$n_got" -eq "$n_want" ]; then
  echo "  ok    $n_want runs -> $n_got distinct directories"; pass=$((pass+1))
else
  echo "  FAIL  run directories collided or missing:${missing:- none}"
  echo "        wanted $n_want, have $n_got: $(ls "$S" 2>/dev/null | tr '\n' ' ')"
  fail=$((fail+1))
fi

# Ablation removes a component, which is the same operation on either checkpoint. The -1
# the sweep applies on the DPO side is right for addition and inverts ablation, where a
# negative coefficient adds the component back instead of taking it out. That degenerated
# 293 of 300 generations before it was caught, and it is invisible until the text is read.
echo "ablation keeps a positive coefficient on the DPO side"
if [ -z "$(ls "$S/dpo_abl_ablate"/lambda_-*.jsonl 2>/dev/null)" ] \
   && [ -n "$(ls "$S/dpo_abl_ablate"/lambda_+*.jsonl 2>/dev/null)" ]; then
  echo "  ok    dpo ablate coefficients are positive"; pass=$((pass+1))
else
  echo "  FAIL  dpo ablate produced: $(ls "$S/dpo_abl_ablate" 2>/dev/null | tr '\n' ' ')"
  fail=$((fail+1))
fi

# The analysis entry points need scored records, which the judge is not run here to make.
# A fixture is enough: what breaks in these two is argument handling and the arithmetic of
# an empty or partial intersection, not the numbers.
echo "analysis on a fixture"
python - "$WORK/an" <<'PY' > "$WORK/fixture.log" 2>&1
import json, os, random, sys, torch
W = sys.argv[1]; random.seed(0); torch.manual_seed(0)
def dump(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows))
def rec(i, r, h, valid=True):
    return {"id": f"p{i}", "response": "x" * 40, "valid": valid,
            "refusal_score": r if valid else None,
            "harmfulness_score": h if valid else None}
it  = [rec(i, 0.2, 0.6) for i in range(40)]
dpo = [rec(i, 0.9, 0.1) for i in range(40)]
dump(f"{W}/it/scored/baseline_scored.jsonl", it)
dump(f"{W}/dpo/scored/baseline_scored.jsonl", dpo)
for tag, s in (("arm", 0.7), ("ctl", 0.2)):
    dump(f"{W}/{tag}/scored/lambda_-0.600_scored.jsonl",
         [rec(i, 0.9 - s * 0.7 + random.gauss(0, .05),
                 0.1 + s * 0.5 + random.gauss(0, .05), valid=i > 2) for i in range(40)])
b = torch.randn(32, 60, 16)
torch.save({"it": b, "dpo": b + torch.randn(32, 1, 16) * .2}, f"{W}/acts.pt")
PY
step conditioned_analysis python scripts/conditioned_analysis.py \
  --it-scored "$WORK/an/it/scored/baseline_scored.jsonl" \
  --dpo-scored "$WORK/an/dpo/scored/baseline_scored.jsonl" \
  --arm "a=$WORK/an/arm" --control "$WORK/an/ctl" --threshold 0.3 --output "$WORK/an/out.json"
step conditioned_analysis_install python scripts/conditioned_analysis.py \
  --it-scored "$WORK/an/it/scored/baseline_scored.jsonl" \
  --dpo-scored "$WORK/an/dpo/scored/baseline_scored.jsonl" \
  --arm "a=$WORK/an/arm" --direction install --strict-common
step conditioned_analysis_compare python scripts/conditioned_analysis.py \
  --it-scored "$WORK/an/it/scored/baseline_scored.jsonl" \
  --dpo-scored "$WORK/an/dpo/scored/baseline_scored.jsonl" \
  --arm "a=$WORK/an/arm" --arm "b=$WORK/an/ctl" --compare "a,b"
step direction_dispersion python scripts/direction_dispersion.py \
  --checkpoint-acts "$WORK/an/acts.pt" --refusal-acts "$WORK/an/acts.pt" \
  --n-eval 20 --hold-out 30

# Reading an arm at one lambda against a control at another compares two amounts of
# intervention. That has to fail loudly, not fall back to whatever the control did run.
echo "a lambda the control never ran is fatal"
if python scripts/conditioned_analysis.py \
     --it-scored "$WORK/an/it/scored/baseline_scored.jsonl" \
     --dpo-scored "$WORK/an/dpo/scored/baseline_scored.jsonl" \
     --arm "a=$WORK/an/arm" --control "$WORK/an/ctl" \
     --at-lambda -0.200 > "$WORK/pin.log" 2>&1; then
  echo "  FAIL  lambda_pin_refuses   read an unmatched lambda without complaining"; fail=$((fail+1))
else
  echo "  ok    lambda_pin_refuses"; pass=$((pass+1))
fi

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
