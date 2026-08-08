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
step screen_model         python scripts/screen_model.py --model "$TINY" --output-dir "$WORK/screen" --n 4 --max-new-tokens 8

# A learned vector and a derived one must not share a directory, or the second run resumes
# on the first's generations and the comparison is between a vector and itself.
echo "run directories are distinct"
if [ -d "$WORK/steer/tiny/harmfulqa/it" ] && [ -d "$WORK/steer/tiny/harmfulqa/it_learned" ]; then
  echo "  ok    distinct run directories"; pass=$((pass+1))
else
  echo "  FAIL  distinct run directories: $(ls "$WORK/steer/tiny/harmfulqa" 2>/dev/null | tr '\n' ' ')"
  fail=$((fail+1))
fi

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
