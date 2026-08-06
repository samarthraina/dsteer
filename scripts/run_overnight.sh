#!/bin/bash
# Run IT eval then DPO eval sequentially. Use after the judge server is up.
#
# Usage:
#   bash scripts/run_overnight.sh configs/llama3_oh.yaml
#
# Assumes:
#   - judge server is already running at localhost:8000
#   - PYTHONPATH includes src/ and /opt/eval_libs
#   - HF auth is set up
#
# Outputs go to outputs/it_eval/<name>/ and outputs/dpo_eval/<name>/
# Each step is resume-safe.

set -e  # exit on first error

MODEL_CFG=${1:-configs/llama3_oh.yaml}

echo "==============================================="
echo "Starting overnight run with model: $MODEL_CFG"
echo "Started at: $(date)"
echo "==============================================="

echo ""
echo ">>> Phase 1: IT eval (capability)"
echo ">>> Started at: $(date)"
python scripts/it_eval.py \
    --model-config "$MODEL_CFG" \
    --eval-config configs/it_eval.yaml

echo ""
echo ">>> Phase 2: DPO eval (safety/refusal)"
echo ">>> Started at: $(date)"
python scripts/dpo_eval.py \
    --model-config "$MODEL_CFG" \
    --eval-config configs/dpo_eval.yaml

echo ""
echo "==============================================="
echo "Overnight run complete."
echo "Finished at: $(date)"
echo "==============================================="
