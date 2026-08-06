#!/usr/bin/env bash
# Bring a fresh Vast instance to a state where experiments can run.
#
# Two environments, deliberately. vLLM pins its own torch, and installing it beside the
# generation stack replaced torch 2.12.0+cu126 with 2.11.0+cu130 -- a silent way to
# invalidate a run already in flight. They talk over HTTP and have no reason to share
# an interpreter.
#
#   /venv/main            extraction, steering, generation, analysis  (image default)
#   /workspace/venv_judge vLLM judge server only
#
# Usage:
#   bash setup_instance.sh [branch]
#
# Then authenticate once, interactively, so the token never passes through a script:
#   source /venv/main/bin/activate && hf auth login
set -uo pipefail

BRANCH=${1:-fix/extraction-and-judge}
REPO=https://github.com/samarthraina/dsteer.git
WORK=/workspace/dsteer

log () { echo "[$(date -u +%H:%M:%S)] $*"; }

log "=== hardware ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "arch: $(uname -m)  cores: $(nproc)  ram: $(free -g | awk '/^Mem:/{print $2}')GB"
df -h /workspace | tail -1

log "=== repo ==="
if [ -d "$WORK/.git" ]; then
  git -C "$WORK" fetch -q origin && git -C "$WORK" checkout -q "$BRANCH" && git -C "$WORK" pull -q
else
  git clone -q "$REPO" "$WORK" && git -C "$WORK" checkout -q "$BRANCH"
fi
log "at $(git -C "$WORK" log --oneline -1)"

log "=== generation env ==="
source /venv/main/bin/activate
# Not vllm: it would pull its own torch into this environment.
uv pip install -q transformers datasets accelerate peft trl \
  pandas matplotlib pyyaml tqdm pytest openai huggingface_hub 2>&1 | tail -2
python - <<'PY'
import torch, transformers
print(f"  torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"transformers {transformers.__version__}")
if torch.cuda.is_available():
    print(f"  gpu: {torch.cuda.get_device_name(0)}")
PY

log "=== judge env ==="
if [ ! -x /workspace/venv_judge/bin/python ]; then
  python3 -m venv /workspace/venv_judge
  /workspace/venv_judge/bin/pip install -q --upgrade pip
  /workspace/venv_judge/bin/pip install -q vllm 2>&1 | tail -3
fi
/workspace/venv_judge/bin/python -c "import torch, vllm; print(f'  judge torch {torch.__version__} vllm {vllm.__version__}')"

log "=== isolation check ==="
# The whole point of two environments: confirm the judge install did not reach into main.
MAIN_T=$(/venv/main/bin/python -c "import torch; print(torch.__version__)")
JUDGE_T=$(/workspace/venv_judge/bin/python -c "import torch; print(torch.__version__)")
echo "  main=$MAIN_T judge=$JUDGE_T"
[ "$MAIN_T" != "$JUDGE_T" ] && echo "  (differ, as intended)" || echo "  (same version -- fine, but check nothing was downgraded)"

log "=== tests ==="
cd "$WORK" && PYTHONPATH="$WORK/src" python -m pytest tests/ -q --no-header 2>&1 | tail -3

log "=== done ==="
echo "Next:  source /venv/main/bin/activate && hf auth login"
echo "Then:  tmux new -s work"
