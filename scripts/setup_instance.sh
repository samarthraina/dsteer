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

# The working tree is not on any remote. `main` is still the v1 site, and the branch this
# script used to default to was deleted on 6 Aug for carrying review material, so a clone
# would either fail or fetch the wrong code. Copy the tree in instead:
#
#   tar czf - --exclude=.git --exclude=outputs . | ssh -p PORT root@HOST \
#     'mkdir -p /workspace/dsteer && tar xzf - -C /workspace/dsteer'
#
# The clone path below stays for the day the tree is published, and is skipped whenever a
# tree is already present.
BRANCH=${1:-main}
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
  log "at $(git -C "$WORK" log --oneline -1)"
elif [ -d "$WORK/src/steering" ]; then
  log "tree present without git metadata -- copied in, not cloned. Left alone."
else
  git clone -q "$REPO" "$WORK" && git -C "$WORK" checkout -q "$BRANCH"
  log "at $(git -C "$WORK" log --oneline -1)"
fi

log "=== generation env ==="
source /venv/main/bin/activate
# Not vllm: it would pull its own torch into this environment.
# tiktoken, tensorboard, sentencepiece and protobuf are here for training/dpo.py, not for
# generation. Three separate launches died on the first two being absent -- the trainer
# sets report_to="tensorboard" and the tokenizers want sentencepiece -- and each failure
# came after the checkpoint download rather than at import.
uv pip install -q transformers datasets accelerate peft trl \
  pandas matplotlib pyyaml tqdm pytest openai huggingface_hub \
  tiktoken tensorboard sentencepiece protobuf 2>&1 | tail -2
python - <<'PY'
import torch, transformers
print(f"  torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"transformers {transformers.__version__}")
if torch.cuda.is_available():
    print(f"  gpu: {torch.cuda.get_device_name(0)}")
PY

log "=== judge env ==="
# An unpinned `pip install vllm` takes whatever is newest, and current releases ship a
# CUDA 13 torch. On a host whose driver is 12.x that fails at model load with "The NVIDIA
# driver on your system is too old", after the box is rented and the checkpoints are down --
# generation and training are unaffected, because /venv/main is cu126, so the failure looks
# like a broken judge rather than a wrong wheel. Pin to a build that runs on 12.x drivers;
# CUDA's minor-version compatibility covers the rest.
# grep, not sed. The backreference here was written as a literal 0x01 byte by an earlier
# edit, so this returned a control character on every host: non-empty, so the `:-13`
# fallback never fired, and the comparison below died with "integer expression expected".
DRIVER_CUDA=$(nvidia-smi | grep -o 'CUDA Version: [0-9]*' | grep -o '[0-9]*$' | head -1)
if [ ! -x /workspace/venv_judge/bin/python ]; then
  python3 -m venv /workspace/venv_judge
  /workspace/venv_judge/bin/pip install -q --upgrade pip
  if [ "${DRIVER_CUDA:-13}" -lt 13 ]; then
    log "  driver is CUDA ${DRIVER_CUDA}.x -- pinning vllm<0.20 for a cu12x torch"
    /workspace/venv_judge/bin/pip install -q "vllm<0.20" 2>&1 | tail -3
  else
    /workspace/venv_judge/bin/pip install -q vllm 2>&1 | tail -3
  fi
fi
/workspace/venv_judge/bin/python -c "import torch, vllm; print(f'  judge torch {torch.__version__} vllm {vllm.__version__}')"
# Proves the wheel matches the driver in a second, rather than at model load ten minutes in.
/workspace/venv_judge/bin/python -c "
import torch, sys
if not torch.cuda.is_available(): sys.exit('  FATAL: judge venv cannot see the GPU')
(torch.ones(64, 64, device='cuda') @ torch.ones(64, 64, device='cuda')).sum().item()
print('  judge venv allocates on the GPU: ok')
" || log "  judge will not be able to load a model on this host"

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
