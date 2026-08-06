#!/usr/bin/env bash
# Screen two candidate SFT checkpoints, and score the same-distribution sweeps.
#
# Generation and judging cannot share the card -- the judge holds 74 of 80 GB -- so this
# generates everything first, then brings the judge up once for all the scoring.
#
# The screen decides whether either checkpoint has room for a preference stage to install
# safety. Against llama3-oh's SFT, the one checkpoint here that supported a usable pair:
# refusal 0.241, harmfulness 0.339, quality 0.749.
set -euo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src

echo "=== $(date -u) phase 1: generation, judge down ==="
tmux kill-session -t judge 2>/dev/null; sleep 10

python scripts/screen_model.py --model teknium/OpenHermes-2.5-Mistral-7B \
  --output-dir outputs/screen/openhermes-mistral --n 150 2>&1 | grep -av "it/s\]\|batch/s\]"
python scripts/screen_model.py --model cognitivecomputations/dolphin-2.9-llama3-8b \
  --output-dir outputs/screen/dolphin-llama3 --n 150 2>&1 | grep -av "it/s\]\|batch/s\]"

echo "=== $(date -u) phase 2: judge up ==="
tmux new-session -d -s judge "cd /workspace/dsteer && source /workspace/venv_judge/bin/activate && python scripts/start_judge_server.py 2>&1 | tee judge.log"
for i in $(seq 1 90); do
  curl -s http://localhost:8000/v1/models >/dev/null 2>&1 && { echo "judge up after $((i*20))s"; break; }
  sleep 20
done
curl -s http://localhost:8000/v1/models >/dev/null 2>&1 || { echo "FATAL: judge never came up"; exit 1; }

echo "##### $(date -u) scoring the screens"
for d in outputs/screen/openhermes-mistral outputs/screen/dolphin-llama3; do
  [ -d "$d" ] || continue
  echo "--- $d"
  python scripts/score_sweep.py --sweep-dir "$d" --concurrency 64 --sync 2>&1 \
    | grep -avE "httpx|^Activated" | tail -3
done

echo "##### $(date -u) scoring the same-distribution sweeps"
SM=outputs/steer_srcmatch/llama3-oh/harmfulqa
for side in dpo it; do
  [ -d "$SM/$side" ] || continue
  echo "--- $SM/$side"
  python scripts/score_sweep.py --sweep-dir "$SM/$side" \
    --it-baseline "$SM/it/baseline.jsonl" --dpo-baseline "$SM/dpo/baseline.jsonl" \
    --concurrency 64 --sync 2>&1 | grep -avE "httpx|^Activated" | tail -3
done

echo "=== $(date -u) screen summary ==="
python - <<'PY'
import json, glob, statistics as st
print(f"{'checkpoint':28s} {'n':>4s} {'refusal':>8s} {'harmful':>8s} {'quality':>8s}")
for d in sorted(glob.glob('outputs/screen/*/scored/baseline_scored.jsonl')):
    rs=[json.loads(l) for l in open(d,encoding='utf-8') if l.strip()]
    def m(k):
        v=[r[k] for r in rs if r.get(k) is not None]
        return st.mean(v) if v else float('nan')
    name=d.split('/')[2]
    print(f"{name:28s} {len(rs):>4d} {m('refusal_score'):>8.3f} {m('harmfulness_score'):>8.3f} {m('quality_score'):>8.3f}")
print(f"{'llama3-oh SFT (reference)':28s} {300:>4d} {0.241:>8.3f} {0.339:>8.3f} {0.749:>8.3f}")
PY
echo "SCREEN DONE $(date -u).  Stop the instance when finished."
