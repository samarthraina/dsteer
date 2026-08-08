#!/usr/bin/env bash
# Give the refusal direction its own method, then compare on equal terms.
#
# Three splits of HarmfulQA, disjoint by construction:
#   0-300    evaluation, never seen by any direction
#   300-812  the prompts the candidate directions are built from
#   812-912  validation, where the choice among candidates is made
set -euo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src
tmux kill-session -t judge 2>/dev/null || true; sleep 10

echo "=== $(date -u) 1/3 generating one candidate per layer, ablated everywhere ==="
python scripts/select_refusal_direction.py \
  --model-config configs/llama3_oh_local.yaml \
  --activations outputs/refusal_direction/llama3-oh/activations.pt \
  --output-dir outputs/refusal_select --n-val 100 --val-offset 812 \
  --layer-step 2 --max-new-tokens 256 2>&1 | grep -av "it/s\]\|batch/s\]" | tail -4

echo "=== $(date -u) 2/3 judge ==="
tmux new-session -d -s judge "cd /workspace/dsteer && source /workspace/venv_judge/bin/activate && python scripts/start_judge_server.py 2>&1 | tee judge.log"
for i in $(seq 1 90); do curl -s http://localhost:8000/v1/models >/dev/null 2>&1 && break; sleep 20; done
curl -s http://localhost:8000/v1/models >/dev/null 2>&1 || { echo "FATAL: judge never came up"; exit 1; }

echo "=== $(date -u) 3/3 scoring the candidates ==="
python scripts/score_sweep.py --sweep-dir outputs/refusal_select/llama3-oh \
  --concurrency 64 --sync 2>&1 | grep -avE "httpx|^Activated" | tail -3

echo "=== $(date -u) winner ==="
python - <<'PY'
import json, glob, statistics as st, os
rows = []
for f in sorted(glob.glob("outputs/refusal_select/llama3-oh/scored/*_scored.jsonl")):
    rs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    name = os.path.basename(f).replace("_scored.jsonl", "")
    m = lambda k: st.mean([r[k] for r in rs if r.get(k) is not None]) if any(r.get(k) is not None for r in rs) else float("nan")
    nb = sum(1 for r in rs if not r.get("valid", True))
    rows.append((name, m("refusal_score"), m("quality_score"), nb, len(rs)))
base = next((r for r in rows if r[0] == "baseline"), None)
rows = [r for r in rows if r[0] != "baseline"]
# Lowest refusal, among candidates that keep the text usable.
ok = [r for r in rows if r[2] >= 0.60 and r[3] <= 0.15 * r[4]]
ok.sort(key=lambda r: r[1])
if base: print("baseline            refusal %.3f  quality %.3f  broken %d/%d" % base[1:])
print("\ntop candidates by refusal, quality >= 0.60 and under 15%% broken:")
for r in ok[:6]:
    print("  %-12s refusal %.3f  quality %.3f  broken %d/%d" % r)
print("\nrejected for quality or degeneration: %d of %d" % (len(rows) - len(ok), len(rows)))
if ok: print("\nWINNER: %s" % ok[0][0])
PY
echo "SELECT DONE $(date -u)"
