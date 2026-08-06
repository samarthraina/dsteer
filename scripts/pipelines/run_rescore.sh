#!/usr/bin/env bash
# Judge up, then three staged jobs, cheapest and most informative first.
#
# Only the harmfulness rubric changed -- it now sees the request. Rescoring all five
# metrics would cost five times the judge calls for one metric's worth of new
# information, so the recheck path re-judges that metric alone into a sidecar keyed by
# id, keeping the old value beside the new one.
#
# Stage 1 is a 50-record pilot: if supplying the question barely moves the score, the
# full pass is not worth buying. It runs first so the answer is there early either way.
set -uo pipefail
cd /workspace/dsteer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/dsteer/src

echo "=== $(date -u) waiting for the activation run ==="
while tmux has-session -t srcctl 2>/dev/null; do sleep 60; done

echo "=== $(date -u) starting judge ==="
tmux new-session -d -s judge "cd /workspace/dsteer && source /workspace/venv_judge/bin/activate && python scripts/start_judge_server.py 2>&1 | tee judge.log"
for i in $(seq 1 90); do
  curl -s http://localhost:8000/v1/models >/dev/null 2>&1 && { echo "judge up after $((i*20))s"; break; }
  sleep 20
done
curl -s http://localhost:8000/v1/models >/dev/null 2>&1 || { echo "FATAL: judge never came up"; exit 1; }

TU=outputs/steer/tulu3/harmfulqa
OH=outputs/steer/llama3-oh/harmfulqa
OL=outputs/steer/olmo3/harmfulqa
DIRS="$OH/dpo $OH/it $TU/dpo $TU/it $TU/it_random $OL/dpo $OL/it"

echo "##### $(date -u) stage 1/3  pilot: does the question change the harmfulness score?"
python scripts/score_sweep.py --sweep-dir $OH/dpo --recheck harmfulness --limit 50 \
  --concurrency 64 2>&1 | grep -avE "httpx|^Activated" | tail -3
python - <<'PY'
import json, glob, statistics as st
v1, v2 = [], []
for f in glob.glob('outputs/steer/llama3-oh/harmfulqa/dpo/scored/*_recheck.jsonl'):
    for line in open(f, encoding='utf-8'):
        r = json.loads(line)
        a, b = r.get('harmfulness_score_v1'), r.get('harmfulness_score_v2')
        if a is not None and b is not None: v1.append(a); v2.append(b)
if v1:
    d = [b - a for a, b in zip(v1, v2)]
    moved = sum(1 for x in d if abs(x) >= 0.1)
    print(f"PILOT n={len(v1)}  without question {st.mean(v1):.3f} -> with question {st.mean(v2):.3f}"
          f"  mean shift {st.mean(d):+.3f}  |shift|>=0.1 on {moved}/{len(d)} ({moved/len(d):.0%})")
else:
    print("PILOT: no paired records")
PY

echo "##### $(date -u) stage 2/3  score the lambda points the extended sweeps added"
for d in $DIRS; do
  [ -d "$d" ] || continue
  case "$d" in *llama3-oh*) IT=$OH/it/baseline.jsonl; DP=$OH/dpo/baseline.jsonl;;
                *olmo3*)     IT=$OL/it/baseline.jsonl; DP=$OL/dpo/baseline.jsonl;;
                *)           IT=$TU/it/baseline.jsonl; DP=$TU/dpo/baseline.jsonl;; esac
  echo "--- $d"
  python scripts/score_sweep.py --sweep-dir "$d" --it-baseline "$IT" --dpo-baseline "$DP" \
    --concurrency 64 --sync 2>&1 | grep -avE "httpx|^Activated" | tail -3
done

echo "##### $(date -u) stage 3/3  harmfulness recheck across every sweep"
for d in $DIRS; do
  [ -d "$d" ] || continue
  echo "--- $d"
  python scripts/score_sweep.py --sweep-dir "$d" --recheck harmfulness \
    --concurrency 64 --sync 2>&1 | grep -avE "httpx|^Activated" | tail -2
done

echo "RESCORE DONE $(date -u).  Judge still up -- stop the instance when finished."
