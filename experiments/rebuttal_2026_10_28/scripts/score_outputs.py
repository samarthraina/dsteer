#!/usr/bin/env python3
"""Score responses with one evaluator, binding every score to the exact response text.

Evaluators:
  original     frozen Qwen3.5-27B @ fc05daec via the vendored Judge and rubrics, unchanged:
               temperature 0.7 / top_k 20 / top_p 0.8 / presence 1.5, 256 tokens, <=3 attempts,
               per-request seed derive_request_seed(20260815, source_id, metric) -- verified to
               reproduce all 43,240 seeds in the recovered scores. NEVER run on a recovered
               Task 027 condition: those already carry original-judge scores, and re-scoring
               them would double-count.
  independent  microsoft/phi-4 @ 2db69c1c (manifests/independent_judge.json), same rubric text,
               system prompt and response text, deterministic decoding recorded separately.
  xstest       official XSTest rubric text on the original judge model, official decoding
               (temperature 0, 16 tokens), plus the ported string-match label.
  ifeval       official strict/loose checkers, CPU, no model.

As in the original pipeline, a response that failed validity is not sent to an LLM judge; it
is recorded as not-scored with its reason, so judges see identical eligible sets.

Run:  python scripts/score_outputs.py --evaluator independent --panel harmful_test --pair A \
          --condition it_base --metrics harmfulness --judge-url http://localhost:8001/v1 \
          --stage independent_judging --estimate-hours 0.1
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor" / "primary_snapshot"))
sys.path.insert(0, str(ROOT / "src"))

from protocol import RECOVERED_HARMFUL, load_protocol  # noqa: E402
from records import ScoreStore, read_jsonl, sha256_text, utc  # noqa: E402

GLOBAL_SEED = 20260815


def load_responses(run_root: Path, panel: str, pair: str, condition: str) -> tuple[list[dict], str]:
    """Responses with response_hash, from the recovered originals or this run's generations."""
    key = (pair, condition)
    if panel == "harmful_test" and key in RECOVERED_HARMFUL:
        f = glob.glob(str(run_root / "inputs/recovered/task027/steer" / RECOVERED_HARMFUL[key] / "*.jsonl"))[0]
        from steering.validity import check
        rows = []
        for r in read_jsonl(Path(f)):
            v = check(r["response"])
            rows.append({"source_id": r["source_id"], "prompt": r["prompt"], "response": r["response"],
                         "response_hash": sha256_text(r["response"]),
                         "validity": {"ok": v.ok, "reason": v.reason}})
        return rows, f"recovered:{RECOVERED_HARMFUL[key]}"
    p = run_root / "outputs/generations" / panel / pair / f"{condition}.jsonl"
    return list(read_jsonl(p)), str(p.relative_to(run_root))


def judge_for(evaluator: str, url: str):
    from steering.config import JudgeConfig
    from steering.judge import Judge
    if evaluator == "original":
        return Judge(JudgeConfig.frozen_qwen35(server_url=url)), {"judge": "Qwen/Qwen3.5-27B@fc05daec18b0a78c049392ed2e771dde82bdf654",
                                                                    "config": "JudgeConfig.frozen_qwen35"}
    man = json.loads((ROOT / "manifests/independent_judge.json").read_text())
    dec = man["decoding"]
    # Non-sampling fields mirror frozen_qwen35 exactly (strict parsing, attempt cap, seed
    # derivation), so the two judges differ only in model and decoding.
    cfg = JudgeConfig(model_name=f"{man['repository']}@{man['revision']}", server_url=url,
                      max_tokens=dec["max_tokens"], temperature=dec["temperature"], top_p=dec["top_p"],
                      seed=dec["seed"], seed_derivation_version=1,
                      max_attempts=dec["max_attempts"], strict_response_parsing=True)
    return Judge(cfg), {"judge": f"{man['repository']}@{man['revision']}", "decoding": dec,
                        "manifest_sha256": __import__("hashlib").sha256(
                            (ROOT / "manifests/independent_judge.json").read_bytes()).hexdigest()}


def score_one(judge, metric: str, row: dict):
    from steering import metrics as M
    fn = {"harmfulness": M.score_harmfulness, "refusal": M.score_refusal,
          "helpfulness": M.score_helpfulness}.get(metric)
    if metric == "quality":
        return M.score_quality(judge, row["response"], record_id=row["source_id"])
    return fn(judge, row["prompt"], row["response"], record_id=row["source_id"])


def judge_record(row: dict, metric: str, res, judge_id: dict) -> dict:
    from steering.metrics import ACTIVE_RUBRICS, rubric_hash
    abst = res.discrete_score is None
    return {"source_id": row["source_id"], "response_hash": row["response_hash"], "metric": metric,
            "rubric_hash": rubric_hash(ACTIVE_RUBRICS[metric]), "judge": judge_id,
            "attempt_count": res.attempt_count, "attempts": res.attempts,
            "discrete_score": res.discrete_score, "score": res.score, "abstained": abst,
            "abstention_reason": res.error if abst else None, "reason": res.reason,
            "resolved_seed": res.resolved_seed, "scored_utc": utc()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default=str(ROOT))
    ap.add_argument("--evaluator", required=True, choices=["original", "independent", "xstest", "ifeval"])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--pair", required=True, choices=["A", "B"])
    ap.add_argument("--condition", required=True)
    ap.add_argument("--metrics", default="harmfulness")
    ap.add_argument("--judge-url", default="http://localhost:8000/v1")
    ap.add_argument("--subset", default=None, help="panel jsonl restricting source ids (e.g. fallback100)")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--estimate-hours", type=float, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run_root = Path(a.run_root).resolve()
    _, phash = load_protocol()

    if a.evaluator == "original" and a.panel == "harmful_test" and (a.pair, a.condition) in RECOVERED_HARMFUL:
        raise SystemExit("REFUSED: recovered Task 027 responses already carry original-judge scores; "
                         "re-scoring them would double-count")
    rows, source = load_responses(run_root, a.panel, a.pair, a.condition)
    if a.subset:
        keep = {r["source_id"] for r in read_jsonl(run_root / "manifests/panels" / a.subset)}
        rows = [r for r in rows if r["source_id"] in keep]
    eligible = [r for r in rows if r["validity"]["ok"]]
    metrics = a.metrics.split(",") if a.evaluator in ("original", "independent") else [a.evaluator]
    out = run_root / "outputs/scores" / a.evaluator / a.panel / a.pair / f"{a.condition}.jsonl"
    identity = {"evaluator": a.evaluator, "panel": a.panel, "pair": a.pair, "condition": a.condition,
                "source": source, "protocol_hash": phash, "subset": a.subset}
    store = ScoreStore(out, identity)
    todo = store.pending(eligible, metrics)
    print(f"plan: {a.evaluator} on {a.pair}/{a.condition}/{a.panel}: {len(rows)} responses, "
          f"{len(eligible)} valid, {len(todo)} metric calls pending ({','.join(metrics)}) -> {out}")
    if a.dry_run:
        return 0

    if a.evaluator == "ifeval":
        import ifeval_score
        panel = {r["source_id"]: r for r in read_jsonl(run_root / "manifests/panels/instruction_test_100.jsonl")}
        for row, m in todo:
            res = ifeval_score.score(panel[row["source_id"]], row["response"])
            store.append_bound(row, {"source_id": row["source_id"], "response_hash": row["response_hash"],
                                     "metric": m, "checker": ifeval_score.CHECKER_REF,
                                     "deviations": ifeval_score.DEVIATIONS, **res, "scored_utc": utc()})
        store.finalise_meta()
        return 0

    from budget_guard import guard_for
    guard = guard_for(run_root)
    guard.require_schedulable(f"{a.stage}:{a.evaluator}:{a.pair}:{a.condition}:{a.panel}", a.estimate_hours)

    if a.evaluator == "xstest":
        import xstest_classify as X
        from openai import OpenAI
        client = OpenAI(base_url=a.judge_url, api_key="EMPTY")
        model = "Qwen/Qwen3.5-27B@fc05daec18b0a78c049392ed2e771dde82bdf654"

        def work(item):
            row, m = item
            raw, err = None, None
            try:
                resp = client.chat.completions.create(
                    model=model, messages=[{"role": "system", "content": X.SYSTEM},
                                           {"role": "user", "content": X.render(row["prompt"], row["response"])}],
                    temperature=X.DECODING["temperature"], max_tokens=X.DECODING["max_tokens"],
                    top_p=X.DECODING["top_p"], extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                raw = resp.choices[0].message.content
            except Exception as exc:  # noqa: BLE001 -- an abstention, recorded with its cause
                err = f"{type(exc).__name__}: {exc}"
            label = X.parse_label(raw)
            return row, {"source_id": row["source_id"], "response_hash": row["response_hash"], "metric": m,
                         "rubric_hash": X.rubric_hash(), "judge": {"model": model, "decoding": X.DECODING},
                         "raw": raw, "error": err, "label": label, "abstained": label is None,
                         "strmatch_label": X.strmatch_label(row["response"]), "scored_utc": utc()}
    else:
        judge, judge_id = judge_for(a.evaluator, a.judge_url)

        def work(item):
            row, m = item
            return row, judge_record(row, m, score_one(judge, m, row), judge_id)

    # a judge that cannot be reached is an infrastructure failure, never an abstention: on
    # 2026-10-29 a skipped judge start turned ~3,200 calls into stored "abstentions" and exit 0
    import urllib.request
    try:
        req = urllib.request.Request(a.judge_url.rstrip("/") + "/models", headers={"Authorization": "Bearer EMPTY"})
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as exc:  # noqa: BLE001
        print(f"judge unreachable at {a.judge_url} ({type(exc).__name__}); nothing scored")
        return 1

    def transport_failure(rec: dict) -> bool:
        why = " ".join(str(x.get("error")) for x in rec.get("attempts") or []) + " " + str(rec.get("error"))
        return rec.get("abstained") and any(t in why for t in ("APIConnectionError", "Connection error", "ConnectError"))

    n, lost = 0, 0
    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        for i in range(0, len(todo), 256):
            if not guard.state().may_schedule:
                print("budget scheduling stop reached; finishing with partial scores")
                break
            for row, rec in pool.map(work, todo[i:i + 256]):
                if transport_failure(rec):
                    lost += 1          # left unscored, so a retry resumes exactly these
                    continue
                store.append_bound(row, rec)
                n += 1
            if lost:
                print(f"judge connection failures on {lost} calls; stopping (infrastructure, retried)")
                return 1
    not_scored = [{"source_id": r["source_id"], "reason": r["validity"]["reason"]} for r in rows if not r["validity"]["ok"]]
    store.finalise_meta(n_responses=len(rows), n_valid=len(eligible), not_scored_invalid=not_scored)
    print(f"scored {n} metric calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
