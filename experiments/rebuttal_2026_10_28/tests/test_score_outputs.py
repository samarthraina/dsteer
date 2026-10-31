#!/usr/bin/env python3
"""Scorer plumbing on REAL recovered responses, against a local OpenAI-compatible stub.

The stub returns fixed text. This therefore validates binding, eligibility, abstention,
attempt caps, resume and refusals -- NOT judge behaviour or accuracy, which only the GPU run
can show. (A mocked judge once passed while the real judge was broken; this test does not
claim otherwise.)

Pins:
  - original-judge re-scoring of recovered Task 027 responses (double counting);
  - an eligible set that differs from the original pipeline's scored set;
  - an abstention coerced to a score of 0;
  - more than three total attempts per call;
  - a score not bound to the response it was computed on;
  - resume re-issuing completed calls.

Run:  .venv/bin/python -I tests/test_score_outputs.py
"""
from __future__ import annotations

import csv
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
FAILURES: list[str] = []
MODE = {"content": '{"score": 3, "reason": "stub"}', "calls": 0}


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


class Stub(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        b = json.dumps({"object": "list", "data": [{"id": "stub"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        MODE["calls"] += 1
        out = {"id": "x", "object": "chat.completion", "created": 0, "model": body.get("model"),
               "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": MODE["content"]}}],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        b = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


def temp_root() -> Path:
    d = Path(tempfile.mkdtemp())
    for sub in ("inputs", "manifests", "configs", "vendor", "src", "scripts"):
        os.symlink(ROOT / sub, d / sub)
    (d / "logs").mkdir()
    return d


def run(root: Path, port: int, *args) -> subprocess.CompletedProcess:
    cmd = [PY, "-I", str(ROOT / "scripts" / "score_outputs.py"), "--run-root", str(root),
           "--judge-url", f"http://127.0.0.1:{port}/v1", "--stage", "test", "--estimate-hours", "0",
           "--concurrency", "8", *args]
    return subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "OPENAI_API_KEY": "EMPTY"})


def main() -> int:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    root = temp_root()
    try:
        print("test_refusals")
        r = run(root, port, "--evaluator", "original", "--panel", "harmful_test", "--pair", "A",
                "--condition", "it_base")
        check("original judge refused on a recovered original", r.returncode != 0 and "REFUSED" in r.stdout + r.stderr)

        print("test_judge_down_is_not_an_abstention")
        # Pins: a skipped judge start (2026-10-29) stored every call as an "abstention" and exited 0
        dead = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        dead_port = dead.server_address[1]
        dead.server_close()
        r = run(root, dead_port, "--evaluator", "independent", "--panel", "harmful_test", "--pair", "A",
                "--condition", "it_base", "--metrics", "harmfulness")
        out = root / "outputs/scores/independent/harmful_test/A/it_base.jsonl"
        check("unreachable judge -> non-zero exit", r.returncode != 0, (r.stdout or r.stderr)[-200:])
        check("unreachable judge -> no score records written",
              not out.exists() or not out.read_text().strip())

        print("test_independent_on_recovered")
        r = run(root, port, "--evaluator", "independent", "--panel", "harmful_test", "--pair", "A",
                "--condition", "it_base", "--metrics", "harmfulness")
        check("run completes", r.returncode == 0, (r.stderr or r.stdout)[-300:])
        out = root / "outputs/scores/independent/harmful_test/A/it_base.jsonl"
        recs = [json.loads(l) for l in out.open()] if out.exists() else []
        conds = {x["condition"]: x for x in csv.DictReader(open(ROOT.parent.parent / "figures-final/data/task027-final/analysis/conditions.csv"))}
        expect = int(conds["M0-A_baseline"]["n_scored_harm"])
        check("eligible set equals the original pipeline's scored set", len(recs) == expect, f"{len(recs)} vs {expect}")
        check("every score carries a 64-hex response hash", all(len(x["response_hash"]) == 64 for x in recs))
        check("stub score parsed to 0.3", all(x["score"] == 0.3 for x in recs))
        meta = json.loads(out.with_suffix(".meta.json").read_text())
        check("invalid responses recorded as not scored, with reasons", len(meta["not_scored_invalid"]) == 300 - expect)
        before = MODE["calls"]
        run(root, port, "--evaluator", "independent", "--panel", "harmful_test", "--pair", "A",
            "--condition", "it_base", "--metrics", "harmfulness")
        check("resume issues no new calls", MODE["calls"] == before, f"{MODE['calls'] - before} new")

        print("test_abstention")
        MODE["content"] = "not json at all"
        before = MODE["calls"]
        r = run(root, port, "--evaluator", "independent", "--panel", "harmful_test", "--pair", "B",
                "--condition", "dpo_base", "--metrics", "harmfulness", "--subset", "harmful_test_fallback100.jsonl")
        out = root / "outputs/scores/independent/harmful_test/B/dpo_base.jsonl"
        recs = [json.loads(l) for l in out.open()]
        check("malformed output -> abstention, never 0", all(x["abstained"] and x["score"] is None for x in recs))
        check("at most 3 attempts per call", all(x["attempt_count"] <= 3 for x in recs) and
              MODE["calls"] - before <= 3 * len(recs), f"{MODE['calls'] - before} calls / {len(recs)} records")
        check("subset restricts to the frozen fallback", len(recs) <= 100)

        print("test_xstest_path")
        MODE["content"] = "2_full_refusal"
        gen = root / "outputs/generations/xstest_safe/A"
        gen.mkdir(parents=True)
        panel = [json.loads(l) for l in open(ROOT / "manifests/panels/xstest_safe_250.jsonl")][:5]
        with (gen / "it_base.jsonl").open("w") as fh:
            for p in panel:
                text = "Sorry, I can't help with that."
                fh.write(json.dumps({"source_id": p["source_id"], "prompt": p["prompt"], "response": text,
                                     "response_hash": __import__("hashlib").sha256(text.encode()).hexdigest(),
                                     "validity": {"ok": True, "reason": None}}) + "\n")
        r = run(root, port, "--evaluator", "xstest", "--panel", "xstest_safe", "--pair", "A", "--condition", "it_base")
        recs = [json.loads(l) for l in (root / "outputs/scores/xstest/xstest_safe/A/it_base.jsonl").open()]
        check("xstest label parsed", all(x["label"] == "2_full_refusal" for x in recs), (r.stderr or "")[-200:])
        check("strmatch secondary label present", all(x["strmatch_label"] == "2_full_refusal" for x in recs))

        print("test_binding")
        sys.path.insert(0, str(ROOT / "src"))
        from records import BindingError, ScoreStore
        st = ScoreStore(root / "b.jsonl", {"t": 1})
        try:
            st.append_bound({"source_id": "s", "response": "abc"},
                            {"source_id": "s", "response_hash": "0" * 64, "metric": "harmfulness"})
            check("score bound to a different text refused", False)
        except BindingError:
            check("score bound to a different text refused", True)
    finally:
        srv.shutdown()
        shutil.rmtree(root, ignore_errors=True)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
