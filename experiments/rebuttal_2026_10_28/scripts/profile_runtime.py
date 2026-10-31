#!/usr/bin/env python3
"""Measured runtime profile from work already executed, then a re-planned budget.

Profiling reuses needed work rather than throwaway runs: the S1 development baselines are
the calibration baselines, and their scoring is the original-judge profile; S5 opens with an
independent-judge pass over development responses only. Rates come from wall-clock step
times in logs/plan_progress.jsonl and record counts on disk, so loads and start-ups are
inside the measured times, not assumed.

Writes reports/PROFILE.json and rebuilds the plan with `plan.py --profile`. A rate with no
measurement keeps its provisional value and is flagged.

Run on the box:  python scripts/profile_runtime.py [--judge independent]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from records import read_jsonl  # noqa: E402


def steps() -> list[dict]:
    p = ROOT / "logs/plan_progress.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def arg(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=None)
    a = ap.parse_args()
    prof = json.loads((ROOT / "reports/PROFILE.json").read_text()) if (ROOT / "reports/PROFILE.json").exists() else {}
    gen, orig, ind = [], [], []
    for s in steps():
        if s.get("status") != "ok":
            continue
        argv = s["argv"]
        if "scripts/run_panel.py" in argv and "--parity" not in argv:
            f = ROOT / "outputs/generations" / arg(argv, "--panel") / arg(argv, "--pair") / f"{arg(argv, '--condition')}.jsonl"
            n = sum(1 for _ in read_jsonl(f))
            gen.append({"pair": arg(argv, "--pair"), "condition": arg(argv, "--condition"), "n": n, "hours": s["hours"]})
        if "scripts/score_outputs.py" in argv and "--metrics" in argv:   # IFEval steps are CPU checkers, no judge calls
            ev = arg(argv, "--evaluator")
            calls = len(arg(argv, "--metrics").split(",")) * sum(
                1 for r in read_jsonl(ROOT / "outputs/generations" / arg(argv, "--panel") / arg(argv, "--pair")
                                      / f"{arg(argv, '--condition')}.jsonl") if r["validity"]["ok"])
            (orig if ev == "original" else ind if ev == "independent" else []).append({"calls": calls, "hours": s["hours"]})
    srv = json.loads((ROOT / "logs/judge_servers.json").read_text()) if (ROOT / "logs/judge_servers.json").exists() else {}

    def rate(rows, key):
        h = sum(r["hours"] for r in rows)
        return round(sum(r[key] for r in rows) / h, 1) if h > 0 else None

    prof.update({
        "generation_responses_per_h": rate(gen, "n"), "generation_steps": gen,
        "original_judge_calls_per_h": rate(orig, "calls"), "original_scoring_steps": orig,
        "independent_judge_calls_per_h": rate(ind, "calls") if (a.judge == "independent" or ind) else prof.get("independent_judge_calls_per_h"),
        "judge_ready_seconds": {k: v.get("ready_s") for k, v in srv.items()},
        "note": "rates include each step's own model load / judge client set-up; judge start-up is reported separately",
    })
    prof["unmeasured"] = [k for k in ("generation_responses_per_h", "original_judge_calls_per_h",
                                      "independent_judge_calls_per_h") if not prof.get(k)]
    (ROOT / "reports/PROFILE.json").write_text(json.dumps(prof, indent=2) + "\n")
    print(json.dumps({k: prof[k] for k in ("generation_responses_per_h", "original_judge_calls_per_h",
                                           "independent_judge_calls_per_h", "unmeasured")}, indent=1))
    subprocess.run([sys.executable, str(ROOT / "scripts/plan.py"), "--profile", str(ROOT / "reports/PROFILE.json")], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
