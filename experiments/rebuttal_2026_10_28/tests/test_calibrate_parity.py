#!/usr/bin/env python3
"""Calibration choice rules and the generation-parity check. CPU.

The calibration rows are constructed (pure decision logic: gates, ordering, tie-break, one
verification). The parity check runs on REAL recovered responses.

Pins:
  - choosing a candidate that fails a gate, or the larger magnitude on a tie;
  - re-tuning after a failed calibration verification;
  - a zero-intervention parity failure passing as drift instead of a hard stop.

Run:  .venv/bin/python -I tests/test_calibrate_parity.py
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import calibrate_refusal as C  # noqa: E402

FAILURES: list[str] = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def row(m, eff, ok=True):
    return {"multiplier": m, "harm_effect": eff, "all_pass": ok}


def test_choose():
    print("test_choose")
    check("largest movement among passing", C.choose([row(0.5, .1), row(1.0, .3), row(1.5, .5, ok=False)])["multiplier"] == 1.0)
    check("tie goes to the smaller magnitude", C.choose([row(1.5, .2), row(0.5, .2), row(1.0, .1)])["multiplier"] == 0.5)
    check("no passing candidate -> None, no fallback", C.choose([row(0.5, .4, False), row(1.0, .5, False)]) is None)
    check("nan effect never chosen", C.choose([row(0.5, float("nan")), row(1.0, .1)])["multiplier"] == 1.0)


def test_gates():
    print("test_gates")
    gen = [{"validity": {"ok": True, "max_new_tokens_hit": False}} for _ in range(19)] + \
          [{"validity": {"ok": False, "max_new_tokens_hit": True}}]
    ids = [f"p{i}" for i in range(20)]
    hb, hs = {i: 0.8 for i in ids}, {i: 0.3 for i in ids}
    qb, qs = {i: 1.0 for i in ids}, {i: 0.85 for i in ids}
    r = C.evaluate(gen, hb, hs, qb, qs, +1)
    check("5% invalid passes (<= 5%)", r["invalid_pass"] and abs(r["invalid_rate"] - 0.05) < 1e-12)
    check("quality loss 0.15 fails the 0.10 gate", not r["quality_pass"] and not r["all_pass"])
    check("intended movement computed", abs(r["harm_effect"] - 0.5) < 1e-9)


def test_decide_once():
    print("test_decide_once")
    with tempfile.TemporaryDirectory() as d:
        rr = Path(d)
        (rr / "manifests").mkdir()
        (rr / "manifests/comparator_selection.json").write_text(json.dumps(
            {"tuned": {"A_installation": {"dev_selected_multiplier": 1.5, "status": "failed_calibration", "multiplier": None}}}))
        r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/calibrate_refusal.py"), "decide",
                            "--run-root", d, "--pair", "A", "--direction", "installation"], capture_output=True, text=True)
        check("a decided cell is never re-verified or re-tuned", r.returncode != 0 and "never retuned" in r.stdout + r.stderr)


def test_parity():
    print("test_parity")
    sys.path.insert(0, str(ROOT / "scripts"))
    import parity_check as P
    from protocol import RECOVERED_HARMFUL
    tmp = Path(tempfile.mkdtemp())
    try:
        for sub in ("inputs", "src"):
            os.symlink(ROOT / sub, tmp / sub)
        (tmp / "reports").mkdir()
        d = tmp / "outputs/parity/harmful_test/A"
        d.mkdir(parents=True)
        orig = [json.loads(l) for l in open(glob.glob(str(ROOT / "inputs/recovered/task027/steer/M0-A_baseline/*.jsonl"))[0])][:5]
        with (d / "it_base.jsonl").open("w") as fh:
            for r in orig:
                fh.write(json.dumps({"source_id": r["source_id"], "response": r["response"], "finish_reason": r["stop_reason"]}) + "\n")
        P.ROOT = tmp
        check("identical responses pass parity", P.main() == 0)
        with (d / "it_base.jsonl").open("w") as fh:
            for i, r in enumerate(orig):
                fh.write(json.dumps({"source_id": r["source_id"], "response": r["response"] + ("x" if i == 2 else ""),
                                     "finish_reason": r["stop_reason"]}) + "\n")
        rc = P.main()
        rep = json.loads((tmp / "reports/PARITY.json").read_text())["rows"][0]
        check("one changed zero-intervention response is a HARD_STOP", rc != 0 and rep["verdict"] == "HARD_STOP",
              f"{rep['exact_text_matches']}/{rep['n']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    for t in (test_choose, test_gates, test_decide_once, test_parity):
        t()
        print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
