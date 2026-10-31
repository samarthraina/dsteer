#!/usr/bin/env python3
"""Recovered Task 027 responses must reproduce the submitted numbers. Real data, no mocks.

Pins three failures that would silently corrupt every rescoring result:
  - a recovered file that is not the submitted one (wrong revision, wrong prefix, partial);
  - a score bound to the wrong response text;
  - an analysis path that drifts from the submitted estimand (intersection, sign, validity).

CIs are compared loosely on purpose: the submission used a different bootstrap RNG stream,
so interval endpoints differ by Monte Carlo noise. Paired n and point estimates are
deterministic and must match exactly.

Run:  python3 -I tests/test_recovered_reproduction.py
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from paired_stats import directed_effect  # noqa: E402

REC = ROOT / "inputs" / "recovered" / "task027"
REF = ROOT.parent.parent / "figures-final" / "data" / "task027-final" / "analysis"
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def scored(cond: str) -> dict:
    f = glob.glob(str(REC / "scored" / cond / "*_scored.jsonl"))[0]
    return {r["source_id"]: r for r in map(json.loads, open(f))}


def harm(rows: dict) -> dict:
    return {k: (r["harmfulness_score"] if r.get("valid") and not r.get("harmfulness_abstained")
                and r.get("harmfulness_score") is not None else None) for k, r in rows.items()}


def test_inventory_and_counts() -> None:
    """Pins: a partial or substituted recovery."""
    print("test_inventory_and_counts")
    man = json.loads((ROOT / "manifests" / "recovered_artifacts.json").read_text())
    s = man["summary"]
    check("58 response files verified locally", s["responses_verified_local"] == 58,
          str(s["responses_verified_local"]))
    check("74,032,462 response bytes", s["responses_bytes_local"] == 74_032_462,
          str(s["responses_bytes_local"]))
    check("inventory matches local reference", man["inventory"]["downloaded_matches_reference"])
    check("inventory matches expected hash", man["inventory"]["downloaded_matches_expected"])
    check("pinned revision", man["revision"] == "4f72531f763eff1c43f29769a6f76d6c06b7ca5a")
    conds = sorted(Path(p).name for p in glob.glob(str(REC / "scored" / "*")))
    check("29 scored conditions", len(conds) == 29, str(len(conds)))
    check("every scored file has 300 rows",
          all(len(scored(c)) == 300 for c in conds))


def test_score_to_response_binding() -> None:
    """Pins: a score attached to a response it was not computed on."""
    print("test_score_to_response_binding")
    tot = bad = 0
    for d in glob.glob(str(REC / "scored" / "*")):
        c = Path(d).name
        gen = {r["source_id"]: r["response"]
               for r in map(json.loads, open(glob.glob(str(REC / "steer" / c / "*.jsonl"))[0]))}
        for k, r in scored(c).items():
            tot += 1
            bad += gen.get(k) != r["response"]
    check("every scored response byte-identical to its generation", bad == 0, f"{tot - bad}/{tot}")


def test_panel_identity() -> None:
    """Pins: recovered rows drawn from a different panel than the frozen one."""
    print("test_panel_identity")
    panel = {json.loads(l)["source_id"]
             for l in (ROOT / "manifests" / "panels" / "harmful_test.jsonl").read_text().splitlines()}
    for c in ("M0-A_baseline", "M+-B_reversal", "rand_A_installation_s11"):
        check(f"{c} source ids == frozen 300-prompt panel", set(scored(c)) == panel)


def test_submitted_effects_reproduce() -> None:
    """Pins: an analysis path that no longer reproduces the submitted estimand."""
    print("test_submitted_effects_reproduce")
    ref = {(r["pair"], r["direction"]): r for r in csv.DictReader(open(REF / "real_effects.csv"))}
    cells = [("A", "installation", "M0-A_baseline", "M0-A_installation", +1),
             ("A", "reversal", "M+-A_baseline", "M+-A_reversal", -1),
             ("B", "installation", "M0-B_baseline", "M0-B_installation", +1),
             ("B", "reversal", "M+-B_baseline", "M+-B_reversal", -1)]
    for p, d, b, i, s in cells:
        e = directed_effect(harm(scored(b)), harm(scored(i)), s)
        r = ref[(p, d)]
        check(f"{p} {d}: paired n == {r['n_paired']}", e.n_paired == int(r["n_paired"]), str(e.n_paired))
        check(f"{p} {d}: point == {float(r['harm_effect']):.6f}",
              abs(e.point - float(r["harm_effect"])) < 1e-9, f"{e.point:.6f}")
        check(f"{p} {d}: CI within bootstrap noise",
              abs(e.ci_lo - float(r["harm_ci_lo"])) < 5e-3 and abs(e.ci_hi - float(r["harm_ci_hi"])) < 5e-3,
              f"[{e.ci_lo:.4f},{e.ci_hi:.4f}] vs [{float(r['harm_ci_lo']):.4f},{float(r['harm_ci_hi']):.4f}]")


def main() -> int:
    if not REC.is_dir():
        print("recovered artifacts absent; run scripts/fetch_recovered.py first")
        return 1
    for t in (test_inventory_and_counts, test_score_to_response_binding,
              test_panel_identity, test_submitted_effects_reproduce):
        t()
        print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
