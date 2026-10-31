#!/usr/bin/env python3
"""Tests for the Stage 0 panel freeze. Each pins one specific failure.

Scope: only what preflight.py actually produces. Tests for vector scaling/sign,
zero-intervention parity, budget cutoff, score-to-response binding and safe resume are
NOT here, because the code they would cover does not exist yet. Writing them now would
assert against nothing and report false coverage.

Run:  python3 -I tests/test_panels.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PANELS = ROOT / "manifests" / "panels"
SALT = "dsteer-rebuttal-20261028"
HARMFUL_DIGEST = "7851299f7e15743e47c0e4604697a0770b8e9607a9dd4420a0f31a3529a53953"
MANIFEST_HASH = "0e1d0a8177d59eba601141a74da19a189527821dab573e203a3261cff54eec70"

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def load(panel: str) -> list[dict]:
    return [json.loads(l) for l in (PANELS / f"{panel}.jsonl").read_text().splitlines()]


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def test_no_split_leakage() -> None:
    """Pins: a tuning prompt reappearing in the test panel, which would void the result."""
    print("test_no_split_leakage")
    names = ["construction_harmful", "development", "calibration", "harmful_test"]
    panels = {n: load(n) for n in names}
    for a, b in combinations(names, 2):
        ia = {r["source_id"] for r in panels[a]}
        ib = {r["source_id"] for r in panels[b]}
        ha = {r["prompt_hash"] for r in panels[a]}
        hb = {r["prompt_hash"] for r in panels[b]}
        check(f"{a} x {b} disjoint by source_id", not (ia & ib), f"{len(ia & ib)} shared")
        check(f"{a} x {b} disjoint by prompt_hash", not (ha & hb), f"{len(ha & hb)} shared")

    allr = [r for n in names for r in panels[n]]
    ids = [r["source_id"] for r in allr]
    hs = [r["prompt_hash"] for r in allr]
    check("union size is 1938", len(allr) == 1938, f"{len(allr)}")
    check("no duplicate source_id in union", len(ids) == len(set(ids)))
    check("no duplicate prompt_hash in union", len(hs) == len(set(hs)),
          f"{len(hs) - len(set(hs))} collisions")
    counts = {n: len(panels[n]) for n in names}
    check("partition sizes 1378/60/200/300",
          counts == {"construction_harmful": 1378, "development": 60,
                     "calibration": 200, "harmful_test": 300}, str(counts))


def test_harmful_test_digest_is_the_submitted_panel() -> None:
    """Pins: silently evaluating a different or reordered 300-prompt panel."""
    print("test_harmful_test_digest_is_the_submitted_panel")
    rs = sorted(load("harmful_test"), key=lambda r: r["permuted_position"])
    dig = sha_text("\n".join(f"{r['source_id']}:{r['prompt_hash']}" for r in rs))
    check("digest matches submitted panel", dig == HARMFUL_DIGEST, dig)

    meta = json.loads((ROOT.parent.parent / "figures-final" / "data" / "task027-final" /
                       "steer" / "M0-A_installation" / "run_meta.json").read_text())
    check("digest matches run_meta field",
          meta.get("panel_ordered_identity_digest") == dig)
    check("manifest hash matches run_meta field",
          meta.get("harmfulqa_manifest_hash") == MANIFEST_HASH)


def test_fallback_is_nested_and_deterministic() -> None:
    """Pins: a fallback that drifts between runs or escapes the test panel."""
    print("test_fallback_is_nested_and_deterministic")
    fe = load("harmful_test")
    fb = load("harmful_test_fallback100")
    check("fallback size is 100", len(fb) == 100, str(len(fb)))
    check("fallback nested in harmful_test",
          {r["source_id"] for r in fb} <= {r["source_id"] for r in fe})

    recomputed = sorted(fe, key=lambda r: sha_text(f"{SALT}:{r['source_id']}"))[:100]
    check("fallback reproduces from salt rule",
          {r["source_id"] for r in recomputed} == {r["source_id"] for r in fb})
    check("fallback is ordered by permuted_position",
          [r["permuted_position"] for r in fb] == sorted(r["permuted_position"] for r in fb))


def test_no_prompt_text_claimed_present() -> None:
    """Pins: a run proceeding as if prompt text were available when only hashes are."""
    print("test_no_prompt_text_claimed_present")
    panels = json.loads((ROOT / "manifests" / "panels.json").read_text())
    for name in ("construction_harmful", "development", "calibration", "harmful_test"):
        check(f"{name} declares prompt text absent",
              panels["panels"][name]["prompt_text_present"] is False)
    for name in ("construction_harmless", "xstest_safe", "instruction_test"):
        check(f"{name} is marked blocked, not built",
              panels["panels"][name]["status"] == "BLOCKED_NOT_BUILT")


def test_budget_ledger_is_zero_and_closed() -> None:
    """Pins: a report claiming zero spend while the ledger shows otherwise."""
    print("test_budget_ledger_is_zero_and_closed")
    rows = [json.loads(l) for l in (ROOT / "logs" / "gpu_budget.jsonl").read_text().splitlines()]
    if any(r.get("event") == "instance_start" for r in rows):
        print("  SKIP  the CPU-stage zero-spend state ended with the first instance interval (2026-10-28)")
        return
    check("every ledger row reports 0.0 consumed hours",
          all(r.get("consumed_a100_hours") == 0.0 for r in rows))
    check("ledger is closed", rows[-1]["event"] == "ledger_close")
    check("no instance was started", rows[-1]["instance_started"] is False)


def main() -> int:
    for t in (test_no_split_leakage,
              test_harmful_test_digest_is_the_submitted_panel,
              test_fallback_is_nested_and_deterministic,
              test_no_prompt_text_claimed_present,
              test_budget_ledger_is_zero_and_closed):
        t()
        print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
