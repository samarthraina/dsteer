#!/usr/bin/env python3
"""Tests for src/budget_guard.py. Each pins one specific spending failure.

Run:  python3 -I tests/test_budget_guard.py
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from budget_guard import BudgetExceeded, BudgetGuard  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def guard(tmp: Path, name: str = "l.jsonl") -> BudgetGuard:
    return BudgetGuard(tmp / name)


def backdate(g: BudgetGuard, hours_ago: float, **fields) -> None:
    """Append a row stamped in the past, to simulate elapsed billing without waiting."""
    ts = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=hours_ago))
    row = {"utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), **fields}
    with g.path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def test_fresh_ledger_is_zero() -> None:
    """Pins: a guard that reports spend before anything started."""
    print("test_fresh_ledger_is_zero")
    with tempfile.TemporaryDirectory() as d:
        g = guard(Path(d))
        g.open_ledger("test")
        st = g.state()
        check("consumed is 0.0", st.consumed_h == 0.0, str(st.consumed_h))
        check("remaining is 30.0", st.remaining_h == 30.0)
        check("may schedule", st.may_schedule)
        check("no instance open", not st.instance_open)


def test_open_interval_bills_across_restart() -> None:
    """Pins: a crash losing an open instance interval, so spend silently stops counting."""
    print("test_open_interval_bills_across_restart")
    with tempfile.TemporaryDirectory() as d:
        g = guard(Path(d))
        g.open_ledger()
        backdate(g, 5.0, event="instance_start", instance_id="i-1", gpu_count=1)
        # a *new* guard object, as after a crash and restart
        st = guard(Path(d)).state()
        check("open interval still billed", 4.9 < st.consumed_h < 5.1, f"{st.consumed_h:.2f}h")
        check("instance reported open", st.instance_open)
        check("instance id recovered", st.instance_id == "i-1")


def test_multi_gpu_bills_proportionally() -> None:
    """Pins: N concurrent A100s counted as one, understating spend N-fold."""
    print("test_multi_gpu_bills_proportionally")
    with tempfile.TemporaryDirectory() as d:
        g = guard(Path(d))
        g.open_ledger()
        backdate(g, 2.0, event="instance_start", instance_id="i-2", gpu_count=4)
        st = guard(Path(d)).state()
        check("2h x 4 GPUs = ~8h", 7.8 < st.consumed_h < 8.2, f"{st.consumed_h:.2f}h")


def test_schedule_cutoff_at_28_5() -> None:
    """Pins: new GPU work admitted past the scheduling stop."""
    print("test_schedule_cutoff_at_28_5")
    with tempfile.TemporaryDirectory() as d:
        g = guard(Path(d))
        g.open_ledger()
        g.charge(28.6, "simulated prior spend")
        st = g.state()
        check("may_schedule is False", not st.may_schedule, f"{st.consumed_h:.2f}h")
        try:
            g.require_schedulable("stage_x", 0.1)
            check("require_schedulable raises", False, "no exception")
        except BudgetExceeded as e:
            check("require_schedulable raises", True, str(e)[:60])


def test_estimate_that_would_breach_ceiling_is_refused() -> None:
    """Pins: admitting a stage whose own estimate already exceeds 30h."""
    print("test_estimate_that_would_breach_ceiling_is_refused")
    with tempfile.TemporaryDirectory() as d:
        g = guard(Path(d))
        g.open_ledger()
        g.charge(25.0, "simulated")
        try:
            g.require_schedulable("big_stage", 6.0)  # 25 + 6 = 31 > 30
            check("refuses over-ceiling estimate", False, "no exception")
        except BudgetExceeded as e:
            check("refuses over-ceiling estimate", True, str(e)[:70])
        # and a stage that fits is admitted
        try:
            g.require_schedulable("small_stage", 2.0)
            check("admits a fitting stage", True)
        except BudgetExceeded:
            check("admits a fitting stage", False)


def test_export_and_terminate_flags() -> None:
    """Pins: passing 29h/29.5h without the export and terminate signals firing."""
    print("test_export_and_terminate_flags")
    for charged, exp, term in ((28.0, False, False), (29.1, True, False), (29.6, True, True)):
        with tempfile.TemporaryDirectory() as d:
            g = guard(Path(d))
            g.open_ledger()
            g.charge(charged, "simulated")
            st = g.state()
            check(f"{charged}h -> must_export={exp}", st.must_export is exp)
            check(f"{charged}h -> must_terminate={term}", st.must_terminate is term)


def test_unverified_stop_raises_but_is_recorded() -> None:
    """Pins: an unverified instance stop passing silently -- the real money risk."""
    print("test_unverified_stop_raises_but_is_recorded")
    with tempfile.TemporaryDirectory() as d:
        g = guard(Path(d))
        g.open_ledger()
        g.instance_start("i-3")
        raised = False
        try:
            g.instance_stop("i-3", stop_verified=False)
        except BudgetExceeded:
            raised = True
        check("unverified stop raises", raised)
        rows = [json.loads(l) for l in g.path.read_text().splitlines()]
        stops = [r for r in rows if r.get("event") == "instance_stop"]
        check("stop row persisted anyway", len(stops) == 1)
        check("row marks stop_verified False", stops[0]["stop_verified"] is False)


def test_double_open_refused() -> None:
    """Pins: two overlapping open intervals, which would corrupt the running total."""
    print("test_double_open_refused")
    with tempfile.TemporaryDirectory() as d:
        g = guard(Path(d))
        g.open_ledger()
        g.instance_start("i-4")
        try:
            g.instance_start("i-5")
            check("second open refused", False, "no exception")
        except BudgetExceeded:
            check("second open refused", True)


def test_ledger_is_append_only() -> None:
    """Pins: a rewrite that erases earlier spend."""
    print("test_ledger_is_append_only")
    with tempfile.TemporaryDirectory() as d:
        g = guard(Path(d))
        g.open_ledger()
        g.charge(1.0, "a")
        n1 = len(g.path.read_text().splitlines())
        g.charge(2.0, "b")
        g.close_ledger()
        rows = g.path.read_text().splitlines()
        check("rows only grew", len(rows) > n1, f"{n1} -> {len(rows)}")
        check("earlier charge retained", any('"reason": "a"' in r for r in rows))
        check("total is cumulative", abs(g.state().consumed_h - 3.0) < 1e-9,
              f"{g.state().consumed_h}")


def test_amendment_cannot_raise_ceiling() -> None:
    """Pins: 'reserve' being treated as permission to exceed 30h."""
    print("test_amendment_cannot_raise_ceiling")
    with tempfile.TemporaryDirectory() as d:
        g = guard(Path(d))
        g.open_ledger()
        g.charge(29.0, "simulated")
        try:
            g.amend_reserve("stage_y", 2.0, "want more")
            check("amendment refused past ceiling", False, "no exception")
        except BudgetExceeded:
            check("amendment refused past ceiling", True)


def test_box_reads_mirror_and_fails_safe() -> None:
    """Pins: the box seeing zero spend because it reads a box-local ledger with no instance
    intervals; a lost Mac link letting the box keep scheduling; the box rewriting billing."""
    print("test_box_reads_mirror_and_fails_safe")
    import os
    import time as _t
    from budget_guard import guard_for
    with tempfile.TemporaryDirectory() as d:
        logs = Path(d) / "logs"
        logs.mkdir()
        mac = BudgetGuard(logs / "gpu_budget.jsonl")
        mac.open_ledger()
        mac.charge(12.0, "simulated Mac-side billing")
        gdir = Path(d) / "guard"
        (gdir / "flags").mkdir(parents=True)
        os.environ["DSTEER_ROLE"] = "box"
        os.environ["DSTEER_GUARD_DIR"] = str(gdir)
        try:
            try:
                guard_for(d)
                check("missing mirror refuses scheduling", False)
            except BudgetExceeded:
                check("missing mirror refuses scheduling", True)
            mirror = logs / "gpu_budget.mirror.jsonl"
            mirror.write_bytes((logs / "gpu_budget.jsonl").read_bytes())
            try:
                guard_for(d)
                check("no instance-resident guard heartbeat refuses scheduling", False)
            except BudgetExceeded:
                check("no instance-resident guard heartbeat refuses scheduling", True)
            (gdir / "heartbeat.json").write_text("{}")
            g = guard_for(d)
            check("box reads Mac billing through the mirror", abs(g.state().consumed_h - 12.0) < 1e-9,
                  f"{g.state().consumed_h}")
            before = mirror.read_bytes()
            g.require_schedulable("stage_z", 1.0)
            check("box stage events go to box_events, mirror untouched",
                  mirror.read_bytes() == before and (logs / "box_events.jsonl").exists())
            try:
                g.charge(1.0, "box trying to bill")
                check("billing event from the box refused", False)
            except BudgetExceeded:
                check("billing event from the box refused", True)
            (gdir / "flags" / "STOP_SCHEDULING").write_text("x")
            try:
                guard_for(d)
                check("guard's STOP_SCHEDULING flag refuses scheduling", False)
            except BudgetExceeded:
                check("guard's STOP_SCHEDULING flag refuses scheduling", True)
            (gdir / "flags" / "STOP_SCHEDULING").unlink()
            old = _t.time() - 3600
            os.utime(mirror, (old, old))
            try:
                guard_for(d)
                check("stale mirror refuses scheduling", False)
            except BudgetExceeded:
                check("stale mirror refuses scheduling", True)
        finally:
            os.environ.pop("DSTEER_ROLE", None)
            os.environ.pop("DSTEER_GUARD_DIR", None)


def main() -> int:
    for t in (test_box_reads_mirror_and_fails_safe,
              test_fresh_ledger_is_zero,
              test_open_interval_bills_across_restart,
              test_multi_gpu_bills_proportionally,
              test_schedule_cutoff_at_28_5,
              test_estimate_that_would_breach_ceiling_is_refused,
              test_export_and_terminate_flags,
              test_unverified_stop_raises_but_is_recorded,
              test_double_open_refused,
              test_ledger_is_append_only,
              test_amendment_cannot_raise_ceiling):
        t()
        print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
