#!/usr/bin/env python3
"""Append-only GPU budget ledger with hard stop thresholds.

A script-local elapsed timer cannot guard spending across crashes and restarts, so the
ledger is the authority: every entry is appended to disk and consumption is always
recomputed from the file, never from memory.

Thresholds from the handoff, against a 30.0 h ceiling:
  28.5 h  stop scheduling new GPU work
  29.0 h  save and export
  29.5 h  terminate GPU workers and stop the paid instance
  30.0 h  ceiling; reserve is not permission to exceed it

Billing is counted from instance start, including idle, setup, download and retry time.
N concurrent A100s consume N x wall time.

Fully CPU-testable: see tests/test_budget_guard.py.
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path

CEILING_H = 30.0
STOP_SCHEDULING_H = 28.5
EXPORT_H = 29.0
TERMINATE_H = 29.5


class BudgetExceeded(RuntimeError):
    """Raised when work is requested past a hard threshold."""


@dataclass(frozen=True)
class BudgetState:
    consumed_h: float
    remaining_h: float
    may_schedule: bool
    must_export: bool
    must_terminate: bool
    instance_open: bool
    instance_id: str | None

    def as_dict(self) -> dict:
        return {
            "consumed_a100_hours": round(self.consumed_h, 4),
            "remaining_a100_hours": round(self.remaining_h, 4),
            "may_schedule_new_work": self.may_schedule,
            "must_export_now": self.must_export,
            "must_terminate_now": self.must_terminate,
            "instance_open": self.instance_open,
            "instance_id": self.instance_id,
        }


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


MIRROR_MAX_AGE_S = 600
GUARD_MAX_AGE_S = 120


def guard_for(run_root: str | Path) -> "BudgetGuard":
    """The right guard for where this process runs.

    On the Mac (default) the ledger of record is logs/gpu_budget.jsonl. On the GPU box
    (DSTEER_ROLE=box) billing is NOT known locally: the Mac watchdog pushes a read-only mirror
    to logs/gpu_budget.mirror.jsonl every tick, box-side events go to logs/box_events.jsonl,
    and a mirror older than MIRROR_MAX_AGE_S refuses all scheduling -- a lost link fails safe.
    """
    import os
    import time
    logs = Path(run_root) / "logs"
    if os.environ.get("DSTEER_ROLE") == "box":
        mirror = logs / "gpu_budget.mirror.jsonl"
        if not mirror.exists() or time.time() - mirror.stat().st_mtime > MIRROR_MAX_AGE_S:
            raise BudgetExceeded("ledger mirror missing or stale: refusing to schedule GPU work")
        gdir = Path(os.environ.get("DSTEER_GUARD_DIR", "/root/dsteer-guard"))
        hb = gdir / "heartbeat.json"
        if not hb.exists() or time.time() - hb.stat().st_mtime > GUARD_MAX_AGE_S:
            raise BudgetExceeded("instance-resident deadline guard not running: refusing to schedule")
        if (gdir / "flags" / "STOP_SCHEDULING").exists():
            raise BudgetExceeded("deadline guard raised STOP_SCHEDULING")
        return BudgetGuard(mirror, events_path=logs / "box_events.jsonl")
    return BudgetGuard(logs / "gpu_budget.jsonl")


class BudgetGuard:
    """Append-only ledger. State is always recomputed from disk."""

    def __init__(self, ledger_path: str | Path, ceiling_h: float = CEILING_H,
                 events_path: str | Path | None = None):
        self.path = Path(ledger_path)
        self.events_path = Path(events_path) if events_path else None
        self.ceiling_h = float(ceiling_h)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- reading

    def _rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    def state(self) -> BudgetState:
        """Recompute consumption from the ledger file.

        Consumption is the sum of closed intervals plus any still-open interval measured
        to now, each multiplied by its GPU count. An open interval is what makes a crash
        safe: restarting re-reads it and keeps billing.
        """
        consumed = 0.0
        open_start: _dt.datetime | None = None
        open_gpus = 1
        open_id: str | None = None

        for r in self._rows():
            ev = r.get("event")
            if ev == "instance_start":
                open_start = _dt.datetime.fromisoformat(r["utc"].replace("Z", "+00:00"))
                open_gpus = int(r.get("gpu_count", 1))
                open_id = r.get("instance_id")
            elif ev == "instance_stop":
                if open_start is not None:
                    end = _dt.datetime.fromisoformat(r["utc"].replace("Z", "+00:00"))
                    consumed += (end - open_start).total_seconds() / 3600.0 * open_gpus
                    open_start, open_gpus, open_id = None, 1, None
            elif ev == "manual_charge":
                consumed += float(r.get("hours", 0.0))

        if open_start is not None:
            now = _dt.datetime.now(_dt.timezone.utc)
            consumed += (now - open_start).total_seconds() / 3600.0 * open_gpus

        return BudgetState(
            consumed_h=consumed,
            remaining_h=max(0.0, self.ceiling_h - consumed),
            may_schedule=consumed < STOP_SCHEDULING_H,
            must_export=consumed >= EXPORT_H,
            must_terminate=consumed >= TERMINATE_H,
            instance_open=open_start is not None,
            instance_id=open_id,
        )

    # ---------------------------------------------------------------- writing

    def _append(self, **fields) -> dict:
        row = {"utc": _utc(), **fields}
        billing = fields.get("event") in ("instance_start", "instance_stop", "manual_charge",
                                          "ledger_open", "ledger_close")
        if self.events_path is not None and billing:
            raise BudgetExceeded("billing events are written only by the ledger of record, never from a mirror")
        target = self.events_path if self.events_path is not None else self.path
        with target.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        return row

    def open_ledger(self, note: str = "") -> dict:
        return self._append(event="ledger_open", ceiling_a100_hours=self.ceiling_h,
                            consumed_a100_hours=0.0, note=note)

    def instance_start(self, instance_id: str, gpu_count: int = 1, note: str = "") -> dict:
        if self.state().instance_open:
            raise BudgetExceeded("an instance interval is already open; stop it first")
        return self._append(event="instance_start", instance_id=instance_id,
                            gpu_count=int(gpu_count), note=note)

    def instance_stop(self, instance_id: str, stop_verified: bool, note: str = "") -> dict:
        row = self._append(event="instance_stop", instance_id=instance_id,
                           stop_verified=bool(stop_verified), note=note)
        if not stop_verified:
            # recorded, then surfaced: an unverified stop is a spend risk, not a detail
            raise BudgetExceeded(
                f"instance {instance_id} stop NOT verified -- verify state and report immediately")
        return row

    def charge(self, hours: float, reason: str) -> dict:
        """Record provider-billed hours that wall-clock intervals do not capture."""
        return self._append(event="manual_charge", hours=float(hours), reason=reason)

    def close_ledger(self, note: str = "") -> dict:
        st = self.state()
        return self._append(event="ledger_close", consumed_a100_hours=round(st.consumed_h, 4),
                            remaining_a100_hours=round(st.remaining_h, 4),
                            instance_open=st.instance_open, note=note)

    # ----------------------------------------------------------------- gating

    def require_schedulable(self, stage: str, estimated_h: float) -> BudgetState:
        """Gate before launching GPU work. Raises rather than letting spend drift."""
        st = self.state()
        if st.must_terminate:
            raise BudgetExceeded(
                f"{stage}: {st.consumed_h:.2f}h consumed >= {TERMINATE_H}h -- terminate now")
        if not st.may_schedule:
            raise BudgetExceeded(
                f"{stage}: {st.consumed_h:.2f}h consumed >= {STOP_SCHEDULING_H}h -- "
                "no new GPU work may be scheduled")
        if st.consumed_h + estimated_h > self.ceiling_h:
            raise BudgetExceeded(
                f"{stage}: {st.consumed_h:.2f}h + {estimated_h:.2f}h estimated would exceed "
                f"the {self.ceiling_h}h ceiling -- reduce scope through a logged amendment")
        self._append(event="stage_admitted", stage=stage, estimated_hours=estimated_h,
                     consumed_a100_hours=round(st.consumed_h, 4))
        return st

    def amend_reserve(self, stage: str, hours: float, reason: str) -> dict:
        """Reserve may be reassigned only through a logged amendment, never silently."""
        st = self.state()
        if st.consumed_h + hours > self.ceiling_h:
            raise BudgetExceeded("an amendment cannot raise the 30h ceiling")
        return self._append(event="reserve_amendment", stage=stage, hours=float(hours),
                            reason=reason, remaining_a100_hours=round(st.remaining_h, 4))
