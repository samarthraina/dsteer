#!/usr/bin/env python3
"""Execute a stage plan on the GPU box, under stage caps and the global ledger.

  - steps run in plan order (grouped by model to limit reloads; data dependencies kept);
  - before each step: global guard (mirror of the Mac ledger), then the stage cap. A
    mandatory step that would exceed its cap may draw S7 reserve only within the
    pre-registered amendment, logged via amend_reserve; otherwise the run stops and reports.
    Optional steps that do not fit are skipped and recorded, never squeezed in;
  - exit codes 2-6 are scientific/identity stops (hash mismatch, parity failure, refused
    condition, digest drift) and are never retried; other failures get at most two retries;
  - '@selected:P:D' placeholders resolve from manifests/comparator_selection.json; with no
    dev-selected candidate the step is skipped and recorded, not substituted;
  - only successful invocations are appended to reports/COMMANDS.txt.

Run:  DSTEER_ROLE=box python scripts/run_plan.py --scope full [--dry-run] [--from-stage S2...]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from budget_guard import BudgetExceeded, guard_for  # noqa: E402

SCIENTIFIC_STOP = {2, 3, 4, 5, 6}
MAX_INFRA_RETRIES = 2
RESERVE_RULE = {"S1_load_parity_profile": 1.0, "S2_refusal_construction_calibration": 0.5}
SHORT = {"installation": "install", "reversal": "reverse"}


def caps() -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    from plan import CAPS
    return dict(CAPS)


def progress(run_root: Path) -> list[dict]:
    p = run_root / "logs" / "plan_progress.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def log_progress(run_root: Path, row: dict) -> None:
    with (run_root / "logs" / "plan_progress.jsonl").open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def step_key(argv: list[str]) -> tuple:
    """A step's identity without its --estimate-hours value, which re-planning from measured
    rates rewrites; otherwise every completed step would look new after profiling."""
    out, skip = [], False
    for x in argv:
        if skip:
            skip = False
            continue
        if x == "--estimate-hours":
            skip = True
            continue
        out.append(x)
    return tuple(out)


def resolve(step: dict, run_root: Path) -> tuple[list[str] | None, str | None]:
    argv = list(step["argv"])
    if "refusal_ablate_reference" in argv and "--pair" in argv:
        pair = argv[argv.index("--pair") + 1]
        sel = run_root / "outputs" / "directions" / pair / "dpo" / "reference_selection.json"
        rec = json.loads(sel.read_text()) if sel.exists() else {}
        if not rec.get("selected"):
            return None, f"reference selection for pair {pair} not available ({rec.get('blocked', 'not run')}); arm not constructed"
    for i, a in enumerate(argv):
        if a.startswith("@selected:"):
            _, pair, direction = a.split(":")
            sel = run_root / "manifests" / "comparator_selection.json"
            entry = json.loads(sel.read_text())["tuned"].get(f"{pair}_{direction}", {}) if sel.exists() else {}
            m = entry.get("dev_selected_multiplier")
            if m is None:
                return None, f"no dev-selected candidate for {pair}_{direction} ({entry.get('status')})"
            argv[i] = f"cal_refusal_add_{SHORT[direction]}_m{m}"
        if a.startswith("refusal_add_tuned_") and "--condition" in argv[max(0, i - 1):i + 1]:
            d = "installation" if a.endswith("install") else "reversal"
            pair = argv[argv.index("--pair") + 1]
            sel = run_root / "manifests" / "comparator_selection.json"
            entry = json.loads(sel.read_text())["tuned"].get(f"{pair}_{d}", {}) if sel.exists() else {}
            if entry.get("status") != "verified":
                return None, f"tuned row not verified for {pair}_{d} ({entry.get('status')})"
            if not entry.get("tuned_differs_from_matched"):
                return None, f"tuned multiplier equals the matched 1.0 row for {pair}_{d}; not duplicated"
    return argv, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default=str(ROOT))
    ap.add_argument("--scope", default="full", choices=["full", "reduced"])
    ap.add_argument("--from-stage", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-optional", action="store_true", help="defer every optional step (recorded as skipped)")
    ap.add_argument("--stop-after", default=None, help="stage prefix; scheduling ends once a later stage is reached")
    a = ap.parse_args()
    rr = Path(a.run_root).resolve()
    steps = json.loads((rr / "configs" / f"stage_plan_{a.scope}.json").read_text())
    cap = caps()
    # a multiset: S2 legitimately runs `start_judge start` twice, and one completed start must
    # not mark the second as done (that skipped a judge start on 2026-10-29)
    from collections import Counter
    done = Counter((p["stage"], step_key(p["argv"])) for p in progress(rr) if p["status"] == "ok")
    used: dict[str, float] = {}
    for p in progress(rr):
        used[p["stage"]] = used.get(p["stage"], 0.0) + p.get("hours", 0.0)
    reserve_drawn = {}
    started = a.from_stage is None
    total_est = 0.0
    seen_stop = False

    for step in steps:
        st = step["stage"]
        if a.stop_after and seen_stop and not st.startswith(a.stop_after):
            print(f"END   --stop-after {a.stop_after}: not scheduling {st} or later")
            break
        seen_stop = seen_stop or bool(a.stop_after and st.startswith(a.stop_after))
        started = started or st.startswith(a.from_stage or "")
        if not started or step["env"] == "mac":
            continue
        argv, skip = resolve(step, rr)
        if a.skip_optional and step.get("optional"):
            skip = skip or "optional step deferred (--skip-optional)"
        if skip:
            print(f"SKIP  [{st}] {skip}")
            if not a.dry_run:
                log_progress(rr, {"stage": st, "argv": step["argv"], "status": "skipped", "reason": skip, "hours": 0.0})
            continue
        if done[(st, step_key(argv))] > 0:
            done[(st, step_key(argv))] -= 1
            continue
        est = step["est_h"]
        room = cap[st] + reserve_drawn.get(st, 0.0) - used.get(st, 0.0)
        if est > room:
            if step.get("optional"):
                print(f"SKIP  [{st}] optional step does not fit the stage cap ({est:.3f} h > {room:.3f} h)")
                if not a.dry_run:
                    log_progress(rr, {"stage": st, "argv": argv, "status": "skipped_cap", "hours": 0.0})
                continue
            extra = min(RESERVE_RULE.get(st, 0.0) - reserve_drawn.get(st, 0.0), est - room)
            if extra > 0 and est <= room + extra:
                if not a.dry_run:
                    g = guard_for(rr)
                    g.amend_reserve(st, extra, "pre-registered rule in configs/amendments/2026-10-28-01-recovery-design-and-reserve.json")
                reserve_drawn[st] = reserve_drawn.get(st, 0.0) + extra
                print(f"RESERVE [{st}] +{extra:.3f} h drawn from S7 under the pre-registered amendment")
            else:
                print(f"STOP  [{st}] mandatory step exceeds the stage cap and permitted reserve; scheduling stops here")
                return 7
        total_est += est
        print(f"{'PLAN ' if a.dry_run else 'RUN  '}[{st}] ~{est:.3f} h  {' '.join(argv)}")
        if a.dry_run:
            used[st] = used.get(st, 0.0) + est
            continue
        try:
            guard_for(rr).require_schedulable(st, est)
        except BudgetExceeded as exc:
            print(f"STOP  global budget: {exc}")
            return 8
        for attempt in range(MAX_INFRA_RETRIES + 1):
            t0 = time.time()
            env = {**os.environ, "DSTEER_ROLE": "box", "HF_HOME": str(rr / ".cache" / "hf"),
                   "NLTK_DATA": str(rr / ".cache" / "nltk_data")}
            gdir = Path(os.environ.get("DSTEER_GUARD_DIR", "/root/dsteer-guard"))
            proc = subprocess.Popen(argv, cwd=rr, env=env, start_new_session=True)
            pidf = gdir / "workers" / f"{proc.pid}.pid"
            try:
                pidf.parent.mkdir(parents=True, exist_ok=True)
                pidf.write_text(str(proc.pid))
                proc.wait()
            finally:
                pidf.unlink(missing_ok=True)
            r = proc
            hours = (time.time() - t0) / 3600
            used[st] = used.get(st, 0.0) + hours
            row = {"stage": st, "argv": argv, "attempt": attempt + 1, "returncode": r.returncode,
                   "hours": round(hours, 4), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            if r.returncode == 0:
                log_progress(rr, {**row, "status": "ok"})
                with (rr / "reports" / "COMMANDS.txt").open("a") as fh:
                    fh.write(f"# {row['utc']} [{st}] ok in {hours:.3f} h\n{' '.join(argv)}\n")
                break
            if r.returncode in SCIENTIFIC_STOP:
                log_progress(rr, {**row, "status": "scientific_stop"})
                print(f"STOP  scientific/identity stop (exit {r.returncode}); not retried")
                return r.returncode
            log_progress(rr, {**row, "status": "infra_failure"})
            if attempt == MAX_INFRA_RETRIES:
                print(f"STOP  infrastructure failure after {MAX_INFRA_RETRIES} retries")
                return 9
    if a.dry_run:
        print(f"\nplan total (raw step estimates): {total_est:.2f} h")
        for st in cap:
            print(f"  {st:40s} {used.get(st, 0.0):6.3f} h of cap {cap[st]}"
                  + (f" (+{reserve_drawn[st]:.2f} reserve)" if st in reserve_drawn else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
