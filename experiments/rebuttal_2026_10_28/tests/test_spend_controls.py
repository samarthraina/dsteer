#!/usr/bin/env python3
"""Spend controls against a stub vastai CLI. No real instance, no network, no real ledger.

The stub keeps instance states in a JSON file. The account always holds a second, unrelated
running instance, so every test also checks that it is never touched.

Pins:
  - creating a paid instance without explicit spend authorisation;
  - acting on an instance this run did not create;
  - a stop self-test that does not actually observe the stopped state;
  - the watchdog failing to stop at the 29.5 h threshold, or stopping the wrong instance;
  - unledgered spend (instance running with no open ledger interval) going unnoticed;
  - an unverifiable stop being recorded as verified.

Run:  python3 -I tests/test_spend_controls.py
"""
from __future__ import annotations

import datetime as _dt
import json
import time
import threading
import signal
import http.server
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAILURES: list[str] = []

STUB = r'''#!/usr/bin/env python3
import json, sys, os
db = os.environ["STUB_DB"]
s = json.load(open(db))
a = [x for x in sys.argv[1:] if x != "--raw"]
def save(): json.dump(s, open(db, "w"))
if a[:2] == ["show", "user"]: print(json.dumps({"id": 1, "username": "u"}))
elif a[:2] == ["show", "instances"]: print(json.dumps([{"id": int(k), "actual_status": v} for k, v in s["inst"].items()]))
elif a[:2] == ["show", "instance"]: print(json.dumps({"id": int(a[2]), "actual_status": s["inst"][a[2]]}))
elif a[:2] == ["stop", "instance"]:
    if not s.get("stuck"): s["inst"][a[2]] = "stopped"
    s.setdefault("log", []).append(["stop", a[2]]); save(); print("ok")
elif a[:2] == ["start", "instance"]:
    s["inst"][a[2]] = "running"; s.setdefault("log", []).append(["start", a[2]]); save(); print("ok")
elif a[:2] == ["create", "instance"]:
    s["inst"]["777"] = "running"; s.setdefault("log", []).append(["create", a[2]]); s["create_args"] = a; save()
    print(json.dumps({"success": True, "new_contract": 777}))
else: sys.exit(2)
'''


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def env(tmp: Path) -> dict:
    return {**os.environ, "VASTAI_BIN": str(tmp / "vastai"), "STUB_DB": str(tmp / "db.json"),
            "DSTEER_RUN_ROOT": str(tmp), "DSTEER_POLL_S": "0.2", "VASTAI_BOX_EXEC": "1",
            "DSTEER_BOX_GUARD_DIR": str(tmp / "guard")}


def vs(tmp, *args):
    return subprocess.run([sys.executable, "-I", str(ROOT / "scripts/vast_session.py"), *args],
                          capture_output=True, text=True, env=env(tmp))


def wd(tmp):
    return subprocess.run([sys.executable, "-I", str(ROOT / "scripts/watchdog.py"), "--once"],
                          capture_output=True, text=True, env=env(tmp))


def db(tmp):
    return json.loads((tmp / "db.json").read_text())


def ledger(tmp):
    p = tmp / "logs/gpu_budget.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def setup() -> Path:
    tmp = Path(tempfile.mkdtemp())
    (tmp / "logs").mkdir()
    (tmp / "vastai").write_text(STUB)
    (tmp / "vastai").chmod(0o755)
    (tmp / "db.json").write_text(json.dumps({"inst": {"555": "running"}}))   # unrelated instance
    return tmp

BOX_KEY = "per-instance-key-test-3c9d"


class _Api(http.server.BaseHTTPRequestHandler):
    db_path = None

    def log_message(self, *a):
        pass

    def do_PUT(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        iid = self.path.rstrip("/").split("/")[-1]
        s = json.loads(Path(self.db_path).read_text())
        ok = self.headers.get("Authorization") == f"Bearer {BOX_KEY}" and iid in s["inst"]
        if ok and body.get("state") == "stopped":
            s["inst"][iid] = "stopped"
            s.setdefault("log", []).append(["api_stop", iid, "per_instance_key"])
            Path(self.db_path).write_text(json.dumps(s))
        out = json.dumps({"success": ok}).encode()
        self.send_response(200 if ok else 401)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def start_box(t: Path, stop_at: float) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    import onstart
    _Api.db_path = str(t / "db.json")
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Api)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    init = subprocess.Popen(["sleep", "600"], start_new_session=True)
    g = t / "guard"
    env = {"PATH": os.environ["PATH"], "HOME": str(g), "DSTEER_GUARD_DIR": str(g), "CONTAINER_ID": "777",
           "CONTAINER_API_KEY": BOX_KEY, "DSTEER_STOP_AT": str(stop_at), "DSTEER_GUARD_INIT_PID": str(init.pid),
           "DSTEER_VAST_API": f"http://127.0.0.1:{srv.server_address[1]}", "DSTEER_GUARD_TICK_S": "0.3",
           "DSTEER_GUARD_RESPAWN_S": "0.5"}
    subprocess.run(["bash", "-c", onstart.build()], env=env, check=True, timeout=30, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 10 and not (g / "heartbeat.json").exists():
        time.sleep(0.1)
    return {"srv": srv, "init": init, "g": g}


def stop_box(box: dict) -> None:
    mine = os.getpgid(0)
    for f in ("loop.pid", "backstop.pid"):
        try:
            pid = int((box["g"] / f).read_text())
            if os.getpgid(pid) != mine:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
    subprocess.run(["pkill", "-f", str(box["g"] / "deadline_guard.py")], capture_output=True)
    box["init"].kill()
    box["srv"].shutdown()


def main() -> int:
    real_ledger = ROOT / "logs/gpu_budget.jsonl"
    before = real_ledger.read_bytes() if real_ledger.exists() else None

    print("test_create_requires_authorisation")
    t = setup()
    r = vs(t, "create", "--offer", "42")
    check("create without --authorize-spend refused", r.returncode != 0 and "REFUSED" in r.stdout + r.stderr)
    check("nothing created", "777" not in db(t)["inst"])
    r = vs(t, "create", "--offer", "42", "--authorize-spend", "test authorisation")
    check("authorised create records instance and opens ledger",
          r.returncode == 0 and (t / "logs/instance.json").exists() and ledger(t)[-1]["event"] == "instance_start")
    r = vs(t, "create", "--offer", "43", "--authorize-spend", "again")
    check("second create refused", r.returncode != 0)

    print("test_create_carries_guard_and_deadline")
    ca = db(t)["create_args"]
    inst = json.loads((t / "logs/instance.json").read_text())
    env_arg = ca[ca.index("--env") + 1] if "--env" in ca else ""
    check("hard deadline passed into the instance environment",
          f"DSTEER_STOP_AT={inst['hard_stop_at_epoch']}" in env_arg, env_arg)
    dl_h = (inst["hard_stop_at_epoch"] - time.time()) / 3600
    check("deadline equals the 29.5 h terminate threshold minus spend so far", 29.3 < dl_h <= 29.5, f"{dl_h:.3f} h")
    check("guard embedded via --onstart-cmd", "--onstart-cmd" in ca and "deadline_guard.py" in ca[ca.index("--onstart-cmd") + 1])

    print("test_selftest_both_stop_paths_with_real_guard")
    box = start_box(t, inst["hard_stop_at_epoch"])
    try:
        r = vs(t, "selftest-stop")
        res = json.loads((t / "logs/stop_selftest.json").read_text())
        check("selftest PASSED", res.get("verdict") == "PASSED", (r.stdout + r.stderr)[-300:])
        log = db(t)["log"]
        check("box-initiated stop made by the guard via the API with the per-instance key",
              ["api_stop", "777", "per_instance_key"] in log, str(log))
        check("mac-initiated stop also exercised", ["stop", "777"] in log)
        check("box-initiated stop verified by the Mac via instance state",
              res["box_initiated_stop"]["verified_by_mac_via_api"] is True)
        check("guard alive after each restart", res["guard_alive_after_restart"]["ok"] and res["guard_alive_final"]["ok"])
        check("self-stop nonce consumed (cannot repeat)", res["nonce_consumed"] is True)
        check("unrelated instance untouched", db(t)["inst"]["555"] == "running" and all(x[1] != "555" for x in log))
        mac_files = [p for p in t.rglob("*") if p.is_file() and "guard" not in p.parts]
        check("per-instance key never reaches Mac-side files",
              not any(BOX_KEY in p.read_text(errors="ignore") for p in mac_files))
        check("instance running again after a passed selftest", db(t)["inst"]["777"] == "running")
    finally:
        stop_box(box)

    print("test_selftest_fails_without_guard")
    t0 = setup()
    vs(t0, "create", "--offer", "42", "--authorize-spend", "x")
    r = vs(t0, "selftest-stop")
    res = json.loads((t0 / "logs/stop_selftest.json").read_text())
    check("no guard -> selftest FAILED", res.get("verdict", "").startswith("FAILED"), res.get("verdict"))
    check("instance stopped after the failed selftest", db(t0)["inst"]["777"] == "stopped")

    print("test_watchdog_threshold")
    ts = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=29.7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with (t / "logs/gpu_budget.jsonl").open("a") as fh:
        fh.write(json.dumps({"event": "manual_charge", "hours": 29.6, "reason": "simulated", "utc": ts}) + "\n")
    r = wd(t)
    beat = json.loads((t / "logs/watchdog_heartbeat.json").read_text())
    check("watchdog stops at >= 29.5 h", (beat["action"] or "").startswith("terminate_threshold_stop:verified"), str(beat))
    check("dedicated instance stopped", db(t)["inst"]["777"] == "stopped")
    check("unrelated instance untouched", db(t)["inst"]["555"] == "running")
    check("scheduling and export flags written", (t / "logs/flags/STOP_SCHEDULING").exists() and (t / "logs/flags/EXPORT_NOW").exists())

    print("test_unledgered_spend")
    t2 = setup()
    vs(t2, "create", "--offer", "42", "--authorize-spend", "x")
    vs(t2, "stop")
    s = db(t2); s["inst"]["777"] = "running"; (t2 / "db.json").write_text(json.dumps(s))  # restarted outside the ledger
    wd(t2)
    beat = json.loads((t2 / "logs/watchdog_heartbeat.json").read_text())
    check("instance running with no ledger interval is stopped", (beat["action"] or "").startswith("unledgered_running_stop"), str(beat))
    check("unrelated instance untouched", db(t2)["inst"]["555"] == "running")

    print("test_unverifiable_stop")
    t3 = setup()
    vs(t3, "create", "--offer", "42", "--authorize-spend", "x")
    s = db(t3); s["stuck"] = True; (t3 / "db.json").write_text(json.dumps(s))
    import importlib, types
    os.environ.update({"VASTAI_BIN": str(t3 / "vastai"), "STUB_DB": str(t3 / "db.json"), "DSTEER_RUN_ROOT": str(t3)})
    sys.path.insert(0, str(ROOT / "scripts"))
    import vast_session as V
    importlib.reload(V)
    V.wait_for = lambda iid, wanted, timeout_s=600: V.state(iid)   # no 10-minute wait in a test
    ok = V._stop(V.BudgetGuard(V.LOGS / "gpu_budget.jsonl"), 777, "test")
    check("stuck instance reported NOT verified", ok is False)
    last = ledger(t3)[-1]
    check("ledger records stop_verified False, never True", last["event"] == "instance_stop" and last["stop_verified"] is False)

    after = real_ledger.read_bytes() if real_ledger.exists() else None
    check("real run ledger untouched by these tests", before == after)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
