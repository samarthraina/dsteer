#!/usr/bin/env python3
"""Instance-resident shutdown: the real generated onstart script, run with NO Mac component.

No watchdog, no ledger mirror, no SSH and no Mac process takes part in any test below; this is
the Mac-asleep / Mac-disconnected case. A local stub plays the Vast REST API and records every
request. Real child processes stand in for GPU workers and for the container's init process.

Pins:
  - the instance not stopping itself at the deadline when the Mac is gone;
  - stopping by killing processes only, without the API stop that ends GPU billing;
  - acting on any instance other than $CONTAINER_ID, or ever destroying;
  - the per-instance key leaking into any file;
  - a crashed guard not being respawned;
  - a deadline extended from the instance side;
  - a rate-limited (429) stop being abandoned;
  - an unbounded instance (no deadline) being allowed to run;
  - the self-stop test repeating after a restart.

Run:  python3 -I tests/test_deadline_guard.py
"""
from __future__ import annotations

import http.server
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import onstart  # noqa: E402

SECRET = "test-container-key-7f3a9c"
FAILURES: list[str] = []
REQ: list[dict] = []
MODE = {"status": 200, "fail_first": 0}


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


class Stub(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _rec(self, method):
        n = int(self.headers.get("Content-Length", 0) or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        REQ.append({"method": method, "path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        if MODE["fail_first"] > 0:
            MODE["fail_first"] -= 1
            code, out = 429, {"detail": "API requests too frequent"}
        else:
            code, out = MODE["status"], {"success": MODE["status"] == 200}
        b = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_PUT(self):
        self._rec("PUT")

    def do_DELETE(self):
        self._rec("DELETE")


def wait_for(pred, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.1)
    return False


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


class Box:
    """One simulated instance: the onstart script, a dummy init process, fake GPU workers."""

    def __init__(self, port, stop_in_s, **env):
        self.g = Path(tempfile.mkdtemp())
        self.init = subprocess.Popen(["sleep", "600"], start_new_session=True)
        self.workers = [subprocess.Popen(["sleep", "600"], start_new_session=True) for _ in range(2)]
        (self.g / "workers").mkdir(parents=True)
        # left over from before this container started: onstart must clear it, since the pid may be reused
        (self.g / "workers" / "stale.pid").write_text(str(self.init.pid))
        self.env = {"PATH": os.environ["PATH"], "HOME": str(self.g), "DSTEER_GUARD_DIR": str(self.g),
                    "DSTEER_GUARD_INIT_PID": str(self.init.pid), "DSTEER_VAST_API": f"http://127.0.0.1:{port}",
                    "CONTAINER_ID": "777", "CONTAINER_API_KEY": SECRET,
                    "DSTEER_GUARD_TICK_S": "0.3", "DSTEER_GUARD_RESPAWN_S": "0.5",
                    "DSTEER_GUARD_TERM_GRACE_S": "1", "DSTEER_GUARD_API_GRACE_S": "3", **env}
        if stop_in_s is not None:
            self.env["DSTEER_STOP_AT"] = str(time.time() + stop_in_s)
        # own session: the backstop timer and loop must never share the test runner's process group
        subprocess.run(["bash", "-c", onstart.build()], env=self.env, check=True, timeout=30,
                       start_new_session=True)
        wait_for(lambda: (self.g / "loop.pid").exists(), 5)
        self.stale_cleared = not (self.g / "workers" / "stale.pid").exists()
        for i, w in enumerate(self.workers):    # jobs register after the container has started
            (self.g / "workers" / f"w{i}.pid").write_text(str(w.pid))

    def stops(self):
        return [r for r in REQ if r["method"] == "PUT" and r["body"] == {"state": "stopped"}]

    def close(self):
        mine = os.getpgid(0)
        for f in ("loop.pid", "backstop.pid"):
            try:
                pid = int((self.g / f).read_text())
                pg = os.getpgid(pid)
                if pg != mine:                      # never signal our own process group
                    os.killpg(pg, signal.SIGKILL)
                else:
                    os.kill(pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        subprocess.run(["pkill", "-f", str(self.g / "deadline_guard.py")], capture_output=True)
        for p in [self.init, *self.workers]:
            try:
                if os.getpgid(p.pid) != mine:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(self.g, ignore_errors=True)


def secret_in_files(g: Path) -> list[str]:
    return [str(p) for p in g.rglob("*") if p.is_file() and SECRET in p.read_text(errors="ignore")]


def main() -> int:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        print("test_stops_itself_with_no_mac")
        REQ.clear(); MODE.update(status=200, fail_first=0)
        b = Box(port, stop_in_s=4)
        try:
            hb = json.loads((b.g / "heartbeat.json").read_text()) if wait_for(lambda: (b.g / "heartbeat.json").exists(), 10) else {}
            check("guard running with deadline and key present (key not logged)",
                  hb.get("container_id") == "777" and hb.get("key_present") is True and hb.get("seconds_left", 0) > 0, str(hb))
            check("scheduling and export flags raised before the deadline",
                  wait_for(lambda: (b.g / "flags/STOP_SCHEDULING").exists() and (b.g / "flags/EXPORT_NOW").exists(), 3))
            check("instance stops itself via the API after the deadline", wait_for(lambda: b.stops(), 15))
            s = b.stops()[0] if b.stops() else {}
            check("stop targets only $CONTAINER_ID", all(r["path"].rstrip("/") == "/api/v0/instances/777" for r in REQ), str({r['path'] for r in REQ}))
            check("authorised with the per-instance key", s.get("auth") == f"Bearer {SECRET}")
            check("never destroys", not any(r["method"] == "DELETE" for r in REQ))
            # poll() reaps: a killed child of this test stays a zombie, which kill(pid, 0) still reports
            check("GPU workers terminated before the API stop", all(w.poll() is not None for w in b.workers))
            check("worker pids from before the container start are cleared", b.stale_cleared)
            check("pid files of killed workers are removed", not list((b.g / "workers").glob("w*.pid")))
            check("self_stop.json records the accepted stop", (b.g / "self_stop.json").exists())
            check("per-instance key appears in no file", not secret_in_files(b.g), str(secret_in_files(b.g)))
        finally:
            b.close()

        print("test_respawn_after_guard_crash")
        REQ.clear()
        b = Box(port, stop_in_s=5)
        try:
            wait_for(lambda: (b.g / "heartbeat.json").exists(), 10)
            pid1 = json.loads((b.g / "heartbeat.json").read_text())["pid"]
            os.kill(pid1, signal.SIGKILL)
            respawned = wait_for(lambda: json.loads((b.g / "heartbeat.json").read_text()).get("pid") != pid1, 6)
            check("killed guard is respawned by the loop", respawned)
            check("respawned guard still stops the instance", wait_for(lambda: b.stops(), 15))
        finally:
            b.close()

        print("test_rate_limited_stop_retried")
        REQ.clear(); MODE.update(status=200, fail_first=2)
        b = Box(port, stop_in_s=1, DSTEER_GUARD_API_GRACE_S="30")   # backoff 2 s + 4 s must fit
        try:
            check("stop accepted after two 429 responses", wait_for(lambda: (b.g / "self_stop.json").exists(), 20))
            att = json.loads((b.g / "self_stop.json").read_text())["attempts"] if (b.g / "self_stop.json").exists() else None
            check("third attempt accepted", att == 3, str(att))
        finally:
            b.close()

        print("test_api_unreachable_best_effort_fallback")
        REQ.clear(); MODE.update(status=200, fail_first=0)
        b = Box(srv.server_address[1] + 1 if False else 9, stop_in_s=1)   # port 9: nothing listens
        try:
            check("fallback signals the init process after the API grace", wait_for(lambda: b.init.poll() is not None, 20))
            log = (b.g / "guard.jsonl").read_text()
            check("fallback logged as best-effort, not as a verified stop",
                  "best_effort_container_exit" in log and not (b.g / "self_stop.json").exists())
        finally:
            b.close()

        print("test_no_deadline_fails_safe")
        REQ.clear()
        b = Box(port, stop_in_s=None)
        try:
            check("an instance with no deadline stops itself immediately", wait_for(lambda: b.stops(), 10))
        finally:
            b.close()

        print("test_deadline_cannot_be_extended")
        REQ.clear()
        b = Box(port, stop_in_s=3)
        try:
            (b.g / "stop_at.json").write_text(json.dumps({"stop_at_epoch": time.time() + 3600}))
            check("a later file deadline does not extend the hard deadline", wait_for(lambda: b.stops(), 15))
        finally:
            b.close()
        REQ.clear()
        b = Box(port, stop_in_s=3600)
        try:
            (b.g / "stop_at.json").write_text(json.dumps({"stop_at_epoch": time.time() + 1}))
            check("an earlier file deadline shortens it", wait_for(lambda: b.stops(), 15))
        finally:
            b.close()

        print("test_self_stop_test_runs_once")
        REQ.clear()
        b = Box(port, stop_in_s=3600)
        try:
            (b.g / "self_stop_test.json").write_text(json.dumps({"nonce": "n-123"}))
            check("one-shot self-stop test stops the instance", wait_for(lambda: b.stops(), 10))
            n1 = len(b.stops())
            pid = json.loads((b.g / "heartbeat.json").read_text())["pid"]
            os.kill(pid, signal.SIGKILL)                    # simulated restart
            time.sleep(3)
            check("nonce consumed: no repeat after restart", len(b.stops()) == n1, f"{n1} -> {len(b.stops())}")
        finally:
            b.close()
    finally:
        srv.shutdown()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
