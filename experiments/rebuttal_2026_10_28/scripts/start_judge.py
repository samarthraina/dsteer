#!/usr/bin/env python3
"""Start or stop one judge server on the box, through the vendored primary launcher.

  original     start_judge_server.frozen_command(): Qwen3.5-27B @ fc05daec, BF16, text-only,
               prefix caching, max_model_len 8192, vllm==0.26.0 verified -- unchanged.
  independent  start_judge_server.build_command() for microsoft/phi-4 at the frozen revision,
               BF16, unquantised, served under the alias the scorer requests.

Waits for /v1/models and checks the served alias before returning, so a scorer can never
talk to the wrong model. Run inside the judge environment.

Run:  python scripts/start_judge.py start --judge original --port 8000
      python scripts/start_judge.py stop --judge original
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor/primary_snapshot"))
sys.path.insert(0, str(ROOT / "vendor/primary_snapshot/scripts"))
PIDS = ROOT / "logs" / "judge_servers.json"
ORIGINAL_ALIAS = "Qwen/Qwen3.5-27B@fc05daec18b0a78c049392ed2e771dde82bdf654"


def command(judge: str, port: int, gpu_util: float) -> tuple[list[str], str]:
    import start_judge_server as sjs
    if judge == "original":
        return sjs.frozen_command(port=port, gpu_util=gpu_util, api_key="EMPTY"), ORIGINAL_ALIAS
    man = json.loads((ROOT / "manifests/independent_judge.json").read_text())
    alias = f"{man['repository']}@{man['revision']}"
    return sjs.build_command(model=man["repository"], port=port, gpu_util=gpu_util, dtype="bfloat16",
                             max_model_len=8192, prefix_caching=True,  # as the frozen original judge
                             revision=man["revision"], served_model_name=alias,
                             api_key="EMPTY"), alias


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["start", "stop", "command"])
    ap.add_argument("--judge", required=True, choices=["original", "independent"])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    pids = json.loads(PIDS.read_text()) if PIDS.exists() else {}

    if a.action == "stop":
        p = pids.pop(a.judge, None)
        if p:
            try:
                os.killpg(p["pgid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
        PIDS.write_text(json.dumps(pids, indent=2))
        print(f"{a.judge} judge stopped")
        return 0

    cmd, alias = command(a.judge, a.port, a.gpu_util)
    if a.action == "command":
        print(" ".join(cmd))
        return 0
    log = (ROOT / "logs" / f"judge_{a.judge}.log").open("a")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pids[a.judge] = {"pid": proc.pid, "pgid": os.getpgid(proc.pid), "port": a.port, "alias": alias,
                     "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cmd": cmd}
    PIDS.write_text(json.dumps(pids, indent=2))
    t0 = time.monotonic()
    while time.monotonic() - t0 < a.timeout:
        if proc.poll() is not None:
            print(f"judge server exited with {proc.returncode}; see logs/judge_{a.judge}.log")
            return 2
        try:
            # the frozen command serves with --api-key EMPTY, so an unauthenticated probe gets 401 forever
            req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/models",
                                         headers={"Authorization": "Bearer EMPTY"})
            with urllib.request.urlopen(req, timeout=5) as r:
                served = [m["id"] for m in json.loads(r.read())["data"]]
            if alias not in served:
                print(f"HARD STOP: server serves {served}, expected {alias}")
                os.killpg(pids[a.judge]["pgid"], signal.SIGTERM)
                return 3
            pids[a.judge]["ready_s"] = round(time.monotonic() - t0, 1)
            PIDS.write_text(json.dumps(pids, indent=2))
            print(f"{a.judge} judge ready as {alias} after {pids[a.judge]['ready_s']} s")
            return 0
        except Exception:  # noqa: BLE001 -- still starting
            time.sleep(5)
    print("judge server did not become ready in time; stopping it")
    try:
        os.killpg(pids[a.judge]["pgid"], signal.SIGTERM)
    except ProcessLookupError:
        pass
    return 1   # infrastructure: retried by run_plan, unlike an alias mismatch


if __name__ == "__main__":
    raise SystemExit(main())
