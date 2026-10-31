"""Build the instance onstart script that installs and supervises the deadline guard.

The guard must exist before any SSH session or GPU work, so it travels inside the onstart
script itself (Vast caps onstart at 4,048 characters; the guard is embedded docstring-stripped,
gzip-compressed and base64-encoded, as the API reference recommends for long scripts).

The script, in order:
  1. unpacks the guard to $G (default /root/dsteer-guard) and clears worker pids left from before
     the container started (those processes are gone; their pids may be reused);
  2. starts a coreutils-only backstop timer that fires 15 minutes after the hard deadline even if
     Python never installs (best-effort: it signals the init process, which may ignore it);
  3. installs python3 only if absent;
  4. runs the guard in a detached respawn loop (setsid + nohup on Linux; nohup alone where setsid
     is absent, e.g. the macOS test host), so it outlives SSH sessions and restarts on a crash.
Nothing in it reads or writes any API key; the guard reads the per-instance key from its own
environment at stop time.
"""
from __future__ import annotations

import ast
import base64
import gzip
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ONSTART_LIMIT = 4048


def guard_payload() -> str:
    tree = ast.parse((ROOT / "scripts" / "deadline_guard.py").read_text())
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef)) and n.body and isinstance(n.body[0], ast.Expr) \
                and isinstance(getattr(n.body[0], "value", None), ast.Constant) and isinstance(n.body[0].value.value, str):
            n.body = n.body[1:] or [ast.Pass()]
    src = "from __future__ import annotations\n" + ast.unparse(tree).replace("from __future__ import annotations\n", "")
    return base64.b64encode(gzip.compress(src.encode(), 9, mtime=0)).decode()


def build() -> str:
    script = f"""#!/bin/bash
G=${{DSTEER_GUARD_DIR:-/root/dsteer-guard}}; mkdir -p $G/workers $G/flags; rm -f $G/workers/*.pid
echo '{guard_payload()}' | base64 -d | gunzip > $G/deadline_guard.py
( d=${{DSTEER_STOP_AT%.*}}; s=$(( ${{d:-0}} + 900 - $(date +%s) )); [ $s -gt 0 ] && sleep $s; kill -TERM ${{DSTEER_GUARD_INIT_PID:-1}} ) >/dev/null 2>&1 &
echo $! > $G/backstop.pid
command -v python3 >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 >/dev/null)
S=$(command -v setsid || true)
nohup $S bash -c "echo \\$\\$ > $G/loop.pid; while true; do python3 $G/deadline_guard.py >> $G/guard.out 2>&1; sleep ${{DSTEER_GUARD_RESPAWN_S:-10}}; done" >/dev/null 2>&1 &
"""
    if len(script) > ONSTART_LIMIT:
        raise ValueError(f"onstart script is {len(script)} chars; Vast's limit is {ONSTART_LIMIT}")
    return script
