#!/usr/bin/env python3
"""Pre-GPU check: do the pinned endpoint files still exist remotely with the recorded hashes?

Metadata only -- compares the hub's server-reported LFS SHA-256 (and size) at each pinned
revision with the hashes recorded in the original run metadata. This establishes REMOTE
PRESENCE and SERVER-REPORTED identity. It is not local byte verification, which
prepare_endpoints.py performs on the GPU box after download.

Run:  python3 -I scripts/check_endpoint_remotes.py
"""
from __future__ import annotations

import json
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_recovered as fr  # noqa: E402  (token handling, cross-host auth stripping)


def tree(repo: str, rtype: str, rev: str, sub: str) -> dict:
    base = f"https://{fr.HOST}/api/{'datasets' if rtype == 'dataset' else 'models'}/{repo}/tree/{rev}"
    url = base + (f"/{urllib.parse.quote(sub)}" if sub else "") + "?recursive=true&expand=true"
    out, nxt = [], url
    while nxt:
        with fr._OPENER.open(fr._req(nxt), timeout=60) as resp:
            out.extend(json.loads(resp.read()))
            link = resp.headers.get("Link", "")
        nxt = None
        for part in link.split(","):
            if 'rel="next"' in part:
                nxt = part[part.find("<") + 1: part.find(">")]
    pre = (sub.rstrip("/") + "/") if sub else ""
    return {e["path"][len(pre):]: {"size": e.get("size"), "lfs_sha256": (e.get("lfs") or {}).get("oid"),
                                   "git_oid": e.get("oid")} for e in out if e.get("type") == "file"}


def main() -> int:
    src = json.loads((ROOT / "manifests/endpoint_sources.json").read_text())
    report, problems = {}, 0
    for key, ep in src.items():
        if ep["status"] == "direct":
            remote = tree(ep["repository"], ep["repository_type"], ep["revision"], ep["subpath"] or "")
            expect = ep["verified_files"]
            what = "endpoint files"
        else:  # merged: the adapter inputs are what exists remotely; outputs are made on the box
            remote = tree(ep["repository"], ep["repository_type"], ep["revision"], ep["subpath"] or "")
            expect = ep["merge"]["adapter_input_files"]
            what = "adapter inputs (merged outputs are produced and verified on the box)"
        rows = []
        for f in expect:
            r = remote.get(f["path"])
            if r is None:
                rows.append({"path": f["path"], "status": "ABSENT_REMOTE"}); problems += 1; continue
            if r["lfs_sha256"]:
                ok = r["lfs_sha256"] == f["sha256"] and (f.get("size_bytes") in (None, r["size"]))
                rows.append({"path": f["path"], "status": "server_hash_match" if ok else "SERVER_HASH_MISMATCH",
                             "server_lfs_sha256": r["lfs_sha256"], "recorded_sha256": f["sha256"]})
            else:
                # small non-LFS file: the hub reports a git blob id, not a sha256 of content
                rows.append({"path": f["path"], "status": "present_non_lfs_size_" +
                             ("match" if f.get("size_bytes") in (None, r["size"]) else "MISMATCH"),
                             "server_size": r["size"], "recorded_size": f.get("size_bytes")})
            problems += rows[-1]["status"].endswith(("MISMATCH",)) or "MISMATCH" in rows[-1]["status"]
        report[key] = {"repository": ep["repository"], "revision": ep["revision"], "subpath": ep["subpath"],
                       "checked": what, "files": rows}
        lfs = sum(r["status"] == "server_hash_match" for r in rows)
        small = sum(r["status"].startswith("present_non_lfs") for r in rows)
        print(f"{key:6s} {ep['repository']}@{ep['revision'][:12]} {ep['subpath'] or ''}: "
              f"{lfs} LFS hash matches, {small} small files present, "
              f"{len(rows) - lfs - small} problems ({what})")
    if src["B_dpo"]["base"]:
        b = src["B_dpo"]["base"]
        report["B_dpo_merge_base"] = {"repository": b["repository"], "revision": b["revision"],
                                      "note": "same repository and revision as B_it, checked above"}
    report["level"] = "remote presence + server-reported hash; NOT local byte verification"
    (ROOT / "manifests/endpoint_remote_check.json").write_text(json.dumps(report, indent=2) + "\n")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
