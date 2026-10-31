#!/usr/bin/env python3
"""Fetch recovered Task 027 responses and Task 025 construction tensors from the private
backup lineage, pinned to one immutable revision, and verify every byte locally.

Stdlib only. The HF token is read in-process from ~/.cache/huggingface/token (or HF_TOKEN)
and is never printed, logged or written anywhere. It is sent only to huggingface.co; on a
redirect to any other host (the LFS/CDN signed URL) the Authorization header is dropped.

Remote presence (the server's listing) and local verification (sha256 of the bytes we hold)
are recorded as separate facts. Only the second counts as verified.

Run:  python3 -I scripts/fetch_recovered.py --run-root . --project-root ../.. [--list-only]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = "samarthraina/dsteer-results"
REPO_TYPE = "dataset"
REVISION = "4f72531f763eff1c43f29769a6f76d6c06b7ca5a"
HOST = "huggingface.co"

T027_PREFIX = "backups/aaai-safety-audit/2026-08-18-task027-final-harmfulqa-evaluation-84b50df"
T025_PREFIX = "backups/aaai-safety-audit/2026-08-17-task025-primary-calibration-84b50df"
TENSORS = {
    "A": (f"{T025_PREFIX}/activation-artifacts/endpoint-A-3e74fd550f63/construction/activations.pt",
          "5c2feb3efd84599a8e7475931f7448545107c30246a60ff8022012f89cd7ffa1"),
    "B": (f"{T025_PREFIX}/activation-artifacts/endpoint-B-3e74fd550f63/construction/activations.pt",
          "c02721c604964d4261b85852654276e49496a6fede78263d9330d6a5307adfc5"),
}
TENSOR_BYTES = 1_445_073_373
INVENTORY_EXPECTED = "681217320bdee41c077d72cb64f3c52cd0690f66ea5c21015df6b339fd7796f7"
MAX_ATTEMPTS = 3


def utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _token(required: bool = True) -> str:
    t = os.environ.get("HF_TOKEN")
    if not t:
        p = Path.home() / ".cache" / "huggingface" / "token"
        t = p.read_text().strip() if p.is_file() else ""
    if not t and required:
        raise SystemExit("no Hugging Face credential resolves (HF_TOKEN or ~/.cache/huggingface/token)")
    return t


class _StripAuthOnHostChange(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).hostname != HOST:
            new.headers.pop("Authorization", None)
            new.unredirected_hdrs.pop("Authorization", None)
        return new


_OPENER = urllib.request.build_opener(_StripAuthOnHostChange)


def _req(url: str) -> urllib.request.Request:
    r = urllib.request.Request(url)
    # the GPU box holds no HF credential: public files go unauthenticated, private ones are
    # pre-fetched on the Mac and arrive already hash-verified
    if urllib.parse.urlparse(url).hostname == HOST and (t := _token(required=os.environ.get("DSTEER_ROLE") != "box")):
        r.add_unredirected_header("Authorization", f"Bearer {t}")
    r.add_header("User-Agent", "dsteer-rebuttal-fetch/1")
    return r


def list_tree(prefix: str) -> list[dict]:
    q = urllib.parse.quote(prefix)
    url = f"https://{HOST}/api/datasets/{REPO}/tree/{REVISION}/{q}?recursive=true&expand=true"
    out, nxt = [], url
    while nxt:
        with _OPENER.open(_req(nxt), timeout=60) as resp:
            out.extend(json.loads(resp.read()))
            link = resp.headers.get("Link", "")
        nxt = None
        for part in link.split(","):
            if 'rel="next"' in part:
                nxt = part[part.find("<") + 1: part.find(">")]
    return [e for e in out if e.get("type") == "file"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def download(path: str, dest: Path, expect_sha: str | None, expect_size: int | None) -> dict:
    """Atomic download with bounded retries. Never overwrites a verified file."""
    if dest.is_file() and expect_sha and sha256_file(dest) == expect_sha:
        return {"status": "already_verified", "attempts": 0}
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://{HOST}/datasets/{REPO}/resolve/{REVISION}/{urllib.parse.quote(path)}"
    part = dest.with_name(dest.name + ".part")
    last = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            h = hashlib.sha256()
            n = 0
            with _OPENER.open(_req(url), timeout=120) as resp, part.open("wb") as fh:
                for blk in iter(lambda: resp.read(1 << 22), b""):
                    fh.write(blk); h.update(blk); n += len(blk)
            got = h.hexdigest()
            if expect_size is not None and n != expect_size:
                raise IOError(f"size {n} != expected {expect_size}")
            if expect_sha and got != expect_sha:
                raise IOError(f"sha256 {got} != expected {expect_sha}")
            part.replace(dest)
            return {"status": "downloaded", "attempts": attempt, "bytes": n}
        except (urllib.error.URLError, IOError, TimeoutError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            part.unlink(missing_ok=True)
            time.sleep(2 * attempt)
    return {"status": "FAILED", "attempts": MAX_ATTEMPTS, "error": last}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default=".")
    ap.add_argument("--project-root", default="../..")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--skip-tensors", action="store_true")
    a = ap.parse_args()
    run_root, project = Path(a.run_root).resolve(), Path(a.project_root).resolve()
    dest_root = run_root / "inputs" / "recovered"

    # ---------- remote listing (presence only)
    t027 = list_tree(T027_PREFIX)
    t025 = list_tree(f"{T025_PREFIX}/activation-artifacts")
    remote = {e["path"]: {"size": e.get("size"),
                          "lfs_sha256": (e.get("lfs") or {}).get("oid")} for e in t027 + t025}

    rel = lambda p: p[len(T027_PREFIX) + 1:]
    steer = sorted(p for p in remote if p.startswith(T027_PREFIX) and "/steer/" in f"/{rel(p)}" and p.endswith(".jsonl"))
    scored = sorted(p for p in remote if p.startswith(T027_PREFIX) and "/scored/" in f"/{rel(p)}" and p.endswith(".jsonl"))
    inv_remote = f"{T027_PREFIX}/sha256_inventory.txt"
    print(f"remote listing @ {REVISION[:12]}: {len(t027)} files under task027 prefix; "
          f"steer jsonl {len(steer)}, scored jsonl {len(scored)}; "
          f"inventory {'present' if inv_remote in remote else 'ABSENT'}")
    for k, (p, s) in TENSORS.items():
        r = remote.get(p, {})
        print(f"tensor {k}: remote size {r.get('size')} lfs-sha {str(r.get('lfs_sha256'))[:16]}… "
              f"{'(listed)' if p in remote else '(NOT LISTED)'}")
    if a.list_only:
        return 0

    # ---------- inventory: download, compare with local reference
    record = {"generated_utc": utc(), "repository": REPO, "repo_type": REPO_TYPE,
              "revision": REVISION, "task027_prefix": T027_PREFIX, "task025_prefix": T025_PREFIX,
              "credential_handling": "token read in-process; never printed or stored; stripped on cross-host redirect",
              "files": []}
    inv_dest = dest_root / "task027" / "sha256_inventory.txt"
    r = download(inv_remote, inv_dest, None, remote.get(inv_remote, {}).get("size"))
    inv_hash = sha256_file(inv_dest) if inv_dest.is_file() else None
    local_ref = project / "figures-final/data/task027-final/sha256_inventory.txt"
    ref_hash = sha256_file(local_ref)
    record["inventory"] = {
        "remote_path": inv_remote, "download": r, "local_path": str(inv_dest),
        "sha256_downloaded": inv_hash, "sha256_local_reference": ref_hash,
        "sha256_expected": INVENTORY_EXPECTED,
        "downloaded_matches_reference": inv_hash == ref_hash,
        "downloaded_matches_expected": inv_hash == INVENTORY_EXPECTED,
    }
    print(f"inventory: downloaded {str(inv_hash)[:16]}… ref {ref_hash[:16]}… "
          f"match={inv_hash == ref_hash == INVENTORY_EXPECTED}")

    expected = {}
    for line in inv_dest.read_text().splitlines():
        if line.strip():
            h, p = line.split(None, 1)
            expected[p.strip()] = h

    # ---------- responses: download every one, hash against the inventory
    for p in steer + scored:
        rp = rel(p)
        exp = expected.get(rp)
        dest = dest_root / "task027" / rp
        res = download(p, dest, exp, remote[p]["size"])
        got = sha256_file(dest) if dest.is_file() else None
        record["files"].append({
            "kind": "steer" if p in steer else "scored", "remote_path": p, "relative": rp,
            "remote_size": remote[p]["size"], "remote_lfs_sha256": remote[p]["lfs_sha256"],
            "inventory_sha256": exp, "local_sha256": got,
            "local_bytes": dest.stat().st_size if dest.is_file() else None,
            "verified": bool(exp) and got == exp, "download": res})

    # ---------- tensors
    if not a.skip_tensors:
        for k, (p, exp) in TENSORS.items():
            dest = dest_root / "task025" / f"pair_{k}" / "activations.pt"
            res = download(p, dest, exp, TENSOR_BYTES)
            got = sha256_file(dest) if dest.is_file() else None
            record["files"].append({
                "kind": f"tensor_{k}", "remote_path": p, "remote_size": remote.get(p, {}).get("size"),
                "remote_lfs_sha256": remote.get(p, {}).get("lfs_sha256"), "expected_sha256": exp,
                "expected_bytes": TENSOR_BYTES, "local_sha256": got,
                "local_bytes": dest.stat().st_size if dest.is_file() else None,
                "verified": got == exp and dest.stat().st_size == TENSOR_BYTES if dest.is_file() else False,
                "download": res})

    rs = [f for f in record["files"] if f["kind"] in ("steer", "scored")]
    record["summary"] = {
        "responses_listed_remote": len(steer) + len(scored),
        "responses_verified_local": sum(f["verified"] for f in rs),
        "responses_bytes_local": sum(f["local_bytes"] or 0 for f in rs),
        "tensors_verified_local": sum(f["verified"] for f in record["files"] if f["kind"].startswith("tensor")),
        "failures": [f["remote_path"] for f in record["files"] if not f["verified"]],
    }
    out = run_root / "manifests" / "recovered_artifacts.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    s = record["summary"]
    print(f"responses verified {s['responses_verified_local']}/{s['responses_listed_remote']} "
          f"({s['responses_bytes_local']} bytes); tensors verified {s['tensors_verified_local']}/2; "
          f"failures {len(s['failures'])}")
    print(f"-> {out}")
    return 0 if not s["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
