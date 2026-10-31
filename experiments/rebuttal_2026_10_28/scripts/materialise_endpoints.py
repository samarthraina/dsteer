#!/usr/bin/env python3
"""Materialise and verify the four endpoints on the GPU box. Writes manifests/endpoints.json.

Direct endpoints (A_it, A_dpo, B_it): download exactly the recorded files at the pinned
revision and verify every byte against the SHA-256 recorded in the original run metadata.

Merged endpoint (B_dpo): verify the adapter inputs byte-for-byte, then merge with the
ORIGINAL procedure -- vendored prepare_endpoints._resolve_merged_endpoint from frozen commit
84b50df (torch.manual_seed(20260815) before the append-only vocabulary resize, bfloat16,
5GB shards, merge once, save, reload, validate) -- and compare every written file's SHA-256
with the merged files recorded for the submitted run.

An endpoint is 'verified' only when every recorded file matches. Anything else is recorded
with per-file detail and blocks that pair: identity checks are never relaxed to continue.

Run on the box (generation environment):
  python scripts/materialise_endpoints.py --stage load_parity_profile --estimate-hours 0.6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_recovered as fr  # noqa: E402

CACHE = ROOT / ".cache" / "endpoints"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def fetch(repo: str, rtype: str, rev: str, sub: str, files: list[dict], dest: Path) -> list[dict]:
    rows = []
    for f in files:
        rel = f"{sub.rstrip('/')}/{f['path']}" if sub else f["path"]
        base = f"https://{fr.HOST}/{'datasets/' if rtype == 'dataset' else ''}{repo}/resolve/{rev}/"
        url = base + urllib.parse.quote(rel)
        target = dest / f["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        error = None
        if not (target.is_file() and sha256_file(target) == f["sha256"]):
            part = target.with_name(target.name + ".part")
            for attempt in range(1, 4):  # bounded: at most three tries per file
                try:
                    with fr._OPENER.open(fr._req(url), timeout=300) as resp, part.open("wb") as fh:
                        shutil.copyfileobj(resp, fh, 1 << 22)
                    part.replace(target)
                    error = None
                    break
                except Exception as exc:  # noqa: BLE001 -- recorded below, never swallowed
                    part.unlink(missing_ok=True)
                    error = f"{type(exc).__name__}: {exc}"
        got = sha256_file(target) if target.is_file() and error is None else None
        rows.append({"path": f["path"], "recorded_sha256": f["sha256"], "local_sha256": got,
                     "bytes": target.stat().st_size if got else None,
                     "match": got == f["sha256"], "error": error})
    return rows


def identity(files: list[dict]) -> str:
    return hashlib.sha256("\n".join(f"{r['path']}:{r['local_sha256']}" for r in sorted(files, key=lambda r: r["path"])).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default=str(ROOT))
    ap.add_argument("--stage", required=True)
    ap.add_argument("--estimate-hours", type=float, required=True)
    ap.add_argument("--only", default=None, help="comma list of endpoint keys")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rr = Path(a.run_root).resolve()
    src = json.loads((rr / "manifests/endpoint_sources.json").read_text())
    keys = a.only.split(",") if a.only else ["A_it", "A_dpo", "B_it", "B_dpo"]
    out_p = rr / "manifests/endpoints.json"
    out = json.loads(out_p.read_text()) if out_p.exists() else {"endpoints": {}}
    for k in keys:
        s = src[k]
        print(f"plan: {k} {s['status']} {s['repository']}@{s['revision'][:12]} {s['subpath'] or ''}")
    if a.dry_run:
        return 0
    from budget_guard import guard_for
    guard_for(rr).require_schedulable(f"{a.stage}:materialise", a.estimate_hours)

    for k in [x for x in keys if src[x]["status"] == "direct"] + [x for x in keys if src[x]["status"] == "merged"]:
        s = src[k]
        if s["status"] == "direct":
            d = CACHE / k
            files = fetch(s["repository"], s["repository_type"], s["revision"], s["subpath"] or "", s["verified_files"], d)
            ok = len(files) == len(s["verified_files"]) and all(f["match"] for f in files)
        else:
            base = out["endpoints"].get("B_it")
            if not base or base.get("status") != "verified":
                out["endpoints"][k] = {"status": "blocked_base_unverified"}
                continue
            ad = CACHE / "B_dpo_adapter"
            ain = fetch(s["repository"], s["repository_type"], s["revision"], s["subpath"] or "",
                        s["merge"]["adapter_input_files"], ad)
            if not all(f["match"] for f in ain):
                out["endpoints"][k] = {"status": "adapter_hash_mismatch", "adapter_inputs": ain}
                continue
            sys.path.insert(0, str(ROOT / "vendor/primary_snapshot"))
            sys.path.insert(0, str(ROOT / "vendor/primary_snapshot/scripts"))
            import prepare_endpoints as pe  # the frozen-commit merge, unchanged
            staging = CACHE / "staging"
            shutil.rmtree(staging / "M+-B", ignore_errors=True)
            staging.mkdir(parents=True, exist_ok=True)
            res = pe._resolve_merged_endpoint(
                {"role": "M+-B", "source_artifact_id": "pair_b_dpo", "base_artifact_id": "pair_b_sft"},
                {"pair_b_sft": Path(base["local_path"]), "pair_b_dpo": ad}, staging,
                s["merge"]["dtype"], s["merge"]["max_shard_size"], "cuda", {})
            d = staging / "M+-B"
            rec = {f["path"]: f["sha256"] for f in s["verified_files"]}
            files = [{"path": p, "recorded_sha256": rec[p], "local_sha256": sha256_file(d / p),
                      "bytes": (d / p).stat().st_size, "match": sha256_file(d / p) == rec[p]}
                     for p in rec if (d / p).is_file()]
            ok = len(files) == len(rec) and all(f["match"] for f in files)
            s = {**s, "merge_result_tokenizer_resolution": res.get("merge", {}).get("tokenizer_resolution")
                 if isinstance(res, dict) else None}
        out["endpoints"][k] = {
            "status": "verified" if ok else "hash_mismatch",
            "repository": s["repository"], "revision": s["revision"], "subpath": s["subpath"],
            "local_path": str(d), "identity_sha256": identity(files), "files": files,
            "verification": "local byte-level SHA-256 against the submitted run's recorded hashes"}
        print(f"{k}: {out['endpoints'][k]['status']} ({sum(f['match'] for f in files)}/{len(files)} files)")
        out_p.write_text(json.dumps(out, indent=2) + "\n")
    out_p.write_text(json.dumps(out, indent=2) + "\n")
    return 0 if all(out["endpoints"].get(k, {}).get("status") == "verified" for k in keys) else 5


if __name__ == "__main__":
    raise SystemExit(main())
