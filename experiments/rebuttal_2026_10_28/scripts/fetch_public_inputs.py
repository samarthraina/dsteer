#!/usr/bin/env python3
"""Fetch the public inputs at immutable refs and record what was fetched.

Every file goes to inputs/public/<source>/, one source per directory, and is treated as
data. The two code bundles fetched here are handled differently:
  - the reference refusal-direction code is READ for the method audit and never executed;
  - the IFEval official checkers are executed for scoring (the handoff requires the
    official checks), so they are copied into vendor/ifeval_official/ only after review,
    with hashes, and imported from there.

Run:  python3 -I scripts/fetch_public_inputs.py
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUB = ROOT / "inputs" / "public"

GH = "https://raw.githubusercontent.com"
HF = "https://huggingface.co/datasets"
SOURCES = {
    "harmfulqa": {
        "ref": "6f1a78aed47d16c0695e4595d0159abc38197bfd",
        "files": {"data_for_hub.json": f"{HF}/declare-lab/HarmfulQA/resolve/{{ref}}/data_for_hub.json"},
        "use": "prompt text for the frozen HarmfulQA partitions; verified against manifest prompt_hash",
    },
    "ifeval": {
        "ref": "966cd89545d6b6acfd7638bc708b98261ca58e84",
        "files": {"ifeval_input_data.jsonl": f"{HF}/google/IFEval/resolve/{{ref}}/ifeval_input_data.jsonl"},
        "use": "instruction-following panel (Experiment B)",
    },
    "ifeval_checkers": {
        "ref": "e6890f85757dd84e27ca6df2dd30651dafad28e0",
        "files": {f: f"{GH}/google-research/google-research/{{ref}}/instruction_following_eval/{f}"
                  for f in ("instructions.py", "instructions_registry.py", "instructions_util.py",
                            "evaluation_lib.py")},
        "use": "official strict/loose prompt-level checkers (executed, after review)",
    },
    "xstest": {
        "ref": "475f10bf0a3d6a9dfb174b6de1a38afbfdff98a5",
        "files": {"xstest_prompts.csv": f"{GH}/paul-rottger/xstest/{{ref}}/xstest_prompts.csv"},
        "use": "250 safe prompts (Experiment B); unsafe contrast rows excluded from the denominator",
    },
    "refusal_direction_reference": {
        "ref": "0b9c8bc56a73b42f595fd173156d4aefe9a9143c",
        "files": {f.replace("/", "__"): f"{GH}/andyrdt/refusal_direction/{{ref}}/{f}"
                  for f in ("dataset/splits/harmless_train.json", "dataset/splits/harmless_val.json",
                            "dataset/splits/harmless_test.json",
                            "pipeline/submodules/generate_directions.py",
                            "pipeline/submodules/select_direction.py",
                            "pipeline/utils/hook_utils.py", "pipeline/config.py",
                            "pipeline/run_pipeline.py",
                            "pipeline/model_utils/model_base.py",
                            "pipeline/model_utils/llama3_model.py",
                            "pipeline/model_utils/qwen_model.py")},
        "use": "harmless construction prompts (Alpaca-derived, as in the reference); reference code READ for the audit, never executed",
    },
}


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    record = {"fetched_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "sources": {}}
    fails = 0
    for name, spec in SOURCES.items():
        d = PUB / name
        d.mkdir(parents=True, exist_ok=True)
        files = {}
        for fname, url_t in spec["files"].items():
            url = url_t.format(ref=spec["ref"])
            dest = d / fname
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "dsteer-rebuttal-fetch/1"})
                data = urllib.request.urlopen(req, timeout=120).read()
                dest.write_bytes(data)
                files[fname] = {"url": url, "bytes": len(data), "sha256": sha256(data)}
            except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
                files[fname] = {"url": url, "error": f"{type(exc).__name__}: {exc}"}
                fails += 1
        record["sources"][name] = {"ref": spec["ref"], "use": spec["use"], "files": files}
        ok = sum("sha256" in f for f in files.values())
        print(f"{name:30s} @ {spec['ref'][:12]}  {ok}/{len(files)} files")
    (ROOT / "manifests" / "public_inputs.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"failures: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
