#!/usr/bin/env python3
"""Prepare the GPU box: two isolated environments from the exact frozen freezes, caches.

  .venv-gen    configs/gpu_env/generation-v1.txt (torch 2.12.0+cu126, transformers 5.14.1,
               peft 0.19.1) -- the archived generation stack of the submitted run
  .venv-judge  configs/gpu_env/judge-v1.txt (vllm 0.26.0) -- the archived judge stack

Everything lives under this run directory (HF_HOME=.cache/hf, pip cache .cache/pip); no global
environment is modified. Records interpreter, driver and package versions to
logs/box_setup.json, and refuses to continue on a Python other than 3.12 or a driver that
cannot run the judge stack's CUDA 13 libraries.

Run on the box:  python3 scripts/box_setup.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENVS = {".venv-gen": ("configs/gpu_env/generation-v1.txt", ["--extra-index-url", "https://download.pytorch.org/whl/cu126"]),
        ".venv-judge": ("configs/gpu_env/judge-v1.txt", [])}


def sh(*cmd, **kw) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def _norm(pin: str) -> str:
    """PEP 503 name normalisation: freeze prints 'Jinja2'/'huggingface_hub', the archive 'jinja2'/'huggingface-hub'."""
    import re
    name, ver = pin.strip().split("==", 1)
    return f"{re.sub(r'[-_.]+', '-', name).lower()}=={ver}"


def main() -> int:
    rec = {"python": sys.version, "platform": sh("uname", "-a").strip()}
    if sys.version_info[:2] != (3, 12):
        print(f"REFUSED: Python {sys.version.split()[0]}; the archived stacks were built on 3.12")
        return 2
    smi = sh("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")
    rec["gpu"] = smi.strip()
    cuda = sh("nvidia-smi").split("CUDA Version:")[1].split()[0]
    rec["driver_cuda"] = cuda
    if float(cuda) < 13.0:
        print(f"REFUSED: driver supports CUDA {cuda}; the judge stack needs >= 13.0")
        return 2
    if "A100" not in smi or "81920" not in smi.replace(" ", ""):
        print(f"REFUSED: expected one A100 80GB, got: {smi.strip()}")
        return 2
    env = {**os.environ, "PIP_CACHE_DIR": str(ROOT / ".cache/pip"), "HF_HOME": str(ROOT / ".cache/hf")}
    for venv, (req, extra) in ENVS.items():
        if not (ROOT / venv / "bin/python").exists():
            sh(sys.executable, "-m", "venv", str(ROOT / venv))
        # --no-deps: each file is a complete freeze of the archived environment, and the archived
        # generation stack itself pairs datasets 5.0.1 with fsspec 2026.7.0 although datasets
        # declares fsspec<=2026.6.0, so the resolver refuses the exact pins it once ran with
        sh(str(ROOT / venv / "bin/python"), "-m", "pip", "install", "-q", "--no-deps", "-r", str(ROOT / req), *extra, env=env)
        rec[venv] = sh(str(ROOT / venv / "bin/python"), "-m", "pip", "freeze").splitlines()
        chk = subprocess.run([str(ROOT / venv / "bin/python"), "-m", "pip", "check"], capture_output=True, text=True)
        rec[f"{venv}:pip_check"] = chk.stdout.strip().splitlines()
        missing = sorted(set(_norm(l) for l in (ROOT / req).read_text().splitlines() if "==" in l)
                         - set(_norm(l) for l in rec[venv] if "==" in l))
        if missing:
            print(f"REFUSED: {venv} does not match its freeze: {missing[:5]}")
            return 2
        print(f"{venv}: {len(rec[venv])} packages installed from {req}")
    # pinned downloads land in the run-local HF cache; endpoints are fetched and verified by
    # materialise_endpoints.py, the judges by vLLM at their pinned revisions on first start
    sh(str(ROOT / ".venv-gen/bin/python"), "-c",
       "import nltk; nltk.download('punkt_tab', download_dir='.cache/nltk_data', quiet=True)", cwd=ROOT, env=env)
    for d in ("logs", "outputs", ".cache/hf"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    (ROOT / "logs/box_setup.json").write_text(json.dumps(rec, indent=2) + "\n")
    print("box setup complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
