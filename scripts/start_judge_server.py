"""Launch the vLLM server hosting the judge model.

Run this from the judge's own environment, not the one the eval scripts use:

    /workspace/venv_judge/bin/python scripts/start_judge_server.py

vLLM pins its own torch -- installing it beside the generation stack replaced
torch 2.12.0+cu126 with 2.11.0+cu130, which is a silent way to invalidate a run
that is already in flight. The two talk over HTTP, so they have no reason to share
an interpreter.

Keep this process running for the duration of scoring; the eval scripts connect to
it as clients.

Prefix caching is on by default and matters more here than in most serving setups.
Every call for a given metric begins with the same rubric, so without it that
prefix is prefilled once per request -- tens of thousands of times over a sweep,
in a workload that is prefill-bound to begin with.

Quantisation is deliberately not offered. Scores are compared against earlier runs
made in bf16, and a quantised judge shifts borderline cases, which is precisely
where a banded rubric changes its answer.

`--frozen` launches exactly the confirmatory Qwen3.5 judge protocol (protocol Section
10 / `steering.judge_identity.FROZEN_JUDGE_IDENTITY`): the pinned repository/revision
served under its revision-bearing alias, bf16, no quantization, an 8192-token context,
language-model-only mode, and prefix caching -- after verifying the installed vLLM is
exactly the frozen version. It never falls back to launching an unverified version.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steering.judge_identity import FROZEN_JUDGE_IDENTITY, served_model_alias

FROZEN_VLLM_VERSION = FROZEN_JUDGE_IDENTITY["judge"]["vllm_version"]


class VllmVersionError(RuntimeError):
    """The installed vLLM does not match the frozen protocol version."""


def verify_vllm_version(expected: str = FROZEN_VLLM_VERSION, version_getter: Optional[Callable[[str], str]] = None) -> str:
    """Confirm the installed `vllm` package matches `expected` before a frozen-server
    command is ever built. Raises `VllmVersionError` on any mismatch or missing
    package -- this never launches on an unverified version.

    `version_getter` defaults to `importlib.metadata.version`, injectable so tests can
    check both outcomes without vLLM installed in the CPU test environment.
    """
    if version_getter is None:
        import importlib.metadata
        version_getter = importlib.metadata.version
    try:
        installed = version_getter("vllm")
    except Exception as e:  # noqa: BLE001 -- package not installed, or lookup failed
        raise VllmVersionError(f"could not determine installed vllm version: {type(e).__name__}: {e}") from e
    if installed != expected:
        raise VllmVersionError(f"frozen judge protocol requires vllm=={expected}, found vllm=={installed}")
    return installed


def build_command(
    model: str, port: int, gpu_util: float, dtype: str, api_key: str,
    max_model_len: int, prefix_caching: bool,
    revision: Optional[str] = None, served_model_name: Optional[str] = None,
    runner: Optional[str] = None, language_model_only: bool = False,
    quantization: Optional[str] = None,
) -> List[str]:
    """Pure vLLM server-launch command construction -- no subprocess is started here,
    so this is exercised directly by tests without launching vLLM. Never adds
    speculative-decoding or MTP flags.

    `runner` is vLLM 0.26's `--runner` flag (its predecessor `--task` is not sent);
    `language_model_only` adds `--language-model-only`, the text-only language-model
    mode flag -- distinct from `runner`, which selects generate/embed/etc.
    """
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--dtype", dtype,
        "--gpu-memory-utilization", str(gpu_util),
        "--max-model-len", str(max_model_len),
        "--port", str(port),
        "--api-key", api_key,
    ]
    if revision:
        cmd += ["--revision", revision]
    if served_model_name:
        cmd += ["--served-model-name", served_model_name]
    if runner:
        cmd += ["--runner", runner]
    if language_model_only:
        cmd.append("--language-model-only")
    if quantization:
        cmd += ["--quantization", quantization]
    if prefix_caching:
        cmd.append("--enable-prefix-caching")
    return cmd


def frozen_command(
    port: int, gpu_util: float, api_key: str, version_getter: Optional[Callable[[str], str]] = None,
) -> List[str]:
    """The exact frozen Qwen3.5 judge-server command (protocol Section 10). Verifies
    `vllm==0.26.0` first and raises rather than returning a command for the wrong
    version -- there is no code path from a version mismatch to a launched process."""
    verify_vllm_version(version_getter=version_getter)
    f = FROZEN_JUDGE_IDENTITY
    repository, revision = f["judge"]["repository"], f["judge"]["revision"]
    return build_command(
        model=repository, port=port, gpu_util=gpu_util, dtype=f["judge"]["dtype"], api_key=api_key,
        max_model_len=f["max_model_len"], prefix_caching=f["prefix_caching"],
        revision=revision, served_model_name=served_model_alias(repository, revision),
        runner="generate", language_model_only=True,  # text-only language-model mode
        quantization=f["judge"]["quantization"],
    )


def main():
    parser = argparse.ArgumentParser(description="Launch vLLM server for the judge model.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-util", type=float, default=0.90)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--max-model-len", type=int, default=4096,
        help="Judge inputs are a rubric plus one prompt and response; a smaller window "
             "leaves more memory for the KV cache and so a larger effective batch.",
    )
    parser.add_argument(
        "--no-prefix-caching", action="store_true",
        help="Disable prefix caching. Only useful for measuring what it is worth.",
    )
    parser.add_argument(
        "--frozen", action="store_true",
        help="Launch exactly the frozen Qwen3.5 confirmatory judge protocol "
             "(steering.judge_identity.FROZEN_JUDGE_IDENTITY), ignoring --model/--dtype/"
             "--max-model-len/--no-prefix-caching. Verifies vllm==0.26.0 before launch "
             "and refuses to launch an unverified version.",
    )
    args = parser.parse_args()

    if args.frozen:
        cmd = frozen_command(port=args.port, gpu_util=args.gpu_util, api_key=args.api_key)
    else:
        cmd = build_command(
            model=args.model, port=args.port, gpu_util=args.gpu_util, dtype=args.dtype,
            api_key=args.api_key, max_model_len=args.max_model_len,
            prefix_caching=not args.no_prefix_caching,
        )

    print(f"Launching vLLM with: {' '.join(cmd)}")
    print(f"Server will accept requests at http://localhost:{args.port}/v1")
    print("Keep this terminal open. Press Ctrl+C to stop.")
    print("-" * 60)

    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in iter(proc.stdout.readline, ""):
            print(line, end="")
    except KeyboardInterrupt:
        print("\nStopping vLLM server...")
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
