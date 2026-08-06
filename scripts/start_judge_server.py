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
"""

from __future__ import annotations

import argparse
import subprocess
import sys


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
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--dtype", args.dtype,
        "--gpu-memory-utilization", str(args.gpu_util),
        "--max-model-len", str(args.max_model_len),
        "--port", str(args.port),
        "--api-key", args.api_key,
    ]
    if not args.no_prefix_caching:
        cmd.append("--enable-prefix-caching")

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
