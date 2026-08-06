"""Run metadata, checkpointing, and pushing results off the machine.

Rented instances go away. Anything that only exists in /workspace is one expired
rental away from being lost, and re-running a sweep costs more than the storage does.
So: every run writes a metadata file saying how it was produced, long extractions
checkpoint as they go, and finished directories get pushed to a dataset repo.

Layout on the hub, so runs stay findable a month later:

    runs/{experiment}/{model}/{run_id}/

`run_id` defaults to a UTC timestamp. Nothing here overwrites: a re-run gets its own
directory, and `run_meta.json` records which commit produced it.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

log = logging.getLogger(__name__)

DEFAULT_REPO = "samarthraina/dsteer-results"

# Checkpoints and raw activations are large and regenerable; the hub copy is for
# results others need to read. Pass include_weights=True to override.
LARGE_PATTERNS = ("*.partial.pt", "*.tmp", "__pycache__/*")


def git_commit(repo_root: Optional[Union[str, Path]] = None) -> Optional[str]:
    """Current commit, with a -dirty suffix if the tree has uncommitted changes."""
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=root, stderr=subprocess.DEVNULL
        )
        return f"{sha}-dirty" if dirty else sha
    except Exception:  # noqa: BLE001 -- not a git checkout, or no git
        return None


def capture_environment() -> Dict[str, Any]:
    """Everything needed to say what produced a number, without reading the logs."""
    env: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
    }

    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["gpu_count"] = torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        pass

    for name in ("transformers", "datasets", "peft", "trl", "accelerate", "vllm"):
        try:
            env[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            pass

    return env


def write_run_metadata(
    output_dir: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write run_meta.json next to the results. Call once at the start of a run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"environment": capture_environment()}
    if config is not None:
        payload["config"] = config
    if extra:
        payload.update(extra)

    path = output_dir / "run_meta.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info(f"Wrote {path}")
    return path


def sync_to_hub(
    local_dir: Union[str, Path],
    experiment: str,
    model: str,
    repo_id: str = DEFAULT_REPO,
    run_id: Optional[str] = None,
    include_weights: bool = False,
    token: Optional[str] = None,
) -> str:
    """Upload a run directory to the dataset repo. Returns the path in the repo.

    Idempotent: re-uploading the same run_id replaces changed files and leaves the
    rest, so this is safe to call repeatedly while a sweep is still writing.
    """
    from huggingface_hub import HfApi

    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(local_dir)

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path_in_repo = f"runs/{experiment}/{model}/{run_id}"

    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    ignore = list(LARGE_PATTERNS)
    if not include_weights:
        # activations.pt is ~2 GB per run and is regenerable from the config.
        ignore.append("activations.pt")

    log.info(f"Uploading {local_dir} -> {repo_id}:{path_in_repo}")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
        ignore_patterns=ignore,
        commit_message=f"{experiment}/{model} @ {git_commit() or 'unknown'}",
    )
    log.info(f"Uploaded to https://huggingface.co/datasets/{repo_id}/tree/main/{path_in_repo}")
    return path_in_repo


class GpuMonitor:
    """Sample GPU utilisation for the duration of a run.

    Two things this answers that a wall-clock timer does not. Whether the GPU was
    actually busy -- a sweep that leaves it at 15% is bounded by something else, and
    on rented hardware that is money going nowhere. And how much headroom is left,
    which is what sets the batch size for the next run.

    Use:
        with GpuMonitor(output_dir, hourly_rate=1.338) as gpu:
            ...
        # writes gpu_usage.json; gpu.summary() also returns the numbers
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        interval: float = 15.0,
        hourly_rate: Optional[float] = None,
    ):
        self.output_dir = Path(output_dir) if output_dir else None
        self.interval = interval
        self.hourly_rate = hourly_rate
        self.samples: list = []
        self._stop = None
        self._thread = None
        self._start = None

    def _sample(self) -> Optional[Dict[str, float]]:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, timeout=10,
            ).decode().strip().splitlines()[0]
            util, used, total, temp = (float(x) for x in out.split(","))
            return {"util_pct": util, "mem_used_mb": used, "mem_total_mb": total, "temp_c": temp}
        except Exception:  # noqa: BLE001 -- monitoring must never break the run
            return None

    def __enter__(self) -> "GpuMonitor":
        import threading

        self._start = time.monotonic()
        self._stop = threading.Event()

        def loop():
            while not self._stop.wait(self.interval):
                s = self._sample()
                if s:
                    self.samples.append(s)

        # daemon: a hung sampler must not keep the process alive
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._stop:
            self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 5)
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "gpu_usage.json").write_text(
                json.dumps(self.summary(), indent=2), encoding="utf-8"
            )

    def summary(self) -> Dict[str, Any]:
        elapsed = time.monotonic() - self._start if self._start else 0.0
        out: Dict[str, Any] = {
            "elapsed_seconds": round(elapsed, 1),
            "elapsed_hours": round(elapsed / 3600, 3),
            "n_samples": len(self.samples),
        }
        if self.hourly_rate:
            out["estimated_cost_usd"] = round(elapsed / 3600 * self.hourly_rate, 3)

        if self.samples:
            util = [s["util_pct"] for s in self.samples]
            mem = [s["mem_used_mb"] for s in self.samples]
            out.update({
                "gpu_util_mean_pct": round(sum(util) / len(util), 1),
                "gpu_util_max_pct": max(util),
                # Time spent below 20% is the part of the bill doing nothing useful.
                "gpu_idle_fraction": round(sum(1 for u in util if u < 20) / len(util), 3),
                "mem_used_peak_mb": max(mem),
                "mem_total_mb": self.samples[0]["mem_total_mb"],
                "mem_peak_fraction": round(max(mem) / self.samples[0]["mem_total_mb"], 3),
                "temp_max_c": max(s["temp_c"] for s in self.samples),
            })

        try:
            import torch

            if torch.cuda.is_available():
                out["torch_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
                out["torch_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1e6, 1)
        except Exception:  # noqa: BLE001
            pass

        return out


class TensorCheckpoint:
    """Incremental save/resume for a long extraction loop.

    Extracting activations for 2000 prompts across two models takes long enough that
    losing it to a dropped connection or an expired rental is a real cost. This holds
    a growing tensor and flushes it every `every` items, so a restart picks up from
    the last flush instead of from zero.

    Use:
        ckpt = TensorCheckpoint(path, shape=(n_layers, n_prompts, hidden), every=200)
        for i in range(ckpt.done, n_prompts):
            ckpt.tensor[:, i, :] = extract(i)
            ckpt.advance()
        ckpt.finalise()
    """

    def __init__(self, path: Union[str, Path], shape, every: int = 200, dtype=None):
        import torch

        self.path = Path(path)
        self.every = every
        self.done = 0
        self.tensor = None

        if self.path.exists():
            blob = torch.load(self.path, map_location="cpu")
            if tuple(blob["shape"]) == tuple(shape):
                self.tensor = blob["tensor"]
                self.done = int(blob["done"])
                log.info(f"Resuming from {self.path}: {self.done}/{shape[1]} done")
            else:
                # A shape change means a different config; the old partial is useless.
                log.warning(
                    f"{self.path} has shape {blob['shape']}, expected {tuple(shape)} -- ignoring"
                )

        if self.tensor is None:
            self.tensor = torch.zeros(*shape, dtype=dtype or torch.float32)

    def advance(self) -> None:
        self.done += 1
        if self.done % self.every == 0:
            self.flush()

    def flush(self) -> None:
        import torch

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        torch.save({"tensor": self.tensor, "done": self.done, "shape": tuple(self.tensor.shape)}, tmp)
        tmp.replace(self.path)  # atomic, so a crash mid-write cannot corrupt the checkpoint

    def finalise(self) -> None:
        """Drop the checkpoint once the caller has saved the finished tensor."""
        if self.path.exists():
            self.path.unlink()
