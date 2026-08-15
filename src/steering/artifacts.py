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

import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

log = logging.getLogger(__name__)

DEFAULT_REPO = "samarthraina/dsteer-results"

# Checkpoints and raw activations are large and regenerable; the hub copy is for
# results others need to read. Pass include_weights=True to override.
LARGE_PATTERNS = ("*.partial.pt", "*.tmp", "__pycache__/*")


class RunMetadataError(RuntimeError):
    """A run_meta.json write-once/resume invariant was violated."""


# Code state
#
# A short commit label is not enough to say what code actually ran: it misses
# uncommitted changes entirely, including untracked files that were never staged. This
# captures the working tree's exact identity -- and, when it isn't clean, a hash that
# changes with the actual dirty content, not just a boolean.

_UNAVAILABLE_CODE_STATE: Dict[str, Any] = {
    "available": False,
    "commit": None,
    "branch": None,
    "dirty": None,
    "dirty_state_hash": None,
    "dirty_paths": [],
}


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_status_entries(root: Path) -> List[Dict[str, Optional[str]]]:
    """Parse `git status --porcelain=v2 -z --untracked-files=all`.

    -z gives NUL-delimited records so spaces/Unicode in paths are never ambiguous with
    the record separator. Every record type is one NUL-terminated field except renames/
    copies ("2"), which carry a second field (the origin path) -- so the record type's
    leading marker must be inspected to know how many NUL-delimited tokens to consume;
    naively splitting on NUL and treating every token as one record would misalign a
    rename's origin path onto the next record. `--untracked-files=all` lists every
    untracked file individually rather than collapsing a new directory into one entry,
    so each file's content can be hashed on its own. Ignored files are excluded by
    Git's own default (no `--ignored` flag), so this reports exactly what Git considers
    dirty -- never a wider scan of the filesystem.
    """
    raw = subprocess.check_output(
        ["git", "status", "--porcelain=v2", "-z", "--untracked-files=all"],
        cwd=root, stderr=subprocess.DEVNULL,
    ).decode("utf-8")
    tokens = raw.split("\x00")
    if tokens and tokens[-1] == "":
        tokens.pop()

    entries: List[Dict[str, Optional[str]]] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        marker = token[:1]
        if marker == "1":
            # "1 XY sub mH mI mW hH hI path"
            parts = token.split(" ", 8)
            identity = ":".join(parts[2:8])  # sub mH mI mW hH hI
            entries.append({"status": parts[1], "path": parts[8], "orig_path": None, "index_identity": identity})
            i += 1
        elif marker == "2":
            # "2 XY sub mH mI mW hH hI Xscore path" + NUL + origPath
            parts = token.split(" ", 9)
            orig_path = tokens[i + 1] if i + 1 < len(tokens) else None
            identity = ":".join(parts[2:9])  # sub mH mI mW hH hI Xscore
            entries.append({"status": parts[1], "path": parts[9], "orig_path": orig_path, "index_identity": identity})
            i += 2
        elif marker == "u":
            # "u XY sub m1 m2 m3 mW h1 h2 h3 path" -- m1/h1=ancestor, m2/h2=ours, m3/h3=theirs
            parts = token.split(" ", 10)
            identity = ":".join(parts[2:10])  # sub m1 m2 m3 mW h1 h2 h3
            entries.append({"status": parts[1], "path": parts[10], "orig_path": None, "index_identity": identity})
            i += 1
        elif marker == "?":
            entries.append({"status": "??", "path": token[2:], "orig_path": None, "index_identity": ""})
            i += 1
        elif marker == "!":
            entries.append({"status": "!!", "path": token[2:], "orig_path": None, "index_identity": ""})
            i += 1
        else:
            i += 1  # blank/unexpected token; skip defensively rather than crash

    return entries


def _stream_sha256_update(hasher, path: Path, chunk_size: int = 1 << 20) -> None:
    """Feed `path`'s content into `hasher` in fixed-size chunks -- an untracked file
    can be model-sized, so it is never read whole into memory."""
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)


def _dirty_state_hash(root: Path, entries: Sequence[Mapping[str, Optional[str]]]) -> str:
    """SHA-256 over every dirty entry's status, path, origin path, HEAD/index object
    identity, and current on-disk content -- sorted by path first, so the result does
    not depend on the order Git happened to report entries in.

    The HEAD/index identity (`index_identity`: submodule state, file modes, and blob
    object hashes at HEAD/index/worktree from the porcelain-v2 record -- stage
    ancestor/ours/theirs for a conflict) is what makes this sound rather than merely
    convenient. Two states can share the same status code and, after a later edit,
    coincidentally identical working-tree bytes while having staged completely
    different content in between -- reading only the current on-disk bytes cannot see
    that, but the reported index blob hash always identifies exactly what is staged.
    On-disk content is still hashed too (chunked, so an untracked or large working-tree
    file is never read whole into memory), since Git does not compute a worktree blob
    hash without reading the file itself.
    """
    hasher = hashlib.sha256()
    ordered = sorted(entries, key=lambda e: (e["path"] or "", e["orig_path"] or ""))
    for entry in ordered:
        hasher.update((entry["status"] or "").encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update((entry["path"] or "").encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update((entry["orig_path"] or "").encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update((entry.get("index_identity") or "").encode("utf-8"))
        hasher.update(b"\x00")

        full = root / entry["path"]
        if full.is_file():
            _stream_sha256_update(hasher, full)
        else:
            hasher.update(b"<absent>")  # deleted, or a directory-only rename artifact
        hasher.update(b"\x00")

    return hasher.hexdigest()


def capture_code_state(repo_root: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Structured Git identity for `repo_root` (default: this repository's root).

    Returns an explicit unavailable state -- never a misleading "clean" one -- when
    `repo_root` is not inside a Git work tree or the `git` binary itself cannot run.
    """
    root = Path(repo_root) if repo_root is not None else _default_repo_root()
    try:
        subprocess.check_output(
            ["git", "rev-parse", "--is-inside-work-tree"], cwd=root, stderr=subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001 -- not a git checkout, git missing, or root absent
        return dict(_UNAVAILABLE_CODE_STATE)

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        if branch == "HEAD":
            branch = f"detached@{commit}"

        entries = _git_status_entries(root)
        dirty = bool(entries)
        dirty_paths = [
            {"status": e["status"], "path": e["path"], "orig_path": e["orig_path"]}
            for e in sorted(entries, key=lambda e: (e["path"] or "", e["orig_path"] or ""))
        ]
        dirty_hash = _dirty_state_hash(root, entries) if dirty else None

        return {
            "available": True,
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
            "dirty_state_hash": dirty_hash,
            "dirty_paths": dirty_paths,
        }
    except Exception:  # noqa: BLE001 -- a git command failed unexpectedly after the repo check passed
        return dict(_UNAVAILABLE_CODE_STATE)


def _short_commit_label(repo_root: Optional[Union[str, Path]] = None) -> str:
    """A short human-readable label for log/commit messages -- not the scientific
    identity, which lives in `capture_code_state`."""
    state = capture_code_state(repo_root)
    if not state["available"]:
        return "unknown"
    label = state["commit"][:7]
    return f"{label}-dirty" if state["dirty"] else label


def capture_environment(repo_root: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Everything needed to say what produced a number, without reading the logs."""
    env: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_state": capture_code_state(repo_root),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
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


# Run identity


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    """Deterministic UTF-8 JSON: sorted keys, compact stable separators. Two payloads
    that differ only in dict insertion order serialize identically."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode("utf-8")


def _compute_run_identity_hash(payload: Mapping[str, Any]) -> str:
    """Canonical SHA-256 over the scientifically relevant subset of `payload`.

    Excludes `run_identity_hash` itself (it cannot depend on its own value) and the two
    fields that are expected to change between an otherwise-identical run and its later
    resume: `environment.timestamp_utc` (the creation instant) and `environment.hostname`
    (moving the same run directory to another machine must not by itself change the
    identity of the run that produced it).
    """
    reduced = {k: v for k, v in payload.items() if k != "run_identity_hash"}
    env = reduced.get("environment")
    if isinstance(env, dict):
        reduced = dict(reduced)
        reduced["environment"] = {k: v for k, v in env.items() if k not in ("timestamp_utc", "hostname")}
    return hashlib.sha256(_canonical_json(reduced)).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write `payload` to `path` atomically: the full content is built in a sibling
    temp file first, then renamed into place, so a crash mid-write can never leave a
    partially written `run_meta.json` -- the rename either fully lands or doesn't
    happen at all. On any failure, only the exact temp file this call created is
    removed; nothing else in the directory is touched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def write_run_metadata(
    output_dir: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    argv: Optional[Sequence[str]] = None,
    repo_root: Optional[Union[str, Path]] = None,
) -> Path:
    """Write `run_meta.json` once, atomically, in `output_dir`.

    A second call is a resume check, not a rewrite: if the current run's identity
    (code state, `argv`, `config`, `extra`, and the software/CUDA/GPU environment --
    excluding only the creation timestamp and hostname) matches what's already on disk,
    the file is left untouched and its path is returned. If it differs, this raises
    `RunMetadataError` and leaves the existing file byte-identical. A legacy file with
    no valid `run_identity_hash`, malformed JSON, or a stored hash that does not match
    its own content is rejected the same way -- never silently upgraded or overwritten.

    `argv`, when given, is stored as a JSON list (never shell-joined), so it round-trips
    exactly regardless of quoting.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_meta.json"

    payload: Dict[str, Any] = {"environment": capture_environment(repo_root)}
    if argv is not None:
        payload["argv"] = list(argv)
    if config is not None:
        payload["config"] = config
    if extra:
        payload.update(extra)
    payload["run_identity_hash"] = _compute_run_identity_hash(payload)

    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise RunMetadataError(f"{path} exists but is not valid JSON: {e}") from e

        stored_hash = existing.get("run_identity_hash") if isinstance(existing, dict) else None
        if not isinstance(stored_hash, str) or not stored_hash:
            raise RunMetadataError(
                f"{path} exists but has no valid 'run_identity_hash' (legacy metadata file); "
                "refusing to resume onto it or overwrite it"
            )
        recomputed_existing = _compute_run_identity_hash(existing)
        if recomputed_existing != stored_hash:
            raise RunMetadataError(
                f"{path} exists but its stored run_identity_hash does not match its own "
                f"content (stored {stored_hash!r}, recomputed {recomputed_existing!r}); "
                "refusing to resume onto an internally inconsistent metadata file"
            )

        new_hash = payload["run_identity_hash"]
        if new_hash == stored_hash:
            log.info(f"{path} already matches this run's identity; not rewriting")
            return path
        raise RunMetadataError(
            f"{path} already exists with a different run identity "
            f"(existing {stored_hash!r}, this run {new_hash!r}); refusing to overwrite. "
            "Use a fresh output directory for a different configuration/code state."
        )

    _atomic_write_json(path, payload)
    log.info(f"Wrote {path} ({_short_commit_label(repo_root)})")
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
        commit_message=f"{experiment}/{model} @ {_short_commit_label()}",
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
