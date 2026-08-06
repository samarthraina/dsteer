"""Guards on run metadata, checkpointing, and GPU accounting.

The failure mode these protect against is losing a long run, or discovering after
the fact that a number cannot be traced to the code that produced it.

Run with:
    pytest tests/test_artifacts.py -v

CPU only; nothing here touches the network or a GPU.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from steering.artifacts import (
    GpuMonitor,
    TensorCheckpoint,
    capture_environment,
    write_run_metadata,
)


# Metadata


def test_environment_capture_has_provenance():
    env = capture_environment()
    assert env["timestamp_utc"].endswith("+00:00")
    assert env["hostname"]
    assert "torch" in env
    # git_commit is None outside a checkout, but the key must exist either way
    assert "git_commit" in env


def test_run_metadata_round_trips(tmp_path):
    cfg = {"model": {"name": "tulu3"}, "seed": 42}
    path = write_run_metadata(tmp_path, config=cfg)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["config"] == cfg
    assert "environment" in payload


# Checkpointing


def test_checkpoint_resumes_where_it_stopped(tmp_path):
    shape = (3, 10, 4)
    path = tmp_path / "x.partial.pt"

    first = TensorCheckpoint(path, shape, every=2)
    for i in range(6):
        first.tensor[:, i, :] = i + 1
        first.advance()

    second = TensorCheckpoint(path, shape, every=2)
    assert second.done == 6
    assert (second.tensor[:, 5, :] == 6).all()
    # Untouched slots must still be zero, not stale data.
    assert (second.tensor[:, 6, :] == 0).all()


def test_checkpoint_rejects_a_shape_change(tmp_path):
    """A different config must not silently resume onto an incompatible tensor."""
    path = tmp_path / "x.partial.pt"
    a = TensorCheckpoint(path, (3, 10, 4), every=2)
    a.tensor[:, 0, :] = 1.0
    a.advance()
    a.flush()

    b = TensorCheckpoint(path, (3, 99, 4), every=2)
    assert b.done == 0
    assert b.tensor.shape == (3, 99, 4)


def test_checkpoint_flush_is_atomic(tmp_path):
    """No .tmp file should survive a flush, or a crash could leave a half-written file."""
    path = tmp_path / "x.partial.pt"
    c = TensorCheckpoint(path, (2, 4, 3), every=1)
    c.tensor[:, 0, :] = 5.0
    c.advance()
    assert path.exists()
    assert not path.with_suffix(".tmp").exists()


def test_finalise_removes_the_checkpoint(tmp_path):
    path = tmp_path / "x.partial.pt"
    c = TensorCheckpoint(path, (2, 4, 3), every=1)
    c.advance()
    assert path.exists()
    c.finalise()
    assert not path.exists()


# GPU accounting


def test_gpu_monitor_writes_usage_even_without_a_gpu(tmp_path):
    """Monitoring must never be the thing that breaks a run."""
    with GpuMonitor(tmp_path, interval=0.05) as gpu:
        pass
    out = tmp_path / "gpu_usage.json"
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["elapsed_seconds"] >= 0
    assert "n_samples" in payload


def test_gpu_monitor_estimates_cost():
    mon = GpuMonitor(hourly_rate=3600.0)  # $1 per second, to keep the arithmetic obvious
    with mon:
        pass
    summary = mon.summary()
    assert "estimated_cost_usd" in summary
    assert summary["estimated_cost_usd"] >= 0
