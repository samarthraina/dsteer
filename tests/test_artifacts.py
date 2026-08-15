"""Guards on run metadata, checkpointing, and GPU accounting.

The failure mode these protect against is losing a long run, or discovering after
the fact that a number cannot be traced to the code that produced it -- including a
resumed output directory silently mixing results from two different code states.

Run with:
    pytest tests/test_artifacts.py -v

CPU only, offline; nothing here touches the network or a GPU. The code-state tests use
tiny temporary Git repositories rather than this repository's own working tree.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import steering.artifacts as artifacts_module
from steering.artifacts import (
    GpuMonitor,
    RunMetadataError,
    TensorCheckpoint,
    capture_code_state,
    capture_environment,
    write_run_metadata,
)


# Temporary Git repo helper


def _git(args, cwd, check=True):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=check, capture_output=True, text=True,
    )


def _init_repo(root: Path, files=None) -> Path:
    """A tiny local Git repo, committed once, with local (not global) identity config."""
    root.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q"], root)
    _git(["config", "user.email", "test@example.com"], root)
    _git(["config", "user.name", "Test"], root)
    for rel, content in (files or {"file.txt": "hello\n"}).items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(["add", "-A"], root)
    _git(["commit", "-q", "-m", "init"], root)
    return root


# Metadata: environment capture


def test_environment_capture_has_provenance():
    env = capture_environment()
    assert env["timestamp_utc"].endswith("+00:00")
    assert env["hostname"]
    assert "torch" in env
    assert "platform" in env
    # code_state is unavailable outside a checkout, but the key must exist either way
    assert "code_state" in env
    assert "available" in env["code_state"]


def test_run_metadata_round_trips(tmp_path):
    cfg = {"model": {"name": "tulu3"}, "seed": 42}
    path = write_run_metadata(tmp_path, config=cfg, argv=["prog", "--seed", "42"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["config"] == cfg
    assert payload["argv"] == ["prog", "--seed", "42"]
    assert "environment" in payload
    assert "run_identity_hash" in payload


# Code state: capture_code_state against tiny temporary repos


def test_a_clean_repo_reports_full_commit_branch_and_no_dirty_hash(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    state = capture_code_state(repo)
    assert state["available"] is True
    assert len(state["commit"]) == 40
    assert all(c in "0123456789abcdef" for c in state["commit"])
    assert state["branch"]  # non-empty; default branch name varies by git version/config
    assert state["dirty"] is False
    assert state["dirty_state_hash"] is None
    assert state["dirty_paths"] == []


def test_a_staged_change_is_detected_as_dirty(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    (repo / "file.txt").write_text("changed\n", encoding="utf-8")
    _git(["add", "file.txt"], repo)

    state = capture_code_state(repo)
    assert state["dirty"] is True
    assert state["dirty_state_hash"] is not None
    assert any(e["path"] == "file.txt" for e in state["dirty_paths"])


def test_an_unstaged_tracked_change_is_detected_as_dirty(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    (repo / "file.txt").write_text("changed but not staged\n", encoding="utf-8")

    state = capture_code_state(repo)
    assert state["dirty"] is True
    entry = next(e for e in state["dirty_paths"] if e["path"] == "file.txt")
    assert entry["status"] == ".M"


def test_a_deletion_is_detected_as_dirty(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    (repo / "file.txt").unlink()

    state = capture_code_state(repo)
    assert state["dirty"] is True
    entry = next(e for e in state["dirty_paths"] if e["path"] == "file.txt")
    assert "D" in entry["status"]


def test_an_untracked_file_is_detected_as_dirty(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    (repo / "new_file.txt").write_text("brand new\n", encoding="utf-8")

    state = capture_code_state(repo)
    assert state["dirty"] is True
    entry = next(e for e in state["dirty_paths"] if e["path"] == "new_file.txt")
    assert entry["status"] == "??"


def test_a_rename_with_spaces_and_unicode_is_represented_without_ambiguity(tmp_path):
    repo = _init_repo(tmp_path / "repo", files={"a file.txt": "content\n"})
    _git(["mv", "a file.txt", "renamed üñíçødé.txt"], repo)

    state = capture_code_state(repo)
    assert state["dirty"] is True
    entry = next(e for e in state["dirty_paths"] if e["orig_path"] == "a file.txt")
    assert entry["path"] == "renamed üñíçødé.txt"
    assert entry["status"][0] == "R"


def test_changing_an_untracked_files_bytes_changes_the_dirty_state_hash(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    target = repo / "new_file.txt"
    target.write_text("version one\n", encoding="utf-8")
    hash_a = capture_code_state(repo)["dirty_state_hash"]

    target.write_text("version two\n", encoding="utf-8")
    hash_b = capture_code_state(repo)["dirty_state_hash"]

    assert hash_a != hash_b


def test_ignored_files_do_not_dirty_the_state(tmp_path):
    repo = _init_repo(tmp_path / "repo", files={"file.txt": "hello\n", ".gitignore": "ignored.txt\n"})
    (repo / "ignored.txt").write_text("should not count\n", encoding="utf-8")

    state = capture_code_state(repo)
    assert state["dirty"] is False
    assert state["dirty_state_hash"] is None


def test_the_same_dirty_state_produces_the_same_hash(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    (repo / "file.txt").write_text("changed\n", encoding="utf-8")

    a = capture_code_state(repo)["dirty_state_hash"]
    b = capture_code_state(repo)["dirty_state_hash"]
    assert a == b


def test_dirty_state_hash_distinguishes_staged_content_despite_identical_status_and_worktree_bytes(tmp_path):
    """Same status code (MM) and the same working-tree bytes at two different moments,
    with different staged content in between -- hashing status + worktree bytes alone
    cannot see that, since neither differs; the reported index blob identity must.

        1. HEAD contains "base".
        2. Stage "staged-version-one".
        3. Change the working tree to fixed content "working-version".
        4. Capture the hash.
        5. Stage "staged-version-two".
        6. Restore the working tree to the same fixed "working-version".
        7. Status is "MM" both times, worktree bytes are identical both times, but the
           index differs -- the dirty-state hash must differ.
    """
    repo = _init_repo(tmp_path / "repo", files={"file.txt": "base\n"})
    target = repo / "file.txt"

    target.write_text("staged-version-one\n", encoding="utf-8")
    _git(["add", "file.txt"], repo)
    target.write_text("working-version\n", encoding="utf-8")
    state_a = capture_code_state(repo)
    entry_a = next(e for e in state_a["dirty_paths"] if e["path"] == "file.txt")
    bytes_a = target.read_bytes()

    target.write_text("staged-version-two\n", encoding="utf-8")
    _git(["add", "file.txt"], repo)
    target.write_text("working-version\n", encoding="utf-8")
    state_b = capture_code_state(repo)
    entry_b = next(e for e in state_b["dirty_paths"] if e["path"] == "file.txt")
    bytes_b = target.read_bytes()

    assert entry_a["status"] == "MM"
    assert entry_b["status"] == "MM"
    assert bytes_a == bytes_b  # identical worktree bytes both times
    assert state_a["dirty_state_hash"] != state_b["dirty_state_hash"]


def test_capture_outside_git_reports_unavailable_not_clean(tmp_path):
    not_a_repo = tmp_path / "plain_dir"
    not_a_repo.mkdir()

    state = capture_code_state(not_a_repo)
    assert state["available"] is False
    assert state["dirty"] is None  # never a misleading "clean"
    assert state["commit"] is None
    assert state["dirty_state_hash"] is None


def test_no_captured_payload_contains_raw_diff_text_or_file_contents(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    marker = "UNIQUE_MARKER_MUST_NOT_APPEAR_IN_METADATA_9f3c2a"
    (repo / "file.txt").write_text(f"{marker}\n", encoding="utf-8")
    (repo / "untracked.txt").write_text(f"also {marker}\n", encoding="utf-8")

    out = tmp_path / "out"
    path = write_run_metadata(out, repo_root=repo)
    text = path.read_text(encoding="utf-8")

    assert marker not in text
    payload = json.loads(text)
    assert payload["environment"]["code_state"]["dirty_state_hash"] is not None


# Run identity: canonical hashing


def test_run_identity_hash_is_independent_of_mapping_insertion_order(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"

    cfg_a = {"model": "x", "seed": 1, "nested": {"p": 1, "q": 2}}
    cfg_b = {"nested": {"q": 2, "p": 1}, "seed": 1, "model": "x"}  # same content, different order
    extra_a = {"side": "it", "lambda": 0.5}
    extra_b = {"lambda": 0.5, "side": "it"}

    path_a = write_run_metadata(out_a, config=cfg_a, extra=extra_a, argv=["p"], repo_root=repo)
    path_b = write_run_metadata(out_b, config=cfg_b, extra=extra_b, argv=["p"], repo_root=repo)

    hash_a = json.loads(path_a.read_text())["run_identity_hash"]
    hash_b = json.loads(path_b.read_text())["run_identity_hash"]
    assert hash_a == hash_b


# Run identity: write-once and resume


def test_the_first_call_writes_a_valid_self_consistent_identity(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    out = tmp_path / "out"
    path = write_run_metadata(out, config={"a": 1}, repo_root=repo)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert artifacts_module._compute_run_identity_hash(payload) == payload["run_identity_hash"]


def test_an_identical_second_call_does_not_rewrite_or_change_mtime(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    out = tmp_path / "out"
    p1 = write_run_metadata(out, config={"a": 1}, extra={"seed": 1}, argv=["p"], repo_root=repo)
    mtime1 = p1.stat().st_mtime_ns
    text1 = p1.read_text(encoding="utf-8")

    p2 = write_run_metadata(out, config={"a": 1}, extra={"seed": 1}, argv=["p"], repo_root=repo)
    assert p2 == p1
    assert p2.stat().st_mtime_ns == mtime1
    assert p2.read_text(encoding="utf-8") == text1


def test_a_code_state_change_is_rejected_and_the_file_stays_byte_identical(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    out = tmp_path / "out"
    write_run_metadata(out, config={"a": 1}, repo_root=repo)
    before = (out / "run_meta.json").read_text(encoding="utf-8")

    (repo / "file.txt").write_text("now dirty\n", encoding="utf-8")  # changes code_state

    with pytest.raises(RunMetadataError):
        write_run_metadata(out, config={"a": 1}, repo_root=repo)
    assert (out / "run_meta.json").read_text(encoding="utf-8") == before


def test_a_cli_change_is_rejected(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    out = tmp_path / "out"
    write_run_metadata(out, config={"a": 1}, argv=["p", "--x", "1"], repo_root=repo)
    before = (out / "run_meta.json").read_text(encoding="utf-8")

    with pytest.raises(RunMetadataError):
        write_run_metadata(out, config={"a": 1}, argv=["p", "--x", "2"], repo_root=repo)
    assert (out / "run_meta.json").read_text(encoding="utf-8") == before


def test_a_config_change_is_rejected(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    out = tmp_path / "out"
    write_run_metadata(out, config={"a": 1}, repo_root=repo)
    before = (out / "run_meta.json").read_text(encoding="utf-8")

    with pytest.raises(RunMetadataError):
        write_run_metadata(out, config={"a": 2}, repo_root=repo)
    assert (out / "run_meta.json").read_text(encoding="utf-8") == before


def test_an_extra_change_is_rejected(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    out = tmp_path / "out"
    write_run_metadata(out, config={"a": 1}, extra={"seed": 1}, repo_root=repo)
    before = (out / "run_meta.json").read_text(encoding="utf-8")

    with pytest.raises(RunMetadataError):
        write_run_metadata(out, config={"a": 1}, extra={"seed": 2}, repo_root=repo)
    assert (out / "run_meta.json").read_text(encoding="utf-8") == before


def test_a_scientific_environment_change_is_rejected(tmp_path, monkeypatch):
    out = tmp_path / "out"
    calls = {"n": 0}

    def fake_env(repo_root=None):
        calls["n"] += 1
        return {
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "code_state": dict(artifacts_module._UNAVAILABLE_CODE_STATE),
            "hostname": "host-a",
            "python": "3.12.0",
            "torch": "2.0.0" if calls["n"] == 1 else "2.1.0",
        }

    monkeypatch.setattr(artifacts_module, "capture_environment", fake_env)
    write_run_metadata(out)
    before = (out / "run_meta.json").read_text(encoding="utf-8")

    with pytest.raises(RunMetadataError):
        write_run_metadata(out)
    assert (out / "run_meta.json").read_text(encoding="utf-8") == before


def test_hostname_only_difference_does_not_change_run_identity(tmp_path, monkeypatch):
    out = tmp_path / "out"
    calls = {"n": 0}

    def fake_env(repo_root=None):
        calls["n"] += 1
        return {
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "code_state": dict(artifacts_module._UNAVAILABLE_CODE_STATE),
            "hostname": "host-a" if calls["n"] == 1 else "host-b",
            "python": "3.12.0",
        }

    monkeypatch.setattr(artifacts_module, "capture_environment", fake_env)
    p1 = write_run_metadata(out)
    mtime1 = p1.stat().st_mtime_ns

    p2 = write_run_metadata(out)  # must not raise
    assert p2 == p1
    assert p2.stat().st_mtime_ns == mtime1


def test_creation_time_only_difference_does_not_change_run_identity(tmp_path, monkeypatch):
    out = tmp_path / "out"
    calls = {"n": 0}

    def fake_env(repo_root=None):
        calls["n"] += 1
        ts = "2026-01-01T00:00:00+00:00" if calls["n"] == 1 else "2026-06-15T12:34:56+00:00"
        return {
            "timestamp_utc": ts,
            "code_state": dict(artifacts_module._UNAVAILABLE_CODE_STATE),
            "hostname": "host-a",
            "python": "3.12.0",
        }

    monkeypatch.setattr(artifacts_module, "capture_environment", fake_env)
    p1 = write_run_metadata(out)
    mtime1 = p1.stat().st_mtime_ns

    p2 = write_run_metadata(out)  # must not raise
    assert p2 == p1
    assert p2.stat().st_mtime_ns == mtime1


def test_a_malformed_existing_json_file_is_rejected_without_modification(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    bad_text = "{not valid json"
    (out / "run_meta.json").write_text(bad_text, encoding="utf-8")

    with pytest.raises(RunMetadataError):
        write_run_metadata(out, config={"a": 1})
    assert (out / "run_meta.json").read_text(encoding="utf-8") == bad_text


def test_a_legacy_metadata_file_without_run_identity_hash_is_rejected(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    legacy = json.dumps({"environment": {"git_commit": "abc1234"}, "config": {"a": 1}})
    (out / "run_meta.json").write_text(legacy, encoding="utf-8")

    with pytest.raises(RunMetadataError):
        write_run_metadata(out, config={"a": 1})
    assert (out / "run_meta.json").read_text(encoding="utf-8") == legacy


def test_an_internally_inconsistent_existing_metadata_file_is_rejected(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    out = tmp_path / "out"
    write_run_metadata(out, config={"a": 1}, repo_root=repo)
    path = out / "run_meta.json"

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config"]["a"] = 999  # tamper content without recomputing the stored hash
    tampered = json.dumps(payload)
    path.write_text(tampered, encoding="utf-8")

    with pytest.raises(RunMetadataError):
        write_run_metadata(out, config={"a": 1}, repo_root=repo)
    assert path.read_text(encoding="utf-8") == tampered


def test_simulated_atomic_write_failure_leaves_no_partial_file(tmp_path, monkeypatch):
    def failing_replace(self, target):
        raise OSError("simulated failure during rename")

    monkeypatch.setattr(Path, "replace", failing_replace)

    out = tmp_path / "out"
    with pytest.raises(OSError):
        write_run_metadata(out, config={"a": 1})

    assert not (out / "run_meta.json").exists()
    assert list(out.glob("run_meta.json.tmp-*")) == []


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
