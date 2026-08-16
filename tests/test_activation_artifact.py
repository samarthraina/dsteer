"""Guards on `src/steering/activation_artifact.py` (Task 016): the sidecar that binds
one construction `activations.pt` to its exact originating `run_meta.json` and
endpoint pair.

CPU-only, offline: tiny synthetic tensors, hand-built run_meta.json payloads (signed
with the real `_compute_run_identity_hash` algorithm), and a mocked frozen HarmfulQA
manifest. No real model, dataset, network, GPU, or judge access.

Run with:
    pytest tests/test_activation_artifact.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steering import activation_artifact as aa  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: a valid primary_v1 run_meta.json, a matching construction blob, and a
# frozen construction manifest mock (real hashing, synthetic identities)
# ---------------------------------------------------------------------------


def _signed_run_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(payload)
    payload["run_identity_hash"] = aa._compute_run_identity_hash(payload)
    return payload


def _fake_construction_manifest(n: int = 1378) -> Dict[str, Any]:
    return {
        "manifest_hash": "construction-manifest-hash",
        "records": [
            {"source_id": f"construction-{i}", "prompt_hash": f"construction-hash-{i}", "partition": "construction", "permuted_position": i}
            for i in range(n)
        ],
    }


def _mock_frozen_construction_manifest(monkeypatch, n: int = 1378) -> None:
    monkeypatch.setattr(aa, "load_manifest", lambda path: _fake_construction_manifest(n))
    monkeypatch.setattr(aa, "validate_manifest_identity", lambda m: None)


def _fake_blob(n: int = 1378, num_layers: int = 4, hidden: int = 8) -> Dict[str, Any]:
    return {
        "it": torch.randn(num_layers, n, hidden), "dpo": torch.randn(num_layers, n, hidden),
        "source_ids": [f"construction-{i}" for i in range(n)],
        "prompt_hashes": [f"construction-hash-{i}" for i in range(n)],
        "partition": "construction", "manifest_hash": "construction-manifest-hash",
    }


def _endpoint(pair: str = "A") -> Dict[str, Any]:
    it_role, dpo_role = ("M0-A", "M+-A") if pair == "A" else ("M0-B", "M+-B")
    return {
        "pair": pair, "candidate_manifest_hash": "c" * 64, "source_manifest_hash": "s" * 64,
        "roles": {"it": it_role, "dpo": dpo_role},
    }


def _valid_run_meta(pair: str = "A", n: int = 1378) -> Dict[str, Any]:
    return _signed_run_meta({
        "protocol_profile": "primary_v1", "endpoint_backed": True,
        "endpoint": _endpoint(pair),
        "harmfulqa_partition": "construction", "harmfulqa_manifest_hash": "construction-manifest-hash",
        "harmfulqa_record_count": n,
        "config": {"eval": {"token_position": "prompt_last"}},
        "model_loading_policy": dict(aa.SAFE_LOADING_POLICY),
    })


def _write_construction_dir(
    tmp_path: Path, name: str, *, pair: str = "A", n: int = 1378, blob: Dict[str, Any] = None,
    run_meta: Dict[str, Any] = None, publish: bool = True,
) -> Path:
    """A construction directory with activations.pt and run_meta.json, optionally
    published (real sidecar, via the actual publish_activation_artifact) or left
    without one for tests that build/omit a sidecar themselves."""
    d = tmp_path / name
    d.mkdir(parents=True)
    blob = blob if blob is not None else _fake_blob(n)
    torch.save(blob, d / "activations.pt")
    run_meta = run_meta if run_meta is not None else _valid_run_meta(pair, n)
    (d / "run_meta.json").write_text(json.dumps(run_meta), encoding="utf-8")
    if publish:
        aa.publish_activation_artifact(d)
    return d


def _hand_build_sidecar(
    d: Path, run_meta: Dict[str, Any], blob: Dict[str, Any], harmfulqa_manifest_hash_override: str = None,
) -> Dict[str, Any]:
    """Build and write a sidecar directly (bypassing publish_activation_artifact
    entirely), for simulating a manually rehashed sidecar/run_meta.json pair that
    consistently claims something publish itself would now refuse to write -- the
    exact tampering scenario the central validator (not the publisher) must catch."""
    it_tensor, dpo_tensor = blob["it"], blob["dpo"]
    endpoint = run_meta.get("endpoint") or {}
    roles = endpoint.get("roles") or {}
    sidecar = {
        "schema_version": aa.SCHEMA_VERSION, "artifact_kind": aa.ARTIFACT_KIND,
        "activation_file": aa.ACTIVATION_FILENAME, "activation_sha256": aa._stream_sha256(d / "activations.pt"),
        "activation_size_bytes": (d / "activations.pt").stat().st_size,
        "run_meta_file": aa.RUN_META_FILENAME, "run_identity_hash": run_meta["run_identity_hash"],
        "protocol_profile": run_meta.get("protocol_profile"), "endpoint_backed": run_meta.get("endpoint_backed"),
        "endpoint": {
            "pair": endpoint.get("pair"), "candidate_manifest_hash": endpoint.get("candidate_manifest_hash"),
            "source_manifest_hash": endpoint.get("source_manifest_hash"),
            "roles": {"it": roles.get("it"), "dpo": roles.get("dpo")},
        },
        "harmfulqa": {
            "partition": run_meta.get("harmfulqa_partition"),
            "manifest_hash": (
                harmfulqa_manifest_hash_override if harmfulqa_manifest_hash_override is not None
                else run_meta.get("harmfulqa_manifest_hash")
            ),
            "record_count": run_meta.get("harmfulqa_record_count"),
        },
        "tensors": {
            "it_shape": list(it_tensor.shape), "dpo_shape": list(dpo_tensor.shape),
            "it_dtype": str(it_tensor.dtype), "dpo_dtype": str(dpo_tensor.dtype),
        },
    }
    sidecar["manifest_hash"] = aa._compute_sidecar_hash(sidecar)
    (d / aa.SIDECAR_FILENAME).write_text(json.dumps(sidecar), encoding="utf-8")
    return sidecar


# ---------------------------------------------------------------------------
# verify_run_metadata_identity
# ---------------------------------------------------------------------------


def test_verify_run_metadata_identity_accepts_a_correctly_signed_payload():
    payload = _signed_run_meta({"a": 1})
    aa.verify_run_metadata_identity(payload, "ctx")  # must not raise


def test_verify_run_metadata_identity_rejects_a_non_dict():
    with pytest.raises(aa.ActivationArtifactError):
        aa.verify_run_metadata_identity(["not", "a", "dict"], "ctx")


@pytest.mark.parametrize("bad_hash", [None, "", 123, "not-recomputed"])
def test_verify_run_metadata_identity_rejects_a_missing_or_wrong_hash(bad_hash):
    payload = _signed_run_meta({"a": 1})
    payload["run_identity_hash"] = bad_hash
    with pytest.raises(aa.ActivationArtifactError):
        aa.verify_run_metadata_identity(payload, "ctx")


def test_verify_run_metadata_identity_rejects_a_field_edited_without_recomputing():
    payload = _signed_run_meta({"a": 1})
    payload["a"] = 2  # edited, hash left stale
    with pytest.raises(aa.ActivationArtifactError):
        aa.verify_run_metadata_identity(payload, "ctx")


# ---------------------------------------------------------------------------
# save_activations_atomic
# ---------------------------------------------------------------------------


def test_save_activations_atomic_writes_the_file_and_no_leftover_temp(tmp_path):
    path = tmp_path / "sub" / "activations.pt"
    aa.save_activations_atomic({"it": torch.zeros(2, 3, 4), "dpo": torch.zeros(2, 3, 4)}, path)
    assert path.exists()
    assert list(path.parent.glob("activations.pt.tmp-*")) == []
    reloaded = torch.load(path, map_location="cpu")
    assert reloaded["it"].shape == (2, 3, 4)


def test_save_activations_atomic_cleans_up_its_temp_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "activations.pt"

    def boom_save(obj, f):
        raise RuntimeError("simulated torch.save failure")

    monkeypatch.setattr(aa.torch, "save", boom_save)
    with pytest.raises(RuntimeError):
        aa.save_activations_atomic({"it": torch.zeros(1)}, path)
    assert not path.exists()
    assert list(tmp_path.glob("activations.pt.tmp-*")) == []


# ---------------------------------------------------------------------------
# publish_activation_artifact
# ---------------------------------------------------------------------------


def test_publish_activation_artifact_builds_the_exact_schema(tmp_path, monkeypatch):
    _mock_frozen_construction_manifest(monkeypatch)
    n = 1378
    blob = _fake_blob(n)
    run_meta = _valid_run_meta("A", n)
    d = _write_construction_dir(tmp_path, "acts", blob=blob, run_meta=run_meta, publish=False)

    sidecar = aa.publish_activation_artifact(d)

    assert sidecar["schema_version"] == 1
    assert sidecar["artifact_kind"] == "dsteer_construction_activations"
    assert sidecar["activation_file"] == "activations.pt"
    assert sidecar["run_meta_file"] == "run_meta.json"
    assert sidecar["run_identity_hash"] == run_meta["run_identity_hash"]
    assert sidecar["protocol_profile"] == "primary_v1"
    assert sidecar["endpoint_backed"] is True
    assert sidecar["endpoint"] == {
        "pair": "A", "candidate_manifest_hash": "c" * 64, "source_manifest_hash": "s" * 64,
        "roles": {"it": "M0-A", "dpo": "M+-A"},
    }
    assert sidecar["harmfulqa"] == {"partition": "construction", "manifest_hash": "construction-manifest-hash", "record_count": n}
    assert sidecar["tensors"] == {
        "it_shape": list(blob["it"].shape), "dpo_shape": list(blob["dpo"].shape),
        "it_dtype": str(blob["it"].dtype), "dpo_dtype": str(blob["dpo"].dtype),
    }
    assert sidecar["activation_sha256"] == aa._stream_sha256(d / "activations.pt")
    assert sidecar["activation_size_bytes"] == (d / "activations.pt").stat().st_size
    assert sidecar["manifest_hash"] == aa._compute_sidecar_hash(sidecar)

    on_disk = json.loads((d / aa.SIDECAR_FILENAME).read_text(encoding="utf-8"))
    assert on_disk == sidecar


def test_publish_activation_artifact_rejects_missing_activations_file(tmp_path):
    d = tmp_path / "acts"
    d.mkdir()
    (d / "run_meta.json").write_text(json.dumps(_valid_run_meta()), encoding="utf-8")
    with pytest.raises(aa.ActivationArtifactError):
        aa.publish_activation_artifact(d)


def test_publish_activation_artifact_rejects_missing_run_meta(tmp_path):
    d = tmp_path / "acts"
    d.mkdir()
    torch.save(_fake_blob(10), d / "activations.pt")
    with pytest.raises(aa.ActivationArtifactError):
        aa.publish_activation_artifact(d)


def test_publish_activation_artifact_rejects_a_run_meta_with_no_valid_self_hash(tmp_path):
    d = tmp_path / "acts"
    d.mkdir()
    torch.save(_fake_blob(10), d / "activations.pt")
    bad = _valid_run_meta(n=10)
    del bad["run_identity_hash"]
    (d / "run_meta.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(aa.ActivationArtifactError):
        aa.publish_activation_artifact(d)


def test_publish_activation_artifact_refuses_a_non_primary_v1_run(tmp_path):
    d = tmp_path / "acts"
    d.mkdir()
    torch.save(_fake_blob(10), d / "activations.pt")
    run_meta = _signed_run_meta({
        "protocol_profile": "legacy_nonconfirmatory", "endpoint_backed": False,
    })
    (d / "run_meta.json").write_text(json.dumps(run_meta), encoding="utf-8")
    with pytest.raises(aa.ActivationArtifactError):
        aa.publish_activation_artifact(d)


def test_publish_activation_artifact_refuses_a_run_not_marked_endpoint_backed(tmp_path):
    d = tmp_path / "acts"
    d.mkdir()
    torch.save(_fake_blob(10), d / "activations.pt")
    run_meta = _signed_run_meta({"protocol_profile": "primary_v1", "endpoint_backed": False})
    (d / "run_meta.json").write_text(json.dumps(run_meta), encoding="utf-8")
    with pytest.raises(aa.ActivationArtifactError):
        aa.publish_activation_artifact(d)


# ---------------------------------------------------------------------------
# load_and_validate_activation_artifact -- the central loader/validator
# ---------------------------------------------------------------------------


def test_load_and_validate_accepts_a_valid_bound_artifact(tmp_path, monkeypatch):
    _mock_frozen_construction_manifest(monkeypatch)
    d = _write_construction_dir(tmp_path, "acts")
    artifact = aa.load_and_validate_activation_artifact(
        d, expected_pair="A", expected_candidate_manifest_hash="c" * 64,
        expected_source_manifest_hash="s" * 64, expected_it_role="M0-A", expected_dpo_role="M+-A",
    )
    assert artifact.acts_path == d / "activations.pt"
    assert artifact.sha256 == aa._stream_sha256(d / "activations.pt")
    assert artifact.size_bytes == (d / "activations.pt").stat().st_size
    assert "it" in artifact.blob and "dpo" in artifact.blob
    assert artifact.run_meta["protocol_profile"] == "primary_v1"
    assert artifact.sidecar["schema_version"] == 1


@pytest.mark.parametrize("missing", ["activations.pt", "run_meta.json", aa.SIDECAR_FILENAME])
def test_load_and_validate_rejects_a_missing_sibling_file(tmp_path, monkeypatch, missing):
    _mock_frozen_construction_manifest(monkeypatch)
    d = _write_construction_dir(tmp_path, "acts")
    (d / missing).unlink()
    with pytest.raises(aa.ActivationArtifactError):
        aa.load_and_validate_activation_artifact(d)


def test_load_and_validate_rejects_activations_bytes_changed_after_publish(tmp_path, monkeypatch):
    _mock_frozen_construction_manifest(monkeypatch)
    d = _write_construction_dir(tmp_path, "acts")
    blob = torch.load(d / "activations.pt", map_location="cpu")
    blob["it"] = blob["it"] + 1.0
    torch.save(blob, d / "activations.pt")
    with pytest.raises(aa.ActivationArtifactError):
        aa.load_and_validate_activation_artifact(d)


def test_load_and_validate_rejects_a_sidecar_edited_without_recomputing_its_hash(tmp_path, monkeypatch):
    _mock_frozen_construction_manifest(monkeypatch)
    d = _write_construction_dir(tmp_path, "acts")
    sidecar_path = d / aa.SIDECAR_FILENAME
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["activation_size_bytes"] = payload["activation_size_bytes"] + 1
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(aa.ActivationArtifactError):
        aa.load_and_validate_activation_artifact(d)


def test_load_and_validate_rejects_a_manually_rehashed_sidecar_with_the_wrong_run_identity_hash(tmp_path, monkeypatch):
    """The sidecar's own self-hash is recomputed correctly (so the self-check alone
    would pass) -- only the run_identity_hash it claims for run_meta.json is wrong,
    proving the binding is a genuine cross-file comparison."""
    _mock_frozen_construction_manifest(monkeypatch)
    d = _write_construction_dir(tmp_path, "acts")
    sidecar_path = d / aa.SIDECAR_FILENAME
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["run_identity_hash"] = "0" * 64
    payload["manifest_hash"] = aa._compute_sidecar_hash(payload)  # rehashed
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(aa.ActivationArtifactError):
        aa.load_and_validate_activation_artifact(d)


def test_load_and_validate_rejects_a_rehashed_run_meta_and_sidecar_consistently_claiming_the_wrong_manifest(tmp_path, monkeypatch):
    """activations.pt itself carries the real, frozen-verified HarmfulQA manifest hash
    -- but run_meta.json and its sidecar were manually rehashed to consistently claim a
    *different* one. Neither file's own self-hash check catches this (each is
    internally consistent, and the two agree with each other); only the cross-file
    equality against the actual blob does."""
    _mock_frozen_construction_manifest(monkeypatch)
    blob = _fake_blob(1378)  # blob["manifest_hash"] == "construction-manifest-hash" (the real one)
    stale = {k: v for k, v in _valid_run_meta(n=1378).items() if k != "run_identity_hash"}
    stale["harmfulqa_manifest_hash"] = "a-different-manifest-hash"
    run_meta = _signed_run_meta(stale)  # rehashed -- self-consistent
    d = tmp_path / "acts"
    d.mkdir()
    torch.save(blob, d / "activations.pt")
    (d / "run_meta.json").write_text(json.dumps(run_meta), encoding="utf-8")
    _hand_build_sidecar(d, run_meta, blob)  # mirrors run_meta's (wrong) claim, self-consistent
    with pytest.raises(aa.ActivationArtifactError):
        aa.load_and_validate_activation_artifact(d)


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("artifact_kind", "something_else"),
    ("activation_file", "not-activations.pt"), ("activation_file", "/etc/passwd"),
    ("run_meta_file", "not-run-meta.json"), ("run_meta_file", "../run_meta.json"),
])
def test_load_and_validate_rejects_wrong_sidecar_schema_fields(tmp_path, monkeypatch, field, value):
    _mock_frozen_construction_manifest(monkeypatch)
    d = _write_construction_dir(tmp_path, "acts")
    sidecar_path = d / aa.SIDECAR_FILENAME
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload[field] = value
    payload["manifest_hash"] = aa._compute_sidecar_hash(payload)
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(aa.ActivationArtifactError):
        aa.load_and_validate_activation_artifact(d)


def test_load_and_validate_rejects_protocol_profile_mismatch_between_sidecar_and_run_meta(tmp_path, monkeypatch):
    _mock_frozen_construction_manifest(monkeypatch)
    d = _write_construction_dir(tmp_path, "acts")
    sidecar_path = d / aa.SIDECAR_FILENAME
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["protocol_profile"] = "legacy_nonconfirmatory"
    payload["manifest_hash"] = aa._compute_sidecar_hash(payload)
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(aa.ActivationArtifactError):
        aa.load_and_validate_activation_artifact(d)


def test_load_and_validate_rejects_endpoint_backed_false(tmp_path, monkeypatch):
    _mock_frozen_construction_manifest(monkeypatch)
    d = _write_construction_dir(tmp_path, "acts")
    sidecar_path = d / aa.SIDECAR_FILENAME
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["endpoint_backed"] = False
    payload["manifest_hash"] = aa._compute_sidecar_hash(payload)
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(aa.ActivationArtifactError):
        aa.load_and_validate_activation_artifact(d)


@pytest.mark.parametrize("path_in_sidecar", [
    ["endpoint", "pair"], ["endpoint", "candidate_manifest_hash"], ["endpoint", "source_manifest_hash"],
    ["endpoint", "roles", "it"], ["endpoint", "roles", "dpo"],
    ["harmfulqa", "partition"], ["harmfulqa", "manifest_hash"], ["harmfulqa", "record_count"],
])
def test_load_and_validate_rejects_sidecar_disagreeing_with_run_meta(tmp_path, monkeypatch, path_in_sidecar):
    """A sidecar field that no longer matches run_meta.json's own claim -- the
    cross-file self-consistency check, independent of what the caller expects."""
    _mock_frozen_construction_manifest(monkeypatch)
    d = _write_construction_dir(tmp_path, "acts")
    sidecar_path = d / aa.SIDECAR_FILENAME
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    node = payload
    for key in path_in_sidecar[:-1]:
        node = node[key]
    node[path_in_sidecar[-1]] = "tampered-value"
    payload["manifest_hash"] = aa._compute_sidecar_hash(payload)
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(aa.ActivationArtifactError):
        aa.load_and_validate_activation_artifact(d)


def test_publish_activation_artifact_rejects_run_meta_partition_mismatched_with_blob(tmp_path, monkeypatch):
    """publish_activation_artifact now validates the payload itself before writing the
    sidecar: a run_meta.json claiming harmfulqa_partition="calibration" beside a blob
    whose actual partition is "construction" must fail, and must produce no sidecar."""
    _mock_frozen_construction_manifest(monkeypatch)
    stale = _valid_run_meta()
    stale = {k: v for k, v in stale.items() if k != "run_identity_hash"}
    stale["harmfulqa_partition"] = "calibration"
    run_meta = _signed_run_meta(stale)
    d = _write_construction_dir(tmp_path, "acts", run_meta=run_meta, publish=False)
    with pytest.raises(aa.ActivationArtifactError):
        aa.publish_activation_artifact(d)
    assert not (d / aa.SIDECAR_FILENAME).exists()


def test_publish_activation_artifact_rejects_wrong_construction_record_count(tmp_path, monkeypatch):
    """The blob has only 1377 records against the (mocked) frozen manifest's 1378 --
    validate_construction_identity, now run before the sidecar is written, catches this
    at publish time; no sidecar is ever produced for it."""
    _mock_frozen_construction_manifest(monkeypatch)
    blob = _fake_blob(1377)
    run_meta = _valid_run_meta(n=1377)
    d = _write_construction_dir(tmp_path, "acts", n=1377, blob=blob, run_meta=run_meta, publish=False)
    with pytest.raises(aa.ActivationArtifactError):
        aa.publish_activation_artifact(d)
    assert not (d / aa.SIDECAR_FILENAME).exists()


def test_load_and_validate_rejects_tensor_shape_claim_mismatch(tmp_path, monkeypatch):
    """The sidecar claims a tensor shape that does not match the actual saved tensor --
    a claim alone is never trusted over the real, freshly-loaded tensor."""
    _mock_frozen_construction_manifest(monkeypatch)
    d = _write_construction_dir(tmp_path, "acts")
    sidecar_path = d / aa.SIDECAR_FILENAME
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["tensors"]["it_shape"] = [999, 999, 999]
    payload["manifest_hash"] = aa._compute_sidecar_hash(payload)
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(aa.ActivationArtifactError):
        aa.load_and_validate_activation_artifact(d)


def test_load_and_validate_rejects_tensor_dtype_claim_mismatch(tmp_path, monkeypatch):
    _mock_frozen_construction_manifest(monkeypatch)
    d = _write_construction_dir(tmp_path, "acts")
    sidecar_path = d / aa.SIDECAR_FILENAME
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["tensors"]["dpo_dtype"] = "torch.float64"
    payload["manifest_hash"] = aa._compute_sidecar_hash(payload)
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(aa.ActivationArtifactError):
        aa.load_and_validate_activation_artifact(d)


def test_publish_activation_artifact_rejects_a_malformed_frozen_identity(tmp_path, monkeypatch):
    """Reordered source_ids -- a malformed frozen identity -- is now caught by
    validate_construction_identity at publish time, not merely left for the validator
    to discover later; no sidecar is produced for it."""
    _mock_frozen_construction_manifest(monkeypatch)
    blob = _fake_blob(1378)
    blob["source_ids"][0], blob["source_ids"][1] = blob["source_ids"][1], blob["source_ids"][0]
    d = _write_construction_dir(tmp_path, "acts", blob=blob, publish=False)
    with pytest.raises(ValueError):
        aa.publish_activation_artifact(d)
    assert not (d / aa.SIDECAR_FILENAME).exists()


# Caller-expected-identity checks: the actual Pair A / Pair B substitution defense.


def test_load_and_validate_rejects_wrong_expected_pair(tmp_path, monkeypatch):
    _mock_frozen_construction_manifest(monkeypatch)
    d = _write_construction_dir(tmp_path, "acts", pair="A")
    with pytest.raises(aa.ActivationArtifactError):
        aa.load_and_validate_activation_artifact(d, expected_pair="B")


def test_load_and_validate_rejects_wrong_expected_candidate_manifest_hash(tmp_path, monkeypatch):
    _mock_frozen_construction_manifest(monkeypatch)
    d = _write_construction_dir(tmp_path, "acts", pair="A")
    with pytest.raises(aa.ActivationArtifactError):
        aa.load_and_validate_activation_artifact(d, expected_candidate_manifest_hash="0" * 64)


def test_load_and_validate_rejects_wrong_expected_source_manifest_hash(tmp_path, monkeypatch):
    _mock_frozen_construction_manifest(monkeypatch)
    d = _write_construction_dir(tmp_path, "acts", pair="A")
    with pytest.raises(aa.ActivationArtifactError):
        aa.load_and_validate_activation_artifact(d, expected_source_manifest_hash="0" * 64)


def test_load_and_validate_rejects_wrong_expected_roles(tmp_path, monkeypatch):
    _mock_frozen_construction_manifest(monkeypatch)
    d = _write_construction_dir(tmp_path, "acts", pair="A")
    with pytest.raises(aa.ActivationArtifactError):
        aa.load_and_validate_activation_artifact(d, expected_it_role="M0-B")
    with pytest.raises(aa.ActivationArtifactError):
        aa.load_and_validate_activation_artifact(d, expected_dpo_role="M+-B")


def test_load_and_validate_rejects_a_pair_b_artifact_substituted_for_pair_a(tmp_path, monkeypatch):
    """The direct Task 016 scenario: a genuinely valid, fully self-consistent,
    correctly-published Pair B artifact -- rejected only because the caller checks it
    against Pair A's own resolved identity."""
    _mock_frozen_construction_manifest(monkeypatch)
    pair_b_dir = _write_construction_dir(tmp_path, "acts_B", pair="B")
    with pytest.raises(aa.ActivationArtifactError):
        aa.load_and_validate_activation_artifact(
            pair_b_dir, expected_pair="A", expected_it_role="M0-A", expected_dpo_role="M+-A",
        )


def test_load_and_validate_with_no_expected_identity_given_accepts_either_pair(tmp_path, monkeypatch):
    """When the caller supplies no expected identity at all, only internal binding is
    checked -- documents that the substitution defense is opt-in via expected_*, which
    every real caller (steer_sweep.py, development_pilot.py) always supplies."""
    _mock_frozen_construction_manifest(monkeypatch)
    pair_b_dir = _write_construction_dir(tmp_path, "acts_B", pair="B")
    artifact = aa.load_and_validate_activation_artifact(pair_b_dir)
    assert artifact.sidecar["endpoint"]["pair"] == "B"


# ---------------------------------------------------------------------------
# validate_construction_identity (the relocated, canonical frozen-manifest check;
# steer_sweep.validate_construction_activations is a thin wrapper around this)
# ---------------------------------------------------------------------------


def test_validate_construction_identity_accepts_a_matching_blob(monkeypatch):
    _mock_frozen_construction_manifest(monkeypatch)
    aa.validate_construction_identity(_fake_blob(1378))  # must not raise


def test_validate_construction_identity_rejects_a_legacy_blob_missing_provenance(monkeypatch):
    _mock_frozen_construction_manifest(monkeypatch)
    with pytest.raises(ValueError):
        aa.validate_construction_identity({"it": torch.zeros(2, 3, 4), "dpo": torch.zeros(2, 3, 4)})


def test_validate_construction_identity_rejects_wrong_manifest_hash(monkeypatch):
    _mock_frozen_construction_manifest(monkeypatch)
    blob = _fake_blob(1378)
    blob["manifest_hash"] = "wrong"
    with pytest.raises(ValueError):
        aa.validate_construction_identity(blob)
