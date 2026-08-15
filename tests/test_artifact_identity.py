"""Guards on the frozen model source-artifact manifest (protocol Gate 0).

Mirrors the HarmfulQA split-manifest tests in shape: canonical-hash order-independence,
frozen-identity pinning against tampering or a merely self-consistent substitute,
structural-invariant rejection, and offline streaming verification against tiny local
files. Entirely offline -- no network, no model download, no GPU, no merge.

Run with:
    pytest tests/test_artifact_identity.py -v
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from steering.artifact_identity import (
    ArtifactIdentityError,
    EXPECTED_ARTIFACT_IDS,
    FROZEN_MODEL_ARTIFACT_IDENTITY,
    build_manifest,
    load_manifest,
    save_manifest,
    validate_frozen_identity,
    validate_manifest_structure,
    verify_local_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_MANIFEST_PATH = REPO_ROOT / "manifests" / "model_artifacts_v1.json"
HARMFULQA_MANIFEST_PATH = REPO_ROOT / "manifests" / "harmfulqa_v1.json"
HARMFULQA_MANIFEST_HASH = "0e1d0a8177d59eba601141a74da19a189527821dab573e203a3261cff54eec70"


def _hex40(fill: str = "a") -> str:
    return fill * 40


def _hex64(fill: str = "a") -> str:
    return fill * 64


def _synthetic_artifacts(overrides=None):
    """Five structurally valid artifacts, one per EXPECTED_ARTIFACT_IDS, with
    well-formed but dummy hex identifiers -- independent of the real frozen values, so
    structural/order-independence tests don't depend on the audit's real hashes."""
    base = {
        "pair_a_sft": {
            "artifact_id": "pair_a_sft", "repository_type": "model", "repository": "org/repo-a",
            "revision": _hex40("1"), "subpath": "sft", "kind": "checkpoint", "base_artifact_id": None,
            "inference_ready": True,
            "files": [{"path": "model.safetensors", "sha256": _hex64("1"), "size_bytes": None}],
            "lineage": {},
        },
        "pair_a_dpo": {
            "artifact_id": "pair_a_dpo", "repository_type": "model", "repository": "org/repo-a",
            "revision": _hex40("1"), "subpath": "dpo", "kind": "checkpoint", "base_artifact_id": None,
            "inference_ready": True,
            "files": [{"path": "model.safetensors", "sha256": _hex64("2"), "size_bytes": None}],
            "lineage": {},
        },
        "pair_b_sft": {
            "artifact_id": "pair_b_sft", "repository_type": "model", "repository": "org/repo-b",
            "revision": _hex40("3"), "subpath": "", "kind": "checkpoint", "base_artifact_id": None,
            "inference_ready": True,
            "files": [{"path": "model.safetensors", "sha256": _hex64("3"), "size_bytes": None}],
            "lineage": {},
        },
        "pair_b_dpo_adapter": {
            "artifact_id": "pair_b_dpo_adapter", "repository_type": "dataset", "repository": "org/results",
            "revision": _hex40("4"), "subpath": "runs/b", "kind": "adapter", "base_artifact_id": "pair_b_sft",
            "inference_ready": False,
            "files": [{"path": "adapter_model.safetensors", "sha256": _hex64("4"), "size_bytes": None}],
            "lineage": {},
        },
        "pair_a_flip_adapter": {
            "artifact_id": "pair_a_flip_adapter", "repository_type": "dataset", "repository": "org/results",
            "revision": _hex40("4"), "subpath": "runs/a-flip", "kind": "adapter", "base_artifact_id": "pair_a_sft",
            "inference_ready": False,
            "files": [{"path": "adapter_model.safetensors", "sha256": _hex64("5"), "size_bytes": None}],
            "lineage": {},
        },
    }
    if overrides:
        for aid, patch in overrides.items():
            base[aid] = {**base[aid], **patch}
    return list(base.values())


# Canonical hash order-independence


def test_canonical_hash_is_independent_of_artifact_list_order():
    forward = _synthetic_artifacts()
    backward = list(reversed(forward))
    a = build_manifest(forward)
    b = build_manifest(backward)
    assert a == b
    assert a["manifest_hash"] == b["manifest_hash"]


def test_canonical_hash_is_independent_of_per_artifact_key_order():
    artifacts = _synthetic_artifacts()
    reordered = [dict(reversed(list(a.items()))) for a in artifacts]
    a = build_manifest(artifacts)
    b = build_manifest(reordered)
    assert a["manifest_hash"] == b["manifest_hash"]


def test_canonical_hash_is_independent_of_file_list_order_within_an_artifact():
    artifacts = _synthetic_artifacts(overrides={
        "pair_a_sft": {"files": [
            {"path": "model.safetensors", "sha256": _hex64("1"), "size_bytes": None},
            {"path": "tokenizer.json", "sha256": _hex64("9"), "size_bytes": None},
        ]},
    })
    swapped = copy.deepcopy(artifacts)
    for a in swapped:
        if a["artifact_id"] == "pair_a_sft":
            a["files"] = list(reversed(a["files"]))
    x = build_manifest(artifacts)
    y = build_manifest(swapped)
    assert x["manifest_hash"] == y["manifest_hash"]


def test_same_inputs_reproduce_the_same_hash():
    artifacts = _synthetic_artifacts()
    a = build_manifest(artifacts)
    b = build_manifest(artifacts)
    assert a["manifest_hash"] == b["manifest_hash"]


def test_changing_a_hash_changes_the_manifest_hash():
    a = build_manifest(_synthetic_artifacts())
    b = build_manifest(_synthetic_artifacts(overrides={
        "pair_a_sft": {"files": [{"path": "model.safetensors", "sha256": _hex64("f"), "size_bytes": None}]},
    }))
    assert a["manifest_hash"] != b["manifest_hash"]


# Committed manifest: load, self-verify, frozen identity


def test_committed_manifest_loads_verifies_and_passes_frozen_identity():
    manifest = load_manifest(COMMITTED_MANIFEST_PATH)  # raises if the stored hash is stale
    validate_manifest_structure(manifest)
    validate_frozen_identity(manifest)  # must not raise
    assert manifest["manifest_hash"] == FROZEN_MODEL_ARTIFACT_IDENTITY["manifest_hash"]
    assert {a["artifact_id"] for a in manifest["artifacts"]} == set(EXPECTED_ARTIFACT_IDS)


def test_a_content_change_with_a_stale_stored_hash_fails_to_load(tmp_path):
    manifest = load_manifest(COMMITTED_MANIFEST_PATH)
    tampered = copy.deepcopy(manifest)
    tampered["artifacts"][0]["revision"] = _hex40("9")  # manifest_hash left stale
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ArtifactIdentityError):
        load_manifest(path)


# Tampering rejected by frozen identity


def test_tampering_a_revision_is_rejected_by_frozen_identity():
    manifest = load_manifest(COMMITTED_MANIFEST_PATH)
    tampered = copy.deepcopy(manifest)
    for a in tampered["artifacts"]:
        if a["artifact_id"] == "pair_a_sft":
            a["revision"] = _hex40("9")
    tampered["manifest_hash"] = "0" * 64  # any self-consistency check is separate; force past it
    with pytest.raises(ArtifactIdentityError):
        validate_frozen_identity(tampered)


def test_tampering_a_base_link_is_rejected_by_frozen_identity():
    manifest = load_manifest(COMMITTED_MANIFEST_PATH)
    tampered = copy.deepcopy(manifest)
    for a in tampered["artifacts"]:
        if a["artifact_id"] == "pair_a_flip_adapter":
            a["base_artifact_id"] = "pair_b_sft"  # a real artifact, but the wrong one
    with pytest.raises(ArtifactIdentityError):
        validate_frozen_identity(tampered)


def test_tampering_a_key_file_hash_is_rejected_by_frozen_identity():
    manifest = load_manifest(COMMITTED_MANIFEST_PATH)
    tampered = copy.deepcopy(manifest)
    for a in tampered["artifacts"]:
        if a["artifact_id"] == "pair_b_dpo_adapter":
            for f in a["files"]:
                if f["path"] == "adapter_model.safetensors":
                    f["sha256"] = _hex64("9")
    with pytest.raises(ArtifactIdentityError):
        validate_frozen_identity(tampered)


def test_an_alternate_self_consistent_manifest_still_fails_frozen_identity():
    """A manifest that is internally valid and even hashes correctly, but simply is not
    the one artifact set this task is reviewed against, must not be accepted."""
    alternate = build_manifest(_synthetic_artifacts())
    validate_manifest_structure(alternate)  # passes on its own
    with pytest.raises(ArtifactIdentityError):
        validate_frozen_identity(alternate)


# Structural invariants


def test_malformed_revision_is_rejected():
    bad = _synthetic_artifacts(overrides={"pair_a_sft": {"revision": "not-forty-hex-chars"}})
    with pytest.raises(ArtifactIdentityError):
        build_manifest(bad)


def test_uppercase_revision_is_rejected():
    bad = _synthetic_artifacts(overrides={"pair_a_sft": {"revision": _hex40("A")}})
    with pytest.raises(ArtifactIdentityError):
        build_manifest(bad)


def test_malformed_sha256_is_rejected():
    bad = _synthetic_artifacts(overrides={
        "pair_a_sft": {"files": [{"path": "model.safetensors", "sha256": "short", "size_bytes": None}]},
    })
    with pytest.raises(ArtifactIdentityError):
        build_manifest(bad)


def test_duplicate_artifact_ids_are_rejected():
    artifacts = _synthetic_artifacts()
    duplicate = dict(artifacts[0])
    with pytest.raises(ArtifactIdentityError):
        build_manifest(artifacts + [duplicate])


def test_a_missing_expected_artifact_id_is_rejected():
    artifacts = [a for a in _synthetic_artifacts() if a["artifact_id"] != "pair_b_sft"]
    with pytest.raises(ArtifactIdentityError):
        build_manifest(artifacts)


def test_duplicate_file_paths_within_one_artifact_are_rejected():
    bad = _synthetic_artifacts(overrides={
        "pair_a_sft": {"files": [
            {"path": "model.safetensors", "sha256": _hex64("1"), "size_bytes": None},
            {"path": "model.safetensors", "sha256": _hex64("2"), "size_bytes": None},
        ]},
    })
    with pytest.raises(ArtifactIdentityError):
        build_manifest(bad)


def test_an_absolute_path_is_rejected():
    bad = _synthetic_artifacts(overrides={
        "pair_a_sft": {"files": [{"path": "/etc/passwd", "sha256": _hex64("1"), "size_bytes": None}]},
    })
    with pytest.raises(ArtifactIdentityError):
        build_manifest(bad)


def test_a_windows_drive_absolute_path_is_rejected():
    bad = _synthetic_artifacts(overrides={
        "pair_a_sft": {"files": [{"path": "C:\\secrets\\model.safetensors", "sha256": _hex64("1"), "size_bytes": None}]},
    })
    with pytest.raises(ArtifactIdentityError):
        build_manifest(bad)


def test_a_traversal_path_is_rejected():
    bad = _synthetic_artifacts(overrides={
        "pair_a_sft": {"files": [{"path": "../../etc/passwd", "sha256": _hex64("1"), "size_bytes": None}]},
    })
    with pytest.raises(ArtifactIdentityError):
        build_manifest(bad)


def test_an_adapter_without_a_base_is_rejected():
    bad = _synthetic_artifacts(overrides={"pair_b_dpo_adapter": {"base_artifact_id": None}})
    with pytest.raises(ArtifactIdentityError):
        build_manifest(bad)


def test_an_adapter_based_on_another_adapter_is_rejected():
    bad = _synthetic_artifacts(overrides={"pair_b_dpo_adapter": {"base_artifact_id": "pair_a_flip_adapter"}})
    with pytest.raises(ArtifactIdentityError):
        build_manifest(bad)


def test_a_checkpoint_with_a_base_artifact_is_rejected():
    bad = _synthetic_artifacts(overrides={"pair_a_sft": {"base_artifact_id": "pair_b_sft"}})
    with pytest.raises(ArtifactIdentityError):
        build_manifest(bad)


# save_manifest immutability


def test_an_existing_different_manifest_is_not_overwritten(tmp_path):
    original = build_manifest(_synthetic_artifacts())
    path = tmp_path / "manifest.json"
    save_manifest(original, path)
    on_disk_before = path.read_text(encoding="utf-8")

    different = build_manifest(_synthetic_artifacts(overrides={
        "pair_a_sft": {"revision": _hex40("f")},
    }))
    with pytest.raises(ArtifactIdentityError):
        save_manifest(different, path)
    assert path.read_text(encoding="utf-8") == on_disk_before


def test_rebuilding_an_identical_manifest_does_not_rewrite(tmp_path):
    payload = build_manifest(_synthetic_artifacts())
    path = tmp_path / "manifest.json"
    save_manifest(payload, path)
    mtime_before = path.stat().st_mtime_ns
    written_again = save_manifest(payload, path)
    assert written_again is False
    assert path.stat().st_mtime_ns == mtime_before


# Offline local-file verification


def _write(path: Path, content: bytes) -> str:
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def test_a_correct_tiny_local_artifact_verifies_successfully(tmp_path):
    weight = tmp_path / "model.safetensors"
    tok = tmp_path / "tokenizer.json"
    weight_sha = _write(weight, b"tiny fake weight bytes")
    tok_sha = _write(tok, b"{}")

    manifest = build_manifest(_synthetic_artifacts(overrides={
        "pair_a_sft": {"files": [
            {"path": "model.safetensors", "sha256": weight_sha, "size_bytes": weight.stat().st_size},
            {"path": "tokenizer.json", "sha256": tok_sha, "size_bytes": None},
        ]},
    }))

    verified = verify_local_artifact(manifest, "pair_a_sft", tmp_path)
    assert {v.path for v in verified} == {"model.safetensors", "tokenizer.json"}
    by_path = {v.path: v for v in verified}
    assert by_path["model.safetensors"].sha256 == weight_sha
    assert by_path["model.safetensors"].size_bytes == weight.stat().st_size


def test_verification_rejects_an_unknown_artifact_id(tmp_path):
    manifest = build_manifest(_synthetic_artifacts())
    with pytest.raises(ArtifactIdentityError):
        verify_local_artifact(manifest, "not_a_real_artifact", tmp_path)


def test_verification_rejects_a_missing_file(tmp_path):
    manifest = build_manifest(_synthetic_artifacts(overrides={
        "pair_a_sft": {"files": [{"path": "model.safetensors", "sha256": _hex64("1"), "size_bytes": None}]},
    }))
    # tmp_path is empty -- the declared file does not exist.
    with pytest.raises(ArtifactIdentityError):
        verify_local_artifact(manifest, "pair_a_sft", tmp_path)


def test_verification_rejects_a_wrong_size(tmp_path):
    weight = tmp_path / "model.safetensors"
    sha = _write(weight, b"some content")
    manifest = build_manifest(_synthetic_artifacts(overrides={
        "pair_a_sft": {"files": [{"path": "model.safetensors", "sha256": sha, "size_bytes": 999999}]},
    }))
    with pytest.raises(ArtifactIdentityError):
        verify_local_artifact(manifest, "pair_a_sft", tmp_path)


def test_verification_rejects_a_wrong_hash(tmp_path):
    weight = tmp_path / "model.safetensors"
    _write(weight, b"some content")
    manifest = build_manifest(_synthetic_artifacts(overrides={
        "pair_a_sft": {"files": [{"path": "model.safetensors", "sha256": _hex64("f"), "size_bytes": None}]},
    }))
    with pytest.raises(ArtifactIdentityError):
        verify_local_artifact(manifest, "pair_a_sft", tmp_path)


def test_verification_rejects_a_directory_instead_of_a_file(tmp_path):
    (tmp_path / "model.safetensors").mkdir()
    manifest = build_manifest(_synthetic_artifacts(overrides={
        "pair_a_sft": {"files": [{"path": "model.safetensors", "sha256": _hex64("1"), "size_bytes": None}]},
    }))
    with pytest.raises(ArtifactIdentityError):
        verify_local_artifact(manifest, "pair_a_sft", tmp_path)


def test_verification_rejects_a_root_escaping_declared_path_even_in_an_ad_hoc_manifest(tmp_path):
    """The manifest here is constructed in-memory, not loaded/validated first -- the
    escape guard inside verify_local_artifact itself must still catch it."""
    escape_target = tmp_path.parent / "outside_secret.txt"
    escape_target.write_bytes(b"should never be read")
    root = tmp_path / "root"
    root.mkdir()

    manifest = {
        "schema_version": 1,
        "artifacts": [{
            "artifact_id": "pair_a_sft", "repository_type": "model", "repository": "org/repo-a",
            "revision": _hex40("1"), "subpath": "sft", "kind": "checkpoint", "base_artifact_id": None,
            "inference_ready": True, "lineage": {},
            "files": [{"path": "../outside_secret.txt", "sha256": _hex64("1"), "size_bytes": None}],
        }],
    }
    with pytest.raises(ArtifactIdentityError):
        verify_local_artifact(manifest, "pair_a_sft", root)


# HarmfulQA manifest untouched


def test_harmfulqa_manifest_remains_byte_identical():
    assert HARMFULQA_MANIFEST_PATH.exists(), "this task must not remove the HarmfulQA manifest"
    payload = json.loads(HARMFULQA_MANIFEST_PATH.read_text(encoding="utf-8"))
    body = {k: v for k, v in payload.items() if k != "manifest_hash"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == HARMFULQA_MANIFEST_HASH
    assert payload["manifest_hash"] == HARMFULQA_MANIFEST_HASH
