"""Guards on `steering.endpoint_binding` (protocol Sections 4/13/14, Task 012).

Endpoint-backed mode resolves a Pair A/B (IT, DPO) model pair from a hash- and
structurally-verified candidate endpoint manifest, binds it to the verified frozen
source-artifact manifest (`manifests/model_artifacts_v1.json`, via
`steering.artifact_identity`), and stream-verifies every declared inference file
against local disk before any of it is trusted. Entirely offline and CPU-only: tiny
real files and synthetic (but real-hash-verified) candidate/source manifests, no
network, no model, no GPU, no real endpoint creation.

The real committed `manifests/model_artifacts_v1.json` is never used here -- every test
builds its own synthetic-but-internally-consistent source manifest and points
`source_manifest_path` at it, so the *pinned frozen identity* check
(`artifact_identity.validate_frozen_identity` against the real
`FROZEN_MODEL_ARTIFACT_IDENTITY`) is bypassed via the module-level autouse fixture
below; that pin is already exercised by tests/test_artifact_identity.py. Everything
else -- hash verification, structural validation, and this module's own
candidate-vs-frozen-source cross-check -- runs for real.

Run with:
    pytest tests/test_endpoint_binding.py -v
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steering import artifact_identity as art_id
from steering import endpoint_binding as eb
from steering import endpoint_manifest as em

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _bypass_real_frozen_identity(monkeypatch):
    """Every test here builds its own synthetic (but internally consistent,
    real-hash-verified) source manifest -- it will never match the actual pinned
    identity in `artifact_identity.FROZEN_MODEL_ARTIFACT_IDENTITY`, which is exercised
    separately in tests/test_artifact_identity.py. Bypass only that one check."""
    monkeypatch.setattr(eb, "validate_frozen_source_identity", lambda m: None)


# ---------------------------------------------------------------------------
# Synthetic (but real-hash-verified) candidate manifest + matching frozen source
# manifest + real local roots
# ---------------------------------------------------------------------------


def _hex64(fill: str = "a") -> str:
    return fill * 64


def _hex40(fill: str = "2") -> str:
    return fill * 40


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_file(root: Path, rel_path: str, content: bytes) -> Dict[str, object]:
    full = root / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(content)
    return {"path": rel_path, "sha256": _sha256(content), "size_bytes": len(content)}


_A_ARCH = {"model_type": "llama", "hidden_size": 8, "num_hidden_layers": 2, "vocab_size": 12, "embedding_rows": 12, "lm_head_rows": 12}
_B_ARCH = {"model_type": "mistral", "hidden_size": 8, "num_hidden_layers": 2, "vocab_size": 16, "embedding_rows": 16, "lm_head_rows": 16}
_A_FP = _hex64("a")
_A_SPECIAL = {"bos_token_id": 1}
_B_FP = _hex64("b")
_B_SPECIAL = {"bos_token_id": 1}

_MMA_ADAPTER_ANCHOR = {"path": "adapter_model.safetensors", "sha256": _hex64("f"), "size_bytes": 10}
_MPB_ADAPTER_ANCHOR = {"path": "adapter_model.safetensors", "sha256": _hex64("e"), "size_bytes": 10}


def _validation(arch, fp, special):
    return {
        **arch, "tied_embeddings": False, "tokenizer_loadable": True, "no_residual_peft_modules": True,
        "forward_pass_smoke_test": True, "tokenizer_fingerprint": fp, "special_token_ids": special,
    }


def _tr_base_only(vocab_size, fp, special):
    return {
        "status": "base_only", "base_vocab_size": vocab_size, "adapter_vocab_size": None,
        "old_vocab_size": None, "new_vocab_size": None, "added_token_ids": [],
        "base_fingerprint": fp, "adapter_fingerprint": None,
        "base_special_token_ids": special, "adapter_special_token_ids": None,
    }


def build_candidate_env(tmp_path: Path):
    """Real local roots (direct sources, plus a bundle root for merged endpoints) with
    tiny real files whose hashes are computed (not invented). Builds a candidate
    endpoint manifest (`endpoint_manifest.build_manifest`) and a matching synthetic
    frozen source manifest (`artifact_identity.build_manifest`) whose
    repository/revision/subpath/anchor-file hashes are exactly what the candidate
    declares -- so the "happy path" binds cleanly, and a test can tamper one side or
    the other to exercise a specific mismatch.

    Returns (manifest_path, source_roots, bundle_root, candidate, source_manifest_path).
    """
    a_sft_root = tmp_path / "src" / "pair_a_sft"
    a_dpo_root = tmp_path / "src" / "pair_a_dpo"
    b_sft_root = tmp_path / "src" / "pair_b_sft"

    a_sft_files = [
        _write_file(a_sft_root, "model.safetensors", b"a-sft-weights"),
        _write_file(a_sft_root, "tokenizer.json", b"a-tokenizer"),
    ]
    a_dpo_files = [
        _write_file(a_dpo_root, "model.safetensors", b"a-dpo-weights"),
        _write_file(a_dpo_root, "tokenizer.json", b"a-tokenizer"),
    ]
    b_sft_files = [
        _write_file(b_sft_root, "model.safetensors", b"b-sft-weights"),
        _write_file(b_sft_root, "tokenizer.json", b"b-tokenizer"),
    ]

    bundle_root = tmp_path / "bundle"
    mma_files = [
        _write_file(bundle_root / "M--A", "model.safetensors", b"a-flip-weights"),
        _write_file(bundle_root / "M--A", "tokenizer.json", b"a-tokenizer"),
    ]
    mpb_files = [
        _write_file(bundle_root / "M+-B", "model.safetensors", b"b-dpo-weights"),
        _write_file(bundle_root / "M+-B", "tokenizer.json", b"b-tokenizer"),
    ]

    source_artifacts = [
        {
            "artifact_id": "pair_a_sft", "repository_type": "model", "repository": "org/pair-a",
            "revision": _hex40("2"), "subpath": "SFT_merged", "kind": "checkpoint", "base_artifact_id": None,
            "inference_ready": True, "lineage": {}, "files": a_sft_files,
        },
        {
            "artifact_id": "pair_a_dpo", "repository_type": "model", "repository": "org/pair-a",
            "revision": _hex40("2"), "subpath": "DPO_merged", "kind": "checkpoint", "base_artifact_id": None,
            "inference_ready": True, "lineage": {}, "files": a_dpo_files,
        },
        {
            "artifact_id": "pair_b_sft", "repository_type": "model", "repository": "org/pair-b",
            "revision": _hex40("3"), "subpath": "", "kind": "checkpoint", "base_artifact_id": None,
            "inference_ready": True, "lineage": {}, "files": b_sft_files,
        },
        {
            "artifact_id": "pair_a_flip_adapter", "repository_type": "dataset", "repository": "org/results",
            "revision": _hex40("4"), "subpath": "runs/a-flip", "kind": "adapter", "base_artifact_id": "pair_a_sft",
            "inference_ready": False, "lineage": {}, "files": [_MMA_ADAPTER_ANCHOR],
        },
        {
            "artifact_id": "pair_b_dpo_adapter", "repository_type": "dataset", "repository": "org/results",
            "revision": _hex40("4"), "subpath": "runs/b-dpo", "kind": "adapter", "base_artifact_id": "pair_b_sft",
            "inference_ready": False, "lineage": {}, "files": [_MPB_ADAPTER_ANCHOR],
        },
    ]
    source_manifest = art_id.build_manifest(source_artifacts)
    source_manifest_path = tmp_path / "model_artifacts_v1.json"
    art_id.save_manifest(source_manifest, source_manifest_path)
    source_manifest_hash = source_manifest["manifest_hash"]

    endpoints = [
        {
            "role": "M0-A", "status": "direct", "source_manifest_hash": source_manifest_hash,
            "source_artifact_id": "pair_a_sft", "base_artifact_id": None,
            "location": {"kind": "source", "repository": "org/pair-a", "revision": _hex40("2"), "subpath": "SFT_merged"},
            "merge": None, "device": "cpu", "library_versions": {"torch": "0.0.0"},
            "validation": _validation(_A_ARCH, _A_FP, _A_SPECIAL), "files": a_sft_files,
        },
        {
            "role": "M+-A", "status": "direct", "source_manifest_hash": source_manifest_hash,
            "source_artifact_id": "pair_a_dpo", "base_artifact_id": None,
            "location": {"kind": "source", "repository": "org/pair-a", "revision": _hex40("2"), "subpath": "DPO_merged"},
            "merge": None, "device": "cpu", "library_versions": {"torch": "0.0.0"},
            "validation": _validation(_A_ARCH, _A_FP, _A_SPECIAL), "files": a_dpo_files,
        },
        {
            "role": "M--A", "status": "merged", "source_manifest_hash": source_manifest_hash,
            "source_artifact_id": "pair_a_flip_adapter", "base_artifact_id": "pair_a_sft",
            "location": {"kind": "bundle", "path": "M--A"}, "device": "cpu",
            "merge": {
                "dtype": "bfloat16", "max_shard_size": "5GB", "merge_seed": 1,
                "tokenizer_resolution": _tr_base_only(_A_ARCH["vocab_size"], _A_FP, _A_SPECIAL),
                "adapter_input_files": [dict(_MMA_ADAPTER_ANCHOR)],
                "flip_lineage": {
                    "archived_trainer_state_path": "runs/x/checkpoint-2510/trainer_state.json",
                    "sha256": _hex64("7"), "training_step": 2510,
                    "training_script": None, "launch_config": None,
                    "label_swap_lineage_verified": False, "confirmatory_eligible": False,
                },
            },
            "library_versions": {"torch": "0.0.0", "peft": "0.0.0"},
            "validation": _validation(_A_ARCH, _A_FP, _A_SPECIAL), "files": mma_files,
        },
        {
            "role": "M0-B", "status": "direct", "source_manifest_hash": source_manifest_hash,
            "source_artifact_id": "pair_b_sft", "base_artifact_id": None,
            "location": {"kind": "source", "repository": "org/pair-b", "revision": _hex40("3"), "subpath": ""},
            "merge": None, "device": "cpu", "library_versions": {"torch": "0.0.0"},
            "validation": _validation(_B_ARCH, _B_FP, _B_SPECIAL), "files": b_sft_files,
        },
        {
            "role": "M+-B", "status": "merged", "source_manifest_hash": source_manifest_hash,
            "source_artifact_id": "pair_b_dpo_adapter", "base_artifact_id": "pair_b_sft",
            "location": {"kind": "bundle", "path": "M+-B"}, "device": "cpu",
            "merge": {
                "dtype": "bfloat16", "max_shard_size": "5GB", "merge_seed": 1,
                "tokenizer_resolution": _tr_base_only(_B_ARCH["vocab_size"], _B_FP, _B_SPECIAL),
                "adapter_input_files": [dict(_MPB_ADAPTER_ANCHOR)],
            },
            "library_versions": {"torch": "0.0.0", "peft": "0.0.0"},
            "validation": _validation(_B_ARCH, _B_FP, _B_SPECIAL), "files": mpb_files,
        },
    ]

    candidate = em.build_manifest(source_manifest_hash, endpoints)
    manifest_path = tmp_path / "endpoint_manifest_candidate_v1.json"
    em.save_manifest(candidate, manifest_path)

    source_roots = {"pair_a_sft": a_sft_root, "pair_a_dpo": a_dpo_root, "pair_b_sft": b_sft_root}
    return manifest_path, source_roots, bundle_root, candidate, source_manifest_path


def _pair_a_roots(source_roots):
    return {"pair_a_sft": source_roots["pair_a_sft"], "pair_a_dpo": source_roots["pair_a_dpo"]}


def _tamper_and_rehash(manifest_path: Path, mutate) -> None:
    """Load the candidate off disk, apply `mutate` in place, and recompute+rewrite its
    own `manifest_hash` -- simulating a manually constructed candidate that is
    perfectly self-consistent (its stored hash matches its own content) but lies about
    something an external authority must catch."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(payload)
    payload["manifest_hash"] = em.compute_manifest_hash(payload)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")


def _set_all_source_manifest_hash(payload, new_hash):
    payload["source_manifest_hash"] = new_hash
    for e in payload["endpoints"]:
        e["source_manifest_hash"] = new_hash


def _tamper_direct_revision(payload):
    for e in payload["endpoints"]:
        if e["role"] == "M0-A":
            e["location"]["revision"] = _hex40("9")


def _tamper_direct_anchor_hash(payload):
    for e in payload["endpoints"]:
        if e["role"] == "M0-A":
            for f in e["files"]:
                if f["path"] == "model.safetensors":
                    f["sha256"] = _hex64("0")


def _tamper_merged_adapter_anchor_hash(payload):
    for e in payload["endpoints"]:
        if e["role"] == "M--A":
            e["merge"]["adapter_input_files"][0]["sha256"] = _hex64("0")


# Role mapping


def test_roles_for_pair_a():
    assert eb.roles_for_pair("A") == ("M0-A", "M+-A")


def test_roles_for_pair_b():
    assert eb.roles_for_pair("B") == ("M0-B", "M+-B")


def test_roles_for_an_unknown_pair_is_rejected():
    with pytest.raises(eb.EndpointBindingError):
        eb.roles_for_pair("C")


def test_resolve_role_entry_can_resolve_a_role_outside_the_wired_pairs(tmp_path):
    """The role resolver must be reusable for a future flip experiment: it can resolve
    M--A directly, not only the two roles currently wired into a pair."""
    _, _, _, candidate, _ = build_candidate_env(tmp_path)
    entry = eb.resolve_role_entry(candidate, "M--A")
    assert entry["role"] == "M--A"
    assert entry["status"] == "merged"


def test_resolve_role_entry_rejects_an_unknown_role(tmp_path):
    _, _, _, candidate, _ = build_candidate_env(tmp_path)
    with pytest.raises(eb.EndpointBindingError):
        eb.resolve_role_entry(candidate, "M-Z")


# Direct and merged path resolution


def test_resolve_pair_a_resolves_both_direct_endpoints(tmp_path):
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)
    resolved = eb.resolve_pair(manifest_path, "A", _pair_a_roots(source_roots), bundle_root, source_manifest_path=source_manifest_path)

    assert resolved.it.role == "M0-A" and resolved.it.status == "direct"
    assert resolved.it.local_path == source_roots["pair_a_sft"].resolve()
    assert resolved.dpo.role == "M+-A" and resolved.dpo.status == "direct"
    assert resolved.dpo.local_path == source_roots["pair_a_dpo"].resolve()
    assert {f["path"] for f in resolved.it.verified_files} == {"model.safetensors", "tokenizer.json"}


def test_resolve_pair_b_resolves_direct_it_and_merged_dpo(tmp_path):
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)
    resolved = eb.resolve_pair(
        manifest_path, "B", {"pair_b_sft": source_roots["pair_b_sft"]}, bundle_root, source_manifest_path=source_manifest_path,
    )

    assert resolved.it.role == "M0-B" and resolved.it.status == "direct"
    assert resolved.it.local_path == source_roots["pair_b_sft"].resolve()
    assert resolved.dpo.role == "M+-B" and resolved.dpo.status == "merged"
    assert resolved.dpo.local_path == (Path(bundle_root) / "M+-B").resolve()


# Manifest hash / structural failure (self-consistency, independent of frozen binding)


def test_resolve_pair_rejects_a_tampered_manifest_hash(tmp_path):
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["endpoints"][0]["device"] = "cuda:0"  # manifest_hash left stale
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(em.EndpointManifestError):
        eb.resolve_pair(manifest_path, "A", _pair_a_roots(source_roots), bundle_root, source_manifest_path=source_manifest_path)


def test_resolve_pair_rejects_a_structurally_invalid_but_correctly_hashed_manifest(tmp_path):
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)
    _tamper_and_rehash(manifest_path, lambda p: p["endpoints"][0]["validation"].__delitem__("forward_pass_smoke_test"))

    with pytest.raises(em.EndpointManifestError):
        eb.resolve_pair(manifest_path, "A", _pair_a_roots(source_roots), bundle_root, source_manifest_path=source_manifest_path)


# Frozen source-artifact binding (correction round 1): a correctly self-rehashed
# candidate must still be rejected if it disagrees with the verified frozen source
# manifest, not only with itself.


def test_wrong_but_well_formed_source_manifest_hash_is_rejected(tmp_path):
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)
    _tamper_and_rehash(manifest_path, lambda p: _set_all_source_manifest_hash(p, _hex64("5")))

    with pytest.raises(eb.EndpointBindingError, match="source_manifest_hash"):
        eb.resolve_pair(manifest_path, "A", _pair_a_roots(source_roots), bundle_root, source_manifest_path=source_manifest_path)


def test_correctly_rehashed_candidate_with_a_changed_direct_revision_is_rejected(tmp_path):
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)
    _tamper_and_rehash(manifest_path, _tamper_direct_revision)

    with pytest.raises(eb.EndpointBindingError, match="location.revision"):
        eb.resolve_pair(manifest_path, "A", _pair_a_roots(source_roots), bundle_root, source_manifest_path=source_manifest_path)


def test_correctly_rehashed_candidate_with_a_changed_direct_anchor_hash_is_rejected(tmp_path):
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)
    _tamper_and_rehash(manifest_path, _tamper_direct_anchor_hash)

    with pytest.raises(eb.EndpointBindingError, match="frozen anchor file"):
        eb.resolve_pair(manifest_path, "A", _pair_a_roots(source_roots), bundle_root, source_manifest_path=source_manifest_path)


def test_correctly_rehashed_candidate_with_a_changed_merged_adapter_anchor_hash_is_rejected(tmp_path):
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)
    _tamper_and_rehash(manifest_path, _tamper_merged_adapter_anchor_hash)

    with pytest.raises(eb.EndpointBindingError, match="frozen anchor file"):
        eb.resolve_pair(manifest_path, "A", _pair_a_roots(source_roots), bundle_root, source_manifest_path=source_manifest_path)


def test_frozen_source_binding_rejects_an_unknown_source_artifact_id(tmp_path):
    """Corrupt the frozen source manifest side (remove an artifact the candidate still
    references). Structural validation of the source manifest itself requires exactly
    the five expected artifact_ids, so this is caught there -- a different, but still
    valid, rejection point from the candidate-vs-frozen cross-check."""
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)

    source_payload = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_payload["artifacts"] = [a for a in source_payload["artifacts"] if a["artifact_id"] != "pair_a_sft"]
    source_payload["manifest_hash"] = art_id.compute_manifest_hash(source_payload)
    source_manifest_path.write_text(json.dumps(source_payload), encoding="utf-8")

    with pytest.raises(art_id.ArtifactIdentityError):
        eb.resolve_pair(manifest_path, "A", _pair_a_roots(source_roots), bundle_root, source_manifest_path=source_manifest_path)


def test_frozen_source_binding_runs_for_the_whole_candidate_not_only_the_requested_pair(tmp_path):
    """A candidate that lies about Pair B must be rejected even when only Pair A is
    being resolved -- the binding check covers every declared endpoint."""
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)

    def mutate(p):
        for e in p["endpoints"]:
            if e["role"] == "M0-B":
                e["location"]["revision"] = _hex40("9")

    _tamper_and_rehash(manifest_path, mutate)

    with pytest.raises(eb.EndpointBindingError, match="location.revision"):
        eb.resolve_pair(manifest_path, "A", _pair_a_roots(source_roots), bundle_root, source_manifest_path=source_manifest_path)


# Complete run-metadata provenance (correction round 1, item 2)


def test_metadata_includes_frozen_source_repository_revision_subpath_and_anchor_files(tmp_path):
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)
    resolved = eb.resolve_pair(manifest_path, "A", _pair_a_roots(source_roots), bundle_root, source_manifest_path=source_manifest_path)

    it_meta = resolved.it.as_metadata()
    assert it_meta["location"] == resolved.it.location
    assert it_meta["frozen_source"] == {
        "repository": "org/pair-a", "repository_type": "model", "revision": _hex40("2"), "subpath": "SFT_merged",
        "anchor_files": [
            {"path": "model.safetensors", "sha256": _sha256(b"a-sft-weights")},
            {"path": "tokenizer.json", "sha256": _sha256(b"a-tokenizer")},
        ],
    }
    assert it_meta["base_frozen_source"] is None  # M0-A has no base


def test_merged_endpoint_metadata_includes_its_own_and_its_base_frozen_source(tmp_path):
    """Requirement: a merged endpoint's metadata must record the repository/revision of
    its own source adapter, not only the artifact_id -- and identify its base too."""
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)
    resolved = eb.resolve_pair(
        manifest_path, "B", {"pair_b_sft": source_roots["pair_b_sft"]}, bundle_root, source_manifest_path=source_manifest_path,
    )

    dpo_meta = resolved.dpo.as_metadata()  # M+-B, merged, source_artifact_id=pair_b_dpo_adapter
    assert dpo_meta["status"] == "merged"
    assert dpo_meta["frozen_source"]["repository"] == "org/results"
    assert dpo_meta["frozen_source"]["revision"] == _hex40("4")
    assert dpo_meta["frozen_source"]["subpath"] == "runs/b-dpo"
    assert dpo_meta["frozen_source"]["anchor_files"] == [{"path": "adapter_model.safetensors", "sha256": _hex64("e")}]

    assert dpo_meta["base_frozen_source"] is not None
    assert dpo_meta["base_frozen_source"]["repository"] == "org/pair-b"
    assert dpo_meta["base_frozen_source"]["revision"] == _hex40("3")


# Source-mapping validation: missing, duplicate, unknown, wrong-role


def test_missing_source_mapping_is_rejected(tmp_path):
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)
    incomplete = {"pair_a_sft": source_roots["pair_a_sft"]}  # pair_a_dpo missing
    with pytest.raises(eb.EndpointBindingError, match="missing"):
        eb.resolve_pair(manifest_path, "A", incomplete, bundle_root, source_manifest_path=source_manifest_path)


def test_duplicate_source_mapping_is_rejected():
    with pytest.raises(eb.EndpointBindingError, match="duplicate"):
        eb.parse_source_mappings(["pair_a_sft=/one", "pair_a_sft=/two"])


def test_malformed_source_mapping_is_rejected():
    with pytest.raises(eb.EndpointBindingError):
        eb.parse_source_mappings(["pair_a_sft-without-equals-sign"])


def test_unknown_source_mapping_is_rejected(tmp_path):
    """A mapping for an artifact_id that is valid *elsewhere* in the manifest but not
    needed for this pair is still rejected, not silently ignored."""
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)
    extra = dict(source_roots)  # pair_a_sft, pair_a_dpo, pair_b_sft
    with pytest.raises(eb.EndpointBindingError, match="unexpected"):
        eb.resolve_pair(manifest_path, "A", extra, bundle_root, source_manifest_path=source_manifest_path)


def test_a_wrong_role_swapped_mapping_fails_file_verification(tmp_path):
    """Swapping which local root is declared for which artifact_id is not caught by
    key-level validation (both keys are individually valid and complete) -- it is
    caught by content verification, since the swapped root's real file hashes do not
    match the role they were mapped to."""
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)
    swapped = {"pair_a_sft": source_roots["pair_a_dpo"], "pair_a_dpo": source_roots["pair_a_sft"]}
    with pytest.raises(eb.EndpointBindingError, match="sha256 mismatch"):
        eb.resolve_pair(manifest_path, "A", swapped, bundle_root, source_manifest_path=source_manifest_path)


# File verification: size mismatch, same-size content mismatch, missing/unexpected file


def test_file_size_mismatch_is_rejected(tmp_path):
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)
    (source_roots["pair_a_sft"] / "model.safetensors").write_bytes(b"a-sft-weights-but-longer-now")
    with pytest.raises(eb.EndpointBindingError, match="size mismatch"):
        eb.resolve_pair(manifest_path, "A", _pair_a_roots(source_roots), bundle_root, source_manifest_path=source_manifest_path)


def test_same_size_content_hash_mismatch_is_rejected(tmp_path):
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)
    original = source_roots["pair_a_sft"] / "model.safetensors"
    original_len = len(original.read_bytes())
    tampered = b"X" * original_len  # same length, different content
    assert tampered != b"a-sft-weights"
    original.write_bytes(tampered)
    with pytest.raises(eb.EndpointBindingError, match="sha256 mismatch"):
        eb.resolve_pair(manifest_path, "A", _pair_a_roots(source_roots), bundle_root, source_manifest_path=source_manifest_path)


def test_missing_inference_file_is_rejected(tmp_path):
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)
    (source_roots["pair_a_sft"] / "tokenizer.json").unlink()
    with pytest.raises(eb.EndpointBindingError, match="missing declared file"):
        eb.resolve_pair(manifest_path, "A", _pair_a_roots(source_roots), bundle_root, source_manifest_path=source_manifest_path)


def test_unexpected_inference_file_is_rejected(tmp_path):
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)
    (source_roots["pair_a_sft"] / "extra_not_declared.bin").write_bytes(b"surprise")
    with pytest.raises(eb.EndpointBindingError, match="unexpected file"):
        eb.resolve_pair(manifest_path, "A", _pair_a_roots(source_roots), bundle_root, source_manifest_path=source_manifest_path)


def test_verify_endpoint_files_never_reads_a_mismatched_file_before_checking_size(tmp_path):
    """Size is checked before hashing -- a corrupt/huge file with the wrong size must
    fail without ever being streamed."""
    root = tmp_path / "root"
    (root).mkdir()
    (root / "model.safetensors").write_bytes(b"short")
    declared = [{"path": "model.safetensors", "sha256": _hex64("0"), "size_bytes": 999999}]
    with pytest.raises(eb.EndpointBindingError, match="size mismatch"):
        eb.verify_endpoint_files(root, declared)


# Bundle-path traversal / absolute-path / symlink-escape rejection


def test_merged_location_traversal_is_rejected(tmp_path):
    entry = {"role": "M--A", "location": {"kind": "bundle", "path": "../escape"}}
    with pytest.raises(eb.EndpointBindingError, match="traversal"):
        eb._resolve_merged_local_root(entry, tmp_path / "bundle")


def test_merged_location_absolute_path_is_rejected(tmp_path):
    entry = {"role": "M--A", "location": {"kind": "bundle", "path": "/etc/passwd"}}
    with pytest.raises(eb.EndpointBindingError, match="relative"):
        eb._resolve_merged_local_root(entry, tmp_path / "bundle")


def test_merged_location_windows_absolute_path_is_rejected(tmp_path):
    entry = {"role": "M--A", "location": {"kind": "bundle", "path": "C:\\Users\\me\\model"}}
    with pytest.raises(eb.EndpointBindingError, match="relative"):
        eb._resolve_merged_local_root(entry, tmp_path / "bundle")


def test_merged_location_wrong_kind_is_rejected(tmp_path):
    entry = {"role": "M--A", "location": {"kind": "source", "path": "M--A"}}
    with pytest.raises(eb.EndpointBindingError, match="bundle location"):
        eb._resolve_merged_local_root(entry, tmp_path / "bundle")


def test_verify_endpoint_files_rejects_an_escaping_symlink(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside content")
    link = root / "model.safetensors"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("creating symlinks requires elevated privileges on this platform")

    declared = [{"path": "model.safetensors", "sha256": _sha256(b"outside content"), "size_bytes": len(b"outside content")}]
    with pytest.raises(eb.EndpointBindingError, match="escaping"):
        eb.verify_endpoint_files(root, declared)


def test_verify_endpoint_files_accepts_a_symlink_that_stays_inside_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    real = root / "real.bin"
    real.write_bytes(b"content")
    link = root / "alias.bin"
    try:
        link.symlink_to(real)
    except OSError:
        pytest.skip("creating symlinks requires elevated privileges on this platform")

    declared = [
        {"path": "real.bin", "sha256": _sha256(b"content"), "size_bytes": len(b"content")},
        {"path": "alias.bin", "sha256": _sha256(b"content"), "size_bytes": len(b"content")},
    ]
    verified = eb.verify_endpoint_files(root, declared)
    assert {f["path"] for f in verified} == {"real.bin", "alias.bin"}


# resolve_model_source: mutual exclusivity, incomplete-mode rejection, metadata content


def _write_yaml_model_config(path: Path) -> Path:
    import yaml
    path.write_text(yaml.safe_dump({
        "name": "legacy-pair", "base_model": "org/base", "it_model": "org/it",
        "dpo_model": "org/dpo", "architecture": "llama", "num_layers": 4,
    }), encoding="utf-8")
    return path


def test_resolve_model_source_rejects_both_modes_given_at_once(tmp_path):
    model_yaml = _write_yaml_model_config(tmp_path / "model.yaml")
    with pytest.raises(eb.EndpointBindingError, match="mutually exclusive"):
        eb.resolve_model_source(
            model_config=str(model_yaml), endpoint_manifest="whatever.json",
            endpoint_bundle_root=None, pair=None, endpoint_source=[],
        )


def test_resolve_model_source_rejects_neither_mode_given():
    with pytest.raises(eb.EndpointBindingError, match="no model source given"):
        eb.resolve_model_source(
            model_config=None, endpoint_manifest=None,
            endpoint_bundle_root=None, pair=None, endpoint_source=[],
        )


@pytest.mark.parametrize("missing_field", ["endpoint_manifest", "endpoint_bundle_root", "pair", "endpoint_source"])
def test_resolve_model_source_rejects_an_incomplete_endpoint_mode(tmp_path, missing_field):
    kwargs = dict(
        endpoint_manifest=str(tmp_path / "m.json"), endpoint_bundle_root=str(tmp_path / "bundle"),
        pair="A", endpoint_source=["pair_a_sft=/x", "pair_a_dpo=/y"],
    )
    kwargs[missing_field] = None if missing_field != "endpoint_source" else []
    with pytest.raises(eb.EndpointBindingError, match="missing required flag"):
        eb.resolve_model_source(model_config=None, **kwargs)


def test_resolve_model_source_legacy_mode_returns_none_metadata(tmp_path):
    model_yaml = _write_yaml_model_config(tmp_path / "model.yaml")
    model_cfg, meta = eb.resolve_model_source(
        model_config=str(model_yaml), endpoint_manifest=None,
        endpoint_bundle_root=None, pair=None, endpoint_source=[],
    )
    assert model_cfg.name == "legacy-pair"
    assert meta is None


def test_resolve_model_source_endpoint_mode_builds_config_and_metadata(tmp_path):
    manifest_path, source_roots, bundle_root, candidate, source_manifest_path = build_candidate_env(tmp_path)
    model_cfg, meta = eb.resolve_model_source(
        model_config=None, endpoint_manifest=str(manifest_path), endpoint_bundle_root=str(bundle_root),
        pair="A", endpoint_source=[f"pair_a_sft={source_roots['pair_a_sft']}", f"pair_a_dpo={source_roots['pair_a_dpo']}"],
        source_manifest_path=source_manifest_path,
    )

    # Architecture/layer count/tokenizer come from validated endpoint facts, never CLI input.
    assert model_cfg.architecture == "llama"
    assert model_cfg.num_layers == 2
    assert model_cfg.tokenizer_id is None
    assert model_cfg.it_subfolder is None and model_cfg.dpo_subfolder is None
    assert model_cfg.it_model == str(source_roots["pair_a_sft"].resolve())
    assert model_cfg.dpo_model == str(source_roots["pair_a_dpo"].resolve())

    # Complete endpoint provenance for run_meta.json (protocol Section 14).
    assert meta["mode"] == "endpoint"
    assert meta["pair"] == "A"
    assert meta["candidate_manifest_hash"] == candidate["manifest_hash"]
    assert meta["source_manifest_hash"] == candidate["source_manifest_hash"]
    assert meta["roles"] == {"it": "M0-A", "dpo": "M+-A"}
    for side in ("it", "dpo"):
        e = meta["endpoints"][side]
        assert e["status"] == "direct"
        assert e["validation"]["model_type"] == "llama"
        assert e["local_path"]
        assert e["location"]["kind"] == "source"
        assert e["frozen_source"]["repository"] == "org/pair-a"
        assert {f["path"] for f in e["verified_files"]} == {"model.safetensors", "tokenizer.json"}
        for f in e["verified_files"]:
            assert isinstance(f["sha256"], str) and len(f["sha256"]) == 64
            assert isinstance(f["size_bytes"], int)


def test_resolve_model_source_is_deterministic_for_the_same_manifest_and_pair(tmp_path):
    """layer_profile.py and steer_sweep.py must land on the same `model_cfg.name` for
    the same candidate manifest and pair, so steer_sweep's activation lookup finds what
    layer_profile wrote -- without either script hardcoding the other's path."""
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)
    kwargs = dict(
        model_config=None, endpoint_manifest=str(manifest_path), endpoint_bundle_root=str(bundle_root),
        pair="A", endpoint_source=[f"pair_a_sft={source_roots['pair_a_sft']}", f"pair_a_dpo={source_roots['pair_a_dpo']}"],
        source_manifest_path=source_manifest_path,
    )
    cfg1, _ = eb.resolve_model_source(**kwargs)
    cfg2, _ = eb.resolve_model_source(**kwargs)
    assert cfg1.name == cfg2.name


def test_default_run_name_is_deterministic_and_pair_specific():
    a = eb.default_run_name("A", "abc123def456")
    b = eb.default_run_name("B", "abc123def456")
    assert a != b
    assert eb.default_run_name("A", "abc123def456") == a


# to_model_config: never trusts a model_type mismatch even if it somehow occurs


def _fake_resolved_endpoint(role, source_artifact_id, model_type, local_path):
    return eb.ResolvedEndpoint(
        role=role, status="direct", source_artifact_id=source_artifact_id, base_artifact_id=None,
        local_path=Path(local_path), validation={"model_type": model_type, "num_hidden_layers": 2},
        location={}, device="cpu", library_versions={}, merge=None, verified_files=[],
        frozen_source={}, base_frozen_source=None,
    )


def test_to_model_config_rejects_a_model_type_mismatch_between_it_and_dpo():
    it = _fake_resolved_endpoint("M0-A", "pair_a_sft", "llama", "/it")
    dpo = _fake_resolved_endpoint("M+-A", "pair_a_dpo", "mistral", "/dpo")
    resolved = eb.ResolvedPair(pair="A", it=it, dpo=dpo, candidate_manifest_hash=_hex64("1"), source_manifest_hash=_hex64("2"))
    with pytest.raises(eb.EndpointBindingError, match="model_type disagree"):
        resolved.to_model_config("name")


# resolve_all_roles / ALL_ROLES (Task 013): the reusable all-role counterpart of
# resolve_pair, used by the Gate 2 smoke test.


def _all_five_roots(source_roots):
    return {"pair_a_sft": source_roots["pair_a_sft"], "pair_a_dpo": source_roots["pair_a_dpo"], "pair_b_sft": source_roots["pair_b_sft"]}


def test_all_roles_is_the_frozen_five_role_order():
    assert eb.ALL_ROLES == ("M0-A", "M+-A", "M--A", "M0-B", "M+-B")


def test_resolve_all_roles_resolves_every_role_with_correct_status(tmp_path):
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)
    resolved = eb.resolve_all_roles(
        manifest_path, _all_five_roots(source_roots), bundle_root, source_manifest_path=source_manifest_path,
    )

    assert set(resolved.roles) == set(eb.ALL_ROLES)
    assert resolved.roles["M0-A"].status == "direct"
    assert resolved.roles["M+-A"].status == "direct"
    assert resolved.roles["M--A"].status == "merged"
    assert resolved.roles["M0-B"].status == "direct"
    assert resolved.roles["M+-B"].status == "merged"
    assert resolved.roles["M0-A"].local_path == source_roots["pair_a_sft"].resolve()
    assert resolved.roles["M--A"].local_path == (Path(bundle_root) / "M--A").resolve()


def test_resolve_all_roles_requires_exactly_the_three_direct_source_artifact_ids(tmp_path):
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)

    with pytest.raises(eb.EndpointBindingError, match="missing"):
        eb.resolve_all_roles(manifest_path, {}, bundle_root, source_manifest_path=source_manifest_path)

    incomplete = {"pair_a_sft": source_roots["pair_a_sft"]}
    with pytest.raises(eb.EndpointBindingError, match="missing"):
        eb.resolve_all_roles(manifest_path, incomplete, bundle_root, source_manifest_path=source_manifest_path)


def test_resolve_all_roles_rejects_an_unexpected_source_mapping(tmp_path):
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)
    extra = dict(_all_five_roots(source_roots))
    extra["pair_a_flip_adapter"] = bundle_root / "M--A"  # never a valid direct-source key
    with pytest.raises(eb.EndpointBindingError, match="unexpected"):
        eb.resolve_all_roles(manifest_path, extra, bundle_root, source_manifest_path=source_manifest_path)


def test_resolve_all_roles_also_binds_against_the_frozen_source_manifest(tmp_path):
    """resolve_all_roles must not skip the frozen-source cross-check that resolve_pair
    performs -- a correctly self-rehashed candidate with a tampered direct revision
    must still be rejected."""
    manifest_path, source_roots, bundle_root, _, source_manifest_path = build_candidate_env(tmp_path)
    _tamper_and_rehash(manifest_path, _tamper_direct_revision)

    with pytest.raises(eb.EndpointBindingError, match="location.revision"):
        eb.resolve_all_roles(manifest_path, _all_five_roots(source_roots), bundle_root, source_manifest_path=source_manifest_path)


def test_resolved_endpoint_set_run_metadata_covers_all_five_roles(tmp_path):
    manifest_path, source_roots, bundle_root, candidate, source_manifest_path = build_candidate_env(tmp_path)
    resolved = eb.resolve_all_roles(
        manifest_path, _all_five_roots(source_roots), bundle_root, source_manifest_path=source_manifest_path,
    )

    meta = resolved.run_metadata()
    assert meta["mode"] == "endpoint_all_roles"
    assert meta["candidate_manifest_hash"] == candidate["manifest_hash"]
    assert meta["roles"] == list(eb.ALL_ROLES)
    assert set(meta["endpoints"]) == set(eb.ALL_ROLES)
    for role in eb.ALL_ROLES:
        assert meta["endpoints"][role]["role"] == role
