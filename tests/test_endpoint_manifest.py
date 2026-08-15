"""Guards on the candidate inference-endpoint manifest (protocol Sections 4/13, Task 011).

Mirrors the model-artifact and judge-protocol manifest tests in shape: canonical-hash
order-independence, structural-invariant rejection, write-once save behavior, the frozen
role/status/artifact mapping, Pair A/Pair B matched-family consistency, and the pure
tokenizer-compatibility/vocab/tied-embedding checks used to accept or reject a resolved
endpoint. Entirely offline -- no network, no model, no GPU, no merge.

Run with:
    pytest tests/test_endpoint_manifest.py -v
"""

from __future__ import annotations

import copy
import json

import pytest

from steering.endpoint_manifest import (
    EXPECTED_ENDPOINT_ROLES,
    FROZEN_ENDPOINT_PLAN,
    EndpointManifestError,
    TokenizerCompatibilityError,
    build_manifest,
    check_tied_embedding_consistency,
    check_tokenizer_extension,
    check_vocab_consistency,
    compute_manifest_hash,
    load_manifest,
    save_manifest,
    tokenizer_fingerprint,
    validate_manifest_structure,
)


def _hex64(fill: str = "a") -> str:
    return fill * 64


def _hex40(fill: str = "2") -> str:
    return fill * 40


SOURCE_HASH = _hex64("2")

_A_ARCH = {"model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 128256, "embedding_rows": 128256, "lm_head_rows": 128256}
_B_ARCH = {"model_type": "mistral", "hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 32002, "embedding_rows": 32002, "lm_head_rows": 32002}

_A_TOK_FP = _hex64("a")
_A_SPECIAL = {"bos_token_id": 1, "eos_token_id": 2}
_B_TOK_FP = _hex64("b")
_B_SPECIAL = {"bos_token_id": 1, "eos_token_id": 2}

_M0A_LOCATION = {"kind": "source", "repository": "sirius5005/SFT-and-DPO", "revision": _hex40("2"), "subpath": "SFT_merged"}
_MPA_LOCATION = {"kind": "source", "repository": "sirius5005/SFT-and-DPO", "revision": _hex40("2"), "subpath": "DPO_merged"}
_M0B_LOCATION = {"kind": "source", "repository": "teknium/OpenHermes-2.5-Mistral-7B", "revision": _hex40("3"), "subpath": ""}
_MMA_LOCATION = {"kind": "bundle", "path": "M--A"}
_MPB_LOCATION = {"kind": "bundle", "path": "M+-B"}


def _validation(arch, tied=False, tokenizer_fp=_A_TOK_FP, special_token_ids=None):
    if special_token_ids is None:
        special_token_ids = _A_SPECIAL
    return {
        **arch, "tied_embeddings": tied, "tokenizer_loadable": True, "no_residual_peft_modules": True,
        "forward_pass_smoke_test": True, "tokenizer_fingerprint": tokenizer_fp, "special_token_ids": special_token_ids,
    }


def _files(fill: str = "1"):
    return [{"path": "model.safetensors", "sha256": _hex64(fill), "size_bytes": 123}]


def _adapter_input_files(fill: str = "d"):
    return [
        {"path": "adapter_config.json", "sha256": _hex64(fill), "size_bytes": 111},
        {"path": "adapter_model.safetensors", "sha256": _hex64(fill), "size_bytes": 222},
        {"path": "training_args.bin", "sha256": _hex64(fill), "size_bytes": 333},
    ]


def _tokenizer_resolution_base_only():
    return {
        "status": "base_only", "base_vocab_size": 128256, "adapter_vocab_size": None,
        "old_vocab_size": None, "new_vocab_size": None, "added_token_ids": [],
        "base_fingerprint": _hex64("f"), "adapter_fingerprint": None,
        "base_special_token_ids": {"bos_token_id": 1}, "adapter_special_token_ids": None,
    }


def _flip_lineage(**overrides):
    base = {
        "archived_trainer_state_path": "runs/x/checkpoint-2510/trainer_state.json",
        "sha256": _hex64("9"), "training_step": 2510,
        "training_script": None, "launch_config": None,
        "label_swap_lineage_verified": False, "confirmatory_eligible": False,
    }
    base.update(overrides)
    return base


def _synthetic_endpoints(overrides=None):
    """Five structurally valid, family-consistent endpoints. Independent of any real
    source manifest, so structural/order-independence tests don't depend on real hashes."""
    base = {
        "M0-A": {
            "role": "M0-A", "status": "direct", "source_manifest_hash": SOURCE_HASH,
            "source_artifact_id": "pair_a_sft", "base_artifact_id": None,
            "location": dict(_M0A_LOCATION), "merge": None, "device": "cuda:0",
            "library_versions": {"torch": "2.0.0", "transformers": "4.49.0"},
            "validation": _validation(_A_ARCH), "files": _files("1"),
        },
        "M+-A": {
            "role": "M+-A", "status": "direct", "source_manifest_hash": SOURCE_HASH,
            "source_artifact_id": "pair_a_dpo", "base_artifact_id": None,
            "location": dict(_MPA_LOCATION), "merge": None, "device": "cuda:0",
            "library_versions": {"torch": "2.0.0", "transformers": "4.49.0"},
            "validation": _validation(_A_ARCH), "files": _files("2"),
        },
        "M--A": {
            "role": "M--A", "status": "merged", "source_manifest_hash": SOURCE_HASH,
            "source_artifact_id": "pair_a_flip_adapter", "base_artifact_id": "pair_a_sft",
            "location": dict(_MMA_LOCATION), "device": "cuda:0",
            "merge": {
                "dtype": "bfloat16", "max_shard_size": "5GB", "merge_seed": 20260815,
                "tokenizer_resolution": _tokenizer_resolution_base_only(),
                "adapter_input_files": _adapter_input_files("3"),
                "flip_lineage": _flip_lineage(),
            },
            "library_versions": {"torch": "2.0.0", "transformers": "4.49.0", "peft": "0.11.0"},
            "validation": _validation(_A_ARCH), "files": _files("3"),
        },
        "M0-B": {
            "role": "M0-B", "status": "direct", "source_manifest_hash": SOURCE_HASH,
            "source_artifact_id": "pair_b_sft", "base_artifact_id": None,
            "location": dict(_M0B_LOCATION), "merge": None, "device": "cuda:0",
            "library_versions": {"torch": "2.0.0", "transformers": "4.49.0"},
            "validation": _validation(_B_ARCH, tokenizer_fp=_B_TOK_FP, special_token_ids=_B_SPECIAL), "files": _files("4"),
        },
        "M+-B": {
            "role": "M+-B", "status": "merged", "source_manifest_hash": SOURCE_HASH,
            "source_artifact_id": "pair_b_dpo_adapter", "base_artifact_id": "pair_b_sft",
            "location": dict(_MPB_LOCATION), "device": "cuda:0",
            "merge": {
                "dtype": "bfloat16", "max_shard_size": "5GB", "merge_seed": 20260815,
                "tokenizer_resolution": {
                    "status": "base_only", "base_vocab_size": 32002, "adapter_vocab_size": None,
                    "old_vocab_size": None, "new_vocab_size": None, "added_token_ids": [],
                    "base_fingerprint": _hex64("e"), "adapter_fingerprint": None,
                    "base_special_token_ids": {"bos_token_id": 1}, "adapter_special_token_ids": None,
                },
                "adapter_input_files": _adapter_input_files("5"),
            },
            "library_versions": {"torch": "2.0.0", "transformers": "4.49.0", "peft": "0.11.0"},
            "validation": _validation(_B_ARCH, tokenizer_fp=_B_TOK_FP, special_token_ids=_B_SPECIAL), "files": _files("5"),
        },
    }
    if overrides:
        for role, patch in overrides.items():
            base[role] = {**base[role], **patch}
    return list(base.values())


def _build(endpoints=None):
    return build_manifest(SOURCE_HASH, endpoints if endpoints is not None else _synthetic_endpoints())


# Canonical hash order-independence


def test_hash_is_independent_of_endpoint_list_order():
    forward = _synthetic_endpoints()
    backward = list(reversed(forward))
    a = build_manifest(SOURCE_HASH, forward)
    b = build_manifest(SOURCE_HASH, backward)
    assert a == b
    assert a["manifest_hash"] == b["manifest_hash"]


def test_hash_is_independent_of_per_endpoint_key_order():
    endpoints = _synthetic_endpoints()
    reordered = [dict(reversed(list(e.items()))) for e in endpoints]
    a = build_manifest(SOURCE_HASH, endpoints)
    b = build_manifest(SOURCE_HASH, reordered)
    assert a["manifest_hash"] == b["manifest_hash"]


def test_hash_is_independent_of_file_list_order_within_an_endpoint():
    endpoints = _synthetic_endpoints(overrides={
        "M0-A": {"files": [
            {"path": "model.safetensors", "sha256": _hex64("1"), "size_bytes": 1},
            {"path": "tokenizer.json", "sha256": _hex64("9"), "size_bytes": 2},
        ]},
    })
    swapped = copy.deepcopy(endpoints)
    for e in swapped:
        if e["role"] == "M0-A":
            e["files"] = list(reversed(e["files"]))
    x = build_manifest(SOURCE_HASH, endpoints)
    y = build_manifest(SOURCE_HASH, swapped)
    assert x["manifest_hash"] == y["manifest_hash"]


def test_same_inputs_reproduce_the_same_hash():
    a = _build()
    b = _build()
    assert a["manifest_hash"] == b["manifest_hash"]


def test_manifest_hash_matches_compute_manifest_hash():
    m = _build()
    assert m["manifest_hash"] == compute_manifest_hash(m)


def test_changing_a_file_hash_changes_the_manifest_hash():
    a = _build()
    b = _build(_synthetic_endpoints(overrides={"M0-A": {"files": _files("f")}}))
    assert a["manifest_hash"] != b["manifest_hash"]


def test_top_level_source_manifest_hash_is_stamped_onto_every_endpoint_by_default():
    endpoints = _synthetic_endpoints()
    for e in endpoints:
        del e["source_manifest_hash"]
    m = build_manifest(_hex64("7"), endpoints)
    assert m["source_manifest_hash"] == _hex64("7")
    assert all(e["source_manifest_hash"] == _hex64("7") for e in m["endpoints"])


# Frozen role/status/source-artifact/base-artifact mapping


def test_frozen_endpoint_plan_matches_the_synthetic_fixture():
    for e in _synthetic_endpoints():
        frozen = FROZEN_ENDPOINT_PLAN[e["role"]]
        assert e["status"] == frozen["status"]
        assert e["source_artifact_id"] == frozen["source_artifact_id"]
        assert e["base_artifact_id"] == frozen["base_artifact_id"]


def test_a_wrong_source_artifact_id_for_a_role_is_rejected():
    bad = _synthetic_endpoints(overrides={"M0-A": {"source_artifact_id": "pair_a_dpo"}})
    with pytest.raises(EndpointManifestError):
        _build(bad)


def test_a_wrong_base_artifact_id_for_a_role_is_rejected():
    bad = _synthetic_endpoints(overrides={"M--A": {"base_artifact_id": "pair_b_sft"}})
    with pytest.raises(EndpointManifestError):
        _build(bad)


def test_a_wrong_status_for_a_role_is_rejected():
    bad = copy.deepcopy(_synthetic_endpoints())
    for e in bad:
        if e["role"] == "M0-A":
            e["status"] = "merged"
            e["base_artifact_id"] = "pair_a_dpo"
            e["merge"] = {"dtype": "bfloat16", "max_shard_size": "5GB", "merge_seed": 1, "tokenizer_resolution": _tokenizer_resolution_base_only()}
    with pytest.raises(EndpointManifestError):
        _build(bad)


# Top-level source_manifest_hash consistency


def test_an_endpoint_with_a_mismatched_source_manifest_hash_is_rejected():
    bad = _synthetic_endpoints(overrides={"M0-A": {"source_manifest_hash": _hex64("9")}})
    with pytest.raises(EndpointManifestError):
        _build(bad)


# Structural invariants


def test_missing_expected_role_is_rejected():
    endpoints = [e for e in _synthetic_endpoints() if e["role"] != "M0-B"]
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_duplicate_roles_are_rejected():
    endpoints = _synthetic_endpoints()
    duplicate = dict(endpoints[0])
    with pytest.raises(EndpointManifestError):
        _build(endpoints + [duplicate])


def test_a_direct_endpoint_with_a_base_artifact_id_is_rejected():
    bad = copy.deepcopy(_synthetic_endpoints())
    for e in bad:
        if e["role"] == "M0-A":
            e["base_artifact_id"] = "pair_a_dpo"
    with pytest.raises(EndpointManifestError):
        _build(bad)


def test_a_direct_endpoint_with_merge_details_is_rejected():
    endpoints = _synthetic_endpoints(overrides={
        "M0-A": {"merge": {"dtype": "bfloat16", "max_shard_size": "5GB", "merge_seed": 1, "tokenizer_resolution": _tokenizer_resolution_base_only()}},
    })
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_a_merged_endpoint_without_merge_details_is_rejected():
    endpoints = _synthetic_endpoints(overrides={"M--A": {"merge": None}})
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_a_merged_endpoint_missing_merge_seed_is_rejected():
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M+-B":
            del e["merge"]["merge_seed"]
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


# Tokenizer-resolution status-specific validation -- each status may only carry the
# exact fields consistent with what it claims, not merely the right shape.


def _tr_override(role, **fields):
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == role:
            e["merge"]["tokenizer_resolution"] = fields
    return endpoints


def test_base_only_with_a_non_null_adapter_vocab_size_is_rejected():
    fields = dict(_tokenizer_resolution_base_only())
    fields["adapter_vocab_size"] = 128260  # must be null for base_only
    with pytest.raises(EndpointManifestError):
        _build(_tr_override("M--A", **fields))


def test_base_only_with_a_nonempty_added_token_ids_is_rejected():
    fields = dict(_tokenizer_resolution_base_only())
    fields["added_token_ids"] = [128256]
    with pytest.raises(EndpointManifestError):
        _build(_tr_override("M--A", **fields))


def test_base_only_missing_base_fingerprint_is_rejected():
    fields = dict(_tokenizer_resolution_base_only())
    fields["base_fingerprint"] = None
    with pytest.raises(EndpointManifestError):
        _build(_tr_override("M--A", **fields))


def _identical_tr(fp=None, special=None):
    fp = fp or _hex64("c")
    special = special if special is not None else {"bos_token_id": 1}
    return {
        "status": "identical", "base_vocab_size": 128256, "adapter_vocab_size": 128256,
        "old_vocab_size": 128256, "new_vocab_size": 128256, "added_token_ids": [],
        "base_fingerprint": fp, "adapter_fingerprint": fp,
        "base_special_token_ids": special, "adapter_special_token_ids": special,
    }


def test_identical_status_with_matching_fields_is_accepted():
    m = _build(_tr_override("M--A", **_identical_tr()))
    assert m["manifest_hash"]


def test_identical_status_with_a_size_mismatch_is_rejected():
    fields = _identical_tr()
    fields["adapter_vocab_size"] = 128260
    with pytest.raises(EndpointManifestError):
        _build(_tr_override("M--A", **fields))


def test_identical_status_with_mismatched_fingerprints_is_rejected():
    fields = _identical_tr()
    fields["adapter_fingerprint"] = _hex64("9")
    with pytest.raises(EndpointManifestError):
        _build(_tr_override("M--A", **fields))


def test_identical_status_with_mismatched_special_token_ids_is_rejected():
    fields = _identical_tr()
    fields["adapter_special_token_ids"] = {"bos_token_id": 99}
    with pytest.raises(EndpointManifestError):
        _build(_tr_override("M--A", **fields))


def test_identical_status_with_a_nonempty_added_token_ids_is_rejected():
    fields = _identical_tr()
    fields["added_token_ids"] = [128256]
    with pytest.raises(EndpointManifestError):
        _build(_tr_override("M--A", **fields))


def _extension_tr(old=128256, new=128260, added=None, base_fp=None, adapter_fp=None):
    added = added if added is not None else list(range(old, new))
    return {
        "status": "append_only_extension", "base_vocab_size": old, "adapter_vocab_size": new,
        "old_vocab_size": old, "new_vocab_size": new, "added_token_ids": added,
        "base_fingerprint": base_fp or _hex64("e"), "adapter_fingerprint": adapter_fp or _hex64("d"),
        "base_special_token_ids": {"bos_token_id": 1}, "adapter_special_token_ids": {"bos_token_id": 1, "new_id": 128256},
    }


def test_append_only_extension_with_non_increasing_size_is_rejected():
    fields = _extension_tr(old=128256, new=128256, added=[])
    with pytest.raises(EndpointManifestError):
        _build(_tr_override("M--A", **fields))


def test_append_only_extension_with_a_non_contiguous_added_range_is_rejected():
    fields = _extension_tr(old=128256, new=128260, added=[128256, 128257, 128259])  # gap
    with pytest.raises(EndpointManifestError):
        _build(_tr_override("M--A", **fields))


def test_append_only_extension_with_base_vocab_size_not_matching_old_is_rejected():
    fields = _extension_tr()
    fields["base_vocab_size"] = 999
    with pytest.raises(EndpointManifestError):
        _build(_tr_override("M--A", **fields))


def test_append_only_extension_missing_adapter_fingerprint_is_rejected():
    fields = _extension_tr()
    fields["adapter_fingerprint"] = None
    with pytest.raises(EndpointManifestError):
        _build(_tr_override("M--A", **fields))


def test_append_only_extension_missing_special_token_ids_is_rejected():
    fields = _extension_tr()
    fields["adapter_special_token_ids"] = None
    with pytest.raises(EndpointManifestError):
        _build(_tr_override("M--A", **fields))


# Merge-critical adapter input binding -- every regular file under the verified
# adapter root, not only the two source-manifest-pinned anchor files.


def test_a_merged_endpoint_missing_adapter_input_files_is_rejected():
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M--A":
            del e["merge"]["adapter_input_files"]
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_a_merged_endpoint_with_empty_adapter_input_files_is_rejected():
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M+-B":
            e["merge"]["adapter_input_files"] = []
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_a_merged_endpoint_adapter_input_files_with_a_bad_hash_is_rejected():
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M--A":
            e["merge"]["adapter_input_files"] = [{"path": "adapter_config.json", "sha256": "not-hex", "size_bytes": 1}]
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_adapter_input_files_order_does_not_affect_the_manifest_hash():
    endpoints = _synthetic_endpoints()
    swapped = copy.deepcopy(endpoints)
    for e in swapped:
        if e["role"] == "M--A":
            e["merge"]["adapter_input_files"] = list(reversed(e["merge"]["adapter_input_files"]))
    x = build_manifest(SOURCE_HASH, endpoints)
    y = build_manifest(SOURCE_HASH, swapped)
    assert x["manifest_hash"] == y["manifest_hash"]


# Endpoint location semantics (direct = source locator, merged = bundle path)


def test_location_must_be_a_dict():
    endpoints = _synthetic_endpoints(overrides={"M0-A": {"location": "M0-A"}})
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_a_direct_endpoint_location_must_have_kind_source():
    """A direct endpoint's weights are never copied into the bundle, so a bundle-style
    location for it is not merely wrong shape, it is a false claim about where the
    file lives."""
    endpoints = _synthetic_endpoints(overrides={"M0-A": {"location": dict(_MMA_LOCATION)}})
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_a_merged_endpoint_location_must_have_kind_bundle():
    endpoints = _synthetic_endpoints(overrides={"M--A": {"location": dict(_M0A_LOCATION)}})
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_a_direct_endpoint_location_missing_repository_is_rejected():
    bad = {**_M0A_LOCATION, "repository": ""}
    endpoints = _synthetic_endpoints(overrides={"M0-A": {"location": bad}})
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_a_direct_endpoint_location_revision_must_be_40_hex():
    bad = {**_M0A_LOCATION, "revision": "not-a-revision"}
    endpoints = _synthetic_endpoints(overrides={"M0-A": {"location": bad}})
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_a_direct_endpoint_location_subpath_must_be_relative():
    bad = {**_M0A_LOCATION, "subpath": "/abs/path"}
    endpoints = _synthetic_endpoints(overrides={"M0-A": {"location": bad}})
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_a_direct_endpoint_location_subpath_rejects_a_windows_path():
    bad = {**_M0A_LOCATION, "subpath": "C:\\Users\\me\\model"}
    endpoints = _synthetic_endpoints(overrides={"M0-A": {"location": bad}})
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_a_direct_endpoint_location_subpath_rejects_traversal():
    bad = {**_M0A_LOCATION, "subpath": "../../etc/passwd"}
    endpoints = _synthetic_endpoints(overrides={"M0-A": {"location": bad}})
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_a_direct_endpoint_location_with_an_empty_subpath_is_accepted():
    """Some source artifacts (e.g. a repository root) declare subpath=''."""
    endpoints = _synthetic_endpoints(overrides={"M0-A": {"location": {**_M0A_LOCATION, "subpath": ""}}})
    m = _build(endpoints)
    assert m["manifest_hash"]


def test_a_direct_endpoint_location_with_unexpected_keys_is_rejected():
    bad = {**_M0A_LOCATION, "extra": "nope"}
    endpoints = _synthetic_endpoints(overrides={"M0-A": {"location": bad}})
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_a_merged_endpoint_location_path_must_be_relative():
    endpoints = _synthetic_endpoints(overrides={"M--A": {"location": {"kind": "bundle", "path": "/abs/M--A"}}})
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_a_merged_endpoint_location_with_unexpected_keys_is_rejected():
    endpoints = _synthetic_endpoints(overrides={"M--A": {"location": {"kind": "bundle", "path": "M--A", "extra": 1}}})
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_a_merged_endpoint_location_path_must_equal_its_own_role():
    """Correction requirement 3: M--A's bundle path must be exactly "M--A", and
    M+-B's must be exactly "M+-B" -- a relative, well-formed path pointing at a
    *different* role's directory is still wrong."""
    endpoints = _synthetic_endpoints(overrides={"M--A": {"location": {"kind": "bundle", "path": "M+-B"}}})
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_m_plus_b_location_path_must_equal_its_own_role():
    endpoints = _synthetic_endpoints(overrides={"M+-B": {"location": {"kind": "bundle", "path": "M--A"}}})
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_a_missing_device_is_rejected():
    endpoints = _synthetic_endpoints(overrides={"M0-A": {"device": ""}})
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_m_minus_a_without_flip_lineage_is_rejected():
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M--A":
            del e["merge"]["flip_lineage"]
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_m_minus_a_flip_lineage_with_a_non_bool_confirmatory_eligible_is_rejected():
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M--A":
            e["merge"]["flip_lineage"]["confirmatory_eligible"] = "false"
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_m_minus_a_flip_lineage_confirmatory_eligible_true_is_rejected():
    """Correction requirement 4: schema_version 1 freezes both flip-lineage flags at
    exactly False -- the archived trainer-state checkpoint proves training reached the
    expected step, never that preference labels were actually swapped. A manifest
    cannot claim otherwise just by flipping the bit and rehashing."""
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M--A":
            e["merge"]["flip_lineage"]["confirmatory_eligible"] = True
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_m_minus_a_flip_lineage_label_swap_lineage_verified_true_is_rejected():
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M--A":
            e["merge"]["flip_lineage"]["label_swap_lineage_verified"] = True
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_load_manifest_rejects_a_manually_flipped_confirmatory_eligible_even_when_rehashed(tmp_path):
    """Regression for the exact threat named in the correction: manually editing
    confirmatory_eligible to True and recomputing manifest_hash so the file is
    internally self-consistent must still be rejected by structural validation."""
    payload = _build()
    tampered = copy.deepcopy(payload)
    for e in tampered["endpoints"]:
        if e["role"] == "M--A":
            e["merge"]["flip_lineage"]["confirmatory_eligible"] = True
    tampered["manifest_hash"] = compute_manifest_hash(tampered)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(EndpointManifestError):
        load_manifest(path)


def test_validation_missing_keys_are_rejected():
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M0-A":
            del e["validation"]["forward_pass_smoke_test"]
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


@pytest.mark.parametrize("key", ["tokenizer_loadable", "no_residual_peft_modules", "forward_pass_smoke_test"])
def test_a_false_required_success_field_is_rejected(key):
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M0-A":
            e["validation"][key] = False
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


@pytest.mark.parametrize("key", ["hidden_size", "num_hidden_layers", "vocab_size", "embedding_rows", "lm_head_rows"])
def test_a_nonpositive_architecture_value_is_rejected(key):
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M0-A":
            e["validation"][key] = 0
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_a_null_model_type_is_rejected():
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M0-A":
            e["validation"]["model_type"] = None
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_a_malformed_tokenizer_fingerprint_is_rejected():
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M0-A":
            e["validation"]["tokenizer_fingerprint"] = "not-hex"
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_a_non_dict_special_token_ids_is_rejected():
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M0-A":
            e["validation"]["special_token_ids"] = ["bos_token_id"]
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_empty_library_versions_is_rejected():
    endpoints = _synthetic_endpoints(overrides={"M0-A": {"library_versions": {}}})
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_an_absolute_or_traversal_file_path_is_rejected():
    bad = _synthetic_endpoints(overrides={
        "M0-A": {"files": [{"path": "../../etc/passwd", "sha256": _hex64("1"), "size_bytes": 1}]},
    })
    with pytest.raises(EndpointManifestError):
        _build(bad)


def test_a_duplicate_file_path_within_one_endpoint_is_rejected():
    bad = _synthetic_endpoints(overrides={
        "M0-A": {"files": [
            {"path": "model.safetensors", "sha256": _hex64("1"), "size_bytes": 1},
            {"path": "model.safetensors", "sha256": _hex64("2"), "size_bytes": 2},
        ]},
    })
    with pytest.raises(EndpointManifestError):
        _build(bad)


def test_a_negative_size_bytes_is_rejected():
    bad = _synthetic_endpoints(overrides={
        "M0-A": {"files": [{"path": "model.safetensors", "sha256": _hex64("1"), "size_bytes": -1}]},
    })
    with pytest.raises(EndpointManifestError):
        _build(bad)


def test_validate_manifest_structure_accepts_a_well_formed_manifest():
    m = _build()
    validate_manifest_structure(m)  # must not raise


# Pair A / Pair B matched-family consistency


def test_pair_a_hidden_size_mismatch_is_rejected():
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M--A":
            e["validation"]["hidden_size"] = 999
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_pair_a_model_type_mismatch_is_rejected():
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M+-A":
            e["validation"]["model_type"] = "mistral"
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_pair_a_vocab_mismatch_is_rejected_with_no_extension_exception():
    """Unlike Pair B, Pair A permits no tokenizer-extension exception at all."""
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M--A":
            e["validation"]["vocab_size"] = _A_ARCH["vocab_size"] + 4
            e["validation"]["embedding_rows"] = _A_ARCH["embedding_rows"] + 4
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_pair_b_layer_count_mismatch_is_rejected():
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M+-B":
            e["validation"]["num_hidden_layers"] = 16
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_pair_b_vocab_mismatch_without_a_recorded_extension_is_rejected():
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M+-B":
            e["validation"]["vocab_size"] = _B_ARCH["vocab_size"] + 4
            e["validation"]["embedding_rows"] = _B_ARCH["embedding_rows"] + 4
            # tokenizer_resolution left as "base_only" -- no extension recorded.
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_pair_b_vocab_mismatch_with_a_matching_verified_extension_is_accepted():
    """A verified append-only extension legitimately changes both vocab size and the
    endpoint's own tokenizer_fingerprint (it hashes the whole vocabulary) -- both must
    differ here for the fixture to actually exercise the exception path. The recorded
    resolution's base_fingerprint/base_special_token_ids must equal M0-B's own
    validated facts, and its adapter_fingerprint/adapter_special_token_ids must equal
    M+-B's -- matching old/new sizes alone is not sufficient."""
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M+-B":
            e["validation"]["vocab_size"] = _B_ARCH["vocab_size"] + 4
            e["validation"]["embedding_rows"] = _B_ARCH["embedding_rows"] + 4
            e["validation"]["lm_head_rows"] = _B_ARCH["lm_head_rows"] + 4
            e["validation"]["tokenizer_fingerprint"] = _hex64("f")
            e["merge"]["tokenizer_resolution"] = {
                "status": "append_only_extension",
                "base_vocab_size": _B_ARCH["vocab_size"], "adapter_vocab_size": _B_ARCH["vocab_size"] + 4,
                "old_vocab_size": _B_ARCH["vocab_size"], "new_vocab_size": _B_ARCH["vocab_size"] + 4,
                "added_token_ids": list(range(_B_ARCH["vocab_size"], _B_ARCH["vocab_size"] + 4)),
                "base_fingerprint": _B_TOK_FP, "adapter_fingerprint": _hex64("f"),
                "base_special_token_ids": _B_SPECIAL, "adapter_special_token_ids": _B_SPECIAL,
            }
    m = _build(endpoints)  # must not raise
    assert m["manifest_hash"]


def test_pair_b_extension_with_matching_sizes_but_wrong_base_fingerprint_is_rejected():
    """Correction requirement 2: matching old/new sizes is not sufficient on its own --
    the recorded base_fingerprint must equal M0-B's own validated tokenizer_fingerprint."""
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M+-B":
            e["validation"]["vocab_size"] = _B_ARCH["vocab_size"] + 4
            e["validation"]["embedding_rows"] = _B_ARCH["embedding_rows"] + 4
            e["validation"]["lm_head_rows"] = _B_ARCH["lm_head_rows"] + 4
            e["validation"]["tokenizer_fingerprint"] = _hex64("f")
            e["merge"]["tokenizer_resolution"] = {
                "status": "append_only_extension",
                "base_vocab_size": _B_ARCH["vocab_size"], "adapter_vocab_size": _B_ARCH["vocab_size"] + 4,
                "old_vocab_size": _B_ARCH["vocab_size"], "new_vocab_size": _B_ARCH["vocab_size"] + 4,
                "added_token_ids": list(range(_B_ARCH["vocab_size"], _B_ARCH["vocab_size"] + 4)),
                "base_fingerprint": _hex64("9"), "adapter_fingerprint": _hex64("f"),  # wrong base_fingerprint
                "base_special_token_ids": _B_SPECIAL, "adapter_special_token_ids": _B_SPECIAL,
            }
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_pair_b_extension_with_matching_sizes_but_wrong_adapter_special_token_ids_is_rejected():
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M+-B":
            e["validation"]["vocab_size"] = _B_ARCH["vocab_size"] + 4
            e["validation"]["embedding_rows"] = _B_ARCH["embedding_rows"] + 4
            e["validation"]["lm_head_rows"] = _B_ARCH["lm_head_rows"] + 4
            e["validation"]["tokenizer_fingerprint"] = _hex64("f")
            e["merge"]["tokenizer_resolution"] = {
                "status": "append_only_extension",
                "base_vocab_size": _B_ARCH["vocab_size"], "adapter_vocab_size": _B_ARCH["vocab_size"] + 4,
                "old_vocab_size": _B_ARCH["vocab_size"], "new_vocab_size": _B_ARCH["vocab_size"] + 4,
                "added_token_ids": list(range(_B_ARCH["vocab_size"], _B_ARCH["vocab_size"] + 4)),
                "base_fingerprint": _B_TOK_FP, "adapter_fingerprint": _hex64("f"),
                "base_special_token_ids": _B_SPECIAL,
                "adapter_special_token_ids": {"bos_token_id": 1, "eos_token_id": 999},  # wrong
            }
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_pair_a_lm_head_rows_mismatch_is_rejected():
    """lm_head_rows is compared in addition to vocab_size/embedding_rows -- a model
    with an untied, independently-sized LM head must not slip past family consistency."""
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M--A":
            e["validation"]["lm_head_rows"] = _A_ARCH["lm_head_rows"] + 4
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_pair_a_tokenizer_fingerprint_mismatch_is_rejected():
    """Dimensions can agree while the actual vocabulary content differs -- the
    fingerprint check is what catches that, and Pair A permits no exception for it."""
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M+-A":
            e["validation"]["tokenizer_fingerprint"] = _hex64("9")
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_pair_a_special_token_ids_mismatch_is_rejected():
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M--A":
            e["validation"]["special_token_ids"] = {"bos_token_id": 1, "eos_token_id": 999}
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_pair_b_tokenizer_fingerprint_mismatch_without_a_recorded_extension_is_rejected():
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M+-B":
            e["validation"]["tokenizer_fingerprint"] = _hex64("9")
            # tokenizer_resolution left as "base_only" -- no extension recorded.
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_pair_b_special_token_ids_mismatch_without_a_recorded_extension_is_rejected():
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M+-B":
            e["validation"]["special_token_ids"] = {"bos_token_id": 1, "eos_token_id": 999}
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


def test_pair_b_vocab_mismatch_with_a_mismatched_extension_size_is_rejected():
    """The extension's own recorded old/new sizes must match the actual facts, not just
    claim status='append_only_extension'."""
    endpoints = copy.deepcopy(_synthetic_endpoints())
    for e in endpoints:
        if e["role"] == "M+-B":
            e["validation"]["vocab_size"] = _B_ARCH["vocab_size"] + 4
            e["validation"]["embedding_rows"] = _B_ARCH["embedding_rows"] + 4
            e["merge"]["tokenizer_resolution"] = {
                "status": "append_only_extension",
                "base_vocab_size": _B_ARCH["vocab_size"], "adapter_vocab_size": _B_ARCH["vocab_size"] + 999,
                "old_vocab_size": _B_ARCH["vocab_size"], "new_vocab_size": _B_ARCH["vocab_size"] + 999,  # wrong
                "added_token_ids": [], "base_fingerprint": _hex64("e"), "adapter_fingerprint": _hex64("d"),
                "base_special_token_ids": {}, "adapter_special_token_ids": {},
            }
    with pytest.raises(EndpointManifestError):
        _build(endpoints)


# save_manifest / load_manifest immutability


def test_an_existing_different_manifest_is_not_overwritten(tmp_path):
    original = _build()
    path = tmp_path / "candidate.json"
    save_manifest(original, path)
    on_disk_before = path.read_text(encoding="utf-8")

    different = _build(_synthetic_endpoints(overrides={"M0-A": {"files": _files("f")}}))
    with pytest.raises(EndpointManifestError):
        save_manifest(different, path)
    assert path.read_text(encoding="utf-8") == on_disk_before


def test_rebuilding_an_identical_manifest_does_not_rewrite(tmp_path):
    payload = _build()
    path = tmp_path / "candidate.json"
    save_manifest(payload, path)
    mtime_before = path.stat().st_mtime_ns
    written_again = save_manifest(payload, path)
    assert written_again is False
    assert path.stat().st_mtime_ns == mtime_before


def test_load_manifest_round_trips(tmp_path):
    payload = _build()
    path = tmp_path / "candidate.json"
    save_manifest(payload, path)
    loaded = load_manifest(path)
    assert loaded == payload


def test_load_manifest_rejects_a_stale_hash(tmp_path):
    payload = _build()
    tampered = copy.deepcopy(payload)
    tampered["endpoints"][0]["device"] = "cpu"  # manifest_hash left stale
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(EndpointManifestError):
        load_manifest(path)


def test_load_manifest_rejects_a_structurally_invalid_but_correctly_hashed_payload(tmp_path):
    """Regression: a stored hash matching its own content only proves the file was not
    corrupted or hand-edited after being written -- it says nothing about structural
    validity. The old `load_manifest` only checked the hash and returned this payload
    unmodified; it must now also run `validate_manifest_structure`."""
    payload = _build()
    tampered = copy.deepcopy(payload)
    del tampered["endpoints"][0]["validation"]["forward_pass_smoke_test"]
    tampered["manifest_hash"] = compute_manifest_hash(tampered)  # rehash so the stored hash is internally consistent
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(EndpointManifestError):
        load_manifest(path)


# Pure validation-fact helpers


def test_check_vocab_consistency_accepts_exact_match():
    assert check_vocab_consistency(vocab_size=100, embedding_rows=100, lm_head_rows=100) is True


def test_check_vocab_consistency_accepts_padded_rows():
    assert check_vocab_consistency(vocab_size=100, embedding_rows=128, lm_head_rows=128) is True


def test_check_vocab_consistency_rejects_insufficient_embedding_rows():
    assert check_vocab_consistency(vocab_size=100, embedding_rows=90, lm_head_rows=100) is False


def test_check_vocab_consistency_rejects_insufficient_lm_head_rows():
    assert check_vocab_consistency(vocab_size=100, embedding_rows=100, lm_head_rows=90) is False


def test_check_tied_embedding_consistency_not_applicable_when_untied():
    assert check_tied_embedding_consistency(tie_word_embeddings=False, embeddings_are_equal=False) is True


def test_check_tied_embedding_consistency_requires_equal_weights_when_tied():
    assert check_tied_embedding_consistency(tie_word_embeddings=True, embeddings_are_equal=False) is False
    assert check_tied_embedding_consistency(tie_word_embeddings=True, embeddings_are_equal=True) is True


# check_tokenizer_extension


def _vocab(n):
    return {f"tok{i}": i for i in range(n)}


def test_check_tokenizer_extension_identical_vocab():
    v = _vocab(10)
    result = check_tokenizer_extension(v, dict(v), {"bos_token_id": 0}, {"bos_token_id": 0})
    assert result == {"status": "identical", "old_vocab_size": 10, "new_vocab_size": 10, "added_token_ids": []}


def test_check_tokenizer_extension_append_only():
    base = _vocab(10)
    adapter = dict(base)
    adapter.update({"newtok0": 10, "newtok1": 11})
    result = check_tokenizer_extension(base, adapter, {}, {})
    assert result == {"status": "append_only_extension", "old_vocab_size": 10, "new_vocab_size": 12, "added_token_ids": [10, 11]}


def test_check_tokenizer_extension_rejects_shrinking():
    base, adapter = _vocab(10), _vocab(8)
    with pytest.raises(TokenizerCompatibilityError, match="smaller"):
        check_tokenizer_extension(base, adapter, {}, {})


def test_check_tokenizer_extension_rejects_a_reordered_shared_token():
    base = _vocab(10)
    adapter = dict(base)
    adapter["tok0"], adapter["tok1"] = adapter["tok1"], adapter["tok0"]  # ids swapped
    with pytest.raises(TokenizerCompatibilityError, match="changed id"):
        check_tokenizer_extension(base, adapter, {}, {})


def test_check_tokenizer_extension_rejects_a_missing_shared_token():
    base = _vocab(10)
    adapter = _vocab(10)
    del adapter["tok3"]
    adapter["othertok"] = 3
    with pytest.raises(TokenizerCompatibilityError, match="missing"):
        check_tokenizer_extension(base, adapter, {}, {})


def test_check_tokenizer_extension_rejects_a_non_contiguous_extension():
    base = _vocab(10)
    adapter = dict(base)
    adapter["gaptok"] = 15  # not 10, leaves a gap
    with pytest.raises(TokenizerCompatibilityError, match="contiguous"):
        check_tokenizer_extension(base, adapter, {}, {})


def test_check_tokenizer_extension_rejects_an_incompatible_special_token():
    base_special = {"bos_token_id": 1, "eos_token_id": 2}
    adapter_special = {"bos_token_id": 1, "eos_token_id": 99}
    with pytest.raises(TokenizerCompatibilityError, match="special token"):
        check_tokenizer_extension(_vocab(10), _vocab(10), base_special, adapter_special)


def test_check_tokenizer_extension_rejects_a_special_token_missing_from_the_adapter_dict():
    """Regression: a defined (non-null) base special token that has no key at all in the
    adapter dict must fail, not be silently treated as not applicable."""
    base_special = {"bos_token_id": 1, "eos_token_id": 2}
    adapter_special = {"bos_token_id": 1}  # eos_token_id key entirely absent
    with pytest.raises(TokenizerCompatibilityError, match="missing"):
        check_tokenizer_extension(_vocab(10), _vocab(10), base_special, adapter_special)


def test_check_tokenizer_extension_rejects_a_null_special_token_value_in_the_adapter():
    """Regression: same failure mode as above, but the adapter dict explicitly maps the
    key to None instead of omitting it. The previous `adapter_id is not None` guard let
    this through silently since a None value skipped the comparison entirely."""
    base_special = {"bos_token_id": 1, "eos_token_id": 2}
    adapter_special = {"bos_token_id": 1, "eos_token_id": None}
    with pytest.raises(TokenizerCompatibilityError, match="missing"):
        check_tokenizer_extension(_vocab(10), _vocab(10), base_special, adapter_special)


def test_check_tokenizer_extension_allows_a_new_special_token_only_in_the_extension():
    base = _vocab(10)
    adapter = dict(base)
    adapter["<new_special>"] = 10
    result = check_tokenizer_extension(base, adapter, {"bos_token_id": 0}, {"bos_token_id": 0, "new_special_id": 10})
    assert result["status"] == "append_only_extension"


def test_tokenizer_fingerprint_is_deterministic_and_content_sensitive():
    v1 = _vocab(5)
    v2 = _vocab(5)
    v3 = _vocab(6)
    assert tokenizer_fingerprint(v1) == tokenizer_fingerprint(v2)
    assert tokenizer_fingerprint(v1) != tokenizer_fingerprint(v3)


def test_tokenizer_fingerprint_is_independent_of_dict_insertion_order():
    v1 = _vocab(5)
    v2 = {k: v1[k] for k in reversed(list(v1))}
    assert tokenizer_fingerprint(v1) == tokenizer_fingerprint(v2)
