"""Candidate inference-endpoint manifest (protocol Sections 4 and 13, Gate 0).

Mirrors the shape of `steering.artifact_identity` and `steering.judge_identity` -- a
schema-versioned JSON payload with a canonical SHA-256 over itself and write-once save
behavior -- applied to the five *derived* inference endpoints (three used directly, two
produced by merging a declared adapter into a declared base) instead of source
artifacts or the judge protocol. Kept independent of `steering.artifact_identity` (no
cross-import) so this module stays self-contained; callers pass in already-verified
source-artifact facts rather than this module re-deriving them.

This module never loads, merges, or downloads a model. It only canonicalizes, hashes,
and structurally validates a payload the caller has already assembled from real
(or, in tests, fake) verification/merge/validation results.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

SCHEMA_VERSION = 1

EXPECTED_ENDPOINT_ROLES = ("M0-A", "M+-A", "M--A", "M0-B", "M+-B")

VALID_STATUSES = ("direct", "merged")

#: The one frozen role -> (status, source_artifact_id, base_artifact_id) mapping
#: (protocol Sections 4 & 13). Both `scripts/prepare_endpoints.py`'s own plan and this
#: module's structural validation are checked against this single authority, so a
#: manifest that is merely internally self-consistent but declares a different mapping
#: is rejected even though its shape is otherwise valid.
FROZEN_ENDPOINT_PLAN: Dict[str, Dict[str, Optional[str]]] = {
    "M0-A": {"status": "direct", "source_artifact_id": "pair_a_sft", "base_artifact_id": None},
    "M+-A": {"status": "direct", "source_artifact_id": "pair_a_dpo", "base_artifact_id": None},
    "M--A": {"status": "merged", "source_artifact_id": "pair_a_flip_adapter", "base_artifact_id": "pair_a_sft"},
    "M0-B": {"status": "direct", "source_artifact_id": "pair_b_sft", "base_artifact_id": None},
    "M+-B": {"status": "merged", "source_artifact_id": "pair_b_dpo_adapter", "base_artifact_id": "pair_b_sft"},
}

_PAIR_A_ROLES = ("M0-A", "M+-A", "M--A")
_PAIR_B_ROLES = ("M0-B", "M+-B")
_FAMILY_MATCH_FIELDS = ("model_type", "hidden_size", "num_hidden_layers")

_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED_VALIDATION_KEYS = (
    "model_type", "hidden_size", "num_hidden_layers", "vocab_size",
    "embedding_rows", "lm_head_rows", "tied_embeddings",
    "tokenizer_loadable", "no_residual_peft_modules", "forward_pass_smoke_test",
    "tokenizer_fingerprint", "special_token_ids",
)
_DIMENSION_MATCH_FIELDS = ("vocab_size", "embedding_rows", "lm_head_rows")
_REQUIRED_TRUE_VALIDATION_KEYS = ("tokenizer_loadable", "no_residual_peft_modules", "forward_pass_smoke_test")
_POSITIVE_INT_VALIDATION_KEYS = ("hidden_size", "num_hidden_layers", "vocab_size", "embedding_rows", "lm_head_rows")


class EndpointManifestError(ValueError):
    """A candidate-endpoint-manifest construction, structural-validation, or
    write-once invariant was violated."""


class TokenizerCompatibilityError(EndpointManifestError):
    """An adapter tokenizer is neither identical to, nor a pure append-only
    extension of, its base tokenizer."""


# Canonical serialization / manifest hash (identical convention to artifact_identity
# and judge_identity)


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    """The exact serialization the manifest hash is computed over. Compact,
    key-sorted, ASCII-only, with `manifest_hash` itself excluded."""
    body = {k: v for k, v in payload.items() if k != "manifest_hash"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def compute_manifest_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def dumps_manifest(payload: Mapping[str, Any]) -> str:
    """Human-readable on-disk form. Deterministic for a given payload, so repeated
    writes of the same content are byte-identical."""
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def save_manifest(payload: Mapping[str, Any], path: Union[str, Path]) -> bool:
    """Write `payload` to `path` if absent. If present, require byte-identical content.

    Returns True if bytes were written, False if an identical manifest already existed.
    Raises `EndpointManifestError` if an existing manifest differs -- a candidate
    manifest is never silently overwritten with different content.
    """
    path = Path(path)
    text = dumps_manifest(payload)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != text:
            raise EndpointManifestError(f"manifest at {path} already exists with different content; refusing to overwrite")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def load_manifest(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a manifest, verify its stored hash against a fresh recomputation, and run
    full structural validation.

    A stored hash matching its own content only proves the file was not corrupted or
    hand-edited after being written -- it says nothing about whether the payload is
    structurally valid (a payload can be internally self-consistent and still be
    missing a required key, describe the wrong frozen mapping, or fail matched-family
    consistency). Both checks are required to accept a manifest.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored = payload.get("manifest_hash")
    recomputed = compute_manifest_hash(payload)
    if stored != recomputed:
        raise EndpointManifestError(f"manifest hash mismatch at {path}: stored {stored!r}, recomputed {recomputed!r}")
    validate_manifest_structure(payload)
    return payload


# Structural validation


def _check_hex64(label: str, value: Any) -> None:
    if not isinstance(value, str) or not _HEX64_RE.match(value):
        raise EndpointManifestError(f"{label} must be 64 lowercase hex characters, got {value!r}")


def _check_nonempty_str(label: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise EndpointManifestError(f"{label} must be a nonempty string, got {value!r}")


def _check_positive_int(label: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EndpointManifestError(f"{label} must be a positive integer, got {value!r}")


def _check_relative_path(label: str, path: Any) -> None:
    """Same guard `steering.artifact_identity` applies to declared source-file paths,
    reused here for both file paths and a merged endpoint's bundle-relative location --
    no absolute path, no drive letter, no traversal.
    """
    if not isinstance(path, str) or not path or path != path.strip():
        raise EndpointManifestError(f"{label} must be a nonempty string, got {path!r}")
    normalized = path.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized) or ".." in normalized.split("/"):
        raise EndpointManifestError(f"{label} must be relative and traversal-free, got {path!r}")


def _validate_location(role: str, status: str, location: Any) -> None:
    """A direct endpoint is never copied or rewritten, so it has no bundle path at all
    -- its location is a stable *source locator* (repository/revision/subpath, exactly
    as verified against the frozen source manifest), never the operator's
    machine-specific local root. A merged endpoint's location is the bundle-relative
    directory it was actually written to (its role, e.g. "M--A")."""
    if not isinstance(location, dict):
        raise EndpointManifestError(f"{role}: location must be a dict")
    kind = location.get("kind")
    if status == "direct":
        if kind != "source":
            raise EndpointManifestError(f"{role}: direct endpoint location.kind must be 'source', got {kind!r}")
        _check_nonempty_str(f"{role}.location.repository", location.get("repository"))
        revision = location.get("revision")
        if not isinstance(revision, str) or not _HEX40_RE.match(revision):
            raise EndpointManifestError(f"{role}: location.revision must be 40 lowercase hex characters, got {revision!r}")
        subpath = location.get("subpath")
        if not isinstance(subpath, str):
            raise EndpointManifestError(f"{role}: location.subpath must be a string, got {subpath!r}")
        if subpath:  # some source artifacts (e.g. a repository root) declare "" -- valid
            _check_relative_path(f"{role}.location.subpath", subpath)
        extra = set(location) - {"kind", "repository", "revision", "subpath"}
        if extra:
            raise EndpointManifestError(f"{role}: location has unexpected keys {sorted(extra)}")
    else:
        if kind != "bundle":
            raise EndpointManifestError(f"{role}: merged endpoint location.kind must be 'bundle', got {kind!r}")
        _check_relative_path(f"{role}.location.path", location.get("path"))
        # The bundle directory a merged endpoint is written to is always its own role
        # (see prepare_endpoints._resolve_merged_endpoint) -- a location claiming a
        # different role's directory is never legitimate, whatever else it looks like.
        if location.get("path") != role:
            raise EndpointManifestError(
                f"{role}: merged endpoint location.path must equal its own role {role!r}, got {location.get('path')!r}"
            )
        extra = set(location) - {"kind", "path"}
        if extra:
            raise EndpointManifestError(f"{role}: location has unexpected keys {sorted(extra)}")


def _validate_files(label: str, files: Any) -> None:
    if not isinstance(files, list) or not files:
        raise EndpointManifestError(f"{label}: files must be a nonempty list")
    seen = set()
    for f in files:
        path = f.get("path")
        _check_relative_path(f"{label}.files[].path", path)
        if path in seen:
            raise EndpointManifestError(f"{label}: duplicate declared file path {path!r}")
        seen.add(path)
        _check_hex64(f"{label}.files[{path!r}].sha256", f.get("sha256"))
        size = f.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise EndpointManifestError(f"{label}.files[{path!r}].size_bytes must be a nonnegative integer, got {size!r}")


def _tr_check_hex64(label: str, tr: Mapping[str, Any], key: str) -> None:
    _check_hex64(f"{label}.merge.tokenizer_resolution.{key}", tr.get(key))


def _tr_check_dict(label: str, tr: Mapping[str, Any], key: str) -> None:
    if not isinstance(tr.get(key), dict):
        raise EndpointManifestError(f"{label}.merge.tokenizer_resolution.{key} must be a dict")


def _tr_check_null(label: str, tr: Mapping[str, Any], status: str, *keys: str) -> None:
    for key in keys:
        if tr.get(key) is not None:
            raise EndpointManifestError(
                f"{label}.merge.tokenizer_resolution.{key} must be null for status={status!r}, got {tr.get(key)!r}"
            )


def _validate_tokenizer_resolution(label: str, tr: Any) -> None:
    """Enforce the exact fields a given `status` may legitimately carry -- not just
    that the whole dict is shaped like *some* valid resolution. A resolution record
    that is internally self-consistent (e.g. right types) but describes a physically
    impossible or self-contradictory tokenizer relationship for its own claimed status
    must still be rejected.
    """
    if not isinstance(tr, dict):
        raise EndpointManifestError(f"{label}.merge.tokenizer_resolution must be a dict")
    status = tr.get("status")
    if status not in ("base_only", "identical", "append_only_extension"):
        raise EndpointManifestError(f"{label}.merge.tokenizer_resolution.status invalid: {status!r}")
    added = tr.get("added_token_ids")
    if not isinstance(added, list):
        raise EndpointManifestError(f"{label}.merge.tokenizer_resolution.added_token_ids must be a list")

    if status == "base_only":
        _check_positive_int(f"{label}.merge.tokenizer_resolution.base_vocab_size", tr.get("base_vocab_size"))
        _tr_check_hex64(label, tr, "base_fingerprint")
        _tr_check_dict(label, tr, "base_special_token_ids")
        _tr_check_null(label, tr, status, "adapter_vocab_size", "old_vocab_size", "new_vocab_size", "adapter_fingerprint", "adapter_special_token_ids")
        if added:
            raise EndpointManifestError(f"{label}.merge.tokenizer_resolution.added_token_ids must be empty for status='base_only'")

    elif status == "identical":
        for key in ("base_vocab_size", "adapter_vocab_size", "old_vocab_size", "new_vocab_size"):
            _check_positive_int(f"{label}.merge.tokenizer_resolution.{key}", tr.get(key))
        sizes = {tr.get(k) for k in ("base_vocab_size", "adapter_vocab_size", "old_vocab_size", "new_vocab_size")}
        if len(sizes) != 1:
            raise EndpointManifestError(
                f"{label}.merge.tokenizer_resolution: base/adapter/old/new vocabulary sizes must all match "
                f"for status='identical', got {tr!r}"
            )
        _tr_check_hex64(label, tr, "base_fingerprint")
        _tr_check_hex64(label, tr, "adapter_fingerprint")
        if tr["base_fingerprint"] != tr["adapter_fingerprint"]:
            raise EndpointManifestError(f"{label}.merge.tokenizer_resolution: base_fingerprint and adapter_fingerprint must match for status='identical'")
        _tr_check_dict(label, tr, "base_special_token_ids")
        _tr_check_dict(label, tr, "adapter_special_token_ids")
        if tr["base_special_token_ids"] != tr["adapter_special_token_ids"]:
            raise EndpointManifestError(f"{label}.merge.tokenizer_resolution: base_special_token_ids and adapter_special_token_ids must match for status='identical'")
        if added:
            raise EndpointManifestError(f"{label}.merge.tokenizer_resolution.added_token_ids must be empty for status='identical'")

    else:  # append_only_extension
        for key in ("base_vocab_size", "adapter_vocab_size", "old_vocab_size", "new_vocab_size"):
            _check_positive_int(f"{label}.merge.tokenizer_resolution.{key}", tr.get(key))
        old_size, new_size = tr["old_vocab_size"], tr["new_vocab_size"]
        if new_size <= old_size:
            raise EndpointManifestError(
                f"{label}.merge.tokenizer_resolution: new_vocab_size ({new_size}) must exceed "
                f"old_vocab_size ({old_size}) for status='append_only_extension'"
            )
        if tr["base_vocab_size"] != old_size or tr["adapter_vocab_size"] != new_size:
            raise EndpointManifestError(
                f"{label}.merge.tokenizer_resolution: base_vocab_size must equal old_vocab_size and "
                f"adapter_vocab_size must equal new_vocab_size for status='append_only_extension'"
            )
        if added != list(range(old_size, new_size)):
            raise EndpointManifestError(
                f"{label}.merge.tokenizer_resolution.added_token_ids must equal the exact contiguous range "
                f"[{old_size}, {new_size}), got {added!r}"
            )
        _tr_check_hex64(label, tr, "base_fingerprint")
        _tr_check_hex64(label, tr, "adapter_fingerprint")
        _tr_check_dict(label, tr, "base_special_token_ids")
        _tr_check_dict(label, tr, "adapter_special_token_ids")


def validate_manifest_structure(payload: Mapping[str, Any]) -> None:
    """Enforce the candidate endpoint manifest's structural invariants. Raises on the
    first violation found; never repairs, drops, or renumbers an endpoint or file.

    Also enforces the frozen role/status/source-artifact/base-artifact mapping and
    Pair A/Pair B matched-family consistency, so a manifest that is merely internally
    self-consistent but describes the wrong mapping or a family mismatch is rejected.
    """
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise EndpointManifestError(f"schema_version must be {SCHEMA_VERSION}, got {payload.get('schema_version')!r}")

    top_source_hash = payload.get("source_manifest_hash")
    _check_hex64("source_manifest_hash", top_source_hash)

    endpoints = payload.get("endpoints")
    if not isinstance(endpoints, list) or not endpoints:
        raise EndpointManifestError("payload['endpoints'] must be a nonempty list")

    roles = [e.get("role") for e in endpoints]
    if len(set(roles)) != len(roles):
        raise EndpointManifestError(f"duplicate endpoint role(s): {roles}")
    if set(roles) != set(EXPECTED_ENDPOINT_ROLES):
        raise EndpointManifestError(
            f"manifest must declare exactly {sorted(EXPECTED_ENDPOINT_ROLES)}, got {sorted(set(roles))}"
        )

    by_role: Dict[str, Mapping[str, Any]] = {}
    for e in endpoints:
        role = e.get("role")
        status = e.get("status")
        if status not in VALID_STATUSES:
            raise EndpointManifestError(f"{role}: status must be one of {VALID_STATUSES}, got {status!r}")

        # Frozen role -> (status, source_artifact_id, base_artifact_id) mapping.
        frozen = FROZEN_ENDPOINT_PLAN.get(role)
        if frozen is not None:
            if status != frozen["status"]:
                raise EndpointManifestError(f"{role}: status must be {frozen['status']!r}, got {status!r}")
            if e.get("source_artifact_id") != frozen["source_artifact_id"]:
                raise EndpointManifestError(
                    f"{role}: source_artifact_id must be {frozen['source_artifact_id']!r}, got {e.get('source_artifact_id')!r}"
                )
            if e.get("base_artifact_id") != frozen["base_artifact_id"]:
                raise EndpointManifestError(
                    f"{role}: base_artifact_id must be {frozen['base_artifact_id']!r}, got {e.get('base_artifact_id')!r}"
                )

        _check_hex64(f"{role}.source_manifest_hash", e.get("source_manifest_hash"))
        if e.get("source_manifest_hash") != top_source_hash:
            raise EndpointManifestError(
                f"{role}: source_manifest_hash {e.get('source_manifest_hash')!r} does not match the "
                f"top-level source_manifest_hash {top_source_hash!r}"
            )
        _check_nonempty_str(f"{role}.source_artifact_id", e.get("source_artifact_id"))
        _validate_location(role, status, e.get("location"))
        _check_nonempty_str(f"{role}.device", e.get("device"))

        base_artifact_id = e.get("base_artifact_id")
        merge = e.get("merge")
        if status == "direct":
            if base_artifact_id is not None:
                raise EndpointManifestError(f"{role}: a direct endpoint must not declare base_artifact_id")
            if merge is not None:
                raise EndpointManifestError(f"{role}: a direct endpoint must not declare merge details")
        else:
            _check_nonempty_str(f"{role}.base_artifact_id", base_artifact_id)
            if not isinstance(merge, dict):
                raise EndpointManifestError(f"{role}: a merged endpoint must declare merge details")
            _check_nonempty_str(f"{role}.merge.dtype", merge.get("dtype"))
            _check_nonempty_str(f"{role}.merge.max_shard_size", merge.get("max_shard_size"))
            _check_positive_int(f"{role}.merge.merge_seed", merge.get("merge_seed"))
            _validate_tokenizer_resolution(role, merge.get("tokenizer_resolution"))
            # Every merge-critical adapter input file, not only the source manifest's
            # anchor files -- the base's own inputs are already fully recorded under
            # the corresponding direct endpoint elsewhere in this same manifest.
            _validate_files(f"{role}.merge.adapter_input_files", merge.get("adapter_input_files"))
            if role == "M--A":
                flip = merge.get("flip_lineage")
                if not isinstance(flip, dict):
                    raise EndpointManifestError(f"{role}: merged endpoint M--A must declare merge.flip_lineage")
                # Frozen for schema_version 1: the archived trainer-state checkpoint proves
                # only that training reached the expected step, never that preference
                # labels were actually swapped. A manifest cannot claim otherwise merely by
                # setting the bit and rehashing -- resolving this requires new archived
                # evidence and a schema/protocol amendment, not a manifest edit.
                for key in ("label_swap_lineage_verified", "confirmatory_eligible"):
                    if flip.get(key) is not False:
                        raise EndpointManifestError(
                            f"{role}: merge.flip_lineage.{key} must be exactly False in schema_version "
                            f"{SCHEMA_VERSION} (label-swap lineage is not resolvable from archived "
                            f"material alone), got {flip.get(key)!r}"
                        )
                if "training_step" not in flip:
                    raise EndpointManifestError(f"{role}: merge.flip_lineage.training_step is required")

        validation = e.get("validation")
        if not isinstance(validation, dict):
            raise EndpointManifestError(f"{role}: validation must be a dict")
        missing = [k for k in _REQUIRED_VALIDATION_KEYS if k not in validation]
        if missing:
            raise EndpointManifestError(f"{role}: validation is missing {missing}")
        for key in _REQUIRED_TRUE_VALIDATION_KEYS:
            if validation.get(key) is not True:
                raise EndpointManifestError(f"{role}: validation.{key} must be True for an accepted endpoint, got {validation.get(key)!r}")
        for key in _POSITIVE_INT_VALIDATION_KEYS:
            _check_positive_int(f"{role}.validation.{key}", validation.get(key))
        if not isinstance(validation.get("tied_embeddings"), bool):
            raise EndpointManifestError(f"{role}: validation.tied_embeddings must be a bool")
        if not isinstance(validation.get("model_type"), str) or not validation["model_type"]:
            raise EndpointManifestError(f"{role}: validation.model_type must be a nonempty string")
        _check_hex64(f"{role}.validation.tokenizer_fingerprint", validation.get("tokenizer_fingerprint"))
        if not isinstance(validation.get("special_token_ids"), dict):
            raise EndpointManifestError(f"{role}: validation.special_token_ids must be a dict")

        library_versions = e.get("library_versions")
        if not isinstance(library_versions, dict) or not library_versions:
            raise EndpointManifestError(f"{role}: library_versions must be a nonempty dict")

        _validate_files(role, e.get("files"))
        by_role[role] = e

    _check_family_consistency(by_role)


def _check_family_consistency(by_role: Mapping[str, Mapping[str, Any]]) -> None:
    """M0-A/M+-A/M--A must agree exactly on model type, hidden size, layer count,
    vocabulary/embedding/lm-head dimensions, and tokenizer identity (full-vocabulary
    fingerprint plus special-token IDs) -- no exception. M0-B/M+-B must agree on the
    same facts unless the difference is exactly the recorded, verified append-only
    tokenizer extension."""

    def facts(role: str) -> Mapping[str, Any]:
        return by_role[role]["validation"]

    for roles, label in ((_PAIR_A_ROLES, "Pair A"), (_PAIR_B_ROLES, "Pair B")):
        for field in _FAMILY_MATCH_FIELDS:
            values = {facts(r)[field] for r in roles}
            if len(values) > 1:
                detail = ", ".join(f"{r}={facts(r)[field]!r}" for r in roles)
                raise EndpointManifestError(f"{label} endpoints disagree on {field}: {detail}")

    a_dims_differ = any(len({facts(r)[field] for r in _PAIR_A_ROLES}) > 1 for field in _DIMENSION_MATCH_FIELDS)
    a_fp_differ = len({facts(r)["tokenizer_fingerprint"] for r in _PAIR_A_ROLES}) > 1
    a_special_differ = len({_frozen_dict(facts(r)["special_token_ids"]) for r in _PAIR_A_ROLES}) > 1
    if a_dims_differ or a_fp_differ or a_special_differ:
        raise EndpointManifestError(
            "Pair A endpoints (M0-A, M+-A, M--A) must share identical vocabulary/embedding/"
            "lm_head dimensions and tokenizer identity (fingerprint, special-token IDs); "
            "no tokenizer extension is permitted for Pair A"
        )

    m0b, mpb_entry = facts("M0-B"), facts("M+-B")
    b_dims_differ = any(m0b[field] != mpb_entry[field] for field in _DIMENSION_MATCH_FIELDS)
    b_fp_differ = m0b["tokenizer_fingerprint"] != mpb_entry["tokenizer_fingerprint"]
    b_special_differ = m0b["special_token_ids"] != mpb_entry["special_token_ids"]
    if b_dims_differ or b_fp_differ or b_special_differ:
        tr = ((by_role["M+-B"].get("merge") or {}).get("tokenizer_resolution")) or {}
        # Matching old/new sizes alone is not enough: the recorded resolution must be
        # anchored to these exact two endpoints' own validated tokenizer identity, not
        # merely claim compatible sizes coincidentally.
        ok = (
            tr.get("status") == "append_only_extension"
            and tr.get("old_vocab_size") == m0b["vocab_size"]
            and tr.get("new_vocab_size") == mpb_entry["vocab_size"]
            and tr.get("base_fingerprint") == m0b["tokenizer_fingerprint"]
            and tr.get("base_special_token_ids") == m0b["special_token_ids"]
            and tr.get("adapter_fingerprint") == mpb_entry["tokenizer_fingerprint"]
            and tr.get("adapter_special_token_ids") == mpb_entry["special_token_ids"]
        )
        if not ok:
            raise EndpointManifestError(
                "M0-B and M+-B disagree on vocabulary/embedding/lm_head dimensions or "
                "tokenizer identity without a verified append-only tokenizer extension "
                "recording the exact old/new sizes and matching the endpoints' own "
                "validated tokenizer fingerprints and special-token IDs"
            )


def _frozen_dict(d: Mapping[str, Any]):
    return tuple(sorted(d.items()))


def build_manifest(
    source_manifest_hash: str, endpoints: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build the schema-versioned, hash-computed candidate endpoint manifest.

    Endpoints are sorted by role and each endpoint's files (and, for a merged endpoint,
    its adapter-input files) are sorted by path before hashing, so the result is
    byte-identical regardless of the order `endpoints` was supplied in.
    `source_manifest_hash` is the one top-level authority; it is stamped onto every
    endpoint entry that does not already carry one. Validates structure (including the
    frozen mapping and family consistency) and raises `EndpointManifestError` rather
    than silently accepting a malformed payload.
    """
    normalized = []
    for e in endpoints:
        e = dict(e)
        e.setdefault("source_manifest_hash", source_manifest_hash)
        e["files"] = sorted((dict(f) for f in e["files"]), key=lambda f: f["path"])
        if e.get("merge") is not None:
            merge = dict(e["merge"])
            if "adapter_input_files" in merge:
                merge["adapter_input_files"] = sorted(
                    (dict(f) for f in merge["adapter_input_files"]), key=lambda f: f["path"],
                )
            e["merge"] = merge
        normalized.append(e)
    normalized.sort(key=lambda e: e["role"])

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest_hash": source_manifest_hash,
        "endpoints": normalized,
    }
    validate_manifest_structure(payload)
    payload["manifest_hash"] = compute_manifest_hash(payload)
    return payload


# Pure validation-fact checks (no model object involved -- the caller extracts plain
# ints/bools/strings/dicts from a real or fake model/tokenizer and passes them in here)


def check_vocab_consistency(vocab_size: int, embedding_rows: int, lm_head_rows: int) -> bool:
    """The tokenizer's declared vocabulary size must not exceed either weight matrix's
    row count (extra rows beyond vocab_size are fine -- padding to a multiple is
    routine -- but a vocabulary the weights cannot address is not)."""
    return embedding_rows >= vocab_size and lm_head_rows >= vocab_size


def check_tied_embedding_consistency(tie_word_embeddings: bool, embeddings_are_equal: bool) -> bool:
    """If the config declares tied embeddings, the embedding and LM-head weights must
    actually be equal (or the same tensor); if it declares them untied, this check does
    not apply -- an untied model is not required to have coincidentally equal weights."""
    if not tie_word_embeddings:
        return True
    return embeddings_are_equal


def tokenizer_fingerprint(vocab: Mapping[str, int]) -> str:
    """A deterministic SHA-256 fingerprint of a tokenizer's full vocabulary content --
    lets a manifest record and later re-verify a tokenizer's exact identity without
    storing the entire vocabulary."""
    canonical = json.dumps(sorted(vocab.items()), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def check_tokenizer_extension(
    base_vocab: Mapping[str, int],
    adapter_vocab: Mapping[str, int],
    base_special_token_ids: Mapping[str, Optional[int]],
    adapter_special_token_ids: Mapping[str, Optional[int]],
) -> Dict[str, Any]:
    """Classify the relationship between an adapter's own tokenizer vocabulary and its
    base's.

    Returns `{"status": "identical" | "append_only_extension", "old_vocab_size":,
    "new_vocab_size":, "added_token_ids": [...]}` on success. Raises
    `TokenizerCompatibilityError`, naming the exact reason, on any of: a smaller
    adapter vocabulary (shrinking), a shared token whose ID changed (reordering), a
    changed/incompatible special token, or new tokens that are not a contiguous append
    at the end of the ID space.
    """
    base_size, adapter_size = len(base_vocab), len(adapter_vocab)
    if adapter_size < base_size:
        raise TokenizerCompatibilityError(
            f"adapter vocabulary ({adapter_size}) is smaller than base ({base_size}); shrinking is not permitted"
        )

    for token, base_id in base_vocab.items():
        if token not in adapter_vocab:
            raise TokenizerCompatibilityError(f"shared token {token!r} (base id {base_id}) is missing from the adapter tokenizer")
        if adapter_vocab[token] != base_id:
            raise TokenizerCompatibilityError(
                f"shared token {token!r} changed id: base={base_id}, adapter={adapter_vocab[token]}"
            )

    for name, base_id in base_special_token_ids.items():
        if base_id is None:
            continue
        # A base special token that is defined (non-null) must be present in the
        # adapter with the identical id -- a missing adapter value is a real
        # incompatibility, never silently treated as "not applicable".
        if name not in adapter_special_token_ids or adapter_special_token_ids[name] is None:
            raise TokenizerCompatibilityError(
                f"special token {name!r} (base id {base_id}) is missing from the adapter tokenizer"
            )
        adapter_id = adapter_special_token_ids[name]
        if adapter_id != base_id:
            raise TokenizerCompatibilityError(
                f"special token {name!r} changed id: base={base_id}, adapter={adapter_id}"
            )

    if adapter_size == base_size:
        return {"status": "identical", "old_vocab_size": base_size, "new_vocab_size": base_size, "added_token_ids": []}

    added_ids = sorted(set(adapter_vocab.values()) - set(base_vocab.values()))
    expected_added_ids = list(range(base_size, adapter_size))
    if added_ids != expected_added_ids:
        raise TokenizerCompatibilityError(
            f"adapter vocabulary extension is not a contiguous append: expected new ids "
            f"{expected_added_ids}, got {added_ids}"
        )
    return {
        "status": "append_only_extension", "old_vocab_size": base_size,
        "new_vocab_size": adapter_size, "added_token_ids": added_ids,
    }
