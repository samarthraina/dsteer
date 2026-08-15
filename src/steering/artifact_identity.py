"""Frozen model source-artifact identities (protocol Gate 0).

Mirrors the shape of `steering.splits`' HarmfulQA manifest -- a schema-versioned JSON
payload with a canonical SHA-256 over itself, a frozen-identity pin against tampering,
and offline verification against local files -- applied to the five model/adapter
source artifacts identified in `Brain/ARTIFACT_IDENTITY_AUDIT_2026-08-15.md` instead of
a dataset partition. Kept independent of `steering.splits` (no cross-import) even though
the canonicalization convention is the same, so this module stays self-contained and
reviewable without pulling in unrelated dataset-partition code.

This module never downloads, merges, or loads a model. It locks identity and verifies
already-downloaded files against declared hashes -- artifact preparation is a separate,
later concern.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

SCHEMA_VERSION = 1

EXPECTED_ARTIFACT_IDS = (
    "pair_a_sft",
    "pair_a_dpo",
    "pair_b_sft",
    "pair_b_dpo_adapter",
    "pair_a_flip_adapter",
)

VALID_REPOSITORY_TYPES = ("model", "dataset")
VALID_KINDS = ("checkpoint", "adapter")

_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class ArtifactIdentityError(ValueError):
    """A manifest-construction, structural-validation, frozen-identity, or offline
    file-verification invariant was violated."""


# Canonical serialization / manifest hash


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    """The exact serialization the manifest hash is computed over.

    Compact, key-sorted, ASCII-only, with `manifest_hash` itself excluded -- the hash
    cannot depend on its own value. Writing and verification must both call this.
    """
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
    Raises `ArtifactIdentityError` if an existing manifest differs -- a manifest is
    never silently overwritten with different content.
    """
    path = Path(path)
    text = dumps_manifest(payload)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != text:
            raise ArtifactIdentityError(f"manifest at {path} already exists with different content; refusing to overwrite")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def load_manifest(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a manifest and verify its stored hash against a fresh recomputation."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored = payload.get("manifest_hash")
    recomputed = compute_manifest_hash(payload)
    if stored != recomputed:
        raise ArtifactIdentityError(f"manifest hash mismatch at {path}: stored {stored!r}, recomputed {recomputed!r}")
    return payload


# Structural validation


def _validate_relative_path(path: Any) -> None:
    """Reject anything but a clean, relative, traversal-free path -- checked both at
    manifest-build time and again at verification time (an in-memory manifest built by
    a caller, not loaded from disk, must get the same protection)."""
    if not isinstance(path, str) or not path or path != path.strip():
        raise ArtifactIdentityError(f"declared file path must be a nonempty string: {path!r}")
    normalized = path.replace("\\", "/")
    if normalized.startswith("/"):
        raise ArtifactIdentityError(f"declared file path must be relative, got {path!r}")
    if re.match(r"^[A-Za-z]:", normalized):
        raise ArtifactIdentityError(f"declared file path must be relative, got {path!r}")
    if ".." in normalized.split("/"):
        raise ArtifactIdentityError(f"declared file path must not contain '..' traversal, got {path!r}")


def validate_manifest_structure(payload: Mapping[str, Any]) -> None:
    """Enforce the frozen manifest's structural invariants. Raises on the first
    violation found; never repairs, drops, or renumbers an artifact or file.
    """
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ArtifactIdentityError("payload['artifacts'] must be a nonempty list")

    ids = [a.get("artifact_id") for a in artifacts]
    if len(set(ids)) != len(ids):
        raise ArtifactIdentityError(f"duplicate artifact_id(s) in manifest: {ids}")
    if set(ids) != set(EXPECTED_ARTIFACT_IDS):
        raise ArtifactIdentityError(
            f"manifest must declare exactly {sorted(EXPECTED_ARTIFACT_IDS)}, got {sorted(set(ids))}"
        )

    by_id = {a["artifact_id"]: a for a in artifacts}

    for artifact_id, artifact in by_id.items():
        rtype = artifact.get("repository_type")
        if rtype not in VALID_REPOSITORY_TYPES:
            raise ArtifactIdentityError(
                f"{artifact_id}: repository_type must be one of {VALID_REPOSITORY_TYPES}, got {rtype!r}"
            )

        if not artifact.get("repository"):
            raise ArtifactIdentityError(f"{artifact_id}: repository must be a nonempty string")

        revision = artifact.get("revision")
        if not isinstance(revision, str) or not _HEX40_RE.match(revision):
            raise ArtifactIdentityError(
                f"{artifact_id}: revision must be 40 lowercase hex characters, got {revision!r}"
            )

        kind = artifact.get("kind")
        if kind not in VALID_KINDS:
            raise ArtifactIdentityError(f"{artifact_id}: kind must be one of {VALID_KINDS}, got {kind!r}")

        files = artifact.get("files")
        if not isinstance(files, list) or not files:
            raise ArtifactIdentityError(f"{artifact_id}: files must be a nonempty list")

        seen_paths = set()
        for f in files:
            path = f.get("path")
            _validate_relative_path(path)
            if path in seen_paths:
                raise ArtifactIdentityError(f"{artifact_id}: duplicate declared file path {path!r}")
            seen_paths.add(path)

            sha256 = f.get("sha256")
            if not isinstance(sha256, str) or not _HEX64_RE.match(sha256):
                raise ArtifactIdentityError(
                    f"{artifact_id}: sha256 for {path!r} must be 64 lowercase hex characters, got {sha256!r}"
                )

            size = f.get("size_bytes")
            if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 0):
                raise ArtifactIdentityError(
                    f"{artifact_id}: size_bytes for {path!r} must be a nonnegative integer or null, got {size!r}"
                )

    for artifact_id, artifact in by_id.items():
        kind = artifact["kind"]
        base_id = artifact.get("base_artifact_id")
        if kind == "adapter":
            if base_id is None:
                raise ArtifactIdentityError(f"{artifact_id}: an adapter must declare base_artifact_id")
            if base_id not in by_id:
                raise ArtifactIdentityError(f"{artifact_id}: base_artifact_id {base_id!r} does not exist")
            if by_id[base_id]["kind"] == "adapter":
                raise ArtifactIdentityError(f"{artifact_id}: base_artifact_id {base_id!r} is itself an adapter")
        elif base_id is not None:
            raise ArtifactIdentityError(f"{artifact_id}: a checkpoint must not declare base_artifact_id")


def build_manifest(artifacts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Build the schema-versioned, hash-computed model-artifact manifest.

    Artifacts and each artifact's files are sorted into a canonical order
    (`artifact_id`, then `path`) before hashing, so the result is byte-identical
    regardless of the order `artifacts` was supplied in. Validates structure first and
    raises `ArtifactIdentityError` rather than silently accepting a malformed record.
    """
    normalized = []
    for a in artifacts:
        a = dict(a)
        a["files"] = sorted((dict(f) for f in a["files"]), key=lambda f: f["path"])
        normalized.append(a)
    normalized.sort(key=lambda a: a["artifact_id"])

    payload: Dict[str, Any] = {"schema_version": SCHEMA_VERSION, "artifacts": normalized}
    validate_manifest_structure(payload)
    payload["manifest_hash"] = compute_manifest_hash(payload)
    return payload


# Frozen identity (Gate 0)
#
# Pins the one manifest this task is reviewed against -- schema, artifact IDs,
# revisions, every declared file hash, and the final manifest hash -- so a different,
# internally self-consistent manifest is rejected even if it passes structural
# validation on its own.

FROZEN_MODEL_ARTIFACT_IDENTITY: Dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "manifest_hash": "a16d842cb1fc0212a0a081b4c49183b2053e5902b00b90dfda255992a1a6a609",
    "artifacts": {
        "pair_a_sft": {
            "revision": "2932781b06bb062fb0fde146be0ebc13315fbbd3",
            "base_artifact_id": None,
            "file_hashes": {
                "model.safetensors": "1a2a404db755b04ac9385d8477f1853d86a586c8ec691abef9931701c0da50d5",
                "tokenizer.json": "8c5bbfc28fa7ce7c55165a2c11eee1765f6eed6ee6fcdc69ef6d9c2f17a41bda",
            },
        },
        "pair_a_dpo": {
            "revision": "2932781b06bb062fb0fde146be0ebc13315fbbd3",
            "base_artifact_id": None,
            "file_hashes": {
                "model.safetensors": "5eedaac41dcbc1fcab7446ba40b53b21d7c392e0ffe896da1b801c3be873b004",
                "tokenizer.json": "8c5bbfc28fa7ce7c55165a2c11eee1765f6eed6ee6fcdc69ef6d9c2f17a41bda",
            },
        },
        "pair_b_sft": {
            "revision": "24c0bea14d53e6f67f1fbe2eca5bfe7cae389b33",
            "base_artifact_id": None,
            "file_hashes": {
                "model-00001-of-00002.safetensors": "0b712f11ea29f3b34fa132403f7cafc0568c722ba3a33f42b55ed77b47fa299d",
                "model-00002-of-00002.safetensors": "5e6249c1a1ceb365e219a0fe667a77f71ec005b3aecb145ff2d8adf46cdb574f",
            },
        },
        "pair_b_dpo_adapter": {
            "revision": "613e550682436b83593aee0b6444ab0dfa56b659",
            "base_artifact_id": "pair_b_sft",
            "file_hashes": {
                "adapter_model.safetensors": "79c81ebc54c040c305fd35524fafe0edb3facc6be91284ad3e5ddc97bb758517",
                "training_args.bin": "67afea1a9303f2ff05cbb845452e4396bb10ca2b82bf7ba9dbe4fdca81225642",
            },
        },
        "pair_a_flip_adapter": {
            "revision": "613e550682436b83593aee0b6444ab0dfa56b659",
            "base_artifact_id": "pair_a_sft",
            "file_hashes": {
                "adapter_model.safetensors": "bf241922adfddd09559c32f157570344e12558a6300726d862cfda5ae32d3682",
                "training_args.bin": "4ec20c965ef181d607d756ff690047fbb89a110624d79301ade070b9ebe62f3d",
            },
        },
    },
}


def validate_frozen_identity(
    manifest: Mapping[str, Any], expected: Mapping[str, Any] = FROZEN_MODEL_ARTIFACT_IDENTITY,
) -> None:
    """Check a loaded manifest against a frozen identity. Raises `ArtifactIdentityError`
    listing every field that disagrees."""
    mismatches: List[str] = []

    def check(label: str, actual: Any, want: Any) -> None:
        if actual != want:
            mismatches.append(f"{label}: expected {want!r}, got {actual!r}")

    check("schema_version", manifest.get("schema_version"), expected["schema_version"])
    check("manifest_hash", manifest.get("manifest_hash"), expected["manifest_hash"])

    by_id = {a.get("artifact_id"): a for a in manifest.get("artifacts", [])}
    expected_artifacts = expected["artifacts"]
    check("artifact_ids", sorted(x for x in by_id if x is not None), sorted(expected_artifacts))

    for artifact_id, exp in expected_artifacts.items():
        actual = by_id.get(artifact_id)
        if actual is None:
            mismatches.append(f"{artifact_id}: missing from manifest")
            continue
        check(f"{artifact_id}.revision", actual.get("revision"), exp["revision"])
        check(f"{artifact_id}.base_artifact_id", actual.get("base_artifact_id"), exp["base_artifact_id"])
        actual_hashes = {f.get("path"): f.get("sha256") for f in actual.get("files", [])}
        for path, sha256 in exp["file_hashes"].items():
            check(f"{artifact_id}.files[{path!r}].sha256", actual_hashes.get(path), sha256)

    if mismatches:
        raise ArtifactIdentityError(
            "manifest does not match the frozen model-artifact identity: " + "; ".join(mismatches)
        )


# Offline local-file verification


@dataclass
class VerifiedFile:
    """One declared file, confirmed present and matching on disk."""
    path: str
    sha256: str
    size_bytes: int


def _stream_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 over `path`, read in fixed-size chunks -- never the whole file into
    memory, since these are model-sized weight files."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_local_artifact(
    manifest: Mapping[str, Any], artifact_id: str, local_root: Union[str, Path],
) -> List[VerifiedFile]:
    """Confirm every file `manifest` declares for `artifact_id` is present under
    `local_root` and matches its declared size (if any) and SHA-256 exactly.

    `local_root` is the caller's already-downloaded copy of the artifact's source
    subfolder -- declared paths are resolved relative to it, never embedded as
    platform-specific paths in the manifest itself. Streams each file's hash rather
    than reading it whole. Raises `ArtifactIdentityError` on the first problem found
    (unknown artifact, missing file, non-file path, a root-escaping declared path, a
    size mismatch, or a hash mismatch) and never repairs, downloads, renames, deletes,
    overwrites, or merges anything. Returns the verified files only if every one passes.
    """
    by_id = {a.get("artifact_id"): a for a in manifest.get("artifacts", [])}
    artifact = by_id.get(artifact_id)
    if artifact is None:
        raise ArtifactIdentityError(f"unknown artifact_id {artifact_id!r}")

    root = Path(local_root).resolve()
    verified: List[VerifiedFile] = []

    for f in artifact.get("files", []):
        rel_path = f.get("path")
        _validate_relative_path(rel_path)

        full_path = (root / rel_path).resolve()
        if not full_path.is_relative_to(root):
            raise ArtifactIdentityError(
                f"{artifact_id}: declared path {rel_path!r} resolves outside {root}"
            )

        if not full_path.exists():
            raise ArtifactIdentityError(f"{artifact_id}: missing file {full_path}")
        if not full_path.is_file():
            raise ArtifactIdentityError(f"{artifact_id}: not a regular file: {full_path}")

        actual_size = full_path.stat().st_size
        declared_size = f.get("size_bytes")
        if declared_size is not None and actual_size != declared_size:
            raise ArtifactIdentityError(
                f"{artifact_id}: size mismatch for {full_path}: declared {declared_size}, got {actual_size}"
            )

        actual_sha256 = _stream_sha256(full_path)
        declared_sha256 = f.get("sha256")
        if actual_sha256 != declared_sha256:
            raise ArtifactIdentityError(
                f"{artifact_id}: sha256 mismatch for {full_path}: "
                f"declared {declared_sha256!r}, got {actual_sha256!r}"
            )

        verified.append(VerifiedFile(path=rel_path, sha256=actual_sha256, size_bytes=actual_size))

    return verified
