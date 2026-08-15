"""Bind candidate endpoint identities to the active pipelines (protocol Sections 4/13/14,
Task 012).

Endpoint-backed mode resolves a Pair A or Pair B (IT, DPO) model pair from a hash- and
structurally-verified candidate endpoint manifest (`steering.endpoint_manifest`), instead
of trusting an arbitrary user-supplied YAML model config. Every architecture/tokenizer
fact used downstream comes from the endpoint's own recorded validation facts, never from
CLI/YAML input -- and every declared inference file is stream-verified (SHA-256 and byte
size) against what is actually on local disk before any of it is used.

This module never downloads, loads, or merges a model, and never touches a GPU. It only
resolves local filesystem paths and verifies local file content against an
already-verified manifest.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .artifact_identity import (
    load_manifest as load_source_manifest,
    validate_frozen_identity as validate_frozen_source_identity,
    validate_manifest_structure as validate_source_manifest_structure,
)
from .config import ModelConfig
from .endpoint_manifest import load_manifest as load_endpoint_manifest

#: Reusable role resolver: Pair A/B (IT, DPO) today. A future task can add "flip":
#: "M--A" to Pair A's mapping (or a standalone lookup through `resolve_role_entry`,
#: which already accepts any declared role) without touching this task's callers.
PAIR_ROLES: Dict[str, Dict[str, str]] = {
    "A": {"it": "M0-A", "dpo": "M+-A"},
    "B": {"it": "M0-B", "dpo": "M+-B"},
}
VALID_PAIRS: Tuple[str, ...] = tuple(sorted(PAIR_ROLES))

#: Every role the candidate manifest declares, in the frozen order the protocol lists
#: them (Section 4). The Gate 2 smoke test (and any future entrypoint needing every
#: endpoint at once, not one pair) resolves against this instead of PAIR_ROLES.
ALL_ROLES: Tuple[str, ...] = ("M0-A", "M+-A", "M--A", "M0-B", "M+-B")

#: The committed frozen source-artifact manifest -- the one authority a candidate
#: endpoint manifest is bound against. Never overridden by CLI/YAML input; the
#: `source_manifest_path` parameter threaded through below exists only so tests can
#: point at a synthetic frozen manifest instead of the real committed one.
DEFAULT_SOURCE_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "manifests" / "model_artifacts_v1.json"


class EndpointBindingError(RuntimeError):
    """A CLI-mode, endpoint-manifest, source-mapping, or local-file-verification
    invariant was violated while binding an endpoint identity to a pipeline."""


# Role resolution -- works against any manifest role, not only the two wired into a pair.


def resolve_role_entry(manifest: Mapping[str, Any], role: str) -> Dict[str, Any]:
    """The candidate manifest's own entry for `role`."""
    by_role = {e["role"]: e for e in manifest["endpoints"]}
    entry = by_role.get(role)
    if entry is None:
        raise EndpointBindingError(f"candidate manifest has no entry for role {role!r}")
    return entry


def roles_for_pair(pair: str) -> Tuple[str, str]:
    """(it_role, dpo_role) for "A" or "B"."""
    if pair not in PAIR_ROLES:
        raise EndpointBindingError(f"unknown pair {pair!r}; must be one of {VALID_PAIRS}")
    roles = PAIR_ROLES[pair]
    return roles["it"], roles["dpo"]


# Binding the candidate manifest to the frozen source-artifact authority
#
# `endpoint_manifest.load_manifest` only proves a candidate is self-consistent -- its
# stored hash matches its own content, and its structure is well-formed. That says
# nothing about whether the *source artifacts it claims* are the real, frozen ones: a
# manually constructed, correctly self-rehashed candidate can declare any
# source_manifest_hash, repository, revision, or anchor-file hash it likes. Every check
# below is against `manifests/model_artifacts_v1.json`, verified independently through
# `steering.artifact_identity`'s own hash/structure/frozen-identity checks -- never
# merely a value the candidate itself supplies.


def load_verified_frozen_source_manifest(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load, hash-verify, structurally validate, and frozen-identity-verify the
    committed source-artifact manifest. `path` defaults to the real committed
    `manifests/model_artifacts_v1.json`; the parameter exists only so a test can point
    at a synthetic (but equally hash/structure-verified) manifest instead."""
    source_manifest = load_source_manifest(path if path is not None else DEFAULT_SOURCE_MANIFEST_PATH)
    validate_source_manifest_structure(source_manifest)
    validate_frozen_source_identity(source_manifest)
    return source_manifest


def _check_frozen_anchor_files(
    role: str, frozen_files: Sequence[Mapping[str, Any]], declared_files: Sequence[Mapping[str, Any]], context: str,
) -> None:
    """Every frozen anchor file (path + sha256) must appear identically among
    `declared_files` -- a candidate is free to declare more files than the frozen
    anchors (e.g. a full inference bundle vs. two pinned weight files), but never a
    different hash for a file the frozen manifest already pins."""
    declared_by_path = {f["path"]: f for f in declared_files}
    for f in frozen_files:
        path = f["path"]
        declared = declared_by_path.get(path)
        if declared is None:
            raise EndpointBindingError(f"{role}: frozen anchor file {path!r} ({context}) is missing from the candidate manifest")
        if declared.get("sha256") != f["sha256"]:
            raise EndpointBindingError(
                f"{role}: frozen anchor file {path!r} ({context}) hash mismatch: "
                f"frozen {f['sha256']!r}, candidate {declared.get('sha256')!r}"
            )


def validate_candidate_against_frozen_source(candidate: Mapping[str, Any], source_manifest: Mapping[str, Any]) -> None:
    """Cross-check the candidate endpoint manifest against the verified frozen source
    manifest, for every declared endpoint -- not only the pair about to be resolved.

    Requires the candidate's `source_manifest_hash` to equal the frozen manifest's own
    hash; requires every endpoint's `base_artifact_id` to match its frozen record's; for
    a direct endpoint, requires `location.repository/revision/subpath` to match the
    frozen record exactly and every frozen anchor file to appear identically among the
    endpoint's declared files; for a merged endpoint, requires every frozen adapter
    anchor file to appear identically in `merge.adapter_input_files`. Raises on the
    first mismatch found.
    """
    if candidate.get("source_manifest_hash") != source_manifest.get("manifest_hash"):
        raise EndpointBindingError(
            f"candidate source_manifest_hash {candidate.get('source_manifest_hash')!r} does not match "
            f"the frozen source manifest hash {source_manifest.get('manifest_hash')!r}"
        )

    by_artifact_id = {a["artifact_id"]: a for a in source_manifest["artifacts"]}

    for endpoint in candidate["endpoints"]:
        role = endpoint["role"]
        artifact_id = endpoint["source_artifact_id"]
        frozen = by_artifact_id.get(artifact_id)
        if frozen is None:
            raise EndpointBindingError(f"{role}: source_artifact_id {artifact_id!r} is not declared in the frozen source manifest")

        if endpoint.get("base_artifact_id") != frozen.get("base_artifact_id"):
            raise EndpointBindingError(
                f"{role}: base_artifact_id {endpoint.get('base_artifact_id')!r} does not match the frozen "
                f"record's base_artifact_id {frozen.get('base_artifact_id')!r} for {artifact_id!r}"
            )

        if endpoint["status"] == "direct":
            location = endpoint["location"]
            for field in ("repository", "revision", "subpath"):
                if location.get(field) != frozen.get(field):
                    raise EndpointBindingError(
                        f"{role}: location.{field} {location.get(field)!r} does not match the frozen "
                        f"source record's {field} {frozen.get(field)!r} for {artifact_id!r}"
                    )
            _check_frozen_anchor_files(role, frozen["files"], endpoint["files"], f"source artifact {artifact_id!r}")
        else:
            merge = endpoint.get("merge") or {}
            adapter_input_files = merge.get("adapter_input_files") or []
            _check_frozen_anchor_files(role, frozen["files"], adapter_input_files, f"adapter {artifact_id!r}")


def _frozen_source_summary(frozen: Mapping[str, Any]) -> Dict[str, Any]:
    """The frozen source record's repository/revision/subpath and anchor files, for
    binding into run metadata -- so a merged endpoint's provenance shows the
    repository/revision of the adapter (or base) that produced it, not only its
    artifact_id."""
    return {
        "repository": frozen["repository"], "repository_type": frozen["repository_type"],
        "revision": frozen["revision"], "subpath": frozen["subpath"],
        "anchor_files": [{"path": f["path"], "sha256": f["sha256"]} for f in frozen["files"]],
    }


# Repeatable ARTIFACT_ID=LOCAL_ROOT CLI values


def parse_source_mappings(raw: Sequence[str]) -> Dict[str, Path]:
    """Parse repeatable `--endpoint-source ARTIFACT_ID=LOCAL_ROOT` values.

    Rejects a malformed entry or the same artifact_id supplied twice -- silently
    keeping "the last one wins" would quietly discard operator intent for exactly the
    input meant to prevent an ambiguous or wrong-role mapping.
    """
    out: Dict[str, Path] = {}
    for item in raw:
        artifact_id, sep, root = item.partition("=")
        artifact_id, root = artifact_id.strip(), root.strip()
        if not sep or not artifact_id or not root:
            raise EndpointBindingError(f"--endpoint-source must be ARTIFACT_ID=LOCAL_ROOT, got {item!r}")
        if artifact_id in out:
            raise EndpointBindingError(f"duplicate --endpoint-source mapping for artifact_id {artifact_id!r}")
        out[artifact_id] = Path(root)
    return out


def required_direct_artifact_ids(manifest: Mapping[str, Any], roles: Sequence[str]) -> set:
    """The `source_artifact_id` of every *direct* endpoint among `roles`. A merged
    endpoint (e.g. M+-B) is resolved entirely under the bundle root and needs no
    source-root mapping of its own."""
    required = set()
    for role in roles:
        entry = resolve_role_entry(manifest, role)
        if entry["status"] == "direct":
            required.add(entry["source_artifact_id"])
    return required


def validate_source_mappings(manifest: Mapping[str, Any], roles: Sequence[str], source_roots: Mapping[str, Path]) -> None:
    """Exactly the required set of artifact_ids must be supplied -- no fewer (a missing
    mapping), no more (an unknown artifact_id, or one that belongs to a different pair
    or role than the one being resolved)."""
    required = required_direct_artifact_ids(manifest, roles)
    supplied = set(source_roots)
    missing = sorted(required - supplied)
    if missing:
        raise EndpointBindingError(f"missing --endpoint-source mapping(s) for {missing}")
    unexpected = sorted(supplied - required)
    if unexpected:
        raise EndpointBindingError(
            f"unexpected --endpoint-source mapping(s) {unexpected}; this pair/role selection "
            f"only needs {sorted(required)}"
        )


# Local file verification -- stream-verify every declared file's SHA-256 and byte size;
# never read a file wholly into memory, and require the local file set to match exactly.


def _stream_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _local_regular_files(root: Path) -> Dict[str, Path]:
    """Every regular file under `root`, keyed by its root-relative forward-slash path.
    A symlink resolving outside `root` is rejected outright rather than followed; one
    resolving inside `root` is treated like any other file."""
    root = root.resolve()
    out: Dict[str, Path] = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            target = p.resolve()
            if not target.is_relative_to(root):
                raise EndpointBindingError(f"refusing to verify a symlink escaping the endpoint directory: {p} -> {target}")
            if not target.is_file():
                continue
        elif not p.is_file():
            continue
        out[str(p.relative_to(root)).replace("\\", "/")] = p
    return out


def verify_endpoint_files(root: Path, declared_files: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Stream-verify every declared file's SHA-256 and byte size against `root`, and
    require the local inference-critical regular-file set to match the manifest exactly
    -- neither a missing declared file nor an unexplained extra one is silently
    accepted. Size is checked before hashing, so a mismatched file is never hashed."""
    root = Path(root)
    if not root.is_dir():
        raise EndpointBindingError(f"endpoint root does not exist or is not a directory: {root}")

    declared_by_path = {f["path"]: f for f in declared_files}
    local = _local_regular_files(root)

    missing = sorted(set(declared_by_path) - set(local))
    if missing:
        raise EndpointBindingError(f"{root}: missing declared file(s): {missing}")
    unexpected = sorted(set(local) - set(declared_by_path))
    if unexpected:
        raise EndpointBindingError(f"{root}: unexpected file(s) not declared in the manifest: {unexpected}")

    verified: List[Dict[str, Any]] = []
    for path in sorted(declared_by_path):
        declared = declared_by_path[path]
        full = local[path]
        actual_size = full.stat().st_size
        if actual_size != declared["size_bytes"]:
            raise EndpointBindingError(
                f"{root}: size mismatch for {path!r}: declared {declared['size_bytes']}, got {actual_size}"
            )
        actual_sha256 = _stream_sha256(full)
        if actual_sha256 != declared["sha256"]:
            raise EndpointBindingError(
                f"{root}: sha256 mismatch for {path!r}: declared {declared['sha256']!r}, got {actual_sha256!r}"
            )
        verified.append({"path": path, "sha256": actual_sha256, "size_bytes": actual_size})
    return verified


# Local root resolution per endpoint status


def _resolve_direct_local_root(entry: Mapping[str, Any], source_roots: Mapping[str, Path]) -> Path:
    """Canonicalize immediately -- the same resolved path is stream-verified here and
    later reused unchanged for model loading and run metadata, so there is never a
    second, independently-resolved copy of "where this endpoint actually is" to drift
    from the one that was verified."""
    artifact_id = entry["source_artifact_id"]
    root = source_roots.get(artifact_id)
    if root is None:
        raise EndpointBindingError(f"{entry['role']}: no --endpoint-source mapping supplied for artifact_id {artifact_id!r}")
    return Path(root).resolve()


def _resolve_merged_local_root(entry: Mapping[str, Any], bundle_root: Path) -> Path:
    """Resolve strictly below `bundle_root` using the manifest's own bundle-relative
    role path. The candidate manifest's own structural validation already requires
    `location.path` to be a clean, relative, traversal-free path equal to the
    endpoint's own role -- this re-derives and re-checks the resolved path itself
    rather than trusting that alone.
    """
    location = entry["location"]
    if location.get("kind") != "bundle":
        raise EndpointBindingError(f"{entry['role']}: expected a bundle location, got {location!r}")
    rel = location.get("path")
    if not isinstance(rel, str) or not rel or rel != rel.strip():
        raise EndpointBindingError(f"{entry['role']}: malformed bundle location path {rel!r}")
    normalized = rel.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized) or ".." in normalized.split("/"):
        raise EndpointBindingError(f"{entry['role']}: bundle location path must be relative and traversal-free, got {rel!r}")

    bundle_root = Path(bundle_root).resolve()
    local = (bundle_root / rel).resolve()
    if not local.is_relative_to(bundle_root):
        raise EndpointBindingError(f"{entry['role']}: bundle location resolves outside the bundle root: {rel!r}")
    if local.is_symlink():
        target = local.resolve()
        if not target.is_relative_to(bundle_root):
            raise EndpointBindingError(f"{entry['role']}: bundle location is a symlink escaping the bundle root: {local} -> {target}")
    return local


@dataclass
class ResolvedEndpoint:
    """One endpoint, fully resolved and file-verified against local disk, with its
    identity bound to the frozen source-artifact record."""

    role: str
    status: str
    source_artifact_id: str
    base_artifact_id: Optional[str]
    local_path: Path
    validation: Dict[str, Any]
    location: Dict[str, Any]
    device: str
    library_versions: Dict[str, str]
    merge: Optional[Dict[str, Any]]
    verified_files: List[Dict[str, Any]]
    frozen_source: Dict[str, Any]
    base_frozen_source: Optional[Dict[str, Any]]

    def as_metadata(self) -> Dict[str, Any]:
        return {
            "role": self.role, "status": self.status,
            "source_artifact_id": self.source_artifact_id, "base_artifact_id": self.base_artifact_id,
            "local_path": str(self.local_path), "location": self.location,
            "validation": self.validation,
            "device": self.device, "library_versions": self.library_versions,
            "merge": self.merge, "verified_files": self.verified_files,
            "frozen_source": self.frozen_source, "base_frozen_source": self.base_frozen_source,
        }


def resolve_endpoint(
    manifest: Mapping[str, Any], role: str, source_manifest: Mapping[str, Any],
    source_roots: Mapping[str, Path], bundle_root: Union[str, Path],
) -> ResolvedEndpoint:
    """Resolve and file-verify one endpoint. A direct endpoint is resolved only from
    the explicitly supplied source root matching its declared source artifact ID; a
    merged endpoint is resolved only below `bundle_root` using its own bundle-relative
    role path. `source_manifest` is the already frozen-verified source manifest
    (`load_verified_frozen_source_manifest`), used only to attach frozen provenance to
    the result -- the caller must already have run
    `validate_candidate_against_frozen_source` before this is reached.
    """
    entry = resolve_role_entry(manifest, role)
    status = entry["status"]
    if status == "direct":
        local_root = _resolve_direct_local_root(entry, source_roots)
    elif status == "merged":
        local_root = _resolve_merged_local_root(entry, bundle_root)
    else:
        raise EndpointBindingError(f"{role}: unknown endpoint status {status!r}")

    verified_files = verify_endpoint_files(local_root, entry["files"])

    by_artifact_id = {a["artifact_id"]: a for a in source_manifest["artifacts"]}
    frozen = by_artifact_id[entry["source_artifact_id"]]
    base_artifact_id = entry.get("base_artifact_id")
    base_frozen_source = _frozen_source_summary(by_artifact_id[base_artifact_id]) if base_artifact_id else None

    return ResolvedEndpoint(
        role=role, status=status,
        source_artifact_id=entry["source_artifact_id"], base_artifact_id=base_artifact_id,
        local_path=local_root, validation=dict(entry["validation"]), location=dict(entry["location"]),
        device=entry["device"], library_versions=dict(entry["library_versions"]),
        merge=(dict(entry["merge"]) if entry.get("merge") is not None else None),
        verified_files=verified_files,
        frozen_source=_frozen_source_summary(frozen), base_frozen_source=base_frozen_source,
    )


@dataclass
class ResolvedPair:
    """A fully resolved and file-verified (IT, DPO) endpoint pair."""

    pair: str
    it: ResolvedEndpoint
    dpo: ResolvedEndpoint
    candidate_manifest_hash: str
    source_manifest_hash: str

    def to_model_config(self, name: str) -> ModelConfig:
        """Construct the in-memory model-pair configuration from validated endpoint
        facts only. Architecture, layer count, and tokenizer source never come from
        user-supplied YAML in endpoint-backed mode:

        - `architecture`/`num_layers` come from the IT endpoint's own recorded
          validation facts (matched-family manifest validation already guarantees
          these agree with the DPO endpoint's).
        - `tokenizer_id=None` deliberately leaves each script's own existing fallback
          (`tokenizer_id or it_model` / `tokenizer_id or which`) to pick the right
          tokenizer: the IT/M0 endpoint for matched activation construction, or the
          selected side's own endpoint for a side-specific generation load -- exactly
          the split the protocol requires, with no new tokenizer-selection logic.
        - No subfolder: a resolved endpoint is always a self-contained directory.
        """
        if self.it.validation["model_type"] != self.dpo.validation["model_type"]:
            raise EndpointBindingError(
                f"pair {self.pair}: IT/DPO model_type disagree "
                f"({self.it.validation['model_type']!r} vs {self.dpo.validation['model_type']!r}) "
                "despite matched-family manifest validation"
            )
        return ModelConfig(
            name=name,
            base_model=f"endpoint:{self.it.source_artifact_id}",
            it_model=str(self.it.local_path), dpo_model=str(self.dpo.local_path),
            architecture=self.it.validation["model_type"], num_layers=self.it.validation["num_hidden_layers"],
            tokenizer_id=None, it_subfolder=None, dpo_subfolder=None, tokenizer_subfolder=None,
        )

    def run_metadata(self) -> Dict[str, Any]:
        """Endpoint provenance for `run_meta.json` (protocol Section 14): candidate
        manifest hash, selected pair and roles, source manifest hash, per-endpoint
        status/source/base artifact IDs, validated facts, resolved local paths, and
        declared file hashes/sizes."""
        return {
            "mode": "endpoint",
            "pair": self.pair,
            "candidate_manifest_hash": self.candidate_manifest_hash,
            "source_manifest_hash": self.source_manifest_hash,
            "roles": {"it": self.it.role, "dpo": self.dpo.role},
            "endpoints": {"it": self.it.as_metadata(), "dpo": self.dpo.as_metadata()},
        }


def default_run_name(pair: str, candidate_manifest_hash: str) -> str:
    """Deterministic, filesystem-safe run name shared by any endpoint-backed script
    given the same candidate manifest and pair -- so `layer_profile.py`'s activations
    and `steer_sweep.py`'s vector-construction lookup land on the same
    `<output_dir>/<name>/...` tree without either script hardcoding the other's path."""
    return f"endpoint-{pair}-{candidate_manifest_hash[:12]}"


def _load_and_bind_candidate(
    manifest_path: Union[str, Path], source_manifest_path: Optional[Union[str, Path]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load the candidate manifest (existing hash and structural validation), load and
    verify the frozen source-artifact manifest, and bind the two together -- shared by
    every entrypoint that resolves one or more roles, whether one pair or all five.

    The frozen-source binding runs for the *entire* candidate manifest (every declared
    endpoint), not only the roles about to be resolved -- a candidate that lies about
    an endpoint the caller does not happen to need is still a candidate that lied.
    """
    manifest = load_endpoint_manifest(manifest_path)
    source_manifest = load_verified_frozen_source_manifest(source_manifest_path)
    validate_candidate_against_frozen_source(manifest, source_manifest)
    return manifest, source_manifest


def resolve_pair(
    manifest_path: Union[str, Path], pair: str,
    source_roots: Mapping[str, Path], bundle_root: Union[str, Path],
    source_manifest_path: Optional[Union[str, Path]] = None,
) -> ResolvedPair:
    """Resolve Pair A/B's IT and DPO endpoints, and stream-verify every declared file
    for both, after binding the candidate to the verified frozen source manifest."""
    if pair not in PAIR_ROLES:
        raise EndpointBindingError(f"unknown pair {pair!r}; must be one of {VALID_PAIRS}")

    manifest, source_manifest = _load_and_bind_candidate(manifest_path, source_manifest_path)

    it_role, dpo_role = roles_for_pair(pair)
    validate_source_mappings(manifest, (it_role, dpo_role), source_roots)

    it = resolve_endpoint(manifest, it_role, source_manifest, source_roots, bundle_root)
    dpo = resolve_endpoint(manifest, dpo_role, source_manifest, source_roots, bundle_root)

    return ResolvedPair(
        pair=pair, it=it, dpo=dpo,
        candidate_manifest_hash=manifest["manifest_hash"],
        source_manifest_hash=manifest["source_manifest_hash"],
    )


@dataclass
class ResolvedEndpointSet:
    """Every declared candidate-manifest role, resolved and file-verified at once, in
    `ALL_ROLES` order -- the all-role counterpart of `ResolvedPair`, for an entrypoint
    (the Gate 2 smoke test) that needs every endpoint rather than one pair."""

    roles: Dict[str, ResolvedEndpoint]
    candidate_manifest_hash: str
    source_manifest_hash: str

    def run_metadata(self) -> Dict[str, Any]:
        """Complete endpoint provenance for all five roles, for `run_meta.json`."""
        return {
            "mode": "endpoint_all_roles",
            "candidate_manifest_hash": self.candidate_manifest_hash,
            "source_manifest_hash": self.source_manifest_hash,
            "roles": list(ALL_ROLES),
            "endpoints": {role: self.roles[role].as_metadata() for role in ALL_ROLES},
        }


def resolve_all_roles(
    manifest_path: Union[str, Path],
    source_roots: Mapping[str, Path], bundle_root: Union[str, Path],
    source_manifest_path: Optional[Union[str, Path]] = None,
) -> ResolvedEndpointSet:
    """Resolve and file-verify every declared candidate-manifest role in `ALL_ROLES`
    order (`M0-A, M+-A, M--A, M0-B, M+-B`), after binding the candidate to the verified
    frozen source manifest. Direct endpoints (`M0-A`, `M+-A`, `M0-B`) are resolved only
    from the matching supplied source root; merged endpoints (`M--A`, `M+-B`) are
    resolved only beneath `bundle_root`. Exactly the direct-source artifact IDs among
    `ALL_ROLES` are required in `source_roots` -- no fewer, no more.
    """
    manifest, source_manifest = _load_and_bind_candidate(manifest_path, source_manifest_path)

    validate_source_mappings(manifest, ALL_ROLES, source_roots)

    roles = {
        role: resolve_endpoint(manifest, role, source_manifest, source_roots, bundle_root)
        for role in ALL_ROLES
    }

    return ResolvedEndpointSet(
        roles=roles,
        candidate_manifest_hash=manifest["manifest_hash"],
        source_manifest_hash=manifest["source_manifest_hash"],
    )


# CLI-mode resolution shared by every endpoint-backed-capable script


def resolve_model_source(
    *,
    model_config: Optional[str],
    endpoint_manifest: Optional[str],
    endpoint_bundle_root: Optional[str],
    pair: Optional[str],
    endpoint_source: Sequence[str],
    source_manifest_path: Optional[Union[str, Path]] = None,
) -> Tuple[ModelConfig, Optional[Dict[str, Any]]]:
    """Resolve CLI model-source flags into `(ModelConfig, endpoint_metadata_or_None)`.

    Legacy YAML mode (`model_config`) and endpoint-backed mode (every other argument
    here) are mutually exclusive, and endpoint-backed mode is never accepted
    half-specified. Every caller must run this before creating output, initializing
    logging, removing a partial file, or loading a prompt, model, or GPU resource --
    a mismatch here has no side effects.
    """
    endpoint_flags = {
        "--endpoint-manifest": endpoint_manifest,
        "--endpoint-bundle-root": endpoint_bundle_root,
        "--pair": pair,
        "--endpoint-source": list(endpoint_source) if endpoint_source else None,
    }
    endpoint_requested = any(v for v in endpoint_flags.values())
    legacy_requested = bool(model_config)

    if legacy_requested and endpoint_requested:
        raise EndpointBindingError(
            "--model-config (legacy YAML mode) and endpoint-backed mode "
            "(--endpoint-manifest/--endpoint-bundle-root/--pair/--endpoint-source) "
            "are mutually exclusive; pass exactly one"
        )
    if not legacy_requested and not endpoint_requested:
        raise EndpointBindingError(
            "no model source given: pass --model-config for legacy YAML mode (not confirmatory), "
            "or --endpoint-manifest/--endpoint-bundle-root/--pair/--endpoint-source for endpoint-backed mode"
        )

    if legacy_requested:
        return ModelConfig.from_yaml(model_config), None

    missing = sorted(flag for flag, v in endpoint_flags.items() if not v)
    if missing:
        raise EndpointBindingError(f"endpoint-backed mode is missing required flag(s): {missing}")

    source_roots = parse_source_mappings(endpoint_source)
    resolved = resolve_pair(endpoint_manifest, pair, source_roots, endpoint_bundle_root, source_manifest_path=source_manifest_path)
    name = default_run_name(pair, resolved.candidate_manifest_hash)
    return resolved.to_model_config(name), resolved.run_metadata()
