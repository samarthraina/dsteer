"""Bind one construction `activations.pt` to its exact originating `run_meta.json` and
endpoint pair (Task 016).

Before this module, `activations.pt` held the frozen construction identities and
tensor data, and its sibling `run_meta.json` recorded the endpoint pair -- but nothing
cryptographically tied the exact bytes of one to the exact content of the other. An
`activations.pt` from Pair B could be placed beside Pair A's `run_meta.json` and pass
every existing check, since each file was only ever verified against itself.

`activations_manifest_v1.json` is a small, self-hashed sidecar published only for an
endpoint-backed `primary_v1` construction run. It records the activation file's
streamed SHA-256/size, the *verified* `run_meta.json`'s own `run_identity_hash`, and
the endpoint/HarmfulQA/tensor-shape facts that file bound -- so a consumer can prove,
from the three files alone, that this exact `activations.pt` is the one that exact
`run_meta.json` produced, for this exact endpoint pair.

Not a general artifact framework: one schema, one publisher, one validator, all
specific to this one construction-activation binding problem.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import torch

# _compute_run_identity_hash is the exact canonical self-hash algorithm
# write_run_metadata itself resumes against (Task 008) -- imported narrowly so a
# consumed run_meta.json is re-verified against the same algorithm that produced it,
# not a subtly different reimplementation that could disagree with it.
from .artifacts import _compute_run_identity_hash
from .splits import load_manifest, validate_manifest_identity

SCHEMA_VERSION = 1
ARTIFACT_KIND = "dsteer_construction_activations"
ACTIVATION_FILENAME = "activations.pt"
RUN_META_FILENAME = "run_meta.json"
SIDECAR_FILENAME = "activations_manifest_v1.json"

#: What a valid primary_v1 construction run_meta.json must additionally satisfy,
#: beyond protocol_profile/endpoint_backed -- mirrors layer_profile.py's own frozen
#: construction constants and development_pilot.py's pre-Task-016 checks (kept
#: file-local rather than shared; see _default_harmfulqa_manifest_path below for why).
PRIMARY_CONSTRUCTION_PARTITION = "construction"
PRIMARY_CONSTRUCTION_RECORD_COUNT = 1378
PRIMARY_TOKEN_POSITION = "prompt_last"
SAFE_LOADING_POLICY: Dict[str, bool] = {"local_files_only": True, "trust_remote_code": False}


class ActivationArtifactError(ValueError):
    """A construction-activation sidecar, its `activations.pt`, or its `run_meta.json`
    failed to bind, self-verify, or match the caller's expected endpoint identity.

    Subclasses `ValueError` so it is caught by any existing `pytest.raises(ValueError)`
    around the frozen-manifest checks this module also performs (see
    `validate_construction_identity`, reused by `steer_sweep.validate_construction_activations`).
    """


@dataclass
class ActivationArtifact:
    """The result of a fully validated construction-activation sidecar: the already
    loaded activation payload plus the verified metadata that bound it, so a caller
    never needs to implement a second provenance check of its own."""

    blob: Dict[str, Any]
    run_meta: Dict[str, Any]
    sidecar: Dict[str, Any]
    acts_path: Path
    sha256: str
    size_bytes: int


# Hashing helpers


def _stream_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode("utf-8")


def _compute_sidecar_hash(payload: Mapping[str, Any]) -> str:
    reduced = {k: v for k, v in payload.items() if k != "manifest_hash"}
    return hashlib.sha256(_canonical_json(reduced)).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def save_activations_atomic(payload: Mapping[str, Any], path: Union[str, Path]) -> None:
    """Save an activations.pt payload through a sibling temporary file and an atomic
    rename, so a crash mid-write can never leave a partially written activations.pt --
    the rename either fully lands or does not happen at all."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        torch.save(dict(payload), tmp)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def verify_run_metadata_identity(run_meta: Any, context: str = "run_meta.json") -> None:
    """Reject `run_meta` unless it is a JSON object carrying a `run_identity_hash` that
    recomputes to itself under `steering.artifacts`'s own canonical algorithm (Task
    008). Raises on a missing, empty, non-string, or mismatched hash -- the last case
    covers any edit made to the file without recomputing it, not merely a missing
    field.
    """
    if not isinstance(run_meta, dict):
        raise ActivationArtifactError(f"{context}: not a JSON object")
    stored = run_meta.get("run_identity_hash")
    if not isinstance(stored, str) or not stored:
        raise ActivationArtifactError(f"{context}: no valid run_identity_hash")
    recomputed = _compute_run_identity_hash(run_meta)
    if recomputed != stored:
        raise ActivationArtifactError(
            f"{context}: run_identity_hash does not match its own content "
            f"(stored {stored!r}, recomputed {recomputed!r}) -- refusing to trust a file "
            "that was edited without recomputing its identity hash"
        )


def _verify_sidecar_self_hash(sidecar: Any, context: str) -> None:
    if not isinstance(sidecar, dict):
        raise ActivationArtifactError(f"{context}: not a JSON object")
    stored = sidecar.get("manifest_hash")
    if not isinstance(stored, str) or not stored:
        raise ActivationArtifactError(f"{context}: no valid manifest_hash")
    recomputed = _compute_sidecar_hash(sidecar)
    if recomputed != stored:
        raise ActivationArtifactError(
            f"{context}: manifest_hash does not match its own content "
            f"(stored {stored!r}, recomputed {recomputed!r}) -- refusing to trust a "
            "sidecar that was edited without recomputing its hash"
        )


# Frozen-manifest construction-identity checks (relocated canonical implementation;
# scripts/steer_sweep.py's validate_construction_activations delegates here so
# existing callers/tests keep working against the same, single implementation rather
# than a second, possibly-diverging one).


def _default_harmfulqa_manifest_path() -> Path:
    """manifests/harmfulqa_v1.json relative to the repository root, independent of the
    caller's current working directory. Mirrors steer_sweep.py's and layer_profile.py's
    own helpers of the same name -- kept file-local rather than shared, since all three
    are tiny and otherwise share no module."""
    return Path(__file__).resolve().parents[2] / "manifests" / "harmfulqa_v1.json"


def _construction_records(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """This manifest's construction-partition records, in ascending permuted_position --
    the exact order layer_profile.py extracted activations in."""
    return sorted(
        (e for e in manifest["records"] if e["partition"] == "construction"),
        key=lambda e: e["permuted_position"],
    )


def validate_construction_identity(blob: Mapping[str, Any], manifest_path: Optional[Union[str, Path]] = None) -> None:
    """Guard the vector-construction input against a legacy, mismatched, or drifted
    artifact -- checked against the frozen manifest itself, not just internal
    self-consistency.

    `layer_profile`'s manifest-backed construction path always writes `source_ids`,
    `prompt_hashes`, `partition`, and `manifest_hash` alongside `it`/`dpo`. A tensor
    missing them is a pre-manifest artifact or came from a different partition. Beyond
    that, this loads and hash-verifies the committed manifest, validates its frozen
    identity, and requires the tensor's ordered source_ids and prompt_hashes to exactly
    match the construction partition's records -- matching lengths and a correct
    top-level manifest_hash alone would still let a corrupted, reordered, or drifted
    artifact through.
    """
    required = ("source_ids", "prompt_hashes", "partition", "manifest_hash")
    missing = [k for k in required if k not in blob]
    if missing:
        raise ActivationArtifactError(
            f"activations.pt is missing {missing} -- looks like a legacy (pre-manifest) "
            "tensor with only 'it'/'dpo'; rerun layer_profile with the manifest-backed "
            "construction config"
        )
    if blob["partition"] != "construction":
        raise ActivationArtifactError(f"activations.pt partition is {blob['partition']!r}, expected 'construction'")

    path = Path(manifest_path) if manifest_path is not None else _default_harmfulqa_manifest_path()
    manifest = load_manifest(path)
    validate_manifest_identity(manifest)

    if blob["manifest_hash"] != manifest["manifest_hash"]:
        raise ActivationArtifactError(
            f"activations.pt manifest_hash {blob['manifest_hash']!r} does not match the "
            f"frozen manifest {manifest['manifest_hash']!r}"
        )

    construction = _construction_records(manifest)
    expected_source_ids = [e["source_id"] for e in construction]
    expected_prompt_hashes = [e["prompt_hash"] for e in construction]

    if blob["source_ids"] != expected_source_ids:
        raise ActivationArtifactError(
            "activations.pt source_ids do not exactly match the construction partition's "
            "ordered source IDs -- altered, reordered, or drifted provenance"
        )
    if blob["prompt_hashes"] != expected_prompt_hashes:
        raise ActivationArtifactError(
            "activations.pt prompt_hashes do not exactly match the construction partition's "
            "ordered prompt hashes -- altered, reordered, or drifted provenance"
        )

    it_shape, dpo_shape = tuple(blob["it"].shape), tuple(blob["dpo"].shape)
    if it_shape != dpo_shape:
        raise ActivationArtifactError(f"activations.pt it/dpo tensor shapes differ: {it_shape} vs {dpo_shape}")

    n_prompts = it_shape[1]
    if n_prompts != len(construction):
        raise ActivationArtifactError(
            f"activations.pt prompt dimension is {n_prompts}, expected {len(construction)} "
            "(the construction partition's record count)"
        )


# Publication (layer_profile.py, endpoint-backed primary_v1 extraction only)


def publish_activation_artifact(construction_dir: Union[str, Path]) -> Dict[str, Any]:
    """Publish `activations_manifest_v1.json`, binding the exact on-disk
    `activations.pt` bytes to the exact sibling `run_meta.json`.

    Must only be called after `activations.pt` has already been saved (via
    `save_activations_atomic`) and `run_meta.json` has already been written (via
    `steering.artifacts.write_run_metadata`) -- both are read back from disk here
    rather than trusted from anything already held in memory, so the sidecar describes
    exactly what is durably on disk, not what a caller merely believes it wrote.
    Refuses to publish for anything but a verified `primary_v1`, `endpoint_backed`
    `run_meta.json`, so a legacy extraction can never be falsely labelled primary_v1 by
    way of an over-eager sidecar.
    """
    construction_dir = Path(construction_dir)
    acts_path = construction_dir / ACTIVATION_FILENAME
    run_meta_path = construction_dir / RUN_META_FILENAME

    if not acts_path.exists():
        raise ActivationArtifactError(f"{acts_path} does not exist; cannot publish a sidecar for it")
    if not run_meta_path.exists():
        raise ActivationArtifactError(f"{run_meta_path} does not exist; cannot publish a sidecar without it")

    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    verify_run_metadata_identity(run_meta, str(run_meta_path))

    if run_meta.get("protocol_profile") != "primary_v1":
        raise ActivationArtifactError(
            "refusing to publish a construction-activation sidecar for a run whose "
            f"run_meta.json protocol_profile is {run_meta.get('protocol_profile')!r}, not 'primary_v1'"
        )
    if run_meta.get("endpoint_backed") is not True:
        raise ActivationArtifactError("refusing to publish a sidecar for a run that is not endpoint_backed")

    endpoint = run_meta.get("endpoint") or {}
    roles = endpoint.get("roles") or {}

    sha256 = _stream_sha256(acts_path)
    size_bytes = acts_path.stat().st_size

    blob = torch.load(acts_path, map_location="cpu")
    it_tensor, dpo_tensor = blob.get("it"), blob.get("dpo")
    if it_tensor is None or dpo_tensor is None:
        raise ActivationArtifactError(f"{acts_path} is missing 'it'/'dpo' tensors; cannot publish a sidecar for it")

    # Validate the saved payload itself before treating it as publishable -- the
    # sidecar is a completion marker, so it must never be written for a payload that
    # fails the same frozen-manifest/identity checks a consumer would later apply, and
    # run_meta.json's own harmfulqa fields must actually describe this activations.pt,
    # not merely be self-consistent within run_meta.json itself.
    validate_construction_identity(blob)
    if run_meta.get("harmfulqa_partition") != blob.get("partition"):
        raise ActivationArtifactError(
            f"{run_meta_path}: harmfulqa_partition {run_meta.get('harmfulqa_partition')!r} does not "
            f"match {acts_path}'s actual partition {blob.get('partition')!r}"
        )
    if run_meta.get("harmfulqa_manifest_hash") != blob.get("manifest_hash"):
        raise ActivationArtifactError(
            f"{run_meta_path}: harmfulqa_manifest_hash {run_meta.get('harmfulqa_manifest_hash')!r} does not "
            f"match {acts_path}'s actual manifest_hash {blob.get('manifest_hash')!r}"
        )
    actual_record_count = len(blob.get("source_ids") or [])
    if (
        run_meta.get("harmfulqa_record_count") != actual_record_count
        or actual_record_count != len(blob.get("prompt_hashes") or [])
        or actual_record_count != it_tensor.shape[1]
    ):
        raise ActivationArtifactError(
            f"{run_meta_path}: harmfulqa_record_count {run_meta.get('harmfulqa_record_count')!r} does not "
            f"agree with {acts_path}'s actual source_ids/prompt_hashes counts and tensor prompt dimension"
        )

    sidecar: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "activation_file": ACTIVATION_FILENAME,
        "activation_sha256": sha256,
        "activation_size_bytes": size_bytes,
        "run_meta_file": RUN_META_FILENAME,
        "run_identity_hash": run_meta["run_identity_hash"],
        "protocol_profile": run_meta.get("protocol_profile"),
        "endpoint_backed": run_meta.get("endpoint_backed"),
        "endpoint": {
            "pair": endpoint.get("pair"),
            "candidate_manifest_hash": endpoint.get("candidate_manifest_hash"),
            "source_manifest_hash": endpoint.get("source_manifest_hash"),
            "roles": {"it": roles.get("it"), "dpo": roles.get("dpo")},
        },
        "harmfulqa": {
            "partition": run_meta.get("harmfulqa_partition"),
            "manifest_hash": run_meta.get("harmfulqa_manifest_hash"),
            "record_count": run_meta.get("harmfulqa_record_count"),
        },
        "tensors": {
            "it_shape": list(it_tensor.shape), "dpo_shape": list(dpo_tensor.shape),
            "it_dtype": str(it_tensor.dtype), "dpo_dtype": str(dpo_tensor.dtype),
        },
    }
    sidecar["manifest_hash"] = _compute_sidecar_hash(sidecar)

    _atomic_write_json(construction_dir / SIDECAR_FILENAME, sidecar)
    return sidecar


# Central loader/validator (steer_sweep.py, development_pilot.py)


def load_and_validate_activation_artifact(
    construction_dir: Union[str, Path],
    *,
    expected_pair: Optional[str] = None,
    expected_candidate_manifest_hash: Optional[str] = None,
    expected_source_manifest_hash: Optional[str] = None,
    expected_it_role: Optional[str] = None,
    expected_dpo_role: Optional[str] = None,
    manifest_path: Optional[Union[str, Path]] = None,
) -> ActivationArtifact:
    """Require and fully validate the three sibling files -- `activations.pt`,
    `activations_manifest_v1.json`, `run_meta.json` -- binding one to the others before
    returning the already-loaded activation payload and verified metadata, so a caller
    never needs to implement a second provenance validator of its own.

    `expected_*` are this run's *own* resolved endpoint facts (never read from the
    files being validated): when given, a sidecar whose endpoint identity disagrees is
    rejected here even though the sidecar and its run_meta.json are perfectly
    self-consistent with *each other* -- this is what catches a Pair B artifact placed
    beside Pair A's run_meta.json, which neither file's own self-hash could ever
    detect on its own.
    """
    construction_dir = Path(construction_dir)
    acts_path = construction_dir / ACTIVATION_FILENAME
    run_meta_path = construction_dir / RUN_META_FILENAME
    sidecar_path = construction_dir / SIDECAR_FILENAME

    for path, label in (
        (acts_path, ACTIVATION_FILENAME), (run_meta_path, RUN_META_FILENAME), (sidecar_path, SIDECAR_FILENAME),
    ):
        if not path.exists():
            raise ActivationArtifactError(f"{construction_dir}: missing required sibling file {label}")

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    _verify_sidecar_self_hash(sidecar, str(sidecar_path))

    if sidecar.get("schema_version") != SCHEMA_VERSION:
        raise ActivationArtifactError(f"{sidecar_path}: schema_version {sidecar.get('schema_version')!r}, expected {SCHEMA_VERSION}")
    if sidecar.get("artifact_kind") != ARTIFACT_KIND:
        raise ActivationArtifactError(f"{sidecar_path}: artifact_kind {sidecar.get('artifact_kind')!r}, expected {ARTIFACT_KIND!r}")
    if sidecar.get("activation_file") != ACTIVATION_FILENAME:
        raise ActivationArtifactError(f"{sidecar_path}: activation_file must be exactly {ACTIVATION_FILENAME!r}, got {sidecar.get('activation_file')!r}")
    if sidecar.get("run_meta_file") != RUN_META_FILENAME:
        raise ActivationArtifactError(f"{sidecar_path}: run_meta_file must be exactly {RUN_META_FILENAME!r}, got {sidecar.get('run_meta_file')!r}")

    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    verify_run_metadata_identity(run_meta, str(run_meta_path))

    sidecar_run_hash = sidecar.get("run_identity_hash")
    if not isinstance(sidecar_run_hash, str) or sidecar_run_hash != run_meta.get("run_identity_hash"):
        raise ActivationArtifactError(
            f"{sidecar_path}: run_identity_hash does not equal {run_meta_path}'s verified "
            "run_identity_hash -- this sidecar does not bind to this exact run_meta.json"
        )

    if sidecar.get("protocol_profile") != "primary_v1" or run_meta.get("protocol_profile") != "primary_v1":
        raise ActivationArtifactError(
            f"{construction_dir}: protocol_profile must be 'primary_v1' in both the sidecar "
            f"({sidecar.get('protocol_profile')!r}) and run_meta.json ({run_meta.get('protocol_profile')!r})"
        )
    if sidecar.get("endpoint_backed") is not True or run_meta.get("endpoint_backed") is not True:
        raise ActivationArtifactError(f"{construction_dir}: endpoint_backed must be exactly true in both the sidecar and run_meta.json")

    token_position = ((run_meta.get("config") or {}).get("eval") or {}).get("token_position")
    if token_position != PRIMARY_TOKEN_POSITION:
        raise ActivationArtifactError(f"{run_meta_path}: token_position is {token_position!r}, expected {PRIMARY_TOKEN_POSITION!r}")
    if run_meta.get("model_loading_policy") != SAFE_LOADING_POLICY:
        raise ActivationArtifactError(
            f"{run_meta_path}: model_loading_policy is {run_meta.get('model_loading_policy')!r}, "
            f"expected {SAFE_LOADING_POLICY!r} (safe local loading)"
        )

    run_endpoint = run_meta.get("endpoint") or {}
    run_roles = run_endpoint.get("roles") or {}
    sidecar_endpoint = sidecar.get("endpoint") or {}
    sidecar_roles = sidecar_endpoint.get("roles") or {}
    for sidecar_val, run_val, field in (
        (sidecar_endpoint.get("pair"), run_endpoint.get("pair"), "endpoint.pair"),
        (sidecar_endpoint.get("candidate_manifest_hash"), run_endpoint.get("candidate_manifest_hash"), "endpoint.candidate_manifest_hash"),
        (sidecar_endpoint.get("source_manifest_hash"), run_endpoint.get("source_manifest_hash"), "endpoint.source_manifest_hash"),
        (sidecar_roles.get("it"), run_roles.get("it"), "endpoint.roles.it"),
        (sidecar_roles.get("dpo"), run_roles.get("dpo"), "endpoint.roles.dpo"),
    ):
        if sidecar_val != run_val:
            raise ActivationArtifactError(
                f"{construction_dir}: sidecar {field} ({sidecar_val!r}) does not match run_meta.json's ({run_val!r})"
            )

    sidecar_hqa = sidecar.get("harmfulqa") or {}
    for sidecar_val, run_val, field in (
        (sidecar_hqa.get("partition"), run_meta.get("harmfulqa_partition"), "harmfulqa.partition"),
        (sidecar_hqa.get("manifest_hash"), run_meta.get("harmfulqa_manifest_hash"), "harmfulqa.manifest_hash"),
        (sidecar_hqa.get("record_count"), run_meta.get("harmfulqa_record_count"), "harmfulqa.record_count"),
    ):
        if sidecar_val != run_val:
            raise ActivationArtifactError(
                f"{construction_dir}: sidecar {field} ({sidecar_val!r}) does not match run_meta.json's ({run_val!r})"
            )
    if sidecar_hqa.get("partition") != PRIMARY_CONSTRUCTION_PARTITION:
        raise ActivationArtifactError(f"{construction_dir}: harmfulqa.partition must be {PRIMARY_CONSTRUCTION_PARTITION!r}")
    if sidecar_hqa.get("record_count") != PRIMARY_CONSTRUCTION_RECORD_COUNT:
        raise ActivationArtifactError(f"{construction_dir}: harmfulqa.record_count must be {PRIMARY_CONSTRUCTION_RECORD_COUNT}")

    # Stream-verify the actual activation file against the sidecar's claim -- never
    # trust the recorded hash/size without re-deriving them from the bytes on disk.
    actual_sha256 = _stream_sha256(acts_path)
    actual_size = acts_path.stat().st_size
    if sidecar.get("activation_sha256") != actual_sha256:
        raise ActivationArtifactError(
            f"{acts_path}: sha256 {actual_sha256!r} does not match the sidecar's recorded "
            f"{sidecar.get('activation_sha256')!r} -- this is not the exact file the sidecar was published for"
        )
    if sidecar.get("activation_size_bytes") != actual_size:
        raise ActivationArtifactError(
            f"{acts_path}: size {actual_size} does not match the sidecar's recorded "
            f"{sidecar.get('activation_size_bytes')!r}"
        )

    blob = torch.load(acts_path, map_location="cpu")
    it_tensor, dpo_tensor = blob.get("it"), blob.get("dpo")
    if it_tensor is None or dpo_tensor is None:
        raise ActivationArtifactError(f"{acts_path} is missing 'it'/'dpo' tensors")
    sidecar_tensors = sidecar.get("tensors") or {}
    for actual, claimed, field in (
        (list(it_tensor.shape), sidecar_tensors.get("it_shape"), "tensors.it_shape"),
        (list(dpo_tensor.shape), sidecar_tensors.get("dpo_shape"), "tensors.dpo_shape"),
        (str(it_tensor.dtype), sidecar_tensors.get("it_dtype"), "tensors.it_dtype"),
        (str(dpo_tensor.dtype), sidecar_tensors.get("dpo_dtype"), "tensors.dpo_dtype"),
    ):
        if actual != claimed:
            raise ActivationArtifactError(f"{acts_path}: actual {field} {actual!r} does not match the sidecar's claimed {claimed!r}")

    # The existing frozen-manifest checks: exact ordered construction identities,
    # partition/manifest_hash, IT/DPO shape agreement, and the 1,378-prompt dimension.
    validate_construction_identity(blob, manifest_path)

    # Close the cross-file equality chain: the check above only proves
    # blob["manifest_hash"] itself is the real frozen one -- it says nothing about
    # whether the sidecar/run_meta.json's own harmfulqa.manifest_hash (already checked
    # against *each other* above) is that *same* value. A manually rehashed
    # run_meta.json and sidecar could otherwise consistently claim a different
    # HarmfulQA manifest while activations.pt itself carries the correct one.
    if sidecar_hqa.get("manifest_hash") != blob.get("manifest_hash"):
        raise ActivationArtifactError(
            f"{acts_path}: manifest_hash {blob.get('manifest_hash')!r} does not match the "
            f"sidecar/run_meta.json harmfulqa.manifest_hash {sidecar_hqa.get('manifest_hash')!r}"
        )
    if sidecar_hqa.get("partition") != blob.get("partition"):
        raise ActivationArtifactError(
            f"{acts_path}: partition {blob.get('partition')!r} does not match the "
            f"sidecar/run_meta.json harmfulqa.partition {sidecar_hqa.get('partition')!r}"
        )
    actual_record_count = len(blob.get("source_ids") or [])
    if (
        sidecar_hqa.get("record_count") != actual_record_count
        or actual_record_count != len(blob.get("prompt_hashes") or [])
        or actual_record_count != it_tensor.shape[1]
    ):
        raise ActivationArtifactError(
            f"{acts_path}: record count does not agree across source_ids/prompt_hashes/the tensor "
            f"prompt dimension and the sidecar/run_meta.json's harmfulqa.record_count "
            f"({sidecar_hqa.get('record_count')!r})"
        )

    # Optional caller-expected endpoint identity -- the actual substitution defense:
    # the caller states which pair/roles/manifest hashes *it* resolved for this run,
    # and a Pair-B artifact placed beside Pair-A's run_meta.json is rejected here even
    # though the sidecar/run_meta pair remains internally self-consistent.
    if expected_pair is not None and sidecar_endpoint.get("pair") != expected_pair:
        raise ActivationArtifactError(
            f"{construction_dir}: sidecar endpoint.pair {sidecar_endpoint.get('pair')!r} "
            f"does not match the expected pair {expected_pair!r}"
        )
    if expected_candidate_manifest_hash is not None and sidecar_endpoint.get("candidate_manifest_hash") != expected_candidate_manifest_hash:
        raise ActivationArtifactError(f"{construction_dir}: sidecar candidate_manifest_hash does not match the currently resolved endpoints")
    if expected_source_manifest_hash is not None and sidecar_endpoint.get("source_manifest_hash") != expected_source_manifest_hash:
        raise ActivationArtifactError(f"{construction_dir}: sidecar source_manifest_hash does not match the currently resolved endpoints")
    if expected_it_role is not None and sidecar_roles.get("it") != expected_it_role:
        raise ActivationArtifactError(
            f"{construction_dir}: sidecar endpoint.roles.it {sidecar_roles.get('it')!r} does not match expected {expected_it_role!r}"
        )
    if expected_dpo_role is not None and sidecar_roles.get("dpo") != expected_dpo_role:
        raise ActivationArtifactError(
            f"{construction_dir}: sidecar endpoint.roles.dpo {sidecar_roles.get('dpo')!r} does not match expected {expected_dpo_role!r}"
        )

    return ActivationArtifact(
        blob=blob, run_meta=run_meta, sidecar=sidecar,
        acts_path=acts_path, sha256=actual_sha256, size_bytes=actual_size,
    )
