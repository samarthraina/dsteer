"""Gate 3: frozen 50-prompt development-pilot runner (protocol Sections 6-9, 13 Gate 3).

A small, bounded engineering pilot -- not a general experiment manager. It generates
under the frozen primary intervention plan (layers [27,28,29,30,31], mean checkpoint-
difference vectors, relative normalisation, all positions, additive, greedy,
max_new_tokens=512, max_input_length=2048) at a single pilot-only coefficient (0.05,
the frozen grid's midpoint) against exactly the 50 development-partition prompts Gate 2
did not already touch (positions 10-59 of the frozen 60-record `development` partition;
Gate 2 used positions 0-9). It exercises the ordinary Pair A / Pair B installation and
reversal paths plus one small, explicitly pilot-only Pair A random-direction diagnostic
(dedicated random seed 11, reused for both the installation and the reversal arm) --
never Pair B random arms, never projection ablation, never a single-layer diagnostic,
never a second coefficient.

    python scripts/development_pilot.py \\
        --endpoint-manifest outputs/endpoints/endpoint_manifest_candidate_v1.json \\
        --endpoint-bundle-root outputs/endpoints \\
        --endpoint-source pair_a_sft=/data/pair_a/SFT_merged \\
        --endpoint-source pair_a_dpo=/data/pair_a/DPO_merged \\
        --endpoint-source pair_b_sft=/data/pair_b/sft \\
        --gate2-dir outputs/gate2_smoke \\
        --pair-a-activation-run outputs/layer_profile/endpoint-A-.../construction \\
        --pair-b-activation-run outputs/layer_profile/endpoint-B-.../construction \\
        --output-dir outputs/gate3_pilot

Endpoint resolution, a successful and identity-matching Gate 2 run, and manifest-
verified `primary_v1` construction activations for both pairs are all required, and all
verified before any output is created, any seed is set, any vector is built, or any
tokenizer/model/GPU resource is touched (protocol Section 13, Gate 3). Models are loaded
and released strictly one at a time, in the order M0-A, M+-A, M--A, M0-B, M+-B -- never
two resident together.

This script writes frozen, judge-ready JSONL only. It never imports or constructs a
`Judge`, never starts a judge server, never scores a response, and never invokes
`score_sweep.py` -- judging is a separate, later, manually-triggered action. Its own
`gate3_summary.json` reports record-count/identity/schema/stopping/validity integrity
only -- never an effect-based ("which arm looks better") pass/fail call -- and always
sets `manual_review_required=True` and `automatically_continue_to_calibration=False`
regardless of its own verdict. The completed output is always preserved; only the
process exit code reflects `automated_integrity_pass`, exactly as Gate 2's own summary
and exit code do.

Unsteered baseline records never claim an intervention that did not happen (no layers,
no vector method/normalisation, `coefficient_status="not_applicable_unsteered"`); a
real-direction arm is `protocol_profile="primary_v1"`, a random-direction arm is
`protocol_profile="secondary_random_control_v1"` -- never one generic label shared by
every arm. The general run seed is frozen at 42 (no `--seed` flag); the generation
batch size defaults to 10 and is always recorded, never an unrecorded automatic
estimate. M--A's `merge.flip_lineage` (`label_swap_lineage_verified`,
`confirmatory_eligible`) is independently re-verified to be exactly `False` before any
generation and is recorded in run metadata, its arm-plan entry, and every M--A record.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generation_smoke  # noqa: E402 -- reused only for select_smoke_records (Gate 2's own definition of "the first 10")
import steer_sweep  # noqa: E402 -- reused only for validate_construction_activations

# _compute_run_identity_hash is the exact canonical self-hash algorithm write_run_metadata
# itself resumes against (Task 008) -- imported narrowly so a consumed run_meta.json is
# re-verified against the same algorithm that produced it, not a subtly different
# reimplementation that could disagree with it.
from steering.artifacts import GpuMonitor, _compute_run_identity_hash, write_run_metadata  # noqa: E402
from steering.data import load_harmfulqa_partition  # noqa: E402
from steering.endpoint_binding import (  # noqa: E402
    ALL_ROLES, EndpointBindingError, ResolvedEndpointSet,
    parse_source_mappings, resolve_all_roles, roles_for_pair,
)
from steering.generate import build_chat_prompts, generate_batched, generation_terminators  # noqa: E402
from steering.models import load_model, load_tokenizer  # noqa: E402
from steering.steer import ActivationSteering, build_vectors, random_vectors_like  # noqa: E402
from steering.utils import read_jsonl, set_all_seeds, setup_logging  # noqa: E402
from steering.validity import check as check_validity  # noqa: E402

#: Frozen for Gate 3 -- not CLI-overridable.
GATE3_PARTITION = "development"
DEVELOPMENT_PARTITION_SIZE = 60
GATE2_RECORD_COUNT = 10
PILOT_RECORD_COUNT = 50

#: The frozen primary intervention plan (protocol Sections 6-7), applied at a single
#: pilot-only coefficient rather than the full calibration grid.
PILOT_LAYERS: Tuple[int, ...] = (27, 28, 29, 30, 31)
PILOT_VECTOR_METHOD = "mean"
PILOT_RANDOM_VECTOR_METHOD = "random_norm_matched"
PILOT_VECTOR_NORMALISE = "relative"
PILOT_POSITIONS = "all"
PILOT_PRESERVE_NORM = False
PILOT_MODE = "add"
PILOT_MAX_NEW_TOKENS = 512
PILOT_MAX_INPUT_LENGTH = 2048

#: The general run seed is frozen, not a scientific degree of freedom -- there is no
#: --seed CLI flag (protocol Section 7).
PILOT_SEED = 42

#: Not CLI-overridable to zero/negative; a positive --batch-size may still override this
#: default for hardware efficiency only, and the resolved value is always recorded.
PILOT_DEFAULT_BATCH_SIZE = 10

#: The fixed midpoint of the frozen calibration grid -- pipeline/metric-sensitivity
#: testing only, never a selected, final, optimal, or confirmatory coefficient.
PILOT_COEFFICIENT = 0.05
PILOT_COEFFICIENT_STATUS = "pilot_only_not_calibrated"
COEFFICIENT_STATUS_UNSTEERED = "not_applicable_unsteered"

#: The one dedicated random-direction identity this pilot uses, for Pair A only. Built
#: once and reused unchanged for both the installation and the reversal arm.
PILOT_RANDOM_SEED = 11

#: What produced this run (constant across every record) versus what protocol/
#: intervention identity a given *arm* claims -- the latter must never collapse to one
#: generic value shared by unsteered, real-direction, and random-direction arms alike.
PILOT_RUN_PROFILE = "gate3_development_pilot"
PROTOCOL_PROFILE_REAL = "primary_v1"
PROTOCOL_PROFILE_RANDOM = "secondary_random_control_v1"
INTERVENTION_PROFILE_BASELINE = "unsteered_baseline"
INTERVENTION_PROFILE_REAL = "real_direction_pilot"
INTERVENTION_PROFILE_RANDOM = "random_direction_pilot"

#: Frozen construction requirements an activation run must satisfy (mirrors
#: layer_profile.py's own PRIMARY_CONSTRUCTION_* constants; kept file-local rather than
#: shared, since both are tiny and the scripts otherwise share no module).
PRIMARY_CONSTRUCTION_PARTITION = "construction"
PRIMARY_CONSTRUCTION_RECORD_COUNT = 1378
PRIMARY_TOKEN_POSITION = "prompt_last"
ENDPOINT_LOADING_POLICY: Dict[str, bool] = {"local_files_only": True, "trust_remote_code": False}


class DevelopmentPilotError(RuntimeError):
    """A Gate 3 precondition (record selection, Gate 2 identity, activation-run
    provenance, terminator availability, or generation-count integrity) was violated."""


def _baseline_arm(arm_id: str, pair: str, role: str, lineage_flag: Optional[str] = None) -> Dict[str, Any]:
    """An unsteered baseline never claims an intervention that did not occur: no
    layers, no vector method/normalisation/positions/preserve_norm, no protocol
    classification, and a coefficient status that says so explicitly."""
    return {
        "arm_id": arm_id, "pair": pair, "endpoint_role": role, "arm_type": "unsteered_baseline",
        "intervention_operation": "none", "coefficient": 0.0,
        "vector_source": None, "random_vector_seed": None, "lineage_flag": lineage_flag,
        "protocol_profile": None, "intervention_profile": INTERVENTION_PROFILE_BASELINE,
        "coefficient_status": COEFFICIENT_STATUS_UNSTEERED,
        "vector_method": None, "layers": [], "normalization": None, "positions": None, "preserve_norm": None,
    }


def _real_direction_arm(arm_id: str, pair: str, role: str, arm_type: str, signed_coefficient: float, vector_source: str) -> Dict[str, Any]:
    return {
        "arm_id": arm_id, "pair": pair, "endpoint_role": role, "arm_type": arm_type,
        "intervention_operation": "add", "coefficient": signed_coefficient,
        "vector_source": vector_source, "random_vector_seed": None, "lineage_flag": None,
        "protocol_profile": PROTOCOL_PROFILE_REAL, "intervention_profile": INTERVENTION_PROFILE_REAL,
        "coefficient_status": PILOT_COEFFICIENT_STATUS,
        "vector_method": PILOT_VECTOR_METHOD, "layers": list(PILOT_LAYERS),
        "normalization": PILOT_VECTOR_NORMALISE, "positions": PILOT_POSITIONS, "preserve_norm": PILOT_PRESERVE_NORM,
    }


def _random_direction_arm(arm_id: str, pair: str, role: str, arm_type: str, signed_coefficient: float, vector_source: str) -> Dict[str, Any]:
    return {
        "arm_id": arm_id, "pair": pair, "endpoint_role": role, "arm_type": arm_type,
        "intervention_operation": "add", "coefficient": signed_coefficient,
        "vector_source": vector_source, "random_vector_seed": PILOT_RANDOM_SEED, "lineage_flag": None,
        "protocol_profile": PROTOCOL_PROFILE_RANDOM, "intervention_profile": INTERVENTION_PROFILE_RANDOM,
        "coefficient_status": PILOT_COEFFICIENT_STATUS,
        "vector_method": PILOT_RANDOM_VECTOR_METHOD, "layers": list(PILOT_LAYERS),
        "normalization": PILOT_VECTOR_NORMALISE, "positions": PILOT_POSITIONS, "preserve_norm": PILOT_PRESERVE_NORM,
    }


#: The exact eleven-arm plan (protocol Section 8). Fully static: which model an arm
#: loads, what it does, and how it is labelled are all fixed ahead of any run, so this
#: is testable without any endpoint, vector, or generation machinery. `vector_source`
#: keys into the per-run `vector_hashes`/`vectors_by_source` maps built in main().
ARM_PLAN: Tuple[Dict[str, Any], ...] = (
    _baseline_arm("M0-A_baseline", "A", "M0-A"),
    _baseline_arm("M+-A_baseline", "A", "M+-A"),
    _baseline_arm("M--A_baseline", "A", "M--A", lineage_flag="exploratory_lineage_unverified"),
    _real_direction_arm("M0-A_install_real", "A", "M0-A", "real_direction_installation", PILOT_COEFFICIENT, "real_mean_pair_A"),
    _real_direction_arm("M+-A_reverse_real", "A", "M+-A", "real_direction_reversal", -PILOT_COEFFICIENT, "real_mean_pair_A"),
    _random_direction_arm("M0-A_install_random_s11", "A", "M0-A", "random_direction_installation", PILOT_COEFFICIENT, "random_seed_11_pair_A"),
    _random_direction_arm("M+-A_reverse_random_s11", "A", "M+-A", "random_direction_reversal", -PILOT_COEFFICIENT, "random_seed_11_pair_A"),
    _baseline_arm("M0-B_baseline", "B", "M0-B"),
    _baseline_arm("M+-B_baseline", "B", "M+-B"),
    _real_direction_arm("M0-B_install_real", "B", "M0-B", "real_direction_installation", PILOT_COEFFICIENT, "real_mean_pair_B"),
    _real_direction_arm("M+-B_reverse_real", "B", "M+-B", "real_direction_reversal", -PILOT_COEFFICIENT, "real_mean_pair_B"),
)


def arms_by_role() -> Dict[str, List[Dict[str, Any]]]:
    """`ARM_PLAN` grouped by the endpoint role that generates it, preserving plan order
    within each role -- iterating this in `ALL_ROLES` order is exactly the sequential
    load/generate/release grouping protocol Section 9 describes."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for arm in ARM_PLAN:
        out.setdefault(arm["endpoint_role"], []).append(arm)
    return out


# Frozen record selection (protocol Section 6)


def select_pilot_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Exactly the 50 development-partition records Gate 2 did not use: positions 10-59
    of the frozen, ascending-`permuted_position` ordering, after Gate 2's own first ten
    (`generation_smoke.select_smoke_records`) are excluded.

    `records` is re-sorted here rather than trusted to already be in order. Requires
    exactly `DEVELOPMENT_PARTITION_SIZE` (60) records -- never truncates, samples, or
    substitutes another partition to make up a shortfall.
    """
    ordered = sorted(records, key=lambda r: r["permuted_position"])
    if len(ordered) != DEVELOPMENT_PARTITION_SIZE:
        raise DevelopmentPilotError(
            f"{GATE3_PARTITION!r} partition has {len(ordered)} records; Gate 3 selection "
            f"requires exactly {DEVELOPMENT_PARTITION_SIZE}"
        )
    first_ten = generation_smoke.select_smoke_records(ordered)
    pilot = ordered[GATE2_RECORD_COUNT:DEVELOPMENT_PARTITION_SIZE]
    if len(pilot) != PILOT_RECORD_COUNT:
        raise DevelopmentPilotError(
            f"selected {len(pilot)} pilot records, expected exactly {PILOT_RECORD_COUNT}"
        )

    first_ten_ids = {r["source_id"] for r in first_ten}
    first_ten_hashes = {r["prompt_hash"] for r in first_ten}
    overlap_ids = [r["source_id"] for r in pilot if r["source_id"] in first_ten_ids]
    overlap_hashes = [r["prompt_hash"] for r in pilot if r["prompt_hash"] in first_ten_hashes]
    if overlap_ids or overlap_hashes:
        raise DevelopmentPilotError(
            f"pilot selection overlaps Gate 2's first ten: source_id overlap {overlap_ids}, "
            f"prompt_hash overlap {overlap_hashes}"
        )
    return pilot


# Consumed run_meta.json self-hash verification
#
# Gate 2's run_meta.json and both activation runs' run_meta.json files are read and
# trusted for cross-checks below -- but a JSON file sitting on disk is only as
# trustworthy as its own self-hash proves it to be. A field could have been hand-edited
# (or overwritten by some unrelated process) after the file was written, and nothing
# about its *shape* would look wrong. Re-verifying `run_identity_hash` here, with the
# exact algorithm write_run_metadata itself resumes against, catches that before any
# field from the file is trusted for anything downstream.


def verify_run_metadata_identity(run_meta: Any, context: str) -> None:
    """Reject `run_meta` unless it is a JSON object carrying a `run_identity_hash` that
    recomputes to itself under `steering.artifacts`'s own canonical algorithm (Task
    008). Raises on a missing, empty, non-string, or mismatched hash -- the last case
    covers any edit made to the file without recomputing it, not merely a missing
    field.
    """
    if not isinstance(run_meta, dict):
        raise DevelopmentPilotError(f"{context}: run_meta.json is not a JSON object")
    stored = run_meta.get("run_identity_hash")
    if not isinstance(stored, str) or not stored:
        raise DevelopmentPilotError(f"{context}: run_meta.json has no valid run_identity_hash")
    recomputed = _compute_run_identity_hash(run_meta)
    if recomputed != stored:
        raise DevelopmentPilotError(
            f"{context}: run_meta.json run_identity_hash does not match its own content "
            f"(stored {stored!r}, recomputed {recomputed!r}) -- refusing to trust a file "
            "that was edited without recomputing its identity hash"
        )


# Gate 2 prerequisite (protocol Section 13, Gate 2 -> Gate 3)


def verify_gate2_prerequisite(
    gate2_dir: Path, resolved: ResolvedEndpointSet, expected_first_ten: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """A successful, identity-matching Gate 2 run is required before anything else in
    Gate 3 touches output, a tokenizer/model, a seed, or a vector.

    Never trusts `gate2_summary.json`'s boolean alone: cross-checks every role's JSONL
    file against the frozen first-ten identities this run independently computed (not
    merely against what the file claims about itself), and cross-checks Gate 2's own
    `run_meta.json` endpoint binding against the endpoints resolved for *this* run.
    """
    summary_path = gate2_dir / "gate2_summary.json"
    if not summary_path.exists():
        raise DevelopmentPilotError(f"Gate 2 summary not found at {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if summary.get("gate") != "gate2_generation_smoke":
        raise DevelopmentPilotError(f"{summary_path}: gate is {summary.get('gate')!r}, expected 'gate2_generation_smoke'")
    if summary.get("pass") is not True:
        raise DevelopmentPilotError(f"{summary_path}: Gate 2 did not pass (pass={summary.get('pass')!r})")
    if summary.get("candidate_manifest_hash") != resolved.candidate_manifest_hash:
        raise DevelopmentPilotError(
            f"Gate 2 candidate_manifest_hash {summary.get('candidate_manifest_hash')!r} does not match "
            f"the currently resolved endpoints {resolved.candidate_manifest_hash!r}"
        )
    if summary.get("source_manifest_hash") != resolved.source_manifest_hash:
        raise DevelopmentPilotError(
            f"Gate 2 source_manifest_hash {summary.get('source_manifest_hash')!r} does not match "
            f"the currently resolved endpoints {resolved.source_manifest_hash!r}"
        )

    roles_summary = summary.get("roles", {})
    for role in ALL_ROLES:
        role_info = roles_summary.get(role)
        if role_info is None or role_info.get("pass") is not True:
            raise DevelopmentPilotError(f"Gate 2 role {role!r} is missing or failing in {summary_path}")

    expected_triples = [(r["source_id"], r["prompt_hash"], r["permuted_position"]) for r in expected_first_ten]
    for role in ALL_ROLES:
        role_path = gate2_dir / f"{role}.jsonl"
        if not role_path.exists():
            raise DevelopmentPilotError(f"Gate 2 role file missing: {role_path}")
        role_records = read_jsonl(role_path)
        if len(role_records) != GATE2_RECORD_COUNT:
            raise DevelopmentPilotError(
                f"Gate 2 role {role!r} has {len(role_records)} records, expected exactly {GATE2_RECORD_COUNT}"
            )
        actual_triples = [(r.get("source_id"), r.get("prompt_hash"), r.get("permuted_position")) for r in role_records]
        if actual_triples != expected_triples:
            raise DevelopmentPilotError(
                f"Gate 2 role {role!r} identities do not exactly match the frozen first ten "
                "development records in order -- a different partition or record set may have been used"
            )
        partitions = {r.get("harmfulqa_partition") for r in role_records}
        if partitions != {GATE3_PARTITION}:
            raise DevelopmentPilotError(f"Gate 2 role {role!r} used partition(s) {partitions}, expected only {GATE3_PARTITION!r}")

    run_meta_path = gate2_dir / "run_meta.json"
    if not run_meta_path.exists():
        raise DevelopmentPilotError(f"Gate 2 run_meta.json not found at {run_meta_path}")
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    verify_run_metadata_identity(run_meta, f"Gate 2 ({run_meta_path})")
    gate2_endpoint = run_meta.get("endpoint") or {}
    if gate2_endpoint.get("candidate_manifest_hash") != resolved.candidate_manifest_hash:
        raise DevelopmentPilotError(
            "Gate 2 run_meta.json does not bind the same candidate endpoint manifest as the "
            "endpoints currently resolved for this Gate 3 run"
        )
    if gate2_endpoint.get("source_manifest_hash") != resolved.source_manifest_hash:
        raise DevelopmentPilotError(
            "Gate 2 run_meta.json does not bind the same source manifest as the endpoints "
            "currently resolved for this Gate 3 run"
        )

    return {"summary": summary, "run_meta_endpoint": gate2_endpoint}


# Activation-run prerequisite (protocol Section 6)


def _stream_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_activation_run(pair: str, activation_run_dir: Path, resolved: ResolvedEndpointSet) -> Dict[str, Any]:
    """A `primary_v1`, endpoint-backed, correctly-paired, manifest-verified construction
    activation run is required for each pair before any vector is built, any output is
    created, or any tokenizer/model/GPU resource is touched.

    Cross-checks `run_meta.json`'s own recorded facts (protocol profile, endpoint
    binding, HarmfulQA partition/record count/anchor, loading policy) against what this
    run independently resolved, then independently re-validates the activation tensors
    and their ordered identities against the frozen manifest via
    `steer_sweep.validate_construction_activations` -- never trusts the tensor file
    just because a passing-looking `run_meta.json` sits beside it.
    """
    acts_path = activation_run_dir / "activations.pt"
    meta_path = activation_run_dir / "run_meta.json"
    if not acts_path.exists():
        raise DevelopmentPilotError(f"pair {pair}: activations.pt not found at {acts_path}")
    if not meta_path.exists():
        raise DevelopmentPilotError(f"pair {pair}: run_meta.json not found at {meta_path}")

    run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    verify_run_metadata_identity(run_meta, f"pair {pair} activation run ({meta_path})")

    if run_meta.get("protocol_profile") != "primary_v1":
        raise DevelopmentPilotError(
            f"pair {pair}: activation run protocol_profile is {run_meta.get('protocol_profile')!r}, "
            "expected 'primary_v1'"
        )
    if run_meta.get("endpoint_backed") is not True:
        raise DevelopmentPilotError(f"pair {pair}: activation run is not endpoint_backed")

    endpoint = run_meta.get("endpoint") or {}
    if endpoint.get("pair") != pair:
        raise DevelopmentPilotError(f"pair {pair}: activation run's endpoint.pair is {endpoint.get('pair')!r}")
    if endpoint.get("candidate_manifest_hash") != resolved.candidate_manifest_hash:
        raise DevelopmentPilotError(
            f"pair {pair}: activation run candidate_manifest_hash does not match the currently resolved endpoints"
        )
    if endpoint.get("source_manifest_hash") != resolved.source_manifest_hash:
        raise DevelopmentPilotError(
            f"pair {pair}: activation run source_manifest_hash does not match the currently resolved endpoints"
        )

    it_role, dpo_role = roles_for_pair(pair)
    roles = endpoint.get("roles") or {}
    if roles.get("it") != it_role or roles.get("dpo") != dpo_role:
        raise DevelopmentPilotError(
            f"pair {pair}: activation run roles {roles!r} do not correspond to pair {pair} "
            f"(expected it={it_role!r}, dpo={dpo_role!r})"
        )

    if run_meta.get("harmfulqa_partition") != PRIMARY_CONSTRUCTION_PARTITION:
        raise DevelopmentPilotError(
            f"pair {pair}: activation run harmfulqa_partition is {run_meta.get('harmfulqa_partition')!r}, "
            f"expected {PRIMARY_CONSTRUCTION_PARTITION!r}"
        )
    if run_meta.get("harmfulqa_record_count") != PRIMARY_CONSTRUCTION_RECORD_COUNT:
        raise DevelopmentPilotError(
            f"pair {pair}: activation run harmfulqa_record_count is "
            f"{run_meta.get('harmfulqa_record_count')!r}, expected {PRIMARY_CONSTRUCTION_RECORD_COUNT}"
        )
    token_position = ((run_meta.get("config") or {}).get("eval") or {}).get("token_position")
    if token_position != PRIMARY_TOKEN_POSITION:
        raise DevelopmentPilotError(
            f"pair {pair}: activation run token_position is {token_position!r}, expected {PRIMARY_TOKEN_POSITION!r}"
        )
    if run_meta.get("model_loading_policy") != ENDPOINT_LOADING_POLICY:
        raise DevelopmentPilotError(
            f"pair {pair}: activation run model_loading_policy is {run_meta.get('model_loading_policy')!r}, "
            f"expected {ENDPOINT_LOADING_POLICY!r} (safe local loading)"
        )

    blob = torch.load(acts_path, map_location="cpu")
    steer_sweep.validate_construction_activations(blob)  # frozen-manifest identity + it/dpo shape agreement

    sha256 = _stream_sha256(acts_path)
    size_bytes = acts_path.stat().st_size

    return {
        "blob": blob, "run_meta": run_meta, "acts_path": acts_path,
        "sha256": sha256, "size_bytes": size_bytes,
    }


# M--A lineage prerequisite (protocol Section 4)


def verify_m_minus_a_lineage(resolved: ResolvedEndpointSet) -> Dict[str, bool]:
    """M--A's own resolved endpoint metadata carries a `merge.flip_lineage` record with
    two frozen invariants: it was never independently verified for label-swap lineage,
    and it was never marked confirmatory-eligible. Both must still read exactly
    `False` before this pilot ever generates from it -- a manifest silently claiming
    otherwise would let an unverified checkpoint masquerade as confirmatory. Returns the
    two verified booleans, for propagation into run metadata and every M--A record.
    """
    merge = resolved.roles["M--A"].merge or {}
    flip_lineage = merge.get("flip_lineage") or {}
    label_swap_lineage_verified = flip_lineage.get("label_swap_lineage_verified")
    confirmatory_eligible = flip_lineage.get("confirmatory_eligible")
    if label_swap_lineage_verified is not False or confirmatory_eligible is not False:
        raise DevelopmentPilotError(
            "M--A's flip_lineage does not match the frozen unverified-exploratory invariant "
            f"(both must be exactly False): label_swap_lineage_verified={label_swap_lineage_verified!r}, "
            f"confirmatory_eligible={confirmatory_eligible!r}"
        )
    return {"label_swap_lineage_verified": label_swap_lineage_verified, "confirmatory_eligible": confirmatory_eligible}


# IO helpers (atomic writes; a staging directory plus one final rename, so a crash
# mid-write can never leave a partial JSONL, summary, or CSV -- mirrors
# generation_smoke.py's own IO helpers, duplicated rather than imported since both
# scripts are otherwise independent and these are a few lines each)


def _release_log_handlers_under(directory: Path) -> None:
    directory = Path(directory).resolve()
    root = logging.getLogger()
    for handler in list(root.handlers):
        base_filename = getattr(handler, "baseFilename", None)
        if base_filename and Path(base_filename).resolve().is_relative_to(directory):
            handler.close()
            root.removeHandler(handler)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def write_jsonl_atomic(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    _atomic_write_text(path, text)


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Per-arm generation


def generate_arm_records(
    arm: Mapping[str, Any], model: Any, tokenizer: Any, terminators: Sequence[int],
    rendered_prompts: Sequence[str], rendered_hashes: Sequence[str],
    pilot_records: Sequence[Dict[str, Any]], vectors_by_source: Mapping[str, Dict[int, torch.Tensor]],
    vector_hashes: Mapping[str, str], batch_size: int, harmfulqa_manifest_hash: str,
    m_minus_a_lineage: Optional[Dict[str, bool]] = None,
) -> List[Dict[str, Any]]:
    """Generate one arm's 50 responses and assemble one complete, judge-ready,
    `score_sweep.py`-compatible record per response.

    Every field describing the intervention (protocol/intervention profile,
    coefficient status, vector method/normalisation/positions/preserve_norm, layers)
    comes directly from the arm's own plan entry -- an unsteered baseline's `None`/`[]`
    values propagate through exactly as declared, rather than the frozen primary
    constants being applied uniformly regardless of whether steering happened.
    `m_minus_a_lineage`'s two verified booleans are attached only to M--A's own records.
    """
    vector_source = arm["vector_source"]
    ctx = None
    if vector_source is not None:
        vec = vectors_by_source[vector_source]
        coeff = arm["coefficient"]
        positions = arm["positions"]
        preserve_norm = arm["preserve_norm"]

        def ctx():  # noqa: E306 -- rebuilt per batch so hooks never outlive it
            return ActivationSteering(
                model, vec, coefficient=coeff, positions=positions,
                preserve_norm=preserve_norm, mode=PILOT_MODE,
            )

    results = generate_batched(
        model, tokenizer, rendered_prompts,
        max_new_tokens=PILOT_MAX_NEW_TOKENS, batch_size=batch_size, max_input_length=PILOT_MAX_INPUT_LENGTH,
        context=ctx, desc=arm["arm_id"], return_metadata=True,
    )
    if len(results) != len(pilot_records):
        raise DevelopmentPilotError(
            f"{arm['arm_id']}: generate_batched returned {len(results)} results for "
            f"{len(pilot_records)} records; refusing to write a silently truncated batch"
        )

    is_m_minus_a = arm["endpoint_role"] == "M--A"
    out: List[Dict[str, Any]] = []
    for rec, sha, result in zip(pilot_records, rendered_hashes, results):
        validity = check_validity(result.text)
        record = {
            "id": rec["source_id"], "source_id": rec["source_id"], "prompt": rec["prompt"],
            "prompt_hash": rec["prompt_hash"], "permuted_position": rec["permuted_position"],
            "harmfulqa_partition": GATE3_PARTITION, "harmfulqa_manifest_hash": harmfulqa_manifest_hash,
            "pair": arm["pair"], "endpoint_role": arm["endpoint_role"],
            "arm_id": arm["arm_id"], "arm_type": arm["arm_type"],
            "run_profile": PILOT_RUN_PROFILE, "protocol_profile": arm["protocol_profile"],
            "intervention_profile": arm["intervention_profile"],
            "intervention_operation": arm["intervention_operation"],
            "coefficient": arm["coefficient"], "lambda": arm["coefficient"],
            "coefficient_status": arm["coefficient_status"],
            "vector_source": vector_source, "vector_hash": vector_hashes.get(vector_source) if vector_source else None,
            "random_vector_seed": arm["random_vector_seed"],
            "layers": list(arm["layers"]), "vector_method": arm["vector_method"],
            "normalization": arm["normalization"], "positions": arm["positions"], "preserve_norm": arm["preserve_norm"],
            "rendered_prompt_sha256": sha,
            "response": result.text, "generated_token_count": result.generated_token_count,
            "stop_reason": result.stop_reason, "stop_token_id": result.stop_token_id,
            "terminator_ids": list(terminators), "has_post_terminator_continuation": result.has_post_terminator_continuation,
            "validity_ok": validity.ok, "validity_reason": validity.reason, "validity_repetition": validity.repetition,
            "lineage_flag": arm["lineage_flag"], "batch_size": batch_size,
        }
        if is_m_minus_a and m_minus_a_lineage is not None:
            record["label_swap_lineage_verified"] = m_minus_a_lineage["label_swap_lineage_verified"]
            record["confirmatory_eligible"] = m_minus_a_lineage["confirmatory_eligible"]
        out.append(record)
    return out


# Gate 3 summary and inspection table

#: Every judge-ready field a record must carry, regardless of arm type -- checked as
#: part of automated integrity, not merely documented in a docstring.
BASE_REQUIRED_RECORD_FIELDS: Tuple[str, ...] = (
    "id", "source_id", "prompt", "prompt_hash", "permuted_position",
    "harmfulqa_partition", "harmfulqa_manifest_hash", "pair", "endpoint_role",
    "arm_id", "arm_type", "run_profile", "protocol_profile", "intervention_profile",
    "intervention_operation", "coefficient", "lambda", "coefficient_status",
    "vector_source", "vector_hash", "random_vector_seed", "layers", "vector_method",
    "normalization", "positions", "preserve_norm", "rendered_prompt_sha256",
    "response", "generated_token_count", "stop_reason", "stop_token_id",
    "terminator_ids", "has_post_terminator_continuation",
    "validity_ok", "validity_reason", "validity_repetition", "lineage_flag", "batch_size",
)
#: Only M--A's own records carry these -- "no other arm should carry flip-lineage
#: fields" is enforced here, not merely stated.
M_MINUS_A_ONLY_FIELDS: Tuple[str, ...] = ("label_swap_lineage_verified", "confirmatory_eligible")

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

#: Every arm-plan-derived field a row must match exactly, and the arm-dict key it is
#: compared against -- "run_profile" has no arm-dict counterpart (it is the one
#: constant every record carries, PILOT_RUN_PROFILE) so it is handled separately below.
_ARM_PROVENANCE_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("protocol_profile", "protocol_profile"), ("intervention_profile", "intervention_profile"),
    ("intervention_operation", "intervention_operation"), ("coefficient", "coefficient"),
    ("coefficient_status", "coefficient_status"), ("vector_source", "vector_source"),
    ("random_vector_seed", "random_vector_seed"), ("vector_method", "vector_method"),
    ("normalization", "normalization"), ("positions", "positions"), ("preserve_norm", "preserve_norm"),
)


def _record_provenance_ok(
    r: Mapping[str, Any], arm: Mapping[str, Any], expected_vector_hash: Optional[str],
    expected_manifest_hash: Any, expected_layers: List[int], batch_size: int, is_m_minus_a: bool,
) -> bool:
    """One row's provenance against its frozen `ARM_PLAN` entry and the externally
    resolved facts (`expected_vector_hash`/`expected_manifest_hash`/`batch_size`) this
    run actually produced -- never inferred from the row itself or its siblings, so a
    uniformly wrong field across every row of an arm is still caught.
    """
    if r.get("run_profile") != PILOT_RUN_PROFILE:
        return False
    for record_key, arm_key in _ARM_PROVENANCE_FIELDS:
        if r.get(record_key) != arm[arm_key]:
            return False
    if r.get("lambda") != arm["coefficient"]:
        return False
    if r.get("layers") != expected_layers:
        return False
    if r.get("batch_size") != batch_size:
        return False
    if r.get("id") != r.get("source_id"):
        return False
    if r.get("harmfulqa_partition") != GATE3_PARTITION:
        return False
    if r.get("harmfulqa_manifest_hash") != expected_manifest_hash:
        return False
    rendered_hash = r.get("rendered_prompt_sha256")
    if not isinstance(rendered_hash, str) or not _SHA256_HEX_RE.match(rendered_hash):
        return False
    if arm["vector_source"] is not None:
        if not r.get("vector_hash") or r.get("vector_hash") != expected_vector_hash:
            return False
    elif r.get("vector_hash") is not None:
        return False
    terminator_ids = r.get("terminator_ids")
    if not terminator_ids:
        return False
    stop_token_id = r.get("stop_token_id")
    if stop_token_id is not None and stop_token_id not in terminator_ids:
        return False
    if is_m_minus_a:
        if r.get("label_swap_lineage_verified") is not False or r.get("confirmatory_eligible") is not False:
            return False
    else:
        if "label_swap_lineage_verified" in r or "confirmatory_eligible" in r:
            return False
    return True


def evaluate_arm_integrity(
    arm: Mapping[str, Any], records: Sequence[Dict[str, Any]], pilot_records: Sequence[Dict[str, Any]],
    vector_hashes: Mapping[str, str], batch_size: int,
) -> Dict[str, Any]:
    """Record/schema/provenance/stopping integrity for one arm -- never an effect-based
    (harmfulness/refusal/quality) judgement, which this script has no access to.

    `arm_integrity_pass` requires every one of: exactly 50 records; the exact ordered
    (source_id, prompt_hash, permuted_position) identity triples; a stable arm_id/pair/
    endpoint_role/arm_type on every row; the full required schema present (and, for
    M--A, exactly its two extra lineage fields -- present there, absent everywhere
    else); no duplicate ids; zero invalid, empty, unknown-stop, max-token-stop, or
    post-terminator-continuation records; and zero rows whose per-row provenance
    disagrees with the frozen `ARM_PLAN` entry and the externally resolved
    `vector_hashes`/`batch_size` (`_record_provenance_ok`). All of these are still
    *reported* even when they fail -- nothing here drops or hides a record.
    """
    arm_id, pair, role, arm_type = arm["arm_id"], arm["pair"], arm["endpoint_role"], arm["arm_type"]
    is_m_minus_a = role == "M--A"

    expected_triples = [(r["source_id"], r["prompt_hash"], r["permuted_position"]) for r in pilot_records]
    actual_triples = [(r.get("source_id"), r.get("prompt_hash"), r.get("permuted_position")) for r in records]
    identity_order_ok = actual_triples == expected_triples

    stable_fields_ok = all(
        r.get("arm_id") == arm_id and r.get("pair") == pair
        and r.get("endpoint_role") == role and r.get("arm_type") == arm_type
        for r in records
    )

    required_fields = set(BASE_REQUIRED_RECORD_FIELDS) | (set(M_MINUS_A_ONLY_FIELDS) if is_m_minus_a else set())
    forbidden_fields = set() if is_m_minus_a else set(M_MINUS_A_ONLY_FIELDS)
    schema_ok = all(
        required_fields.issubset(r.keys()) and forbidden_fields.isdisjoint(r.keys())
        for r in records
    )

    ids = [r.get("id") for r in records]
    duplicate_ids = len(set(ids)) != len(ids)

    n = len(records)
    n_invalid = sum(1 for r in records if not r.get("validity_ok"))
    n_unknown = sum(1 for r in records if r.get("stop_reason") == "unknown")
    n_max_new_tokens = sum(1 for r in records if r.get("stop_reason") == "max_new_tokens")
    n_post_terminator = sum(1 for r in records if r.get("has_post_terminator_continuation"))
    n_empty = sum(1 for r in records if not (r.get("response") or "").strip())

    expected_vector_hash = vector_hashes.get(arm["vector_source"]) if arm["vector_source"] else None
    expected_manifest_hash = pilot_records[0]["manifest_hash"] if pilot_records else None
    expected_layers = list(arm["layers"])
    n_provenance_mismatch = sum(
        1 for r in records
        if not _record_provenance_ok(r, arm, expected_vector_hash, expected_manifest_hash, expected_layers, batch_size, is_m_minus_a)
    )

    stop_reason_distribution: Dict[str, int] = {}
    for r in records:
        stop_reason_distribution[r.get("stop_reason")] = stop_reason_distribution.get(r.get("stop_reason"), 0) + 1

    arm_integrity_pass = (
        n == PILOT_RECORD_COUNT and identity_order_ok and stable_fields_ok and schema_ok
        and not duplicate_ids and n_invalid == 0 and n_unknown == 0
        and n_max_new_tokens == 0 and n_post_terminator == 0 and n_empty == 0
        and n_provenance_mismatch == 0
    )

    return {
        "expected_record_count": PILOT_RECORD_COUNT, "actual_record_count": n,
        "identity_order_ok": identity_order_ok, "stable_fields_ok": stable_fields_ok,
        "schema_ok": schema_ok, "duplicate_ids": duplicate_ids,
        "n_invalid": n_invalid, "invalid_rate": (n_invalid / n) if n else None,
        "n_stop_reason_unknown": n_unknown, "n_max_new_tokens": n_max_new_tokens,
        "n_post_terminator_continuation": n_post_terminator,
        "n_empty_or_missing_response": n_empty,
        "n_provenance_mismatch": n_provenance_mismatch,
        "stop_reason_distribution": stop_reason_distribution,
        "arm_integrity_pass": arm_integrity_pass,
    }


def build_gate3_summary(
    arm_records: Mapping[str, Sequence[Dict[str, Any]]], pilot_records: Sequence[Dict[str, Any]],
    batch_size: int, vector_hashes: Mapping[str, str],
) -> Dict[str, Any]:
    """Structural integrity only -- expected/actual counts, exact identity/order,
    schema, per-row provenance, and stopping/invalidity rates with their denominators
    kept visible, never dropped. Never an effect-based (which arm "looks better")
    pass/fail call, and never a signal that continues the pipeline:
    `manual_review_required` is always true and `automatically_continue_to_calibration`
    is always false.

    `automated_integrity_pass` requires the exact `ARM_PLAN` arm-id set (no missing, no
    extra arm) *and* every arm's own `arm_integrity_pass`. A missing/extra arm is
    reported by name in `missing_arms`/`unexpected_arms` regardless of the per-arm
    results, since an unexpected arm has no frozen plan entry to score against.
    """
    expected_arm_ids = {arm["arm_id"] for arm in ARM_PLAN}
    actual_arm_ids = set(arm_records)
    missing_arms = sorted(expected_arm_ids - actual_arm_ids)
    unexpected_arms = sorted(actual_arm_ids - expected_arm_ids)
    arm_set_ok = not missing_arms and not unexpected_arms

    arm_by_id = {arm["arm_id"]: arm for arm in ARM_PLAN}
    arms = {
        arm_id: evaluate_arm_integrity(arm_by_id[arm_id], records, pilot_records, vector_hashes, batch_size)
        for arm_id, records in arm_records.items() if arm_id in arm_by_id
    }
    automated_integrity_pass = arm_set_ok and all(a["arm_integrity_pass"] for a in arms.values())

    return {
        "gate": "gate3_development_pilot",
        "batch_size": batch_size,
        "arm_set_ok": arm_set_ok, "missing_arms": missing_arms, "unexpected_arms": unexpected_arms,
        "arms": arms,
        "automated_integrity_pass": automated_integrity_pass,
        "manual_review_required": True,
        "automatically_continue_to_calibration": False,
    }


def write_inspection_csv(path: Path, arm_records: Mapping[str, Sequence[Dict[str, Any]]]) -> None:
    """One row per prompt per arm, ordered deterministically by pair, arm-plan order,
    then `permuted_position` -- never by dict/filesystem iteration order."""
    arm_order = {arm["arm_id"]: i for i, arm in enumerate(ARM_PLAN)}
    rows = []
    for arm_id, records in arm_records.items():
        for r in records:
            rows.append({
                "pair": r["pair"], "arm_id": r["arm_id"], "arm_type": r["arm_type"],
                "endpoint_role": r["endpoint_role"],
                "coefficient": r["coefficient"], "lambda": r["lambda"],
                "permuted_position": r["permuted_position"], "source_id": r["source_id"],
                "prompt": r["prompt"], "response": r["response"],
                "validity_ok": r["validity_ok"], "validity_reason": r["validity_reason"],
                "stop_reason": r["stop_reason"], "stop_token_id": r["stop_token_id"],
                "generated_token_count": r["generated_token_count"],
                "has_post_terminator_continuation": r["has_post_terminator_continuation"],
                "_arm_order": arm_order[arm_id],
            })
    df = pd.DataFrame(rows)
    df = df.sort_values(["pair", "_arm_order", "permuted_position"]).drop(columns=["_arm_order"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# CLI


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate 3: frozen 50-prompt development pilot.")
    parser.add_argument("--endpoint-manifest", required=True)
    parser.add_argument("--endpoint-bundle-root", required=True)
    parser.add_argument(
        "--endpoint-source", action="append", default=[], metavar="ARTIFACT_ID=LOCAL_ROOT",
        help="Repeatable; requires exactly pair_a_sft, pair_a_dpo, and pair_b_sft.",
    )
    parser.add_argument("--gate2-dir", required=True, help="A completed, passing Gate 2 (generation_smoke.py) output directory.")
    parser.add_argument("--pair-a-activation-run", required=True, help="Pair A primary_v1 construction activation-run directory.")
    parser.add_argument("--pair-b-activation-run", required=True, help="Pair B primary_v1 construction activation-run directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--batch-size", type=int, default=None, metavar="N",
        help=f"Generation batch size, for hardware efficiency only; must be positive. "
             f"Defaults to {PILOT_DEFAULT_BATCH_SIZE}. Always recorded in run_meta.json, "
             "every generated record, and gate3_summary.json.",
    )
    parser.add_argument("--hourly-rate", type=float, default=None)
    return parser


def resolve_batch_size(cli_value: Optional[int]) -> int:
    """The frozen default, or a positive CLI override -- never an unrecorded automatic
    estimate. Rejected before any endpoint, dataset, output, tokenizer, model, or GPU
    access."""
    batch_size = cli_value if cli_value is not None else PILOT_DEFAULT_BATCH_SIZE
    if batch_size <= 0:
        raise DevelopmentPilotError(f"--batch-size must be positive, got {batch_size}")
    return batch_size


def main() -> int:
    args = _build_arg_parser().parse_args()

    # A pure CLI-shape check -- before endpoint resolution, dataset loading, output
    # creation, or any tokenizer/model/GPU access, since it needs nothing from them.
    batch_size = resolve_batch_size(args.batch_size)

    # Endpoint resolution and complete verification -- before anything below touches
    # output, a seed, a vector, or a tokenizer/model/GPU resource. A mismatch has no
    # side effects.
    source_roots = parse_source_mappings(args.endpoint_source)
    resolved = resolve_all_roles(args.endpoint_manifest, source_roots, args.endpoint_bundle_root)

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        print(f"FAIL: {output_dir} already exists; refusing to overwrite an existing output directory", file=sys.stderr)
        return 1

    # Read-only preparation: loading the frozen development partition and selecting the
    # pilot's 50 records mutates nothing on disk.
    development_records = load_harmfulqa_partition(GATE3_PARTITION)
    pilot_records = select_pilot_records(development_records)
    first_ten = generation_smoke.select_smoke_records(development_records)
    harmfulqa_manifest_hash = pilot_records[0]["manifest_hash"]

    # Gate 2, activation-run, and M--A lineage prerequisites -- still before
    # set_all_seeds (CUDA), output creation, vector construction, or any
    # tokenizer/model/GPU access.
    gate2_info = verify_gate2_prerequisite(Path(args.gate2_dir), resolved, first_ten)
    acts_a = verify_activation_run("A", Path(args.pair_a_activation_run), resolved)
    acts_b = verify_activation_run("B", Path(args.pair_b_activation_run), resolved)
    m_minus_a_lineage = verify_m_minus_a_lineage(resolved)

    # The general run seed is frozen (protocol Section 7); there is no --seed CLI flag.
    set_all_seeds(PILOT_SEED)

    staging_dir = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    try:
        staging_dir.mkdir(parents=True)

        vector_a = build_vectors(acts_a["acts_path"], method=PILOT_VECTOR_METHOD, layers=list(PILOT_LAYERS), normalise=PILOT_VECTOR_NORMALISE)
        vector_b = build_vectors(acts_b["acts_path"], method=PILOT_VECTOR_METHOD, layers=list(PILOT_LAYERS), normalise=PILOT_VECTOR_NORMALISE)
        random_vector_a = random_vectors_like(vector_a, seed=PILOT_RANDOM_SEED)

        # Recorded as paths relative to the run root, not the staging directory -- the
        # staging directory is renamed to the final output directory below, so an
        # absolute staging path would no longer exist by the time anyone reads
        # run_meta.json.
        vector_relpaths = {
            "real_mean_pair_A": Path("vectors") / "pair_A_mean.pt",
            "real_mean_pair_B": Path("vectors") / "pair_B_mean.pt",
            "random_seed_11_pair_A": Path("vectors") / "pair_A_random_s11.pt",
        }
        vectors_dir = staging_dir / "vectors"
        vectors_dir.mkdir(parents=True, exist_ok=True)
        vector_abspaths = {name: staging_dir / relpath for name, relpath in vector_relpaths.items()}
        torch.save(vector_a, vector_abspaths["real_mean_pair_A"])
        torch.save(vector_b, vector_abspaths["real_mean_pair_B"])
        torch.save(random_vector_a, vector_abspaths["random_seed_11_pair_A"])
        vector_hashes = {name: _stream_sha256(path) for name, path in vector_abspaths.items()}

        vectors_by_source = {
            "real_mean_pair_A": vector_a, "real_mean_pair_B": vector_b, "random_seed_11_pair_A": random_vector_a,
        }

        def _vector_metadata(name: str, vector_source: str, method: str, random_seed: Optional[int]) -> Dict[str, Any]:
            return {
                "path": vector_relpaths[name].as_posix(), "sha256": vector_hashes[name],
                "size_bytes": vector_abspaths[name].stat().st_size,
                "vector_source": vector_source, "layers": list(PILOT_LAYERS),
                "method": method, "normalization": PILOT_VECTOR_NORMALISE, "random_seed": random_seed,
            }

        vector_metadata = {
            "real_mean_pair_A": _vector_metadata("real_mean_pair_A", "real_mean_pair_A", PILOT_VECTOR_METHOD, None),
            "real_mean_pair_B": _vector_metadata("real_mean_pair_B", "real_mean_pair_B", PILOT_VECTOR_METHOD, None),
            "random_seed_11_pair_A": _vector_metadata(
                "random_seed_11_pair_A", "random_seed_11_pair_A", PILOT_RANDOM_VECTOR_METHOD, PILOT_RANDOM_SEED,
            ),
        }

        run_meta_extra = {
            "gate": "gate3_development_pilot",
            "pilot_records": [
                {"source_id": r["source_id"], "prompt_hash": r["prompt_hash"], "permuted_position": r["permuted_position"]}
                for r in pilot_records
            ],
            "harmfulqa_partition": GATE3_PARTITION,
            "harmfulqa_manifest_hash": harmfulqa_manifest_hash,
            "harmfulqa_record_count": len(pilot_records),
            "candidate_manifest_hash": resolved.candidate_manifest_hash,
            "source_manifest_hash": resolved.source_manifest_hash,
            "endpoint": resolved.run_metadata(),
            "gate2": {
                "gate2_dir": str(args.gate2_dir),
                "summary": gate2_info["summary"],
                "run_meta_endpoint": gate2_info["run_meta_endpoint"],
            },
            "activation_runs": {
                "A": {
                    "path": str(acts_a["acts_path"]), "sha256": acts_a["sha256"], "size_bytes": acts_a["size_bytes"],
                    "protocol_profile": acts_a["run_meta"].get("protocol_profile"), "endpoint": acts_a["run_meta"].get("endpoint"),
                },
                "B": {
                    "path": str(acts_b["acts_path"]), "sha256": acts_b["sha256"], "size_bytes": acts_b["size_bytes"],
                    "protocol_profile": acts_b["run_meta"].get("protocol_profile"), "endpoint": acts_b["run_meta"].get("endpoint"),
                },
            },
            "vectors": vector_metadata,
            "m_minus_a_lineage": m_minus_a_lineage,
            "arm_plan": [
                (dict(arm, **m_minus_a_lineage) if arm["endpoint_role"] == "M--A" else dict(arm))
                for arm in ARM_PLAN
            ],
            "decoding": {
                "do_sample": False, "greedy": True,
                "max_new_tokens": PILOT_MAX_NEW_TOKENS, "max_input_length": PILOT_MAX_INPUT_LENGTH,
                "batch_size": batch_size, "sequential_endpoint_loading": True,
            },
            "intervention": {
                "layers": list(PILOT_LAYERS), "vector_method": PILOT_VECTOR_METHOD,
                "vector_normalise": PILOT_VECTOR_NORMALISE, "positions": PILOT_POSITIONS,
                "preserve_norm": PILOT_PRESERVE_NORM, "mode": PILOT_MODE,
                "coefficient": PILOT_COEFFICIENT, "coefficient_status": PILOT_COEFFICIENT_STATUS,
            },
        }
        write_run_metadata(staging_dir, config={"cli": vars(args)}, extra=run_meta_extra, argv=list(sys.argv))

        log = setup_logging(staging_dir / "development_pilot.log")
        log.info(f"Gate 3 development pilot: {len(pilot_records)} prompts from {GATE3_PARTITION}")

        pair_dirs = {
            "A": staging_dir / "judge_ready" / "pair_A",
            "B": staging_dir / "judge_ready" / "pair_B",
        }
        by_role = arms_by_role()
        arm_records: Dict[str, List[Dict[str, Any]]] = {}

        with GpuMonitor(staging_dir, hourly_rate=args.hourly_rate) as gpu:
            for role in ALL_ROLES:
                role_arms = by_role.get(role, [])
                if not role_arms:
                    continue
                ep = resolved.roles[role]
                log.info(f"{role}: loading tokenizer from {ep.local_path}")
                tokenizer = load_tokenizer(str(ep.local_path), **ENDPOINT_LOADING_POLICY)
                terminators = generation_terminators(tokenizer)
                if not terminators:
                    raise DevelopmentPilotError(
                        f"{role}: tokenizer has no usable EOS/end-of-turn terminator; refusing to load its model"
                    )
                rendered_prompts = build_chat_prompts(tokenizer, [r["prompt"] for r in pilot_records])
                rendered_hashes = [_sha256_text(p) for p in rendered_prompts]

                log.info(f"{role}: loading model from {ep.local_path}")
                model = load_model(str(ep.local_path), **ENDPOINT_LOADING_POLICY)

                for arm in role_arms:
                    records = generate_arm_records(
                        arm, model, tokenizer, terminators, rendered_prompts, rendered_hashes,
                        pilot_records, vectors_by_source, vector_hashes, batch_size, harmfulqa_manifest_hash,
                        m_minus_a_lineage=m_minus_a_lineage,
                    )
                    arm_records[arm["arm_id"]] = records
                    dest = pair_dirs[arm["pair"]] / f"{arm['arm_id']}.jsonl"
                    write_jsonl_atomic(dest, records)
                    log.info(f"{arm['arm_id']}: wrote {len(records)} records -> {dest}")

                # One model resident at a time -- released before the next role loads.
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        log.info(f"GPU usage: {json.dumps(gpu.summary())}")

        summary = build_gate3_summary(arm_records, pilot_records, batch_size, vector_hashes)
        write_json_atomic(staging_dir / "gate3_summary.json", summary)
        write_inspection_csv(staging_dir / "pilot_inspection.csv", arm_records)
    except Exception as e:  # noqa: BLE001 -- any failure anywhere here must promote nothing
        _release_log_handlers_under(staging_dir)
        shutil.rmtree(staging_dir, ignore_errors=True)
        print(f"FAIL: Gate 3 development pilot: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if output_dir.exists():  # race guard immediately before the one promotion
        _release_log_handlers_under(staging_dir)
        shutil.rmtree(staging_dir, ignore_errors=True)
        print(f"FAIL: {output_dir} appeared during the run; refusing to promote", file=sys.stderr)
        return 1
    _release_log_handlers_under(staging_dir)
    staging_dir.replace(output_dir)  # the single atomic promotion for the whole run

    # The completed output is preserved either way -- only the exit code reflects the
    # automated integrity verdict. manual_review_required and
    # automatically_continue_to_calibration are always set regardless (see
    # build_gate3_summary): a passing verdict here is not itself a launch signal for
    # any later stage.
    print(
        f"Gate 3 development pilot complete: see {output_dir}/gate3_summary.json "
        f"(automated_integrity_pass={summary['automated_integrity_pass']}; "
        "manual review required; this never auto-continues to calibration)"
    )
    return 0 if summary["automated_integrity_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
