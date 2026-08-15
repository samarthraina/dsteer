"""Frozen judge-protocol identity (protocol Section 10).

Mirrors the shape of `steering.artifact_identity` and `steering.splits`' HarmfulQA
manifest -- a schema-versioned JSON payload with a canonical SHA-256 over itself and a
frozen-identity pin against tampering or a merely self-consistent substitute -- applied
to the confirmatory Qwen3.5 judge protocol instead of a dataset partition or model
artifact set. Kept independent of `steering.judge`, `steering.config`, and
`steering.metrics` (no imports of them) so this module stays self-contained and
reviewable: callers pass in the live system-prompt/rubric text and frozen numeric
settings; this module only canonicalizes, hashes, and validates them.

This module never contacts a server, loads a model, or launches vLLM.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

SCHEMA_VERSION = 1

#: Version of the (global_seed, record_id, metric) -> request-seed derivation below.
#: Bumping this changes every derived seed, so it is itself part of the frozen identity
#: rather than an unpinned implementation detail.
SEED_DERIVATION_VERSION = 1

#: The five active metrics a `manifests/judge_protocol_v1.json` must hash rubrics for.
#: Matches `steering.metrics.ACTIVE_RUBRICS`'s keys.
ACTIVE_METRICS = ("quality", "helpfulness", "refusal", "harmfulness", "steering_shift")

#: The frozen structured-output schema (protocol Section 10): exactly `score` and
#: `reason`, no additional properties. Sent as a guided-decoding schema on every judge
#: request and enforced again by the parser.
STRUCTURED_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 10},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}

_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class JudgeIdentityError(ValueError):
    """A judge-protocol manifest-construction, structural-validation, or
    frozen-identity invariant was violated."""


# Canonical serialization / manifest hash (identical convention to artifact_identity)


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    """The exact serialization the manifest hash is computed over. Compact, key-sorted,
    ASCII-only, with `manifest_hash` itself excluded."""
    body = {k: v for k, v in payload.items() if k != "manifest_hash"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def compute_manifest_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def dumps_manifest(payload: Mapping[str, Any]) -> str:
    """Human-readable on-disk form. Deterministic for a given payload."""
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def save_manifest(payload: Mapping[str, Any], path: Union[str, Path]) -> bool:
    """Write `payload` to `path` if absent. If present, require byte-identical content.

    Returns True if bytes were written, False if an identical manifest already existed.
    Raises `JudgeIdentityError` if an existing manifest differs.
    """
    path = Path(path)
    text = dumps_manifest(payload)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != text:
            raise JudgeIdentityError(f"manifest at {path} already exists with different content; refusing to overwrite")
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
        raise JudgeIdentityError(f"manifest hash mismatch at {path}: stored {stored!r}, recomputed {recomputed!r}")
    return payload


def sha256_text(text: str) -> str:
    """SHA-256 hex of a rubric/system-prompt string, exactly as given -- no
    stripping or normalization, so a one-character wording change always changes the
    hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def served_model_alias(repository: str, revision: str) -> str:
    """A revision-bearing served-model alias: never the bare mutable repository name.

    `repository@revision` is deterministic and lossless (the full revision, not a
    prefix), so it can be recomputed from the frozen identity and compared exactly
    against what a server or a request actually used.
    """
    return f"{repository}@{revision}"


# Deterministic per-request seed derivation (protocol Section 10)


def derive_request_seed(
    global_seed: int, record_id: str, metric: str, version: int = SEED_DERIVATION_VERSION,
) -> int:
    """A deterministic request seed from (global_seed, record_id, metric).

    Repeated scoring of the same (record, metric) pair always derives the same seed;
    a different metric on the same record, or a different record on the same metric,
    must not collide in practice. Bounded to a nonnegative 31-bit range for broad
    client/server seed-field compatibility.
    """
    digest = hashlib.sha256(f"{version}:{global_seed}:{record_id}:{metric}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


# Structural validation


def _check_hex(label: str, value: Any, pattern: "re.Pattern[str]", length: int) -> None:
    if not isinstance(value, str) or not pattern.match(value):
        raise JudgeIdentityError(f"{label} must be {length} lowercase hex characters, got {value!r}")


def _check_unit_interval(label: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise JudgeIdentityError(f"{label} must be a number in [0, 1], got {value!r}")


def validate_manifest_structure(payload: Mapping[str, Any]) -> None:
    """Enforce the frozen judge-protocol manifest's structural invariants. Raises on the
    first violation found."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise JudgeIdentityError(f"schema_version must be {SCHEMA_VERSION}, got {payload.get('schema_version')!r}")

    judge = payload.get("judge")
    if not isinstance(judge, dict):
        raise JudgeIdentityError("payload['judge'] must be a dict")
    if not judge.get("repository"):
        raise JudgeIdentityError("judge.repository must be a nonempty string")
    _check_hex("judge.revision", judge.get("revision"), _HEX40_RE, 40)
    expected_alias = served_model_alias(judge["repository"], judge["revision"])
    if judge.get("served_model_alias") != expected_alias:
        raise JudgeIdentityError(
            f"judge.served_model_alias must be {expected_alias!r} (repository@revision), "
            f"got {judge.get('served_model_alias')!r}"
        )
    if not judge.get("vllm_version"):
        raise JudgeIdentityError("judge.vllm_version must be a nonempty string")
    for flag in ("text_only", "thinking_enabled"):
        if not isinstance(judge.get(flag), bool):
            raise JudgeIdentityError(f"judge.{flag} must be a bool, got {judge.get(flag)!r}")
    if judge.get("dtype") != "bfloat16":
        raise JudgeIdentityError(f"judge.dtype must be 'bfloat16', got {judge.get('dtype')!r}")
    if judge.get("quantization") is not None:
        raise JudgeIdentityError(f"judge.quantization must be null (no quantization), got {judge.get('quantization')!r}")

    sampling = payload.get("sampling")
    if not isinstance(sampling, dict):
        raise JudgeIdentityError("payload['sampling'] must be a dict")
    required_sampling = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")
    missing = [k for k in required_sampling if k not in sampling]
    if missing:
        raise JudgeIdentityError(f"sampling is missing {missing}")
    _check_unit_interval("sampling.top_p", sampling["top_p"])
    _check_unit_interval("sampling.min_p", sampling["min_p"])
    if isinstance(sampling["top_k"], bool) or not isinstance(sampling["top_k"], int) or sampling["top_k"] < 1:
        raise JudgeIdentityError(f"sampling.top_k must be a positive integer, got {sampling['top_k']!r}")
    if isinstance(sampling["temperature"], bool) or not isinstance(sampling["temperature"], (int, float)) or sampling["temperature"] < 0:
        raise JudgeIdentityError(f"sampling.temperature must be a nonnegative number, got {sampling['temperature']!r}")

    if not isinstance(payload.get("global_seed"), int) or isinstance(payload.get("global_seed"), bool):
        raise JudgeIdentityError(f"global_seed must be an int, got {payload.get('global_seed')!r}")
    if payload.get("seed_derivation_version") != SEED_DERIVATION_VERSION:
        raise JudgeIdentityError(
            f"seed_derivation_version must be {SEED_DERIVATION_VERSION}, got {payload.get('seed_derivation_version')!r}"
        )

    for field, minimum in (
        ("max_response_tokens", 1), ("max_model_len", 1), ("max_attempts", 1), ("concurrency", 1),
    ):
        v = payload.get(field)
        if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
            raise JudgeIdentityError(f"{field} must be an int >= {minimum}, got {v!r}")

    if not isinstance(payload.get("prefix_caching"), bool):
        raise JudgeIdentityError(f"prefix_caching must be a bool, got {payload.get('prefix_caching')!r}")

    if payload.get("structured_output_schema") != STRUCTURED_OUTPUT_SCHEMA:
        raise JudgeIdentityError("structured_output_schema does not match the frozen STRUCTURED_OUTPUT_SCHEMA")

    _check_hex("system_prompt_hash", payload.get("system_prompt_hash"), _HEX64_RE, 64)

    active_hashes = payload.get("active_rubric_hashes")
    if not isinstance(active_hashes, dict) or set(active_hashes) != set(ACTIVE_METRICS):
        raise JudgeIdentityError(
            f"active_rubric_hashes must have exactly the keys {sorted(ACTIVE_METRICS)}, got {sorted((active_hashes or {}).keys())}"
        )
    for metric, h in active_hashes.items():
        _check_hex(f"active_rubric_hashes[{metric!r}]", h, _HEX64_RE, 64)

    _check_hex("legacy_harmfulness_hash", payload.get("legacy_harmfulness_hash"), _HEX64_RE, 64)


def build_manifest(
    *,
    repository: str,
    revision: str,
    vllm_version: str,
    text_only: bool,
    thinking_enabled: bool,
    dtype: str,
    quantization: Optional[str],
    sampling: Mapping[str, Any],
    global_seed: int,
    seed_derivation_version: int,
    max_response_tokens: int,
    max_model_len: int,
    max_attempts: int,
    concurrency: int,
    prefix_caching: bool,
    structured_output_schema: Mapping[str, Any],
    system_prompt: str,
    active_rubrics: Mapping[str, str],
    legacy_harmfulness_rubric: str,
) -> Dict[str, Any]:
    """Build the schema-versioned, hash-computed judge-protocol manifest.

    Every hash is computed fresh from the live text passed in (`system_prompt`,
    `active_rubrics`, `legacy_harmfulness_rubric`) rather than accepted as a
    precomputed value -- there is exactly one place a rubric edit can fail to show up
    as a hash change, and it is not here. Validates structure and raises
    `JudgeIdentityError` rather than silently accepting a malformed payload.
    """
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "judge": {
            "repository": repository,
            "revision": revision,
            "served_model_alias": served_model_alias(repository, revision),
            "vllm_version": vllm_version,
            "text_only": text_only,
            "thinking_enabled": thinking_enabled,
            "dtype": dtype,
            "quantization": quantization,
        },
        "sampling": dict(sampling),
        "global_seed": global_seed,
        "seed_derivation_version": seed_derivation_version,
        "max_response_tokens": max_response_tokens,
        "max_model_len": max_model_len,
        "max_attempts": max_attempts,
        "concurrency": concurrency,
        "prefix_caching": prefix_caching,
        "structured_output_schema": dict(structured_output_schema),
        "system_prompt_hash": sha256_text(system_prompt),
        "active_rubric_hashes": {k: sha256_text(v) for k, v in sorted(active_rubrics.items())},
        "legacy_harmfulness_hash": sha256_text(legacy_harmfulness_rubric),
    }
    validate_manifest_structure(payload)
    payload["manifest_hash"] = compute_manifest_hash(payload)
    return payload


# Frozen identity (protocol Section 10)
#
# Pins the one judge protocol this task is reviewed against -- repository, revision,
# vLLM version, mode, precision, every sampling field, seed policy, limits, structured
# schema, and every rubric/system-prompt hash -- so a manifest that is merely
# internally self-consistent but is not this exact protocol is rejected.

FROZEN_JUDGE_IDENTITY: Dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "judge": {
        "repository": "Qwen/Qwen3.5-27B",
        "revision": "fc05daec18b0a78c049392ed2e771dde82bdf654",
        "served_model_alias": "Qwen/Qwen3.5-27B@fc05daec18b0a78c049392ed2e771dde82bdf654",
        "vllm_version": "0.26.0",
        "text_only": True,
        "thinking_enabled": False,
        "dtype": "bfloat16",
        "quantization": None,
    },
    "sampling": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    "global_seed": 20260815,
    "seed_derivation_version": SEED_DERIVATION_VERSION,
    "max_response_tokens": 256,
    "max_model_len": 8192,
    "max_attempts": 3,
    "concurrency": 32,
    "prefix_caching": True,
    "structured_output_schema": STRUCTURED_OUTPUT_SCHEMA,
    # Filled in from the real, committed source text by
    # scripts/build_judge_protocol_manifest.py; pinned here as literal hex so a rubric
    # or system-prompt edit that isn't reflected in a fresh build is caught as a
    # mismatch rather than silently accepted.
    "system_prompt_hash": "023d3fbb40f8dc7bc932b7a31192b5b32c48fb005f31586deabd7b46c73bbd2e",
    "active_rubric_hashes": {
        "harmfulness": "6276ce26f2284797970d2d37de584acd04e1ecf92a4a7d7776dea0b46a623e59",
        "helpfulness": "62e9e001ad9ee26655d8dbef9cc39ef10804b812cb9d93e2f0cbddb0865620e0",
        "quality": "22a7f608b89b994932614b2aa168ff3ff1731d963b81446715e41f4c69e32bed",
        "refusal": "76c1a315669f7af07d840db913c9c766783e98a68d20dd366ae325e575afccf2",
        "steering_shift": "aa10ee6d8dd29dea9b51b29c6c9977dadfc9ea52aead6baca839ff197e004694",
    },
    "legacy_harmfulness_hash": "0317421aab66fbcd329c9bcdceda2d00452c663d0c1eb624be9e01889486597e",
    "manifest_hash": "1c693c172a7e7a0f47fbda28988a2036ed9bff88273977664fc59b5206bf96bf",
}


def validate_frozen_identity(
    manifest: Mapping[str, Any], expected: Mapping[str, Any] = FROZEN_JUDGE_IDENTITY,
) -> None:
    """Check a loaded manifest against a frozen identity. Raises `JudgeIdentityError`
    listing every field that disagrees."""
    mismatches: List[str] = []

    def check(label: str, actual: Any, want: Any) -> None:
        if actual != want:
            mismatches.append(f"{label}: expected {want!r}, got {actual!r}")

    check("schema_version", manifest.get("schema_version"), expected["schema_version"])
    check("manifest_hash", manifest.get("manifest_hash"), expected["manifest_hash"])

    exp_judge = expected["judge"]
    actual_judge = manifest.get("judge") or {}
    for key, want in exp_judge.items():
        check(f"judge.{key}", actual_judge.get(key), want)

    exp_sampling = expected["sampling"]
    actual_sampling = manifest.get("sampling") or {}
    for key, want in exp_sampling.items():
        check(f"sampling.{key}", actual_sampling.get(key), want)

    for key in (
        "global_seed", "seed_derivation_version", "max_response_tokens", "max_model_len",
        "max_attempts", "concurrency", "prefix_caching", "structured_output_schema",
        "system_prompt_hash", "legacy_harmfulness_hash",
    ):
        check(key, manifest.get(key), expected[key])

    exp_hashes = expected["active_rubric_hashes"]
    actual_hashes = manifest.get("active_rubric_hashes") or {}
    for metric, want in exp_hashes.items():
        check(f"active_rubric_hashes[{metric!r}]", actual_hashes.get(metric), want)

    if mismatches:
        raise JudgeIdentityError(
            "manifest does not match the frozen judge-protocol identity: " + "; ".join(mismatches)
        )


def validate_live_evaluator_identity(
    manifest: Mapping[str, Any],
    *,
    system_prompt: str,
    active_rubrics: Mapping[str, str],
    legacy_harmfulness_rubric: str,
    structured_output_schema: Mapping[str, Any],
) -> None:
    """Recompute hashes from the live, currently-imported system prompt and rubric text
    and compare them against what `manifest` claims.

    `validate_frozen_identity` alone cannot catch drift here: it only checks a loaded
    manifest against the hardcoded `FROZEN_JUDGE_IDENTITY` pin, never against the code
    that is actually running. If a rubric or the system prompt is edited without
    rebuilding `manifests/judge_protocol_v1.json` (and, correspondingly,
    `FROZEN_JUDGE_IDENTITY`), a stale manifest would still self-verify and still match
    the stale frozen pin -- this is the check that catches "the manifest and the
    running code no longer agree." Raises `JudgeIdentityError` listing every field that
    disagrees.
    """
    mismatches: List[str] = []

    def check(label: str, actual: Any, want: Any) -> None:
        if actual != want:
            mismatches.append(f"{label}: expected {want!r}, got {actual!r}")

    check("system_prompt_hash", sha256_text(system_prompt), manifest.get("system_prompt_hash"))

    live_hashes = {k: sha256_text(v) for k, v in active_rubrics.items()}
    manifest_hashes = manifest.get("active_rubric_hashes") or {}
    if set(live_hashes) != set(manifest_hashes):
        mismatches.append(
            f"active_rubric_hashes metric set differs: live={sorted(live_hashes)}, "
            f"manifest={sorted(manifest_hashes)}"
        )
    else:
        for metric, want in manifest_hashes.items():
            check(f"active_rubric_hashes[{metric!r}]", live_hashes.get(metric), want)

    check("legacy_harmfulness_hash", sha256_text(legacy_harmfulness_rubric), manifest.get("legacy_harmfulness_hash"))
    check("structured_output_schema", dict(structured_output_schema), manifest.get("structured_output_schema"))

    if mismatches:
        raise JudgeIdentityError(
            "live evaluator identity (system prompt / active rubrics / legacy rubric / "
            "structured-output schema currently imported in code) does not match the "
            "loaded judge-protocol manifest: " + "; ".join(mismatches)
        )


def validate_judge_config(cfg: Any, expected: Mapping[str, Any] = FROZEN_JUDGE_IDENTITY) -> None:
    """Reject a `JudgeConfig` whose confirmatory-relevant fields do not exactly match
    the frozen identity -- the "reject non-frozen settings on the confirmatory path"
    guard. Duck-typed on attribute access so this module never imports `steering.config`.
    """
    mismatches: List[str] = []

    def check(label: str, actual: Any, want: Any) -> None:
        if actual != want:
            mismatches.append(f"{label}: expected {want!r}, got {actual!r}")

    exp_judge = expected["judge"]
    check("model_name", getattr(cfg, "model_name", None), exp_judge["served_model_alias"])
    check("max_tokens", getattr(cfg, "max_tokens", None), expected["max_response_tokens"])

    exp_sampling = expected["sampling"]
    check("temperature", getattr(cfg, "temperature", None), exp_sampling["temperature"])
    check("top_p", getattr(cfg, "top_p", None), exp_sampling["top_p"])
    check("top_k", getattr(cfg, "top_k", None), exp_sampling["top_k"])
    check("min_p", getattr(cfg, "min_p", None), exp_sampling["min_p"])
    check("presence_penalty", getattr(cfg, "presence_penalty", None), exp_sampling["presence_penalty"])
    check("repetition_penalty", getattr(cfg, "repetition_penalty", None), exp_sampling["repetition_penalty"])

    check("enable_thinking", getattr(cfg, "enable_thinking", None), exp_judge["thinking_enabled"])
    check("seed", getattr(cfg, "seed", None), expected["global_seed"])
    check("seed_derivation_version", getattr(cfg, "seed_derivation_version", None), expected["seed_derivation_version"])
    check("max_attempts", getattr(cfg, "max_attempts", None), expected["max_attempts"])
    check("max_score", getattr(cfg, "max_score", None), 10)
    check("use_logprobs", getattr(cfg, "use_logprobs", None), False)

    if mismatches:
        raise JudgeIdentityError(
            "JudgeConfig does not match the frozen judge-protocol identity: " + "; ".join(mismatches)
        )
