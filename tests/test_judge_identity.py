"""Guards on the frozen judge-protocol manifest (protocol Section 10, Task 010).

Mirrors the model-artifact manifest tests in shape: canonical-hash order-independence,
byte-for-byte rebuild, frozen-identity pinning against tampering or a merely
self-consistent substitute, structural-invariant rejection, and deterministic
per-request seed derivation. Entirely offline -- no network, no model, no GPU, no
judge server.

Run with:
    pytest tests/test_judge_identity.py -v
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from steering.judge import Judge
from steering.judge_identity import (
    ACTIVE_METRICS,
    FROZEN_JUDGE_IDENTITY,
    JudgeIdentityError,
    SEED_DERIVATION_VERSION,
    STRUCTURED_OUTPUT_SCHEMA,
    build_manifest,
    compute_manifest_hash,
    derive_request_seed,
    load_manifest,
    save_manifest,
    served_model_alias,
    validate_frozen_identity,
    validate_judge_config,
    validate_live_evaluator_identity,
    validate_manifest_structure,
)
from steering.metrics import ACTIVE_RUBRICS, LEGACY_HARMFULNESS_RUBRIC_PRE_REPAIR

import build_judge_protocol_manifest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_MANIFEST_PATH = REPO_ROOT / "manifests" / "judge_protocol_v1.json"


def _kwargs(**overrides):
    f = FROZEN_JUDGE_IDENTITY
    base = dict(
        repository=f["judge"]["repository"], revision=f["judge"]["revision"],
        vllm_version=f["judge"]["vllm_version"], text_only=f["judge"]["text_only"],
        thinking_enabled=f["judge"]["thinking_enabled"], dtype=f["judge"]["dtype"],
        quantization=f["judge"]["quantization"], sampling=dict(f["sampling"]),
        global_seed=f["global_seed"], seed_derivation_version=f["seed_derivation_version"],
        max_response_tokens=f["max_response_tokens"], max_model_len=f["max_model_len"],
        max_attempts=f["max_attempts"], concurrency=f["concurrency"], prefix_caching=f["prefix_caching"],
        structured_output_schema=dict(STRUCTURED_OUTPUT_SCHEMA), system_prompt=Judge.SYSTEM_PROMPT,
        active_rubrics=dict(ACTIVE_RUBRICS), legacy_harmfulness_rubric=LEGACY_HARMFULNESS_RUBRIC_PRE_REPAIR,
    )
    base.update(overrides)
    return base


def _real_manifest():
    return build_manifest(**_kwargs())


# Canonical hash order-independence


def test_hash_is_independent_of_sampling_dict_key_order():
    a = build_manifest(**_kwargs())
    reordered_sampling = dict(reversed(list(_kwargs()["sampling"].items())))
    b = build_manifest(**_kwargs(sampling=reordered_sampling))
    assert a["manifest_hash"] == b["manifest_hash"]


def test_hash_is_independent_of_active_rubrics_dict_key_order():
    a = build_manifest(**_kwargs())
    reordered = dict(reversed(list(ACTIVE_RUBRICS.items())))
    b = build_manifest(**_kwargs(active_rubrics=reordered))
    assert a["manifest_hash"] == b["manifest_hash"]


def test_same_inputs_reproduce_the_same_hash():
    a = build_manifest(**_kwargs())
    b = build_manifest(**_kwargs())
    assert a["manifest_hash"] == b["manifest_hash"]


def test_manifest_hash_matches_compute_manifest_hash():
    m = _real_manifest()
    assert m["manifest_hash"] == compute_manifest_hash(m)


# Committed manifest: load, self-verify, frozen identity, byte-for-byte rebuild


def test_committed_manifest_loads_verifies_and_passes_frozen_identity():
    manifest = load_manifest(COMMITTED_MANIFEST_PATH)  # raises if the stored hash is stale
    validate_manifest_structure(manifest)
    validate_frozen_identity(manifest)  # must not raise
    assert manifest["manifest_hash"] == FROZEN_JUDGE_IDENTITY["manifest_hash"]


def test_rebuilding_the_committed_manifest_is_byte_for_byte_identical():
    rebuilt = build_judge_protocol_manifest.build()
    on_disk = json.loads(COMMITTED_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert rebuilt == on_disk


def test_a_content_change_with_a_stale_stored_hash_fails_to_load(tmp_path):
    manifest = load_manifest(COMMITTED_MANIFEST_PATH)
    tampered = copy.deepcopy(manifest)
    tampered["judge"]["revision"] = "9" * 40  # manifest_hash left stale
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(JudgeIdentityError):
        load_manifest(path)


# Frozen-identity rejection: rubric/prompt text and every frozen configuration field


def test_a_one_character_rubric_change_is_rejected():
    tampered = dict(ACTIVE_RUBRICS)
    tampered["quality"] = tampered["quality"][:-2] + "X\n"  # one character different
    m = build_manifest(**_kwargs(active_rubrics=tampered))
    with pytest.raises(JudgeIdentityError):
        validate_frozen_identity(m)


def test_a_one_character_system_prompt_change_is_rejected():
    m = build_manifest(**_kwargs(system_prompt=Judge.SYSTEM_PROMPT + " "))
    with pytest.raises(JudgeIdentityError):
        validate_frozen_identity(m)


@pytest.mark.parametrize("field,value", [
    ("revision", "0" * 40),
    ("vllm_version", "0.25.0"),
    ("dtype", "float16"),
    ("thinking_enabled", True),
    ("text_only", False),
])
def test_a_frozen_judge_field_change_is_rejected(field, value):
    m = build_manifest(**_kwargs())
    tampered = copy.deepcopy(m)
    tampered["judge"][field] = value
    with pytest.raises(JudgeIdentityError):
        validate_frozen_identity(tampered)


@pytest.mark.parametrize("field,value", [
    ("temperature", 0.0), ("top_p", 1.0), ("top_k", 1), ("min_p", 0.1),
    ("presence_penalty", 0.0), ("repetition_penalty", 1.1),
])
def test_a_frozen_sampling_field_change_is_rejected(field, value):
    m = build_manifest(**_kwargs())
    tampered = copy.deepcopy(m)
    tampered["sampling"][field] = value
    with pytest.raises(JudgeIdentityError):
        validate_frozen_identity(tampered)


@pytest.mark.parametrize("field,value", [
    ("global_seed", 1), ("seed_derivation_version", 2), ("max_response_tokens", 512),
    ("max_model_len", 4096), ("max_attempts", 5), ("concurrency", 64), ("prefix_caching", False),
])
def test_a_frozen_top_level_field_change_is_rejected(field, value):
    m = build_manifest(**_kwargs())
    tampered = copy.deepcopy(m)
    tampered[field] = value
    with pytest.raises(JudgeIdentityError):
        validate_frozen_identity(tampered)


def test_a_structured_output_schema_change_is_rejected_at_build_time():
    """The schema is frozen exactly -- `build_manifest` itself refuses a schema that
    differs from `STRUCTURED_OUTPUT_SCHEMA`, before frozen-identity comparison is even
    reached."""
    bad_schema = dict(STRUCTURED_OUTPUT_SCHEMA)
    bad_schema["additionalProperties"] = True
    with pytest.raises(JudgeIdentityError):
        build_manifest(**_kwargs(structured_output_schema=bad_schema))


def test_an_alternate_self_consistent_manifest_still_fails_frozen_identity():
    """A manifest that is internally valid and hashes correctly, but uses a different
    (still well-formed) revision, must not be accepted."""
    alternate = build_manifest(**_kwargs(revision="1" * 40))
    validate_manifest_structure(alternate)  # passes on its own
    with pytest.raises(JudgeIdentityError):
        validate_frozen_identity(alternate)


# Legacy harmfulness is preserved but cannot enter active scoring


def test_legacy_harmfulness_hash_differs_from_the_active_harmfulness_hash():
    m = _real_manifest()
    assert m["legacy_harmfulness_hash"] != m["active_rubric_hashes"]["harmfulness"]


def test_legacy_harmfulness_text_is_not_the_active_harmfulness_text():
    assert LEGACY_HARMFULNESS_RUBRIC_PRE_REPAIR != ACTIVE_RUBRICS["harmfulness"]


def test_active_rubrics_has_exactly_the_five_frozen_metrics():
    assert set(ACTIVE_RUBRICS) == set(ACTIVE_METRICS)


def test_legacy_harmfulness_rubric_is_absent_from_active_rubrics_values():
    assert LEGACY_HARMFULNESS_RUBRIC_PRE_REPAIR not in ACTIVE_RUBRICS.values()


# Structural invariants


def test_missing_sampling_field_is_rejected():
    bad = _kwargs()
    del bad["sampling"]["top_k"]
    with pytest.raises(JudgeIdentityError):
        build_manifest(**bad)


def test_out_of_range_top_p_is_rejected():
    bad = _kwargs()
    bad["sampling"]["top_p"] = 1.5
    with pytest.raises(JudgeIdentityError):
        build_manifest(**bad)


def test_negative_top_k_is_rejected():
    bad = _kwargs()
    bad["sampling"]["top_k"] = -1
    with pytest.raises(JudgeIdentityError):
        build_manifest(**bad)


def test_a_boolean_top_k_is_rejected_even_though_bool_subclasses_int():
    bad = _kwargs()
    bad["sampling"]["top_k"] = True
    with pytest.raises(JudgeIdentityError):
        build_manifest(**bad)


def test_a_nonhex_revision_is_rejected():
    with pytest.raises(JudgeIdentityError):
        build_manifest(**_kwargs(revision="not-forty-hex-characters"))


def test_an_uppercase_revision_is_rejected():
    with pytest.raises(JudgeIdentityError):
        build_manifest(**_kwargs(revision="A" * 40))


def test_nonnull_quantization_is_rejected():
    with pytest.raises(JudgeIdentityError):
        build_manifest(**_kwargs(quantization="awq"))


def test_non_bfloat16_dtype_is_rejected():
    with pytest.raises(JudgeIdentityError):
        build_manifest(**_kwargs(dtype="float16"))


def test_zero_concurrency_is_rejected():
    with pytest.raises(JudgeIdentityError):
        build_manifest(**_kwargs(concurrency=0))


def test_served_model_alias_is_repository_at_revision():
    assert served_model_alias("org/repo", "a" * 40) == f"org/repo@{'a' * 40}"


# save_manifest immutability


def test_an_existing_different_manifest_is_not_overwritten(tmp_path):
    original = _real_manifest()
    path = tmp_path / "manifest.json"
    save_manifest(original, path)
    on_disk_before = path.read_text(encoding="utf-8")

    different = build_manifest(**_kwargs(revision="2" * 40))
    with pytest.raises(JudgeIdentityError):
        save_manifest(different, path)
    assert path.read_text(encoding="utf-8") == on_disk_before


def test_rebuilding_an_identical_manifest_does_not_rewrite(tmp_path):
    payload = _real_manifest()
    path = tmp_path / "manifest.json"
    save_manifest(payload, path)
    mtime_before = path.stat().st_mtime_ns
    written_again = save_manifest(payload, path)
    assert written_again is False
    assert path.stat().st_mtime_ns == mtime_before


# Deterministic per-request seed derivation


def test_identical_record_and_metric_pairs_get_identical_seeds():
    a = derive_request_seed(20260815, "harmfulqa-42", "refusal")
    b = derive_request_seed(20260815, "harmfulqa-42", "refusal")
    assert a == b


def test_different_metrics_on_the_same_record_do_not_collide():
    a = derive_request_seed(20260815, "harmfulqa-42", "refusal")
    b = derive_request_seed(20260815, "harmfulqa-42", "harmfulness")
    assert a != b


def test_different_records_on_the_same_metric_do_not_collide():
    a = derive_request_seed(20260815, "harmfulqa-42", "refusal")
    b = derive_request_seed(20260815, "harmfulqa-43", "refusal")
    assert a != b


def test_a_different_seed_derivation_version_changes_the_seed():
    a = derive_request_seed(20260815, "harmfulqa-42", "refusal", version=1)
    b = derive_request_seed(20260815, "harmfulqa-42", "refusal", version=2)
    assert a != b


def test_derived_seed_is_a_nonnegative_int():
    s = derive_request_seed(20260815, "harmfulqa-42", "refusal", version=SEED_DERIVATION_VERSION)
    assert isinstance(s, int)
    assert s >= 0


# validate_judge_config


class _FakeCfg:
    def __init__(self, **overrides):
        f = FROZEN_JUDGE_IDENTITY
        self.model_name = f["judge"]["served_model_alias"]
        self.max_tokens = f["max_response_tokens"]
        self.temperature = f["sampling"]["temperature"]
        self.top_p = f["sampling"]["top_p"]
        self.top_k = f["sampling"]["top_k"]
        self.min_p = f["sampling"]["min_p"]
        self.presence_penalty = f["sampling"]["presence_penalty"]
        self.repetition_penalty = f["sampling"]["repetition_penalty"]
        self.enable_thinking = f["judge"]["thinking_enabled"]
        self.seed = f["global_seed"]
        self.seed_derivation_version = f["seed_derivation_version"]
        self.max_attempts = f["max_attempts"]
        self.max_score = 10
        self.use_logprobs = False
        for k, v in overrides.items():
            setattr(self, k, v)


def test_a_matching_judge_config_passes():
    validate_judge_config(_FakeCfg())  # must not raise


def test_a_judge_config_with_the_bare_mutable_model_name_is_rejected():
    cfg = _FakeCfg(model_name=FROZEN_JUDGE_IDENTITY["judge"]["repository"])
    with pytest.raises(JudgeIdentityError):
        validate_judge_config(cfg)


def test_a_judge_config_with_a_drifted_temperature_is_rejected():
    cfg = _FakeCfg(temperature=0.0)
    with pytest.raises(JudgeIdentityError):
        validate_judge_config(cfg)


def test_a_judge_config_allowing_more_than_the_frozen_attempt_limit_is_rejected():
    cfg = _FakeCfg(max_attempts=10)
    with pytest.raises(JudgeIdentityError):
        validate_judge_config(cfg)


def test_a_judge_config_that_still_allows_logprobs_is_rejected():
    cfg = _FakeCfg(use_logprobs=True)
    with pytest.raises(JudgeIdentityError):
        validate_judge_config(cfg)


# validate_live_evaluator_identity (Task 010 correction): catches drift between the
# manifest on disk and the code actually running right now, which
# validate_frozen_identity alone cannot see (it only checks the manifest against the
# hardcoded FROZEN_JUDGE_IDENTITY pin, never against the live imports).


def _live_kwargs(**overrides):
    base = dict(
        system_prompt=Judge.SYSTEM_PROMPT, active_rubrics=dict(ACTIVE_RUBRICS),
        legacy_harmfulness_rubric=LEGACY_HARMFULNESS_RUBRIC_PRE_REPAIR,
        structured_output_schema=dict(STRUCTURED_OUTPUT_SCHEMA),
    )
    base.update(overrides)
    return base


def test_live_identity_matching_the_manifest_passes():
    validate_live_evaluator_identity(_real_manifest(), **_live_kwargs())  # must not raise


def test_live_identity_rejects_a_one_character_system_prompt_drift():
    with pytest.raises(JudgeIdentityError):
        validate_live_evaluator_identity(
            _real_manifest(), **_live_kwargs(system_prompt=Judge.SYSTEM_PROMPT + " "),
        )


def test_live_identity_rejects_a_one_character_active_rubric_drift():
    tampered = dict(ACTIVE_RUBRICS)
    tampered["refusal"] = tampered["refusal"][:-1] + "X\n"
    with pytest.raises(JudgeIdentityError):
        validate_live_evaluator_identity(_real_manifest(), **_live_kwargs(active_rubrics=tampered))


def test_live_identity_rejects_a_missing_active_rubric():
    tampered = dict(ACTIVE_RUBRICS)
    del tampered["quality"]
    with pytest.raises(JudgeIdentityError):
        validate_live_evaluator_identity(_real_manifest(), **_live_kwargs(active_rubrics=tampered))


def test_live_identity_rejects_a_legacy_rubric_drift():
    with pytest.raises(JudgeIdentityError):
        validate_live_evaluator_identity(
            _real_manifest(),
            **_live_kwargs(legacy_harmfulness_rubric=LEGACY_HARMFULNESS_RUBRIC_PRE_REPAIR + " "),
        )


def test_live_identity_rejects_a_structured_output_schema_drift():
    bad_schema = dict(STRUCTURED_OUTPUT_SCHEMA)
    bad_schema["additionalProperties"] = True
    with pytest.raises(JudgeIdentityError):
        validate_live_evaluator_identity(_real_manifest(), **_live_kwargs(structured_output_schema=bad_schema))


def test_live_identity_passes_even_when_the_manifest_still_matches_the_stale_frozen_pin():
    """The scenario the frozen-identity-only check cannot catch: a manifest that is
    internally self-consistent AND still matches FROZEN_JUDGE_IDENTITY, built from text
    that has since drifted from what's actually imported. Simulated here by tampering
    the live text directly (frozen-identity checks the manifest, not the live code) --
    `validate_frozen_identity` alone would not catch this, `validate_live_evaluator_
    identity` does."""
    manifest = _real_manifest()  # matches FROZEN_JUDGE_IDENTITY exactly
    validate_frozen_identity(manifest)  # passes: the manifest itself hasn't changed
    with pytest.raises(JudgeIdentityError):
        validate_live_evaluator_identity(manifest, **_live_kwargs(system_prompt="drifted live text"))
