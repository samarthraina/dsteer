"""Guards on score_sweep.py's direct discrete/normalized output schema (Task 005).

Every judged metric, including steering_shift and metric rechecks, must record
`<metric>_discrete_score` (the judge's emitted integer or None) and `<metric>_score`
(that integer / 10, or None) -- the sole authoritative pair per protocol Section 10.
Invalid generations must get None in both fields without ever reaching the judge.

Run with:
    pytest tests/test_score_sweep.py -v

Server-free: the judge is a stub returning a fixed discrete score.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import score_sweep  # noqa: E402
import build_judge_protocol_manifest  # noqa: E402

from steering.judge import JudgeResponse
from steering.judge_identity import FROZEN_JUDGE_IDENTITY, compute_manifest_hash, save_manifest
from steering.utils import read_jsonl, write_jsonl


class _FakeJudge:
    """Returns a fixed discrete score (or a fixed error) for every rubric call, so the
    output schema can be checked without a real judge server.

    `attempts`/`attempt_count`/`resolved_seed` default to a plausible single-attempt
    `JudgeResponse` when not given, so provenance fields always propagate through
    `score_one`/`recheck_file` with realistic (non-empty) values by default; tests that
    care about exact attempt-level content pass their own.
    """

    def __init__(self, discrete_score=7, error=None, attempts=None, attempt_count=None, resolved_seed=None):
        self.discrete_score = discrete_score
        self.error = error
        self.calls = 0
        self._attempts = attempts
        self._attempt_count = attempt_count
        self._resolved_seed = resolved_seed

    def score(self, **kwargs):
        self.calls += 1
        if self.error:
            attempts = self._attempts if self._attempts is not None else [{"raw": None, "error": self.error}]
            attempt_count = self._attempt_count if self._attempt_count is not None else len(attempts)
            return JudgeResponse(
                score=None, discrete_score=None, reason="", raw="", error=self.error,
                attempts=attempts, attempt_count=attempt_count, resolved_seed=self._resolved_seed,
            )
        attempts = self._attempts if self._attempts is not None else [{"raw": "{}", "error": None}]
        attempt_count = self._attempt_count if self._attempt_count is not None else len(attempts)
        return JudgeResponse(
            score=self.discrete_score / 10.0, discrete_score=self.discrete_score,
            reason="stub", raw="{}",
            attempts=attempts, attempt_count=attempt_count, resolved_seed=self._resolved_seed,
        )


def _valid_record(rid="r1"):
    return {
        "id": rid, "prompt": "why is the sky blue",
        "response": "because of Rayleigh scattering, which is a long enough response to pass the validity screen.",
    }


def _invalid_record(rid="r2"):
    return {"id": rid, "prompt": "hi", "response": "no."}  # too short: fails the validity screen


def test_score_one_records_discrete_and_normalized_fields_for_every_metric():
    judge = _FakeJudge(discrete_score=6)
    rec = _valid_record()
    out = score_sweep.score_one(judge, rec, it_ref={}, dpo_ref={}, min_length=10, repetition_threshold=0.5)

    for m in score_sweep.METRICS:
        assert out[f"{m}_discrete_score"] == 6
        assert out[f"{m}_score"] == pytest.approx(0.6)
    assert out["steering_shift_score"] is None  # no IT/DPO references given
    assert out["steering_shift_discrete_score"] is None


def test_score_one_applies_the_same_schema_to_steering_shift():
    judge = _FakeJudge(discrete_score=8)
    rec = _valid_record()
    it_ref = {rec["id"]: "an IT reference response"}
    dpo_ref = {rec["id"]: "a DPO reference response"}
    out = score_sweep.score_one(judge, rec, it_ref=it_ref, dpo_ref=dpo_ref, min_length=10, repetition_threshold=0.5)

    assert out["steering_shift_discrete_score"] == 8
    assert out["steering_shift_score"] == pytest.approx(0.8)


def test_score_one_records_steering_shift_presented_order_and_inversion():
    """"r1" presents IT as REFERENCE A (not inverted); "r2" presents DPO as REFERENCE A
    (inverted, so the judge's raw 8 is mapped to 10-8=2 before being recorded)."""
    it_ref = {"r1": "an IT reference response", "r2": "an IT reference response"}
    dpo_ref = {"r1": "a DPO reference response", "r2": "a DPO reference response"}

    out_it_first = score_sweep.score_one(_FakeJudge(discrete_score=8), _valid_record(rid="r1"),
                                         it_ref=it_ref, dpo_ref=dpo_ref, min_length=10, repetition_threshold=0.5)
    assert out_it_first["steering_shift_presented_order"] == "it_first"
    assert out_it_first["steering_shift_inverted"] is False
    assert out_it_first["steering_shift_discrete_score"] == 8  # unmapped: not inverted

    out_dpo_first = score_sweep.score_one(_FakeJudge(discrete_score=8), _valid_record(rid="r2"),
                                          it_ref=it_ref, dpo_ref=dpo_ref, min_length=10, repetition_threshold=0.5)
    assert out_dpo_first["steering_shift_presented_order"] == "dpo_first"
    assert out_dpo_first["steering_shift_inverted"] is True
    assert out_dpo_first["steering_shift_discrete_score"] == 2  # 10 - 8, mapped back to IT-to-DPO orientation


def test_score_one_leaves_the_steering_shift_midpoint_unchanged_by_inversion():
    """Score 5 (the symmetric midpoint) must round-trip to 5 whichever order was
    presented -- 10 - 5 == 5."""
    it_ref = {"r2": "an IT reference response"}
    dpo_ref = {"r2": "a DPO reference response"}
    out = score_sweep.score_one(_FakeJudge(discrete_score=5), _valid_record(rid="r2"),
                                it_ref=it_ref, dpo_ref=dpo_ref, min_length=10, repetition_threshold=0.5)
    assert out["steering_shift_presented_order"] == "dpo_first"  # confirms inversion is exercised
    assert out["steering_shift_discrete_score"] == 5


def test_score_one_omits_steering_shift_presented_order_when_no_reference_pair():
    out = score_sweep.score_one(_FakeJudge(discrete_score=8), _valid_record(rid="r1"),
                                it_ref={}, dpo_ref={}, min_length=10, repetition_threshold=0.5)
    assert out["steering_shift_presented_order"] is None
    assert out["steering_shift_inverted"] is None


def test_score_one_never_judges_an_invalid_generation():
    judge = _FakeJudge(discrete_score=9)
    rec = _invalid_record()
    out = score_sweep.score_one(judge, rec, it_ref={}, dpo_ref={}, min_length=10, repetition_threshold=0.5)

    assert judge.calls == 0
    assert out["valid"] is False
    for m in score_sweep.METRICS:
        assert out[f"{m}_score"] is None
        assert out[f"{m}_discrete_score"] is None
    assert out["steering_shift_score"] is None
    assert out["steering_shift_discrete_score"] is None
    assert out["steering_shift_presented_order"] is None
    assert out["steering_shift_inverted"] is None


def test_score_one_records_a_judge_error_without_a_discrete_score():
    judge = _FakeJudge(error="boom: could not parse")
    rec = _valid_record()
    out = score_sweep.score_one(judge, rec, it_ref={}, dpo_ref={}, min_length=10, repetition_threshold=0.5)
    for m in score_sweep.METRICS:
        assert out[f"{m}_score"] is None
        assert out[f"{m}_discrete_score"] is None
        assert out[f"{m}_error"] == "boom: could not parse"


# Output-level per-metric provenance (Task 010 correction 3): attempts, attempt count,
# resolved seed, and abstention status, for every metric including steering shift, with
# invalid/not-judged/no-reference distinguished from a genuine judge abstention.


def test_score_one_persists_full_provenance_for_every_metric_on_success():
    attempts = [{"raw": '{"score": 7, "reason": "bad"}', "error": "not real"}, {"raw": "{}", "error": None}]
    judge = _FakeJudge(discrete_score=7, attempts=attempts, attempt_count=2, resolved_seed=123456)
    rec = _valid_record()
    out = score_sweep.score_one(judge, rec, it_ref={}, dpo_ref={}, min_length=10, repetition_threshold=0.5)

    for m in score_sweep.METRICS:
        assert out[f"{m}_attempts"] == attempts
        assert out[f"{m}_attempt_count"] == 2
        assert out[f"{m}_resolved_seed"] == 123456
        assert out[f"{m}_abstained"] is False  # a call happened and returned a valid score
        # Existing fields untouched.
        assert out[f"{m}_discrete_score"] == 7
        assert out[f"{m}_score"] == pytest.approx(0.7)
        assert out[f"{m}_reason"] == "stub"


def test_score_one_persists_full_provenance_for_every_metric_on_a_genuine_abstention():
    attempts = [{"raw": None, "error": "timeout"}, {"raw": "not json", "error": "could not parse"}]
    judge = _FakeJudge(error="could not parse", attempts=attempts, attempt_count=2, resolved_seed=99)
    rec = _valid_record()
    out = score_sweep.score_one(judge, rec, it_ref={}, dpo_ref={}, min_length=10, repetition_threshold=0.5)

    for m in score_sweep.METRICS:
        assert out[f"{m}_attempts"] == attempts
        assert out[f"{m}_attempt_count"] == 2
        assert out[f"{m}_resolved_seed"] == 99
        assert out[f"{m}_abstained"] is True  # a call happened; the judge never returned a valid score
        assert out[f"{m}_discrete_score"] is None


def test_score_one_uses_none_abstention_status_for_an_invalid_not_judged_generation():
    """An invalid generation never reaches the judge -- `_abstained` must be None, not
    True, so it is never confused with a genuine judge abstention."""
    judge = _FakeJudge(discrete_score=9)
    rec = _invalid_record()
    out = score_sweep.score_one(judge, rec, it_ref={}, dpo_ref={}, min_length=10, repetition_threshold=0.5)

    assert judge.calls == 0
    for m in score_sweep.METRICS:
        assert out[f"{m}_attempts"] == []
        assert out[f"{m}_attempt_count"] == 0
        assert out[f"{m}_resolved_seed"] is None
        assert out[f"{m}_abstained"] is None
    assert out["steering_shift_attempts"] == []
    assert out["steering_shift_attempt_count"] == 0
    assert out["steering_shift_resolved_seed"] is None
    assert out["steering_shift_abstained"] is None


def test_score_one_persists_steering_shift_provenance_when_a_reference_pair_exists():
    attempts = [{"raw": '{"score": 8, "reason": "ok"}', "error": None}]
    judge = _FakeJudge(discrete_score=8, attempts=attempts, attempt_count=1, resolved_seed=42)
    it_ref = {"r1": "an IT reference response"}
    dpo_ref = {"r1": "a DPO reference response"}
    out = score_sweep.score_one(judge, _valid_record(rid="r1"), it_ref=it_ref, dpo_ref=dpo_ref,
                                min_length=10, repetition_threshold=0.5)

    assert out["steering_shift_attempts"] == attempts
    assert out["steering_shift_attempt_count"] == 1
    assert out["steering_shift_resolved_seed"] == 42
    assert out["steering_shift_abstained"] is False


def test_score_one_uses_none_abstention_status_for_steering_shift_without_a_reference_pair():
    """No IT/DPO reference pair means Steering Shift is never evaluated for this
    record -- `_abstained` must be None, not a judge abstention."""
    judge = _FakeJudge(discrete_score=8)
    out = score_sweep.score_one(judge, _valid_record(rid="r1"), it_ref={}, dpo_ref={},
                                min_length=10, repetition_threshold=0.5)
    assert out["steering_shift_attempts"] == []
    assert out["steering_shift_attempt_count"] == 0
    assert out["steering_shift_resolved_seed"] is None
    assert out["steering_shift_abstained"] is None


def test_recheck_file_persists_v2_provenance_including_abstention(tmp_path):
    scored = tmp_path / "baseline_scored.jsonl"
    write_jsonl([
        {
            "id": "r1", "lambda": 0.0, "response": "an original valid response text here.",
            "valid": True, "refusal_score": 0.5, "refusal_discrete_score": 5,
        },
    ], scored)

    attempts = [{"raw": '{"score": 9, "reason": "ok"}', "error": None}]
    judge = _FakeJudge(discrete_score=9, attempts=attempts, attempt_count=1, resolved_seed=777)
    dst = tmp_path / "baseline_recheck.jsonl"
    n = score_sweep.recheck_file(judge, scored, dst, metrics=["refusal"], concurrency=1, chunk=1)
    assert n == 1

    [row] = read_jsonl(dst)
    assert row["refusal_attempts_v2"] == attempts
    assert row["refusal_attempt_count_v2"] == 1
    assert row["refusal_resolved_seed_v2"] == 777
    assert row["refusal_abstained_v2"] is False  # recheck always calls the judge -- never None


def test_recheck_file_marks_a_genuine_abstention_in_v2_provenance(tmp_path):
    scored = tmp_path / "baseline_scored.jsonl"
    write_jsonl([
        {
            "id": "r1", "lambda": 0.0, "response": "an original valid response text here.",
            "valid": True, "refusal_score": 0.5, "refusal_discrete_score": 5,
        },
    ], scored)

    judge = _FakeJudge(error="boom: could not parse")
    dst = tmp_path / "baseline_recheck.jsonl"
    n = score_sweep.recheck_file(judge, scored, dst, metrics=["refusal"], concurrency=1, chunk=1)
    assert n == 1

    [row] = read_jsonl(dst)
    assert row["refusal_abstained_v2"] is True


def test_recheck_uses_the_same_direct_score_schema(tmp_path):
    scored = tmp_path / "baseline_scored.jsonl"
    write_jsonl([
        {
            "id": "r1", "lambda": 0.0, "response": "an original valid response text here.",
            "valid": True, "refusal_score": 0.5, "refusal_discrete_score": 5,
        },
    ], scored)

    judge = _FakeJudge(discrete_score=9)
    dst = tmp_path / "baseline_recheck.jsonl"
    n = score_sweep.recheck_file(judge, scored, dst, metrics=["refusal"], concurrency=1, chunk=1)
    assert n == 1

    [row] = read_jsonl(dst)
    assert row["refusal_discrete_score_v2"] == 9
    assert row["refusal_score_v2"] == pytest.approx(0.9)
    assert row["refusal_discrete_score_v1"] == 5
    assert row["refusal_score_v1"] == pytest.approx(0.5)


def test_recheck_records_the_judge_error_without_a_v2_discrete_score(tmp_path):
    scored = tmp_path / "baseline_scored.jsonl"
    write_jsonl([
        {
            "id": "r1", "lambda": 0.0, "response": "an original valid response text here.",
            "valid": True, "refusal_score": 0.5, "refusal_discrete_score": 5,
        },
    ], scored)

    judge = _FakeJudge(error="boom: could not parse")
    dst = tmp_path / "baseline_recheck.jsonl"
    n = score_sweep.recheck_file(judge, scored, dst, metrics=["refusal"], concurrency=1, chunk=1)
    assert n == 1

    [row] = read_jsonl(dst)
    assert row["refusal_discrete_score_v2"] is None
    assert row["refusal_score_v2"] is None
    assert row["refusal_error_v2"] == "boom: could not parse"


def test_no_active_output_is_changed_by_alternative_token_probabilities():
    """A stub judge cannot carry logprobs at all (score_sweep only ever sees the
    already-parsed JudgeResponse) -- this pins that the discrete/normalized pair is
    exactly the stub's emitted value, with no weighting concept anywhere in the path."""
    judge = _FakeJudge(discrete_score=3)
    rec = _valid_record()
    out = score_sweep.score_one(judge, rec, it_ref={}, dpo_ref={}, min_length=10, repetition_threshold=0.5)
    for m in score_sweep.METRICS:
        assert out[f"{m}_discrete_score"] == 3
        assert out[f"{m}_score"] == pytest.approx(0.3)
        assert "weighted" not in out


# build_run_config


def _sweep_dir(tmp_path, names=("b.jsonl", "a.jsonl")):
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    for name in names:
        (sweep / name).write_text("", encoding="utf-8")
    return sweep


def _score_sweep_parser() -> argparse.ArgumentParser:
    """Mirrors score_sweep.main()'s real CLI surface, for testing build_run_config and
    metadata wiring without running main() end to end."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--it-baseline", default=None)
    parser.add_argument("--dpo-baseline", default=None)
    parser.add_argument("--judge-url", default="http://localhost:8000/v1")
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--min-length", type=int, default=10)
    parser.add_argument("--repetition-threshold", type=float, default=0.5)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--recheck", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--confirmatory", action="store_true")
    parser.add_argument("--judge-protocol-manifest", default=None)
    return parser


def _expected_cli(sweep, **overrides):
    base = {
        "sweep_dir": str(sweep), "output_dir": None, "it_baseline": None, "dpo_baseline": None,
        "judge_url": "http://localhost:8000/v1", "judge_model": "Qwen/Qwen2.5-32B-Instruct",
        "min_length": 10, "repetition_threshold": 0.5, "concurrency": 32, "sync": False,
        "recheck": None, "limit": None, "confirmatory": False, "judge_protocol_manifest": None,
    }
    base.update(overrides)
    return base


def test_build_run_config_binds_full_cli_ordered_input_files_and_metric_names(tmp_path):
    sweep = _sweep_dir(tmp_path)
    files = sorted(sweep.glob("*.jsonl"))
    args = _score_sweep_parser().parse_args(["--sweep-dir", str(sweep)])

    config = score_sweep.build_run_config(args, files)

    assert config["input_files"] == [str(p) for p in files]
    assert config["input_files"] == sorted(config["input_files"])  # resolved, ordered
    assert config["metrics"] == list(score_sweep.METRICS)
    assert config["cli"] == _expected_cli(sweep)
    assert "judge_protocol" not in config  # not on the legacy (non-confirmatory) path


def test_build_run_config_binds_judge_protocol_when_given(tmp_path):
    sweep = _sweep_dir(tmp_path)
    files = sorted(sweep.glob("*.jsonl"))
    args = _score_sweep_parser().parse_args(["--sweep-dir", str(sweep)])
    judge_protocol = {"manifest_hash": "abc", "resolved_concurrency": 32}

    config = score_sweep.build_run_config(args, files, judge_protocol=judge_protocol)

    assert config["judge_protocol"] == judge_protocol


# main(): metadata wiring (Task 009)


class _MetadataSentinel(Exception):
    """Raised by a mocked write_run_metadata to simulate an identity mismatch, so a
    test can prove nothing after the metadata call ran -- without mocking the rest of
    the pipeline."""


def _spy_write_run_metadata(monkeypatch, calls, raise_sentinel=True):
    def fake(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        if raise_sentinel:
            raise _MetadataSentinel()
        return Path("run_meta.json")

    monkeypatch.setattr(score_sweep, "write_run_metadata", fake)


def _forbid(monkeypatch, name):
    """Fail loudly if `name` is ever called -- used to prove no side effect happens
    after a simulated metadata mismatch."""
    def boom(*a, **k):
        raise AssertionError(f"{name} must not be called after a metadata identity mismatch")

    monkeypatch.setattr(score_sweep, name, boom)


def _forbid_judge_construction(monkeypatch):
    """Fail loudly only on actual `Judge(cfg)` construction. Unlike `_forbid(...,
    "Judge")` (which replaces the whole name and so also breaks a harmless read of
    `Judge.SYSTEM_PROMPT` for the live-identity check), this patches `__init__` on the
    real class -- the class itself, and its attributes, remain readable."""
    def boom(self, cfg):
        raise AssertionError("Judge must not be constructed after a metadata identity mismatch")

    monkeypatch.setattr(score_sweep.Judge, "__init__", boom)


def test_argv_is_passed_as_an_exact_json_list_preserving_spaces_and_unicode(tmp_path, monkeypatch):
    sweep = _sweep_dir(tmp_path)
    argv = ["score_sweep.py", "--sweep-dir", str(sweep), "--judge-model", "unicode üñíçødé and spaces here"]
    monkeypatch.setattr(sys, "argv", argv)

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)

    with pytest.raises(_MetadataSentinel):
        score_sweep.main()

    assert calls[0]["kwargs"]["argv"] == argv
    assert isinstance(calls[0]["kwargs"]["argv"], list)


def test_metadata_config_includes_the_full_parsed_cli_namespace_with_defaults(tmp_path, monkeypatch):
    sweep = _sweep_dir(tmp_path)
    argv = ["score_sweep.py", "--sweep-dir", str(sweep)]
    monkeypatch.setattr(sys, "argv", argv)

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)

    with pytest.raises(_MetadataSentinel):
        score_sweep.main()

    cli = calls[0]["kwargs"]["config"]["cli"]
    assert cli == _expected_cli(sweep)


def test_metadata_includes_ordered_discovered_input_files_and_metric_names(tmp_path, monkeypatch):
    sweep = _sweep_dir(tmp_path, names=("c.jsonl", "a.jsonl", "b.jsonl"))
    expected_files = sorted(str(p) for p in sweep.glob("*.jsonl"))
    monkeypatch.setattr(sys, "argv", ["score_sweep.py", "--sweep-dir", str(sweep)])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)

    with pytest.raises(_MetadataSentinel):
        score_sweep.main()

    config = calls[0]["kwargs"]["config"]
    assert config["input_files"] == expected_files
    assert config["metrics"] == list(score_sweep.METRICS)


def test_write_run_metadata_is_called_before_setup_logging(tmp_path, monkeypatch):
    sweep = _sweep_dir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["score_sweep.py", "--sweep-dir", str(sweep)])

    order = []

    def fake_write_run_metadata(*a, **k):
        order.append("write_run_metadata")
        raise _MetadataSentinel()

    monkeypatch.setattr(score_sweep, "write_run_metadata", fake_write_run_metadata)
    monkeypatch.setattr(score_sweep, "setup_logging", lambda *a, **k: order.append("setup_logging"))

    with pytest.raises(_MetadataSentinel):
        score_sweep.main()

    assert order == ["write_run_metadata"]  # setup_logging never ran


def test_a_metadata_mismatch_prevents_all_later_side_effects(tmp_path, monkeypatch):
    sweep = _sweep_dir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["score_sweep.py", "--sweep-dir", str(sweep)])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)
    _forbid(monkeypatch, "setup_logging")
    _forbid(monkeypatch, "Judge")
    _forbid(monkeypatch, "sync_to_hub")

    with pytest.raises(_MetadataSentinel):
        score_sweep.main()

    assert len(calls) == 1  # metadata was attempted exactly once, then main() stopped


# main(): --confirmatory judge-protocol verification (Task 010)


def _write_real_judge_manifest(tmp_path) -> Path:
    """The exact frozen manifest -- byte-for-byte what build_judge_protocol_manifest.py
    produces -- written to a tmp path so tests never depend on, or mutate, the
    committed `manifests/judge_protocol_v1.json`."""
    manifest = build_judge_protocol_manifest.build()
    path = tmp_path / "judge_protocol_v1.json"
    save_manifest(manifest, path)
    return path


def test_confirmatory_binds_the_verified_judge_identity_manifest_hash_seed_policy_and_concurrency(tmp_path, monkeypatch):
    sweep = _sweep_dir(tmp_path)
    manifest_path = _write_real_judge_manifest(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "score_sweep.py", "--sweep-dir", str(sweep), "--confirmatory",
        "--judge-protocol-manifest", str(manifest_path),
    ])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)

    with pytest.raises(_MetadataSentinel):
        score_sweep.main()

    config = calls[0]["kwargs"]["config"]
    jp = config["judge_protocol"]
    assert jp["manifest_hash"] == FROZEN_JUDGE_IDENTITY["manifest_hash"]
    # The complete verified manifest -- judge identity, sampling, schema, every hash --
    # not only the "judge" subsection.
    assert jp["manifest"]["judge"] == FROZEN_JUDGE_IDENTITY["judge"]
    assert jp["manifest"]["sampling"] == FROZEN_JUDGE_IDENTITY["sampling"]
    assert jp["manifest"]["structured_output_schema"] == FROZEN_JUDGE_IDENTITY["structured_output_schema"]
    assert jp["manifest"]["active_rubric_hashes"] == FROZEN_JUDGE_IDENTITY["active_rubric_hashes"]
    assert jp["manifest"]["legacy_harmfulness_hash"] == FROZEN_JUDGE_IDENTITY["legacy_harmfulness_hash"]
    assert jp["seed_policy"] == {
        "global_seed": FROZEN_JUDGE_IDENTITY["global_seed"],
        "seed_derivation_version": FROZEN_JUDGE_IDENTITY["seed_derivation_version"],
    }
    assert jp["resolved_concurrency"] == 32
    # Unambiguous even though --judge-model's CLI default is still the legacy Qwen2.5
    # name: this is always the resolved revision-bearing Qwen3.5 alias.
    assert jp["resolved_model_alias"] == FROZEN_JUDGE_IDENTITY["judge"]["served_model_alias"]
    assert config["cli"]["judge_model"] == score_sweep.LEGACY_JUDGE_MODEL_DEFAULT  # the CLI default, unresolved
    # The CLI namespace itself also carries confirmatory=True through the existing cli key.
    assert config["cli"]["confirmatory"] is True


def test_confirmatory_rejects_a_tampered_manifest_before_any_mutation(tmp_path, monkeypatch):
    manifest = build_judge_protocol_manifest.build()
    tampered = copy.deepcopy(manifest)
    tampered["sampling"]["temperature"] = 0.0  # one frozen field changed
    # Recompute a self-consistent (but non-frozen) hash so load_manifest's own
    # self-check passes and the rejection comes from validate_frozen_identity, not from
    # a stale-hash load failure.
    tampered["manifest_hash"] = compute_manifest_hash(tampered)

    manifest_path = tmp_path / "tampered_judge_protocol.json"
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")

    sweep = _sweep_dir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "score_sweep.py", "--sweep-dir", str(sweep), "--confirmatory",
        "--judge-protocol-manifest", str(manifest_path),
    ])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)
    _forbid(monkeypatch, "setup_logging")
    _forbid(monkeypatch, "Judge")
    _forbid(monkeypatch, "sync_to_hub")

    with pytest.raises(SystemExit):
        score_sweep.main()

    assert calls == []  # write_run_metadata was never reached


def test_confirmatory_rejects_a_stale_hash_manifest_before_any_mutation(tmp_path, monkeypatch):
    manifest = build_judge_protocol_manifest.build()
    tampered = copy.deepcopy(manifest)
    tampered["global_seed"] = 1  # manifest_hash left stale -- load_manifest itself must reject this
    manifest_path = tmp_path / "stale_hash_judge_protocol.json"
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")

    sweep = _sweep_dir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "score_sweep.py", "--sweep-dir", str(sweep), "--confirmatory",
        "--judge-protocol-manifest", str(manifest_path),
    ])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)
    _forbid(monkeypatch, "setup_logging")
    _forbid(monkeypatch, "Judge")

    with pytest.raises(SystemExit):
        score_sweep.main()

    assert calls == []


def test_confirmatory_rejects_a_non_frozen_judge_model_override_before_any_mutation(tmp_path, monkeypatch):
    manifest_path = _write_real_judge_manifest(tmp_path)
    sweep = _sweep_dir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "score_sweep.py", "--sweep-dir", str(sweep), "--confirmatory",
        "--judge-protocol-manifest", str(manifest_path),
        "--judge-model", "some/other-model",
    ])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)
    _forbid(monkeypatch, "setup_logging")
    _forbid_judge_construction(monkeypatch)

    with pytest.raises(SystemExit):
        score_sweep.main()

    assert calls == []


def test_confirmatory_rejects_a_non_frozen_concurrency_override_before_any_mutation(tmp_path, monkeypatch):
    manifest_path = _write_real_judge_manifest(tmp_path)
    sweep = _sweep_dir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "score_sweep.py", "--sweep-dir", str(sweep), "--confirmatory",
        "--judge-protocol-manifest", str(manifest_path),
        "--concurrency", "8",
    ])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)
    _forbid(monkeypatch, "setup_logging")
    _forbid_judge_construction(monkeypatch)

    with pytest.raises(SystemExit):
        score_sweep.main()

    assert calls == []


def test_confirmatory_rejects_a_one_character_live_system_prompt_change_before_any_mutation(tmp_path, monkeypatch):
    """The manifest on disk is the real, matching one -- only the live, currently
    imported `Judge.SYSTEM_PROMPT` has drifted by one character. `validate_frozen_
    judge_identity` alone cannot see this (the manifest still matches
    FROZEN_JUDGE_IDENTITY); only the live-evaluator-identity check catches it."""
    manifest_path = _write_real_judge_manifest(tmp_path)
    sweep = _sweep_dir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "score_sweep.py", "--sweep-dir", str(sweep), "--confirmatory",
        "--judge-protocol-manifest", str(manifest_path),
    ])
    monkeypatch.setattr(score_sweep.Judge, "SYSTEM_PROMPT", score_sweep.Judge.SYSTEM_PROMPT + " ")

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)
    _forbid(monkeypatch, "setup_logging")
    _forbid_judge_construction(monkeypatch)

    with pytest.raises(SystemExit):
        score_sweep.main()

    assert calls == []


def test_confirmatory_rejects_a_one_character_live_rubric_change_before_any_mutation(tmp_path, monkeypatch):
    """Same as above but for an active rubric instead of the system prompt."""
    manifest_path = _write_real_judge_manifest(tmp_path)
    sweep = _sweep_dir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "score_sweep.py", "--sweep-dir", str(sweep), "--confirmatory",
        "--judge-protocol-manifest", str(manifest_path),
    ])
    tampered_rubrics = dict(score_sweep.ACTIVE_RUBRICS)
    tampered_rubrics["quality"] = tampered_rubrics["quality"][:-1] + "X\n"
    monkeypatch.setattr(score_sweep, "ACTIVE_RUBRICS", tampered_rubrics)

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)
    _forbid(monkeypatch, "setup_logging")
    _forbid_judge_construction(monkeypatch)

    with pytest.raises(SystemExit):
        score_sweep.main()

    assert calls == []


def test_confirmatory_accepts_the_frozen_alias_passed_explicitly(tmp_path, monkeypatch):
    """--judge-model is not required to be omitted -- passing exactly the frozen alias
    is also accepted, not just the legacy default sentinel."""
    manifest_path = _write_real_judge_manifest(tmp_path)
    sweep = _sweep_dir(tmp_path)
    frozen_alias = FROZEN_JUDGE_IDENTITY["judge"]["served_model_alias"]
    monkeypatch.setattr(sys, "argv", [
        "score_sweep.py", "--sweep-dir", str(sweep), "--confirmatory",
        "--judge-protocol-manifest", str(manifest_path),
        "--judge-model", frozen_alias,
    ])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)

    with pytest.raises(_MetadataSentinel):
        score_sweep.main()

    assert len(calls) == 1
    jp = calls[0]["kwargs"]["config"]["judge_protocol"]
    assert jp["manifest"]["judge"]["served_model_alias"] == frozen_alias
    assert jp["resolved_model_alias"] == frozen_alias
