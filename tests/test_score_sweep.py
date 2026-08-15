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
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import score_sweep  # noqa: E402

from steering.judge import JudgeResponse
from steering.utils import read_jsonl, write_jsonl


class _FakeJudge:
    """Returns a fixed discrete score (or a fixed error) for every rubric call, so the
    output schema can be checked without a real judge server."""

    def __init__(self, discrete_score=7, error=None):
        self.discrete_score = discrete_score
        self.error = error
        self.calls = 0

    def score(self, **kwargs):
        self.calls += 1
        if self.error:
            return JudgeResponse(score=None, discrete_score=None, reason="", raw="", error=self.error)
        return JudgeResponse(
            score=self.discrete_score / 10.0, discrete_score=self.discrete_score,
            reason="stub", raw="{}",
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


def test_score_one_records_a_judge_error_without_a_discrete_score():
    judge = _FakeJudge(error="boom: could not parse")
    rec = _valid_record()
    out = score_sweep.score_one(judge, rec, it_ref={}, dpo_ref={}, min_length=10, repetition_threshold=0.5)
    for m in score_sweep.METRICS:
        assert out[f"{m}_score"] is None
        assert out[f"{m}_discrete_score"] is None
        assert out[f"{m}_error"] == "boom: could not parse"


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
    return parser


def _expected_cli(sweep, **overrides):
    base = {
        "sweep_dir": str(sweep), "output_dir": None, "it_baseline": None, "dpo_baseline": None,
        "judge_url": "http://localhost:8000/v1", "judge_model": "Qwen/Qwen2.5-32B-Instruct",
        "min_length": 10, "repetition_threshold": 0.5, "concurrency": 32, "sync": False,
        "recheck": None, "limit": None,
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
