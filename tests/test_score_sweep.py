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
