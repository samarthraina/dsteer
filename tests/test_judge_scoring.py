"""Guards on the judge's discrete-score authority (protocol Section 10).

The judge-emitted JSON integer is the sole authoritative score. An earlier version of
this module re-weighted it by the token-probability alternatives at its position
(G-Eval); that estimator is ambiguous for multi-token scores and is not the outcome the
protocol froze, so it was removed entirely. These pin that removal: alternatives are
still visible in the stub completions below (to prove they have no effect), the judge
request itself must never ask for logprobs, and `use_logprobs=True` must fail
immediately at construction rather than silently reviving the old estimator. They also
pin that an invalid emitted score is rejected outright -- never coerced, clamped, or
rounded into range.

Run with:
    pytest tests/test_judge_scoring.py -v

No server required; the completion objects are stubs.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from steering.config import JudgeConfig
from steering.judge import Judge


def make_completion(text: str, score_token: str = None, alternatives: dict = None):
    """A stub chat completion, optionally carrying logprobs for the score token -- used
    only to prove they are ignored, since the judge no longer requests them."""
    message = SimpleNamespace(content=text)
    logprobs = None
    if score_token is not None:
        top = [
            SimpleNamespace(token=tok, logprob=math.log(p))
            for tok, p in (alternatives or {}).items()
        ]
        logprobs = SimpleNamespace(
            content=[SimpleNamespace(token=score_token, top_logprobs=top)]
        )
    return SimpleNamespace(choices=[SimpleNamespace(message=message, logprobs=logprobs)])


@pytest.fixture
def judge(monkeypatch):
    monkeypatch.setattr("steering.judge.OpenAI", lambda **kw: SimpleNamespace())
    return Judge(JudgeConfig())


# Construction guards


def test_use_logprobs_true_fails_at_construction(monkeypatch):
    monkeypatch.setattr("steering.judge.OpenAI", lambda **kw: SimpleNamespace())
    with pytest.raises(ValueError):
        Judge(JudgeConfig(use_logprobs=True))


def test_a_non_ten_max_score_fails_at_construction(monkeypatch):
    monkeypatch.setattr("steering.judge.OpenAI", lambda **kw: SimpleNamespace())
    with pytest.raises(ValueError):
        Judge(JudgeConfig(max_score=5))


def test_the_safe_default_disables_logprob_weighting():
    assert JudgeConfig().use_logprobs is False


def test_the_safe_default_constructs_without_raising(monkeypatch):
    monkeypatch.setattr("steering.judge.OpenAI", lambda **kw: SimpleNamespace())
    Judge(JudgeConfig())  # must not raise


# Accepted discrete scores, normalized exactly


def test_emitted_zero_is_accepted_and_normalized_exactly(judge, monkeypatch):
    completion = make_completion('{"score": 0, "reason": "none"}')
    monkeypatch.setattr(judge, "_call", lambda msg: completion)
    res = judge.score(rubric="r", output_text="o")
    assert res.discrete_score == 0
    assert res.score == pytest.approx(0.0)
    assert res.error is None


def test_an_interior_emitted_score_is_accepted_and_normalized_exactly(judge, monkeypatch):
    completion = make_completion('{"score": 7, "reason": "ok"}')
    monkeypatch.setattr(judge, "_call", lambda msg: completion)
    res = judge.score(rubric="r", output_text="o")
    assert res.discrete_score == 7
    assert res.score == pytest.approx(0.7)


def test_emitted_ten_is_accepted_and_normalized_exactly(judge, monkeypatch):
    completion = make_completion('{"score": 10, "reason": "max"}')
    monkeypatch.setattr(judge, "_call", lambda msg: completion)
    res = judge.score(rubric="r", output_text="o")
    assert res.discrete_score == 10
    assert res.score == pytest.approx(1.0)


def test_raw_score_and_weighted_are_deprecated_compatibility_fields(judge, monkeypatch):
    completion = make_completion('{"score": 4, "reason": "ok"}')
    monkeypatch.setattr(judge, "_call", lambda msg: completion)
    res = judge.score(rubric="r", output_text="o")
    assert res.raw_score == pytest.approx(res.score)
    assert res.weighted is False


# Logprobs cannot alter the emitted score


def test_logprobs_on_the_completion_cannot_alter_the_emitted_score(judge, monkeypatch):
    """Alternatives strongly favor 8 over the emitted 7; the discrete score must stay 7."""
    completion = make_completion(
        '{"score": 7, "reason": "ok"}', score_token="7", alternatives={"7": 0.05, "8": 0.95},
    )
    monkeypatch.setattr(judge, "_call", lambda msg: completion)
    res = judge.score(rubric="r", output_text="o")
    assert res.discrete_score == 7
    assert res.score == pytest.approx(0.7)
    assert res.weighted is False


def test_logprobs_at_the_top_of_scale_cannot_alter_a_two_token_emitted_score(judge, monkeypatch):
    """10 is emitted as '1' then '0'; alternatives at that position must not matter."""
    completion = make_completion(
        '{"score": 10, "reason": "max"}', score_token="1", alternatives={"1": 0.1, "9": 0.9},
    )
    monkeypatch.setattr(judge, "_call", lambda msg: completion)
    res = judge.score(rubric="r", output_text="o")
    assert res.discrete_score == 10
    assert res.score == pytest.approx(1.0)


def test_judge_request_never_asks_for_logprobs(judge):
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return make_completion('{"score": 5, "reason": "ok"}')

    judge.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    judge.score(rubric="r", output_text="o")
    assert "logprobs" not in captured
    assert "top_logprobs" not in captured


# Rejected / invalid scores -- never coerced, clamped, or rounded


def test_a_string_score_is_rejected_not_coerced(judge, monkeypatch):
    completion = make_completion('{"score": "7", "reason": "x"}')
    monkeypatch.setattr(judge, "_call", lambda msg: completion)
    res = judge.score(rubric="r", output_text="o", max_retries=0)
    assert res.score is None
    assert res.discrete_score is None
    assert res.error is not None


def test_a_float_score_is_rejected_not_truncated(judge, monkeypatch):
    completion = make_completion('{"score": 7.5, "reason": "x"}')
    monkeypatch.setattr(judge, "_call", lambda msg: completion)
    res = judge.score(rubric="r", output_text="o", max_retries=0)
    assert res.score is None
    assert res.discrete_score is None


def test_a_whole_number_float_score_is_still_rejected(judge, monkeypatch):
    """7.0 is a JSON float, not the integer 7 -- must not be silently accepted."""
    completion = make_completion('{"score": 7.0, "reason": "x"}')
    monkeypatch.setattr(judge, "_call", lambda msg: completion)
    res = judge.score(rubric="r", output_text="o", max_retries=0)
    assert res.score is None
    assert res.discrete_score is None


def test_a_boolean_score_is_rejected_even_though_bool_subclasses_int(judge, monkeypatch):
    completion = make_completion('{"score": true, "reason": "x"}')
    monkeypatch.setattr(judge, "_call", lambda msg: completion)
    res = judge.score(rubric="r", output_text="o", max_retries=0)
    assert res.score is None
    assert res.discrete_score is None


def test_a_negative_score_is_rejected_not_clamped(judge, monkeypatch):
    completion = make_completion('{"score": -1, "reason": "x"}')
    monkeypatch.setattr(judge, "_call", lambda msg: completion)
    res = judge.score(rubric="r", output_text="o", max_retries=0)
    assert res.score is None
    assert res.discrete_score is None


def test_an_above_range_score_is_rejected_not_clamped(judge, monkeypatch):
    completion = make_completion('{"score": 42, "reason": "!"}')
    monkeypatch.setattr(judge, "_call", lambda msg: completion)
    res = judge.score(rubric="r", output_text="o", max_retries=0)
    assert res.score is None
    assert res.discrete_score is None


def test_a_missing_score_key_is_rejected(judge, monkeypatch):
    completion = make_completion('{"reason": "no score given"}')
    monkeypatch.setattr(judge, "_call", lambda msg: completion)
    res = judge.score(rubric="r", output_text="o", max_retries=0)
    assert res.score is None
    assert res.discrete_score is None
    assert res.error is not None


def test_unparseable_json_is_rejected(judge, monkeypatch):
    completion = make_completion("not json at all")
    monkeypatch.setattr(judge, "_call", lambda msg: completion)
    res = judge.score(rubric="r", output_text="o", max_retries=0)
    assert res.score is None
    assert res.discrete_score is None
    assert res.error is not None


# Retry behavior


def test_malformed_then_valid_retry_returns_the_valid_discrete_score(judge, monkeypatch):
    attempts = [
        make_completion('{"score": "not a number", "reason": "bad"}'),
        make_completion('{"score": 6, "reason": "good"}'),
    ]
    calls = iter(attempts)
    monkeypatch.setattr(judge, "_call", lambda msg: next(calls))

    res = judge.score(rubric="r", output_text="o", max_retries=2)
    assert res.discrete_score == 6
    assert res.score == pytest.approx(0.6)
    assert res.error is None


def test_all_invalid_attempts_return_an_abstention_with_the_final_error(judge, monkeypatch):
    bad = [
        make_completion('{"score": "x", "reason": "1"}'),
        make_completion('{"score": -5, "reason": "2"}'),
        make_completion('{"score": 99, "reason": "3"}'),
    ]
    calls = iter(bad)
    monkeypatch.setattr(judge, "_call", lambda msg: next(calls))

    res = judge.score(rubric="r", output_text="o", max_retries=2)
    assert res.score is None
    assert res.discrete_score is None
    assert res.error is not None
    assert "99" in res.error  # the final attempt's error is what's recorded


# Existing JSON-extraction behavior (unrelated to weighting; must still pass)


def test_json_extraction_from_a_markdown_fence():
    text = '```json\n{"score": 6, "reason": "fenced"}\n```'
    assert Judge._extract_json(text) == {"score": 6, "reason": "fenced"}


def test_json_extraction_from_surrounding_prose():
    text = 'Sure, here you go: {"score": 3, "reason": "prose"} -- hope that helps.'
    assert Judge._extract_json(text) == {"score": 3, "reason": "prose"}
