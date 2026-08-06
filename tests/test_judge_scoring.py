"""Guards on judge score weighting.

The failure this protects against is silent: deepeval requests logprobs, but falls back to
a plain integer whenever the model wrapper lacks `a_generate_raw_response`, and never says
so. Our earlier runs took that path without anyone noticing, which is how a nominally
11-point scale ended up behaving like a 3-point one.

So the weighting must be observable (`JudgeResponse.weighted`) and must degrade to the raw
integer rather than to nothing.

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
    """A stub chat completion, optionally carrying logprobs for the score token."""
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


def test_weighted_score_is_a_probability_weighted_mean(judge, monkeypatch):
    """7 at 0.6 and 8 at 0.4 must give 7.4/10, not 7/10."""
    completion = make_completion(
        '{"score": 7, "reason": "ok"}', score_token="7", alternatives={"7": 0.6, "8": 0.4}
    )
    monkeypatch.setattr(judge, "_call", lambda msg: completion)

    res = judge.score(rubric="r", output_text="o")
    assert res.weighted is True
    assert res.raw_score == pytest.approx(0.7)
    assert res.score == pytest.approx(0.74)


def test_alternatives_are_renormalised(judge, monkeypatch):
    """Non-numeric alternatives are dropped, and the rest renormalised."""
    completion = make_completion(
        '{"score": 3, "reason": "x"}',
        score_token="3",
        alternatives={"3": 0.3, "4": 0.3, "banana": 0.4},
    )
    monkeypatch.setattr(judge, "_call", lambda msg: completion)

    res = judge.score(rubric="r", output_text="o")
    # banana discarded, 3 and 4 renormalise to 0.5 each -> 3.5
    assert res.score == pytest.approx(0.35)


def test_falls_back_to_raw_integer_without_logprobs(judge, monkeypatch):
    """The deepeval failure mode: no logprobs must degrade visibly, not silently."""
    completion = make_completion('{"score": 9, "reason": "y"}')
    monkeypatch.setattr(judge, "_call", lambda msg: completion)

    res = judge.score(rubric="r", output_text="o")
    assert res.weighted is False
    assert res.score == pytest.approx(0.9)
    assert res.raw_score == pytest.approx(0.9)


def test_falls_back_when_score_token_not_found(judge, monkeypatch):
    """Logprobs present but for some other token: keep the integer."""
    completion = make_completion(
        '{"score": 5, "reason": "z"}', score_token="reason", alternatives={"1": 1.0}
    )
    monkeypatch.setattr(judge, "_call", lambda msg: completion)

    res = judge.score(rubric="r", output_text="o")
    assert res.weighted is False
    assert res.score == pytest.approx(0.5)


def test_use_logprobs_false_skips_weighting(monkeypatch):
    monkeypatch.setattr("steering.judge.OpenAI", lambda **kw: SimpleNamespace())
    judge = Judge(JudgeConfig(use_logprobs=False))
    completion = make_completion(
        '{"score": 7, "reason": "ok"}', score_token="7", alternatives={"7": 0.5, "8": 0.5}
    )
    monkeypatch.setattr(judge, "_call", lambda msg: completion)

    res = judge.score(rubric="r", output_text="o")
    assert res.weighted is False
    assert res.score == pytest.approx(0.7)


def test_scores_stay_in_unit_range(judge, monkeypatch):
    """A judge that ignores the scale must not produce out-of-range values."""
    completion = make_completion('{"score": 42, "reason": "!"}')
    monkeypatch.setattr(judge, "_call", lambda msg: completion)

    res = judge.score(rubric="r", output_text="o")
    assert 0.0 <= res.score <= 1.0


def test_unicode_digit_alternatives_do_not_break_scoring(judge, monkeypatch):
    """str.isdigit() accepts subscripts and superscripts; int() does not.

    The judge really does return tokens like the subscript one among its alternatives at
    a score position, and the resulting ValueError failed the call, retried, and left the
    record unscored -- 86% of one metric before it was caught.
    """
    completion = make_completion(
        '{"score": 7, "reason": "ok"}',
        score_token="7",
        alternatives={"7": 0.5, "8": 0.3, "\u2081": 0.1, "\u00b2": 0.1},
    )
    monkeypatch.setattr(judge, "_call", lambda msg: completion)

    res = judge.score(rubric="r", output_text="o")
    assert res.error is None
    assert res.weighted is True
    # Only 7 and 8 count, renormalised to 0.625/0.375 -> 7.375
    assert res.score == pytest.approx(0.7375)


def test_two_token_top_of_scale_is_weighted(judge, monkeypatch):
    """A score of 10 arrives as '1' then '0'.

    Matching the literal "10" finds nothing, which is why weighting silently never fired.
    At the first digit token the alternatives are ['1', '9', ...] where '1' opens 10.
    """
    completion = make_completion(
        '{"score": 10, "reason": "total refusal"}',
        score_token="1",
        alternatives={"1": 0.7, "9": 0.3},
    )
    monkeypatch.setattr(judge, "_call", lambda msg: completion)

    res = judge.score(rubric="r", output_text="o")
    assert res.weighted is True
    # '1' stands for the emitted 10: 0.7*10 + 0.3*9 = 9.7
    assert res.score == pytest.approx(0.97)
