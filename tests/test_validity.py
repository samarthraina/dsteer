"""Guards on the two screens that stand between generations and the reported mean.

Both exist because the alternative is a silent bias. Letting a degenerate generation be
judged flatters the over-steered regime; coding a judge abstention as zero flatters the
de-aligned one. Neither shows up as an error -- just as a number that is wrong in a
convenient direction.

Run with:
    pytest tests/test_validity.py -v
"""

from __future__ import annotations

import math

import pytest

from steering.validity import check, ngram_repetition, summarise


# Screening


def test_accepts_ordinary_prose():
    v = check("I can't help with that request, but I can explain why it is dangerous.")
    assert v.ok and v.reason is None


def test_rejects_empty_and_short():
    assert not check("").ok
    assert not check(None).ok
    assert not check("   ").ok
    assert check("no.", min_length=10).reason.startswith("shorter than")


def test_rejects_a_repetition_loop():
    """The observed over-steering failure: one phrase, over and over."""
    text = "safety and understanding and respect. " * 30
    v = check(text)
    assert not v.ok
    assert v.reason.startswith("repetition")
    assert v.repetition > 0.5


def test_rejects_a_leaked_role_header():
    """The observed template fault: the model writes its own header and keeps going."""
    text = "Here is a reply to your question about pens.\nassistant\nAnd here I go again at length."
    assert not check(text).ok
    assert "role token" in check(text).reason


def test_role_token_inside_a_sentence_is_not_a_leak():
    """A leak is a header on its own line; the bare word is ordinary English."""
    text = "A helpful assistant should decline this request, and explain the reason clearly."
    assert check(text).ok


def test_rejects_a_punctuation_periodicity_loop():
    """The observed no-whitespace failure: one short punctuation unit, tiled with no
    spaces to break it into separate n-gram words."""
    text = ".://.//.//.//.//.//.//.//.//.//.//.//"
    v = check(text)
    assert not v.ok
    assert v.reason == "punctuation/subword loop"


def test_rejects_a_repeated_numeric_dot_loop():
    text = "1." * 20
    v = check(text)
    assert not v.ok
    assert v.reason == "punctuation/subword loop"


def test_rejects_a_repeated_subword_unit():
    text = "lolol" * 10
    v = check(text)
    assert not v.ok
    assert v.reason == "punctuation/subword loop"


def test_normal_url_is_not_a_loop():
    text = "See https://example.com/docs/api/v2/reference for the full schema."
    assert check(text).ok


def test_decimal_containing_sentence_is_not_a_loop():
    text = "The reading moved from 3.14159 to 2.71828 over the trial, a real change."
    assert check(text).ok


def test_code_fragment_is_not_a_loop():
    text = "for i in range(10):\n    print(i, i * 2, i ** 2)\nreturn total"
    assert check(text).ok


def test_decimal_with_trailing_zeroes_is_not_a_loop():
    """A single decimal token, not a loop, no matter how much of the response it is."""
    assert check("The result was 0.000000000000 after rounding.").ok


def test_dash_divider_in_code_is_not_a_loop():
    """A repeated separator inside one token must not condemn the whole response."""
    assert check("return ----------------;").ok


def test_formulaic_refusals_survive_the_repetition_screen():
    """Safety boilerplate repeats phrases legitimately.

    A tight threshold would discard exactly the responses a safety sweep is about, so
    this pins the loose default against a realistic refusal.
    """
    text = (
        "I'm sorry, but I can't help with that. It's important to remember that this "
        "is illegal and unsafe. If you have another question, I'm happy to help with that."
    )
    assert check(text).ok


def test_repetition_is_zero_for_varied_text_and_high_for_loops():
    assert ngram_repetition("every single word here is different from all the others") == 0.0
    assert ngram_repetition("loop loop loop loop loop loop loop loop") > 0.5


# Aggregation


def test_abstentions_are_excluded_not_zeroed():
    """The bias this prevents: three refusals to score would halve a mean of 1.0."""
    stats = summarise([1.0, 1.0, 1.0, None, None, None])
    assert stats["mean"] == pytest.approx(1.0)
    assert stats["n_scored"] == 3
    assert stats["n_abstained"] == 3
    assert stats["abstention_rate"] == pytest.approx(0.5)


def test_all_abstained_is_nan_not_zero():
    """No data must read as no data, not as a score of zero."""
    stats = summarise([None, None])
    assert math.isnan(stats["mean"])
    assert stats["n_scored"] == 0


def test_summarise_accepts_a_generator():
    stats = summarise(x for x in [0.5, None, 1.0])
    assert stats["mean"] == pytest.approx(0.75)
    assert stats["n_abstained"] == 1


def test_summarise_of_nothing():
    stats = summarise([])
    assert math.isnan(stats["mean"])
    assert stats["abstention_rate"] == 0.0
