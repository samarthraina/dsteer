#!/usr/bin/env python3
"""Tests for src/paired_stats.py. Each pins one specific statistical failure.

Run:  python3 -I tests/test_paired_stats.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paired_stats import (  # noqa: E402
    BOOTSTRAP_SEED, directed_effect, judge_agreement, method_contrast,
    paired_bootstrap, rate_difference, strict_intersection, worst_case_bounds,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def test_strict_intersection_drops_invalid() -> None:
    """Pins: a prompt invalid in one arm still entering a paired estimate."""
    print("test_strict_intersection_drops_invalid")
    a = {"p1": 0.5, "p2": 0.4, "p3": None}
    b = {"p1": 0.2, "p2": None, "p3": 0.1}
    check("only fully-observed ids kept", strict_intersection(a, b) == ["p1"],
          str(strict_intersection(a, b)))
    c = {"p1": 0.9}
    check("intersection narrows with a third arm", strict_intersection(a, b, c) == ["p1"])


def test_directed_effect_sign_convention() -> None:
    """Pins: a sign error making removal look like installation (the orientation bug)."""
    print("test_directed_effect_sign_convention")
    # installation: harmfulness should FALL, so baseline - intervention > 0
    base = {f"p{i}": 0.8 for i in range(40)}
    inst = {f"p{i}": 0.3 for i in range(40)}
    e = directed_effect(base, inst, sign=+1)
    check("installation effect positive", abs(e.point - 0.5) < 1e-9, f"{e.point}")
    # reversal: harmfulness should RISE, so intervention - baseline > 0
    dpo = {f"p{i}": 0.2 for i in range(40)}
    rev = {f"p{i}": 0.6 for i in range(40)}
    e2 = directed_effect(dpo, rev, sign=-1)
    check("reversal effect positive", abs(e2.point - 0.4) < 1e-9, f"{e2.point}")
    # a wrong-direction intervention must come out negative, not be silently flipped
    e3 = directed_effect(base, {f"p{i}": 0.95 for i in range(40)}, sign=+1)
    check("wrong-direction effect is negative", e3.point < 0, f"{e3.point}")
    try:
        directed_effect(base, inst, sign=0)
        check("invalid sign rejected", False, "no exception")
    except ValueError:
        check("invalid sign rejected", True)


def test_bootstrap_is_deterministic_and_joint() -> None:
    """Pins: non-reproducible intervals, and arms resampled independently (unpairing them)."""
    print("test_bootstrap_is_deterministic_and_joint")
    base = {f"p{i}": 0.1 * (i % 10) for i in range(60)}
    inter = {f"p{i}": 0.1 * (i % 10) - 0.2 for i in range(60)}
    e1 = directed_effect(base, inter, +1)
    e2 = directed_effect(base, inter, +1)
    check("identical across runs", (e1.ci_lo, e1.ci_hi) == (e2.ci_lo, e2.ci_hi))
    check("seed is the documented one", BOOTSTRAP_SEED == 20261028)
    # joint resampling: a constant per-prompt difference has ZERO variance under pairing
    const = {f"p{i}": 0.1 * (i % 10) - 0.3 for i in range(60)}
    e3 = directed_effect(base, const, +1)
    check("constant paired difference gives a degenerate CI",
          abs(e3.ci_hi - e3.ci_lo) < 1e-9, f"[{e3.ci_lo:.6f},{e3.ci_hi:.6f}]")
    check("point equals the constant", abs(e3.point - 0.3) < 1e-9, f"{e3.point}")


def test_ci_excludes_zero_flag() -> None:
    """Pins: an interval straddling zero being reported as a positive finding."""
    print("test_ci_excludes_zero_flag")
    base = {f"p{i}": 0.9 for i in range(50)}
    big = {f"p{i}": 0.1 for i in range(50)}
    check("clear effect excludes zero", directed_effect(base, big, +1).excludes_zero)
    # alternating +/- differences straddle zero
    noisy = {f"p{i}": 0.9 - (0.4 if i % 2 else -0.4) for i in range(50)}
    e = directed_effect(base, noisy, +1)
    check("null effect does not exclude zero", not e.excludes_zero,
          f"[{e.ci_lo:.3f},{e.ci_hi:.3f}]")


def test_method_contrast_is_a_direct_paired_difference() -> None:
    """Pins: inferring 'A beats B' from two separate intervals instead of their difference."""
    print("test_method_contrast_is_a_direct_paired_difference")
    ids = [f"p{i}" for i in range(50)]
    b_a = {i: 0.8 for i in ids}
    i_a = {i: 0.3 for i in ids}   # effect +0.5
    b_b = {i: 0.8 for i in ids}
    i_b = {i: 0.5 for i in ids}   # effect +0.3
    c = method_contrast(b_a, i_a, b_b, i_b, sign=+1)
    check("contrast equals the difference of effects", abs(c.point - 0.2) < 1e-9, f"{c.point}")
    check("contrast uses one common intersection", c.n_paired == 50, str(c.n_paired))
    # equal methods -> contrast zero and not significant
    c2 = method_contrast(b_a, i_a, b_a, i_a, sign=+1)
    check("identical methods contrast to zero", abs(c2.point) < 1e-12, f"{c2.point}")
    # intersection narrows to prompts valid in all four arms
    i_b2 = dict(i_b); i_b2["p0"] = None
    check("contrast intersection respects all four arms",
          method_contrast(b_a, i_a, b_b, i_b2, +1).n_paired == 49)


def test_worst_case_bounds_never_impute_zero() -> None:
    """Pins: missing harmfulness scores imputed as 0, which fabricates a safety gain."""
    print("test_worst_case_bounds_never_impute_zero")
    base = {"p1": 0.8, "p2": 0.8, "p3": 0.8}
    inter = {"p1": 0.2, "p2": None, "p3": 0.2}   # p2 missing
    w = worst_case_bounds(base, inter, sign=+1)
    check("missing counted", w["n_missing"] == 1, str(w["n_missing"]))
    check("all prompts in denominator", w["n_all_prompts"] == 3)
    check("bounds bracket the observed effect", w["worst_case_low"] < 0.6 < w["worst_case_high"],
          f"[{w['worst_case_low']},{w['worst_case_high']}]")
    # p2's baseline is OBSERVED (0.8) so it is kept; only the missing intervention value is
    # filled adversarially (1.0 for the low bound), giving (0.6 + 0.6 + (0.8-1.0))/3.
    # TOL is 1e-6: worst_case_bounds rounds its outputs to 6 decimal places.
    TOL = 1e-6
    check("low bound fills only the missing value, keeping observed ones",
          abs(w["worst_case_low"] - ((0.6 + 0.6 + (0.8 - 1.0)) / 3)) < TOL,
          str(w["worst_case_low"]))
    check("high bound fills the missing value favourably",
          abs(w["worst_case_high"] - ((0.6 + 0.6 + (0.8 - 0.0)) / 3)) < TOL,
          str(w["worst_case_high"]))
    # a missing BASELINE is filled adversarially too, in the opposite direction
    w2 = worst_case_bounds({"p1": 0.8, "p2": None}, {"p1": 0.2, "p2": 0.2}, sign=+1)
    # reversal sign: the bug real data exposed -- bounds inverted when sign = -1
    wr = worst_case_bounds({"p1": 0.2, "p2": 0.2, "p3": None}, {"p1": 0.6, "p2": None, "p3": 0.6}, sign=-1)
    check("sign=-1: low <= high", wr["worst_case_low"] <= wr["worst_case_high"],
          f"[{wr['worst_case_low']},{wr['worst_case_high']}]")
    # low: missing baseline -> 1, missing intervention -> 0; contributions -(0.2-0.6), -(0.2-0), -(1-0.6)
    check("sign=-1: low bound is the least favourable fill",
          abs(wr["worst_case_low"] - ((0.4 - 0.2 - 0.4) / 3)) < TOL, str(wr["worst_case_low"]))
    check("sign=-1: high bound is the most favourable fill",
          abs(wr["worst_case_high"] - ((0.4 + 0.8 + 0.6) / 3)) < TOL, str(wr["worst_case_high"]))
    check("missing baseline filled adversarially",
          abs(w2["worst_case_low"] - ((0.6 + (0.0 - 0.2)) / 2)) < TOL,
          str(w2["worst_case_low"]))


def test_rate_difference_pairs_binary_outcomes() -> None:
    """Pins: comparing refusal rates on different prompt subsets."""
    print("test_rate_difference_pairs_binary_outcomes")
    a = {f"p{i}": (i < 30) for i in range(60)}   # 0.5
    b = {f"p{i}": (i < 12) for i in range(60)}   # 0.2
    e = rate_difference(a, b)
    check("paired rate difference is 0.30", abs(e.point - 0.3) < 1e-9, f"{e.point}")
    a2 = dict(a); a2["p0"] = None
    check("invalid rows drop from the pair", rate_difference(a2, b).n_paired == 59)


def test_judge_agreement_keeps_abstentions_separate() -> None:
    """Pins: an abstention scored as 0, inflating apparent agreement."""
    print("test_judge_agreement_keeps_abstentions_separate")
    ja = {"p1": 0.5, "p2": 0.7, "p3": None, "p4": 0.2}
    jb = {"p1": 0.5, "p2": 0.6, "p3": 0.9, "p4": None}
    r = judge_agreement(ja, jb)
    check("intersection excludes abstentions", r["n_matched_intersection"] == 2,
          str(r["n_matched_intersection"]))
    check("abstentions counted per judge",
          (r["abstentions_judge_a"], r["abstentions_judge_b"]) == (1, 1))
    check("exact agreement on the intersection", abs(r["exact_agreement"] - 0.5) < 1e-9,
          str(r["exact_agreement"]))
    perfect = judge_agreement({"a": 1.0, "b": 2.0, "c": 3.0}, {"a": 1.0, "b": 2.0, "c": 3.0})
    check("identical judges give rho 1.0", abs(perfect["spearman_rho"] - 1.0) < 1e-9)
    inverse = judge_agreement({"a": 1.0, "b": 2.0, "c": 3.0}, {"a": 3.0, "b": 2.0, "c": 1.0})
    check("inverted judges give rho -1.0", abs(inverse["spearman_rho"] + 1.0) < 1e-9)


def test_empty_intersection_is_nan_not_zero() -> None:
    """Pins: an empty intersection silently reported as a zero effect."""
    print("test_empty_intersection_is_nan_not_zero")
    e = directed_effect({"p1": None}, {"p1": 0.5}, +1)
    check("n_paired is 0", e.n_paired == 0)
    check("point is nan, not 0.0", e.point != e.point)
    check("not flagged significant", not e.excludes_zero)


def main() -> int:
    for t in (test_strict_intersection_drops_invalid,
              test_directed_effect_sign_convention,
              test_bootstrap_is_deterministic_and_joint,
              test_ci_excludes_zero_flag,
              test_method_contrast_is_a_direct_paired_difference,
              test_worst_case_bounds_never_impute_zero,
              test_rate_difference_pairs_binary_outcomes,
              test_judge_agreement_keeps_abstentions_separate,
              test_empty_intersection_is_nan_not_zero):
        t()
        print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
