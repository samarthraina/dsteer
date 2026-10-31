#!/usr/bin/env python3
"""Paired prompt-bootstrap statistics. Pure stdlib, CPU only, fully testable.

Rules enforced here, from the handoff:
  - 10,000 replicates, seed 20261028
  - prompt IDs resampled JOINTLY across arms, never independently
  - strict per-estimand intersections: a prompt counts only if present and valid in every
    arm that estimand needs
  - a direct paired difference for method contrasts; a difference is never inferred from
    two separate significance tests
  - worst-case bounds on [0,1] for missing scores, never zero-imputation
  - model pairs and datasets are never pooled (the caller passes one cell at a time)
"""
from __future__ import annotations

import random
from dataclasses import dataclass

BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261028


@dataclass(frozen=True)
class Estimate:
    point: float
    ci_lo: float
    ci_hi: float
    n_paired: int
    excludes_zero: bool

    def as_dict(self) -> dict:
        return {"point": round(self.point, 6), "ci_lo": round(self.ci_lo, 6),
                "ci_hi": round(self.ci_hi, 6), "n_paired": self.n_paired,
                "excludes_zero": self.excludes_zero}


def strict_intersection(*arms: dict[str, float | None]) -> list[str]:
    """Prompt ids present AND non-None in every arm. Sorted, so results are deterministic."""
    if not arms:
        return []
    keep = set(arms[0])
    for a in arms:
        keep &= {k for k, v in a.items() if v is not None}
    return sorted(keep)


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def paired_bootstrap(
    ids: list[str],
    statistic,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> Estimate:
    """Bootstrap `statistic(sample_ids)` by resampling prompt ids jointly.

    `statistic` receives a list of ids (with repeats) and returns a scalar. Because every
    arm is indexed by the same resampled id list, arms stay paired by construction.
    """
    if not ids:
        return Estimate(float("nan"), float("nan"), float("nan"), 0, False)
    point = statistic(ids)
    rng = random.Random(seed)
    n = len(ids)
    reps_out = []
    for _ in range(reps):
        sample = [ids[rng.randrange(n)] for _ in range(n)]
        reps_out.append(statistic(sample))
    lo, hi = _percentile(reps_out, 0.025), _percentile(reps_out, 0.975)
    return Estimate(point, lo, hi, n, (lo > 0 and hi > 0) or (lo < 0 and hi < 0))


def directed_effect(
    baseline: dict[str, float | None],
    intervention: dict[str, float | None],
    sign: int,
) -> Estimate:
    """Direction-oriented paired effect, so positive always means the intended movement.

    sign=+1 for installation (baseline minus intervention: harmfulness should fall)
    sign=-1 for reversal     (intervention minus baseline: harmfulness should rise)
    """
    if sign not in (1, -1):
        raise ValueError("sign must be +1 or -1")
    ids = strict_intersection(baseline, intervention)

    def stat(sample: list[str]) -> float:
        if not sample:
            return float("nan")
        return sum(sign * (baseline[i] - intervention[i]) for i in sample) / len(sample)

    return paired_bootstrap(ids, stat)


def method_contrast(
    baseline_a: dict[str, float | None], intervention_a: dict[str, float | None],
    baseline_b: dict[str, float | None], intervention_b: dict[str, float | None],
    sign: int,
) -> Estimate:
    """Direct paired difference of two methods' directed effects, on ONE common intersection.

    This is the only admissible way to claim one method beats another. Comparing two
    separately-computed confidence intervals is not a test and is not offered here.
    """
    ids = strict_intersection(baseline_a, intervention_a, baseline_b, intervention_b)

    def stat(sample: list[str]) -> float:
        if not sample:
            return float("nan")
        tot = 0.0
        for i in sample:
            ea = sign * (baseline_a[i] - intervention_a[i])
            eb = sign * (baseline_b[i] - intervention_b[i])
            tot += ea - eb
        return tot / len(sample)

    return paired_bootstrap(ids, stat)


def worst_case_bounds(
    baseline: dict[str, float | None],
    intervention: dict[str, float | None],
    sign: int,
    score_min: float = 0.0,
    score_max: float = 1.0,
) -> dict:
    """Bound the directed effect over all prompts, filling missing scores adversarially.

    Missing harmfulness scores are never imputed as zero. Each bound substitutes the value
    that is least and most favourable to the intended direction, on the stated score range.
    """
    all_ids = sorted(set(baseline) | set(intervention))

    def fill(low: bool) -> float:
        # the contribution is sign*(b - v); to minimise it, push b down and v up when sign>0,
        # and the reverse when sign<0. The maximum is the mirror image.
        b_fill = score_min if (low == (sign > 0)) else score_max
        v_fill = score_max if (low == (sign > 0)) else score_min
        tot = 0.0
        for i in all_ids:
            b, v = baseline.get(i), intervention.get(i)
            tot += sign * ((b_fill if b is None else b) - (v_fill if v is None else v))
        return tot / len(all_ids) if all_ids else float("nan")

    observed = strict_intersection(baseline, intervention)
    return {
        "n_all_prompts": len(all_ids),
        "n_observed_intersection": len(observed),
        "n_missing": len(all_ids) - len(observed),
        "worst_case_low": round(fill(True), 6),
        "worst_case_high": round(fill(False), 6),
        "score_range": [score_min, score_max],
        "method": "adversarial substitution on the score range; zero-imputation is not used",
    }


def rate_difference(
    arm_a: dict[str, bool | None], arm_b: dict[str, bool | None]
) -> Estimate:
    """Paired difference in a binary rate (e.g. full-refusal on safe prompts), a minus b."""
    ids = strict_intersection(
        {k: (None if v is None else float(v)) for k, v in arm_a.items()},
        {k: (None if v is None else float(v)) for k, v in arm_b.items()},
    )

    def stat(sample: list[str]) -> float:
        if not sample:
            return float("nan")
        return sum(float(arm_a[i]) - float(arm_b[i]) for i in sample) / len(sample)

    return paired_bootstrap(ids, stat)


def judge_agreement(
    judge_a: dict[str, float | None], judge_b: dict[str, float | None]
) -> dict:
    """Agreement between two judges on their matched intersection.

    Abstentions are reported separately, never coerced to a score.
    """
    ids = strict_intersection(judge_a, judge_b)
    a = [judge_a[i] for i in ids]
    b = [judge_b[i] for i in ids]
    n = len(ids)

    exact = sum(1 for x, y in zip(a, b) if x == y) / n if n else float("nan")
    mean_abs = sum(abs(x - y) for x, y in zip(a, b)) / n if n else float("nan")

    # Spearman via rank Pearson, average ranks for ties
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rho = float("nan")
    if n > 1:
        ra, rb = ranks(a), ranks(b)
        ma, mb = sum(ra) / n, sum(rb) / n
        num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
        da = sum((x - ma) ** 2 for x in ra) ** 0.5
        db = sum((y - mb) ** 2 for y in rb) ** 0.5
        rho = num / (da * db) if da > 0 and db > 0 else float("nan")

    return {
        "n_matched_intersection": n,
        "exact_agreement": None if n == 0 else round(exact, 6),
        "mean_absolute_difference": None if n == 0 else round(mean_abs, 6),
        "spearman_rho": None if n <= 1 else round(rho, 6),
        "abstentions_judge_a": sum(1 for v in judge_a.values() if v is None),
        "abstentions_judge_b": sum(1 for v in judge_b.values() if v is None),
        "note": "abstentions excluded from the intersection and reported separately, never scored 0",
    }
