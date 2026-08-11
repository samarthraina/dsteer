"""Score steering arms on the prompts where the two checkpoints actually differ.

Most of a harmful-prompt benchmark is handled the same way by both checkpoints, so a mean
over the whole set averages the informative prompts into the uninformative ones and
reports a diluted effect as a null. Conditioning on the prompts that carry a difference is
what makes a transfer number mean anything -- and it is also what the resemblance metric
scores *highest* on the wrong half of, which is the point of the paper.

The subset is fixed by the two unsteered checkpoints alone: prompts where their refusal
scores differ by at least `--threshold`. It never looks at a steered arm, so adding an arm
cannot move the subset it is judged on.

Effects are reported against each arm's own random control where one is given, because a
perturbation of any direction moves behaviour and the size of that non-specific movement
grows with the magnitude applied.

Three things here exist because getting them wrong produces a plausible number rather than
an error.

**Every quantity is read on one common set of prompts.** An arm that degenerates on eight
prompts has no score for them, and a mean over the remaining ninety-two compared against a
gap computed over all hundred is a comparison between two different prompt sets. Baseline,
floor, arm and control are intersected first, and what was dropped is reported by reason:
degeneration and judge abstention are different failures and are counted apart.

**The interval is on the reported effect, not on an intermediate.** The bootstrap resamples
prompts and recomputes the gap, the arm and the null on every draw, so uncertainty in the
endpoints reaches the number the paper quotes.

**Which lambda is read is printed, always.** An addition sweep holds a grid, and picking
the largest magnitude is right only while the largest magnitude is the measured ceiling.
Scoring one more point past it would otherwise move a headline number silently.

    python scripts/conditioned_analysis.py \
        --it-scored  outputs/steer/llama3-oh/harmfulqa/it/scored/baseline_scored.jsonl \
        --dpo-scored outputs/steer/llama3-oh/harmfulqa/dpo_ho300/scored/baseline_scored.jsonl \
        --arm checkpoint=outputs/steer/llama3-oh/harmfulqa/dpo_cpsame_ablate_ho300 \
        --arm refusal=outputs/steer/llama3-oh/harmfulqa/dpo_refusal_ablate \
        --control outputs/steer/llama3-oh/harmfulqa/dpo_refusal_ablate_random
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steering.utils import read_jsonl

METRICS = ("refusal", "harmfulness")
BOOT = 4000


def load(path: Path) -> Dict[str, dict]:
    """{id: record} for one scored file."""
    return {r["id"]: r for r in read_jsonl(path)}


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def scored_ids(recs: Dict[str, dict], metric: str) -> Set[str]:
    return {i for i, r in recs.items() if r.get(f"{metric}_score") is not None}


def losses(recs: Dict[str, dict], metric: str, within: Set[str]) -> Dict[str, int]:
    """Why records inside `within` carry no score.

    Degeneration and abstention pull in opposite directions and are not interchangeable:
    a broken generation is the model failing, an abstention is the judge declining -- and
    in the de-align direction the responses a judge declines to score are the unsafe ones.
    """
    broken = sum(1 for i in within if i in recs and not recs[i].get("valid", True))
    missing = sum(1 for i in within if i not in recs)
    unscored = sum(1 for i in within
                   if i in recs and recs[i].get("valid", True)
                   and recs[i].get(f"{metric}_score") is None)
    return {"broken": broken, "abstained": unscored, "absent": missing}


def steered_file(arm: Path, at_lambda: Optional[float]) -> Path:
    """The scored file for an arm: given directly, chosen by lambda, or the largest run.

    An ablation arm has the single coefficient 1.0. An addition sweep carries a grid and is
    reported at its own measured ceiling, which is the largest magnitude it ran -- true only
    while nothing past the ceiling has been scored, so the choice is printed and the
    alternatives with it.
    """
    if arm.is_file():
        return arm
    hits = sorted(glob.glob(str(arm / "scored" / "lambda_*_scored.jsonl")))
    if not hits:
        raise FileNotFoundError(f"no scored lambda file under {arm}/scored")

    def lam(p: str) -> float:
        return float(Path(p).name.split("lambda_")[1].split("_scored")[0])

    if at_lambda is not None:
        for h in hits:
            if abs(lam(h) - at_lambda) < 1e-9:
                return Path(h)
        raise FileNotFoundError(
            f"no lambda {at_lambda:+.3f} under {arm}/scored; have "
            f"{', '.join(format(lam(h), '+.3f') for h in hits)}")

    pick = max(hits, key=lambda p: abs(lam(p)))
    if len(hits) > 1:
        print(f"  {arm.name}: read at {lam(pick):+.3f} of "
              f"{', '.join(format(lam(h), '+.3f') for h in hits)}")
    return Path(pick)


def interval(ids: Sequence[str], base: Dict[str, float], floor: Dict[str, float],
             arm: Dict[str, float], ctrl: Optional[Dict[str, float]],
             lo: float = 2.5, hi: float = 97.5) -> tuple:
    """Bootstrap the reported effect, resampling prompts.

    The gap, the arm and the null are all recomputed on each draw, so uncertainty in the
    two unsteered endpoints reaches the interval instead of being treated as exact.
    """
    rng, k, out = random.Random(0), len(ids), []
    for _ in range(BOOT):
        s = [ids[rng.randrange(k)] for _ in range(k)]
        b = mean(base[i] for i in s)
        gap = b - mean(floor[i] for i in s)
        if gap == 0:
            continue
        a = 100.0 * (b - mean(arm[i] for i in s)) / gap
        c = 100.0 * (b - mean(ctrl[i] for i in s)) / gap if ctrl else 0.0
        out.append(a - c)
    if not out:
        return float("nan"), float("nan")
    out.sort()
    return out[int(lo / 100 * len(out))], out[int(hi / 100 * len(out))]


def main() -> int:
    p = argparse.ArgumentParser(description="Conditioned analysis on the prompts that carry a difference.")
    p.add_argument("--it-scored", required=True, help="Scored baseline of the instruction-tuned checkpoint.")
    p.add_argument("--dpo-scored", required=True, help="Scored baseline of the aligned checkpoint.")
    p.add_argument("--arm", action="append", default=[], metavar="NAME=DIR[::CONTROL]",
                   help="A steering arm to evaluate. Repeatable.")
    p.add_argument("--control", default=None,
                   help="Arm directory whose effect is subtracted from every arm without "
                        "its own. Under ablation one control serves every direction at the "
                        "same layer set, because the direction is normalised before "
                        "projecting; under addition each arm needs its own, norm-matched.")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Minimum refusal difference between the checkpoints. Fixed a "
                        "priori as 'clearly different'; not tuned on the split.")
    p.add_argument("--direction", choices=["remove", "install"], default="remove",
                   help="Which way the arm steers. 'remove' starts at the aligned "
                        "checkpoint and moves toward the unaligned one; 'install' is the "
                        "reverse. This decides which checkpoint is the starting point and "
                        "which is the target, and a share-of-gap computed against the "
                        "wrong one is not small or noisy -- it is meaningless, while still "
                        "printing a plausible percentage.")
    p.add_argument("--at-lambda", type=float, default=None,
                   help="Read every sweep at this lambda instead of at its largest "
                        "magnitude. Use it to pin a grid that may grow.")
    p.add_argument("--strict-common", action="store_true",
                   help="Read every arm on the prompts scored in *all* of them, not each "
                        "arm's own intersection. Costs power; makes the cells exactly "
                        "comparable, which is what a table of arms side by side implies.")
    p.add_argument("--compare", action="append", default=[], metavar="ARM_A,ARM_B",
                   help="Bootstrap the difference between two arms on the prompts both "
                        "scored. Repeatable. Two overlapping single-arm intervals do not "
                        "establish that the arms are the same; this does.")
    p.add_argument("--output", default=None, help="Write the numbers as JSON.")
    args = p.parse_args()

    it_recs, dpo_recs = load(Path(args.it_scored)), load(Path(args.dpo_scored))
    it_ref = {i: r["refusal_score"] for i, r in it_recs.items() if r.get("refusal_score") is not None}
    dpo_ref = {i: r["refusal_score"] for i, r in dpo_recs.items() if r.get("refusal_score") is not None}
    shared = it_ref.keys() & dpo_ref.keys()
    subset = {i for i in shared if abs(dpo_ref[i] - it_ref[i]) >= args.threshold}
    print(f"{len(shared)} prompts scored on both checkpoints; "
          f"{len(subset)} differ by >= {args.threshold}\n")

    print("reading arms")
    arms = {}
    for spec in args.arm:
        name, _, rest = spec.partition("=")
        arm_path, _, ctrl_path = rest.partition("::")
        arms[name] = (Path(arm_path), ctrl_path or args.control)
    # An arm read at one lambda against a control read at another compares two amounts of
    # intervention, so a control that never ran the pinned point is fatal rather than
    # something to fall back from.
    try:
        files = {n: steered_file(a, args.at_lambda) for n, (a, _) in arms.items()}
        ctrl_files = {c: steered_file(Path(c), args.at_lambda)
                      for _, c in arms.values() if c}
    except FileNotFoundError as e:
        raise SystemExit(f"{e}\nArm and control must both have been run at the "
                         f"lambda being read.")
    recs = {n: load(f) for n, f in files.items()}
    ctrl_recs = {c: load(f) for c, f in ctrl_files.items()}

    # Restricting to the prompts every arm scored makes the cells comparable at the cost of
    # the union of their failures; each arm's own intersection keeps more power but lets
    # two cells sit on slightly different prompts.
    def common(metric: str, names: Optional[Sequence[str]]) -> List[str]:
        ok = subset & scored_ids(it_recs, metric) & scored_ids(dpo_recs, metric)
        wanted = list(recs) if args.strict_common or names is None else list(names)
        for n in wanted:
            ok &= scored_ids(recs[n], metric)
            c = arms[n][1]
            if c:
                ok &= scored_ids(ctrl_recs[c], metric)
        return sorted(ok)

    def specific(ids, metric, name, base, floor):
        """One arm's null-corrected effect on a given prompt set."""
        b = mean(base[i] for i in ids)
        gap = b - mean(floor[i] for i in ids)
        if gap == 0:
            return float("nan")
        a = 100.0 * (b - mean(recs[name][i][f"{metric}_score"] for i in ids)) / gap
        c = arms[name][1]
        n0 = (100.0 * (b - mean(ctrl_recs[c][i][f"{metric}_score"] for i in ids)) / gap
              if c else 0.0)
        return a - n0

    print()
    start_lbl, target_lbl = ("aligned", "unaligned") if args.direction == "remove" \
        else ("unaligned", "aligned")
    results: Dict[str, dict] = {}
    header = (f"\n{'arm':<20}{'metric':<13}{'n':>5}{'mean':>8}{'% gap':>8}{'null':>7}"
              f"{'specific':>10}{'95% interval':>20}{'brk':>5}{'abs':>5}")
    print(header)

    for name, (arm_path, ctrl_path) in arms.items():
        row: Dict[str, dict] = {"file": str(files[name]), "control": ctrl_path}
        for m in METRICS:
            ids = common(m, [name])
            if not ids:
                print(f"{name:<20}{m:<13}    0   no prompts scored in every file")
                continue
            it_m = {i: it_recs[i][f"{m}_score"] for i in ids}
            dpo_m = {i: dpo_recs[i][f"{m}_score"] for i in ids}
            base, floor = (dpo_m, it_m) if args.direction == "remove" else (it_m, dpo_m)
            arm_m = {i: recs[name][i][f"{m}_score"] for i in ids}
            ctrl_m = ({i: ctrl_recs[ctrl_path][i][f"{m}_score"] for i in ids}
                      if ctrl_path else None)

            b, f = mean(base[i] for i in ids), mean(floor[i] for i in ids)
            gap = b - f
            pct = 100.0 * (b - mean(arm_m[i] for i in ids)) / gap if gap else float("nan")
            null = (100.0 * (b - mean(ctrl_m[i] for i in ids)) / gap
                    if ctrl_m and gap else 0.0)
            lo, hi = interval(ids, base, floor, arm_m, ctrl_m)
            lost = losses(recs[name], m, subset)

            row[m] = {"n": len(ids), "mean": mean(arm_m[i] for i in ids),
                      "start": b, "target": f, "gap": gap,
                      "pct_of_gap": pct, "null_pct": null, "specific_points": pct - null,
                      "ci_lo": lo, "ci_hi": hi, **lost}
            print(f"{name:<20}{m:<13}{len(ids):>5}{row[m]['mean']:>8.3f}{pct:>7.1f}%"
                  f"{null:>6.1f}%{pct - null:>10.1f}"
                  f"{'[' + format(lo, '.1f') + ', ' + format(hi, '.1f') + ']':>20}"
                  f"{lost['broken']:>5}{lost['abstained']:>5}")
        results[name] = row

    # Two arms read on the same prompts share their baseline, their floor and most of their
    # prompt-level noise, so the interval on the *difference* is much tighter than the two
    # arms' own intervals suggest. Overlapping single-arm intervals are only weak evidence
    # of no difference; this is the test that settles it.
    for spec in args.compare:
        a_name, _, b_name = spec.partition(",")
        if a_name not in arms or b_name not in arms:
            raise SystemExit(f"--compare {spec}: unknown arm")
        print(f"\n{a_name} minus {b_name}, paired on the prompts both scored")
        for m in METRICS:
            ids = common(m, [a_name, b_name])
            if not ids:
                print(f"  {m:<13} no prompts scored in both"); continue
            it_m = {i: it_recs[i][f"{m}_score"] for i in ids}
            dpo_m = {i: dpo_recs[i][f"{m}_score"] for i in ids}
            base, floor = (dpo_m, it_m) if args.direction == "remove" else (it_m, dpo_m)
            point = specific(ids, m, a_name, base, floor) - specific(ids, m, b_name, base, floor)
            rng, k, draws = random.Random(0), len(ids), []
            for _ in range(BOOT):
                s = [ids[rng.randrange(k)] for _ in range(k)]
                d = specific(s, m, a_name, base, floor) - specific(s, m, b_name, base, floor)
                if d == d:
                    draws.append(d)
            draws.sort()
            lo, hi = draws[int(.025 * len(draws))], draws[int(.975 * len(draws))]
            verdict = "excludes zero" if lo > 0 or hi < 0 else "includes zero"
            print(f"  {m:<13} n={len(ids):<4} {point:+7.1f}  "
                  f"[{lo:+.1f}, {hi:+.1f}]  {verdict}")

    # The endpoints move with the prompt set, so they are reported per metric on the set
    # actually used rather than once at the top.
    print(f"\nendpoints on each metric's common set ({start_lbl} start -> {target_lbl} target)")
    for m in METRICS:
        ids = common(m, None)
        if not ids:
            continue
        it_mean = mean(it_recs[i][f"{m}_score"] for i in ids)
        dpo_mean = mean(dpo_recs[i][f"{m}_score"] for i in ids)
        b, f = (dpo_mean, it_mean) if args.direction == "remove" else (it_mean, dpo_mean)
        print(f"  {m:<13} n={len(ids):<5} {b:.3f} -> {f:.3f}  gap {b - f:+.3f}")
    print(f"\n  brk = degenerate, judged invalid.  abs = judge declined to score.\n"
          f"  Both are excluded from the mean and from the gap it is read against.")

    if args.output:
        payload = {"n_subset": len(subset), "threshold": args.threshold,
                   "direction": args.direction, "at_lambda": args.at_lambda,
                   "strict_common": args.strict_common, "arms": results}
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
