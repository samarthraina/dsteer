#!/usr/bin/env python3
"""Budget estimate from MEASURED throughput of the original A100 runs.

Throughput is not guessed. It is derived from timestamps and record counts in the archived
run metadata of the original runs, on the same hardware (A100-SXM4-80GB, 1 GPU), same
decoding and same token caps:

  generation  figures-final/data/task027-final/steer/*/run_meta.json
              29 conditions x 300 prompts, 2026-08-18T06:11:32Z -> 08:00:48Z
              median 3.8 min/condition, range 3.3-4.5 min
              -> conservative 4.5 min per 300 responses = 4000 responses/GPU-h

  judging     figures-final/data/task026a-final/outputs-026a/scored/*/run_meta.json
              5 scoring jobs, input_files x 200-prompt calibration panel
              observed 1543-2697 responses/GPU-h across consecutive jobs
              -> conservative 1543 responses/GPU-h (full metric set)

Both use the SLOWEST observed rate. Deltas include the following job's setup, so the
underlying rates are faster than modelled. Activation extraction is prefill-only and was
not separately timed; it is allocated generously rather than extrapolated.

Run:  python3 -I scripts/estimate_budget.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import statistics
from pathlib import Path

# ---- measured constants (slowest observed; see module docstring)
GEN_PER_H = 4000.0          # responses/GPU-hour, greedy, 768-token cap
JUDGE_PER_H = 1543.0        # responses/GPU-hour, full metric set, 256 output tokens
MODEL_LOAD_H = 5 / 60       # per load/unload cycle
EXTRACT_H_PER_ENDPOINT = 0.5  # prefill-only activation extraction, generous allowance
HEADROOM = 1.30             # handoff requires +30%
CEILING = 30.0


def measure_generation(project: Path) -> dict:
    rows = []
    for f in glob.glob(str(project / "figures-final/data/task027-final/steer/*/run_meta.json")):
        d = json.loads(Path(f).read_text())
        rows.append((_dt.datetime.fromisoformat(d["environment"]["timestamp_utc"]),
                     d.get("harmfulqa_record_count")))
    rows.sort()
    deltas = [(rows[i + 1][0] - rows[i][0]).total_seconds() / 60 for i in range(len(rows) - 1)]
    usable = [d for d in deltas if 0 < d < 240]
    n = rows[0][1] or 300
    return {
        "source": "task027-final/steer/*/run_meta.json",
        "conditions": len(rows), "prompts_per_condition": n,
        "span_hours": round((rows[-1][0] - rows[0][0]).total_seconds() / 3600, 3),
        "median_min_per_condition": round(statistics.median(usable), 2),
        "max_min_per_condition": round(max(usable), 2),
        "median_responses_per_gpu_hour": round(n / (statistics.median(usable) / 60)),
        "conservative_responses_per_gpu_hour": round(n / (max(usable) / 60)),
        "hardware": "NVIDIA A100-SXM4-80GB x1",
    }


def h_gen(n: int) -> float:
    return n / GEN_PER_H


def h_judge(n: int) -> float:
    return n / JUDGE_PER_H


def build(scenario: str) -> dict:
    """scenario: 'recovered' (originals available) or 'replication' (originals unavailable)."""
    recovered = scenario == "recovered"
    stages: list[dict] = []

    def add(stage: str, detail: str, hours: float, n: int | None = None) -> None:
        stages.append({"stage": stage, "detail": detail, "responses": n,
                       "hours_raw": round(hours, 3)})

    # ---- loading, parity, profiling
    add("load_parity_profile",
        "4 endpoint loads + judge load + zero-intervention parity + throughput profile",
        5 * MODEL_LOAD_H + h_gen(240) + h_judge(240), 480)

    # ---- direction reconstruction (required: tensors absent)
    add("dsteer_direction_reconstruction",
        "activation extraction, 1378 construction prompts x 2 endpoints x 2 pairs",
        4 * EXTRACT_H_PER_ENDPOINT + 4 * MODEL_LOAD_H)
    add("refusal_direction_construction",
        "256 harmful + 256 harmless x 2 intervened endpoints x 2 pairs",
        4 * EXTRACT_H_PER_ENDPOINT + 4 * MODEL_LOAD_H)

    # ---- Experiment A calibration
    cal = 3 * 60 * 2 * 2          # 3 multipliers x 60 dev x 2 directions x 2 pairs
    ver = 200 * 1 * 2 * 2         # selected candidate on 200 calibration x 2 dir x 2 pairs
    add("refusal_calibration", f"dev screen {cal} + calibration verify {ver}",
        h_gen(cal + ver) + h_judge(cal + ver) + 2 * MODEL_LOAD_H, cal + ver)

    # ---- Experiment A harmful panel
    if recovered:
        new_cond = 2              # only the two refusal comparator arms are new
        note_a = "4 of 6 conditions per pair reused from recovered originals"
    else:
        new_cond = 6              # everything regenerated under one common protocol
        note_a = "all 6 conditions per pair regenerated -- REPLICATION, not reuse"
    harm = new_cond * 300 * 2
    add("harmful_panel_generation", f"{new_cond} conditions x 300 x 2 pairs ({note_a})",
        h_gen(harm) + 4 * MODEL_LOAD_H, harm)
    add("harmful_panel_original_judge", "original Qwen3.5-27B scoring of the new arms",
        h_judge(harm) + MODEL_LOAD_H, harm)

    # ---- Experiment B
    xs, ife = 250 * 6 * 2, 100 * 6 * 2
    add("xstest_generation", f"250 safe x 6 conditions x 2 pairs", h_gen(xs) + 4 * MODEL_LOAD_H, xs)
    add("xstest_rubric_scoring", "frozen XSTest-style refusal rubric via judge",
        h_judge(xs) + MODEL_LOAD_H, xs)
    add("ifeval_generation", f"100 prompts x 6 conditions x 2 pairs",
        h_gen(ife) + 4 * MODEL_LOAD_H, ife)
    add("ifeval_checkers", "official strict/loose checkers -- CPU, no GPU", 0.0, ife)

    # ---- Experiment C
    if recovered:
        add("independent_judge", "2400 recovered originals, independent judge only",
            h_judge(2400) + MODEL_LOAD_H, 2400)
    else:
        both = harm * 2
        add("independent_judge",
            f"REPLICATION: both judges score the SAME {harm} new responses ({both} judge items)",
            h_judge(both) + 2 * MODEL_LOAD_H, both)

    raw = sum(s["hours_raw"] for s in stages)
    withhead = raw * HEADROOM
    for s in stages:
        s["hours_with_headroom"] = round(s["hours_raw"] * HEADROOM, 2)

    return {
        "scenario": scenario,
        "stages": stages,
        "total_hours_raw": round(raw, 2),
        "total_hours_with_30pct_headroom": round(withhead, 2),
        "ceiling": CEILING,
        "fits": withhead <= CEILING,
        "slack_hours": round(CEILING - withhead, 2),
    }


def reduced(full: dict) -> dict:
    """Apply the handoff's priority cuts, in its stated order, until it fits."""
    cuts = []
    stages = {s["stage"]: dict(s) for s in full["stages"]}

    # cut 1: IFEval 100 -> 50
    stages["ifeval_generation"]["hours_raw"] = h_gen(50 * 6 * 2) + 4 * MODEL_LOAD_H
    stages["ifeval_generation"]["responses"] = 50 * 6 * 2
    cuts.append("IFEval 100 -> 50 prompts (preselected nested subset)")

    # cut 2: harmful test 300 -> 100
    n_cond = 6 if full["scenario"] == "replication" else 2
    harm100 = n_cond * 100 * 2
    stages["harmful_panel_generation"]["hours_raw"] = h_gen(harm100) + 4 * MODEL_LOAD_H
    stages["harmful_panel_generation"]["responses"] = harm100
    stages["harmful_panel_original_judge"]["hours_raw"] = h_judge(harm100) + MODEL_LOAD_H
    stages["harmful_panel_original_judge"]["responses"] = harm100
    cuts.append("harmful test 300 -> 100 (frozen nested fallback, salt dsteer-rebuttal-20261028)")

    # cut 3: independent judging -> the fixed 800-response fallback
    if full["scenario"] == "replication":
        stages["independent_judge"]["hours_raw"] = h_judge(800 * 2) + 2 * MODEL_LOAD_H
        stages["independent_judge"]["responses"] = 800 * 2
        stages["independent_judge"]["detail"] = ("REPLICATION fallback: both judges score the "
                                                 "same 800 preselected responses (1600 items)")
    else:
        stages["independent_judge"]["hours_raw"] = h_judge(800) + MODEL_LOAD_H
        stages["independent_judge"]["responses"] = 800
    cuts.append("independent judging -> fixed 800-response preselected subset")

    # cut 4: XSTest to the same 100-prompt scale if still needed
    out = list(stages.values())
    raw = sum(s["hours_raw"] for s in out)
    if raw * HEADROOM > CEILING:
        stages["xstest_generation"]["hours_raw"] = h_gen(125 * 6 * 2) + 4 * MODEL_LOAD_H
        stages["xstest_generation"]["responses"] = 125 * 6 * 2
        stages["xstest_rubric_scoring"]["hours_raw"] = h_judge(125 * 6 * 2) + MODEL_LOAD_H
        stages["xstest_rubric_scoring"]["responses"] = 125 * 6 * 2
        cuts.append("XSTest 250 -> 125 safe prompts (disclosed deviation)")
        out = list(stages.values())
        raw = sum(s["hours_raw"] for s in out)

    for s in out:
        s["hours_with_headroom"] = round(s["hours_raw"] * HEADROOM, 2)
        s["hours_raw"] = round(s["hours_raw"], 3)

    return {
        "scenario": full["scenario"] + "_reduced",
        "cuts_applied_in_handoff_priority_order": cuts,
        "preserved": ["both model pairs", "the matched additive equal-norm comparator",
                      "reference-baseline incompleteness remains visible"],
        "stages": out,
        "total_hours_raw": round(raw, 2),
        "total_hours_with_30pct_headroom": round(raw * HEADROOM, 2),
        "ceiling": CEILING,
        "fits": raw * HEADROOM <= CEILING,
        "slack_hours": round(CEILING - raw * HEADROOM, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default=".")
    ap.add_argument("--project-root", default="../..")
    a = ap.parse_args()
    run_root, project = Path(a.run_root).resolve(), Path(a.project_root).resolve()

    measured = measure_generation(project)
    out = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "basis": "MEASURED from archived original-run metadata, not estimated from nothing",
        "measured_generation": measured,
        "measured_judging": {
            "source": "task026a-final/outputs-026a/scored/*/run_meta.json",
            "jobs": 5, "observed_responses_per_gpu_hour_range": [1543, 2697],
            "conservative_responses_per_gpu_hour": JUDGE_PER_H,
            "note": "deltas include the next job's setup, so true rates are faster",
        },
        "modelling_constants": {
            "generation_responses_per_gpu_hour": GEN_PER_H,
            "judging_responses_per_gpu_hour": JUDGE_PER_H,
            "model_load_hours_per_cycle": round(MODEL_LOAD_H, 4),
            "activation_extraction_hours_per_endpoint": EXTRACT_H_PER_ENDPOINT,
            "headroom_multiplier": HEADROOM,
        },
        "unmeasured_assumptions": [
            "activation extraction allocated at 0.5 GPU-h per endpoint (prefill-only; never separately timed)",
            "model load/unload at 5 min per cycle",
            "download time for endpoints, judges and datasets excluded -- it bills from instance start and must be added once network conditions are known",
            "IFEval official checkers assumed CPU-only",
        ],
    }
    for sc in ("recovered", "replication"):
        full = build(sc)
        out[f"scope_{sc}_full"] = full
        out[f"scope_{sc}_reduced"] = reduced(full)

    p = run_root / "reports" / "BUDGET_ESTIMATE.json"
    p.write_text(json.dumps(out, indent=2) + "\n")

    print(f"MEASURED generation: median {measured['median_responses_per_gpu_hour']} resp/GPU-h, "
          f"conservative {measured['conservative_responses_per_gpu_hour']} resp/GPU-h")
    print(f"MEASURED judging   : conservative {JUDGE_PER_H:.0f} resp/GPU-h\n")
    for sc in ("recovered", "replication"):
        for v in ("full", "reduced"):
            d = out[f"scope_{sc}_{v}"]
            print(f"  {sc:12s} {v:8s} {d['total_hours_with_30pct_headroom']:6.2f} h "
                  f"(+30% headroom)  fits={d['fits']}  slack={d['slack_hours']:+.2f} h")
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
