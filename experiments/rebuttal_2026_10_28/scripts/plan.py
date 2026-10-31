#!/usr/bin/env python3
"""The executable stage plan and its budget, from one source of truth.

Every step carries the command run_plan.py executes AND the hours it is estimated to cost,
so the budget is the plan's cost by construction. Writes configs/stage_plan_<scope>.json and
reports/BUDGET_ESTIMATE_V2.json.

Rates are PROVISIONAL until profile_runtime replaces them on the box:
  generation  task027 run_meta: 29 run-START timestamps -> 28 inter-start intervals (median
              3.8 min, max 4.5 min per 300-prompt condition, each interval including that run's
              own model load). That is NOT a verified completion time for 29 runs: the last run's
              duration is unobserved. Modelled at the slowest interval, 300 / 4.5 min.
  orig judge  task026a scored run_meta: 5 start timestamps -> 4 intervals, 4 metrics per
              response; slowest 1,543 responses/h -> 6,172 metric calls/h. Intervals include the
              next job's start-up, so this is conservative.
  extraction  task025 gpu_usage.json: ~100 s for 1,378 prompts x 2 endpoints per pair.
  independent unmeasured: provisionally the original judge's per-call rate (phi-4 is about half
              the size, so likely conservative). MUST be profiled before its stage is admitted.
  unmeasured  judge start-up 10 min, generator load 2 min (added on top of the generation rate,
              which already includes a load -- deliberate double count), downloads at 50 MB/s,
              environment setup 25 min, sync-back 10 min.
Totals add a 10% retry allowance and then the handoff's 30% headroom.

Run:  python3 -I scripts/plan.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN_PER_H = 300 / (4.5 / 60)            # 4000 responses / GPU-h
ORIG_CALLS_PER_H = 1543 * 4             # 6172 metric calls / GPU-h
IND_CALLS_PER_H = ORIG_CALLS_PER_H       # provisional until profiled
EXTRACT_H_PER_PAIR = 100 / 3600 * 1.5    # measured 100 s, x1.5 for the second readout
SELECT_H_PER_PAIR = 130 * 180 / 32 * 0.12 / 3600   # 130 candidates x 180 prompts, batch 32, ~0.12 s/batch
GEN_LOAD_H, JUDGE_START_H = 2 / 60, 10 / 60
DOWNLOAD_GB = {"A_it": 16.1, "A_dpo": 16.1, "B_it": 14.5, "B_dpo_adapter": 0.7,
               "judge_original": 54.0, "judge_independent": 29.4, "envs": 12.0}
DL_H = sum(DOWNLOAD_GB.values()) * 1024 / 50 / 3600
ENV_H, SYNC_H, MERGE_H = 25 / 60, 10 / 60, 6 / 60
RETRY, HEADROOM, CEILING = 1.10, 1.30, 30.0
CAPS = {"S1_load_parity_profile": 2.0, "S2_refusal_construction_calibration": 4.0,
        "S3_harmful_generation_original_judging": 7.0, "S4_benign_panels": 6.0,
        "S5_independent_judging": 4.0, "S6_optional_projection": 2.0, "S7_reserve_export_shutdown": 5.0}
PY_GEN = ".venv-gen/bin/python"
PY_JUDGE = ".venv-judge/bin/python"


def gen(stage, pair, cond, panel, n, extra=()):
    return {"stage": stage, "kind": "generation", "env": "gen", "responses": n,
            "est_h": GEN_LOAD_H + n / GEN_PER_H,
            "argv": [PY_GEN, "scripts/run_panel.py", "--pair", pair, "--condition", cond, "--panel", panel,
                     "--stage", stage, "--estimate-hours", f"{GEN_LOAD_H + n / GEN_PER_H:.3f}", *extra]}


def score(stage, evaluator, pair, cond, panel, n, metrics, rate, extra=()):
    calls = n * len(metrics)
    return {"stage": stage, "kind": f"scoring:{evaluator}", "env": "judge", "metric_calls": calls,
            "est_h": calls / rate,
            "argv": [PY_JUDGE, "scripts/score_outputs.py", "--evaluator", evaluator, "--pair", pair,
                     "--condition", cond, "--panel", panel, "--metrics", ",".join(metrics),
                     "--judge-url", "http://localhost:" + ("8001" if evaluator == "independent" else "8000") + "/v1",
                     "--stage", stage, "--estimate-hours", f"{calls / rate:.3f}", *extra]}


def judge(stage, which, action):
    return {"stage": stage, "kind": f"judge_{action}", "env": "judge",
            "est_h": JUDGE_START_H if action == "start" else 0.0,
            "argv": [PY_JUDGE, "scripts/start_judge.py", action, "--judge", which,
                     "--port", "8001" if which == "independent" else "8000"]}


def cpu(stage, argv, est=0.02, env="gen"):
    return {"stage": stage, "kind": "cpu", "env": env, "est_h": est, "argv": [PY_GEN if env == "gen" else PY_JUDGE, *argv]}


def build(scope: str) -> list[dict]:
    reduced = scope == "reduced"
    H = 100 if reduced else 300                       # harmful test panel
    IFE = 50 if reduced else 100
    hpanel = "harmful_test_fallback100" if reduced else "harmful_test"
    ipanel = "instruction_fallback50" if reduced else "instruction_test"
    S1, S2, S3, S4, S5, S7 = (k for k in CAPS if not k.startswith("S6"))
    steps: list[dict] = []
    sides = {"A": ("it", "dpo"), "B": ("it", "dpo")}
    base = {"it": "it_base", "dpo": "dpo_base"}

    # ---------------- S1: environments, endpoints, dev baselines, parity, judge profiles
    dl_s1 = sum(v for k, v in DOWNLOAD_GB.items() if k != "judge_independent") * 1024 / 50 / 3600
    steps.append({"stage": S1, "kind": "setup", "env": "system", "est_h": ENV_H + dl_s1,
                  "argv": ["python3", "scripts/box_setup.py"],
                  "note": "envs + endpoint and original-judge downloads; phi-4 is fetched in S5"})
    steps.append({"stage": S1, "kind": "endpoints", "env": "gen", "est_h": MERGE_H + 4 * GEN_LOAD_H,
                  "argv": [PY_GEN, "scripts/materialise_endpoints.py", "--stage", S1, "--estimate-hours", f"{MERGE_H:.3f}"]})
    for p in "AB":
        for s in sides[p]:
            steps.append(gen(S1, p, base[s], "development", 60))
    for p, c in (("A", "it_base"), ("A", "dsteer_install"), ("B", "dpo_base"), ("B", "dsteer_reverse")):
        # one whole original batch: the frozen generator batches contiguous slices of 64 and the
        # recovered files keep the original order, so a 20-prompt prefix changes batch composition
        # (left-padding in bf16) and cannot test byte parity -- the first attempt did exactly that
        steps.append(gen(S1, p, c, "harmful_test", 64, ("--parity", "--limit", "64")))
    steps.append(cpu(S1, ["scripts/parity_check.py"]))
    steps.append(judge(S1, "original", "start"))
    for p in "AB":
        for s in sides[p]:
            steps.append(score(S1, "original", p, base[s], "development", 60, ["harmfulness", "quality", "refusal"], ORIG_CALLS_PER_H))
    steps.append(judge(S1, "original", "stop"))
    steps.append(cpu(S1, ["scripts/profile_runtime.py"], env="gen"))

    # ---------------- S2: refusal directions, reference selection, additive calibration
    for p in "AB":
        for s in sides[p]:
            steps.append({"stage": S2, "kind": "extraction", "env": "gen", "est_h": GEN_LOAD_H + EXTRACT_H_PER_PAIR / 2,
                          "argv": [PY_GEN, "scripts/build_refusal_directions.py", "--pair", p, "--side", s,
                                   "--stage", S2, "--estimate-hours", "0.06"]})
        steps.append({"stage": S2, "kind": "reference_selection", "env": "gen", "est_h": GEN_LOAD_H + SELECT_H_PER_PAIR,
                      "argv": [PY_GEN, "scripts/select_refusal_reference.py", "--pair", p, "--stage", S2,
                               "--estimate-hours", f"{SELECT_H_PER_PAIR:.3f}"]})
    cal = []
    for p in "AB":
        for d, short in (("installation", "install"), ("reversal", "reverse")):
            for m in (0.5, 1.0, 1.5):
                cal.append((p, f"cal_refusal_add_{short}_m{m}"))
                steps.append(gen(S2, p, f"cal_refusal_add_{short}_m{m}", "development", 60))
    steps.append(judge(S2, "original", "start"))
    for p, c in cal:
        steps.append(score(S2, "original", p, c, "development", 60, ["harmfulness", "quality", "refusal"], ORIG_CALLS_PER_H))
    steps.append(judge(S2, "original", "stop"))
    for p in "AB":
        for d in ("installation", "reversal"):
            steps.append(cpu(S2, ["scripts/calibrate_refusal.py", "select", "--pair", p, "--direction", d]))
    # verification: one selected candidate + its baseline per cell, on 200 calibration prompts
    for p in "AB":
        for d, s in (("installation", "it"), ("reversal", "dpo")):
            steps.append({**gen(S2, p, base[s], "calibration", 200)})
            steps.append({**gen(S2, p, f"@selected:{p}:{d}", "calibration", 200), "resolve": "selected_candidate"})
    steps.append(judge(S2, "original", "start"))
    for p in "AB":
        for d, s in (("installation", "it"), ("reversal", "dpo")):
            steps.append(score(S2, "original", p, base[s], "calibration", 200, ["harmfulness", "quality"], ORIG_CALLS_PER_H))
            steps.append({**score(S2, "original", p, f"@selected:{p}:{d}", "calibration", 200, ["harmfulness", "quality"], ORIG_CALLS_PER_H),
                          "resolve": "selected_candidate"})
    steps.append(judge(S2, "original", "stop"))
    for p in "AB":
        for d in ("installation", "reversal"):
            steps.append(cpu(S2, ["scripts/calibrate_refusal.py", "decide", "--pair", p, "--direction", d]))

    # ---------------- S3: harmful panel, new arms only; original judge on those only
    new_arms = ["refusal_add_install", "refusal_add_reverse", "refusal_ablate_reference"]
    tuned = ["refusal_add_tuned_install", "refusal_add_tuned_reverse"]
    for p in "AB":
        for c in new_arms + tuned:
            steps.append({**gen(S3, p, c, hpanel, H), **({"optional": True} if c in tuned + ["refusal_ablate_reference"] else {})})
    steps.append(judge(S3, "original", "start"))
    for p in "AB":
        for c in new_arms + tuned:
            steps.append({**score(S3, "original", p, c, hpanel, H, ["harmfulness", "refusal", "quality"], ORIG_CALLS_PER_H),
                          **({"optional": True} if c in tuned + ["refusal_ablate_reference"] else {})})
    steps.append(judge(S3, "original", "stop"))

    # ---------------- S4: benign panels, all six mandatory conditions (+ optional rows)
    six = ["it_base", "dpo_base", "dsteer_install", "dsteer_reverse", "refusal_add_install", "refusal_add_reverse"]
    for p in "AB":
        for c in six + ["refusal_ablate_reference"]:
            for panel, n in (("xstest_safe", 250), (ipanel, IFE)):
                steps.append({**gen(S4, p, c, panel, n), **({"optional": True} if c == "refusal_ablate_reference" else {})})
    steps.append(judge(S4, "original", "start"))
    for p in "AB":
        for c in six + ["refusal_ablate_reference"]:
            steps.append({**score(S4, "xstest", p, c, "xstest_safe", 250, ["xstest"], ORIG_CALLS_PER_H * 4),
                          **({"optional": True} if c == "refusal_ablate_reference" else {})})
    steps.append(judge(S4, "original", "stop"))
    for p in "AB":
        for c in six + ["refusal_ablate_reference"]:
            steps.append(cpu(S4, ["scripts/score_outputs.py", "--evaluator", "ifeval", "--pair", p, "--condition", c,
                                  "--panel", ipanel, "--stage", S4, "--estimate-hours", "0"], est=0.0))

    # ---------------- S5: independent judge on the RECOVERED originals (rescoring design)
    dl_s5 = DOWNLOAD_GB["judge_independent"] * 1024 / 50 / 3600
    steps.append({**judge(S5, "independent", "start"), "est_h": JUDGE_START_H + dl_s5})
    for p in "AB":   # independent-judge profile on development responses only; gates S5 admission
        steps.append({**score(S5, "independent", p, "it_base", "development", 60, ["harmfulness"], IND_CALLS_PER_H),
                      "kind": "profile:independent"})
    steps.append(cpu(S5, ["scripts/profile_runtime.py", "--judge", "independent"], env="gen"))
    sub = ["--subset", "harmful_test_fallback100.jsonl"] if reduced else []
    for p in "AB":
        for c in ("it_base", "dpo_base", "dsteer_install", "dsteer_reverse"):
            steps.append(score(S5, "independent", p, c, "harmful_test", H, ["harmfulness"], IND_CALLS_PER_H, sub))
    for p in "AB":
        for c in ("it_base", "dpo_base", "dsteer_install", "dsteer_reverse"):
            steps.append({**score(S5, "independent", p, c, "harmful_test", H, ["refusal", "quality"], IND_CALLS_PER_H, sub),
                          "optional": True, "note": "only within the S5 cap, after harmfulness"})
    steps.append(judge(S5, "independent", "stop"))

    # ---------------- S7: analysis, report, export, shutdown latency
    steps.append(cpu(S7, ["scripts/analyse.py"], est=0.05))
    steps.append(cpu(S7, ["scripts/build_report.py"], est=0.02))
    steps.append({"stage": S7, "kind": "sync_back", "env": "mac", "est_h": SYNC_H, "argv": ["(Mac) rsync outputs back"]})
    return steps


def summarise(steps: list[dict], scope: str) -> dict:
    by_stage, mand, opt = {}, 0.0, 0.0
    for s in steps:
        by_stage.setdefault(s["stage"], {"mandatory_h": 0.0, "optional_h": 0.0})
        k = "optional_h" if s.get("optional") else "mandatory_h"
        by_stage[s["stage"]][k] += s["est_h"]
        if s.get("optional"):
            opt += s["est_h"]
        else:
            mand += s["est_h"]
    for st, v in by_stage.items():
        v["mandatory_with_retry_headroom_h"] = round(v["mandatory_h"] * RETRY * HEADROOM, 3)
        v["all_with_retry_headroom_h"] = round((v["mandatory_h"] + v["optional_h"]) * RETRY * HEADROOM, 3)
        v["cap_h"] = CAPS[st]
        v["mandatory_fits_cap"] = v["mandatory_with_retry_headroom_h"] <= CAPS[st]
        v["mandatory_h"], v["optional_h"] = round(v["mandatory_h"], 3), round(v["optional_h"], 3)
    gen_n = sum(s.get("responses", 0) for s in steps if not s.get("optional"))
    calls = sum(s.get("metric_calls", 0) for s in steps if not s.get("optional"))
    return {"scope": scope, "steps": len(steps), "new_responses_mandatory": gen_n,
            "metric_calls_mandatory": calls,
            "judge_starts": sum(s["kind"] == "judge_start" for s in steps),
            "generator_invocations": sum(s["kind"] == "generation" for s in steps),
            "mandatory_raw_h": round(mand, 3), "optional_raw_h": round(opt, 3),
            "mandatory_with_retry_headroom_h": round(mand * RETRY * HEADROOM, 2),
            "all_with_retry_headroom_h": round((mand + opt) * RETRY * HEADROOM, 2),
            "ceiling_h": CEILING, "mandatory_fits": mand * RETRY * HEADROOM <= CEILING,
            "all_fits": (mand + opt) * RETRY * HEADROOM <= CEILING, "by_stage": by_stage}


def apply_profile(path: str) -> list[str]:
    """Replace provisional rates with measured ones; return which were replaced."""
    global GEN_PER_H, ORIG_CALLS_PER_H, IND_CALLS_PER_H, JUDGE_START_H
    prof = json.loads(Path(path).read_text())
    done = []
    if prof.get("generation_responses_per_h"):
        GEN_PER_H = float(prof["generation_responses_per_h"]); done.append("generation")
    if prof.get("original_judge_calls_per_h"):
        ORIG_CALLS_PER_H = float(prof["original_judge_calls_per_h"]); done.append("original_judge")
    if prof.get("independent_judge_calls_per_h"):
        IND_CALLS_PER_H = float(prof["independent_judge_calls_per_h"]); done.append("independent_judge")
    ready = [v for v in (prof.get("judge_ready_seconds") or {}).values() if v]
    if ready:
        JUDGE_START_H = max(ready) / 3600; done.append("judge_startup")
    return done


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None)
    measured = apply_profile(ap.parse_args().profile) if ap.parse_args().profile else []
    out = {"basis": __doc__.split("Rates are PROVISIONAL", 1)[1].split("Run:")[0].strip(),
           "status": ("MEASURED: " + ", ".join(measured)) if measured else
                     "PROVISIONAL -- replace rates with reports/PROFILE.json measurements before admitting S2",
           "measured_rates_applied": measured,
           "constants": {"generation_responses_per_h": GEN_PER_H, "original_judge_calls_per_h": ORIG_CALLS_PER_H,
                         "independent_judge_calls_per_h": IND_CALLS_PER_H, "download_gb": DOWNLOAD_GB,
                         "download_h": round(DL_H, 3), "retry_allowance": RETRY, "headroom": HEADROOM},
           "corrections_from_v1": [
               "D-STEER direction reconstruction removed: vectors built on CPU from the verified recovered tensors.",
               "Original-judge scoring of recovered arms removed: they already carry original scores (scorer refuses).",
               "Every metric call counted (3 per new harmful arm, 3 per dev arm, 2 per calibration check, 1 XSTest call).",
               "Model loads, judge start-ups, downloads, environment setup and sync-back added explicitly.",
               "Generation rate described as 28 inter-start intervals, not a verified total for 29 runs.",
               "Independent judge rate marked unmeasured; profiled in S1 before S5 is admitted.",
               "Extraction now uses measured task025 GPU time (~100 s per pair) instead of 0.5 h per endpoint."],
           "scopes": {}}
    for scope in ("full", "reduced"):
        steps = build(scope)
        (ROOT / "configs" / f"stage_plan_{scope}.json").write_text(json.dumps(steps, indent=2) + "\n")
        out["scopes"][scope] = summarise(steps, scope)
    (ROOT / "reports" / "BUDGET_ESTIMATE_V2.json").write_text(json.dumps(out, indent=2) + "\n")
    for scope, s in out["scopes"].items():
        print(f"{scope:8s} mandatory {s['mandatory_with_retry_headroom_h']:5.2f} h | with optional "
              f"{s['all_with_retry_headroom_h']:5.2f} h (ceiling 30) | {s['new_responses_mandatory']} new responses, "
              f"{s['metric_calls_mandatory']} metric calls, {s['judge_starts']} judge starts")
        for st, v in s["by_stage"].items():
            print(f"    {st:40s} {v['mandatory_with_retry_headroom_h']:5.2f} h mand / "
                  f"{v['all_with_retry_headroom_h']:5.2f} h all  (cap {v['cap_h']})"
                  f"{'' if v['mandatory_fits_cap'] else '  ** OVER CAP **'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
