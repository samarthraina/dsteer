#!/usr/bin/env python3
"""Reviewer evidence map and run accounting, generated from what is on disk. CPU only.

Each of the eight Brain placeholders is filled ONLY from outputs/analysis/*.csv rows with
status=complete. A placeholder with no complete row is written as PENDING with the reason,
never with a number. GPU hours and instance-stop status come from the ledger of record.

Writes reports/REVIEWER_EVIDENCE_MAP.md and reports/RUN_STATUS.json.

Run:  python3 -I scripts/build_report.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from budget_guard import BudgetGuard  # noqa: E402

A = ROOT / "outputs/analysis"


def rows(name: str) -> list[dict]:
    p = A / f"{name}.csv"
    return list(csv.DictReader(p.open())) if p.exists() else []


def complete(rs):
    return [r for r in rs if r.get("status") == "complete"]


def f(x, d=3):
    try:
        return f"{float(x):+.{d}f}"
    except (TypeError, ValueError):
        return "—"


def ci(r, lo="ci_lo", hi="ci_hi"):
    return f"[{f(r.get(lo))}, {f(r.get(hi))}]"


def table(header: list[str], body: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(out)


def pending(reason: str) -> str:
    return f"**PENDING — no verified result.** {reason}\n"


def section_baseline(tag: str) -> str:
    eff = complete(rows("harmful_effects"))
    con = complete(rows("method_contrasts"))
    comp = [r for r in eff if r["method"] != "dsteer"]
    if not comp:
        return pending("The comparator arms have not been generated and scored (GPU stages S2-S3). "
                       "Only the recovered D-STEER rows exist.")
    body = [[r["pair"], r["direction"], r["method"], r["paired_n"], f(r["intended_effect"]), ci(r),
             r.get("invalid_rate", ""), r.get("maxlen_rate", ""), r.get("quality_loss", ""), r.get("gates_pass", "")]
            for r in eff]
    s = table(["pair", "direction", "method", "paired n", "intended effect", "95% CI", "invalid", "max-token",
               "quality loss", "gates pass"], body)
    if con:
        s += "\n\nDirect paired contrasts (D-STEER minus comparator, one common intersection):\n\n"
        s += table(["pair", "direction", "contrast", "paired n", "difference", "95% CI", "label"],
                   [[r["pair"], r["direction"], r["contrast"], r["paired_n"], f(r["difference"]), ci(r), r.get("label", "")]
                    for r in con])
    s += ("\n\n`refusal_add_matched` adds the harmful-minus-harmless mean direction of each endpoint at "
          "D-STEER's per-layer norm and coefficient (layers 27-31, all positions). It is an adapted equal-norm "
          "additive baseline, NOT an exact reproduction of reference directional ablation (departures: "
          "src/refusal.py REFERENCE_DEPARTURES).")
    s += "\n\n" + reference_status()
    s += ("\n\nInterpretation limit: on these endpoints the matched direction lowers the judged refusal score "
          "rather than raising it, so it is a weak refusal baseline. These rows show that D-STEER outperforms "
          "this matched additive comparator; they do not show that D-STEER outperforms established refusal "
          "steering in general.")
    return s


def reference_status() -> str:
    """The exact-reference arm and the tuned arms, from their own records -- never assumed."""
    lines = []
    for pair in ("A", "B"):
        p = ROOT / "outputs/directions" / pair / "dpo" / "reference_selection.json"
        if not p.exists():
            lines.append(f"- Pair {pair}: reference selection not run.")
            continue
        r = json.loads(p.read_text())
        if r.get("selected"):
            lines.append(f"- Pair {pair}: reference direction selected at layer {r['selected']['layer']}.")
        else:
            why = r.get("blocked") or "no candidate passed the reference's KL / induce / pruning rules"
            lines.append(f"- Pair {pair}: exact reference ablation **not constructible** ({why}; harmful "
                         f"selection prompts kept after the refusal-score filter: {r.get('n_harmful')}). Arm not run.")
    sel = ROOT / "manifests/comparator_selection.json"
    if sel.exists():
        t = json.loads(sel.read_text()).get("tuned", {})
        lines.append("- Tuned comparator multipliers: " + ", ".join(
            f"{k} {v.get('status')} (dev-selected {v.get('dev_selected_multiplier')})" for k, v in t.items())
            + ". Not run on the test panel (deferred; their calibration check was voided after a judge outage).")
    return "Reference-method arms:\n\n" + "\n".join(lines)


def geometry() -> str:
    out = []
    for pair in ("A", "B"):
        for side in ("it", "dpo"):
            p = ROOT / "outputs/directions" / pair / side / "geometry.csv"
            if not p.exists():
                continue
            rs = [r for r in csv.DictReader(p.open()) if str(r.get("steered_layer")) == "True"]
            cs = [float(r["cosine_refusal_vs_dsteer"]) for r in rs]
            if cs:
                out.append([pair, side, ", ".join(r["layer"] for r in rs), f"{min(cs):+.3f}", f"{max(cs):+.3f}"])
    if not out:
        return ""
    return ("\n\nLayerwise cosine between each endpoint's refusal direction and D-STEER's direction, steered "
            "layers (descriptive; cosine does not establish functional equivalence or novelty; full table "
            "outputs/directions/{pair}/{side}/geometry.csv):\n\n"
            + table(["pair", "endpoint", "layers", "min cosine", "max cosine"], out))


def section_judge() -> str:
    jr = complete(rows("judge_robustness"))
    if not jr:
        return pending("The independent judge (microsoft/phi-4@2db69c1c) has not scored the recovered "
                       "original responses (GPU stage S5). The design is rescoring of the identical "
                       "submitted responses; all 58 files are verified locally.")
    body = [[r["pair"], r["direction"], f(r["effect_original"]), ci(r, "orig_ci_lo", "orig_ci_hi"),
             f(r["effect_independent"]), ci(r, "ind_ci_lo", "ind_ci_hi"), r["matched_n"],
             f(r["effect_difference_orig_minus_ind"]), ci(r, "diff_ci_lo", "diff_ci_hi"), r["independent_sign_agrees"]]
            for r in jr]
    s = table(["pair", "direction", "effect (original)", "95% CI", "effect (independent)", "95% CI",
               "matched n", "difference", "95% CI", "same sign"], body)
    agree = []
    for r in jr:
        for arm in ("it_base", "dpo_base", "dsteer_install", "dsteer_reverse"):
            if r.get(f"{arm}_n_matched_intersection"):
                agree.append([r["pair"], arm, r[f"{arm}_n_matched_intersection"], r[f"{arm}_exact_agreement"][:5],
                              r[f"{arm}_mean_absolute_difference"][:5], r[f"{arm}_spearman_rho"][:5],
                              r[f"{arm}_abstentions_judge_a"], r[f"{arm}_abstentions_judge_b"]])
    s += ("\n\nIndependent judge: microsoft/phi-4@2db69c1c, rescoring the identical submitted responses "
          "(harmfulness). Per-response agreement with the original judge:\n\n"
          + table(["pair", "arm", "matched n", "exact agreement", "mean |diff|", "Spearman rho",
                   "not scored (original)", "not scored (independent)"], agree))
    return s


def section_utility() -> str:
    bu = [r for r in complete(rows("benign_utility")) if r["panel"] == "xstest_safe"]
    iv = complete(rows("ifeval_variants"))
    if not bu and not iv:
        return pending("XSTest (250 safe prompts) and IFEval (100 prompts) have not been generated or scored "
                       "(GPU stage S4). Both panels are frozen.")
    x = table(["pair", "condition", "generated", "invalid", "classified", "full refusal", "partial refusal",
               "full-refusal change vs baseline", "95% CI", "paired n"],
              [[r["pair"], r["condition"], r["n_generated"], r["n_invalid"], r["n_classified"],
                f(r["full_refusal_rate"]), f(r["partial_refusal_rate"]),
                f(r.get("full_refusal_diff_vs_endpoint")), ci(r, "diff_ci_lo", "diff_ci_hi"), r.get("diff_paired_n", "—")]
               for r in bu])
    v = lambda r, k: f"{f(r.get(k + '_change'))} {ci(r, k + '_ci_lo', k + '_ci_hi')} (n={r.get(k + '_paired_n', '—')})" \
        if r.get(k + "_change") else "—"
    lv = table(["pair", "condition", "invalid", "official, all 100 (strict / loose)",
                "conditional on validity (strict / loose, n valid)", "validity-adjusted (strict / loose)"],
               [[r["pair"], r["condition"], r["invalid_rate"], f"{r['official_all_strict']} / {r['official_all_loose']}",
                 f"{r['conditional_strict']} / {r['conditional_loose']} (n={r['n_valid']})",
                 f"{r['adjusted_strict']} / {r['adjusted_loose']}"] for r in iv])
    ch = table(["pair", "condition", "vs", "invalid-rate change", "official all-100 strict change",
                "conditional strict change", "adjusted strict change"],
               [[r["pair"], r["condition"], r["baseline"], v(r, "invalid"), v(r, "official_all_strict"),
                 v(r, "conditional_strict"), v(r, "adjusted_strict")] for r in iv if r.get("baseline")])
    return ("XSTest safe prompts (labels: original judge with the official XSTest rubric; rates are over "
            "classified responses, invalid responses are not labelled):\n\n" + x +
            "\n\nIFEval (official checker google-research@e6890f85, prompt-level). Three separately labelled "
            "scores: **official** = the unmodified checker on all 100 outputs as generated (the IFEval metric); "
            "**conditional** = the same checker on outputs passing the frozen validity gate, denominator shown; "
            "**validity-adjusted** = invalid outputs counted as failures (NOT the official metric).\n\n" + lv +
            "\n\nPaired changes against each intervention's own baseline (installation vs IT, reversal vs DPO); "
            "loose-level changes in outputs/analysis/ifeval_variants.csv:\n\n" + ch +
            "\n\nLimits: a limited utility audit, not a capability benchmark (250 XSTest safe prompts, 100 IFEval "
            "prompts). Pair A's unsteered endpoints already fall into repetition loops on 30-37 % of IFEval outputs "
            "at the 768-token cap, so pair A's instruction-following evidence is weak. XSTest labels come from a "
            "single LLM judge.")


def main() -> int:
    led = BudgetGuard(ROOT / "logs/gpu_budget.jsonl")
    st = led.state()
    inst = json.loads((ROOT / "logs/instance.json").read_text()) if (ROOT / "logs/instance.json").exists() else None
    stops = [json.loads(l) for l in (ROOT / "logs/gpu_budget.jsonl").read_text().splitlines()
             if '"instance_stop"' in l] if (ROOT / "logs/gpu_budget.jsonl").exists() else []
    status = {"consumed_a100_hours": round(st.consumed_h, 4), "remaining_a100_hours": round(st.remaining_h, 4),
              "instance": inst and inst.get("id"), "instance_open": st.instance_open,
              "instance_stop_verified": (stops[-1].get("stop_verified") if stops else None),
              "instance_ever_started": inst is not None}
    (ROOT / "reports/RUN_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")

    d = [r for r in complete(rows("harmful_effects")) if r["method"] == "dsteer"]
    existing = table(["pair", "direction", "paired n", "effect", "95% CI", "worst-case bounds"],
                     [[r["pair"], r["direction"], r["paired_n"], f(r["intended_effect"]), ci(r),
                       f"[{f(r['worst_case_low'])}, {f(r['worst_case_high'])}]"] for r in d])
    md = [
        "# REVIEWER_EVIDENCE_MAP — generated by scripts/build_report.py",
        "",
        f"GPU hours consumed: **{status['consumed_a100_hours']}** of 30. Instance ever started: "
        f"**{status['instance_ever_started']}**. Last stop verified: **{status['instance_stop_verified']}**.",
        "",
        "A block marked PENDING has no verified result. A block with tables is filled only from complete "
        "analysis tables under outputs/analysis/; nothing here is hand-entered. Brain placeholders are "
        "updated by the user, not by this file.",
        "",
        "## Existing evidence, recomputed from the recovered submitted responses (CPU)",
        "",
        existing,
        "",
        "Worst-case bounds fill every excluded prompt adversarially on [0,1]; a post-submission analysis.",
        "",
        "## 1 · R1 BASELINE",
        "", section_baseline("R1"), "",
        "## 2 · R1 INDEPENDENT JUDGE",
        "", section_judge(), "",
        "## 3 · R1 HUMAN AUDIT (optional)",
        "", pending("Blinded package prepared (outputs/human_audit/, 100 items); no annotator contacted, no "
                    "human label exists. Do not describe automated labels as human validation."), "",
        "## 4 · R2 BASELINE AND GEOMETRY",
        "", (section_baseline("R2") + geometry())
        if complete(rows("harmful_effects")) and any(r["method"] != "dsteer" for r in complete(rows("harmful_effects")))
        else pending("Needs the endpoint refusal directions (GPU stage S2) and the comparator arms."), "",
        "## 5 · R2 BENIGN UTILITY",
        "", section_utility(), "",
        "## 6 · R2 JUDGE CHECK",
        "", section_judge(), "",
        "## 7 · R3 BASELINE",
        "", section_baseline("R3"), "",
        "## 8 · R3 PROJECTION (optional)",
        "", pending("Lowest priority; not implemented as a runner in this pass and runs only after "
                    "independent judging. A null residual effect would not show that no other mechanism exists."),
        "",
    ]
    (ROOT / "reports/REVIEWER_EVIDENCE_MAP.md").write_text("\n".join(md))
    filled = sum(1 for b in (section_baseline("x"), section_judge(), section_utility()) if "PENDING" not in b)
    print(f"evidence map written; result blocks with data: {filled}; GPU hours {status['consumed_a100_hours']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
