#!/usr/bin/env python3
"""IFEval reported three ways, never conflated. CPU only.

  official_all   the unmodified official checker on all 100 outputs as generated
                 (this is the IFEval metric; degenerate outputs are checked like any other)
  conditional    the same checker on outputs that pass the frozen validity gate; the
                 denominator is printed with every rate
  adjusted       validity-adjusted, NOT the official metric: an invalid output counts as a
                 failure, a valid one keeps its official verdict

Paired changes are each intervention minus its own baseline endpoint (installation vs IT,
reversal vs DPO), bootstrapped over prompts (10,000 reps, seed 20261028). The conditional
change uses only prompts valid in both arms. Invalid-rate changes are reported alongside.

Run:  .venv/bin/python -I scripts/ifeval_variants.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import ifeval_score  # noqa: E402
from paired_stats import rate_difference  # noqa: E402
from records import read_jsonl  # noqa: E402

PAIRS = ("A", "B")
CONDS = ("it_base", "dpo_base", "dsteer_install", "dsteer_reverse", "refusal_add_install", "refusal_add_reverse")
REF = {"dsteer_install": "it_base", "refusal_add_install": "it_base",
       "dsteer_reverse": "dpo_base", "refusal_add_reverse": "dpo_base"}


def per_prompt(pair: str, cond: str) -> dict | None:
    p = ROOT / "outputs/generations/instruction_test" / pair / f"{cond}.jsonl"
    if not p.exists():
        return None
    panel = {r["source_id"]: r for r in read_jsonl(ROOT / "manifests/panels/instruction_test_100.jsonl")}
    out = {}
    for g in read_jsonl(p):
        s = ifeval_score.score(panel[g["source_id"]], g["response"])
        out[g["source_id"]] = {"valid": bool(g["validity"]["ok"]), "strict": s["strict_prompt_level"],
                               "loose": s["loose_prompt_level"]}
    return out


def rate(xs: list[bool]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def main() -> int:
    data = {(p, c): per_prompt(p, c) for p in PAIRS for c in CONDS}
    rows = []
    for (pair, cond), d in data.items():
        if d is None:
            rows.append({"pair": pair, "condition": cond, "status": "not_run"})
            continue
        ids = sorted(d)
        valid = [i for i in ids if d[i]["valid"]]
        r = {"pair": pair, "condition": cond, "status": "complete", "n": len(ids), "n_valid": len(valid),
             "invalid_rate": rate([not d[i]["valid"] for i in ids])}
        for lvl in ("strict", "loose"):
            r[f"official_all_{lvl}"] = rate([d[i][lvl] for i in ids])
            r[f"conditional_{lvl}"] = rate([d[i][lvl] for i in valid])
            r[f"adjusted_{lvl}"] = rate([d[i][lvl] and d[i]["valid"] for i in ids])
        base = REF.get(cond)
        if base and data.get((pair, base)):
            b = data[(pair, base)]

            def diff(name, fa, fb, only_valid=False):
                a = {i: (fa(d[i]) if (not only_valid or d[i]["valid"]) else None) for i in d}
                bb = {i: (fb(b[i]) if (not only_valid or b[i]["valid"]) else None) for i in b}
                e = rate_difference(a, bb)
                r.update({f"{name}_change": round(e.point, 4), f"{name}_ci_lo": round(e.ci_lo, 4),
                          f"{name}_ci_hi": round(e.ci_hi, 4), f"{name}_paired_n": e.n_paired})

            r["baseline"] = base
            diff("invalid", lambda x: not x["valid"], lambda x: not x["valid"])
            for lvl in ("strict", "loose"):
                diff(f"official_all_{lvl}", lambda x, l=lvl: x[l], lambda x, l=lvl: x[l])
                diff(f"conditional_{lvl}", lambda x, l=lvl: x[l], lambda x, l=lvl: x[l], only_valid=True)
                diff(f"adjusted_{lvl}", lambda x, l=lvl: x[l] and x["valid"], lambda x, l=lvl: x[l] and x["valid"])
        rows.append(r)
    out = ROOT / "outputs/analysis/ifeval_variants.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows); checker {ifeval_score.CHECKER_REF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
