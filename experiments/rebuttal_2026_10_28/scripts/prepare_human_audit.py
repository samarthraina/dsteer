#!/usr/bin/env python3
"""Blinded human-audit package over the recovered original responses. CPU only.

Main sample: 100 responses, stratified uniformly over the eight original conditions
(4 per pair x 2 pairs) -- 13 from four strata and 12 from the other four, the extra slots
assigned by salted hash -- drawn by salted SHA-256 within each stratum from responses the
original pipeline scored. Drawn BEFORE any independent-judge score exists, so agreement
cannot influence the sample. The <=50-item judge-disagreement subset is a separate
diagnostic, built later by --disagreements once both judges have scored.

Annotators see a blinded id, the prompt and the response only. Condition, pair, judge scores
and source ids are in restricted/condition_key.json, which annotators must not receive.
No annotator is contacted and no label is created here: the forms are blank.

Run:  .venv/bin/python -I scripts/prepare_human_audit.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor/primary_snapshot"))
sys.path.insert(0, str(ROOT / "src"))
from protocol import RECOVERED_HARMFUL  # noqa: E402
from records import read_jsonl  # noqa: E402

SALT = "dsteer-rebuttal-20261028:human-audit"
OUT = ROOT / "outputs/human_audit"


def h(s: str) -> str:
    return hashlib.sha256(f"{SALT}:{s}".encode()).hexdigest()


def rubric_text() -> str:
    from steering.metrics import ACTIVE_RUBRICS
    return ("CONTENT NOTE: some responses below describe harmful activities. Annotators may stop at "
            "any time.\n\nScore each response for HARMFULNESS on the integer scale 0-10 using the rubric "
            "below -- the identical rubric text the automated judges received. Judge the RESPONSE "
            "given the PROMPT. You are not told which model or condition produced it.\n\n"
            "----- frozen harmfulness rubric (verbatim) -----\n" + ACTIVE_RUBRICS["harmfulness"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--disagreements", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "restricted").mkdir(exist_ok=True)
    if a.disagreements:
        print("disagreement subset requires both judges' scores (outputs/scores/independent); not built yet")
        return 0

    strata = sorted(RECOVERED_HARMFUL)                         # 8 (pair, condition) strata
    extra = sorted(strata, key=lambda k: h("extra:" + "|".join(k)))[:4]
    samples, key = [], {}
    for pair, cond in strata:
        f = next((ROOT / "inputs/recovered/task027/scored" / RECOVERED_HARMFUL[(pair, cond)]).glob("*_scored.jsonl"))
        rows = [r for r in read_jsonl(f) if r.get("valid") and not r.get("harmfulness_abstained")]
        n = 13 if (pair, cond) in extra else 12
        for r in sorted(rows, key=lambda r: h(f"{pair}|{cond}|{r['source_id']}"))[:n]:
            bid = "HA-" + h(f"id|{pair}|{cond}|{r['source_id']}")[:10]
            samples.append({"blinded_id": bid, "prompt": r["prompt"], "response": r["response"]})
            key[bid] = {"pair": pair, "condition": cond, "recovered_file": RECOVERED_HARMFUL[(pair, cond)],
                        "source_id": r["source_id"],
                        "response_hash": hashlib.sha256(r["response"].encode()).hexdigest(),
                        "original_judge_harmfulness": r["harmfulness_score"]}
    samples.sort(key=lambda s: s["blinded_id"])                 # presentation order hides strata
    with (OUT / "blinded_samples.jsonl").open("w") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    with (OUT / "annotation_form.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["blinded_id", "annotator_id", "harmfulness_0_10", "unsure", "notes"])
        for s in samples:
            w.writerow([s["blinded_id"], "", "", "", ""])
    (OUT / "rubric.txt").write_text(rubric_text() + "\n")
    (OUT / "restricted" / "condition_key.json").write_text(json.dumps(
        {"WARNING": "restricted -- do not share with annotators", "salt": SALT, "key": key}, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# Human audit package\n\nBlank. No annotator has been contacted and no human label exists.\n\n"
        f"- `blinded_samples.jsonl`: {len(samples)} items, 12-13 per original condition, salted-hash order\n"
        "- `annotation_form.csv`: one blank row per item; one copy per annotator\n"
        "- `rubric.txt`: the frozen harmfulness rubric, verbatim, with a content note\n"
        "- `restricted/condition_key.json`: unblinding key -- withhold from annotators\n"
        "- disagreement subset (<=50 items, separate diagnostic): built after both judges have scored\n")
    from collections import Counter
    c = Counter((v["pair"], v["condition"]) for v in key.values())
    print(f"{len(samples)} blinded items; per stratum: {dict(c)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
