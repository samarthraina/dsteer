#!/usr/bin/env python3
"""Freeze every remaining panel before any new outcome exists. CPU only.

Selection is by salted SHA-256 of a stable key, never by content or by any model output.
Overlap is checked across ALL panels at three strengths: exact text, the frozen
normalisation (NFC, newline folding, strip -- splits.prompt_hash), and an aggressive
normalisation (casefold, collapse whitespace, strip punctuation). Any hit at any strength
is excluded from the later-priority panel and disclosed.

Run:  .venv/bin/python -I scripts/build_panels_v2.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor" / "primary_snapshot"))
from steering.splits import prompt_hash  # noqa: E402

SALT = "dsteer-rebuttal-20261028"
PUB = ROOT / "inputs" / "public"
PANELS = ROOT / "manifests" / "panels"
N_CONSTRUCT = 256


def salted(key: str) -> str:
    return hashlib.sha256(f"{SALT}:{key}".encode()).hexdigest()


def aggressive(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).casefold()
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def write(name: str, rows: list[dict]) -> dict:
    p = PANELS / f"{name}.jsonl"
    with p.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    digest = hashlib.sha256("\n".join(f"{r['source_id']}:{r['prompt_hash']}" for r in rows).encode()).hexdigest()
    return {"n": len(rows), "jsonl": str(p.relative_to(ROOT)),
            "jsonl_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "ordered_identity_digest": digest}


def main() -> int:
    man = json.loads((ROOT.parent.parent / "figures-final/data/task027-final/manifests/harmfulqa_v1.json").read_text())
    raw = json.loads((PUB / "harmfulqa" / "data_for_hub.json").read_text())
    hq_text = {f"harmfulqa-{i}": r["question"] for i, r in enumerate(raw)}
    hq_topic = {f"harmfulqa-{i}": r.get("topic") for i, r in enumerate(raw)}

    def hq_rows(partition: str) -> list[dict]:
        out = []
        for r in sorted((r for r in man["records"] if r["partition"] == partition),
                        key=lambda r: r["permuted_position"]):
            t = hq_text[r["source_id"]]
            if prompt_hash(t) != r["prompt_hash"]:
                raise SystemExit(f"HARD STOP: {r['source_id']} text does not match frozen hash")
            out.append({"source_id": r["source_id"], "prompt": t, "prompt_hash": r["prompt_hash"],
                        "dataset": "declare-lab/HarmfulQA@6f1a78aed47d16c0695e4595d0159abc38197bfd",
                        "partition": partition, "topic": hq_topic[r["source_id"]]})
        return out

    harmful_test = hq_rows("final_evaluation")
    development = hq_rows("development")
    calibration = hq_rows("calibration")
    construction_all = hq_rows("construction")
    construction_harmful = sorted(construction_all, key=lambda r: salted(r["source_id"]))[:N_CONSTRUCT]

    # ---- harmless, from the reference repository's own Alpaca-derived splits
    def ref_split(split: str) -> list[dict]:
        d = json.loads((PUB / "refusal_direction_reference" / f"dataset__splits__harmless_{split}.json").read_text())
        return [{"source_id": f"rdref-harmless-{split}-{i}", "prompt": r["instruction"],
                 "prompt_hash": prompt_hash(r["instruction"]), "category": r.get("category"),
                 "dataset": "andyrdt/refusal_direction@0b9c8bc56a73b42f595fd173156d4aefe9a9143c dataset/splits/harmless_" + split}
                for i, r in enumerate(d)]

    # ---- XSTest safe rows only; unsafe contrast rows never enter the denominator
    xs = list(csv.DictReader((PUB / "xstest" / "xstest_prompts.csv").open(encoding="utf-8")))
    xstest_safe = [{"source_id": f"xstest-{r['id']}", "prompt": r["prompt"], "prompt_hash": prompt_hash(r["prompt"]),
                    "type": r["type"], "label": r["label"], "focus": r["focus"],
                    "dataset": "paul-rottger/xstest@475f10bf0a3d6a9dfb174b6de1a38afbfdff98a5"}
                   for r in xs if r["label"] == "safe"]
    xstest_unsafe_excluded = sum(r["label"] == "unsafe" for r in xs)

    # ---- IFEval
    ife = [json.loads(l) for l in (PUB / "ifeval" / "ifeval_input_data.jsonl").open()]
    ifeval_all = [{"source_id": f"ifeval-{r['key']}", "prompt": r["prompt"], "prompt_hash": prompt_hash(r["prompt"]),
                   "instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"], "key": r["key"],
                   "dataset": "google/IFEval@966cd89545d6b6acfd7638bc708b98261ca58e84"} for r in ife]

    # ---- overlap screening: higher-priority panels are fixed; lower-priority ones are filtered
    fixed = {"harmful_test": harmful_test, "development": development, "calibration": calibration,
             "construction_harmful": construction_harmful}
    keys = lambda t: {("exact", t), ("frozen_norm", prompt_hash(t)), ("aggressive", aggressive(t))}
    taken: dict[tuple, str] = {}
    for name, rows in fixed.items():
        for r in rows:
            for k in keys(r["prompt"]):
                taken.setdefault(k, name)

    exclusions: list[dict] = []

    def admit(name: str, pool: list[dict], n: int | None) -> list[dict]:
        out = []
        for r in sorted(pool, key=lambda r: salted(r["source_id"])) if n else pool:
            hit = next((taken[k] for k in keys(r["prompt"]) if k in taken), None)
            if hit:
                exclusions.append({"panel": name, "source_id": r["source_id"], "collides_with": hit})
                continue
            out.append(r)
            for k in keys(r["prompt"]):
                taken.setdefault(k, name)
            if n and len(out) == n:
                break
        return out

    # Benign test panels are fixed before construction/selection pools draw from the harmless corpus,
    # so a harmless construction prompt can never coincide with a benign test prompt.
    xstest_safe = admit("xstest_safe", xstest_safe, None)
    instruction_test = admit("instruction_test", ifeval_all, 100)
    construction_harmless = admit("construction_harmless", ref_split("train"), N_CONSTRUCT)
    selection_harmless = admit("selection_harmless", ref_split("val"), len(development))
    instruction_fallback50 = sorted(instruction_test, key=lambda r: salted("nested50:" + r["source_id"]))[:50]

    panels = {
        "harmful_test": write("harmful_test_text", harmful_test),
        "development": write("development_text", development),
        "calibration": write("calibration_text", calibration),
        "construction_harmful": write("construction_harmful_256", construction_harmful),
        "construction_harmless": write("construction_harmless_256", construction_harmless),
        "selection_harmless": write("selection_harmless_60", selection_harmless),
        "xstest_safe": write("xstest_safe_250", xstest_safe),
        "instruction_test": write("instruction_test_100", instruction_test),
        "instruction_fallback50": write("instruction_fallback50", instruction_fallback50),
    }
    assert {r["source_id"] for r in instruction_fallback50} <= {r["source_id"] for r in instruction_test}
    assert panels["harmful_test"]["ordered_identity_digest"] == \
        "7851299f7e15743e47c0e4604697a0770b8e9607a9dd4420a0f31a3529a53953"

    # ---- confounds of the harmful-vs-harmless contrast (documented, not corrected)
    def stats(rows):
        w = [len(r["prompt"].split()) for r in rows]
        c = [len(r["prompt"]) for r in rows]
        return {"words_median": statistics.median(w), "words_mean": round(statistics.mean(w), 2),
                "chars_median": statistics.median(c), "question_mark_share": round(sum(r["prompt"].rstrip().endswith("?") for r in rows) / len(rows), 3)}
    from collections import Counter
    confounds = {
        "harmful": {"source": "HarmfulQA construction partition", **stats(construction_harmful),
                    "topics": dict(Counter(r["topic"] for r in construction_harmful).most_common())},
        "harmless": {"source": "Alpaca-derived harmless_train (reference repository)", **stats(construction_harmless),
                     "categories": dict(Counter(r["category"] for r in construction_harmless).most_common(8))},
        "interpretation": (
            "The contrast differs in harmful intent AND in corpus, topic distribution, phrasing (HarmfulQA is "
            "almost entirely questions; Alpaca mostly imperatives) and length. The resulting direction is a "
            "harmful-vs-general prompt contrast. It is a refusal-associated direction in the sense used by the "
            "reference method, NOT an isolated refusal mechanism."),
    }

    summary = {"salt": SALT, "panels": panels, "exclusions": exclusions,
               "exclusion_counts": dict(Counter(e["panel"] for e in exclusions)),
               "xstest_unsafe_rows_excluded_from_denominator": xstest_unsafe_excluded,
               "overlap_strengths": ["exact", "frozen_norm (splits.prompt_hash)", "aggressive (NFKC, casefold, strip punctuation, collapse whitespace)"],
               "confounds": confounds}
    (ROOT / "manifests" / "panels_v2.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    for k, v in panels.items():
        print(f"  {k:24s} n={v['n']:4d}  {v['jsonl']}")
    print("exclusions:", summary["exclusion_counts"] or "none")
    print("xstest unsafe rows excluded from denominator:", xstest_unsafe_excluded)
    h, s = confounds["harmful"], confounds["harmless"]
    print(f"confound: harmful median {h['words_median']} words, {100*h['question_mark_share']:.0f}% questions | "
          f"harmless median {s['words_median']} words, {100*s['question_mark_share']:.0f}% questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
