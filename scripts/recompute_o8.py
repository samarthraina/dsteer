"""Recompute the resemblance-versus-transfer split under the corrected harmfulness rubric.

The original numbers were scored before the rubric was given the request, so their
harmfulness column is on a scoring that nothing else in the project still uses -- its gap
denominator and its numerator are both pre-correction, which makes the figure internally
consistent and externally incomparable. Refusal is unaffected, because that rubric never
changed, and refusal is the half the argument actually needs.

Re-judging was already done once, for every scored record, into `*_recheck.jsonl` sidecars
carrying `harmfulness_score_v2` beside the original `_v1`. So this needs no judge and no
GPU: it is the same generations, read through the corrected scoring.

    python scripts/recompute_o8.py --it-run <hub run id> --dpo-run <hub run id>
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO = "samarthraina/dsteer-results"


def load(path: str) -> list:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def pick_run(files, side: str) -> str:
    """The run directory with the most recheck files for this side."""
    counts = defaultdict(int)
    for f in files:
        m = re.search(rf"runs/steer_scored/llama3-oh_{side}/([^/]+)/.*_recheck\.jsonl$", f)
        if m:
            counts[m.group(1)] += 1
    if not counts:
        raise SystemExit(f"no recheck files for side {side!r}")
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def fetch(side: str, run: str, stem: str, kind: str) -> Optional[str]:
    p = f"runs/steer_scored/llama3-oh_{side}/{run}/{stem}_{kind}.jsonl"
    try:
        return hf_hub_download(REPO, p, repo_type="dataset")
    except Exception:
        return None


def series(side: str, run: str, stem: str) -> Dict[str, Dict[str, float]]:
    """{id: {refusal, harm_v1, harm_v2}} for one setting."""
    out: Dict[str, Dict[str, float]] = {}
    sc = fetch(side, run, stem, "scored")
    if sc is None:
        return out
    for r in load(sc):
        if r.get("refusal_score") is None:
            continue
        out[r["id"]] = {"refusal": r["refusal_score"], "harm_v1": r.get("harmfulness_score")}
    rc = fetch(side, run, stem, "recheck")
    if rc:
        for r in load(rc):
            if r["id"] in out and r.get("harmfulness_score_v2") is not None:
                out[r["id"]]["harm_v2"] = r["harmfulness_score_v2"]
    return out


def mean(vals) -> float:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def boot_delta(pairs, n: int = 4000, lo: float = 10.0, hi: float = 90.0):
    """Paired bootstrap of (steered - unsteered) on the same prompts."""
    if not pairs:
        return (float("nan"),) * 3
    rng = random.Random(0)
    k = len(pairs)
    ds = []
    for _ in range(n):
        s = [pairs[rng.randrange(k)] for _ in range(k)]
        ds.append(sum(b - a for a, b in s) / k)
    ds.sort()
    return sum(b - a for a, b in pairs) / k, ds[int(lo / 100 * n)], ds[int(hi / 100 * n)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Recompute the split under the corrected rubric.")
    ap.add_argument("--it-run", default=None)
    ap.add_argument("--dpo-run", default=None)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--lambdas", default="+0.200,+0.400,+0.600")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    it_run = args.it_run or pick_run(files, "it")
    dpo_run = args.dpo_run or pick_run(files, "dpo")
    print(f"IT run  {it_run}\nDPO run {dpo_run}\n")

    it0, dpo0 = series("it", it_run, "baseline"), series("dpo", dpo_run, "baseline")
    shared = it0.keys() & dpo0.keys()
    differ = {i for i in shared if abs(dpo0[i]["refusal"] - it0[i]["refusal"]) >= args.threshold}
    agree = shared - differ
    print(f"{len(shared)} scored on both; {len(differ)} differ, {len(agree)} agree\n")

    for key, label in (("harm_v1", "as published (pre-correction)"),
                       ("harm_v2", "corrected rubric")):
        print(f"--- harmfulness, {label} ---")
        for name, ids in (("checkpoints differ", differ), ("checkpoints agree", agree)):
            print(f"  {name:20s} n={len(ids):<4} "
                  f"IT {mean(it0[i].get(key) for i in ids):.3f}  "
                  f"DPO {mean(dpo0[i].get(key) for i in ids):.3f}  "
                  f"gap {mean(it0[i].get(key) for i in ids) - mean(dpo0[i].get(key) for i in ids):+.3f}")
        print()

    print("--- steered, conditioned on the prompts that differ ---")
    print(f"{'lambda':<9}{'d refusal':>26}{'d harm v1':>26}{'d harm v2':>26}")
    rows = {}
    for lam in args.lambdas.split(","):
        s = series("it", it_run, f"lambda_{lam}")
        ids = sorted(differ & s.keys())
        cells = []
        for metric, key in (("refusal", "refusal"), ("v1", "harm_v1"), ("v2", "harm_v2")):
            pairs = [(it0[i][key], s[i][key]) for i in ids
                     if it0[i].get(key) is not None and s[i].get(key) is not None]
            d, lo, hi = boot_delta(pairs)
            cells.append(f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        rows[lam] = cells
        print(f"{lam:<9}" + "".join(f"{c:>26}" for c in cells))

    if args.output:
        Path(args.output).write_text(json.dumps(
            {"it_run": it_run, "dpo_run": dpo_run, "n_differ": len(differ),
             "n_agree": len(agree), "rows": rows}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
