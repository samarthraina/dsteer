#!/usr/bin/env python3
"""Generation parity: does the new stack reproduce the recovered Task 027 responses?

Compares outputs/parity/harmful_test/{pair}/{condition}.jsonl, generated on the box for a
small prefix of the panel, against the recovered originals for the same source ids.
Byte-identical text is the target. The report gives exact-match rate, the first differing
character offset, and whether stop reasons agree -- so a library-version drift is measured,
not assumed away. A zero-intervention arm (it_base, dpo_base) failing parity is a hard stop
for that pair; a steered arm failing is reported with its divergence profile.

Run:  python scripts/parity_check.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from protocol import RECOVERED_HARMFUL  # noqa: E402
from records import read_jsonl  # noqa: E402


def first_diff(a: str, b: str) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> int:
    rows = []
    for f in sorted(glob.glob(str(ROOT / "outputs/parity/harmful_test/*/*.jsonl"))):
        p = Path(f)
        pair, cond = p.parent.name, p.stem
        if (pair, cond) not in RECOVERED_HARMFUL:
            continue
        orig = {r["source_id"]: r for r in read_jsonl(Path(glob.glob(str(
            ROOT / "inputs/recovered/task027/steer" / RECOVERED_HARMFUL[(pair, cond)] / "*.jsonl"))[0]))}
        new = list(read_jsonl(p))
        exact = sum(r["response"] == orig[r["source_id"]]["response"] for r in new)
        stop = sum(r["finish_reason"] == orig[r["source_id"]]["stop_reason"] for r in new)
        diffs = [first_diff(r["response"], orig[r["source_id"]]["response"]) for r in new]
        diffs = [d for d in diffs if d is not None]
        zero = cond in ("it_base", "dpo_base")
        rows.append({"pair": pair, "condition": cond, "zero_intervention": zero, "n": len(new),
                     "exact_text_matches": exact, "stop_reason_matches": stop,
                     "median_first_diff_char": sorted(diffs)[len(diffs) // 2] if diffs else None,
                     "verdict": "PASS" if exact == len(new) else ("HARD_STOP" if zero else "DRIFT_REPORTED")})
    out = ROOT / "reports/PARITY.json"
    out.write_text(json.dumps({"rows": rows, "target": "byte-identical responses on a panel prefix"}, indent=2) + "\n")
    for r in rows:
        print(f"{r['pair']} {r['condition']:16s} exact {r['exact_text_matches']}/{r['n']} "
              f"stop {r['stop_reason_matches']}/{r['n']} -> {r['verdict']}")
    return 0 if rows and not any(r["verdict"] == "HARD_STOP" for r in rows) else 6


if __name__ == "__main__":
    raise SystemExit(main())
