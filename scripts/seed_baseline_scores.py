"""Reuse an already-scored baseline in another arm whose baseline is identical.

Every arm of a multi-arm sweep writes its own baseline.jsonl: the unsteered checkpoint,
same prompts, greedy decoding, hooks inactive. The text is therefore identical across
arms, and judging it again buys nothing. At roughly 35 minutes of judge time per baseline
this is the largest avoidable cost in a run with several arms -- on six arms it is most of
two hours.

Identity is checked, not assumed. If any response differs the copy is refused: a baseline
that is not byte-identical is a different measurement, and reusing its scores would
compare an arm against a baseline it never produced. That is the same failure as a run
resuming on another run's generations, arrived at from the other end.

score_sweep resumes by record id, so once the scored file is in place the baseline costs
zero judge calls and still contributes its row to summary.csv.

    python scripts/seed_baseline_scores.py \
        --from outputs/steer/llama3-oh/harmfulqa/dpo_refusal \
        --to   outputs/steer/llama3-oh/harmfulqa/dpo_refusal_ablate
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steering.utils import read_jsonl

BASELINE = "baseline.jsonl"
SCORED = "scored/baseline_scored.jsonl"


def responses(path: Path) -> dict:
    return {r["id"]: r.get("response", "") for r in read_jsonl(path)}


def main() -> int:
    p = argparse.ArgumentParser(description="Copy a scored baseline between arms with identical generations.")
    p.add_argument("--from", dest="src", required=True, help="Arm directory holding the scored baseline.")
    p.add_argument("--to", dest="dst", required=True, help="Arm directory to seed.")
    p.add_argument("--force", action="store_true", help="Overwrite a scored baseline already in --to.")
    args = p.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    src_scored, dst_scored = src / SCORED, dst / SCORED

    for path in (src / BASELINE, dst / BASELINE, src_scored):
        if not path.exists():
            print(f"FATAL: missing {path}")
            return 1

    if dst_scored.exists() and not args.force:
        print(f"FATAL: {dst_scored} exists; pass --force to replace it")
        return 1

    a, b = responses(src / BASELINE), responses(dst / BASELINE)
    if a.keys() != b.keys():
        print(f"FATAL: different prompt ids -- {len(a)} in --from, {len(b)} in --to, "
              f"{len(a.keys() & b.keys())} shared")
        return 1

    differing = [k for k in a if a[k] != b[k]]
    if differing:
        print(f"FATAL: {len(differing)}/{len(a)} responses differ; these are not the same "
              f"baseline. First: {differing[0]}")
        return 1

    scored_ids = {r["id"] for r in read_jsonl(src_scored)}
    missing = a.keys() - scored_ids
    if missing:
        print(f"FATAL: {len(missing)}/{len(a)} baseline records are unscored in --from")
        return 1

    dst_scored.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_scored, dst_scored)
    print(f"{len(a)}/{len(a)} responses identical; copied {len(scored_ids)} scored records")
    print(f"  {src_scored} -> {dst_scored}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
