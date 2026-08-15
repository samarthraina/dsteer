"""Build (or re-verify) the frozen HarmfulQA partition manifest (protocol Section 5).

Loads the 1,960 raw rows at the pinned revision, assigns stable source IDs before any
shuffling, and hands them to `steering.splits.build_manifest`, which excludes the 22
documented verbatim-duplicate rows (retaining the lower source index from each pair),
partitions the 1,938 retained prompts, and records the full exclusion lineage. An
existing manifest is never overwritten with different content -- rerunning against an
unchanged source reports success without rewriting; any difference fails loudly instead
of silently updating.

Needs network access to pull HarmfulQA at the pinned revision; does no model or GPU work.

    python scripts/build_harmfulqa_manifest.py [--out manifests/harmfulqa_v1.json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset

from steering.splits import SplitError, build_manifest, prompt_hash, save_manifest

REPO_ID = "declare-lab/HarmfulQA"
CONFIG = "default"
SPLIT = "train"
# Pinned per protocol Section 5 -- source validation at this revision is what
# established the 1,960-row / 22-duplicate-pair counts this script enforces below.
REVISION = "6f1a78aed47d16c0695e4595d0159abc38197bfd"
SEED = 20260815


def load_raw_records(revision: str):
    """Stable source IDs from the row's position in the unshuffled dataset at
    `revision` -- never from a post-shuffle enumeration."""
    ds = load_dataset(REPO_ID, name=CONFIG, split=SPLIT, revision=revision)
    records = []
    for i, row in enumerate(ds):
        prompt = row["question"]
        records.append({
            "source_id": f"harmfulqa-{i}",
            "source_index": i,
            "prompt": prompt,
            "prompt_hash": prompt_hash(prompt),
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="manifests/harmfulqa_v1.json")
    args = parser.parse_args()

    records = load_raw_records(REVISION)
    print(f"loaded {len(records)} raw records from {REPO_ID}@{REVISION} (config={CONFIG})")

    try:
        payload = build_manifest(
            records,
            seed=SEED,
            dataset_repo=REPO_ID,
            dataset_split=SPLIT,
            dataset_revision=REVISION,
            dataset_config=CONFIG,
        )
    except SplitError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    try:
        written = save_manifest(payload, out_path)
    except SplitError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"manifest hash: {payload['manifest_hash']}")
    print(f"raw/retained/excluded: {payload['raw_record_count']}/{payload['retained_record_count']}/{payload['excluded_record_count']}")
    print(f"partition counts: {payload['partition_counts']}")
    print(f"{'wrote' if written else 'unchanged, already present at'}: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
