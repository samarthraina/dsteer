"""Build (or re-verify) the frozen AdvBench OOD evaluation manifest (Task 018).

Downloads only the pinned source file (`walledai/AdvBench@<revision>`,
`data/train-00000-of-00001.parquet`), validates every raw row's prompt, deduplicates by
canonical prompt hash, selects exactly 200 records by the fixed permutation, and writes
`manifests/advbench_v1.json`. Refuses to overwrite an existing manifest with different
content (see `advbench.save_manifest`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import hf_hub_download

from steering.advbench import (
    AdvBenchError,
    REPOSITORY,
    REVISION,
    SOURCE_FILE_PATH,
    build_manifest,
    iter_raw_rows,
    save_manifest,
    source_file_identity,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="manifests/advbench_v1.json")
    args = parser.parse_args()

    print(f"downloading {REPOSITORY}@{REVISION} :: {SOURCE_FILE_PATH}")
    source_path = hf_hub_download(
        repo_id=REPOSITORY, filename=SOURCE_FILE_PATH, repo_type="dataset", revision=REVISION,
    )
    source_file = source_file_identity(source_path)
    print(f"source file: {source_file['size_bytes']} bytes, sha256={source_file['sha256']}")

    rows = iter_raw_rows(source_path)
    print(f"loaded {len(rows)} raw rows")

    try:
        payload = build_manifest(rows, source_file=source_file)
    except AdvBenchError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    try:
        written = save_manifest(payload, out_path)
    except AdvBenchError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"manifest hash: {payload['manifest_hash']}")
    print(f"raw: {payload['raw_record_count']}  "
          f"invalid-excluded: {payload['invalid_excluded_count']}  "
          f"duplicate-excluded: {payload['duplicate_excluded_count']}  "
          f"eligible: {payload['eligible_record_count']}  "
          f"selected: {payload['selected_record_count']}")
    print(f"{'wrote' if written else 'unchanged, already present at'}: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
