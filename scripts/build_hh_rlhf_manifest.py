"""Build (or re-verify) the frozen HH-RLHF harmless-base evaluation manifest (Task 017).

Downloads only the pinned source file (`Anthropic/hh-rlhf@<revision>`,
`harmless-base/test.jsonl.gz`), parses every raw row with the corrected transcript
parser, deduplicates by canonical prompt hash, selects exactly 200 records by the fixed
permutation, and writes `manifests/hh_rlhf_harmless_test_v1.json`. Refuses to overwrite
an existing manifest with different content (see `hh_rlhf.save_manifest`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import hf_hub_download

from steering.hh_rlhf import (
    HHRLHFError,
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
    parser.add_argument("--out", default="manifests/hh_rlhf_harmless_test_v1.json")
    args = parser.parse_args()

    print(f"downloading {REPOSITORY}@{REVISION} :: {SOURCE_FILE_PATH}")
    source_path = hf_hub_download(
        repo_id=REPOSITORY, filename=SOURCE_FILE_PATH, repo_type="dataset", revision=REVISION,
    )
    source_file = source_file_identity(source_path)
    print(f"compressed:   {source_file['compressed_size_bytes']} bytes, sha256={source_file['compressed_sha256']}")
    print(f"decompressed: {source_file['canonical_size_bytes']} bytes, sha256={source_file['canonical_sha256']}")

    rows = iter_raw_rows(source_path)
    print(f"loaded {len(rows)} raw rows")

    try:
        payload = build_manifest(rows, source_file=source_file)
    except HHRLHFError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    try:
        written = save_manifest(payload, out_path)
    except HHRLHFError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"manifest hash: {payload['manifest_hash']}")
    print(f"raw: {payload['raw_record_count']}  "
          f"parse-excluded: {payload['parse_excluded_count']} {payload['parse_excluded_by_reason']}  "
          f"duplicate-excluded: {payload['duplicate_excluded_count']}  "
          f"eligible: {payload['eligible_record_count']}  "
          f"selected: {payload['selected_record_count']}")
    print(f"{'wrote' if written else 'unchanged, already present at'}: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
