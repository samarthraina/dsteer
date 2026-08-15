"""Immutable partition manifests (protocol Section 5).

A manifest fixes, once, which source records fall in construction/calibration/
final_evaluation/development. The permutation depends only on a fixed seed and each
record's stable source ID -- never on the order records happen to arrive in -- so the
same source corpus always produces the same manifest, and a leakage check can compare
IDs and prompt hashes instead of trusting that nobody re-shuffled since.

HarmfulQA's frozen source (`declare-lab/HarmfulQA`, config `default`, split `train`,
revision `6f1a78aed47d16c0695e4595d0159abc38197bfd`) has 1,960 raw rows but only 1,938
unique normalized prompts: 22 verbatim duplicate pairs. The amendment authorizing their
exclusion predates any pilot or evaluation (protocol Section 5). `resolve_duplicates`
keeps the lower source-index row from each pair and records the full exclusion lineage,
so the manifest documents exactly which rows were dropped and why.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

SCHEMA_VERSION = 2

# sha256(f"{seed}:{source_id}") sorted ascending, source_id as tie-breaker for the
# (astronomically unlikely) case of a hash collision. Documented so a reimplementation
# in another language can reproduce the same permutation bit-for-bit.
PERMUTATION_ALGORITHM = "sha256(seed:source_id)-sort-source_id-tiebreak-v1"

# The frozen HarmfulQA source: 1,960 raw rows, 22 verbatim duplicate pairs excluded,
# 1,938 retained and partitioned. These are defaults, not hard limits -- callers
# building a manifest from a different snapshot pass their own expected counts.
RAW_RECORD_COUNT = 1960
EXPECTED_DUPLICATE_PAIRS = 22
RETAINED_RECORD_COUNT = RAW_RECORD_COUNT - EXPECTED_DUPLICATE_PAIRS

# (partition name, start position inclusive, end position exclusive)
PARTITION_BOUNDS = (
    ("construction", 0, 1378),
    ("calibration", 1378, 1578),
    ("final_evaluation", 1578, 1878),
    ("development", 1878, 1938),
)
TOTAL_RECORDS = PARTITION_BOUNDS[-1][2]
assert TOTAL_RECORDS == RETAINED_RECORD_COUNT


class SplitError(ValueError):
    """A manifest-construction or manifest-verification invariant was violated."""


def normalize_prompt(text: str) -> str:
    """NFC-normalize, fold CRLF/CR to LF, and strip leading/trailing whitespace only.

    Deliberately does not lowercase or collapse internal whitespace -- either would
    make the hash blind to real prompt-rendering differences.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def prompt_hash(text: str) -> str:
    """SHA-256 hex digest of the normalized prompt."""
    return hashlib.sha256(normalize_prompt(text).encode("utf-8")).hexdigest()


def permutation_sort_key(seed: int, source_id: str) -> str:
    return hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).hexdigest()


def partition_for_position(position: int) -> str:
    for name, lo, hi in PARTITION_BOUNDS:
        if lo <= position < hi:
            return name
    raise SplitError(f"position {position} is outside the defined partitions (0-{TOTAL_RECORDS - 1})")


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    """The exact serialization the manifest hash is computed over.

    Compact, key-sorted, ASCII-only, with `manifest_hash` itself excluded -- the hash
    cannot depend on its own value. Writing and verification must both call this.
    """
    body = {k: v for k, v in payload.items() if k != "manifest_hash"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def compute_manifest_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def resolve_duplicates(
    raw_records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Group `raw_records` by normalized prompt hash; keep the lowest `source_index`
    from each group, exclude the rest, and record the full exclusion lineage.

    Each `raw_records` entry must carry `source_id`, integer `source_index`, and either
    `prompt` or a precomputed `prompt_hash`. A group larger than two is refused rather
    than silently resolved -- the frozen source has only verbatim pairs, and a bigger
    group means the source changed underneath the manifest.

    Returns (retained, exclusions):
      retained:   [{"source_id", "source_index", "prompt_hash"}, ...] one per group.
      exclusions: [{"excluded_source_id", "excluded_source_index",
                     "retained_source_id", "retained_source_index",
                     "prompt_hash"}, ...] one per dropped row.
    """
    seen_source_ids = set()
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for rec in raw_records:
        source_id = rec["source_id"]
        if source_id in seen_source_ids:
            raise SplitError(f"duplicate source_id in raw input: {source_id!r}")
        seen_source_ids.add(source_id)

        phash = rec.get("prompt_hash") or prompt_hash(rec["prompt"])
        entry = {"source_id": source_id, "source_index": rec["source_index"], "prompt_hash": phash}
        groups.setdefault(phash, []).append(entry)

    retained: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    for phash, group in groups.items():
        if len(group) > 2:
            ids = [g["source_id"] for g in group]
            raise SplitError(
                f"prompt hash {phash} has {len(group)} verbatim duplicates {ids}; "
                "the frozen source has only pairs -- refusing to guess a resolution"
            )
        group_sorted = sorted(group, key=lambda g: g["source_index"])
        keep = group_sorted[0]
        retained.append({"source_id": keep["source_id"], "source_index": keep["source_index"], "prompt_hash": phash})
        for dropped in group_sorted[1:]:
            exclusions.append({
                "excluded_source_id": dropped["source_id"],
                "excluded_source_index": dropped["source_index"],
                "retained_source_id": keep["source_id"],
                "retained_source_index": keep["source_index"],
                "prompt_hash": phash,
            })

    return retained, exclusions


def build_manifest(
    raw_records: Sequence[Mapping[str, Any]],
    seed: int,
    dataset_repo: str,
    dataset_split: str,
    dataset_revision: str,
    dataset_config: Optional[str] = None,
    expected_raw_count: int = RAW_RECORD_COUNT,
    expected_retained_count: int = RETAINED_RECORD_COUNT,
    expected_duplicate_pairs: int = EXPECTED_DUPLICATE_PAIRS,
) -> Dict[str, Any]:
    """Build the frozen partition manifest from the raw (pre-dedup) source rows.

    `raw_records` must each carry a stable `source_id`, integer `source_index`, and
    either a `prompt` (hashed here) or a precomputed `prompt_hash`. Raises `SplitError`
    on any frozen-requirement violation -- including an unexpected raw count or
    duplicate-pair count -- instead of silently adjusting counts or dropping records.
    """
    if not dataset_revision:
        raise SplitError("dataset_revision must be a nonempty exact revision")
    if len(raw_records) != expected_raw_count:
        raise SplitError(f"expected exactly {expected_raw_count} raw records, got {len(raw_records)}")

    retained, exclusions = resolve_duplicates(raw_records)

    if len(exclusions) != expected_duplicate_pairs:
        raise SplitError(
            f"expected exactly {expected_duplicate_pairs} duplicate exclusions, "
            f"got {len(exclusions)}"
        )
    if len(retained) != expected_retained_count:
        raise SplitError(f"expected exactly {expected_retained_count} retained records, got {len(retained)}")

    retained_hashes = {r["prompt_hash"] for r in retained}
    if len(retained_hashes) != len(retained):
        raise SplitError("retained records still contain a duplicate normalized prompt hash")

    retained_ids = {r["source_id"] for r in retained}
    excluded_ids = {e["excluded_source_id"] for e in exclusions}
    if retained_ids & excluded_ids:
        raise SplitError("a source_id appears both retained and excluded")
    for excl in exclusions:
        if excl["retained_source_id"] not in retained_ids:
            raise SplitError(f"exclusion {excl['excluded_source_id']!r} references a retained_source_id that is not retained")

    entries = [{"source_id": r["source_id"], "prompt_hash": r["prompt_hash"]} for r in retained]
    entries.sort(key=lambda e: (permutation_sort_key(seed, e["source_id"]), e["source_id"]))
    for position, entry in enumerate(entries):
        entry["permuted_position"] = position
        entry["partition"] = partition_for_position(position)

    counts = {name: 0 for name, _, _ in PARTITION_BOUNDS}
    for entry in entries:
        counts[entry["partition"]] += 1
    expected_partition_counts = {name: hi - lo for name, lo, hi in PARTITION_BOUNDS}
    if counts != expected_partition_counts:
        raise SplitError(f"partition counts {counts} do not match {expected_partition_counts}")

    # Canonical order in the manifest is independent of the caller's input order --
    # required for byte-identical manifests regardless of how `raw_records` was
    # assembled. Records by source_id; exclusions by the numeric index that was
    # dropped, which is stable and human-readable.
    entries_by_id = sorted(entries, key=lambda e: e["source_id"])
    exclusions_sorted = sorted(exclusions, key=lambda e: e["excluded_source_index"])

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "repository": dataset_repo,
            "config": dataset_config,
            "split": dataset_split,
            "revision": dataset_revision,
        },
        "permutation": {
            "seed": seed,
            "algorithm": PERMUTATION_ALGORITHM,
        },
        "raw_record_count": len(raw_records),
        "retained_record_count": len(retained),
        "excluded_record_count": len(exclusions),
        "duplicate_exclusions": exclusions_sorted,
        "partition_counts": counts,
        "records": entries_by_id,
    }
    payload["manifest_hash"] = compute_manifest_hash(payload)
    return payload


def dumps_manifest(payload: Mapping[str, Any]) -> str:
    """Human-readable on-disk form. Deterministic for a given payload, so repeated
    writes of the same content are byte-identical."""
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def save_manifest(payload: Mapping[str, Any], path: Union[str, Path]) -> bool:
    """Write `payload` to `path` if absent. If present, require byte-identical content.

    Returns True if bytes were written, False if an identical manifest already existed.
    Raises `SplitError` if an existing manifest differs -- a manifest is never silently
    overwritten with different content.
    """
    path = Path(path)
    text = dumps_manifest(payload)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != text:
            raise SplitError(f"manifest at {path} already exists with different content; refusing to overwrite")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def load_manifest(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a manifest and verify its stored hash against a fresh recomputation."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored = payload.get("manifest_hash")
    recomputed = compute_manifest_hash(payload)
    if stored != recomputed:
        raise SplitError(f"manifest hash mismatch at {path}: stored {stored!r}, recomputed {recomputed!r}")
    return payload


# The one manifest this task is reviewed against (protocol Section 5 / Task 002,
# commit ad96b54654e823f9ce58910b8df2274cffcc4f1b). `load_manifest` already verifies
# the stored hash against a fresh recomputation of the file's own content; this pins
# that hash -- and everything upstream of it -- to the single value a partition loader
# is allowed to trust, so a manifest swapped in from elsewhere is rejected even if it is
# internally self-consistent.
FROZEN_HARMFULQA_IDENTITY: Dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "repository": "declare-lab/HarmfulQA",
    "config": "default",
    "split": "train",
    "revision": "6f1a78aed47d16c0695e4595d0159abc38197bfd",
    "seed": 20260815,
    "algorithm": PERMUTATION_ALGORITHM,
    "manifest_hash": "0e1d0a8177d59eba601141a74da19a189527821dab573e203a3261cff54eec70",
    "raw_record_count": RAW_RECORD_COUNT,
    "retained_record_count": RETAINED_RECORD_COUNT,
    "excluded_record_count": EXPECTED_DUPLICATE_PAIRS,
    "partition_counts": {name: hi - lo for name, lo, hi in PARTITION_BOUNDS},
}


def validate_manifest_identity(
    manifest: Mapping[str, Any], expected: Mapping[str, Any] = FROZEN_HARMFULQA_IDENTITY,
) -> None:
    """Check a loaded manifest against a frozen identity before it selects any
    partition. Raises `SplitError` listing every field that disagrees.
    """
    mismatches: List[str] = []

    def check(label: str, actual: Any, want: Any) -> None:
        if actual != want:
            mismatches.append(f"{label}: expected {want!r}, got {actual!r}")

    check("schema_version", manifest.get("schema_version"), expected["schema_version"])
    dataset = manifest.get("dataset", {})
    for key in ("repository", "config", "split", "revision"):
        check(f"dataset.{key}", dataset.get(key), expected[key])
    permutation = manifest.get("permutation", {})
    check("permutation.seed", permutation.get("seed"), expected["seed"])
    check("permutation.algorithm", permutation.get("algorithm"), expected["algorithm"])
    check("manifest_hash", manifest.get("manifest_hash"), expected["manifest_hash"])
    for key in ("raw_record_count", "retained_record_count", "excluded_record_count"):
        check(key, manifest.get(key), expected[key])
    check("partition_counts", manifest.get("partition_counts"), expected["partition_counts"])

    if mismatches:
        raise SplitError("manifest does not match the frozen identity: " + "; ".join(mismatches))


def validate_source_binding(manifest: Mapping[str, Any], raw_records: Sequence[Mapping[str, Any]]) -> None:
    """Cross-check freshly reconstructed raw source rows against a validated manifest
    before any partition is returned. Each `raw_records` entry must carry `source_id`,
    `source_index`, and `prompt_hash`. Raises `SplitError` on any mismatch -- a row is
    never silently skipped or repaired.
    """
    if len(raw_records) != manifest["raw_record_count"]:
        raise SplitError(f"expected {manifest['raw_record_count']} raw rows, got {len(raw_records)}")

    raw_by_id: Dict[str, Mapping[str, Any]] = {}
    for rec in raw_records:
        sid = rec["source_id"]
        if sid in raw_by_id:
            raise SplitError(f"duplicate source_id reconstructed from raw source: {sid!r}")
        raw_by_id[sid] = rec
    if len({r["source_index"] for r in raw_records}) != len(raw_records):
        raise SplitError("duplicate source_index reconstructed from raw source")

    retained_entries = manifest["records"]
    if len(retained_entries) != manifest["retained_record_count"]:
        raise SplitError("manifest retained_record_count does not match its own records list")

    retained_ids = set()
    for entry in retained_entries:
        sid = entry["source_id"]
        if sid in retained_ids:
            raise SplitError(f"manifest lists retained source_id {sid!r} more than once")
        retained_ids.add(sid)
        raw = raw_by_id.get(sid)
        if raw is None:
            raise SplitError(f"manifest retained source_id {sid!r} is missing from the raw source")
        if raw["prompt_hash"] != entry["prompt_hash"]:
            raise SplitError(f"prompt hash mismatch for retained {sid!r}: source has drifted from the manifest")

    excluded_entries = manifest["duplicate_exclusions"]
    if len(excluded_entries) != manifest["excluded_record_count"]:
        raise SplitError("manifest excluded_record_count does not match its own duplicate_exclusions list")

    for excl in excluded_entries:
        excluded_id = excl["excluded_source_id"]
        retained_id = excl["retained_source_id"]
        if excluded_id in retained_ids:
            raise SplitError(f"excluded source_id {excluded_id!r} also appears retained")
        raw_excluded = raw_by_id.get(excluded_id)
        if raw_excluded is None:
            raise SplitError(f"excluded source_id {excluded_id!r} is missing from the raw source")
        if retained_id not in retained_ids:
            raise SplitError(f"exclusion for {excluded_id!r} points to a retained_source_id that is not retained")
        raw_retained = raw_by_id[retained_id]
        if raw_excluded["prompt_hash"] != excl["prompt_hash"] or raw_retained["prompt_hash"] != excl["prompt_hash"]:
            raise SplitError(f"exclusion lineage for {excluded_id!r} does not match recomputed prompt hashes")
        if raw_retained["source_index"] >= raw_excluded["source_index"]:
            raise SplitError(f"exclusion for {excluded_id!r} does not retain the lower source index")

    retained_hashes = [e["prompt_hash"] for e in retained_entries]
    if len(set(retained_hashes)) != len(retained_hashes):
        raise SplitError("retained prompt hashes are not unique")

    positions = sorted(e["permuted_position"] for e in retained_entries)
    if positions != list(range(len(retained_entries))):
        raise SplitError("permuted positions are not exactly contiguous from 0")

    by_partition: Dict[str, List[int]] = {}
    for entry in retained_entries:
        by_partition.setdefault(entry["partition"], []).append(entry["permuted_position"])
    for name, lo, hi in PARTITION_BOUNDS:
        actual = sorted(by_partition.get(name, []))
        if actual != list(range(lo, hi)):
            raise SplitError(f"partition {name!r} does not occupy positions [{lo}, {hi})")

    # No source ID or prompt hash can appear in more than one partition -- guaranteed by
    # the checks above (each retained source_id/hash appears exactly once in `records`,
    # each record has exactly one partition), verified explicitly for defense.
    seen_ids: set = set()
    seen_hashes: set = set()
    for entry in retained_entries:
        if entry["source_id"] in seen_ids:
            raise SplitError(f"source_id {entry['source_id']!r} appears more than once across partitions")
        if entry["prompt_hash"] in seen_hashes:
            raise SplitError(f"prompt_hash {entry['prompt_hash']!r} appears more than once across partitions")
        seen_ids.add(entry["source_id"])
        seen_hashes.add(entry["prompt_hash"])
