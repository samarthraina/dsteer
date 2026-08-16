"""Frozen AdvBench OOD evaluation panel (Task 018).

AdvBench is evaluation-only in this protocol: it never selects vectors, layers,
coefficients, rubric wording, or exclusions. Intervention vectors come exclusively from
the frozen HarmfulQA construction partition (`steering.splits`); coefficients come
exclusively from HarmfulQA calibration. This module only freezes and validates the
200-prompt evaluation panel itself -- it does not touch the experiment runners. Target
responses in the source (the "jailbreak" completions AdvBench ships alongside each
prompt) are ignored entirely; only prompts are ever read or hashed here.

Kept as its own module rather than folding into `steering.hh_rlhf`, because the source
shape is different: a single flat parquet table of (prompt, target) pairs with no
transcript structure, no chosen/rejected agreement to check, and no multi-turn parsing.
Reuses the same permutation convention as the HH-RLHF panel
(`sha256(seed:source_id)-sort-source_id-tiebreak-v1`) so both frozen evaluation panels
are selected the same way.

The frozen source (`walledai/AdvBench`, `data/train-00000-of-00001.parquet`, revision
`9d4730540082fa4017450b65ca1c0e1d8d30446e`) has 520 raw rows, all non-empty strings, no
duplicate normalized prompts -- so all 520 are eligible, of which exactly 200 are
selected.

The manifest never stores raw prompt text, only source IDs, source indices, prompt
hashes, permutation positions, and partition -- AdvBench prompts are harmful-behavior
strings that should not be committed to the repository. The frozen runtime loader
reconstructs prompt text fresh from the pinned source at every call and verifies it
against the manifest's hashes before returning anything.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

SCHEMA_VERSION = 1

# Same convention as steering.hh_rlhf.PERMUTATION_ALGORITHM: sha256(f"{seed}:{source_id}")
# sorted ascending, source_id as tie-breaker, selecting the first N from a larger
# eligible pool rather than partitioning the whole pool.
PERMUTATION_ALGORITHM = "sha256(seed:source_id)-sort-source_id-tiebreak-v1"

REPOSITORY = "walledai/AdvBench"
CONFIG = "default"
SPLIT = "train"
REVISION = "9d4730540082fa4017450b65ca1c0e1d8d30446e"
SOURCE_FILE_PATH = "data/train-00000-of-00001.parquet"

SEED = 20260815
PARTITION = "evaluation"

# Empirically derived from the pinned source (see module docstring); `build_manifest`
# refuses to proceed if a fresh read produces different counts instead of silently
# accepting whatever it finds.
RAW_RECORD_COUNT = 520
EXPECTED_INVALID_EXCLUDED_COUNT = 0
EXPECTED_DUPLICATE_EXCLUDED_COUNT = 0
EXPECTED_ELIGIBLE_RECORD_COUNT = 520
SELECTED_RECORD_COUNT = 200


class AdvBenchError(ValueError):
    """A canonicalization, manifest-construction, or manifest-verification invariant
    was violated."""


# Canonicalization + hashing


def normalize_prompt(text: str) -> str:
    """NFC-normalize and fold CRLF/CR to LF. Deliberately does not strip or collapse
    whitespace -- conservative canonicalization per the frozen contract."""
    text = unicodedata.normalize("NFC", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def prompt_hash(text: str) -> str:
    """SHA-256 hex digest of the normalized prompt."""
    return hashlib.sha256(normalize_prompt(text).encode("utf-8")).hexdigest()


def is_valid_prompt(value: Any) -> bool:
    """A valid prompt is a string that is non-empty after normalization. Whitespace
    that survives normalization (never stripped) counts as content, not emptiness."""
    return isinstance(value, str) and len(normalize_prompt(value)) > 0


def permutation_sort_key(seed: int, source_id: str) -> str:
    return hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).hexdigest()


# Source file loading


def iter_raw_rows(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read every row from the pinned parquet file, in on-disk row order -- that order
    is each row's stable `source_index`. Only "prompt" is used; "target" (the source's
    jailbreak completion) is carried through but never read by anything here."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pylist()


def hash_file(path: Union[str, Path]) -> Tuple[str, int]:
    """(sha256_hexdigest, size_bytes) of the file at `path`, read as raw bytes."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def source_file_identity(path: Union[str, Path]) -> Dict[str, Any]:
    sha256, size = hash_file(path)
    return {"path": SOURCE_FILE_PATH, "sha256": sha256, "size_bytes": size}


# Parsing + dedup over the full raw source


def parse_all_rows(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Validate every raw row's "prompt" field. Returns (valid_records, invalid_exclusions),
    both in source_index order.

    `valid_records` carry the raw prompt text in memory (needed by the runtime loader)
    but that text is never included in anything written to disk -- `build_manifest`
    only takes id/index/hash fields from these records.
    """
    valid: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        source_id = f"advbench-{i}"
        raw = row.get("prompt") if isinstance(row, Mapping) else None
        if not is_valid_prompt(raw):
            invalid.append({"source_id": source_id, "source_index": i, "reason": "invalid_prompt"})
            continue
        valid.append({
            "source_id": source_id,
            "source_index": i,
            "prompt": raw,
            "prompt_hash": prompt_hash(raw),
        })
    return valid, invalid


def resolve_duplicates(records: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Group valid records by canonical prompt hash; keep the lowest `source_index`
    from each group, exclude the rest, and record the full exclusion lineage. Mirrors
    `steering.hh_rlhf.resolve_duplicates`: a group larger than two is refused rather
    than silently resolved -- an unexpectedly large duplicate group means the source
    changed underneath the manifest.

    Returns (retained, exclusions):
      retained:   one full record per group (the lowest source_index member).
      exclusions: [{"excluded_source_id", "excluded_source_index",
                     "retained_source_id", "retained_source_index", "prompt_hash"}, ...]
    """
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for rec in records:
        groups.setdefault(rec["prompt_hash"], []).append(rec)

    retained: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    for phash, group in groups.items():
        if len(group) > 2:
            ids = [g["source_id"] for g in group]
            raise AdvBenchError(
                f"prompt hash {phash} has {len(group)} verbatim duplicates {ids}; "
                "the pinned source has only pairs at most -- refusing to guess a resolution"
            )
        group_sorted = sorted(group, key=lambda g: g["source_index"])
        keep = group_sorted[0]
        retained.append(dict(keep))
        for dropped in group_sorted[1:]:
            exclusions.append({
                "excluded_source_id": dropped["source_id"],
                "excluded_source_index": dropped["source_index"],
                "retained_source_id": keep["source_id"],
                "retained_source_index": keep["source_index"],
                "prompt_hash": phash,
            })
    return retained, exclusions


# Manifest construction, hashing, save/load


def canonical_manifest_json(payload: Mapping[str, Any]) -> bytes:
    """The exact serialization the manifest hash is computed over: compact, key-sorted,
    ASCII-only, with `manifest_hash` itself excluded."""
    body = {k: v for k, v in payload.items() if k != "manifest_hash"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def compute_manifest_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_manifest_json(payload)).hexdigest()


def build_manifest(
    rows: Sequence[Mapping[str, Any]],
    source_file: Mapping[str, Any],
    seed: int = SEED,
    dataset_revision: str = REVISION,
    expected_raw_count: int = RAW_RECORD_COUNT,
    expected_selected_count: int = SELECTED_RECORD_COUNT,
) -> Dict[str, Any]:
    """Build the frozen 200-prompt evaluation manifest from the raw (pre-filter,
    pre-selection) source rows, in file order.

    Raises `AdvBenchError` on any frozen-requirement violation -- including an
    unexpected raw count -- instead of silently adjusting counts or dropping records.
    Does not assume any particular invalid/duplicate count in advance; those are
    derived here and simply recorded. The returned payload never contains raw prompt
    text, only IDs, indices, and hashes.
    """
    if not dataset_revision:
        raise AdvBenchError("dataset_revision must be a nonempty exact revision")
    if len(rows) != expected_raw_count:
        raise AdvBenchError(f"expected exactly {expected_raw_count} raw rows, got {len(rows)}")

    valid, invalid_exclusions = parse_all_rows(rows)
    retained, duplicate_exclusions = resolve_duplicates(valid)

    eligible_count = len(retained)
    if eligible_count < expected_selected_count:
        raise AdvBenchError(f"only {eligible_count} eligible records, need at least {expected_selected_count}")

    entries = [
        {"source_id": r["source_id"], "source_index": r["source_index"], "prompt_hash": r["prompt_hash"]}
        for r in retained
    ]
    entries.sort(key=lambda e: (permutation_sort_key(seed, e["source_id"]), e["source_id"]))
    selected = entries[:expected_selected_count]
    for position, entry in enumerate(selected):
        entry["permuted_position"] = position
        entry["partition"] = PARTITION

    # Canonical order in the manifest is independent of the caller's input/selection
    # order -- required for byte-identical manifests regardless of iteration order.
    selected_by_id = sorted(selected, key=lambda e: e["source_id"])
    invalid_exclusions_sorted = sorted(invalid_exclusions, key=lambda e: e["source_index"])
    duplicate_exclusions_sorted = sorted(duplicate_exclusions, key=lambda e: e["excluded_source_index"])

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "repository": REPOSITORY,
            "config": CONFIG,
            "split": SPLIT,
            "revision": dataset_revision,
        },
        "source_file": dict(source_file),
        "permutation": {
            "seed": seed,
            "algorithm": PERMUTATION_ALGORITHM,
        },
        "raw_record_count": len(rows),
        "invalid_excluded_count": len(invalid_exclusions),
        "duplicate_excluded_count": len(duplicate_exclusions),
        "eligible_record_count": eligible_count,
        "selected_record_count": len(selected),
        "invalid_exclusions": invalid_exclusions_sorted,
        "duplicate_exclusions": duplicate_exclusions_sorted,
        "records": selected_by_id,
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
    Raises `AdvBenchError` if an existing manifest differs -- a manifest is never
    silently overwritten with different content.
    """
    path = Path(path)
    text = dumps_manifest(payload)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != text:
            raise AdvBenchError(f"manifest at {path} already exists with different content; refusing to overwrite")
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
        raise AdvBenchError(f"manifest hash mismatch at {path}: stored {stored!r}, recomputed {recomputed!r}")
    return payload


# Frozen identity + source-binding verification
#
# FROZEN_ADVBENCH_IDENTITY is filled in with the real manifest_hash and derived counts
# once `scripts/build_advbench_manifest.py` has built the manifest for real (Task 018).
# `load_manifest` already verifies the stored hash against a fresh recomputation of the
# file's own content; this pins that hash -- and everything upstream of it -- to the
# single value the frozen loader is allowed to trust, so a manifest that is internally
# self-consistent but was rebuilt from different content is still rejected.
FROZEN_ADVBENCH_IDENTITY: Dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "repository": REPOSITORY,
    "config": CONFIG,
    "split": SPLIT,
    "revision": REVISION,
    "seed": SEED,
    "algorithm": PERMUTATION_ALGORITHM,
    "manifest_hash": "0bb0269883d6137157fd5f41f48e74f7d4d74251430452ff5508daa9bf5313f9",
    "raw_record_count": RAW_RECORD_COUNT,
    "invalid_excluded_count": EXPECTED_INVALID_EXCLUDED_COUNT,
    "duplicate_excluded_count": EXPECTED_DUPLICATE_EXCLUDED_COUNT,
    "eligible_record_count": EXPECTED_ELIGIBLE_RECORD_COUNT,
    "selected_record_count": SELECTED_RECORD_COUNT,
    "source_sha256": "168902da5a510479f69e17cd22b2c99699cd0e6980141ee6af18ddeb036a80d3",
}


def validate_manifest_identity(
    manifest: Mapping[str, Any], expected: Mapping[str, Any] = FROZEN_ADVBENCH_IDENTITY,
) -> None:
    """Check a loaded manifest against a frozen identity before it selects any record.
    Raises `AdvBenchError` listing every field that disagrees."""
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
    for key in ("raw_record_count", "invalid_excluded_count", "duplicate_excluded_count",
                "eligible_record_count", "selected_record_count"):
        check(key, manifest.get(key), expected[key])
    check("source_file.sha256", manifest.get("source_file", {}).get("sha256"), expected["source_sha256"])

    if mismatches:
        raise AdvBenchError("manifest does not match the frozen identity: " + "; ".join(mismatches))


def validate_source_binding(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    source_file: Mapping[str, Any],
) -> None:
    """Cross-check a freshly re-downloaded/re-parsed source against a validated
    manifest before any record is returned. Raises `AdvBenchError` on any mismatch --
    source drift, changed prompt content, changed selected membership, reordered
    records, or altered exclusions all fail loudly, never silently.
    """
    if source_file.get("sha256") != manifest["source_file"]["sha256"]:
        raise AdvBenchError("source file content has drifted from the manifest")

    if len(rows) != manifest["raw_record_count"]:
        raise AdvBenchError(f"expected {manifest['raw_record_count']} raw rows, got {len(rows)}")

    valid, invalid_exclusions = parse_all_rows(rows)
    if len(invalid_exclusions) != manifest["invalid_excluded_count"]:
        raise AdvBenchError(
            f"expected {manifest['invalid_excluded_count']} invalid exclusions, got {len(invalid_exclusions)}"
        )
    reconstructed_invalid_ids = {e["source_id"] for e in invalid_exclusions}
    manifest_invalid_ids = {e["source_id"] for e in manifest["invalid_exclusions"]}
    if reconstructed_invalid_ids != manifest_invalid_ids:
        raise AdvBenchError("reconstructed invalid-prompt exclusions do not match the manifest's lineage")

    retained, duplicate_exclusions = resolve_duplicates(valid)
    if len(duplicate_exclusions) != manifest["duplicate_excluded_count"]:
        raise AdvBenchError(
            f"expected {manifest['duplicate_excluded_count']} duplicate exclusions, got {len(duplicate_exclusions)}"
        )
    reconstructed_dup_ids = {e["excluded_source_id"] for e in duplicate_exclusions}
    manifest_dup_ids = {e["excluded_source_id"] for e in manifest["duplicate_exclusions"]}
    if reconstructed_dup_ids != manifest_dup_ids:
        raise AdvBenchError("reconstructed duplicate exclusions do not match the manifest's lineage")

    if len(retained) != manifest["eligible_record_count"]:
        raise AdvBenchError(f"expected {manifest['eligible_record_count']} eligible records, got {len(retained)}")

    retained_by_id = {r["source_id"]: r for r in retained}
    manifest_records = manifest["records"]
    if len(manifest_records) != manifest["selected_record_count"]:
        raise AdvBenchError("manifest selected_record_count does not match its own records list")

    seen_positions = set()
    for entry in manifest_records:
        sid = entry["source_id"]
        raw = retained_by_id.get(sid)
        if raw is None:
            raise AdvBenchError(f"selected source_id {sid!r} is not among the reconstructed eligible records")
        if raw["prompt_hash"] != entry["prompt_hash"]:
            raise AdvBenchError(f"content for selected {sid!r} has drifted from the manifest")
        if entry["permuted_position"] in seen_positions:
            raise AdvBenchError(f"permuted_position {entry['permuted_position']} appears more than once")
        seen_positions.add(entry["permuted_position"])

    if sorted(seen_positions) != list(range(manifest["selected_record_count"])):
        raise AdvBenchError("permuted positions are not exactly contiguous from 0")


# Frozen runtime loader


def _default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[2] / "manifests" / "advbench_v1.json"


def load_advbench_evaluation(
    manifest_path: Optional[Union[str, Path]] = None,
    source_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Load the frozen 200-prompt AdvBench evaluation panel exactly as fixed by the
    manifest.

    Downloads (or reuses a locally cached copy of) only the pinned source file,
    re-verifies its content identity, reconstructs every raw stable identity, and
    re-verifies all manifest exclusions and selected membership before returning
    anything -- membership and order are entirely determined by the manifest, never by
    a caller-supplied `n`/`seed`. Target completions in the source are never read.

    Returns exactly `SELECTED_RECORD_COUNT` records in ascending `permuted_position`,
    each with: source_id, source_index, prompt, prompt_hash, partition,
    permuted_position, manifest_hash.
    """
    path = Path(manifest_path) if manifest_path is not None else _default_manifest_path()
    manifest = load_manifest(path)
    validate_manifest_identity(manifest)

    if source_path is None:
        from huggingface_hub import hf_hub_download

        source_path = hf_hub_download(
            repo_id=manifest["dataset"]["repository"],
            filename=manifest["source_file"]["path"],
            repo_type="dataset",
            revision=manifest["dataset"]["revision"],
        )

    source_file = source_file_identity(source_path)
    rows = iter_raw_rows(source_path)
    validate_source_binding(manifest, rows, source_file)

    valid, _ = parse_all_rows(rows)
    by_id = {r["source_id"]: r for r in valid}

    selected = []
    for entry in manifest["records"]:
        raw = by_id[entry["source_id"]]
        selected.append({
            "source_id": raw["source_id"],
            "source_index": raw["source_index"],
            "prompt": raw["prompt"],
            "prompt_hash": raw["prompt_hash"],
            "partition": entry["partition"],
            "permuted_position": entry["permuted_position"],
            "manifest_hash": manifest["manifest_hash"],
        })
    selected.sort(key=lambda r: r["permuted_position"])
    return selected
