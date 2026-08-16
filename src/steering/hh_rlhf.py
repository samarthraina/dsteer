"""Frozen HH-RLHF harmless-base evaluation panel (Task 017).

HH-RLHF is evaluation-only in this protocol: it never selects layers, coefficients,
vector methods, judge wording, or exclusions, and it does not restore the invalid
historical HH-RLHF response_last activation experiment. Intervention vectors come
exclusively from the frozen HarmfulQA construction partition (`steering.splits`); the
eventual HH-RLHF panel will use coefficients selected exclusively on HarmfulQA
calibration. This module only freezes the 200-record evaluation panel itself.

Kept as its own module rather than generalizing `steering.splits`, because the
selection shape is different: HarmfulQA partitions its entire deduplicated retained set
into four contiguous ranges; HH-RLHF selects exactly 200 records out of a much larger
eligible pool and leaves the rest unselected. Making `splits.py` support both shapes
would mean parameterizing partition geometry for a module that otherwise has nothing to
do with this dataset.

The frozen source (`Anthropic/hh-rlhf`, `harmless-base/test.jsonl.gz`, revision
`09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa`) has 2,312 raw rows. The original parser
(`steering.data._split_hh_conversation`) split transcripts on every blank line and
silently dropped any paragraph that did not start with a fresh "Human:"/"Assistant:"
marker -- corrupting any multi-paragraph turn. The corrected parser here finds turn
boundaries only at a blank line immediately followed by a role marker
(`\\n\\n(Human|Assistant): `), so internal blank lines within a turn are preserved.
Under the corrected parser: 12 rows have a malformed `chosen` transcript, 1 has a
malformed `rejected` transcript, 0 have a chosen/rejected prompt-history mismatch, and 2
pairs of the remaining 2,299 rows share an identical canonical prompt -- leaving 2,297
eligible records, of which exactly 200 are selected.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

SCHEMA_VERSION = 1

# Same convention as steering.splits.PERMUTATION_ALGORITHM: sha256(f"{seed}:{source_id}")
# sorted ascending, source_id as tie-breaker. Here it selects the first N from a larger
# eligible pool rather than partitioning the whole pool.
PERMUTATION_ALGORITHM = "sha256(seed:source_id)-sort-source_id-tiebreak-v1"

REPOSITORY = "Anthropic/hh-rlhf"
DATA_DIR = "harmless-base"
SPLIT = "test"
REVISION = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"
SOURCE_FILE_PATH = "harmless-base/test.jsonl.gz"

SEED = 20260815
PARTITION = "evaluation"

# Empirically derived from the pinned source (see module docstring); `build_manifest`
# refuses to proceed if a fresh parse produces different counts instead of silently
# accepting whatever it finds.
RAW_RECORD_COUNT = 2312
EXPECTED_PARSE_EXCLUDED_COUNT = 13
EXPECTED_HISTORY_MISMATCH_COUNT = 0
EXPECTED_DUPLICATE_EXCLUDED_COUNT = 2
EXPECTED_ELIGIBLE_RECORD_COUNT = 2297
SELECTED_RECORD_COUNT = 200


class HHRLHFError(ValueError):
    """A transcript-parsing, manifest-construction, or manifest-verification invariant
    was violated."""


# Text normalization + canonical hashing


def normalize_text(text: str) -> str:
    """NFC-normalize and fold CRLF/CR to LF. Deliberately does not strip or collapse
    whitespace -- turn boundaries are found structurally (see `_find_turns`), and
    stripping here would risk merging or misplacing turn content."""
    text = unicodedata.normalize("NFC", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def conversation_hash(messages: Sequence[Mapping[str, str]]) -> str:
    """SHA-256 of the canonical JSON of the full ordered prompt-message list -- the
    entire history, never only the first user turn."""
    return hashlib.sha256(_canonical_json(list(messages))).hexdigest()


def response_hash(text: str) -> str:
    """SHA-256 of the normalized final response text."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


# Transcript parsing


_TURN_BOUNDARY_RE = re.compile(r"\n\n(Human|Assistant): ")
_ROLE_MAP = {"Human": "user", "Assistant": "assistant"}


def _find_turns(text: str) -> List[Tuple[str, str]]:
    """Split a normalized transcript into (role_label, content) turns.

    A turn boundary is a blank line immediately followed by "Human: " or "Assistant: ",
    never a bare blank line -- a multi-paragraph turn's own internal blank lines are not
    followed by a role marker and so stay inside `content` untouched. The leading
    "\\n\\n" prepended below lets the same regex catch the transcript's first turn
    without a separate case.
    """
    padded = "\n\n" + text.strip()
    matches = list(_TURN_BOUNDARY_RE.finditer(padded))
    turns = []
    for i, m in enumerate(matches):
        role_label = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(padded)
        content = padded[start:end].strip("\n")
        turns.append((role_label, content))
    return turns


def parse_transcript(raw_text: str) -> Tuple[List[Dict[str, str]], str]:
    """Parse one HH-RLHF transcript into (prompt_messages, final_response).

    Raises `HHRLHFError` if the transcript is empty, has no recognisable turns, does not
    strictly alternate Human/Assistant starting with Human, or does not end with an
    Assistant turn. A malformed transcript is never silently repaired or partially
    accepted -- the caller must exclude it.
    """
    if not raw_text or not raw_text.strip():
        raise HHRLHFError("empty transcript")
    turns = _find_turns(normalize_text(raw_text))
    if not turns:
        raise HHRLHFError("no Human:/Assistant: turns found")

    expected = "Human"
    for role_label, _content in turns:
        if role_label != expected:
            raise HHRLHFError(
                f"turns are not strictly alternating starting with Human "
                f"(found {role_label!r} where {expected!r} was expected)"
            )
        expected = "Assistant" if expected == "Human" else "Human"
    if turns[-1][0] != "Assistant":
        raise HHRLHFError("transcript does not end with an Assistant turn")

    messages = [{"role": _ROLE_MAP[role], "content": content} for role, content in turns]
    return messages[:-1], messages[-1]["content"]


def parse_row(source_index: int, chosen_raw: str, rejected_raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Parse one raw row's chosen/rejected transcripts independently and require their
    prompt-message histories to be identical.

    Returns (record, None) on success or (None, exclusion) on failure. `exclusion` is
    `{"source_id", "source_index", "reason", "detail"}` with `reason` one of
    "chosen_malformed", "rejected_malformed", "history_mismatch" -- distinguishable so a
    manifest can report parse and history-mismatch exclusions separately, and never
    silently repaired or accepted.
    """
    source_id = f"hh-harmless-test-{source_index}"
    try:
        chosen_prompt, chosen_response = parse_transcript(chosen_raw)
    except HHRLHFError as exc:
        return None, {"source_id": source_id, "source_index": source_index, "reason": "chosen_malformed", "detail": str(exc)}
    try:
        rejected_prompt, rejected_response = parse_transcript(rejected_raw)
    except HHRLHFError as exc:
        return None, {"source_id": source_id, "source_index": source_index, "reason": "rejected_malformed", "detail": str(exc)}
    if chosen_prompt != rejected_prompt:
        return None, {
            "source_id": source_id, "source_index": source_index,
            "reason": "history_mismatch", "detail": "chosen/rejected prompt-message histories disagree",
        }

    record = {
        "source_id": source_id,
        "source_index": source_index,
        "prompt": chosen_prompt,
        "prompt_hash": conversation_hash(chosen_prompt),
        "chosen": chosen_response,
        "chosen_hash": response_hash(chosen_response),
        "rejected": rejected_response,
        "rejected_hash": response_hash(rejected_response),
    }
    return record, None


# Source file loading


def iter_raw_rows(path: Union[str, Path]) -> List[Dict[str, str]]:
    """Read every non-blank JSONL line from the gzip-compressed source file, in file
    order. Each row is `{"chosen": str, "rejected": str}`."""
    rows: List[Dict[str, str]] = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def hash_file(path: Union[str, Path]) -> Tuple[str, int]:
    """(sha256_hexdigest, size_bytes) of the file at `path`, read as raw bytes."""
    path = Path(path)
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def hash_canonical_content(path: Union[str, Path]) -> Tuple[str, int]:
    """(sha256_hexdigest, size_bytes) of the decompressed content of the gzip source
    file -- the "canonical decompressed representation" distinct from the compressed
    file's own hash."""
    digest = hashlib.sha256()
    size = 0
    with gzip.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def source_file_identity(path: Union[str, Path]) -> Dict[str, Any]:
    compressed_sha256, compressed_size = hash_file(path)
    canonical_sha256, canonical_size = hash_canonical_content(path)
    return {
        "path": SOURCE_FILE_PATH,
        "compressed_sha256": compressed_sha256,
        "compressed_size_bytes": compressed_size,
        "canonical_sha256": canonical_sha256,
        "canonical_size_bytes": canonical_size,
    }


# Parsing + dedup over the full raw source


def parse_all_rows(rows: Sequence[Mapping[str, str]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse every raw row. Returns (well_formed_records, parse_exclusions), both in
    source_index order."""
    records: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        record, exclusion = parse_row(i, row["chosen"], row["rejected"])
        if record is not None:
            records.append(record)
        else:
            exclusions.append(exclusion)
    return records, exclusions


def resolve_duplicates(records: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Group well-formed records by canonical prompt hash; keep the lowest
    `source_index` from each group, exclude the rest, and record the full exclusion
    lineage. A group larger than two is refused rather than silently resolved, mirroring
    `steering.splits.resolve_duplicates` -- the pinned source is known to have only
    verbatim pairs, and a bigger group means the source changed underneath the manifest.

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
            raise HHRLHFError(
                f"prompt hash {phash} has {len(group)} verbatim duplicates {ids}; "
                "the pinned source has only pairs -- refusing to guess a resolution"
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


def permutation_sort_key(seed: int, source_id: str) -> str:
    return hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).hexdigest()


# Manifest construction, hashing, save/load


def canonical_manifest_json(payload: Mapping[str, Any]) -> bytes:
    """The exact serialization the manifest hash is computed over: compact, key-sorted,
    ASCII-only, with `manifest_hash` itself excluded."""
    body = {k: v for k, v in payload.items() if k != "manifest_hash"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def compute_manifest_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_manifest_json(payload)).hexdigest()


def build_manifest(
    rows: Sequence[Mapping[str, str]],
    source_file: Mapping[str, Any],
    seed: int = SEED,
    dataset_revision: str = REVISION,
    expected_raw_count: int = RAW_RECORD_COUNT,
    expected_selected_count: int = SELECTED_RECORD_COUNT,
) -> Dict[str, Any]:
    """Build the frozen 200-record evaluation manifest from the raw (pre-filter,
    pre-selection) source rows, in file order.

    Raises `HHRLHFError` on any frozen-requirement violation -- including an unexpected
    raw count -- instead of silently adjusting counts or dropping records. Does not
    assume any particular exclusion/duplicate count in advance; those are derived here
    and simply recorded.
    """
    if not dataset_revision:
        raise HHRLHFError("dataset_revision must be a nonempty exact revision")
    if len(rows) != expected_raw_count:
        raise HHRLHFError(f"expected exactly {expected_raw_count} raw rows, got {len(rows)}")

    well_formed, parse_exclusions = parse_all_rows(rows)
    retained, duplicate_exclusions = resolve_duplicates(well_formed)

    eligible_count = len(retained)
    if eligible_count < expected_selected_count:
        raise HHRLHFError(f"only {eligible_count} eligible records, need at least {expected_selected_count}")

    entries = [
        {
            "source_id": r["source_id"],
            "source_index": r["source_index"],
            "prompt_hash": r["prompt_hash"],
            "chosen_hash": r["chosen_hash"],
            "rejected_hash": r["rejected_hash"],
        }
        for r in retained
    ]
    entries.sort(key=lambda e: (permutation_sort_key(seed, e["source_id"]), e["source_id"]))
    selected = entries[:expected_selected_count]
    for position, entry in enumerate(selected):
        entry["permuted_position"] = position
        entry["partition"] = PARTITION

    reason_counts: Dict[str, int] = {}
    for excl in parse_exclusions:
        reason_counts[excl["reason"]] = reason_counts.get(excl["reason"], 0) + 1

    # Canonical order in the manifest is independent of the caller's input/selection
    # order -- required for byte-identical manifests regardless of iteration order.
    selected_by_id = sorted(selected, key=lambda e: e["source_id"])
    parse_exclusions_sorted = sorted(parse_exclusions, key=lambda e: e["source_index"])
    duplicate_exclusions_sorted = sorted(duplicate_exclusions, key=lambda e: e["excluded_source_index"])

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "repository": REPOSITORY,
            "data_dir": DATA_DIR,
            "split": SPLIT,
            "revision": dataset_revision,
        },
        "source_file": dict(source_file),
        "permutation": {
            "seed": seed,
            "algorithm": PERMUTATION_ALGORITHM,
        },
        "raw_record_count": len(rows),
        "parse_excluded_count": len(parse_exclusions),
        "parse_excluded_by_reason": reason_counts,
        "duplicate_excluded_count": len(duplicate_exclusions),
        "eligible_record_count": eligible_count,
        "selected_record_count": len(selected),
        "parse_exclusions": parse_exclusions_sorted,
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
    Raises `HHRLHFError` if an existing manifest differs -- a manifest is never silently
    overwritten with different content.
    """
    path = Path(path)
    text = dumps_manifest(payload)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != text:
            raise HHRLHFError(f"manifest at {path} already exists with different content; refusing to overwrite")
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
        raise HHRLHFError(f"manifest hash mismatch at {path}: stored {stored!r}, recomputed {recomputed!r}")
    return payload


# Frozen identity + source-binding verification
#
# FROZEN_HH_RLHF_IDENTITY is filled in with the real manifest_hash and derived counts
# once `scripts/build_hh_rlhf_manifest.py` has built the manifest for real (Task 017).
# `load_manifest` already verifies the stored hash against a fresh recomputation of the
# file's own content; this pins that hash -- and everything upstream of it -- to the
# single value the frozen loader is allowed to trust, so a manifest that is internally
# self-consistent but was rebuilt from different content is still rejected.
FROZEN_HH_RLHF_IDENTITY: Dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "repository": REPOSITORY,
    "data_dir": DATA_DIR,
    "split": SPLIT,
    "revision": REVISION,
    "seed": SEED,
    "algorithm": PERMUTATION_ALGORITHM,
    "manifest_hash": "44f93ff71f229e11c1ac41db6a4bb3b07625ad53270594d0f207e14d76c6068b",
    "raw_record_count": RAW_RECORD_COUNT,
    "parse_excluded_count": EXPECTED_PARSE_EXCLUDED_COUNT,
    "duplicate_excluded_count": EXPECTED_DUPLICATE_EXCLUDED_COUNT,
    "eligible_record_count": EXPECTED_ELIGIBLE_RECORD_COUNT,
    "selected_record_count": SELECTED_RECORD_COUNT,
    "source_compressed_sha256": "ebfaaed21162a4de120ae85366075b34425e2a6303fac59b99555002f7016f03",
}


def validate_manifest_identity(
    manifest: Mapping[str, Any], expected: Mapping[str, Any] = FROZEN_HH_RLHF_IDENTITY,
) -> None:
    """Check a loaded manifest against a frozen identity before it selects any record.
    Raises `HHRLHFError` listing every field that disagrees."""
    mismatches: List[str] = []

    def check(label: str, actual: Any, want: Any) -> None:
        if actual != want:
            mismatches.append(f"{label}: expected {want!r}, got {actual!r}")

    check("schema_version", manifest.get("schema_version"), expected["schema_version"])
    dataset = manifest.get("dataset", {})
    for key in ("repository", "data_dir", "split", "revision"):
        check(f"dataset.{key}", dataset.get(key), expected[key])
    permutation = manifest.get("permutation", {})
    check("permutation.seed", permutation.get("seed"), expected["seed"])
    check("permutation.algorithm", permutation.get("algorithm"), expected["algorithm"])
    check("manifest_hash", manifest.get("manifest_hash"), expected["manifest_hash"])
    for key in ("raw_record_count", "parse_excluded_count", "duplicate_excluded_count",
                "eligible_record_count", "selected_record_count"):
        check(key, manifest.get(key), expected[key])
    check(
        "source_file.compressed_sha256",
        manifest.get("source_file", {}).get("compressed_sha256"),
        expected["source_compressed_sha256"],
    )

    if mismatches:
        raise HHRLHFError("manifest does not match the frozen identity: " + "; ".join(mismatches))


def validate_source_binding(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
    source_file: Mapping[str, Any],
) -> None:
    """Cross-check a freshly re-downloaded/re-parsed source against a validated
    manifest before any record is returned. Raises `HHRLHFError` on any mismatch --
    source drift, changed transcript content, changed selected membership, reordered
    records, or altered exclusions all fail loudly, never silently.
    """
    if source_file.get("compressed_sha256") != manifest["source_file"]["compressed_sha256"]:
        raise HHRLHFError("source file compressed content has drifted from the manifest")
    if source_file.get("canonical_sha256") != manifest["source_file"]["canonical_sha256"]:
        raise HHRLHFError("source file decompressed content has drifted from the manifest")

    if len(rows) != manifest["raw_record_count"]:
        raise HHRLHFError(f"expected {manifest['raw_record_count']} raw rows, got {len(rows)}")

    well_formed, parse_exclusions = parse_all_rows(rows)
    if len(parse_exclusions) != manifest["parse_excluded_count"]:
        raise HHRLHFError(
            f"expected {manifest['parse_excluded_count']} parse/history exclusions, "
            f"got {len(parse_exclusions)}"
        )
    reconstructed_parse_ids = {e["source_id"] for e in parse_exclusions}
    manifest_parse_ids = {e["source_id"] for e in manifest["parse_exclusions"]}
    if reconstructed_parse_ids != manifest_parse_ids:
        raise HHRLHFError("reconstructed parse/history exclusions do not match the manifest's lineage")
    for excl in parse_exclusions:
        recorded = next(e for e in manifest["parse_exclusions"] if e["source_id"] == excl["source_id"])
        if recorded["reason"] != excl["reason"]:
            raise HHRLHFError(f"exclusion reason for {excl['source_id']!r} has drifted from the manifest")

    retained, duplicate_exclusions = resolve_duplicates(well_formed)
    if len(duplicate_exclusions) != manifest["duplicate_excluded_count"]:
        raise HHRLHFError(
            f"expected {manifest['duplicate_excluded_count']} duplicate exclusions, "
            f"got {len(duplicate_exclusions)}"
        )
    reconstructed_dup_ids = {e["excluded_source_id"] for e in duplicate_exclusions}
    manifest_dup_ids = {e["excluded_source_id"] for e in manifest["duplicate_exclusions"]}
    if reconstructed_dup_ids != manifest_dup_ids:
        raise HHRLHFError("reconstructed duplicate exclusions do not match the manifest's lineage")

    if len(retained) != manifest["eligible_record_count"]:
        raise HHRLHFError(f"expected {manifest['eligible_record_count']} eligible records, got {len(retained)}")

    retained_by_id = {r["source_id"]: r for r in retained}
    manifest_records = manifest["records"]
    if len(manifest_records) != manifest["selected_record_count"]:
        raise HHRLHFError("manifest selected_record_count does not match its own records list")

    seen_positions = set()
    for entry in manifest_records:
        sid = entry["source_id"]
        raw = retained_by_id.get(sid)
        if raw is None:
            raise HHRLHFError(f"selected source_id {sid!r} is not among the reconstructed eligible records")
        if (raw["prompt_hash"] != entry["prompt_hash"]
                or raw["chosen_hash"] != entry["chosen_hash"]
                or raw["rejected_hash"] != entry["rejected_hash"]):
            raise HHRLHFError(f"content for selected {sid!r} has drifted from the manifest")
        if entry["permuted_position"] in seen_positions:
            raise HHRLHFError(f"permuted_position {entry['permuted_position']} appears more than once")
        seen_positions.add(entry["permuted_position"])

    if sorted(seen_positions) != list(range(manifest["selected_record_count"])):
        raise HHRLHFError("permuted positions are not exactly contiguous from 0")


# Frozen runtime loader


def _default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[2] / "manifests" / "hh_rlhf_harmless_test_v1.json"


def load_hh_rlhf_evaluation(
    manifest_path: Optional[Union[str, Path]] = None,
    source_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Load the frozen 200-record HH-RLHF evaluation panel exactly as fixed by the
    manifest.

    Downloads (or reuses a locally cached copy of) only the pinned source file,
    re-verifies its content identity, reconstructs every raw stable identity, and
    re-verifies all manifest exclusions and selected membership before returning
    anything -- membership and order are entirely determined by the manifest, never by
    a caller-supplied `n`/`seed`.

    Returns exactly `SELECTED_RECORD_COUNT` records in ascending `permuted_position`,
    each with: source_id, source_index, prompt (messages), prompt_hash, chosen,
    rejected, chosen_hash, rejected_hash, partition, permuted_position, manifest_hash.
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

    well_formed, _ = parse_all_rows(rows)
    by_id = {r["source_id"]: r for r in well_formed}

    selected = []
    for entry in manifest["records"]:
        raw = by_id[entry["source_id"]]
        selected.append({
            "source_id": raw["source_id"],
            "source_index": raw["source_index"],
            "prompt": raw["prompt"],
            "prompt_hash": raw["prompt_hash"],
            "chosen": raw["chosen"],
            "rejected": raw["rejected"],
            "chosen_hash": raw["chosen_hash"],
            "rejected_hash": raw["rejected_hash"],
            "partition": entry["partition"],
            "permuted_position": entry["permuted_position"],
            "manifest_hash": manifest["manifest_hash"],
        })
    selected.sort(key=lambda r: r["permuted_position"])
    return selected
