"""Guards on the frozen HarmfulQA partition manifest (protocol Section 5, as amended).

Every check here pins a specific frozen requirement from the handoff: exact raw/
retained/exclusion counts, a permutation that depends only on seed + stable ID (never
on input order), byte-identical canonical output, full duplicate-exclusion lineage, and
a manifest that is never silently overwritten.

Run with:
    pytest tests/test_splits.py -v
"""

from __future__ import annotations

import random

import pytest

from steering.splits import (
    PARTITION_BOUNDS,
    SplitError,
    build_manifest,
    load_manifest,
    normalize_prompt,
    prompt_hash,
    resolve_duplicates,
    save_manifest,
)

RETAINED_TOTAL = sum(hi - lo for _, lo, hi in PARTITION_BOUNDS)  # 1938
N_PAIRS = 22
RAW_TOTAL = RETAINED_TOTAL + N_PAIRS  # 1960


def _unique_records(n, prefix="rec", start=0):
    return [{"source_id": f"{prefix}-{i}", "source_index": i, "prompt": f"prompt text number {i}"}
            for i in range(start, start + n)]


def _raw_records_with_pairs(n_unique=RETAINED_TOTAL, n_pairs=N_PAIRS):
    """`n_unique` distinct prompts plus `n_pairs` verbatim duplicates of the first
    `n_pairs` of them, each duplicate given a strictly higher source_index than its
    original -- exactly the shape of the real HarmfulQA source."""
    records = _unique_records(n_unique)
    dupes = []
    for i in range(n_pairs):
        dupes.append({
            "source_id": f"rec-dup-{i}",
            "source_index": n_unique + i,
            "prompt": records[i]["prompt"],
        })
    return records + dupes


def _build(records, **overrides):
    kwargs = dict(
        seed=20260815, dataset_repo="test/repo", dataset_split="train", dataset_revision="deadbeef",
        expected_raw_count=len(records) if "expected_raw_count" not in overrides else overrides.pop("expected_raw_count"),
    )
    kwargs.update(overrides)
    return build_manifest(records, **kwargs)


# Normalization and hashing


def test_normalize_strips_only_leading_and_trailing_whitespace():
    assert normalize_prompt("  hello   world  \n") == "hello   world"


def test_normalize_folds_crlf_and_cr_to_lf():
    assert normalize_prompt("line one\r\nline two\rline three") == "line one\nline two\nline three"


def test_normalize_applies_nfc():
    # "e" + combining acute accent (decomposed) must match precomposed U+00E9 after NFC.
    decomposed = "caf" + "é"
    precomposed = "café"
    assert decomposed != precomposed  # confirms the test actually exercises NFC
    assert normalize_prompt(decomposed) == normalize_prompt(precomposed)


def test_normalize_does_not_lowercase_or_collapse_internal_whitespace():
    assert normalize_prompt("Hello    World") == "Hello    World"


def test_prompt_hash_is_sha256_hex_of_the_normalized_text():
    import hashlib

    text = "  Hello\r\nWorld  "
    expected = hashlib.sha256(normalize_prompt(text).encode("utf-8")).hexdigest()
    assert prompt_hash(text) == expected
    assert len(prompt_hash(text)) == 64


# Duplicate resolution


def test_resolve_duplicates_keeps_the_lower_source_index():
    records = _raw_records_with_pairs(n_unique=5, n_pairs=1)
    retained, exclusions = resolve_duplicates(records)

    assert len(retained) == 5
    assert len(exclusions) == 1
    excl = exclusions[0]
    assert excl["excluded_source_id"] == "rec-dup-0"
    assert excl["excluded_source_index"] == 5
    assert excl["retained_source_id"] == "rec-0"
    assert excl["retained_source_index"] == 0
    assert excl["prompt_hash"] == prompt_hash("prompt text number 0")
    retained_ids = {r["source_id"] for r in retained}
    assert "rec-dup-0" not in retained_ids
    assert "rec-0" in retained_ids


def test_resolve_duplicates_rejects_a_group_larger_than_two():
    records = _unique_records(3)
    records.append({"source_id": "rec-extra-1", "source_index": 99, "prompt": records[0]["prompt"]})
    records.append({"source_id": "rec-extra-2", "source_index": 100, "prompt": records[0]["prompt"]})
    with pytest.raises(SplitError):
        resolve_duplicates(records)


def test_resolve_duplicates_rejects_duplicate_source_ids_in_raw_input():
    records = _unique_records(3)
    records.append({"source_id": records[0]["source_id"], "source_index": 99, "prompt": "an unrelated prompt"})
    with pytest.raises(SplitError):
        resolve_duplicates(records)


# Manifest construction: counts, partitioning, exclusion lineage


def test_full_synthetic_source_retains_1938_and_records_22_exclusions():
    records = _raw_records_with_pairs()
    payload = _build(records)

    assert payload["raw_record_count"] == RAW_TOTAL
    assert payload["retained_record_count"] == RETAINED_TOTAL
    assert payload["excluded_record_count"] == N_PAIRS
    assert len(payload["duplicate_exclusions"]) == N_PAIRS
    assert len(payload["records"]) == RETAINED_TOTAL

    assert payload["partition_counts"] == {
        "construction": 1378, "calibration": 200, "final_evaluation": 300, "development": 60,
    }
    positions_by_partition = {}
    for entry in payload["records"]:
        positions_by_partition.setdefault(entry["partition"], []).append(entry["permuted_position"])
    for name, lo, hi in PARTITION_BOUNDS:
        assert sorted(positions_by_partition[name]) == list(range(lo, hi))


def test_duplicate_exclusion_lineage_is_complete_and_consistent():
    records = _raw_records_with_pairs()
    payload = _build(records)

    excluded_ids = {e["excluded_source_id"] for e in payload["duplicate_exclusions"]}
    retained_ids = {e["source_id"] for e in payload["records"]}
    assert excluded_ids.isdisjoint(retained_ids)

    for excl in payload["duplicate_exclusions"]:
        assert excl["retained_source_id"] in retained_ids
        # The retained twin must actually carry the same normalized prompt hash.
        retained_entry = next(e for e in payload["records"] if e["source_id"] == excl["retained_source_id"])
        assert retained_entry["prompt_hash"] == excl["prompt_hash"]
        assert excl["excluded_source_id"] not in {e["source_id"] for e in payload["records"]}
        # Retained index is always lower than the excluded one, by construction.
        assert excl["retained_source_index"] < excl["excluded_source_index"]


def test_wrong_raw_record_count_is_rejected():
    records = _raw_records_with_pairs()[:-1]
    with pytest.raises(SplitError):
        _build(records, expected_raw_count=RAW_TOTAL)


def test_unexpected_duplicate_pair_count_is_rejected():
    """One fewer duplicate pair than the frozen source documents must stop the build,
    not silently accept a different exclusion count."""
    records = _raw_records_with_pairs(n_unique=RETAINED_TOTAL + 1, n_pairs=N_PAIRS - 1)
    with pytest.raises(SplitError):
        _build(records, expected_raw_count=RAW_TOTAL)


def test_a_triple_duplicate_group_is_rejected_even_at_the_right_raw_count():
    """A raw count that matches expectations must not paper over a bad group shape."""
    records = _raw_records_with_pairs()  # RAW_TOTAL rows, 22 clean pairs
    records.append({
        "source_id": "rec-triple-extra",
        "source_index": max(r["source_index"] for r in records) + 1,
        "prompt": records[0]["prompt"],  # a third copy of an already-paired prompt
    })
    with pytest.raises(SplitError):
        _build(records)  # expected_raw_count defaults to len(records): the count "looks right"


def test_empty_revision_is_rejected():
    records = _raw_records_with_pairs()
    with pytest.raises(SplitError):
        build_manifest(
            records, seed=20260815,
            dataset_repo="test/repo", dataset_split="train", dataset_revision="",
            expected_raw_count=RAW_TOTAL,
        )


def test_duplicate_source_ids_in_raw_input_are_rejected_by_build_manifest():
    records = _raw_records_with_pairs()
    records.append({"source_id": records[0]["source_id"], "source_index": 99999, "prompt": "yet another prompt"})
    with pytest.raises(SplitError):
        _build(records, expected_raw_count=len(records))


# Order-independence and reproducibility


def test_permuting_input_order_produces_a_byte_identical_manifest():
    records = _raw_records_with_pairs()
    shuffled = list(records)
    random.Random(0).shuffle(shuffled)

    a = _build(records)
    b = _build(shuffled)

    assert a == b
    assert a["manifest_hash"] == b["manifest_hash"]


def test_same_inputs_and_seed_reproduce_the_same_hash():
    records = _raw_records_with_pairs()
    a = _build(records)
    b = _build(records)
    assert a["manifest_hash"] == b["manifest_hash"]


def test_changing_the_seed_changes_the_hash_and_assignment():
    records = _raw_records_with_pairs()
    a = _build(records, seed=20260815)
    b = _build(records, seed=1)

    assert a["manifest_hash"] != b["manifest_hash"]
    positions_a = {e["source_id"]: e["permuted_position"] for e in a["records"]}
    positions_b = {e["source_id"]: e["permuted_position"] for e in b["records"]}
    assert positions_a != positions_b


# Save / load / immutability


def test_save_and_load_round_trip(tmp_path):
    payload = _build(_raw_records_with_pairs())
    path = tmp_path / "manifest.json"
    written = save_manifest(payload, path)
    assert written is True

    loaded = load_manifest(path)
    assert loaded == payload


def test_rebuilding_an_identical_manifest_does_not_rewrite(tmp_path):
    payload = _build(_raw_records_with_pairs())
    path = tmp_path / "manifest.json"
    save_manifest(payload, path)
    mtime_before = path.stat().st_mtime_ns

    written_again = save_manifest(payload, path)
    assert written_again is False
    assert path.stat().st_mtime_ns == mtime_before


def test_an_existing_different_manifest_is_not_overwritten(tmp_path):
    original = _build(_raw_records_with_pairs())
    path = tmp_path / "manifest.json"
    save_manifest(original, path)
    on_disk_before = path.read_text(encoding="utf-8")

    different = _build(_raw_records_with_pairs(), seed=1)
    with pytest.raises(SplitError):
        save_manifest(different, path)

    assert path.read_text(encoding="utf-8") == on_disk_before


def test_tampering_causes_verification_to_fail(tmp_path):
    payload = _build(_raw_records_with_pairs())
    path = tmp_path / "manifest.json"
    save_manifest(payload, path)

    tampered = path.read_text(encoding="utf-8").replace(
        payload["records"][0]["prompt_hash"], "0" * 64,
    )
    path.write_text(tampered, encoding="utf-8")

    with pytest.raises(SplitError):
        load_manifest(path)
