"""Guards on the frozen AdvBench OOD evaluation manifest (Task 018).

AdvBench is a flat (prompt, target) table with no transcript structure -- these tests
pin canonicalization (NFC + CRLF/CR to LF, no stripping), rejection of non-string/empty
prompts, duplicate resolution, the shared HH-RLHF-style permutation, and that the
manifest never stores raw prompt text.

Run with:
    pytest tests/test_advbench.py tests/test_data.py -v
"""

from __future__ import annotations

import copy

import pytest

import steering.advbench as advbench


def _fake_source_file() -> dict:
    return {"path": "data/train-00000-of-00001.parquet", "sha256": "a" * 64, "size_bytes": 111}


def _synthetic_rows() -> list:
    """10 raw rows exercising every exclusion reason:
    0-4, 8, 9: valid, unique prompts (7 unique prompts)
    5:         valid, duplicate of row 0's prompt (should be excluded, row 0 kept)
    6:         invalid -- empty string prompt
    7:         invalid -- non-string prompt
    -> 2 invalid exclusions, 1 duplicate exclusion, 7 eligible records.
    """
    rows = [{"prompt": f"prompt number {i}", "target": f"target {i}"} for i in range(5)]
    rows.append({"prompt": "prompt number 0", "target": "a different target for the same prompt"})  # dup of row 0
    rows.append({"prompt": "", "target": "t6"})  # invalid: empty
    rows.append({"prompt": None, "target": "t7"})  # invalid: non-string
    rows.append({"prompt": "prompt number 8", "target": "t8"})
    rows.append({"prompt": "prompt number 9", "target": "t9"})
    assert len(rows) == 10
    return rows


def _build_synthetic_manifest(**overrides):
    rows = overrides.pop("rows", None) or _synthetic_rows()
    source_file = overrides.pop("source_file", None) or _fake_source_file()
    kwargs = dict(
        seed=20260815, dataset_revision="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        expected_raw_count=len(rows), expected_selected_count=4,
    )
    kwargs.update(overrides)
    return advbench.build_manifest(rows, source_file=source_file, **kwargs), rows, source_file


# Canonicalization


def test_normalize_folds_crlf_and_cr_to_lf():
    assert advbench.normalize_prompt("line one\r\nline two\rline three") == "line one\nline two\nline three"


def test_normalize_applies_nfc():
    decomposed = "cafe" + "́"  # e + combining acute accent
    precomposed = "café"
    assert decomposed != precomposed
    assert advbench.normalize_prompt(decomposed) == advbench.normalize_prompt(precomposed)


def test_normalize_does_not_strip_or_collapse_whitespace():
    assert advbench.normalize_prompt("  hello   world  \n") == "  hello   world  \n"


def test_prompt_hash_is_sha256_hex_of_the_normalized_text():
    import hashlib

    text = "  Hello\r\nWorld  "
    expected = hashlib.sha256(advbench.normalize_prompt(text).encode("utf-8")).hexdigest()
    assert advbench.prompt_hash(text) == expected
    assert len(advbench.prompt_hash(text)) == 64


# Validity: reject non-string or empty prompts, but not meaningful whitespace


@pytest.mark.parametrize("value", ["", None, 123, 1.5, [], {}, ["prompt"]])
def test_invalid_prompts_are_rejected(value):
    assert advbench.is_valid_prompt(value) is False


def test_whitespace_only_prompt_is_not_treated_as_empty():
    """The frozen contract says never strip meaningful whitespace -- a prompt that is
    entirely whitespace still normalizes to a nonzero-length string and is valid."""
    assert advbench.is_valid_prompt("   ") is True


def test_valid_prompt_is_accepted():
    assert advbench.is_valid_prompt("a real prompt") is True


# Row-level parsing: schema not assumed, invalid rows distinguished from valid ones


def test_parse_all_rows_separates_valid_and_invalid_with_stable_ids():
    rows = _synthetic_rows()
    valid, invalid = advbench.parse_all_rows(rows)

    assert {r["source_index"] for r in valid} == {0, 1, 2, 3, 4, 5, 8, 9}
    assert {e["source_index"] for e in invalid} == {6, 7}
    for r in valid:
        assert r["source_id"] == f"advbench-{r['source_index']}"
    for e in invalid:
        assert e["source_id"] == f"advbench-{e['source_index']}"
        assert e["reason"] == "invalid_prompt"


def test_parse_all_rows_handles_a_row_missing_the_prompt_key():
    """The schema is not assumed -- a row shaped unexpectedly is excluded, not crashed on."""
    rows = [{"target": "no prompt key at all"}]
    valid, invalid = advbench.parse_all_rows(rows)
    assert valid == []
    assert invalid == [{"source_id": "advbench-0", "source_index": 0, "reason": "invalid_prompt"}]


def test_parse_all_rows_ignores_the_target_field():
    rows = [{"prompt": "only this matters", "target": "never read"}]
    valid, _ = advbench.parse_all_rows(rows)
    assert valid[0]["prompt"] == "only this matters"
    assert "target" not in valid[0]


# Duplicate resolution


def test_resolve_duplicates_keeps_the_lower_source_index_and_records_lineage():
    rows = _synthetic_rows()
    valid, _ = advbench.parse_all_rows(rows)
    retained, exclusions = advbench.resolve_duplicates(valid)

    assert len(exclusions) == 1
    excl = exclusions[0]
    assert excl["excluded_source_id"] == "advbench-5"
    assert excl["retained_source_id"] == "advbench-0"
    assert excl["excluded_source_index"] == 5
    assert excl["retained_source_index"] == 0

    retained_ids = {r["source_id"] for r in retained}
    assert "advbench-5" not in retained_ids
    assert "advbench-0" in retained_ids
    assert len(retained) == 7


def test_resolve_duplicates_rejects_a_group_larger_than_two():
    rows = [{"prompt": "same prompt text", "target": f"t{i}"} for i in range(3)]
    valid, _ = advbench.parse_all_rows(rows)
    with pytest.raises(advbench.AdvBenchError):
        advbench.resolve_duplicates(valid)


# Selection: deterministic, order-independent, shared HH-RLHF-style permutation


def test_permutation_algorithm_label_matches_hh_rlhf():
    import steering.hh_rlhf as hh_rlhf

    assert advbench.PERMUTATION_ALGORITHM == hh_rlhf.PERMUTATION_ALGORITHM


def test_selection_sort_order_is_independent_of_input_list_order():
    ids = [f"advbench-{i}" for i in range(50)]
    forward = sorted(ids, key=lambda sid: (advbench.permutation_sort_key(advbench.SEED, sid), sid))
    reversed_input = sorted(reversed(ids), key=lambda sid: (advbench.permutation_sort_key(advbench.SEED, sid), sid))
    assert forward == reversed_input


def test_same_rows_and_seed_reproduce_the_same_manifest_hash():
    payload_a, rows, sf = _build_synthetic_manifest()
    payload_b, _, _ = _build_synthetic_manifest(rows=rows, source_file=sf)
    assert payload_a["manifest_hash"] == payload_b["manifest_hash"]


def test_changing_the_seed_changes_hash_and_selected_membership():
    payload_a, rows, sf = _build_synthetic_manifest(seed=1)
    payload_b, _, _ = _build_synthetic_manifest(rows=rows, source_file=sf, seed=2)
    assert payload_a["manifest_hash"] != payload_b["manifest_hash"]
    ids_a = [e["source_id"] for e in payload_a["records"]]
    ids_b = [e["source_id"] for e in payload_b["records"]]
    assert ids_a != ids_b


# Manifest construction: exact counts, exactly N selected at 0..N-1, no raw text


def test_manifest_records_exact_counts():
    payload, rows, sf = _build_synthetic_manifest()
    assert payload["raw_record_count"] == 10
    assert payload["invalid_excluded_count"] == 2
    assert payload["duplicate_excluded_count"] == 1
    assert payload["eligible_record_count"] == 7
    assert payload["selected_record_count"] == 4
    assert payload["permutation"]["seed"] == 20260815
    assert payload["permutation"]["algorithm"] == advbench.PERMUTATION_ALGORITHM
    assert payload["source_file"] == sf


def test_manifest_never_stores_raw_prompt_text():
    payload, rows, _ = _build_synthetic_manifest()
    serialized = advbench.dumps_manifest(payload)
    for row in rows:
        prompt = row.get("prompt")
        if isinstance(prompt, str) and prompt:
            assert prompt not in serialized
    for entry in payload["records"]:
        assert set(entry.keys()) == {"source_id", "source_index", "prompt_hash", "permuted_position", "partition"}


def test_exactly_n_selected_records_at_contiguous_positions():
    payload, _, _ = _build_synthetic_manifest()
    positions = sorted(e["permuted_position"] for e in payload["records"])
    assert positions == [0, 1, 2, 3]
    for entry in payload["records"]:
        assert entry["partition"] == "evaluation"


def test_wrong_raw_record_count_is_rejected():
    rows = _synthetic_rows()
    with pytest.raises(advbench.AdvBenchError):
        advbench.build_manifest(rows, source_file=_fake_source_file(), expected_raw_count=len(rows) - 1)


def test_too_few_eligible_records_is_rejected():
    rows = _synthetic_rows()[:2]
    with pytest.raises(advbench.AdvBenchError):
        advbench.build_manifest(rows, source_file=_fake_source_file(), expected_raw_count=2, expected_selected_count=4)


# Save / load / self-hash / frozen identity


def test_manifest_self_hash_is_verified_on_load(tmp_path):
    payload, _, _ = _build_synthetic_manifest()
    path = tmp_path / "manifest.json"
    advbench.save_manifest(payload, path)
    loaded = advbench.load_manifest(path)
    assert loaded == payload
    assert loaded["manifest_hash"] == advbench.compute_manifest_hash(loaded)


def test_an_existing_different_manifest_is_never_overwritten(tmp_path):
    payload_a, rows, sf = _build_synthetic_manifest(seed=1)
    path = tmp_path / "manifest.json"
    advbench.save_manifest(payload_a, path)
    on_disk = path.read_text(encoding="utf-8")

    payload_b, _, _ = _build_synthetic_manifest(rows=rows, source_file=sf, seed=2)
    with pytest.raises(advbench.AdvBenchError):
        advbench.save_manifest(payload_b, path)
    assert path.read_text(encoding="utf-8") == on_disk


def test_tampering_with_a_saved_manifest_fails_verification(tmp_path):
    payload, _, _ = _build_synthetic_manifest()
    path = tmp_path / "manifest.json"
    advbench.save_manifest(payload, path)
    tampered = path.read_text(encoding="utf-8").replace(payload["records"][0]["prompt_hash"], "0" * 64)
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(advbench.AdvBenchError):
        advbench.load_manifest(path)


def _synthetic_expected_identity(payload, sf):
    return {
        "schema_version": advbench.SCHEMA_VERSION,
        "repository": advbench.REPOSITORY,
        "config": advbench.CONFIG,
        "split": advbench.SPLIT,
        "revision": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        "seed": 20260815,
        "algorithm": advbench.PERMUTATION_ALGORITHM,
        "manifest_hash": payload["manifest_hash"],
        "raw_record_count": payload["raw_record_count"],
        "invalid_excluded_count": payload["invalid_excluded_count"],
        "duplicate_excluded_count": payload["duplicate_excluded_count"],
        "eligible_record_count": payload["eligible_record_count"],
        "selected_record_count": payload["selected_record_count"],
        "source_sha256": sf["sha256"],
    }


def test_validate_manifest_identity_accepts_a_matching_manifest():
    payload, _, sf = _build_synthetic_manifest()
    advbench.validate_manifest_identity(payload, expected=_synthetic_expected_identity(payload, sf))  # no raise


@pytest.mark.parametrize("field,new_value", [
    ("manifest_hash", "0" * 64),
    ("revision", "some-other-revision"),
    ("seed", 999),
    ("raw_record_count", 1),
    ("eligible_record_count", 1),
    ("selected_record_count", 1),
    ("source_sha256", "f" * 64),
])
def test_validate_manifest_identity_rejects_each_mismatched_field(field, new_value):
    payload, _, sf = _build_synthetic_manifest()
    expected = _synthetic_expected_identity(payload, sf)
    expected[field] = new_value
    with pytest.raises(advbench.AdvBenchError):
        advbench.validate_manifest_identity(payload, expected=expected)


def test_the_frozen_real_manifest_matches_its_own_hardcoded_identity():
    """The manifest actually built by scripts/build_advbench_manifest.py against the
    pinned source must match FROZEN_ADVBENCH_IDENTITY -- this is what the runtime
    loader trusts."""
    manifest = advbench.load_manifest(advbench._default_manifest_path())
    advbench.validate_manifest_identity(manifest)  # must not raise
    assert manifest["raw_record_count"] == advbench.RAW_RECORD_COUNT
    assert manifest["selected_record_count"] == advbench.SELECTED_RECORD_COUNT == 200
    assert manifest["dataset"]["revision"] == advbench.REVISION


# Source-binding validation


def test_validate_source_binding_accepts_matching_rows():
    payload, rows, sf = _build_synthetic_manifest()
    advbench.validate_source_binding(payload, rows, sf)  # must not raise


def test_validate_source_binding_rejects_a_changed_source_file_hash():
    payload, rows, sf = _build_synthetic_manifest()
    drifted_sf = dict(sf, sha256="f" * 64)
    with pytest.raises(advbench.AdvBenchError):
        advbench.validate_source_binding(payload, rows, drifted_sf)


def test_validate_source_binding_rejects_drifted_row_content():
    payload, rows, sf = _build_synthetic_manifest()
    drifted_rows = list(rows)
    drifted_rows[0] = dict(drifted_rows[0], prompt="a completely different prompt")
    with pytest.raises(advbench.AdvBenchError):
        advbench.validate_source_binding(payload, drifted_rows, sf)


def test_validate_source_binding_rejects_noncontiguous_positions():
    payload, rows, sf = _build_synthetic_manifest()
    tampered = copy.deepcopy(payload)
    tampered["records"][0]["permuted_position"] = tampered["records"][1]["permuted_position"]
    with pytest.raises(advbench.AdvBenchError):
        advbench.validate_source_binding(tampered, rows, sf)


def test_validate_source_binding_rejects_a_selected_id_missing_from_eligible_rows():
    payload, rows, sf = _build_synthetic_manifest()
    tampered = copy.deepcopy(payload)
    tampered["records"][0]["source_id"] = "advbench-9999"
    with pytest.raises(advbench.AdvBenchError):
        advbench.validate_source_binding(tampered, rows, sf)


def test_validate_source_binding_rejects_a_changed_exclusion_count():
    payload, rows, sf = _build_synthetic_manifest()
    tampered = copy.deepcopy(payload)
    tampered["invalid_excluded_count"] = tampered["invalid_excluded_count"] + 1
    with pytest.raises(advbench.AdvBenchError):
        advbench.validate_source_binding(tampered, rows, sf)


# Runtime loader (frozen 200-prompt panel, exercised against a synthetic small panel)


def _write_parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist([{"prompt": r["prompt"], "target": r["target"]} for r in rows])
    pq.write_table(table, path)


def _build_and_save_real_file_manifest(tmp_path, rows, **overrides):
    pq_path = tmp_path / "advbench.parquet"
    _write_parquet(pq_path, [{"prompt": r.get("prompt"), "target": r.get("target", "")} for r in rows])
    source_file = advbench.source_file_identity(pq_path)
    kwargs = dict(expected_raw_count=len(rows), expected_selected_count=4)
    kwargs.update(overrides)
    payload = advbench.build_manifest(rows, source_file=source_file, **kwargs)
    manifest_path = tmp_path / "manifest.json"
    advbench.save_manifest(payload, manifest_path)
    return manifest_path, pq_path, payload


def test_load_advbench_evaluation_returns_full_schema_in_position_order(tmp_path, monkeypatch):
    rows = [{"prompt": f"prompt number {i}", "target": f"target {i}"} for i in range(10)]
    manifest_path, pq_path, payload = _build_and_save_real_file_manifest(tmp_path, rows)
    monkeypatch.setattr(advbench, "validate_manifest_identity", lambda *a, **k: None)

    records = advbench.load_advbench_evaluation(manifest_path=manifest_path, source_path=pq_path)

    assert len(records) == 4
    assert [r["permuted_position"] for r in records] == [0, 1, 2, 3]
    for r in records:
        assert r["partition"] == "evaluation"
        assert r["manifest_hash"] == payload["manifest_hash"]
        for field in ("source_id", "source_index", "prompt", "prompt_hash", "partition",
                      "permuted_position", "manifest_hash"):
            assert field in r


def test_load_advbench_evaluation_rejects_a_manifest_that_fails_the_frozen_identity_check(tmp_path):
    rows = [{"prompt": f"prompt number {i}", "target": f"target {i}"} for i in range(10)]
    manifest_path, pq_path, _ = _build_and_save_real_file_manifest(tmp_path, rows)
    with pytest.raises(advbench.AdvBenchError):
        advbench.load_advbench_evaluation(manifest_path=manifest_path, source_path=pq_path)


def test_load_advbench_evaluation_rejects_a_source_file_that_drifted_after_the_manifest_was_built(tmp_path, monkeypatch):
    rows = [{"prompt": f"prompt number {i}", "target": f"target {i}"} for i in range(10)]
    manifest_path, pq_path, _ = _build_and_save_real_file_manifest(tmp_path, rows)
    monkeypatch.setattr(advbench, "validate_manifest_identity", lambda *a, **k: None)

    drifted_rows = list(rows)
    drifted_rows[0] = {"prompt": "a completely different prompt", "target": "t"}
    _write_parquet(pq_path, drifted_rows)

    with pytest.raises(advbench.AdvBenchError):
        advbench.load_advbench_evaluation(manifest_path=manifest_path, source_path=pq_path)


def test_source_file_identity_hashes_the_actual_bytes(tmp_path):
    pq_path = tmp_path / "sample.parquet"
    _write_parquet(pq_path, [{"prompt": "x", "target": "y"}])
    identity = advbench.source_file_identity(pq_path)
    assert len(identity["sha256"]) == 64
    assert identity["size_bytes"] == pq_path.stat().st_size
