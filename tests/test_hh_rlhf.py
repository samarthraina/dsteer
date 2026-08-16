"""Guards on the corrected HH-RLHF transcript parser and the frozen evaluation manifest
(Task 017).

The bug this pins: the original parser (`steering.data._split_hh_conversation`) split
transcripts on every blank line and silently dropped any paragraph that did not start
with a fresh "Human:"/"Assistant:" marker -- corrupting any multi-paragraph turn. The
corrected parser here (`steering.hh_rlhf`) finds turn boundaries only at a blank line
immediately followed by a role marker, so internal blank lines within a turn survive.

Run with:
    pytest tests/test_hh_rlhf.py tests/test_data.py -v
"""

from __future__ import annotations

import copy
import gzip
import json

import pytest

import steering.hh_rlhf as hh_rlhf


def _mk_row(human: str, chosen_reply: str, rejected_reply: str = None, human_rejected: str = None) -> dict:
    """One synthetic raw HH-RLHF row: {"chosen", "rejected"} transcripts."""
    rejected_reply = rejected_reply if rejected_reply is not None else chosen_reply + " (rejected variant)"
    human_rejected = human_rejected if human_rejected is not None else human
    return {
        "chosen": f"\n\nHuman: {human}\n\nAssistant: {chosen_reply}",
        "rejected": f"\n\nHuman: {human_rejected}\n\nAssistant: {rejected_reply}",
    }


def _fake_source_file() -> dict:
    return {
        "path": "harmless-base/test.jsonl.gz",
        "compressed_sha256": "a" * 64,
        "compressed_size_bytes": 111,
        "canonical_sha256": "b" * 64,
        "canonical_size_bytes": 222,
    }


def _synthetic_rows() -> list:
    """10 raw rows exercising every exclusion reason:
    0-4, 9: well-formed, unique prompts (6 unique prompts)
    5:      well-formed, duplicate of row 0's prompt (should be excluded, row 0 kept)
    6:      malformed chosen (does not end with Assistant)
    7:      malformed rejected (does not strictly alternate)
    8:      chosen/rejected prompt-history mismatch
    -> 3 parse exclusions, 1 duplicate exclusion, 6 eligible records.
    """
    rows = [_mk_row(f"question {i}", f"answer {i}") for i in range(5)]
    rows.append(_mk_row("question 0", "a different answer to the same question"))  # dup of row 0
    rows.append({
        "chosen": "\n\nHuman: bad\n\nAssistant: partial\n\nHuman: trailing human turn",
        "rejected": "\n\nHuman: bad\n\nAssistant: fine",
    })
    rows.append({
        "chosen": "\n\nHuman: q\n\nAssistant: a",
        "rejected": "\n\nHuman: q\n\nHuman: q again\n\nAssistant: a",
    })
    rows.append(_mk_row("question eight chosen", "answer 8", human_rejected="question eight rejected"))
    rows.append(_mk_row("question 9", "answer 9"))
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
    return hh_rlhf.build_manifest(rows, source_file=source_file, **kwargs), rows, source_file


# Transcript parsing


def test_parses_a_single_human_assistant_turn():
    prompt, response = hh_rlhf.parse_transcript("\n\nHuman: hi there\n\nAssistant: hello!")
    assert prompt == [{"role": "user", "content": "hi there"}]
    assert response == "hello!"


def test_parses_multiple_alternating_turns():
    text = "\n\nHuman: q1\n\nAssistant: a1\n\nHuman: q2\n\nAssistant: a2"
    prompt, response = hh_rlhf.parse_transcript(text)
    assert prompt == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    assert response == "a2"


def test_preserves_internal_blank_lines_within_a_turn():
    text = (
        "\n\nHuman: pranks?\n\nAssistant: first paragraph.\n\nsecond paragraph, previously dropped."
        "\n\nHuman: ok\n\nAssistant: done"
    )
    prompt, response = hh_rlhf.parse_transcript(text)
    assistant_turn = prompt[1]
    assert assistant_turn["content"] == "first paragraph.\n\nsecond paragraph, previously dropped."


def test_excludes_only_the_final_assistant_response_from_the_prompt():
    text = "\n\nHuman: q1\n\nAssistant: a1\n\nHuman: q2\n\nAssistant: a2 (final)"
    prompt, response = hh_rlhf.parse_transcript(text)
    assert response == "a2 (final)"
    assert all(m["content"] != "a2 (final)" for m in prompt)
    assert len(prompt) == 3


def test_crlf_and_cr_are_normalized_to_lf_before_parsing():
    lf_text = "\n\nHuman: hi\n\nAssistant: hello"
    crlf_text = "\r\n\r\nHuman: hi\r\n\r\nAssistant: hello"
    cr_text = "\r\rHuman: hi\r\rAssistant: hello"
    assert hh_rlhf.parse_transcript(crlf_text) == hh_rlhf.parse_transcript(lf_text)
    assert hh_rlhf.parse_transcript(cr_text) == hh_rlhf.parse_transcript(lf_text)


def test_nfc_normalization_makes_decomposed_and_precomposed_unicode_equal():
    decomposed = "cafe" + "́"  # e + combining acute accent
    precomposed = "café"
    assert decomposed != precomposed
    text_a = f"\n\nHuman: {decomposed}\n\nAssistant: ok"
    text_b = f"\n\nHuman: {precomposed}\n\nAssistant: ok"
    assert hh_rlhf.parse_transcript(text_a) == hh_rlhf.parse_transcript(text_b)


@pytest.mark.parametrize("bad_text", [
    "",
    "   ",
    "just plain text with no role markers at all",
    "\n\nHuman: q\n\nAssistant: a\n\nHuman: trailing",
    "\n\nHuman: q\n\nHuman: q again\n\nAssistant: a",
    "\n\nAssistant: a reply with no human turn first",
])
def test_malformed_transcripts_are_rejected_not_repaired(bad_text):
    with pytest.raises(hh_rlhf.HHRLHFError):
        hh_rlhf.parse_transcript(bad_text)


# Row-level parsing: chosen/rejected agreement


def test_chosen_and_rejected_full_histories_must_agree():
    row = _mk_row("shared question", "chosen answer")
    record, exclusion = hh_rlhf.parse_row(0, row["chosen"], row["rejected"])
    assert exclusion is None
    assert record["prompt"] == [{"role": "user", "content": "shared question"}]


def test_history_mismatch_is_excluded_with_a_distinguishable_reason():
    row = _mk_row("chosen question", "answer", human_rejected="different rejected question")
    record, exclusion = hh_rlhf.parse_row(3, row["chosen"], row["rejected"])
    assert record is None
    assert exclusion["reason"] == "history_mismatch"
    assert exclusion["source_id"] == "hh-harmless-test-3"
    assert exclusion["source_index"] == 3


def test_malformed_chosen_and_rejected_are_distinguished():
    _, chosen_bad = hh_rlhf.parse_row(0, "not a transcript", "\n\nHuman: q\n\nAssistant: a")
    assert chosen_bad["reason"] == "chosen_malformed"
    _, rejected_bad = hh_rlhf.parse_row(0, "\n\nHuman: q\n\nAssistant: a", "not a transcript")
    assert rejected_bad["reason"] == "rejected_malformed"


# Stable identities and canonical hashing


def test_source_ids_are_assigned_from_row_position_before_any_filtering():
    rows = _synthetic_rows()
    well_formed, exclusions = hh_rlhf.parse_all_rows(rows)
    for rec in well_formed:
        assert rec["source_id"] == f"hh-harmless-test-{rec['source_index']}"
    for excl in exclusions:
        assert excl["source_id"] == f"hh-harmless-test-{excl['source_index']}"
    assert {r["source_index"] for r in well_formed} == {0, 1, 2, 3, 4, 5, 9}
    assert {e["source_index"] for e in exclusions} == {6, 7, 8}


def test_conversation_hash_covers_the_full_message_list_not_just_the_first_turn():
    short = [{"role": "user", "content": "q1"}]
    long_ = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}, {"role": "user", "content": "q2"}]
    assert hh_rlhf.conversation_hash(short) != hh_rlhf.conversation_hash(long_)
    assert len(hh_rlhf.conversation_hash(short)) == 64


def test_conversation_hash_is_order_sensitive_and_deterministic():
    a = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    b = [{"role": "assistant", "content": "y"}, {"role": "user", "content": "x"}]
    assert hh_rlhf.conversation_hash(a) != hh_rlhf.conversation_hash(b)
    assert hh_rlhf.conversation_hash(a) == hh_rlhf.conversation_hash(list(a))


def test_response_hash_differs_for_different_text_and_is_length_64():
    h1 = hh_rlhf.response_hash("answer one")
    h2 = hh_rlhf.response_hash("answer two")
    assert h1 != h2
    assert len(h1) == 64


def test_parse_row_records_prompt_chosen_rejected_hashes():
    row = _mk_row("q", "chosen text", rejected_reply="rejected text")
    record, _ = hh_rlhf.parse_row(0, row["chosen"], row["rejected"])
    assert record["chosen_hash"] == hh_rlhf.response_hash("chosen text")
    assert record["rejected_hash"] == hh_rlhf.response_hash("rejected text")
    assert record["prompt_hash"] == hh_rlhf.conversation_hash(record["prompt"])


# Duplicate resolution


def test_resolve_duplicates_keeps_the_lower_source_index_and_records_lineage():
    rows = _synthetic_rows()
    well_formed, _ = hh_rlhf.parse_all_rows(rows)
    retained, exclusions = hh_rlhf.resolve_duplicates(well_formed)

    assert len(exclusions) == 1
    excl = exclusions[0]
    assert excl["excluded_source_id"] == "hh-harmless-test-5"
    assert excl["retained_source_id"] == "hh-harmless-test-0"
    assert excl["excluded_source_index"] == 5
    assert excl["retained_source_index"] == 0

    retained_ids = {r["source_id"] for r in retained}
    assert "hh-harmless-test-5" not in retained_ids
    assert "hh-harmless-test-0" in retained_ids
    assert len(retained) == 6


def test_resolve_duplicates_rejects_a_group_larger_than_two():
    rows = [_mk_row("same question", f"answer {i}") for i in range(3)]
    well_formed, _ = hh_rlhf.parse_all_rows(rows)
    with pytest.raises(hh_rlhf.HHRLHFError):
        hh_rlhf.resolve_duplicates(well_formed)


# Selection: deterministic, order-independent, no library shuffle


def test_selection_sort_order_is_independent_of_input_list_order():
    ids = [f"hh-harmless-test-{i}" for i in range(50)]
    forward = sorted(ids, key=lambda sid: (hh_rlhf.permutation_sort_key(hh_rlhf.SEED, sid), sid))
    reversed_input = sorted(reversed(ids), key=lambda sid: (hh_rlhf.permutation_sort_key(hh_rlhf.SEED, sid), sid))
    assert forward == reversed_input


def test_same_rows_and_seed_reproduce_the_same_manifest_hash():
    payload_a, rows, sf = _build_synthetic_manifest()
    payload_b, _, _ = _build_synthetic_manifest(rows=rows, source_file=sf)
    assert payload_a["manifest_hash"] == payload_b["manifest_hash"]


def test_changing_the_seed_changes_hash_and_selected_membership():
    payload_a, rows, sf = _build_synthetic_manifest(seed=1)
    payload_b, _, _ = _build_synthetic_manifest(rows=rows, source_file=sf, seed=2)
    assert payload_a["manifest_hash"] != payload_b["manifest_hash"]
    ids_a = {e["source_id"] for e in payload_a["records"]}
    ids_b = {e["source_id"] for e in payload_b["records"]}
    assert ids_a != ids_b or [e["permuted_position"] for e in payload_a["records"]] != [e["permuted_position"] for e in payload_b["records"]]


# Manifest construction: exact counts, exactly N selected at 0..N-1


def test_manifest_records_exact_counts():
    payload, rows, sf = _build_synthetic_manifest()
    assert payload["raw_record_count"] == 10
    assert payload["parse_excluded_count"] == 3
    assert payload["parse_excluded_by_reason"] == {
        "chosen_malformed": 1, "rejected_malformed": 1, "history_mismatch": 1,
    }
    assert payload["duplicate_excluded_count"] == 1
    assert payload["eligible_record_count"] == 6
    assert payload["selected_record_count"] == 4
    assert payload["permutation"]["seed"] == 20260815
    assert payload["permutation"]["algorithm"] == hh_rlhf.PERMUTATION_ALGORITHM
    assert payload["source_file"] == sf


def test_exactly_n_selected_records_at_contiguous_positions():
    payload, _, _ = _build_synthetic_manifest()
    positions = sorted(e["permuted_position"] for e in payload["records"])
    assert positions == [0, 1, 2, 3]
    for entry in payload["records"]:
        assert entry["partition"] == "evaluation"


def test_wrong_raw_record_count_is_rejected():
    rows = _synthetic_rows()
    with pytest.raises(hh_rlhf.HHRLHFError):
        hh_rlhf.build_manifest(rows, source_file=_fake_source_file(), expected_raw_count=len(rows) - 1)


def test_too_few_eligible_records_is_rejected():
    rows = _synthetic_rows()[:2]  # only 2 well-formed unique rows, need 4 selected
    with pytest.raises(hh_rlhf.HHRLHFError):
        hh_rlhf.build_manifest(rows, source_file=_fake_source_file(), expected_raw_count=2, expected_selected_count=4)


# Save / load / self-hash / frozen identity


def test_manifest_self_hash_is_verified_on_load(tmp_path):
    payload, _, _ = _build_synthetic_manifest()
    path = tmp_path / "manifest.json"
    hh_rlhf.save_manifest(payload, path)
    loaded = hh_rlhf.load_manifest(path)
    assert loaded == payload
    assert loaded["manifest_hash"] == hh_rlhf.compute_manifest_hash(loaded)


def test_an_existing_different_manifest_is_never_overwritten(tmp_path):
    payload_a, rows, sf = _build_synthetic_manifest(seed=1)
    path = tmp_path / "manifest.json"
    hh_rlhf.save_manifest(payload_a, path)
    on_disk = path.read_text(encoding="utf-8")

    payload_b, _, _ = _build_synthetic_manifest(rows=rows, source_file=sf, seed=2)
    with pytest.raises(hh_rlhf.HHRLHFError):
        hh_rlhf.save_manifest(payload_b, path)
    assert path.read_text(encoding="utf-8") == on_disk


def test_tampering_with_a_saved_manifest_fails_verification(tmp_path):
    payload, _, _ = _build_synthetic_manifest()
    path = tmp_path / "manifest.json"
    hh_rlhf.save_manifest(payload, path)
    tampered = path.read_text(encoding="utf-8").replace(payload["records"][0]["prompt_hash"], "0" * 64)
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(hh_rlhf.HHRLHFError):
        hh_rlhf.load_manifest(path)


def _synthetic_expected_identity(payload, sf):
    return {
        "schema_version": hh_rlhf.SCHEMA_VERSION,
        "repository": hh_rlhf.REPOSITORY,
        "data_dir": hh_rlhf.DATA_DIR,
        "split": hh_rlhf.SPLIT,
        "revision": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        "seed": 20260815,
        "algorithm": hh_rlhf.PERMUTATION_ALGORITHM,
        "manifest_hash": payload["manifest_hash"],
        "raw_record_count": payload["raw_record_count"],
        "parse_excluded_count": payload["parse_excluded_count"],
        "duplicate_excluded_count": payload["duplicate_excluded_count"],
        "eligible_record_count": payload["eligible_record_count"],
        "selected_record_count": payload["selected_record_count"],
        "source_compressed_sha256": sf["compressed_sha256"],
    }


def test_validate_manifest_identity_accepts_a_matching_manifest():
    payload, _, sf = _build_synthetic_manifest()
    hh_rlhf.validate_manifest_identity(payload, expected=_synthetic_expected_identity(payload, sf))  # no raise


@pytest.mark.parametrize("field,new_value", [
    ("manifest_hash", "0" * 64),
    ("revision", "some-other-revision"),
    ("seed", 999),
    ("raw_record_count", 1),
    ("eligible_record_count", 1),
    ("selected_record_count", 1),
    ("source_compressed_sha256", "f" * 64),
])
def test_validate_manifest_identity_rejects_each_mismatched_field(field, new_value):
    payload, _, sf = _build_synthetic_manifest()
    expected = _synthetic_expected_identity(payload, sf)
    expected[field] = new_value
    with pytest.raises(hh_rlhf.HHRLHFError):
        hh_rlhf.validate_manifest_identity(payload, expected=expected)


def test_the_frozen_real_manifest_matches_its_own_hardcoded_identity():
    """The manifest actually built by scripts/build_hh_rlhf_manifest.py against the
    pinned source must match FROZEN_HH_RLHF_IDENTITY -- this is what the runtime loader
    trusts. A differently-rehashed manifest, even if internally self-consistent, must
    still be rejected (exercised above); this pins that the frozen constant is correct
    for the manifest actually on disk."""
    manifest = hh_rlhf.load_manifest(hh_rlhf._default_manifest_path())
    hh_rlhf.validate_manifest_identity(manifest)  # must not raise
    assert manifest["raw_record_count"] == hh_rlhf.RAW_RECORD_COUNT
    assert manifest["selected_record_count"] == hh_rlhf.SELECTED_RECORD_COUNT == 200
    assert manifest["dataset"]["revision"] == hh_rlhf.REVISION


# Source-binding validation


def test_validate_source_binding_accepts_matching_rows():
    payload, rows, sf = _build_synthetic_manifest()
    hh_rlhf.validate_source_binding(payload, rows, sf)  # must not raise


def test_validate_source_binding_rejects_a_changed_source_file_hash():
    payload, rows, sf = _build_synthetic_manifest()
    drifted_sf = dict(sf, compressed_sha256="f" * 64)
    with pytest.raises(hh_rlhf.HHRLHFError):
        hh_rlhf.validate_source_binding(payload, rows, drifted_sf)


def test_validate_source_binding_rejects_drifted_row_content():
    payload, rows, sf = _build_synthetic_manifest()
    drifted_rows = list(rows)
    drifted_rows[0] = dict(drifted_rows[0], chosen=drifted_rows[0]["chosen"] + "\n\nHuman: extra\n\nAssistant: more")
    with pytest.raises(hh_rlhf.HHRLHFError):
        hh_rlhf.validate_source_binding(payload, drifted_rows, sf)


def test_validate_source_binding_rejects_a_changed_exclusion_reason():
    payload, rows, sf = _build_synthetic_manifest()
    tampered = copy.deepcopy(payload)
    for excl in tampered["parse_exclusions"]:
        if excl["reason"] == "chosen_malformed":
            excl["reason"] = "rejected_malformed"
    with pytest.raises(hh_rlhf.HHRLHFError):
        hh_rlhf.validate_source_binding(tampered, rows, sf)


def test_validate_source_binding_rejects_noncontiguous_positions():
    payload, rows, sf = _build_synthetic_manifest()
    tampered = copy.deepcopy(payload)
    tampered["records"][0]["permuted_position"] = tampered["records"][1]["permuted_position"]
    with pytest.raises(hh_rlhf.HHRLHFError):
        hh_rlhf.validate_source_binding(tampered, rows, sf)


def test_validate_source_binding_rejects_a_selected_id_missing_from_eligible_rows():
    payload, rows, sf = _build_synthetic_manifest()
    tampered = copy.deepcopy(payload)
    tampered["records"][0]["source_id"] = "hh-harmless-test-9999"
    with pytest.raises(hh_rlhf.HHRLHFError):
        hh_rlhf.validate_source_binding(tampered, rows, sf)


# Runtime loader (frozen 200-record panel, exercised against a synthetic small panel)


def _write_gzip_jsonl(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _build_and_save_real_file_manifest(tmp_path, rows, **overrides):
    gz_path = tmp_path / "test.jsonl.gz"
    _write_gzip_jsonl(gz_path, rows)
    source_file = hh_rlhf.source_file_identity(gz_path)
    kwargs = dict(expected_raw_count=len(rows), expected_selected_count=4)
    kwargs.update(overrides)
    payload = hh_rlhf.build_manifest(rows, source_file=source_file, **kwargs)
    manifest_path = tmp_path / "manifest.json"
    hh_rlhf.save_manifest(payload, manifest_path)
    return manifest_path, gz_path, payload


def test_load_hh_rlhf_evaluation_returns_full_schema_in_position_order(tmp_path, monkeypatch):
    rows = _synthetic_rows()
    manifest_path, gz_path, payload = _build_and_save_real_file_manifest(tmp_path, rows)
    monkeypatch.setattr(hh_rlhf, "validate_manifest_identity", lambda *a, **k: None)

    records = hh_rlhf.load_hh_rlhf_evaluation(manifest_path=manifest_path, source_path=gz_path)

    assert len(records) == 4
    assert [r["permuted_position"] for r in records] == [0, 1, 2, 3]
    for r in records:
        assert r["partition"] == "evaluation"
        assert r["manifest_hash"] == payload["manifest_hash"]
        for field in ("source_id", "source_index", "prompt", "prompt_hash", "chosen", "rejected",
                      "chosen_hash", "rejected_hash", "partition", "permuted_position", "manifest_hash"):
            assert field in r


def test_load_hh_rlhf_evaluation_rejects_a_manifest_that_fails_the_frozen_identity_check(tmp_path):
    rows = _synthetic_rows()
    manifest_path, gz_path, _ = _build_and_save_real_file_manifest(tmp_path, rows)
    with pytest.raises(hh_rlhf.HHRLHFError):
        hh_rlhf.load_hh_rlhf_evaluation(manifest_path=manifest_path, source_path=gz_path)


def test_load_hh_rlhf_evaluation_rejects_a_source_file_that_drifted_after_the_manifest_was_built(tmp_path, monkeypatch):
    rows = _synthetic_rows()
    manifest_path, gz_path, _ = _build_and_save_real_file_manifest(tmp_path, rows)
    monkeypatch.setattr(hh_rlhf, "validate_manifest_identity", lambda *a, **k: None)

    drifted_rows = list(rows)
    drifted_rows[0] = _mk_row("a completely different question", "a completely different answer")
    _write_gzip_jsonl(gz_path, drifted_rows)

    with pytest.raises(hh_rlhf.HHRLHFError):
        hh_rlhf.load_hh_rlhf_evaluation(manifest_path=manifest_path, source_path=gz_path)


# Source file identity: compressed vs. canonical (decompressed) representation


def test_source_file_identity_distinguishes_compressed_from_canonical_hash(tmp_path):
    gz_path = tmp_path / "sample.jsonl.gz"
    _write_gzip_jsonl(gz_path, [{"chosen": "x", "rejected": "y"}])
    identity = hh_rlhf.source_file_identity(gz_path)
    assert identity["compressed_sha256"] != identity["canonical_sha256"]
    assert len(identity["compressed_sha256"]) == 64
    assert len(identity["canonical_sha256"]) == 64
    assert identity["compressed_size_bytes"] == gz_path.stat().st_size
