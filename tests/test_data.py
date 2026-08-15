"""Guards on stable HarmfulQA identities.

The bug this pins: `load_harmfulqa` used to assign `id` from the post-shuffle
enumeration, so the same source row got a different ID whenever the seed or requested
sample size changed. `_FakeDataset` mocks just enough of the `datasets.Dataset` surface
(`map`/`shuffle`/`select`/iteration) that this needs no network access.

Run with:
    pytest tests/test_data.py -v
"""

from __future__ import annotations

import random

import pytest

import steering.data as data_module
import steering.splits as splits_module

RETAINED_TOTAL = sum(hi - lo for _, lo, hi in splits_module.PARTITION_BOUNDS)  # 1938
N_PAIRS = 22
RAW_TOTAL = RETAINED_TOTAL + N_PAIRS  # 1960


class _FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def map(self, fn, with_indices=False):
        if with_indices:
            new_rows = [{**row, **fn(row, i)} for i, row in enumerate(self.rows)]
        else:
            new_rows = [{**row, **fn(row)} for row in self.rows]
        return _FakeDataset(new_rows)

    def shuffle(self, seed=None):
        rows = list(self.rows)
        random.Random(seed).shuffle(rows)
        return _FakeDataset(rows)

    def select(self, indices):
        return _FakeDataset([self.rows[i] for i in indices])

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)


def _patch_dataset(monkeypatch, n_rows=50):
    rows = [{"question": f"question number {i}"} for i in range(n_rows)]
    monkeypatch.setattr(data_module, "load_dataset", lambda *a, **k: _FakeDataset(rows))
    return rows


def test_harmfulqa_id_source_id_and_hash_are_stable_within_a_seed(monkeypatch):
    """A row picked by both a small and a larger request under the same seed is a
    prefix relationship (same shuffle order) -- it must carry identical identity."""
    _patch_dataset(monkeypatch, n_rows=50)

    small = data_module.load_harmfulqa(n=10, seed=1)
    large = data_module.load_harmfulqa(n=40, seed=1)

    large_by_prompt = {r["prompt"]: r for r in large}
    for rec in small:
        match = large_by_prompt[rec["prompt"]]
        assert match["id"] == rec["id"]
        assert match["source_id"] == rec["source_id"]
        assert match["source_index"] == rec["source_index"]
        assert match["prompt_hash"] == rec["prompt_hash"]


def test_harmfulqa_id_and_hash_are_stable_across_seeds(monkeypatch):
    """Two disjoint-seed samples of 40 out of 50 rows must overlap by pigeonhole
    (40 + 40 - 50 = 30 shared rows); every shared row must carry the same identity."""
    _patch_dataset(monkeypatch, n_rows=50)

    a = data_module.load_harmfulqa(n=40, seed=1)
    b = data_module.load_harmfulqa(n=40, seed=7)

    by_prompt_a = {r["prompt"]: r for r in a}
    by_prompt_b = {r["prompt"]: r for r in b}
    shared = set(by_prompt_a) & set(by_prompt_b)
    assert len(shared) >= 30

    for prompt in shared:
        ra, rb = by_prompt_a[prompt], by_prompt_b[prompt]
        assert ra["id"] == rb["id"]
        assert ra["source_id"] == rb["source_id"]
        assert ra["source_index"] == rb["source_index"]
        assert ra["prompt_hash"] == rb["prompt_hash"]


def test_harmfulqa_source_ids_are_unique_within_a_load(monkeypatch):
    _patch_dataset(monkeypatch, n_rows=50)
    records = data_module.load_harmfulqa(n=30, seed=3)
    ids = [r["source_id"] for r in records]
    assert len(ids) == len(set(ids))
    indices = [r["source_index"] for r in records]
    assert len(indices) == len(set(indices))


def test_harmfulqa_records_carry_id_source_id_source_index_and_prompt_hash(monkeypatch):
    _patch_dataset(monkeypatch, n_rows=10)
    records = data_module.load_harmfulqa(n=5, seed=0)
    for rec in records:
        assert rec["id"] == rec["source_id"]
        assert rec["id"].startswith("harmfulqa-")
        assert rec["id"] == f"harmfulqa-{rec['source_index']}"
        assert isinstance(rec["source_index"], int)
        assert len(rec["prompt_hash"]) == 64


# Pinned revision


def test_harmfulqa_default_revision_is_pinned_not_latest(monkeypatch):
    """Requirement: `load_harmfulqa`'s default must be the frozen revision, not
    whatever HEAD of the dataset repo happens to be when the call runs."""
    calls = []

    def spy_load_dataset(*args, **kwargs):
        calls.append(kwargs)
        return _FakeDataset([{"question": "only question"}])

    monkeypatch.setattr(data_module, "load_dataset", spy_load_dataset)

    data_module.load_harmfulqa(n=1, seed=0)

    assert calls, "load_dataset was never called"
    assert calls[0].get("revision") == data_module.HARMFULQA_REVISION
    assert calls[0].get("name") == data_module.HARMFULQA_CONFIG
    assert data_module.HARMFULQA_REVISION == "6f1a78aed47d16c0695e4595d0159abc38197bfd"
    assert data_module.HARMFULQA_CONFIG == "default"


# load_harmfulqa_partition (Task 003)


def _synthetic_prompts(n_unique=RETAINED_TOTAL, n_pairs=N_PAIRS):
    """1,960 raw prompts (1,938 unique, 22 verbatim duplicates appended at higher
    indices) -- the same shape as the real HarmfulQA source, index = list position."""
    prompts = [f"synthetic prompt number {i}" for i in range(n_unique)]
    prompts += [prompts[i] for i in range(n_pairs)]
    return prompts


def _raw_records_from_prompts(prompts):
    return [{"source_id": f"harmfulqa-{i}", "source_index": i, "prompt": p} for i, p in enumerate(prompts)]


def _build_synthetic_manifest(tmp_path, prompts=None, **build_overrides):
    prompts = prompts if prompts is not None else _synthetic_prompts()
    raw_records = _raw_records_from_prompts(prompts)
    kwargs = dict(
        seed=20260815, dataset_repo="declare-lab/HarmfulQA", dataset_split="train",
        dataset_revision="6f1a78aed47d16c0695e4595d0159abc38197bfd", dataset_config="default",
    )
    kwargs.update(build_overrides)
    payload = splits_module.build_manifest(raw_records, **kwargs)
    path = tmp_path / "harmfulqa_v1.json"
    splits_module.save_manifest(payload, path)
    return path, payload, prompts


def _patch_harmfulqa_source(monkeypatch, prompts):
    monkeypatch.setattr(data_module, "load_dataset", lambda *a, **k: [{"question": p} for p in prompts])


def _skip_identity_check(monkeypatch):
    """The happy-path / source-binding tests below use a synthetic manifest whose
    repository/revision/hash legitimately differ from the real frozen identity --
    `validate_manifest_identity` against the real constant is exercised separately with
    the real manifest_path unset, and unit-tested directly in test_splits.py."""
    monkeypatch.setattr(data_module, "validate_manifest_identity", lambda manifest, expected=None: None)


def test_load_harmfulqa_partition_returns_exact_counts_order_and_provenance(tmp_path, monkeypatch):
    manifest_path, payload, prompts = _build_synthetic_manifest(tmp_path)
    _patch_harmfulqa_source(monkeypatch, prompts)
    _skip_identity_check(monkeypatch)

    expected_counts = {
        "construction": 1378, "calibration": 200, "final_evaluation": 300, "development": 60,
    }
    by_partition = {}
    for partition, expected_n in expected_counts.items():
        records = data_module.load_harmfulqa_partition(partition, manifest_path=manifest_path)
        assert len(records) == expected_n

        positions = [r["permuted_position"] for r in records]
        assert positions == sorted(positions), "must be returned in ascending permuted_position"

        for r in records:
            assert r["partition"] == partition
            assert r["id"] == r["source_id"]
            assert r["manifest_hash"] == payload["manifest_hash"]
            for field in ("id", "source_id", "source_index", "prompt", "prompt_hash", "partition", "permuted_position", "manifest_hash"):
                assert field in r
        by_partition[partition] = records

    assert [r["permuted_position"] for r in by_partition["construction"]] == list(range(0, 1378))
    assert [r["permuted_position"] for r in by_partition["calibration"]] == list(range(1378, 1578))
    assert [r["permuted_position"] for r in by_partition["final_evaluation"]] == list(range(1578, 1878))
    assert [r["permuted_position"] for r in by_partition["development"]] == list(range(1878, 1938))

    id_sets = {p: {r["source_id"] for r in recs} for p, recs in by_partition.items()}
    hash_sets = {p: {r["prompt_hash"] for r in recs} for p, recs in by_partition.items()}
    names = list(expected_counts)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert id_sets[names[i]].isdisjoint(id_sets[names[j]])
            assert hash_sets[names[i]].isdisjoint(hash_sets[names[j]])

    union_ids = set().union(*id_sets.values())
    assert len(union_ids) == RETAINED_TOTAL

    excluded_ids = {e["excluded_source_id"] for e in payload["duplicate_exclusions"]}
    assert excluded_ids.isdisjoint(union_ids)
    assert len(excluded_ids) == N_PAIRS


def test_load_harmfulqa_partition_rejects_an_invalid_partition_before_any_dataset_load(monkeypatch):
    calls = []
    monkeypatch.setattr(data_module, "load_dataset", lambda *a, **k: calls.append(1) or [])

    with pytest.raises(ValueError):
        data_module.load_harmfulqa_partition("nonexistent_partition")

    assert not calls, "load_dataset must not be called for an invalid partition name"


def test_load_harmfulqa_partition_rejects_a_tampered_manifest_hash(tmp_path, monkeypatch):
    manifest_path, payload, prompts = _build_synthetic_manifest(tmp_path)
    tampered = manifest_path.read_text(encoding="utf-8").replace(payload["records"][0]["prompt_hash"], "0" * 64)
    manifest_path.write_text(tampered, encoding="utf-8")

    _patch_harmfulqa_source(monkeypatch, prompts)

    with pytest.raises(splits_module.SplitError):
        data_module.load_harmfulqa_partition("development", manifest_path=manifest_path)


def test_load_harmfulqa_partition_rejects_incorrect_metadata_via_identity_check(tmp_path, monkeypatch):
    """No identity-check bypass here: a manifest built against the wrong revision must
    be rejected by the real `validate_manifest_identity` against the frozen identity."""
    manifest_path, payload, prompts = _build_synthetic_manifest(
        tmp_path, dataset_revision="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
    )
    _patch_harmfulqa_source(monkeypatch, prompts)

    with pytest.raises(splits_module.SplitError):
        data_module.load_harmfulqa_partition("development", manifest_path=manifest_path)


def test_load_harmfulqa_partition_rejects_a_changed_source_prompt(tmp_path, monkeypatch):
    manifest_path, payload, prompts = _build_synthetic_manifest(tmp_path)
    drifted = list(prompts)
    drifted[0] = "this prompt text changed after the manifest was built"

    _patch_harmfulqa_source(monkeypatch, drifted)
    _skip_identity_check(monkeypatch)

    with pytest.raises(splits_module.SplitError):
        data_module.load_harmfulqa_partition("construction", manifest_path=manifest_path)


def test_load_harmfulqa_partition_rejects_a_missing_source_row(tmp_path, monkeypatch):
    manifest_path, payload, prompts = _build_synthetic_manifest(tmp_path)
    truncated = prompts[:-1]

    _patch_harmfulqa_source(monkeypatch, truncated)
    _skip_identity_check(monkeypatch)

    with pytest.raises(splits_module.SplitError):
        data_module.load_harmfulqa_partition("development", manifest_path=manifest_path)


def test_default_harmfulqa_manifest_path_is_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = data_module._default_harmfulqa_manifest_path()
    assert path.name == "harmfulqa_v1.json"
    assert path.parent.name == "manifests"
    assert str(tmp_path) not in str(path)
    assert path.is_absolute()
