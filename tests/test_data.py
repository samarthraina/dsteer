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

import steering.data as data_module


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
