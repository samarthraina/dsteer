"""Guards on the manifest-backed HarmfulQA partitions in `steer_sweep.py`.

The bug this pins: the active sweep script called the ad hoc `load_harmfulqa(n, seed)`
path for calibration/development sweeps, and read `activations.pt` from a plain
directory with no check that it actually came from the frozen construction partition.
These pin partition resolution/rejection, output-path separation, hold-out rejection,
provenance pass-through into JSONL, and activation-artifact validation.

Run with:
    pytest tests/test_steer_sweep.py -v

CPU-only, network-free: the frozen partition loader and generation are mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import steer_sweep  # noqa: E402

from steering.generate import GenerationResult
from steering.utils import read_jsonl


def _cfg(**overrides):
    return steer_sweep.SteerSweepConfig(**overrides)


# resolve_harmfulqa_partition


def test_resolve_partition_cli_overrides_yaml():
    cfg = _cfg(prompt_source="harmfulqa", prompt_partition="calibration")
    assert steer_sweep.resolve_harmfulqa_partition(cfg, "development") == "development"


def test_resolve_partition_falls_back_to_yaml_when_no_cli_override():
    cfg = _cfg(prompt_source="harmfulqa", prompt_partition="calibration")
    assert steer_sweep.resolve_harmfulqa_partition(cfg, None) == "calibration"


def test_resolve_partition_rejects_construction():
    cfg = _cfg(prompt_source="harmfulqa", prompt_partition="construction")
    with pytest.raises(ValueError):
        steer_sweep.resolve_harmfulqa_partition(cfg, None)


def test_resolve_partition_rejects_final_evaluation():
    cfg = _cfg(prompt_source="harmfulqa", prompt_partition="final_evaluation")
    with pytest.raises(ValueError):
        steer_sweep.resolve_harmfulqa_partition(cfg, None)


def test_resolve_partition_rejects_a_missing_partition():
    """No silent default: an unset partition for prompt_source='harmfulqa' must raise."""
    cfg = _cfg(prompt_source="harmfulqa", prompt_partition=None)
    with pytest.raises(ValueError):
        steer_sweep.resolve_harmfulqa_partition(cfg, None)


def test_resolve_partition_is_a_noop_for_non_harmfulqa_sources():
    cfg = _cfg(prompt_source="advbench", prompt_partition=None)
    assert steer_sweep.resolve_harmfulqa_partition(cfg, None) is None
    assert steer_sweep.resolve_harmfulqa_partition(cfg, "development") == "development"


# load_prompts


def test_load_prompts_harmfulqa_uses_the_partition_loader_ignoring_n_and_seed(monkeypatch):
    fake_records = [{"id": f"harmfulqa-{i}", "prompt": f"p{i}"} for i in range(200)]
    calls = []

    def fake_loader(partition):
        calls.append(partition)
        return list(fake_records)

    monkeypatch.setattr(steer_sweep, "load_harmfulqa_partition", fake_loader)

    a = steer_sweep.load_prompts("harmfulqa", n=1, seed=1, partition="calibration")
    b = steer_sweep.load_prompts("harmfulqa", n=999, seed=999999, partition="calibration")

    assert calls == ["calibration", "calibration"]
    assert a == b == fake_records


def test_load_prompts_advbench_unaffected(monkeypatch):
    """Existing non-HarmfulQA behavior (n-slicing via LOADERS) must be untouched."""
    fake_records = [{"id": f"advbench-{i}", "prompt": f"p{i}"} for i in range(10)]
    monkeypatch.setitem(steer_sweep.LOADERS, "advbench", lambda n, seed: fake_records[:n])
    out = steer_sweep.load_prompts("advbench", n=4, seed=1)
    assert len(out) == 4


def test_load_prompts_rejects_an_unknown_source():
    with pytest.raises(ValueError):
        steer_sweep.load_prompts("nonsense", n=1, seed=1)


# output_source_segment


def test_output_segment_differs_across_partitions():
    dev = steer_sweep.output_source_segment("harmfulqa", "development")
    cal = steer_sweep.output_source_segment("harmfulqa", "calibration")
    assert dev != cal
    assert dev == Path("harmfulqa") / "development"
    assert cal == Path("harmfulqa") / "calibration"


def test_output_segment_unchanged_for_non_harmfulqa():
    assert steer_sweep.output_source_segment("advbench", None) == Path("advbench")


# reject_hold_out_for_harmfulqa


def test_hold_out_rejected_for_harmfulqa_when_nonzero():
    with pytest.raises(ValueError):
        steer_sweep.reject_hold_out_for_harmfulqa("harmfulqa", 5)


def test_hold_out_allowed_for_harmfulqa_when_zero():
    steer_sweep.reject_hold_out_for_harmfulqa("harmfulqa", 0)  # must not raise


def test_hold_out_unrestricted_for_non_harmfulqa():
    steer_sweep.reject_hold_out_for_harmfulqa("advbench", 300)  # must not raise


# validate_construction_activations
#
# These validate against the real committed manifest -- local file read plus SHA-256,
# no network -- so "matches the frozen construction partition" is checked against the
# actual provenance a run would be compared to, not a synthetic stand-in.


def _real_construction_ids_and_hashes():
    manifest = steer_sweep.load_manifest(steer_sweep._default_harmfulqa_manifest_path())
    construction = steer_sweep._construction_records(manifest)
    return (
        [e["source_id"] for e in construction],
        [e["prompt_hash"] for e in construction],
        manifest["manifest_hash"],
    )


def _valid_blob():
    source_ids, prompt_hashes, manifest_hash = _real_construction_ids_and_hashes()
    n = len(source_ids)
    return {
        "it": torch.zeros(4, n, 8), "dpo": torch.zeros(4, n, 8),
        "source_ids": list(source_ids), "prompt_hashes": list(prompt_hashes),
        "partition": "construction", "manifest_hash": manifest_hash,
    }


def test_validate_construction_activations_accepts_a_blob_matching_the_real_manifest():
    steer_sweep.validate_construction_activations(_valid_blob())  # must not raise


def test_validate_construction_activations_rejects_a_legacy_blob():
    blob = {"it": torch.zeros(4, 5, 8), "dpo": torch.zeros(4, 5, 8)}
    with pytest.raises(ValueError):
        steer_sweep.validate_construction_activations(blob)


def test_validate_construction_activations_rejects_the_wrong_partition():
    blob = _valid_blob()
    blob["partition"] = "calibration"
    with pytest.raises(ValueError):
        steer_sweep.validate_construction_activations(blob)


def test_validate_construction_activations_rejects_the_wrong_manifest_hash():
    blob = _valid_blob()
    blob["manifest_hash"] = "0" * 64
    with pytest.raises(ValueError):
        steer_sweep.validate_construction_activations(blob)


def test_validate_construction_activations_rejects_one_incorrect_source_id():
    blob = _valid_blob()
    mid = len(blob["source_ids"]) // 2
    blob["source_ids"][mid] = "harmfulqa-not-a-real-construction-id"
    with pytest.raises(ValueError):
        steer_sweep.validate_construction_activations(blob)


def test_validate_construction_activations_rejects_one_incorrect_prompt_hash():
    blob = _valid_blob()
    mid = len(blob["prompt_hashes"]) // 2
    blob["prompt_hashes"][mid] = "f" * 64
    with pytest.raises(ValueError):
        steer_sweep.validate_construction_activations(blob)


def test_validate_construction_activations_rejects_reordered_ids_and_hashes():
    blob = _valid_blob()
    # Swap two entries -- same set of IDs/hashes, correct length, wrong order.
    blob["source_ids"][0], blob["source_ids"][1] = blob["source_ids"][1], blob["source_ids"][0]
    blob["prompt_hashes"][0], blob["prompt_hashes"][1] = blob["prompt_hashes"][1], blob["prompt_hashes"][0]
    with pytest.raises(ValueError):
        steer_sweep.validate_construction_activations(blob)


def test_validate_construction_activations_rejects_mismatched_it_dpo_tensor_shapes():
    blob = _valid_blob()
    n = blob["it"].shape[1]
    blob["dpo"] = torch.zeros(4, n, 16)  # different hidden dim from `it`
    with pytest.raises(ValueError):
        steer_sweep.validate_construction_activations(blob)


def test_validate_construction_activations_rejects_a_prompt_dimension_mismatch():
    """it/dpo shapes agree with each other but disagree with the manifest's record
    count -- source_ids/prompt_hashes stay fully correct so this isolates the
    tensor-vs-manifest-count check from the provenance-list checks above."""
    source_ids, prompt_hashes, manifest_hash = _real_construction_ids_and_hashes()
    n = len(source_ids) - 1  # one short of the real construction count
    blob = {
        "it": torch.zeros(4, n, 8), "dpo": torch.zeros(4, n, 8),
        "source_ids": source_ids, "prompt_hashes": prompt_hashes,
        "partition": "construction", "manifest_hash": manifest_hash,
    }
    with pytest.raises(ValueError):
        steer_sweep.validate_construction_activations(blob)


# run_one: provenance pass-through into JSONL


def test_run_one_preserves_provenance_fields_in_jsonl(tmp_path, monkeypatch):
    manifest_hash = "0" * 64  # run_one passes provenance through as-is; the value itself is irrelevant here
    records = [
        {
            "id": "harmfulqa-10", "source_id": "harmfulqa-10", "source_index": 10, "prompt": "p10",
            "prompt_hash": "hash10", "partition": "development", "permuted_position": 1878,
            "manifest_hash": manifest_hash,
        },
        {
            "id": "harmfulqa-11", "source_id": "harmfulqa-11", "source_index": 11, "prompt": "p11",
            "prompt_hash": "hash11", "partition": "development", "permuted_position": 1879,
            "manifest_hash": manifest_hash,
        },
    ]
    chats = ["chat-p10", "chat-p11"]

    def fake_generate_batched(model, tokenizer, todo_chats, **kwargs):
        assert kwargs.get("return_metadata") is True
        return [
            GenerationResult(text=f"response for {c}", generated_token_count=5,
                              stop_reason="eos_token", stop_token_id=2)
            for c in todo_chats
        ]

    monkeypatch.setattr(steer_sweep, "generate_batched", fake_generate_batched)

    cfg = _cfg()
    path = tmp_path / "baseline.jsonl"
    n = steer_sweep.run_one(
        model=None, tokenizer=None, records=records, chats=chats,
        coefficient=0.0, vectors={}, cfg=cfg, path=path, label="test", batch_size=2,
    )
    assert n == 2

    written = read_jsonl(path)
    assert len(written) == 2
    for rec, chat, out in zip(records, chats, written):
        for key in ("source_id", "source_index", "prompt_hash", "partition", "permuted_position", "manifest_hash"):
            assert out[key] == rec[key]
        assert out["response"] == f"response for {chat}"
        assert out["lambda"] == 0.0


# run_one: generation metadata (Task 006)


def test_run_one_persists_all_four_generation_metadata_fields(tmp_path, monkeypatch):
    records = [{"id": "r1", "prompt": "p1"}, {"id": "r2", "prompt": "p2"}]
    chats = ["chat-1", "chat-2"]
    scripted = [
        GenerationResult(text="first response", generated_token_count=12,
                          stop_reason="eos_token", stop_token_id=2),
        GenerationResult(text="second response", generated_token_count=512,
                          stop_reason="max_new_tokens", stop_token_id=None),
    ]

    def fake_generate_batched(model, tokenizer, todo_chats, **kwargs):
        return list(scripted)

    monkeypatch.setattr(steer_sweep, "generate_batched", fake_generate_batched)

    cfg = _cfg()
    path = tmp_path / "baseline.jsonl"
    n = steer_sweep.run_one(
        model=None, tokenizer=None, records=records, chats=chats,
        coefficient=0.0, vectors={}, cfg=cfg, path=path, label="test", batch_size=2,
    )
    assert n == 2

    written = read_jsonl(path)
    for result, out in zip(scripted, written):
        assert out["response"] == result.text
        assert out["generated_token_count"] == result.generated_token_count
        assert out["stop_reason"] == result.stop_reason
        assert out["stop_token_id"] == result.stop_token_id


def test_run_one_raises_clearly_on_a_result_count_mismatch(tmp_path, monkeypatch):
    """generate_batched returning fewer results than pending records must fail loudly,
    not silently truncate via zip(...) and write a partial batch."""
    records = [{"id": "r1", "prompt": "p1"}, {"id": "r2", "prompt": "p2"}]
    chats = ["chat-1", "chat-2"]

    def fake_generate_batched(model, tokenizer, todo_chats, **kwargs):
        return [GenerationResult(text="only one", generated_token_count=3,
                                  stop_reason="eos_token", stop_token_id=2)]

    monkeypatch.setattr(steer_sweep, "generate_batched", fake_generate_batched)

    cfg = _cfg()
    path = tmp_path / "baseline.jsonl"
    with pytest.raises(RuntimeError):
        steer_sweep.run_one(
            model=None, tokenizer=None, records=records, chats=chats,
            coefficient=0.0, vectors={}, cfg=cfg, path=path, label="test", batch_size=2,
        )
    assert not path.exists()  # nothing written before the mismatch was caught


# Config file


def test_steer_sweep_yaml_parses_to_protocol_values():
    repo_root = Path(__file__).resolve().parents[1]
    cfg = steer_sweep.SteerSweepConfig.from_yaml(repo_root / "configs" / "steer_sweep.yaml")
    assert cfg.prompt_source == "harmfulqa"
    assert cfg.prompt_partition == "calibration"
    assert cfg.activations_dir == "outputs/layer_profile"
    assert steer_sweep.resolve_harmfulqa_partition(cfg, None) == "calibration"
