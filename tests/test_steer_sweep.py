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

import argparse
import sys
from pathlib import Path

import pytest
import yaml

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


# build_run_config


def _model_cfg_dict(**overrides):
    base = {
        "name": "tinytest", "base_model": "org/base", "it_model": "org/it",
        "dpo_model": "org/dpo", "architecture": "llama", "num_layers": 4,
    }
    base.update(overrides)
    return base


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _steer_sweep_parser() -> argparse.ArgumentParser:
    """Mirrors steer_sweep.main()'s real CLI surface, for testing build_run_config and
    metadata wiring without running main() end to end."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config")
    parser.add_argument("--eval-config")
    parser.add_argument("--side", choices=["it", "dpo"])
    parser.add_argument("--random-control", action="store_true")
    parser.add_argument("--mode", choices=["add", "ablate"], default="add")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--vectors", default=None)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--hold-out", type=int, default=0)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--hourly-rate", type=float, default=None)
    return parser


def test_build_run_config_binds_model_eval_and_full_cli_including_defaults(tmp_path):
    model_yaml = _write_yaml(tmp_path / "model.yaml", _model_cfg_dict())
    eval_yaml = _write_yaml(tmp_path / "eval.yaml", {"prompt_source": "advbench", "n_prompts": 5})

    model_cfg = steer_sweep.ModelConfig.from_yaml(model_yaml)
    eval_cfg = steer_sweep.SteerSweepConfig.from_yaml(eval_yaml)

    args = _steer_sweep_parser().parse_args([
        "--model-config", str(model_yaml), "--eval-config", str(eval_yaml), "--side", "it",
    ])

    config = steer_sweep.build_run_config(model_cfg, eval_cfg, args)

    assert config["model"]["name"] == "tinytest"
    assert config["eval"]["prompt_source"] == "advbench"
    # every parsed CLI value, including ones never explicitly passed (defaulted).
    assert config["cli"] == {
        "model_config": str(model_yaml), "eval_config": str(eval_yaml), "side": "it",
        "random_control": False, "mode": "add", "tag": None, "vectors": None, "layers": None,
        "hold_out": 0, "partition": None, "seed": 42, "sync": False, "hourly_rate": None,
    }


# main(): metadata wiring (Task 009)


class _MetadataSentinel(Exception):
    """Raised by a mocked write_run_metadata to simulate an identity mismatch, so a
    test can prove nothing after the metadata call ran -- without mocking the rest of
    the pipeline."""


def _spy_write_run_metadata(monkeypatch, calls, raise_sentinel=True):
    def fake(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        if raise_sentinel:
            raise _MetadataSentinel()
        return Path("run_meta.json")

    monkeypatch.setattr(steer_sweep, "write_run_metadata", fake)


def _forbid(monkeypatch, name):
    """Fail loudly if `name` is ever called -- used to prove no side effect happens
    after a simulated metadata mismatch."""
    def boom(*a, **k):
        raise AssertionError(f"{name} must not be called after a metadata identity mismatch")

    monkeypatch.setattr(steer_sweep, name, boom)


def _setup_advbench_run(tmp_path, monkeypatch, extra_eval=None):
    model_yaml = _write_yaml(tmp_path / "model.yaml", _model_cfg_dict())
    eval_cfg_dict = {"prompt_source": "advbench", "n_prompts": 3, "output_dir": str(tmp_path / "out")}
    eval_cfg_dict.update(extra_eval or {})
    eval_yaml = _write_yaml(tmp_path / "eval.yaml", eval_cfg_dict)

    fake_records = [{"id": f"advbench-{i}", "prompt": f"p{i}"} for i in range(3)]
    monkeypatch.setitem(steer_sweep.LOADERS, "advbench", lambda n, seed: fake_records[:n])
    return model_yaml, eval_yaml


def test_argv_is_passed_as_an_exact_json_list_preserving_spaces_and_unicode(tmp_path, monkeypatch):
    model_yaml, eval_yaml = _setup_advbench_run(tmp_path, monkeypatch)
    argv = [
        "steer_sweep.py", "--model-config", str(model_yaml), "--eval-config", str(eval_yaml), "--side", "it",
        "--tag", "unicode üñíçødé and spaces here",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)

    with pytest.raises(_MetadataSentinel):
        steer_sweep.main()

    assert calls[0]["kwargs"]["argv"] == argv
    assert isinstance(calls[0]["kwargs"]["argv"], list)


def test_metadata_config_includes_the_full_parsed_cli_namespace_with_defaults(tmp_path, monkeypatch):
    model_yaml, eval_yaml = _setup_advbench_run(tmp_path, monkeypatch)
    argv = ["steer_sweep.py", "--model-config", str(model_yaml), "--eval-config", str(eval_yaml), "--side", "it"]
    monkeypatch.setattr(sys, "argv", argv)

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)

    with pytest.raises(_MetadataSentinel):
        steer_sweep.main()

    cli = calls[0]["kwargs"]["config"]["cli"]
    assert cli == {
        "model_config": str(model_yaml), "eval_config": str(eval_yaml), "side": "it",
        "random_control": False, "mode": "add", "tag": None, "vectors": None, "layers": None,
        "hold_out": 0, "partition": None, "seed": 42, "sync": False, "hourly_rate": None,
    }


def test_metadata_retains_resolved_model_and_eval_configs(tmp_path, monkeypatch):
    model_yaml, eval_yaml = _setup_advbench_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "steer_sweep.py", "--model-config", str(model_yaml), "--eval-config", str(eval_yaml), "--side", "dpo",
    ])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)

    with pytest.raises(_MetadataSentinel):
        steer_sweep.main()

    config = calls[0]["kwargs"]["config"]
    assert config["model"]["name"] == "tinytest"
    assert config["model"]["architecture"] == "llama"
    assert config["eval"]["prompt_source"] == "advbench"


def test_metadata_carries_exact_harmfulqa_provenance(tmp_path, monkeypatch):
    model_yaml = _write_yaml(tmp_path / "model.yaml", _model_cfg_dict())
    eval_yaml = _write_yaml(tmp_path / "eval.yaml", {
        "prompt_source": "harmfulqa", "prompt_partition": "calibration", "output_dir": str(tmp_path / "out"),
    })
    fake_records = [
        {"id": f"harmfulqa-{i}", "prompt": f"p{i}", "manifest_hash": "manifest-abc"}
        for i in range(4)
    ]
    monkeypatch.setattr(steer_sweep, "load_harmfulqa_partition", lambda partition: fake_records)
    monkeypatch.setattr(sys, "argv", [
        "steer_sweep.py", "--model-config", str(model_yaml), "--eval-config", str(eval_yaml), "--side", "it",
    ])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)

    with pytest.raises(_MetadataSentinel):
        steer_sweep.main()

    extra = calls[0]["kwargs"]["extra"]
    assert extra == {
        "harmfulqa_partition": "calibration",
        "harmfulqa_manifest_hash": "manifest-abc",
        "harmfulqa_record_count": 4,
    }


def test_metadata_carries_the_resolved_partition_when_supplied_through_the_cli_override(tmp_path, monkeypatch):
    """The eval config says calibration; --partition overrides it to development, and the
    metadata must reflect the CLI-resolved value, not the YAML default."""
    model_yaml = _write_yaml(tmp_path / "model.yaml", _model_cfg_dict())
    eval_yaml = _write_yaml(tmp_path / "eval.yaml", {
        "prompt_source": "harmfulqa", "prompt_partition": "calibration", "output_dir": str(tmp_path / "out"),
    })
    fake_records = [
        {"id": f"harmfulqa-{i}", "prompt": f"p{i}", "manifest_hash": "manifest-xyz"}
        for i in range(2)
    ]
    calls_to_loader = []

    def fake_loader(partition):
        calls_to_loader.append(partition)
        return fake_records

    monkeypatch.setattr(steer_sweep, "load_harmfulqa_partition", fake_loader)
    monkeypatch.setattr(sys, "argv", [
        "steer_sweep.py", "--model-config", str(model_yaml), "--eval-config", str(eval_yaml),
        "--side", "it", "--partition", "development",
    ])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)

    with pytest.raises(_MetadataSentinel):
        steer_sweep.main()

    assert calls_to_loader == ["development"]  # the CLI override, not the YAML's "calibration"
    extra = calls[0]["kwargs"]["extra"]
    assert extra["harmfulqa_partition"] == "development"
    assert calls[0]["kwargs"]["config"]["cli"]["partition"] == "development"


def test_write_run_metadata_is_called_before_setup_logging(tmp_path, monkeypatch):
    model_yaml, eval_yaml = _setup_advbench_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "steer_sweep.py", "--model-config", str(model_yaml), "--eval-config", str(eval_yaml), "--side", "it",
    ])

    order = []

    def fake_write_run_metadata(*a, **k):
        order.append("write_run_metadata")
        raise _MetadataSentinel()

    monkeypatch.setattr(steer_sweep, "write_run_metadata", fake_write_run_metadata)
    monkeypatch.setattr(steer_sweep, "setup_logging", lambda *a, **k: order.append("setup_logging"))

    with pytest.raises(_MetadataSentinel):
        steer_sweep.main()

    assert order == ["write_run_metadata"]  # setup_logging never ran


def test_a_metadata_mismatch_prevents_all_later_side_effects(tmp_path, monkeypatch):
    model_yaml, eval_yaml = _setup_advbench_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "steer_sweep.py", "--model-config", str(model_yaml), "--eval-config", str(eval_yaml), "--side", "it",
    ])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)
    _forbid(monkeypatch, "setup_logging")
    _forbid(monkeypatch, "load_tokenizer")
    _forbid(monkeypatch, "load_model")
    _forbid(monkeypatch, "sync_to_hub")

    with pytest.raises(_MetadataSentinel):
        steer_sweep.main()

    assert len(calls) == 1  # metadata was attempted exactly once, then main() stopped
