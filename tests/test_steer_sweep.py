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

from steering.endpoint_binding import EndpointBindingError
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


def test_run_one_persists_the_post_terminator_continuation_field(tmp_path, monkeypatch):
    """Task 013: `GenerationResult.has_post_terminator_continuation` must reach the
    JSONL output, not only the four original generation metadata fields."""
    records = [{"id": "r1", "prompt": "p1"}, {"id": "r2", "prompt": "p2"}]
    chats = ["chat-1", "chat-2"]
    scripted = [
        GenerationResult(text="clean", generated_token_count=3, stop_reason="eos_token",
                          stop_token_id=2, has_post_terminator_continuation=False),
        GenerationResult(text="anomalous", generated_token_count=3, stop_reason="eos_token",
                          stop_token_id=2, has_post_terminator_continuation=True),
    ]

    monkeypatch.setattr(steer_sweep, "generate_batched", lambda model, tokenizer, todo_chats, **kwargs: list(scripted))

    cfg = _cfg()
    path = tmp_path / "baseline.jsonl"
    steer_sweep.run_one(
        model=None, tokenizer=None, records=records, chats=chats,
        coefficient=0.0, vectors={}, cfg=cfg, path=path, label="test", batch_size=2,
    )

    written = read_jsonl(path)
    assert written[0]["has_post_terminator_continuation"] is False
    assert written[1]["has_post_terminator_continuation"] is True


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


def test_activations_dir_default_matches_the_active_yaml_and_the_manifest_backed_construction_path():
    """The dataclass default must agree with the active YAML -- both must point at the
    manifest-backed construction output path, not the stale
    outputs/layer_profile_response_token."""
    repo_root = Path(__file__).resolve().parents[1]
    yaml_cfg = steer_sweep.SteerSweepConfig.from_yaml(repo_root / "configs" / "steer_sweep.yaml")
    assert steer_sweep.SteerSweepConfig().activations_dir == "outputs/layer_profile"
    assert steer_sweep.SteerSweepConfig().activations_dir == yaml_cfg.activations_dir


# build_run_config


def _model_cfg_dict(**overrides):
    base = {
        "name": "tinytest", "base_model": "org/base", "it_model": "org/it",
        # 32 layers matches both frozen pairs (Task 014's steered_layers(32, 5) ==
        # PRIMARY_LAYERS) -- main() now resolves the effective layer set for protocol
        # classification before write_run_metadata, so a fixture smaller than
        # layers_last_k=5 would fail steered_layers() before these metadata-only tests
        # ever reach the mocked write_run_metadata call.
        "dpo_model": "org/dpo", "architecture": "llama", "num_layers": 32,
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
        "model_config": str(model_yaml),
        "endpoint_manifest": None, "endpoint_bundle_root": None, "pair": None, "endpoint_source": [],
        "eval_config": str(eval_yaml), "side": "it",
        "random_control": False, "random_seed": None, "mode": "add", "tag": None, "vectors": None, "layers": None,
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
    assert extra["harmfulqa_partition"] == "calibration"
    assert extra["harmfulqa_manifest_hash"] == "manifest-abc"
    assert extra["harmfulqa_record_count"] == 4


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


# main(): endpoint-backed mode (Task 012)


def test_mutually_exclusive_modes_are_rejected_before_any_side_effect(tmp_path, monkeypatch):
    model_yaml, eval_yaml = _setup_advbench_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "steer_sweep.py", "--model-config", str(model_yaml), "--eval-config", str(eval_yaml), "--side", "it",
        "--endpoint-manifest", "whatever.json",
    ])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)
    _forbid(monkeypatch, "setup_logging")
    _forbid(monkeypatch, "load_tokenizer")
    _forbid(monkeypatch, "load_model")

    with pytest.raises(EndpointBindingError, match="mutually exclusive"):
        steer_sweep.main()

    assert calls == []


def test_incomplete_endpoint_mode_is_rejected_before_any_side_effect(tmp_path, monkeypatch):
    _, eval_yaml = _setup_advbench_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "steer_sweep.py", "--eval-config", str(eval_yaml), "--side", "it",
        "--endpoint-manifest", str(tmp_path / "m.json"), "--pair", "A",
        # --endpoint-bundle-root and --endpoint-source are both missing.
    ])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)
    _forbid(monkeypatch, "setup_logging")
    _forbid(monkeypatch, "load_tokenizer")
    _forbid(monkeypatch, "load_model")

    with pytest.raises(EndpointBindingError, match="missing required flag"):
        steer_sweep.main()

    assert calls == []


def test_no_model_source_is_rejected_before_any_side_effect(tmp_path, monkeypatch):
    _, eval_yaml = _setup_advbench_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["steer_sweep.py", "--eval-config", str(eval_yaml), "--side", "it"])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)
    _forbid(monkeypatch, "setup_logging")

    with pytest.raises(EndpointBindingError, match="no model source given"):
        steer_sweep.main()

    assert calls == []


def test_endpoint_verification_failure_prevents_all_side_effects(tmp_path, monkeypatch):
    """A candidate manifest that fails hash/structural verification (here: simply does
    not exist) must fail before any output, logging, seeding, or model-loading side
    effect -- exercising the real `resolve_model_source`, not a mock. `set_all_seeds`
    is explicitly forbidden here because it touches CUDA
    (`torch.cuda.is_available()`/`manual_seed_all`), which must never happen before
    endpoint verification completes on a GPU machine."""
    _, eval_yaml = _setup_advbench_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "steer_sweep.py", "--eval-config", str(eval_yaml), "--side", "it",
        "--endpoint-manifest", str(tmp_path / "does-not-exist.json"),
        "--endpoint-bundle-root", str(tmp_path / "bundle"), "--pair", "A",
        "--endpoint-source", "pair_a_sft=/x", "--endpoint-source", "pair_a_dpo=/y",
    ])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)
    _forbid(monkeypatch, "set_all_seeds")
    _forbid(monkeypatch, "setup_logging")
    _forbid(monkeypatch, "load_tokenizer")
    _forbid(monkeypatch, "load_model")
    _forbid(monkeypatch, "load_harmfulqa_partition")

    with pytest.raises(Exception):
        steer_sweep.main()

    assert calls == []


def test_endpoint_resolution_runs_before_set_all_seeds_and_config_prompt_loading(tmp_path, monkeypatch):
    """Correction round 1: endpoint-mode validation and file verification must run
    before `set_all_seeds` -- not merely before `write_run_metadata`. Proven by
    asserting the raised error names the (nonexistent) candidate manifest path, not the
    (also nonexistent, but never-reached) eval-config path -- if eval-config loading
    had run first, the failure would instead be a FileNotFoundError for that path."""
    bad_manifest = tmp_path / "does-not-exist.json"
    never_read_eval_config = tmp_path / "should-never-be-read.yaml"  # deliberately never created
    monkeypatch.setattr(sys, "argv", [
        "steer_sweep.py", "--eval-config", str(never_read_eval_config), "--side", "it",
        "--endpoint-manifest", str(bad_manifest),
        "--endpoint-bundle-root", str(tmp_path / "bundle"), "--pair", "A",
        "--endpoint-source", "pair_a_sft=/x", "--endpoint-source", "pair_a_dpo=/y",
    ])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)
    _forbid(monkeypatch, "set_all_seeds")
    _forbid(monkeypatch, "setup_logging")
    _forbid(monkeypatch, "load_tokenizer")
    _forbid(monkeypatch, "load_model")
    _forbid(monkeypatch, "load_harmfulqa_partition")

    with pytest.raises(FileNotFoundError) as exc_info:
        steer_sweep.main()

    # Compare by filename, not the full path, to sidestep platform-specific backslash
    # escaping in the exception's string form.
    assert bad_manifest.name in str(exc_info.value)
    assert never_read_eval_config.name not in str(exc_info.value)
    assert calls == []


def test_endpoint_metadata_is_merged_into_run_meta_extra(tmp_path, monkeypatch):
    """Wiring check: whatever `resolve_model_source` returns as endpoint metadata must
    reach `write_run_metadata`'s `extra["endpoint"]` unchanged. The resolution logic
    itself is exercised exhaustively in tests/test_endpoint_binding.py; this only
    checks that steer_sweep.py threads the result through correctly."""
    model_cfg_dict = _model_cfg_dict(name="endpoint-A-abc123")
    fake_model_cfg = steer_sweep.ModelConfig(**model_cfg_dict)
    fake_endpoint_meta = {
        "mode": "endpoint", "pair": "A", "candidate_manifest_hash": "c" * 64,
        "source_manifest_hash": "s" * 64, "roles": {"it": "M0-A", "dpo": "M+-A"},
        "endpoints": {"it": {"role": "M0-A"}, "dpo": {"role": "M+-A"}},
    }
    monkeypatch.setattr(steer_sweep, "resolve_model_source", lambda **kw: (fake_model_cfg, fake_endpoint_meta))

    eval_yaml = _write_yaml(tmp_path / "eval.yaml", {
        "prompt_source": "advbench", "n_prompts": 2, "output_dir": str(tmp_path / "out"),
    })
    fake_records = [{"id": "advbench-0", "prompt": "p0"}]
    monkeypatch.setitem(steer_sweep.LOADERS, "advbench", lambda n, seed: fake_records[:n])
    monkeypatch.setattr(sys, "argv", [
        "steer_sweep.py", "--endpoint-manifest", "m.json", "--endpoint-bundle-root", "b",
        "--pair", "A", "--endpoint-source", "pair_a_sft=/x", "--eval-config", str(eval_yaml), "--side", "it",
    ])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)

    with pytest.raises(_MetadataSentinel):
        steer_sweep.main()

    assert calls[0]["kwargs"]["extra"]["endpoint"] == fake_endpoint_meta
    assert calls[0]["kwargs"]["config"]["model"]["name"] == "endpoint-A-abc123"


# Frozen primary steering protocol -- YAML and dataclass defaults (Task 014)


def test_steer_sweep_yaml_matches_the_frozen_primary_protocol():
    repo_root = Path(__file__).resolve().parents[1]
    cfg = steer_sweep.SteerSweepConfig.from_yaml(repo_root / "configs" / "steer_sweep.yaml")
    assert tuple(cfg.lambdas) == steer_sweep.PRIMARY_COEFFICIENTS
    assert cfg.layers_last_k == steer_sweep.PRIMARY_LAYERS_LAST_K
    assert cfg.vector_method == steer_sweep.PRIMARY_VECTOR_METHOD
    assert cfg.vector_normalise == steer_sweep.PRIMARY_VECTOR_NORMALISE
    assert cfg.positions == steer_sweep.PRIMARY_POSITIONS
    assert cfg.preserve_norm == steer_sweep.PRIMARY_PRESERVE_NORM
    assert cfg.max_new_tokens == steer_sweep.PRIMARY_MAX_NEW_TOKENS
    assert cfg.max_input_length == steer_sweep.PRIMARY_MAX_INPUT_LENGTH
    assert cfg.prompt_partition == steer_sweep.PRIMARY_HARMFULQA_PARTITION


def test_steer_sweep_config_dataclass_defaults_match_the_frozen_primary_protocol():
    """Constructing SteerSweepConfig() with no YAML at all must already be the frozen
    primary protocol -- the dataclass defaults and configs/steer_sweep.yaml must never
    silently drift apart."""
    cfg = steer_sweep.SteerSweepConfig()
    assert tuple(cfg.lambdas) == steer_sweep.PRIMARY_COEFFICIENTS
    assert cfg.layers_last_k == steer_sweep.PRIMARY_LAYERS_LAST_K
    assert cfg.vector_method == steer_sweep.PRIMARY_VECTOR_METHOD
    assert cfg.vector_normalise == steer_sweep.PRIMARY_VECTOR_NORMALISE
    assert cfg.positions == steer_sweep.PRIMARY_POSITIONS
    assert cfg.preserve_norm == steer_sweep.PRIMARY_PRESERVE_NORM
    assert cfg.max_new_tokens == steer_sweep.PRIMARY_MAX_NEW_TOKENS
    assert cfg.max_input_length == steer_sweep.PRIMARY_MAX_INPUT_LENGTH
    assert cfg.prompt_partition == steer_sweep.PRIMARY_HARMFULQA_PARTITION


def test_frozen_layers_last_k_resolves_to_the_frozen_primary_layers_on_a_32_layer_model():
    """Both frozen pairs are 32-layer checkpoints; layers_last_k=5 must resolve to
    exactly the protocol's absolute layer indices."""
    assert tuple(steer_sweep.steered_layers(32, steer_sweep.PRIMARY_LAYERS_LAST_K)) == steer_sweep.PRIMARY_LAYERS


# resolve_effective_layers


def test_resolve_effective_layers_uses_layers_last_k_by_default():
    args = argparse.Namespace(layers=None)
    cfg = _cfg(layers_last_k=5)
    model_cfg = steer_sweep.ModelConfig(**_model_cfg_dict())
    assert steer_sweep.resolve_effective_layers(args, cfg, model_cfg) == [27, 28, 29, 30, 31]


def test_resolve_effective_layers_uses_an_explicit_override():
    args = argparse.Namespace(layers="5,10,15")
    cfg = _cfg()
    model_cfg = steer_sweep.ModelConfig(**_model_cfg_dict())
    assert steer_sweep.resolve_effective_layers(args, cfg, model_cfg) == [5, 10, 15]


def test_resolve_effective_layers_rejects_an_out_of_range_layer():
    args = argparse.Namespace(layers="999")
    cfg = _cfg()
    model_cfg = steer_sweep.ModelConfig(**_model_cfg_dict())
    with pytest.raises(ValueError):
        steer_sweep.resolve_effective_layers(args, cfg, model_cfg)


# Protocol-profile classification: pure classify_protocol_profile (Task 014)


def _primary_inputs(**overrides):
    base = dict(
        endpoint_backed=True,
        lambdas=steer_sweep.PRIMARY_COEFFICIENTS, layers=steer_sweep.PRIMARY_LAYERS,
        vector_method=steer_sweep.PRIMARY_VECTOR_METHOD, vector_normalise=steer_sweep.PRIMARY_VECTOR_NORMALISE,
        positions=steer_sweep.PRIMARY_POSITIONS, preserve_norm=steer_sweep.PRIMARY_PRESERVE_NORM,
        max_new_tokens=steer_sweep.PRIMARY_MAX_NEW_TOKENS, max_input_length=steer_sweep.PRIMARY_MAX_INPUT_LENGTH,
        mode="add", random_control=False, random_seed=None,
        external_vectors=False, single_layer_override=False,
    )
    base.update(overrides)
    return steer_sweep.RunProfileInputs(**base)


def _ablation_inputs(**overrides):
    """A conforming secondary_ablation_v1 run: the frozen primary layer set, but the
    single full-removal coefficient instead of the primary grid."""
    return _primary_inputs(mode="ablate", lambdas=steer_sweep.ABLATION_LAMBDAS, **overrides)


def test_conforming_endpoint_backed_run_is_classified_primary_v1():
    assert steer_sweep.classify_protocol_profile(_primary_inputs()) == "primary_v1"


@pytest.mark.parametrize("field,bad_value", [
    ("lambdas", (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)),  # the stale active default
    ("layers", (1, 2, 3, 4, 5)),
    ("vector_method", "pc3"),
    ("vector_normalise", None),
    ("vector_normalise", "unit"),
    ("positions", "new"),
    ("preserve_norm", True),
    ("max_new_tokens", 256),
    ("max_input_length", 1024),
])
def test_each_form_of_primary_protocol_drift_is_rejected(field, bad_value):
    inputs = _primary_inputs(**{field: bad_value})
    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.classify_protocol_profile(inputs)


@pytest.mark.parametrize("random_seed", [11, 22, 33, 44, 55])
def test_random_control_accepts_each_of_the_five_frozen_seeds(random_seed):
    inputs = _primary_inputs(random_control=True, random_seed=random_seed)
    assert steer_sweep.classify_protocol_profile(inputs) == "secondary_random_control_v1"


@pytest.mark.parametrize("random_seed", [0, 1, 12, 42, 100, -11, None])
def test_random_control_rejects_a_random_seed_outside_the_frozen_five(random_seed):
    """random_seed=None (never supplied) must be rejected exactly like an
    unauthorized value -- there is no silent fallback."""
    inputs = _primary_inputs(random_control=True, random_seed=random_seed)
    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.classify_protocol_profile(inputs)


def test_random_control_rejects_a_run_that_does_not_otherwise_preserve_the_frozen_core():
    inputs = _primary_inputs(random_control=True, random_seed=11, layers=(1, 2, 3, 4, 5))
    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.classify_protocol_profile(inputs)


@pytest.mark.parametrize("combo", [
    dict(random_control=True, mode="ablate", random_seed=11, lambdas=steer_sweep.ABLATION_LAMBDAS),
    dict(random_control=True, single_layer_override=True, random_seed=11, layers=(27,)),
    dict(mode="ablate", single_layer_override=True, layers=(27,), lambdas=steer_sweep.ABLATION_LAMBDAS),
])
def test_ambiguous_secondary_combinations_are_rejected(combo):
    inputs = _primary_inputs(**combo)
    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.classify_protocol_profile(inputs)


# Projection ablation: exactly lambdas=[1.0], the frozen primary layer set, and
# otherwise the frozen primary settings -- never the primary coefficient grid, never a
# single (diagnostic) layer, never any other drift.


def test_projection_ablation_with_the_full_removal_coefficient_is_secondary_ablation_v1():
    inputs = _ablation_inputs()
    assert steer_sweep.classify_protocol_profile(inputs) == "secondary_ablation_v1"


def test_projection_ablation_rejects_the_primary_coefficient_grid():
    """The active primary coefficients (0.01-0.10) are an installation/reversal
    sweep, not the predeclared full-projection-removal diagnostic (lambdas=[1.0])."""
    inputs = _primary_inputs(mode="ablate")  # lambdas left at PRIMARY_COEFFICIENTS
    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.classify_protocol_profile(inputs)


def test_projection_ablation_rejects_a_non_frozen_layer_set():
    inputs = _ablation_inputs(layers=(27,))
    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.classify_protocol_profile(inputs)


def test_projection_ablation_rejects_a_non_mean_vector_method():
    """Regression: vector_method="pc3" must never be labelled secondary_ablation_v1 --
    the profile requires the real manifest-backed mean direction."""
    inputs = _ablation_inputs(vector_method="pc3")
    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.classify_protocol_profile(inputs)


def test_projection_ablation_rejects_a_non_relative_normalisation():
    inputs = _ablation_inputs(vector_normalise="unit")
    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.classify_protocol_profile(inputs)


def test_projection_ablation_rejects_changed_positions():
    inputs = _ablation_inputs(positions="new")
    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.classify_protocol_profile(inputs)


def test_projection_ablation_rejects_preserve_norm():
    inputs = _ablation_inputs(preserve_norm=True)
    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.classify_protocol_profile(inputs)


def test_projection_ablation_rejects_changed_generation_limits():
    inputs = _ablation_inputs(max_new_tokens=256)
    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.classify_protocol_profile(inputs)


def test_ablation_is_never_primary_even_with_the_conforming_ablation_settings():
    inputs = _ablation_inputs()
    assert steer_sweep.classify_protocol_profile(inputs) != "primary_v1"


# Single-layer diagnostic: exactly one predeclared diagnostic layer (11, 19, 27, 31),
# and otherwise the frozen primary settings.


@pytest.mark.parametrize("layer", [11, 19, 27, 31])
def test_layer_diagnostic_accepts_each_predeclared_diagnostic_layer(layer):
    inputs = _primary_inputs(single_layer_override=True, layers=(layer,))
    assert steer_sweep.classify_protocol_profile(inputs) == "secondary_layer_diagnostic_v1"


@pytest.mark.parametrize("layer", [0, 5, 10, 12, 20, 26, 28, 31 + 1])
def test_layer_diagnostic_rejects_a_non_predeclared_layer(layer):
    inputs = _primary_inputs(single_layer_override=True, layers=(layer,))
    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.classify_protocol_profile(inputs)


def test_layer_diagnostic_rejects_more_than_one_layer_even_if_all_are_diagnostic_layers():
    """single_layer_override is only ever True for a real single-layer --layers
    override (set by resolve_effective_layers's caller); this directly guards the
    "exactly one" requirement at the classifier level too."""
    inputs = _primary_inputs(single_layer_override=True, layers=(11, 19))
    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.classify_protocol_profile(inputs)


def test_layer_diagnostic_rejects_a_non_mean_vector_method():
    inputs = _primary_inputs(single_layer_override=True, layers=(27,), vector_method="pc3")
    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.classify_protocol_profile(inputs)


def test_layer_diagnostic_rejects_a_non_relative_normalisation():
    inputs = _primary_inputs(single_layer_override=True, layers=(27,), vector_normalise="unit")
    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.classify_protocol_profile(inputs)


def test_layer_diagnostic_rejects_preserve_norm():
    inputs = _primary_inputs(single_layer_override=True, layers=(27,), preserve_norm=True)
    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.classify_protocol_profile(inputs)


def test_layer_diagnostic_rejects_changed_generation_limits():
    inputs = _primary_inputs(single_layer_override=True, layers=(27,), max_input_length=1024)
    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.classify_protocol_profile(inputs)


def test_single_layer_diagnostic_is_never_primary_even_with_otherwise_frozen_settings():
    inputs = _primary_inputs(single_layer_override=True, layers=(27,))
    assert steer_sweep.classify_protocol_profile(inputs) != "primary_v1"


def test_external_vectors_cannot_be_labelled_primary():
    inputs = _primary_inputs(external_vectors=True)
    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.classify_protocol_profile(inputs)


def test_non_mean_vector_method_with_no_secondary_flag_cannot_be_labelled_primary():
    inputs = _primary_inputs(vector_method="pc3")
    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.classify_protocol_profile(inputs)


def test_legacy_runs_are_always_legacy_nonconfirmatory_regardless_of_settings():
    inputs = _primary_inputs(
        endpoint_backed=False, lambdas=(9.9,), layers=(0,),
        mode="ablate", random_control=True, random_seed=1,
    )
    assert steer_sweep.classify_protocol_profile(inputs) == "legacy_nonconfirmatory"


# main(): protocol-profile fail-before-mutation ordering, metadata, and safe loading


def _endpoint_meta():
    return {
        "mode": "endpoint", "pair": "A", "candidate_manifest_hash": "c" * 64,
        "source_manifest_hash": "s" * 64, "roles": {"it": "M0-A", "dpo": "M+-A"},
        "endpoints": {"it": {"role": "M0-A"}, "dpo": {"role": "M+-A"}},
    }


def _mock_endpoint_backed(monkeypatch, num_layers=32, name="endpoint-A-abc123"):
    model_cfg = steer_sweep.ModelConfig(**_model_cfg_dict(name=name, num_layers=num_layers))
    endpoint_meta = _endpoint_meta()
    monkeypatch.setattr(steer_sweep, "resolve_model_source", lambda **kw: (model_cfg, endpoint_meta))
    return model_cfg, endpoint_meta


def _primary_eval_yaml(tmp_path, **overrides):
    d = {
        "prompt_source": "harmfulqa", "prompt_partition": "calibration",
        "lambdas": list(steer_sweep.PRIMARY_COEFFICIENTS), "layers_last_k": 5,
        "vector_method": "mean", "vector_normalise": "relative",
        "positions": "all", "preserve_norm": False,
        "max_new_tokens": 512, "max_input_length": 2048,
        "output_dir": str(tmp_path / "out"),
    }
    d.update(overrides)
    return _write_yaml(tmp_path / "eval.yaml", d)


def _endpoint_backed_argv(eval_yaml, extra=None):
    argv = [
        "steer_sweep.py", "--endpoint-manifest", "m.json", "--endpoint-bundle-root", "b",
        "--pair", "A", "--endpoint-source", "pair_a_sft=/x", "--eval-config", str(eval_yaml), "--side", "it",
    ]
    return argv + (extra or [])


def test_protocol_mismatch_fails_before_any_side_effect(tmp_path, monkeypatch):
    """An endpoint-backed run whose config drifts from the frozen primary protocol,
    using no recognized secondary flag, must fail before prompt loading, output
    creation, metadata writing, logging setup, checkpoint/model/tokenizer loading,
    vector loading, or GPU/CUDA access."""
    _mock_endpoint_backed(monkeypatch)
    eval_yaml = _primary_eval_yaml(tmp_path, lambdas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # the stale active default
    monkeypatch.setattr(sys, "argv", _endpoint_backed_argv(eval_yaml))

    _forbid(monkeypatch, "write_run_metadata")
    _forbid(monkeypatch, "setup_logging")
    _forbid(monkeypatch, "load_tokenizer")
    _forbid(monkeypatch, "load_model")
    _forbid(monkeypatch, "build_vectors")
    _forbid(monkeypatch, "set_all_seeds")
    _forbid(monkeypatch, "load_harmfulqa_partition")

    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.main()


def test_ambiguous_secondary_combination_fails_before_any_side_effect(tmp_path, monkeypatch):
    _mock_endpoint_backed(monkeypatch)
    eval_yaml = _primary_eval_yaml(tmp_path)
    monkeypatch.setattr(sys, "argv", _endpoint_backed_argv(eval_yaml, ["--random-control", "--mode", "ablate", "--random-seed", "11"]))

    _forbid(monkeypatch, "write_run_metadata")
    _forbid(monkeypatch, "setup_logging")
    _forbid(monkeypatch, "load_tokenizer")
    _forbid(monkeypatch, "load_model")
    _forbid(monkeypatch, "set_all_seeds")
    _forbid(monkeypatch, "load_harmfulqa_partition")

    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.main()


def test_external_vectors_on_an_endpoint_backed_run_fail_before_any_side_effect(tmp_path, monkeypatch):
    _mock_endpoint_backed(monkeypatch)
    eval_yaml = _primary_eval_yaml(tmp_path)
    monkeypatch.setattr(sys, "argv", _endpoint_backed_argv(eval_yaml, ["--vectors", str(tmp_path / "learned.pt")]))

    _forbid(monkeypatch, "write_run_metadata")
    _forbid(monkeypatch, "setup_logging")
    _forbid(monkeypatch, "load_tokenizer")
    _forbid(monkeypatch, "load_model")
    _forbid(monkeypatch, "set_all_seeds")
    _forbid(monkeypatch, "load_harmfulqa_partition")

    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.main()


def test_run_metadata_contains_protocol_profile_and_loading_policy(tmp_path, monkeypatch):
    _mock_endpoint_backed(monkeypatch)
    eval_yaml = _primary_eval_yaml(tmp_path)
    fake_records = [
        {"id": f"harmfulqa-{i}", "prompt": f"p{i}", "manifest_hash": "hash-abc", "partition": "calibration"}
        for i in range(5)
    ]
    monkeypatch.setattr(steer_sweep, "load_harmfulqa_partition", lambda partition: fake_records)
    monkeypatch.setattr(sys, "argv", _endpoint_backed_argv(eval_yaml))

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)

    with pytest.raises(_MetadataSentinel):
        steer_sweep.main()

    extra = calls[0]["kwargs"]["extra"]
    assert extra["protocol_profile"] == "primary_v1"
    assert extra["endpoint_backed"] is True
    assert extra["model_loading_policy"] == {"local_files_only": True, "trust_remote_code": False}
    assert extra["resolved_layers"] == [27, 28, 29, 30, 31]
    assert extra["random_control_seed"] is None


def test_run_metadata_records_the_random_control_seed_when_applicable(tmp_path, monkeypatch):
    _mock_endpoint_backed(monkeypatch)
    eval_yaml = _primary_eval_yaml(tmp_path)
    fake_records = [
        {"id": f"harmfulqa-{i}", "prompt": f"p{i}", "manifest_hash": "hash-abc", "partition": "calibration"}
        for i in range(5)
    ]
    monkeypatch.setattr(steer_sweep, "load_harmfulqa_partition", lambda partition: fake_records)
    monkeypatch.setattr(sys, "argv", _endpoint_backed_argv(eval_yaml, ["--random-control", "--random-seed", "11", "--seed", "42"]))

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)

    with pytest.raises(_MetadataSentinel):
        steer_sweep.main()

    extra = calls[0]["kwargs"]["extra"]
    assert extra["protocol_profile"] == "secondary_random_control_v1"
    assert extra["random_control_seed"] == 11
    # the general run seed is recorded independently through the full CLI namespace,
    # and must never be conflated with the dedicated random-direction identity.
    assert calls[0]["kwargs"]["config"]["cli"]["seed"] == 42
    assert calls[0]["kwargs"]["config"]["cli"]["random_seed"] == 11


def test_random_control_without_random_seed_is_rejected_before_any_side_effect(tmp_path, monkeypatch):
    _mock_endpoint_backed(monkeypatch)
    eval_yaml = _primary_eval_yaml(tmp_path)
    monkeypatch.setattr(sys, "argv", _endpoint_backed_argv(eval_yaml, ["--random-control"]))

    _forbid(monkeypatch, "write_run_metadata")
    _forbid(monkeypatch, "setup_logging")
    _forbid(monkeypatch, "load_tokenizer")
    _forbid(monkeypatch, "load_model")
    _forbid(monkeypatch, "set_all_seeds")
    _forbid(monkeypatch, "load_harmfulqa_partition")
    _forbid(monkeypatch, "resolve_model_source")

    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.main()


def test_random_seed_without_random_control_is_rejected_before_any_side_effect(tmp_path, monkeypatch):
    _mock_endpoint_backed(monkeypatch)
    eval_yaml = _primary_eval_yaml(tmp_path)
    monkeypatch.setattr(sys, "argv", _endpoint_backed_argv(eval_yaml, ["--random-seed", "11"]))

    _forbid(monkeypatch, "write_run_metadata")
    _forbid(monkeypatch, "setup_logging")
    _forbid(monkeypatch, "load_tokenizer")
    _forbid(monkeypatch, "load_model")
    _forbid(monkeypatch, "set_all_seeds")
    _forbid(monkeypatch, "load_harmfulqa_partition")
    _forbid(monkeypatch, "resolve_model_source")

    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.main()


@pytest.mark.parametrize("random_seed_a,random_seed_b", [(11, 22)])
def test_different_random_seeds_resolve_to_different_run_directories_while_the_general_seed_is_unchanged(
    tmp_path, monkeypatch, random_seed_a, random_seed_b,
):
    """Regression for the pre-correction bug: all five frozen-seed random controls
    shared one `_random` tag regardless of seed, so they would collide, fail the
    metadata resume check, or silently reuse each other's run directory."""
    parser = _steer_sweep_parser()
    args_a = parser.parse_args(["--side", "it", "--random-control"])
    args_a.random_seed = random_seed_a
    args_a.seed = 42
    args_b = parser.parse_args(["--side", "it", "--random-control"])
    args_b.random_seed = random_seed_b
    args_b.seed = 42

    tag_a = steer_sweep.run_tag("it", args_a)
    tag_b = steer_sweep.run_tag("it", args_b)

    assert tag_a != tag_b
    assert f"_random_s{random_seed_a}" in tag_a
    assert f"_random_s{random_seed_b}" in tag_b
    assert args_a.seed == args_b.seed == 42


def test_validate_random_seed_args_accepts_random_control_with_a_random_seed():
    args = argparse.Namespace(random_control=True, random_seed=11)
    steer_sweep.validate_random_seed_args(args)  # must not raise


def test_validate_random_seed_args_accepts_neither_flag():
    args = argparse.Namespace(random_control=False, random_seed=None)
    steer_sweep.validate_random_seed_args(args)  # must not raise


def test_validate_random_seed_args_rejects_random_control_without_random_seed():
    args = argparse.Namespace(random_control=True, random_seed=None)
    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.validate_random_seed_args(args)


def test_validate_random_seed_args_rejects_random_seed_without_random_control():
    args = argparse.Namespace(random_control=False, random_seed=11)
    with pytest.raises(steer_sweep.ProtocolProfileError):
        steer_sweep.validate_random_seed_args(args)


class _FakeTokenizerForLoadPolicy:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "chat"


def _run_full_for_loading_policy(tmp_path, monkeypatch, endpoint_backed: bool):
    """Runs main() all the way through model/tokenizer loading (with generation itself
    stubbed out), so the exact kwargs load_tokenizer/load_model were called with can be
    inspected -- the only way to prove the *effective* loading policy without loading a
    real model."""
    if endpoint_backed:
        model_cfg, _ = _mock_endpoint_backed(monkeypatch)
        mode_argv = ["--endpoint-manifest", "m.json", "--endpoint-bundle-root", "b",
                     "--pair", "A", "--endpoint-source", "pair_a_sft=/x"]
    else:
        model_yaml = _write_yaml(tmp_path / "model.yaml", _model_cfg_dict())
        model_cfg = steer_sweep.ModelConfig(**_model_cfg_dict())
        mode_argv = ["--model-config", str(model_yaml)]

    eval_yaml = _primary_eval_yaml(tmp_path, activations_dir=str(tmp_path / "acts"))
    fake_records = [
        {"id": f"harmfulqa-{i}", "prompt": f"p{i}", "manifest_hash": "hash-abc", "partition": "calibration"}
        for i in range(3)
    ]
    monkeypatch.setattr(steer_sweep, "load_harmfulqa_partition", lambda partition: fake_records)

    acts_path = tmp_path / "acts" / model_cfg.name / "construction" / "activations.pt"
    acts_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"placeholder": True}, acts_path)
    monkeypatch.setattr(steer_sweep, "validate_construction_activations", lambda *a, **k: None)
    monkeypatch.setattr(steer_sweep, "build_vectors", lambda *a, **k: {l: torch.zeros(4) for l in steer_sweep.PRIMARY_LAYERS})

    load_calls = []

    def fake_load_tokenizer(model_path, subfolder="", **kw):
        load_calls.append(("tokenizer", kw))
        return _FakeTokenizerForLoadPolicy()

    def fake_load_model(model_path, subfolder="", **kw):
        load_calls.append(("model", kw))
        return object()

    monkeypatch.setattr(steer_sweep, "load_tokenizer", fake_load_tokenizer)
    monkeypatch.setattr(steer_sweep, "load_model", fake_load_model)
    monkeypatch.setattr(steer_sweep, "run_one", lambda *a, **k: 0)

    monkeypatch.setattr(sys, "argv", ["steer_sweep.py", *mode_argv, "--eval-config", str(eval_yaml), "--side", "it"])
    steer_sweep.main()
    return load_calls


def test_endpoint_backed_loads_pass_local_files_only_and_no_remote_code(tmp_path, monkeypatch):
    load_calls = _run_full_for_loading_policy(tmp_path, monkeypatch, endpoint_backed=True)
    assert len(load_calls) == 2  # tokenizer + model
    for _, kwargs in load_calls:
        assert kwargs.get("local_files_only") is True
        assert kwargs.get("trust_remote_code") is False


def test_legacy_loads_keep_the_historical_permissive_policy(tmp_path, monkeypatch):
    load_calls = _run_full_for_loading_policy(tmp_path, monkeypatch, endpoint_backed=False)
    assert len(load_calls) == 2
    for _, kwargs in load_calls:
        assert kwargs.get("local_files_only") is False
        assert kwargs.get("trust_remote_code") is True
