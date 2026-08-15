"""Guards on the manifest-backed HarmfulQA construction path in `layer_profile.py`.

The bug this pins: the active construction script called the ad hoc
`load_harmfulqa(n, seed)` path, so a run's activation tensors could not be tied back to
the frozen manifest. These pin partition enforcement, provenance alignment with the
tensor prompt dimension, and that `hh_rlhf`'s existing diagnostic behavior (including
`response_last`) is untouched.

Run with:
    pytest tests/test_layer_profile.py -v

CPU-only, network-free: the frozen partition loader and HH-RLHF loader are mocked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import layer_profile  # noqa: E402


def _cfg(**overrides):
    return layer_profile.LayerProfileConfig(**overrides)


def _fake_partition_records(n=5, partition="construction", manifest_hash="abc123"):
    return [
        {
            "prompt": f"p{i}", "chosen": None, "source_id": f"harmfulqa-{i}", "prompt_hash": f"hash{i}",
            "partition": partition, "permuted_position": i, "manifest_hash": manifest_hash,
        }
        for i in range(n)
    ]


# validate_harmfulqa_construction_config


def test_construction_config_accepts_the_protocol_settings():
    cfg = _cfg(prompt_source="harmfulqa", prompt_partition="construction", token_position="prompt_last")
    layer_profile.validate_harmfulqa_construction_config(cfg)  # must not raise


def test_construction_config_rejects_a_non_construction_partition():
    cfg = _cfg(prompt_source="harmfulqa", prompt_partition="calibration", token_position="prompt_last")
    with pytest.raises(ValueError):
        layer_profile.validate_harmfulqa_construction_config(cfg)


def test_construction_config_rejects_a_missing_partition():
    cfg = _cfg(prompt_source="harmfulqa", prompt_partition=None, token_position="prompt_last")
    with pytest.raises(ValueError):
        layer_profile.validate_harmfulqa_construction_config(cfg)


def test_construction_config_rejects_response_last():
    cfg = _cfg(prompt_source="harmfulqa", prompt_partition="construction", token_position="response_last")
    with pytest.raises(ValueError):
        layer_profile.validate_harmfulqa_construction_config(cfg)


def test_construction_config_rejects_mixed():
    cfg = _cfg(prompt_source="mixed")
    with pytest.raises(ValueError):
        layer_profile.validate_harmfulqa_construction_config(cfg)


def test_construction_config_leaves_hh_rlhf_diagnostic_behavior_untouched():
    """hh_rlhf + response_last, the existing diagnostic path, must remain unaffected."""
    cfg = _cfg(prompt_source="hh_rlhf", token_position="response_last", prompt_partition=None)
    layer_profile.validate_harmfulqa_construction_config(cfg)  # must not raise


# load_prompts


def test_load_prompts_harmfulqa_requests_exactly_construction_and_returns_everything(monkeypatch):
    fake_records = _fake_partition_records(n=1378)
    calls = []

    def fake_loader(partition):
        calls.append(partition)
        return fake_records

    monkeypatch.setattr(layer_profile, "load_harmfulqa_partition", fake_loader)

    out = layer_profile.load_prompts("harmfulqa", n=5, seed=999, partition="construction")

    assert calls == ["construction"]
    assert len(out) == 1378  # not sliced to n=5
    assert [o["prompt"] for o in out] == [r["prompt"] for r in fake_records]  # order preserved
    for o, r in zip(out, fake_records):
        assert o["source_id"] == r["source_id"]
        assert o["prompt_hash"] == r["prompt_hash"]
        assert o["partition"] == r["partition"]
        assert o["permuted_position"] == r["permuted_position"]
        assert o["manifest_hash"] == r["manifest_hash"]
        assert o["chosen"] is None


def test_load_prompts_harmfulqa_rejects_another_partition(monkeypatch):
    def boom(partition):
        raise AssertionError("load_harmfulqa_partition must not be called for a rejected partition")

    monkeypatch.setattr(layer_profile, "load_harmfulqa_partition", boom)
    with pytest.raises(ValueError):
        layer_profile.load_prompts("harmfulqa", n=5, seed=1, partition="calibration")


def test_load_prompts_harmfulqa_rejects_a_missing_partition(monkeypatch):
    def boom(partition):
        raise AssertionError("load_harmfulqa_partition must not be called for a rejected partition")

    monkeypatch.setattr(layer_profile, "load_harmfulqa_partition", boom)
    with pytest.raises(ValueError):
        layer_profile.load_prompts("harmfulqa", n=5, seed=1, partition=None)


def test_load_prompts_rejects_mixed():
    with pytest.raises(ValueError):
        layer_profile.load_prompts("mixed", n=10, seed=1)


def test_load_prompts_hh_rlhf_unaffected(monkeypatch):
    """Existing hh_rlhf behavior (sampling via n/seed) must be untouched."""
    fake_hh_records = [
        {"id": f"hh-{i}", "prompt": [{"role": "user", "content": f"q{i}"}], "chosen": f"a{i}"}
        for i in range(5)
    ]
    monkeypatch.setattr(layer_profile, "load_hh_rlhf_test", lambda n, seed: fake_hh_records[:n])
    out = layer_profile.load_prompts("hh_rlhf", n=3, seed=7)
    assert len(out) == 3
    assert out[0]["prompt"] == "q0"
    assert out[0]["chosen"] == "a0"


# harmfulqa_provenance / activation-tensor alignment


def test_harmfulqa_provenance_aligns_with_prompt_order():
    prompts = _fake_partition_records(n=5)
    prov = layer_profile.harmfulqa_provenance(prompts)
    assert prov["source_ids"] == [p["source_id"] for p in prompts]
    assert prov["prompt_hashes"] == [p["prompt_hash"] for p in prompts]
    assert prov["partition"] == "construction"
    assert prov["manifest_hash"] == "abc123"


def test_activation_provenance_aligns_with_tensor_prompt_dimension():
    prompts = _fake_partition_records(n=7)
    prov = layer_profile.harmfulqa_provenance(prompts)
    fake_it = torch.zeros(4, len(prompts), 8)  # (layers, prompts, hidden)
    assert len(prov["source_ids"]) == fake_it.shape[1]
    assert len(prov["prompt_hashes"]) == fake_it.shape[1]


def test_harmfulqa_provenance_rejects_mixed_partitions_or_hashes():
    prompts = _fake_partition_records(n=3, partition="construction")
    prompts[1] = dict(prompts[1], partition="calibration")
    with pytest.raises(ValueError):
        layer_profile.harmfulqa_provenance(prompts)


# Output paths


def test_output_root_is_partition_specific_for_harmfulqa():
    root = layer_profile.output_root_for("outputs/layer_profile", "llama3-oh", "harmfulqa", "construction")
    assert root == Path("outputs/layer_profile") / "llama3-oh" / "construction"


def test_output_root_is_unchanged_for_non_harmfulqa_sources():
    root = layer_profile.output_root_for("outputs/layer_profile", "llama3-oh", "hh_rlhf", None)
    assert root == Path("outputs/layer_profile") / "llama3-oh"


# Config file


def test_layer_profile_yaml_parses_to_protocol_values():
    repo_root = Path(__file__).resolve().parents[1]
    cfg = layer_profile.LayerProfileConfig.from_yaml(repo_root / "configs" / "layer_profile.yaml")
    assert cfg.prompt_source == "harmfulqa"
    assert cfg.prompt_partition == "construction"
    assert cfg.token_position == "prompt_last"
    layer_profile.validate_harmfulqa_construction_config(cfg)  # must not raise


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


def test_build_run_config_binds_model_eval_and_full_cli_including_defaults(tmp_path):
    model_yaml = _write_yaml(tmp_path / "model.yaml", _model_cfg_dict())
    eval_yaml = _write_yaml(tmp_path / "eval.yaml", {"prompt_source": "hh_rlhf", "n_prompts": 5})

    model_cfg = layer_profile.ModelConfig.from_yaml(model_yaml)
    eval_cfg = layer_profile.LayerProfileConfig.from_yaml(eval_yaml)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config")
    parser.add_argument("--eval-config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hourly-rate", type=float, default=None)
    args = parser.parse_args(["--model-config", str(model_yaml), "--eval-config", str(eval_yaml)])

    config = layer_profile.build_run_config(model_cfg, eval_cfg, args)

    assert config["model"]["name"] == "tinytest"
    assert config["eval"]["prompt_source"] == "hh_rlhf"
    # every parsed CLI value, including ones never explicitly passed (defaulted).
    assert config["cli"] == {
        "model_config": str(model_yaml), "eval_config": str(eval_yaml),
        "seed": 42, "sync": False, "no_resume": False, "device": "auto", "hourly_rate": None,
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

    monkeypatch.setattr(layer_profile, "write_run_metadata", fake)


def _forbid(monkeypatch, name):
    """Fail loudly if `name` is ever called -- used to prove no side effect happens
    after a simulated metadata mismatch."""
    def boom(*a, **k):
        raise AssertionError(f"{name} must not be called after a metadata identity mismatch")

    monkeypatch.setattr(layer_profile, name, boom)


def _setup_hh_rlhf_run(tmp_path, monkeypatch, extra_eval=None):
    model_yaml = _write_yaml(tmp_path / "model.yaml", _model_cfg_dict())
    eval_cfg_dict = {"prompt_source": "hh_rlhf", "n_prompts": 3, "output_dir": str(tmp_path / "out")}
    eval_cfg_dict.update(extra_eval or {})
    eval_yaml = _write_yaml(tmp_path / "eval.yaml", eval_cfg_dict)

    fake_hh_records = [
        {"id": f"hh-{i}", "prompt": [{"role": "user", "content": f"q{i}"}], "chosen": f"a{i}"}
        for i in range(3)
    ]
    monkeypatch.setattr(layer_profile, "load_hh_rlhf_test", lambda n, seed: fake_hh_records[:n])
    return model_yaml, eval_yaml


def test_argv_is_passed_as_an_exact_json_list_preserving_spaces_and_unicode(tmp_path, monkeypatch):
    model_yaml, eval_yaml = _setup_hh_rlhf_run(tmp_path, monkeypatch)
    argv = [
        "layer_profile.py", "--model-config", str(model_yaml), "--eval-config", str(eval_yaml),
        "--device", "unicode üñíçødé and spaces here",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)

    with pytest.raises(_MetadataSentinel):
        layer_profile.main()

    assert calls[0]["kwargs"]["argv"] == argv
    assert isinstance(calls[0]["kwargs"]["argv"], list)


def test_metadata_config_includes_the_full_parsed_cli_namespace_with_defaults(tmp_path, monkeypatch):
    model_yaml, eval_yaml = _setup_hh_rlhf_run(tmp_path, monkeypatch)
    argv = ["layer_profile.py", "--model-config", str(model_yaml), "--eval-config", str(eval_yaml)]
    monkeypatch.setattr(sys, "argv", argv)

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)

    with pytest.raises(_MetadataSentinel):
        layer_profile.main()

    cli = calls[0]["kwargs"]["config"]["cli"]
    assert cli == {
        "model_config": str(model_yaml), "eval_config": str(eval_yaml),
        "seed": 42, "sync": False, "no_resume": False, "device": "auto", "hourly_rate": None,
    }


def test_metadata_retains_resolved_model_and_eval_configs(tmp_path, monkeypatch):
    model_yaml, eval_yaml = _setup_hh_rlhf_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["layer_profile.py", "--model-config", str(model_yaml), "--eval-config", str(eval_yaml)])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)

    with pytest.raises(_MetadataSentinel):
        layer_profile.main()

    config = calls[0]["kwargs"]["config"]
    assert config["model"]["name"] == "tinytest"
    assert config["model"]["architecture"] == "llama"
    assert config["eval"]["prompt_source"] == "hh_rlhf"


def test_metadata_carries_exact_harmfulqa_provenance(tmp_path, monkeypatch):
    model_yaml = _write_yaml(tmp_path / "model.yaml", _model_cfg_dict())
    eval_yaml = _write_yaml(tmp_path / "eval.yaml", {
        "prompt_source": "harmfulqa", "prompt_partition": "construction",
        "token_position": "prompt_last", "output_dir": str(tmp_path / "out"),
    })
    fake_records = [
        {
            "prompt": f"p{i}", "chosen": None, "source_id": f"harmfulqa-{i}", "prompt_hash": f"hash{i}",
            "partition": "construction", "permuted_position": i, "manifest_hash": "manifest-abc",
        }
        for i in range(4)
    ]
    monkeypatch.setattr(layer_profile, "load_harmfulqa_partition", lambda partition: fake_records)
    monkeypatch.setattr(sys, "argv", ["layer_profile.py", "--model-config", str(model_yaml), "--eval-config", str(eval_yaml)])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)

    with pytest.raises(_MetadataSentinel):
        layer_profile.main()

    extra = calls[0]["kwargs"]["extra"]
    assert extra == {
        "harmfulqa_partition": "construction",
        "harmfulqa_manifest_hash": "manifest-abc",
        "harmfulqa_record_count": 4,
    }


def test_write_run_metadata_is_called_before_setup_logging(tmp_path, monkeypatch):
    model_yaml, eval_yaml = _setup_hh_rlhf_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["layer_profile.py", "--model-config", str(model_yaml), "--eval-config", str(eval_yaml)])

    order = []

    def fake_write_run_metadata(*a, **k):
        order.append("write_run_metadata")
        raise _MetadataSentinel()

    monkeypatch.setattr(layer_profile, "write_run_metadata", fake_write_run_metadata)
    monkeypatch.setattr(layer_profile, "setup_logging", lambda *a, **k: order.append("setup_logging"))

    with pytest.raises(_MetadataSentinel):
        layer_profile.main()

    assert order == ["write_run_metadata"]  # setup_logging never ran


def test_a_metadata_mismatch_prevents_all_later_side_effects(tmp_path, monkeypatch):
    model_yaml, eval_yaml = _setup_hh_rlhf_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["layer_profile.py", "--model-config", str(model_yaml), "--eval-config", str(eval_yaml)])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)
    _forbid(monkeypatch, "setup_logging")
    _forbid(monkeypatch, "load_tokenizer")
    _forbid(monkeypatch, "load_model")
    _forbid(monkeypatch, "sync_to_hub")

    with pytest.raises(_MetadataSentinel):
        layer_profile.main()

    assert len(calls) == 1  # metadata was attempted exactly once, then main() stopped


def test_a_metadata_mismatch_under_no_resume_leaves_a_partial_checkpoint_byte_identical(tmp_path, monkeypatch):
    model_yaml, eval_yaml = _setup_hh_rlhf_run(tmp_path, monkeypatch)
    out_dir = tmp_path / "out" / "tinytest"
    out_dir.mkdir(parents=True)
    partial = out_dir / "it.partial.pt"
    original_bytes = b"not a real checkpoint, just needs to survive untouched"
    partial.write_bytes(original_bytes)

    monkeypatch.setattr(sys, "argv", [
        "layer_profile.py", "--model-config", str(model_yaml), "--eval-config", str(eval_yaml), "--no-resume",
    ])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)

    with pytest.raises(_MetadataSentinel):
        layer_profile.main()

    assert partial.read_bytes() == original_bytes
    assert partial.exists()
