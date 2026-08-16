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

from steering.endpoint_binding import EndpointBindingError  # noqa: E402


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
        "model_config": str(model_yaml),
        "endpoint_manifest": None, "endpoint_bundle_root": None, "pair": None, "endpoint_source": [],
        "eval_config": str(eval_yaml),
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
    assert extra["harmfulqa_partition"] == "construction"
    assert extra["harmfulqa_manifest_hash"] == "manifest-abc"
    assert extra["harmfulqa_record_count"] == 4


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


# main(): endpoint-backed mode (Task 012)


def test_mutually_exclusive_modes_are_rejected_before_any_side_effect(tmp_path, monkeypatch):
    model_yaml, eval_yaml = _setup_hh_rlhf_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "layer_profile.py", "--model-config", str(model_yaml), "--eval-config", str(eval_yaml),
        "--endpoint-manifest", "whatever.json",
    ])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)
    _forbid(monkeypatch, "setup_logging")
    _forbid(monkeypatch, "load_tokenizer")
    _forbid(monkeypatch, "load_model")

    with pytest.raises(EndpointBindingError, match="mutually exclusive"):
        layer_profile.main()

    assert calls == []  # metadata was never even attempted


def test_incomplete_endpoint_mode_is_rejected_before_any_side_effect(tmp_path, monkeypatch):
    model_yaml, eval_yaml = _setup_hh_rlhf_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "layer_profile.py", "--eval-config", str(eval_yaml),
        "--endpoint-manifest", str(tmp_path / "m.json"), "--pair", "A",
        # --endpoint-bundle-root and --endpoint-source are both missing.
    ])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)
    _forbid(monkeypatch, "setup_logging")
    _forbid(monkeypatch, "load_tokenizer")
    _forbid(monkeypatch, "load_model")

    with pytest.raises(EndpointBindingError, match="missing required flag"):
        layer_profile.main()

    assert calls == []


def test_no_model_source_is_rejected_before_any_side_effect(tmp_path, monkeypatch):
    _, eval_yaml = _setup_hh_rlhf_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["layer_profile.py", "--eval-config", str(eval_yaml)])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)
    _forbid(monkeypatch, "setup_logging")

    with pytest.raises(EndpointBindingError, match="no model source given"):
        layer_profile.main()

    assert calls == []


def test_endpoint_verification_failure_prevents_all_side_effects(tmp_path, monkeypatch):
    """A candidate manifest that fails hash/structural verification (here: simply does
    not exist) must fail before any output, logging, seeding, or model-loading side
    effect -- exercising the real `resolve_model_source`, not a mock, so the actual
    verification call runs and actually fails. `set_all_seeds` is explicitly forbidden
    here because it touches CUDA (`torch.cuda.is_available()`/`manual_seed_all`), which
    must never happen before endpoint verification completes on a GPU machine."""
    _, eval_yaml = _setup_hh_rlhf_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "layer_profile.py", "--eval-config", str(eval_yaml),
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
    _forbid(monkeypatch, "load_hh_rlhf_test")
    _forbid(monkeypatch, "load_harmfulqa_partition")

    with pytest.raises(Exception):
        layer_profile.main()

    assert calls == []


def test_endpoint_resolution_runs_before_set_all_seeds_and_config_prompt_loading(tmp_path, monkeypatch):
    """Correction round 1: endpoint-mode validation and file verification must run
    before `set_all_seeds` -- not merely before `write_run_metadata` -- since
    `set_all_seeds` can seed CUDA on a GPU machine. Proven by asserting the raised
    error names the (nonexistent) candidate manifest path, not the (also nonexistent,
    but never-reached) eval-config path -- if eval-config loading had run first, the
    failure would instead be a FileNotFoundError for that path."""
    bad_manifest = tmp_path / "does-not-exist.json"
    never_read_eval_config = tmp_path / "should-never-be-read.yaml"  # deliberately never created
    monkeypatch.setattr(sys, "argv", [
        "layer_profile.py", "--eval-config", str(never_read_eval_config),
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
    _forbid(monkeypatch, "load_hh_rlhf_test")
    _forbid(monkeypatch, "load_harmfulqa_partition")

    with pytest.raises(FileNotFoundError) as exc_info:
        layer_profile.main()

    # Compare by filename, not the full path, to sidestep platform-specific backslash
    # escaping in the exception's string form.
    assert bad_manifest.name in str(exc_info.value)
    assert never_read_eval_config.name not in str(exc_info.value)
    assert calls == []


def test_endpoint_metadata_is_merged_into_run_meta_extra(tmp_path, monkeypatch):
    """Wiring check: whatever `resolve_model_source` returns as endpoint metadata must
    reach `write_run_metadata`'s `extra["endpoint"]` unchanged. The resolution logic
    itself is exercised exhaustively in tests/test_endpoint_binding.py; this only
    checks that layer_profile.py threads the result through correctly."""
    model_cfg_dict = _model_cfg_dict(name="endpoint-A-abc123")
    fake_model_cfg = layer_profile.ModelConfig(**model_cfg_dict)
    fake_endpoint_meta = {
        "mode": "endpoint", "pair": "A", "candidate_manifest_hash": "c" * 64,
        "source_manifest_hash": "s" * 64, "roles": {"it": "M0-A", "dpo": "M+-A"},
        "endpoints": {"it": {"role": "M0-A"}, "dpo": {"role": "M+-A"}},
    }
    monkeypatch.setattr(layer_profile, "resolve_model_source", lambda **kw: (fake_model_cfg, fake_endpoint_meta))
    monkeypatch.setattr(layer_profile, "verify_construction_prompt_identity", lambda *a, **k: None)

    # Endpoint-backed activation extraction is only accepted on the frozen primary
    # construction path (Task 014); the fixture must satisfy it for this "does the
    # endpoint metadata reach run_meta.json" wiring check to reach write_run_metadata
    # at all.
    eval_yaml = _write_yaml(tmp_path / "eval.yaml", {
        "prompt_source": "harmfulqa", "prompt_partition": "construction",
        "token_position": "prompt_last", "output_dir": str(tmp_path / "out"),
    })
    fake_records = [
        {
            "prompt": f"p{i}", "chosen": None, "source_id": f"harmfulqa-{i}", "prompt_hash": f"hash{i}",
            "partition": "construction", "permuted_position": i, "manifest_hash": "manifest-abc",
        }
        for i in range(layer_profile.PRIMARY_CONSTRUCTION_RECORD_COUNT)
    ]
    monkeypatch.setattr(layer_profile, "load_harmfulqa_partition", lambda partition: fake_records)
    monkeypatch.setattr(sys, "argv", [
        "layer_profile.py", "--endpoint-manifest", "m.json", "--endpoint-bundle-root", "b",
        "--pair", "A", "--endpoint-source", "pair_a_sft=/x", "--eval-config", str(eval_yaml),
    ])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)

    with pytest.raises(_MetadataSentinel):
        layer_profile.main()

    assert calls[0]["kwargs"]["extra"]["endpoint"] == fake_endpoint_meta
    assert calls[0]["kwargs"]["extra"]["protocol_profile"] == "primary_v1"
    assert calls[0]["kwargs"]["config"]["model"]["name"] == "endpoint-A-abc123"


# Protocol-profile classification: primary construction requirements (Task 014)


def _fake_construction_manifest(n: int):
    return {
        "records": [
            {"source_id": f"harmfulqa-{i}", "prompt_hash": f"hash{i}", "partition": "construction", "permuted_position": i}
            for i in range(n)
        ],
    }


def _valid_construction_prompts(n: int):
    return [
        {
            "prompt": f"p{i}", "chosen": None, "source_id": f"harmfulqa-{i}", "prompt_hash": f"hash{i}",
            "partition": "construction", "permuted_position": i, "manifest_hash": "manifest-abc",
        }
        for i in range(n)
    ]


def _construction_cfg(**overrides):
    d = dict(prompt_source="harmfulqa", prompt_partition="construction", token_position="prompt_last")
    d.update(overrides)
    return layer_profile.LayerProfileConfig(**d)


def _mock_frozen_manifest(monkeypatch, n: int):
    monkeypatch.setattr(layer_profile, "load_manifest", lambda path: _fake_construction_manifest(n))
    monkeypatch.setattr(layer_profile, "validate_manifest_identity", lambda m: None)


# verify_construction_prompt_identity


def test_verify_construction_prompt_identity_accepts_a_correctly_ordered_match(monkeypatch):
    _mock_frozen_manifest(monkeypatch, 5)
    prompts = [{"source_id": f"harmfulqa-{i}", "prompt_hash": f"hash{i}"} for i in range(5)]
    layer_profile.verify_construction_prompt_identity(prompts)  # must not raise


def test_verify_construction_prompt_identity_rejects_reordered_records(monkeypatch):
    _mock_frozen_manifest(monkeypatch, 5)
    prompts = [{"source_id": f"harmfulqa-{i}", "prompt_hash": f"hash{i}"} for i in [0, 2, 1, 3, 4]]
    with pytest.raises(layer_profile.ProtocolProfileError):
        layer_profile.verify_construction_prompt_identity(prompts)


def test_verify_construction_prompt_identity_rejects_a_wrong_prompt_hash(monkeypatch):
    _mock_frozen_manifest(monkeypatch, 5)
    prompts = [{"source_id": f"harmfulqa-{i}", "prompt_hash": ("tampered" if i == 2 else f"hash{i}")} for i in range(5)]
    with pytest.raises(layer_profile.ProtocolProfileError):
        layer_profile.verify_construction_prompt_identity(prompts)


def test_verify_construction_prompt_identity_rejects_a_wrong_source_id(monkeypatch):
    _mock_frozen_manifest(monkeypatch, 5)
    prompts = [{"source_id": ("tampered" if i == 3 else f"harmfulqa-{i}"), "prompt_hash": f"hash{i}"} for i in range(5)]
    with pytest.raises(layer_profile.ProtocolProfileError):
        layer_profile.verify_construction_prompt_identity(prompts)


# classify_activation_profile (phase 2: assumes phase 1 already passed -- record
# count and manifest-verified identity, which need the prompts already loaded)


def test_classify_activation_profile_primary_v1_for_a_conforming_endpoint_backed_run(monkeypatch):
    n = layer_profile.PRIMARY_CONSTRUCTION_RECORD_COUNT
    _mock_frozen_manifest(monkeypatch, n)
    profile = layer_profile.classify_activation_profile(True, _construction_cfg(), _valid_construction_prompts(n))
    assert profile == "primary_v1"


def test_classify_activation_profile_legacy_is_unconditional_regardless_of_settings():
    """A non-endpoint-backed run is always legacy_nonconfirmatory, even with settings
    that would otherwise satisfy (or badly violate) the primary construction path."""
    cfg = _construction_cfg(prompt_source="hh_rlhf", prompt_partition=None, token_position="response_last")
    assert layer_profile.classify_activation_profile(False, cfg, [{"prompt": "x"}]) == "legacy_nonconfirmatory"


# validate_endpoint_backed_construction_config (phase 1: config-only, before
# load_prompts -- source, partition, and token_position need no loaded prompts)


def test_validate_endpoint_backed_construction_config_accepts_a_conforming_config():
    layer_profile.validate_endpoint_backed_construction_config(True, _construction_cfg())  # must not raise


def test_validate_endpoint_backed_construction_config_is_a_noop_for_legacy_runs():
    cfg = layer_profile.LayerProfileConfig(prompt_source="hh_rlhf", prompt_partition=None, token_position="response_last")
    layer_profile.validate_endpoint_backed_construction_config(False, cfg)  # must not raise


def test_validate_endpoint_backed_construction_config_rejects_hh_rlhf_for_endpoint_backed():
    cfg = layer_profile.LayerProfileConfig(prompt_source="hh_rlhf", token_position="prompt_last")
    with pytest.raises(layer_profile.ProtocolProfileError):
        layer_profile.validate_endpoint_backed_construction_config(True, cfg)


def test_validate_endpoint_backed_construction_config_rejects_mixed_for_endpoint_backed():
    cfg = layer_profile.LayerProfileConfig(prompt_source="mixed", token_position="prompt_last")
    with pytest.raises(layer_profile.ProtocolProfileError):
        layer_profile.validate_endpoint_backed_construction_config(True, cfg)


def test_validate_endpoint_backed_construction_config_rejects_a_non_construction_partition_for_endpoint_backed():
    cfg = _construction_cfg(prompt_partition="development")
    with pytest.raises(layer_profile.ProtocolProfileError):
        layer_profile.validate_endpoint_backed_construction_config(True, cfg)


def test_validate_endpoint_backed_construction_config_rejects_response_last_for_endpoint_backed():
    cfg = _construction_cfg(token_position="response_last")
    with pytest.raises(layer_profile.ProtocolProfileError):
        layer_profile.validate_endpoint_backed_construction_config(True, cfg)


@pytest.mark.parametrize("n", [1377, 1379, 100, 0])
def test_classify_activation_profile_rejects_a_wrong_construction_record_count(n):
    with pytest.raises(layer_profile.ProtocolProfileError):
        layer_profile.classify_activation_profile(True, _construction_cfg(), _valid_construction_prompts(n))


def test_classify_activation_profile_rejects_reordered_construction_identities(monkeypatch):
    n = layer_profile.PRIMARY_CONSTRUCTION_RECORD_COUNT
    _mock_frozen_manifest(monkeypatch, n)
    prompts = _valid_construction_prompts(n)
    prompts[0], prompts[1] = prompts[1], prompts[0]  # same set of records, wrong order
    with pytest.raises(layer_profile.ProtocolProfileError):
        layer_profile.classify_activation_profile(True, _construction_cfg(), prompts)


def test_classify_activation_profile_rejects_an_unverified_construction_set(monkeypatch):
    """The frozen manifest disagrees with every record -- simulating a drifted or
    fabricated prompt list that happens to have the right count and partition label."""
    n = layer_profile.PRIMARY_CONSTRUCTION_RECORD_COUNT
    monkeypatch.setattr(layer_profile, "load_manifest", lambda path: _fake_construction_manifest(n))
    monkeypatch.setattr(layer_profile, "validate_manifest_identity", lambda m: None)
    prompts = [
        {
            "prompt": f"p{i}", "chosen": None, "source_id": f"not-the-real-id-{i}", "prompt_hash": f"not-the-real-hash-{i}",
            "partition": "construction", "permuted_position": i, "manifest_hash": "manifest-abc",
        }
        for i in range(n)
    ]
    with pytest.raises(layer_profile.ProtocolProfileError):
        layer_profile.classify_activation_profile(True, _construction_cfg(), prompts)


# main(): fail-before-mutation ordering, metadata, and safe loading (Task 014)


@pytest.mark.parametrize("eval_overrides", [
    {"prompt_source": "hh_rlhf"},
    {"prompt_source": "mixed"},
    {"prompt_source": "harmfulqa", "prompt_partition": "development", "token_position": "prompt_last"},
    {"prompt_source": "harmfulqa", "prompt_partition": "construction", "token_position": "response_last"},
], ids=["hh_rlhf", "mixed", "wrong_partition", "response_last"])
def test_layer_profile_endpoint_backed_protocol_mismatch_fails_before_any_side_effect(tmp_path, monkeypatch, eval_overrides):
    """Phase 1 (validate_endpoint_backed_construction_config) runs before
    load_prompts, so an endpoint-backed HH/mixed/wrong-partition/response_last
    configuration must fail before any prompt loader, output creation, metadata
    writing, logging, checkpoint deletion, or model/tokenizer/GPU access occurs.

    mixed, wrong_partition, and response_last are already rejected earlier still, by
    the pre-existing, endpoint-agnostic validate_harmfulqa_construction_config (a
    ValueError); only hh_rlhf reaches the endpoint-backed phase-1 check itself (a
    ProtocolProfileError). Either is a legitimate fail-before-mutation exit -- the
    forbidden calls below are what actually pins the "no side effects" guarantee.
    """
    fake_model_cfg = layer_profile.ModelConfig(**_model_cfg_dict())
    fake_endpoint_meta = {"mode": "endpoint", "pair": "A"}
    monkeypatch.setattr(layer_profile, "resolve_model_source", lambda **kw: (fake_model_cfg, fake_endpoint_meta))

    eval_cfg_dict = {"n_prompts": 3, "output_dir": str(tmp_path / "out")}
    eval_cfg_dict.update(eval_overrides)
    eval_yaml = _write_yaml(tmp_path / "eval.yaml", eval_cfg_dict)
    monkeypatch.setattr(sys, "argv", [
        "layer_profile.py", "--endpoint-manifest", "m.json", "--endpoint-bundle-root", "b",
        "--pair", "A", "--endpoint-source", "pair_a_sft=/x", "--eval-config", str(eval_yaml),
    ])

    _forbid(monkeypatch, "load_prompts")
    _forbid(monkeypatch, "load_hh_rlhf_test")
    _forbid(monkeypatch, "load_harmfulqa_partition")
    _forbid(monkeypatch, "set_all_seeds")
    _forbid(monkeypatch, "write_run_metadata")
    _forbid(monkeypatch, "setup_logging")
    _forbid(monkeypatch, "load_tokenizer")
    _forbid(monkeypatch, "load_model")

    with pytest.raises((layer_profile.ProtocolProfileError, ValueError)):
        layer_profile.main()


def test_layer_profile_metadata_contains_protocol_profile_and_loading_policy(tmp_path, monkeypatch):
    fake_model_cfg = layer_profile.ModelConfig(**_model_cfg_dict(name="endpoint-A-abc123"))
    fake_endpoint_meta = {"mode": "endpoint", "pair": "A"}
    monkeypatch.setattr(layer_profile, "resolve_model_source", lambda **kw: (fake_model_cfg, fake_endpoint_meta))
    monkeypatch.setattr(layer_profile, "verify_construction_prompt_identity", lambda *a, **k: None)

    n = layer_profile.PRIMARY_CONSTRUCTION_RECORD_COUNT
    fake_records = [
        {"prompt": f"p{i}", "chosen": None, "source_id": f"harmfulqa-{i}", "prompt_hash": f"hash{i}",
         "partition": "construction", "permuted_position": i, "manifest_hash": "manifest-abc"}
        for i in range(n)
    ]
    monkeypatch.setattr(layer_profile, "load_harmfulqa_partition", lambda partition: fake_records)

    eval_yaml = _write_yaml(tmp_path / "eval.yaml", {
        "prompt_source": "harmfulqa", "prompt_partition": "construction",
        "token_position": "prompt_last", "output_dir": str(tmp_path / "out"),
    })
    monkeypatch.setattr(sys, "argv", [
        "layer_profile.py", "--endpoint-manifest", "m.json", "--endpoint-bundle-root", "b",
        "--pair", "A", "--endpoint-source", "pair_a_sft=/x", "--eval-config", str(eval_yaml),
    ])

    calls = []
    _spy_write_run_metadata(monkeypatch, calls)

    with pytest.raises(_MetadataSentinel):
        layer_profile.main()

    extra = calls[0]["kwargs"]["extra"]
    assert extra["protocol_profile"] == "primary_v1"
    assert extra["endpoint_backed"] is True
    assert extra["model_loading_policy"] == {"local_files_only": True, "trust_remote_code": False}


def test_endpoint_backed_run_passes_the_safe_loading_policy_to_tokenizer_and_activation_extraction(tmp_path, monkeypatch):
    fake_model_cfg = layer_profile.ModelConfig(**_model_cfg_dict(name="endpoint-A-abc123"))
    fake_endpoint_meta = {"mode": "endpoint", "pair": "A"}
    monkeypatch.setattr(layer_profile, "resolve_model_source", lambda **kw: (fake_model_cfg, fake_endpoint_meta))
    monkeypatch.setattr(layer_profile, "verify_construction_prompt_identity", lambda *a, **k: None)

    n = layer_profile.PRIMARY_CONSTRUCTION_RECORD_COUNT
    fake_records = [
        {"prompt": f"p{i}", "chosen": None, "source_id": f"harmfulqa-{i}", "prompt_hash": f"hash{i}",
         "partition": "construction", "permuted_position": i, "manifest_hash": "manifest-abc"}
        for i in range(n)
    ]
    monkeypatch.setattr(layer_profile, "load_harmfulqa_partition", lambda partition: fake_records)

    tokenizer_calls = []
    monkeypatch.setattr(layer_profile, "load_tokenizer", lambda *a, **k: tokenizer_calls.append(k) or object())

    extraction_calls = []

    def fake_extract_activations(model_path, subfolder, tokenizer, prompts, cfg, checkpoint_path=None, loading_policy=None):
        extraction_calls.append(loading_policy)
        return torch.zeros(2, len(prompts), 2)

    monkeypatch.setattr(layer_profile, "extract_activations", fake_extract_activations)

    eval_yaml = _write_yaml(tmp_path / "eval.yaml", {
        "prompt_source": "harmfulqa", "prompt_partition": "construction",
        "token_position": "prompt_last", "output_dir": str(tmp_path / "out"),
    })
    monkeypatch.setattr(sys, "argv", [
        "layer_profile.py", "--endpoint-manifest", "m.json", "--endpoint-bundle-root", "b",
        "--pair", "A", "--endpoint-source", "pair_a_sft=/x", "--eval-config", str(eval_yaml),
    ])

    layer_profile.main()

    assert len(tokenizer_calls) == 1
    assert tokenizer_calls[0]["local_files_only"] is True
    assert tokenizer_calls[0]["trust_remote_code"] is False
    assert extraction_calls == [
        {"local_files_only": True, "trust_remote_code": False},
        {"local_files_only": True, "trust_remote_code": False},
    ]


def test_legacy_run_passes_the_historical_permissive_policy_to_tokenizer_and_activation_extraction(tmp_path, monkeypatch):
    """Legacy behavior remains usable, and is explicitly non-confirmatory -- the loading
    policy stays the historical, more permissive default, unchanged."""
    model_yaml, eval_yaml = _setup_hh_rlhf_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["layer_profile.py", "--model-config", str(model_yaml), "--eval-config", str(eval_yaml)])

    tokenizer_calls = []
    monkeypatch.setattr(layer_profile, "load_tokenizer", lambda *a, **k: tokenizer_calls.append(k) or object())

    extraction_calls = []

    def fake_extract_activations(model_path, subfolder, tokenizer, prompts, cfg, checkpoint_path=None, loading_policy=None):
        extraction_calls.append(loading_policy)
        return torch.zeros(2, len(prompts), 2)

    monkeypatch.setattr(layer_profile, "extract_activations", fake_extract_activations)

    calls = []
    _spy_write_run_metadata(monkeypatch, calls, raise_sentinel=False)

    layer_profile.main()

    assert len(tokenizer_calls) == 1
    assert tokenizer_calls[0]["local_files_only"] is False
    assert tokenizer_calls[0]["trust_remote_code"] is True
    assert extraction_calls == [
        {"local_files_only": False, "trust_remote_code": True},
        {"local_files_only": False, "trust_remote_code": True},
    ]
    assert calls[0]["kwargs"]["extra"]["protocol_profile"] == "legacy_nonconfirmatory"
    assert calls[0]["kwargs"]["extra"]["endpoint_backed"] is False


def test_summary_write_uses_utf8_so_the_real_arrow_character_round_trips(tmp_path):
    """Pins the fix for a real bug: summary.txt.write_text(summary) with no explicit
    encoding used the Windows cp1252 codepage, which cannot represent the U+2192 arrow
    that summarize_findings's real output always contains -- crashing main() on
    Windows the first time a test exercised its summary-writing step all the way
    through. The fix is an explicit encoding="utf-8", not stubbing the summary away."""
    stats_df = layer_profile.compute_layer_stats(torch.randn(4, 5, 8), torch.randn(4, 5, 8))
    summary = layer_profile.summarize_findings(stats_df, num_layers=4)
    assert "→" in summary

    path = tmp_path / "summary.txt"
    path.write_text(summary, encoding="utf-8")
    assert path.read_text(encoding="utf-8") == summary
