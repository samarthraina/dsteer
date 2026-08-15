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

import sys
from pathlib import Path

import pytest

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
