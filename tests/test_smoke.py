"""Smoke tests: minimal sanity checks for data loaders and config parsing.

Run with:
    pytest tests/test_smoke.py -v

These tests do NOT require GPU or a running judge server.
The judge connection test is skipped unless a server is available.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from steering.config import ITEvalConfig, ModelConfig
from steering.data import (
    load_advbench,
    load_alpacaeval,
    load_harmfulqa,
    load_hh_rlhf_test,
    load_ifeval,
)
from steering.utils import read_jsonl, write_jsonl, load_yaml


REPO_ROOT = Path(__file__).parent.parent


# Config


def test_model_config_loads():
    cfg = ModelConfig.from_yaml(REPO_ROOT / "configs" / "llama3_oh.yaml")
    assert cfg.name == "llama3-oh"
    assert cfg.architecture == "llama"
    assert cfg.num_layers == 32
    assert cfg.it_model
    assert cfg.dpo_model


def test_iteval_config_loads():
    """Check the shape of the parsed config, not specific sample counts.

    Those get tuned per run, so asserting a literal here just goes stale.
    """
    cfg = ITEvalConfig.from_yaml(REPO_ROOT / "configs" / "it_eval.yaml")
    assert isinstance(cfg.alpaca_n, int) and cfg.alpaca_n > 0
    assert cfg.max_new_tokens > 0
    assert cfg.judge.model_name.startswith("Qwen")
    assert cfg.judge.server_url.startswith("http")
    # Nested judge settings must survive the YAML round-trip. it_eval.yaml does not
    # override these, so they must resolve to the safe defaults: probability weighting
    # off (Task 005 made the emitted discrete integer the sole authoritative score) and
    # the scale frozen at 10.
    assert cfg.judge.use_logprobs is False
    assert cfg.judge.max_score == 10


# IO helpers


def test_jsonl_roundtrip(tmp_path):
    records = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    path = tmp_path / "test.jsonl"
    write_jsonl(records, path)
    loaded = read_jsonl(path)
    assert loaded == records


# Data loaders (small samples to keep tests fast)


@pytest.mark.parametrize(
    "loader, n",
    [
        (load_ifeval, 5),
        (load_alpacaeval, 5),
        (load_hh_rlhf_test, 5),
        (load_harmfulqa, 5),
        (load_advbench, 5),
    ],
)
def test_loaders_return_prompts(loader, n):
    """Each loader returns at most n records, each with an 'id' and a 'prompt' field."""
    records = loader(n=n)
    assert len(records) <= n
    assert len(records) > 0, f"{loader.__name__} returned zero records"
    for rec in records:
        assert "id" in rec
        assert "prompt" in rec
        assert isinstance(rec["prompt"], (str, list))
        assert rec["prompt"], f"empty prompt in {loader.__name__}"
