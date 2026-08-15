"""Guards on `scripts/generation_smoke.py` (Task 013, protocol Sections 5, 13 Gate 2, 14).

CPU-only, offline: a synthetic (but real-hash-verified) candidate/source manifest pair,
tiny real local files, and fake tokenizer/model/generation objects -- no real model, no
GPU, no judge, no network. `generate_batched` itself is already exhaustively tested in
tests/test_generate.py; here it is mocked so these tests focus on orchestration: role
order, record selection, verification-before-side-effects ordering, sequential
load/release, metadata completeness, atomic output, and Gate 2 pass/fail logic.

Run with:
    pytest tests/test_generation_smoke.py -v
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import generation_smoke  # noqa: E402

from steering import artifact_identity as art_id  # noqa: E402
from steering import endpoint_manifest as em  # noqa: E402
from steering.endpoint_binding import ALL_ROLES, EndpointBindingError  # noqa: E402
from steering.generate import GenerationResult  # noqa: E402


@pytest.fixture(autouse=True)
def _bypass_real_frozen_identity(monkeypatch):
    """Every test builds its own synthetic (but internally consistent,
    real-hash-verified) source manifest -- it will never match the real pinned
    identity, which is exercised separately in tests/test_artifact_identity.py."""
    import steering.endpoint_binding as eb
    monkeypatch.setattr(eb, "validate_frozen_source_identity", lambda m: None)


def _hex64(fill: str = "a") -> str:
    return fill * 64


def _hex40(fill: str = "2") -> str:
    return fill * 40


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_file(root: Path, rel_path: str, content: bytes) -> Dict[str, object]:
    full = root / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(content)
    return {"path": rel_path, "sha256": _sha256(content), "size_bytes": len(content)}


_A_ARCH = {"model_type": "llama", "hidden_size": 8, "num_hidden_layers": 2, "vocab_size": 12, "embedding_rows": 12, "lm_head_rows": 12}
_B_ARCH = {"model_type": "mistral", "hidden_size": 8, "num_hidden_layers": 2, "vocab_size": 16, "embedding_rows": 16, "lm_head_rows": 16}
_A_FP = _hex64("a")
_A_SPECIAL = {"bos_token_id": 1}
_B_FP = _hex64("b")
_B_SPECIAL = {"bos_token_id": 1}


def _validation(arch, fp, special):
    return {
        **arch, "tied_embeddings": False, "tokenizer_loadable": True, "no_residual_peft_modules": True,
        "forward_pass_smoke_test": True, "tokenizer_fingerprint": fp, "special_token_ids": special,
    }


def _tr_base_only(vocab_size, fp, special):
    return {
        "status": "base_only", "base_vocab_size": vocab_size, "adapter_vocab_size": None,
        "old_vocab_size": None, "new_vocab_size": None, "added_token_ids": [],
        "base_fingerprint": fp, "adapter_fingerprint": None,
        "base_special_token_ids": special, "adapter_special_token_ids": None,
    }


def build_gate2_env(tmp_path: Path):
    """Real local roots for all five roles (three direct source roots, plus a bundle
    root for the two merged endpoints), a candidate endpoint manifest, and a matching
    synthetic frozen source manifest. Returns a dict with everything a test needs:
    manifest_path, source_roots (only the three direct artifact IDs), bundle_root,
    source_manifest_path, candidate, and path_to_role (local path -> role, for fake
    load_tokenizer/load_model to identify which role they were asked to load).
    """
    a_sft_root = tmp_path / "src" / "pair_a_sft"
    a_dpo_root = tmp_path / "src" / "pair_a_dpo"
    b_sft_root = tmp_path / "src" / "pair_b_sft"

    a_sft_files = [_write_file(a_sft_root, "model.safetensors", b"a-sft-weights")]
    a_dpo_files = [_write_file(a_dpo_root, "model.safetensors", b"a-dpo-weights")]
    b_sft_files = [_write_file(b_sft_root, "model.safetensors", b"b-sft-weights")]

    bundle_root = tmp_path / "bundle"
    mma_files = [_write_file(bundle_root / "M--A", "model.safetensors", b"a-flip-weights")]
    mpb_files = [_write_file(bundle_root / "M+-B", "model.safetensors", b"b-dpo-weights")]

    mma_anchor = {"path": "adapter_model.safetensors", "sha256": _hex64("f"), "size_bytes": 10}
    mpb_anchor = {"path": "adapter_model.safetensors", "sha256": _hex64("e"), "size_bytes": 10}

    source_artifacts = [
        {"artifact_id": "pair_a_sft", "repository_type": "model", "repository": "org/pair-a",
         "revision": _hex40("2"), "subpath": "SFT_merged", "kind": "checkpoint", "base_artifact_id": None,
         "inference_ready": True, "lineage": {}, "files": a_sft_files},
        {"artifact_id": "pair_a_dpo", "repository_type": "model", "repository": "org/pair-a",
         "revision": _hex40("2"), "subpath": "DPO_merged", "kind": "checkpoint", "base_artifact_id": None,
         "inference_ready": True, "lineage": {}, "files": a_dpo_files},
        {"artifact_id": "pair_b_sft", "repository_type": "model", "repository": "org/pair-b",
         "revision": _hex40("3"), "subpath": "", "kind": "checkpoint", "base_artifact_id": None,
         "inference_ready": True, "lineage": {}, "files": b_sft_files},
        {"artifact_id": "pair_a_flip_adapter", "repository_type": "dataset", "repository": "org/results",
         "revision": _hex40("4"), "subpath": "runs/a-flip", "kind": "adapter", "base_artifact_id": "pair_a_sft",
         "inference_ready": False, "lineage": {}, "files": [mma_anchor]},
        {"artifact_id": "pair_b_dpo_adapter", "repository_type": "dataset", "repository": "org/results",
         "revision": _hex40("4"), "subpath": "runs/b-dpo", "kind": "adapter", "base_artifact_id": "pair_b_sft",
         "inference_ready": False, "lineage": {}, "files": [mpb_anchor]},
    ]
    source_manifest = art_id.build_manifest(source_artifacts)
    source_manifest_path = tmp_path / "model_artifacts_v1.json"
    art_id.save_manifest(source_manifest, source_manifest_path)
    source_manifest_hash = source_manifest["manifest_hash"]

    endpoints = [
        {"role": "M0-A", "status": "direct", "source_manifest_hash": source_manifest_hash,
         "source_artifact_id": "pair_a_sft", "base_artifact_id": None,
         "location": {"kind": "source", "repository": "org/pair-a", "revision": _hex40("2"), "subpath": "SFT_merged"},
         "merge": None, "device": "cpu", "library_versions": {"torch": "0.0.0"},
         "validation": _validation(_A_ARCH, _A_FP, _A_SPECIAL), "files": a_sft_files},
        {"role": "M+-A", "status": "direct", "source_manifest_hash": source_manifest_hash,
         "source_artifact_id": "pair_a_dpo", "base_artifact_id": None,
         "location": {"kind": "source", "repository": "org/pair-a", "revision": _hex40("2"), "subpath": "DPO_merged"},
         "merge": None, "device": "cpu", "library_versions": {"torch": "0.0.0"},
         "validation": _validation(_A_ARCH, _A_FP, _A_SPECIAL), "files": a_dpo_files},
        {"role": "M--A", "status": "merged", "source_manifest_hash": source_manifest_hash,
         "source_artifact_id": "pair_a_flip_adapter", "base_artifact_id": "pair_a_sft",
         "location": {"kind": "bundle", "path": "M--A"}, "device": "cpu",
         "merge": {
             "dtype": "bfloat16", "max_shard_size": "5GB", "merge_seed": 1,
             "tokenizer_resolution": _tr_base_only(_A_ARCH["vocab_size"], _A_FP, _A_SPECIAL),
             "adapter_input_files": [dict(mma_anchor)],
             "flip_lineage": {
                 "archived_trainer_state_path": "runs/x/checkpoint-2510/trainer_state.json",
                 "sha256": _hex64("7"), "training_step": 2510,
                 "training_script": None, "launch_config": None,
                 "label_swap_lineage_verified": False, "confirmatory_eligible": False,
             },
         },
         "library_versions": {"torch": "0.0.0", "peft": "0.0.0"},
         "validation": _validation(_A_ARCH, _A_FP, _A_SPECIAL), "files": mma_files},
        {"role": "M0-B", "status": "direct", "source_manifest_hash": source_manifest_hash,
         "source_artifact_id": "pair_b_sft", "base_artifact_id": None,
         "location": {"kind": "source", "repository": "org/pair-b", "revision": _hex40("3"), "subpath": ""},
         "merge": None, "device": "cpu", "library_versions": {"torch": "0.0.0"},
         "validation": _validation(_B_ARCH, _B_FP, _B_SPECIAL), "files": b_sft_files},
        {"role": "M+-B", "status": "merged", "source_manifest_hash": source_manifest_hash,
         "source_artifact_id": "pair_b_dpo_adapter", "base_artifact_id": "pair_b_sft",
         "location": {"kind": "bundle", "path": "M+-B"}, "device": "cpu",
         "merge": {
             "dtype": "bfloat16", "max_shard_size": "5GB", "merge_seed": 1,
             "tokenizer_resolution": _tr_base_only(_B_ARCH["vocab_size"], _B_FP, _B_SPECIAL),
             "adapter_input_files": [dict(mpb_anchor)],
         },
         "library_versions": {"torch": "0.0.0", "peft": "0.0.0"},
         "validation": _validation(_B_ARCH, _B_FP, _B_SPECIAL), "files": mpb_files},
    ]

    candidate = em.build_manifest(source_manifest_hash, endpoints)
    manifest_path = tmp_path / "endpoint_manifest_candidate_v1.json"
    em.save_manifest(candidate, manifest_path)

    source_roots = {"pair_a_sft": a_sft_root, "pair_a_dpo": a_dpo_root, "pair_b_sft": b_sft_root}
    path_to_role = {
        str(a_sft_root.resolve()): "M0-A", str(a_dpo_root.resolve()): "M+-A",
        str((bundle_root / "M--A").resolve()): "M--A",
        str(b_sft_root.resolve()): "M0-B", str((bundle_root / "M+-B").resolve()): "M+-B",
    }
    return {
        "manifest_path": manifest_path, "source_roots": source_roots, "bundle_root": bundle_root,
        "source_manifest_path": source_manifest_path, "candidate": candidate, "path_to_role": path_to_role,
    }


def _endpoint_source_args(source_roots) -> List[str]:
    return [f"{k}={v}" for k, v in source_roots.items()]


# ---------------------------------------------------------------------------
# Fake tokenizer/model/generation plumbing
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Just enough surface for `generation_terminators` and `build_chat_prompts`."""

    def __init__(self, role, eos_token_id=2, extra_vocab=None):
        self.role = role
        self.eos_token_id = eos_token_id
        self._vocab = dict(extra_vocab if extra_vocab is not None else {"<|eot_id|>": 128009})

    def get_vocab(self):
        return dict(self._vocab)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"[{self.role}]{messages[0]['content']}"


def _clean_result(text="a perfectly ordinary unsteered response with enough words") -> GenerationResult:
    return GenerationResult(text=text, generated_token_count=8, stop_reason="eos_token",
                             stop_token_id=2, has_post_terminator_continuation=False)


def _all_clean_results(n=10):
    return [_clean_result(f"a perfectly ordinary unsteered response number {i}") for i in range(n)]


def _fake_records(n=10, partition="development", manifest_hash="hqa-hash", start=0):
    return [
        {
            "id": f"harmfulqa-{i}", "source_id": f"harmfulqa-{i}", "source_index": i,
            "prompt": f"prompt {i}", "prompt_hash": f"hash-{i}",
            "partition": partition, "permuted_position": start + i, "manifest_hash": manifest_hash,
        }
        for i in range(n)
    ]


def _install_fakes(monkeypatch, scripted_by_role: Dict[str, List[GenerationResult]], path_to_role: Dict[str, str], call_log: list):
    def fake_load_tokenizer(model_path, subfolder="", fallback_pad_token="<|endoftext|>",
                             trust_remote_code=True, local_files_only=False):
        assert trust_remote_code is False
        assert local_files_only is True
        role = path_to_role[model_path]
        call_log.append(f"load_tokenizer:{role}")
        return _FakeTokenizer(role)

    def fake_load_model(model_path, subfolder="", dtype=None, device_map="auto",
                         trust_remote_code=True, local_files_only=False):
        assert trust_remote_code is False
        assert local_files_only is True
        role = path_to_role[model_path]
        call_log.append(f"load_model:{role}")
        return object()

    def fake_generate_batched(model, tokenizer, prompts, max_new_tokens=None, batch_size=None,
                               max_input_length=None, return_metadata=True, **kwargs):
        assert return_metadata is True
        assert max_new_tokens == generation_smoke.SMOKE_MAX_NEW_TOKENS
        assert max_input_length == generation_smoke.SMOKE_MAX_INPUT_LENGTH
        call_log.append(f"generate:{tokenizer.role}")
        return list(scripted_by_role[tokenizer.role])

    monkeypatch.setattr(generation_smoke, "load_tokenizer", fake_load_tokenizer)
    monkeypatch.setattr(generation_smoke, "load_model", fake_load_model)
    monkeypatch.setattr(generation_smoke, "generate_batched", fake_generate_batched)


def _run_full(tmp_path, monkeypatch, env, scripted_by_role=None, records=None, batch_size=None, extra_args=None):
    scripted_by_role = scripted_by_role or {role: _all_clean_results() for role in ALL_ROLES}
    records = records if records is not None else _fake_records()
    call_log = []
    _install_fakes(monkeypatch, scripted_by_role, env["path_to_role"], call_log)
    monkeypatch.setattr(generation_smoke, "load_harmfulqa_partition", lambda partition: records)

    output_dir = tmp_path / "out"
    args = [
        "generation_smoke.py",
        "--endpoint-manifest", str(env["manifest_path"]), "--endpoint-bundle-root", str(env["bundle_root"]),
        *sum((["--endpoint-source", s] for s in _endpoint_source_args(env["source_roots"])), []),
        "--output-dir", str(output_dir),
    ]
    if batch_size is not None:
        args += ["--batch-size", str(batch_size)]
    args += extra_args or []
    monkeypatch.setattr(sys, "argv", args)
    # generation_smoke resolves manifests through the real committed default unless
    # given source_manifest_path -- but resolve_all_roles takes it as an explicit arg,
    # not a CLI flag, so route it through by monkeypatching the module-level default.
    import steering.endpoint_binding as eb
    monkeypatch.setattr(eb, "DEFAULT_SOURCE_MANIFEST_PATH", env["source_manifest_path"])

    rc = generation_smoke.main()
    return rc, output_dir, call_log


def _no_staging_dirs_left(output_dir: Path) -> bool:
    parent = output_dir.parent
    if not parent.exists():
        return True
    return list(parent.glob(f".{output_dir.name}.staging-*")) == []


# Role order and mapping


def test_all_roles_resolved_in_correct_order(tmp_path, monkeypatch):
    env = build_gate2_env(tmp_path)
    rc, output_dir, call_log = _run_full(tmp_path, monkeypatch, env)
    assert rc == 0
    load_calls = [c.split(":")[1] for c in call_log if c.startswith("load_tokenizer:")]
    assert load_calls == list(ALL_ROLES)


def test_role_statuses_are_direct_direct_merged_direct_merged():
    from steering.endpoint_binding import PAIR_ROLES
    assert ALL_ROLES == ("M0-A", "M+-A", "M--A", "M0-B", "M+-B")


# Record selection: exactly the first 10 development records


def test_select_smoke_records_takes_first_10_by_permuted_position():
    shuffled = _fake_records(n=15)
    import random
    rng = random.Random(0)
    rng.shuffle(shuffled)

    selected = generation_smoke.select_smoke_records(shuffled)

    assert len(selected) == 10
    assert [r["permuted_position"] for r in selected] == list(range(10))
    assert [r["source_id"] for r in selected] == [f"harmfulqa-{i}" for i in range(10)]


def test_select_smoke_records_rejects_fewer_than_ten():
    with pytest.raises(generation_smoke.GenerationSmokeError):
        generation_smoke.select_smoke_records(_fake_records(n=5))


def test_full_run_uses_the_development_partition_and_exactly_ten_records(tmp_path, monkeypatch):
    env = build_gate2_env(tmp_path)
    records = _fake_records(n=20)
    calls = []
    orig = generation_smoke.load_harmfulqa_partition

    def spy_loader(partition):
        calls.append(partition)
        return records

    monkeypatch.setattr(generation_smoke, "load_harmfulqa_partition", lambda partition: spy_loader(partition))
    scripted = {role: _all_clean_results() for role in ALL_ROLES}
    call_log = []
    _install_fakes(monkeypatch, scripted, env["path_to_role"], call_log)

    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "generation_smoke.py", "--endpoint-manifest", str(env["manifest_path"]),
        "--endpoint-bundle-root", str(env["bundle_root"]),
        *sum((["--endpoint-source", s] for s in _endpoint_source_args(env["source_roots"])), []),
        "--output-dir", str(output_dir),
    ])
    import steering.endpoint_binding as eb
    monkeypatch.setattr(eb, "DEFAULT_SOURCE_MANIFEST_PATH", env["source_manifest_path"])

    rc = generation_smoke.main()
    assert rc == 0
    assert calls == ["development"]

    for role in ALL_ROLES:
        records_written = _read_jsonl(output_dir / f"{role}.jsonl")
        assert len(records_written) == 10
        assert [r["permuted_position"] for r in records_written] == list(range(10))


# Per-endpoint tokenizer selection


def test_each_role_gets_its_own_tokenizer_object(tmp_path, monkeypatch):
    env = build_gate2_env(tmp_path)
    rc, output_dir, call_log = _run_full(tmp_path, monkeypatch, env)
    assert rc == 0
    # Five distinct load_tokenizer calls, one per role -- not a shared tokenizer reused
    # across roles.
    tokenizer_calls = [c for c in call_log if c.startswith("load_tokenizer:")]
    assert len(tokenizer_calls) == 5
    assert len(set(tokenizer_calls)) == 5


# Endpoint verification before all side effects


def test_endpoint_verification_runs_before_all_side_effects(tmp_path, monkeypatch):
    """A candidate manifest that fails hash/structural verification (here: does not
    exist) must fail before set_all_seeds, dataset loading, output creation, or
    logging -- exercising the real resolve_all_roles, not a mock."""
    bad_manifest = tmp_path / "does-not-exist.json"
    monkeypatch.setattr(sys, "argv", [
        "generation_smoke.py", "--endpoint-manifest", str(bad_manifest),
        "--endpoint-bundle-root", str(tmp_path / "bundle"),
        "--endpoint-source", "pair_a_sft=/x", "--endpoint-source", "pair_a_dpo=/y", "--endpoint-source", "pair_b_sft=/z",
        "--output-dir", str(tmp_path / "out"),
    ])

    def boom(*a, **k):
        raise AssertionError("must not be called before endpoint verification succeeds")

    monkeypatch.setattr(generation_smoke, "set_all_seeds", boom)
    monkeypatch.setattr(generation_smoke, "load_harmfulqa_partition", boom)
    monkeypatch.setattr(generation_smoke, "setup_logging", boom)
    monkeypatch.setattr(generation_smoke, "load_tokenizer", boom)
    monkeypatch.setattr(generation_smoke, "load_model", boom)
    monkeypatch.setattr(generation_smoke, "write_run_metadata", boom)

    with pytest.raises(FileNotFoundError):
        generation_smoke.main()

    assert not (tmp_path / "out").exists()


def test_incomplete_source_mappings_are_rejected_before_side_effects(tmp_path, monkeypatch):
    env = build_gate2_env(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "generation_smoke.py", "--endpoint-manifest", str(env["manifest_path"]),
        "--endpoint-bundle-root", str(env["bundle_root"]),
        "--endpoint-source", "pair_a_sft=/x",  # missing pair_a_dpo and pair_b_sft
        "--output-dir", str(tmp_path / "out"),
    ])
    import steering.endpoint_binding as eb
    monkeypatch.setattr(eb, "DEFAULT_SOURCE_MANIFEST_PATH", env["source_manifest_path"])

    def boom(*a, **k):
        raise AssertionError("must not be called before source-mapping validation succeeds")

    monkeypatch.setattr(generation_smoke, "set_all_seeds", boom)
    monkeypatch.setattr(generation_smoke, "load_harmfulqa_partition", boom)

    with pytest.raises(EndpointBindingError, match="missing"):
        generation_smoke.main()

    assert not (tmp_path / "out").exists()


# Nonempty terminator requirement


def test_a_role_with_no_terminator_blocks_before_any_model_load(tmp_path, monkeypatch):
    env = build_gate2_env(tmp_path)
    call_log = []

    def fake_load_tokenizer(model_path, subfolder="", fallback_pad_token="<|endoftext|>",
                             trust_remote_code=True, local_files_only=False):
        role = env["path_to_role"][model_path]
        call_log.append(f"load_tokenizer:{role}")
        if role == "M--A":
            return _FakeTokenizer(role, eos_token_id=None, extra_vocab={})  # no terminators at all
        return _FakeTokenizer(role)

    def boom_load_model(*a, **k):
        raise AssertionError("load_model must not be called when a terminator is missing")

    monkeypatch.setattr(generation_smoke, "load_tokenizer", fake_load_tokenizer)
    monkeypatch.setattr(generation_smoke, "load_model", boom_load_model)
    monkeypatch.setattr(generation_smoke, "load_harmfulqa_partition", lambda partition: _fake_records())

    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "generation_smoke.py", "--endpoint-manifest", str(env["manifest_path"]),
        "--endpoint-bundle-root", str(env["bundle_root"]),
        *sum((["--endpoint-source", s] for s in _endpoint_source_args(env["source_roots"])), []),
        "--output-dir", str(output_dir),
    ])
    import steering.endpoint_binding as eb
    monkeypatch.setattr(eb, "DEFAULT_SOURCE_MANIFEST_PATH", env["source_manifest_path"])

    rc = generation_smoke.main()
    assert rc == 1
    assert not output_dir.exists()
    assert _no_staging_dirs_left(output_dir)
    assert not any(c.startswith("load_model:") for c in call_log)


# Sequential model load/release


def test_models_are_loaded_and_released_strictly_sequentially(tmp_path, monkeypatch):
    env = build_gate2_env(tmp_path)
    rc, output_dir, call_log = _run_full(tmp_path, monkeypatch, env)
    assert rc == 0

    # All five tokenizers load first (needed for metadata before any model loads),
    # then load_model/generate strictly alternate one role at a time, in ALL_ROLES order.
    tail = [c for c in call_log if c.startswith("load_model:") or c.startswith("generate:")]
    expected = []
    for role in ALL_ROLES:
        expected.append(f"load_model:{role}")
        expected.append(f"generate:{role}")
    assert tail == expected


# Generated-token count, stop metadata, terminator IDs


def test_generated_token_count_and_stop_metadata_are_recorded(tmp_path, monkeypatch):
    env = build_gate2_env(tmp_path)
    scripted = {role: _all_clean_results() for role in ALL_ROLES}
    scripted["M0-A"][3] = GenerationResult(
        text="a perfectly ordinary response with plenty of words in it",
        generated_token_count=42, stop_reason="end_of_turn_token", stop_token_id=128009,
        has_post_terminator_continuation=False,
    )
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch, env, scripted_by_role=scripted)
    assert rc == 0

    records = _read_jsonl(output_dir / "M0-A.jsonl")
    assert records[3]["generated_token_count"] == 42
    assert records[3]["stop_reason"] == "end_of_turn_token"
    assert records[3]["stop_token_id"] == 128009
    assert records[3]["terminator_ids"] == [2, 128009]


# Post-terminator continuation flag propagation


def test_clean_padding_is_not_flagged_as_continuation(tmp_path, monkeypatch):
    env = build_gate2_env(tmp_path)
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch, env)
    assert rc == 0
    for role in ALL_ROLES:
        for rec in _read_jsonl(output_dir / f"{role}.jsonl"):
            assert rec["has_post_terminator_continuation"] is False


def test_real_content_after_terminator_fails_that_roles_gate(tmp_path, monkeypatch):
    env = build_gate2_env(tmp_path)
    scripted = {role: _all_clean_results() for role in ALL_ROLES}
    scripted["M+-B"][0] = GenerationResult(
        text="a response that technically continued past its own stop point",
        generated_token_count=9, stop_reason="eos_token", stop_token_id=2,
        has_post_terminator_continuation=True,
    )
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch, env, scripted_by_role=scripted)
    assert rc == 1

    records = _read_jsonl(output_dir / "M+-B.jsonl")
    assert records[0]["has_post_terminator_continuation"] is True

    summary = json.loads((output_dir / "gate2_summary.json").read_text(encoding="utf-8"))
    assert summary["pass"] is False
    assert summary["roles"]["M+-B"]["n_post_terminator_continuation"] == 1
    assert summary["roles"]["M+-B"]["pass"] is False
    assert summary["roles"]["M0-A"]["pass"] is True  # unaffected roles still pass


# Validity and rendered-prompt hashes


def test_validity_screening_is_applied_and_recorded(tmp_path, monkeypatch):
    env = build_gate2_env(tmp_path)
    scripted = {role: _all_clean_results() for role in ALL_ROLES}
    scripted["M0-B"][5] = GenerationResult(text="", generated_token_count=0, stop_reason="eos_token", stop_token_id=2)
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch, env, scripted_by_role=scripted)
    assert rc == 1

    records = _read_jsonl(output_dir / "M0-B.jsonl")
    assert records[5]["validity_ok"] is False
    assert records[5]["validity_reason"] == "empty"
    for i, rec in enumerate(records):
        if i != 5:
            assert rec["validity_ok"] is True

    summary = json.loads((output_dir / "gate2_summary.json").read_text(encoding="utf-8"))
    assert summary["roles"]["M0-B"]["n_invalid"] == 1
    assert summary["roles"]["M0-B"]["pass"] is False


def test_rendered_prompt_sha256_matches_the_actual_chat_template_output(tmp_path, monkeypatch):
    env = build_gate2_env(tmp_path)
    records_in = _fake_records()
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch, env, records=records_in)
    assert rc == 0

    out_records = _read_jsonl(output_dir / "M0-A.jsonl")
    for rec_in, rec_out in zip(records_in, out_records):
        expected_text = f"[M0-A]{rec_in['prompt']}"
        assert rec_out["rendered_prompt_sha256"] == hashlib.sha256(expected_text.encode("utf-8")).hexdigest()


# Complete metadata


def test_run_metadata_contains_complete_endpoint_and_generation_provenance(tmp_path, monkeypatch):
    env = build_gate2_env(tmp_path)
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch, env, batch_size=7)
    assert rc == 0

    meta = json.loads((output_dir / "run_meta.json").read_text(encoding="utf-8"))

    assert meta["harmfulqa_partition"] == "development"
    assert meta["harmfulqa_manifest_hash"] == "hqa-hash"
    assert [r["source_id"] for r in meta["records"]] == [f"harmfulqa-{i}" for i in range(10)]
    assert [r["permuted_position"] for r in meta["records"]] == list(range(10))

    assert meta["decoding"]["do_sample"] is False
    assert meta["decoding"]["max_new_tokens"] == 512
    assert meta["decoding"]["max_input_length"] == 2048
    assert meta["decoding"]["batch_size"] == 7
    assert meta["decoding"]["steering"] is None
    assert meta["decoding"]["quantization"] is None

    assert meta["endpoint"]["candidate_manifest_hash"] == env["candidate"]["manifest_hash"]
    assert meta["endpoint"]["source_manifest_hash"] == env["candidate"]["source_manifest_hash"]
    assert set(meta["endpoint"]["endpoints"]) == set(ALL_ROLES)
    for role in ALL_ROLES:
        e = meta["endpoint"]["endpoints"][role]
        assert e["role"] == role
        assert "location" in e
        assert "frozen_source" in e

    for role in ALL_ROLES:
        gen = meta["per_endpoint_generation"][role]
        assert gen["terminator_ids"] == [2, 128009]
        assert len(gen["rendered_prompt_sha256"]) == 10

    assert "environment" in meta
    assert meta["config"]["cli"]["batch_size"] == 7
    assert "argv" in meta


# Atomic output behavior


def test_an_existing_output_directory_is_never_overwritten(tmp_path, monkeypatch):
    env = build_gate2_env(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)
    sentinel = output_dir / "keep-me.txt"
    sentinel.write_text("pre-existing content", encoding="utf-8")

    def boom(*a, **k):
        raise AssertionError("must not be called when the output directory already exists")

    monkeypatch.setattr(generation_smoke, "load_harmfulqa_partition", boom)
    monkeypatch.setattr(generation_smoke, "load_tokenizer", boom)
    monkeypatch.setattr(generation_smoke, "load_model", boom)

    monkeypatch.setattr(sys, "argv", [
        "generation_smoke.py", "--endpoint-manifest", str(env["manifest_path"]),
        "--endpoint-bundle-root", str(env["bundle_root"]),
        *sum((["--endpoint-source", s] for s in _endpoint_source_args(env["source_roots"])), []),
        "--output-dir", str(output_dir),
    ])
    import steering.endpoint_binding as eb
    monkeypatch.setattr(eb, "DEFAULT_SOURCE_MANIFEST_PATH", env["source_manifest_path"])

    rc = generation_smoke.main()
    assert rc == 1
    assert sentinel.read_text(encoding="utf-8") == "pre-existing content"


def test_no_staging_directory_remains_after_a_successful_run(tmp_path, monkeypatch):
    env = build_gate2_env(tmp_path)
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch, env)
    assert rc == 0
    assert _no_staging_dirs_left(output_dir)
    assert output_dir.exists()


def test_no_staging_directory_or_output_remains_after_a_generation_failure(tmp_path, monkeypatch):
    env = build_gate2_env(tmp_path)
    scripted = {role: _all_clean_results() for role in ALL_ROLES}
    scripted["M0-A"] = _all_clean_results(9)  # one short -- count mismatch

    rc, output_dir, _ = _run_full(tmp_path, monkeypatch, env, scripted_by_role=scripted)
    assert rc == 1
    assert not output_dir.exists()
    assert _no_staging_dirs_left(output_dir)


def test_jsonl_and_summary_files_are_written_via_atomic_rename(tmp_path, monkeypatch):
    """The atomic-write helper must never leave its own .tmp-* artifact behind."""
    env = build_gate2_env(tmp_path)
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch, env)
    assert rc == 0
    for role in ALL_ROLES:
        assert (output_dir / f"{role}.jsonl").exists()
        assert list(output_dir.glob(f"{role}.jsonl.tmp-*")) == []
    assert (output_dir / "gate2_summary.json").exists()
    assert list(output_dir.glob("gate2_summary.json.tmp-*")) == []


# Pass and fail summaries; nonzero exit on a failed gate


def test_pass_summary_and_zero_exit_code_on_a_clean_run(tmp_path, monkeypatch):
    env = build_gate2_env(tmp_path)
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch, env)
    assert rc == 0

    summary = json.loads((output_dir / "gate2_summary.json").read_text(encoding="utf-8"))
    assert summary["pass"] is True
    for role in ALL_ROLES:
        assert summary["roles"][role]["pass"] is True
        assert summary["roles"][role]["n_records"] == 10


def test_fail_summary_and_nonzero_exit_code_are_both_still_written(tmp_path, monkeypatch):
    env = build_gate2_env(tmp_path)
    scripted = {role: _all_clean_results() for role in ALL_ROLES}
    scripted["M--A"][2] = GenerationResult(text="short but nonempty", generated_token_count=8,
                                            stop_reason="unknown", stop_token_id=None)
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch, env, scripted_by_role=scripted)

    assert rc == 1  # nonzero, but...
    assert output_dir.exists()  # ...the outputs are still fully written and promoted
    for role in ALL_ROLES:
        assert (output_dir / f"{role}.jsonl").exists()
        assert len(_read_jsonl(output_dir / f"{role}.jsonl")) == 10

    summary = json.loads((output_dir / "gate2_summary.json").read_text(encoding="utf-8"))
    assert summary["pass"] is False
    assert summary["roles"]["M--A"]["n_stop_reason_unknown"] == 1
    assert summary["roles"]["M--A"]["pass"] is False
    assert summary["roles"]["M0-A"]["pass"] is True


def test_max_new_tokens_stop_fails_the_gate(tmp_path, monkeypatch):
    env = build_gate2_env(tmp_path)
    scripted = {role: _all_clean_results() for role in ALL_ROLES}
    scripted["M0-B"][0] = GenerationResult(text="reached the budget without stopping cleanly at all",
                                            generated_token_count=512, stop_reason="max_new_tokens", stop_token_id=None)
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch, env, scripted_by_role=scripted)
    assert rc == 1
    summary = json.loads((output_dir / "gate2_summary.json").read_text(encoding="utf-8"))
    assert summary["roles"]["M0-B"]["n_max_new_tokens"] == 1
    assert summary["roles"]["M0-B"]["pass"] is False


# M--A remains confirmatory-ineligible regardless of smoke-test outcome


def test_m_minus_a_remains_confirmatory_ineligible_on_a_passing_run(tmp_path, monkeypatch):
    env = build_gate2_env(tmp_path)
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch, env)
    assert rc == 0

    meta = json.loads((output_dir / "run_meta.json").read_text(encoding="utf-8"))
    flip = meta["endpoint"]["endpoints"]["M--A"]["merge"]["flip_lineage"]
    assert flip["confirmatory_eligible"] is False
    assert flip["label_swap_lineage_verified"] is False


def test_m_minus_a_remains_confirmatory_ineligible_even_when_its_own_gate_fails(tmp_path, monkeypatch):
    env = build_gate2_env(tmp_path)
    scripted = {role: _all_clean_results() for role in ALL_ROLES}
    scripted["M--A"][0] = GenerationResult(text="x", generated_token_count=1, stop_reason="unknown", stop_token_id=None)
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch, env, scripted_by_role=scripted)
    assert rc == 1

    meta = json.loads((output_dir / "run_meta.json").read_text(encoding="utf-8"))
    flip = meta["endpoint"]["endpoints"]["M--A"]["merge"]["flip_lineage"]
    assert flip["confirmatory_eligible"] is False
    assert flip["label_swap_lineage_verified"] is False


# Small local helper (avoids importing steering.utils.read_jsonl just for this)


def _read_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
