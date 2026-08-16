"""Guards on `scripts/development_pilot.py` (Task 015, protocol Sections 6-9, 13 Gate 3).

CPU-only, offline: synthetic (but real-hash-verified) candidate/source/construction
manifests, tiny real local files and tensors, and fake tokenizer/model/generation
objects -- no real model, no GPU, no judge, no network. `generate_batched`,
`ActivationSteering`, and `validate_construction_activations` are each already
exhaustively tested elsewhere; here they are exercised through development_pilot's own
orchestration: record selection, Gate 2/activation-run prerequisite verification,
fail-before-mutation ordering, the frozen arm plan, sequential load/release, judge-ready
schema, the Gate 3 summary/inspection table, and atomic output.

Run with:
    pytest tests/test_development_pilot.py -v
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import development_pilot as dp  # noqa: E402

from steering import artifact_identity as art_id  # noqa: E402
from steering import endpoint_manifest as em  # noqa: E402
from steering.endpoint_binding import ALL_ROLES  # noqa: E402
from steering.generate import GenerationResult  # noqa: E402


@pytest.fixture(autouse=True)
def _bypass_real_frozen_identity(monkeypatch):
    """Every test builds its own synthetic (but internally consistent,
    real-hash-verified) source manifest -- exercised against the real pinned identity
    separately in tests/test_artifact_identity.py."""
    import steering.endpoint_binding as eb
    monkeypatch.setattr(eb, "validate_frozen_source_identity", lambda m: None)


# ---------------------------------------------------------------------------
# Endpoint environment (mirrors tests/test_generation_smoke.py's build_gate2_env)
# ---------------------------------------------------------------------------


def _hex64(fill: str = "a") -> str:
    return fill * 64


def _hex40(fill: str = "2") -> str:
    return fill * 40


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_file(root: Path, rel_path: str, content: bytes) -> Dict[str, object]:
    full = root / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(content)
    return {"path": rel_path, "sha256": _sha256_bytes(content), "size_bytes": len(content)}


_A_ARCH = {"model_type": "llama", "hidden_size": 8, "num_hidden_layers": 32, "vocab_size": 12, "embedding_rows": 12, "lm_head_rows": 12}
_B_ARCH = {"model_type": "mistral", "hidden_size": 8, "num_hidden_layers": 32, "vocab_size": 16, "embedding_rows": 16, "lm_head_rows": 16}
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


def build_pilot_env(tmp_path: Path, tag: str = "env"):
    """Real local roots for all five roles, a candidate endpoint manifest, and a
    matching synthetic frozen source manifest -- everything `resolve_all_roles` needs.
    `tag` lets a test build two distinct (non-matching) environments in one tmp_path.
    """
    root = tmp_path / tag
    a_sft_root = root / "src" / "pair_a_sft"
    a_dpo_root = root / "src" / "pair_a_dpo"
    b_sft_root = root / "src" / "pair_b_sft"

    a_sft_files = [_write_file(a_sft_root, "model.safetensors", f"{tag}-a-sft-weights".encode())]
    a_dpo_files = [_write_file(a_dpo_root, "model.safetensors", f"{tag}-a-dpo-weights".encode())]
    b_sft_files = [_write_file(b_sft_root, "model.safetensors", f"{tag}-b-sft-weights".encode())]

    bundle_root = root / "bundle"
    mma_files = [_write_file(bundle_root / "M--A", "model.safetensors", f"{tag}-a-flip-weights".encode())]
    mpb_files = [_write_file(bundle_root / "M+-B", "model.safetensors", f"{tag}-b-dpo-weights".encode())]

    mma_anchor = {"path": "adapter_model.safetensors", "sha256": _hex64("f"), "size_bytes": 10}
    mpb_anchor = {"path": "adapter_model.safetensors", "sha256": _hex64("e"), "size_bytes": 10}

    source_artifacts = [
        {"artifact_id": "pair_a_sft", "repository_type": "model", "repository": f"org/{tag}-pair-a",
         "revision": _hex40("2"), "subpath": "SFT_merged", "kind": "checkpoint", "base_artifact_id": None,
         "inference_ready": True, "lineage": {}, "files": a_sft_files},
        {"artifact_id": "pair_a_dpo", "repository_type": "model", "repository": f"org/{tag}-pair-a",
         "revision": _hex40("2"), "subpath": "DPO_merged", "kind": "checkpoint", "base_artifact_id": None,
         "inference_ready": True, "lineage": {}, "files": a_dpo_files},
        {"artifact_id": "pair_b_sft", "repository_type": "model", "repository": f"org/{tag}-pair-b",
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
    source_manifest_path = root / "model_artifacts_v1.json"
    art_id.save_manifest(source_manifest, source_manifest_path)
    source_manifest_hash = source_manifest["manifest_hash"]

    endpoints = [
        {"role": "M0-A", "status": "direct", "source_manifest_hash": source_manifest_hash,
         "source_artifact_id": "pair_a_sft", "base_artifact_id": None,
         "location": {"kind": "source", "repository": f"org/{tag}-pair-a", "revision": _hex40("2"), "subpath": "SFT_merged"},
         "merge": None, "device": "cpu", "library_versions": {"torch": "0.0.0"},
         "validation": _validation(_A_ARCH, _A_FP, _A_SPECIAL), "files": a_sft_files},
        {"role": "M+-A", "status": "direct", "source_manifest_hash": source_manifest_hash,
         "source_artifact_id": "pair_a_dpo", "base_artifact_id": None,
         "location": {"kind": "source", "repository": f"org/{tag}-pair-a", "revision": _hex40("2"), "subpath": "DPO_merged"},
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
         "location": {"kind": "source", "repository": f"org/{tag}-pair-b", "revision": _hex40("3"), "subpath": ""},
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
    manifest_path = root / "endpoint_manifest_candidate_v1.json"
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


def _endpoint_argv(env) -> List[str]:
    return [
        "--endpoint-manifest", str(env["manifest_path"]), "--endpoint-bundle-root", str(env["bundle_root"]),
        *sum((["--endpoint-source", s] for s in _endpoint_source_args(env["source_roots"])), []),
    ]


def _patch_default_source_manifest(monkeypatch, env) -> None:
    import steering.endpoint_binding as eb
    monkeypatch.setattr(eb, "DEFAULT_SOURCE_MANIFEST_PATH", env["source_manifest_path"])


def _resolved(env):
    from steering.endpoint_binding import resolve_all_roles
    return resolve_all_roles(env["manifest_path"], env["source_roots"], env["bundle_root"], source_manifest_path=env["source_manifest_path"])


# ---------------------------------------------------------------------------
# HarmfulQA development partition (60 records) and construction manifest/blob fakes
# ---------------------------------------------------------------------------


def _fake_development_records(n: int = 60, manifest_hash: str = "dev-manifest-hash") -> List[Dict[str, Any]]:
    return [
        {
            "id": f"harmfulqa-{i}", "source_id": f"harmfulqa-{i}", "source_index": i,
            "prompt": f"prompt text {i}", "prompt_hash": f"dev-hash-{i}",
            "partition": "development", "permuted_position": i, "manifest_hash": manifest_hash,
        }
        for i in range(n)
    ]


def _fake_construction_manifest(n: int = 1378) -> Dict[str, Any]:
    return {
        "manifest_hash": "construction-manifest-hash",
        "records": [
            {"source_id": f"construction-{i}", "prompt_hash": f"construction-hash-{i}", "partition": "construction", "permuted_position": i}
            for i in range(n)
        ],
    }


def _mock_frozen_construction_manifest(monkeypatch, n: int = 1378) -> None:
    monkeypatch.setattr(dp.activation_artifact, "load_manifest", lambda path: _fake_construction_manifest(n))
    monkeypatch.setattr(dp.activation_artifact, "validate_manifest_identity", lambda m: None)


def _fake_construction_blob(n: int = 1378, num_layers: int = 32, hidden: int = 8) -> Dict[str, Any]:
    return {
        "it": torch.randn(num_layers, n, hidden), "dpo": torch.randn(num_layers, n, hidden),
        "source_ids": [f"construction-{i}" for i in range(n)],
        "prompt_hashes": [f"construction-hash-{i}" for i in range(n)],
        "partition": "construction", "manifest_hash": "construction-manifest-hash",
    }


# ---------------------------------------------------------------------------
# Gate 2 directory and activation-run directory builders
# ---------------------------------------------------------------------------


def _signed_run_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
    """A run_meta.json-shaped payload with a real, correctly-computed
    `run_identity_hash` (the exact algorithm `verify_run_metadata_identity` checks
    against), so fixtures produce metadata the pilot actually trusts by default."""
    payload = dict(payload)
    payload["run_identity_hash"] = dp._compute_run_identity_hash(payload)
    return payload


def _tamper_run_meta_field(path: Path, key: str, value: Any) -> None:
    """Overwrite one field in an on-disk run_meta.json *without* recomputing its
    run_identity_hash -- simulating an edited file, for rejection tests."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[key] = value
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_fake_gate2_dir(
    tmp_path: Path, name: str, env: Dict[str, Any], first_ten: List[Dict[str, Any]],
    resolved, gate_pass: bool = True, role_overrides: Dict[str, List[Dict[str, Any]]] = None,
) -> Path:
    gate2_dir = tmp_path / name
    gate2_dir.mkdir(parents=True)

    roles_summary = {}
    for role in ALL_ROLES:
        roles_summary[role] = {
            "role": role, "n_records": 10, "expected_records": 10,
            "n_stop_reason_unknown": 0, "n_max_new_tokens": 0,
            "n_post_terminator_continuation": 0, "n_invalid": 0, "pass": gate_pass,
        }
    summary = {
        "gate": "gate2_generation_smoke", "pass": gate_pass, "roles": roles_summary,
        "candidate_manifest_hash": resolved.candidate_manifest_hash,
        "source_manifest_hash": resolved.source_manifest_hash,
    }
    (gate2_dir / "gate2_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    role_overrides = role_overrides or {}
    for role in ALL_ROLES:
        records = role_overrides.get(role)
        if records is None:
            records = [
                {
                    "source_id": r["source_id"], "prompt_hash": r["prompt_hash"], "permuted_position": r["permuted_position"],
                    "harmfulqa_manifest_hash": r["manifest_hash"], "harmfulqa_partition": "development",
                    "endpoint_role": role, "rendered_prompt_sha256": "x", "response": "a clean unsteered response with enough words",
                    "generated_token_count": 8, "stop_reason": "eos_token", "stop_token_id": 2,
                    "terminator_ids": [2], "has_post_terminator_continuation": False,
                    "validity_ok": True, "validity_reason": None, "validity_repetition": 0.0,
                }
                for r in first_ten
            ]
        text = "".join(json.dumps(r) + "\n" for r in records)
        (gate2_dir / f"{role}.jsonl").write_text(text, encoding="utf-8")

    run_meta = _signed_run_meta({
        "harmfulqa_partition": "development", "harmfulqa_manifest_hash": first_ten[0]["manifest_hash"],
        "endpoint": resolved.run_metadata(),
    })
    (gate2_dir / "run_meta.json").write_text(json.dumps(run_meta), encoding="utf-8")
    return gate2_dir


def _build_matching_sidecar(run_dir: Path, blob: Dict[str, Any], run_meta: Dict[str, Any]) -> Dict[str, Any]:
    """A sidecar built to exactly match whatever `run_meta`/`blob` claim -- including a
    deliberately "wrong" claim a fixture is testing rejection of -- so a test isolates
    exactly the field it varies rather than tripping a different (missing-file/
    self-hash) check first. Uses the real production hashing (streamed SHA-256, the
    real sidecar self-hash algorithm) so the sidecar itself is genuinely well-formed
    unless a test explicitly tampers with it afterward."""
    acts_path = run_dir / "activations.pt"
    endpoint = run_meta.get("endpoint") or {}
    roles = endpoint.get("roles") or {}
    sidecar = {
        "schema_version": dp.activation_artifact.SCHEMA_VERSION,
        "artifact_kind": dp.activation_artifact.ARTIFACT_KIND,
        "activation_file": dp.activation_artifact.ACTIVATION_FILENAME,
        "activation_sha256": dp.activation_artifact._stream_sha256(acts_path),
        "activation_size_bytes": acts_path.stat().st_size,
        "run_meta_file": dp.activation_artifact.RUN_META_FILENAME,
        "run_identity_hash": run_meta.get("run_identity_hash"),
        "protocol_profile": run_meta.get("protocol_profile"),
        "endpoint_backed": run_meta.get("endpoint_backed"),
        "endpoint": {
            "pair": endpoint.get("pair"), "candidate_manifest_hash": endpoint.get("candidate_manifest_hash"),
            "source_manifest_hash": endpoint.get("source_manifest_hash"),
            "roles": {"it": roles.get("it"), "dpo": roles.get("dpo")},
        },
        "harmfulqa": {
            "partition": run_meta.get("harmfulqa_partition"), "manifest_hash": run_meta.get("harmfulqa_manifest_hash"),
            "record_count": run_meta.get("harmfulqa_record_count"),
        },
        "tensors": {
            "it_shape": list(blob["it"].shape), "dpo_shape": list(blob["dpo"].shape),
            "it_dtype": str(blob["it"].dtype), "dpo_dtype": str(blob["dpo"].dtype),
        },
    }
    sidecar["manifest_hash"] = dp.activation_artifact._compute_sidecar_hash(sidecar)
    return sidecar


def _write_sidecar(run_dir: Path, sidecar: Dict[str, Any]) -> None:
    (run_dir / dp.activation_artifact.SIDECAR_FILENAME).write_text(json.dumps(sidecar), encoding="utf-8")


def write_fake_activation_run_dir(
    tmp_path: Path, name: str, pair: str, resolved, n: int = 1378,
    protocol_profile: str = "primary_v1", endpoint_backed: bool = True,
    harmfulqa_partition: str = "construction", harmfulqa_record_count: int = None,
    token_position: str = "prompt_last",
    loading_policy: Dict[str, bool] = None, endpoint_pair_override: str = None,
    blob: Dict[str, Any] = None, write_sidecar: bool = True,
    sidecar_overrides: Dict[str, Any] = None, tamper_sidecar_hash: bool = False,
    endpoint_meta_overrides: Dict[str, Any] = None,
) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)

    blob = blob if blob is not None else _fake_construction_blob(n)
    torch.save(blob, run_dir / "activations.pt")

    endpoint_meta = None
    if endpoint_backed:
        from steering.endpoint_binding import roles_for_pair
        it_role, dpo_role = roles_for_pair(pair)
        endpoint_meta = {
            "mode": "endpoint", "pair": endpoint_pair_override or pair,
            "candidate_manifest_hash": resolved.candidate_manifest_hash,
            "source_manifest_hash": resolved.source_manifest_hash,
            "roles": {"it": it_role, "dpo": dpo_role},
            "endpoints": {"it": resolved.roles[it_role].as_metadata(), "dpo": resolved.roles[dpo_role].as_metadata()},
        }
        if endpoint_meta_overrides:
            # Merged into run_meta's *own* endpoint block (not just the sidecar), so
            # run_meta and its sidecar (built from it below) stay mutually consistent
            # with each other -- isolating a mismatch against the caller's *own*
            # resolved endpoint identity, rather than tripping the sidecar-vs-run_meta
            # cross-check first.
            endpoint_meta = dict(endpoint_meta, **endpoint_meta_overrides)

    run_meta = _signed_run_meta({
        "protocol_profile": protocol_profile, "endpoint_backed": endpoint_backed,
        "endpoint": endpoint_meta,
        "harmfulqa_partition": harmfulqa_partition,
        # Matches the blob's own manifest_hash by default, so the fixture's run_meta
        # genuinely describes the activations.pt it sits beside -- required now that
        # the central validator (and publish_activation_artifact) cross-check this
        # against the actual blob, not merely against the sidecar/run_meta pair.
        "harmfulqa_manifest_hash": blob.get("manifest_hash"),
        "harmfulqa_record_count": harmfulqa_record_count if harmfulqa_record_count is not None else n,
        "config": {"eval": {"token_position": token_position}},
        "model_loading_policy": loading_policy if loading_policy is not None else {"local_files_only": True, "trust_remote_code": False},
    })
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta), encoding="utf-8")

    if write_sidecar:
        sidecar = _build_matching_sidecar(run_dir, blob, run_meta)
        if sidecar_overrides:
            sidecar.update(sidecar_overrides)
            if not tamper_sidecar_hash:
                sidecar["manifest_hash"] = dp.activation_artifact._compute_sidecar_hash(sidecar)
        _write_sidecar(run_dir, sidecar)
    return run_dir


# ---------------------------------------------------------------------------
# Fake tokenizer/model/generation plumbing (mirrors tests/test_generation_smoke.py)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    def __init__(self, role, eos_token_id=2, extra_vocab=None):
        self.role = role
        self.eos_token_id = eos_token_id
        self._vocab = dict(extra_vocab if extra_vocab is not None else {"<|eot_id|>": 128009})

    def get_vocab(self):
        return dict(self._vocab)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"[{self.role}]{messages[0]['content']}"


class _FakeHookHandle:
    def remove(self):
        pass


class _FakeDecoderLayer:
    def register_forward_hook(self, fn):
        return _FakeHookHandle()


class _FakeSteerableModel:
    """Just enough surface for `steering.activations.decoder_layers` (a `.layers` list
    of hookable modules) so `ActivationSteering.__enter__`/`__exit__` can run for real
    against a steered arm without a real transformer."""

    def __init__(self, num_layers: int = 32):
        self.layers = [_FakeDecoderLayer() for _ in range(num_layers)]


def _clean_result(text="a perfectly ordinary unsteered response with enough words") -> GenerationResult:
    return GenerationResult(text=text, generated_token_count=8, stop_reason="eos_token",
                             stop_token_id=2, has_post_terminator_continuation=False)


def _clean_results(n: int) -> List[GenerationResult]:
    return [_clean_result(f"a perfectly ordinary unsteered response number {i}") for i in range(n)]


def _install_fakes(monkeypatch, env, call_log: list, results_for_arm=None, results_for_role=None):
    """`results_for_arm`: {arm_id: [GenerationResult,...]} takes priority; falls back to
    `results_for_role`: {role: [GenerationResult,...]}; falls back to all-clean."""

    def fake_load_tokenizer(model_path, subfolder="", fallback_pad_token="<|endoftext|>",
                             trust_remote_code=True, local_files_only=False):
        assert trust_remote_code is False
        assert local_files_only is True
        role = env["path_to_role"][model_path]
        call_log.append(f"load_tokenizer:{role}")
        return _FakeTokenizer(role)

    def fake_load_model(model_path, subfolder="", dtype=None, device_map="auto",
                         trust_remote_code=True, local_files_only=False):
        assert trust_remote_code is False
        assert local_files_only is True
        role = env["path_to_role"][model_path]
        call_log.append(f"load_model:{role}")
        return _FakeSteerableModel()

    def fake_generate_batched(model, tokenizer, prompts, max_new_tokens=None, batch_size=None,
                               max_input_length=None, context=None, desc=None, return_metadata=True, **kwargs):
        assert return_metadata is True
        assert max_new_tokens == dp.PILOT_MAX_NEW_TOKENS
        assert max_input_length == dp.PILOT_MAX_INPUT_LENGTH
        call_log.append(f"generate:{tokenizer.role}:{desc}")
        if context is not None:
            with context():
                pass  # exercise hook registration/removal without a real model
        if results_for_arm and desc in results_for_arm:
            return list(results_for_arm[desc])
        if results_for_role and tokenizer.role in results_for_role:
            return list(results_for_role[tokenizer.role])
        return _clean_results(len(prompts))

    monkeypatch.setattr(dp, "load_tokenizer", fake_load_tokenizer)
    monkeypatch.setattr(dp, "load_model", fake_load_model)
    monkeypatch.setattr(dp, "generate_batched", fake_generate_batched)


def _run_full(
    tmp_path, monkeypatch, env=None, dev_records=None, results_for_arm=None, results_for_role=None,
    gate2_dir=None, acts_a_dir=None, acts_b_dir=None, extra_args=None, output_dir=None,
):
    env = env or build_pilot_env(tmp_path)
    resolved = _resolved(env)
    dev_records = dev_records if dev_records is not None else _fake_development_records()
    first_ten = dp.generation_smoke.select_smoke_records(dev_records)

    gate2_dir = gate2_dir or write_fake_gate2_dir(tmp_path, "gate2", env, first_ten, resolved)

    _mock_frozen_construction_manifest(monkeypatch)
    acts_a_dir = acts_a_dir or write_fake_activation_run_dir(tmp_path, "acts_A", "A", resolved)
    acts_b_dir = acts_b_dir or write_fake_activation_run_dir(tmp_path, "acts_B", "B", resolved)

    call_log: List[str] = []
    _install_fakes(monkeypatch, env, call_log, results_for_arm=results_for_arm, results_for_role=results_for_role)
    monkeypatch.setattr(dp, "load_harmfulqa_partition", lambda partition: dev_records)

    output_dir = output_dir or (tmp_path / "out")
    argv = [
        "development_pilot.py", *_endpoint_argv(env),
        "--gate2-dir", str(gate2_dir), "--pair-a-activation-run", str(acts_a_dir), "--pair-b-activation-run", str(acts_b_dir),
        "--output-dir", str(output_dir),
    ] + (extra_args or [])
    monkeypatch.setattr(sys, "argv", argv)
    _patch_default_source_manifest(monkeypatch, env)

    rc = dp.main()
    return rc, output_dir, call_log


def _read_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ===========================================================================
# 1-4: Pure record selection
# ===========================================================================


def test_select_pilot_records_returns_positions_10_through_59():
    records = _fake_development_records(60)
    pilot = dp.select_pilot_records(records)
    assert len(pilot) == 50
    assert [r["permuted_position"] for r in pilot] == list(range(10, 60))
    assert [r["source_id"] for r in pilot] == [f"harmfulqa-{i}" for i in range(10, 60)]


def test_select_pilot_records_sorts_reordered_input():
    records = _fake_development_records(60)
    import random
    shuffled = list(records)
    random.Random(0).shuffle(shuffled)
    pilot = dp.select_pilot_records(shuffled)
    assert [r["permuted_position"] for r in pilot] == list(range(10, 60))


@pytest.mark.parametrize("n", [59, 61, 0, 10])
def test_select_pilot_records_rejects_wrong_development_size(n):
    with pytest.raises(dp.DevelopmentPilotError):
        dp.select_pilot_records(_fake_development_records(n))


def test_select_pilot_records_rejects_overlap_with_first_ten():
    records = _fake_development_records(60)
    # Corrupt position 15 to duplicate position 3's identity -- a genuine partition
    # would never do this, but the pure function must still catch it defensively.
    records[15] = dict(records[15])
    records[15]["source_id"] = records[3]["source_id"]
    records[15]["prompt_hash"] = records[3]["prompt_hash"]
    with pytest.raises(dp.DevelopmentPilotError):
        dp.select_pilot_records(records)


# ===========================================================================
# 5-7: Gate 2 prerequisite
# ===========================================================================


def test_verify_gate2_prerequisite_accepts_a_conforming_gate2_dir(tmp_path):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    dev_records = _fake_development_records()
    first_ten = dp.generation_smoke.select_smoke_records(dev_records)
    gate2_dir = write_fake_gate2_dir(tmp_path, "gate2", env, first_ten, resolved)
    info = dp.verify_gate2_prerequisite(gate2_dir, resolved, first_ten)
    assert info["summary"]["pass"] is True


def test_verify_gate2_prerequisite_rejects_a_failing_gate2(tmp_path):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    dev_records = _fake_development_records()
    first_ten = dp.generation_smoke.select_smoke_records(dev_records)
    gate2_dir = write_fake_gate2_dir(tmp_path, "gate2", env, first_ten, resolved, gate_pass=False)
    with pytest.raises(dp.DevelopmentPilotError):
        dp.verify_gate2_prerequisite(gate2_dir, resolved, first_ten)


def test_verify_gate2_prerequisite_rejects_candidate_manifest_hash_mismatch(tmp_path):
    env_gate2 = build_pilot_env(tmp_path, tag="gate2env")
    env_pilot = build_pilot_env(tmp_path, tag="pilotenv")
    resolved_gate2 = _resolved(env_gate2)
    resolved_pilot = _resolved(env_pilot)
    dev_records = _fake_development_records()
    first_ten = dp.generation_smoke.select_smoke_records(dev_records)
    gate2_dir = write_fake_gate2_dir(tmp_path, "gate2", env_gate2, first_ten, resolved_gate2)
    with pytest.raises(dp.DevelopmentPilotError):
        dp.verify_gate2_prerequisite(gate2_dir, resolved_pilot, first_ten)


def test_verify_gate2_prerequisite_rejects_source_manifest_hash_mismatch_in_run_meta(tmp_path):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    dev_records = _fake_development_records()
    first_ten = dp.generation_smoke.select_smoke_records(dev_records)
    gate2_dir = write_fake_gate2_dir(tmp_path, "gate2", env, first_ten, resolved)
    run_meta = json.loads((gate2_dir / "run_meta.json").read_text(encoding="utf-8"))
    del run_meta["run_identity_hash"]
    run_meta["endpoint"]["source_manifest_hash"] = "tampered"
    run_meta = _signed_run_meta(run_meta)  # resigned, so this exercises the field check, not the hash check
    (gate2_dir / "run_meta.json").write_text(json.dumps(run_meta), encoding="utf-8")
    with pytest.raises(dp.DevelopmentPilotError):
        dp.verify_gate2_prerequisite(gate2_dir, resolved, first_ten)


def test_verify_gate2_prerequisite_rejects_a_forged_passing_summary_with_a_wrong_role_file(tmp_path):
    """The gate2_summary.json boolean alone is never trusted -- a role file whose
    identities disagree with the frozen first ten must be caught even if the summary
    claims that role passed."""
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    dev_records = _fake_development_records()
    first_ten = dp.generation_smoke.select_smoke_records(dev_records)
    tampered_first_ten = list(first_ten)
    tampered_first_ten[0] = dict(tampered_first_ten[0])
    tampered_first_ten[0]["source_id"] = "not-the-real-id"
    gate2_dir = write_fake_gate2_dir(
        tmp_path, "gate2", env, first_ten, resolved,
        role_overrides={"M0-A": [
            {
                "source_id": r["source_id"], "prompt_hash": r["prompt_hash"], "permuted_position": r["permuted_position"],
                "harmfulqa_manifest_hash": r["manifest_hash"], "harmfulqa_partition": "development",
            }
            for r in tampered_first_ten
        ]},
    )
    with pytest.raises(dp.DevelopmentPilotError):
        dp.verify_gate2_prerequisite(gate2_dir, resolved, first_ten)


def test_verify_gate2_prerequisite_rejects_a_missing_role_file(tmp_path):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    dev_records = _fake_development_records()
    first_ten = dp.generation_smoke.select_smoke_records(dev_records)
    gate2_dir = write_fake_gate2_dir(tmp_path, "gate2", env, first_ten, resolved)
    (gate2_dir / "M--A.jsonl").unlink()
    with pytest.raises(dp.DevelopmentPilotError):
        dp.verify_gate2_prerequisite(gate2_dir, resolved, first_ten)


def test_verify_gate2_prerequisite_rejects_a_role_that_used_a_different_partition(tmp_path):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    dev_records = _fake_development_records()
    first_ten = dp.generation_smoke.select_smoke_records(dev_records)
    gate2_dir = write_fake_gate2_dir(
        tmp_path, "gate2", env, first_ten, resolved,
        role_overrides={"M+-B": [
            {
                "source_id": r["source_id"], "prompt_hash": r["prompt_hash"], "permuted_position": r["permuted_position"],
                "harmfulqa_manifest_hash": r["manifest_hash"], "harmfulqa_partition": "calibration",
            }
            for r in first_ten
        ]},
    )
    with pytest.raises(dp.DevelopmentPilotError):
        dp.verify_gate2_prerequisite(gate2_dir, resolved, first_ten)


def test_verify_gate2_prerequisite_rejects_missing_summary_file(tmp_path):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    dev_records = _fake_development_records()
    first_ten = dp.generation_smoke.select_smoke_records(dev_records)
    with pytest.raises(dp.DevelopmentPilotError):
        dp.verify_gate2_prerequisite(tmp_path / "does-not-exist", resolved, first_ten)


# ===========================================================================
# 8-9: Activation-run prerequisite
# ===========================================================================


def test_verify_activation_run_accepts_a_conforming_run(tmp_path, monkeypatch):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    _mock_frozen_construction_manifest(monkeypatch)
    run_dir = write_fake_activation_run_dir(tmp_path, "acts_A", "A", resolved)
    info = dp.verify_activation_run("A", run_dir, resolved)
    assert info["sha256"]
    assert info["size_bytes"] > 0


def test_verify_activation_run_rejects_wrong_pair(tmp_path, monkeypatch):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    _mock_frozen_construction_manifest(monkeypatch)
    run_dir = write_fake_activation_run_dir(tmp_path, "acts_A", "A", resolved, endpoint_pair_override="B")
    with pytest.raises(dp.activation_artifact.ActivationArtifactError):
        dp.verify_activation_run("A", run_dir, resolved)


def test_verify_activation_run_rejects_candidate_manifest_hash_mismatch(tmp_path, monkeypatch):
    env_a = build_pilot_env(tmp_path, tag="a")
    env_b = build_pilot_env(tmp_path, tag="b")
    resolved_a = _resolved(env_a)
    resolved_b = _resolved(env_b)
    _mock_frozen_construction_manifest(monkeypatch)
    run_dir = write_fake_activation_run_dir(tmp_path, "acts_A", "A", resolved_a)
    with pytest.raises(dp.activation_artifact.ActivationArtifactError):
        dp.verify_activation_run("A", run_dir, resolved_b)


def test_verify_activation_run_rejects_non_primary_protocol_profile(tmp_path, monkeypatch):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    _mock_frozen_construction_manifest(monkeypatch)
    run_dir = write_fake_activation_run_dir(tmp_path, "acts_A", "A", resolved, protocol_profile="legacy_nonconfirmatory")
    with pytest.raises(dp.activation_artifact.ActivationArtifactError):
        dp.verify_activation_run("A", run_dir, resolved)


def test_verify_activation_run_rejects_not_endpoint_backed(tmp_path, monkeypatch):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    _mock_frozen_construction_manifest(monkeypatch)
    run_dir = write_fake_activation_run_dir(tmp_path, "acts_A", "A", resolved, endpoint_backed=False)
    with pytest.raises(dp.activation_artifact.ActivationArtifactError):
        dp.verify_activation_run("A", run_dir, resolved)


def test_verify_activation_run_rejects_wrong_partition(tmp_path, monkeypatch):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    _mock_frozen_construction_manifest(monkeypatch)
    run_dir = write_fake_activation_run_dir(tmp_path, "acts_A", "A", resolved, harmfulqa_partition="calibration")
    with pytest.raises(dp.activation_artifact.ActivationArtifactError):
        dp.verify_activation_run("A", run_dir, resolved)


def test_verify_activation_run_rejects_wrong_record_count(tmp_path, monkeypatch):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    _mock_frozen_construction_manifest(monkeypatch)
    run_dir = write_fake_activation_run_dir(tmp_path, "acts_A", "A", resolved, harmfulqa_record_count=1377)
    with pytest.raises(dp.activation_artifact.ActivationArtifactError):
        dp.verify_activation_run("A", run_dir, resolved)


def test_verify_activation_run_rejects_wrong_token_position(tmp_path, monkeypatch):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    _mock_frozen_construction_manifest(monkeypatch)
    run_dir = write_fake_activation_run_dir(tmp_path, "acts_A", "A", resolved, token_position="response_last")
    with pytest.raises(dp.activation_artifact.ActivationArtifactError):
        dp.verify_activation_run("A", run_dir, resolved)


def test_verify_activation_run_rejects_an_unsafe_loading_policy(tmp_path, monkeypatch):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    _mock_frozen_construction_manifest(monkeypatch)
    run_dir = write_fake_activation_run_dir(
        tmp_path, "acts_A", "A", resolved, loading_policy={"local_files_only": False, "trust_remote_code": True},
    )
    with pytest.raises(dp.activation_artifact.ActivationArtifactError):
        dp.verify_activation_run("A", run_dir, resolved)


def test_verify_activation_run_rejects_missing_activations_file(tmp_path, monkeypatch):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    _mock_frozen_construction_manifest(monkeypatch)
    run_dir = write_fake_activation_run_dir(tmp_path, "acts_A", "A", resolved)
    (run_dir / "activations.pt").unlink()
    with pytest.raises(dp.activation_artifact.ActivationArtifactError):
        dp.verify_activation_run("A", run_dir, resolved)


def test_verify_activation_run_rejects_missing_run_meta(tmp_path, monkeypatch):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    _mock_frozen_construction_manifest(monkeypatch)
    run_dir = write_fake_activation_run_dir(tmp_path, "acts_A", "A", resolved)
    (run_dir / "run_meta.json").unlink()
    with pytest.raises(dp.activation_artifact.ActivationArtifactError):
        dp.verify_activation_run("A", run_dir, resolved)


def test_verify_activation_run_enforces_tensor_identity_via_the_existing_validator(tmp_path, monkeypatch):
    """development_pilot re-uses steer_sweep.validate_construction_activations rather
    than reimplementing tensor/identity checking -- a reordered construction identity
    (caught by the real, unmocked validator logic) must propagate as a failure here."""
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    _mock_frozen_construction_manifest(monkeypatch)
    blob = _fake_construction_blob()
    blob["source_ids"][0], blob["source_ids"][1] = blob["source_ids"][1], blob["source_ids"][0]
    run_dir = write_fake_activation_run_dir(tmp_path, "acts_A", "A", resolved, blob=blob)
    with pytest.raises(ValueError):
        dp.verify_activation_run("A", run_dir, resolved)


def test_verify_activation_run_enforces_it_dpo_shape_agreement(tmp_path, monkeypatch):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    _mock_frozen_construction_manifest(monkeypatch)
    blob = _fake_construction_blob()
    blob["dpo"] = blob["dpo"][:, :-1, :]  # one fewer prompt than "it"
    run_dir = write_fake_activation_run_dir(tmp_path, "acts_A", "A", resolved, blob=blob)
    with pytest.raises(ValueError):
        dp.verify_activation_run("A", run_dir, resolved)


# Task 016: sidecar-bound activation-artifact provenance


def test_verify_activation_run_rejects_a_missing_sidecar(tmp_path, monkeypatch):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    _mock_frozen_construction_manifest(monkeypatch)
    run_dir = write_fake_activation_run_dir(tmp_path, "acts_A", "A", resolved, write_sidecar=False)
    with pytest.raises(dp.activation_artifact.ActivationArtifactError):
        dp.verify_activation_run("A", run_dir, resolved)


def test_verify_activation_run_rejects_activations_pt_bytes_changed_after_publish(tmp_path, monkeypatch):
    """The sidecar and run_meta.json are both untouched and self-consistent -- only the
    activation file's actual bytes changed after the sidecar was published."""
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    _mock_frozen_construction_manifest(monkeypatch)
    run_dir = write_fake_activation_run_dir(tmp_path, "acts_A", "A", resolved)
    blob = torch.load(run_dir / "activations.pt", map_location="cpu")
    blob["it"] = blob["it"] + 1.0
    torch.save(blob, run_dir / "activations.pt")
    with pytest.raises(dp.activation_artifact.ActivationArtifactError):
        dp.verify_activation_run("A", run_dir, resolved)


def test_verify_activation_run_rejects_a_sidecar_edited_without_recomputing_its_hash(tmp_path, monkeypatch):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    _mock_frozen_construction_manifest(monkeypatch)
    run_dir = write_fake_activation_run_dir(tmp_path, "acts_A", "A", resolved)
    sidecar_path = run_dir / dp.activation_artifact.SIDECAR_FILENAME
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["activation_size_bytes"] = payload["activation_size_bytes"] + 1  # edited, hash NOT recomputed
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(dp.activation_artifact.ActivationArtifactError):
        dp.verify_activation_run("A", run_dir, resolved)


def test_verify_activation_run_rejects_a_manually_rehashed_sidecar_with_the_wrong_run_identity_hash(tmp_path, monkeypatch):
    """The sidecar's own manifest_hash is recomputed correctly (so its self-hash check
    passes), but its run_identity_hash was set to a value that does not equal the
    verified run_meta.json's own hash -- proving the sidecar-to-run_meta binding is a
    real cross-file comparison, not merely "does the sidecar check out against itself"."""
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    _mock_frozen_construction_manifest(monkeypatch)
    run_dir = write_fake_activation_run_dir(
        tmp_path, "acts_A", "A", resolved,
        sidecar_overrides={"run_identity_hash": "0" * 64},  # rehashed (tamper_sidecar_hash=False)
    )
    with pytest.raises(dp.activation_artifact.ActivationArtifactError):
        dp.verify_activation_run("A", run_dir, resolved)


def test_verify_activation_run_rejects_wrong_roles(tmp_path, monkeypatch):
    """run_meta.json and its sidecar agree with *each other* (roles=M0-B/M+-B, pair=A)
    but disagree with what pair A actually requires (M0-A/M+-A) -- caught only because
    the caller's own expected identity is checked, not merely internal consistency."""
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    _mock_frozen_construction_manifest(monkeypatch)
    run_dir = write_fake_activation_run_dir(
        tmp_path, "acts_A", "A", resolved,
        endpoint_meta_overrides={"roles": {"it": "M0-B", "dpo": "M+-B"}},
    )
    with pytest.raises(dp.activation_artifact.ActivationArtifactError):
        dp.verify_activation_run("A", run_dir, resolved)


def test_verify_activation_run_rejects_source_manifest_hash_mismatch_alone(tmp_path, monkeypatch):
    """Isolates source_manifest_hash specifically (candidate_manifest_hash and pair are
    both left correct and mutually consistent) -- a Pair A artifact resolved against a
    different frozen source-artifact manifest must still be rejected."""
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    _mock_frozen_construction_manifest(monkeypatch)
    run_dir = write_fake_activation_run_dir(
        tmp_path, "acts_A", "A", resolved,
        endpoint_meta_overrides={"source_manifest_hash": "0" * 64},
    )
    with pytest.raises(dp.activation_artifact.ActivationArtifactError):
        dp.verify_activation_run("A", run_dir, resolved)


def test_verify_activation_run_rejects_pair_b_artifact_substituted_for_pair_a(tmp_path, monkeypatch):
    """The direct Task 016 substitution scenario: a genuinely valid, sidecar-bound Pair
    B activation run, presented as if it were Pair A's. The sidecar and run_meta.json
    are perfectly self-consistent with each other (this really is a complete, correctly
    published Pair B artifact) -- rejected only because the caller checks it against
    Pair A's own resolved identity, exactly as development_pilot.py does for real."""
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    _mock_frozen_construction_manifest(monkeypatch)
    pair_b_dir = write_fake_activation_run_dir(tmp_path, "acts_B", "B", resolved)
    with pytest.raises(dp.activation_artifact.ActivationArtifactError):
        dp.verify_activation_run("A", pair_b_dir, resolved)


# ===========================================================================
# 10-13: The frozen arm plan
# ===========================================================================


def test_arm_plan_has_exactly_eleven_arms():
    assert len(dp.ARM_PLAN) == 11
    assert len({arm["arm_id"] for arm in dp.ARM_PLAN}) == 11  # all distinct


def test_arm_plan_matches_the_frozen_matrix_exactly():
    ids = [arm["arm_id"] for arm in dp.ARM_PLAN]
    assert ids == [
        "M0-A_baseline", "M+-A_baseline", "M--A_baseline",
        "M0-A_install_real", "M+-A_reverse_real",
        "M0-A_install_random_s11", "M+-A_reverse_random_s11",
        "M0-B_baseline", "M+-B_baseline",
        "M0-B_install_real", "M+-B_reverse_real",
    ]
    pair_a = [a for a in dp.ARM_PLAN if a["pair"] == "A"]
    pair_b = [a for a in dp.ARM_PLAN if a["pair"] == "B"]
    assert len(pair_a) == 7
    assert len(pair_b) == 4


def test_no_m_minus_a_steering_arm_exists():
    m_minus_a_arms = [a for a in dp.ARM_PLAN if a["endpoint_role"] == "M--A"]
    assert len(m_minus_a_arms) == 1
    arm = m_minus_a_arms[0]
    assert arm["arm_type"] == "unsteered_baseline"
    assert arm["coefficient"] == 0.0
    assert arm["vector_source"] is None
    assert arm["lineage_flag"] == "exploratory_lineage_unverified"


def test_no_pair_b_random_arms_exist():
    for arm in dp.ARM_PLAN:
        if arm["pair"] == "B":
            assert arm["random_vector_seed"] is None
            assert arm["vector_source"] != "random_seed_11_pair_A"


def test_pair_a_random_arms_use_only_seed_11_and_share_the_same_vector_source():
    random_arms = [a for a in dp.ARM_PLAN if a["random_vector_seed"] is not None]
    assert len(random_arms) == 2
    assert {a["arm_id"] for a in random_arms} == {"M0-A_install_random_s11", "M+-A_reverse_random_s11"}
    for arm in random_arms:
        assert arm["random_vector_seed"] == 11
        assert arm["vector_source"] == "random_seed_11_pair_A"
        assert arm["pair"] == "A"


def test_pilot_coefficient_is_frozen_at_point_zero_five_and_marked_non_calibrated():
    assert dp.PILOT_COEFFICIENT == 0.05
    assert dp.PILOT_COEFFICIENT_STATUS == "pilot_only_not_calibrated"
    steered = [a for a in dp.ARM_PLAN if a["vector_source"] is not None]
    assert steered  # sanity: there are steered arms
    for arm in steered:
        assert abs(arm["coefficient"]) == dp.PILOT_COEFFICIENT
    installs = [a for a in steered if "install" in a["arm_type"]]
    reversals = [a for a in steered if "reversal" in a["arm_type"]]
    assert all(a["coefficient"] == dp.PILOT_COEFFICIENT for a in installs)
    assert all(a["coefficient"] == -dp.PILOT_COEFFICIENT for a in reversals)


def test_no_projection_ablation_single_layer_or_extra_coefficients_in_the_plan():
    for arm in dp.ARM_PLAN:
        assert arm["intervention_operation"] in ("none", "add")
        assert abs(arm["coefficient"]) in (0.0, dp.PILOT_COEFFICIENT)


# ===========================================================================
# 14-19, 23: Full-run behaviour
# ===========================================================================


def test_pair_a_and_pair_b_vectors_are_built_independently(tmp_path, monkeypatch):
    calls = []
    real_build_vectors = dp.build_vectors

    def spy_build_vectors(path, **kwargs):
        calls.append(Path(path))
        return real_build_vectors(path, **kwargs)

    monkeypatch.setattr(dp, "build_vectors", spy_build_vectors)
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch)
    assert rc == 0
    assert len(calls) == 2
    assert calls[0] != calls[1]


def test_all_arm_outputs_contain_the_same_50_ordered_record_identities(tmp_path, monkeypatch):
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch)
    assert rc == 0
    expected_ids = [f"harmfulqa-{i}" for i in range(10, 60)]
    for arm in dp.ARM_PLAN:
        pair_dir = "pair_A" if arm["pair"] == "A" else "pair_B"
        records = _read_jsonl(output_dir / "judge_ready" / pair_dir / f"{arm['arm_id']}.jsonl")
        assert len(records) == 50
        assert [r["id"] for r in records] == expected_ids
        assert [r["source_id"] for r in records] == expected_ids


def test_judge_ready_schema_is_compatible_with_score_sweep(tmp_path, monkeypatch):
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch)
    assert rc == 0
    records = _read_jsonl(output_dir / "judge_ready" / "pair_A" / "M0-A_install_real.jsonl")
    for r in records:
        assert isinstance(r["id"], str) and r["id"]
        assert isinstance(r["prompt"], str)
        assert isinstance(r["response"], str)
        assert r["lambda"] == r["coefficient"]

    baseline = {r["id"]: r for r in _read_jsonl(output_dir / "judge_ready" / "pair_A" / "M0-A_baseline.jsonl")}
    for r in records:
        assert r["id"] in baseline  # steering-shift reference lookup by id must resolve


def test_tokenizer_and_model_loads_use_safe_local_policy(tmp_path, monkeypatch):
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch)
    assert rc == 0  # the fake loaders themselves assert local_files_only=True, trust_remote_code=False


def test_missing_terminators_fail_before_model_loading(tmp_path, monkeypatch):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    dev_records = _fake_development_records()
    first_ten = dp.generation_smoke.select_smoke_records(dev_records)
    gate2_dir = write_fake_gate2_dir(tmp_path, "gate2", env, first_ten, resolved)
    _mock_frozen_construction_manifest(monkeypatch)
    acts_a_dir = write_fake_activation_run_dir(tmp_path, "acts_A", "A", resolved)
    acts_b_dir = write_fake_activation_run_dir(tmp_path, "acts_B", "B", resolved)

    call_log: List[str] = []

    def fake_load_tokenizer(model_path, subfolder="", fallback_pad_token="<|endoftext|>",
                             trust_remote_code=True, local_files_only=False):
        role = env["path_to_role"][model_path]
        call_log.append(f"load_tokenizer:{role}")
        if role == "M0-A":
            return _FakeTokenizer(role, eos_token_id=None, extra_vocab={})
        return _FakeTokenizer(role)

    def boom_load_model(*a, **k):
        raise AssertionError("load_model must not be called when a terminator is missing")

    monkeypatch.setattr(dp, "load_tokenizer", fake_load_tokenizer)
    monkeypatch.setattr(dp, "load_model", boom_load_model)
    monkeypatch.setattr(dp, "load_harmfulqa_partition", lambda partition: dev_records)

    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "development_pilot.py", *_endpoint_argv(env),
        "--gate2-dir", str(gate2_dir), "--pair-a-activation-run", str(acts_a_dir), "--pair-b-activation-run", str(acts_b_dir),
        "--output-dir", str(output_dir),
    ])
    _patch_default_source_manifest(monkeypatch, env)

    rc = dp.main()
    assert rc == 1
    assert not output_dir.exists()
    assert not any(c.startswith("load_model:") for c in call_log)


def test_models_are_loaded_and_released_strictly_sequentially(tmp_path, monkeypatch):
    rc, output_dir, call_log = _run_full(tmp_path, monkeypatch)
    assert rc == 0
    load_model_calls = [c.split(":")[1] for c in call_log if c.startswith("load_model:")]
    # Each role's model is loaded exactly once even though M0-A/M+-A each generate three
    # arms and M0-B/M+-B each generate two -- one load, several arms, one release.
    assert load_model_calls == ["M0-A", "M+-A", "M--A", "M0-B", "M+-B"]

    generate_roles_in_order = [c.split(":")[1] for c in call_log if c.startswith("generate:")]
    # Every M0-A generation happens before M+-A's model is ever loaded, etc.
    first_index = {}
    last_index = {}
    for i, role in enumerate(generate_roles_in_order):
        first_index.setdefault(role, i)
        last_index[role] = i
    for role in ALL_ROLES:
        if role not in first_index:
            continue
        span = list(range(first_index[role], last_index[role] + 1))
        assert all(generate_roles_in_order[i] == role for i in span), "a role's generations must be contiguous"


def test_module_never_imports_or_references_judge():
    assert "judge" not in dir(dp)
    assert "Judge" not in dir(dp)
    assert "score_sweep" not in dir(dp)
    assert "steering.judge" not in sys.modules or not hasattr(dp, "Judge")


# ===========================================================================
# 20-22: Gate 3 summary and inspection table
# ===========================================================================


def test_gate3_summary_denominators_are_exact(tmp_path, monkeypatch):
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch)
    assert rc == 0
    summary = json.loads((output_dir / "gate3_summary.json").read_text(encoding="utf-8"))
    assert summary["gate"] == "gate3_development_pilot"
    assert len(summary["arms"]) == 11
    for arm_id, stats in summary["arms"].items():
        assert stats["expected_record_count"] == 50
        assert stats["actual_record_count"] == 50
        assert stats["identity_order_ok"] is True
        assert stats["n_invalid"] == 0
        assert stats["invalid_rate"] == 0.0
        assert stats["n_stop_reason_unknown"] == 0
        assert stats["n_max_new_tokens"] == 0
        assert stats["n_post_terminator_continuation"] == 0
        assert stats["n_empty_or_missing_response"] == 0
        assert stats["stop_reason_distribution"] == {"eos_token": 50}
    assert summary["automated_integrity_pass"] is True


def test_invalid_and_max_length_records_remain_in_output_and_summary(tmp_path, monkeypatch):
    """Invalid/max-length records affect the gate verdict (and hence the exit code),
    never record retention: the output is fully written and promoted either way."""
    scripted = _clean_results(50)
    scripted[0] = GenerationResult(text="", generated_token_count=0, stop_reason="eos_token", stop_token_id=2)
    scripted[1] = GenerationResult(text="reached the token budget without ever stopping cleanly at all",
                                    generated_token_count=512, stop_reason="max_new_tokens", stop_token_id=None)
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch, results_for_arm={"M0-A_baseline": scripted})
    assert rc == 1  # the automated verdict fails...
    assert output_dir.exists()  # ...but the completed output is still fully preserved

    records = _read_jsonl(output_dir / "judge_ready" / "pair_A" / "M0-A_baseline.jsonl")
    assert len(records) == 50  # nothing dropped
    assert records[0]["validity_ok"] is False
    assert records[1]["stop_reason"] == "max_new_tokens"

    summary = json.loads((output_dir / "gate3_summary.json").read_text(encoding="utf-8"))
    assert summary["automated_integrity_pass"] is False
    assert summary["manual_review_required"] is True
    assert summary["automatically_continue_to_calibration"] is False

    stats = summary["arms"]["M0-A_baseline"]
    assert stats["actual_record_count"] == 50
    assert stats["n_invalid"] == 1
    assert stats["invalid_rate"] == pytest.approx(1 / 50)
    assert stats["n_max_new_tokens"] == 1
    assert stats["stop_reason_distribution"]["max_new_tokens"] == 1
    assert stats["arm_integrity_pass"] is False

    # Every other arm is unaffected -- one bad arm does not contaminate the rest.
    for arm_id, arm_stats in summary["arms"].items():
        if arm_id != "M0-A_baseline":
            assert arm_stats["arm_integrity_pass"] is True


def test_pilot_inspection_csv_is_ordered_by_pair_arm_plan_permuted_position(tmp_path, monkeypatch):
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch)
    assert rc == 0
    df = pd.read_csv(output_dir / "pilot_inspection.csv")
    assert len(df) == 11 * 50

    arm_order = {arm["arm_id"]: i for i, arm in enumerate(dp.ARM_PLAN)}
    prev_key = None
    for _, row in df.iterrows():
        key = (row["pair"], arm_order[row["arm_id"]], row["permuted_position"])
        if prev_key is not None:
            assert key >= prev_key
        prev_key = key

    # Pair A rows precede Pair B rows, and within a pair, arm-plan order is preserved.
    assert list(df["pair"].unique()) == ["A", "B"]


# ===========================================================================
# 24-25: Atomic output
# ===========================================================================


def test_existing_output_directory_is_never_overwritten(tmp_path, monkeypatch):
    env = build_pilot_env(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)
    sentinel = output_dir / "keep-me.txt"
    sentinel.write_text("pre-existing content", encoding="utf-8")

    def boom(*a, **k):
        raise AssertionError("must not be called when the output directory already exists")

    monkeypatch.setattr(dp, "load_harmfulqa_partition", boom)
    monkeypatch.setattr(dp, "load_tokenizer", boom)
    monkeypatch.setattr(dp, "load_model", boom)

    monkeypatch.setattr(sys, "argv", [
        "development_pilot.py", *_endpoint_argv(env),
        "--gate2-dir", str(tmp_path / "nonexistent-gate2"),
        "--pair-a-activation-run", str(tmp_path / "nonexistent-a"),
        "--pair-b-activation-run", str(tmp_path / "nonexistent-b"),
        "--output-dir", str(output_dir),
    ])
    _patch_default_source_manifest(monkeypatch, env)

    rc = dp.main()
    assert rc == 1
    assert sentinel.read_text(encoding="utf-8") == "pre-existing content"


def test_generation_failure_does_not_promote_a_partial_final_directory(tmp_path, monkeypatch):
    """A truncated batch (generate_batched returning fewer results than requested)
    raises before any output is written for that arm -- and nothing already staged for
    earlier arms is promoted either, since promotion is a single atomic rename."""
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    dev_records = _fake_development_records()
    first_ten = dp.generation_smoke.select_smoke_records(dev_records)
    gate2_dir = write_fake_gate2_dir(tmp_path, "gate2", env, first_ten, resolved)
    _mock_frozen_construction_manifest(monkeypatch)
    acts_a_dir = write_fake_activation_run_dir(tmp_path, "acts_A", "A", resolved)
    acts_b_dir = write_fake_activation_run_dir(tmp_path, "acts_B", "B", resolved)

    call_log: List[str] = []

    def fake_load_tokenizer(model_path, subfolder="", fallback_pad_token="<|endoftext|>",
                             trust_remote_code=True, local_files_only=False):
        role = env["path_to_role"][model_path]
        return _FakeTokenizer(role)

    def fake_load_model(model_path, subfolder="", dtype=None, device_map="auto",
                         trust_remote_code=True, local_files_only=False):
        return _FakeSteerableModel()

    def fake_generate_batched(model, tokenizer, prompts, max_new_tokens=None, batch_size=None,
                               max_input_length=None, context=None, desc=None, return_metadata=True, **kwargs):
        call_log.append(desc)
        if desc == "M+-A_baseline":
            return _clean_results(len(prompts) - 1)  # one short
        return _clean_results(len(prompts))

    monkeypatch.setattr(dp, "load_tokenizer", fake_load_tokenizer)
    monkeypatch.setattr(dp, "load_model", fake_load_model)
    monkeypatch.setattr(dp, "generate_batched", fake_generate_batched)
    monkeypatch.setattr(dp, "load_harmfulqa_partition", lambda partition: dev_records)

    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "development_pilot.py", *_endpoint_argv(env),
        "--gate2-dir", str(gate2_dir), "--pair-a-activation-run", str(acts_a_dir), "--pair-b-activation-run", str(acts_b_dir),
        "--output-dir", str(output_dir),
    ])
    _patch_default_source_manifest(monkeypatch, env)

    rc = dp.main()
    assert rc == 1
    assert not output_dir.exists()
    assert list(output_dir.parent.glob(f".{output_dir.name}.staging-*")) == []
    # M0-A's arms (generated before M+-A) are not left behind anywhere on disk either.
    assert "M0-A_baseline" in call_log


# ===========================================================================
# 26-27: Run metadata provenance; no automatic continuation
# ===========================================================================


def test_run_metadata_contains_full_provenance(tmp_path, monkeypatch):
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch)
    assert rc == 0
    meta = json.loads((output_dir / "run_meta.json").read_text(encoding="utf-8"))

    assert meta["gate"] == "gate3_development_pilot"
    assert "cli" in meta["config"]
    assert "argv" in meta

    assert [r["permuted_position"] for r in meta["pilot_records"]] == list(range(10, 60))
    assert meta["candidate_manifest_hash"]
    assert meta["source_manifest_hash"]
    assert set(meta["endpoint"]["endpoints"]) == set(ALL_ROLES)

    assert meta["gate2"]["gate2_dir"]
    assert meta["gate2"]["summary"]["pass"] is True

    assert set(meta["activation_runs"]) == {"A", "B"}
    for pair in ("A", "B"):
        entry = meta["activation_runs"][pair]
        assert entry["sha256"]
        assert entry["size_bytes"] > 0
        assert entry["protocol_profile"] == "primary_v1"
        assert entry["endpoint"]["pair"] == pair

    assert meta["vectors"]["real_mean_pair_A"]["sha256"]
    assert meta["vectors"]["real_mean_pair_B"]["sha256"]
    assert meta["vectors"]["random_seed_11_pair_A"]["sha256"]
    assert meta["vectors"]["random_seed_11_pair_A"]["random_seed"] == 11
    assert meta["vectors"]["real_mean_pair_A"]["random_seed"] is None

    # Every vector's provenance: relative path (resolves under the FINAL output
    # directory, not the staging path it was written under), sha256, size, source,
    # layers, construction method, and normalisation.
    for name, expected_method in [
        ("real_mean_pair_A", "mean"), ("real_mean_pair_B", "mean"),
        ("random_seed_11_pair_A", "random_norm_matched"),
    ]:
        entry = meta["vectors"][name]
        resolved_path = output_dir / entry["path"]
        assert resolved_path.is_file()
        assert not Path(entry["path"]).is_absolute()
        assert ".staging-" not in entry["path"]
        assert resolved_path.stat().st_size == entry["size_bytes"]
        assert dp._stream_sha256(resolved_path) == entry["sha256"]
        assert entry["vector_source"] == name
        assert entry["layers"] == [27, 28, 29, 30, 31]
        assert entry["method"] == expected_method
        assert entry["normalization"] == "relative"

    assert meta["m_minus_a_lineage"] == {"label_swap_lineage_verified": False, "confirmatory_eligible": False}

    assert len(meta["arm_plan"]) == 11
    m_minus_a_entry = next(a for a in meta["arm_plan"] if a["endpoint_role"] == "M--A")
    assert m_minus_a_entry["label_swap_lineage_verified"] is False
    assert m_minus_a_entry["confirmatory_eligible"] is False
    for a in meta["arm_plan"]:
        if a["endpoint_role"] != "M--A":
            assert "label_swap_lineage_verified" not in a
            assert "confirmatory_eligible" not in a

    assert meta["decoding"]["do_sample"] is False
    assert meta["decoding"]["max_new_tokens"] == 512
    assert meta["decoding"]["max_input_length"] == 2048
    assert meta["decoding"]["sequential_endpoint_loading"] is True
    assert meta["decoding"]["batch_size"] == dp.PILOT_DEFAULT_BATCH_SIZE

    assert meta["intervention"]["layers"] == [27, 28, 29, 30, 31]
    assert meta["intervention"]["vector_normalise"] == "relative"
    assert meta["intervention"]["mode"] == "add"
    assert meta["intervention"]["coefficient"] == 0.05
    assert meta["intervention"]["coefficient_status"] == "pilot_only_not_calibrated"

    assert "environment" in meta
    assert "seed" not in meta["config"]["cli"]  # no --seed CLI flag exists


def test_summary_never_signals_automatic_continuation(tmp_path, monkeypatch):
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch)
    assert rc == 0
    summary = json.loads((output_dir / "gate3_summary.json").read_text(encoding="utf-8"))
    assert summary["manual_review_required"] is True
    assert summary["automatically_continue_to_calibration"] is False
    assert "auto_continue" not in json.dumps(summary).lower().replace("automatically_continue_to_calibration", "")


# ===========================================================================
# Correction round: vector-path provenance after atomic promotion (item 1)
# ===========================================================================


def test_vector_paths_resolve_under_the_final_output_directory_with_matching_hash_and_size(tmp_path, monkeypatch):
    """Regression: run_meta.json must never record a staging-directory path -- once the
    staging directory is renamed to the final output directory, such a path would no
    longer exist. Every recorded vector path must resolve, relative, under the *final*
    output root, and its recorded hash/size must match the promoted file."""
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch)
    assert rc == 0
    meta = json.loads((output_dir / "run_meta.json").read_text(encoding="utf-8"))

    for name, entry in meta["vectors"].items():
        assert not Path(entry["path"]).is_absolute(), f"{name}: path must be relative"
        assert "staging" not in entry["path"].lower(), f"{name}: path must not reference the staging directory"
        resolved_path = output_dir / entry["path"]
        assert resolved_path.is_file(), f"{name}: {resolved_path} does not exist under the final output directory"
        assert resolved_path.stat().st_size == entry["size_bytes"]
        assert dp._stream_sha256(resolved_path) == entry["sha256"]


# ===========================================================================
# Correction round: meaningful automated_integrity_pass (item 2)
# ===========================================================================


def _promoted_arm_records(tmp_path, monkeypatch):
    """A clean, fully-promoted run's arm_records (read back from disk), the exact
    pilot_records identity list, and the real vector_hashes mapping the run actually
    used (read back from run_meta.json, not recomputed) -- a real, schema-correct
    baseline that individual failure-mode tests corrupt one field at a time."""
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch)
    assert rc == 0
    arm_records = {}
    for arm in dp.ARM_PLAN:
        pair_dir = "pair_A" if arm["pair"] == "A" else "pair_B"
        arm_records[arm["arm_id"]] = _read_jsonl(output_dir / "judge_ready" / pair_dir / f"{arm['arm_id']}.jsonl")
    pilot_records = [
        {"source_id": f"harmfulqa-{i}", "prompt_hash": f"dev-hash-{i}", "permuted_position": i,
         "manifest_hash": "dev-manifest-hash"}
        for i in range(10, 60)
    ]
    meta = json.loads((output_dir / "run_meta.json").read_text(encoding="utf-8"))
    vector_hashes = {name: entry["sha256"] for name, entry in meta["vectors"].items()}
    return arm_records, pilot_records, vector_hashes


def test_build_gate3_summary_passes_on_a_clean_run(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is True
    assert summary["arm_set_ok"] is True
    assert summary["missing_arms"] == []
    assert summary["unexpected_arms"] == []


def test_build_gate3_summary_fails_when_an_arm_is_missing(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    del arm_records["M--A_baseline"]
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arm_set_ok"] is False
    assert summary["missing_arms"] == ["M--A_baseline"]


def test_build_gate3_summary_fails_on_an_unexpected_arm(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    arm_records["not_a_real_arm"] = list(arm_records["M0-A_baseline"])
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arm_set_ok"] is False
    assert summary["unexpected_arms"] == ["not_a_real_arm"]


def test_build_gate3_summary_fails_on_a_wrong_prompt_hash(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M0-A_baseline"]]
    recs[3]["prompt_hash"] = "tampered-hash"
    arm_records["M0-A_baseline"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M0-A_baseline"]["identity_order_ok"] is False
    assert summary["arms"]["M0-A_baseline"]["arm_integrity_pass"] is False


def test_build_gate3_summary_fails_on_a_wrong_permuted_position(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M0-A_baseline"]]
    recs[3]["permuted_position"] = 9999
    arm_records["M0-A_baseline"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M0-A_baseline"]["identity_order_ok"] is False


def test_build_gate3_summary_fails_on_an_empty_response(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M0-B_baseline"]]
    recs[0]["response"] = ""
    arm_records["M0-B_baseline"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M0-B_baseline"]["n_empty_or_missing_response"] == 1
    assert summary["arms"]["M0-B_baseline"]["arm_integrity_pass"] is False


def test_build_gate3_summary_fails_on_an_unknown_stop_reason(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M0-A_install_real"]]
    recs[0]["stop_reason"] = "unknown"
    arm_records["M0-A_install_real"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M0-A_install_real"]["n_stop_reason_unknown"] == 1
    assert summary["arms"]["M0-A_install_real"]["arm_integrity_pass"] is False


def test_build_gate3_summary_fails_on_a_max_new_tokens_stop(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M+-A_reverse_real"]]
    recs[0]["stop_reason"] = "max_new_tokens"
    arm_records["M+-A_reverse_real"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M+-A_reverse_real"]["n_max_new_tokens"] == 1


def test_build_gate3_summary_fails_on_a_post_terminator_continuation(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M0-A_install_random_s11"]]
    recs[0]["has_post_terminator_continuation"] = True
    arm_records["M0-A_install_random_s11"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M0-A_install_random_s11"]["n_post_terminator_continuation"] == 1


def test_build_gate3_summary_fails_on_an_invalid_output(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M+-B_reverse_real"]]
    recs[0]["validity_ok"] = False
    arm_records["M+-B_reverse_real"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M+-B_reverse_real"]["n_invalid"] == 1


def test_build_gate3_summary_fails_on_duplicate_ids(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M0-B_install_real"]]
    recs[1] = dict(recs[1], id=recs[0]["id"])
    arm_records["M0-B_install_real"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M0-B_install_real"]["duplicate_ids"] is True


def test_build_gate3_summary_fails_on_a_missing_schema_field(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M0-A_baseline"]]
    del recs[0]["vector_hash"]
    arm_records["M0-A_baseline"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M0-A_baseline"]["schema_ok"] is False


def test_build_gate3_summary_fails_on_an_unstable_arm_id(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M0-A_baseline"]]
    recs[2]["arm_id"] = "M0-A_install_real"  # claims to be a different arm mid-file
    arm_records["M0-A_baseline"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M0-A_baseline"]["stable_fields_ok"] is False


def test_build_gate3_summary_fails_when_a_non_m_minus_a_arm_carries_a_lineage_field(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M0-B_baseline"]]
    recs[0]["confirmatory_eligible"] = False  # forbidden on any arm but M--A
    arm_records["M0-B_baseline"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M0-B_baseline"]["schema_ok"] is False


def test_gate3_summary_explicit_fields_present(tmp_path, monkeypatch):
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch)
    assert rc == 0
    summary = json.loads((output_dir / "gate3_summary.json").read_text(encoding="utf-8"))
    assert "arm_set_ok" in summary and "missing_arms" in summary and "unexpected_arms" in summary
    for arm_id, stats in summary["arms"].items():
        assert set(stats) >= {
            "expected_record_count", "actual_record_count", "identity_order_ok", "stable_fields_ok",
            "schema_ok", "duplicate_ids", "n_invalid", "invalid_rate", "n_stop_reason_unknown",
            "n_max_new_tokens", "n_post_terminator_continuation", "n_empty_or_missing_response",
            "stop_reason_distribution", "arm_integrity_pass",
        }


def test_exit_code_reflects_automated_integrity_pass_and_output_is_preserved_either_way(tmp_path, monkeypatch):
    rc_clean, output_dir_clean, _ = _run_full(tmp_path / "clean", monkeypatch, output_dir=tmp_path / "out_clean")
    assert rc_clean == 0
    assert output_dir_clean.exists()

    scripted = _clean_results(50)
    scripted[0] = GenerationResult(text="", generated_token_count=0, stop_reason="eos_token", stop_token_id=2)
    rc_dirty, output_dir_dirty, _ = _run_full(
        tmp_path / "dirty", monkeypatch, results_for_arm={"M0-A_baseline": scripted}, output_dir=tmp_path / "out_dirty",
    )
    assert rc_dirty == 1
    assert output_dir_dirty.exists()  # still preserved despite the failing verdict
    dirty_summary = json.loads((output_dir_dirty / "gate3_summary.json").read_text(encoding="utf-8"))
    assert dirty_summary["manual_review_required"] is True
    assert dirty_summary["automatically_continue_to_calibration"] is False


# ===========================================================================
# Correction round: baseline / real / random record-schema semantics (item 3)
# ===========================================================================


def test_baseline_record_schema_is_exact(tmp_path, monkeypatch):
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch)
    assert rc == 0
    rec = _read_jsonl(output_dir / "judge_ready" / "pair_A" / "M0-A_baseline.jsonl")[0]
    assert rec["coefficient"] == 0.0
    assert rec["lambda"] == 0.0
    assert rec["coefficient_status"] == "not_applicable_unsteered"
    assert rec["vector_source"] is None
    assert rec["vector_hash"] is None
    assert rec["random_vector_seed"] is None
    assert rec["layers"] == []
    assert rec["vector_method"] is None
    assert rec["normalization"] is None
    assert rec["positions"] is None
    assert rec["preserve_norm"] is None
    assert rec["intervention_operation"] == "none"
    assert rec["intervention_profile"] == "unsteered_baseline"
    assert rec["protocol_profile"] is None
    assert rec["run_profile"] == "gate3_development_pilot"


def test_real_direction_record_schema_is_exact(tmp_path, monkeypatch):
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch)
    assert rc == 0
    rec = _read_jsonl(output_dir / "judge_ready" / "pair_A" / "M0-A_install_real.jsonl")[0]
    assert rec["run_profile"] == "gate3_development_pilot"
    assert rec["protocol_profile"] == "primary_v1"
    assert rec["intervention_profile"] == "real_direction_pilot"
    assert rec["coefficient_status"] == "pilot_only_not_calibrated"
    assert rec["coefficient"] == 0.05
    assert rec["lambda"] == 0.05
    assert rec["vector_source"] == "real_mean_pair_A"
    assert rec["vector_hash"]
    assert rec["random_vector_seed"] is None
    assert rec["layers"] == [27, 28, 29, 30, 31]
    assert rec["vector_method"] == "mean"
    assert rec["normalization"] == "relative"
    assert rec["positions"] == "all"
    assert rec["preserve_norm"] is False

    reversal = _read_jsonl(output_dir / "judge_ready" / "pair_A" / "M+-A_reverse_real.jsonl")[0]
    assert reversal["coefficient"] == -0.05
    assert reversal["lambda"] == -0.05
    assert reversal["protocol_profile"] == "primary_v1"


def test_random_direction_record_schema_is_exact(tmp_path, monkeypatch):
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch)
    assert rc == 0
    rec = _read_jsonl(output_dir / "judge_ready" / "pair_A" / "M0-A_install_random_s11.jsonl")[0]
    assert rec["run_profile"] == "gate3_development_pilot"
    assert rec["protocol_profile"] == "secondary_random_control_v1"
    assert rec["intervention_profile"] == "random_direction_pilot"
    assert rec["coefficient_status"] == "pilot_only_not_calibrated"
    assert rec["coefficient"] == 0.05
    assert rec["vector_source"] == "random_seed_11_pair_A"
    assert rec["vector_hash"]
    assert rec["random_vector_seed"] == 11
    assert rec["layers"] == [27, 28, 29, 30, 31]
    assert rec["vector_method"] == "random_norm_matched"
    assert rec["normalization"] == "relative"
    assert rec["positions"] == "all"
    assert rec["preserve_norm"] is False

    reversal = _read_jsonl(output_dir / "judge_ready" / "pair_A" / "M+-A_reverse_random_s11.jsonl")[0]
    assert reversal["coefficient"] == -0.05
    assert reversal["random_vector_seed"] == 11
    assert reversal["protocol_profile"] == "secondary_random_control_v1"


def test_protocol_profile_is_not_one_generic_value_across_arm_types(tmp_path, monkeypatch):
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch)
    assert rc == 0
    baseline = _read_jsonl(output_dir / "judge_ready" / "pair_A" / "M0-A_baseline.jsonl")[0]
    real = _read_jsonl(output_dir / "judge_ready" / "pair_A" / "M0-A_install_real.jsonl")[0]
    random_ = _read_jsonl(output_dir / "judge_ready" / "pair_A" / "M0-A_install_random_s11.jsonl")[0]
    values = {baseline["protocol_profile"], real["protocol_profile"], random_["protocol_profile"]}
    assert values == {None, "primary_v1", "secondary_random_control_v1"}
    assert len(values) == 3  # three distinct values, never one generic label


# ===========================================================================
# Correction round: frozen seed and resolved batch size (item 4)
# ===========================================================================


def test_no_seed_cli_flag_exists():
    parser = dp._build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--seed", "1"])


def test_set_all_seeds_always_uses_the_frozen_pilot_seed(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dp, "set_all_seeds", lambda seed: calls.append(seed))
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch)
    assert rc == 0
    assert calls == [42] == [dp.PILOT_SEED]


def test_resolve_batch_size_defaults_to_ten():
    assert dp.resolve_batch_size(None) == 10 == dp.PILOT_DEFAULT_BATCH_SIZE


def test_resolve_batch_size_accepts_a_positive_override():
    assert dp.resolve_batch_size(4) == 4


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_resolve_batch_size_rejects_non_positive_values(bad):
    with pytest.raises(dp.DevelopmentPilotError):
        dp.resolve_batch_size(bad)


def test_non_positive_batch_size_is_rejected_before_any_side_effect(tmp_path, monkeypatch):
    env = build_pilot_env(tmp_path)

    def boom(*a, **k):
        raise AssertionError("must not be called when --batch-size is non-positive")

    monkeypatch.setattr(dp, "resolve_all_roles", boom)
    monkeypatch.setattr(dp, "load_harmfulqa_partition", boom)
    monkeypatch.setattr(dp, "set_all_seeds", boom)

    monkeypatch.setattr(sys, "argv", [
        "development_pilot.py", *_endpoint_argv(env),
        "--gate2-dir", str(tmp_path / "nonexistent-gate2"),
        "--pair-a-activation-run", str(tmp_path / "nonexistent-a"),
        "--pair-b-activation-run", str(tmp_path / "nonexistent-b"),
        "--output-dir", str(tmp_path / "out"), "--batch-size", "0",
    ])

    with pytest.raises(dp.DevelopmentPilotError):
        dp.main()


def test_resolved_batch_size_is_recorded_in_records_run_meta_and_summary(tmp_path, monkeypatch):
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch, extra_args=["--batch-size", "7"])
    assert rc == 0
    for arm in dp.ARM_PLAN:
        pair_dir = "pair_A" if arm["pair"] == "A" else "pair_B"
        recs = _read_jsonl(output_dir / "judge_ready" / pair_dir / f"{arm['arm_id']}.jsonl")
        assert all(r["batch_size"] == 7 for r in recs)

    meta = json.loads((output_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["decoding"]["batch_size"] == 7

    summary = json.loads((output_dir / "gate3_summary.json").read_text(encoding="utf-8"))
    assert summary["batch_size"] == 7


# ===========================================================================
# Correction round: M--A frozen lineage propagation (item 5)
# ===========================================================================


def test_verify_m_minus_a_lineage_accepts_the_frozen_false_false_pair(tmp_path):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    result = dp.verify_m_minus_a_lineage(resolved)
    assert result == {"label_swap_lineage_verified": False, "confirmatory_eligible": False}


def test_verify_m_minus_a_lineage_rejects_a_true_confirmatory_eligible_flag(tmp_path):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    resolved.roles["M--A"].merge["flip_lineage"]["confirmatory_eligible"] = True
    with pytest.raises(dp.DevelopmentPilotError):
        dp.verify_m_minus_a_lineage(resolved)


def test_verify_m_minus_a_lineage_rejects_a_true_label_swap_verified_flag(tmp_path):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    resolved.roles["M--A"].merge["flip_lineage"]["label_swap_lineage_verified"] = True
    with pytest.raises(dp.DevelopmentPilotError):
        dp.verify_m_minus_a_lineage(resolved)


def test_verify_m_minus_a_lineage_rejects_a_missing_flip_lineage(tmp_path):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    resolved.roles["M--A"].merge.pop("flip_lineage")
    with pytest.raises(dp.DevelopmentPilotError):
        dp.verify_m_minus_a_lineage(resolved)


def test_m_minus_a_is_baseline_only_in_the_frozen_plan():
    """Re-pins the pre-existing guarantee (already covered by
    test_no_m_minus_a_steering_arm_exists) in the context of this correction round:
    M--A never gets a real or random direction arm, so it never carries a vector
    source/hash that lineage fields could be confused with."""
    m_minus_a_arms = [a for a in dp.ARM_PLAN if a["endpoint_role"] == "M--A"]
    assert len(m_minus_a_arms) == 1
    assert m_minus_a_arms[0]["arm_type"] == "unsteered_baseline"
    assert m_minus_a_arms[0]["vector_source"] is None


def test_m_minus_a_records_contain_the_verified_false_flags_and_lineage_flag(tmp_path, monkeypatch):
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch)
    assert rc == 0
    recs = _read_jsonl(output_dir / "judge_ready" / "pair_A" / "M--A_baseline.jsonl")
    assert len(recs) == 50
    for r in recs:
        assert r["label_swap_lineage_verified"] is False
        assert r["confirmatory_eligible"] is False
        assert r["lineage_flag"] == "exploratory_lineage_unverified"


def test_only_m_minus_a_records_carry_flip_lineage_fields(tmp_path, monkeypatch):
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch)
    assert rc == 0
    for arm in dp.ARM_PLAN:
        if arm["endpoint_role"] == "M--A":
            continue
        pair_dir = "pair_A" if arm["pair"] == "A" else "pair_B"
        recs = _read_jsonl(output_dir / "judge_ready" / pair_dir / f"{arm['arm_id']}.jsonl")
        for r in recs:
            assert "label_swap_lineage_verified" not in r
            assert "confirmatory_eligible" not in r


def test_run_metadata_and_arm_plan_carry_m_minus_a_lineage(tmp_path, monkeypatch):
    rc, output_dir, _ = _run_full(tmp_path, monkeypatch)
    assert rc == 0
    meta = json.loads((output_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["m_minus_a_lineage"] == {"label_swap_lineage_verified": False, "confirmatory_eligible": False}
    m_minus_a_entry = next(a for a in meta["arm_plan"] if a["endpoint_role"] == "M--A")
    assert m_minus_a_entry["label_swap_lineage_verified"] is False
    assert m_minus_a_entry["confirmatory_eligible"] is False


def test_no_code_path_marks_m_minus_a_confirmatory_when_the_manifest_lies(tmp_path, monkeypatch):
    """If a resolved M--A endpoint ever claimed confirmatory eligibility, the pilot must
    refuse to run at all -- there is no code path that silently drops or overrides a
    True flag into the frozen False/False pair; the whole run is rejected instead,
    before any seed, tokenizer, model, or GPU access."""
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    dev_records = _fake_development_records()
    first_ten = dp.generation_smoke.select_smoke_records(dev_records)
    gate2_dir = write_fake_gate2_dir(tmp_path, "gate2", env, first_ten, resolved)
    _mock_frozen_construction_manifest(monkeypatch)
    acts_a_dir = write_fake_activation_run_dir(tmp_path, "acts_A", "A", resolved)
    acts_b_dir = write_fake_activation_run_dir(tmp_path, "acts_B", "B", resolved)

    call_log: List[str] = []
    _install_fakes(monkeypatch, env, call_log)
    monkeypatch.setattr(dp, "load_harmfulqa_partition", lambda partition: dev_records)

    orig_resolve_all_roles = dp.resolve_all_roles

    def tampered_resolve_all_roles(*a, **k):
        r = orig_resolve_all_roles(*a, **k)
        r.roles["M--A"].merge["flip_lineage"]["confirmatory_eligible"] = True
        return r

    monkeypatch.setattr(dp, "resolve_all_roles", tampered_resolve_all_roles)

    def boom(*a, **k):
        raise AssertionError("must not be called after M--A lineage verification fails")

    monkeypatch.setattr(dp, "load_tokenizer", boom)
    monkeypatch.setattr(dp, "load_model", boom)
    monkeypatch.setattr(dp, "set_all_seeds", boom)

    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "development_pilot.py", *_endpoint_argv(env),
        "--gate2-dir", str(gate2_dir), "--pair-a-activation-run", str(acts_a_dir), "--pair-b-activation-run", str(acts_b_dir),
        "--output-dir", str(output_dir),
    ])
    _patch_default_source_manifest(monkeypatch, env)

    with pytest.raises(dp.DevelopmentPilotError):
        dp.main()
    assert not output_dir.exists()


# ===========================================================================
# Second correction round, item 1: consumed run_meta.json self-hash verification
# ===========================================================================


def test_verify_run_metadata_identity_accepts_a_correctly_signed_payload():
    payload = _signed_run_meta({"a": 1, "b": {"c": 2}})
    dp.verify_run_metadata_identity(payload, "test")  # must not raise


@pytest.mark.parametrize("bad_payload", [[], "a string", 123, None, True])
def test_verify_run_metadata_identity_rejects_a_non_dict_payload(bad_payload):
    with pytest.raises(dp.DevelopmentPilotError):
        dp.verify_run_metadata_identity(bad_payload, "test")


@pytest.mark.parametrize("bad_hash", ["", 12345, None, True, []])
def test_verify_run_metadata_identity_rejects_a_malformed_hash(bad_hash):
    payload = {"a": 1, "run_identity_hash": bad_hash}
    with pytest.raises(dp.DevelopmentPilotError):
        dp.verify_run_metadata_identity(payload, "test")


def test_verify_run_metadata_identity_rejects_an_edited_field_without_a_recomputed_hash():
    payload = _signed_run_meta({"a": 1, "b": 2})
    payload["a"] = 999  # edited after signing; hash not recomputed
    with pytest.raises(dp.DevelopmentPilotError):
        dp.verify_run_metadata_identity(payload, "test")


def test_verify_gate2_prerequisite_rejects_a_missing_run_identity_hash(tmp_path):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    dev_records = _fake_development_records()
    first_ten = dp.generation_smoke.select_smoke_records(dev_records)
    gate2_dir = write_fake_gate2_dir(tmp_path, "gate2", env, first_ten, resolved)
    payload = json.loads((gate2_dir / "run_meta.json").read_text(encoding="utf-8"))
    del payload["run_identity_hash"]
    (gate2_dir / "run_meta.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(dp.DevelopmentPilotError):
        dp.verify_gate2_prerequisite(gate2_dir, resolved, first_ten)


def test_verify_gate2_prerequisite_rejects_a_field_tampered_without_recomputing_the_hash(tmp_path):
    """A trusted field (here: harmfulqa_partition, which the pilot cross-checks against
    every Gate 2 role file) edited without recomputing run_identity_hash must be
    rejected, even though the edit itself looks structurally ordinary."""
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    dev_records = _fake_development_records()
    first_ten = dp.generation_smoke.select_smoke_records(dev_records)
    gate2_dir = write_fake_gate2_dir(tmp_path, "gate2", env, first_ten, resolved)
    _tamper_run_meta_field(gate2_dir / "run_meta.json", "harmfulqa_partition", "calibration")
    with pytest.raises(dp.DevelopmentPilotError):
        dp.verify_gate2_prerequisite(gate2_dir, resolved, first_ten)


def test_verify_activation_run_rejects_a_missing_run_identity_hash(tmp_path, monkeypatch):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    _mock_frozen_construction_manifest(monkeypatch)
    run_dir = write_fake_activation_run_dir(tmp_path, "acts_A", "A", resolved)
    payload = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    del payload["run_identity_hash"]
    (run_dir / "run_meta.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(dp.activation_artifact.ActivationArtifactError):
        dp.verify_activation_run("A", run_dir, resolved)


def test_verify_activation_run_rejects_a_field_tampered_without_recomputing_the_hash(tmp_path, monkeypatch):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    _mock_frozen_construction_manifest(monkeypatch)
    run_dir = write_fake_activation_run_dir(tmp_path, "acts_A", "A", resolved)
    _tamper_run_meta_field(run_dir / "run_meta.json", "harmfulqa_record_count", 9999)
    with pytest.raises(dp.activation_artifact.ActivationArtifactError):
        dp.verify_activation_run("A", run_dir, resolved)


def test_tampered_gate2_run_meta_fails_before_any_side_effect(tmp_path, monkeypatch):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    dev_records = _fake_development_records()
    first_ten = dp.generation_smoke.select_smoke_records(dev_records)
    gate2_dir = write_fake_gate2_dir(tmp_path, "gate2", env, first_ten, resolved)
    _tamper_run_meta_field(gate2_dir / "run_meta.json", "harmfulqa_manifest_hash", "tampered-without-resigning")
    _mock_frozen_construction_manifest(monkeypatch)
    acts_a_dir = write_fake_activation_run_dir(tmp_path, "acts_A", "A", resolved)
    acts_b_dir = write_fake_activation_run_dir(tmp_path, "acts_B", "B", resolved)

    call_log: List[str] = []
    _install_fakes(monkeypatch, env, call_log)
    monkeypatch.setattr(dp, "load_harmfulqa_partition", lambda partition: dev_records)

    def boom(*a, **k):
        raise AssertionError("must not be called after a run_identity_hash mismatch")

    monkeypatch.setattr(dp, "set_all_seeds", boom)
    monkeypatch.setattr(dp, "load_tokenizer", boom)
    monkeypatch.setattr(dp, "load_model", boom)

    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "development_pilot.py", *_endpoint_argv(env),
        "--gate2-dir", str(gate2_dir), "--pair-a-activation-run", str(acts_a_dir), "--pair-b-activation-run", str(acts_b_dir),
        "--output-dir", str(output_dir),
    ])
    _patch_default_source_manifest(monkeypatch, env)

    with pytest.raises(dp.DevelopmentPilotError):
        dp.main()
    assert not output_dir.exists()


def test_tampered_activation_run_meta_fails_before_any_side_effect(tmp_path, monkeypatch):
    env = build_pilot_env(tmp_path)
    resolved = _resolved(env)
    dev_records = _fake_development_records()
    first_ten = dp.generation_smoke.select_smoke_records(dev_records)
    gate2_dir = write_fake_gate2_dir(tmp_path, "gate2", env, first_ten, resolved)
    _mock_frozen_construction_manifest(monkeypatch)
    acts_a_dir = write_fake_activation_run_dir(tmp_path, "acts_A", "A", resolved)
    acts_b_dir = write_fake_activation_run_dir(tmp_path, "acts_B", "B", resolved)
    _tamper_run_meta_field(acts_a_dir / "run_meta.json", "protocol_profile", "legacy_nonconfirmatory")

    call_log: List[str] = []
    _install_fakes(monkeypatch, env, call_log)
    monkeypatch.setattr(dp, "load_harmfulqa_partition", lambda partition: dev_records)

    def boom(*a, **k):
        raise AssertionError("must not be called after a run_identity_hash mismatch")

    monkeypatch.setattr(dp, "set_all_seeds", boom)
    monkeypatch.setattr(dp, "load_tokenizer", boom)
    monkeypatch.setattr(dp, "load_model", boom)

    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "development_pilot.py", *_endpoint_argv(env),
        "--gate2-dir", str(gate2_dir), "--pair-a-activation-run", str(acts_a_dir), "--pair-b-activation-run", str(acts_b_dir),
        "--output-dir", str(output_dir),
    ])
    _patch_default_source_manifest(monkeypatch, env)

    with pytest.raises(dp.activation_artifact.ActivationArtifactError):
        dp.main()
    assert not output_dir.exists()


# ===========================================================================
# Second correction round, item 2: exact per-row arm provenance
# ===========================================================================


def test_provenance_check_fails_on_a_wrong_coefficient(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M0-A_install_real"]]
    recs[0]["coefficient"] = 0.10  # not the frozen pilot coefficient
    arm_records["M0-A_install_real"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M0-A_install_real"]["n_provenance_mismatch"] == 1
    assert summary["arms"]["M0-A_install_real"]["arm_integrity_pass"] is False


def test_provenance_check_fails_on_a_lambda_out_of_sync_with_coefficient(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M+-A_reverse_real"]]
    recs[0]["lambda"] = 0.05  # disagrees with its own coefficient field (-0.05)
    arm_records["M+-A_reverse_real"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M+-A_reverse_real"]["n_provenance_mismatch"] == 1


def test_provenance_check_fails_on_a_wrong_vector_hash(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M0-B_install_real"]]
    recs[0]["vector_hash"] = "0" * 64  # not the real resolved vector hash
    arm_records["M0-B_install_real"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M0-B_install_real"]["n_provenance_mismatch"] == 1


def test_provenance_check_fails_when_a_baseline_row_carries_a_vector_hash(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M0-B_baseline"]]
    recs[0]["vector_hash"] = "a" * 64
    arm_records["M0-B_baseline"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M0-B_baseline"]["n_provenance_mismatch"] == 1


def test_provenance_check_fails_on_a_wrong_random_seed(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M0-A_install_random_s11"]]
    recs[0]["random_vector_seed"] = 22  # not the frozen dedicated seed 11
    arm_records["M0-A_install_random_s11"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M0-A_install_random_s11"]["n_provenance_mismatch"] == 1


def test_provenance_check_fails_on_wrong_layers(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M+-A_reverse_random_s11"]]
    recs[0]["layers"] = [27, 28, 29, 30]  # dropped a frozen layer
    arm_records["M+-A_reverse_random_s11"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M+-A_reverse_random_s11"]["n_provenance_mismatch"] == 1


def test_provenance_check_fails_on_a_wrong_protocol_profile(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M0-A_install_real"]]
    recs[0]["protocol_profile"] = "secondary_random_control_v1"  # real arm claiming random's profile
    arm_records["M0-A_install_real"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M0-A_install_real"]["n_provenance_mismatch"] == 1


def test_provenance_check_fails_on_a_wrong_intervention_profile(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M0-A_baseline"]]
    recs[0]["intervention_profile"] = "real_direction_pilot"  # baseline claiming a real intervention
    arm_records["M0-A_baseline"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M0-A_baseline"]["n_provenance_mismatch"] == 1


def test_provenance_check_fails_on_a_wrong_batch_size(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M0-B_baseline"]]
    recs[0]["batch_size"] = 999
    arm_records["M0-B_baseline"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M0-B_baseline"]["n_provenance_mismatch"] == 1


def test_provenance_check_fails_on_a_wrong_harmfulqa_manifest_hash(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M+-B_reverse_real"]]
    recs[0]["harmfulqa_manifest_hash"] = "not-the-real-manifest-hash"
    arm_records["M+-B_reverse_real"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M+-B_reverse_real"]["n_provenance_mismatch"] == 1


def test_provenance_check_fails_when_id_does_not_equal_source_id(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M0-A_baseline"]]
    recs[0]["id"] = "not-the-source-id"
    arm_records["M0-A_baseline"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M0-A_baseline"]["n_provenance_mismatch"] == 1


def test_provenance_check_fails_when_m_minus_a_eligibility_is_changed_to_true(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M--A_baseline"]]
    recs[0]["confirmatory_eligible"] = True
    arm_records["M--A_baseline"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M--A_baseline"]["n_provenance_mismatch"] == 1


def test_provenance_check_fails_when_m_minus_a_label_swap_verified_is_changed_to_true(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M--A_baseline"]]
    recs[0]["label_swap_lineage_verified"] = True
    arm_records["M--A_baseline"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M--A_baseline"]["n_provenance_mismatch"] == 1


def test_provenance_check_fails_on_a_malformed_rendered_prompt_hash(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M0-A_baseline"]]
    recs[0]["rendered_prompt_sha256"] = "not-a-real-sha256"
    arm_records["M0-A_baseline"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M0-A_baseline"]["n_provenance_mismatch"] == 1


def test_provenance_check_fails_on_empty_terminator_ids(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M0-A_baseline"]]
    recs[0]["terminator_ids"] = []
    arm_records["M0-A_baseline"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M0-A_baseline"]["n_provenance_mismatch"] == 1


def test_provenance_check_fails_when_stop_token_id_is_not_in_the_terminator_set(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    recs = [dict(r) for r in arm_records["M0-A_baseline"]]
    recs[0]["stop_token_id"] = 999999
    arm_records["M0-A_baseline"] = recs
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is False
    assert summary["arms"]["M0-A_baseline"]["n_provenance_mismatch"] == 1


def test_provenance_mismatch_field_present_and_zero_on_a_clean_run(tmp_path, monkeypatch):
    arm_records, pilot_records, vector_hashes = _promoted_arm_records(tmp_path, monkeypatch)
    summary = dp.build_gate3_summary(arm_records, pilot_records, dp.PILOT_DEFAULT_BATCH_SIZE, vector_hashes)
    assert summary["automated_integrity_pass"] is True
    for arm_id, stats in summary["arms"].items():
        assert stats["n_provenance_mismatch"] == 0, arm_id
