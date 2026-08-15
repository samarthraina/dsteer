"""Guards on `scripts/prepare_endpoints.py` (Task 011, protocol Sections 4/13 Gate 0).

Every Transformers/PEFT/torch call lives behind a thin, individually monkeypatchable
wrapper -- these tests use tiny real `torch.nn.Module` fakes (no download, no GPU, no
real pretrained weights) so `named_modules()`, `get_input_embeddings()`, and
`torch.equal(...)` behave exactly like a real model without loading one. `--device cpu`
is passed in every test invocation (the script's own default, for the eventual A100
run, is `cuda:0`) so nothing here needs a GPU.

The real, hash-pinned `manifests/model_artifacts_v1.json` hashes are of real model
weight files this environment does not have; `validate_frozen_source_identity` is
monkeypatched to a no-op in tests that need a full run, using a synthetic (but
internally self-consistent, real-hash-verified) source manifest instead -- the frozen
pin itself is already covered by `tests/test_artifact_identity.py`.

Run with:
    pytest tests/test_prepare_endpoints.py -v

CPU-only, offline: no real model is ever loaded.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_endpoints  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steering.artifact_identity import build_manifest as build_source_manifest  # noqa: E402
from steering.endpoint_manifest import load_manifest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Tiny real nn.Module / tokenizer fakes
# ---------------------------------------------------------------------------


class _FakeCausalLM(nn.Module):
    def __init__(self, vocab_size=12, hidden_size=8, num_layers=2, model_type="llama", tie=False):
        super().__init__()
        self.config = SimpleNamespace(
            model_type=model_type, hidden_size=hidden_size,
            num_hidden_layers=num_layers, tie_word_embeddings=tie,
        )
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie:
            self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids):
        return self.lm_head(self.embed_tokens(input_ids))

    def resize_token_embeddings(self, new_size):
        old = self.embed_tokens
        new_embed = nn.Embedding(new_size, old.embedding_dim)
        with torch.no_grad():
            new_embed.weight[: old.num_embeddings] = old.weight
        self.embed_tokens = new_embed

        old_head = self.lm_head
        new_head = nn.Linear(old_head.in_features, new_size, bias=False)
        with torch.no_grad():
            new_head.weight[: old_head.out_features] = old_head.weight
        self.lm_head = new_head
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def save_pretrained(self, path, safe_serialization=True, max_shard_size="5GB"):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "config.json").write_text(json.dumps({"model_type": self.config.model_type}), encoding="utf-8")
        (p / "model.safetensors").write_bytes(f"fake-weights-{id(self)}".encode())


class _ResidualLoraLayer(nn.Module):
    def forward(self, x):
        return x


class _FakeCausalLMWithResidualAdapter(_FakeCausalLM):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.lora_leftover = _ResidualLoraLayer()


class _FakeTokenizer:
    def __init__(self, vocab_size=12, special_ids=None):
        self._vocab = {f"tok{i}": i for i in range(vocab_size)}
        ids = special_ids or {}
        for name in prepare_endpoints._SPECIAL_TOKEN_ATTRS:
            setattr(self, name, ids.get(name))

    def __len__(self):
        return len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def save_pretrained(self, path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "tokenizer.json").write_text(json.dumps({"vocab_size": len(self)}), encoding="utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _extended_tokenizer(base: _FakeTokenizer, extra: int) -> _FakeTokenizer:
    tok = _FakeTokenizer(len(base) + extra)
    for name in prepare_endpoints._SPECIAL_TOKEN_ATTRS:
        setattr(tok, name, getattr(base, name))
    return tok


# ---------------------------------------------------------------------------
# Synthetic (but real-hash-verified) source manifest + local roots
# ---------------------------------------------------------------------------


def _hex40(fill: str) -> str:
    return fill * 40


def _write_file(root: Path, rel_path: str, content: bytes) -> str:
    full = root / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(content)
    return _sha256(content)


def _build_synthetic_env(tmp_path):
    """Five local artifact roots with tiny real files whose hashes are computed (not
    invented) and folded into a synthetic source manifest built with the real
    `artifact_identity.build_manifest`. A separate archive root holds the flip run's
    trainer-state JSON at the expected final step. Returns (source_manifest, roots,
    archive_root). Adapter roots carry no tokenizer files by default."""
    roots = {
        "pair_a_sft": tmp_path / "pair_a_sft",
        "pair_a_dpo": tmp_path / "pair_a_dpo",
        "pair_b_sft": tmp_path / "pair_b_sft",
        "pair_a_flip_adapter": tmp_path / "pair_a_flip_adapter",
        "pair_b_dpo_adapter": tmp_path / "pair_b_dpo_adapter",
    }

    def declared(artifact_id, filenames):
        return [
            {"path": name, "sha256": _write_file(roots[artifact_id], name, f"{artifact_id}:{name}".encode()), "size_bytes": None}
            for name in filenames
        ]

    archive_root = tmp_path / "archive"
    trainer_state_rel = "runs/dpo_training/llama3-oh-flip/20260812T060638Z/checkpoint-2510/trainer_state.json"
    (archive_root / Path(trainer_state_rel).parent).mkdir(parents=True, exist_ok=True)
    (archive_root / trainer_state_rel).write_text(json.dumps({"global_step": 2510}), encoding="utf-8")

    artifacts = [
        {
            "artifact_id": "pair_a_sft", "repository_type": "model", "repository": "org/pair-a",
            "revision": _hex40("1"), "subpath": "SFT_merged", "kind": "checkpoint", "base_artifact_id": None,
            "inference_ready": True, "lineage": {},
            "files": declared("pair_a_sft", ["model.safetensors", "tokenizer.json"]),
        },
        {
            "artifact_id": "pair_a_dpo", "repository_type": "model", "repository": "org/pair-a",
            "revision": _hex40("1"), "subpath": "DPO_merged", "kind": "checkpoint", "base_artifact_id": None,
            "inference_ready": True, "lineage": {},
            "files": declared("pair_a_dpo", ["model.safetensors", "tokenizer.json"]),
        },
        {
            "artifact_id": "pair_b_sft", "repository_type": "model", "repository": "org/pair-b",
            "revision": _hex40("2"), "subpath": "", "kind": "checkpoint", "base_artifact_id": None,
            "inference_ready": True, "lineage": {},
            "files": declared("pair_b_sft", ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]),
        },
        {
            "artifact_id": "pair_b_dpo_adapter", "repository_type": "dataset", "repository": "org/results",
            "revision": _hex40("3"), "subpath": "runs/b", "kind": "adapter", "base_artifact_id": "pair_b_sft",
            "inference_ready": False, "lineage": {},
            "files": declared("pair_b_dpo_adapter", ["adapter_model.safetensors", "training_args.bin"]),
        },
        {
            "artifact_id": "pair_a_flip_adapter", "repository_type": "dataset", "repository": "org/results",
            "revision": _hex40("3"), "subpath": "runs/a-flip", "kind": "adapter", "base_artifact_id": "pair_a_sft",
            "inference_ready": False,
            "lineage": {"archived_trainer_state_path": trainer_state_rel},
            "files": declared("pair_a_flip_adapter", ["adapter_model.safetensors", "training_args.bin"]),
        },
    ]
    source_manifest = build_source_manifest(artifacts)
    return source_manifest, roots, archive_root


def _install_fake_models(monkeypatch, base_models, tokenizers, adapter_calls=None, merged_models=None):
    """Monkeypatch every model/tokenizer wrapper to look up a fake by root path. Also
    registers whatever `_save_endpoint` actually saves under the staging role dir back
    into these same lookup dicts, so the post-save reload (`_resolve_merged_endpoint`)
    finds the object that was "written" rather than a KeyError."""
    adapter_calls = adapter_calls if adapter_calls is not None else []

    def fake_load_base_config(root):
        return base_models[str(root)].config

    def fake_load_tokenizer_from(root):
        key = str(root)
        if key not in tokenizers:
            raise FileNotFoundError(f"no tokenizer at {root}")
        return tokenizers[key]

    def fake_load_base_model(root, dtype_name, device):
        return base_models[str(root)]

    def fake_load_adapter(model, adapter_root):
        adapter_calls.append({"model": model, "adapter_root": Path(adapter_root)})
        merged = (merged_models or {}).get(str(adapter_root), model)
        return SimpleNamespace(merge_and_unload=lambda: merged)

    def fake_save_endpoint(model, tokenizer, path, max_shard_size):
        model.save_pretrained(str(path), safe_serialization=True, max_shard_size=max_shard_size)
        tokenizer.save_pretrained(str(path))
        base_models[str(path)] = model
        tokenizers[str(path)] = tokenizer

    monkeypatch.setattr(prepare_endpoints, "_load_base_config", fake_load_base_config)
    monkeypatch.setattr(prepare_endpoints, "_load_tokenizer_from", fake_load_tokenizer_from)
    monkeypatch.setattr(prepare_endpoints, "_load_base_model", fake_load_base_model)
    monkeypatch.setattr(prepare_endpoints, "_load_adapter", fake_load_adapter)
    monkeypatch.setattr(prepare_endpoints, "_save_endpoint", fake_save_endpoint)
    # A real merge genuinely needs peft installed; this test environment does not have
    # it, so library-version reporting (incidental metadata, not merge behavior) is
    # faked here rather than exercising a real `import peft`.
    monkeypatch.setattr(
        prepare_endpoints, "_library_versions",
        lambda include_peft: {"torch": "0.0.0-fake", "transformers": "0.0.0-fake", **({"peft": "0.0.0-fake"} if include_peft else {})},
    )
    return adapter_calls


def _forbid(monkeypatch, name):
    def boom(*a, **k):
        raise AssertionError(f"{name} must not be called")
    monkeypatch.setattr(prepare_endpoints, name, boom)


def _base_args(tmp_path, roots, archive_root, output_dir, extra=None):
    args = [
        "--pair-a-sft-root", str(roots["pair_a_sft"]),
        "--pair-a-dpo-root", str(roots["pair_a_dpo"]),
        "--pair-b-sft-root", str(roots["pair_b_sft"]),
        "--pair-a-flip-adapter-root", str(roots["pair_a_flip_adapter"]),
        "--pair-b-dpo-adapter-root", str(roots["pair_b_dpo_adapter"]),
        "--pair-a-flip-archive-root", str(archive_root),
        "--output-dir", str(output_dir),
        "--device", "cpu",
    ]
    return args + (extra or [])


def _default_models_and_tokenizers(roots):
    base_models = {
        str(roots["pair_a_sft"]): _FakeCausalLM(vocab_size=12, model_type="llama"),
        str(roots["pair_a_dpo"]): _FakeCausalLM(vocab_size=12, model_type="llama"),
        str(roots["pair_b_sft"]): _FakeCausalLM(vocab_size=12, model_type="mistral"),
    }
    tokenizers = {
        str(roots["pair_a_sft"]): _FakeTokenizer(12),
        str(roots["pair_a_dpo"]): _FakeTokenizer(12),
        str(roots["pair_b_sft"]): _FakeTokenizer(12),
    }
    return base_models, tokenizers


def _run_full(tmp_path, monkeypatch, base_models, tokenizers, merged_models=None, extra_args=None):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    monkeypatch.setattr(prepare_endpoints, "load_source_manifest", lambda path: source_manifest)
    monkeypatch.setattr(prepare_endpoints, "validate_frozen_source_identity", lambda m: None)
    adapter_calls = _install_fake_models(monkeypatch, base_models, tokenizers, merged_models=merged_models)

    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "prepare_endpoints.py", *_base_args(tmp_path, roots, archive_root, output_dir, extra=extra_args),
    ])
    rc = prepare_endpoints.main()
    return rc, output_dir, adapter_calls, roots


def _no_staging_dirs_left(output_dir: Path) -> bool:
    parent = output_dir.parent
    if not parent.exists():
        return True
    return list(parent.glob(f".{output_dir.name}.staging-*")) == []


# plan_endpoints: exact mapping


def test_plan_endpoints_has_the_exact_required_mapping():
    plan = {p["role"]: p for p in prepare_endpoints.plan_endpoints()}
    assert set(plan) == {"M0-A", "M+-A", "M--A", "M0-B", "M+-B"}
    assert plan["M0-A"] == {"role": "M0-A", "status": "direct", "source_artifact_id": "pair_a_sft", "base_artifact_id": None}
    assert plan["M+-A"] == {"role": "M+-A", "status": "direct", "source_artifact_id": "pair_a_dpo", "base_artifact_id": None}
    assert plan["M0-B"] == {"role": "M0-B", "status": "direct", "source_artifact_id": "pair_b_sft", "base_artifact_id": None}
    assert plan["M--A"] == {"role": "M--A", "status": "merged", "source_artifact_id": "pair_a_flip_adapter", "base_artifact_id": "pair_a_sft"}
    assert plan["M+-B"] == {"role": "M+-B", "status": "merged", "source_artifact_id": "pair_b_dpo_adapter", "base_artifact_id": "pair_b_sft"}


# No Transformers/PEFT import before verification -- module import itself


def test_module_import_does_not_import_torch_transformers_or_peft():
    """A fresh subprocess import (never contaminated by another test file's real
    torch/transformers import into the shared pytest process) confirms the module
    itself imports none of them at load time."""
    code = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "import prepare_endpoints\n"
        "assert 'torch' not in sys.modules, 'torch imported at module load'\n"
        "assert 'transformers' not in sys.modules, 'transformers imported at module load'\n"
        "assert 'peft' not in sys.modules, 'peft imported at module load'\n"
        "print('OK')\n"
    ) % str(REPO_ROOT / "scripts").replace("\\", "\\\\")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_dry_run_works_in_a_subprocess_with_transformers_and_peft_hidden(tmp_path):
    """`--dry-run` must succeed even when `transformers`/`peft` cannot be imported at
    all -- simulated by shadowing them with modules that raise on import."""
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    fake_pkgs_dir = tmp_path / "fake_pkgs"
    fake_pkgs_dir.mkdir()
    for name in ("transformers", "peft"):
        (fake_pkgs_dir / f"{name}.py").write_text("raise ImportError('should never be imported by --dry-run')\n", encoding="utf-8")

    output_dir = tmp_path / "out"
    args = _base_args(tmp_path, roots, archive_root, output_dir, extra=[
        "--source-manifest", str(REPO_ROOT / "manifests" / "model_artifacts_v1.json"), "--dry-run",
    ])
    # Use the real committed source manifest for this one -- verify_local_artifact
    # would fail against our synthetic roots' fake hashes, so --dry-run must fail
    # *before* reaching any model import for a different, legitimate reason; the
    # assertion below only cares that it never touches the shadowed transformers/peft.
    code = (
        "import sys; sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s')\n"
        "import prepare_endpoints\n"
        "sys.argv = ['prepare_endpoints.py'] + %r\n"
        "prepare_endpoints.main()\n"
        "assert 'transformers' not in sys.modules\n"
        "assert 'peft' not in sys.modules\n"
        "print('OK')\n"
    ) % (str(fake_pkgs_dir).replace("\\", "\\\\"), str(REPO_ROOT / "scripts").replace("\\", "\\\\"), args)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert "OK" in result.stdout, result.stdout + result.stderr


# Source verification before import/load/output mutation


def test_a_source_hash_mismatch_fails_before_any_model_load_or_output_mutation(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    (roots["pair_a_sft"] / "model.safetensors").write_bytes(b"tampered content")

    monkeypatch.setattr(prepare_endpoints, "load_source_manifest", lambda path: source_manifest)
    monkeypatch.setattr(prepare_endpoints, "validate_frozen_source_identity", lambda m: None)
    for name in ("_load_base_config", "_load_tokenizer_from", "_load_base_model", "_load_adapter", "_save_endpoint"):
        _forbid(monkeypatch, name)

    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prepare_endpoints.py", *_base_args(tmp_path, roots, archive_root, output_dir)])
    rc = prepare_endpoints.main()
    assert rc != 0
    assert not output_dir.exists()
    assert _no_staging_dirs_left(output_dir)


def test_a_missing_local_root_fails_before_any_model_load(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    monkeypatch.setattr(prepare_endpoints, "load_source_manifest", lambda path: source_manifest)
    monkeypatch.setattr(prepare_endpoints, "validate_frozen_source_identity", lambda m: None)
    for name in ("_load_base_config", "_load_tokenizer_from", "_load_base_model", "_load_adapter"):
        _forbid(monkeypatch, name)

    output_dir = tmp_path / "out"
    args = _base_args(tmp_path, roots, archive_root, output_dir)
    idx = args.index("--pair-b-sft-root") + 1
    args[idx] = str(tmp_path / "does-not-exist")
    monkeypatch.setattr(sys, "argv", ["prepare_endpoints.py", *args])
    rc = prepare_endpoints.main()
    assert rc != 0
    assert not output_dir.exists()


# Flip lineage


def test_verify_flip_lineage_records_hash_and_step(tmp_path):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    facts = prepare_endpoints.verify_flip_lineage(source_manifest, archive_root)
    assert facts["training_step"] == 2510
    flip = [a for a in source_manifest["artifacts"] if a["artifact_id"] == "pair_a_flip_adapter"][0]
    expected_path = flip["lineage"]["archived_trainer_state_path"]
    assert facts["archived_trainer_state_path"] == expected_path
    real_bytes = (archive_root / expected_path).read_bytes()
    assert facts["sha256"] == hashlib.sha256(real_bytes).hexdigest()


def test_verify_flip_lineage_always_records_lineage_unverified(tmp_path):
    """Presence + hash + correct step of the archived trainer state is not, by itself,
    proof of label-swap lineage -- both flags must always be False from this function."""
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    facts = prepare_endpoints.verify_flip_lineage(source_manifest, archive_root)
    assert facts["label_swap_lineage_verified"] is False
    assert facts["confirmatory_eligible"] is False


def test_verify_flip_lineage_rejects_a_wrong_training_step(tmp_path):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    trainer_state_rel = "runs/dpo_training/llama3-oh-flip/20260812T060638Z/checkpoint-2510/trainer_state.json"
    (archive_root / trainer_state_rel).write_text(json.dumps({"global_step": 2000}), encoding="utf-8")
    with pytest.raises(prepare_endpoints.EndpointPreparationError, match="2000"):
        prepare_endpoints.verify_flip_lineage(source_manifest, archive_root)


def test_verify_flip_lineage_fails_clearly_when_trainer_state_is_missing(tmp_path):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    empty_archive = tmp_path / "empty_archive"
    empty_archive.mkdir()
    with pytest.raises(prepare_endpoints.EndpointPreparationError):
        prepare_endpoints.verify_flip_lineage(source_manifest, empty_archive)


def test_verify_flip_lineage_rejects_a_root_escaping_declared_path(tmp_path):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    tampered = json.loads(json.dumps(source_manifest))
    for a in tampered["artifacts"]:
        if a["artifact_id"] == "pair_a_flip_adapter":
            a["lineage"]["archived_trainer_state_path"] = "../outside.json"
    with pytest.raises(prepare_endpoints.EndpointPreparationError):
        prepare_endpoints.verify_flip_lineage(tampered, archive_root)


def test_verify_flip_lineage_hashes_an_optional_training_script_and_launch_config(tmp_path):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    script = tmp_path / "flip_train.py"
    script.write_bytes(b"# DPO_FLIP=1 implementation")
    launch = tmp_path / "launch.yaml"
    launch.write_bytes(b"flip: true")

    facts = prepare_endpoints.verify_flip_lineage(
        source_manifest, archive_root, training_script_path=str(script), launch_config_path=str(launch),
    )
    assert facts["training_script"] == {"filename": "flip_train.py", "sha256": hashlib.sha256(script.read_bytes()).hexdigest(), "size_bytes": script.stat().st_size}
    assert facts["launch_config"]["filename"] == "launch.yaml"
    # Hashing these does not change the verdict.
    assert facts["label_swap_lineage_verified"] is False
    assert facts["confirmatory_eligible"] is False


def test_verify_flip_lineage_fails_on_a_missing_declared_training_script(tmp_path):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    with pytest.raises(prepare_endpoints.EndpointPreparationError):
        prepare_endpoints.verify_flip_lineage(source_manifest, archive_root, training_script_path=str(tmp_path / "missing.py"))


# --dry-run: no model load, merge, output write, or side effect


def test_dry_run_has_no_model_load_merge_or_output_write(tmp_path, monkeypatch, capsys):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    monkeypatch.setattr(prepare_endpoints, "load_source_manifest", lambda path: source_manifest)
    monkeypatch.setattr(prepare_endpoints, "validate_frozen_source_identity", lambda m: None)
    for name in (
        "_load_base_config", "_load_tokenizer_from", "_load_base_model",
        "_load_adapter", "_merge_and_unload", "_save_endpoint", "_forward_pass_smoke_test",
    ):
        _forbid(monkeypatch, name)

    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "prepare_endpoints.py", *_base_args(tmp_path, roots, archive_root, output_dir, extra=["--dry-run"]),
    ])
    rc = prepare_endpoints.main()
    assert rc == 0
    assert not output_dir.exists()
    out = capsys.readouterr().out
    assert "M--A" in out and "M+-B" in out


# Full run: correct base, single adapter application, validation gating, promotion,
# and manifest


def test_full_run_succeeds_with_exactly_one_staging_directory_and_promotion(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers)
    assert rc == 0
    for role in ("M--A", "M+-B"):
        assert (output_dir / role).exists()
    assert (output_dir / "endpoint_manifest_candidate_v1.json").exists()
    assert _no_staging_dirs_left(output_dir)  # the staging dir was renamed away, not left behind


def test_full_run_never_uses_the_wrong_base_for_an_adapter(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers)
    assert rc == 0

    by_root = {str(c["adapter_root"]): c["model"] for c in adapter_calls}
    assert by_root[str(roots["pair_a_flip_adapter"])] is base_models[str(roots["pair_a_sft"])]
    assert by_root[str(roots["pair_b_dpo_adapter"])] is base_models[str(roots["pair_b_sft"])]
    assert by_root[str(roots["pair_a_flip_adapter"])] is not base_models[str(roots["pair_b_sft"])]
    assert by_root[str(roots["pair_b_dpo_adapter"])] is not base_models[str(roots["pair_a_sft"])]


def test_full_run_applies_each_adapter_exactly_once(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers)
    assert rc == 0
    assert len(adapter_calls) == 2


def test_direct_endpoints_are_validated_but_never_saved_or_written(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    monkeypatch.setattr(prepare_endpoints, "load_source_manifest", lambda path: source_manifest)
    monkeypatch.setattr(prepare_endpoints, "validate_frozen_source_identity", lambda m: None)
    _install_fake_models(monkeypatch, base_models, tokenizers)

    save_calls = []
    real_save = prepare_endpoints._save_endpoint

    def counting_save(model, tokenizer, path, max_shard_size):
        save_calls.append(path)
        return real_save(model, tokenizer, path, max_shard_size)

    monkeypatch.setattr(prepare_endpoints, "_save_endpoint", counting_save)

    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prepare_endpoints.py", *_base_args(tmp_path, roots, archive_root, output_dir)])
    rc = prepare_endpoints.main()
    assert rc == 0
    assert len(save_calls) == 2  # M--A and M+-B only
    for role in ("M0-A", "M+-A", "M0-B"):
        assert not (output_dir / role).exists()


# Tokenizer handling


def test_no_tokenizer_artifacts_at_the_adapter_uses_the_base_tokenizer(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    resize_calls = []
    monkeypatch.setattr(prepare_endpoints, "_resize_token_embeddings", lambda *a: resize_calls.append(a))
    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers)
    assert rc == 0
    assert resize_calls == []

    manifest = load_manifest(output_dir / "endpoint_manifest_candidate_v1.json")
    by_role = {e["role"]: e for e in manifest["endpoints"]}
    assert by_role["M--A"]["merge"]["tokenizer_resolution"]["status"] == "base_only"
    assert by_role["M+-B"]["merge"]["tokenizer_resolution"]["status"] == "base_only"


def test_tokenizer_artifacts_present_but_unloadable_fails(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    # Detection must find these files, but no fake is registered for this root, so the
    # (fake) load raises -- this must NOT be silently treated as "no tokenizer".
    (roots["pair_a_flip_adapter"] / "tokenizer.json").write_text("{}", encoding="utf-8")

    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers)
    assert rc != 0
    assert not output_dir.exists()
    assert _no_staging_dirs_left(output_dir)


def test_an_identical_adapter_tokenizer_is_accepted_without_resize(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    (roots["pair_a_flip_adapter"] / "tokenizer.json").write_text("{}", encoding="utf-8")
    tokenizers[str(roots["pair_a_flip_adapter"])] = _FakeTokenizer(12)  # identical vocab to base

    resize_calls = []
    monkeypatch.setattr(prepare_endpoints, "_resize_token_embeddings", lambda *a: resize_calls.append(a))
    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers)
    assert rc == 0
    assert resize_calls == []

    manifest = load_manifest(output_dir / "endpoint_manifest_candidate_v1.json")
    by_role = {e["role"]: e for e in manifest["endpoints"]}
    assert by_role["M--A"]["merge"]["tokenizer_resolution"]["status"] == "identical"


def test_an_append_only_adapter_tokenizer_resizes_before_adapter_load_with_the_fixed_merge_seed(tmp_path, monkeypatch):
    """Uses Pair B (M+-B), the only pair where a tokenizer extension is a permitted
    exception to matched-family consistency -- Pair A permits no extension at all
    (see test_pair_a_vocab_mismatch_is_rejected_with_no_extension_exception in
    tests/test_endpoint_manifest.py), so a resize on M--A would itself be rejected."""
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    (roots["pair_b_dpo_adapter"] / "tokenizer.json").write_text("{}", encoding="utf-8")
    tokenizers[str(roots["pair_b_dpo_adapter"])] = _extended_tokenizer(tokenizers[str(roots["pair_b_sft"])], extra=4)

    order = []
    seeds_used = []
    real_resize = prepare_endpoints._resize_token_embeddings

    def spy_resize(model, new_size, seed):
        order.append(("resize", new_size))
        seeds_used.append(seed)
        real_resize(model, new_size, seed)

    monkeypatch.setattr(prepare_endpoints, "_resize_token_embeddings", spy_resize)
    _install_fake_models(monkeypatch, base_models, tokenizers)
    fake_load_adapter = prepare_endpoints._load_adapter

    def spy_load_adapter(model, adapter_root):
        order.append(("load_adapter", str(adapter_root)))
        return fake_load_adapter(model, adapter_root)

    monkeypatch.setattr(prepare_endpoints, "_load_adapter", spy_load_adapter)

    monkeypatch.setattr(prepare_endpoints, "load_source_manifest", lambda path: source_manifest)
    monkeypatch.setattr(prepare_endpoints, "validate_frozen_source_identity", lambda m: None)
    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prepare_endpoints.py", *_base_args(tmp_path, roots, archive_root, output_dir)])
    rc = prepare_endpoints.main()
    assert rc == 0

    assert seeds_used == [prepare_endpoints.MERGE_SEED]
    resize_idx = order.index(("resize", 16))
    adapter_idx = order.index(("load_adapter", str(roots["pair_b_dpo_adapter"])))
    assert resize_idx < adapter_idx

    manifest = load_manifest(output_dir / "endpoint_manifest_candidate_v1.json")
    by_role = {e["role"]: e for e in manifest["endpoints"]}
    tr = by_role["M+-B"]["merge"]["tokenizer_resolution"]
    assert tr["status"] == "append_only_extension"
    assert tr["old_vocab_size"] == 12 and tr["new_vocab_size"] == 16
    assert by_role["M+-B"]["merge"]["merge_seed"] == prepare_endpoints.MERGE_SEED


def test_a_shrinking_adapter_tokenizer_is_rejected(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    (roots["pair_a_flip_adapter"] / "tokenizer.json").write_text("{}", encoding="utf-8")
    tokenizers[str(roots["pair_a_flip_adapter"])] = _FakeTokenizer(8)  # smaller than base's 12

    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers)
    assert rc != 0
    assert not output_dir.exists()
    assert _no_staging_dirs_left(output_dir)


def test_a_reordered_shared_token_adapter_tokenizer_is_rejected(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    (roots["pair_a_flip_adapter"] / "tokenizer.json").write_text("{}", encoding="utf-8")
    reordered = _FakeTokenizer(12)
    reordered._vocab["tok0"], reordered._vocab["tok1"] = reordered._vocab["tok1"], reordered._vocab["tok0"]
    tokenizers[str(roots["pair_a_flip_adapter"])] = reordered

    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers)
    assert rc != 0
    assert not output_dir.exists()


def test_an_incompatible_special_token_adapter_tokenizer_is_rejected(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    tokenizers[str(roots["pair_a_sft"])] = _FakeTokenizer(12, special_ids={"bos_token_id": 1})
    (roots["pair_a_flip_adapter"] / "tokenizer.json").write_text("{}", encoding="utf-8")
    tokenizers[str(roots["pair_a_flip_adapter"])] = _FakeTokenizer(12, special_ids={"bos_token_id": 99})

    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers)
    assert rc != 0
    assert not output_dir.exists()


# Validation failures reject the endpoint before promotion


def test_a_vocab_inconsistency_rejects_the_endpoint(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    tokenizers[str(roots["pair_b_sft"])] = _FakeTokenizer(999)  # exceeds the fake model's embedding rows

    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers)
    assert rc != 0
    assert not (output_dir / "M+-B").exists()
    assert not output_dir.exists()


def test_a_residual_peft_module_on_a_merged_endpoint_rejects_it(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    broken_merge = _FakeCausalLMWithResidualAdapter(vocab_size=12, model_type="llama")
    merged_models = {str(roots["pair_a_flip_adapter"]): broken_merge}

    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers, merged_models=merged_models)
    assert rc != 0
    assert not output_dir.exists()


def test_a_residual_peft_module_on_a_direct_endpoint_rejects_it(tmp_path, monkeypatch):
    """Requirement 4: residual-adapter absence is required for all five endpoints, not
    only merged ones."""
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    base_models[str(roots["pair_a_dpo"])] = _FakeCausalLMWithResidualAdapter(vocab_size=12, model_type="llama")

    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers)
    assert rc != 0
    assert not output_dir.exists()


# Validate the in-memory merge before saving -- a separate, earlier gate than the
# existing independent reload-and-validate of what was actually written. Both are
# required; neither is a substitute for the other.


def test_a_residual_peft_module_in_the_in_memory_merge_is_rejected_before_saving(tmp_path, monkeypatch):
    """The fake save-and-reload path here is deliberately rigged to produce a clean
    plain model regardless of what was merged -- so if the pre-save check on the
    in-memory merged object were missing, this run would otherwise succeed. Only the
    immediate post-merge_and_unload() validation can catch it, and it must do so before
    _save_endpoint is ever called."""
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    monkeypatch.setattr(prepare_endpoints, "load_source_manifest", lambda path: source_manifest)
    monkeypatch.setattr(prepare_endpoints, "validate_frozen_source_identity", lambda m: None)

    broken_merge = _FakeCausalLMWithResidualAdapter(vocab_size=12, model_type="llama")
    merged_models = {str(roots["pair_a_flip_adapter"]): broken_merge}
    _install_fake_models(monkeypatch, base_models, tokenizers, merged_models=merged_models)

    save_calls = []

    def clean_reload_save(model, tokenizer, path, max_shard_size):
        save_calls.append(path)
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        base_models[str(p)] = _FakeCausalLM(vocab_size=12, model_type="llama")  # clean, unlike what was merged
        tokenizers[str(p)] = tokenizer

    monkeypatch.setattr(prepare_endpoints, "_save_endpoint", clean_reload_save)

    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prepare_endpoints.py", *_base_args(tmp_path, roots, archive_root, output_dir)])
    rc = prepare_endpoints.main()
    assert rc != 0
    assert not output_dir.exists()
    assert save_calls == []  # must fail before _save_endpoint is ever reached
    assert _no_staging_dirs_left(output_dir)


# Strengthened residual-PEFT detection (_looks_like_peft_module): a `peft_config`
# attribute, a class defined in the `peft` package by __module__ (regardless of name),
# or a class name marking a tuner/ModulesToSaveWrapper -- not only "lora" in the name.


def test_looks_like_peft_module_detects_a_peft_config_attribute():
    obj = SimpleNamespace(peft_config={"x": 1})
    assert prepare_endpoints._looks_like_peft_module(obj) is True


def test_looks_like_peft_module_detects_a_class_from_the_peft_package_regardless_of_name():
    class InnocuouslyNamedClass:
        __module__ = "peft.tuners.lora.layer"
    assert prepare_endpoints._looks_like_peft_module(InnocuouslyNamedClass()) is True


def test_looks_like_peft_module_detects_a_tuner_name_marker():
    class SomeTuner:
        pass
    assert prepare_endpoints._looks_like_peft_module(SomeTuner()) is True


def test_looks_like_peft_module_detects_a_modules_to_save_wrapper_name_marker():
    class ModulesToSaveWrapper:
        pass
    assert prepare_endpoints._looks_like_peft_module(ModulesToSaveWrapper()) is True


def test_looks_like_peft_module_is_false_for_an_ordinary_object():
    assert prepare_endpoints._looks_like_peft_module(nn.Linear(2, 2)) is False


def test_has_no_residual_peft_modules_detects_a_peft_config_attribute_on_the_model_itself():
    model = _FakeCausalLM(vocab_size=12)
    model.peft_config = {"fake": True}
    assert prepare_endpoints._has_no_residual_peft_modules(model) is False


def test_has_no_residual_peft_modules_detects_a_submodule_whose_class_is_from_the_peft_package():
    """Regression: the old check only substring-matched "lora" in a submodule's class
    name -- a non-LoRA peft-package module (e.g. a different tuner type) with no such
    marker in its name would have silently passed."""
    class _PeftPackageSubmodule(nn.Module):
        __module__ = "peft.tuners.other_tuner"
        def forward(self, x):
            return x

    model = _FakeCausalLM(vocab_size=12)
    model.leftover = _PeftPackageSubmodule()
    assert prepare_endpoints._has_no_residual_peft_modules(model) is False


def test_has_no_residual_peft_modules_detects_a_modules_to_save_wrapper_by_name():
    class ModulesToSaveWrapper(nn.Module):
        def forward(self, x):
            return x

    model = _FakeCausalLM(vocab_size=12)
    model.classifier = ModulesToSaveWrapper()
    assert prepare_endpoints._has_no_residual_peft_modules(model) is False


def test_has_no_residual_peft_modules_accepts_a_clean_model():
    assert prepare_endpoints._has_no_residual_peft_modules(_FakeCausalLM(vocab_size=12)) is True


def test_a_peft_config_attribute_on_a_direct_endpoint_rejects_it(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    base_models[str(roots["pair_a_dpo"])].peft_config = {"fake": True}

    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers)
    assert rc != 0
    assert not output_dir.exists()


def test_a_peft_package_submodule_without_a_lora_name_marker_on_a_merged_endpoint_rejects_it(tmp_path, monkeypatch):
    """The old detection only ever caught "lora" in a class name; a peft-package tuner
    module with an unrelated class name must still be caught on the in-memory merged
    model (validated via the reload path, same as any other merged endpoint)."""
    class _OtherPeftSubmodule(nn.Module):
        __module__ = "peft.tuners.other_tuner"
        def forward(self, x):
            return x

    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    broken_merge = _FakeCausalLM(vocab_size=12, model_type="llama")
    broken_merge.leftover = _OtherPeftSubmodule()
    merged_models = {str(roots["pair_a_flip_adapter"]): broken_merge}

    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers, merged_models=merged_models)
    assert rc != 0
    assert not output_dir.exists()


def test_a_peft_config_attribute_on_the_reloaded_merged_model_rejects_it(tmp_path, monkeypatch):
    """Validation of a merged endpoint always runs against the reloaded-from-disk
    object, never the pre-save in-memory one -- confirms the strengthened detection is
    reached on that reload path too, not only for the in-memory merge."""
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    monkeypatch.setattr(prepare_endpoints, "load_source_manifest", lambda path: source_manifest)
    monkeypatch.setattr(prepare_endpoints, "validate_frozen_source_identity", lambda m: None)
    _install_fake_models(monkeypatch, base_models, tokenizers)

    def bad_save(model, tokenizer, path, max_shard_size):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        reloaded = _FakeCausalLM(vocab_size=12, model_type="llama")
        reloaded.peft_config = {"fake": True}
        base_models[str(p)] = reloaded
        tokenizers[str(p)] = tokenizer

    monkeypatch.setattr(prepare_endpoints, "_save_endpoint", bad_save)

    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prepare_endpoints.py", *_base_args(tmp_path, roots, archive_root, output_dir)])
    rc = prepare_endpoints.main()
    assert rc != 0
    assert not output_dir.exists()


def test_a_tied_embedding_inconsistency_rejects_the_endpoint(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    base_models[str(roots["pair_b_sft"])] = _FakeCausalLM(vocab_size=12, model_type="mistral", tie=True)
    base_models[str(roots["pair_b_sft"])].lm_head = nn.Linear(8, 12, bias=False)  # breaks the tie

    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers)
    assert rc != 0
    assert not output_dir.exists()


def test_a_forward_pass_failure_rejects_the_endpoint(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)

    def boom(model, tokenizer, device):
        raise RuntimeError("forward pass failed")

    monkeypatch.setattr(prepare_endpoints, "_forward_pass_smoke_test", boom)
    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers)
    assert rc != 0
    assert not output_dir.exists()


def test_a_missing_model_type_rejects_the_endpoint(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    base_models[str(roots["pair_a_sft"])].config.model_type = None

    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers)
    assert rc != 0
    assert not output_dir.exists()


def test_a_zero_hidden_size_rejects_the_endpoint(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    base_models[str(roots["pair_b_sft"])].config.hidden_size = 0

    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers)
    assert rc != 0
    assert not output_dir.exists()


# Matched-family consistency (full-run level -- structural cases already covered in
# tests/test_endpoint_manifest.py)


def test_pair_a_family_mismatch_rejects_the_whole_run(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    base_models[str(roots["pair_a_dpo"])] = _FakeCausalLM(vocab_size=12, hidden_size=999, model_type="llama")

    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers)
    assert rc != 0
    assert not output_dir.exists()
    assert _no_staging_dirs_left(output_dir)


# Required failure-mode tests: endpoint saving, file hashing, manifest construction,
# manifest writing, final promotion


def test_endpoint_saving_failure_leaves_no_promotion(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    monkeypatch.setattr(prepare_endpoints, "load_source_manifest", lambda path: source_manifest)
    monkeypatch.setattr(prepare_endpoints, "validate_frozen_source_identity", lambda m: None)
    _install_fake_models(monkeypatch, base_models, tokenizers)

    def boom_save(model, tokenizer, path, max_shard_size):
        raise OSError("simulated disk-full while saving endpoint")

    monkeypatch.setattr(prepare_endpoints, "_save_endpoint", boom_save)

    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prepare_endpoints.py", *_base_args(tmp_path, roots, archive_root, output_dir)])
    rc = prepare_endpoints.main()
    assert rc != 0
    assert not output_dir.exists()
    assert _no_staging_dirs_left(output_dir)


def test_manifest_construction_failure_leaves_no_promotion(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    monkeypatch.setattr(prepare_endpoints, "load_source_manifest", lambda path: source_manifest)
    monkeypatch.setattr(prepare_endpoints, "validate_frozen_source_identity", lambda m: None)
    _install_fake_models(monkeypatch, base_models, tokenizers)

    def boom_build(source_manifest_hash, entries):
        raise prepare_endpoints.EndpointPreparationError("simulated manifest construction failure")

    monkeypatch.setattr(prepare_endpoints, "build_endpoint_manifest", boom_build)

    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prepare_endpoints.py", *_base_args(tmp_path, roots, archive_root, output_dir)])
    rc = prepare_endpoints.main()
    assert rc != 0
    assert not output_dir.exists()
    # Endpoints were merged successfully into staging before the manifest step failed --
    # they must not survive as a final, accepted endpoint either.
    assert _no_staging_dirs_left(output_dir)


def test_manifest_writing_failure_leaves_no_promotion(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    monkeypatch.setattr(prepare_endpoints, "load_source_manifest", lambda path: source_manifest)
    monkeypatch.setattr(prepare_endpoints, "validate_frozen_source_identity", lambda m: None)
    _install_fake_models(monkeypatch, base_models, tokenizers)

    def boom_save_manifest(payload, path):
        raise OSError("simulated disk-full while writing the candidate manifest")

    monkeypatch.setattr(prepare_endpoints, "save_endpoint_manifest", boom_save_manifest)

    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prepare_endpoints.py", *_base_args(tmp_path, roots, archive_root, output_dir)])
    rc = prepare_endpoints.main()
    assert rc != 0
    assert not output_dir.exists()
    assert _no_staging_dirs_left(output_dir)


def test_final_promotion_failure_removes_only_the_staging_directory(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    monkeypatch.setattr(prepare_endpoints, "load_source_manifest", lambda path: source_manifest)
    monkeypatch.setattr(prepare_endpoints, "validate_frozen_source_identity", lambda m: None)
    _install_fake_models(monkeypatch, base_models, tokenizers)

    output_dir = tmp_path / "out"
    real_replace = pathlib.Path.replace

    def flaky_replace(self, target):
        if Path(target) == output_dir:
            raise OSError("simulated promotion failure (e.g. cross-device rename)")
        return real_replace(self, target)

    monkeypatch.setattr(pathlib.Path, "replace", flaky_replace)

    monkeypatch.setattr(sys, "argv", ["prepare_endpoints.py", *_base_args(tmp_path, roots, archive_root, output_dir)])
    rc = prepare_endpoints.main()
    assert rc != 0
    assert not output_dir.exists()
    assert _no_staging_dirs_left(output_dir)


def test_file_hashing_failure_rejects_the_endpoint_and_leaves_no_promotion(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    monkeypatch.setattr(prepare_endpoints, "load_source_manifest", lambda path: source_manifest)
    monkeypatch.setattr(prepare_endpoints, "validate_frozen_source_identity", lambda m: None)
    _install_fake_models(monkeypatch, base_models, tokenizers)

    real_hash_output_files = prepare_endpoints._hash_output_files
    calls = {"n": 0}

    def flaky_hash(root):
        calls["n"] += 1
        if calls["n"] == 1:  # fail on the very first endpoint hashed
            raise prepare_endpoints.EndpointPreparationError("simulated file-hashing failure")
        return real_hash_output_files(root)

    monkeypatch.setattr(prepare_endpoints, "_hash_output_files", flaky_hash)

    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prepare_endpoints.py", *_base_args(tmp_path, roots, archive_root, output_dir)])
    rc = prepare_endpoints.main()
    assert rc != 0
    assert not output_dir.exists()
    assert _no_staging_dirs_left(output_dir)


def test_hash_output_files_rejects_an_escaping_symlink(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside content")
    link = root / "escape_link"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("creating symlinks requires elevated privileges on this platform")

    with pytest.raises(prepare_endpoints.EndpointPreparationError, match="escaping"):
        prepare_endpoints._hash_output_files(root)


def test_hash_output_files_accepts_a_symlink_that_stays_inside_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    real = root / "real.bin"
    real.write_bytes(b"content")
    link = root / "alias.bin"
    try:
        link.symlink_to(real)
    except OSError:
        pytest.skip("creating symlinks requires elevated privileges on this platform")

    files = prepare_endpoints._hash_output_files(root)
    paths = {f["path"] for f in files}
    assert "real.bin" in paths
    assert "alias.bin" in paths


def test_hash_output_files_returns_stable_sorted_order(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    for name in ("z.bin", "a.bin", "m.bin"):
        (root / name).write_bytes(name.encode())
    files = prepare_endpoints._hash_output_files(root)
    assert [f["path"] for f in files] == ["a.bin", "m.bin", "z.bin"]


# An existing output directory is never overwritten


def test_an_existing_output_directory_is_never_overwritten(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)
    sentinel = output_dir / "keep-me.txt"
    sentinel.write_text("pre-existing content", encoding="utf-8")

    monkeypatch.setattr(prepare_endpoints, "load_source_manifest", lambda path: source_manifest)
    monkeypatch.setattr(prepare_endpoints, "validate_frozen_source_identity", lambda m: None)
    for name in ("_load_base_model", "_load_adapter", "_save_endpoint"):
        _forbid(monkeypatch, name)

    monkeypatch.setattr(sys, "argv", ["prepare_endpoints.py", *_base_args(tmp_path, roots, archive_root, output_dir)])
    rc = prepare_endpoints.main()
    assert rc != 0
    assert sentinel.read_text(encoding="utf-8") == "pre-existing content"


def test_a_pre_existing_temp_looking_directory_is_never_touched(tmp_path, monkeypatch):
    """A directory that merely looks like a staging directory (same naming pattern) but
    was not created by this run must never be deleted or reused."""
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)

    output_dir = tmp_path / "out"
    decoy = output_dir.parent / f".{output_dir.name}.staging-deadbeefdeadbeefdeadbeefdeadbeef"
    decoy.mkdir(parents=True)
    (decoy / "not-mine.txt").write_text("do not touch", encoding="utf-8")

    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers)
    assert rc == 0
    assert decoy.exists()
    assert (decoy / "not-mine.txt").read_text(encoding="utf-8") == "do not touch"


# Candidate manifest content sanity (real hashing of saved output files, reload-based
# validation, device/library/merge-seed recording)


def test_the_candidate_manifest_records_real_hashes_of_saved_output_files(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers)
    assert rc == 0

    manifest = load_manifest(output_dir / "endpoint_manifest_candidate_v1.json")
    assert manifest["source_manifest_hash"] == source_manifest["manifest_hash"]
    by_role = {e["role"]: e for e in manifest["endpoints"]}

    for role in ("M--A", "M+-B"):
        entry = by_role[role]
        assert entry["status"] == "merged"
        assert entry["device"] == "cpu"
        assert entry["files"] == sorted(entry["files"], key=lambda f: f["path"])
        for f in entry["files"]:
            on_disk = output_dir / role / f["path"]
            assert on_disk.exists()
            assert f["sha256"] == hashlib.sha256(on_disk.read_bytes()).hexdigest()
            assert f["size_bytes"] == on_disk.stat().st_size
        assert entry["merge"]["dtype"] == "bfloat16"
        assert entry["merge"]["merge_seed"] == prepare_endpoints.MERGE_SEED
        assert entry["library_versions"]["peft"]

    flip_entry = by_role["M--A"]
    assert flip_entry["merge"]["flip_lineage"]["training_step"] == 2510
    assert flip_entry["merge"]["flip_lineage"]["confirmatory_eligible"] is False
    assert flip_entry["merge"]["flip_lineage"]["label_swap_lineage_verified"] is False

    # Direct endpoints hash *every* file at the source root, not only the source
    # manifest's declared anchor file(s) -- confirmed against each role's actual files.
    expected_anchor_files = {
        "M0-A": {"model.safetensors", "tokenizer.json"},
        "M+-A": {"model.safetensors", "tokenizer.json"},
        "M0-B": {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"},
    }
    for role in ("M0-A", "M+-A", "M0-B"):
        entry = by_role[role]
        assert entry["status"] == "direct"
        assert entry["merge"] is None
        assert entry["device"] == "cpu"
        assert {f["path"] for f in entry["files"]} >= expected_anchor_files[role]


def test_direct_endpoint_location_is_a_source_locator_matching_the_source_manifest(tmp_path, monkeypatch):
    """A direct endpoint's weights are never copied into the bundle -- its location must
    be the stable source locator from the verified source manifest, never the
    operator's machine-specific local root."""
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers)
    assert rc == 0

    manifest = load_manifest(output_dir / "endpoint_manifest_candidate_v1.json")
    by_role = {e["role"]: e for e in manifest["endpoints"]}
    by_artifact = {a["artifact_id"]: a for a in source_manifest["artifacts"]}

    for role, artifact_id in (("M0-A", "pair_a_sft"), ("M+-A", "pair_a_dpo"), ("M0-B", "pair_b_sft")):
        loc = by_role[role]["location"]
        artifact = by_artifact[artifact_id]
        assert loc == {
            "kind": "source", "repository": artifact["repository"],
            "revision": artifact["revision"], "subpath": artifact["subpath"],
        }
        assert str(roots[artifact_id]) not in json.dumps(loc)


def test_merged_endpoint_location_is_the_bundle_relative_role(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers)
    assert rc == 0

    manifest = load_manifest(output_dir / "endpoint_manifest_candidate_v1.json")
    by_role = {e["role"]: e for e in manifest["endpoints"]}
    assert by_role["M--A"]["location"] == {"kind": "bundle", "path": "M--A"}
    assert by_role["M+-B"]["location"] == {"kind": "bundle", "path": "M+-B"}


def test_the_candidate_manifest_records_tokenizer_fingerprint_and_special_token_ids(tmp_path, monkeypatch):
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers)
    assert rc == 0

    from steering.endpoint_manifest import tokenizer_fingerprint as compute_fp

    manifest = load_manifest(output_dir / "endpoint_manifest_candidate_v1.json")
    by_role = {e["role"]: e for e in manifest["endpoints"]}
    expected_fp = compute_fp({f"tok{i}": i for i in range(12)})
    expected_special = {name: None for name in prepare_endpoints._SPECIAL_TOKEN_ATTRS}
    for role in ("M0-A", "M+-A", "M--A", "M0-B", "M+-B"):
        v = by_role[role]["validation"]
        assert v["tokenizer_fingerprint"] == expected_fp
        assert v["special_token_ids"] == expected_special

    # Every Pair A role must agree exactly (enforced separately by
    # tests/test_endpoint_manifest.py; here we confirm the real pipeline actually
    # populates a *consistent* value, not just *some* value, for each pair).
    assert len({by_role[r]["validation"]["tokenizer_fingerprint"] for r in ("M0-A", "M+-A", "M--A")}) == 1
    assert len({by_role[r]["validation"]["tokenizer_fingerprint"] for r in ("M0-B", "M+-B")}) == 1


def test_the_candidate_manifest_binds_every_regular_adapter_root_file_including_adapter_config(tmp_path, monkeypatch):
    """Correction requirement 3: bind every merge-critical adapter input, not only the
    two source-manifest-pinned anchor files (adapter_model.safetensors,
    training_args.bin)."""
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    (roots["pair_a_flip_adapter"] / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")

    rc, output_dir, adapter_calls, roots = _run_full(tmp_path, monkeypatch, base_models, tokenizers)
    assert rc == 0

    manifest = load_manifest(output_dir / "endpoint_manifest_candidate_v1.json")
    by_role = {e["role"]: e for e in manifest["endpoints"]}
    adapter_files = {f["path"]: f for f in by_role["M--A"]["merge"]["adapter_input_files"]}
    assert {"adapter_config.json", "adapter_model.safetensors", "training_args.bin"} <= set(adapter_files)

    on_disk = roots["pair_a_flip_adapter"] / "adapter_config.json"
    assert adapter_files["adapter_config.json"]["sha256"] == hashlib.sha256(on_disk.read_bytes()).hexdigest()
    assert adapter_files["adapter_config.json"]["size_bytes"] == on_disk.stat().st_size


def test_main_reloads_and_validates_the_candidate_manifest_before_promotion(tmp_path, monkeypatch):
    """Correction requirement 5: after writing the candidate manifest inside staging,
    load it back and run full structural validation before the single atomic
    promotion -- a failure there must block promotion and clean up only this run's
    staging directory, the same as every other failure mode."""
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    monkeypatch.setattr(prepare_endpoints, "load_source_manifest", lambda path: source_manifest)
    monkeypatch.setattr(prepare_endpoints, "validate_frozen_source_identity", lambda m: None)
    _install_fake_models(monkeypatch, base_models, tokenizers)

    load_calls = []

    def boom_load(path):
        load_calls.append(Path(path))
        raise prepare_endpoints.EndpointPreparationError("simulated reload-validation failure")

    monkeypatch.setattr(prepare_endpoints, "load_endpoint_manifest", boom_load)

    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prepare_endpoints.py", *_base_args(tmp_path, roots, archive_root, output_dir)])
    rc = prepare_endpoints.main()
    assert rc != 0
    assert not output_dir.exists()
    assert _no_staging_dirs_left(output_dir)
    assert len(load_calls) == 1
    assert load_calls[0].name == "endpoint_manifest_candidate_v1.json"


def test_merged_endpoint_validation_reflects_the_reloaded_saved_model_not_the_pre_save_one(tmp_path, monkeypatch):
    """If the object actually written differs from the pre-merge in-memory model (e.g.
    a broken save), validation must see the *reloaded* one."""
    source_manifest, roots, archive_root = _build_synthetic_env(tmp_path)
    base_models, tokenizers = _default_models_and_tokenizers(roots)
    monkeypatch.setattr(prepare_endpoints, "load_source_manifest", lambda path: source_manifest)
    monkeypatch.setattr(prepare_endpoints, "validate_frozen_source_identity", lambda m: None)
    _install_fake_models(monkeypatch, base_models, tokenizers)

    # Force the post-save reload to hand back a broken (residual-adapter) model,
    # distinct from whatever was merged in memory -- proves validation runs against
    # what reload actually returns, not the pre-save object.
    def bad_save(model, tokenizer, path, max_shard_size):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        base_models[str(p)] = _FakeCausalLMWithResidualAdapter(vocab_size=12, model_type="llama")
        tokenizers[str(p)] = tokenizer

    monkeypatch.setattr(prepare_endpoints, "_save_endpoint", bad_save)

    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prepare_endpoints.py", *_base_args(tmp_path, roots, archive_root, output_dir)])
    rc = prepare_endpoints.main()
    assert rc != 0  # the reloaded (broken) model must fail validation
    assert not output_dir.exists()
