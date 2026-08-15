"""Prepare deterministic inference endpoints and close Gate 0 (protocol Sections 4, 13).

Five frozen source artifacts (`manifests/model_artifacts_v1.json`) become five inference
endpoints. Three are already-merged checkpoints and are used directly, unmodified. Two
are LoRA adapters that must be deterministically merged into their declared base before
they are inference-ready:

    M0-A = pair_a_sft                                          (direct)
    M+-A = pair_a_dpo                                          (direct)
    M--A = pair_a_flip_adapter merged into pair_a_sft           (merged)
    M0-B = pair_b_sft                                          (direct)
    M+-B = pair_b_dpo_adapter merged into pair_b_sft            (merged)

This script verifies every source artifact's declared hash, and the flip run's archived
trainer-state lineage, entirely with the standard library -- before Transformers or PEFT
is ever imported. It then loads/merges/validates every endpoint into one unique staging
directory (a sibling of the final `--output-dir`), and -- only if every endpoint and the
candidate manifest itself succeed -- promotes that single staging directory to the final
output directory with one atomic rename. Any failure before that point removes only the
staging directory this run created; the final output directory, which must not already
exist, is never touched. It never substitutes, retrains, stacks adapters onto adapters,
quantizes, or converts a model.

M--A (the flipped-DPO endpoint) is always prepared and validated like any other endpoint,
but its manifest entry always records `confirmatory_eligible: false`: the archived
trainer-state checkpoint proves the training run reached the expected final step, not
that chosen/rejected preference labels were actually swapped. See `verify_flip_lineage`.

    python scripts/prepare_endpoints.py --dry-run \\
        --pair-a-sft-root /data/pair_a/SFT_merged \\
        --pair-a-dpo-root /data/pair_a/DPO_merged \\
        --pair-b-sft-root /data/pair_b/sft \\
        --pair-a-flip-adapter-root /data/results/llama3-oh-flip/final_dpo_adapter \\
        --pair-b-dpo-adapter-root /data/results/openhermes-mistral-adapter \\
        --pair-a-flip-archive-root /data/results \\
        --output-dir outputs/endpoints --device cuda:0

Omit --dry-run to actually load, merge, validate, and write real endpoints. That step
loads real multi-billion-parameter models and is a separately approved execution step,
never exercised by the CPU test suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steering.artifact_identity import (
    ArtifactIdentityError,
    EXPECTED_ARTIFACT_IDS,
    load_manifest as load_source_manifest,
    validate_frozen_identity as validate_frozen_source_identity,
    validate_manifest_structure as validate_source_manifest_structure,
    verify_local_artifact,
)
from steering.endpoint_manifest import (
    FROZEN_ENDPOINT_PLAN,
    build_manifest as build_endpoint_manifest,
    check_tied_embedding_consistency,
    check_tokenizer_extension,
    check_vocab_consistency,
    load_manifest as load_endpoint_manifest,
    save_manifest as save_endpoint_manifest,
    tokenizer_fingerprint,
)

DEFAULT_SOURCE_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifests" / "model_artifacts_v1.json"

_VALID_DTYPE_NAMES = ("bfloat16", "float16", "float32")

#: Fixed, not CLI-overridable: a "fixed merge seed" that a caller could freely change
#: would not be fixed. Seeds `torch.manual_seed` immediately before a tokenizer-driven
#: embedding resize, so any random initialization of new rows is reproducible.
MERGE_SEED = 20260815

#: The archive's own final checkpoint (protocol / audit: "checkpoints at steps 500,
#: 1000, 1500, 2000, 2500, and 2510" -- the final adapter corresponds to step 2510).
EXPECTED_FLIP_TRAINING_STEP = 2510

_TOKENIZER_ARTIFACT_NAMES = (
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    "vocab.json", "merges.txt", "special_tokens_map.json",
)
_SPECIAL_TOKEN_ATTRS = (
    "bos_token_id", "eos_token_id", "pad_token_id",
    "unk_token_id", "sep_token_id", "cls_token_id", "mask_token_id",
)


class EndpointPreparationError(RuntimeError):
    """A source-verification, lineage, load, merge, or validation invariant was
    violated while preparing an endpoint."""


def plan_endpoints() -> List[Dict[str, Any]]:
    """The frozen endpoint-to-source/base/adapter mapping (protocol Sections 4 & 13),
    read from the one authority (`endpoint_manifest.FROZEN_ENDPOINT_PLAN`) that the
    candidate manifest's own structural validation is also checked against."""
    return [{"role": role, **spec} for role, spec in FROZEN_ENDPOINT_PLAN.items()]


# Source and lineage verification -- pure stdlib (hashlib/json/pathlib) only. No
# Transformers/PEFT import happens anywhere above this section, and none of it is
# reached from here.


def verify_all_sources(source_manifest: Mapping[str, Any], roots: Mapping[str, Path]) -> None:
    """Verify every declared file/hash for all five source artifacts. Raises on the
    first missing local root or the first hash/size mismatch -- before any output
    directory is created and before any adapter or base model is loaded."""
    for artifact_id in EXPECTED_ARTIFACT_IDS:
        if artifact_id not in roots or roots[artifact_id] is None:
            raise EndpointPreparationError(f"no local root supplied for source artifact {artifact_id!r}")
        verify_local_artifact(source_manifest, artifact_id, roots[artifact_id])


def _stream_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hash_optional_file(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Hash a caller-supplied audit file (training/launch script). Only its filename is
    recorded, never the operator's absolute local path."""
    if not path:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise EndpointPreparationError(f"declared file for hashing does not exist: {p}")
    return {"filename": p.name, "sha256": _stream_sha256(p), "size_bytes": p.stat().st_size}


def verify_flip_lineage(
    source_manifest: Mapping[str, Any], archive_root: Path,
    training_script_path: Optional[str] = None, launch_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Confirm the frozen archived trainer-state path declared for `pair_a_flip_adapter`
    exists under `archive_root`, hash it, and require it recorded the expected final
    training step. Optionally hash a training script and/or launch config for audit.

    This is deliberately narrow, and deliberately never claims more than it can prove:
    `trainer_state.json` (loss/LR/step history) establishes that a training run reached
    step `EXPECTED_FLIP_TRAINING_STEP`, nothing about how the training data was
    constructed. Hashing an independently-supplied training/launch script records that
    *a* script implementing a label swap exists and matches a given hash -- it does not
    establish that *this archived run* actually used it. `label_swap_lineage_verified`
    and `confirmatory_eligible` are therefore always recorded `False` here; see the
    completion report for what archived evidence or rerun would resolve this.
    """
    by_id = {a["artifact_id"]: a for a in source_manifest["artifacts"]}
    flip = by_id.get("pair_a_flip_adapter")
    if flip is None:
        raise EndpointPreparationError("source manifest has no pair_a_flip_adapter entry")
    declared_path = (flip.get("lineage") or {}).get("archived_trainer_state_path")
    if not declared_path:
        raise EndpointPreparationError(
            "pair_a_flip_adapter manifest entry has no lineage.archived_trainer_state_path"
        )

    root = Path(archive_root).resolve()
    full_path = (root / declared_path).resolve()
    if not full_path.is_relative_to(root):
        raise EndpointPreparationError(f"declared trainer-state path {declared_path!r} resolves outside {root}")
    if not full_path.exists() or not full_path.is_file():
        raise EndpointPreparationError(
            f"flip lineage verification failed: archived trainer state not found at {full_path}. "
            "The adapter's presence alone does not establish label-swap lineage."
        )

    sha256 = _stream_sha256(full_path)
    try:
        state = json.loads(full_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise EndpointPreparationError(f"archived trainer state at {full_path} is not valid JSON: {e}") from e
    step = state.get("global_step")
    if step is None:
        raise EndpointPreparationError(f"archived trainer state at {full_path} has no 'global_step' field")
    if step != EXPECTED_FLIP_TRAINING_STEP:
        raise EndpointPreparationError(
            f"archived trainer state at {full_path} has global_step={step}, expected {EXPECTED_FLIP_TRAINING_STEP}"
        )

    return {
        "archived_trainer_state_path": declared_path, "sha256": sha256, "training_step": step,
        "training_script": _hash_optional_file(training_script_path),
        "launch_config": _hash_optional_file(launch_config_path),
        "label_swap_lineage_verified": False,
        "confirmatory_eligible": False,
    }


# Model/tokenizer I/O -- every real Transformers/PEFT/torch call is deferred-imported
# inside one of these thin wrappers, so `--dry-run` and offline tests never need any of
# those packages installed, and none of them is imported before the checks above run.


def _torch_dtype(name: str) -> Any:
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _load_base_config(root: Path) -> Any:
    """`trust_remote_code` is never set: these are standard, natively supported
    architectures, so unverified repository code must never execute."""
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(str(root), local_files_only=True)


def _load_tokenizer_from(root: Path) -> Any:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(root), local_files_only=True)


def _load_base_model(root: Path, dtype_name: str, device: str) -> Any:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(root), torch_dtype=_torch_dtype(dtype_name), local_files_only=True,
    )
    return model.to(device)


def _resize_token_embeddings(model: Any, new_size: int, seed: int) -> None:
    import torch
    torch.manual_seed(seed)
    model.resize_token_embeddings(new_size)


def _load_adapter(model: Any, adapter_root: Path) -> Any:
    """Attaches only the declared adapter to only the already-loaded, declared base
    model object -- `adapter_config.json`'s own `base_model_name_or_path` is never
    consulted to choose a base; the base is always the object the caller already
    verified and loaded."""
    from peft import PeftModel
    return PeftModel.from_pretrained(model, str(adapter_root))


def _merge_and_unload(peft_model: Any) -> Any:
    return peft_model.merge_and_unload()


def _forward_pass_smoke_test(model: Any, tokenizer: Any, device: str) -> None:
    """A no-generation forward pass on a tiny local input. Raises on any exception --
    never swallowed, since an endpoint that cannot run a forward pass must not be
    accepted."""
    import torch
    vocab_size = len(tokenizer)
    ids = torch.tensor([[i % vocab_size for i in range(1, 5)]], dtype=torch.long, device=device)
    with torch.no_grad():
        model(input_ids=ids)


def _save_endpoint(model: Any, tokenizer: Any, path: Path, max_shard_size: str) -> None:
    model.save_pretrained(str(path), safe_serialization=True, max_shard_size=max_shard_size)
    tokenizer.save_pretrained(str(path))


def _library_versions(include_peft: bool) -> Dict[str, str]:
    import torch
    import transformers as _transformers

    versions = {"torch": torch.__version__, "transformers": _transformers.__version__}
    if include_peft:
        import peft as _peft
        versions["peft"] = _peft.__version__
    return versions


# Tokenizer detection -- filesystem check first, load second; a missing tokenizer is a
# normal case, a present-but-broken one is a real failure and is never downgraded.


def _tokenizer_artifacts_present(root: Path) -> bool:
    return any((root / name).exists() for name in _TOKENIZER_ARTIFACT_NAMES)


def _load_adapter_tokenizer_if_present(root: Path) -> Optional[Any]:
    if not _tokenizer_artifacts_present(root):
        return None
    return _load_tokenizer_from(root)  # raises on failure -- never caught here


def _tokenizer_vocab(tokenizer: Any) -> Dict[str, int]:
    return dict(tokenizer.get_vocab())


def _tokenizer_special_ids(tokenizer: Any) -> Dict[str, Optional[int]]:
    return {name: getattr(tokenizer, name, None) for name in _SPECIAL_TOKEN_ATTRS}


# Validation


_PEFT_NAME_MARKERS = ("lora", "tuner", "modulestosavewrapper")


def _looks_like_peft_module(obj: Any) -> bool:
    """True if `obj` (a model or any submodule) still carries PEFT state by any of
    three independent signals, so a PEFT wrapper that happens to avoid one of them
    (e.g. a non-LoRA tuner with no "lora" in its class name) is still caught:

    - it exposes a `peft_config` attribute (PEFT's own marker of an active wrapper);
    - its class (or the class of the module holding it) is defined in the `peft`
      package, regardless of what the class is named;
    - its class name contains a known PEFT-family marker ("lora", "tuner",
      "modulestosavewrapper" -- covering LoRA, other tuner types, and PEFT's
      save-both-branches wrapper for modules_to_save).
    """
    if hasattr(obj, "peft_config"):
        return True
    cls = type(obj)
    module_name = getattr(cls, "__module__", "") or ""
    if module_name == "peft" or module_name.startswith("peft."):
        return True
    cls_name = cls.__name__.lower()
    return any(marker in cls_name for marker in _PEFT_NAME_MARKERS)


def _has_no_residual_peft_modules(model: Any) -> bool:
    if _looks_like_peft_module(model):
        return False
    return not any(_looks_like_peft_module(m) for _, m in model.named_modules())


def _embeddings_are_equal(model: Any) -> bool:
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        return True
    import torch
    return bool(torch.equal(model.get_input_embeddings().weight, lm_head.weight))


def validate_endpoint(model: Any, tokenizer: Any, device: str) -> Dict[str, Any]:
    """Run every Gate 0 validation check against a loaded (real, or in tests, fake)
    model/tokenizer pair -- for every endpoint, direct or merged. Raises
    `EndpointPreparationError` on the first failing check; returns the recorded
    validation facts only once every check, including positive-value checks on every
    architecture fact, has passed."""
    config = model.config
    model_type = getattr(config, "model_type", None)
    if not model_type:
        raise EndpointPreparationError("config.model_type is missing or empty")
    hidden_size = getattr(config, "hidden_size", None)
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
        raise EndpointPreparationError(f"config.hidden_size must be a positive int, got {hidden_size!r}")
    num_layers = getattr(config, "num_hidden_layers", None)
    if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers <= 0:
        raise EndpointPreparationError(f"config.num_hidden_layers must be a positive int, got {num_layers!r}")

    embedding_rows = int(model.get_input_embeddings().weight.shape[0])
    lm_head = model.get_output_embeddings()
    lm_head_rows = int(lm_head.weight.shape[0]) if lm_head is not None else embedding_rows
    vocab_size = len(tokenizer)
    if vocab_size <= 0 or embedding_rows <= 0 or lm_head_rows <= 0:
        raise EndpointPreparationError(
            f"vocab_size={vocab_size}, embedding_rows={embedding_rows}, lm_head_rows={lm_head_rows} must all be positive"
        )
    tied = bool(getattr(config, "tie_word_embeddings", False))

    if not check_vocab_consistency(vocab_size, embedding_rows, lm_head_rows):
        raise EndpointPreparationError(
            f"vocab_size={vocab_size} exceeds embedding_rows={embedding_rows} or lm_head_rows={lm_head_rows}"
        )

    embeddings_equal = _embeddings_are_equal(model) if tied else True
    if not check_tied_embedding_consistency(tied, embeddings_equal):
        raise EndpointPreparationError("config declares tied embeddings but embedding/LM-head weights differ")

    no_residual = _has_no_residual_peft_modules(model)
    if not no_residual:
        raise EndpointPreparationError("endpoint still has residual PEFT/LoRA modules")

    _forward_pass_smoke_test(model, tokenizer, device)  # raises on failure

    vocab = _tokenizer_vocab(tokenizer)
    return {
        "model_type": model_type, "hidden_size": hidden_size, "num_hidden_layers": num_layers,
        "vocab_size": vocab_size, "embedding_rows": embedding_rows, "lm_head_rows": lm_head_rows,
        "tied_embeddings": tied, "tokenizer_loadable": True,
        "no_residual_peft_modules": no_residual, "forward_pass_smoke_test": True,
        "tokenizer_fingerprint": tokenizer_fingerprint(vocab),
        "special_token_ids": _tokenizer_special_ids(tokenizer),
    }


def _hash_output_files(root: Path) -> List[Dict[str, Any]]:
    """Hash every regular inference file under `root`, direct or merged endpoint alike.
    A symlink resolving outside `root` is rejected outright rather than followed;
    a symlink resolving inside `root` is hashed like any other file. Returned in
    stable (path-sorted) order."""
    root = root.resolve()
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            target = p.resolve()
            if not target.is_relative_to(root):
                raise EndpointPreparationError(f"refusing to hash a symlink escaping the endpoint directory: {p} -> {target}")
            if not target.is_file():
                continue
        elif not p.is_file():
            continue
        files.append({
            "path": str(p.relative_to(root)).replace("\\", "/"),
            "sha256": _stream_sha256(p),
            "size_bytes": p.stat().st_size,
        })
    return sorted(files, key=lambda f: f["path"])


# Endpoint resolution


def _source_location(source_manifest: Mapping[str, Any], artifact_id: str) -> Dict[str, Any]:
    """A direct endpoint's weights are never copied into the output bundle, so it has
    no bundle path -- its location is the stable source locator the caller already
    verified this exact endpoint against (`repository`/`revision`/`subpath` from the
    frozen source manifest), never the operator's machine-specific local root."""
    by_id = {a["artifact_id"]: a for a in source_manifest["artifacts"]}
    artifact = by_id[artifact_id]
    return {
        "kind": "source",
        "repository": artifact["repository"],
        "revision": artifact["revision"],
        "subpath": artifact.get("subpath", ""),
    }


def _resolve_direct_endpoint(
    spec: Mapping[str, Any], source_manifest: Mapping[str, Any], roots: Mapping[str, Path], dtype_name: str, device: str,
) -> Dict[str, Any]:
    """A direct endpoint is loaded (read-only) for validation and never written --
    the already-verified source root's weights are never copied or rewritten merely to
    normalize layout."""
    role, artifact_id = spec["role"], spec["source_artifact_id"]
    root = roots[artifact_id]
    _load_base_config(root)  # config-loadability check; facts folded into validate_endpoint via model.config
    tokenizer = _load_tokenizer_from(root)
    model = _load_base_model(root, dtype_name, device)
    validation = validate_endpoint(model, tokenizer, device)
    files = _hash_output_files(root)
    return {
        "role": role, "status": "direct",
        "source_artifact_id": artifact_id, "base_artifact_id": None,
        "location": _source_location(source_manifest, artifact_id), "merge": None, "device": device,
        "library_versions": _library_versions(include_peft=False),
        "validation": validation, "files": files,
    }


def _resolve_merged_endpoint(
    spec: Mapping[str, Any], roots: Mapping[str, Path], staging_dir: Path,
    dtype_name: str, max_shard_size: str, device: str, flip_lineage: Mapping[str, Any],
) -> Dict[str, Any]:
    """Load the declared base, resolve the tokenizer relationship, resize only for a
    verified append-only extension, attach only the declared adapter, merge exactly
    once, save into the staging directory, then reload the config/tokenizer/model from
    what was actually written and validate *that* -- never the in-memory pre-save
    model."""
    role = spec["role"]
    adapter_artifact_id, base_artifact_id = spec["source_artifact_id"], spec["base_artifact_id"]
    base_root, adapter_root = roots[base_artifact_id], roots[adapter_artifact_id]

    base_tokenizer = _load_tokenizer_from(base_root)
    adapter_tokenizer = _load_adapter_tokenizer_if_present(adapter_root)
    base_model = _load_base_model(base_root, dtype_name, device)

    if adapter_tokenizer is None:
        resolved_tokenizer = base_tokenizer
        base_vocab = _tokenizer_vocab(base_tokenizer)
        tokenizer_resolution: Dict[str, Any] = {
            "status": "base_only",
            "base_vocab_size": len(base_vocab), "adapter_vocab_size": None,
            "old_vocab_size": None, "new_vocab_size": None, "added_token_ids": [],
            "base_fingerprint": tokenizer_fingerprint(base_vocab), "adapter_fingerprint": None,
            "base_special_token_ids": _tokenizer_special_ids(base_tokenizer), "adapter_special_token_ids": None,
        }
    else:
        base_vocab = _tokenizer_vocab(base_tokenizer)
        adapter_vocab = _tokenizer_vocab(adapter_tokenizer)
        base_special = _tokenizer_special_ids(base_tokenizer)
        adapter_special = _tokenizer_special_ids(adapter_tokenizer)
        compat = check_tokenizer_extension(base_vocab, adapter_vocab, base_special, adapter_special)  # raises on incompatibility
        resolved_tokenizer = adapter_tokenizer
        if compat["status"] == "append_only_extension":
            _resize_token_embeddings(base_model, compat["new_vocab_size"], MERGE_SEED)
        tokenizer_resolution = {
            "status": compat["status"],
            "base_vocab_size": len(base_vocab), "adapter_vocab_size": len(adapter_vocab),
            "old_vocab_size": compat["old_vocab_size"], "new_vocab_size": compat["new_vocab_size"],
            "added_token_ids": compat["added_token_ids"],
            "base_fingerprint": tokenizer_fingerprint(base_vocab), "adapter_fingerprint": tokenizer_fingerprint(adapter_vocab),
            "base_special_token_ids": base_special, "adapter_special_token_ids": adapter_special,
        }

    peft_model = _load_adapter(base_model, adapter_root)
    merged_model = _merge_and_unload(peft_model)

    # Validate the in-memory merged object immediately -- before staging is touched at
    # all. The independent reload-and-validate below (of what was actually written) is
    # not a substitute for this: a save/reload path that happens to produce a clean
    # object would otherwise let residual PEFT state on the in-memory merge through
    # unnoticed. Both checks are required; neither alone is sufficient.
    validate_endpoint(merged_model, resolved_tokenizer, device)

    staging_role_dir = staging_dir / role
    staging_role_dir.mkdir(parents=True)
    _save_endpoint(merged_model, resolved_tokenizer, staging_role_dir, max_shard_size)

    # Validate what was actually written, not the in-memory pre-save objects.
    del merged_model, base_model, peft_model, resolved_tokenizer
    _load_base_config(staging_role_dir)
    reloaded_tokenizer = _load_tokenizer_from(staging_role_dir)
    reloaded_model = _load_base_model(staging_role_dir, dtype_name, device)
    validation = validate_endpoint(reloaded_model, reloaded_tokenizer, device)
    files = _hash_output_files(staging_role_dir)

    # Every merge-critical adapter input file, not only the two source-manifest-pinned
    # anchor files -- the base's own inputs are already fully recorded under the
    # corresponding direct endpoint elsewhere in this same manifest.
    adapter_input_files = _hash_output_files(adapter_root)

    merge_details: Dict[str, Any] = {
        "dtype": dtype_name, "max_shard_size": max_shard_size, "merge_seed": MERGE_SEED,
        "tokenizer_resolution": tokenizer_resolution, "adapter_input_files": adapter_input_files,
    }
    if role == "M--A":
        merge_details["flip_lineage"] = dict(flip_lineage)

    return {
        "role": role, "status": "merged",
        "source_artifact_id": adapter_artifact_id, "base_artifact_id": base_artifact_id,
        "location": {"kind": "bundle", "path": role}, "merge": merge_details, "device": device,
        "library_versions": _library_versions(include_peft=True),
        "validation": validation, "files": files,
    }


def _print_plan(plan: Sequence[Mapping[str, Any]], source_manifest: Mapping[str, Any], roots: Mapping[str, Path], output_dir: Path, device: str) -> None:
    print(f"source manifest hash: {source_manifest['manifest_hash']}")
    print(f"output dir: {output_dir}")
    print(f"device: {device}")
    for spec in plan:
        role, status = spec["role"], spec["status"]
        if status == "direct":
            print(f"  {role}: direct  <- {spec['source_artifact_id']} @ {roots[spec['source_artifact_id']]}")
        else:
            print(
                f"  {role}: merge   <- adapter {spec['source_artifact_id']} @ {roots[spec['source_artifact_id']]} "
                f"INTO base {spec['base_artifact_id']} @ {roots[spec['base_artifact_id']]}"
            )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare deterministic inference endpoints and close Gate 0.")
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST_PATH))
    parser.add_argument("--pair-a-sft-root", required=True)
    parser.add_argument("--pair-a-dpo-root", required=True)
    parser.add_argument("--pair-b-sft-root", required=True)
    parser.add_argument("--pair-a-flip-adapter-root", required=True)
    parser.add_argument("--pair-b-dpo-adapter-root", required=True)
    parser.add_argument(
        "--pair-a-flip-archive-root", required=True,
        help="Local root of the archived results repository snapshot containing the "
             "flip run's trainer-state checkpoint directory (a wider root than "
             "--pair-a-flip-adapter-root, which only holds the final adapter files).",
    )
    parser.add_argument(
        "--flip-training-script", default=None,
        help="Optional: path to the training script implementing the label swap. "
             "Hashed and recorded for audit only -- does not by itself verify this "
             "archived adapter was produced by it.",
    )
    parser.add_argument(
        "--flip-launch-config", default=None,
        help="Optional: path to the flip run's launch/run configuration. Hashed and "
             "recorded for audit only.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=_VALID_DTYPE_NAMES)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument(
        "--device", default="cuda:0",
        help="Explicit device for the eventual A100 execution, e.g. cuda:0 or cpu. "
             "Every endpoint is loaded onto exactly this device (no device_map='auto' "
             "sharding); the recorded RAM/VRAM budget assumes this placement.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Verify sources and lineage and print the exact endpoint plan; import no "
             "model library, load no model, merge nothing, and write no output.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    try:
        source_manifest = load_source_manifest(args.source_manifest)
        validate_source_manifest_structure(source_manifest)
        validate_frozen_source_identity(source_manifest)
    except ArtifactIdentityError as e:
        print(f"FAIL: source manifest verification: {e}", file=sys.stderr)
        return 1

    roots = {
        "pair_a_sft": Path(args.pair_a_sft_root),
        "pair_a_dpo": Path(args.pair_a_dpo_root),
        "pair_b_sft": Path(args.pair_b_sft_root),
        "pair_a_flip_adapter": Path(args.pair_a_flip_adapter_root),
        "pair_b_dpo_adapter": Path(args.pair_b_dpo_adapter_root),
    }

    try:
        verify_all_sources(source_manifest, roots)
        flip_lineage = verify_flip_lineage(
            source_manifest, Path(args.pair_a_flip_archive_root),
            training_script_path=args.flip_training_script, launch_config_path=args.flip_launch_config,
        )
    except (ArtifactIdentityError, EndpointPreparationError) as e:
        print(f"FAIL: source/lineage verification: {e}", file=sys.stderr)
        return 1

    plan = plan_endpoints()
    output_dir = Path(args.output_dir)

    if args.dry_run:
        _print_plan(plan, source_manifest, roots, output_dir, args.device)
        return 0

    if not flip_lineage["confirmatory_eligible"]:
        print(
            "NOTE: M--A label-swap lineage is not verified from archived material alone. "
            "M--A will still be prepared and validated, but recorded as "
            "confirmatory_eligible=false. Gate 0 is not being reported closed for M--A.",
            file=sys.stderr,
        )

    if output_dir.exists():
        print(f"FAIL: {output_dir} already exists; refusing to overwrite an existing output directory", file=sys.stderr)
        return 1

    staging_dir = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    candidate: Optional[Dict[str, Any]] = None
    try:
        staging_dir.mkdir(parents=True)
        entries: List[Dict[str, Any]] = []
        for spec in plan:
            if spec["status"] == "direct":
                entry = _resolve_direct_endpoint(spec, source_manifest, roots, args.dtype, args.device)
            else:
                entry = _resolve_merged_endpoint(
                    spec, roots, staging_dir, args.dtype, args.max_shard_size, args.device, flip_lineage,
                )
            entries.append(entry)

        candidate = build_endpoint_manifest(source_manifest["manifest_hash"], entries)
        candidate_path = staging_dir / "endpoint_manifest_candidate_v1.json"
        save_endpoint_manifest(candidate, candidate_path)
        # Load back what was actually written and re-run full structural validation --
        # not a trust-what-we-just-built shortcut. `load_manifest` verifies the self-hash
        # and validates structure; a hash-consistent but structurally invalid payload
        # must still fail here, before the one promotion below.
        load_endpoint_manifest(candidate_path)
    except Exception as e:  # noqa: BLE001 -- any failure anywhere here must promote nothing
        shutil.rmtree(staging_dir, ignore_errors=True)
        print(f"FAIL: endpoint preparation: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if output_dir.exists():  # race guard immediately before the one promotion
        shutil.rmtree(staging_dir, ignore_errors=True)
        print(f"FAIL: {output_dir} appeared during preparation; refusing to promote", file=sys.stderr)
        return 1
    try:
        staging_dir.replace(output_dir)  # the single atomic promotion for the whole run
    except OSError as e:
        shutil.rmtree(staging_dir, ignore_errors=True)
        print(f"FAIL: promotion failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"candidate manifest hash: {candidate['manifest_hash']}")
    if not flip_lineage["confirmatory_eligible"]:
        print("M--A: confirmatory_eligible=false (label-swap lineage not verified from archived material)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
