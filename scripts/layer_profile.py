"""Layer-wise activation analysis comparing IT and DPO models.

For each transformer layer, measures:
1. Cosine similarity between IT and DPO hidden states (mean across prompts)
2. L2 norm of the activation difference (mean across prompts)
3. SVD of the (h_DPO - h_IT) matrix across prompts:
   - PC0 explained variance ratio (is the shift low-rank?)
   - Top-3 PCs explained variance (cumulative)
4. Angle between mean direction and PC0 (is the mean a good steering vector?)

Output:
- layer_stats.csv with one row per layer
- cosine_similarity.png  (per-layer curve)
- explained_variance.png (PC0 variance per layer)
- norm_curve.png         (||h_DPO - h_IT|| per layer)
- summary.txt            (text summary of findings)

Readout position is configurable, because the two are not interchangeable:

  prompt_last     final token of the prompt, where the model is about to generate.
                  Most relevant to steering generation behaviour.
  response_last   final token of prompt + chosen response, which is what the paper's
                  Eq. 1 uses. Requires a dataset with reference responses (hh_rlhf).

Activations are read with forward hooks on the decoder layers rather than from
`output_hidden_states`. That tuple is [embeddings, L0_out, ..., L(n-2)_out, norm(L(n-1)_out)]
-- its final entry is the post-final-norm output, so the raw last-layer residual stream is
not in it at all, and indexing it as if it were makes the last row incomparable to the rest.

Usage:
    python scripts/layer_profile.py --model-config configs/llama3_oh.yaml --eval-config configs/layer_profile.yaml

This script does NOT require the judge server - it's pure tensor computation.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from steering.activations import LastTokenCapture, build_input_text
from steering.artifacts import GpuMonitor, TensorCheckpoint, sync_to_hub, write_run_metadata
from steering.config import ModelConfig
from steering.data import load_hh_rlhf_test, load_harmfulqa_partition
from steering.endpoint_binding import resolve_model_source
from steering.models import load_model, load_tokenizer
from steering.splits import load_manifest, validate_manifest_identity
from steering.utils import load_yaml, resolve_device, set_all_seeds, setup_logging

#: Frozen primary construction requirements (protocol Sections 6, 13 Gate 1; Task 014).
#: An endpoint-backed activation-extraction run is only protocol_profile="primary_v1"
#: when it matches every one of these exactly -- see classify_activation_profile below.
PRIMARY_CONSTRUCTION_SOURCE = "harmfulqa"
PRIMARY_CONSTRUCTION_PARTITION = "construction"
PRIMARY_CONSTRUCTION_RECORD_COUNT = 1378
PRIMARY_TOKEN_POSITION = "prompt_last"

PROTOCOL_PROFILES = ("primary_v1", "legacy_nonconfirmatory")


class ProtocolProfileError(ValueError):
    """An endpoint-backed activation-extraction run does not satisfy the frozen
    primary construction requirements."""


# Config


@dataclass
class LayerProfileConfig:
    """Settings for layer profile analysis."""

    # Source of prompts to extract activations from.
    # Options: "hh_rlhf", "harmfulqa", "mixed"
    prompt_source: str = "hh_rlhf"

    # Which frozen manifest partition to use when prompt_source == "harmfulqa". Protocol
    # construction (Section 6) is the only partition permitted here; membership and
    # order come entirely from the manifest, not from n_prompts/seed.
    prompt_partition: Optional[str] = None

    # Number of prompts to use. Not consulted when prompt_source == "harmfulqa": the
    # named partition is loaded in full.
    n_prompts: int = 100

    # Readout position: "prompt_last" or "response_last".
    # "response_last" needs reference responses, so it requires prompt_source = "hh_rlhf".
    token_position: str = "prompt_last"

    # Maximum input length (tokens) - prompts longer than this are truncated.
    max_input_length: int = 512

    # Flush the activation checkpoint every N prompts, so a dropped connection or an
    # expired rental costs at most this many forward passes.
    checkpoint_every: int = 200

    # Output location.
    output_dir: str = "outputs/layer_profile"

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "LayerProfileConfig":
        raw = load_yaml(path)
        return cls(**raw)


def main():
    parser = argparse.ArgumentParser(description="Run layer-wise activation profile analysis.")
    parser.add_argument(
        "--model-config", default=None,
        help="Legacy YAML model-pair config. Historical/non-confirmatory use only; "
             "mutually exclusive with endpoint-backed mode below.",
    )
    parser.add_argument(
        "--endpoint-manifest", default=None,
        help="Endpoint-backed mode: candidate endpoint manifest path (protocol Section 4).",
    )
    parser.add_argument(
        "--endpoint-bundle-root", default=None,
        help="Endpoint-backed mode: root directory holding merged endpoint bundles.",
    )
    parser.add_argument(
        "--pair", choices=["A", "B"], default=None,
        help="Endpoint-backed mode: which model pair to resolve from the candidate manifest.",
    )
    parser.add_argument(
        "--endpoint-source", action="append", default=[], metavar="ARTIFACT_ID=LOCAL_ROOT",
        help="Endpoint-backed mode: repeatable, local root for one direct source artifact "
             "(e.g. pair_a_sft=/data/pair_a/SFT_merged). Not needed for a merged endpoint, "
             "which resolves entirely under --endpoint-bundle-root.",
    )
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sync", action="store_true",
        help="Push the finished run to the results dataset on the hub.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Ignore any existing checkpoint.")
    parser.add_argument(
        "--device", default="auto",
        help="auto | cuda | cpu, for the per-layer statistics after extraction.",
    )
    parser.add_argument(
        "--hourly-rate", type=float, default=None,
        help="Instance $/hr, to record an estimated cost alongside GPU usage.",
    )
    args = parser.parse_args()

    # Endpoint verification (frozen-source binding, candidate-manifest hash/structural
    # validation, and per-file SHA-256/size streaming, for endpoint-backed mode)
    # happens here, before anything below touches output, logging, a checkpoint, or a
    # model/GPU resource. A mismatch has no side effects.
    model_cfg, endpoint_meta = resolve_model_source(
        model_config=args.model_config, endpoint_manifest=args.endpoint_manifest,
        endpoint_bundle_root=args.endpoint_bundle_root, pair=args.pair, endpoint_source=args.endpoint_source,
    )
    endpoint_backed = endpoint_meta is not None

    eval_cfg = LayerProfileConfig.from_yaml(args.eval_config)
    validate_harmfulqa_construction_config(eval_cfg)

    # Phase 1 of endpoint-backed primary construction eligibility (protocol Sections 6,
    # 13 Gate 1): prompt_source/prompt_partition/token_position are checkable from
    # config alone, so they are rejected here -- before any prompt loader runs, before
    # set_all_seeds (CUDA), output creation, metadata writing, logging, checkpoint
    # deletion, or any model/tokenizer/GPU access.
    validate_endpoint_backed_construction_config(endpoint_backed, eval_cfg)

    # Read-only preparation, safe before set_all_seeds (CUDA) and phase 2 of the
    # protocol-profile check below: loading prompts (and, for HarmfulQA, the frozen
    # manifest partition they came from) mutates nothing on disk.
    prompts = load_prompts(eval_cfg.prompt_source, eval_cfg.n_prompts, args.seed, eval_cfg.prompt_partition)
    if eval_cfg.token_position == "response_last":
        prompts = [p for p in prompts if p.get("chosen")]
        if not prompts:
            raise ValueError(
                "response_last needs reference responses; prompt_source must be hh_rlhf"
            )

    # Phase 2: record count and manifest-verified identity, which need the prompts
    # already loaded -- still before set_all_seeds, output creation, metadata writing,
    # logging, checkpoint deletion, or any model/tokenizer/GPU access. Legacy runs are
    # always legacy_nonconfirmatory, unconditionally.
    protocol_profile = classify_activation_profile(endpoint_backed, eval_cfg, prompts)
    loading_policy = _model_loading_policy(endpoint_backed)

    set_all_seeds(args.seed)

    output_root = output_root_for(eval_cfg.output_dir, model_cfg.name, eval_cfg.prompt_source, eval_cfg.prompt_partition)

    run_meta_extra = None
    if eval_cfg.prompt_source == "harmfulqa":
        prov = harmfulqa_provenance(prompts)
        run_meta_extra = {
            "harmfulqa_partition": prov["partition"],
            "harmfulqa_manifest_hash": prov["manifest_hash"],
            "harmfulqa_record_count": len(prompts),
        }
    run_meta_extra = dict(run_meta_extra or {})
    run_meta_extra["protocol_profile"] = protocol_profile
    run_meta_extra["endpoint_backed"] = endpoint_backed
    run_meta_extra["model_loading_policy"] = loading_policy
    if endpoint_meta is not None:
        run_meta_extra["endpoint"] = endpoint_meta
    write_run_metadata(
        output_root,
        config=build_run_config(model_cfg, eval_cfg, args),
        extra=run_meta_extra,
        argv=list(sys.argv),
    )

    # Nothing below may run until the identity check above has succeeded: the log is
    # not initialised, no stale checkpoint is removed, and no model is loaded before
    # this point -- a mismatched resume fails here, before it can touch anything.
    log = setup_logging(output_root / "layer_profile.log")
    log.info(f"Model pair: {model_cfg.name}")
    log.info(f"Output: {output_root}")
    log.info(f"Architecture: {model_cfg.architecture}, layers: {model_cfg.num_layers}")
    log.info(f"Loaded {len(prompts)} prompts from {eval_cfg.prompt_source} (readout: {eval_cfg.token_position})")
    log.info(f"protocol_profile={protocol_profile}")

    if args.no_resume:
        for stale in output_root.glob("*.partial.pt"):
            stale.unlink()
            log.info(f"Removed checkpoint {stale}")

    # Load tokenizer
    tokenizer_path = model_cfg.tokenizer_id or model_cfg.it_model
    tokenizer_subfolder = model_cfg.tokenizer_subfolder or (model_cfg.it_subfolder or "")
    tokenizer = load_tokenizer(tokenizer_path, subfolder=tokenizer_subfolder, **loading_policy)

    # Extract activations from both models on the same prompts.
    with GpuMonitor(output_root, hourly_rate=args.hourly_rate) as gpu:
        log.info("\n" + "=" * 60)
        log.info("Extracting activations from IT model")
        log.info("=" * 60)
        activations_it = extract_activations(
            model_cfg.it_model, model_cfg.it_subfolder or "",
            tokenizer, prompts, eval_cfg,
            checkpoint_path=output_root / "it.partial.pt",
            loading_policy=loading_policy,
        )
        log.info(f"IT activations shape: {activations_it.shape}  (layers, prompts, hidden)")

        log.info("\n" + "=" * 60)
        log.info("Extracting activations from DPO model")
        log.info("=" * 60)
        activations_dpo = extract_activations(
            model_cfg.dpo_model, model_cfg.dpo_subfolder or "",
            tokenizer, prompts, eval_cfg,
            checkpoint_path=output_root / "dpo.partial.pt",
            loading_policy=loading_policy,
        )
        log.info(f"DPO activations shape: {activations_dpo.shape}")

    log.info(f"GPU usage: {json.dumps(gpu.summary())}")

    # Save raw activations for later reuse (steering vector building). For the
    # manifest-backed HarmfulQA path, provenance travels with the tensors so a later
    # consumer (steer_sweep's vector construction) can verify what produced them rather
    # than trust a directory name.
    activation_payload = {"it": activations_it, "dpo": activations_dpo}
    if eval_cfg.prompt_source == "harmfulqa":
        activation_payload.update(harmfulqa_provenance(prompts))
    torch.save(activation_payload, output_root / "activations.pt")
    log.info(f"Saved raw activations to {output_root}/activations.pt")

    # Both tensors are on disk in their final form; the partials are now redundant.
    for name in ("it.partial.pt", "dpo.partial.pt"):
        (output_root / name).unlink(missing_ok=True)

    # Compute per-layer statistics.
    log.info("\n" + "=" * 60)
    log.info("Computing per-layer statistics")
    log.info("=" * 60)
    stats_df = compute_layer_stats(activations_it, activations_dpo,
                                   device=resolve_device(args.device))
    stats_df.to_csv(output_root / "layer_stats.csv", index=False)
    log.info(f"Saved stats to {output_root}/layer_stats.csv")
    log.info("\n" + stats_df.to_string(index=False))

    # Generate plots.
    plot_curves(stats_df, output_root)
    log.info(f"Saved plots to {output_root}/")

    # Text summary of findings.
    summary = summarize_findings(stats_df, model_cfg.num_layers)
    (output_root / "summary.txt").write_text(summary, encoding="utf-8")
    log.info("\n" + summary)

    if args.sync:
        sync_to_hub(output_root, experiment="layer_profile", model=model_cfg.name)


# Prompt loading


def build_run_config(model_cfg: ModelConfig, eval_cfg: LayerProfileConfig, args: argparse.Namespace) -> Dict[str, Any]:
    """The resolved run identity for `write_run_metadata`: the model/eval dataclasses
    plus the complete parsed CLI namespace, including defaulted values, under `cli`."""
    return {"model": asdict(model_cfg), "eval": asdict(eval_cfg), "cli": vars(args)}


def output_root_for(base_output_dir: str, model_name: str, prompt_source: str, prompt_partition: Optional[str]) -> Path:
    """<output_dir>/<model_name>/, with a partition-specific suffix for manifest-backed
    HarmfulQA construction so it cannot resume on a historical or other-partition
    checkpoint."""
    root = Path(base_output_dir) / model_name
    if prompt_source == "harmfulqa":
        root = root / prompt_partition
    return root


def validate_harmfulqa_construction_config(cfg: LayerProfileConfig) -> None:
    """Protocol construction (Section 6) uses the frozen manifest, not ad hoc sampling.

    `mixed` always includes a 50/50 ad hoc HarmfulQA component (see `load_prompts`), so
    it is not a protocol construction path and is rejected outright rather than silently
    drawing HarmfulQA prompts outside the manifest.
    """
    if cfg.prompt_source == "harmfulqa":
        if cfg.prompt_partition != "construction":
            raise ValueError(
                "prompt_source='harmfulqa' requires prompt_partition='construction' "
                f"(the only protocol construction partition); got {cfg.prompt_partition!r}"
            )
        if cfg.token_position != "prompt_last":
            raise ValueError(
                "prompt_source='harmfulqa' requires token_position='prompt_last'; "
                f"got {cfg.token_position!r}"
            )
    elif cfg.prompt_source == "mixed":
        raise ValueError(
            "prompt_source='mixed' is not a protocol construction path -- it always "
            "includes an ad hoc HarmfulQA component outside the frozen manifest. Use "
            "'hh_rlhf', or 'harmfulqa' with prompt_partition='construction', instead."
        )


# Protocol-profile classification (Task 014)


def _default_harmfulqa_manifest_path() -> Path:
    """manifests/harmfulqa_v1.json relative to the repository root, independent of the
    caller's current working directory. Mirrors steer_sweep.py's own helper of the
    same name -- kept file-local rather than shared, since both are tiny and the two
    scripts otherwise share no module."""
    return Path(__file__).resolve().parents[1] / "manifests" / "harmfulqa_v1.json"


def _construction_records(manifest: Dict) -> List[Dict]:
    """This manifest's construction-partition records, in ascending permuted_position."""
    return sorted(
        (e for e in manifest["records"] if e["partition"] == "construction"),
        key=lambda e: e["permuted_position"],
    )


def verify_construction_prompt_identity(prompts: List[Dict], manifest_path: Optional[Union[str, Path]] = None) -> None:
    """Re-verify the loaded construction prompts against the frozen manifest itself --
    not merely trusted because `load_harmfulqa_partition` is the one path that
    produces them -- so a corrupted, reordered, or drifted prompt list is caught
    before it ever reaches activation extraction. Mirrors
    `steer_sweep.validate_construction_activations`'s not-just-internal-self-consistency
    approach, applied to the prompt list rather than the post-extraction tensor.
    """
    path = Path(manifest_path) if manifest_path is not None else _default_harmfulqa_manifest_path()
    manifest = load_manifest(path)
    validate_manifest_identity(manifest)

    construction = _construction_records(manifest)
    expected_source_ids = [e["source_id"] for e in construction]
    expected_prompt_hashes = [e["prompt_hash"] for e in construction]

    actual_source_ids = [p["source_id"] for p in prompts]
    actual_prompt_hashes = [p["prompt_hash"] for p in prompts]

    if actual_source_ids != expected_source_ids[:len(actual_source_ids)]:
        raise ProtocolProfileError(
            "construction prompt source IDs do not exactly match the frozen manifest's "
            "ordered construction partition -- altered, reordered, or drifted provenance"
        )
    if actual_prompt_hashes != expected_prompt_hashes[:len(actual_prompt_hashes)]:
        raise ProtocolProfileError(
            "construction prompt hashes do not exactly match the frozen manifest's "
            "ordered construction partition -- altered, reordered, or drifted provenance"
        )


def validate_endpoint_backed_construction_config(endpoint_backed: bool, eval_cfg: LayerProfileConfig) -> None:
    """Phase 1: everything about endpoint-backed primary construction eligibility that
    is checkable from config alone -- prompt_source, prompt_partition, and
    token_position -- before `load_prompts()` is ever called. Record count and
    identity (phase 2, `classify_activation_profile`) require the prompts to already
    be loaded and so cannot be checked here.

    A no-op for legacy (non-endpoint-backed) runs: legacy prompt-source flexibility
    (hh_rlhf, and whatever `validate_harmfulqa_construction_config` already permits) is
    unrestricted by this function.
    """
    if not endpoint_backed:
        return

    problems: List[str] = []
    if eval_cfg.prompt_source != PRIMARY_CONSTRUCTION_SOURCE:
        problems.append(f"prompt_source must be {PRIMARY_CONSTRUCTION_SOURCE!r}, got {eval_cfg.prompt_source!r}")
    if eval_cfg.prompt_source == PRIMARY_CONSTRUCTION_SOURCE and eval_cfg.prompt_partition != PRIMARY_CONSTRUCTION_PARTITION:
        problems.append(f"prompt_partition must be {PRIMARY_CONSTRUCTION_PARTITION!r}, got {eval_cfg.prompt_partition!r}")
    if eval_cfg.token_position != PRIMARY_TOKEN_POSITION:
        problems.append(f"token_position must be {PRIMARY_TOKEN_POSITION!r}, got {eval_cfg.token_position!r}")

    if problems:
        raise ProtocolProfileError(
            "endpoint-backed activation extraction does not match the frozen primary "
            "construction protocol: " + "; ".join(problems)
        )


def classify_activation_profile(
    endpoint_backed: bool, eval_cfg: LayerProfileConfig, prompts: List[Dict],
    manifest_path: Optional[Union[str, Path]] = None,
) -> str:
    """Phase 2: assumes `validate_endpoint_backed_construction_config` (phase 1) has
    already passed -- prompt_source/prompt_partition/token_position are correct.
    Checks what can only be known once the prompts are actually loaded: the exact
    record count and re-verified identity against the frozen manifest. Returns
    "primary_v1" for the frozen primary construction path; "legacy_nonconfirmatory"
    for the historical (non-endpoint-backed) path, unconditionally, regardless of its
    own settings.

    An endpoint-backed run that drifts from the primary construction requirements is
    rejected outright (raises), not silently downgraded to legacy_nonconfirmatory -- a
    verified endpoint claiming a non-primary activation profile is a configuration
    error, not a legitimate secondary use. Nothing here reads a model, touches a GPU,
    or mutates output.
    """
    if not endpoint_backed:
        return "legacy_nonconfirmatory"

    if len(prompts) != PRIMARY_CONSTRUCTION_RECORD_COUNT:
        raise ProtocolProfileError(
            "endpoint-backed activation extraction does not match the frozen primary "
            f"construction protocol: construction record count must be exactly "
            f"{PRIMARY_CONSTRUCTION_RECORD_COUNT}, got {len(prompts)}"
        )
    verify_construction_prompt_identity(prompts, manifest_path)  # raises ProtocolProfileError on any mismatch
    return "primary_v1"


def _model_loading_policy(endpoint_backed: bool) -> Dict[str, bool]:
    """The effective model/tokenizer loading policy (Tasks 012/013): endpoint-backed
    loads are restricted to verified local files with no remote code execution; legacy
    loads keep the historical, more permissive defaults unchanged."""
    if endpoint_backed:
        return {"local_files_only": True, "trust_remote_code": False}
    return {"local_files_only": False, "trust_remote_code": True}


def harmfulqa_provenance(prompts: List[Dict]) -> Dict[str, object]:
    """Ordered source_ids/prompt_hashes plus the shared partition/manifest_hash, from
    HarmfulQA-partition records that must all belong to one partition and manifest.

    These arrays are saved alongside the activation tensors and align exactly with
    tensor prompt dimension 1, since `prompts` is never sliced or reordered after
    `load_harmfulqa_partition` returns it.
    """
    manifest_hashes = {p["manifest_hash"] for p in prompts}
    partitions = {p["partition"] for p in prompts}
    if len(manifest_hashes) > 1 or len(partitions) > 1:
        raise ValueError("prompts do not share a single partition/manifest_hash")
    return {
        "source_ids": [p["source_id"] for p in prompts],
        "prompt_hashes": [p["prompt_hash"] for p in prompts],
        "partition": next(iter(partitions)) if partitions else None,
        "manifest_hash": next(iter(manifest_hashes)) if manifest_hashes else None,
    }


def load_prompts(source: str, n: int, seed: int, partition: Optional[str] = None) -> List[Dict[str, str]]:
    """Load records with a "prompt" string and, where available, a "chosen" response.

    For HH-RLHF, we extract just the first user turn (single-turn for clean activations)
    and carry its reference response through for response_last readout.

    For HarmfulQA, `partition` must be "construction" (enforced by
    `validate_harmfulqa_construction_config` before this is called); the entire
    partition is returned in manifest order -- `n` and `seed` play no part in
    membership or order.
    """
    if source == "hh_rlhf":
        records = load_hh_rlhf_test(n=n, seed=seed)
        out = []
        for rec in records:
            p = rec["prompt"]
            if isinstance(p, list) and p:
                # Take the first user message only (avoid multi-turn confusion).
                user_msgs = [m for m in p if m.get("role") == "user"]
                if not user_msgs:
                    continue
                text = user_msgs[0]["content"]
            elif isinstance(p, str):
                text = p
            else:
                continue
            out.append({"prompt": text, "chosen": rec.get("chosen")})
        return out[:n]
    elif source == "harmfulqa":
        if partition != "construction":
            raise ValueError(
                "prompt_source='harmfulqa' requires prompt_partition='construction'; "
                f"got {partition!r}"
            )
        records = load_harmfulqa_partition("construction")
        return [
            {
                "prompt": r["prompt"], "chosen": None,
                "source_id": r["source_id"], "prompt_hash": r["prompt_hash"],
                "partition": r["partition"], "permuted_position": r["permuted_position"],
                "manifest_hash": r["manifest_hash"],
            }
            for r in records
        ]
    elif source == "mixed":
        raise ValueError(
            "prompt_source='mixed' is not a protocol construction path -- it always "
            "includes an ad hoc HarmfulQA component outside the frozen manifest. Use "
            "'hh_rlhf', or 'harmfulqa' with prompt_partition='construction', instead."
        )
    else:
        raise ValueError(f"Unknown prompt source: {source!r}")


# Activation extraction


def extract_activations(
    model_path: str,
    subfolder: str,
    tokenizer,
    prompts: List[Dict[str, str]],
    cfg: LayerProfileConfig,
    checkpoint_path: Optional[Path] = None,
    loading_policy: Optional[Dict[str, bool]] = None,
) -> torch.Tensor:
    """Extract the residual stream at the readout token, for every layer.

    Returns: tensor of shape (num_layers, n_prompts, hidden_dim)

    With a checkpoint_path, progress is flushed periodically and a re-run resumes
    from the last flush rather than starting over. Two models x 2000 prompts is long
    enough that losing it to a dropped connection matters.

    `loading_policy` (`local_files_only`/`trust_remote_code`) defaults to the
    historical, more permissive load when omitted, so a direct call from an existing
    caller/test is unaffected; `main()` always passes the resolved policy explicitly.
    """
    log = logging.getLogger(__name__)
    log.info(f"Loading: {model_path} subfolder={subfolder!r}")

    model = load_model(model_path, subfolder=subfolder, **(loading_policy or {}))
    device = next(model.parameters()).device

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    shape = (num_layers, len(prompts), hidden_dim)

    ckpt = TensorCheckpoint(checkpoint_path, shape, every=cfg.checkpoint_every) if checkpoint_path else None
    acts = ckpt.tensor if ckpt else torch.zeros(*shape, dtype=torch.float32)
    start = ckpt.done if ckpt else 0

    if start >= len(prompts):
        log.info(f"Checkpoint already complete ({start}/{len(prompts)}); skipping extraction")
    else:
        if start:
            log.info(f"Resuming at prompt {start}/{len(prompts)}")
        with LastTokenCapture(model) as cap:
            for i in tqdm(range(start, len(prompts)), desc="Extracting", unit="prompt",
                          initial=start, total=len(prompts)):
                text = build_input_text(tokenizer, prompts[i], cfg.token_position)
                inputs = tokenizer(
                    text, return_tensors="pt",
                    truncation=True, max_length=cfg.max_input_length,
                ).to(device)

                cap.clear()
                with torch.no_grad():
                    model(**inputs, use_cache=False)
                acts[:, i, :] = cap.stack()
                if ckpt:
                    ckpt.advance()

        if ckpt:
            ckpt.flush()

    # Free model from GPU before returning.
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return acts


# Statistics


def compute_layer_stats(act_it: torch.Tensor, act_dpo: torch.Tensor,
                        device: str = "cpu") -> pd.DataFrame:
    """Compute per-layer statistics. Returns DataFrame with one row per layer."""
    num_layers = act_it.shape[0]
    rows = []

    for layer in range(num_layers):
        h_it = act_it[layer].to(device)      # (n_prompts, hidden)
        h_dpo = act_dpo[layer].to(device)
        diff = h_dpo - h_it       # (n_prompts, hidden)

        # 1. Cosine similarity between IT and DPO per prompt, then mean.
        cos_per_prompt = torch.nn.functional.cosine_similarity(h_it, h_dpo, dim=-1)
        cos_mean = cos_per_prompt.mean().item()
        cos_std = cos_per_prompt.std().item()

        # 2. L2 norm of the difference per prompt, then mean.
        diff_norms = torch.norm(diff, dim=-1)
        diff_norm_mean = diff_norms.mean().item()
        diff_norm_std = diff_norms.std().item()

        # 3. Norms of the activations themselves (for context).
        it_norm = torch.norm(h_it, dim=-1).mean().item()
        dpo_norm = torch.norm(h_dpo, dim=-1).mean().item()

        # 4. SVD of the difference matrix.
        # diff is (n_prompts, hidden). For SVD we want directions in hidden space.
        # U: (n_prompts, k), S: (k,), Vh: (k, hidden)
        # Right singular vectors (Vh rows) are directions in hidden space.
        #
        # NOTE: this is the CENTRED decomposition, so it describes the scatter of the
        # displacements around their mean, not the mean displacement itself. It says
        # nothing about whether h_dpo ~= h_it + a*v holds -- centring removes exactly that
        # component. For the low-rank question use scripts/rank_analysis.py, which runs
        # the uncentred decomposition.
        diff_centered = diff - diff.mean(dim=0, keepdim=True)
        try:
            U, S, Vh = torch.linalg.svd(diff_centered, full_matrices=False)
            sv = S.float()
            # Explained variance ratios (S^2 / sum(S^2)).
            ev = (sv ** 2) / (sv ** 2).sum()
            pc0_ev = ev[0].item()
            pc1_ev = ev[1].item() if len(ev) > 1 else 0.0
            pc2_ev = ev[2].item() if len(ev) > 2 else 0.0
            top3_ev = pc0_ev + pc1_ev + pc2_ev

            # 5. Angle between mean direction and PC0 direction.
            mean_dir = diff.mean(dim=0)  # (hidden,)
            mean_dir_norm = mean_dir / (mean_dir.norm() + 1e-8)
            pc0_dir = Vh[0]  # (hidden,)
            # Both are unit vectors; cosine = dot product.
            cos_mean_pc0 = abs((mean_dir_norm * pc0_dir).sum().item())
        except Exception as e:
            log = logging.getLogger(__name__)
            log.warning(f"SVD failed at layer {layer}: {e}")
            pc0_ev = pc1_ev = pc2_ev = top3_ev = float("nan")
            cos_mean_pc0 = float("nan")

        rows.append({
            "layer": layer,
            "cos_it_dpo_mean": cos_mean,
            "cos_it_dpo_std": cos_std,
            "diff_norm_mean": diff_norm_mean,
            "diff_norm_std": diff_norm_std,
            "it_norm_mean": it_norm,
            "dpo_norm_mean": dpo_norm,
            "pc0_explained_variance": pc0_ev,
            "pc1_explained_variance": pc1_ev,
            "pc2_explained_variance": pc2_ev,
            "top3_explained_variance": top3_ev,
            "cos_mean_vs_pc0": cos_mean_pc0,
        })

    return pd.DataFrame(rows)


# Plots


def plot_curves(stats_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate the three diagnostic plots."""
    layers = stats_df["layer"].values

    # Cosine similarity per layer.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, stats_df["cos_it_dpo_mean"], marker="o", linewidth=1.5)
    ax.fill_between(
        layers,
        stats_df["cos_it_dpo_mean"] - stats_df["cos_it_dpo_std"],
        stats_df["cos_it_dpo_mean"] + stats_df["cos_it_dpo_std"],
        alpha=0.2,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Per-layer cosine similarity between IT and DPO hidden states")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(top=1.0)
    fig.tight_layout()
    fig.savefig(output_dir / "cosine_similarity.png", dpi=140)
    plt.close(fig)

    # Explained variance per layer.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, stats_df["pc0_explained_variance"], marker="o", label="PC0", linewidth=1.5)
    ax.plot(layers, stats_df["top3_explained_variance"], marker="s", label="Top 3 PCs",
            linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PC0 and top-3 PC variance of (h_DPO − h_IT) per layer")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(output_dir / "explained_variance.png", dpi=140)
    plt.close(fig)

    # Difference norm per layer.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, stats_df["diff_norm_mean"], marker="o", linewidth=1.5,
            label="||h_DPO − h_IT||")
    ax.fill_between(
        layers,
        stats_df["diff_norm_mean"] - stats_df["diff_norm_std"],
        stats_df["diff_norm_mean"] + stats_df["diff_norm_std"],
        alpha=0.2,
    )
    ax.plot(layers, stats_df["it_norm_mean"], linestyle="--", alpha=0.5, label="||h_IT||")
    ax.plot(layers, stats_df["dpo_norm_mean"], linestyle="--", alpha=0.5, label="||h_DPO||")
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 norm")
    ax.set_title("Activation norms and difference norms per layer")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "norm_curve.png", dpi=140)
    plt.close(fig)


# Summary


def summarize_findings(stats_df: pd.DataFrame, num_layers: int) -> str:
    """Produce a human-readable summary of the layer profile."""
    # Average cosine similarity in early (first 25%), middle, and late (last 25%) layers.
    n = num_layers
    early = stats_df.iloc[:n // 4]
    middle = stats_df.iloc[n // 4: 3 * n // 4]
    late = stats_df.iloc[3 * n // 4:]

    pc0_late = late["pc0_explained_variance"].mean()
    cos_late = late["cos_it_dpo_mean"].mean()
    cos_early = early["cos_it_dpo_mean"].mean()
    cos_drop = cos_early - cos_late
    mean_vs_pc0_late = late["cos_mean_vs_pc0"].mean()

    diff_norm_early = early["diff_norm_mean"].mean()
    diff_norm_late = late["diff_norm_mean"].mean()

    lines = [
        "=" * 60,
        "LAYER PROFILE SUMMARY",
        "=" * 60,
        "",
        f"Model has {num_layers} transformer layers.",
        "",
        "COSINE SIMILARITY (IT vs DPO hidden states):",
        f"  Early layers (0-{n//4-1}):  mean = {cos_early:.4f}",
        f"  Middle layers ({n//4}-{3*n//4-1}): mean = {middle['cos_it_dpo_mean'].mean():.4f}",
        f"  Late layers ({3*n//4}-{n-1}):  mean = {cos_late:.4f}",
        f"  Drop from early to late: {cos_drop:+.4f}",
        "",
        "DIFFERENCE NORM ||h_DPO - h_IT||:",
        f"  Early layers: {diff_norm_early:.2f}",
        f"  Late layers:  {diff_norm_late:.2f}",
        f"  Ratio (late/early): {diff_norm_late/max(diff_norm_early, 1e-8):.2f}x",
        "",
        "SCATTER AROUND THE MEAN DISPLACEMENT (centred PC0, late layers):",
        f"  Mean centred-PC0 explained variance in late layers: {pc0_late:.4f}",
        f"  Mean cos(mean_dir, centred PC0) in late layers: {mean_vs_pc0_late:.4f}",
        "",
        "INTERPRETATION:",
    ]

    # Heuristic interpretations.
    if cos_drop > 0.005:
        lines.append("  - Cosine similarity DROPS in late layers → shift is layer-localized.")
    else:
        lines.append("  - Cosine similarity is FLAT → activations shifted uniformly.")

    lines += [
        "",
        "  The two numbers above describe variation AROUND the mean displacement, because",
        "  the decomposition is centred. They are not a measure of whether the shift is",
        "  low-rank, and a low value here is fully compatible with a strong shared shift.",
        "  cos(mean_dir, centred PC0) is near zero by construction and carries no signal.",
        "",
        "  For the low-rank question — uncentred PC0, effective rank, and the fraction of",
        "  displacement a single direction captures — run:",
        "",
        "      python scripts/rank_analysis.py --run <label>=<this dir> --output-dir <out>",
        "      python scripts/vector_analysis.py --run <label>=<this dir> --output-dir <out>",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    main()
