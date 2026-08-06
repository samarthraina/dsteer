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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

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
from steering.data import load_hh_rlhf_test, load_harmfulqa
from steering.models import load_model, load_tokenizer
from steering.utils import load_yaml, resolve_device, set_all_seeds, setup_logging


# Config


@dataclass
class LayerProfileConfig:
    """Settings for layer profile analysis."""

    # Source of prompts to extract activations from.
    # Options: "hh_rlhf", "harmfulqa", "mixed"
    prompt_source: str = "hh_rlhf"

    # Number of prompts to use.
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
    parser.add_argument("--model-config", required=True)
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

    set_all_seeds(args.seed)

    model_cfg = ModelConfig.from_yaml(args.model_config)
    eval_cfg = LayerProfileConfig.from_yaml(args.eval_config)

    output_root = Path(eval_cfg.output_dir) / model_cfg.name
    output_root.mkdir(parents=True, exist_ok=True)

    log = setup_logging(output_root / "layer_profile.log")
    log.info(f"Model pair: {model_cfg.name}")
    log.info(f"Output: {output_root}")
    log.info(f"Architecture: {model_cfg.architecture}, layers: {model_cfg.num_layers}")

    write_run_metadata(
        output_root,
        config={"model": asdict(model_cfg), "eval": asdict(eval_cfg), "seed": args.seed},
    )

    if args.no_resume:
        for stale in output_root.glob("*.partial.pt"):
            stale.unlink()
            log.info(f"Removed checkpoint {stale}")

    # Load prompts
    log.info(f"Loading {eval_cfg.n_prompts} prompts from {eval_cfg.prompt_source}")
    log.info(f"Readout position: {eval_cfg.token_position}")
    prompts = load_prompts(eval_cfg.prompt_source, eval_cfg.n_prompts, args.seed)
    if eval_cfg.token_position == "response_last":
        prompts = [p for p in prompts if p.get("chosen")]
        if not prompts:
            raise ValueError(
                "response_last needs reference responses; prompt_source must be hh_rlhf"
            )
    log.info(f"Loaded {len(prompts)} prompts")

    # Load tokenizer
    tokenizer_path = model_cfg.tokenizer_id or model_cfg.it_model
    tokenizer_subfolder = model_cfg.tokenizer_subfolder or (model_cfg.it_subfolder or "")
    tokenizer = load_tokenizer(tokenizer_path, subfolder=tokenizer_subfolder)

    # Extract activations from both models on the same prompts.
    with GpuMonitor(output_root, hourly_rate=args.hourly_rate) as gpu:
        log.info("\n" + "=" * 60)
        log.info("Extracting activations from IT model")
        log.info("=" * 60)
        activations_it = extract_activations(
            model_cfg.it_model, model_cfg.it_subfolder or "",
            tokenizer, prompts, eval_cfg,
            checkpoint_path=output_root / "it.partial.pt",
        )
        log.info(f"IT activations shape: {activations_it.shape}  (layers, prompts, hidden)")

        log.info("\n" + "=" * 60)
        log.info("Extracting activations from DPO model")
        log.info("=" * 60)
        activations_dpo = extract_activations(
            model_cfg.dpo_model, model_cfg.dpo_subfolder or "",
            tokenizer, prompts, eval_cfg,
            checkpoint_path=output_root / "dpo.partial.pt",
        )
        log.info(f"DPO activations shape: {activations_dpo.shape}")

    log.info(f"GPU usage: {json.dumps(gpu.summary())}")

    # Save raw activations for later reuse (steering vector building).
    torch.save({"it": activations_it, "dpo": activations_dpo},
               output_root / "activations.pt")
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
    (output_root / "summary.txt").write_text(summary)
    log.info("\n" + summary)

    if args.sync:
        sync_to_hub(output_root, experiment="layer_profile", model=model_cfg.name)


# Prompt loading


def load_prompts(source: str, n: int, seed: int) -> List[Dict[str, str]]:
    """Load records with a "prompt" string and, where available, a "chosen" response.

    For HH-RLHF, we extract just the first user turn (single-turn for clean activations)
    and carry its reference response through for response_last readout.
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
        return [{"prompt": r["prompt"], "chosen": None} for r in load_harmfulqa(n=n, seed=seed)]
    elif source == "mixed":
        n_each = n // 2
        a = load_prompts("hh_rlhf", n_each, seed)
        b = load_prompts("harmfulqa", n - n_each, seed)
        return a + b
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
) -> torch.Tensor:
    """Extract the residual stream at the readout token, for every layer.

    Returns: tensor of shape (num_layers, n_prompts, hidden_dim)

    With a checkpoint_path, progress is flushed periodically and a re-run resumes
    from the last flush rather than starting over. Two models x 2000 prompts is long
    enough that losing it to a dropped connection matters.
    """
    log = logging.getLogger(__name__)
    log.info(f"Loading: {model_path} subfolder={subfolder!r}")

    model = load_model(model_path, subfolder=subfolder)
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
