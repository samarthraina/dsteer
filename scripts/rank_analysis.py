"""Low-rank structure of an activation shift, measured without centring.

layer_profile.py runs SVD on the difference matrix *after* subtracting its column
mean. That removes the shared component -- exactly the part the h_dpo ~= h_it + a*v
picture is about -- so its pc0_explained_variance describes the scatter around the
shift, not the shift. This script measures the shift itself.

For the difference matrix D (n_prompts x hidden) at each layer, with v_bar its mean row:

  mean_frac       n*||v_bar||^2 / ||D||_F^2, the share of total squared displacement
                  carried by the shared direction. This is the rank-1 test.
  pc0 / top3      explained variance of the leading directions, NOT centred.
  eff_rank        participation ratio (sum s^2)^2 / sum s^4. 1.0 means a pure rank-1
                  shift; higher means the displacement is spread over more directions.
  cos_to_mean     per-sample cos(d_i, v_bar), mean and sd, plus the fraction above 0.5.
                  Answers whether individual displacements share the mean's direction,
                  rather than the mean merely existing.
  pc0_centred     reproduces layer_profile.py's number, for comparison.

Two runs can be compared in one pass, which is how the base->IT shift is used as a
control for the IT->DPO shift: if both look alike, "localised and low-rank" does not
distinguish preference optimisation from ordinary fine-tuning.

Usage:
    python scripts/rank_analysis.py \
        --run it_dpo=outputs/layer_profile/tulu3 \
        --run base_it=outputs/layer_profile/tulu3_basesft \
        --output-dir outputs/rank_analysis/tulu3

Each run directory must contain the activations.pt written by layer_profile.py.
No judge server needed. Uses the GPU for the decompositions when one is present
(--device cpu to force otherwise): on CPU these dominate the runtime of the whole
geometry pipeline, taking longer than the extraction that produced the tensors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steering.utils import resolve_device, setup_logging


# Statistics


def layer_stats(diff: torch.Tensor, ref: torch.Tensor, device: str = "cpu") -> Dict[str, float]:
    """Rank statistics for one layer's displacement matrix.

    diff: (n_prompts, hidden) displacement. ref: (n_prompts, hidden) baseline
    activations, used only to normalise the displacement magnitude.
    """
    # float64: the squared-singular-value ratios below lose too much precision in bf16/fp32
    # when the spectrum spans several orders of magnitude. Two decompositions per layer
    # is minutes on CPU and seconds on a GPU, and this runs once per model pair.
    d = diff.to(device=device, dtype=torch.float64)
    ref = ref.to(device=device, dtype=torch.float64)
    n = d.shape[0]

    v_bar = d.mean(dim=0)
    energy = (d ** 2).sum().item()

    sv = torch.linalg.svdvals(d)
    ev = (sv ** 2) / (sv ** 2).sum()
    eff_rank = ((sv ** 2).sum() ** 2 / (sv ** 4).sum()).item()

    sv_c = torch.linalg.svdvals(d - v_bar)
    pc0_centred = ((sv_c ** 2) / (sv_c ** 2).sum())[0].item()

    unit = v_bar / (v_bar.norm() + 1e-12)
    cos_i = (d / (d.norm(dim=1, keepdim=True) + 1e-12)) @ unit

    return {
        "rel_norm": (d.norm(dim=1).mean() / ref.norm(dim=1).mean()).item(),
        "mean_frac": (n * (v_bar @ v_bar).item()) / energy if energy > 0 else float("nan"),
        "pc0": ev[0].item(),
        "top3": ev[:3].sum().item(),
        "eff_rank": eff_rank,
        "cos_to_mean_mean": cos_i.mean().item(),
        "cos_to_mean_sd": cos_i.std().item(),
        "frac_aligned": (cos_i > 0.5).double().mean().item(),
        "pc0_centred": pc0_centred,
    }


def profile_run(path: Path, log, device: str = "cpu") -> pd.DataFrame:
    """Per-layer statistics for one activations.pt (keys "it" and "dpo")."""
    blob = torch.load(path, map_location="cpu")
    a, b = blob["it"], blob["dpo"]
    log.info(f"{path}: layers={a.shape[0]} prompts={a.shape[1]} hidden={a.shape[2]} device={device}")

    rows = []
    for layer in range(a.shape[0]):
        row = layer_stats(b[layer] - a[layer], a[layer], device=device)
        row["layer"] = layer
        rows.append(row)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return pd.DataFrame(rows).set_index("layer")


# Plots


def plot_comparison(frames: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Effective rank and displacement magnitude against depth, one line per run."""
    for column, ylabel, fname in [
        ("eff_rank", "Participation ratio", "effective_rank.png"),
        ("rel_norm", "||delta h|| / ||h||", "relative_norm.png"),
        ("cos_to_mean_mean", "cos(d_i, mean)", "directional_alignment.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, df in frames.items():
            ax.plot(df.index, df[column], marker="o", linewidth=1.5, label=label)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=140)
        plt.close(fig)


# Summary


def summarize(frames: Dict[str, pd.DataFrame], late_frac: float = 0.25) -> str:
    lines = ["=" * 60, "RANK ANALYSIS (uncentred)", "=" * 60, ""]
    for label, df in frames.items():
        n_late = max(1, int(len(df) * late_frac))
        early, late = df.iloc[0], df.iloc[-n_late:]
        lines += [
            f"[{label}]  {len(df)} layers, late = last {n_late}",
            f"  effective rank:  layer 0 = {early.eff_rank:.2f}"
            f"   late mean = {late.eff_rank.mean():.2f}",
            f"  PC0 (uncentred): layer 0 = {early.pc0:.3f}"
            f"   late mean = {late.pc0.mean():.3f}",
            f"  shared-direction share of displacement, late: {late.mean_frac.mean():.3f}",
            f"  cos(d_i, mean), late: {late.cos_to_mean_mean.mean():.3f}"
            f"  ({late.frac_aligned.mean():.1%} of samples above 0.5)",
            "",
        ]

    if len(frames) > 1:
        labels = list(frames)
        lines += ["COMPARISON (late layers):", ""]
        for label in labels:
            df = frames[label]
            n_late = max(1, int(len(df) * late_frac))
            late = df.iloc[-n_late:]
            lines.append(
                f"  {label:>12}: eff_rank {late.eff_rank.mean():6.2f}"
                f"   rel_norm {late.rel_norm.mean():.3f}"
                f"   mean_frac {late.mean_frac.mean():.3f}"
            )
        lines += [
            "",
            "  If the control shift matches the DPO shift on these, then 'localised and",
            "  low-rank' does not separate preference optimisation from fine-tuning.",
        ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Measure low-rank structure of an activation shift.")
    parser.add_argument(
        "--run", action="append", required=True, metavar="LABEL=DIR",
        help="Run directory containing activations.pt. Repeatable to compare runs.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--device", default="auto",
        help="auto | cuda | cpu. The per-layer decompositions dominate runtime on CPU.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(output_dir / "rank_analysis.log")
    device = resolve_device(args.device)

    runs: List[tuple] = []
    for spec in args.run:
        if "=" not in spec:
            parser.error(f"--run expects LABEL=DIR, got {spec!r}")
        label, directory = spec.split("=", 1)
        runs.append((label, Path(directory)))

    frames = {}
    for label, directory in runs:
        path = directory / "activations.pt"
        if not path.exists():
            parser.error(f"no activations.pt in {directory}")
        df = profile_run(path, log, device=device)
        df.to_csv(output_dir / f"rank_{label}.csv")
        frames[label] = df

    plot_comparison(frames, output_dir)
    summary = summarize(frames)
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    log.info("\n" + summary)


if __name__ == "__main__":
    main()
