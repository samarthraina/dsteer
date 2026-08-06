"""Which steering vector, and what coefficient?

Two candidate directions have been used in this project:

  mean   v = (1/N) sum_i dh_i                     -- what the paper's Eq. 2 defines
  pc3    v = Vt[3] of the *centred* dh matrix,    -- what the v1 notebook actually ran
         rescaled to ||mean||                        (steering_vector_svd_component = 3)

Because centring removes the mean, pc3 is close to orthogonal to the mean displacement,
so these are different vectors and the choice is not cosmetic.

For each candidate this reports, per layer:

  alpha_mean/sd     per-sample coefficient alpha_i = <dh_i, v> / ||v||^2. The spread says
                    how input-dependent the displacement is; a constant lambda can only be
                    right for all inputs if this is tight.
  lambda_star       the least-squares coefficient over the whole set, i.e. the geometric
                    optimum. For v = mean this is 1 by construction, which is the point:
                    it is a sanity check, not a result.
  captured          1 - ||dh - alpha* v||^2 / ||dh||^2. Fraction of squared displacement the
                    single direction explains. Low values mean steering along v cannot
                    reproduce the displacement however lambda is chosen.
  frac_negative     share of samples wanting a coefficient of the opposite sign.
  cos_mean_pc3      angle between the two candidates.

Usage:
    python scripts/vector_analysis.py \
        --run tulu3=outputs/layer_profile/tulu3 \
        --output-dir outputs/vector_analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steering.utils import resolve_device, setup_logging


def candidate_vectors(D: torch.Tensor, pc_index: int = 3) -> Dict[str, torch.Tensor]:
    """Build the two candidate steering directions from a displacement matrix."""
    v_mean = D.mean(dim=0)
    centred = D - v_mean
    # Vt rows are directions in hidden space. v1 rescaled the component to ||mean||.
    _, _, Vt = torch.linalg.svd(centred, full_matrices=False)
    v_pc = Vt[pc_index] * v_mean.norm()
    return {"mean": v_mean, f"pc{pc_index}": v_pc}


def coefficient_stats(D: torch.Tensor, v: torch.Tensor) -> Dict[str, float]:
    """Per-sample projection coefficients onto v, and how much of D that direction explains."""
    vv = (v @ v).item()
    if vv == 0:
        return {k: float("nan") for k in
                ("alpha_mean", "alpha_sd", "lambda_star", "captured", "frac_negative")}

    alpha = (D @ v) / vv
    lambda_star = alpha.mean().item()          # least-squares optimum over the set
    residual = D - lambda_star * v
    captured = 1.0 - (residual ** 2).sum().item() / (D ** 2).sum().item()

    return {
        "alpha_mean": alpha.mean().item(),
        "alpha_sd": alpha.std().item(),
        "lambda_star": lambda_star,
        "captured": captured,
        "frac_negative": (alpha < 0).double().mean().item(),
    }


def profile(path: Path, log, device: str = "cpu") -> pd.DataFrame:
    blob = torch.load(path, map_location="cpu")
    it, dpo = blob["it"], blob["dpo"]
    log.info(f"{path}: layers={it.shape[0]} prompts={it.shape[1]} hidden={it.shape[2]} device={device}")

    rows = []
    for layer in range(it.shape[0]):
        D = (dpo[layer] - it[layer]).to(device=device, dtype=torch.float64)
        vecs = candidate_vectors(D)
        row = {"layer": layer}
        for name, v in vecs.items():
            for k, val in coefficient_stats(D, v).items():
                row[f"{name}_{k}"] = val
        a, b = vecs["mean"], vecs["pc3"]
        row["cos_mean_pc3"] = abs(
            ((a / a.norm()) @ (b / b.norm())).item()
        )
        rows.append(row)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return pd.DataFrame(rows).set_index("layer")


def plot(frames: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, df in frames.items():
        axes[0].plot(df.index, df.mean_captured, marker="o", label=f"{label}: mean")
        axes[0].plot(df.index, df.pc3_captured, marker="s", linestyle="--", label=f"{label}: pc3")
        axes[1].plot(df.index, df.mean_alpha_sd, marker="o", label=f"{label}: sd(alpha)")
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Fraction of displacement captured")
    axes[0].grid(True, alpha=0.3); axes[0].legend(); axes[0].set_ylim(0, 1)
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("sd of per-sample coefficient")
    axes[1].grid(True, alpha=0.3); axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "vector_comparison.png", dpi=140)
    plt.close(fig)


def summarize(frames: Dict[str, pd.DataFrame], late_frac: float = 0.25) -> str:
    lines = ["=" * 64, "STEERING VECTOR COMPARISON", "=" * 64, ""]
    for label, df in frames.items():
        n_late = max(1, int(len(df) * late_frac))
        late = df.iloc[-n_late:]
        lines += [
            f"[{label}]  late layers = last {n_late}",
            f"  displacement captured by mean : {late.mean_captured.mean():.3f}",
            f"  displacement captured by pc3  : {late.pc3_captured.mean():.3f}",
            f"  cos(mean, pc3)                : {late.cos_mean_pc3.mean():.3f}",
            f"  lambda* for mean              : {late.mean_lambda_star.mean():.3f}",
            f"  lambda* for pc3               : {late.pc3_lambda_star.mean():.3f}",
            f"  sd of alpha (mean vector)     : {late.mean_alpha_sd.mean():.3f}",
            f"  samples wanting opposite sign : {late.mean_frac_negative.mean():.1%}",
            "",
        ]
    lines += [
        "Reading: 'captured' bounds what any constant lambda can achieve along that",
        "direction. A large sd of alpha means no single lambda suits all inputs.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare candidate steering vectors and coefficients.")
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--device", default="auto",
        help="auto | cuda | cpu. The per-layer decompositions dominate runtime on CPU.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(output_dir / "vector_analysis.log")
    device = resolve_device(args.device)

    frames = {}
    for spec in args.run:
        if "=" not in spec:
            parser.error(f"--run expects LABEL=DIR, got {spec!r}")
        label, directory = spec.split("=", 1)
        path = Path(directory) / "activations.pt"
        if not path.exists():
            parser.error(f"no activations.pt in {directory}")
        df = profile(path, log, device=device)
        df.to_csv(output_dir / f"vectors_{label}.csv")
        frames[label] = df

    plot(frames, output_dir)
    summary = summarize(frames)
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    log.info("\n" + summary)


if __name__ == "__main__":
    main()
