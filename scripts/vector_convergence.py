"""How many samples does the mean displacement actually need?

The v1 runs used 10,000; these use 2,000; the paper reports no number at all. None of
that says whether the estimate had converged. The mean of a set of vectors that mostly
point the same way converges fast, so this measures it directly instead: draw subsets,
build the vector, and take its angle to the full-sample vector.

Reported per layer as the median over draws, with the 10th percentile, since the question
is not how good a typical subset is but how bad an unlucky one gets.

    python scripts/vector_convergence.py --run tulu3=outputs/layer_profile_response_token/tulu3 \
        --output-dir outputs/vector_convergence
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch

from steering.utils import resolve_device, setup_logging


def convergence(D: torch.Tensor, sizes: List[int], draws: int, seed: int) -> List[Dict]:
    """Cosine between a subset mean and the full-sample mean, over random draws."""
    full = D.mean(dim=0)
    full = full / full.norm()
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = D.shape[0]

    rows = []
    for size in sizes:
        if size > n:
            continue
        cosines = []
        for _ in range(draws):
            idx = torch.randperm(n, generator=g)[:size]
            v = D[idx].mean(dim=0)
            cosines.append((v / v.norm() @ full).item())
        c = torch.tensor(cosines)
        rows.append({
            "n": size,
            "cos_median": c.median().item(),
            "cos_p10": c.quantile(0.10).item(),
            "cos_min": c.min().item(),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Sample size needed for the steering vector.")
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers-last-k", type=int, default=5)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out / "vector_convergence.log")
    device = resolve_device(args.device)
    sizes = [25, 50, 100, 250, 500, 1000, 1500]

    summary = {}
    for spec in args.run:
        label, directory = spec.split("=", 1)
        path = Path(directory) / "activations.pt"
        if not path.exists():
            log.warning(f"skip {label}: no activations at {path}")
            continue
        blob = torch.load(path, map_location="cpu")
        it, dpo = blob["it"], blob["dpo"]
        n_layers = it.shape[0]
        layers = list(range(n_layers - args.layers_last_k, n_layers))

        rows = []
        for layer in layers:
            D = (dpo[layer] - it[layer]).to(device=device, dtype=torch.float64)
            for r in convergence(D, sizes, args.draws, args.seed):
                rows.append({"layer": layer, **r})
        df = pd.DataFrame(rows)
        df.to_csv(out / f"convergence_{label}.csv", index=False)

        # Worst layer at each size is what the answer has to survive.
        worst = df.groupby("n")[["cos_median", "cos_p10"]].min()
        summary[label] = {
            "n_available": int(it.shape[1]),
            "worst_layer_by_size": {int(k): {c: round(v, 4) for c, v in row.items()}
                                    for k, row in worst.iterrows()},
        }
        log.info(f"\n[{label}] worst layer, cosine to the full-sample vector\n{worst.to_string()}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
