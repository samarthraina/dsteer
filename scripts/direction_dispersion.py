"""How much of each direction do the activations already carry?

Ablation removes (h . v_hat) v_hat, so what it does to a prompt depends entirely on how
much that prompt's activation lies along v_hat. Addition does not care. That asymmetry is
a candidate mechanical explanation for an ablation result differing from an addition one,
and it costs nothing to measure: the activations are already on disk.

Three controls decide whether the answer means anything.

**Both directions have to be out of sample**, and confirming that is not free: the two
activation files hold different slices of one shuffled prompt list, so a row index means
something different in each. The checkpoint file runs from prompt 0; the refusal file was
built with `--hold-out 300` and its row 0 is prompt 300. Aligning them is what the guard
below does, by checking that the overlapping rows are the same activations rather than
assuming it from the file names.

**A direction estimated from a few hundred samples carries estimation noise.** Two disjoint
halves of the refusal set are compared to each other first, so the spread between any two
estimates of the direction is visible before anything is read into a difference.

**A harmful-vs-harmless direction is large on harmful prompts by construction.** Every
evaluation prompt is harmful, so a big uniform projection onto a direction fitted to
separate harmful from harmless is close to arithmetic. The harmless activations are
projected too. If they sit near zero, the projection is class separation rather than a
standing feature of the residual stream, and the ablation reading has to say so.

Projections are reported as a fraction of activation norm, because h . v_hat in raw units
says nothing without knowing how big h is.

    python scripts/direction_dispersion.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LAYERS = [27, 28, 29, 30, 31]


def mean_diff(blob: dict, layers: Sequence[int], lo: int, hi: int) -> Dict[int, torch.Tensor]:
    """Eq. 2 on a chosen slice of samples: mean(dpo) - mean(it) per layer."""
    it, dpo = blob["it"], blob["dpo"]
    n = min(it.shape[1], dpo.shape[1], hi) if hi else min(it.shape[1], dpo.shape[1])
    if lo >= n:
        raise ValueError(f"slice [{lo}:{hi}] leaves nothing of {it.shape[1]} samples")
    return {l: (dpo[l][lo:n].double().mean(0) - it[l][lo:n].double().mean(0)) for l in layers}


def cos(a: Dict[int, torch.Tensor], b: Dict[int, torch.Tensor], layers) -> float:
    vals = [(a[l] / a[l].norm() @ (b[l] / b[l].norm())).item() for l in layers]
    return sum(vals) / len(vals)


def profile(h: torch.Tensor, v: torch.Tensor) -> tuple:
    """Projection onto a unit direction, in units of the activation norm."""
    proj = h.double() @ (v / v.norm())
    scale = h.double().norm(dim=1).mean().item()
    return proj.mean().item(), proj.std().item(), scale


def main() -> int:
    p = argparse.ArgumentParser(description="Dispersion of activation projections onto each direction.")
    p.add_argument("--checkpoint-acts", default="outputs/layer_profile_harmfulqa/llama3-oh/activations.pt")
    p.add_argument("--refusal-acts", default="outputs/refusal_direction/llama3-oh/activations.pt",
                   help="Omit with 'none' where a pair has no refusal direction: the "
                        "checkpoint profile alone still says whether the difference sits "
                        "near-orthogonal to the activations, which is the half that "
                        "replicates.")
    p.add_argument("--n-eval", type=int, default=300,
                   help="Evaluation prompts, taken from the front -- the same ones the arms score.")
    p.add_argument("--hold-out", type=int, default=300,
                   help="Leading prompts the checkpoint vector drops, matching the arm.")
    p.add_argument("--refusal-offset", type=int, default=300,
                   help="What the refusal file's row 0 is in the shared prompt order: the "
                        "--hold-out it was built with, read from its run_meta.json. It is "
                        "already disjoint from the evaluation set, so nothing is dropped "
                        "here; the offset only says how to line the two files up.")
    args = p.parse_args()

    cp_blob = torch.load(args.checkpoint_acts, map_location="cpu")
    v_cp = mean_diff(cp_blob, LAYERS, args.hold_out, 0)
    have_rd = args.refusal_acts.lower() not in ("none", "")
    directions = [("checkpoint", v_cp)]

    if have_rd:
        rd_blob = torch.load(args.refusal_acts, map_location="cpu")
        n_rd = rd_blob["dpo"].shape[1]
        half = n_rd // 2
        v_rd = mean_diff(rd_blob, LAYERS, 0, 0)
        directions.append(("refusal", v_rd))

        print(f"refusal direction: {n_rd} samples, prompts "
              f"{args.refusal_offset}-{args.refusal_offset + n_rd - 1} of the shared order, "
              f"so disjoint from the first {args.n_eval}\n")
        print("how well resolved is the direction at this sample size?")
        print(f"  cos(first {half}, last {n_rd - half})   "
              f"{cos(mean_diff(rd_blob, LAYERS, 0, half), mean_diff(rd_blob, LAYERS, half, 0), LAYERS):+.4f}"
              f"   <- two disjoint halves; the spread between any two estimates")
        print(f"  cos(refusal, checkpoint)   {cos(v_rd, v_cp, LAYERS):+.4f}")

        # Both files hold the aligned checkpoint read the same way on one shuffled order,
        # so once the offset is applied the overlapping rows are the same tensor. Comparing
        # row 0 to row 0 instead shows two unrelated prompts at about 0.78, which is chance
        # for this stream and reads like a difference in extraction rather than in offset.
        l0, o = LAYERS[0], args.refusal_offset
        k = min(cp_blob["dpo"].shape[1] - o, n_rd)
        a = cp_blob["dpo"][l0][o:o + k].double()
        b = rd_blob["dpo"][l0][:k].double()
        drift = ((a - b).norm(dim=1) / a.norm(dim=1)).mean().item()
        print(f"\nrows line up at offset {o}? layer {l0}, {k} overlapping rows: "
              f"mean relative difference {drift:.2e}")
        if drift > 1e-3:
            raise SystemExit(
                "  The two files do not line up at this offset, so a projection across them\n"
                "  compares different prompts. Read --refusal-offset off the refusal run's\n"
                "  run_meta.json (its hold_out) rather than assuming the flag's default.")

    print(f"\naligned checkpoint on the {args.n_eval} evaluation prompts, "
          f"projection as a share of |h|\n")
    print(f"{'layer':<7}{'direction':<22}{'h.v/|h|':>10}{'sd/|h|':>9}{'|h|':>9}{'|v|':>9}")
    agg: Dict[str, list] = {}
    for layer in LAYERS:
        h = cp_blob["dpo"][layer][: args.n_eval]
        for name, vecs in directions:
            m, sd, scale = profile(h, vecs[layer])
            agg.setdefault(name, []).append((m / scale, sd / scale))
            print(f"{layer:<7}{name:<22}{m / scale:>10.3f}{sd / scale:>9.3f}"
                  f"{scale:>9.2f}{vecs[layer].norm().item():>9.3f}")

    print(f"\nmean over layers {LAYERS[0]}-{LAYERS[-1]}")
    for name, rows in agg.items():
        m = sum(r[0] for r in rows) / len(rows)
        sd = sum(r[1] for r in rows) / len(rows)
        print(f"  {name:<22} {m:>+7.3f} +- {sd:.3f}")

    if not have_rd:
        return 0

    # Every evaluation prompt is harmful, and the refusal direction was fitted to separate
    # harmful from harmless. If the harmless side projects near zero, a large uniform
    # projection on harmful prompts is that separation and not a property of the stream.
    print(f"\nthe same direction on the harmless prompts it was built against")
    print(f"{'layer':<7}{'set':<22}{'h.v/|h|':>10}{'sd/|h|':>9}")
    for layer in LAYERS:
        for name, key in (("harmful (built from)", "dpo"), ("harmless (built from)", "it")):
            m, sd, scale = profile(rd_blob[key][layer], v_rd[layer])
            print(f"{layer:<7}{name:<22}{m / scale:>10.3f}{sd / scale:>9.3f}")
    print("\n  Harmless near zero means the direction separates the two classes, which is")
    print("  what it was built to do -- so 'the activations already carry it' is true of")
    print("  harmful prompts specifically, not of the model in general.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
