#!/usr/bin/env python3
"""D-STEER directions and their actual perturbation norms, from the recovered tensors.

CPU only. Uses the vendored primary `build_vectors` unchanged -- the same call gen_final.py
makes (method=mean, normalise=relative, layers 27-31, device=cpu) -- so the vectors here are
the submitted construction, not a reimplementation.

Writes the five-layer vectors per pair (~80 KB) so the GPU box need not receive the 2.9 GB
tensors. The box does not rebuild them: the end-to-end check is the S1 parity run, which
regenerates the steered D-STEER arms on a panel prefix and compares them with the recovered
originals (scripts/parity_check.py) -- a wrong vector cannot pass that.

Run:  .venv/bin/python -I scripts/compute_dsteer_geometry.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor" / "primary_snapshot"))
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from steering.steer import build_vectors  # noqa: E402

LAYERS = [27, 28, 29, 30, 31]
LAMBDA = {"A": {"installation": 0.05, "reversal": -0.05},
          "B": {"installation": 0.135, "reversal": -0.055}}
TENSOR_SHA = {"A": "5c2feb3efd84599a8e7475931f7448545107c30246a60ff8022012f89cd7ffa1",
              "B": "c02721c604964d4261b85852654276e49496a6fede78263d9330d6a5307adfc5"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> int:
    outdir = ROOT / "outputs" / "directions"
    outdir.mkdir(parents=True, exist_ok=True)
    report = {"layers": LAYERS, "construction": "build_vectors(method=mean, normalise=relative, device=cpu)",
              "torch": torch.__version__, "pairs": {}}
    rows = []
    for pair in ("A", "B"):
        t = ROOT / "inputs" / "recovered" / "task025" / f"pair_{pair}" / "activations.pt"
        got = sha256_file(t)
        if got != TENSOR_SHA[pair]:
            print(f"HARD STOP: pair {pair} tensor hash {got} != {TENSOR_SHA[pair]}")
            return 2
        blob = torch.load(t, map_location="cpu", weights_only=True)
        it, dpo = blob["it"], blob["dpo"]
        shape = {"it": list(it.shape), "dpo": list(dpo.shape), "dtype": str(it.dtype),
                 "keys": sorted(blob.keys())}

        vecs = build_vectors(t, method="mean", layers=LAYERS, normalise="relative", device="cpu")
        raw = {}
        for L in LAYERS:
            d = (dpo[L] - it[L]).double()
            raw[L] = d.mean(dim=0)
            it_norm = it[L].double().norm(dim=1).mean().item()
            dpo_norm = dpo[L].double().norm(dim=1).mean().item()
            # prompt-level directional coherence: share of prompts whose displacement agrees
            # in sign with the mean direction
            unit = raw[L] / raw[L].norm()
            coh = (d @ unit > 0).double().mean().item()
            for direction, lam in LAMBDA[pair].items():
                rows.append({
                    "pair": pair, "layer": L, "direction": direction, "lambda": lam,
                    "raw_mean_displacement_norm": round(raw[L].norm().item(), 6),
                    "mean_it_activation_norm": round(it_norm, 6),
                    "mean_dpo_activation_norm": round(dpo_norm, 6),
                    "relative_vector_norm": round(vecs[L].norm().item(), 6),
                    "actual_perturbation_norm": round(abs(lam) * vecs[L].norm().item(), 6),
                    "perturbation_over_it_activation": round(abs(lam) * vecs[L].norm().item() / it_norm, 6),
                    "prompt_sign_agreement_with_mean": round(coh, 6),
                })
            # relative vectors must point along the raw mean displacement
            cos = torch.nn.functional.cosine_similarity(vecs[L].double(), raw[L], dim=0).item()
            if cos < 1 - 1e-6:
                print(f"HARD STOP: pair {pair} layer {L} relative/raw cosine {cos}")
                return 2

        vpath = outdir / f"dsteer_vectors_pair_{pair}.pt"
        torch.save({L: vecs[L] for L in LAYERS}, vpath)
        # raw mean displacement at every layer, for the all-layer geometry against refusal axes
        allraw = (dpo.double() - it.double()).mean(dim=1)
        apath = outdir / f"dsteer_raw_meandiff_all_layers_pair_{pair}.pt"
        torch.save(allraw, apath)
        report["pairs"][pair] = {
            "tensor_sha256": got, "tensor_shape": shape,
            "n_construction_prompts": int(it.shape[1]),
            "vectors_path": str(vpath.relative_to(ROOT)), "vectors_sha256": sha256_file(vpath),
            "relative_vector_norms": {L: round(vecs[L].norm().item(), 6) for L in LAYERS},
        }
        del blob, it, dpo

    import csv
    with (outdir / "dsteer_geometry.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (outdir / "dsteer_geometry.json").write_text(json.dumps(report, indent=2) + "\n")
    for p, r in report["pairs"].items():
        print(f"pair {p}: tensor {r['tensor_shape']['it']} {r['tensor_shape']['dtype']}, "
              f"n={r['n_construction_prompts']}, |v_L| {r['relative_vector_norms']}")
    for r in rows:
        if r["layer"] in (27, 31):
            print(f"  {r['pair']} L{r['layer']} {r['direction']:12s} λ={r['lambda']:+.3f} "
                  f"perturbation {r['actual_perturbation_norm']:8.4f} "
                  f"(= {100*r['perturbation_over_it_activation']:.2f}% of mean IT norm), "
                  f"raw disp {r['raw_mean_displacement_norm']:.4f}, coherence {r['prompt_sign_agreement_with_mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
