"""Inference-time activation steering.

Adds a per-layer vector to the residual stream during the forward pass:

    h_l  <-  h_l + lambda * v_l

which is the intervention the paper describes, applied without touching weights.

Three choices here are not cosmetic, so they are explicit rather than implied.

**Which vector.** `mean` is the mean displacement, i.e. the paper's Eq. 2. `pcN` is
the N-th principal direction of the *centred* displacement matrix rescaled to the
mean's norm -- what the v1 notebook actually ran, with N=3. Centring removes the mean,
so pc3 is close to orthogonal to it: these are different directions and the choice
changes what the experiment tests.

**What lambda means.** Raw lambda is in the units of ||v||, which differ per layer and
per model, so "optimal lambda = 0.4" does not transfer between models. With
`normalise="relative"` the vector is rescaled to the mean activation norm at its layer,
making lambda a dimensionless fraction of typical activation magnitude and comparable
across models. Use raw to reproduce the paper, relative to compare pairs.

**Where it is applied.** Default is every token position, prompt and generation alike,
which matches the activation-steering literature. `positions="new"` restricts it to
generated tokens, leaving the prompt encoding untouched.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import torch

from .activations import decoder_layers

log = logging.getLogger(__name__)


# Vector construction


def build_vectors(
    activations_path: Union[str, Path],
    method: str = "mean",
    layers: Optional[Sequence[int]] = None,
    normalise: Optional[str] = None,
    device: str = "cpu",
    skip_first: int = 0,
) -> Dict[int, torch.Tensor]:
    """Build per-layer steering vectors from a layer_profile activations.pt.

    method:     "mean" (Eq. 2) or "pcN" for the N-th centred principal direction,
                rescaled to the mean's norm.
    normalise:  None keeps the natural scale; "unit" gives unit norm; "relative"
                rescales to the mean activation norm at that layer, so lambda reads as
                a fraction of typical activation magnitude.
    skip_first: drop this many leading samples before averaging.

    `skip_first` exists because the prompt loaders shuffle on a fixed seed and take the
    first n, so a profile of 1900 prompts and a sweep of 300 share their first 300 -- the
    evaluation set sits inside the set the vector was estimated from. Where the two are
    drawn from the same corpus, skipping the sweep's count makes them disjoint and the
    measurement held out. It does nothing when the vector comes from a different corpus.
    """
    blob = torch.load(Path(activations_path), map_location="cpu")
    it, dpo = blob["it"], blob["dpo"]
    if skip_first:
        if skip_first >= it.shape[1]:
            raise ValueError(
                f"skip_first={skip_first} leaves nothing of {it.shape[1]} samples")
        it, dpo = it[:, skip_first:], dpo[:, skip_first:]
        log.info(f"Held out the first {skip_first} samples; "
                 f"{it.shape[1]} remain for the vector")
    n_layers = it.shape[0]
    wanted = list(layers) if layers is not None else list(range(n_layers))

    out: Dict[int, torch.Tensor] = {}
    for layer in wanted:
        if not 0 <= layer < n_layers:
            raise ValueError(f"layer {layer} outside 0..{n_layers - 1}")
        d = (dpo[layer] - it[layer]).to(device=device, dtype=torch.float64)
        v_mean = d.mean(dim=0)

        if method == "mean":
            v = v_mean
        elif method.startswith("pc"):
            k = int(method[2:])
            # Centred, matching the v1 construction: the mean is projected out first.
            _, _, Vt = torch.linalg.svd(d - v_mean, full_matrices=False)
            if k >= Vt.shape[0]:
                raise ValueError(f"{method}: only {Vt.shape[0]} components available")
            v = Vt[k] * v_mean.norm()
        else:
            raise ValueError(f"unknown method {method!r}; use 'mean' or 'pcN'")

        if normalise == "unit":
            v = v / (v.norm() + 1e-12)
        elif normalise == "relative":
            v = v / (v.norm() + 1e-12) * it[layer].to(device).double().norm(dim=1).mean()
        elif normalise is not None:
            raise ValueError(f"unknown normalise {normalise!r}")

        out[layer] = v.float().cpu()

    log.info(f"Built {len(out)} vectors from {activations_path} "
             f"(method={method}, normalise={normalise})")
    return out


def random_vectors_like(
    vectors: Dict[int, torch.Tensor], seed: int = 0
) -> Dict[int, torch.Tensor]:
    """Norm-matched random directions, for the control.

    The point of the control: if a random vector of the same magnitude moves behaviour
    as much as the DPO direction does, then the sweep is measuring perturbation size,
    not alignment content.
    """
    g = torch.Generator().manual_seed(seed)
    out = {}
    for layer, v in vectors.items():
        r = torch.randn(v.shape, generator=g, dtype=torch.float32)
        out[layer] = r / r.norm() * v.norm()
    return out


# The intervention


class ActivationSteering:
    """Modify the residual stream at chosen layers, for the duration of a block.

    Use:
        with ActivationSteering(model, vectors, coefficient=0.4):
            model.generate(...)

    coefficient may be a float (same for every steered layer) or {layer: float}, which
    is what a per-layer schedule needs.

    Two interventions, and the difference matters when comparing against published work.
    `mode="add"` puts lambda * v into the stream, which is what a checkpoint-difference
    vector is built for. `mode="ablate"` removes the component along v instead --
    h <- h - (h . v_hat) v_hat -- so the model cannot represent that direction at all.
    Ablation is what the refusal-direction line uses, and it is not a special case of
    addition: it is input-dependent, subtracting exactly as much as each activation
    happens to carry rather than a fixed amount. `coefficient` scales how much of the
    component is taken out, so 1.0 removes it entirely.
    """

    def __init__(
        self,
        model,
        vectors: Dict[int, torch.Tensor],
        coefficient: Union[float, Dict[int, float]] = 1.0,
        positions: str = "all",
        preserve_norm: bool = False,
        mode: str = "add",
    ):
        if positions not in ("all", "new"):
            raise ValueError(f"positions must be 'all' or 'new', got {positions!r}")
        if mode not in ("add", "ablate"):
            raise ValueError(f"mode must be 'add' or 'ablate', got {mode!r}")
        self.mode = mode
        self.model = model
        self.vectors = vectors
        self.positions = positions
        self.preserve_norm = preserve_norm
        self.coefficients = (
            {l: float(coefficient) for l in vectors}
            if isinstance(coefficient, (int, float)) else dict(coefficient)
        )
        missing = set(self.coefficients) - set(vectors)
        if missing:
            raise ValueError(f"coefficients given for layers without vectors: {sorted(missing)}")
        self._handles = []
        self._prefill_done = False

    def __enter__(self) -> "ActivationSteering":
        layers = decoder_layers(self.model)
        for idx, coeff in self.coefficients.items():
            if not 0 <= idx < len(layers):
                raise ValueError(f"layer {idx} outside 0..{len(layers) - 1}")
            self._handles.append(
                layers[idx].register_forward_hook(self._make_hook(idx, coeff))
            )
        self._prefill_done = False
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def _make_hook(self, layer: int, coeff: float):
        vec = self.vectors[layer]

        def hook(_module, _args, output):
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output

            # During cached generation every step after the prefill has seq_len 1, so
            # "new tokens only" is exactly the seq_len==1 case.
            if self.positions == "new" and h.shape[1] > 1:
                return output

            v = vec.to(device=h.device, dtype=h.dtype)

            if self.mode == "ablate":
                # Project the component along v out of every position. How much is
                # removed depends on the activation, not on a fixed step.
                unit = v / (v.norm() + 1e-6)
                h = h - coeff * (h * unit).sum(dim=-1, keepdim=True) * unit
                return (h,) + output[1:] if is_tuple else h

            if self.preserve_norm:
                before = h.norm(dim=-1, keepdim=True)
                h = h + coeff * v
                h = h * (before / (h.norm(dim=-1, keepdim=True) + 1e-6))
            else:
                h = h + coeff * v

            return (h,) + output[1:] if is_tuple else h

        return hook


def steered_layers(n_layers: int, last_k: int) -> list:
    """The final k layer indices, which is how the paper selects its steering set."""
    if not 1 <= last_k <= n_layers:
        raise ValueError(f"last_k must be in 1..{n_layers}, got {last_k}")
    return list(range(n_layers - last_k, n_layers))
