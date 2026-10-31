"""Reference-site directional ablation and activation addition.

The primary package's ActivationSteering(mode="ablate") hooks decoder-layer OUTPUTS only. The
reference method (andyrdt/refusal_direction @ 0b9c8bc, pipeline/utils/hook_utils.py) removes
the direction from every residual-stream write: the input to every block, and the output of
every attention and every MLP module. That set is what makes hook ablation equivalent to
orthogonalising the weights, and tests/test_hooks.py checks that equivalence on a real model.

D-STEER and the matched additive comparator do not use this module; they use the primary
ActivationSteering unchanged, so their intervention is the submitted one.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

import torch


def _layers(model) -> Sequence[torch.nn.Module]:
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError("cannot locate decoder layers")


def _unit(direction: torch.Tensor) -> torch.Tensor:
    return direction / (direction.norm() + 1e-8)


def _project_out(x: torch.Tensor, unit: torch.Tensor) -> torch.Tensor:
    u = unit.to(device=x.device, dtype=x.dtype)
    return x - (x @ u).unsqueeze(-1) * u


@contextmanager
def reference_ablation(model, direction: torch.Tensor) -> Iterator[None]:
    """Remove `direction` from every residual-stream write, at every position, every layer."""
    unit = _unit(direction.detach().float())
    handles = []

    def pre(_m, args, kwargs):
        if args:
            return (_project_out(args[0], unit),) + tuple(args[1:]), kwargs
        kwargs["hidden_states"] = _project_out(kwargs["hidden_states"], unit)
        return args, kwargs

    def post(_m, _i, out):
        if isinstance(out, tuple):
            return (_project_out(out[0], unit),) + tuple(out[1:])
        return _project_out(out, unit)

    try:
        for layer in _layers(model):
            handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
            handles.append(layer.self_attn.register_forward_hook(post))
            handles.append(layer.mlp.register_forward_hook(post))
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def addition_at_block_input(model, layer: int, vector: torch.Tensor, coeff: float = 1.0) -> Iterator[None]:
    """Reference 'induce' intervention: add coeff * vector to the input of one block."""
    v = vector.detach().float()

    def pre(_m, args, kwargs):
        if args:
            h = args[0]
            return (h + coeff * v.to(device=h.device, dtype=h.dtype),) + tuple(args[1:]), kwargs
        h = kwargs["hidden_states"]
        kwargs["hidden_states"] = h + coeff * v.to(device=h.device, dtype=h.dtype)
        return args, kwargs

    h = _layers(model)[layer].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        yield
    finally:
        h.remove()


@contextmanager
def capture_block_inputs(model, positions: Sequence[int]) -> Iterator[dict]:
    """Collect residual-stream inputs to every block at the given (negative) positions.

    Yields a dict layer -> list of [batch, len(positions), d] tensors (float32, CPU).
    """
    store: dict[int, list] = {}
    handles = []

    def make(i):
        def pre(_m, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            store.setdefault(i, []).append(h[:, list(positions), :].detach().float().cpu())
        return pre

    try:
        for i, layer in enumerate(_layers(model)):
            handles.append(layer.register_forward_pre_hook(make(i), with_kwargs=True))
        yield store
    finally:
        for h in handles:
            h.remove()


def orthogonalise_weights_(model, direction: torch.Tensor) -> None:
    """In-place weight orthogonalisation, the reference's equivalent of hook ablation.

    Test-only: used to check that reference_ablation is the same transformation. Never
    applied to an evaluation endpoint.
    """
    u = _unit(direction.detach().float())

    def ortho_rows_out(w):  # W maps into the residual stream along dim 0
        uu = u.to(w.device, w.dtype)
        return w - torch.outer(uu, uu @ w)

    emb = model.get_input_embeddings().weight
    emb.data = emb.data - (emb.data @ u.to(emb.device, emb.dtype)).unsqueeze(-1) * u.to(emb.device, emb.dtype)
    for layer in _layers(model):
        layer.self_attn.o_proj.weight.data = ortho_rows_out(layer.self_attn.o_proj.weight.data)
        layer.mlp.down_proj.weight.data = ortho_rows_out(layer.mlp.down_proj.weight.data)
