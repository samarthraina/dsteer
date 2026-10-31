"""Torch-side extraction and scoring for the two refusal comparators.

Two readouts, deliberately different, each matching the method it serves:

extract_prompt_last   D-STEER's construction readout, reproduced exactly from layer_profile:
                      build_input_text(prompt_last), tokenizer(text, truncation, max_length=512)
                      with default special tokens, one prompt per forward, use_cache=False,
                      LastTokenCapture on decoder-layer OUTPUTS. Feeds refusal_add_matched.

mean_at_eoi_inputs    the reference readout: residual stream entering each block (forward
                      pre-hook) at every end-of-instruction position, left-padded batches,
                      float64 running mean. Feeds refusal_ablate_reference.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Optional, Sequence

import torch

from hooks import _layers
from refusal import refusal_score

CONSTRUCTION_MAX_INPUT = 512   # Task 025 run_meta: config.eval.max_input_length


@torch.no_grad()
def extract_prompt_last(model, tokenizer, prompts: Sequence[str]) -> torch.Tensor:
    from steering.activations import LastTokenCapture, build_input_text
    device = next(model.parameters()).device
    n_layers = len(_layers(model))
    acts = torch.zeros(n_layers, len(prompts), model.config.hidden_size, dtype=torch.float32)
    with LastTokenCapture(model) as cap:
        for i, p in enumerate(prompts):
            text = build_input_text(tokenizer, {"prompt": p}, "prompt_last")
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=CONSTRUCTION_MAX_INPUT).to(device)
            cap.clear()
            model(**enc, use_cache=False)
            acts[:, i, :] = cap.stack()
    return acts


def _chat(tokenizer, prompts: Sequence[str]) -> list[str]:
    return [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                          add_generation_prompt=True) for p in prompts]


def _positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Position ids from the mask, as generate() derives them.

    A bare forward uses arange, offsetting each left-padded row by a constant. Under RoPE that
    is harmless -- attention depends only on relative positions, which a uniform shift
    preserves, and pads are masked -- so this mirrors generate() rather than fixing a bug.
    """
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)


def _batches(tokenizer, prompts, batch_size):
    side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        texts = _chat(tokenizer, prompts)
        for i in range(0, len(texts), batch_size):
            # default special tokens: the same call generate_batched makes (and the reference's
            # tokenizer call), so selection sees inputs exactly as generation will
            yield tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = side


@torch.no_grad()
def mean_at_eoi_inputs(model, tokenizer, prompts: Sequence[str], positions: Sequence[int],
                       batch_size: int = 32) -> torch.Tensor:
    """[n_pos, n_layers, d] float64 mean of block-input residuals at the given positions."""
    device = next(model.parameters()).device
    layers = _layers(model)
    out = torch.zeros(len(positions), len(layers), model.config.hidden_size, dtype=torch.float64)
    n = len(prompts)
    handles = []

    def make(i):
        def pre(_m, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            out[:, i] += h[:, list(positions), :].double().sum(dim=0).cpu() / n
        return pre

    try:
        for i, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(make(i), with_kwargs=True))
        for enc in _batches(tokenizer, prompts, batch_size):
            model(**{k: v.to(device) for k, v in enc.items()},
                  position_ids=_positions(enc["attention_mask"]).to(device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return out


@torch.no_grad()
def last_logits(model, tokenizer, prompts: Sequence[str], context: Optional[Callable] = None,
                batch_size: int = 32) -> torch.Tensor:
    device = next(model.parameters()).device
    rows = []
    for enc in _batches(tokenizer, prompts, batch_size):
        with (context() if context else nullcontext()):
            logits = model(**{k: v.to(device) for k, v in enc.items()},
                           position_ids=_positions(enc["attention_mask"]).to(device), use_cache=False).logits
        rows.append(logits[:, -1, :].float().cpu())
    return torch.cat(rows)


def scores(model, tokenizer, prompts, refusal_toks, context=None, batch_size=32) -> torch.Tensor:
    return refusal_score(last_logits(model, tokenizer, prompts, context, batch_size), refusal_toks)
