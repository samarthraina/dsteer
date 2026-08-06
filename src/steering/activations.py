"""Reading the residual stream out of a decoder-only model.

Why hooks rather than `output_hidden_states=True`: that tuple is laid out as

    [embeddings, L0_out, ..., L(n-2)_out, norm(L(n-1)_out)]

because the model appends before each block and once more after the final norm. The raw
last-layer residual stream is therefore absent, and treating entry i+1 as "layer i" makes
the final row a differently-scaled quantity from the rest -- visible as a 3-4x jump in
activation norm at the last layer. Hooks give the same object at every depth.

Readout position matters too, and the two options are not interchangeable:

    prompt_last     final token of the prompt, where the model is about to generate
    response_last   final token of prompt + reference response
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch


def decoder_layers(model) -> List:
    """The list of transformer blocks. Covers llama/mistral/qwen-style models."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise AttributeError(f"cannot locate decoder layers on {type(model).__name__}")
    return layers


def build_input_text(tokenizer, record: Dict[str, Optional[str]], token_position: str) -> str:
    """Text whose final token is the position to read.

    prompt_last:   template through the assistant header, so the last token is the one the
                   model would generate from.
    response_last: the same, plus the reference response appended raw, so the last token is
                   the end of the response rather than a trailing template token.
    """
    messages = [{"role": "user", "content": record["prompt"]}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if token_position == "prompt_last":
        return chat
    if token_position == "response_last":
        chosen = record.get("chosen")
        if not chosen:
            raise ValueError("response_last needs a 'chosen' response; use prompt_source=hh_rlhf")
        return chat + chosen
    raise ValueError(f"Unknown token_position: {token_position!r}")


class LastTokenCapture:
    """Context manager collecting the final-token residual stream at every layer.

    Use:
        with LastTokenCapture(model) as cap:
            model(**inputs, use_cache=False)
            acts = cap.stack()          # (num_layers, hidden)
    """

    def __init__(self, model):
        self.layers = decoder_layers(model)
        self.captured: Dict[int, torch.Tensor] = {}
        self._handles = []

    def __enter__(self) -> "LastTokenCapture":
        def make_hook(idx: int):
            def hook(_module, _args, output):
                h = output[0] if isinstance(output, tuple) else output
                self.captured[idx] = h[0, -1, :].detach().float().cpu()
            return hook

        self._handles = [
            layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(self.layers)
        ]
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def clear(self) -> None:
        self.captured.clear()

    def stack(self) -> torch.Tensor:
        """(num_layers, hidden). Raises if any layer failed to fire."""
        if len(self.captured) != len(self.layers):
            raise RuntimeError(
                f"captured {len(self.captured)} layers, expected {len(self.layers)}"
            )
        return torch.stack([self.captured[i] for i in range(len(self.layers))])
