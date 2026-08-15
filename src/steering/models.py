"""Model and tokenizer loading helpers."""

from __future__ import annotations

import logging
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from .config import ModelConfig

log = logging.getLogger(__name__)


def load_tokenizer(
    model_path: str, subfolder: str = "", fallback_pad_token: str = "<|endoftext|>",
    trust_remote_code: bool = True, local_files_only: bool = False,
) -> PreTrainedTokenizer:
    """Load a tokenizer. Ensures pad_token is set.

    If pad_token is missing, use eos_token (standard for causal LMs).

    `trust_remote_code`/`local_files_only` default to the historical behavior (trusted,
    network-allowed) so every existing caller is unaffected; a caller loading from an
    already-verified local endpoint (Task 013's Gate 2 smoke test) passes
    `trust_remote_code=False, local_files_only=True` explicitly instead.
    """
    kwargs = {"trust_remote_code": trust_remote_code, "local_files_only": local_files_only}
    if subfolder:
        kwargs["subfolder"] = subfolder
    try:
        tok = AutoTokenizer.from_pretrained(model_path, **kwargs)
    except (ValueError, KeyError) as e:
        # Some checkpoints declare tokenizer_class="TokenizersBackend" (a
        # transformers 5.x class) which older transformers can't resolve.
        # Fall back to a fast tokenizer backed by tokenizer.json.
        log.warning(f"AutoTokenizer failed ({e}); falling back to PreTrainedTokenizerFast for {model_path}")
        from transformers import PreTrainedTokenizerFast
        fast_kwargs = {"trust_remote_code": trust_remote_code, "local_files_only": local_files_only}
        if subfolder:
            fast_kwargs["subfolder"] = subfolder
        tok = PreTrainedTokenizerFast.from_pretrained(model_path, **fast_kwargs)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
            log.info(f"Set pad_token = eos_token ({tok.eos_token!r}) for {model_path}")
        else:
            tok.pad_token = fallback_pad_token
            log.warning(f"No eos_token; set pad_token = {fallback_pad_token!r} for {model_path}")
    return tok


def load_model(
    model_path: str,
    subfolder: str = "",
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    trust_remote_code: bool = True,
    local_files_only: bool = False,
) -> PreTrainedModel:
    """Load a model in bf16. Sets to eval mode.

    `trust_remote_code`/`local_files_only` default to the historical behavior (trusted,
    network-allowed) so every existing caller is unaffected; a caller loading from an
    already-verified local endpoint (Task 013's Gate 2 smoke test) passes
    `trust_remote_code=False, local_files_only=True` explicitly instead.
    """
    log.info(f"Loading model: {model_path} subfolder={subfolder!r} (dtype={dtype}, device_map={device_map})")
    kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if subfolder:
        kwargs["subfolder"] = subfolder
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return model


def load_pair(cfg: ModelConfig) -> Tuple[PreTrainedTokenizer, PreTrainedModel, PreTrainedModel]:
    """Load the IT/DPO pair specified in a ModelConfig.

    Returns: (tokenizer, it_model, dpo_model)

    Note: both models live on GPU simultaneously. For 8B + 8B in bf16, this needs ~32GB VRAM.
    If you have less VRAM, load and evaluate them sequentially in the calling script.
    """
    tokenizer_path = cfg.tokenizer_id or cfg.it_model
    tok = load_tokenizer(tokenizer_path)

    it = load_model(cfg.it_model)
    dpo = load_model(cfg.dpo_model)

    return tok, it, dpo
