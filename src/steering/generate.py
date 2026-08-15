"""Batched greedy generation.

The existing eval scripts generate one prompt at a time. Measured on an A100 that
leaves the card at 38-45% utilisation and under a fifth of its memory during a plain
forward pass; decoding 512 tokens one sequence at a time is worse still, because decode
is bandwidth-bound and a batch of one wastes almost all of it. A lambda sweep is
prompts x lambda values x directions, so the constant matters: at batch 1 the full grid
is days, and batched it is hours.

Left padding is required, not preferred. A decoder-only model reads the last position
as "what comes next"; right-padded sequences put padding there and the continuation is
generated from pad tokens.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import torch
from tqdm import tqdm

log = logging.getLogger(__name__)

# Chat templates end an assistant turn with a model-specific token, not always EOS --
# Llama-3's `<|eot_id|>`, Qwen/ChatML's `<|im_end|>`. Add more here as new families
# enter the sweep.
KNOWN_EOT_TOKENS = ("<|eot_id|>", "<|im_end|>")


def generation_terminators(tokenizer) -> List[int]:
    """Unique stop-token IDs for `model.generate`: EOS plus any known chat end-of-turn
    token the tokenizer actually has.

    `tokenizer.eos_token_id` alone misses the real stop for templates like Llama-3's,
    so generation runs past the model's own turn boundary to `max_new_tokens`. Looking
    tokens up in `get_vocab()` rather than `convert_tokens_to_ids` means a token the
    tokenizer does not have is simply absent, not silently aliased to `<unk>`.
    """
    ids: List[int] = []
    seen = set()

    def add(token_id):
        if isinstance(token_id, int) and token_id >= 0 and token_id not in seen:
            seen.add(token_id)
            ids.append(token_id)

    add(tokenizer.eos_token_id)
    vocab = tokenizer.get_vocab()
    for token in KNOWN_EOT_TOKENS:
        add(vocab.get(token))

    return ids


def build_chat_prompts(tokenizer, prompts: Sequence) -> List[str]:
    """Apply the chat template to each prompt, ready for generation.

    Accepts either a plain string or an already-structured message list (HH-RLHF).
    """
    out = []
    for p in prompts:
        messages = p if isinstance(p, list) else [{"role": "user", "content": p}]
        out.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    return out


@torch.no_grad()
def generate_batched(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_new_tokens: int = 512,
    batch_size: int = 32,
    max_input_length: int = 2048,
    context: Optional[Callable] = None,
    desc: str = "Generating",
) -> List[str]:
    """Greedy-decode `prompts`, returning the continuations only.

    context: a zero-argument callable returning a context manager wrapped around each
    batch -- this is where ActivationSteering goes. Kept as a factory rather than an
    instance so hooks are registered and removed per batch instead of living across the
    whole sweep.
    """
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    outputs: List[str] = []

    terminators = generation_terminators(tokenizer)
    stop_ids = terminators[0] if len(terminators) == 1 else terminators

    try:
        for start in tqdm(range(0, len(prompts), batch_size), desc=desc, unit="batch"):
            chunk = list(prompts[start:start + batch_size])
            enc = tokenizer(
                chunk, return_tensors="pt", padding=True,
                truncation=True, max_length=max_input_length,
            ).to(device)

            ctx = context() if context is not None else nullcontext()
            with ctx:
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=stop_ids,
                )

            # Left padding means every sequence's continuation starts at the same index.
            gen = out[:, enc["input_ids"].shape[1]:]
            outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = original_side

    return [o.strip() for o in outputs]


def suggest_batch_size(model, hidden_context_tokens: int = 2560, safety: float = 0.75) -> int:
    """A batch size that fits the free VRAM, from the KV-cache cost per sequence.

    Rough by design -- it exists so the sweep does not run at batch 1 on an 80 GB card
    because nobody tuned it, and it is capped low enough to stay clear of the ragged
    edge where a long batch OOMs mid-sweep.
    """
    if not torch.cuda.is_available():
        return 8

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    bytes_per_elem = 2  # bf16/fp16 cache

    per_seq = 2 * n_layers * n_kv * head_dim * hidden_context_tokens * bytes_per_elem
    free, _total = torch.cuda.mem_get_info()
    est = int(free * safety // max(per_seq, 1))
    return max(1, min(est, 64))
