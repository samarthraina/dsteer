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
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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


@dataclass
class GenerationResult:
    """One generated continuation, with exact stop provenance.

    Computed from the generated token IDs, not the decoded text -- special tokens are
    removed during decoding (`skip_special_tokens=True`) and cannot be recovered
    reliably from the string afterward.

    stop_reason:
        "eos_token"         -- the first terminator hit was the tokenizer's EOS ID.
        "end_of_turn_token" -- the first terminator hit was a known chat end-of-turn ID
                                distinct from EOS (e.g. Llama-3's `<|eot_id|>`).
        "max_new_tokens"    -- no terminator occurred and generation used the full budget.
        "unknown"           -- generation stopped short of the budget with no known
                                terminator in this row's own continuation. Not assumed to
                                be a normal stop -- recorded distinctly so it can be audited.
    stop_token_id: the terminator's ID, or None when no terminator caused the stop
                   ("max_new_tokens" or "unknown").
    has_post_terminator_continuation: True only when a terminator was hit AND some later
        position in this row's continuation holds a token that is neither the pad token
        nor any terminator -- genuine generated content past the model's own stop point,
        as opposed to the batch simply padding a finished row out to the batch's common
        length (Task 013, Gate 2). Always False when no terminator was hit, since the
        question does not apply. Defaults to False so existing construction sites and
        the pre-Task-013 API are unaffected.
    """
    text: str
    generated_token_count: int
    stop_reason: str
    stop_token_id: Optional[int]
    has_post_terminator_continuation: bool = False


def _first_terminator_metadata(
    ids: Sequence[int],
    eos_token_id: Optional[int],
    terminators: Sequence[int],
    max_new_tokens: int,
    pad_token_id: Optional[int] = None,
) -> Tuple[int, str, Optional[int], bool]:
    """One row's (generated_token_count, stop_reason, stop_token_id,
    has_post_terminator_continuation).

    `ids` must already be just this row's own continuation (the slice after the common
    left-padded input width) -- scanning stops at the first terminator found, so tokens
    or batch-padding positions after it, and any other row's tokens, never count toward
    `generated_token_count`/`stop_reason`/`stop_token_id`. The post-terminator flag is
    the one exception: it is computed from exactly what remains after that point, to
    distinguish ordinary batch padding from a genuine unexplained continuation.
    """
    terminator_set = set(terminators)
    for i, tok in enumerate(ids):
        if tok in terminator_set:
            reason = "eos_token" if tok == eos_token_id else "end_of_turn_token"
            rest = ids[i + 1:]
            has_post = any(t != pad_token_id and t not in terminator_set for t in rest)
            return i + 1, reason, tok, has_post

    length = len(ids)
    if length >= max_new_tokens:
        return max_new_tokens, "max_new_tokens", None, False
    return length, "unknown", None, False


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
    return_metadata: bool = False,
) -> Union[List[str], List[GenerationResult]]:
    """Greedy-decode `prompts`, returning the continuations only.

    context: a zero-argument callable returning a context manager wrapped around each
    batch -- this is where ActivationSteering goes. Kept as a factory rather than an
    instance so hooks are registered and removed per batch instead of living across the
    whole sweep.

    return_metadata: False (default) returns `List[str]`, unchanged from before this
    existed. True returns `List[GenerationResult]` -- the same decoded text plus exact
    per-row stop provenance, computed independently for every row so one sequence's
    early termination cannot affect another's count or reason.
    """
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    text_outputs: List[str] = []
    meta_outputs: List[GenerationResult] = []

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
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)

            if return_metadata:
                for row_ids, text in zip(gen.tolist(), decoded):
                    count, reason, stop_id, has_post = _first_terminator_metadata(
                        row_ids, tokenizer.eos_token_id, terminators, max_new_tokens, tokenizer.pad_token_id,
                    )
                    meta_outputs.append(GenerationResult(
                        text=text.strip(), generated_token_count=count,
                        stop_reason=reason, stop_token_id=stop_id,
                        has_post_terminator_continuation=has_post,
                    ))
            else:
                text_outputs.extend(decoded)
    finally:
        tokenizer.padding_side = original_side

    if return_metadata:
        return meta_outputs
    return [o.strip() for o in text_outputs]


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
