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
# Llama-3's `<|eot_id|>`, Qwen/ChatML's `<|im_end|>`. `<|end_of_text|>` is Llama-3's own
# base-model terminator, which some endpoints emit in place of (or in addition to)
# `<|eot_id|>` (Task 021C). Add more here as new families enter the sweep.
KNOWN_EOT_TOKENS = ("<|eot_id|>", "<|im_end|>", "<|end_of_text|>")

#: Frozen exact stop-string collection (Task 021D/022): validated against real Pair-B
#: DPO-merged generations as a structurally-complete boundary marking the start of an
#: unwanted next training example, never mid-answer. Exact literal match only -- never
#: regex, never `### Response:`, never broad role-word matching.
STOP_STRINGS: Tuple[str, ...] = ("### Instruction:",)


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
        "eos_token"         -- the first genuine terminator hit was the tokenizer's
                                EOS ID.
        "end_of_turn_token" -- the first genuine terminator hit was a known chat
                                end-of-turn ID distinct from EOS (e.g. Llama-3's
                                `<|eot_id|>`).
        "stop_string"        -- this row's own content contains an exact
                                `STOP_STRINGS` match strictly before any genuine
                                token-ID terminator (Task 021D/022) -- an unwanted
                                next-example boundary, not a real end-of-turn. `text`
                                is truncated to everything strictly before the match.
        "max_new_tokens"    -- no terminator or stop string occurred and generation
                                used the full budget.
        "unknown"           -- generation stopped short of the budget with no known
                                terminator or stop string in this row's own
                                continuation. Not assumed to be a normal stop --
                                recorded distinctly so it can be audited.
    stop_token_id: the terminator's ID, or None when no single-token terminator caused
                   the stop ("stop_string", "max_new_tokens", or "unknown").
    has_post_terminator_continuation: True only when a terminator was hit AND some later
        position in this row's continuation holds a token that is neither the pad token
        nor any terminator -- genuine generated content past the model's own stop point,
        as opposed to the batch simply padding a finished row out to the batch's common
        length (Task 013, Gate 2). Always False when no terminator was hit, since the
        question does not apply. Defaults to False so existing construction sites and
        the pre-Task-013 API are unaffected.
    matched_stop_string: the exact `STOP_STRINGS` member matched, when
        stop_reason=="stop_string" -- otherwise None. Defaults to None so existing
        construction sites and the pre-Task-022 API are unaffected.
    """
    text: str
    generated_token_count: int
    stop_reason: str
    stop_token_id: Optional[int]
    has_post_terminator_continuation: bool = False
    matched_stop_string: Optional[str] = None


def _first_terminator_metadata(
    ids: Sequence[int],
    eos_token_id: Optional[int],
    terminators: Sequence[int],
    max_new_tokens: int,
    pad_token_id: Optional[int] = None,
) -> Tuple[int, str, Optional[int], bool]:
    """One row's (generated_token_count, stop_reason, stop_token_id,
    has_post_terminator_continuation), from the first token-ID terminator this row's
    own continuation contains -- with no awareness of stop strings at all.

    `ids` must already be just this row's own continuation (the slice after the common
    left-padded input width) -- scanning stops at the first terminator found, so tokens
    or batch-padding positions after it, and any other row's tokens, never count toward
    `generated_token_count`/`stop_reason`/`stop_token_id`. The post-terminator flag is
    the one exception: it is computed from exactly what remains after that point, to
    distinguish ordinary batch padding from a genuine unexplained continuation.

    This function's "first terminator" is only a *candidate* boundary, not necessarily
    the row's true earliest stop: when `pad_token_id` equals a terminator ID (a common
    real configuration -- `tokenizer.pad_token = tokenizer.eos_token`), a row that
    really stopped via an exact stop string gets padded with that same ID afterward,
    and this scan cannot tell that padding from a genuine terminator token by ID alone.
    `_row_generation_result` is the one that resolves that ambiguity, by checking
    whether an exact stop string occurs even earlier in this row's own content than
    whatever this function reports -- callers must go through it rather than trusting
    this function's output directly whenever stop strings are in play.

    The `length >= max_new_tokens` fallback below is exact only when no terminator is
    found *and* nothing else could have caused an early stop -- true before stop
    strings existed, since `ids` (the batch-padded slice) then had no way to be longer
    than this row's own real content. Task 022 introduces exactly that possibility, so
    callers that also check for a stop string must not trust this fallback's `length`
    for that case -- see `_row_generation_result`, which recomputes it from the
    pad-stripped own-content length instead of taking this fallback at face value.
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


def _own_content_length(ids: Sequence[int], pad_token_id: Optional[int]) -> int:
    """This row's own generated length, before any trailing run of `pad_token_id` --
    ordinary batch padding added when another row in the same batch continues longer.

    Needed because a row that stops via an exact stop string (Task 022) carries no
    single terminator token for `_first_terminator_metadata` to find, so without this
    its batch-padded length would be mistaken for genuine content and misreported as
    "max_new_tokens".
    """
    if pad_token_id is None:
        return len(ids)
    n = len(ids)
    while n > 0 and ids[n - 1] == pad_token_id:
        n -= 1
    return n


def _find_stop_string(text: str, stop_strings: Sequence[str]) -> Optional[Tuple[str, int]]:
    """The earliest occurrence of any `stop_strings` member in `text`, as
    `(matched_string, char_index)` -- or None if none occurs.

    Exact substring search only (`str.find`) -- never regex, never fuzzy/approximate
    matching (Task 022).
    """
    earliest: Optional[Tuple[str, int]] = None
    for s in stop_strings:
        idx = text.find(s)
        if idx != -1 and (earliest is None or idx < earliest[1]):
            earliest = (s, idx)
    return earliest


def _row_generation_result(
    ids: Sequence[int],
    full_text: str,
    eos_token_id: Optional[int],
    terminators: Sequence[int],
    stop_strings: Sequence[str],
    max_new_tokens: int,
    pad_token_id: Optional[int],
    decode_own_content: Callable[[Sequence[int]], str],
) -> "GenerationResult":
    """One row's complete `GenerationResult`, using whichever boundary -- a token-ID
    terminator or an exact stop string -- this row's own content actually reaches
    first (Task 021D/022 correction).

    `_first_terminator_metadata` finds a candidate token-ID terminator, if any, but
    that candidate can be a false positive: `model.generate` pads a row that has
    already finished with `pad_token_id`, and when `pad_token_id` equals a terminator
    ID (e.g. `tokenizer.pad_token = tokenizer.eos_token`), a row that really stopped
    via an exact stop string is then padded with a value indistinguishable, by ID
    alone, from a genuine terminator. So a found candidate is only trusted once this
    row's own content *strictly before* it has been checked for an earlier exact
    stop-string match: under real `model.generate` semantics with both `eos_token_id`
    and `stop_strings` registered, a row can never carry a genuine terminator token
    after a point where a stop string was already satisfied -- generation would have
    ended there first -- so finding a stop string in that earlier prefix can only mean
    the candidate terminator is padding, never a second, independent event. If no
    earlier stop string is found there, the candidate is genuine and this row's result
    is exactly what it always was, using the full batch-decoded text unmodified
    (existing behavior, never truncated for that case).

    Only when no token-ID terminator candidate exists at all does this fall back to
    searching this row's own content (batch padding stripped first --
    `_own_content_length`) for a stop string, so another row continuing longer in the
    same batch can never misclassify this one as "max_new_tokens" or inflate its token
    count. A stop-string match (whichever branch finds it) truncates the returned text
    at the marker, matching the existing `.strip()` normalisation applied everywhere
    else; every other outcome (no terminator, no stop string) is unchanged from before
    stop strings existed except for using this corrected own-content length instead of
    the raw (potentially padded) one.
    """
    count, reason, stop_id, has_post = _first_terminator_metadata(
        ids, eos_token_id, terminators, max_new_tokens, pad_token_id,
    )
    if reason in ("eos_token", "end_of_turn_token"):
        if stop_strings:
            terminator_index = count - 1
            pre_terminator_text = decode_own_content(ids[:terminator_index])
            earlier_match = _find_stop_string(pre_terminator_text, stop_strings)
            if earlier_match is not None:
                matched_string, char_idx = earlier_match
                return GenerationResult(
                    text=pre_terminator_text[:char_idx].strip(), generated_token_count=terminator_index,
                    stop_reason="stop_string", stop_token_id=None,
                    has_post_terminator_continuation=False,
                    matched_stop_string=matched_string,
                )
        return GenerationResult(
            text=full_text.strip(), generated_token_count=count,
            stop_reason=reason, stop_token_id=stop_id,
            has_post_terminator_continuation=has_post,
        )

    own_length = _own_content_length(ids, pad_token_id)
    own_text = decode_own_content(ids[:own_length]) if stop_strings else ""
    match = _find_stop_string(own_text, stop_strings) if stop_strings else None
    if match is not None:
        matched_string, char_idx = match
        return GenerationResult(
            text=own_text[:char_idx].strip(), generated_token_count=own_length,
            stop_reason="stop_string", stop_token_id=None,
            has_post_terminator_continuation=False,
            matched_stop_string=matched_string,
        )

    if own_length >= max_new_tokens:
        return GenerationResult(
            text=full_text.strip(), generated_token_count=max_new_tokens,
            stop_reason="max_new_tokens", stop_token_id=None,
            has_post_terminator_continuation=False,
        )
    return GenerationResult(
        text=full_text.strip(), generated_token_count=own_length,
        stop_reason="unknown", stop_token_id=None,
        has_post_terminator_continuation=False,
    )


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

    def decode_own_content(own_ids: Sequence[int]) -> str:
        return tokenizer.decode(own_ids, skip_special_tokens=True)

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
                    stop_strings=list(STOP_STRINGS),
                    tokenizer=tokenizer,
                )

            # Left padding means every sequence's continuation starts at the same index.
            gen = out[:, enc["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)

            if return_metadata:
                for row_ids, text in zip(gen.tolist(), decoded):
                    meta_outputs.append(_row_generation_result(
                        row_ids, text, tokenizer.eos_token_id, terminators, STOP_STRINGS,
                        max_new_tokens, tokenizer.pad_token_id, decode_own_content,
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
