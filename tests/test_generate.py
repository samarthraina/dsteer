"""Guards on generation termination.

The observed fault: `generate_batched` stopped only on `tokenizer.eos_token_id`, but a
chat template like Llama-3's ends the assistant turn with `<|eot_id|>` instead, so
decoding ran to `max_new_tokens` past the model's real stop. These pin the terminator
lookup and its plumbing into `model.generate` without loading a real model or tokenizer.

Run with:
    pytest tests/test_generate.py -v
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from steering.generate import generate_batched, generation_terminators


class _FakeTokenizer:
    """Just enough of the HF tokenizer surface for `generation_terminators` and
    `generate_batched`, with no real vocab or model download."""

    def __init__(self, vocab, eos_token_id=None, pad_token_id=0):
        self._vocab = dict(vocab)
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.padding_side = "right"

    def get_vocab(self):
        return self._vocab

    def __call__(self, texts, return_tensors=None, padding=None, truncation=None, max_length=None):
        ids = torch.tensor([[1, 2, 3] for _ in texts])
        return _Batch(input_ids=ids, attention_mask=torch.ones_like(ids))

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["x" for _ in ids]


class _Batch(dict):
    """Stand-in for a `BatchEncoding`: dict for `**enc`/`enc[...]`, with `.to(device)`."""

    def to(self, device):
        return self


class _FakeModel:
    def __init__(self):
        self.calls = []

    def parameters(self):
        yield torch.zeros(1)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        extra = torch.zeros((input_ids.shape[0], 1), dtype=input_ids.dtype)
        return torch.cat([input_ids, extra], dim=1)


# Terminator resolution


def test_llama_style_tokenizer_gets_eos_plus_eot():
    tok = _FakeTokenizer({"<|eot_id|>": 128009, "<|im_end|>": 999, "other": 5}, eos_token_id=2)
    assert generation_terminators(tok) == [2, 128009, 999]


def test_tokenizer_without_chat_eot_returns_eos_only():
    tok = _FakeTokenizer({"other": 5}, eos_token_id=2)
    assert generation_terminators(tok) == [2]


def test_duplicate_eos_and_eot_id_is_not_repeated():
    tok = _FakeTokenizer({"<|eot_id|>": 2}, eos_token_id=2)
    assert generation_terminators(tok) == [2]


def test_missing_eot_token_is_not_aliased_to_unknown():
    """`<|im_end|>` absent from vocab must be skipped, not resolved to some other id."""
    tok = _FakeTokenizer({}, eos_token_id=2)
    assert generation_terminators(tok) == [2]


def test_no_eos_and_no_known_eot_returns_empty():
    tok = _FakeTokenizer({}, eos_token_id=None)
    assert generation_terminators(tok) == []


# Plumbing into generate_batched


def test_generate_batched_passes_multiple_terminators_as_a_list():
    tok = _FakeTokenizer({"<|eot_id|>": 7}, eos_token_id=2)
    model = _FakeModel()

    generate_batched(model, tok, ["hello", "world"], max_new_tokens=1, batch_size=2)

    assert model.calls[0]["eos_token_id"] == [2, 7]


def test_generate_batched_passes_single_terminator_as_an_int():
    tok = _FakeTokenizer({}, eos_token_id=2)
    model = _FakeModel()

    generate_batched(model, tok, ["hello"], max_new_tokens=1, batch_size=1)

    assert model.calls[0]["eos_token_id"] == 2
