"""Guards on how activations are read out of a model.

The bug these protect against: `output_hidden_states=True` returns
`[embeddings, L0_out, ..., L(n-2)_out, norm(L(n-1)_out)]`. The final entry has already
passed through the model's final norm, and the raw last-layer residual stream is not in
the tuple at all. Indexing it as `hidden_states[i + 1]` for i in range(num_layers) makes
the last row a differently-scaled quantity from every other row, which shows up as a 3-4x
jump in activation norm at the final layer.

`layer_profile.extract_activations` therefore uses forward hooks instead.

Run with:
    pytest tests/test_activation_extraction.py -v

CPU only; pulls a tiny random model on first run.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from steering.activations import LastTokenCapture, build_input_text, decoder_layers

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(TINY)
    model.eval()
    return model


def test_final_hidden_state_is_post_norm(tiny_model):
    """The documented failure mode. If this ever fails, the tuple layout changed."""
    n = tiny_model.config.num_hidden_layers
    ids = torch.tensor([[1, 2, 3, 4, 5]])

    with torch.no_grad():
        out = tiny_model(input_ids=ids, output_hidden_states=True, use_cache=False)

    assert len(out.hidden_states) == n + 1

    raw_last = {}

    def hook(_m, _a, output):
        raw_last["h"] = output[0] if isinstance(output, tuple) else output

    handle = tiny_model.model.layers[-1].register_forward_hook(hook)
    try:
        with torch.no_grad():
            tiny_model(input_ids=ids, use_cache=False)
    finally:
        handle.remove()

    # The tuple's last entry is NOT the raw residual stream ...
    assert not torch.allclose(out.hidden_states[n], raw_last["h"], atol=1e-4)
    # ... it is that stream after the final norm.
    assert torch.allclose(
        out.hidden_states[n], tiny_model.model.norm(raw_last["h"]), atol=1e-4
    )


def test_hooks_capture_every_layer(tiny_model):
    """Hook-based extraction yields one residual-stream vector per layer."""
    layers = decoder_layers(tiny_model)
    assert len(layers) == tiny_model.config.num_hidden_layers

    ids = torch.tensor([[1, 2, 3]])
    with LastTokenCapture(tiny_model) as cap:
        with torch.no_grad():
            tiny_model(input_ids=ids, use_cache=False)
        acts = cap.stack()

    assert acts.shape == (len(layers), tiny_model.config.hidden_size)

    # Layers 0..n-2 must match the tuple; the last one is where they diverge.
    with torch.no_grad():
        out = tiny_model(input_ids=ids, output_hidden_states=True, use_cache=False)
    for i in range(len(layers) - 1):
        assert torch.allclose(acts[i], out.hidden_states[i + 1][0, -1, :], atol=1e-4)
    assert not torch.allclose(acts[-1], out.hidden_states[len(layers)][0, -1, :], atol=1e-4)


def test_stack_fails_if_a_layer_did_not_fire(tiny_model):
    """A silently-missing layer must raise, not produce a zero row."""
    with LastTokenCapture(tiny_model) as cap:
        with pytest.raises(RuntimeError):
            cap.stack()


def test_build_input_text_positions():
    """response_last extends prompt_last rather than replacing it, and needs a response."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TINY)
    if tok.chat_template is None:
        tok.chat_template = (
            "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
            "{% if add_generation_prompt %}assistant: {% endif %}"
        )

    rec = {"prompt": "why is the sky blue?", "chosen": "rayleigh scattering."}

    prompt_only = build_input_text(tok, rec, "prompt_last")
    with_response = build_input_text(tok, rec, "response_last")

    assert with_response.startswith(prompt_only)
    assert with_response.endswith(rec["chosen"])
    assert len(with_response) > len(prompt_only)

    with pytest.raises(ValueError):
        build_input_text(tok, {"prompt": "x", "chosen": None}, "response_last")

    with pytest.raises(ValueError):
        build_input_text(tok, rec, "middle_token")
