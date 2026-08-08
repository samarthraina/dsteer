"""Guards on the steering intervention itself.

Every failure mode here is silent. A hook registered on the wrong layer, a coefficient
that never reaches the residual stream, a vector built by the wrong recipe, or right-
padded batched generation all produce plausible text and wrong numbers. So each is
checked against an arithmetic expectation rather than eyeballed.

Run with:
    pytest tests/test_steering.py -v

CPU only; pulls a tiny random model on first run.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from steering.activations import LastTokenCapture, decoder_layers
from steering.steer import (
    ActivationSteering,
    build_vectors,
    random_vectors_like,
    steered_layers,
)

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(TINY)
    m.eval()
    return m


@pytest.fixture
def acts_file(tmp_path):
    """A stand-in activations.pt with a known displacement."""
    n_layers, n_prompts, hidden = 4, 40, 16
    torch.manual_seed(0)
    it = torch.randn(n_layers, n_prompts, hidden)
    direction = torch.randn(hidden)
    direction = direction / direction.norm()
    # A shared shift plus per-sample noise: the mean must recover the shared part.
    dpo = it + 3.0 * direction + 0.1 * torch.randn(n_layers, n_prompts, hidden)
    path = tmp_path / "activations.pt"
    torch.save({"it": it, "dpo": dpo}, path)
    return path, direction


# Vector construction


def test_mean_vector_recovers_the_shared_shift(acts_file):
    path, direction = acts_file
    v = build_vectors(path, method="mean")
    assert set(v) == {0, 1, 2, 3}
    got = v[0] / v[0].norm()
    assert torch.dot(got, direction).abs() > 0.99
    assert v[0].norm() == pytest.approx(3.0, rel=0.05)


def test_pc_vector_is_not_the_mean(acts_file):
    """pc3 comes from the centred matrix, so it is near-orthogonal to the mean.

    This is the fork between what the paper describes and what v1 ran.
    """
    path, _ = acts_file
    mean = build_vectors(path, method="mean")[0]
    pc = build_vectors(path, method="pc0")[0]
    cos = torch.dot(mean / mean.norm(), pc / pc.norm()).abs()
    assert cos < 0.3
    # pc is rescaled to the mean's norm, so magnitude alone cannot distinguish them.
    assert pc.norm() == pytest.approx(mean.norm(), rel=1e-4)


def test_normalise_modes(acts_file):
    path, _ = acts_file
    assert build_vectors(path, method="mean", normalise="unit")[0].norm() == pytest.approx(1.0)
    natural = build_vectors(path, method="mean")[0].norm()
    relative = build_vectors(path, method="mean", normalise="relative")[0].norm()
    assert relative != pytest.approx(natural)


def test_layer_selection_and_bounds(acts_file):
    path, _ = acts_file
    assert set(build_vectors(path, method="mean", layers=[1, 3])) == {1, 3}
    with pytest.raises(ValueError):
        build_vectors(path, method="mean", layers=[99])
    with pytest.raises(ValueError):
        build_vectors(path, method="nonsense")


def test_random_control_matches_norms_but_not_direction(acts_file):
    path, _ = acts_file
    real = build_vectors(path, method="mean")
    rand = random_vectors_like(real, seed=1)
    for layer, v in real.items():
        assert rand[layer].norm() == pytest.approx(v.norm(), rel=1e-5)
        cos = torch.dot(v / v.norm(), rand[layer] / rand[layer].norm()).abs()
        assert cos < 0.5


def test_steered_layers_counts_from_the_top():
    assert steered_layers(32, 5) == [27, 28, 29, 30, 31]
    assert steered_layers(32, 1) == [31]
    with pytest.raises(ValueError):
        steered_layers(32, 0)


# The intervention


def test_steering_shifts_the_residual_stream_by_exactly_lambda_v(tiny_model):
    """h_steered - h_clean must equal lambda * v at the steered layer."""
    hidden = tiny_model.config.hidden_size
    n = tiny_model.config.num_hidden_layers
    target = n - 1
    v = torch.randn(hidden)
    ids = torch.tensor([[1, 2, 3, 4]])

    with LastTokenCapture(tiny_model) as cap:
        with torch.no_grad():
            tiny_model(input_ids=ids, use_cache=False)
        clean = cap.stack()

    with ActivationSteering(tiny_model, {target: v}, coefficient=0.4):
        with LastTokenCapture(tiny_model) as cap:
            with torch.no_grad():
                tiny_model(input_ids=ids, use_cache=False)
            steered = cap.stack()

    assert torch.allclose(steered[target] - clean[target], 0.4 * v, atol=1e-4)
    # Earlier layers run before the hook, so they must be untouched.
    for i in range(target):
        assert torch.allclose(steered[i], clean[i], atol=1e-5)


def test_hooks_are_removed_on_exit(tiny_model):
    """A leaked hook would silently steer every later run in the process."""
    hidden = tiny_model.config.hidden_size
    v = torch.randn(hidden)
    ids = torch.tensor([[1, 2, 3]])

    with LastTokenCapture(tiny_model) as cap:
        with torch.no_grad():
            tiny_model(input_ids=ids, use_cache=False)
        before = cap.stack()

    with ActivationSteering(tiny_model, {0: v}, coefficient=1.0):
        with torch.no_grad():
            tiny_model(input_ids=ids, use_cache=False)

    with LastTokenCapture(tiny_model) as cap:
        with torch.no_grad():
            tiny_model(input_ids=ids, use_cache=False)
        after = cap.stack()

    assert torch.allclose(before, after, atol=1e-6)


def test_zero_coefficient_is_a_no_op(tiny_model):
    hidden = tiny_model.config.hidden_size
    v = torch.randn(hidden)
    ids = torch.tensor([[1, 2, 3]])

    with LastTokenCapture(tiny_model) as cap:
        with torch.no_grad():
            tiny_model(input_ids=ids, use_cache=False)
        clean = cap.stack()

    with ActivationSteering(tiny_model, {1: v}, coefficient=0.0):
        with LastTokenCapture(tiny_model) as cap:
            with torch.no_grad():
                tiny_model(input_ids=ids, use_cache=False)
            steered = cap.stack()

    assert torch.allclose(clean, steered, atol=1e-6)


def test_per_layer_coefficients(tiny_model):
    hidden = tiny_model.config.hidden_size
    v0, v1 = torch.randn(hidden), torch.randn(hidden)
    ids = torch.tensor([[1, 2, 3]])

    with LastTokenCapture(tiny_model) as cap:
        with torch.no_grad():
            tiny_model(input_ids=ids, use_cache=False)
        clean = cap.stack()

    with ActivationSteering(tiny_model, {0: v0, 1: v1}, coefficient={0: 0.25, 1: 0.75}):
        with LastTokenCapture(tiny_model) as cap:
            with torch.no_grad():
                tiny_model(input_ids=ids, use_cache=False)
            steered = cap.stack()

    assert torch.allclose(steered[0] - clean[0], 0.25 * v0, atol=1e-4)


def test_preserve_norm_keeps_magnitude(tiny_model):
    hidden = tiny_model.config.hidden_size
    v = torch.randn(hidden) * 5
    ids = torch.tensor([[1, 2, 3]])

    with LastTokenCapture(tiny_model) as cap:
        with torch.no_grad():
            tiny_model(input_ids=ids, use_cache=False)
        clean = cap.stack()

    with ActivationSteering(tiny_model, {0: v}, coefficient=1.0, preserve_norm=True):
        with LastTokenCapture(tiny_model) as cap:
            with torch.no_grad():
                tiny_model(input_ids=ids, use_cache=False)
            steered = cap.stack()

    assert steered[0].norm() == pytest.approx(clean[0].norm(), rel=2e-3)
    assert not torch.allclose(steered[0], clean[0], atol=1e-4)


def test_rejects_bad_arguments(tiny_model):
    v = torch.randn(tiny_model.config.hidden_size)
    with pytest.raises(ValueError):
        ActivationSteering(tiny_model, {0: v}, positions="sometimes")
    with pytest.raises(ValueError):
        ActivationSteering(tiny_model, {0: v}, coefficient={5: 1.0})
    with pytest.raises(ValueError):
        with ActivationSteering(tiny_model, {999: v}, coefficient=1.0):
            pass


# Batched generation


def test_batched_generation_matches_single(tiny_model):
    """Left padding must make a batch give the same greedy output as one at a time.

    Right padding puts pad tokens at the final position and the model continues from
    those instead of the prompt -- plausible text, wrong experiment.
    """
    from transformers import AutoTokenizer

    from steering.generate import generate_batched

    tok = AutoTokenizer.from_pretrained(TINY)
    tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        tok.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"

    prompts = ["the capital of France", "a much longer prompt about something else entirely", "hi"]
    batched = generate_batched(tiny_model, tok, prompts, max_new_tokens=6, batch_size=3)
    one_at_a_time = generate_batched(tiny_model, tok, prompts, max_new_tokens=6, batch_size=1)
    assert batched == one_at_a_time


def test_generation_restores_padding_side(tiny_model):
    from transformers import AutoTokenizer

    from steering.generate import generate_batched

    tok = AutoTokenizer.from_pretrained(TINY)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    generate_batched(tiny_model, tok, ["hello"], max_new_tokens=2, batch_size=1)
    assert tok.padding_side == "right"


def test_hold_out_excludes_the_evaluation_slice(tmp_path):
    """Vectors must be buildable from samples the sweep will not be scored on.

    The prompt loaders shuffle on a fixed seed and take the first n, so a profile of 1900
    prompts and a sweep of 300 share their first 300 -- the vector is then estimated
    partly from the prompts it is measured on. `skip_first` removes that overlap, and
    this pins it by giving the leading slice a different displacement from the rest:
    holding it out has to recover the rest's direction, keeping it must not.
    """
    torch.manual_seed(0)
    n_layers, n_prompts, hidden, held = 2, 1000, 16, 300
    it = torch.randn(n_layers, n_prompts, hidden)
    rest = torch.randn(hidden)
    rest = rest / rest.norm()
    lead = torch.randn(hidden)
    lead = lead / lead.norm()

    dpo = it + rest
    dpo[:, :held] = it[:, :held] + 5.0 * lead

    path = tmp_path / "activations.pt"
    torch.save({"it": it, "dpo": dpo}, path)

    full = build_vectors(path, method="mean", layers=[0])[0]
    kept = build_vectors(path, method="mean", layers=[0], skip_first=held)[0]

    assert torch.dot(kept / kept.norm(), rest) > 0.99
    assert torch.dot(full / full.norm(), rest) < 0.9
    assert torch.dot(full / full.norm(), kept / kept.norm()).abs() < 0.9

    with pytest.raises(ValueError):
        build_vectors(path, method="mean", layers=[0], skip_first=n_prompts)
