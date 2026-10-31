#!/usr/bin/env python3
"""Intervention tests on a real (tiny, random-weight) Llama and Mistral, CPU.

Random weights mean these pin the MATHS of each intervention and the plumbing, not model
behaviour. Each test names the failure it pins. Behavioural validation happens on GPU.

Run:  .venv/bin/python -I tests/test_hooks.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
sys.path.insert(0, str(ROOT / "vendor" / "primary_snapshot"))
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from steering.steer import ActivationSteering  # noqa: E402
from hooks import capture_block_inputs, orthogonalise_weights_, reference_ablation  # noqa: E402
from refusal import eoi_positions, match_norms, refusal_token_ids  # noqa: E402

FAILURES: list[str] = []
MODELS = json.loads((ROOT / "configs" / "tiny_test_models.json").read_text())
CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
          "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def load(repo: str):
    m = AutoModelForCausalLM.from_pretrained(MODELS[repo]["path"], torch_dtype=torch.float32).eval()
    t = AutoTokenizer.from_pretrained(MODELS[repo]["path"])
    if t.pad_token is None:
        t.pad_token = t.eos_token
    t.chat_template = CHATML
    return m, t


def inputs(tok, n=3):
    texts = [tok.apply_chat_template([{"role": "user", "content": f"question number {i} about tea"}],
                                     tokenize=False, add_generation_prompt=True) for i in range(n)]
    tok.padding_side = "left"
    return tok(texts, return_tensors="pt", padding=True)


def block_outputs(model, enc):
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    return out.logits, out.hidden_states  # hidden_states[i+1] is block i output (last is normed)


def vectors_for(model, layers, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = model.config.hidden_size
    return {L: torch.randn(d, generator=g) for L in layers}


def run(repo: str) -> None:
    print(f"== {repo.split('/')[-1]}")
    model, tok = load(repo)
    enc = inputs(tok)
    n_layers = model.config.num_hidden_layers
    L0 = 0
    layers = [L0]
    base_logits, base_h = block_outputs(model, enc)

    # 1. zero-intervention parity -- pins: hooks that perturb the model even at coefficient 0
    v = vectors_for(model, layers)
    with torch.no_grad(), ActivationSteering(model, v, coefficient=0.0):
        z_logits, _ = block_outputs(model, enc)
    check("coefficient 0 leaves logits bit-identical", torch.equal(z_logits, base_logits))
    with torch.no_grad():
        g0 = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
        with ActivationSteering(model, v, coefficient=0.0):
            g1 = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
    check("coefficient 0 leaves greedy generation identical", torch.equal(g0, g1))

    # 2. addition sign and norm -- pins: wrong sign, wrong layer, or wrong scale
    lam = 0.37
    with torch.no_grad(), ActivationSteering(model, v, coefficient=lam):
        _, s_h = block_outputs(model, enc)
    delta = s_h[L0 + 1] - base_h[L0 + 1]
    expected = (lam * v[L0]).expand_as(delta)
    check("steered layer output shifts by exactly +λv at every position",
          torch.allclose(delta, expected, atol=1e-5), f"max err {float((delta - expected).abs().max()):.2e}")
    with torch.no_grad(), ActivationSteering(model, v, coefficient=-lam):
        _, n_h = block_outputs(model, enc)
    check("negative coefficient shifts by exactly -λv",
          torch.allclose(n_h[L0 + 1] - base_h[L0 + 1], -expected, atol=1e-5))
    check("earlier layers untouched", L0 == 0 or torch.equal(s_h[L0], base_h[L0]))

    # 3. matched norms -- pins: a comparator with a different perturbation energy than D-STEER
    r = vectors_for(model, layers, seed=7)
    r = {L: x * 13.0 for L, x in r.items()}       # deliberately different raw scale
    matched, rows = match_norms(r, v)
    with torch.no_grad(), ActivationSteering(model, matched, coefficient=lam):
        _, m_h = block_outputs(model, enc)
    d_ds = (s_h[L0 + 1] - base_h[L0 + 1])[0, -1].norm().item()
    d_rf = (m_h[L0 + 1] - base_h[L0 + 1])[0, -1].norm().item()
    check("realised comparator perturbation norm == D-STEER's", abs(d_ds - d_rf) < 1e-4,
          f"{d_ds:.6f} vs {d_rf:.6f}")
    check("scaling preserves direction exactly", all(abs(x.cos_scaled_vs_raw - 1) < 1e-9 for x in rows))
    check("pre/post norms recorded", all(abs(x.scaled_norm - x.target_norm) < 1e-6 for x in rows))

    # 4. reference ablation removes the component from every block input
    u = r[L0] / r[L0].norm()
    with torch.no_grad(), reference_ablation(model, r[L0]), capture_block_inputs(model, [-1, -2]) as cap:
        a_logits, _ = block_outputs(model, enc)
    worst = max(float((torch.cat(cap[i]) @ u).abs().max()) for i in range(n_layers))
    check("no residual component along the direction at any block input", worst < 1e-4, f"max |proj| {worst:.2e}")

    # 5. hook ablation == weight orthogonalisation -- pins: missing write sites (e.g. mid-layer)
    m2, _ = load(repo)
    orthogonalise_weights_(m2, r[L0])
    with torch.no_grad():
        o_logits = m2(**enc).logits
    err = float((o_logits - a_logits).abs().max())
    check("hook ablation equals weight orthogonalisation", err < 1e-3, f"max logit err {err:.2e}")

    # 6. ablation is not negative addition -- pins: labelling the additive comparator an ablation
    with torch.no_grad(), ActivationSteering(model, {L: -r[L0] for L in range(n_layers)}, coefficient=1.0):
        neg_logits, _ = block_outputs(model, enc)
    check("ablation differs from subtracting the direction", not torch.allclose(neg_logits, a_logits, atol=1e-3))

    # 7. template-derived end-of-instruction span and refusal ids
    pos = eoi_positions(tok)
    sfx = tok("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    check("eoi span length equals the template suffix", len(pos) == len(sfx), f"{len(pos)} vs {len(sfx)}")
    check("refusal ids resolve", len(refusal_token_ids(tok, "chatml")) >= 1)


def main() -> int:
    torch.manual_seed(0)
    for repo in MODELS:
        run(repo)
        print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
