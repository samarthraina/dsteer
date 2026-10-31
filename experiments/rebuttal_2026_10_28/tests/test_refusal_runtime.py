#!/usr/bin/env python3
"""Refusal-comparator extraction and selection on a real tiny model, CPU.

Pins:
  - batched selection scores that depend on padding, e.g. a dropped attention mask
    (a uniform rotary offset is provably harmless under RoPE and is not what this pins);
  - a reference readout taken at the wrong site or position;
  - a D-STEER-parity readout that differs from layer_profile's per-prompt capture;
  - a matched comparator whose saved vectors do not carry D-STEER's norms;
  - an empty post-filter class crashing instead of being recorded as blocked;
  - selection rules diverging from the reference (prune, KL, induce, min bypass).

Run:  .venv/bin/python -I tests/test_refusal_runtime.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(ROOT / "vendor" / "primary_snapshot"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from refusal import Candidate, select  # noqa: E402
from refusal_runtime import extract_prompt_last, last_logits, mean_at_eoi_inputs  # noqa: E402

FAILURES: list[str] = []
CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
          "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def tiny():
    path = json.loads((ROOT / "configs" / "tiny_test_models.json").read_text())[
        "hf-internal-testing/tiny-random-LlamaForCausalLM"]["path"]
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()
    t = AutoTokenizer.from_pretrained(path)
    t.pad_token = t.pad_token or t.eos_token
    t.chat_template = CHATML
    return m, t


PROMPTS = ["hi", "a much longer prompt about several different topics in detail", "medium length prompt"]


def test_padding_invariance():
    print("test_padding_invariance")
    m, t = tiny()
    batched = last_logits(m, t, PROMPTS, batch_size=3)
    single = torch.cat([last_logits(m, t, [p], batch_size=1) for p in PROMPTS])
    err = float((batched - single).abs().max())
    check("batched last-position logits equal single-prompt logits", err < 1e-5, f"max err {err:.2e}")
    # sensitivity: the same comparison with the mask dropped must fail, or this test pins nothing
    from refusal_runtime import _batches
    rows = []
    with torch.no_grad():
        for enc in _batches(t, PROMPTS, 3):
            rows.append(m(input_ids=enc["input_ids"], use_cache=False).logits[:, -1, :].float())
    bad = float((torch.cat(rows) - single).abs().max())
    check("test detects a dropped attention mask", bad > 100 * max(err, 1e-9), f"err without mask {bad:.2e}")


def test_reference_readout_site():
    print("test_reference_readout_site")
    m, t = tiny()
    pos = [-2, -1]
    got = mean_at_eoi_inputs(m, t, PROMPTS, pos, batch_size=1)
    # independent computation: block i input == hidden_states[i] (embeddings for i=0)
    ref = torch.zeros_like(got)
    for p in PROMPTS:
        text = t.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        with torch.no_grad():
            hs = m(**t(text, return_tensors="pt"), output_hidden_states=True).hidden_states
        for li in range(m.config.num_hidden_layers):
            ref[:, li] += hs[li][0, pos, :].double() / len(PROMPTS)
    err = float((got - ref).abs().max())
    check("block-input mean at end-of-instruction positions", err < 1e-4, f"max err {err:.2e}")
    batched = mean_at_eoi_inputs(m, t, PROMPTS, pos, batch_size=3)
    check("reference readout independent of batch size", float((batched - got).abs().max()) < 1e-4)


def test_dsteer_parity_readout():
    print("test_dsteer_parity_readout")
    m, t = tiny()
    acts = extract_prompt_last(m, t, PROMPTS)
    # layer_profile semantics: decoder-layer OUTPUT, final token, default special tokens
    for i, p in enumerate(PROMPTS):
        text = t.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        with torch.no_grad():
            hs = m(**t(text, return_tensors="pt", truncation=True, max_length=512), output_hidden_states=True).hidden_states
        # hidden_states[L+1] is block L output for every block except the last, which HF normalises
        for L in range(m.config.num_hidden_layers - 1):
            if not torch.allclose(acts[L, i], hs[L + 1][0, -1].float(), atol=1e-5):
                check(f"prompt {i} layer {L} equals block output", False)
                return
    check("per-prompt readout equals decoder-layer outputs at the final token", True)


def test_build_matched_and_blocked_filter():
    print("test_build_matched_and_blocked_filter")
    import build_refusal_directions as brd
    m, t = tiny()
    n_layers, d = m.config.num_hidden_layers, m.config.hidden_size
    layers = tuple(range(n_layers))
    brd.LAYERS = layers
    g = torch.Generator().manual_seed(3)
    dv = {L: torch.randn(d, generator=g) * (L + 2) for L in layers}
    dr = torch.randn(n_layers, d, generator=g)
    other = ["tell me about bread", "list three rivers", "explain photosynthesis briefly"]
    brd.prompts = lambda name: [{"prompt": p} for p in (PROMPTS if "harmful" in name else other)]
    with tempfile.TemporaryDirectory() as tmp:
        s = brd.build(m, t, "A", "dpo", Path(tmp), dv, dr, with_candidates=True)
        blob = torch.load(Path(tmp) / "directions.pt", weights_only=False)
        ok = all(abs(float(blob["matched"][L].norm()) - float(dv[L].norm())) < 1e-4 for L in layers)
        check("saved matched vectors carry D-STEER's per-layer norms", ok)
        rows = list(__import__("csv").DictReader(open(Path(tmp) / "geometry.csv")))
        check("geometry rows for every layer", len(rows) == n_layers)
        rf = s.get("reference_filter", {})
        check("filter counts recorded", {"harmful_kept", "harmless_kept"} <= set(rf))
        if rf.get("harmful_kept", 0) == 0 or rf.get("harmless_kept", 0) == 0:
            check("empty post-filter class recorded as blocked, not crashed", "blocked" in rf)


def test_selection_rules():
    print("test_selection_rules")
    n = 10  # layers 8, 9 are the last 20%
    cs = [Candidate(0, 1, bypass=-1.0, induce=0.5, kl=0.05),
          Candidate(0, 2, bypass=-3.0, induce=0.5, kl=0.20),   # kl filter
          Candidate(0, 3, bypass=-4.0, induce=-0.1, kl=0.01),  # induce filter
          Candidate(0, 9, bypass=-9.0, induce=1.0, kl=0.00),   # pruned
          Candidate(1, 4, bypass=-2.0, induce=0.2, kl=0.09),
          Candidate(1, 5, bypass=float("nan"), induce=0.2, kl=0.01)]
    best, all_c = select(cs, n)
    check("minimum bypass among survivors chosen", (best.pos, best.layer) == (1, 4), f"{best.pos},{best.layer}")
    reasons = {(c.pos, c.layer): c.filtered_reason for c in all_c}
    check("kl filter applied", reasons[(0, 2)] == "kl>0.1")
    check("induce filter applied", reasons[(0, 3)] == "induce<0.0")
    check("last-20% layers pruned", reasons[(0, 9)] == "last_20pct_layers")
    check("nan scores filtered", reasons[(1, 5)] == "nan_score")
    none, _ = select([Candidate(0, 9, -1, 1, 0)], n)
    check("no survivor -> no selection, not a fallback", none is None)


def main() -> int:
    torch.manual_seed(0)
    for t in (test_padding_invariance, test_reference_readout_site, test_dsteer_parity_readout,
              test_build_matched_and_blocked_filter, test_selection_rules):
        t()
        print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
