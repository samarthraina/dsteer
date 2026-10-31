#!/usr/bin/env python3
"""Generation runner on a real tiny model, CPU: schema, resume, immutability, refusals.

Pins:
  - a record missing a field the handoff requires;
  - resume that regenerates or double-counts completed prompts;
  - an output file silently reused by a different condition;
  - regeneration of a recovered Task 027 original on the harmful panel;
  - zero-intervention drift through the full runner path.

Run:  .venv/bin/python -I tests/test_run_panel.py
"""
from __future__ import annotations

import json
import os
import subprocess
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

import run_panel  # noqa: E402
from protocol import CONDITIONS  # noqa: E402
from records import DuplicateRecord, IdentityConflict, ResponseStore  # noqa: E402

FAILURES: list[str] = []
REQUIRED = {"run_id", "protocol_hash", "source_id", "prompt_hash", "pair", "endpoint_identity",
            "intervention_identity", "decoding_identity", "seed", "response", "response_hash",
            "finish_reason", "input_tokens", "output_tokens", "validity", "elapsed_s",
            "started_utc", "finished_utc"}
CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
          "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def check(name: str, cond: bool, detail: str = "") -> None:
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


def rows(n):
    return [{"source_id": f"t-{i}", "prompt": f"short question {i}?", "prompt_hash": f"h{i}"} for i in range(n)]


def meta(name):
    return {"run_id": "test", "protocol_hash": "p", "panel": "unit", "pair": "A",
            "condition": {"name": name}, "endpoint": {"repository": "tiny"}}


def test_schema_resume_and_immutability() -> None:
    print("test_schema_resume_and_immutability")
    m, t = tiny()
    run_panel.DECODING["max_new_tokens"] = 6  # keep CPU test fast; the frozen 768 is checked below
    with tempfile.TemporaryDirectory() as d:
        ident = {"condition": "it_base", "panel": "unit"}
        st = ResponseStore(Path(d) / "c.jsonl", ident)
        run_panel.generate_into(st, m, t, rows(3), None, batch_size=2, meta=meta("it_base"))
        recs = st.records()
        check("3 records written", len(recs) == 3)
        check("every handoff field present", all(REQUIRED <= set(r) for r in recs),
              str(sorted(REQUIRED - set(recs[0]))))
        check("response_hash binds the text",
              all(r["response_hash"] == __import__("hashlib").sha256(r["response"].encode()).hexdigest() for r in recs))
        before = (Path(d) / "c.jsonl").read_bytes()
        st2 = ResponseStore(Path(d) / "c.jsonl", ident)
        run_panel.generate_into(st2, m, t, rows(5), None, batch_size=2, meta=meta("it_base"))
        after = (Path(d) / "c.jsonl").read_bytes()
        check("resume generates only the 2 missing prompts", len(st2.records()) == 5)
        check("completed records byte-identical after resume", after.startswith(before))
        try:
            st2.append(recs[0])
            check("duplicate append refused", False)
        except DuplicateRecord:
            check("duplicate append refused", True)
        try:
            ResponseStore(Path(d) / "c.jsonl", {"condition": "refusal_add_install", "panel": "unit"})
            check("different identity on same file refused", False)
        except IdentityConflict:
            check("different identity on same file refused", True)
        g = json.loads((Path(d) / "c.meta.json").read_text())["generation_gates"]
        check("gates computed over all records and panel marked complete",
              g["n"] == 5 and g["complete"] is True, json.dumps(g))
        st3 = ResponseStore(Path(d) / "c.jsonl", ident)
        run_panel.generate_into(st3, m, t, rows(7)[:5] + [], None, batch_size=2, meta=meta("it_base"))
        g2 = json.loads((Path(d) / "c.meta.json").read_text())["generation_gates"]
        check("a no-op resume writes nothing new", g2["n"] == 5)
    run_panel.DECODING["max_new_tokens"] = 768


def test_zero_intervention_through_runner() -> None:
    print("test_zero_intervention_through_runner")
    from steering.steer import ActivationSteering
    m, t = tiny()
    run_panel.DECODING["max_new_tokens"] = 8
    d = m.config.hidden_size
    v = {0: torch.randn(d, generator=torch.Generator().manual_seed(1))}
    with tempfile.TemporaryDirectory() as tmp:
        a = ResponseStore(Path(tmp) / "a.jsonl", {"x": 1})
        b = ResponseStore(Path(tmp) / "b.jsonl", {"x": 2})
        run_panel.generate_into(a, m, t, rows(4), None, batch_size=2, meta=meta("it_base"))
        run_panel.generate_into(b, m, t, rows(4), lambda: ActivationSteering(m, v, coefficient=0.0),
                                batch_size=2, meta=meta("zero"))
        ra = {r["source_id"]: r["response"] for r in a.records()}
        rb = {r["source_id"]: r["response"] for r in b.records()}
        check("coefficient-0 runner output identical to unsteered", ra == rb)
    run_panel.DECODING["max_new_tokens"] = 768


def test_frozen_decoding_and_refusals() -> None:
    print("test_frozen_decoding_and_refusals")
    check("frozen max_new_tokens 768", run_panel.DECODING["max_new_tokens"] == 768)
    check("frozen max_input_length 2048", run_panel.DECODING["max_input_length"] == 2048)
    check("frozen seed 42", run_panel.DECODING["seed"] == 42)
    check("greedy", run_panel.DECODING["do_sample"] is False)
    py = str(ROOT / ".venv" / "bin" / "python")
    base = [py, "-I", str(ROOT / "scripts" / "run_panel.py"), "--pair", "A", "--panel", "harmful_test",
            "--stage", "t", "--estimate-hours", "0", "--dry-run"]
    r = subprocess.run(base + ["--condition", "dsteer_install"], capture_output=True, text=True)
    check("recovered original refused on harmful panel", r.returncode != 0 and "REFUSED" in (r.stdout + r.stderr))
    r = subprocess.run(base + ["--condition", "it_base", "--parity"], capture_output=True, text=True)
    check("parity without --limit refused", r.returncode != 0 and "--limit" in (r.stdout + r.stderr))
    rows_, dig = run_panel.load_panel("harmful_test")
    check("harmful panel loads with the frozen digest", len(rows_) == 300 and dig == run_panel.HARMFUL_DIGEST)
    fb, _ = run_panel.load_panel("harmful_test_fallback100")
    check("fallback panel carries prompt text", len(fb) == 100 and all(r.get("prompt") for r in fb))
    check("all mandatory conditions present",
          {"it_base", "dpo_base", "dsteer_install", "dsteer_reverse",
           "refusal_add_install", "refusal_add_reverse"} <= set(CONDITIONS))


def main() -> int:
    for t in (test_schema_resume_and_immutability, test_zero_intervention_through_runner,
              test_frozen_decoding_and_refusals):
        t()
        print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
