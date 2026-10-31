#!/usr/bin/env python3
"""Frozen-condition generation: one (pair, condition, panel) per invocation, resumable.

The scientific path is the primary one, imported from the vendored snapshot unchanged:
build_chat_prompts, generate_batched (greedy, 768 new tokens, 2048 input tokens, frozen stop
handling), ActivationSteering (add, all positions, no norm preservation), validity.check,
set_all_seeds(42). The only new intervention is reference_ablation (src/hooks.py), used by
refusal_ablate_reference alone.

Refusals, by design:
  - a recovered Task 027 condition on the harmful panel is never regenerated; --parity runs a
    small check into outputs/parity/ instead and never touches outputs/generations/;
  - an endpoint without a verified entry in manifests/endpoints.json is not loaded;
  - an edited protocol.json, a drifted panel digest, or an output file bound to a different
    identity stops the run before a model is loaded;
  - the budget ledger is consulted before loading and before every chunk.

Run on the GPU box (example):
  python scripts/run_panel.py --pair A --condition refusal_add_install --panel harmful_test \
      --stage harmful_generation --estimate-hours 0.35
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor" / "primary_snapshot"))
sys.path.insert(0, str(ROOT / "src"))

from protocol import CONDITIONS, LAYERS, RECOVERED_HARMFUL, load_protocol, tuned_condition  # noqa: E402
from records import ResponseStore, read_jsonl, response_record, sha256_text, utc  # noqa: E402

PANEL_FILES = {
    "harmful_test": "harmful_test_text.jsonl",
    "harmful_test_fallback100": "harmful_test_fallback100.jsonl",
    "development": "development_text.jsonl",
    "calibration": "calibration_text.jsonl",
    "xstest_safe": "xstest_safe_250.jsonl",
    "instruction_test": "instruction_test_100.jsonl",
    "instruction_fallback50": "instruction_fallback50.jsonl",
}
HARMFUL_DIGEST = "7851299f7e15743e47c0e4604697a0770b8e9607a9dd4420a0f31a3529a53953"
DECODING = {"mode": "greedy", "do_sample": False, "temperature": None, "top_p": None,
            "max_new_tokens": 768, "max_input_length": 2048, "stop_strings": ["### Instruction:"],
            "seed": 42, "batch_size_policy": "suggest_batch_size(model), as frozen (batch_size: null)"}
GATE = 0.05


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_panel(name: str) -> tuple[list[dict], str]:
    rows = list(read_jsonl(ROOT / "manifests" / "panels" / PANEL_FILES[name]))
    if name == "harmful_test_fallback100":
        text = {r["source_id"]: r for r in read_jsonl(ROOT / "manifests" / "panels" / "harmful_test_text.jsonl")}
        rows = [{**text[r["source_id"]]} for r in rows]
    digest = sha256_text("\n".join(f"{r['source_id']}:{r['prompt_hash']}" for r in rows))
    if name == "harmful_test" and digest != HARMFUL_DIGEST:
        raise SystemExit(f"HARD STOP: harmful panel digest drifted: {digest}")
    return rows, digest


CALIBRATION_PANELS = ("development", "calibration")
CAL_MULTIPLIERS = (0.5, 1.0, 1.5)


def resolve_condition(name: str, run_root: Path, panel: str | None = None, pair: str | None = None):
    if name.startswith("cal_refusal_add_"):
        # cal_refusal_add_{install|reverse}_m{0.5|1.0|1.5}: calibration screen only
        if panel not in CALIBRATION_PANELS:
            raise SystemExit(f"{name} is a calibration-screen condition; not allowed on {panel}")
        stem, mult = name.rsplit("_m", 1)
        if float(mult) not in CAL_MULTIPLIERS:
            raise SystemExit(f"{name}: multiplier {mult} is not in the frozen screen {CAL_MULTIPLIERS}")
        direction = "installation" if stem.endswith("install") else "reversal"
        c = tuned_condition(direction, float(mult))
        from dataclasses import replace
        return replace(c, name=name, note=f"calibration screen, multiplier {mult}")
    if name in CONDITIONS:
        return CONDITIONS[name]
    if name.startswith("refusal_add_tuned_"):
        sel = json.loads((run_root / "manifests" / "comparator_selection.json").read_text())
        direction = "installation" if name.endswith("install") else "reversal"
        entry = sel["tuned"].get(f"{pair}_{direction}", {})
        if entry.get("status") != "verified" or entry.get("multiplier") is None:
            raise SystemExit(f"{name}: calibration produced no usable tuned candidate; nothing to run")
        return tuned_condition(direction, float(entry["multiplier"]))
    raise SystemExit(f"unknown condition {name}")


def intervention_payload(cond, pair: str, run_root: Path) -> tuple[dict | None, object, dict]:
    """Returns (vectors, ablation_direction, identity) without touching a GPU."""
    import torch
    ident = cond.identity(pair)
    if cond.intervention == "none":
        return None, None, ident
    if cond.intervention == "dsteer_add":
        p = run_root / "outputs" / "directions" / f"dsteer_vectors_pair_{pair}.pt"
        vecs = torch.load(p, map_location="cpu", weights_only=True)
        ident["vector_file"], ident["vector_sha256"] = str(p.relative_to(run_root)), sha256_file(p)
    elif cond.intervention == "refusal_add":
        p = run_root / "outputs" / "directions" / pair / cond.side / "directions.pt"
        blob = torch.load(p, map_location="cpu", weights_only=True)
        vecs = {int(L): v for L, v in blob["matched"].items()}
        ident["vector_file"], ident["vector_sha256"] = str(p.relative_to(run_root)), sha256_file(p)
    elif cond.intervention == "refusal_ablate":
        p = run_root / "outputs" / "directions" / pair / "dpo" / "reference_selected.pt"
        blob = torch.load(p, map_location="cpu", weights_only=True)
        ident["direction_file"], ident["direction_sha256"] = str(p.relative_to(run_root)), sha256_file(p)
        ident["selected_pos"], ident["selected_layer"] = int(blob["pos"]), int(blob["layer"])
        return None, blob["direction"], ident
    else:
        raise SystemExit(f"unknown intervention {cond.intervention}")
    if sorted(vecs) != list(LAYERS):
        raise SystemExit(f"HARD STOP: vectors cover layers {sorted(vecs)}, expected {list(LAYERS)}")
    ident["vector_norms"] = {str(L): float(v.norm()) for L, v in vecs.items()}
    return vecs, None, ident


def verified_endpoint(run_root: Path, pair: str, side: str) -> dict:
    p = run_root / "manifests" / "endpoints.json"
    if not p.exists():
        raise SystemExit("manifests/endpoints.json missing; run scripts/materialise_endpoints.py first")
    ep = json.loads(p.read_text())["endpoints"].get(f"{pair}_{side}")
    if not ep or ep.get("status") != "verified":
        raise SystemExit(f"HARD STOP: endpoint {pair}_{side} not verified: {ep and ep.get('status')}")
    return ep


def make_context(model, cond, vectors, ablation_direction, coefficient):
    from steering.steer import ActivationSteering
    from hooks import reference_ablation
    if cond.intervention == "none":
        return None
    if cond.intervention == "refusal_ablate":
        return lambda: reference_ablation(model, ablation_direction)
    return lambda: ActivationSteering(model, vectors, coefficient=coefficient,
                                      positions="all", preserve_norm=False, mode="add")


def generate_into(store: ResponseStore, model, tokenizer, rows: list[dict], context, *, batch_size: int,
                  meta: dict, guard=None, chunk: int | None = None) -> dict:
    """Generate every pending row into `store`, chunk by chunk. Shared by the CLI and tests."""
    from steering.generate import build_chat_prompts, generate_batched
    from steering.validity import check
    todo = store.pending(rows)
    chunk = chunk or max(batch_size * 4, batch_size)
    stopped = None
    for i in range(0, len(todo), chunk):
        if guard is not None:
            st = guard.state()
            if not st.may_schedule:
                stopped = f"budget: {st.consumed_h:.2f}h >= scheduling stop"
                break
        part = todo[i:i + chunk]
        chats = build_chat_prompts(tokenizer, [r["prompt"] for r in part])
        started, t0 = utc(), time.monotonic()
        try:
            results = generate_batched(model, tokenizer, chats, max_new_tokens=DECODING["max_new_tokens"],
                                       batch_size=batch_size, max_input_length=DECODING["max_input_length"],
                                       context=context, desc=meta["condition"]["name"], return_metadata=True)
        except Exception as exc:  # noqa: BLE001 -- recorded as a failed attempt, never as a sample
            for r in part:
                store.record_failure({"source_id": r["source_id"], "error": f"{type(exc).__name__}: {exc}"})
            raise
        if len(results) != len(part):
            raise RuntimeError(f"{len(results)} results for {len(part)} prompts; refusing a truncated write")
        per = (time.monotonic() - t0) / len(part)
        for r, chat, res in zip(part, chats, results):
            v = check(res.text)
            n_in = len(tokenizer(chat, add_special_tokens=False, truncation=True,
                                 max_length=DECODING["max_input_length"])["input_ids"])
            store.append(response_record(
                run_id=meta["run_id"], protocol_hash=meta["protocol_hash"], panel=meta["panel"],
                pair=meta["pair"], condition=meta["condition"], endpoint=meta["endpoint"],
                decoding={**DECODING, "batch_size": batch_size}, prompt_row=r, result=res,
                prompt_tokens=n_in, started_utc=started, elapsed_s=per,
                validity={"ok": v.ok, "reason": v.reason,
                          "max_new_tokens_hit": res.stop_reason == "max_new_tokens"}))
    recs = store.records()
    n = len(recs)
    n_inv = sum(not r["validity"]["ok"] for r in recs)
    n_max = sum(r["validity"]["max_new_tokens_hit"] for r in recs)
    gates = {"n": n, "n_invalid": n_inv, "n_maxlen": n_max,
             "invalid_rate": n_inv / n if n else None, "maxlen_rate": n_max / n if n else None,
             "invalid_pass": n_inv <= GATE * n, "maxlen_pass": n_max <= GATE * n,
             "complete": n == len(rows), "stopped_early": stopped}
    store.finalise_meta(generation_gates=gates)
    return gates


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default=str(ROOT))
    ap.add_argument("--pair", required=True, choices=["A", "B"])
    ap.add_argument("--condition", required=True)
    ap.add_argument("--panel", required=True, choices=sorted(PANEL_FILES))
    ap.add_argument("--stage", required=True)
    ap.add_argument("--estimate-hours", type=float, required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N rows only (profiling)")
    ap.add_argument("--parity", action="store_true",
                    help="regenerate a recovered condition on N harmful prompts into outputs/parity/")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run_root = Path(a.run_root).resolve()

    protocol, protocol_hash = load_protocol()
    cond = resolve_condition(a.condition, run_root, a.panel, a.pair)
    rows, digest = load_panel(a.panel)
    recovered = a.panel.startswith("harmful_test") and (a.pair, cond.name) in RECOVERED_HARMFUL
    if recovered and not a.parity:
        raise SystemExit(f"REFUSED: {a.pair}/{cond.name} on {a.panel} is a recovered original "
                         f"({RECOVERED_HARMFUL[(a.pair, cond.name)]}); it is reused, never regenerated")
    if a.parity and not recovered:
        raise SystemExit("--parity only applies to recovered harmful-panel conditions")
    if a.parity and not a.limit:
        raise SystemExit("--parity requires --limit (a small check, not a regeneration)")
    if a.limit:
        rows = rows[:a.limit]

    vectors, abl, ident = intervention_payload(cond, a.pair, run_root)
    endpoint = verified_endpoint(run_root, a.pair, cond.side)
    out_dir = run_root / "outputs" / ("parity" if a.parity else "generations") / a.panel / a.pair
    if a.limit and not a.parity:
        out_dir = run_root / "outputs" / "profile" / a.panel / a.pair
    identity = {"protocol_hash": protocol_hash, "panel": a.panel, "panel_digest": digest, "pair": a.pair,
                "condition": ident, "endpoint": {k: endpoint[k] for k in ("repository", "revision", "subpath", "identity_sha256")},
                "decoding": DECODING, "limit": a.limit}
    print(f"plan: {a.pair}/{cond.name} on {a.panel} (n={len(rows)}) -> {out_dir}/{cond.name}.jsonl "
          f"| coefficient {cond.coefficient(a.pair):+.4f} | endpoint {endpoint['repository']}@{endpoint['revision'][:12]}")
    if a.dry_run:
        return 0

    from budget_guard import guard_for
    guard = guard_for(run_root)
    guard.require_schedulable(f"{a.stage}:{a.pair}:{cond.name}:{a.panel}", a.estimate_hours)

    import torch
    from steering.models import load_model, load_tokenizer
    from steering.generate import suggest_batch_size
    from steering.utils import set_all_seeds
    set_all_seeds(42)
    tok = load_tokenizer(endpoint["local_path"], subfolder="", trust_remote_code=False, local_files_only=True)
    model = load_model(endpoint["local_path"], subfolder="", trust_remote_code=False, local_files_only=True)
    batch = suggest_batch_size(model)
    store = ResponseStore(out_dir / f"{cond.name}.jsonl", identity)
    meta = {"run_id": f"{a.stage}-{uuid.uuid4().hex[:8]}", "protocol_hash": protocol_hash, "panel": a.panel,
            "pair": a.pair, "condition": ident, "endpoint": identity["endpoint"]}
    ctx = make_context(model, cond, vectors, abl, cond.coefficient(a.pair))
    gates = generate_into(store, model, tok, rows, ctx, batch_size=batch, meta=meta, guard=guard)
    print("GATES", json.dumps(gates))
    del model
    torch.cuda.empty_cache()
    return 0 if gates["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
