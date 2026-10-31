#!/usr/bin/env python3
"""Reference candidate selection for refusal_ablate_reference, on the DPO endpoint.

Mirrors andyrdt/refusal_direction @ 0b9c8bc pipeline/submodules/select_direction.py:
  bypass  mean refusal score on harmful selection prompts under all-site ablation
  induce  mean refusal score on harmless selection prompts with the raw candidate added at
          its source block input, coefficient 1.0
  kl      mean last-position KL on harmless selection prompts under ablation
  filter  last 20% of layers, KL > 0.1, induce < 0; choose minimum bypass
Selection data: development partition (60 harmful) + selection_harmless_60, both filtered on
refusal-score sign as the reference filters its validation split. Never a test panel, never
the independent judge, never a generated response.

Pruned-layer candidates are recorded but not scored: the reference computes their scores and
then discards them unconditionally, so skipping them cannot change the selection.

Run:  python scripts/select_refusal_reference.py --pair A --stage refusal_selection --estimate-hours 0.2
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor" / "primary_snapshot"))
sys.path.insert(0, str(ROOT / "src"))

from protocol import load_protocol  # noqa: E402
from records import read_jsonl  # noqa: E402


def run_selection(model, tokenizer, blob: dict, harmful: list[str], harmless: list[str]) -> dict:
    import torch
    from hooks import addition_at_block_input, reference_ablation
    from refusal import Candidate, kl_last_position, pruned, refusal_score, select
    from refusal_runtime import last_logits, scores

    toks, cand = blob["refusal_toks"], blob["candidates"]
    n_pos, n_layers, _ = cand.shape
    s_h, s_l = scores(model, tokenizer, harmful, toks), scores(model, tokenizer, harmless, toks)
    harmful = [p for p, s in zip(harmful, s_h.tolist()) if s > 0]       # reference filter_val
    harmless = [p for p, s in zip(harmless, s_l.tolist()) if s < 0]
    if not harmful or not harmless:
        return {"selected": None, "blocked": "a selection class is empty after filtering",
                "n_harmful": len(harmful), "n_harmless": len(harmless), "candidates": []}
    base_l = last_logits(model, tokenizer, harmless)

    cands = []
    for pi in range(n_pos):
        for L in range(n_layers):
            d = cand[pi, L]
            if pruned(L, n_layers):
                cands.append(Candidate(pi, L, float("nan"), float("nan"), float("nan")))
                continue
            abl = lambda d=d: reference_ablation(model, d)
            bypass = refusal_score(last_logits(model, tokenizer, harmful, abl), toks).mean().item()
            kl = kl_last_position(base_l, last_logits(model, tokenizer, harmless, abl)).mean().item()
            add = lambda d=d, L=L: addition_at_block_input(model, L, d, 1.0)
            induce = refusal_score(last_logits(model, tokenizer, harmless, add), toks).mean().item()
            cands.append(Candidate(pi, L, bypass, induce, kl))
    best, all_c = select(cands, n_layers)
    return {"selected": None if best is None else {"pos": best.pos, "layer": best.layer,
                                                   "bypass": best.bypass, "induce": best.induce, "kl": best.kl},
            "baseline_harmful_refusal_score": s_h.mean().item(),
            "baseline_harmless_refusal_score": s_l.mean().item(),
            "n_harmful": len(harmful), "n_harmless": len(harmless),
            "candidates": [c.__dict__ for c in all_c]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default=str(ROOT))
    ap.add_argument("--pair", required=True, choices=["A", "B"])
    ap.add_argument("--stage", required=True)
    ap.add_argument("--estimate-hours", type=float, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run_root = Path(a.run_root).resolve()
    _, phash = load_protocol()
    d = run_root / "outputs" / "directions" / a.pair / "dpo"
    harmful = [r["prompt"] for r in read_jsonl(run_root / "manifests/panels/development_text.jsonl")]
    harmless = [r["prompt"] for r in read_jsonl(run_root / "manifests/panels/selection_harmless_60.jsonl")]
    print(f"plan: reference selection on {a.pair}_dpo, {len(harmful)} harmful dev + {len(harmless)} harmless selection")
    if a.dry_run:
        return 0
    import torch
    blob = torch.load(d / "directions.pt", weights_only=False)
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_panel import verified_endpoint
    from budget_guard import guard_for
    guard_for(run_root).require_schedulable(f"{a.stage}:{a.pair}", a.estimate_hours)
    ep = verified_endpoint(run_root, a.pair, "dpo")
    from steering.models import load_model, load_tokenizer
    from steering.utils import set_all_seeds
    set_all_seeds(42)
    tok = load_tokenizer(ep["local_path"], trust_remote_code=False, local_files_only=True)
    model = load_model(ep["local_path"], trust_remote_code=False, local_files_only=True).eval()
    res = run_selection(model, tok, blob, harmful, harmless)
    res.update({"pair": a.pair, "protocol_hash": phash, "reference_commit": "0b9c8bc56a73b42f595fd173156d4aefe9a9143c"})
    with (d / "reference_candidates.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pos", "layer", "bypass", "induce", "kl", "filtered_reason"])
        w.writeheader(); w.writerows(res["candidates"])
    if res["selected"]:
        s = res["selected"]
        p = d / "reference_selected.pt"
        torch.save({"pos": s["pos"], "layer": s["layer"],
                    "direction": blob["candidates"][s["pos"], s["layer"]]}, p)
        res["selected"]["direction_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    res.pop("candidates")
    (d / "reference_selection.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))
    # a blocked selection is a finding about the reference method on this endpoint, recorded in
    # reference_selection.json; it removes only the optional refusal_ablate_reference arm for this
    # pair (run_plan skips it), so it does not halt the mandatory matched comparator
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
