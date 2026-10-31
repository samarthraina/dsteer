#!/usr/bin/env python3
"""Build refusal directions on the endpoint each comparator intervenes on.

Per (pair, side):
  matched     256 harmful (HarmfulQA construction) vs 256 harmless (reference Alpaca-derived
              train split), D-STEER readout, every layer; harmful-minus-harmless mean; layers
              27-31 rescaled to D-STEER's per-layer norm. Records raw and scaled norms and
              cosines against D-STEER at every layer.
  candidates  (DPO side only) reference readout: block inputs at every end-of-instruction
              position, after the reference's refusal-score filter on the train prompts.

Panels and the selection pools are frozen manifests; nothing here reads a test panel.

Run:  python scripts/build_refusal_directions.py --pair A --side it --stage refusal_construction --estimate-hours 0.05
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

from protocol import LAYERS, load_protocol  # noqa: E402
from records import read_jsonl  # noqa: E402


def prompts(name: str) -> list[dict]:
    return list(read_jsonl(ROOT / "manifests" / "panels" / name))


def family(tokenizer) -> str:
    return "llama3" if len(tokenizer) > 100_000 else "chatml"


def build(model, tokenizer, pair: str, side: str, out_dir: Path, dsteer_vectors: dict,
          dsteer_raw_all, with_candidates: bool) -> dict:
    import torch
    from refusal import eoi_positions, match_norms, mean_diff, refusal_token_ids
    from refusal_runtime import extract_prompt_last, mean_at_eoi_inputs, scores

    harmful = prompts("construction_harmful_256.jsonl")
    harmless = prompts("construction_harmless_256.jsonl")
    hp, lp = [r["prompt"] for r in harmful], [r["prompt"] for r in harmless]

    a_h = extract_prompt_last(model, tokenizer, hp)
    a_l = extract_prompt_last(model, tokenizer, lp)
    raw = mean_diff(a_h, a_l)                                     # [n_layers, d]
    matched, rows = match_norms({L: raw[L] for L in LAYERS}, dsteer_vectors)

    geo = []
    for L in range(raw.shape[0]):
        cos = torch.nn.functional.cosine_similarity(raw[L], dsteer_raw_all[L].double(), dim=0).item()
        geo.append({"pair": pair, "endpoint": side, "layer": L, "refusal_raw_norm": raw[L].norm().item(),
                    "dsteer_raw_norm": dsteer_raw_all[L].norm().item(), "cosine_refusal_vs_dsteer": cos,
                    "steered_layer": L in LAYERS})

    blob = {"raw_all_layers": raw.float(), "matched": {L: matched[L] for L in LAYERS},
            "layers": list(LAYERS), "n_harmful": len(hp), "n_harmless": len(lp)}
    summary = {"pair": pair, "side": side, "matched_layers": [r.__dict__ for r in rows]}

    if with_candidates:
        toks = refusal_token_ids(tokenizer, family(tokenizer))
        pos = eoi_positions(tokenizer)
        s_h, s_l = scores(model, tokenizer, hp, toks), scores(model, tokenizer, lp, toks)
        keep_h = [p for p, s in zip(hp, s_h.tolist()) if s > 0]   # reference filter_train
        keep_l = [p for p, s in zip(lp, s_l.tolist()) if s < 0]
        cand = mean_at_eoi_inputs(model, tokenizer, keep_h, pos) - mean_at_eoi_inputs(model, tokenizer, keep_l, pos)
        blob["candidates"] = cand.float()                           # [n_pos, n_layers, d]
        blob["eoi_positions"] = pos
        blob["refusal_toks"] = toks
        summary["reference_filter"] = {"harmful_kept": len(keep_h), "harmful_dropped": len(hp) - len(keep_h),
                                       "harmless_kept": len(keep_l), "harmless_dropped": len(lp) - len(keep_l),
                                       "refusal_toks": toks, "eoi_positions": pos}
        if not keep_h or not keep_l:
            summary["reference_filter"]["blocked"] = "a class is empty after filtering; candidates undefined"

    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "directions.pt"
    torch.save(blob, p)
    summary["directions_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    (out_dir / "directions_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out_dir / "geometry.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(geo[0]))
        w.writeheader(); w.writerows(geo)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default=str(ROOT))
    ap.add_argument("--pair", required=True, choices=["A", "B"])
    ap.add_argument("--side", required=True, choices=["it", "dpo"])
    ap.add_argument("--stage", required=True)
    ap.add_argument("--estimate-hours", type=float, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run_root = Path(a.run_root).resolve()
    _, phash = load_protocol()
    import torch
    dv = torch.load(run_root / "outputs/directions" / f"dsteer_vectors_pair_{a.pair}.pt", weights_only=True)
    dr = torch.load(run_root / "outputs/directions" / f"dsteer_raw_meandiff_all_layers_pair_{a.pair}.pt", weights_only=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_panel import verified_endpoint
    ep = verified_endpoint(run_root, a.pair, a.side)
    print(f"plan: refusal directions on {a.pair}_{a.side} ({ep['repository']}@{ep['revision'][:12]}), "
          f"candidates={'yes' if a.side == 'dpo' else 'no'}")
    if a.dry_run:
        return 0
    from budget_guard import guard_for
    guard_for(run_root).require_schedulable(
        f"{a.stage}:{a.pair}:{a.side}", a.estimate_hours)
    from steering.models import load_model, load_tokenizer
    from steering.utils import set_all_seeds
    set_all_seeds(42)
    tok = load_tokenizer(ep["local_path"], trust_remote_code=False, local_files_only=True)
    model = load_model(ep["local_path"], trust_remote_code=False, local_files_only=True).eval()
    s = build(model, tok, a.pair, a.side, run_root / "outputs/directions" / a.pair / a.side, dv, dr,
              with_candidates=(a.side == "dpo"))
    s["protocol_hash"] = phash
    print(json.dumps(s, indent=1)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
