"""Build a refusal direction from prompt contrasts on a single checkpoint.

The vector used everywhere else here is a *checkpoint* difference: the same prompt run
through the model before and after preference optimisation. This builds the other kind,
the one the interpretability line uses — harmful prompts against harmless ones, on one
model, no pair required.

Two reasons to want it. It is the realistic threat model, since most releases publish
only the aligned model. And where the checkpoint-difference vector saturates well short
of removing the safety a preference stage installed, a refusal direction is reported to
jailbreak nearly completely — which can only be true if the two directions differ. This
measures the angle between them, and puts both through the same ceiling and transfer
procedure so the reach is compared on equal terms.

Saved in the layout `layer_profile.py` writes, so `lambda_range.py` and `steer_sweep.py`
read it with no changes: `it` holds the harmless activations and `dpo` the harmful ones,
so the mean difference is (harmful - harmless), pointing toward refusal. Steering with
-lambda therefore suppresses refusal, matching the sign convention of the DPO vector,
where -lambda undoes the preference stage.

    python scripts/refusal_direction.py --model-config configs/llama3_oh_local.yaml \
        --side dpo --n 512 --output-dir outputs/refusal_direction

Then point a sweep config's `activations_dir` at the output and run the usual pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from steering.artifacts import write_run_metadata
from steering.config import ModelConfig
from steering.data import load_alpacaeval, load_harmfulqa
from steering.models import load_tokenizer
from steering.steer import build_vectors, steered_layers
from steering.utils import set_all_seeds, setup_logging

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer_profile import LayerProfileConfig, extract_activations  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Refusal direction by prompt contrast, and its angle to the DPO direction.")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--side", choices=["it", "dpo"], default="dpo",
                        help="Which checkpoint to read the direction off. The aligned one "
                             "is the realistic case: it is what gets published.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n", type=int, default=512,
                        help="Prompts per side. The mean converges by ~500 (see the "
                             "convergence check), so more buys little.")
    parser.add_argument("--hold-out", type=int, default=0,
                        help="Leading harmful prompts to exclude, so the direction is not "
                             "built from the prompts a sweep will be scored on. Set this "
                             "to the sweep's prompt count.")
    parser.add_argument("--compare-to", default=None, metavar="ACTIVATIONS.PT",
                        help="Checkpoint-difference activations to measure the angle against.")
    parser.add_argument("--layers-last-k", type=int, default=5)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_all_seeds(args.seed)
    model_cfg = ModelConfig.from_yaml(args.model_config)
    out = Path(args.output_dir) / model_cfg.name
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out / "refusal_direction.log")
    write_run_metadata(out, config={
        "model": asdict(model_cfg), "side": args.side, "n": args.n,
        "hold_out": args.hold_out, "seed": args.seed,
    })

    # Harmful prompts share the loader and seed with the sweeps, so --hold-out lines up
    # with the evaluation set. Harmless ones are general instructions: the contrast has to
    # isolate refusal, not topic, so they come from a different corpus on purpose.
    harmful = load_harmfulqa(n=args.n + args.hold_out, seed=args.seed)[args.hold_out:]
    harmless = load_alpacaeval(n=args.n, seed=args.seed)
    n = min(len(harmful), len(harmless))
    harmful, harmless = harmful[:n], harmless[:n]
    log.info(f"{n} prompts per side (harmful held out: {args.hold_out})")

    which = model_cfg.it_model if args.side == "it" else model_cfg.dpo_model
    sub = (model_cfg.it_subfolder if args.side == "it" else model_cfg.dpo_subfolder) or ""
    tokenizer = load_tokenizer(model_cfg.tokenizer_id or which,
                               subfolder=model_cfg.tokenizer_subfolder or sub)

    # Prompt token, because these prompts have no reference response to read after.
    cfg = LayerProfileConfig(token_position="prompt_last",
                             max_input_length=args.max_input_length)

    acts = {}
    for label, prompts in [("harmless", harmless), ("harmful", harmful)]:
        acts[label] = extract_activations(
            which, sub, tokenizer, prompts, cfg,
            checkpoint_path=out / f"_{label}.partial.pt",
        )
        log.info(f"{label}: {tuple(acts[label].shape)}")

    torch.save({"it": acts["harmless"], "dpo": acts["harmful"]}, out / "activations.pt")
    log.info(f"Saved to {out}/activations.pt (it=harmless, dpo=harmful)")

    summary = {"model": model_cfg.name, "side": args.side, "n": n, "hold_out": args.hold_out}

    if args.compare_to:
        layers = steered_layers(model_cfg.num_layers, args.layers_last_k)
        refusal = build_vectors(out / "activations.pt", method="mean", layers=layers)
        dpo_dir = build_vectors(Path(args.compare_to), method="mean", layers=layers,
                                skip_first=args.hold_out)
        rows = []
        for layer in layers:
            a, b = refusal[layer], dpo_dir[layer]
            rows.append({
                "layer": layer,
                "cos": (a / a.norm() @ (b / b.norm())).item(),
                "norm_refusal": a.norm().item(),
                "norm_dpo": b.norm().item(),
            })
        summary["angle_to_dpo_direction"] = rows
        log.info("\nlayer   cos(refusal, dpo)   |v_refusal|   |v_dpo|")
        for r in rows:
            log.info(f"{r['layer']:5d}   {r['cos']:+.4f}          "
                     f"{r['norm_refusal']:9.3f}   {r['norm_dpo']:8.3f}")
        mean_cos = sum(r["cos"] for r in rows) / len(rows)
        log.info(f"mean cosine over the steered layers: {mean_cos:+.4f}")
        log.info("Near zero means preference optimisation does not move the model along "
                 "its refusal direction, which would explain a reach that stops short.")

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
