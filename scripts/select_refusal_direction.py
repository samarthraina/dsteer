"""Choose the refusal direction the way the interpretability line does, on validation data.

Our first attempt took the mean difference over the same five layers the checkpoint vector
uses, and added it. That is our recipe applied to their contrast, not their method, and it
under-builds their side of the comparison -- which matters when the conclusion is that the
checkpoint difference wins.

What they actually do, and this reproduces:

  - a candidate direction at *every* layer, not a fixed late block
  - the chosen direction applied by **ablation at all layers**, so the model cannot
    represent it anywhere, rather than added at a few
  - the choice made on a **validation split** disjoint from both the prompts the
    directions were built from and the evaluation set

This writes one generation file per candidate into a single directory, so `score_sweep.py`
judges them all in one pass and the winner is picked on measured refusal rather than on a
keyword rule.

    python scripts/select_refusal_direction.py --model-config configs/llama3_oh_local.yaml \
        --activations outputs/refusal_direction/llama3-oh/activations.pt \
        --output-dir outputs/refusal_select --n-val 100 --val-offset 812
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from steering.artifacts import write_run_metadata
from steering.config import ModelConfig
from steering.data import load_harmfulqa
from steering.generate import build_chat_prompts, generate_batched, suggest_batch_size
from steering.models import load_model, load_tokenizer
from steering.steer import ActivationSteering, build_vectors
from steering.utils import append_jsonl, read_jsonl, set_all_seeds, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Select a refusal direction on validation data.")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--activations", required=True,
                        help="From refusal_direction.py: it=harmless, dpo=harmful.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--side", choices=["it", "dpo"], default="dpo")
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--val-offset", type=int, default=812,
                        help="Skip this many HarmfulQA prompts. Must clear both the "
                             "evaluation set and the prompts the directions come from.")
    parser.add_argument("--coefficient", type=float, default=1.0,
                        help="Share of the component removed. 1.0 ablates it entirely.")
    parser.add_argument("--layer-step", type=int, default=1,
                        help="Stride over candidate layers, to trade cost for resolution.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_all_seeds(args.seed)
    model_cfg = ModelConfig.from_yaml(args.model_config)
    out = Path(args.output_dir) / model_cfg.name
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out / "select_refusal_direction.log")
    write_run_metadata(out, config={"model": asdict(model_cfg), **vars(args)})

    records = load_harmfulqa(n=args.val_offset + args.n_val, seed=args.seed)[args.val_offset:]
    log.info(f"{len(records)} validation prompts, offset {args.val_offset}")

    which = model_cfg.it_model if args.side == "it" else model_cfg.dpo_model
    sub = (model_cfg.it_subfolder if args.side == "it" else model_cfg.dpo_subfolder) or ""
    tokenizer = load_tokenizer(model_cfg.tokenizer_id or which,
                               subfolder=model_cfg.tokenizer_subfolder or sub)
    model = load_model(which, subfolder=sub)
    chats = build_chat_prompts(tokenizer, [r["prompt"] for r in records])
    batch_size = args.batch_size if getattr(args, "batch_size", None) else suggest_batch_size(model)

    n_layers = model_cfg.num_layers
    candidates = list(range(0, n_layers, args.layer_step))
    all_layers = list(range(n_layers))
    log.info(f"{len(candidates)} candidate directions, ablated across all {n_layers} layers")

    # Unsteered reference, so the selection has something to improve on.
    settings = [("baseline", None)] + [(f"cand_L{l:02d}", l) for l in candidates]

    for name, layer in settings:
        path = out / f"{name}.jsonl"
        done = {r["id"] for r in read_jsonl(path)} if path.exists() else set()
        todo = [r for r in records if r["id"] not in done]
        if not todo:
            continue
        idx = {r["id"]: i for i, r in enumerate(records)}
        todo_chats = [chats[idx[r["id"]]] for r in todo]

        ctx = None
        if layer is not None:
            # One direction, taken from this layer, removed at every layer.
            v = build_vectors(Path(args.activations), method="mean", layers=[layer])[layer]
            vectors = {l: v for l in all_layers}

            def ctx(vectors=vectors):
                return ActivationSteering(model, vectors, coefficient=args.coefficient,
                                          mode="ablate")

        texts = generate_batched(model, tokenizer, todo_chats,
                                 max_new_tokens=args.max_new_tokens, batch_size=batch_size,
                                 max_input_length=args.max_input_length, context=ctx, desc=name)
        for rec, text in zip(todo, texts):
            append_jsonl({**rec, "prompt_str": str(rec["prompt"]), "response": text,
                          "lambda": 0.0 if layer is None else float(layer),
                          "candidate_layer": layer}, path)
        log.info(f"{name}: {len(todo)} generations")

    log.info(f"Wrote {len(settings)} files to {out}. Score that directory, then pick the "
             f"candidate with the lowest refusal.")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
