"""Find how far a given model can be steered before its generations fall apart.

A shared lambda grid is the wrong instrument. On HarmfulQA at lambda=0.6, tulu3 breaks 38%
of its generations while llama3-oh breaks 5% with quality still flat -- so the same number
sits past the cliff on one pair and nowhere near it on another, and any cross-model
comparison read off that grid is comparing different amounts of intervention.

This locates each model's own ceiling first, so the scored sweep can span the range that
actually exists. It needs no judge: degeneration is detectable locally by
`validity.check`, which makes the search cheap enough to run before every sweep.

    python scripts/lambda_range.py --model-config configs/models/llama3-oh.yaml \
        --eval-config configs/steer_sweep.yaml --side it --output-dir outputs/lambda_range

Search: double from the seed value until the break rate is exceeded, then bisect. Doubling
assumes only that degeneration grows with |lambda| -- more perturbation, more broken text
-- which is a property of the intervention and not of whatever we are trying to measure.
It deliberately assumes *nothing* about refusal, harmfulness or shift, none of which need
be monotone. That is why the ceiling is found on validity alone and the metrics are then
gridded across the whole clean interval rather than read at the ceiling.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from steering.artifacts import write_run_metadata
from steering.config import ModelConfig
from steering.generate import build_chat_prompts, generate_batched, suggest_batch_size
from steering.models import load_model, load_tokenizer
from steering.steer import ActivationSteering, build_vectors, random_vectors_like, steered_layers
from steering.utils import set_all_seeds, setup_logging
from steering.validity import check

sys.path.insert(0, str(Path(__file__).resolve().parent))
from steer_sweep import SteerSweepConfig, load_prompts  # noqa: E402


def break_rate(model, tokenizer, chats, coeff, vectors, cfg, batch_size, label,
               mode: str = "add") -> Dict:
    """Share of generations that fail the degeneracy screen at this coefficient."""
    ctx = None
    if coeff != 0.0:
        def ctx():
            return ActivationSteering(
                model, vectors, coefficient=coeff,
                positions=cfg.positions, preserve_norm=cfg.preserve_norm,
                mode=mode,
            )

    texts = generate_batched(
        model, tokenizer, chats,
        max_new_tokens=cfg.max_new_tokens, batch_size=batch_size,
        max_input_length=cfg.max_input_length, context=ctx, desc=label,
    )
    verdicts = [check(t) for t in texts]
    broken = sum(1 for v in verdicts if not v.ok)
    reasons: Dict[str, int] = {}
    for v in verdicts:
        if not v.ok:
            reasons[v.reason] = reasons.get(v.reason, 0) + 1
    return {
        "lambda": coeff,
        "n": len(texts),
        "broken": broken,
        "break_rate": broken / len(texts) if texts else 0.0,
        "reasons": reasons,
        "sample": texts[0][:200] if texts else "",
    }


def search(probe, seed_lambda: float, max_lambda: float, threshold: float,
           tolerance: float, log) -> Dict:
    """Double to bracket the ceiling, then bisect to `tolerance`.

    Returns the largest lambda whose break rate stays at or below `threshold`, together
    with every point evaluated -- the trace is worth keeping, since a model that breaks
    at 0.5 and a model that never breaks are different findings and the bare ceiling
    does not distinguish them.
    """
    trace: List[Dict] = []

    def rate(lam: float) -> float:
        hit = next((t for t in trace if abs(t["lambda"] - lam) < 1e-9), None)
        if hit is None:
            hit = probe(lam)
            trace.append(hit)
            log.info(f"lambda={lam:+.3f}  broken {hit['broken']}/{hit['n']} "
                     f"({hit['break_rate']:.1%})  {hit['reasons']}")
        return hit["break_rate"]

    lo = 0.0                       # known clean: the unsteered model
    hi: Optional[float] = None     # smallest lambda known to break

    lam = seed_lambda
    while lam <= max_lambda:
        if rate(lam) > threshold:
            hi = lam
            break
        lo = lam
        lam *= 2

    if hi is None:
        log.warning(f"no breakage up to {max_lambda:+.3f} -- ceiling is above the search range")
        return {"ceiling": lo, "bracketed": False, "trace": sorted(trace, key=lambda t: t["lambda"])}

    while hi - lo > tolerance:
        mid = (lo + hi) / 2
        if rate(mid) > threshold:
            hi = mid
        else:
            lo = mid

    return {"ceiling": lo, "bracketed": True, "trace": sorted(trace, key=lambda t: t["lambda"])}


def suggest_grid(ceiling: float, existing: Optional[List[float]] = None,
                 n_new: int = 3) -> List[float]:
    """Keep the magnitudes already generated, then extend to the ceiling.

    An evenly spaced grid over [0, ceiling] would land on none of the values already
    swept and scored, so it would re-buy generations we own and re-judge them, while
    spending most of its points below the region the search was run to find. Anchoring on
    what exists keeps the old sweep comparable and puts every new point where nothing has
    been measured.
    """
    kept = sorted(v for v in (existing or []) if v <= ceiling)
    top = kept[-1] if kept else 0.0
    # A ceiling that lands on an existing magnitude leaves nothing above it to explore,
    # and dividing the remainder anyway yields three copies of the same rounded value.
    if ceiling - top < 0.02:
        return kept
    step = (ceiling - top) / n_new
    grid = kept + [round(top + step * i, 3) for i in range(1, n_new + 1)]
    return sorted(set(grid))



def run_tag(side: str, args, extra: str = "") -> str:
    """Directory name for a run, distinguishing the vector it used.

    Generations resume by record id within their directory, so two runs that share a
    directory silently reuse each other's outputs. A learned vector and a mean-difference
    vector on the same checkpoint would otherwise collide, and the second would report the
    first's generations as its own -- which is exactly the comparison the frozen-vector
    experiment exists to make.
    """
    parts = [side, extra]
    # The vector's source is not otherwise in the path, so a run using a different
    # activations directory would land on an earlier run's generations and resume on
    # them -- reporting success while comparing a vector against itself.
    if getattr(args, "tag", None):
        parts.append("_" + args.tag)
    if getattr(args, "mode", "add") == "ablate":
        parts.append("_ablate")
    if getattr(args, "vectors", None):
        parts.append("_learned")
    if getattr(args, "layers", None):
        parts.append("_L" + args.layers.replace(",", "-"))
    if getattr(args, "hold_out", 0):
        parts.append(f"_ho{args.hold_out}")
    if getattr(args, "random_control", False):
        parts.append("_random")
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Locate a model's steering ceiling.")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--side", choices=["it", "dpo"], required=True,
                        help="Which checkpoint to steer.")
    parser.add_argument(
        "--sign", type=int, choices=[1, -1], default=None,
        help="Steering direction, overriding the default (+1 from IT, -1 from DPO). "
             "The two differ in checkpoint as well as direction, so the default pairing "
             "cannot tell a fragile direction from a fragile checkpoint. Setting this "
             "crossways -- -1 on IT, +1 on DPO -- separates them.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--random-control", action="store_true")
    parser.add_argument("--n-probe", type=int, default=50,
                        help="Prompts per probe. Only the break rate is needed, so this is "
                             "far smaller than a scored sweep.")
    parser.add_argument("--seed-lambda", type=float, default=0.4,
                        help="First value tried. 0.4 is clean on every pair measured so far.")
    parser.add_argument("--max-lambda", type=float, default=6.4)
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Break rate that counts as past the ceiling.")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="Stop bisecting once the bracket is this narrow.")
    parser.add_argument(
        "--mode", choices=["add", "ablate"], default="add",
        help="How the vector is applied. 'add' puts lambda*v into the stream; 'ablate' "
             "projects the component along v out, which is what the refusal-direction "
             "line does and is not a rescaling of addition.",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Appended to the run directory. Use it whenever the vector comes from a "
             "different activations source, so runs cannot resume on each other.",
    )
    parser.add_argument(
        "--vectors", default=None, metavar="VECTORS.PT",
        help="Use pre-built per-layer vectors instead of deriving them from activations, "
             "e.g. one trained by train_steering_vector.py. Overrides --layers.",
    )
    parser.add_argument(
        "--layers", default=None, metavar="L[,L...]",
        help="Steer exactly these layers, overriding layers_last_k. One layer at a time "
             "is how 'why only the final layers' gets an answer rather than an assertion; "
             "a shared scalar over five of them is not equal intervention, since the "
             "displacement grows with depth.",
    )
    parser.add_argument(
        "--hold-out", type=int, default=0,
        help="Drop this many leading samples when building the vector. Use the "
             "sweep's prompt count when the vector and the evaluation come from "
             "the same corpus, so the two sets are disjoint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_all_seeds(args.seed)
    model_cfg = ModelConfig.from_yaml(args.model_config)
    cfg = SteerSweepConfig.from_yaml(args.eval_config)

    sign = float(args.sign) if args.sign is not None else (1.0 if args.side == "it" else -1.0)
    crossed = "_crossed" if (args.sign is not None and sign != (1.0 if args.side == "it" else -1.0)) else ""
    tag = run_tag(args.side, args, crossed)
    out = Path(args.output_dir) / model_cfg.name / tag
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out / "lambda_range.log")
    write_run_metadata(out, config={
        "model": asdict(model_cfg), "side": args.side, "sign": sign, "n_probe": args.n_probe,
        "threshold": args.threshold, "tolerance": args.tolerance, "seed": args.seed,
    })

    records = load_prompts(cfg.prompt_source, args.n_probe, args.seed)
    if args.vectors:
        # A vector trained under the DPO loss, rather than read off a checkpoint pair.
        # It arrives already per-layer, so the layer set comes from the file.
        loaded = torch.load(Path(args.vectors), map_location="cpu")
        vectors = {int(k): v.float() for k, v in loaded.items()}
        layers = sorted(vectors)
        log.info(f"Loaded {len(vectors)} vectors from {args.vectors}; layers {layers}")
    else:
        if args.layers:
            layers = [int(x) for x in args.layers.split(",")]
            bad = [l for l in layers if not 0 <= l < model_cfg.num_layers]
            if bad:
                parser.error(f"layers {bad} outside 0..{model_cfg.num_layers - 1}")
        else:
            layers = steered_layers(model_cfg.num_layers, cfg.layers_last_k)
        acts = Path(cfg.activations_dir) / model_cfg.name / "activations.pt"
        if not acts.exists():
            raise FileNotFoundError(f"no activations at {acts}; run layer_profile first")
        vectors = build_vectors(acts, method=cfg.vector_method, layers=layers,
                                normalise=cfg.vector_normalise,
                                skip_first=args.hold_out)
    if args.random_control:
        vectors = random_vectors_like(vectors, seed=args.seed)

    which = model_cfg.it_model if args.side == "it" else model_cfg.dpo_model
    sub = (model_cfg.it_subfolder if args.side == "it" else model_cfg.dpo_subfolder) or ""
    tokenizer = load_tokenizer(model_cfg.tokenizer_id or which,
                               subfolder=model_cfg.tokenizer_subfolder or sub)
    model = load_model(which, subfolder=sub)
    chats = build_chat_prompts(tokenizer, [r["prompt"] for r in records])
    batch_size = cfg.batch_size or suggest_batch_size(model)

    log.info(f"{model_cfg.name} {tag}: {len(records)} probe prompts, "
             f"layers {layers[0]}-{layers[-1]}, threshold {args.threshold:.0%}")

    def probe(magnitude: float) -> Dict:
        return break_rate(model, tokenizer, chats, sign * magnitude, vectors, cfg,
                          batch_size, f"{model_cfg.name} {tag} lambda={sign * magnitude:+.2f}",
                          mode=args.mode)

    result = search(probe, args.seed_lambda, args.max_lambda, args.threshold,
                    args.tolerance, log)
    result["side"] = args.side
    result["sign"] = sign
    result["suggested_grid"] = [
        sign * v for v in suggest_grid(result["ceiling"], existing=list(cfg.lambdas))
    ]

    (out / "range.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info(f"ceiling |lambda| = {result['ceiling']:.3f} "
             f"({'bracketed' if result['bracketed'] else 'not reached'})")
    log.info(f"suggested grid: {result['suggested_grid']}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
