"""Generate from a single checkpoint, to see whether it is worth training a pair from.

A preference stage can only install safety a checkpoint does not already have. Two of the
three pairs measured here have no safety difference between their checkpoints at all,
because their instruction-tuning mix already covered it -- so the expensive mistake is
training a DPO stage onto a base with no headroom and discovering it afterwards.

This writes generations in the layout `score_sweep.py` reads, so screening is: run this,
then score that directory. What to look for, against the one checkpoint here that did
support a usable pair (refusal 0.241, harmfulness 0.339, quality 0.749):

  low refusal      room for a safety stage to change something
  high harmfulness the headroom itself
  decent quality   without which low harmfulness is incoherence, not safety

    python scripts/screen_model.py --model teknium/OpenHermes-2.5-Mistral-7B \
        --output-dir outputs/screen/openhermes-mistral --n 150
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from steering.artifacts import write_run_metadata
from steering.data import load_harmfulqa
from steering.generate import build_chat_prompts, generate_batched, suggest_batch_size
from steering.models import load_model, load_tokenizer
from steering.utils import append_jsonl, read_jsonl, set_all_seeds, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Generate from one checkpoint for screening.")
    parser.add_argument("--model", required=True, help="HF id or local path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-input-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_all_seeds(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out / "screen_model.log")
    write_run_metadata(out, config={"model": args.model, "n": args.n, "seed": args.seed})

    # The same seed and loader as the sweeps, so a screened checkpoint is being asked the
    # same questions as everything else here.
    records = load_harmfulqa(n=args.n, seed=args.seed)
    path = out / "baseline.jsonl"
    done = {r["id"] for r in read_jsonl(path)} if path.exists() else set()
    todo = [r for r in records if r["id"] not in done]
    if not todo:
        log.info(f"{path} already complete")
        return

    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model)
    chats = build_chat_prompts(tokenizer, [r["prompt"] for r in todo])
    batch_size = args.batch_size or suggest_batch_size(model)
    log.info(f"{args.model}: {len(todo)} prompts, batch {batch_size}")

    texts = generate_batched(
        model, tokenizer, chats,
        max_new_tokens=args.max_new_tokens, batch_size=batch_size,
        max_input_length=args.max_input_length, desc=Path(args.model).name,
    )
    for rec, text in zip(todo, texts):
        append_jsonl({**rec, "prompt_str": str(rec["prompt"]),
                      "response": text, "lambda": 0.0, "model": args.model}, path)

    # Fail loudly. A tokenizer that will not load, or a download that half-finished,
    # otherwise leaves an empty directory that the scoring stage happily reads as zero
    # records -- which is how a pipeline reports a screen it never ran.
    written = sum(1 for _ in read_jsonl(path))
    if written < len(records):
        raise RuntimeError(
            f"{path} has {written} of {len(records)} generations; refusing to continue"
        )

    log.info(f"wrote {len(todo)} generations to {path}")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
