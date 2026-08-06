"""Instruction-following capability evaluation.

Evaluates BOTH the IT and DPO checkpoints on:
1. IFEval — rule-based verifiable-instruction scoring (no judge needed)
2. AlpacaEval prompts — quality scoring via the judge

Output: a single CSV per model pair summarizing performance, and per-prompt
JSONL files with raw generations.

Usage:
    # Start the judge server first in a separate terminal:
    python scripts/start_judge_server.py

    # Then run the eval:
    python scripts/it_eval.py --model-config configs/llama3_oh.yaml --eval-config configs/it_eval.yaml

Dependencies:
    Requires the official IFEval scorer:
        pip install git+https://github.com/google-research/google-research.git#subdirectory=instruction_following_eval

    The script exits with an error if this is not installed. We do NOT provide
    a simplified fallback because partial IFEval scoring would silently produce
    non-comparable numbers.
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm

from steering.config import ITEvalConfig, ModelConfig
from steering.data import load_alpacaeval, load_ifeval
from steering.judge import Judge
from steering.metrics import score_quality
from steering.models import load_model, load_tokenizer
from steering.utils import append_jsonl, read_jsonl, set_all_seeds, setup_logging, write_jsonl


def main():
    parser = argparse.ArgumentParser(description="Run IT capability evaluation.")
    parser.add_argument("--model-config", required=True, help="Path to model YAML config")
    parser.add_argument("--eval-config", required=True, help="Path to IT-eval YAML config")
    parser.add_argument(
        "--phase",
        choices=["generate", "score", "both"],
        default="both",
        help="generate: only generate responses (no judge/server or IFEval scorer needed, generator owns the GPU); "
             "score: only score existing generations (no generator loaded, judge owns the GPU); "
             "both: full run (default; behavior unchanged from before this flag existed).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_all_seeds(args.seed)

    model_cfg = ModelConfig.from_yaml(args.model_config)
    eval_cfg = ITEvalConfig.from_yaml(args.eval_config)

    output_root = Path(eval_cfg.output_dir) / model_cfg.name
    output_root.mkdir(parents=True, exist_ok=True)

    log = setup_logging(output_root / "it_eval.log")
    log.info(f"Model pair: {model_cfg.name}")
    log.info(f"Output directory: {output_root}")
    log.info(f"Phase: {args.phase}")

    do_generate = args.phase in ("generate", "both")
    do_score = args.phase in ("score", "both")

    # The official IFEval scorer is only needed for scoring; verify it before doing
    # anything expensive (fail-fast preserved for --phase both and score).
    if do_score:
        require_official_ifeval()

    checkpoints = [
        ("it", model_cfg.it_model, model_cfg.it_subfolder or ""),
        ("dpo", model_cfg.dpo_model, model_cfg.dpo_subfolder or ""),
    ]

    # Prompts + tokenizer are only needed when generating.
    tokenizer = None
    ifeval_prompts = None
    alpaca_prompts = None
    if do_generate:
        # Load prompts (same prompts for IT and DPO so results are comparable).
        ifeval_prompts = load_ifeval(n=eval_cfg.ifeval_n)
        alpaca_prompts = load_alpacaeval(n=eval_cfg.alpaca_n)
        tokenizer_path = model_cfg.tokenizer_id or model_cfg.it_model
        tokenizer_subfolder = model_cfg.tokenizer_subfolder or (model_cfg.it_subfolder or "")
        tokenizer = load_tokenizer(tokenizer_path, subfolder=tokenizer_subfolder)

    # The judge (vLLM HTTP client) is only needed for AlpacaEval quality scoring.
    judge = Judge(eval_cfg.judge) if do_score else None

    # Process each checkpoint: generate (if requested) and free the model, then score (if requested).
    # For --phase both this is the original per-checkpoint order.
    summary_rows = []
    for checkpoint_label, model_path, subfolder in checkpoints:
        ifeval_path = output_root / f"ifeval_{checkpoint_label}.jsonl"
        alpaca_path = output_root / f"alpaca_{checkpoint_label}.jsonl"

        if do_generate:
            log.info(f"\n{'=' * 60}\nGenerating {checkpoint_label.upper()}: {model_path} (subfolder={subfolder!r})\n{'=' * 60}")
            model = load_model(model_path, subfolder=subfolder)
            generate_for_prompts(
                model, tokenizer, ifeval_prompts, eval_cfg, output_path=ifeval_path, label="IFEval",
            )
            generate_for_prompts(
                model, tokenizer, alpaca_prompts, eval_cfg, output_path=alpaca_path, label="AlpacaEval",
            )
            # Free the model before scoring so the judge can own the GPU.
            del model
            gc.collect()
            torch.cuda.empty_cache()

        if do_score:
            log.info(f"\n{'=' * 60}\nScoring {checkpoint_label.upper()}\n{'=' * 60}")
            if not ifeval_path.exists() or not alpaca_path.exists():
                log.warning(
                    f"  {checkpoint_label}: missing generations "
                    f"(ifeval={ifeval_path.exists()}, alpaca={alpaca_path.exists()}), skipping"
                )
                continue
            ifeval_records = read_jsonl(ifeval_path)
            alpaca_records = read_jsonl(alpaca_path)

            # IFEval scoring (rule-based, no judge).
            ifeval_summary = score_ifeval(ifeval_records)
            log.info(f"IFEval pass-rate ({checkpoint_label}): "
                     f"prompt-strict={ifeval_summary['prompt_level_strict']:.3f}, "
                     f"prompt-loose={ifeval_summary['prompt_level_loose']:.3f}, "
                     f"inst-strict={ifeval_summary['inst_level_strict']:.3f}, "
                     f"inst-loose={ifeval_summary['inst_level_loose']:.3f}")

            # Judge-based quality scoring on AlpacaEval.
            alpaca_scored = score_outputs_with_judge(judge, alpaca_records)
            write_jsonl(alpaca_scored, output_root / f"alpaca_{checkpoint_label}_scored.jsonl")
            quality_scores = [r["quality_score"] for r in alpaca_scored if r.get("quality_score") is not None]
            mean_quality = sum(quality_scores) / len(quality_scores) if quality_scores else float("nan")
            log.info(f"Mean quality ({checkpoint_label}): {mean_quality:.3f}")

            summary_rows.append({
                "checkpoint": checkpoint_label,
                "ifeval_prompt_strict": ifeval_summary["prompt_level_strict"],
                "ifeval_prompt_loose": ifeval_summary["prompt_level_loose"],
                "ifeval_inst_strict": ifeval_summary["inst_level_strict"],
                "ifeval_inst_loose": ifeval_summary["inst_level_loose"],
                "mean_quality": mean_quality,
                "n_ifeval": len(ifeval_records),
                "n_alpaca": len(alpaca_records),
                "n_quality_scored": len(quality_scores),
            })

    # Save summary CSV (scoring phases only).
    if do_score:
        if not summary_rows:
            log.warning("No scored results to summarize (no generations found).")
        else:
            summary_path = output_root / "summary.csv"
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
            log.info(f"\nSummary saved to {summary_path}")
            log.info("\nResults:")
            log.info("\n" + pd.DataFrame(summary_rows).to_string(index=False))
    elif do_generate:
        log.info("\nGeneration phase complete. Run with --phase score (judge server up) to score.")


# Dependency check


def require_official_ifeval():
    """Hard requirement: the official IFEval scorer must be importable.

    We refuse to run with a simplified fallback because partial scoring would
    silently produce numbers that don't match the published IFEval benchmark.
    """
    try:
        from instruction_following_eval import evaluation_main  # noqa: F401
    except ImportError:
        print(
            "ERROR: The official IFEval scorer is required but not installed.\n"
            "\n"
            "Install it with:\n"
            "  pip install git+https://github.com/google-research/google-research.git"
            "#subdirectory=instruction_following_eval\n"
            "\n"
            "We do not provide a fallback scorer because it would silently produce\n"
            "non-comparable IFEval numbers.",
            file=sys.stderr,
        )
        sys.exit(1)


# Generation


def generate_for_prompts(
    model,
    tokenizer,
    prompts: List[Dict],
    cfg: ITEvalConfig,
    output_path: Path,
    label: str,
) -> List[Dict]:
    """Generate responses for each prompt and save to JSONL.

    Resume-safe: if output_path already exists, skip prompts whose ids are already present.
    """
    existing_ids = set()
    if output_path.exists():
        for rec in read_jsonl(output_path):
            existing_ids.add(rec["id"])
        if existing_ids:
            logging.info(f"{label}: resuming, {len(existing_ids)} already done")

    pending = [p for p in prompts if p["id"] not in existing_ids]
    device = next(model.parameters()).device

    for prompt_rec in tqdm(pending, desc=f"Generating {label}", unit="prompt"):
        messages = [{"role": "user", "content": prompt_rec["prompt"]}]
        chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat, return_tensors="pt", truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
                temperature=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = out[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        record = dict(prompt_rec)
        record["response"] = response
        append_jsonl(record, output_path)

    # Reload combined records (existing + new) so downstream scoring sees everything.
    return read_jsonl(output_path)


# IFEval rule-based scoring


def score_ifeval(records: List[Dict]) -> Dict[str, float]:
    """Score IFEval responses using Google's official scorer.

    Returns four pass rates: prompt-level strict/loose, instruction-level strict/loose.

    The official `instruction_following_eval` package exposes its scoring via
    `test_instruction_following_strict()` and `test_instruction_following_loose()`,
    one prompt at a time. We loop over our records, score each, and aggregate.
    """
    from instruction_following_eval import evaluation_lib

    prompt_strict_results = []
    prompt_loose_results = []
    inst_strict_results = []
    inst_loose_results = []

    # Map from prompt text to response, required by the scorer's API.
    prompt_to_response = {rec["prompt"]: rec["response"] for rec in records}

    for i, rec in enumerate(records):
        # The HF IFEval dataset has kwargs with all-fields-or-null structure;
        # the official scorer expects only the relevant fields per instruction.
        raw_kwargs = rec.get("kwargs", [{}] * len(rec["instruction_id_list"]))
        filtered_kwargs = [
            {k: v for k, v in (kw or {}).items() if v is not None}
            for kw in raw_kwargs
        ]
        inp = evaluation_lib.InputExample(
            key=i,
            instruction_id_list=rec["instruction_id_list"],
            prompt=rec["prompt"],
            kwargs=filtered_kwargs,
        )

        # Strict scoring
        out_strict = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
        prompt_strict_results.append(out_strict.follow_all_instructions)
        inst_strict_results.extend(out_strict.follow_instruction_list)

        # Loose scoring (more permissive, accepts certain variations)
        out_loose = evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)
        prompt_loose_results.append(out_loose.follow_all_instructions)
        inst_loose_results.extend(out_loose.follow_instruction_list)

    def rate(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "prompt_level_strict": rate(prompt_strict_results),
        "prompt_level_loose": rate(prompt_loose_results),
        "inst_level_strict": rate(inst_strict_results),
        "inst_level_loose": rate(inst_loose_results),
    }


# Judge-based scoring


def score_outputs_with_judge(judge: Judge, records: List[Dict]) -> List[Dict]:
    """Apply quality scoring via the judge to each record. Returns enriched records."""
    out = []
    for rec in tqdm(records, desc="Scoring quality", unit="prompt"):
        result = score_quality(judge, rec["response"])
        enriched = dict(rec)
        enriched["quality_score"] = result.score
        enriched["quality_reason"] = result.reason
        if result.error:
            enriched["quality_error"] = result.error
        out.append(enriched)
    return out


if __name__ == "__main__":
    main()
