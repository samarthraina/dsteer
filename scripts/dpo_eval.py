"""Safety/refusal evaluation. Compares IT and DPO models on:

- HH-RLHF (harmless-base, test split) — Anthropic's harmlessness benchmark
- HarmfulQA — non-optimized adversarial questions
- AdvBench — optimized jailbreak prompts

For each model on each dataset, generates responses and scores them with the judge on:
- refusal: did the model decline?
- helpfulness: was the response useful?
- harmfulness: did the output enable harm?
- quality: is the output coherent? (sanity check)

Output: per-prompt JSONL files with raw generations + scores, and a summary CSV
with means per (model, dataset).

Usage:
    # Start the judge server first in a separate terminal:
    python scripts/start_judge_server.py

    # Then run:
    python scripts/dpo_eval.py --model-config configs/llama3_oh.yaml --eval-config configs/dpo_eval.yaml
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd
import torch
from tqdm import tqdm

from steering.config import ModelConfig
from steering.data import load_advbench, load_harmfulqa, load_hh_rlhf_test
from steering.judge import Judge, JudgeConfig
from steering.metrics import score_harmfulness, score_helpfulness, score_quality, score_refusal
from steering.models import load_model, load_tokenizer
from steering.utils import (
    append_jsonl,
    load_yaml,
    read_jsonl,
    set_all_seeds,
    setup_logging,
    write_jsonl,
)


# Config


from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class DPOEvalConfig:
    """Settings for the DPO safety/refusal evaluation."""

    # Sample sizes (None = use full datasets)
    hh_rlhf_n: Optional[int] = 200
    harmfulqa_n: Optional[int] = 200
    advbench_n: Optional[int] = 200

    # Generation
    max_new_tokens: int = 512

    # Output location (per-model subdirs created automatically)
    output_dir: str = "outputs/dpo_eval"

    # Judge config (nested)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DPOEvalConfig":
        raw = load_yaml(path)
        judge_raw = raw.pop("judge", {})
        return cls(judge=JudgeConfig(**judge_raw), **raw)


# Main


def main():
    parser = argparse.ArgumentParser(description="Run DPO safety/refusal evaluation.")
    parser.add_argument("--model-config", required=True, help="Path to model YAML config")
    parser.add_argument("--eval-config", required=True, help="Path to DPO-eval YAML config")
    parser.add_argument(
        "--phase",
        choices=["generate", "score", "both"],
        default="both",
        help="generate: only generate responses (no judge/server needed, generator owns the GPU); "
             "score: only judge existing generations (no generator loaded, judge owns the GPU); "
             "both: full run (default; behavior unchanged from before this flag existed).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_all_seeds(args.seed)

    model_cfg = ModelConfig.from_yaml(args.model_config)
    eval_cfg = DPOEvalConfig.from_yaml(args.eval_config)

    output_root = Path(eval_cfg.output_dir) / model_cfg.name
    output_root.mkdir(parents=True, exist_ok=True)

    log = setup_logging(output_root / "dpo_eval.log")
    log.info(f"Model pair: {model_cfg.name}")
    log.info(f"Output directory: {output_root}")
    log.info(f"Phase: {args.phase}")

    do_generate = args.phase in ("generate", "both")
    do_score = args.phase in ("score", "both")

    dataset_specs = [
        ("hh_rlhf", load_hh_rlhf_test, eval_cfg.hh_rlhf_n),
        ("harmfulqa", load_harmfulqa, eval_cfg.harmfulqa_n),
        ("advbench", load_advbench, eval_cfg.advbench_n),
    ]
    checkpoints = [
        ("it", model_cfg.it_model, model_cfg.it_subfolder or ""),
        ("dpo", model_cfg.dpo_model, model_cfg.dpo_subfolder or ""),
    ]

    # Datasets + tokenizer are only needed when generating.
    datasets = {}
    tokenizer = None
    if do_generate:
        log.info("Loading datasets...")
        datasets = {name: loader(n=n) for name, loader, n in dataset_specs}
        for name, prompts in datasets.items():
            log.info(f"  {name}: {len(prompts)} prompts")
        tokenizer_path = model_cfg.tokenizer_id or model_cfg.it_model
        tokenizer_subfolder = model_cfg.tokenizer_subfolder or (model_cfg.it_subfolder or "")
        tokenizer = load_tokenizer(tokenizer_path, subfolder=tokenizer_subfolder)

    # The judge (vLLM HTTP client) is only needed when scoring.
    judge = Judge(eval_cfg.judge) if do_score else None

    # Process each checkpoint: generate (if requested) and free the model, then score (if requested).
    # For --phase both this is the original per-checkpoint order: generate IT, score IT, generate DPO, score DPO.
    summary_rows = []
    for checkpoint_label, model_path, subfolder in checkpoints:
        generated = {}
        if do_generate:
            log.info(f"\n{'=' * 60}\nGenerating {checkpoint_label.upper()}: {model_path} (subfolder={subfolder!r})\n{'=' * 60}")
            model = load_model(model_path, subfolder=subfolder)
            for dataset_name, prompts in datasets.items():
                output_path = output_root / f"{dataset_name}_{checkpoint_label}.jsonl"
                generated[dataset_name] = generate_for_prompts(
                    model, tokenizer, prompts, eval_cfg,
                    output_path=output_path,
                    label=f"{dataset_name}/{checkpoint_label}",
                )
            # Free the model before scoring so the judge can own the GPU.
            del model
            gc.collect()
            torch.cuda.empty_cache()

        if do_score:
            log.info(f"\n{'=' * 60}\nScoring {checkpoint_label.upper()}\n{'=' * 60}")
            for dataset_name, _loader, _n in dataset_specs:
                gen_path = output_root / f"{dataset_name}_{checkpoint_label}.jsonl"
                if dataset_name in generated:
                    # Just generated this run — use the in-memory records.
                    records = generated[dataset_name]
                elif gen_path.exists():
                    # Score-only run — read the generations from disk.
                    records = read_jsonl(gen_path)
                else:
                    log.warning(f"  {dataset_name}/{checkpoint_label}: no generations at {gen_path}, skipping")
                    continue

                scored_path = output_root / f"{dataset_name}_{checkpoint_label}_scored.jsonl"
                scored = score_records_with_judge(
                    judge, records,
                    output_path=scored_path,
                    label=f"{dataset_name}/{checkpoint_label}",
                )

                # Compute per-dataset means.
                stats = compute_dataset_stats(scored)
                stats["checkpoint"] = checkpoint_label
                stats["dataset"] = dataset_name
                stats["n"] = len(scored)
                summary_rows.append(stats)

                log.info(
                    f"  {dataset_name}/{checkpoint_label}: "
                    f"refusal={stats['refusal_mean']:.3f}, "
                    f"helpfulness={stats['helpfulness_mean']:.3f}, "
                    f"harmfulness={stats['harmfulness_mean']:.3f}, "
                    f"quality={stats['quality_mean']:.3f}"
                )

    # Save summary CSV (scoring phases only).
    if do_score:
        if not summary_rows:
            log.warning("No scored results to summarize (no generations found).")
        else:
            summary_path = output_root / "summary.csv"
            cols = ["checkpoint", "dataset", "n",
                    "refusal_mean", "helpfulness_mean", "harmfulness_mean", "quality_mean",
                    "refusal_n_scored", "helpfulness_n_scored", "harmfulness_n_scored", "quality_n_scored"]
            df = pd.DataFrame(summary_rows)[cols]
            df.to_csv(summary_path, index=False)
            log.info(f"\nSummary saved to {summary_path}")
            log.info("\nResults:")
            log.info("\n" + df.to_string(index=False))
    elif do_generate:
        log.info("\nGeneration phase complete. Run with --phase score (judge server up) to score.")


# Generation


def generate_for_prompts(
    model,
    tokenizer,
    prompts: List[Dict],
    cfg: DPOEvalConfig,
    output_path: Path,
    label: str,
) -> List[Dict]:
    """Generate responses for each prompt and save to JSONL.

    Resume-safe: if output_path already exists, skip prompts whose ids are already present.

    For HH-RLHF prompts where 'prompt' is a list of chat messages, applies the
    template directly. For others where 'prompt' is a string, wraps in a single user turn.
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
        # Build chat messages.
        prompt = prompt_rec["prompt"]
        if isinstance(prompt, list):
            # Already a list of messages (HH-RLHF).
            messages = prompt
        else:
            # Plain string prompt → single user turn.
            messages = [{"role": "user", "content": prompt}]

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
        # For HH-RLHF the prompt is a list of messages; for serialization, store the
        # final user query as a string for convenience.
        if isinstance(prompt, list):
            record["prompt_str"] = "\n".join(
                f"{m['role']}: {m['content']}" for m in prompt
            )
        record["response"] = response
        append_jsonl(record, output_path)

    return read_jsonl(output_path)


# Scoring


def score_records_with_judge(
    judge: Judge,
    records: List[Dict],
    output_path: Path,
    label: str,
) -> List[Dict]:
    """Apply refusal, helpfulness, harmfulness, and quality scoring to each record.

    Resume-safe: if output_path exists, skip records whose ids are already present
    AND already have all four scores populated.
    """
    existing = {}
    if output_path.exists():
        for rec in read_jsonl(output_path):
            if _is_fully_scored(rec):
                existing[rec["id"]] = rec
        if existing:
            logging.info(f"{label} scoring: resuming, {len(existing)} already scored")

    pending = [r for r in records if r["id"] not in existing]

    for rec in tqdm(pending, desc=f"Scoring {label}", unit="prompt"):
        # Resolve the prompt text the judge should see.
        prompt_text = rec.get("prompt_str") or rec["prompt"]
        if isinstance(prompt_text, list):
            prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in prompt_text)

        response = rec["response"]
        enriched = dict(rec)

        # Quality is the only property of the output alone. Harm is contextual -- the same
        # sentence reads differently against different requests -- so it gets the prompt too.
        scorers = [
            ("refusal", lambda: score_refusal(judge, input_text=prompt_text, output_text=response)),
            ("helpfulness", lambda: score_helpfulness(judge, input_text=prompt_text, output_text=response)),
            ("harmfulness", lambda: score_harmfulness(judge, input_text=prompt_text, output_text=response)),
            ("quality", lambda: score_quality(judge, output_text=response)),
        ]

        for name, call in scorers:
            res = call()
            enriched[f"{name}_score"] = res.score
            # Keep the unweighted integer too: it is what earlier runs recorded, so the
            # two are comparable, and it makes the weighting's effect measurable.
            enriched[f"{name}_raw_score"] = res.raw_score
            enriched[f"{name}_weighted"] = res.weighted
            enriched[f"{name}_reason"] = res.reason
            if res.error:
                enriched[f"{name}_error"] = res.error

        existing[rec["id"]] = enriched
        append_jsonl(enriched, output_path)

    # Return in the original record order.
    return [existing[r["id"]] for r in records if r["id"] in existing]


def _is_fully_scored(rec: Dict) -> bool:
    """Returns True if a record has all four score fields populated (non-None)."""
    return all(
        rec.get(f"{metric}_score") is not None
        for metric in ("refusal", "helpfulness", "harmfulness", "quality")
    )


def compute_dataset_stats(scored: List[Dict]) -> Dict[str, float]:
    """Compute mean scores across a list of scored records, ignoring None values."""
    stats = {}
    for metric in ("refusal", "helpfulness", "harmfulness", "quality"):
        scores = [r[f"{metric}_score"] for r in scored if r.get(f"{metric}_score") is not None]
        stats[f"{metric}_mean"] = sum(scores) / len(scores) if scores else float("nan")
        stats[f"{metric}_n_scored"] = len(scores)
    return stats


if __name__ == "__main__":
    main()
