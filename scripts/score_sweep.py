"""Score a steering sweep.

Takes the JSONL files a sweep produced and judges each generation on the rubrics, plus
Steering Shift against fixed IT and DPO references. Separate from generation so the
judge server is only needed for this half, and so a rescore does not regenerate.

    python scripts/start_judge_server.py                      # terminal 1
    python scripts/score_sweep.py --sweep-dir outputs/steer/tulu3/harmfulqa/it \
        --it-baseline  outputs/steer/tulu3/harmfulqa/it/baseline.jsonl \
        --dpo-baseline outputs/steer/tulu3/harmfulqa/dpo/baseline.jsonl

Two things it refuses to average away:

**Degenerate generations** are screened before judging, not after. Over-steering yields
empty strings and repetition loops, and letting those through means a broken generation
contributes a real number to the mean -- which flatters exactly the region where the
model has fallen apart.

**Judge abstentions** are counted, not zeroed. In the de-alignment direction the outputs
a judge declines to score are the unsafe ones, so coding them as zero would make a
de-aligned model look safe. Every metric reports its own n_scored beside its mean.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steering.artifacts import sync_to_hub, write_run_metadata
from steering.config import JudgeConfig
from steering.judge import Judge
from steering.metrics import (
    score_harmfulness, score_helpfulness, score_quality, score_refusal, score_steering_shift,
)
from steering.utils import append_jsonl, read_jsonl, setup_logging
from steering.validity import check, summarise

METRICS = ("refusal", "helpfulness", "harmfulness", "quality")


def prompt_text(record: Dict) -> str:
    p = record.get("prompt_str") or record.get("prompt")
    if isinstance(p, list):
        return "\n".join(f"{m['role']}: {m['content']}" for m in p)
    return str(p)


def load_references(path: Optional[str]) -> Dict[str, str]:
    """{prompt id: response} from a baseline file."""
    if not path:
        return {}
    return {r["id"]: r.get("response", "") for r in read_jsonl(Path(path))}


def build_run_config(args: argparse.Namespace, input_files: List[Path]) -> Dict[str, object]:
    """The resolved run identity for `write_run_metadata`: the complete parsed CLI
    namespace (including defaulted values), the resolved ordered input JSONL paths this
    run discovered, and the active direct metric names."""
    return {
        "cli": vars(args),
        "input_files": [str(p) for p in input_files],
        "metrics": list(METRICS),
    }


def score_one(judge: Judge, rec: Dict, it_ref: Dict[str, str], dpo_ref: Dict[str, str],
              min_length: int, repetition_threshold: float) -> Dict:
    """Screen and judge a single record. Pure with respect to shared state."""
    out = dict(rec)
    response = rec.get("response", "")
    verdict = check(response, min_length=min_length, repetition_threshold=repetition_threshold)
    out["valid"] = verdict.ok
    out["invalid_reason"] = verdict.reason
    out["repetition"] = round(verdict.repetition, 4)

    if not verdict.ok:
        # Recorded, not judged: a broken generation has no meaningful rubric score, and
        # guessing one would put it into the mean.
        for m in METRICS:
            out[f"{m}_score"] = None
            out[f"{m}_discrete_score"] = None
        out["steering_shift_score"] = None
        out["steering_shift_discrete_score"] = None
        return out

    text = prompt_text(rec)
    for name, res in [
        ("refusal", score_refusal(judge, input_text=text, output_text=response)),
        ("helpfulness", score_helpfulness(judge, input_text=text, output_text=response)),
        ("harmfulness", score_harmfulness(judge, input_text=text, output_text=response)),
        ("quality", score_quality(judge, output_text=response)),
    ]:
        # discrete_score (0-10, or None) and score (the same value / 10, or None) are
        # the judge's emitted integer -- the sole authoritative value (protocol
        # Section 10). No weighted/probability-derived value is recorded here.
        out[f"{name}_discrete_score"] = res.discrete_score
        out[f"{name}_score"] = res.score
        out[f"{name}_reason"] = res.reason
        if res.error:
            out[f"{name}_error"] = res.error

    a, b = it_ref.get(rec["id"]), dpo_ref.get(rec["id"])
    if a and b:
        shift = score_steering_shift(judge, input_text=text, output_text=response,
                                     reference_a=a, reference_b=b)
        out["steering_shift_discrete_score"] = shift.discrete_score
        out["steering_shift_score"] = shift.score
        out["steering_shift_reason"] = shift.reason
        if shift.error:
            out["steering_shift_error"] = shift.error
    else:
        out["steering_shift_score"] = None
        out["steering_shift_discrete_score"] = None
    return out


SCORERS = {
    "refusal": lambda j, t, r: score_refusal(j, input_text=t, output_text=r),
    "helpfulness": lambda j, t, r: score_helpfulness(j, input_text=t, output_text=r),
    "harmfulness": lambda j, t, r: score_harmfulness(j, input_text=t, output_text=r),
    "quality": lambda j, t, r: score_quality(j, output_text=r),
}


def recheck_file(judge: Judge, scored: Path, dst: Path, metrics: List[str],
                 concurrency: int = 64, chunk: int = 64, limit: Optional[int] = None) -> int:
    """Re-judge named metrics on an already-scored file, into a sidecar.

    When a rubric changes, only that metric is stale. Re-running the scorer would redo
    all of them -- five times the judge calls for one metric's worth of new information.

    Results land in a separate file keyed by id, rather than rewriting the scored file,
    so the old values survive for comparison and an interrupted run resumes by id. How
    far the two disagree is itself worth measuring: it says what the change was worth.
    """
    records = read_jsonl(scored)
    done = {r["id"] for r in read_jsonl(dst)} if dst.exists() else set()
    todo = [r for r in records
            if r["id"] not in done and r.get("valid", True) and r.get("response")]
    if limit is not None:
        todo = todo[:limit]

    def one(rec: Dict) -> Dict:
        out = {"id": rec["id"], "lambda": rec.get("lambda")}
        text = prompt_text(rec)
        for m in metrics:
            res = SCORERS[m](judge, text, rec["response"])
            out[f"{m}_discrete_score_v2"] = res.discrete_score
            out[f"{m}_score_v2"] = res.score
            out[f"{m}_reason_v2"] = res.reason
            if res.error:
                out[f"{m}_error_v2"] = res.error
            out[f"{m}_discrete_score_v1"] = rec.get(f"{m}_discrete_score")
            out[f"{m}_score_v1"] = rec.get(f"{m}_score")
        return out

    written = 0
    with tqdm(total=len(todo), desc=f"recheck {scored.stem}", unit="rec", leave=False) as bar:
        for start in range(0, len(todo), chunk):
            batch, results = todo[start:start + chunk], []
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {pool.submit(one, rec): rec for rec in batch}
                for fut in as_completed(futures):
                    try:
                        results.append(fut.result())
                    except Exception as e:  # noqa: BLE001
                        logging.warning(f"recheck {futures[fut].get('id')}: {type(e).__name__}: {e}")
                    bar.update(1)
            for rec in sorted(results, key=lambda r: str(r["id"])):
                append_jsonl(rec, dst)
                written += 1
    return written


def score_file(
    judge: Judge,
    src: Path,
    dst: Path,
    it_ref: Dict[str, str],
    dpo_ref: Dict[str, str],
    min_length: int,
    repetition_threshold: float,
    concurrency: int = 32,
    chunk: int = 64,
) -> List[Dict]:
    """Score one lambda's generations. Resume-safe by record id.

    Records are judged concurrently. A single judge call takes about two seconds, almost
    all of it waiting on the server, so issuing them one at a time puts the whole sweep
    in the tens of hours while leaving the GPU mostly idle -- vLLM batches concurrent
    requests server-side, and the client was the bottleneck. Writes still happen in
    chunks so an interrupted run resumes near where it stopped rather than at the start.
    """
    records = read_jsonl(src)
    done = {r["id"]: r for r in read_jsonl(dst)} if dst.exists() else {}
    todo = [r for r in records if r["id"] not in done]

    with tqdm(total=len(todo), desc=src.stem, unit="rec", leave=False) as bar:
        for start in range(0, len(todo), chunk):
            batch = todo[start:start + chunk]
            results: List[Dict] = []
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {
                    pool.submit(score_one, judge, rec, it_ref, dpo_ref,
                                min_length, repetition_threshold): rec
                    for rec in batch
                }
                for fut in as_completed(futures):
                    rec = futures[fut]
                    try:
                        results.append(fut.result())
                    except Exception as e:  # noqa: BLE001 -- one bad record must not stop the file
                        logging.warning(f"{src.stem} id={rec.get('id')}: {type(e).__name__}: {e}")
                    bar.update(1)

            # Written in submission order so the file stays stable across reruns.
            for rec in sorted(results, key=lambda r: str(r["id"])):
                done[rec["id"]] = rec
                append_jsonl(rec, dst)

    return list(done.values())


def main():
    parser = argparse.ArgumentParser(description="Score a steering sweep with the judge.")
    parser.add_argument("--sweep-dir", required=True, help="Directory of *.jsonl from steer_sweep.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <sweep-dir>/scored.")
    parser.add_argument("--it-baseline", help="Unsteered IT generations (reference A).")
    parser.add_argument("--dpo-baseline", help="Unsteered DPO generations (reference B).")
    parser.add_argument("--judge-url", default="http://localhost:8000/v1")
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--min-length", type=int, default=10)
    parser.add_argument("--repetition-threshold", type=float, default=0.5)
    parser.add_argument(
        "--concurrency", type=int, default=32,
        help="Judge calls in flight. The server batches them; the client was the limit.",
    )
    parser.add_argument("--sync", action="store_true")
    parser.add_argument(
        "--recheck", default=None, metavar="METRIC[,METRIC]",
        help="Re-judge only these metrics on an already-scored sweep, into *_recheck.jsonl. "
             "Use when a rubric changed: rescoring everything costs five times the calls "
             "for one metric's worth of new information.",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many records per file. For a pilot before "
                             "committing to the full set.")
    args = parser.parse_args()

    sweep = Path(args.sweep_dir)
    out_dir = Path(args.output_dir) if args.output_dir else sweep / "scored"

    # Read-only preparation for the metadata identity check below -- discovering input
    # file names and reading small reference files mutates nothing in out_dir.
    files = sorted(p for p in sweep.glob("*.jsonl"))
    if not files:
        parser.error(f"no .jsonl in {sweep}")

    it_ref = load_references(args.it_baseline)
    dpo_ref = load_references(args.dpo_baseline)

    write_run_metadata(
        out_dir,
        config=build_run_config(args, files),
        argv=list(sys.argv),
    )

    # Nothing below may run until the identity check above has succeeded: the log is
    # not initialised and no judge client is constructed before this point -- a
    # mismatched resume fails here, before it can touch anything.
    log = setup_logging(out_dir / "score_sweep.log")
    log.info(f"{len(files)} files to score in {sweep}")
    if it_ref and dpo_ref:
        log.info(f"Steering Shift references: {len(it_ref)} IT, {len(dpo_ref)} DPO")
    else:
        log.warning("No reference pair given -- Steering Shift will be skipped")

    judge = Judge(JudgeConfig(model_name=args.judge_model, server_url=args.judge_url))

    if args.recheck:
        metrics = [m.strip() for m in args.recheck.split(",")]
        bad = [m for m in metrics if m not in SCORERS]
        if bad:
            parser.error(f"unknown metric(s) {bad}; use {sorted(SCORERS)}")
        total = 0
        for src in files:
            scored = out_dir / f"{src.stem}_scored.jsonl"
            if not scored.exists():
                log.warning(f"no scores for {src.stem}; skipping")
                continue
            n = recheck_file(judge, scored, out_dir / f"{src.stem}_recheck.jsonl",
                             metrics, concurrency=args.concurrency, limit=args.limit)
            total += n
            log.info(f"{src.stem}: +{n} rechecked")
        log.info(f"{total} records rechecked on {metrics}")
        if args.sync:
            sync_to_hub(out_dir, experiment="steer_scored", model=sweep.parent.parent.name)
        return

    rows = []
    for src in files:
        scored = score_file(judge, src, out_dir / f"{src.stem}_scored.jsonl",
                            it_ref, dpo_ref, args.min_length, args.repetition_threshold,
                            concurrency=args.concurrency)
        row = {
            "setting": src.stem,
            "lambda": scored[0].get("lambda") if scored else None,
            "n": len(scored),
            "n_invalid": sum(1 for r in scored if not r.get("valid", True)),
        }
        for m in list(METRICS) + ["steering_shift"]:
            stats = summarise([r.get(f"{m}_score") for r in scored])
            row[f"{m}_mean"] = stats["mean"]
            row[f"{m}_n"] = stats["n_scored"]
            row[f"{m}_abstained"] = stats["n_abstained"]
        rows.append(row)
        log.info(f"{src.stem}: n={row['n']} invalid={row['n_invalid']} "
                 f"shift={row['steering_shift_mean']:.3f} refusal={row['refusal_mean']:.3f}")

    df = pd.DataFrame(rows).sort_values("lambda", na_position="first")
    df.to_csv(out_dir / "summary.csv", index=False)
    log.info("\n" + df.to_string(index=False))

    if args.sync:
        sync_to_hub(out_dir, experiment="steer_scored", model=sweep.parent.parent.name)


if __name__ == "__main__":
    main()
