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
from steering.judge_identity import (
    JudgeIdentityError,
    STRUCTURED_OUTPUT_SCHEMA,
    load_manifest as load_judge_manifest,
    validate_frozen_identity as validate_frozen_judge_identity,
    validate_judge_config,
    validate_live_evaluator_identity,
)
from steering.metrics import (
    ACTIVE_RUBRICS, LEGACY_HARMFULNESS_RUBRIC_PRE_REPAIR,
    score_harmfulness, score_helpfulness, score_quality, score_refusal, score_steering_shift,
)
from steering.utils import append_jsonl, read_jsonl, setup_logging
from steering.validity import check, summarise

METRICS = ("refusal", "helpfulness", "harmfulness", "quality")

#: Legacy default (protocol Section 10 permits a historical-judge sensitivity pass on
#: this model): unaffected by --confirmatory validation unless explicitly overridden to
#: something else.
LEGACY_JUDGE_MODEL_DEFAULT = "Qwen/Qwen2.5-32B-Instruct"

DEFAULT_JUDGE_PROTOCOL_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifests" / "judge_protocol_v1.json"


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


def build_run_config(
    args: argparse.Namespace, input_files: List[Path], judge_protocol: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """The resolved run identity for `write_run_metadata`: the complete parsed CLI
    namespace (including defaulted values), the resolved ordered input JSONL paths this
    run discovered, and the active direct metric names. On the confirmatory path,
    `judge_protocol` additionally binds the verified judge identity, its manifest hash,
    the request-seed policy, and the resolved concurrency (Task 010)."""
    config: Dict[str, object] = {
        "cli": vars(args),
        "input_files": [str(p) for p in input_files],
        "metrics": list(METRICS),
    }
    if judge_protocol is not None:
        config["judge_protocol"] = judge_protocol
    return config


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
        # guessing one would put it into the mean. `_abstained=None` (not True/False)
        # marks that no judge call happened at all -- distinct from a genuine judge
        # abstention, where a call was made and the judge failed to return a valid score.
        for m in METRICS:
            out[f"{m}_score"] = None
            out[f"{m}_discrete_score"] = None
            out[f"{m}_attempts"] = []
            out[f"{m}_attempt_count"] = 0
            out[f"{m}_resolved_seed"] = None
            out[f"{m}_abstained"] = None
        out["steering_shift_score"] = None
        out["steering_shift_discrete_score"] = None
        out["steering_shift_presented_order"] = None
        out["steering_shift_inverted"] = None
        out["steering_shift_attempts"] = []
        out["steering_shift_attempt_count"] = 0
        out["steering_shift_resolved_seed"] = None
        out["steering_shift_abstained"] = None
        return out

    record_id = rec["id"]
    text = prompt_text(rec)
    for name, res in [
        ("refusal", score_refusal(judge, input_text=text, output_text=response, record_id=record_id)),
        ("helpfulness", score_helpfulness(judge, input_text=text, output_text=response, record_id=record_id)),
        ("harmfulness", score_harmfulness(judge, input_text=text, output_text=response, record_id=record_id)),
        ("quality", score_quality(judge, output_text=response, record_id=record_id)),
    ]:
        # discrete_score (0-10, or None) and score (the same value / 10, or None) are
        # the judge's emitted integer -- the sole authoritative value (protocol
        # Section 10). No weighted/probability-derived value is recorded here.
        out[f"{name}_discrete_score"] = res.discrete_score
        out[f"{name}_score"] = res.score
        out[f"{name}_reason"] = res.reason
        if res.error:
            out[f"{name}_error"] = res.error
        # Full attempt-level provenance (protocol Section 10 / Task 010): every
        # attempt, the resolved seed, and whether this call genuinely abstained (a
        # call happened but no valid score came back) -- distinct from the
        # not-judged case above, which is never a judge abstention.
        out[f"{name}_attempts"] = res.attempts
        out[f"{name}_attempt_count"] = res.attempt_count
        out[f"{name}_resolved_seed"] = res.resolved_seed
        out[f"{name}_abstained"] = res.discrete_score is None

    a, b = it_ref.get(record_id), dpo_ref.get(record_id)
    if a and b:
        shift = score_steering_shift(judge, record_id=record_id, input_text=text, output_text=response,
                                     it_reference=a, dpo_reference=b)
        out["steering_shift_discrete_score"] = shift.discrete_score
        out["steering_shift_score"] = shift.score
        out["steering_shift_reason"] = shift.reason
        out["steering_shift_presented_order"] = shift.presented_order
        out["steering_shift_inverted"] = shift.inverted
        out["steering_shift_attempts"] = shift.attempts
        out["steering_shift_attempt_count"] = shift.attempt_count
        out["steering_shift_resolved_seed"] = shift.resolved_seed
        out["steering_shift_abstained"] = shift.abstained
        if shift.error:
            out["steering_shift_error"] = shift.error
    else:
        # No reference pair -- Steering Shift is not evaluated for this record at all,
        # not merely abstained: `_abstained=None` matches the not-judged convention above.
        out["steering_shift_score"] = None
        out["steering_shift_discrete_score"] = None
        out["steering_shift_presented_order"] = None
        out["steering_shift_inverted"] = None
        out["steering_shift_attempts"] = []
        out["steering_shift_attempt_count"] = 0
        out["steering_shift_resolved_seed"] = None
        out["steering_shift_abstained"] = None
    return out


SCORERS = {
    "refusal": lambda j, t, r, rid: score_refusal(j, input_text=t, output_text=r, record_id=rid),
    "helpfulness": lambda j, t, r, rid: score_helpfulness(j, input_text=t, output_text=r, record_id=rid),
    "harmfulness": lambda j, t, r, rid: score_harmfulness(j, input_text=t, output_text=r, record_id=rid),
    "quality": lambda j, t, r, rid: score_quality(j, output_text=r, record_id=rid),
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
            res = SCORERS[m](judge, text, rec["response"], rec["id"])
            out[f"{m}_discrete_score_v2"] = res.discrete_score
            out[f"{m}_score_v2"] = res.score
            out[f"{m}_reason_v2"] = res.reason
            if res.error:
                out[f"{m}_error_v2"] = res.error
            # A recheck always calls the judge (`todo` above is already filtered to
            # valid, response-bearing records), so `_abstained_v2` is always a real
            # bool here, never None.
            out[f"{m}_attempts_v2"] = res.attempts
            out[f"{m}_attempt_count_v2"] = res.attempt_count
            out[f"{m}_resolved_seed_v2"] = res.resolved_seed
            out[f"{m}_abstained_v2"] = res.discrete_score is None
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
    parser.add_argument(
        "--confirmatory", action="store_true",
        help="Use exactly the frozen Qwen3.5 judge protocol (protocol Section 10): "
             "loads and verifies manifests/judge_protocol_v1.json (self-hash, frozen "
             "identity), builds a JudgeConfig pinned to it, and rejects any "
             "non-frozen --judge-model/--concurrency override before any mutation.",
    )
    parser.add_argument(
        "--judge-protocol-manifest", default=None,
        help=f"Path to the judge-protocol manifest (default: {DEFAULT_JUDGE_PROTOCOL_MANIFEST_PATH}). "
             "Only consulted with --confirmatory.",
    )
    args = parser.parse_args()

    sweep = Path(args.sweep_dir)
    out_dir = Path(args.output_dir) if args.output_dir else sweep / "scored"

    # Read-only preparation for the metadata identity check below -- discovering input
    # file names, reading small reference files, and loading/verifying the static
    # judge-protocol manifest all mutate nothing in out_dir.
    files = sorted(p for p in sweep.glob("*.jsonl"))
    if not files:
        parser.error(f"no .jsonl in {sweep}")

    it_ref = load_references(args.it_baseline)
    dpo_ref = load_references(args.dpo_baseline)

    judge_protocol_extra = None
    if args.confirmatory:
        manifest_path = Path(args.judge_protocol_manifest) if args.judge_protocol_manifest else DEFAULT_JUDGE_PROTOCOL_MANIFEST_PATH
        try:
            judge_manifest = load_judge_manifest(manifest_path)
            validate_frozen_judge_identity(judge_manifest)
            # The manifest can self-verify and match the hardcoded FROZEN_JUDGE_IDENTITY
            # pin and still be stale, if the system prompt or a rubric was edited after
            # the manifest was last built. This re-hashes the text actually imported
            # right now (Judge.SYSTEM_PROMPT, ACTIVE_RUBRICS, the legacy rubric, and the
            # structured-output schema) and requires it to match what the manifest
            # claims -- catching drift that manifest-vs-itself checks cannot see.
            validate_live_evaluator_identity(
                judge_manifest,
                system_prompt=Judge.SYSTEM_PROMPT,
                active_rubrics=ACTIVE_RUBRICS,
                legacy_harmfulness_rubric=LEGACY_HARMFULNESS_RUBRIC_PRE_REPAIR,
                structured_output_schema=STRUCTURED_OUTPUT_SCHEMA,
            )
        except JudgeIdentityError as e:
            parser.error(f"--confirmatory judge-protocol manifest failed verification: {e}")

        frozen_alias = judge_manifest["judge"]["served_model_alias"]
        if args.judge_model not in (LEGACY_JUDGE_MODEL_DEFAULT, frozen_alias):
            parser.error(f"--confirmatory requires --judge-model={frozen_alias!r} (or omit it); got {args.judge_model!r}")
        if args.concurrency != judge_manifest["concurrency"]:
            parser.error(
                f"--confirmatory requires --concurrency={judge_manifest['concurrency']} "
                f"(or omit it); got {args.concurrency}"
            )

        judge_cfg = JudgeConfig.frozen_qwen35(server_url=args.judge_url)
        try:
            validate_judge_config(judge_cfg, expected=judge_manifest)
        except JudgeIdentityError as e:
            parser.error(f"--confirmatory resolved JudgeConfig failed verification: {e}")

        judge_protocol_extra = {
            "manifest_hash": judge_manifest["manifest_hash"],
            # The complete verified manifest (judge identity, sampling, schema,
            # every hash), not only the `judge` subsection -- the reader should never
            # need to reconstruct sampling/schema/hash fields from elsewhere.
            "manifest": judge_manifest,
            "seed_policy": {
                "global_seed": judge_manifest["global_seed"],
                "seed_derivation_version": judge_manifest["seed_derivation_version"],
            },
            "resolved_concurrency": args.concurrency,
            # The model name actually used, unambiguous even though --judge-model's CLI
            # default (recorded verbatim in `cli.judge_model`) is still the historical
            # Qwen2.5 name when omitted: `judge_cfg.model_name` is always the resolved
            # revision-bearing Qwen3.5 alias on this path, never the CLI default string.
            "resolved_model_alias": judge_cfg.model_name,
        }
    else:
        judge_cfg = JudgeConfig(model_name=args.judge_model, server_url=args.judge_url)

    write_run_metadata(
        out_dir,
        config=build_run_config(args, files, judge_protocol=judge_protocol_extra),
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

    judge = Judge(judge_cfg)

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
