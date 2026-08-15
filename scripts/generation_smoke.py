"""Gate 2: five-endpoint, 10-prompt generation smoke test (protocol Sections 5, 13
Gate 2, and 14).

Generates unsteered greedy responses for exactly the first 10 records (by ascending
`permuted_position`) of the frozen HarmfulQA `development` partition, from all five
verified endpoints -- M0-A, M+-A, M--A, M0-B, M+-B -- one at a time. This is the
smallest auditable check that every endpoint stops correctly and produces no
unexplained continuation after its end-of-turn token, before any larger pilot or
confirmatory run is attempted. It does not chain into the 50-prompt pilot (Gate 3) or
any later gate -- that is a separate, later step.

    python scripts/generation_smoke.py \\
        --endpoint-manifest outputs/endpoints/endpoint_manifest_candidate_v1.json \\
        --endpoint-bundle-root outputs/endpoints \\
        --endpoint-source pair_a_sft=/data/pair_a/SFT_merged \\
        --endpoint-source pair_a_dpo=/data/pair_a/DPO_merged \\
        --endpoint-source pair_b_sft=/data/pair_b/sft \\
        --output-dir outputs/gate2_smoke

Endpoint resolution and complete local-file verification (candidate-manifest hash and
structure, frozen source-artifact binding, and per-file SHA-256/size streaming) happen
before anything else: before CUDA seeding or any GPU check, before the HarmfulQA
dataset loads, before any tokenizer or model loads, before the output directory is
created, and before logging starts. A verification failure has no side effects.

The decoding configuration is frozen (greedy, `max_new_tokens=512`,
`max_input_length=2048`, no steering, no quantization) and not CLI-overridable; only
the generation batch size may be tuned for hardware efficiency, and it is always
recorded. Models are loaded and released strictly one at a time -- never two 7B/8B
checkpoints resident together.

Gate 2 passes only if every endpoint produces exactly 10 records, none stops with
reason "unknown", none reaches `max_new_tokens`, none has any non-padding,
non-terminator token after its first terminator, and none fails validity screening. A
failing gate still writes every output and the summary; only the process exit code
reflects the failure, so nothing here can silently swallow a bad run.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steering.artifacts import GpuMonitor, write_run_metadata
from steering.data import load_harmfulqa_partition
from steering.endpoint_binding import ALL_ROLES, ResolvedEndpointSet, parse_source_mappings, resolve_all_roles
from steering.generate import build_chat_prompts, generate_batched, generation_terminators
from steering.models import load_model, load_tokenizer
from steering.utils import set_all_seeds, setup_logging
from steering.validity import check as check_validity

#: Frozen for Gate 2 -- not CLI-overridable. A configurable smoke test would not be a
#: fixed, auditable gate.
SMOKE_RECORD_COUNT = 10
SMOKE_MAX_NEW_TOKENS = 512
SMOKE_MAX_INPUT_LENGTH = 2048
SMOKE_PARTITION = "development"


class GenerationSmokeError(RuntimeError):
    """A Gate 2 precondition (record selection, terminator availability, or
    generation-count integrity) was violated."""


def select_smoke_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Exactly the first `SMOKE_RECORD_COUNT` records of `records`, in ascending
    `permuted_position` -- never sampled, shuffled, or selected by ID. `records` is
    re-sorted here rather than trusted to already be in order, so this stays correct
    even if a future change to `load_harmfulqa_partition` altered its own ordering."""
    ordered = sorted(records, key=lambda r: r["permuted_position"])
    if len(ordered) < SMOKE_RECORD_COUNT:
        raise GenerationSmokeError(
            f"{SMOKE_PARTITION!r} partition has only {len(ordered)} records; Gate 2 needs {SMOKE_RECORD_COUNT}"
        )
    return ordered[:SMOKE_RECORD_COUNT]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_tokenizers(resolved: ResolvedEndpointSet, records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Load every endpoint's own tokenizer (local-only, no remote code), require a
    nonempty terminator set before any model is loaded, and pre-render and hash this
    run's prompts under each endpoint's own chat template. Entirely read-only: no
    model is loaded here, and no GPU is touched.

    Returns `{role: {"tokenizer", "terminators", "rendered_prompts", "rendered_sha256"}}`.
    """
    prompts = [r["prompt"] for r in records]
    out: Dict[str, Dict[str, Any]] = {}
    for role in ALL_ROLES:
        ep = resolved.roles[role]
        tokenizer = load_tokenizer(str(ep.local_path), trust_remote_code=False, local_files_only=True)
        terminators = generation_terminators(tokenizer)
        if not terminators:
            raise GenerationSmokeError(
                f"{role}: tokenizer has no usable EOS/end-of-turn terminator; refusing to load its model"
            )
        rendered = build_chat_prompts(tokenizer, prompts)
        out[role] = {
            "tokenizer": tokenizer, "terminators": terminators,
            "rendered_prompts": rendered, "rendered_sha256": [_sha256_text(p) for p in rendered],
        }
    return out


# Atomic file writes -- each completed file lands via write-to-temp-then-rename, so a
# crash mid-write can never leave a partial JSONL or summary. The whole run is *also*
# built in a staging directory and promoted with one atomic rename at the very end
# (see main()), so no output directory ever contains a partial or mixed-run result.


def _release_log_handlers_under(directory: Path) -> None:
    """Close and detach any logging handler (from `setup_logging`) writing into
    `directory`, so a directory rename can succeed. `logging.shutdown()` would do this
    too, but it is process-global and this script may run inside a longer-lived
    process (e.g. a test session) alongside other logging -- only the handler this
    run's own `setup_logging` call created is touched here."""
    directory = Path(directory).resolve()
    root = logging.getLogger()
    for handler in list(root.handlers):
        base_filename = getattr(handler, "baseFilename", None)
        if base_filename and Path(base_filename).resolve().is_relative_to(directory):
            handler.close()
            root.removeHandler(handler)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def write_role_jsonl_atomic(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    _atomic_write_text(path, text)


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


# Per-endpoint generation and Gate 2 evaluation


def generate_role_records(
    role: str, model: Any, tokenizer_info: Mapping[str, Any], records: Sequence[Dict[str, Any]],
    harmfulqa_manifest_hash: str, batch_size: int,
) -> List[Dict[str, Any]]:
    """Generate this endpoint's 10 responses and assemble one complete record per
    response: dataset provenance, endpoint role, rendered-prompt hash, raw response,
    exact stop provenance, resolved terminator IDs, the post-terminator-continuation
    flag, and validity screening."""
    results = generate_batched(
        model, tokenizer_info["tokenizer"], tokenizer_info["rendered_prompts"],
        max_new_tokens=SMOKE_MAX_NEW_TOKENS, batch_size=batch_size,
        max_input_length=SMOKE_MAX_INPUT_LENGTH, return_metadata=True,
    )
    if len(results) != len(records):
        raise GenerationSmokeError(
            f"{role}: generate_batched returned {len(results)} results for {len(records)} records; "
            "refusing to write a silently truncated batch"
        )

    out: List[Dict[str, Any]] = []
    for rec, sha, result in zip(records, tokenizer_info["rendered_sha256"], results):
        validity = check_validity(result.text)
        out.append({
            "source_id": rec["source_id"], "prompt_hash": rec["prompt_hash"],
            "permuted_position": rec["permuted_position"],
            "harmfulqa_manifest_hash": harmfulqa_manifest_hash, "harmfulqa_partition": SMOKE_PARTITION,
            "endpoint_role": role,
            "rendered_prompt_sha256": sha,
            "response": result.text,
            "generated_token_count": result.generated_token_count,
            "stop_reason": result.stop_reason, "stop_token_id": result.stop_token_id,
            "terminator_ids": list(tokenizer_info["terminators"]),
            "has_post_terminator_continuation": result.has_post_terminator_continuation,
            "validity_ok": validity.ok, "validity_reason": validity.reason,
            "validity_repetition": validity.repetition,
        })
    return out


def evaluate_role_gate(role: str, role_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """One endpoint's Gate 2 verdict (protocol Section 13, Gate 2)."""
    n = len(role_records)
    n_unknown = sum(1 for r in role_records if r["stop_reason"] == "unknown")
    n_max_new_tokens = sum(1 for r in role_records if r["stop_reason"] == "max_new_tokens")
    n_post_terminator = sum(1 for r in role_records if r["has_post_terminator_continuation"])
    n_invalid = sum(1 for r in role_records if not r["validity_ok"])
    role_pass = (
        n == SMOKE_RECORD_COUNT and n_unknown == 0 and n_max_new_tokens == 0
        and n_post_terminator == 0 and n_invalid == 0
    )
    return {
        "role": role, "n_records": n, "expected_records": SMOKE_RECORD_COUNT,
        "n_stop_reason_unknown": n_unknown, "n_max_new_tokens": n_max_new_tokens,
        "n_post_terminator_continuation": n_post_terminator, "n_invalid": n_invalid,
        "pass": role_pass,
    }


def build_summary(role_gates: Mapping[str, Dict[str, Any]], resolved: ResolvedEndpointSet) -> Dict[str, Any]:
    return {
        "gate": "gate2_generation_smoke",
        "pass": all(role_gates[role]["pass"] for role in ALL_ROLES),
        "roles": {role: role_gates[role] for role in ALL_ROLES},
        "candidate_manifest_hash": resolved.candidate_manifest_hash,
        "source_manifest_hash": resolved.source_manifest_hash,
    }


# CLI


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate 2: five-endpoint, 10-prompt generation smoke test.")
    parser.add_argument("--endpoint-manifest", required=True, help="Candidate endpoint manifest path.")
    parser.add_argument("--endpoint-bundle-root", required=True, help="Root directory holding merged endpoint bundles.")
    parser.add_argument(
        "--endpoint-source", action="append", default=[], metavar="ARTIFACT_ID=LOCAL_ROOT",
        help="Repeatable; requires exactly pair_a_sft, pair_a_dpo, and pair_b_sft "
             "(the three direct-source endpoints among all five roles).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--batch-size", type=int, default=SMOKE_RECORD_COUNT,
        help="Generation batch size, for hardware efficiency only -- the frozen "
             "decoding configuration (greedy, max_new_tokens, max_input_length) is "
             "unaffected. Always recorded in run_meta.json.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hourly-rate", type=float, default=None)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    # Endpoint resolution and complete verification -- before any CUDA seeding or GPU
    # check, dataset load, tokenizer/model load, output creation, or logging. A
    # mismatch has no side effects: nothing below this point has run yet.
    source_roots = parse_source_mappings(args.endpoint_source)
    resolved = resolve_all_roles(args.endpoint_manifest, source_roots, args.endpoint_bundle_root)

    set_all_seeds(args.seed)

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        print(f"FAIL: {output_dir} already exists; refusing to overwrite an existing output directory", file=sys.stderr)
        return 1

    records = load_harmfulqa_partition(SMOKE_PARTITION)
    smoke_records = select_smoke_records(records)
    harmfulqa_manifest_hash = smoke_records[0]["manifest_hash"]

    staging_dir = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    try:
        staging_dir.mkdir(parents=True)

        # Tokenizers only, read-only -- required before writing metadata (rendered-
        # prompt hashes and terminator IDs are endpoint-specific) and before any model
        # is loaded (Requirement 4: reject a missing terminator before that model loads).
        tokenizer_info = prepare_tokenizers(resolved, smoke_records)

        extra = {
            "gate": "gate2_generation_smoke",
            "harmfulqa_partition": SMOKE_PARTITION,
            "harmfulqa_manifest_hash": harmfulqa_manifest_hash,
            "records": [
                {"source_id": r["source_id"], "prompt_hash": r["prompt_hash"], "permuted_position": r["permuted_position"]}
                for r in smoke_records
            ],
            "decoding": {
                "do_sample": False, "greedy": True,
                "max_new_tokens": SMOKE_MAX_NEW_TOKENS, "max_input_length": SMOKE_MAX_INPUT_LENGTH,
                "batch_size": args.batch_size, "steering": None, "quantization": None,
                "sequential_endpoint_loading": True,
            },
            "endpoint": resolved.run_metadata(),
            "per_endpoint_generation": {
                role: {
                    "terminator_ids": list(tokenizer_info[role]["terminators"]),
                    "rendered_prompt_sha256": list(tokenizer_info[role]["rendered_sha256"]),
                }
                for role in ALL_ROLES
            },
        }
        write_run_metadata(staging_dir, config={"cli": vars(args)}, extra=extra, argv=list(sys.argv))

        log = setup_logging(staging_dir / "generation_smoke.log")
        log.info(f"Gate 2 smoke test: {len(smoke_records)} prompts from {SMOKE_PARTITION}")

        role_gates: Dict[str, Dict[str, Any]] = {}
        with GpuMonitor(staging_dir, hourly_rate=args.hourly_rate) as gpu:
            for role in ALL_ROLES:
                ep = resolved.roles[role]
                log.info(f"{role}: loading model from {ep.local_path}")
                model = load_model(str(ep.local_path), trust_remote_code=False, local_files_only=True)

                role_records = generate_role_records(
                    role, model, tokenizer_info[role], smoke_records, harmfulqa_manifest_hash, args.batch_size,
                )
                write_role_jsonl_atomic(staging_dir / f"{role}.jsonl", role_records)
                role_gates[role] = evaluate_role_gate(role, role_records)
                log.info(f"{role}: pass={role_gates[role]['pass']}")

                # One model resident at a time -- released before the next endpoint loads.
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        log.info(f"GPU usage: {json.dumps(gpu.summary())}")

        summary = build_summary(role_gates, resolved)
        write_json_atomic(staging_dir / "gate2_summary.json", summary)
    except Exception as e:  # noqa: BLE001 -- any failure anywhere here must promote nothing
        # The log file handler (once setup_logging has run) still holds
        # staging_dir/generation_smoke.log open; on Windows, deleting or renaming a
        # directory containing an open file handle silently fails (or leaves it
        # behind), so that handler is released before touching staging_dir at all.
        _release_log_handlers_under(staging_dir)
        shutil.rmtree(staging_dir, ignore_errors=True)
        print(f"FAIL: Gate 2 smoke test: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if output_dir.exists():  # race guard immediately before the one promotion
        _release_log_handlers_under(staging_dir)
        shutil.rmtree(staging_dir, ignore_errors=True)
        print(f"FAIL: {output_dir} appeared during the run; refusing to promote", file=sys.stderr)
        return 1
    _release_log_handlers_under(staging_dir)
    staging_dir.replace(output_dir)  # the single atomic promotion for the whole run

    # A failing gate is still fully recorded and promoted above; only the exit code
    # reflects the failure, and this script never chains into the 50-prompt pilot.
    print(f"Gate 2 {'PASS' if summary['pass'] else 'FAIL'}: see {output_dir}/gate2_summary.json")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
