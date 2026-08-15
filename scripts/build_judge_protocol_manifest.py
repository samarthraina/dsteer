"""Offline, deterministic builder for `manifests/judge_protocol_v1.json`.

Binds every frozen judge-protocol field (`Brain/EXPERIMENT_PROTOCOL_V1.md` Section 10)
plus the live system-prompt and active-rubric hashes, computed fresh from
`Judge.SYSTEM_PROMPT` and `steering.metrics.ACTIVE_RUBRICS` -- not from a separately
typed-out copy. Rebuilding must reproduce the committed file byte-for-byte: a rubric or
prompt edit that is not reflected in `judge_identity.FROZEN_JUDGE_IDENTITY` shows up
here as a frozen-identity mismatch rather than a silent drift.

No model, network, GPU, or judge server: this only hashes strings and writes JSON.

    python scripts/build_judge_protocol_manifest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steering.judge import Judge
from steering.judge_identity import (
    FROZEN_JUDGE_IDENTITY,
    STRUCTURED_OUTPUT_SCHEMA,
    build_manifest,
    save_manifest,
    validate_frozen_identity,
)
from steering.metrics import ACTIVE_RUBRICS, LEGACY_HARMFULNESS_RUBRIC_PRE_REPAIR

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifests" / "judge_protocol_v1.json"


def build() -> dict:
    f = FROZEN_JUDGE_IDENTITY
    return build_manifest(
        repository=f["judge"]["repository"],
        revision=f["judge"]["revision"],
        vllm_version=f["judge"]["vllm_version"],
        text_only=f["judge"]["text_only"],
        thinking_enabled=f["judge"]["thinking_enabled"],
        dtype=f["judge"]["dtype"],
        quantization=f["judge"]["quantization"],
        sampling=f["sampling"],
        global_seed=f["global_seed"],
        seed_derivation_version=f["seed_derivation_version"],
        max_response_tokens=f["max_response_tokens"],
        max_model_len=f["max_model_len"],
        max_attempts=f["max_attempts"],
        concurrency=f["concurrency"],
        prefix_caching=f["prefix_caching"],
        structured_output_schema=STRUCTURED_OUTPUT_SCHEMA,
        system_prompt=Judge.SYSTEM_PROMPT,
        active_rubrics=ACTIVE_RUBRICS,
        legacy_harmfulness_rubric=LEGACY_HARMFULNESS_RUBRIC_PRE_REPAIR,
    )


def main() -> None:
    manifest = build()
    # Must already match the pinned identity before it is ever written: a manifest that
    # only agrees with itself is not what this task is reviewed against.
    validate_frozen_identity(manifest)
    written = save_manifest(manifest, MANIFEST_PATH)
    print(f"{'wrote' if written else 'unchanged (already byte-identical)'}: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
