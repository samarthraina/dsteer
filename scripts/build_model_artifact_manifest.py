"""Build (or re-verify) the frozen model source-artifact manifest (protocol Gate 0).

Encodes the five source artifacts identified in
`Brain/ARTIFACT_IDENTITY_AUDIT_2026-08-15.md` -- Pair A's SFT/DPO merged checkpoints,
Pair B's SFT checkpoint, and the two recovered DPO adapters -- as one schema-versioned,
hash-verified manifest via `steering.artifact_identity.build_manifest`. An existing
manifest is never overwritten with different content: rerunning against the same
literal values reports success without rewriting; any difference fails loudly instead
of silently updating.

Entirely offline: every value here is a literal transcription of the audit document.
No network access, no model download, no merge.

    python scripts/build_model_artifact_manifest.py [--out manifests/model_artifacts_v1.json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steering.artifact_identity import ArtifactIdentityError, build_manifest, save_manifest

PAIR_A_REPO = "sirius5005/SFT-and-DPO"
PAIR_A_REVISION = "2932781b06bb062fb0fde146be0ebc13315fbbd3"
PAIR_A_TOKENIZER_SHA256 = "8c5bbfc28fa7ce7c55165a2c11eee1765f6eed6ee6fcdc69ef6d9c2f17a41bda"

PAIR_B_SFT_REPO = "teknium/OpenHermes-2.5-Mistral-7B"
PAIR_B_SFT_REVISION = "24c0bea14d53e6f67f1fbe2eca5bfe7cae389b33"

RESULTS_REPO = "samarthraina/dsteer-results"
RESULTS_REVISION = "613e550682436b83593aee0b6444ab0dfa56b659"

ARTIFACTS = [
    {
        "artifact_id": "pair_a_sft",
        "repository_type": "model",
        "repository": PAIR_A_REPO,
        "revision": PAIR_A_REVISION,
        "subpath": "SFT_merged",
        "kind": "checkpoint",
        "base_artifact_id": None,
        "inference_ready": True,
        "files": [
            {
                "path": "model.safetensors",
                "sha256": "1a2a404db755b04ac9385d8477f1853d86a586c8ec691abef9931701c0da50d5",
                "size_bytes": None,
            },
            {
                "path": "tokenizer.json",
                "sha256": PAIR_A_TOKENIZER_SHA256,
                "size_bytes": None,
            },
        ],
        "lineage": {},
    },
    {
        "artifact_id": "pair_a_dpo",
        "repository_type": "model",
        "repository": PAIR_A_REPO,
        "revision": PAIR_A_REVISION,
        "subpath": "DPO_merged",
        "kind": "checkpoint",
        "base_artifact_id": None,
        "inference_ready": True,
        "files": [
            {
                "path": "model.safetensors",
                "sha256": "5eedaac41dcbc1fcab7446ba40b53b21d7c392e0ffe896da1b801c3be873b004",
                "size_bytes": None,
            },
            {
                "path": "tokenizer.json",
                "sha256": PAIR_A_TOKENIZER_SHA256,
                "size_bytes": None,
            },
        ],
        "lineage": {},
    },
    {
        "artifact_id": "pair_b_sft",
        "repository_type": "model",
        "repository": PAIR_B_SFT_REPO,
        "revision": PAIR_B_SFT_REVISION,
        "subpath": "",
        "kind": "checkpoint",
        "base_artifact_id": None,
        "inference_ready": True,
        "files": [
            {
                "path": "model-00001-of-00002.safetensors",
                "sha256": "0b712f11ea29f3b34fa132403f7cafc0568c722ba3a33f42b55ed77b47fa299d",
                "size_bytes": None,
            },
            {
                "path": "model-00002-of-00002.safetensors",
                "sha256": "5e6249c1a1ceb365e219a0fe667a77f71ec005b3aecb145ff2d8adf46cdb574f",
                "size_bytes": None,
            },
        ],
        "lineage": {},
    },
    {
        "artifact_id": "pair_b_dpo_adapter",
        "repository_type": "dataset",
        "repository": RESULTS_REPO,
        "revision": RESULTS_REVISION,
        "subpath": "runs/dpo_training/openhermes-mistral-adapter/20260809T155221Z",
        "kind": "adapter",
        "base_artifact_id": "pair_b_sft",
        "inference_ready": False,
        "files": [
            {
                "path": "adapter_model.safetensors",
                "sha256": "79c81ebc54c040c305fd35524fafe0edb3facc6be91284ad3e5ddc97bb758517",
                "size_bytes": None,
            },
            {
                "path": "training_args.bin",
                "sha256": "67afea1a9303f2ff05cbb845452e4396bb10ca2b82bf7ba9dbe4fdca81225642",
                "size_bytes": None,
            },
        ],
        "lineage": {},
    },
    {
        "artifact_id": "pair_a_flip_adapter",
        "repository_type": "dataset",
        "repository": RESULTS_REPO,
        "revision": RESULTS_REVISION,
        "subpath": "runs/dpo_training/llama3-oh-flip/20260812T060638Z/final_dpo_adapter",
        "kind": "adapter",
        "base_artifact_id": "pair_a_sft",
        "inference_ready": False,
        "files": [
            {
                "path": "adapter_model.safetensors",
                "sha256": "bf241922adfddd09559c32f157570344e12558a6300726d862cfda5ae32d3682",
                "size_bytes": None,
            },
            {
                "path": "training_args.bin",
                "sha256": "4ec20c965ef181d607d756ff690047fbb89a110624d79301ade070b9ebe62f3d",
                "size_bytes": None,
            },
        ],
        # Not a declared hashed file -- a lineage pointer only. The archive's Git blob
        # ID for this JSON is not a SHA-256 of file content and must never be presented
        # as one.
        "lineage": {
            "archived_trainer_state_path": (
                "runs/dpo_training/llama3-oh-flip/20260812T060638Z/"
                "checkpoint-2510/trainer_state.json"
            ),
        },
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="manifests/model_artifacts_v1.json")
    args = parser.parse_args()

    try:
        payload = build_manifest(ARTIFACTS)
    except ArtifactIdentityError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    try:
        written = save_manifest(payload, out_path)
    except ArtifactIdentityError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"manifest hash: {payload['manifest_hash']}")
    print(f"artifacts: {sorted(a['artifact_id'] for a in payload['artifacts'])}")
    print(f"{'wrote' if written else 'unchanged, already present at'}: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
