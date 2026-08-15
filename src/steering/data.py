"""Dataset loaders. All return List[Dict[str, str]] with at least a "prompt" key.

Sources:
- IFEval (Google): verifiable-instruction prompts
- AlpacaEval: general instruction-following prompts
- HH-RLHF (harmless-base, test): conversational safety prompts
- HarmfulQA: adversarial questions
- AdvBench: optimized jailbreak prompts
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset

from steering.splits import (
    load_manifest,
    prompt_hash,
    validate_manifest_identity,
    validate_source_binding,
)

log = logging.getLogger(__name__)


def load_ifeval(n: Optional[int] = None, seed: int = 42) -> List[Dict[str, str]]:
    """Load IFEval prompts with their verifiable instructions.

    Each record: {"id", "prompt", "instruction_id_list", "kwargs"}
    The instruction_id_list + kwargs are used by the scorer (rule-based, no judge).
    """
    ds = load_dataset("google/IFEval", split="train")
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    records = []
    for i, row in enumerate(ds):
        records.append({
            "id": f"ifeval-{i}",
            "prompt": row["prompt"],
            "instruction_id_list": row["instruction_id_list"],
            "kwargs": row["kwargs"],
        })
    log.info(f"Loaded IFEval: {len(records)} prompts")
    return records


def load_alpacaeval(n: int = 50, seed: int = 42) -> List[Dict[str, str]]:
    """Load AlpacaEval prompts for general instruction-following quality.

    Fetched as a plain file rather than through `load_dataset`, because newer versions of
    `datasets` refuse to run the dataset's loading script. Through the hub client rather
    than urllib, which brings retries, caching and the local token with it -- a bare
    urlopen fails on any network that resets the connection, and does so on every call.
    """
    import json
    import random

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        filename="alpaca_eval.json",
        repo_type="dataset",
    )
    with open(path, encoding="utf-8") as fh:
        items = json.load(fh)

    rng = random.Random(seed)
    if n < len(items):
        items = rng.sample(items, n)

    records = []
    for i, row in enumerate(items):
        records.append({
            "id": f"alpaca-{i}",
            "prompt": row["instruction"],
        })
    log.info(f"Loaded AlpacaEval: {len(records)} prompts")
    return records


def load_hh_rlhf_test(n: Optional[int] = None, seed: int = 42) -> List[Dict[str, str]]:
    """Load HH-RLHF harmless-base test split.

    Each record: {"id", "prompt", "chosen", "rejected"}
    The "prompt" is the multi-turn conversation up to (but not including) the final assistant turn.
    """
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="test")
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))

    records = []
    for i, row in enumerate(ds):
        prompt, chosen = _split_hh_conversation(row["chosen"])
        _, rejected = _split_hh_conversation(row["rejected"])
        if prompt is None:
            continue
        records.append({
            "id": f"hh-{i}",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    log.info(f"Loaded HH-RLHF (harmless-base, test): {len(records)} prompts")
    return records


HARMFULQA_CONFIG = "default"
# Pinned per protocol Section 5. Source validation at this revision found 1,960 raw
# rows / 22 verbatim duplicate pairs -- an unpinned "latest" default would silently
# invalidate that count and the manifest built from it.
HARMFULQA_REVISION = "6f1a78aed47d16c0695e4595d0159abc38197bfd"

VALID_HARMFULQA_PARTITIONS = ("construction", "calibration", "final_evaluation", "development")


def _harmfulqa_record(source_index: int, prompt: str) -> Dict[str, Any]:
    """The one place a HarmfulQA row becomes an identity: `source_id`/`id` from its raw
    position, plus its normalized `prompt_hash` -- shared by `load_harmfulqa` and
    `load_harmfulqa_partition` so the two never compute identity differently."""
    source_id = f"harmfulqa-{source_index}"
    return {
        "id": source_id,
        "source_id": source_id,
        "source_index": source_index,
        "prompt_hash": prompt_hash(prompt),
        "prompt": prompt,
    }


def load_harmfulqa(
    n: Optional[int] = None, seed: int = 42, revision: Optional[str] = HARMFULQA_REVISION,
) -> List[Dict[str, str]]:
    """Load HarmfulQA prompts (non-optimized adversarial questions).

    `id`/`source_id`/`source_index` are the row's position in the unshuffled dataset at
    `revision`, assigned before any sampling -- the same source row keeps the same
    identity no matter what `n` or `seed` a caller requests. Kept out of the
    shuffle/select call below so existing sampling behavior for `n` is unchanged.

    Does not exclude the 22 verbatim duplicate rows documented in the split manifest --
    that exclusion applies to the frozen partition manifest (`steering.splits`), not to
    this general-purpose loader used for ad hoc sampling elsewhere. For the frozen,
    manifest-backed partitions, use `load_harmfulqa_partition`.
    """
    ds = load_dataset("declare-lab/HarmfulQA", name=HARMFULQA_CONFIG, split="train", revision=revision)
    ds = ds.map(lambda _row, idx: {"_source_index": idx}, with_indices=True)
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    records = [_harmfulqa_record(row["_source_index"], row["question"]) for row in ds]
    log.info(f"Loaded HarmfulQA: {len(records)} prompts")
    return records


def _default_harmfulqa_manifest_path() -> Path:
    """`manifests/harmfulqa_v1.json` relative to the repository root, independent of the
    caller's current working directory."""
    return Path(__file__).resolve().parents[2] / "manifests" / "harmfulqa_v1.json"


def load_harmfulqa_partition(
    partition: str, manifest_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Load one frozen HarmfulQA partition exactly as fixed by the manifest.

    No `n`/`seed`: membership and order are entirely determined by the manifest. Every
    returned prompt is proven, at load time, against the pinned dataset revision and the
    manifest's own hash and lineage -- not merely trusted to still match.
    """
    if partition not in VALID_HARMFULQA_PARTITIONS:
        raise ValueError(f"partition must be one of {VALID_HARMFULQA_PARTITIONS}, got {partition!r}")

    path = Path(manifest_path) if manifest_path is not None else _default_harmfulqa_manifest_path()
    manifest = load_manifest(path)
    validate_manifest_identity(manifest)

    dataset = manifest["dataset"]
    ds = load_dataset(
        dataset["repository"], name=dataset["config"], split=dataset["split"], revision=dataset["revision"],
    )
    raw_records = [_harmfulqa_record(i, row["question"]) for i, row in enumerate(ds)]
    validate_source_binding(manifest, raw_records)

    raw_by_id = {r["source_id"]: r for r in raw_records}
    selected = [
        {
            "id": entry["source_id"],
            "source_id": entry["source_id"],
            "source_index": raw_by_id[entry["source_id"]]["source_index"],
            "prompt": raw_by_id[entry["source_id"]]["prompt"],
            "prompt_hash": entry["prompt_hash"],
            "partition": entry["partition"],
            "permuted_position": entry["permuted_position"],
            "manifest_hash": manifest["manifest_hash"],
        }
        for entry in manifest["records"]
        if entry["partition"] == partition
    ]
    selected.sort(key=lambda r: r["permuted_position"])
    log.info(f"Loaded HarmfulQA partition {partition!r}: {len(selected)} prompts")
    return selected


def load_advbench(n: Optional[int] = None, seed: int = 42) -> List[Dict[str, str]]:
    """Load AdvBench harmful behaviors (used as prompts; ignore the target strings)."""
    ds = load_dataset("walledai/AdvBench", split="train")
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    records = []
    for i, row in enumerate(ds):
        records.append({
            "id": f"advbench-{i}",
            "prompt": row["prompt"],
        })
    log.info(f"Loaded AdvBench: {len(records)} prompts")
    return records


# Helpers

def _split_hh_conversation(text: str):
    """Split an HH-RLHF transcript into (prompt_messages, final_assistant_response).

    The prompt is returned as a list of {"role", "content"} dicts (chat-template ready).
    """
    if not text:
        return None, None

    # HH transcripts look like: "\n\nHuman: ... \n\nAssistant: ... \n\nHuman: ... \n\nAssistant: ..."
    parts = text.split("\n\n")
    messages = []
    last_assistant = None
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("Human:"):
            messages.append({"role": "user", "content": part[len("Human:"):].strip()})
        elif part.startswith("Assistant:"):
            content = part[len("Assistant:"):].strip()
            messages.append({"role": "assistant", "content": content})
            last_assistant = content

    if not messages or messages[-1]["role"] != "assistant":
        return None, None

    # Strip the final assistant turn — that's our reference answer.
    prompt_messages = messages[:-1]
    return prompt_messages, last_assistant
