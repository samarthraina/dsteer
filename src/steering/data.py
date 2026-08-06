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
from typing import Dict, List, Optional

from datasets import load_dataset

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

    Loads directly from the upstream JSON file because newer versions of the
    `datasets` library refuse to run the dataset's loading script.
    """
    import json
    import random
    import urllib.request

    url = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json"
    with urllib.request.urlopen(url) as resp:
        items = json.load(resp)

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


def load_harmfulqa(n: Optional[int] = None, seed: int = 42) -> List[Dict[str, str]]:
    """Load HarmfulQA prompts (non-optimized adversarial questions)."""
    ds = load_dataset("declare-lab/HarmfulQA", split="train")
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    records = []
    for i, row in enumerate(ds):
        records.append({
            "id": f"harmfulqa-{i}",
            "prompt": row["question"],
        })
    log.info(f"Loaded HarmfulQA: {len(records)} prompts")
    return records


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
