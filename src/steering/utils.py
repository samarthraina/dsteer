"""Shared utilities: random seeds, IO helpers, logging."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import numpy as np
import torch
import yaml


def set_all_seeds(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and PyTorch (CPU + CUDA).

    Call this at the top of every script that uses randomness.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Hash-based randomness in Python's str hashing — only affects multi-process
    # but cheap to set.
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logging(log_path: Union[str, Path, None] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure root logger. Writes to stdout and optionally to a file."""
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("dsteer")


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], path: Union[str, Path]) -> None:
    """Write an iterable of dicts to a JSONL file. Creates parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(record: Dict[str, Any], path: Union[str, Path]) -> None:
    """Append a single record to a JSONL file. Used for resume-friendly writes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_device(preference: str = "auto") -> str:
    """Pick a compute device for the analysis scripts.

    The per-layer decompositions are the slow part of the geometry pipeline -- a few
    hundred SVDs of an (n_prompts x hidden) matrix -- and on CPU they dominate the
    wall time of a run whose GPU work took minutes. cuSOLVER turns that into seconds.
    """
    if preference != "auto":
        return preference
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 -- torch missing is a valid CPU-only setup
        return "cpu"


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a YAML file into a dict."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
