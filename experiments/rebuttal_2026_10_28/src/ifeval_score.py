"""Official IFEval strict and loose prompt-level checks, CPU.

Uses the official checkers from google-research/google-research @ e6890f85
(instruction_following_eval/), vendored unchanged after review. One deviation, for
determinism: langdetect is randomised unless seeded, so DetectorFactory.seed is fixed to 0
before any check runs. Accuracy is the checker's verdict, never an LLM impression.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor" / "ifeval_official"))
os.environ.setdefault("NLTK_DATA", str(ROOT / ".cache" / "nltk_data"))

import langdetect  # noqa: E402
import nltk  # noqa: E402

nltk.data.path.insert(0, os.environ["NLTK_DATA"])
langdetect.DetectorFactory.seed = 0

from instruction_following_eval import evaluation_lib as ev  # noqa: E402

CHECKER_REF = "google-research/google-research@e6890f85757dd84e27ca6df2dd30651dafad28e0"
DEVIATIONS = ["langdetect.DetectorFactory.seed = 0 for deterministic language detection"]


def score(row: dict, response: str) -> dict:
    inp = ev.InputExample(key=row["key"], instruction_id_list=row["instruction_id_list"],
                          prompt=row["prompt"], kwargs=row["kwargs"])
    s = ev.test_instruction_following_strict(inp, {row["prompt"]: response})
    l = ev.test_instruction_following_loose(inp, {row["prompt"]: response})
    return {"strict_prompt_level": bool(s.follow_all_instructions),
            "loose_prompt_level": bool(l.follow_all_instructions),
            "strict_instruction_level": [bool(x) for x in s.follow_instruction_list],
            "loose_instruction_level": [bool(x) for x in l.follow_instruction_list],
            "instruction_id_list": list(row["instruction_id_list"])}
