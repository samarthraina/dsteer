"""Config loading. Loads YAML files into typed dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .utils import load_yaml


@dataclass
class ModelConfig:
    """Specification of an IT/DPO model pair."""

    name: str                          # short identifier, e.g. "llama3-oh"
    base_model: str                    # HF base model ID, e.g. "meta-llama/Meta-Llama-3-8B"
    it_model: str                      # HF path to IT (instruction-tuned) checkpoint
    dpo_model: str                     # HF path to DPO checkpoint
    architecture: str                  # "llama" | "mistral" | "qwen"
    num_layers: int                    # transformer layer count (e.g. 32 for 8B Llama)
    tokenizer_id: Optional[str] = None  # if None, use it_model's tokenizer

    it_subfolder: Optional[str] = None
    dpo_subfolder: Optional[str] = None
    tokenizer_subfolder: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ModelConfig":
        return cls(**load_yaml(path))


@dataclass
class JudgeConfig:
    """Settings for the LLM-as-judge (Qwen 32B served via vLLM)."""

    model_name: str = "Qwen/Qwen2.5-32B-Instruct"
    server_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    max_tokens: int = 1024
    temperature: float = 0.0
    timeout: float = 180.0

    # Deprecated (Task 005): G-Eval probability weighting used to re-weight the emitted
    # score by token probabilities (Liu et al. 2023 sec. 2.3). The protocol (Section 10)
    # freezes the judge's emitted discrete integer as the sole authoritative score, so
    # `Judge` now refuses to construct at all if `use_logprobs=True` -- kept here only so
    # a historical config setting it fails with a clear error instead of silently
    # re-enabling the old estimator.
    use_logprobs: bool = False
    top_logprobs: int = 20  # unused; the judge request never asks for logprobs anymore
    max_score: int = 10  # frozen at 10 for this protocol path; any other value fails clearly


@dataclass
class ITEvalConfig:
    """Settings for the IT capability evaluation."""

    # Sample sizes (None = use all)
    ifeval_n: Optional[int] = None
    alpaca_n: int = 50

    # Generation
    max_new_tokens: int = 512
    batch_size: int = 8

    # Outputs
    output_dir: str = "outputs/it_eval"

    # Judge config (nested)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ITEvalConfig":
        raw = load_yaml(path)
        judge_raw = raw.pop("judge", {})
        return cls(judge=JudgeConfig(**judge_raw), **raw)
