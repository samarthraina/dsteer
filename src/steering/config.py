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
    """Settings for the LLM-as-judge (served via vLLM)."""

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

    # Task 010: frozen Qwen3.5 confirmatory-protocol sampling/seed fields (protocol
    # Section 10). `None` means "not set -- do not send this field", which is what
    # every pre-existing (legacy Qwen2.5) config still does, so a plain `JudgeConfig()`
    # sends exactly the request shape it always has. Use `frozen_qwen35()` to build a
    # config with all of these pinned to the frozen identity.
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    enable_thinking: Optional[bool] = None
    seed: Optional[int] = None
    seed_derivation_version: Optional[int] = None

    # Task 010 correction: the frozen three-total-attempt limit, bound through the
    # config so a confirmatory caller cannot silently exceed it by passing a larger
    # `Judge.score(max_retries=...)`. `None` (the default) means no limit is enforced --
    # every pre-existing call site is unaffected.
    max_attempts: Optional[int] = None

    # Task 010 correction: on the confirmatory path, `Judge.score` parses only a direct
    # `json.loads` of the raw completion -- no markdown-fence stripping, no
    # brace-scanning through surrounding prose. `None` (the default) keeps the existing
    # lenient multi-strategy extraction for every pre-existing (legacy) call site.
    strict_response_parsing: Optional[bool] = None

    @classmethod
    def frozen_qwen35(cls, server_url: str, api_key: str = "EMPTY") -> "JudgeConfig":
        """A `JudgeConfig` fully pinned to the frozen Qwen3.5 confirmatory protocol
        (`Brain/EXPERIMENT_PROTOCOL_V1.md` Section 10). Every sampling/seed/limit value
        comes from `judge_identity.FROZEN_JUDGE_IDENTITY` -- the single source of
        truth, so this can never drift from what `manifests/judge_protocol_v1.json` was
        built against. `model_name` is the revision-bearing served alias, never the
        bare mutable repository name.
        """
        from .judge_identity import FROZEN_JUDGE_IDENTITY

        f = FROZEN_JUDGE_IDENTITY
        return cls(
            model_name=f["judge"]["served_model_alias"],
            server_url=server_url, api_key=api_key,
            max_tokens=f["max_response_tokens"], temperature=f["sampling"]["temperature"],
            top_p=f["sampling"]["top_p"], top_k=f["sampling"]["top_k"],
            min_p=f["sampling"]["min_p"], presence_penalty=f["sampling"]["presence_penalty"],
            repetition_penalty=f["sampling"]["repetition_penalty"],
            enable_thinking=f["judge"]["thinking_enabled"], seed=f["global_seed"],
            seed_derivation_version=f["seed_derivation_version"],
            max_attempts=f["max_attempts"], strict_response_parsing=True,
        )


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
