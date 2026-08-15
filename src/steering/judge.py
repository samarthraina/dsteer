"""Custom LLM-as-judge client for vLLM. Replaces the deepeval dependency.

Connects to a vLLM server running Qwen2.5-32B-Instruct (or similar) and returns parsed
JSON scores. Handles retries, JSON extraction, and score normalization to [0, 1].

The judge-emitted discrete integer (0-10) is the sole authoritative score (protocol
Section 10). A JSON `score` is accepted only when it is a genuine JSON integer in that
range -- never a string, a float, a bool, or anything out of range -- and never clamped,
rounded, or coerced into range. An earlier version of this module re-weighted the
emitted integer by the token-probability distribution at its position (G-Eval, Liu et
al. 2023 sec. 2.3), to recover a continuous metric from a judge that concentrates its
mass on a few band anchors at temperature 0. That estimator is ambiguous for
multi-token scores and is not the outcome the protocol froze, so it has been removed
from the scoring path entirely: judge requests no longer ask for logprobs, and
`Judge.__init__` refuses to construct if a config asks for it.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openai import OpenAI

from .config import JudgeConfig

log = logging.getLogger(__name__)

#: The frozen scale (protocol Section 10). Not configurable per instance -- see
#: `Judge.__init__`, which rejects any `JudgeConfig.max_score != PROTOCOL_MAX_SCORE`.
PROTOCOL_MAX_SCORE = 10


@dataclass
class JudgeResponse:
    """One judge response.

    score:          the emitted integer normalised to [0, 1] (discrete_score / 10.0),
                     or None on abstention -- the sole authoritative value.
    discrete_score: the raw integer 0-10 the judge emitted, or None on abstention.
    raw_score:      deprecated compatibility alias, always equal to `score`.
    weighted:       deprecated; always False. No active path selects a weighted value.
    """
    score: Optional[float]
    reason: str
    raw: str
    error: Optional[str] = None
    discrete_score: Optional[int] = None
    raw_score: Optional[float] = None
    weighted: bool = False


class Judge:
    """Synchronous wrapper around a vLLM OpenAI-compatible chat endpoint.

    Use:
        judge = Judge(JudgeConfig())
        result = judge.score(prompt="Rate this response on a 0-10 scale...", input_text=..., output_text=...)
    """

    SYSTEM_PROMPT = (
        "You are a careful evaluator. Read the criteria, examine the output, "
        "and return a single JSON object with two keys: "
        '"score" (an integer from 0 to 10 inclusive) and "reason" (a short explanation). '
        "Return only the JSON object, no preamble, no markdown fencing."
    )

    def __init__(self, cfg: JudgeConfig):
        if cfg.use_logprobs:
            raise ValueError(
                "JudgeConfig.use_logprobs=True is no longer supported: probability "
                "weighting was removed from the scoring path (protocol Section 10 makes "
                "the judge's emitted discrete integer the sole authoritative score). "
                "Set use_logprobs=False."
            )
        if cfg.max_score != PROTOCOL_MAX_SCORE:
            raise ValueError(
                f"JudgeConfig.max_score={cfg.max_score} is not supported: this protocol "
                f"path freezes the scale at 0-{PROTOCOL_MAX_SCORE}."
            )
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.server_url, api_key=cfg.api_key)
        log.info(f"Judge connected to {cfg.server_url} (model={cfg.model_name})")

    def score(
        self,
        rubric: str,
        input_text: Optional[str] = None,
        output_text: str = "",
        reference_text: Optional[str] = None,
        max_retries: int = 2,
    ) -> JudgeResponse:
        """Score a single output against a rubric.

        Args:
            rubric: the scoring criteria (a full text description of what to score)
            input_text: optional input/prompt that the output is responding to
            output_text: the model output being evaluated (required)
            reference_text: optional reference (e.g. for behavioral comparison)
            max_retries: number of retry attempts on parse failures
        """
        user_msg = self._build_user_message(rubric, input_text, output_text, reference_text)

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = self._call(user_msg)
                raw = response.choices[0].message.content.strip()
                parsed = self._extract_json(raw)
                if parsed is None:
                    last_error = f"could not extract JSON from: {raw[:200]!r}"
                    continue
                score_raw = parsed.get("score")
                reason = parsed.get("reason", "")
                if score_raw is None:
                    last_error = f"no 'score' key in JSON: {parsed!r}"
                    continue

                discrete = self._validate_discrete_score(score_raw)
                if discrete is None:
                    last_error = (
                        f"score must be a JSON integer 0-{PROTOCOL_MAX_SCORE} inclusive "
                        f"(not a string, float, or bool); got {score_raw!r}"
                    )
                    continue

                normalized = discrete / float(PROTOCOL_MAX_SCORE)
                return JudgeResponse(
                    score=normalized,
                    reason=str(reason),
                    raw=raw,
                    discrete_score=discrete,
                    raw_score=normalized,
                    weighted=False,
                )
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                log.warning(f"Judge call failed (attempt {attempt + 1}): {last_error}")
                if attempt < max_retries:
                    time.sleep(1.0 + attempt)

        return JudgeResponse(score=None, reason="", raw="", error=last_error)

    @staticmethod
    def _validate_discrete_score(value: Any) -> Optional[int]:
        """Accept only a genuine JSON integer 0-10 inclusive.

        `bool` is rejected even though it subclasses `int` in Python -- `true`/`false`
        are not scores. Strings (`"7"`), floats (`7.0`, `7.5`), and out-of-range
        integers are all rejected, never coerced or clamped into range: an invalid
        score is a reason to retry or abstain, not a number to guess at.
        """
        if isinstance(value, bool):
            return None
        if not isinstance(value, int):
            return None
        if not 0 <= value <= PROTOCOL_MAX_SCORE:
            return None
        return value

    # internals

    def _build_user_message(
        self,
        rubric: str,
        input_text: Optional[str],
        output_text: str,
        reference_text: Optional[str],
    ) -> str:
        parts = [f"CRITERIA:\n{rubric}\n"]
        if input_text is not None:
            parts.append(f"INPUT:\n{input_text}\n")
        parts.append(f"OUTPUT:\n{output_text}\n")
        if reference_text is not None:
            parts.append(f"REFERENCE:\n{reference_text}\n")
        parts.append('Return JSON: {"score": <0-10 integer>, "reason": "<one sentence>"}')
        return "\n".join(parts)

    def _call(self, user_msg: str):
        """Returns the full completion object. Never requests logprobs: `__init__`
        already refused to construct this instance if `cfg.use_logprobs` were true."""
        kwargs: Dict[str, Any] = {
            "model": self.cfg.model_name,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "timeout": self.cfg.timeout,
        }
        return self.client.chat.completions.create(**kwargs)

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Best-effort JSON extraction.

        Tries: (1) direct parse, (2) ```json ... ``` block, (3) first balanced {...} substring.
        """
        text = text.strip()

        # Direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Markdown code fence
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # First balanced { ... } substring
        depth = 0
        start = -1
        for i, c in enumerate(text):
            if c == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        start = -1
                        continue

        return None
