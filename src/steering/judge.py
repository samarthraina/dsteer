"""Custom LLM-as-judge client for vLLM. Replaces the deepeval dependency.

Connects to a vLLM server running Qwen2.5-32B-Instruct (or similar) and
returns parsed JSON scores. Handles retries, JSON extraction, and
score normalization to [0, 1].

Scores are probability-weighted where the server returns logprobs. A judge asked for an
integer at temperature 0 concentrates on a few band anchors -- in our runs one metric put
87.5% of its mass on a single value -- which leaves a nominally 11-point scale behaving
like a 3-point one. G-Eval (Liu et al., 2023) handles this by weighting the score by the
token probabilities, sum(p(s) * s), and that is what `_weighted_score` does. The raw
integer is kept alongside so the two can be compared.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openai import OpenAI

from .config import JudgeConfig

log = logging.getLogger(__name__)


def _is_ascii_digits(text: str) -> bool:
    """True only for plain 0-9.

    str.isdigit() is not this test. It accepts superscripts and subscripts -- the judge's
    alternatives at a score position include tokens like the subscript one -- and int()
    then raises on them. That took out 86% of one metric's scores as failed calls before
    it was caught, which is the failure mode this whole path is meant to avoid: a
    silently unscored record looks the same as a judge that declined.
    """
    return bool(text) and text.isascii() and text.isdigit()


@dataclass
class JudgeResponse:
    """One judge response.

    score:     value in [0, 1] -- probability-weighted when logprobs were available
    raw_score: the integer the judge emitted, normalised to [0, 1], always populated
    weighted:  whether `score` used probability weighting
    """
    score: Optional[float]
    reason: str
    raw: str
    error: Optional[str] = None
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

                def norm(v: float) -> float:
                    return max(0.0, min(1.0, v / self.cfg.max_score))

                raw_score = norm(float(score_raw))

                weighted = None
                if self.cfg.use_logprobs:
                    weighted = self._weighted_score(score_raw, response)

                return JudgeResponse(
                    score=raw_score if weighted is None else norm(weighted),
                    reason=str(reason),
                    raw=raw,
                    raw_score=raw_score,
                    weighted=weighted is not None,
                )
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                log.warning(f"Judge call failed (attempt {attempt + 1}): {last_error}")
                if attempt < max_retries:
                    time.sleep(1.0 + attempt)

        return JudgeResponse(score=None, reason="", raw="", error=last_error)

    def _weighted_score(self, emitted_score: Any, response: Any) -> Optional[float]:
        """G-Eval probability weighting: sum(p(s) * s) over the score token's alternatives.

        The score is located by position rather than by string match. On a 0-10 scale the
        top of the range is emitted as two tokens -- '1' then '0' -- so searching for the
        literal "10" finds nothing and the weighting silently never fires, which is what
        it did before this. Instead: walk to the first digit token, and read the
        alternatives there.

        At that position the alternatives are the model's candidate scores. For an
        emitted 10 they look like ['1', '9', '8', '7', ...], where '1' is the opening
        digit of 10 and the rest are whole single-digit scores. So an alternative equal
        to the emitted value's first digit is read as the full emitted value, and every
        other digit as itself. That holds for any scale whose only multi-digit value is
        its maximum, which covers 0-10.

        Returns None when there are no logprobs or no digit token, and the caller keeps
        the raw integer.
        """
        try:
            content = response.choices[0].logprobs.content
        except AttributeError:
            return None
        if not content:
            return None

        emitted = str(emitted_score).strip()
        if not _is_ascii_digits(emitted):
            return None

        for token_info in content:
            text = token_info.token.strip()
            if not _is_ascii_digits(text):
                continue

            # First digit token: this is the score position.
            probs: Dict[int, float] = {}
            for alt in getattr(token_info, "top_logprobs", None) or []:
                alt_text = alt.token.strip()
                if not _is_ascii_digits(alt_text):
                    continue
                # A single digit matching the emitted value's opening digit stands for
                # the whole emitted value (the '1' of '10'); anything else is itself.
                if len(emitted) > 1 and alt_text == emitted[0]:
                    value = int(emitted)
                else:
                    value = int(alt_text)
                if 0 <= value <= self.cfg.max_score:
                    # vLLM can return the same value under different token strings.
                    probs[value] = probs.get(value, 0.0) + math.exp(alt.logprob)

            total = sum(probs.values())
            if not probs or total <= 0:
                return None
            return sum(value * p for value, p in probs.items()) / total

        return None

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
        """Returns the full completion object; the caller needs logprobs as well as text."""
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
        if self.cfg.use_logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = self.cfg.top_logprobs
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
