"""Groq client with model fallback, retries and token accounting.

The rest of the app depends on `complete` and `complete_json`, never on the SDK,
so swapping providers is a one-file change.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from core.config import GroqConfig
from tools.guard import _repair_json

log = logging.getLogger("agent.llm")

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_RETRY_IN = re.compile(r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE)


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "ratelimit" in type(exc).__name__.lower() or "429" in text or "rate limit" in text


def _retry_delay(exc: Exception, cap: float = 60.0) -> float:
    """Read the wait the provider asked for, else back off exponentially."""
    match = _RETRY_IN.search(str(exc))
    if match:
        try:
            return min(cap, float(match.group(1)) + 1.0)
        except ValueError:
            pass
    return min(cap, 5.0)


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict[str, Any]]
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    latency_ms: int = 0
    error: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class TokenMeter:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    errors: int = 0
    by_model: dict[str, int] = field(default_factory=dict)

    def record(self, response: LLMResponse) -> None:
        self.calls += 1
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens
        self.by_model[response.model] = (
            self.by_model.get(response.model, 0) + response.total_tokens
        )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class GroqClient:
    def __init__(self, config: GroqConfig, meter: TokenMeter | None = None) -> None:
        if not config.configured:
            raise LLMError(
                "GROQ_API_KEY is not configured. Add it to .env before running."
            )
        from groq import Groq

        self.config = config
        self.meter = meter or TokenMeter()
        self._client = Groq(api_key=config.api_key, timeout=config.timeout_s, max_retries=0)
        self._degraded = False

    @property
    def model(self) -> str:
        if self._degraded:
            return self.config.model_fallback
        return self.config.model_primary

    def _next_model(self) -> str:
        """After a primary failure, fall back permanently for the run."""
        self._degraded = True
        return self.config.model_fallback

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        max_tokens: int = 1400,
        temperature: float = 0.1,
    ) -> LLMResponse:
        started = time.perf_counter()
        attempt_model = self.model
        last_error = ""

        for attempt in range(self.config.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": attempt_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = tool_choice or "auto"
                completion = self._client.chat.completions.create(**kwargs)
                choice = completion.choices[0]
                message = choice.message
                usage = completion.usage

                raw_tool_calls: list[Any] = list(getattr(message, "tool_calls", None) or [])
                parsed_calls = [
                    {
                        "id": getattr(call, "id", f"call_{i}"),
                        "name": call.function.name,
                        "arguments": call.function.arguments or "{}",
                    }
                    for i, call in enumerate(raw_tool_calls)
                ]

                response = LLMResponse(
                    content=(message.content or "").strip(),
                    tool_calls=parsed_calls,
                    model=attempt_model,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    finish_reason=choice.finish_reason or "",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
                self.meter.record(response)
                return response

            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                message = str(exc).lower()

                # A token/minute limit is an ORGANISATION-wide limit: every
                # model shares it, so falling back to a second model just burns
                # the same budget. Honour the requested delay instead.
                if _is_rate_limit(exc):
                    delay = _retry_delay(exc)
                    log.warning(
                        "rate limited (attempt %s/%s, model=%s): waiting %.1fs",
                        attempt + 1,
                        self.config.max_retries,
                        attempt_model,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                log.warning(
                    "llm call failed (attempt %s/%s, model=%s): %s",
                    attempt + 1,
                    self.config.max_retries,
                    attempt_model,
                    last_error[:300],
                )
                if attempt_model != self.config.model_fallback:
                    attempt_model = self._next_model()
                    log.info("falling back to model %s", attempt_model)
                else:
                    time.sleep(min(2**attempt, 8))

        response = LLMResponse(
            content="",
            tool_calls=[],
            model=attempt_model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            error=last_error,
        )
        self.meter.errors += 1
        self.meter.record(response)
        return response

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        schema_hint: str = "",
        max_tokens: int = 1600,
        temperature: float = 0.1,
    ) -> tuple[dict[str, Any] | None, str, str]:
        """Ask for JSON and parse defensively.

        Returns (parsed_or_None, raw_text, finish_reason) so the caller can
        distinguish a truncation from a genuine syntax error and decide whether
        to retry rather than having a silent parse failure deep in the pipeline.
        """
        instruction = (
            "Respond with a single valid JSON object and nothing else. "
            "No prose, no markdown fences, no trailing commentary."
        )
        if schema_hint:
            instruction += f"\nShape: {schema_hint}"

        enriched = list(messages) + [
            {"role": "system", "content": instruction},
        ]
        response = self.complete(enriched, max_tokens=max_tokens, temperature=temperature)
        parsed = extract_json(response.content)
        return parsed, response.content, response.finish_reason


def extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    candidate = text.strip()

    direct = _repair_json(candidate)
    if direct is not None:
        return direct

    fenced = _JSON_FENCE.search(candidate)
    if fenced:
        return _repair_json(fenced.group(1))

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end > start:
        return _repair_json(candidate[start : end + 1])
    return None
