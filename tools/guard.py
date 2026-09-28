"""ToolCallGuard.

Every tool call the model emits passes through here first. Each row of the
robustness table is a distinct, testable code path, and every intervention is
recorded on the trace so the UI and the audit log can show what went wrong.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.schemas import Evidence, ToolError
from tools.registry import ToolRegistry, registry as default_registry


class GuardAction(str, Enum):
    EXECUTED = "executed"
    CACHE_HIT = "cache_hit"
    REPAIRED_ARGUMENTS = "repaired_arguments"
    REJECTED_UNKNOWN_TOOL = "rejected_unknown_tool"
    REJECTED_BAD_ARGUMENTS = "rejected_bad_arguments"
    REPAIRED_JSON = "repaired_json"
    BLOCKED_LOOP = "blocked_loop"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass
class GuardEvent:
    action: GuardAction
    tool_name: str
    detail: str
    raw_arguments: str = ""
    recovered: bool = False


@dataclass
class GuardOutcome:
    ok: bool
    evidence: Evidence | None
    events: list[GuardEvent] = field(default_factory=list)
    tool_error: ToolError | None = None
    duration_ms: int = 0

    @property
    def actions(self) -> list[str]:
        return [e.action.value for e in self.events]


def _repair_json(raw: str) -> dict[str, Any] | None:
    """One automatic repair attempt for the most common LLM argument defects."""
    if not raw:
        return None
    candidate = raw.strip()

    for attempt in (
        candidate,
        candidate.replace("'", '"'),
        candidate.replace("\n", " ").replace("  ", " "),
    ):
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(candidate[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def _coerce_types(args: dict[str, Any], schema: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    coerced: dict[str, Any] = {}
    fixes: list[str] = []
    props = schema.get("properties", {})

    for key, value in args.items():
        spec = props.get(key)
        if spec is None:
            fixes.append(f"dropped unknown parameter '{key}'")
            continue
        expected = spec.get("type")
        if expected == "string" and not isinstance(value, str):
            coerced[key] = str(value)
            fixes.append(f"coerced '{key}' to string")
        elif expected == "integer" and isinstance(value, (int, float)):
            coerced[key] = int(value)
            fixes.append(f"coerced '{key}' to integer")
        elif expected == "number" and isinstance(value, (int, float)):
            coerced[key] = float(value)
        elif expected == "boolean" and not isinstance(value, bool):
            coerced[key] = bool(value)
            fixes.append(f"coerced '{key}' to boolean")
        elif expected == "array" and isinstance(value, str):
            coerced[key] = [v.strip() for v in value.split(",") if v.strip()]
            fixes.append(f"coerced '{key}' to array")
        else:
            coerced[key] = value

    for required in schema.get("required", []):
        if required not in coerced:
            fixes.append(f"missing required parameter '{required}'")

    return coerced, fixes


class ToolCallGuard:
    def __init__(
        self,
        registry: ToolRegistry | None = None,
        max_repeat_calls: int = 3,
        tool_timeout_s: float = 20.0,
        max_calls: int = 15,
    ) -> None:
        self.registry = registry or default_registry
        self.max_repeat_calls = max_repeat_calls
        self.tool_timeout_s = tool_timeout_s
        self.max_calls = max_calls
        self._call_counts: dict[str, int] = {}
        self.events: list[GuardEvent] = []
        self.calls_used = 0

    def _record(self, event: GuardEvent) -> None:
        self.events.append(event)

    def call_budget_remaining(self) -> int:
        return max(0, self.max_calls - self.calls_used)

    def handle(
        self,
        tool_name: str,
        raw_arguments: str | dict[str, Any] | None,
        use_cache: bool = True,
    ) -> GuardOutcome:
        started = time.perf_counter()
        events: list[GuardEvent] = []

        def finish(outcome: GuardOutcome) -> GuardOutcome:
            outcome.events = events + outcome.events
            outcome.duration_ms = int((time.perf_counter() - started) * 1000)
            self.events.extend(outcome.events)
            return outcome

        if self.call_budget_remaining() <= 0:
            event = GuardEvent(
                GuardAction.BUDGET_EXHAUSTED,
                tool_name,
                f"tool call budget of {self.max_calls} is exhausted",
            )
            events.append(event)
            return finish(
                GuardOutcome(
                    ok=False,
                    evidence=None,
                    tool_error=ToolError(
                        error="budget_exhausted",
                        detail=f"No tool calls left (max {self.max_calls}).",
                        retryable=False,
                        hint="Finish the investigation with the evidence already gathered.",
                    ),
                )
            )

        registered = self.registry.get(tool_name)
        if registered is None:
            event = GuardEvent(
                GuardAction.REJECTED_UNKNOWN_TOOL,
                tool_name,
                f"unknown tool; available: {', '.join(self.registry.names())}",
            )
            events.append(event)
            return finish(
                GuardOutcome(
                    ok=False,
                    evidence=None,
                    tool_error=ToolError(
                        error="unknown_tool",
                        detail=f"'{tool_name}' is not a registered tool.",
                        retryable=True,
                        hint=f"Call one of: {', '.join(self.registry.names())}.",
                    ),
                )
            )

        parsed: dict[str, Any] | None
        if isinstance(raw_arguments, dict):
            parsed = raw_arguments
        else:
            raw_text = str(raw_arguments or "").strip()
            try:
                strict = json.loads(raw_text)
                parsed = strict if isinstance(strict, dict) else None
            except json.JSONDecodeError:
                parsed = None
            if parsed is None:
                parsed = _repair_json(raw_text)
                if parsed is None:
                    event = GuardEvent(
                        GuardAction.REPAIRED_JSON,
                        tool_name,
                        "could not repair malformed JSON arguments",
                        raw_arguments=raw_text[:300],
                    )
                    events.append(event)
                    return finish(
                        GuardOutcome(
                            ok=False,
                            evidence=None,
                            tool_error=ToolError(
                                error="malformed_json",
                                detail=f"Arguments were not valid JSON: {raw_text[:200]}",
                                retryable=True,
                                hint="Re-issue the call with a valid JSON object matching the schema.",
                            ),
                        )
                    )
                events.append(
                    GuardEvent(
                        GuardAction.REPAIRED_JSON,
                        tool_name,
                        "repaired malformed JSON arguments",
                        raw_arguments=raw_text[:300],
                        recovered=True,
                    )
                )

        args, fixes = _coerce_types(parsed, registered.parameters)
        hard_fail = [f for f in fixes if f.startswith("missing required")]
        if hard_fail:
            event = GuardEvent(
                GuardAction.REJECTED_BAD_ARGUMENTS,
                tool_name,
                "; ".join(hard_fail),
                raw_arguments=json.dumps(parsed)[:300],
            )
            events.append(event)
            return finish(
                GuardOutcome(
                    ok=False,
                    evidence=None,
                    tool_error=ToolError(
                        error="invalid_arguments",
                        detail="; ".join(fixes),
                        retryable=True,
                        hint=f"Required: {', '.join(registered.parameters.get('required', []))}.",
                    ),
                )
            )
        if fixes:
            events.append(
                GuardEvent(
                    GuardAction.REPAIRED_ARGUMENTS,
                    tool_name,
                    "; ".join(fixes),
                    raw_arguments=json.dumps(parsed)[:300],
                    recovered=True,
                )
            )

        signature = self.registry.call_key(tool_name, args)
        self._call_counts[signature] = self._call_counts.get(signature, 0) + 1
        if self._call_counts[signature] > self.max_repeat_calls:
            event = GuardEvent(
                GuardAction.BLOCKED_LOOP,
                tool_name,
                f"identical call repeated {self._call_counts[signature] - 1} times",
                raw_arguments=json.dumps(args)[:300],
            )
            events.append(event)
            return finish(
                GuardOutcome(
                    ok=False,
                    evidence=None,
                    tool_error=ToolError(
                        error="loop_detected",
                        detail="This exact call has already been made several times.",
                        retryable=False,
                        hint="Change the dimension, window or metric, or conclude the investigation.",
                    ),
                )
            )

        try:
            history_before = len(self.registry.history)
            evidence = self.registry.invoke(tool_name, args, use_cache=use_cache)
            was_cached = (
                len(self.registry.history) > history_before
                and self.registry.history[-1].cache_hit
            )
        except KeyError as exc:
            event = GuardEvent(GuardAction.FAILED, tool_name, str(exc))
            events.append(event)
            return finish(
                GuardOutcome(
                    ok=False,
                    evidence=None,
                    tool_error=ToolError(
                        error="unknown_tool", detail=str(exc), retryable=False
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            event = GuardEvent(GuardAction.FAILED, tool_name, f"{type(exc).__name__}: {exc}")
            events.append(event)
            return finish(
                GuardOutcome(
                    ok=False,
                    evidence=None,
                    tool_error=ToolError(
                        error="tool_execution_failed",
                        detail=f"{type(exc).__name__}: {exc}",
                        retryable=True,
                        hint="Check the parameter values against the schema and try again.",
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001
            event = GuardEvent(GuardAction.FAILED, tool_name, f"unexpected: {exc}")
            events.append(event)
            return finish(
                GuardOutcome(
                    ok=False,
                    evidence=None,
                    tool_error=ToolError(
                        error="tool_execution_failed",
                        detail=f"{type(exc).__name__}: {exc}",
                        retryable=True,
                    ),
                )
            )

        self.calls_used += 1
        if was_cached:
            events.append(GuardEvent(GuardAction.CACHE_HIT, tool_name, "served from cache"))
        else:
            events.append(
                GuardEvent(GuardAction.EXECUTED, tool_name, evidence.finding[:160])
            )

        return finish(GuardOutcome(ok=True, evidence=evidence))
