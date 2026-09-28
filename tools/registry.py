"""Tool registry.

Decorator-based so adding a tool is one function. Every tool declares a JSON
Schema the LLM sees, and the registry is what the ToolCallGuard validates
against. Tables and columns are allow-listed: the agent can never express a
query outside this surface.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from core.schemas import Evidence

ALLOWED_TABLES = {"fact_daily", "business_events", "log_signatures", "config_changes"}
MAX_ROWS = 500
MAX_CELLS = 200_000

Handler = Callable[..., Evidence | dict[str, Any]]


def stable_evidence_id(prefix: str, **parts: Any) -> str:
    """Deterministic evidence id.

    Two requirements, both learned the hard way:
      1. It must be stable across processes. Python's built-in hash() is salted
         per interpreter, so ids built from it cannot be resolved by a later
         run reading the same trace, which defeats the audit trail entirely.
      2. It must vary with every argument that changes the answer, otherwise a
         second call with a different dimension silently collides with the first
         and the new evidence is discarded as a duplicate.
    """
    canonical = "|".join(f"{key}={parts[key]!r}" for key in sorted(parts))
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:10]
    return f"EV-{prefix}-{digest}"


@dataclass
class RegisteredTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler
    cost_weight: int = 1


@dataclass
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    result: Evidence | None
    error: str | None
    duration_ms: int
    cache_hit: bool = False


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._cache: dict[str, Evidence] = {}
        self.history: list[ToolCallRecord] = []

    def register(
        self, name: str, description: str, parameters: dict[str, Any], cost_weight: int = 1
    ) -> Callable[[Handler], Handler]:
        def decorator(fn: Handler) -> Handler:
            self._tools[name] = RegisteredTool(
                name=name,
                description=description,
                parameters=parameters,
                handler=fn,
                cost_weight=cost_weight,
            )
            return fn

        return decorator

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def openai_schemas(self, subset: list[str] | None = None) -> list[dict[str, Any]]:
        chosen = subset or self.names()
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for name in chosen
            if (t := self._tools.get(name)) is not None
        ]

    def call_key(self, name: str, args: dict[str, Any]) -> str:
        canonical = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(f"{name}:{canonical}".encode()).hexdigest()[:24]

    def invoke(self, name: str, args: dict[str, Any], use_cache: bool = True) -> Evidence:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"unknown tool: {name}")

        key = self.call_key(name, args)
        if use_cache and key in self._cache:
            cached = self._cache[key]
            self.history.append(
                ToolCallRecord(name, args, cached, None, 0, cache_hit=True)
            )
            return cached

        result = tool.handler(**args)
        evidence = result if isinstance(result, Evidence) else Evidence(**result)
        self._cache[key] = evidence
        self.history.append(ToolCallRecord(name, args, evidence, None, 0))
        return evidence


registry = ToolRegistry()
tool = registry.register


def guard_sql(where_clause: str) -> str:
    """Defence in depth: reject anything that is not a plain filtered SELECT."""
    lowered = f" {where_clause.strip().lower()} "
    forbidden = (
        " insert ",
        " update ",
        " delete ",
        " drop ",
        " create ",
        " alter ",
        " attach ",
        " copy ",
        " pragma ",
        " install ",
        " load ",
        ";",
    )
    if not lowered.startswith(" select ") and not lowered.startswith(" with "):
        raise ValueError("only SELECT statements are permitted")
    for token in forbidden:
        if token in lowered:
            raise ValueError(f"forbidden token in query: {token.strip()}")
    return where_clause


def clamp_rows(df: pd.DataFrame, limit: int = MAX_ROWS) -> pd.DataFrame:
    return df.head(limit) if len(df) > limit else df


def to_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(v) for v in value]
    if hasattr(value, "item") and hasattr(value, "dtype"):
        return value.item()
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value
