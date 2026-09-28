"""Smoke-test the tool layer and the guard without any LLM involvement.

    python -m tests.test_tools
"""

from __future__ import annotations

import json

import tools.investigation  # noqa: F401  (import registers the tools)
from tools.guard import ToolCallGuard
from tools.registry import registry

ANOMALY = {"start": "2026-09-26", "end": "2026-09-26"}
BASELINE = {"baseline_start": "2026-08-29", "baseline_end": "2026-09-25"}


def show(guard: ToolCallGuard, title: str, name: str, args: dict) -> None:
    outcome = guard.handle(name, json.dumps(args))
    print(f"\n=== {title} ===")
    print(f"guard: {outcome.actions} ({outcome.duration_ms} ms)")
    if not outcome.ok:
        print(f"ERROR: {outcome.tool_error}")
        return
    ev = outcome.evidence
    assert ev is not None
    print(f"finding: {ev.finding}")
    print(f"confidence_hint: {ev.confidence_hint}")
    return ev


def main() -> None:
    print(f"tools: {registry.names()}")
    guard = ToolCallGuard()

    ev = show(
        guard,
        "analyze_metric daily_revenue",
        "analyze_metric",
        {"metric": "daily_revenue", **ANOMALY, **BASELINE},
    )
    assert ev is not None
    numbers = ev.numbers
    assert numbers["anomaly_is_real"] is True
    assert numbers["changepoint"] == "2026-09-26"

    ev = show(
        guard,
        "find_related_metrics (mechanism + traffic flat?)",
        "find_related_metrics",
        {"metric": "daily_revenue", **ANOMALY, **BASELINE},
    )
    assert ev is not None
    assert ev.numbers["traffic_flat"] is True

    ev = show(
        guard,
        "breakdown_by_dimension daily_revenue x payment_method",
        "breakdown_by_dimension",
        {"metric": "daily_revenue", "dimension": "payment_method", **ANOMALY, **BASELINE},
    )
    assert ev is not None
    top = ev.numbers["segments"][0]
    assert top["segment"] == "upi", f"expected upi, got {top['segment']}"
    assert top["baseline_payment_success_rate"] is not None

    ev = show(
        guard,
        "breakdown_by_dimension payment_success_rate x country",
        "breakdown_by_dimension",
        {"metric": "payment_success_rate", "dimension": "country", **ANOMALY, **BASELINE},
    )
    assert ev is not None
    countries = {s["segment"]: s for s in ev.numbers["segments"]}
    assert "IN" in countries
    assert countries["IN"]["delta_pct"] < -30

    ev = show(
        guard,
        "query_business_events (did the deploy happen before the changepoint?)",
        "query_business_events",
        {"start": "2026-09-25", "end": "2026-09-27"},
    )
    assert ev is not None
    types = [e["type"] for e in ev.numbers["events"]]
    assert "release" in types and "incident" in types

    print("\n--- guard failure paths ---")
    cases = [
        ("malformed json", "analyze_metric", "{not json at all"),
        ("unknown tool", "definitely_not_a_tool", "{}"),
        ("missing required param", "analyze_metric", json.dumps({"metric": "daily_revenue"})),
        ("bad param types", "analyze_metric", json.dumps(
            {"metric": "daily_revenue", "start": 20260926, "end": "2026-09-26"})),
        ("unknown param", "analyze_metric", json.dumps(
            {"metric": "daily_revenue", "start": "2026-09-26", "end": "2026-09-26", "bogus": 1})),
    ]
    for title, tool_name, raw in cases:
        out = guard.handle(tool_name, raw)
        status = "recovered" if out.ok else f"rejected: {out.tool_error.error if out.tool_error else '?'}"
        print(f"  {title:24s} -> {out.actions} {status}")

    print("\n--- loop detection (same call 4x) ---")
    loop_guard = ToolCallGuard(max_repeat_calls=3)
    args = json.dumps({"metric": "orders", **ANOMALY, **BASELINE})
    for i in range(4):
        out = loop_guard.handle("analyze_metric", args)
        print(f"  call {i + 1}: ok={out.ok} {out.actions}")

    print("\n--- budget exhaustion ---")
    small = ToolCallGuard(max_calls=1)
    small.handle("analyze_metric", args)
    out = small.handle("analyze_metric", args)
    print(f"  after budget: ok={out.ok} actions={out.actions}")

    print("\n--- cache hit ---")
    cache_guard = ToolCallGuard()
    cache_guard.handle("analyze_metric", args)
    out = cache_guard.handle("analyze_metric", args)
    print(f"  second identical call: {out.actions}")

    print("\nALL TOOL + GUARD CHECKS PASSED")


if __name__ == "__main__":
    main()
