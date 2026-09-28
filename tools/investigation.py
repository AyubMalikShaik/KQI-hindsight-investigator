"""Investigation tools.

Generic and metric-agnostic: everything is driven by the semantic catalog, so
the same agent works on revenue, conversion, churn or latency. Every tool returns
an Evidence with compact numbers and a declarative chart spec.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from core.schemas import ChartSpec, Evidence
from data import catalog as catalog_mod
from tools.registry import stable_evidence_id, tool
from tools import sql

SUM_METRICS = {"daily_revenue", "orders", "sessions"}


def _iso(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _window(args: dict[str, Any]) -> tuple[str, str, str, str]:
    start = _iso(args["start"])
    end = _iso(args["end"])
    baseline_start = _iso(args.get("baseline_start") or args["start"])
    baseline_end = _iso(args.get("baseline_end") or args["end"])
    return start, end, baseline_start, baseline_end


def _baseline_per_day(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df["value"].mean())


def _weekday_baseline(anomaly_day: pd.Timestamp, baseline: pd.DataFrame) -> tuple[float, int]:
    """Mean of the same weekday across the baseline window.

    Comparing a Saturday to a flat 28-day mean understates the change because the
    weekend lift is counted as part of the baseline. A real monitor compares like
    with like, and so does this tool.
    """
    if baseline.empty:
        return 0.0, 0
    same = baseline[baseline["date"].dt.dayofweek == anomaly_day.dayofweek]
    if same.empty:
        return float(baseline["value"].mean()), len(baseline)
    return float(same["value"].mean()), len(same)


def _traffic_z(anomaly_day: pd.Timestamp, baseline: pd.DataFrame, observed: float) -> float:
    """z-score of observed traffic against like-for-like baseline days.

    The spread comes from the same weekday, because a 28-day std folds the 11%
    weekend lift into it. Dispersion alone is not a sufficient test for
    "unchanged" though, so callers must also gate on the size of the change:
    see traffic_is_flat.
    """
    if baseline.empty:
        return 0.0
    expected, _ = _weekday_baseline(anomaly_day, baseline)
    if expected == 0:
        return 0.0
    same = baseline[baseline["date"].dt.dayofweek == anomaly_day.dayofweek]
    reference = same if len(same) > 1 else baseline
    std = float(reference["value"].std()) if len(reference) > 1 else 0.0
    if std <= 0:
        std = abs(expected) * 0.02
    return (observed - expected) / std


def traffic_is_flat(z: float, delta_pct: float, flat_pct: float = 5.0) -> bool:
    """Traffic counts as unchanged only if it is both statistically and materially flat.

    A z-score alone let the tool call a 12% traffic drop "statistically flat"
    while listing sessions among the metrics that had moved, because the
    baseline spread was inflated by weekend and promo-day swings. "Unchanged" is
    a claim about size, so it gets tested against size as well.
    """
    return abs(z) < 2.0 and abs(delta_pct) < flat_pct


def _changepoint(series: pd.DataFrame, expected: float, std: float) -> str | None:
    """First day in the window that is a genuine outlier versus expectation."""
    if series.empty or expected == 0:
        return None
    tolerance = max(std, expected * 0.02)
    flagged = series[(series["value"] - expected).abs() > 2.0 * tolerance]
    if flagged.empty:
        return None
    return _iso(flagged.iloc[0]["date"])


@tool(
    name="analyze_metric",
    description=(
        "Confirm whether a metric anomaly is real. Returns the daily series, the "
        "baseline level, day-over-day deviation, the changepoint date, z-score and "
        "data-quality flags. Use this first to validate the alert before theorising."
    ),
    parameters={
        "type": "object",
        "properties": {
            "metric": {
                "type": "string",
                "description": "Catalog metric name, e.g. daily_revenue, conversion_rate, orders.",
                "enum": sorted(sql.METRIC_COLUMN),
            },
            "start": {"type": "string", "description": "Anomaly window start, YYYY-MM-DD."},
            "end": {"type": "string", "description": "Anomaly window end, YYYY-MM-DD."},
            "baseline_start": {
                "type": "string",
                "description": "Baseline window start, YYYY-MM-DD. Defaults to start.",
            },
            "baseline_end": {
                "type": "string",
                "description": "Baseline window end, YYYY-MM-DD. Defaults to end.",
            },
        },
        "required": ["metric", "start", "end"],
    },
    cost_weight=1,
)
def analyze_metric(
    metric: str,
    start: str,
    end: str,
    baseline_start: str | None = None,
    baseline_end: str | None = None,
    **_: Any,
) -> Evidence:
    catalog_mod.get_catalog_entry(metric) or (_ for _ in ()).throw(
        ValueError(f"unknown metric '{metric}'")
    )
    a_start, a_end, b_start, b_end = _window(
        {"start": start, "end": end, "baseline_start": baseline_start, "baseline_end": baseline_end}
    )

    series = sql.totals_by_day(metric, a_start, a_end)
    baseline = sql.totals_by_day(metric, b_start, b_end)
    baseline_value = _baseline_per_day(baseline)
    anomaly_value = float(series["value"].mean()) if not series.empty else 0.0

    anomaly_day = pd.Timestamp(a_end)
    expected, weekday_samples = _weekday_baseline(anomaly_day, baseline)
    reference = expected if expected else baseline_value

    deviation_pct = (anomaly_value - reference) / reference * 100.0 if reference else 0.0
    naive_deviation_pct = (
        (anomaly_value - baseline_value) / baseline_value * 100.0 if baseline_value else 0.0
    )

    std = float(baseline["value"].std()) if len(baseline) > 1 else 0.0
    z_score = (anomaly_value - reference) / std if std > 0 else 0.0
    changepoint = _changepoint(series, reference, std)

    flags = sql.data_quality_flags(metric, a_start, a_end)
    metric_is_real = abs(deviation_pct) >= 3.0 and not flags

    basis = (
        f"same-weekday baseline over {weekday_samples} sample(s)"
        if weekday_samples and weekday_samples != len(baseline)
        else f"{b_start}..{b_end} baseline"
    )
    verdict = (
        f"{metric} moved {deviation_pct:+.1f}% against a {basis} "
        f"({anomaly_value:,.2f} observed vs {reference:,.2f} expected), z={z_score:.2f}."
    )
    if abs(naive_deviation_pct - deviation_pct) > 2.0:
        verdict += (
            f" A flat window mean would put it at {naive_deviation_pct:+.1f}% "
            "(day-of-week adjusted here)."
        )
    if not metric_is_real:
        verdict += (
            " Deviation is within normal variance or the window has data-quality "
            "problems, so treat the alert as unconfirmed."
        )

    return Evidence(
        evidence_id=stable_evidence_id(
            "MET", tool="analyze_metric", metric=metric, start=a_start, end=a_end, base=b_start
        ),
        tool="analyze_metric",
        finding=verdict,
        numbers={
            "metric": metric,
            "anomaly_mean": round(anomaly_value, 4),
            "baseline_mean": round(baseline_value, 4),
            "expected_same_weekday": round(reference, 4),
            "deviation_pct": round(deviation_pct, 2),
            "deviation_pct_unadjusted": round(naive_deviation_pct, 2),
            "z_score": round(z_score, 2),
            "changepoint": changepoint,
            "baseline_std": round(std, 4),
            "anomaly_window": [a_start, a_end],
            "baseline_window": [b_start, b_end],
            "anomaly_is_real": metric_is_real,
            "series": {
                _iso(r["date"]): round(float(r["value"]), 4)
                for _, r in series.iterrows()
            },
        },
        confidence_hint=0.85 if metric_is_real else 0.35,
        chart=ChartSpec(
            kind="line",
            title=f"{metric}: anomaly vs baseline",
            categories=[_iso(r["date"]) for _, r in series.iterrows()],
            series={"value": [round(float(v), 4) for v in series["value"]]},
            annotations=[{"label": "baseline", "y": round(baseline_value, 4)}],
        ),
        data_quality_flags=flags,
    )


@tool(
    name="breakdown_by_dimension",
    description=(
        "Find which segment caused the change. For each segment value of a "
        "dimension, returns the anomaly value, baseline value, absolute and "
        "percentage contribution to the total change, and a full funnel "
        "decomposition (sessions, payment attempts, payment successes, orders, "
        "revenue) that distinguishes a VOLUME effect from a RATE effect. This is "
        "the decisive tool for attributing a drop to a specific segment."
    ),
    parameters={
        "type": "object",
        "properties": {
            "metric": {
                "type": "string",
                "description": "Catalog metric to attribute.",
                "enum": sorted(sql.METRIC_COLUMN),
            },
            "dimension": {
                "type": "string",
                "description": "Dimension to split by.",
                "enum": catalog_mod.CORE_DIMENSIONS,
            },
            "start": {"type": "string", "description": "Anomaly window start."},
            "end": {"type": "string", "description": "Anomaly window end."},
            "baseline_start": {"type": "string", "description": "Baseline window start."},
            "baseline_end": {"type": "string", "description": "Baseline window end."},
        },
        "required": ["metric", "dimension", "start", "end"],
    },
    cost_weight=2,
)
def breakdown_by_dimension(
    metric: str,
    dimension: str,
    start: str,
    end: str,
    baseline_start: str | None = None,
    baseline_end: str | None = None,
    **_: Any,
) -> Evidence:
    a_start, a_end, b_start, b_end = _window(
        {"start": start, "end": end, "baseline_start": baseline_start, "baseline_end": baseline_end}
    )
    if metric not in sql.METRIC_COLUMN:
        raise ValueError(
            f"metric '{metric}' is not in the catalog. "
            f"Allowed: {', '.join(sorted(sql.METRIC_COLUMN))}"
        )

    segments = sql.decompose_metric(metric, dimension, a_start, a_end, b_start, b_end)

    if segments.empty:
        return Evidence(
            evidence_id=stable_evidence_id(
                "BRK",
                tool="breakdown_by_dimension",
                metric=metric,
                dimension=dimension,
                start=a_start,
                end=a_end,
                base=b_start,
            ),
            tool="breakdown_by_dimension",
            finding=f"No data for {dimension} in the requested window.",
            numbers={},
            confidence_hint=0.2,
        )

    total_delta = float(segments["delta"].sum()) or 0.0
    gross_negative = float(-segments.loc[segments["delta"] < 0, "delta"].sum()) or 0.0
    contributors = segments[segments["delta"] < 0].head(10).copy()
    top = contributors.iloc[0] if not contributors.empty else None

    finding_parts: list[str] = []
    if top is not None:
        share = float(top["delta"]) / gross_negative * 100.0 if gross_negative else 0.0
        finding_parts.append(
            f"{top['segment']} is the largest negative contributor to {metric} "
            f"({share:.0f}% of the total decline, {float(top['delta_pct'] or 0):+.1f}%)."
        )
        base_sessions = float(top["baseline_sessions"])
        anomaly_sessions = float(top["anomaly_sessions"])
        session_change = (
            (anomaly_sessions - base_sessions) / base_sessions * 100.0 if base_sessions else 0.0
        )
        volume_effect = float(top["volume_effect"])
        rate_effect = float(top["rate_effect"])
        base_psr = top["baseline_payment_success_rate"]
        anomaly_psr = top["anomaly_payment_success_rate"]
        base_cr = top["baseline_conversion_rate"]
        anomaly_cr = top["anomaly_conversion_rate"]

        session_series = sql.metric_by_dimension(
            "sessions", a_start, a_end, dimension, str(top["segment"])
        )
        session_baseline = sql.metric_by_dimension(
            "sessions", b_start, b_end, dimension, str(top["segment"])
        )
        session_expected, _ = _weekday_baseline(pd.Timestamp(a_end), session_baseline)
        if not session_expected:
            session_expected = (
                float(session_baseline["value"].mean())
                if not session_baseline.empty
                else anomaly_sessions
            )
        session_z = _traffic_z(pd.Timestamp(a_end), session_baseline, anomaly_sessions)

        traffic_flat = traffic_is_flat(session_z, session_change)
        magnitude = abs(rate_effect) + abs(volume_effect)
        rate_share = abs(rate_effect) / magnitude if magnitude else 0.0

        if traffic_flat:
            finding_parts.append(
                f"Sessions in this segment are flat ({session_change:+.1f}%, z={session_z:.2f}), "
                "so this is a RATE effect, not a traffic effect."
            )
        elif rate_share >= 0.65:
            finding_parts.append(
                f"Sessions changed {session_change:+.1f}% (z={session_z:.2f}) but the rate "
                f"effect dominates ({rate_share * 100:.0f}% of the change), so the cause is "
                "conversion quality rather than traffic volume."
            )
        elif abs(volume_effect) >= abs(rate_effect):
            finding_parts.append(
                f"Sessions in this segment changed {session_change:+.1f}% (z={session_z:.2f}) "
                "and the volume effect dominates, so this is a VOLUME effect."
            )
        else:
            finding_parts.append(
                f"Sessions changed {session_change:+.1f}% (z={session_z:.2f}) and the rate "
                "effect is larger, so the cause is mixed but primarily conversion quality."
            )

        if base_psr and anomaly_psr and abs(anomaly_psr - base_psr) > 0.05:
            finding_parts.append(
                f"Payment success rate in this segment fell from {float(base_psr):.3f} to "
                f"{float(anomaly_psr):.3f}."
            )
        if base_cr and anomaly_cr and abs(anomaly_cr - base_cr) > 0.001:
            finding_parts.append(
                f"Conversion rate in this segment fell from {float(base_cr):.4f} to "
                f"{float(anomaly_cr):.4f}."
            )

        if metric in SUM_METRICS:
            finding_parts.append(
                f"Decomposition of the {float(top['delta']):,.0f} change: volume effect "
                f"{float(top['volume_effect']):,.0f}, rate effect {float(top['rate_effect']):,.0f}."
            )
        else:
            finding_parts.append(
                f"This is a ratio metric, so the change is a rate change by construction: "
                f"{float(top['anomaly_value']):.4f} versus a baseline of "
                f"{float(top['baseline_value']):.4f}."
            )

        isolated = segments[
            (segments["delta"] < 0) & (segments["segment"] != top["segment"])
        ]
        if not isolated.empty:
            finding_parts.append(
                "Other segments also declined: "
                + ", ".join(str(s) for s in isolated["segment"].head(4))
                + f" (largest {float(isolated['delta_pct'].min() or 0):+.1f}%)."
            )
        offsetting = segments[segments["delta"] > 0]
        if not offsetting.empty:
            finding_parts.append(
                "Segments that held up or improved: "
                + ", ".join(str(s) for s in offsetting.nlargest(3, "delta")["segment"])
                + f", together offsetting {float(offsetting['delta'].sum()):,.0f}."
            )
    else:
        finding_parts.append(f"No segment of {dimension} contributed negatively to {metric}.")

    numbers = {
        "metric": metric,
        "dimension": dimension,
        "total_change": round(total_delta, 2),
        "gross_decline": round(gross_negative, 2),
        "segments": [
            {
                "segment": str(row["segment"]),
                "anomaly_value": round(float(row["anomaly_value"]), 4),
                "baseline_value": round(float(row["baseline_value"]), 4),
                "delta": round(float(row["delta"]), 2),
                "delta_pct": round(float(row["delta_pct"] or 0), 2),
                "share_of_decline_pct": round(
                    float(row["delta"]) / gross_negative * 100.0 if gross_negative else 0.0, 1
                ),
                "volume_effect": round(float(row["volume_effect"]), 2),
                "rate_effect": round(float(row["rate_effect"]), 2),
                "anomaly_sessions": int(row["anomaly_sessions"]),
                "baseline_sessions": int(row["baseline_sessions"]),
                "anomaly_orders": int(row["anomaly_orders"]),
                "baseline_orders": int(row["baseline_orders"]),
                "anomaly_payment_success_rate": (
                    round(float(row["anomaly_payment_success_rate"]), 4)
                    if row["anomaly_payment_success_rate"] is not None
                    else None
                ),
                "baseline_payment_success_rate": (
                    round(float(row["baseline_payment_success_rate"]), 4)
                    if row["baseline_payment_success_rate"] is not None
                    else None
                ),
                "anomaly_conversion_rate": (
                    round(float(row["anomaly_conversion_rate"]), 5)
                    if row["anomaly_conversion_rate"] is not None
                    else None
                ),
                "baseline_conversion_rate": (
                    round(float(row["baseline_conversion_rate"]), 5)
                    if row["baseline_conversion_rate"] is not None
                    else None
                ),
            }
            for _, row in contributors.iterrows()
        ],
        # Expose the dominant negative segment structurally. It is the single
        # most important fact in a decomposition, and the confidence scorer
        # needs it as data rather than having to parse it back out of prose.
        "top": {
            "dimension": dimension,
            "segment": str(top["segment"]) if top is not None else None,
            "delta": round(float(top["delta"]), 2) if top is not None else 0.0,
            "delta_pct": round(float(top["delta_pct"] or 0.0), 2) if top is not None else 0.0,
            "share_of_decline_pct": (
                round(float(top["delta"]) / gross_negative * 100.0, 1)
                if top is not None and gross_negative
                else 0.0
            ),
        },
    }

    return Evidence(
        evidence_id=stable_evidence_id(
            "BRK",
            tool="breakdown_by_dimension",
            metric=metric,
            dimension=dimension,
            start=a_start,
            end=a_end,
            base=b_start,
        ),
        tool="breakdown_by_dimension",
        finding=" ".join(finding_parts),
        numbers=numbers,
        confidence_hint=0.8 if top is not None else 0.3,
        chart=ChartSpec(
            kind="bar",
            title=f"{metric} contribution to change by {dimension}",
            categories=[str(s) for s in contributors["segment"]],
            series={"delta": [round(float(d), 2) for d in contributors["delta"]]},
        ),
    )


@tool(
    name="find_related_metrics",
    description=(
        "Check every metric related to the primary one in the semantic catalog and "
        "report which of them moved. Use this to find the mechanism behind a change, "
        "for example that conversion fell while traffic stayed flat."
    ),
    parameters={
        "type": "object",
        "properties": {
            "metric": {"type": "string", "description": "Primary metric.", "enum": sorted(sql.METRIC_COLUMN)},
            "start": {"type": "string", "description": "Anomaly window start."},
            "end": {"type": "string", "description": "Anomaly window end."},
            "baseline_start": {"type": "string", "description": "Baseline window start."},
            "baseline_end": {"type": "string", "description": "Baseline window end."},
            "dimension": {
                "type": "string",
                "description": "Optional dimension to restrict the comparison to.",
                "enum": catalog_mod.CORE_DIMENSIONS,
            },
            "dimension_value": {
                "type": "string",
                "description": "Optional segment value, used with dimension.",
            },
        },
        "required": ["metric", "start", "end"],
    },
    cost_weight=1,
)
def find_related_metrics(
    metric: str,
    start: str,
    end: str,
    baseline_start: str | None = None,
    baseline_end: str | None = None,
    dimension: str | None = None,
    dimension_value: str | None = None,
    **_: Any,
) -> Evidence:
    a_start, a_end, b_start, b_end = _window(
        {"start": start, "end": end, "baseline_start": baseline_start, "baseline_end": baseline_end}
    )
    candidates = [m for m in catalog_mod.related_metrics(metric) if m in sql.METRIC_COLUMN]
    if metric not in candidates:
        candidates = [metric] + candidates

    moved: list[dict[str, Any]] = []
    anomaly_day = pd.Timestamp(a_end)
    segment_note = (
        f" within {dimension}={dimension_value}" if dimension and dimension_value else ""
    )

    for candidate in candidates:
        a_series = sql.totals_by_day(
            candidate, a_start, a_end, dimension=dimension, dimension_value=dimension_value
        )
        b_series = sql.totals_by_day(
            candidate, b_start, b_end, dimension=dimension, dimension_value=dimension_value
        )
        a_val = float(a_series["value"].mean()) if not a_series.empty else 0.0
        b_val, _ = _weekday_baseline(anomaly_day, b_series)
        if not b_val:
            b_val = float(b_series["value"].mean()) if not b_series.empty else 0.0
        delta_pct = (a_val - b_val) / b_val * 100.0 if b_val else 0.0
        moved.append(
            {
                "metric": candidate,
                "anomaly": round(a_val, 4),
                "baseline": round(b_val, 4),
                "delta_pct": round(delta_pct, 2),
                "moved": abs(delta_pct) >= 3.0,
            }
        )

    changed = [m for m in moved if m["moved"]]
    traffic = next((m for m in moved if m["metric"] == "sessions"), None)
    mechanism = next(
        (m for m in moved if m["metric"] in ("payment_success_rate", "conversion_rate", "aov")),
        None,
    )

    parts: list[str] = []
    if changed:
        parts.append(
            "Metrics that moved"
            + segment_note
            + ": "
            + ", ".join(f"{m['metric']} {m['delta_pct']:+.1f}%" for m in changed)
            + "."
        )
    else:
        parts.append(f"No related metric moved by more than 3%{segment_note}.")

    traffic_flat = False
    if traffic is not None:
        sessions_series = sql.totals_by_day(
            "sessions", a_start, a_end, dimension=dimension, dimension_value=dimension_value
        )
        baseline_series = sql.totals_by_day(
            "sessions", b_start, b_end, dimension=dimension, dimension_value=dimension_value
        )
        a_val = float(sessions_series["value"].mean()) if not sessions_series.empty else 0.0
        b_val, _ = _weekday_baseline(anomaly_day, baseline_series)
        b_val = b_val or (float(baseline_series["value"].mean()) if not baseline_series.empty else 0.0)
        session_z = _traffic_z(anomaly_day, baseline_series, a_val)
        traffic_flat = traffic_is_flat(session_z, float(traffic["delta_pct"]))
        if traffic_flat:
            parts.append(
                f"Traffic (sessions) is statistically flat ({traffic['delta_pct']:+.1f}%, "
                f"z={session_z:.2f}), which rules out a demand or traffic-loss explanation."
            )
        else:
            parts.append(
                f"Traffic (sessions) also moved {traffic['delta_pct']:+.1f}% (z={session_z:.2f}), "
                "so a traffic-loss explanation is still in play."
            )
    if mechanism and mechanism["moved"]:
        parts.append(
            f"Mechanism indicator {mechanism['metric']} moved "
            f"{mechanism['delta_pct']:+.1f}%{segment_note}."
        )

    return Evidence(
        evidence_id=stable_evidence_id(
            "REL",
            tool="find_related_metrics",
            metric=metric,
            start=a_start,
            end=a_end,
            base=b_start,
            dimension=dimension or "",
            value=dimension_value or "",
        ),
        tool="find_related_metrics",
        finding=" ".join(parts),
        numbers={
            "metric": metric,
            "dimension": dimension,
            "dimension_value": dimension_value,
            "related": moved,
            "traffic_flat": traffic_flat,
        },
        confidence_hint=0.7 if changed else 0.3,
        chart=ChartSpec(
            kind="bar",
            title="Related metric deviation",
            categories=[m["metric"] for m in moved],
            series={"delta_pct": [m["delta_pct"] for m in moved]},
        ),
    )


@tool(
    name="query_business_events",
    description=(
        "List deployments, pricing changes, campaigns, vendor incidents and "
        "incidents around a time range, plus any log error spikes. Use this to test "
        "whether a candidate cause actually happened before the metric changed."
    ),
    parameters={
        "type": "object",
        "properties": {
            "start": {"type": "string", "description": "Range start, YYYY-MM-DD or datetime."},
            "end": {"type": "string", "description": "Range end, YYYY-MM-DD or datetime."},
            "event_type": {
                "type": "string",
                "description": "Optional filter.",
                "enum": ["release", "pricing", "campaign", "vendor_incident", "incident"],
            },
            "include_logs": {
                "type": "boolean",
                "description": "Include log error spikes. Defaults to true.",
            },
        },
        "required": ["start", "end"],
    },
    cost_weight=1,
)
def query_business_events(
    start: str,
    end: str,
    event_type: str | None = None,
    include_logs: bool = True,
    **_: Any,
) -> Evidence:
    a_start = pd.Timestamp(start).strftime("%Y-%m-%d")
    a_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    events = sql.business_events_between(a_start, a_end, event_type)
    logs = sql.log_spikes_between(a_start, a_end) if include_logs else pd.DataFrame()

    parts: list[str] = []
    if not events.empty:
        parts.append(f"{len(events)} business event(s) in {a_start}..{a_end}:")
        for _, row in events.iterrows():
            parts.append(
                f"{pd.Timestamp(row['ts']).strftime('%Y-%m-%d %H:%M')} "
                f"[{row['type']}] {row['description']} (owner: {row['owner']})"
            )
    else:
        parts.append(f"No business events recorded in {a_start}..{a_end}.")

    if not logs.empty:
        top = logs.iloc[0]
        parts.append(
            f"Largest log spike: '{top['signature']}' on {top['service']} with "
            f"{int(top['count']):,} occurrences starting "
            f"{pd.Timestamp(top['start']).strftime('%Y-%m-%d %H:%M')}."
        )

    return Evidence(
        evidence_id=stable_evidence_id(
            "EVT",
            tool="query_business_events",
            start=a_start,
            end=a_end,
            event_type=event_type or "",
            include_logs=bool(include_logs),
        ),
        tool="query_business_events",
        finding=" ".join(parts),
        numbers={
            "window": [a_start, a_end],
            "events": [
                {
                    "event_id": row["event_id"],
                    "ts": pd.Timestamp(row["ts"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "type": row["type"],
                    "service": row["service"],
                    "description": row["description"],
                    "owner": row["owner"],
                    "severity": row["severity"],
                }
                for _, row in events.iterrows()
            ],
            "log_spikes": [
                {
                    "signature": row["signature"],
                    "service": row["service"],
                    "count": int(row["count"]),
                    "start": pd.Timestamp(row["start"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                for _, row in logs.iterrows()
            ],
        },
        confidence_hint=0.6 if not events.empty else 0.3,
        chart=ChartSpec(
            kind="scatter",
            title="Business events in window",
            categories=[
                pd.Timestamp(row["ts"]).strftime("%Y-%m-%d %H:%M") for _, row in events.iterrows()
            ],
            series={},
        ),
    )


@tool(
    name="query_config_changes",
    description=(
        "List runtime configuration changes applied around a time range, newest last. "
        "Reach for this when a metric moved with no release, deploy or campaign in the "
        "event log: a config push changes behaviour without ever appearing in the "
        "release history."
    ),
    parameters={
        "type": "object",
        "properties": {
            "start": {"type": "string", "description": "Range start, YYYY-MM-DD or datetime."},
            "end": {"type": "string", "description": "Range end, YYYY-MM-DD or datetime."},
            "service": {
                "type": "string",
                "description": "Optional service filter, e.g. checkout-service.",
            },
        },
        "required": ["start", "end"],
    },
    cost_weight=1,
)
def query_config_changes(
    start: str,
    end: str,
    service: str | None = None,
    **_: Any,
) -> Evidence:
    a_start = pd.Timestamp(start).strftime("%Y-%m-%d")
    a_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    changes = sql.config_changes_between(a_start, a_end)
    if service:
        changes = changes[changes["service"].str.lower() == service.strip().lower()]

    parts: list[str] = []
    if changes.empty:
        parts.append(f"No config changes recorded in {a_start}..{a_end}.")
    else:
        parts.append(f"{len(changes)} config change(s) in {a_start}..{a_end}:")
        for _, row in changes.iterrows():
            parts.append(
                f"{pd.Timestamp(row['ts']).strftime('%Y-%m-%d %H:%M')} "
                f"{row['config_key']} on {row['service']}: {row['old_value']} -> "
                f"{row['new_value']} (by {row['changed_by']})"
            )

    return Evidence(
        evidence_id=stable_evidence_id(
            "CFG",
            tool="query_config_changes",
            start=a_start,
            end=a_end,
            service=service or "",
        ),
        tool="query_config_changes",
        finding=" ".join(parts),
        numbers={
            "window": [a_start, a_end],
            "changes": [
                {
                    "change_id": row["change_id"],
                    "ts": pd.Timestamp(row["ts"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "service": row["service"],
                    "config_key": row["config_key"],
                    "old_value": row["old_value"],
                    "new_value": row["new_value"],
                    "changed_by": row["changed_by"],
                }
                for _, row in changes.iterrows()
            ],
        },
        confidence_hint=0.6 if not changes.empty else 0.3,
        chart=ChartSpec(
            kind="bar",
            title="Config changes in window",
            categories=[row["config_key"] for _, row in changes.iterrows()],
            series={"change_count": [1] * len(changes)},
        ),
    )
