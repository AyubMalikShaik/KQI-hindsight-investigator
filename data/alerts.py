"""Alerts derived from the planted scenarios.

An alert is a *measurement* the monitoring layer would have emitted, not a
hand-written number: observed, expected and deviation are all recomputed from
the warehouse so an alert can never drift away from the data it describes.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from core.schemas import Anomaly, Direction, Severity, Window
from data.scenarios import SCENARIOS

ALERT_METRIC = "daily_revenue"
BASELINE_DAYS = 28
ALERT_IDS = {"SCN-001": "ALR-2026-0142"}


def _daily_revenue(db_path) -> pd.DataFrame:
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(
            "SELECT date, SUM(revenue)::DOUBLE AS revenue FROM fact_daily GROUP BY date ORDER BY date"
        ).fetchdf()
    finally:
        con.close()


def _severity(deviation_pct: float) -> Severity:
    magnitude = abs(deviation_pct)
    if magnitude >= 20:
        return Severity.critical
    if magnitude >= 10:
        return Severity.high
    if magnitude >= 5:
        return Severity.medium
    return Severity.low


def _weekday_expected(series: pd.DataFrame, day: pd.Timestamp) -> float:
    """Expected value from the same weekday in the previous four weeks.

    Weekends carry a 1.11 traffic multiplier, so comparing a Saturday against a
    flat 28-day mean reports a healthy weekend as a large gain. The tool layer
    makes the same adjustment; an alert that did not would hand the agent a
    deviation it can never explain.
    """
    by_date = dict(zip(series["date"], series["revenue"]))
    samples = [
        by_date.get((day - pd.Timedelta(days=7 * k)).normalize(), None) for k in (1, 2, 3, 4)
    ]
    usable = [float(v) for v in samples if v is not None and v == v]
    if usable:
        return sum(usable) / len(usable)
    baseline = series.loc[
        (series["date"] > day - pd.Timedelta(days=BASELINE_DAYS)) & (series["date"] < day),
        "revenue",
    ]
    return float(baseline.mean())


def build_alert(scenario_id: str, db_path) -> Anomaly:
    """Recompute the alert a monitor would raise for a scenario."""
    if scenario_id not in SCENARIOS:
        raise ValueError(
            f"unknown scenario '{scenario_id}'. Known: {', '.join(sorted(SCENARIOS))}"
        )

    truth = SCENARIOS[scenario_id].truth
    changepoint = pd.Timestamp(truth.changepoint)
    series = _daily_revenue(db_path)

    observed = float(series.loc[series["date"] == changepoint, "revenue"].sum())
    expected = _weekday_expected(series, changepoint)
    deviation = (observed - expected) / expected * 100.0
    baseline_start = changepoint - pd.Timedelta(days=BASELINE_DAYS)

    end = changepoint + pd.Timedelta(days=1)
    return Anomaly(
        alert_id=ALERT_IDS.get(scenario_id, f"ALR-{scenario_id}"),
        metric=ALERT_METRIC,
        grain="day",
        detected_at=datetime(
            end.year, end.month, end.day, 6, 0, 0, tzinfo=timezone.utc
        ),
        anomaly_window=Window(start=truth.changepoint, end=truth.changepoint),
        baseline_window=Window(
            start=baseline_start.strftime("%Y-%m-%d"),
            end=(changepoint - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        ),
        observed=round(observed, 2),
        expected=round(expected, 2),
        deviation_pct=round(deviation, 1),
        direction=Direction.drop if deviation < 0 else Direction.spike,
        severity=_severity(deviation),
        source="grafana",
    )


def describe_ground_truth(scenario_id: str) -> str:
    truth = SCENARIOS[scenario_id].truth
    segment = " x ".join(f"{k}={v}" for k, v in truth.affected_segment.items())
    lines = [
        f"  scenario : {scenario_id} {SCENARIOS[scenario_id].title}",
        f"  cause    : {truth.root_cause}",
        f"  type     : {truth.cause_type}",
        f"  segment  : {segment}",
        f"  decisive : {truth.decisive_tool}",
    ]
    for item in truth.distinguishing_evidence:
        lines.append(f"    - {item}")
    return "\n".join(lines)


__all__ = [
    "ALERT_METRIC",
    "build_alert",
    "describe_ground_truth",
]
