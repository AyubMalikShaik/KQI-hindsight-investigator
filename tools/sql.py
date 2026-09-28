"""Shared query helpers for the tool layer.

Ratio metrics are always recomputed from their numerator and denominator, never
averaged across daily rates, which is the classic silent-error trap.
"""

from __future__ import annotations

import json
import re
from typing import Any

import duckdb
import pandas as pd

from data import catalog as catalog_mod
from data.db import read_only_conn
from tools.registry import ALLOWED_TABLES, MAX_CELLS, MAX_ROWS, clamp_rows

_CONN: duckdb.DuckDBPyConnection | None = None

NUMERIC_COLUMNS = [
    "sessions",
    "orders",
    "revenue",
    "revenue_gross",
    "refund_amount",
    "payment_attempts",
    "payment_successes",
]

METRIC_COLUMN = {
    "daily_revenue": "revenue",
    "orders": "orders",
    "sessions": "sessions",
    "conversion_rate": ("orders", "sessions"),
    "aov": ("revenue", "orders"),
    "payment_success_rate": ("payment_successes", "payment_attempts"),
}


def connection() -> duckdb.DuckDBPyConnection:
    global _CONN
    if _CONN is None:
        _CONN = read_only_conn()
    return _CONN


def close_connection() -> None:
    global _CONN
    if _CONN is not None:
        _CONN.close()
        _CONN = None


def validate_table(table: str) -> str:
    if table not in ALLOWED_TABLES:
        raise ValueError(f"table '{table}' is not allow-listed")
    return table


def validate_dimension(dimension: str) -> str:
    if dimension not in catalog_mod.CORE_DIMENSIONS:
        raise ValueError(
            f"dimension '{dimension}' is not in the catalog. "
            f"Allowed: {', '.join(catalog_mod.CORE_DIMENSIONS)}"
        )
    return dimension


def metric_expression(metric: str) -> str:
    """SQL expression that yields the metric, recomputing ratios correctly."""
    if metric not in METRIC_COLUMN:
        raise ValueError(
            f"metric '{metric}' is not in the catalog. "
            f"Allowed: {', '.join(sorted(METRIC_COLUMN))}"
        )
    spec = METRIC_COLUMN[metric]
    if isinstance(spec, tuple):
        numerator, denominator = spec
        return f"SUM({numerator})::DOUBLE / NULLIF(SUM({denominator}), 0)"
    return f"SUM({spec})::DOUBLE"


def _where_clauses(
    dimension: str | None,
    value: str | None,
    extra_filters: list[str] | None = None,
) -> tuple[str, list[Any]]:
    clauses: list[str] = ["1=1"]
    params: list[Any] = []
    if dimension and value:
        validate_dimension(dimension)
        clauses.append(f"{dimension} = ?")
        params.append(value)
    for clause in extra_filters or []:
        if re.search(r"[;]|--|/\*", clause):
            raise ValueError("illegal token in filter")
        clauses.append(f"({clause})")
    return " AND ".join(clauses), params


def metric_by_dimension(
    metric: str,
    start: str,
    end: str,
    dimension: str | None = None,
    value: str | None = None,
    limit: int = MAX_ROWS,
) -> pd.DataFrame:
    validate_table("fact_daily")

    # Clause order and param order MUST be built together. Appending the date
    # params after the fact while the dimension placeholder came first silently
    # bound the country value to the date range and returned an empty frame.
    clauses: list[str] = ["date BETWEEN ? AND ?"]
    params: list[Any] = [start, end]
    if dimension and value:
        validate_dimension(dimension)
        clauses.append(f"{dimension} = ?")
        params.append(value)

    where = " AND ".join(clauses)
    expr = metric_expression(metric)

    sql = f"""
        SELECT date, {expr} AS value
        FROM fact_daily
        WHERE {where}
        GROUP BY date
        ORDER BY date
    """
    return clamp_rows(connection().execute(sql, params).fetchdf(), limit)


def segment_aggregates(
    dimension: str,
    start: str,
    end: str,
    baseline_start: str,
    baseline_end: str,
    limit: int = 60,
) -> pd.DataFrame:
    """Raw funnel sums per segment for both windows.

    Metric arithmetic is done in Python from these sums, so a ratio metric is
    always recomputed from its numerator and denominator rather than averaged
    across daily rates.
    """
    validate_table("fact_daily")
    validate_dimension(dimension)

    def block(lo: str, hi: str, prefix: str, params: list[Any]) -> str:
        params.extend([lo, hi] * 6)
        return f"""
            SUM(CASE WHEN date >= ? AND date <= ? THEN sessions ELSE 0 END) AS {prefix}_sessions,
            SUM(CASE WHEN date >= ? AND date <= ? THEN payment_attempts ELSE 0 END) AS {prefix}_attempts,
            SUM(CASE WHEN date >= ? AND date <= ? THEN payment_successes ELSE 0 END) AS {prefix}_successes,
            SUM(CASE WHEN date >= ? AND date <= ? THEN orders ELSE 0 END) AS {prefix}_orders,
            SUM(CASE WHEN date >= ? AND date <= ? THEN revenue ELSE 0 END) AS {prefix}_revenue,
            SUM(CASE WHEN date >= ? AND date <= ? THEN revenue_gross ELSE 0 END) AS {prefix}_revenue_gross
        """

    params: list[Any] = []
    anomaly_block = block(start, end, "a", params)
    baseline_block = block(baseline_start, baseline_end, "b", params)
    params.extend([min(start, baseline_start), max(end, baseline_end), limit])

    sql_text = f"""
        SELECT {dimension} AS segment, {anomaly_block}, {baseline_block}
        FROM fact_daily
        WHERE date >= ? AND date <= ?
        GROUP BY 1
        ORDER BY a_revenue ASC
        LIMIT ?
    """
    df = connection().execute(sql_text, params).fetchdf()
    if df.empty:
        return df
    df = df[(df["a_revenue"] > 0) | (df["b_revenue"] > 0)].reset_index(drop=True)

    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    baseline_days = (pd.Timestamp(baseline_end) - pd.Timestamp(baseline_start)).days + 1
    df.attrs["baseline_scaled"] = baseline_days != days
    df.attrs["baseline_days"] = baseline_days
    return df


def _scale_baseline(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Scale a multi-week baseline down to a single-week comparison window.

    A 28-day baseline summed over 7 anomaly days would otherwise make every
    segment look like a 75% decline.
    """
    if not df.attrs.get("baseline_scaled"):
        return df
    factor = window_days / float(df.attrs["baseline_days"])
    out = df.copy()
    for column in out.columns:
        if column.startswith("b_"):
            out[column] = out[column] * factor
    return out


AGG_COLUMN = {
    "sessions": "sessions",
    "orders": "orders",
    "revenue": "revenue",
    "revenue_gross": "revenue_gross",
    "payment_attempts": "attempts",
    "payment_successes": "successes",
}


def value_from_sums(row: pd.Series, metric: str, prefix: str) -> float:
    spec = METRIC_COLUMN.get(metric)
    if spec is None:
        raise ValueError(f"metric '{metric}' is not in the catalog")
    if isinstance(spec, tuple):
        numerator, denominator = spec
        num = float(row[f"{prefix}_{AGG_COLUMN[numerator]}"])
        den = float(row[f"{prefix}_{AGG_COLUMN[denominator]}"])
        return num / den if den else 0.0
    return float(row[f"{prefix}_{AGG_COLUMN[spec]}"])


def decompose_metric(
    metric: str, dimension: str, start: str, end: str, baseline_start: str, baseline_end: str
) -> pd.DataFrame:
    """Segment-level attribution of a metric change, with the volume/rate split.

    For a sum metric the change in each segment is split into the part explained
    by traffic volume and the part explained by value per session, which is what
    makes an attribution defensible rather than a guess.
    """
    raw = segment_aggregates(dimension, start, end, baseline_start, baseline_end)
    if raw.empty:
        return raw

    window_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    raw = _scale_baseline(raw, window_days)

    rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        a_value = value_from_sums(row, metric, "a")
        b_value = value_from_sums(row, metric, "b")
        a_sessions = float(row["a_sessions"])
        b_sessions = float(row["b_sessions"])
        a_per_session = a_value / a_sessions if a_sessions else 0.0
        b_per_session = b_value / b_sessions if b_sessions else 0.0

        volume_effect = (a_sessions - b_sessions) * b_per_session
        rate_effect = a_sessions * (a_per_session - b_per_session)

        rows.append(
            {
                "segment": row["segment"],
                "anomaly_value": a_value,
                "baseline_value": b_value,
                "delta": a_value - b_value,
                "delta_pct": (
                    (a_value - b_value) / b_value * 100.0 if b_value else None
                ),
                "anomaly_sessions": int(a_sessions),
                "baseline_sessions": int(b_sessions),
                "anomaly_orders": int(row["a_orders"]),
                "baseline_orders": int(row["b_orders"]),
                "anomaly_revenue": float(row["a_revenue"]),
                "baseline_revenue": float(row["b_revenue"]),
                "anomaly_payment_success_rate": (
                    float(row["a_successes"]) / float(row["a_attempts"])
                    if float(row["a_attempts"])
                    else None
                ),
                "baseline_payment_success_rate": (
                    float(row["b_successes"]) / float(row["b_attempts"])
                    if float(row["b_attempts"])
                    else None
                ),
                "anomaly_conversion_rate": (
                    float(row["a_orders"]) / a_sessions if a_sessions else None
                ),
                "baseline_conversion_rate": (
                    float(row["b_orders"]) / b_sessions if b_sessions else None
                ),
                "volume_effect": volume_effect,
                "rate_effect": rate_effect,
            }
        )

    out = pd.DataFrame(rows)
    total_delta = float(out["delta"].sum())
    if total_delta:
        out["contribution_pct"] = out["delta"] / total_delta * 100.0
    else:
        out["contribution_pct"] = 0.0
    return out.sort_values("delta").reset_index(drop=True)


def totals_by_day(
    metric: str,
    start: str,
    end: str,
    limit: int = MAX_ROWS,
    dimension: str | None = None,
    dimension_value: str | None = None,
) -> pd.DataFrame:
    return metric_by_dimension(
        metric, start, end, limit=limit, dimension=dimension, value=dimension_value
    )


def business_events_between(
    start: str, end: str, event_type: str | None = None, limit: int = 50
) -> pd.DataFrame:
    validate_table("business_events")
    clauses = ["ts >= ? AND ts < ?"]
    params: list[Any] = [start, end]
    if event_type:
        clauses.append("type = ?")
        params.append(event_type)
    sql = f"""
        SELECT event_id, ts, type, service, description, owner, severity, metadata
        FROM business_events
        WHERE {' AND '.join(clauses)}
        ORDER BY ts
        LIMIT ?
    """
    params.append(limit)
    return clamp_rows(connection().execute(sql, params).fetchdf(), limit)


def log_spikes_between(start: str, end: str, limit: int = 50) -> pd.DataFrame:
    validate_table("log_signatures")
    sql = """
        SELECT service, level, signature, start, "end", count
        FROM log_signatures
        WHERE start >= ? AND start < ?
        ORDER BY count DESC
        LIMIT ?
    """
    return clamp_rows(connection().execute(sql, [start, end, limit]).fetchdf(), limit)


def config_changes_between(start: str, end: str, limit: int = 50) -> pd.DataFrame:
    validate_table("config_changes")
    sql = """
        SELECT change_id, ts, service, config_key, old_value, new_value, changed_by, description
        FROM config_changes
        WHERE ts >= ? AND ts < ?
        ORDER BY ts
        LIMIT ?
    """
    return clamp_rows(connection().execute(sql, [start, end, limit]).fetchdf(), limit)


def data_quality_flags(metric: str, start: str, end: str) -> list[str]:
    """Null spikes, duplicate loads and late-arriving data on the target window."""
    validate_table("fact_daily")
    flags: list[str] = []
    row = connection().execute(
        """
        SELECT
            COUNT(*) AS rows,
            SUM(CASE WHEN revenue IS NULL THEN 1 ELSE 0 END) AS null_revenue,
            SUM(CASE WHEN sessions IS NULL OR sessions < 0 THEN 1 ELSE 0 END) AS bad_sessions,
            COUNT(DISTINCT date) AS distinct_days
        FROM fact_daily
        WHERE date BETWEEN ? AND ?
        """,
        [start, end],
    ).fetchdf()
    if row.empty:
        return ["no rows in window"]
    r = row.iloc[0]
    if int(r["null_revenue"] or 0) > 0:
        flags.append(f"{int(r['null_revenue'])} rows with null revenue")
    if int(r["bad_sessions"] or 0) > 0:
        flags.append(f"{int(r['bad_sessions'])} rows with missing or negative sessions")
    expected_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    if int(r["distinct_days"]) < expected_days:
        flags.append(
            f"missing {expected_days - int(r['distinct_days'])} day(s) of data in window"
        )
    dup = connection().execute(
        """
        SELECT COUNT(*) AS dup_groups FROM (
            SELECT date, country, platform, app_version, payment_method, channel,
                   product_category, COUNT(*) AS n
            FROM fact_daily
            WHERE date BETWEEN ? AND ?
            GROUP BY ALL
            HAVING COUNT(*) > 1
        )
        """,
        [start, end],
    ).fetchdf()
    if int(dup.iloc[0]["dup_groups"] or 0) > 0:
        flags.append(f"{int(dup.iloc[0]['dup_groups'])} duplicated dimension rows")
    return flags


def safe_json(value: Any) -> str:
    return json.dumps(value, default=str)
