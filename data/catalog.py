"""Semantic catalog.

This is what makes the agent metric-agnostic: onboarding a KPI is a catalog row,
not a code change. Tools read their allow-listed dimensions and relationships
from here, never from hardcoded logic.
"""

from __future__ import annotations

import json
from typing import Any

from core.config import DATA_DIR

CORE_DIMENSIONS = [
    "country",
    "platform",
    "app_version",
    "payment_method",
    "channel",
    "product_category",
]

CATALOG_PATH = DATA_DIR / "dim_catalog.json"

CATALOG: list[dict[str, Any]] = [
    {
        "metric": "daily_revenue",
        "description": (
            "Net recognised revenue per day, in USD. Sum of order_value for orders "
            "with status='paid', minus refunds refunded on that day. Excludes tax and "
            "shipping. This is the primary business KPI monitored by the agent."
        ),
        "unit": "USD",
        "aggregation": "sum",
        "grain": "day",
        "dimensions": CORE_DIMENSIONS,
        "related_metrics": [
            "orders",
            "sessions",
            "conversion_rate",
            "aov",
            "payment_success_rate",
        ],
        "identity": "daily_revenue = sessions * conversion_rate * aov",
        "pii_columns": [],
        "owner": "revenue-operations",
    },
    {
        "metric": "orders",
        "description": "Count of orders created per day across all statuses.",
        "unit": "count",
        "aggregation": "sum",
        "grain": "day",
        "dimensions": CORE_DIMENSIONS,
        "related_metrics": ["daily_revenue", "conversion_rate", "aov"],
        "identity": "orders = sessions * conversion_rate",
        "pii_columns": [],
        "owner": "revenue-operations",
    },
    {
        "metric": "sessions",
        "description": "Count of unique storefront sessions per day. Upstream of conversion.",
        "unit": "count",
        "aggregation": "sum",
        "grain": "day",
        "dimensions": CORE_DIMENSIONS,
        "related_metrics": ["orders", "conversion_rate", "daily_revenue"],
        "identity": "traffic input to conversion_rate",
        "pii_columns": [],
        "owner": "growth-analytics",
    },
    {
        "metric": "conversion_rate",
        "description": (
            "orders divided by sessions. Ratio metric: recompute the ratio from the "
            "underlying numerator and denominator, never average the daily rates."
        ),
        "unit": "ratio",
        "aggregation": "ratio",
        "grain": "day",
        "ratio_numerator": "orders",
        "ratio_denominator": "sessions",
        "dimensions": CORE_DIMENSIONS,
        "related_metrics": ["orders", "sessions", "payment_success_rate", "daily_revenue"],
        "identity": "conversion_rate = orders / sessions",
        "pii_columns": [],
        "owner": "growth-analytics",
    },
    {
        "metric": "aov",
        "description": "Average order value. Ratio metric: daily_revenue divided by orders.",
        "unit": "USD",
        "aggregation": "ratio",
        "grain": "day",
        "ratio_numerator": "daily_revenue",
        "ratio_denominator": "orders",
        "dimensions": CORE_DIMENSIONS,
        "related_metrics": ["daily_revenue", "orders"],
        "identity": "aov = daily_revenue / orders",
        "pii_columns": [],
        "owner": "revenue-operations",
    },
    {
        "metric": "payment_success_rate",
        "description": (
            "Successful payment attempts divided by total payment attempts. The "
            "leading indicator for conversion_rate: a drop here precedes a "
            "conversion drop with no lag on the same day."
        ),
        "unit": "ratio",
        "aggregation": "ratio",
        "grain": "day",
        "ratio_numerator": "payment_successes",
        "ratio_denominator": "payment_attempts",
        "dimensions": CORE_DIMENSIONS,
        "related_metrics": ["conversion_rate", "orders", "daily_revenue"],
        "identity": "payment_success_rate = payment_successes / payment_attempts",
        "pii_columns": [],
        "owner": "payments-platform",
    },
]

CATALOG_BY_METRIC: dict[str, dict[str, Any]] = {c["metric"]: c for c in CATALOG}


def get_catalog_entry(metric: str) -> dict[str, Any] | None:
    return CATALOG_BY_METRIC.get(metric)


def allowed_dimensions(metric: str) -> list[str]:
    entry = get_catalog_entry(metric)
    return list(entry["dimensions"]) if entry else []


def is_ratio(metric: str) -> bool:
    entry = get_catalog_entry(metric)
    return bool(entry) and entry.get("aggregation") == "ratio"


def ratio_parts(metric: str) -> tuple[str, str] | None:
    entry = get_catalog_entry(metric)
    if not entry or entry.get("aggregation") != "ratio":
        return None
    return entry["ratio_numerator"], entry["ratio_denominator"]


def related_metrics(metric: str) -> list[str]:
    entry = get_catalog_entry(metric)
    return list(entry.get("related_metrics", [])) if entry else []


def write_catalog() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CATALOG_PATH.write_text(json.dumps(CATALOG, indent=2), encoding="utf-8")
