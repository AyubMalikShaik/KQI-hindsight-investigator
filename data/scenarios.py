"""Planted scenarios.

Each scenario is a named intervention on the generated facts plus the ground
truth needed to score the agent automatically. Scenarios are applied in order,
so a later scenario can be layered on top of an earlier one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

PSR_MULTIPLIER = 0.40
ATTEMPT_RETRY_INFLATION = 1.08
ABANDON_AFTER_FAILURE = 0.93
SURVIVOR_AOV_UPLIFT = 1.06
REFUND_RATE = 0.021


@dataclass(frozen=True)
class GroundTruth:
    scenario_id: str
    root_cause: str
    cause_type: str
    affected_segment: dict[str, str]
    changepoint: str
    expected_metric_impact_pct: float
    decisive_tool: str
    distinguishing_evidence: list[str] = field(default_factory=list)
    is_data_quality_issue: bool = False


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    title: str
    description: str
    apply: Callable[[pd.DataFrame], pd.DataFrame]
    events: list[dict[str, Any]]
    log_spikes: list[dict[str, Any]]
    truth: GroundTruth
    config_changes: list[dict[str, Any]] = field(default_factory=list)


def _mask(df: pd.DataFrame, **conditions: str) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    for column, value in conditions.items():
        mask &= (df[column] == value).to_numpy()
    return mask


def _between(df: pd.DataFrame, start: str, end: str) -> np.ndarray:
    return ((df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))).to_numpy()


def _v12_upi_regression(df: pd.DataFrame) -> pd.DataFrame:
    """UPI SDK 4.12.0 keepalive regression: Indian UPI checkout fails.

    The SDK is a shared dependency of the web and mobile checkout, so the blast
    radius is every Indian UPI authorisation, not just the mobile app. Sessions
    are untouched, so this is a conversion-rate effect isolated to one
    country-and-payment-method, preceded by the release that shipped it.
    """
    df = df.copy()
    target = _mask(df, country="IN", payment_method="upi") & (
        df["date"] >= pd.Timestamp("2026-09-26")
    )
    if target.sum() == 0:
        return df

    sessions = df.loc[target, "sessions"].to_numpy()
    attempted_orders = df.loc[target, "orders"].to_numpy()
    base_psr = df.loc[target, "payment_success_rate"].fillna(1.0).to_numpy()
    aov = df.loc[target, "aov"].to_numpy() * SURVIVOR_AOV_UPLIFT

    degraded_psr = np.clip(base_psr * PSR_MULTIPLIER, 0.05, 0.999)
    attempts = np.maximum(np.round(attempted_orders * ATTEMPT_RETRY_INFLATION), 0.0)
    surviving_orders = np.maximum(
        np.round(attempts * degraded_psr * ABANDON_AFTER_FAILURE), 0.0
    )

    df.loc[target, "payment_attempts"] = attempts.astype("int64")
    df.loc[target, "payment_successes"] = surviving_orders.astype("int64")
    df.loc[target, "orders"] = surviving_orders.astype("int64")
    df.loc[target, "aov"] = np.round(aov, 3)
    df.loc[target, "revenue"] = np.round(surviving_orders * aov, 2)
    df.loc[target, "revenue_gross"] = np.round(surviving_orders * aov / (1 - REFUND_RATE), 2)
    df.loc[target, "refund_amount"] = np.round(
        surviving_orders * aov / (1 - REFUND_RATE) - surviving_orders * aov, 2
    )
    df.loc[target, "order_value"] = np.round(aov, 3)
    df.loc[target, "payment_success_rate"] = np.where(
        attempts > 0, np.round(surviving_orders / np.maximum(attempts, 1.0), 5), np.nan
    )
    return df


V12_UPI_REGRESSION = Scenario(
    scenario_id="SCN-001",
    title="UPI SDK 4.12.0 keepalive regression on Android",
    description=(
        "checkout-service v4.12.0 ships upi-sdk@4.12.0 with a reduced keepalive "
        "timeout. Indian UPI authorisations time out at the gateway across web and "
        "mobile. Sessions are unaffected, so this is a conversion-rate effect."
    ),
    apply=_v12_upi_regression,
    events=[
        {
            "event_id": "EVT-0101",
            "ts": pd.Timestamp("2026-09-26 06:05:00"),
            "type": "incident",
            "service": "checkout-service",
            "description": "UPI authorisation timeouts reported by support, 5xx from payment-gateway",
            "owner": "payments-platform",
            "severity": "critical",
            "metadata": json.dumps({"error_rate_5xx": 0.27, "region": "IN"}),
        }
    ],
    log_spikes=[
        {
            "service": "checkout-service",
            "level": "error",
            "signature": "PaymentGatewayTimeout: upi-sdk (504)",
            "start": "2026-09-26 05:55:00",
            "end": "2026-09-26 23:00:00",
            "count": 41_820,
        }
    ],
    truth=GroundTruth(
        scenario_id="SCN-001",
        root_cause="upi-sdk 4.12.0 keepalive regression introduced by checkout-service v4.12.0",
        cause_type="release_regression",
        affected_segment={"country": "IN", "payment_method": "upi"},
        changepoint="2026-09-26",
        expected_metric_impact_pct=-20.9,
        decisive_tool="breakdown_by_dimension",
        distinguishing_evidence=[
            "sessions flat in the affected segment (volume effect absent)",
            "payment_success_rate collapses from ~0.94 to ~0.38",
            "conversion_rate collapses only in IN x upi, all other segments flat",
            "deploy of checkout-service v4.12.0 at 05:40 UTC precedes the changepoint",
        ],
    ),
)


def _silent_upi_config_regression(df: pd.DataFrame) -> pd.DataFrame:
    """Same fingerprint as SCN-001, but no deploy: a config push did it.

    The distinguishing evidence is *absence*. The collapse is identical to the
    release-driven regression (IN x upi, sessions flat, payment success down),
    but nothing lands in the release log. The only artefact of the cause is a
    config change, which an investigator has no reason to query unless a prior
    incident has already told them that this failure mode arrives without a
    deploy.
    """
    df = df.copy()
    target = _mask(df, country="IN", payment_method="upi") & _between(
        df, "2026-07-09", "2026-08-12"
    )
    if target.sum() == 0:
        return df

    attempted_orders = df.loc[target, "orders"].to_numpy()
    base_psr = df.loc[target, "payment_success_rate"].fillna(1.0).to_numpy()
    aov = df.loc[target, "aov"].to_numpy() * SURVIVOR_AOV_UPLIFT

    degraded_psr = np.clip(base_psr * 0.44, 0.05, 0.999)
    attempts = np.maximum(np.round(attempted_orders * ATTEMPT_RETRY_INFLATION), 0.0)
    surviving_orders = np.maximum(
        np.round(attempts * degraded_psr * ABANDON_AFTER_FAILURE), 0.0
    )

    df.loc[target, "payment_attempts"] = attempts.astype("int64")
    df.loc[target, "payment_successes"] = surviving_orders.astype("int64")
    df.loc[target, "orders"] = surviving_orders.astype("int64")
    df.loc[target, "aov"] = np.round(aov, 3)
    df.loc[target, "revenue"] = np.round(surviving_orders * aov, 2)
    df.loc[target, "revenue_gross"] = np.round(surviving_orders * aov / (1 - REFUND_RATE), 2)
    df.loc[target, "refund_amount"] = np.round(
        surviving_orders * aov / (1 - REFUND_RATE) - surviving_orders * aov, 2
    )
    df.loc[target, "order_value"] = np.round(aov, 3)
    df.loc[target, "payment_success_rate"] = np.where(
        attempts > 0, np.round(surviving_orders / np.maximum(attempts, 1.0), 5), np.nan
    )
    return df


SILENT_UPI_CONFIG = Scenario(
    scenario_id="SCN-002",
    title="Silent UPI keepalive regression from a config push (no deploy)",
    description=(
        "A payments-platform config change lowers the UPI keepalive interval from "
        "30000ms to 5000ms. Indian UPI authorisations start timing out at the "
        "gateway. No release ships, so the release log is empty across the "
        "changepoint and the only record of the cause is a config change."
    ),
    apply=_silent_upi_config_regression,
    events=[],
    log_spikes=[
        {
            "service": "checkout-service",
            "level": "error",
            "signature": "PaymentGatewayTimeout: upi-sdk (504)",
            "start": "2026-07-09 05:50:00",
            "end": "2026-07-12 23:00:00",
            "count": 33_410,
        }
    ],
    truth=GroundTruth(
        scenario_id="SCN-002",
        root_cause=(
            "payment.upi.keepalive_ms config push lowered the UPI keepalive interval "
            "from 30000 to 5000, no deploy involved"
        ),
        # Canonical taxonomy name: Hypothesis.cause_type has no
        # "config_regression", and a ground-truth label the agent cannot emit
        # makes every comparison fail on the label rather than on the finding.
        cause_type="config_change",
        affected_segment={"country": "IN", "payment_method": "upi"},
        changepoint="2026-07-09",
        expected_metric_impact_pct=-13.4,
        decisive_tool="query_config_changes",
        distinguishing_evidence=[
            "no release or deploy event within days of the changepoint",
            "config change to payment.upi.keepalive_ms (30000 -> 5000) at 05:35 UTC",
            "config change to payment.upi.conn_pool_max (60 -> 30) at 05:44 UTC, "
            "the more recent of the two and the wrong one",
            "payment_success_rate collapses in IN x upi while sessions stay flat",
            "PaymentGatewayTimeout: upi-sdk (504) log spike begins 15 minutes after "
            "the keepalive change and six minutes after the pool change",
        ],
    ),
    config_changes=[
        {
            "change_id": "CFG-0101",
            "ts": pd.Timestamp("2026-07-09 05:35:00"),
            "service": "checkout-service",
            "config_key": "payment.upi.keepalive_ms",
            "old_value": "30000",
            "new_value": "5000",
            "changed_by": "payments-platform",
            "description": "lowered UPI keepalive interval during gateway latency tuning",
        },
        # A second, genuinely plausible change in the same window. Tightening the
        # UPI connection pool would also produce gateway timeouts, and it lands
        # six minutes before the symptom rather than fifteen, so recency alone
        # points at this one. Metrics cannot separate the two: both are config
        # pushes, both precede the changepoint, both touch payment.upi. This is
        # the ambiguity the memory prior has to resolve, and it is why the
        # incident needs two changes rather than one smoking gun.
        {
            "change_id": "CFG-0102",
            "ts": pd.Timestamp("2026-07-09 05:44:00"),
            "service": "checkout-service",
            "config_key": "payment.upi.conn_pool_max",
            "old_value": "60",
            "new_value": "30",
            "changed_by": "payments-platform",
            "description": "tightened the UPI gateway connection pool during capacity tuning",
        },
    ],
)


def _revenue_recognition_failure(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue falls while every volume and rate metric stays healthy.

    A demand or payment fault cannot move revenue alone: sessions, orders,
    conversion and payment success all hold flat, so average order value is
    forced down as the residual. Nothing about the customer behaviour changed,
    which means the drop is in how revenue was recognised, not in revenue itself.
    """
    df = df.copy()
    target = _between(df, "2026-08-13", "2026-09-04")
    if target.sum() == 0:
        return df

    factor = 0.82
    for column in ("revenue", "revenue_gross", "refund_amount", "aov", "order_value"):
        df.loc[target, column] = np.round(df.loc[target, column].to_numpy() * factor, 4)
    return df


REVENUE_RECOGNITION_FAILURE = Scenario(
    scenario_id="SCN-003",
    title="Revenue recognition shortfall reported as a revenue collapse",
    description=(
        "A settlement batch for 2026-08-13 fails to commit, so 18% of the day's "
        "revenue is never recognised. Sessions, orders, conversion and payment "
        "success are all healthy, so the only movement is revenue and the aov "
        "residual it implies. This is a measurement failure, not a business one."
    ),
    apply=_revenue_recognition_failure,
    events=[],
    log_spikes=[
        {
            "service": "settlement-worker",
            "level": "warn",
            "signature": "SettlementBatch: commit lag exceeded retry budget",
            "start": "2026-08-13 02:10:00",
            "end": "2026-08-14 01:40:00",
            "count": 1_284,
        }
    ],
    truth=GroundTruth(
        scenario_id="SCN-003",
        root_cause=(
            "settlement batch commit failure left 18% of 2026-08-13 revenue "
            "unrecognised; the business was healthy"
        ),
        cause_type="data_quality",
        affected_segment={"country": "ALL"},
        changepoint="2026-08-13",
        expected_metric_impact_pct=-18.0,
        decisive_tool="find_related_metrics",
        distinguishing_evidence=[
            "sessions, orders and conversion_rate are flat across the changepoint",
            "payment_success_rate is flat: no payment fault",
            "revenue falls uniformly across every country and channel",
            "aov falls as the arithmetic residual of revenue / orders",
        ],
        is_data_quality_issue=True,
    ),
)


def _paid_channel_traffic_loss(df: pd.DataFrame) -> pd.DataFrame:
    """Volume leaves through one acquisition channel. Nothing else moves.

    Sessions, orders, revenue, payment attempts and payment successes all fall
    by the same factor inside channel=paid, while average order value,
    conversion and payment success hold. Every rate is healthy, so the only
    correct reading is a traffic mix change rather than a checkout or payment
    fault.
    """
    df = df.copy()
    target = _mask(df, channel="paid") & _between(df, "2026-09-05", "2026-09-26")
    if target.sum() == 0:
        return df

    factor = 0.72
    for column in (
        "sessions",
        "orders",
        "payment_attempts",
        "payment_successes",
        "revenue",
        "revenue_gross",
        "refund_amount",
    ):
        df.loc[target, column] = np.maximum(
            np.round(df.loc[target, column].to_numpy() * factor), 0.0
        )
    return df


PAID_CHANNEL_TRAFFIC_LOSS = Scenario(
    scenario_id="SCN-004",
    title="Paid acquisition traffic loss (negative control)",
    description=(
        "Sessions in channel=paid drop 28% and carry orders, revenue, payment "
        "attempts and payment successes down with them. Average order value, "
        "conversion and payment success are unchanged. This is a volume effect "
        "confined to one acquisition channel and must not be reported as a "
        "checkout, payment or data-quality fault."
    ),
    apply=_paid_channel_traffic_loss,
    events=[],
    log_spikes=[],
    truth=GroundTruth(
        scenario_id="SCN-004",
        root_cause=(
            "traffic loss in channel=paid: sessions -28% with conversion, aov and "
            "payment success unchanged, an external acquisition mix change rather "
            "than a platform fault"
        ),
        cause_type="traffic_loss",
        affected_segment={"channel": "paid"},
        changepoint="2026-09-05",
        expected_metric_impact_pct=-7.4,
        decisive_tool="breakdown_by_dimension",
        distinguishing_evidence=[
            "sessions fall only in channel=paid, all other channels flat",
            "conversion_rate and aov are unchanged",
            "payment_success_rate is unchanged",
            "orders, attempts and successes all fall by the same factor as sessions",
        ],
    ),
)


SCENARIOS: dict[str, Scenario] = {
    s.scenario_id: s
    for s in [
        V12_UPI_REGRESSION,
        SILENT_UPI_CONFIG,
        REVENUE_RECOGNITION_FAILURE,
        PAID_CHANNEL_TRAFFIC_LOSS,
    ]
}


def apply_scenarios(df: pd.DataFrame, scenario_ids: list[str]) -> pd.DataFrame:
    for scenario_id in scenario_ids:
        df = SCENARIOS[scenario_id].apply(df)
    return df
