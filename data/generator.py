"""Synthetic but internally coherent dataset for Lumen, an India-first mobile
commerce platform.

The causal chain is modelled explicitly so the data is genuinely explainable:

    sessions -> payment attempts -> payment successes -> orders -> revenue

Every metric in the catalog is a consequence of that funnel, which is what makes
the volume-vs-rate decomposition in the agent defensible rather than decorative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

END_DATE = pd.Timestamp("2026-09-26")
HISTORY_DAYS = 180
START_DATE = END_DATE - pd.Timedelta(days=HISTORY_DAYS - 1)

RELEASES = [
    ("4.9.0", "2026-04-10"),
    ("4.10.0", "2026-05-08"),
    ("4.11.0", "2026-06-12"),
    ("4.11.1", "2026-08-07"),
    ("4.12.0", "2026-09-26"),
]

COUNTRIES = {
    "IN": {"weight": 0.66, "aov_factor": 1.04, "cr_factor": 1.14, "android": 0.66},
    "ID": {"weight": 0.11, "aov_factor": 0.74, "cr_factor": 0.94, "android": 0.71},
    "BR": {"weight": 0.08, "aov_factor": 0.98, "cr_factor": 1.02, "android": 0.31},
    "US": {"weight": 0.07, "aov_factor": 1.86, "cr_factor": 1.36, "android": 0.34},
    "DE": {"weight": 0.05, "aov_factor": 1.72, "cr_factor": 1.20, "android": 0.24},
    "GB": {"weight": 0.03, "aov_factor": 1.66, "cr_factor": 1.22, "android": 0.29},
}

PAYMENT_METHODS = {
    "IN": {"upi": 0.64, "card": 0.19, "netbanking": 0.10, "wallet": 0.07},
    "ID": {"qris": 0.58, "card": 0.24, "wallet": 0.12, "bank_transfer": 0.06},
    "BR": {"pix": 0.55, "card": 0.37, "boleto": 0.08},
    "US": {"card": 0.66, "wallet": 0.22, "paypal": 0.12},
    "DE": {"card": 0.58, "paypal": 0.24, "sepa": 0.18},
    "GB": {"card": 0.71, "paypal": 0.29},
}

PLATFORMS = ["android", "ios", "web"]
CHANNELS = {
    "organic": 0.34,
    "paid": 0.26,
    "email": 0.13,
    "social": 0.15,
    "direct": 0.12,
}
CATEGORIES = {
    "electronics": {"weight": 0.24, "aov_factor": 2.10, "cr_factor": 0.78},
    "fashion": {"weight": 0.28, "aov_factor": 0.72, "cr_factor": 1.24},
    "home": {"weight": 0.19, "aov_factor": 1.18, "cr_factor": 0.96},
    "beauty": {"weight": 0.17, "aov_factor": 0.66, "cr_factor": 1.35},
    "grocery": {"weight": 0.12, "aov_factor": 0.48, "cr_factor": 1.48},
}

BASE_SESSIONS = 412_000
BASE_CONVERSION = 0.0292
BASE_AOV = 71.0
BASE_PAYMENT_SUCCESS = 0.941
REFUND_RATE = 0.021


def _dates() -> pd.DatetimeIndex:
    return pd.date_range(START_DATE, END_DATE, freq="D")


def _version_for(dates: pd.DatetimeIndex) -> np.ndarray:
    out = np.empty(len(dates), dtype=object)
    for i, d in enumerate(dates.date):
        current = RELEASES[0][0]
        for version, released in RELEASES:
            if d >= pd.Timestamp(released).date():
                current = version
        out[i] = current
    return out


def _combo_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for country, cfg in COUNTRIES.items():
        android_share = cfg["android"]  # type: ignore[index]
        platform_weights = {
            "android": android_share,
            "ios": (1 - android_share) * 0.66,
            "web": (1 - android_share) * 0.34,
        }
        for platform, p_weight in platform_weights.items():
            for method, m_weight in PAYMENT_METHODS[country].items():
                for channel, c_weight in CHANNELS.items():
                    for category, cat_cfg in CATEGORIES.items():
                        rows.append(
                            {
                                "country": country,
                                "platform": platform,
                                "payment_method": method,
                                "channel": channel,
                                "product_category": category,
                                "weight": (
                                    cfg["weight"]  # type: ignore[index]
                                    * p_weight
                                    * m_weight
                                    * c_weight
                                    * cat_cfg["weight"]
                                ),
                                "aov_factor": cfg["aov_factor"] * cat_cfg["aov_factor"],  # type: ignore[index]
                                "cr_factor": cfg["cr_factor"] * cat_cfg["cr_factor"],  # type: ignore[index]
                            }
                        )
    return pd.DataFrame(rows)


def _day_factors(dates: pd.DatetimeIndex, rng: np.random.Generator) -> dict[str, np.ndarray]:
    n = len(dates)
    growth = 1.0 + np.linspace(0.0, 0.34, n)
    weekly = np.where(dates.dayofweek >= 5, 1.11, 1.0)
    monthly_promo = np.where(dates.day == 15, 1.34, 1.0)
    monthly_promo = np.where(dates.day == 16, 1.19, monthly_promo)
    noise = 1.0 + rng.normal(0.0, 0.021, n)
    return {
        "sessions": growth * weekly * monthly_promo * noise,
        "conversion": 1.0 + np.linspace(0.0, 0.06, n) + rng.normal(0.0, 0.013, n),
        "aov": 1.0 + rng.normal(0.0, 0.017, n),
        "payment": 1.0 + rng.normal(0.0, 0.0055, n),
    }


def build_daily_facts(seed: int = 20260926) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = _dates()
    combos = _combo_frame()
    factors = _day_factors(dates, rng)

    grid = pd.MultiIndex.from_product(
        [dates, range(len(combos))], names=["date", "combo_idx"]
    )
    df = grid.to_frame(index=False)
    df = df.join(combos, on="combo_idx")

    n_dates = len(dates)
    df["app_version"] = _version_for(dates)[np.repeat(np.arange(n_dates), len(combos))]

    n_rows = len(df)
    day_idx = np.repeat(np.arange(n_dates), len(combos))

    sessions = (
        BASE_SESSIONS
        * df["weight"].to_numpy()
        * factors["sessions"][day_idx]
        * (1.0 + rng.normal(0.0, 0.045, n_rows))
    )
    sessions = np.maximum(np.round(sessions), 1.0)

    conversion = (
        BASE_CONVERSION
        * df["cr_factor"].to_numpy()
        * factors["conversion"][day_idx]
        * (1.0 + rng.normal(0.0, 0.031, n_rows))
    )
    conversion = np.clip(conversion, 0.002, 0.24)

    aov = (
        BASE_AOV
        * df["aov_factor"].to_numpy()
        * factors["aov"][day_idx]
        * (1.0 + rng.normal(0.0, 0.038, n_rows))
    )
    aov = np.maximum(aov, 4.0)

    pay_success = np.clip(
        BASE_PAYMENT_SUCCESS * factors["payment"][day_idx] * (1.0 + rng.normal(0.0, 0.006, n_rows)),
        0.2,
        0.999,
    )

    orders = np.maximum(np.round(sessions * conversion), 0.0)
    attempts = np.maximum(np.round(orders / pay_success), 0.0)
    successes = np.minimum(orders, attempts)

    df["sessions"] = sessions.astype("int64")
    df["orders"] = orders.astype("int64")
    df["payment_attempts"] = attempts.astype("int64")
    df["payment_successes"] = successes.astype("int64")
    df["aov"] = np.round(aov, 3)
    df["revenue"] = np.round(orders * aov, 2)
    df["revenue_gross"] = np.round(orders * aov / (1 - REFUND_RATE), 2)
    df["refund_amount"] = np.round(orders * aov / (1 - REFUND_RATE) - orders * aov, 2)
    df["order_value"] = np.round(aov, 3)
    df["payment_success_rate"] = np.where(
        attempts > 0, np.round(successes / np.maximum(attempts, 1.0), 5), np.nan
    )

    ordered = [
        "date",
        "country",
        "platform",
        "app_version",
        "payment_method",
        "channel",
        "product_category",
        "sessions",
        "orders",
        "revenue",
        "revenue_gross",
        "refund_amount",
        "aov",
        "payment_attempts",
        "payment_successes",
        "payment_success_rate",
        "order_value",
    ]
    return df[ordered].sort_values(ordered[:7]).reset_index(drop=True)


def build_business_events() -> pd.DataFrame:
    rows = [
        {
            "event_id": "EVT-0001",
            "ts": pd.Timestamp("2026-05-08 09:12:00"),
            "type": "release",
            "service": "checkout-service",
            "description": "checkout-service v4.10.0 rolled out to 100% - idempotency keys for split payments",
            "owner": "platform-eng",
            "severity": "routine",
            "metadata": json_of(version="4.10.0", rollout="100%"),
        },
        {
            "event_id": "EVT-0002",
            "ts": pd.Timestamp("2026-06-01 00:00:00"),
            "type": "pricing",
            "service": "growth",
            "description": "Free delivery threshold raised from Rs 499 to Rs 599 in metro PIN codes",
            "owner": "growth",
            "severity": "planned",
            "metadata": json_of(metric_affected="aov", direction="up"),
        },
        {
            "event_id": "EVT-0003",
            "ts": pd.Timestamp("2026-06-12 07:40:00"),
            "type": "release",
            "service": "checkout-service",
            "description": "checkout-service v4.11.0 - UPI mandate flow, saved payment instruments",
            "owner": "platform-eng",
            "severity": "routine",
            "metadata": json_of(version="4.11.0", rollout="100%"),
        },
        {
            "event_id": "EVT-0004",
            "ts": pd.Timestamp("2026-07-15 00:00:00"),
            "type": "campaign",
            "service": "growth",
            "description": "Monsoon Essentials mega-sale across fashion and home",
            "owner": "growth",
            "severity": "planned",
            "metadata": json_of(campaign="monsoon-essentials"),
        },
        {
            "event_id": "EVT-0005",
            "ts": pd.Timestamp("2026-08-07 11:25:00"),
            "type": "release",
            "service": "checkout-service",
            "description": "checkout-service v4.11.1 - patch for voucher stacking in cart",
            "owner": "platform-eng",
            "severity": "routine",
            "metadata": json_of(version="4.11.1", rollout="100%"),
        },
        {
            "event_id": "EVT-0006",
            "ts": pd.Timestamp("2026-08-19 00:00:00"),
            "type": "vendor_incident",
            "service": "payments-partner:paytm-payments",
            "description": "Paytm payment partner elevated error rates 18:00-21:30 IST, auto-failed over to UPI",
            "owner": "payments-platform",
            "severity": "degraded",
            "metadata": json_of(partner="paytm", duration_hours=3.5),
        },
        {
            "event_id": "EVT-0007",
            "ts": pd.Timestamp("2026-09-03 10:00:00"),
            "type": "campaign",
            "service": "growth",
            "description": "Back-to-school pricing on electronics, 5-12% discount",
            "owner": "growth",
            "severity": "planned",
            "metadata": json_of(campaign="back-to-school"),
        },
        {
            "event_id": "EVT-0008",
            "ts": pd.Timestamp("2026-09-26 05:40:00"),
            "type": "release",
            "service": "checkout-service",
            "description": (
                "checkout-service v4.12.0 rolled out to 100% - bundles UPI SDK 4.12.0 "
                "with reduced keepalive timeout"
            ),
            "owner": "platform-eng",
            "severity": "routine",
            "metadata": json_of(version="4.12.0", rollout="100%", dependency="upi-sdk@4.12.0"),
        },
    ]
    return pd.DataFrame(rows)


def json_of(**kwargs: object) -> str:
    import json

    return json.dumps(kwargs)


def build_config_changes() -> pd.DataFrame:
    """Runtime configuration changes.

    Deliberately a separate surface from business_events: most config pushes are
    routine, and a change that materially alters behaviour can be delivered
    here without a deploy ever appearing in the release log. That gap is the
    reason an investigator can be looking at a metric collapse with no release
    to point at.
    """
    rows = [
        {
            "change_id": "CFG-0001",
            "ts": pd.Timestamp("2026-05-19 11:20:00"),
            "service": "checkout-service",
            "config_key": "cart.max_line_items",
            "old_value": "50",
            "new_value": "50",
            "changed_by": "platform-eng",
            "description": "no-op re-apply of cart limits during routine config sync",
        },
        {
            "change_id": "CFG-0002",
            "ts": pd.Timestamp("2026-06-24 09:05:00"),
            "service": "search-service",
            "config_key": "search.result_window",
            "old_value": "24",
            "new_value": "48",
            "changed_by": "search-eng",
            "description": "widened search result window for relevance tuning",
        },
        {
            "change_id": "CFG-0003",
            "ts": pd.Timestamp("2026-07-28 16:40:00"),
            "service": "checkout-service",
            "config_key": "payments.retry.max_attempts",
            "old_value": "2",
            "new_value": "3",
            "changed_by": "payments-platform",
            "description": "raised card retry attempts to reduce soft declines",
        },
        {
            "change_id": "CFG-0004",
            "ts": pd.Timestamp("2026-08-25 13:10:00"),
            "service": "checkout-service",
            "config_key": "cart.abandon_prompt_delay_s",
            "old_value": "30",
            "new_value": "45",
            "changed_by": "growth",
            "description": "delayed abandon-cart prompt, campaign experiment",
        },
        {
            "change_id": "CFG-0005",
            "ts": pd.Timestamp("2026-09-11 10:25:00"),
            "service": "checkout-service",
            "config_key": "payments.tax_rounding",
            "old_value": "half_up",
            "new_value": "half_up",
            "changed_by": "payments-platform",
            "description": "no-op re-apply of tax rounding during routine config sync",
        },
    ]
    return pd.DataFrame(rows)


def build_log_signatures() -> pd.DataFrame:
    rows = [
        {
            "signature": "PaymentGatewayTimeout: upi-sdk (504)",
            "service": "checkout-service",
            "level": "error",
            "typical_count_day": 120,
        },
        {
            "signature": "PaymentDeclined: insufficient_instrument (402)",
            "service": "checkout-service",
            "level": "warn",
            "typical_count_day": 900,
        },
        {
            "signature": "CartValidationFailed: voucher_stacking (422)",
            "service": "checkout-service",
            "level": "error",
            "typical_count_day": 40,
        },
        {
            "signature": "InventoryReservationFailed: oversell_race (409)",
            "service": "inventory",
            "level": "error",
            "typical_count_day": 70,
        },
        {
            "signature": "UpstreamTimeout: search-index (504)",
            "service": "search",
            "level": "error",
            "typical_count_day": 15,
        },
    ]
    return pd.DataFrame(rows)
