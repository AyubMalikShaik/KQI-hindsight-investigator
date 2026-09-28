"""Build the DuckDB dataset, optionally with planted scenarios.

    python -m data.build --scenarios SCN-001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import ARTIFACTS_DIR, get_settings
from data import catalog as catalog_mod
from data import db, generator, scenarios


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="*", default=[], help="scenario ids to plant")
    parser.add_argument("--rebuild", action="store_true", help="delete the existing database first")
    parser.add_argument(
        "--db",
        help="write the warehouse here instead of the default path, so each incident "
        "can be benchmarked in isolation",
    )
    args = parser.parse_args()

    settings = get_settings()
    db_path = Path(args.db).resolve() if args.db else settings.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    staging = ARTIFACTS_DIR / "staging"
    staging.mkdir(parents=True, exist_ok=True)

    if args.rebuild and db_path.exists():
        db_path.unlink()
        for suffix in (".wal", ".tmp"):
            sidecar = db_path.with_suffix(db_path.suffix + suffix)
            if sidecar.exists():
                sidecar.unlink()

    print("generating daily facts ...")
    daily = generator.build_daily_facts()

    baseline = daily.copy()
    planted = scenarios.apply_scenarios(daily, args.scenarios)

    events = generator.build_business_events()
    planted_events = events.copy()
    for scenario_id in args.scenarios:
        scenario = scenarios.SCENARIOS[scenario_id]
        if scenario.events:
            planted_events = pd.concat([planted_events, pd.DataFrame(scenario.events)], ignore_index=True)

    logs = generator.build_log_signatures()
    spikes = [s for sid in args.scenarios for s in scenarios.SCENARIOS[sid].log_spikes]
    planted_logs = pd.concat([logs, pd.DataFrame(spikes)], ignore_index=True) if spikes else logs

    config = generator.build_config_changes()
    config_rows = [c for sid in args.scenarios for c in scenarios.SCENARIOS[sid].config_changes]
    planted_config = (
        pd.concat([config, pd.DataFrame(config_rows)], ignore_index=True) if config_rows else config
    )

    truth_rows = []
    for scenario_id in args.scenarios:
        truth = scenarios.SCENARIOS[scenario_id].truth
        truth_rows.append(
            {
                "scenario_id": truth.scenario_id,
                "root_cause": truth.root_cause,
                "cause_type": truth.cause_type,
                "affected_segment": json.dumps(truth.affected_segment),
                "changepoint": truth.changepoint,
                "expected_impact": truth.expected_metric_impact_pct,
                "applied_from": truth.changepoint,
            }
        )

    fact_path = staging / "fact_daily.parquet"
    events_path = staging / "business_events.parquet"
    logs_path = staging / "log_signatures.parquet"
    config_path = staging / "config_changes.parquet"
    truth_path = staging / "scenario_ground_truth.parquet"

    planted.to_parquet(fact_path, index=False)
    planted_events.to_parquet(events_path, index=False)
    planted_logs.to_parquet(logs_path, index=False)
    planted_config.to_parquet(config_path, index=False)
    if truth_rows:
        pd.DataFrame(truth_rows).to_parquet(truth_path, index=False)

    con = db.read_write_conn(str(db_path))
    try:
        con.execute(
            """
            CREATE OR REPLACE TABLE fact_daily AS SELECT * FROM read_parquet(?)
            """,
            [str(fact_path)],
        )
        con.execute(
            "CREATE OR REPLACE TABLE business_events AS SELECT * FROM read_parquet(?)",
            [str(events_path)],
        )
        con.execute(
            "CREATE OR REPLACE TABLE log_signatures AS SELECT * FROM read_parquet(?)",
            [str(logs_path)],
        )
        con.execute(
            "CREATE OR REPLACE TABLE config_changes AS SELECT * FROM read_parquet(?)",
            [str(config_path)],
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE scenario_ground_truth AS
            SELECT * FROM read_parquet(?)
            """,
            [str(truth_path)],
        ) if truth_rows else con.execute("CREATE OR REPLACE TABLE scenario_ground_truth (scenario_id VARCHAR)")
    finally:
        con.close()

    catalog_mod.write_catalog()

    report_impact(baseline, planted, args.scenarios)
    print(f"\ndatabase: {db_path}")
    print(f"rows in fact_daily: {len(planted):,}")


def report_impact(baseline: pd.DataFrame, planted: pd.DataFrame, scenario_ids: list[str]) -> None:
    if not scenario_ids:
        print("no scenarios planted; baseline dataset only")
        return

    def total(df: pd.DataFrame, day: str) -> float:
        return float(df.loc[df["date"] == pd.Timestamp(day), "revenue"].sum())

    anomaly_day = max(scenarios.SCENARIOS[s].truth.changepoint for s in scenario_ids)
    anomaly_ts = pd.Timestamp(anomaly_day)
    anomaly_total = total(planted, anomaly_day)
    baseline_same_day = total(baseline, anomaly_day)
    deviation = (anomaly_total - baseline_same_day) / baseline_same_day * 100

    same_weekday = [
        (anomaly_ts - pd.Timedelta(days=7 * k)).strftime("%Y-%m-%d") for k in (1, 2, 3, 4)
    ]
    weekday_baseline = sum(total(baseline, d) for d in same_weekday) / len(same_weekday)
    vs_weekday = (anomaly_total - weekday_baseline) / weekday_baseline * 100

    print(f"\n=== impact check ({', '.join(scenario_ids)}) ===")
    print(f"changepoint            : {anomaly_day} ({anomaly_ts.day_name()})")
    print(f"revenue that day      : ${anomaly_total:,.0f}")
    print(f"counterfactual (no bug): ${baseline_same_day:,.0f}  -> deviation {deviation:+.1f}%")
    print(f"vs same weekday -4wks : {vs_weekday:+.1f}%  (day-of-week aware)")

    truth = scenarios.SCENARIOS[scenario_ids[0]].truth if len(scenario_ids) == 1 else None
    seg = truth.affected_segment if truth else {"country": "IN", "payment_method": "upi"}
    seg_mask = np.ones(len(planted), dtype=bool)
    for column, value in seg.items():
        seg_mask &= (planted[column] == value).to_numpy()
    affected = planted.loc[seg_mask & (planted["date"] == anomaly_ts)]
    base_aff = baseline.loc[seg_mask & (baseline["date"] == anomaly_ts)]
    if len(affected):
        sess = float(affected["sessions"].sum())
        aff_rev = float(affected["revenue"].sum())
        lost = float(base_aff["revenue"].sum()) - aff_rev
        psr = float(affected["payment_successes"].sum() / affected["payment_attempts"].sum())
        base_psr = float(
            base_aff["payment_successes"].sum() / base_aff["payment_attempts"].sum()
        )
        label = " x ".join(f"{k}={v}" for k, v in seg.items())
        base_seg_rev = float(base_aff["revenue"].sum())
        print(f"affected segment  : {label}, {len(affected)} cells")
        print(f"  share of company : {base_seg_rev / baseline_same_day * 100:.1f}% of pre-incident revenue")
        print(f"  lost revenue    : ${lost:,.0f}  ({lost / baseline_same_day * 100:.1f}% of company)")
        print(f"  sessions        : {sess:,.0f}  (volume effect)")
        print(f"  pay success rate: {psr:.3f} vs {base_psr:.3f} baseline  (rate effect)")


if __name__ == "__main__":
    main()
