"""DuckDB access.

Read-only for the agent: the tool layer opens the connection in read_only mode
so an LLM-driven code path can never mutate the warehouse.
"""

from __future__ import annotations

from typing import Any

import duckdb
import pandas as pd

from core.config import Settings, get_settings

SCHEMA = """
CREATE OR REPLACE TABLE fact_daily AS
SELECT * FROM read_parquet('{fact_path}');

CREATE OR REPLACE TABLE business_events AS
SELECT * FROM read_parquet('{events_path}');

CREATE OR REPLACE TABLE log_signatures AS
SELECT * FROM read_parquet('{logs_path}');

CREATE OR REPLACE TABLE config_changes AS
SELECT * FROM read_parquet('{config_path}');

CREATE OR REPLACE TABLE investigations (
    investigation_id VARCHAR PRIMARY KEY,
    alert_id        VARCHAR,
    tenant_id       VARCHAR,
    metric          VARCHAR,
    status          VARCHAR,
    started_at      TIMESTAMP,
    finished_at     TIMESTAMP,
    report_json     VARCHAR,
    feedback_json   VARCHAR,
    memory_enabled  BOOLEAN DEFAULT TRUE,
    tool_calls      INTEGER DEFAULT 0,
    tokens_used     INTEGER DEFAULT 0,
    top_cause       VARCHAR,
    cause_confirmed VARCHAR
);

CREATE OR REPLACE TABLE scenario_ground_truth (
    scenario_id       VARCHAR,
    root_cause        VARCHAR,
    cause_type        VARCHAR,
    affected_segment  VARCHAR,
    changepoint       VARCHAR,
    expected_impact   DOUBLE,
    applied_from      DATE
);
"""


def read_write_conn(db_path: str | None = None) -> duckdb.DuckDBPyConnection:
    path = db_path or str(get_settings().db_path)
    return duckdb.connect(path)


def read_only_conn(settings: Settings | None = None) -> duckdb.DuckDBPyConnection:
    s = settings or get_settings()
    return duckdb.connect(str(s.db_path), read_only=True)


def query(sql: str, params: list[Any] | None = None) -> pd.DataFrame:
    con = read_only_conn()
    try:
        return con.execute(sql, params or []).fetchdf()
    finally:
        con.close()
