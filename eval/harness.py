"""Shared harness for running one scenario down the deterministic path.

Both the A/B comparison and the learning curve need the same thing: a run whose
only variable is the memory it can see, summarised in a way that can be diffed
across arms. Keeping it in one place is what makes those comparisons
comparable -- two copies of the scoping/recall/gather/score sequence drift.

The deterministic path is the path production actually takes when Groq rate
limits force the fallback, so a result here is a result a real user gets, not a
result that only exists in a test.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tools.investigation  # noqa: F401,E402
from tools import sql as tools_sql  # noqa: E402

from agent.orchestrator import Investigation  # noqa: E402
from core.config import get_settings  # noqa: E402
from data import build as build_mod  # noqa: E402
from data.alerts import build_alert  # noqa: E402
from memory.memory_service import MemoryService  # noqa: E402

# An identifier a hypothesis can blame beyond its cause category. Only the first
# one is reported: these runs emit one candidate per config change, and the whole
# point of a comparison is which of them the prior promotes.
_IDENTIFIER = re.compile(r"\b(?:v?\d+\.\d+\.\d+|[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+)\b")


def bench_db_path(scenario_id: str) -> Path:
    return PROJECT_ROOT / "artifacts" / f"bench_{scenario_id}.duckdb"


def ensure_db(scenario_id: str) -> Path:
    """Build the scenario warehouse if it is not already on disk."""
    path = bench_db_path(scenario_id)
    if path.exists():
        return path
    print(f"  building {path.name} ...")
    argv = sys.argv
    sys.argv = ["build", "--scenarios", scenario_id, "--rebuild", "--db", str(path)]
    try:
        build_mod.main()
    finally:
        sys.argv = argv
    return path


def mechanism(hypothesis) -> str:
    """The specific identifier a top hypothesis names, or "-" if it names none."""
    match = _IDENTIFIER.search(hypothesis.statement or "")
    return match.group(0) if match else "-"


def with_scenario_db(scenario_id: str, run):
    """Run `run` with LUMEN_DB_PATH pointed at this scenario's warehouse.

    Settings are cached per process, so the cache has to be cleared on the way
    in *and* on the way out; otherwise the next scenario silently reuses the
    previous one's connection and every result after the first is wrong.
    """
    ensure_db(scenario_id)
    previous_db = os.environ.get("LUMEN_DB_PATH")
    os.environ["LUMEN_DB_PATH"] = str(bench_db_path(scenario_id))
    get_settings.cache_clear()
    try:
        return run()
    finally:
        if previous_db is None:
            os.environ.pop("LUMEN_DB_PATH", None)
        else:
            os.environ["LUMEN_DB_PATH"] = previous_db
        get_settings.cache_clear()


def run_once(
    scenario_id: str,
    *,
    bank: str,
    memory_enabled: bool,
    on_event=None,
) -> dict[str, Any]:
    """One investigation, summarised for diffing across arms.

    `bank` is passed explicitly rather than read from settings so a caller can
    point a run at a bank of its own choosing -- that is how the learning curve
    varies how much history the agent has.
    """
    settings = get_settings()
    alert = build_alert(scenario_id, bench_db_path(scenario_id))
    memory = MemoryService(settings.hindsight, bank)
    investigation = Investigation(
        anomaly=alert,
        settings=settings,
        memory=memory,
        memory_enabled=memory_enabled,
        llm=None,
        on_event=on_event,
    )
    try:
        investigation._scoping()
        investigation._recall("after_scoping", investigation._signature_query())
        investigation._evidence_gathering()
        investigation._recall("after_evidence", investigation._evidence_query())
        recall_failed = any(
            entry.helped_reason.startswith("recall error")
            for entry in investigation.memory_used
        )
        investigation._evidence_driven_hypotheses()
        if not investigation.hypotheses:
            return {
                "recall_failed": recall_failed,
                "cause_type": None,
                "mechanism": "-",
                "confidence": 0.0,
                "priors": 0,
                "prior_relevance": 0.0,
                "candidates": 0,
                "evidence": len(investigation.evidence),
                "tool_calls": investigation.guard.calls_used,
                "scored": [],
            }
        scored = investigation._score(
            investigation._changepoint(), investigation._candidate_events()
        )
        scored.sort(key=lambda item: item[1].total, reverse=True)
        winner, confidence = scored[0]
        return {
            "recall_failed": recall_failed,
            "cause_type": winner.cause_type,
            "mechanism": mechanism(winner),
            "confidence": confidence.total,
            "priors": len(winner.memory_prior_ids or []),
            "prior_relevance": getattr(winner, "memory_prior_relevance", 0.0),
            "candidates": len(investigation.hypotheses),
            "evidence": len(investigation.evidence),
            "tool_calls": investigation.guard.calls_used,
            "scored": [(h.cause_type, mechanism(h), c.total) for h, c in scored],
        }
    finally:
        memory.close()
        # tools.sql caches one module-level connection, so without this every
        # scenario after the first would silently read the first one's data.
        tools_sql.close_connection()


def run_scenario(scenario_id: str, *, bank: str, memory_enabled: bool) -> dict[str, Any]:
    """Point the process at this scenario's warehouse, then run once."""
    def _run():
        return run_once(
            scenario_id, bank=bank, memory_enabled=memory_enabled
        )

    return with_scenario_db(scenario_id, _run)
