"""Check every planted incident resolves to the cause that was planted.

Each scenario is built into its own warehouse, because a shared one lets one
incident's planted effect contaminate another's baseline. The agent runs the
deterministic path (no LLM), because that is the path rate limits actually put
us on, and a verdict that only holds when the model cooperates is not a
verdict.

Memory is exercised for real against Hindsight: the assertion is that the
correct prior is attached to the winning hypothesis, not that confidence went
up by some amount.

    python -m tests.test_scenarios
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tools.investigation  # noqa: F401,E402
from tools import sql as tools_sql  # noqa: E402

from agent.orchestrator import Investigation  # noqa: E402
from core.config import get_settings  # noqa: E402
from data import build as build_mod  # noqa: E402
from data.alerts import build_alert  # noqa: E402
from data.scenarios import SCENARIOS  # noqa: E402
from memory.memory_service import MemoryService  # noqa: E402

# What the deterministic path is expected to conclude, per scenario. This is
# the whole point of the suite: each incident is diagnosable from tool evidence
# plus prior cases, and no incident is allowed to collapse into "unknown".
EXPECTED_CAUSE = {
    "SCN-001": "release_regression",
    "SCN-002": "config_change",
    "SCN-003": "data_quality",
    "SCN-004": "traffic_loss",
}

# Incidents whose correct answer cannot be reached from metrics and the event
# log alone, so a memory prior is the only route to them.
MEMORY_DECISIVE = {"SCN-002"}

# Incidents where the evidence supports more than one reading and the recalled
# history is what breaks the tie. Without the prior the run settles on the
# shallower symptom-level cause; with it, on the actual root cause. These are
# the cases that justify the memory weight, so a regression here is the one
# failure that matters: confidence drifting while the answer stays wrong is
# exactly the failure mode that a confidence-only comparison would hide.
MEMORY_CHANGES_CONCLUSION = {
    "SCN-001": ("payment_failure", "release_regression"),
}

# Incidents where the cause *type* is settled either way but the evidence
# carries two rival mechanisms of that same type, so scoring them is a tie and
# only the prior separates them. Recorded as the config key each run names:
# cause_type alone cannot see this, because both candidates are config_change
# and a type-level assertion would pass with memory switched off.
MEMORY_CHANGES_MECHANISM = {
    # Two config pushes land in the same window. Recency is the tie-break the
    # evidence-only run falls back on, and it points at the wrong one.
    "SCN-002": ("payment.upi.conn_pool_max", "payment.upi.keepalive_ms"),
}


def named_config_key(hypothesis) -> str | None:
    """The config key a config_change hypothesis actually blames, if any."""
    if hypothesis.cause_type != "config_change":
        return None
    for match in re.findall(r"payment\.upi\.[a-z_]+", hypothesis.statement):
        return match
    return None


def bench_db_path(scenario_id: str) -> Path:
    return PROJECT_ROOT / "artifacts" / f"bench_{scenario_id}.duckdb"


def ensure_db(scenario_id: str) -> Path:
    path = bench_db_path(scenario_id)
    if path.exists():
        return path
    print(f"  building {path.name} ...")
    argv = sys.argv
    sys.argv = [
        "build",
        "--scenarios",
        scenario_id,
        "--rebuild",
        "--db",
        str(path),
    ]
    try:
        build_mod.main()
    finally:
        sys.argv = argv
    return path


def investigate(scenario_id: str, db_path: Path, memory_enabled: bool):
    """Run scoping, evidence and the deterministic hypothesis path.

    Mirrors the state order in Investigation.run(), including both recall
    stages: memory that is never recalled cannot be cited, and the priors are
    exactly what this suite is checking.

    Returns (top scored hypothesis, all hypotheses, recall_succeeded). Recall
    is reported separately so that a Hindsight outage is never mistaken for
    memory failing to be useful.
    """
    settings = get_settings()
    alert = build_alert(scenario_id, db_path)
    memory = MemoryService(settings.hindsight, settings.bank)
    investigation = Investigation(
        anomaly=alert,
        settings=settings,
        memory=memory,
        memory_enabled=memory_enabled,
        on_event=None,
    )
    try:
        investigation._scoping()
        investigation._recall("after_scoping", investigation._signature_query())
        investigation._evidence_gathering()
        investigation._recall("after_evidence", investigation._evidence_query())
        # The orchestrator records transport failures in the reason field, so a
        # Hindsight outage is distinguishable from memory that was reachable
        # and still went unused.
        recall_errors = [
            entry.helped_reason
            for entry in investigation.memory_used
            if entry.helped_reason.startswith("recall error")
        ]
        recall_ok = not recall_errors
        for message in recall_errors:
            print(f"    recall     : FAILED ({message[:80]})")
        investigation._evidence_driven_hypotheses()
        if not investigation.hypotheses:
            return None, [], recall_ok
        changepoint = investigation._changepoint()
        scored = investigation._score(changepoint, investigation._candidate_events())
        scored.sort(key=lambda item: item[1].total, reverse=True)
        return scored[0], investigation.hypotheses, recall_ok
    finally:
        memory.close()
        # tools.sql caches one module-level connection, so without this every
        # scenario after the first would silently read the first one's data.
        tools_sql.close_connection()


def main() -> int:
    get_settings.cache_clear()
    failures: list[str] = []
    memory_confidence: dict[str, tuple[float, float]] = {}
    conclusion_flips: list[str] = []
    mechanism_flips: list[str] = []

    for scenario_id in sorted(SCENARIOS):
        truth = SCENARIOS[scenario_id].truth
        print(f"\n=== {scenario_id}: {SCENARIOS[scenario_id].title}")
        print(f"    planted cause: {truth.cause_type} -> {truth.root_cause[:70]}")

        db_path = ensure_db(scenario_id)
        previous_db = os.environ.get("LUMEN_DB_PATH")
        os.environ["LUMEN_DB_PATH"] = str(db_path)
        get_settings.cache_clear()
        try:
            with_memory, hypotheses, recall_ok = investigate(scenario_id, db_path, True)
            without_memory, _, _ = investigate(scenario_id, db_path, False)
        finally:
            if previous_db is None:
                os.environ.pop("LUMEN_DB_PATH", None)
            else:
                os.environ["LUMEN_DB_PATH"] = previous_db
            get_settings.cache_clear()

        if with_memory is None:
            failures.append(f"{scenario_id}: no hypothesis was produced")
            continue

        hypothesis, confidence = with_memory
        expected = EXPECTED_CAUSE[scenario_id]
        print(f"    concluded   : {hypothesis.cause_type} (confidence {confidence.total:.2f})")
        print(
            f"    breakdown   : support={confidence.evidence_support} "
            f"magnitude={confidence.magnitude_explained} "
            f"temporal={confidence.temporal_alignment} prior={confidence.memory_prior}"
        )
        print(
            f"    priors      : {len(hypothesis.memory_prior_ids)} recalled case(s) "
            f"cited by {hypothesis.hypothesis_id}"
        )
        if without_memory is not None and recall_ok:
            memory_confidence[scenario_id] = (without_memory[1].total, confidence.total)
            print(
                f"    no memory   : {without_memory[0].cause_type} "
                f"(confidence {without_memory[1].total:.2f}, "
                f"prior {without_memory[1].memory_prior})"
            )

        if hypothesis.cause_type != expected:
            failures.append(
                f"{scenario_id}: concluded {hypothesis.cause_type}, expected {expected}"
            )

        if scenario_id in MEMORY_DECISIVE and recall_ok and not hypothesis.memory_prior_ids:
            failures.append(
                f"{scenario_id}: memory-decisive incident settled with no prior cited"
            )

        if scenario_id in MEMORY_CHANGES_CONCLUSION and recall_ok and without_memory is not None:
            shallow, deep = MEMORY_CHANGES_CONCLUSION[scenario_id]
            if without_memory[0].cause_type != shallow or hypothesis.cause_type != deep:
                failures.append(
                    f"{scenario_id}: memory should move the conclusion {shallow} -> {deep}, "
                    f"got {without_memory[0].cause_type} -> {hypothesis.cause_type}"
                )
            else:
                conclusion_flips.append(scenario_id)

        if scenario_id in MEMORY_CHANGES_MECHANISM and recall_ok and without_memory is not None:
            wrong, right = MEMORY_CHANGES_MECHANISM[scenario_id]
            got_wrong, got_right = (
                named_config_key(without_memory[0]),
                named_config_key(hypothesis),
            )
            if got_wrong != wrong or got_right != right:
                failures.append(
                    f"{scenario_id}: memory should move the mechanism {wrong} -> {right}, "
                    f"got {got_wrong} -> {got_right}"
                )
            else:
                mechanism_flips.append(scenario_id)

    print("\n" + "=" * 74)
    if memory_confidence:
        print("top-hypothesis confidence, memory off vs on:")
        for scenario_id, (baseline, with_mem) in sorted(memory_confidence.items()):
            print(
                f"  {scenario_id}: {baseline:.2f} -> {with_mem:.2f} "
                f"(+{with_mem - baseline:.2f} from the memory prior)"
            )
    if conclusion_flips:
        print("\nconclusion changed by memory (the reason memory is weighted at all):")
        for scenario_id in conclusion_flips:
            shallow, deep = MEMORY_CHANGES_CONCLUSION[scenario_id]
            print(f"  {scenario_id}: {shallow} -> {deep}")
    if mechanism_flips:
        print("\nmechanism changed by memory (same cause type, rival candidates):")
        for scenario_id in mechanism_flips:
            wrong, right = MEMORY_CHANGES_MECHANISM[scenario_id]
            print(f"  {scenario_id}: {wrong} -> {right}")
    if failures:
        print("\nPROBLEMS:")
        for problem in failures:
            print(f"  - {problem}")
        return 1
    print("\nALL SCENARIOS RESOLVED TO THEIR PLANTED CAUSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
