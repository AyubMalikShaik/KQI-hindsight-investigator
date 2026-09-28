"""Run one investigation end to end.

    python -m run_investigation
    python -m run_investigation --no-memory
    python -m run_investigation --scenario SCN-002
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import tools.investigation  # noqa: E402,F401  (registers the investigation tools)

from agent.orchestrator import Investigation, TraceEvent  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.schemas import Anomaly, Direction, Severity, Window  # noqa: E402
from data.alerts import build_alert, describe_ground_truth  # noqa: E402
from data.scenarios import SCENARIOS  # noqa: E402
from memory.memory_service import MemoryService  # noqa: E402

DEFAULT_SCENARIO = "SCN-001"

ALERT = Anomaly(
    alert_id="ALR-2026-0142",
    metric="daily_revenue",
    grain="day",
    detected_at=datetime(2026, 9, 27, 6, 0, 0, tzinfo=timezone.utc),
    anomaly_window=Window(start="2026-09-26", end="2026-09-26"),
    baseline_window=Window(start="2026-08-29", end="2026-09-25"),
    observed=1_547_169.0,
    expected=1_856_660.0,
    deviation_pct=-16.7,
    direction=Direction.drop,
    severity=Severity.high,
    source="grafana",
)

ICONS = {
    "state": ">>",
    "evidence": "  [evidence]",
    "hypothesis": "  [hypothesis]",
    "verdict": "  [verdict]",
    "memory_recall": "  [memory]",
    "memory_skipped": "  [memory]",
    "guard": "  [guard]",
    "plan": "  [plan]",
    "llm_parse_error": "  [warn]",
    "citation_violation": "  [warn]",
}


def on_event(event: TraceEvent) -> None:
    icon = ICONS.get(event.kind, "  .")
    text = event.message
    if event.kind == "evidence" and len(text) > 300:
        text = text[:300] + "..."
    print(f"{icon} {text}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-memory", action="store_true", help="disable Hindsight recall")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--scenario",
        default=DEFAULT_SCENARIO,
        choices=sorted(SCENARIOS),
        help="which planted incident to investigate (default: %(default)s)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    if not settings.groq.configured:
        print("GROQ_API_KEY missing from .env", file=sys.stderr)
        return 2

    alert = ALERT if args.scenario == DEFAULT_SCENARIO else build_alert(
        args.scenario, settings.db_path
    )

    memory = MemoryService(settings.hindsight, settings.bank)
    if memory.available:
        memory.ensure_bank()

    print("=" * 78)
    print(f"INVESTIGATING {alert.alert_id}: {alert.metric} {alert.deviation_pct:+.1f}%")
    print(f"scenario: {args.scenario}")
    print(
        f"memory: {'on' if memory.available else 'off'}"
        f"{' (disabled for this run)' if args.no_memory else ''}"
    )
    print("=" * 78)

    try:
        report = _investigate(settings, memory, args, alert)
        # Stage 1 of the retain cycle: the agent's conclusion goes in
        # immediately, before anyone has judged it, tagged unconfirmed. Stage 2
        # (scripts/feedback.py) replaces the same document once an analyst
        # confirms or rejects it. Has to happen before close(), which drops the
        # client and makes available False.
        if memory.available and not args.no_memory:
            retained = memory.retain_report(report, report.investigation_id)
            print(f"\nretained (unconfirmed): {'yes' if retained else 'no'}")
    finally:
        # The Hindsight client holds an aiohttp session; leaking it prints
        # "Unclosed client session" noise on every run.
        memory.close()

    print("\n" + "=" * 78)
    print("REPORT")
    print("=" * 78)
    print(json.dumps(json.loads(report.model_dump_json()), indent=2)[:6000])
    print(f"\ntrace: {report.trace_id}")
    print(
        f"llm turns: {report.llm_turns} | tool calls: {report.tool_calls}"
        f" | tokens: {report.tokens_used:,} | llm errors: {report.llm_errors}"
        f" | duration: {report.duration_s}s"
    )
    print("\nGROUND TRUTH (planted in the dataset, not shown to the agent):")
    print(describe_ground_truth(args.scenario))
    return 0


def _investigate(settings, memory, args, alert: Anomaly):
    investigation = Investigation(
        anomaly=alert,
        settings=settings,
        memory=memory,
        memory_enabled=not args.no_memory,
        on_event=None if args.quiet else on_event,
    )
    try:
        report = investigation.run()
    except Exception:
        # Keep the audit trail even when the investigation dies mid-pipeline.
        print(f"\ninvestigation failed; partial trace: {investigation.write_trace()}", file=sys.stderr)
        raise

    investigation.write_trace()
    report_path = investigation.persist(report)
    print(f"\ntrace written: {investigation.trace_id}")
    print(f"report written: {report_path}")
    return report


if __name__ == "__main__":
    raise SystemExit(main())
