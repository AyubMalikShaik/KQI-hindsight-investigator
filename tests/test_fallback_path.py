"""Force the hypothesizer to fail and check the evidence-driven fallback.

Runs the real tools, then builds hypotheses with no LLM involvement, so the
degraded path is verified rather than assumed.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import tools.investigation  # noqa: F401,E402

from agent.orchestrator import Investigation  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.schemas import Anomaly, Direction, Severity, Window  # noqa: E402
from memory.memory_service import MemoryService  # noqa: E402

ALERT = Anomaly(
    alert_id="ALR-FALLBACK-TEST",
    metric="daily_revenue",
    grain="day",
    detected_at=datetime(2026, 9, 27, tzinfo=timezone.utc),
    anomaly_window=Window(start="2026-09-26", end="2026-09-26"),
    baseline_window=Window(start="2026-08-29", end="2026-09-25"),
    observed=1_547_169.0,
    expected=1_856_660.0,
    deviation_pct=-16.7,
    direction=Direction.drop,
    severity=Severity.high,
)


def main() -> int:
    settings = get_settings()
    events: list[str] = []
    investigation = Investigation(
        anomaly=ALERT,
        settings=settings,
        memory=MemoryService(settings.hindsight, settings.bank),
        memory_enabled=False,
        on_event=lambda event: events.append(f"[{event.kind}] {event.message}"),
    )

    investigation._scoping()
    investigation._evidence_gathering()
    print(f"gathered {len(investigation.evidence)} evidence items\n")

    investigation._evidence_driven_hypotheses()
    if not investigation.hypotheses:
        print("FAIL: fallback produced no hypotheses")
        return 1

    print(f"fallback produced {len(investigation.hypotheses)} hypothesis(es):\n")
    for hypothesis in investigation.hypotheses:
        print(f"{hypothesis.hypothesis_id}  cause_type={hypothesis.cause_type}")
        print(f"  statement: {hypothesis.statement}")
        print(f"  affected_segments: {hypothesis.affected_segments}")
        print(f"  predicted_evidence: {hypothesis.predicted_evidence}")
        print()

    changepoint = investigation._changepoint()
    event_times = investigation._candidate_events()
    print("-" * 78)
    scored = investigation._score(changepoint, event_times)
    for hypothesis, confidence in scored:
        print(f"{hypothesis.hypothesis_id} confidence {confidence.total:.2f}")
        for note in confidence.notes:
            print(f"    {note}")

    print("\nsummary fallback:\n ", investigation._fallback_summary())

    # Now the full degraded path: no validator verdicts and no reporter output,
    # which is what a rate-limited Groq run looks like. The report must not
    # contradict itself in either direction.
    print("\n" + "=" * 78)
    print("degraded path: no verdicts, no reporter output")
    print("=" * 78)
    problems: list[str] = []

    top = max(scored, key=lambda pair: pair[1].total)
    investigation._settle_without_verdict(top)
    report = investigation._assemble_report(None, scored)

    print(f"status:       {report.status}")
    print(f"root_causes:  {len(report.root_causes)}")
    for cause in report.root_causes:
        print(f"  {cause.hypothesis_id} conf={cause.confidence} {cause.statement[:70]}")
    print(f"summary:      {report.summary[:150]}")

    if report.status == "partial" and report.root_causes:
        problems.append("status is partial while root causes are listed")
    if report.status == "cannot_determine" and report.root_causes:
        problems.append("status is cannot_determine while root causes are listed")
    if "no hypothesis reached" in report.summary.lower() and report.root_causes:
        problems.append("summary denies a root cause while listing one")
    if not report.root_causes:
        problems.append("a hypothesis above the threshold produced no root cause")
    for cause in report.root_causes:
        if cause.confidence < 0.75:
            problems.append(
                f"{cause.hypothesis_id} reported at {cause.confidence}, below threshold"
            )
        if not cause.evidence_ids:
            problems.append(f"{cause.hypothesis_id} has no evidence ids")

    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nDEGRADED PATH OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
