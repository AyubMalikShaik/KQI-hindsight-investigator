"""Static checks that keep the agent's prompt/tool plumbing honest.

These are the failures that only show up as a KeyError or a silently discarded
tool call in the middle of an expensive LLM run, so they are worth catching
before spending a single token.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import tools.investigation  # noqa: F401,E402

from agent import prompts  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.schemas import (  # noqa: E402
    Anomaly,
    Direction,
    HypothesisVerdict,
    InvestigationReport,
    RootCause,
    RuledOut,
    Severity,
    SimilarPastCase,
    Window,
)
from tools.registry import registry  # noqa: E402
from validation.confidence import validate_citations  # noqa: E402

PROMPT_NAMES = [
    "SYSTEM",
    "PLANNER",
    "INVESTIGATOR",
    "HYPOTHESIZER",
    "VALIDATOR",
    "REPORTER",
]

PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z_0-9]*)\}")


def placeholders(template: str) -> set[str]:
    scrubbed = template.replace("{{", "\x00").replace("}}", "\x00")
    return set(PLACEHOLDER.findall(scrubbed))


def sample_alert() -> Anomaly:
    return Anomaly(
        alert_id="ALR-TEST-1",
        metric="daily_revenue",
        grain="day",
        detected_at=datetime(2026, 9, 27, tzinfo=timezone.utc),
        anomaly_window=Window(start="2026-09-26", end="2026-09-26"),
        baseline_window=Window(start="2026-08-29", end="2026-09-25"),
        observed=1_000.0,
        expected=1_200.0,
        deviation_pct=-16.7,
        direction=Direction.drop,
        severity=Severity.high,
    )


def check_prompts() -> list[str]:
    failures: list[str] = []
    alert = sample_alert()
    values = {
        "alert": prompts.format_alert(alert),
        "tool_names": ", ".join(registry.names()),
        "tool_schemas": prompts.format_tool_schemas(registry),
        "evidence_block": "E1: test",
        "hypotheses": "H1: test",
        "verdicts": "H1: supported",
        "memory_block": "no memory",
        "scope_evidence": "none",
    }
    for name in PROMPT_NAMES:
        template = getattr(prompts, name, None)
        if template is None:
            failures.append(f"{name}: prompt does not exist")
            continue
        needed = placeholders(template)
        missing = needed - set(values)
        if missing:
            failures.append(f"{name}: unsupplied placeholders {sorted(missing)}")
            continue
        try:
            template.format(**{key: values[key] for key in needed})
        except (KeyError, IndexError, ValueError) as exc:
            failures.append(f"{name}: format() raised {exc!r}")
    return failures


def check_tools_registered() -> list[str]:
    expected = {
        "analyze_metric",
        "breakdown_by_dimension",
        "find_related_metrics",
        "query_business_events",
    }
    missing = expected - set(registry.names())
    return [f"tools never registered: {sorted(missing)}"] if missing else []


def check_tool_schemas_rendered() -> list[str]:
    rendered = prompts.format_tool_schemas(registry)
    problems: list[str] = []
    if "no tools registered" in rendered:
        problems.append("format_tool_schemas produced no output")
    for name in registry.names():
        if f"- {name}(" not in rendered:
            problems.append(f"{name} missing from the rendered schema block")
    return problems


def check_dimension_filter_binds_correctly() -> list[str]:
    """Regression: the date params used to be prepended after the dimension
    placeholder, so country=IN was compared against the date range."""
    from tools import sql

    problems: list[str] = []
    filtered = sql.totals_by_day(
        "daily_revenue", "2026-09-26", "2026-09-26", dimension="country", dimension_value="IN"
    )
    total = sql.totals_by_day("daily_revenue", "2026-09-26", "2026-09-26")
    if filtered.empty:
        problems.append("dimension filter returned no rows (parameter order regression)")
    elif not total.empty:
        filtered_value = float(filtered["value"].iloc[0])
        total_value = float(total["value"].iloc[0])
        if filtered_value >= total_value:
            problems.append(
                f"country=IN revenue {filtered_value} should be below global {total_value}"
            )
    return problems


def check_citation_validation_catches_fabrication() -> list[str]:
    report = InvestigationReport(
        trace_id="t",
        alert_id="a",
        metric="daily_revenue",
        status="root_cause_found",
        summary="s",
        root_causes=[
            RootCause(
                hypothesis_id="H1",
                statement="made up",
                cause_type="release_regression",
                confidence=0.9,
                evidence_ids=["EV-DOES-NOT-EXIST"],
            )
        ],
        ruled_out=[RuledOut(hypothesis="x", why="y", evidence_ids=["EV-ALSO-FAKE"])],
    )
    problems = validate_citations(report, {"EV-REAL"})
    if not any("unknown ids" in p for p in problems):
        return ["validate_citations did not flag a hallucinated evidence id"]
    if report.uncited_claims() != []:
        return ["uncited_claims should be empty when ids exist, however fake"]
    return []


def check_uncited_claims_returns_bare_ids() -> list[str]:
    report = InvestigationReport(
        trace_id="t",
        alert_id="a",
        metric="daily_revenue",
        status="partial",
        summary="s",
        root_causes=[
            RootCause(
                hypothesis_id="H7",
                statement="no evidence attached",
                cause_type="unknown",
                confidence=0.1,
                evidence_ids=[],
            )
        ],
    )
    uncited = report.uncited_claims()
    if uncited != ["H7"]:
        return [f"uncited_claims() returned {uncited!r}, expected ['H7'] (bare ids)"]
    return []


def check_verdict_evidence_ids_usable() -> list[str]:
    verdict = HypothesisVerdict(hypothesis_id="H1", verdict="supported", reason="r")
    if not hasattr(verdict, "evidence_ids"):
        return ["HypothesisVerdict is missing evidence_ids"]
    if hasattr(verdict, "evidence_id"):
        return ["HypothesisVerdict still has the singular evidence_id attribute"]
    return []


def check_evidence_ids_are_stable_across_processes() -> list[str]:
    """A trace written by one run must be resolvable by the next one."""
    import subprocess
    import sys as _sys

    from tools.registry import stable_evidence_id

    first = stable_evidence_id("MET", tool="analyze_metric", metric="daily_revenue")
    probe = (
        "import sys; sys.path.insert(0, '.');"
        "from tools.registry import stable_evidence_id;"
        "print(stable_evidence_id('MET', tool='analyze_metric', metric='daily_revenue'))"
    )
    completed = subprocess.run(
        [_sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=".",
    )
    second = completed.stdout.strip()
    if completed.returncode != 0:
        return [f"subprocess probe failed: {completed.stderr.strip()[:200]}"]
    if first != second:
        return [
            f"evidence id is not reproducible across processes: {first!r} vs {second!r} "
            "(built-in hash() is salted per interpreter)"
        ]
    return []


def check_evidence_ids_vary_with_arguments() -> list[str]:
    """A different dimension must not collide with the previous evidence id."""
    from tools.registry import stable_evidence_id

    problems: list[str] = []
    by_country = stable_evidence_id("BRK", tool="breakdown_by_dimension", dimension="country")
    by_payment = stable_evidence_id("BRK", tool="breakdown_by_dimension", dimension="payment_method")
    by_window = stable_evidence_id(
        "BRK", tool="breakdown_by_dimension", dimension="country", start="2026-09-01"
    )
    if by_country == by_payment:
        problems.append("different dimensions produced the same evidence id")
    if by_country == by_window:
        problems.append("different windows produced the same evidence id")
    return problems


def check_out_of_window_queries_are_snapped() -> list[str]:
    """Regression: the validator killed the correct cause by querying a quiet
    week, so follow-up windows must be anchored to the anomaly window."""
    import tools.investigation  # noqa: F401
    from agent.orchestrator import Investigation
    from core.config import get_settings
    from memory.memory_service import MemoryService

    settings = get_settings()
    investigation = Investigation(
        anomaly=sample_alert(),
        settings=settings,
        memory=MemoryService(settings.hindsight, settings.bank),
        memory_enabled=False,
        on_event=lambda event: None,
    )
    problems: list[str] = []

    overlapping = {"start": "2026-09-20", "end": "2026-09-30"}
    if investigation._constrain_window("analyze_metric", overlapping, "validation") != overlapping:
        problems.append("a window that overlaps the anomaly was modified")

    outside = {"start": "2026-06-01", "end": "2026-06-07"}
    adjusted = investigation._constrain_window("analyze_metric", outside, "validation")
    if adjusted["start"] != "2026-09-26" or adjusted["end"] != "2026-09-26":
        problems.append(
            f"a non-overlapping window was not snapped to the anomaly window: {adjusted}"
        )
    if outside["start"] != "2026-06-01":
        problems.append("_constrain_window mutated the caller's dict")
    return problems


def check_magnitude_scoring_handles_declining_shares() -> list[str]:
    """Regression: a declining segment reports a negative share, and comparing
    the raw sign made max() always pick 0.0, so every declining metric scored
    the 0.4 'segment not isolated' fallback regardless of how much of the
    decline it explained."""
    from core.schemas import Evidence, Hypothesis
    from validation.confidence import magnitude_explained

    evidence = [
        Evidence(
            evidence_id="ev_decline",
            tool="breakdown_by_dimension",
            finding="upi explains the decline",
            args={"metric": "payment_success_rate", "dimension": "payment_method"},
            numbers={
                "segments": [
                    {"segment": "upi", "share_of_decline_pct": -99.2},
                    {"segment": "card", "share_of_decline_pct": -0.8},
                ]
            },
        )
    ]
    problems: list[str] = []

    hypothesis = Hypothesis(
        hypothesis_id="H1",
        statement="UPI payments failed.",
        cause_type="payment_failure",
        affected_segments={"payment_method": "upi"},
    )
    score, note = magnitude_explained(hypothesis, evidence, report_total_change=1000.0)
    if "not isolated" in note:
        problems.append(f"a -99.2% share was treated as unquantified: {note!r}")
    if score < 0.9:
        problems.append(f"a 99.2% share scored {score:.2f}, expected >= 0.90")

    unmatched = Hypothesis(
        hypothesis_id="H2",
        statement="Something else failed.",
        cause_type="payment_failure",
        affected_segments={"payment_method": "mystery"},
    )
    unmatched_score, unmatched_note = magnitude_explained(
        unmatched, evidence, report_total_change=1000.0
    )
    if "not found in the gathered breakdowns" not in unmatched_note:
        problems.append(
            f"an unmatched segment should say so explicitly, got {unmatched_note!r}"
        )
    if unmatched_score > 0.5:
        problems.append(f"an unmatched segment scored {unmatched_score:.2f}, expected <= 0.50")
    return problems


def check_impact_figures_are_mutually_consistent() -> list[str]:
    """Regression: impact_abs came from a breakdown's gross_decline, which nets
    out segments that grew, so it disagreed with impact_pct. A $306,067.51
    figure sat next to a -16.7% drop that implies $310,062."""
    from agent.orchestrator import Investigation
    from core.config import get_settings
    from core.schemas import Evidence
    from memory.memory_service import MemoryService

    settings = get_settings()
    anomaly = sample_alert()
    investigation = Investigation(
        anomaly=anomaly,
        settings=settings,
        memory=MemoryService(settings.hindsight, settings.bank),
        memory_enabled=False,
        on_event=lambda event: None,
    )
    # A breakdown whose negative segments sum to less than the total shortfall,
    # because one segment grew and offset part of the drop.
    investigation.evidence.append(
        Evidence(
            evidence_id="EV-BRK-mixed",
            tool="breakdown_by_dimension",
            finding="segments partially offset",
            args={"metric": "daily_revenue", "dimension": "country"},
            numbers={"gross_decline": 306_067.51, "total_change": -309_490.78},
        )
    )

    problems: list[str] = []
    impact_abs, impact_pct = investigation._impact()
    expected = abs(anomaly.expected - anomaly.observed)
    if abs(impact_abs - expected) > 0.01:
        problems.append(
            f"impact_abs {impact_abs} should be the alert shortfall {round(expected, 2)}"
        )

    implied_pct = impact_abs / anomaly.expected * 100
    if abs(implied_pct - abs(impact_pct)) > 0.1:
        problems.append(
            f"impact_abs {impact_abs} implies {implied_pct:.2f}% but impact_pct is "
            f"{impact_pct}%"
        )

    note = investigation._impact_note(impact_abs, impact_pct)
    if f"{impact_abs:,.0f}" not in note:
        problems.append(f"impact note does not quote the computed figure: {note!r}")
    if f"{impact_pct:+.1f}" not in note:
        problems.append(f"impact note does not quote the computed percentage: {note!r}")
    return problems


def check_segment_matching_survives_model_wording() -> list[str]:
    """Regression: hypotheses state "UPI"/"India" while the warehouse stores
    "upi"/"IN", and matching was verbatim, so a correct hypothesis silently
    lost its magnitude score and could drop below the confidence threshold."""
    from core.schemas import Evidence, Hypothesis
    from validation.confidence import magnitude_explained

    evidence = [
        Evidence(
            evidence_id="EV-BRK-seg",
            tool="breakdown_by_dimension",
            finding="country and payment_method breakdowns",
            args={"metric": "daily_revenue", "dimension": "payment_method"},
            numbers={
                "dimension": "payment_method",
                "top": {"segment": "upi", "share_of_decline_pct": -100.0},
                "segments": [
                    {"segment": "upi", "share_of_decline_pct": -100.0},
                    {"segment": "card", "share_of_decline_pct": 0.0},
                ],
            },
        ),
        Evidence(
            evidence_id="EV-BRK-country",
            tool="breakdown_by_dimension",
            finding="country breakdown",
            args={"metric": "daily_revenue", "dimension": "country"},
            numbers={
                "dimension": "country",
                "segments": [{"segment": "IN", "share_of_decline_pct": -100.0}],
            },
        ),
    ]
    problems: list[str] = []

    for claimed, description in (
        ({"payment_method": "upi"}, "lowercase canonical value"),
        ({"payment_method": "UPI"}, "uppercase value as the model writes it"),
        ({"country": "IN"}, "country code"),
        ({"country": "India"}, "country name rather than code"),
    ):
        hypothesis = Hypothesis(
            hypothesis_id="H1",
            statement="UPI payments failed in India.",
            cause_type="payment_failure",
            affected_segments=claimed,
        )
        score, note = magnitude_explained(hypothesis, evidence, report_total_change=1000.0)
        if score < 0.9:
            problems.append(
                f"{description} ({claimed}) scored {score:.2f}: {note!r}"
            )

    # With no explicit segments the scorer falls back to the dominant segment
    # and asks whether the prose refers to it. Short codes must not match
    # inside an unrelated word.
    for statement, should_match in (
        ("UPI authorisation timeouts hit India.", True),
        ("Conversion increased after the release.", False),
        ("A short incremental rollout began.", False),
    ):
        hypothesis = Hypothesis(
            hypothesis_id="H2",
            statement=statement,
            cause_type="payment_failure",
            affected_segments={},
        )
        score, note = magnitude_explained(hypothesis, evidence, report_total_change=1000.0)
        matched = score > 0.5
        if matched != should_match:
            problems.append(
                f"prose {statement!r} -> dominant segment matched={matched} "
                f"(score {score:.2f}, {note!r}), expected {should_match}"
            )

    wrong = Hypothesis(
        hypothesis_id="H3",
        statement="Card payments failed.",
        cause_type="payment_failure",
        affected_segments={"payment_method": "card"},
    )
    wrong_score, wrong_note = magnitude_explained(wrong, evidence, report_total_change=1000.0)
    if wrong_score > 0.5:
        problems.append(
            f"a 0%-share segment was credited with magnitude: {wrong_score:.2f} ({wrong_note!r})"
        )
    return problems


def check_retain_tags_the_alert_it_belongs_to() -> list[str]:
    """Regression: reports were retained without an alert tag, so recall could not
    tell an investigation's own earlier report from someone else's history.

    That made an investigation read back its own conclusion as a prior and then
    confirm itself with it -- the exact circularity the seeded bank exists to
    avoid. Both retain stages must carry the tag, and recall must be able to
    drop anything carrying the alert under investigation while leaving every
    other memory in place.
    """
    from core.schemas import Feedback, InvestigationReport
    from memory.memory_service import MemoryService

    problems: list[str] = []
    report = InvestigationReport(
        trace_id="trace-retain",
        investigation_id="INV-RETAIN-1",
        alert_id="ALR-2026-0142",
        metric="daily_revenue",
        status="root_cause_found",
        summary="UPI keepalive regression",
    )

    class _FakeClient:
        def __init__(self) -> None:
            self.retains: list[dict] = []
            self.memories: list[dict] = []

        def retain(self, **kwargs):  # noqa: ANN003
            self.retains.append(kwargs)
            self.memories.append(
                {
                    "id": f"mem-{len(self.retains)}",
                    "tags": kwargs.get("tags") or [],
                    "content": kwargs.get("content") or "",
                }
            )
            return True

        def list_memories(self, **kwargs):  # noqa: ANN003
            class _Response:
                pass

            response = _Response()
            items = []
            for memory in self.memories:
                item = type("Item", (), {})()
                item.id = memory["id"]
                item.tags = memory["tags"]
                items.append(item)
            response.memories = items
            return response

        def close(self) -> None:
            pass

    memory = MemoryService.__new__(MemoryService)
    memory.bank = "test-bank"
    memory._unavailable_reason = ""
    memory.config = type("C", (), {"configured": True})()
    memory._client = _FakeClient()

    feedback = Feedback(investigation_id="INV-RETAIN-1", verdict="confirmed", confirmed_cause="keepalive")
    if not memory.retain_report(report, "INV-RETAIN-1"):
        problems.append("retain_report returned False against a working client")
    if not memory.retain_feedback(report, feedback, "INV-RETAIN-1"):
        problems.append("retain_feedback returned False against a working client")

    for kwargs in memory._client.retains:
        tags = kwargs.get("tags") or []
        if "alert:ALR-2026-0142" not in tags:
            problems.append(f"retained without an alert tag: {tags}")
        if not any(str(tag).startswith("stage:") for tag in tags):
            # _retain appends the stage tag itself; if it is missing the
            # confirmed/unconfirmed distinction is invisible to the report.
            problems.append(f"retained without a stage tag: {tags}")

    # Both stages must address one document, or recall sees the guess and the
    # correction side by side.
    doc_ids = {kwargs.get("document_id") for kwargs in memory._client.retains}
    if len(doc_ids) != 1:
        problems.append(f"the two retain stages used {len(doc_ids)} document ids: {doc_ids}")

    # Stage 2 must supersede stage 1 rather than sit alongside it.
    if memory._client.retains[-1].get("update_mode") != "replace":
        problems.append("stage 2 did not use update_mode=replace, so both versions survive")

    # Now prove recall can separate them.
    from memory.memory_service import RecallResult

    class _RecallClient(_FakeClient):
        def recall(self, **kwargs):  # noqa: ANN003
            class _Item:
                def __init__(self, index: int) -> None:
                    self.id = f"mem-{index}"
                    self.text = self.id
                    self.scores = {"final": 0.9}
                    self.type = ""
                    self.context = ""
                    self.occurred_start = ""
                    self.document_id = ""

            class _Response:
                results = [_Item(1), _Item(2)]

            return _Response()

    memory._client = _RecallClient()
    memory._client.memories = [
        {"id": "mem-1", "tags": ["alert:ALR-2026-0142", "stage:human_feedback"]},
        {"id": "mem-2", "tags": ["stage:human_feedback"]},
    ]

    kept = memory.recall("query", "after_scoping", exclude_alert="ALR-2026-0142")
    if isinstance(kept, RecallResult) and kept.error:
        problems.append(f"recall errored: {kept.error}")
    elif [m.memory_id for m in kept.memories] != ["mem-2"]:
        problems.append(
            "exclude_alert did not drop only the alert under investigation: "
            f"kept {[m.memory_id for m in kept.memories]}"
        )

    unfiltered = memory.recall("query", "after_scoping")
    if [m.memory_id for m in unfiltered.memories] != ["mem-1", "mem-2"]:
        problems.append("recall without exclude_alert changed what comes back")

    # Stage lookup must still work off the same tag read.
    stages = memory.stages_by_memory_id(["mem-1", "mem-2"])
    if stages.get("mem-1") != "human_feedback":
        problems.append(f"stage tag not read back: {stages}")

    memory._client = None
    return problems


def main() -> int:
    checks = [
        check_prompts,
        check_tools_registered,
        check_tool_schemas_rendered,
        check_dimension_filter_binds_correctly,
        check_citation_validation_catches_fabrication,
        check_uncited_claims_returns_bare_ids,
        check_verdict_evidence_ids_usable,
        check_evidence_ids_are_stable_across_processes,
        check_evidence_ids_vary_with_arguments,
        check_out_of_window_queries_are_snapped,
        check_magnitude_scoring_handles_declining_shares,
        check_impact_figures_are_mutually_consistent,
        check_segment_matching_survives_model_wording,
        check_retain_tags_the_alert_it_belongs_to,
    ]
    failures: list[str] = []
    for check in checks:
        try:
            problems = check()
        except Exception as exc:  # noqa: BLE001
            problems = [f"{check.__name__} raised {type(exc).__name__}: {exc}"]
        status = "ok" if not problems else "FAIL"
        print(f"  {check.__name__:46} {status}")
        for problem in problems:
            print(f"      - {problem}")
        failures.extend(problems)

    print()
    if failures:
        print(f"{len(failures)} PROBLEM(S)")
        return 1
    print("ALL PLUMBING CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
