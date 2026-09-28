"""Lumen investigation console.

    streamlit run app/app.py

Features:
  * Alert Inbox / Scenario picker with one-click "Run Live Investigation"
  * Real-time trace event timeline during live runs
  * Evidence ChartSpec rendering (Plotly bar, line, waterfall charts)
  * Memory recall analysis and confidence score delta (with vs without memory prior)
  * Retain Stage 2 Feedback submission and reflect synthesis
"""

from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from core.config import get_settings  # noqa: E402
from core.schemas import Anomaly, Direction, InvestigationReport, Severity, Window  # noqa: E402
from data.alerts import build_alert  # noqa: E402
from data.scenarios import SCENARIOS  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"
TRACE_DIR = PROJECT_ROOT / "artifacts" / "traces"
CURVE_PATH = PROJECT_ROOT / "artifacts" / "learning_curve.json"
MEMORY_WEIGHT = 0.15

REPORT_DIR.mkdir(parents=True, exist_ok=True)
TRACE_DIR.mkdir(parents=True, exist_ok=True)


def report_paths() -> list[Path]:
    return sorted(REPORT_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)


def load_report(path: Path) -> InvestigationReport | None:
    try:
        return InvestigationReport.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def load_trace(trace_id: str) -> list[dict]:
    path = TRACE_DIR / f"{trace_id}.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def score_events(events: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for event in events:
        if event.get("kind") != "score":
            continue
        message = str(event.get("message", ""))
        hypothesis_id = str((event.get("data") or {}).get("hypothesis_id", ""))
        values: dict[str, float] = {}
        for part in message.split("(")[-1].rstrip(")").split():
            if "=" in part:
                key, _, value = part.partition("=")
                try:
                    values[key] = float(value)
                except ValueError:
                    continue
        if hypothesis_id:
            out[hypothesis_id] = values
    return out


def without_memory(total: float, values: dict) -> float:
    prior = values.get("prior", 0.0)
    return round(max(0.0, total - MEMORY_WEIGHT * prior), 3)


def render_chart_spec(chart_dict: dict) -> None:
    """Render a ChartSpec dictionary using Plotly."""
    if not chart_dict or "type" not in chart_dict:
        return

    chart_type = chart_dict.get("type")
    title = chart_dict.get("title", "")
    series = chart_dict.get("series", [])
    categories = chart_dict.get("categories", [])

    if chart_type in ("bar", "column") and series and categories:
        fig = go.Figure()
        for s in series:
            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=s.get("values", []),
                    name=s.get("name", "series"),
                )
            )
        fig.update_layout(title=title, barmode="group", height=320)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "line" and series and categories:
        fig = go.Figure()
        for s in series:
            fig.add_trace(
                go.Scatter(
                    x=categories,
                    y=s.get("values", []),
                    mode="lines+markers",
                    name=s.get("name", "series"),
                )
            )
        fig.update_layout(title=title, height=320)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "waterfall" and series and categories:
        values = series[0].get("values", []) if series else []
        fig = go.Figure(
            go.Waterfall(
                name="Decomposition",
                orientation="v",
                measure=["relative"] * len(values),
                x=categories,
                y=values,
            )
        )
        fig.update_layout(title=title, height=340)
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
def render_report(report: InvestigationReport, events: list[dict]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", report.status.replace("_", " "))
    col2.metric("Impact", f"{report.impact_pct:+.1f}%")
    col3.metric("Root causes", len(report.root_causes))
    col4.metric("Tool calls", report.tool_calls)
    st.write(report.summary)

    if not report.root_causes:
        st.info("No root cause cleared the confidence threshold.")
        return

    scores = score_events(events)
    for index, cause in enumerate(report.root_causes, start=1):
        values = scores.get(cause.hypothesis_id, {})
        base = without_memory(cause.confidence, values)
        lift = round(cause.confidence - base, 3)
        with st.container(border=True):
            st.markdown(
                f"**H{index}. {cause.cause_type}** — confidence {cause.confidence:.3f}"
            )
            st.write(cause.statement)
            left, right, middle = st.columns(3)
            left.metric("Without memory", f"{base:.3f}")
            right.metric("With memory", f"{cause.confidence:.3f}")
            middle.metric("Prior lift", f"{lift:+.3f}")
            if values:
                st.caption(
                    "signals: "
                    + ", ".join(
                        f"{k}={v:g}"
                        for k, v in sorted(values.items())
                        if k in {"support", "magnitude", "temporal", "prior"}
                    )
                )
            st.caption("evidence: " + ", ".join(cause.evidence_ids) or "none")

    if report.ruled_out:
        st.subheader("Ruled out")
        for entry in report.ruled_out:
            st.markdown(f"- **{entry.hypothesis}** — {entry.why}")

    # Evidence charts visualization
    evidence_events = [e for e in events if e.get("kind") == "evidence"]
    if evidence_events:
        st.subheader("Evidence Visualizations")
        for ev in evidence_events:
            data = ev.get("data") or {}
            chart = data.get("chart")
            if chart:
                render_chart_spec(chart)


def render_memory(report: InvestigationReport, events: list[dict]) -> None:
    st.subheader("Cases recalled this run")
    if not report.memory_used:
        st.info("Memory was unavailable for this run, so nothing was recalled.")
        return

    for entry in report.memory_used:
        verdict = "helped" if entry.helped else "recalled but unused"
        with st.expander(
            f"{entry.recall_stage} — {len(entry.memory_ids)} recalled, {verdict}",
            expanded=entry.helped,
        ):
            st.caption(f"query: {entry.query}")
            if entry.helped_reason:
                st.caption(entry.helped_reason)
            for memory_id, summary, score in zip(
                entry.memory_ids, entry.summaries, entry.scores
            ):
                st.markdown(f"- `{score:.3f}` {summary}")

    if report.similar_past_cases:
        st.subheader("Prior cases by relevance")
        st.caption(
            "Hindsight retrieval score, not a probability: comparable within one "
            "run, not across runs."
        )
        for case in report.similar_past_cases:
            badge = "confirmed" if case.confirmed else "unconfirmed"
            st.markdown(
                f"- `{case.similarity:.3f}` **{badge}** — {case.outcome}"
            )

    attached = [
        event for event in events if event.get("kind") == "memory_attached"
    ]
    if attached:
        st.subheader("Priors attached to hypotheses")
        for event in attached:
            data = event.get("data") or {}
            st.markdown(
                f"- **{data.get('hypothesis_id')}** ({data.get('cause_type')}) "
                f"at relevance {data.get('relevance')} — "
                f"{len(data.get('memory_ids') or [])} case(s)"
            )
        st.caption(
            "A prior only counts as help if the hypothesis carrying it survived "
            "validation on current evidence; memory never overrides a tool result."
        )

    st.subheader("Timeline")
    if not events:
        st.caption("no trace on disk for this run")
        return
    for event in events:
        kind = event.get("kind", "")
        message = str(event.get("message", ""))
        icon = {
            "evidence": "📊",
            "hypothesis": "💡",
            "memory_recall": "🧠",
            "memory_attached": "🧠",
            "score": "🎯",
            "state": "🔄",
            "done": "✅",
        }.get(kind, "•")
        st.markdown(f"`{icon}` {message}")


def render_history() -> None:
    st.subheader("Backfilled history")
    st.caption(
        "Closed cases seeded into the bank. Every one is from a different "
        "incident, so a correct recall is a lead rather than the answer."
    )
    from scripts.seed_bank import SEED_CASES  # noqa: PLC0415

    for case in SEED_CASES:
        with st.expander(f"{case['occurred_at'][:10]} — {case['document_id']}"):
            st.write(case["content"])
            st.caption("tags: " + ", ".join(case.get("tags", [])))


def render_learning_curve() -> None:
    st.subheader("Learning curve")
    if not CURVE_PATH.exists():
        st.info(
            "No curve on disk. Run `python -m eval.learning_curve` — it seeds a "
            "bank per size and records the result here."
        )
        return

    data = json.loads(CURVE_PATH.read_text(encoding="utf-8"))
    scenarios = data["scenarios"]
    rows = data["results"]
    if not rows:
        st.info("curve file is empty")
        return

    by_scenario: dict[str, dict[int, dict]] = {sid: {} for sid in scenarios}
    for row in rows:
        if "error" not in row:
            by_scenario.setdefault(row["scenario"], {})[row["size"]] = row

    st.caption(
        "Accuracy and confidence as the bank fills. History accumulates in date "
        "order, because that is the only order it ever arrives in."
    )

    chart = {
        sid: {str(size): values["confidence"] for size, values in sorted(sizes.items())}
        for sid, sizes in by_scenario.items()
        if sizes
    }
    st.line_chart(chart, height=280)

    header = "| size | " + " | ".join(scenarios) + " |"
    divider = "|" + "---|" * (len(scenarios) + 1)
    lines = [header, divider]
    for size in data["sizes"]:
        cells = []
        for sid in scenarios:
            row = by_scenario.get(sid, {}).get(size)
            if row is None:
                cells.append("—")
            else:
                mark = "ok" if row.get("correct") else "X"
                cells.append(f"{mark} {row['confidence']:.2f} pri={row['priors']}")
        lines.append(f"| {size} | " + " | ".join(cells) + " |")
    st.markdown("\n".join(lines))


def render_feedback(report: InvestigationReport) -> None:
    st.subheader("Confirm or reject this conclusion")
    st.caption(
        "Stage 2 of the retain cycle. Your verdict replaces the agent's "
        "unconfirmed guess under the same document id, so recall never serves "
        "both the guess and its correction."
    )
    if not report.root_causes:
        st.info("Nothing settled to confirm.")
        return

    top = report.root_causes[0]
    st.markdown(f"**Agent's leading hypothesis:** {top.statement}")
    st.caption(f"cause type {top.cause_type}, confidence {top.confidence:.3f}")

    if not report.investigation_id:
        st.error(
            "This report has no investigation id, so feedback cannot be filed "
            "against it. Re-run the investigation."
        )
        return

    with st.form("feedback"):
        verdict = st.selectbox(
            "Was the agent right?",
            ["confirmed", "partially_correct", "wrong"],
        )
        cause = st.text_input(
            "The real root cause, in your words",
            value=top.statement if verdict != "wrong" else "",
        )
        action = st.text_input("Action taken")
        owner = st.text_input("Owner")
        outcome = st.text_input("Outcome")
        notes = st.text_area("Notes")
        do_reflect = st.checkbox(
            "Ask reflect() what this confirmation means for the next similar case",
            value=True,
        )
        submitted = st.form_submit_button("File feedback")

    if not submitted:
        return

    from core.config import get_settings  # noqa: PLC0415
    from core.schemas import Feedback  # noqa: PLC0415
    from memory.memory_service import MemoryService  # noqa: PLC0415

    settings = get_settings()
    memory = MemoryService(settings.hindsight, settings.bank)
    if not memory.available:
        st.error(f"Hindsight unavailable: {memory._unavailable_reason}")
        return

    try:
        if not memory.ensure_bank():
            st.error("Could not use the memory bank.")
            return
        feedback = Feedback(
            investigation_id=report.investigation_id,
            verdict=verdict,  # type: ignore[arg-type]
            confirmed_cause=cause,
            action_taken=action,
            owner=owner,
            outcome=outcome,
            notes=notes,
        )
        ok = memory.retain_feedback(report, feedback, report.investigation_id)
        if not ok:
            st.error("Retain failed; check the Hindsight connection.")
            return
        st.success(
            f"Filed. Stage moved agent_report → human_feedback "
            f"(verdict={verdict})."
        )

        if do_reflect:
            with st.spinner("reflecting..."):
                result = memory.reflect(
                    query=(
                        f"After a human confirmed this incident as '{verdict}', "
                        f"what should the next investigator know before opening a "
                        f"similar case on {report.metric}? Agent said: "
                        f"{top.statement}. Confirmed cause: {cause or verdict}. "
                        "What transfers across incidents and what was specific "
                        "to this one?"
                    ),
                    budget="mid",
                )
            if result.get("error"):
                st.warning(f"reflect skipped: {result['error']}")
            else:
                out = REPORT_DIR / f"{report.trace_id}.reflection.txt"
                out.write_text(str(result.get("text") or ""), encoding="utf-8")
                st.subheader("Reflection")
                st.markdown(str(result.get("text") or ""))
                st.caption(
                    f"{len(result.get('based_on') or [])} memory unit(s) cited; "
                    f"written to {out.name}. Not retained back into the bank -- "
                    "that would hand recall a second copy of the same case."
                )
    finally:
        memory.close()


def run_live_investigation(scenario_id: str, enable_memory: bool) -> None:
    st.subheader(f"⚡ Live Investigation: {scenario_id}")

    import tools.investigation  # noqa: F401,E402
    from agent.orchestrator import Investigation, TraceEvent  # noqa: E402
    from memory.memory_service import MemoryService  # noqa: E402

    settings = get_settings()
    if not settings.groq.configured:
        st.error("GROQ_API_KEY is missing from .env. Cannot run LLM investigation.")
        return

    # Build DB if needed
    alert = build_alert(scenario_id, settings.db_path)
    memory = MemoryService(settings.hindsight, settings.bank)
    if memory.available and enable_memory:
        memory.ensure_bank()

    timeline_placeholder = st.empty()
    live_events: list[dict] = []

    def on_event(event: TraceEvent) -> None:
        live_events.append({"kind": event.kind, "message": event.message, "data": event.data})
        with timeline_placeholder.container():
            st.write(f"**Step:** {event.kind} - {event.message[:200]}")

    investigation = Investigation(
        anomaly=alert,
        settings=settings,
        memory=memory,
        memory_enabled=enable_memory and memory.available,
        on_event=on_event,
    )

    with st.spinner("Investigating scenario..."):
        try:
            report = investigation.run()
            investigation.write_trace()
            report_path = investigation.persist(report)
            if memory.available and enable_memory:
                memory.retain_report(report, report.investigation_id)
            st.success(f"Investigation complete! Written to {report_path.name}")
            st.rerun()
        except Exception as err:
            st.error(f"Investigation failed: {err}")
        finally:
            memory.close()


# --------------------------------------------------------------------------- #
# shell
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Lumen", page_icon="🔎", layout="wide")
    st.title("Lumen — Alert-Triggered Root Cause Investigator")

    paths = report_paths()

    with st.sidebar:
        st.header("🚨 Alert Inbox & Scenario Trigger")
        selected_scenario = st.selectbox(
            "Select Scenario Alert",
            options=sorted(SCENARIOS.keys()),
            format_func=lambda s: f"{s}: {SCENARIOS[s].title}",
        )
        scenario_meta = SCENARIOS[selected_scenario]
        st.info(f"**Title:** {scenario_meta.title}\n\n**Cause Type:** {scenario_meta.truth.cause_type}")

        use_memory = st.checkbox("Enable Hindsight Memory", value=True)
        if st.button("🚀 Start Live Investigation", type="primary"):
            run_live_investigation(selected_scenario, use_memory)

        st.divider()
        st.subheader("Past Investigation Reports")
        labels = []
        for path in paths:
            report = load_report(path)
            if report is None:
                continue
            labels.append(
                (
                    f"{report.investigation_id or report.trace_id} · "
                    f"{report.status.replace('_', ' ')}",
                    path,
                    report,
                )
            )

        if labels:
            choice = st.selectbox(
                "Select Report",
                options=range(len(labels)),
                format_func=lambda i: labels[i][0],
            )
            _, path, report = labels[choice]
            st.caption(path.name)
            memory_on = bool(report.memory_used)
            st.caption("memory: on" if memory_on else "memory: off / unavailable")
        else:
            report = None

    if report is None:
        st.warning("No completed reports found. Use the sidebar to start an investigation!")
        return

    tabs = st.tabs(
        ["Report", "Memory & Trace", "History", "Learning curve", "Feedback"]
    )
    events = load_trace(report.trace_id)
    with tabs[0]:
        render_report(report, events)
    with tabs[1]:
        render_memory(report, events)
    with tabs[2]:
        render_history()
    with tabs[3]:
        render_learning_curve()
    with tabs[4]:
        render_feedback(report)


if __name__ == "__main__":
    main()
