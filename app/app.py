"""Lumen investigation console.

    streamlit run app/app.py
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
    """Render a ChartSpec dictionary or chart structure using Plotly."""
    if not chart_dict or not isinstance(chart_dict, dict):
        return

    kind = chart_dict.get("kind") or chart_dict.get("type")
    title = chart_dict.get("title", "Evidence Chart")
    series = chart_dict.get("series", {})
    categories = chart_dict.get("categories", [])

    if isinstance(series, dict) and series:
        fig = go.Figure()
        for name, vals in series.items():
            if not vals:
                continue
            x_vals = categories if categories and len(categories) == len(vals) else list(range(len(vals)))
            if kind in ("bar", "column"):
                fig.add_trace(go.Bar(x=x_vals, y=vals, name=str(name)))
            else:
                fig.add_trace(go.Scatter(x=x_vals, y=vals, mode="lines+markers", name=str(name)))
        fig.update_layout(title=title, margin=dict(l=20, r=20, t=40, b=20), height=300)
        st.plotly_chart(fig, use_container_width=True)

    elif isinstance(series, list) and series and categories:
        fig = go.Figure()
        for s in series:
            vals = s.get("values", []) if isinstance(s, dict) else []
            s_name = s.get("name", "series") if isinstance(s, dict) else "series"
            if kind == "waterfall":
                fig.add_trace(
                    go.Waterfall(
                        name=s_name,
                        orientation="v",
                        measure=["relative"] * len(vals),
                        x=categories,
                        y=vals,
                    )
                )
            elif kind in ("bar", "column"):
                fig.add_trace(go.Bar(x=categories, y=vals, name=s_name))
            else:
                fig.add_trace(go.Scatter(x=categories, y=vals, mode="lines+markers", name=s_name))
        fig.update_layout(title=title, margin=dict(l=20, r=20, t=40, b=20), height=300)
        st.plotly_chart(fig, use_container_width=True)


def render_evidence_visualizations(events: list[dict], report: InvestigationReport) -> None:
    st.subheader("📊 Evidence Visualizations")
    rendered_any = False

    for ev in events:
        if ev.get("kind") == "evidence":
            data = ev.get("data") or {}
            chart = data.get("chart")
            finding = data.get("finding") or ev.get("message", "")
            if chart and (chart.get("series") or chart.get("categories")):
                st.markdown(f"**{data.get('tool', 'Evidence Tool')}**: {finding}")
                render_chart_spec(chart)
                rendered_any = True

    if not rendered_any:
        st.info("No numerical evidence charts generated for this run.")


def render_memory_comparison_banner(report: InvestigationReport, events: list[dict]) -> None:
    st.subheader("⚖️ Memory Impact Comparison (With vs Without Memory)")

    scores = score_events(events)
    if not report.root_causes:
        st.info("No root cause met the confidence threshold for memory comparison.")
        return

    top_cause = report.root_causes[0]
    values = scores.get(top_cause.hypothesis_id, {})
    base_conf = without_memory(top_cause.confidence, values)
    with_conf = top_cause.confidence
    lift = round(with_conf - base_conf, 3)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("With Memory Confidence", f"{with_conf:.3f}")
    col2.metric("Without Memory Confidence", f"{base_conf:.3f}")
    col3.metric("Memory Lift Delta", f"{lift:+.3f}")
    col4.metric("Recalled Memory Cases", len(report.similar_past_cases) if report.similar_past_cases else 0)

    with st.container(border=True):
        st.markdown("### Top Hypothesis Memory Comparison")
        st.markdown(f"**Root Cause:** {top_cause.statement}")
        st.markdown(f"**Cause Type:** `{top_cause.cause_type}`")

        if lift > 0:
            st.success(
                f"🧠 **Memory Boosted Confidence by {lift:+.3f}**: "
                f"Historical incident memory reinforced evidence support (from `{base_conf:.3f}` to `{with_conf:.3f}`)."
            )
        else:
            st.info("ℹ️ Memory was neutral or uninfluenced for this hypothesis (no memory lift applied).")

        # Table comparison of all root causes with vs without memory
        comp_rows = []
        for rc in report.root_causes:
            rc_vals = scores.get(rc.hypothesis_id, {})
            rc_base = without_memory(rc.confidence, rc_vals)
            rc_lift = round(rc.confidence - rc_base, 3)
            comp_rows.append({
                "Hypothesis ID": rc.hypothesis_id,
                "Cause Type": rc.cause_type,
                "With Memory Score": f"{rc.confidence:.3f}",
                "Without Memory Score": f"{rc.base_score:.3f}" if hasattr(rc, "base_score") else f"{rc_base:.3f}",
                "Prior Lift Delta": f"{rc_lift:+.3f}",
            })

        if comp_rows:
            st.dataframe(comp_rows, use_container_width=True)


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
def render_report(report: InvestigationReport, events: list[dict]) -> None:
    st.subheader("📋 Executive Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", report.status.replace("_", " ").title())
    col2.metric("Impact", f"{report.impact_pct:+.1f}%")
    col3.metric("Root Causes", len(report.root_causes))
    col4.metric("Tool Calls", report.tool_calls)

    st.write(report.summary)

    # Top-Level Memory Comparison
    st.divider()
    render_memory_comparison_banner(report, events)

    # Render Visualizations
    st.divider()
    render_evidence_visualizations(events, report)

    # Render Root Cause Details
    st.divider()
    st.subheader("🔍 Identified Root Causes")
    if not report.root_causes:
        st.info("No root cause cleared the confidence threshold.")
        return

    scores = score_events(events)
    for index, cause in enumerate(report.root_causes, start=1):
        values = scores.get(cause.hypothesis_id, {})
        base = without_memory(cause.confidence, values)
        lift = round(cause.confidence - base, 3)
        with st.container(border=True):
            st.markdown(f"### #{index} {cause.cause_type.replace('_', ' ').title()}")
            st.markdown(f"**Statement:** {cause.statement}")

            c1, c2, c3 = st.columns(3)
            c1.metric("With Memory Confidence", f"{cause.confidence:.3f}")
            c2.metric("Without Memory Confidence", f"{base:.3f}")
            c3.metric("Memory Prior Lift", f"{lift:+.3f}")

    if report.ruled_out:
        st.divider()
        st.subheader("🚫 Ruled Out Hypotheses")
        for entry in report.ruled_out:
            st.markdown(f"- **{entry.hypothesis}**: {entry.why}")


def render_memory(report: InvestigationReport, events: list[dict]) -> None:
    st.subheader("🧠 Recalled Memory Cases")
    if not report.memory_used:
        st.info("Memory was not enabled or unavailable for this run.")
        return

    for entry in report.memory_used:
        status_label = "Helpful" if entry.helped else "Unused"
        with st.expander(f"{entry.recall_stage.title()} Stage ({len(entry.memory_ids)} cases recalled - {status_label})", expanded=entry.helped):
            st.markdown(f"**Query:** `{entry.query}`")
            if entry.helped_reason:
                st.caption(entry.helped_reason)
            for summary, score in zip(entry.summaries, entry.scores):
                st.markdown(f"- **Score `{score:.3f}`**: {summary}")

    st.divider()
    st.subheader("⏱️ Investigation Timeline & Trace")
    if not events:
        st.caption("No trace events recorded on disk.")
        return
    for event in events:
        kind = event.get("kind", "")
        message = str(event.get("message", ""))
        icon = {
            "evidence": "📊",
            "hypothesis": "💡",
            "memory_recall": "🧠",
            "memory_attached": "🎯",
            "score": "📈",
            "state": "🔄",
            "done": "✅",
        }.get(kind, "•")
        st.markdown(f"`{icon} {kind}` {message}")


def render_history() -> None:
    st.subheader("📜 Seeded Historical Incident Bank")
    st.caption("Closed prior cases stored in memory to guide hypothesis ranking.")
    from scripts.seed_bank import SEED_CASES  # noqa: PLC0415

    for case in SEED_CASES:
        with st.expander(f"{case['occurred_at'][:10]} — {case['document_id']}"):
            st.write(case["content"])


def render_learning_curve() -> None:
    st.subheader("📈 Learning Curve (Accuracy vs Memory Size)")
    if not CURVE_PATH.exists():
        st.info("No learning curve benchmark on disk. Run `python -m eval.learning_curve` to generate.")
        return

    data = json.loads(CURVE_PATH.read_text(encoding="utf-8"))
    scenarios = data["scenarios"]
    rows = data["results"]
    if not rows:
        st.info("Learning curve file is empty.")
        return

    by_scenario: dict[str, dict[int, dict]] = {sid: {} for sid in scenarios}
    for row in rows:
        if "error" not in row:
            by_scenario.setdefault(row["scenario"], {})[row["size"]] = row

    chart = {
        sid: {str(size): values["confidence"] for size, values in sorted(sizes.items())}
        for sid, sizes in by_scenario.items()
        if sizes
    }
    st.line_chart(chart, height=280)


def render_feedback(report: InvestigationReport) -> None:
    st.subheader("📝 Analyst Feedback & Hindsight Retain")
    st.caption("Submit human verdict to update Hindsight memory bank.")
    if not report.root_causes:
        st.info("No root cause available to evaluate.")
        return

    top = report.root_causes[0]
    st.markdown(f"**Leading Finding:** {top.statement}")

    if not report.investigation_id:
        st.error("No investigation ID found for this report.")
        return

    with st.form("feedback"):
        verdict = st.selectbox("Verdict", ["confirmed", "partially_correct", "wrong"])
        cause = st.text_input("Confirmed Root Cause", value=top.statement if verdict != "wrong" else "")
        action = st.text_input("Action Taken")
        owner = st.text_input("Owner / Team")
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Submit Feedback")

    if submitted:
        from core.config import get_settings  # noqa: PLC0415
        from core.schemas import Feedback  # noqa: PLC0415
        from memory.memory_service import MemoryService  # noqa: PLC0415

        settings = get_settings()
        memory = MemoryService(settings.hindsight, settings.bank)
        if not memory.available:
            st.error(f"Hindsight unavailable: {memory._unavailable_reason}")
            return

        try:
            if memory.ensure_bank():
                fb = Feedback(
                    investigation_id=report.investigation_id,
                    verdict=verdict,  # type: ignore[arg-type]
                    confirmed_cause=cause,
                    action_taken=action,
                    owner=owner,
                    outcome="Feedback submitted via Streamlit UI",
                    notes=notes,
                )
                if memory.retain_feedback(report, fb, report.investigation_id):
                    st.success("Feedback submitted and saved to memory!")
        finally:
            memory.close()


def run_live_investigation(scenario_id: str, enable_memory: bool) -> None:
    st.subheader(f"⚡ Running Live Investigation: {scenario_id}")

    import tools.investigation  # noqa: F401,E402
    from agent.orchestrator import Investigation, TraceEvent  # noqa: E402
    from memory.memory_service import MemoryService  # noqa: E402

    settings = get_settings()
    if not settings.groq.configured:
        st.error("GROQ_API_KEY is missing from .env. Add key to run live investigations.")
        return

    alert = build_alert(scenario_id, settings.db_path)
    memory = MemoryService(settings.hindsight, settings.bank)
    if memory.available and enable_memory:
        memory.ensure_bank()

    timeline_placeholder = st.empty()

    def on_event(event: TraceEvent) -> None:
        with timeline_placeholder.container():
            st.info(f"**{event.kind}**: {event.message}")

    investigation = Investigation(
        anomaly=alert,
        settings=settings,
        memory=memory,
        memory_enabled=enable_memory and memory.available,
        on_event=on_event,
    )

    with st.spinner("Analyzing metrics, logs, and events..."):
        try:
            report = investigation.run()
            investigation.write_trace()
            report_path = investigation.persist(report)
            if memory.available and enable_memory:
                memory.retain_report(report, report.investigation_id)
            st.success(f"Investigation completed! Saved to {report_path.name}")
            st.rerun()
        except Exception as err:
            st.error(f"Investigation failed: {err}")
        finally:
            memory.close()


# --------------------------------------------------------------------------- #
# shell
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Lumen Anomaly Investigator", page_icon="🔎", layout="wide")
    st.title("🔎 KQI Anomaly Root Cause Investigator")

    paths = report_paths()

    with st.sidebar:
        st.header("🚨 Alert Inbox & Simulator")
        selected_scenario = st.selectbox(
            "Select Scenario Alert",
            options=sorted(SCENARIOS.keys()),
            format_func=lambda s: f"{s}: {SCENARIOS[s].title}",
        )
        scenario_meta = SCENARIOS[selected_scenario]
        st.info(f"**Title:** {scenario_meta.title}\n\n**Cause:** {scenario_meta.truth.cause_type}")

        use_memory = st.checkbox("Enable Hindsight Memory", value=True)
        if st.button("🚀 Start Live Investigation", type="primary", use_container_width=True):
            run_live_investigation(selected_scenario, use_memory)

        st.divider()
        st.header("📂 Saved Reports")

        report_options = ["-- Choose an investigation --"]
        report_map = {}
        for path in paths:
            rep = load_report(path)
            if rep:
                label = f"{rep.investigation_id or path.stem} ({rep.status})"
                report_options.append(label)
                report_map[label] = (path, rep)

        selected_option = st.selectbox("Select Past Investigation", options=report_options)
        if selected_option != "-- Choose an investigation --":
            path, report = report_map[selected_option]
        else:
            report = None

    if report is None:
        st.markdown("""
        ### Welcome to Lumen Anomaly Investigator

        Lumen investigates revenue & KQI metric anomalies by analyzing metric breakdowns, business events, deployment logs, and historical memory.

        **To get started:**
        - Select an alert scenario in the sidebar and click **🚀 Start Live Investigation**, OR
        - Select an existing report under **📂 Saved Reports** in the sidebar.
        """)
        return

    tabs = st.tabs(["Report & Visualizations", "Memory & Trace", "Historical Bank", "Learning Curve", "Analyst Feedback"])
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
