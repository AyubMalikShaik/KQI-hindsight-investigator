"""Investigation orchestrator.

An explicit bounded state machine rather than one large prompt, so every run is
traceable, every budget is enforced in code, and the UI can render progress.

    RECEIVED -> SCOPING -> RECALL_PRIOR -> PLANNING -> EVIDENCE_GATHERING
    -> RECALL_EVIDENCE -> HYPOTHESIZING -> VALIDATING -> (loop or) REPORTING -> CLOSED
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from agent import prompts
from agent.llm import GroqClient, LLMResponse
from core.config import ARTIFACTS_DIR, Settings, get_settings
from core.schemas import (
    Anomaly,
    Evidence,
    Hypothesis,
    HypothesisVerdict,
    InvestigationPlan,
    InvestigationReport,
    MemoryUsed,
    RecommendedAction,
    RootCause,
    RuledOut,
    SimilarPastCase,
    ToolError,
)
from memory.memory_service import STAGE_CONFIRMED, MemoryService, RecalledMemory
from tools.guard import ToolCallGuard
from tools.registry import registry

import tools.investigation  # noqa: F401  (importing registers the investigation tools)

from validation import confidence as confidence_mod

log = logging.getLogger("agent.orchestrator")

TRACE_DIR = ARTIFACTS_DIR / "traces"
REPORT_DIR = ARTIFACTS_DIR / "reports"


class State(str, Enum):
    RECEIVED = "RECEIVED"
    SCOPING = "SCOPING"
    RECALL_PRIOR = "RECALL_PRIOR"
    PLANNING = "PLANNING"
    EVIDENCE_GATHERING = "EVIDENCE_GATHERING"
    RECALL_EVIDENCE = "RECALL_EVIDENCE"
    HYPOTHESIZING = "HYPOTHESIZING"
    VALIDATING = "VALIDATING"
    REPORTING = "REPORTING"
    CLOSED = "CLOSED"


@dataclass
class TraceEvent:
    state: str
    kind: str
    message: str
    ts: str
    data: dict[str, Any] = field(default_factory=dict)


class Investigation:
    def __init__(
        self,
        anomaly: Anomaly,
        settings: Settings,
        memory: MemoryService | None = None,
        memory_enabled: bool = True,
        llm: GroqClient | None = None,
        on_event: Callable[[TraceEvent], None] | None = None,
    ) -> None:
        self.anomaly = anomaly
        self.settings = settings
        self.trace_id = uuid.uuid4().hex[:16]
        self.investigation_id = f"INV-{datetime.now(timezone.utc):%Y%m%d}-{self.trace_id[:6]}"
        self.memory_enabled = memory_enabled
        self.memory = memory
        self.on_event = on_event
        self.budget = settings.budget

        self.llm = llm or GroqClient(settings.groq)
        self.guard = ToolCallGuard(
            registry,
            max_repeat_calls=self.budget.max_repeat_calls,
            tool_timeout_s=self.budget.tool_timeout_s,
            max_calls=self.budget.max_tool_calls,
        )

        self.evidence: list[Evidence] = []
        self.hypotheses: list[Hypothesis] = []
        self.verdicts: list[HypothesisVerdict] = []
        self.memory_used: list[MemoryUsed] = []
        self.plan: InvestigationPlan | None = None
        self.recalled: dict[str, list[RecalledMemory]] = {}
        self.events: list[TraceEvent] = []
        self.started = time.perf_counter()
        self.state = State.RECEIVED
        self.llm_turns = 0
        self.validation_loops = 0
        self._valid_ids: set[str] = set()

    # ------------------------------------------------------------------ #
    # trace
    # ------------------------------------------------------------------ #
    def emit(self, kind: str, message: str, **data: Any) -> TraceEvent:
        event = TraceEvent(
            state=self.state.value,
            kind=kind,
            message=message,
            ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            data=data,
        )
        self.events.append(event)
        if self.on_event:
            self.on_event(event)
        return event

    def transition(self, state: State, message: str = "") -> None:
        previous = self.state
        self.state = state
        self.emit("state", message or f"{previous.value} -> {state.value}", **{
            "from": previous.value,
            "to": state.value,
        })

    def write_trace(self) -> Path:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        path = TRACE_DIR / f"{self.trace_id}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event.__dict__, default=str) + "\n")
        return path

    # ------------------------------------------------------------------ #
    # memory helpers
    # ------------------------------------------------------------------ #
    def _memory_available(self) -> bool:
        return bool(self.memory_enabled and self.memory and self.memory.available)

    def _as_of(self) -> datetime:
        return datetime.combine(
            self.anomaly.detected_at.date(), datetime.min.time(), tzinfo=timezone.utc
        )

    def _signature_query(self) -> str:
        direction = self.anomaly.direction.value
        return (
            f"{self.anomaly.metric} {direction} of {abs(self.anomaly.deviation_pct):.0f}%, "
            f"anomaly on {self.anomaly.anomaly_window.start}. "
            "What root cause did we find last time something like this happened?"
        )

    def _recall(
        self, stage: str, query: str, max_tokens: int | None = None
    ) -> None:
        if not self._memory_available():
            reason = (
                "disabled for this run"
                if not self.memory_enabled
                else "memory service unavailable"
            )
            self.emit("memory_skipped", f"Memory recall skipped: {reason}", stage=stage)
            self.memory_used.append(
                MemoryUsed(query=query, recall_stage=stage, helped_reason=reason)
            )
            return

        assert self.memory is not None
        result = self.memory.recall(
            query=query,
            stage=stage,
            as_of=self._as_of(),
            max_tokens=max_tokens or self.budget.recall_max_tokens,
            budget=self.budget.recall_budget,
            # An investigation must not read back the report written about the
            # alert it is currently working, or its own conclusion becomes the
            # prior that confirms it.
            exclude_alert=self.anomaly.alert_id,
        )
        if not result.ok:
            self.emit("memory_error", f"Recall failed: {result.error}", stage=stage)
            self.memory_used.append(
                MemoryUsed(
                    query=query, recall_stage=stage, helped_reason=f"recall error: {result.error}"
                )
            )
            return

        self.recalled[stage] = result.memories
        self.emit(
            "memory_recall",
            f"Recalled {len(result.memories)} prior case(s) at {stage}",
            stage=stage,
            query=query,
            memory_ids=result.ids(),
            summaries=[m.text for m in result.memories],
        )
        self.memory_used.append(
            MemoryUsed(
                query=query,
                recall_stage=stage,
                memory_ids=result.ids(),
                summaries=[m.text for m in result.memories],
                scores=[m.score for m in result.memories],
                confirmed_causes=result.confirmed_causes(),
            )
        )

    def _memory_block(self, stage: str) -> str:
        memories = self.recalled.get(stage, [])
        return prompts.format_memory(memories, stage, self.memory_used)

    def _attribute_memory(self) -> None:
        """Decide whether recalled memory actually influenced the conclusion.

        Deliberately conservative and evidence-based rather than optimistic:
        a memory counts as having helped only when a surviving hypothesis
        explicitly carries it as a prior, or the planner pruned a check because
        of it. Everything else is reported as recalled-but-unused, which is the
        honest answer and is exactly what the memory A/B comparison needs.
        """
        cited: dict[str, list[str]] = {}
        for hypothesis in self.hypotheses:
            for memory_id in hypothesis.memory_prior_ids:
                cited.setdefault(memory_id, []).append(hypothesis.hypothesis_id)

        pruned_checks = {p.lower() for p in (self.plan.pruned_checks if self.plan else [])}

        for entry in self.memory_used:
            entry.cited_by_hypothesis_ids = sorted(
                {
                    hypothesis_id
                    for memory_id in entry.memory_ids
                    for hypothesis_id in cited.get(memory_id, [])
                }
            )
            survived = [
                hypothesis_id
                for hypothesis_id in entry.cited_by_hypothesis_ids
                if any(
                    h.hypothesis_id == hypothesis_id and h.status == "supported"
                    for h in self.hypotheses
                )
            ]
            if survived:
                entry.helped = True
                entry.helped_reason = (
                    f"hypothesis {', '.join(survived)} was formed with this case as a prior "
                    "and survived validation on current evidence"
                )
            elif entry.cited_by_hypothesis_ids:
                entry.helped = False
                entry.helped_reason = (
                    "cited as a prior, but the hypothesis that used it did not survive "
                    "validation, so the memory did not carry the conclusion"
                )
            elif pruned_checks:
                entry.helped = True
                entry.helped_reason = (
                    f"planner pruned checks ({', '.join(sorted(pruned_checks))}) on the "
                    "strength of a recalled case"
                )
            elif entry.memory_ids:
                entry.helped = False
                entry.helped_reason = (
                    "recalled and shown to the model, but no hypothesis claimed it as a prior "
                    "and no check was pruned, so it did not change the investigation"
                )

    # ------------------------------------------------------------------ #
    # llm helper
    # ------------------------------------------------------------------ #
    def _json_call(
        self, purpose: str, messages: list[dict[str, Any]], schema_hint: str, max_tokens: int = 1600
    ) -> tuple[dict[str, Any] | None, str]:
        """Call the model for JSON, retrying once if the answer is truncated.

        `gpt-oss` is a chatty model: with a low token ceiling it returns
        syntactically incomplete JSON. Retrying with an explicit brevity
        instruction is far cheaper than failing the whole investigation.
        """
        for attempt in range(2):
            if self.llm_turns >= self.budget.max_llm_turns:
                self.emit("budget", f"LLM turn budget exhausted before {purpose}")
                return None, ""
            self.llm_turns += 1
            parsed, raw, finish = self.llm.complete_json(
                messages, schema_hint=schema_hint, max_tokens=max_tokens
            )
            if parsed is not None:
                self.emit("llm", f"{purpose} complete", model=self.llm.model)
                return parsed, raw
            if finish == "length":
                self.emit(
                    "llm_truncated",
                    f"{purpose}: response hit the {max_tokens}-token ceiling mid-JSON; "
                    "retrying with a brevity instruction",
                )
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous answer was cut off. Reply with the complete JSON object only. "
                            "No prose, no preamble, no markdown fence. Keep every reason under 30 words."
                        ),
                    }
                ]
                max_tokens = int(max_tokens * 1.75)
                continue
            self.emit("llm_parse_error", f"{purpose}: model did not return valid JSON", raw=raw[:400])
            return None, raw
        self.emit("llm_parse_error", f"{purpose}: still invalid JSON after a brevity retry", raw=raw[:400])
        return None, raw

    # ------------------------------------------------------------------ #
    # states
    # ------------------------------------------------------------------ #
    def run(self) -> InvestigationReport:
        self.transition(State.SCOPING, "Confirming the anomaly is real")
        self._scoping()

        self.transition(State.RECALL_PRIOR, "Recalling prior investigations by signature")
        self._recall("after_scoping", self._signature_query())

        self.transition(State.PLANNING, "Planning the investigation")
        self._planning()

        self.transition(State.EVIDENCE_GATHERING, "Gathering evidence")
        self._evidence_gathering()

        self.transition(State.RECALL_EVIDENCE, "Recalling cases matching the evidence")
        self._recall("after_evidence", self._evidence_query())

        self.transition(State.HYPOTHESIZING, "Forming candidate causes")
        self._hypothesizing()

        self.transition(State.VALIDATING, "Validating hypotheses against evidence")
        self._validating()

        # Must happen after validation: whether memory helped depends on which
        # hypotheses that carried it as a prior actually survived.
        self._attribute_memory()

        self.transition(State.REPORTING, "Writing the report")
        report = self._reporting()

        self.transition(State.CLOSED, "Investigation complete")
        self.emit(
            "done",
            f"Completed in {time.perf_counter() - self.started:.1f}s, "
            f"{self.llm_turns} LLM turns, {self.guard.calls_used} tool calls",
        )
        return report

    def _scoping(self) -> None:
        self.evidence.clear()
        self._call_tool(
            "analyze_metric",
            {
                "metric": self.anomaly.metric,
                "start": str(self.anomaly.anomaly_window.start),
                "end": str(self.anomaly.anomaly_window.end),
                "baseline_start": str(self.anomaly.baseline_window.start),
                "baseline_end": str(self.anomaly.baseline_window.end),
            },
            stage="scoping",
        )

    def _planning(self) -> None:
        scope = [e for e in self.evidence if e.tool == "analyze_metric"]
        messages = [
            {"role": "system", "content": prompts.SYSTEM},
            {
                "role": "user",
                "content": prompts.PLANNER.format(
                    alert=prompts.format_alert(self.anomaly),
                    tool_names=", ".join(registry.names()),
                    scope_evidence=prompts.format_evidence(scope),
                    memory_block=self._memory_block("after_scoping"),
                ),
            },
        ]
        parsed, _ = self._json_call(
            "planning", messages, '{"objective":str,"steps":[str],"tools_to_call":[str],"pruned_checks":[str],"memory_informed":bool}'
        )
        if parsed:
            self.plan = InvestigationPlan(**{k: v for k, v in parsed.items() if k in InvestigationPlan.model_fields})
            self.emit(
                "plan",
                self.plan.objective,
                steps=self.plan.steps,
                tools_to_call=self.plan.tools_to_call,
                pruned_checks=self.plan.pruned_checks,
                memory_informed=self.plan.memory_informed,
            )
        else:
            self.emit("plan_skipped", "Planner returned no usable plan; using a default sequence")

    def _evidence_gathering(self) -> None:
        preferred = list(self.plan.tools_to_call) if self.plan else []
        pruned = {p.lower() for p in (self.plan.pruned_checks if self.plan else [])}
        executed: set[str] = set()

        for name in preferred:
            if self.guard.call_budget_remaining() <= 0:
                break
            if name not in registry.names():
                self.emit("plan_invalid", f"Planner named unknown tool '{name}'; ignoring")
                continue
            if name in executed:
                continue
            if any(p in name.lower() for p in pruned):
                self.emit(
                    "check_pruned",
                    f"Skipped {name}: a recalled prior case already ruled it out",
                    tool=name,
                )
                continue
            args = self._default_args(name)
            if args is None:
                continue
            executed.add(name)
            self._call_tool(name, args, stage="evidence")

        if not executed:
            self.emit(
                "fallback_sequence",
                "Plan produced no usable calls; running the systematic evidence sequence",
            )
            for name, args in self._evidence_sequence():
                if self.guard.call_budget_remaining() <= 0:
                    self.emit("budget", "Tool call budget exhausted during evidence gathering")
                    break
                key = f"{name}:{args.get('dimension', '')}"
                if key in executed:
                    continue
                executed.add(key)
                self._call_tool(name, args, stage="evidence")

    def _evidence_sequence(self) -> list[tuple[str, dict[str, Any]]]:
        start = str(self.anomaly.anomaly_window.start)
        end = str(self.anomaly.anomaly_window.end)
        base_start = str(self.anomaly.baseline_window.start)
        base_end = str(self.anomaly.baseline_window.end)
        metric = self.anomaly.metric

        sequence: list[tuple[str, dict[str, Any]]] = [
            (
                "find_related_metrics",
                {
                    "metric": metric,
                    "start": start,
                    "end": end,
                    "baseline_start": base_start,
                    "baseline_end": base_end,
                },
            ),
            (
                "breakdown_by_dimension",
                {
                    "metric": metric,
                    "dimension": "country",
                    "start": start,
                    "end": end,
                    "baseline_start": base_start,
                    "baseline_end": base_end,
                },
            ),
        ]
        for dimension in ("platform", "payment_method", "channel", "app_version"):
            sequence.append(
                (
                    "breakdown_by_dimension",
                    {
                        "metric": metric,
                        "dimension": dimension,
                        "start": start,
                        "end": end,
                        "baseline_start": base_start,
                        "baseline_end": base_end,
                    },
                )
            )
        changepoint = self._changepoint()
        if changepoint:
            from datetime import timedelta

            center = datetime.fromisoformat(changepoint)
            sequence.append(
                (
                    "query_business_events",
                    {
                        "start": str((center - timedelta(days=2)).date()),
                        "end": str((center + timedelta(days=1)).date()),
                    },
                )
            )
        return sequence

    def _default_args(self, name: str) -> dict[str, Any] | None:
        if name == "query_config_changes":
            changepoint = self._changepoint()
            if changepoint:
                from datetime import timedelta

                center = datetime.fromisoformat(changepoint)
                return {
                    "start": str((center - timedelta(days=2)).date()),
                    "end": str((center + timedelta(days=1)).date()),
                }
            return None
        for candidate, args in self._evidence_sequence():
            if candidate == name:
                return args
        return None

    def _changepoint(self) -> str | None:
        for item in self.evidence:
            if item.tool == "analyze_metric":
                value = item.numbers.get("changepoint")
                if value:
                    return str(value)
        return str(self.anomaly.anomaly_window.end)

    def _evidence_query(self) -> str:
        parts: list[str] = []
        for item in self.evidence:
            if item.tool == "breakdown_by_dimension":
                segments = item.numbers.get("segments") or []
                if segments:
                    top = segments[0]
                    parts.append(
                        f"{item.numbers.get('dimension')}={top.get('segment')} "
                        f"accounted for {top.get('share_of_decline_pct')}% of the decline"
                    )
            elif item.tool == "find_related_metrics":
                moved = [
                    f"{m['metric']} {m['delta_pct']:+.0f}%"
                    for m in item.numbers.get("related", [])
                    if m.get("moved")
                ]
                if moved:
                    parts.append("moved metrics: " + ", ".join(moved))
        summary = "; ".join(parts) or "no localised segment yet"
        return (
            f"{self.anomaly.metric} anomaly on {self.anomaly.anomaly_window.start}. "
            f"Evidence so far: {summary}. "
            "What past investigation had this same evidence pattern, and what was the confirmed cause?"
        )

    def _hypothesizing(self) -> None:
        all_memory_ids = [i for m in self.memory_used for i in m.memory_ids]
        memory_ids = all_memory_ids
        messages = [
            {"role": "system", "content": prompts.SYSTEM},
            {
                "role": "user",
                "content": prompts.HYPOTHESIZER.format(
                    alert=prompts.format_alert(self.anomaly),
                    evidence_block=prompts.format_evidence(self.evidence),
                    memory_block=self._memory_block("after_evidence"),
                ),
            },
        ]
        parsed, _ = self._json_call(
            "hypothesizing",
            messages,
            '{"hypotheses":[{"hypothesis_id":str,"statement":str,"cause_type":str,'
            '"predicted_evidence":[str],"affected_segments":{},'
            f'"memory_prior_ids":[str]}}]}} recalled memory ids: {sorted(all_memory_ids) or "none"}',
            max_tokens=2000,
        )
        if not parsed or not parsed.get("hypotheses"):
            self.emit(
                "hypotheses_failed",
                "The model returned no hypotheses; building them from tool evidence instead",
            )
            self._evidence_driven_hypotheses()
            return

        for index, raw in enumerate(parsed["hypotheses"][:6], start=1):
            try:
                hypothesis = Hypothesis(
                    hypothesis_id=str(raw.get("hypothesis_id") or f"H{index}"),
                    statement=str(raw.get("statement", "")).strip(),
                    cause_type=raw.get("cause_type", "unknown"),
                    predicted_evidence=[str(p) for p in raw.get("predicted_evidence", [])],
                    affected_segments={
                        str(k): str(v) for k, v in (raw.get("affected_segments") or {}).items()
                    },
                    memory_prior_ids=(
                        [m for m in (raw.get("memory_prior_ids") or []) if m in memory_ids]
                    ),
                    source="memory_prior" if raw.get("memory_prior_ids") else "fresh",
                )
            except Exception as exc:  # noqa: BLE001
                self.emit("hypothesis_rejected", f"Dropped malformed hypothesis: {exc}")
                continue
            self.hypotheses.append(hypothesis)
            self.emit(
                "hypothesis",
                f"{hypothesis.hypothesis_id}: {hypothesis.statement}",
                hypothesis_id=hypothesis.hypothesis_id,
                cause_type=hypothesis.cause_type,
                source=hypothesis.source,
                predicted_evidence=hypothesis.predicted_evidence,
            )

    def _candidate_events(self) -> list[tuple[str, str]]:
        """Recorded changes that could be a candidate cause, as (ts, source).

        The source matters: a config push can only ever explain a config
        hypothesis, and a release can never explain one. Scoring a payment
        failure as temporally aligned because a config change happened to land
        nearby let the vaguer hypothesis borrow the sharper one's evidence.
        """
        events: list[tuple[str, str]] = []
        for item in self.evidence:
            if item.tool == "query_business_events":
                events.extend((e["ts"], "business") for e in item.numbers.get("events", []))
            elif item.tool == "query_config_changes":
                events.extend((c["ts"], "config") for c in item.numbers.get("changes", []))
        return events

    def _validating(self) -> None:
        changepoint = self._changepoint()
        event_times = self._candidate_events()

        for _ in range(self.budget.max_validation_loops + 1):
            messages = [
                {"role": "system", "content": prompts.SYSTEM},
                {
                    "role": "user",
                    "content": prompts.VALIDATOR.format(
                        alert=prompts.format_alert(self.anomaly),
                        evidence_block=prompts.format_evidence(self.evidence),
                        hypotheses=prompts.format_hypotheses(self.hypotheses),
                        tool_names=", ".join(registry.names()),
                        tool_schemas=prompts.format_tool_schemas(registry),
                    ),
                },
            ]
            parsed, _ = self._json_call(
                "validating",
                messages,
                '{"verdicts":[{"hypothesis_id":str,"verdict":str,"reason":str,'
                '"evidence_ids":[str],"requested_tools":[]}]}',
                max_tokens=1800,
            )
            if not parsed or not parsed.get("verdicts"):
                # Do not break here. Scoring is the system's own authority and
                # does not depend on the validator answering, so bailing out
                # before _score left every hypothesis unscored, which in turn
                # left the report claiming nothing cleared the threshold.
                self.emit("validation_failed", "Validator returned no verdicts this loop")
                scored = self._score(changepoint, event_times)
                for hypothesis, confidence in scored:
                    self.emit(
                        "score",
                        f"{hypothesis.hypothesis_id} confidence {confidence.total:.2f} "
                        f"(support={confidence.evidence_support} "
                        f"magnitude={confidence.magnitude_explained} "
                        f"temporal={confidence.temporal_alignment} "
                        f"prior={confidence.memory_prior} "
                        f"contradictions={confidence.contradictions})",
                        hypothesis_id=hypothesis.hypothesis_id,
                        notes=confidence.notes,
                    )
                if scored:
                    self._settle_without_verdict(max(scored, key=lambda pair: pair[1].total))
                break

            self.verdicts = []
            for raw in parsed["verdicts"]:
                verdict = HypothesisVerdict(
                    hypothesis_id=str(raw.get("hypothesis_id", "")),
                    verdict=raw.get("verdict", "needs_more_evidence"),
                    reason=str(raw.get("reason", "")),
                    evidence_ids=[str(i) for i in raw.get("evidence_ids", [])],
                )
                if verdict.hypothesis_id and verdict.hypothesis_id in {
                    h.hypothesis_id for h in self.hypotheses
                }:
                    self.verdicts.append(verdict)
                    self.emit(
                        "verdict",
                        f"{verdict.hypothesis_id}: {verdict.verdict}",
                        hypothesis_id=verdict.hypothesis_id,
                        verdict=verdict.verdict,
                        reason=verdict.reason,
                    )

            for raw in parsed["verdicts"]:
                for request in raw.get("requested_tools", []) or []:
                    name = request.get("name")
                    args = request.get("arguments") or {}
                    if name not in registry.names():
                        self.emit(
                            "guard",
                            f"Validator asked for unknown tool '{name}'; "
                            f"available: {', '.join(registry.names())}",
                            tool=str(name),
                        )
                        continue
                    if self.guard.call_budget_remaining() <= 0:
                        self.emit("budget", "Tool budget exhausted; ignoring validator tool request")
                        continue
                    self._call_tool(name, args, stage="validation")

            for hypothesis in self.hypotheses:
                verdict = next(
                    (v for v in self.verdicts if v.hypothesis_id == hypothesis.hypothesis_id), None
                )
                if not verdict:
                    continue

                # A kill must cite the evidence that contradicts the hypothesis.
                # An uncited kill is the model's opinion, not a refutation, and
                # treating it as final is how the correct answer gets discarded.
                cited = [i for i in verdict.evidence_ids if i in self._valid_ids]
                if verdict.verdict == "killed" and not cited:
                    self.emit(
                        "kill_downgraded",
                        f"{hypothesis.hypothesis_id}: killed with no citable evidence "
                        f"({verdict.reason[:80]!r}); downgraded to needs_more_evidence",
                        hypothesis_id=hypothesis.hypothesis_id,
                    )
                    verdict.verdict = "needs_more_evidence"
                    self.emit(
                        "hypothesis_open",
                        f"{hypothesis.hypothesis_id}: needs_more_evidence",
                        hypothesis_id=hypothesis.hypothesis_id,
                        reason=verdict.reason,
                    )
                    continue

                # A kill, once made, is final. Without this a later loop that
                # sees fresher evidence can quietly resurrect a hypothesis the
                # validator already refuted, making the run non-monotonic.
                if hypothesis.status == "killed":
                    if verdict.verdict == "supported":
                        self.emit(
                            "kill_upheld",
                            f"{hypothesis.hypothesis_id}: was already killed "
                            f"({hypothesis.kill_reason}); ignoring a later 'supported' verdict",
                            hypothesis_id=hypothesis.hypothesis_id,
                        )
                    continue

                if verdict.verdict == "killed":
                    hypothesis.status = "killed"
                    hypothesis.kill_reason = verdict.reason
                elif verdict.verdict == "supported":
                    hypothesis.status = "supported"

            scored = self._score(changepoint, event_times)
            for hypothesis, confidence in scored:
                self.emit(
                    "score",
                    f"{hypothesis.hypothesis_id} confidence {confidence.total:.2f}"
                    f" (support={confidence.evidence_support} magnitude={confidence.magnitude_explained}"
                    f" temporal={confidence.temporal_alignment} prior={confidence.memory_prior}"
                    f" contradictions={confidence.contradictions})",
                    hypothesis_id=hypothesis.hypothesis_id,
                    notes=confidence.notes,
                )
            top = max(scored, key=lambda s: s[1].total) if scored else None
            if top and top[1].total >= self.budget.confidence_threshold:
                self.emit(
                    "confidence",
                    f"Top hypothesis {top[0].hypothesis_id} reached "
                    f"{top[1].total:.2f} (threshold {self.budget.confidence_threshold})",
                    notes=top[1].notes,
                )
                break

            # The validator was unavailable this pass. Settle the leading
            # hypothesis on its computed score so the report cannot contradict
            # itself, and record the gap rather than hiding it.
            if not self.verdicts and top is not None:
                self._settle_without_verdict(top)
                break

            self.validation_loops += 1
            if self.validation_loops > self.budget.max_validation_loops:
                self.emit("budget", "Validation loop budget exhausted")
                break
            self.emit("validate_again", "Confidence below threshold; running another validation pass")

    def _settle_without_verdict(self, top: tuple) -> None:
        """Carry the leading hypothesis on its computed score when no verdict exists.

        A hypothesis can only be marked 'supported' by a validator verdict, so a
        rate-limited validator used to force 'partial' even with a 0.95 computed
        score, and the report then listed root causes beside a status and
        summary that said otherwise. The computed score is the system's own
        authority; the missing corroboration is emitted explicitly.
        """
        hypothesis, confidence = top
        if hypothesis.status == "killed":
            return
        if confidence.total < self.budget.confidence_threshold:
            self.emit(
                "verdict_unavailable",
                f"Validator produced no verdicts and {hypothesis.hypothesis_id} scored "
                f"{confidence.total:.2f}, below the {self.budget.confidence_threshold} "
                "threshold; leaving it unconfirmed",
                hypothesis_id=hypothesis.hypothesis_id,
                confidence=confidence.total,
            )
            return

        hypothesis.status = "supported"
        self.emit(
            "verdict_unavailable",
            f"Validator produced no verdicts this pass; {hypothesis.hypothesis_id} is "
            f"carried on its computed score of {confidence.total:.2f} (threshold "
            f"{self.budget.confidence_threshold}) without a corroborating verdict",
            hypothesis_id=hypothesis.hypothesis_id,
            confidence=confidence.total,
        )

    def _score(self, changepoint: str | None, event_times: list[tuple[str, str]]) -> list[tuple]:
        """Return (hypothesis, confidence) pairs, scored deterministically.

        The score is written back onto the hypothesis so that anything reading
        hypothesis.confidence later (the report summary, the status decision)
        sees the computed value rather than the field's 0.0 default.
        """
        scored: list[tuple] = []
        for hypothesis in self.hypotheses:
            confidence = confidence_mod.score_hypothesis(
                hypothesis, self.evidence, self.verdicts, changepoint, event_times, self.memory_used
            )
            hypothesis.confidence = confidence.total
            scored.append((hypothesis, confidence))
        return scored

    def _top_data_quality(self) -> Hypothesis | None:
        """The best-scored hypothesis, if it fully observed a data-quality reading."""
        ranked = sorted(
            (h for h in self.hypotheses if h.status != "killed"),
            key=lambda h: h.confidence,
            reverse=True,
        )
        if not ranked:
            return None
        best = ranked[0]
        if best.cause_type != "data_quality" or not best.predicted_evidence:
            return None
        support, _ = confidence_mod.evidence_support(best, self.evidence, self.verdicts)
        return best if support >= 0.999 else None

    def _reporting(self) -> InvestigationReport:
        changepoint = self._changepoint()
        event_times = self._candidate_events()

        scored = self._score(changepoint, event_times)
        scored.sort(key=lambda s: s[1].total, reverse=True)

        messages = [
            {"role": "system", "content": prompts.SYSTEM},
            {
                "role": "user",
                "content": prompts.REPORTER.format(
                    alert=prompts.format_alert(self.anomaly),
                    evidence_block=prompts.format_evidence(self.evidence),
                    verdicts=prompts.format_hypotheses(self.hypotheses)
                    + "\n\nVERDICTS:\n"
                    + "\n".join(
                        f"{v.hypothesis_id}: {v.verdict} - {v.reason}" for v in self.verdicts
                    ),
                    memory_block=self._memory_block("after_evidence"),
                    tool_names=", ".join(registry.names()),
                ),
            },
        ]

        parsed, _ = self._json_call(
            "reporting",
            messages,
            '{"status":str,"summary":str,"impact_note":str,"root_causes":[],'
            '"ruled_out":[],"recommended_actions":[],"open_questions":[],"data_quality_notes":[]}',
            max_tokens=2200,
        )

        valid_ids = self._valid_ids
        report = self._assemble_report(parsed, scored)

        problems = confidence_mod.validate_citations(report, valid_ids)
        if problems:
            self.emit("citation_violation", "; ".join(problems))

        # Repair rather than merely complain: strip hallucinated ids, then fall
        # back to the evidence that actually supported the hypothesis.
        for cause in report.root_causes:
            filtered = [i for i in cause.evidence_ids if i in valid_ids]
            if len(filtered) != len(cause.evidence_ids):
                self.emit(
                    "citation_repaired",
                    f"{cause.hypothesis_id}: dropped "
                    f"{len(cause.evidence_ids) - len(filtered)} untraceable evidence id(s)",
                )
            cause.evidence_ids = filtered

        for cause in report.root_causes:
            if cause.evidence_ids:
                continue
            replacement = self._supporting_evidence(
                next(
                    (
                        h
                        for h in self.hypotheses
                        if h.hypothesis_id == cause.hypothesis_id
                    ),
                    None,
                )
            )
            fallback = [i for i in replacement if i in valid_ids]
            if fallback:
                cause.evidence_ids = fallback
                self.emit(
                    "citation_repaired",
                    f"{cause.hypothesis_id}: attached the evidence that supported it",
                )
            else:
                self.emit(
                    "citation_unresolvable",
                    f"{cause.hypothesis_id} has no traceable evidence; "
                    "the report should not present it as a finding",
                )

        return report

    def _assemble_report(
        self, parsed: dict[str, Any] | None, scored: list
    ) -> InvestigationReport:
        confidence_by_id = {h.hypothesis_id: c.total for h, c in scored}
        impact_abs, impact_pct = self._impact()
        reporter_impact_note = str((parsed or {}).get("impact_note", "")).strip()
        if reporter_impact_note:
            self.emit(
                "impact_note_replaced",
                f"reporter impact note discarded in favour of computed figures: "
                f"{reporter_impact_note[:120]}",
            )

        root_causes: list[RootCause] = []
        if parsed and parsed.get("root_causes"):
            for raw in parsed["root_causes"]:
                hypothesis_id = str(raw.get("hypothesis_id", ""))
                # The computed score is authoritative. The model may restate it
                # in prose, but it must not be able to inflate its own number.
                computed = confidence_by_id.get(hypothesis_id)
                claimed = raw.get("confidence")
                confidence = computed if computed is not None else float(claimed or 0.0)
                if computed is not None and claimed is not None:
                    claimed_value = float(claimed)
                    if abs(claimed_value - computed) > 0.15:
                        self.emit(
                            "confidence_mismatch",
                            f"{hypothesis_id}: model claimed {claimed_value:.2f}, "
                            f"computed {computed:.2f}; using the computed value",
                        )
                root_causes.append(
                    RootCause(
                        hypothesis_id=hypothesis_id,
                        statement=str(raw.get("statement", "")),
                        cause_type=str(raw.get("cause_type", "unknown")),
                        confidence=confidence,
                        evidence_ids=[str(i) for i in raw.get("evidence_ids", [])],
                        contribution_pct=float(raw.get("contribution_pct", 0.0) or 0.0),
                    )
                )
        else:
            # The reporter named no root causes, so fall back to the
            # hypotheses the run actually settled on. Listing any candidate
            # above 0.2 here is what produced a report that showed a 0.85
            # "root cause" beside a status of partial and a summary saying
            # nothing had cleared the threshold. Only settled hypotheses
            # belong in this field; an unsettled one is an open question.
            for hypothesis, confidence in scored:
                if hypothesis.status != "supported":
                    continue
                if confidence.total < self.budget.confidence_threshold:
                    continue
                root_causes.append(
                    RootCause(
                        hypothesis_id=hypothesis.hypothesis_id,
                        statement=hypothesis.statement,
                        cause_type=hypothesis.cause_type,
                        confidence=confidence.total,
                        evidence_ids=self._supporting_evidence(hypothesis),
                        contribution_pct=0.0,
                    )
                )

        # Killed hypotheses and the model's own ruled_out list describe the same
        # set, so merge them on the hypothesis id instead of appending both.
        killed = [h for h in self.hypotheses if h.status == "killed"]
        model_reasons: dict[str, tuple[str, list[str]]] = {}
        if parsed and parsed.get("ruled_out"):
            for raw in parsed["ruled_out"]:
                text = str(raw.get("hypothesis", ""))
                model_reasons[text] = (
                    str(raw.get("why", "")),
                    [str(i) for i in raw.get("evidence_ids", [])],
                )

        statement_to_hypothesis = {h.statement: h for h in self.hypotheses}
        # The reporter may identify a hypothesis by its id ("H2") rather than
        # by restating it, so match on either form before treating a model
        # ruled_out entry as an additional finding.
        killed_keys: set[str] = set()
        for hypothesis in killed:
            killed_keys.add(hypothesis.statement.strip().lower())
            killed_keys.add(hypothesis.hypothesis_id.strip().lower())

        ruled_out: list[RuledOut] = []
        seen: set[str] = set()

        def add_ruled_out(hypothesis_text: str, why: str, cited: list[str]) -> None:
            key = (hypothesis_text or "").strip().lower()
            if not key or key in seen:
                return
            seen.add(key)
            ruled_out.append(
                RuledOut(hypothesis=hypothesis_text, why=why, evidence_ids=cited)
            )

        for hypothesis in killed:
            why, cited = hypothesis.kill_reason or "contradicted by the evidence gathered", (
                self._supporting_evidence(hypothesis)
            )
            model_entry = model_reasons.get(hypothesis.statement) or model_reasons.get(
                hypothesis.hypothesis_id
            )
            if model_entry and model_entry[0]:
                why = model_entry[0]
            if model_entry and model_entry[1]:
                cited = model_entry[1]
            add_ruled_out(hypothesis.statement, why, cited)

        for text, (why, cited) in model_reasons.items():
            match = statement_to_hypothesis.get(text)
            if match is None:
                match = next(
                    (
                        h
                        for h in killed
                        if h.hypothesis_id.strip().lower() == text.strip().lower()
                    ),
                    None,
                )
            if match and match.status == "killed":
                continue
            if text.strip().lower() in killed_keys:
                continue
            if any(text.strip().lower() == r.hypothesis.strip().lower() for r in ruled_out):
                continue
            add_ruled_out(text, why, cited)

        # Similarity must come from Hindsight's own relevance score. A hardcoded
        # 0.5 would be a fabricated number in a field the UI shows to users.
        recalled_contexts = {
            memory.memory_id: memory.context
            for memories in self.recalled.values()
            for memory in memories
        }
        # A case is recalled at more than one stage and each recall scores it
        # differently, so keep the most relevant score seen. Reporting the first
        # recall's number would show one case with two different similarities.
        best_scores: dict[str, float] = {}
        best_summaries: dict[str, str] = {}
        for entry in self.memory_used:
            for index, (memory_id, text) in enumerate(
                zip(entry.memory_ids, entry.summaries)
            ):
                score = entry.scores[index] if index < len(entry.scores) else 0.0
                if score > best_scores.get(memory_id, float("-inf")):
                    best_scores[memory_id] = score
                    best_summaries[memory_id] = text

        # A case counts as confirmed only if it was retained with the confirmed
        # stage tag, never because the text happens to say so. Hindsight's
        # recall does not return the context field, so the stage is read back
        # from the bank instead of being assumed absent.
        stage_by_id: dict[str, str] = {}
        try:
            stage_by_id = self.memory.stages_by_memory_id(best_scores)
        except Exception as exc:  # noqa: BLE001
            self.emit("memory_stage_unknown", f"could not read retention stages: {exc}")

        similar_cases: list[SimilarPastCase] = []
        for memory_id, score in sorted(
            best_scores.items(), key=lambda item: item[1], reverse=True
        ):
            stage = stage_by_id.get(memory_id) or recalled_contexts.get(memory_id) or ""
            similar_cases.append(
                SimilarPastCase(
                    case_id=memory_id,
                    similarity=round(score, 3),
                    outcome=best_summaries[memory_id][:200],
                    confirmed=STAGE_CONFIRMED in stage,
                )
            )
            if len(similar_cases) >= 6:
                break

        actions: list[RecommendedAction] = []
        if parsed and parsed.get("recommended_actions"):
            for raw in parsed["recommended_actions"]:
                actions.append(
                    RecommendedAction(
                        action=str(raw.get("action", "")),
                        owner_hint=str(raw.get("owner_hint", "")),
                        priority=raw.get("priority", "P1"),
                        expected_effect=str(raw.get("expected_effect", "")),
                    )
                )

        status = str((parsed or {}).get("status", "partial"))
        if status not in {
            "root_cause_found",
            "partial",
            "cannot_determine",
            "data_quality_issue",
        }:
            status = "partial"
        if not root_causes and status == "root_cause_found":
            status = "cannot_determine"
        if parsed is None and root_causes:
            # The reporter never returned a status, but the run did settle a
            # root cause above the confidence threshold. Deriving the status
            # from the model's wording alone is what let a report claim
            # 'partial' while listing a 0.85 root cause.
            status = "root_cause_found"
            self.emit(
                "status_derived",
                "Reporter produced no status; derived root_cause_found from the "
                f"computed confidence of {root_causes[0].confidence}",
                status=status,
            )
        elif parsed is None:
            # Naming the broken pipeline needs more confidence than classifying
            # the incident. When every prediction behind a data-quality reading
            # was observed, "this is a measurement failure, not a business one"
            # is itself a usable answer, and reporting it as a bare 'partial'
            # would throw away the one thing the run actually established.
            top = self._top_data_quality()
            status = "data_quality_issue" if top else "partial"
            self.emit(
                "status_derived",
                (
                    f"Reporter produced no status and no root cause cleared the threshold, but "
                    f"{top.hypothesis_id} classified the incident as a data quality issue with "
                    f"all {len(top.predicted_evidence)} predictions observed"
                    if top
                    else "Reporter produced no status and no hypothesis was settled; "
                    "reporting partial"
                ),
                status=status,
            )

        report = InvestigationReport(
            trace_id=self.trace_id,
            investigation_id=self.investigation_id,
            alert_id=self.anomaly.alert_id,
            metric=self.anomaly.metric,
            status=status,
            summary=str((parsed or {}).get("summary", "") or self._fallback_summary()),
            impact_abs=impact_abs,
            impact_pct=impact_pct,
            impact_note=self._impact_note(impact_abs, impact_pct),
            root_causes=root_causes,
            ruled_out=ruled_out,
            similar_past_cases=similar_cases,
            memory_used=self.memory_used,
            recommended_actions=actions,
            open_questions=[str(q) for q in (parsed or {}).get("open_questions", [])],
            data_quality_notes=[str(n) for n in (parsed or {}).get("data_quality_notes", [])],
            tools_used=[e.tool for e in self.evidence],
            duration_s=round(time.perf_counter() - self.started, 2),
            tokens_used=self.llm.meter.total_tokens,
            llm_turns=self.llm_turns,
            tool_calls=self.guard.calls_used,
            llm_errors=self.llm.meter.errors,
        )
        return report

    def _supporting_evidence(self, hypothesis: Hypothesis | None) -> list[str]:
        if hypothesis is None:
            return []
        verdict = next(
            (v for v in self.verdicts if v.hypothesis_id == hypothesis.hypothesis_id), None
        )
        if verdict and verdict.evidence_ids:
            return [i for i in verdict.evidence_ids if i in self._valid_ids]
        return sorted(self._valid_ids)[:2]

    def _evidence_driven_hypotheses(self) -> None:
        """Derive candidate causes from tool output with no LLM involvement.

        Used when the hypothesizer call fails (rate limit, truncation). The
        point is that the investigation still returns grounded, cited findings
        rather than an empty report, and every prediction below is something a
        tool actually observed.
        """
        if not self.evidence:
            self.emit("hypotheses_empty", "No evidence to derive hypotheses from")
            return

        # dimension -> (segment value, share of decline, evidence id)
        top_segments: dict[str, tuple[str, float, str]] = {}
        for item in self.evidence:
            if item.tool != "breakdown_by_dimension":
                continue
            top = item.numbers.get("top") or {}
            dimension = str(item.numbers.get("dimension") or "")
            segment = top.get("segment")
            if not dimension or not segment:
                continue
            top_segments[dimension] = (
                str(segment),
                float(top.get("share_of_decline_pct") or 0.0),
                item.evidence_id,
            )

        related: dict[str, float] = {}
        traffic_flat = False
        for item in self.evidence:
            if item.tool != "find_related_metrics":
                continue
            traffic_flat = bool(item.numbers.get("traffic_flat"))
            for row in item.numbers.get("related", []):
                if row.get("moved"):
                    related[str(row["metric"])] = float(row.get("delta_pct") or 0.0)

        changepoint = self._changepoint()
        events: list[dict[str, Any]] = []
        for item in self.evidence:
            if item.tool == "query_business_events":
                events.extend(item.numbers.get("events", []))
        release = next(
            (e for e in events if e.get("type") == "release"),
            events[0] if events else None,
        )

        mechanism = next(
            (
                metric
                for metric in ("payment_success_rate", "conversion_rate", "aov")
                if metric in related
            ),
            None,
        )
        volume_healthy = all(
            abs(related.get(metric, 0.0)) < 5.0 for metric in ("sessions", "orders")
        )
        rates_healthy = all(
            abs(related.get(metric, 0.0)) < 5.0
            for metric in ("conversion_rate", "payment_success_rate")
        )
        candidates: list[Hypothesis] = []

        # A revenue or aov move on its own, with volume and every rate healthy,
        # is arithmetically impossible as a demand or checkout fault. Whatever
        # moved is a measurement fault, so read it as one before blaming a cause.
        if volume_healthy and rates_healthy and mechanism == "aov":
            candidates.append(
                Hypothesis(
                    hypothesis_id="H1",
                    statement=(
                        f"{self.anomaly.metric} fell while sessions, orders, conversion and "
                        f"payment success all held flat, leaving aov down "
                        f"{abs(related['aov']):.1f}% as the only residual. Revenue cannot move "
                        f"on its own, so this looks like a recognition or settlement failure "
                        f"rather than a business fault."
                    ),
                    cause_type="data_quality",
                    predicted_evidence=[
                        f"{self.anomaly.metric} moved but orders and sessions did not",
                        "conversion_rate and payment_success_rate did not move",
                        "aov moved as the residual of the metric over orders",
                    ],
                    affected_segments={},
                    source="evidence_derived",
                )
            )
            mechanism = None

        if mechanism and traffic_flat:
            payment_segment = top_segments.get("payment_method")
            segment_note = (
                f" in {payment_segment[0]}" if payment_segment else ""
            )
            trigger = (
                f" after the {release.get('type')} at {release.get('ts')}"
                if release
                else ""
            )
            candidates.append(
                Hypothesis(
                    hypothesis_id="H1",
                    statement=(
                        f"A conversion-quality failure{segment_note}{trigger} reduced "
                        f"{mechanism} by {abs(related[mechanism]):.1f}% while traffic stayed "
                        f"flat, which accounts for the decline in {self.anomaly.metric}."
                    ),
                    cause_type="payment_failure",
                    predicted_evidence=[
                        f"{mechanism} fell materially",
                        "sessions are flat, so this is a rate effect",
                        f"the decline is concentrated in one segment of {self.anomaly.metric}",
                    ],
                    affected_segments=(
                        {"payment_method": payment_segment[0]} if payment_segment else {}
                    ),
                    source="evidence_derived",
                )
            )

        if release is not None and changepoint:
            release_segment = top_segments.get("app_version") or top_segments.get("platform")
            release_dimension = (
                "app_version" if top_segments.get("app_version") else "platform"
            )
            candidates.append(
                Hypothesis(
                    hypothesis_id=f"H{len(candidates) + 1}",
                    statement=(
                        f"A release at {release.get('ts')} ({release.get('description', '')}) "
                        f"coincides with the onset of the change and could have caused it."
                    ),
                    cause_type="release_regression",
                    predicted_evidence=[
                        "a business event immediately precedes the changepoint",
                        "the release coincides with the changepoint",
                    ],
                    affected_segments=(
                        {release_dimension: release_segment[0]} if release_segment else {}
                    ),
                    source="evidence_derived",
                )
            )

        if not traffic_flat and "sessions" in related:
            traffic_segment = next(
                (
                    {dimension: top_segments[dimension][0]}
                    for dimension in ("channel", "platform", "country")
                    if dimension in top_segments
                ),
                {},
            )
            candidates.append(
                Hypothesis(
                    hypothesis_id=f"H{len(candidates) + 1}",
                    statement=(
                        f"Traffic itself fell {abs(related['sessions']):.1f}%, so the decline "
                        f"may be a demand or acquisition problem rather than a conversion problem."
                    ),
                    cause_type="traffic_loss",
                    predicted_evidence=[
                        "sessions moved with the metric",
                        f"the decline is concentrated in {', '.join(traffic_segment) or 'no single segment'}",
                    ],
                    affected_segments=traffic_segment,
                    source="evidence_derived",
                )
            )

        # No release event, but a real conversion-quality failure. A deploy
        # cannot explain it, and the release log is the only place a
        # configuration change would show up if anyone looked for one.
        events_tool = next(
            (i for i in self.evidence if i.tool == "query_business_events"), None
        )
        no_events = bool(
            events_tool is not None and not (events_tool.numbers.get("events") or [])
        )
        if no_events and mechanism and traffic_flat and related.get("payment_success_rate"):
            config_evidence = self._call_tool(
                "query_config_changes", self._default_args("query_config_changes") or {}, "hypothesis"
            )
            changes = list(config_evidence.numbers.get("changes") or []) if config_evidence else []
            # One hypothesis per change, newest first. Collapsing them into a
            # single "some config changed" candidate threw away the only thing
            # that distinguishes them: which key it was. Two changes in the
            # window are equally consistent with the metrics, so the evidence
            # alone cannot separate them and recency is the tie-break an
            # evidence-only run falls back on -- which here picks the wrong one.
            for change in sorted(changes, key=lambda row: str(row.get("ts", "")), reverse=True):
                key = str(change.get("config_key", "unknown key"))
                candidates.append(
                    Hypothesis(
                        hypothesis_id=f"H{len(candidates) + 1}",
                        statement=(
                            f"payment_success_rate fell "
                            f"{abs(related['payment_success_rate']):.1f}% with flat traffic and "
                            f"no release in the window. Config change {key} went from "
                            f"{change.get('old_value')} to {change.get('new_value')} at "
                            f"{str(change.get('ts', ''))[:16].replace('T', ' ')} on "
                            f"{change.get('service')}, so the behaviour change arrived as a "
                            f"config push rather than a deploy."
                        ),
                        cause_type="config_change",
                        predicted_evidence=[
                            "no release or deploy event precedes the changepoint",
                            "payment_success_rate moved in one segment while traffic held",
                            f"a config change to {key} precedes the changepoint",
                        ],
                        affected_segments=(
                            {
                                **(
                                    {"payment_method": top_segments["payment_method"][0]}
                                    if "payment_method" in top_segments
                                    else {}
                                ),
                                "config_key": key,
                            }
                        ),
                        memory_prior_ids=[],
                        source="evidence_derived",
                    )
                )

        if not candidates:
            candidates.append(
                Hypothesis(
                    hypothesis_id="H1",
                    statement=(
                        f"{self.anomaly.metric} moved {self.anomaly.deviation_pct:+.1f}% but the "
                        "evidence gathered did not isolate a segment or mechanism."
                    ),
                    cause_type="unknown",
                    predicted_evidence=[],
                    affected_segments={},
                    source="evidence_derived",
                )
            )

        for hypothesis in candidates:
            self.hypotheses.append(hypothesis)
            self.emit(
                "hypothesis",
                f"{hypothesis.hypothesis_id}: {hypothesis.statement}",
                hypothesis_id=hypothesis.hypothesis_id,
                cause_type=hypothesis.cause_type,
                source=hypothesis.source,
            )

        self._attach_memory_priors(candidates)

    def _attach_memory_priors(self, candidates: list[Hypothesis]) -> None:
        """Give each evidence-derived hypothesis the recalled cases that match it.

        Without this the deterministic path is completely memory-blind: it emits
        candidates, none of them carry a prior, and the memory weight is zero
        however relevant the recall was. Attaching priors keeps the ranking
        evidence-led while letting a prior case that describes this exact
        failure mode lift the matching candidate.
        """
        if not self._memory_available() or not candidates:
            return

        memories: list[Any] = []
        for stage_memories in self.recalled.values():
            memories.extend(stage_memories)
        if not memories:
            return

        for hypothesis in candidates:
            matched = confidence_mod.matching_memory_ids(hypothesis, memories)
            if not matched:
                continue
            hypothesis.memory_prior_ids = matched
            hypothesis.memory_prior_relevance = confidence_mod.mean_relevance(
                hypothesis, memories
            )
            self.emit(
                "memory_attached",
                f"{hypothesis.hypothesis_id} ({hypothesis.cause_type}) carries "
                f"{len(matched)} recalled prior case(s) at mean relevance "
                f"{hypothesis.memory_prior_relevance:.2f}",
                hypothesis_id=hypothesis.hypothesis_id,
                cause_type=hypothesis.cause_type,
                memory_ids=matched,
                relevance=hypothesis.memory_prior_relevance,
            )

    def _fallback_summary(self) -> str:
        """Assemble a summary from the evidence when the reporter fails.

        A reporter that dies on a rate limit must not erase the investigation:
        the tool evidence is still valid and the user deserves a real answer
        with its confidence, clearly marked as machine-assembled.
        """
        if not self.evidence:
            return (
                f"{self.anomaly.metric} moved {self.anomaly.deviation_pct:+.1f}% and no "
                "usable evidence could be gathered."
            )

        parts = [
            f"{self.anomaly.metric} moved {self.anomaly.deviation_pct:+.1f}% over "
            f"{self.anomaly.anomaly_window.start} to {self.anomaly.anomaly_window.end}."
        ]
        top = next(
            (
                item.numbers.get("top")
                for item in self.evidence
                if item.tool == "breakdown_by_dimension" and item.numbers.get("top")
            ),
            None,
        )
        if top and top.get("segment"):
            parts.append(
                f"The decline is concentrated in {top['dimension']}={top['segment']}, "
                f"which accounts for {top.get('share_of_decline_pct', 0):.0f}% of it."
            )
        changepoint = self._changepoint()
        if changepoint:
            events = [
                event
                for item in self.evidence
                if item.tool == "query_business_events"
                for event in item.numbers.get("events", [])
            ]
            if events:
                first = events[0]
                parts.append(
                    f"The change starts at {changepoint}, and the earliest recorded "
                    f"business event is {first.get('ts')} ({first.get('type')})."
                )
        supported = [
            h
            for h in self.hypotheses
            if h.status == "supported" and h.confidence >= self.budget.confidence_threshold
        ]
        if supported:
            best = max(supported, key=lambda h: h.confidence)
            parts.append(
                f"Best supported cause: {best.statement.strip().rstrip('.')}. "
                f"Confidence {best.confidence:.2f}."
            )
        elif self.hypotheses:
            parts.append(
                "No hypothesis reached the confidence threshold of "
                f"{self.budget.confidence_threshold}, so this is reported as partial "
                "rather than a confirmed root cause."
            )
        parts.append("(Assembled from tool evidence because the reporter call failed.)")
        return " ".join(parts)

    def _impact(self) -> tuple[float, float]:
        """Headline impact: the shortfall against the alert's own expectation.

        Preferring a breakdown's gross_decline here was wrong: gross_decline
        sums only the negative segment contributions, so growing segments that
        offset the drop are netted out of one number but not the other, and
        impact_abs (306,067.51) came out inconsistent with impact_pct (-16.7%
        implies 310,062). The shortfall keeps the two figures consistent.
        """
        return (
            round(abs(self.anomaly.expected - self.anomaly.observed), 2),
            self.anomaly.deviation_pct,
        )

    def _impact_note(self, impact_abs: float, impact_pct: float) -> str:
        """Build the impact sentence from the computed figures.

        The reporter's version restates the numbers in its own words, which is
        how a $309,500 note ended up next to a $306,067.51 field. Any figure a
        user reads has to come from the same computation as the field beside it.
        """
        direction = "below" if impact_pct < 0 else "above"
        return (
            f"{self.anomaly.metric} came in at {self.anomaly.observed:,.0f}, "
            f"{impact_abs:,.0f} {direction} the {self.anomaly.expected:,.0f} expected for "
            f"{self.anomaly.anomaly_window.start} ({impact_pct:+.1f}%)."
        )

    # ------------------------------------------------------------------ #
    # tools
    # ------------------------------------------------------------------ #
    def _constrain_window(
        self, name: str, args: dict[str, Any], stage: str
    ) -> dict[str, Any]:
        """Keep follow-up queries anchored to the anomaly window.

        A model-chosen window can land entirely outside the incident. The tools
        then correctly report "no segment contributed negatively", and the model
        reads that as evidence against its own hypothesis. Observed in a real
        run: the validator killed the correct cause by querying a quiet week.
        The fix is to snap a non-overlapping window onto the anomaly window and
        say so in the trace, rather than silently accepting a misleading query.
        """
        start = args.get("start")
        end = args.get("end")
        if not isinstance(start, str) or not isinstance(end, str):
            return args

        # Window bounds are dates in the model but strings in tool arguments.
        anomaly_start = str(self.anomaly.anomaly_window.start)
        anomaly_end = str(self.anomaly.anomaly_window.end)
        if start <= anomaly_end and end >= anomaly_start:
            return args

        adjusted = dict(args)
        adjusted["start"] = anomaly_start
        adjusted["end"] = anomaly_end
        self.emit(
            "tool_window_corrected",
            f"{name} asked for {start}..{end}, which does not overlap the anomaly "
            f"window {anomaly_start}..{anomaly_end}; queried the anomaly window instead",
            stage=stage,
            tool=name,
            requested={"start": start, "end": end},
            used={"start": anomaly_start, "end": anomaly_end},
        )
        return adjusted

    def _call_tool(self, name: str, args: dict[str, Any], stage: str) -> Evidence | None:
        args = self._constrain_window(name, args, stage)
        outcome = self.guard.handle(name, json.dumps(args))
        actions = ", ".join(outcome.actions)
        if not outcome.ok:
            error = outcome.tool_error
            self.emit(
                "guard",
                f"{name} failed: {error.error if error else 'unknown'}",
                stage=stage,
                tool=name,
                actions=outcome.actions,
                error=error.model_dump() if error else None,
            )
            return None

        evidence = outcome.evidence
        assert evidence is not None
        if evidence.evidence_id in {e.evidence_id for e in self.evidence}:
            self.emit("duplicate_evidence", f"{name} returned an existing id; skipped", tool=name)
            return evidence

        self._valid_ids.add(evidence.evidence_id)
        self.evidence.append(evidence)
        self.emit(
            "evidence",
            evidence.finding,
            stage=stage,
            tool=name,
            evidence_id=evidence.evidence_id,
            actions=outcome.actions,
            confidence_hint=evidence.confidence_hint,
            numbers=evidence.numbers,
            chart=evidence.chart.model_dump() if evidence.chart else None,
        )
        return evidence

    def persist(self, report: InvestigationReport) -> Path:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / f"{self.trace_id}.json"
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return path
