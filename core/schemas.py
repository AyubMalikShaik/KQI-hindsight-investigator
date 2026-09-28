"""Typed contracts shared across every layer.

These are the schemas the LLM is forced to produce and the tools are forced to
return. Nothing downstream is allowed to hand-wave a shape.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# L1: Ingestion
# --------------------------------------------------------------------------- #
class Direction(str, Enum):
    drop = "drop"
    spike = "spike"


class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class Window(BaseModel):
    start: date
    end: date

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1


class Anomaly(BaseModel):
    """Monitoring-tool agnostic alert, normalized to one shape."""

    alert_id: str
    metric: str
    grain: Literal["hour", "day", "week"] = "day"
    detected_at: datetime
    anomaly_window: Window
    baseline_window: Window
    observed: float
    expected: float
    deviation_pct: float
    direction: Direction
    severity: Severity
    source: str = "simulator"

    @field_validator("deviation_pct")
    @classmethod
    def _check_sign(cls, v: float) -> float:
        if not -100.0 <= v <= 1000.0:
            raise ValueError("deviation_pct out of plausible range")
        return v


# --------------------------------------------------------------------------- #
# L4: Tool results
# --------------------------------------------------------------------------- #
class ChartSpec(BaseModel):
    """Declarative chart so Streamlit can render evidence without bespoke code."""

    kind: Literal["line", "bar", "waterfall", "heatmap", "scatter"] = "bar"
    title: str
    x: str | None = None
    y: str | list[str] | None = None
    series: dict[str, list[float]] = Field(default_factory=dict)
    categories: list[str] = Field(default_factory=list)
    annotations: list[dict[str, Any]] = Field(default_factory=list)


class Evidence(BaseModel):
    """What every tool returns. Compact by design - never raw rows."""

    evidence_id: str
    tool: str
    finding: str
    numbers: dict[str, Any] = Field(default_factory=dict)
    confidence_hint: float = 0.5
    chart: ChartSpec | None = None
    data_quality_flags: list[str] = Field(default_factory=list)

    @field_validator("confidence_hint")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class ToolError(BaseModel):
    """Structured errors go back to the model so it can self-correct."""

    error: str
    detail: str = ""
    retryable: bool = True
    hint: str = ""


# --------------------------------------------------------------------------- #
# L3: Agent reasoning
# --------------------------------------------------------------------------- #
class InvestigationPlan(BaseModel):
    objective: str
    steps: list[str] = Field(default_factory=list)
    tools_to_call: list[str] = Field(default_factory=list)
    pruned_checks: list[str] = Field(
        default_factory=list,
        description="Checks skipped because a recalled case already ruled them out.",
    )
    memory_informed: bool = False


class Hypothesis(BaseModel):
    """A falsifiable candidate cause. Predicted evidence is what makes it a test."""

    hypothesis_id: str
    statement: str
    cause_type: Literal[
        "release_regression",
        "payment_failure",
        "traffic_loss",
        "pricing_change",
        "seasonality",
        "data_quality",
        "config_change",
        "competitor",
        "unknown",
    ] = "unknown"
    predicted_evidence: list[str] = Field(default_factory=list)
    affected_segments: dict[str, str] = Field(default_factory=dict)
    memory_prior_ids: list[str] = Field(default_factory=list)
    memory_prior_relevance: float = 0.0
    source: Literal["memory_prior", "fresh", "evidence_derived"] = "fresh"
    confidence: float = 0.0
    status: Literal["open", "supported", "killed", "confirmed"] = "open"
    kill_reason: str = ""


class HypothesisVerdict(BaseModel):
    hypothesis_id: str
    verdict: Literal["supported", "killed", "needs_more_evidence"]
    reason: str
    evidence_ids: list[str] = Field(default_factory=list)
    requested_tools: list[dict[str, Any]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# L7: Validation
# --------------------------------------------------------------------------- #
class ConfidenceBreakdown(BaseModel):
    total: float
    evidence_support: float = 0.0
    temporal_alignment: float = 0.0
    magnitude_explained: float = 0.0
    memory_prior: float = 0.0
    contradictions: float = 0.0
    weights: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ScoredHypothesis(BaseModel):
    hypothesis: Hypothesis
    confidence: ConfidenceBreakdown


# --------------------------------------------------------------------------- #
# L8: Report
# --------------------------------------------------------------------------- #
class RootCause(BaseModel):
    hypothesis_id: str
    statement: str
    cause_type: str
    confidence: float
    evidence_ids: list[str] = Field(
        default_factory=list, description="Non-empty is enforced by ReportValidator."
    )
    contribution_pct: float = 0.0


class RuledOut(BaseModel):
    hypothesis: str
    why: str
    evidence_ids: list[str] = Field(default_factory=list)


class MemoryUsed(BaseModel):
    """Per-recall record. This is the 'Memory used' card on screen."""

    query: str
    recall_stage: Literal["after_scoping", "after_evidence"]
    memory_ids: list[str] = Field(default_factory=list)
    summaries: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    cited_by_hypothesis_ids: list[str] = Field(default_factory=list)
    confirmed_causes: list[str] = Field(default_factory=list)
    helped: bool = False
    helped_reason: str = ""


class SimilarPastCase(BaseModel):
    case_id: str
    # Hindsight's own relevance score, not a probability. It is a hybrid
    # retrieval score and can exceed 1.0 on strong matches, so do not clamp it
    # and do not read it as "percent similar". It is comparable across cases
    # within one report because all of them come from the same scorer.
    similarity: float
    outcome: str
    confirmed: bool = False


class RecommendedAction(BaseModel):
    action: str
    owner_hint: str = ""
    priority: Literal["P0", "P1", "P2"] = "P1"
    expected_effect: str = ""


class InvestigationReport(BaseModel):
    trace_id: str
    # Stage 1 of the retain cycle keys on this, and stage 2 must address the
    # same document. Without it on the report, feedback can only be filed by
    # guessing which investigation it belongs to.
    investigation_id: str = ""
    alert_id: str
    metric: str
    status: Literal["root_cause_found", "partial", "cannot_determine", "data_quality_issue"]
    summary: str
    impact_abs: float = 0.0
    impact_pct: float = 0.0
    impact_note: str = ""
    root_causes: list[RootCause] = Field(default_factory=list)
    ruled_out: list[RuledOut] = Field(default_factory=list)
    similar_past_cases: list[SimilarPastCase] = Field(default_factory=list)
    memory_used: list[MemoryUsed] = Field(default_factory=list)
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    data_quality_notes: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    duration_s: float = 0.0
    tokens_used: int = 0
    llm_turns: int = 0
    tool_calls: int = 0
    llm_errors: int = 0

    def uncited_claims(self) -> list[str]:
        """Hypothesis ids of root causes that no evidence supports.

        Returns bare hypothesis ids so callers can match them against
        RootCause.hypothesis_id directly.
        """
        return [rc.hypothesis_id for rc in self.root_causes if not rc.evidence_ids]


# --------------------------------------------------------------------------- #
# Human feedback
# --------------------------------------------------------------------------- #
class Feedback(BaseModel):
    investigation_id: str
    verdict: Literal["confirmed", "wrong", "partially_correct"]
    confirmed_cause: str = ""
    rejected_causes: list[str] = Field(default_factory=list)
    action_taken: str = ""
    owner: str = ""
    outcome: str = ""
    time_to_resolve_hours: float | None = None
    notes: str = ""


__all__ = [
    "Anomaly",
    "ChartSpec",
    "ConfidenceBreakdown",
    "Direction",
    "Evidence",
    "Feedback",
    "Hypothesis",
    "HypothesisVerdict",
    "InvestigationPlan",
    "InvestigationReport",
    "MemoryUsed",
    "RecommendedAction",
    "RootCause",
    "RuledOut",
    "ScoredHypothesis",
    "Severity",
    "SimilarPastCase",
    "ToolError",
    "Window",
]
