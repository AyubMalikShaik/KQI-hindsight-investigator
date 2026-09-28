"""Hindsight wrapper.

The rest of the application depends on retain / recall / reflect and nothing else,
so Hindsight SDK details never leak into the agent.

Two things here are load-bearing and easy to get wrong:

1. Two-stage retain reuses the same document_id with update_mode="replace" so the
   confirmed outcome supersedes the agent's unconfirmed guess. Without it recall
   returns both, and a later investigation gets re-anchored on a hypothesis that
   was already disproved.

2. recall passes query_timestamp. Without it Hindsight anchors recency scoring to
   server time, which mis-ranks the backfilled historical cases and makes the
   learning-curve numbers meaningless.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from core.config import HindsightConfig
from core.schemas import Feedback, InvestigationReport

log = logging.getLogger("memory")

STAGE_UNCONFIRMED = "agent_report"
STAGE_CONFIRMED = "human_feedback"


@dataclass
class RecalledMemory:
    memory_id: str
    text: str
    score: float = 0.0
    type: str = ""
    context: str = ""
    occurred_at: str = ""
    document_id: str = ""


def _best_score(result: Any) -> float:
    """Hindsight returns a `scores` model/dict, not a single `score` field.

    Prefer the final relevance score so callers can threshold on something
    meaningful instead of a silent 0.0.
    """
    scores = getattr(result, "scores", None)
    if scores is None:
        legacy = getattr(result, "score", None)
        try:
            return float(legacy) if legacy is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    if not isinstance(scores, dict):
        scores = {
            key: value
            for key, value in {
                name: getattr(scores, name, None) for name in dir(scores)
                if not name.startswith("_")
            }.items()
            if isinstance(value, (int, float))
        }

    for key in ("final", "reranker", "rerank", "relevance", "score", "similarity", "semantic"):
        value = scores.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    numeric = [float(v) for v in scores.values() if isinstance(v, (int, float))]
    return max(numeric) if numeric else 0.0


@dataclass
class RecallResult:
    query: str
    stage: str
    memories: list[RecalledMemory] = field(default_factory=list)
    as_of: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def ids(self) -> list[str]:
        return [m.memory_id for m in self.memories]

    def confirmed_causes(self) -> list[str]:
        causes: list[str] = []
        for memory in self.memories:
            if "confirmed" in memory.text.lower() or "root cause" in memory.text.lower():
                causes.append(memory.text)
        return causes


class MemoryService:
    def __init__(self, config: HindsightConfig, bank: str) -> None:
        self.config = config
        self.bank = bank
        self._client: Any | None = None
        self._unavailable_reason = ""
        if not config.configured:
            self._unavailable_reason = (
                "HINDSIGHT_BASE_URL / HINDSIGHT_API_KEY are not configured"
            )
            log.warning("memory disabled: %s", self._unavailable_reason)
        else:
            try:
                from hindsight_client import Hindsight

                self._client = Hindsight(
                    base_url=config.base_url, api_key=config.api_key
                )
            except Exception as exc:  # noqa: BLE001
                self._unavailable_reason = f"{type(exc).__name__}: {exc}"
                log.warning("memory disabled: %s", self._unavailable_reason)

    @property
    def available(self) -> bool:
        return self._client is not None

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    def ensure_bank(self) -> bool:
        """Create the bank if it is missing. Returns True when it is usable.

        Must not lie: a swallowed exception here shows up much later as an
        opaque 404 on every recall.
        """
        if not self.available:
            return False
        try:
            # hindsight_client exposes create_bank on the client; BanksApi's
            # upsert helper is create_or_update_bank.
            create = getattr(self._client, "create_bank", None)
            if create is not None:
                create(bank_id=self.bank, name="Lumen anomaly investigations")
            else:
                self._client.banks.create_or_update_bank(
                    bank_id=self.bank, name="Lumen anomaly investigations"
                )
            log.info("created Hindsight bank %s", self.bank)
            return True
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}".lower()
            if "already exists" in message or "conflict" in message:
                return True
            log.warning("could not create Hindsight bank %s: %s", self.bank, exc)
            self._unavailable_reason = f"bank {self.bank} unusable: {exc}"
            return False

    # ------------------------------------------------------------------ #
    # retain
    # ------------------------------------------------------------------ #
    def retain_report(
        self,
        report: InvestigationReport,
        investigation_id: str,
        occurred_at: datetime | None = None,
    ) -> bool:
        """Stage 1: store the agent's unconfirmed conclusion immediately."""
        if not self.available:
            return False
        return self._retain(
            content=self._format_report(report),
            context=STAGE_UNCONFIRMED,
            document_id=f"{self.bank}:{investigation_id}",
            occurred_at=occurred_at or datetime.now(timezone.utc),
            tags=[
                "metric:" + report.metric,
                "status:unconfirmed",
                "kind:report",
                # Lets recall drop this report when the same alert comes back
                # around, so an investigation never reads its own answer.
                f"alert:{report.alert_id}",
            ],
            replace=True,
        )

    def retain_feedback(
        self,
        report: InvestigationReport,
        feedback: Feedback,
        investigation_id: str,
        occurred_at: datetime | None = None,
    ) -> bool:
        """Stage 2: supersede stage 1 with the human-confirmed outcome.

        Same document_id plus update_mode=replace, so recall never sees both the
        wrong agent guess and its correction.
        """
        if not self.available:
            return False
        return self._retain(
            content=self._format_feedback(report, feedback),
            context=STAGE_CONFIRMED,
            document_id=f"{self.bank}:{investigation_id}",
            occurred_at=occurred_at or datetime.now(timezone.utc),
            tags=[
                "metric:" + report.metric,
                f"verdict:{feedback.verdict}",
                "kind:closed_case",
                f"alert:{report.alert_id}",
            ]
            + [f"cause:{t}" for t in self._cause_tags(report)],
            replace=True,
        )

    def retain_case(
        self,
        content: str,
        document_id: str,
        occurred_at: datetime,
        tags: Iterable[str] | None = None,
        context: str = "historical_case",
    ) -> bool:
        if not self.available:
            return False
        return self._retain(
            content=content,
            context=context,
            document_id=document_id,
            occurred_at=occurred_at,
            tags=list(tags or ["kind:closed_case"]),
            replace=True,
        )

    def _retain(
        self,
        content: str,
        context: str,
        document_id: str,
        occurred_at: datetime,
        tags: list[str] | None = None,
        replace: bool = False,
    ) -> bool:
        # Hindsight's consolidation splits a document into derived memory units
        # that keep their tags but lose the context field, so recall can never
        # tell an agent guess from a human-confirmed closure. Carrying the
        # stage as a tag is the only channel that survives.
        all_tags = list(tags or [])
        stage_tag = f"stage:{context}"
        if stage_tag not in all_tags:
            all_tags.append(stage_tag)
        try:
            self._client.retain(
                bank_id=self.bank,
                content=content,
                context=context,
                document_id=document_id,
                timestamp=occurred_at,
                tags=all_tags or None,
                update_mode="replace" if replace else None,
                retain_async=False,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("retain failed for %s: %s", document_id, exc)
            return False

    def tags_by_memory_id(self, memory_ids: Iterable[str]) -> dict[str, list[str]]:
        """Map memory id -> its tags, read back from the bank.

        Recall does not return tags, so anything the caller needs to know about
        provenance -- retention stage, or which alert a report belongs to -- has
        to be looked up here. One list_memories call covers a whole recall.
        """
        wanted = {str(i) for i in memory_ids}
        found: dict[str, list[str]] = {}
        if not self.available or not wanted:
            return found
        try:
            response = self._client.list_memories(bank_id=self.bank)
        except Exception as exc:  # noqa: BLE001
            log.warning("list_memories failed; memory tags unknown: %s", exc)
            return found
        items = getattr(response, "memories", None) or getattr(response, "items", None) or []
        for item in items:
            memory_id = str(getattr(item, "id", "") or "")
            if not memory_id or memory_id not in wanted:
                continue
            found[memory_id] = [str(tag) for tag in (getattr(item, "tags", None) or [])]
        return found

    def stages_by_memory_id(self, memory_ids: Iterable[str]) -> dict[str, str]:
        """Map memory id -> retention stage, read from the stage tag.

        Used to decide whether a recalled case was human-confirmed. On failure
        the caller should treat the stages as unknown rather than as unconfirmed.
        """
        stages: dict[str, str] = {}
        for memory_id, tags in self.tags_by_memory_id(memory_ids).items():
            for tag in tags:
                if tag.startswith("stage:"):
                    stages[memory_id] = tag.split(":", 1)[1]
                    break
        return stages

    # ------------------------------------------------------------------ #
    # recall
    # ------------------------------------------------------------------ #
    def recall(
        self,
        query: str,
        stage: str,
        as_of: datetime | None = None,
        max_tokens: int = 2048,
        budget: str = "mid",
        tags: list[str] | None = None,
        exclude_alert: str | None = None,
    ) -> RecallResult:
        if not self.available:
            return RecallResult(query=query, stage=stage, error=self._unavailable_reason)
        try:
            response = self._client.recall(
                bank_id=self.bank,
                query=query,
                max_tokens=max_tokens,
                budget=budget,
                query_timestamp=(as_of or datetime.now(timezone.utc)).isoformat(),
                tags=tags,
                prefer_observations=True,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("recall failed: %s", exc)
            return RecallResult(query=query, stage=stage, error=f"{type(exc).__name__}: {exc}")

        memories: list[RecalledMemory] = []
        for rank, result in enumerate(getattr(response, "results", None) or []):
            memories.append(
                RecalledMemory(
                    memory_id=str(getattr(result, "id", f"m{rank}")),
                    text=str(getattr(result, "text", "")),
                    score=_best_score(result),
                    type=str(getattr(result, "type", "") or ""),
                    context=str(getattr(result, "context", "") or ""),
                    occurred_at=str(getattr(result, "occurred_start", "") or ""),
                    document_id=str(getattr(result, "document_id", "") or ""),
                )
            )

        if exclude_alert and memories:
            own = f"alert:{exclude_alert}"
            by_id = self.tags_by_memory_id(m.memory_id for m in memories)
            kept = [m for m in memories if own not in by_id.get(m.memory_id, [])]
            if len(kept) != len(memories):
                log.info(
                    "recall: dropped %d memor(y/ies) about the alert under "
                    "investigation (%s)",
                    len(memories) - len(kept),
                    exclude_alert,
                )
                memories = kept

        return RecallResult(
            query=query,
            stage=stage,
            memories=memories,
            as_of=(as_of or datetime.now(timezone.utc)).isoformat(),
        )

    # ------------------------------------------------------------------ #
    # reflect
    # ------------------------------------------------------------------ #
    def reflect(
        self,
        query: str,
        budget: str = "mid",
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.available:
            return {"text": "", "structured_output": None, "error": self._unavailable_reason}
        try:
            response = self._client.reflect(
                bank_id=self.bank,
                query=query,
                budget=budget,
                include_facts=True,
                response_schema=schema or None,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("reflect failed: %s", exc)
            return {"text": "", "structured_output": None, "error": f"{type(exc).__name__}: {exc}"}

        based_on = getattr(response, "based_on", None)
        cited: list[str] = []
        if based_on is not None:
            for memory in getattr(based_on, "memories", None) or []:
                cited.append(str(getattr(memory, "text", "")))
        return {
            "text": str(getattr(response, "text", "") or ""),
            "structured_output": getattr(response, "structured_output", None),
            "based_on": cited,
            "error": None,
        }

    # ------------------------------------------------------------------ #
    # formatting
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cause_tags(report: InvestigationReport) -> list[str]:
        tags: list[str] = []
        for cause in report.root_causes:
            tags.append(cause.cause_type)
        return tags[:3]

    @staticmethod
    def _format_report(report: InvestigationReport) -> str:
        lines = [
            f"Metric: {report.metric} ({report.impact_pct:+.1f}%) on {report.alert_id}",
            f"Status: {report.status}",
        ]
        for cause in report.root_causes:
            lines.append(
                f"Agent's leading hypothesis: {cause.statement} "
                f"(confidence {cause.confidence:.2f}, cause type {cause.cause_type})"
            )
        for ruled in report.ruled_out:
            lines.append(f"Ruled out: {ruled.hypothesis}")
        lines.append(f"Summary: {report.summary}")
        return "\n".join(lines)

    @staticmethod
    def _format_feedback(report: InvestigationReport, feedback: Feedback) -> str:
        lines = [
            f"Metric: {report.metric} ({report.impact_pct:+.1f}%) on {report.alert_id}",
            f"Human verdict: {feedback.verdict}",
        ]
        if feedback.confirmed_cause:
            lines.append(f"Root cause (confirmed by analyst): {feedback.confirmed_cause}")
        if feedback.action_taken:
            lines.append(f"Action taken: {feedback.action_taken}")
        if feedback.owner:
            lines.append(f"Owner: {feedback.owner}")
        if feedback.outcome:
            lines.append(f"Outcome: {feedback.outcome}")
        if feedback.time_to_resolve_hours is not None:
            lines.append(f"Time to resolve: {feedback.time_to_resolve_hours}h")
        agent_guess = report.root_causes[0].statement if report.root_causes else "none"
        was_right = "yes" if feedback.confirmed_cause and _similar(
            agent_guess, feedback.confirmed_cause
        ) else "no"
        lines.append(f"Agent's initial top hypothesis: {agent_guess} | Correct? {was_right}")
        if feedback.rejected_causes:
            lines.append(f"Rejected causes: {'; '.join(feedback.rejected_causes)}")
        if feedback.notes:
            lines.append(f"Notes: {feedback.notes}")
        return "\n".join(lines)


def _similar(a: str, b: str) -> bool:
    a_words = {w for w in a.lower().split() if len(w) > 3}
    b_words = {w for w in b.lower().split() if len(w) > 3}
    if not a_words or not b_words:
        return False
    return len(a_words & b_words) / len(a_words | b_words) > 0.15
