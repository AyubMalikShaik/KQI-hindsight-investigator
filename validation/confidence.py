"""Confidence engine.

The LLM never states a confidence number. It is computed here from the evidence
so the score is reproducible and auditable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from data.catalog import CATALOG
from core.schemas import (
    ConfidenceBreakdown,
    Evidence,
    Hypothesis,
    HypothesisVerdict,
    InvestigationReport,
)

WEIGHTS = {
    "evidence_support": 0.40,
    "magnitude_explained": 0.25,
    "temporal_alignment": 0.20,
    "memory_prior": 0.15,
    "contradictions": 0.30,
}

KEYWORD_GROUPS: dict[str, tuple[str, ...]] = {
    "payment_failure": ("payment", "upi", "gateway", "psp", "authoris", "authoriz", "timeout", "decline"),
    "release_regression": ("deploy", "release", "rollout", "version", "regression", "shipped", "v4."),
    "traffic_loss": ("traffic", "sessions", "demand", "seo", "campaign", "impression", "click"),
    "pricing_change": ("price", "pricing", "discount", "promo", "threshold", "aov", "basket"),
    "data_quality": (
        "duplicate",
        "late",
        "backfill",
        "null",
        "etl",
        "pipeline",
        "tracking",
        "missing",
        "recognition",
        "settle",
        "revenue",
        "under-report",
    ),
    "config_change": ("config push", "config", "configuration", "flag toggle", "runtime change"),
    "seasonality": ("season", "holiday", "weekend", "festival", "diwali", "payday"),
    "competitor": ("competitor", "rival", "market share", "outprice"),
}

# Vocabulary that appears in nearly every stored incident. Matching on these
# tells you the case is from the same domain, not that it explains this cause,
# so they cannot on their own establish relevance.
GENERIC_TERMS = frozenset(
    {
        "payment",
        "payments",
        "revenue",
        "sessions",
        "traffic",
        "orders",
        "conversion",
        "timeout",
        "config",
        "gateway",
        "checkout",
        "india",
    }
)

# Below this a recalled case is not on-topic enough to cite as a prior at all.
MIN_RELEVANT_PRIOR = 0.3

# Metric names, and the numerator/denominator they are built from. A hypothesis
# always states which metric moved, so these appear in every sibling's
# statement and in nearly every stored case: they are observations, not
# mechanisms. Letting them count meant two hypotheses blaming different config
# keys shared a token, both matched the memory, and the mechanism that was
# supposed to separate them was diluted by the thing they agreed on.
OBSERVED_METRICS: frozenset[str] = frozenset(
    name
    for entry in CATALOG
    for name in (
        entry["metric"],
        entry.get("ratio_numerator", ""),
        entry.get("ratio_denominator", ""),
    )
    if name
)

# A percentage or ratio quoted as evidence ("25.6", "0.941"). Version numbers
# also look like dotted numbers, so they are told apart by shape: a version has
# at least three components (4.12.0) or is written v-prefixed (v4.12), while a
# measurement never does.
_MEASUREMENT = re.compile(r"^\d+\.\d+$")


def _is_measurement(token: str) -> bool:
    if token.startswith("v"):
        return False
    return bool(_MEASUREMENT.match(token))


def _is_negated(text: str, term: str) -> bool:
    """True when a term appears only inside a phrase that denies it.

    Looks for a negation cue shortly before the term ("no release", "without a
    deploy", "never shipped"), which is the common shape in a lesson learned
    about what a symptom is *not* evidence of.
    """
    pattern = re.compile(
        r"\b(?:no|not|never|without|absent|ruled out|no evidence of)\b[^.;]{0,24}"
        + re.escape(term),
    )
    return bool(pattern.search(text))


def identifier_tokens(text: str) -> set[str]:
    """Dotted keys and version numbers appearing in free text.

    Restricted to things that can name a specific mechanism -- a config key, a
    service flag, a version -- and nothing else. Counting prose words as
    identifiers would make every hypothesis overlap every memory ("the", "was",
    "drop") and leave two same-category hypotheses scored identically again.

    Metric names and quoted measurements are excluded for the same reason in
    reverse: they are the observation every sibling shares, so including them
    hands both candidates a match against the same memory and washes out the
    one token that tells them apart.
    """
    lowered = str(text or "").lower()
    found = set(re.findall(r"[a-z0-9]+(?:[._][a-z0-9]+)+", lowered)) | set(
        re.findall(r"\bv?\d+(?:\.\d+)+", lowered)
    )
    return {
        token
        for token in found
        if token not in OBSERVED_METRICS and not _is_measurement(token)
    }


def identifier_segments(text: str) -> set[str]:
    """Word-level parts of dotted keys and versions, e.g. upi, keepalive_ms.

    A hypothesis names payment.upi.keepalive_ms while a recalled case usually
    writes only the tail, so requiring the whole key to repeat would score every
    pair as unrelated and drop us back to one undifferentiated prior.
    """
    segments: set[str] = set()
    for token in identifier_tokens(text):
        segments.update(part for part in re.split(r"[._]+", token) if part)
    return {s for s in segments if len(s) >= 3 and s not in GENERIC_TERMS}


def mechanism_tokens(hypothesis: Any) -> set[str]:
    """Identifiers this hypothesis specifically blames, not its cause category.

    Two hypotheses can share cause_type "config_change" and still blame
    completely different things -- one payment.upi.keepalive_ms, one
    payment.timeout_ms -- and a cause_type-keyed prior hands both the identical
    set. That made the prior a constant across same-category hypotheses, so it
    could never break a tie between them, which is exactly the job a prior has.

    Only identifiers are taken: dotted keys, version numbers and the segment
    values the hypothesis explicitly claims, drawn from its statement. Its
    predictions are excluded because siblings in the same incident predict the
    same metrics, and those shared observations would make two different
    mechanisms look identical again.
    """
    # Only the statement and the declared segments: those say what the
    # hypothesis *blames*. predicted_evidence is deliberately excluded -- it
    # describes what the hypothesis *expects to observe*, and sibling hypotheses
    # in the same incident predict the same metrics. Parsing
    # "payment_success_rate moved" as the dotted key payment_success_rate put a
    # shared observation into both mechanisms sets and re-collapsed them.
    corpus = str(getattr(hypothesis, "statement", "") or "").lower()

    tokens = identifier_tokens(corpus) | identifier_segments(corpus)
    for value in (getattr(hypothesis, "affected_segments", None) or {}).values():
        # Declared segment values are chosen, not prose, so a bare word here
        # ("upi", "paid") is a real identifier and is kept verbatim.
        cleaned = str(value).strip().lower()
        if len(cleaned) >= 3 and cleaned not in GENERIC_TERMS:
            tokens.add(cleaned)

    return {t for t in tokens if len(t) >= 3 and t not in GENERIC_TERMS}


def mechanism_overlap(hypothesis: Any, text: str) -> tuple[float, int]:
    """Fraction of a hypothesis's identifiers named in a memory, and its count.

    Compared against the memory's own identifiers rather than its raw prose,
    so a hypothesis token counts as matched only when the case actually names
    that mechanism or a tail of it ("payment.upi.keepalive_ms" against a case
    that says "keepalive_ms").

    Two traps, both of which collapsed sibling hypotheses back onto each other.
    Matching *any* segment of a dotted key gave payment.upi.conn_pool_max
    credit against a case that only says payment.upi.keepalive_ms, because both
    share the namespace payment.upi -- the namespace is the one part that is
    never the mechanism. Matching bare tokens as substrings let "conn" hit
    "connections", so a token scored against a word it is not.
    """
    tokens = mechanism_tokens(hypothesis)
    if not tokens:
        return 1.0, 0
    named = identifier_tokens(text) | identifier_segments(text)
    haystack = text.lower()
    hits = 0
    for token in tokens:
        tail = token.rsplit(".", 1)[-1]
        if token in named or tail in named:
            hits += 1
            continue
        if re.search(r"(?<![\w.])" + re.escape(token) + r"(?![\w])", haystack):
            hits += 1
    return round(hits / len(tokens), 3), len(tokens)


def matching_hypothesis_relevance(
    hypothesis: Any, memories: list[Any], min_score: float = 0.0
) -> dict[str, float]:
    """Per-hypothesis relevance: cause category, narrowed by the exact mechanism.

    Cause-type relevance decides whether a memory is about this class of
    failure at all; the identifier overlap then decides whether it is about
    *this* failure. The second term only applies when the memory itself names
    some mechanism -- a case that discusses no identifiers cannot be faulted
    for not mentioning this one, so it is left at cause relevance rather than
    penalised for staying general.
    """
    cause_relevance = matching_memory_relevance(hypothesis.cause_type, memories, min_score)
    if not cause_relevance:
        return {}

    hypothesis_identifiers = mechanism_tokens(hypothesis)
    out: dict[str, float] = {}
    for memory in memories:
        memory_id = getattr(memory, "memory_id", None)
        if memory_id not in cause_relevance:
            continue
        text = str(getattr(memory, "text", "") or "").lower()
        if not hypothesis_identifiers or not identifier_tokens(text):
            # Neither side names identifiers, so there is nothing to tell two
            # same-category hypotheses apart; the cause term stands on its own.
            out[memory_id] = cause_relevance[memory_id]
            continue
        overlap, _ = mechanism_overlap(hypothesis, text)
        out[memory_id] = round(cause_relevance[memory_id] * (0.3 + 0.7 * overlap), 3)

    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def matching_memory_relevance(
    cause_type: str, memories: list[Any], min_score: float = 0.0
) -> dict[str, float]:
    """Map recall ids to how strongly each one speaks to a given cause type.

    Returns id -> relevance in [0, 1], most relevant first. A bare substring test
    is not enough: words like "payment" and "revenue" appear in almost every
    stored case, so an on-topic test made nearly every hypothesis cite nearly the
    whole bank. That inflated the prior uniformly and made memory read as a flat
    +0.045 offset on every scenario instead of a signal that discriminates
    between them.

    Relevance rewards distinctive vocabulary for the cause type ("config push",
    "keepalive", "backfill") and discounts a case whose only tie is a generic
    term, so a config regression draws on the config case and not the traffic one.
    """
    keywords = KEYWORD_GROUPS.get(cause_type, ())
    if not keywords:
        return {}

    out: dict[str, float] = {}
    for memory in memories:
        if float(getattr(memory, "score", 0.0) or 0.0) < min_score:
            continue
        text = str(getattr(memory, "text", "") or "").lower()
        memory_id = getattr(memory, "memory_id", None)
        if not text or not memory_id:
            continue

        hits = [k for k in keywords if k in text]
        if not hits:
            continue

        # A negated mention points the other way. "checkout faults without
        # release events" argues against a release regression, so counting the
        # bare word "release" as support would let the one case that most
        # strongly refutes a deploy hypothesis act as its prior.
        negated = {k for k in hits if _is_negated(text, k)}
        distinctive = [
            k for k in hits if k not in GENERIC_TERMS and k not in negated
        ]
        if not distinctive:
            if negated and not [k for k in hits if k not in GENERIC_TERMS]:
                out[memory_id] = 0.0
            else:
                # Matched only on vocabulary every incident shares, so it says
                # almost nothing about this particular cause type.
                out[memory_id] = round(0.15 * len(hits) / len(keywords), 3)
            continue

        # One distinctive term is already meaningful evidence that this case is
        # about this failure mode; additional terms add confidence but saturate,
        # so a case repeating itself cannot outrank an independent match.
        relevance = 0.45 + 0.55 * (1.0 - 0.55 ** len(distinctive))
        out[memory_id] = round(min(1.0, relevance), 3)

    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def matching_memory_ids(hypothesis: Any, memories: list[Any], min_score: float = 0.0) -> list[str]:
    """Recall ids worth citing as priors for one hypothesis, most relevant first.

    Takes the hypothesis rather than its cause_type: two candidates can share a
    cause and still blame different mechanisms, and keying on the category gave
    them the identical prior, which is the one thing a prior must not do.
    Only genuinely on-topic cases are returned, so the count means something.
    """
    return [
        memory_id
        for memory_id, relevance in matching_hypothesis_relevance(
            hypothesis, memories, min_score
        ).items()
        if relevance >= MIN_RELEVANT_PRIOR
    ]


def mean_relevance(hypothesis: Any, memories: list[Any], min_score: float = 0.0) -> float:
    """Mean relevance of the priors a hypothesis would cite, or 0.0 if none."""
    keep = [
        relevance
        for relevance in matching_hypothesis_relevance(hypothesis, memories, min_score).values()
        if relevance >= MIN_RELEVANT_PRIOR
    ]
    if not keep:
        return 0.0
    return round(sum(keep) / len(keep), 3)


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9.]+", text.lower()) if len(w) > 2}


def evidence_support(
    hypothesis: Hypothesis, evidence: list[Evidence], verdicts: list[HypothesisVerdict]
) -> tuple[float, str]:
    """Fraction of the hypothesis's predicted evidence actually observed."""
    if not hypothesis.predicted_evidence:
        return 0.0, "hypothesis made no predictions, so nothing supports it"

    corpus = " ".join(f"{e.finding} {e.numbers}" for e in evidence)
    corpus_tokens = _tokens(corpus)
    confirmed_ids: set[str] = set()
    for verdict in verdicts:
        if verdict.hypothesis_id == hypothesis.hypothesis_id and verdict.verdict == "supported":
            confirmed_ids.update(verdict.evidence_ids)

    hits = 0
    # A prediction that asserts an absence ("no release precedes the change")
    # can never score by token overlap, because the tokens it names are exactly
    # the ones the evidence lacks. Left unhandled it read as unsupported, which
    # penalised exactly the hypothesis that had ruled a cause out.
    absence_observed = bool(
        re.search(r"\bno (business )?(events?|releases?|deploys?)\b|\bnone\b", corpus.lower())
    )
    for prediction in hypothesis.predicted_evidence:
        if prediction.strip().lower().startswith("no "):
            if absence_observed:
                hits += 1
            continue
        prediction_tokens = _tokens(prediction)
        if not prediction_tokens:
            continue
        overlap = len(prediction_tokens & corpus_tokens) / len(prediction_tokens)
        if overlap >= 0.35:
            hits += 1

    if confirmed_ids:
        hits = max(hits, 1)

    ratio = hits / len(hypothesis.predicted_evidence)
    return min(1.0, ratio), f"{hits}/{len(hypothesis.predicted_evidence)} predictions observed"


SCORING_DIMENSIONS = (
    "country",
    "platform",
    "payment_method",
    "channel",
    "product_category",
    "app_version",
)

# The warehouse stores canonical codes ("IN", "upi") but a hypothesis states
# what a human would say ("India", "UPI"). Comparing the two verbatim meant a
# correct hypothesis silently lost its magnitude score. Targets are lowercased
# so they match the canonical form of the code itself.
SEGMENT_ALIASES: dict[str, str] = {
    "india": "in",
    "united states": "us",
    "usa": "us",
    "united kingdom": "gb",
    "uk": "gb",
    "singapore": "sg",
    "indonesia": "id",
    "brazil": "br",
    "canada": "ca",
    "mexico": "mx",
    "australia": "au",
    "nigeria": "ng",
    "south africa": "za",
}


def _canonical_segment(value: object) -> str:
    """Normalise a segment value for comparison: trim, lowercase, alias."""
    text = str(value or "").strip().lower()
    return SEGMENT_ALIASES.get(text, text)


def _mention_candidates(value: str) -> set[str]:
    """The segment value plus any names that refer to it ("IN" -> "india")."""
    canonical = _canonical_segment(value)
    names = {canonical}
    for alias, target in SEGMENT_ALIASES.items():
        if target == canonical:
            names.add(alias)
    return names


def _mentions(text: str, value: str) -> bool:
    """Whether prose refers to a segment, without matching inside other words.

    Short codes need word boundaries: "in" appears inside "conversion" and
    "increase", so a plain substring test credits unrelated hypotheses.
    """
    haystack = text.lower()
    for candidate in _mention_candidates(value):
        if len(candidate) <= 3:
            if re.search(rf"\b{re.escape(candidate)}\b", haystack):
                return True
        elif candidate in haystack:
            return True
    return False


def _dominant_segment(evidence: list[Evidence]) -> tuple[str, str, float] | None:
    """The segment the decomposition blames most, as (dimension, value, share)."""
    for item in evidence:
        if item.tool != "breakdown_by_dimension":
            continue
        top = item.numbers.get("top") or {}
        value = top.get("segment")
        if not value:
            continue
        return (
            str(item.numbers.get("dimension") or ""),
            str(value),
            float(top.get("share_of_decline_pct") or 0.0),
        )
    return None


def magnitude_explained(
    hypothesis: Hypothesis, evidence: list[Evidence], report_total_change: float | None
) -> tuple[float, str]:
    """Share of the observed decline attributable to the hypothesis's segment."""
    if not hypothesis.affected_segments:
        # The model often omits affected_segments even when it clearly names the
        # affected segment in prose. Fall back to the decomposition's dominant
        # segment, but only when the hypothesis actually refers to it, so we
        # never credit a cause for a segment it never claimed.
        dominant = _dominant_segment(evidence)
        if dominant and dominant[0]:
            dimension, value, share = dominant
            if _mentions(hypothesis.statement, value):
                # A declining segment reports a negative share; scoring it as
                # signed produced a negative confidence contribution.
                magnitude = abs(share)
                return (
                    min(1.0, magnitude / 100.0 * 1.2),
                    f"hypothesis names {dimension}={value}, which explains "
                    f"{magnitude:.0f}% of the decline",
                )
        return 0.4, "no segment quantification available"

    best_share = 0.0
    best_evidence_id = ""
    claimed = {
        _canonical_segment(value)
        for key, value in hypothesis.affected_segments.items()
        if key in SCORING_DIMENSIONS
    }
    for item in evidence:
        if item.tool != "breakdown_by_dimension":
            continue
        for segment in item.numbers.get("segments") or []:
            if _canonical_segment(segment.get("segment")) not in claimed:
                continue
            # Shares are negative for declines, so compare on magnitude:
            # using the raw sign made max() always pick 0.0 and pinned
            # magnitude_explained at its 0.4 fallback for every declining metric.
            share = abs(float(segment.get("share_of_decline_pct") or 0.0))
            if share > best_share:
                best_share = share
                best_evidence_id = item.evidence_id

    if best_share:
        return (
            min(1.0, best_share / 100.0 * 1.2),
            f"segment explains {best_share:.0f}% of the decline ({best_evidence_id})",
        )
    if claimed:
        return (
            0.4,
            "affected segment "
            + ", ".join(sorted(claimed))
            + " not found in the gathered breakdowns",
        )
    return 0.4, "affected segment not isolated in the evidence"


# Cause types whose mechanism happens outside the platform. A partner stops
# delivering or a campaign ends; nothing is deployed and nothing is configured,
# so there is no internal change to find a timestamp for.
EXTERNAL_CAUSES: tuple[str, ...] = ("traffic_loss", "seasonality", "competitor")


def temporal_alignment(
    hypothesis: Hypothesis, changepoint: str | None, event_times: list[tuple[str, str]]
) -> tuple[float, str]:
    """A cause must precede the change, not follow it.

    ``event_times`` is a list of ``(timestamp, source)`` pairs where source is
    "business" or "config". A config push can only explain a config hypothesis
    and a release can never explain one, so a hypothesis is only credited with
    an event of a kind that could actually produce its cause.
    """
    if not changepoint:
        return 0.5, "changepoint unknown, temporal ordering unverified"

    own_group = KEYWORD_GROUPS.get(hypothesis.cause_type, ())
    groups = (own_group,) if own_group else tuple(KEYWORD_GROUPS.values())
    statement = hypothesis.statement.lower()
    wants_config = hypothesis.cause_type == "config_change"

    for event_time, source in sorted(event_times):
        if event_time[:10] > changepoint:
            continue
        if (source == "config") != wants_config:
            continue
        if any(keyword in statement for group in groups for keyword in group):
            return 1.0, f"candidate cause recorded {event_time} which precedes the changepoint {changepoint}"

    if hypothesis.cause_type in EXTERNAL_CAUSES:
        return 0.6, (
            "no internal change is expected for an external cause, and nothing "
            "recorded contradicts this one"
        )
    return 0.15, "no recorded event matching the cause precedes the changepoint"

def memory_prior(hypothesis: Hypothesis, used: list, verdicts: list[HypothesisVerdict]) -> tuple[float, str]:
    """How much the recalled history should raise confidence, in [0, 1].

    Graded by how relevant the cited priors actually are, not merely by their
    existence. Returning a flat value whenever any prior was attached meant
    every scenario received the same +0.045, which is indistinguishable from
    memory not mattering; a weak match should now lift confidence less than a
    close one, and an unrelated incident should not be lifted at all.

    A prior that is independently corroborated by current evidence is worth
    more, but memory can never reach certainty on its own.
    """
    if not hypothesis.memory_prior_ids:
        return 0.0, "no memory prior"

    relevance = float(getattr(hypothesis, "memory_prior_relevance", 0.0) or 0.0)
    if relevance <= 0.0:
        # Older callers and any path that never ran the relevance pass.
        relevance = 0.5

    surviving = [
        v for v in verdicts if v.hypothesis_id == hypothesis.hypothesis_id and v.verdict == "supported"
    ]
    ceiling = 0.8 if surviving else 0.4
    prior = round(relevance * ceiling, 3)
    corroboration = (
        "independently confirmed on current evidence"
        if surviving
        else "current evidence has not confirmed it yet"
    )
    return prior, (
        f"{len(hypothesis.memory_prior_ids)} recalled case(s) at mean relevance "
        f"{relevance:.2f} ({corroboration})"
    )


def contradictions(hypothesis: Hypothesis, evidence: list[Evidence], verdicts: list[HypothesisVerdict]) -> tuple[float, str]:
    penalty = 0.0
    reasons: list[str] = []

    for verdict in verdicts:
        if verdict.hypothesis_id == hypothesis.hypothesis_id and verdict.verdict == "killed":
            penalty += 0.6
            reasons.append(verdict.reason[:120])

    if hypothesis.cause_type == "traffic_loss":
        for item in evidence:
            if item.tool == "find_related_metrics" and item.numbers.get("traffic_flat"):
                penalty += 0.5
                reasons.append("sessions were statistically flat, contradicting a traffic explanation")
    if hypothesis.cause_type == "pricing_change":
        for item in evidence:
            if item.tool == "breakdown_by_dimension":
                for segment in item.numbers.get("segments") or []:
                    baseline = segment.get("baseline_value")
                    anomaly = segment.get("anomaly_value")
                    if baseline and anomaly and anomaly >= baseline * 0.98:
                        penalty += 0.3
                        reasons.append("basket size did not move, which a pricing change would usually move")

    return min(1.0, penalty), "; ".join(reasons) if reasons else "none found"


def score_hypothesis(
    hypothesis: Hypothesis,
    evidence: list[Evidence],
    verdicts: list[HypothesisVerdict],
    changepoint: str | None,
    event_times: list[tuple[str, str]],
    used_memory: list,
) -> ConfidenceBreakdown:
    support, support_note = evidence_support(hypothesis, evidence, verdicts)
    magnitude, magnitude_note = magnitude_explained(hypothesis, evidence, None)
    temporal, temporal_note = temporal_alignment(hypothesis, changepoint, event_times)
    prior, prior_note = memory_prior(hypothesis, used_memory, verdicts)
    contradiction, contradiction_note = contradictions(hypothesis, evidence, verdicts)

    total = (
        WEIGHTS["evidence_support"] * support
        + WEIGHTS["magnitude_explained"] * magnitude
        + WEIGHTS["temporal_alignment"] * temporal
        + WEIGHTS["memory_prior"] * prior
        - WEIGHTS["contradictions"] * contradiction
    )
    total = max(0.0, min(1.0, total))

    return ConfidenceBreakdown(
        total=round(total, 3),
        evidence_support=round(support, 3),
        temporal_alignment=round(temporal, 3),
        magnitude_explained=round(magnitude, 3),
        memory_prior=round(prior, 3),
        contradictions=round(contradiction, 3),
        weights=dict(WEIGHTS),
        notes=[
            f"evidence support: {support_note}",
            f"magnitude explained: {magnitude_note}",
            f"temporal alignment: {temporal_note}",
            f"memory prior: {prior_note}",
            f"contradictions: {contradiction_note}",
        ],
    )


def build_inferred_kill(
    hypothesis: Hypothesis, scored_confidence: float, threshold: float
) -> Hypothesis:
    """Below threshold, the hypothesis is killed rather than reported as a guess."""
    if hypothesis.status == "killed":
        return hypothesis
    if scored_confidence < threshold * 0.6:
        return hypothesis.model_copy(
            update={
                "status": "killed",
                "kill_reason": f"confidence {scored_confidence:.2f} far below the {threshold:.2f} threshold",
            }
        )
    return hypothesis.model_copy(
        update={"status": "supported" if scored_confidence >= threshold else "open"}
    )


def validate_citations(report: InvestigationReport, valid_ids: set[str]) -> list[str]:
    """Reject any claim that cites nothing, or cites an id no tool produced."""
    problems: list[str] = []
    for cause in report.root_causes:
        if not cause.evidence_ids:
            problems.append(f"root cause '{cause.statement[:60]}' cites no evidence")
        unknown = [i for i in cause.evidence_ids if i not in valid_ids]
        if unknown:
            problems.append(f"root cause '{cause.statement[:40]}' cites unknown ids {unknown}")
    for ruled in report.ruled_out:
        unknown = [i for i in ruled.evidence_ids if i not in valid_ids]
        if unknown:
            problems.append(f"ruled-out '{ruled.hypothesis[:40]}' cites unknown ids {unknown}")
    return problems
