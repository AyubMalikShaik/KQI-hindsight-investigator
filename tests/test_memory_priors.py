"""Check that a memory prior tells sibling hypotheses apart.

The regression these guards: priors were keyed on cause_type, so two candidates
sharing a category received an identical prior set and an identical relevance.
A prior that cannot distinguish between the candidates competing for the same
conclusion cannot change which one wins, so memory would move confidence while
the answer stayed put -- and the metric that makes memory look like it is
working is exactly the metric that hides it.
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.schemas import Hypothesis  # noqa: E402
from validation import confidence as C  # noqa: E402


class _Memory:
    def __init__(self, memory_id: str, text: str, score: float = 0.9):
        self.memory_id = memory_id
        self.text = text
        self.score = score


KEEPALIVE = Hypothesis(
    hypothesis_id="H1",
    statement=(
        "A config push to payment.upi.keepalive_ms lowered the UPI keepalive "
        "interval from 30000 to 5000 at 05:35 UTC"
    ),
    cause_type="config_change",
    predicted_evidence=[
        "no release or deploy event precedes the changepoint",
        "payment_success_rate moved in one segment while traffic held",
    ],
    affected_segments={"config_key": "payment.upi.keepalive_ms"},
)

TIMEOUT = Hypothesis(
    hypothesis_id="H2",
    statement=(
        "A config push to payment.timeout_ms raised the gateway read timeout "
        "at 05:35 UTC"
    ),
    cause_type="config_change",
    predicted_evidence=[
        "no release or deploy event precedes the changepoint",
        "payment_success_rate moved in one segment while traffic held",
    ],
    affected_segments={"config_key": "payment.timeout_ms"},
)

NAMES_KEEPALIVE = _Memory(
    "m-keepalive",
    "A configuration push lowered payment.upi.keepalive_ms from 30000 to 5000, "
    "and UPI authorisations began timing out at the gateway.",
)
NAMES_RETRY_WINDOW = _Memory(
    "m-retry",
    "A configuration push shortened payments.card.retry_window_s during a "
    "provider migration, cutting card conversion in India.",
)
NAMES_TIMEOUT = _Memory(
    "m-timeout",
    "A configuration push raised payment.timeout_ms from 8000 to 15000, and "
    "gateway reads began timing out before the response returned.",
)
GENERAL_LESSON = _Memory(
    "m-lesson",
    "Lesson learned: checkout faults without release events should be treated "
    "as configuration changes; always check runtime config history.",
)
TRAFFIC_CASE = _Memory(
    "m-traffic",
    "Lumen daily revenue fell due to a drop in sessions from the paid "
    "acquisition channel while conversion rates remained healthy.",
)


def main() -> int:
    failures: list[str] = []

    # Both name the same cause type, so cause-level matching alone hands them
    # the same answer. That is the bug under test.
    if KEEPALIVE.cause_type != TIMEOUT.cause_type:
        failures.append("fixture drifted: the two hypotheses no longer share a cause type")

    tokens = C.mechanism_tokens(KEEPALIVE)
    if not tokens:
        failures.append("keepalive hypothesis names no identifiers, so it cannot be told apart")
    if tokens & C.mechanism_tokens(TIMEOUT):
        failures.append(
            "the two hypotheses share an identifier, so they are not genuinely "
            f"different mechanisms: {sorted(tokens & C.mechanism_tokens(TIMEOUT))}"
        )

    def relevance(hypothesis: Hypothesis, memory: _Memory) -> float:
        values = list(C.matching_hypothesis_relevance(hypothesis, [memory]).values())
        return values[0] if values else 0.0

    cases = [
        ("names the keepalive key", NAMES_KEEPALIVE, KEEPALIVE, TIMEOUT),
        ("names the timeout key", NAMES_TIMEOUT, TIMEOUT, KEEPALIVE),
        ("names an unrelated key", NAMES_RETRY_WINDOW, None, KEEPALIVE),
    ]
    for label, memory, expected, other_hypothesis in cases:
        keepalive_value = relevance(KEEPALIVE, memory)
        timeout_value = relevance(TIMEOUT, memory)
        print(
            f"  {label}: keepalive={keepalive_value:.3f} timeout={timeout_value:.3f}"
        )
        if expected is KEEPALIVE and not keepalive_value > timeout_value:
            failures.append(
                f"{label}: keepalive scored {keepalive_value:.3f} against "
                f"{timeout_value:.3f}"
            )
        if expected is TIMEOUT and not timeout_value > keepalive_value:
            failures.append(
                f"{label}: timeout scored {timeout_value:.3f} against "
                f"{keepalive_value:.3f}"
            )
        if expected is None:
            # A case about neither mechanism must not be inflated for one of
            # them just because it shares the cause category.
            if keepalive_value != timeout_value:
                failures.append(
                    f"{label}: a case naming neither key scored "
                    f"{keepalive_value:.3f} vs {timeout_value:.3f}, so an "
                    "unrelated mechanism is being preferred over the other"
                )
            if other_hypothesis and keepalive_value >= C.MIN_RELEVANT_PRIOR:
                failures.append(
                    f"{label}: scored {keepalive_value:.3f}, above the cite "
                    "threshold, for a mechanism it never names"
                )

    # A case that names no identifier cannot be faulted for not naming this one,
    # so it must not be penalised into uselessness.
    for hypothesis in (KEEPALIVE, TIMEOUT):
        value = relevance(hypothesis, GENERAL_LESSON)
        if value < C.MIN_RELEVANT_PRIOR:
            failures.append(
                f"{hypothesis.hypothesis_id}: general lesson dropped to {value:.3f}, "
                "below the cite threshold, so an unidentifiable memory is discarded"
            )
        if value != relevance(KEEPALIVE if hypothesis is TIMEOUT else TIMEOUT, GENERAL_LESSON):
            failures.append(
                f"{hypothesis.hypothesis_id}: a case naming no identifier scored "
                "differently against sibling hypotheses"
            )

    # Cause type still gates: a traffic case must not back a config hypothesis.
    for hypothesis in (KEEPALIVE, TIMEOUT):
        value = relevance(hypothesis, TRAFFIC_CASE)
        if value >= C.MIN_RELEVANT_PRIOR:
            failures.append(
                f"{hypothesis.hypothesis_id}: a traffic case scored {value:.3f} as a "
                "prior for a config hypothesis"
            )

    # Citing must be selective rather than whole-bank.
    bank = [NAMES_KEEPALIVE, NAMES_RETRY_WINDOW, GENERAL_LESSON, TRAFFIC_CASE]
    ids = C.matching_memory_ids(KEEPALIVE, bank)
    if len(ids) >= len(bank):
        failures.append(f"all {len(bank)} memories were cited as priors for one hypothesis")
    print(f"  cited for keepalive: {ids}")

    # The number must not silently drop to nothing either.
    if not ids:
        failures.append("the memory describing this exact mechanism was not cited at all")

    mean = C.mean_relevance(KEEPALIVE, bank)
    if not 0.0 < mean <= 1.0:
        failures.append(f"mean relevance out of range: {mean}")
    print(f"  mean relevance: {mean:.3f}")

    if failures:
        print("\nPROBLEMS:")
        for problem in failures:
            print(f"  - {problem}")
        return 1
    print("\nMEMORY PRIORS DISTINGUISH SIBLING HYPOTHESES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
