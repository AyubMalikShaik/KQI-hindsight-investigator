"""Reset the Hindsight bank and seed it with genuinely prior incidents.

The bank used to contain the self-test facts for the very incident under
investigation, so recalling "prior cases" just handed the agent its own answer
and the memory-prior score became circular. A believable bank holds closed
cases from other incidents: similar enough to be useful, different enough that
the agent still has to find the cause.

Usage:
    python -m scripts.seed_bank --list
    python -m scripts.seed_bank --reset
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.config import get_settings  # noqa: E402
from memory.memory_service import STAGE_CONFIRMED, MemoryService  # noqa: E402

# Closed cases from *other* incidents, each dated before the incident it informs,
# so recall can only ever supply a lead. None of them is the incident under
# investigation: they are near misses that should sharpen the search without
# supplying the answer.
#
#   SCN-001 (2026-09-26, deploy)   <- case-2026-07-03-checkout-deploy
#   SCN-002 (2026-07-09, config)   <- case-2026-05-20-card-retry-config-push
#                                  <- case-2026-06-15-upi-keepalive-push
#   SCN-003 (2026-08-13, revenue)  <- case-2026-06-30-revenue-recognition-tax-schema
#   SCN-004 (2026-09-05, traffic)  <- case-2026-07-21-paid-channel-misdiagnosed
#
# The two payment cases are deliberately near misses in the same market: the
# memory has to discriminate, not just match keywords. The keepalive case is the
# one that has to discriminate *within* a scenario: SCN-002 now carries two
# competing config changes in the same window, and this case is what tells the
# agent which key to suspect first.
SEED_CASES: list[dict[str, str]] = [
    {
        "document_id": "case-2026-05-12-us-card-decline",
        "occurred_at": "2026-05-12T08:00:00+00:00",
        "tags": ["kind:closed_case", "metric:daily_revenue", "cause:payment_failure"],
        "content": (
            "2026-05-12, Lumen. daily_revenue fell 9% for two days. Root cause: the "
            "US acquiring bank's TLS certificate expired, so card authorisations 5xx'd "
            "in the US only. Sessions were flat. Fixed by rotating the certificate; "
            "monitoring added for gateway 5xx rate by country. Lesson: when a payment "
            "success drop is confined to one market, check certificates and provider "
            "status before blaming the app."
        ),
    },
    {
        "document_id": "case-2026-07-03-checkout-deploy",
        "occurred_at": "2026-07-03T14:00:00+00:00",
        "content": (
            "2026-07-03, Lumen. conversion_rate dropped 11% in IN for about four hours "
            "and recovered on its own. Root cause: a bad checkout-service deploy that "
            "served a stale price list, reverted by the on-call engineer. Lesson: a "
            "regression that self-recovers within hours is usually a deploy; correlate "
            "the onset with release events before investigating data or demand."
        ),
    },
    {
        "document_id": "case-2026-08-19-payment-partner-outage",
        "occurred_at": "2026-08-19T02:00:00+00:00",
        "content": (
            "2026-08-19, Lumen. payment_success_rate dipped 4% across all countries for "
            "roughly 90 minutes. Root cause: a scheduled maintenance window at an "
            "external payment partner, not a Lumen defect. Closed with no code change. "
            "Lesson: broad, shallow, short drops across every market usually indicate an "
            "external dependency, whereas a deep drop in a single market points at that "
            "market's integration."
        ),
    },
    {
        "document_id": "case-2026-04-28-festival-demand",
        "occurred_at": "2026-04-28T18:00:00+00:00",
        "content": (
            "2026-04-28, Lumen. sessions rose 40% and conversion fell 8% with flat "
            "revenue. Root cause: a promotional traffic spike from a partner campaign "
            "brought low-intent sessions. Lesson: when sessions move, the story is "
            "demand, not a defect."
        ),
    },
    {
        "document_id": "case-2026-05-20-card-retry-config-push",
        "occurred_at": "2026-05-20T07:30:00+00:00",
        "tags": ["kind:closed_case", "metric:conversion_rate", "cause:config_change"],
        "content": (
            "2026-05-20, Lumen. conversion_rate fell 14% in IN across card payments for "
            "most of a day, with sessions flat and no release in the change window. "
            "Triage burned two hours looking for a deploy before the release log was "
            "confirmed empty. Root cause: a config push had shortened "
            "payments.card.retry_window_s during a provider migration. Lesson: a "
            "checkout fault with no release event is a config change until proven "
            "otherwise; always read the runtime config change history for the window, "
            "not just the release history. A config push alters behaviour without ever "
            "appearing in a deploy log."
        ),
    },
    {
        "document_id": "case-2026-06-15-upi-keepalive-push",
        "occurred_at": "2026-06-15T09:00:00+00:00",
        "tags": ["kind:closed_case", "metric:payment_success_rate", "cause:config_change"],
        "content": (
            "2026-06-15, Lumen. payment_success_rate fell 11% in IN for UPI while "
            "sessions stayed flat and no release landed in the change window. Root "
            "cause: a config push had lowered payment.upi.keepalive_ms to 8000 during "
            "a routine gateway tuning pass, so idle UPI connections were dropped "
            "before the bank responded. Restoring 45000 fixed it within minutes. "
            "Lesson: a config push is a configuration change rather than a deploy and "
            "never appears in a release log, so when a UPI success rate moves with "
            "flat traffic, read the runtime config change history for the "
            "payment.upi keys and check which key changed, not merely that one did. "
            "Several keys move in the same window on a busy service; only one of them "
            "is load-bearing."
        ),
    },
    {
        "document_id": "case-2026-06-30-revenue-recognition-tax-schema",
        "occurred_at": "2026-06-30T04:00:00+00:00",
        "tags": ["kind:closed_case", "metric:daily_revenue", "cause:data_quality"],
        "content": (
            "2026-06-30, Lumen. daily_revenue fell 15% company-wide while sessions, "
            "orders, conversion and payment success all held flat, so average order "
            "value looked like the culprit. Root cause: a tax-service schema change "
            "began dropping high-value orders from the revenue recognition job; the "
            "business was healthy and the drop was purely a measurement failure. "
            "Lesson: revenue cannot move on its own. If revenue falls while volume and "
            "every rate hold, the movement is in recognition or settlement, so classify "
            "it as a data quality incident rather than chasing a demand or payment "
            "cause that the evidence does not support."
        ),
    },
    {
        "document_id": "case-2026-07-21-paid-channel-misdiagnosed",
        "occurred_at": "2026-07-21T11:00:00+00:00",
        "tags": ["kind:closed_case", "metric:daily_revenue", "cause:traffic_loss"],
        "content": (
            "2026-07-21, Lumen. daily_revenue fell 9% and the first hypothesis was a "
            "payment failure, which sent engineers into the checkout logs. Root cause: "
            "the ads platform had stopped delivering a partner roster, so sessions in "
            "the paid channel fell while conversion, average order value and payment "
            "success were all unchanged. Lesson: a drop confined to one acquisition "
            "channel where every rate stays healthy is a volume effect, not a platform "
            "fault. Do not anchor on a payment or checkout cause when no rate moved; "
            "break the drop down by channel first."
        ),
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list what is in the bank now")
    parser.add_argument("--reset", action="store_true", help="delete the bank and reseed it")
    args = parser.parse_args()

    settings = get_settings()
    memory = MemoryService(settings.hindsight, settings.bank)
    if not memory.available:
        print("Hindsight is not available; check HINDSIGHT_API_KEY in .env")
        return 1

    try:
        if args.list:
            memories = memory._client.list_memories(bank_id=memory.bank)
            items = getattr(memories, "memories", None) or getattr(memories, "items", None) or []
            print(f"bank {memory.bank}: {len(items)} memory/memories")
            for item in items:
                text = getattr(item, "content", None) or getattr(item, "text", "") or ""
                print(f"  - {text[:160]}")
            return 0

        if args.reset:
            try:
                memory._client.delete_bank(bank_id=memory.bank)
                print(f"deleted bank {memory.bank}")
            except Exception as exc:  # noqa: BLE001
                print(f"delete_bank failed ({exc}); continuing to overwrite by document id")

        if not memory.ensure_bank():
            print("could not create the bank")
            return 1

        kept = 0
        for case in SEED_CASES:
            ok = memory.retain_case(
                content=case["content"],
                document_id=case["document_id"],
                occurred_at=datetime.fromisoformat(case["occurred_at"]),
                tags=case.get("tags", ["kind:closed_case"]),
                # Human-confirmed closures, so the report can show these as
                # confirmed history rather than as the agent's own guesses.
                context=STAGE_CONFIRMED,
            )
            kept += int(ok)
            print(f"  {'ok  ' if ok else 'FAIL'} {case['document_id']}")

        print(f"\nseeded {kept}/{len(SEED_CASES)} prior cases into {memory.bank}")
        print("each one is a closed case from a different incident, so a correct")
        print("recall is a lead rather than the answer")

        # Prove recall surfaces them.
        result = memory.recall(
            query="daily_revenue fell and payment success dropped in India; what did we find before?",
            stage="after_scoping",
        )
        print(f"\nrecall returned {len(result.memories)} case(s):")
        for case in result.memories:
            print(f"  score={case.score:.3f} {case.memory_id[:8]} {case.text[:120]}")
        return 0
    finally:
        memory.close()


if __name__ == "__main__":
    raise SystemExit(main())
