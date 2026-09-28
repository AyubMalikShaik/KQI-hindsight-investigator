"""File analyst feedback on a report: stage 2 of the retain cycle.

Stage 1 wrote the agent's unconfirmed guess when the run finished. This
supersedes it with what a human actually concluded, so recall never serves both
the wrong guess and its correction for the same investigation -- both stages use
one document id with update_mode=replace.

After the correction lands, reflect() asks Hindsight what the confirmation says
in the context of the whole bank. The synthesis is written next to the report
rather than retained: echoing it back into the bank would hand recall a second
copy of the same case and inflate the next investigation's prior.

    python -m scripts.feedback --list
    python -m scripts.feedback INV-20260926-abc123 --verdict confirmed \
        --cause "UPI SDK 4.12.0 keepalive regression" \
        --action "rolled back to 4.11.1" --owner platform-eng
    python -m scripts.feedback <id> --verdict wrong --cause "config push" \
        --notes "keepalive was tuned on purpose, not a defect"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import get_settings  # noqa: E402
from core.schemas import Feedback, InvestigationReport  # noqa: E402
from memory.memory_service import MemoryService  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"


def find_report(investigation_id: str) -> tuple[Path, InvestigationReport] | None:
    for path in sorted(REPORT_DIR.glob("*.json")):
        try:
            report = InvestigationReport.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception:  # noqa: BLE001
            continue
        if report.investigation_id == investigation_id or report.trace_id == investigation_id:
            return path, report
    return None


def list_reports() -> int:
    reports = sorted(REPORT_DIR.glob("*.json"), key=os.path.getmtime)
    if not reports:
        print("no reports yet")
        return 0
    print(f"{'investigation':<24}{'trace':<18}{'status':<20}metric")
    for path in reports:
        try:
            report = InvestigationReport.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception:  # noqa: BLE001
            continue
        print(
            f"{report.investigation_id or '-':<24}{report.trace_id:<18}"
            f"{report.status:<20}{report.metric}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("investigation_id", nargs="?", help="id from --list")
    parser.add_argument("--list", action="store_true", help="list reports")
    parser.add_argument(
        "--verdict",
        choices=["confirmed", "wrong", "partially_correct"],
        help="was the agent's leading cause right?",
    )
    parser.add_argument("--cause", default="", help="the real root cause, in your words")
    parser.add_argument("--action", default="", help="what was done about it")
    parser.add_argument("--owner", default="", help="who owned the fix")
    parser.add_argument("--outcome", default="", help="how it ended")
    parser.add_argument("--reject", action="append", default=[], dest="rejected")
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--no-reflect", action="store_true", help="skip the post-confirmation synthesis"
    )
    args = parser.parse_args()

    if args.list or not args.investigation_id:
        return list_reports()
    if not args.verdict:
        parser.error("--verdict is required to file feedback")

    found = find_report(args.investigation_id)
    if not found:
        print(f"no report matches {args.investigation_id!r}; try --list")
        return 1
    path, report = found
    print(f"report: {path.name} ({report.investigation_id})")

    settings = get_settings()
    memory = MemoryService(settings.hindsight, settings.bank)
    if not memory.available:
        print(f"Hindsight unavailable: {memory._unavailable_reason}")
        return 1
    if not memory.ensure_bank():
        print("could not use the bank")
        return 1

    feedback = Feedback(
        investigation_id=report.investigation_id,
        verdict=args.verdict,
        confirmed_cause=args.cause,
        rejected_causes=args.rejected,
        action_taken=args.action,
        owner=args.owner,
        outcome=args.outcome,
        notes=args.notes,
    )

    try:
        ok = memory.retain_feedback(report, feedback, report.investigation_id)
        print(f"retained (confirmed): {'yes' if ok else 'no'}")
        if not ok:
            return 1

        top = report.root_causes[0].statement if report.root_causes else "no hypothesis"
        print(
            f"stage transition: agent_report -> human_feedback "
            f"(verdict={feedback.verdict})"
        )

        if args.no_reflect:
            return 0

        result = memory.reflect(
            query=(
                f"After a human confirmed this incident as '{feedback.verdict}', "
                f"what should the next investigator know before opening a similar "
                f"case on {report.metric}? Agent said: {top}. "
                f"Confirmed cause: {feedback.confirmed_cause or feedback.verdict}. "
                f"What transfers across incidents and what was specific to this one?"
            ),
            budget="mid",
        )
        if result.get("error"):
            print(f"reflect skipped: {result['error']}")
            return 0

        text = str(result.get("text") or "").strip()
        out = path.with_suffix(".reflection.txt")
        cited = result.get("based_on") or []
        out.write_text(
            f"investigation: {report.investigation_id}\n"
            f"verdict: {feedback.verdict}\n"
            f"cited memories: {len(cited)}\n\n{text}\n",
            encoding="utf-8",
        )
        print(f"reflection written: {out.name} ({len(text)} chars, {len(cited)} cited)")
        for line in (text.splitlines() or [""])[0:1]:
            print(f"  {line[:200]}")
        return 0
    finally:
        memory.close()


if __name__ == "__main__":
    raise SystemExit(main())
