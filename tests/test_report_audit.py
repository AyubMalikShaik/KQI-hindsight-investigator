"""Audit the most recent persisted report against the architecture invariants.

Verifies the report, not the console: computed confidence, citation validity,
deduplication, real similarity scores, and memory attribution.
"""

from __future__ import annotations

import glob
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    # A partial or data-quality report legitimately has no root cause, so
    # requiring citations unconditionally failed runs that were correct to
    # stop short of naming one.
    requested = [a for a in sys.argv[1:] if not a.startswith("-")]
    if requested:
        candidates = [f"artifacts/reports/{a}.json" for a in requested]
    else:
        candidates = glob.glob("artifacts/reports/*.json")
    if not candidates:
        print("no reports found; run an investigation first")
        return 1
    path = max(candidates, key=os.path.getmtime)
    report = json.load(open(path, encoding="utf-8"))
    problems: list[str] = []

    print(f"file:            {path}")
    print(f"status:          {report['status']}")
    print(
        f"root_causes:     {len(report['root_causes'])}   "
        f"ruled_out: {len(report['ruled_out'])}   "
        f"similar_cases: {len(report['similar_past_cases'])}"
    )
    print(f"impact:          {report['impact_abs']} / {report['impact_pct']}%")
    print(f"impact_note:     {report['impact_note'][:90]}")
    print(f"llm_turns:       {report.get('llm_turns')}  tool_calls: {report.get('tool_calls')}")
    print()

    cited = [i for rc in report["root_causes"] for i in rc.get("evidence_ids", [])]
    print(f"cited evidence:  {len(cited)} ({len(set(cited))} unique)")
    if len(cited) != len(set(cited)):
        problems.append("a root cause cites the same evidence id twice")
    if report["root_causes"] and not cited:
        problems.append("root causes carry no evidence ids")
    if not report["root_causes"] and report["status"] == "root_cause_found":
        problems.append("status claims a root cause but none is listed")
    if not report["root_causes"] and report["status"] not in {
        "partial",
        "cannot_determine",
        "data_quality_issue",
    }:
        problems.append(f"status {report['status']} has no root cause to justify it")

    for cause in report["root_causes"]:
        print(f"  {cause['hypothesis_id']} conf={cause['confidence']} {cause['cause_type']}")
    print()

    keys = [ro.get("hypothesis") for ro in report["ruled_out"]]
    if len(keys) != len(set(keys)):
        problems.append(f"duplicate ruled_out entries: {keys}")
    for ro in report["ruled_out"]:
        print(f"  ruled_out: {str(ro.get('hypothesis'))[:70]}")
        print(f"             why: {str(ro.get('why'))[:70]}")
        print(f"             cited: {ro.get('evidence_ids')}")
    print()

    recall_scores: dict[str, float] = {}
    for recall in report["memory_used"]:
        scores = recall.get("scores") or []
        print(
            f"  memory {recall['recall_stage']}: {len(recall['memory_ids'])} ids, "
            f"scores={[round(s, 3) for s in scores]}"
        )
        if len(scores) != len(recall["memory_ids"]):
            problems.append(
                f"memory recall '{recall['recall_stage']}' has {len(scores)} scores "
                f"for {len(recall['memory_ids'])} ids"
            )
        for memory_id, score in zip(recall["memory_ids"], scores):
            recall_scores[memory_id] = max(recall_scores.get(memory_id, float("-inf")), score)
    print()

    for case in report["similar_past_cases"]:
        case_id = case.get("case_id")
        sim = case.get("similarity")
        print(f"  similar {str(case_id)[:8]} relevance={sim} confirmed={case.get('confirmed')}")
        if sim is None or sim < 0:
            problems.append(f"case {case_id} has a missing or negative score: {sim!r}")
        # The same case must not appear with two different similarity values.
        expected = recall_scores.get(case_id)
        if expected is not None and abs(expected - sim) > 0.001:
            problems.append(
                f"case {case_id} reports similarity {sim} but memory_used best score "
                f"is {round(expected, 3)}"
            )
    print()

    uncited = report.get("uncited_claims") or []
    if uncited:
        problems.append(f"uncited claims: {uncited}")
    else:
        print("uncited_claims:   none")

    # Internal consistency: the report must not contradict itself. A run that
    # reported 'partial' while listing a root cause above the threshold, and a
    # summary saying nothing cleared the bar, all came from the same bug.
    if report["root_causes"]:
        if report["status"] == "partial":
            problems.append(
                f"status is partial but {len(report['root_causes'])} root cause(s) are listed"
            )
        if report["status"] == "cannot_determine":
            problems.append("status is cannot_determine but root causes are listed")
        summary = (report.get("summary") or "").lower()
        if "no hypothesis reached" in summary:
            problems.append(
                "summary claims no hypothesis reached the threshold, but root "
                "causes are listed"
            )
        if "(assembled from tool evidence" not in summary.lower() and (
            "partial" in summary and "root cause" in summary
        ):
            print("note: summary mentions both 'partial' and 'root cause'")
    else:
        if report["status"] == "root_cause_found":
            problems.append("status is root_cause_found but no root causes are listed")
        summary = (report.get("summary") or "").lower()
        if "best supported cause" in summary:
            problems.append("summary claims a supported cause, but none is listed")

    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nREPORT AUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
