"""Print the memory section of a persisted report: prior cases and attribution.

    python -m tests.show_memory [trace_id]
"""

from __future__ import annotations

import glob
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    if len(sys.argv) > 1:
        path = f"artifacts/reports/{sys.argv[1]}.json"
    else:
        path = max(glob.glob("artifacts/reports/*.json"), key=os.path.getmtime)
    report = json.load(open(path, encoding="utf-8"))

    print(f"file:     {path}")
    print(f"status:   {report['status']}")
    causes = report["root_causes"]
    print(f"cause:    {causes[0]['confidence'] if causes else None} "
          f"{causes[0]['cause_type'] if causes else ''}")
    print(f"\nprior cases ({len(report['similar_past_cases'])}):")
    for case in report["similar_past_cases"]:
        print(f"  relevance={case['similarity']:<8} confirmed={case['confirmed']}")
        print(f"    {case['outcome'][:110]}")
    print("\nattribution:")
    for recall in report["memory_used"]:
        print(f"  {recall['recall_stage']}: {len(recall['memory_ids'])} memories, "
              f"helped={recall.get('helped')}")
        print(f"    cited_by={recall.get('cited_by_hypothesis_ids')}")
        print(f"    {recall.get('helped_reason', '')[:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
