"""Explain one hypothesis's confidence breakdown from a persisted trace.

Useful when a score looks wrong: prints the per-signal notes the scorer
recorded rather than just the total.
"""

from __future__ import annotations

import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    trace_id = sys.argv[1] if len(sys.argv) > 1 else None
    if trace_id is None:
        from pathlib import Path

        traces = sorted(Path("artifacts/traces").glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not traces:
            print("no traces found")
            return 1
        trace_id = traces[-1].stem
        print(f"(no trace id given, using newest: {trace_id})")

    print(f"\ntrace {trace_id}")
    for line in open(f"artifacts/traces/{trace_id}.jsonl", encoding="utf-8"):
        event = json.loads(line)
        if event["kind"] == "hypothesis":
            extra = {k: v for k, v in event.items() if k not in ("kind", "message", "ts")}
            print(f"\n{event['message'][:100]}")
            print(f"  {extra}")
        elif event["kind"] == "confidence":
            print(f"\n  {event['message']}")
            for note in event.get("notes", []):
                print(f"      {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
