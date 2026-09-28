"""Scenario-aware memory A/B: run every incident twice and diff the outcomes.

The point of the memory layer is that it changes conclusions, so the comparison
has to be per scenario rather than "the two newest report files", which only
works if somebody happened to run the same incident twice in the right order.

Each scenario is run down the deterministic path -- the path rate limits actually
put us on -- with memory on and again with memory off. Memory-off is not a
crippled control: it runs the identical evidence gathering, the identical
hypothesis generator and the identical scorer, and forms the same candidate set.
The only difference is that recalled priors are absent, so anything that moves is
attributable to the prior rather than to a handicapped baseline.

Reported per scenario:
  conclusion   cause type of the top hypothesis, on vs off
  mechanism    the specific identifier blamed, where one is named
  confidence   top score, on vs off
  priors       recalled cases cited by the winner
  evidence     evidence items gathered (should be identical)
  tool calls   guard counter (should be identical -- same plan, same tools)

    python -m tests.compare_memory_ab
    python -m tests.compare_memory_ab --scenarios SCN-002
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import get_settings  # noqa: E402
from data.scenarios import SCENARIOS  # noqa: E402
from eval.harness import run_scenario  # noqa: E402


def compare(scenario_id: str) -> tuple[list[str], bool]:
    """Run one scenario both ways. Returns (problems, change_recorded)."""
    truth = SCENARIOS[scenario_id].truth
    print(f"\n=== {scenario_id}: {SCENARIOS[scenario_id].title}")
    print(f"    planted      : {truth.cause_type} -> {truth.root_cause[:72]}")

    bank = get_settings().bank
    off = run_scenario(scenario_id, bank=bank, memory_enabled=False)
    on = run_scenario(scenario_id, bank=bank, memory_enabled=True)

    problems: list[str] = []
    if not on or on.get("recall_failed"):
        problems.append(
            f"{scenario_id}: recall failed, so this run says nothing about memory"
        )
        return problems, False
    if not off or off.get("cause_type") is None:
        problems.append(f"{scenario_id}: the memory-off control produced no hypothesis")
        return problems, False

    print(
        f"    memory off   : {off['cause_type']} / {off['mechanism']} "
        f"conf={off['confidence']:.3f} priors={off['priors']}"
    )
    print(
        f"    memory on    : {on['cause_type']} / {on['mechanism']} "
        f"conf={on['confidence']:.3f} priors={on['priors']} "
        f"rel={on['prior_relevance']:.2f}"
    )

    # The control must be genuinely capable: same candidates, same evidence,
    # same tool budget. If memory-off gathered less, the diff would be measuring
    # a handicapped baseline rather than the prior.
    for field, label in (
        ("candidates", "candidate hypotheses"),
        ("evidence", "evidence items"),
        ("tool_calls", "tool calls"),
    ):
        print(f"    {field:<12}: off={off[field]} on={on[field]}")
        if off[field] != on[field]:
            problems.append(
                f"{scenario_id}: {label} differ ({off[field]} vs {on[field]}), so the "
                "two arms did not run the same investigation"
            )

    if on["priors"] == 0:
        problems.append(f"{scenario_id}: memory-on cited no prior at all")
    if off["priors"] != 0:
        problems.append(
            f"{scenario_id}: memory-off cited {off['priors']} prior(s), which it cannot "
            "have recalled -- the arms are not separated"
        )

    changed = (
        on["cause_type"] != off["cause_type"] or on["mechanism"] != off["mechanism"]
    )
    lifted = on["confidence"] > off["confidence"]
    correct = on["cause_type"] == truth.cause_type

    if not correct:
        problems.append(
            f"{scenario_id}: memory-on concluded {on['cause_type']}, "
            f"planted cause is {truth.cause_type}"
        )
    if changed:
        verdict = "conclusion" if on["cause_type"] != off["cause_type"] else "mechanism"
        print(
            f"    => memory changed the {verdict}: "
            f"{off['cause_type']}/{off['mechanism']} -> {on['cause_type']}/{on['mechanism']}"
        )
    elif lifted:
        print("    => memory lifted confidence only; the answer was already settled")
    else:
        problems.append(
            f"{scenario_id}: memory changed neither the answer nor the confidence"
        )
    return problems, changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=sorted(SCENARIOS),
        help="scenario ids to compare (default: all)",
    )
    args = parser.parse_args()

    get_settings.cache_clear()
    problems: list[str] = []
    changed: list[str] = []

    for scenario_id in args.scenarios:
        if scenario_id not in SCENARIOS:
            problems.append(f"unknown scenario {scenario_id}")
            continue
        found, did_change = compare(scenario_id)
        problems.extend(found)
        if did_change:
            changed.append(scenario_id)

    print("\n" + "=" * 74)
    if changed:
        print("memory changed the answer for:")
        for scenario_id in changed:
            print(f"  {scenario_id}")
    else:
        print("memory changed no answer -- only confidence moved")
    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nA/B COMPARISON OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
