"""Learning curve: does more history make the agent better?

A memory layer is only worth having if the answer improves as the bank fills.
This walks the bank size from empty to full, running every scenario at each
size on the deterministic path, and reports accuracy, confidence and how much
of the prior actually landed.

History accumulates in date order, because that is the only way it ever
accumulates in production: nobody gets to choose that the case they need
arrives first. So the curve is a real prediction of what an operator sees as
their incident log grows, not a cherry-picked ordering.

Sizes 0 and full are the meaningful endpoints:
  0     the control -- memory on but nothing to remember
  full  what the seeded bank gives us

Everything between them shows whether the improvement is gradual (a real
learning signal) or a single lucky hit at one particular size (noise).

    python -m eval.learning_curve
    python -m eval.learning_curve --sizes 0 4 8
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import get_settings  # noqa: E402
from data.scenarios import SCENARIOS  # noqa: E402
from eval.harness import run_scenario  # noqa: E402
from memory.memory_service import MemoryService  # noqa: E402
from scripts.seed_bank import SEED_CASES  # noqa: E402

# A size has to be able to hold every case we will ever ask it for, so the
# default ladder tops out at the whole bank.
DEFAULT_SIZES = (0, 2, 4, 6, 8)


def chronological() -> list[dict[str, str]]:
    """Seed cases in the order they were filed."""
    return sorted(SEED_CASES, key=lambda case: case["occurred_at"])


def seed_bank(bank: str, cases: list[dict[str, str]]) -> int:
    """Put `cases` into a bank of their own. Returns how many landed."""
    settings = get_settings()
    memory = MemoryService(settings.hindsight, bank)
    if not memory.available:
        print("Hindsight is not available; check HINDSIGHT_API_KEY in .env")
        return -1
    kept = 0
    try:
        if not memory.ensure_bank():
            print(f"could not create bank {bank}")
            return -1
        for case in cases:
            kept += int(
                memory.retain_case(
                    content=case["content"],
                    document_id=case["document_id"],
                    occurred_at=datetime.fromisoformat(case["occurred_at"]),
                    tags=case.get("tags", ["kind:closed_case"]),
                    context="agent_report",
                )
            )
    finally:
        memory.close()
    return kept


def drop_bank(bank: str) -> None:
    settings = get_settings()
    memory = MemoryService(settings.hindsight, bank)
    try:
        if memory.available:
            memory._client.delete_bank(bank_id=bank)
    except Exception as exc:  # noqa: BLE001
        print(f"  could not delete {bank}: {exc}")
    finally:
        memory.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        nargs="*",
        type=int,
        default=list(DEFAULT_SIZES),
        help="bank sizes to measure (default: %(default)s)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="leave the curve banks behind instead of deleting them",
    )
    args = parser.parse_args()

    cases = chronological()
    sizes = sorted(set(args.sizes))
    if any(size < 0 or size > len(cases) for size in sizes):
        print(f"sizes must be within 0..{len(cases)}")
        return 1

    get_settings.cache_clear()
    scenario_ids = sorted(SCENARIOS)
    results: dict[tuple[int, str], dict[str, Any]] = {}
    seeded: dict[int, bool] = {}

    for size in sizes:
        bank = f"lumen-curve-{size}"
        print(f"\n--- bank size {size} ({bank}) ---")
        drop_bank(bank)
        # Size 0 still needs the bank to exist: recall against a missing bank is
        # a 404, which is a different failure from "nothing worth remembering".
        kept = seed_bank(bank, cases[:size])
        seeded[size] = kept == size
        if size:
            print(f"  seeded {kept}/{size} case(s)")

        for scenario_id in scenario_ids:
            outcome = run_scenario(scenario_id, bank=bank, memory_enabled=True)
            truth = SCENARIOS[scenario_id].truth
            if outcome.get("recall_failed"):
                print(f"  {scenario_id}: recall failed")
                results[(size, scenario_id)] = {"error": "recall_failed"}
                continue
            results[(size, scenario_id)] = {
                "correct": outcome["cause_type"] == truth.cause_type,
                "cause_type": outcome["cause_type"],
                "mechanism": outcome["mechanism"],
                "confidence": outcome["confidence"],
                "priors": outcome["priors"],
                "prior_relevance": outcome["prior_relevance"],
                "tool_calls": outcome["tool_calls"],
                "evidence": outcome["evidence"],
            }

        if size != 0 and not args.keep:
            drop_bank(bank)

    # ----------------------------------------------------------------- #
    # report
    # ----------------------------------------------------------------- #
    # Written before the verdict so a failed run still leaves the numbers the
    # chart needs; the UI reads this rather than recomputing the curve.
    curve_path = PROJECT_ROOT / "artifacts" / "learning_curve.json"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve_path.write_text(
        json.dumps(
            {
                "sizes": sizes,
                "scenarios": scenario_ids,
                "cases": [
                    {"document_id": c["document_id"], "occurred_at": c["occurred_at"]}
                    for c in cases
                ],
                "results": [
                    {
                        "size": size,
                        "scenario": scenario_id,
                        **row,
                    }
                    for (size, scenario_id), row in sorted(results.items())
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\ncurve data written: {curve_path.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 78)
    print("accuracy and confidence as the bank fills")
    print("=" * 78)
    header = f"{'size':>4} | " + " | ".join(f"{sid}" for sid in scenario_ids)
    print(header)
    print("-" * len(header))
    for size in sizes:
        cells = []
        for scenario_id in scenario_ids:
            row = results.get((size, scenario_id), {})
            if "error" in row:
                cells.append(f"{'err':>14}")
            else:
                mark = "ok" if row["correct"] else "X "
                cells.append(f"{mark} {row['confidence']:.2f} pri={row['priors']}")
        print(f"{size:>4} | " + " | ".join(f"{c:<14}" for c in cells))

    print("\nprior detail (confidence / relevance / cases cited):")
    for size in sizes:
        for scenario_id in scenario_ids:
            row = results.get((size, scenario_id), {})
            if not row or "error" in row:
                continue
            print(
                f"  size={size:>2} {scenario_id}: conf={row['confidence']:.3f} "
                f"rel={row['prior_relevance']:.2f} priors={row['priors']} "
                f"cause={row['cause_type']}"
            )

    # ----------------------------------------------------------------- #
    # verdicts
    # ----------------------------------------------------------------- #
    problems: list[str] = []
    for size, ok in sorted(seeded.items()):
        if not ok:
            problems.append(f"bank size {size} did not seed completely")

    errors = [
        f"{sid} at size {size}"
        for (size, sid), row in sorted(results.items())
        if "error" in row
    ]
    if errors:
        problems.append("recall failed: " + ", ".join(errors))

    baseline = sizes[0]
    final = sizes[-1]
    improved: list[str] = []
    regressed: list[str] = []
    for scenario_id in scenario_ids:
        start = results.get((baseline, scenario_id), {})
        end = results.get((final, scenario_id), {})
        if not start or not end or "error" in start or "error" in end:
            continue
        if not end["correct"]:
            problems.append(
                f"{scenario_id}: still wrong at size {final}, so the bank did not help"
            )
        elif not start["correct"]:
            # The interesting case: an empty bank gets it wrong and history
            # gets it right. That is what the curve is for.
            improved.append(scenario_id)
        if start.get("correct") and not end["correct"]:
            regressed.append(scenario_id)
        if end["confidence"] < start["confidence"] and start["correct"]:
            problems.append(
                f"{scenario_id}: confidence fell as history grew "
                f"({start['confidence']:.3f} -> {end['confidence']:.3f})"
            )

    if regressed:
        problems.append(
            "answer was correct with no history and wrong with full history: "
            + ", ".join(regressed)
        )

    if improved:
        print(
            "\ncorrect only once history accumulated (size "
            f"{baseline} -> {final}):"
        )
        for scenario_id in improved:
            print(f"  {scenario_id}")
    elif final > baseline:
        print(
            f"\nnote: every answer was already correct at size {baseline}, so this "
            "curve measures confidence, not accuracy"
        )

    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nLEARNING CURVE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
