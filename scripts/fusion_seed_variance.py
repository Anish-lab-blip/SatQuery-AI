"""Predeclared seed-variance pass for the Phase 14 fusion-head A/B comparison.

This tool implements ONLY the predeclared variance rule. It answers a single
question: *given the deciding metric across the seeds of one arm, what is the
comparison floor?* It reports that floor and nothing else.

    python scripts/fusion_seed_variance.py \
        --run runs/seed1/run_record.json \
        --run runs/seed2/run_record.json \
        --run runs/seed3/run_record.json \
        --run runs/seed4/run_record.json \
        --run runs/seed5/run_record.json \
        --report variance_report.json

THE RULE (fixed by the owner; not a parameter of this tool)
-----------------------------------------------------------
    * The variance pass uses FIVE independent seeds of arm A (Gate F D-05).
      Five satisfies the preregistration's >= 3 floor.
    * For each run, the DECIDING metric is `best_val_accuracy` (Gate F D-02);
      macro-F1 is co-reported and secondary, and is never read here.
    * spread = max(best_val_accuracy) - min(best_val_accuracy) across the seeds.
    * Gate F D-03: if the spread is EXACTLY 0, the comparison floor is the
      validation-accuracy resolution `1 / N_val` (for N_val = 4000 that is
      0.00025). N_val is read from the run records, never hardcoded.
    * Otherwise the comparison floor is the measured spread itself.

WHAT THIS TOOL MUST NOT DO
--------------------------
This tool NEVER declares a winner, NEVER compares a delta between the two arms
against a threshold, and NEVER picks or proposes a threshold. It reports the
floor only. The A/B decision belongs to the owner and is made by a different,
later step. Any change that makes this file choose between arms is out of scope
by construction.

EXIT CODES
----------
    0   a report was produced (a floor was reported)
    2   fewer than 3 runs were supplied, a record lacks `best_val_accuracy`,
        the records disagree on `val_samples`, or a record is missing/invalid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

#: The preregistration floor: the variance pass needs at least this many runs.
MIN_RUNS = 3

#: Gate F D-05: the predeclared number of independent arm-A seeds.
D05_TARGET_SEEDS = 5

#: The deciding metric key, exactly as the run record writes it (Gate F D-02).
DECIDING_METRIC_KEY = "best_val_accuracy"

#: The run-record key carrying the validation-split size, `N_val`.
VAL_SAMPLES_KEY = "val_samples"

__all__ = [
    "MIN_RUNS",
    "D05_TARGET_SEEDS",
    "DECIDING_METRIC_KEY",
    "compute_floor",
    "read_run_record",
    "build_report",
    "main",
]


def hr(title: str = "") -> None:
    print("-" * 70)
    if title:
        print(title)
        print("-" * 70)


def read_run_record(path: str | Path) -> dict[str, Any]:
    """Load one `run_record.json`. Raises `FusionSeedVarianceError` on failure."""
    p = Path(path)
    if not p.exists():
        raise FusionSeedVarianceError(f"run record does not exist: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FusionSeedVarianceError(
            f"run record {p} could not be parsed as JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise FusionSeedVarianceError(
            f"run record {p} is a {type(data).__name__}, not a JSON object"
        )
    if DECIDING_METRIC_KEY not in data:
        raise FusionSeedVarianceError(
            f"run record {p} has no {DECIDING_METRIC_KEY!r}; the deciding metric "
            f"(Gate F D-02) cannot be read from it"
        )
    return data


class FusionSeedVarianceError(RuntimeError):
    """A variance pass that cannot be computed as predeclared."""


def compute_floor(
    accuracies: Sequence[float], n_val: int
) -> tuple[float, float, float]:
    """Return `(spread, resolution, floor)` for one arm's seed set.

    `spread` is `max - min` across the seeds. `resolution` is `1 / N_val`, the
    validation-accuracy resolution. Gate F D-03: `floor` is `resolution` when
    the spread is EXACTLY 0 (the seeds are indistinguishable at this metric's
    resolution), and the measured `spread` otherwise.
    """
    if not accuracies:
        raise FusionSeedVarianceError("no accuracies were supplied")
    if n_val <= 0:
        raise FusionSeedVarianceError(
            f"N_val must be positive to form a resolution; got {n_val!r}"
        )
    spread = float(max(accuracies) - min(accuracies))
    resolution = 1.0 / float(n_val)
    floor = resolution if spread == 0.0 else spread
    return spread, resolution, floor


def build_report(runs: Sequence[dict[str, Any]], *, arm: str = "A") -> dict[str, Any]:
    """Assemble the JSON report from the run records. Reports the floor only."""
    if len(runs) < MIN_RUNS:
        raise FusionSeedVarianceError(
            f"{len(runs)} run(s) supplied; the predeclared variance pass needs at "
            f"least {MIN_RUNS} (Gate F D-05 uses {D05_TARGET_SEEDS})"
        )

    accuracies: list[float] = []
    seeds: list[Any] = []
    per_seed: list[dict[str, Any]] = []
    n_vals: set[int] = set()
    for run in runs:
        acc = float(run[DECIDING_METRIC_KEY])
        n_val = int(run[VAL_SAMPLES_KEY])
        seed = run.get("seed")
        accuracies.append(acc)
        seeds.append(seed)
        n_vals.add(n_val)
        per_seed.append(
            {
                "seed": seed,
                DECIDING_METRIC_KEY: acc,
                VAL_SAMPLES_KEY: n_val,
                "arm": run.get("arm"),
            }
        )

    if len(n_vals) != 1:
        raise FusionSeedVarianceError(
            f"the run records disagree on {VAL_SAMPLES_KEY!r}: {sorted(n_vals)}; "
            f"the resolution 1/N_val is undefined across differing validation sets"
        )
    n_val = n_vals.pop()

    spread, resolution, floor = compute_floor(accuracies, n_val)

    return {
        "tool": "fusion_seed_variance",
        "arm": arm,
        "deciding_metric": DECIDING_METRIC_KEY,
        "n_runs": len(runs),
        "seeds": seeds,
        "per_seed": per_seed,
        "accuracies": accuracies,
        "spread": spread,
        "n_val": n_val,
        "resolution": resolution,
        "floor": floor,
        "floor_rule": (
            "spread == 0 -> validation-accuracy resolution 1/N_val; "
            "otherwise -> the measured spread"
        ),
        "d05_target_seeds": D05_TARGET_SEEDS,
        "d05_target_met": len(runs) >= D05_TARGET_SEEDS,
        "advisory": (
            "This report states the comparison floor only. It does not declare a "
            "winner, does not compare any inter-arm delta to a threshold, and does "
            "not pick a threshold. The A/B decision is the owner's."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Predeclared seed-variance pass for the fusion-head A/B comparison. "
            "Reports the comparison floor; never decides a winner."
        )
    )
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="RUN_RECORD.json",
        help=(
            "a run_record.json to include; repeat once per seed "
            f"(>= {MIN_RUNS} required, Gate F D-05 uses {D05_TARGET_SEEDS})"
        ),
    )
    ap.add_argument(
        "--report",
        required=True,
        metavar="PATH",
        help="where to write the JSON variance report",
    )
    ap.add_argument(
        "--arm",
        default="A",
        help="the arm label these runs belong to (Gate F D-05 fixes this at A)",
    )
    args = ap.parse_args(argv)

    print("=" * 70)
    print("PHASE 14 — FUSION-HEAD SEED-VARIANCE PASS")
    print("=" * 70)
    print(f"arm         : {args.arm}")
    print(f"deciding    : {DECIDING_METRIC_KEY} (Gate F D-02)")
    print(f"runs        : {len(args.run)}")
    print(f"report      : {args.report}")
    print()

    try:
        runs = [read_run_record(path) for path in args.run]
        report = build_report(runs, arm=args.arm)
    except FusionSeedVarianceError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 2

    hr("PER-SEED DECIDING METRIC")
    for entry in report["per_seed"]:
        acc = entry[DECIDING_METRIC_KEY]
        print(
            f"  seed {str(entry['seed']):>6}  {DECIDING_METRIC_KEY}"
            f" = {acc:.6f}  ({VAL_SAMPLES_KEY}={entry[VAL_SAMPLES_KEY]})"
        )
    print()

    hr("VARIANCE")
    print(f"  seeds        : {report['seeds']}")
    print(f"  N_val        : {report['n_val']}")
    print(f"  resolution   : {report['resolution']:.10g}  (1 / N_val)")
    print(f"  spread       : {report['spread']:.10g}  (max - min)")
    print(f"  floor        : {report['floor']:.10g}")
    print(f"  floor rule   : {report['floor_rule']}")
    if not report["d05_target_met"]:
        print(
            f"  note         : {report['n_runs']} runs < D-05 target "
            f"{D05_TARGET_SEEDS}; the preregistration floor is {MIN_RUNS}"
        )
    print()

    Path(args.report).write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    hr("WROTE")
    print(f"  {args.report}")
    print()
    print(report["advisory"])
    hr("=")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
