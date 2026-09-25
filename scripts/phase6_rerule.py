#!/usr/bin/env python3
"""Phase 6 re-rule harness — re-score RECORDED metrics under a rule version.

WHAT THIS IS
------------
A run's `run_manifest.json` records the baseline and adapted `MetricSet`s (their
confusion matrices, per-class accuracies and counts) but NOT the raw predictions.
This script rebuilds those `MetricSet`s from the manifest and re-applies the
acceptance rule, so a change to the rule can be audited **without a GPU, without
retraining, and without downloading a model**.

WHAT THIS IS NOT
----------------
It is NOT a new evaluation. It re-scores numbers that were already produced by a
real run. It cannot change the predictions, the metrics, or the split. In
particular, run 1 evaluated the **test** split but never persisted its metrics,
so no invocation of this script can produce a test-split verdict — an ACCEPTED
here is an ACCEPTED **on validation only**.

Usage:
    python scripts/phase6_rerule.py <run_manifest.json>
    python scripts/phase6_rerule.py <run_manifest.json> --rule-version v001
    python scripts/phase6_rerule.py <run_manifest.json> --compare
    python scripts/phase6_rerule.py <run_manifest.json> --compare --json

Exit codes:
    0  the analysis ran (the verdict is in the output, not the exit code)
    2  the manifest could not be read, or an unknown rule version was requested

Only the standard library and `training.vlm.evaluate` are used. No torch, no
transformers, no network.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.vlm.evaluate import (  # noqa: E402
    ACCEPTANCE_RULE_VERSION,
    CLASS_DROP_Z,
    LEGACY_RULE_VERSIONS,
    MAX_CLASS_DROP_PP,
    MIN_CLASS_DROP_QUESTIONS,
    MIN_CLASS_QUESTIONS,
    MetricSet,
    decide_acceptance,
)

BANNER = """\
================================================================================
  PHASE 6 RE-RULE HARNESS  --  an AUDIT, not an evaluation
--------------------------------------------------------------------------------
  This re-scores metrics that a run ALREADY RECORDED in its run_manifest.json.
  It does not load a model, generate an answer, or touch a GPU. It cannot change
  the predictions or the metrics, and it cannot produce a TEST-SPLIT verdict:
  run 1's test-split metrics were never persisted (the manifest's test_delta_pp
  is null). An ACCEPTED below is an ACCEPTED on VALIDATION ONLY.
================================================================================
"""


def _metric_set_from_block(block: dict[str, Any]) -> MetricSet:
    """Rebuild a `MetricSet` from a recorded manifest block.

    The manifest stores exactly the fields `MetricSet.to_dict()` emits, so the
    round trip is lossless. No metric is re-derived from raw predictions -- the
    manifest does not contain them, and inventing them would not be a replay.
    """
    return MetricSet(
        n=int(block.get("n", 0)),
        exact_match=float(block.get("exact_match", 0.0)),
        per_class_accuracy={
            k: float(v) for k, v in (block.get("per_class_accuracy") or {}).items()
        },
        precision=float(block.get("precision", 0.0)),
        recall=float(block.get("recall", 0.0)),
        f1=float(block.get("f1", 0.0)),
        confusion={k: int(v) for k, v in (block.get("confusion") or {}).items()},
        per_class_counts={
            k: int(v) for k, v in (block.get("per_class_counts") or {}).items()
        },
        truncated=bool(block.get("truncated", False)),
        n_available=int(block.get("n_available", 0)),
    )


def _per_class_rows(
    baseline: MetricSet, adapted: MetricSet
) -> list[dict[str, Any]]:
    """One row per class the V2 guardrail considers, with its full evidence.

    Rows use the same filter as the rule (`>= MIN_CLASS_QUESTIONS` questions in
    the baseline and present in both splits). SE and z use the rule's own
    formula so the table and the verdict cannot disagree about arithmetic.
    """
    rows: list[dict[str, Any]] = []
    for cls, count in baseline.per_class_counts.items():
        if count < MIN_CLASS_QUESTIONS:
            continue
        if cls not in adapted.per_class_accuracy:
            continue
        base_pp = baseline.per_class_accuracy.get(cls, 0.0) * 100.0
        adapted_pp = adapted.per_class_accuracy.get(cls, 0.0) * 100.0
        drop_pp = base_pp - adapted_pp
        lost_questions = round((drop_pp / 100.0) * count)
        p_base = base_pp / 100.0
        p_adapted = adapted_pp / 100.0
        se_pp = (
            math.sqrt(
                max(
                    p_base * (1.0 - p_base) + p_adapted * (1.0 - p_adapted),
                    1e-12,
                )
                / count
            )
            * 100.0
        )
        z = drop_pp / se_pp
        rows.append(
            {
                "class": cls,
                "n_questions": count,
                "baseline_pp": round(base_pp, 4),
                "adapted_pp": round(adapted_pp, 4),
                "drop_pp": round(drop_pp, 4),
                "se_pp": round(se_pp, 4),
                "z": round(z, 4),
                "lost_questions": lost_questions,
                "fails_v001": adapted_pp < base_pp - MAX_CLASS_DROP_PP,
                "fails_v002": (
                    lost_questions >= MIN_CLASS_DROP_QUESTIONS
                    and z >= CLASS_DROP_Z
                ),
            }
        )
    # Largest drops first: the classes a reader most wants to judge.
    rows.sort(key=lambda r: r["drop_pp"], reverse=True)
    return rows


def _fmt_class(name: str, width: int = 42) -> str:
    return name if len(name) <= width else name[: width - 1] + "\u2026"


def _print_table(rows: list[dict[str, Any]]) -> None:
    header = (
        f"  {'class':<42} {'n':>4} {'base pp':>8} {'adap pp':>8} "
        f"{'drop pp':>8} {'SE pp':>7} {'z':>7} {'lost':>5}  {'v001':>5} {'v002':>5}"
    )
    print(f"\nPer-class V2 evidence (classes with >= {MIN_CLASS_QUESTIONS} questions):")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        print(
            f"  {_fmt_class(r['class']):<42} {r['n_questions']:>4} "
            f"{r['baseline_pp']:>8.3f} {r['adapted_pp']:>8.3f} "
            f"{r['drop_pp']:>8.3f} {r['se_pp']:>7.3f} {r['z']:>7.3f} "
            f"{r['lost_questions']:>5}  "
            f"{'FAIL' if r['fails_v001'] else '.':>5} "
            f"{'FAIL' if r['fails_v002'] else '.':>5}"
        )
    n_v001 = sum(1 for r in rows if r["fails_v001"])
    n_v002 = sum(1 for r in rows if r["fails_v002"])
    print(
        f"\n  v001 criterion (flat drop > {MAX_CLASS_DROP_PP:.1f} pp): "
        f"{n_v001} class(es) fail"
    )
    print(
        f"  v002 criterion (lost >= {MIN_CLASS_DROP_QUESTIONS} questions AND "
        f"z >= {CLASS_DROP_Z}): {n_v002} class(es) fail"
    )


def _decide(
    baseline: MetricSet, adapted: MetricSet, rule_version: str
) -> dict[str, Any]:
    """Apply the rule to the recorded metrics, with no test metric sets."""
    decision = decide_acceptance(
        baseline,
        adapted,
        run_completed=True,   # V3: the run completed its predeclared budget
        artifact_ok=True,     # V4: the artifact verified (recorded in the manifest)
        rule_version=rule_version,
    )
    return decision.to_dict()


def _print_decision(rule_version: str, decision: dict[str, Any]) -> None:
    print(f"\n  Decision under {rule_version}: {decision['status']}")
    print(f"    val delta: {decision['val_delta_pp']:+.2f} pp")
    for reason in decision["reasons"]:
        print(f"    - {reason}")
    if decision["class_failures"]:
        print("    class failures:")
        for f in decision["class_failures"]:
            extra = ""
            if "lost_questions" in f:
                extra = f", lost={f['lost_questions']}, z={f['z']}"
            print(
                f"      {f['class']}: {f['baseline_pp']:.3f} -> "
                f"{f['adapted_pp']:.3f} pp (drop {f['drop_pp']:.3f}){extra}"
            )
    if decision["status"] == "ACCEPTED":
        print(
            "    NOTE: this is ACCEPTED on VALIDATION. Run 1's test-split metrics "
            "were never persisted, so this is not an acceptance."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-score a run's RECORDED metrics under a chosen acceptance-rule "
            "version. An audit, not an evaluation."
        )
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="path to a run_manifest.json containing 'baseline' and 'metrics'",
    )
    parser.add_argument(
        "--rule-version",
        default=ACCEPTANCE_RULE_VERSION,
        help=(
            f"acceptance-rule version to apply (default: {ACCEPTANCE_RULE_VERSION}); "
            f"known versions: {ACCEPTANCE_RULE_VERSION}, "
            f"{', '.join(LEGACY_RULE_VERSIONS)}"
        ),
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="print the decision under BOTH v001 and v002 side by side",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of (or in addition to) the report",
    )
    args = parser.parse_args(argv)

    print(BANNER)

    if not args.manifest.is_file():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not read manifest {args.manifest}: {exc}", file=sys.stderr)
        return 2

    missing = [k for k in ("baseline", "metrics") if k not in manifest]
    if missing:
        print(
            f"ERROR: manifest is missing {missing}; it must carry the recorded "
            f"'baseline' and 'metrics' blocks to be replayable",
            file=sys.stderr,
        )
        return 2

    baseline = _metric_set_from_block(manifest["baseline"])
    adapted = _metric_set_from_block(manifest["metrics"])

    versions = ["v001", "v002"] if args.compare else [args.rule_version]
    known = (ACCEPTANCE_RULE_VERSION, *LEGACY_RULE_VERSIONS)
    unknown = [v for v in versions if v not in known]
    if unknown:
        print(
            f"ERROR: unknown rule version(s) {unknown}; known: {known}",
            file=sys.stderr,
        )
        return 2

    rows = _per_class_rows(baseline, adapted)

    print(f"  manifest        : {args.manifest}")
    print(f"  manifest version: acceptance_rule_version={manifest.get('acceptance_rule_version')!r}")
    print(
        f"  recorded decision: {manifest.get('decision', {}).get('status')} "
        f"(val delta {manifest.get('decision', {}).get('val_delta_pp')} pp)"
    )
    print(
        f"  val metrics     : baseline exact_match={baseline.exact_match:.6f} "
        f"(n={baseline.n}), adapted exact_match={adapted.exact_match:.6f} "
        f"(n={adapted.n})"
    )
    # The test split's status is read from the manifest, not assumed: run 1 DID
    # evaluate it, but the metrics were never persisted, so it is un-replayable.
    n_eval = (manifest.get("evaluation_budget") or {}).get("n_evaluated") or {}
    if any(k.endswith("_test") for k in n_eval):
        test_note = (
            f"evaluated in the run (n_evaluated={ {k: v for k, v in n_eval.items() if k.endswith('_test')} }) "
            f"but its metrics were NOT persisted -> no test-split verdict is possible"
        )
    else:
        test_note = (
            "NOT PRESENT in the manifest (no test-split evaluation recorded) "
            "-> no test-split verdict"
        )
    print(f"  test metrics    : {test_note}")
    print(
        "  replay note     : the *_pp values below are computed from the "
        "manifest's 6-dp-rounded per_class_accuracy, so they can differ in the "
        "last decimal from the run's own computation (which used raw counts)"
    )

    _print_table(rows)

    decisions: dict[str, Any] = {}
    for version in versions:
        decision = _decide(baseline, adapted, version)
        decisions[version] = decision
        _print_decision(version, decision)
        # Cross-check: the rule's own failure list must match the table.
        table_fails = sum(1 for r in rows if r[f"fails_{version}"])
        if table_fails != len(decision["class_failures"]):
            print(
                f"    WARNING: table shows {table_fails} failing class(es) but the "
                f"rule returned {len(decision['class_failures'])}; the harness and "
                f"the rule disagree",
                file=sys.stderr,
            )

    if args.compare:
        print("\n  --- SIDE BY SIDE ---")
        for version in versions:
            d = decisions[version]
            print(
                f"    {version}: {d['status']:<10} "
                f"class_failures={len(d['class_failures'])}  "
                f"val_delta={d['val_delta_pp']:+.2f} pp"
            )
        print(
            "    (both verdicts are on VALIDATION; neither is a test-split "
            "verdict, because run 1's test-split metrics were never persisted)"
        )

    if args.json:
        payload = {
            "manifest": str(args.manifest),
            "manifest_acceptance_rule_version": manifest.get("acceptance_rule_version"),
            "recorded_decision": manifest.get("decision"),
            "is_audit_not_evaluation": True,
            "test_split_metrics_persisted": False,
            "per_class": rows,
            "decisions": decisions,
        }
        print("\nJSON:")
        print(json.dumps(payload, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
