"""Phase 4/13 -- select the ROUTER confidence threshold on the VALIDATION split.

    python scripts/sweep_router_threshold.py

WHAT IS REPORTED, AND WHY A SWEEP AT ALL
----------------------------------------
`router/classifier.py` gates the learned router behind a confidence threshold
(`IntentRouter.confidence_threshold`, default `0.70`, read from
`router.confidence_threshold` in the config). Below the gate, the deterministic
lexical fallback answers instead. `docs/PHASE4_ROUTER_REPORT.md` records that
**"Threshold 0.70 is uncalibrated"** -- it was a placeholder chosen before any
validation number existed, and both routing paths currently agree on every
canonical query, so nothing ever forced the value to be examined.

This script forwards the validation split ONCE through the frozen MiniLM
encoder + the trained adapter, and then scores the SAME per-query confidences at
every threshold in a grid. For each threshold it reports:

    * coverage  -- the fraction of val queries the learned router would SERVE
                   (confidence >= threshold); the rest fall back to lexical.
    * covered_task_accuracy -- accuracy among the served queries only.
    * fallback_rate -- 1 - coverage.

The point is to make the coverage/accuracy trade-off visible before a value is
committed, instead of shipping a number nobody measured.

WHAT THIS SCRIPT DELIBERATELY DOES NOT CLAIM
--------------------------------------------
The plan (§59, `plan:3261-3263`) asks for **>= 500 validation queries and 100
hard negatives**. The corpus has **86 validation queries** (train 410 / val 86 /
test 80); the minimum per-class support in val is 8 (`caption`). The val split
contains **zero** hard negatives, because `hn_*` families are held out to TEST
by construction (`router/dataset.py`, `hard_negatives_to_test=True`).

So this is a **corpus-limited** sweep. The artifact records `corpus_limited:
true`, the plan minimums, and the honest counts, and its `notes` state plainly
that the threshold is **NOT calibrated** -- the corpus is too small and too
synthetic to calibrate anything. Selecting a number here is a *justified
default*, not a calibration. **The corpus is NOT padded with generated queries
to reach 500**: that would be fabricating evidence.

Backlog correction, recorded here because the backlog is wrong: backlog item
P1-9 says "n=80". That is the **test** split size, not the validation split. The
sweep target is **val, n=86**. The sweep never touches test.

THE THRESHOLD IS SELECTED ON VALIDATION, AND ONLY ON VALIDATION
---------------------------------------------------------------
Enforced, not documented:

    * `val`   -- allowed. Held out from fitting, so selecting here is fair.
    * `test`  -- REFUSED. The one-shot benchmark. Selecting on it spends it.
    * `train` -- REFUSED. The fitting set. Tuning here measures memorisation.

Any `--split` other than `val` exits 4 and writes nothing. There is no override
flag, on purpose.

SELECTION AND THE TIE-BREAK
---------------------------
`--select-by` chooses the scalar:

    * `covered_accuracy` (default) -- maximise the accuracy among SERVED
      queries; on an exact tie the LOWEST threshold wins (the most permissive
      gate that is no less accurate -- serve as much as possible without losing
      accuracy).
    * `coverage_at_full_accuracy` -- the LOWEST threshold whose served-set
      accuracy is 1.0 (the most permissive gate that admits no error among
      served queries). If no threshold reaches 1.0, the highest threshold.

OVERWRITING IS DELIBERATE, NOT ACCIDENTAL
-----------------------------------------
An existing `threshold_sweep_val.json` is preserved by default (exit 2), because
a sweep is evidence. `--force` overwrites it and stamps `forced_overwrite=true`.

EXIT CODES
----------
    0   swept; artifact written
    2   the thresholds are malformed or outside [0,1]; or the artifact already
        exists and --force was not given; or the corpus/val split is empty
    4   a forbidden split was requested (anything other than `val`)
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "artifacts" / "router"

#: The split a threshold may be selected on. Anything else exits 4.
ALLOWED_SPLIT = "val"

#: The shipped, untuned threshold (docs/PHASE4_ROUTER_REPORT.md:163). The sweep
#: reports the row nearest this so the delta against what was actually shipped
#: is always visible, whatever the grid contains.
SHIPPED_THRESHOLD = 0.70

#: Plan §59 (plan:3261-3263) minimums, recorded so the shortfall is explicit.
PLAN_MIN_VAL_QUERIES = 500
PLAN_MIN_HARD_NEGATIVES = 100

#: The default artifact filename.
ARTIFACT_NAME = "threshold_sweep_val.json"


def hr(title: str = "") -> None:
    print("-" * 70)
    if title:
        print(title)
        print("-" * 70)


def _environment_report(device: str) -> dict:
    facts: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "requested_device": device,
    }
    try:
        import torch

        facts["torch"] = torch.__version__
        facts["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        facts["torch"] = f"unavailable ({type(exc).__name__})"
    return facts


def _build_thresholds(args: argparse.Namespace) -> list[float]:
    """The thresholds to score: an explicit list, or start/stop/step inclusive.

    Rounded to 6 dp so a generated grid does not carry float noise into the
    artifact. Every failure mode raises ValueError HERE, before the encoder or
    the corpus is touched, so a typo costs nothing.
    """
    if args.thresholds is not None:
        values = [part.strip() for part in args.thresholds.split(",")]
        values = [part for part in values if part]
        if not values:
            raise ValueError("--thresholds was given but parsed to nothing")
        out: list[float] = []
        for raw in values:
            try:
                t = round(float(raw), 6)
            except ValueError:
                raise ValueError(f"--thresholds entry is not a number: {raw!r}") from None
            if not 0.0 <= t <= 1.0:
                raise ValueError(
                    f"--thresholds entry outside [0,1]: {t} (from {raw!r})"
                )
            out.append(t)
        return out

    if args.step <= 0:
        raise ValueError(f"--step must be > 0, got {args.step}")

    out = []
    t = args.start
    while t <= args.stop + args.step / 2:
        out.append(round(t, 6))
        t += args.step
    if not out:
        raise ValueError(
            f"no thresholds between start={args.start} and stop={args.stop} "
            f"at step={args.step}"
        )
    for t in out:
        if not 0.0 <= t <= 1.0:
            raise ValueError(f"generated threshold outside [0,1]: {t}")
    return out


# ---------------------------------------------------------------------------
# The sweep itself -- a pure function over (confidence, correctness) pairs so it
# can be unit-tested without loading the encoder or the adapter.
# ---------------------------------------------------------------------------


def _select_row(rows: list[dict], select_by: str) -> dict:
    """Apply the stated selection rule. Ties break toward the LOWER threshold."""
    served = [r for r in rows if r["n_covered"] > 0]
    if not served:  # pragma: no cover - defensive; caller guarantees non-empty
        return rows[0]

    if select_by == "coverage_at_full_accuracy":
        perfect = [r for r in served if r["covered_task_accuracy"] == 1.0]
        if perfect:
            return min(perfect, key=lambda r: r["threshold"])
        return max(served, key=lambda r: r["threshold"])

    if select_by == "covered_accuracy":
        best = max(r["covered_task_accuracy"] for r in served)
        candidates = [r for r in served if r["covered_task_accuracy"] == best]
        return min(candidates, key=lambda r: r["threshold"])

    raise ValueError(f"unknown select_by: {select_by!r}")


def _split_contamination(
    scored_texts: list[str],
    val_texts: set[str],
    test_texts: set[str],
) -> tuple[bool, int, int]:
    """MEASURE whether any TEST example reached the scoring path.

    `test_split_touched` must be a measurement, not a literal. A hardcoded
    `False` is a claim that no change can falsify — precisely the "check that
    cannot fail" this sweep exists to remove. This derives the fact from the
    texts ACTUALLY handed to the encoder: if any scored text belongs to the test
    split, the split firewall has been bypassed and the sweep is invalid.

    Args:
        scored_texts: the exact texts sent to the encoder.
        val_texts: texts in the validation split.
        test_texts: texts in the test split.

    Returns:
        (touched, n_test_scored, n_val_scored).
    """
    scored = set(scored_texts)
    n_test = len(scored & set(test_texts))
    n_val = len(scored & set(val_texts))
    return bool(n_test), n_test, n_val


def sweep_confidence(
    confidences: list[float],
    correct: list[bool],
    thresholds: list[float],
    select_by: str = "covered_accuracy",
) -> tuple[list[dict], dict]:
    """Score every threshold over the SAME per-query confidences.

    Args:
        confidences: the learned router's task confidence for each query
            (max softmax probability).
        correct: whether the learned router's argmax task was the true task.
        thresholds: the grid to score.
        select_by: the selection rule (see `_select_row`).

    Returns:
        (rows, selected) where each row carries the threshold, coverage and
        served-set accuracy, and `selected` is the chosen row.
    """
    n = len(confidences)
    if n == 0:
        raise ValueError("cannot sweep an empty validation set")
    if len(correct) != n:
        raise ValueError(
            f"confidences and correct differ in length: {n} vs {len(correct)}"
        )

    rows: list[dict] = []
    for t in thresholds:
        idx = [i for i in range(n) if confidences[i] >= t]
        n_covered = len(idx)
        coverage = n_covered / n
        acc = (sum(1 for i in idx if correct[i]) / n_covered) if n_covered else None
        rows.append(
            {
                "threshold": t,
                "n_covered": n_covered,
                "coverage": round(coverage, 6),
                "covered_task_accuracy": None if acc is None else round(acc, 6),
                "fallback_rate": round(1.0 - coverage, 6),
            }
        )

    return rows, _select_row(rows, select_by)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sweep the router confidence threshold on the validation split"
    )
    ap.add_argument("--output-dir", default=None,
                    help="defaults to artifacts/router")
    ap.add_argument("--adapter", default=None,
                    help="adapter directory; defaults to artifacts/router/router_adapter_v001")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default=ALLOWED_SPLIT,
                    help="the split to select on; ONLY 'val' is accepted")
    ap.add_argument("--thresholds", default=None,
                    help="explicit comma-separated list, e.g. '0.3,0.4,0.5'")
    ap.add_argument("--start", type=float, default=0.50)
    ap.add_argument("--stop", type=float, default=0.99)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--select-by", default="covered_accuracy",
                    choices=("covered_accuracy", "coverage_at_full_accuracy"))
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing artifact; without it the artifact "
                         "is preserved (exit 2)")
    args = ap.parse_args()

    # -- HARD SPLIT FIREWALL, before anything is loaded or written -----------
    if args.split != ALLOWED_SPLIT:
        print("=" * 70)
        print("REFUSING -- a confidence threshold may only be selected on validation")
        print("=" * 70)
        print(f"  requested split : {args.split!r}")
        print(f"  allowed split   : {ALLOWED_SPLIT!r}")
        print()
        print("  The threshold has to be chosen on the VALIDATION split:")
        print("    * val   -- held out from fitting, so selecting here is fair")
        print("    * test  -- the ONE-SHOT benchmark. Selecting on it spends it:")
        print("               the number stops being held-out and can never be")
        print("               reported as such again.")
        print("    * train -- the fitting set. Tuning here measures memorisation,")
        print("               not generalisation.")
        print()
        print("  There is no override for this. Pass --split val.")
        return 4

    from core.config import load_config

    cfg = load_config()
    device = args.device or cfg.device_preference

    # -- threshold input validation, BEFORE the encoder is loaded -----------
    try:
        thresholds = _build_thresholds(args)
    except ValueError as exc:
        print(f"bad --thresholds / --start/--stop/--step: {exc}")
        return 2

    out_dir = Path(args.output_dir) if args.output_dir else OUT_DIR
    out_path = out_dir / ARTIFACT_NAME
    if out_path.exists() and not args.force:
        print(f"refusing to overwrite an existing artifact: {out_path}")
        print("  Move or remove it deliberately, pass a different --output-dir,")
        print("  or pass --force to overwrite it (the new artifact will record")
        print("  force=true).")
        return 2

    adapter_path = Path(args.adapter) if args.adapter else (
        REPO_ROOT / "artifacts" / "router" / "router_adapter_v001"
    )

    print("=" * 70)
    print("ROUTER -- CONFIDENCE-THRESHOLD SWEEP (VALIDATION ONLY)")
    print("=" * 70)
    print(f"adapter     : {adapter_path}")
    print(f"split       : {args.split}")
    print(f"device      : {device}")
    print(f"config hash : {cfg.hash}")
    print(f"select by   : {args.select_by}")
    print(f"thresholds  : {len(thresholds)} points "
          f"[{thresholds[0]:.2f} .. {thresholds[-1]:.2f}]")
    print()

    hr("ENVIRONMENT")
    env = _environment_report(device)
    for key in sorted(env):
        print(f"  {key:<22} {env[key]}")
    print()

    # -- corpus + val split -------------------------------------------------
    from router.dataset import (
        HARD_NEGATIVE_PREFIX,
        build_corpus,
        split_by_group,
        split_leakage_report,
    )

    hr("CORPUS")
    corpus = build_corpus(n_template_repeats=1, seed=args.seed)
    problems = corpus.validate()
    if problems:
        print("corpus failed validation:")
        for p in problems:
            print(f"  - {p}")
        return 2
    corpus, conflicts = corpus.dedupe()
    if conflicts:
        print("corpus contains labelling conflicts:")
        for c in conflicts:
            print(f"  - {c}")
        return 2

    split = split_by_group(
        corpus, seed=args.seed, val_ratio=0.15, hard_negatives_to_test=True
    )
    report = split_leakage_report(split)
    if not report["clean"]:
        print(f"split failed the group-leakage audit: {report['groups_across_splits']}")
        return 2

    val_examples = split[ALLOWED_SPLIT]
    if not val_examples:
        print(f"no examples in the {ALLOWED_SPLIT!r} split")
        return 2

    val_task_counts = report["task_counts"][ALLOWED_SPLIT]
    n_val = len(val_examples)
    min_support = min(val_task_counts.values()) if val_task_counts else 0
    hard_negatives_in_val = sum(
        1 for e in val_examples if e.group.startswith(HARD_NEGATIVE_PREFIX)
    )
    corpus_limited = (
        n_val < PLAN_MIN_VAL_QUERIES
        or hard_negatives_in_val < PLAN_MIN_HARD_NEGATIVES
    )

    print(f"  total examples   : {len(corpus):,}")
    print(f"  groups           : {len(corpus.groups()):,}")
    print(f"  split sizes      : train {report['sizes']['train']}  "
          f"val {report['sizes']['val']}  test {report['sizes']['test']}")
    print(f"  val per-class    : {val_task_counts}")
    print(f"  val min support  : {min_support}")
    print(f"  hard negatives in val : {hard_negatives_in_val} "
          f"(hn_* families are held out to TEST by design)")
    print(f"  corpus limited   : {corpus_limited} "
          f"(plan wants >={PLAN_MIN_VAL_QUERIES} val / "
          f"{PLAN_MIN_HARD_NEGATIVES} hard negatives)")
    print()

    # -- forward the val split ONCE ----------------------------------------
    from router.classifier import load_adapter
    from router.encoder import build_encoder
    from router.label_space import TASK_TO_INDEX

    hr("SCORING")
    try:
        encoder = build_encoder(cfg, device=device)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED to load the encoder: {type(exc).__name__}: {exc}")
        return 2
    print(f"  encoder : {encoder}")

    try:
        adapter, adapter_meta = load_adapter(adapter_path, device=device)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED to load the adapter: {type(exc).__name__}: {exc}")
        return 2
    print(f"  adapter : {adapter}")

    import torch

    started = datetime.now(timezone.utc)
    texts = [e.text for e in val_examples]
    embeddings = encoder.encode(texts, batch_size=64, normalize=True)
    with torch.no_grad():
        out = adapter(torch.from_numpy(embeddings).to(device))
        task_probs = torch.softmax(out.task_logits, dim=-1).cpu()
        confidences = task_probs.max(dim=-1).values.tolist()
        predicted = task_probs.argmax(dim=-1).tolist()
    true = [TASK_TO_INDEX[e.task] for e in val_examples]
    correct = [predicted[i] == true[i] for i in range(n_val)]
    seconds = (datetime.now(timezone.utc) - started).total_seconds()

    # MEASURE, do not assert. `test_split_touched` is derived from the texts
    # ACTUALLY handed to the encoder, so it is falsifiable: a future edit that
    # widened the scoring loop to include test examples would flip it to True
    # and be refused here, rather than silently keeping a hardcoded `False`.
    val_texts = {e.text for e in val_examples}
    test_texts = {e.text for e in split["test"]}
    test_split_touched, n_test_scored, n_val_scored = _split_contamination(
        texts, val_texts, test_texts
    )
    if test_split_touched:
        print(
            f"BUG: {n_test_scored} TEST example(s) reached the encoder; refusing "
            f"to write a contaminated sweep"
        )
        return 2

    overall_accuracy = sum(1 for c in correct if c) / n_val

    try:
        rows, selected = sweep_confidence(
            confidences, correct, thresholds, select_by=args.select_by
        )
    except ValueError as exc:
        print(f"FAILED to sweep: {exc}")
        return 2

    # -- the curve ---------------------------------------------------------
    shipped_row = min(rows, key=lambda r: abs(r["threshold"] - SHIPPED_THRESHOLD))
    delta = {
        "covered_task_accuracy": (
            None if selected["covered_task_accuracy"] is None
            or shipped_row["covered_task_accuracy"] is None
            else round(
                selected["covered_task_accuracy"]
                - shipped_row["covered_task_accuracy"],
                4,
            )
        ),
        "coverage": round(selected["coverage"] - shipped_row["coverage"], 4),
    }

    print()
    hr("THRESHOLD CURVE")
    print(f"  {'thr':>5}  {'covered':>8}  {'coverage':>9}  "
          f"{'served_acc':>10}  {'fallback':>9}")
    for row in rows:
        acc = row["covered_task_accuracy"]
        acc_s = "  n/a" if acc is None else f"{acc:>6.4f}"
        mark = "  <== selected" if row["threshold"] == selected["threshold"] else ""
        print(f"  {row['threshold']:>5.2f}  {row['n_covered']:>8}  "
              f"{row['coverage']:>9.4f}  {acc_s:>10}  "
              f"{row['fallback_rate']:>9.4f}{mark}")
    print()
    print(f"  val queries        : {n_val}")
    print(f"  ungated accuracy   : {overall_accuracy:.4f} "
          f"(all {n_val} queries, threshold-free)")
    print(f"  selected           : threshold {selected['threshold']:.2f} "
          f"by {args.select_by}")
    print(f"  shipped            : threshold {SHIPPED_THRESHOLD:.2f} "
          f"(docs/PHASE4_ROUTER_REPORT.md:163, uncalibrated)")
    print(f"  DELTA vs shipped   : served_acc {delta['covered_task_accuracy']}  "
          f"coverage {delta['coverage']:+.4f}")
    print()

    # -- artifact ----------------------------------------------------------
    note = (
        "corpus-limited: val n=86 vs plan >=500. This is NOT a calibration -- "
        "the corpus is synthetic and too small (min per-class support 8, caption) "
        "and val carries 0 hard negatives (hn_* families are held out to TEST by "
        "design). Selecting a threshold here yields a justified default, not a "
        "calibrated value. The corpus was NOT padded with generated queries. "
        "Backlog P1-9's 'n=80' is the TEST split; the sweep target is val n=86. "
        "The test split was NOT touched."
    )
    payload = {
        "artifact": "router_threshold_sweep",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "select_by": args.select_by,
        "n_val": n_val,
        "val_task_counts": val_task_counts,
        "val_min_support": min_support,
        "hard_negatives_in_val": hard_negatives_in_val,
        "corpus_limited": corpus_limited,
        "plan_min_val_queries": PLAN_MIN_VAL_QUERIES,
        "plan_min_hard_negatives": PLAN_MIN_HARD_NEGATIVES,
        "split_sizes": report["sizes"],
        "corpus_total": len(corpus),
        "corpus_groups": len(corpus.groups()),
        "thresholds": thresholds,
        "rows": rows,
        "selected": selected,
        "shipped_threshold": SHIPPED_THRESHOLD,
        "shipped_row": shipped_row,
        "delta_vs_shipped": delta,
        "overall_ungated_accuracy": round(overall_accuracy, 6),
        "force": bool(args.force),
        "config_hash": cfg.hash,
        "adapter_path": str(adapter_path),
        "adapter_config_hash": adapter_meta.get("config_hash"),
        "adapter_encoder": adapter_meta.get("encoder"),
        "encoder_type": adapter_meta.get("encoder_type"),
        "device": device,
        "environment": env,
        "seconds": round(seconds, 3),
        # Derived from the texts actually scored (see `_split_contamination`),
        # not a literal: n_test_scored > 0 would have refused before writing.
        "test_split_touched": test_split_touched,
        "n_test_examples_scored": n_test_scored,
        "n_val_examples_scored": n_val_scored,
        "note": note,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    hr("WROTE")
    print(f"  {out_path}")
    hr("=")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
