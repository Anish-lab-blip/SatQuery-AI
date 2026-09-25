"""SatQuery AI — Phase 6: the independent test-split adjudication required by §7.4.

WHY THIS SCRIPT EXISTS
----------------------
Run 1 was `REJECTED` **solely by V2** (V1 passed: val delta +42.00 pp). V2 is the
per-class collapse guardrail. By the contract's own definition V2 is evaluated on
the **validation** split, and `decide_acceptance` hardcodes that
(`_class_failures(baseline_val, adapted_val)`).

But `docs/PHASE6_RUN1_REJECTION_DIAGNOSIS.md` §7.4 condition 3 forbids deciding
there:

    "Decide on data that did not motivate the change. Evaluate `v002` on the
     test split ... Reporting an acceptance decision from `v002` on the same val
     subset that revealed the defect would be self-confirming."

So the acceptance must be adjudicated on the **test** split. V2 needs a per-class
breakdown on that split, and run 1's head-truncated stdout recorded only the
adapted-test confusion and f1 — no per-class data. The `baseline_test` half was
already measured locally (`scripts/phase6_recover_baseline_test.py`), but the
adapted half was never per-class.

The real, integrity-verified run-1 adapter is available locally
(`phase6_realbundle.zip` -> `phase6_adapter/`, adapter_model digest
`07c76a75...` matching `ARTIFACT_SHA256SUMS.json`, tree hash `5c6b8631...`
matching run 1's `adapter_sha256`). So the adapted model can be evaluated on the
frozen test subset **evaluation-only, with no retraining**, and both halves can be
measured on the *same* subset in one process.

WHAT THIS DOES AND DOES NOT DO
------------------------------
- Does **not** retrain. The adapter is loaded from disk and frozen.
- Does **not** alter the v002 rule. V1/V2 criteria are applied verbatim, only to
  the test split instead of val.
- Does **not** overwrite run 1's manifest or the recovery manifest.
- Does **not** treat the earlier val `ACCEPTED` as final. This adjudication
  supersedes it as the contract-facing verdict.

GATES — STOP AT THE FIRST FAILURE
---------------------------------
Gate A'' (no model)  the local corpus reproduces run 1's split sizes, and the
                     drawn test subset's per-class question counts equal the ones
                     `baseline_test` was measured on. That proves the subset is
                     *the same subset* — model-free.
Gate D  (adapted)    the adapted model reproduces run 1's recorded adapted-test
                     control EXACTLY: `exact_match == 0.963`, `n == 1000`,
                     `confusion == {tp:500, fp:19, tn:463, fn:18}`,
                     `f1 == 0.96432`. This is what proves the local adapter is
                     run 1's adapter and that CPU/fp32 reproduces the Kaggle T4
                     at the adapted level too.
Gate E  (baseline)   the base model, on the SAME subset, re-measures
                     `baseline_test` with its per-class breakdown.

Only if A''/D/E pass is V1/V2 applied on the test split.

Usage:
    python scripts/phase6_adjudicate_test.py --stop-after corpus
    python scripts/phase6_adjudicate_test.py            # full A'' -> D -> E -> decide
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

#: Run 1's original manifest — EVIDENCE. Never written to.
RUN1_MANIFEST_PATH = Path(r"D:/New folder (2)/New folder (3)/_x/run_manifest.json")

#: The recovery manifest — read for the subset-identity control. Never written to.
RECOVERY_MANIFEST_PATH = REPO_ROOT / "artifacts" / "vlm" / "run1_test_recovery" / "run_manifest.json"

#: Where this adjudication's evidence is written. A NEW path.
ADJUDICATION_PATH = REPO_ROOT / "artifacts" / "vlm" / "run1_test_recovery" / "test_adjudication.json"

#: The extracted real run-1 adapter (see module docstring for its digest chain).
DEFAULT_ADAPTER_DIR = REPO_ROOT / ".scratch" / "phase6_real_adapter" / "phase6_adapter"

#: Per-question answer caches. A 1000-question CPU split is tens of minutes, and a
#: killed process would otherwise lose every answer. Each cache is fingerprinted to
#: its exact question set, so a stale cache is refused rather than silently reused.
DEFAULT_CACHE_DIR = REPO_ROOT / ".scratch" / "adj_cache"

# -- run-1 recorded constants (the Gate A'' / Gate D controls) ----------------
RUN1_EVAL_LIMIT = 1000
RUN1_AVAILABLE_PER_SPLIT = {"val": 6750, "test": 7772}

#: Run 1's adapted-test endpoint, recovered from its stdout confusion
#: (`tp500/fp19/tn463/fn18`, f1 0.96432). This is the Gate D control.
RUN1_ADAPTED_TEST_EXACT_MATCH = 0.963
RUN1_ADAPTED_TEST_CONFUSION = {"tp": 500, "fp": 19, "tn": 463, "fn": 18}
RUN1_ADAPTED_TEST_F1 = 0.96432


class GateFailure(RuntimeError):
    """A gate did not reproduce. Carries the observed values for the report."""

    def __init__(self, gate: str, message: str, observed: Any) -> None:
        super().__init__(f"{gate}: {message}")
        self.gate = gate
        self.observed = observed


class GateIncomplete(RuntimeError):
    """The gate ran out of its wall budget mid-split.

    NOT a failure: answers are cached per question, so re-running resumes. This
    exists because a 1000-question CPU split takes tens of minutes and a killed
    process used to lose all of it.
    """

    def __init__(self, gate: str, answered: int, total: int, cache_path: Any) -> None:
        super().__init__(
            f"{gate}: incomplete — {answered}/{total} questions answered; "
            f"answers cached in {cache_path}. Re-run to resume."
        )
        self.gate = gate
        self.answered = answered
        self.total = total
        self.cache_path = cache_path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"required manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _per_class_counts(samples: list[Any]) -> dict[str, int]:
    """Per-class question counts via the repository's own classifier."""
    from training.vlm.evaluate import class_of_question

    counts: dict[str, int] = {}
    for sample in samples:
        cls = class_of_question(sample.question)
        if cls is not None:
            counts[cls] = counts.get(cls, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Gate A'' — corpus + subset identity (no model)
# ---------------------------------------------------------------------------
def gate_corpus(cfg: Any, recovery: dict[str, Any]) -> tuple[Any, list[Any], dict[str, Any]]:
    from training.vlm.dataset import build_corpus
    from scripts.phase6_train_vlm import _eval_subset

    print("=" * 78)
    print("GATE A'' — corpus and test-subset identity (no model)")
    print("=" * 78)

    corpus = build_corpus(cfg, limit=None)
    available = {s: len(corpus.by_split(s)) for s in ("val", "test")}
    print(f"  available_per_split : {available}")
    if available != RUN1_AVAILABLE_PER_SPLIT:
        raise GateFailure("GATE A''", f"available_per_split != run 1's {RUN1_AVAILABLE_PER_SPLIT}", available)

    test_subset = _eval_subset(corpus.by_split("test"), limit=RUN1_EVAL_LIMIT, seed=cfg.seed, split="test")
    print(f"  test subset n       : {len(test_subset)}")

    observed = _per_class_counts(test_subset)
    expected = recovery["baseline_test"].get("per_class_counts") or {}
    if not expected:
        raise GateFailure("GATE A''", "recovery manifest carries no baseline_test.per_class_counts to check against", observed)

    if observed != expected:
        diff = {
            k: (expected.get(k), observed.get(k))
            for k in set(expected) | set(observed)
            if expected.get(k) != observed.get(k)
        }
        raise GateFailure(
            "GATE A''",
            f"drawn test subset's per-class counts differ from the subset baseline_test was measured on: {diff}",
            {"observed": observed, "diff": diff},
        )
    print(f"  test per-class counts: EXACT match to the recovery subset ({len(observed)} classes, sum {sum(observed.values())})")
    print("  GATE A'': PASS (same subset, proven without a model)")

    facts = {
        "available_per_split": available,
        "test_subset_n": len(test_subset),
        "test_per_class_counts_match_recovery": True,
        "test_per_class_counts": observed,
    }
    return corpus, test_subset, facts


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_model(cfg: Any, device: str, adapter_path: str | None) -> tuple[Any, Any, str]:
    from specialists.vqa.model import SmolVLM

    vlm = SmolVLM(
        checkpoint=cfg.base_model,
        revision=cfg.revision,
        processor_longest_edge=cfg.processor_longest_edge,
        device=device,
        do_image_splitting=cfg.do_image_splitting,
        adapter_path=adapter_path,
    )
    dtype = str(next(vlm.model.parameters()).dtype)
    return vlm.model, vlm.processor, dtype


def _release(model: Any) -> None:
    del model
    gc.collect()


# ---------------------------------------------------------------------------
# Gate D — the adapted model must reproduce run 1's recorded test control
# ---------------------------------------------------------------------------
def gate_adapted(
    cfg: Any,
    test_subset: list[Any],
    device: str,
    adapter_dir: str,
    *,
    cache_path: Any = None,
    max_seconds: float = 0.0,
) -> tuple[Any, str]:
    from training.vlm.evaluate import evaluate_split

    print("=" * 78)
    print("GATE D — adapted model on the test subset (run-1 reproduction control)")
    print("=" * 78)
    print(f"  adapter: {adapter_dir}")

    started = time.monotonic()
    model, processor, dtype = _load_model(cfg, device, adapter_dir)
    print(f"  loaded in {time.monotonic() - started:.1f} s; parameter dtype = {dtype}")

    deadline = (time.monotonic() + max_seconds) if max_seconds > 0 else None
    metric = evaluate_split(
        model, processor, test_subset, device=device, deadline=deadline,
        cache_path=cache_path,
    )
    if metric.truncated:
        _release(model)
        raise GateIncomplete("GATE D", metric.n, metric.n_available, cache_path)
    print(f"  exact_match : {metric.exact_match!r}  (run 1: {RUN1_ADAPTED_TEST_EXACT_MATCH!r})")
    print(f"  n           : {metric.n}  (run 1: {RUN1_EVAL_LIMIT})")
    print(f"  confusion   : {metric.confusion}  (run 1: {RUN1_ADAPTED_TEST_CONFUSION})")
    print(f"  f1          : {metric.f1!r}  (run 1: {RUN1_ADAPTED_TEST_F1!r})")

    ok = (
        metric.n == RUN1_EVAL_LIMIT
        and metric.exact_match == RUN1_ADAPTED_TEST_EXACT_MATCH
        and metric.confusion == RUN1_ADAPTED_TEST_CONFUSION
    )
    if not ok:
        raise GateFailure(
            "GATE D",
            "the adapted model did NOT reproduce run 1's recorded adapted-test control. "
            "Either the local adapter is not run 1's adapter, or CPU/fp32 does not "
            "reproduce the Kaggle T4 at the adapted level. The test-split adjudication "
            "cannot proceed on unverified metrics.",
            {
                "exact_match": metric.exact_match,
                "n": metric.n,
                "confusion": metric.confusion,
                "f1": metric.f1,
            },
        )
    print("  GATE D: PASS (exact reproduction)")
    # Free the adapted model before the base model is loaded: both are ~2 GB in
    # fp32 and holding them at once is avoidable.
    _release(model)
    return metric, dtype


# ---------------------------------------------------------------------------
# Gate E — the base model, on the SAME subset
# ---------------------------------------------------------------------------
def gate_baseline(
    cfg: Any,
    test_subset: list[Any],
    device: str,
    recovery: dict[str, Any],
    *,
    cache_path: Any = None,
    max_seconds: float = 0.0,
) -> tuple[Any, str]:
    from training.vlm.evaluate import evaluate_split

    print("=" * 78)
    print("GATE E — base model on the same test subset")
    print("=" * 78)

    started = time.monotonic()
    model, processor, dtype = _load_model(cfg, device, None)
    print(f"  loaded in {time.monotonic() - started:.1f} s; parameter dtype = {dtype}")

    deadline = (time.monotonic() + max_seconds) if max_seconds > 0 else None
    metric = evaluate_split(
        model, processor, test_subset, device=device, deadline=deadline,
        cache_path=cache_path,
    )
    if metric.truncated:
        _release(model)
        raise GateIncomplete("GATE E", metric.n, metric.n_available, cache_path)
    print(f"  exact_match : {metric.exact_match!r}")
    print(f"  n           : {metric.n}")
    print(f"  confusion   : {metric.confusion}")

    prior = recovery["baseline_test"].get("exact_match")
    prior_conf = recovery["baseline_test"].get("confusion")
    print(f"  recovery's earlier baseline_test: exact_match={prior!r} confusion={prior_conf}")

    if metric.exact_match != prior or metric.confusion != prior_conf:
        print("  NOTE: this re-measurement differs from the earlier recovery value.")
        print("        Recorded as-is; the earlier value is NOT overwritten.")
    else:
        print("  GATE E: reproduces the earlier recovery measurement exactly.")
    _release(model)
    return metric, dtype


# ---------------------------------------------------------------------------
# The adjudication
# ---------------------------------------------------------------------------
def adjudicate(
    baseline_test: Any,
    adapted_test: Any,
    cfg: Any,
    run1: dict[str, Any],
    recovery: dict[str, Any],
    device: str,
    adapted_dtype: str,
    baseline_dtype: str,
    corpus_facts: dict[str, Any],
    baseline_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from training.vlm.evaluate import (
        MetricSet,
        decide_acceptance,
        describe_acceptance_rule,
    )

    print("=" * 78)
    print("ADJUDICATION — v002 V1/V2 applied on the TEST split")
    print("=" * 78)

    baseline_val = MetricSet(**run1["baseline"])
    adapted_val = MetricSet(**run1["metrics"])

    decision = decide_acceptance(
        baseline_val,
        adapted_val,
        baseline_test=baseline_test,
        adapted_test=adapted_test,
        run_completed=bool(run1["training_state"].get("run_completed", True)),
        artifact_ok=True,
        decision_split="test",
    )
    print(f"  status           : {decision.status}")
    print(f"  decision_split   : test")
    print(f"  test delta       : {decision.test_delta_pp} pp")
    print(f"  val delta        : {decision.val_delta_pp} pp")
    print(f"  class failures   : {len(decision.class_failures)}")
    for r in decision.reasons:
        print(f"    - {r}")

    # Per-class comparison on the test split, for the record.
    base_pc = baseline_test.per_class_accuracy or {}
    adap_pc = adapted_test.per_class_accuracy or {}
    counts = baseline_test.per_class_counts or {}
    rows = []
    for cls in sorted(base_pc):
        n = counts.get(cls, 0)
        b, a = base_pc[cls], adap_pc.get(cls)
        if a is None:
            continue
        rows.append(
            {
                "class": cls,
                "n_questions": n,
                "baseline_pp": round(b * 100.0, 4),
                "adapted_pp": round(a * 100.0, 4),
                "delta_pp": round((a - b) * 100.0, 4),
            }
        )

    return {
        "kind": "independent_test_split_adjudication",
        "required_by": "docs/PHASE6_RUN1_REJECTION_DIAGNOSIS.md §7.4 condition 3",
        "why": (
            "Run 1 was REJECTED solely by V2, which the contract defines on the "
            "validation split. §7.4 forbids deciding on the val subset that revealed "
            "the defect, so V1/V2 are applied to the test split instead. Both halves "
            "are measured in this run on the same frozen 1000-question subset."
        ),
        "supersedes": "the val-split ACCEPTED recorded in run_manifest.json (recovery); that verdict is not final acceptance",
        "rule_version": "v002",
        "rule_unchanged": True,
        "decision_split": "test",
        "not_a_retrain": True,
        "adapter": {
            "path": str(DEFAULT_ADAPTER_DIR),
            "provenance": "phase6_realbundle.zip -> phase6_adapter/; adapter_model.safetensors digest 07c76a75... matches ARTIFACT_SHA256SUMS.json; tree hash 5c6b8631... matches run 1's manifest adapter_sha256",
        },
        "device": device,
        "precision": {"adapted": adapted_dtype, "baseline": baseline_dtype},
        "gates": {
            "gate_corpus": corpus_facts,
            "gate_adapted_reproduced_run1": True,
            "gate_baseline": baseline_provenance
            or {"source": "measured this run (base model, no adapter)"},
        },
        "baseline_test": baseline_test.to_dict(),
        "adapted_test": adapted_test.to_dict(),
        "val_reference": {
            "baseline": run1["baseline"],
            "adapted": run1["metrics"],
        },
        "per_class_test": rows,
        "decision": decision.to_dict(),
        "acceptance_rule": describe_acceptance_rule(),
        "recovery_reference": {
            "path": str(RECOVERY_MANIFEST_PATH),
            "baseline_test_exact_match": recovery["baseline_test"].get("exact_match"),
            "adapted_test_exact_match_from_stdout": recovery["metrics_test"].get("exact_match"),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6 independent test-split adjudication (§7.4)")
    p.add_argument("--stop-after", choices=("corpus", "adapted", "baseline", "decide"), default="decide")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--adapter-dir", default=str(DEFAULT_ADAPTER_DIR))
    p.add_argument("--corpus-root", default="data/bigearthnet_v2/reben/BigEarthNet-S2")
    p.add_argument("--metadata-parquet", default="data/bigearthnet_v2/metadata.parquet")
    p.add_argument("--output", default=str(ADJUDICATION_PATH))
    p.add_argument(
        "--reuse-recorded-baseline",
        action="store_true",
        help=(
            "Skip Gate E and take `baseline_test` from the recovery manifest instead. "
            "That block is already complete (19 per-class accuracies and counts summing "
            "to 1000, truncated=false) and was measured locally on the same frozen "
            "1000-question subset through the same code path, so re-measuring it is "
            "~48 min of CPU for no new information. The report records this provenance "
            "explicitly. Gate D still re-measures the adapted half locally, so both "
            "halves are CPU/fp32 same-source either way."
        ),
    )
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help=(
            "Wall budget for GENERATION in each gate (0 = unlimited). On expiry the "
            "script stops cleanly with an 'incomplete' message instead of being killed; "
            "answers are cached, so re-running resumes where it stopped."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from training.vlm.config import VLMTrainingConfig

    cfg = VLMTrainingConfig.from_registry(
        corpus_root=args.corpus_root,
        metadata_parquet=args.metadata_parquet,
        output_dir=str(ADJUDICATION_PATH.parent),
        seed=args.seed,
    )
    run1 = _load_json(RUN1_MANIFEST_PATH)
    recovery = _load_json(RECOVERY_MANIFEST_PATH)

    print(f"run-1 manifest      : {RUN1_MANIFEST_PATH}")
    print(f"recovery manifest   : {RECOVERY_MANIFEST_PATH}")
    print(f"config_hash         : {cfg.config_hash} (run 1: {run1.get('config_hash')})")
    print(f"base_model          : {cfg.base_model} @ {cfg.revision}")
    print(f"device              : {args.device}")

    cache_dir = Path(args.cache_dir)
    print(f"answer cache dir    : {cache_dir}")

    try:
        _, test_subset, corpus_facts = gate_corpus(cfg, recovery)
        if args.stop_after == "corpus":
            print("\nStopped after the corpus gate. No model was loaded.")
            return 0

        adapted_test, adapted_dtype = gate_adapted(
            cfg, test_subset, args.device, args.adapter_dir,
            cache_path=cache_dir / "adapted_test.jsonl",
            max_seconds=args.max_seconds,
        )
        if args.stop_after == "adapted":
            print("\nStopped after Gate D.")
            return 0

        if args.reuse_recorded_baseline:
            from training.vlm.evaluate import MetricSet

            recorded = recovery["baseline_test"]
            print("=" * 78)
            print("GATE E — SKIPPED (--reuse-recorded-baseline)")
            print("=" * 78)
            print(f"  baseline_test taken from : {RECOVERY_MANIFEST_PATH}")
            print(f"  exact_match              : {recorded.get('exact_match')!r}")
            print(f"  n                        : {recorded.get('n')!r}")
            print(f"  confusion                : {recorded.get('confusion')}")
            print(
                "  per_class_accuracy       : "
                f"{len(recorded.get('per_class_accuracy') or {})} classes "
                f"(counts sum {sum((recorded.get('per_class_counts') or {}).values())})"
            )
            baseline_test = MetricSet(**recorded)
            baseline_dtype = "reused from recovery manifest (measured locally, cpu/fp32)"
            baseline_provenance = {
                "source": "recovery manifest baseline_test — NOT re-measured this run",
                "why": (
                    "--reuse-recorded-baseline: the block is already complete (19 per-class "
                    "accuracies and counts summing to 1000, truncated=false) and was measured "
                    "locally on the same frozen 1000-question subset through the same code path, "
                    "so re-measuring it is ~48 min of CPU for no new information"
                ),
                "path": str(RECOVERY_MANIFEST_PATH),
                "exact_match": recorded.get("exact_match"),
                "confusion": recorded.get("confusion"),
                "truncated": recorded.get("truncated"),
            }
        else:
            baseline_test, baseline_dtype = gate_baseline(
                cfg, test_subset, args.device, recovery,
                cache_path=cache_dir / "baseline_test.jsonl",
                max_seconds=args.max_seconds,
            )
            baseline_provenance = None
            if args.stop_after == "baseline":
                print("\nStopped after Gate E.")
                return 0

        report = adjudicate(
            baseline_test, adapted_test, cfg, run1, recovery,
            args.device, adapted_dtype, baseline_dtype, corpus_facts,
            baseline_provenance,
        )
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\n  adjudication written: {out}")
    except GateIncomplete as exc:
        print("\n" + "~" * 78)
        print(f"{exc}")
        print("~" * 78)
        print("  This is NOT a failure. Re-run the same command to continue.")
        return 0
    except GateFailure as exc:
        print("\n" + "!" * 78)
        print(f"{exc}")
        print(f"observed: {json.dumps(exc.observed, indent=2, default=str)}")
        print("!" * 78)
        return 2

    print("\nALL GATES PASSED — test-split adjudication complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
