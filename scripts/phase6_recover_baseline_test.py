"""SatQuery AI — recover run 1's missing `baseline_test` on a local CPU host.

WHY THIS SCRIPT EXISTS
----------------------
Run 1 (Phase 6, Kaggle T4) evaluated the test split but its V2 short-circuit
returned before the test-delta arithmetic, so `baseline_test`/`metrics_test` were
never persisted (`test_delta_pp: null`). The executed notebook's stdout is
head-truncated and holds no `baseline_test` block, so the number is not
recoverable from any surviving artifact.

The base model is **deterministic and CPU-runnable**, so the missing half — the
*un-adapted* model's test accuracy — can be re-derived locally **without
retraining and without re-running the adapted model**. The adapted half
(`adapted_test = 96.30 pp`) is already established from run 1's stdout confusion
(`tp500/fp19/tn463/fn18`, `f1 0.96432`) and is NOT recomputed here.

WHAT THIS DOES AND DOES NOT CLAIM
---------------------------------
This is **not** a re-run of Phase 6 and it is **not** run 1's original record. It
is a *locally recovered* `baseline_test`, paired with run 1's stdout
`adapted_test`. The resulting `test_delta_pp` therefore mixes two sources (a
locally measured baseline and a Kaggle-recorded adapted), and the recovery
manifest says so explicitly. Run 1's original manifest is never overwritten.

THREE GATES — STOP AT THE FIRST FAILURE
---------------------------------------
Gate A (no model)  the local corpus reproduces run 1's split sizes and the val
                   subset's per-class question counts EXACTLY. If this fails the
                   question set differs and nothing downstream is comparable.
Gate B (val)       the local baseline reproduces run 1's val control EXACTLY:
                   `exact_match == 0.491`, `n == 1000`,
                   `confusion == {tp:28, fp:35, tn:463, fn:474}`. A Gate-B
                   mismatch with Gate A passing is **numeric divergence (CPU vs
                   T4)**, not a different question set.
Gate C (test)      only if B reproduces exactly: measure `baseline_test` and
                   persist it.

The protocol reuses the **real** code path — `build_corpus`, `_eval_subset`,
`SmolVLM`, `evaluate_split`, `decide_acceptance`, `build_run_manifest` — rather
than re-implementing any of it, so the recovered number is produced by the same
code that produced run 1.

DEVICE AND PRECISION — REPORTED, NOT ASSUMED
--------------------------------------------
Run 1 recorded `precision: fp16` on a Kaggle T4. On this CPU-only host
(`torch.cuda.is_available() is False`) `SmolVLM` loads the model in **float32**
(`specialists/vqa/model.py`: `torch.float16 if device == "cuda" else
torch.float32`). The precision actually used is measured from the loaded
parameters and recorded in the manifest — it is never written as `fp16`.

Usage:
    python scripts/phase6_recover_baseline_test.py --stop-after a
    python scripts/phase6_recover_baseline_test.py --prefix-samples 16
    python scripts/phase6_recover_baseline_test.py            # full A -> B -> C
"""

from __future__ import annotations

import argparse
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

#: Where the recovered manifest is written. A NEW path; run 1's is untouched.
RECOVERY_MANIFEST_PATH = REPO_ROOT / "artifacts" / "vlm" / "run1_test_recovery" / "run_manifest.json"

# -- run-1 recorded constants (the Gate A / Gate B / Gate C controls) ---------
#: Run 1's evaluation limit. NOT a CLI knob: contract item U freezes the
#: evaluation set, and Gate A's per-class control is only meaningful at exactly
#: this size. A different limit would draw a different subset and Gate A would
#: fail against run 1's counts for the wrong reason.
RUN1_EVAL_LIMIT = 1000

RUN1_AVAILABLE_PER_SPLIT = {"val": 6750, "test": 7772}

#: Run 1's val-subset per-class question counts, verbatim from the manifest.
#: Sums to 1000.
RUN1_VAL_PER_CLASS_COUNTS: dict[str, int] = {
    "Agro-forestry areas": 43,
    "Arable land": 56,
    "Beaches, dunes, sands": 26,
    "Broad-leaved forest": 85,
    "Coastal wetlands": 35,
    "Complex cultivation patterns": 24,
    "Coniferous forest": 62,
    "Industrial or commercial units": 28,
    "Inland waters": 66,
    "Inland wetlands": 35,
    "Land principally occupied by agriculture, with significant areas of natural vegetation": 26,
    "Marine waters": 163,
    "Mixed forest": 31,
    "Moors, heathland and sclerophyllous vegetation": 25,
    "Natural grassland and sparsely vegetated areas": 30,
    "Pastures": 162,
    "Permanent crops": 36,
    "Transitional woodland, shrub": 32,
    "Urban fabric": 35,
}

#: Run 1's val control.
RUN1_BASELINE_VAL_EXACT_MATCH = 0.491
RUN1_BASELINE_VAL_N = 1000
RUN1_BASELINE_VAL_CONFUSION = {"tp": 28, "fp": 35, "tn": 463, "fn": 474}

#: Run 1's adapted-test endpoint, from the stdout confusion (tp500/fp19/tn463/fn18,
#: f1 0.96432). Established; NOT recomputed here.
RUN1_ADAPTED_TEST_PP = 96.30
RUN1_ADAPTED_TEST_CONFUSION = {"tp": 500, "fp": 19, "tn": 463, "fn": 18}
RUN1_ADAPTED_TEST_F1 = 0.96432


class GateFailure(RuntimeError):
    """A gate did not reproduce. Carries the observed values for the report."""

    def __init__(self, gate: str, message: str, observed: Any) -> None:
        super().__init__(f"{gate}: {message}")
        self.gate = gate
        self.observed = observed


def _load_run1() -> dict[str, Any]:
    if not RUN1_MANIFEST_PATH.is_file():
        raise SystemExit(f"run-1 manifest not found: {RUN1_MANIFEST_PATH}")
    return json.loads(RUN1_MANIFEST_PATH.read_text(encoding="utf-8"))


def _per_class_counts(samples: list[Any]) -> dict[str, int]:
    """Per-class question counts, using the repository's own classifier.

    Reuses `training.vlm.evaluate.class_of_question` rather than re-deriving the
    presence template, so the count is the same one `_metric_set` would produce.
    """
    from training.vlm.evaluate import class_of_question

    counts: dict[str, int] = {}
    for sample in samples:
        cls = class_of_question(sample.question)
        if cls is not None:
            counts[cls] = counts.get(cls, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Gate A — corpus identity (no model)
# ---------------------------------------------------------------------------
def gate_a(cfg: Any) -> tuple[Any, dict[str, list[Any]]]:
    from training.vlm.dataset import build_corpus
    from scripts.phase6_train_vlm import _eval_subset

    print("=" * 78)
    print("GATE A — corpus identity (no model)")
    print("=" * 78)

    corpus = build_corpus(cfg, limit=None)
    available = {s: len(corpus.by_split(s)) for s in ("val", "test")}
    print(f"  available_per_split : {available}")
    if available != RUN1_AVAILABLE_PER_SPLIT:
        raise GateFailure(
            "GATE A",
            f"available_per_split != run 1's {RUN1_AVAILABLE_PER_SPLIT}",
            available,
        )

    eval_sets = {
        split: _eval_subset(
            corpus.by_split(split), limit=RUN1_EVAL_LIMIT, seed=cfg.seed, split=split
        )
        for split in ("val", "test")
    }
    print(f"  selected_per_split  : {{'val': {len(eval_sets['val'])}, 'test': {len(eval_sets['test'])}}}")

    observed = _per_class_counts(eval_sets["val"])
    if observed != RUN1_VAL_PER_CLASS_COUNTS:
        diff = {
            k: (RUN1_VAL_PER_CLASS_COUNTS.get(k), observed.get(k))
            for k in set(RUN1_VAL_PER_CLASS_COUNTS) | set(observed)
            if RUN1_VAL_PER_CLASS_COUNTS.get(k) != observed.get(k)
        }
        raise GateFailure(
            "GATE A",
            f"val per-class counts differ from run 1 (expected/observed): {diff}",
            {"observed": observed, "diff": diff},
        )
    print(f"  val per-class counts: EXACT match ({len(observed)} classes, sum {sum(observed.values())})")
    print("  GATE A: PASS")
    return corpus, eval_sets


# ---------------------------------------------------------------------------
# Gate B — validation control
# ---------------------------------------------------------------------------
def _load_model(cfg: Any, device: str) -> tuple[Any, Any, str]:
    from specialists.vqa.model import SmolVLM

    vlm = SmolVLM(
        checkpoint=cfg.base_model,
        revision=cfg.revision,
        processor_longest_edge=cfg.processor_longest_edge,
        device=device,
        do_image_splitting=cfg.do_image_splitting,
    )
    # The dtype actually used, read off the loaded parameters -- not assumed.
    dtype = str(next(vlm.model.parameters()).dtype)
    return vlm.model, vlm.processor, dtype


def gate_b(model: Any, processor: Any, val_subset: list[Any], device: str) -> Any:
    from training.vlm.evaluate import evaluate_split

    print("=" * 78)
    print("GATE B — validation control")
    print("=" * 78)
    metric = evaluate_split(model, processor, val_subset, device=device, deadline=None)
    print(f"  exact_match : {metric.exact_match!r}  (run 1: {RUN1_BASELINE_VAL_EXACT_MATCH!r})")
    print(f"  n           : {metric.n}  (run 1: {RUN1_BASELINE_VAL_N})")
    print(f"  confusion   : {metric.confusion}  (run 1: {RUN1_BASELINE_VAL_CONFUSION})")

    ok = (
        metric.n == RUN1_BASELINE_VAL_N
        and metric.exact_match == RUN1_BASELINE_VAL_EXACT_MATCH
        and metric.confusion == RUN1_BASELINE_VAL_CONFUSION
    )
    if not ok:
        raise GateFailure(
            "GATE B",
            "val control did NOT reproduce exactly. Gate A passed, so this is "
            "NUMERIC DIVERGENCE (CPU float32 vs Kaggle T4 fp16), NOT a different "
            "question set.",
            {
                "exact_match": metric.exact_match,
                "n": metric.n,
                "confusion": metric.confusion,
            },
        )
    print("  GATE B: PASS (exact)")
    return metric


# ---------------------------------------------------------------------------
# Gate C — test split
# ---------------------------------------------------------------------------
def _adapted_test_dict() -> dict[str, Any]:
    """Run 1's adapted-test MetricSet, reconstructed from its stdout confusion."""
    tp, fp, tn, fn = (
        RUN1_ADAPTED_TEST_CONFUSION["tp"],
        RUN1_ADAPTED_TEST_CONFUSION["fp"],
        RUN1_ADAPTED_TEST_CONFUSION["tn"],
        RUN1_ADAPTED_TEST_CONFUSION["fn"],
    )
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "n": n,
        "n_available": n,
        "truncated": False,
        "exact_match": round(RUN1_ADAPTED_TEST_PP / 100.0, 6),
        "per_class_accuracy": {},
        "per_class_counts": {},
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": RUN1_ADAPTED_TEST_F1,
        "confusion": dict(RUN1_ADAPTED_TEST_CONFUSION),
        "source": "run 1 Kaggle stdout (tp500/fp19/tn463/fn18, f1 0.96432)",
        "note": (
            "per_class_accuracy/per_class_counts are unavailable: run 1's stdout "
            "records only the confusion and f1. exact_match is the endpoint the "
            "test delta needs; the per-class split is not reconstructible."
        ),
    }


def gate_c(
    model: Any,
    processor: Any,
    test_subset: list[Any],
    cfg: Any,
    run1: dict[str, Any],
    device: str,
    dtype: str,
    gate_a_facts: dict[str, Any],
    gate_b_facts: dict[str, Any],
) -> dict[str, Any]:
    from training.vlm.evaluate import (
        MetricSet,
        decide_acceptance,
        describe_acceptance_rule,
        evaluate_split,
    )
    from training.vlm.run_manifest import build_run_manifest, manifest_hash, write_run_manifest

    print("=" * 78)
    print("GATE C — test split (baseline only)")
    print("=" * 78)
    baseline_test = evaluate_split(model, processor, test_subset, device=device, deadline=None)
    print(f"  baseline_test exact_match : {baseline_test.exact_match!r} ({baseline_test.exact_match*100:.2f} pp)")
    print(f"  baseline_test n           : {baseline_test.n}")
    print(f"  baseline_test confusion   : {baseline_test.confusion}")
    if baseline_test.n != RUN1_EVAL_LIMIT:
        raise GateFailure(
            "GATE C",
            f"test split scored {baseline_test.n} questions, expected {RUN1_EVAL_LIMIT}",
            {"n": baseline_test.n},
        )

    test_delta_pp = round(RUN1_ADAPTED_TEST_PP - baseline_test.exact_match * 100.0, 4)
    print(f"  test_delta_pp = round({RUN1_ADAPTED_TEST_PP} - {baseline_test.exact_match*100:.2f}, 4) = {test_delta_pp}")

    # -- rebuild the manifest through the fixed path -----------------------
    adapted_test = _adapted_test_dict()
    baseline_test_dict = baseline_test.to_dict()

    manifest = build_run_manifest(
        config=cfg,
        corpus_summary=run1["corpus"],
        lora_report=run1["lora"],
        frozen_report=run1["frozen_params"],
        prompt_contract=run1["prompt_contract"],
        state=run1["training_state"],
        # run 1's VAL metrics, carried unchanged.
        metrics=run1["metrics"],
        baseline=run1["baseline"],
        # the recovered / established TEST blocks.
        metrics_test=adapted_test,
        baseline_test=baseline_test_dict,
        evaluation_budget=run1["evaluation_budget"],
        started_at=run1["started_at"],
        finished_at=run1["finished_at"],
    )

    # The v002 decision on the TEST split -- the thing this recovery enables.
    # Built from the run-1 val MetricSets (unchanged) and the recovered/established
    # test blocks. `artifact_ok`/`run_completed` mirror run 1's own record: run 1's
    # REJECTED verdict is a V1/V2 verdict, which means V3/V4 passed.
    baseline_val = MetricSet(**run1["baseline"])
    adapted_val = MetricSet(**run1["metrics"])
    decision = decide_acceptance(
        baseline_val,
        adapted_val,
        baseline_test=baseline_test,
        adapted_test=MetricSet(**{k: v for k, v in adapted_test.items() if k in MetricSet.__dataclass_fields__}),
        run_completed=bool(run1["training_state"].get("run_completed", True)),
        artifact_ok=True,
    )
    print(f"  v002 decision (test) : {decision.status}  test_delta_pp={decision.test_delta_pp}")

    manifest["decision"] = decision.to_dict()
    manifest["acceptance_rule"] = describe_acceptance_rule()
    manifest["recovery"] = {
        "kind": "locally_recovered_baseline_test",
        "not_run1_original": True,
        "reason": (
            "run 1 evaluated the test split but the V2 short-circuit returned "
            "before the test-delta arithmetic, so baseline_test/metrics_test were "
            "never persisted. The un-adapted model is deterministic and "
            "CPU-runnable, so baseline_test is re-derived locally. The adapted "
            "test half is run 1's stdout confusion, NOT recomputed."
        ),
        "original_manifest_path": str(RUN1_MANIFEST_PATH),
        "original_manifest_hash": run1.get("manifest_hash"),
        "original_acceptance_rule_version": run1.get("acceptance_rule_version"),
        "baseline_test_source": "measured locally this run (base model, no adapter)",
        "adapted_test_source": "run 1 Kaggle stdout (tp500/fp19/tn463/fn18, f1 0.96432)",
        "delta_is_cross_source": True,
        "delta_is_cross_source_note": (
            "test_delta_pp pairs a locally measured baseline_test with a "
            "Kaggle-recorded adapted_test. Both are on the same frozen 1000-"
            "question test subset (Gate A), but the two halves were produced on "
            "different hardware."
        ),
        "device": device,
        "precision_actually_used": dtype,
        "precision_note": (
            "run 1 recorded precision=fp16 on a Kaggle T4; this host is CPU-only "
            "(torch.cuda.is_available()=False) and SmolVLM loads float32 here. The "
            "value above is read off the loaded parameters."
        ),
        "gate_a": gate_a_facts,
        "gate_b": gate_b_facts,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest["manifest_hash"] = manifest_hash(manifest)
    path = write_run_manifest(manifest, RECOVERY_MANIFEST_PATH)
    print(f"  recovery manifest written: {path}")
    print(f"  manifest_hash            : {manifest['manifest_hash']}")
    print("  GATE C: PASS")
    return manifest


def _time_prefix(model: Any, processor: Any, samples: list[Any], device: str, n: int) -> float:
    from training.vlm.evaluate import evaluate_split

    print(f"  timing a {n}-sample prefix ...")
    started = time.monotonic()
    _ = evaluate_split(model, processor, samples[:n], device=device, deadline=None)
    elapsed = time.monotonic() - started
    per_sample = elapsed / n
    print(
        f"  {n} samples in {elapsed:.1f} s -> {per_sample:.2f} s/sample "
        f"(est. {per_sample * 1000 / 60:.1f} min per 1000-sample split)"
    )
    return per_sample


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recover run 1's missing baseline_test locally")
    p.add_argument("--stop-after", choices=("a", "prefix", "b", "c"), default="c",
                   help="stop after this gate (default: c); 'prefix' times a short "
                        "run and stops without scoring the full val split")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix-samples", type=int, default=16,
                   help="samples used to measure throughput before the full split")
    p.add_argument("--corpus-root", default="data/bigearthnet_v2/reben/BigEarthNet-S2")
    p.add_argument("--metadata-parquet", default="data/bigearthnet_v2/metadata.parquet")
    p.add_argument("--output-dir", default="artifacts/vlm/run1_test_recovery")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from training.vlm.config import VLMTrainingConfig

    cfg = VLMTrainingConfig.from_registry(
        corpus_root=args.corpus_root,
        metadata_parquet=args.metadata_parquet,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    run1 = _load_run1()
    print(f"run-1 manifest      : {RUN1_MANIFEST_PATH}")
    print(f"config_hash         : {cfg.config_hash} (run 1: {run1.get('config_hash')})")
    print(f"base_model          : {cfg.base_model} @ {cfg.revision}")
    print(f"device              : {args.device}")

    try:
        corpus, eval_sets = gate_a(cfg)
        gate_a_facts = {
            "available_per_split": RUN1_AVAILABLE_PER_SPLIT,
            "val_per_class_counts_match": True,
            "val_subset_n": len(eval_sets["val"]),
            "test_subset_n": len(eval_sets["test"]),
        }
        if args.stop_after == "a":
            print("\nStopped after Gate A (--stop-after a). No model was loaded.")
            return 0

        print("=" * 78)
        print("loading base model ...")
        print("=" * 78)
        load_started = time.monotonic()
        model, processor, dtype = _load_model(cfg, args.device)
        print(f"  loaded in {time.monotonic() - load_started:.1f} s; parameter dtype = {dtype}")

        _time_prefix(model, processor, eval_sets["val"], args.device, args.prefix_samples)
        if args.stop_after == "prefix":
            print("\nStopped after the timing prefix (--stop-after prefix).")
            return 0

        baseline_val = gate_b(model, processor, eval_sets["val"], args.device)
        gate_b_facts = {
            "exact_match": baseline_val.exact_match,
            "n": baseline_val.n,
            "confusion": baseline_val.confusion,
        }
        if args.stop_after == "b":
            print("\nStopped after Gate B (--stop-after b).")
            return 0

        gate_c(
            model, processor, eval_sets["test"], cfg, run1,
            args.device, dtype, gate_a_facts, gate_b_facts,
        )
    except GateFailure as exc:
        print("\n" + "!" * 78)
        print(f"{exc}")
        print(f"observed: {json.dumps(exc.observed, indent=2, default=str)}")
        print("!" * 78)
        return 2

    print("\nALL GATES PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
