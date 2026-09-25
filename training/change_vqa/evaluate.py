"""SatQuery AI — change-VQA evaluation (R-02).

WHAT IS MEASURED, AND WHAT IS NOT
---------------------------------
CDVQA's answer space is a closed set of 19 short strings. The task is therefore
**classification**, and the metrics are classification metrics:

    answer accuracy        the primary number; the benchmark's own criterion
    macro-F1               reported because accuracy alone hides a collapsed
                           minority class, and `yes`/`no` are 58% of Train
    per-type accuracy      because a single aggregate hides the eight types
    top-3 accuracy         because several ratio bins are adjacent and a near
                           miss is not the same error as a wild one
    estimator MAE          the class-wise change estimate against label1/label2

The plan's section 2.2 also lists BLEU/ROUGE-L/CIDEr/BERTScore under
CAPTIONING. Those are generative-language metrics and are deliberately NOT
computed here: this task does not generate language, so reporting them would be
reporting a number that does not describe the system. R-16 (which captioning
metrics to build, against which definition) is a separate, open ruling.

Every report carries its own provenance — dataset, split, counts, checkpoint
hash, config hash, preprocessing version, metric version, inference
configuration and timestamp — so two numbers can never be compared without
knowing whether they describe the same thing. Failed and skipped samples are
counted, never dropped from the denominator silently.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from training.change_vqa.dataset import (
    DATASET_ID,
    PREPROCESSING_VERSION,
    ChangeVQARecord,
    SceneTargets,
)
from training.change_vqa.features import FEATURE_SPEC
from training.change_vqa.vocab import (
    CHANGE_CLASS_ORDER,
    INDEX_TO_ANSWER,
    N_ANSWERS,
    QUESTION_TYPE_ORDER,
)

#: Bumped when a metric definition changes. Recorded in every report so a
#: number produced under an older definition is identifiable as such.
METRIC_IMPLEMENTATION = "change_vqa_metrics_v1"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def answer_accuracy(predicted: Sequence[int], gold: Sequence[int]) -> float:
    """Exact-match accuracy. The primary metric."""
    if not gold:
        return 0.0
    hits = sum(1 for p, g in zip(predicted, gold) if int(p) == int(g))
    return hits / len(gold)


def top_k_accuracy(
    probabilities: Any, gold: Sequence[int], k: int = 3
) -> float:
    """Fraction of rows whose gold answer is in the top `k` by probability."""
    import numpy as np

    if not gold:
        return 0.0
    probs = np.asarray(probabilities)
    if probs.ndim != 2 or probs.shape[1] != N_ANSWERS:
        raise ValueError(
            f"probabilities must be (N, {N_ANSWERS}), got {probs.shape}"
        )
    k = max(1, min(int(k), N_ANSWERS))
    top = np.argsort(-probs, axis=1)[:, :k]
    hits = sum(
        1 for row, g in zip(top, gold) if int(g) in {int(v) for v in row}
    )
    return hits / len(gold)


def macro_f1(predicted: Sequence[int], gold: Sequence[int]) -> float:
    """Macro-averaged F1 over the answers that actually occur in `gold`.

    Hand-rolled rather than imported: the project has no scikit-learn
    dependency, and adding one to compute a nine-line formula would put a
    ~30 MB package on the deployment path for no benefit.

    The label set is the union of observed gold and predicted labels. Averaging
    over all 19 would count an answer that never appears as a zero-F1 class and
    depress the score by an amount that depends on the split's vocabulary — a
    number that would move when the split changed for reasons unrelated to the
    model.
    """
    labels = sorted({int(g) for g in gold} | {int(p) for p in predicted})
    if not labels:
        return 0.0
    scores: list[float] = []
    for label in labels:
        tp = sum(1 for p, g in zip(predicted, gold) if int(p) == label and int(g) == label)
        fp = sum(1 for p, g in zip(predicted, gold) if int(p) == label and int(g) != label)
        fn = sum(1 for p, g in zip(predicted, gold) if int(p) != label and int(g) == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
    return sum(scores) / len(scores)


def confusion_matrix(
    predicted: Sequence[int], gold: Sequence[int], *, normalise: bool = True
) -> Any:
    """Rows = gold, columns = predicted, over the full 19-answer space."""
    import numpy as np

    matrix = np.zeros((N_ANSWERS, N_ANSWERS), dtype=np.int64)
    for p, g in zip(predicted, gold):
        matrix[int(g), int(p)] += 1
    if not normalise:
        return matrix
    totals = matrix.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(totals > 0, matrix / totals, 0.0)


def per_type_accuracy(
    records: Sequence[ChangeVQARecord],
    predicted: Sequence[int],
) -> dict[str, dict[str, float]]:
    """Accuracy and support for each of the eight question types.

    Types with no rows are reported with support 0 and accuracy None rather than
    0.0: a type that was never asked has no accuracy, and printing 0.0 would
    read as total failure.
    """
    buckets: dict[str, list[tuple[int, int]]] = {
        qtype: [] for qtype in QUESTION_TYPE_ORDER
    }
    for record, prediction in zip(records, predicted):
        buckets.setdefault(record.qtype, []).append(
            (int(prediction), record.answer_index)
        )
    out: dict[str, dict[str, float]] = {}
    for qtype, pairs in buckets.items():
        if not pairs:
            out[qtype] = {"support": 0, "accuracy": None}
            continue
        hits = sum(1 for p, g in pairs if p == g)
        out[qtype] = {
            "support": len(pairs),
            "accuracy": round(hits / len(pairs), 6),
        }
    return out


def majority_baselines(
    records: Sequence[ChangeVQARecord],
    gold: Sequence[int],
) -> dict[str, Any]:
    """The majority-class baselines, computed so they are TRACEABLE.

    WHY THIS EXISTS (provenance defect §6c)
    ---------------------------------------
    The published R-02 table quotes a `global-majority` baseline and a
    `per-type-majority` baseline. Until now neither was written into any export:
    the strings `per-type-majority`, `global-majority`, `0.5084` and `0.4560`
    appear nowhere in `eval_summary.json`, and the per-type figures were
    screenshot-only. A reader therefore had no way to check them, which is the
    one thing a baseline must support.

    Both are computed from the GOLD labels alone -- no model prediction enters --
    so they are exactly reproducible from the export and cannot drift when the
    model changes.

    DEFINITIONS, stated because "majority baseline" is ambiguous:

        global-majority   predict the single most frequent answer in the split,
                          for every question. This is the weakest baseline and
                          the one quoted as `global-majority` in the report.

        per-type-majority the single most frequent answer WITHIN each question
                          type, applied to every question of that type. This is
                          strictly stronger than global-majority, because the
                          question type is itself information the system is given.

    Both are computed with ties broken by the LOWEST answer index, which is
    deterministic and independent of dict iteration order.
    """
    from collections import Counter

    gold_list = [int(g) for g in gold]
    n = len(gold_list)

    # A length mismatch must be rejected, not silently truncated. `zip` below
    # stops at the shorter sequence, so a short `gold` would quietly compute the
    # baseline over an UNSTATED subset -- a confident-looking number derived from
    # the wrong population. Found by test_a_mismatched_length_is_rejected.
    if len(records) != n:
        raise ValueError(
            f"records and gold must align: {len(records)} records vs {n} gold "
            "labels. A baseline computed over a truncated pair would not "
            "describe the population it names."
        )

    if n == 0:
        return {
            "n": 0,
            "global_majority": None,
            "per_type_majority": None,
            "note": "no scored samples",
        }

    # -- global ---------------------------------------------------------------
    overall = Counter(gold_list)
    top_count = max(overall.values())
    global_answer = min(a for a, c in overall.items() if c == top_count)
    global_baseline = top_count / n

    # -- per type -------------------------------------------------------------
    per_type_counts: dict[str, Counter] = {
        qtype: Counter() for qtype in QUESTION_TYPE_ORDER
    }
    for record, answer in zip(records, gold_list):
        per_type_counts.setdefault(record.qtype, Counter())[answer] += 1

    per_type_detail: dict[str, Any] = {}
    hits = 0
    for qtype, counts in per_type_counts.items():
        if not counts:
            per_type_detail[qtype] = {
                "support": 0,
                "majority_answer": None,
                "majority_count": 0,
                "majority_share": None,
            }
            continue
        support = sum(counts.values())
        best = max(counts.values())
        answer = min(a for a, c in counts.items() if c == best)
        hits += best
        per_type_detail[qtype] = {
            "support": support,
            "majority_answer": answer,
            "majority_answer_text": (
                INDEX_TO_ANSWER[answer] if 0 <= answer < N_ANSWERS else None
            ),
            "majority_count": best,
            "majority_share": round(best / support, 6),
        }

    return {
        "n": n,
        "global_majority": round(global_baseline, 6),
        "global_majority_answer": global_answer,
        "global_majority_answer_text": (
            INDEX_TO_ANSWER[global_answer] if 0 <= global_answer < N_ANSWERS else None
        ),
        "global_majority_count": top_count,
        "per_type_majority": round(hits / n, 6),
        "per_type": per_type_detail,
        "gold_histogram": {
            INDEX_TO_ANSWER[a] if 0 <= a < N_ANSWERS else str(a): c
            for a, c in sorted(overall.items())
        },
        "tie_break": "lowest answer index",
        "derivation": (
            "Computed from gold labels only; no prediction enters. Reproducible "
            "from this export without re-running the model."
        ),
        "caveat": (
            "A majority baseline is a property of the split's label "
            "distribution, not of the task. It is quoted for context and is NOT "
            "an official metric."
        ),
    }


def estimator_errors(
    predicted_mag: Any,
    predicted_delta: Any,
    predicted_total: Any,
    targets: Sequence[SceneTargets],
) -> dict[str, Any]:
    """Mean absolute error of the class-wise change estimate.

    This is the only place the label maps are used at evaluation time, and it is
    a *diagnostic*: the deliverable is the answer. Reporting it makes the
    two-stage architecture auditable — if the answers are poor, this says
    whether the change estimate or the answer head is at fault.
    """
    import numpy as np

    mag = np.asarray(predicted_mag, dtype=np.float64)
    delta = np.asarray(predicted_delta, dtype=np.float64)
    total = np.asarray(predicted_total, dtype=np.float64)
    gold_mag = np.asarray([t.class_mag for t in targets], dtype=np.float64)
    gold_delta = np.asarray([t.class_delta for t in targets], dtype=np.float64)
    gold_total = np.asarray([t.total_changed for t in targets], dtype=np.float64)

    per_class = {
        name: round(float(np.abs(mag[:, i] - gold_mag[:, i]).mean()), 6)
        for i, name in enumerate(CHANGE_CLASS_ORDER)
    }
    return {
        "class_magnitude_mae": round(float(np.abs(mag - gold_mag).mean()), 6),
        "class_magnitude_mae_per_class": per_class,
        "class_delta_mae": round(float(np.abs(delta - gold_delta).mean()), 6),
        "total_changed_mae": round(float(np.abs(total - gold_total).mean()), 6),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class EvaluationReport:
    """A complete, self-describing evaluation result."""

    payload: dict[str, Any]

    @property
    def accuracy(self) -> float:
        return float(self.payload["metrics"]["answer_accuracy"])

    def to_dict(self) -> dict[str, Any]:
        return self.payload

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return p

    def canonical_sha256(self) -> str:
        """SHA256 of the CANONICAL COMPACT serialisation of the payload.

        This is a content digest: it is stable across reformatting, key order and
        whitespace, so two reports with the same content hash equal even if one
        was written with `indent=2` and the other with `indent=4`.

        It is **not** the hash of any file on disk. See `file_bytes_sha256`.

        WHY THE NAME CHANGED (provenance defect §6a)
        --------------------------------------------
        This method used to be called `sha256()` and its result was written to a
        key named `report_sha256`. `write()` emits `indent=2`, so a reviewer who
        hashed the written file with `sha256sum` got a DIFFERENT value and
        reasonably concluded the report had been corrupted or edited after
        generation. Both hashes were correct; the name was wrong.

        The old name is still accepted as an alias so existing callers and the
        recorded `eval_summary.json` files remain readable -- but the value is
        now also written under an unambiguous key. A value that no longer claims
        to be something it is not.
        """
        blob = json.dumps(self.payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()

    #: Backwards-compatible alias. Kept so a caller written against the old name
    #: keeps working; new code should call `canonical_sha256` and read
    #: `report_canonical_sha256`.
    sha256 = canonical_sha256

    def file_bytes_sha256(self, path: str | Path) -> str:
        """SHA256 of the report's BYTES on disk.

        This is the value that matches `sha256sum <file>`, and therefore the one
        a reviewer can independently reproduce. It requires the report to have
        been written first, which is why it is a method taking a path rather than
        a property.
        """
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()


def evaluate_records(
    records: Sequence[ChangeVQARecord],
    predictions: dict[str, Any],
    *,
    targets: Sequence[SceneTargets],
    split: str,
    checkpoint_path: str | Path | None,
    checkpoint_sha256: str | None,
    config_hash: str | None,
    inference_config: dict[str, Any],
    n_failed: int = 0,
    n_skipped: int = 0,
) -> EvaluationReport:
    """Assemble the full report. Counts are explicit, never inferred."""
    import numpy as np

    gold = [record.answer_index for record in records]
    predicted = [int(v) for v in predictions["answer_index"]]
    probabilities = predictions.get("probabilities")

    metrics: dict[str, Any] = {
        "answer_accuracy": round(answer_accuracy(predicted, gold), 6),
        "macro_f1": round(macro_f1(predicted, gold), 6),
        "per_type": per_type_accuracy(records, predicted),
        "n_answers_in_gold": len({int(g) for g in gold}),
    }
    if probabilities is not None:
        metrics["top_3_accuracy"] = round(
            top_k_accuracy(probabilities, gold, k=3), 6
        )
    if targets:
        metrics["estimator"] = estimator_errors(
            predictions["class_mag"],
            predictions["class_delta"],
            predictions["total_changed"],
            targets,
        )
    metrics["mean_confidence"] = (
        round(float(np.mean(predictions["confidence"])), 6)
        if len(predictions["confidence"])
        else 0.0
    )

    baselines = majority_baselines(records, gold)

    return EvaluationReport(
        payload={
            "metric_implementation": METRIC_IMPLEMENTATION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "dataset_id": DATASET_ID,
                "split": split,
                "n_samples": len(records),
                "n_unique_scenes": len({r.file_name for r in records}),
                "preprocessing_version": PREPROCESSING_VERSION,
                "feature_spec": FEATURE_SPEC,
            },
            "checkpoint": {
                "path": str(checkpoint_path) if checkpoint_path else None,
                "sha256": checkpoint_sha256 or "unavailable",
            },
            "config_hash": config_hash or "unavailable",
            "inference_config": inference_config,
            "counts": {
                "n_scored": len(records),
                "n_failed": int(n_failed),
                "n_skipped": int(n_skipped),
            },
            "metrics": metrics,
            # §6c: the baselines are exported, not left to a screenshot.
            "baselines": baselines,
            "confusion_matrix": confusion_matrix(
                predicted, gold, normalise=False
            ).tolist(),
            "confusion_matrix_normalised": confusion_matrix(
                predicted, gold, normalise=True
            ).round(6).tolist(),
            "answer_vocabulary": list(INDEX_TO_ANSWER),
            "metric_provenance": {
                # Which metrics are RECONSTRUCTIBLE from this export and which
                # are not. A confusion matrix yields accuracy and macro-F1; it
                # cannot yield top-3 or a mean confidence, because both need the
                # per-sample score vector. Saying so here stops a later reader
                # from trying to re-derive them and concluding the report is
                # inconsistent when the numbers do not fall out.
                "answer_derived": [
                    "answer_accuracy",
                    "macro_f1",
                    "per_type",
                    "confusion_matrix",
                    "confusion_matrix_normalised",
                ],
                "recorded_only": [
                    "top_3_accuracy",
                    "mean_confidence",
                ],
                "recorded_only_reason": (
                    "top-3 needs the full per-sample (19,) score vector and "
                    "mean_confidence needs the per-sample chosen probability; a "
                    "confusion matrix cannot reproduce either."
                ),
                "baseline_derived": [
                    "global_majority",
                    "per_type_majority",
                ],
                "baseline_derived_reason": (
                    "Both come from the gold labels alone, so they are "
                    "reconstructible from this export."
                ),
                "confidence_method": (
                    inference_config.get("confidence_method", "uncalibrated")
                ),
                "calibration_note": (
                    "These confidences are the raw softmax of the chosen answer. "
                    "Nothing is fitted at evaluation time. Whether a fitted "
                    "temperature should be applied when REPORTING this metric is "
                    "a separate, open question (R-03); the artifact that would "
                    "supply it is produced by scripts/fit_calibration.py and is "
                    "NOT applied here, so this number stays comparable to the "
                    "already-published figures."
                ),
            },
            "note": (
                "Classification metrics only. This task does not generate "
                "language, so BLEU/ROUGE-L/CIDEr/BERTScore are deliberately "
                "not reported (plan section 2.2 lists them under CAPTIONING; "
                "R-16 governs them and is open)."
            ),
        }
    )


# ---------------------------------------------------------------------------
# Qualitative examples
# ---------------------------------------------------------------------------


def qualitative_examples(
    records: Sequence[ChangeVQARecord],
    predictions: dict[str, Any],
    *,
    targets: Sequence[SceneTargets] | None = None,
    n_success: int = 5,
    n_failure: int = 5,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Representative successes AND failures, never successes alone.

    A qualitative section that shows only correct predictions is a marketing
    document. Both are returned, failures first in the payload so a reader who
    stops after one entry reads a failure.

    Each example carries the two image paths, their identity, the question, the
    gold answer, the prediction, the raw confidence and the class-wise change
    estimate — the facts a reviewer needs to judge the case without re-running
    anything.
    """
    target_by_file = (
        {t.file_name: t for t in targets} if targets is not None else {}
    )
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        prediction = int(predictions["answer_index"][index])
        correct = prediction == record.answer_index
        entry: dict[str, Any] = {
            "question_id": record.question_id,
            "split": record.split,
            "scene_key": record.scene_key,
            "file_name": record.file_name,
            "t1_path": record.t1_path,
            "t2_path": record.t2_path,
            "temporal_reference": record.temporal_ref,
            "qtype": record.qtype,
            "question": record.question,
            "gold_answer": record.answer,
            "predicted_answer": INDEX_TO_ANSWER[prediction],
            "correct": correct,
            "raw_confidence": round(float(predictions["confidence"][index]), 6),
            "raw_margin": round(float(predictions["margin"][index]), 6),
            "confidence_is_calibrated": False,
            "class_change_magnitude": {
                name: round(float(predictions["class_mag"][index][i]), 6)
                for i, name in enumerate(CHANGE_CLASS_ORDER)
            },
            "class_change_delta": {
                name: round(float(predictions["class_delta"][index][i]), 6)
                for i, name in enumerate(CHANGE_CLASS_ORDER)
            },
            "total_changed_estimate": round(
                float(predictions["total_changed"][index]), 6
            ),
        }
        scene_targets = target_by_file.get(record.file_name)
        if scene_targets is not None:
            entry["label_derived_targets"] = {
                "class_change_magnitude": {
                    name: round(scene_targets.class_mag[i], 6)
                    for i, name in enumerate(CHANGE_CLASS_ORDER)
                },
                "total_changed": round(scene_targets.total_changed, 6),
                "label1_sha256": scene_targets.label1_sha256,
                "label2_sha256": scene_targets.label2_sha256,
            }
        (successes if correct else failures).append(entry)
        if (
            max_examples is not None
            and len(successes) >= n_success
            and len(failures) >= n_failure
        ):
            break

    return {
        "n_scored": len(records),
        "n_correct": len(successes),
        "n_incorrect": len(failures),
        "note": (
            "Failures are listed first and are not filtered. Both sets are "
            "truncated to the requested counts from the full scored set, in "
            "dataset order."
        ),
        "failures": failures[:n_failure],
        "successes": successes[:n_success],
    }


__all__ = [
    "METRIC_IMPLEMENTATION",
    "EvaluationReport",
    "answer_accuracy",
    "top_k_accuracy",
    "macro_f1",
    "confusion_matrix",
    "per_type_accuracy",
    "estimator_errors",
    "evaluate_records",
    "qualitative_examples",
]
