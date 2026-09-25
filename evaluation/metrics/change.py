"""Change-detection metrics — precision, recall, F1, IoU, mIoU.

Pure NumPy. No model, no torch, no dataset. These are the numbers a change
detector is judged by, so they are the most boring code in the repo:
boolean-array arithmetic, every division guarded, tested against
hand-computed values.

TWO AGGREGATION CONVENTIONS, BOTH REPORTED
------------------------------------------
LEVIR-CD papers typically report POOLED metrics — total TP/FP/FN summed
across every test tile, then one precision/recall/F1. Some venues report
MACRO metrics — per-tile metrics averaged over tiles.

They differ when tile sizes or change fractions differ, and they can
disagree by several points. Rather than pick one and hope, both are computed
and labelled. Whichever the benchmark specifies can then be read directly
instead of re-derived.

    pooled  : sum the confusion counts, then compute
    macro   : compute per tile, then average

An empty-change tile (all background) contributes a recall of 0.0 under the
naive macro average, which drags the number down for a property of the
dataset rather than of the model. Such tiles are excluded from the macro
average and counted separately, exactly as `score_dataset` in the grounding
metrics excludes target-less images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

#: Default probability threshold. A config value, not a tuned one.
DEFAULT_THRESHOLD = 0.5


def _as_bool_mask(mask: Any) -> np.ndarray:
    """Coerce to a 2-D boolean array. Accepts bool, 0/1 ints, or probabilities."""
    arr = np.asarray(mask)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {arr.shape}")
    if arr.dtype == np.bool_:
        return arr
    return arr > 0.5


def binarize(prob: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Threshold a probability map into a boolean mask.

    `>=` rather than `>`: a pixel at exactly the threshold is a change. The
    asymmetry matters for reproducibility, so it is pinned by a test.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0,1], got {threshold}")
    return np.asarray(prob) >= threshold


@dataclass(frozen=True)
class ConfusionCounts:
    """Raw counts. Everything else is derived, so there is one source of truth."""

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def n_pixels(self) -> int:
        return (
            self.true_positive + self.false_positive
            + self.false_negative + self.true_negative
        )

    @property
    def n_positive_true(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def n_positive_pred(self) -> int:
        return self.true_positive + self.false_positive

    def __add__(self, other: "ConfusionCounts") -> "ConfusionCounts":
        return ConfusionCounts(
            self.true_positive + other.true_positive,
            self.false_positive + other.false_positive,
            self.false_negative + other.false_negative,
            self.true_negative + other.true_negative,
        )


def confusion(y_true: Any, y_pred: Any) -> ConfusionCounts:
    """Confusion counts for two boolean masks of identical shape."""
    t = _as_bool_mask(y_true)
    p = _as_bool_mask(y_pred)
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {p.shape}")

    tp = int(np.sum(t & p))
    fp = int(np.sum(~t & p))
    fn = int(np.sum(t & ~p))
    tn = int(np.sum(~t & ~p))
    return ConfusionCounts(tp, fp, fn, tn)


@dataclass(frozen=True)
class ChangeScores:
    """Metrics for one mask, or for a pooled set of masks.

    `iou` is the IoU of the CHANGE class. `miou` averages the change and
    background IoU, which is the more common reporting convention in
    segmentation literature and is less pessimistic when change is rare.
    """

    precision: float
    recall: float
    f1: float
    iou: float
    miou: float
    counts: ConfusionCounts

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "iou": round(self.iou, 4),
            "miou": round(self.miou, 4),
            "tp": self.counts.true_positive,
            "fp": self.counts.false_positive,
            "fn": self.counts.false_negative,
            "tn": self.counts.true_negative,
            "n_pixels": self.counts.n_pixels,
        }


def _safe_div(numerator: float, denominator: float) -> float:
    """Division that returns 0.0 rather than NaN when the denominator is zero.

    A tile with no predicted change and no true change has precision 0/0. That
    is "nothing was asserted and nothing was wrong", which is not the same as
    "everything was wrong" — but NaN poisons any average it enters. 0.0 is the
    honest choice here because it is the *no-information* value, and callers
    that care about the distinction can read `counts` directly.
    """
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def scores_from_counts(counts: ConfusionCounts) -> ChangeScores:
    """Derive every metric from raw counts."""
    tp, fp, fn, tn = (
        counts.true_positive, counts.false_positive,
        counts.false_negative, counts.true_negative,
    )

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    iou_change = _safe_div(tp, tp + fp + fn)
    iou_background = _safe_div(tn, tn + fp + fn)
    miou = 0.5 * (iou_change + iou_background)

    return ChangeScores(
        precision=precision,
        recall=recall,
        f1=f1,
        iou=iou_change,
        miou=miou,
        counts=counts,
    )


def score_one(
    y_true: Any, y_pred: Any, threshold: float = DEFAULT_THRESHOLD
) -> ChangeScores:
    """Metrics for one mask pair. `y_pred` may be a probability map."""
    t = _as_bool_mask(y_true)
    if np.asarray(y_pred).dtype == np.bool_:
        p = _as_bool_mask(y_pred)
    else:
        p = binarize(np.asarray(y_pred), threshold)
    return scores_from_counts(confusion(t, p))


@dataclass
class ChangeReport:
    """Aggregate over a dataset, under both conventions."""

    n_images: int
    n_images_with_change: int
    pooled: ChangeScores
    macro: ChangeScores
    per_image_change_fraction: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_images": self.n_images,
            "n_images_with_change": self.n_images_with_change,
            "pooled": self.pooled.to_dict(),
            "macro": self.macro.to_dict(),
            "mean_change_fraction": (
                round(float(np.mean(self.per_image_change_fraction)), 4)
                if self.per_image_change_fraction else 0.0
            ),
        }

    def summary(self) -> str:
        lines = [
            f"images              : {self.n_images} "
            f"({self.n_images_with_change} with change)",
            "",
            f"  {'':<10}{'precision':>11}{'recall':>9}{'f1':>9}{'iou':>9}{'miou':>9}",
        ]
        for label, s in (("pooled", self.pooled), ("macro", self.macro)):
            lines.append(
                f"  {label:<10}{s.precision:>11.4f}{s.recall:>9.4f}"
                f"{s.f1:>9.4f}{s.iou:>9.4f}{s.miou:>9.4f}"
            )
        if self.per_image_change_fraction:
            lines.append("")
            lines.append(
                f"mean change fraction : "
                f"{float(np.mean(self.per_image_change_fraction)):.4f}"
            )
        return "\n".join(lines)


def score_dataset(
    pairs: Sequence[tuple[Any, Any]],
    threshold: float = DEFAULT_THRESHOLD,
) -> ChangeReport:
    """Aggregate over (y_true, y_pred) pairs.

    Args:
        pairs: one (ground_truth_mask, predicted_probability_or_mask) per tile.
        threshold: applied to probability predictions.

    Returns:
        ChangeReport carrying BOTH the pooled and macro conventions.

    Note on macro aggregation: tiles whose ground truth has no change at all
    are excluded from the macro average. Including them would score recall 0.0
    for a property of the dataset (many LEVIR-CD tiles are unchanged) rather
    than of the model.
    """
    if not pairs:
        empty = scores_from_counts(ConfusionCounts(0, 0, 0, 0))
        return ChangeReport(0, 0, empty, empty, [])

    pooled = ConfusionCounts(0, 0, 0, 0)
    per_image: list[ChangeScores] = []
    fractions: list[float] = []
    n_with_change = 0

    for y_true, y_pred in pairs:
        t = _as_bool_mask(y_true)
        if np.asarray(y_pred).dtype == np.bool_:
            p = _as_bool_mask(y_pred)
        else:
            p = binarize(np.asarray(y_pred), threshold)

        c = confusion(t, p)
        pooled = pooled + c

        frac = _safe_div(c.n_positive_true, c.n_pixels)
        fractions.append(frac)
        if c.n_positive_true > 0:
            n_with_change += 1
            per_image.append(scores_from_counts(c))

    if per_image:
        macro = ChangeScores(
            precision=float(np.mean([s.precision for s in per_image])),
            recall=float(np.mean([s.recall for s in per_image])),
            f1=float(np.mean([s.f1 for s in per_image])),
            iou=float(np.mean([s.iou for s in per_image])),
            miou=float(np.mean([s.miou for s in per_image])),
            counts=pooled,
        )
    else:
        macro = scores_from_counts(pooled)

    return ChangeReport(
        n_images=len(pairs),
        n_images_with_change=n_with_change,
        pooled=scores_from_counts(pooled),
        macro=macro,
        per_image_change_fraction=fractions,
    )


__all__ = [
    "DEFAULT_THRESHOLD",
    "ConfusionCounts",
    "ChangeScores",
    "ChangeReport",
    "binarize",
    "confusion",
    "scores_from_counts",
    "score_one",
    "score_dataset",
]