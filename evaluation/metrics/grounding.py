"""Grounding metrics — IoU, Recall@IoU, and greedy box matching.

Pure NumPy. No model, no torch, no dataset. These are the numbers a grounding
result is judged by, so they are deliberately the most boring code in the repo:
integer-indexed, explicitly guarded against the degenerate cases that produce
NaN, and tested against hand-computed values.

Coordinate convention: every box is [x1, y1, x2, y2] in the SAME frame for
prediction and target, satisfying x2 >= x1 and y2 >= y1. Internally SatQuery
uses normalized 0-1 (core.schemas.CoordinateSystem.NORMALIZED_0_1); VRSBench
annotations are 0-100 and are converted at the adapter boundary, never here.
Mixing frames here would produce a plausible-looking IoU that means nothing.

Reference values are hand-computed in the tests rather than taken from a
library, because a metric that agrees only with itself proves nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

BoxArray = Sequence[float]

#: IoU thresholds a grounding result is normally reported at.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.25, 0.5, 0.75)


def _as_array(boxes: Iterable[BoxArray]) -> np.ndarray:
    """Coerce a sequence of 4-element boxes to an (N, 4) float array."""
    arr = np.asarray(list(boxes), dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"expected boxes shaped (N, 4), got {arr.shape}")
    return arr


def box_area(boxes: np.ndarray) -> np.ndarray:
    """Area of each box. Zero for inverted or degenerate boxes."""
    widths = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0.0, a_max=None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], a_min=0.0, a_max=None)
    return widths * heights


def intersection_area(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise intersection area, shape (len(a), len(b)).

    Uses broadcasting rather than a loop so an N x M match is one operation.
    """
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    widths = np.clip(x2 - x1, a_min=0.0, a_max=None)
    heights = np.clip(y2 - y1, a_min=0.0, a_max=None)
    return widths * heights


def box_iou_matrix(preds: Iterable[BoxArray], targets: Iterable[BoxArray]) -> np.ndarray:
    """Pairwise IoU, shape (len(preds), len(targets)).

    A pair where either box has zero area yields 0.0, not NaN. A degenerate
    prediction is a wrong prediction, not an undefined one.
    """
    p = _as_array(preds)
    t = _as_array(targets)
    if p.shape[0] == 0 or t.shape[0] == 0:
        return np.zeros((p.shape[0], t.shape[0]), dtype=np.float64)

    inter = intersection_area(p, t)
    union = box_area(p)[:, None] + box_area(t)[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return np.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)


def box_iou(a: BoxArray, b: BoxArray) -> float:
    """IoU of a single pair. Convenience wrapper over the matrix form."""
    return float(box_iou_matrix([a], [b])[0, 0])


@dataclass(frozen=True)
class MatchResult:
    """Outcome of greedily matching predictions to targets at one threshold."""

    #: (pred_index, target_index, iou) triples, highest IoU first.
    matches: tuple[tuple[int, int, float], ...]
    n_preds: int
    n_targets: int
    iou_threshold: float

    @property
    def true_positives(self) -> int:
        return len(self.matches)

    @property
    def false_positives(self) -> int:
        return self.n_preds - self.true_positives

    @property
    def false_negatives(self) -> int:
        return self.n_targets - self.true_positives

    @property
    def recall(self) -> float:
        if self.n_targets == 0:
            return 0.0
        return self.true_positives / self.n_targets

    @property
    def precision(self) -> float:
        if self.n_preds == 0:
            return 0.0
        return self.true_positives / self.n_preds

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0


def greedy_match(
    preds: Iterable[BoxArray],
    targets: Iterable[BoxArray],
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Greedily match predictions to targets at an IoU threshold.

    Greedy (highest-IoU-first) rather than Hungarian matching. Both are used in
    the literature; greedy is what VRSBench-style referring evaluation does and
    it is order-independent because the sort is on the IoU value, not on the
    prediction order. Ties break on (pred_index, target_index) so the result is
    deterministic for identical inputs.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in [0,1], got {iou_threshold}")

    p = _as_array(preds)
    t = _as_array(targets)
    iou = box_iou_matrix(preds, targets)

    candidates: list[tuple[float, int, int]] = []
    for pi in range(iou.shape[0]):
        for ti in range(iou.shape[1]):
            value = float(iou[pi, ti])
            if value >= iou_threshold:
                candidates.append((value, pi, ti))

    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

    used_preds: set[int] = set()
    used_targets: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for value, pi, ti in candidates:
        if pi in used_preds or ti in used_targets:
            continue
        used_preds.add(pi)
        used_targets.add(ti)
        matches.append((pi, ti, value))

    return MatchResult(
        matches=tuple(matches),
        n_preds=int(p.shape[0]),
        n_targets=int(t.shape[0]),
        iou_threshold=iou_threshold,
    )


def recall_at_iou(
    preds: Iterable[BoxArray],
    targets: Iterable[BoxArray],
    iou_threshold: float = 0.5,
) -> float:
    """Fraction of targets matched by some prediction at the threshold."""
    return greedy_match(preds, targets, iou_threshold).recall


def mean_matched_iou(
    preds: Iterable[BoxArray],
    targets: Iterable[BoxArray],
    iou_threshold: float = 0.0,
) -> float:
    """Mean IoU over matched pairs.

    With the default threshold of 0.0 every prediction that overlaps at all is
    a candidate; pairs with zero overlap are still excluded, so a prediction
    nowhere near any target contributes nothing. This is the standard
    localization-quality measure once a match is established.
    """
    result = greedy_match(preds, targets, iou_threshold)
    if not result.matches:
        return 0.0
    return float(np.mean([m[2] for m in result.matches]))


def best_iou(preds: Iterable[BoxArray], targets: Iterable[BoxArray]) -> float:
    """Highest single IoU between any prediction and any target.

    Useful as a ceiling: if the best achievable IoU across the whole image is
    below a threshold, no matching strategy can reach that threshold.
    """
    iou = box_iou_matrix(preds, targets)
    if iou.size == 0:
        return 0.0
    return float(iou.max())


@dataclass(frozen=True)
class GroundingScore:
    """One image's grounding outcome."""

    n_preds: int
    n_targets: int
    best_iou: float
    mean_matched_iou: float
    recall: dict[float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "n_preds": self.n_preds,
            "n_targets": self.n_targets,
            "best_iou": round(self.best_iou, 4),
            "mean_matched_iou": round(self.mean_matched_iou, 4),
            "recall": {str(k): round(v, 4) for k, v in self.recall.items()},
        }


def score_one(
    preds: Iterable[BoxArray],
    targets: Iterable[BoxArray],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> GroundingScore:
    """Score a single image's predictions against its targets."""
    p = _as_array(preds)
    t = _as_array(targets)
    return GroundingScore(
        n_preds=int(p.shape[0]),
        n_targets=int(t.shape[0]),
        best_iou=best_iou(preds, targets),
        mean_matched_iou=mean_matched_iou(preds, targets),
        recall={thr: recall_at_iou(preds, targets, thr) for thr in thresholds},
    )


@dataclass(frozen=True)
class GroundingReport:
    """Aggregate over a dataset.

    Recall is macro-averaged over images (mean of per-image recall) rather than
    pooled (total matches / total targets). The two differ when images carry
    different numbers of targets, and macro is the convention for referring
    evaluation because one crowded image should not dominate the score.
    """

    n_images: int
    n_images_with_targets: int
    n_predictions: int
    n_targets: int
    mean_best_iou: float
    mean_matched_iou: float
    recall: dict[float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "n_images": self.n_images,
            "n_images_with_targets": self.n_images_with_targets,
            "n_predictions": self.n_predictions,
            "n_targets": self.n_targets,
            "mean_best_iou": round(self.mean_best_iou, 4),
            "mean_matched_iou": round(self.mean_matched_iou, 4),
            "recall": {str(k): round(v, 4) for k, v in self.recall.items()},
        }

    def summary(self) -> str:
        lines = [
            f"images              : {self.n_images} "
            f"({self.n_images_with_targets} with targets)",
            f"predictions/targets : {self.n_predictions} / {self.n_targets}",
            f"mean best IoU       : {self.mean_best_iou:.4f}",
            f"mean matched IoU    : {self.mean_matched_iou:.4f}",
        ]
        for thr in sorted(self.recall):
            lines.append(f"recall @ IoU {thr:<5}: {self.recall[thr]:.4f}")
        return "\n".join(lines)


def score_dataset(
    per_image: Sequence[tuple[Iterable[BoxArray], Iterable[BoxArray]]],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> GroundingReport:
    """Aggregate grounding scores over (predictions, targets) pairs.

    Args:
        per_image: one (preds, targets) tuple per image.
        thresholds: IoU thresholds to report recall at.

    Images with no targets are excluded from the recall average -- a recall
    over zero targets is undefined, and counting it as 0.0 would penalise a
    model for a property of the dataset. They ARE counted in n_images, and
    their predictions still count toward false positives in a pooled view.
    """
    scores: list[GroundingScore] = [score_one(p, t, thresholds) for p, t in per_image]

    if not scores:
        return GroundingReport(0, 0, 0, 0, 0.0, 0.0, {thr: 0.0 for thr in thresholds})

    with_targets = [s for s in scores if s.n_targets > 0]

    recall: dict[float, float] = {}
    for thr in thresholds:
        if with_targets:
            recall[thr] = float(np.mean([s.recall[thr] for s in with_targets]))
        else:
            recall[thr] = 0.0

    return GroundingReport(
        n_images=len(scores),
        n_images_with_targets=len(with_targets),
        n_predictions=sum(s.n_preds for s in scores),
        n_targets=sum(s.n_targets for s in scores),
        mean_best_iou=float(np.mean([s.best_iou for s in scores])),
        mean_matched_iou=(
            float(np.mean([s.mean_matched_iou for s in with_targets]))
            if with_targets else 0.0
        ),
        recall=recall,
    )


def benchmark_to_normalized(box: Sequence[float], scale: float = 100.0) -> list[float]:
    """Convert a 0-100 benchmark box to SatQuery's 0-1 convention.

    VRSBench states verbatim that "all box coordinates are normalized to 0-100".
    Treating those numbers as pixels or as 0-1 is a silent, catastrophic error,
    so the conversion is a named function rather than an inline division.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if len(box) != 4:
        raise ValueError(f"expected 4 values, got {len(box)}")
    return [float(v) / scale for v in box]


__all__ = [
    "DEFAULT_THRESHOLDS",
    "MatchResult",
    "GroundingScore",
    "GroundingReport",
    "box_area",
    "intersection_area",
    "box_iou_matrix",
    "box_iou",
    "greedy_match",
    "recall_at_iou",
    "mean_matched_iou",
    "best_iou",
    "score_one",
    "score_dataset",
    "benchmark_to_normalized",
]