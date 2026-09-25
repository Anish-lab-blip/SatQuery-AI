"""Calibration quality metrics — ECE, NLL, reliability diagram.

WHAT THESE ARE FOR
------------------
A fitted temperature is a claim that confidence now means something. These
metrics are how that claim gets checked rather than asserted. They are REPORTED,
never thresholded: the plan specifies no calibration quality bar, so this module
computes numbers and stops. Deciding whether `ECE = 0.04` is acceptable is a
maintainer ruling, exactly like the R-02 accuracy / macro-F1 question.

DEFINITIONS, WRITTEN OUT BECAUSE THEY GET SILENTLY REDEFINED
------------------------------------------------------------
Expected Calibration Error (equal-width binning, the standard formulation):

    ECE = sum_over_bins  (n_bin / N) * |accuracy_in_bin - mean_confidence_in_bin|

Confidence here is the **predicted-class probability** (`max_c p_c`), and accuracy
is the fraction of those argmax predictions that are correct. This is the
top-1 confidence view, which is the one a user of a single-answer system
actually experiences.

Mean NLL:

    NLL = -(1/N) * sum_i log p_i[gold_i]

Lower is better for both. ECE lower-bounds nothing and is sensitive to the bin
count; that limitation is stated in the artifact rather than papered over.

WHAT THESE METRICS MUST NOT BE USED FOR
---------------------------------------
They are not an aggregate score. `evaluation/normalize.py` refuses to aggregate
while `official_weights is None`, and nothing here changes that: ECE and NLL are
diagnostics on a confidence mapping, not a task metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

#: Equal-width bin count for ECE. 15 is the value used across the calibration
#: literature (Guo et al. 2017); it is recorded in the artifact so a reader is
#: never guessing which binning produced a reported ECE.
DEFAULT_N_BINS = 15


def _as_arrays(probabilities: Any, targets: Sequence[int]) -> tuple[Any, Any]:
    import numpy as np

    probs = np.asarray(probabilities, dtype=np.float64)
    gold = np.asarray(list(targets), dtype=np.int64)
    if probs.ndim != 2:
        raise ValueError(f"probabilities must be 2-D (N, C), got {probs.shape}")
    if len(gold) != probs.shape[0]:
        raise ValueError(
            f"probabilities has {probs.shape[0]} rows but {len(gold)} targets"
        )
    if len(gold) == 0:
        raise ValueError("no samples")
    return probs, gold


def expected_calibration_error(
    probabilities: Any,
    targets: Sequence[int],
    *,
    n_bins: int = DEFAULT_N_BINS,
) -> float:
    """Equal-width ECE over the predicted-class confidence.

    Args:
        probabilities: `(N, C)` probabilities (post-softmax, possibly calibrated).
        targets: `N` gold class indices.
        n_bins: number of equal-width bins on `[0, 1]`.

    Returns:
        The ECE as a float in `[0, 1]`.

    Raises:
        ValueError: on malformed input, or `n_bins < 1`.
    """
    import numpy as np

    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    probs, gold = _as_arrays(probabilities, targets)

    confidence = np.max(probs, axis=-1)
    predicted = np.argmax(probs, axis=-1)
    correct = (predicted == gold).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = np.float64(len(gold))
    ece = np.float64(0.0)
    for index in range(n_bins):
        lo, hi = edges[index], edges[index + 1]
        # Half-open [lo, hi) so each sample lands in exactly one bin; the final
        # bin is closed so a confidence of exactly 1.0 is not dropped.
        if index == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        bin_acc = float(np.mean(correct[mask]))
        bin_conf = float(np.mean(confidence[mask]))
        ece += (count / total) * abs(bin_acc - bin_conf)
    return float(ece)


def mean_negative_log_likelihood(
    probabilities: Any,
    targets: Sequence[int],
    *,
    eps: float = 1e-12,
) -> float:
    """Mean NLL of the gold class under the given probabilities.

    Clamped at `eps` so a zero probability cannot produce an infinite score.

    Raises:
        ValueError: on malformed input.
    """
    import numpy as np

    probs, gold = _as_arrays(probabilities, targets)
    picked = probs[np.arange(len(gold)), gold]
    return float(-np.mean(np.log(np.clip(picked, eps, 1.0))))


def reliability_diagram(
    probabilities: Any,
    targets: Sequence[int],
    *,
    n_bins: int = DEFAULT_N_BINS,
) -> dict[str, Any]:
    """Per-bin accuracy vs confidence, the data behind an ECE number.

    An ECE alone is unfalsifiable -- two very different miscalibrations can
    average to the same scalar. The diagram is what lets a reader see the SHAPE:
    uniform overconfidence and a single bad high-confidence bin are different
    failures with the same ECE.

    Returns:
        `{"n_bins", "edges", "bins": [{"lo","hi","count","accuracy",
        "confidence","gap"}...], "ece"}`. Empty bins keep `count: 0` with null
        accuracy/confidence so the table has fixed length and is directly
        plottable.
    """
    import numpy as np

    probs, gold = _as_arrays(probabilities, targets)
    confidence = np.max(probs, axis=-1)
    predicted = np.argmax(probs, axis=-1)
    correct = (predicted == gold).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, Any]] = []
    total = np.float64(len(gold))
    ece = np.float64(0.0)
    for index in range(n_bins):
        lo, hi = float(edges[index]), float(edges[index + 1])
        if index == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = int(np.count_nonzero(mask))
        if count == 0:
            bins.append(
                {"lo": lo, "hi": hi, "count": 0, "accuracy": None,
                 "confidence": None, "gap": None}
            )
            continue
        bin_acc = float(np.mean(correct[mask]))
        bin_conf = float(np.mean(confidence[mask]))
        ece += (count / total) * abs(bin_acc - bin_conf)
        bins.append(
            {"lo": lo, "hi": hi, "count": count, "accuracy": round(bin_acc, 6),
             "confidence": round(bin_conf, 6),
             "gap": round(bin_acc - bin_conf, 6)}
        )

    return {
        "n_bins": n_bins,
        "edges": [float(e) for e in edges],
        "bins": bins,
        "ece": round(float(ece), 6),
        "note": (
            "Equal-width bins over predicted-class confidence. ECE is "
            "bin-count sensitive and is not an aggregate score."
        ),
    }


@dataclass(frozen=True)
class CalibrationMetrics:
    """The before/after metrics block written into the artifact."""

    ece_before: float
    ece_after: float
    nll_before: float
    nll_after: float
    n_samples: int
    n_bins: int
    n_classes: int

    @property
    def ece_improvement(self) -> float:
        """ECE reduction. Negative means calibration made reliability worse."""
        return self.ece_before - self.ece_after

    @property
    def nll_improvement(self) -> float:
        return self.nll_before - self.nll_after

    def describe(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
            "n_bins": self.n_bins,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
            "ece_improvement": round(self.ece_improvement, 6),
            "nll_before": self.nll_before,
            "nll_after": self.nll_after,
            "nll_improvement": round(self.nll_improvement, 6),
        }


def compare_calibration(
    probabilities: Any,
    targets: Sequence[int],
    calibration: Any,
    *,
    n_bins: int = DEFAULT_N_BINS,
) -> CalibrationMetrics:
    """Before/after metrics for a fitted calibration, on the fitting split.

    IMPORTANT: because `probabilities` are the same rows the temperature was
    fitted on, these are **fit-set** numbers. They are diagnostic -- they show
    whether the fit reduced NLL -- and they are optimistic by construction. The
    artifact records them with that caveat rather than presenting them as a
    held-out result.

    Args:
        probabilities: `(N, C)` raw probabilities.
        targets: `N` gold indices.
        calibration: a `TemperatureCalibration` (duck-typed: needs `apply` on a
            scalar). Rows are temperature-scaled in logit space then re-softmaxed.
        n_bins: ECE bin count.
    """
    import numpy as np

    from training.calibration.fitter import logits_from_probabilities, softmax

    probs, gold = _as_arrays(probabilities, targets)
    n_classes = probs.shape[1]

    ece_before = expected_calibration_error(probs, gold, n_bins=n_bins)
    nll_before = mean_negative_log_likelihood(probs, gold)

    temperature = float(getattr(calibration, "temperature"))
    scaled = softmax(logits_from_probabilities(probs) / temperature)
    ece_after = expected_calibration_error(scaled, gold, n_bins=n_bins)
    nll_after = mean_negative_log_likelihood(scaled, gold)

    return CalibrationMetrics(
        ece_before=round(ece_before, 6),
        ece_after=round(ece_after, 6),
        nll_before=round(nll_before, 6),
        nll_after=round(nll_after, 6),
        n_samples=int(len(gold)),
        n_bins=n_bins,
        n_classes=n_classes,
    )
