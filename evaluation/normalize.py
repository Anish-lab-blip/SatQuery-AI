"""SatQuery AI — metric normalisation (plan section 63).

The plan's "Metric Normalisation" contract is:

    raw -> task-specific transform -> 0-1 normalized

and it fixes the aggregation surface explicitly:

    aggregate:
      official_weights: null
    ...until the organizers publish them.

Plan section 63 states that the uploaded specification **explicitly prohibits
inventing an official aggregate formula**. This module is written to honour that
prohibition literally, so it is deliberately small:

  * The transform set is **deliberately minimal.** The ONLY transform registered
    is the identity map, and it is registered only for metrics whose native range
    is already `[0, 1]` (IoU, mIoU, F1, precision, recall, accuracy, mAP). Each
    such metric is listed explicitly in `TRANSFORMS`.
  * A metric that is not registered **raises** (`UnregisteredMetricError`). It is
    never guessed at, never clamped, and never inverted. A lower-is-better metric
    (e.g. an error rate) is NOT handled here, because doing so would require a
    bound this module has no authority to choose.
  * A value that falls outside `[0, 1]` after its transform **raises**
    (`NormalisationRangeError`); it is never clamped silently.
  * `official_weights` is `None`, and `aggregate(...)` **raises** while it is
    `None` (`OfficialWeightsUnavailableError`). No aggregate formula exists here
    and none may be invented.

Adding a transform (or a weighting) is therefore a **contract decision** backed
by an authoritative source — not a default this module is allowed to assume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from core.errors import SatQueryError

__all__ = [
    "Transform",
    "TRANSFORMS",
    "normalize",
    "official_weights",
    "aggregate",
    "NormalisationError",
    "UnregisteredMetricError",
    "NormalisationRangeError",
    "OfficialWeightsUnavailableError",
]


# ---------------------------------------------------------------------------
# Errors — kept in this module (a new, additive file) so no existing error
# taxonomy is edited. They subclass the project's `SatQueryError` so they carry
# the same `code` / `user_message` / `to_trace()` contract as every other error.
# ---------------------------------------------------------------------------
class NormalisationError(SatQueryError):
    """A metric could not be normalised under the plan section 63 contract."""

    code = "normalisation_error"
    user_message = "A metric could not be normalised."


class UnregisteredMetricError(NormalisationError):
    """The metric has no registered transform, and none may be guessed."""

    code = "unregistered_metric"
    user_message = "No normalisation transform is registered for this metric."


class NormalisationRangeError(NormalisationError):
    """The transformed value fell outside [0, 1]; clamping is forbidden."""

    code = "normalisation_range_error"
    user_message = "A metric normalised outside the valid [0, 1] range."


class OfficialWeightsUnavailableError(NormalisationError):
    """The official aggregate weights are not published; no formula is invented."""

    code = "official_weights_unavailable"
    user_message = "The official aggregate weights have not been published."


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Transform:
    """One declared `raw -> [0, 1]` transform for a metric.

    Attributes:
        name: the transform's identity, recorded so a caller can see WHAT was
            applied rather than only the result.
        function: the callable applied to the raw value.
        note: why this transform is the correct one for its metric(s).
    """

    name: str
    function: Callable[[float], float]
    note: str


def _identity(raw: float) -> float:
    """Return the raw value unchanged; the metric is already on [0, 1]."""
    return float(raw)


#: The ONLY transform this module registers: the identity map, valid solely for
#: metrics whose native range is already [0, 1]. A metric outside that set must
#: NOT be mapped through this — it must be registered with its own transform,
#: which is a contract decision (plan section 63).
_NATIVE_UNIT = Transform(
    name="identity",
    function=_identity,
    note="native range already [0,1]; the identity transform is exact",
)

#: Declarative registry: metric name -> transform. Every entry is explicit.
#: Each metric below is natively in [0, 1] and therefore needs no scaling:
#:   iou        intersection-over-union, a ratio in [0, 1]
#:   miou       mean IoU over classes, a mean of ratios in [0, 1]
#:   f1         harmonic mean of precision and recall, in [0, 1]
#:   precision  TP / (TP + FP), a ratio in [0, 1]
#:   recall     TP / (TP + FN), a ratio in [0, 1]
#:   accuracy   correct / total, a ratio in [0, 1]
#:   map        mean average precision, a mean of precisions in [0, 1]
TRANSFORMS: dict[str, Transform] = {
    "iou": _NATIVE_UNIT,
    "miou": _NATIVE_UNIT,
    "f1": _NATIVE_UNIT,
    "precision": _NATIVE_UNIT,
    "recall": _NATIVE_UNIT,
    "accuracy": _NATIVE_UNIT,
    "map": _NATIVE_UNIT,
}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def normalize(metric_name: str, raw: float) -> float:
    """Normalise one raw metric to `[0, 1]` under plan section 63.

    Args:
        metric_name: the metric's registered name (case-insensitive).
        raw: the raw metric value.

    Returns:
        The normalised value, guaranteed to lie in the closed interval [0, 1].

    Raises:
        UnregisteredMetricError: the metric has no registered transform. This is
            the non-invention guard: an unknown metric is never guessed at.
        NormalisationRangeError: the transformed value is outside [0, 1]. It is
            refused rather than clamped, so a wrong transform cannot hide.
    """
    key = str(metric_name).strip().lower()
    transform = TRANSFORMS.get(key)
    if transform is None:
        raise UnregisteredMetricError(
            f"no normalisation transform is registered for metric "
            f"{metric_name!r}; registered metrics are {sorted(TRANSFORMS)}. "
            f"Adding a transform is a contract decision (plan section 63), not a "
            f"default.",
            context={"metric": metric_name, "registered": sorted(TRANSFORMS)},
        )

    value = float(transform.function(raw))
    if not (0.0 <= value <= 1.0):
        raise NormalisationRangeError(
            f"metric {metric_name!r} (transform={transform.name!r}) normalised to "
            f"{value!r}, which is outside [0, 1]; refusing to clamp silently",
            context={
                "metric": metric_name,
                "transform": transform.name,
                "raw": raw,
                "normalized": value,
            },
        )
    return value


# ---------------------------------------------------------------------------
# Aggregation — deliberately absent until the organizers publish the weights
# ---------------------------------------------------------------------------
#: The official aggregate weights, as fixed by plan section 63. `None` until the
#: organizers publish them; there is deliberately no default.
official_weights: None = None


def aggregate(metric_scores: Mapping[str, float]) -> float:
    """Combine normalised metrics into the official score.

    This function exists to make the absence of an official formula explicit and
    testable. Plan section 63 prohibits inventing an aggregate formula, so while
    `official_weights is None` there is no legal way to combine metrics, and this
    function raises.

    Args:
        metric_scores: normalised per-metric scores (unused while weights are
            absent; accepted so the contract is visible and the failure is
            testable).

    Raises:
        OfficialWeightsUnavailableError: always, while `official_weights` is
            `None`. There is no fallback weighting.
    """
    if official_weights is None:
        raise OfficialWeightsUnavailableError(
            "the official aggregate weights have not been published; plan section "
            "63 prohibits inventing an aggregate formula, so no score can be "
            "combined yet.",
            context={"metrics": sorted(metric_scores)},
        )
    # Unreachable while `official_weights` is None. Kept explicit so that if the
    # organizers ever publish weights, the weighting is still a deliberate act
    # rather than a silently-invented default.
    raise OfficialWeightsUnavailableError(
        "aggregate() has no formula: the organizers' weighting is not implemented "
        "here, and plan section 63 forbids inventing one.",
        context={"metrics": sorted(metric_scores), "weights": official_weights},
    )
