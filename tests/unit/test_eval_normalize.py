"""Tests for `evaluation.normalize` — the plan section 63 metric normaliser.

The point of these tests is **non-invention**: the module must refuse to guess a
transform for an unregistered metric, must refuse to clamp an out-of-range value,
and must refuse to aggregate while the official weights are unpublished.

Mutation check: if `normalize` returned a clamped value instead of raising, or if
`aggregate` returned a made-up weighted sum, at least one test here fails.
"""

from __future__ import annotations

import pytest

import evaluation.normalize as normalize_mod
from evaluation.normalize import (
    TRANSFORMS,
    OfficialWeightsUnavailableError,
    UnregisteredMetricError,
    NormalisationRangeError,
    aggregate,
    normalize,
)

#: The exact, deliberately minimal metric set the module registers.
EXPECTED_METRICS = {"iou", "miou", "f1", "precision", "recall", "accuracy", "map"}


def test_registered_metrics_are_the_declared_minimal_set() -> None:
    assert set(TRANSFORMS) == EXPECTED_METRICS
    # Every registered transform is the identity map, and says so.
    for transform in TRANSFORMS.values():
        assert transform.name == "identity"
        assert "native range already [0,1]" in transform.note


@pytest.mark.parametrize("metric", sorted(EXPECTED_METRICS))
@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_identity_metrics_normalize_to_themselves_and_stay_in_unit_range(
    metric: str, value: float
) -> None:
    result = normalize(metric, value)
    assert result == pytest.approx(value)
    assert 0.0 <= result <= 1.0


def test_metric_name_is_case_and_whitespace_insensitive() -> None:
    assert normalize("  IoU ", 0.42) == pytest.approx(0.42)
    assert normalize("MiOu", 0.3) == pytest.approx(0.3)


def test_unregistered_metric_raises() -> None:
    with pytest.raises(UnregisteredMetricError):
        normalize("psnr", 0.9)
    # A lower-is-better metric is NOT silently handled: no registration, no guess.
    with pytest.raises(UnregisteredMetricError):
        normalize("error_rate", 0.1)


@pytest.mark.parametrize("bad", [1.0000001, 1.5, 2.0, -0.1, -1.0])
def test_out_of_range_raises_rather_than_clamping(bad: float) -> None:
    with pytest.raises(NormalisationRangeError):
        normalize("iou", bad)


def test_official_weights_is_none() -> None:
    assert normalize_mod.official_weights is None


@pytest.mark.parametrize(
    "scores",
    [
        {},
        {"iou": 0.5},
        {"iou": 0.9, "f1": 0.4},
        {"iou": 0.9, "miou": 0.8, "f1": 0.7, "accuracy": 0.95},
    ],
)
def test_aggregate_raises_while_weights_are_unpublished(scores: dict) -> None:
    with pytest.raises(OfficialWeightsUnavailableError):
        aggregate(scores)


def test_no_aggregate_formula_is_invented() -> None:
    # `aggregate` has no reachable returning path while weights are None: every
    # input raises, so no weighting can have been invented.
    for scores in ({"iou": 0.1}, {"iou": 0.9, "f1": 0.2}, {"accuracy": 1.0}):
        with pytest.raises(OfficialWeightsUnavailableError):
            aggregate(scores)
    # And there is no hidden default weighting table anywhere in the module.
    weight_like = {
        name: value
        for name, value in vars(normalize_mod).items()
        if "weight" in name.lower()
        and not isinstance(value, type)
        and value is not None
    }
    assert weight_like == {}
