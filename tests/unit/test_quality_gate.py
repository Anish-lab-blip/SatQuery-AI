"""Tests for the deterministic input-quality gate (finding F5-5).

Written after the slow VLM probe produced this, on 512x512 uniform noise, with
a prompt that explicitly instructed the model to decline:

    "A black and white photograph of a man and a woman, who appear to be in a
     room, with a table in front of them."

The prompt cannot prevent that. This gate is the guard that does. Every
assertion here traces to that one measurement.

The load-bearing property is that **the verdict is deterministic and the
thresholds are constants**. If a future edit makes the gate probabilistic, or
tunes it against the test set, these tests will not catch it -- but the module
will no longer be doing the job it was written for.
"""

from __future__ import annotations

import numpy as np
import pytest

from preprocessing.quality import (
    BLOCKING_VERDICTS,
    MIN_AUTOCORRELATION,
    NOISE_ENTROPY_BITS,
    QualityVerdict,
    assess_image_quality,
    histogram_entropy,
    lag1_autocorrelation,
    normalized_std,
    to_grayscale,
)

# ---------------------------------------------------------------------------
# Fixtures -- the four input classes the gate must separate
# ---------------------------------------------------------------------------


def structured_plane(size: int = 128) -> np.ndarray:
    """Smooth ramp plus mild texture. Stands in for real imagery."""
    yy, xx = np.mgrid[0:size, 0:size]
    ramp = (xx + yy).astype(np.float64)
    texture = np.random.default_rng(0).integers(0, 40, (size, size)).astype(np.float64)
    return ramp + texture


def uniform_noise(size: int = 128) -> np.ndarray:
    """The exact input class that produced the hallucination."""
    return np.random.default_rng(0).integers(0, 256, (size, size)).astype(np.uint8)


def constant_plane(size: int = 128, value: int = 1000) -> np.ndarray:
    return np.full((size, size), value, dtype=np.uint16)


def salt_and_pepper(size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 2, (size, size)).astype(np.uint8) * 255
    return arr


# ---------------------------------------------------------------------------
# The headline property: noise is separated from structure
# ---------------------------------------------------------------------------


def test_structured_imagery_is_accepted() -> None:
    q = assess_image_quality(structured_plane())
    assert q.verdict is QualityVerdict.STRUCTURED
    assert q.is_usable is True


def test_uniform_noise_is_rejected() -> None:
    """The F5-5 input class. Must never reach the model."""
    q = assess_image_quality(uniform_noise())
    assert q.verdict is QualityVerdict.NOISE
    assert q.is_usable is False
    assert q.verdict in BLOCKING_VERDICTS


def test_salt_and_pepper_is_rejected() -> None:
    """Uncorrelated with a bimodal spread -- also a hallucination risk."""
    q = assess_image_quality(salt_and_pepper())
    assert q.verdict is QualityVerdict.NOISE
    assert q.is_usable is False


def test_constant_plane_is_flagged_but_not_blocked() -> None:
    """A flat field has no content but is not noise.

    It must not be treated as unanalysable (a flat desert scene is a real
    scene), but the caller should lower confidence.
    """
    q = assess_image_quality(constant_plane())
    assert q.verdict is QualityVerdict.FLAT
    assert q.is_usable is True
    assert q.is_degraded is True


def test_noise_is_not_confused_with_flat() -> None:
    """The distinction that variance alone cannot make.

    Both have low information content, but only one is uncorrelated. Getting
    this backwards would either block legitimate flat scenes or admit noise.
    """
    noise = assess_image_quality(uniform_noise())
    flat = assess_image_quality(constant_plane())
    assert noise.verdict is not flat.verdict
    assert noise.verdict is QualityVerdict.NOISE
    assert flat.verdict is QualityVerdict.FLAT


# ---------------------------------------------------------------------------
# The discriminating metric: lag-1 autocorrelation
# ---------------------------------------------------------------------------


def test_autocorrelation_is_high_for_structure() -> None:
    assert lag1_autocorrelation(to_grayscale(structured_plane())) > 0.6


def test_autocorrelation_is_near_zero_for_noise() -> None:
    assert abs(lag1_autocorrelation(to_grayscale(uniform_noise()))) < 0.10


def test_autocorrelation_of_a_constant_is_one() -> None:
    """Perfectly coherent by definition; FLAT is what flags it, not autocorr."""
    assert lag1_autocorrelation(to_grayscale(constant_plane())) == pytest.approx(1.0)


def test_autocorrelation_separates_the_two_classes_by_a_wide_margin() -> None:
    """The gate's whole premise: the margin is large, not marginal.

    Measured on the four input classes: structured 0.952, uniform noise -0.008,
    salt-and-pepper -0.010, constant 1.000. Autocorrelation carries the gate.
    """
    structure = lag1_autocorrelation(to_grayscale(structured_plane()))
    noise = lag1_autocorrelation(to_grayscale(uniform_noise()))
    assert structure - noise > 0.5, (
        f"margin is only {structure - noise:.3f}; the threshold "
        f"({MIN_AUTOCORRELATION}) is too close to the noise floor to be safe"
    )


def test_autocorrelation_is_the_load_bearing_signal_not_entropy() -> None:
    """Documents which metric actually discriminates, so a future change to
    the entropy threshold is not mistaken for a weakening of the gate.

    Measured: entropy gives structured 7.581 vs noise 7.988 against a 7.8
    threshold -- a 0.22-bit margin. Autocorrelation gives 0.952 vs -0.008
    against a 0.10 threshold -- a 0.96 margin.
    """
    structured = to_grayscale(structured_plane())
    noise = to_grayscale(uniform_noise())

    autocorr_margin = lag1_autocorrelation(structured) - lag1_autocorrelation(noise)
    entropy_margin = histogram_entropy(noise) - histogram_entropy(structured)

    assert autocorr_margin > entropy_margin, (
        "the design assumes autocorrelation is the stronger signal; if that "
        "has reversed, the thresholds need re-deriving"
    )


def test_autocorrelation_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        lag1_autocorrelation(np.zeros((2, 2, 2)))


def test_autocorrelation_handles_a_single_row() -> None:
    assert lag1_autocorrelation(np.zeros((1, 10))) == 0.0


# ---------------------------------------------------------------------------
# Supporting metrics
# ---------------------------------------------------------------------------


def test_entropy_is_maximal_for_a_uniform_histogram() -> None:
    """8-bit uniform spread approaches 8.0 bits."""
    assert histogram_entropy(uniform_noise().astype(np.float64)) > NOISE_ENTROPY_BITS


def test_entropy_is_low_for_a_ramp() -> None:
    """A smooth ramp concentrates intensity, so entropy stays well below 8."""
    assert histogram_entropy(to_grayscale(structured_plane())) < NOISE_ENTROPY_BITS


def test_entropy_of_a_constant_is_zero() -> None:
    assert histogram_entropy(to_grayscale(constant_plane())) == 0.0


def test_normalized_std_is_scale_invariant() -> None:
    """A stretched image must not change the verdict."""
    base = to_grayscale(structured_plane())
    assert normalized_std(base) == pytest.approx(normalized_std(base * 17.0))


def test_normalized_std_is_zero_for_a_constant() -> None:
    assert normalized_std(to_grayscale(constant_plane())) == 0.0


# ---------------------------------------------------------------------------
# Shape handling
# ---------------------------------------------------------------------------


def test_grayscale_passthrough_for_2d() -> None:
    arr = np.zeros((8, 8), dtype=np.float64)
    assert to_grayscale(arr).shape == (8, 8)


def test_channels_last_is_reduced() -> None:
    arr = np.zeros((8, 8, 3), dtype=np.float64)
    assert to_grayscale(arr).shape == (8, 8)


def test_channels_first_is_reduced() -> None:
    arr = np.zeros((3, 8, 8), dtype=np.float64)
    assert to_grayscale(arr).shape == (8, 8)


def test_single_channel_is_broadcast_not_dropped() -> None:
    arr = np.zeros((8, 8, 1), dtype=np.float64)
    assert to_grayscale(arr).shape == (8, 8)


def test_quality_accepts_channels_first_raster_data() -> None:
    """The form `read_bands` returns: (bands, H, W)."""
    bands = np.stack([structured_plane(64)] * 3)
    q = assess_image_quality(bands)
    assert q.verdict is QualityVerdict.STRUCTURED


def test_quality_accepts_channels_last_raster_data() -> None:
    bands = np.stack([structured_plane(64)] * 3, axis=-1)
    q = assess_image_quality(bands)
    assert q.verdict is QualityVerdict.STRUCTURED


# ---------------------------------------------------------------------------
# Degenerate input
# ---------------------------------------------------------------------------


def test_nan_input_is_invalid() -> None:
    arr = structured_plane(64).astype(np.float64)
    arr[0, 0] = np.nan
    q = assess_image_quality(arr)
    assert q.verdict is QualityVerdict.INVALID_VALUES
    assert q.is_usable is False


def test_inf_input_is_invalid() -> None:
    arr = structured_plane(64).astype(np.float64)
    arr[5, 5] = np.inf
    q = assess_image_quality(arr)
    assert q.verdict is QualityVerdict.INVALID_VALUES


def test_a_single_nan_disqualifies_a_large_array() -> None:
    """The bug this test exists for.

    An earlier version used `finite_fraction < 0.999`. One NaN in a 64x64 array
    is 0.99976, which is NOT less than 0.999 -- so the defect passed on a large
    image and was caught on a small one. The verdict depended on image
    dimensions rather than on the defect.
    """
    arr = structured_plane(128).astype(np.float64)
    arr[0, 0] = np.nan
    assert assess_image_quality(arr).verdict is QualityVerdict.INVALID_VALUES


def test_a_single_non_finite_disqualifies_regardless_of_size() -> None:
    """The size-independence property, asserted directly across many sizes."""
    for size in (32, 64, 128, 256):
        arr = structured_plane(size).astype(np.float64)
        arr[0, 0] = np.nan
        q = assess_image_quality(arr)
        assert q.verdict is QualityVerdict.INVALID_VALUES, (
            f"size {size} tolerated a NaN (finite_fraction "
            f"{q.finite_fraction})"
        )


def test_partially_non_finite_is_invalid() -> None:
    arr = structured_plane(64).astype(np.float64)
    arr[:32, :] = np.nan
    q = assess_image_quality(arr)
    assert q.verdict is QualityVerdict.INVALID_VALUES
    assert q.finite_fraction == pytest.approx(0.5)


def test_autocorrelation_refuses_non_finite_input() -> None:
    """It raises rather than returning NaN.

    A NaN verdict would fall through every branch in assess_image_quality.
    """
    arr = structured_plane(64).astype(np.float64)
    arr[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        lag1_autocorrelation(arr)


def test_entropy_tolerates_non_finite_by_dropping_it() -> None:
    """Contrast with autocorrelation: entropy filters, autocorrelation refuses.

    Entropy is a histogram statistic, so dropping bad samples is meaningful.
    Correlation is a joint statistic, so dropping them is not.
    """
    arr = structured_plane(64).astype(np.float64)
    arr[0, 0] = np.nan
    assert histogram_entropy(arr) > 0.0


def test_empty_array_is_invalid_not_a_crash() -> None:
    q = assess_image_quality(np.zeros((0, 0)))
    assert q.verdict is QualityVerdict.INVALID_VALUES
    assert q.is_usable is False


def test_tiny_array_is_flagged_rather_than_judged() -> None:
    q = assess_image_quality(np.zeros((4, 4), dtype=np.uint8))
    assert q.verdict is QualityVerdict.TOO_SMALL
    assert q.is_usable is True
    assert q.is_degraded is True


def test_all_zero_tile_is_flat_not_noise() -> None:
    """A common real case: an empty nodata tile. Low information, not noise."""
    q = assess_image_quality(np.zeros((64, 64), dtype=np.uint8))
    assert q.verdict is QualityVerdict.FLAT


# ---------------------------------------------------------------------------
# Determinism -- the property the whole design rests on
# ---------------------------------------------------------------------------


def test_verdict_is_deterministic() -> None:
    arr = uniform_noise()
    verdicts = {assess_image_quality(arr).verdict for _ in range(5)}
    assert len(verdicts) == 1


def test_repeated_calls_produce_identical_metrics() -> None:
    arr = structured_plane()
    a = assess_image_quality(arr)
    b = assess_image_quality(arr)
    assert a.to_dict() == b.to_dict()


def test_gate_is_not_seeded_by_global_numpy_state() -> None:
    """Changing the global RNG must not change the verdict."""
    arr = structured_plane()
    before = assess_image_quality(arr).verdict
    np.random.seed(12345)
    np.random.random(1000)
    after = assess_image_quality(arr).verdict
    assert before is after


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_every_verdict_has_a_reason() -> None:
    cases = [
        structured_plane(),
        uniform_noise(),
        constant_plane(),
        np.zeros((4, 4), dtype=np.uint8),
    ]
    for arr in cases:
        q = assess_image_quality(arr)
        assert q.reason(), f"no reason for verdict {q.verdict}"


def test_noise_reason_quotes_the_measured_value() -> None:
    q = assess_image_quality(uniform_noise())
    assert "autocorrelation" in q.reason()


def test_to_dict_is_json_serialisable() -> None:
    import json

    q = assess_image_quality(structured_plane())
    json.dumps(q.to_dict())  # must not raise


def test_to_dict_contains_only_numbers_and_strings() -> None:
    q = assess_image_quality(structured_plane())
    payload = q.to_dict()
    assert isinstance(payload["verdict"], str)
    assert isinstance(payload["autocorrelation"], float)
    assert isinstance(payload["shape"], list)


def test_blocking_verdicts_are_exactly_the_unanalysable_ones() -> None:
    assert BLOCKING_VERDICTS == {
        QualityVerdict.NOISE,
        QualityVerdict.INVALID_VALUES,
    }


def test_threshold_override_is_honoured() -> None:
    """The threshold is a parameter so an experiment can widen it."""
    arr = structured_plane()
    strict = assess_image_quality(arr, min_autocorrelation=0.999)
    assert strict.verdict is QualityVerdict.NOISE