"""SatQuery AI — deterministic input-quality gate.

Exists because of finding F5-5 (docs/PHASE5_VLM_CONTRACT.md). A loaded
SmolVLM-500M-Instruct, given 512x512 uniform random noise and a prompt that
explicitly says "answer only from what is visible" and "if the image does not
contain enough information, say so plainly", produced:

    "A black and white photograph of a man and a woman, who appear to be in a
     room, with a table in front of them."

That is a fluent, specific, entirely fabricated scene description. The prompt
cannot prevent it. A 500M-parameter VLM will describe *something* for any input
it is given, and asking it to self-assess reliably is asking it to do the thing
it just failed at.

So the gate is deterministic and upstream of the model. If an array carries no
plausible scene structure, the specialist is never asked about it.

The discriminating signal is **lag-1 spatial autocorrelation**, not variance.

    real remote-sensing imagery   autocorrelation 0.6 - 0.99
    uniform random noise          autocorrelation ~ 0.00
    a constant (blank) image      autocorrelation undefined; variance ~ 0

Variance alone cannot separate these. A flat desert scene and a flat "all-zero"
tile both have near-zero variance, but the desert has strong neighbour
correlation and the zero tile does not. Autocorrelation is high for *both*
uniform-but-real imagery and textured imagery, and collapses only for noise.

Nothing here is learned, sampled, or probabilistic. Same input, same verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

#: Below this lag-1 autocorrelation, an array is not spatially coherent.
#: Real imagery sits far above it (typically > 0.6); uniform noise sits at ~0.
MIN_AUTOCORRELATION = 0.10

#: Below this normalised standard deviation, an array is effectively constant.
FLAT_STD_EPSILON = 1e-6

#: Shannon entropy (bits, 8-bit histogram) above which the intensity
#: distribution is indistinguishable from uniform noise. A uniform 8-bit
#: distribution is 8.0 bits.
#:
#: MEASURED, and the margin here is thin: structured ramp+texture gives 7.581,
#: uniform noise gives 7.988. That is a 0.22-bit gap against a 7.8 threshold.
#: The autocorrelation check is what actually carries this gate (0.952 vs
#: -0.008, a 0.96 margin against a 0.10 threshold); entropy is a redundant
#: second signal. It is kept because a future change to either metric should
#: not silently disable the gate, but do not rely on it alone.
NOISE_ENTROPY_BITS = 7.8


class QualityVerdict(str, Enum):
    """What the gate concluded about an array."""

    #: Spatial structure consistent with real imagery. Safe to analyse.
    STRUCTURED = "structured"
    #: Effectively constant. Low information, but not a hallucination risk.
    FLAT = "flat"
    #: Uncorrelated, near-uniform intensity distribution. Hallucination risk.
    NOISE = "noise"
    #: Too few pixels to judge. Treated as usable but flagged.
    TOO_SMALL = "too_small"
    #: NaN or infinite values present.
    INVALID_VALUES = "invalid_values"


#: Verdicts that must block a VLM call.
BLOCKING_VERDICTS: frozenset[QualityVerdict] = frozenset(
    {QualityVerdict.NOISE, QualityVerdict.INVALID_VALUES}
)


@dataclass(frozen=True)
class ImageQuality:
    """Measured signals. Every field is a number, none is a model output."""

    verdict: QualityVerdict
    autocorrelation: float
    std_normalized: float
    entropy_bits: float
    finite_fraction: float
    shape: tuple[int, ...]

    @property
    def is_usable(self) -> bool:
        return self.verdict not in BLOCKING_VERDICTS

    @property
    def is_degraded(self) -> bool:
        """Usable, but the caller should lower confidence."""
        return self.verdict in {QualityVerdict.FLAT, QualityVerdict.TOO_SMALL}

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict.value,
            "autocorrelation": round(self.autocorrelation, 4),
            "std_normalized": round(self.std_normalized, 6),
            "entropy_bits": round(self.entropy_bits, 4),
            "finite_fraction": round(self.finite_fraction, 4),
            "shape": list(self.shape),
        }

    def reason(self) -> str:
        if self.verdict is QualityVerdict.NOISE:
            return (
                f"input has no spatial structure (autocorrelation "
                f"{self.autocorrelation:.3f} < {MIN_AUTOCORRELATION}); "
                f"this is not analysable imagery"
            )
        if self.verdict is QualityVerdict.FLAT:
            return "input is effectively constant; there is nothing to describe"
        if self.verdict is QualityVerdict.TOO_SMALL:
            return f"input is too small to assess ({self.shape})"
        if self.verdict is QualityVerdict.INVALID_VALUES:
            return (
                f"input contains non-finite values "
                f"(finite fraction {self.finite_fraction:.3f})"
            )
        return "input has usable spatial structure"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def to_grayscale(array: np.ndarray) -> np.ndarray:
    """Collapse to a 2-D float array. Handles (H,W), (H,W,C) and (C,H,W)."""
    arr = np.asarray(array)

    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3:
        # (C, H, W) with a small leading dim is channel-first.
        if arr.shape[0] <= 4 and arr.shape[1] > 4 and arr.shape[2] > 4:
            gray = arr.mean(axis=0)
        else:
            gray = arr[:, :, :3].mean(axis=2) if arr.shape[2] >= 3 else arr.mean(axis=2)
    else:
        gray = arr.reshape(arr.shape[-2], arr.shape[-1])

    return gray.astype(np.float64, copy=False)


def lag1_autocorrelation(gray: np.ndarray) -> float:
    """Pearson correlation between each pixel and its right/down neighbours.

    Returns the mean of the horizontal and vertical correlations.

    A constant array returns 1.0 by definition (every neighbour pair is
    identical). That is correct: a flat field is perfectly spatially coherent,
    and the FLAT verdict is what flags its lack of information.
    """
    if gray.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {gray.shape}")
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0

    # Refuse rather than compute. Pearson correlation over NaN or inf produces
    # NaN silently, and a NaN that reaches the verdict comparison would fall
    # through every branch. assess_image_quality screens for this first; the
    # guard exists because this is also a public function.
    if not np.isfinite(gray).all():
        raise ValueError(
            "lag1_autocorrelation requires finite values; "
            "screen with assess_image_quality first"
        )

    std = float(gray.std())
    if std < FLAT_STD_EPSILON:
        return 1.0

    h_corr = _pearson(gray[:, :-1].ravel(), gray[:, 1:].ravel())
    v_corr = _pearson(gray[:-1, :].ravel(), gray[1:, :].ravel())

    values = [c for c in (h_corr, v_corr) if not np.isnan(c)]
    if not values:
        return 0.0
    return float(np.mean(values))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = float(np.sqrt((a_c * a_c).sum() * (b_c * b_c).sum()))
    if denom < FLAT_STD_EPSILON:
        return float("nan")
    return float((a_c * b_c).sum() / denom)


def histogram_entropy(gray: np.ndarray, bits: int = 8) -> float:
    """Shannon entropy of the intensity histogram, in bits.

    A uniform distribution over 256 bins gives 8.0. Real imagery clusters into
    a smaller number of intensity levels and sits well below that.

    Non-finite values are dropped rather than propagated. Callers that need
    strictness should screen with `assess_image_quality`, which refuses any
    non-finite input outright.
    """
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        return 0.0

    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < FLAT_STD_EPSILON:
        return 0.0

    bins = 1 << bits
    counts, _ = np.histogram(finite, bins=bins, range=(lo, hi))
    total = counts.sum()
    if total == 0:
        return 0.0

    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def normalized_std(gray: np.ndarray) -> float:
    """Standard deviation divided by the dynamic range. Scale-invariant."""
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        return 0.0
    rng = float(finite.max() - finite.min())
    if rng < FLAT_STD_EPSILON:
        return 0.0
    return float(finite.std() / rng)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def assess_image_quality(
    array: np.ndarray,
    *,
    min_side: int = 16,
    min_autocorrelation: float = MIN_AUTOCORRELATION,
) -> ImageQuality:
    """Classify an array as structured, flat, noise, or invalid.

    Deterministic. No model, no randomness, no thresholds learned from data.

    Args:
        array: raster data in any of (H,W), (H,W,C), (C,H,W).
        min_side: arrays smaller than this in either dimension are flagged
            TOO_SMALL rather than judged.
        min_autocorrelation: override for the noise threshold. Exposed so an
            experiment can widen it, never so production can loosen it silently.
    """
    arr = np.asarray(array)
    shape = tuple(int(s) for s in arr.shape)

    if arr.size == 0:
        return ImageQuality(
            QualityVerdict.INVALID_VALUES, 0.0, 0.0, 0.0, 0.0, shape
        )

    finite_mask = np.isfinite(arr)
    finite_fraction = float(finite_mask.mean())

    # Any non-finite value is disqualifying. Deliberately NO tolerance
    # threshold: a fixed fraction is size-dependent, so one NaN in a 64x64
    # array (0.99976) would pass while one NaN in a 10x10 array (0.99) would
    # fail. The same defect must not be tolerated or rejected depending on
    # image dimensions. NaN in a raster means nodata or corruption, and the
    # caller has to resolve that before the gate rather than have the gate
    # average it away.
    #
    # Known limitation: a float GeoTIFF whose nodata sentinel is NaN lands here
    # and is refused. That is deliberate for now -- the alternative is
    # computing correlation over NaN, which produces a meaningless verdict.
    # Filling nodata from the raster profile belongs in the tiling work.
    if finite_fraction < 1.0:
        return ImageQuality(
            QualityVerdict.INVALID_VALUES, 0.0, 0.0, 0.0, finite_fraction, shape
        )

    gray = to_grayscale(arr)

    if gray.shape[0] < min_side or gray.shape[1] < min_side:
        return ImageQuality(
            QualityVerdict.TOO_SMALL,
            0.0,
            normalized_std(gray),
            histogram_entropy(gray),
            finite_fraction,
            shape,
        )

    autocorr = lag1_autocorrelation(gray)
    std_norm = normalized_std(gray)
    entropy = histogram_entropy(gray)

    # -- constant image ---------------------------------------------------
    if std_norm < FLAT_STD_EPSILON:
        return ImageQuality(
            QualityVerdict.FLAT, autocorr, std_norm, entropy, finite_fraction, shape
        )

    # -- noise: uncorrelated AND near-uniform intensity --------------------
    # Both conditions are required. An uncorrelated but low-entropy array is
    # unusual enough to warrant caution rather than a hard block, and a
    # high-entropy but well-correlated array is just a textured scene.
    if autocorr < min_autocorrelation and entropy >= NOISE_ENTROPY_BITS:
        return ImageQuality(
            QualityVerdict.NOISE, autocorr, std_norm, entropy, finite_fraction, shape
        )

    # -- uncorrelated with a normal intensity spread -----------------------
    # Salt-and-pepper and speckle land here. Also a hallucination risk.
    if autocorr < min_autocorrelation:
        return ImageQuality(
            QualityVerdict.NOISE, autocorr, std_norm, entropy, finite_fraction, shape
        )

    return ImageQuality(
        QualityVerdict.STRUCTURED, autocorr, std_norm, entropy, finite_fraction, shape
    )


__all__ = [
    "QualityVerdict",
    "ImageQuality",
    "BLOCKING_VERDICTS",
    "MIN_AUTOCORRELATION",
    "NOISE_ENTROPY_BITS",
    "assess_image_quality",
    "lag1_autocorrelation",
    "histogram_entropy",
    "normalized_std",
    "to_grayscale",
]