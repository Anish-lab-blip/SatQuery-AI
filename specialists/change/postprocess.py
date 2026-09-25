"""Change-map post-processing and registration quality.

Turns a probability map into discrete change regions, and measures how well
the two acquisitions are aligned. Both are deterministic — no model, no
sampling, same input same output.

WHY REGISTRATION QUALITY IS MEASURED, NOT ASSUMED
-------------------------------------------------
Plan section 16: "If registration quality is too poor: lower confidence,
optionally refuse precise spatial claims. Never convert bad registration into
artificial certainty."

A change detector fed two mis-registered images reports change along every
edge in the scene. That output is not wrong so much as meaningless, and it
looks exactly like a confident detection. So alignment is measured BEFORE the
change map is trusted, and the measurement travels with the result.

Method: `cv2.phaseCorrelate` on the two grayscale images. It returns the
sub-pixel shift and a response in [0, 1]. A response near 0 means the images
share no translatable structure — either genuinely different scenes, or one is
noise. A large shift means the pair is offset.

Deliberately a TRANSLATION model. Real mis-registration includes rotation and
warp, which phase correlation does not recover; a small residual after a
translation correction is therefore not proof of good alignment. What the
measurement can do honestly is flag the *bad* cases, and that is how it is
used: as a gate, not a certificate.

MORPHOLOGY ORDER MATTERS
------------------------
Opening then closing, not the reverse. Opening removes isolated speckle first,
so the closing that follows merges genuine regions rather than fusing specks
into false blobs. Reversing the order produces measurably larger regions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.errors import SpecialistError
from core.schemas import Box, ChangeRegion, CoordinateSystem
from evaluation.metrics.change import DEFAULT_THRESHOLD, binarize

#: Kernel size for the morphological open/close pair. 3 is the smallest that
#: removes single-pixel speckle without eroding real region boundaries.
DEFAULT_KERNEL_SIZE = 3


@dataclass(frozen=True)
class RegistrationQuality:
    """Measured alignment between two acquisitions."""

    shift_x: float
    shift_y: float
    response: float
    max_shift_px: int
    is_usable: bool
    reason: str

    @property
    def shift_magnitude(self) -> float:
        return float(np.hypot(self.shift_x, self.shift_y))

    def to_dict(self) -> dict[str, Any]:
        return {
            "shift_x": round(self.shift_x, 3),
            "shift_y": round(self.shift_y, 3),
            "shift_magnitude": round(self.shift_magnitude, 3),
            "response": round(self.response, 4),
            "max_shift_px": self.max_shift_px,
            "is_usable": self.is_usable,
            "reason": self.reason,
        }

    def confidence_factor(self) -> float:
        """Multiplier in [0, 1] applied to change confidence.

        Two independent penalties, whichever is worse:

            shift    1.0 at zero shift, 0.0 at max_shift_px
            response 0.0 at zero response, 1.0 at response >= 0.15

        A pair that is offset AND uncorrelated takes the worse of the two,
        which is the conservative choice.
        """
        if self.max_shift_px <= 0:
            shift_penalty = 0.0
        else:
            shift_penalty = max(0.0, 1.0 - self.shift_magnitude / self.max_shift_px)
        response_factor = min(1.0, max(0.0, self.response / 0.15))
        return float(min(shift_penalty, response_factor))


def to_grayscale_float(array: Any) -> np.ndarray:
    """Collapse to a 2-D float32 array in [0, 1].

    Accepts (H, W), (H, W, C) and (C, H, W). Percentile-stretched so that a
    uint16 raster and a uint8 raster of the same scene correlate identically.
    """
    arr = np.asarray(array)
    if arr.ndim == 3:
        if arr.shape[0] <= 4 and arr.shape[1] > 4 and arr.shape[2] > 4:
            gray = arr.mean(axis=0)
        else:
            gray = arr[:, :, :3].mean(axis=2) if arr.shape[2] >= 3 else arr.mean(axis=2)
    elif arr.ndim == 2:
        gray = arr
    else:
        raise ValueError(f"expected 2-D or 3-D input, got shape {arr.shape}")

    out = gray.astype(np.float32)
    finite = out[np.isfinite(out)]
    if finite.size:
        lo, hi = np.percentile(finite, (2, 98))
        if hi > lo:
            out = (out - lo) / (hi - lo)
    # The astype here is NOT redundant, and removing it breaks
    # measure_registration.
    #
    # np.percentile returns float64 scalars even for a float32 input, so
    # `out - lo` promotes the whole array to float64. Without this cast the
    # function returns float64 while its docstring promises float32 -- and
    # cv2.phaseCorrelate then fails its own type assertion:
    #
    #     src1.type() == window.type()  in cv::phaseCorrelate
    #
    # with a CV_32F Hanning window. Measured, not theorised.
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def measure_registration(
    t1: Any,
    t2: Any,
    *,
    max_shift_px: int = 8,
    min_response: float = 0.15,
) -> RegistrationQuality:
    """Measure translation between two acquisitions via phase correlation.

    Args:
        t1, t2: arrays in any of (H,W), (H,W,C), (C,H,W).
        max_shift_px: shifts beyond this are unusable.
        min_response: correlation response below this is unusable.

    Raises:
        SpecialistError: the arrays differ in shape, or the measurement fails.
    """
    a = to_grayscale_float(t1)
    b = to_grayscale_float(t2)

    if a.shape != b.shape:
        raise SpecialistError(
            f"cannot measure registration between different shapes: "
            f"{a.shape} vs {b.shape}",
            specialist="change",
        )

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise SpecialistError(
            f"opencv is required for registration measurement: {exc}",
            specialist="change",
        ) from exc

    # phaseCorrelate needs a window; without one the FFT edge effects dominate
    # and a perfectly aligned pair can score badly.
    #
    # The window type must match the source type exactly, or OpenCV asserts.
    # `to_grayscale_float` guarantees float32, so CV_32F is correct here --
    # and the check below makes the coupling explicit rather than implied, so
    # a future change to either side fails with a readable message instead of
    # an OpenCV assertion.
    h, w = a.shape
    if a.dtype != np.float32 or b.dtype != np.float32:
        raise SpecialistError(
            f"phase correlation requires float32 inputs; got {a.dtype} and "
            f"{b.dtype}. to_grayscale_float should guarantee this.",
            specialist="change",
        )
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)

    try:
        (dx, dy), response = cv2.phaseCorrelate(a, b, win)
    except Exception as exc:  # noqa: BLE001
        raise SpecialistError(
            f"phase correlation failed: {exc}", specialist="change"
        ) from exc

    response = float(response) if np.isfinite(response) else 0.0
    dx = float(dx) if np.isfinite(dx) else 0.0
    dy = float(dy) if np.isfinite(dy) else 0.0

    magnitude = float(np.hypot(dx, dy))

    if magnitude > max_shift_px:
        usable, reason = False, (
            f"acquisitions are offset by {magnitude:.1f}px "
            f"(limit {max_shift_px}px); change along every edge is expected"
        )
    elif response < min_response:
        usable, reason = False, (
            f"phase-correlation response {response:.4f} is below "
            f"{min_response}; the pair shares no translatable structure"
        )
    else:
        usable, reason = True, "translation within tolerance"

    return RegistrationQuality(
        shift_x=dx, shift_y=dy, response=response,
        max_shift_px=max_shift_px, is_usable=usable, reason=reason,
    )


# ---------------------------------------------------------------------------
# Region extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionStats:
    """One connected component, in pixel space."""

    label: int
    area_pixels: int
    bbox_px: tuple[int, int, int, int]   # col_min, row_min, col_max, row_max
    mean_probability: float

    @property
    def width(self) -> int:
        return self.bbox_px[2] - self.bbox_px[0]

    @property
    def height(self) -> int:
        return self.bbox_px[3] - self.bbox_px[1]


@dataclass
class PostprocessResult:
    """Everything the change specialist needs from a probability map."""

    binary_mask: np.ndarray
    cleaned_mask: np.ndarray
    regions: list[RegionStats] = field(default_factory=list)
    n_components_raw: int = 0
    n_components_kept: int = 0
    total_change_pixels: int = 0

    @property
    def change_fraction(self) -> float:
        if self.binary_mask.size == 0:
            return 0.0
        return float(self.total_change_pixels) / float(self.binary_mask.size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_components_raw": self.n_components_raw,
            "n_components_kept": self.n_components_kept,
            "total_change_pixels": self.total_change_pixels,
            "change_fraction": round(self.change_fraction, 6),
            "regions": [
                {
                    "label": r.label,
                    "area_pixels": r.area_pixels,
                    "bbox_px": list(r.bbox_px),
                    "mean_probability": round(r.mean_probability, 4),
                }
                for r in self.regions
            ],
        }


def morphological_cleanup(
    mask: np.ndarray,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    open_iterations: int = 1,
    close_iterations: int = 2,
) -> np.ndarray:
    """Open then close. Speckle first, then merge.

    Order is deliberate — see the module docstring. Returning uint8 0/1.

    With kernel_size <= 1 this is a no-op passthrough, so a caller can disable
    it without branching.
    """
    binary = np.asarray(mask).astype(np.uint8)
    if kernel_size <= 1 or (open_iterations <= 0 and close_iterations <= 0):
        return binary

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise SpecialistError(
            f"opencv is required for morphological cleanup: {exc}",
            specialist="change",
        ) from exc

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    out = binary
    if open_iterations > 0:
        out = cv2.morphologyEx(
            out, cv2.MORPH_OPEN, kernel, iterations=open_iterations
        )
    if close_iterations > 0:
        out = cv2.morphologyEx(
            out, cv2.MORPH_CLOSE, kernel, iterations=close_iterations
        )
    return (out > 0).astype(np.uint8)


def connected_regions(
    mask: np.ndarray,
    probabilities: np.ndarray | None = None,
    min_pixels: int = 32,
) -> tuple[list[RegionStats], int]:
    """Connected components, filtered by area.

    Args:
        mask: binary 0/1 or boolean mask.
        probabilities: optional probability map of the same shape, used to
            report each region's mean probability.
        min_pixels: components smaller than this are dropped.

    Returns:
        (kept_regions, n_components_before_filtering)

    The raw count is returned so a caller can report how many were discarded.
    Dropping without recording would hide how noisy the map actually was.

    Connectivity is 8-connected: a diagonal chain of changed pixels is one
    object on the ground, and 4-connectivity would fragment it.
    """
    binary = np.asarray(mask).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {binary.shape}")

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise SpecialistError(
            f"opencv is required for connected components: {exc}",
            specialist="change",
        ) from exc

    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    # Label 0 is the background.
    n_components = max(0, n_labels - 1)

    prob = None
    if probabilities is not None:
        prob = np.asarray(probabilities, dtype=np.float64)
        if prob.shape != binary.shape:
            raise ValueError(
                f"probabilities shape {prob.shape} != mask shape {binary.shape}"
            )

    kept: list[RegionStats] = []
    for label in range(1, n_labels):
        col_min = int(stats[label, cv2.CC_STAT_LEFT])
        row_min = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < min_pixels:
            continue

        if prob is not None:
            component = labels == label
            mean_p = float(prob[component].mean()) if component.any() else 0.0
        else:
            mean_p = 0.0

        kept.append(
            RegionStats(
                label=label,
                area_pixels=area,
                bbox_px=(col_min, row_min, col_min + width, row_min + height),
                mean_probability=mean_p,
            )
        )

    kept.sort(key=lambda r: r.area_pixels, reverse=True)
    return kept, n_components


def postprocess_change_map(
    probabilities: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    open_iterations: int = 1,
    close_iterations: int = 2,
    min_component_pixels: int = 32,
    apply_morphology: bool = True,
) -> PostprocessResult:
    """Probability map -> binary mask -> cleaned mask -> regions.

    The RAW binary mask is retained alongside the cleaned one. Metrics are
    reported on the raw mask by default, because morphology is a presentation
    aid and applying it before scoring would inflate the numbers relative to
    the literature. The cleaned mask is what gets drawn.
    """
    prob = np.asarray(probabilities, dtype=np.float64)
    if prob.ndim == 3 and prob.shape[0] == 1:
        prob = prob[0]
    if prob.ndim != 2:
        raise ValueError(f"expected a 2-D probability map, got {prob.shape}")

    raw = binarize(prob, threshold)
    cleaned = (
        morphological_cleanup(raw, kernel_size, open_iterations, close_iterations)
        if apply_morphology
        else raw.astype(np.uint8)
    )

    regions, n_raw = connected_regions(
        cleaned, probabilities=prob, min_pixels=min_component_pixels
    )

    return PostprocessResult(
        binary_mask=raw,
        cleaned_mask=cleaned.astype(bool),
        regions=regions,
        n_components_raw=n_raw,
        n_components_kept=len(regions),
        total_change_pixels=int(np.sum(cleaned)),
    )


# ---------------------------------------------------------------------------
# Schema bridge
# ---------------------------------------------------------------------------


def regions_to_schema(
    regions: list[RegionStats],
    image_width: int,
    image_height: int,
    *,
    coordinate_system: CoordinateSystem = CoordinateSystem.NORMALIZED_0_1,
) -> list[ChangeRegion]:
    """Convert pixel-space RegionStats into schema `ChangeRegion` objects.

    The Box is built in the REQUESTED coordinate system. Callers that want
    geographic coordinates convert afterwards via `geospatial.transform`, which
    keeps this function free of rasterio and CRS concerns.

    Raises:
        ValueError: image dimensions are not positive, or a region falls
            entirely outside the frame.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"image dimensions must be positive, got {image_width}x{image_height}"
        )

    out: list[ChangeRegion] = []
    for r in regions:
        c0, r0, c1, r1 = r.bbox_px

        if coordinate_system is CoordinateSystem.NORMALIZED_0_1:
            x1, y1 = c0 / image_width, r0 / image_height
            x2, y2 = c1 / image_width, r1 / image_height
        elif coordinate_system is CoordinateSystem.PIXEL:
            x1, y1, x2, y2 = float(c0), float(r0), float(c1), float(r1)
        else:
            raise ValueError(
                f"regions_to_schema cannot produce {coordinate_system.value} "
                f"coordinates; convert after this call"
            )

        box = Box(
            x1=x1, y1=y1, x2=x2, y2=y2,
            score=min(1.0, max(0.0, r.mean_probability)),
            label="change",
            coordinate_system=coordinate_system,
        )
        out.append(
            ChangeRegion(
                box=box,
                area_pixels=r.area_pixels,
                mean_probability=min(1.0, max(0.0, r.mean_probability)),
                coordinate_system=coordinate_system,
            )
        )
    return out


__all__ = [
    "DEFAULT_KERNEL_SIZE",
    "RegistrationQuality",
    "RegionStats",
    "PostprocessResult",
    "to_grayscale_float",
    "measure_registration",
    "morphological_cleanup",
    "connected_regions",
    "postprocess_change_map",
    "regions_to_schema",
]