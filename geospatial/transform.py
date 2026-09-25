"""SatQuery AI — geospatial coordinate transforms.

This module makes the `coordinate_system` field in core/schemas.py real. Every spatial
result the system produces passes through here at least once, so this is the single place
where a normalised 0-1 box, a pixel box, and a geographic box are converted between.

Non-negotiable rules (docs/ARCHITECTURE_FREEZE.md section 2.6):
  * CRS and affine transform are preserved, never silently stripped.
  * Conversions are explicit about their source and target frames.
  * A conversion that cannot be performed raises, rather than returning a plausible lie.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from affine import Affine

from core.errors import CoordinateError
from core.schemas import Box, CoordinateSystem, GeoMetadata

# ---------------------------------------------------------------------------
# Core frame types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PixelWindow:
    """A bounding region in pixel space: [col_min, row_min, col_max, row_max]."""

    col_min: float
    row_min: float
    col_max: float
    row_max: float

    def __post_init__(self) -> None:
        if self.col_max < self.col_min or self.row_max < self.row_min:
            raise CoordinateError(
                f"pixel window is inverted: {self.as_list()}"
            )

    def as_list(self) -> list[float]:
        return [self.col_min, self.row_min, self.col_max, self.row_max]

    @property
    def width(self) -> float:
        return self.col_max - self.col_min

    @property
    def height(self) -> float:
        return self.row_max - self.row_min

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class GeoBounds:
    """Geographic bounds in CRS units: (minx, miny, maxx, maxy)."""

    minx: float
    miny: float
    maxx: float
    maxy: float

    def as_list(self) -> list[float]:
        return [self.minx, self.miny, self.maxx, self.maxy]


# ---------------------------------------------------------------------------
# Normalised <-> pixel
# ---------------------------------------------------------------------------


def normalized_to_pixel(
    box: Sequence[float], width: int, height: int
) -> PixelWindow:
    """Convert a normalised [0,1] box to pixel coordinates.

    Values are NOT clipped: a box slightly outside the frame is preserved so the
    caller can decide whether that is a bug or a legitimate edge case.
    """
    if len(box) != 4:
        raise CoordinateError(f"expected 4 values, got {len(box)}")
    if width <= 0 or height <= 0:
        raise CoordinateError(f"invalid raster dimensions: {width}x{height}")

    x1, y1, x2, y2 = (float(v) for v in box)
    return PixelWindow(
        col_min=x1 * width,
        row_min=y1 * height,
        col_max=x2 * width,
        row_max=y2 * height,
    )


def pixel_to_normalized(
    window: PixelWindow, width: int, height: int
) -> list[float]:
    """Convert a pixel window back to a normalised [0,1] box."""
    if width <= 0 or height <= 0:
        raise CoordinateError(f"invalid raster dimensions: {width}x{height}")
    return [
        window.col_min / width,
        window.row_min / height,
        window.col_max / width,
        window.row_max / height,
    ]


# ---------------------------------------------------------------------------
# Pixel <-> geographic
# ---------------------------------------------------------------------------


def pixel_to_geo(
    window: PixelWindow, transform: Affine
) -> GeoBounds:
    """Convert a pixel window to geographic bounds using the raster's affine transform.

    rasterio/affine transforms map (col, row) -> (x, y). The y axis usually points
    down in pixel space, so the geographic miny comes from the BOTTOM row.
    """
    if transform is None:
        raise CoordinateError("cannot convert to geo: raster has no affine transform")

    # Corners. row_max is the bottom in pixel space -> smaller y in most CRS.
    # Affine supports both * and @; @ is the non-deprecated matmul form.
    left, top = transform @ (window.col_min, window.row_min)
    right, bottom = transform @ (window.col_max, window.row_max)

    return GeoBounds(
        minx=min(left, right),
        miny=min(top, bottom),
        maxx=max(left, right),
        maxy=max(top, bottom),
    )


def geo_to_pixel(bounds: GeoBounds, transform: Affine) -> PixelWindow:
    """Invert the affine transform to recover a pixel window from geographic bounds."""
    if transform is None:
        raise CoordinateError("cannot convert from geo: raster has no affine transform")

    try:
        inverse = ~transform
    except Exception as exc:  # noqa: BLE001
        raise CoordinateError(f"affine transform is not invertible: {exc}") from exc

    col_min, row_min = inverse @ (bounds.minx, bounds.maxy)
    col_max, row_max = inverse @ (bounds.maxx, bounds.miny)

    return PixelWindow(
        col_min=min(col_min, col_max),
        row_min=min(row_min, row_max),
        col_max=max(col_min, col_max),
        row_max=max(row_min, row_max),
    )


# ---------------------------------------------------------------------------
# Box-object conversions (schema-aware)
# ---------------------------------------------------------------------------


def to_normalized(
    box: Box, width: int | None = None, height: int | None = None,
    transform: Affine | None = None,
) -> Box:
    """Coerce any Box to normalised 0-1 coordinates.

    Requires enough context for the source frame:
      pixel -> width, height
      geo   -> width, height, transform
    """
    if box.coordinate_system is CoordinateSystem.NORMALIZED_0_1:
        return box

    if box.coordinate_system is CoordinateSystem.PIXEL:
        if width is None or height is None:
            raise CoordinateError("pixel->normalized requires width and height")
        win = PixelWindow(box.x1, box.y1, box.x2, box.y2)
        nx1, ny1, nx2, ny2 = pixel_to_normalized(win, width, height)
        return box.model_copy(
            update={
                "x1": nx1, "y1": ny1, "x2": nx2, "y2": ny2,
                "coordinate_system": CoordinateSystem.NORMALIZED_0_1,
            }
        )

    if box.coordinate_system is CoordinateSystem.GEO:
        if width is None or height is None or transform is None:
            raise CoordinateError(
                "geo->normalized requires width, height and the affine transform"
            )
        win = geo_to_pixel(GeoBounds(box.x1, box.y1, box.x2, box.y2), transform)
        nx1, ny1, nx2, ny2 = pixel_to_normalized(win, width, height)
        return box.model_copy(
            update={
                "x1": nx1, "y1": ny1, "x2": nx2, "y2": ny2,
                "coordinate_system": CoordinateSystem.NORMALIZED_0_1,
            }
        )

    raise CoordinateError(f"unsupported coordinate system: {box.coordinate_system}")


def to_pixel(
    box: Box, width: int, height: int, transform: Affine | None = None
) -> Box:
    """Coerce any Box to pixel coordinates."""
    if box.coordinate_system is CoordinateSystem.PIXEL:
        return box

    if box.coordinate_system is CoordinateSystem.NORMALIZED_0_1:
        win = normalized_to_pixel([box.x1, box.y1, box.x2, box.y2], width, height)
    elif box.coordinate_system is CoordinateSystem.GEO:
        if transform is None:
            raise CoordinateError("geo->pixel requires the affine transform")
        win = geo_to_pixel(GeoBounds(box.x1, box.y1, box.x2, box.y2), transform)
    else:
        raise CoordinateError(f"unsupported coordinate system: {box.coordinate_system}")

    return box.model_copy(
        update={
            "x1": win.col_min, "y1": win.row_min,
            "x2": win.col_max, "y2": win.row_max,
            "coordinate_system": CoordinateSystem.PIXEL,
        }
    )


def to_geo(box: Box, width: int, height: int, transform: Affine) -> Box:
    """Coerce any Box to geographic coordinates."""
    if box.coordinate_system is CoordinateSystem.GEO:
        return box

    pixel_box = to_pixel(box, width, height, transform)
    bounds = pixel_to_geo(
        PixelWindow(pixel_box.x1, pixel_box.y1, pixel_box.x2, pixel_box.y2),
        transform,
    )
    return box.model_copy(
        update={
            "x1": bounds.minx, "y1": bounds.miny,
            "x2": bounds.maxx, "y2": bounds.maxy,
            "coordinate_system": CoordinateSystem.GEO,
        }
    )


# ---------------------------------------------------------------------------
# Intersection / IoU — used by grounding NMS and change-region overlap
# ---------------------------------------------------------------------------


def intersect(a: PixelWindow, b: PixelWindow) -> PixelWindow | None:
    """Axis-aligned intersection of two pixel windows, or None if disjoint."""
    col_min = max(a.col_min, b.col_min)
    row_min = max(a.row_min, b.row_min)
    col_max = min(a.col_max, b.col_max)
    row_max = min(a.row_max, b.row_max)
    if col_max <= col_min or row_max <= row_min:
        return None
    return PixelWindow(col_min, row_min, col_max, row_max)


def iou(a: PixelWindow, b: PixelWindow) -> float:
    """Intersection-over-union of two axis-aligned windows. Range [0, 1]."""
    inter = intersect(a, b)
    if inter is None:
        return 0.0
    union = a.area + b.area - inter.area
    if union <= 0:
        return 0.0
    return inter.area / union


# ---------------------------------------------------------------------------
# Normalisation helper for the 0-100 benchmark convention
# ---------------------------------------------------------------------------


def benchmark_boxes_to_normalized(
    boxes: Iterable[Sequence[float]], scale: float = 100.0
) -> list[list[float]]:
    """Convert benchmark boxes (e.g. VRSBench's 0-100) to our internal 0-1.

    Finding from Phase 0: VRSBench states verbatim that "all box coordinates are
    normalized to 0-100". Treating those numbers as pixels is a silent, catastrophic
    bug — this function exists so that mistake can only be made once, explicitly.
    """
    if scale <= 0:
        raise CoordinateError(f"box scale must be positive, got {scale}")
    out: list[list[float]] = []
    for b in boxes:
        if len(b) != 4:
            raise CoordinateError(f"expected 4 values per box, got {len(b)}")
        out.append([float(v) / scale for v in b])
    return out


def normalized_boxes_to_benchmark(
    boxes: Iterable[Sequence[float]], scale: float = 100.0
) -> list[list[float]]:
    """Inverse of benchmark_boxes_to_normalized."""
    if scale <= 0:
        raise CoordinateError(f"box scale must be positive, got {scale}")
    out: list[list[float]] = []
    for b in boxes:
        if len(b) != 4:
            raise CoordinateError(f"expected 4 values per box, got {len(b)}")
        out.append([float(v) * scale for v in b])
    return out


# ---------------------------------------------------------------------------
# GeoMetadata convenience
# ---------------------------------------------------------------------------


def affine_from_metadata(geo: GeoMetadata) -> Affine | None:
    """Reconstruct an Affine from a GeoMetadata transform list."""
    if not geo.transform or len(geo.transform) != 6:
        return None
    a, b, c, d, e, f = geo.transform
    return Affine(a, b, c, d, e, f)


__all__ = [
    "PixelWindow",
    "GeoBounds",
    "normalized_to_pixel",
    "pixel_to_normalized",
    "pixel_to_geo",
    "geo_to_pixel",
    "to_normalized",
    "to_pixel",
    "to_geo",
    "intersect",
    "iou",
    "benchmark_boxes_to_normalized",
    "normalized_boxes_to_benchmark",
    "affine_from_metadata",
]