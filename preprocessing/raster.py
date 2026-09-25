"""SatQuery AI — GeoTIFF / raster input contract.

The first thing every uploaded asset meets. Implements the validation chain from the
frozen architecture:

    file -> dimensions -> bands -> dtype -> CRS -> transform -> bounds -> nodata
         -> modality -> temporal metadata

Design rules:
  * Never raise a bare exception. Every failure is a typed SatQueryError.
  * Never silently drop geospatial metadata. If the source had a CRS, the returned
    AssetMetadata says so.
  * A missing CRS degrades to non-geospatial mode; it does not abort.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from core.errors import (
    OversizedImageError,
    RasterReadError,
    UnsupportedBandsError,
)
from core.schemas import AssetMetadata, GeoMetadata, Modality, SensorDescriptor

# ---------------------------------------------------------------------------
# Modality inference
# ---------------------------------------------------------------------------

#: Band-count heuristics for modality detection when explicit metadata is absent.
#: These are heuristics, not ground truth — the sensor adapter is authoritative
#: when a sensor descriptor is supplied.
_OPTICAL_BAND_COUNTS = {3, 4, 8, 11, 12, 13}
_SAR_BAND_COUNTS = {1, 2}


def _safe_is_tiled(dataset: Any) -> bool:
    """Read the tiled flag without tripping rasterio's pending deprecation.

    `DatasetReader.is_tiled` is scheduled for removal. It is purely informational
    metadata for us, so a missing value degrades to False rather than aborting.
    """
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            return bool(dataset.is_tiled)
    except Exception:  # noqa: BLE001 - metadata only, never fatal
        return False


def infer_modality(band_count: int, explicit: str | None = None) -> Modality:
    """Infer modality from band count, honouring an explicit label if given."""
    if explicit:
        try:
            return Modality(explicit.lower())
        except ValueError:
            pass

    if band_count in _SAR_BAND_COUNTS:
        return Modality.SAR
    if band_count in _OPTICAL_BAND_COUNTS:
        return Modality.OPTICAL
    # 12-band optical and 2-band SAR are both plausible; default to optical only
    # when the count clearly favours it.
    if band_count >= 4:
        return Modality.OPTICAL
    return Modality.UNKNOWN


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def file_sha256(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file. Used for dataset manifests and dedup detection."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_bytes):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def inspect_raster(
    path: str | Path,
    *,
    max_pixels: int | None = None,
    explicit_modality: str | None = None,
    compute_hash: bool = False,
    sensor: SensorDescriptor | None = None,
) -> AssetMetadata:
    """Validate and describe a raster without loading its pixel data.

    Args:
        path: TIFF/GeoTIFF path.
        max_pixels: optional pixel budget; exceeding it raises OversizedImageError
            (recoverable — the caller may downscale).
        explicit_modality: override modality inference.
        compute_hash: compute the file SHA-256 (slower; needed for manifests).
        sensor: optional sensor descriptor from the sensor adapter.

    Returns:
        AssetMetadata with full geospatial detail preserved.

    Raises:
        RasterReadError: the file is not a readable raster.
        OversizedImageError: the raster exceeds the configured pixel budget.
    """
    p = Path(path)
    if not p.exists():
        raise RasterReadError(f"file does not exist: {p}")
    if not p.is_file():
        raise RasterReadError(f"not a regular file: {p}")

    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RasterReadError(f"rasterio is not installed: {exc}") from exc

    try:
        with rasterio.open(p) as ds:
            width, height = ds.width, ds.height
            band_count = ds.count
            dtypes = sorted({str(d) for d in ds.dtypes})
            crs_value = ds.crs.to_string() if ds.crs else None
            transform = list(ds.transform)[:6] if ds.transform else None
            bounds = list(ds.bounds) if ds.bounds else None
            nodata = ds.nodata
            resolution = list(ds.res) if ds.res else None
            driver = ds.driver
            is_tiled = _safe_is_tiled(ds)
    except RasterReadError:
        raise
    except Exception as exc:  # noqa: BLE001 - rasterio raises many types
        raise RasterReadError(f"could not open raster '{p}': {exc}") from exc

    if width <= 0 or height <= 0:
        raise RasterReadError(f"raster has degenerate dimensions {width}x{height}")

    if max_pixels is not None and width * height > max_pixels:
        raise OversizedImageError(
            f"raster is {width * height} pixels, exceeding the budget of {max_pixels}",
            context={"width": width, "height": height, "max_pixels": max_pixels},
        )

    if band_count <= 0:
        raise UnsupportedBandsError(f"raster reports {band_count} bands")

    geo = GeoMetadata(
        crs=crs_value,
        transform=transform,
        bounds=bounds,
        width=width,
        height=height,
        band_count=band_count,
        dtype=",".join(dtypes) if dtypes else None,
        nodata=float(nodata) if nodata is not None else None,
        resolution=resolution,
        has_crs=crs_value is not None,
        is_georeferenced=bool(crs_value and transform),
        driver=driver,
        is_tiled=is_tiled,
    )

    modality = infer_modality(band_count, explicit_modality)

    return AssetMetadata(
        path=str(p.resolve()),
        modality=modality,
        sha256=file_sha256(p) if compute_hash else None,
        geo=geo,
        sensor=sensor,
    )


def read_bands(
    path: str | Path,
    *,
    indexes: list[int] | None = None,
    out_dtype: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read raster bands as a (bands, H, W) array plus its profile.

    The profile retains crs/transform so downstream code never loses georeferencing.
    """
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover
        raise RasterReadError(f"rasterio is not installed: {exc}") from exc

    p = Path(path)
    try:
        with rasterio.open(p) as ds:
            arr = ds.read(indexes=indexes) if indexes else ds.read()
            profile = ds.profile.copy()
            profile["crs"] = ds.crs.to_string() if ds.crs else None
            profile["transform"] = list(ds.transform)[:6] if ds.transform else None
            profile["bounds"] = list(ds.bounds) if ds.bounds else None
            profile["nodata"] = ds.nodata
    except Exception as exc:  # noqa: BLE001
        raise RasterReadError(f"could not read bands from '{p}': {exc}") from exc

    if out_dtype and arr.dtype != np.dtype(out_dtype):
        arr = arr.astype(out_dtype)

    return arr, profile


def write_raster(
    path: str | Path,
    array: np.ndarray,
    profile: dict[str, Any],
) -> Path:
    """Write an array using a profile captured by read_bands.

    Preserves crs/transform/bounds from the profile. Used by evidence generation
    (change maps, masks) so those artifacts stay georeferenced.
    """
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover
        raise RasterReadError(f"rasterio is not installed: {exc}") from exc

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    out_profile = dict(profile)
    if array.ndim == 2:
        out_profile.update(count=1, height=array.shape[0], width=array.shape[1],
                           dtype=str(array.dtype))
    else:
        out_profile.update(count=array.shape[0], height=array.shape[1],
                           width=array.shape[2], dtype=str(array.dtype))

    try:
        with rasterio.open(p, "w", **{k: v for k, v in out_profile.items()
                                      if k not in {"crs", "transform", "bounds"}},
                           crs=profile.get("crs"),
                           transform=profile.get("transform")) as ds:
            ds.write(array if array.ndim == 3 else array[np.newaxis, ...])
    except Exception as exc:  # noqa: BLE001
        raise RasterReadError(f"could not write raster '{p}': {exc}") from exc

    return p


def pixel_area_m2(meta: AssetMetadata) -> float | None:
    """Area of one pixel in square metres, when the CRS is metric.

    Returns None for geographic CRS or unknown resolution — callers must not guess.
    """
    from geospatial.crs import is_metric

    if not is_metric(meta.geo.crs):
        return None
    if not meta.geo.resolution or len(meta.geo.resolution) < 2:
        return None
    xres, yres = meta.geo.resolution[0], meta.geo.resolution[1]
    return abs(xres * yres)


__all__ = [
    "infer_modality",
    "file_sha256",
    "inspect_raster",
    "read_bands",
    "write_raster",
    "pixel_area_m2",
]