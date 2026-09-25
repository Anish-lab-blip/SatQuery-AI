"""SatQuery AI — raster-to-displayable-image conversion.

One implementation, shared by every specialist that needs pixels. Both the VQA
specialist and the grounding specialist read a GeoTIFF and need a uint8 (H, W, 3)
array; duplicating that logic in two modules is how the two drift apart.

The percentile stretch is deterministic: the same file always yields the same
array, so a grounding box and a VQA answer describe identical pixels.

What this does NOT do
---------------------
It does not resample, crop, or reproject. Those change the pixel grid, and the
grounding specialist converts normalized boxes to pixel coordinates using the
ORIGINAL raster's dimensions — a silent resize here would put every box in the
wrong place. Size changes belong in the tiling stage, which records what it did.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.errors import SpecialistError


def load_image_array(path: str | Path) -> Any:
    """Read a raster and percentile-stretch it to a displayable (H, W, 3) uint8 array.

    Deterministic: the same file always yields the same array. The stretch uses
    only finite values, so a nodata sentinel does not crush the dynamic range.

    Args:
        path: TIFF/GeoTIFF path.

    Returns:
        (H, W, 3) uint8 numpy array.

    Raises:
        SpecialistError: numpy is unavailable, or the raster cannot be read.
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise SpecialistError(f"numpy not installed: {exc}") from exc

    from preprocessing.raster import read_bands

    array, _profile = read_bands(path)

    # (bands, H, W) -> (H, W, bands); take the first three bands as RGB.
    if array.ndim == 3:
        array = np.transpose(array, (1, 2, 0))
        if array.shape[2] > 3:
            array = array[:, :, :3]
        elif array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
    elif array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)

    arr = array.astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size:
        lo, hi = np.percentile(finite, (2, 98))
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def load_pil_image(path: str | Path) -> Any:
    """Read a raster as a displayable PIL image.

    Georeferencing is not lost: the caller's `AssetMetadata` already carries the
    CRS and transform, and this function never touches them.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise SpecialistError(f"PIL not installed: {exc}") from exc

    return Image.fromarray(load_image_array(path))


__all__ = ["load_image_array", "load_pil_image"]