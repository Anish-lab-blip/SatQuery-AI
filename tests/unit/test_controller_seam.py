"""The controller's asset-resolution seam — regression tests for a real defect.

THE DEFECT
----------
`AnalysisController._resolve_assets` built metadata from the path alone:

    return [AssetMetadata(path=path) for path in request.assets]

`AssetMetadata`'s `geo` and `modality` are defaulted, so every asset arrived at
every specialist with `band_count=None`, `modality=UNKNOWN`, `crs=None` — even
when the file on disk was a perfectly ordinary 4-band georeferenced GeoTIFF.

The consequence is user-visible. `OpticalSarSpecialist._infer_modality`
(specialists/optical_sar/specialist.py:295-315) reads `asset.geo.band_count`;
with `None` it returns `Modality.UNKNOWN` for both assets, so the pair is
indeterminate and `validate_request` rejects it. A valid optical/SAR pair is
refused as unpairable. The same loss starves every modality- and
resolution-dependent branch in every specialist.

These tests are written to FAIL against the old implementation, so they pin the
seam rather than describe it. Each one constructs a real GeoTIFF: the defect
only manifests against real files, because reading nothing is what produced it.

`inspect_raster` (preprocessing/raster.py:96) already existed and already did
exactly this work — it had zero callers before this fix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core.controller import AnalysisController
from core.errors import RasterReadError
from core.planner import PolicyPlanner
from core.registry import SpecialistRegistry
from core.schemas import AnalysisRequest, Modality, Task

rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_geotiff(
    path: Path,
    *,
    count: int,
    width: int = 64,
    height: int = 64,
    crs: str | None = "EPSG:32643",
    dtype: str = "uint16",
) -> Path:
    """A real, small, georeferenced GeoTIFF. No stub bytes."""
    transform = from_origin(500_000.0, 4_000_000.0, 10.0, 10.0) if crs else None
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": dtype,
    }
    if crs:
        profile["crs"] = crs
        profile["transform"] = transform
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(np.zeros((count, height, width), dtype=dtype))
    return path


class _Cfg:
    device_preference = "cpu"

    def get(self, path: str, default: Any = None) -> Any:
        return default

    @property
    def hash(self) -> str:
        return "seamcfg000000000"


def _controller() -> AnalysisController:
    registry = SpecialistRegistry.discover(_Cfg())
    return AnalysisController(registry=registry, planner=PolicyPlanner(registry))


# ===========================================================================
# TEST 1 — band count and modality survive resolution
# ===========================================================================
def test_resolved_assets_carry_band_count_and_modality(tmp_path: Path) -> None:
    """A real 4-band optical and 2-band SAR must resolve with real metadata.

    This is the core of the defect. Before the fix both assets came back with
    `band_count=None` and `modality=UNKNOWN`, which is indistinguishable from
    two unreadable files.
    """
    optical = _write_geotiff(tmp_path / "opt_4band.tif", count=4)
    sar = _write_geotiff(tmp_path / "sar_2band.tif", count=2)

    resolved = _controller()._resolve_assets(
        AnalysisRequest(assets=[str(optical), str(sar)], query="compare these")
    )

    assert len(resolved) == 2
    assert resolved[0].geo.band_count == 4
    assert resolved[1].geo.band_count == 2
    # `infer_modality`: 4 bands -> optical; 2 bands -> sar.
    assert resolved[0].modality is Modality.OPTICAL
    assert resolved[1].modality is Modality.SAR
    # And the pixels/dtype are described, not just the band count.
    assert resolved[0].geo.width == 64
    assert resolved[0].geo.height == 64
    assert resolved[0].geo.dtype == "uint16"


# ===========================================================================
# TEST 2 — the optical/SAR pair is no longer refused as indeterminate
# ===========================================================================
def test_optical_sar_pair_is_accepted_end_to_end(tmp_path: Path) -> None:
    """The user-visible consequence: a valid pair must not be refused.

    Before the fix, both assets inferred `UNKNOWN`, the pair was indeterminate,
    and `OpticalSarSpecialist.validate_request` raised `InvalidRequestError`.
    """
    from specialists.base import SpecialistRequest
    from specialists.optical_sar.specialist import OpticalSarSpecialist

    optical = _write_geotiff(tmp_path / "opt_4band.tif", count=4)
    sar = _write_geotiff(tmp_path / "sar_2band.tif", count=2)

    resolved = _controller()._resolve_assets(
        AnalysisRequest(assets=[str(optical), str(sar)], query="compare these")
    )

    specialist = OpticalSarSpecialist.__new__(OpticalSarSpecialist)
    specialist.name = "optical_sar"
    specialist.version = "0.1.0"
    specialist.capabilities = ("optical_sar",)

    # Must not raise. Before the fix this raised InvalidRequestError because
    # neither asset could be identified as optical or SAR.
    specialist.validate_request(
        SpecialistRequest(assets=resolved, query="compare these")
    )


# ===========================================================================
# TEST 3 — geospatial metadata survives, and unreadable inputs are typed
# ===========================================================================
def test_resolved_assets_carry_crs_and_resolution(tmp_path: Path) -> None:
    """Georeferencing must not be dropped on the way to the specialists.

    A specialist that cannot see the CRS cannot report a geolocation or check
    that a pair shares a datum, and `GeoMetadata.has_crs` stays False, which
    silently changes what the evidence engine will emit.
    """
    optical = _write_geotiff(tmp_path / "geo.tif", count=4, crs="EPSG:32643")

    resolved = _controller()._resolve_assets(
        AnalysisRequest(assets=[str(optical)], query="where is this")
    )

    geo = resolved[0].geo
    assert geo.crs == "EPSG:32643"
    assert geo.has_crs is True
    assert geo.is_georeferenced is True
    assert geo.transform is not None and len(geo.transform) == 6
    assert geo.bounds is not None and len(geo.bounds) == 4
    assert geo.resolution is not None and geo.resolution[0] == pytest.approx(10.0)
    assert geo.band_count == 4


def test_unreadable_asset_raises_a_typed_error(tmp_path: Path) -> None:
    """A non-raster must fail with `RasterReadError`, not a bare exception.

    The controller maps typed codes to user messages, so the failure has to be
    typed. A missing file is the cheapest way to prove the error path survives
    the wiring.
    """
    missing = tmp_path / "does_not_exist.tif"
    with pytest.raises(RasterReadError):
        _controller()._resolve_assets(
            AnalysisRequest(assets=[str(missing)], query="what is here")
        )


def test_non_raster_file_raises_a_typed_error(tmp_path: Path) -> None:
    """A text file named `.tif` is not a raster and must be reported as such."""
    bogus = tmp_path / "not_a_raster.tif"
    bogus.write_text("this is not a tiff", encoding="utf-8")
    with pytest.raises(RasterReadError):
        _controller()._resolve_assets(
            AnalysisRequest(assets=[str(bogus)], query="what is here")
        )


def test_declared_modalities_override_inference(tmp_path: Path) -> None:
    """An explicit modality is authoritative; inference is the fallback.

    `inspect_raster(explicit_modality=...)` already exists for this. A 1-band
    file is ambiguous by count, and a caller who knows it is SAR must be able
    to say so rather than having the heuristic decide.
    """
    one_band = _write_geotiff(tmp_path / "ambiguous_1band.tif", count=1)

    resolved = _controller()._resolve_assets(
        AnalysisRequest(assets=[str(one_band)], query="what is here"),
        asset_modalities={"ambiguous_1band.tif": "sar"},
    )

    assert resolved[0].modality is Modality.SAR


def test_unknown_declared_modality_is_rejected(tmp_path: Path) -> None:
    """A nonsense declaration is a caller error, not silently ignored."""
    optical = _write_geotiff(tmp_path / "opt.tif", count=4)
    with pytest.raises(ValueError):
        _controller()._resolve_assets(
            AnalysisRequest(assets=[str(optical)], query="what is here"),
            asset_modalities={"opt.tif": "microwave"},
        )


def test_asset_modalities_keyed_by_full_path_also_works(tmp_path: Path) -> None:
    """Callers may key by basename or by full path; both must resolve.

    The GUI hands over file-picker paths while a batch caller has bare names,
    and requiring one spelling would make the other silently fall back to
    inference.
    """
    optical = _write_geotiff(tmp_path / "opt.tif", count=1)

    resolved = _controller()._resolve_assets(
        AnalysisRequest(assets=[str(optical)], query="what is here"),
        asset_modalities={str(optical): "optical"},
    )

    assert resolved[0].modality is Modality.OPTICAL


def test_resolution_is_still_lazy_about_pixels(tmp_path: Path) -> None:
    """`inspect_raster` must describe without decoding the pixel array.

    The reason the controller does not simply call `read_bands` is that a
    12-band Sentinel-2 tile is ~100 MB and a 4-specialist plan would decode it
    several times. `inspect_raster` opens the header only, so resolution stays
    cheap enough to do once per asset.
    """
    big = _write_geotiff(tmp_path / "big.tif", count=12, width=2048, height=2048)

    resolved = _controller()._resolve_assets(
        AnalysisRequest(assets=[str(big)], query="describe")
    )

    assert resolved[0].geo.band_count == 12
    assert resolved[0].geo.width == 2048
    # No pixel data is held on the metadata — only the description.
    assert not hasattr(resolved[0], "array")
    assert resolved[0].sha256 is None  # hashing is opt-in: it reads the file
