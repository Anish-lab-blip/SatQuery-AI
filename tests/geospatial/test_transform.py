"""Phase 2 tests — geospatial transforms and the raster input contract.

These tests build real GeoTIFFs in a temp directory and round-trip coordinates through
them. Nothing here is mocked: if the affine arithmetic is wrong, these fail.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from affine import Affine

from core.errors import (
    CoordinateError,
    MissingCRSError,
    OversizedImageError,
    RasterReadError,
)
from core.schemas import Box, CoordinateSystem
from geospatial.crs import compare_crs, describe_crs, is_metric, parse_crs
from geospatial.transform import (
    GeoBounds,
    PixelWindow,
    affine_from_metadata,
    benchmark_boxes_to_normalized,
    geo_to_pixel,
    intersect,
    iou,
    normalized_boxes_to_benchmark,
    normalized_to_pixel,
    pixel_to_geo,
    pixel_to_normalized,
    to_geo,
    to_normalized,
    to_pixel,
)
from preprocessing.raster import (
    file_sha256,
    infer_modality,
    inspect_raster,
    pixel_area_m2,
    read_bands,
    write_raster,
)

# ---------------------------------------------------------------------------
# Fixtures: a real 10 m/px UTM GeoTIFF
# ---------------------------------------------------------------------------

# UTM zone 43N — plausible for Indian subcontinent imagery
UTM43N = "EPSG:32643"
ORIGIN_X = 500_000.0
ORIGIN_Y = 2_500_000.0
PIXEL_SIZE = 10.0
WIDTH = 64
HEIGHT = 64

# rasterio/affine convention: (a, b, c, d, e, f) maps (col,row) -> (x,y)
# y decreases as row increases (north-up)
TEST_TRANSFORM = Affine(PIXEL_SIZE, 0.0, ORIGIN_X, 0.0, -PIXEL_SIZE, ORIGIN_Y)


@pytest.fixture()
def geotiff(tmp_path: Path) -> Path:
    """A small, georeferenced 3-band optical GeoTIFF."""
    import rasterio

    path = tmp_path / "sample.tif"
    data = np.arange(3 * HEIGHT * WIDTH, dtype=np.uint16).reshape(3, HEIGHT, WIDTH)
    with rasterio.open(
        path, "w", driver="GTiff", width=WIDTH, height=HEIGHT, count=3,
        dtype="uint16", crs=UTM43N, transform=TEST_TRANSFORM, nodata=0,
    ) as ds:
        ds.write(data)
    return path


@pytest.fixture()
def plain_tiff(tmp_path: Path) -> Path:
    """A TIFF with no CRS and no transform — the non-geospatial case."""
    import rasterio

    path = tmp_path / "plain.tif"
    data = np.zeros((1, 32, 32), dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", width=32, height=32, count=1, dtype="uint8",
    ) as ds:
        ds.write(data)
    return path


@pytest.fixture()
def sar_tiff(tmp_path: Path) -> Path:
    """A 2-band SAR GeoTIFF (VV, VH)."""
    import rasterio

    path = tmp_path / "sar.tif"
    data = np.full((2, 32, 32), 1000, dtype=np.uint16)
    with rasterio.open(
        path, "w", driver="GTiff", width=32, height=32, count=2, dtype="uint16",
        crs=UTM43N, transform=Affine(PIXEL_SIZE, 0, ORIGIN_X, 0, -PIXEL_SIZE, ORIGIN_Y),
    ) as ds:
        ds.write(data)
    return path


# ---------------------------------------------------------------------------
# Normalised <-> pixel
# ---------------------------------------------------------------------------


def test_normalized_to_pixel_full_frame() -> None:
    win = normalized_to_pixel([0.0, 0.0, 1.0, 1.0], 100, 200)
    assert win.as_list() == [0.0, 0.0, 100.0, 200.0]


def test_normalized_pixel_roundtrip() -> None:
    original = [0.25, 0.5, 0.75, 1.0]
    win = normalized_to_pixel(original, 400, 800)
    back = pixel_to_normalized(win, 400, 800)
    assert back == pytest.approx(original)


def test_normalized_to_pixel_rejects_bad_dimensions() -> None:
    with pytest.raises(CoordinateError):
        normalized_to_pixel([0, 0, 1, 1], 0, 10)


def test_pixel_window_rejects_inverted() -> None:
    with pytest.raises(CoordinateError):
        PixelWindow(10, 10, 5, 5)


def test_pixel_window_area_and_extent() -> None:
    w = PixelWindow(0, 0, 10, 20)
    assert w.width == 10 and w.height == 20 and w.area == 200


# ---------------------------------------------------------------------------
# Pixel <-> geographic
# ---------------------------------------------------------------------------


def test_pixel_origin_maps_to_transform_origin() -> None:
    win = PixelWindow(0, 0, 0, 0)
    g = pixel_to_geo(win, TEST_TRANSFORM)
    assert g.minx == pytest.approx(ORIGIN_X)
    assert g.maxy == pytest.approx(ORIGIN_Y)


def test_pixel_to_geo_handles_north_up_axis() -> None:
    # A 10x10 pixel box at the top-left
    win = PixelWindow(0, 0, 10, 10)
    g = pixel_to_geo(win, TEST_TRANSFORM)
    assert g.minx == pytest.approx(ORIGIN_X)
    assert g.maxx == pytest.approx(ORIGIN_X + 100.0)   # 10 px * 10 m
    assert g.maxy == pytest.approx(ORIGIN_Y)           # top row is the max y
    assert g.miny == pytest.approx(ORIGIN_Y - 100.0)   # bottom row is the min y


def test_pixel_geo_roundtrip() -> None:
    win = PixelWindow(7, 11, 23, 41)
    bounds = pixel_to_geo(win, TEST_TRANSFORM)
    back = geo_to_pixel(bounds, TEST_TRANSFORM)
    assert back.as_list() == pytest.approx(win.as_list())


def test_geo_to_pixel_rejects_missing_transform() -> None:
    with pytest.raises(CoordinateError):
        pixel_to_geo(PixelWindow(0, 0, 1, 1), None)  # type: ignore[arg-type]


def test_pixel_to_geo_rejects_missing_transform() -> None:
    with pytest.raises(CoordinateError):
        geo_to_pixel(GeoBounds(0, 0, 1, 1), None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Box-object conversions — C-5 made real
# ---------------------------------------------------------------------------


def test_box_to_normalized_is_identity_for_normalized() -> None:
    b = Box(x1=0.1, y1=0.2, x2=0.3, y2=0.4)
    assert to_normalized(b) is b


def test_pixel_box_to_normalized() -> None:
    b = Box(x1=0, y1=0, x2=50, y2=100, coordinate_system=CoordinateSystem.PIXEL)
    out = to_normalized(b, width=100, height=200)
    assert out.coordinate_system is CoordinateSystem.NORMALIZED_0_1
    assert [out.x1, out.y1, out.x2, out.y2] == pytest.approx([0.0, 0.0, 0.5, 0.5])


def test_pixel_to_normalized_requires_dimensions() -> None:
    b = Box(x1=0, y1=0, x2=1, y2=1, coordinate_system=CoordinateSystem.PIXEL)
    with pytest.raises(CoordinateError):
        to_normalized(b)


def test_normalized_box_to_pixel() -> None:
    b = Box(x1=0.0, y1=0.0, x2=0.5, y2=0.5)
    out = to_pixel(b, width=200, height=400)
    assert out.coordinate_system is CoordinateSystem.PIXEL
    assert [out.x1, out.y1, out.x2, out.y2] == pytest.approx([0, 0, 100, 200])


def test_normalized_box_to_geo_and_back() -> None:
    b = Box(x1=0.0, y1=0.0, x2=0.5, y2=0.5)
    geo_box = to_geo(b, WIDTH, HEIGHT, TEST_TRANSFORM)
    assert geo_box.coordinate_system is CoordinateSystem.GEO
    # top-left half of a 64x64 @10m image = 320m x 320m
    assert geo_box.x2 - geo_box.x1 == pytest.approx(320.0)
    assert geo_box.y2 - geo_box.y1 == pytest.approx(320.0)

    back = to_normalized(
        geo_box, width=WIDTH, height=HEIGHT, transform=TEST_TRANSFORM
    )
    assert [back.x1, back.y1, back.x2, back.y2] == pytest.approx([0.0, 0.0, 0.5, 0.5])


def test_geo_to_pixel_requires_transform() -> None:
    b = Box(x1=0, y1=0, x2=1, y2=1, coordinate_system=CoordinateSystem.GEO)
    with pytest.raises(CoordinateError):
        to_pixel(b, width=10, height=10)


# ---------------------------------------------------------------------------
# Intersection / IoU — used by grounding NMS
# ---------------------------------------------------------------------------


def test_intersect_disjoint_returns_none() -> None:
    assert intersect(PixelWindow(0, 0, 10, 10), PixelWindow(20, 20, 30, 30)) is None


def test_intersect_overlapping() -> None:
    got = intersect(PixelWindow(0, 0, 10, 10), PixelWindow(5, 5, 15, 15))
    assert got is not None
    assert got.as_list() == [5, 5, 10, 10]


def test_iou_identical_is_one() -> None:
    w = PixelWindow(0, 0, 10, 10)
    assert iou(w, w) == pytest.approx(1.0)


def test_iou_disjoint_is_zero() -> None:
    assert iou(PixelWindow(0, 0, 10, 10), PixelWindow(50, 50, 60, 60)) == 0.0


def test_iou_half_overlap() -> None:
    # two 10x10 squares sharing half their width
    a = PixelWindow(0, 0, 10, 10)
    b = PixelWindow(5, 0, 15, 10)
    # inter = 50, union = 150
    assert iou(a, b) == pytest.approx(50 / 150)


# ---------------------------------------------------------------------------
# Benchmark 0-100 convention (Phase 0 finding, verified against VRSBench)
# ---------------------------------------------------------------------------


def test_benchmark_boxes_scale_to_normalized() -> None:
    got = benchmark_boxes_to_normalized([[0, 0, 50, 100]])
    assert got == [[0.0, 0.0, 0.5, 1.0]]


def test_benchmark_roundtrip() -> None:
    original = [[12.5, 30.0, 88.0, 99.9]]
    norm = benchmark_boxes_to_normalized(original)
    back = normalized_boxes_to_benchmark(norm)
    # pytest.approx does not accept nested sequences; compare the inner box.
    assert len(back) == 1
    assert back[0] == pytest.approx(original[0])


def test_benchmark_rejects_non_positive_scale() -> None:
    with pytest.raises(CoordinateError):
        benchmark_boxes_to_normalized([[0, 0, 1, 1]], scale=0)


# ---------------------------------------------------------------------------
# CRS
# ---------------------------------------------------------------------------


def test_parse_crs_none_and_valid() -> None:
    assert parse_crs(None) is None
    assert parse_crs("EPSG:32643") is not None


def test_parse_crs_rejects_garbage() -> None:
    with pytest.raises(MissingCRSError):
        parse_crs("not-a-crs")


def test_compare_crs_identical() -> None:
    c = compare_crs("EPSG:32643", "EPSG:32643")
    assert c.compatible and c.identical and not c.requires_reprojection


def test_compare_crs_different_projected_requires_reprojection() -> None:
    c = compare_crs("EPSG:32643", "EPSG:32644")
    assert c.compatible and not c.identical and c.requires_reprojection


def test_compare_crs_missing_is_incompatible() -> None:
    c = compare_crs(None, "EPSG:32643")
    assert not c.compatible
    assert "lack a CRS" in c.reason


def test_is_metric_true_for_utm() -> None:
    assert is_metric("EPSG:32643") is True


def test_is_metric_false_for_wgs84() -> None:
    assert is_metric("EPSG:4326") is False


def test_describe_crs_reports_units() -> None:
    d = describe_crs("EPSG:32643")
    assert d["present"] is True
    assert d["is_projected"] is True


# ---------------------------------------------------------------------------
# Raster inspection
# ---------------------------------------------------------------------------


def test_inspect_geotiff_preserves_crs_and_transform(geotiff: Path) -> None:
    meta = inspect_raster(geotiff)
    assert meta.geo.has_crs is True
    assert meta.geo.is_georeferenced is True
    assert meta.geo.crs is not None and "32643" in meta.geo.crs
    assert meta.geo.width == WIDTH and meta.geo.height == HEIGHT
    assert meta.geo.band_count == 3
    assert meta.geo.transform is not None and len(meta.geo.transform) == 6
    assert meta.geo.nodata == 0.0
    assert meta.modality.value == "optical"


def test_inspect_plain_tiff_degrades_without_crs(plain_tiff: Path) -> None:
    meta = inspect_raster(plain_tiff)
    assert meta.geo.has_crs is False
    assert meta.geo.is_georeferenced is False
    # must still be readable — missing CRS is a degradation, not a failure
    assert meta.geo.width == 32 and meta.geo.height == 32


def test_inspect_detects_sar_from_two_bands(sar_tiff: Path) -> None:
    meta = inspect_raster(sar_tiff)
    assert meta.modality.value == "sar"
    assert meta.geo.band_count == 2


def test_explicit_modality_overrides_inference(geotiff: Path) -> None:
    meta = inspect_raster(geotiff, explicit_modality="sar")
    assert meta.modality.value == "sar"


def test_inspect_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RasterReadError):
        inspect_raster(tmp_path / "nope.tif")


def test_inspect_non_raster_raises(tmp_path: Path) -> None:
    bad = tmp_path / "not_a_raster.tif"
    bad.write_text("this is not a tiff")
    with pytest.raises(RasterReadError):
        inspect_raster(bad)


def test_oversized_image_is_recoverable(geotiff: Path) -> None:
    with pytest.raises(OversizedImageError) as exc:
        inspect_raster(geotiff, max_pixels=10)
    assert exc.value.recoverable is True


def test_inspect_with_hash_produces_sha256(geotiff: Path) -> None:
    meta = inspect_raster(geotiff, compute_hash=True)
    assert meta.sha256 is not None and len(meta.sha256) == 64


def test_hash_is_stable_and_content_sensitive(geotiff: Path, plain_tiff: Path) -> None:
    assert file_sha256(geotiff) == file_sha256(geotiff)
    assert file_sha256(geotiff) != file_sha256(plain_tiff)


# ---------------------------------------------------------------------------
# Band reading / writing preserves georeferencing
# ---------------------------------------------------------------------------


def test_read_bands_shape_and_profile(geotiff: Path) -> None:
    arr, profile = read_bands(geotiff)
    assert arr.shape == (3, HEIGHT, WIDTH)
    assert profile["crs"] is not None
    assert profile["transform"] is not None


def test_read_specific_band_index(geotiff: Path) -> None:
    arr, _ = read_bands(geotiff, indexes=[1])
    assert arr.shape == (1, HEIGHT, WIDTH)


def test_write_raster_preserves_georeferencing(geotiff: Path, tmp_path: Path) -> None:
    arr, profile = read_bands(geotiff, indexes=[1])
    out = tmp_path / "out" / "band1.tif"
    write_raster(out, arr, profile)
    assert out.exists()

    meta = inspect_raster(out)
    assert meta.geo.has_crs is True
    assert meta.geo.width == WIDTH


def test_pixel_area_metric_crs(geotiff: Path) -> None:
    meta = inspect_raster(geotiff)
    # 10 m x 10 m pixels
    assert pixel_area_m2(meta) == pytest.approx(100.0)


def test_pixel_area_none_for_non_metric(plain_tiff: Path) -> None:
    meta = inspect_raster(plain_tiff)
    assert pixel_area_m2(meta) is None


# ---------------------------------------------------------------------------
# Modality inference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bands,expected",
    [
        (1, "sar"),
        (2, "sar"),
        (3, "optical"),
        (4, "optical"),
        (12, "optical"),
        (13, "optical"),
    ],
)
def test_infer_modality_by_band_count(bands: int, expected: str) -> None:
    assert infer_modality(bands).value == expected


def test_infer_modality_explicit_wins() -> None:
    assert infer_modality(3, explicit="sar").value == "sar"


def test_infer_modality_unknown_explicit_falls_back() -> None:
    assert infer_modality(3, explicit="nonsense").value == "optical"


# ---------------------------------------------------------------------------
# affine_from_metadata round-trip
# ---------------------------------------------------------------------------


def test_affine_from_metadata_roundtrip(geotiff: Path) -> None:
    meta = inspect_raster(geotiff)
    affine = affine_from_metadata(meta.geo)
    assert affine is not None
    assert affine.a == pytest.approx(PIXEL_SIZE)
    assert affine.c == pytest.approx(ORIGIN_X)
    assert affine.f == pytest.approx(ORIGIN_Y)


def test_affine_from_metadata_none_when_absent() -> None:
    from core.schemas import GeoMetadata

    assert affine_from_metadata(GeoMetadata()) is None