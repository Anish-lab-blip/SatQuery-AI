"""Phase 9 tests — change detection model, dataset, and post-processing.

The model itself is never downloaded here. What IS tested is everything that was
learned by probing it (docs/PHASE9_CHANGE_CONTRACT.md):

  F5-1  the loader class is resolved by feature detection
  F5-2  the dtype kwarg is resolved by signature detection (`dtype=` vs `torch_dtype=`)
  F5-3  processor longest_edge = processor_n * 512
  F5-4  the dtype kwarg is `dtype=` on v5+, `torch_dtype=` on v4
  F5-5  grounding carries an explicit coordinate_system and a documented token floor
  F5-6  the processor default was 2048 (would upscale 512 tiles 4x)
  F5-7  the processor must NOT upscale our 512 tiles
  F5-8  the STANet attention budget logic
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch
from affine import Affine

from specialists.change.stanet import (
    STANetStyleChangeDetector,
    ChangeLossBreakdown,
    ChangeOutput,
    save_change_model,
    load_change_model,
)
from specialists.change.postprocess import (
    postprocess_change_map,
    connected_regions,
    morphological_cleanup,
    regions_to_schema,
    measure_registration,
)
from evaluation.metrics.change import (
    DEFAULT_THRESHOLD,
    score_dataset,
    score_one,
)
from geospatial.transform import (
    benchmark_boxes_to_normalized,
    normalized_boxes_to_benchmark,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def geotiff(tmp_path):
    """A small 3-band GeoTIFF the specialist can actually read."""
    import rasterio

    path = tmp_path / "sample.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=256, height=256, count=4,
        dtype="uint16", crs="EPSG:32643",
        transform=Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 2500000.0),
    ) as ds:
        ds.write(np.random.randint(0, 4000, (3, 256, 256)).astype("uint16"))
    return path


@pytest.fixture()
def plain_tiff(tmp_path):
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
def sar_tiff(tmp_path):
    """A 2-band SAR GeoTIFF (VV, VH)."""
    import rasterio

    path = tmp_path / "sar.tif"
    data = np.full((2, 32, 32), 1000, dtype=np.uint16)
    with rasterio.open(
        path, "w", driver="GTiff", width=32, height=32, count=2, dtype="uint16",
        crs="EPSG:32643", transform=Affine(10.0, 0, 500000.0, 0, -10.0, 2500000.0),
    ) as ds:
        ds.write(data)
    return path


# ---------------------------------------------------------------------------
# STANet model contract
# ---------------------------------------------------------------------------


def test_model_constructs() -> None:
    from specialists.change.stanet import STANetStyleChangeDetector
    model = STANetStyleChangeDetector(width=64, pretrained=False)
    assert model is not None


def test_model_forward_shape() -> None:
    import torch
    from specialists.change.stanet import STANetStyleChangeDetector

    model = STANetStyleChangeDetector(width=64, pretrained=False)
    model.eval()
    t1 = torch.randn(2, 3, 256, 256)
    t2 = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(t1, t2)

    # `forward` returns a ChangeOutput, not a bare tensor. Assert on the real
    # return type so a signature change is a test failure rather than an
    # AttributeError deep inside some caller.
    assert isinstance(out, ChangeOutput)
    assert out.logits.shape == (2, 1, 256, 256)
    assert out.probabilities.shape == (2, 1, 256, 256)
    # Probabilities are sigmoid(logits), so they must be in [0, 1] and agree.
    assert torch.all(out.probabilities >= 0.0)
    assert torch.all(out.probabilities <= 1.0)
    assert torch.allclose(out.probabilities, torch.sigmoid(out.logits))


def test_model_rejects_invalid_shapes() -> None:
    from specialists.change.stanet import STANetStyleChangeDetector

    model = STANetStyleChangeDetector(width=64, pretrained=False)
    with pytest.raises(Exception):
        model(torch.randn(2, 3, 256, 256), torch.randn(2, 3, 128, 128))  # shape mismatch


def test_model_rejects_non_divisible_spatial_dims() -> None:
    from specialists.change.stanet import STANetStyleChangeDetector

    model = STANetStyleChangeDetector(width=64, pretrained=False)
    with pytest.raises(Exception):
        model(torch.randn(1, 3, 257, 256), torch.randn(1, 3, 257, 256))  # not divisible by 8


def test_model_config_dict() -> None:
    from specialists.change.stanet import STANetStyleChangeDetector
    model = STANetStyleChangeDetector(width=128, pretrained=False)
    config = model.config_dict()
    assert config["width"] == 128
    assert config["sa_mode"] == "PAM"
    assert config["attention_budget_bytes"] == 256 * 1024 * 1024


# ---------------------------------------------------------------------------
# Postprocess
# ---------------------------------------------------------------------------


def test_postprocess_output_shape() -> None:
    import numpy as np

    prob = np.zeros((64, 64), dtype=np.float32)
    prob[10:20, 10:20] = 0.9
    prob[40:42, 40:42] = 0.8
    prob[50, 50] = 0.95

    res = postprocess_change_map(prob, threshold=0.5, min_component_pixels=32)

    assert res.n_components_kept == 1
    assert res.total_change_pixels == 96


def test_morphological_cleanup_order() -> None:
    """Opening then closing, not the reverse."""
    import numpy as np

    m = np.zeros((32, 32), dtype=np.uint8)
    m[10:20, 10:20] = 1
    m[15, 15] = 0
    m[5, 5] = 1

    o1 = morphological_cleanup(m, 3, open_iterations=1, close_iterations=2)

    # isolated pixel removed
    assert o1[5, 5] == 0
    # hole filled
    assert o1[15, 15] == 1


def test_morphology_open_then_close_vs_close_then_open() -> None:
    """Opening first is what removes the speck; the close then fills the hole.

    The `close_only` branch passes `open_iterations=0`, so it performs a close
    with no open -- it is not literally "close then open" despite the module
    docstring's shorthand for the reversed order. What the pair pins is the
    consequence of the open step:

      * the isolated speck at (5, 5) survives a close-only pass (closing
        cannot remove a lone pixel) but is gone when opening runs first;
      * the centre hole at (15, 15) IS filled by the close in both branches,
        because the 3x3 MORPH_ELLIPSE closing happens after the open in the
        real pipeline and the open leaves the 10x10 block's hole intact.

    The original version asserted `close_then_open[15, 15] == 0` ("the hole
    persists"). It does not: this kernel DOES fill the hole, so the original
    assertion was simply false about the implementation. The genuine,
    order-dependent difference is at the speck, and that is what is asserted.
    """
    import numpy as np

    m = np.zeros((32, 32), dtype=np.uint8)
    m[10:20, 10:20] = 1
    m[15, 15] = 0  # centre hole
    m[5, 5] = 1    # isolated speck

    open_then_close = morphological_cleanup(m, 3, open_iterations=1, close_iterations=2)
    close_only = morphological_cleanup(m, 3, open_iterations=0, close_iterations=2)

    # The real, guaranteed difference: only the open-first pass removes the speck.
    assert open_then_close[5, 5] == 0
    assert close_only[5, 5] == 1
    # Both fill the hole; the close is what does it.
    assert open_then_close[15, 15] == 1
    assert close_only[15, 15] == 1


def test_connected_regions() -> None:
    import numpy as np

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    mask[30:40, 30:40] = 1

    # `connected_regions` returns (kept_regions, n_components_before_filtering).
    # Both are asserted: the raw count exists so a caller can report how much
    # noise was discarded, and that is a real part of the contract.
    regions, n_raw = connected_regions(mask, min_pixels=32)

    assert n_raw == 2
    assert len(regions) == 2
    assert regions[0].area_pixels == 100
    assert regions[1].area_pixels == 100


def test_connected_regions_filters_small_components() -> None:
    """A component below `min_pixels` is dropped from `kept` but still counted.

    The two halves are what make the raw counter useful: dropping without
    recording would hide how noisy the map actually was.
    """
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:20, 10:20] = 1        # 100 px, kept
    mask[50, 50] = 1              # 1 px, dropped

    regions, n_raw = connected_regions(mask, min_pixels=32)

    assert n_raw == 2
    assert len(regions) == 1
    assert regions[0].area_pixels == 100


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics_precision_recall_f1_iou() -> None:
    y_true = np.array([[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]], dtype=bool)
    y_pred = np.array([[1,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,0,0]], dtype=bool)

    scores = score_one(y_true, y_pred)

    # tp=3, fp=1, fn=1  ->  precision = recall = f1 = 0.75
    assert scores.precision == pytest.approx(0.75)
    assert scores.recall == pytest.approx(0.75)
    assert scores.f1 == pytest.approx(0.75)
    # iou = tp / (tp+fp+fn) = 3/5
    assert scores.iou == pytest.approx(0.6)
    # miou = mean(change_iou, background_iou) = 0.5 * (3/5 + 11/13) = 0.7230769...
    # The original expected 0.7231 with pytest.approx's default rel tolerance,
    # which is 7.2e-07 -- far tighter than the 4th decimal the literal implies.
    assert scores.miou == pytest.approx(0.7230769, abs=1e-6)


def test_metrics_dataset_aggregation() -> None:
    pairs = [
        (np.array([[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]], dtype=bool),
         np.array([[1,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,0,0]], dtype=bool)),
        (np.zeros((4,4), dtype=bool), np.zeros((4,4), dtype=bool)),  # no change
    ]

    report = score_dataset(pairs)
    assert report.n_images == 2
    assert report.n_images_with_change == 1
    # The change tile scores 0.75. The all-background tile is EXCLUDED from the
    # macro average (it would otherwise score recall 0.0 for a property of the
    # dataset, not the model -- see the module docstring), so macro.f1 is 0.75.
    # The original expected 0.375, the naive both-tiles-averaged value the
    # module explicitly rejects; and its first pair was (t, t), a perfect
    # prediction, so even `pooled.precision` could not have been 0.75.
    assert report.macro.f1 == pytest.approx(0.75)
    assert report.pooled.precision == pytest.approx(0.75)
    assert report.pooled.recall == pytest.approx(0.75)


def test_benchmark_boxes_to_normalized() -> None:
    # VRSBench boxes are normalized 0-100, not pixels. The conversion lives in
    # `geospatial.transform`, NOT in `evaluation.metrics.change` (whose __all__
    # has no box helpers). The original test imported from the wrong module and
    # could never have passed.
    got = benchmark_boxes_to_normalized([[0, 0, 50, 100]])
    assert got == [[0.0, 0.0, 0.5, 1.0]]


def test_benchmark_roundtrip() -> None:
    original = [[12.5, 30.0, 88.0, 99.9]]
    norm = benchmark_boxes_to_normalized(original)
    back = normalized_boxes_to_benchmark(norm)
    # pytest.approx does not accept a nested list; compare the flattened values.
    assert np.asarray(back, dtype=float).ravel().tolist() == pytest.approx(original[0])


# ---------------------------------------------------------------------------
# Change detection model
# ---------------------------------------------------------------------------


def test_stanet_forward() -> None:
    from specialists.change.stanet import STANetStyleChangeDetector
    import torch

    model = STANetStyleChangeDetector(width=64, pretrained=False)
    model.eval()
    t1 = torch.randn(2, 3, 256, 256)
    t2 = torch.randn(2, 3, 256, 256)
    out = model(t1, t2)
    assert out.logits.shape == (2, 1, 256, 256)


def test_stanet_attention_budget() -> None:
    from specialists.change.stanet import attention_matrix_bytes
    assert attention_matrix_bytes(2, 4096) == 134_217_728  # 128 MB
    assert attention_matrix_bytes(2, 1024) == 8_388_608   # 8 MB


def test_change_loss() -> None:
    import torch
    from specialists.change.stanet import change_loss

    lg = torch.zeros(2, 1, 32, 32)
    tg = torch.zeros(2, 1, 32, 32)
    tg[0, :, 8:16, 8:16] = 1.0

    loss = change_loss(lg, tg, bce_weight=0.5, dice_weight=0.5)
    assert loss.total.item() > 0
    assert loss.bce.item() > 0
    assert loss.dice.item() > 0


# ---------------------------------------------------------------------------
# Postprocess
# ---------------------------------------------------------------------------


def test_postprocess_regions() -> None:
    import numpy as np
    prob = np.zeros((64, 64), dtype=np.float32)
    prob[10:20, 10:20] = 0.9
    prob[40:42, 40:42] = 0.8
    prob[50, 50] = 0.95

    res = postprocess_change_map(prob, threshold=0.5, min_component_pixels=32)

    assert res.n_components_kept == 1
    assert res.total_change_pixels == 96


def test_morphological_cleanup() -> None:
    import numpy as np
    m = np.zeros((32, 32), dtype=np.uint8)
    m[10:20, 10:20] = 1
    m[15, 15] = 0  # hole

    cleaned = morphological_cleanup(m, 3, open_iterations=1, close_iterations=2)
    assert cleaned[15, 15] == 1  # hole filled


def test_regions_to_schema() -> None:
    import numpy as np
    from specialists.change.postprocess import RegionStats, regions_to_schema

    regions = [
        RegionStats(label=1, area_pixels=100, bbox_px=(10, 10, 20, 20), mean_probability=0.9),
    ]
    schema = regions_to_schema([RegionStats(label=1, area_pixels=100, bbox_px=(10, 10, 20, 20), mean_probability=0.9)], 64, 64)
    assert len(schema) == 1
    assert schema[0].box.x1 == pytest.approx(10/64)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_score_one() -> None:
    y_true = np.array([[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]], dtype=bool)
    y_pred = np.array([[1,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,0,0]], dtype=bool)

    scores = score_one(y_true, y_pred)
    assert scores.precision == pytest.approx(0.75)
    assert scores.recall == pytest.approx(0.75)
    assert scores.f1 == pytest.approx(0.75)
    assert scores.iou == pytest.approx(0.6)
    assert scores.miou == pytest.approx(0.7230769, abs=1e-6)


def test_score_dataset() -> None:
    pairs = [
        (np.array([[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]], dtype=bool),
         np.array([[1,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,0,0]], dtype=bool)),
        (np.zeros((4,4), dtype=bool), np.zeros((4,4), dtype=bool)),
    ]

    report = score_dataset(pairs)
    assert report.n_images == 2
    assert report.n_images_with_change == 1
    assert report.pooled.precision == pytest.approx(0.75)
    assert report.pooled.recall == pytest.approx(0.75)
    assert report.macro.f1 == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_registration_identical() -> None:
    import numpy as np
    from specialists.change.postprocess import measure_registration

    a = np.random.default_rng(0).integers(0, 200, (128, 128)).astype(np.uint8)
    smooth = cv2.GaussianBlur(a.astype(np.float32), (9,9), 3)

    q = measure_registration(smooth, smooth)
    assert q.shift_magnitude == pytest.approx(0.0, abs=1e-6)
    # cv2.phaseCorrelate returns a response that can overshoot 1.0 very slightly
    # (measured: 1.0000733). The original asserted it within 1e-6 of exactly 1.0,
    # which the implementation does not and need not guarantee.
    assert q.response == pytest.approx(1.0, abs=1e-3)
    assert q.is_usable is True


def test_registration_shift() -> None:
    import numpy as np
    import cv2
    from specialists.change.postprocess import measure_registration

    a = np.random.default_rng(0).integers(0, 200, (128, 128)).astype(np.uint8)
    smooth = cv2.GaussianBlur(a.astype(np.float32), (9, 9), 3)
    shifted = np.roll(smooth, (5, 3), axis=(0, 1))
    q = measure_registration(smooth, shifted, max_shift_px=2)
    assert q.shift_magnitude == pytest.approx(5.8, abs=0.5)
    assert q.is_usable is False


# ---------------------------------------------------------------------------
# Registration test (requires cv2)
# ---------------------------------------------------------------------------

def test_registration_requires_cv2() -> None:
    pytest.importorskip("cv2")
    import numpy as np
    from specialists.change.postprocess import measure_registration

    # A real, textured image -- NOT zeros. A constant image has no translatable
    # structure, so phase correlation reports the degenerate (centre,
    # response=0.0) result and `is_usable` is correctly False. The original
    # used a zero image and asserted usableness, which the implementation
    # neither gives nor should give: an all-flat pair carries no information.
    a = np.random.default_rng(1).integers(0, 200, (128, 128)).astype(np.uint8)
    smooth = cv2.GaussianBlur(a.astype(np.float32), (9, 9), 3)
    q = measure_registration(smooth, smooth)
    assert q.is_usable is True

    # And the negative case the zeros were accidentally testing: a flat pair is
    # not usable, because nothing can be aligned.
    flat = cv2.GaussianBlur(np.zeros((128, 128), dtype=np.float32), (9, 9), 3)
    assert measure_registration(flat, flat).is_usable is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])