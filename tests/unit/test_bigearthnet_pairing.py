"""Tests for the BigEarthNet optical<->SAR pairing step (Phase 12).

Everything runs against synthetic patch directories in a temp dir with an
INJECTED band loader, so there is no download, no network and no geospatial
reader in the loop. The loader returns a per-band constant, which is what lets
the tests check WHERE a band landed, not merely that the tensor has the right
shape.

The tests that matter most are the two structural ones:

  * the canonical optical order is CROMA's 12-band order (B10 absent), and a
    band with no file is ZERO-FILLED rather than the neighbours being shifted
    into its slot -- a shift produces a right-shaped tensor that means the wrong
    thing
  * a missing modality is masked and zero-filled, never dropped and never a
    crash
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from specialists.optical_sar.sensor_adapter import OPTICAL_CANONICAL, SAR_CANONICAL
from training.data.bigearthnet import (
    CANONICAL_OPTICAL_BANDS,
    CANONICAL_SAR_BANDS,
    BigEarthNetError,
    discover_patches,
    pair_patch,
    pair_patches,
)

#: What real BigEarthNet-S2 ships: 12 bands, no cirrus (B10).
REAL_S2_BANDS: tuple[str, ...] = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09",
    "B11", "B12",
)

#: The loader's S2 token list is 13 (it includes B10). Truncating it to 12 --
#: which is what a naive "take the first 12 tokens" does -- yields a set that
#: includes B10 and drops B12. That is the trap this file pins.
LOADER_S2_TOKENS: tuple[str, ...] = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09",
    "B10", "B11", "B12",
)

#: A distinct constant per band token, so a placed band is identifiable by value.
BAND_VALUE: dict[str, float] = {
    band: float(i + 1) for i, band in enumerate(REAL_S2_BANDS)
}
BAND_VALUE["B10"] = 99.0  # non-canonical; must never appear in a slot
BAND_VALUE["VV"] = 21.0
BAND_VALUE["VH"] = 22.0

SHAPE = (4, 4)


def fake_loader(path: Path) -> np.ndarray:
    """Return a constant array identifying the band token in the file name."""
    token = path.stem.split("_")[-1]
    return np.full(SHAPE, BAND_VALUE[token], dtype=np.float32)


def write_patch(
    tile_dir: Path,
    patch_name: str,
    tokens: tuple[str, ...],
    *,
    labels: tuple[str, ...] = ("Arable land",),
    split: str | None = None,
) -> Path:
    """Write one BigEarthNet-shaped patch directory (stub band files + metadata)."""
    patch_dir = tile_dir / patch_name
    patch_dir.mkdir(parents=True, exist_ok=True)
    for token in tokens:
        (patch_dir / f"{patch_name}_{token}.tif").write_bytes(b"II*\x00stub")
    meta: dict[str, object] = {"labels": list(labels), "patch_id": patch_name}
    if split:
        meta["split"] = split
    (patch_dir / f"{patch_name}_metadata.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )
    return patch_dir


@pytest.fixture()
def paired_root(tmp_path: Path) -> Path:
    """One S2 tile (p0, p1) and one S1 tile (p0 only)."""
    root = tmp_path / "BEN"
    s2 = root / "S2A_TILE_1"
    s1 = root / "S1A_TILE_1"
    s2.mkdir(parents=True)
    s1.mkdir(parents=True)
    write_patch(s2, "p0", REAL_S2_BANDS)
    write_patch(s2, "p1", REAL_S2_BANDS)
    write_patch(s1, "p0", ("VV", "VH"))
    return root


# ---------------------------------------------------------------------------
# The canonical order is imported, not re-declared
# ---------------------------------------------------------------------------


def test_canonical_orders_are_the_frozen_ones() -> None:
    assert CANONICAL_OPTICAL_BANDS == OPTICAL_CANONICAL
    assert CANONICAL_SAR_BANDS == SAR_CANONICAL
    assert len(CANONICAL_OPTICAL_BANDS) == 12
    assert "B10" not in CANONICAL_OPTICAL_BANDS, (
        "the cirrus band must not be a canonical optical slot"
    )
    assert CANONICAL_SAR_BANDS == ("VV", "VH")


# ---------------------------------------------------------------------------
# Pairing returns BOTH modalities
# ---------------------------------------------------------------------------


def test_pairing_returns_both_modalities(paired_root: Path) -> None:
    sample = pair_patch(paired_root, "p0", loader=fake_loader)
    assert sample.optical.shape == (12, *SHAPE)
    assert sample.sar.shape == (2, *SHAPE)
    assert sample.optical_mask.tolist() == [1.0] * 12
    assert sample.sar_mask.tolist() == [1.0, 1.0]
    assert sample.optical_present and sample.sar_present


def test_optical_bands_land_in_canonical_slots(paired_root: Path) -> None:
    sample = pair_patch(paired_root, "p0", loader=fake_loader)
    for slot, band in enumerate(CANONICAL_OPTICAL_BANDS):
        assert np.all(sample.optical[slot] == BAND_VALUE[band]), (
            f"{band} did not land in slot {slot}"
        )


def test_sar_bands_land_in_canonical_slots(paired_root: Path) -> None:
    sample = pair_patch(paired_root, "p0", loader=fake_loader)
    assert np.all(sample.sar[0] == BAND_VALUE["VV"])
    assert np.all(sample.sar[1] == BAND_VALUE["VH"])


# ---------------------------------------------------------------------------
# The no-shift rule -- the trap
# ---------------------------------------------------------------------------


def test_absent_b10_does_not_shift_bands(tmp_path: Path) -> None:
    """The loader's 13-token list truncated to 12 includes B10 and drops B12.

    A positional mapping would slide B11 into B12's slot and put B10 where B11
    belongs. The correct mapping is by identity: B12 (no file) is zero-filled and
    masked, B11 stays in its own slot, and B10 -- not canonical -- is recorded as
    ignored.
    """
    root = tmp_path / "BEN"
    s2 = root / "S2A_TILE_1"
    s2.mkdir(parents=True)
    tokens = LOADER_S2_TOKENS[:12]
    assert "B10" in tokens and "B12" not in tokens  # the trap is real
    write_patch(s2, "p0", tokens)

    sample = pair_patch(root, "p0", loader=fake_loader)

    assert sample.optical.shape[0] == 12
    b12 = CANONICAL_OPTICAL_BANDS.index("B12")
    assert np.all(sample.optical[b12] == 0.0), "B12's slot must be zero-filled"
    assert sample.optical_mask[b12] == 0.0
    assert "B12" in sample.optical_missing_bands

    b11 = CANONICAL_OPTICAL_BANDS.index("B11")
    assert np.all(sample.optical[b11] == BAND_VALUE["B11"]), (
        "B11 was shifted out of its slot"
    )

    assert "B10" in sample.optical_ignored_bands
    assert not any(
        np.all(sample.optical[i] == BAND_VALUE["B10"]) for i in range(12)
    ), "B10 (non-canonical) occupied a slot"


def test_a_missing_canonical_band_zero_fills_its_slot(tmp_path: Path) -> None:
    """Any absent canonical band is zero-filled, never skipped over."""
    root = tmp_path / "BEN"
    s2 = root / "S2A_TILE_1"
    s2.mkdir(parents=True)
    tokens = tuple(b for b in REAL_S2_BANDS if b != "B09")
    write_patch(s2, "p0", tokens)

    sample = pair_patch(root, "p0", loader=fake_loader)

    b09 = CANONICAL_OPTICAL_BANDS.index("B09")
    assert np.all(sample.optical[b09] == 0.0)
    assert sample.optical_mask[b09] == 0.0
    assert "B09" in sample.optical_missing_bands

    b11 = CANONICAL_OPTICAL_BANDS.index("B11")
    assert np.all(sample.optical[b11] == BAND_VALUE["B11"]), (
        "B11 slid into B09's empty slot"
    )
    assert sample.optical_mask[b11] == 1.0


# ---------------------------------------------------------------------------
# A missing modality is masked, not dropped
# ---------------------------------------------------------------------------


def test_a_missing_modality_is_masked_not_dropped(paired_root: Path) -> None:
    """p1 has no SAR. That is the intended absent-modality case, not an error."""
    sample = pair_patch(paired_root, "p1", loader=fake_loader)
    assert sample.optical_present is True
    assert sample.sar_present is False
    assert sample.sar_mask.tolist() == [0.0, 0.0]
    assert np.all(sample.sar == 0.0)
    assert sample.sar_missing_bands == CANONICAL_SAR_BANDS
    assert np.all(sample.optical[0] == BAND_VALUE["B01"])  # optical intact


# ---------------------------------------------------------------------------
# Existing discovery behaviour is untouched
# ---------------------------------------------------------------------------


def test_existing_discovery_is_unchanged(paired_root: Path) -> None:
    s2 = discover_patches(paired_root, "s2")
    s1 = discover_patches(paired_root, "s1")
    assert {p.patch_id for p in s2} == {"p0", "p1"}
    assert {p.patch_id for p in s1} == {"p0"}
    assert all(p.modality == "s2" and p.band_count == 12 for p in s2)
    assert all(p.modality == "s1" and p.band_count == 2 for p in s1)


# ---------------------------------------------------------------------------
# Pairing API
# ---------------------------------------------------------------------------


def test_pair_patches_returns_one_sample_per_id(paired_root: Path) -> None:
    samples = pair_patches(paired_root, loader=fake_loader)
    assert [s.patch_id for s in samples] == ["p0", "p1"]


def test_pair_patches_respects_a_limit(paired_root: Path) -> None:
    assert len(pair_patches(paired_root, limit=1, loader=fake_loader)) == 1


def test_pairing_across_separate_roots(tmp_path: Path) -> None:
    optical = tmp_path / "S2"
    sar = tmp_path / "S1"
    (optical / "T").mkdir(parents=True)
    (sar / "T").mkdir(parents=True)
    write_patch(optical / "T", "p0", REAL_S2_BANDS)
    write_patch(sar / "T", "p0", ("VV", "VH"))

    sample = pair_patch(optical, "p0", sar_root=sar, loader=fake_loader)
    assert sample.optical_present and sample.sar_present
    assert np.all(sample.sar[0] == BAND_VALUE["VV"])


def test_ambiguous_patch_id_raises(tmp_path: Path) -> None:
    root = tmp_path / "BEN"
    for tile in ("TILE_A", "TILE_B"):
        d = root / tile
        d.mkdir(parents=True)
        write_patch(d, "p0", REAL_S2_BANDS)
    with pytest.raises(BigEarthNetError, match="ambiguous"):
        pair_patches(root, loader=fake_loader)


def test_unknown_patch_id_raises(paired_root: Path) -> None:
    with pytest.raises(BigEarthNetError, match="not found"):
        pair_patch(paired_root, "nope", loader=fake_loader)


def test_sample_to_dict_reports_channel_counts(paired_root: Path) -> None:
    report = pair_patch(paired_root, "p0", loader=fake_loader).to_dict()
    assert report["optical_channels_present"] == 12
    assert report["sar_channels_present"] == 2
    assert report["optical_ignored_bands"] == []
