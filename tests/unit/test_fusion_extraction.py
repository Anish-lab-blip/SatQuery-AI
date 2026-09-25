"""Tests for the Phase 12 frozen-feature extraction pipeline (fixtures only).

No real data, no CROMA, no network, no GPU. The encoder and the band loader are
both injected fakes, which is the whole reason the producer takes them as
parameters.

Two kinds of test live here, and the second is the one that matters:

  * ordinary tests — the pipeline runs, the cache round-trips, the frozen width
    and mask offsets hold, resumability works, the verifier catches corruption.
  * MUTATION PROOFS — for each load-bearing guard, a defective implementation is
    patched in with `monkeypatch`, and the test asserts that the guard FIRES.
    A test that passes under both a correct and a defective implementation
    proves nothing; these prove the checks are load-bearing. Nothing is edited
    on disk and `monkeypatch` restores every patch at teardown.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import training.data.bigearthnet as ben_module  # noqa: E402
import training.fusion.extract as extract_module  # noqa: E402
import training.fusion.train as train_module  # noqa: E402
from specialists.optical_sar.inference import resize_to_canonical  # noqa: E402
from specialists.optical_sar.radiometry import (  # noqa: E402
    MODALITY_OPTICAL,
    normalise_for_croma,
)
from training.data.bigearthnet import (  # noqa: E402
    CANONICAL_OPTICAL_BANDS,
    CANONICAL_SAR_BANDS,
    PairedSample,
    pair_patches,
)
from training.fusion.extract import (  # noqa: E402
    CACHE_VERSION,
    EXTRACTION_METADATA_VERSION,
    FROZEN_FUSION_DIM,
    AmbiguousLabelError,
    CacheProvenanceError,
    ExtractionError,
    ExtractionInputError,
    assert_scene_disjoint,
    assert_split_homogeneous,
    assert_train_val_disjoint,
    extract_fusion_features,
    load_split_assignments,
    plan_extraction,
    policy_name,
    require_single_label,
    resolve_label,
    select_samples,
    skip_ambiguous,
    verify_feature_cache,
)
from training.fusion.train import (  # noqa: E402
    USE_8_BIT_ENV,
    FusionFeature,
    FusionTrainingError,
    prepare_fusion_batch,
    read_feature_cache,
    train_fusion_head,
    write_feature_cache,
)

#: Column offsets of the frozen concatenation: 3*768 GAP, then the 12+2 masks.
MASK_START = 2304
SAR_MASK_START = 2316
FROZEN_CONFIG_HASH = "78f1e3700da15aa1"
DIM = 768
#: `croma.image_resolution` in the frozen config; the size the provenance
#: checks compare against, so fixtures must use it unless they say otherwise.
CONFIG_RESOLUTION = 120

#: What real BigEarthNet-S2 ships: 12 bands, no cirrus (B10).
REAL_S2_BANDS: tuple[str, ...] = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09",
    "B11", "B12",
)
BAND_VALUE: dict[str, float] = {
    band: float(index + 1) for index, band in enumerate(REAL_S2_BANDS)
}
BAND_VALUE["VV"] = 21.0
BAND_VALUE["VH"] = 22.0
BAND_SHAPE = (8, 8)


@pytest.fixture(autouse=True)
def _isolate_use_8_bit():
    """The arm rides on an env var; no test may inherit another's value."""
    prior = os.environ.pop(USE_8_BIT_ENV, None)
    yield
    if prior is None:
        os.environ.pop(USE_8_BIT_ENV, None)
    else:
        os.environ[USE_8_BIT_ENV] = prior


# ---------------------------------------------------------------------------
# Fakes and factories
# ---------------------------------------------------------------------------


def _gap_from(array: np.ndarray, dim: int = DIM) -> np.ndarray:
    """Deterministic, content-dependent (B, dim) vector. No RNG anywhere."""
    arr = np.asarray(array, dtype=np.float32)
    batch, channels = int(arr.shape[0]), int(arr.shape[1])
    per_channel = arr.reshape(batch, channels, -1).mean(axis=2)
    out = np.empty((batch, dim), dtype=np.float32)
    for k in range(dim):
        out[:, k] = per_channel[:, k % channels] + float(k // channels)
    return out


class FakeEncoding:
    """Stands in for `CROMAEncoding`: `.dim` plus `.as_forward_dict()`."""

    def __init__(self, optical_gap, sar_gap, joint_gap) -> None:
        self.optical_gap = optical_gap
        self.sar_gap = sar_gap
        self.joint_gap = joint_gap

    @property
    def dim(self) -> int:
        return int(self.optical_gap.shape[1])

    def as_forward_dict(self) -> dict[str, np.ndarray]:
        return {
            "optical_GAP": self.optical_gap,
            "SAR_GAP": self.sar_gap,
            "joint_GAP": self.joint_gap,
        }


class FakeEncoder:
    """Stands in for CROMA. Records every batch, so tests can assert what the
    pipeline actually fed it.

    `resolution` is None by default, meaning "this fake does not declare one":
    the pipeline only checks a declared resolution, and most tests deliberately
    run at a size the config does not name. `normalize_input = False` is the raw
    encoder the pipeline demands.
    """

    normalize_input = False

    def __init__(self, dim: int = DIM, resolution: int | None = None) -> None:
        self._dim = int(dim)
        self.resolution = resolution
        self.optical_batches: list[np.ndarray] = []
        self.sar_batches: list[np.ndarray] = []

    @property
    def encoder_dim(self) -> int:
        return self._dim

    @property
    def n_calls(self) -> int:
        return len(self.optical_batches)

    def encode(self, optical, sar) -> FakeEncoding:
        optical = np.asarray(optical, dtype=np.float32)
        sar = np.asarray(sar, dtype=np.float32)
        self.optical_batches.append(optical.copy())
        self.sar_batches.append(sar.copy())
        joint = np.concatenate([optical, sar], axis=1)
        return FakeEncoding(
            _gap_from(optical, self._dim),
            _gap_from(sar, self._dim),
            _gap_from(joint, self._dim),
        )


def make_sample(
    patch_id: str,
    *,
    tile: str = "T01",
    labels: tuple[str, ...] = ("Arable land",),
    split: str | None = "train",
    optical_present: int = 12,
    sar_present: int = 2,
    size: int = 8,
) -> PairedSample:
    """One synthetic `PairedSample` in canonical order.

    Each patch gets a deterministic per-id CURVE SHAPE, not a per-id offset: the
    encoder-input stretch is affine-invariant (mean +/- 2std), so an offset alone
    would normalise away and two "different" patches would produce identical
    features. The exponent survives it, so the fixtures are genuinely distinct.
    """
    level = sum(ord(character) for character in patch_id) % 11
    base = np.linspace(0.0, 1.0, size * size, dtype=np.float64)
    ramp = (base ** (1.0 + level * 0.15)).reshape(size, size).astype(np.float32)
    optical = np.zeros((12, size, size), dtype=np.float32)
    for channel in range(optical_present):
        optical[channel] = float(channel + 1) + ramp
    sar = np.zeros((2, size, size), dtype=np.float32)
    for channel in range(sar_present):
        sar[channel] = float(channel + 1) * 2.0 + ramp

    optical_mask = np.zeros(12, dtype=np.float32)
    optical_mask[:optical_present] = 1.0
    sar_mask = np.zeros(2, dtype=np.float32)
    sar_mask[:sar_present] = 1.0

    return PairedSample(
        patch_id=patch_id,
        tile=tile,
        optical=optical,
        sar=sar,
        optical_mask=optical_mask,
        sar_mask=sar_mask,
        optical_present_bands=CANONICAL_OPTICAL_BANDS[:optical_present],
        sar_present_bands=CANONICAL_SAR_BANDS[:sar_present],
        optical_missing_bands=CANONICAL_OPTICAL_BANDS[optical_present:],
        sar_missing_bands=CANONICAL_SAR_BANDS[sar_present:],
        optical_ignored_bands=(),
        sar_ignored_bands=(),
        labels=tuple(labels),
        official_split=split,
    )


def make_feature(sample_id: str, scene_id: str, *, label_index: int = 0, seed: int = 0):
    rng = np.random.default_rng(seed)
    return FusionFeature(
        sample_id=sample_id,
        scene_id=scene_id,
        optical_gap=rng.standard_normal(DIM).astype(np.float32),
        sar_gap=rng.standard_normal(DIM).astype(np.float32),
        joint_gap=rng.standard_normal(DIM).astype(np.float32),
        optical_mask=np.ones(12, dtype=np.float32),
        sar_mask=np.ones(2, dtype=np.float32),
        label_index=label_index,
    )


def extract(
    samples,
    *,
    cache_path: Path,
    encoder=None,
    resolution: int | None = None,
    split: str | None = "train",
    declared_splits=None,
    **kwargs,
):
    """One extraction with the fixtures' defaults, so tests stay readable.

    The default resolution is the CONFIG's (None -> `croma.image_resolution`,
    120), not a test-local 8: `verify_feature_cache` now compares the recorded
    resolution against the live config, so a cache built at a size the config
    does not name is a cache whose provenance checks fail — correctly.
    """
    return extract_fusion_features(
        samples,
        encoder=encoder if encoder is not None else FakeEncoder(),
        resolution=resolution,
        split=split,
        declared_splits=declared_splits,
        label_policy=kwargs.pop("label_policy", require_single_label),
        cache_path=cache_path,
        **kwargs,
    )


def write_stub_patch(
    tile_dir: Path,
    patch_name: str,
    tokens: tuple[str, ...],
    *,
    labels: tuple[str, ...] = ("Arable land",),
    split: str | None = "train",
) -> Path:
    """A BigEarthNet-shaped patch directory: stub band files + metadata JSON.

    The bytes are never read — an injected loader supplies the arrays — but the
    discovery walk needs the file names to exist.
    """
    patch_dir = tile_dir / patch_name
    patch_dir.mkdir(parents=True, exist_ok=True)
    for token in tokens:
        (patch_dir / f"{patch_name}_{token}.tif").write_bytes(b"II*\x00stub")
    (patch_dir / f"{patch_name}_metadata.json").write_text(
        json.dumps({"labels": list(labels), "split": split}), encoding="utf-8"
    )
    return patch_dir


def fake_band_loader(path: Path) -> np.ndarray:
    """Constant per band token, so a placed band is identifiable by value."""
    token = Path(path).stem.split("_")[-1]
    return np.full(BAND_SHAPE, BAND_VALUE[token], dtype=np.float32)


def ramp_band_loader(path: Path) -> np.ndarray:
    """A non-constant band, so `normalise_for_croma` has a real window to use."""
    token = Path(path).stem.split("_")[-1]
    ramp = np.linspace(0.0, 1.0, BAND_SHAPE[0] * BAND_SHAPE[1], dtype=np.float32)
    return (BAND_VALUE[token] + ramp).reshape(BAND_SHAPE)


# ---------------------------------------------------------------------------
# End to end, against fakes
# ---------------------------------------------------------------------------


def test_extraction_runs_end_to_end_and_round_trips(tmp_path: Path) -> None:
    samples = [make_sample(f"p{i}", tile=f"T{i:02d}") for i in range(3)]
    encoder = FakeEncoder()
    out = tmp_path / "cache.npz"

    result = extract(samples, cache_path=out, encoder=encoder)

    assert result.written is True
    assert result.n_encoded == 3
    assert result.n_total == 3
    assert result.encoder_dim == DIM
    assert result.n_scenes == 3
    assert out.exists() and out.with_suffix(".json").exists()

    loaded, _ = read_feature_cache(out)
    assert [feature.sample_id for feature in loaded] == ["p0", "p1", "p2"]
    assert [feature.scene_id for feature in loaded] == ["ben_T00", "ben_T01", "ben_T02"]
    for original, cached in zip(samples, loaded):
        assert np.array_equal(cached.optical_mask, original.optical_mask)
        assert np.array_equal(cached.sar_mask, original.sar_mask)

    # The pipeline fed the encoder exactly normalise-then-resize output.
    expected, _ = normalise_for_croma(
        samples[0].optical, use_8_bit=True, modality=MODALITY_OPTICAL
    )
    expected = resize_to_canonical(expected[0], CONFIG_RESOLUTION)
    assert np.allclose(encoder.optical_batches[0][0], expected, atol=1e-6)

    # And the cached GAPs are exactly what that encoder returned for that input.
    fed = np.concatenate(encoder.optical_batches, axis=0)
    assert np.allclose(loaded[0].optical_gap, _gap_from(fed)[0], atol=1e-6)

    # Distinct patches give distinct vectors — the fixture is not degenerate.
    assert not np.allclose(loaded[0].optical_gap, loaded[1].optical_gap)


def test_extraction_is_deterministic(tmp_path: Path) -> None:
    samples = [make_sample(f"p{i}") for i in range(2)]
    first = extract(samples, cache_path=tmp_path / "a.npz")
    second = extract(samples, cache_path=tmp_path / "b.npz")

    a, _ = read_feature_cache(first.cache_path)
    b, _ = read_feature_cache(second.cache_path)
    for left, right in zip(a, b):
        assert np.array_equal(left.optical_gap, right.optical_gap)
        assert np.array_equal(left.joint_gap, right.joint_gap)


def test_batching_never_encodes_the_whole_corpus_at_once(tmp_path: Path) -> None:
    samples = [make_sample(f"p{i}") for i in range(5)]
    encoder = FakeEncoder()

    extract(samples, cache_path=tmp_path / "c.npz", encoder=encoder, batch_size=2)

    assert encoder.n_calls == 3
    assert [batch.shape[0] for batch in encoder.optical_batches] == [2, 2, 1]
    assert all(
        batch.shape[1:] == (12, CONFIG_RESOLUTION, CONFIG_RESOLUTION)
        for batch in encoder.optical_batches
    )
    assert all(
        batch.shape[1:] == (2, CONFIG_RESOLUTION, CONFIG_RESOLUTION)
        for batch in encoder.sar_batches
    )


def test_extraction_from_roots_uses_the_injected_band_loader(tmp_path: Path) -> None:
    """The loader seam: stub patch files on disk, arrays from the fake."""
    tile = tmp_path / "S2A_MSIL2A_20200101T000000_T01ABC"
    write_stub_patch(tile, "p0", REAL_S2_BANDS)
    write_stub_patch(tile, "p1", REAL_S2_BANDS)

    out = tmp_path / "cache.npz"
    result = extract_fusion_features(
        encoder=FakeEncoder(),
        optical_root=tmp_path,
        loader=ramp_band_loader,
        split="train",
        label_policy=require_single_label,
        cache_path=out,
    )

    assert result.n_encoded == 2
    features, _ = read_feature_cache(out)
    assert sorted(feature.sample_id for feature in features) == ["p0", "p1"]
    # All 12 canonical bands were placed, so the mask is full.
    assert np.array_equal(features[0].optical_mask, np.ones(12, dtype=np.float32))
    assert verify_feature_cache(out)["all_checks_passed"] is True


def test_a_band_loader_with_explicit_samples_is_refused(tmp_path: Path) -> None:
    """A loader that would be silently ignored is an error, not a no-op."""
    with pytest.raises(ExtractionInputError, match="loader"):
        extract_fusion_features(
            [make_sample("p0")],
            encoder=FakeEncoder(),
            loader=ramp_band_loader,
            resolution=8,
            label_policy=require_single_label,
            cache_path=tmp_path / "c.npz",
        )


def test_an_encoder_that_normalises_is_refused(tmp_path: Path) -> None:
    """The stretch must run exactly once; a second one is a silent shift."""

    class NormalisingEncoder(FakeEncoder):
        normalize_input = True

    with pytest.raises(ExtractionError, match="TWICE"):
        extract(
            [make_sample("p0")],
            cache_path=tmp_path / "c.npz",
            encoder=NormalisingEncoder(),
        )
    assert not (tmp_path / "c.npz").exists()


# ---------------------------------------------------------------------------
# The produced cache
# ---------------------------------------------------------------------------


def test_produced_cache_passes_verify_feature_cache(tmp_path: Path) -> None:
    out = tmp_path / "cache.npz"
    extract([make_sample(f"p{i}") for i in range(3)], cache_path=out)

    report = verify_feature_cache(out)

    assert report["all_checks_passed"] is True
    assert report["verdict"] is True
    assert report["failed_checks"] == []
    assert report["n"] == 3
    assert report["cache_version"] == CACHE_VERSION
    assert {item["check"] for item in report["checks"]} >= {
        # the frozen contract
        "gap_vectors_are_768_d",
        "optical_mask_is_12_d",
        "sar_mask_is_2_d",
        "total_width_is_2318",
        "no_nan_or_inf",
        "label_index_in_range",
        "sample_ids_unique",
        "scene_ids_present",
        "row_counts_agree",
        "cache_version_matches",
        # WHICH run produced these numbers
        "provenance_present",
        "extraction_metadata_version_matches",
        "config_hash_matches",
        "arm_use_8_bit_consistent",
        "croma_image_resolution_matches",
        "encoder_dim_matches",
        "label_policy_recorded",
    }
    assert report["metadata"]["label_policy"] == "require_single_label"


def test_frozen_width_and_mask_offsets(tmp_path: Path) -> None:
    """The (B, 2318) layout: 3x768 GAP, then the optical mask at 2304, SAR at 2316."""
    sample = make_sample("p0", optical_present=4, sar_present=1)
    out = tmp_path / "cache.npz"
    extract([sample], cache_path=out)
    features, _ = read_feature_cache(out)

    gaps = np.stack(
        [np.concatenate([f.optical_gap, f.sar_gap, f.joint_gap]) for f in features]
    )
    o_mask = np.stack([f.optical_mask for f in features])
    s_mask = np.stack([f.sar_mask for f in features])
    assembled = prepare_fusion_batch(gaps, o_mask, s_mask, apply_dropout=False)

    assert assembled.shape == (1, FROZEN_FUSION_DIM) == (1, 2318)
    assert np.array_equal(assembled[0, MASK_START:SAR_MASK_START], o_mask[0])
    assert np.array_equal(assembled[0, SAR_MASK_START:], s_mask[0])
    assert np.array_equal(o_mask[0], np.array([1, 1, 1, 1] + [0] * 8, np.float32))
    assert np.array_equal(s_mask[0], np.array([1, 0], np.float32))
    assert np.array_equal(assembled[0, :768], features[0].optical_gap)
    assert np.array_equal(assembled[0, 768:1536], features[0].sar_gap)
    assert np.array_equal(assembled[0, 1536:2304], features[0].joint_gap)


def test_provenance_metadata_is_written(tmp_path: Path) -> None:
    out = tmp_path / "cache.npz"
    extract([make_sample("p0")], cache_path=out, arm="A")
    _, meta = read_feature_cache(out)

    assert meta["config_hash"] == FROZEN_CONFIG_HASH
    assert meta["arm"] == "A"
    assert meta["use_8_bit"] is True
    assert meta["croma_image_resolution"] == CONFIG_RESOLUTION
    assert meta["encoder_dim"] == DIM
    assert meta["extraction_metadata_version"] == EXTRACTION_METADATA_VERSION
    assert meta["label_policy"] == "require_single_label"
    assert meta["split"] == "train"
    assert meta["cache_version"] == CACHE_VERSION
    assert meta["num_classes"] == 19
    assert meta["deterministic"] is True
    assert meta["normalisation"]["transform"] == "per_channel_mean_pm_2std"
    assert meta["normalisation"]["use_8_bit"] is True
    assert meta["source"]["kind"] == "samples"
    assert meta["n_samples"] == 1
    # Ramp bands have a real window, so nothing is degenerate in this fixture.
    assert meta["n_samples_with_degenerate_channels"] == 0


def test_a_present_but_flat_band_is_counted_as_degenerate(tmp_path: Path) -> None:
    """A constant present band has no window; the aggregate must be live."""
    sample = make_sample("p0")
    flat = np.full((1, 8, 8), 5.0, dtype=np.float32)
    sample = replace(
        sample, optical=np.concatenate([flat, sample.optical[1:]], axis=0)
    )
    out = tmp_path / "cache.npz"
    extract([sample], cache_path=out)

    _, meta = read_feature_cache(out)
    assert meta["n_samples_with_degenerate_channels"] == 1


def test_the_arm_selects_the_stretch_and_never_moves_the_config_hash(
    tmp_path: Path,
) -> None:
    a_out = tmp_path / "a.npz"
    b_out = tmp_path / "b.npz"
    extract([make_sample("p0")], cache_path=a_out, arm="A")
    extract([make_sample("p0")], cache_path=b_out, arm="B")

    _, meta_a = read_feature_cache(a_out)
    _, meta_b = read_feature_cache(b_out)

    assert meta_a["use_8_bit"] is True and meta_a["arm"] == "A"
    assert meta_b["use_8_bit"] is False and meta_b["arm"] == "B"
    assert meta_a["config_hash"] == meta_b["config_hash"] == FROZEN_CONFIG_HASH


def test_resolution_defaults_to_the_config(tmp_path: Path) -> None:
    """Not overriding the resolution must read `croma.image_resolution`."""
    from core.config import load_config

    out = tmp_path / "cache.npz"
    encoder = FakeEncoder()
    extract([make_sample("p0")], cache_path=out, encoder=encoder, resolution=None)

    assert encoder.optical_batches[0].shape[2:] == (120, 120)
    _, meta = read_feature_cache(out)
    assert meta["croma_image_resolution"] == int(
        load_config().get("croma.image_resolution")
    ) == 120


def test_an_arm_that_contradicts_use_8_bit_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ExtractionError, match="contradicts"):
        extract(
            [make_sample("p0")],
            cache_path=tmp_path / "c.npz",
            arm="A",
            use_8_bit=False,
        )


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------


def test_rerun_skips_existing_ids_without_duplicating_or_corrupting(
    tmp_path: Path,
) -> None:
    samples = [make_sample(f"p{i}") for i in range(4)]
    out = tmp_path / "cache.npz"

    extract(samples[:2], cache_path=out)
    before, _ = read_feature_cache(out)

    encoder = FakeEncoder()
    second = extract(samples, cache_path=out, encoder=encoder)

    assert second.n_skipped_existing == 2
    assert second.n_encoded == 2
    assert second.n_total == 4
    # The second pass encoded only the two new samples.
    assert encoder.n_calls == 1
    assert encoder.optical_batches[0].shape[0] == 2

    after, _ = read_feature_cache(out)
    assert [feature.sample_id for feature in after] == ["p0", "p1", "p2", "p3"]
    for old, new in zip(before, after[:2]):
        assert old.sample_id == new.sample_id
        assert np.array_equal(old.optical_gap, new.optical_gap)
        assert np.array_equal(old.optical_mask, new.optical_mask)
    assert verify_feature_cache(out)["all_checks_passed"] is True


def test_a_fully_present_cache_is_left_untouched(tmp_path: Path) -> None:
    samples = [make_sample(f"p{i}") for i in range(2)]
    out = tmp_path / "cache.npz"
    extract(samples, cache_path=out)
    before = out.read_bytes()

    second = extract(samples, cache_path=out)

    assert second.n_skipped_existing == 2
    assert second.n_encoded == 0
    assert second.written is False
    assert out.read_bytes() == before


def test_no_resume_overwrites_from_scratch(tmp_path: Path) -> None:
    out = tmp_path / "cache.npz"
    extract([make_sample("p0"), make_sample("p1")], cache_path=out)
    result = extract([make_sample("p2")], cache_path=out, resume=False)

    assert result.n_skipped_existing == 0
    assert result.n_encoded == 1
    features, _ = read_feature_cache(out)
    assert [feature.sample_id for feature in features] == ["p2"]


def test_a_duplicate_id_in_the_input_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ExtractionError, match="duplicate sample id"):
        extract(
            [make_sample("p0"), make_sample("p0")],
            cache_path=tmp_path / "c.npz",
        )


# ---------------------------------------------------------------------------
# The label policy — the open decision
# ---------------------------------------------------------------------------


def test_multi_label_without_a_policy_raises(tmp_path: Path) -> None:
    """The ambiguity is surfaced, never silently resolved."""
    sample = make_sample("p0", labels=("Arable land", "Pastures"))
    out = tmp_path / "c.npz"

    with pytest.raises(AmbiguousLabelError, match="2 labels"):
        extract_fusion_features(
            [sample],
            encoder=FakeEncoder(),
            resolution=8,
            split="train",
            label_policy=None,
            cache_path=out,
        )
    assert not out.exists()


def test_the_same_ambiguity_raises_through_the_default_policy(
    tmp_path: Path,
) -> None:
    with pytest.raises(AmbiguousLabelError, match="Pastures"):
        extract(
            [make_sample("p0", labels=("Arable land", "Pastures"))],
            cache_path=tmp_path / "c.npz",
        )


def test_a_zero_label_sample_is_skipped_and_counted_not_mapped_to_class_zero(
    tmp_path: Path,
) -> None:
    samples = [
        make_sample("p0"),
        make_sample("p1", labels=()),
        make_sample("p2"),
    ]
    out = tmp_path / "cache.npz"

    result = extract(samples, cache_path=out)

    assert result.n_requested == 3
    assert result.n_skipped_unlabelled == 1
    assert result.n_skipped_by_policy == 1
    assert result.n_encoded == 2

    features, _ = read_feature_cache(out)
    ids = [feature.sample_id for feature in features]
    assert ids == ["p0", "p2"]
    assert "p1" not in ids, "a zero-label sample was mapped to a class instead of skipped"
    # 'Arable land' is class 2 in the CLC-19 order; the mapping is real.
    assert {feature.label_index for feature in features} == {2}


def test_a_zero_label_sample_is_also_skipped_with_no_policy(tmp_path: Path) -> None:
    """`label_policy=None` skips zero-label samples rather than defaulting."""
    out = tmp_path / "cache.npz"
    result = extract(
        [make_sample("p0"), make_sample("p1", labels=())],
        cache_path=out,
        label_policy=None,
    )
    assert result.n_skipped_unlabelled == 1
    features, _ = read_feature_cache(out)
    assert [feature.sample_id for feature in features] == ["p0"]


def test_skip_ambiguous_excludes_rather_than_chooses(tmp_path: Path) -> None:
    samples = [
        make_sample("p0"),
        make_sample("p1", labels=("Arable land", "Pastures")),
    ]
    out = tmp_path / "cache.npz"
    result = extract(samples, cache_path=out, label_policy=skip_ambiguous)

    assert result.n_skipped_by_policy == 1
    assert result.n_skipped_unlabelled == 0
    features, _ = read_feature_cache(out)
    assert [feature.sample_id for feature in features] == ["p0"]
    _, meta = read_feature_cache(out)
    assert meta["label_policy"] == "skip_ambiguous"


def test_a_policy_returning_an_out_of_range_index_is_refused() -> None:
    with pytest.raises(ExtractionError, match="outside"):
        resolve_label(("Arable land",), lambda labels: 19, sample_id="p0")


def test_an_unknown_class_is_an_error_not_a_drop() -> None:
    with pytest.raises(ExtractionError, match="not one of the"):
        resolve_label(("Sown grass",), require_single_label, sample_id="p0")


def test_policy_names_are_recorded_and_distinguishable() -> None:
    assert policy_name(None) == "none"
    assert policy_name(require_single_label) == "require_single_label"
    assert policy_name(skip_ambiguous) == "skip_ambiguous"
    # An anonymous policy gets a fingerprint, not the useless bare type name —
    # see `test_a_lambda_policy_gets_a_stable_distinct_identity`.
    assert policy_name(lambda labels: 0).startswith("function:")


def test_an_empty_extraction_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ExtractionInputError, match="empty cache"):
        extract(
            [make_sample("p0", labels=())],
            cache_path=tmp_path / "c.npz",
        )


# ---------------------------------------------------------------------------
# The split firewall
# ---------------------------------------------------------------------------


def test_split_homogeneity_is_enforced() -> None:
    features = [make_feature("s0", "scene_a"), make_feature("s1", "scene_b")]

    assert_split_homogeneous(features, ["train", "train"], requested="train")

    with pytest.raises(ExtractionInputError, match="do not belong"):
        assert_split_homogeneous(features, ["train", "val"], requested="train")


def test_an_undeclared_split_is_a_failure_not_a_wildcard() -> None:
    features = [make_feature("s0", "scene_a")]
    with pytest.raises(ExtractionInputError, match="do not belong"):
        assert_split_homogeneous(features, [None], requested="train")


def test_extraction_refuses_a_mixed_split(tmp_path: Path) -> None:
    samples = [
        make_sample("p0", split="train"),
        make_sample("p1", split="val"),
    ]
    out = tmp_path / "c.npz"
    with pytest.raises(ExtractionInputError, match="homogeneous|do not belong"):
        extract(samples, cache_path=out)
    assert not out.exists()


def test_train_val_scene_disjointness_is_enforced() -> None:
    train = [make_feature("s0", "scene_a")]
    val = [make_feature("s1", "scene_b")]
    assert_train_val_disjoint(train, val)

    leaked = [make_feature("s1", "scene_a")]
    with pytest.raises(FusionTrainingError, match="scene"):
        assert_train_val_disjoint(train, leaked)


def test_scene_disjointness_works_on_plain_scene_ids() -> None:
    assert_scene_disjoint({"a", "b"}, {"c"})
    with pytest.raises(ExtractionInputError, match="leaks|both"):
        assert_scene_disjoint({"a", "b"}, {"b", "c"})


def test_declared_splits_override_what_the_sample_says(tmp_path: Path) -> None:
    """A metadata table's assignment is the split that is checked."""
    samples = [make_sample("p0", split=None)]
    out = tmp_path / "c.npz"
    result = extract(samples, cache_path=out, declared_splits=["train"])
    assert result.n_encoded == 1

    with pytest.raises(ExtractionInputError):
        extract(samples, cache_path=tmp_path / "d.npz", declared_splits=["val"])


# ---------------------------------------------------------------------------
# Metadata tables
# ---------------------------------------------------------------------------


def test_load_split_assignments_from_csv(tmp_path: Path) -> None:
    table = tmp_path / "metadata.csv"
    table.write_text(
        "patch_id,split\np0,train\np1,validation\np2,test\np3,nonsense\n",
        encoding="utf-8",
    )
    assignments = load_split_assignments(table)

    assert assignments == {"p0": "train", "p1": "val", "p2": "test"}


def test_load_split_assignments_rejects_an_unrecognised_schema(
    tmp_path: Path,
) -> None:
    table = tmp_path / "metadata.csv"
    table.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ExtractionInputError, match="columns"):
        load_split_assignments(table)


def test_load_split_assignments_reports_a_missing_table(tmp_path: Path) -> None:
    with pytest.raises(ExtractionInputError, match="does not exist"):
        load_split_assignments(tmp_path / "absent.parquet")


# ---------------------------------------------------------------------------
# verify_feature_cache — corruption classes, one test each
# ---------------------------------------------------------------------------


def _valid_cache(tmp_path: Path, n: int = 3) -> Path:
    return extract(
        [make_sample(f"p{i}", tile=f"T{i:02d}") for i in range(n)],
        cache_path=tmp_path / "cache.npz",
    ).cache_path


def _rewrite_arrays(path: Path, **overrides) -> None:
    with np.load(str(path)) as data:
        arrays = {key: data[key] for key in data.files}
    arrays.update(overrides)
    np.savez_compressed(str(path), **arrays)


def _rewrite_sidecar(path: Path, **overrides) -> None:
    sidecar = path.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload.update(overrides)
    sidecar.write_text(json.dumps(payload), encoding="utf-8")


#: Sentinel for `_rewrite_metadata`: remove this KEY. It pops a dict entry; it
#: never deletes a file.
_DROP = object()


def _rewrite_metadata(path: Path, **overrides) -> None:
    """Corrupt ONE provenance field, leaving the arrays and the ids alone."""
    sidecar = path.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata") or {})
    for key, value in overrides.items():
        if value is _DROP:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    payload["metadata"] = metadata
    sidecar.write_text(json.dumps(payload), encoding="utf-8")


def test_verify_passes_the_valid_cache_control(tmp_path: Path) -> None:
    assert verify_feature_cache(_valid_cache(tmp_path))["all_checks_passed"] is True


def test_verify_fails_on_a_wrong_mask_width(tmp_path: Path) -> None:
    cache = _valid_cache(tmp_path)
    _rewrite_arrays(cache, optical_mask=np.zeros((3, 11), dtype=np.float32))

    report = verify_feature_cache(cache)

    assert report["all_checks_passed"] is False
    assert "optical_mask_is_12_d" in report["failed_checks"]
    assert "total_width_is_2318" in report["failed_checks"]


def test_verify_fails_on_nan(tmp_path: Path) -> None:
    cache = _valid_cache(tmp_path)
    with np.load(str(cache)) as data:
        joint = data["joint_gap"].copy()
    joint[1, 0] = np.nan
    _rewrite_arrays(cache, joint_gap=joint)

    report = verify_feature_cache(cache)

    assert report["all_checks_passed"] is False
    assert "no_nan_or_inf" in report["failed_checks"]
    assert "joint_gap" in next(
        item["detail"] for item in report["checks"] if item["check"] == "no_nan_or_inf"
    )


def test_verify_fails_on_an_out_of_range_label(tmp_path: Path) -> None:
    cache = _valid_cache(tmp_path)
    _rewrite_arrays(cache, label_index=np.array([0, 1, 19], dtype=np.int64))

    report = verify_feature_cache(cache)

    assert report["all_checks_passed"] is False
    assert "label_index_in_range" in report["failed_checks"]


def test_verify_fails_on_a_duplicate_sample_id(tmp_path: Path) -> None:
    cache = _valid_cache(tmp_path)
    _rewrite_sidecar(cache, sample_ids=["p0", "p0", "p2"])

    report = verify_feature_cache(cache)

    assert report["all_checks_passed"] is False
    assert "sample_ids_unique" in report["failed_checks"]


def test_verify_fails_on_a_missing_scene_id(tmp_path: Path) -> None:
    cache = _valid_cache(tmp_path)
    _rewrite_sidecar(cache, scene_ids=["ben_T00", "", "ben_T02"])

    report = verify_feature_cache(cache)

    assert report["all_checks_passed"] is False
    assert "scene_ids_present" in report["failed_checks"]


def test_verify_fails_on_a_row_count_disagreement(tmp_path: Path) -> None:
    cache = _valid_cache(tmp_path)
    _rewrite_sidecar(cache, sample_ids=["p0", "p1"])

    report = verify_feature_cache(cache)

    assert report["all_checks_passed"] is False
    assert "row_counts_agree" in report["failed_checks"]


def test_verify_fails_on_a_stale_cache_version(tmp_path: Path) -> None:
    cache = _valid_cache(tmp_path)
    _rewrite_sidecar(cache, cache_version="v0")

    report = verify_feature_cache(cache)

    assert report["all_checks_passed"] is False
    assert "cache_version_matches" in report["failed_checks"]


def test_verify_fails_on_a_missing_sidecar(tmp_path: Path) -> None:
    """An interrupted write: complete arrays on disk, no sidecar beside them.

    The fixture is BUILT in that state — the arrays are copied to a fresh path
    and no sidecar is written — rather than deleting the sidecar of a valid
    cache. This suite never removes a file: the previous version of this test
    called `Path.unlink()`, which is exactly the defect the Round 2 audit found
    (it is also what trips the sandbox's safe-delete guard at teardown).
    """
    source = _valid_cache(tmp_path)
    orphan = tmp_path / "orphan.npz"
    with np.load(str(source)) as data:
        arrays = {key: data[key] for key in data.files}
    np.savez_compressed(str(orphan), **arrays)

    assert orphan.exists()
    assert not orphan.with_suffix(".json").exists(), "the fixture must have no sidecar"

    report = verify_feature_cache(orphan)

    assert report["all_checks_passed"] is False
    assert report["failed_checks"] == ["sidecar_exists"]
    assert "interrupted write" in report["checks"][1]["detail"]


def test_verify_reports_a_missing_cache_without_raising(tmp_path: Path) -> None:
    report = verify_feature_cache(tmp_path / "absent.npz")
    assert report["all_checks_passed"] is False
    assert report["failed_checks"] == ["npz_exists"]


# ---------------------------------------------------------------------------
# verify_feature_cache — PROVENANCE: which run produced these numbers
#
# A cache can satisfy every shape check and still hold features from another
# arm, another resolution or another config. A trainer that read it would report
# a number belonging to a different experiment.
# ---------------------------------------------------------------------------


class _StubConfig:
    """A config with just the two fields the provenance checks read."""

    def __init__(self, *, config_hash: str = FROZEN_CONFIG_HASH,
                 resolution: int = CONFIG_RESOLUTION) -> None:
        self.hash = config_hash
        self._resolution = resolution

    def get(self, key: str, default=None):
        if key == "croma.image_resolution":
            return self._resolution
        return default


def test_verify_accepts_the_provenance_it_recorded(tmp_path: Path) -> None:
    report = verify_feature_cache(_valid_cache(tmp_path))
    provenance = [
        item
        for item in report["checks"]
        if item["check"]
        in {
            "provenance_present",
            "extraction_metadata_version_matches",
            "config_hash_matches",
            "arm_use_8_bit_consistent",
            "croma_image_resolution_matches",
            "encoder_dim_matches",
            "label_policy_recorded",
        }
    ]
    assert len(provenance) == 7
    assert all(item["passed"] for item in provenance), report["failed_checks"]


def test_verify_fails_when_the_config_hash_differs(tmp_path: Path) -> None:
    cache = _valid_cache(tmp_path)
    _rewrite_metadata(cache, config_hash="deadbeefdeadbeef")

    report = verify_feature_cache(cache)

    assert report["all_checks_passed"] is False
    assert "config_hash_matches" in report["failed_checks"]
    assert "deadbeefdeadbeef" in next(
        item["detail"] for item in report["checks"]
        if item["check"] == "config_hash_matches"
    )


def test_verify_fails_when_the_resolution_differs_from_the_config(
    tmp_path: Path,
) -> None:
    cache = _valid_cache(tmp_path)
    _rewrite_metadata(cache, croma_image_resolution=512)

    report = verify_feature_cache(cache)

    assert report["all_checks_passed"] is False
    assert "croma_image_resolution_matches" in report["failed_checks"]


def test_verify_fails_when_the_arm_and_use_8_bit_disagree(tmp_path: Path) -> None:
    cache = _valid_cache(tmp_path)
    _rewrite_metadata(cache, arm="A", use_8_bit=False)

    report = verify_feature_cache(cache)

    assert report["all_checks_passed"] is False
    assert "arm_use_8_bit_consistent" in report["failed_checks"]


def test_verify_fails_when_the_encoder_width_is_not_the_frozen_one(
    tmp_path: Path,
) -> None:
    cache = _valid_cache(tmp_path)
    _rewrite_metadata(cache, encoder_dim=512)

    report = verify_feature_cache(cache)

    assert report["all_checks_passed"] is False
    assert "encoder_dim_matches" in report["failed_checks"]


def test_verify_fails_when_the_metadata_version_is_unknown(tmp_path: Path) -> None:
    cache = _valid_cache(tmp_path)
    _rewrite_metadata(cache, extraction_metadata_version=0)

    report = verify_feature_cache(cache)

    assert report["all_checks_passed"] is False
    assert "extraction_metadata_version_matches" in report["failed_checks"]


def test_verify_fails_when_there_is_no_provenance_at_all(tmp_path: Path) -> None:
    cache = _valid_cache(tmp_path)
    _rewrite_sidecar(cache, metadata={})

    report = verify_feature_cache(cache)

    assert report["all_checks_passed"] is False
    assert "provenance_present" in report["failed_checks"]
    # The dependent checks fail too, rather than silently passing on nothing.
    assert "config_hash_matches" in report["failed_checks"]
    assert "encoder_dim_matches" in report["failed_checks"]


def test_verify_compares_against_the_config_it_is_given(tmp_path: Path) -> None:
    """`config=` must be used, not silently replaced by a fresh load."""
    cache = _valid_cache(tmp_path)

    report = verify_feature_cache(cache, config=_StubConfig(config_hash="deadbeef"))

    assert report["all_checks_passed"] is False
    assert "config_hash_matches" in report["failed_checks"]
    # The resolution agrees, so exactly the hash check is the failure.
    assert "croma_image_resolution_matches" not in report["failed_checks"]


def test_a_lambda_policy_gets_a_stable_distinct_identity(tmp_path: Path) -> None:
    """`type(policy).__name__` called every lambda "function" — useless provenance."""
    first = lambda labels: 0  # noqa: E731 - an anonymous policy is the case here
    second = lambda labels: 1  # noqa: E731

    assert policy_name(first) != policy_name(second)
    assert policy_name(first) == policy_name(first), "must be stable across calls"
    assert policy_name(first) != "function"
    assert policy_name(first).startswith("function:")

    class CallablePolicy:
        def __call__(self, labels):
            return 0

    assert policy_name(CallablePolicy()).startswith("CallablePolicy:")

    a_out, b_out = tmp_path / "a.npz", tmp_path / "b.npz"
    extract([make_sample("p0")], cache_path=a_out, label_policy=first)
    extract([make_sample("p0")], cache_path=b_out, label_policy=second)

    _, meta_a = read_feature_cache(a_out)
    _, meta_b = read_feature_cache(b_out)
    assert meta_a["label_policy"] == policy_name(first)
    assert meta_a["label_policy"] != meta_b["label_policy"], (
        "two different policies recorded the same provenance string"
    )


# ---------------------------------------------------------------------------
# Resumability is conditional on the provenance matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("cache_version", "v0"),
        ("extraction_metadata_version", 0),
        ("config_hash", "deadbeefdeadbeef"),
        ("arm", "B"),
        ("use_8_bit", False),
        ("croma_image_resolution", 512),
        ("encoder_dim", 512),
        ("split", "val"),
        # The fixtures build caches under `require_single_label` (see `extract`),
        # so this rewrites the record to the OTHER real policy.
        ("label_policy", "skip_ambiguous"),
    ],
)
def test_resume_refuses_a_provenance_mismatch(
    tmp_path: Path, field: str, value
) -> None:
    """One test per comparable field: the field AND both values are named."""
    out = tmp_path / "cache.npz"
    extract([make_sample("p0")], cache_path=out)
    _rewrite_metadata(out, **{field: value})

    with pytest.raises(CacheProvenanceError) as excinfo:
        extract([make_sample("p0"), make_sample("p1")], cache_path=out)

    message = str(excinfo.value)
    assert field in message, message
    assert repr(value) in message, message
    # Nothing was merged and nothing was re-encoded.
    features, _ = read_feature_cache(out)
    assert [feature.sample_id for feature in features] == ["p0"]


def test_resume_refuses_a_cache_with_no_provenance_metadata(tmp_path: Path) -> None:
    out = tmp_path / "cache.npz"
    extract([make_sample("p0")], cache_path=out)
    _rewrite_sidecar(out, metadata={})

    with pytest.raises(CacheProvenanceError, match="no provenance metadata"):
        extract([make_sample("p1")], cache_path=out)


def test_resume_refuses_when_the_existing_cache_records_no_encoder_width(
    tmp_path: Path,
) -> None:
    """A field that is absent is a mismatch, not a wildcard."""
    out = tmp_path / "cache.npz"
    extract([make_sample("p0")], cache_path=out)
    _rewrite_metadata(out, encoder_dim=_DROP)

    with pytest.raises(CacheProvenanceError, match="records nothing"):
        extract([make_sample("p1")], cache_path=out)


def test_resume_still_merges_when_the_provenance_matches(tmp_path: Path) -> None:
    """The control: the check must not refuse an ordinary re-run."""
    out = tmp_path / "cache.npz"
    extract([make_sample("p0")], cache_path=out)

    second = extract([make_sample("p0"), make_sample("p1")], cache_path=out)

    assert second.n_skipped_existing == 1
    assert second.n_encoded == 1
    features, _ = read_feature_cache(out)
    assert [feature.sample_id for feature in features] == ["p0", "p1"]


def test_resume_refuses_when_the_existing_cache_records_no_label_policy(
    tmp_path: Path,
) -> None:
    """A cache whose record has no `label_policy` is not a wildcard.

    This is the case that was actually observed to be WRONG, not a hypothesis.
    Before the field was compared, `check_resume_provenance` only inspected the
    keys it had been told about, so a record that named no policy at all was
    indistinguishable from a record whose policy matched: the merge proceeded
    and the single surviving provenance record named one selection rule while
    the rows had been selected by two. `split` says which partition the rows
    came from; `label_policy` says which rows of it were kept -- and the open
    data decision in this project turns on exactly that field.
    """
    out = tmp_path / "cache.npz"
    extract([make_sample("p0")], cache_path=out)
    _rewrite_metadata(out, label_policy=_DROP)

    with pytest.raises(CacheProvenanceError, match="records nothing"):
        extract([make_sample("p1")], cache_path=out)

    features, _ = read_feature_cache(out)
    assert [feature.sample_id for feature in features] == ["p0"]


def test_the_two_real_label_policies_record_distinct_names() -> None:
    """Non-vacuity for every check above: the names must actually differ.

    If `skip_ambiguous` and `require_single_label` stringified the same way,
    `test_resume_refuses_a_provenance_mismatch[label_policy-...]` would pass
    while comparing a constant against itself, and the guard would be dead.
    """
    assert policy_name(skip_ambiguous) == "skip_ambiguous"
    assert policy_name(require_single_label) == "require_single_label"
    assert policy_name(None) == "none"
    assert policy_name(skip_ambiguous) != policy_name(require_single_label)


#: The provenance fields a resume compares, pinned as data.
#:
#: This list IS the guard, so the list itself is asserted. Every behavioural
#: test above names ONE field, and all of them would still pass if a DIFFERENT
#: field were dropped from the tuple: the remaining tests still exercise the
#: remaining fields, and nothing else asserts membership. That is precisely how
#: `label_policy` came to be written into every cache's metadata while being
#: compared by nothing -- a green suite over a tuple with a hole in it.
EXPECTED_COMPARED_PROVENANCE_FIELDS = (
    "cache_version",
    "extraction_metadata_version",
    "config_hash",
    "arm",
    "use_8_bit",
    "croma_image_resolution",
    "encoder_dim",
    "split",
    "label_policy",
)


def test_the_compared_provenance_fields_are_pinned_exactly() -> None:
    """Pin the SET, not only the behaviour of the fields currently in it."""
    compared = tuple(key for key, _ in extract_module.RESUME_PROVENANCE_FIELDS)
    assert compared == EXPECTED_COMPARED_PROVENANCE_FIELDS


def test_every_compared_provenance_field_has_a_distinct_human_label() -> None:
    """The labels are printed in the refusal; a blank or reused label would
    leave the message ambiguous about WHICH field disagreed."""
    labels = [label for _, label in extract_module.RESUME_PROVENANCE_FIELDS]
    assert all(label.strip() for label in labels), labels
    assert len(set(labels)) == len(labels), labels


def test_the_writer_records_every_field_the_resume_check_compares(
    tmp_path: Path,
) -> None:
    """Writer and checker must agree in BOTH directions.

    Comparing a field the writer never records would refuse every resume with
    "records nothing"; recording a field the checker never compares is the
    `label_policy` defect itself. One cache, both directions checked.
    """
    out = tmp_path / "cache.npz"
    extract([make_sample("p0")], cache_path=out)
    _features, metadata = read_feature_cache(out)

    compared = [key for key, _ in extract_module.RESUME_PROVENANCE_FIELDS]
    never_written = sorted(key for key in compared if key not in metadata)
    assert not never_written, f"compared but never written: {never_written}"


def test_a_forgotten_required_field_is_a_loud_error_not_a_silent_pass() -> None:
    """The hole that HID this defect, closed.

    `check_resume_provenance` skipped any key absent from `expected`. Omitting a
    field therefore did not fail -- it disabled that field's check, silently,
    and the suite stayed green. Here `expected` is deliberately missing
    `label_policy` and the call must complain instead of allowing the merge.
    """
    expected = {
        "cache_version": CACHE_VERSION,
        "extraction_metadata_version": EXTRACTION_METADATA_VERSION,
        "config_hash": "0" * 16,
        "arm": "A",
        "use_8_bit": False,
        "croma_image_resolution": 120,
        "split": "train",
    }
    assert "label_policy" not in expected

    with pytest.raises(ValueError, match="label_policy"):
        extract_module.check_resume_provenance(
            dict(expected), expected, path="cache.npz"
        )


def test_the_optional_provenance_set_is_pinned_to_encoder_dim_only() -> None:
    """Pinning the exemption list, because the exemption IS the hole.

    Only `encoder_dim` may be omitted (an encoder that declares no width cannot
    state it). Widening this set would silently re-open the defect for whatever
    field was added, so the set is asserted rather than trusted.
    """
    optional = extract_module.RESUME_PROVENANCE_OPTIONAL_FIELDS
    assert optional == frozenset({"encoder_dim"})
    compared = {key for key, _ in extract_module.RESUME_PROVENANCE_FIELDS}
    assert optional <= compared, "an exemption for a field that is not compared"


# ---------------------------------------------------------------------------
# The write is committed by rename, so a crash cannot half-overwrite a cache
# ---------------------------------------------------------------------------


def test_an_interrupted_write_leaves_the_previous_cache_untouched(
    monkeypatch, tmp_path: Path
) -> None:
    """The commit is `os.replace`; interrupt it and the old cache must survive.

    Before Round 2 the `.npz` was written in place and then the sidecar was
    written in place, so an interruption anywhere in the slow compressed write
    left a truncated array beside a stale sidecar.
    """
    out = tmp_path / "cache.npz"
    extract([make_sample("p0"), make_sample("p1")], cache_path=out)
    npz_before = out.read_bytes()
    sidecar_before = out.with_suffix(".json").read_bytes()

    def interrupted(src, dst):
        raise OSError("simulated crash before the commit")

    monkeypatch.setattr(train_module.os, "replace", interrupted)

    with pytest.raises(OSError, match="simulated crash"):
        train_module.write_feature_cache(out, [make_feature("p9", "ben_T99")])

    assert out.read_bytes() == npz_before
    assert out.with_suffix(".json").read_bytes() == sidecar_before
    features, _ = read_feature_cache(out)
    assert [feature.sample_id for feature in features] == ["p0", "p1"]
    assert verify_feature_cache(out)["all_checks_passed"] is True

    # And the write really did go through a temporary name.
    assert list(tmp_path.glob(".cache.tmp-*.npz")), "no temporary file was used"


# ---------------------------------------------------------------------------
# The trainer's leak firewall, on the rows about to be written
# ---------------------------------------------------------------------------


def test_a_sample_id_leak_into_the_other_partition_is_refused(
    tmp_path: Path,
) -> None:
    """A leak the scene-level pre-check cannot see: same patch, different scene."""
    samples = [make_sample("p0", tile="T01"), make_sample("p1", tile="T02")]
    other = [make_feature("p0", "ben_T99")]
    out = tmp_path / "cache.npz"

    with pytest.raises(ExtractionInputError, match="leak") as excinfo:
        extract(
            samples,
            cache_path=out,
            other_partition=other,
            other_partition_name="the val cache",
        )

    assert "the val cache" in str(excinfo.value)
    assert "sample" in str(excinfo.value)
    assert not out.exists(), "a leaky cache must never be written"


def test_a_scene_leak_into_the_other_partition_is_refused(tmp_path: Path) -> None:
    samples = [make_sample("p0", tile="T01")]
    other = [make_feature("v0", "ben_T01")]
    out = tmp_path / "cache.npz"

    with pytest.raises(ExtractionInputError, match="leak"):
        extract(samples, cache_path=out, other_partition=other)

    assert not out.exists()


def test_a_disjoint_other_partition_is_accepted(tmp_path: Path) -> None:
    out = tmp_path / "cache.npz"
    result = extract(
        [make_sample("p0", tile="T01")],
        cache_path=out,
        other_partition=[make_feature("v0", "ben_T88")],
    )

    assert result.written is True
    assert verify_feature_cache(out)["all_checks_passed"] is True


# ---------------------------------------------------------------------------
# The size handed to the encoder is checked, not assumed
# ---------------------------------------------------------------------------


def test_an_encoder_declaring_another_resolution_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "cache.npz"

    with pytest.raises(ExtractionError, match="built for 64px"):
        extract([make_sample("p0")], cache_path=out, encoder=FakeEncoder(resolution=64))

    assert not out.exists()


def test_the_size_handed_to_encode_is_checked_not_trusted(
    monkeypatch, tmp_path: Path
) -> None:
    """A resize that silently does nothing keeps every downstream shape valid."""
    monkeypatch.setattr(
        extract_module,
        "resize_to_canonical",
        lambda array, size: np.asarray(array, dtype=np.float32),
    )
    out = tmp_path / "cache.npz"

    with pytest.raises(ExtractionError, match="handed to encode"):
        extract([make_sample("p0")], cache_path=out)

    assert not out.exists()


# ---------------------------------------------------------------------------
# The mask-update guard in prepare_fusion_batch
# ---------------------------------------------------------------------------


def _mask_block(assembled: np.ndarray) -> np.ndarray:
    return assembled[:, MASK_START:SAR_MASK_START]


def test_the_mask_guard_accepts_a_consistent_dropout() -> None:
    """The real `channel_dropout` satisfies the assertion at every rate."""
    gaps = np.zeros((4, 3 * DIM), dtype=np.float32)
    optical_mask = np.ones((4, 12), dtype=np.float32)
    sar_mask = np.ones((4, 2), dtype=np.float32)

    for rates in ((1.0,), (0.0,), (1.0, 0.8, 0.6, 0.4)):
        assembled = prepare_fusion_batch(
            gaps, optical_mask, sar_mask,
            rng=np.random.default_rng(0),
            optical_rates=rates, sar_rates=(1.0,),
        )
        assert assembled.shape == (4, FROZEN_FUSION_DIM)


def test_the_returned_mask_reaches_the_head_not_the_pre_dropout_one() -> None:
    gaps = np.zeros((4, 3 * DIM), dtype=np.float32)
    ones_optical = np.ones((4, 12), dtype=np.float32)
    ones_sar = np.ones((4, 2), dtype=np.float32)

    assembled = prepare_fusion_batch(
        gaps, ones_optical, ones_sar,
        rng=np.random.default_rng(0),
        optical_rates=(0.0,), sar_rates=(0.0,),
    )
    assert np.all(_mask_block(assembled) == 0.0)
    assert np.all(assembled[:, SAR_MASK_START:] == 0.0)


# ---------------------------------------------------------------------------
# MUTATION PROOFS
#
# Each test patches the function under test with a deliberately defective
# version, asserts that the guard FIRES, and lets `monkeypatch` restore it.
# Nothing on disk is edited and no mutation survives the test.
# ---------------------------------------------------------------------------


def test_mutation_discarded_mask_update_must_fail(monkeypatch) -> None:
    """MUTATION 1 — `channel_dropout` stops updating the mask.

    A version that returns the PRE-dropout mask is exactly the defect the `_`
    discard used to hide. The new guard must catch it, because a head trained
    against a mask that lies trains fine and is quietly wrong.
    """
    real = train_module.channel_dropout
    gaps = np.zeros((4, 3 * DIM), dtype=np.float32)
    ones_optical = np.ones((4, 12), dtype=np.float32)
    ones_sar = np.ones((4, 2), dtype=np.float32)

    def discarded_update(features, mask, *, keep_probabilities, rng):
        dropped, _updated = real(
            features, mask, keep_probabilities=keep_probabilities, rng=rng
        )
        return dropped, np.asarray(mask, dtype=np.float32).copy()

    # NON-VACUITY: the mutant is observably different from the real function at
    # keep=0.0 (real mask all-zero, mutant mask all-one). Without this, the
    # `pytest.raises` below could be passing because the patch did nothing.
    real_block, real_mask = real(
        ones_optical, ones_optical,
        keep_probabilities=(0.0,), rng=np.random.default_rng(0),
    )
    bad_block, bad_mask = discarded_update(
        ones_optical, ones_optical,
        keep_probabilities=(0.0,), rng=np.random.default_rng(0),
    )
    assert np.array_equal(real_block, bad_block)
    assert np.all(real_mask == 0.0) and np.all(bad_mask == 1.0)
    assert not np.array_equal(real_mask, bad_mask), "the mutation is a no-op"

    # Control: the real implementation passes the guard.
    prepare_fusion_batch(
        gaps, ones_optical, ones_sar,
        rng=np.random.default_rng(0), optical_rates=(0.0,), sar_rates=(0.0,),
    )

    monkeypatch.setattr(train_module, "channel_dropout", discarded_update)

    with pytest.raises(FusionTrainingError, match="disagree"):
        prepare_fusion_batch(
            gaps, ones_optical, ones_sar,
            rng=np.random.default_rng(0), optical_rates=(0.0,), sar_rates=(0.0,),
        )


def test_mutation_shifted_band_placement_must_fail(monkeypatch, tmp_path: Path) -> None:
    """MUTATION 2 — a missing band's neighbours slide down into its slot.

    This protects `pair_patches`' canonical placement, which `extract.py`
    CONSUMES: the frozen 2318 vector's 12-d mask block at offset 2304 is only
    meaningful if slot i is band i. A shift yields a tensor of exactly the right
    shape whose channels mean the wrong thing, so no downstream shape check can
    see it — only the property asserted below.

    Non-vacuity: the mutant sample is compared against the correct one and shown
    to differ (slot 3 holds B05's data and mask=1), while the shape is identical.
    """
    tile = tmp_path / "T01ABC"
    # B04 (canonical slot 3) has no file; B05 (slot 4) must stay in slot 4.
    present = tuple(band for band in REAL_S2_BANDS if band != "B04")
    write_stub_patch(tile, "p0", present)

    def no_shift(sample: PairedSample) -> None:
        assert sample.optical_mask[3] == 0.0, "slot 3 must be masked off"
        assert np.all(sample.optical[3] == 0.0), "slot 3 must be zero-filled"
        assert sample.optical_mask[4] == 1.0
        assert np.allclose(sample.optical[4], BAND_VALUE["B05"])

    # Control: the real implementation satisfies the no-shift property.
    correct = ben_module.pair_patch(tmp_path, "p0", loader=fake_band_loader)
    no_shift(correct)

    def shifting_assemble(loaded, canonical, shape):
        array = np.zeros((len(canonical), shape[0], shape[1]), dtype=np.float32)
        mask = np.zeros((len(canonical),), dtype=np.float32)
        for packed, slot in enumerate(sorted(loaded)):
            array[packed] = loaded[slot]
            mask[packed] = 1.0
        return array, mask

    monkeypatch.setattr(ben_module, "_assemble_modality", shifting_assemble)

    mutant = ben_module.pair_patch(tmp_path, "p0", loader=fake_band_loader)
    # Shape-silent but observable, in the data AND in the mask block.
    assert mutant.optical.shape == correct.optical.shape
    assert not np.array_equal(mutant.optical[3], correct.optical[3])
    assert np.allclose(mutant.optical[3], BAND_VALUE["B05"])
    assert not np.array_equal(mutant.optical_mask, correct.optical_mask)

    with pytest.raises(AssertionError, match="masked off"):
        no_shift(mutant)


def test_mutation_dropout_resurrecting_an_absent_band_must_fail(monkeypatch) -> None:
    """MUTATION 3 — dropout ignores the sensor mask and revives absent bands.

    Note this defect is SELF-CONSISTENT (block == mask), so mutation 1's
    structural guard cannot see it. Only the semantic property can, which is why
    both checks exist.
    """
    gaps = np.zeros((4, 3 * DIM), dtype=np.float32)
    optical_mask = np.zeros((4, 12), dtype=np.float32)
    optical_mask[:, :4] = 1.0  # Cartosat-2S: 4 real channels, 8 masked off
    sar_mask = np.ones((4, 2), dtype=np.float32)

    def no_resurrection() -> None:
        assembled = prepare_fusion_batch(
            gaps, optical_mask, sar_mask,
            rng=np.random.default_rng(1), optical_rates=(1.0,), sar_rates=(1.0,),
        )
        assert np.all(assembled[:, MASK_START + 4 : SAR_MASK_START] == 0.0), (
            "dropout resurrected a band the sensor never measured"
        )

    # Control: the real implementation keeps the absent bands absent.
    no_resurrection()

    def resurrecting(features, mask, *, keep_probabilities, rng):
        """`keep = draws & (mask > 0)` with the `& (mask > 0)` removed.

        Because the block IS the mask in this representation, the defect returns
        the same (wrong) tensor twice — it is SELF-CONSISTENT, so mutation 1's
        structural guard cannot see it.
        """
        arr = np.asarray(features, dtype=np.float32)
        probabilities = np.asarray(keep_probabilities, dtype=np.float64)
        keep_fraction = rng.choice(probabilities, size=arr.shape[0])
        draws = (rng.random(arr.shape) < keep_fraction[:, None]).astype(np.float32)
        return draws, draws

    # NON-VACUITY, part 1: the mutant is self-consistent AND wrong.
    first, second = resurrecting(
        optical_mask, optical_mask,
        keep_probabilities=(1.0,), rng=np.random.default_rng(1),
    )
    assert np.array_equal(first, second), "the mutant must be self-consistent"
    assert np.all(first[:, 4:] == 1.0), "the mutant must resurrect absent bands"

    monkeypatch.setattr(train_module, "channel_dropout", resurrecting)

    # NON-VACUITY, part 2: because it is self-consistent, the STRUCTURAL guard
    # in `prepare_fusion_batch` accepts it — only the semantic property catches
    # it. If this call raised, mutation 3 would be proving mutation 1's guard.
    prepare_fusion_batch(
        gaps, optical_mask, sar_mask,
        rng=np.random.default_rng(1), optical_rates=(1.0,), sar_rates=(1.0,),
    )

    with pytest.raises(AssertionError, match="resurrected"):
        no_resurrection()


def test_mutation_skipping_the_split_firewall_must_fail(
    monkeypatch, tmp_path: Path
) -> None:
    """MUTATION 4 — the leak guard is a no-op, so a shared scene trains.

    Observation, both ways: with the guard in place the run is refused and no
    record is written; with `assert_split_disjoint` neutralised the same call
    trains and writes one. The guard is what stopped it, not something else.
    """
    train = [make_feature(f"s{i}", "tile_a", seed=i) for i in range(6)]
    val = [make_feature(f"v{i}", "tile_a", seed=100 + i) for i in range(3)]

    # Control: the firewall refuses the leak, and nothing is written.
    with pytest.raises(FusionTrainingError, match="scene"):
        train_fusion_head(
            train, val, output_dir=tmp_path / "control",
            epochs=1, batch_size=4, seed=1, verbose=False,
        )
    assert not (tmp_path / "control" / "run_record.json").exists()

    monkeypatch.setattr(train_module, "assert_split_disjoint", lambda left, right: None)

    result = train_fusion_head(
        train, val, output_dir=tmp_path / "mutated",
        epochs=1, batch_size=4, seed=1, verbose=False,
    )
    # With the firewall disabled the leak is no longer refused: the guard is
    # what caught it, not something else.
    assert result is not None
    assert (tmp_path / "mutated" / "run_record.json").exists()


def test_mutation_silently_mapping_a_multi_label_sample_must_fail(
    monkeypatch, tmp_path: Path
) -> None:
    """MUTATION 5 — the ambiguity guard is replaced by 'take the first label'.

    Extraction then SUCCEEDS and writes a cache whose label was chosen for it.
    That silent rewrite is exactly what the guard exists to prevent.
    """
    sample = make_sample("p0", labels=("Arable land", "Pastures"))

    # Control: no policy -> the ambiguity is refused and nothing is written.
    with pytest.raises(AmbiguousLabelError):
        extract_fusion_features(
            [sample], encoder=FakeEncoder(), resolution=8, split="train",
            label_policy=None, cache_path=tmp_path / "control.npz",
        )
    assert not (tmp_path / "control.npz").exists()

    monkeypatch.setattr(
        extract_module, "resolve_label", lambda labels, policy, **kwargs: 0
    )

    out = tmp_path / "mutated.npz"
    result = extract_fusion_features(
        [sample], encoder=FakeEncoder(), resolution=8, split="train",
        label_policy=None, cache_path=out,
    )
    features, _ = read_feature_cache(out)
    assert result.n_encoded == 1
    assert features[0].label_index == 0  # silently chosen, never declared


# ---------------------------------------------------------------------------
# The CLI's exit codes, and the leak firewall's wiring into the real path
#
# The heavy seams are faked — the corpus walk (`pair_patches`) and CROMA
# (`build_extraction_encoder`) — so the CONTROL FLOW under test is the real
# one: input validation, the split firewall, the extractor, the verifier.
# ---------------------------------------------------------------------------


class RasterioIOError(OSError):
    """What rasterio raises for an unreadable band file.

    Defined here rather than imported. `RasterioIOError` IS an `OSError`
    subclass, and the CLI catches `OSError`, so this proves the handler covers
    the real exception without making rasterio a test dependency.
    """


def _cli_argv(root: Path, out: Path, *, checkpoint_dir: Path | None = None) -> list[str]:
    """Arguments for a non-dry run. The checkpoint is only `is_file()`-checked."""
    base = root if checkpoint_dir is None else checkpoint_dir
    checkpoint = base / "CROMA_base.pt"
    checkpoint.write_bytes(b"stub: never loaded, only stat-ed")
    return [
        "--optical-root", str(root),
        "--split", "train",
        "--out-cache", str(out),
        "--croma-checkpoint", str(checkpoint),
    ]


def _run_cli(
    monkeypatch,
    argv: list[str],
    *,
    samples=None,
    pair_callable=None,
    encoder=None,
    pair_raises=None,
    extractor_raises=None,
) -> int:
    import scripts.extract_fusion_features as cli

    monkeypatch.setattr(sys, "argv", ["extract_fusion_features.py", *argv])

    if pair_raises is not None:
        def pair_boom(*args, **kwargs):
            raise pair_raises

        monkeypatch.setattr(cli, "pair_patches", pair_boom)
    elif pair_callable is not None:
        monkeypatch.setattr(cli, "pair_patches", pair_callable)
    elif samples is not None:
        monkeypatch.setattr(cli, "pair_patches", lambda *a, **k: list(samples))

    if encoder is not None:
        monkeypatch.setattr(cli, "build_extraction_encoder", lambda *a, **k: encoder)

    if extractor_raises is not None:
        def extractor_boom(*args, **kwargs):
            raise extractor_raises

        monkeypatch.setattr(cli, "extract_fusion_features", extractor_boom)

    return cli.main()


def test_cli_exits_zero_on_a_successful_run(monkeypatch, tmp_path: Path) -> None:
    out = tmp_path / "cache.npz"
    code = _run_cli(
        monkeypatch,
        _cli_argv(tmp_path, out),
        samples=[make_sample("p0", tile="T01"), make_sample("p1", tile="T02")],
        encoder=FakeEncoder(resolution=CONFIG_RESOLUTION),
    )

    assert code == 0
    assert out.exists() and out.with_suffix(".json").exists()
    # The CLI's own verifier (provenance included) passed on what it wrote.
    assert verify_feature_cache(out)["all_checks_passed"] is True


def test_cli_exits_two_when_a_band_file_cannot_be_read(
    monkeypatch, tmp_path: Path
) -> None:
    """A read failure is a corpus problem, refused BEFORE anything is encoded."""
    code = _run_cli(
        monkeypatch,
        _cli_argv(tmp_path, tmp_path / "cache.npz"),
        pair_raises=RasterioIOError("cannot open band file"),
    )

    assert code == 2
    assert not (tmp_path / "cache.npz").exists()


def test_cli_exits_three_when_the_extractor_raises_a_typed_error(
    monkeypatch, tmp_path: Path
) -> None:
    """A wrong-width feature is a FusionTrainingError raised mid-run: exit 3."""
    code = _run_cli(
        monkeypatch,
        _cli_argv(tmp_path, tmp_path / "cache.npz"),
        samples=[make_sample("p0")],
        encoder=FakeEncoder(resolution=CONFIG_RESOLUTION),
        extractor_raises=FusionTrainingError("feature width 2317 is not 2318"),
    )

    assert code == 3


def test_cli_exits_two_when_the_out_cache_is_unwritable(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """Refused in the input checks, not after an hour of encoding."""
    target = tmp_path / "a_directory"
    target.mkdir()

    code = _run_cli(monkeypatch, _cli_argv(tmp_path, target), samples=[make_sample("p0")])

    assert code == 2
    assert "cannot write" in capsys.readouterr().out


def test_cli_exits_three_when_the_destination_fails_mid_run(
    monkeypatch, tmp_path: Path
) -> None:
    """A PermissionError after the run started is 3, not a pre-flight 2."""
    code = _run_cli(
        monkeypatch,
        _cli_argv(tmp_path, tmp_path / "cache.npz"),
        samples=[make_sample("p0")],
        encoder=FakeEncoder(resolution=CONFIG_RESOLUTION),
        extractor_raises=PermissionError("the destination is not writable"),
    )

    assert code == 3


# ---------------------------------------------------------------------------
# The dry run applies the label policy — it is a plan, not a head count
#
# BigEarthNet v2.0 is MULTI-LABEL, so counting candidates before the policy
# over-reports by every ambiguous patch — largest exactly where a dry run is
# most useful. These tests pin the post-policy count, from both paths.
# ---------------------------------------------------------------------------

#: The reported defect's shape, reproduced: 5 train candidates of which one is
#: MULTI-LABEL, and 3 val patches the split firewall must exclude. The pre-policy
#: count is 5 and the post-policy count is 4, so a plan that skips the policy is
#: off by exactly one and the assertions below can tell the two apart.
SYNTH_TRAIN: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("p0", ("Arable land",)),
    ("p1", ("Arable land",)),
    ("p2", ("Arable land",)),
    ("p3", ("Arable land", "Pastures")),  # multi-label -> skip_ambiguous drops it
    ("p4", ("Arable land",)),
)
#: A patch with NO labels. It is a real corpus case, and `pair_patches` drops it
#: during discovery (`require_labels=True`), so it never reaches the policy and
#: the CLI's unlabelled count is legitimately 0. Pinned below rather than
#: assumed: if discovery ever stops filtering, the candidate count moves.
SYNTH_UNLABELLED = "p5"
SYNTH_VAL: tuple[str, ...] = ("p6", "p7", "p8")

SYNTH_PAIRED = 8
SYNTH_TRAIN_CANDIDATES = 5
SYNTH_SELECTED = 4
SYNTH_SKIPPED_AMBIGUOUS = 1


def build_synth_corpus(root: Path) -> Path:
    """A two-tile corpus with a known multi-label patch (and one label-less)."""
    train_tile = root / "T01ABC"
    for name, labels in SYNTH_TRAIN:
        write_stub_patch(train_tile, name, REAL_S2_BANDS, labels=labels, split="train")
    write_stub_patch(
        train_tile, SYNTH_UNLABELLED, REAL_S2_BANDS, labels=(), split="train"
    )
    val_tile = root / "T02ABC"
    for name in SYNTH_VAL:
        write_stub_patch(
            val_tile, name, REAL_S2_BANDS, labels=("Arable land",), split="val"
        )
    return root


def _train_samples(root: Path):
    """The corpus as the CLI sees it, filtered to the requested split."""
    paired = list(pair_patches(root, loader=ramp_band_loader))
    assert len(paired) == SYNTH_PAIRED, "discovery must drop the label-less patch"
    return [sample for sample in paired if sample.official_split == "train"]


def test_the_plan_and_the_run_report_the_same_post_policy_count(
    tmp_path: Path,
) -> None:
    root = build_synth_corpus(tmp_path / "corpus")
    train = _train_samples(root)

    plan = plan_extraction(train, label_policy=skip_ambiguous, split="train")

    assert plan.n_requested == SYNTH_TRAIN_CANDIDATES
    assert plan.n_selected == SYNTH_SELECTED, "the plan counted pre-policy candidates"
    assert plan.n_skipped_by_policy == SYNTH_SKIPPED_AMBIGUOUS
    assert plan.n_skipped_unlabelled == 0, (
        "discovery drops label-less patches before the policy ever sees them"
    )
    assert plan.n_would_write == SYNTH_SELECTED
    assert plan.n_scenes == 1
    assert plan.label_policy == "skip_ambiguous"

    result = extract_fusion_features(
        train,
        encoder=FakeEncoder(),
        label_policy=skip_ambiguous,
        split="train",
        cache_path=tmp_path / "cache.npz",
    )

    # Every number the plan predicted, the run reports.
    assert result.n_encoded == plan.n_selected == SYNTH_SELECTED
    assert result.n_total == plan.n_would_write == SYNTH_SELECTED
    assert result.n_skipped_by_policy == plan.n_skipped_by_policy
    assert result.n_skipped_unlabelled == plan.n_skipped_unlabelled == 0
    assert result.n_scenes == plan.n_scenes == 1

    features, _ = read_feature_cache(tmp_path / "cache.npz")
    assert [feature.sample_id for feature in features] == ["p0", "p1", "p2", "p4"]


def test_the_plan_accounts_for_unlabelled_samples_when_the_caller_supplies_them(
    tmp_path: Path,
) -> None:
    """`require_labels=False` callers DO reach the policy with no labels.

    The CLI's corpus walk filters them out, so this pins the accounting on the
    path where they are present — an unlabelled sample is counted, never mapped
    to class 0.
    """
    samples = [
        make_sample("p0"),
        make_sample("p1", labels=()),
        make_sample("p2", labels=("Arable land", "Pastures")),
        make_sample("p3"),
    ]

    plan = plan_extraction(samples, label_policy=skip_ambiguous, split="train")

    assert plan.n_requested == 4
    assert plan.n_selected == 2
    assert plan.n_skipped_unlabelled == 1
    assert plan.n_skipped_by_policy == 2
    assert plan.n_would_write == 2

    result = extract_fusion_features(
        samples,
        encoder=FakeEncoder(),
        label_policy=skip_ambiguous,
        split="train",
        cache_path=tmp_path / "cache.npz",
    )

    assert result.n_encoded == plan.n_selected == 2
    assert result.n_skipped_unlabelled == plan.n_skipped_unlabelled == 1
    assert result.n_skipped_by_policy == plan.n_skipped_by_policy == 2
    features, _ = read_feature_cache(tmp_path / "cache.npz")
    assert [feature.sample_id for feature in features] == ["p0", "p3"]


def test_the_cli_dry_run_and_the_cli_run_agree(monkeypatch, tmp_path, capsys) -> None:
    root = build_synth_corpus(tmp_path / "corpus")
    out = tmp_path / "cache.npz"

    def reader(*args, **kwargs):
        return list(pair_patches(root, loader=ramp_band_loader))

    argv = [*_cli_argv(root, out, checkpoint_dir=tmp_path), "--label-policy",
            "skip_ambiguous"]

    dry_code = _run_cli(monkeypatch, [*argv, "--dry-run"], pair_callable=reader)
    dry = capsys.readouterr().out

    assert dry_code == 0
    assert f"Planned: {SYNTH_SELECTED} sample(s)" in dry
    assert f"will encode      : {SYNTH_SELECTED}" in dry
    assert f"Would encode: {SYNTH_SELECTED} sample(s)" in dry
    assert f"skipped ambiguous: {SYNTH_SKIPPED_AMBIGUOUS}" in dry
    assert "skipped unlabelled: 0" in dry
    assert f"Planned: {SYNTH_TRAIN_CANDIDATES} sample(s)" not in dry, (
        "the dry run planned the PRE-policy candidate count"
    )

    real_code = _run_cli(
        monkeypatch,
        argv,
        pair_callable=reader,
        encoder=FakeEncoder(resolution=CONFIG_RESOLUTION),
    )
    real = capsys.readouterr().out

    assert real_code == 0
    assert re.search(r"n_encoded\s+4\b", real), real
    assert re.search(r"n_total\s+4\b", real), real
    # The CLI checks the plan against the run itself and reports the verdict.
    assert re.search(r"agrees\s+True", real), real

    features, _ = read_feature_cache(out)
    assert [feature.sample_id for feature in features] == ["p0", "p1", "p2", "p4"]


def test_the_dry_run_counts_the_rows_already_cached(monkeypatch, tmp_path, capsys) -> None:
    root = build_synth_corpus(tmp_path / "corpus")
    out = tmp_path / "cache.npz"

    def reader(*args, **kwargs):
        return list(pair_patches(root, loader=ramp_band_loader))

    argv = [*_cli_argv(root, out, checkpoint_dir=tmp_path), "--label-policy",
            "skip_ambiguous"]

    assert _run_cli(
        monkeypatch, argv, pair_callable=reader,
        encoder=FakeEncoder(resolution=CONFIG_RESOLUTION),
    ) == 0
    capsys.readouterr()

    assert _run_cli(monkeypatch, [*argv, "--dry-run"], pair_callable=reader) == 0
    dry = capsys.readouterr().out

    assert f"already cached   : {SYNTH_SELECTED} row(s)" in dry
    assert f"{SYNTH_SELECTED} candidate(s) skipped as already present" in dry
    assert "Planned: 0 sample(s)" in dry
    assert f"rows after write : {SYNTH_SELECTED}" in dry


def test_a_dry_run_fails_on_an_ambiguous_sample_like_the_run_does(
    monkeypatch, tmp_path, capsys
) -> None:
    """`require_single_label` must not be planned around — it must fail."""
    root = build_synth_corpus(tmp_path / "corpus")
    train = _train_samples(root)

    with pytest.raises(AmbiguousLabelError, match="Pastures"):
        plan_extraction(train, label_policy=require_single_label, split="train")

    def reader(*args, **kwargs):
        return list(pair_patches(root, loader=ramp_band_loader))

    argv = [*_cli_argv(root, tmp_path / "cache.npz", checkpoint_dir=tmp_path),
            "--label-policy", "require_single_label"]

    dry_code = _run_cli(monkeypatch, [*argv, "--dry-run"], pair_callable=reader)
    assert "AmbiguousLabelError" in capsys.readouterr().out
    assert dry_code == 3, "the dry run planned around an ambiguous sample"

    real_code = _run_cli(
        monkeypatch, argv, pair_callable=reader,
        encoder=FakeEncoder(resolution=CONFIG_RESOLUTION),
    )
    assert real_code == dry_code == 3
    assert not (tmp_path / "cache.npz").exists()


def test_select_samples_is_the_one_place_the_policy_is_applied() -> None:
    """The shared decision, asserted directly so a drift is caught here."""
    samples = [
        make_sample("a"),
        make_sample("b", labels=("Arable land", "Pastures")),
        make_sample("c", labels=()),
        make_sample("d"),
    ]

    selection = select_samples(
        samples, label_policy=skip_ambiguous, existing_ids={"d"}
    )

    assert [sample.patch_id for sample, _ in selection.to_encode] == ["a"]
    assert selection.n_skipped_existing == 1
    assert selection.n_skipped_by_policy == 2
    assert selection.n_skipped_unlabelled == 1
    assert selection.scene_ids == ("ben_T01",)


def test_cli_refuses_a_patch_id_that_leaks_into_the_other_cache(
    monkeypatch, tmp_path: Path
) -> None:
    """The row-level firewall is wired into the real path, before the write.

    The scene-level pre-check cannot see this one: the same patch id is recorded
    in both partitions under two different scene ids. Only
    `assert_train_val_disjoint`, run over the rows about to be written, catches
    it — and the cache is never produced.
    """
    other = tmp_path / "val.npz"
    write_feature_cache(other, [make_feature("p0", "ben_T99")])
    out = tmp_path / "train.npz"

    code = _run_cli(
        monkeypatch,
        [*_cli_argv(tmp_path, out), "--other-cache", str(other)],
        samples=[make_sample("p0", tile="T01"), make_sample("p1", tile="T02")],
        encoder=FakeEncoder(resolution=CONFIG_RESOLUTION),
    )

    assert code == 2
    assert not out.exists(), "the leaky cache was written anyway"
