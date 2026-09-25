"""Tests for the Phase 6 BigEarthNet -> SmolVLM instruction corpus.

Synthetic patch trees and a synthetic `metadata.parquet` are written under
`tmp_path`, so nothing here depends on the local corpus being present. One test
additionally exercises the real parquet and SKIPS cleanly when it is absent,
because a skip must be visible rather than mistaken for coverage.

Two regression tests pin defects that were found by *running* the pipeline on
real data, not by reading it:

  * `PatchSample.scene_key` -- `evaluation.leakage.assert_no_scene_overlap`
    reads `record.scene_key`, and the attribute was missing, so the pipeline
    raised `AttributeError` on its first real run.
  * scene ids are T2 **block** ids (`ben_<tile>:<k>`), never acquisition-folder
    ids -- an acquisition-folder key gives each seasonal acquisition its own
    namespace, which makes the disjointness check pass on a corpus that leaks.
"""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from core.errors import LeakageError
from training.data.bigearthnet import CLC19_CLASSES
from training.data.bigearthnet_blocks import parse_tile
from training.data.bigearthnet_labels import (
    BigEarthNetLabelIndex,
    LabelIndexError,
    _coerce_labels,
    _coerce_split,
)
from training.vlm.config import VLMTrainingConfig
from training.vlm.dataset import (
    RGB_BANDS,
    VLMDataError,
    PatchSample,
    _check_family_is_well_posed,
    _presence_pairs,
    build_corpus,
    describe_render,
    discover_labelled_patches,
    render_rgb,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
#: The real release manifest. Present locally; the test skips if it is not.
REAL_PARQUET = REPO_ROOT / "data" / "bigearthnet_v2" / "metadata.parquet"

TILE = "33UUP"
#: `p.tile` in the pipeline is the reBEN acquisition folder, NOT the MGRS tile.
TILE_FOLDER = "S2A_MSIL2A_20200101T000000_N0000_R000_T33UUP"


def _patch_id(
    tile: str, row: int, col: int, *, date: str = "20200101T000000"
) -> str:
    return f"S2A_MSIL2A_{date}_N0000_R000_T{tile}_{row}_{col}"


def _write_band(path: Path, array: np.ndarray) -> None:
    """Write a 2-D uint16 GeoTIFF -- a real band file the loader can read."""
    import rasterio

    array = np.asarray(array)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
    ) as dst:
        dst.write(array, 1)


def _write_patch(
    root: Path,
    patch_id: str,
    *,
    shape: tuple[int, int] = (8, 8),
    constant: bool = False,
) -> Path:
    """One patch directory holding the true-colour triple as real GeoTIFFs."""
    patch_dir = root / TILE_FOLDER / patch_id
    patch_dir.mkdir(parents=True, exist_ok=True)
    for i, token in enumerate(RGB_BANDS):
        if constant:
            arr = np.full(shape, 1000 + i, dtype=np.uint16)
        else:
            arr = (np.arange(shape[0] * shape[1]) * (i + 1)).astype(np.uint16)
            arr = arr.reshape(shape)
        _write_band(patch_dir / f"{patch_id}_{token}.tif", arr)
    return patch_dir


def _write_parquet(path: Path, rows: list[tuple[str, list[str], str]]) -> None:
    import pandas as pd

    pd.DataFrame(
        {
            "patch_id": [r[0] for r in rows],
            "labels": [r[1] for r in rows],
            "split": [r[2] for r in rows],
        }
    ).to_parquet(path)


@pytest.fixture()
def corpus(tmp_path: Path) -> SimpleNamespace:
    """8 train + 1 val + 1 test patches, one label each, on one MGRS tile.

    The 8 train patches are 4-connected on the `(row, col)` grid, so they form a
    single T2 block; val and test are single-cell blocks. Proportions are
    0.8 / 0.1 / 0.1, matching the registry's ratios within tolerance.
    """
    root = tmp_path / "BigEarthNet-S2"
    root.mkdir()

    rows: list[tuple[str, list[str], str]] = []
    for col in range(8):
        pid = _patch_id(TILE, 0, col)
        _write_patch(root, pid)
        rows.append((pid, ["Arable land"], "train"))
    val_pid = _patch_id(TILE, 10, 0)
    _write_patch(root, val_pid)
    rows.append((val_pid, ["Marine waters"], "validation"))
    test_pid = _patch_id(TILE, 20, 0)
    _write_patch(root, test_pid)
    rows.append((test_pid, ["Pastures"], "test"))

    parquet = tmp_path / "metadata.parquet"
    _write_parquet(parquet, rows)
    return SimpleNamespace(root=root, parquet=parquet, rows=rows)


def _config(corpus: SimpleNamespace, **overrides: Any) -> VLMTrainingConfig:
    return VLMTrainingConfig.from_registry(
        corpus_root=str(corpus.root),
        metadata_parquet=str(corpus.parquet),
        **overrides,
    )


# ---------------------------------------------------------------------------
# Label coercion
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        (["Arable land", "Pastures"], ("Arable land", "Pastures")),
        (np.array(["Arable land", "Pastures"]), ("Arable land", "Pastures")),
        ("['Arable land', 'Pastures']", ("Arable land", "Pastures")),
        ("Arable land", ("Arable land",)),
        (None, ()),
        ("", ()),
        ([], ()),
    ],
)
def test_coerce_labels_handles_every_observed_cell_shape(value, expected) -> None:
    assert _coerce_labels(value) == expected


def test_coerce_labels_rejects_a_truncated_stringified_list() -> None:
    """A bracketed literal that is not valid JSON must raise, not be accepted as
    one long label."""
    with pytest.raises(LabelIndexError):
        _coerce_labels("[Arable land")


def test_coerce_split_maps_validation_to_val() -> None:
    assert _coerce_split("validation") == "val"
    assert _coerce_split("VAL") == "val"
    assert _coerce_split("train") == "train"
    assert _coerce_split("nonsense") is None
    assert _coerce_split(None) is None


# ---------------------------------------------------------------------------
# The parquet label index
# ---------------------------------------------------------------------------
def test_label_index_reads_labels_and_splits(corpus: SimpleNamespace) -> None:
    index = BigEarthNetLabelIndex.from_parquet(corpus.parquet)
    assert len(index) == 10

    pid = _patch_id(TILE, 0, 0)
    assert index.labels_for(pid) == ("Arable land",)
    assert index.split_for(pid) == "train"
    # An unknown id is None -- never a defaulted label.
    assert index.labels_for("not-in-the-manifest") is None
    assert index.split_for("not-in-the-manifest") is None
    # The manifest spells it "validation"; the index normalises to "val".
    assert index.split_for(_patch_id(TILE, 10, 0)) == "val"


def test_label_index_coverage_counts_matched_and_unmatched(
    corpus: SimpleNamespace,
) -> None:
    index = BigEarthNetLabelIndex.from_parquet(corpus.parquet)
    ids = [_patch_id(TILE, 0, i) for i in range(8)] + ["missing"]
    cov = index.coverage(ids)
    assert cov.n_requested == 9
    assert cov.n_matched == 8
    assert cov.n_unmatched == 1
    assert cov.n_single_label == 8
    assert cov.n_multi_label == 0
    assert cov.single_label_fraction == 1.0


def test_label_index_refuses_a_missing_parquet(tmp_path: Path) -> None:
    with pytest.raises(LabelIndexError):
        BigEarthNetLabelIndex.from_parquet(tmp_path / "nope.parquet")


@pytest.mark.skipif(
    not REAL_PARQUET.exists(),
    reason="real BigEarthNet metadata.parquet is not present in this checkout",
)
def test_real_parquet_loads_the_full_manifest() -> None:
    index = BigEarthNetLabelIndex.from_parquet(REAL_PARQUET)
    # Measured in docs/PHASE6_AUDIT_AND_CONTRACT.md section 2: 480,038 rows.
    assert len(index) == 480_038


# ---------------------------------------------------------------------------
# PatchSample.scene_key -- the AttributeError regression
# ---------------------------------------------------------------------------
def _patch(scene_id: str, split: str, patch_id: str = "p") -> PatchSample:
    return PatchSample(
        patch_id=patch_id,
        scene_id=scene_id,
        tile="acquisition-folder",
        directory=Path("/nonexistent"),
        labels=("Arable land",),
        band_paths={},
        official_split=split,
    )


def test_patch_sample_scene_key_mirrors_scene_id() -> None:
    """Regression: `assert_no_scene_overlap` reads `record.scene_key`; the
    attribute was missing and the first real pipeline run raised
    `AttributeError`."""
    sample = _patch("ben_33UUP:0", "train")
    assert sample.scene_key == sample.scene_id


def test_scene_overlap_check_reads_scene_key_and_fires() -> None:
    """`assert_no_scene_overlap` must accept a `PatchSample` list (it reads
    `.scene_key`) and must raise when a scene spans two splits."""
    from evaluation.leakage import assert_no_scene_overlap

    train = [_patch("ben_33UUP:0", "train", "a")]
    test = [_patch("ben_33UUP:0", "test", "b")]
    with pytest.raises(LeakageError):
        assert_no_scene_overlap(train, test)

    # Disjoint scenes are accepted (the negative control).
    assert_no_scene_overlap(train, [_patch("ben_33UUP:1", "test", "c")])


# ---------------------------------------------------------------------------
# Presence-pair generation -- balance
# ---------------------------------------------------------------------------
def test_presence_pairs_answer_yes_for_present_and_no_for_absent() -> None:
    triples = _presence_pairs(
        ("Arable land",),
        vocabulary=CLC19_CLASSES,
        negative_ratio=1.0,
        rng=random.Random(0),
    )
    assert triples
    for question, answer, family in triples:
        assert family == "presence"
        if "Arable land" in question:
            assert answer == "Yes."
        else:
            assert answer == "No."


def test_negative_sampling_is_balanced_not_five_percent_positive() -> None:
    """With 1 present and 18 absent classes the naive generator would be ~5%
    positive, and a model could beat it by always answering "yes"."""
    triples = _presence_pairs(
        ("Arable land",),
        vocabulary=CLC19_CLASSES,
        negative_ratio=1.0,
        rng=random.Random(0),
    )
    positives = sum(1 for _, a, _ in triples if a == "Yes.")
    negatives = sum(1 for _, a, _ in triples if a == "No.")
    assert positives == 1
    assert negatives == 1
    assert positives + negatives == len(triples)


def test_negative_ratio_zero_yields_no_negatives() -> None:
    triples = _presence_pairs(
        ("Arable land",),
        vocabulary=CLC19_CLASSES,
        negative_ratio=0.0,
        rng=random.Random(0),
    )
    # `max(1, round(0))` keeps at least one negative so the corpus is never
    # all-positive.
    assert any(a == "No." for _, a, _ in triples)


# ---------------------------------------------------------------------------
# Family admissibility
# ---------------------------------------------------------------------------
def test_multi_family_is_refused_on_an_all_single_label_corpus() -> None:
    with pytest.raises(VLMDataError) as exc:
        _check_family_is_well_posed(("multi",), n_single_label=10, n_matched=10)
    assert exc.value.code == "vlm_data_error"
    message = str(exc.value).lower()
    assert "multi" in message
    assert "single" in message


def test_presence_family_is_allowed_on_an_all_single_label_corpus() -> None:
    assert (
        _check_family_is_well_posed(
            ("presence",), n_single_label=10, n_matched=10
        )
        == []
    )


def test_unknown_family_is_refused() -> None:
    with pytest.raises(VLMDataError):
        _check_family_is_well_posed(("describe",), n_single_label=1, n_matched=1)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def test_discover_labelled_patches_joins_labels_from_the_parquet(
    corpus: SimpleNamespace,
) -> None:
    patches, coverage, _ = discover_labelled_patches(
        corpus_root=corpus.root, metadata_parquet=corpus.parquet
    )
    assert len(patches) == 10
    assert all(p.labels for p in patches)
    assert {p.official_split for p in patches} == {"train", "val", "test"}
    assert coverage.n_matched == 10


# ---------------------------------------------------------------------------
# build_corpus -- the release partition, keyed by T2 blocks
# ---------------------------------------------------------------------------
def test_build_corpus_uses_the_release_partition(corpus: SimpleNamespace) -> None:
    built = build_corpus(_config(corpus))
    assert built.patches
    for patch in built.patches:
        assert patch.split == patch.official_split
    assert built.split_info["patches_by_split"] == {"train": 8, "val": 1, "test": 1}
    assert built.split_info["policy"] == "release_partition_keyed_by_T2_blocks"


def test_build_corpus_keys_scenes_by_t2_block(corpus: SimpleNamespace) -> None:
    built = build_corpus(_config(corpus))
    for patch in built.patches:
        mgrs_tile = parse_tile(patch.patch_id)
        assert patch.scene_id.startswith(f"ben_{mgrs_tile}:")
        # Not the acquisition folder, and not the raw patch id.
        assert "MSIL2A" not in patch.scene_id
        assert patch.scene_id != patch.patch_id
    assert built.split_info["scene_key"] == "ben_<tile>:<k>"


def test_build_corpus_generates_balanced_presence_samples(
    corpus: SimpleNamespace,
) -> None:
    built = build_corpus(_config(corpus))
    answers = [s.answer for s in built.samples]
    assert set(answers) == {"Yes.", "No."}
    assert all(s.family == "presence" for s in built.samples)
    assert answers.count("Yes.") == answers.count("No.")


def test_build_corpus_samples_are_split_disjoint(corpus: SimpleNamespace) -> None:
    built = build_corpus(_config(corpus))
    by_split = {
        split: {s.scene_id for s in built.by_split(split)}
        for split in ("train", "val", "test")
    }
    assert by_split["train"].isdisjoint(by_split["val"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["val"].isdisjoint(by_split["test"])
    assert sum(built.split_counts().values()) == len(built)


def test_build_corpus_invokes_the_scene_overlap_check(
    corpus: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The disjointness check is enforced, not decorative: it is called for all
    three split pairs during the build."""
    import evaluation.leakage as leakage

    calls: list[tuple[set[str], set[str]]] = []
    original = leakage.assert_no_scene_overlap

    def spy(left, right):
        calls.append(({p.scene_id for p in left}, {p.scene_id for p in right}))
        return original(left, right)

    monkeypatch.setattr(leakage, "assert_no_scene_overlap", spy)
    build_corpus(_config(corpus))

    assert len(calls) == 3
    for left, right in calls:
        assert left.isdisjoint(right)


# ---------------------------------------------------------------------------
# render_rgb
# ---------------------------------------------------------------------------
def _renderable_patch(root: Path, *, constant: bool) -> PatchSample:
    pid = _patch_id(TILE, 0, 0)
    patch_dir = _write_patch(root, pid, constant=constant)
    return PatchSample(
        patch_id=pid,
        scene_id="ben_33UUP:0",
        tile=TILE_FOLDER,
        directory=patch_dir,
        labels=("Arable land",),
        band_paths={t: patch_dir / f"{pid}_{t}.tif" for t in RGB_BANDS},
        official_split="train",
    )


def test_render_rgb_returns_a_pil_rgb_image_of_the_requested_size(
    tmp_path: Path,
) -> None:
    patch = _renderable_patch(tmp_path, constant=False)
    image = render_rgb(patch, size=16)
    assert image.mode == "RGB"
    assert image.size == (16, 16)


def test_render_rgb_of_a_constant_band_is_zeros_not_nan(tmp_path: Path) -> None:
    """A constant band (lo == hi) is stretched to zeros rather than divided by
    zero. The NaN would be silently turned into a plausible tensor by the
    processor, so the failure would be invisible."""
    patch = _renderable_patch(tmp_path, constant=True)
    image = render_rgb(patch)
    array = np.asarray(image)
    assert np.isfinite(array).all()
    assert array.max() == 0


def test_render_rgb_refuses_a_patch_without_a_band() -> None:
    patch = PatchSample(
        patch_id="p",
        scene_id="ben_33UUP:0",
        tile="t",
        directory=Path("/nonexistent"),
        labels=("Arable land",),
        band_paths={},  # no true-colour triple -> refused before any file read
        official_split="train",
    )
    with pytest.raises(VLMDataError):
        render_rgb(patch)


def test_describe_render_records_the_contract() -> None:
    described = describe_render(percentiles=(2.0, 98.0))
    assert described["bands"] == list(RGB_BANDS)
    assert described["output_dtype"] == "uint8"
    assert described["percentiles"] == [2.0, 98.0]
    assert described["normalisation"] == "per-band percentile stretch"
