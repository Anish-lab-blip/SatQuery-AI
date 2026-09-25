"""Tests for the reBEN v2 -> PairedSample adapter (Phase 12).

Everything runs against a synthetic reBEN v2 tree in a temp dir with an INJECTED
band reader, so there is no network, no download and no geospatial reader in the
loop. The reader returns a per-band constant, which is what lets the tests check
WHERE a band landed rather than merely that the tensor has the right shape.

The tests that matter most are the ones that pin the four real-data defects this
module exists to fix -- each is written so that it FAILS under the defect it
targets:

  * D1 labels and the split come from the metadata parquet, because the
    extracted reBEN tree ships band `.tif` files and no per-patch metadata at all
  * D2 the S1 modality is joined on the released `s1_name`, NOT on the patch id
    -- the two name sets are disjoint in the real corpus (0 of 28,000)
  * D3 every band is resampled onto one common grid, because reBEN stores 10 m,
    20 m and 60 m bands inside the SAME patch
  * D4 the leakage boundary is the manifest's block id (`ben_<tile>:<n>`), not
    `ben_<tile>` -- 12 of the 47 tiles in the shipped selection carry blocks in
    more than one split, so the tile key raises a spurious leak abort

WHY THE SYNTHETIC NAMES ARE ABBREVIATED
---------------------------------------
The real release names are ~50 characters, and
`reben/BigEarthNet-S2/<granule>/<patch>/<patch>_B01.tif` built from them lands at
284 characters under a `pytest` temp root. That fails, and it fails *confusingly*:
`Path.write_bytes` goes through the CRT's `open()`, which is NOT long-path aware,
while Python's `os.mkdir` DOES add the `\\?\` prefix -- so `mkdir(parents=True)`
reports success and the very next `write_bytes` raises `FileNotFoundError` for a
directory that demonstrably exists. Windows' limit is 260 characters.

The constants below therefore keep the real names' TOKEN COUNTS (which is what
`_granule_of_patch_id` / `_granule_of_s1_name` depend on) at a fraction of their
length. The real names are still exercised, off the filesystem, by
`test_granule_rules_hold_for_real_corpus_names`, and
`_write_bands` refuses to write past a budget so a future edit that lengthens a
name fails with an explanation instead of a bare `FileNotFoundError`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from training.data.bigearthnet import (
    CANONICAL_OPTICAL_BANDS,
    CANONICAL_SAR_BANDS,
    PairedSample,
)
from training.fusion.reben_adapter import (
    DEFAULT_TARGET_SHAPE,
    METADATA_FILES,
    RESAMPLING_MODES,
    RebenLayoutError,
    build_reben_samples,
    load_metadata_index,
    load_selection_manifest,
    resolve_patch_dirs,
)

#: What real BigEarthNet-S2 ships: 12 bands, no cirrus (B10).
REAL_S2_BANDS: tuple[str, ...] = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09",
    "B11", "B12",
)

#: A distinct constant per band token, so a placed band is identifiable by value.
BAND_VALUE: dict[str, float] = {
    band: float(i + 1) for i, band in enumerate(REAL_S2_BANDS)
}
BAND_VALUE["VV"] = 21.0
BAND_VALUE["VH"] = 22.0

#: The adapter's target grid; deliberately NOT the source shape, so a reader that
#: ignored `shape` would be caught.
TARGET = (8, 8)

#: Deliberately unlike the S2 patch id, exactly as the real corpus is. Both keep
#: the real names' token counts -- 6 and 6 -- so the "strip the last 2 / last 3
#: tokens" granule rules are exercised on the same shape the release uses.
PATCH_ID = "S2A_MSIL2A_20200101_T33UUP_43_85"
S1_NAME = "S1A_IW_GRDH_20200101_33UUP_68_74"
GRANULE_S2 = "S2A_MSIL2A_20200101_T33UUP"
GRANULE_S1 = "S1A_IW_GRDH_20200101"
TILE = "33UUP"
BLOCK = "ben_33UUP:1"

#: The REAL release names, ~50 chars each, used by the off-filesystem granule test.
REAL_PATCH_ID = "S2B_MSIL2A_20170927T094019_N9999_R036_T35VLC_09_85"
REAL_S1_NAME = "S1A_IW_GRDH_1SDV_20170928T161150_35VLC_9_85"

#: Windows' non-long-path-aware limit, and the budget the fixtures hold themselves
#: to so they still fit under a default `pytest` basetemp.
MAX_PATH = 260
PATH_BUDGET = 250

#: Records captured from the injected reader, so a test can assert what it was
#: asked for (the resampling mode and the target shape).
CALLS: list[tuple[Path, tuple[int, int], str]] = []


def fake_reader(path: Path, shape: tuple[int, int], resampling: str) -> np.ndarray:
    """Return a constant array identifying the band token in the file name."""
    CALLS.append((path, shape, resampling))
    token = path.stem.split("_")[-1]
    return np.full(shape, BAND_VALUE[token], dtype=np.float32)


def _write_bands(directory: Path, stem: str, tokens: tuple[str, ...]) -> None:
    """Create the band stubs, refusing to build a path Windows cannot open.

    Without this guard an over-long fixture name fails as
    `FileNotFoundError: ...` pointing at a directory `mkdir` had just created,
    which reads as a bug in the adapter. It is not: it is MAX_PATH.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for token in tokens:
        target = directory / f"{stem}_{token}.tif"
        if len(str(target)) >= PATH_BUDGET:
            raise AssertionError(
                f"fixture path is {len(str(target))} chars, at or over the "
                f"{PATH_BUDGET}-char budget (Windows MAX_PATH is {MAX_PATH}, and "
                f"the CRT open() behind write_bytes is not long-path aware): "
                f"{target}\n"
                f"Shorten the synthetic names, or run pytest with a shallower "
                f"--basetemp. Do not 'fix' this in the adapter."
            )
        target.write_bytes(b"stub")


def _parquet(path: Path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(
        pa.table(
            {
                "patch_id": pa.array([r["patch_id"] for r in rows], type=pa.string()),
                "labels": pa.array([r["labels"] for r in rows],
                                   type=pa.list_(pa.string())),
                "split": pa.array([r["split"] for r in rows], type=pa.string()),
                "s1_name": pa.array([r["s1_name"] for r in rows], type=pa.string()),
            }
        ),
        path,
    )


def build_corpus(
    root: Path,
    *,
    records: list[dict] | None = None,
    missing_optical_bands: tuple[str, ...] = (),
    make_sar: bool = True,
    make_optical: bool = True,
) -> Path:
    """Write a synthetic reBEN v2 tree + manifest + metadata parquet."""
    if records is None:
        records = [
            {
                "sample_id": PATCH_ID,
                "source_scene": TILE,
                "scene_id": BLOCK,
                "split": "train",
                "dataset_id": "bigearthnet",
                "labels": ["Urban fabric"],
                "s1_name": S1_NAME,
                "official_split": "train",
            }
        ]

    reben = root / "reben"
    if make_optical:
        for r in records:
            stem = r["sample_id"]
            granule = "_".join(stem.split("_")[:-2])
            tokens = tuple(t for t in REAL_S2_BANDS
                           if t not in missing_optical_bands)
            _write_bands(reben / "BigEarthNet-S2" / granule / stem, stem, tokens)
    if make_sar:
        for r in records:
            stem = r["s1_name"]
            granule = "_".join(stem.split("_")[:-3])
            _write_bands(reben / "BigEarthNet-S1" / granule / stem, stem,
                         ("VV", "VH"))

    header = {
        "_manifest": {
            "name": "synthetic_t2",
            "hash": "deadbeef",
            "count": len(records),
            "scenes": len({r["scene_id"] for r in records}),
            "notes": {"label_policy": "skip_ambiguous", "scene_key": "T2"},
        }
    }
    manifest = root / "selection.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(o) for o in [header, *records]) + "\n",
        encoding="utf-8",
    )

    tables = root / "tables"
    tables.mkdir(exist_ok=True)
    _parquet(
        tables / METADATA_FILES[0],
        [
            {"patch_id": r["sample_id"], "labels": r["labels"],
             "split": r["official_split"], "s1_name": r["s1_name"]}
            for r in records
        ],
    )
    return root


def build(root: Path, **kwargs):
    """Shorthand: build the corpus then run the adapter against it."""
    build_corpus(root, **kwargs)
    return build_reben_samples(
        manifest_path=root / "selection.jsonl",
        reben_root=root / "reben",
        parquet_dir=root / "tables",
        target_shape=TARGET,
        reader=fake_reader,
    )


@pytest.fixture(autouse=True)
def _clear_calls():
    CALLS.clear()
    yield
    CALLS.clear()


# ---------------------------------------------------------------------------
# Fixture hygiene -- the constants themselves
# ---------------------------------------------------------------------------


#: A `pytest` root is the basetemp plus the per-test directory pytest appends.
#: Measured at 85 chars for a default temp basetemp and 109 for the deep
#: `--basetemp` this suite was first run under.
SHALLOWEST_ROOT = 85
HEADROOM = 100


def test_fixture_names_leave_room_for_a_temp_root() -> None:
    """Guard: the synthetic names must not outgrow Windows' 260-char MAX_PATH.

    Measured contributions of the path below `tmp_path`:

      * abbreviated names (below): 122 chars -> 122 + 109 = 231, fits
      * the real release names:    175 chars -> 175 + 109 = 284, does NOT fit

    That 284 is the exact failure this fixture hit. `Path.write_bytes` goes
    through the CRT `open()`, which is not long-path aware, so the failure
    surfaced as `FileNotFoundError` on a directory `mkdir` had just created.
    """
    deepest = f"reben/BigEarthNet-S2/{GRANULE_S2}/{PATCH_ID}/{PATCH_ID}_B01.tif"
    real_granule = "_".join(REAL_PATCH_ID.split("_")[:-2])
    real = (
        f"reben/BigEarthNet-S2/{real_granule}/{REAL_PATCH_ID}/{REAL_PATCH_ID}_B01.tif"
    )

    assert len(deepest) + HEADROOM < MAX_PATH, (
        f"the fixture now contributes {len(deepest)} chars; with {HEADROOM} chars "
        f"of headroom for the pytest root that is {len(deepest) + HEADROOM}, too "
        f"close to MAX_PATH ({MAX_PATH})"
    )
    assert len(real) + SHALLOWEST_ROOT >= MAX_PATH, (
        "if the real names ever fit the path budget, the abbreviations below are "
        "no longer justified -- switch the fixtures back to the real names"
    )


def test_the_granule_rules_agree_on_the_abbreviated_and_real_names() -> None:
    """The abbreviations are shorter (6 tokens vs the real 8) but must resolve
    to a granule the same way, because that is what `resolve_patch_dirs` relies
    on. The rule is positional -- strip the LAST 2 tokens for S2, the LAST 3 for
    S1 -- so it is the trailing structure that has to match, not the length.
    """
    assert "_".join(PATCH_ID.split("_")[:-2]) == GRANULE_S2
    assert "_".join(REAL_PATCH_ID.split("_")[:-2]) == (
        "S2B_MSIL2A_20170927T094019_N9999_R036_T35VLC"
    )
    assert "_".join(S1_NAME.split("_")[:-3]) == GRANULE_S1
    assert "_".join(REAL_S1_NAME.split("_")[:-3]) == (
        "S1A_IW_GRDH_1SDV_20170928T161150"
    )


def test_the_fixture_bands_cover_every_canonical_channel() -> None:
    """A canonical band with no fixture file would read as a missing modality."""
    assert set(REAL_S2_BANDS) == set(CANONICAL_OPTICAL_BANDS)
    assert len(CANONICAL_OPTICAL_BANDS) == 12
    assert tuple(CANONICAL_SAR_BANDS) == ("VV", "VH")


# ---------------------------------------------------------------------------
# D3 -- one common grid
# ---------------------------------------------------------------------------


def test_every_band_lands_on_the_common_grid(tmp_path: Path) -> None:
    """D3: a real patch mixes 10/20/60 m bands; all must come out one size."""
    (sample,) = build(tmp_path)
    assert sample.optical.shape == (12, *TARGET)
    assert sample.sar.shape == (2, *TARGET)
    # the reader was asked for the target grid, never for a native size
    assert {shape for _p, shape, _m in CALLS} == {TARGET}


def test_resampling_mode_reaches_the_reader(tmp_path: Path) -> None:
    build_corpus(tmp_path)
    build_reben_samples(
        manifest_path=tmp_path / "selection.jsonl",
        reben_root=tmp_path / "reben",
        parquet_dir=tmp_path / "tables",
        target_shape=TARGET,
        resampling="nearest",
        reader=fake_reader,
    )
    assert {mode for _p, _s, mode in CALLS} == {"nearest"}


def test_unknown_resampling_mode_is_refused(tmp_path: Path) -> None:
    build_corpus(tmp_path)
    with pytest.raises(RebenLayoutError, match="unknown resampling mode"):
        build_reben_samples(
            manifest_path=tmp_path / "selection.jsonl",
            reben_root=tmp_path / "reben",
            parquet_dir=tmp_path / "tables",
            resampling="lanczos",
            reader=fake_reader,
        )


def test_default_target_shape_is_croma_resolution() -> None:
    assert DEFAULT_TARGET_SHAPE == (120, 120)
    assert "bilinear" in RESAMPLING_MODES


# ---------------------------------------------------------------------------
# D1 -- labels and the split come from the parquet
# ---------------------------------------------------------------------------


def test_labels_and_split_come_from_the_metadata_table(tmp_path: Path) -> None:
    """D1: the tree has no per-patch metadata, so discovery cannot supply these."""
    (sample,) = build(tmp_path)
    assert sample.labels == ("Urban fabric",)
    assert sample.official_split == "train"


def test_a_patch_absent_from_the_table_is_refused(tmp_path: Path) -> None:
    """D1: the tree can hold a patch the released table does not describe.

    The labels and the split exist ONLY in the table, so this must be a loud
    failure -- there is no fallback that could invent either one.
    """
    build_corpus(tmp_path)
    _parquet(
        tmp_path / "tables" / METADATA_FILES[0],
        [{"patch_id": "S2A_MSIL2A_20200101_T33UUP_99_99", "labels": ["Urban fabric"],
          "split": "train", "s1_name": S1_NAME}],
    )
    with pytest.raises(RebenLayoutError, match="is not in either metadata table"):
        build_reben_samples(
            manifest_path=tmp_path / "selection.jsonl",
            reben_root=tmp_path / "reben",
            parquet_dir=tmp_path / "tables",
            reader=fake_reader,
        )


def test_a_missing_label_is_refused(tmp_path: Path) -> None:
    build_corpus(tmp_path)
    _parquet(
        tmp_path / "tables" / METADATA_FILES[0],
        [{"patch_id": PATCH_ID, "labels": [], "split": "train", "s1_name": S1_NAME}],
    )
    with pytest.raises(RebenLayoutError, match="has no labels"):
        build_reben_samples(
            manifest_path=tmp_path / "selection.jsonl",
            reben_root=tmp_path / "reben",
            parquet_dir=tmp_path / "tables",
            reader=fake_reader,
        )


def test_official_split_aliases_are_normalised(tmp_path: Path) -> None:
    """The release spells it `validation`; the pipeline's vocabulary is `val`."""
    build_corpus(
        tmp_path,
        records=[{
            "sample_id": PATCH_ID, "source_scene": TILE, "scene_id": BLOCK,
            "split": "val", "dataset_id": "bigearthnet",
            "labels": ["Urban fabric"], "s1_name": S1_NAME,
            "official_split": "validation",
        }],
    )
    (sample,) = build_reben_samples(
        manifest_path=tmp_path / "selection.jsonl",
        reben_root=tmp_path / "reben",
        parquet_dir=tmp_path / "tables",
        target_shape=TARGET,
        reader=fake_reader,
    )
    assert sample.official_split == "val"


def test_no_metadata_table_at_all_is_refused(tmp_path: Path) -> None:
    """An absent table is refused -- built by NOT writing one, not by deleting.

    Deliberately avoids `Path.unlink()`: this sandbox routes it through a trash
    shim whose bulk-delete guard counts deletions cumulatively across the whole
    turn, so a single `unlink()` in a test can abort the entire pytest process
    with `SystemExit(1)` once earlier runs have banked enough deletions.
    """
    tables = tmp_path / "tables"
    tables.mkdir()
    with pytest.raises(RebenLayoutError, match="no metadata table"):
        load_metadata_index(tables)


def test_a_metadata_table_with_no_rows_is_refused(tmp_path: Path) -> None:
    """A present-but-empty table is a defect, not a corpus with no patches.

    Returning `{}` here would defer the failure to a per-patch
    "not in either metadata table" much later in the run, for every patch.
    """
    build_corpus(tmp_path)
    _parquet(tmp_path / "tables" / METADATA_FILES[0], [])
    with pytest.raises(RebenLayoutError, match="hold no rows"):
        load_metadata_index(tmp_path / "tables")


# ---------------------------------------------------------------------------
# D2 -- the S1 join key
# ---------------------------------------------------------------------------


def test_sar_is_joined_on_s1_name_not_the_patch_id(tmp_path: Path) -> None:
    """D2: the two name sets are disjoint; joining on patch id finds no SAR."""
    assert PATCH_ID != S1_NAME
    (sample,) = build(tmp_path)
    assert sample.sar_mask.sum() == 2, (
        "SAR was not joined: the adapter must use the parquet's `s1_name`, not "
        "the patch id"
    )
    assert sample.sar_present_bands == tuple(CANONICAL_SAR_BANDS)


def test_a_missing_sar_directory_is_refused_not_zero_filled(tmp_path: Path) -> None:
    """An incomplete extraction must fail loudly, not become an absent modality."""
    build_corpus(tmp_path, make_sar=False)
    with pytest.raises(RebenLayoutError, match="SAR patch directory is missing"):
        build_reben_samples(
            manifest_path=tmp_path / "selection.jsonl",
            reben_root=tmp_path / "reben",
            parquet_dir=tmp_path / "tables",
            reader=fake_reader,
        )


def test_a_missing_optical_directory_is_refused(tmp_path: Path) -> None:
    build_corpus(tmp_path, make_optical=False)
    with pytest.raises(RebenLayoutError, match="optical patch directory is missing"):
        build_reben_samples(
            manifest_path=tmp_path / "selection.jsonl",
            reben_root=tmp_path / "reben",
            parquet_dir=tmp_path / "tables",
            reader=fake_reader,
        )


def test_resolve_patch_dirs_uses_the_documented_granule_rules(tmp_path: Path) -> None:
    build_corpus(tmp_path)
    optical, sar = resolve_patch_dirs(tmp_path / "reben", PATCH_ID, S1_NAME)
    assert optical == tmp_path / "reben" / "BigEarthNet-S2" / GRANULE_S2 / PATCH_ID
    assert sar == tmp_path / "reben" / "BigEarthNet-S1" / GRANULE_S1 / S1_NAME


def test_granule_rules_hold_for_real_corpus_names(tmp_path: Path) -> None:
    """The release's own ~50-char names must resolve to the same granules.

    Deliberately writes NO band file: the real leaf name would be 264 chars
    under a pytest root, past `MAX_PATH`. Directories are fine -- `os.mkdir` is
    long-path aware -- so the S1 rule is reached by creating only the optical
    directory and reading the error the missing SAR one produces.
    """
    reben = tmp_path / "reben"

    with pytest.raises(RebenLayoutError) as optical_exc:
        resolve_patch_dirs(reben, REAL_PATCH_ID, REAL_S1_NAME)
    optical_msg = str(optical_exc.value).replace("\\", "/")
    assert "BigEarthNet-S2/S2B_MSIL2A_20170927T094019_N9999_R036_T35VLC/" in optical_msg
    assert optical_msg.endswith(REAL_PATCH_ID)

    # satisfy the optical side so the SAR side is the one that reports
    optical_dir = (
        reben / "BigEarthNet-S2"
        / "_".join(REAL_PATCH_ID.split("_")[:-2]) / REAL_PATCH_ID
    )
    optical_dir.mkdir(parents=True)
    with pytest.raises(RebenLayoutError) as sar_exc:
        resolve_patch_dirs(reben, REAL_PATCH_ID, REAL_S1_NAME)
    sar_msg = str(sar_exc.value).replace("\\", "/")
    assert "BigEarthNet-S1/S1A_IW_GRDH_1SDV_20170928T161150/" in sar_msg
    assert sar_msg.endswith(REAL_S1_NAME)


# ---------------------------------------------------------------------------
# D4 -- the leakage boundary
# ---------------------------------------------------------------------------


def test_scene_key_is_the_manifest_block_id_not_the_tile(tmp_path: Path) -> None:
    """D4: `ben_<tile>` is too coarse and raises a spurious leak abort."""
    (sample,) = build(tmp_path)
    assert sample.scene_key == BLOCK
    assert sample.scene_id == BLOCK
    assert sample.scene_id != f"ben_{TILE}"


def test_tile_is_the_mgrs_tile_not_the_granule_directory(tmp_path: Path) -> None:
    (sample,) = build(tmp_path)
    assert sample.tile == TILE
    assert sample.tile != GRANULE_S2


def test_scene_id_falls_back_to_the_tile_without_a_key() -> None:
    """The historical behaviour must survive: 190 existing tests depend on it."""
    base = dict(
        patch_id="p", tile="T01",
        optical=np.zeros((12, 2, 2), np.float32), sar=np.zeros((2, 2, 2), np.float32),
        optical_mask=np.ones(12, np.float32), sar_mask=np.ones(2, np.float32),
        optical_present_bands=(), sar_present_bands=(),
        optical_missing_bands=(), sar_missing_bands=(),
        optical_ignored_bands=(), sar_ignored_bands=(),
        labels=("Urban fabric",), official_split="train",
    )
    assert PairedSample(**base).scene_id == "ben_T01"
    assert PairedSample(**base, scene_key=BLOCK).scene_id == BLOCK


def test_manifest_scene_id_is_required(tmp_path: Path) -> None:
    build_corpus(tmp_path)
    lines = (tmp_path / "selection.jsonl").read_text(encoding="utf-8").splitlines()
    header, record = json.loads(lines[0]), json.loads(lines[1])
    del record["scene_id"]
    (tmp_path / "selection.jsonl").write_text(
        "\n".join(json.dumps(o) for o in (header, record)) + "\n", encoding="utf-8"
    )
    with pytest.raises(KeyError):
        build_reben_samples(
            manifest_path=tmp_path / "selection.jsonl",
            reben_root=tmp_path / "reben",
            parquet_dir=tmp_path / "tables",
            reader=fake_reader,
        )


# ---------------------------------------------------------------------------
# The no-shift rule
# ---------------------------------------------------------------------------


def test_a_missing_band_is_zero_filled_and_never_shifted(tmp_path: Path) -> None:
    """A shift yields a right-shaped tensor whose channels mean the wrong thing."""
    (sample,) = build(tmp_path, missing_optical_bands=("B03",))

    slot = CANONICAL_OPTICAL_BANDS.index("B03")
    assert sample.optical_mask[slot] == 0.0
    assert np.all(sample.optical[slot] == 0.0), "the missing slot must be zero"
    assert sample.optical_missing_bands == ("B03",)

    # the band AFTER the hole must still be in its own slot, not slid down
    nxt = CANONICAL_OPTICAL_BANDS.index("B04")
    assert sample.optical_mask[nxt] == 1.0
    assert np.all(sample.optical[nxt] == BAND_VALUE["B04"])
    # and nothing from a later slot leaked into the hole
    assert BAND_VALUE["B04"] not in set(np.unique(sample.optical[slot]).tolist())


def test_band_order_is_the_canonical_croma_order(tmp_path: Path) -> None:
    """Each slot must hold the band the canonical order puts there.

    Built with a band MISSING on purpose. With every band present, "pack the
    present bands" and "place each band in its own slot" produce identical
    tensors, so the test would pass for the wrong reason -- verified by
    mutation: without the hole this test does not catch the packing bug.
    """
    (sample,) = build(tmp_path, missing_optical_bands=("B05",))
    assert sample.optical_missing_bands == ("B05",)

    for slot, band in enumerate(CANONICAL_OPTICAL_BANDS):
        if band == "B05":
            assert sample.optical_mask[slot] == 0.0
            continue
        assert sample.optical_mask[slot] == 1.0, f"slot {slot} should hold {band}"
        assert np.all(sample.optical[slot] == BAND_VALUE[band]), (
            f"slot {slot} should hold {band}, not {sample.optical[slot].flat[0]}"
        )
    for slot, pol in enumerate(CANONICAL_SAR_BANDS):
        assert np.all(sample.sar[slot] == BAND_VALUE[pol]), (
            f"SAR slot {slot} should hold {pol}"
        )


def test_b10_is_never_slotted(tmp_path: Path) -> None:
    """B10 is not a canonical channel; a file for it must be ignored, not placed."""
    build_corpus(tmp_path)
    patch = tmp_path / "reben" / "BigEarthNet-S2" / GRANULE_S2 / PATCH_ID
    (patch / f"{PATCH_ID}_B10.tif").write_bytes(b"stub")
    (sample,) = build_reben_samples(
        manifest_path=tmp_path / "selection.jsonl",
        reben_root=tmp_path / "reben",
        parquet_dir=tmp_path / "tables",
        target_shape=TARGET,
        reader=fake_reader,
    )
    assert sample.optical_ignored_bands == ("B10",)
    assert sample.optical_mask.sum() == 12
    assert 99.0 not in set(np.unique(sample.optical).tolist())


# ---------------------------------------------------------------------------
# D5 -- bounded windows
# ---------------------------------------------------------------------------


def _multi(tmp_path: Path, n: int = 5) -> None:
    records = []
    for i in range(n):
        pid = f"{GRANULE_S2}_{i:02d}_10"
        s1n = f"{GRANULE_S1}_{TILE}_{i}_10"
        records.append({
            "sample_id": pid, "source_scene": TILE, "scene_id": BLOCK,
            "split": "train" if i < 3 else "val", "dataset_id": "bigearthnet",
            "labels": ["Urban fabric"], "s1_name": s1n,
            "official_split": "train" if i < 3 else "validation",
        })
    build_corpus(tmp_path, records=records)


def _run(tmp_path: Path, **kwargs):
    return build_reben_samples(
        manifest_path=tmp_path / "selection.jsonl",
        reben_root=tmp_path / "reben",
        parquet_dir=tmp_path / "tables",
        target_shape=TARGET,
        reader=fake_reader,
        **kwargs,
    )


def test_offset_and_limit_give_a_bounded_window(tmp_path: Path) -> None:
    _multi(tmp_path, 5)
    everything = [s.patch_id for s in _run(tmp_path)]
    first = [s.patch_id for s in _run(tmp_path, limit=2)]
    second = [s.patch_id for s in _run(tmp_path, offset=2, limit=2)]
    assert first == everything[:2]
    assert second == everything[2:4]
    # the windows are disjoint and ordered, which is what makes chunked resume safe
    assert set(first) & set(second) == set()


def test_offset_windows_tile_the_whole_corpus(tmp_path: Path) -> None:
    _multi(tmp_path, 5)
    everything = [s.patch_id for s in _run(tmp_path)]
    stitched: list[str] = []
    for off in range(0, 6, 2):
        stitched += [s.patch_id for s in _run(tmp_path, offset=off, limit=2)]
    assert stitched == everything


def test_split_filter_is_applied_before_limit(tmp_path: Path) -> None:
    """A bounded val run must yield val samples, not the first N of train."""
    _multi(tmp_path, 5)
    (sample,) = _run(tmp_path, split="val", limit=1)
    assert sample.official_split == "val"


def test_an_offset_past_the_end_is_refused(tmp_path: Path) -> None:
    _multi(tmp_path, 5)
    with pytest.raises(RebenLayoutError, match="at offset"):
        _run(tmp_path, offset=99)


def test_a_negative_offset_is_refused(tmp_path: Path) -> None:
    _multi(tmp_path, 5)
    with pytest.raises(RebenLayoutError, match="must not be negative"):
        _run(tmp_path, offset=-1)


# ---------------------------------------------------------------------------
# Manifest reading
# ---------------------------------------------------------------------------


def test_header_is_returned_for_provenance(tmp_path: Path) -> None:
    build_corpus(tmp_path)
    header, records = load_selection_manifest(tmp_path / "selection.jsonl")
    assert header["name"] == "synthetic_t2"
    assert header["notes"]["scene_key"] == "T2"
    assert len(records) == 1


def test_a_manifest_without_a_header_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"sample_id": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(RebenLayoutError, match="_manifest"):
        load_selection_manifest(path)


def test_an_empty_manifest_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(RebenLayoutError, match="empty"):
        load_selection_manifest(path)


def test_a_missing_manifest_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RebenLayoutError, match="does not exist"):
        load_selection_manifest(tmp_path / "nope.jsonl")
