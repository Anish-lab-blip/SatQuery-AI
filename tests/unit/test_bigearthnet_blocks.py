"""Tests for the T2 spatial-block scene key (Phase 12).

Pure functions only: no parquet, no imagery, no network. Every fixture is a
hand-built grid small enough to reason about by eye, which is the point — the
failures that matter here are geometric (a diagonal wrongly welded into one
block, a component that leaks across a partition boundary), and they are
invisible in a corpus-sized aggregate.

The tests that matter most:

  * `test_diagonal_is_not_connected` — 4-connectivity is the definition of the
    block. If diagonals counted, blocks would grow across the corners of the
    official partition's rings and the key would no longer be the partition's
    own unit.
  * `test_blocks_never_cross_a_partition` — components are computed per
    `(tile, split)`. Two cells that touch on the grid but carry different
    official splits must be two blocks, however adjacent they look.
  * `test_scene_id_does_not_encode_the_split` — the id must not assert the very
    property the downstream disjointness check is supposed to measure.
"""

from __future__ import annotations

import random

import pytest

from training.data.bigearthnet_blocks import (
    BigEarthNetBlockError,
    block_cell_counts,
    count_blocks_by_split,
    parse_acquisition_date,
    parse_ground_cell,
    parse_s1_tile,
    parse_tile,
    reconstruct_blocks,
)

#: A real row from `metadata.parquet`.
PATCH_ID = "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57"
#: The matching Sentinel-1 name: same tile, no leading `T`.
S1_NAME = "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57"
#: `s2v1_name` from the same row: NO tile token at all.
S2V1_NAME = "S2A_MSIL2A_20170613T101031_26_57"


# ---------------------------------------------------------------------------
# Name parsing
# ---------------------------------------------------------------------------


def test_parse_tile_reads_the_mgrs_tile():
    assert parse_tile(PATCH_ID) == "33UUP"


def test_parse_tile_rejects_a_name_with_no_tile_token():
    """`s2v1_name` must fail loudly, never return None or an empty tile.

    A silent empty tile would make every patch its own scene — or every patch
    the same scene — with no visible failure anywhere downstream.
    """
    with pytest.raises(BigEarthNetBlockError):
        parse_tile(S2V1_NAME)


def test_parse_tile_rejects_a_sentinel1_name():
    """S1 names have the token but no `T` marker; they need `parse_s1_tile`."""
    with pytest.raises(BigEarthNetBlockError):
        parse_tile(S1_NAME)


def test_parse_tile_rejects_an_unrelated_string():
    with pytest.raises(BigEarthNetBlockError):
        parse_tile("not-a-patch-id")


def test_parse_s1_tile_reads_the_same_tile_as_patch_id():
    assert parse_s1_tile(S1_NAME) == "33UUP"
    assert parse_s1_tile(S1_NAME) == parse_tile(PATCH_ID)


def test_parse_s1_tile_does_not_mistake_the_timestamp_or_orbit_for_a_tile():
    """`_20170612T165809_` and `_R022_` must not parse as a tile."""
    with pytest.raises(BigEarthNetBlockError):
        parse_s1_tile(S2V1_NAME)


def test_parse_ground_cell_reads_the_row_col_suffix():
    assert parse_ground_cell(PATCH_ID) == (26, 57)


def test_parse_ground_cell_ignores_the_orbit_and_tile_fields():
    """Only the trailing `<row>_<col>` is the cell, not `R022` or the tile."""
    assert parse_ground_cell(
        "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_126_57"
    ) == (126, 57)


def test_parse_ground_cell_raises_without_a_suffix():
    with pytest.raises(BigEarthNetBlockError):
        parse_ground_cell("S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP")


def test_parse_acquisition_date_is_iso():
    assert parse_acquisition_date(PATCH_ID) == "2017-06-13"


def test_parse_acquisition_date_raises_without_a_timestamp():
    with pytest.raises(BigEarthNetBlockError):
        parse_acquisition_date("S2A_MSIL2A_N9999_R022_T33UUP_26_57")


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------


def _grid(rows: list[tuple[int, int]], split: str = "train", tile: str = "33UUP"):
    return [(tile, r, c, split) for r, c in rows]


def test_single_cell_is_one_block():
    mapping = reconstruct_blocks(_grid([(0, 0)]))
    assert list(mapping) == [("33UUP", 0, 0)]
    assert len(set(mapping.values())) == 1


def test_adjacent_cells_merge_into_one_block():
    mapping = reconstruct_blocks(_grid([(0, 0), (0, 1), (1, 0)]))
    assert len(set(mapping.values())) == 1


def test_diagonal_is_not_connected():
    """4-connectivity: a diagonal neighbour is a DIFFERENT block."""
    mapping = reconstruct_blocks(_grid([(0, 0), (1, 1)]))
    assert len(set(mapping.values())) == 2
    assert mapping[("33UUP", 0, 0)] != mapping[("33UUP", 1, 1)]


def test_diagonal_only_chain_is_fragmented():
    """A chain joined only at corners is five blocks, not one."""
    mapping = reconstruct_blocks(_grid([(i, i) for i in range(5)]))
    assert len(set(mapping.values())) == 5


def test_a_hole_does_not_split_a_block():
    """A ring of cells around a missing centre is still one 4-connected block."""
    ring = [
        (0, 0), (0, 1), (0, 2),
        (1, 0),         (1, 2),
        (2, 0), (2, 1), (2, 2),
    ]
    mapping = reconstruct_blocks(_grid(ring))
    assert len(set(mapping.values())) == 1
    assert len(mapping) == 8


def test_fragmented_region_yields_one_block_per_fragment():
    cells = [(0, 0), (0, 1), (5, 5), (5, 6), (5, 7), (20, 20)]
    mapping = reconstruct_blocks(_grid(cells))
    scenes = set(mapping.values())
    assert len(scenes) == 3
    assert block_cell_counts(mapping) == {
        mapping[("33UUP", 0, 0)]: 2,
        mapping[("33UUP", 5, 5)]: 3,
        mapping[("33UUP", 20, 20)]: 1,
    }


# ---------------------------------------------------------------------------
# Partition purity
# ---------------------------------------------------------------------------


def test_blocks_never_cross_a_partition():
    """Two touching cells with different official splits are two blocks.

    This is the whole reason components are computed per `(tile, split)`. If
    they were computed per tile, the train frame and the validation ring would
    be welded into one block that spans two partitions.
    """
    cells = [
        ("33UUP", 0, 0, "train"),
        ("33UUP", 0, 1, "train"),
        ("33UUP", 0, 2, "validation"),
        ("33UUP", 0, 3, "validation"),
    ]
    mapping = reconstruct_blocks(cells)
    assert mapping[("33UUP", 0, 0)] == mapping[("33UUP", 0, 1)]
    assert mapping[("33UUP", 0, 2)] == mapping[("33UUP", 0, 3)]
    assert mapping[("33UUP", 0, 1)] != mapping[("33UUP", 0, 2)]

    cell_splits = {(t, r, c): s for t, r, c, s in cells}
    assert count_blocks_by_split(mapping, cell_splits) == {
        "train": 1,
        "validation": 1,
    }


def test_an_impure_cell_is_refused():
    """One ground location cannot belong to two official splits."""
    cells = [
        ("33UUP", 0, 0, "train"),
        ("33UUP", 0, 0, "test"),
    ]
    with pytest.raises(BigEarthNetBlockError):
        reconstruct_blocks(cells)


def test_blocks_are_tile_scoped():
    """Identical grids in two tiles are two blocks, not one."""
    cells = [
        ("33UUP", 0, 0, "train"),
        ("34VER", 0, 0, "train"),
    ]
    mapping = reconstruct_blocks(cells)
    assert mapping[("33UUP", 0, 0)] != mapping[("34VER", 0, 0)]


# ---------------------------------------------------------------------------
# Scene ids
# ---------------------------------------------------------------------------


def test_scene_id_does_not_encode_the_split():
    """The id must not assert the property the disjointness check measures.

    If the id were `ben_<tile>:<split>:<k>`, "no scene id appears in two
    partitions" would be true by construction of the STRING and would pass even
    if the blocks underneath straddled. The id is `ben_<tile>:<k>` instead, so a
    straddling block WOULD show up as one id in two partitions.
    """
    cells = [
        ("33UUP", 0, 0, "train"),
        ("33UUP", 0, 1, "train"),
        ("33UUP", 5, 5, "validation"),
        ("33UUP", 9, 9, "test"),
    ]
    mapping = reconstruct_blocks(cells)
    for scene_id in mapping.values():
        tile, _, ordinal = scene_id.partition(":")
        assert tile == "ben_33UUP"
        assert ordinal.isdigit()
        for split in ("train", "validation", "test"):
            assert split not in scene_id


def test_each_scene_id_belongs_to_exactly_one_split():
    cells = [
        ("33UUP", 0, 0, "train"),
        ("33UUP", 0, 1, "train"),
        ("33UUP", 5, 5, "validation"),
        ("34VER", 0, 0, "test"),
    ]
    mapping = reconstruct_blocks(cells)
    cell_splits = {(t, r, c): s for t, r, c, s in cells}
    scene_split: dict[str, set[str]] = {}
    for cell, scene in mapping.items():
        scene_split.setdefault(scene, set()).add(cell_splits[cell])
    assert all(len(splits) == 1 for splits in scene_split.values())
    assert len(scene_split) == 3


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_reconstruction_is_deterministic_across_calls():
    cells = _grid([(0, 0), (0, 1), (3, 3), (3, 4), (9, 9)])
    assert reconstruct_blocks(cells) == reconstruct_blocks(cells)


def test_reconstruction_is_independent_of_input_order():
    """The scene ids depend on the SET of cells, never on the caller's order."""
    cells = _grid([(0, 0), (0, 1), (0, 2), (3, 3), (3, 4), (9, 9), (9, 10)])
    baseline = reconstruct_blocks(cells)

    rng = random.Random(1234)
    for _ in range(10):
        shuffled = list(cells)
        rng.shuffle(shuffled)
        assert reconstruct_blocks(shuffled) == baseline


def test_scene_ordinals_are_contiguous_from_zero_per_tile():
    cells = _grid([(0, 0), (5, 5), (9, 9)])
    mapping = reconstruct_blocks(cells)
    ordinals = sorted(
        int(scene.partition(":")[2]) for scene in set(mapping.values())
    )
    assert ordinals == [0, 1, 2]


def test_block_cell_counts_sums_to_the_cell_count():
    cells = _grid([(0, 0), (0, 1), (5, 5), (9, 9), (9, 10)])
    mapping = reconstruct_blocks(cells)
    counts = block_cell_counts(mapping)
    assert sum(counts.values()) == len(cells)
    assert sorted(counts.values()) == [1, 2, 2]


def test_count_blocks_by_split_counts_blocks_not_cells():
    cells = [
        ("33UUP", 0, 0, "train"),
        ("33UUP", 0, 1, "train"),
        ("33UUP", 0, 2, "train"),
        ("33UUP", 5, 5, "test"),
    ]
    mapping = reconstruct_blocks(cells)
    cell_splits = {(t, r, c): s for t, r, c, s in cells}
    assert count_blocks_by_split(mapping, cell_splits) == {"train": 1, "test": 1}
