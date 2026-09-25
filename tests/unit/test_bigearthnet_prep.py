"""Tests for the BigEarthNet reader and instruction-pair construction.

Everything runs against synthetic patch directories built in a temp dir. No
dataset download, no network, no GPU.

The tests that matter most are the two structural ones:

  * the tile name is the scene key, and one tile's patches never straddle a
    split boundary
  * no generated question asks for a count or a location, because the source
    annotations do not contain either and a model trained on such questions
    learns to invent them
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.errors import LeakageError
from evaluation.leakage import assign_splits_by_scene, audit_manifest
from evaluation.manifests import DatasetManifest, SampleRecord
from training.data.bigearthnet import (
    CLC19_CLASSES,
    BigEarthNetError,
    BigEarthNetPatch,
    build_instruction_pairs,
    discover_patches,
    summarize,
    warnings_for,
)

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------


def make_patch(
    tile_dir: Path,
    patch_name: str,
    labels: list[str],
    modality: str = "s2",
    split: str | None = None,
    bands: int = 12,
) -> Path:
    """Write one BigEarthNet-shaped patch directory."""
    patch_dir = tile_dir / patch_name
    patch_dir.mkdir(parents=True, exist_ok=True)

    tokens = (
        ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
         "B09", "B10", "B11", "B12"][:bands]
        if modality == "s2"
        else ["VV", "VH"][:bands]
    )
    for token in tokens:
        (patch_dir / f"{patch_name}_{token}.tif").write_bytes(b"II*\x00stub")

    metadata: dict[str, object] = {"labels": labels, "patch_id": patch_name}
    if split:
        metadata["split"] = split
    (patch_dir / f"{patch_name}_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return patch_dir


@pytest.fixture()
def ben_tree(tmp_path: Path) -> Path:
    """Four tiles, three to five patches each, with an official split."""
    root = tmp_path / "BigEarthNet-S2"
    root.mkdir()

    layout = {
        "S2A_TILE_1": ["Arable land", "Pastures"],
        "S2A_TILE_2": ["Urban fabric", "Industrial or commercial units"],
        "S2A_TILE_3": ["Broad-leaved forest", "Mixed forest", "Coniferous forest"],
        "S2B_TILE_4": ["Inland waters", "Marine waters"],
    }
    for tile_name, base_labels in layout.items():
        tile_dir = root / tile_name
        tile_dir.mkdir()
        for i in range(4):
            split = "train" if i < 3 else "test"
            make_patch(
                tile_dir,
                f"{tile_name}_patch_{i}",
                labels=base_labels if i % 2 == 0 else base_labels[:1],
                split=split,
            )
    return root


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discovery_finds_every_patch(ben_tree: Path) -> None:
    patches = discover_patches(ben_tree)
    assert len(patches) == 16


def test_discovery_reads_labels(ben_tree: Path) -> None:
    patches = discover_patches(ben_tree)
    assert all(p.labels for p in patches)


def test_discovery_reads_band_paths(ben_tree: Path) -> None:
    patches = discover_patches(ben_tree)
    assert all(p.band_count == 12 for p in patches)


def test_discovery_respects_a_limit(ben_tree: Path) -> None:
    assert len(discover_patches(ben_tree, limit=5)) == 5


def test_discovery_rejects_a_missing_root(tmp_path: Path) -> None:
    with pytest.raises(BigEarthNetError, match="does not exist"):
        discover_patches(tmp_path / "nope")


def test_discovery_rejects_an_empty_tree(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(BigEarthNetError, match="no S2 patches"):
        discover_patches(empty)


def test_discovery_can_skip_patches_without_labels(tmp_path: Path) -> None:
    root = tmp_path / "root"
    tile = root / "TILE_A"
    tile.mkdir(parents=True)
    make_patch(tile, "labelled", ["Arable land"])
    # An unlabelled patch: bands but no labels key.
    patch = tile / "unlabelled"
    patch.mkdir()
    (patch / "unlabelled_B02.tif").write_bytes(b"stub")
    (patch / "unlabelled_metadata.json").write_text("{}", encoding="utf-8")

    assert len(discover_patches(root, require_labels=True)) == 1
    assert len(discover_patches(root, require_labels=False)) == 2


def test_discovery_handles_a_doubly_nested_layout(tmp_path: Path) -> None:
    """Some releases nest patch/patch/. Tile identity must survive that.

    This is the case the first implementation got wrong: it walked up one
    level only when the parent held band files, and in this layout the parent
    holds none, so the patch was returned as its own tile.
    """
    root = tmp_path / "root"
    outer = root / "TILE_X"
    outer.mkdir(parents=True)
    inner = outer / "patch_0" / "patch_0"
    inner.mkdir(parents=True)
    (inner / "patch_0_B02.tif").write_bytes(b"stub")
    (inner / "patch_0_metadata.json").write_text(
        json.dumps({"labels": ["Arable land"]}), encoding="utf-8"
    )

    patches = discover_patches(root)
    assert len(patches) == 1
    assert patches[0].tile == "TILE_X"


def test_discovery_handles_a_split_directory_layout(tmp_path: Path) -> None:
    """<root>/<split>/<tile>/<patch>/ -- the tile is not the split."""
    root = tmp_path / "root"
    for split in ("train", "test"):
        tile = root / split / f"TILE_{split.upper()}"
        tile.mkdir(parents=True)
        make_patch(tile, f"p_{split}", ["Arable land"], split=split)

    patches = discover_patches(root)
    tiles = {p.tile for p in patches}
    assert tiles == {"TILE_TRAIN", "TILE_TEST"}, (
        f"split directories leaked into tile identity: {tiles}"
    )


def test_doubly_nested_patches_sharing_a_tile_share_a_scene(tmp_path: Path) -> None:
    """The leakage boundary must survive the extra nesting level."""
    root = tmp_path / "root"
    tile = root / "TILE_Y"
    for i in range(3):
        inner = tile / f"p{i}" / f"p{i}"
        inner.mkdir(parents=True)
        (inner / f"p{i}_B02.tif").write_bytes(b"stub")
        (inner / f"p{i}_metadata.json").write_text(
            json.dumps({"labels": ["Arable land"]}), encoding="utf-8"
        )

    patches = discover_patches(root)
    assert len(patches) == 3
    assert len({p.scene_id for p in patches}) == 1, (
        "patches from one tile did not share a scene key"
    )


def test_discovery_reads_the_official_split(ben_tree: Path) -> None:
    patches = discover_patches(ben_tree)
    splits = {p.official_split for p in patches}
    assert splits == {"train", "test"}


def test_discovery_of_sar_patches(tmp_path: Path) -> None:
    root = tmp_path / "BigEarthNet-S1"
    tile = root / "S1A_TILE"
    tile.mkdir(parents=True)
    make_patch(tile, "sar_0", ["Inland waters"], modality="s1", bands=2)

    patches = discover_patches(root, modality="s1")
    assert len(patches) == 1
    assert patches[0].band_count == 2


# ---------------------------------------------------------------------------
# Scene identity -- the leakage boundary
# ---------------------------------------------------------------------------


def test_tile_name_is_the_scene_key(ben_tree: Path) -> None:
    patches = discover_patches(ben_tree)
    assert all(p.scene_id.startswith("s2_") for p in patches)
    assert len({p.scene_id for p in patches}) == 4


def test_patches_from_one_tile_share_a_scene(ben_tree: Path) -> None:
    patches = discover_patches(ben_tree)
    by_tile: dict[str, set[str]] = {}
    for p in patches:
        by_tile.setdefault(p.tile, set()).add(p.scene_id)
    assert all(len(scenes) == 1 for scenes in by_tile.values())


def test_scene_splitting_keeps_a_tile_together(ben_tree: Path) -> None:
    """The property the whole manifest exists for."""
    patches = discover_patches(ben_tree)
    records = [
        SampleRecord(
            dataset_id="bigearthnet",
            sample_id=p.patch_id,
            split="train",
            scene_id=p.scene_id,
            sensor="sentinel-2",
            sha256=p.patch_id.ljust(64, "0")[:64],
        )
        for p in patches
    ]
    assigned = assign_splits_by_scene(records, seed=42)

    scene_splits: dict[str, set[str]] = {}
    for record in assigned:
        scene_splits.setdefault(record.scene_key, set()).add(record.split)
    assert all(len(s) == 1 for s in scene_splits.values())


def test_manifest_audit_passes_on_a_real_bigearthnet_layout(ben_tree: Path) -> None:
    patches = discover_patches(ben_tree)
    records = [
        SampleRecord(
            dataset_id="bigearthnet",
            sample_id=p.patch_id,
            split="train",
            scene_id=p.scene_id,
            sensor="sentinel-2",
            sha256=p.patch_id.ljust(64, "0")[:64],
        )
        for p in patches
    ]
    manifest = DatasetManifest.from_records(
        "bigearthnet", assign_splits_by_scene(records, seed=7)
    )
    report = audit_manifest(manifest)
    assert report.clean is True


def test_a_random_patch_split_is_caught_as_leakage(ben_tree: Path) -> None:
    """The adversarial case: split by PATCH instead of by tile."""
    patches = discover_patches(ben_tree)
    records = []
    for i, p in enumerate(patches):
        records.append(
            SampleRecord(
                dataset_id="bigearthnet",
                sample_id=p.patch_id,
                split="test" if i % 2 else "train",  # type: ignore[arg-type]
                scene_id=p.scene_id,
                sha256=p.patch_id.ljust(64, "0")[:64],
            )
        )

    manifest = DatasetManifest.from_records("leaky", records)
    with pytest.raises(LeakageError):
        audit_manifest(manifest, require_all_splits=False)


# ---------------------------------------------------------------------------
# Instruction pairs -- the honesty constraint
# ---------------------------------------------------------------------------


def _patch(labels: list[str], patch_id: str = "p0") -> BigEarthNetPatch:
    return BigEarthNetPatch(
        patch_id=patch_id,
        tile="TILE",
        modality="s2",
        directory=Path("/nonexistent"),
        labels=tuple(labels),
    )


def test_pairs_are_generated_for_a_labelled_patch() -> None:
    pairs = build_instruction_pairs(_patch(["Arable land", "Pastures"]))
    assert len(pairs) > 0
    assert all({"question", "answer"} == set(p) for p in pairs)


def test_pairs_include_a_multi_label_enumeration() -> None:
    pairs = build_instruction_pairs(_patch(["Arable land", "Pastures"]))
    answers = " ".join(p["answer"] for p in pairs)
    assert "Arable land" in answers
    assert "Pastures" in answers


def test_pairs_include_both_positive_and_negative_presence() -> None:
    """A corpus of only positives teaches the model to answer 'yes' always."""
    pairs = build_instruction_pairs(
        _patch(["Arable land", "Pastures", "Broad-leaved forest"]),
        negative_ratio=1.0,
    )
    answers = {p["answer"] for p in pairs}
    assert "Yes." in answers
    assert "No." in answers


def test_no_question_asks_for_a_count() -> None:
    """Plan section 44: do not manufacture counts the labels do not support.

    BigEarthNet labels are presence labels. A "how many" question has no
    correct answer in the data, so training on one teaches hallucination.
    """
    pairs = build_instruction_pairs(_patch(["Arable land", "Pastures"]))
    for pair in pairs:
        lowered = pair["question"].lower()
        assert "how many" not in lowered
        assert "count" not in lowered
        assert "number of" not in lowered


def test_no_question_asks_for_a_location() -> None:
    """Grounding is a specialist, not a language task (finding C-5)."""
    pairs = build_instruction_pairs(_patch(["Arable land", "Pastures"]))
    for pair in pairs:
        lowered = pair["question"].lower()
        assert "where" not in lowered
        assert "locate" not in lowered
        assert "coordinate" not in lowered
        assert "bounding box" not in lowered


def test_dominance_answer_does_not_invent_an_area_measurement() -> None:
    """The annotation has no per-class area, so no answer may claim one."""
    pairs = build_instruction_pairs(_patch(["Arable land", "Pastures"]))
    dominance = [
        p for p in pairs if "dominant" in p["question"].lower()
        or "most of this patch" in p["question"].lower()
    ]
    assert dominance, "no dominance question was generated"
    for pair in dominance:
        assert "does not specify" in pair["answer"], (
            "a dominance answer claimed a spatial measurement the "
            "annotation does not contain"
        )


def test_a_patch_with_no_labels_produces_no_pairs() -> None:
    assert build_instruction_pairs(_patch([])) == []


def test_generation_is_deterministic_for_a_patch() -> None:
    patch = _patch(["Arable land", "Pastures"], "stable")
    first = build_instruction_pairs(patch, seed=1)
    second = build_instruction_pairs(patch, seed=1)
    assert first == second


def test_generation_differs_across_patches() -> None:
    a = build_instruction_pairs(_patch(["Arable land", "Pastures"], "p1"), seed=1)
    b = build_instruction_pairs(_patch(["Arable land", "Pastures"], "p2"), seed=1)
    assert a != b


def test_single_label_patch_reads_naturally() -> None:
    pairs = build_instruction_pairs(_patch(["Inland waters"]))
    enumeration = next(
        p for p in pairs if "visible" in p["question"] or "List" in p["question"]
        or "Describe" in p["question"]
    )
    assert enumeration["answer"] == "Inland waters"


def test_two_labels_are_joined_with_and() -> None:
    pairs = build_instruction_pairs(_patch(["Arable land", "Pastures"]))
    enumeration = next(
        p for p in pairs if "visible" in p["question"] or "List" in p["question"]
        or "Describe" in p["question"]
    )
    assert enumeration["answer"] == "Arable land and Pastures"


def test_unknown_labels_are_excluded_from_the_vocabulary_index() -> None:
    patch = _patch(["Arable land", "A class from a future release"])
    indices = patch.label_indices(CLC19_CLASSES)
    assert len(indices) == 1


# ---------------------------------------------------------------------------
# Corpus summary
# ---------------------------------------------------------------------------


def test_summary_counts_patches_and_tiles(ben_tree: Path) -> None:
    summary = summarize(discover_patches(ben_tree))
    assert summary["patches"] == 16
    assert summary["tiles"] == 4
    assert summary["patches_per_tile_min"] == 4


def test_summary_reports_official_splits(ben_tree: Path) -> None:
    summary = summarize(discover_patches(ben_tree))
    assert summary["official_splits"] == {"train": 12, "test": 4}


def test_summary_of_an_empty_corpus() -> None:
    assert summarize([]) == {"patches": 0}


def test_warnings_are_empty_for_a_healthy_corpus(ben_tree: Path) -> None:
    summary = summarize(discover_patches(ben_tree))
    problems = warnings_for(summary)
    assert [p for p in problems if "one patch" in p] == []


def test_warning_fires_when_every_tile_holds_one_patch(tmp_path: Path) -> None:
    """The failure mode that makes scene splitting vacuous.

    Three tiles, one patch each. An earlier version of this fixture put the
    bands directly in the tile directory, which collapsed all three into a
    single tile -- so the corpus had one tile with three patches and the
    warning correctly did not fire. The fixture was wrong, not the check.
    """
    root = tmp_path / "root"
    for i in range(3):
        tile = root / f"TILE_{i}"
        make_patch(tile, f"patch_{i}", ["Arable land"])

    summary = summarize(discover_patches(root))
    assert summary["tiles"] == 3
    assert summary["tiles_with_one_patch"] == 3

    problems = warnings_for(summary)
    assert any("one patch" in p for p in problems), (
        f"the vacuous-split warning did not fire: {problems}"
    )


def test_warning_does_not_fire_for_multiple_patches_per_tile(
    ben_tree: Path,
) -> None:
    """The negative case: the check must not fire on healthy data."""
    problems = warnings_for(summarize(discover_patches(ben_tree)))
    assert not any("one patch" in p for p in problems)


def test_resolve_tile_dir_picks_the_highest_non_split_ancestor(
    tmp_path: Path,
) -> None:
    """Direct unit test of the resolver, independent of discovery."""
    from training.data.bigearthnet import _resolve_tile_dir

    base = tmp_path / "root"
    tile = base / "TILE_A"
    nested = tile / "p0" / "p0"
    nested.mkdir(parents=True)

    assert _resolve_tile_dir(nested, base) == tile
    assert _resolve_tile_dir(tile / "p0", base) == tile
    assert _resolve_tile_dir(tile, base) == tile


def test_resolve_tile_dir_skips_a_split_directory(tmp_path: Path) -> None:
    from training.data.bigearthnet import _resolve_tile_dir

    base = tmp_path / "root"
    tile = base / "train" / "TILE_B"
    patch = tile / "p0"
    patch.mkdir(parents=True)

    assert _resolve_tile_dir(patch, base) == tile


def test_warning_fires_when_no_official_split_exists(tmp_path: Path) -> None:
    root = tmp_path / "root"
    tile = root / "TILE"
    tile.mkdir(parents=True)
    make_patch(tile, "p0", ["Arable land"], split=None)

    problems = warnings_for(summarize(discover_patches(root)))
    assert any("no official split" in p for p in problems)


def test_warning_fires_for_an_incomplete_official_split(tmp_path: Path) -> None:
    root = tmp_path / "root"
    tile = root / "TILE"
    tile.mkdir(parents=True)
    make_patch(tile, "p0", ["Arable land"], split="train")
    make_patch(tile, "p1", ["Pastures"], split="train")

    problems = warnings_for(summarize(discover_patches(root)))
    assert any("incomplete" in p for p in problems)