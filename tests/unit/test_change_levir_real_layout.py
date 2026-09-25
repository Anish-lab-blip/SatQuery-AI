"""Phase 9 — the LEVIR-CD loader against the REAL dataset layout.

WHY THIS FILE EXISTS
--------------------
Every Phase 9 test before this one used a fixture shaped `<root>/train/{A,B,label}`.
That shape is real (it is the official 1024px distribution), but it is NOT the
shape of `keykeylv/levir-cd-256`, which is the dataset `configs/base.yaml`
actually names (`change.levir_split: 7120/1024/2048` patches).

The real archive was inspected directly -- a partially-downloaded 519 MB prefix
of the 2.33 GB zip, whose local file headers survive truncation. It contains:

    A/test_100_1.png ... A/test_100_16.png     (16 tiles per scene)
    A/train_216_1.png  ...

i.e. a FLAT tree with the split as a FILENAME PREFIX. Against that real shape:

    discover_levir_dataset(root)  ->  {}      (0 pairs)
    load_levir_dataset(root)      ->  0 items

Both were measured, not assumed. The loader was blind to the dataset it was
written for, and no test could see it because no test used the real shape.

The second half of this file is the consequence that matters more. In the flat
layout a scene spans 16 files. A loader that keys scenes by `sample_id` gives
every tile its own scene, so `assert_image_disjoint` can never fire and the
scene-disjoint split silently becomes a per-tile split -- 15/16 of every scene
crosses the boundary while the guard reports success. That is a leakage bug that
looks exactly like a passing test.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: The real naming convention, taken from the archive itself.
FLAT_NAME = "{split}_{scene}_{tile}.png"


def _write_png(path: Path, *, rgb: int, label: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if label:
        arr = np.full((16, 16), rgb, dtype="uint8")
        Image.fromarray(arr, mode="L").save(path)
    else:
        Image.fromarray(np.full((16, 16, 3), rgb, dtype="uint8")).save(path)


def build_flat_root(
    root: Path,
    *,
    splits: dict[str, int],
    tiles_per_scene: int = 16,
) -> Path:
    """Build a FLAT LEVIR-CD root using the real archive's naming.

    `splits` maps split name -> number of SCENES, and scene numbering starts at
    100 so the names match the archive's own (`test_100_1.png`).
    """
    for sub in ("A", "B", "label"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for split, n_scenes in splits.items():
        for scene_offset in range(n_scenes):
            scene = 100 + scene_offset
            for tile in range(1, tiles_per_scene + 1):
                name = FLAT_NAME.format(split=split, scene=scene, tile=tile)
                base = (scene * 3 + tile) % 200
                _write_png(root / "A" / name, rgb=base)
                _write_png(root / "B" / name, rgb=base + 20)
                # Half the tiles carry change, half do not -- so a binarisation
                # bug that collapses the mask is visible downstream.
                _write_png(root / "label" / name, rgb=255 if tile % 2 else 0, label=True)
    return root


# ---------------------------------------------------------------------------
# 1. The flat layout is detected and loaded at all
# ---------------------------------------------------------------------------


def test_flat_layout_is_detected(tmp_path: Path) -> None:
    from training.change.dataset import LAYOUT_FLAT, detect_levir_layout

    root = build_flat_root(tmp_path / "levir", splits={"train": 2})
    assert detect_levir_layout(root) == LAYOUT_FLAT


def test_flat_layout_is_loaded_not_silently_empty(tmp_path: Path) -> None:
    """The measured regression: this returned 0 items before the fix."""
    from training.change.dataset import load_levir_dataset

    root = build_flat_root(tmp_path / "levir", splits={"train": 2})
    items = load_levir_dataset(root)

    assert len(items) == 2 * 16, (
        f"flat layout must yield scenes x tiles items, got {len(items)}"
    )
    assert all(Path(i["t1_path"]).exists() for i in items)
    assert all(Path(i["t2_path"]).exists() for i in items)
    assert all(Path(i["label_path"]).exists() for i in items)


def test_flat_layout_routes_files_to_the_right_split(tmp_path: Path) -> None:
    """The split is a filename prefix here, so it must be parsed, not guessed."""
    from training.change.dataset import load_levir_dataset

    root = build_flat_root(tmp_path / "levir", splits={"train": 2, "val": 1, "test": 1})
    for split, expected_scenes in (("train", 2), ("val", 1), ("test", 1)):
        items = load_levir_dataset(root, splits=(split,))
        assert len(items) == expected_scenes * 16, split
        assert {i["split"] for i in items} == {split}


# ---------------------------------------------------------------------------
# 2. A scene is 16 tiles, not 16 scenes
# ---------------------------------------------------------------------------


def test_flat_layout_groups_16_tiles_into_one_scene(tmp_path: Path) -> None:
    """The heart of it. Keying by tile makes the leakage guard vacuous."""
    from training.change.dataset import load_levir_dataset

    root = build_flat_root(tmp_path / "levir", splits={"train": 2})
    items = load_levir_dataset(root)

    scenes = {i["group_key"] for i in items}
    assert scenes == {"train_100", "train_101"}, (
        f"expected 2 scenes, got {len(scenes)}: {sorted(scenes)[:5]}"
    )


def test_flat_scene_key_drops_only_the_tile_index(tmp_path: Path) -> None:
    from training.change.dataset import flat_stem_parts

    assert flat_stem_parts("test_100_1") == ("test", "test_100")
    assert flat_stem_parts("train_216_16") == ("train", "train_216")
    # Not LEVIR names -- must refuse rather than invent a split.
    for bad in ("scene00", "test_100", "test_100_x", "other_100_1"):
        assert flat_stem_parts(bad) is None, bad


def test_nested_layout_still_works(tmp_path: Path) -> None:
    """Regression: the official layout must keep loading exactly as before."""
    from training.change.dataset import LAYOUT_NESTED, detect_levir_layout, load_levir_dataset

    root = tmp_path / "levir"
    for index in range(4):
        stem = f"scene{index:02d}"
        _write_png(root / "train" / "A" / f"{stem}.png", rgb=10 + index)
        _write_png(root / "train" / "B" / f"{stem}.png", rgb=50 + index)
        _write_png(root / "train" / "label" / f"{stem}.png", rgb=255, label=True)

    assert detect_levir_layout(root) == LAYOUT_NESTED
    items = load_levir_dataset(root)
    assert len(items) == 4
    # Nested: each file IS a scene.
    assert {i["group_key"] for i in items} == {
        "scene00", "scene01", "scene02", "scene03"
    }


# ---------------------------------------------------------------------------
# 3. The leakage guard must be able to FAIL on the flat layout
# ---------------------------------------------------------------------------


def test_scene_disjoint_split_never_straddles_a_flat_scene(tmp_path: Path) -> None:
    from training.change.dataset import (
        assert_image_disjoint,
        load_levir_dataset,
        split_by_image,
    )

    root = build_flat_root(tmp_path / "levir", splits={"train": 4})
    items = load_levir_dataset(root)
    train, val = split_by_image(items, val_fraction=0.5, seed=0)
    assert_image_disjoint(train, val)  # raises on overlap

    assert len(train) + len(val) == len(items), "an item was lost or duplicated"


def test_the_guard_fires_when_a_flat_scene_is_split_by_tile(tmp_path: Path) -> None:
    """MUTATION TEST of the guard itself.

    Take ONE scene's 16 tiles and cut them down the middle. That is a genuine
    straddle, and the guard must raise.

    Then break the scene key the way the old loader did -- key by `sample_id`,
    i.e. by tile -- and assert the guard goes blind. If the second assertion ever
    stops holding, the guard has been re-armed and this test should be revisited;
    if the FIRST ever stops holding, the guard is decorative.
    """
    from training.change.dataset import LEVIRCDError, assert_image_disjoint, load_levir_dataset

    root = build_flat_root(tmp_path / "levir", splits={"train": 2})
    items = load_levir_dataset(root)

    one_scene = [i for i in items if i["group_key"] == "train_100"]
    assert len(one_scene) == 16
    left, right = one_scene[:8], one_scene[8:]

    # Correct scene key: the guard sees the leak.
    with pytest.raises(LEVIRCDError):
        assert_image_disjoint(left, right)

    # Broken key (the pre-fix behaviour): every tile is its own scene, so the
    # same leak is invisible. This is exactly how the bug hid.
    broken_left = [dict(i, group_key=i["sample_id"]) for i in left]
    broken_right = [dict(i, group_key=i["sample_id"]) for i in right]
    assert_image_disjoint(broken_left, broken_right)


# ---------------------------------------------------------------------------
# 4. The test-split firewall
# ---------------------------------------------------------------------------


def _run_train_change(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "train_change.py"), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_train_change_never_loads_the_test_split(tmp_path: Path) -> None:
    """The training script used to load train+val+test and re-split all of them.

    A public-test firewall violation that no fixture could expose, because the
    only fixture ever used shipped a `train/` directory and nothing else.
    """
    root = build_flat_root(tmp_path / "levir", splits={"train": 2, "val": 1, "test": 1})
    out_dir = tmp_path / "out"

    result = _run_train_change(
        "--data-root", str(root), "--output-dir", str(out_dir), "--dry-run"
    )
    combined = result.stdout + result.stderr

    assert "Traceback" not in combined, combined
    assert result.returncode == 0, combined
    assert "test split      : not loaded" in combined, combined
    # 2 train scenes x 16 tiles; the test scene must not inflate this.
    assert "train items     : 32" in combined, combined
    assert "test_" not in combined.split("DATASET")[-1].split("DRY RUN")[0], combined


def test_train_change_uses_the_dataset_val_split_when_present(tmp_path: Path) -> None:
    """With a real `val` split, the frozen 445/64/128 contract must be honoured
    rather than re-carved -- otherwise the val number is not comparable."""
    root = build_flat_root(tmp_path / "levir", splits={"train": 2, "val": 1})
    result = _run_train_change("--data-root", str(root), "--dry-run")
    combined = result.stdout + result.stderr

    assert result.returncode == 0, combined
    assert "val items       : 16" in combined, combined
    assert "the dataset's own val split" in combined, combined


def test_train_change_carves_val_when_the_root_has_only_train(tmp_path: Path) -> None:
    """The single-split case still works, and says so."""
    root = build_flat_root(tmp_path / "levir", splits={"train": 4})
    result = _run_train_change("--data-root", str(root), "--dry-run")
    combined = result.stdout + result.stderr

    assert result.returncode == 0, combined
    assert "carved from train" in combined, combined


# ---------------------------------------------------------------------------
# 5. The dataset's OWN split declaration beats our filename inference
# ---------------------------------------------------------------------------


def _write_split_list(root: Path, split: str, stems: list[str]) -> None:
    (root / "list").mkdir(parents=True, exist_ok=True)
    (root / "list" / f"{split}.txt").write_text(
        "\n".join(f"{s}.png" for s in stems), encoding="utf-8"
    )


def test_declared_split_list_is_preferred_over_the_filename_prefix(
    tmp_path: Path,
) -> None:
    """`keykeylv/levir-cd-256` ships `list/train.txt` (7120), `list/val.txt`
    (1024) and `list/test.txt` (2048) -- exactly the frozen
    `change.levir_split`. That is the dataset declaring its own split, and it
    should outrank a filename convention that merely happens to agree."""
    from training.change.dataset import load_levir_dataset, read_split_list

    root = build_flat_root(tmp_path / "levir", splits={"train": 2, "test": 2})
    # Declare only ONE of the two train scenes, and say so.
    declared = [f"train_100_{t}" for t in range(1, 17)]
    _write_split_list(root, "train", declared)

    assert len(read_split_list(root, "train")) == 16
    items = load_levir_dataset(root, splits=("train",))
    assert len(items) == 16, "the declared list must win over the prefix scan"
    assert {i["group_key"] for i in items} == {"train_100"}


def test_split_list_still_groups_tiles_into_scenes(tmp_path: Path) -> None:
    """The scene key must survive the list path too -- a list-sourced stem that
    lost its scene grouping would make the leakage guard vacuous again."""
    from training.change.dataset import load_levir_dataset

    root = build_flat_root(tmp_path / "levir", splits={"train": 2})
    _write_split_list(root, "train", [f"train_101_{t}" for t in range(1, 17)])

    items = load_levir_dataset(root, splits=("train",))
    assert {i["group_key"] for i in items} == {"train_101"}


def test_absent_split_list_falls_back_to_the_filename_prefix(tmp_path: Path) -> None:
    """No `list/` directory is not an empty split."""
    from training.change.dataset import load_levir_dataset, read_split_list

    root = build_flat_root(tmp_path / "levir", splits={"train": 2})
    assert read_split_list(root, "train") is None
    assert len(load_levir_dataset(root, splits=("train",))) == 32


def test_verify_split_lists_reports_agreement(tmp_path: Path) -> None:
    from training.change.dataset import verify_split_lists

    root = build_flat_root(tmp_path / "levir", splits={"train": 2, "test": 1})
    _write_split_list(root, "train", [f"train_100_{t}" for t in range(1, 17)])
    _write_split_list(root, "test", [f"test_100_{t}" for t in range(1, 17)])

    report = verify_split_lists(root)
    assert report["train"]["agreement"] == 16
    assert report["train"]["declared_only"] == 0
    # The prefix scan sees both train scenes; the list names one.
    assert report["train"]["derived_only"] == 16
    assert report["test"]["declared_only"] == 0
    assert report["test"]["derived_only"] == 0
