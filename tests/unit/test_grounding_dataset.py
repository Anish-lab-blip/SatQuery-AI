"""Tests for the Phase 8 grounding dataset.

The load-bearing test is `test_split_never_puts_an_image_in_both_partitions`.
VRSBench train has 36,313 referring expressions across 20,262 images, so several
refers share one photograph. A record-level split would put two expressions for
the same image in different partitions — the same failure mode as splitting
tiles from one scene, and the model would be validated on an image it trained
on. The guard mirrors `evaluation.leakage.assign_splits_by_scene` deliberately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from training.grounding.dataset import (
    GroundingDataError,
    GroundingItem,
    assert_image_disjoint,
    attach_cached,
    image_cache_path,
    split_by_image,
    text_cache_path,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIM = 512
GRID = 7


def make_items(n_images: int = 20, per_image: int = 3) -> list[GroundingItem]:
    """Items where several refers share one image — the realistic shape."""
    items: list[GroundingItem] = []
    for img in range(n_images):
        key = f"img_{img:04d}"
        for j in range(per_image):
            items.append(
                GroundingItem(
                    sample_id=f"{key}_ref_{j}",
                    image_key=key,
                    phrase=f"object number {j}",
                    box=[0.2, 0.2, 0.6, 0.6],
                )
            )
    return items


# ---------------------------------------------------------------------------
# Split by image — the leakage guard
# ---------------------------------------------------------------------------

def test_split_never_puts_an_image_in_both_partitions() -> None:
    """The property the whole module exists for."""
    train, val = split_by_image(make_items(n_images=30), val_fraction=0.2, seed=42)
    train_keys = {i.image_key for i in train}
    val_keys = {i.image_key for i in val}
    assert train_keys & val_keys == set(), (
        f"image leaked across the split: {sorted(train_keys & val_keys)[:5]}"
    )


def test_all_refers_of_one_image_stay_together() -> None:
    """A photograph's referring expressions are one indivisible group."""
    items = make_items(n_images=10, per_image=4)
    train, val = split_by_image(items, val_fraction=0.3, seed=7)

    for partition in (train, val):
        by_image: dict[str, int] = {}
        for item in partition:
            by_image[item.image_key] = by_image.get(item.image_key, 0) + 1
        # Every image in a partition has ALL of its refers there.
        assert all(count == 4 for count in by_image.values())


def test_split_covers_every_item_exactly_once() -> None:
    items = make_items(n_images=15, per_image=3)
    train, val = split_by_image(items, val_fraction=0.2, seed=1)
    assert len(train) + len(val) == len(items)
    ids = [i.sample_id for i in train] + [i.sample_id for i in val]
    assert sorted(ids) == sorted(i.sample_id for i in items)


def test_split_is_deterministic_for_a_seed() -> None:
    a_train, a_val = split_by_image(make_items(), val_fraction=0.25, seed=3)
    b_train, b_val = split_by_image(make_items(), val_fraction=0.25, seed=3)
    assert [i.sample_id for i in a_train] == [i.sample_id for i in b_train]
    assert [i.sample_id for i in a_val] == [i.sample_id for i in b_val]


def test_split_differs_across_seeds() -> None:
    def assignment(seed: int) -> dict[str, str]:
        train, val = split_by_image(make_items(n_images=40), val_fraction=0.25, seed=seed)
        return {i.image_key: "train" for i in train} | {
            i.image_key: "val" for i in val
        }

    assert assignment(1) != assignment(999)


def test_split_emits_records_in_sorted_image_order() -> None:
    """Within a partition, records are emitted sorted by image key.

    The seed chooses WHICH image lands in which partition. It must NOT also
    choose the order of the returned list, or downstream code taking
    `train[:N]` would depend on the seed in a way nobody would notice.

    An earlier version of this test asserted the train LIST was identical
    across seeds. That is unachievable and was a bad test: different seeds
    select different images, so the lists legitimately differ in CONTENT, not
    just order. Measured failure: index 15 was 'img_0005' at seed 1 and
    'img_0009' at seed 999. The real invariant is canonical order per
    partition, which is what this asserts.
    """
    for seed in (1, 7, 42, 999):
        train, val = split_by_image(
            make_items(n_images=25), val_fraction=0.2, seed=seed
        )
        train_keys = [i.image_key for i in train]
        val_keys = [i.image_key for i in val]
        assert train_keys == sorted(train_keys), f"train not sorted at seed={seed}"
        assert val_keys == sorted(val_keys), f"val not sorted at seed={seed}"


def test_same_seed_gives_byte_identical_partitions() -> None:
    """The reproducibility property that actually matters.

    Same seed, same corpus -> identical partitions in identical order. This is
    what makes a resumed or reproduced run comparable to the original.
    """
    a_train, a_val = split_by_image(make_items(n_images=25), val_fraction=0.2, seed=5)
    b_train, b_val = split_by_image(make_items(n_images=25), val_fraction=0.2, seed=5)
    assert [i.sample_id for i in a_train] == [i.sample_id for i in b_train]
    assert [i.sample_id for i in a_val] == [i.sample_id for i in b_val]


def test_split_rejects_bad_val_fraction() -> None:
    with pytest.raises(GroundingDataError, match="val_fraction"):
        split_by_image(make_items(), val_fraction=0.0)
    with pytest.raises(GroundingDataError, match="val_fraction"):
        split_by_image(make_items(), val_fraction=1.0)


def test_split_rejects_a_corpus_too_small_to_split() -> None:
    """One image cannot produce both a train and a val partition."""
    with pytest.raises(GroundingDataError, match="empty partition"):
        split_by_image(make_items(n_images=1), val_fraction=0.5)


# ---------------------------------------------------------------------------
# The explicit disjointness assertion
# ---------------------------------------------------------------------------

def test_assert_image_disjoint_passes_for_a_clean_split() -> None:
    train, val = split_by_image(make_items(n_images=20), val_fraction=0.25, seed=5)
    assert_image_disjoint(train, val)  # must not raise


def test_assert_image_disjoint_fires_on_overlap() -> None:
    items = make_items(n_images=6)
    with pytest.raises(GroundingDataError, match="appear in both"):
        assert_image_disjoint(items, items)


# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

def test_image_cache_path_is_deterministic(tmp_path: Path) -> None:
    a = image_cache_path(tmp_path, "img_0001")
    b = image_cache_path(tmp_path, "img_0001")
    assert a == b
    assert a.suffix == ".npz"


def test_image_cache_path_differs_per_key(tmp_path: Path) -> None:
    assert image_cache_path(tmp_path, "a") != image_cache_path(tmp_path, "b")


def test_text_cache_paths_live_in_a_subdirectory(tmp_path: Path) -> None:
    p = text_cache_path(tmp_path, "the water body")
    assert p.parent.name == "text"
    assert p.suffix == ".npy"


def test_cache_paths_are_filesystem_safe(tmp_path: Path) -> None:
    """A phrase with slashes and colons must not escape the cache dir."""
    p = text_cache_path(tmp_path, "a/b\\c:d*e?f")
    assert p.parent.parent == tmp_path


# ---------------------------------------------------------------------------
# attach_cached
# ---------------------------------------------------------------------------

def _write_cache_entry(cache_dir: Path, key: str, phrase: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "text").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        image_cache_path(cache_dir, key),
        patches=np.zeros((GRID * GRID, DIM), dtype=np.float16),
        cls=np.zeros(DIM, dtype=np.float16),
    )
    np.save(text_cache_path(cache_dir, phrase), np.zeros(DIM, dtype=np.float16))


def test_attach_cached_keeps_items_with_both_entries(tmp_path: Path) -> None:
    _write_cache_entry(tmp_path, "img_a", "a thing")
    items = [
        GroundingItem(
            sample_id="s", image_key="img_a", phrase="a thing", box=[0, 0, 1, 1]
        )
    ]
    kept = attach_cached(items, tmp_path, verbose=False)
    assert len(kept) == 1
    assert kept[0].patches_path is not None
    assert kept[0].text is not None


def test_attach_cached_drops_items_with_a_missing_image(tmp_path: Path) -> None:
    """Dropped, never zero-filled — zero patches would train on blank input."""
    _write_cache_entry(tmp_path, "img_a", "a thing")
    items = [
        GroundingItem(sample_id="ok", image_key="img_a", phrase="a thing",
                      box=[0, 0, 1, 1]),
        GroundingItem(sample_id="gone", image_key="img_missing", phrase="a thing",
                      box=[0, 0, 1, 1]),
    ]
    kept = attach_cached(items, tmp_path, verbose=False)
    assert [i.sample_id for i in kept] == ["ok"]


def test_attach_cached_drops_items_with_a_missing_phrase(tmp_path: Path) -> None:
    _write_cache_entry(tmp_path, "img_a", "a thing")
    items = [
        GroundingItem(sample_id="s", image_key="img_a", phrase="different phrase",
                      box=[0, 0, 1, 1])
    ]
    assert attach_cached(items, tmp_path, verbose=False) == []


def test_load_raises_when_text_is_missing() -> None:
    item = GroundingItem(
        sample_id="s", image_key="k", phrase="p", box=[0, 0, 1, 1],
        patches=np.zeros((1, 1), dtype=np.float32),
    )
    with pytest.raises(GroundingDataError, match="no text embedding"):
        item.load()