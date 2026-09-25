"""Tests for the lazy-image mode of the VRSBench loader.

SOFTWARE CORRECTNESS ONLY. These say nothing about grounding quality.

The mode exists because the full eval split is 16,159 records, and the loader's
default eager `Image.open(...).convert("RGB")` per record holds one PIL object
per record -- roughly 12.7 GB at 512x512 RGB, which does not fit a Kaggle
session. The deciding resolution experiment passes `open_images=False` and
opens one image at a time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from training.data.vrsbench import load_referring_samples

# ---------------------------------------------------------------------------
# Fixture: a minimal verified-eval-schema file
# ---------------------------------------------------------------------------


def write_image(path: Path, size: int = 64) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:size, 0:size]
    plane = ((xx + yy) % 256).astype(np.uint8)
    Image.fromarray(np.stack([plane, plane // 2, 255 - plane], axis=-1)).save(path)
    return path


@pytest.fixture()
def eval_tree(tmp_path: Path) -> Path:
    """A verified-eval-schema file: image_id + question + ground_truth string."""
    root = tmp_path / "vrsbench"
    records = []
    for i in range(4):
        name = f"P{i:04d}_{i:04d}.png"
        write_image(root / "images" / name)
        records.append(
            {
                "image_id": name,
                "question": f"the object number {i}",
                "ground_truth": f"{{<25><25><50><50>}}",
                "question_id": i,
            }
        )
    (root / "VRSBench_EVAL_referring.json").write_text(
        json.dumps(records), encoding="utf-8"
    )
    return root


# ---------------------------------------------------------------------------
# Default behaviour must be unchanged
# ---------------------------------------------------------------------------


def test_default_still_opens_images(eval_tree: Path) -> None:
    """Existing callers and tests rely on this; it must not regress."""
    samples = load_referring_samples(eval_tree, limit=None, verbose=False)
    assert samples
    assert all(s.image is not None for s in samples)


def test_default_populates_image_path_too(eval_tree: Path) -> None:
    samples = load_referring_samples(eval_tree, limit=None, verbose=False)
    assert all(s.image_path is not None for s in samples)
    assert all(s.image_path.exists() for s in samples)


# ---------------------------------------------------------------------------
# Lazy mode
# ---------------------------------------------------------------------------


def test_lazy_mode_leaves_image_none(eval_tree: Path) -> None:
    samples = load_referring_samples(
        eval_tree, limit=None, verbose=False, open_images=False
    )
    assert samples
    assert all(s.image is None for s in samples), (
        "lazy mode still opened images; the full eval split would not fit memory"
    )


def test_lazy_mode_still_populates_image_path(eval_tree: Path) -> None:
    samples = load_referring_samples(
        eval_tree, limit=None, verbose=False, open_images=False
    )
    assert all(s.image_path is not None and s.image_path.exists() for s in samples)


def test_lazy_mode_boxes_match_eager_mode(eval_tree: Path) -> None:
    """The image-loading mode must not change the annotation parsing."""
    eager = load_referring_samples(eval_tree, limit=None, verbose=False)
    lazy = load_referring_samples(
        eval_tree, limit=None, verbose=False, open_images=False
    )
    assert [s.sample_id for s in eager] == [s.sample_id for s in lazy]
    assert [s.box for s in eager] == [s.box for s in lazy]
    assert [s.phrase for s in eager] == [s.phrase for s in lazy]


def test_lazy_mode_returns_the_same_count(eval_tree: Path) -> None:
    lazy = load_referring_samples(
        eval_tree, limit=None, verbose=False, open_images=False
    )
    assert len(lazy) == 4


def test_lazy_sample_opens_on_demand(eval_tree: Path) -> None:
    """What the experiment actually does: open, use, close, one at a time."""
    samples = load_referring_samples(
        eval_tree, limit=None, verbose=False, open_images=False
    )
    for sample in samples:
        image = Image.open(sample.image_path).convert("RGB")
        try:
            assert image.size == (64, 64)
            assert image.mode == "RGB"
        finally:
            image.close()


def test_lazy_mode_does_not_leak_file_handles(eval_tree: Path) -> None:
    """Sequential open/close over every sample must not exhaust handles."""
    samples = load_referring_samples(
        eval_tree, limit=None, verbose=False, open_images=False
    )
    for _ in range(50):
        for sample in samples:
            image = Image.open(sample.image_path)
            image.close()