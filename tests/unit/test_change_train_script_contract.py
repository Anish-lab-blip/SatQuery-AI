"""Phase 9 — contracts for `scripts/train_change.py`.

These tests exist because the training script was written, referenced from
`docs/PHASE8_HANDOFF.md` as the next step, and never actually executed. It
could not be imported at all, which hid three separate defects:

  1. it imported `split_by_image` and `assert_image_disjoint` from
     `training.change.dataset`, where neither existed (they were only ever
     defined, differently typed, in `training/grounding/dataset.py`);
  2. it used `@dataclass` without importing `dataclass`;
  3. its local `change_loss` accepted `bce_weight` / `dice_weight` and then
     passed literal `0.5, 0.5` to the real implementation, so every
     `change.bce_weight` / `change.dice_weight` value in `configs/base.yaml`
     was silently ignored.

Each test below pins one of those. The point is not the specific assertions —
it is that a training script that cannot import must not be able to pass CI.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# 1. The script must be importable
# ---------------------------------------------------------------------------


def test_train_change_script_imports() -> None:
    """The script must import cleanly.

    This is the test that would have caught the missing `split_by_image`
    import (and, once that was fixed, the missing `dataclass` import).
    """
    module = importlib.import_module("scripts.train_change")
    assert module is not None


def test_train_change_exposes_expected_entry_points() -> None:
    module = importlib.import_module("scripts.train_change")
    for name in ("change_loss", "train_change_head", "evaluate"):
        assert hasattr(module, name), f"train_change.py lost {name!r}"


# ---------------------------------------------------------------------------
# 2. Loss weights must actually reach the loss
# ---------------------------------------------------------------------------


def test_change_loss_honours_bce_weight() -> None:
    """The weights must change the TOTAL, which is where they are applied.

    `bce` and `dice` are the raw, unweighted components — they are reported
    so the training loop can log them separately, and they correctly do not
    move when only the weighting changes. `total` is the weighted sum and is
    what must respond.

    Before the fix the parameters were accepted and discarded, so every
    weighting produced a byte-identical `total`. That is what this pins: not
    that a specific number is right, but that the knob is wired at all.
    """
    module = importlib.import_module("scripts.train_change")
    logits = torch.randn(2, 1, 32, 32)
    target = (torch.rand(2, 1, 32, 32) > 0.5).float()

    low = module.change_loss(logits, target, bce_weight=0.25, dice_weight=0.75)
    high = module.change_loss(logits, target, bce_weight=0.75, dice_weight=0.25)

    # the weighted sum must respond to the weighting
    assert float(low.total) != pytest.approx(float(high.total))

    # the raw components are unweighted and must not
    assert float(low.bce) == pytest.approx(float(high.bce))
    assert float(low.dice) == pytest.approx(float(high.dice))

    # and the weighting must actually be a weighting, not an inversion:
    # more BCE weight on a term that is the larger of the two raises the sum.
    if float(low.bce) > float(low.dice):
        assert float(high.total) > float(low.total)


def test_change_loss_defaults_are_the_conventional_half_half() -> None:
    """Defaults stay 0.5/0.5 — the fix forwards the caller's value, it does
    not change what a caller who omits the argument gets."""
    module = importlib.import_module("scripts.train_change")
    logits = torch.randn(2, 1, 16, 16)
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()

    default = module.change_loss(logits, target)
    explicit = module.change_loss(logits, target, bce_weight=0.5, dice_weight=0.5)

    assert float(default.total) == pytest.approx(float(explicit.total))


def test_change_loss_returns_a_breakdown_not_a_bare_tuple() -> None:
    """The annotation used to claim `tuple[Tensor, float, float]` while the
    delegate returns `ChangeLossBreakdown`. Components must stay reachable
    by name, because the training loop logs them separately."""
    module = importlib.import_module("scripts.train_change")
    out = module.change_loss(
        torch.randn(1, 1, 8, 8), torch.zeros(1, 1, 8, 8)
    )
    for field in ("total", "bce", "dice"):
        assert hasattr(out, field), f"ChangeLossBreakdown lost {field!r}"


# ---------------------------------------------------------------------------
# 3. Scene-level splitting (the leakage guard the script relies on)
# ---------------------------------------------------------------------------


def _items(scene: str, crops: int = 3) -> list[dict]:
    return [
        {"sample_id": f"{scene}_{i}", "group_key": scene, "split": "train"}
        for i in range(crops)
    ]


def test_split_by_image_imports_from_change_dataset() -> None:
    """The symbols the training script imports must exist where it imports
    them from — not only in the grounding module."""
    from training.change.dataset import (  # noqa: F401
        assert_image_disjoint,
        split_by_image,
    )


def test_split_keeps_every_crop_of_a_scene_together() -> None:
    """LEVIR-CD ships neighbouring crops of one scene. If the split is by
    patch, near-duplicate crops land on both sides and validation measures
    memorisation rather than generalisation."""
    from training.change.dataset import split_by_image

    items: list[dict] = []
    for scene in ("a", "b", "c", "d", "e", "f"):
        items.extend(_items(scene))

    train, val = split_by_image(items, val_fraction=0.5, seed=0)

    train_scenes = {i["group_key"] for i in train}
    val_scenes = {i["group_key"] for i in val}
    assert not (train_scenes & val_scenes), "a scene straddles the boundary"

    # completeness: no item lost or duplicated
    assert len(train) + len(val) == len(items)
    assert sorted(i["sample_id"] for i in train + val) == sorted(
        i["sample_id"] for i in items
    )


def test_split_is_reproducible_for_a_given_seed() -> None:
    from training.change.dataset import split_by_image

    items: list[dict] = []
    for scene in ("a", "b", "c", "d", "e", "f"):
        items.extend(_items(scene))

    first = split_by_image(items, val_fraction=0.34, seed=7)
    second = split_by_image(items, val_fraction=0.34, seed=7)
    assert [i["sample_id"] for i in first[0]] == [i["sample_id"] for i in second[0]]
    assert [i["sample_id"] for i in first[1]] == [i["sample_id"] for i in second[1]]


def test_assert_image_disjoint_raises_on_a_real_overlap() -> None:
    """The guard must fail loudly, not warn."""
    from training.change.dataset import (
        LEVIRCDError,
        assert_image_disjoint,
    )

    train = _items("shared", 2)
    val = _items("shared", 2)
    with pytest.raises(LEVIRCDError):
        assert_image_disjoint(train, val)


def test_assert_image_disjoint_passes_on_a_clean_split() -> None:
    from training.change.dataset import assert_image_disjoint

    assert_image_disjoint(_items("a"), _items("b"))


def test_split_rejects_an_out_of_range_fraction() -> None:
    from training.change.dataset import LEVIRCDError, split_by_image

    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(LEVIRCDError):
            split_by_image(_items("a", 2), val_fraction=bad, seed=0)


# ---------------------------------------------------------------------------
# 4. LEVIR_SPLITS is defined exactly once
# ---------------------------------------------------------------------------


def test_levir_splits_is_defined_once_and_is_public() -> None:
    """It was defined twice in one module (identically). Two definitions of a
    public constant is a trap even when the values agree, because the next
    reader cannot tell which is authoritative."""
    import inspect

    from training.change import dataset as ds

    source = inspect.getsource(ds)
    assert source.count("LEVIR_SPLITS = {") == 1
    assert "LEVIR_SPLITS" in ds.__all__


def test_levir_splits_reports_scene_counts() -> None:
    """Pins the granularity so nobody 'fixes' this to the patch count.

    `configs/base.yaml` (7120/1024/2048) and this constant (445/64/128)
    describe the same dataset at different granularities — patches after
    256px tiling versus scene image pairs. They are not in conflict.
    """
    from training.change.dataset import LEVIR_SPLITS

    assert LEVIR_SPLITS == {"train": 445, "val": 64, "test": 128}
