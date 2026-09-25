"""Unit tests for grounding NMS (specialists/grounding/head.py::nms).

WHY THIS FILE EXISTS
--------------------
§79 GROUNDING lists non-maximum suppression as part of the grounding pipeline
(plan line ~4351). The behaviour is implemented in
`specialists/grounding/head.py:294` and consumed at
`specialists/grounding/specialist.py:302`, but it had NO dedicated test: a grep
for `GroundingHead|enforce_order|decode_cell_relative` across tests/ returned
zero NMS hits. That absence is the entire reason the bullet was graded PARTIAL.

These tests pin the SPECIFIED contract of `nms`, not a snapshot of whatever the
current code emits.

CONTRACT UNDER TEST
-------------------
    nms(boxes, scores, iou_threshold=0.5) -> LongTensor of kept indices, best first

    * greedy: the highest-scoring box is taken first, then each remaining box is
      taken in descending score order;
    * a candidate is DROPPED when its IoU with an already-kept box is STRICTLY
      greater than `iou_threshold` (equality is KEPT: `ious <= iou_threshold`);
    * the return value is an index tensor into the INPUT array, ordered by
      descending score;
    * an empty input returns an empty LongTensor without raising.

All boxes are synthetic. Nothing here loads a model, a dataset, CROMA, or the
GPU; the only tensors are the handful of coordinates defined inline.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from specialists.grounding.head import nms

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _boxes(*rows: list[float]) -> torch.Tensor:
    """(N, 4) float32 normalized xyxy from row literals."""
    return torch.tensor(rows, dtype=torch.float32)


def _scores(*values: float) -> torch.Tensor:
    """(N,) float32 scores from literals."""
    return torch.tensor(values, dtype=torch.float32)


def _iou(a: list[float], b: list[float]) -> float:
    """Independent IoU of two xyxy boxes.

    Hand-computed here (NOT via the module's own `_box_iou`) so the geometry the
    cases rely on is established without testing the implementation against
    itself.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Contract surface
# ---------------------------------------------------------------------------


def test_nms_is_reachable_where_the_specialist_imports_it() -> None:
    # The consumer does `from specialists.grounding.head import ... nms`
    # (specialists/grounding/specialist.py:302). Importing it here is the same
    # contract, so this test fails loudly if the symbol ever moves or is renamed.
    assert callable(nms)


def test_default_iou_threshold_is_one_half() -> None:
    # The consumer passes `self.nms_iou` explicitly, but the default is part of
    # the documented signature and callers may rely on it.
    default = inspect.signature(nms).parameters["iou_threshold"].default
    assert default == 0.5


# ---------------------------------------------------------------------------
# Core suppression behaviour
# ---------------------------------------------------------------------------


def test_overlapping_box_is_suppressed_and_highest_scoring_survives() -> None:
    # Boxes 0 and 1 are identical (IoU 1.0, far above the 0.5 default); box 2 is
    # disjoint from both. Only the highest-scoring of the overlapping pair
    # survives, and it is kept first.
    boxes = _boxes(
        [0.0, 0.0, 0.4, 0.4],
        [0.0, 0.0, 0.4, 0.4],
        [0.6, 0.6, 1.0, 1.0],
    )
    scores = _scores(0.9, 0.5, 0.3)

    keep = nms(boxes, scores)

    assert keep.tolist() == [0, 2]
    assert int(scores.argmax()) in keep.tolist()  # the best box is the survivor


def test_boxes_below_the_threshold_are_all_retained() -> None:
    a = [0.0, 0.0, 0.4, 0.4]
    d = [0.35, 0.0, 0.75, 0.4]
    assert _iou(a, d) < 0.5  # the premise of the case, computed independently

    keep = nms(_boxes(a, d), _scores(0.9, 0.8))

    assert sorted(keep.tolist()) == [0, 1]


def test_single_box_passes_through_unchanged() -> None:
    boxes = _boxes([0.1, 0.2, 0.3, 0.4])
    keep = nms(boxes, _scores(0.7))

    assert keep.tolist() == [0]
    assert torch.equal(boxes[keep], boxes)  # the surviving geometry is untouched


def test_empty_input_returns_empty_without_raising() -> None:
    keep = nms(torch.zeros((0, 4)), torch.zeros((0,)))

    assert keep.shape == (0,)
    assert keep.dtype == torch.long


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_score_tie_is_resolved_deterministically() -> None:
    # Two identical boxes with equal scores: exactly one survives, and repeated
    # calls must agree on which. We do NOT pin *which* index wins (torch's
    # argsort tie order is an implementation detail) -- only that it is stable.
    boxes = _boxes([0.0, 0.0, 0.4, 0.4], [0.0, 0.0, 0.4, 0.4])
    scores = _scores(0.5, 0.5)

    first = nms(boxes, scores)
    assert first.numel() == 1
    assert first.item() in (0, 1)
    for _ in range(5):
        assert torch.equal(nms(boxes, scores), first)


def test_output_ordering_is_deterministic_across_repeated_calls() -> None:
    boxes = _boxes(
        [0.0, 0.0, 0.4, 0.4],  # A
        [0.02, 0.02, 0.42, 0.42],  # B, heavily overlaps A
        [0.6, 0.6, 0.9, 0.9],  # C, disjoint
        [0.6, 0.0, 0.9, 0.3],  # D, disjoint
    )
    scores = _scores(0.9, 0.8, 0.7, 0.6)

    first = nms(boxes, scores)
    for _ in range(5):
        assert torch.equal(nms(boxes, scores), first)

    # "best first": the kept scores are non-increasing.
    kept_scores = scores[first]
    assert torch.all(kept_scores[:-1] >= kept_scores[1:])
    assert int(first[0]) == int(scores.argmax())


def test_result_is_invariant_to_input_permutation() -> None:
    boxes = _boxes(
        [0.0, 0.0, 0.4, 0.4],  # A
        [0.02, 0.02, 0.42, 0.42],  # B, heavily overlaps A
        [0.6, 0.6, 0.9, 0.9],  # C, disjoint
        [0.6, 0.0, 0.9, 0.3],  # D, disjoint
    )
    scores = _scores(0.9, 0.8, 0.7, 0.6)

    base_keep = nms(boxes, scores)
    base_boxes = {tuple(round(v, 6) for v in boxes[i].tolist()) for i in base_keep.tolist()}
    base_kept_scores = [round(float(scores[i]), 6) for i in base_keep.tolist()]

    permutation = [2, 0, 3, 1]
    perm_boxes = boxes[permutation]
    perm_scores = scores[permutation]

    perm_keep = nms(perm_boxes, perm_scores)
    perm_kept_set = {
        tuple(round(v, 6) for v in perm_boxes[i].tolist()) for i in perm_keep.tolist()
    }
    perm_kept_scores = [round(float(perm_scores[i]), 6) for i in perm_keep.tolist()]

    # Same set in -> same set out, in the same (score-descending) order.
    assert perm_kept_set == base_boxes
    assert perm_kept_scores == base_kept_scores


# ---------------------------------------------------------------------------
# Threshold boundary (exact arithmetic, no float fuzz)
# ---------------------------------------------------------------------------
#
# Containment gives an exactly-representable IoU: when box b is fully inside
# box a, IoU = area_b / area_a. The boxes below are chosen so every coordinate,
# area and the resulting IoU are exact in binary floating point, which lets us
# pin the `<=` boundary without an epsilon fudge.


def test_threshold_boundary_at_the_default_one_half() -> None:
    a = [0.0, 0.0, 0.5, 0.5]  # area 0.25
    b = [0.0, 0.0, 0.5, 0.25]  # area 0.125, fully inside a
    assert _iou(a, b) == 0.5  # exact: 0.125 / 0.25

    boxes, scores = _boxes(a, b), _scores(0.9, 0.5)

    # IoU exactly equal to the threshold -> KEPT (the comparison is `<=`).
    assert nms(boxes, scores, 0.5).tolist() == [0, 1]
    # IoU just ABOVE the threshold -> suppressed.
    assert nms(boxes, scores, 0.49).tolist() == [0]
    # IoU just BELOW the threshold -> retained.
    assert nms(boxes, scores, 0.51).tolist() == [0, 1]


def test_threshold_boundary_at_one_quarter() -> None:
    a = [0.0, 0.0, 0.5, 0.5]  # area 0.25
    b = [0.125, 0.125, 0.375, 0.375]  # area 0.0625, fully inside a
    assert _iou(a, b) == 0.25  # exact: 0.0625 / 0.25

    boxes, scores = _boxes(a, b), _scores(0.9, 0.5)

    assert nms(boxes, scores, 0.25).tolist() == [0, 1]  # at the boundary: kept
    assert nms(boxes, scores, 0.24).tolist() == [0]  # IoU just above: suppressed
    assert nms(boxes, scores, 0.26).tolist() == [0, 1]  # IoU just below: kept


def test_threshold_one_keeps_everything_and_zero_drops_full_overlap() -> None:
    # Identical boxes have IoU 1.0 exactly.
    boxes, scores = _boxes([0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.4, 0.4]), _scores(0.9, 0.5)

    # Upper bound: IoU 1.0 <= 1.0 -> both kept.
    assert nms(boxes, scores, 1.0).tolist() == [0, 1]
    # Lower bound: IoU 1.0 > 0.0 -> the second is suppressed.
    assert nms(boxes, scores, 0.0).tolist() == [0]


# ---------------------------------------------------------------------------
# Argument validation (documented guards)
# ---------------------------------------------------------------------------


def test_out_of_range_threshold_is_rejected() -> None:
    boxes, scores = _boxes([0.0, 0.0, 1.0, 1.0]), _scores(0.5)
    with pytest.raises(ValueError):
        nms(boxes, scores, 1.5)
    with pytest.raises(ValueError):
        nms(boxes, scores, -0.1)


def test_malformed_boxes_are_rejected() -> None:
    with pytest.raises(ValueError):
        nms(torch.zeros((2, 3)), _scores(0.5, 0.5))


def test_scores_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        nms(_boxes([0.0, 0.0, 1.0, 1.0]), _scores(0.5, 0.5))
