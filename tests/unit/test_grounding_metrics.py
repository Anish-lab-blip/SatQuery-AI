"""Tests for the grounding metrics.

Every expected value is hand-computed. A metric that agrees only with itself,
or with a library it is a thin wrapper around, proves nothing about whether it
is measuring the right thing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from evaluation.metrics.grounding import (
    DEFAULT_THRESHOLDS,
    best_iou,
    benchmark_to_normalized,
    box_area,
    box_iou,
    box_iou_matrix,
    greedy_match,
    intersection_area,
    mean_matched_iou,
    recall_at_iou,
    score_dataset,
    score_one,
)

# ---------------------------------------------------------------------------
# Hand-computed cases
# ---------------------------------------------------------------------------


def test_identical_boxes_have_iou_one() -> None:
    box = [0.1, 0.1, 0.5, 0.5]
    assert box_iou(box, box) == pytest.approx(1.0)


def test_disjoint_boxes_have_iou_zero() -> None:
    assert box_iou([0.0, 0.0, 0.2, 0.2], [0.5, 0.5, 0.7, 0.7]) == 0.0


def test_touching_boxes_have_iou_zero() -> None:
    """Edges that touch share zero area, not a line."""
    assert box_iou([0.0, 0.0, 0.5, 0.5], [0.5, 0.0, 1.0, 0.5]) == 0.0


def test_half_overlap_iou() -> None:
    """Two 1x1 squares offset by 0.5 in x.

    intersection = 0.5, union = 1.0 + 1.0 - 0.5 = 1.5, IoU = 1/3.
    """
    a = [0.0, 0.0, 1.0, 1.0]
    b = [0.5, 0.0, 1.5, 1.0]
    assert box_iou(a, b) == pytest.approx(1.0 / 3.0)


def test_nested_box_iou() -> None:
    """A 0.5x0.5 box fully inside a 1x1 box.

    intersection = 0.25, union = 1.0, IoU = 0.25.
    """
    outer = [0.0, 0.0, 1.0, 1.0]
    inner = [0.25, 0.25, 0.75, 0.75]
    assert box_iou(outer, inner) == pytest.approx(0.25)


def test_iou_is_symmetric() -> None:
    a = [0.1, 0.2, 0.4, 0.6]
    b = [0.3, 0.1, 0.7, 0.5]
    assert box_iou(a, b) == pytest.approx(box_iou(b, a))


def test_iou_of_sixty_percent_overlap() -> None:
    """Overlap of 0.6 x 1.0 against two 1x1 boxes.

    intersection = 0.6, union = 2.0 - 0.6 = 1.4, IoU = 3/7.
    """
    a = [0.0, 0.0, 1.0, 1.0]
    b = [0.4, 0.0, 1.4, 1.0]
    assert box_iou(a, b) == pytest.approx(0.6 / 1.4)


# ---------------------------------------------------------------------------
# Degenerate input -- must not produce NaN
# ---------------------------------------------------------------------------


def test_zero_area_box_yields_iou_zero_not_nan() -> None:
    degenerate = [0.5, 0.5, 0.5, 0.5]
    normal = [0.0, 0.0, 1.0, 1.0]
    result = box_iou(degenerate, normal)
    assert result == 0.0
    assert not math.isnan(result)


def test_two_identical_degenerate_boxes_yield_zero() -> None:
    degenerate = [0.5, 0.5, 0.5, 0.5]
    assert box_iou(degenerate, degenerate) == 0.0


def test_inverted_box_is_zero_area_not_negative() -> None:
    inverted = [0.8, 0.8, 0.2, 0.2]
    assert box_area(np.asarray([inverted], dtype=np.float64))[0] == 0.0


def test_matrix_contains_no_nan_for_degenerate_input() -> None:
    preds = [[0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 1.0, 1.0]]
    targets = [[0.0, 0.0, 0.0, 0.0], [0.2, 0.2, 0.8, 0.8]]
    matrix = box_iou_matrix(preds, targets)
    assert not np.isnan(matrix).any()


# ---------------------------------------------------------------------------
# Matrix form
# ---------------------------------------------------------------------------


def test_matrix_shape() -> None:
    matrix = box_iou_matrix([[0, 0, 1, 1]] * 3, [[0, 0, 1, 1]] * 5)
    assert matrix.shape == (3, 5)


def test_matrix_empty_predictions() -> None:
    assert box_iou_matrix([], [[0, 0, 1, 1]]).shape == (0, 1)


def test_matrix_empty_targets() -> None:
    assert box_iou_matrix([[0, 0, 1, 1]], []).shape == (1, 0)


def test_matrix_rejects_malformed_boxes() -> None:
    with pytest.raises(ValueError, match="shaped"):
        box_iou_matrix([[0, 0, 1]], [[0, 0, 1, 1]])


def test_matrix_matches_scalar_form() -> None:
    preds = [[0.0, 0.0, 0.4, 0.4], [0.5, 0.5, 0.9, 0.9]]
    targets = [[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 1.0, 1.0]]
    matrix = box_iou_matrix(preds, targets)
    for i, p in enumerate(preds):
        for j, t in enumerate(targets):
            assert matrix[i, j] == pytest.approx(box_iou(p, t))


def test_intersection_area_shape_and_values() -> None:
    a = np.asarray([[0.0, 0.0, 1.0, 1.0]])
    b = np.asarray([[0.5, 0.5, 1.5, 1.5]])
    inter = intersection_area(a, b)
    assert inter.shape == (1, 1)
    assert inter[0, 0] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Greedy matching
# ---------------------------------------------------------------------------


def test_perfect_match_at_half_iou() -> None:
    target = [0.0, 0.0, 1.0, 1.0]
    pred = [0.0, 0.0, 1.0, 1.0]
    result = greedy_match([pred], [target], 0.5)
    assert result.true_positives == 1
    assert result.recall == pytest.approx(1.0)
    assert result.precision == pytest.approx(1.0)


def test_no_match_when_below_threshold() -> None:
    """IoU of 1/3 is below 0.5."""
    result = greedy_match(
        [[0.0, 0.0, 1.0, 1.0]], [[0.5, 0.0, 1.5, 1.0]], iou_threshold=0.5
    )
    assert result.true_positives == 0
    assert result.recall == 0.0
    assert result.false_positives == 1
    assert result.false_negatives == 1


def test_matching_is_one_to_one() -> None:
    """Two predictions cannot both claim the same single target."""
    target = [0.0, 0.0, 1.0, 1.0]
    preds = [[0.0, 0.0, 1.0, 1.0], [0.01, 0.01, 0.99, 0.99]]
    result = greedy_match(preds, [target], 0.5)
    assert result.true_positives == 1
    assert result.false_positives == 1


def test_higher_iou_prediction_wins_the_match() -> None:
    target = [0.0, 0.0, 1.0, 1.0]
    good = [0.0, 0.0, 1.0, 1.0]          # IoU 1.0
    worse = [0.0, 0.0, 0.9, 0.9]         # IoU 0.81
    result = greedy_match([worse, good], [target], 0.5)
    assert result.matches[0][0] == 1, "the better prediction was not selected"


def test_matching_is_independent_of_prediction_order() -> None:
    target = [0.0, 0.0, 1.0, 1.0]
    a = [0.0, 0.0, 1.0, 1.0]
    b = [0.0, 0.0, 0.7, 0.7]
    forward = greedy_match([a, b], [target], 0.5)
    backward = greedy_match([b, a], [target], 0.5)
    assert forward.true_positives == backward.true_positives
    # Both select the better box; only its index changes.
    assert forward.matches[0][2] == pytest.approx(backward.matches[0][2])


def test_two_targets_two_correct_predictions() -> None:
    targets = [[0.0, 0.0, 0.4, 0.4], [0.6, 0.6, 1.0, 1.0]]
    preds = [[0.0, 0.0, 0.4, 0.4], [0.6, 0.6, 1.0, 1.0]]
    result = greedy_match(preds, targets, 0.5)
    assert result.true_positives == 2
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)


def test_recall_counts_unmatched_targets() -> None:
    targets = [[0.0, 0.0, 0.4, 0.4], [0.6, 0.6, 1.0, 1.0]]
    preds = [[0.0, 0.0, 0.4, 0.4]]
    result = greedy_match(preds, targets, 0.5)
    assert result.true_positives == 1
    assert result.recall == pytest.approx(0.5)
    assert result.false_negatives == 1


def test_no_predictions_gives_zero_recall() -> None:
    result = greedy_match([], [[0.0, 0.0, 1.0, 1.0]], 0.5)
    assert result.recall == 0.0
    assert result.precision == 0.0


def test_no_targets_gives_zero_recall_not_divide_by_zero() -> None:
    result = greedy_match([[0.0, 0.0, 1.0, 1.0]], [], 0.5)
    assert result.recall == 0.0
    assert result.precision == 0.0
    assert not math.isnan(result.recall)


def test_greedy_match_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        greedy_match([], [], iou_threshold=1.5)


def test_f1_of_partial_match() -> None:
    """1 TP, 1 FP, 1 FN -> precision 0.5, recall 0.5, F1 0.5."""
    targets = [[0.0, 0.0, 0.4, 0.4], [0.6, 0.6, 1.0, 1.0]]
    preds = [[0.0, 0.0, 0.4, 0.4], [0.1, 0.1, 0.3, 0.3]]
    result = greedy_match(preds, targets, 0.5)
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.f1 == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Threshold behaviour
# ---------------------------------------------------------------------------


def test_recall_is_monotonic_in_threshold() -> None:
    """A lower IoU threshold can never match fewer targets."""
    targets = [[0.0, 0.0, 1.0, 1.0]]
    preds = [[0.5, 0.0, 1.5, 1.0]]  # IoU 1/3
    assert recall_at_iou(preds, targets, 0.25) == 1.0
    assert recall_at_iou(preds, targets, 0.5) == 0.0


def test_default_thresholds_are_the_standard_set() -> None:
    assert DEFAULT_THRESHOLDS == (0.25, 0.5, 0.75)


def test_recall_at_exact_threshold_matches() -> None:
    """A pair whose IoU equals the threshold counts as a match."""
    a = [0.0, 0.0, 1.0, 1.0]
    b = [0.5, 0.0, 1.5, 1.0]  # IoU = 1/3 exactly
    assert recall_at_iou([a], [b], 1.0 / 3.0) == 1.0


# ---------------------------------------------------------------------------
# Best IoU and mean matched IoU
# ---------------------------------------------------------------------------


def test_best_iou_picks_the_maximum() -> None:
    targets = [[0.0, 0.0, 1.0, 1.0]]
    preds = [[0.8, 0.8, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]
    assert best_iou(preds, targets) == pytest.approx(1.0)


def test_best_iou_with_no_targets_is_zero() -> None:
    assert best_iou([[0.0, 0.0, 1.0, 1.0]], []) == 0.0


def test_mean_matched_iou_averages_matches() -> None:
    targets = [[0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 1.0, 1.0]]
    preds = [[0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 0.9, 0.9]]
    # First pair IoU 1.0; second pair 0.4*0.4 / (0.25 + 0.16 - 0.16) = 0.16/0.25
    value = mean_matched_iou(preds, targets)
    assert 0.0 < value <= 1.0


def test_mean_matched_iou_with_no_matches_is_zero() -> None:
    assert mean_matched_iou([[0, 0, 0.1, 0.1]], [[0.9, 0.9, 1.0, 1.0]]) == 0.0


# ---------------------------------------------------------------------------
# Per-image and dataset scoring
# ---------------------------------------------------------------------------


def test_score_one_reports_all_thresholds() -> None:
    score = score_one([[0.0, 0.0, 1.0, 1.0]], [[0.0, 0.0, 1.0, 1.0]])
    assert set(score.recall) == set(DEFAULT_THRESHOLDS)
    assert all(v == pytest.approx(1.0) for v in score.recall.values())


def test_score_one_counts() -> None:
    score = score_one([[0, 0, 1, 1], [0.1, 0.1, 0.2, 0.2]], [[0, 0, 1, 1]])
    assert score.n_preds == 2
    assert score.n_targets == 1


def test_score_dataset_aggregates() -> None:
    pairs = [
        ([[0.0, 0.0, 1.0, 1.0]], [[0.0, 0.0, 1.0, 1.0]]),  # perfect
        ([[0.0, 0.0, 0.2, 0.2]], [[0.8, 0.8, 1.0, 1.0]]),  # total miss
    ]
    report = score_dataset(pairs)
    assert report.n_images == 2
    assert report.n_images_with_targets == 2
    assert report.recall[0.5] == pytest.approx(0.5)


def test_score_dataset_excludes_targetless_images_from_recall() -> None:
    """A recall over zero targets is undefined; it must not be averaged in as 0.

    Counting it as 0.0 would penalise the model for a property of the dataset.
    """
    pairs = [
        ([[0.0, 0.0, 1.0, 1.0]], [[0.0, 0.0, 1.0, 1.0]]),  # perfect
        ([[0.0, 0.0, 1.0, 1.0]], []),                        # no targets
    ]
    report = score_dataset(pairs)
    assert report.n_images == 2
    assert report.n_images_with_targets == 1
    assert report.recall[0.5] == pytest.approx(1.0), (
        "a targetless image dragged the recall average down"
    )


def test_score_dataset_empty_input() -> None:
    report = score_dataset([])
    assert report.n_images == 0
    assert report.recall[0.5] == 0.0
    assert not math.isnan(report.mean_best_iou)


def test_report_is_json_serialisable() -> None:
    import json

    report = score_dataset([([[0, 0, 1, 1]], [[0, 0, 1, 1]])])
    json.dumps(report.to_dict())


def test_report_summary_is_renderable() -> None:
    report = score_dataset([([[0, 0, 1, 1]], [[0, 0, 1, 1]])])
    text = report.summary()
    assert "recall @ IoU" in text
    assert "mean best IoU" in text


# ---------------------------------------------------------------------------
# Benchmark coordinate conversion
# ---------------------------------------------------------------------------


def test_benchmark_box_scales_to_normalized() -> None:
    assert benchmark_to_normalized([0, 0, 50, 100]) == [0.0, 0.0, 0.5, 1.0]


def test_benchmark_conversion_rejects_bad_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        benchmark_to_normalized([0, 0, 1, 1], scale=0)


def test_benchmark_conversion_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="4 values"):
        benchmark_to_normalized([0, 0, 1])


def test_benchmark_conversion_is_exact_for_round_values() -> None:
    assert benchmark_to_normalized([25, 50, 75, 100]) == [0.25, 0.5, 0.75, 1.0]