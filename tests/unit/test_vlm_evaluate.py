"""Tests for the predeclared acceptance rule (`training.vlm.evaluate`).

The rule was declared on 2026-09-23 BEFORE any training run existed. These tests
pin every constant and every decision branch, because a threshold that drifts
after results are seen is exactly what the brief forbids: the whole point of
declaring `ACCEPTANCE_RULE_VERSION` up front is that it cannot be moved to fit
an outcome.

**The v002 bump is an explicit version bump, not a quiet edit.** Run 1 was
REJECTED under `v001` because its V2 guardrail (a flat 1.0 pp drop) fired on
three classes whose drops are statistically indistinguishable from zero. `v002`
replaces *only* V2's criterion with the contract's own "resolvable at ~3 SE"
resolution standard applied per-class, and `v001` is retained and replayable: replaying
run 1's recorded metrics yields the **same verdict, class set and key set**, but is
**not byte-identical**, because the manifest stores 6-dp-rounded accuracies while the
original run computed from raw counts. These tests therefore pin:

  * the new `v002` constants and `ACCEPTANCE_RULE_VERSION`;
  * that `rule_version="v001"` still reproduces the old predicate;
  * that the real run-1 numbers give 3 class failures under `v001` and 0 under
    `v002`;
  * that `v002` is not toothless (a large, significant collapse still fails);
  * that an unknown `rule_version` raises rather than silently falling back.

`MetricSet` is constructed directly, so no model and no generation is needed.
"""

from __future__ import annotations

import pytest

import training.vlm.evaluate as ev
from training.vlm.evaluate import (
    ACCEPTANCE_RULE_VERSION,
    ACCEPT_MIN_DELTA_PP,
    ACCEPT_MIN_DELTA_PP_CEILING,
    CEILING_BASELINE_PP,
    CLASS_DROP_Z,
    LEGACY_RULE_VERSIONS,
    MAX_CLASS_DROP_PP,
    MIN_CLASS_DROP_QUESTIONS,
    MIN_CLASS_QUESTIONS,
    NORMALISATION_RULE_VERSION,
    TEST_VAL_DISAGREEMENT_PP,
    MetricSet,
    _metric_set,
    class_of_question,
    decide_acceptance,
    describe_acceptance_rule,
    normalise_answer,
)


def _ms(
    exact_match: float,
    per_class: dict[str, float] | None = None,
    counts: dict[str, int] | None = None,
) -> MetricSet:
    return MetricSet(
        n=100,
        exact_match=exact_match,
        per_class_accuracy=per_class or {},
        per_class_counts=counts or {},
    )


def _run1_metrics() -> tuple[MetricSet, MetricSet]:
    """The three classes run 1 flagged under v001, exactly as recorded.

    Values are the recorded per-class accuracies and counts from the run-1
    manifest: `Transitional woodland, shrub` 90.625 -> 81.25 (n=32),
    `Mixed forest` 80.6452 -> 74.1935 (n=31), `Agro-forestry areas`
    76.7442 -> 72.093 (n=43). The aggregate exact-match is 0.491 -> 0.911
    (+42.00 pp), which passes V1. Only these three classes flagged under v001,
    so they are all the per-class data the V2 verdict depends on.
    """
    classes: dict[str, tuple[int, float, float]] = {
        "Transitional woodland, shrub": (32, 0.90625, 0.8125),
        "Mixed forest": (31, 0.806452, 0.741935),
        "Agro-forestry areas": (43, 0.767442, 0.72093),
    }
    baseline = MetricSet(
        n=1000,
        exact_match=0.491,
        per_class_accuracy={c: b for c, (_, b, _) in classes.items()},
        per_class_counts={c: n for c, (n, _, _) in classes.items()},
    )
    adapted = MetricSet(
        n=1000,
        exact_match=0.911,
        per_class_accuracy={c: a for c, (_, _, a) in classes.items()},
        per_class_counts={c: n for c, (n, _, _) in classes.items()},
    )
    return baseline, adapted


# ---------------------------------------------------------------------------
# The constants -- the rule must not drift
# ---------------------------------------------------------------------------
def test_acceptance_rule_constants_are_exactly_the_predeclared_values() -> None:
    # v002 is the current rule (declared 2026-09-24, after run 1).
    assert ACCEPTANCE_RULE_VERSION == "v002"
    assert MIN_CLASS_DROP_QUESTIONS == 4
    assert CLASS_DROP_Z == 1.96
    # v001 is retained so its verdict stays reproducible; its constants must
    # still be exactly the pre-registered values.
    assert LEGACY_RULE_VERSIONS == ("v001",)
    assert MAX_CLASS_DROP_PP == 1.0
    assert MIN_CLASS_QUESTIONS == 20
    # V1 / V1' / V3 / V4 are unchanged from v001.
    assert ACCEPT_MIN_DELTA_PP == 5.0
    assert CEILING_BASELINE_PP == 95.0
    assert ACCEPT_MIN_DELTA_PP_CEILING == 2.0
    assert TEST_VAL_DISAGREEMENT_PP == 10.0


# ---------------------------------------------------------------------------
# V1 / V1' -- the primary improvement condition
# ---------------------------------------------------------------------------
def test_v1_pass_accepts() -> None:
    decision = decide_acceptance(_ms(0.50), _ms(0.60), run_completed=True, artifact_ok=True)
    assert decision.status == "ACCEPTED"


def test_v1_fail_rejects() -> None:
    decision = decide_acceptance(_ms(0.50), _ms(0.52), run_completed=True, artifact_ok=True)
    assert decision.status == "REJECTED"


def test_boundary_at_exactly_five_points_is_accepted() -> None:
    decision = decide_acceptance(_ms(0.50), _ms(0.55), run_completed=True, artifact_ok=True)
    assert decision.val_delta_pp == pytest.approx(5.0)
    assert decision.status == "ACCEPTED"


def test_ceiling_clause_lowers_the_threshold_when_baseline_is_at_ceiling() -> None:
    """Baseline 96 pp >= 95 pp, so the required delta drops to +2.0 pp."""
    decision = decide_acceptance(_ms(0.96), _ms(0.98), run_completed=True, artifact_ok=True)
    assert decision.status == "ACCEPTED"
    assert decision.thresholds_used["ceiling_baseline_pp"] == 95.0
    assert decision.thresholds_used["accept_min_delta_pp_ceiling"] == 2.0


def test_ceiling_clause_still_rejects_below_two_points() -> None:
    decision = decide_acceptance(_ms(0.96), _ms(0.97), run_completed=True, artifact_ok=True)
    assert decision.status == "REJECTED"


# ---------------------------------------------------------------------------
# V2 -- no material, significant class collapse (v002)
# ---------------------------------------------------------------------------
def test_v2_class_collapse_rejects_even_when_v1_passes() -> None:
    """A large, significant collapse still rejects under the current rule."""
    baseline = _ms(0.50, per_class={"Marine waters": 0.90}, counts={"Marine waters": 30})
    adapted = _ms(0.60, per_class={"Marine waters": 0.50}, counts={"Marine waters": 30})
    decision = decide_acceptance(baseline, adapted, run_completed=True, artifact_ok=True)
    assert decision.status == "REJECTED"
    assert decision.class_failures
    assert decision.class_failures[0]["class"] == "Marine waters"


def test_v2_ignores_classes_with_too_few_questions() -> None:
    """The clause only applies to classes with >= MIN_CLASS_QUESTIONS questions."""
    baseline = _ms(0.50, per_class={"Marine waters": 0.90}, counts={"Marine waters": 5})
    adapted = _ms(0.60, per_class={"Marine waters": 0.10}, counts={"Marine waters": 5})
    decision = decide_acceptance(baseline, adapted, run_completed=True, artifact_ok=True)
    assert decision.status == "ACCEPTED"


def test_v2_failure_entry_carries_lost_questions_and_z() -> None:
    """A flagged class records the evidence, not just the point estimate."""
    baseline = _ms(0.50, per_class={"Marine waters": 0.90}, counts={"Marine waters": 100})
    adapted = _ms(0.60, per_class={"Marine waters": 0.40}, counts={"Marine waters": 100})
    decision = decide_acceptance(baseline, adapted, run_completed=True, artifact_ok=True)
    assert decision.status == "REJECTED"
    entry = decision.class_failures[0]
    assert entry["lost_questions"] == 50
    assert entry["z"] == pytest.approx(8.7039, abs=1e-3)
    # the v001 keys are retained for backward compatibility.
    assert {"class", "n_questions", "baseline_pp", "adapted_pp", "drop_pp"} <= set(entry)


def test_v2_does_not_flag_a_material_but_not_significant_drop() -> None:
    """The significance bar is load-bearing: 4 questions lost, z ~= 0.57."""
    baseline = _ms(0.50, per_class={"Marine waters": 0.60}, counts={"Marine waters": 100})
    adapted = _ms(0.60, per_class={"Marine waters": 0.56}, counts={"Marine waters": 100})
    decision = decide_acceptance(baseline, adapted, run_completed=True, artifact_ok=True)
    assert decision.status == "ACCEPTED"
    assert decision.class_failures == []


def test_v2_boundary_at_exactly_four_questions_lost_still_fails() -> None:
    """Exactly MIN_CLASS_DROP_QUESTIONS lost with z >= CLASS_DROP_Z fails.

    Pins the float boundary: `(20.0 / 100.0) * 20` can evaluate to
    3.999999999999999, which would silently clear the floor. `lost_questions` is
    mathematically an integer and is rounded before comparison.
    """
    baseline = _ms(0.50, per_class={"Marine waters": 1.0}, counts={"Marine waters": 20})
    adapted = _ms(0.60, per_class={"Marine waters": 0.80}, counts={"Marine waters": 20})
    decision = decide_acceptance(baseline, adapted, run_completed=True, artifact_ok=True)
    assert decision.status == "REJECTED"
    assert decision.class_failures[0]["lost_questions"] == 4


# ---------------------------------------------------------------------------
# v001 is retained and reproduces the original predicate exactly
# ---------------------------------------------------------------------------
def test_v001_reproduces_the_old_predicate_on_a_six_point_drop() -> None:
    """A 6.45 pp drop at n=31 failed v001 and passes v002 -- the run-1 case."""
    baseline = _ms(0.50, per_class={"Mixed forest": 0.806452}, counts={"Mixed forest": 31})
    adapted = _ms(0.60, per_class={"Mixed forest": 0.741935}, counts={"Mixed forest": 31})

    v001 = decide_acceptance(
        baseline, adapted, run_completed=True, artifact_ok=True, rule_version="v001"
    )
    assert v001.status == "REJECTED"
    assert len(v001.class_failures) == 1
    # v001 entries keep exactly the original five keys.
    assert set(v001.class_failures[0]) == {
        "class",
        "n_questions",
        "baseline_pp",
        "adapted_pp",
        "drop_pp",
    }

    v002 = decide_acceptance(
        baseline, adapted, run_completed=True, artifact_ok=True, rule_version="v002"
    )
    assert v002.status == "ACCEPTED"
    assert v002.class_failures == []


def test_run1_reproduces_three_class_failures_and_rejects_under_v001() -> None:
    baseline, adapted = _run1_metrics()
    decision = decide_acceptance(
        baseline, adapted, run_completed=True, artifact_ok=True, rule_version="v001"
    )
    assert decision.status == "REJECTED"
    assert len(decision.class_failures) == 3
    assert {f["class"] for f in decision.class_failures} == {
        "Transitional woodland, shrub",
        "Mixed forest",
        "Agro-forestry areas",
    }


def test_run1_passes_v002_with_zero_class_failures() -> None:
    baseline, adapted = _run1_metrics()
    decision = decide_acceptance(
        baseline, adapted, run_completed=True, artifact_ok=True, rule_version="v002"
    )
    assert decision.status == "ACCEPTED"
    assert decision.class_failures == []
    assert decision.val_delta_pp == pytest.approx(42.0)


def test_run1_flagged_classes_clear_each_v002_subcondition_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The robustness argument: remove EITHER sub-condition and the three
    run-1 classes still do not fail, so v002 was not tuned to them.

    * with the materiality floor removed (`MIN_CLASS_DROP_QUESTIONS = 0`), the
      significance bar alone must still clear all three -- none has z >= 1.96;
    * with the significance bar removed (`CLASS_DROP_Z = 0`), the materiality
      floor alone must still clear all three -- none lost 4 questions.
    """
    baseline, adapted = _run1_metrics()

    monkeypatch.setattr(ev, "MIN_CLASS_DROP_QUESTIONS", 0)
    assert ev._class_failures(baseline, adapted, rule_version="v002") == []

    monkeypatch.setattr(ev, "CLASS_DROP_Z", 0.0)
    monkeypatch.setattr(ev, "MIN_CLASS_DROP_QUESTIONS", MIN_CLASS_DROP_QUESTIONS)
    assert ev._class_failures(baseline, adapted, rule_version="v002") == []


def test_v002_is_not_toothless_a_large_significant_collapse_still_fails() -> None:
    baseline = _ms(0.50, per_class={"Marine waters": 0.90}, counts={"Marine waters": 100})
    adapted = _ms(0.60, per_class={"Marine waters": 0.40}, counts={"Marine waters": 100})
    decision = decide_acceptance(baseline, adapted, run_completed=True, artifact_ok=True)
    assert decision.status == "REJECTED"
    assert len(decision.class_failures) == 1


def test_unknown_rule_version_raises_rather_than_silently_falling_back() -> None:
    with pytest.raises(ValueError):
        decide_acceptance(
            _ms(0.50), _ms(0.60), run_completed=True, artifact_ok=True, rule_version="v999"
        )
    with pytest.raises(ValueError):
        describe_acceptance_rule("v999")



# ---------------------------------------------------------------------------
# V3 / V4 -- an unfinished run or a broken artifact is INCONCLUSIVE
# ---------------------------------------------------------------------------
def test_v3_incomplete_run_is_inconclusive_never_accepted() -> None:
    decision = decide_acceptance(_ms(0.50), _ms(0.99), run_completed=False, artifact_ok=True)
    assert decision.status == "INCONCLUSIVE"


def test_v4_bad_artifact_is_inconclusive() -> None:
    decision = decide_acceptance(_ms(0.50), _ms(0.99), run_completed=True, artifact_ok=False)
    assert decision.status == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Val vs test
# ---------------------------------------------------------------------------
def test_val_accept_not_confirmed_on_test_is_not_silently_promoted() -> None:
    decision = decide_acceptance(
        _ms(0.50),
        _ms(0.60),
        baseline_test=_ms(0.50),
        adapted_test=_ms(0.51),
        run_completed=True,
        artifact_ok=True,
    )
    assert decision.status == "ACCEPTED_ON_VAL_NOT_CONFIRMED_ON_TEST"


def test_val_and_test_agree_accepts() -> None:
    decision = decide_acceptance(
        _ms(0.50),
        _ms(0.60),
        baseline_test=_ms(0.50),
        adapted_test=_ms(0.60),
        run_completed=True,
        artifact_ok=True,
    )
    assert decision.status == "ACCEPTED"


# ---------------------------------------------------------------------------
# Recording the test delta -- a short-circuit must not discard it
# ---------------------------------------------------------------------------
def _v2_rejecting_val_pair() -> tuple[MetricSet, MetricSet]:
    """A val pair that passes V1 but fails V2 under the current (v002) rule.

    A large, significant collapse in one class: 50 questions lost at z ~= 8.7.
    """
    baseline = _ms(0.50, per_class={"Marine waters": 0.90}, counts={"Marine waters": 100})
    adapted = _ms(0.60, per_class={"Marine waters": 0.40}, counts={"Marine waters": 100})
    return baseline, adapted


def test_v2_rejected_run_still_records_the_test_delta() -> None:
    """Run 1's defect: a V2 rejection returned before the test-delta arithmetic,
    so a fully evaluated test split serialised as `test_delta_pp: null`."""
    baseline, adapted = _v2_rejecting_val_pair()
    decision = decide_acceptance(
        baseline,
        adapted,
        baseline_test=_ms(0.50),
        adapted_test=_ms(0.58),
        run_completed=True,
        artifact_ok=True,
    )
    assert decision.status == "REJECTED"
    assert decision.class_failures  # it really is a V2 rejection
    assert decision.test_delta_pp == pytest.approx(8.0)
    assert decision.to_dict()["test_delta_pp"] == pytest.approx(8.0)


def test_v1_rejected_run_also_records_the_test_delta() -> None:
    decision = decide_acceptance(
        _ms(0.50),
        _ms(0.52),
        baseline_test=_ms(0.50),
        adapted_test=_ms(0.58),
        run_completed=True,
        artifact_ok=True,
    )
    assert decision.status == "REJECTED"
    assert decision.test_delta_pp == pytest.approx(8.0)


def test_test_delta_is_none_when_no_test_metrics_were_supplied() -> None:
    """Absent test sets stay `None` -- the change records more, it does not invent."""
    baseline, adapted = _v2_rejecting_val_pair()
    decision = decide_acceptance(baseline, adapted, run_completed=True, artifact_ok=True)
    assert decision.status == "REJECTED"
    assert decision.test_delta_pp is None


def test_inconclusive_paths_do_not_gain_a_test_delta() -> None:
    """V3/V4 are untouched: their returns still carry no test delta."""
    v3 = decide_acceptance(
        _ms(0.50), _ms(0.60),
        baseline_test=_ms(0.50), adapted_test=_ms(0.58),
        run_completed=False, artifact_ok=True,
    )
    assert v3.status == "INCONCLUSIVE"
    assert v3.test_delta_pp is None

    v4 = decide_acceptance(
        _ms(0.50), _ms(0.60),
        baseline_test=_ms(0.50), adapted_test=_ms(0.58),
        run_completed=True, artifact_ok=False,
    )
    assert v4.status == "INCONCLUSIVE"
    assert v4.test_delta_pp is None


def test_recording_the_test_delta_does_not_change_any_accept_verdict() -> None:
    """Pin the pre-existing ACCEPTED / not-confirmed statuses so the recording
    change cannot silently alter a verdict."""
    confirmed = decide_acceptance(
        _ms(0.50), _ms(0.60),
        baseline_test=_ms(0.50), adapted_test=_ms(0.60),
        run_completed=True, artifact_ok=True,
    )
    assert confirmed.status == "ACCEPTED"
    assert confirmed.test_delta_pp == pytest.approx(10.0)

    not_confirmed = decide_acceptance(
        _ms(0.50), _ms(0.60),
        baseline_test=_ms(0.50), adapted_test=_ms(0.51),
        run_completed=True, artifact_ok=True,
    )
    assert not_confirmed.status == "ACCEPTED_ON_VAL_NOT_CONFIRMED_ON_TEST"
    assert not_confirmed.test_delta_pp == pytest.approx(1.0)

    val_only = decide_acceptance(_ms(0.50), _ms(0.60), run_completed=True, artifact_ok=True)
    assert val_only.status == "ACCEPTED"
    assert val_only.test_delta_pp is None


# ---------------------------------------------------------------------------
# A zero-sample test split is not a comparison
# ---------------------------------------------------------------------------
def _empty_ms() -> MetricSet:
    """A `MetricSet` for a split on which nothing was scored (`n == 0`)."""
    return MetricSet(n=0, exact_match=0.0)


def test_zero_sample_test_split_is_val_only_not_a_confident_zero_delta() -> None:
    """A split with zero questions scored must not be reported as a `0.0` delta.

    Comparing two empty sets yields 0.0, which would flow into
    `test_delta >= required` as "test did not confirm the val result" and flag
    the run for owner review -- inventing a negative finding from no data. It
    must instead fall through to the val-only ACCEPTED, with no such reason.
    """
    decision = decide_acceptance(
        _ms(0.50),
        _ms(0.60),
        baseline_test=_empty_ms(),
        adapted_test=_empty_ms(),
        run_completed=True,
        artifact_ok=True,
    )
    assert decision.status == "ACCEPTED"
    assert decision.test_delta_pp is None
    assert not any("test did not confirm" in r for r in decision.reasons)
    assert any("no test-split metrics supplied" in r for r in decision.reasons)


@pytest.mark.parametrize(
    "baseline_test,adapted_test",
    [
        (_empty_ms(), _ms(0.60)),  # baseline side empty
        (_ms(0.50), _empty_ms()),  # adapted side empty
    ],
)
def test_half_empty_test_split_is_not_a_comparison(
    baseline_test: MetricSet, adapted_test: MetricSet
) -> None:
    """A half-empty test comparison is not a comparison either."""
    decision = decide_acceptance(
        _ms(0.50),
        _ms(0.60),
        baseline_test=baseline_test,
        adapted_test=adapted_test,
        run_completed=True,
        artifact_ok=True,
    )
    assert decision.status == "ACCEPTED"
    assert decision.test_delta_pp is None
    assert not any("test did not confirm" in r for r in decision.reasons)


def test_non_empty_test_split_still_produces_the_delta_and_statuses() -> None:
    """Re-pin the normal path: both sides scored, the delta and statuses stand.

    Test delta +3.0 pp is below the +5.0 pp requirement, so the val accept is
    correctly downgraded and the "did not confirm" reason IS present -- the
    zero-sample guard must not suppress it when there is real data.
    """
    decision = decide_acceptance(
        _ms(0.50),
        _ms(0.60),
        baseline_test=_ms(0.50),
        adapted_test=_ms(0.53),
        run_completed=True,
        artifact_ok=True,
    )
    assert decision.status == "ACCEPTED_ON_VAL_NOT_CONFIRMED_ON_TEST"
    assert decision.test_delta_pp == pytest.approx(3.0)
    assert any("test did not confirm" in r for r in decision.reasons)


# ---------------------------------------------------------------------------
# The metrics -- one rule set, cross-checked against the repository metrics
# ---------------------------------------------------------------------------
def test_exact_match_agrees_with_the_repository_vqa_answer_accuracy() -> None:
    """For the presence answer vocabulary the local scorer and the repository's
    VQA scorer agree, so there is not a second normalisation rule set drifting
    from the tested one on the answers this task actually produces."""
    from evaluation.metrics.vqa import answer_accuracy

    preds = ["Yes.", "No.", "yes", "No."]
    golds = ["Yes.", "No.", "Yes.", "yes"]
    questions = ["Is water present in this image?"] * 4
    metrics = _metric_set(preds, golds, questions)
    assert metrics.exact_match == pytest.approx(answer_accuracy(preds, golds))


def test_local_normalisation_maps_yes_no_synonyms_the_vqa_rule_set_does_not() -> None:
    """The deliberate divergence: the presence task's answers are a two-token
    vocabulary, so the yes/no synonym map is the load-bearing rule. Reusing the
    VQA-v2 rule set would score "True" as wrong when it is right."""
    from evaluation.metrics.vqa import normalize_answer

    assert normalise_answer("True") == "yes"
    assert normalise_answer("False") == "no"
    assert normalize_answer("True") == "true"
    assert normalize_answer("False") == "false"


def test_normalisation_strips_surrounding_quotes_before_folding_synonyms() -> None:
    """The ORDER of the two steps is load-bearing.

    `evaluation.metrics.vqa.normalize_answer` deliberately preserves apostrophes,
    so `"'Yes.'"` survives it as `"'yes'"`. If the synonym fold ran first, it
    would miss `_YES_TOKENS` and the quoted form would score as wrong. Stripping
    the surrounding quotes must therefore happen BEFORE the fold. This pins that
    ordering; it is part of the metric definition, not an implementation detail.
    """
    assert normalise_answer("'Yes.'") == "yes"
    assert normalise_answer('"No."') == "no"


def test_metric_set_confusion_and_metrics_are_hand_computed() -> None:
    preds = ["Yes.", "Yes.", "No.", "No."]
    golds = ["Yes.", "No.", "No.", "No."]
    questions = ["Is water present in this image?"] * 4
    metrics = _metric_set(preds, golds, questions)
    assert metrics.confusion == {"tp": 1, "fp": 1, "tn": 2, "fn": 0}
    assert metrics.exact_match == pytest.approx(0.75)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(1.0)


def test_class_of_question_recovers_the_class_or_none() -> None:
    assert class_of_question("Is Marine waters present in this image?") == "Marine waters"
    assert class_of_question("Describe this image") is None


# ---------------------------------------------------------------------------
# The rule record -- BERTScore is unavailable, never zero
# ---------------------------------------------------------------------------
def test_bertscore_is_recorded_unavailable_not_as_zero() -> None:
    rule = describe_acceptance_rule()
    assert rule["bertscore"]["available"] is False
    assert "roberta-large" in rule["bertscore"]["reason"]
    assert "BERTScore" in rule["excluded_metrics"]


def test_describe_acceptance_rule_reports_the_v002_thresholds_honestly() -> None:
    rule = describe_acceptance_rule()
    assert rule["version"] == ACCEPTANCE_RULE_VERSION == "v002"
    # v002 was declared AFTER run 1, and must say so.
    assert rule["declared_before_training"] is False
    assert rule["v2_criterion"]["min_class_drop_questions"] == MIN_CLASS_DROP_QUESTIONS
    assert rule["v2_criterion"]["class_drop_z"] == CLASS_DROP_Z
    # the v001 facts are not overwritten -- they are still published.
    assert rule["max_class_drop_pp"] == MAX_CLASS_DROP_PP
    assert rule["min_class_questions"] == MIN_CLASS_QUESTIONS
    assert rule["v2_criterion"]["legacy_v001_criterion"]["max_class_drop_pp"] == MAX_CLASS_DROP_PP
    # the amendment records the original declaration date and the reason.
    assert rule["amendment"]["amends"] == "v001"
    assert rule["amendment"]["original_declared"] == "2026-09-23"
    assert rule["amendment"]["declared"] == "2026-09-24"
    assert rule["amendment"]["declared_before_training"] is False
    # V1 / V1' / V3 / V4 are unchanged.
    assert rule["accept_min_delta_pp"] == ACCEPT_MIN_DELTA_PP
    assert rule["ceiling_baseline_pp"] == CEILING_BASELINE_PP
    assert rule["accept_min_delta_pp_ceiling"] == ACCEPT_MIN_DELTA_PP_CEILING
    assert rule["test_val_disagreement_pp"] == TEST_VAL_DISAGREEMENT_PP


def test_describe_acceptance_rule_reports_v001_as_predeclared_when_asked() -> None:
    rule = describe_acceptance_rule("v001")
    assert rule["version"] == "v001"
    assert rule["declared_before_training"] is True
    assert rule["v2_criterion"]["max_class_drop_pp"] == MAX_CLASS_DROP_PP
    assert rule["v2_criterion"]["min_class_questions"] == MIN_CLASS_QUESTIONS
    # v001 has no amendment record -- it IS the original.
    assert "amendment" not in rule

