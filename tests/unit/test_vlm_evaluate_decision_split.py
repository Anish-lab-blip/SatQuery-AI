"""Tests for `decide_acceptance(..., decision_split=...)`.

WHY THIS PARAMETER EXISTS
-------------------------
`docs/PHASE6_RUN1_REJECTION_DIAGNOSIS.md` §7.4 condition 3 requires the v002
acceptance to be adjudicated on the **test** split:

    "Decide on data that did not motivate the change. Evaluate `v002` on the test
     split ... Reporting an acceptance decision from `v002` on the same val subset
     that revealed the defect would be self-confirming."

Item V defines V2 on the validation split, and `decide_acceptance` hardcoded that.
`decision_split` selects which split feeds the V1/V2 gates. **The criteria are not
altered** — only which split supplies the metrics.

These tests pin two things that must both hold:

  1. `decision_split="val"` (the default) reproduces the previous behaviour
     **including the exact reason strings**, so no recorded verdict or assertion
     changes.
  2. `decision_split="test"` genuinely runs V2 on the *test* sets — proven by
     constructing a case where val fails V2 and test passes, and showing the
     verdict flips.

`MetricSet` is built directly, so no model and no generation is needed.
"""

from __future__ import annotations

import pytest

from training.vlm.evaluate import (
    MIN_CLASS_QUESTIONS,
    MetricSet,
    decide_acceptance,
)

# A class that fails v002 on a 0.90 -> 0.70 move over 100 questions:
#   lost = round(0.20 * 100) = 20  (>= MIN_CLASS_DROP_QUESTIONS = 4)
#   se   = sqrt((0.9*0.1 + 0.7*0.3) / 100) = 0.054772
#   z    = 0.20 / 0.054772 = 3.65  (>= CLASS_DROP_Z = 1.96)
# It fails BOTH halves of the v002 criterion.
FAILING = ("Transitional woodland, shrub", 100, 0.90, 0.70)

# A class that passes v002: 0.90 -> 0.88 over 100 questions.
#   lost = 2  (< 4) -> fails the materiality floor, so it is never flagged.
# It also passes v001's flat rule only if the drop exceeds 1.0 pp, which 2.00 pp
# does -- see `test_v001_replays_under_the_test_split`, which relies on that.
PASSING = ("Mixed forest", 100, 0.90, 0.88)

# class tuple layout: (name, count, baseline_accuracy, adapted_accuracy)
_CLS, _CNT, _BASE, _ADAP = 0, 1, 2, 3


def _baseline(exact_match: float, classes: list[tuple], n: int = 1000) -> MetricSet:
    return MetricSet(
        n=n,
        exact_match=exact_match,
        per_class_accuracy={c[_CLS]: c[_BASE] for c in classes},
        per_class_counts={c[_CLS]: c[_CNT] for c in classes},
    )


def _adapted(exact_match: float, classes: list[tuple], n: int = 1000) -> MetricSet:
    return MetricSet(
        n=n,
        exact_match=exact_match,
        per_class_accuracy={c[_CLS]: c[_ADAP] for c in classes},
        per_class_counts={c[_CLS]: c[_CNT] for c in classes},
    )


def _val_clean_test_failing():
    """val passes V1/V2; test passes V1 but FAILS V2.

    Returns (baseline_val, adapted_val, baseline_test, adapted_test).
    """
    return (
        _baseline(0.491, [PASSING]),
        _adapted(0.911, [PASSING]),
        _baseline(0.468, [FAILING]),
        _adapted(0.963, [FAILING]),
    )


def _val_failing_test_clean():
    """val passes V1 but FAILS V2; test passes V1/V2."""
    return (
        _baseline(0.491, [FAILING]),
        _adapted(0.911, [FAILING]),
        _baseline(0.468, [PASSING]),
        _adapted(0.963, [PASSING]),
    )


def _decide(baseline_val, adapted_val, baseline_test, adapted_test, **kw):
    return decide_acceptance(
        baseline_val,
        adapted_val,
        baseline_test=baseline_test,
        adapted_test=adapted_test,
        run_completed=True,
        artifact_ok=True,
        **kw,
    )


# ---------------------------------------------------------------------------
# 0. the fixture is real — these two classes must actually fail/pass V2
# ---------------------------------------------------------------------------
def test_fixture_classes_behave_as_documented():
    """Guard the fixtures: if the predicate changes, these tests must not go quiet."""
    b, a = _baseline(0.468, [FAILING]), _adapted(0.963, [FAILING])
    d = _decide(_baseline(0.491, [PASSING]), _adapted(0.911, [PASSING]), b, a,
                decision_split="test")
    assert d.status == "REJECTED"
    assert len(d.class_failures) == 1
    assert d.class_failures[0]["lost_questions"] == 20
    assert d.class_failures[0]["z"] >= 1.96

    b2, a2 = _baseline(0.468, [PASSING]), _adapted(0.963, [PASSING])
    d2 = _decide(_baseline(0.491, [PASSING]), _adapted(0.911, [PASSING]), b2, a2,
                 decision_split="test")
    assert d2.status == "ACCEPTED"
    assert d2.class_failures == []


# ---------------------------------------------------------------------------
# 1. the default path is unchanged, reason strings included
# ---------------------------------------------------------------------------
def test_default_is_val_gated_and_keeps_legacy_reason_strings():
    bv, av, bt, at = _val_clean_test_failing()
    d = _decide(bv, av, bt, at)

    assert d.status == "ACCEPTED"
    assert d.val_delta_pp == pytest.approx(42.0)
    assert d.test_delta_pp == pytest.approx(49.5)
    # The legacy strings, verbatim. A drift here means a recorded verdict's
    # evidence text changed without a rule version bump.
    assert d.reasons[0] == (
        "val baseline=49.10 pp, adapted=91.10 pp, delta=+42.00 pp; "
        "required (V1) >= +5.00 pp"
    )
    assert d.reasons[1] == "V1 and V2 passed on validation"
    assert d.reasons[2] == "test delta=+49.50 pp (val delta +42.00 pp)"


def test_default_is_val_gated_so_a_test_only_class_failure_does_not_reject():
    bv, av, bt, at = _val_clean_test_failing()
    d = _decide(bv, av, bt, at)
    assert d.status == "ACCEPTED"
    assert d.class_failures == []


def test_default_rejects_when_val_has_a_class_failure():
    bv, av, bt, at = _val_failing_test_clean()
    d = _decide(bv, av, bt, at)
    assert d.status == "REJECTED"
    assert len(d.class_failures) == 1
    assert d.class_failures[0]["class"] == FAILING[_CLS]
    assert "on validation" in d.reasons[-1]
    # The test delta is still recorded on a REJECTED run.
    assert d.test_delta_pp == pytest.approx(49.5)


# ---------------------------------------------------------------------------
# 2. decision_split="test" genuinely moves the gates
# ---------------------------------------------------------------------------
def test_test_split_accepts_where_val_split_rejects():
    """The same metrics: val-gated REJECTED, test-gated ACCEPTED.

    This is the load-bearing test. If it ever passes for both modes, the split
    selector is not actually reaching V2.
    """
    bv, av, bt, at = _val_failing_test_clean()

    val_gated = _decide(bv, av, bt, at)
    test_gated = _decide(bv, av, bt, at, decision_split="test")

    assert val_gated.status == "REJECTED"
    assert test_gated.status == "ACCEPTED"
    assert test_gated.class_failures == []


def test_test_split_rejects_where_val_split_accepts():
    """The converse: val clean, test failing -> test-gated REJECTED."""
    bv, av, bt, at = _val_clean_test_failing()

    val_gated = _decide(bv, av, bt, at)
    test_gated = _decide(bv, av, bt, at, decision_split="test")

    assert val_gated.status == "ACCEPTED"
    assert test_gated.status == "REJECTED"
    assert len(test_gated.class_failures) == 1
    assert "on test" in test_gated.reasons[-1]


def test_test_split_reason_strings_are_labelled_test():
    # NOTE: the fixture must be the one whose *test* split passes V2 -- the
    # `_val_clean_test_failing` fixture fails V2 on test by construction, so its
    # reasons[1] is a V2 failure line, not the "passed on test" line.
    bv, av, bt, at = _val_failing_test_clean()
    d = _decide(bv, av, bt, at, decision_split="test")
    assert d.reasons[0] == (
        "test baseline=46.80 pp, adapted=96.30 pp, delta=+49.50 pp; "
        "required (V1) >= +5.00 pp"
    )
    assert d.reasons[1] == "V1 and V2 passed on test"
    assert d.reasons[2] == "val delta=+42.00 pp (test delta +49.50 pp)"


def test_deltas_are_named_by_split_in_both_modes():
    bv, av, bt, at = _val_clean_test_failing()
    for split in ("val", "test"):
        d = _decide(bv, av, bt, at, decision_split=split)
        assert d.val_delta_pp == pytest.approx(42.0)
        assert d.test_delta_pp == pytest.approx(49.5)


# ---------------------------------------------------------------------------
# 3. the confirming split is mirrored
# ---------------------------------------------------------------------------
def test_test_split_is_confirmed_by_val():
    """With test as the gate split, val is what confirms."""
    # val improves only +2.00 pp -> below the +5.00 required -> does not confirm.
    bv = _baseline(0.90, [PASSING])
    av = _adapted(0.92, [PASSING])
    bt, at = _baseline(0.468, [PASSING]), _adapted(0.963, [PASSING])

    d = _decide(bv, av, bt, at, decision_split="test")
    assert d.status == "ACCEPTED_ON_TEST_NOT_CONFIRMED_ON_VAL"
    assert any("val did not confirm the test result" in r for r in d.reasons)


def test_val_split_keeps_the_original_confirmation_status():
    bv = _baseline(0.491, [PASSING])
    av = _adapted(0.911, [PASSING])
    bt = _baseline(0.60, [PASSING])
    at = _adapted(0.61, [PASSING])  # +1.00 pp, below the required +5.00

    d = _decide(bv, av, bt, at)
    assert d.status == "ACCEPTED_ON_VAL_NOT_CONFIRMED_ON_TEST"


# ---------------------------------------------------------------------------
# 4. argument validation
# ---------------------------------------------------------------------------
def test_test_split_without_both_test_sets_raises():
    bv, av = _baseline(0.491, [PASSING]), _adapted(0.911, [PASSING])
    with pytest.raises(ValueError, match="requires both baseline_test and adapted_test"):
        _decide(bv, av, None, None, decision_split="test")


def test_unknown_decision_split_raises():
    bv, av = _baseline(0.491, [PASSING]), _adapted(0.911, [PASSING])
    with pytest.raises(ValueError, match="decision_split must be"):
        _decide(bv, av, None, None, decision_split="validation")


def test_unknown_rule_version_still_raises_under_test_split():
    bv, av = _baseline(0.491, [PASSING]), _adapted(0.911, [PASSING])
    with pytest.raises(ValueError, match="unknown rule_version"):
        _decide(bv, av, bv, av, decision_split="test", rule_version="v999")


# ---------------------------------------------------------------------------
# 5. v001 still replays under either split
# ---------------------------------------------------------------------------
def test_v001_replays_under_the_test_split():
    """v001's flat 1.0 pp rule fires on a 2.00 pp drop that v002 does not."""
    bt, at = _baseline(0.468, [PASSING]), _adapted(0.963, [PASSING])
    bv, av = _baseline(0.491, [PASSING]), _adapted(0.911, [PASSING])

    v002 = _decide(bv, av, bt, at, decision_split="test", rule_version="v002")
    v001 = _decide(bv, av, bt, at, decision_split="test", rule_version="v001")

    assert v002.status == "ACCEPTED"
    assert v001.status == "REJECTED"
    assert "V2 (v001) failed" in v001.reasons[-1]


def test_class_below_the_question_floor_is_ignored_in_both_modes():
    """A class under MIN_CLASS_QUESTIONS is excluded from V2 either way."""
    tiny = ("Rare class", MIN_CLASS_QUESTIONS - 1, 0.90, 0.10)  # huge drop, too few
    bt, at = _baseline(0.468, [tiny]), _adapted(0.963, [tiny])
    bv, av = _baseline(0.491, [tiny]), _adapted(0.911, [tiny])

    for split in ("val", "test"):
        d = _decide(bv, av, bt, at, decision_split=split)
        assert d.status == "ACCEPTED"
        assert d.class_failures == []
