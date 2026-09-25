"""Tests for the Tier-1 VQA / Change-VQA text metrics (`evaluation.metrics.vqa`).

A metric that passes while proving nothing is worse than no metric, so these
tests aim to be MUTATION-PROOF:

  * each normalization rule (R1-R6) is pinned by a case that only that rule
    changes, so an edit that silently drops a rule fails a test;
  * every metric has an agree case, a disagree case, and a HAND-COMPUTED value;
  * a negative control (`test_negative_control_...`) builds deliberately broken
    metric variants and asserts they produce a DIFFERENT, wrong value — that is
    the point of the control: it proves the assertions above are sensitive, not
    tautological.
"""

from __future__ import annotations

import pytest

from evaluation.metrics import vqa
from evaluation.metrics.vqa import (
    NORMALIZATION_RULES,
    answer_accuracy,
    exact_match,
    normalized_exact_match,
    normalize_answer,
    token_f1,
)


# ---------------------------------------------------------------------------
# R1-R6 — each rule pinned by a case only that rule changes
# ---------------------------------------------------------------------------
def test_r1_lowercase() -> None:
    # Only R1 changes "CAT" -> "cat".
    assert normalize_answer("CAT") == "cat"


def test_r2_contractions() -> None:
    # Only R2 changes "dont" -> "don't"; the apostrophe is not stripped.
    assert normalize_answer("dont") == "don't"
    # ...and the two spellings therefore converge.
    assert normalize_answer("dont") == normalize_answer("don't")


def test_r3_number_words_to_digits() -> None:
    # Only R3 changes "two" -> "2".
    assert normalize_answer("two") == "2"
    assert normalize_answer("ten cats") == "10 cats"


def test_r4_removes_articles() -> None:
    # Only R4 removes the article; assert its EFFECT (absence) so the rule is
    # pinned independently of R6's whitespace collapsing.
    assert "the" not in normalize_answer("the cat").split()
    assert normalize_answer("a cat").split() == ["cat"]
    assert normalize_answer("an owl").split() == ["owl"]


def test_r5_strips_punctuation() -> None:
    # Only R5 changes these.
    assert normalize_answer("cat.") == "cat"
    assert normalize_answer("cat!") == "cat"
    assert normalize_answer("(cat)") == "cat"


def test_r5_preserves_digit_internal_separators() -> None:
    # R5 must NOT destroy a decimal point or a thousands separator.
    assert normalize_answer("3.14") == "3.14"
    assert normalize_answer("1,000") == "1,000"


def test_r6_collapses_whitespace_and_strips() -> None:
    # Only R6 changes the run of spaces.
    assert normalize_answer("cat    dog") == "cat dog"
    assert normalize_answer("  cat  ") == "cat"


def test_normalization_rules_are_declared() -> None:
    assert NORMALIZATION_RULES
    assert len(NORMALIZATION_RULES) == 6
    assert all(isinstance(rule, str) and rule for rule in NORMALIZATION_RULES)


def test_package_reexports_the_vqa_public_names() -> None:
    """`evaluation.metrics` must re-export exactly the vqa public names."""
    import evaluation.metrics as metrics

    assert set(metrics.__all__) == set(vqa.__all__)
    for name in vqa.__all__:
        assert getattr(metrics, name) is getattr(vqa, name)


def test_normalize_answer_is_idempotent() -> None:
    for text in ("The Cat!", "  a DOG  ", "dont", "two birds", "3.14"):
        once = normalize_answer(text)
        assert normalize_answer(once) == once


def test_normalize_answer_rejects_non_strings() -> None:
    with pytest.raises(TypeError):
        normalize_answer(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# exact_match / normalized_exact_match
# ---------------------------------------------------------------------------
def test_exact_and_normalized_agree_on_identical_strings() -> None:
    assert exact_match("cat", "cat") == 1.0
    assert normalized_exact_match("cat", "cat") == 1.0


def test_normalized_agrees_where_exact_disagrees_on_case_and_punctuation() -> None:
    # Each difference is isolated so the assertions can actually fail:
    #   * case only        -> a case-INSENSITIVE exact_match would wrongly give 1.0
    #   * punctuation only -> a punctuation-stripping exact_match would wrongly give 1.0
    # The combined fixture below cannot distinguish either on its own (its two
    # strings differ in case AND punctuation at once), which is why it is kept
    # only as a third, weaker case.
    assert exact_match("Cat", "cat") == 0.0  # case only
    assert exact_match("cat!", "cat") == 0.0  # punctuation only
    assert exact_match("The Cat!", "the cat") == 0.0  # both, as before
    assert normalized_exact_match("The Cat!", "the cat") == 1.0


def test_both_disagree_on_genuinely_different_answers() -> None:
    assert exact_match("cat", "dog") == 0.0
    assert normalized_exact_match("cat", "dog") == 0.0


# ---------------------------------------------------------------------------
# token_f1
# ---------------------------------------------------------------------------
def test_token_f1_exact_token_match_is_one() -> None:
    assert token_f1("the cat", "the cat") == pytest.approx(1.0)


def test_token_f1_disjoint_tokens_is_zero() -> None:
    assert token_f1("cat", "dog") == pytest.approx(0.0)


def test_token_f1_hand_computed_partial_overlap() -> None:
    # pred -> ["cat", "sat"] ; gold -> ["cat"]
    #   overlap = 1 ; precision = 1/2 ; recall = 1/1
    #   f1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 2/3
    assert token_f1("the cat sat", "the cat") == pytest.approx(2.0 / 3.0)


def test_token_f1_respects_token_multiplicity() -> None:
    # "cat cat" vs "cat": overlap 1 ; precision 1/2 ; recall 1/1 ; f1 2/3.
    # A set-based implementation would wrongly return 1.0 here.
    assert token_f1("cat cat", "cat") == pytest.approx(2.0 / 3.0)


def test_token_f1_both_empty_is_one() -> None:
    assert token_f1("the", "a") == pytest.approx(1.0)  # both normalize to ""


def test_token_f1_one_side_empty_is_zero() -> None:
    assert token_f1("cat", "the") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# answer_accuracy
# ---------------------------------------------------------------------------
def test_answer_accuracy_hand_computed() -> None:
    preds = ["cat", "a dog", "bird"]
    golds = ["cat", "dog", "fish"]
    # normalized: ("cat"=="cat") 1 ; ("dog"=="dog") 1 ; ("bird"=="fish") 0 -> 2/3
    assert answer_accuracy(preds, golds) == pytest.approx(2.0 / 3.0)


def test_answer_accuracy_all_correct_is_one() -> None:
    assert answer_accuracy(["cat", "dog"], ["cat", "dog"]) == pytest.approx(1.0)


def test_answer_accuracy_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        answer_accuracy(["cat"], ["cat", "dog"])


def test_answer_accuracy_empty_is_zero() -> None:
    assert answer_accuracy([], []) == 0.0


# ---------------------------------------------------------------------------
# Negative control — the assertions above are shown to be sensitive
# ---------------------------------------------------------------------------
def _broken_token_f1_set_based(pred: str, gold: str) -> float:
    """Deliberately WRONG token F1: set-based, so it ignores multiplicity."""
    pred_set = set(normalize_answer(pred).split())
    gold_set = set(normalize_answer(gold).split())
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    overlap = len(pred_set & gold_set)
    precision = overlap / len(pred_set)
    recall = overlap / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _broken_exact_match_case_insensitive(pred: str, gold: str) -> float:
    """Deliberately WRONG exact match: case-folds, so it is no longer exact."""
    return 1.0 if pred.lower() == gold.lower() else 0.0


def _broken_normalize_case_only(text: str) -> str:
    """Deliberately WRONG normalization: lowercases but strips nothing else."""
    return text.lower().strip()


def test_negative_control_broken_variants_produce_wrong_values() -> None:
    """THE POINT OF THIS CONTROL: prove the real assertions are sensitive.

    Each block builds a deliberately broken variant, shows it returns a value
    that DIFFERS from the real metric on a fixture the real tests rely on, and
    then asserts that a naive equality check between the two actually raises.
    If the real metric were ever silently weakened into the broken variant, the
    corresponding real test above would start failing — that is what makes those
    assertions load-bearing rather than tautological.
    """
    # (1) token_f1 must respect multiplicity.
    real_f1 = token_f1("cat cat", "cat")
    broken_f1 = _broken_token_f1_set_based("cat cat", "cat")
    assert real_f1 == pytest.approx(2.0 / 3.0)
    assert broken_f1 == pytest.approx(1.0)
    assert broken_f1 != real_f1
    with pytest.raises(AssertionError):
        assert broken_f1 == real_f1

    # (2) normalization must strip more than case.
    assert normalize_answer("The Cat!") == "cat"
    assert _broken_normalize_case_only("The Cat!") == "the cat!"
    assert _broken_normalize_case_only("The Cat!") != normalize_answer("The Cat!")

    # (3) exact_match must stay case-sensitive (or it duplicates the normalized
    #     variant and the plan's two distinct metrics collapse into one).
    assert exact_match("Cat", "cat") == 0.0
    assert _broken_exact_match_case_insensitive("Cat", "cat") == 1.0
    with pytest.raises(AssertionError):
        assert _broken_exact_match_case_insensitive("Cat", "cat") == exact_match(
            "Cat", "cat"
        )
