"""R-02 test areas A-D — the CDVQA answer space and question ontology.

    A  vocabulary derivation and integrity      (the 19-answer closed set)
    B  question-type resolution                 (free-form text -> one of eight)
    C  temporal-reference extraction            (pre / post / unspecified)
    D  legal-answer masking                     (never an all-False mask)

WHY THESE FOUR ARE SEPARATE AREAS
---------------------------------
They fail differently and they are consumed by different components:

    A  a wrong vocabulary silently re-indexes every label in the corpus
    B  a wrong resolver mis-conditions the head on the wrong question type
    C  a wrong temporal reference changes the correct answer for the SAME scene
    D  a wrong mask turns a legitimate answer into a confident index-0 answer

A and D are the two places where a defect does not crash and does not look like
a defect: A shifts the label space, D collapses the argmax. Both are pinned here
by the property that makes them safe rather than by an example.

NOTE ON THE LETTER LABELS
-------------------------
The R-02 brief specified "20 named test areas A-R" and the enumeration was not
preserved in any repository artifact, so the letter-to-area mapping below is a
reconstruction anchored to the brief's requirement sections. The areas
themselves cover the requirement surface; the letters are bookkeeping.
`A-R` is 18 letters, not 20, and this file set implements exactly A through R
rather than inventing two extra letters to reach a rounder number.
"""

from __future__ import annotations

import pytest

from training.change_vqa import prompts as P
from training.change_vqa import vocab as V


# ===========================================================================
# Area A - vocabulary derivation and integrity
# ===========================================================================
def test_a_vocabulary_is_the_frozen_union_of_the_three_sub_vocabularies() -> None:
    """19 = 2 (yes/no) + 11 (ratio bins) + 6 (change classes).

    Pinned against the frozen constants rather than against a literal set: this
    is the assertion that fails if the vocabulary is ever re-derived from the
    answers OBSERVED in a split, which is the corruption the module docstring
    forbids.
    """
    assert V.N_ANSWERS == 19
    assert len(V.ANSWER_VOCABULARY) == V.N_ANSWERS
    assert set(V.ANSWER_VOCABULARY) == (
        set(V.YES_NO) | set(V.RATIO_BINS) | set(V.CHANGE_CLASS_ORDER)
    )
    assert len(V.YES_NO) + len(V.RATIO_BINS) + len(V.CHANGE_CLASS_ORDER) == 19


def test_a_vocabulary_is_sorted_so_indices_are_stable() -> None:
    """The index of an answer must not depend on the order answers were found."""
    assert V.ANSWER_VOCABULARY == tuple(sorted(V.ANSWER_VOCABULARY))
    assert len(set(V.ANSWER_VOCABULARY)) == V.N_ANSWERS


def test_a_answer_index_maps_are_mutual_inverses() -> None:
    for index, answer in enumerate(V.INDEX_TO_ANSWER):
        assert V.ANSWER_TO_INDEX[answer] == index
    assert set(V.ANSWER_TO_INDEX) == set(V.ANSWER_VOCABULARY)


def test_a_ratio_bins_are_in_numeric_order_not_lexical() -> None:
    """`"10_to_20" < "0_to_10"` lexicographically; a string sort is off by eight."""
    assert V.RATIO_BINS[0] == "0"
    numeric = [int(name.split("_to_")[0]) for name in V.RATIO_BINS[1:]]
    assert numeric == sorted(numeric)
    assert numeric == list(range(0, 100, 10))


def test_a_per_class_bins_are_a_strict_subset_of_the_global_bins() -> None:
    assert len(V.RATIO_TYPE_BINS) == 9
    assert set(V.RATIO_TYPE_BINS) < set(V.RATIO_BINS)
    # The two bins the per-class vocabulary does not carry.
    assert set(V.RATIO_BINS) - set(V.RATIO_TYPE_BINS) == {"80_to_90", "90_to_100"}


def test_a_validate_vocabulary_reports_the_numbers_it_asserted() -> None:
    report = V.validate_vocabulary()
    assert report["n_answers"] == 19
    assert report["n_question_types"] == 8
    assert report["n_classes"] == 6
    assert report["ratio_bins"] == list(V.RATIO_BINS)
    assert set(report["per_type_vocabulary_size"]) == set(V.QUESTION_TYPE_ORDER)


def test_a_validate_vocabulary_fires_when_the_ontology_drifts(monkeypatch) -> None:
    """The gate must be proven to fire, not assumed to.

    Dropping one bin makes the union 18 while `ANSWER_VOCABULARY` still holds 19,
    which is exactly the drift `validate_vocabulary` exists to catch.
    """
    monkeypatch.setattr(V, "RATIO_BINS", V.RATIO_BINS[:-1])
    with pytest.raises(AssertionError, match="answer vocabulary has"):
        V.validate_vocabulary()


def test_a_validate_vocabulary_fires_when_a_class_disappears(monkeypatch) -> None:
    monkeypatch.setattr(V, "CHANGE_CLASS_ORDER", V.CHANGE_CLASS_ORDER[:-1])
    with pytest.raises(AssertionError, match="answer vocabulary has"):
        V.validate_vocabulary()


def test_a_answers_outside_vocabulary_returns_only_the_outsiders() -> None:
    assert V.answers_outside_vocabulary(["yes", "no", "buildings"]) == []
    assert V.answers_outside_vocabulary(["yes", "perhaps", "perhaps"]) == ["perhaps"]


def test_a_question_type_slots_reserve_a_slot_for_the_unresolvable_case() -> None:
    assert V.N_QUESTION_TYPE_SLOTS == 9
    assert V.QUESTION_TYPE_SLOTS[:8] == V.QUESTION_TYPE_ORDER
    assert V.QUESTION_TYPE_SLOTS[-1] == V.UNKNOWN_QUESTION_TYPE
    assert V.UNKNOWN_QUESTION_TYPE_INDEX == 8


# ===========================================================================
# Area B - question-type resolution
# ===========================================================================
def test_b_every_canonical_template_resolves_back_to_its_own_type() -> None:
    """A round trip across two modules: the phrasings and the resolver must agree.

    This is the load-bearing test for area B. If a template is edited and the
    resolver is not, the head is conditioned on the wrong type slot and the
    answer is computed from the wrong question -- with no error anywhere.
    """
    for qtype in V.QUESTION_TYPE_ORDER:
        question = P.canonical_query(qtype)
        resolved, ok = V.resolve_question_type(question)
        assert ok is True, f"{qtype} template did not resolve: {question!r}"
        assert resolved == qtype, (
            f"{qtype} template resolved as {resolved!r}: {question!r}"
        )


def test_b_largest_change_is_not_swallowed_by_the_generic_change_rule() -> None:
    """"largest change" contains "change", so rule ORDER is load-bearing."""
    assert V.resolve_question_type("What is the largest change?")[0] == "largest_change"
    assert V.resolve_question_type("What type of change is the smallest?")[0] == (
        "smallest_change"
    )


def test_b_per_class_ratio_is_distinguished_from_the_global_ratio() -> None:
    """Both are ratio questions; only one names a class."""
    per_class = "What is the change percentage of buildings in the pre-change image?"
    global_ = "What is the percentage of changed regions?"
    assert V.resolve_question_type(per_class)[0] == "change_ratio_types"
    assert V.resolve_question_type(global_)[0] == "change_ratio"


def test_b_resolution_is_case_insensitive() -> None:
    lower = V.resolve_question_type("did the areas of buildings increase?")
    upper = V.resolve_question_type("DID THE AREAS OF BUILDINGS INCREASE?")
    assert lower == upper == ("increase_or_not", True)


def test_b_an_unresolvable_question_is_reported_as_unresolved_not_guessed() -> None:
    """The free-form path must be able to say "I do not know the type"."""
    for question in ("How many cars are parked here?", "Describe this scene.", ""):
        resolved, ok = V.resolve_question_type(question)
        assert ok is False, question
        assert resolved == V.UNKNOWN_QUESTION_TYPE


def test_b_every_type_in_the_ontology_has_a_template() -> None:
    report = P.validate_templates()
    assert report["n_types"] == 8
    assert set(P.TEMPLATES) == set(V.QUESTION_TYPE_ORDER)


# ===========================================================================
# Area C - temporal reference
# ===========================================================================
def test_c_pre_and_post_references_are_extracted() -> None:
    assert V.resolve_temporal_reference(
        "What is the change percentage of buildings in the pre-change image?"
    ) == "pre"
    assert V.resolve_temporal_reference(
        "What is the largest change in the post-change image?"
    ) == "post"


def test_c_post_change_is_never_read_as_pre() -> None:
    """"post-change" contains "change"; a naive scan could return "pre"."""
    for question in (
        "What changed in the post-change image?",
        "What is the largest change in the post-event image?",
        "Change in the second image?",
    ):
        assert V.resolve_temporal_reference(question) == "post", question


def test_c_a_question_naming_both_acquisitions_is_unspecified() -> None:
    """There is no such row in the dataset; a free-form query could be one.

    Preferring one silently would make the answer depend on marker order rather
    than on what was asked.
    """
    assert V.resolve_temporal_reference(
        "Compare the pre-change image with the post-change image."
    ) == "unspecified"


def test_c_a_question_naming_no_acquisition_is_unspecified() -> None:
    assert V.resolve_temporal_reference("What is the largest change?") == (
        "unspecified"
    )
    assert V.resolve_temporal_reference("") == "unspecified"


def test_c_temporal_slots_are_three_with_unspecified_at_index_zero() -> None:
    assert V.N_TEMPORAL_REFERENCE_SLOTS == 3
    assert V.TEMPORAL_REFERENCE_ORDER == ("unspecified", "pre", "post")
    assert V.TEMPORAL_REFERENCE_TO_INDEX["unspecified"] == 0


def test_c_change_class_resolution_prefers_the_longest_surface_form() -> None:
    """The real hazard: "low vegetation" must not be matched as "vegetation"."""
    assert V.resolve_change_class(
        "Did the areas of low vegetation change?"
    ) == "low_vegetation"
    assert V.resolve_change_class(
        "Did the areas of non-vegetated ground surface change?"
    ) == "NVG_surface"
    assert V.resolve_change_class("What is the largest change?") is None


# ===========================================================================
# Area D - legal-answer masking
# ===========================================================================
def test_d_the_mask_width_is_the_full_answer_space() -> None:
    for qtype in V.QUESTION_TYPE_ORDER:
        mask = V.legal_answer_mask(qtype)
        assert mask is not None
        assert len(mask) == V.N_ANSWERS


def test_d_the_mask_admits_exactly_the_types_frozen_vocabulary() -> None:
    for qtype in V.QUESTION_TYPE_ORDER:
        mask = V.legal_answer_mask(qtype)
        admitted = {
            answer for answer, legal in zip(V.ANSWER_VOCABULARY, mask) if legal
        }
        assert admitted == set(V.LEGAL_ANSWERS[qtype]), qtype


def test_d_an_unknown_type_returns_no_mask_rather_than_an_empty_one() -> None:
    """THE critical property of area D.

    An all-False mask would make every answer illegal and collapse the argmax to
    index 0 -- a confident wrong answer with no trace of why. `None` means "do
    not mask", which is the only safe answer for an unknown type.
    """
    assert V.legal_answer_mask(V.UNKNOWN_QUESTION_TYPE) is None
    assert V.legal_answer_mask("not_a_real_type") is None


def test_d_no_known_type_ever_produces_an_all_false_mask() -> None:
    for qtype in V.QUESTION_TYPE_ORDER:
        mask = V.legal_answer_mask(qtype)
        assert mask is not None
        assert any(mask), f"{qtype} has no legal answer; argmax would collapse to 0"


def test_d_the_legal_answer_table_covers_exactly_the_ontology() -> None:
    assert set(V.LEGAL_ANSWERS) == set(V.QUESTION_TYPE_ORDER)
    for qtype, answers in V.LEGAL_ANSWERS.items():
        assert answers, qtype
        assert answers <= set(V.ANSWER_VOCABULARY), qtype


# ===========================================================================
# Area D (cont.) - ratio binning, which feeds the ratio answers
# ===========================================================================
def test_d_ratio_bins_are_half_open_with_exact_zero_its_own_bin() -> None:
    assert V.ratio_bin_for(0.0) == "0"
    assert V.ratio_bin_for(0.0001) == "0_to_10"
    # Exactly on an interior edge belongs to the UPPER bin.
    assert V.ratio_bin_for(0.10) == "10_to_20"
    assert V.ratio_bin_for(0.0999) == "0_to_10"


def test_d_ratio_binning_saturates_rather_than_raising_at_the_top() -> None:
    """The per-class vocabulary has no bin above 80, so the top saturates there."""
    assert V.ratio_bin_for(1.0) == "90_to_100"
    assert V.ratio_bin_for(0.95, per_class=True) == "70_to_80"
    # Out-of-range input is clamped, not indexed off the end.
    assert V.ratio_bin_for(-5.0) == "0"
    assert V.ratio_bin_for(5.0) == "90_to_100"
