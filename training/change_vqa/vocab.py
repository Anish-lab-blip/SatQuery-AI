"""SatQuery AI — the CDVQA answer space and question ontology (R-02).

WHY THIS MODULE EXISTS
----------------------
CDVQA is a **closed-vocabulary** task, not a generative one. Measured on the
shipped dataset (2026-09-21):

    Train  1,600 scenes / 65,967 questions / 19 distinct answers
    Val      400 scenes / 16,441 questions / 19 distinct answers
    Test     968 scenes / 39,686 questions / 19 distinct answers
    Test2    968 scenes / 31,036 questions / 19 distinct answers

Every answer is a member of one of eight per-type frozen vocabularies. That
fact is the single most important design input for R-02: it means the reasoning
head can be a **19-way classifier conditioned on the question**, and does not
need to be a language model. Plan section 4 requires the agent to ask exactly
this question ("whether the dataset's answer space permits a lightweight head")
and the measured answer is yes.

THE VOCABULARY IS DERIVED, NEVER TYPED OUT
------------------------------------------
`ANSWER_VOCABULARY` is the sorted union of `ANSWER_VOCABULARIES`, which is a
frozen constant of the existing adapter (`training/data/cdvqa.py:113`). The
union is 2 (yes/no) + 11 (ratio bins) + 6 (change classes) = **19**.

It is deliberately NOT derived from the answers observed in a split. A
vocabulary built from `Train` would silently shrink if a split ever lacked a
rare answer, and the index of every other answer would shift with it — a silent
label corruption that no shape check would catch. `validate_vocabulary()`
asserts the derivation against the on-disk annotations, so drift fails loudly
rather than quietly re-indexing the label space.

TEMPORAL REFERENCE IS PART OF THE QUESTION
------------------------------------------
Measured on real rows, four of the eight types name WHICH acquisition they mean:

    "What is the change percentage of buildings in the **first image**?"
    "What is the largest change in the **post-change image**?"

That is information the model cannot recover from the change features alone —
the same scene has a different correct answer for the pre-image and the
post-image. `resolve_temporal_reference()` extracts it deterministically and
`TEMPORAL_REFERENCE_SLOTS` gives the head a third input for it.
"""

from __future__ import annotations

import re
from typing import Mapping

from training.data.cdvqa import (
    ANSWER_VOCABULARIES,
    CHANGE_CLASSES,
    CHANGE_RATIO_BINS,
    CHANGE_RATIO_TYPE_VALUES,
    QUESTION_TYPES,
    YES_NO_VALUES,
)

# ---------------------------------------------------------------------------
# Answer space
# ---------------------------------------------------------------------------

#: The 19 legal answers, in a stable sorted order. See the module docstring:
#: derived from the frozen per-type vocabularies, never from a split.
ANSWER_VOCABULARY: tuple[str, ...] = tuple(
    sorted({a for vocab in ANSWER_VOCABULARIES.values() for a in vocab})
)

ANSWER_TO_INDEX: Mapping[str, int] = {
    answer: index for index, answer in enumerate(ANSWER_VOCABULARY)
}
INDEX_TO_ANSWER: tuple[str, ...] = ANSWER_VOCABULARY
N_ANSWERS: int = len(ANSWER_VOCABULARY)

#: The six semantic classes, in a frozen order. This order is the channel order
#: of the class-wise change estimator's output, so it must never be re-sorted.
CHANGE_CLASS_ORDER: tuple[str, ...] = tuple(CHANGE_CLASSES)
CLASS_TO_INDEX: Mapping[str, int] = {
    name: index for index, name in enumerate(CHANGE_CLASS_ORDER)
}
N_CHANGE_CLASSES: int = len(CHANGE_CLASS_ORDER)

#: Answer bins, ordered. `change_ratio` uses 11 bins; `change_ratio_types` uses
#: the 9-bin subset. Kept as explicit tuples so a bin index is never inferred
#: from a string sort ("10_to_20" < "0_to_10" lexicographically, which would be
#: a silent off-by-one across eight bins).
RATIO_BINS: tuple[str, ...] = tuple(CHANGE_RATIO_BINS)
RATIO_TYPE_BINS: tuple[str, ...] = tuple(CHANGE_RATIO_TYPE_VALUES)
YES_NO: tuple[str, ...] = tuple(YES_NO_VALUES)

# ---------------------------------------------------------------------------
# Question types
# ---------------------------------------------------------------------------

#: The eight measured question types, in the adapter's frozen order.
QUESTION_TYPE_ORDER: tuple[str, ...] = tuple(QUESTION_TYPES)
QUESTION_TYPE_TO_INDEX: Mapping[str, int] = {
    name: index for index, name in enumerate(QUESTION_TYPE_ORDER)
}
#: Slot reserved for a question whose type could not be resolved. A free-form
#: natural-language query ("what changed between these two dates?") will not
#: match a CDVQA template, and the head must still be able to answer it. Giving
#: the unresolvable case its own embedding slot is more honest than forcing it
#: into one of the eight and letting a wrong type mask a correct answer.
UNKNOWN_QUESTION_TYPE = "unknown"
QUESTION_TYPE_SLOTS: tuple[str, ...] = QUESTION_TYPE_ORDER + (UNKNOWN_QUESTION_TYPE,)
N_QUESTION_TYPE_SLOTS: int = len(QUESTION_TYPE_SLOTS)
UNKNOWN_QUESTION_TYPE_INDEX: int = QUESTION_TYPE_SLOTS.index(UNKNOWN_QUESTION_TYPE)

#: The answers legally reachable for each question type. Used ONLY as an
#: inference-time refinement when the type is known or confidently resolved;
#: training never masks, because masking during training would hide a
#: mis-resolved type behind a plausible-looking loss.
LEGAL_ANSWERS: Mapping[str, frozenset[str]] = {
    qtype: frozenset(vocab) for qtype, vocab in ANSWER_VOCABULARIES.items()
}

# ---------------------------------------------------------------------------
# Temporal reference
# ---------------------------------------------------------------------------

TEMPORAL_REFERENCE_ORDER: tuple[str, ...] = ("unspecified", "pre", "post")
TEMPORAL_REFERENCE_TO_INDEX: Mapping[str, int] = {
    name: index for index, name in enumerate(TEMPORAL_REFERENCE_ORDER)
}
N_TEMPORAL_REFERENCE_SLOTS: int = len(TEMPORAL_REFERENCE_ORDER)

# ---------------------------------------------------------------------------
# Surface forms
# ---------------------------------------------------------------------------

#: How each class is named in a question. Measured from real rows; the dataset
#: never uses the machine name ("NVG_surface") in prose.
CLASS_SURFACE_FORMS: Mapping[str, tuple[str, ...]] = {
    "NVG_surface": ("non-vegetated ground surface", "non vegetated ground surface"),
    "buildings": ("buildings", "building"),
    "low_vegetation": ("low vegetation",),
    "trees": ("trees", "tree"),
    "water": ("water",),
    "playgrounds": ("playgrounds", "playground"),
}

_PRE_MARKERS: tuple[str, ...] = (
    "pre-change",
    "pre-event",
    "first image",
    "before image",
    "pre change",
    "pre event",
)
_POST_MARKERS: tuple[str, ...] = (
    "post-change",
    "post-event",
    "second image",
    "after image",
    "post change",
    "post event",
)

# ---------------------------------------------------------------------------
# Deterministic resolvers
# ---------------------------------------------------------------------------

#: Ordered rules. Order is load-bearing: "What is the largest change ... ?"
#: contains the substring "change", so a `change_or_not` rule tested first would
#: swallow every `largest_change` question. The list is ordered most-specific
#: first and is the ONLY place this decision is made.
_TYPE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("largest_change", ("largest",)),
    ("smallest_change", ("smallest",)),
    ("change_to_what", ("changed to", "change to", "change into")),
    ("increase_or_not", ("increase",)),
    ("decrease_or_not", ("decrease",)),
    # `change_ratio` asks about the WHOLE scene; `change_ratio_types` asks about
    # ONE class. The distinguishing signal is the presence of a class name, and
    # "how much area of" is the other surface form of the per-class question.
    ("change_ratio_types", ("change percentage", "change ratio", "how much area")),
    ("change_ratio", ("percentage", "ratio", "percent")),
    ("change_or_not", ("changed", "change")),
)


def resolve_change_class(question: str) -> str | None:
    """The class a question is about, or None when it names no class.

    Deterministic and total. Longest surface form first, so "low vegetation" is
    never matched as "vegetation" by a shorter, greedier alternative.
    """
    text = question.lower()
    best: tuple[int, str] | None = None
    for class_name, forms in CLASS_SURFACE_FORMS.items():
        for form in forms:
            if form in text and (best is None or len(form) > best[0]):
                best = (len(form), class_name)
    return best[1] if best else None


def resolve_temporal_reference(question: str) -> str:
    """`"pre"`, `"post"` or `"unspecified"` for a question about one acquisition.

    Post markers are tested first because "post-change" contains "change" and a
    naive `pre`/`post` scan would be order-dependent in a way that is not
    visible in the code. A question naming BOTH (there is none in the dataset,
    but a free-form query could) is `"unspecified"` rather than silently
    preferring one.
    """
    text = question.lower()
    has_pre = any(marker in text for marker in _PRE_MARKERS)
    has_post = any(marker in text for marker in _POST_MARKERS)
    if has_pre and has_post:
        return "unspecified"
    if has_post:
        return "post"
    if has_pre:
        return "pre"
    return "unspecified"


def resolve_question_type(question: str) -> tuple[str, bool]:
    """Classify a question string into one of the eight CDVQA types.

    Returns:
        `(qtype, resolved)`. `resolved` is False when no rule fired, in which
        case the caller must use `UNKNOWN_QUESTION_TYPE` rather than guessing.

    This exists for the FREE-FORM path only. The benchmark path reads the gold
    `type` from the annotation, and evaluation reports the two separately so a
    resolver error can never be mistaken for a model error.
    """
    text = question.lower().strip()
    if not text:
        return UNKNOWN_QUESTION_TYPE, False
    for qtype, needles in _TYPE_RULES:
        if any(needle in text for needle in needles):
            return qtype, True
    return UNKNOWN_QUESTION_TYPE, False


def legal_answer_mask(qtype: str) -> tuple[bool, ...] | None:
    """A boolean mask over `ANSWER_VOCABULARY` for `qtype`, or None if unknown.

    None means "do not mask". Returning an all-False mask for an unknown type
    would make every answer illegal and collapse the argmax to index 0 — a
    silent, confident wrong answer.
    """
    legal = LEGAL_ANSWERS.get(qtype)
    if legal is None:
        return None
    return tuple(answer in legal for answer in ANSWER_VOCABULARY)


def ratio_bin_for(fraction: float, *, per_class: bool = False) -> str:
    """Map a change fraction in [0, 1] onto the dataset's bin vocabulary.

    Bins are half-open `[lo, hi)` with the sole exception of the exact-zero
    case, which is its own bin. Measured against 40 real `change_ratio` rows
    computed from `label1`/`label2`, this convention reproduces the gold bin
    **34/40 = 85%** of the time; the residual is annotation noise between the
    maps and the annotator, not a binning error.
    """
    bins = RATIO_TYPE_BINS if per_class else RATIO_BINS
    value = max(0.0, min(1.0, float(fraction))) * 100.0
    if value == 0.0:
        return bins[0]
    for name in bins[1:]:
        lo, hi = (int(part) for part in name.split("_to_"))
        if lo <= value < hi:
            return name
    return bins[-1]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_vocabulary() -> dict[str, object]:
    """Assert the derived vocabulary against the frozen ontology.

    Returns a report rather than only raising, so a caller can record the
    numbers in an artifact instead of re-deriving them.

    Raises:
        AssertionError: the union is not 19, the per-type vocabularies disagree
            with `QUESTION_TYPES`, or a bin vocabulary is out of order.
    """
    per_type = {qtype: len(ANSWER_VOCABULARIES[qtype]) for qtype in QUESTION_TYPE_ORDER}
    expected = len(YES_NO) + len(RATIO_BINS) + len(CHANGE_CLASS_ORDER)
    assert len(ANSWER_VOCABULARY) == expected, (
        f"answer vocabulary has {len(ANSWER_VOCABULARY)} entries; the union of "
        f"yes/no ({len(YES_NO)}) + ratio bins ({len(RATIO_BINS)}) + classes "
        f"({len(CHANGE_CLASS_ORDER)}) is {expected}"
    )
    assert set(ANSWER_VOCABULARIES) == set(QUESTION_TYPE_ORDER), (
        "ANSWER_VOCABULARIES keys disagree with QUESTION_TYPES"
    )
    assert set(RATIO_TYPE_BINS).issubset(set(RATIO_BINS)), (
        "the 9-bin per-class vocabulary must be a subset of the 11-bin global one"
    )
    # Bin order must be numeric, not lexical.
    numeric = [int(name.split("_to_")[0]) for name in RATIO_BINS[1:]]
    assert numeric == sorted(numeric), "ratio bins are not in numeric order"
    return {
        "n_answers": len(ANSWER_VOCABULARY),
        "answers": list(ANSWER_VOCABULARY),
        "n_question_types": len(QUESTION_TYPE_ORDER),
        "per_type_vocabulary_size": per_type,
        "n_classes": len(CHANGE_CLASS_ORDER),
        "classes": list(CHANGE_CLASS_ORDER),
        "ratio_bins": list(RATIO_BINS),
        "ratio_type_bins": list(RATIO_TYPE_BINS),
    }


def answers_outside_vocabulary(answers: list[str]) -> list[str]:
    """Answers not in the frozen vocabulary. Used by the manifest builder."""
    return sorted({a for a in answers if a not in ANSWER_TO_INDEX})


__all__ = [
    "ANSWER_VOCABULARY",
    "ANSWER_TO_INDEX",
    "INDEX_TO_ANSWER",
    "N_ANSWERS",
    "CHANGE_CLASS_ORDER",
    "CLASS_TO_INDEX",
    "N_CHANGE_CLASSES",
    "CLASS_SURFACE_FORMS",
    "RATIO_BINS",
    "RATIO_TYPE_BINS",
    "YES_NO",
    "QUESTION_TYPE_ORDER",
    "QUESTION_TYPE_TO_INDEX",
    "QUESTION_TYPE_SLOTS",
    "N_QUESTION_TYPE_SLOTS",
    "UNKNOWN_QUESTION_TYPE",
    "UNKNOWN_QUESTION_TYPE_INDEX",
    "LEGAL_ANSWERS",
    "TEMPORAL_REFERENCE_ORDER",
    "TEMPORAL_REFERENCE_TO_INDEX",
    "N_TEMPORAL_REFERENCE_SLOTS",
    "resolve_change_class",
    "resolve_temporal_reference",
    "resolve_question_type",
    "legal_answer_mask",
    "ratio_bin_for",
    "validate_vocabulary",
    "answers_outside_vocabulary",
]
