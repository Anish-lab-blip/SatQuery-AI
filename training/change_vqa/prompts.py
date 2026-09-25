"""SatQuery AI — question phrasings and the explanation prompt (R-02).

TWO DISTINCT JOBS, DELIBERATELY IN ONE FILE
-------------------------------------------
1. **Canonical question phrasings.** Measured from real dataset rows, per type.
   Used to generate a natural-language query for a change-VQA request (the GUI
   and the planner both need one), to seed the router's query set, and to test
   that `resolve_question_type` agrees with the ontology it claims to model.

2. **The explanation prompt.** Plan section 25 is explicit about the
   specialist/VLM split: *"Do not ask SmolVLM to hallucinate coordinates"* — the
   specialist produces the fact, the language model produces the sentence. The
   same rule applies to the answer. The prompt built here hands the VLM the
   already-computed answer and the already-computed evidence and asks it to
   explain them. It never asks the VLM for the answer, and the caller is
   expected to discard the explanation if the VLM is unavailable — the answer
   stands on its own.

Both jobs are about *how a change question is phrased*, which is why they share
a module rather than each getting a thin file.
"""

from __future__ import annotations

from typing import Any, Mapping

from training.change_vqa.vocab import (
    CHANGE_CLASS_ORDER,
    CLASS_SURFACE_FORMS,
    QUESTION_TYPE_ORDER,
    UNKNOWN_QUESTION_TYPE,
)

#: Canonical surface form for each question type, with a `{class}` slot where
#: the type is class-scoped. Every template below was checked against real
#: dataset rows; `QUESTION_TYPE_ORDER` is the authority for the key set, and
#: `validate_templates()` asserts the two agree.
TEMPLATES: Mapping[str, str] = {
    "change_or_not": "Have the areas of {class} changed?",
    "change_ratio": "What is the percentage of changed regions?",
    "change_ratio_types": (
        "What is the change percentage of {class} in the pre-change image?"
    ),
    "change_to_what": (
        "What have the areas of {class} in the pre-change image mainly changed to?"
    ),
    "increase_or_not": "Did the areas of {class} increase?",
    "decrease_or_not": "Did the areas of {class} decrease?",
    "largest_change": "What is the largest change?",
    "smallest_change": "What type of change is the smallest?",
}

#: Types whose template carries a `{class}` slot.
CLASS_SCOPED_TYPES: tuple[str, ...] = (
    "change_or_not",
    "change_ratio_types",
    "change_to_what",
    "increase_or_not",
    "decrease_or_not",
)

#: Types that do not name a class.
GLOBAL_TYPES: tuple[str, ...] = ("change_ratio", "largest_change", "smallest_change")

#: The default class used when a template needs one and the caller did not
#: supply one. `buildings` is the most common class-scoped subject in Train
#: (4,416 rows) and, more importantly, it is a real class rather than a
#: placeholder — a placeholder would generate queries the resolver cannot map.
DEFAULT_CLASS = "buildings"


def surface_form(change_class: str) -> str:
    """How the dataset names a class in prose.

    Raises:
        KeyError: the class is not one of the six. Silently passing the machine
            name through would produce a query no resolver and no model has
            ever seen.
    """
    forms = CLASS_SURFACE_FORMS.get(change_class)
    if not forms:
        raise KeyError(
            f"unknown change class {change_class!r}; known: {list(CHANGE_CLASS_ORDER)}"
        )
    return forms[0]


def canonical_query(
    qtype: str, *, change_class: str | None = None, fallback_to_default: bool = True
) -> str:
    """A natural-language query for `qtype`.

    Args:
        qtype: one of the eight, or `"unknown"`.
        change_class: required for a class-scoped type. When omitted and
            `fallback_to_default` is True, `DEFAULT_CLASS` is used.
        fallback_to_default: set False to get a `ValueError` instead of a
            default — the strict mode used by callers that must not invent a
            subject.

    Raises:
        KeyError: unknown type.
        ValueError: a class-scoped type with no class and no fallback allowed.
    """
    if qtype == UNKNOWN_QUESTION_TYPE:
        return "What changed between these two images?"
    template = TEMPLATES.get(qtype)
    if template is None:
        raise KeyError(
            f"unknown question type {qtype!r}; known: {list(QUESTION_TYPE_ORDER)}"
        )
    if "{class}" not in template:
        return template
    if change_class is None:
        if not fallback_to_default:
            raise ValueError(
                f"question type {qtype!r} is class-scoped but no change_class "
                f"was supplied"
            )
        change_class = DEFAULT_CLASS
    return template.format(**{"class": surface_form(change_class)})


def validate_templates() -> dict[str, Any]:
    """Assert the template table covers exactly the ontology.

    Raises:
        AssertionError: a type has no template, or a template names a class the
            dataset does not have.
    """
    missing = [t for t in QUESTION_TYPE_ORDER if t not in TEMPLATES]
    assert not missing, f"question types without a template: {missing}"
    extra = [t for t in TEMPLATES if t not in QUESTION_TYPE_ORDER]
    assert not extra, f"templates for unknown question types: {extra}"
    class_scoped = tuple(t for t in QUESTION_TYPE_ORDER if "{class}" in TEMPLATES[t])
    assert set(class_scoped) == set(CLASS_SCOPED_TYPES), (
        f"class-scoped types disagree: templates say {class_scoped}, "
        f"CLASS_SCOPED_TYPES says {CLASS_SCOPED_TYPES}"
    )
    rendered = {
        qtype: canonical_query(qtype, change_class=CHANGE_CLASS_ORDER[0])
        for qtype in QUESTION_TYPE_ORDER
    }
    return {
        "n_types": len(QUESTION_TYPE_ORDER),
        "class_scoped_types": list(class_scoped),
        "global_types": list(GLOBAL_TYPES),
        "rendered_examples": rendered,
    }


# ---------------------------------------------------------------------------
# The explanation prompt
# ---------------------------------------------------------------------------

EXPLANATION_SYSTEM = (
    "You are explaining a remote-sensing change analysis. The change answer and "
    "all spatial facts were computed by deterministic specialists. Restate them "
    "in one or two plain sentences. Do not add locations, counts, or answers of "
    "your own."
)


def build_explanation_prompt(
    *,
    question: str,
    answer: str,
    class_mag: Mapping[str, float] | None = None,
    class_delta: Mapping[str, float] | None = None,
    total_changed: float | None = None,
    confidence: float | None = None,
    extra_facts: Mapping[str, Any] | None = None,
) -> str:
    """Build the VLM explanation prompt from ALREADY-COMPUTED facts.

    The prompt is constructed so that every number in it came from a specialist.
    The model's job is the sentence, not the fact (plan section 25). The answer
    is included as a given, never as a question, so the VLM cannot be mistaken
    for the component that produced it.

    Returns:
        A prompt string. The caller is responsible for the chat template — the
        VLM specialist already enforces that (`vlm.prompt_must_use_chat_template`).
    """
    lines: list[str] = [
        EXPLANATION_SYSTEM,
        "",
        f"Question asked: {question}",
        f"Computed answer: {answer}",
    ]
    if confidence is not None:
        lines.append(f"Raw model confidence: {confidence:.4f} (uncalibrated)")
    if total_changed is not None:
        lines.append(f"Total changed fraction of the scene: {total_changed:.4f}")
    if class_mag:
        ranked = sorted(class_mag.items(), key=lambda kv: -abs(kv[1]))[:3]
        lines.append(
            "Per-class change magnitude (top 3): "
            + ", ".join(f"{name}={value:.4f}" for name, value in ranked)
        )
    if class_delta:
        ranked = sorted(class_delta.items(), key=lambda kv: -abs(kv[1]))[:3]
        lines.append(
            "Signed per-class change (top 3): "
            + ", ".join(f"{name}={value:+.4f}" for name, value in ranked)
        )
    if extra_facts:
        for key, value in sorted(extra_facts.items()):
            lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("Explain the computed answer in one or two sentences.")
    return "\n".join(lines)


__all__ = [
    "TEMPLATES",
    "CLASS_SCOPED_TYPES",
    "GLOBAL_TYPES",
    "DEFAULT_CLASS",
    "EXPLANATION_SYSTEM",
    "surface_form",
    "canonical_query",
    "validate_templates",
    "build_explanation_prompt",
]
