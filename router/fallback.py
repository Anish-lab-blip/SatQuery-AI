"""SatQuery AI — deterministic lexical intent fallback.

The learned router handles natural phrasing. This module exists for the case
the router itself flags as low-confidence, and for environments where the
encoder cannot be loaded at all (a CPU-only Space with no cached weights).

Design rules:

  * Purely lexical. No model, no embeddings, no randomness.
  * Ordered rules, highest specificity first. The first match wins.
  * Never invents capability. If nothing matches, it returns `unsupported`
    with low confidence rather than guessing a task.
  * Must agree with the learned router on the curated hard negatives. If the
    two disagree on `"describe the water body"` vs `"show me the water body"`,
    the fallback is wrong, not the router.
"""

from __future__ import annotations

from dataclasses import dataclass

from router.label_space import (
    TASK_CLASSES,
    is_valid_modality,
    is_valid_task,
)


@dataclass(frozen=True)
class LexicalMatch:
    """One rule hit, with the evidence that produced it."""

    task: str
    modality: str
    temporal: bool
    spatial_output: bool
    language_output: bool
    confidence: float
    rule: str
    matched_terms: tuple[str, ...]


# ---------------------------------------------------------------------------
# Term tables
# ---------------------------------------------------------------------------

#: "show me where" phrasings. Spatial intent.
_SPATIAL_TERMS: tuple[str, ...] = (
    # Bare "where" is included deliberately. "Where did the change happen?"
    # must resolve to change + spatial_output=True, and the temporal branch
    # consults this table only AFTER the temporal terms have already matched.
    # Without it, "where did" carries no spatial signal at all.
    "where", "where is", "where are", "where's",
    "locate", "location of",
    "show me where", "show where",
    "highlight", "mark the", "point out", "point to",
    "draw a box", "draw box", "bounding box",
    "find the", "find all",
    "show me the", "show the region",
)

#: Temporal phrasings.
_TEMPORAL_TERMS: tuple[str, ...] = (
    "changed", "change", "changes", "changing",
    "before and after", "before-and-after",
    "between these two", "between the two",
    # "Compare these two images." is a temporal request. The superficially
    # similar "Compare the optical and radar images." is claimed by the
    # dual-modality table, which is consulted first.
    "these two images", "two images", "compare these", "compare the two",
    "over time", "temporal", "time series", "time-series",
    "differences between the two dates", "two dates",
    "has the", "did the", "have the",
    "compared to before", "since then",
)

#: Dual-modality phrasings.
_DUAL_MODALITY_TERMS: tuple[str, ...] = (
    "optical and sar", "optical and radar", "sar and optical",
    "radar and optical", "radar and optical", "both images",
    "both sensors", "both modalities", "two modalities",
    "sar image", "radar image", "radar data", "sar data",
    "co-registered", "coregistered", "fuse", "fusion",
    "jointly", "multi-modal", "multimodal",
)

#: Caption phrasings.
_CAPTION_TERMS: tuple[str, ...] = (
    "describe", "description", "caption", "write a caption",
    "summarize", "summarise", "summary of",
    "tell me about this image", "what do you see",
    "what can you see", "explain this image",
)

#: VQA phrasings.
_VQA_TERMS: tuple[str, ...] = (
    # Bare "what" is needed because "What land cover is visible?" carries no
    # other interrogative marker. Precedence protects it: the temporal,
    # spatial and caption tables are all consulted before this one, so
    # "what changed" and "what do you see" never reach here.
    "what",
    "how many", "how much", "is there", "are there",
    "what is", "what are", "what kind", "what type",
    "which", "does this", "do you see", "can you tell",
    "is this", "are these", "identify the type",
)

#: SAR-only phrasings.
_SAR_TERMS: tuple[str, ...] = (
    "sar image", "sar data", "radar image", "radar data",
    "synthetic aperture radar", "backscatter",
)

#: Optical-only phrasings.
_OPTICAL_TERMS: tuple[str, ...] = (
    "optical image", "optical data", "multispectral", "true colour",
    "true color", "rgb image", "visible image",
)


def _hits(text: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(term for term in terms if term in text)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


# ---------------------------------------------------------------------------
# The fallback
# ---------------------------------------------------------------------------


def lexical_route(query: str) -> LexicalMatch:
    """Classify a query with ordered lexical rules.

    Rule order encodes precedence, which matters because the phrasings overlap:

        "show me where the change happened"
            has both a spatial term AND a temporal term -> change + spatial

        "compare optical and radar to locate built-up areas"
            has dual-modality AND spatial -> optical_sar (spatial doesn't apply
            to the joint workflow, whose output is a classification)

    Precedence: dual_modality > temporal > spatial > caption > vqa > unsupported
    """
    if not isinstance(query, str) or not query.strip():
        return LexicalMatch(
            task="unsupported",
            modality="unknown",
            temporal=False,
            spatial_output=False,
            language_output=False,
            confidence=0.0,
            rule="empty_query",
            matched_terms=(),
        )

    text = query.lower().strip()

    dual_hits = _hits(text, _DUAL_MODALITY_TERMS)
    temporal_hits = _hits(text, _TEMPORAL_TERMS)
    spatial_hits = _hits(text, _SPATIAL_TERMS)
    caption_hits = _hits(text, _CAPTION_TERMS)
    vqa_hits = _hits(text, _VQA_TERMS)
    sar_hits = _hits(text, _SAR_TERMS)
    optical_hits = _hits(text, _OPTICAL_TERMS)

    # -- 1. dual modality wins outright ---------------------------------
    # "compare optical and radar" is optical_sar even though it also contains
    # no temporal marker; the modality word is the decisive evidence.
    if dual_hits:
        # "both images" alone is ambiguous; require a modality word too.
        explicit_pair = any(
            term in dual_hits
            for term in ("optical and sar", "optical and radar", "sar and optical",
                         "radar and optical", "both sensors", "both modalities",
                         "two modalities", "fuse", "fusion", "jointly",
                         "multi-modal", "multimodal", "co-registered", "coregistered")
        )
        if explicit_pair:
            return LexicalMatch(
                task="optical_sar",
                modality="optical_sar",
                temporal=False,
                spatial_output=False,
                language_output=True,
                confidence=0.92,
                rule="dual_modality",
                matched_terms=dual_hits,
            )

    # -- 2. temporal ----------------------------------------------------
    if temporal_hits:
        spatial = _contains_any(text, _SPATIAL_TERMS)
        return LexicalMatch(
            task="change",
            modality="optical",
            temporal=True,
            spatial_output=spatial,
            language_output=True,
            confidence=0.90 if spatial else 0.85,
            rule="temporal_spatial" if spatial else "temporal",
            matched_terms=temporal_hits + spatial_hits,
        )

    # -- 3. spatial / grounding -----------------------------------------
    if spatial_hits:
        return LexicalMatch(
            task="grounding",
            modality="optical",
            temporal=False,
            spatial_output=True,
            language_output=True,
            confidence=0.88,
            rule="spatial",
            matched_terms=spatial_hits,
        )

    # -- 4. SAR-only ----------------------------------------------------
    if sar_hits and not optical_hits:
        return LexicalMatch(
            task="vqa",
            modality="sar",
            temporal=False,
            spatial_output=False,
            language_output=True,
            confidence=0.75,
            rule="sar_single",
            matched_terms=sar_hits,
        )

    # -- 5. optical-only ------------------------------------------------
    if optical_hits and not sar_hits:
        return LexicalMatch(
            task="vqa",
            modality="optical",
            temporal=False,
            spatial_output=False,
            language_output=True,
            confidence=0.72,
            rule="optical_single",
            matched_terms=optical_hits,
        )

    # -- 6. caption -----------------------------------------------------
    if caption_hits:
        return LexicalMatch(
            task="caption",
            modality="unknown",
            temporal=False,
            spatial_output=False,
            language_output=True,
            confidence=0.85,
            rule="caption",
            matched_terms=caption_hits,
        )

    # -- 7. vqa ---------------------------------------------------------
    if vqa_hits:
        return LexicalMatch(
            task="vqa",
            modality="unknown",
            temporal=False,
            spatial_output=False,
            language_output=True,
            confidence=0.78,
            rule="vqa",
            matched_terms=vqa_hits,
        )

    # -- 8. nothing matched: refuse rather than guess -------------------
    return LexicalMatch(
        task="unsupported",
        modality="unknown",
        temporal=False,
        spatial_output=False,
        language_output=False,
        confidence=0.30,
        rule="no_match",
        matched_terms=(),
    )


def is_available() -> bool:
    """The fallback has no model dependency and is always available."""
    return True


def self_check() -> list[str]:
    """Assert the fallback agrees with the curated hard negatives.

    Returns a list of failures. An empty list means the fallback is internally
    consistent with the label space.
    """
    cases: tuple[tuple[str, str, bool | None], ...] = (
        ("Describe this image.", "caption", None),
        ("What land cover is visible?", "vqa", None),
        ("Show me the water body.", "grounding", None),
        ("Describe the water body.", "caption", None),
        ("What changed between these images?", "change", None),
        ("Where did the change happen?", "change", True),
        ("What changed?", "change", False),
        ("Compare the optical and radar images.", "optical_sar", None),
        ("Locate the buildings.", "grounding", None),
        ("Book me a flight to Delhi.", "unsupported", None),
    )

    failures: list[str] = []
    for query, expected_task, expected_spatial in cases:
        match = lexical_route(query)
        if match.task != expected_task:
            failures.append(
                f"{query!r}: expected task {expected_task!r}, got {match.task!r} "
                f"(rule={match.rule})"
            )
        if expected_spatial is not None and match.spatial_output != expected_spatial:
            failures.append(
                f"{query!r}: expected spatial_output={expected_spatial}, "
                f"got {match.spatial_output}"
            )
        if not is_valid_task(match.task):
            failures.append(f"{query!r}: produced invalid task {match.task!r}")
        if not is_valid_modality(match.modality):
            failures.append(f"{query!r}: produced invalid modality {match.modality!r}")
        if match.task not in TASK_CLASSES:
            failures.append(f"{query!r}: task outside the label space")

    return failures


__all__ = ["LexicalMatch", "lexical_route", "is_available", "self_check"]