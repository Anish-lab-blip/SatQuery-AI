"""SatQuery AI — optical-SAR prompt templates (frozen before evaluation).

Per freeze section 5, "Prompts are versioned files, frozen before benchmark
evaluation." This module is that file for the optical-SAR workflow.

WHAT THE VLM IS AND IS NOT ALLOWED TO DO HERE
----------------------------------------------
The fused classifier produces the label and the numbers. The VLM's only job is
to put those into a sentence a person can read. It is deliberately not asked:

  * for a confidence -- freeze section 5: "No LLM-generated confidence"
  * for coordinates -- freeze section 5: "No LLM-generated coordinates"
  * for the label itself -- the classifier decided it before the VLM ran

So `build_explanation_prompt` takes the already-computed decision as INPUT and
asks for prose. A VLM asked to classify would produce a plausible label with no
measurement behind it, and that label would sit next to a real confidence score
looking equally authoritative.

The prompt also carries the availability facts, because the single most
important thing a reader needs to know about an optical-SAR result is which
bands were actually measured. "Classified as urban, optical 4 of 12 channels
available" is honest; "classified as urban" alone invites the reader to assume a
full Sentinel-2 scene.

Chat-template discipline (finding F5-3)
---------------------------------------
SmolVLM raises ValueError on prompts without one `<image>` token per image, so
these strings are never passed to the model directly. `render_for_vlm` delegates
to the processor's `apply_chat_template`. Hand-building the token is the exact
mistake F5-3 records.

PROMPT_VERSION is bumped whenever the wording changes, and it travels in the
execution trace so a result can be attributed to the prompt that produced it.
"""

from __future__ import annotations

from typing import Any

#: Bump on ANY wording change. Recorded in the execution trace.
PROMPT_VERSION = "optical_sar_v1"

#: The system instruction. Deliberately constrains the model to narration.
SYSTEM_PROMPT = (
    "You are a remote-sensing analyst assistant. You explain results that have "
    "already been computed by measurement models. Do not invent facts, do not "
    "estimate confidence, and do not state coordinates or locations. If the "
    "supplied facts do not determine the answer, say so."
)


def build_explanation_prompt(
    *,
    predicted_label: str,
    fusion_margin: float,
    optical_available: int,
    optical_total: int,
    sar_available: int,
    sar_total: int,
    query: str = "",
    cross_modal_agreement: float | None = None,
) -> str:
    """Render the narration prompt from ALREADY-COMPUTED facts.

    Every number below was produced by the specialist before this function runs.
    The prompt restates them; it never asks the model to derive them.

    Args:
        predicted_label: the classifier's decision, already made.
        fusion_margin: top-1 minus top-2 probability. A measurement.
        optical_available / optical_total: channel availability, from the mask.
        sar_available / sar_total: polarisation availability, from the mask.
        query: the operator's original question, if any.
        cross_modal_agreement: optical/SAR agreement, when both are present.
    """
    lines = [
        "A prediction has already been made by a measurement model.",
        "",
        f"Predicted class: {predicted_label}",
        f"Fusion margin (top-1 minus top-2 probability): {fusion_margin:.3f}",
        (
            f"Optical channel availability: {optical_available} of "
            f"{optical_total}"
        ),
        f"SAR channel availability: {sar_available} of {sar_total}",
    ]

    if cross_modal_agreement is not None:
        lines.append(f"Cross-modal agreement: {cross_modal_agreement:.3f}")

    if optical_available < optical_total:
        lines.append(
            f"NOTE: {optical_total - optical_available} optical channel(s) were "
            f"not measured by this sensor and are zero-filled. They carry no "
            f"information."
        )
    if sar_available < sar_total:
        lines.append(
            f"NOTE: {sar_total - sar_available} SAR channel(s) were not measured "
            f"and are zero-filled. They carry no information."
        )

    if query.strip():
        lines.append("")
        lines.append(f"The operator asked: {query.strip()}")

    lines.extend(
        [
            "",
            "Write two or three sentences explaining this result to the "
            "operator. Use ONLY the facts above. Do not state a confidence "
            "value and do not state any location or coordinates.",
        ]
    )
    return "\n".join(lines)


def build_messages(prompt: str) -> list[dict[str, Any]]:
    """Wrap a prompt as a chat message for `apply_chat_template`.

    The `<image>` token is inserted by the chat template, never by hand
    (finding F5-3).
    """
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def render_for_vlm(processor: Any, prompt: str) -> Any:
    """Render through the processor's chat template.

    Raises:
        SpecialistError: the processor lacks `apply_chat_template`. Delegating
            is mandatory -- constructing the prompt string by hand is what
            finding F5-3 records as broken, and there is no safe fallback.
    """
    from core.errors import SpecialistError

    if not hasattr(processor, "apply_chat_template"):
        raise SpecialistError(
            "the processor does not expose apply_chat_template; optical-SAR "
            "prompts must be rendered through the chat template (finding F5-3) "
            "and there is no safe hand-built fallback",
            specialist="optical_sar",
        )
    return processor.apply_chat_template(
        build_messages(prompt), add_generation_prompt=True
    )


__all__ = [
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "build_explanation_prompt",
    "build_messages",
    "render_for_vlm",
]
