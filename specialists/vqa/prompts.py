"""SatQuery AI — versioned VLM prompt set.

Rules (docs/ARCHITECTURE_FREEZE.md section 2.2, plan section 19):

  * Prompts are **versioned data**, not f-strings scattered through the code.
    `PROMPT_VERSION` is recorded in the execution trace so an evaluation run
    can be reproduced.
  * Every prompt is **evidence-grounded**: it tells the model to answer only
    from what is visible and to say so when it cannot.
  * No prompt asks the model for coordinates. Grounding is a specialist, not
    a language task (finding C-5).
  * Prompts are built through the processor's chat template, never as raw
    strings — SmolVLM raises ValueError without one <image> token per image
    (finding F5-3).

These must be FROZEN before benchmark evaluation. Editing a prompt invalidates
every recorded metric that used the previous version.
"""

from __future__ import annotations

from typing import Any, Literal

#: Bump when any prompt text changes. Recorded in every trace.
PROMPT_VERSION = "v001"

PromptKind = Literal["vqa", "caption", "explain_grounding", "explain_change"]


# ---------------------------------------------------------------------------
# System preamble — shared by every task
# ---------------------------------------------------------------------------

SYSTEM_PREAMBLE = (
    "You are a remote-sensing image analyst. "
    "Answer only from what is visible in the provided image. "
    "Do not speculate about locations, coordinates, measurements or dates "
    "that you cannot see. "
    "If the image does not contain enough information, say so plainly."
)


# ---------------------------------------------------------------------------
# Task instructions
# ---------------------------------------------------------------------------

_INSTRUCTIONS: dict[PromptKind, str] = {
    "vqa": (
        "Answer the user's question about this satellite image in one short "
        "sentence. If the answer is not visible, reply exactly: "
        "\"Not enough information in the image.\""
    ),
    "caption": (
        "Describe this satellite image in two to three sentences. "
        "Mention the dominant land cover, any notable objects, and the overall "
        "setting. Do not repeat yourself."
    ),
    # The VLM explains a region another specialist found. It never produces the
    # region itself.
    "explain_grounding": (
        "A separate spatial model located the region described below. "
        "Describe what is visible in that region in one sentence. "
        "You did not compute the region; do not comment on its coordinates."
    ),
    # The VLM interprets a change map another specialist computed.
    "explain_change": (
        "A separate change-detection model produced the change regions "
        "described below. Summarise what most likely changed, in one or two "
        "sentences. The change map is the source of truth, not your impression."
    ),
}


def build_messages(
    kind: PromptKind,
    user_text: str,
    *,
    image_count: int = 1,
    evidence_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build a chat-template message list.

    `image_count` placeholders are emitted so the processor sees exactly one
    <image> token per image. Getting this wrong is the F5-3 failure.

    `evidence_context` is rendered as text the model may reference. It never
    carries coordinates the model is expected to echo — it carries labels and
    scores a specialist already computed.
    """
    if image_count < 1:
        raise ValueError(f"image_count must be >= 1, got {image_count}")

    content: list[dict[str, Any]] = [{"type": "image"} for _ in range(image_count)]

    body = _INSTRUCTIONS[kind]
    if evidence_context:
        rendered = ", ".join(
            f"{key}={value}" for key, value in sorted(evidence_context.items())
        )
        body = f"{body}\n\nAdditional context from other specialists: {rendered}"

    content.append({"type": "text", "text": f"{user_text}\n\n{body}"})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PREAMBLE}]},
        {"role": "user", "content": content},
    ]


def describe_prompts() -> dict[str, Any]:
    """Prompt manifest for the execution trace and the run manifest."""
    return {
        "version": PROMPT_VERSION,
        "kinds": sorted(_INSTRUCTIONS),
        "system_preamble_sha": _short_hash(SYSTEM_PREAMBLE),
    }


def _short_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()[:12]


__all__ = [
    "PROMPT_VERSION",
    "PromptKind",
    "SYSTEM_PREAMBLE",
    "build_messages",
    "describe_prompts",
]