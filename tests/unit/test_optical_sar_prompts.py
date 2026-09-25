"""Optical-SAR prompts — the narration boundary.

Freeze section 5:

    "No LLM-generated coordinates. No LLM-generated confidence."
    "Prompts are versioned files, frozen before benchmark evaluation."

The VLM's only job in this workflow is to put an already-computed decision into
a sentence. These tests pin that boundary, because the easy mistake is to ask
the model for the thing the measurement models already produced -- at which
point two answers exist and nothing says which one the system believed.
"""

from __future__ import annotations

import pytest

from core.errors import SpecialistError
from specialists.optical_sar.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_explanation_prompt,
    build_messages,
    render_for_vlm,
)


def test_prompt_version_is_declared_for_the_trace():
    """A result must be attributable to the prompt that produced it."""
    assert PROMPT_VERSION.startswith("optical_sar_v")


def test_prompt_carries_facts_rather_than_asking_for_them():
    """Every number is supplied TO the model, not requested FROM it.

    The predicted label, the margin and both availability counts are inputs to
    this function. Asking the model to derive them would put a second, unmeasured
    answer next to the measured one.
    """
    prompt = build_explanation_prompt(
        predicted_label="urban",
        fusion_margin=0.62,
        optical_available=4,
        optical_total=12,
        sar_available=2,
        sar_total=2,
    )

    assert "urban" in prompt
    assert "0.620" in prompt
    assert "4 of 12" in prompt
    assert "2 of 2" in prompt


def test_prompt_forbids_confidence_and_coordinates():
    """The instruction must actually say what not to do."""
    prompt = build_explanation_prompt(
        predicted_label="urban",
        fusion_margin=0.62,
        optical_available=4,
        optical_total=12,
        sar_available=2,
        sar_total=2,
    )

    lowered = prompt.lower()
    assert "confidence" in lowered
    assert "do not state" in lowered
    assert "coordinate" in lowered or "location" in lowered


def test_missing_channels_are_flagged_in_the_prompt():
    """"Classified as urban" alone invites the reader to assume a full scene.

    The prompt must say how much of each modality was actually measured.
    """
    prompt = build_explanation_prompt(
        predicted_label="urban",
        fusion_margin=0.5,
        optical_available=4,
        optical_total=12,
        sar_available=1,
        sar_total=2,
    )

    assert "8 optical channel(s)" in prompt
    assert "carry no information" in prompt
    assert "1 SAR channel(s)" in prompt


def test_a_full_scene_adds_no_missing_channel_note():
    prompt = build_explanation_prompt(
        predicted_label="urban",
        fusion_margin=0.5,
        optical_available=12,
        optical_total=12,
        sar_available=2,
        sar_total=2,
    )
    assert "not measured" not in prompt


def test_the_operator_query_is_included_when_present():
    prompt = build_explanation_prompt(
        predicted_label="urban",
        fusion_margin=0.5,
        optical_available=12,
        optical_total=12,
        sar_available=2,
        sar_total=2,
        query="is there flooding near the river?",
    )
    assert "is there flooding near the river?" in prompt


def test_agreement_is_included_only_when_measured():
    with_agreement = build_explanation_prompt(
        predicted_label="urban",
        fusion_margin=0.5,
        optical_available=12,
        optical_total=12,
        sar_available=2,
        sar_total=2,
        cross_modal_agreement=0.81,
    )
    assert "0.810" in with_agreement

    without = build_explanation_prompt(
        predicted_label="urban",
        fusion_margin=0.5,
        optical_available=12,
        optical_total=12,
        sar_available=2,
        sar_total=2,
    )
    assert "Cross-modal agreement" not in without


def test_prompts_are_rendered_through_the_chat_template():
    """Finding F5-3: SmolVLM raises ValueError without one <image> token.

    The `<image>` token is placed by the chat template, never by hand.
    """
    class _Processor:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, add_generation_prompt=False):  # noqa: ANN001
            self.calls.append((messages, add_generation_prompt))
            return "<rendered>"

    processor = _Processor()
    rendered = render_for_vlm(processor, "explain this")

    assert rendered == "<rendered>"
    assert len(processor.calls) == 1
    messages, add_generation = processor.calls[0]
    assert add_generation is True
    # The user turn carries an image placeholder.
    user_content = messages[1]["content"]
    assert any(part["type"] == "image" for part in user_content)


def test_a_processor_without_the_chat_template_is_refused():
    """There is no safe hand-built fallback -- delegation is mandatory."""
    class _Bare:
        pass

    with pytest.raises(SpecialistError) as excinfo:
        render_for_vlm(_Bare(), "explain this")
    assert "chat template" in str(excinfo.value.detail).lower()


def test_messages_place_the_system_instruction_first():
    messages = build_messages("explain this")
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert SYSTEM_PROMPT in messages[0]["content"][0]["text"]
