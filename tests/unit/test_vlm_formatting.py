"""Tests for Phase 6 prompt construction and loss masking (`training.vlm.formatting`).

The core coupling this file pins: training prompts are built by the **serving**
code (`specialists.vqa.prompts.build_messages`). An adapter tuned on prompts the
serving path does not produce is tuned for a distribution that never occurs at
inference, and the failure is invisible -- the loss curve is healthy and the
answers are quietly worse than the baseline.

The masking tests use the real processor (not the model) because the mask
boundary is defined against the processor's chat template. The processor is
provided by the session-scoped `smolvlm_processor` fixture, pinned the way the
trainer pins it.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.errors import SatQueryError
from specialists.vqa.prompts import PROMPT_VERSION, build_messages
from training.vlm.formatting import (
    IGNORE_INDEX,
    build_training_messages,
    describe_prompt_contract,
    format_example,
)

QUESTION = "Is water present in this image?"


# ---------------------------------------------------------------------------
# Prompts come from the serving code
# ---------------------------------------------------------------------------
def test_prompts_are_built_by_the_serving_module() -> None:
    assert build_training_messages(QUESTION) == build_messages("vqa", QUESTION)


def test_messages_carry_exactly_one_image_placeholder() -> None:
    """F5-3: SmolVLM raises without exactly one <image> token per image."""
    messages = build_training_messages(QUESTION)
    images = [
        content
        for message in messages
        for content in message["content"]
        if content.get("type") == "image"
    ]
    assert len(images) == 1


def test_describe_prompt_contract_records_the_source_and_version() -> None:
    described = describe_prompt_contract()
    assert described["source"] == "specialists.vqa.prompts"
    assert described["reused"] is True
    assert described["version"] == PROMPT_VERSION
    assert described["ignore_index"] == IGNORE_INDEX


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------
def test_prompt_version_is_recorded(smolvlm_processor: Any) -> None:
    example = format_example(
        smolvlm_processor, question=QUESTION, answer="Yes.", max_seq_length=512
    )
    assert example.prompt_version == PROMPT_VERSION


def test_prompt_tokens_are_masked_and_answer_tokens_supervise(
    smolvlm_processor: Any,
) -> None:
    example = format_example(
        smolvlm_processor, question=QUESTION, answer="Yes.", max_seq_length=512
    )
    assert example.prompt_length < example.full_length
    assert len(example.labels) == example.full_length
    assert all(token == IGNORE_INDEX for token in example.labels[: example.prompt_length])
    assert all(
        token != IGNORE_INDEX for token in example.labels[example.prompt_length :]
    )
    assert example.n_supervised_tokens > 0
    assert example.n_supervised_tokens == example.full_length - example.prompt_length


def test_a_valid_answer_supervises_the_same_token_count_as_before(
    smolvlm_processor: Any,
) -> None:
    """Measured: `"Yes."` -> suffix `"Yes.\\n"` -> 3 supervised tokens.

    Pinned so the empty-answer refusal cannot be implemented by loosening the
    tokenisation path for VALID answers: the count for a real answer must not
    move when the guard is added.
    """
    example = format_example(
        smolvlm_processor, question=QUESTION, answer="Yes.", max_seq_length=512
    )
    assert example.n_supervised_tokens == 3


@pytest.mark.parametrize("answer", ["", "   ", "\n", "\t\n "])
def test_an_empty_or_whitespace_only_answer_is_refused(
    smolvlm_processor: Any, answer: str
) -> None:
    """Finding A: an empty answer must be REFUSED, not silently scored.

    The defect this replaces: `suffix = answer if answer.endswith("\\n") else
    answer + "\\n"` turns `""` into `"\\n"`, which tokenises to exactly one token
    (the newline token, id 198). The token-count guard therefore never fires for a
    whitespace-only answer and the example trains on a bare newline -- a
    supervised token that carries no answer at all. `answer.strip() == ""` is the
    condition that actually catches it.

    The exception is matched against `(ValueError, SatQueryError)`: either the
    module's existing `ValueError` convention or a `SatQueryError`-derived module
    error (the convention of the sibling `collate`/`lora`/`dataset` modules). The
    message must state the answer is empty after stripping.
    """
    with pytest.raises((ValueError, SatQueryError), match="empty"):
        format_example(
            smolvlm_processor, question=QUESTION, answer=answer, max_seq_length=512
        )


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def test_over_budget_example_is_refused(smolvlm_processor: Any) -> None:
    with pytest.raises(ValueError, match="max_seq_length"):
        format_example(
            smolvlm_processor, question=QUESTION, answer="Yes.", max_seq_length=8
        )


def test_an_answer_that_adds_no_tokens_is_refused() -> None:
    """The token-count guard's branch, which stays AFTER the empty-answer guard.

    The empty-answer guard refuses whitespace-only answers before this point, so
    this branch is reached only by a NON-empty answer whose tokenisation adds
    nothing (see `test_an_empty_or_whitespace_only_answer_is_refused` for the
    guard that fires first). It is exercised here with a tokenizer that adds
    nothing for the suffix, to prove this guard still exists and is wired.
    """

    class _NoOpTokenizer:
        def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
            return {"input_ids": [1, 2, 3]}

    class _NoOpProcessor:
        tokenizer = _NoOpTokenizer()

        def apply_chat_template(
            self, messages: list, add_generation_prompt: bool = False, tokenize: bool = False
        ) -> str:
            return "prompt"

    with pytest.raises(ValueError, match="no tokens"):
        format_example(
            _NoOpProcessor(), question=QUESTION, answer="Yes.", max_seq_length=512
        )
