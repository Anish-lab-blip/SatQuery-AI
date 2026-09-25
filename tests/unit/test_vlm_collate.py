"""Tests for Phase 6 batch assembly (`training.vlm.collate`).

The processor is loaded (cheap and cached -- NOT the model), because the whole
point of this module is alignment with the processor's image-expanded token
stream, and a hand-built stub would test the stub. It is provided by the
session-scoped `smolvlm_processor` fixture, pinned the way the trainer pins it:
an unpinned processor upscales a 512 px tile 4x and splits it into 17
sub-images, a different token distribution from the trainer's (finding F5-2).

The real pixel key names are asserted from measurement, not guessed:
`pixel_values` and `pixel_attention_mask`, with `pixel_values` 5-D
`(batch, n_images, C, H, W)`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from training.vlm.collate import (
    IGNORE_INDEX,
    CollationError,
    Collator,
    as_patch_sample,
    make_item,
)
from training.vlm.dataset import InstructionSample
from training.vlm.formatting import FormattedExample

#: Measured from the pinned processor.
EXPECTED_PIXEL_KEYS = {"pixel_values", "pixel_attention_mask"}


def _image(size: int = 512) -> Any:
    from PIL import Image

    return Image.new("RGB", (size, size), (10, 20, 30))


def _two_items(processor: Any) -> list[Any]:
    """Two items whose prompts differ in length, so the batch must pad."""
    return [
        make_item(
            processor,
            question="Is water present in this image?",
            answer="Yes.",
            image=_image(),
            max_seq_length=512,
        ),
        make_item(
            processor,
            question=(
                "Is arable land present in this image, and please answer as "
                "briefly as you can?"
            ),
            answer="No.",
            image=_image(),
            max_seq_length=512,
        ),
    ]


def test_single_item_batch_has_no_padding(smolvlm_processor: Any) -> None:
    collator = Collator(smolvlm_processor, 512)
    batch = collator([_two_items(smolvlm_processor)[0]])
    assert batch["input_ids"].shape[0] == 1
    assert batch["attention_mask"].all()
    # The prompt is still masked; only the answer supervises.
    assert (batch["labels"] == IGNORE_INDEX).sum() > 0
    assert (batch["labels"] != IGNORE_INDEX).sum() > 0


def test_differing_lengths_pad_labels_with_ignore_index(
    smolvlm_processor: Any,
) -> None:
    collator = Collator(smolvlm_processor, 512)
    batch = collator(_two_items(smolvlm_processor))
    assert batch["input_ids"].shape[0] == 2

    pad = batch["attention_mask"] == 0
    assert pad.any(), "the batch was expected to require padding"
    # Labels are padded with the ignore index, never with the pad token id --
    # padding a label with the pad id would train the model to emit pad tokens.
    assert (batch["labels"][pad] == IGNORE_INDEX).all()
    assert not (batch["labels"][pad] == collator.pad_token_id).any()
    # input_ids pad with the tokenizer's pad id; attention_mask is 0 there.
    assert (batch["input_ids"][pad] == collator.pad_token_id).all()


def test_pixel_keys_are_carried_through(smolvlm_processor: Any) -> None:
    collator = Collator(smolvlm_processor, 512)
    batch = collator(_two_items(smolvlm_processor))

    assert collator.pixel_keys == EXPECTED_PIXEL_KEYS
    for key in EXPECTED_PIXEL_KEYS:
        assert key in batch
    # (batch, n_images, C, H, W); n_images == 1 because the pin is applied.
    assert batch["pixel_values"].dim() == 5
    assert batch["pixel_values"].shape[0] == 2
    assert batch["pixel_values"].shape[1] == 1


def test_a_bare_formatted_example_is_refused(smolvlm_processor: Any) -> None:
    """A tokenizer-only example cannot be aligned with pixel_values, because the
    processor expands <image> into the image grid."""
    collator = Collator(smolvlm_processor, 512)
    bare = FormattedExample(
        input_ids=[1, 2, 3],
        labels=[IGNORE_INDEX, IGNORE_INDEX, 1],
        prompt_length=2,
        full_length=3,
        prompt_version="v001",
    )
    with pytest.raises(CollationError):
        collator([bare])


def test_over_budget_example_is_refused(smolvlm_processor: Any) -> None:
    collator = Collator(smolvlm_processor, 16)
    with pytest.raises(CollationError):
        collator([_two_items(smolvlm_processor)[0]])


def test_empty_batch_is_refused(smolvlm_processor: Any) -> None:
    collator = Collator(smolvlm_processor, 512)
    with pytest.raises(CollationError):
        collator([])


def test_describe_collation_records_the_pad_contract(
    smolvlm_processor: Any,
) -> None:
    collator = Collator(smolvlm_processor, 512)
    collator(_two_items(smolvlm_processor))
    described = collator.describe_collation()
    assert described["pad_side"] == "right"
    assert described["ignore_index"] == IGNORE_INDEX
    assert described["max_seq_length"] == 512
    assert described["pixel_keys"] == ["pixel_attention_mask", "pixel_values"]


def test_as_patch_sample_coerces_an_instruction_sample() -> None:
    sample = InstructionSample(
        sample_id="s",
        patch_id="p",
        scene_id="ben_33UUP:0",
        split="train",
        question="q",
        answer="Yes.",
        family="presence",
        directory=Path("/nonexistent"),
        band_paths={},
    )
    patch = as_patch_sample(sample)
    assert patch.patch_id == "p"
    assert patch.scene_id == "ben_33UUP:0"
