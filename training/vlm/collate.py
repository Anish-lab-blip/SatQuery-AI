"""SatQuery AI — Phase 6 batch assembly for SmolVLM.

THE SHAPE HAZARD THIS MODULE EXISTS TO HANDLE (measured 2026-09-23)
-------------------------------------------------------------------
SmolVLM's processor with `do_image_splitting=True` expands one input image into
a variable number of sub-images, so `pixel_values` is **5-D**:

    processor(images=<1 patch>, text=<prompt>)["pixel_values"].shape
        == (1, 1, 3, 512, 512)     # (batch, n_images, channels, H, W)

and a batch of three gives `(3, 1, 3, 512, 512)`. Getting the leading
`n_images` axis wrong is the most likely cause of a shape crash on Kaggle.

THE TOKEN-ALIGNMENT HAZARD, WHICH IS SUBTLER AND WORSE
------------------------------------------------------
`training.vlm.formatting.format_example` builds its token ids through the
**plain tokenizer**, where the `<image>` placeholder is a single token. The
processor, given the image, expands that placeholder into the full image grid.
Measured on the real corpus:

    tokenizer-only prompt length  : 105   (one <image> token)
    processor-with-image length   : 171   (the image expanded to 66 more tokens)

So the ids `format_example` produces **do not align** with the `pixel_values`
the processor emits. A collator that stacked `FormattedExample.input_ids`
beside `processor(...)["pixel_values"]` would hand the model a prompt with one
image token and a 66-token image tensor. The failure is silent -- it does not
raise, it just trains against a misaligned image.

This collator therefore re-tokenises through the **processor** (the image-aware
path) and derives the loss mask from the expanded prompt length. The
`FormattedExample` is still carried for provenance (prompt version, supervised
token count), and a bare `FormattedExample` with no image is refused with an
explicit `CollationError` rather than silently trained on.

PADDING SIDE
------------
The tokenizer's `padding_side` is `right` (measured). Right padding is the norm
for causal-LM training: the pads sit after the answer, never inside the prompt,
so no pad token is ever attended to as context. The side is read from the live
tokenizer and recorded by `describe_collation()` rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from core.errors import SatQueryError
from training.vlm.formatting import IGNORE_INDEX, FormattedExample

__all__ = [
    "IGNORE_INDEX",
    "CollationError",
    "CollationItem",
    "Collator",
    "as_patch_sample",
    "render_sample",
    "make_item",
]


class CollationError(SatQueryError):
    """A batch could not be assembled without misaligning image and text."""

    code = "collation_error"
    user_message = "The Phase 6 training batch could not be assembled."


def _read_longest_edge(processor: Any) -> int | None:
    """The processor's pinned `longest_edge`, or `None` when it cannot be read.

    The pin is what keeps F5-2 from firing. It lives on the image processor's
    `size` (a `SizeDict`, measured: `Idefics3ImageProcessor`), with a
    `max_image_size` dict fallback. It is **read, not assumed**, so
    `describe_collation` records the value actually in force. `None` is returned
    -- never a guessed 512 -- when neither is present, because a fabricated pin
    would make the manifest claim a guarantee that was never checked.
    """
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return None
    size = getattr(image_processor, "size", None)
    edge = getattr(size, "longest_edge", None) if size is not None else None
    if edge is None:
        max_image_size = getattr(image_processor, "max_image_size", None)
        if isinstance(max_image_size, dict):
            edge = max_image_size.get("longest_edge")
    try:
        return int(edge) if edge is not None else None
    except (TypeError, ValueError):
        return None


def _count_subimages(processor_output: dict[str, Any]) -> int:
    """How many images the processor emitted for ONE input patch.

    Mirrors `specialists.vqa.model.SmolVLM._count_images`: under
    `do_image_splitting` the processor returns `pixel_values` shaped
    `(batch, n_images, C, H, W)`, so the image count is the second axis. For a
    single input patch, more than one image is the F5-2 signature.
    """
    pixel_values = processor_output.get("pixel_values")
    if pixel_values is None or not hasattr(pixel_values, "shape"):
        return 0
    if len(pixel_values.shape) == 5:
        return int(pixel_values.shape[1])
    return int(pixel_values.shape[0])


def as_patch_sample(sample: Any) -> Any:
    """Coerce a corpus `InstructionSample` into a `PatchSample` for rendering.

    `render_rgb` reads `patch.band(token)`; an `InstructionSample` carries the
    same `band_paths` but not that method. Rather than edit the frozen dataset
    module, this is the one place the conversion lives, so the trainer and the
    evaluator cannot diverge.
    """
    from training.vlm.dataset import PatchSample

    if isinstance(sample, PatchSample):
        return sample
    return PatchSample(
        patch_id=sample.patch_id,
        scene_id=sample.scene_id,
        tile=sample.scene_id,
        directory=sample.directory,
        labels=(),
        band_paths=dict(sample.band_paths),
        official_split=None,
    )


def render_sample(sample: Any, *, percentiles: tuple[float, float] = (2.0, 98.0)) -> Any:
    """Render any corpus sample (patch or instruction) to a true-colour image."""
    from training.vlm.dataset import render_rgb

    return render_rgb(as_patch_sample(sample), percentiles=percentiles)


@dataclass(frozen=True)
class CollationItem(FormattedExample):
    """A `FormattedExample` that also carries the image and its rendered prompt.

    Subclassing (rather than editing) `FormattedExample` keeps the lead's frozen
    formatting module untouched while supplying the two things the processor
    path needs and the tokenizer-only path cannot provide: the image, and the
    chat-templated prompt text whose processor tokenisation the mask is built
    against.
    """

    image: Any = None
    prompt_text: str = ""
    answer_text: str = ""


def make_item(
    processor: Any,
    *,
    question: str,
    answer: str,
    image: Any,
    max_seq_length: int,
    eos_token: str | None = None,
) -> CollationItem:
    """Build a `CollationItem` from one (image, question, answer) triple.

    `format_example` is still called: it validates that the answer contributes
    supervised tokens and that the example fits the budget, and its
    `prompt_version` is carried for the manifest. The processor-rendered prompt
    text is stored alongside it so the collator can build the aligned ids.
    """
    from training.vlm.formatting import build_training_messages, format_example

    example = format_example(
        processor,
        question=question,
        answer=answer,
        max_seq_length=max_seq_length,
        eos_token=eos_token,
    )
    messages = build_training_messages(question)
    prompt_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    suffix = answer if answer.endswith("\n") else answer + "\n"
    if eos_token:
        suffix += eos_token
    return CollationItem(
        input_ids=list(example.input_ids),
        labels=list(example.labels),
        prompt_length=example.prompt_length,
        full_length=example.full_length,
        prompt_version=example.prompt_version,
        image=image,
        prompt_text=prompt_text,
        answer_text=suffix,
    )


class Collator:
    """Assemble a batch of `FormattedExample`s into model-ready tensors.

    Args:
        processor: the SmolVLM processor. Its tokenizer's `padding_side` and
            `pad_token_id` are read from the live object.
        max_seq_length: the hard token budget (contract item P). An example that
            exceeds it raises rather than being silently truncated to nothing.
        pad_token_id: override for the padding id. Defaults to the tokenizer's.
    """

    def __init__(
        self,
        processor: Any,
        max_seq_length: int,
        pad_token_id: int | None = None,
    ) -> None:
        if max_seq_length < 1:
            raise CollationError(
                f"max_seq_length must be >= 1, got {max_seq_length}",
                context={"max_seq_length": max_seq_length},
            )
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.pad_side = str(getattr(processor.tokenizer, "padding_side", "right"))
        #: The processor's resolution pin, read from the live object (`None` when
        #: it cannot be read). Recorded in the manifest; the F5-2 guard in
        #: `__call__` is what actually enforces it, by measurement.
        self.processor_longest_edge = _read_longest_edge(processor)
        if self.pad_side not in ("right", "left"):
            raise CollationError(
                f"unsupported padding_side {self.pad_side!r}; expected 'right' or "
                f"'left'",
                context={"padding_side": self.pad_side},
            )
        if pad_token_id is not None:
            self.pad_token_id = int(pad_token_id)
        else:
            token_id = getattr(processor.tokenizer, "pad_token_id", None)
            if token_id is None:
                raise CollationError(
                    "the tokenizer has no pad_token_id and none was supplied; "
                    "padding would silently use an arbitrary id",
                    context={"padding_side": self.pad_side},
                )
            self.pad_token_id = int(token_id)
        #: Pixel keys actually observed, for the manifest.
        self.pixel_keys: set[str] = set()

    # -- public ------------------------------------------------------------
    def __call__(self, examples: Sequence[FormattedExample]) -> dict[str, torch.Tensor]:
        """Collate a batch. Returns `input_ids`, `attention_mask`, `labels`, and
        the processor's pixel keys, correctly stacked across the batch.

        Raises:
            CollationError: an example carries no image, an example exceeds
                `max_seq_length`, or the batch's per-image counts differ (which
                would make a rectangular pixel tensor impossible).
        """
        if not examples:
            raise CollationError("cannot collate an empty batch")

        rows: list[dict[str, Any]] = []
        pixel_batches: dict[str, list[torch.Tensor]] = {}
        n_images_seen: list[int] = []

        for item in examples:
            image = getattr(item, "image", None)
            prompt_text = getattr(item, "prompt_text", "")
            answer_text = getattr(item, "answer_text", "")
            if image is None or not prompt_text or not answer_text:
                raise CollationError(
                    "a batch item has no image/prompt_text/answer_text. Build "
                    "items with training.vlm.collate.make_item: a bare "
                    "FormattedExample cannot be aligned with pixel_values because "
                    "the processor expands <image> into the image grid and the "
                    "tokenizer-only ids do not.",
                    context={"type": type(item).__name__},
                )

            out = self.processor(
                images=image, text=prompt_text, return_tensors="pt"
            )
            expanded_prompt = out["input_ids"][0].tolist()
            prompt_len = len(expanded_prompt)

            # -- F5-2 guard, placed BEFORE the length check ----------------
            # Unpinned, the processor upscales a 512 px tile and splits it into
            # ~17 sub-images / ~1,147 tokens. Every batch then trips the
            # max_seq_length check below and is refused with a bare length error
            # that misattributes the cause to the answer. Detect the split here
            # and name the real cause instead.
            n_subimages = _count_subimages(out)
            if n_subimages > 1:
                raise CollationError(
                    f"the processor split a single rendered patch into "
                    f"{n_subimages} sub-images (the prompt expanded to "
                    f"{prompt_len} tokens). This is finding F5-2: the processor "
                    f"is not pinned. Unpinned, its default longest_edge=2048 "
                    f"upscales a 512 px tile and do_image_splitting cuts it into "
                    f"~17 sub-images / ~1,147 tokens, so every batch is refused. "
                    f"Pin it with size={{'longest_edge': 512}}, as "
                    f"specialists.vqa.model.SmolVLM does.",
                    context={
                        "n_subimages": n_subimages,
                        "prompt_tokens": prompt_len,
                        "processor_longest_edge": self.processor_longest_edge,
                    },
                )

            answer_ids = self.processor.tokenizer(
                answer_text, add_special_tokens=False
            )["input_ids"]
            if not answer_ids:
                raise CollationError(
                    "the answer tokenised to nothing; the example would train on "
                    "no supervised tokens",
                    context={"answer_text": answer_text[:80]},
                )

            full_ids = expanded_prompt + list(answer_ids)
            if len(full_ids) > self.max_seq_length:
                raise CollationError(
                    f"example is {len(full_ids)} tokens, over the "
                    f"max_seq_length={self.max_seq_length} budget; raise the "
                    f"budget or shorten the answer",
                    context={
                        "n_tokens": len(full_ids),
                        "max_seq_length": self.max_seq_length,
                    },
                )
            labels = [IGNORE_INDEX] * prompt_len + list(answer_ids)

            rows.append({"input_ids": full_ids, "labels": labels})

            for key, value in out.items():
                if key in ("input_ids", "attention_mask"):
                    continue
                if not isinstance(value, torch.Tensor):
                    continue
                self.pixel_keys.add(key)
                pixel_batches.setdefault(key, []).append(value)
                if key == "pixel_values":
                    n_images_seen.append(
                        int(value.shape[1]) if value.dim() == 5 else 1
                    )

        if not pixel_batches:
            raise CollationError(
                "the processor emitted no pixel tensors; a vision-language batch "
                "without pixel_values cannot train the image path",
                context={"processor_keys": sorted(pixel_batches)},
            )

        # A rectangular pixel tensor requires a constant image count per sample.
        if len(set(n_images_seen)) > 1:
            raise CollationError(
                f"the batch produced differing per-sample image counts "
                f"{sorted(set(n_images_seen))}; a rectangular pixel tensor cannot "
                f"be built. This happens when patches render to different sizes; "
                f"render every patch to a fixed size before collating.",
                context={"n_images": sorted(set(n_images_seen))},
            )

        batch = self._pad_text(rows)
        for key, tensors in pixel_batches.items():
            batch[key] = torch.cat(tensors, dim=0)
        return batch

    def describe_collation(self) -> dict[str, Any]:
        """The collation contract, for the run manifest."""
        return {
            "pad_token_id": self.pad_token_id,
            "pad_side": self.pad_side,
            "ignore_index": IGNORE_INDEX,
            "max_seq_length": self.max_seq_length,
            "processor_longest_edge": self.processor_longest_edge,
            "f5_2_guard": {
                "enforced_by": "measurement in Collator.__call__",
                "expected_subimages_per_patch": 1,
                "unpinned_signature": (
                    "~17 sub-images / ~1,147 prompt tokens; raises a CollationError "
                    "naming F5-2 before the length check can misattribute the cause"
                ),
            },
            "pixel_keys": sorted(self.pixel_keys),
            "note": (
                "pixel_values is (batch, n_images, C, H, W) with do_image_splitting; "
                "ids come from the processor (image-expanded) path, not the plain "
                "tokenizer, so text and image align"
            ),
        }

    # -- internals ---------------------------------------------------------
    def _pad_text(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Pad `input_ids`/`labels` and build the `attention_mask`.

        Padding side is `self.pad_side`. `input_ids` pad with the tokenizer's pad
        id; `labels` pad with `IGNORE_INDEX` so padded positions never contribute
        to the loss (padding a label with the pad id would train the model to
        emit pad tokens).
        """
        max_len = max(len(r["input_ids"]) for r in rows)
        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []

        for row in rows:
            ids = list(row["input_ids"])
            labels = list(row["labels"])
            n_pad = max_len - len(ids)
            if self.pad_side == "right":
                input_rows.append(ids + [self.pad_token_id] * n_pad)
                label_rows.append(labels + [IGNORE_INDEX] * n_pad)
                mask_rows.append([1] * len(ids) + [0] * n_pad)
            else:
                input_rows.append([self.pad_token_id] * n_pad + ids)
                label_rows.append([IGNORE_INDEX] * n_pad + labels)
                mask_rows.append([0] * n_pad + [1] * len(ids))

        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
        }
