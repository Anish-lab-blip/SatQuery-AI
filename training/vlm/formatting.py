"""SatQuery AI — Phase 6 prompt construction and loss masking.

THE ONE RULE THIS MODULE ENFORCES
---------------------------------
**Training prompts are built by the serving code.** `specialists.vqa.prompts`
owns `build_messages()`, `SYSTEM_PREAMBLE` and `PROMPT_VERSION`, and this module
calls it rather than composing its own strings.

That is not tidiness. An adapter trained on prompts the serving path does not
produce is an adapter tuned for a distribution that never occurs at inference --
and the failure is invisible: the loss curve is healthy, the adapter loads, and
the answers are worse than the baseline for reasons nothing in the logs explains.
The version is recorded in every run manifest so the coupling is auditable.

Two serving rules are inherited and must not be broken:

  * **F5-3** -- SmolVLM raises `ValueError` without exactly one `<image>` token per
    image. `build_messages` emits them; this module never hand-writes a prompt.
  * **C-5** -- no prompt asks for coordinates. Grounding is a specialist, not a
    language task. The instruction pairs carry no location, so nothing here can
    accidentally train one.

LOSS MASKING
------------
Only the **answer** tokens contribute to the loss. The prompt tokens are set to
`-100`, which is the ignore index. Training on prompt tokens would spend most of
the gradient teaching the model to reproduce a system preamble it is always given,
which is why an unmasked loss looks *better* and generalises worse.

The mask is built by tokenising the prompt alone and taking its length, rather
than by searching for a delimiter string. A delimiter search breaks silently when
the template changes; a length does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "IGNORE_INDEX",
    "FormattedExample",
    "build_training_messages",
    "format_example",
    "describe_prompt_contract",
]

#: PyTorch's cross-entropy ignore index.
IGNORE_INDEX = -100


@dataclass(frozen=True)
class FormattedExample:
    """One tokenised example plus the mask boundary that defines the loss."""

    input_ids: list[int]
    labels: list[int]
    prompt_length: int
    full_length: int
    prompt_version: str

    @property
    def n_supervised_tokens(self) -> int:
        """Answer tokens. Zero means the example teaches nothing -- refused."""
        return sum(1 for t in self.labels if t != IGNORE_INDEX)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_length": self.prompt_length,
            "full_length": self.full_length,
            "n_supervised_tokens": self.n_supervised_tokens,
            "prompt_version": self.prompt_version,
        }


def build_training_messages(question: str, *, image_count: int = 1) -> list[dict[str, Any]]:
    """The serving message list for a question. Delegates; never re-implements."""
    from specialists.vqa.prompts import build_messages

    return build_messages("vqa", question, image_count=image_count)


def _render(processor: Any, messages: list[dict[str, Any]], *, generation_prompt: bool) -> str:
    """Render messages through the processor's chat template.

    `add_generation_prompt=True` appends the assistant turn header, which is what
    the model conditions on at inference. Training without it would leave a gap
    between the training and serving prefixes.
    """
    return processor.apply_chat_template(
        messages, add_generation_prompt=generation_prompt, tokenize=False
    )


def format_example(
    processor: Any,
    *,
    question: str,
    answer: str,
    max_seq_length: int,
    eos_token: str | None = None,
) -> FormattedExample:
    """Tokenise one (question, answer) pair with the prompt tokens masked.

    Raises `ValueError` when the answer is empty/whitespace-only, when the
    example has no supervised tokens, or when it exceeds `max_seq_length`.
    All three are silent-quality failures otherwise: the first teaches the model
    to emit nothing, the second trains on nothing, and the third is truncated to
    nothing if the prompt alone is longer than the budget.

    The empty-answer case is refused **explicitly**, and the token-count check
    below cannot substitute for it. `"\\n"` is a real token (measured: id 198),
    so an empty answer still produces a well-formed-looking example whose only
    supervised token is the newline separator -- the token count is non-zero and
    the example passes every length test while supervising nothing but a line
    break. The token-count check is kept as defence in depth for *other*
    degenerate cases, not this one.
    """
    from specialists.vqa.prompts import PROMPT_VERSION

    if not answer.strip():
        raise ValueError(
            "the answer is empty or whitespace-only; an empty target would "
            "supervise only the newline separator. This is not caught by the "
            "token-count check below: '\\n' is a real token (measured id 198), so "
            "the example would look well-formed while teaching the model to emit "
            "nothing. Supply a non-empty answer."
        )

    messages = build_training_messages(question)
    prompt_text = _render(processor, messages, generation_prompt=True)

    suffix = answer if answer.endswith("\n") else answer + "\n"
    if eos_token:
        suffix += eos_token

    prompt_ids = processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = processor.tokenizer(
        prompt_text + suffix, add_special_tokens=False
    )["input_ids"]

    if len(full_ids) > max_seq_length:
        raise ValueError(
            f"example is {len(full_ids)} tokens, over the "
            f"max_seq_length={max_seq_length} budget; raise max_seq_length or "
            f"shorten the answer"
        )
    if len(prompt_ids) >= len(full_ids):
        raise ValueError(
            "the answer contributed no tokens; the example would train on nothing"
        )

    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]
    if all(t == IGNORE_INDEX for t in labels):
        raise ValueError("every token is masked; the example teaches nothing")

    return FormattedExample(
        input_ids=list(full_ids),
        labels=labels,
        prompt_length=len(prompt_ids),
        full_length=len(full_ids),
        prompt_version=PROMPT_VERSION,
    )


def describe_prompt_contract(processor: Any | None = None) -> dict[str, Any]:
    """The prompt contract, for the run manifest.

    Records the serving module and version so a run can be tied to the exact
    prompt set it trained against. When `processor` is given, the rendered prompt
    for a probe question is included as evidence that the template is the one
    actually used.
    """
    from specialists.vqa.prompts import PROMPT_VERSION, describe_prompts

    out: dict[str, Any] = {
        "source": "specialists.vqa.prompts",
        "reused": True,
        "reason": (
            "training prompts are built by the serving code so an adapter cannot "
            "be tuned for a prompt distribution that never occurs at inference"
        ),
        "ignore_index": IGNORE_INDEX,
        "masking": "prompt tokens masked; only answer tokens contribute to the loss",
    }
    out.update(describe_prompts())
    out["version"] = PROMPT_VERSION

    if processor is not None:
        try:
            messages = build_training_messages("Is water present in this image?")
            out["probe_prompt"] = _render(
                processor, messages, generation_prompt=True
            )
        except Exception as exc:  # noqa: BLE001 - a probe must not fail the run
            out["probe_prompt_error"] = f"{type(exc).__name__}: {exc}"
    return out
