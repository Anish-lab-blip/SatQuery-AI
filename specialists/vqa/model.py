"""SatQuery AI — SmolVLM loader.

This module exists because the plan was wrong in four specific ways, and each
correction is enforced here rather than remembered (docs/PHASE5_VLM_CONTRACT.md).

    F5-1  `AutoModelForVision2Seq` does NOT EXIST in transformers 5.17.0.
          It is not deprecated — referencing it raises AttributeError. The
          loader class is resolved by feature detection, never hardcoded.

    F5-2  The processor's default `longest_edge` is 2048. A 512 px tile is
          upscaled 4x and then split by `do_image_splitting` into 4x4
          sub-images + 1 overview = 17 images and 1142 prompt tokens, versus
          1 image when pinned. MEASURED, not estimated.

    F5-3  SmolVLM requires one `<image>` token per image in the prompt. Raw
          prompt strings raise ValueError. All prompts go through
          `apply_chat_template`.

    F5-4  The dtype kwarg is not discoverable by signature. Neither
          `AutoModelForImageTextToText.from_pretrained` nor even
          `PreTrainedModel.from_pretrained` names it; both absorb it via
          **kwargs. Resolved by call-time fallback.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from core.errors import ModelLoadError

#: Environment variable naming a Phase 6 LoRA adapter to attach at load time.
#: It exists because `configs/base.yaml` is frozen (`Config.hash` is cited by
#: every run manifest), so the adapter cannot be a registry key.
ADAPTER_ENV_VAR = "SATQUERY_VLM_ADAPTER"

#: Preference order for the loader class. Resolved by feature detection.
LOADER_CANDIDATES: tuple[str, ...] = (
    "AutoModelForImageTextToText",   # transformers v5 name
    "AutoModelForVision2Seq",        # present in some v4 versions only
    "AutoModelForMultimodalLM",      # fallback spelling seen in the wild
)

#: dtype kwarg spellings, in preference order. v5 uses `dtype`; v4 `torch_dtype`.
DTYPE_KWARG_CANDIDATES: tuple[str, ...] = ("dtype", "torch_dtype")


@dataclass
class VLMLoadInfo:
    """Facts recorded in the execution trace so a run is reproducible."""

    checkpoint: str
    revision: str | None
    loader_class: str
    dtype_kwarg: str
    device: str
    parameters: int
    load_seconds: float
    processor_longest_edge: int
    max_images_seen: int
    #: Phase 6 — the LoRA adapter attached at load time, if any, so an inference
    #: result can be traced to an artifact. `None` means the base model served.
    adapter_path: str | None = None
    #: SHA256 of the attached adapter's weights, so the trace names the bytes.
    adapter_sha256: str | None = None

    def to_trace(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "loader_class": self.loader_class,
            "dtype_kwarg": self.dtype_kwarg,
            "device": self.device,
            "parameters": self.parameters,
            "load_seconds": round(self.load_seconds, 2),
            "processor_longest_edge": self.processor_longest_edge,
            "adapter_path": self.adapter_path,
            "adapter_sha256": self.adapter_sha256,
        }


def resolve_loader_class() -> type:
    """Return the first available loader class.

    F5-1: this is not paranoia. `AutoModelForVision2Seq` is simply gone in
    transformers 5.17.0, so a hardcoded reference would crash at import.
    """
    import transformers

    for name in LOADER_CANDIDATES:
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls

    available = [
        n for n in dir(transformers)
        if n.startswith("AutoModel") and n[9:10].isupper()
    ]
    raise ModelLoadError(
        "no usable vision-language loader class found in this transformers "
        f"version; looked for {list(LOADER_CANDIDATES)}. "
        f"Available AutoModel* classes: {sorted(available)}",
        specialist="vqa",
    )


def resolve_dtype_kwarg(loader_cls: type) -> str:
    """Return the dtype kwarg spelling this transformers version accepts.

    F5-4: `from_pretrained` is `**kwargs`-only on both the Auto* classes and
    on PreTrainedModel, so signature inspection cannot answer this. We return
    the preferred spelling and let `_load_with_dtype_fallback` correct it.
    """
    import inspect

    try:
        from transformers import PreTrainedModel

        params = inspect.signature(PreTrainedModel.from_pretrained).parameters
        if "dtype" in params:
            return "dtype"
        if "torch_dtype" in params:
            return "torch_dtype"
    except Exception:  # noqa: BLE001 - introspection is best-effort
        pass

    _ = loader_cls  # kept for signature stability; unused by design
    return DTYPE_KWARG_CANDIDATES[0]


def _load_with_dtype_fallback(
    loader_cls: type,
    checkpoint: str,
    revision: str | None,
    torch_dtype: torch.dtype,
) -> tuple[Any, str]:
    """Load the model, correcting the dtype kwarg on TypeError.

    Returns (model, dtype_kwarg_actually_used).
    """
    kwargs: dict[str, Any] = {}
    if revision:
        kwargs["revision"] = revision

    first = resolve_dtype_kwarg(loader_cls)
    order = [first] + [k for k in DTYPE_KWARG_CANDIDATES if k != first]

    last_error: Exception | None = None
    for kwarg in order:
        attempt = dict(kwargs)
        attempt[kwarg] = torch_dtype
        try:
            model = loader_cls.from_pretrained(checkpoint, **attempt)
            return model, kwarg
        except TypeError as exc:
            last_error = exc
            continue
        except Exception as exc:  # noqa: BLE001 - hub errors are numerous
            raise ModelLoadError(
                f"could not load VLM '{checkpoint}'"
                f"{f' @ {revision}' if revision else ''}: {exc}",
                specialist="vqa",
                context={"checkpoint": checkpoint, "revision": revision},
            ) from exc

    raise ModelLoadError(
        f"could not load VLM '{checkpoint}': no accepted dtype kwarg among "
        f"{order}. Last error: {last_error}",
        specialist="vqa",
    )


def _adapter_sha256(adapter_path: str) -> str | None:
    """SHA256 of an adapter's weight file, or `None` when it cannot be read.

    Prefers `adapter_model.safetensors`, then `adapter_model.bin`, then the
    `adapter_config.json`. Returns `None` rather than a fabricated digest when no
    file is present -- a made-up hash would be worse than an absent one.
    """
    directory = Path(adapter_path)
    for name in ("adapter_model.safetensors", "adapter_model.bin", "adapter_config.json"):
        candidate = directory / name
        if candidate.is_file():
            digest = hashlib.sha256()
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            return digest.hexdigest()
    return None


class SmolVLM:
    """Loaded SmolVLM plus its pinned processor.

    The processor pin is the load-bearing part. See F5-2.
    """

    def __init__(
        self,
        checkpoint: str,
        revision: str | None,
        processor_longest_edge: int,
        device: str = "cpu",
        torch_dtype: torch.dtype | None = None,
        do_image_splitting: bool = True,
        adapter_path: str | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.revision = revision
        self.device = device
        self.processor_longest_edge = processor_longest_edge
        self.do_image_splitting = do_image_splitting

        if processor_longest_edge < 1:
            raise ModelLoadError(
                f"processor_longest_edge must be >= 1, got {processor_longest_edge}",
                specialist="vqa",
            )

        try:
            from transformers import AutoProcessor
        except ImportError as exc:  # pragma: no cover
            raise ModelLoadError(
                f"transformers is not installed: {exc}", specialist="vqa"
            ) from exc

        dtype = torch_dtype or (
            torch.float16 if device == "cuda" else torch.float32
        )

        started = time.time()
        loader_cls = resolve_loader_class()
        model, dtype_kwarg = _load_with_dtype_fallback(
            loader_cls, checkpoint, revision, dtype
        )
        model.eval()
        model.to(device)

        # The pin is passed explicitly, NOT re-read from global config, so the
        # value the guard reports is the value actually enforced.
        processor = self._build_processor(
            AutoProcessor, checkpoint, revision, processor_longest_edge
        )

        # Phase 6 — an optional LoRA adapter, attached additively. Resolution
        # order: the explicit argument, then SATQUERY_VLM_ADAPTER, then none.
        # When none is set the model is byte-for-byte the base model.
        resolved_adapter = adapter_path or os.environ.get(ADAPTER_ENV_VAR) or None
        adapter_sha: str | None = None
        if resolved_adapter:
            model = self._attach_adapter(model, resolved_adapter)
            adapter_sha = _adapter_sha256(resolved_adapter)

        self.model = model
        self.processor = processor
        self.load_info = VLMLoadInfo(
            checkpoint=checkpoint,
            revision=revision,
            loader_class=loader_cls.__name__,
            dtype_kwarg=dtype_kwarg,
            device=device,
            parameters=sum(p.numel() for p in model.parameters()),
            load_seconds=time.time() - started,
            processor_longest_edge=processor_longest_edge,
            max_images_seen=0,
            adapter_path=resolved_adapter,
            adapter_sha256=adapter_sha,
        )

    @staticmethod
    def _attach_adapter(model: Any, adapter_path: str) -> Any:
        """Wrap `model` with a PEFT adapter loaded from `adapter_path`.

        A load failure raises rather than silently serving the base model: an
        inference result attributed to an adapter that did not actually attach is
        worse than a hard failure.
        """
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover
            raise ModelLoadError(
                f"an adapter was requested ({adapter_path}) but peft is not "
                f"installed: {exc}",
                specialist="vqa",
                context={"adapter_path": adapter_path},
            ) from exc

        directory = Path(adapter_path)
        if not directory.exists():
            raise ModelLoadError(
                f"adapter path does not exist: {directory}",
                specialist="vqa",
                context={"adapter_path": str(directory)},
            )
        try:
            return PeftModel.from_pretrained(model, str(directory))
        except Exception as exc:  # noqa: BLE001 - PEFT raises a wide range
            raise ModelLoadError(
                f"could not attach the LoRA adapter at {directory}: {exc}",
                specialist="vqa",
                context={"adapter_path": str(directory)},
            ) from exc

    @staticmethod
    def _build_processor(
        auto_processor: type,
        checkpoint: str,
        revision: str | None,
        longest_edge: int,
    ) -> Any:
        """Construct the processor with the resolution pinned.

        F5-2 is enforced here. The default would upscale and split. The edge is
        a parameter rather than a global read so the enforced value and the
        reported value cannot diverge.
        """
        kwargs: dict[str, Any] = {}
        if revision:
            kwargs["revision"] = revision
        # The pin. See the module docstring.
        kwargs["size"] = {"longest_edge": longest_edge}
        try:
            return auto_processor.from_pretrained(checkpoint, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"could not load processor for '{checkpoint}' with an explicit "
                f"longest_edge pin: {exc}",
                specialist="vqa",
            ) from exc

    # -- inference ---------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        image: Any,
        messages: list[dict[str, Any]],
        max_new_tokens: int = 128,
        do_sample: bool = False,
    ) -> str:
        """Run one deterministic generation. Returns the decoded new text."""
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # F5-2 regression guard: if the pin were lost, pixel_values would carry
        # a leading 17 instead of 1. Fail loudly rather than silently paying it.
        images_in_batch = self._count_images(inputs)
        self.load_info.max_images_seen = max(
            self.load_info.max_images_seen, images_in_batch
        )
        if images_in_batch > 1:
            raise ModelLoadError(
                f"processor produced {images_in_batch} images for a single input; "
                f"the longest_edge pin (currently "
                f"{self.processor_longest_edge}) is not being applied "
                f"(finding F5-2)",
                specialist="vqa",
                context={"images_in_batch": images_in_batch},
            )

        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

        prompt_len = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(
            generated[:, prompt_len:], skip_special_tokens=True
        )[0].strip()

    @staticmethod
    def _count_images(inputs: dict[str, Any]) -> int:
        pv = inputs.get("pixel_values")
        if pv is None or not hasattr(pv, "shape"):
            return 0
        # (batch, images, channels, h, w) -> images
        if len(pv.shape) == 5:
            return int(pv.shape[0] * pv.shape[1])
        return int(pv.shape[0])

    def unload(self) -> None:
        """Release the model. Used by the lazy model manager (plan section 49)."""
        del self.model
        self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        return (
            f"<SmolVLM {self.checkpoint} loader={self.load_info.loader_class} "
            f"edge={self.processor_longest_edge} device={self.device}>"
        )


def build_vlm(config, device: str | None = None, adapter_path: str | None = None) -> SmolVLM:
    """Construct the VLM from the central configuration.

    `adapter_path` optionally attaches a Phase 6 LoRA adapter. It defaults to the
    `SATQUERY_VLM_ADAPTER` environment variable inside `SmolVLM`; the registry is
    frozen and carries no adapter key.
    """
    if device is None:
        device = config.device_preference

    return SmolVLM(
        checkpoint=config.require("vlm.checkpoint"),
        revision=config.get("vlm.revision"),
        processor_longest_edge=int(config.get("vlm.processor_longest_edge", 512)),
        do_image_splitting=bool(config.get("vlm.do_image_splitting", True)),
        device=device,
        adapter_path=adapter_path,
    )


__all__ = [
    "SmolVLM",
    "VLMLoadInfo",
    "build_vlm",
    "resolve_loader_class",
    "resolve_dtype_kwarg",
    "LOADER_CANDIDATES",
    "DTYPE_KWARG_CANDIDATES",
    "ADAPTER_ENV_VAR",
]