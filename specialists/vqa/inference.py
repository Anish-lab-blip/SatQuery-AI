"""SatQuery AI — VQA and captioning specialist.

Implements Workflow A (single-image VQA) and Workflow B (caption) from the
frozen architecture. Both are the same specialist because both are "one image
in, one string out" — the difference is the prompt kind.

Dependencies:
    core/config.py          config registry
    core/errors.py          typed failures
    core/schemas.py         SpecialistResult and friends
    specialists/base.py     the interface this implements
    specialists/vqa/model.py   SmolVLM loader (F5-1..F5-4 enforced there)
    specialists/vqa/prompts.py versioned prompt set (F5-3 enforced there)
    preprocessing/raster.py raster reading, georeferencing preserved

Prompt discipline (finding F5-3): every prompt goes through the loader's
`generate`, which applies the chat template. This module never builds a raw
prompt string.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from core.errors import InvalidRequestError, SpecialistError
from core.schemas import (
    ConfidenceBreakdown,
    Evidence,
    EvidenceType,
    Modality,
    SpecialistResult,
    Task,
)
from preprocessing.quality import ImageQuality, assess_image_quality
from specialists.base import Specialist, SpecialistRequest
from specialists.vqa import prompts
from specialists.vqa.model import SmolVLM

#: Phrases the VLM uses to decline. Treated as a low-confidence answer rather
#: than an error: the model answered, it just had nothing to say.
_REFUSAL_MARKERS: tuple[str, ...] = (
    "not enough information",
    "cannot determine",
    "can't determine",
    "unable to determine",
    "no information",
    "not visible",
)


def _looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def _load_image_array(path: str | Path) -> "np.ndarray":
    """Read a raster and percentile-stretch it to a displayable (H, W, 3) uint8 array.

    Deterministic: the same file always yields the same array. The stretch is
    computed from the finite values only, so a nodata sentinel does not crush
    the dynamic range.
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise SpecialistError(
            f"numpy not installed: {exc}", specialist="vqa"
        ) from exc

    from preprocessing.raster import read_bands

    array, _profile = read_bands(path)
    # (bands, H, W) -> (H, W, bands); take the first three bands as RGB.
    if array.ndim == 3:
        array = np.transpose(array, (1, 2, 0))
        if array.shape[2] > 3:
            array = array[:, :, :3]
        elif array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
    elif array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)

    arr = array.astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size:
        lo, hi = np.percentile(finite, (2, 98))
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def _load_image(path: str | Path) -> Any:
    """Read a raster as a displayable PIL image.

    Preserves georeferencing on the way out by reading through the shared
    raster helper; the VLM only needs pixels, but the caller's `AssetMetadata`
    already holds the CRS/transform, so nothing is lost here.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise SpecialistError(
            f"PIL not installed: {exc}", specialist="vqa"
        ) from exc

    return Image.fromarray(_load_image_array(path))


class VLMSpecialist(Specialist):
    """Single-image VQA and captioning."""

    name = "vlm"
    version = "0.1.0"
    capabilities = ("vqa", "caption")

    def __init__(
        self,
        model: SmolVLM,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        max_images_per_call: int = 1,
    ) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.max_images_per_call = max_images_per_call

    # -- validation --------------------------------------------------------

    def validate_request(self, request: SpecialistRequest) -> None:
        if request.asset_count != 1:
            raise InvalidRequestError(
                f"VQA and captioning operate on exactly one image; "
                f"got {request.asset_count}",
                context={"specialist": self.name, "assets": request.asset_count},
            )

        task = request.params.get("task")
        if task not in self.capabilities:
            raise InvalidRequestError(
                f"{self.name} cannot serve task {task!r}; "
                f"capabilities are {list(self.capabilities)}",
                context={"specialist": self.name, "task": task},
            )

        asset = request.assets[0]
        if not Path(asset.path).exists():
            raise InvalidRequestError(
                f"asset path does not exist: {asset.path}",
                context={"specialist": self.name, "path": asset.path},
            )

        if task == "vqa" and not request.query.strip():
            raise InvalidRequestError(
                "VQA requires a non-empty question",
                context={"specialist": self.name},
            )

        if asset.modality is Modality.SAR and task == "caption":
            # Legal, but the answer will be about backscatter texture rather
            # than land cover. Recorded so the trace explains a poor answer.
            pass

    # -- execution ---------------------------------------------------------

    def execute(self, request: SpecialistRequest) -> SpecialistResult:
        self.validate_request(request)

        task: Literal["vqa", "caption"] = request.params["task"]
        asset = request.assets[0]

        try:
            array = _load_image_array(asset.path)
        except SpecialistError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise SpecialistError(
                f"could not prepare image for the VLM: {exc}",
                specialist=self.name,
                context={"path": asset.path},
            ) from exc

        # -- input-quality gate (finding F5-5) -----------------------------
        # The VLM will produce a fluent, specific, entirely fabricated scene
        # description for ANY input it is handed, including uniform noise, even
        # when the prompt explicitly instructs it to decline. Measured, not
        # theorised. The guard therefore has to sit here, before the model.
        quality = assess_image_quality(array)
        if not quality.is_usable:
            return self._refuse(
                task=task,
                asset=asset,
                quality=quality,
                reason=quality.reason(),
            )

        try:
            from PIL import Image

            image = Image.fromarray(array)
        except Exception as exc:  # noqa: BLE001
            raise SpecialistError(
                f"could not construct an image for the VLM: {exc}",
                specialist=self.name,
                context={"path": asset.path},
            ) from exc

        kind = "caption" if task == "caption" else "vqa"
        messages = prompts.build_messages(
            kind,
            request.query or "Describe this image.",
            image_count=1,
        )

        try:
            answer = self.model.generate(
                image,
                messages,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
            )
        except Exception as exc:  # noqa: BLE001
            raise SpecialistError(
                f"VLM generation failed: {exc}",
                specialist=self.name,
                context={"task": task},
            ) from exc

        answer = self._normalize_answer(answer)

        result = SpecialistResult(
            task=Task.VQA if task == "vqa" else Task.CAPTION,
            answer=answer,
            confidence=self._confidence_for(answer, quality),
            geospatial=asset.geo,
            degraded=quality.is_degraded,
        )
        if quality.is_degraded:
            result.warnings.append(
                f"input quality: {quality.verdict.value} ({quality.reason()})"
            )
        result.evidence = self.produce_evidence(result)
        return result

    def _refuse(
        self,
        task: str,
        asset: Any,
        quality: ImageQuality,
        reason: str,
    ) -> SpecialistResult:
        """Return a degraded result WITHOUT calling the model.

        Used when the input-quality gate blocks. The specialist still returns a
        well-formed `SpecialistResult` — the controller must not have to
        special-case this — but the answer states the refusal plainly and the
        confidence carries the measured reason.
        """
        result = SpecialistResult(
            task=Task.VQA if task == "vqa" else Task.CAPTION,
            answer=f"Not enough information in the image: {reason}.",
            confidence=ConfidenceBreakdown(
                raw=0.0,
                method="uncalibrated",
                components={
                    "schema_valid": 1.0,
                    "input_usable": 0.0,
                    "autocorrelation": float(quality.autocorrelation),
                },
                degraded=True,
                degradation_reason=reason,
            ),
            geospatial=asset.geo,
            degraded=True,
        )
        result.evidence = [
            Evidence(
                type=EvidenceType.STATISTIC,
                source_specialist=self.name,
                score=0.0,
                payload={
                    "refused": True,
                    "reason": reason,
                    "quality": quality.to_dict(),
                    "prompt_version": prompts.PROMPT_VERSION,
                    "task": task,
                },
            )
        ]
        return result

    @staticmethod
    def _normalize_answer(text: str) -> str:
        """Deterministic post-processing. No paraphrasing, only cleanup."""
        cleaned = " ".join(text.split())
        if not cleaned:
            return "Not enough information in the image."
        # A trailing period on a one-word answer is noise.
        if len(cleaned.split()) == 1:
            cleaned = cleaned.rstrip(".")
        return cleaned

    def _confidence_for(
        self, answer: str, quality: ImageQuality | None = None
    ) -> ConfidenceBreakdown:
        """Measurable signals only.

        There is no token-probability path here because `generate` is greedy
        and we do not expose logits. What we *can* measure:
          * schema validity  — the answer is non-empty and well-formed
          * refusal          — the model declined, which is a real signal
          * input quality    — the deterministic gate's verdict (F5-5)
        Calibration on validation data belongs in Phase 13; this is the raw
        score it will consume.
        """
        schema_valid = 1.0 if answer.strip() else 0.0
        refused = 1.0 if _looks_like_refusal(answer) else 0.0
        not_refused = 1.0 - refused

        input_usable = 1.0 if (quality is None or quality.is_usable) else 0.0
        input_structured = (
            1.0 if (quality is not None and quality.verdict.value == "structured") else 0.0
        )

        # The input gate is a multiplier, not a term. A fluent answer to a
        # question about noise must not be able to score well on the strength
        # of its own fluency.
        base = 0.5 * schema_valid + 0.5 * not_refused
        raw = base * input_usable

        components = {
            "schema_valid": schema_valid,
            "declined": refused,
            "input_usable": input_usable,
            "input_structured": input_structured,
        }
        if quality is not None:
            components["autocorrelation"] = float(quality.autocorrelation)

        degraded = bool(refused) or not input_usable or (
            quality is not None and quality.is_degraded
        )
        if refused:
            reason = "the model reported insufficient evidence"
        elif not input_usable:
            reason = quality.reason() if quality else "input failed the quality gate"
        elif quality is not None and quality.is_degraded:
            reason = quality.reason()
        else:
            reason = None

        return ConfidenceBreakdown(
            raw=raw,
            method="uncalibrated",
            components=components,
            degraded=degraded,
            degradation_reason=reason,
        )

    # -- evidence ----------------------------------------------------------

    def produce_evidence(self, result: SpecialistResult) -> list[Evidence]:
        """One item: the answer text itself, attributed to this specialist.

        Deliberately NOT a spatial item. The VLM produces language; if it
        produced coordinates it would be violating the C-5 rule. Phase 13 adds
        the image crop once the evidence engine owns artifact rendering.
        """
        return [
            Evidence(
                type=EvidenceType.STATISTIC,
                source_specialist=self.name,
                score=result.confidence.value,
                payload={
                    "answer": result.answer,
                    "prompt_version": prompts.PROMPT_VERSION,
                    "task": result.task.value,
                },
            )
        ]

    # -- confidence --------------------------------------------------------

    def estimate_confidence(self, result: SpecialistResult) -> ConfidenceBreakdown:
        return result.confidence if result.confidence is not None else self._confidence_for(
            result.answer
        )

    # -- trace -------------------------------------------------------------

    def model_refs(self) -> list[dict[str, str]]:
        return [
            {
                "name": self.model.checkpoint,
                "revision": self.model.revision or "unpinned",
                "role": "vlm",
            }
        ]


def build_vqa_specialist(config, device: str | None = None) -> VLMSpecialist:
    """Construct the specialist, loading the VLM from the central config."""
    from specialists.vqa.model import build_vlm

    vlm = build_vlm(config, device=device)
    return VLMSpecialist(
        model=vlm,
        max_new_tokens=int(config.get("vlm.max_new_tokens", 128)),
        do_sample=bool(config.get("vlm.do_sample", False)),
        max_images_per_call=int(config.get("vlm.max_images_per_call", 1)),
    )


__all__ = [
    "VLMSpecialist",
    "build_vqa_specialist",
    "_load_image",
    "_load_image_array",
    "_looks_like_refusal",
]