"""SatQuery AI — optical-SAR inference pipeline (freeze section 2.5).

The seam between "an image on disk" and "three GAP vectors plus two masks".
Kept separate from `specialist.py` so the tensor plumbing can be tested without
a `Specialist` and without a raster, and separate from `croma.py` so that module
stays a thin wrapper over someone else's model.

THE PIPELINE
------------
    optical raster -> sensor adapter -> (12, H, W) + mask[12]
    SAR raster     -> sensor adapter -> ( 2, H, W) + mask[2]
    resize to CROMA's 120x120
    CROMA(SAR_images=..., optical_images=...)     <- no mask, ever
    assemble_fusion_input                         <- mask enters HERE
    fusion head -> logits -> probabilities

DEGRADATION IS A FIRST-CLASS OUTCOME
------------------------------------
Two artifacts are needed and neither is guaranteed to exist in this environment:
`CROMA_base.pt` (777.6 MB) and the vendored `use_croma.py`, plus a trained fusion
head (which has never been trained).

Per Phase 8's precedent (`grounding/specialist.py:has_head`) and Phase 9's
(`change/specialist.py:has_checkpoint`), a missing artifact is a DEPLOYMENT case,
not a crash. `run_pipeline` reports which pieces were missing and returns a
result marked degraded rather than raising -- but it never *invents* a
prediction to fill the gap. When the head is absent there is no class output at
all, and the caller is told so in `warnings`.

WHAT IS COMPUTED WITHOUT ANY ARTIFACT
-------------------------------------
Everything on the sensor side. The masks, the canonical channel placement, the
availability statistics and the modality validation are all real work that
requires no weights, and they are exactly the facts that tell an operator why a
result is or is not trustworthy on Cartosat-2S + RISAT. Those keep running when
the models do not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.errors import ModelLoadError, SpecialistError
from core.schemas import SensorDescriptor
from specialists.optical_sar.fusion_head import (
    FusionInput,
    assemble_fusion_input,
    expected_fusion_dim,
    mask_availability_stats,
)
from specialists.optical_sar.sensor_adapter import (
    SensorAdapterOutput,
    adapt_optical,
    adapt_sar,
)


@dataclass
class OpticalSarInference:
    """Everything one forward pass produced, plus what was missing."""

    fusion_input: FusionInput
    optical: SensorAdapterOutput
    sar: SensorAdapterOutput
    probabilities: np.ndarray | None = None      # (B, task_dim) or None
    predicted_index: int | None = None
    margin: float = 0.0
    agreement: float | None = None
    warnings: list[str] = field(default_factory=list)
    degraded: bool = False
    degradation_reasons: list[str] = field(default_factory=list)

    @property
    def has_prediction(self) -> bool:
        return self.probabilities is not None and self.predicted_index is not None

    def availability(self) -> dict[str, float]:
        return mask_availability_stats(
            self.fusion_input.optical_mask, self.fusion_input.sar_mask
        )


def resize_to_canonical(array: np.ndarray, resolution: int) -> np.ndarray:
    """Resize a (C, H, W) array to (C, resolution, resolution), bilinearly.

    Everything in the chain is in the same units, so plain interpolation is
    correct. Nearest-neighbour would be wrong for the continuous optical
    reflectance channels; for the zero-filled channels it costs nothing either
    way, since interpolating zeros gives zeros.
    """
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 3:
        raise SpecialistError(
            f"expected (C, H, W), got {arr.shape}", specialist="optical_sar"
        )
    if arr.shape[1] == resolution and arr.shape[2] == resolution:
        return arr

    try:
        import cv2
    except ImportError:  # pragma: no cover
        # Fall back to torch's interpolate, which is already a dependency.
        import torch

        t = torch.from_numpy(arr).unsqueeze(0)
        out = torch.nn.functional.interpolate(
            t, size=(resolution, resolution), mode="bilinear", align_corners=False
        )
        return out.squeeze(0).numpy()

    channels = [
        cv2.resize(arr[c], (resolution, resolution), interpolation=cv2.INTER_LINEAR)
        for c in range(arr.shape[0])
    ]
    return np.stack(channels, axis=0).astype(np.float32)


def build_masks(optical: SensorAdapterOutput, sar: SensorAdapterOutput) -> tuple[np.ndarray, np.ndarray]:
    """The two availability masks as (1, C) float32, ready for concatenation."""
    return (
        optical.mask.astype(np.float32)[np.newaxis, :],
        sar.mask.astype(np.float32)[np.newaxis, :],
    )


def run_pipeline(
    *,
    optical_array: np.ndarray,
    sar_array: np.ndarray,
    optical_descriptor: SensorDescriptor,
    sar_descriptor: SensorDescriptor,
    encoder: Any | None = None,
    head: Any | None = None,
    resolution: int = 120,
    task_dim: int = 19,
    device: str = "cpu",
) -> OpticalSarInference:
    """Canonicalise, encode and (if possible) classify.

    Args:
        optical_array: raw (bands, H, W) optical.
        sar_array: raw (bands, H, W) SAR.
        optical_descriptor / sar_descriptor: from the sensor adapter.
        encoder: a `CROMAEncoder`, or None when CROMA is unavailable.
        head: a trained fusion head, or None when there is no trained artifact.
        resolution: CROMA input size.
        task_dim: classifier output width.
        device: torch device.

    Returns:
        An `OpticalSarInference`. NEVER raises for a missing model: the
        degradation is reported, because "we have no trained head" is a fact
        about the deployment, not a failure of the request.

    Raises:
        SpecialistError: the caller passed arrays that cannot be canonicalised at
            all -- a programming error, not a deployment state.
    """
    warnings: list[str] = []
    reasons: list[str] = []

    # -- sensor side: always runs -----------------------------------------
    optical = adapt_optical(optical_array, optical_descriptor)
    sar = adapt_sar(sar_array, sar_descriptor)

    optical_mask, sar_mask = build_masks(optical, sar)
    stats = mask_availability_stats(optical_mask, sar_mask)

    if stats["optical_channels_present"] == 0:
        raise SpecialistError(
            "no optical channel could be placed; the optical input carries no "
            "usable band for the canonical layout",
            specialist="optical_sar",
            context={"sensor": optical_descriptor.sensor},
        )
    if stats["sar_channels_present"] == 0:
        raise SpecialistError(
            "no SAR polarisation could be placed; the SAR input carries no "
            "usable band for the canonical layout",
            specialist="optical_sar",
            context={"sensor": sar_descriptor.sensor},
        )

    if stats["optical_channels_present"] < optical_descriptor.canonical_channels:
        warnings.append(
            f"{int(optical_descriptor.canonical_channels - stats['optical_channels_present'])} "
            f"of {optical_descriptor.canonical_channels} optical channels are "
            f"absent from sensor '{optical_descriptor.sensor}' and are "
            f"zero-filled with the availability mask set false. No band was "
            f"invented to fill them."
        )
    if stats["sar_channels_present"] < sar_descriptor.canonical_channels:
        warnings.append(
            f"{int(sar_descriptor.canonical_channels - stats['sar_channels_present'])} "
            f"of {sar_descriptor.canonical_channels} SAR channels are absent "
            f"from sensor '{sar_descriptor.sensor}' and are zero-filled."
        )

    # -- CROMA -------------------------------------------------------------
    if encoder is None:
        reasons.append("CROMA is not loaded")
        warnings.append(
            "CROMA is not loaded, so no optical/SAR/joint representation was "
            "computed. Only the sensor-side facts (canonical channel placement "
            "and availability) are available. Supply CROMA_base.pt and the "
            "vendored use_croma.py to enable fusion."
        )
        placeholder = _empty_fusion_input(expected_fusion_dim(), optical_mask, sar_mask, device)
        return OpticalSarInference(
            fusion_input=placeholder,
            optical=optical,
            sar=sar,
            warnings=warnings,
            degraded=True,
            degradation_reasons=reasons,
        )

    optical_tensor = resize_to_canonical(optical.canonical, resolution)
    sar_tensor = resize_to_canonical(sar.canonical, resolution)

    encoding = encoder.encode(
        optical_tensor[np.newaxis, ...], sar_tensor[np.newaxis, ...]
    )

    if encoding.n_patches != (resolution // 8) ** 2:
        warnings.append(
            f"CROMA returned {encoding.n_patches} patches; at {resolution}px "
            f"with stride 8 the frozen spec expects {(resolution // 8) ** 2}"
        )

    fusion_input = assemble_fusion_input(
        encoding.as_forward_dict(),
        optical_mask=optical_mask,
        sar_mask=sar_mask,
        encoder_dim=encoding.dim,
    )

    # -- fusion head -------------------------------------------------------
    if head is None:
        reasons.append("no trained fusion head")
        warnings.append(
            "no trained fusion head is loaded, so no class prediction was "
            "produced. A randomly-initialised head would emit a label with no "
            "learned signal behind it, so none is reported. The fused "
            "representation is computed and available for inspection."
        )
        return OpticalSarInference(
            fusion_input=fusion_input,
            optical=optical,
            sar=sar,
            warnings=warnings,
            degraded=True,
            degradation_reasons=reasons,
        )

    probabilities = _classify(head, fusion_input.tensor, device=device)
    if probabilities.shape[1] != task_dim:
        raise SpecialistError(
            f"fusion head emitted {probabilities.shape[1]} classes, expected "
            f"{task_dim}",
            specialist="optical_sar",
        )

    order = np.argsort(probabilities[0])[::-1]
    predicted = int(order[0])
    margin = float(probabilities[0][order[0]] - probabilities[0][order[1]])
    agreement = cross_modal_agreement(encoding)

    if not bool(getattr(head, "_satquery_trained", False)):
        reasons.append("fusion head is not a trained artifact")
        warnings.append(
            "the fusion head is not a trained artifact; its output carries no "
            "learned signal."
        )

    return OpticalSarInference(
        fusion_input=fusion_input,
        optical=optical,
        sar=sar,
        probabilities=probabilities,
        predicted_index=predicted,
        margin=margin,
        agreement=agreement,
        warnings=warnings,
        degraded=bool(reasons),
        degradation_reasons=reasons,
    )


def _classify(head: Any, tensor: np.ndarray, *, device: str) -> np.ndarray:
    """Run the fusion head and return softmax probabilities."""
    try:
        import torch

        with torch.no_grad():
            logits = head(torch.from_numpy(tensor).to(device))
        return torch.softmax(logits.float(), dim=-1).cpu().numpy()
    except SpecialistError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SpecialistError(
            f"fusion head forward pass failed: {exc}",
            specialist="optical_sar",
            context={"input_shape": tuple(tensor.shape)},
        ) from exc


def cross_modal_agreement(encoding: Any) -> float | None:
    """Cosine similarity between the optical and SAR global vectors.

    Plan section 26 lists cross-modal agreement as one of the four optical-SAR
    confidence components. This is the cheapest honest reading of it: how
    similarly do the two encoders describe the same scene? A high value means
    both modalities point the same way; a low one means the fused decision rests
    on a disagreement.

    It is a MEASUREMENT on the encoded vectors, not a learned quantity, and it is
    NOT used as a probability -- it feeds `estimate_confidence` as one component
    among four.
    """
    try:
        import torch

        a = torch.from_numpy(np.asarray(encoding.optical_gap, dtype=np.float32))
        b = torch.from_numpy(np.asarray(encoding.sar_gap, dtype=np.float32))
        if a.shape != b.shape or a.numel() == 0:
            return None
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        return float(cos.mean().clamp(-1.0, 1.0).item())
    except Exception:  # noqa: BLE001
        return None


def _empty_fusion_input(
    dim: int,
    optical_mask: np.ndarray,
    sar_mask: np.ndarray,
    device: str,
) -> FusionInput:
    """A zero tensor of the right shape, for the no-CROMA path.

    Deliberately NOT used for a prediction. It exists so the result object has a
    consistent shape and the caller can read the masks; `probabilities` stays
    None so nothing can mistake it for an inference.
    """
    del device  # reserved: the placeholder is numpy regardless of device
    encoder_dim = 768
    zeros_gap = np.zeros((1, encoder_dim), dtype=np.float32)
    tensor = np.concatenate(
        [zeros_gap, zeros_gap, zeros_gap, optical_mask, sar_mask], axis=1
    )
    return FusionInput(
        tensor=tensor,
        optical_gap=zeros_gap,
        sar_gap=zeros_gap,
        joint_gap=zeros_gap.copy(),
        optical_mask=optical_mask,
        sar_mask=sar_mask,
    )


def build_head_from_config(config: Any, *, head_path: str | None = None, device: str = "cpu"):
    """Construct or load the fusion head.

    Returns None when no trained artifact exists, rather than an untrained head.
    That distinction is the whole point: `build_fusion_head` would happily
    produce a randomly-initialised module, and returning it would put a real
    tensor behind `has_prediction` with no learned signal in it.

    Raises:
        ModelLoadError: `head_path` was given, exists, and cannot be read. A
            corrupt artifact must never be silently downgraded to "untrained".
    """
    from specialists.optical_sar.fusion_head import build_fusion_head, load_fusion_head

    kwargs = {
        "input_dim": int(config.get("fusion.input_dim", 2318)),
        "hidden_dim": int(config.get("fusion.hidden_dim", 512)),
        "task_dim": int(config.get("fusion.num_classes", 19)),
        "dropout": float(config.get("fusion.dropout", 0.2)),
    }

    if head_path is None:
        return None

    from pathlib import Path

    if not Path(head_path).exists():
        return None

    head = load_fusion_head(head_path, device=device, **kwargs)
    return head


__all__ = [
    "OpticalSarInference",
    "run_pipeline",
    "resize_to_canonical",
    "build_masks",
    "cross_modal_agreement",
    "build_head_from_config",
]
