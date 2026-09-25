"""SatQuery AI — optical-SAR fusion head (freeze section 2.5, plan section 17).

CROMA produces three 768-dimensional vectors per sample: an optical GAP, a SAR
GAP, and a joint GAP. The fusion head turns those, plus the availability masks,
into a task prediction.

THE FROZEN CONCATENATION
------------------------
    optical_GAP      (B, 768)
    SAR_GAP          (B, 768)
    joint_GAP        (B, 768)
    optical_mask     (B,  12)     <- availability, from the sensor adapter
    sar_mask         (B,   2)     <- availability, from the sensor adapter
                     ---------
    concat           (B, 2318)

`fusion.input_dim = 2318` in configs/base.yaml, and `core/config.py` recomputes
it as `len(modalities_used) * encoder_dim + optical_channels + sar_channels` at
load time (finding C-1). This module recomputes it AGAIN from the CROMA config
and refuses to build on a mismatch, so a config edit cannot silently reshape the
first Linear layer into something that trains but means nothing.

WHY THE MASK GOES HERE AND NOT INTO CROMA  (finding C-1)
--------------------------------------------------------
CROMA is a masked autoencoder. Handing it an availability mask invites it to
reconstruct the missing channels -- which is precisely the fabrication the
sensor adapter exists to prevent: the model would output plausible values for
bands no sensor measured, and those values would then be treated as data. The
mask is therefore consumed HERE, where it can only do one thing: tell the
classifier which inputs to distrust.

MANDATORY CHANNEL/BAND DROPOUT
------------------------------
Freeze section 2.5: "Channel/band dropout during fusion-head training is
**mandatory**; it is what teaches the head to trust the availability mask."

The two are the same trick. With no dropout, the head learns to read channel 4
of the optical vector unconditionally, because in training channel 4 was always
present. On the hidden set (Cartosat-2S, 4 bands) channels 5-12 are *always*
zero -- and a head that never saw a masked channel treats those zeros as a
measurement of blackness rather than as absence. Dropout makes "this channel is
missing" a state the head has been trained under, so the mask becomes
informative instead of decorative.

`channel_dropout` is a real function, not a flag: it is tested directly, because
a training-time transform that never actually ran is one of the easiest things
to ship broken.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.errors import ModelLoadError, SpecialistError

#: Output keys CROMA's forward pass returns, verified against the official
#: README (github.com/antofuller/CROMA). Declared here so the rest of the
#: package refers to one spelling.
CROMA_GAP_KEYS: tuple[str, ...] = ("optical_GAP", "SAR_GAP", "joint_GAP")

#: Task dimensionality the head projects to. `fusion.num_classes` in config.
DEFAULT_TASK_DIM = 19


def expected_fusion_dim(
    *,
    encoder_dim: int = 768,
    optical_channels: int = 12,
    sar_channels: int = 2,
    modalities_used: int = 3,
) -> int:
    """The frozen concatenation width. 3*768 + 12 + 2 = 2318."""
    return int(modalities_used) * int(encoder_dim) + int(optical_channels) + int(sar_channels)


@dataclass(frozen=True)
class FusionInput:
    """The (B, 2318) tensor plus the pieces it was assembled from.

    Carrying the parts is not redundancy: the confidence system (plan section 26)
    needs `optical_mask` and `sar_mask` separately to report how much of each
    modality was actually present, and recovering them by slicing the
    concatenation re-derives the layout in a second place.
    """

    tensor: np.ndarray               # (B, 2318)
    optical_gap: np.ndarray          # (B, 768)
    sar_gap: np.ndarray              # (B, 768)
    joint_gap: np.ndarray            # (B, 768)
    optical_mask: np.ndarray         # (B, 12)
    sar_mask: np.ndarray             # (B, 2)

    @property
    def batch_size(self) -> int:
        return int(self.tensor.shape[0])

    @property
    def dim(self) -> int:
        return int(self.tensor.shape[1])


def assemble_fusion_input(
    croma_output: dict[str, np.ndarray],
    *,
    optical_mask: np.ndarray,
    sar_mask: np.ndarray,
    encoder_dim: int = 768,
    optical_channels: int = 12,
    sar_channels: int = 2,
) -> FusionInput:
    """Concatenate CROMA's GAP vectors with the availability masks.

    Order is frozen (freeze section 2.5). It is asserted rather than assumed,
    because a permutation here produces a tensor of exactly the right shape that
    trains to a worse number -- the hardest kind of bug to notice.

    Args:
        croma_output: the dict CROMA's forward pass returns.
        optical_mask: (B, optical_channels) float/bool availability.
        sar_mask: (B, sar_channels) float/bool availability.

    Raises:
        SpecialistError: a required key is missing, or a component has the wrong
            width.
    """
    missing = [k for k in CROMA_GAP_KEYS if k not in croma_output]
    if missing:
        raise SpecialistError(
            f"CROMA output is missing {missing}; expected keys "
            f"{list(CROMA_GAP_KEYS)}. The real forward pass was verified to "
            f"return exactly these (github.com/antofuller/CROMA README).",
            specialist="optical_sar",
            context={"available_keys": sorted(croma_output)},
        )

    def _as_2d(name: str, array: np.ndarray, width: int) -> np.ndarray:
        arr = np.asarray(array, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2 or arr.shape[1] != width:
            raise SpecialistError(
                f"{name} has shape {arr.shape}, expected (B, {width})",
                specialist="optical_sar",
            )
        return arr

    optical_gap = _as_2d("optical_GAP", croma_output["optical_GAP"], encoder_dim)
    sar_gap = _as_2d("SAR_GAP", croma_output["SAR_GAP"], encoder_dim)
    joint_gap = _as_2d("joint_GAP", croma_output["joint_GAP"], encoder_dim)

    batch = optical_gap.shape[0]
    if sar_gap.shape[0] != batch or joint_gap.shape[0] != batch:
        raise SpecialistError(
            f"CROMA GAP vectors disagree on batch size: optical "
            f"{optical_gap.shape[0]}, SAR {sar_gap.shape[0]}, joint "
            f"{joint_gap.shape[0]}",
            specialist="optical_sar",
        )

    o_mask = _as_2d("optical_mask", optical_mask, optical_channels).astype(np.float32)
    s_mask = _as_2d("sar_mask", sar_mask, sar_channels).astype(np.float32)
    if o_mask.shape[0] != batch or s_mask.shape[0] != batch:
        raise SpecialistError(
            f"availability masks disagree with the batch size: optical mask "
            f"{o_mask.shape[0]}, SAR mask {s_mask.shape[0]}, features {batch}",
            specialist="optical_sar",
        )

    tensor = np.concatenate(
        [optical_gap, sar_gap, joint_gap, o_mask, s_mask], axis=1
    )

    expected = expected_fusion_dim(
        encoder_dim=encoder_dim,
        optical_channels=optical_channels,
        sar_channels=sar_channels,
        modalities_used=len(CROMA_GAP_KEYS),
    )
    if tensor.shape[1] != expected:
        raise SpecialistError(
            f"assembled fusion input is {tensor.shape[1]}-d, expected {expected}",
            specialist="optical_sar",
        )

    return FusionInput(
        tensor=tensor,
        optical_gap=optical_gap,
        sar_gap=sar_gap,
        joint_gap=joint_gap,
        optical_mask=o_mask,
        sar_mask=s_mask,
    )


def channel_dropout(
    features: np.ndarray,
    mask: np.ndarray,
    *,
    keep_probabilities: tuple[float, ...],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop whole channels from a feature block and update its mask.

    Plan section 20's robustness schedule: optical is trained at 100/80/60/40%
    channel availability and SAR at 100/50%. Applying it means: choose, per
    sample, a fraction of the channels that WERE available, and mark the rest
    unavailable.

    The mask is updated to match the dropped features. That is the entire point
    -- dropping features while leaving the mask saying "present" would teach the
    head that the mask lies.

    Args:
        features: (B, C) feature block.
        mask: (B, C) availability, 1.0 available.
        keep_probabilities: fractions to sample from, uniformly.
        rng: a seeded generator. Seeded deliberately: an unseeded training
            transform makes a run unreproducible.

    Returns:
        (dropped_features, updated_mask). Dropped channels are zeroed, not
        renormalised -- renormalising would fabricate a scale that the real
        missing-channel case does not have.
    """
    arr = np.asarray(features, dtype=np.float32)
    msk = np.asarray(mask, dtype=np.float32)
    if arr.shape != msk.shape:
        raise SpecialistError(
            f"features {arr.shape} and mask {msk.shape} must have the same shape",
            specialist="optical_sar",
        )
    if not keep_probabilities:
        return arr.copy(), msk.copy()

    batch, channels = arr.shape
    out = arr.copy()
    out_mask = msk.copy()

    probabilities = np.asarray(keep_probabilities, dtype=np.float64)
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise SpecialistError(
            f"keep_probabilities must be in [0,1], got {list(keep_probabilities)}",
            specialist="optical_sar",
        )

    keep_fraction = rng.choice(probabilities, size=batch)
    # Bernoulli draws against the per-sample keep fraction.
    draws = rng.random((batch, channels)) < keep_fraction[:, None]

    # A channel that was already unavailable stays unavailable: the mask is a
    # statement about the SENSOR, and no training transform may resurrect a band
    # the sensor did not measure.
    keep = draws & (msk > 0.0)
    out_mask = keep.astype(np.float32)
    out = out * out_mask
    return out, out_mask


def mask_availability_stats(
    optical_mask: np.ndarray, sar_mask: np.ndarray
) -> dict[str, float]:
    """Fraction of each modality actually present. Feeds the confidence system."""
    o = np.asarray(optical_mask, dtype=np.float32)
    s = np.asarray(sar_mask, dtype=np.float32)
    return {
        "optical_available_fraction": float(o.mean()) if o.size else 0.0,
        "sar_available_fraction": float(s.mean()) if s.size else 0.0,
        "optical_channels_present": float(o.sum()),
        "sar_channels_present": float(s.sum()),
    }


# ---------------------------------------------------------------------------
# Torch head
# ---------------------------------------------------------------------------


def build_fusion_head(
    *,
    input_dim: int,
    hidden_dim: int,
    task_dim: int,
    dropout: float,
    device: str = "cpu",
):
    """Construct the frozen fusion head.

    Frozen architecture (freeze section 2.5, plan section 17):

        LayerNorm -> Linear(input_dim, hidden_dim) -> GELU -> Dropout
                  -> Linear(hidden_dim, task_dim)

    `input_dim` is checked against the recomputed frozen concatenation, so the
    first Linear layer cannot be built to the wrong width.

    Raises:
        SpecialistError: device is not 'cpu'.
        ModelLoadError: torch is unavailable, or input_dim is wrong.
    """
    if device != "cpu":
        # The plan's deployment boundary allows CPU-only operation (freeze
        # section 4: cpu_mode required). A device string that would silently fall
        # back is worse than a refusal.
        raise SpecialistError(
            f"optical_sar fusion head supports device='cpu' only; got {device!r}",
            specialist="optical_sar",
        )

    if int(input_dim) != 2318:
        raise ModelLoadError(
            f"fusion head input_dim={input_dim} does not match the frozen "
            f"concatenation 2318 (3*768 + 12 + 2). Finding C-1: CROMA emits "
            f"optical/SAR/joint GAP vectors and the availability mask is "
            f"consumed by this head.",
            specialist="optical_sar",
        )

    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - environment issue
        raise ModelLoadError(
            f"torch is not installed: {exc}", specialist="optical_sar"
        ) from exc

    class FusionHead(nn.Module):
        """LayerNorm -> Linear -> GELU -> Dropout -> Linear.

        No activation on the output: this is a logit-producing task head, and a
        softmax here would be applied twice once a loss function adds its own.
        """

        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(input_dim)
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.act = nn.GELU()
            self.drop = nn.Dropout(float(dropout))
            self.fc2 = nn.Linear(hidden_dim, task_dim)

        def forward(self, x):  # noqa: ANN001, ANN201 - torch signature
            x = self.norm(x)
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            return self.fc2(x)

        def num_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters())

    head = FusionHead()
    head.eval()
    return head


def load_fusion_head(path: str, *, device: str = "cpu", **kwargs):
    """Load a trained fusion head from disk.

    Raises:
        ModelLoadError: the file exists but cannot be read. A corrupt artifact is
            NOT the same as a missing one, and must never be silently replaced by
            an untrained head.
    """
    head = build_fusion_head(device=device, **kwargs)
    try:
        import torch

        state = torch.load(path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        head.load_state_dict(state)
        head.eval()
        head._satquery_trained = True  # type: ignore[attr-defined]
        return head
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(
            f"could not load fusion head from '{path}': {exc}",
            specialist="optical_sar",
            context={"path": str(path)},
        ) from exc


__all__ = [
    "CROMA_GAP_KEYS",
    "DEFAULT_TASK_DIM",
    "FusionInput",
    "expected_fusion_dim",
    "assemble_fusion_input",
    "channel_dropout",
    "mask_availability_stats",
    "build_fusion_head",
    "load_fusion_head",
]
