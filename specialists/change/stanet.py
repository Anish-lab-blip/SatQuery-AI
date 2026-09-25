"""STANet-style Siamese change detector.

Bi-temporal change detection over a pair of co-registered images. The
architecture follows STANet's shape — shared Siamese encoder, spatial-temporal
attention over feature differences, feature-difference aggregation decoder —
reimplemented rather than vendored, per the Phase 0 resolution of finding C-9
(the upstream repo is Python 3.6-era and depends on `visdom`/`apex`).

    T1 ---> [ shared encoder ] ---> f1  (4 levels)
    T2 ---> [ shared encoder ] ---> f2  (4 levels)
                                      |
                          |f1 - f2| + concat([f1, f2, |f1-f2|])
                                      |
                              spatial attention (PAM)
                                      |
                            progressive decoder + skips
                                      |
                              1-channel change logit

WEIGHTS ARE TIED, NOT COPIED
----------------------------
Both branches call the SAME module instance. There is no second encoder to
fall out of sync. This is the "shared encoder" requirement from plan section 16
enforced by construction, not by convention.

MEASURED ENCODER SHAPES (ResNet18 trunk, 256x256 input)
-------------------------------------------------------
    layer1   64 ch @ 64x64    4096 positions
    layer2  128 ch @ 32x32    1024 positions
    layer3  256 ch @ 16x16     256 positions
    layer4  512 ch @  8x8       64 positions

THE ATTENTION MEMORY LINE IS COMPUTED, NOT ASSUMED
--------------------------------------------------
STANet's PAM builds a full `positions x positions` attention matrix. At batch
8, fp32, that costs:

    layer1   8 * 4096^2 * 4 =  537.0 MB   <- exceeds a Kaggle session's budget
    layer2   8 * 1024^2 * 4 =   33.5 MB   <- fine
    layer3   8 *  256^2 * 4 =    2.1 MB   <- fine
    layer4   8 *   64^2 * 4 =    0.13 MB  <- fine

So attention is applied at layers 2, 3 and 4 and SKIPPED at layer 1, where the
difference features are fused directly instead. The decision is recomputed on
every forward pass from the ACTUAL batch size and a byte budget, so changing
batch size or tile size moves the line correctly rather than silently
overrunning memory. Skips are returned in the output so the trace can report
them.

This is a deviation from a literal STANet reproduction, and it is recorded as
one: attention at 64x64 would not fit. It is not a silent substitution.

PRETRAINED WEIGHTS ARE NOT A HARD DEPENDENCY
--------------------------------------------
Config says `change.pretrained: true`. A first run with no local cache needs a
download from download.pytorch.org, which may be unavailable. Resolution order:

    1. a local weights file, if one is given
    2. torchvision's IMAGENET1K_V1 download (cached after the first success)
    3. random initialisation, with an explicit warning

Random init is a legitimate starting point for change detection — the task is
pixel-pair comparison, not ImageNet classification — but the caller is told,
because "pretrained: true" in a config must not become a lie when the download
fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.errors import ModelLoadError, SpecialistError

#: Attention is skipped at any level whose attention matrix would exceed this.
#: 256 MB leaves room for activations, gradients and the optimizer state inside
#: a 15 GB T4 while comfortably admitting layer2 (33.5 MB) and excluding
#: layer1 (537 MB) at the standard batch size of 8.
DEFAULT_ATTENTION_BUDGET_BYTES = 256 * 1024 * 1024

#: Encoder output channels, verified by probe against torchvision resnet18.
ENCODER_CHANNELS: tuple[int, ...] = (64, 128, 256, 512)

#: Negative bias on the final change logit. LEVIR-CD's changed-pixel fraction is
#: roughly 5-15%, so a bias of -2.0 starts the prior near 0.12 rather than 0.5
#: and stops the first epochs being spent un-learning a saturated sigmoid.
FINAL_BIAS_INIT = -2.0


@dataclass
class ChangeOutput:
    """Model output plus the facts needed to interpret it."""

    logits: torch.Tensor          # (B, 1, H, W) at input resolution
    probabilities: torch.Tensor   # sigmoid(logits)
    attention_applied: list[int] = field(default_factory=list)
    attention_skipped: list[int] = field(default_factory=list)
    attention_bytes: dict[int, int] = field(default_factory=dict)

    def to_trace(self) -> dict[str, Any]:
        return {
            "attention_applied": list(self.attention_applied),
            "attention_skipped": list(self.attention_skipped),
            "attention_bytes": {str(k): v for k, v in self.attention_bytes.items()},
        }


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class SharedResNetEncoder(nn.Module):
    """ResNet18 trunk producing a 4-level feature pyramid.

    One instance, called twice. Tied weights are the whole point.
    """

    def __init__(
        self,
        pretrained: bool = False,
        weights_path: str | Path | None = None,
        frozen: bool = False,
    ) -> None:
        super().__init__()

        try:
            from torchvision.models import resnet18
        except ImportError as exc:  # pragma: no cover
            raise ModelLoadError(
                f"torchvision is required for the change encoder: {exc}",
                specialist="change",
            ) from exc

        self.pretrained_used = False
        self.load_warning: str | None = None

        weights = None
        if weights_path is not None:
            # Local file: no network, fully pinned. Preferred.
            path = Path(weights_path)
            if not path.exists():
                raise ModelLoadError(
                    f"encoder weights_path does not exist: {path}",
                    specialist="change",
                )
            try:
                state = torch.load(str(path), map_location="cpu", weights_only=True)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
            except Exception as exc:  # noqa: BLE001
                raise ModelLoadError(
                    f"could not read encoder weights from {path}: {exc}",
                    specialist="change",
                ) from exc
            model = resnet18(weights=None)
            missing, unexpected = model.load_state_dict(state, strict=False)
            # strict=False because a checkpoints dict may carry head weights we
            # do not use; anything MISSING from the trunk is a real problem.
            trunk_missing = [
                k for k in missing
                if not any(k.startswith(p) for p in ("fc.",))
            ]
            if trunk_missing:
                raise ModelLoadError(
                    f"encoder weights at {path} are missing {len(trunk_missing)} "
                    f"trunk parameter(s), e.g. {trunk_missing[:3]}",
                    specialist="change",
                    context={"weights_path": str(path)},
                )
            self.pretrained_used = True
        elif pretrained:
            try:
                from torchvision.models import ResNet18_Weights

                model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                self.pretrained_used = True
            except Exception as exc:  # noqa: BLE001
                model = resnet18(weights=None)
                self.load_warning = (
                    f"pretrained ResNet18 weights could not be obtained "
                    f"({type(exc).__name__}: {exc}); fell back to random "
                    f"initialisation. Change detection compares pixel pairs "
                    f"rather than classifying scenes, so this is a legitimate "
                    f"starting point -- but it is not what change.pretrained "
                    f"requested."
                )
        else:
            model = resnet18(weights=None)

        self.stem = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.frozen = frozen
        if frozen:
            for param in self.parameters():
                param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Returns [L1, L2, L3, L4], coarse last."""
        h = self.stem(x)
        h = self.layer1(h)
        l1 = h
        h = self.layer2(h)
        l2 = h
        h = self.layer3(h)
        l3 = h
        h = self.layer4(h)
        l4 = h
        return [l1, l2, l3, l4]

    def out_channels(self) -> tuple[int, ...]:
        return ENCODER_CHANNELS


# ---------------------------------------------------------------------------
# Spatial-temporal attention
# ---------------------------------------------------------------------------


def attention_matrix_bytes(batch: int, positions: int, dtype_bytes: int = 4) -> int:
    """Bytes for a full `positions x positions` attention matrix."""
    return int(batch) * int(positions) * int(positions) * dtype_bytes


class SpatialAttention(nn.Module):
    """PAM-style self-attention over the spatial positions of one feature map.

    STANet's position attention module. Applied to the fused difference
    representation, it lets a changed region reinforce itself across the whole
    feature map rather than being judged from a local receptive field alone.

    `forward` returns the attended features AND whether attention actually ran,
    so a caller can distinguish "attention was applied" from "attention was
    skipped for memory" without inspecting internals.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.query = nn.Conv2d(channels, hidden, 1)
        self.key = nn.Conv2d(channels, hidden, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        self.scale = float(hidden) ** 0.5

    def forward(
        self, x: torch.Tensor, budget_bytes: int
    ) -> tuple[torch.Tensor, bool, int]:
        """Args:
            x: (B, C, H, W).
            budget_bytes: skip attention if the matrix would exceed this.

        Returns:
            (features, applied, bytes)
        """
        b, c, h, w = x.shape
        positions = h * w
        needed = attention_matrix_bytes(b, positions, x.element_size())

        if needed > budget_bytes:
            # Skip. The difference features pass through unchanged; the decoder
            # still sees them, it just does not get global self-attention at
            # this resolution.
            return x, False, needed

        q = self.query(x).view(b, -1, positions).permute(0, 2, 1)   # (B,N,hidden)
        k = self.key(x).view(b, -1, positions)                       # (B,hidden,N)
        attn = torch.softmax(q @ k / self.scale, dim=-1)             # (B,N,N)

        v = self.value(x).view(b, -1, positions).permute(0, 2, 1)    # (B,N,C)
        attended = (attn @ v).permute(0, 2, 1).view(b, c, h, w)
        return x + self.out(attended), True, needed


# ---------------------------------------------------------------------------
# Difference fusion
# ---------------------------------------------------------------------------


class DifferenceFusion(nn.Module):
    """Reduce `concat([f1, f2, |f1-f2|])` back to a working width."""

    def __init__(self, channels: int, out_channels: int) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(channels * 3, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(f1 - f2)
        return self.reduce(torch.cat([f1, f2, diff], dim=1))


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class DecoderBlock(nn.Module):
    """Upsample, concatenate a skip, then convolve."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------


class STANetStyleChangeDetector(nn.Module):
    """Siamese change detector with spatial-temporal attention."""

    def __init__(
        self,
        *,
        width: int = 128,
        pretrained: bool = False,
        weights_path: str | Path | None = None,
        frozen_encoder: bool = False,
        sa_mode: str = "PAM",
        attention_budget_bytes: int = DEFAULT_ATTENTION_BUDGET_BYTES,
    ) -> None:
        super().__init__()

        if sa_mode not in {"PAM", "BAM"}:
            raise SpecialistError(
                f"sa_mode must be 'PAM' or 'BAM', got {sa_mode!r}",
                specialist="change",
            )
        if width < 8:
            raise SpecialistError(
                f"decoder width must be >= 8, got {width}", specialist="change"
            )
        if attention_budget_bytes < 0:
            raise SpecialistError(
                f"attention budget must be >= 0, got {attention_budget_bytes}",
                specialist="change",
            )

        self.width = width
        self.sa_mode = sa_mode
        self.attention_budget_bytes = attention_budget_bytes

        self.encoder = SharedResNetEncoder(
            pretrained=pretrained,
            weights_path=weights_path,
            frozen=frozen_encoder,
        )

        channels = self.encoder.out_channels()

        # Per-level difference fusion, then attention where it fits.
        self.fuse = nn.ModuleList(
            [DifferenceFusion(c, width) for c in channels]
        )
        self.attention = nn.ModuleList(
            [SpatialAttention(width) for _ in channels]
        )

        # Progressive decoder: 8x8 -> 16 -> 32 -> 64 -> 256
        self.dec3 = DecoderBlock(width, width, width)
        self.dec2 = DecoderBlock(width, width, width)
        self.dec1 = DecoderBlock(width, width, width // 2)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(width // 2, width // 2, 4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(width // 2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(width // 2, 1, 1)

        self._init_head()

    def _init_head(self) -> None:
        nn.init.normal_(self.head.weight, std=0.01)
        with torch.no_grad():
            self.head.bias.fill_(FINAL_BIAS_INIT)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> ChangeOutput:
        """Args:
            t1, t2: (B, 3, H, W) float tensors, already normalized consistently.

        Returns:
            ChangeOutput with logits at input resolution.

        Raises:
            SpecialistError: shapes disagree, or the spatial dimensions are not
                divisible by 8 (the encoder's total stride).
        """
        if t1.shape != t2.shape:
            raise SpecialistError(
                f"T1 and T2 must have the same shape; got {tuple(t1.shape)} "
                f"and {tuple(t2.shape)}",
                specialist="change",
            )
        if t1.dim() != 4:
            raise SpecialistError(
                f"expected (B, C, H, W) input, got {tuple(t1.shape)}",
                specialist="change",
            )
        h, w = t1.shape[-2:]
        if h % 8 or w % 8:
            raise SpecialistError(
                f"spatial dimensions must be divisible by 8 (encoder stride); "
                f"got {h}x{w}",
                specialist="change",
            )

        f1_levels = self.encoder(t1)
        f2_levels = self.encoder(t2)

        applied: list[int] = []
        skipped: list[int] = []
        byte_map: dict[int, int] = {}

        fused: list[torch.Tensor] = []
        for level, (a, b) in enumerate(zip(f1_levels, f2_levels)):
            diff = self.fuse[level](a, b)
            if self.sa_mode == "PAM":
                diff, ran, needed = self.attention[level](
                    diff, self.attention_budget_bytes
                )
                byte_map[level] = needed
                (applied if ran else skipped).append(level)
            else:
                # BAM is the channel/co-occurrence variant in STANet. Kept as a
                # declared-but-unimplemented branch rather than silently
                # aliasing PAM, so a config asking for BAM fails visibly here.
                raise SpecialistError(
                    "sa_mode='BAM' is declared in config but not implemented; "
                    "set change.sa_mode to 'PAM'",
                    specialist="change",
                )
            fused.append(diff)

        # Coarse-to-fine.
        x = fused[3]
        x = self.dec3(x, fused[2])
        x = self.dec2(x, fused[1])
        x = self.dec1(x, fused[0])
        x = self.final_up(x)

        if x.shape[-2:] != (h, w):
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

        logits = self.head(x)
        return ChangeOutput(
            logits=logits,
            probabilities=torch.sigmoid(logits),
            attention_applied=applied,
            attention_skipped=skipped,
            attention_bytes=byte_map,
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "sa_mode": self.sa_mode,
            "attention_budget_bytes": self.attention_budget_bytes,
            "encoder": "resnet18",
            "encoder_channels": list(ENCODER_CHANNELS),
            "frozen_encoder": self.encoder.frozen,
            "pretrained_used": self.encoder.pretrained_used,
        }

    @classmethod
    def from_config_dict(cls, payload: dict[str, Any]) -> "STANetStyleChangeDetector":
        return cls(
            width=int(payload["width"]),
            sa_mode=str(payload.get("sa_mode", "PAM")),
            attention_budget_bytes=int(
                payload.get("attention_budget_bytes", DEFAULT_ATTENTION_BUDGET_BYTES)
            ),
            frozen_encoder=bool(payload.get("frozen_encoder", False)),
            pretrained=False,
        )

    def __repr__(self) -> str:
        return (
            f"<STANetStyleChangeDetector width={self.width} sa={self.sa_mode} "
            f"params={self.num_parameters():,} "
            f"trainable={self.trainable_parameters():,}>"
        )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


@dataclass
class ChangeLossBreakdown:
    total: torch.Tensor
    bce: torch.Tensor
    dice: torch.Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "bce": float(self.bce.detach()),
            "dice": float(self.dice.detach()),
        }


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice over the change class.

    BCE alone is minimised by predicting "no change" everywhere when change is
    rare, which is exactly the imbalance LEVIR-CD has. Dice is computed on the
    soft probabilities so it is differentiable and directly optimises region
    overlap.

    eps guards the empty-target case: a tile with no change has numerator 0 and
    denominator 0, which would be NaN without it.
    """
    prob = torch.sigmoid(logits)
    prob_flat = prob.reshape(prob.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    intersection = (prob_flat * target_flat).sum(dim=1)
    denominator = prob_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def change_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    pos_weight: float | None = None,
) -> ChangeLossBreakdown:
    """Composite BCE + Dice. Weights sum to 1 by convention, not requirement.

    Args:
        logits: (B, 1, H, W) raw model output.
        target: (B, 1, H, W) binary change mask, 0/1 floats.
        bce_weight, dice_weight: component weights.
        pos_weight: optional positive-class weight for the BCE term. When None
            the BCE is unweighted; the Dice term already addresses the class
            imbalance, so this is left as an explicit tuning knob rather than a
            hidden default.
    """
    if logits.shape != target.shape:
        raise SpecialistError(
            f"logits {tuple(logits.shape)} and target {tuple(target.shape)} "
            f"must have the same shape",
            specialist="change",
        )

    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight
    )
    dice = dice_loss(logits, target)
    total = bce_weight * bce + dice_weight * dice
    return ChangeLossBreakdown(total=total, bce=bce, dice=dice)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_change_model(
    config: Any,
    *,
    weights_path: str | Path | None = None,
    device: str | None = None,
) -> STANetStyleChangeDetector:
    """Construct the detector from the central config."""
    device = device or config.device_preference

    model = STANetStyleChangeDetector(
        width=int(config.get("change.decoder_width", 128)),
        pretrained=bool(config.get("change.pretrained", False)),
        weights_path=weights_path,
        frozen_encoder=bool(config.get("change.frozen_encoder", False)),
        sa_mode=str(config.get("change.sa_mode", "PAM")),
        attention_budget_bytes=int(
            config.get(
                "change.attention_budget_bytes", DEFAULT_ATTENTION_BUDGET_BYTES
            )
        ),
    )
    model.to(device)
    return model


def save_change_model(path: str | Path, model: STANetStyleChangeDetector, metadata: dict[str, Any] | None = None) -> Path:
    """Write the model plus its reconstruction config."""
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {"state_dict": model.state_dict(), "config": model.config_dict()},
        str(p),
    )
    if metadata is not None:
        (p.parent / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return p


def load_change_model(path: str | Path, device: str = "cpu") -> STANetStyleChangeDetector:
    """Load a trained detector. Strict, so an architecture drift fails loudly."""
    p = Path(path)
    if not p.exists():
        raise ModelLoadError(
            f"change model not found: {p}", specialist="change"
        )
    try:
        payload = torch.load(str(p), map_location=device, weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(
            f"could not read change model from {p}: {exc}", specialist="change"
        ) from exc

    config = payload.get("config")
    if not config:
        raise ModelLoadError(
            "change model checkpoint has no embedded config; it cannot be "
            "reconstructed without guessing the architecture",
            specialist="change",
        )

    model = STANetStyleChangeDetector.from_config_dict(config)
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(
            f"change model state_dict does not match the reconstructed "
            f"architecture: {exc}",
            specialist="change",
        ) from exc

    model.to(device)
    model.eval()
    return model


__all__ = [
    "ChangeOutput",
    "ChangeLossBreakdown",
    "SharedResNetEncoder",
    "SpatialAttention",
    "DifferenceFusion",
    "DecoderBlock",
    "STANetStyleChangeDetector",
    "attention_matrix_bytes",
    "dice_loss",
    "change_loss",
    "build_change_model",
    "save_change_model",
    "load_change_model",
    "DEFAULT_ATTENTION_BUDGET_BYTES",
    "ENCODER_CHANNELS",
    "FINAL_BIAS_INIT",
]