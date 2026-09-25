"""SatQuery AI — learned grounding head.

Sits on top of the FROZEN RemoteCLIP encoder and turns per-patch tokens into
boxes. This is the plan's "lightweight learned grounding head" and it is the
Phase 8 deliverable.

FROZEN INPUTS (measured, docs/PHASE7_GROUNDING_CONTRACT.md):
    patch tokens : (B, 49, 512) at 224px   -- 7x7 grid, projected dim 512
    text emb     : (B, 512)

The encoder is never trained here. Only this head moves.

WHY 224 AND NOT 448
-------------------
docs/PHASE7_RESOLUTION_DECISION.md. Over all 16,159 VRSBench eval records, 448
was worse on mean best IoU (-0.0147), worse at EVERY recall threshold
(-0.0699 / -0.0243 / -0.0022), and 1.59x the latency, with paired t = -22.63
and a 95% CI of [-0.0160, -0.0134] that excludes zero. The head is trained at
224 and the grid is 7x7.

PER-CELL FEATURE
----------------
    f_i = concat([p_i, t, p_i * t, global_pool])       -> 4 * 512 = 2048

    p_i          the patch token
    t            the text embedding, broadcast to every cell
    p_i * t      element-wise alignment (cheap cross-modal interaction)
    global_pool  mean over all patches

    The global term matters: a per-cell MLP otherwise cannot see anything
    outside its own patch, and a 1/7-of-image receptive field is too small for
    objects that span several cells.

BOX PARAMETERISATION (cell-relative, YOLO-style)
------------------------------------------------
Cell (r, c) of a grid_h x grid_w map predicts [tx, ty, tw, th, obj]:

    cx = (c + sigmoid(tx)) / grid_w
    cy = (r + sigmoid(ty)) / grid_h
    w  = sigmoid(tw)
    h  = sigmoid(th)
    box = clip((cx - w/2, cy - h/2, cx + w/2, cy + h/2), 0, 1)

Cell-relative rather than absolute because 7x7 is coarse. An absolute
regressor must learn 49 separate mappings onto the same global coordinates; a
cell-relative one only learns a local offset.

POSITIVE ASSIGNMENT
-------------------
Exactly one cell per target: the one containing the ground-truth box CENTRE.
Standard single-stage-detector convention (YOLO/FCOS), and it makes the 1-of-49
confidence imbalance explicit and therefore correctable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Cells are assigned by box centre. Ties go to the earlier cell so the
#: assignment is deterministic.
EPS = 1e-7


@dataclass
class HeadOutput:
    """Raw head output before decoding. Logits, not probabilities."""

    raw: torch.Tensor          # (B, N, 5) = [tx, ty, tw, th, objectness_logit]

    @property
    def box_params(self) -> torch.Tensor:
        return self.raw[..., :4]

    @property
    def objectness_logit(self) -> torch.Tensor:
        return self.raw[..., 4]


class GroundingHead(nn.Module):
    """Text-conditioned per-cell box regressor over frozen RemoteCLIP tokens."""

    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 512,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1:
            raise ValueError(
                f"feature_dim and hidden_dim must be positive, "
                f"got {feature_dim}, {hidden_dim}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout

        self.proj = nn.Linear(feature_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        # 5 outputs: [tx, ty, tw, th, objectness]
        self.out = nn.Linear(hidden_dim, 5)

        self._init_weights()

    def _init_weights(self) -> None:
        """Small-std init, and objectness bias low.

        With one positive cell in 49 the task starts 1:48 imbalanced. A bias of
        -2.0 starts the objectness prior near 0.12, which is closer to the
        truth than 0.5 and stops the first epochs being spent un-learning a
        saturated sigmoid.
        """
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.out.weight, std=0.01)
        with torch.no_grad():
            self.out.bias.zero_()
            self.out.bias[4] = -2.0

    def forward(self, patch_tokens: torch.Tensor, text_embedding: torch.Tensor) -> HeadOutput:
        """Args:
            patch_tokens:   (B, N, D) frozen encoder patch tokens.
            text_embedding: (B, D)     frozen encoder text embedding.

        Returns:
            HeadOutput with raw (B, N, 5) logits.

        Raises:
            ValueError: shapes disagree with the declared feature_dim.
        """
        if patch_tokens.dim() != 3:
            raise ValueError(
                f"patch_tokens must be (B, N, D), got {tuple(patch_tokens.shape)}"
            )
        if text_embedding.dim() != 2:
            raise ValueError(
                f"text_embedding must be (B, D), got {tuple(text_embedding.shape)}"
            )
        b, n, d = patch_tokens.shape
        if text_embedding.shape != (b, d):
            raise ValueError(
                f"text batch/dim {tuple(text_embedding.shape)} does not match "
                f"patch tokens ({b}, {d})"
            )

        text = text_embedding.unsqueeze(1).expand(-1, n, -1)          # (B, N, D)
        prod = patch_tokens * text
        global_pool = patch_tokens.mean(dim=1, keepdim=True).expand(-1, n, -1)

        feat = torch.cat([patch_tokens, text, prod, global_pool], dim=-1)  # (B,N,4D)
        if feat.shape[-1] != self.feature_dim:
            raise ValueError(
                f"assembled feature is {feat.shape[-1]}-d but this head was built "
                f"for feature_dim={self.feature_dim}. The encoder's projected "
                f"dim changed, or grounding_head.feature_dim is wrong."
            )

        h = self.proj(feat)
        h = F.gelu(h)
        h = self.norm(h)
        h = self.drop(h)
        return HeadOutput(raw=self.out(h))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config_dict(self) -> dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_p,
            "outputs": ["tx", "ty", "tw", "th", "objectness"],
            "decode": "cell_relative",
        }

    @classmethod
    def from_config_dict(cls, payload: dict[str, Any]) -> "GroundingHead":
        return cls(
            feature_dim=int(payload["feature_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            dropout=float(payload["dropout"]),
        )

    def __repr__(self) -> str:
        return (
            f"<GroundingHead in={self.feature_dim} hidden={self.hidden_dim} "
            f"params={self.num_parameters()}>"
        )


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def decode_cell_relative(
    box_params: torch.Tensor, grid_h: int, grid_w: int
) -> torch.Tensor:
    """Turn (B, N, 4) cell-relative params into (B, N, 4) normalized xyxy.

    Differentiable: the output feeds the box and GIoU losses directly, so the
    loss is computed on exactly the coordinates the metric measures.

    Args:
        box_params: (B, N, 4) = [tx, ty, tw, th], where N == grid_h * grid_w and
            row-major.
        grid_h, grid_w: grid dimensions.

    Returns:
        (B, N, 4) normalized xyxy, clipped to [0, 1] and order-corrected.
    """
    if box_params.dim() != 3 or box_params.shape[-1] != 4:
        raise ValueError(
            f"box_params must be (B, N, 4), got {tuple(box_params.shape)}"
        )
    n = box_params.shape[1]
    if n != grid_h * grid_w:
        raise ValueError(
            f"expected {grid_h * grid_w} cells for a {grid_h}x{grid_w} grid, "
            f"got {n}"
        )

    tx, ty, tw, th = box_params.unbind(dim=-1)

    idx = torch.arange(n, device=box_params.device, dtype=box_params.dtype)
    rows = torch.div(idx, grid_w, rounding_mode="floor")
    cols = idx - rows * grid_w

    cx = (cols.view(1, -1) + torch.sigmoid(tx)) / grid_w
    cy = (rows.view(1, -1) + torch.sigmoid(ty)) / grid_h
    w = torch.sigmoid(tw)
    h = torch.sigmoid(th)

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    boxes = boxes.clamp(0.0, 1.0)
    return enforce_order(boxes)


def enforce_order(boxes: torch.Tensor) -> torch.Tensor:
    """Guarantee x2 >= x1 and y2 >= y1 by swapping where violated.

    Clipping to [0,1] can push a box past its own edge; without this a decoded
    box can be inverted, which makes IoU zero and silently kills the gradient.
    """
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    lo_x = torch.minimum(x1, x2)
    hi_x = torch.maximum(x1, x2)
    lo_y = torch.minimum(y1, y2)
    hi_y = torch.maximum(y1, y2)
    return torch.stack([lo_x, lo_y, hi_x, hi_y], dim=-1)


def assign_positive_cell(
    targets: torch.Tensor, grid_h: int, grid_w: int
) -> torch.Tensor:
    """Index of the cell containing each target's centre.

    Args:
        targets: (B, 4) normalized xyxy.
        grid_h, grid_w: grid dimensions.

    Returns:
        (B,) long tensor of row-major cell indices, clamped into range so a
        target exactly on x=1.0 does not index off the end.
    """
    if targets.dim() != 2 or targets.shape[-1] != 4:
        raise ValueError(f"targets must be (B, 4), got {tuple(targets.shape)}")

    cx = (targets[:, 0] + targets[:, 2]) / 2.0
    cy = (targets[:, 1] + targets[:, 3]) / 2.0

    col = torch.clamp((cx * grid_w).floor().long(), 0, grid_w - 1)
    row = torch.clamp((cy * grid_h).floor().long(), 0, grid_h - 1)
    return row * grid_w + col


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    """Greedy non-maximum suppression. Returns kept indices, best first.

    Pure torch, no torchvision dependency, so the deployed Space does not need
    an extra package and the behaviour is identical on CPU and GPU.

    Args:
        boxes: (N, 4) normalized xyxy.
        scores: (N,).
        iou_threshold: boxes overlapping a kept box above this are dropped.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in [0,1], got {iou_threshold}")
    if boxes.dim() != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes must be (N, 4), got {tuple(boxes.shape)}")
    if scores.dim() != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError(
            f"scores {tuple(scores.shape)} does not match boxes "
            f"{tuple(boxes.shape)}"
        )
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep: list[torch.Tensor] = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = _box_iou(boxes[i].unsqueeze(0), boxes[rest]).squeeze(0)
        order = rest[ious <= iou_threshold]

    return torch.stack(keep)


def _box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU, (len(a), len(b)). Zero for degenerate boxes, never NaN."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device, dtype=a.dtype)

    x1 = torch.maximum(a[:, None, 0], b[None, :, 0])
    y1 = torch.maximum(a[:, None, 1], b[None, :, 1])
    x2 = torch.minimum(a[:, None, 2], b[None, :, 2])
    y2 = torch.minimum(a[:, None, 3], b[None, :, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - GIoU, mean over the batch. Both (B, 4) normalized xyxy.

    GIoU is used alongside L1 because L1 alone is scale-blind: a box that is
    the right size but in the wrong place and a box that is the wrong size but
    overlapping both attract similar gradients. GIoU is zero when the boxes do
    not overlap at all and rises smoothly as they approach.
    """
    x1 = torch.maximum(pred[:, 0], target[:, 0])
    y1 = torch.maximum(pred[:, 1], target[:, 1])
    x2 = torch.minimum(pred[:, 2], target[:, 2])
    y2 = torch.minimum(pred[:, 3], target[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_p = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    area_t = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)
    union = area_p + area_t - inter
    iou = torch.where(union > 0, inter / union, torch.zeros_like(union))

    cx1 = torch.minimum(pred[:, 0], target[:, 0])
    cy1 = torch.minimum(pred[:, 1], target[:, 1])
    cx2 = torch.maximum(pred[:, 2], target[:, 2])
    cy2 = torch.maximum(pred[:, 3], target[:, 3])
    convex = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0)
    giou = iou - torch.where(convex > 0, (convex - union) / convex,
                             torch.zeros_like(convex))
    return (1.0 - giou).mean()


@dataclass
class LossBreakdown:
    """Separate components so a collapse in one is visible in the logs."""

    total: torch.Tensor
    box: torch.Tensor
    giou: torch.Tensor
    confidence: torch.Tensor
    n_positive: int

    def to_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "box": float(self.box.detach()),
            "giou": float(self.giou.detach()),
            "confidence": float(self.confidence.detach()),
            "n_positive": int(self.n_positive),
        }


def grounding_loss(
    raw: torch.Tensor,
    targets: torch.Tensor,
    grid_h: int,
    grid_w: int,
    *,
    box_weight: float = 0.5,
    giou_weight: float = 0.3,
    confidence_weight: float = 0.2,
    positive_confidence_weight: float = 20.0,
) -> LossBreakdown:
    """Composite grounding loss.

    Box and GIoU are computed on the DECODED coordinates of the single positive
    cell per sample. Confidence is BCE over every cell, with the single positive
    up-weighted: at 1-of-49 an unweighted mean BCE is minimised by predicting
    "no object" everywhere, which is the classic collapse this weight prevents.

    Args:
        raw: (B, N, 5) head output.
        targets: (B, 4) normalized xyxy.
        grid_h, grid_w: grid dimensions.
        box_weight, giou_weight, confidence_weight: loss component weights.
        positive_confidence_weight: BCE weight on the positive cell.
    """
    if raw.dim() != 3 or raw.shape[-1] != 5:
        raise ValueError(f"raw must be (B, N, 5), got {tuple(raw.shape)}")
    b = raw.shape[0]
    if targets.shape != (b, 4):
        raise ValueError(
            f"targets must be ({b}, 4), got {tuple(targets.shape)}"
        )

    decoded = decode_cell_relative(raw[..., :4], grid_h, grid_w)     # (B, N, 4)
    pos = assign_positive_cell(targets, grid_h, grid_w)              # (B,)

    rows = torch.arange(b, device=raw.device)
    pos_boxes = decoded[rows, pos]                                   # (B, 4)

    box = F.l1_loss(pos_boxes, targets)
    giou = giou_loss(pos_boxes, targets)

    obj_logits = raw[..., 4]                                         # (B, N)
    conf_targets = torch.zeros_like(obj_logits)
    conf_targets[rows, pos] = 1.0
    weights = torch.ones_like(obj_logits)
    weights[rows, pos] = positive_confidence_weight
    conf = F.binary_cross_entropy_with_logits(
        obj_logits, conf_targets, weight=weights
    )

    total = box_weight * box + giou_weight * giou + confidence_weight * conf
    return LossBreakdown(
        total=total, box=box, giou=giou, confidence=conf, n_positive=int(b)
    )


def build_head(config) -> GroundingHead:
    """Construct the head from the central config, asserting the frozen input contract."""
    from specialists.grounding.remoteclip import VERIFIED_PROJECTED_DIM

    feature_dim = int(config.get("grounding_head.feature_dim", 2048))
    expected = 4 * VERIFIED_PROJECTED_DIM
    if feature_dim != expected:
        raise ValueError(
            f"grounding_head.feature_dim={feature_dim} but the frozen encoder "
            f"projects to {VERIFIED_PROJECTED_DIM}, so the assembled per-cell "
            f"feature is {expected}-d. Fix configs/base.yaml before training."
        )
    return GroundingHead(
        feature_dim=feature_dim,
        hidden_dim=int(config.get("grounding_head.hidden_dim", 512)),
        dropout=float(config.get("grounding_head.dropout", 0.10)),
    )


__all__ = [
    "GroundingHead",
    "HeadOutput",
    "LossBreakdown",
    "decode_cell_relative",
    "enforce_order",
    "assign_positive_cell",
    "nms",
    "giou_loss",
    "grounding_loss",
    "build_head",
    "EPS",
]