"""SatQuery AI — Phase 8 grounding head training.

Trains ONE small head (1,052,677 params) over FROZEN RemoteCLIP features. The
encoder never moves. This is the plan's "freeze RemoteCLIP initially; train the
projection and the grounding head".

THE BAR, SET BY PHASE 7
-----------------------
The zero-shot baseline over all 16,159 VRSBench eval records at 224 measured:

    mean best IoU  0.0972
    Recall@0.50    0.0234

That is weak, and it is the floor this head must beat. A head that does not
exceed 0.0972 mean best IoU on validation has not justified existing.

VALIDATION USES THE EVALUATION METRIC
-------------------------------------
Validation boxes go through the SAME decode -> NMS -> metric path as the
benchmark (`evaluation.metrics.grounding`). Training on a differentiable box
loss while validating on a different coordinate convention is how a model ends
up with a good loss and a bad score.

Two decode strategies are reported, because they measure different things:

    argmax     the single highest-scoring cell's box. Directly comparable to
               the Phase 7 zero-shot baseline, which is also an argmax.
    threshold  NMS over every cell scoring above a threshold. Usually better
               on multi-cell objects, sometimes worse on a spurious peak.

Leakage: the split is by IMAGE (see dataset.py). `assert_image_disjoint` runs
before any training and raises rather than warning.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from core.errors import SatQueryError
from specialists.grounding.head import (
    GroundingHead,
    build_head,
    decode_cell_relative,
    grounding_loss,
    nms,
)
from training.grounding.dataset import (
    GroundingDataError,
    GroundingItem,
    assert_image_disjoint,
    split_by_image,
)

#: The Phase 7 zero-shot baseline this head must beat. Recorded as a constant
#: so every training log states the comparison rather than assuming the reader
#: remembers it. Source: docs/PHASE7_RESOLUTION_DECISION.md.
ZERO_SHOT_BASELINE_IOU = 0.0972
ZERO_SHOT_BASELINE_RECALL_50 = 0.0234

#: A head that cannot beat the baseline by this margin has not earned its place.
MIN_IMPROVEMENT_IOU = 0.02

RECALL_THRESHOLDS: tuple[float, ...] = (0.10, 0.25, 0.50)


class GroundingTrainingError(SatQueryError):
    code = "grounding_training_error"
    user_message = "Grounding head training could not proceed."


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class FeatureTensorDataset(torch.utils.data.Dataset):
    """Serves (patches, text, box) triples from the frozen feature cache.

    `in_memory=True` materializes everything once. At 49x512 fp16 a patch block
    is ~50 KB, so 36,313 items is ~1.8 GB and a 10k-item subsample is ~500 MB —
    both fine on a Kaggle session with 30 GB RAM, and it removes per-item disk
    I/O from the inner loop.
    """

    def __init__(self, items: Sequence[GroundingItem], in_memory: bool = True) -> None:
        self.items = list(items)
        self._cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        if in_memory:
            for item in self.items:
                patches, text = item.load()
                self._cache.append(
                    (
                        patches.astype(np.float32),
                        text.astype(np.float32),
                        np.asarray(item.box, dtype=np.float32),
                    )
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._cache:
            patches, text, box = self._cache[index]
        else:
            patches, text = self.items[index].load()
            box = np.asarray(self.items[index].box, dtype=np.float32)
        return (
            torch.from_numpy(patches),
            torch.from_numpy(text),
            torch.from_numpy(box),
        )


def collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    patches = torch.stack([b[0] for b in batch])
    text = torch.stack([b[1] for b in batch])
    boxes = torch.stack([b[2] for b in batch])
    return patches, text, boxes


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _decode_predictions(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    *,
    strategy: str,
    top_k: int,
    score_threshold: float,
    nms_iou: float,
) -> list[list[float]]:
    """Turn one image's (N,4) boxes and (N,) scores into a list of predictions."""
    if boxes.numel() == 0:
        return []

    if strategy == "argmax":
        peak = int(scores.argmax())
        return [boxes[peak].tolist()]

    keep = scores >= score_threshold
    if int(keep.sum()) == 0:
        return []
    b = boxes[keep]
    s = scores[keep]
    if strategy == "threshold":
        idx = nms(b, s, nms_iou)[:top_k]
        return [b[int(j)].tolist() for j in idx]

    raise GroundingTrainingError(f"unknown decode strategy {strategy!r}")


@dataclass
class ValidationResult:
    """Validation metrics under one decode strategy."""

    strategy: str
    n: int
    mean_best_iou: float
    mean_matched_iou: float
    recall: dict[str, float]
    latency_ms_per_image: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "n": self.n,
            "mean_best_iou": round(self.mean_best_iou, 4),
            "mean_matched_iou": round(self.mean_matched_iou, 4),
            "recall": {k: round(v, 4) for k, v in self.recall.items()},
            "latency_ms_per_image": round(self.latency_ms_per_image, 3),
        }

    @property
    def beats_baseline(self) -> bool:
        return self.mean_best_iou > ZERO_SHOT_BASELINE_IOU + MIN_IMPROVEMENT_IOU


@torch.no_grad()
def evaluate(
    head: GroundingHead,
    items: Sequence[GroundingItem],
    *,
    grid: int,
    device: str,
    strategy: str = "argmax",
    top_k: int = 5,
    score_threshold: float = 0.30,
    nms_iou: float = 0.50,
    batch_size: int = 128,
    in_memory: bool = True,
) -> ValidationResult:
    """Score the head on a split using the benchmark metric."""
    from evaluation.metrics.grounding import score_dataset

    head.eval()
    dataset = FeatureTensorDataset(items, in_memory=in_memory)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
    )

    per_image: list[tuple[list[list[float]], list[list[float]]]] = []
    total_images = 0
    t0 = time.time()

    for patches, text, boxes_gt in loader:
        patches = patches.to(device)
        text = text.to(device)
        out = head(patches, text)
        decoded = decode_cell_relative(out.raw[..., :4], grid, grid)
        scores = torch.sigmoid(out.raw[..., 4])

        for i in range(patches.shape[0]):
            preds = _decode_predictions(
                decoded[i].cpu(),
                scores[i].cpu(),
                strategy=strategy,
                top_k=top_k,
                score_threshold=score_threshold,
                nms_iou=nms_iou,
            )
            per_image.append((preds, [boxes_gt[i].tolist()]))
        total_images += patches.shape[0]

    report = score_dataset(per_image, thresholds=RECALL_THRESHOLDS)
    elapsed_ms = (time.time() - t0) * 1000.0

    return ValidationResult(
        strategy=strategy,
        n=report.n_images,
        mean_best_iou=report.mean_best_iou,
        mean_matched_iou=report.mean_matched_iou,
        recall={f"{k:.2f}": v for k, v in report.recall.items()},
        latency_ms_per_image=elapsed_ms / max(1, total_images),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class EpochRecord:
    epoch: int
    loss_total: float
    loss_box: float
    loss_giou: float
    loss_confidence: float
    val_iou_argmax: float
    val_recall50_argmax: float
    lr: float
    seconds: float

    def to_dict(self) -> dict[str, float]:
        return {
            "epoch": self.epoch,
            "loss_total": round(self.loss_total, 6),
            "loss_box": round(self.loss_box, 6),
            "loss_giou": round(self.loss_giou, 6),
            "loss_confidence": round(self.loss_confidence, 6),
            "val_iou_argmax": round(self.val_iou_argmax, 4),
            "val_recall50_argmax": round(self.val_recall50_argmax, 4),
            "lr": self.lr,
            "seconds": round(self.seconds, 2),
        }


@dataclass
class TrainingResult:
    artifact_dir: Path
    history: list[EpochRecord]
    val_argmax: ValidationResult
    val_threshold: ValidationResult
    best_val_iou: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def beats_baseline(self) -> bool:
        return self.best_val_iou > ZERO_SHOT_BASELINE_IOU + MIN_IMPROVEMENT_IOU

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "PHASE 8 — GROUNDING HEAD TRAINING",
            "=" * 70,
            f"epochs run        : {len(self.history)}",
            f"best val IoU      : {self.best_val_iou:.4f}",
            f"zero-shot baseline: {ZERO_SHOT_BASELINE_IOU:.4f}",
            f"improvement       : {self.best_val_iou - ZERO_SHOT_BASELINE_IOU:+.4f}",
            f"required margin   : +{MIN_IMPROVEMENT_IOU:.2f}",
            "",
            "FINAL VALIDATION",
            f"  {'strategy':<12}{'IoU':>9}{'Recall@.1':>11}{'Recall@.25':>12}"
            f"{'Recall@.5':>11}",
        ]
        for r in (self.val_argmax, self.val_threshold):
            lines.append(
                f"  {r.strategy:<12}{r.mean_best_iou:>9.4f}"
                f"{r.recall.get('0.10', 0.0):>11.4f}"
                f"{r.recall.get('0.25', 0.0):>12.4f}"
                f"{r.recall.get('0.50', 0.0):>11.4f}"
            )
        lines.append("")
        verdict = "BEATS BASELINE" if self.beats_baseline else "DOES NOT BEAT BASELINE"
        lines.append(f"VERDICT: {verdict}")
        lines.append(f"artifact: {self.artifact_dir}")
        lines.append("=" * 70)
        return "\n".join(lines)


def train_grounding_head(
    train_items: Sequence[GroundingItem],
    val_items: Sequence[GroundingItem],
    config,
    *,
    output_dir: str | Path,
    device: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    seed: int | None = None,
    max_steps: int | None = None,
    in_memory: bool = True,
    resume_from: str | Path | None = None,
    verbose: bool = True,
) -> TrainingResult:
    """Train the head. Returns a result carrying every number needed to judge it."""
    from core.config import load_config

    cfg = config if config is not None else load_config()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    grid = int(cfg.get("grounding.image_size", 224)) // 32
    epochs = epochs if epochs is not None else int(cfg.get("grounding_training.epochs", 20))
    batch_size = batch_size if batch_size is not None else int(cfg.get("grounding_training.batch_size", 16))
    learning_rate = learning_rate if learning_rate is not None else float(cfg.get("grounding_training.learning_rate", 1e-4))
    seed = seed if seed is not None else int(cfg.get("project.seed", 42))
    weight_decay = float(cfg.get("grounding_training.weight_decay", 1e-4))
    grad_clip = float(cfg.get("grounding_training.grad_clip", 1.0))
    warmup_ratio = float(cfg.get("grounding_training.warmup_ratio", 0.05))
    box_w = float(cfg.get("grounding_training.box_loss_weight", 0.5))
    giou_w = float(cfg.get("grounding_training.giou_loss_weight", 0.3))
    conf_w = float(cfg.get("grounding_training.confidence_loss_weight", 0.2))
    pos_conf_w = float(cfg.get("grounding_head.positive_confidence_weight", 20.0))
    nms_iou = float(cfg.get("grounding.nms_iou", 0.50))
    top_k = int(cfg.get("grounding.max_candidates", 20))
    score_threshold = float(cfg.get("grounding.confidence_threshold", 0.40))

    # -- leakage guard, before anything is built --------------------------
    assert_image_disjoint(train_items, val_items)

    if not train_items or not val_items:
        raise GroundingTrainingError(
            f"empty split: {len(train_items)} train / {len(val_items)} val items"
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    head = build_head(cfg).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    train_ds = FeatureTensorDataset(train_items, in_memory=in_memory)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=False,
    )

    steps_per_epoch = max(1, math.ceil(len(train_ds) / batch_size))
    total_steps = steps_per_epoch * epochs
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return learning_rate * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * learning_rate * (1.0 + math.cos(math.pi * progress))

    start_epoch = 0
    if resume_from is not None:
        ckpt = torch.load(str(resume_from), map_location=device, weights_only=False)
        head.load_state_dict(ckpt["state_dict"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))
        if verbose:
            print(f"  resumed from {resume_from} at epoch {start_epoch}")

    history: list[EpochRecord] = []
    global_step = start_epoch * steps_per_epoch
    best_val_iou = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(start_epoch, epochs):
        head.train()
        epoch_t0 = time.time()
        sums = {"total": 0.0, "box": 0.0, "giou": 0.0, "conf": 0.0}
        batches = 0

        for patches, text, boxes_gt in train_loader:
            lr = lr_at(global_step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            patches = patches.to(device)
            text = text.to(device)
            boxes_gt = boxes_gt.to(device)

            out = head(patches, text)
            breakdown = grounding_loss(
                out.raw,
                boxes_gt,
                grid,
                grid,
                box_weight=box_w,
                giou_weight=giou_w,
                confidence_weight=conf_w,
                positive_confidence_weight=pos_conf_w,
            )

            optimizer.zero_grad(set_to_none=True)
            breakdown.total.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(head.parameters(), max_norm=grad_clip)
            optimizer.step()

            sums["total"] += float(breakdown.total.detach())
            sums["box"] += float(breakdown.box.detach())
            sums["giou"] += float(breakdown.giou.detach())
            sums["conf"] += float(breakdown.confidence.detach())
            batches += 1
            global_step += 1

            if max_steps is not None and global_step >= max_steps:
                break

        val = evaluate(
            head,
            val_items,
            grid=grid,
            device=device,
            strategy="argmax",
            top_k=top_k,
            score_threshold=score_threshold,
            nms_iou=nms_iou,
            in_memory=in_memory,
        )

        record = EpochRecord(
            epoch=epoch + 1,
            loss_total=sums["total"] / max(1, batches),
            loss_box=sums["box"] / max(1, batches),
            loss_giou=sums["giou"] / max(1, batches),
            loss_confidence=sums["conf"] / max(1, batches),
            val_iou_argmax=val.mean_best_iou,
            val_recall50_argmax=val.recall.get("0.50", 0.0),
            lr=lr_at(global_step),
            seconds=time.time() - epoch_t0,
        )
        history.append(record)

        if val.mean_best_iou > best_val_iou:
            best_val_iou = val.mean_best_iou
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

        if verbose:
            print(
                f"  epoch {record.epoch:>3}/{epochs}  "
                f"loss {record.loss_total:.4f}  "
                f"box {record.loss_box:.4f}  giou {record.loss_giou:.4f}  "
                f"conf {record.loss_confidence:.4f}  "
                f"val_iou {record.val_iou_argmax:.4f}  "
                f"({record.seconds:.1f}s)"
            )

        # Checkpoint every epoch: the head is 4 MB, so the cost is nil and a
        # session that dies at epoch 19 does not lose 18 epochs.
        _save_checkpoint(
            artifact_dir / "checkpoint_last.pt", head, optimizer, epoch + 1, cfg
        )

        if max_steps is not None and global_step >= max_steps:
            if verbose:
                print(f"  reached max_steps={max_steps}; stopping")
            break

    if best_state is not None:
        head.load_state_dict(best_state)

    val_argmax = evaluate(
        head, val_items, grid=grid, device=device, strategy="argmax",
        top_k=top_k, score_threshold=score_threshold, nms_iou=nms_iou,
        in_memory=in_memory,
    )
    val_threshold = evaluate(
        head, val_items, grid=grid, device=device, strategy="threshold",
        top_k=top_k, score_threshold=score_threshold, nms_iou=nms_iou,
        in_memory=in_memory,
    )

    metadata: dict[str, Any] = {
        "artifact": "grounding_head",
        "version": "v001",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": cfg.hash,
        "seed": seed,
        "device": device,
        "grid": grid,
        "resolution": int(cfg.get("grounding.image_size", 224)),
        "head": head.config_dict(),
        "head_parameters": head.num_parameters(),
        "train_items": len(train_items),
        "val_items": len(val_items),
        "train_images": len({i.image_key for i in train_items}),
        "val_images": len({i.image_key for i in val_items}),
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "warmup_ratio": warmup_ratio,
            "box_loss_weight": box_w,
            "giou_loss_weight": giou_w,
            "confidence_loss_weight": conf_w,
            "positive_confidence_weight": pos_conf_w,
            "max_steps": max_steps,
        },
        "baseline": {
            "source": "docs/PHASE7_RESOLUTION_DECISION.md",
            "mean_best_iou": ZERO_SHOT_BASELINE_IOU,
            "recall_at_0.50": ZERO_SHOT_BASELINE_RECALL_50,
            "required_margin": MIN_IMPROVEMENT_IOU,
        },
        "history": [r.to_dict() for r in history],
        "val_argmax": val_argmax.to_dict(),
        "val_threshold": val_threshold.to_dict(),
        "best_val_iou": round(best_val_iou, 4),
        "improvement_over_baseline": round(best_val_iou - ZERO_SHOT_BASELINE_IOU, 4),
        "beats_baseline": bool(best_val_iou > ZERO_SHOT_BASELINE_IOU + MIN_IMPROVEMENT_IOU),
    }
    if device == "cuda":
        metadata["peak_vram_mb"] = round(
            torch.cuda.max_memory_allocated() / (1024 ** 2), 1
        )
        metadata["gpu_name"] = torch.cuda.get_device_name(0)

    _save_checkpoint(artifact_dir / "head.pt", head, optimizer, epochs, cfg)

    with (artifact_dir / "training_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True, default=str)

    return TrainingResult(
        artifact_dir=artifact_dir,
        history=history,
        val_argmax=val_argmax,
        val_threshold=val_threshold,
        best_val_iou=best_val_iou,
        metadata=metadata,
    )


def _save_checkpoint(path: Path, head, optimizer, epoch: int, config) -> None:
    torch.save(
        {
            "state_dict": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "head_config": head.config_dict(),
            "config_hash": config.hash,
        },
        str(path),
    )


def load_trained_head(path: str | Path, device: str = "cpu") -> GroundingHead:
    """Load a trained head from a checkpoint. Strict, so a shape drift fails."""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    head = GroundingHead.from_config_dict(ckpt["head_config"])
    head.load_state_dict(ckpt["state_dict"], strict=True)
    head.to(device)
    head.eval()
    return head


__all__ = [
    "FeatureTensorDataset",
    "ValidationResult",
    "EpochRecord",
    "TrainingResult",
    "train_grounding_head",
    "load_trained_head",
    "evaluate",
    "collate",
    "ZERO_SHOT_BASELINE_IOU",
    "ZERO_SHOT_BASELINE_RECALL_50",
    "MIN_IMPROVEMENT_IOU",
    "RECALL_THRESHOLDS",
]