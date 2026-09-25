"""SatQuery AI -- Phase 9 change-detection training.

Trains the STANet-style Siamese detector (`specialists/change/stanet.py`) on
LEVIR-CD bi-temporal pairs, on CPU or GPU, with the same shape as the Phase 8
grounding trainer (`training/grounding/train.py`).

WHAT THIS MODULE OWNS
---------------------
    ChangePairDataset          A/B/label PNGs -> (3,T,T) / (1,T,T) tensors
    collate                    stacks the pair batches
    change_loss                composite BCE + Dice, weights FORWARDED
    collect_change_predictions forward ONCE, keep the probability maps
    evaluate                   validation on the benchmark metric
    sweep_thresholds           score those maps at many thresholds, pick one
    train_change_head          the real loop
    load_trained_change_model  reload a saved detector

WHY THE FORWARD PASS IS SEPARATE FROM THE SCORING
-------------------------------------------------
`evaluate` used to forward and score in a single pass, discarding the
probability maps as it went. Selecting a threshold requires the opposite shape:
forward ONCE over a split, then score the SAME maps at many thresholds. So the
forward loop lives in `collect_change_predictions` and both `evaluate` and
`sweep_thresholds` consume its output through the one `score_dataset` call. The
single-shot number and every swept number therefore come off one code path; two
implementations would be free to drift apart, and a sweep that disagreed with
the shipped evaluation would be worse than no sweep.

WHY THE VALIDATION MASK IS THE RAW ONE
--------------------------------------
`specialists/change/postprocess.py` states the convention in its own docstring:
metrics are reported on the RAW binarized mask, because morphological cleanup is
a presentation aid and applying it before scoring inflates the number relative to
the literature. `evaluate` therefore hands
`evaluation.metrics.change.score_dataset` the model's PROBABILITY map and lets
that module apply its own `>= threshold` rule -- the same rule its own test pins
-- and never calls `postprocess_change_map`. Morphology is for drawing, not for
scoring. A val number produced by a different mask than the one the benchmark
scores is not a val number.

WHY THE LEAKAGE GUARD RUNS FIRST
--------------------------------
LEVIR-CD ships neighbouring 256px crops of the same 1024px scene. A split by
patch puts near-duplicate crops on both sides of the boundary, and validation
then measures memorisation. `assert_image_disjoint` is called BEFORE the model is
built, so a leaky split costs nothing and cannot be trained past. It raises
rather than warns.

THE 0.5/0.5 LOSS DEFAULT IS A CONFIG VALUE, NOT A TUNED ONE
-----------------------------------------------------------
`configs/base.yaml` exposes `change.bce_weight` and `change.dice_weight`. They are
read here and forwarded to the specialist. An earlier version of the training
script accepted those arguments and then passed literal 0.5/0.5, making every
config value a silent no-op. `tests/unit/test_change_train_script_contract.py`
pins that this cannot come back; `change_loss` below is the module-scope symbol
that test reaches for.

TILE SIZE
---------
`change.tile_size` (256 in `configs/base.yaml`) is the tensor the detector sees.
LEVIR-CD ships 1024x1024 scenes, so `_fit_to_tile` takes a deterministic CENTRE
CROP when the source is larger and zero-pads when it is smaller. Both T1 and T2
receive the identical crop/pad, so the padding contributes nothing to the
difference features. One tile per item keeps `len(dataset) == len(items)`, which
keeps the scene-disjointness accounting exact; expanding to a 4x4 tile grid is a
throughput decision that belongs to the caller, not to the leakage guard.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from core.errors import SatQueryError
from evaluation.metrics.change import ChangeReport, ChangeScores, score_dataset
from specialists.change.stanet import (
    ChangeLossBreakdown,
    STANetStyleChangeDetector,
    build_change_model,
    change_loss as stanet_change_loss,
    load_change_model,
    save_change_model,
)
from training.change.dataset import assert_image_disjoint

#: The tensor edge the detector is trained on when the config says nothing.
DEFAULT_TILE_SIZE = 256

#: Probability threshold used for validation. Read from `change.threshold`; this
#: is the fallback, and it matches `evaluation.metrics.change.DEFAULT_THRESHOLD`.
DEFAULT_THRESHOLD = 0.5

#: Label PNGs are 0/255. Anything at or above this counts as change. 128 rather
#: than 255 so a resampled or JPEG-compressed label mask still binarizes sanely.
LABEL_CHANGE_LEVEL = 128


class ChangeTrainingError(SatQueryError):
    code = "change_training_error"
    user_message = "Change-detection training could not proceed."


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _load_rgb(path: str | Path) -> np.ndarray:
    """PNG -> (3, H, W) float32 in [0, 1].

    Raises rather than substituting zeros: a silently zero-filled tile trains
    the model on a scene that does not exist.
    """
    from PIL import Image

    p = Path(path)
    if not p.exists():
        raise ChangeTrainingError(
            f"image does not exist: {p}", context={"path": str(p)}
        )
    try:
        with Image.open(p) as handle:
            arr = np.asarray(handle.convert("RGB"), dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        raise ChangeTrainingError(
            f"could not decode image {p}: {type(exc).__name__}: {exc}",
            context={"path": str(p)},
        ) from exc
    return np.ascontiguousarray(np.transpose(arr / 255.0, (2, 0, 1)))


def _load_label(path: str | Path) -> np.ndarray:
    """Label PNG (0/255) -> (1, H, W) float32 in {0.0, 1.0}."""
    from PIL import Image

    p = Path(path)
    if not p.exists():
        raise ChangeTrainingError(
            f"label does not exist: {p}", context={"path": str(p)}
        )
    try:
        with Image.open(p) as handle:
            arr = np.asarray(handle.convert("L"))
    except Exception as exc:  # noqa: BLE001
        raise ChangeTrainingError(
            f"could not decode label {p}: {type(exc).__name__}: {exc}",
            context={"path": str(p)},
        ) from exc
    binary = (arr >= LABEL_CHANGE_LEVEL).astype(np.float32)
    return binary[None, :, :]


def _fit_to_tile(arrays: list[np.ndarray], tile_size: int) -> list[np.ndarray]:
    """Centre-crop or zero-pad every array to a `tile_size` square.

    One function for the pair AND the label, so the three tensors can never be
    cropped differently -- a mismatched crop is a silent label shift, which
    trains the model to be wrong in a way no metric would reveal.

    Padding is constant zero for all three. Both acquisitions receive the same
    border, so `|f1 - f2|` is unaffected by it, and a zero-padded label asserts
    no change, which is the honest reading of "no data here".
    """
    h, w = arrays[0].shape[-2:]
    if h >= tile_size and w >= tile_size:
        top = (h - tile_size) // 2
        left = (w - tile_size) // 2
        return [a[..., top:top + tile_size, left:left + tile_size] for a in arrays]

    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    spec = ((0, 0), (top, bottom), (left, right))
    return [np.pad(a, spec, mode="constant", constant_values=0.0) for a in arrays]


class ChangePairDataset(torch.utils.data.Dataset):
    """Serves (T1, T2, label) tiles from LEVIR-CD PNG triples.

    Items are the dicts `training.change.dataset.load_levir_dataset` returns, so
    the split, the leakage guard and this dataset all speak about the same
    objects. See the module docstring for why one tile per item.
    """

    def __init__(
        self, items: Sequence[Any], *, tile_size: int = DEFAULT_TILE_SIZE
    ) -> None:
        if tile_size < 8:
            # The encoder's total stride is 8; anything smaller cannot be
            # downsampled four times.
            raise ChangeTrainingError(
                f"tile_size must be >= 8, got {tile_size}",
                context={"tile_size": tile_size},
            )
        self.items = list(items)
        self.tile_size = int(tile_size)

    def __len__(self) -> int:
        return len(self.items)

    def _paths(self, index: int) -> tuple[str, str, str]:
        item = self.items[index]
        try:
            return (
                str(item["t1_path"]),
                str(item["t2_path"]),
                str(item["label_path"]),
            )
        except (TypeError, KeyError) as exc:
            raise ChangeTrainingError(
                f"item {index} is missing a t1_path/t2_path/label_path key: "
                f"{type(exc).__name__}: {exc}",
                context={"index": index},
            ) from exc

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t1_path, t2_path, label_path = self._paths(index)
        t1, t2, label = _fit_to_tile(
            [_load_rgb(t1_path), _load_rgb(t2_path), _load_label(label_path)],
            self.tile_size,
        )
        return (
            torch.from_numpy(t1),
            torch.from_numpy(t2),
            torch.from_numpy(label),
        )


def collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack a batch of (T1, T2, label) triples."""
    return (
        torch.stack([b[0] for b in batch]),
        torch.stack([b[1] for b in batch]),
        torch.stack([b[2] for b in batch]),
    )


def _scene_key(item: Any) -> str:
    """The grouping key `split_by_image` / `assert_image_disjoint` use."""
    if isinstance(item, dict):
        return str(
            item.get("group_key") or item.get("image_key") or item.get("sample_id")
        )
    for attr in ("group_key", "image_key", "sample_id"):
        value = getattr(item, attr, None)
        if value:
            return str(value)
    return ""


def _n_scenes(items: Sequence[Any]) -> int:
    return len({_scene_key(i) for i in items})


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def change_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    pos_weight: float | None = None,
) -> ChangeLossBreakdown:
    """Composite BCE + Dice, delegating to the specialist implementation.

    The weights are FORWARDED, not pinned. They were previously hardcoded to
    literal 0.5/0.5 here, which silently discarded the caller's values and made
    every `change.bce_weight` / `change.dice_weight` entry in `configs/base.yaml`
    a no-op -- exactly the knobs the spec requires tuning on validation.
    """
    return stanet_change_loss(
        logits,
        target,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        pos_weight=pos_weight,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Validation metrics under BOTH aggregation conventions.

    Pooled sums the confusion counts across tiles; macro averages per-tile
    metrics. `evaluation.metrics.change` computes both and warns they can
    disagree by several points, so both are carried rather than one being picked
    and hoped for.
    """

    n: int
    n_images_with_change: int
    threshold: float
    pooled: ChangeScores
    macro: ChangeScores
    mean_change_fraction: float
    seconds: float
    per_image_change_fraction: list[float] = field(default_factory=list)

    @property
    def iou(self) -> float:
        """Pooled IoU of the change class -- the number training selects on."""
        return float(self.pooled.iou)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "n_images_with_change": self.n_images_with_change,
            "threshold": self.threshold,
            "pooled": self.pooled.to_dict(),
            "macro": self.macro.to_dict(),
            "mean_change_fraction": round(self.mean_change_fraction, 4),
            "seconds": round(self.seconds, 3),
        }

    def summary(self) -> str:
        lines = [
            f"tiles            : {self.n} "
            f"({self.n_images_with_change} with change)",
            f"threshold        : {self.threshold:.2f}",
            f"mean change frac : {self.mean_change_fraction:.4f}",
            f"  {'':<8}{'precision':>11}{'recall':>9}{'f1':>9}{'iou':>9}{'miou':>9}",
        ]
        for label, s in (("pooled", self.pooled), ("macro", self.macro)):
            lines.append(
                f"  {label:<8}{s.precision:>11.4f}{s.recall:>9.4f}"
                f"{s.f1:>9.4f}{s.iou:>9.4f}{s.miou:>9.4f}"
            )
        return "\n".join(lines)


@torch.no_grad()
def collect_change_predictions(
    model: STANetStyleChangeDetector,
    items: Sequence[Any],
    *,
    device: str = "cpu",
    batch_size: int = 8,
    tile_size: int = DEFAULT_TILE_SIZE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Forward the detector once, returning one (truth, probability) pair per tile.

    The pairs come back in DATASET order (`shuffle=False`), so a caller can score
    them at any number of thresholds and know every score describes the same
    tiles in the same sequence. `truth` is the raw 2-D label mask (0.0/1.0) and
    `probability` is the 2-D sigmoid map as float32 -- exactly what
    `score_dataset` expects, and the same objects `evaluate` used to score
    inline.

    This is a pure extraction of the loop `evaluate` ran: same `model.eval()`,
    same `ChangePairDataset`, same `DataLoader(shuffle=False, collate_fn=collate)`
    and the same `truth[i, 0]` / `probabilities[i, 0]` squeeze.
    """
    model.eval()
    dataset = ChangePairDataset(items, tile_size=tile_size)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
    )

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for t1, t2, labels in loader:
        out = model(t1.to(device), t2.to(device))
        probabilities = out.probabilities.detach().cpu().numpy()
        truth = labels.numpy()
        for i in range(probabilities.shape[0]):
            # (1,H,W) -> (H,W); score_dataset's _as_bool_mask accepts both but a
            # 2-D mask keeps the shapes unambiguous.
            pairs.append((truth[i, 0], probabilities[i, 0]))
    return pairs


@torch.no_grad()
def evaluate(
    model: STANetStyleChangeDetector,
    items: Sequence[Any],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    device: str = "cpu",
    batch_size: int = 8,
    tile_size: int = DEFAULT_TILE_SIZE,
) -> ValidationResult:
    """Score the detector on a split using the benchmark metric.

    The PROBABILITY map goes to `score_dataset`, which applies its own
    `>= threshold` rule. No morphology is applied: see the module docstring.

    A thin wrapper: `collect_change_predictions` does the forward and
    `score_dataset` does the arithmetic, so `evaluate` and `sweep_thresholds`
    cannot disagree about what a threshold scores. `seconds` still times the
    whole thing -- forward plus scoring -- as it did before the extraction.
    """
    t0 = time.time()
    pairs = collect_change_predictions(
        model,
        items,
        device=device,
        batch_size=batch_size,
        tile_size=tile_size,
    )
    report: ChangeReport = score_dataset(pairs, threshold=threshold)
    elapsed = time.time() - t0

    mean_fraction = (
        float(np.mean(report.per_image_change_fraction))
        if report.per_image_change_fraction
        else 0.0
    )
    return ValidationResult(
        n=report.n_images,
        n_images_with_change=report.n_images_with_change,
        threshold=float(threshold),
        pooled=report.pooled,
        macro=report.macro,
        mean_change_fraction=mean_fraction,
        seconds=elapsed,
        per_image_change_fraction=list(report.per_image_change_fraction),
    )


# ---------------------------------------------------------------------------
# Threshold selection (validation only)
# ---------------------------------------------------------------------------

#: The scalars `sweep_thresholds` will select on. Kept as a tuple so the error
#: message and the guard can never list different sets.
SWEEP_SELECTORS: tuple[str, ...] = ("macro_iou", "pooled_iou", "f1")


@dataclass(frozen=True)
class ThresholdSweep:
    """The whole threshold curve plus the row that won.

    `rows` holds one dict per threshold IN THE ORDER GIVEN, so the caller can
    plot the curve; `selected` is a reference to the winning row. `n_pairs` is
    the number of tiles scored -- fixed across thresholds, since the ground truth
    does not depend on the threshold.
    """

    select_by: str
    thresholds: list[float]
    rows: list[dict]
    selected: dict
    n_pairs: int
    n_images_with_change: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "select_by": self.select_by,
            "thresholds": list(self.thresholds),
            "rows": [dict(row) for row in self.rows],
            "selected": dict(self.selected),
            "n_pairs": self.n_pairs,
            "n_images_with_change": self.n_images_with_change,
        }


def _sweep_metric(report: ChangeReport, select_by: str) -> float:
    """The scalar a sweep selects on, read at FULL precision from the report.

    Deliberately reads the `ChangeScores` field rather than the rounded
    `to_dict()` value: rounding to 4 decimals could manufacture a tie that does
    not exist, or hide one that does, and the tie-break rule depends on knowing
    which.
    """
    if select_by == "macro_iou":
        return float(report.macro.iou)
    if select_by == "pooled_iou":
        return float(report.pooled.iou)
    if select_by == "f1":
        return float(report.pooled.f1)
    raise ChangeTrainingError(
        f"unknown select_by {select_by!r}; expected one of {SWEEP_SELECTORS}",
        context={"select_by": select_by},
    )


def sweep_thresholds(
    pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    thresholds: Sequence[float],
    *,
    select_by: str = "macro_iou",
) -> ThresholdSweep:
    """Score one set of probability maps at many thresholds and select one.

    Every threshold is scored by `score_dataset(pairs, threshold=t)` -- the SAME
    function the single-shot evaluation uses. No metric arithmetic is
    re-implemented here, so the sweep cannot drift away from the shipped number.

    Args:
        pairs: (truth_mask, probability_map) per tile, as returned by
            `collect_change_predictions`. The SAME pairs are reused for every
            threshold, which is the entire point: one forward pass, many scores.
        thresholds: the thresholds to score, in the order to report them.
        select_by: which scalar chooses the winner --
            `"macro_iou"` (default) reads `report.macro.iou`,
            `"pooled_iou"` reads `report.pooled.iou`,
            `"f1"` reads `report.pooled.f1`.

    Tie-break: on an EXACT tie the HIGHER threshold wins. A higher threshold
    asserts change on fewer pixels, so it produces fewer false positives. On
    LEVIR-CD only ~5% of pixels change, and the shipped 0.50 already leans
    toward precision over recall, so breaking a tie downward would spend
    precision to buy recall on a corpus that cannot afford it. The conservative
    direction is up.

    Raises:
        ChangeTrainingError: `thresholds` is empty, or `select_by` is not one of
            `SWEEP_SELECTORS`.
    """
    if not thresholds:
        raise ChangeTrainingError(
            "sweep_thresholds needs at least one threshold",
            context={"select_by": select_by},
        )
    if select_by not in SWEEP_SELECTORS:
        raise ChangeTrainingError(
            f"unknown select_by {select_by!r}; expected one of {SWEEP_SELECTORS}",
            context={"select_by": select_by},
        )

    rows: list[dict] = []
    scores: list[float] = []
    n_images_with_change = 0
    for t in thresholds:
        report = score_dataset(pairs, threshold=float(t))
        n_images_with_change = report.n_images_with_change
        rows.append({
            "threshold": float(t),
            "pooled": report.pooled.to_dict(),
            "macro": report.macro.to_dict(),
            "n_images_with_change": report.n_images_with_change,
        })
        scores.append(_sweep_metric(report, select_by))

    # Argmax with the tie broken toward the larger threshold, regardless of the
    # order the thresholds arrived in.
    best = 0
    for i in range(1, len(rows)):
        if scores[i] > scores[best]:
            best = i
        elif scores[i] == scores[best] and rows[i]["threshold"] > rows[best]["threshold"]:
            best = i

    return ThresholdSweep(
        select_by=select_by,
        thresholds=[float(t) for t in thresholds],
        rows=rows,
        selected=rows[best],
        n_pairs=len(pairs),
        n_images_with_change=n_images_with_change,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class EpochRecord:
    """One epoch's training loss and validation IoU."""

    epoch: int
    loss_total: float
    loss_bce: float
    loss_dice: float
    val_iou: float
    val_f1: float
    lr: float
    seconds: float

    def to_dict(self) -> dict[str, float]:
        return {
            "epoch": self.epoch,
            "loss_total": round(self.loss_total, 6),
            "loss_bce": round(self.loss_bce, 6),
            "loss_dice": round(self.loss_dice, 6),
            "val_iou": round(self.val_iou, 4),
            "val_f1": round(self.val_f1, 4),
            "lr": self.lr,
            "seconds": round(self.seconds, 2),
        }


@dataclass
class TrainingResult:
    """Everything needed to judge the run, including the numbers that say it failed."""

    artifact_dir: Path
    history: list[EpochRecord]
    best_val_iou: float
    final_val: ValidationResult
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def first_val_iou(self) -> float:
        """Validation IoU after the first epoch. The floor the run started from."""
        return float(self.history[0].val_iou) if self.history else 0.0

    @property
    def improved(self) -> bool:
        """Whether training moved validation IoU above where it started.

        Reported alongside the absolute number: a run that ends at 0.02 having
        started at 0.0 has learned something, and a run that ends at 0.02 having
        started at 0.02 has not.
        """
        return self.best_val_iou > self.first_val_iou

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "PHASE 9 -- CHANGE-DETECTION HEAD TRAINING",
            "=" * 70,
            f"epochs run       : {len(self.history)}",
            f"first-epoch val  : {self.first_val_iou:.4f}",
            f"best val IoU     : {self.best_val_iou:.4f}",
            f"improvement      : {self.best_val_iou - self.first_val_iou:+.4f}",
            "",
            "FINAL VALIDATION (best checkpoint)",
            self.final_val.summary(),
            "",
            f"artifact         : {self.artifact_dir}",
            "=" * 70,
        ]
        return "\n".join(lines)


def _save_checkpoint(
    path: Path,
    model: STANetStyleChangeDetector,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Any,
) -> None:
    """Resumable checkpoint. Carries the optimizer state and the config hash."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": model.config_dict(),
            "config_hash": config.hash,
        },
        str(path),
    )


def train_change_head(
    train_items: Sequence[Any],
    val_items: Sequence[Any],
    config: Any,
    *,
    output_dir: str | Path,
    device: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    seed: int | None = None,
    max_steps: int | None = None,
    resume_from: str | Path | None = None,
    verbose: bool = True,
) -> TrainingResult:
    """Train the change detector and return every number needed to judge it.

    Args:
        train_items, val_items: LEVIR item dicts. Checked for scene overlap
            before anything is built.
        config: the central `core.config.Config`. Read for `change.*` keys.
        output_dir: receives `head.pt`, `model_metadata.json` and
            `checkpoint_last.pt`.
        max_steps: cap on OPTIMIZER steps, not epochs. Used by the smoke test to
            bound a CPU run.
        resume_from: a `checkpoint_last.pt`. The scheduler is NOT restored --
            the cosine schedule is re-derived from `epochs` and the resumed
            epoch, which is what a deterministic re-run needs.

    Raises:
        ChangeTrainingError: empty split, or a split that leaks.
    """
    from core.config import load_config

    cfg = config if config is not None else load_config()
    device = device or cfg.device_preference
    tile_size = int(cfg.get("change.tile_size", DEFAULT_TILE_SIZE))
    threshold = float(cfg.get("change.threshold", DEFAULT_THRESHOLD))
    epochs = epochs if epochs is not None else int(cfg.get("change.epochs", 20))
    batch_size = (
        batch_size if batch_size is not None else int(cfg.get("change.batch_size", 8))
    )
    learning_rate = (
        learning_rate
        if learning_rate is not None
        else float(cfg.get("change.learning_rate", 1e-3))
    )
    seed = seed if seed is not None else int(cfg.get("project.seed", 42))
    weight_decay = float(cfg.get("change.weight_decay", 1e-4))
    grad_clip = float(cfg.get("change.grad_clip", 1.0))
    bce_weight = float(cfg.get("change.bce_weight", 0.5))
    dice_weight = float(cfg.get("change.dice_weight", 0.5))
    pos_weight = cfg.get("change.pos_weight", None)
    pos_weight = float(pos_weight) if pos_weight is not None else None

    # -- leakage guard, BEFORE the model is built --------------------------
    assert_image_disjoint(list(train_items), list(val_items))

    if not train_items or not val_items:
        raise ChangeTrainingError(
            f"empty split: {len(train_items)} train / {len(val_items)} val items"
        )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model = build_change_model(cfg, device=device)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )

    train_loader = torch.utils.data.DataLoader(
        ChangePairDataset(train_items, tile_size=tile_size),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=False,
    )

    steps_per_epoch = max(1, math.ceil(len(train_items) / batch_size))

    start_epoch = 0
    if resume_from is not None:
        ckpt = torch.load(str(resume_from), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))
        if verbose:
            print(f"  resumed from {resume_from} at epoch {start_epoch}")

    history: list[EpochRecord] = []
    best_val_iou = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    global_step = start_epoch * steps_per_epoch
    started = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_t0 = time.time()
        sums = {"total": 0.0, "bce": 0.0, "dice": 0.0}
        batches = 0

        for t1, t2, labels in train_loader:
            t1 = t1.to(device)
            t2 = t2.to(device)
            labels = labels.to(device)

            out = model(t1, t2)
            breakdown = change_loss(
                out.logits,
                labels,
                bce_weight=bce_weight,
                dice_weight=dice_weight,
                pos_weight=pos_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            breakdown.total.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            sums["total"] += float(breakdown.total.detach())
            sums["bce"] += float(breakdown.bce.detach())
            sums["dice"] += float(breakdown.dice.detach())
            batches += 1
            global_step += 1

            if max_steps is not None and global_step >= max_steps:
                break

        scheduler.step()

        val = evaluate(
            model,
            val_items,
            threshold=threshold,
            device=device,
            batch_size=batch_size,
            tile_size=tile_size,
        )

        record = EpochRecord(
            epoch=epoch + 1,
            loss_total=sums["total"] / max(1, batches),
            loss_bce=sums["bce"] / max(1, batches),
            loss_dice=sums["dice"] / max(1, batches),
            val_iou=val.iou,
            val_f1=float(val.pooled.f1),
            lr=float(optimizer.param_groups[0]["lr"]),
            seconds=time.time() - epoch_t0,
        )
        history.append(record)

        if val.iou > best_val_iou:
            best_val_iou = val.iou
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(
                f"  epoch {record.epoch:>3}/{epochs}  "
                f"loss {record.loss_total:.4f}  "
                f"bce {record.loss_bce:.4f}  dice {record.loss_dice:.4f}  "
                f"val_iou {record.val_iou:.4f}  val_f1 {record.val_f1:.4f}  "
                f"({record.seconds:.1f}s)"
            )

        # Checkpoint every epoch: the detector is ~63 MB, so the cost is small
        # and a session that dies at epoch 19 does not lose 18 epochs.
        _save_checkpoint(
            artifact_dir / "checkpoint_last.pt", model, optimizer, epoch + 1, cfg
        )

        if max_steps is not None and global_step >= max_steps:
            if verbose:
                print(f"  reached max_steps={max_steps}; stopping")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val = evaluate(
        model,
        val_items,
        threshold=threshold,
        device=device,
        batch_size=batch_size,
        tile_size=tile_size,
    )
    duration = time.time() - started

    metadata: dict[str, Any] = {
        "artifact": "change_head",
        "version": "v001",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": cfg.hash,
        "seed": seed,
        "device": device,
        "tile_size": tile_size,
        "threshold": threshold,
        "model": model.config_dict(),
        "model_parameters": model.num_parameters(),
        "trainable_parameters": model.trainable_parameters(),
        "train_items": len(train_items),
        "val_items": len(val_items),
        "train_scenes": _n_scenes(train_items),
        "val_scenes": _n_scenes(val_items),
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "bce_weight": bce_weight,
            "dice_weight": dice_weight,
            "pos_weight": pos_weight,
            "max_steps": max_steps,
        },
        "history": [r.to_dict() for r in history],
        "best_val_iou": round(best_val_iou, 4),
        "first_val_iou": round(history[0].val_iou, 4) if history else 0.0,
        "improved": bool(history) and best_val_iou > history[0].val_iou,
        "final_val": final_val.to_dict(),
        "duration_seconds": round(duration, 2),
    }
    if device == "cuda":
        metadata["peak_vram_mb"] = round(
            torch.cuda.max_memory_allocated() / (1024 ** 2), 1
        )

    save_change_model(artifact_dir / "head.pt", model, metadata)
    with (artifact_dir / "training_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True, default=str)

    return TrainingResult(
        artifact_dir=artifact_dir,
        history=history,
        best_val_iou=best_val_iou,
        final_val=final_val,
        metadata=metadata,
    )


def load_trained_change_model(
    path: str | Path, device: str = "cpu"
) -> STANetStyleChangeDetector:
    """Load a detector written by `train_change_head`.

    Delegates to `specialists.change.stanet.load_change_model`, which is strict
    about the architecture, and marks the model as trained so
    `ChangeSpecialist.has_checkpoint` reports the truth rather than guessing.
    """
    model = load_change_model(path, device=device)
    model._satquery_trained = True  # type: ignore[attr-defined]
    return model


def checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    """Read the `model_metadata.json` written beside a checkpoint.

    `save_change_model` embeds only the architecture in the `.pt`; the run's
    `config_hash` lives in the sidecar. Returns {} when the sidecar is absent,
    so a caller can distinguish "no metadata" from "metadata says they match"
    instead of treating both as agreement.
    """
    sidecar = Path(path).parent / "model_metadata.json"
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


__all__ = [
    "ChangeTrainingError",
    "ChangePairDataset",
    "ValidationResult",
    "EpochRecord",
    "TrainingResult",
    "ThresholdSweep",
    "SWEEP_SELECTORS",
    "change_loss",
    "collect_change_predictions",
    "evaluate",
    "sweep_thresholds",
    "train_change_head",
    "load_trained_change_model",
    "checkpoint_metadata",
    "collate",
    "DEFAULT_TILE_SIZE",
    "DEFAULT_THRESHOLD",
    "LABEL_CHANGE_LEVEL",
]
