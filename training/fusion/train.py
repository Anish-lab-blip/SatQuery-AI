"""SatQuery AI — Phase 12 optical-SAR fusion-head training (frozen-feature).

WHAT THIS MODULE IS
-------------------
The Phase 12 training LOOP for the optical-SAR fusion head. It is a
FROZEN-FEATURE trainer: CROMA never moves, its three 768-d GAP vectors are
encoded once and cached, and this loop trains only the small head
(`build_fusion_head`, 2318 -> hidden -> num_classes) over those cached vectors.

It reuses the proven shape of `train_grounding_head` (frozen features, `no_grad`
evaluation, warmup+cosine LR, best-state tracking, a `training_metadata.json`)
and borrows the discipline of `train_change_head` (a leak guard before anything
is built, explicit seeding, a checkpoint that carries `config_hash`).

WHAT THIS MODULE IS NOT — READ THIS BEFORE QUOTING A NUMBER
-----------------------------------------------------------
**No number this module produces is a result.** The loop has never seen the real
paired BigEarthNet-S1+S2 corpus; on the fixtures it runs against, it is PLUMBING
EVIDENCE — proof that the loop runs, that the frozen contract holds and that
channel dropout is wired. Nothing more.

In particular the **pre-registered 11.5 decision metric is NOT computed here**
(`docs/PHASE14_CROMA_NORMALISATION_CHANGE.md` section 4). That metric requires a
paired design, a >=3-seed variance-estimation pass, a held-out split on the real
corpus and a decision rule fixed in advance. This module supplies the loop those
need; it does not, and must not be read to, supply the metric. Every run record
carries `result_status` and `pre_registered_metric_computed` fields saying so.

THE FROZEN CONTRACT (freeze section 2.5)
----------------------------------------
    input = concat(optical_GAP, SAR_GAP, joint_GAP, optical_mask(12), sar_mask(2))
          = (B, 2318)
    head  = LayerNorm -> Linear(2318, W) -> GELU -> Dropout -> Linear(W, 19)

The width is not re-declared here. It is taken from
`specialists.optical_sar.fusion_head.expected_fusion_dim` and enforced by
`build_fusion_head`, which refuses anything but 2318. A second copy of "2318" in
this file is exactly the kind of number that drifts.

CHANNEL DROPOUT IS MANDATORY (freeze section 2.5)
-------------------------------------------------
`channel_dropout` is applied PER MODALITY each step: optical at 100/80/60/40%,
SAR at 100/50% (plan section 20). Its `(dropped_features, updated_mask)` return
is fed forward — the updated mask, never the pre-dropout one. In the frozen head
input the channel-level block IS the availability mask (the 2318 vector carries
no per-channel values), so the two returns coincide; the load-bearing rule is
that the returned mask is what reaches the head. Feeding the base mask would
teach the head the mask lies — a defect that trains fine and is quietly wrong.

THE ARM MECHANISM — no config edit (`docs/PHASE12_ENTRY_GATE.md` section 4)
--------------------------------------------------------------------------
`arm="A"|"B"` selects the pre-registered normalisation arm by driving the
EXISTING hash-exempt environment channel `SATQUERY_CROMA_USE_8_BIT`, NOT by
editing `configs/base.yaml`. Adding a `fusion_training:` block would move the
frozen `Config.hash` off `78f1e3700da15aa1` and detach the Phase 9 benchmark
from its config. The arm is recorded in the run record; the config hash stays
put.

THE FEATURE CACHE
-----------------
Features are encoded ONCE and cached to disk. The cache lives under
`DEFAULT_FEATURE_CACHE_DIR` = `artifacts/optical_sar/fusion_features/` — under
`artifacts/` because that is the repository's generated-artefact root (not
source, not config, not `data/` inputs), scoped to this specialist, and NOT
under the frozen `artifacts/change/` tree. See `feature_cache_path`.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from core.errors import ModelLoadError, SatQueryError
from specialists.optical_sar.fusion_head import (
    CROMA_GAP_KEYS,
    assemble_fusion_input,
    build_fusion_head,
    channel_dropout,
    expected_fusion_dim,
)
from specialists.optical_sar.radiometry import ENV_USE_8_BIT as USE_8_BIT_ENV

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Feature-cache layout version. Bump when the encoder, the resolution or the set
#: of stored fields changes, so a stale cache is a MISS rather than a silent
#: shape error in the middle of training.
CACHE_VERSION = "v1"

#: Where the one-off CROMA feature cache lives. See the module docstring.
DEFAULT_FEATURE_CACHE_DIR = REPO_ROOT / "artifacts" / "optical_sar" / "fusion_features"

#: Plan section 20 robustness schedule.
OPTICAL_DROPOUT_RATES: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4)
SAR_DROPOUT_RATES: tuple[float, ...] = (1.0, 0.5)

#: Hyperparameter defaults. There is deliberately NO `fusion_training:` block in
#: `configs/base.yaml`: adding one would move the frozen `Config.hash`. These
#: defaults plus CLI overrides are the whole surface.
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_WARMUP_RATIO = 0.05

#: Stamped into every run record so a fixture run cannot be mistaken for a result.
#:
#: PASS 15 (2026-09-23) — THE TRAILING CLAUSE WAS FALSE AND IS CORRECTED.
#:
#: It used to read "the pre-registered 11.5 metric is not computed
#: (docs/PHASE14_CROMA_NORMALISATION_CHANGE.md §4)". That was TRUE when written
#: -- no production head existed, so the metric could not be computed -- and it
#: became FALSE on 2026-09-23, when the frozen production head was recovered,
#: verified byte-for-byte, and the pre-registered metric WAS computed and
#: independently reproduced:
#:
#:     artifacts/optical_sar/fusion_head_production_v001/pre_registered_115_metric.json
#:     accuracy 0.931000   macro-F1 0.434161   loss 0.254592   n_scored 4000
#:
#: A false claim in this constant is not cosmetic: it is stamped into every run
#: record this module writes, so it would propagate into all FUTURE records. The
#: TEN HISTORICAL records are deliberately NOT rewritten -- see
#: `docs/PHASE12_R08_TRUTHFUL_BACKFILL.md`, which preserves each record byte-for-
#: byte and adds the clarification beside it rather than editing history.
#:
#: What remains true, and is the whole point of the flag: the numbers a TRAINING
#: LOOP run produces are fixture/loop evidence about the plumbing. The
#: pre-registered metric is a separate, frozen evaluation of a separate artifact.
RESULT_STATUS = (
    "PLUMBING_ONLY — fixture/loop evidence, NOT a result; this run computes no "
    "benchmark metric. The pre-registered 11.5 metric is a SEPARATE frozen "
    "evaluation and IS computed — see "
    "artifacts/optical_sar/fusion_head_production_v001/"
    "pre_registered_115_metric.json (docs/PHASE12_R08_TRUTHFUL_BACKFILL.md)"
)


class FusionTrainingError(SatQueryError):
    code = "fusion_training_error"
    user_message = "Fusion-head training could not proceed."


# ---------------------------------------------------------------------------
# The normalisation arm
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One pre-registered normalisation arm. Recorded, never a config edit."""

    name: str
    use_8_bit: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "use_8_bit": self.use_8_bit,
            "description": self.description,
        }


ARM_CONTROL = "A"
ARM_VARIANT = "B"

#: The two arms the production comparison actually runs, keyed by the uint8 axis.
#:
#: WARNING -- this `B` is NOT the arm B registered in
#: `docs/PHASE14_CROMA_NORMALISATION_CHANGE.md` section 4. Both arms below run
#: the SAME per-channel mean+-2std encoder-input stretch; they differ ONLY on
#: whether the result then takes the uint8 round-trip. The axis this dict
#: expresses is therefore the uint8 axis, not the registered
#: "stretch vs no-stretch" axis.
#:
#: The ORIGINALLY preregistered arm B ("percentile/dB conditioning only, no
#: encoder-input stretch") is NOT CONSTRUCTIBLE from this repository: stage 1
#: (the percentile/dB transform) is unimplemented, and `Arm` exposes no field
#: that can express "skip the encoder-input stretch" -- the stretch in
#: `training/fusion/extract.py` is applied unconditionally. There is no arm C.
ARMS: dict[str, Arm] = {
    ARM_CONTROL: Arm(
        ARM_CONTROL,
        True,
        "uint8 axis, use_8_bit=true: per-channel mean+-2std encoder-input "
        "stretch, then the uint8 round-trip (scale to 0..255, clip, quantise, "
        "/255) -> {0/255,...,1}. This is the transform actually executed; the "
        "registered PHASE14 §4 stage-1 percentile/dB conditioning is not run.",
    ),
    ARM_VARIANT: Arm(
        ARM_VARIANT,
        False,
        "uint8 axis, use_8_bit=false -- NOT the registered PHASE14 §4 arm B: "
        "the SAME per-channel mean+-2std encoder-input stretch with NO uint8 "
        "round-trip, clipped to [0,1] directly. The registered arm B "
        "('percentile/dB conditioning only, no encoder-input stretch') is not "
        "constructible from this repository.",
    ),
}


def resolve_arm(name: str) -> Arm:
    """Look up an arm by name. Raises on anything but A/B."""
    key = str(name).strip().upper()
    if key not in ARMS:
        raise FusionTrainingError(
            f"unknown arm {name!r}; accepted values are {sorted(ARMS)}"
        )
    return ARMS[key]


def apply_arm(arm: Arm, *, env: Any | None = None) -> str:
    """Select the arm through the hash-exempt env channel. No config is touched.

    Returns the value written, so a caller can record it.
    """
    target = os.environ if env is None else env
    value = "true" if arm.use_8_bit else "false"
    target[USE_8_BIT_ENV] = value
    return value


# ---------------------------------------------------------------------------
# Cached features
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusionFeature:
    """One sample's cached CROMA outputs plus its label.

    Attributes:
        sample_id: unique sample identifier (the split-firewall key).
        scene_id: the leakage boundary (the Sentinel tile).
        optical_gap / sar_gap / joint_gap: (768,) float32 CROMA GAP vectors.
        optical_mask / sar_mask: (C,) float32 availability.
        label_index: the class index in [0, num_classes).
    """

    sample_id: str
    scene_id: str
    optical_gap: np.ndarray
    sar_gap: np.ndarray
    joint_gap: np.ndarray
    optical_mask: np.ndarray
    sar_mask: np.ndarray
    label_index: int

    def __post_init__(self) -> None:
        gaps = [
            np.asarray(self.optical_gap, dtype=np.float32),
            np.asarray(self.sar_gap, dtype=np.float32),
            np.asarray(self.joint_gap, dtype=np.float32),
        ]
        for name, gap in zip(CROMA_GAP_KEYS, gaps):
            if gap.ndim != 1:
                raise FusionTrainingError(f"{name} must be 1-D, got {gap.shape}")
        if not (gaps[0].shape == gaps[1].shape == gaps[2].shape):
            raise FusionTrainingError(
                f"GAP vectors disagree on length: "
                f"{[int(g.shape[0]) for g in gaps]}"
            )
        o_mask = np.asarray(self.optical_mask, dtype=np.float32)
        s_mask = np.asarray(self.sar_mask, dtype=np.float32)
        if o_mask.ndim != 1 or s_mask.ndim != 1:
            raise FusionTrainingError(
                f"masks must be 1-D, got {o_mask.shape} and {s_mask.shape}"
            )
        total = 3 * int(gaps[0].shape[0]) + int(o_mask.shape[0]) + int(s_mask.shape[0])
        if total != expected_fusion_dim():
            raise FusionTrainingError(
                f"feature width {total} does not match the frozen fusion input "
                f"{expected_fusion_dim()}; check the cached GAP/mask widths"
            )
        if int(self.label_index) < 0:
            raise FusionTrainingError(
                f"label_index must be >= 0, got {self.label_index}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "scene_id": self.scene_id,
            "label_index": int(self.label_index),
            "optical_channels_present": int(np.asarray(self.optical_mask).sum()),
            "sar_channels_present": int(np.asarray(self.sar_mask).sum()),
        }


def feature_cache_path(cache_dir: str | Path | None = None, tag: str = "") -> Path:
    """The cache file path. `tag` distinguishes a slice (e.g. a split name)."""
    base = Path(cache_dir) if cache_dir is not None else DEFAULT_FEATURE_CACHE_DIR
    suffix = f"_{tag}" if tag else ""
    return base / f"croma_features_{CACHE_VERSION}{suffix}.npz"


def write_feature_cache(
    path: str | Path,
    features: Sequence[FusionFeature],
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write features to a single `.npz` (arrays) plus a JSON sidecar (ids/meta).

    ATOMIC COMMIT — READ BEFORE CHANGING THE WRITE ORDER
    ---------------------------------------------------
    Both files are written to unique temporary names in the SAME directory and
    only then moved onto their final names with `os.replace` (an atomic rename
    within one filesystem). No reader ever sees a half-written `.npz`: the
    compressed write is the slow part (seconds to minutes on a real corpus), and
    it now happens entirely on the temp name.

    The `.npz` and its `.json` sidecar cannot be replaced as ONE operation, so
    the commit is two adjacent renames. A crash inside that window leaves a
    complete-but-stale pair — new arrays beside the old sidecar, or the reverse
    — which `verify_feature_cache` reports (`row_counts_agree`,
    `sidecar_n_matches_rows`) rather than a truncated array that loads cleanly
    and is silently wrong. Closing that last window needs a single-file cache
    layout, which is a `CACHE_VERSION` bump and not a write-order change.

    A crashed run may leave one `.<name>.tmp-<pid>-<uuid>.npz` / `.json` behind.
    Nothing reads those names; this function never deletes them (a delete is not
    a safe operation to perform on someone else's filesystem).
    """
    items = list(features)
    if not items:
        raise FusionTrainingError("refusing to write an empty feature cache")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = out.with_suffix(".json")

    payload = {
        "optical_gap": np.stack([np.asarray(f.optical_gap, np.float32) for f in items]),
        "sar_gap": np.stack([np.asarray(f.sar_gap, np.float32) for f in items]),
        "joint_gap": np.stack([np.asarray(f.joint_gap, np.float32) for f in items]),
        "optical_mask": np.stack(
            [np.asarray(f.optical_mask, np.float32) for f in items]
        ),
        "sar_mask": np.stack([np.asarray(f.sar_mask, np.float32) for f in items]),
        "label_index": np.asarray([int(f.label_index) for f in items], dtype=np.int64),
    }
    sidecar = {
        "cache_version": CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n": len(items),
        "sample_ids": [f.sample_id for f in items],
        "scene_ids": [f.scene_id for f in items],
        "metadata": dict(metadata or {}),
    }

    stamp = f"{os.getpid()}-{uuid.uuid4().hex}"
    # The `.npz` suffix is load-bearing: `np.savez_compressed` appends one when
    # the name lacks it, and the rename would then miss the file it just wrote.
    tmp_npz = out.parent / f".{out.stem}.tmp-{stamp}.npz"
    tmp_sidecar = sidecar_path.parent / f".{sidecar_path.name}.tmp-{stamp}"

    np.savez_compressed(str(tmp_npz), **payload)
    tmp_sidecar.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8"
    )

    os.replace(tmp_npz, out)
    os.replace(tmp_sidecar, sidecar_path)
    return out


def read_feature_cache(
    path: str | Path,
) -> tuple[list[FusionFeature], dict[str, Any]]:
    """Read a feature cache. Returns (features, metadata)."""
    src = Path(path)
    if not src.exists():
        raise FusionTrainingError(f"feature cache does not exist: {src}")

    sidecar_path = src.with_suffix(".json")
    if not sidecar_path.exists():
        raise FusionTrainingError(
            f"feature cache sidecar is missing: {sidecar_path} (an interrupted "
            f"write); re-run feature extraction"
        )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    with np.load(str(src)) as data:
        gaps_o = data["optical_gap"]
        gaps_s = data["sar_gap"]
        gaps_j = data["joint_gap"]
        o_masks = data["optical_mask"]
        s_masks = data["sar_mask"]
        labels = data["label_index"]

    sample_ids = sidecar["sample_ids"]
    scene_ids = sidecar["scene_ids"]
    if not (len(sample_ids) == len(scene_ids) == len(labels)):
        raise FusionTrainingError(
            f"feature cache is inconsistent: {len(sample_ids)} ids vs "
            f"{len(labels)} label rows"
        )

    features = [
        FusionFeature(
            sample_id=str(sample_ids[i]),
            scene_id=str(scene_ids[i]),
            optical_gap=gaps_o[i],
            sar_gap=gaps_s[i],
            joint_gap=gaps_j[i],
            optical_mask=o_masks[i],
            sar_mask=s_masks[i],
            label_index=int(labels[i]),
        )
        for i in range(len(labels))
    ]
    return features, dict(sidecar.get("metadata", {}))


# ---------------------------------------------------------------------------
# Batch assembly — channel dropout lives here
# ---------------------------------------------------------------------------


def prepare_fusion_batch(
    gaps: np.ndarray,
    optical_mask: np.ndarray,
    sar_mask: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    optical_rates: tuple[float, ...] = OPTICAL_DROPOUT_RATES,
    sar_rates: tuple[float, ...] = SAR_DROPOUT_RATES,
    apply_dropout: bool = True,
) -> np.ndarray:
    """Assemble the (B, 2318) frozen head input, applying channel dropout.

    `gaps` is (B, 3*encoder_dim) = concat(optical_GAP, SAR_GAP, joint_GAP); the
    masks are (B, 12) and (B, 2). Channel dropout is applied PER MODALITY, and
    the UPDATED mask is what reaches the head — never the pre-dropout one.

    In the frozen head input the channel-level block IS the availability mask
    (the 2318 vector carries no per-channel values), so `channel_dropout`'s two
    returns coincide. The load-bearing rule is therefore that the returned block
    is the one fed forward. Both returns are bound to names and compared: a
    disagreement raises rather than being resolved silently.

    Args:
        apply_dropout: False for evaluation, which must be deterministic.

    Raises:
        FusionTrainingError: dropout was requested without a generator.
    """
    gap_block = np.asarray(gaps, dtype=np.float32)
    o_mask = np.atleast_2d(np.asarray(optical_mask, dtype=np.float32))
    s_mask = np.atleast_2d(np.asarray(sar_mask, dtype=np.float32))

    if apply_dropout:
        if rng is None:
            raise FusionTrainingError(
                "channel dropout needs a seeded generator; pass rng="
            )
        # BOTH returns are consumed. `channel_dropout` returns
        # (dropped_features, updated_mask), and the RETURNED mask is what
        # reaches the head -- never the pre-dropout one.
        #
        # WHY THE TWO COINCIDE HERE: the frozen 2318-vector carries no
        # per-channel feature block, so there is nothing for dropout to zero
        # except the mask itself. The pooling already collapsed each modality to
        # a single 768-vector, so the "channel-level block" passed in IS the
        # availability mask. Dropping a channel and updating the mask are
        # therefore the same edit to the same tensor.
        #
        # The equality is ASSERTED rather than assumed. `_` used to discard the
        # block return, which meant a future change to `channel_dropout` could
        # stop updating the mask and nothing here would notice: the head would
        # be trained on a mask that lies, which trains fine and is quietly wrong.
        o_block, o_mask = channel_dropout(
            o_mask, o_mask, keep_probabilities=optical_rates, rng=rng
        )
        s_block, s_mask = channel_dropout(
            s_mask, s_mask, keep_probabilities=sar_rates, rng=rng
        )
        if not np.array_equal(o_block, o_mask):
            raise FusionTrainingError(
                "channel_dropout's dropped block and updated mask disagree for "
                "the optical modality; in the frozen representation they are the "
                "same tensor, so one of them is not the dropout's output. "
                "Refusing to train the head against a mask it cannot trust."
            )
        if not np.array_equal(s_block, s_mask):
            raise FusionTrainingError(
                "channel_dropout's dropped block and updated mask disagree for "
                "the SAR modality; in the frozen representation they are the "
                "same tensor, so one of them is not the dropout's output. "
                "Refusing to train the head against a mask it cannot trust."
            )

    width = gap_block.shape[1] // len(CROMA_GAP_KEYS)
    assembled = assemble_fusion_input(
        {
            "optical_GAP": gap_block[:, 0:width],
            "SAR_GAP": gap_block[:, width : 2 * width],
            "joint_GAP": gap_block[:, 2 * width : 3 * width],
        },
        optical_mask=o_mask,
        sar_mask=s_mask,
    )
    return assembled.tensor


def _features_to_arrays(
    features: Sequence[FusionFeature],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack features into (gaps, optical_mask, sar_mask, labels) arrays."""
    items = list(features)
    gaps = np.stack(
        [
            np.concatenate(
                [
                    np.asarray(f.optical_gap, np.float32),
                    np.asarray(f.sar_gap, np.float32),
                    np.asarray(f.joint_gap, np.float32),
                ]
            )
            for f in items
        ]
    ).astype(np.float32)
    o_mask = np.stack([np.asarray(f.optical_mask, np.float32) for f in items])
    s_mask = np.stack([np.asarray(f.sar_mask, np.float32) for f in items])
    labels = np.asarray([int(f.label_index) for f in items], dtype=np.int64)
    return gaps, o_mask, s_mask, labels


# ---------------------------------------------------------------------------
# Split firewall
# ---------------------------------------------------------------------------


def assert_split_disjoint(
    train: Sequence[FusionFeature], val: Sequence[FusionFeature]
) -> None:
    """Hard assertion that no scene or sample crosses the partition boundary.

    Raises rather than warns: a leakage guard that only prints is a guard that
    gets ignored.
    """
    scene_overlap = {f.scene_id for f in train} & {f.scene_id for f in val}
    if scene_overlap:
        raise FusionTrainingError(
            f"{len(scene_overlap)} scene(s) appear in both train and val, e.g. "
            f"{sorted(scene_overlap)[:5]}"
        )
    sample_overlap = {f.sample_id for f in train} & {f.sample_id for f in val}
    if sample_overlap:
        raise FusionTrainingError(
            f"{len(sample_overlap)} sample(s) appear in both train and val, e.g. "
            f"{sorted(sample_overlap)[:5]}"
        )


# ---------------------------------------------------------------------------
# Torch dataset
# ---------------------------------------------------------------------------


class FusionFeatureDataset(torch.utils.data.Dataset):
    """Serves (gaps, optical_mask, sar_mask, label) from the frozen feature cache.

    The cache is materialised in memory: a 2318-d float32 row is ~9 KB, so even
    100k samples is ~0.9 GB and per-item disk I/O leaves the inner loop.
    """

    def __init__(self, features: Sequence[FusionFeature]) -> None:
        self.gaps, self.o_mask, self.s_mask, self.labels = _features_to_arrays(
            features
        )

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.gaps[index]),
            torch.from_numpy(self.o_mask[index]),
            torch.from_numpy(self.s_mask[index]),
            torch.tensor(int(self.labels[index])),
        )


def collate(batch):
    gaps = torch.stack([b[0] for b in batch])
    o_mask = torch.stack([b[1] for b in batch])
    s_mask = torch.stack([b[2] for b in batch])
    labels = torch.stack([b[3] for b in batch])
    return gaps, o_mask, s_mask, labels


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _per_class_f1(
    preds: np.ndarray, labels: np.ndarray, num_classes: int
) -> list[float]:
    """Per-class F1 for all `num_classes` slots; absent classes score 0.0.

    Split out from `_macro_f1` so the vector can be recorded and re-derived.
    A macro-F1 scalar over 19 slots is not comparable to one over the slots that
    happen to be present, and the difference is large enough to mislead (0.434 vs
    0.589 on the frozen test split), so exposing the terms is worth the few lines.
    """
    scores: list[float] = []
    for c in range(num_classes):
        tp = int(np.sum((preds == c) & (labels == c)))
        fp = int(np.sum((preds == c) & (labels != c)))
        fn = int(np.sum((preds != c) & (labels == c)))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2.0 * tp / denom)
    return scores


def _macro_f1(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Unweighted mean of per-class F1 over all `num_classes` classes."""
    scores = _per_class_f1(preds, labels, num_classes)
    return float(np.mean(scores)) if scores else 0.0


@dataclass
class ValidationResult:
    """Validation diagnostics. NOT the pre-registered 11.5 metric."""

    n: int
    loss: float
    accuracy: float
    macro_f1: float
    # Class presence in the scored split. `macro_f1` is a mean over ALL
    # `num_classes` slots, so a class with no examples contributes 0.0 and lowers
    # the mean without saying anything about head quality. Recording which classes
    # were actually present keeps 0.434 (over 19) distinguishable from 0.589 (over
    # the 14 present) for anyone reading the number later.
    classes_present: tuple[int, ...] = ()
    classes_absent: tuple[int, ...] = ()
    per_class_f1: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "loss": round(self.loss, 6),
            "accuracy": round(self.accuracy, 6),
            "macro_f1": round(self.macro_f1, 6),
            "note": RESULT_STATUS,
        }


@torch.no_grad()
def evaluate_fusion_head(
    head,
    features: Sequence[FusionFeature],
    *,
    batch_size: int = 256,
    num_classes: int = 19,
) -> ValidationResult:
    """Score the head on a split. Deterministic: no dropout at evaluation."""
    items = list(features)
    if not items:
        raise FusionTrainingError("cannot evaluate on an empty split")

    head.eval()
    gaps, o_mask, s_mask, labels = _features_to_arrays(items)
    total_loss = 0.0
    preds_all: list[np.ndarray] = []

    for start in range(0, len(items), batch_size):
        stop = start + batch_size
        x = prepare_fusion_batch(
            gaps[start:stop], o_mask[start:stop], s_mask[start:stop],
            apply_dropout=False,
        )
        y = torch.from_numpy(labels[start:stop])
        logits = head(torch.from_numpy(x))
        total_loss += float(nn.functional.cross_entropy(logits, y)) * (
            stop - start
        )
        preds_all.append(logits.argmax(dim=1).numpy())

    preds = np.concatenate(preds_all) if preds_all else np.zeros((0,), np.int64)
    accuracy = float(np.mean(preds == labels)) if labels.size else 0.0
    present = tuple(sorted(int(c) for c in np.unique(labels)))
    absent = tuple(c for c in range(num_classes) if c not in set(present))
    return ValidationResult(
        n=int(labels.size),
        loss=total_loss / max(1, len(items)),
        accuracy=accuracy,
        macro_f1=_macro_f1(preds, labels, num_classes),
        classes_present=present,
        classes_absent=absent,
        per_class_f1=tuple(_per_class_f1(preds, labels, num_classes)),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class EpochRecord:
    epoch: int
    loss: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    lr: float
    seconds: float

    def to_dict(self) -> dict[str, float]:
        return {
            "epoch": self.epoch,
            "loss": round(self.loss, 6),
            "val_loss": round(self.val_loss, 6),
            "val_accuracy": round(self.val_accuracy, 6),
            "val_macro_f1": round(self.val_macro_f1, 6),
            "lr": self.lr,
            "seconds": round(self.seconds, 2),
        }


@dataclass
class TrainingResult:
    artifact_dir: Path
    history: list[EpochRecord]
    final_val: ValidationResult
    best_val_accuracy: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "PHASE 12 — OPTICAL-SAR FUSION-HEAD TRAINING",
            "=" * 70,
            f"epochs run        : {len(self.history)}",
            f"best val accuracy : {self.best_val_accuracy:.6f}",
            "",
            "RESULT STATUS: PLUMBING EVIDENCE ONLY — not a result.",
            "The pre-registered 11.5 metric is NOT computed here.",
            "",
            f"artifact          : {self.artifact_dir}",
            "=" * 70,
        ]
        return "\n".join(lines)


def environment_report(device: str) -> dict[str, Any]:
    """What ran, where. Recorded in the run record so a number has a context."""
    facts: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "requested_device": device,
    }
    try:
        facts["torch"] = torch.__version__
        facts["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        facts["torch"] = f"unavailable ({type(exc).__name__})"
    return facts


def _save_checkpoint(
    path: Path,
    head,
    optimizer,
    epoch: int,
    config,
    arm: Arm,
    head_config: dict[str, Any],
) -> None:
    """Resumable checkpoint. Carries the optimizer state and the config hash."""
    torch.save(
        {
            "state_dict": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config_hash": config.hash,
            "arm": arm.name,
            "use_8_bit": arm.use_8_bit,
            "head_config": dict(head_config),
        },
        str(path),
    )


def load_trained_fusion_head(path: str | Path, device: str = "cpu"):
    """Load a trained fusion head from a checkpoint. Strict, so a drift fails."""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    head = build_fusion_head(device=device, **ckpt["head_config"])
    head.load_state_dict(ckpt["state_dict"], strict=True)
    head.eval()
    return head


def train_fusion_head(
    train_features: Sequence[FusionFeature],
    val_features: Sequence[FusionFeature],
    config: Any | None = None,
    *,
    output_dir: str | Path,
    arm: str = ARM_CONTROL,
    device: str = "cpu",
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    seed: int | None = None,
    max_steps: int | None = None,
    hidden_dim: int | None = None,
    dropout: float | None = None,
    optical_rates: tuple[float, ...] = OPTICAL_DROPOUT_RATES,
    sar_rates: tuple[float, ...] = SAR_DROPOUT_RATES,
    radiometry: dict[str, Any] | None = None,
    feature_cache: str | Path | None = None,
    resume_from: str | Path | None = None,
    verbose: bool = True,
) -> TrainingResult:
    """Train the fusion head over cached features. Returns a judgeable result.

    Raises:
        FusionTrainingError: empty split, a split that leaks, an unknown arm, or
            a `fusion.input_dim` that disagrees with the frozen concatenation.
    """
    from core.config import load_config

    cfg = config if config is not None else load_config()

    # -- the frozen contract, enforced by calling through -------------------
    declared_input_dim = cfg.get("fusion.input_dim")
    expected = expected_fusion_dim(
        encoder_dim=int(cfg.get("croma.encoder_dim", 768)),
        optical_channels=int(cfg.get("croma.optical_channels", 12)),
        sar_channels=int(cfg.get("croma.sar_channels", 2)),
        modalities_used=len(CROMA_GAP_KEYS),
    )
    if declared_input_dim != expected:
        raise FusionTrainingError(
            f"fusion.input_dim={declared_input_dim} disagrees with the frozen "
            f"concatenation {expected} (3*encoder_dim + optical_channels + "
            f"sar_channels); refusing to train a head whose first Linear layer "
            f"would mean nothing"
        )
    input_dim = int(declared_input_dim)

    hidden_dim = int(
        hidden_dim if hidden_dim is not None else cfg.get("fusion.hidden_dim", 512)
    )
    dropout = float(
        dropout if dropout is not None else cfg.get("fusion.dropout", 0.2)
    )
    num_classes = int(cfg.get("fusion.num_classes", 19))

    epochs = int(epochs if epochs is not None else DEFAULT_EPOCHS)
    batch_size = int(batch_size if batch_size is not None else DEFAULT_BATCH_SIZE)
    learning_rate = float(
        learning_rate if learning_rate is not None else DEFAULT_LEARNING_RATE
    )
    seed = int(seed if seed is not None else cfg.get("project.seed", 42))
    weight_decay = DEFAULT_WEIGHT_DECAY
    grad_clip = DEFAULT_GRAD_CLIP
    warmup_ratio = DEFAULT_WARMUP_RATIO

    resolved_arm = resolve_arm(arm)

    # -- leak guard, BEFORE anything is built -------------------------------
    assert_split_disjoint(list(train_features), list(val_features))
    if not train_features or not val_features:
        raise FusionTrainingError(
            f"empty split: {len(train_features)} train / "
            f"{len(val_features)} val items"
        )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    shuffle_rng = np.random.default_rng(seed)
    dropout_rng = np.random.default_rng(seed + 1)

    # -- select the arm through the hash-exempt env channel (no config edit) --
    apply_arm(resolved_arm, env=os.environ)

    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    head_config = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "task_dim": num_classes,
        "dropout": dropout,
    }
    try:
        head = build_fusion_head(device=device, **head_config)
    except ModelLoadError as exc:
        raise FusionTrainingError(
            f"the fusion head refused to build: {exc}"
        ) from exc

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    gaps, o_mask, s_mask, labels = _features_to_arrays(list(train_features))
    n_train = int(labels.shape[0])
    steps_per_epoch = max(1, math.ceil(n_train / batch_size))
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

        # A checkpoint carries the provenance of the run that wrote it. Resuming
        # across a different arm, config or head shape would blend two runs into
        # one artifact whose record is true for only half of it -- the same defect
        # `check_resume_provenance` already refuses on the extraction side.
        #
        # The fields are REQUIRED, not optional. `_save_checkpoint` always writes
        # all four, so an absent one means this file did not come from this trainer
        # and its provenance cannot be checked. Refuse, never assume.
        required = ("config_hash", "arm", "use_8_bit", "head_config")
        absent = [key for key in required if key not in ckpt]
        if absent:
            raise FusionTrainingError(
                f"refusing to resume from {resume_from}: it records no "
                f"{', '.join(absent)}, so the run that wrote it cannot be "
                f"identified. `_save_checkpoint` always writes these fields."
            )

        mismatches: list[str] = []
        if ckpt["config_hash"] != cfg.hash:
            mismatches.append(
                f"config_hash: checkpoint {ckpt['config_hash']!r}, this run {cfg.hash!r}"
            )
        if ckpt["arm"] != resolved_arm.name:
            mismatches.append(
                f"arm: checkpoint {ckpt['arm']!r}, this run {resolved_arm.name!r}"
            )
        if bool(ckpt["use_8_bit"]) != resolved_arm.use_8_bit:
            mismatches.append(
                f"use_8_bit: checkpoint {ckpt['use_8_bit']!r}, "
                f"this run {resolved_arm.use_8_bit!r}"
            )
        # `dropout` has NO parameters, so `strict=True` is structurally blind to a
        # dropout change -- it must be compared here or not at all.
        ck_head = ckpt["head_config"] or {}
        for key in ("input_dim", "hidden_dim", "task_dim", "dropout"):
            if key not in ck_head:
                mismatches.append(f"head_config[{key}]: not recorded by the checkpoint")
            elif ck_head[key] != head_config[key]:
                mismatches.append(
                    f"head_config[{key}]: checkpoint {ck_head[key]!r}, "
                    f"this run {head_config[key]!r}"
                )
        if mismatches:
            raise FusionTrainingError(
                f"refusing to resume from {resume_from}: it was written by a "
                f"different run and the result would carry one record over two. "
                + " | ".join(mismatches)
            )

        head.load_state_dict(ckpt["state_dict"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))
        if verbose:
            print(f"  resumed from {resume_from} at epoch {start_epoch}")

    history: list[EpochRecord] = []
    global_step = start_epoch * steps_per_epoch
    best_val_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(start_epoch, epochs):
        head.train()
        epoch_t0 = time.time()
        order = shuffle_rng.permutation(n_train)
        running_loss = 0.0
        batches = 0

        for start in range(0, n_train, batch_size):
            idx = order[start : start + batch_size]
            x = prepare_fusion_batch(
                gaps[idx], o_mask[idx], s_mask[idx],
                rng=dropout_rng, optical_rates=optical_rates, sar_rates=sar_rates,
            )
            y = torch.from_numpy(labels[idx])

            lr = lr_at(global_step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            logits = head(torch.from_numpy(x))
            loss = nn.functional.cross_entropy(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(head.parameters(), max_norm=grad_clip)
            optimizer.step()

            running_loss += float(loss.detach())
            batches += 1
            global_step += 1

            if max_steps is not None and global_step >= max_steps:
                break

        val = evaluate_fusion_head(
            head, list(val_features), batch_size=batch_size, num_classes=num_classes
        )
        record = EpochRecord(
            epoch=epoch + 1,
            loss=running_loss / max(1, batches),
            val_loss=val.loss,
            val_accuracy=val.accuracy,
            val_macro_f1=val.macro_f1,
            lr=lr_at(global_step),
            seconds=time.time() - epoch_t0,
        )
        history.append(record)

        if val.accuracy > best_val_accuracy:
            best_val_accuracy = val.accuracy
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

        if verbose:
            print(
                f"  epoch {record.epoch:>3}/{epochs}  "
                f"loss {record.loss:.4f}  "
                f"val_loss {record.val_loss:.4f}  "
                f"val_acc {record.val_accuracy:.4f}  "
                f"val_f1 {record.val_macro_f1:.4f}  "
                f"({record.seconds:.1f}s)"
            )

        _save_checkpoint(
            artifact_dir / "checkpoint_last.pt",
            head,
            optimizer,
            epoch + 1,
            cfg,
            resolved_arm,
            head_config,
        )

        if max_steps is not None and global_step >= max_steps:
            if verbose:
                print(f"  reached max_steps={max_steps}; stopping")
            break

    if best_state is not None:
        head.load_state_dict(best_state)

    final_val = evaluate_fusion_head(
        head, list(val_features), batch_size=batch_size, num_classes=num_classes
    )

    radiometry = dict(radiometry or {})
    metadata: dict[str, Any] = {
        "artifact": "fusion_head",
        "version": "v001",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": cfg.hash,
        "seed": seed,
        "device": device,
        "arm": resolved_arm.name,
        "use_8_bit": resolved_arm.use_8_bit,
        "normalization": resolved_arm.description,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "task_dim": num_classes,
        "dropout": dropout,
        "head_parameters": int(head.num_parameters()),
        "train_samples": len(train_features),
        "val_samples": len(val_features),
        "train_scenes": len({f.scene_id for f in train_features}),
        "val_scenes": len({f.scene_id for f in val_features}),
        "channel_dropout": {
            "optical_rates": list(optical_rates),
            "sar_rates": list(sar_rates),
        },
        "radiometry": radiometry,
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "warmup_ratio": warmup_ratio,
            "max_steps": max_steps,
        },
        "history": [r.to_dict() for r in history],
        "final_val": final_val.to_dict(),
        "best_val_accuracy": round(best_val_accuracy, 6),
        "result_status": RESULT_STATUS,
        "pre_registered_metric_computed": False,
        "deciding_metric": {
            "key": "best_val_accuracy",
            "definition": "maximum validation accuracy over epochs",
            "secondary_reported": [
                "final_val.macro_f1",
                "history[].val_macro_f1",
            ],
            "note": "Gate F D-02: best validation accuracy decides; macro-F1 "
                    "is co-reported only and is not deciding.",
        },
    }

    _save_checkpoint(
        artifact_dir / "head.pt",
        head,
        optimizer,
        epochs,
        cfg,
        resolved_arm,
        head_config,
    )

    with (artifact_dir / "training_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True, default=str)

    run_record: dict[str, Any] = {
        "run_id": (
            f"fusion_head_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": cfg.hash,
        "seed": seed,
        "device": device,
        "arm": resolved_arm.name,
        "use_8_bit": resolved_arm.use_8_bit,
        "normalization": resolved_arm.description,
        "per_channel_windows": radiometry.get("per_channel_windows"),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "task_dim": num_classes,
        "dropout": dropout,
        "channel_dropout": {
            "optical_rates": list(optical_rates),
            "sar_rates": list(sar_rates),
        },
        "train_samples": len(train_features),
        "val_samples": len(val_features),
        "train_scenes": len({f.scene_id for f in train_features}),
        "val_scenes": len({f.scene_id for f in val_features}),
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_steps": max_steps,
        },
        "history": [r.to_dict() for r in history],
        "final_val": final_val.to_dict(),
        "best_val_accuracy": round(best_val_accuracy, 6),
        "result_status": RESULT_STATUS,
        "pre_registered_metric_computed": False,
        "deciding_metric": {
            "key": "best_val_accuracy",
            "definition": "maximum validation accuracy over epochs",
            "secondary_reported": [
                "final_val.macro_f1",
                "history[].val_macro_f1",
            ],
            "note": "Gate F D-02: best validation accuracy decides; macro-F1 "
                    "is co-reported only and is not deciding.",
        },
        "environment": environment_report(device),
        "feature_cache": str(feature_cache) if feature_cache is not None else None,
    }
    (artifact_dir / "run_record.json").write_text(
        json.dumps(run_record, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    return TrainingResult(
        artifact_dir=artifact_dir,
        history=history,
        final_val=final_val,
        best_val_accuracy=best_val_accuracy,
        metadata=metadata,
    )


__all__ = [
    "CACHE_VERSION",
    "DEFAULT_FEATURE_CACHE_DIR",
    "OPTICAL_DROPOUT_RATES",
    "SAR_DROPOUT_RATES",
    "DEFAULT_EPOCHS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LEARNING_RATE",
    "RESULT_STATUS",
    "USE_8_BIT_ENV",
    "ARM_CONTROL",
    "ARM_VARIANT",
    "ARMS",
    "Arm",
    "resolve_arm",
    "apply_arm",
    "FusionTrainingError",
    "FusionFeature",
    "feature_cache_path",
    "write_feature_cache",
    "read_feature_cache",
    "prepare_fusion_batch",
    "assert_split_disjoint",
    "FusionFeatureDataset",
    "collate",
    "ValidationResult",
    "evaluate_fusion_head",
    "EpochRecord",
    "TrainingResult",
    "train_fusion_head",
    "load_trained_fusion_head",
    "environment_report",
]
