"""SatQuery AI — batching for change-VQA training and evaluation (R-02).

WHY A SEPARATE MODULE
---------------------
Three things must hold at once and none of them is a modelling decision:

    1. a record whose scene is missing from the feature cache must be SKIPPED
       AND COUNTED, never silently dropped (a silently dropped sample is a
       training set the caller believes is larger than it is);
    2. shuffling must be reproducible from a seed and must not touch the global
       `random` state, because the router and the fusion trainer already seed
       globally and a second consumer of that state makes a run unreproducible
       for reasons that are invisible in the training loop;
    3. the batch must carry its own provenance — which question ids it holds —
       so a checkpoint's training data can be reconstructed from the log.

This module does exactly those three things and nothing else.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from core.errors import SpecialistError
from training.change_vqa.dataset import ChangeVQARecord, SceneTargets
from training.change_vqa.features import ChangeFeatureCache, TextFeatureCache

#: Sentinel used when a record has no text feature cached. Never reached in a
#: correct run; raising here instead would abort a whole epoch because of one
#: missing row, which is the wrong trade for a training loop.
MISSING_TEXT_POLICY = "skip"


@dataclass
class ChangeVQABatch:
    """One assembled batch, plus the record-level facts needed to audit it."""

    change_features: Any
    text_features: Any
    qtype_index: Any
    temporal_index: Any
    answer_index: Any
    class_mag: Any
    class_delta: Any
    total_changed: Any
    records: list[ChangeVQARecord] = field(default_factory=list)
    n_skipped_missing_change_feature: int = 0
    n_skipped_missing_text_feature: int = 0
    n_skipped_missing_targets: int = 0

    def __len__(self) -> int:
        return len(self.records)

    @property
    def question_ids(self) -> list[int]:
        return [record.question_id for record in self.records]

    @property
    def n_skipped(self) -> int:
        return (
            self.n_skipped_missing_change_feature
            + self.n_skipped_missing_text_feature
            + self.n_skipped_missing_targets
        )

    def skip_summary(self) -> dict[str, int]:
        return {
            "missing_change_feature": self.n_skipped_missing_change_feature,
            "missing_text_feature": self.n_skipped_missing_text_feature,
            "missing_targets": self.n_skipped_missing_targets,
        }


def build_batch(
    records: Sequence[ChangeVQARecord],
    *,
    change_cache: ChangeFeatureCache,
    text_cache: TextFeatureCache,
    targets: dict[str, SceneTargets],
) -> ChangeVQABatch:
    """Assemble one batch, skipping (and counting) anything unusable.

    Raises:
        SpecialistError: nothing in `records` could be assembled. A silently
            empty batch would make a training loop run to completion while
            training on nothing.
    """
    import numpy as np

    change_rows: list[Any] = []
    text_rows: list[Any] = []
    qtype_rows: list[int] = []
    temporal_rows: list[int] = []
    answer_rows: list[int] = []
    mag_rows: list[Any] = []
    delta_rows: list[Any] = []
    total_rows: list[float] = []
    kept: list[ChangeVQARecord] = []
    missing_change = 0
    missing_text = 0
    missing_targets = 0

    text_index = {qid: i for i, qid in enumerate(text_cache.question_ids)}

    for record in records:
        if not change_cache.has(record.scene_key):
            missing_change += 1
            continue
        text_position = text_index.get(record.question_id)
        if text_position is None:
            missing_text += 1
            continue
        scene_targets = targets.get(record.file_name)
        if scene_targets is None:
            missing_targets += 1
            continue

        change_rows.append(change_cache.get(record.scene_key))
        text_rows.append(text_cache.features[text_position])
        qtype_rows.append(record.qtype_index)
        temporal_rows.append(
            _temporal_index(record.temporal_ref)
        )
        answer_rows.append(record.answer_index)
        mag_rows.append(scene_targets.class_mag)
        delta_rows.append(scene_targets.class_delta)
        total_rows.append(scene_targets.total_changed)
        kept.append(record)

    if not kept:
        raise SpecialistError(
            f"none of {len(records)} record(s) could be assembled: "
            f"{missing_change} missing a change feature, {missing_text} missing "
            f"a text feature, {missing_targets} missing scene targets",
            specialist="change_vqa",
        )

    return ChangeVQABatch(
        change_features=np.stack(change_rows).astype(np.float32),
        text_features=np.stack(text_rows).astype(np.float32),
        qtype_index=np.asarray(qtype_rows, dtype=np.int64),
        temporal_index=np.asarray(temporal_rows, dtype=np.int64),
        answer_index=np.asarray(answer_rows, dtype=np.int64),
        class_mag=np.stack(mag_rows).astype(np.float32),
        class_delta=np.stack(delta_rows).astype(np.float32),
        total_changed=np.asarray(total_rows, dtype=np.float32),
        records=kept,
        n_skipped_missing_change_feature=missing_change,
        n_skipped_missing_text_feature=missing_text,
        n_skipped_missing_targets=missing_targets,
    )


def _temporal_index(name: str) -> int:
    from training.change_vqa.vocab import TEMPORAL_REFERENCE_TO_INDEX

    return TEMPORAL_REFERENCE_TO_INDEX.get(name, 0)


def iter_batches(
    records: Sequence[ChangeVQARecord],
    *,
    change_cache: ChangeFeatureCache,
    text_cache: TextFeatureCache,
    targets: dict[str, SceneTargets],
    batch_size: int,
    shuffle: bool = False,
    seed: int = 42,
) -> Iterator[ChangeVQABatch]:
    """Yield batches over `records`, deterministically when `shuffle=True`.

    The shuffle uses a private `random.Random(seed)`, never the module-global
    generator, so two loops over the same data with the same seed produce the
    same order regardless of what else the process has drawn from `random`.
    """
    if batch_size < 1:
        raise SpecialistError(
            f"batch_size must be >= 1, got {batch_size}", specialist="change_vqa"
        )

    order = list(range(len(records)))
    if shuffle:
        random.Random(seed).shuffle(order)

    for start in range(0, len(order), batch_size):
        chunk = [records[i] for i in order[start : start + batch_size]]
        if not chunk:
            continue
        yield build_batch(
            chunk,
            change_cache=change_cache,
            text_cache=text_cache,
            targets=targets,
        )


def summarise_skips(batches: Sequence[ChangeVQABatch]) -> dict[str, int]:
    """Total skip counts across batches, for the training log."""
    out = {
        "missing_change_feature": 0,
        "missing_text_feature": 0,
        "missing_targets": 0,
    }
    for batch in batches:
        for key, value in batch.skip_summary().items():
            out[key] += value
    return out


__all__ = [
    "ChangeVQABatch",
    "build_batch",
    "iter_batches",
    "summarise_skips",
]
