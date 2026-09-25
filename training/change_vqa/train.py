"""SatQuery AI — change-VQA training loop (R-02).

WHAT THIS TRAINS, AND WHAT IT DOES NOT
--------------------------------------
It trains exactly one thing: the small reasoning head in `model.py`. The STANet
detector is FROZEN and is never updated here — its features are read from a cache
built beforehand (`scripts/prepare_change_vqa.py`). That is what makes the
plan's <= 3 h / T4x2 budget reachable: there is no 15.8 M-parameter backbone in
the optimizer, and no image is decoded during training.

THE THREE THINGS THAT MAKE THIS RUN USABLE AS EVIDENCE
------------------------------------------------------
1.  **It refuses to touch the test splits.** `Test` and `Test2` are not readable
    through any option here. A run that selected on, or even looked at, the data
    it will be graded on produces a number that means nothing. There is no flag
    to override this, because a flag is a thing someone eventually passes.

2.  **It stops on the clock, not on the epoch count.** `--time-limit-seconds`
    defaults to the plan's 3 h. The budget is checked before every batch, and on
    exhaustion the run saves `checkpoint_last.pt`, flushes the log and exits
    with a structured `time_limit_reached` reason. The Kaggle notebook wraps this
    in a 12 h session wall, so a run that cannot finish must still leave a
    usable, self-describing artifact behind.

3.  **It records what it does not know as `unavailable`.** Hardware, CUDA and
    driver details, and the wall-clock split between epochs are measured. A
    field that could not be measured is written as `"unavailable"`. It is never
    estimated, and it is never left looking like a zero.

CONFIDENCE IS NOT FITTED HERE
-----------------------------
The head emits raw softmax and a top1-top2 margin. Nothing in this file
calibrates anything, and the run record says so explicitly
(`"confidence_method": "uncalibrated"`), because the R-03 calibration contract is
a separate, unresolved ruling. A fitted temperature written here would be a
fabricated claim about a capability the project has not decided to build.

Usage:
    python -m training.change_vqa.train --help
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from core.code_revision import REPO_ROOT, revision_block
from core.errors import SpecialistError
from training.change_vqa.collate import (
    ChangeVQABatch,
    iter_batches,
    summarise_skips,
)
from training.change_vqa.dataset import (
    DATASET_ID,
    EXPECTED_SPLIT_SIZES,
    PREPROCESSING_VERSION,
    ChangeVQARecord,
    SceneTargets,
    dataset_statistics,
    group_by_native_split,
    load_change_vqa_records,
    read_scene_targets,
    verify_split_integrity,
)
from training.change_vqa.evaluate import evaluate_records, qualitative_examples
from training.change_vqa.features import (
    CHANGE_FEATURE_DIM,
    FEATURE_SPEC,
    TEXT_FEATURE_DIM,
    ChangeFeatureCache,
    TextFeatureCache,
)
from training.change_vqa.model import (
    ARCHITECTURE_VERSION,
    DEFAULT_LOSS_WEIGHTS,
    build_change_vqa_head,
    change_vqa_loss,
    predict_answers,
    save_change_vqa_head,
)

#: Fixed seed. Every stochastic choice in this file derives from it, so a run is
#: reproducible from its own record.
SEED = 42

#: The plan's budget for this stage (section 46): 3 h on T4x2.
DEFAULT_TIME_LIMIT_SECONDS = 3 * 3600

DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_PATIENCE = 6
DEFAULT_MIN_DELTA = 1e-4

#: Splits this trainer may read. Test/Test2 are absent by construction, not by a
#: runtime check that could be relaxed.
TRAINING_SPLIT = "Train"
SELECTION_SPLIT = "Val"
READABLE_SPLITS: tuple[str, ...] = (TRAINING_SPLIT, SELECTION_SPLIT)
FORBIDDEN_SPLITS: tuple[str, ...] = ("Test", "Test2")

#: The honest status of an artifact produced by this script. Training produces a
#: checkpoint; it does not produce a verified capability. R-02 reaches VERIFIED
#: only after the returned artifact has been evaluated, and this string is the
#: reminder that the two are different.
POST_TRAINING_STATE = "TRAINED_UNVERIFIED"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def set_seeds(seed: int = SEED) -> dict[str, Any]:
    """Seed every generator this run can reach, and report what was set.

    `torch.use_deterministic_algorithms(True)` is deliberately NOT enabled: it
    raises on operators that have no deterministic implementation, which on this
    stack turns a reproducibility preference into a hard crash mid-epoch. The
    cuDNN switches are set, and the actual values are reported so the record
    states what was done rather than what was intended.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return {
        "seed": seed,
        "python_random_seeded": True,
        "numpy_seeded": True,
        "torch_seeded": True,
        "torch_cuda_manual_seed_all": cuda_available,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms_enabled": False,
        "deterministic_algorithms_note": (
            "not enabled: it raises on operators lacking a deterministic "
            "implementation, which would abort the run rather than make it "
            "reproducible"
        ),
    }


# ---------------------------------------------------------------------------
# Time budget
# ---------------------------------------------------------------------------


@dataclass
class TimeBudget:
    """A monotonic-clock budget with a hard stop.

    Uses `time.monotonic`, not wall time: a clock adjustment or an NTP step must
    not be able to extend or truncate a training run.
    """

    limit_seconds: float | None
    started: float = field(default_factory=time.monotonic)

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def remaining(self) -> float:
        if self.limit_seconds is None:
            return math.inf
        return self.limit_seconds - self.elapsed()

    def exhausted(self) -> bool:
        return self.limit_seconds is not None and self.elapsed() >= self.limit_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "limit_seconds": self.limit_seconds,
            "elapsed_seconds": round(self.elapsed(), 3),
            "remaining_seconds": (
                None if self.limit_seconds is None
                else round(max(0.0, self.remaining()), 3)
            ),
            "clock": "time.monotonic",
        }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class RunLog:
    """Append-only text log, flushed per line.

    Flushed per line on purpose: a hard stop or a crash must not lose the
    evidence of how far the run got. A buffered log that disappears is worse than
    no log, because it looks like the run never started.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")
        self.lines: list[str] = []

    def __call__(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        self.lines.append(line)
        self._handle.write(line + "\n")
        self._handle.flush()
        print(line, flush=True)

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def environment_report(device: str) -> dict[str, Any]:
    """Measured hardware and library facts. Unmeasurable fields say so."""
    import torch

    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unavailable",
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "requested_device": device,
        "cpu_count": __import__("os").cpu_count(),
    }
    try:
        import torchvision

        report["torchvision"] = torchvision.__version__
    except Exception:  # noqa: BLE001
        report["torchvision"] = "unavailable"

    if torch.cuda.is_available():
        try:
            report["cuda_version"] = torch.version.cuda or "unavailable"
            report["cudnn_version"] = (
                torch.backends.cudnn.version() or "unavailable"
            )
            report["gpu_count"] = torch.cuda.device_count()
            report["gpu_names"] = [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]
            report["gpu_total_memory_bytes"] = [
                torch.cuda.get_device_properties(i).total_memory
                for i in range(torch.cuda.device_count())
            ]
        except Exception as exc:  # noqa: BLE001
            report["cuda_details_error"] = f"{type(exc).__name__}: {exc}"
    else:
        report["cuda_version"] = "unavailable"
        report["cudnn_version"] = "unavailable"
        report["gpu_count"] = 0
        report["gpu_names"] = []
        report["gpu_total_memory_bytes"] = []
    return report


def native_bfloat16_supported(index: int = 0) -> bool:
    """True only where bfloat16 is a HARDWARE feature — Ampere and newer.

    `torch.cuda.is_bf16_supported()` is **not** that predicate. It answers "can
    this device run bf16 at all", which since PyTorch 2.x includes *emulated*
    bf16, and it therefore returns True on Turing (T4, compute capability 7.5).
    The plan's hardware row for R-02 is T4x2, so that difference decides the
    dtype of the real run:

      * emulated bf16 does not use the tensor cores, so on a T4 it is slower
        than fp16 — the opposite of what enabling AMP is for;
      * it still carries bf16's 8 mantissa bits, so it is not more accurate
        either (measured on this repo: 1.1e-3 absolute error on a sigmoid).

    fp16 is the right choice on Turing, and it is the dtype GradScaler exists
    for. Returns False on any non-CUDA host rather than raising.
    """
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        return int(torch.cuda.get_device_capability(index)[0]) >= 8
    except Exception:  # noqa: BLE001 - a probe must not be able to kill a run
        return False


def select_amp(amp_enabled: bool) -> tuple[Any, bool]:
    """Decide the autocast dtype, and whether a GradScaler belongs with it.

    Returns `(amp_dtype, use_grad_scaler)`.

    The pairing is the point. `GradScaler` exists to keep fp16 gradients out of
    the subnormal range; bfloat16 has fp32's exponent range and needs no
    scaling at all. Using one with bf16 is not harmful so much as meaningless —
    it adds a scale/unscale round trip and a `step()` that can silently skip an
    update while the run record still claims mixed precision. So the scaler is
    attached to fp16 and to nothing else.
    """
    import torch

    if not amp_enabled:
        return torch.float32, False
    if native_bfloat16_supported():
        return torch.bfloat16, False
    return torch.float16, True


def resolve_amp(requested: bool, device: str) -> tuple[bool, str]:
    """Decide whether mixed precision can actually run, and say why.

    A capability probe, not a flag echo: `--amp` on a CPU host is a request that
    cannot be honoured, and silently training in fp32 while the record claims AMP
    would misdescribe the run.

    The reason string names the dtype AND the scaler decision, because those two
    together are what the run record has to be able to reproduce later.
    """
    import torch

    if not requested:
        return False, "disabled by --no-amp"
    if device.startswith("cuda") and torch.cuda.is_available():
        dtype, use_scaler = select_amp(True)
        name = "bfloat16" if dtype is torch.bfloat16 else "float16"
        scaler = "with GradScaler" if use_scaler else "no GradScaler (bf16 needs none)"
        return True, f"cuda autocast ({name}) {scaler}"
    return False, f"requested but unavailable on device {device!r}; running fp32"


# ---------------------------------------------------------------------------
# Record selection
# ---------------------------------------------------------------------------


def filter_available(
    records: Sequence[ChangeVQARecord],
    *,
    change_cache: ChangeFeatureCache,
    text_cache: TextFeatureCache,
) -> tuple[list[ChangeVQARecord], dict[str, int]]:
    """Keep only records whose change feature and question feature both exist.

    This runs BEFORE batching, and it is not an optimisation. `iter_batches`
    groups records into fixed-size chunks and then assembles each one, and
    `build_batch` raises when a chunk assembles nothing — which is the right
    guard against a run that trains on an empty set. But with a partially built
    cache (a resumed extraction, or a smoke run over a subset of scenes) a chunk
    can be *entirely* unusable by bad luck, and the run dies with a message about
    an empty batch rather than about the missing features that caused it.

    Filtering here makes the skip accounting explicit and total, so
    `--max-train-records` means "this many usable records" instead of "this many
    records, most of which will be dropped".
    """
    text_ids = set(text_cache.question_ids)
    kept: list[ChangeVQARecord] = []
    counts = {"missing_change_feature": 0, "missing_text_feature": 0}
    for record in records:
        if not change_cache.has(record.scene_key):
            counts["missing_change_feature"] += 1
            continue
        if record.question_id not in text_ids:
            counts["missing_text_feature"] += 1
            continue
        kept.append(record)
    return kept, counts


def subsample(
    records: Sequence[ChangeVQARecord], limit: int | None, *, seed: int
) -> list[ChangeVQARecord]:
    """Deterministic subsample, for a smoke run. Order preserved."""
    if limit is None or limit >= len(records):
        return list(records)
    if limit < 1:
        raise SpecialistError(
            f"subsample limit must be >= 1, got {limit}", specialist="change_vqa"
        )
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    keep = sorted(order[:limit])
    return [records[i] for i in keep]


def balance_by_question_type(
    records: Sequence[ChangeVQARecord], *, seed: int
) -> tuple[list[ChangeVQARecord], dict[str, Any]]:
    """Undersample every question type to the smallest type's count.

    The alternative remedy is a weighted loss. This one is chosen because it is
    auditable without inspecting gradients: the resulting epoch is a real subset
    of the data with a known composition, and the cost is reported as the number
    of records actually discarded. A weighted loss would leave the epoch size and
    composition unchanged while changing the objective, which is harder to see in
    a log and easier to get subtly wrong.
    """
    by_type: dict[str, list[ChangeVQARecord]] = {}
    for record in records:
        by_type.setdefault(record.qtype, []).append(record)
    if not by_type:
        return [], {"enabled": True, "n_records": 0, "per_type": {}}

    target = min(len(rows) for rows in by_type.values())
    generator = random.Random(seed)
    kept: list[ChangeVQARecord] = []
    per_type: dict[str, dict[str, int]] = {}
    for qtype in sorted(by_type):
        rows = by_type[qtype]
        if len(rows) > target:
            chosen = sorted(generator.sample(range(len(rows)), target))
            rows = [rows[i] for i in chosen]
        kept.extend(rows)
        per_type[qtype] = {"available": len(by_type[qtype]), "kept": len(rows)}
    return kept, {
        "enabled": True,
        "target_per_type": target,
        "n_records": len(kept),
        "n_discarded": len(records) - len(kept),
        "per_type": per_type,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class EpochResult:
    epoch: int
    n_batches: int
    n_records: int
    train_loss: float
    loss_components: dict[str, float]
    val_accuracy: float
    val_macro_f1: float
    val_mean_confidence: float
    seconds: float
    skips: dict[str, int]
    time_limit_reached: bool = False


def _to_device(batch: ChangeVQABatch, device: str) -> dict[str, Any]:
    import torch

    return {
        "change_features": torch.as_tensor(
            batch.change_features, dtype=torch.float32, device=device
        ),
        "text_features": torch.as_tensor(
            batch.text_features, dtype=torch.float32, device=device
        ),
        "qtype_index": torch.as_tensor(
            batch.qtype_index, dtype=torch.long, device=device
        ),
        "temporal_index": torch.as_tensor(
            batch.temporal_index, dtype=torch.long, device=device
        ),
        "answer_index": torch.as_tensor(
            batch.answer_index, dtype=torch.long, device=device
        ),
        "class_mag": torch.as_tensor(
            batch.class_mag, dtype=torch.float32, device=device
        ),
        "class_delta": torch.as_tensor(
            batch.class_delta, dtype=torch.float32, device=device
        ),
        "total_changed": torch.as_tensor(
            batch.total_changed, dtype=torch.float32, device=device
        ),
    }


def assert_finite_step(
    loss: Any, *, epoch: int, batch_index: int, n_batches: int
) -> None:
    """Refuse to take an optimizer step on a non-finite loss.

    Nothing else in this module guards against a poisoned step, and the failure
    it prevents is the expensive kind: a non-finite loss does not raise. It
    propagates through `optimizer.step()` into every weight, and the run then
    saves a checkpoint of NaNs that still loads, still predicts, and still
    reports an accuracy near chance — which is precisely the artifact this
    project must never hand to a reviewer.

    Failing here costs one batch and names the epoch, the batch and the loss
    components. Under mixed precision the likely cause is a saturated sigmoid:
    BCE's gradient is `(p - t) / (p * (1 - p))`, so a probability pinned at
    exactly 0 or 1 makes the term infinite. The message says so, because the
    first thing worth trying is `--no-amp`.
    """
    import torch

    if bool(torch.isfinite(loss.total)):
        return
    components = {
        k: v for k, v in loss.to_dict().items() if not k.startswith("weight_")
    }
    raise SpecialistError(
        f"epoch {epoch} batch {batch_index + 1}/{n_batches}: the loss is "
        f"{float(loss.total.detach())!r}, which is not finite. Components: "
        f"{components}. No optimizer step was taken and no checkpoint was "
        f"written from this step. A saturated sigmoid makes binary "
        f"cross-entropy infinite, so re-run with --no-amp to rule mixed "
        f"precision in or out before touching the learning rate.",
        specialist="change_vqa",
    )


def assert_finite_parameters(model: Any, *, epoch: int) -> None:
    """One sweep per epoch, after the batches.

    A finite loss with non-finite weights is rare, but it is the failure that
    silently survives all the way into the artifact, so it is worth the ~2 ms
    per epoch it costs to scan 1.45 M parameters.
    """
    import torch

    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise SpecialistError(
                f"epoch {epoch}: parameter {name!r} holds non-finite values "
                f"after the optimizer step. Refusing to validate or to "
                f"checkpoint a poisoned model.",
                specialist="change_vqa",
            )


def _evaluate(
    model: Any,
    *,
    records: Sequence[ChangeVQARecord],
    change_cache: ChangeFeatureCache,
    text_cache: TextFeatureCache,
    targets: dict[str, SceneTargets],
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    """Score `records`. Never touches the test splits — see `main`."""
    import numpy as np

    if not records:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "mean_confidence": 0.0,
            "n_scored": 0,
            "n_skipped": 0,
            "predictions": None,
            "scored_records": [],
        }

    index: list[ChangeVQARecord] = []
    change_rows: list[Any] = []
    text_rows: list[Any] = []
    qtype_rows: list[int] = []
    temporal_rows: list[int] = []
    skipped = 0
    text_position = {q: i for i, q in enumerate(text_cache.question_ids)}

    for record in records:
        if not change_cache.has(record.scene_key):
            skipped += 1
            continue
        position = text_position.get(record.question_id)
        if position is None:
            skipped += 1
            continue
        index.append(record)
        change_rows.append(change_cache.get(record.scene_key))
        text_rows.append(text_cache.features[position])
        qtype_rows.append(record.qtype_index)
        temporal_rows.append(_temporal_index(record.temporal_ref))

    if not index:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "mean_confidence": 0.0,
            "n_scored": 0,
            "n_skipped": skipped,
            "predictions": None,
            "scored_records": [],
        }

    from training.change_vqa.evaluate import answer_accuracy, macro_f1

    predictions = predict_answers(
        model,
        change_features=np.stack(change_rows).astype(np.float32),
        text_features=np.stack(text_rows).astype(np.float32),
        qtype_indices=np.asarray(qtype_rows, dtype=np.int64),
        temporal_indices=np.asarray(temporal_rows, dtype=np.int64),
        qtypes=[r.qtype for r in index],
        apply_type_mask=False,
        device=device,
    )
    gold = [r.answer_index for r in index]
    predicted = [int(v) for v in predictions["answer_index"]]
    return {
        "accuracy": answer_accuracy(predicted, gold),
        "macro_f1": macro_f1(predicted, gold),
        "mean_confidence": float(np.mean(predictions["confidence"])),
        "n_scored": len(index),
        "n_skipped": skipped,
        "predictions": predictions,
        "scored_records": index,
    }


def _temporal_index(name: str) -> int:
    from training.change_vqa.vocab import TEMPORAL_REFERENCE_TO_INDEX

    return TEMPORAL_REFERENCE_TO_INDEX.get(name, 0)


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Run one training job. Returns the run record; never fabricates a field."""
    import numpy as np
    import torch

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(output_dir / "train_log.txt")
    budget = TimeBudget(args.time_limit_seconds)

    log("=" * 72)
    log(f"SatQuery AI change-VQA training (R-02) — {ARCHITECTURE_VERSION}")
    log("=" * 72)

    # -- 0. refuse the test splits, before any work ----------------------
    for split in FORBIDDEN_SPLITS:
        if split in READABLE_SPLITS:
            raise SpecialistError(
                f"internal error: {split} is in READABLE_SPLITS",
                specialist="change_vqa",
            )
    log(f"readable splits      : {list(READABLE_SPLITS)}")
    log(f"forbidden splits     : {list(FORBIDDEN_SPLITS)} (not readable here)")

    device = args.device
    if args.require_cuda and not device.startswith("cuda"):
        raise SpecialistError(
            f"--require-cuda was given but the resolved device is {device!r}; "
            f"refusing to fall back silently",
            specialist="change_vqa",
        )
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SpecialistError(
            f"device {device!r} was requested but CUDA is unavailable",
            specialist="change_vqa",
        )

    seeds = set_seeds(args.seed)
    amp_enabled, amp_reason = resolve_amp(args.amp, device)
    environment = environment_report(device)

    log(f"device               : {device}")
    log(f"mixed precision      : {amp_enabled} ({amp_reason})")
    log(f"seed                 : {args.seed}")
    log(f"time limit           : {args.time_limit_seconds}s")
    log(f"torch                : {environment['torch']}")

    # -- 1. data ---------------------------------------------------------
    log("")
    log("--- data ---")
    records = load_change_vqa_records(
        args.data_root, splits=READABLE_SPLITS, require_images=False
    )
    if not records:
        raise SpecialistError(
            f"no change-VQA records resolved from {args.data_root!r}",
            specialist="change_vqa",
        )
    integrity = verify_split_integrity(
        records,
        expect_full_split=False,
        expected_splits=READABLE_SPLITS,
    )
    if not integrity["is_clean"]:
        raise SpecialistError(
            "CDVQA split integrity check failed: "
            f"{integrity['size_errors']} {integrity['leakage']} "
            f"({integrity['records_with_inconsistent_imagery']} record(s) with "
            f"mismatched imagery). Training on mis-paired (scene, question, "
            f"answer) units would produce a number that means nothing.",
            specialist="change_vqa",
        )
    by_split = group_by_native_split(records)
    train_records = by_split[TRAINING_SPLIT]
    val_records = by_split[SELECTION_SPLIT]
    if not train_records or not val_records:
        raise SpecialistError(
            f"need both {TRAINING_SPLIT} and {SELECTION_SPLIT} records; got "
            f"{len(train_records)} and {len(val_records)}",
            specialist="change_vqa",
        )
    log(f"integrity            : clean "
        f"(scenes/split {integrity['scenes_per_split']})")

    # -- 2. caches and targets -------------------------------------------
    # Loaded BEFORE record selection, because selection has to know which
    # records are actually trainable. See `filter_available`.
    log("")
    log("--- cached inputs ---")
    change_cache = ChangeFeatureCache.read(
        args.change_cache, expect_spec_hash=args.expect_change_spec
    )
    text_cache_train = TextFeatureCache.read(
        args.text_cache_train, expect_spec_hash=args.expect_text_spec
    )
    text_cache_val = TextFeatureCache.read(
        args.text_cache_val, expect_spec_hash=args.expect_text_spec
    )
    targets = read_scene_targets(args.targets)

    log(f"change cache         : {len(change_cache)} scenes, "
        f"spec {change_cache.spec_hash}")
    log(f"  trained detector   : {change_cache.extractor_config.get('trained')}")
    log(f"text cache (train)   : {len(text_cache_train)} questions, "
        f"spec {text_cache_train.spec_hash}")
    log(f"text cache (val)     : {len(text_cache_val)} questions")
    log(f"scene targets        : {len(targets)} scenes")

    if not change_cache.extractor_config.get("trained", False):
        log("WARNING: the change feature cache was built with an UNTRAINED "
            "detector. The head can still be trained, but its inputs are not "
            "the trained STANet representation the plan specifies, and any "
            "result must be reported as such.")

    if args.resume:
        raise SpecialistError(
            "resume is not implemented: a partially-trained head resumed from a "
            "checkpoint whose optimizer state was not saved would silently "
            "restart the schedule. Not offered rather than offered wrongly.",
            specialist="change_vqa",
        )

    # -- 3. keep only what is actually trainable -------------------------
    log("")
    log("--- record selection ---")
    train_records, train_drops = filter_available(
        train_records, change_cache=change_cache, text_cache=text_cache_train
    )
    val_records, val_drops = filter_available(
        val_records, change_cache=change_cache, text_cache=text_cache_val
    )
    log(f"train usable         : {len(train_records)} "
        f"(dropped {train_drops['missing_change_feature']} without a change "
        f"feature, {train_drops['missing_text_feature']} without a text feature)")
    log(f"val usable           : {len(val_records)} "
        f"(dropped {val_drops['missing_change_feature']} without a change "
        f"feature, {val_drops['missing_text_feature']} without a text feature)")
    if not train_records:
        raise SpecialistError(
            "no train record has both a cached change feature and a cached "
            "question feature. The caches and the records do not describe the "
            "same scenes — check that prepare_change_vqa.py ran over the same "
            "splits the trainer is reading.",
            specialist="change_vqa",
        )
    if not val_records:
        raise SpecialistError(
            "no VAL record has both cached features, so there is nothing to "
            "select a checkpoint on. Prepare the Val split, or the run would "
            "select on nothing and report an accuracy of zero.",
            specialist="change_vqa",
        )

    balancing: dict[str, Any] = {"enabled": False}
    if args.balance_by_type:
        train_records, balancing = balance_by_question_type(
            train_records, seed=args.seed
        )
        log(f"balanced by type     : {balancing['n_records']} kept, "
            f"{balancing['n_discarded']} discarded "
            f"(target {balancing['target_per_type']}/type)")

    train_records = subsample(train_records, args.max_train_records, seed=args.seed)
    val_records = subsample(val_records, args.max_val_records, seed=args.seed + 1)

    stats = dataset_statistics(train_records + val_records)
    log(f"train records        : {len(train_records)}")
    log(f"val records          : {len(val_records)}")
    log(f"distinct answers     : {stats['n_distinct_answers']}")
    log(f"outside vocabulary   : {stats['answers_outside_frozen_vocabulary']}")
    if stats["answers_outside_frozen_vocabulary"]:
        raise SpecialistError(
            f"answers outside the frozen vocabulary: "
            f"{stats['answers_outside_frozen_vocabulary']}",
            specialist="change_vqa",
        )

    # -- 4. model --------------------------------------------------------
    log("")
    log("--- model ---")
    model = build_change_vqa_head(
        change_feature_dim=CHANGE_FEATURE_DIM,
        text_feature_dim=TEXT_FEATURE_DIM,
        trunk_dim=args.trunk_dim,
        text_dim=args.text_dim,
        dropout=args.dropout,
    )
    model.to(device)
    n_params = model.num_parameters()
    log(f"head parameters      : {n_params:,} (trainable {n_params:,})")
    log(f"change feature dim   : {CHANGE_FEATURE_DIM}")
    log(f"text feature dim     : {TEXT_FEATURE_DIM}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    batches_per_epoch = max(1, math.ceil(len(train_records) / args.batch_size))
    total_steps = batches_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    # AMP dtype and scaler are chosen together — see `select_amp`. bf16 gets no
    # GradScaler, so the step path below keys off the scaler, not off `amp_enabled`.
    amp_dtype, use_grad_scaler = select_amp(amp_enabled)
    scaler = (
        torch.amp.GradScaler("cuda")
        if (amp_enabled and use_grad_scaler)
        else None
    )
    amp_context = (
        torch.amp.autocast("cuda", dtype=amp_dtype)
        if amp_enabled
        else contextlib.nullcontext()
    )

    log(f"optimizer            : AdamW(lr={args.lr}, wd={args.weight_decay})")
    log(f"scheduler            : cosine, {warmup_steps} warmup of {total_steps} "
        f"steps")
    log(f"batch size           : {args.batch_size} "
        f"({batches_per_epoch} batches/epoch)")
    log(f"loss weights         : {DEFAULT_LOSS_WEIGHTS}")

    # -- 5. epochs -------------------------------------------------------
    log("")
    log("--- training ---")
    history: list[EpochResult] = []
    best_accuracy = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    global_step = 0
    stop_reason = "epochs_exhausted"
    checkpoint_path = output_dir / "head.pt"
    last_checkpoint_path = output_dir / "checkpoint_last.pt"

    for epoch in range(1, args.epochs + 1):
        if budget.exhausted():
            stop_reason = "time_limit_reached"
            log(f"epoch {epoch}: time limit reached before the epoch started")
            break

        epoch_started = time.monotonic()
        model.train()
        batches = list(
            iter_batches(
                train_records,
                change_cache=change_cache,
                text_cache=text_cache_train,
                targets=targets,
                batch_size=args.batch_size,
                shuffle=True,
                seed=args.seed + epoch,
            )
        )
        running = 0.0
        components: dict[str, float] = {}
        n_records = 0
        time_limit_hit = False

        for batch_index, batch in enumerate(batches):
            if budget.exhausted():
                time_limit_hit = True
                stop_reason = "time_limit_reached"
                log(f"epoch {epoch}: time limit reached at batch "
                    f"{batch_index + 1}/{len(batches)}")
                break

            tensors = _to_device(batch, device)
            lr = lr_at(global_step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with amp_context:
                output = model(
                    tensors["change_features"],
                    tensors["text_features"],
                    tensors["qtype_index"],
                    tensors["temporal_index"],
                )
                loss = change_vqa_loss(
                    output,
                    answer_index=tensors["answer_index"],
                    class_mag=tensors["class_mag"],
                    class_delta=tensors["class_delta"],
                    total_changed=tensors["total_changed"],
                )
            assert_finite_step(
                loss, epoch=epoch, batch_index=batch_index, n_batches=len(batches)
            )
            if scaler is not None:
                # fp16 only. bf16 takes the plain path below: it needs no
                # scaling, and a scaler there would add a skipped-step failure
                # mode for no benefit.
                scaler.scale(loss.total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            running += float(loss.total.detach()) * len(batch)
            for key, value in loss.to_dict().items():
                components[key] = components.get(key, 0.0) + value * len(batch)
            n_records += len(batch)
            global_step += 1

        if n_records == 0:
            raise SpecialistError(
                f"epoch {epoch} assembled no usable batches; the caches and the "
                f"records do not overlap",
                specialist="change_vqa",
            )
        assert_finite_parameters(model, epoch=epoch)

        train_loss = running / n_records
        loss_components = {k: v / n_records for k, v in components.items()}
        skips = summarise_skips(batches)

        val = _evaluate(
            model,
            records=val_records,
            change_cache=change_cache,
            text_cache=text_cache_val,
            targets=targets,
            batch_size=args.batch_size,
            device=device,
        )
        epoch_seconds = time.monotonic() - epoch_started

        result = EpochResult(
            epoch=epoch,
            n_batches=len(batches),
            n_records=n_records,
            train_loss=train_loss,
            loss_components=loss_components,
            val_accuracy=val["accuracy"],
            val_macro_f1=val["macro_f1"],
            val_mean_confidence=val["mean_confidence"],
            seconds=epoch_seconds,
            skips=skips,
            time_limit_reached=time_limit_hit,
        )
        history.append(result)

        improved = val["accuracy"] > best_accuracy + args.min_delta
        marker = ""
        if improved:
            best_accuracy = val["accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_change_vqa_head(
                checkpoint_path,
                model,
                metadata={
                    "architecture": ARCHITECTURE_VERSION,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "state": POST_TRAINING_STATE,
                    "selected_on": f"{SELECTION_SPLIT} answer accuracy",
                    "val_answer_accuracy": round(best_accuracy, 6),
                    "epoch": epoch,
                    "seed": args.seed,
                    "dataset_id": DATASET_ID,
                    "preprocessing_version": PREPROCESSING_VERSION,
                    "feature_spec": FEATURE_SPEC,
                    "change_cache_spec": change_cache.spec_hash,
                    "text_cache_spec": text_cache_train.spec_hash,
                    "detector_trained": bool(
                        change_cache.extractor_config.get("trained", False)
                    ),
                    "config_hash": args.config_hash or "unavailable",
                    "confidence_method": "uncalibrated",
                    "test_splits_used": False,
                },
            )
            marker = "  <- best (saved)"
        else:
            epochs_without_improvement += 1

        log(
            f"epoch {epoch:>3}/{args.epochs}  "
            f"loss {train_loss:.4f}  "
            f"val_acc {val['accuracy']:.4f}  "
            f"val_f1 {val['macro_f1']:.4f}  "
            f"conf {val['mean_confidence']:.4f}  "
            f"lr {lr:.2e}  "
            f"{epoch_seconds:.1f}s  "
            f"scored {val['n_scored']}"
            f"{marker}"
        )

        # Always leave a recoverable artifact behind.
        save_change_vqa_head(last_checkpoint_path, model)

        if time_limit_hit:
            log("stopping: time limit reached")
            break
        if (
            not improved
            and epochs_without_improvement >= args.patience
        ):
            stop_reason = "early_stopping"
            log(f"stopping: no improvement for {args.patience} epoch(s)")
            break

    if not history:
        raise SpecialistError(
            "no epoch completed; the time limit is shorter than one epoch. "
            "Increase --time-limit-seconds or reduce --max-train-records.",
            specialist="change_vqa",
        )

    # -- 6. final report -------------------------------------------------
    log("")
    log("--- done ---")
    log(f"stop reason          : {stop_reason}")
    log(f"best epoch           : {best_epoch} (val acc {best_accuracy:.4f})")
    log(f"elapsed              : {budget.elapsed():.1f}s")

    val_predictions = _evaluate(
        model,
        records=val_records,
        change_cache=change_cache,
        text_cache=text_cache_val,
        targets=targets,
        batch_size=args.batch_size,
        device=device,
    )
    report = None
    if val_predictions["predictions"] is not None:
        report = evaluate_records(
            val_predictions["scored_records"],
            val_predictions["predictions"],
            targets=[targets[r.file_name] for r in val_predictions["scored_records"]
                     if r.file_name in targets],
            split=SELECTION_SPLIT,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=_sha256(checkpoint_path),
            config_hash=args.config_hash,
            inference_config={
                "batch_size": args.batch_size,
                "device": device,
                "apply_type_mask": False,
                "decode": "argmax",
                "note": (
                    "masking off so masked and unmasked accuracy can be "
                    "compared; a masked number reported alone would hide what "
                    "the mask contributed"
                ),
            },
            n_skipped=val_predictions["n_skipped"],
        )
        report.write(output_dir / "eval_val.json")
        log(f"val accuracy         : {report.accuracy:.4f} "
            f"(written to eval_val.json)")

        qualitative = qualitative_examples(
            val_predictions["scored_records"],
            val_predictions["predictions"],
            targets=[targets[r.file_name] for r in val_predictions["scored_records"]
                     if r.file_name in targets],
        )
        (output_dir / "qualitative_val.json").write_text(
            json.dumps(qualitative, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    run_record: dict[str, Any] = {
        "schema": "change_vqa_run_v1",
        "architecture": ARCHITECTURE_VERSION,
        "state": POST_TRAINING_STATE,
        "state_note": (
            "training produces an artifact, not a verified capability. R-02 "
            "reaches VERIFIED only after the returned checkpoint has been "
            "evaluated on the held-out split."
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "dataset_id": DATASET_ID,
            "preprocessing_version": PREPROCESSING_VERSION,
            "root": str(args.data_root),
            "train_records": len(train_records),
            "val_records": len(val_records),
            "test_records": "not_read",
            "selection_drops": {
                "train": train_drops,
                "val": val_drops,
            },
            "expected_split_sizes": {
                k: {"scenes": s, "questions": q}
                for k, (s, q) in EXPECTED_SPLIT_SIZES.items()
            },
            "integrity": integrity,
            "statistics": stats,
            "balancing": balancing,
        },
        "features": {
            "feature_spec": FEATURE_SPEC,
            "change_cache": {
                "path": str(args.change_cache),
                "scenes": len(change_cache),
                "spec_hash": change_cache.spec_hash,
                "detector_trained": bool(
                    change_cache.extractor_config.get("trained", False)
                ),
                "extractor_config": change_cache.extractor_config,
            },
            "text_cache_train": {
                "path": str(args.text_cache_train),
                "questions": len(text_cache_train),
                "spec_hash": text_cache_train.spec_hash,
            },
            "text_cache_val": {
                "path": str(args.text_cache_val),
                "questions": len(text_cache_val),
                "spec_hash": text_cache_val.spec_hash,
            },
            "targets_path": str(args.targets),
            "target_scenes": len(targets),
        },
        "model": {
            "parameters": n_params,
            "trunk_dim": args.trunk_dim,
            "text_dim": args.text_dim,
            "dropout": args.dropout,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "last_checkpoint": str(last_checkpoint_path),
            "last_checkpoint_sha256": _sha256(last_checkpoint_path),
        },
        "optimization": {
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler": "cosine_with_warmup",
            "warmup_steps": warmup_steps,
            "total_steps_planned": total_steps,
            "grad_clip": args.grad_clip,
            "batch_size": args.batch_size,
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "loss_weights": DEFAULT_LOSS_WEIGHTS,
            "amp_enabled": amp_enabled,
            "amp_reason": amp_reason,
            "amp_dtype": str(amp_dtype) if amp_enabled else "float32",
            "grad_scaler": scaler is not None,
            "native_bfloat16": native_bfloat16_supported(),
        },
        "selection": {
            "metric": f"{SELECTION_SPLIT} answer accuracy",
            "best_epoch": best_epoch,
            "best_accuracy": round(best_accuracy, 6),
            "patience": args.patience,
            "min_delta": args.min_delta,
            "stop_reason": stop_reason,
        },
        "confidence": {
            "method": "uncalibrated",
            "note": (
                "raw softmax plus top1-top2 margin. Nothing is fitted here; the "
                "R-03 calibration contract is a separate, open ruling."
            ),
        },
        "reproducibility": seeds,
        # Provenance defect section 6b. `hashes.json` pinned the checkpoint by
        # sha256 but identified the CODE only as a path, and the Kaggle revision
        # probe returned `git_available: false` because the code arrives as a
        # dataset with no `.git`. A path is not a revision. This records the git
        # commit when there genuinely is one (with its dirty flag, since a dirty
        # tree means the commit does NOT identify the files on disk) plus a
        # reproducible content digest over the code tree. Missing identifiers are
        # recorded as missing, never as placeholders.
        "code_revision": revision_block(REPO_ROOT),
        "environment": environment,
        "budget": {
            "time_limit_seconds": args.time_limit_seconds,
            "elapsed_seconds": round(budget.elapsed(), 3),
            "clock": "time.monotonic",
        },
        "history": [asdict(h) for h in history],
        "config_hash": args.config_hash or "unavailable",
        "argv": sys.argv[1:],
    }
    (output_dir / "run_record.json").write_text(
        json.dumps(run_record, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    log(f"run record           : {output_dir / 'run_record.json'}")
    log.close()
    return run_record


def _sha256(path: Path) -> str:
    """Hash a file, or say it is unavailable. Never invent a digest."""
    import hashlib

    if not path.exists():
        return "unavailable"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.change_vqa.train",
        description=(
            "Train the CDVQA reasoning head (R-02). The STANet detector is "
            "frozen; its features are read from a cache. Test splits are not "
            "readable by this program."
        ),
    )
    parser.add_argument("--data-root", default="data/cdvqa")
    parser.add_argument("--change-cache", required=True,
                        help="npz written by scripts/prepare_change_vqa.py")
    parser.add_argument("--text-cache-train", required=True)
    parser.add_argument("--text-cache-val", required=True)
    parser.add_argument("--targets", required=True,
                        help="JSONL of label-derived scene targets")
    parser.add_argument("--output-dir", default="outputs/change_vqa")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-cuda", action="store_true",
                        help="fail rather than fall back to CPU")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--time-limit-seconds", type=float,
                        default=DEFAULT_TIME_LIMIT_SECONDS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--trunk-dim", type=int, default=512)
    parser.add_argument("--text-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--balance-by-type", action="store_true",
                        help="undersample every question type to the smallest "
                             "type's count; the cost is logged")
    parser.add_argument("--max-train-records", type=int, default=None,
                        help="deterministic subsample, for a smoke run")
    parser.add_argument("--max-val-records", type=int, default=None)
    parser.add_argument("--resume", default=None,
                        help="not implemented; passing it raises rather than "
                             "silently restarting the schedule")
    parser.add_argument("--expect-change-spec", default=None,
                        help="refuse a change cache built under another spec")
    parser.add_argument("--expect-text-spec", default=None)
    parser.add_argument("--config-hash", default=None,
                        help="Config.hash of the config this run used")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.time_limit_seconds is not None and args.time_limit_seconds <= 0:
        raise SystemExit("--time-limit-seconds must be positive or omitted")
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    record = train(args)
    print()
    print(f"state: {record['state']}")
    print(f"best val accuracy: {record['selection']['best_accuracy']}")
    print(f"stop reason: {record['selection']['stop_reason']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
