"""R-02 — the local smoke training run, and the artifact it leaves behind.

WHY THIS MODULE EXISTS
----------------------
`tests/unit/test_change_vqa_amp.py` pins the AMP *decision* and the NaN guards,
and `tests/unit/test_change_vqa_head.py` section Q pins the BCE helper. Neither
of them runs the trainer. A decision function that is never driven by the real
loop is a decision function that can be wired up wrongly, and that is exactly
the failure mode this repository has already been bitten by once (the Kaggle
run died on the first optimizer step, after every static check passed).

So this module builds a **small, real CDVQA dataset on disk**, runs the actual
`training.change_vqa.train.train()` end to end, and inspects what it produced.
Nothing here reimplements the training loop; every assertion is made against an
artifact the real code wrote.

WHAT IS REAL AND WHAT IS A STAND-IN
-----------------------------------
Real, driven through the production code path:

  * the CDVQA annotation schema (`{Split}_{images,questions,answers}.json`),
    read by the shipped adapter;
  * `im1`/`im2` layout detection and the two-path pairing contract;
  * the label palette decoder and `compute_scene_targets` — the supervision
    targets are computed from actual PNGs, not typed in;
  * the **frozen STANet detector** at
    `artifacts/change/levir_change_v001/head.pt`, producing real change
    features through `ChangeFeatureExtractor`;
  * the full trainer: record selection, batching, forward, all four loss terms,
    backward, AdamW step, cosine schedule, validation, checkpoint save;
  * `load_change_vqa_head` and the serving specialist's `execute()`.

Stand-ins, and why:

  * **Question text features are synthetic** (384-d, L2-normalised). MiniLM
    loads in ~21 s from the local HF cache and this suite runs it once per
    session; the text encoder's own contract is `TextFeatureExtractor.encode`,
    which is not what this module is testing. The spec hash is the real
    `text_spec_hash()`, so the cache identity is exercised.
  * **Working resolution is 64 px**, not the default 256. The feature
    *dimension* and the whole pooling code path are unchanged; only the pooled
    values differ, and a 64 px scene keeps the suite near a second. The
    resolution is part of the spec hash, so the fixture is self-consistent and
    the serving tests below deliberately exploit the difference.
  * **The dataset is 9 scenes**, not 2,968. `verify_split_integrity` is called
    with `expect_full_split=False`, which is the documented path for a
    deliberate subsample.

WHAT THIS MODULE DOES NOT PROVE — READ THIS BEFORE CITING IT
------------------------------------------------------------
**CUDA AMP was not exercised.** This host has no CUDA device
(`torch.cuda.is_available()` is False, torch 2.14.0+cpu). `torch.amp.autocast("cuda")`
on a CPU-only host is a no-op that warns and leaves `is_autocast_enabled("cuda")
== False`, so the CUDA dispatch key that raises

    RuntimeError: torch.nn.functional.binary_cross_entropy ... unsafe to autocast

cannot be reached here at all. What IS proven locally is the *mechanism*: the
BCE call sites are invoked with autocast disabled and fp32 operands while a
genuinely active autocast context is running around them (see
`test_smoke_the_bce_helper_is_protecting_a_genuinely_bfloat16_probability`).
CPU autocast with bfloat16 does engage on this build, which makes that a real
test rather than a vacuous one — but it is still not a T4.

Consequently the honest statement after this module is: **the training path runs
end to end in fp32 on CPU, and the AMP-relevant code is exercised against a real
active autocast context, but full CUDA AMP behaviour has not been observed.**
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import tempfile
import warnings
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest
from PIL import Image

from core.errors import SpecialistError
from specialists.change.vqa_specialist import ChangeVQASpecialist
from training.change_vqa import dataset as D
from training.change_vqa import model as M
from training.change_vqa import train as T
from training.change_vqa import vocab as V
from training.change_vqa.features import (
    CHANGE_FEATURE_DIM,
    TEXT_FEATURE_DIM,
    ChangeFeatureCache,
    ChangeFeatureExtractor,
    TextFeatureCache,
    extract_change_features,
    text_spec_hash,
)
from training.change_vqa.dataset import read_scene_targets
from training.change_vqa.evaluate import answer_accuracy

REPO_ROOT = Path(__file__).resolve().parents[2]
STANET_CHECKPOINT = REPO_ROOT / "artifacts" / "change" / "levir_change_v001" / "head.pt"

#: The frozen artifact's recorded identity. Checked here because this module is
#: the one place that actually loads it, and "never regenerate the frozen
#: artifacts" is only enforceable if something notices when one moves.
STANET_SIZE_BYTES = 63_231_009
STANET_SHA256 = (
    "c5ef31277b67aa01a593aec0eac503eeaccc6d674349fda20ca44c9cc6f8e9fa"
)
STANET_PARAMETERS = 15_779_969

#: The reasoning head's parameter count. Pinned because the whole R-02 budget
#: argument ("train ~1.45 M, not 15.8 M") rests on it, and a silent architecture
#: change would move it.
HEAD_PARAMETERS = 1_453_912

#: Reduced working resolution for speed. See the module docstring.
SMOKE_IMAGE_SIZE = 64

TRAIN_SCENES = ("00001", "00002", "00003", "00004", "00005", "00006")
VAL_SCENES = ("00011", "00012", "00013")

#: Scenes that exist ONLY in the forbidden splits. Named with a token that
#: cannot occur by coincidence, so "was Test ever read?" is answerable by
#: searching the serialized run record for it.
TEST_ONLY_TOKEN = "zztestonly"
TEST_ONLY_SCENES = (f"{TEST_ONLY_TOKEN}0001", f"{TEST_ONLY_TOKEN}0002")

#: One question per type, with a question string the resolver actually fires on.
#: The `type` column is what the trainer reads; the prose is what the free-form
#: resolver sees, and the serving test below depends on it resolving.
QUESTIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("change_or_not", "Have the areas of buildings changed?", ("yes", "no")),
    (
        "change_ratio_types",
        "What is the change percentage of buildings in the first image?",
        V.RATIO_TYPE_BINS,
    ),
    ("increase_or_not", "Has the area of water increased?", ("yes", "no")),
    ("decrease_or_not", "Has the area of trees decreased?", ("yes", "no")),
    (
        "change_to_what",
        "What has the area of low vegetation changed to?",
        V.CHANGE_CLASS_ORDER,
    ),
    (
        "smallest_change",
        "What is the smallest change in the post-change image?",
        V.CHANGE_CLASS_ORDER,
    ),
    (
        "largest_change",
        "What is the largest change in the image?",
        V.CHANGE_CLASS_ORDER,
    ),
    (
        "change_ratio",
        "What is the percentage of change in the image?",
        V.RATIO_BINS,
    ),
)


# ---------------------------------------------------------------------------
# Fixture plumbing
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _png_rasterio_is_quiet() -> Iterator[None]:
    """Silence rasterio's "no geotransform" warning for the PNG stand-ins.

    `load_image_array` goes through rasterio, which warns that a dataset has no
    geotransform. For a real GeoTIFF that warning is meaningful; for a PNG
    stand-in it is a statement about the fixture, not about the code under test.
    It is suppressed narrowly rather than globally so that `-W error::UserWarning`
    still catches warnings raised by this repository's own code.
    """
    try:
        from rasterio.errors import NotGeoreferencedWarning as _Warning
    except Exception:  # noqa: BLE001 - rasterio absent: nothing to silence
        _Warning = UserWarning  # type: ignore[assignment,misc]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", _Warning)
        yield


def _load_image(path: Any) -> Any:
    from preprocessing.imagery import load_image_array

    with _png_rasterio_is_quiet():
        return load_image_array(path)


def _write_rgb_png(path: Path, pixels: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode="RGB").save(path)
    return path


def _write_label_png(
    path: Path, colours: list[tuple[int, int, int]], size: int = 8
) -> Path:
    """A label map in the frozen palette (white = shared background)."""
    assert len(colours) == size * size
    image = Image.new("RGB", (size, size))
    image.putdata(colours)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _label_colours(*, variant: int) -> list[tuple[int, int, int]]:
    """A 8x8 palette map. `variant` changes a few pixels, so label2 != label1.

    The change is deliberately small and located, so `class_mag` is a fraction
    rather than 0 or 1 and `total_changed` sits strictly inside (0, 1) — a
    target pinned at its bounds would let a saturated model look correct.
    """
    colours = [D.BACKGROUND_RGB] * 64
    for index in range(8):
        colours[index] = D.LABEL_PALETTE["buildings"]
    for index in range(8, 14):
        colours[index] = D.LABEL_PALETTE["water"]
    if variant:
        for index in range(14, 14 + variant):
            colours[index] = D.LABEL_PALETTE["trees"]
    return colours


def _write_annotations(
    root: Path, split: str, scenes: tuple[str, ...]
) -> list[int]:
    """Write the three annotation tables for one split. Returns its question ids.

    `question_id` restarts at 0 for every split, matching the shipped corpus —
    which is the property that made the cross-split collision possible and is
    therefore worth reproducing in the fixture.
    """
    annotations = root / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)

    image_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    question_ids: list[int] = []
    question_id = 0
    answer_id = 0

    for scene_index, scene in enumerate(scenes):
        refs: list[int] = []
        for q_index, (qtype, text, answers) in enumerate(QUESTIONS):
            answer = answers[(scene_index + q_index) % len(answers)]
            question_rows.append(
                {
                    "id": question_id,
                    "date_added": 1631214265.0,
                    "img_id": scene_index,
                    "type": qtype,
                    "question": text,
                    "answers_ids": [answer_id],
                    "active": True,
                }
            )
            answer_rows.append(
                {
                    "id": answer_id,
                    "date_added": 1631214265.0,
                    "question_id": question_id,
                    "answer": answer,
                    "active": True,
                }
            )
            refs.append(question_id)
            question_ids.append(question_id)
            question_id += 1
            answer_id += 1
        image_rows.append(
            {
                "id": scene_index,
                "res_x": str(SMOKE_IMAGE_SIZE),
                "res_y": str(SMOKE_IMAGE_SIZE),
                "questions_ids": refs,
                "file_name": f"{scene}.png",
                "active": True,
            }
        )

    for name, key, rows in (
        ("images", "images", image_rows),
        ("questions", "questions", question_rows),
        ("answers", "answers", answer_rows),
    ):
        (annotations / f"{split}_{name}.json").write_text(
            json.dumps({key: rows}), encoding="utf-8"
        )
    return question_ids


def _build_dataset(root: Path) -> None:
    """A tiny but structurally complete CDVQA root, plus deliberately broken
    Test/Test2 splits (see `test_smoke_the_trainer_never_reads_the_test_splits`).
    """
    for split, scenes in (
        ("Train", TRAIN_SCENES),
        ("Val", VAL_SCENES),
        ("Test", TEST_ONLY_SCENES),
        ("Test2", TEST_ONLY_SCENES),
    ):
        _write_annotations(root, split, scenes)

    rng = np.random.default_rng(20260921)
    for scene_index, scene in enumerate(TRAIN_SCENES + VAL_SCENES):
        # A deterministic, non-constant image, so the frozen encoder sees
        # structure rather than a flat field.
        base = rng.integers(0, 256, size=(SMOKE_IMAGE_SIZE, SMOKE_IMAGE_SIZE, 3))
        _write_rgb_png(root / "im1" / f"{scene}.png", base)
        shifted = np.roll(base, shift=3 + scene_index, axis=0)
        _write_rgb_png(root / "im2" / f"{scene}.png", shifted)
        _write_label_png(root / "label1" / f"{scene}.png", _label_colours(variant=0))
        _write_label_png(
            root / "label2" / f"{scene}.png", _label_colours(variant=2 + scene_index % 4)
        )

    # The forbidden splits are present but UNREADABLE. Nothing reads them, so a
    # trainer that widens its split list fails here instead of quietly training
    # on the data it will be graded on.
    for split in ("Test", "Test2"):
        for table in ("images", "questions", "answers"):
            (root / "annotations" / f"{split}_{table}.json").write_text(
                "{ this split is deliberately unreadable", encoding="utf-8"
            )


def _synthetic_text_features(count: int, *, seed: int) -> Any:
    """384-d L2-normalised stand-ins, shaped like MiniLM's output."""
    generator = np.random.default_rng(seed)
    vectors = generator.normal(size=(count, TEXT_FEATURE_DIM)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def _build_inputs(root: Path, work: Path) -> dict[str, Any]:
    """Targets + change cache + text caches, all through the production code."""
    scenes = TRAIN_SCENES + VAL_SCENES

    targets = [
        D.compute_scene_targets(
            root / "label1" / f"{scene}.png",
            root / "label2" / f"{scene}.png",
            f"{scene}.png",
        )
        for scene in scenes
    ]
    targets_path = D.write_scene_targets(targets, work / "targets.jsonl")

    extractor = _extractor()
    change_cache = extract_change_features(
        extractor,
        [
            (
                scene,
                str(root / "im1" / f"{scene}.png"),
                str(root / "im2" / f"{scene}.png"),
            )
            for scene in scenes
        ],
        loader=_load_image,
    )
    assert not change_cache.failures, change_cache.failures  # type: ignore[attr-defined]
    change_cache_path = change_cache.write(work / "change_cache.npz")

    records = D.load_change_vqa_records(root, splits=("Train", "Val"))
    by_split = D.group_by_native_split(records)

    spec = text_spec_hash(encoder="synthetic-smoke-fixture")
    text_paths: dict[str, Path] = {}
    for split in ("Train", "Val"):
        rows = by_split[split]
        cache = TextFeatureCache(
            question_ids=tuple(record.question_id for record in rows),
            features=_synthetic_text_features(len(rows), seed=split_seed(split)),
            spec_hash=spec,
        )
        text_paths[split] = cache.write(work / f"text_cache_{split.lower()}.npz")

    return {
        "root": root,
        "work": work,
        "targets": targets_path,
        "change_cache": change_cache_path,
        "text_cache_train": text_paths["Train"],
        "text_cache_val": text_paths["Val"],
        "records": records,
    }


def split_seed(split: str) -> int:
    return 11 if split == "Train" else 12


def _smoke_argv(inputs: dict[str, Any], output_dir: Path) -> list[str]:
    return [
        "--data-root", str(inputs["root"]),
        "--change-cache", str(inputs["change_cache"]),
        "--text-cache-train", str(inputs["text_cache_train"]),
        "--text-cache-val", str(inputs["text_cache_val"]),
        "--targets", str(inputs["targets"]),
        "--output-dir", str(output_dir),
        "--device", "cpu",
        # `--amp` is the DEFAULT, and it is the flag Kaggle will pass. On a CPU
        # host the trainer must degrade to fp32 AND record that it did, rather
        # than training in fp32 while the run record claims mixed precision.
        "--amp",
        "--epochs", "2",
        "--batch-size", "4",
        "--time-limit-seconds", "900",
        "--seed", "42",
    ]


# ---------------------------------------------------------------------------
# One real run, memoised
# ---------------------------------------------------------------------------
_SCRATCH: Path | None = None
_EXTRACTOR: ChangeFeatureExtractor | None = None
_SMOKE: dict[str, Any] | None = None


def _scratch() -> Path:
    """A `mkdtemp` directory that is never deleted.

    pytest's `tmp_path` performs directory removals, which a sandbox delete
    guard in this environment has blocked before; a directory that is never
    removed needs no delete permission. The OS temp reaper owns it.
    """
    global _SCRATCH
    if _SCRATCH is None:
        _SCRATCH = Path(tempfile.mkdtemp(prefix="sq_r02_smoke_"))
    return _SCRATCH


def _extractor() -> ChangeFeatureExtractor:
    """The frozen STANet at the reduced working resolution, loaded once."""
    global _EXTRACTOR
    if _EXTRACTOR is None:
        if not STANET_CHECKPOINT.exists():
            pytest.skip(
                f"the frozen STANet checkpoint is absent ({STANET_CHECKPOINT}); "
                f"the smoke run cannot build real change features"
            )
        from specialists.change.stanet import load_change_model

        model = load_change_model(str(STANET_CHECKPOINT), device="cpu")
        _EXTRACTOR = ChangeFeatureExtractor(
            model=model,
            device="cpu",
            image_size=SMOKE_IMAGE_SIZE,
            trained=True,
        )
    return _EXTRACTOR


def smoke() -> dict[str, Any]:
    """Build the dataset and run the real trainer ONCE for the whole module.

    Memoised because the run costs a frozen-checkpoint load and two epochs, and
    every test below reads a different part of the same artifact. The result is
    treated as read-only; tests that need a fresh model reload it from disk.
    """
    global _SMOKE
    if _SMOKE is not None:
        return _SMOKE

    scratch = _scratch()
    root = scratch / "cdvqa"
    work = scratch / "inputs"
    work.mkdir(parents=True, exist_ok=True)
    _build_dataset(root)
    inputs = _build_inputs(root, work)

    output_dir = scratch / "run"
    args = T.build_parser().parse_args(_smoke_argv(inputs, output_dir))
    record = T.train(args)

    _SMOKE = {
        "scratch": scratch,
        "root": root,
        "output_dir": output_dir,
        "inputs": inputs,
        "record": record,
    }
    return _SMOKE


def _record() -> dict[str, Any]:
    return smoke()["record"]


def _reloaded_head() -> Any:
    return M.load_change_vqa_head(smoke()["output_dir"] / "head.pt", device="cpu")


def _read_eval_val() -> dict[str, Any]:
    return json.loads(
        (smoke()["output_dir"] / "eval_val.json").read_text(encoding="utf-8")
    )


# ===========================================================================
# The run itself
# ===========================================================================
def test_smoke_the_real_trainer_completes_two_epochs_and_reports_trained_unverified() -> None:
    """It ran, it finished on its own terms, and it did not claim verification.

    `TRAINED_UNVERIFIED` is the whole point of the status vocabulary: training
    produces an artifact, and an artifact is not a verified capability.
    """
    record = _record()
    assert record["state"] == T.POST_TRAINING_STATE == "TRAINED_UNVERIFIED"
    assert record["schema"] == "change_vqa_run_v1"
    assert record["architecture"] == M.ARCHITECTURE_VERSION
    assert record["optimization"]["epochs_requested"] == 2
    assert record["optimization"]["epochs_completed"] == 2
    assert record["selection"]["stop_reason"] == "epochs_exhausted"
    assert len(record["history"]) == 2
    assert record["dataset"]["train_records"] == len(TRAIN_SCENES) * len(QUESTIONS)
    assert record["dataset"]["val_records"] == len(VAL_SCENES) * len(QUESTIONS)


def test_smoke_all_four_loss_terms_are_computed_and_finite() -> None:
    """Every term of the composite loss must actually have been evaluated.

    A term that silently collapsed to zero (a bad broadcast, a detached branch)
    would leave `total` finite and the run would look healthy while training
    only the answer classifier.
    """
    record = _record()
    for entry in record["history"]:
        components = entry["loss_components"]
        for key in (
            "total",
            "answer_ce",
            "class_magnitude_bce",
            "class_delta_mse",
            "total_changed_bce",
        ):
            assert key in components, key
            assert np.isfinite(components[key]), (entry["epoch"], key, components[key])

        # The three estimator terms and the answer term are all strictly
        # positive at random initialisation; a zero would mean the term never
        # ran.
        assert components["answer_ce"] > 0.0
        assert components["class_magnitude_bce"] > 0.0
        assert components["total_changed_bce"] > 0.0
        assert components["class_delta_mse"] > 0.0
        assert entry["train_loss"] > 0.0
        assert entry["n_records"] > 0
        assert entry["skips"] == {
            "missing_change_feature": 0,
            "missing_text_feature": 0,
            "missing_targets": 0,
        }


def test_smoke_the_optimizer_moved_the_weights_not_just_the_log() -> None:
    """A run that logs loss but never steps would pass every check above."""
    import torch

    saved = torch.load(
        str(smoke()["output_dir"] / "head.pt"), map_location="cpu", weights_only=False
    )
    fresh = M.build_change_vqa_head(
        change_feature_dim=CHANGE_FEATURE_DIM, text_feature_dim=TEXT_FEATURE_DIM
    )
    reference = fresh.state_dict()

    assert set(saved["state_dict"]) == set(reference)
    differing = [
        name
        for name, tensor in saved["state_dict"].items()
        if not torch.equal(tensor, reference[name])
    ]
    # Every tensor in the model receives gradient on the first step, so a run
    # that really stepped moves essentially all of them.
    assert len(differing) == len(reference), sorted(set(reference) - set(differing))


def test_smoke_the_head_has_the_recorded_parameter_count() -> None:
    """The 1.45 M figure is the entire justification for the <=3 h budget."""
    record = _record()
    assert record["model"]["parameters"] == HEAD_PARAMETERS
    head = M.build_change_vqa_head(
        change_feature_dim=CHANGE_FEATURE_DIM, text_feature_dim=TEXT_FEATURE_DIM
    )
    assert head.num_parameters() == HEAD_PARAMETERS
    assert _reloaded_head().num_parameters() == HEAD_PARAMETERS


def test_smoke_the_run_record_measures_what_it_can_and_says_unavailable_otherwise() -> None:
    """No field may be invented. A measured field is measured; an unmeasured one
    says so."""
    record = _record()
    assert record["model"]["checkpoint_sha256"] != "unavailable"
    assert record["model"]["last_checkpoint_sha256"] != "unavailable"
    # No --config-hash was passed, so the honest value is "unavailable" — not a
    # plausible-looking digest.
    assert record["config_hash"] == "unavailable"
    assert record["reproducibility"]["seed"] == 42
    assert record["reproducibility"]["torch_seeded"] is True
    assert record["environment"]["cuda_available"] is False
    assert record["environment"]["torch"].startswith("2.")
    assert record["confidence"]["method"] == "uncalibrated"


def test_smoke_amp_requested_on_a_cpu_host_is_recorded_as_fp32_not_claimed() -> None:
    """`--amp` was passed and could not be honoured. The record must say so.

    This is the difference between "trained with mixed precision" and "asked for
    mixed precision"; a record that conflates them misdescribes the run.
    """
    record = _record()
    optimization = record["optimization"]
    assert optimization["amp_enabled"] is False
    assert optimization["amp_dtype"] == "float32"
    assert optimization["grad_scaler"] is False
    assert optimization["native_bfloat16"] is False
    assert "unavailable" in optimization["amp_reason"]
    assert "cpu" in optimization["amp_reason"]


# ===========================================================================
# Test / Test2 protection
# ===========================================================================
def test_smoke_the_trainer_never_reads_the_test_splits_even_when_they_are_broken() -> None:
    """The strongest available proof that Test/Test2 are unreachable.

    The fixture's `Test_*.json` and `Test2_*.json` files are not valid JSON. If
    any code path in training read them, the run would have died with a
    `CDVQAError` instead of producing the record asserted here. The run
    succeeding IS the assertion.
    """
    assert T.READABLE_SPLITS == ("Train", "Val")
    assert T.FORBIDDEN_SPLITS == ("Test", "Test2")
    assert not set(T.FORBIDDEN_SPLITS) & set(T.READABLE_SPLITS)

    record = _record()
    assert record["dataset"]["test_records"] == "not_read"
    assert record["dataset"]["expected_split_sizes"]["Test"]["scenes"] == 968

    serialized = json.dumps(record, default=str)
    assert TEST_ONLY_TOKEN not in serialized, (
        "a scene that exists only in Test/Test2 appears in the run record, so "
        "something read a forbidden split"
    )


def test_smoke_a_truncated_run_refuses_to_report_itself_as_a_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that completed no epoch must raise, not return a record.

    The brief is explicit that a truncated run must never be reported as the
    final result. `train()` checks the budget before each epoch; if the budget
    is already gone, `history` is empty and this guard fires. The budget is
    stubbed rather than set to a tiny number so the test is deterministic.
    """

    class _Exhausted:
        limit_seconds = 0.0

        def elapsed(self) -> float:
            return 0.0

        def remaining(self) -> float:
            return 0.0

        def exhausted(self) -> bool:
            return True

        def to_dict(self) -> dict[str, Any]:
            return {"limit_seconds": 0.0, "elapsed_seconds": 0.0}

    scratch = _scratch()
    inputs = smoke()["inputs"]
    output_dir = scratch / "truncated"
    args = T.build_parser().parse_args(_smoke_argv(inputs, output_dir))

    monkeypatch.setattr(T, "TimeBudget", lambda _limit: _Exhausted())
    with pytest.raises(SpecialistError, match="no epoch completed"):
        T.train(args)


def test_smoke_the_frozen_stanet_artifact_is_byte_identical_to_the_recorded_digest() -> None:
    """Nothing in this module may have rewritten the frozen detector.

    Checked here rather than assumed because this is the only module that loads
    it, and "the frozen artifacts must never be regenerated" is only enforceable
    if a test notices when one moves.
    """
    if not STANET_CHECKPOINT.exists():
        pytest.skip(f"frozen checkpoint absent: {STANET_CHECKPOINT}")
    assert STANET_CHECKPOINT.stat().st_size == STANET_SIZE_BYTES

    digest = hashlib.sha256()
    with STANET_CHECKPOINT.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    assert digest.hexdigest() == STANET_SHA256

    import torch

    payload = torch.load(str(STANET_CHECKPOINT), map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    assert sum(t.numel() for t in state.values() if hasattr(t, "numel")) >= 0
    assert _extractor().model is not None
    assert (
        sum(p.numel() for p in _extractor().model.parameters()) == STANET_PARAMETERS
    )


# ===========================================================================
# The checkpoint -> serving round trip
# ===========================================================================
def test_smoke_the_checkpoint_reloads_and_reproduces_the_selected_validation_accuracy() -> None:
    """The artifact Kaggle uploads must answer the validation set identically.

    `T._evaluate` is called directly rather than reimplemented: the point is
    that the *same* evaluation, run against the *reloaded* checkpoint, gives the
    same number as the run that selected it. A reimplementation could agree for
    the wrong reason.
    """
    smoke_run = smoke()
    reloaded = _reloaded_head()
    inputs = smoke_run["inputs"]
    work = inputs["work"]

    change_cache = ChangeFeatureCache.read(work / "change_cache.npz")
    text_cache_val = TextFeatureCache.read(work / "text_cache_val.npz")
    targets = read_scene_targets(inputs["targets"])
    val_records = D.group_by_native_split(inputs["records"])["Val"]

    rescored = T._evaluate(
        reloaded,
        records=val_records,
        change_cache=change_cache,
        text_cache=text_cache_val,
        targets=targets,
        batch_size=4,
        device="cpu",
    )
    recorded = _read_eval_val()
    assert rescored["n_scored"] == len(val_records)
    # The record rounds to 6 dp, so the round trip is asserted at the precision
    # the number is actually recorded at. Comparing raw floats with a float
    # tolerance would only be testing `round()`.
    assert round(rescored["accuracy"], 6) == recorded["metrics"]["answer_accuracy"]


def test_smoke_the_published_probability_contract_survives_the_checkpoint() -> None:
    """`class_mag`/`total_changed` are sigmoid probabilities and `class_delta` is
    tanh — after a save/load round trip, not only in memory.

    This is the contract that makes `binary_cross_entropy` the correct loss and
    `binary_cross_entropy_with_logits` the wrong one, so it is pinned on the
    artifact rather than on a freshly built module.
    """
    reloaded = _reloaded_head()
    assert getattr(reloaded, "_satquery_trained", False) is True
    assert reloaded.training is False

    inputs = smoke()["inputs"]
    work = inputs["work"]
    change_cache = ChangeFeatureCache.read(work / "change_cache.npz")
    text_cache_val = TextFeatureCache.read(work / "text_cache_val.npz")
    val_records = D.group_by_native_split(inputs["records"])["Val"]

    change_rows = np.stack(
        [change_cache.get(record.scene_key) for record in val_records]
    ).astype(np.float32)
    position = {q: i for i, q in enumerate(text_cache_val.question_ids)}
    text_rows = np.stack(
        [text_cache_val.features[position[record.question_id]] for record in val_records]
    ).astype(np.float32)

    predictions = M.predict_answers(
        reloaded,
        change_features=change_rows,
        text_features=text_rows,
        qtype_indices=np.asarray([r.qtype_index for r in val_records], dtype=np.int64),
        temporal_indices=np.asarray(
            [V.TEMPORAL_REFERENCE_TO_INDEX.get(r.temporal_ref, 0) for r in val_records],
            dtype=np.int64,
        ),
        qtypes=[r.qtype for r in val_records],
        apply_type_mask=False,
        device="cpu",
    )

    mag = predictions["class_mag"]
    delta = predictions["class_delta"]
    total = predictions["total_changed"]
    probabilities = predictions["probabilities"]

    assert mag.shape == (len(val_records), V.N_CHANGE_CLASSES)
    assert float(mag.min()) >= 0.0 and float(mag.max()) <= 1.0
    assert float(delta.min()) >= -1.0 and float(delta.max()) <= 1.0
    assert float(total.min()) >= 0.0 and float(total.max()) <= 1.0
    assert probabilities.shape == (len(val_records), V.N_ANSWERS)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    for answer in predictions["answer"]:
        assert answer in V.ANSWER_TO_INDEX


def test_smoke_the_serving_path_loads_the_checkpoint_this_run_produced() -> None:
    """Train/serve skew is checked by driving the specialist, not by reading it.

    The specialist must accept the head, agree on the feature spec, and return a
    real answer from the 19-answer vocabulary — which also proves the checkpoint
    carries everything serving needs.
    """
    smoke_run = smoke()
    head_path = smoke_run["output_dir"] / "head.pt"
    metadata = json.loads(
        (smoke_run["output_dir"] / "model_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["state"] == "TRAINED_UNVERIFIED"
    assert metadata["test_splits_used"] is False
    assert metadata["confidence_method"] == "uncalibrated"

    specialist = ChangeVQASpecialist(
        head=_reloaded_head(),
        feature_extractor=_extractor(),
        text_encoder=_StubTextEncoder(),
        head_path=head_path,
        head_metadata=metadata,
    )

    assert specialist.has_head is True
    assert specialist.feature_spec_mismatch() is None
    assert specialist.unavailable_reason() is None

    request = _serving_request(
        smoke_run["root"], "What is the largest change in the image?"
    )
    with _png_rasterio_is_quiet():
        result = specialist.execute(request)

    assert result.answer != "", "a loaded, matched head must be able to answer"
    assert result.answer in V.ANSWER_TO_INDEX
    assert result.degraded is False


def test_smoke_the_serving_path_refuses_the_checkpoint_under_a_mismatched_feature_spec() -> None:
    """The same head, served at a DIFFERENT resolution, must refuse to answer.

    The spec hash encodes the resolution and the trained/untrained state, so a
    head fitted on one representation and served another is a real skew. The
    specialist must refuse rather than emit a fluent answer computed from a
    representation the head never saw.
    """
    smoke_run = smoke()
    metadata = json.loads(
        (smoke_run["output_dir"] / "model_metadata.json").read_text(encoding="utf-8")
    )
    mismatched = ChangeFeatureExtractor(
        model=_extractor().model, device="cpu", image_size=256, trained=True
    )
    assert mismatched.spec_hash != metadata["change_cache_spec"]

    specialist = ChangeVQASpecialist(
        head=_reloaded_head(),
        feature_extractor=mismatched,
        text_encoder=_StubTextEncoder(),
        head_metadata=metadata,
    )
    mismatch = specialist.feature_spec_mismatch()
    assert mismatch is not None and "never fitted on" in mismatch

    request = _serving_request(
        smoke_run["root"], "What is the largest change in the image?"
    )
    with _png_rasterio_is_quiet():
        result = specialist.execute(request)
    assert result.answer == ""
    assert result.degraded is True


# ===========================================================================
# Mixed precision — what CAN and CANNOT be exercised on this host
# ===========================================================================
def test_smoke_a_real_step_runs_under_an_active_reduced_precision_autocast() -> None:
    """The full step — forward, all four loss terms, backward, AdamW — under a
    genuinely ACTIVE autocast context.

    The first assertion is the important one. `torch.amp.autocast("cuda")` on a
    CPU-only host silently does nothing, so a test that merely wrapped a step in
    it would pass without exercising anything. CPU autocast with bfloat16 does
    engage on this build, and that is asserted before anything else.
    """
    import torch

    torch.manual_seed(42)
    head = M.build_change_vqa_head(
        change_feature_dim=CHANGE_FEATURE_DIM, text_feature_dim=TEXT_FEATURE_DIM
    )
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)

    batch = 4
    generator = torch.Generator().manual_seed(7)
    change = torch.randn(batch, CHANGE_FEATURE_DIM, generator=generator)
    text = torch.randn(batch, TEXT_FEATURE_DIM, generator=generator)
    qtype = torch.zeros(batch, dtype=torch.long)
    temporal = torch.zeros(batch, dtype=torch.long)
    answers = torch.zeros(batch, dtype=torch.long)
    magnitude = torch.rand(batch, V.N_CHANGE_CLASSES, generator=generator)
    delta = torch.rand(batch, V.N_CHANGE_CLASSES, generator=generator) * 2 - 1
    total = torch.rand(batch, generator=generator)

    before = {name: tensor.detach().clone() for name, tensor in head.state_dict().items()}

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        # Not a no-op: this is what makes the rest of the test meaningful.
        assert torch.is_autocast_enabled("cpu") is True
        assert torch.get_autocast_dtype("cpu") is torch.bfloat16

        output = head(change, text, qtype, temporal)
        assert output.answer_logits.dtype is torch.bfloat16, (
            "the autocast context did not reduce the compute dtype, so this "
            "test would prove nothing"
        )
        loss = M.change_vqa_loss(
            output,
            answer_index=answers,
            class_mag=magnitude,
            class_delta=delta,
            total_changed=total,
        )
        assert torch.isfinite(loss.total).item() is True
        loss.total.backward()

    # Every loss term is fp32 despite the bf16 context — cross_entropy and
    # mse_loss are autocast-safe, and the BCE helper forces fp32 itself.
    for term in (loss.total, loss.answer, loss.magnitude, loss.delta, loss.total_changed):
        assert term.dtype is torch.float32

    for name, parameter in head.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name

    optimizer.step()
    moved = [
        name
        for name, tensor in head.state_dict().items()
        if not torch.equal(tensor, before[name])
    ]
    assert len(moved) == len(before)


def test_smoke_the_loss_takes_the_logits_and_keeps_them_fp32_under_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loss path the trainer actually takes, inside a real autocast block.

    Two facts are pinned, and both matter:

    1. `change_vqa_loss` computes the two BCE terms from the LOGITS, not from
       the emitted probabilities. That is the whole fix for the saturated-sigmoid
       NaN — see section S of `test_change_vqa_head.py`.
    2. The head still emits reduced-precision PROBABILITIES under autocast, which
       is the published output. The loss simply no longer consumes them, so the
       precision loss that used to reach the loss cannot reach it any more.

    The recorded call arguments are the proof: autocast OFF at the call, and both
    operands fp32.
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(42)
    head = M.build_change_vqa_head(
        change_feature_dim=CHANGE_FEATURE_DIM, text_feature_dim=TEXT_FEATURE_DIM
    )
    head.train()

    observed: list[tuple[bool, Any, Any]] = []
    real = F.binary_cross_entropy_with_logits

    def _spy(logits: Any, target: Any, *args: Any, **kwargs: Any) -> Any:
        observed.append(
            (
                bool(torch.is_autocast_enabled(logits.device.type)),
                logits.dtype,
                target.dtype,
            )
        )
        return real(logits, target, *args, **kwargs)

    monkeypatch.setattr(F, "binary_cross_entropy_with_logits", _spy)

    batch = 4
    generator = torch.Generator().manual_seed(7)
    change = torch.randn(batch, CHANGE_FEATURE_DIM, generator=generator)
    text = torch.randn(batch, TEXT_FEATURE_DIM, generator=generator)
    qtype = torch.zeros(batch, dtype=torch.long)
    temporal = torch.zeros(batch, dtype=torch.long)

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu") is True
        output = head(change, text, qtype, temporal)
        # The premise: the PUBLISHED probabilities really are reduced precision.
        assert output.class_mag.dtype is torch.bfloat16
        assert output.total_changed.dtype is torch.bfloat16
        # ...and the logits the loss will use are carried alongside them.
        assert output.mag_logits is not None
        assert output.total_logits is not None

        M.change_vqa_loss(
            output,
            answer_index=torch.zeros(batch, dtype=torch.long),
            class_mag=torch.rand(batch, V.N_CHANGE_CLASSES, generator=generator),
            class_delta=torch.rand(batch, V.N_CHANGE_CLASSES, generator=generator),
            total_changed=torch.rand(batch, generator=generator),
        )

    # Two call sites: class magnitude and total changed. Both on fp32 logits.
    assert len(observed) == 2, observed
    for autocast_active, logits_dtype, target_dtype in observed:
        assert autocast_active is False, (
            "the loss invoked a BCE op while autocast was active; on CUDA that "
            "is the exact call that raises RuntimeError"
        )
        assert logits_dtype is torch.float32
        assert target_dtype is torch.float32


def test_smoke_the_accuracy_helper_agrees_with_the_recorded_metric() -> None:
    """A guard on the assertion above: `answer_accuracy` is the metric the run
    record reports, so the round-trip test compares like with like."""
    smoke_run = smoke()
    record = smoke_run["record"]
    reloaded = _reloaded_head()

    inputs = smoke_run["inputs"]
    change_cache = ChangeFeatureCache.read(inputs["work"] / "change_cache.npz")
    text_cache_val = TextFeatureCache.read(inputs["work"] / "text_cache_val.npz")
    val_records = D.group_by_native_split(inputs["records"])["Val"]
    targets = read_scene_targets(inputs["targets"])

    rescored = T._evaluate(
        reloaded,
        records=val_records,
        change_cache=change_cache,
        text_cache=text_cache_val,
        targets=targets,
        batch_size=4,
        device="cpu",
    )
    gold = [record_.answer_index for record_ in rescored["scored_records"]]
    predicted = [int(v) for v in rescored["predictions"]["answer_index"]]
    assert rescored["accuracy"] == answer_accuracy(predicted, gold)
    assert record["selection"]["best_accuracy"] == round(
        max(entry["val_accuracy"] for entry in record["history"]), 6
    )


# ---------------------------------------------------------------------------
# Stand-in text encoder
# ---------------------------------------------------------------------------
class _StubTextEncoder:
    """A deterministic 384-d encoder, so the serving tests need no MiniLM.

    `TextFeatureExtractor.encode` is the contract; this satisfies it without a
    ~21 s model load. The text encoder's own behaviour is not under test here.
    """

    def encode(self, texts: Any) -> Any:
        rows = list(texts)
        generator = np.random.default_rng(len(rows))
        vectors = generator.normal(size=(len(rows), TEXT_FEATURE_DIM)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors


def _serving_request(root: Path, query: str) -> Any:
    from core.schemas import AssetMetadata
    from specialists.base import SpecialistRequest

    scene = TRAIN_SCENES[0]
    return SpecialistRequest(
        assets=[
            AssetMetadata(path=str(root / "im1" / f"{scene}.png")),
            AssetMetadata(path=str(root / "im2" / f"{scene}.png")),
        ],
        query=query,
    )
