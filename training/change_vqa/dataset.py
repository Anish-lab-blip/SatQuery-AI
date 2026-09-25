"""SatQuery AI — the CDVQA dataset for change-VQA reasoning (R-02).

WHAT THIS ADDS OVER `training/data/cdvqa.py`
--------------------------------------------
The existing adapter is a **annotation-only** layer. Its own docstring is
explicit: *"No model code, no training loop, no text embeddings: this module
only prepares data."* It returns example dicts of paths and strings and never
opens an image. Measured 2026-09-21: zero image decoding, zero tensors, zero
label decoding, no reasoning path.

R-02 needs three things it does not provide, and this module adds exactly those:

    1. **pair alignment materialised as a typed record** — T1/T2 resolved, the
       temporal reference decoded from the question, and the question type
       resolved to a slot index;
    2. **label-derived supervision targets** — `label1`/`label2` ship with all
       2,968 scenes, so the class-wise change magnitude and signed delta are
       COMPUTABLE, not guessed. These are the targets of the change estimator;
    3. **leakage-safe split materialisation** — scene-disjoint train/val, and a
       hard refusal to pool Test with Test2 (they share all 968 scenes).

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
No tensors, no torch, no model. Loading pixels is `features.py`'s job. Keeping
this module import-light means the manifest, the statistics and the leakage
checks are testable without torch — the same discipline `core/planner.py` uses.

THE MANIFEST IS SCENE-LEVEL, AND THAT IS DELIBERATE
---------------------------------------------------
153,130 question rows would produce a ~50 MB JSONL that duplicates text already
authoritative in `annotations/*.json`. The manifest therefore records the
**2,968 scenes** (identity, integrity, split, question counts) and the
question-level records are derived from the annotations on demand through the
existing adapter. One source of truth for the text; one manifest for identity.

`SampleRecord` and `DatasetManifest` are REUSED from `evaluation/manifests.py`.
This is not a second manifest system: the split label is mapped onto the
existing `train|val|test` vocabulary and the native CDVQA split name is carried
in `extra["native_split"]`. Test2 maps to `test` because that is what it is —
but `by_native_split()` is the accessor, so Test and Test2 can never be pooled
by accident.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from core.errors import SatQueryError
from evaluation.manifests import DatasetManifest, SampleRecord
from training.change_vqa.vocab import (
    ANSWER_TO_INDEX,
    CHANGE_CLASS_ORDER,
    CLASS_TO_INDEX,
    N_CHANGE_CLASSES,
    QUESTION_TYPE_TO_INDEX,
    UNKNOWN_QUESTION_TYPE,
    UNKNOWN_QUESTION_TYPE_INDEX,
    answers_outside_vocabulary,
    resolve_change_class,
    resolve_question_type,
    resolve_temporal_reference,
)
from training.data.cdvqa import (
    CDVQAError,
    CDVQA_SPLITS,
    CDVQAImage,
    CDVQAQuestion,
    build_cdvqa_examples,
    load_cdvqa,
)

#: The dataset identity recorded in every manifest, artifact and evaluation.
DATASET_ID = "cdvqa"

#: Bumped when the record shape or the target definition changes. Recorded in
#: every evaluation so two numbers computed under different definitions can
#: never be silently compared.
PREPROCESSING_VERSION = "change_vqa_preproc_v1"

#: Native split -> manifest split. Test2 is a SECOND QUESTION SET over the SAME
#: 968 scenes as Test, not an independent sample; it must never be pooled.
SPLIT_MAP: dict[str, str] = {
    "Train": "train",
    "Val": "val",
    "Test": "test",
    "Test2": "test",
}

#: The scene-label palette, measured from the shipped maps. Agreement 1.0000
#: over all 2,968 scenes against `label1=pre` (artifacts/cdvqa/
#: temporal_order_evidence_v2.json). White is a SHARED BACKGROUND, not a class:
#: it covers ~80% of pixels and is not one of the six CDVQA change classes.
LABEL_PALETTE: dict[str, tuple[int, int, int]] = {
    "NVG_surface": (128, 128, 128),
    "trees": (0, 255, 0),
    "low_vegetation": (0, 128, 0),
    "water": (0, 0, 255),
    "buildings": (128, 0, 0),
    "playgrounds": (255, 0, 0),
}
BACKGROUND_RGB: tuple[int, int, int] = (255, 255, 255)


class ChangeVQAError(SatQueryError):
    """A dataset-level defect: a missing scene, an undecodable label, leakage."""

    code = "change_vqa_error"
    user_message = "The change-VQA dataset could not be prepared."


# ---------------------------------------------------------------------------
# Scene-level targets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneTargets:
    """Label-derived supervision for ONE scene, computed from label1/label2.

    Definitions (all fractions of total pixels, so scenes are comparable):

        class_mag[c]    (gained[c] + lost[c]) / total    in [0, 1]
        class_delta[c]  (gained[c] - lost[c]) / total    in [-1, 1]
        class_ratio[c]  (gained[c] + lost[c]) / count1[c]  per-class ratio
        total_changed   count(label1 != label2) / total  in [0, 1]

    where `gained[c] = |label2==c and label1!=c|` and
    `lost[c] = |label1==c and label2!=c|`.

    The `total_changed` definition was VALIDATED against the gold answers before
    being adopted: over 40 real `change_ratio` rows, `count(label1 != label2) /
    total` reproduces the annotated bin **34/40 = 85%** of the time. The
    competing definition (change among non-background pixels only) reproduced
    **1/40**. The residual 15% is annotator/map disagreement, which is the noise
    floor of the target — recorded here rather than smoothed away.
    """

    file_name: str
    scene_key: str
    class_mag: tuple[float, ...]
    class_delta: tuple[float, ...]
    class_ratio: tuple[float, ...]
    total_changed: float
    n_pixels: int
    n_labeled_pixels: int
    label1_sha256: str
    label2_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "scene_key": self.scene_key,
            "class_mag": list(self.class_mag),
            "class_delta": list(self.class_delta),
            "class_ratio": list(self.class_ratio),
            "total_changed": self.total_changed,
            "n_pixels": self.n_pixels,
            "n_labeled_pixels": self.n_labeled_pixels,
            "label1_sha256": self.label1_sha256,
            "label2_sha256": self.label2_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SceneTargets":
        return cls(
            file_name=payload["file_name"],
            scene_key=payload["scene_key"],
            class_mag=tuple(float(v) for v in payload["class_mag"]),
            class_delta=tuple(float(v) for v in payload["class_delta"]),
            class_ratio=tuple(float(v) for v in payload["class_ratio"]),
            total_changed=float(payload["total_changed"]),
            n_pixels=int(payload["n_pixels"]),
            n_labeled_pixels=int(payload["n_labeled_pixels"]),
            label1_sha256=payload["label1_sha256"],
            label2_sha256=payload["label2_sha256"],
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_label_map(path: str | Path) -> Any:
    """Decode a CDVQA scene-label PNG to an int8 class map.

    Returns:
        An `(H, W)` int8 array. Value `-1` is background (white); values
        `0..5` index `CHANGE_CLASS_ORDER`.

    Raises:
        ChangeVQAError: the image cannot be read, or it contains a colour that
            is neither a palette class nor the background. An unknown colour is
            a real defect — silently mapping it to background would delete
            changed pixels from the supervision and quietly weaken the target.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ChangeVQAError(f"numpy and Pillow are required: {exc}") from exc

    p = Path(path)
    if not p.exists():
        raise ChangeVQAError(f"label map not found: {p}")
    try:
        rgb = np.asarray(Image.open(p).convert("RGB"))
    except Exception as exc:  # noqa: BLE001
        raise ChangeVQAError(f"could not read label map {p}: {exc}") from exc

    out = np.full(rgb.shape[:2], -1, dtype=np.int8)
    for class_name, colour in LABEL_PALETTE.items():
        out[np.all(rgb == np.array(colour, dtype=rgb.dtype), axis=-1)] = (
            CLASS_TO_INDEX[class_name]
        )
    out[np.all(rgb == np.array(BACKGROUND_RGB, dtype=rgb.dtype), axis=-1)] = -1

    known = np.zeros(rgb.shape[:2], dtype=bool)
    for colour in list(LABEL_PALETTE.values()) + [BACKGROUND_RGB]:
        known |= np.all(rgb == np.array(colour, dtype=rgb.dtype), axis=-1)
    if not known.all():
        unknown_colours = np.unique(rgb[~known].reshape(-1, 3), axis=0)[:5]
        raise ChangeVQAError(
            f"label map {p.name} contains {int((~known).sum())} pixel(s) in "
            f"colours outside the frozen palette, e.g. "
            f"{[tuple(int(c) for c in row) for row in unknown_colours]}"
        )
    return out


def compute_scene_targets(
    label1_path: str | Path,
    label2_path: str | Path,
    file_name: str,
) -> SceneTargets:
    """Compute the class-wise change targets for one scene.

    `label1` is the PRE map and `label2` the POST map. That orientation is not
    assumed: it is the measured verdict `label1=pre,label2=post` of
    `artifacts/cdvqa/temporal_order_evidence_v2.json` (agreement 1.0000 over
    all 2,968 scenes; the reverse hypothesis scores 0.3588 with zero
    directional hits). The image-level leg (`im1`=pre, `im2`=post) is
    SUPPORTED but not proven, and the distinction is carried into every
    artifact this module produces.
    """
    import numpy as np

    a = decode_label_map(label1_path)
    b = decode_label_map(label2_path)
    if a.shape != b.shape:
        raise ChangeVQAError(
            f"label maps disagree in shape for {file_name}: "
            f"{tuple(a.shape)} vs {tuple(b.shape)}"
        )

    total = int(a.size)
    if total == 0:
        raise ChangeVQAError(f"label map for {file_name} is empty")

    mag: list[float] = []
    delta: list[float] = []
    ratio: list[float] = []
    for index in range(N_CHANGE_CLASSES):
        in1 = a == index
        in2 = b == index
        gained = int(np.logical_and(in2, ~in1).sum())
        lost = int(np.logical_and(in1, ~in2).sum())
        count1 = int(in1.sum())
        mag.append((gained + lost) / total)
        delta.append((gained - lost) / total)
        # A class absent from the PRE map has no denominator. 0.0 is the honest
        # value: nothing of that class could have changed out of it, and a
        # divide-by-zero would put NaN into the training target.
        ratio.append((gained + lost) / count1 if count1 else 0.0)

    return SceneTargets(
        file_name=file_name,
        scene_key=Path(file_name).stem,
        class_mag=tuple(mag),
        class_delta=tuple(delta),
        class_ratio=tuple(ratio),
        total_changed=float((a != b).mean()),
        n_pixels=total,
        n_labeled_pixels=int((a != -1).sum() + (b != -1).sum()),
        label1_sha256=_sha256_file(Path(label1_path)),
        label2_sha256=_sha256_file(Path(label2_path)),
    )


def compute_scene_targets_cached(
    label_dir: str | Path,
    file_name: str,
    cache: dict[str, SceneTargets] | None = None,
) -> SceneTargets:
    """`compute_scene_targets` with an in-process memo keyed on the file name."""
    if cache is not None and file_name in cache:
        return cache[file_name]
    base = Path(label_dir)
    targets = compute_scene_targets(
        base / "label1" / file_name, base / "label2" / file_name, file_name
    )
    if cache is not None:
        cache[file_name] = targets
    return targets


# ---------------------------------------------------------------------------
# Question-level records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChangeVQARecord:
    """One (scene, question, answer) training/evaluation unit."""

    split: str          # native CDVQA split: Train | Val | Test | Test2
    question_id: int
    file_name: str
    scene_key: str
    qtype: str
    qtype_index: int
    resolved_qtype: str
    qtype_resolved: bool
    temporal_ref: str
    change_class: str | None
    question: str
    answer: str
    answer_index: int
    t1_path: str
    t2_path: str

    @property
    def is_answer_legal_for_type(self) -> bool:
        from training.change_vqa.vocab import LEGAL_ANSWERS

        legal = LEGAL_ANSWERS.get(self.qtype)
        return True if legal is None else self.answer in legal

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "question_id": self.question_id,
            "file_name": self.file_name,
            "scene_key": self.scene_key,
            "qtype": self.qtype,
            "qtype_index": self.qtype_index,
            "resolved_qtype": self.resolved_qtype,
            "qtype_resolved": self.qtype_resolved,
            "temporal_ref": self.temporal_ref,
            "change_class": self.change_class,
            "question": self.question,
            "answer": self.answer,
            "answer_index": self.answer_index,
            "t1_path": self.t1_path,
            "t2_path": self.t2_path,
        }


def _record_from_example(example: dict[str, Any]) -> ChangeVQARecord:
    """Turn one `build_cdvqa_examples` dict into a typed record.

    The example dict is the existing adapter's contract; this function adds no
    interpretation of its own beyond indexing the frozen vocabulary.

    The adapter emits exactly these keys (see
    `training.data.cdvqa.build_cdvqa_examples`):

        image_path, image_paths, layout, file_name, scene_key, split, qtype,
        question, answer, question_id

    There is deliberately **no** image row id: the adapter deduplicates scenes
    and routes each question to its scene through the unioned `question_ids`, so
    `question_id` already determines the scene and `scene_key` names it. A field
    read here that the adapter does not emit would be an invented column.
    """
    qtype = str(example["qtype"])
    answer = str(example["answer"])
    if answer not in ANSWER_TO_INDEX:
        raise ChangeVQAError(
            f"answer {answer!r} (question {example['question_id']}) is outside "
            f"the frozen 19-answer vocabulary; the label space would silently "
            f"shift if it were admitted"
        )
    resolved_qtype, resolved = resolve_question_type(str(example["question"]))
    paths = list(example["image_paths"])
    if len(paths) != 2:
        raise ChangeVQAError(
            f"expected exactly two temporal image paths, got {len(paths)} for "
            f"scene {example['file_name']}; the imagery layout did not resolve "
            f"to SECOND's im1/im2 pair layout, so this is a single-image scene "
            f"and cannot support a change question"
        )
    return ChangeVQARecord(
        split=str(example["split"]),
        question_id=int(example["question_id"]),
        file_name=str(example["file_name"]),
        scene_key=str(example["scene_key"]),
        qtype=qtype,
        qtype_index=QUESTION_TYPE_TO_INDEX.get(
            qtype, UNKNOWN_QUESTION_TYPE_INDEX
        ),
        resolved_qtype=resolved_qtype,
        qtype_resolved=resolved,
        temporal_ref=resolve_temporal_reference(str(example["question"])),
        change_class=resolve_change_class(str(example["question"])),
        question=str(example["question"]),
        answer=answer,
        answer_index=ANSWER_TO_INDEX[answer],
        # T1 is the PRE acquisition and T2 the POST one. See the module
        # docstring and `compute_scene_targets` for the strength of that claim.
        t1_path=str(paths[0]),
        t2_path=str(paths[1]),
    )


def load_change_vqa_records(
    root: str | Path,
    *,
    splits: Sequence[str] = CDVQA_SPLITS,
    image_dir: str | Path | None = None,
    require_images: bool = False,
) -> list[ChangeVQARecord]:
    """Every resolvable (scene, question, answer) unit for `splits`.

    Delegates parsing to the existing adapter so the annotation format is
    interpreted in exactly one place — but calls it **once per split**, which is
    not a style choice. See `_examples_for_one_split`.

    Returns:
        Records in `splits` order, each split internally in the adapter's
        deterministic order.
    """
    base = Path(root)
    resolved_image_dir = str(image_dir) if image_dir is not None else str(base)
    out: list[ChangeVQARecord] = []
    for split in splits:
        for example in _examples_for_one_split(
            base, split, resolved_image_dir, require_images
        ):
            out.append(_record_from_example(example))
    return out


def _examples_for_one_split(
    base: Path, split: str, image_dir: str, require_images: bool
) -> list[dict[str, Any]]:
    """Run the adapter for ONE native split and return its example dicts.

    WHY THIS IS PER-SPLIT AND NOT ONE CALL OVER ALL SPLITS
    ------------------------------------------------------
    Historically: a defect. Measured 2026-09-21, on this corpus, with the
    shipped annotations:

        `CDVQAQuestion.question_id` is **per-split sequential** — each split
        restarts at 0 (Train 0..65966, Val 0..16440, Test 0..39685,
        Test2 0..31035; all four verified, 0 gaps).

        `build_cdvqa_examples` built its question->scene index as

            question_to_image: dict[int, CDVQAImage] = {}
            for image in images:
                for question_id in image.question_ids:
                    question_to_image[question_id] = image

        over the *concatenation* of the requested splits. Because the ids
        restart, later splits overwrote earlier ones and the last writer won.

    The consequence was not a crash but a silent mis-pairing:

        splits=("Train",)          -> 65,967 questions / 1,600 scenes, 0 wrong
        splits=("Val",)            -> 16,441 questions /   400 scenes, 0 wrong
        splits=("Train","Val")     -> 82,408 questions / **1,601** scenes
                                      (the correct figure is 2,000)
        and 32,882 of those questions were routed to imagery belonging to the
        *other* split.

    **That defect is now FIXED at its root**: `build_cdvqa_examples` keys the
    index on `(split, question_id)`, so a multi-split call pairs every question
    with its own split's imagery. The fix is pinned by
    `tests/unit/test_change_vqa_dataset.py` (the multi-split counts) and by
    `tests/unit/test_cdvqa_adapter.py` (a synthetic two-split collision).

    This function is RETAINED rather than collapsed into one call, for two
    reasons that are still true and still worth having:

    * it is the equivalence check. A test asserts that one call over all splits
      now produces exactly what this per-split loop produces; if the adapter
      ever regresses, the two disagree and the test fails. That is a stronger
      guard than either alone.
    * `require_images` is evaluated per split, so a split whose imagery is
      missing is named by the split that is missing it.

    Within one split the adapter is exact, and this is verified rather than
    assumed: `sum(len(image.question_ids))` equals the split's question count
    (65,967 / 16,441 / 39,686 / 31,036 — all exact), and no question id is
    claimed by more than one scene in any split (0 contested, all four).
    """
    corpus = load_cdvqa(base, splits=(split,), require_images=require_images)
    return build_cdvqa_examples(
        corpus.images, corpus.questions, image_dir=image_dir
    )


def group_by_native_split(
    records: Iterable[ChangeVQARecord],
) -> dict[str, list[ChangeVQARecord]]:
    """Group records by the NATIVE split name, never by the mapped one."""
    out: dict[str, list[ChangeVQARecord]] = {name: [] for name in CDVQA_SPLITS}
    for record in records:
        out.setdefault(record.split, []).append(record)
    return out


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_scene_manifest(
    root: str | Path,
    *,
    splits: Sequence[str] = CDVQA_SPLITS,
    require_images: bool = True,
) -> DatasetManifest:
    """Build the scene-level manifest for the CDVQA corpus.

    One `SampleRecord` per **(native split, scene)** pair — 3,936 records across
    all four splits, covering 2,968 distinct scenes. The two figures differ
    because Test and Test2 are the same 968 scenes asked about twice, and the
    manifest keeps them separate so `by_native_split` can address each.

    Each record carries the two image hashes, the scene's question count and the
    native split name. `scene_id` is the file-name stem — the dataset ships no
    coarser grouping, so the scene IS the image.

    The per-scene question count comes from `CDVQAImage.question_ids`, which is
    this split's own question index. It is NOT derived from
    `(question.split, question.image_id)`: a scene occupies ~16 image *rows*
    while `image_id` keeps only the first row's id, so that join counts the
    questions of one row and undercounts every scene by roughly a factor of 16.
    `sum(len(image.question_ids))` is verified equal to the split's question
    count (65,967 / 16,441 / 39,686 / 31,036), so this index is complete.
    """
    base = Path(root)
    corpus = load_cdvqa(base, splits=tuple(splits), require_images=require_images)

    records: list[SampleRecord] = []
    for image in corpus.images:
        t1 = Path(image.image_paths(base)[0])
        t2 = Path(image.image_paths(base)[1])
        if require_images and not (t1.exists() and t2.exists()):
            raise ChangeVQAError(
                f"scene {image.file_name} is declared but its imagery is "
                f"missing ({t1.name} / {t2.name}); refusing to freeze a "
                f"manifest over absent data"
            )
        records.append(
            SampleRecord(
                dataset_id=DATASET_ID,
                sample_id=f"{image.split}:{image.scene_key}",
                split=SPLIT_MAP[image.split],
                scene_id=image.scene_key,
                source_scene=image.file_name,
                sensor="second",
                sha256=_sha256_file(t1) if t1.exists() else None,
                path=str(t1),
                extra={
                    "native_split": image.split,
                    "file_name": image.file_name,
                    "t1_path": str(t1),
                    "t2_path": str(t2),
                    "t2_sha256": _sha256_file(t2) if t2.exists() else None,
                    "n_questions": len(image.question_ids),
                    "preprocessing_version": PREPROCESSING_VERSION,
                    "temporal_order_evidence": "label1=pre,label2=post (proven); "
                    "im1=pre,im2=post (statistical)",
                },
            )
        )

    records.sort(key=lambda r: (r.extra["native_split"], r.scene_id or ""))
    return DatasetManifest(
        name="cdvqa_scenes",
        records=records,
        notes={
            "dataset_id": DATASET_ID,
            "preprocessing_version": PREPROCESSING_VERSION,
            "native_splits": list(splits),
            "split_map": SPLIT_MAP,
            "label_palette": {k: list(v) for k, v in LABEL_PALETTE.items()},
            "background_rgb": list(BACKGROUND_RGB),
        },
    )


def by_native_split(
    manifest: DatasetManifest, native_split: str
) -> list[SampleRecord]:
    """The scenes of one NATIVE split. The only safe split accessor.

    `DatasetManifest.by_split("test")` returns Test **and** Test2 pooled, which
    is a leakage hazard for this dataset: both native splits cover the same 968
    scenes. Use this accessor instead.
    """
    if native_split not in SPLIT_MAP:
        raise ChangeVQAError(
            f"unknown native split {native_split!r}; known: {list(SPLIT_MAP)}"
        )
    return [
        record
        for record in manifest.records
        if record.extra.get("native_split") == native_split
    ]


def leakage_report(manifest: DatasetManifest) -> dict[str, Any]:
    """Scene-level overlap between every pair of native splits.

    Reports rather than judges: the Test/Test2 overlap is a PROPERTY of the
    dataset, not a defect in it. `assert_no_leakage` is what turns the report
    into a gate.
    """
    scenes = {
        name: {r.scene_id for r in by_native_split(manifest, name)}
        for name in SPLIT_MAP
    }
    overlap: dict[str, int] = {}
    for i, a in enumerate(SPLIT_MAP):
        for b in list(SPLIT_MAP)[i + 1:]:
            overlap[f"{a}|{b}"] = len(scenes[a] & scenes[b])
    return {
        "scenes_per_native_split": {k: len(v) for k, v in scenes.items()},
        "pairwise_scene_overlap": overlap,
        "train_val_disjoint": overlap["Train|Val"] == 0,
        "train_test_disjoint": overlap["Train|Test"] == 0,
        "val_test_disjoint": overlap["Val|Test"] == 0,
        "test_test2_share_all_scenes": (
            overlap["Test|Test2"] == len(scenes["Test"]) == len(scenes["Test2"])
        ),
    }


def assert_no_leakage(manifest: DatasetManifest) -> None:
    """Raise unless the splits a model is trained and scored on are disjoint.

    The gate is deliberately narrower than "no overlap anywhere": Test and
    Test2 DO share all their scenes by construction, and a check that failed on
    that would be a check nobody could pass. What must hold is that nothing a
    model is fitted on touches anything it is scored on.

    Raises:
        ChangeVQAError: Train overlaps Val, Train overlaps Test, or Val
            overlaps Test.
    """
    report = leakage_report(manifest)
    bad = [
        key
        for key in ("Train|Val", "Train|Test", "Val|Test")
        if report["pairwise_scene_overlap"][key] != 0
    ]
    if bad:
        raise ChangeVQAError(
            "scene-level leakage between splits: "
            + ", ".join(
                f"{key}={report['pairwise_scene_overlap'][key]}" for key in bad
            )
        )


# ---------------------------------------------------------------------------
# Split integrity — the gate that catches a mis-paired question
# ---------------------------------------------------------------------------

#: Ground truth for this corpus revision, measured 2026-09-21 by reading the
#: twelve annotation files **directly** rather than through any adapter:
#: `(unique_file_names, resolved_questions)` per native split.
#:
#: These numbers are the arbiter. When a pipeline disagrees with them, the
#: pipeline is what is wrong.
EXPECTED_SPLIT_SIZES: dict[str, tuple[int, int]] = {
    "Train": (1600, 65967),
    "Val": (400, 16441),
    "Test": (968, 39686),
    "Test2": (968, 31036),
}

EXPECTED_TOTAL_QUESTIONS = sum(q for _s, q in EXPECTED_SPLIT_SIZES.values())

#: Distinct scenes across the corpus — the union over splits. Test and Test2
#: cover the SAME 968 scenes, so this is less than the sum below.
EXPECTED_TOTAL_SCENES = 2968

#: Records in a full scene manifest: one per (native split, scene) pair. This is
#: deliberately NOT `EXPECTED_TOTAL_SCENES` — a manifest must keep Test and Test2
#: as separate records so `by_native_split` can address each, and those two
#: splits duplicate 968 scenes. Collapsing them to 2,968 would silently destroy
#: the Test/Test2 distinction the evaluation depends on.
EXPECTED_MANIFEST_RECORDS = sum(s for s, _q in EXPECTED_SPLIT_SIZES.values())

#: Native splits that are allowed to share scenes. Test2 is a second question
#: set over Test's scenes — a property of the release, not a defect.
_ALLOWED_SHARED_PAIRS: frozenset[frozenset[str]] = frozenset(
    {frozenset({"Test", "Test2"})}
)


def verify_split_integrity(
    records: Sequence[ChangeVQARecord],
    *,
    expect_full_split: bool = True,
    expected_splits: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Check `records` against the corpus's measured ground truth.

    WHY THIS EXISTS
    ---------------
    A question paired with the wrong scene does not crash. It trains, it
    evaluates, and it reports a number. The cross-split question-id collision in
    `build_cdvqa_examples` produced exactly that (see
    `_examples_for_one_split`), and the only reason it was caught is that the
    per-split counts came out wrong — 1,601 scenes where 2,000 were expected.

    So the counts are checked explicitly, and so is split disjointness at the
    RECORD level. The manifest-level `assert_no_leakage` cannot see this defect:
    the manifest is built from the images table, which is correct per split,
    while the mis-pairing happens only in the examples. This is the gate that
    sees it.

    Args:
        records: the records a run is about to use.
        expect_full_split: when True (the default), require each expected split's
            record and scene counts to equal `EXPECTED_SPLIT_SIZES` exactly. Set
            False for a deliberate subsample, which still gets the disjointness
            and internal-consistency checks.
        expected_splits: which native splits the caller asked for. Defaults to
            all four. A caller that legitimately loads only Train and Val must
            say so, otherwise the two absent splits would be reported as size
            errors — which is a false alarm, and a gate that cries wolf is a
            gate people learn to bypass. Records belonging to a split OUTSIDE
            this set are always an error.

    Returns:
        The report, whether or not it is clean. Inspect it; do not assume it.
    """
    expected = tuple(
        expected_splits if expected_splits is not None
        else tuple(EXPECTED_SPLIT_SIZES)
    )
    unknown = [name for name in expected if name not in EXPECTED_SPLIT_SIZES]
    if unknown:
        raise ChangeVQAError(
            f"unknown expected split(s) {unknown}; known: "
            f"{list(EXPECTED_SPLIT_SIZES)}"
        )

    per_split: dict[str, list[ChangeVQARecord]] = {}
    for record in records:
        per_split.setdefault(record.split, []).append(record)

    scenes = {name: {r.scene_key for r in rows} for name, rows in per_split.items()}
    counts = {name: len(rows) for name, rows in per_split.items()}

    size_errors: list[str] = []
    unexpected = sorted(set(per_split) - set(expected))
    if unexpected:
        size_errors.append(
            f"records present for split(s) {unexpected} which the caller did not "
            f"request {list(expected)}; the splits a run may read must be a "
            f"deliberate choice"
        )

    for name in expected:
        exp_scenes, exp_questions = EXPECTED_SPLIT_SIZES[name]
        got_scenes = len(scenes.get(name, ()))
        got_questions = counts.get(name, 0)
        if not expect_full_split:
            if got_scenes > exp_scenes or got_questions > exp_questions:
                size_errors.append(
                    f"{name}: {got_scenes} scenes / {got_questions} questions "
                    f"EXCEEDS the corpus's {exp_scenes}/{exp_questions}"
                )
            continue
        if (got_scenes, got_questions) != (exp_scenes, exp_questions):
            size_errors.append(
                f"{name}: got {got_scenes} scenes / {got_questions} questions, "
                f"expected {exp_scenes}/{exp_questions}"
            )

    # Disjointness is checked over the splits actually present, so a two-split
    # run still gets the check that catches the collision.
    overlap: dict[str, int] = {}
    leakage: list[str] = []
    present = sorted(per_split)
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            shared = len(scenes[a] & scenes[b])
            overlap[f"{a}|{b}"] = shared
            if shared and frozenset({a, b}) not in _ALLOWED_SHARED_PAIRS:
                leakage.append(f"{a}|{b}={shared}")

    # Every record must name a scene the corpus actually ships, and the two
    # temporal paths must belong to that scene.
    inconsistent: list[str] = []
    for record in records:
        if Path(record.t1_path).stem != record.scene_key or (
            Path(record.t2_path).stem != record.scene_key
        ):
            inconsistent.append(
                f"q{record.question_id} ({record.split}) names scene "
                f"{record.scene_key!r} but its imagery is "
                f"{Path(record.t1_path).name}/{Path(record.t2_path).name}"
            )

    report: dict[str, Any] = {
        "n_records": len(records),
        "expected_splits": list(expected),
        "questions_per_split": counts,
        "scenes_per_split": {k: len(v) for k, v in scenes.items()},
        "expected_split_sizes": {
            k: {"scenes": s, "questions": q}
            for k, (s, q) in EXPECTED_SPLIT_SIZES.items()
        },
        "pairwise_scene_overlap": overlap,
        "allowed_shared_pairs": [sorted(p) for p in _ALLOWED_SHARED_PAIRS],
        "unexpected_splits": unexpected,
        "size_errors": size_errors,
        "leakage": leakage,
        "records_with_inconsistent_imagery": len(inconsistent),
        "inconsistent_examples": inconsistent[:5],
        "is_clean": not (size_errors or leakage or inconsistent),
        "expect_full_split": expect_full_split,
    }
    return report


def assert_split_integrity(
    records: Sequence[ChangeVQARecord],
    *,
    expect_full_split: bool = True,
    expected_splits: Sequence[str] | None = None,
) -> dict[str, Any]:
    """`verify_split_integrity`, but raise on any defect.

    Raises:
        ChangeVQAError: sizes are wrong, splits leak into each other, a split was
            read that was not requested, or a record's imagery does not belong to
            the scene it names.
    """
    report = verify_split_integrity(
        records,
        expect_full_split=expect_full_split,
        expected_splits=expected_splits,
    )
    problems = report["size_errors"] + report["leakage"]
    if report["records_with_inconsistent_imagery"]:
        problems.append(
            f"{report['records_with_inconsistent_imagery']} record(s) whose "
            f"imagery does not belong to the scene they name"
        )
    if problems:
        raise ChangeVQAError(
            "CDVQA split integrity check failed — the (scene, question, answer) "
            "units are not trustworthy, so nothing downstream of them is "
            "either: " + "; ".join(problems)
        )
    return report


def dataset_statistics(
    records: Sequence[ChangeVQARecord],
    *,
    targets: dict[str, SceneTargets] | None = None,
) -> dict[str, Any]:
    """Everything the training log must print about the data it is about to use.

    Returns counts, the per-split and per-type distributions, the answer
    vocabulary actually observed, the number of unique scenes and pairs, and
    any answer outside the frozen vocabulary. No image is opened.
    """
    from collections import Counter

    per_split = Counter(r.split for r in records)
    per_type = Counter(r.qtype for r in records)
    per_type_split: dict[str, dict[str, int]] = {}
    for record in records:
        per_type_split.setdefault(record.split, {}).setdefault(record.qtype, 0)
        per_type_split[record.split][record.qtype] += 1

    answers = Counter(r.answer for r in records)
    scenes = {r.file_name for r in records}
    pairs = {(r.file_name, r.split) for r in records}
    resolver_agree = sum(
        1 for r in records if r.qtype_resolved and r.resolved_qtype == r.qtype
    )
    illegal = sum(1 for r in records if not r.is_answer_legal_for_type)

    stats: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "preprocessing_version": PREPROCESSING_VERSION,
        "n_records": len(records),
        "n_unique_scenes": len(scenes),
        "n_unique_pairs": len(pairs),
        "records_per_native_split": dict(sorted(per_split.items())),
        "scenes_per_native_split": {
            split: len({r.file_name for r in records if r.split == split})
            for split in sorted(per_split)
        },
        "question_type_counts": dict(sorted(per_type.items())),
        "question_type_counts_per_split": per_type_split,
        "answer_counts": dict(answers.most_common()),
        "n_distinct_answers": len(answers),
        "answers_outside_frozen_vocabulary": answers_outside_vocabulary(
            list(answers)
        ),
        "answers_illegal_for_their_type": illegal,
        "type_resolver_agreement": resolver_agree,
        "type_resolver_agreement_rate": (
            round(resolver_agree / len(records), 4) if records else 0.0
        ),
    }
    if targets is not None:
        stats["scene_targets"] = {
            "n_scenes_with_targets": len(targets),
            "mean_total_changed": round(
                sum(t.total_changed for t in targets.values()) / len(targets), 6
            )
            if targets
            else 0.0,
            "class_order": list(CHANGE_CLASS_ORDER),
            "mean_class_mag": [
                round(
                    sum(t.class_mag[i] for t in targets.values()) / len(targets), 6
                )
                for i in range(N_CHANGE_CLASSES)
            ]
            if targets
            else [],
        }
    return stats


def write_manifest(manifest: DatasetManifest, path: str | Path) -> Path:
    """Write the scene manifest as JSONL (header line + one record per line)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(manifest.to_jsonl(), encoding="utf-8")
    return p


def read_manifest(path: str | Path) -> DatasetManifest:
    return DatasetManifest.read(path)


def write_scene_targets(
    targets: Iterable[SceneTargets], path: str | Path
) -> Path:
    """Write the label-derived targets as JSONL, one scene per line."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(t.to_dict(), sort_keys=True) for t in targets
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def read_scene_targets(path: str | Path) -> dict[str, SceneTargets]:
    """Read a targets file into a `file_name -> SceneTargets` map."""
    p = Path(path)
    if not p.exists():
        raise ChangeVQAError(f"scene targets file not found: {p}")
    out: dict[str, SceneTargets] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        targets = SceneTargets.from_dict(payload)
        out[targets.file_name] = targets
    return out


__all__ = [
    "DATASET_ID",
    "PREPROCESSING_VERSION",
    "SPLIT_MAP",
    "LABEL_PALETTE",
    "BACKGROUND_RGB",
    "ChangeVQAError",
    "SceneTargets",
    "ChangeVQARecord",
    "decode_label_map",
    "compute_scene_targets",
    "compute_scene_targets_cached",
    "load_change_vqa_records",
    "group_by_native_split",
    "build_scene_manifest",
    "by_native_split",
    "leakage_report",
    "assert_no_leakage",
    "EXPECTED_SPLIT_SIZES",
    "EXPECTED_TOTAL_QUESTIONS",
    "EXPECTED_TOTAL_SCENES",
    "EXPECTED_MANIFEST_RECORDS",
    "verify_split_integrity",
    "assert_split_integrity",
    "dataset_statistics",
    "write_manifest",
    "read_manifest",
    "write_scene_targets",
    "read_scene_targets",
    "CDVQAError",
    "CDVQAImage",
    "CDVQAQuestion",
]
