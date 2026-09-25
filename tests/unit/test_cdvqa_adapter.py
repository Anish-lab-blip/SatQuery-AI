"""Tests for the CDVQA reader, join and example construction.

Everything runs against synthetic annotation trees built in a temp dir -- three
JSON tables per split (``images`` / ``questions`` / ``answers``). No dataset
download, no network, no GPU, and no dependency on the real corpus being
present: the tests must pass on a clean checkout.

The fixtures reproduce the three facts the module exists to handle:

  * ``images`` carries ONE ROW PER QUESTION -- the same ``file_name`` repeats,
    once per question asked about it, each row with its own ``id`` and its own
    ``questions_ids``. Deduplicating on ``file_name`` is mandatory.
  * ``date_added`` is a FLOAT (``1631214265.2322025``), not a string.
  * ``res_x`` / ``res_y`` are the strings ``".1524m"``.

The structural tests that matter most:

  * the dedup trap: N rows, one scene, the union of all N question ids
  * the drop counters are four genuinely distinct reasons and cannot be
    reconstructed from ``summarize``'s single proxy number
  * Test/Test2 sharing images is reported, because pooling them inflates
    confidence in a way no per-example metric reveals
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.errors import SatQueryError
from training.data.cdvqa import (
    ANSWER_VOCABULARIES,
    CHANGE_CLASSES,
    CHANGE_RATIO_BINS,
    CHANGE_RATIO_TYPE_VALUES,
    CDVQA_SPLITS,
    LAYOUT_AUTO,
    LAYOUT_FLAT,
    LAYOUT_SECOND_PAIRS,
    QUESTION_TYPES,
    YES_NO_VALUES,
    CDVQACorpus,
    CDVQAError,
    CDVQAImage,
    CDVQAQuestion,
    _as_float,
    _as_int,
    build_cdvqa_examples,
    detect_layout,
    discover_cdvqa,
    discover_questions,
    load_cdvqa,
    summarize,
    warnings_for,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The exact key set every example must carry -- no more, no less.
EXAMPLE_KEYS = {
    "image_path",
    "image_paths",
    "layout",
    "file_name",
    "scene_key",
    "split",
    "qtype",
    "question",
    "answer",
    "question_id",
}

# ---------------------------------------------------------------------------
# Synthetic annotation tree
# ---------------------------------------------------------------------------


def image_row(
    row_id: int,
    file_name: str,
    questions_ids: list[int],
    *,
    res_x: str = ".1524m",
    res_y: str = ".1524m",
    active: bool = True,
) -> dict:
    """One ``images`` row -- note it is per QUESTION, not per image."""
    return {
        "id": row_id,
        "res_x": res_x,
        "res_y": res_y,
        "questions_ids": list(questions_ids),
        "file_name": file_name,
        "active": active,
    }


def question_row(
    row_id: int,
    img_id: int,
    qtype: str,
    answers_ids: list[int],
    *,
    question: str = "Did something change?",
    date_added: float | str | None = 1631214265.2322025,
    active: bool = True,
) -> dict:
    """One ``questions`` row -- ``date_added`` is a FLOAT in the real corpus."""
    return {
        "id": row_id,
        "date_added": date_added,
        "img_id": img_id,
        "type": qtype,
        "question": question,
        "answers_ids": list(answers_ids),
        "active": active,
    }


def answer_row(
    row_id: int,
    question_id: int,
    answer: str,
    *,
    date_added: float | str | None = 1631214265.2322068,
    active: bool = True,
) -> dict:
    return {
        "id": row_id,
        "date_added": date_added,
        "question_id": question_id,
        "answer": answer,
        "active": active,
    }


def write_table(root: Path, split: str, key: str, rows: list[dict]) -> Path:
    """Write ``{root}/annotations/{split}_{key}.json`` with one top-level key."""
    annotations = root / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    path = annotations / f"{split}_{key}.json"
    path.write_text(json.dumps({key: rows}), encoding="utf-8")
    return path


def write_split(
    root: Path,
    split: str,
    *,
    images: list[dict] | None = None,
    questions: list[dict] | None = None,
    answers: list[dict] | None = None,
) -> None:
    write_table(root, split, "images", images or [])
    write_table(root, split, "questions", questions or [])
    write_table(root, split, "answers", answers or [])


@pytest.fixture()
def cdvqa_root(tmp_path: Path) -> Path:
    """A small, fully-resolved two-split corpus in the real schema shape."""
    root = tmp_path / "cdvqa"

    write_split(
        root,
        "Train",
        # One file_name, three per-question rows, distinct question ids.
        images=[
            image_row(0, "07308.png", [0, 1]),
            image_row(1, "07308.png", [2]),
            image_row(2, "07308.png", [3]),
            image_row(3, "10589.png", [4]),
        ],
        questions=[
            question_row(0, 0, "change_or_not", [0]),
            question_row(1, 0, "change_ratio", [1]),
            question_row(2, 1, "change_to_what", [2]),
            question_row(3, 2, "increase_or_not", [3]),
            question_row(4, 3, "decrease_or_not", [4]),
        ],
        answers=[
            answer_row(0, 0, "yes"),
            answer_row(1, 1, "0_to_10"),
            answer_row(2, 2, "buildings"),
            answer_row(3, 3, "no"),
            answer_row(4, 4, "yes"),
        ],
    )
    return root


@pytest.fixture()
def cdvqa_paired_root(tmp_path: Path) -> Path:
    """A corpus whose imagery is SECOND's PAIRED layout.

    ``im1/`` and ``im2/`` each hold the *same* basenames -- the shared basename
    is exactly why a flat one-path-per-``file_name`` layout cannot represent the
    scene. The PNGs carry real magic bytes so ``is_file()`` semantics are
    exercised without needing a decoder.
    """
    root = tmp_path / "cdvqa_paired"
    write_split(
        root,
        "Train",
        images=[image_row(0, "00003.png", [0, 1]), image_row(1, "00011.png", [2])],
        questions=[
            question_row(0, 0, "change_or_not", [0]),
            question_row(1, 0, "change_to_what", [1]),
            question_row(2, 1, "increase_or_not", [2]),
        ],
        answers=[
            answer_row(0, 0, "yes"),
            answer_row(1, 1, "buildings"),
            answer_row(2, 2, "no"),
        ],
    )
    for directory in ("im1", "im2"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for name in ("00003.png", "00011.png"):
        (root / "im1" / name).write_bytes(b"\x89PNG\r\n\x1a\n im1 " + name.encode())
        (root / "im2" / name).write_bytes(b"\x89PNG\r\n\x1a\n im2 " + name.encode())
    return root


@pytest.fixture()
def cdvqa_flat_root(tmp_path: Path) -> Path:
    """A corpus whose imagery is the FLAT layout: one PNG directly under root."""
    root = tmp_path / "cdvqa_flat"
    write_split(
        root,
        "Train",
        images=[image_row(0, "00003.png", [0])],
        questions=[question_row(0, 0, "change_or_not", [0])],
        answers=[answer_row(0, 0, "yes")],
    )
    (root / "00003.png").write_bytes(b"\x89PNG\r\n\x1a\n flat")
    return root


# ---------------------------------------------------------------------------
# Discovery and the dedup trap
# ---------------------------------------------------------------------------


def test_one_file_name_appearing_as_n_rows_becomes_one_image(
    cdvqa_root: Path,
) -> None:
    """The whole point: the images table is per question, not per image."""
    images = discover_cdvqa(cdvqa_root, splits=("Train",))
    by_name = {img.file_name: img for img in images}
    assert len(images) == 2
    assert set(by_name) == {"07308.png", "10589.png"}


def test_deduped_image_carries_the_union_of_question_ids(
    cdvqa_root: Path,
) -> None:
    """N rows with N question ids -> ONE image whose question_ids is the union."""
    image = next(
        i for i in discover_cdvqa(cdvqa_root, splits=("Train",))
        if i.file_name == "07308.png"
    )
    assert image.question_ids == (0, 1, 2, 3)


def test_dedup_unions_overlapping_question_ids(tmp_path: Path) -> None:
    """A repeated id across rows must not double-count."""
    root = tmp_path / "cdvqa"
    write_split(
        root,
        "Train",
        images=[
            image_row(0, "a.png", [10, 11]),
            image_row(1, "a.png", [11, 12]),
            image_row(2, "a.png", [12, 13]),
        ],
    )
    images = discover_cdvqa(root, splits=("Train",))
    assert len(images) == 1
    assert images[0].question_ids == (10, 11, 12, 13)


def test_deduped_image_keeps_the_first_rows_identity(tmp_path: Path) -> None:
    root = tmp_path / "cdvqa"
    write_split(
        root,
        "Train",
        images=[
            image_row(7, "a.png", [1], res_x=".10m", res_y=".20m"),
            image_row(8, "a.png", [2], res_x=".99m", res_y=".99m"),
        ],
    )
    image = discover_cdvqa(root, splits=("Train",))[0]
    assert image.image_id == 7
    assert image.res_x == ".10m"
    assert image.res_y == ".20m"


def test_discovery_returns_one_record_per_distinct_file_name_per_split(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cdvqa"
    write_split(
        root,
        "Train",
        images=[
            image_row(0, "a.png", [0]),
            image_row(1, "a.png", [1]),
            image_row(2, "b.png", [2]),
        ],
    )
    write_split(
        root,
        "Val",
        images=[image_row(0, "c.png", [0]), image_row(1, "c.png", [1])],
    )
    images = discover_cdvqa(root, splits=("Train", "Val"))
    per_split: dict[str, list[str]] = {}
    for image in images:
        per_split.setdefault(image.split, []).append(image.file_name)
    assert sorted(per_split["Train"]) == ["a.png", "b.png"]
    assert per_split["Val"] == ["c.png"]


def test_discovery_is_sorted_by_split_then_file_name(cdvqa_root: Path) -> None:
    images = discover_cdvqa(cdvqa_root, splits=("Train",))
    names = [img.file_name for img in images]
    assert names == sorted(names)


def test_scene_key_is_the_file_name_stem(cdvqa_root: Path) -> None:
    images = discover_cdvqa(cdvqa_root, splits=("Train",))
    assert all(img.scene_key == Path(img.file_name).stem for img in images)
    assert {img.scene_key for img in images} == {"07308", "10589"}


def test_image_path_without_image_dir_is_the_bare_file_name(
    cdvqa_root: Path,
) -> None:
    image = discover_cdvqa(cdvqa_root, splits=("Train",))[0]
    assert image.image_path() == Path(image.file_name)


def test_image_path_with_image_dir_is_resolved_against_it(
    cdvqa_root: Path,
) -> None:
    image = discover_cdvqa(cdvqa_root, splits=("Train",))[0]
    resolved = image.image_path("/data/cdvqa/png")
    assert resolved == Path("/data/cdvqa/png") / image.file_name


def test_resolution_strings_are_preserved(tmp_path: Path) -> None:
    root = tmp_path / "cdvqa"
    write_split(
        root, "Train", images=[image_row(0, "a.png", [0], res_x=".1524m", res_y=".1524m")]
    )
    image = discover_cdvqa(root, splits=("Train",))[0]
    assert image.res_x == ".1524m"
    assert image.res_y == ".1524m"


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(CDVQAError, match="does not exist"):
        discover_cdvqa(tmp_path / "nope")


def test_root_without_annotations_directory_raises(tmp_path: Path) -> None:
    root = tmp_path / "cdvqa"
    root.mkdir()
    with pytest.raises(CDVQAError, match="no 'annotations' directory"):
        discover_cdvqa(root)


def test_missing_annotation_file_raises(tmp_path: Path) -> None:
    root = tmp_path / "cdvqa"
    # Only the questions and answers tables exist; images is absent.
    write_table(root, "Train", "questions", [])
    write_table(root, "Train", "answers", [])
    with pytest.raises(CDVQAError, match="missing CDVQA annotation file"):
        discover_cdvqa(root, splits=("Train",))


def test_malformed_json_raises(tmp_path: Path) -> None:
    root = tmp_path / "cdvqa"
    annotations = root / "annotations"
    annotations.mkdir(parents=True)
    (annotations / "Train_images.json").write_text("{not valid json", encoding="utf-8")
    write_table(root, "Train", "questions", [])
    write_table(root, "Train", "answers", [])
    with pytest.raises(CDVQAError, match="could not read"):
        discover_cdvqa(root, splits=("Train",))


def test_wrong_top_level_key_raises(tmp_path: Path) -> None:
    root = tmp_path / "cdvqa"
    annotations = root / "annotations"
    annotations.mkdir(parents=True)
    (annotations / "Train_images.json").write_text(
        json.dumps({"imgs": []}), encoding="utf-8"
    )
    write_table(root, "Train", "questions", [])
    write_table(root, "Train", "answers", [])
    with pytest.raises(CDVQAError, match="no top-level 'images' list"):
        discover_cdvqa(root, splits=("Train",))


def test_require_images_true_raises_without_pngs(cdvqa_root: Path) -> None:
    with pytest.raises(CDVQAError, match="require_images=True but no .png"):
        discover_cdvqa(cdvqa_root, splits=("Train",), require_images=True)


def test_require_images_false_loads_from_annotations_alone(
    cdvqa_root: Path,
) -> None:
    """Annotations only is a valid state: loading must not need the imagery."""
    assert len(discover_cdvqa(cdvqa_root, splits=("Train",))) == 2


def test_require_images_true_passes_when_pngs_exist(cdvqa_root: Path) -> None:
    images_dir = cdvqa_root / "images"
    images_dir.mkdir()
    (images_dir / "07308.png").write_bytes(b"\x89PNG\r\n\x1a\nstub")
    assert len(discover_cdvqa(cdvqa_root, splits=("Train",), require_images=True)) == 2


def test_require_images_detects_flat_pngs_in_the_root(tmp_path: Path) -> None:
    """``_has_pngs`` probes ``<base>`` itself -- a flat PNG counts."""
    root = tmp_path / "cdvqa"
    write_split(root, "Train", images=[image_row(0, "a.png", [0])])
    (root / "a.png").write_bytes(b"\x89PNG\r\n\x1a\nstub")
    assert len(discover_cdvqa(root, splits=("Train",), require_images=True)) == 1


def test_require_images_does_not_detect_nested_pngs(tmp_path: Path) -> None:
    """``_has_pngs`` is documented as NON-recursive: a nested PNG is not seen.

    The three probe locations are ``<base>/images``, ``<base>/png`` and
    ``<base>``; a PNG buried deeper is deliberately not counted. This asserts
    that documented (and accepted) limitation rather than a bug.
    """
    root = tmp_path / "cdvqa"
    write_split(root, "Train", images=[image_row(0, "a.png", [0])])
    nested = root / "sub" / "deeper"
    nested.mkdir(parents=True)
    (nested / "a.png").write_bytes(b"\x89PNG\r\n\x1a\nstub")
    with pytest.raises(CDVQAError, match="require_images=True but no .png"):
        discover_cdvqa(root, splits=("Train",), require_images=True)


# ---------------------------------------------------------------------------
# Paired imagery layout (SECOND)
# ---------------------------------------------------------------------------
#
# SECOND ships every scene twice -- ``im1`` and ``im2`` -- and the two
# directories SHARE basenames, so ``im1/00003.png`` and ``im2/00003.png`` both
# exist. The tests below pin the pair behaviour and, just as importantly, pin
# what must NOT change: ``image_path`` keeps its single-``Path`` arity, and
# ``_has_pngs`` stays non-recursive.


def test_layout_constants_are_exported() -> None:
    from training.data import cdvqa as mod

    assert "LAYOUT_FLAT" in mod.__all__
    assert "LAYOUT_SECOND_PAIRS" in mod.__all__
    assert "LAYOUT_AUTO" in mod.__all__


def test_image_paths_resolves_im1_then_im2_under_image_dir() -> None:
    image = CDVQAImage(image_id=1, split="Train", file_name="00003.png")
    first, second = image.image_paths("/data/cdvqa")
    assert first == Path("/data/cdvqa") / "im1" / "00003.png"
    assert second == Path("/data/cdvqa") / "im2" / "00003.png"


def test_image_paths_without_image_dir_is_relative() -> None:
    image = CDVQAImage(image_id=1, split="Train", file_name="00003.png")
    assert image.image_paths() == (Path("im1/00003.png"), Path("im2/00003.png"))


def test_shared_basenames_do_not_collide_in_the_paired_layout(
    tmp_path: Path,
) -> None:
    """The reason the flat layout fails: both ``im1``/``im2`` hold the same name."""
    root = tmp_path / "cdvqa"
    write_split(root, "Train", images=[image_row(0, "00003.png", [0])])
    for sub in ("im1", "im2"):
        directory = root / sub
        directory.mkdir(parents=True)
        (directory / "00003.png").write_bytes(b"\x89PNG\r\n\x1a\nstub")
    image = discover_cdvqa(root, splits=("Train",))[0]
    first, second = image.image_paths(root)
    assert first != second
    assert first.exists() and second.exists()


def test_require_images_true_passes_on_the_paired_layout(tmp_path: Path) -> None:
    root = tmp_path / "cdvqa"
    write_split(root, "Train", images=[image_row(0, "00003.png", [0])])
    im1 = root / "im1"
    im1.mkdir(parents=True)
    (im1 / "00003.png").write_bytes(b"\x89PNG\r\n\x1a\nstub")
    assert len(discover_cdvqa(root, splits=("Train",), require_images=True)) == 1


def test_require_images_true_passes_when_pairs_sit_under_images(
    tmp_path: Path,
) -> None:
    """``<base>/images/im2`` is accepted as well as ``<base>/im2``."""
    root = tmp_path / "cdvqa"
    write_split(root, "Train", images=[image_row(0, "00003.png", [0])])
    nested = root / "images" / "im2"
    nested.mkdir(parents=True)
    (nested / "00003.png").write_bytes(b"\x89PNG\r\n\x1a\nstub")
    assert len(discover_cdvqa(root, splits=("Train",), require_images=True)) == 1


def test_paired_presence_is_any_png_not_both_directories(tmp_path: Path) -> None:
    """``_has_pngs`` answers "is imagery here at all", not "is it complete".

    A half-extracted archive therefore still reads as *present*. That is the
    pre-existing flat-layout semantics, kept deliberately: completeness is
    :func:`summarize`'s job (declared vs present), not this probe's.
    """
    root = tmp_path / "cdvqa"
    write_split(root, "Train", images=[image_row(0, "00003.png", [0])])
    im1 = root / "im1"
    im1.mkdir(parents=True)
    (im1 / "00003.png").write_bytes(b"\x89PNG\r\n\x1a\nstub")
    # im2 is absent entirely -- still detected.
    assert len(discover_cdvqa(root, splits=("Train",), require_images=True)) == 1


def test_paired_probe_stays_non_recursive(tmp_path: Path) -> None:
    """The paired probe must not become a recursive glob.

    ``<base>/im1/deeper/a.png`` is not a recognised location: the paired layout
    is ``<base>/im1/*.png``, one level, exactly like the flat probes. Without
    this guard a recursive implementation would pass every test above while
    silently widening what counts as imagery.
    """
    root = tmp_path / "cdvqa"
    write_split(root, "Train", images=[image_row(0, "a.png", [0])])
    deeper = root / "im1" / "deeper"
    deeper.mkdir(parents=True)
    (deeper / "a.png").write_bytes(b"\x89PNG\r\n\x1a\nstub")
    with pytest.raises(CDVQAError, match="require_images=True but no .png"):
        discover_cdvqa(root, splits=("Train",), require_images=True)


def test_image_path_keeps_its_single_path_arity() -> None:
    """``image_path`` must NOT be widened to return a pair."""
    image = CDVQAImage(image_id=1, split="Train", file_name="a.png")
    result = image.image_path("/data/cdvqa")
    assert isinstance(result, Path)
    assert result == Path("/data/cdvqa") / "a.png"


def test_image_paths_rejects_the_flat_layout() -> None:
    image = CDVQAImage(image_id=1, split="Train", file_name="a.png")
    with pytest.raises(CDVQAError, match="no temporal pair"):
        image.image_paths("/data/cdvqa", layout=LAYOUT_FLAT)


def test_image_paths_rejects_an_unknown_layout() -> None:
    image = CDVQAImage(image_id=1, split="Train", file_name="a.png")
    with pytest.raises(CDVQAError, match="unknown imagery layout"):
        image.image_paths("/data/cdvqa", layout="nope")


def test_missing_imagery_error_names_the_recognised_layouts(tmp_path: Path) -> None:
    """A layout mismatch must not read as missing data."""
    root = tmp_path / "cdvqa"
    write_split(root, "Train", images=[image_row(0, "a.png", [0])])
    with pytest.raises(CDVQAError) as excinfo:
        discover_cdvqa(root, splits=("Train",), require_images=True)
    message = str(excinfo.value)
    assert LAYOUT_SECOND_PAIRS in message
    assert "im1" in message and "im2" in message


def test_reading_a_subset_of_splits(tmp_path: Path) -> None:
    root = tmp_path / "cdvqa"
    for split in CDVQA_SPLITS:
        write_split(root, split, images=[image_row(0, f"{split}.png", [0])])
    images = discover_cdvqa(root, splits=("Test", "Test2"))
    assert {img.split for img in images} == {"Test", "Test2"}
    assert len(images) == 2


def test_load_returns_a_corpus_with_drops_for_each_requested_split(
    cdvqa_root: Path,
) -> None:
    corpus = load_cdvqa(cdvqa_root, splits=("Train",))
    assert isinstance(corpus, CDVQACorpus)
    assert set(corpus.drops) == {"Train"}
    assert len(corpus.images) == 2
    assert len(corpus.questions) == 5


# ---------------------------------------------------------------------------
# The join -- drop accounting
# ---------------------------------------------------------------------------


def _drop_fixture(root: Path) -> None:
    """One split exercising all four drop reasons at once."""
    write_split(
        root,
        "Train",
        # 999 is referenced by the image table but no such question exists.
        images=[
            image_row(0, "a.png", [100, 102, 104]),
            image_row(1, "b.png", [101, 103, 999]),
        ],
        questions=[
            question_row(100, 0, "change_or_not", [0]),
            question_row(101, 1, "change_or_not", [1]),
            question_row(102, 0, "change_or_not", [2, 3]),  # two answers
            question_row(103, 9999, "change_or_not", [4]),  # unknown img_id
            question_row(104, 0, "change_or_not", [777]),  # answer 777 missing
        ],
        answers=[
            answer_row(0, 100, "yes"),
            answer_row(1, 101, "no"),
            answer_row(2, 102, "yes"),
            answer_row(3, 102, "no"),
            answer_row(4, 103, "yes"),
        ],
    )


def test_multi_answer_question_is_dropped_and_counted(tmp_path: Path) -> None:
    root = tmp_path / "cdvqa"
    _drop_fixture(root)
    corpus = load_cdvqa(root, splits=("Train",))
    assert corpus.drops["Train"]["multi_answer_count"] == 1
    assert 102 not in {q.question_id for q in corpus.questions}


def test_unknown_image_question_is_dropped_and_counted(tmp_path: Path) -> None:
    root = tmp_path / "cdvqa"
    _drop_fixture(root)
    corpus = load_cdvqa(root, splits=("Train",))
    assert corpus.drops["Train"]["unknown_image_count"] == 1
    assert 103 not in {q.question_id for q in corpus.questions}


def test_answerless_question_is_dropped_and_counted(tmp_path: Path) -> None:
    root = tmp_path / "cdvqa"
    _drop_fixture(root)
    corpus = load_cdvqa(root, splits=("Train",))
    assert corpus.drops["Train"]["answerless_count"] == 1
    assert 104 not in {q.question_id for q in corpus.questions}


def test_dangling_reference_is_counted(tmp_path: Path) -> None:
    root = tmp_path / "cdvqa"
    _drop_fixture(root)
    corpus = load_cdvqa(root, splits=("Train",))
    assert corpus.drops["Train"]["dangling_reference_count"] == 1
    assert corpus.drops["Train"]["declared_question_refs"] == 6


def test_resolved_questions_are_the_survivors(tmp_path: Path) -> None:
    root = tmp_path / "cdvqa"
    _drop_fixture(root)
    corpus = load_cdvqa(root, splits=("Train",))
    assert {q.question_id for q in corpus.questions} == {100, 101}


def test_questions_are_sorted_by_question_id(cdvqa_root: Path) -> None:
    questions = discover_questions(cdvqa_root, splits=("Train",))
    ids = [q.question_id for q in questions]
    assert ids == sorted(ids)


def test_date_added_is_parsed_as_a_float(cdvqa_root: Path) -> None:
    questions = discover_questions(cdvqa_root, splits=("Train",))
    assert all(isinstance(q.date_added, float) for q in questions)


def test_date_added_tolerates_a_string_and_omission(tmp_path: Path) -> None:
    """A future release emitting a string (or nothing) must still load."""
    root = tmp_path / "cdvqa"
    write_split(
        root,
        "Train",
        images=[image_row(0, "a.png", [0, 1])],
        questions=[
            question_row(0, 0, "change_or_not", [0], date_added="1631214265.5"),
            question_row(1, 0, "change_or_not", [1], date_added=None),
        ],
        answers=[answer_row(0, 0, "yes"), answer_row(1, 1, "no")],
    )
    questions = {q.question_id: q for q in discover_questions(root, splits=("Train",))}
    assert questions[0].date_added == 1631214265.5
    assert questions[1].date_added is None


def test_the_drop_proxy_conflates_the_four_reasons(tmp_path: Path) -> None:
    """THE MUTATION TEST.

    ``summarize`` without ``drops`` reports a single proxy:
    ``len(declared_refs - resolved)``. That number folds together four
    genuinely different losses. Build two corpora that are INDISTINGUISHABLE
    through the proxy and the true dangling count, yet differ in
    ``multi_answer_count`` -- proving the counters are distinct and that the
    proxy cannot reveal them.
    """
    root_a = tmp_path / "a"
    write_split(
        root_a,
        "Train",
        images=[image_row(0, "a.png", [100, 101])],
        questions=[
            question_row(100, 0, "change_or_not", [0, 1]),  # multi-answer
            question_row(101, 0, "change_or_not", [777]),  # answerless
        ],
        answers=[answer_row(0, 100, "yes"), answer_row(1, 100, "no")],
    )

    root_b = tmp_path / "b"
    write_split(
        root_b,
        "Train",
        images=[image_row(0, "a.png", [100, 101])],
        questions=[
            question_row(100, 9999, "change_or_not", [0]),  # unknown image
            question_row(101, 0, "change_or_not", [777]),  # answerless
        ],
        answers=[answer_row(0, 100, "yes")],
    )

    corpus_a = load_cdvqa(root_a, splits=("Train",))
    corpus_b = load_cdvqa(root_b, splits=("Train",))

    summary_a = summarize(corpus_a.images, corpus_a.questions)
    summary_b = summarize(corpus_b.images, corpus_b.questions)

    # The proxy is IDENTICAL for both corpora ...
    assert summary_a["dangling_reference_count"] == 2
    assert summary_b["dangling_reference_count"] == 2

    # ... and so is the TRUE dangling count ...
    assert corpus_a.drops["Train"]["dangling_reference_count"] == 0
    assert corpus_b.drops["Train"]["dangling_reference_count"] == 0

    # ... yet the multi-answer counts DIFFER. The proxy cannot reveal it.
    assert corpus_a.drops["Train"]["multi_answer_count"] == 1
    assert corpus_b.drops["Train"]["multi_answer_count"] == 0

    # And the proxy is exactly the sum of the three non-dangling losses here.
    assert summary_a["dangling_reference_count"] == (
        corpus_a.drops["Train"]["multi_answer_count"]
        + corpus_a.drops["Train"]["unknown_image_count"]
        + corpus_a.drops["Train"]["answerless_count"]
    )


def test_dangling_alone_does_not_reveal_the_other_losses(tmp_path: Path) -> None:
    """A corpus with real drops but zero dangling refs still loses questions."""
    root = tmp_path / "cdvqa"
    _drop_fixture(root)
    corpus = load_cdvqa(root, splits=("Train",))
    drops = corpus.drops["Train"]
    # dangling is 1, but three more questions vanished for other reasons.
    assert drops["dangling_reference_count"] == 1
    other_losses = (
        drops["multi_answer_count"]
        + drops["unknown_image_count"]
        + drops["answerless_count"]
    )
    assert other_losses == 3
    assert other_losses != drops["dangling_reference_count"]


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


def test_one_example_per_resolved_question(cdvqa_root: Path) -> None:
    images = discover_cdvqa(cdvqa_root, splits=("Train",))
    questions = discover_questions(cdvqa_root, splits=("Train",))
    examples = build_cdvqa_examples(images, questions)
    assert len(examples) == len(questions) == 5


def test_example_has_the_exact_expected_key_set(cdvqa_root: Path) -> None:
    images = discover_cdvqa(cdvqa_root, splits=("Train",))
    questions = discover_questions(cdvqa_root, splits=("Train",))
    examples = build_cdvqa_examples(images, questions)
    assert all(set(example) == EXAMPLE_KEYS for example in examples)


def test_example_pairs_the_question_with_its_own_scene(
    cdvqa_root: Path,
) -> None:
    images = discover_cdvqa(cdvqa_root, splits=("Train",))
    questions = {q.question_id: q for q in discover_questions(cdvqa_root, splits=("Train",))}
    examples = {e["question_id"]: e for e in build_cdvqa_examples(images, questions.values())}
    # question 4 was asked about 10589.png, the rest about 07308.png.
    assert examples[4]["file_name"] == "10589.png"
    assert examples[0]["file_name"] == "07308.png"


def test_question_types_filter_is_exact(cdvqa_root: Path) -> None:
    images = discover_cdvqa(cdvqa_root, splits=("Train",))
    questions = discover_questions(cdvqa_root, splits=("Train",))
    examples = build_cdvqa_examples(
        images, questions, question_types=("change_or_not", "change_ratio")
    )
    assert {e["qtype"] for e in examples} == {"change_or_not", "change_ratio"}


def test_question_types_filter_matching_nothing_yields_nothing(
    cdvqa_root: Path,
) -> None:
    images = discover_cdvqa(cdvqa_root, splits=("Train",))
    questions = discover_questions(cdvqa_root, splits=("Train",))
    assert build_cdvqa_examples(images, questions, question_types=("nope",)) == []


def test_example_image_path_resolves_against_image_dir(cdvqa_root: Path) -> None:
    images = discover_cdvqa(cdvqa_root, splits=("Train",))
    questions = discover_questions(cdvqa_root, splits=("Train",))
    examples = build_cdvqa_examples(images, questions, image_dir="/data/png")
    assert all(
        e["image_path"] == str(Path("/data/png") / e["file_name"]) for e in examples
    )


# ---------------------------------------------------------------------------
# Layout detection and the paired-imagery example paths (defect #9)
#
# These are the guards for defect #9: before the fix, ``build_cdvqa_examples``
# emitted the FLAT path for every scene, so against the real SECOND corpus
# 0 / 153,130 example ``image_path`` values resolved on disk. Each test below
# fails under the defective implementation -- that is the point.
# ---------------------------------------------------------------------------


def test_detect_layout_none_is_flat() -> None:
    """No directory named => the flat, single-file expectation."""
    assert detect_layout(None) == LAYOUT_FLAT


def test_detect_layout_recognises_paired_root(cdvqa_paired_root: Path) -> None:
    assert detect_layout(cdvqa_paired_root) == LAYOUT_SECOND_PAIRS


def test_detect_layout_recognises_flat_root(cdvqa_flat_root: Path) -> None:
    assert detect_layout(cdvqa_flat_root) == LAYOUT_FLAT


def test_paired_example_image_path_resolves_on_disk(cdvqa_paired_root: Path) -> None:
    """Defect #9: the flat path never exists; the PRE image path must."""
    images = discover_cdvqa(cdvqa_paired_root, splits=("Train",))
    questions = discover_questions(cdvqa_paired_root, splits=("Train",))
    examples = build_cdvqa_examples(images, questions, image_dir=cdvqa_paired_root)
    assert examples
    assert all(e["layout"] == LAYOUT_SECOND_PAIRS for e in examples)
    assert all(Path(e["image_path"]).is_file() for e in examples)


def test_paired_example_image_paths_all_resolve_on_disk(cdvqa_paired_root: Path) -> None:
    images = discover_cdvqa(cdvqa_paired_root, splits=("Train",))
    questions = discover_questions(cdvqa_paired_root, splits=("Train",))
    examples = build_cdvqa_examples(images, questions, image_dir=cdvqa_paired_root)
    assert all(Path(p).is_file() for e in examples for p in e["image_paths"])


def test_paired_example_image_paths_are_pre_then_post(cdvqa_paired_root: Path) -> None:
    """``image_paths`` is ``[im1/f, im2/f]`` and ``image_path`` is the PRE one."""
    images = discover_cdvqa(cdvqa_paired_root, splits=("Train",))
    questions = discover_questions(cdvqa_paired_root, splits=("Train",))
    examples = build_cdvqa_examples(images, questions, image_dir=cdvqa_paired_root)
    for example in examples:
        assert len(example["image_paths"]) == 2
        assert Path(example["image_paths"][0]).parts[-2] == "im1"
        assert Path(example["image_paths"][1]).parts[-2] == "im2"
        assert Path(example["image_paths"][0]).name == example["file_name"]
        assert Path(example["image_paths"][1]).name == example["file_name"]
        # image_path is the PRE image, i.e. the same file as image_paths[0]
        assert example["image_path"] == example["image_paths"][0]


def test_paired_example_image_path_is_not_the_flat_path(cdvqa_paired_root: Path) -> None:
    """The old defect emitted ``<root>/<file_name>``, which does not exist."""
    images = discover_cdvqa(cdvqa_paired_root, splits=("Train",))
    questions = discover_questions(cdvqa_paired_root, splits=("Train",))
    examples = build_cdvqa_examples(images, questions, image_dir=cdvqa_paired_root)
    assert all(
        e["image_path"] != str(cdvqa_paired_root / e["file_name"]) for e in examples
    )


def test_flat_example_has_a_single_image_path(cdvqa_flat_root: Path) -> None:
    images = discover_cdvqa(cdvqa_flat_root, splits=("Train",))
    questions = discover_questions(cdvqa_flat_root, splits=("Train",))
    examples = build_cdvqa_examples(images, questions, image_dir=cdvqa_flat_root)
    assert all(e["layout"] == LAYOUT_FLAT for e in examples)
    assert all(len(e["image_paths"]) == 1 for e in examples)
    assert all(Path(e["image_paths"][0]).is_file() for e in examples)


def test_no_image_dir_is_flat_back_compat(cdvqa_paired_root: Path) -> None:
    """No ``image_dir`` => unchanged flat behaviour, byte-for-byte."""
    images = discover_cdvqa(cdvqa_paired_root, splits=("Train",))
    questions = discover_questions(cdvqa_paired_root, splits=("Train",))
    examples = build_cdvqa_examples(images, questions)
    assert all(e["layout"] == LAYOUT_FLAT for e in examples)
    assert all(e["image_path"] == e["file_name"] for e in examples)
    assert all(e["image_paths"] == [e["file_name"]] for e in examples)


def test_forced_flat_layout_on_paired_root(cdvqa_paired_root: Path) -> None:
    images = discover_cdvqa(cdvqa_paired_root, splits=("Train",))
    questions = discover_questions(cdvqa_paired_root, splits=("Train",))
    examples = build_cdvqa_examples(
        images, questions, image_dir=cdvqa_paired_root, layout=LAYOUT_FLAT
    )
    assert all(e["layout"] == LAYOUT_FLAT for e in examples)
    assert all(len(e["image_paths"]) == 1 for e in examples)
    assert all(
        e["image_path"] == str(cdvqa_paired_root / e["file_name"]) for e in examples
    )


def test_forced_paired_layout_on_paired_root(cdvqa_paired_root: Path) -> None:
    images = discover_cdvqa(cdvqa_paired_root, splits=("Train",))
    questions = discover_questions(cdvqa_paired_root, splits=("Train",))
    examples = build_cdvqa_examples(
        images, questions, image_dir=cdvqa_paired_root, layout=LAYOUT_SECOND_PAIRS
    )
    assert all(e["layout"] == LAYOUT_SECOND_PAIRS for e in examples)
    assert all(len(e["image_paths"]) == 2 for e in examples)
    assert all(Path(p).is_file() for e in examples for p in e["image_paths"])


def test_unknown_layout_raises_naming_accepted_values(cdvqa_paired_root: Path) -> None:
    images = discover_cdvqa(cdvqa_paired_root, splits=("Train",))
    questions = discover_questions(cdvqa_paired_root, splits=("Train",))
    with pytest.raises(CDVQAError) as excinfo:
        build_cdvqa_examples(
            images, questions, image_dir=cdvqa_paired_root, layout="nope"
        )
    message = str(excinfo.value)
    assert "auto" in message
    assert LAYOUT_FLAT in message
    assert LAYOUT_SECOND_PAIRS in message


def test_build_cdvqa_examples_accepts_the_auto_constant(
    cdvqa_paired_root: Path,
) -> None:
    """``layout=LAYOUT_AUTO`` is the auto-detect path -- the exported constant,
    not a bare literal -- and it resolves exactly as the default does."""
    images = discover_cdvqa(cdvqa_paired_root, splits=("Train",))
    questions = discover_questions(cdvqa_paired_root, splits=("Train",))
    auto = build_cdvqa_examples(
        images, questions, image_dir=cdvqa_paired_root, layout=LAYOUT_AUTO
    )
    default = build_cdvqa_examples(images, questions, image_dir=cdvqa_paired_root)
    assert auto == default
    assert all(e["layout"] == LAYOUT_SECOND_PAIRS for e in auto)
    assert all(Path(e["image_path"]).is_file() for e in auto)


def test_image_paths_rejects_the_auto_sentinel() -> None:
    """``auto`` is resolved by the caller from a dataset root; a single scene
    cannot honour it, so it is rejected with a message that says so -- not the
    generic unknown-layout one."""
    image = CDVQAImage(image_id=1, split="Train", file_name="a.png")
    with pytest.raises(CDVQAError) as excinfo:
        image.image_paths("/data/cdvqa", layout=LAYOUT_AUTO)
    message = str(excinfo.value)
    assert LAYOUT_AUTO in message
    assert "auto" in message
    assert "unknown imagery layout" not in message


def test_question_whose_image_is_unknown_is_skipped_not_mispaired() -> None:
    """A wrong image is worse than a missing example -- so skip, never guess."""
    images = [CDVQAImage(image_id=0, split="Train", file_name="a.png", question_ids=(1,))]
    questions = [
        CDVQAQuestion(
            question_id=1, split="Train", image_id=0, qtype="change_or_not",
            question="q1", answer="yes", answer_id=0,
        ),
        # question 2 is referenced by no image at all.
        CDVQAQuestion(
            question_id=2, split="Train", image_id=0, qtype="change_or_not",
            question="q2", answer="no", answer_id=1,
        ),
    ]
    examples = build_cdvqa_examples(images, questions)
    assert len(examples) == 1
    assert examples[0]["question_id"] == 1
    assert examples[0]["file_name"] == "a.png"


def test_question_is_never_paired_with_another_images_file() -> None:
    images = [
        CDVQAImage(image_id=0, split="Train", file_name="a.png", question_ids=(1,)),
        CDVQAImage(image_id=1, split="Train", file_name="b.png", question_ids=(2,)),
    ]
    questions = [
        CDVQAQuestion(
            question_id=2, split="Train", image_id=1, qtype="change_or_not",
            question="q2", answer="no", answer_id=1,
        )
    ]
    examples = build_cdvqa_examples(images, questions)
    assert examples[0]["file_name"] == "b.png"
    assert examples[0]["scene_key"] == "b"


def test_example_order_is_deterministic(cdvqa_root: Path) -> None:
    images = discover_cdvqa(cdvqa_root, splits=("Train",))
    questions = discover_questions(cdvqa_root, splits=("Train",))
    first = build_cdvqa_examples(images, questions)
    second = build_cdvqa_examples(images, questions)
    assert first == second


# ---------------------------------------------------------------------------
# F1 — cross-split question-id collision
#
# `question_id` is PER-SPLIT sequential: every split restarts at 0 (measured on
# the real corpus 2026-09-21 — Train 0..65966, Val 0..16440, Test 0..39685,
# Test2 0..31035, all four with no gaps). `build_cdvqa_examples` used to index
# its question->scene map on the bare `question_id`, so over a concatenation of
# splits a later split overwrote an earlier one and the last writer won. On the
# real corpus a single call over ("Train", "Val") produced 82,408 examples over
# only 1,601 scenes instead of 2,000, with 32,882 questions paired with the
# OTHER split's imagery. Nothing raised; the examples still trained.
#
# The index is now keyed on `(split, question_id)`. Each test below fails under
# the defective implementation — that is the point.
# ---------------------------------------------------------------------------


def _two_split_collision_corpus(root: Path) -> None:
    """Train and Val that BOTH own question id 0, about different scenes.

    This is the minimal reproduction of the real corpus's layout: ids restart
    per split, so a bare-id index cannot tell the two apart.
    """
    write_split(
        root,
        "Train",
        images=[image_row(0, "train_scene.png", [0])],
        questions=[question_row(0, 0, "change_or_not", [0])],
        answers=[answer_row(0, 0, "yes")],
    )
    write_split(
        root,
        "Val",
        images=[image_row(0, "val_scene.png", [0])],
        questions=[question_row(0, 0, "change_or_not", [0])],
        answers=[answer_row(0, 0, "no")],
    )


def test_multi_split_examples_pair_each_question_with_its_own_split(
    tmp_path: Path,
) -> None:
    """F1: two splits both owning question id 0 must not collide."""
    root = tmp_path / "cdvqa"
    _two_split_collision_corpus(root)

    images = discover_cdvqa(root, splits=("Train", "Val"))
    questions = discover_questions(root, splits=("Train", "Val"))
    examples = {
        (e["split"], e["question_id"]): e
        for e in build_cdvqa_examples(images, questions)
    }

    assert set(examples) == {("Train", 0), ("Val", 0)}
    # The load-bearing assertion: each question keeps ITS OWN split's scene.
    assert examples[("Train", 0)]["file_name"] == "train_scene.png"
    assert examples[("Val", 0)]["file_name"] == "val_scene.png"
    # The answer was always split-scoped; only the pairing was wrong. Pinned so
    # the fix cannot be "made to pass" by moving the answer instead.
    assert examples[("Train", 0)]["answer"] == "yes"
    assert examples[("Val", 0)]["answer"] == "no"


def test_multi_split_examples_never_borrow_another_splits_imagery(
    tmp_path: Path,
) -> None:
    """The general form: no example may name a scene outside its own split."""
    root = tmp_path / "cdvqa"
    _two_split_collision_corpus(root)

    images = discover_cdvqa(root, splits=("Train", "Val"))
    questions = discover_questions(root, splits=("Train", "Val"))
    scenes_by_split = {split: set() for split in ("Train", "Val")}
    for image in images:
        scenes_by_split[image.split].add(image.file_name)

    examples = build_cdvqa_examples(images, questions)
    assert examples
    for example in examples:
        own = scenes_by_split[example["split"]]
        others = set().union(
            *(s for name, s in scenes_by_split.items() if name != example["split"])
        )
        assert example["file_name"] in own, example
        assert example["file_name"] not in others, example


def test_multi_split_and_per_split_calls_agree(tmp_path: Path) -> None:
    """The equivalence that lets `_examples_for_one_split` stay as a guard.

    One call over both splits must produce exactly what two per-split calls
    produce. If the adapter regresses, the two disagree and this fails.
    """
    root = tmp_path / "cdvqa"
    _two_split_collision_corpus(root)

    images = discover_cdvqa(root, splits=("Train", "Val"))
    questions = discover_questions(root, splits=("Train", "Val"))
    together = build_cdvqa_examples(images, questions)

    separately: list[dict] = []
    for split in ("Train", "Val"):
        separately.extend(
            build_cdvqa_examples(
                discover_cdvqa(root, splits=(split,)),
                discover_questions(root, splits=(split,)),
            )
        )
    assert together == separately


def test_a_single_split_call_is_unaffected_by_the_split_scoped_index(
    cdvqa_root: Path,
) -> None:
    """No behaviour change for one split: ids are unique within a split."""
    images = discover_cdvqa(cdvqa_root, splits=("Train",))
    questions = discover_questions(cdvqa_root, splits=("Train",))
    examples = build_cdvqa_examples(images, questions)
    assert len(examples) == len(questions) == 5
    assert all(e["split"] == "Train" for e in examples)
    assert all(set(e) == EXAMPLE_KEYS for e in examples)


def test_a_question_whose_own_split_has_no_imagery_is_skipped_not_guessed(
    tmp_path: Path,
) -> None:
    """The scoped index must SKIP, not fall back to a same-numbered id.

    Passing Val questions against Train-only imagery is a caller error, and the
    correct response is to emit nothing rather than to pair the Val question
    with whatever Train scene happens to own id 0.
    """
    root = tmp_path / "cdvqa"
    _two_split_collision_corpus(root)

    train_images = discover_cdvqa(root, splits=("Train",))
    val_questions = discover_questions(root, splits=("Val",))
    assert build_cdvqa_examples(train_images, val_questions) == []


# ---------------------------------------------------------------------------
# Summarize / warnings
# ---------------------------------------------------------------------------


def _resolved(root: Path, split: str = "Train"):
    return discover_cdvqa(root, splits=(split,)), discover_questions(root, splits=(split,))


def test_image_files_present_is_none_without_image_root(cdvqa_root: Path) -> None:
    images, questions = _resolved(cdvqa_root)
    summary = summarize(images, questions)
    assert summary["image_files_present"] is None
    assert summary["image_files_declared"] == 2


def test_no_absence_claim_when_presence_was_not_checked(cdvqa_root: Path) -> None:
    """``None`` means 'not checked', so no absence may be asserted."""
    images, questions = _resolved(cdvqa_root)
    problems = warnings_for(summarize(images, questions))
    assert not any("are present on disk" in p for p in problems)
    assert not any("NOT ready" in p for p in problems)
    # Only a hedged note that presence was not looked at.
    assert any("not checked" in p for p in problems)


def test_image_files_present_is_zero_for_an_empty_image_root(
    cdvqa_root: Path, tmp_path: Path
) -> None:
    images, questions = _resolved(cdvqa_root)
    empty = tmp_path / "png"
    empty.mkdir()
    summary = summarize(images, questions, image_root=empty)
    assert summary["image_files_present"] == 0
    problems = warnings_for(summary)
    assert any("are present on disk" in p and "NOT ready" in p for p in problems)


def test_partial_presence_warns_with_the_right_count(
    cdvqa_root: Path, tmp_path: Path
) -> None:
    images, questions = _resolved(cdvqa_root)
    png_dir = tmp_path / "png"
    png_dir.mkdir()
    (png_dir / "07308.png").write_bytes(b"\x89PNG\r\n\x1a\nstub")
    summary = summarize(images, questions, image_root=png_dir)
    assert summary["image_files_present"] == 1
    assert summary["image_files_declared"] == 2
    problems = warnings_for(summary)
    assert any("only 1 of 2 declared image files are present" in p for p in problems)


def test_no_image_warning_when_every_file_is_present(
    cdvqa_root: Path, tmp_path: Path
) -> None:
    images, questions = _resolved(cdvqa_root)
    png_dir = tmp_path / "png"
    png_dir.mkdir()
    for name in ("07308.png", "10589.png"):
        (png_dir / name).write_bytes(b"\x89PNG\r\n\x1a\nstub")
    summary = summarize(images, questions, image_root=png_dir)
    assert summary["image_files_present"] == 2
    problems = warnings_for(summary)
    assert not any("image files are present" in p for p in problems)


# ---------------------------------------------------------------------------
# Presence must recognise the PAIRED layout
# ---------------------------------------------------------------------------
#
# Found by running the real corpus end-to-end after extracting SECOND:
# ``summarize`` reported ``image_files_present == 0`` while 11,872 PNGs sat on
# disk, because presence counted ``<root>/<name>`` only. ``warnings_for`` then
# announced the dataset was "NOT ready". Availability and presence have to agree
# on which layouts count.


def test_presence_counts_a_scene_found_in_the_paired_layout(
    cdvqa_root: Path, tmp_path: Path
) -> None:
    images, questions = _resolved(cdvqa_root)
    root = tmp_path / "cdvqa"
    for sub in ("im1", "im2"):
        directory = root / sub
        directory.mkdir(parents=True)
        for name in ("07308.png", "10589.png"):
            (directory / name).write_bytes(b"\x89PNG\r\n\x1a\nstub")
    summary = summarize(images, questions, image_root=root)
    assert summary["image_files_present"] == 2
    assert summary["image_files_declared"] == 2


def test_presence_counts_a_scene_present_in_im2_only(
    cdvqa_root: Path, tmp_path: Path
) -> None:
    """Presence means "imagery is here at all", matching ``_has_pngs``."""
    images, questions = _resolved(cdvqa_root)
    root = tmp_path / "cdvqa"
    (root / "im2").mkdir(parents=True)
    (root / "im2" / "07308.png").write_bytes(b"\x89PNG\r\n\x1a\nstub")
    summary = summarize(images, questions, image_root=root)
    assert summary["image_files_present"] == 1


def test_no_absence_warning_when_paired_imagery_is_complete(
    cdvqa_root: Path, tmp_path: Path
) -> None:
    """The exact false alarm the real corpus produced."""
    images, questions = _resolved(cdvqa_root)
    root = tmp_path / "cdvqa"
    for sub in ("im1", "im2"):
        directory = root / sub
        directory.mkdir(parents=True)
        for name in ("07308.png", "10589.png"):
            (directory / name).write_bytes(b"\x89PNG\r\n\x1a\nstub")
    problems = warnings_for(summarize(images, questions, image_root=root))
    assert not any("are present on disk" in p for p in problems)
    assert not any("NOT ready" in p for p in problems)


def test_flat_presence_still_works_after_the_paired_addition(
    cdvqa_root: Path, tmp_path: Path
) -> None:
    """Regression guard: adding the paired probe must not break the flat case."""
    images, questions = _resolved(cdvqa_root)
    png_dir = tmp_path / "png"
    png_dir.mkdir()
    (png_dir / "07308.png").write_bytes(b"\x89PNG\r\n\x1a\nstub")
    assert summarize(images, questions, image_root=png_dir)["image_files_present"] == 1


def test_presence_probe_stays_non_recursive(cdvqa_root: Path, tmp_path: Path) -> None:
    """A PNG buried deeper than a recognised directory must not count."""
    images, questions = _resolved(cdvqa_root)
    root = tmp_path / "cdvqa"
    deeper = root / "im1" / "deeper"
    deeper.mkdir(parents=True)
    for name in ("07308.png", "10589.png"):
        (deeper / name).write_bytes(b"\x89PNG\r\n\x1a\nstub")
    assert summarize(images, questions, image_root=root)["image_files_present"] == 0


def test_test_and_test2_sharing_all_images_warns_they_are_not_independent(
    tmp_path: Path,
) -> None:
    """The single most important warning in the module."""
    root = tmp_path / "cdvqa"
    names = ["a.png", "b.png", "c.png"]
    for split in ("Test", "Test2"):
        write_split(
            root,
            split,
            images=[image_row(i, n, [i]) for i, n in enumerate(names)],
        )
    images = discover_cdvqa(root, splits=("Test", "Test2"))
    problems = warnings_for(summarize(images, []))
    assert any("SAME" in p and "NOT independent" in p for p in problems), problems


def test_test_and_test2_disjoint_images_do_not_warn(tmp_path: Path) -> None:
    """The negative case: the check must not fire on genuinely separate sets."""
    root = tmp_path / "cdvqa"
    write_split(root, "Test", images=[image_row(0, "t1.png", [0])])
    write_split(root, "Test2", images=[image_row(0, "t2.png", [0])])
    images = discover_cdvqa(root, splits=("Test", "Test2"))
    problems = warnings_for(summarize(images, []))
    # No independence claim of any kind -- neither "SAME" nor "share ...".
    assert not any("independent" in p for p in problems), problems
    assert not any("share" in p for p in problems), problems


def test_partial_test_test2_overlap_warns_but_not_as_identical(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cdvqa"
    write_split(
        root, "Test",
        images=[image_row(0, "a.png", [0]), image_row(1, "b.png", [1])],
    )
    write_split(
        root, "Test2",
        images=[image_row(0, "a.png", [0]), image_row(1, "c.png", [1])],
    )
    images = discover_cdvqa(root, splits=("Test", "Test2"))
    problems = warnings_for(summarize(images, []))
    assert any("share 1 images" in p for p in problems), problems
    assert not any("SAME" in p for p in problems)


def test_cross_split_overlap_matrix_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "cdvqa"
    write_split(root, "Test", images=[image_row(0, "a.png", [0])])
    write_split(root, "Test2", images=[image_row(0, "a.png", [0])])
    summary = summarize(discover_cdvqa(root, splits=("Test", "Test2")), [])
    assert summary["cross_split_overlap"]["Test"]["Test2"] == 1
    assert summary["cross_split_overlap"]["Test"]["Test"] == 1


def test_per_split_vocabulary_drift_warns(tmp_path: Path) -> None:
    root = tmp_path / "cdvqa"
    write_split(
        root,
        "Train",
        images=[image_row(0, "a.png", [0, 1])],
        questions=[
            question_row(0, 0, "change_or_not", [0]),
            question_row(1, 0, "change_or_not", [1]),
        ],
        answers=[answer_row(0, 0, "yes"), answer_row(1, 1, "no")],
    )
    write_split(
        root,
        "Val",
        images=[image_row(0, "b.png", [0])],
        questions=[question_row(0, 0, "change_or_not", [0])],
        answers=[answer_row(0, 0, "yes")],
    )
    images = discover_cdvqa(root, splits=("Train", "Val"))
    questions = discover_questions(root, splits=("Train", "Val"))
    problems = warnings_for(summarize(images, questions))
    assert any(
        "split 'Val' never produced answers ['no']" in p for p in problems
    ), problems


def test_degenerate_single_answer_vocabulary_warns(tmp_path: Path) -> None:
    root = tmp_path / "cdvqa"
    write_split(
        root,
        "Train",
        images=[image_row(0, "a.png", [0, 1])],
        questions=[
            question_row(0, 0, "change_to_what", [0]),
            question_row(1, 0, "change_to_what", [1]),
        ],
        answers=[answer_row(0, 0, "buildings"), answer_row(1, 1, "buildings")],
    )
    images, questions = _resolved(root)
    problems = warnings_for(summarize(images, questions))
    assert any("single-answer vocabulary" in p for p in problems), problems


def test_vocabulary_outside_the_frozen_set_warns(tmp_path: Path) -> None:
    root = tmp_path / "cdvqa"
    write_split(
        root,
        "Train",
        images=[image_row(0, "a.png", [0, 1])],
        questions=[
            question_row(0, 0, "change_or_not", [0]),
            question_row(1, 0, "change_or_not", [1]),
        ],
        answers=[answer_row(0, 0, "yes"), answer_row(1, 1, "maybe")],
    )
    images, questions = _resolved(root)
    problems = warnings_for(summarize(images, questions))
    assert any("outside the frozen vocabulary" in p for p in problems), problems


def test_zero_question_split_reports_zero_and_warns(tmp_path: Path) -> None:
    """A split with imagery but no resolved questions is a LIVE guard.

    This branch used to be unreachable: ``questions_per_split`` was keyed off
    the resolved-question Counter alone, which can never hold a zero, so a
    split with imagery and no questions was simply absent and the "zero
    questions" check never fired. ``summarize`` now keys the map over the
    union of image splits and question splits, so the split appears as an
    explicit ``0`` and the warning fires. This test is the regression that
    stops that guard going vacuous again.
    """
    root = tmp_path / "cdvqa"
    # Imagery for Train, but the questions table is empty.
    write_split(root, "Train", images=[image_row(0, "a.png", [0])])
    images = discover_cdvqa(root, splits=("Train",))
    summary = summarize(images, [])
    assert summary["questions_per_split"] == {"Train": 0}
    problems = warnings_for(summary)
    assert any("split 'Train' has zero questions" in p for p in problems), problems


def test_split_with_neither_images_nor_questions_is_not_invented(
    tmp_path: Path,
) -> None:
    """No split is invented: a requested split with empty tables stays absent."""
    root = tmp_path / "cdvqa"
    write_split(
        root,
        "Train",
        images=[image_row(0, "a.png", [0])],
        questions=[question_row(0, 0, "change_or_not", [0])],
        answers=[answer_row(0, 0, "yes")],
    )
    # Val is requested (its three tables exist) but holds nothing at all.
    write_split(root, "Val")
    images = discover_cdvqa(root, splits=("Train", "Val"))
    questions = discover_questions(root, splits=("Train", "Val"))
    summary = summarize(images, questions)
    assert set(summary["questions_per_split"]) == {"Train"}
    assert "Val" not in summary["questions_per_split"]
    problems = warnings_for(summary)
    assert not any("'Val'" in p for p in problems), problems


def test_zero_question_warning_is_absent_on_a_healthy_corpus(
    tmp_path: Path,
) -> None:
    """The negative case: every split has questions, so nothing fires."""
    root = tmp_path / "cdvqa"
    write_split(
        root,
        "Train",
        images=[image_row(0, "a.png", [0, 1])],
        questions=[
            question_row(0, 0, "change_or_not", [0]),
            question_row(1, 0, "change_or_not", [1]),
        ],
        answers=[answer_row(0, 0, "yes"), answer_row(1, 1, "no")],
    )
    images = discover_cdvqa(root, splits=("Train",))
    questions = discover_questions(root, splits=("Train",))
    summary = summarize(images, questions)
    assert summary["questions_per_split"] == {"Train": 2}
    problems = warnings_for(summary)
    assert not any("zero questions" in p for p in problems), problems


def test_a_healthy_corpus_produces_no_warnings(tmp_path: Path) -> None:
    """The clean baseline: a two-split corpus with full coverage is silent."""
    root = tmp_path / "cdvqa"
    for split in ("Train", "Val"):
        write_split(
            root,
            split,
            images=[image_row(0, f"{split}.png", [0, 1])],
            questions=[
                question_row(0, 0, "change_or_not", [0]),
                question_row(1, 0, "change_or_not", [1]),
            ],
            answers=[answer_row(0, 0, "yes"), answer_row(1, 1, "no")],
        )
    images = discover_cdvqa(root, splits=("Train", "Val"))
    questions = discover_questions(root, splits=("Train", "Val"))
    png_dir = tmp_path / "png"
    png_dir.mkdir()
    for image in images:
        (png_dir / image.file_name).write_bytes(b"\x89PNG\r\n\x1a\nstub")

    problems = warnings_for(summarize(images, questions, image_root=png_dir))
    assert problems == [], problems


def test_drop_counts_is_none_without_the_drops_argument(cdvqa_root: Path) -> None:
    images, questions = _resolved(cdvqa_root)
    assert summarize(images, questions)["drop_counts"] is None


def test_drop_counts_carries_every_counter_per_split(tmp_path: Path) -> None:
    root = tmp_path / "cdvqa"
    _drop_fixture(root)
    corpus = load_cdvqa(root, splits=("Train",))
    summary = summarize(corpus.images, corpus.questions, drops=corpus.drops)
    assert summary["drop_counts"] is not None
    counts = summary["drop_counts"]["Train"]
    assert {
        "declared_question_refs",
        "dangling_reference_count",
        "unknown_image_count",
        "multi_answer_count",
        "answerless_count",
    } <= set(counts)


def test_drop_counts_sum_matches_the_proxy_when_only_dangling(cdvqa_root: Path) -> None:
    corpus = load_cdvqa(cdvqa_root, splits=("Train",))
    with_drops = summarize(
        corpus.images, corpus.questions, drops=corpus.drops
    )
    without_drops = summarize(corpus.images, corpus.questions)
    # No drops at all in this fixture: both agree.
    assert with_drops["dangling_reference_count"] == 0
    assert without_drops["dangling_reference_count"] == 0


def test_summary_of_an_empty_corpus() -> None:
    assert summarize([], []) == {"images": 0, "questions": 0, "splits": []}


def test_examples_are_summarized_when_passed(cdvqa_root: Path) -> None:
    images, questions = _resolved(cdvqa_root)
    examples = build_cdvqa_examples(images, questions)
    summary = summarize(images, questions, examples)
    assert summary["examples_total"] == len(examples)
    assert summary["examples_per_split"] == {"Train": len(examples)}


# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------


def test_answer_vocabularies_has_exactly_the_question_types() -> None:
    assert set(ANSWER_VOCABULARIES) == set(QUESTION_TYPES)
    assert len(ANSWER_VOCABULARIES) == 8


def test_change_ratio_and_change_ratio_types_are_different_constants() -> None:
    """Similar names, different types: 11 bins vs a 9-value union."""
    assert ANSWER_VOCABULARIES["change_ratio"] is not ANSWER_VOCABULARIES[
        "change_ratio_types"
    ]
    assert CHANGE_RATIO_BINS is not CHANGE_RATIO_TYPE_VALUES
    assert len(CHANGE_RATIO_BINS) == 11
    assert len(CHANGE_RATIO_TYPE_VALUES) == 9
    assert "0" in CHANGE_RATIO_BINS
    assert "0" in CHANGE_RATIO_TYPE_VALUES
    assert "90_to_100" in CHANGE_RATIO_BINS
    assert "90_to_100" not in CHANGE_RATIO_TYPE_VALUES


def test_the_three_boolean_types_share_the_yes_no_vocabulary() -> None:
    for qtype in ("change_or_not", "increase_or_not", "decrease_or_not"):
        assert ANSWER_VOCABULARIES[qtype] is YES_NO_VALUES
        assert ANSWER_VOCABULARIES[qtype] == ("yes", "no")


def test_the_three_six_class_types_share_the_change_classes() -> None:
    for qtype in ("change_to_what", "smallest_change", "largest_change"):
        assert ANSWER_VOCABULARIES[qtype] is CHANGE_CLASSES
    assert len(CHANGE_CLASSES) == 6


def test_split_names_are_the_four_shipped_splits() -> None:
    assert CDVQA_SPLITS == ("Train", "Val", "Test", "Test2")


def test_error_carries_the_documented_code() -> None:
    assert issubclass(CDVQAError, SatQueryError)
    assert CDVQAError.code == "cdvqa_error"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("5", 5),
        (5, 5),
        (3.7, 3),
        ("abc", None),
        (None, None),
        ([], None),
        ({}, None),
    ],
)
def test_as_int_coerces_or_returns_none(value, expected) -> None:
    assert _as_int(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1.5", 1.5),
        (2, 2.0),
        (1631214265.2322025, 1631214265.2322025),
        ("abc", None),
        (None, None),
        ([], None),
        ({}, None),
    ],
)
def test_as_float_coerces_or_returns_none(value, expected) -> None:
    assert _as_float(value) == expected


# ---------------------------------------------------------------------------
# Import hygiene
# ---------------------------------------------------------------------------


def test_training_data_package_reexports_cdvqa_names() -> None:
    import training.data as td

    assert td.CDVQAError is CDVQAError
    assert td.CDVQAImage is CDVQAImage
    assert td.CDVQAQuestion is CDVQAQuestion
    assert td.CDVQACorpus is CDVQACorpus
    assert td.load_cdvqa is load_cdvqa
    assert td.discover_cdvqa is discover_cdvqa
    assert td.discover_questions is discover_questions
    assert td.build_cdvqa_examples is build_cdvqa_examples
    assert td.summarize_cdvqa is summarize
    assert td.warnings_for_cdvqa is warnings_for


def test_importing_training_data_has_no_filesystem_side_effects(
    tmp_path: Path,
) -> None:
    """Import in a fresh interpreter, in an empty cwd, and prove nothing landed.

    Note on the environment this test builds. It sets `PYTHONPATH` to the repo
    root *alone*, which **discards any site-packages the parent process was
    reaching via `PYTHONPATH`**. That is usually harmless -- in a normal virtual
    environment third-party packages are on the interpreter's own `sys.path` and
    survive the override. But when a runner reaches its dependencies *through*
    `PYTHONPATH` (as this sandbox does, because it cannot execute the project
    venv's interpreter directly), the child sees only the repo and
    `import numpy` fails with `ModuleNotFoundError`.

    That failure is a **harness artifact, not a code defect**: the property under
    test is "the import writes nothing into the cwd", and it is unaffected by
    which `sys.path` entry supplied numpy. Verified directly -- re-running the
    same child command with the site-packages appended gives `rc=0` and
    `imported`.

    The `PYTHONPATH` is therefore *extended* rather than replaced. This is a
    strictly better test in every environment: it still overrides nothing about
    the repo path, and it no longer depends on how the runner obtained its
    dependencies.
    """
    env = dict(os.environ)
    inherited = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT), inherited) if part
    )
    proc = subprocess.run(
        [sys.executable, "-c", "import training.data; print('imported')"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "imported" in proc.stdout
    assert list(tmp_path.iterdir()) == [], (
        f"importing training.data wrote into the cwd: {list(tmp_path.iterdir())}"
    )
