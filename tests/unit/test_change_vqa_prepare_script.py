"""R-02 — `prepare_record.json` must be the provenance of the whole directory.

WHY THIS FILE EXISTS
--------------------
A full preparation is several invocations with different `--skip-*` flags: the
Kaggle notebook builds targets, then change features, then question features, and
finally re-runs over all four splits with `--skip-text` to extend the targets. The
record was overwritten by each pass, so the last pass — which skipped the question
features — produced a record with NO `text_features` at all.

That is the same failure shape as the one that stopped the second Kaggle run: a
later step silently discarding an earlier step's output. It was invisible for the
same reason — nothing reads the record during the run, so the loss only shows up
for the reviewer who opens it afterwards and concludes the question features were
never built, which is exactly what the record is supposed to rule out.

The sections a pass does not write are now carried forward: single-artifact
sections keep the last pass that wrote them, and the per-split `text_features`
mapping is merged. `written_this_pass` and `carried_forward` name the difference.

These tests drive `_carry_forward` through the real pass sequence, on disk, so
what is proven is the record a reviewer would actually open.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_change_vqa import (
    _PROVENANCE_SECTIONS,
    _carry_forward,
    main,
)
from training.change_vqa.dataset import read_scene_targets


@pytest.fixture()
def scratch() -> Path:
    # `tempfile.mkdtemp`, never pytest's `tmp_path`: this sandbox blocks the
    # directory removals `tmp_path` performs at teardown. The directory is left
    # in place on purpose.
    return Path(tempfile.mkdtemp(prefix="r02_prepare_record_"))


def _run_pass(record_path: Path, **sections) -> dict:
    """One prepare invocation: carry forward, then write, exactly as `prepare`."""
    result = _carry_forward(
        record_path, {"data_root": "synthetic", "argv": [], **sections}
    )
    record_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def _reopen(record_path: Path) -> dict:
    return json.loads(record_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The regression: the notebook's real pass order
# ---------------------------------------------------------------------------


def test_the_question_features_of_every_split_survive_the_pass_sequence(
    scratch: Path,
) -> None:
    """The whole hole, replayed in the order the notebook runs.

    Passes 4 and 5 both skip `--skip-text`, so before the fix the record the
    reviewer opens has no `text_features` key at all.
    """
    path = scratch / "prepare_record.json"

    # cell 19 — targets for the training splits
    _run_pass(path, targets={"path": "scene_targets.jsonl", "scenes": 2000})
    # cell 21 — change features for the training splits
    _run_pass(path, change_features={"path": "change_features.npz", "scenes": 2000})
    # cell 23 — question features for the training splits
    _run_pass(
        path,
        text_features={
            "Train": {"questions": 25935},
            "Val": {"questions": 10000},
        },
    )
    # cell 27a — question features for the held-out splits
    _run_pass(
        path,
        text_features={
            "Test": {"questions": 39686},
            "Test2": {"questions": 31036},
        },
    )
    # cell 27b — targets and change features over all four splits, --skip-text
    final = _run_pass(
        path,
        targets={"path": "scene_targets.jsonl", "scenes": 2968},
        change_features={"path": "change_features.npz", "scenes": 2968},
    )

    assert set(final["text_features"]) == {"Train", "Val", "Test", "Test2"}
    assert final["targets"]["scenes"] == 2968
    assert final["change_features"]["scenes"] == 2968
    assert final["written_this_pass"] == ["targets", "change_features"]
    assert "text_features" in final["carried_forward"]

    # And it must still be there after the write, since that is what a reviewer
    # reads off the exported tree.
    assert set(_reopen(path)["text_features"]) == {"Train", "Val", "Test", "Test2"}


def test_a_later_pass_keeps_a_single_artifact_it_skipped(scratch: Path) -> None:
    """`--skip-features` on a targets-only pass must not erase the feature
    provenance, because the npz is still on disk."""
    path = scratch / "prepare_record.json"
    _run_pass(
        path,
        change_features={"path": "change_features.npz", "scenes": 2000,
                         "spec_hash": "c801326f85a185f8"},
    )
    final = _run_pass(path, targets={"scenes": 2968})

    assert final["change_features"]["spec_hash"] == "c801326f85a185f8"
    assert "change_features" in final["carried_forward"]
    assert final["written_this_pass"] == ["targets"]


# ---------------------------------------------------------------------------
# What must NOT be carried
# ---------------------------------------------------------------------------


def test_a_single_artifact_section_is_replaced_wholesale_not_merged(
    scratch: Path,
) -> None:
    """A pass that writes `targets` describes the file that is now on disk. Merging
    field by field would mix two passes' numbers into one description of one file
    — a record that is wrong in a way no reader can detect."""
    path = scratch / "prepare_record.json"
    _run_pass(path, targets={"path": "a.jsonl", "scenes": 2000, "mean": 0.31})
    final = _run_pass(path, targets={"scenes": 2968})

    assert final["targets"] == {"scenes": 2968}
    assert "targets" not in final.get("carried_forward", [])


def test_nothing_is_carried_on_a_first_pass(scratch: Path) -> None:
    path = scratch / "prepare_record.json"
    final = _run_pass(path, targets={"scenes": 2968})

    assert "carried_forward" not in final
    assert final["written_this_pass"] == ["targets"]


def test_carrying_forward_does_not_invent_sections_that_never_existed(
    scratch: Path,
) -> None:
    """A record must not gain a section because a LATER run happened to write it
    somewhere else. Only a real previous record contributes."""
    path = scratch / "prepare_record.json"
    final = _run_pass(path, targets={"scenes": 2968})

    for absent in ("change_features", "text_features", "manifest"):
        assert absent not in final


def test_a_split_this_pass_did_not_touch_is_reported_as_carried(
    scratch: Path,
) -> None:
    """Both passes write `text_features`; only the second one's key is new. The
    carried flag has to distinguish the two, or a reviewer cannot tell which run
    produced which split."""
    path = scratch / "prepare_record.json"
    _run_pass(path, text_features={"Train": {"questions": 1}})
    final = _run_pass(path, text_features={"Test": {"questions": 2}})

    assert set(final["text_features"]) == {"Train", "Test"}
    assert "text_features" in final["carried_forward"]


def test_a_rewritten_split_alone_is_not_reported_as_carried(scratch: Path) -> None:
    """The control for the test above: when this pass covers every key the
    previous one had, nothing was carried."""
    path = scratch / "prepare_record.json"
    _run_pass(path, text_features={"Train": {"questions": 1}})
    final = _run_pass(path, text_features={"Train": {"questions": 2}})

    assert final["text_features"] == {"Train": {"questions": 2}}
    assert "carried_forward" not in final


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_an_unreadable_previous_record_does_not_fail_the_pass(scratch: Path) -> None:
    """A truncated record — an interrupted run — must not stop the next pass. The
    artifact it was describing is still on disk and still usable."""
    path = scratch / "prepare_record.json"
    path.write_text('{"targets": {"scenes": 2000', encoding="utf-8")  # truncated

    final = _run_pass(path, change_features={"scenes": 2968})

    assert final["change_features"]["scenes"] == 2968
    assert "carried_forward" not in final


def test_a_previous_section_of_the_wrong_shape_is_ignored(scratch: Path) -> None:
    """A hand-edited or older-format record must not raise inside the merge."""
    path = scratch / "prepare_record.json"
    path.write_text(
        json.dumps({"text_features": ["Train", "Val"], "targets": 2000}),
        encoding="utf-8",
    )

    final = _run_pass(path, change_features={"scenes": 2968})

    # `targets` is a replace section and this pass did not write it, so it is
    # carried as-is; `text_features` is a merge section with the wrong shape and
    # is left alone rather than crashed on.
    assert final["targets"] == 2000
    assert "text_features" not in final


def test_the_declared_sections_are_the_ones_the_record_can_hold() -> None:
    """A guard against the mapping and the writer drifting apart: every declared
    section is one `prepare` can actually produce."""
    assert set(_PROVENANCE_SECTIONS) == {
        "manifest", "targets", "change_features", "text_features",
    }
    assert set(_PROVENANCE_SECTIONS.values()) <= {"replace", "merge"}


# ---------------------------------------------------------------------------
# K4 at the ARTIFACT level: the real script, several real passes, on disk
# ---------------------------------------------------------------------------
#
# The tests above drive `_carry_forward` directly, so they prove the record.
# They do NOT prove what the passes leave in `scene_targets.jsonl`, because
# nothing above runs `prepare` itself. That is the gap this closes.
#
# The defect it reproduces is the one that stopped the second Kaggle run:
# `write_scene_targets` OVERWRITES the file, so the only pass that can extend the
# targets to the held-out splits is a pass that does NOT skip them. Every pass
# after the first skipped them, so the file stayed at Train/Val and the evaluator
# reported every held-out question as `missing_targets`.

_HELD_OUT = ("Test", "Test2")
_TRAINING = ("Train", "Val")
_SIMPLE_QUESTIONS = (
    ("change_or_not", "Have the areas of buildings changed?", ("yes", "no")),
    ("increase_or_not", "Has the area of water increased?", ("yes", "no")),
    ("decrease_or_not", "Has the area of trees decreased?", ("yes", "no")),
)


def _write_split(
    root: Path,
    split: str,
    scenes: tuple[str, ...],
    questions: tuple[tuple[str, str, tuple[str, ...]], ...] = _SIMPLE_QUESTIONS,
) -> None:
    """The three annotation tables for one split, in the shipped shape.

    `question_id` restarts at 0 for every split, as in the real corpus.

    `questions` is overridable so a split can be given scenes but no questions —
    the one input state that makes `prepare` write no question-feature cache at
    all, which the evaluator refuses to run on.
    """
    annotations = root / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    images, question_rows, answer_rows = [], [], []
    question_id = answer_id = 0
    for scene_index, scene in enumerate(scenes):
        refs = []
        for q_index, (qtype, text, options) in enumerate(questions):
            question_rows.append({
                "id": question_id, "date_added": 1631214265.0, "img_id": scene_index,
                "type": qtype, "question": text, "answers_ids": [answer_id],
                "active": True,
            })
            answer_rows.append({
                "id": answer_id, "date_added": 1631214265.0,
                "question_id": question_id,
                "answer": options[(scene_index + q_index) % len(options)],
                "active": True,
            })
            refs.append(question_id)
            question_id += 1
            answer_id += 1
        images.append({
            "id": scene_index, "res_x": "64", "res_y": "64",
            "questions_ids": refs, "file_name": f"{scene}.png", "active": True,
        })
    for name, key, rows in (
        ("images", "images", images),
        ("questions", "questions", question_rows),
        ("answers", "answers", answer_rows),
    ):
        (annotations / f"{split}_{name}.json").write_text(
            json.dumps({key: rows}), encoding="utf-8"
        )


def _four_split_root(root: Path) -> dict[str, tuple[str, ...]]:
    """A four-split CDVQA root. Every split is READABLE, so held-out targets can
    be built -- which is the whole point of the pass under test."""
    scenes = {
        "Train": ("00001", "00002"),
        "Val": ("00011", "00012"),
        "Test": ("zztest0001", "zztest0002"),
        "Test2": ("zztest20001", "zztest20002"),
    }
    from PIL import Image

    from training.change_vqa import dataset as D

    rng = np.random.default_rng(20260921)
    for split, names in scenes.items():
        _write_split(root, split, names)
    for index, scene in enumerate(sum(scenes.values(), ())):
        base = rng.integers(0, 256, size=(64, 64, 3))
        for sub, pixels in (("im1", base), ("im2", np.roll(base, 3 + index, axis=0))):
            path = root / sub / f"{scene}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode="RGB").save(path)
        for sub, variant in (("label1", 0), ("label2", 2 + index % 4)):
            colours = [D.BACKGROUND_RGB] * 64
            for pixel in range(8):
                colours[pixel] = D.LABEL_PALETTE["buildings"]
            for pixel in range(8, 14):
                colours[pixel] = D.LABEL_PALETTE["water"]
            for pixel in range(14, 14 + variant):
                colours[pixel] = D.LABEL_PALETTE["trees"]
            image = Image.new("RGB", (8, 8))
            image.putdata(colours)
            path = root / sub / f"{scene}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
    return scenes


def _prepare(data_root: Path, prepared: Path, *argv: str) -> int:
    """One real `prepare_change_vqa.py` invocation, in process.

    `--limit-per-split` is the documented subsample flag: it sets
    `expect_full_split=False`, relaxing the corpus SIZE check (2,968 scenes) and
    nothing else. A synthetic fixture cannot have 1,600 Train scenes and the gate
    must not be weakened for the real run, so it is relaxed here and only here.
    """
    return main([
        "--data-root", str(data_root), "--out-dir", str(prepared),
        "--limit-per-split", "2", "--skip-manifest", *argv,
    ])


def test_a_four_split_pass_is_what_builds_the_held_out_targets(scratch: Path) -> None:
    """K4, end to end through the real script.

    Three passes over one directory. The middle one is the bug exactly as the
    second Kaggle run had it: the right four splits, but `--skip-targets`, so
    `scene_targets.jsonl` is never rewritten and the held-out labels stay absent.
    """
    data_root = scratch / "data"
    prepared = scratch / "prepared"
    data_root.mkdir(parents=True)
    prepared.mkdir(parents=True)
    scenes = _four_split_root(data_root)

    def target_scenes() -> set[str]:
        targets = read_scene_targets(prepared / "scene_targets.jsonl")
        return {t.scene_key for t in targets.values()}

    # --- pass 1: the training splits, as the notebook's section 8 does -------
    assert _prepare(
        data_root, prepared, "--splits", *_TRAINING,
        "--skip-features", "--skip-text",
    ) == 0
    train_val = set(scenes["Train"]) | set(scenes["Val"])
    assert target_scenes() == train_val, (
        "a Train/Val targets pass must produce exactly the Train/Val scenes"
    )

    # --- pass 2, the BUG: all four splits, but targets skipped ---------------
    assert _prepare(
        data_root, prepared, "--splits", *_TRAINING, *_HELD_OUT,
        "--skip-targets", "--skip-features", "--skip-text",
    ) == 0
    bug = target_scenes()
    assert bug == train_val, (
        "a four-split pass that skips targets must leave the file untouched"
    )
    for held in _HELD_OUT:
        assert not (set(scenes[held]) & bug), (
            f"{held} targets appeared even though the pass skipped them -- the "
            "skip flag is not being honoured"
        )

    # --- pass 3, the FIX: the same four splits, targets ENABLED --------------
    assert _prepare(
        data_root, prepared, "--splits", *_TRAINING, *_HELD_OUT,
        "--skip-features", "--skip-text",
    ) == 0
    fixed = target_scenes()
    assert fixed == set().union(*(set(v) for v in scenes.values())), (
        "the four-split pass must build the held-out targets too"
    )
    # And it must not have LOST the training rows -- a held-out-only call would,
    # because `write_scene_targets` overwrites the whole file.
    assert set(scenes["Train"]) | set(scenes["Val"]) <= fixed, (
        "the four-split pass dropped the training splits' targets; a "
        "held-out-only call would do exactly that, which is why the fix extends "
        "the all-splits call instead of adding a Test/Test2-only one"
    )


def test_a_second_feature_pass_does_not_lose_the_first_passs_scenes(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SAME failure shape as K4, one artifact over.

    Pass 2 extracts change features for Train/Val; pass 5 asks for all four
    splits. `change_features.npz` is resumable, so pass 5 must KEEP the cached
    scenes and append the new ones. If the merge dropped the existing rows,
    every Train/Val record would report `missing_change_feature` at scoring
    time — a silent wrong artifact, exactly like the missing targets, and the
    evaluator's coverage guard would NOT see it (it compares the cache against
    the targets, not the records against the cache).

    The extractor is STUBBED. This pins the pending/merge bookkeeping — the part
    that can silently lose data — without paying for 5,936 detector forwards. The
    real extractor path is exercised by the smoke run and by the Kaggle log.
    """
    from training.change_vqa.features import ChangeFeatureCache

    from scripts.prepare_change_vqa import CHANGE_FEATURE_DIM
    import scripts.prepare_change_vqa as P

    class _StubExtractor:
        trained = True
        spec_hash = "stub-spec-hash-for-the-merge-test"

        def config_dict(self) -> dict:
            return {"stub": True}

    def _stub_extract(extractor, pending, on_progress=None):
        keys = tuple(triple[0] for triple in pending)
        return ChangeFeatureCache(
            scene_keys=keys,
            features=np.zeros((len(keys), CHANGE_FEATURE_DIM), dtype=np.float32),
            spec_hash=extractor.spec_hash,
            extractor_config=extractor.config_dict(),
        )

    monkeypatch.setattr(P, "build_change_feature_extractor", lambda *a, **k: _StubExtractor())
    monkeypatch.setattr(P, "extract_change_features", _stub_extract)

    data_root = scratch / "data"
    prepared = scratch / "prepared"
    data_root.mkdir(parents=True)
    prepared.mkdir(parents=True)
    scenes = _four_split_root(data_root)
    train_val = set(scenes["Train"]) | set(scenes["Val"])
    all_scenes = set().union(*(set(v) for v in scenes.values()))

    def cached() -> set[str]:
        return set(ChangeFeatureCache.read(prepared / "change_features.npz").scene_keys)

    # --- pass 2: change features for the training splits ---------------------
    assert _prepare(
        data_root, prepared, "--splits", *_TRAINING,
        "--skip-targets", "--skip-text",
    ) == 0
    assert cached() == train_val, (
        "the first feature pass must cache exactly the training splits"
    )

    # --- pass 5: the same artifact, now over all four splits -----------------
    assert _prepare(
        data_root, prepared, "--splits", *_TRAINING, *_HELD_OUT,
        "--skip-targets", "--skip-text",
    ) == 0
    after = cached()
    assert after == all_scenes, (
        "the second feature pass must APPEND to the resumable cache, not replace "
        "it -- losing the training splits here is the same defect as losing the "
        "held-out targets, and it surfaces the same way"
    )
    assert train_val <= after, "the first pass's cached scenes were dropped"


def test_a_split_with_no_questions_writes_no_text_cache_and_records_that(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one input state the evaluator refuses to run on, made visible here.

    `prepare` writes one question-feature cache per split. A split that resolves
    no questions writes **no file at all** — and `evaluate_change_vqa.py` now
    treats a requested split with no cache as an input error rather than skipping
    it. So the absence must be reported where it happens, not twenty minutes into
    an evaluation.

    The encoder is STUBBED: this pins the per-split bookkeeping and the warning,
    not MiniLM, which takes ~21 s to load.
    """
    import scripts.prepare_change_vqa as P
    from training.change_vqa.features import TEXT_FEATURE_DIM

    class _StubEncoder:
        @classmethod
        def build(cls, config, device="cpu"):  # noqa: ANN001, ANN206
            return cls()

        def encode(self, questions):  # noqa: ANN001, ANN201
            return np.zeros((len(questions), TEXT_FEATURE_DIM), dtype=np.float32)

    monkeypatch.setattr(P, "TextFeatureExtractor", _StubEncoder)

    data_root = scratch / "data"
    prepared = scratch / "prepared"
    data_root.mkdir(parents=True)
    prepared.mkdir(parents=True)
    scenes = _four_split_root(data_root)
    # Test2 keeps its scenes and its imagery, and loses every question.
    _write_split(data_root, "Test2", scenes["Test2"], questions=())

    assert _prepare(
        data_root, prepared, "--splits", *_TRAINING, *_HELD_OUT,
        "--skip-targets", "--skip-features",
    ) == 0

    for split in ("Train", "Val", "Test"):
        assert (prepared / f"text_features_{split}.npz").exists(), (
            f"{split} has questions and must have a cache"
        )
    assert not (prepared / "text_features_Test2.npz").exists(), (
        "a split with no questions must write NO cache -- the evaluator refuses "
        "to score it, so pretending otherwise would be worse than failing"
    )

    record = json.loads(
        (prepared / "prepare_record.json").read_text(encoding="utf-8")
    )
    assert record["text_features_empty_splits"] == ["Test2"], (
        "the empty split must be recorded, not merely printed -- this is the "
        "state that stops the evaluation, so a reviewer needs it in the "
        "provenance file"
    )
    assert "Test2" not in record["text_features"], (
        "a split with no cache must not be claimed in the record"
    )
    assert sorted(record["text_features"]) == ["Test", "Train", "Val"]


# ---------------------------------------------------------------------------
# A scene with imagery but no labels must fail LOUDLY, and atomically
# ---------------------------------------------------------------------------
#
# The K4 class, one layer down. `require_images=True` selects scenes that have
# both `im1` and `im2`; the targets pass then reads `label1`/`label2` for each of
# them. A scene that has imagery and no labels is therefore *selected* and then
# *unreadable*, and the question is whether that shortens `scene_targets.jsonl`
# quietly or stops the run.
#
# It stops the run -- `decode_label_map` raises `ChangeVQAError`, which `main`
# turns into `EXIT_DATA`. That is the right behaviour, and it is load-bearing for
# the notebook's coverage guard: the guard asserts
# `set(change_cache.scene_keys) <= target_scenes`, which is only sound if a pass
# cannot leave a target file that is silently short. If a partial file were
# possible, the guard would fire on a correct run and the fix would block the very
# evaluation it was written to enable.
#
# The atomicity half matters as much as the exit code. `write_scene_targets` is
# called ONCE, after the loop over scenes, so a failure part-way leaves whatever
# was there before -- never a half-written file. A truncated targets file would be
# worse than no file at all, because it is well-formed and the evaluator would
# report the loss as a partial `missing_targets` shortfall rather than a cause.

_MISSING = ("Test2", "zztest20001")


def _prepare_without_labels(
    scratch: Path, *, which: tuple[str, str] = _MISSING
) -> tuple[int, Path, Path, dict[str, tuple[str, ...]]]:
    """A four-split root with one scene's labels deleted, prepared once.

    Returns `(rc, data_root, prepared, scenes)`.
    """
    data_root = scratch / "data"
    prepared = scratch / "prepared"
    data_root.mkdir(parents=True)
    prepared.mkdir(parents=True)
    scenes = _four_split_root(data_root)
    split, scene_key = which
    assert scene_key in scenes[split], f"{scene_key} is not a {split} scene"
    for sub in ("label1", "label2"):
        path = data_root / sub / f"{scene_key}.png"
        assert path.exists(), f"fixture is wrong: {path} does not exist"
        path.unlink()
    # The IMAGERY is deliberately left in place: that is what makes the scene
    # selectable by `require_images=True` while its labels are unreadable.
    for sub in ("im1", "im2"):
        assert (data_root / sub / f"{scene_key}.png").exists()
    rc = _prepare(
        data_root, prepared, "--splits", *_TRAINING, *_HELD_OUT,
        "--skip-features", "--skip-text",
    )
    return rc, data_root, prepared, scenes


def test_a_scene_with_imagery_but_no_labels_stops_the_run(scratch: Path) -> None:
    from scripts.prepare_change_vqa import EXIT_DATA

    rc, _data_root, prepared, _scenes = _prepare_without_labels(scratch)

    assert rc == EXIT_DATA, (
        f"an unreadable label is a DATA problem (exit {EXIT_DATA}), not success "
        f"and not a crash; got {rc}"
    )
    assert not (prepared / "scene_targets.jsonl").exists(), (
        "no targets file may be written when a selected scene has no labels -- a "
        "silently short file is the K4 failure shape one layer down, and the "
        "notebook's coverage guard is only sound because this cannot happen"
    )


def test_the_failure_names_the_scene_whose_labels_are_missing(
    scratch: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The message must name the file, so the fix is one `ls` away."""
    _prepare_without_labels(scratch)

    err = capsys.readouterr().err
    split, scene_key = _MISSING
    assert f"{scene_key}.png" in err, err
    assert "label map not found" in err, err
    assert split  # the fixture's premise; kept so the constant cannot drift


def test_a_failed_pass_leaves_an_EARLIER_targets_file_intact(scratch: Path) -> None:
    """Atomicity across passes, which is the case the notebook actually hits.

    The notebook prepares Train/Val first (section 8), then re-runs over all four
    splits in section 12. If the second pass fails, the first pass's file must
    survive **unchanged** -- the same principle as K5, where a later pass must not
    destroy an earlier one's output.
    """
    data_root = scratch / "data"
    prepared = scratch / "prepared"
    data_root.mkdir(parents=True)
    prepared.mkdir(parents=True)
    scenes = _four_split_root(data_root)

    # --- pass 1: Train/Val, which succeeds --------------------------------
    assert _prepare(
        data_root, prepared, "--splits", *_TRAINING,
        "--skip-features", "--skip-text",
    ) == 0
    targets_path = prepared / "scene_targets.jsonl"
    before = targets_path.read_bytes()
    assert read_scene_targets(targets_path), "pass 1 must have written targets"

    # --- break a held-out scene's labels ----------------------------------
    for sub in ("label1", "label2"):
        (data_root / sub / f"{_MISSING[1]}.png").unlink()

    # --- pass 2: all four splits, which fails ------------------------------
    assert _prepare(
        data_root, prepared, "--splits", *_TRAINING, *_HELD_OUT,
        "--skip-features", "--skip-text",
    ) != 0

    assert targets_path.read_bytes() == before, (
        "a failed pass must leave the previous targets file byte-identical; a "
        "truncated rewrite would be well-formed and would report the loss as a "
        "partial shortfall rather than a cause"
    )
    assert set(scenes["Train"]) | set(scenes["Val"]) == {
        t.scene_key for t in read_scene_targets(targets_path).values()
    }, "the surviving file must still be exactly the training splits"
