"""R-02 test areas E-J — the CDVQA dataset layer.

    E  label-map decoding and target arithmetic   (the supervision is computed)
    F  scene-target serialisation round trip
    G  record construction contract               (the adapter's exact keys)
    H  split-integrity gate                       (the mis-pairing defect)
    I  leakage reporting and the leakage assertion
    J  manifest shape and the safe split accessor

WHY H IS THE MOST IMPORTANT AREA HERE
-------------------------------------
`build_cdvqa_examples` indexes questions by `question_id`, which is PER-SPLIT
SEQUENTIAL, so a multi-split call lets a later split overwrite an earlier one and
routes questions to imagery belonging to another split. Nothing crashes: the
records still train, still evaluate, and still produce a number.

**That defect is now fixed at its root** — the index is keyed on
`(split, question_id)` — and the fix is pinned on the real corpus by
`test_j_one_multi_split_call_pairs_every_question_with_its_own_split` at the end
of this file (Train 65,967/1,600, Val 16,441/400, together 82,408/2,000).

`verify_split_integrity` remains the second line of defence, and is still tested
by being FED the defect -- a record set with two splits sharing scenes -- and
required to report it and raise. A gate that has never been shown to fire is not
a gate, and a root-cause fix is not a reason to let a working gate go untested.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from evaluation.manifests import DatasetManifest, SampleRecord
from training.change_vqa import dataset as D
from training.change_vqa import vocab as V

REPO_ROOT = Path(__file__).resolve().parents[2]
CDVQA_ROOT = REPO_ROOT / "data" / "cdvqa"
ANNOTATIONS = CDVQA_ROOT / "annotations"


@pytest.fixture
def scratch() -> Path:
    """A self-managed scratch directory.

    `tmp_path` is deliberately NOT used. pytest's `tmp_path` performs directory
    removals, and this repository's test runs have been blocked by a sandbox
    delete guard; a `mkdtemp` directory that is never deleted needs no delete
    permission, so these tests run in a restricted environment. The directory is
    left for the OS temp reaper rather than removed here.
    """
    return Path(tempfile.mkdtemp(prefix="sq_r02_dataset_"))

_RGB_BACKGROUND = (255, 255, 255)
_RGB_BUILDINGS = (128, 0, 0)
_RGB_WATER = (0, 0, 255)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_label_png(path: Path, colours: list[tuple[int, int, int]], size: int = 4) -> Path:
    """Write a `size x size` PNG from a flat list of RGB colours."""
    from PIL import Image

    assert len(colours) == size * size
    image = Image.new("RGB", (size, size))
    image.putdata(colours)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _record(**overrides: Any) -> D.ChangeVQARecord:
    """A syntactically valid record; override only the field under test."""
    base: dict[str, Any] = {
        "split": "Train",
        "question_id": 0,
        "file_name": "00001.png",
        "scene_key": "00001",
        "qtype": "change_or_not",
        "qtype_index": V.QUESTION_TYPE_TO_INDEX["change_or_not"],
        "resolved_qtype": "change_or_not",
        "qtype_resolved": True,
        "temporal_ref": "unspecified",
        "change_class": None,
        "question": "Have the areas of buildings changed?",
        "answer": "yes",
        "answer_index": V.ANSWER_TO_INDEX["yes"],
        "t1_path": "/data/im1/00001.png",
        "t2_path": "/data/im2/00001.png",
    }
    base.update(overrides)
    return D.ChangeVQARecord(**base)


def _split_records(split: str, scene_keys: list[str], *, n_questions: int = 1):
    """A consistent little record set for one native split."""
    out: list[D.ChangeVQARecord] = []
    for scene_key in scene_keys:
        for q in range(n_questions):
            out.append(
                _record(
                    split=split,
                    scene_key=scene_key,
                    file_name=f"{scene_key}.png",
                    question_id=q,
                    t1_path=f"/d/im1/{scene_key}.png",
                    t2_path=f"/d/im2/{scene_key}.png",
                )
            )
    return out


def _manifest_record(native_split: str, scene_key: str) -> SampleRecord:
    return SampleRecord(
        dataset_id=D.DATASET_ID,
        sample_id=f"{native_split}:{scene_key}",
        split=D.SPLIT_MAP[native_split],
        scene_id=scene_key,
        source_scene=f"{scene_key}.png",
        sensor="second",
        path=f"/d/im1/{scene_key}.png",
        extra={"native_split": native_split, "n_questions": 1},
    )


# ===========================================================================
# Area E - label-map decoding and target arithmetic
# ===========================================================================
def test_e_decodes_the_palette_and_the_shared_background(scratch: Path) -> None:
    """White is a SHARED BACKGROUND (-1), not a seventh class."""
    colours = [_RGB_BACKGROUND] * 16
    colours[0] = _RGB_BUILDINGS
    colours[5] = _RGB_WATER
    path = _write_label_png(scratch / "label1" / "00001.png", colours)

    decoded = D.decode_label_map(path)
    assert decoded.shape == (4, 4)
    assert decoded[0, 0] == V.CLASS_TO_INDEX["buildings"]
    assert decoded[1, 1] == V.CLASS_TO_INDEX["water"]
    assert decoded[3, 3] == -1
    assert int((decoded == -1).sum()) == 14


def test_e_every_palette_colour_maps_to_its_declared_class(scratch: Path) -> None:
    colours = [_RGB_BACKGROUND] * 16
    for index, (_name, rgb) in enumerate(D.LABEL_PALETTE.items()):
        colours[index] = rgb
    path = _write_label_png(scratch / "label1" / "00002.png", colours)
    decoded = D.decode_label_map(path)
    for index, (name, _rgb) in enumerate(D.LABEL_PALETTE.items()):
        assert decoded.reshape(-1)[index] == V.CLASS_TO_INDEX[name], name


def test_e_an_unknown_colour_is_a_defect_not_a_background_pixel(scratch: Path) -> None:
    """Silently mapping it to background would delete changed pixels from the
    supervision and quietly weaken the target."""
    colours = [_RGB_BACKGROUND] * 16
    colours[7] = (7, 7, 7)
    path = _write_label_png(scratch / "label1" / "00003.png", colours)
    with pytest.raises(D.ChangeVQAError, match="outside the frozen palette"):
        D.decode_label_map(path)


def test_e_a_missing_label_map_raises(scratch: Path) -> None:
    with pytest.raises(D.ChangeVQAError, match="not found"):
        D.decode_label_map(scratch / "nope.png")


def test_e_target_arithmetic_is_exact_on_a_hand_computed_scene(scratch: Path) -> None:
    """Four buildings pixels become water in a 4x4 scene.

    Hand-computed expectations:
        buildings  in1=4 in2=0 -> gained 0, lost 4 -> mag 4/16=0.25, delta -0.25
                                                    -> ratio 4/4 = 1.0
        water      in1=0 in2=4 -> gained 4, lost 0 -> mag 0.25, delta +0.25
                                                    -> ratio 4/0 -> 0.0 (no NaN)
        total_changed = 4/16 = 0.25
    """
    before = [_RGB_BACKGROUND] * 16
    for i in (0, 1, 4, 5):
        before[i] = _RGB_BUILDINGS
    after = list(before)
    for i in (0, 1, 4, 5):
        after[i] = _RGB_WATER

    _write_label_png(scratch / "label1" / "00001.png", before)
    _write_label_png(scratch / "label2" / "00001.png", after)

    targets = D.compute_scene_targets(
        scratch / "label1" / "00001.png",
        scratch / "label2" / "00001.png",
        "00001.png",
    )

    buildings = V.CLASS_TO_INDEX["buildings"]
    water = V.CLASS_TO_INDEX["water"]
    assert targets.scene_key == "00001"
    assert targets.n_pixels == 16
    assert targets.class_mag[buildings] == pytest.approx(0.25)
    assert targets.class_delta[buildings] == pytest.approx(-0.25)
    assert targets.class_ratio[buildings] == pytest.approx(1.0)
    assert targets.class_mag[water] == pytest.approx(0.25)
    assert targets.class_delta[water] == pytest.approx(0.25)
    # A class absent from the PRE map has no denominator; 0.0 is honest, NaN is not.
    assert targets.class_ratio[water] == 0.0
    assert targets.total_changed == pytest.approx(0.25)
    # Both maps label 4 pixels each.
    assert targets.n_labeled_pixels == 8
    # The remaining four classes are untouched.
    for name in ("NVG_surface", "low_vegetation", "trees", "playgrounds"):
        index = V.CLASS_TO_INDEX[name]
        assert targets.class_mag[index] == 0.0
        assert targets.class_delta[index] == 0.0


def test_e_target_vectors_are_in_range(scratch: Path) -> None:
    before = [_RGB_BACKGROUND] * 16
    after = [_RGB_BACKGROUND] * 16
    before[0] = _RGB_BUILDINGS
    after[3] = _RGB_BUILDINGS
    _write_label_png(scratch / "label1" / "00001.png", before)
    _write_label_png(scratch / "label2" / "00001.png", after)
    targets = D.compute_scene_targets(
        scratch / "label1" / "00001.png", scratch / "label2" / "00001.png", "00001.png"
    )
    assert len(targets.class_mag) == V.N_CHANGE_CLASSES
    assert all(0.0 <= v <= 1.0 for v in targets.class_mag)
    assert all(-1.0 <= v <= 1.0 for v in targets.class_delta)
    assert 0.0 <= targets.total_changed <= 1.0


def test_e_an_unchanged_scene_has_zero_change_and_no_false_delta(scratch: Path) -> None:
    colours = [_RGB_BACKGROUND] * 16
    colours[0] = _RGB_BUILDINGS
    _write_label_png(scratch / "label1" / "00001.png", colours)
    _write_label_png(scratch / "label2" / "00001.png", colours)
    targets = D.compute_scene_targets(
        scratch / "label1" / "00001.png", scratch / "label2" / "00001.png", "00001.png"
    )
    assert targets.total_changed == 0.0
    assert all(v == 0.0 for v in targets.class_mag)
    assert all(v == 0.0 for v in targets.class_delta)


def test_e_shape_mismatch_between_the_two_maps_raises(scratch: Path) -> None:
    _write_label_png(scratch / "label1" / "00001.png", [_RGB_BACKGROUND] * 16, size=4)
    _write_label_png(scratch / "label2" / "00001.png", [_RGB_BACKGROUND] * 9, size=3)
    with pytest.raises(D.ChangeVQAError, match="disagree in shape"):
        D.compute_scene_targets(
            scratch / "label1" / "00001.png",
            scratch / "label2" / "00001.png",
            "00001.png",
        )


# ===========================================================================
# Area F - scene-target serialisation
# ===========================================================================
def test_f_scene_targets_survive_a_dict_round_trip(scratch: Path) -> None:
    colours = [_RGB_BACKGROUND] * 16
    colours[0] = _RGB_BUILDINGS
    _write_label_png(scratch / "label1" / "00001.png", colours)
    colours[0] = _RGB_WATER
    _write_label_png(scratch / "label2" / "00001.png", colours)
    original = D.compute_scene_targets(
        scratch / "label1" / "00001.png", scratch / "label2" / "00001.png", "00001.png"
    )
    restored = D.SceneTargets.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original


def test_f_scene_targets_survive_a_jsonl_round_trip(scratch: Path) -> None:
    targets = D.SceneTargets(
        file_name="00001.png",
        scene_key="00001",
        class_mag=(0.1, 0.2, 0.0, 0.0, 0.3, 0.0),
        class_delta=(0.1, -0.2, 0.0, 0.0, 0.3, 0.0),
        class_ratio=(0.5, 0.25, 0.0, 0.0, 1.0, 0.0),
        total_changed=0.42,
        n_pixels=65536,
        n_labeled_pixels=131072,
        label1_sha256="a" * 64,
        label2_sha256="b" * 64,
    )
    path = D.write_scene_targets([targets], scratch / "targets.jsonl")
    loaded = D.read_scene_targets(path)
    assert set(loaded) == {"00001.png"}
    assert loaded["00001.png"] == targets


def test_f_a_missing_targets_file_raises(scratch: Path) -> None:
    with pytest.raises(D.ChangeVQAError, match="not found"):
        D.read_scene_targets(scratch / "absent.jsonl")


# ===========================================================================
# Area G - record construction contract
# ===========================================================================
def _adapter_example(**overrides: Any) -> dict[str, Any]:
    """Exactly the ten keys `build_cdvqa_examples` emits -- no more, no fewer."""
    base: dict[str, Any] = {
        "image_path": "/d/im1/00001.png",
        "image_paths": ["/d/im1/00001.png", "/d/im2/00001.png"],
        "layout": "second",
        "file_name": "00001.png",
        "scene_key": "00001",
        "split": "Train",
        "qtype": "change_or_not",
        "question": "Have the areas of buildings changed?",
        "answer": "yes",
        "question_id": 7,
    }
    base.update(overrides)
    return base


def test_g_the_adapter_contract_is_the_ten_documented_keys() -> None:
    """A field read here that the adapter does not emit would be an invented
    column -- which is exactly the `image_id` defect this contract prevents."""
    assert set(_adapter_example()) == {
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
    record = D._record_from_example(_adapter_example())
    assert record.question_id == 7
    assert record.scene_key == "00001"
    assert record.t1_path == "/d/im1/00001.png"
    assert record.t2_path == "/d/im2/00001.png"


def test_g_the_record_carries_resolved_question_metadata() -> None:
    record = D._record_from_example(
        _adapter_example(
            qtype="change_ratio_types",
            question=(
                "What is the change percentage of low vegetation in the "
                "pre-change image?"
            ),
            answer="10_to_20",
        )
    )
    assert record.qtype_index == V.QUESTION_TYPE_TO_INDEX["change_ratio_types"]
    assert record.resolved_qtype == "change_ratio_types"
    assert record.qtype_resolved is True
    assert record.temporal_ref == "pre"
    assert record.change_class == "low_vegetation"
    assert record.answer_index == V.ANSWER_TO_INDEX["10_to_20"]


def test_g_an_answer_outside_the_frozen_vocabulary_raises() -> None:
    """Admitting it would shift the index of every other answer."""
    with pytest.raises(D.ChangeVQAError, match="outside the frozen 19-answer"):
        D._record_from_example(_adapter_example(answer="maybe"))


def test_g_a_scene_that_is_not_a_temporal_pair_raises() -> None:
    for paths in ([], ["/d/im1/00001.png"], ["a", "b", "c"]):
        with pytest.raises(D.ChangeVQAError, match="exactly two temporal image paths"):
            D._record_from_example(_adapter_example(image_paths=paths))


def test_g_an_unresolvable_question_still_produces_a_record() -> None:
    """The free-form path must not be blocked by the resolver."""
    record = D._record_from_example(
        _adapter_example(
            qtype="change_or_not",
            question="How many cars are parked here?",
            answer="no",
        )
    )
    assert record.qtype_resolved is False
    assert record.resolved_qtype == V.UNKNOWN_QUESTION_TYPE
    assert record.temporal_ref == "unspecified"


def test_g_legality_of_the_answer_for_its_type_is_reported() -> None:
    legal = D._record_from_example(_adapter_example(answer="yes"))
    assert legal.is_answer_legal_for_type is True
    # A class answer on a yes/no question is not legal for its type.
    illegal = D._record_from_example(
        _adapter_example(qtype="change_or_not", answer="buildings")
    )
    assert illegal.is_answer_legal_for_type is False


# ===========================================================================
# Area H - the split-integrity gate
# ===========================================================================
def test_h_a_consistent_two_split_set_is_reported_clean() -> None:
    records = _split_records("Train", ["0001", "0002"]) + _split_records(
        "Val", ["0003"]
    )
    report = D.verify_split_integrity(
        records, expect_full_split=False, expected_splits=("Train", "Val")
    )
    assert report["is_clean"] is True
    assert report["size_errors"] == []
    assert report["leakage"] == []
    assert report["scenes_per_split"] == {"Train": 2, "Val": 1}


def test_h_the_gate_reports_leakage_when_two_splits_share_scenes() -> None:
    """THE test of area H: feed it the defect the gate exists to catch."""
    records = _split_records("Train", ["0001", "0002"]) + _split_records(
        "Val", ["0001", "0002"]
    )
    report = D.verify_split_integrity(
        records, expect_full_split=False, expected_splits=("Train", "Val")
    )
    assert report["is_clean"] is False
    assert report["leakage"] == ["Train|Val=2"]
    assert report["pairwise_scene_overlap"]["Train|Val"] == 2


def test_h_the_gate_raises_rather_than_only_reporting() -> None:
    records = _split_records("Train", ["0001"]) + _split_records("Val", ["0001"])
    with pytest.raises(D.ChangeVQAError, match="split integrity check failed"):
        D.assert_split_integrity(
            records, expect_full_split=False, expected_splits=("Train", "Val")
        )


def test_h_test_and_test2_sharing_scenes_is_allowed_by_construction() -> None:
    """A gate that failed on this would be a gate nobody could pass."""
    records = _split_records("Test", ["0001", "0002"]) + _split_records(
        "Test2", ["0001", "0002"]
    )
    report = D.verify_split_integrity(
        records, expect_full_split=False, expected_splits=("Test", "Test2")
    )
    assert report["is_clean"] is True
    assert report["pairwise_scene_overlap"]["Test|Test2"] == 2
    assert report["allowed_shared_pairs"] == [["Test", "Test2"]]


def test_h_reading_a_split_the_caller_did_not_request_is_an_error() -> None:
    records = _split_records("Train", ["0001"]) + _split_records("Test", ["0009"])
    report = D.verify_split_integrity(
        records, expect_full_split=False, expected_splits=("Train",)
    )
    assert report["unexpected_splits"] == ["Test"]
    assert report["is_clean"] is False
    with pytest.raises(D.ChangeVQAError, match="did not request"):
        D.assert_split_integrity(
            records, expect_full_split=False, expected_splits=("Train",)
        )


def test_h_a_record_whose_imagery_belongs_to_another_scene_is_caught() -> None:
    """The precise signature of the cross-split id collision."""
    records = [
        _record(
            scene_key="0001",
            file_name="0001.png",
            t1_path="/d/im1/0002.png",
            t2_path="/d/im2/0002.png",
        )
    ]
    report = D.verify_split_integrity(
        records, expect_full_split=False, expected_splits=("Train",)
    )
    assert report["records_with_inconsistent_imagery"] == 1
    assert report["is_clean"] is False
    with pytest.raises(D.ChangeVQAError, match="does not belong to the scene"):
        D.assert_split_integrity(
            records, expect_full_split=False, expected_splits=("Train",)
        )


def test_h_full_split_expectation_rejects_wrong_counts() -> None:
    records = _split_records("Train", ["0001", "0002"])
    report = D.verify_split_integrity(records, expected_splits=("Train",))
    assert report["expect_full_split"] is True
    assert report["is_clean"] is False
    assert any("expected 1600/65967" in e for e in report["size_errors"])


def test_h_a_subsample_may_undershoot_but_may_not_exceed_the_corpus() -> None:
    """A gate that cries wolf is a gate people learn to bypass."""
    under = _split_records("Train", ["0001"])
    report = D.verify_split_integrity(
        under, expect_full_split=False, expected_splits=("Train",)
    )
    assert report["is_clean"] is True

    over = _split_records("Val", [f"{i:05d}" for i in range(401)])
    report_over = D.verify_split_integrity(
        over, expect_full_split=False, expected_splits=("Val",)
    )
    assert report_over["is_clean"] is False
    assert any("EXCEEDS the corpus" in e for e in report_over["size_errors"])


def test_h_an_unknown_expected_split_is_rejected() -> None:
    with pytest.raises(D.ChangeVQAError, match="unknown expected split"):
        D.verify_split_integrity([], expected_splits=("Train", "Nope"))


# ===========================================================================
# Area I - leakage reporting
# ===========================================================================
def _leakage_manifest() -> DatasetManifest:
    records = (
        [_manifest_record("Train", f"t{i}") for i in range(2)]
        + [_manifest_record("Val", "v0")]
        + [_manifest_record("Test", "e0"), _manifest_record("Test", "e1")]
        + [_manifest_record("Test2", "e0"), _manifest_record("Test2", "e1")]
    )
    return DatasetManifest(name="cdvqa_scenes", records=records)


def test_i_leakage_report_describes_the_corpus_properties() -> None:
    report = D.leakage_report(_leakage_manifest())
    assert report["scenes_per_native_split"] == {
        "Train": 2,
        "Val": 1,
        "Test": 2,
        "Test2": 2,
    }
    assert report["train_val_disjoint"] is True
    assert report["train_test_disjoint"] is True
    assert report["val_test_disjoint"] is True
    assert report["test_test2_share_all_scenes"] is True
    assert report["pairwise_scene_overlap"]["Test|Test2"] == 2


def test_i_assert_no_leakage_passes_on_a_disjoint_train_val_test() -> None:
    D.assert_no_leakage(_leakage_manifest())


def test_i_assert_no_leakage_fires_when_a_val_scene_appears_in_train() -> None:
    manifest = _leakage_manifest()
    manifest.records.append(_manifest_record("Train", "v0"))
    with pytest.raises(D.ChangeVQAError, match="scene-level leakage"):
        D.assert_no_leakage(manifest)


def test_i_assert_no_leakage_does_not_fire_on_the_test_test2_overlap() -> None:
    """The narrower gate is the deliberate choice: Test/Test2 overlap is a
    property of the release, not a defect in it."""
    manifest = DatasetManifest(
        name="x",
        records=[
            _manifest_record("Test", "e0"),
            _manifest_record("Test2", "e0"),
        ],
    )
    D.assert_no_leakage(manifest)


# ===========================================================================
# Area J - manifest shape and the safe split accessor
# ===========================================================================
def test_j_the_safe_accessor_separates_test_from_test2() -> None:
    manifest = _leakage_manifest()
    test = D.by_native_split(manifest, "Test")
    test2 = D.by_native_split(manifest, "Test2")
    assert len(test) == len(test2) == 2
    assert {r.scene_id for r in test} == {r.scene_id for r in test2}
    assert {r.sample_id for r in test}.isdisjoint({r.sample_id for r in test2})


def test_j_the_mapped_accessor_pools_test_and_test2_which_is_the_hazard() -> None:
    """Pinned deliberately: `by_split("test")` returns BOTH native splits.

    The scenes are the same 968 asked about twice, so pooling them silently
    doubles the denominator with a correlated second question set. This test
    documents why `by_native_split` exists and must not be replaced by the
    convenience accessor.
    """
    manifest = _leakage_manifest()
    pooled = manifest.by_split("test")
    assert len(pooled) == 4 == len(D.by_native_split(manifest, "Test")) * 2
    assert len(pooled) > len(D.by_native_split(manifest, "Test"))


def test_j_an_unknown_native_split_is_rejected() -> None:
    with pytest.raises(D.ChangeVQAError, match="unknown native split"):
        D.by_native_split(_leakage_manifest(), "Training")


def test_j_split_map_routes_test2_to_test_and_keeps_the_native_name() -> None:
    assert D.SPLIT_MAP == {
        "Train": "train",
        "Val": "val",
        "Test": "test",
        "Test2": "test",
    }
    for record in D.by_native_split(_leakage_manifest(), "Test2"):
        assert record.split == "test"
        assert record.extra["native_split"] == "Test2"


def test_j_expected_corpus_figures_are_internally_consistent() -> None:
    """The manifest count is NOT the scene count, and that is deliberate.

    Collapsing them would destroy the Test/Test2 distinction the evaluation
    depends on.
    """
    assert D.EXPECTED_SPLIT_SIZES == {
        "Train": (1600, 65967),
        "Val": (400, 16441),
        "Test": (968, 39686),
        "Test2": (968, 31036),
    }
    assert D.EXPECTED_TOTAL_QUESTIONS == 153130
    assert D.EXPECTED_TOTAL_SCENES == 2968
    assert D.EXPECTED_MANIFEST_RECORDS == 3936
    assert D.EXPECTED_MANIFEST_RECORDS == sum(s for s, _ in D.EXPECTED_SPLIT_SIZES.values())
    # 3,936 manifest rows cover 2,968 scenes because Test and Test2 duplicate 968.
    assert D.EXPECTED_MANIFEST_RECORDS - D.EXPECTED_TOTAL_SCENES == 968


def test_j_dataset_statistics_counts_without_opening_an_image() -> None:
    records = (
        _split_records("Train", ["0001", "0002"])
        + _split_records("Val", ["0003"])
    )
    stats = D.dataset_statistics(records)
    assert stats["dataset_id"] == "cdvqa"
    assert stats["n_records"] == 3
    assert stats["n_unique_scenes"] == 3
    assert stats["records_per_native_split"] == {"Train": 2, "Val": 1}
    assert stats["answers_outside_frozen_vocabulary"] == []
    assert stats["answers_illegal_for_their_type"] == 0


# ===========================================================================
# Area J (cont.) - the ground truth, re-derived from the raw annotations
# ===========================================================================
def _raw_split_figures(split: str) -> tuple[int, int, list[int]]:
    """Read the raw annotation JSON for one split, bypassing every adapter.

    Returns `(unique_file_names, n_questions, question_ids)`. This is the
    arbiter: when a pipeline disagrees with these numbers, the pipeline is what
    is wrong.
    """
    images = json.loads((ANNOTATIONS / f"{split}_images.json").read_text("utf-8"))[
        "images"
    ]
    questions = json.loads(
        (ANNOTATIONS / f"{split}_questions.json").read_text("utf-8")
    )["questions"]
    file_names = {row["file_name"] for row in images}
    ids = sorted(int(row["id"]) for row in questions)
    return len(file_names), len(questions), ids


@pytest.mark.skipif(
    not ANNOTATIONS.exists(), reason="CDVQA annotations are not present on disk"
)
def test_j_raw_annotations_confirm_the_recorded_corpus_ground_truth() -> None:
    """Re-derive EXPECTED_SPLIT_SIZES from the twelve annotation files.

    If this fails, the constants are stale and every count downstream of them is
    wrong -- which is the failure mode the constants exist to make loud.
    """
    observed: dict[str, tuple[int, int]] = {}
    for split in ("Train", "Val", "Test", "Test2"):
        scenes, n_questions, _ids = _raw_split_figures(split)
        observed[split] = (scenes, n_questions)
    assert observed == D.EXPECTED_SPLIT_SIZES


@pytest.mark.skipif(
    not ANNOTATIONS.exists(), reason="CDVQA annotations are not present on disk"
)
def test_j_question_ids_are_per_split_sequential_which_is_the_defect_mechanism() -> None:
    """Proves the MECHANISM behind the cross-split mis-pairing.

    Each split restarts its question ids at 0, so a single dict keyed on
    `question_id` across more than one split lets a later split overwrite an
    earlier one. This test is the evidence for the per-split workaround in
    `_examples_for_one_split`; it is asserted rather than described because a
    comment can drift from the data and this cannot.
    """
    for split in ("Train", "Val", "Test", "Test2"):
        _scenes, n_questions, ids = _raw_split_figures(split)
        assert ids == list(range(n_questions)), split
        assert ids[0] == 0, split
    # The collision is therefore guaranteed: two splits both own id 0.
    assert _raw_split_figures("Train")[2][0] == _raw_split_figures("Val")[2][0] == 0


@pytest.mark.skipif(
    not ANNOTATIONS.exists(), reason="CDVQA annotations are not present on disk"
)
def test_j_every_corpus_answer_is_inside_the_frozen_vocabulary() -> None:
    """The vocabulary is derived from the ontology; the corpus must agree."""
    observed: set[str] = set()
    for split in ("Train", "Val", "Test", "Test2"):
        answers = json.loads(
            (ANNOTATIONS / f"{split}_answers.json").read_text("utf-8")
        )["answers"]
        observed |= {str(row["answer"]) for row in answers}
    assert V.answers_outside_vocabulary(sorted(observed)) == []
    assert observed <= set(V.ANSWER_VOCABULARY)


@pytest.mark.skipif(
    not ANNOTATIONS.exists(), reason="CDVQA annotations are not present on disk"
)
def test_j_train_val_and_test_are_scene_disjoint_in_the_raw_annotations() -> None:
    scene_sets: dict[str, set[str]] = {}
    for split in ("Train", "Val", "Test", "Test2"):
        images = json.loads(
            (ANNOTATIONS / f"{split}_images.json").read_text("utf-8")
        )["images"]
        scene_sets[split] = {row["file_name"] for row in images}

    assert scene_sets["Train"] & scene_sets["Val"] == set()
    assert scene_sets["Train"] & scene_sets["Test"] == set()
    assert scene_sets["Val"] & scene_sets["Test"] == set()
    assert scene_sets["Test"] == scene_sets["Test2"]
    assert len(scene_sets["Train"] | scene_sets["Val"] | scene_sets["Test"]) == (
        D.EXPECTED_TOTAL_SCENES
    )


# ---------------------------------------------------------------------------
# F1 — the cross-split question-id collision, fixed at its root
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not ANNOTATIONS.exists(), reason="CDVQA annotations are not present on disk"
)
def test_j_one_multi_split_call_pairs_every_question_with_its_own_split() -> None:
    """F1 regression, on the real corpus, at the real scale.

    Before the fix, `build_cdvqa_examples` keyed its question->scene index on the
    bare `question_id`. Because every split restarts its ids at 0, a single call
    over ("Train", "Val") produced 82,408 examples over only **1,601** scenes
    instead of 2,000, and 32,882 of those questions were paired with imagery
    belonging to the *other* split. Nothing raised; the examples still trained.

    The figures below are the measured ground truth. They are asserted rather
    than recomputed, so that a regression is a failure and not a new baseline.
    """
    from training.data.cdvqa import build_cdvqa_examples, load_cdvqa

    corpus = load_cdvqa(CDVQA_ROOT, splits=("Train", "Val"))
    examples = build_cdvqa_examples(corpus.images, corpus.questions)

    # -- counts ---------------------------------------------------------------
    assert len(examples) == 82_408
    questions_per_split: dict[str, int] = {"Train": 0, "Val": 0}
    scenes_per_split: dict[str, set[str]] = {"Train": set(), "Val": set()}
    for example in examples:
        questions_per_split[example["split"]] += 1
        scenes_per_split[example["split"]].add(example["file_name"])

    assert questions_per_split == {"Train": 65_967, "Val": 16_441}
    assert {s: len(v) for s, v in scenes_per_split.items()} == {
        "Train": 1_600,
        "Val": 400,
    }
    # Cross-checked against the constants the rest of the suite uses.
    assert D.EXPECTED_SPLIT_SIZES["Train"] == (1_600, 65_967)
    assert D.EXPECTED_SPLIT_SIZES["Val"] == (400, 16_441)

    # -- no cross-split scene leakage ----------------------------------------
    raw_scenes = {
        split: {
            row["file_name"]
            for row in json.loads(
                (ANNOTATIONS / f"{split}_images.json").read_text("utf-8")
            )["images"]
        }
        for split in ("Train", "Val")
    }
    assert raw_scenes["Train"] & raw_scenes["Val"] == set()

    # Every example's scene comes from its OWN split, and no example names a
    # scene from another split -- i.e. no question is paired with imagery that
    # belongs elsewhere.
    assert scenes_per_split["Train"] <= raw_scenes["Train"]
    assert scenes_per_split["Val"] <= raw_scenes["Val"]
    assert scenes_per_split["Train"] & raw_scenes["Val"] == set()
    assert scenes_per_split["Val"] & raw_scenes["Train"] == set()

    # The defect lost 399 Train scenes (1,601 total instead of 2,000). Every
    # scene of both splits must be reachable again.
    assert scenes_per_split["Train"] == raw_scenes["Train"]
    assert scenes_per_split["Val"] == raw_scenes["Val"]


@pytest.mark.skipif(
    not ANNOTATIONS.exists(), reason="CDVQA annotations are not present on disk"
)
def test_j_test_and_test2_questions_also_keep_their_own_pairing() -> None:
    """F1 for the held-out pair: Test2 must not clobber Test.

    Test and Test2 are two question sets over the SAME 968 scenes, and both
    restart their ids at 0 (Test 0..39,685, Test2 0..31,035). Under the old
    bare-id index, Test2's 31,036 ids overwrote Test's for the whole overlap, so
    a Test question was answered with whatever scene Test2 happened to attach to
    that id. Because the scene SET is shared the wrong pairing was invisible in
    the counts -- which is precisely why it needed a dedicated test.
    """
    from training.data.cdvqa import build_cdvqa_examples, load_cdvqa

    corpus = load_cdvqa(CDVQA_ROOT, splits=("Test", "Test2"))
    examples = build_cdvqa_examples(corpus.images, corpus.questions)

    per_split: dict[str, int] = {"Test": 0, "Test2": 0}
    scenes_per_split: dict[str, set[str]] = {"Test": set(), "Test2": set()}
    for example in examples:
        per_split[example["split"]] += 1
        scenes_per_split[example["split"]].add(example["file_name"])

    assert per_split == {"Test": 39_686, "Test2": 31_036}
    assert {s: len(v) for s, v in scenes_per_split.items()} == {
        "Test": 968,
        "Test2": 968,
    }
    # Both question sets still cover the same 968 scenes; only the PAIRING is
    # split-scoped, never the scene population.
    assert scenes_per_split["Test"] == scenes_per_split["Test2"]

    # The pairing itself: re-derive each split's question -> scene map from the
    # raw tables and require the examples to agree exactly.
    for split in ("Test", "Test2"):
        images = json.loads(
            (ANNOTATIONS / f"{split}_images.json").read_text("utf-8")
        )["images"]
        expected: dict[int, str] = {}
        for row in images:
            for qid in row["questions_ids"]:
                expected[int(qid)] = row["file_name"]
        observed = {
            example["question_id"]: example["file_name"]
            for example in examples
            if example["split"] == split
        }
        assert observed, split
        mismatched = {
            qid: (observed[qid], expected[qid])
            for qid in observed
            if qid in expected and observed[qid] != expected[qid]
        }
        assert mismatched == {}, (split, mismatched)
