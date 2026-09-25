"""Tests for the VRSBench referring-expression loader.

The load-bearing test is `test_unparseable_schema_raises_with_diagnostics`.
The VRSBench JSON layout is UNVERIFIED -- I have not inspected a download --
so the loader's most important property is that it FAILS LOUDLY when its
assumptions are wrong, rather than returning an empty list that looks like a
working loader and produces an empty experiment.

The other cluster of tests pins the 0-100 -> 0-1 conversion, because getting
that wrong is silent: boxes a hundred times too large would show up as a
suspiciously high IoU, not as a crash.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from training.data.vrsbench import (
    VRSBENCH_BOX_SCALE,
    ReferringSample,
    VRSBenchError,
    load_referring_samples,
    summarize,
    warnings_for,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def write_image(path: Path, size: int = 64) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:size, 0:size]
    plane = ((xx + yy) % 256).astype(np.uint8)
    Image.fromarray(np.stack([plane, plane // 2, 255 - plane], axis=-1)).save(path)
    return path


@pytest.fixture()
def vrsbench_tree(tmp_path: Path) -> Path:
    """A plausible VRSBench layout with 0-100 boxes."""
    root = tmp_path / "vrsbench"
    (root / "images").mkdir(parents=True)

    records = []
    for i in range(6):
        name = f"img_{i:03d}"
        write_image(root / "images" / f"{name}.jpg")
        # A box in the middle third of the image, in 0-100 coordinates.
        records.append(
            {
                "id": f"ref_{i}",
                "image_id": name,
                "phrase": f"the building number {i}",
                "bbox": [30.0, 30.0, 60.0, 60.0],
            }
        )

    (root / "referring.json").write_text(
        json.dumps({"data": records}), encoding="utf-8"
    )
    return root


# ---------------------------------------------------------------------------
# The 0-100 conversion -- silent when wrong
# ---------------------------------------------------------------------------


def test_zero_to_hundred_boxes_are_scaled(vrsbench_tree: Path) -> None:
    samples = load_referring_samples(vrsbench_tree, limit=None, verbose=False)
    assert samples
    box = samples[0].box
    assert box == pytest.approx([0.30, 0.30, 0.60, 0.60])


def test_boxes_are_within_zero_to_one(vrsbench_tree: Path) -> None:
    for sample in load_referring_samples(vrsbench_tree, limit=None, verbose=False):
        assert all(0.0 <= v <= 1.0 for v in sample.box), sample.box


def test_already_normalized_boxes_are_left_alone(tmp_path: Path) -> None:
    """A 0-1 box must not be divided by 100 a second time."""
    root = tmp_path / "root"
    write_image(root / "images" / "a.jpg")
    (root / "ref.json").write_text(
        json.dumps({"data": [
            {"id": "x", "image_id": "a", "phrase": "a thing",
             "bbox": [0.25, 0.25, 0.5, 0.5]}
        ]}),
        encoding="utf-8",
    )

    samples = load_referring_samples(root, limit=None, verbose=False)
    assert samples[0].box == pytest.approx([0.25, 0.25, 0.5, 0.5])


def test_ambiguous_box_follows_the_documented_convention(tmp_path: Path) -> None:
    """[20, 20, 30, 40] is read as 0-100 xyxy.

    This input is range-valid under BOTH the xyxy and the xywh reading, so the
    documented convention settles it, not a heuristic.

    DISCRIMINATION NOTE: this test does NOT exercise the removed xywh
    heuristic. Scaling gives [0.2, 0.2, 0.3, 0.4] with x2 > x1, so the
    heuristic's `if x2 < x1` branch was never taken and the old
    implementation returned these same values. An earlier version of this test
    claimed otherwise in its docstring; the claim was wrong and the test would
    have passed under both implementations, which makes it an assertion rather
    than a control. The discriminating case is covered by
    `test_the_removed_heuristic_would_have_diverged`.
    """
    root = tmp_path / "root"
    write_image(root / "images" / "a.jpg")
    (root / "ref.json").write_text(
        json.dumps({"data": [
            {"id": "x", "image_id": "a", "phrase": "a thing",
             "bbox": [20.0, 20.0, 30.0, 40.0]}
        ]}),
        encoding="utf-8",
    )

    samples = load_referring_samples(root, limit=None, verbose=False)
    # xyxy in 0-100: x1=0.2, y1=0.2, x2=0.3, y2=0.4
    assert samples[0].box == pytest.approx([0.2, 0.2, 0.3, 0.4])


def test_the_removed_heuristic_would_have_diverged() -> None:
    """The anti-xywh guard, shown to discriminate.

    Reconstructs the removed heuristic inline -- it is NOT imported, because it
    no longer exists in the module -- and asserts the two implementations
    disagree on an input where the heuristic actually fired. If the heuristic
    is reintroduced, or if the loader starts accepting what it accepted, the
    two stop diverging and this fails.

    Measured on the discriminating input:

        [80, 80, 20, 20]   current loader    -> None            (rejected)
                           removed heuristic -> [0.8, 0.8, 1.0, 1.0]
    """
    from training.data.vrsbench import _coerce_box

    def removed_heuristic(raw):
        """The deleted xywh branch, reproduced for comparison only."""
        values = [float(v) for v in raw]
        if (all(0.0 <= v <= 1.0 for v in values)
                and values[2] >= values[0] and values[3] >= values[1]):
            return values
        scaled = [v / VRSBENCH_BOX_SCALE for v in values]
        if scaled[2] < scaled[0] or scaled[3] < scaled[1]:
            x, y, w, h = scaled
            scaled = [x, y, x + w, y + h]
        if not all(-0.001 <= v <= 1.001 for v in scaled):
            return None
        return scaled

    discriminating = [80.0, 80.0, 20.0, 20.0]

    current = _coerce_box(discriminating)
    previous = removed_heuristic(discriminating)

    assert current is None, "the loader accepted a malformed box"
    assert previous == pytest.approx([0.8, 0.8, 1.0, 1.0])
    assert current != previous, (
        "the loader and the removed heuristic agree on the discriminating "
        "input -- either the heuristic is back, or this test has stopped "
        "discriminating and must be replaced"
    )


def test_a_genuinely_inverted_box_is_rejected(tmp_path: Path) -> None:
    """After scaling, x2 < x1 means the record is malformed, not xywh."""
    root = tmp_path / "root"
    write_image(root / "images" / "a.jpg")
    (root / "ref.json").write_text(
        json.dumps({"data": [
            {"id": "x", "image_id": "a", "phrase": "p",
             "bbox": [80.0, 80.0, 20.0, 20.0]}
        ]}),
        encoding="utf-8",
    )

    with pytest.raises(VRSBenchError, match="could not parse"):
        load_referring_samples(root, limit=None, verbose=False)


def test_the_scale_constant_matches_the_documented_convention() -> None:
    assert VRSBENCH_BOX_SCALE == 100.0


def test_double_conversion_is_caught_by_the_warnings() -> None:
    """A mean width near zero means the scale was applied twice."""
    summary = {"samples": 10, "unique_phrases": 10, "box_width_mean": 0.0003}
    assert any("twice" in w for w in warnings_for(summary))


def test_missing_conversion_is_caught_by_the_warnings() -> None:
    """A mean width near 1.0 means raw 0-100 values were treated as 0-1."""
    summary = {"samples": 10, "unique_phrases": 10, "box_width_mean": 0.97}
    assert any("skipped" in w for w in warnings_for(summary))


# ---------------------------------------------------------------------------
# Failing loudly on an unverified schema
# ---------------------------------------------------------------------------


def test_unparseable_schema_raises_with_diagnostics(tmp_path: Path) -> None:
    """The property that keeps an unknown-schema file from silently yielding [].

    The verified VRSBench layouts are documented in the module docstring
    (eval: image_id/question/ground_truth token string; train: conversations
    with [refer] turns; plus a legacy key-fallback table). A file matching
    NONE of those must still raise and name the keys it actually saw, so the
    layout can be learned from the first real run.
    """
    root = tmp_path / "root"
    (root / "annotations").mkdir(parents=True)
    (root / "annotations" / "mystery.json").write_text(
        json.dumps({"totally_unexpected_key": [{"a": 1}]}), encoding="utf-8"
    )

    with pytest.raises(VRSBenchError) as exc:
        load_referring_samples(root, limit=None, verbose=False)

    message = str(exc.value)
    assert "totally_unexpected_key" in message, (
        "the error did not report the keys it actually found"
    )
    assert "matched none of them" in message, (
        "the error no longer distinguishes a verified-schema miss from a crash"
    )


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(VRSBenchError, match="does not exist"):
        load_referring_samples(tmp_path / "nope")


def test_no_json_raises_with_a_useful_message(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "readme.txt").write_text("no annotations here")
    with pytest.raises(VRSBenchError, match="no JSON files"):
        load_referring_samples(root)


def test_records_without_images_are_skipped_not_fatal(tmp_path: Path) -> None:
    """Partial data should degrade, not abort a 118 GB preparation."""
    root = tmp_path / "root"
    write_image(root / "images" / "present.jpg")
    (root / "ref.json").write_text(
        json.dumps({"data": [
            {"id": "a", "image_id": "present", "phrase": "ok",
             "bbox": [10, 10, 20, 20]},
            {"id": "b", "image_id": "absent", "phrase": "missing image",
             "bbox": [10, 10, 20, 20]},
        ]}),
        encoding="utf-8",
    )

    samples = load_referring_samples(root, limit=None, verbose=False)
    assert len(samples) == 1
    assert samples[0].sample_id == "a"


# ---------------------------------------------------------------------------
# Schema tolerance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("list_key", ["data", "annotations", "regions", "objects"])
def test_several_record_list_keys_are_accepted(tmp_path: Path, list_key: str) -> None:
    root = tmp_path / "root"
    write_image(root / "images" / "a.jpg")
    (root / "ref.json").write_text(
        json.dumps({list_key: [
            {"id": "x", "image_id": "a", "phrase": "p", "bbox": [10, 10, 20, 20]}
        ]}),
        encoding="utf-8",
    )
    assert len(load_referring_samples(root, limit=None, verbose=False)) == 1


@pytest.mark.parametrize("phrase_key", ["phrase", "expression", "ref", "sentence"])
def test_several_phrase_keys_are_accepted(tmp_path: Path, phrase_key: str) -> None:
    root = tmp_path / "root"
    write_image(root / "images" / "a.jpg")
    (root / "ref.json").write_text(
        json.dumps({"data": [
            {"id": "x", "image_id": "a", phrase_key: "hello",
             "bbox": [10, 10, 20, 20]}
        ]}),
        encoding="utf-8",
    )
    assert len(load_referring_samples(root, limit=None, verbose=False)) == 1


def test_a_bare_top_level_list_is_accepted(tmp_path: Path) -> None:
    root = tmp_path / "root"
    write_image(root / "images" / "a.jpg")
    (root / "ref.json").write_text(
        json.dumps([{"id": "x", "image_id": "a", "phrase": "p",
                     "bbox": [10, 10, 20, 20]}]),
        encoding="utf-8",
    )
    assert len(load_referring_samples(root, limit=None, verbose=False)) == 1


# ---------------------------------------------------------------------------
# Sampling and reporting
# ---------------------------------------------------------------------------


def test_limit_truncates_deterministically(vrsbench_tree: Path) -> None:
    a = load_referring_samples(vrsbench_tree, limit=3, seed=1, verbose=False)
    b = load_referring_samples(vrsbench_tree, limit=3, seed=1, verbose=False)
    assert len(a) == 3
    assert [s.sample_id for s in a] == [s.sample_id for s in b]


def test_different_seeds_pick_different_samples(vrsbench_tree: Path) -> None:
    a = load_referring_samples(vrsbench_tree, limit=3, seed=1, verbose=False)
    b = load_referring_samples(vrsbench_tree, limit=3, seed=99, verbose=False)
    assert [s.sample_id for s in a] != [s.sample_id for s in b]


def test_limit_none_returns_everything(vrsbench_tree: Path) -> None:
    assert len(load_referring_samples(vrsbench_tree, limit=None, verbose=False)) == 6


def test_summarize_reports_structure(vrsbench_tree: Path) -> None:
    samples = load_referring_samples(vrsbench_tree, limit=None, verbose=False)
    summary = summarize(samples)
    assert summary["samples"] == 6
    assert summary["unique_phrases"] == 6
    assert 0.0 < summary["box_width_mean"] < 1.0


def test_summarize_of_empty() -> None:
    assert summarize([]) == {"samples": 0}


def test_warnings_are_empty_for_a_clean_corpus(vrsbench_tree: Path) -> None:
    summary = summarize(
        load_referring_samples(vrsbench_tree, limit=None, verbose=False)
    )
    assert warnings_for(summary) == []


# ---------------------------------------------------------------------------
# Sample validation
# ---------------------------------------------------------------------------


def test_sample_rejects_a_bad_box_length() -> None:
    with pytest.raises(VRSBenchError, match="4 values"):
        ReferringSample(sample_id="x", image=None, phrase="p", box=[1, 2, 3])


def test_sample_rejects_an_out_of_range_box() -> None:
    with pytest.raises(VRSBenchError, match="outside 0-1"):
        ReferringSample(sample_id="x", image=None, phrase="p", box=[0, 0, 90, 90])


def test_sample_rejects_an_inverted_box() -> None:
    with pytest.raises(VRSBenchError, match="inverted"):
        ReferringSample(sample_id="x", image=None, phrase="p", box=[0.8, 0.8, 0.2, 0.2])


def test_sample_accepts_a_valid_box() -> None:
    s = ReferringSample(sample_id="x", image=None, phrase="p",
                        box=[0.1, 0.2, 0.5, 0.6])
    assert s.box == [0.1, 0.2, 0.5, 0.6]


# ---------------------------------------------------------------------------
# Verified VRSBench schema (evidence: xiang709/VRSBench, checked 2026-09-16)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("{<25><40><33><60>}", [0.25, 0.40, 0.33, 0.60]),   # the real eval sample
        ("{<0><0><100><100>}", [0.0, 0.0, 1.0, 1.0]),       # full-frame corner case
    ],
)
def test_gt_token_string_parses_to_zero_one(raw: str, expected: list[float]) -> None:
    from training.data.vrsbench import _parse_gt_token_string
    assert _parse_gt_token_string(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw,reason",
    [
        ("{<25><40><33>}", "3 tokens, not 4"),
        ("{<-1><40><33><60>}", "negative token"),
        ("{<25><40><33><120>}", "token over 100 (measured train noise reaches 196)"),
        ("no tokens here", "no angle tokens"),
    ],
)
def test_gt_token_string_rejects_out_of_range(raw: str, reason: str) -> None:
    from training.data.vrsbench import _parse_gt_token_string
    assert _parse_gt_token_string(raw) is None, reason


def test_eval_schema_records_parse_end_to_end(tmp_path: Path) -> None:
    """The verified eval layout: bare-filename image refs + token-string GT."""
    root = tmp_path / "root"
    write_image(root / "P0003_0002.png")
    (root / "eval.json").write_text(
        json.dumps([
            {
                "image_id": "P0003_0002.png",
                "question": "The large yellow vehicle situated closest to the green area.",
                "ground_truth": "{<25><40><33><60>}",
                "question_id": 0,
            },
            {
                "image_id": "P0003_0002.png",
                "question": "The toll station in the centre.",
                "ground_truth": "{<45><45><59><59>}",
                "question_id": 1,
            },
        ]),
        encoding="utf-8",
    )

    samples = load_referring_samples(root, limit=None, verbose=False)
    assert len(samples) == 2
    assert samples[0].box == pytest.approx([0.25, 0.40, 0.33, 0.60])
    # ids must be unique: question_id repeats per image, so the stem goes in.
    assert samples[0].sample_id == "P0003_0002_0000"
    assert samples[1].sample_id == "P0003_0002_0001"


def test_train_refer_turns_parse_and_others_are_ignored(tmp_path: Path) -> None:
    """The verified train layout: [refer] turns become samples; caption/vqa don't.

    Out-of-range ground truths (the measured 8,238/36,313 noise in
    VRSBench_train.json) are rejected, not clamped.
    """
    root = tmp_path / "root"
    write_image(root / "00002_0000.png")
    (root / "train.json").write_text(
        json.dumps([
            {
                "id": "Final_Data/v1.2",
                "image": "00002_0000.png",
                "conversations": [
                    {"from": "human",
                     "value": "<image>\n[refer] could you tell me the location "
                              "for <p>The toll station is positioned at the "
                              "center of the image</p>?"},
                    {"from": "gpt", "value": "{<45><45><59><59>}"},
                ],
            },
            {
                "id": "Final_Data/v1.2",
                "image": "00002_0000.png",
                "conversations": [
                    {"from": "human",
                     "value": "<image>\n[caption] Could you describe the "
                              "contents of this image for me?"},
                    {"from": "gpt", "value": "A rural area with a toll station."},
                ],
            },
            {
                "id": "Final_Data/v1.2",
                "image": "00002_0000.png",
                "conversations": [
                    {"from": "human",
                     "value": "<image>\n[refer] location of "
                              "<p>A chimney in the corner</p> is"},
                    {"from": "gpt", "value": "{<45><53><86><95>}"},
                ],
            },
            {
                "id": "Final_Data/v1.2",
                "image": "00002_0000.png",
                "conversations": [
                    {"from": "human",
                     "value": "<image>\n[refer] location of "
                              "<p>An out-of-range object</p> is"},
                    {"from": "gpt", "value": "{<60><83><101><93>}"},
                ],
            },
        ]),
        encoding="utf-8",
    )

    samples = load_referring_samples(root, limit=None, verbose=False)
    # 4 records: 1 caption (ignored, not a skip), 2 valid refer, 1 out-of-range
    # refer (skipped).
    assert len(samples) == 2
    assert all(
        s.sample_id.startswith("Final_Data/v1.2_ref_") for s in samples
    ), "the verified train id must carry the record index for uniqueness"
    phrases = {s.phrase for s in samples}
    assert "The toll station is positioned at the center of the image" in phrases
    assert "A chimney in the corner" in phrases
    assert not any("out-of-range" in p for p in phrases)