"""Regression tests for the degenerate fallback-box guard (Phase 8).

THE DEFECT THESE EXIST FOR
--------------------------
The grounding E2E run in fallback mode (`used_head=False`) returned, as its top
candidate, a box covering the ENTIRE frame:

    x1=0.000 y1=0.000 x2=1.000 y2=1.000  ->  area 1.000

That is not a localisation. It asserts "somewhere in this image", which is what
a failed localisation looks like once it has been smoothed into a rectangle --
and because `core.schemas.Box` accepts it, nothing downstream could tell it
apart from a real detection.

The guard drops such candidates in the ZERO-SHOT FALLBACK PATH ONLY and records
what it dropped. The trained-head path is deliberately untouched: it is a
regressed box with its own learned prior, and filtering it would move the frozen
Phase 8 benchmark numbers.

Boundary semantics: `area > DEGENERATE_AREA_FRACTION` is dropped. A box of
exactly 0.9 area is KEPT. That is pinned below, because an off-by-one on the
comparison operator would silently change which candidates survive.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from core.schemas import EvidenceType, Task
from specialists.base import SpecialistRequest
from specialists.grounding.specialist import (
    CONFIDENCE_WEIGHTS,
    DEGENERATE_AREA_FRACTION,
    DecodedBox,
    GroundingSpecialist,
    _box_area,
)

# ---------------------------------------------------------------------------
# Fixtures and stubs
# ---------------------------------------------------------------------------


@pytest.fixture()
def geotiff(tmp_path: Path) -> Path:
    """A small georeferenced 3-band GeoTIFF with real spatial structure."""
    rasterio = pytest.importorskip("rasterio")
    from affine import Affine

    size = 128
    yy, xx = np.mgrid[0:size, 0:size]
    ramp = (xx + yy).astype(np.float64)
    texture = np.random.default_rng(0).integers(0, 80, (size, size)).astype(np.float64)
    plane = ramp + texture
    data = np.stack([plane, plane * 0.6 + 800.0, plane * 0.4 + 1600.0]).astype("uint16")

    path = tmp_path / "scene.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=size, height=size, count=3,
        dtype="uint16", crs="EPSG:32643",
        transform=Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 2_500_000.0),
    ) as ds:
        ds.write(data)
    return path


class StubEncoder:
    """Minimal RemoteCLIP stand-in. The decode is stubbed, so the values it
    returns are never interpreted -- only its call surface matters."""

    model_name = "stub-remoteclip"
    checkpoint_path = "stub-checkpoint.pt"

    def encode_image(self, image):  # noqa: ARG002
        return types.SimpleNamespace(
            patch_tokens=np.zeros((49, 512), dtype=np.float32),
            cls=np.zeros(512, dtype=np.float32),
        )

    def encode_text(self, texts):
        return np.zeros((len(texts), 512), dtype=np.float32)


class StubHead:
    """Head stand-in. Only `eval()` is called during construction, and its
    presence flips `has_head` to True so the head path can be exercised."""

    def eval(self) -> None:  # noqa: D102
        return None

    def num_parameters(self) -> int:  # noqa: D102
        return 1_052_677


def _stats() -> dict[str, float]:
    return {
        "max_objectness": 0.42,
        "top_mean_objectness": 0.30,
        "score_contrast": 0.12,
    }


def _request(path: Path, query: str = "the water body") -> SpecialistRequest:
    from preprocessing.raster import inspect_raster

    return SpecialistRequest(assets=[inspect_raster(path)], query=query)


def _make(
    *,
    head: object | None = None,
    boxes: list[DecodedBox],
    head_boxes: list[DecodedBox] | None = None,
) -> GroundingSpecialist:
    """Specialist with the decode stubbed to return a controlled box list.

    The stub is assigned to the INSTANCE, not the class, so no other test sees
    it. A plain function on an instance is not bound, so it receives exactly
    the arguments `execute()` passes: (encoded, text).
    """
    specialist = GroundingSpecialist(encoder=StubEncoder(), head=head, grid=7)

    def _zero_shot(encoded, text):  # noqa: ARG001
        return list(boxes), _stats()

    specialist._decode_zero_shot = _zero_shot  # type: ignore[method-assign]

    if head_boxes is not None:

        def _with_head(encoded, text):  # noqa: ARG001
            return list(head_boxes), _stats()

        specialist._decode_with_head = _with_head  # type: ignore[method-assign]

    return specialist


FULL_FRAME = DecodedBox(box=[0.0, 0.0, 1.0, 1.0], score=0.31, source="zero_shot")
SMALL_BOX = DecodedBox(box=[0.25, 0.25, 0.40, 0.40], score=0.55, source="zero_shot")


# ---------------------------------------------------------------------------
# The area helper
# ---------------------------------------------------------------------------


def test_box_area_full_frame_is_one() -> None:
    assert _box_area([0.0, 0.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_box_area_of_a_quadrant() -> None:
    assert _box_area([0.0, 0.0, 0.5, 0.5]) == pytest.approx(0.25)


def test_box_area_of_inverted_box_is_zero_not_negative() -> None:
    assert _box_area([0.8, 0.8, 0.2, 0.2]) == 0.0


def test_box_area_of_zero_width_is_zero() -> None:
    assert _box_area([0.5, 0.1, 0.5, 0.9]) == 0.0


# ---------------------------------------------------------------------------
# _drop_degenerate -- the mechanism
# ---------------------------------------------------------------------------


def test_drop_degenerate_splits_full_frame_out() -> None:
    kept, dropped = GroundingSpecialist._drop_degenerate([FULL_FRAME, SMALL_BOX])
    assert [d.box for d in kept] == [SMALL_BOX.box]
    assert [d.box for d in dropped] == [FULL_FRAME.box]


def test_drop_degenerate_keeps_everything_usable() -> None:
    kept, dropped = GroundingSpecialist._drop_degenerate([SMALL_BOX])
    assert len(kept) == 1
    assert dropped == []


def test_drop_degenerate_drops_everything_when_all_are_full_frame() -> None:
    kept, dropped = GroundingSpecialist._drop_degenerate([FULL_FRAME, FULL_FRAME])
    assert kept == []
    assert len(dropped) == 2


def test_drop_degenerate_on_empty_input() -> None:
    kept, dropped = GroundingSpecialist._drop_degenerate([])
    assert kept == [] and dropped == []


def test_area_exactly_at_threshold_is_KEPT() -> None:
    """Boundary: the comparison is `>`, so 0.9 exactly survives.

    An off-by-one here silently changes which candidates reach the result, so
    the semantics are pinned rather than assumed.
    """
    boundary = DecodedBox(box=[0.0, 0.0, 0.9, 1.0], score=0.4, source="zero_shot")
    assert _box_area(boundary.box) == pytest.approx(DEGENERATE_AREA_FRACTION)
    kept, dropped = GroundingSpecialist._drop_degenerate([boundary])
    assert len(kept) == 1, "a box at exactly the threshold must be kept"
    assert dropped == []


def test_area_just_over_threshold_is_DROPPED() -> None:
    over = DecodedBox(box=[0.0, 0.0, 0.91, 1.0], score=0.4, source="zero_shot")
    assert _box_area(over.box) > DEGENERATE_AREA_FRACTION
    kept, dropped = GroundingSpecialist._drop_degenerate([over])
    assert kept == []
    assert len(dropped) == 1


def test_threshold_override_is_honoured() -> None:
    """The threshold is a parameter, so a caller can tighten or loosen it."""
    kept, dropped = GroundingSpecialist._drop_degenerate(
        [SMALL_BOX], max_area_fraction=0.01
    )
    assert kept == []
    assert len(dropped) == 1


# ---------------------------------------------------------------------------
# execute() in fallback mode -- the reported defect
# ---------------------------------------------------------------------------


def test_fallback_full_frame_box_produces_no_regions(geotiff: Path) -> None:
    """The exact reported case: a full-frame candidate must not become a region."""
    specialist = _make(head=None, boxes=[FULL_FRAME])
    result = specialist.execute(_request(geotiff))

    assert result.boxes == [], "a full-frame box was emitted as a box"
    assert result.regions == [], "a full-frame box was emitted as a region"


def test_fallback_full_frame_box_still_returns_a_valid_result(geotiff: Path) -> None:
    specialist = _make(head=None, boxes=[FULL_FRAME])
    result = specialist.execute(_request(geotiff))

    assert result.task is Task.GROUNDING
    assert result.answer
    assert result.degraded is True


def test_fallback_keeps_usable_boxes_alongside_dropped(geotiff: Path) -> None:
    """Dropping must be per-candidate, not all-or-nothing."""
    specialist = _make(head=None, boxes=[FULL_FRAME, SMALL_BOX])
    result = specialist.execute(_request(geotiff))

    assert len(result.boxes) == 1
    assert result.boxes[0].x1 == pytest.approx(0.25)
    assert len(result.regions) == 1


def test_fallback_keeps_usable_boxes_only(geotiff: Path) -> None:
    specialist = _make(head=None, boxes=[SMALL_BOX])
    result = specialist.execute(_request(geotiff))

    assert len(result.boxes) == 1
    assert len(result.regions) == 1
    # No drop happened, so no drop warning should be present.
    assert not any("degenerate" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Preserved behaviour -- the guard must not change anything else
# ---------------------------------------------------------------------------


def test_degraded_flag_is_still_set(geotiff: Path) -> None:
    specialist = _make(head=None, boxes=[FULL_FRAME])
    result = specialist.execute(_request(geotiff))
    assert result.degraded is True


def test_missing_head_warning_is_still_emitted(geotiff: Path) -> None:
    specialist = _make(head=None, boxes=[FULL_FRAME])
    result = specialist.execute(_request(geotiff))
    assert any("no trained grounding head is loaded" in w for w in result.warnings)


def test_drop_is_also_warned_about(geotiff: Path) -> None:
    specialist = _make(head=None, boxes=[FULL_FRAME])
    result = specialist.execute(_request(geotiff))
    assert any("degenerate" in w for w in result.warnings), result.warnings


def test_confidence_components_are_unchanged(geotiff: Path) -> None:
    """The guard must not touch the measurable confidence signals."""
    specialist = _make(head=None, boxes=[FULL_FRAME])
    result = specialist.execute(_request(geotiff))

    for key in CONFIDENCE_WEIGHTS:
        assert key in result.confidence.components
    assert result.confidence.components["used_head"] == 0.0
    assert result.confidence.degraded is True
    assert result.confidence.degradation_reason


def test_confidence_value_matches_the_unfiltered_stats(geotiff: Path) -> None:
    """Dropping a box must not change the score the decode reported.

    The confidence describes the DECODE, not the surviving candidate list --
    otherwise a dropped candidate would look like a more confident result.
    """
    dropped_run = _make(head=None, boxes=[FULL_FRAME]).execute(_request(geotiff))
    kept_run = _make(head=None, boxes=[SMALL_BOX]).execute(_request(geotiff))
    assert dropped_run.confidence.raw == pytest.approx(kept_run.confidence.raw)


def test_answer_reports_no_region_when_everything_was_dropped(geotiff: Path) -> None:
    specialist = _make(head=None, boxes=[FULL_FRAME])
    result = specialist.execute(_request(geotiff))
    assert "no confident region" in result.answer.lower()
    assert "located" not in result.answer.lower()


def test_answer_still_reports_a_count_when_boxes_survive(geotiff: Path) -> None:
    specialist = _make(head=None, boxes=[SMALL_BOX])
    result = specialist.execute(_request(geotiff))
    assert "located 1 candidate region" in result.answer.lower()


# ---------------------------------------------------------------------------
# Evidence -- the drop must be recorded, not silent
# ---------------------------------------------------------------------------


def test_dropped_candidate_is_recorded_in_evidence(geotiff: Path) -> None:
    specialist = _make(head=None, boxes=[FULL_FRAME])
    result = specialist.execute(_request(geotiff))

    records = [e for e in result.evidence if e.payload.get("dropped_degenerate")]
    assert len(records) == 1, "the drop was not recorded in evidence"

    payload = records[0].payload
    assert payload["n_dropped"] == 1
    assert payload["area_threshold"] == pytest.approx(DEGENERATE_AREA_FRACTION)
    assert payload["dropped"][0]["box"] == [0.0, 0.0, 1.0, 1.0]
    assert payload["dropped"][0]["area"] == pytest.approx(1.0)
    assert payload["dropped"][0]["source"] == "zero_shot"


def test_dropped_candidate_is_NOT_emitted_as_bounding_box_evidence(
    geotiff: Path,
) -> None:
    """Rejecting the box but then shipping it as spatial evidence would defeat
    the entire guard."""
    specialist = _make(head=None, boxes=[FULL_FRAME])
    result = specialist.execute(_request(geotiff))

    boxes_in_evidence = [
        e for e in result.evidence if e.type is EvidenceType.BOUNDING_BOX
    ]
    assert boxes_in_evidence == [], "the rejected box reappeared as evidence"


def test_no_drop_record_when_nothing_was_dropped(geotiff: Path) -> None:
    specialist = _make(head=None, boxes=[SMALL_BOX])
    result = specialist.execute(_request(geotiff))

    records = [e for e in result.evidence if e.payload.get("dropped_degenerate")]
    assert records == []


def test_evidence_ids_stay_unique_after_a_drop(geotiff: Path) -> None:
    specialist = _make(head=None, boxes=[FULL_FRAME, SMALL_BOX])
    result = specialist.execute(_request(geotiff))

    ids = [e.evidence_id for e in result.evidence]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Schema integrity
# ---------------------------------------------------------------------------


def test_result_serialises_after_a_drop(geotiff: Path) -> None:
    import json

    specialist = _make(head=None, boxes=[FULL_FRAME])
    result = specialist.execute(_request(geotiff))
    payload = json.loads(result.model_dump_json())

    assert payload["task"] == "grounding"
    assert payload["boxes"] == []
    assert payload["regions"] == []
    assert payload["degraded"] is True


def test_evidence_survives_the_schema_serialiser(geotiff: Path) -> None:
    import json

    specialist = _make(head=None, boxes=[FULL_FRAME])
    result = specialist.execute(_request(geotiff))
    payload = json.loads(result.model_dump_json())

    drop_records = [
        e for e in payload["evidence"] if e["payload"].get("dropped_degenerate")
    ]
    assert len(drop_records) == 1


# ---------------------------------------------------------------------------
# The trained-head path must be untouched
# ---------------------------------------------------------------------------


def test_head_path_is_not_filtered(geotiff: Path) -> None:
    """A full-frame box from the TRAINED HEAD must survive.

    The guard is scoped to the zero-shot fallback. The head is a regressed box
    with its own learned prior, and filtering it would move the frozen Phase 8
    benchmark numbers -- which are the evidence the head earned its place on.
    """
    specialist = _make(
        head=StubHead(),
        boxes=[],                      # unused: the head path is taken
        head_boxes=[FULL_FRAME],
    )
    assert specialist.has_head is True

    result = specialist.execute(_request(geotiff))

    assert len(result.boxes) == 1, "the head's box was filtered; it must not be"
    assert result.boxes[0].x1 == pytest.approx(0.0)
    assert result.boxes[0].x2 == pytest.approx(1.0)
    assert result.degraded is False


def test_head_path_emits_no_drop_record(geotiff: Path) -> None:
    specialist = _make(head=StubHead(), boxes=[], head_boxes=[FULL_FRAME])
    result = specialist.execute(_request(geotiff))

    records = [e for e in result.evidence if e.payload.get("dropped_degenerate")]
    assert records == [], "the head path recorded a drop it did not perform"


def test_head_path_confidence_uses_head_component(geotiff: Path) -> None:
    specialist = _make(head=StubHead(), boxes=[], head_boxes=[SMALL_BOX])
    result = specialist.execute(_request(geotiff))
    assert result.confidence.components["used_head"] == 1.0
    assert result.confidence.degraded is False