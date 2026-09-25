"""Tests for the evidence engine — the aggregator (Phase 13).

WHAT THESE PIN
--------------
The engine's whole value is that it is *reproducible* and *lossless*. So the
tests are mostly about the two ways that can silently break:

  determinism  -- the same claims must produce the same ordered, identified
                  collection even when specialists finish in a different order,
                  or when the uuids differ between processes.
  no loss      -- no specialist's evidence may vanish without being counted, and
                  agreement between specialists must be recorded rather than
                  collapsed with one name thrown away.

Plus the honesty rule on confidence: an uncalibrated engine must pass the raw
score through and SAY it is uncalibrated, never manufacture a fitted mapping.

The `--basetemp=.pytest_tmp` flag is mandatory (see the project handoff): the
default pytest temp root triggers a sandbox denial on this machine.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from core.schemas import (
    Box,
    ConfidenceBreakdown,
    CoordinateSystem,
    Evidence,
    EvidenceType,
    GeoMetadata,
    Region,
    SpecialistResult,
    Task,
)
from evidence import (
    DEFAULT_MAX_ITEMS,
    METHOD_TEMPERATURE,
    METHOD_UNCALIBRATED,
    EvidenceCollection,
    EvidenceEngine,
    TemperatureCalibration,
    aggregate_evidence,
    calibrate,
    calibrate_result,
    evidence_digest,
    load_calibration,
)

#: Scratch root, repo-local. The built-in `tmp_path` fixture cannot finalize
#: under this machine's sandbox, so artifact tests use this instead. It is the
#: same directory `--basetemp=.pytest_tmp` points pytest at, so nothing is
#: written outside the workspace.
SCRATCH_ROOT = Path(__file__).resolve().parents[2] / ".pytest_tmp"


@pytest.fixture()
def scratch() -> Path:
    """A fresh scratch directory for artifact I/O tests.

    Deliberately NOT cleaned up on teardown: the sandbox's safe-delete guard
    rejects the recursive delete, and a test that writes into the repo's own
    ignored scratch directory is harmless. Each test gets a unique path so no
    state leaks between them.
    """
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    path = SCRATCH_ROOT / f"evidence_{uuid.uuid4().hex[:8]}"
    path.mkdir(exist_ok=True)
    return path

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def box_evidence(
    coords: list[float],
    *,
    score: float | None = 0.5,
    specialist: str = "grounding",
    payload: dict | None = None,
) -> Evidence:
    return Evidence(
        type=EvidenceType.BOUNDING_BOX,
        source_specialist=specialist,
        coordinate_system=CoordinateSystem.NORMALIZED_0_1,
        coordinates=list(coords),
        score=score,
        payload=payload or {},
    )


def stat_evidence(
    *,
    score: float | None = 0.5,
    specialist: str = "vlm",
    payload: dict | None = None,
) -> Evidence:
    return Evidence(
        type=EvidenceType.STATISTIC,
        source_specialist=specialist,
        score=score,
        payload=payload or {},
    )


def result_with(
    evidence: list[Evidence],
    *,
    task: Task = Task.GROUNDING,
    raw: float = 0.5,
) -> SpecialistResult:
    """A `SpecialistResult` carrying exactly the given evidence."""
    return SpecialistResult(
        task=task,
        answer="built for the evidence tests",
        evidence=list(evidence),
        confidence=ConfidenceBreakdown(raw=raw),
    )


# ---------------------------------------------------------------------------
# Determinism and ordering
# ---------------------------------------------------------------------------


def test_aggregate_is_deterministic_across_repeat_calls() -> None:
    """Same input -> byte-identical output. The core purity claim."""
    items = [box_evidence([0.1, 0.1, 0.2, 0.2]), stat_evidence(), box_evidence(
        [0.5, 0.5, 0.6, 0.6], score=0.9
    )]
    engine = EvidenceEngine()

    first = engine.aggregate(evidence=items)
    second = engine.aggregate(evidence=items)

    assert first.ids() == second.ids()
    assert evidence_digest(first) == evidence_digest(second)
    assert [e.model_dump() for e in first.items] == [
        e.model_dump() for e in second.items
    ]


def test_order_is_independent_of_input_order() -> None:
    """Specialists finish in whatever order they finish. The collection must not."""
    a = box_evidence([0.1, 0.1, 0.2, 0.2], score=0.30)
    b = box_evidence([0.7, 0.7, 0.8, 0.8], score=0.90)
    c = stat_evidence(score=0.60)

    engine = EvidenceEngine()
    forward = engine.aggregate(evidence=[a, b, c])
    backward = engine.aggregate(evidence=[c, b, a])

    assert evidence_digest(forward) == evidence_digest(backward)
    assert [e.type for e in forward.items] == [e.type for e in backward.items]


def test_equal_scores_order_deterministically_by_coordinates() -> None:
    """Ties must not fall back to insertion order -- that is input-order
    dependence wearing a disguise. Coordinates break the tie totally."""
    low = box_evidence([0.1, 0.1, 0.2, 0.2], score=0.5)
    high = box_evidence([0.8, 0.8, 0.9, 0.9], score=0.5)

    engine = EvidenceEngine()
    assert engine.aggregate(evidence=[low, high]).ids() == engine.aggregate(
        evidence=[high, low]
    ).ids()
    assert engine.aggregate(evidence=[high, low]).items[0].coordinates == [
        0.1, 0.1, 0.2, 0.2
    ]


def test_higher_score_sorts_first_within_a_type() -> None:
    weak = stat_evidence(score=0.2, specialist="vlm")
    strong = stat_evidence(score=0.9, specialist="vlm")
    collection = EvidenceEngine().aggregate(evidence=[weak, strong])
    assert collection.items[0].score == pytest.approx(0.9)


def test_unscored_items_sort_after_scored_items() -> None:
    """A missing score is not evidence of strength."""
    scored = stat_evidence(score=0.1, specialist="vlm")
    unscored = stat_evidence(score=None, specialist="vlm")
    collection = EvidenceEngine().aggregate(evidence=[unscored, scored])
    assert collection.items[0].score == pytest.approx(0.1)
    assert collection.items[-1].score is None


def test_types_are_grouped_together() -> None:
    items = [
        stat_evidence(specialist="vlm"),
        box_evidence([0.1, 0.1, 0.2, 0.2]),
        stat_evidence(specialist="change", payload={"other": True}),
    ]
    types = [e.type for e in EvidenceEngine().aggregate(evidence=items).items]
    assert types == sorted(types, key=lambda t: t.value)


# ---------------------------------------------------------------------------
# Id assignment
# ---------------------------------------------------------------------------


def test_ids_are_sequential_and_zero_padded() -> None:
    # Distinct scores keep these as distinct claims; see the dedup note above.
    items = [stat_evidence(score=0.1 * i, specialist="vlm") for i in range(1, 4)]
    collection = EvidenceEngine().aggregate(evidence=items)
    assert collection.ids() == ["evidence_001", "evidence_002", "evidence_003"]


def test_payload_only_differences_are_deduplicated_as_one_claim() -> None:
    """The identity key ignores payload, so items differing only in payload are
    the same claim. Pinned because it is the behaviour the id/cap tests rely on,
    and because it is the deliberate choice: a `crs` in a payload is a detail of
    the claim, not a different claim."""
    items = [stat_evidence(score=0.5, specialist="vlm", payload={"i": i})
             for i in range(4)]
    collection = EvidenceEngine().aggregate(evidence=items)

    assert len(collection) == 1
    assert collection.dropped_duplicates == 3


def test_ids_restart_from_one_for_each_aggregation() -> None:
    engine = EvidenceEngine()
    first = engine.aggregate(evidence=[stat_evidence(score=0.3)])
    second = engine.aggregate(evidence=[stat_evidence(score=0.7)])
    assert first.ids() == second.ids() == ["evidence_001"]


def test_source_results_are_not_mutated() -> None:
    """The engine renumbers a COPY. A caller's result must survive intact."""
    original = box_evidence([0.1, 0.1, 0.2, 0.2], specialist="grounding")
    before = original.evidence_id

    collection = EvidenceEngine().aggregate(evidence=[original])

    assert original.evidence_id == before
    assert collection.items[0].evidence_id == "evidence_001"
    assert collection.items[0] is not original


def test_result_objects_are_not_mutated() -> None:
    result = result_with([box_evidence([0.1, 0.1, 0.2, 0.2]), stat_evidence()])
    snapshot = json.loads(result.model_dump_json())

    EvidenceEngine().aggregate(results=result)

    assert json.loads(result.model_dump_json()) == snapshot


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_identical_items_from_one_specialist_collapse() -> None:
    dupe = box_evidence([0.3, 0.3, 0.4, 0.4], score=0.5, specialist="grounding")
    collection = EvidenceEngine().aggregate(
        evidence=[dupe, dupe.model_copy(deep=True)]
    )

    assert len(collection) == 1
    assert collection.dropped_duplicates == 1


def test_deduplication_ignores_random_uuid() -> None:
    """Two structurally identical items have different uuids by default. If the
    key included the uuid, dedup would never fire in production."""
    a = box_evidence([0.3, 0.3, 0.4, 0.4])
    b = box_evidence([0.3, 0.3, 0.4, 0.4])
    assert a.evidence_id != b.evidence_id  # precondition

    collection = EvidenceEngine().aggregate(evidence=[a, b])
    assert len(collection) == 1


def test_deduplication_ignores_payload_differences_and_merges_them() -> None:
    """Same claim, different payload. The claim is one item -- but the payload
    the loser carried must not be lost."""
    a = box_evidence([0.3, 0.3, 0.4, 0.4], payload={"crs": "EPSG:32643"})
    b = box_evidence([0.3, 0.3, 0.4, 0.4], payload={"rank": 0})

    collection = EvidenceEngine().aggregate(evidence=[a, b])
    assert len(collection) == 1
    assert collection.items[0].payload["crs"] == "EPSG:32643"
    assert collection.items[0].payload["rank"] == 0


def test_different_coordinates_are_not_deduplicated() -> None:
    """Two boxes that disagree are two claims. Merging them would hide a
    spatial disagreement, which the freeze forbids."""
    a = box_evidence([0.1, 0.1, 0.2, 0.2])
    b = box_evidence([0.8, 0.8, 0.9, 0.9])
    assert len(EvidenceEngine().aggregate(evidence=[a, b])) == 2


def test_different_coordinate_systems_are_not_deduplicated() -> None:
    """The same four numbers in `pixel` and in `geo` are different claims."""
    normalized = box_evidence([10.0, 10.0, 20.0, 20.0])
    pixel = Evidence(
        type=EvidenceType.BOUNDING_BOX,
        source_specialist="grounding",
        coordinate_system=CoordinateSystem.PIXEL,
        coordinates=[10.0, 10.0, 20.0, 20.0],
        score=0.5,
    )
    assert len(EvidenceEngine().aggregate(evidence=[normalized, pixel])) == 2


def test_same_claim_from_two_specialists_is_kept_and_marked_corroborated() -> None:
    """`source_specialist` is a single string, so two specialists agreeing are
    two items. Both survive -- attribution is never thrown away -- and each
    records the other in its payload."""
    from_vlm = stat_evidence(score=0.7, specialist="vlm")
    from_change = stat_evidence(score=0.7, specialist="change")

    collection = EvidenceEngine().aggregate(evidence=[from_vlm, from_change])

    assert len(collection) == 2
    specialists = {e.source_specialist for e in collection.items}
    assert specialists == {"vlm", "change"}
    for item in collection.items:
        assert len(item.payload["corroborated_by"]) == 1
    by_name = {e.source_specialist: e for e in collection.items}
    assert by_name["vlm"].payload["corroborated_by"] == ["change"]
    assert by_name["change"].payload["corroborated_by"] == ["vlm"]


def test_unrelated_items_carry_no_corroboration_marker() -> None:
    items = [stat_evidence(score=0.1, specialist="vlm", payload={"a": 1}),
             stat_evidence(score=0.9, specialist="vlm", payload={"b": 2})]
    collection = EvidenceEngine().aggregate(evidence=items)
    assert all("corroborated_by" not in e.payload for e in collection.items)


# ---------------------------------------------------------------------------
# Aggregating across specialists
# ---------------------------------------------------------------------------


def test_multiple_results_aggregate_without_loss() -> None:
    vqa = result_with([stat_evidence(score=0.8, specialist="vlm")], task=Task.VQA)
    grounding = result_with(
        [box_evidence([0.1, 0.1, 0.2, 0.2], specialist="grounding")],
        task=Task.GROUNDING,
    )
    change = result_with(
        [stat_evidence(score=0.4, specialist="change", payload={"n": 3})],
        task=Task.CHANGE,
    )

    collection = EvidenceEngine().aggregate([vqa, grounding, change])

    assert len(collection) == 3
    assert collection.sources == ["change", "grounding", "vlm"]
    assert {e.source_specialist for e in collection.items} == {
        "vlm", "grounding", "change"
    }


def test_aggregation_does_not_double_count_a_specialists_own_evidence() -> None:
    """`SpecialistResult.evidence` and `Specialist.produce_evidence` expose the
    same list. Reading both would double every item."""
    result = result_with([box_evidence([0.1, 0.1, 0.2, 0.2]), stat_evidence()])
    collection = EvidenceEngine().aggregate([result, result])

    # Two results, same three-claim set... the second result repeats the first
    # exactly, so the same-specialist items collapse.
    assert len(collection) <= 2


def test_accepts_a_single_result_without_wrapping() -> None:
    result = result_with([box_evidence([0.1, 0.1, 0.2, 0.2])])
    assert len(EvidenceEngine().aggregate(result)) == 1


def test_empty_input_yields_an_empty_collection() -> None:
    collection = EvidenceEngine().aggregate(evidence=[])
    assert len(collection) == 0
    assert collection.sources == []
    assert collection.truncated is False


def test_a_result_with_no_evidence_contributes_nothing() -> None:
    result = SpecialistResult(
        task=Task.VQA, answer="nothing measured", confidence=ConfidenceBreakdown(raw=0.5)
    )
    collection = EvidenceEngine().aggregate([result])
    assert len(collection) == 0


def test_both_results_and_evidence_is_rejected() -> None:
    with pytest.raises(ValueError):
        EvidenceEngine().aggregate([result_with([])], evidence=[])


def test_neither_results_nor_evidence_is_rejected() -> None:
    with pytest.raises(ValueError):
        EvidenceEngine().aggregate()


def test_zero_max_items_is_rejected() -> None:
    """A cap of zero silently discards every piece of evidence. Refuse it."""
    with pytest.raises(ValueError):
        EvidenceEngine(max_items=0)


# ---------------------------------------------------------------------------
# The cap
# ---------------------------------------------------------------------------


def test_cap_is_respected_and_the_drop_is_counted() -> None:
    # Distinct SCORES, not distinct payloads: the identity key deliberately
    # ignores payload, so ten items differing only by payload are one claim.
    items = [stat_evidence(score=i / 20.0, specialist="vlm") for i in range(10)]
    collection = EvidenceEngine(max_items=3).aggregate(evidence=items)

    assert len(collection) == 3
    assert collection.dropped_over_limit == 7
    assert collection.total_before_limit == 10
    assert collection.truncated is True


def test_no_truncation_flag_when_under_the_cap() -> None:
    collection = EvidenceEngine(max_items=8).aggregate(evidence=[stat_evidence()])
    assert collection.truncated is False
    assert collection.dropped_over_limit == 0


def test_default_cap_matches_the_config_value() -> None:
    assert DEFAULT_MAX_ITEMS == 32


def test_sources_are_reported_before_the_cap_is_applied() -> None:
    """Provenance is a property of the run, not of the surviving items."""
    items = [stat_evidence(score=0.9, specialist="vlm", payload={"i": 0}),
             box_evidence([0.1, 0.1, 0.2, 0.2], specialist="grounding",
                          payload={"i": 1})]
    collection = EvidenceEngine(max_items=1).aggregate(evidence=items)
    assert len(collection) == 1
    assert collection.sources == ["grounding", "vlm"]


# ---------------------------------------------------------------------------
# Canonical vocabulary
# ---------------------------------------------------------------------------


def test_every_evidence_type_member_is_reachable() -> None:
    """The canonical vocabulary is exactly `EvidenceType`. Nothing invented."""
    for member in EvidenceType:
        item = Evidence(type=member, source_specialist="test", score=0.5)
        assert EvidenceEngine().aggregate(evidence=[item]).items[0].type is member


@pytest.mark.parametrize(
    "task,expected",
    [
        (Task.GROUNDING, EvidenceType.BOUNDING_BOX),
        (Task.CHANGE, EvidenceType.CHANGE_MAP),
        (Task.OPTICAL_SAR, EvidenceType.JOINT_FEATURE_REGION),
        (Task.VQA, EvidenceType.STATISTIC),
        (Task.CAPTION, EvidenceType.STATISTIC),
    ],
)
def test_task_to_canonical_type_mapping(task: Task, expected: EvidenceType) -> None:
    result = SpecialistResult(task=task, confidence=ConfidenceBreakdown(raw=0.5))
    assert EvidenceEngine.evidence_type_for(result) is expected


def test_box_wrapper_preserves_coordinate_system() -> None:
    """A box in `pixel` must not be relabelled as normalized on the way in."""
    box = Box(x1=1.0, y1=2.0, x2=3.0, y2=4.0, score=0.5,
              coordinate_system=CoordinateSystem.PIXEL)
    item = EvidenceEngine().evidence_from_box(box, source_specialist="grounding")
    assert item.coordinate_system is CoordinateSystem.PIXEL
    assert item.coordinates == [1.0, 2.0, 3.0, 4.0]


def test_region_with_a_mask_ref_becomes_mask_evidence() -> None:
    region = Region(mask_ref="masks/one.png", score=0.5)
    item = EvidenceEngine().evidence_from_region(region, source_specialist="change")
    assert item.type is EvidenceType.MASK
    assert item.artifact_ref == "masks/one.png"


def test_region_with_only_a_box_becomes_box_evidence() -> None:
    box = Box(x1=0.0, y1=0.0, x2=0.5, y2=0.5)
    region = Region(box=box)
    item = EvidenceEngine().evidence_from_region(region, source_specialist="change")
    assert item.type is EvidenceType.BOUNDING_BOX


def test_region_with_neither_geometry_is_a_statistic_not_a_spatial_claim() -> None:
    item = EvidenceEngine().evidence_from_region(
        Region(label="unlocated"), source_specialist="change"
    )
    assert item.type is EvidenceType.STATISTIC
    assert item.coordinates is None


def test_geolocation_evidence_carries_the_crs() -> None:
    geo = GeoMetadata(crs="EPSG:32643", has_crs=True, width=128, height=128)
    item = EvidenceEngine().evidence_from_geospatial(geo, source_specialist="grounding")
    assert item is not None
    assert item.type is EvidenceType.GEOLOCATION
    assert item.payload["crs"] == "EPSG:32643"


def test_no_geolocation_evidence_without_a_crs() -> None:
    """Emitting a placement the data does not support is the failure mode the
    grounding degenerate-box guard exists to prevent. Same rule applies here."""
    item = EvidenceEngine().evidence_from_geospatial(
        GeoMetadata(), source_specialist="grounding"
    )
    assert item is None


# ---------------------------------------------------------------------------
# Confidence: uncalibrated pass-through (the honesty rule)
# ---------------------------------------------------------------------------


def test_no_calibration_passes_raw_through_and_says_so() -> None:
    breakdown = calibrate(0.73, None)
    assert breakdown.method == METHOD_UNCALIBRATED
    assert breakdown.calibrated is None
    assert breakdown.value == pytest.approx(0.73), "value must fall back to raw"
    assert breakdown.components["calibrated_applied"] == 0.0


def test_uncalibrated_is_never_labelled_as_fitted() -> None:
    """The false-claim guard: no artifact -> no calibration claim."""
    for raw in (0.0, 0.25, 0.5, 0.99, 1.0):
        breakdown = calibrate(raw, None)
        assert breakdown.method != METHOD_TEMPERATURE
        assert breakdown.calibrated is None


def test_identity_temperature_is_reported_as_uncalibrated() -> None:
    """A fitted T of exactly 1.0 learned nothing. Claiming a correction would be
    a false claim of reliability."""
    calibration = TemperatureCalibration(temperature=1.0, fitted_on="valid")
    assert calibration.is_effective() is False

    breakdown = calibrate(0.6, calibration)
    assert breakdown.method == METHOD_UNCALIBRATED
    assert breakdown.calibrated is None
    assert breakdown.value == pytest.approx(0.6)
    assert breakdown.components["calibration_identity"] == 1.0
    assert breakdown.components["temperature"] == pytest.approx(1.0)


def test_engine_without_calibration_does_not_claim_calibration() -> None:
    result = result_with([stat_evidence()], raw=0.42)
    breakdown = EvidenceEngine().confidence_for(result)
    assert breakdown.method == METHOD_UNCALIBRATED
    assert breakdown.calibrated is None
    assert breakdown.value == pytest.approx(0.42)


def test_raw_outside_unit_range_is_clamped_not_rejected() -> None:
    assert calibrate(1.5, None).raw == 1.0
    assert calibrate(-0.5, None).raw == 0.0


def test_non_finite_raw_is_refused() -> None:
    """A NaN is an upstream bug. Clamping it to 0.5 would invent a measurement."""
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            calibrate(bad, None)


# ---------------------------------------------------------------------------
# Confidence: temperature scaling
# ---------------------------------------------------------------------------


def test_temperature_scaling_is_applied_when_fitted() -> None:
    calibration = TemperatureCalibration(temperature=2.0, fitted_on="valid")
    breakdown = calibrate(0.73, calibration)

    assert breakdown.method == METHOD_TEMPERATURE
    assert breakdown.calibrated is not None
    assert breakdown.calibrated < 0.73, "T > 1 must soften a confident score"
    assert breakdown.value == pytest.approx(breakdown.calibrated)
    assert breakdown.components["calibrated_applied"] == 1.0


def test_temperature_above_one_softens_and_below_one_sharpens() -> None:
    soft = calibrate(0.8, TemperatureCalibration(temperature=3.0)).calibrated
    sharp = calibrate(0.8, TemperatureCalibration(temperature=0.5)).calibrated
    assert soft is not None and sharp is not None
    assert soft < 0.8 < sharp


def test_temperature_scaling_is_monotonic() -> None:
    """Calibration may not reorder scores -- only respace them."""
    calibration = TemperatureCalibration(temperature=2.5)
    raws = [0.05, 0.2, 0.5, 0.75, 0.95]
    out = [calibrate(r, calibration).calibrated for r in raws]
    assert all(a is not None and b is not None for a, b in zip(out, out[1:]))
    assert out == sorted(out)


def test_temperature_scaling_is_deterministic() -> None:
    calibration = TemperatureCalibration(temperature=1.7)
    assert calibrate(0.42, calibration).calibrated == calibrate(
        0.42, calibration
    ).calibrated


def test_endpoint_scores_do_not_produce_nan() -> None:
    """logit(0) and logit(1) are infinite; the clamping must keep it finite."""
    calibration = TemperatureCalibration(temperature=2.0)
    for raw in (0.0, 1.0):
        breakdown = calibrate(raw, calibration)
        assert breakdown.calibrated is not None
        assert 0.0 <= breakdown.calibrated <= 1.0


def test_invalid_temperatures_are_rejected() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            TemperatureCalibration(temperature=bad)


def test_specialist_degradation_survives_calibration() -> None:
    """Calibrating must not launder a degraded result into a confident one."""
    degraded = ConfidenceBreakdown(
        raw=0.9, method="uncalibrated", degraded=True,
        degradation_reason="zero-shot fallback: no trained head loaded",
    )
    out = EvidenceEngine(
        calibration=TemperatureCalibration(temperature=2.0)
    ).confidence_for(degraded)

    assert out.degraded is True
    assert out.degradation_reason == "zero-shot fallback: no trained head loaded"
    assert out.method == METHOD_TEMPERATURE


def test_specialist_components_are_preserved_through_calibration() -> None:
    original = ConfidenceBreakdown(
        raw=0.5,
        components={"max_objectness": 0.4, "used_head": 1.0},
        degraded=False,
    )
    out = EvidenceEngine(
        calibration=TemperatureCalibration(temperature=2.0)
    ).confidence_for(original)

    assert out.components["max_objectness"] == pytest.approx(0.4)
    assert out.components["used_head"] == pytest.approx(1.0)
    assert out.components["temperature"] == pytest.approx(2.0)


def test_engine_confidence_from_a_result_preserves_components() -> None:
    result = SpecialistResult(
        task=Task.GROUNDING,
        answer="a",
        confidence=ConfidenceBreakdown(
            raw=0.6, components={"score_contrast": 0.2}, degraded=True,
            degradation_reason="zero-shot fallback",
        ),
    )
    out = EvidenceEngine().confidence_for(result)
    assert out.components["score_contrast"] == pytest.approx(0.2)
    assert out.degraded is True
    assert out.degradation_reason == "zero-shot fallback"


def test_calibrate_result_helper_preserves_provenance() -> None:
    breakdown = ConfidenceBreakdown(
        raw=0.7, components={"a": 1.0}, degraded=True, degradation_reason="why"
    )
    out = calibrate_result(breakdown, TemperatureCalibration(temperature=2.0))
    assert out.components["a"] == pytest.approx(1.0)
    assert out.degraded is True
    assert out.degradation_reason == "why"
    assert out.method == METHOD_TEMPERATURE


# ---------------------------------------------------------------------------
# Calibration artifact I/O
# ---------------------------------------------------------------------------


def test_artifact_round_trips_through_json(scratch: Path) -> None:
    path = scratch / "calibration_v001.json"
    path.write_text(
        json.dumps({"temperature": 1.8, "fitted_on": "valid", "n_samples": 500}),
        encoding="utf-8",
    )
    calibration = TemperatureCalibration.from_json(path)
    assert calibration is not None
    assert calibration.temperature == pytest.approx(1.8)
    assert calibration.n_samples == 500
    assert calibration.artifact == str(path)


def test_nested_artifact_form_is_accepted(scratch: Path) -> None:
    path = scratch / "nested.json"
    path.write_text(
        json.dumps({"temperature_scaling": {"temperature": 1.25}}), encoding="utf-8"
    )
    calibration = TemperatureCalibration.from_json(path)
    assert calibration is not None
    assert calibration.temperature == pytest.approx(1.25)


def test_missing_artifact_degrades_instead_of_failing(scratch: Path) -> None:
    """An HF Space may ship without the fitted artifact. That is a legitimate
    deployment state, not an error."""
    assert TemperatureCalibration.from_json(scratch / "absent.json") is None


def test_missing_artifact_raises_when_required(scratch: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TemperatureCalibration.from_json(scratch / "absent.json", required=True)


def test_malformed_artifact_is_not_silently_swallowed(scratch: Path) -> None:
    """A corrupt artifact treated as 'no artifact' would hide an operational
    defect. It raises instead."""
    path = scratch / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        TemperatureCalibration.from_json(path)


def test_artifact_without_a_temperature_is_rejected(scratch: Path) -> None:
    path = scratch / "incomplete.json"
    path.write_text(json.dumps({"fitted_on": "valid"}), encoding="utf-8")
    with pytest.raises(ValueError):
        TemperatureCalibration.from_json(path)


class _StubConfig:
    def __init__(self, data: dict) -> None:
        self._data = data

    def get(self, path: str, default=None):
        node = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def test_load_calibration_respects_the_master_switch(scratch: Path) -> None:
    """The frozen key is `confidence.temperature_scaling`."""
    (scratch / "calibration_v001.json").write_text(
        json.dumps({"temperature": 1.5}), encoding="utf-8"
    )
    config = _StubConfig(
        {"confidence": {"temperature_scaling": False,
                        "calibration_file": "calibration_v001.json"}}
    )
    assert load_calibration(config, base_dir=scratch) is None


def test_load_calibration_reads_the_configured_file(scratch: Path) -> None:
    (scratch / "calibration_v001.json").write_text(
        json.dumps({"temperature": 1.5}), encoding="utf-8"
    )
    config = _StubConfig(
        {"confidence": {"temperature_scaling": True,
                        "calibration_file": "calibration_v001.json"}}
    )
    calibration = load_calibration(config, base_dir=scratch)
    assert calibration is not None
    assert calibration.temperature == pytest.approx(1.5)


def test_load_calibration_without_a_file_configured_is_none(scratch: Path) -> None:
    config = _StubConfig({"confidence": {"temperature_scaling": True}})
    assert load_calibration(config, base_dir=scratch) is None


def test_engine_from_config_uses_the_configured_cap() -> None:
    engine = EvidenceEngine.from_config(_StubConfig({"evidence": {"max_items": 5}}))
    assert engine.max_items == 5


def test_engine_from_config_falls_back_to_the_default_cap() -> None:
    engine = EvidenceEngine.from_config(_StubConfig({}))
    assert engine.max_items == DEFAULT_MAX_ITEMS


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_aggregate_evidence_shorthand_matches_the_engine() -> None:
    result = result_with([box_evidence([0.1, 0.1, 0.2, 0.2]), stat_evidence()])
    shorthand = aggregate_evidence([result])
    direct = EvidenceEngine().aggregate([result])
    assert shorthand.ids() == direct.ids()
    assert evidence_digest(shorthand) == evidence_digest(direct)


def test_collection_helpers_filter_correctly() -> None:
    items = [
        box_evidence([0.1, 0.1, 0.2, 0.2], specialist="grounding"),
        stat_evidence(score=0.5, specialist="vlm"),
        stat_evidence(score=0.9, specialist="change", payload={"x": 1}),
    ]
    collection = EvidenceEngine().aggregate(evidence=items)

    assert len(collection.by_type(EvidenceType.STATISTIC)) == 2
    assert len(collection.by_specialist("grounding")) == 1
    assert len(collection.by_specialist("vlm")) == 1
    assert collection.summary()["returned"] == 3


def test_collection_summary_reports_the_observable_facts() -> None:
    items = [stat_evidence(score=i / 20.0, specialist="vlm") for i in range(5)]
    summary = EvidenceEngine(max_items=2).aggregate(evidence=items).summary()

    assert summary["returned"] == 2
    assert summary["total_before_limit"] == 5
    assert summary["dropped_over_limit"] == 3
    assert summary["truncated"] is True
    assert summary["sources"] == ["vlm"]
    assert summary["types"] == ["statistic"]


def test_digest_changes_when_a_claim_changes() -> None:
    a = EvidenceEngine().aggregate(evidence=[stat_evidence(score=0.5)])
    b = EvidenceEngine().aggregate(evidence=[stat_evidence(score=0.6)])
    assert evidence_digest(a) != evidence_digest(b)


def test_digest_is_stable_across_regenerated_uuids() -> None:
    """The digest tracks claims, not container identity, so it is usable as a
    reproducibility assertion across processes."""
    first = EvidenceEngine().aggregate(evidence=[stat_evidence(score=0.5)])
    second = EvidenceEngine().aggregate(evidence=[stat_evidence(score=0.5)])
    assert first.items[0].evidence_id != stat_evidence(score=0.5).evidence_id
    assert evidence_digest(first) == evidence_digest(second)


def test_collection_is_iterable_and_sized() -> None:
    collection = EvidenceEngine().aggregate(
        evidence=[stat_evidence(score=0.5, specialist="vlm", payload={"a": 1}),
                  stat_evidence(score=0.6, specialist="vlm", payload={"b": 2})]
    )
    assert isinstance(collection, EvidenceCollection)
    assert len(collection) == 2
    assert len(list(collection)) == 2


def test_serialises_through_the_schema_serialiser() -> None:
    """The aggregated evidence must survive the same JSON path the result does."""
    result = result_with([box_evidence([0.1, 0.1, 0.2, 0.2]), stat_evidence()])
    collection = EvidenceEngine().aggregate([result])

    payload = json.loads(json.dumps([e.model_dump(mode="json")
                                     for e in collection.items]))
    assert [e["evidence_id"] for e in payload] == ["evidence_001", "evidence_002"]


def test_module_public_surface_is_declared() -> None:
    import evidence

    for name in evidence.__all__:
        assert hasattr(evidence, name), f"{name} is exported but not importable"
