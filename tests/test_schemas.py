"""Phase 1 tests — canonical schema contract.

These tests are the executable form of docs/ARCHITECTURE_FREEZE.md section 3.
If one of them fails, a specialist could emit a result the rest of the system
cannot consume.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from core.schemas import (
    SCHEMA_VERSION,
    AnalysisRequest,
    Box,
    ChangeRegion,
    ConfidenceBreakdown,
    ControllerState,
    CoordinateSystem,
    Evidence,
    EvidenceType,
    ExecutionTrace,
    GeoMetadata,
    Intent,
    Modality,
    Region,
    ResultEnvelope,
    SpecialistResult,
    Task,
)


def _conf(raw: float = 0.8) -> ConfidenceBreakdown:
    return ConfidenceBreakdown(raw=raw)


# --------------------------------------------------------------------------
# C-5 — coordinate system is mandatory on spatial output
# --------------------------------------------------------------------------
def test_box_defaults_to_normalized_coordinates() -> None:
    b = Box(x1=0.1, y1=0.2, x2=0.5, y2=0.6)
    assert b.coordinate_system is CoordinateSystem.NORMALIZED_0_1


def test_box_rejects_inverted_coordinates() -> None:
    with pytest.raises(ValidationError):
        Box(x1=0.9, y1=0.9, x2=0.1, y2=0.1)


def test_box_rejects_out_of_range_score() -> None:
    with pytest.raises(ValidationError):
        Box(x1=0.0, y1=0.0, x2=1.0, y2=1.0, score=1.5)


def test_box_accepts_pixel_coordinate_system() -> None:
    b = Box(x1=10, y1=20, x2=100, y2=200,
            coordinate_system=CoordinateSystem.PIXEL)
    assert b.coordinate_system is CoordinateSystem.PIXEL


def test_region_carries_its_own_coordinate_system() -> None:
    r = Region(label="water", coordinate_system=CoordinateSystem.GEO)
    assert r.coordinate_system is CoordinateSystem.GEO
    assert r.region_id.startswith("region_")


# --------------------------------------------------------------------------
# Evidence integrity
# --------------------------------------------------------------------------
def test_spatial_evidence_without_coordinate_system_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Evidence(
            type=EvidenceType.BOUNDING_BOX,
            source_specialist="grounding",
            coordinates=[0.1, 0.2, 0.5, 0.6],
        )


def test_spatial_evidence_with_coordinate_system_is_accepted() -> None:
    e = Evidence(
        type=EvidenceType.BOUNDING_BOX,
        source_specialist="grounding",
        coordinates=[0.1, 0.2, 0.5, 0.6],
        coordinate_system=CoordinateSystem.NORMALIZED_0_1,
        score=0.88,
    )
    assert e.evidence_id.startswith("ev_")


def test_statistic_evidence_needs_no_coordinate_system() -> None:
    e = Evidence(
        type=EvidenceType.STATISTIC,
        source_specialist="change",
        payload={"changed_pixels": 1234},
    )
    assert e.coordinate_system is None


def test_availability_mask_evidence_is_first_class() -> None:
    # finding C-1: modality trust is evidence, and it lives outside CROMA.
    e = Evidence(
        type=EvidenceType.AVAILABILITY_MASK,
        source_specialist="optical_sar",
        payload={"optical": [1] * 12, "sar": [1, 0]},
    )
    assert e.type is EvidenceType.AVAILABILITY_MASK


def test_evidence_ids_must_be_unique_within_a_result() -> None:
    e = Evidence(type=EvidenceType.STATISTIC, source_specialist="x")
    with pytest.raises(ValidationError):
        SpecialistResult(
            task=Task.VQA,
            confidence=_conf(),
            evidence=[e, e.model_copy()],
        )


# --------------------------------------------------------------------------
# Router intent
# --------------------------------------------------------------------------
def test_intent_requires_confidence_in_range() -> None:
    with pytest.raises(ValidationError):
        Intent(task=Task.VQA, confidence=1.4)


def test_change_intent_forces_temporal_true() -> None:
    i = Intent(task=Task.CHANGE, confidence=0.9, spatial_output=True)
    assert i.temporal is True


def test_optical_sar_intent_infers_modality() -> None:
    i = Intent(task=Task.OPTICAL_SAR, confidence=0.9)
    assert i.modality is Modality.OPTICAL_SAR


def test_intent_records_its_source() -> None:
    i = Intent(task=Task.VQA, confidence=0.5, source="lexical_fallback")
    assert i.source == "lexical_fallback"


def test_intent_rejects_unknown_source() -> None:
    with pytest.raises(ValidationError):
        Intent(task=Task.VQA, confidence=0.5, source="vibes")


# --------------------------------------------------------------------------
# Result consistency
# --------------------------------------------------------------------------
def test_grounding_without_boxes_self_declares_degraded() -> None:
    r = SpecialistResult(task=Task.GROUNDING, confidence=_conf(0.4))
    assert r.degraded is True
    assert any("no boxes" in w for w in r.warnings)


def test_grounding_with_a_box_is_not_degraded() -> None:
    r = SpecialistResult(
        task=Task.GROUNDING,
        confidence=_conf(0.9),
        boxes=[Box(x1=0.1, y1=0.1, x2=0.4, y2=0.4, score=0.9)],
    )
    assert r.degraded is False


def test_change_output_shape_no_longer_manufactures_degraded() -> None:
    """F-16c (owner ruling 2026-09-23) REVERSED this assertion.

    This test used to be named
    `test_change_without_spatial_output_self_declares_degraded` and asserted
    `degraded is True` for exactly this input. It fired on
    `change_map is None and not regions` -- and `change_map` was a PROXY for "a
    map was produced", because before F-16 it held a filesystem path whenever a
    file had been written.

    F-16 made the ref permanently null (a client cannot retrieve it in v1), so
    the proxy died, the clause collapsed to `not regions`, and a SUCCESSFUL
    no-change analysis started reporting `degraded: true`. A null,
    non-retrievable artifact ref was manufacturing a degradation.

    The clause is REMOVED rather than narrowed, so `degraded` for CHANGE is now
    set entirely by the specialist (`specialists/change/specialist.py:345` no
    trained detector; `:365` co-registration too poor for a spatial claim). The
    old name is deliberately NOT reused: it asserted something the ruling has
    made false, and a green test pinning a reversed spec is worse than no test.
    """
    r = SpecialistResult(task=Task.CHANGE, confidence=_conf(0.6))
    assert r.regions == []
    assert r.change_map is None
    assert r.degraded is False
    assert not any("spatial output" in w for w in r.warnings)


def test_change_with_no_regions_but_an_answer_is_not_degraded() -> None:
    """F-16c, in the shape a real run actually produces.

    A change analysis that ran correctly and found nothing above threshold has
    no regions and no retrievable map -- and is a SUCCESS, not a failure. The
    specialist still emits its "No change detected above threshold ..." answer
    (`specialists/change/specialist.py:402-406`), and the fact is carried
    machine-readably by the empty `regions` plus the CHANGE_MAP evidence's
    `n_components_kept` / `total_change_pixels`.
    """
    r = SpecialistResult(
        task=Task.CHANGE,
        answer=(
            "No change detected above threshold 0.50. 0 raw change pixel(s) "
            "remained after filtering."
        ),
        confidence=_conf(0.8),
    )
    assert r.regions == []
    assert r.change_map is None
    assert r.degraded is False
    assert not any("spatial output" in w for w in r.warnings)


def test_change_with_regions_is_not_degraded() -> None:
    r = SpecialistResult(
        task=Task.CHANGE,
        confidence=_conf(0.7),
        regions=[Region(box=Box(x1=0, y1=0, x2=0.2, y2=0.2))],
    )
    assert r.degraded is False


def test_result_schema_carries_every_mandated_field() -> None:
    r = SpecialistResult(task=Task.VQA, confidence=_conf())
    payload = json.loads(r.model_dump_json())
    for field in (
        "task", "answer", "labels", "regions", "boxes", "masks", "change_map",
        "evidence", "confidence", "geospatial", "schema_version",
    ):
        assert field in payload, f"missing mandated field: {field}"


def test_result_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SpecialistResult(task=Task.VQA, confidence=_conf(), mystery=1)


def test_change_region_requires_non_negative_area() -> None:
    with pytest.raises(ValidationError):
        ChangeRegion(
            box=Box(x1=0, y1=0, x2=0.1, y2=0.1),
            area_pixels=-5,
            mean_probability=0.5,
        )


# --------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------
def test_confidence_value_prefers_calibrated() -> None:
    c = ConfidenceBreakdown(raw=0.9, calibrated=0.62, method="temperature")
    assert c.value == 0.62


def test_confidence_value_falls_back_to_raw() -> None:
    assert ConfidenceBreakdown(raw=0.55).value == 0.55


def test_confidence_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        ConfidenceBreakdown(raw=1.2)


def test_degraded_confidence_requires_a_reason_field_exists() -> None:
    c = ConfidenceBreakdown(raw=0.3, degraded=True,
                            degradation_reason="poor registration")
    assert c.degraded and c.degradation_reason


# --------------------------------------------------------------------------
# Execution trace — observable facts only
# --------------------------------------------------------------------------
def test_trace_records_controller_states() -> None:
    t = ExecutionTrace(workflow=["validate", "preprocess", "execute"])
    assert t.run_id.startswith("run_")
    assert ControllerState.PLAN.value == "PLAN"


def test_trace_has_no_reasoning_field() -> None:
    """No chain-of-thought may be representable in the trace schema."""
    fields = set(ExecutionTrace.model_fields)
    forbidden = {"thoughts", "reasoning", "chain_of_thought", "scratchpad", "cot"}
    assert not (fields & forbidden)


def test_trace_carries_config_hash_slot() -> None:
    assert "config_hash" in ExecutionTrace.model_fields


def test_trace_defaults_to_empty_collections() -> None:
    t = ExecutionTrace()
    assert t.fallbacks == [] and t.errors == [] and t.selected_models == []


# --------------------------------------------------------------------------
# Envelopes
# --------------------------------------------------------------------------
def test_analysis_request_requires_at_least_one_asset() -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest(assets=[], query="what is this?")


def test_analysis_request_accepts_a_query_with_assets() -> None:
    r = AnalysisRequest(assets=["a.tif"], query="Show the water body.")
    assert r.force_task is None


def test_result_envelope_serializes_to_json() -> None:
    env = ResultEnvelope(
        run_id="run_test",
        result=SpecialistResult(task=Task.VQA, confidence=_conf()),
        trace=ExecutionTrace(),
    )
    payload = json.loads(env.model_dump_json())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["result"]["task"] == "vqa"


# --------------------------------------------------------------------------
# Geo metadata
# --------------------------------------------------------------------------
def test_geo_metadata_defaults_to_unreferenced() -> None:
    g = GeoMetadata()
    assert g.has_crs is False and g.is_georeferenced is False


def test_geo_metadata_preserves_transform_and_crs() -> None:
    g = GeoMetadata(
        crs="EPSG:32643",
        transform=[10.0, 0.0, 500000.0, 0.0, -10.0, 2500000.0],
        bounds=[500000.0, 2490000.0, 512000.0, 2500000.0],
        resolution=[10.0, 10.0],
        has_crs=True,
        is_georeferenced=True,
    )
    assert len(g.transform) == 6
    assert g.crs == "EPSG:32643"