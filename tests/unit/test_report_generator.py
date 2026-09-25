"""Unit tests for `reports.generator` — pure, deterministic, fixtures only.

These tests pin the report CONTRACT, not any particular run: the exact field
set, the honesty rule for confidence, evidence round-tripping with its
coordinate system, and byte-stability of `render_json`. No model, no I/O beyond
`tmp_path`, no network.
"""

from __future__ import annotations

import json

from core.schemas import (
    ConfidenceBreakdown,
    ControllerState,
    CoordinateSystem,
    Evidence,
    EvidenceType,
    ExecutionTrace,
    ResultEnvelope,
    SpecialistResult,
    Task,
    TraceStep,
)

from reports.generator import (
    REPORT_FIELDS,
    REPORT_SCHEMA_VERSION,
    UNCALIBRATED_METHOD,
    build_report,
    render_json,
    write_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _confidence(
    *,
    raw: float = 0.5,
    calibrated: float | None = None,
    method: str = UNCALIBRATED_METHOD,
    degraded: bool = False,
    reason: str | None = None,
) -> ConfidenceBreakdown:
    return ConfidenceBreakdown(
        raw=raw,
        calibrated=calibrated,
        method=method,
        degraded=degraded,
        degradation_reason=reason,
    )


def _result(
    *,
    answer: str = "an answer",
    evidence: list[Evidence] | None = None,
    confidence: ConfidenceBreakdown | None = None,
    warnings: list[str] | None = None,
    degraded: bool = False,
    task: Task = Task.VQA,
) -> SpecialistResult:
    return SpecialistResult(
        task=task,
        answer=answer,
        evidence=list(evidence or []),
        confidence=confidence if confidence is not None else _confidence(),
        warnings=list(warnings or []),
        degraded=degraded,
    )


# ---------------------------------------------------------------------------
# Field set
# ---------------------------------------------------------------------------
def test_field_set_is_exactly_the_contract() -> None:
    report = build_report(_result())
    assert set(report) == REPORT_FIELDS
    assert report["report_schema_version"] == REPORT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# The honesty rule — never fabricate a calibrated confidence
# ---------------------------------------------------------------------------
def test_calibrated_none_renders_as_null_and_uncalibrated() -> None:
    report = build_report(
        _result(confidence=_confidence(raw=0.4, calibrated=None))
    )
    assert report["confidence"]["calibrated"] is None
    assert report["confidence"]["method"] == "uncalibrated"


def test_an_inconsistent_source_cannot_claim_a_calibration() -> None:
    """`calibrated is None` must win over a method that claims otherwise."""
    report = build_report(
        _result(
            confidence=_confidence(
                raw=0.4, calibrated=None, method="temperature_scaling"
            )
        )
    )
    assert report["confidence"]["calibrated"] is None
    assert report["confidence"]["method"] == "uncalibrated"


def test_never_fills_in_a_calibrated_value_when_absent() -> None:
    report = build_report(_result(confidence=_confidence(raw=0.42, calibrated=None)))
    # The raw score is preserved verbatim; nothing is invented to fill the gap.
    assert report["confidence"]["raw"] == 0.42
    assert report["confidence"]["calibrated"] is None


def test_a_real_calibrated_value_is_passed_through() -> None:
    report = build_report(
        _result(
            confidence=_confidence(
                raw=0.4, calibrated=0.55, method="temperature_scaling"
            )
        )
    )
    assert report["confidence"]["calibrated"] == 0.55
    assert report["confidence"]["method"] == "temperature_scaling"


def test_degradation_fields_are_carried_through() -> None:
    report = build_report(
        _result(
            degraded=True,
            confidence=_confidence(degraded=True, reason="no fitted artifact"),
        )
    )
    assert report["degraded"] is True
    assert report["confidence"]["degraded"] is True
    assert report["confidence"]["degradation_reason"] == "no fitted artifact"


# ---------------------------------------------------------------------------
# Evidence round-trips verbatim
# ---------------------------------------------------------------------------
def test_evidence_round_trips_with_its_coordinate_system() -> None:
    evidence = Evidence(
        type=EvidenceType.BOUNDING_BOX,
        source_specialist="grounding",
        coordinate_system=CoordinateSystem.NORMALIZED_0_1,
        coordinates=[0.1, 0.2, 0.3, 0.4],
        score=0.6,
        payload={"label": "region"},
    )
    report = build_report(_result(evidence=[evidence]))

    assert len(report["evidence"]) == 1
    item = report["evidence"][0]
    assert item["type"] == "bounding_box"
    assert item["source_specialist"] == "grounding"
    assert item["coordinate_system"] == "normalized_0_1"
    assert item["coordinates"] == [0.1, 0.2, 0.3, 0.4]
    assert item["payload"] == {"label": "region"}


# ---------------------------------------------------------------------------
# Deterministic rendering
# ---------------------------------------------------------------------------
def test_render_json_is_byte_stable_across_two_calls() -> None:
    report = build_report(_result())
    assert render_json(report) == render_json(report)


def test_render_json_is_independent_of_key_insertion_order() -> None:
    report = build_report(_result())
    reordered = {key: report[key] for key in reversed(list(report))}
    assert render_json(report) == render_json(reordered)


def test_render_json_preserves_non_ascii_readably() -> None:
    report = build_report(_result(answer="café — naïve"))
    rendered = render_json(report)
    assert "café" in rendered
    assert "\\u" not in rendered


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def test_write_report_creates_parent_dirs_and_writes_utf8(tmp_path) -> None:
    report = build_report(_result(answer="café — naïve"))
    destination = tmp_path / "nested" / "dir" / "report.json"

    written = write_report(report, destination)

    assert written == destination
    assert destination.exists()
    text = destination.read_text(encoding="utf-8")
    assert json.loads(text) == report


# ---------------------------------------------------------------------------
# Trace states
# ---------------------------------------------------------------------------
def test_trace_states_come_from_the_envelope_trace() -> None:
    result = _result()
    trace = ExecutionTrace(
        run_id="run_test",
        steps=[
            TraceStep(state=ControllerState.RECEIVE),
            TraceStep(state=ControllerState.RESPOND),
        ],
    )
    envelope = ResultEnvelope(run_id="run_test", result=result, trace=trace)

    report = build_report(result, envelope=envelope)

    assert report["trace_states"] == ["RECEIVE", "RESPOND"]


def test_explicit_trace_argument_is_used() -> None:
    result = _result()
    trace = ExecutionTrace(
        run_id="run_test", steps=[TraceStep(state=ControllerState.PLAN)]
    )
    report = build_report(result, trace=trace)
    assert report["trace_states"] == ["PLAN"]


def test_trace_states_are_empty_without_a_trace() -> None:
    report = build_report(_result())
    assert report["trace_states"] == []
