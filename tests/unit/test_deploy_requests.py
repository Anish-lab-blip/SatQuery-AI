"""Phase 18 §79 DEPLOYMENT: cold-start and sequential-request behaviour.

The plan's §79 DEPLOYMENT section requires a **sequential request test** and a
**cold-start test**. Both were missing (no `sequential` / `cold_start` hits
under `tests/`). These pin the REGISTRY's real construction lifecycle.

WHY THE REAL REGISTRY, WITH RECORDING BUILDERS
----------------------------------------------
A cold-start claim checked against a stub *registry* is vacuous: with a stub,
"no specialist was constructed" is true by definition. So these tests use the
real `core.registry.SpecialistRegistry` (`discover` / `build` / `build_all`) with
the real spec table (`default_specs()`), and inject RECORDING builders through
the documented `builders=` seam (`core/registry.py:300`). Every construction is
therefore counted, and the spec table's own assertions and memoisation are the
thing under test.

WHAT THE COLD-START TEST ACTUALLY CATCHES
-----------------------------------------
It catches a regression that constructs specialists EAGERLY — at composition, at
import, or inside the planner — instead of lazily on the first query that needs
them. That laziness is the whole point of the design (plan §49) and is what
keeps a cold serving process cheap.

`health()` IS THE DOCUMENTED EXCEPTION
-------------------------------------
`core/controller.py:331` calls `self.registry.build_all()`, whose docstring says
"Constructs everything." It is the only production caller of `build_all`, so the
cold-start property holds *only while `health()` is not on the request path*. A
readiness probe that calls `health()` constructs every specialist and silently
falsifies the claim. The contrast test pins that distinction so a future change
routing `health()` onto the normal path cannot quietly invalidate cold start.

Model-free by construction: `device="cpu"` (so `Config.device_preference`, a CUDA
probe, is never evaluated), stub builders, and a stub router (so
`core.controller._route` never imports `router.classifier`). No weight loading,
no CROMA, no torch, no Gradio.
"""

from __future__ import annotations

from typing import Any

from core.config import load_config
from core.controller import AnalysisController
from core.planner import PolicyPlanner
from core.registry import SpecialistRegistry, default_specs
from core.schemas import (
    AnalysisRequest,
    AssetMetadata,
    Box,
    ConfidenceBreakdown,
    ControllerState,
    CoordinateSystem,
    Evidence,
    EvidenceType,
    Intent,
    Modality,
    SpecialistResult,
    Task,
)
from specialists.base import Specialist, SpecialistRequest

#: The hash the shipped artifacts were trained under. Unchanged by these tests.
FROZEN_CONFIG_HASH = "78f1e3700da15aa1"

#: The capability tuple each spec in `default_specs()` declares. A stub that
#: disagreed would fail the registry's post-construction assertion
#: (`core/registry.py:466-473`), which is the point of using the real table.
_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "vqa": ("vqa", "caption"),
    "caption": ("vqa", "caption"),
    "grounding": ("grounding",),
    "change": ("change",),
    "change_vqa": ("change_vqa",),
    "optical_sar": ("optical_sar",),
}

_TASK: dict[str, Task] = {
    "vqa": Task.VQA,
    "caption": Task.CAPTION,
    "grounding": Task.GROUNDING,
    "change": Task.CHANGE,
    "change_vqa": Task.CHANGE_VQA,
    "optical_sar": Task.OPTICAL_SAR,
}


class _RecordingStub(Specialist):
    """A minimal, deterministic specialist that constructs no model."""

    version = "0.0.0-test"

    def __init__(self, capability: str) -> None:
        self.name = capability
        self.capabilities = _CAPABILITIES[capability]
        self.task = _TASK[capability]

    def validate_request(self, request: SpecialistRequest) -> None:
        return None

    def execute(self, request: SpecialistRequest) -> SpecialistResult:
        boxes: list[Box] = []
        if self.task is Task.GROUNDING:
            boxes = [
                Box(
                    x1=0.1,
                    y1=0.1,
                    x2=0.5,
                    y2=0.5,
                    coordinate_system=CoordinateSystem.NORMALIZED_0_1,
                )
            ]
        return SpecialistResult(
            task=self.task,
            answer="stub answer",
            boxes=boxes,
            change_map="demo://change_map" if self.task is Task.CHANGE else None,
            evidence=[
                Evidence(
                    type=EvidenceType.STATISTIC,
                    source_specialist=self.name,
                    score=0.5,
                )
            ],
            confidence=ConfidenceBreakdown(
                raw=0.5, calibrated=None, method="uncalibrated"
            ),
        )

    def produce_evidence(self, result: SpecialistResult) -> list[Evidence]:
        return list(result.evidence)

    def estimate_confidence(self, result: SpecialistResult) -> ConfidenceBreakdown:
        return result.confidence


class _Recorder:
    """Counts specialist constructions per capability, in order."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.order: list[str] = []

    def builders(self) -> dict[str, Any]:
        """A `builders=` override map for every capability in the spec table."""
        return {name: self._builder(name) for name in _CAPABILITIES}

    def _builder(self, capability: str):
        def _build(config: Any, **kwargs: Any) -> _RecordingStub:
            self.counts[capability] = self.counts.get(capability, 0) + 1
            self.order.append(capability)
            return _RecordingStub(capability)

        return _build

    def total(self) -> int:
        return sum(self.counts.values())


class _StubPrediction:
    """Duck-typed router output. Not a `RouterPrediction`, so no torch import."""

    def __init__(self, task: Task) -> None:
        self.intent = Intent(task=task, confidence=0.9, source="learned")
        self.above_threshold = True

    def to_trace(self) -> dict[str, Any]:
        return {"task": self.intent.task.value}


class _StubRouter:
    def __init__(self, task: Task) -> None:
        self._prediction = _StubPrediction(task)

    def route(self, query: str) -> _StubPrediction:
        return self._prediction


def _assets(count: int) -> list[AssetMetadata]:
    return [
        AssetMetadata(path=f"deploy://asset_{index}.tif", modality=Modality.OPTICAL)
        for index in range(count)
    ]


def _controller(
    recorder: _Recorder, task: Task
) -> tuple[SpecialistRegistry, AnalysisController]:
    config = load_config()
    registry = SpecialistRegistry.discover(
        config,
        device="cpu",
        specs=default_specs(),
        builders=recorder.builders(),
    )
    controller = AnalysisController(
        registry=registry,
        planner=PolicyPlanner(registry),
        router=_StubRouter(task),
        config=config,
    )
    return registry, controller


def _run(controller: AnalysisController, query: str, count: int = 1):
    assets = _assets(count)
    return controller.run(
        AnalysisRequest(assets=[asset.path for asset in assets], query=query),
        assets=assets,
    )


# ---------------------------------------------------------------------------
# B2 — cold start
# ---------------------------------------------------------------------------
def test_cold_start_constructs_nothing_until_the_first_query() -> None:
    recorder = _Recorder()
    registry, controller = _controller(recorder, Task.VQA)

    # discover() + controller composition construct no specialist.
    assert recorder.total() == 0
    assert registry.available() == (
        "caption",
        "change",
        "change_vqa",
        "grounding",
        "optical_sar",
        "vqa",
    )

    envelope = _run(controller, "cold start")

    # Exactly the capability the query needed, constructed exactly once.
    assert recorder.counts == {"vqa": 1}
    assert envelope.result.task is Task.VQA
    assert ControllerState.RESPOND in [step.state for step in envelope.trace.steps]
    # The one constructed specialist is the recording stub — not a real model.
    entry = registry.build("vqa")
    assert isinstance(entry.specialist, _RecordingStub)


# ---------------------------------------------------------------------------
# B3 — the health() trap, pinned as a contrast
# ---------------------------------------------------------------------------
def test_health_constructs_every_specialist_so_cold_start_holds_only_without_it() -> None:
    """`health()` calls `build_all()`, which constructs EVERY specialist.

    This is the contrast that keeps the cold-start assertion honest: the property
    is true only while `health()` stays off the request path.
    """
    recorder = _Recorder()
    registry, controller = _controller(recorder, Task.VQA)
    assert recorder.total() == 0

    controller.health()

    assert set(recorder.counts) == set(registry.available())
    assert all(count == 1 for count in recorder.counts.values())


# ---------------------------------------------------------------------------
# B4 — sequential requests
# ---------------------------------------------------------------------------
def test_sequential_requests_do_not_reconstruct_the_specialist() -> None:
    recorder = _Recorder()
    registry, controller = _controller(recorder, Task.VQA)

    first = _run(controller, "first")
    second = _run(controller, "second")

    # Both ran to completion, as distinct runs.
    assert first.run_id != second.run_id
    assert ControllerState.RESPOND in [step.state for step in first.trace.steps]
    assert ControllerState.RESPOND in [step.state for step in second.trace.steps]
    assert first.result.task is Task.VQA
    assert second.result.task is Task.VQA

    # The second request reused the memoised specialist: still one construction.
    assert recorder.counts == {"vqa": 1}
    assert recorder.order == ["vqa"]


# ---------------------------------------------------------------------------
# B5 — the request path does not move the config hash
# ---------------------------------------------------------------------------
def test_deploy_request_path_does_not_move_the_config_hash() -> None:
    recorder = _Recorder()
    registry, controller = _controller(recorder, Task.VQA)

    _run(controller, "hash guard")

    assert load_config().hash == FROZEN_CONFIG_HASH
    assert registry.config.hash == FROZEN_CONFIG_HASH
    assert controller.config.hash == FROZEN_CONFIG_HASH
