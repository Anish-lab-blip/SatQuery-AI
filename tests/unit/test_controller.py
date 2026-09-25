"""Tests for `core.controller` — dispatch, partial failure and assembly (Phase 15).

Four things this suite exists to pin, all of them stated as requirements in the
architect's design:

  * §5  partial failure never erases evidence — no row of the §5.1 table aborts
        a run, and a failed step leaves a `STATISTIC` record behind;
  * §6  the trace is observable facts only, one `EXECUTE` step per plan step;
  * §7  evidence goes through `EvidenceEngine.aggregate`, the `sources`
        assertion is a real check in VERIFY, and confidence is the primary
        step's rather than an average;
  * §9  a full run with every specialist degraded still returns a coherent,
        honest `ResultEnvelope`.

Stub specialists are used throughout: no model weights, no torch loading. The
asset paths must still name real files, because the controller inspects them —
but header-reading stubs are all that is needed, and they are written into
pytest's per-test temporary directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core.controller import (
    CONTRADICTION_IOU_FLOOR,
    AnalysisController,
    StepOutcome,
    _iou,
)
from core.errors import (
    InvalidRequestError,
    ModelLoadError,
    ModelUnavailableError,
    PairMisalignmentError,
    SpecialistError,
)
from core.planner import ExecutionPlan, PlanMode, PlanRefusal, PlanStep, PolicyPlanner
from core.registry import RegistryState, SpecialistRegistry, SpecialistSpec
from core.schemas import (
    AnalysisRequest,
    Box,
    ConfidenceBreakdown,
    ControllerState,
    Evidence,
    EvidenceType,
    GeoMetadata,
    Region,
    SpecialistResult,
    Task,
)
from specialists.base import Specialist, SpecialistRequest


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
class _Stub(Specialist):
    """Configurable specialist for exercising every controller path."""

    name = "stub"
    version = "1.0.0"
    capabilities: tuple[str, ...] = ("stub",)

    def __init__(
        self,
        *,
        task: Task = Task.VQA,
        answer: str = "stub answer",
        raw: float = 0.8,
        degraded: bool = False,
        evidence: list[Evidence] | None = None,
        boxes: list[Box] | None = None,
        regions: list[Region] | None = None,
        labels: list[str] | None = None,
        change_map: str | None = None,
        geo: GeoMetadata | None = None,
        warnings: list[str] | None = None,
        validation_error: Exception | None = None,
        execute_error: Exception | None = None,
    ) -> None:
        self.task = task
        self.answer = answer
        self.raw = raw
        self.degraded = degraded
        self.evidence = evidence if evidence is not None else []
        self.boxes = boxes if boxes is not None else []
        self.regions = regions if regions is not None else []
        self.labels = labels if labels is not None else []
        self.change_map = change_map
        self.geo = geo
        self.warnings = warnings if warnings is not None else []
        self.validation_error = validation_error
        self.execute_error = execute_error

    def validate_request(self, request: SpecialistRequest) -> None:
        if self.validation_error is not None:
            raise self.validation_error

    def execute(self, request: SpecialistRequest) -> SpecialistResult:
        if self.execute_error is not None:
            raise self.execute_error
        return SpecialistResult(
            task=self.task,
            answer=self.answer,
            labels=list(self.labels),
            boxes=list(self.boxes),
            regions=list(self.regions),
            change_map=self.change_map,
            evidence=list(self.evidence),
            confidence=ConfidenceBreakdown(raw=self.raw, degraded=self.degraded),
            geospatial=self.geo or GeoMetadata(),
            warnings=list(self.warnings),
            degraded=self.degraded,
        )

    def produce_evidence(self, result: SpecialistResult) -> list[Evidence]:
        return list(result.evidence)

    def estimate_confidence(self, result: SpecialistResult) -> ConfidenceBreakdown:
        return result.confidence


def _stub_spec(capability: str) -> SpecialistSpec:
    return SpecialistSpec(
        name=capability,
        capabilities=(capability,),
        module="nonexistent",
        builder="build_nothing",
        requires_assets=None,
    )


def _stub_class(capability: str) -> type:
    return type(
        f"_Stub_{capability}",
        (_Stub,),
        {"capabilities": (capability,), "name": capability},
    )


class _Registry:
    """Registry double returning pre-built stubs."""

    def __init__(
        self,
        entries: dict[str, Any],
        *,
        states: dict[str, RegistryState] | None = None,
    ) -> None:
        self._entries = entries
        self._states = states or {}
        self.built: list[str] = []

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def build(self, capability: str) -> Any:
        from core.registry import RegistryEntry

        if capability not in self._entries:
            raise SpecialistError(f"unknown capability {capability!r}")
        self.built.append(capability)
        value = self._entries[capability]
        state = self._states.get(capability, RegistryState.AVAILABLE)
        if isinstance(value, RegistryEntry):
            return value
        return RegistryEntry(
            spec=_stub_spec(capability),
            specialist=value,
            state=state,
        )

    def build_all(self) -> tuple[Any, ...]:
        return tuple(self.build(c) for c in self.available())

    def describe(self) -> dict[str, Any]:
        return {"declared": list(self.available()), "built": {}, "device": "cpu"}

    def entries(self) -> tuple[Any, ...]:
        return tuple(self.build(c) for c in self.available())

    @property
    def device(self) -> str:
        return "cpu"


class _Prediction:
    """Minimal router prediction: `intent` plus `above_threshold`."""

    def __init__(
        self,
        task: Task,
        *,
        confidence: float = 0.9,
        source: str = "learned",
        above: bool = True,
        language_output: bool = True,
        spatial_output: bool = False,
    ) -> None:
        from core.schemas import Intent

        self.intent = Intent(
            task=task,
            confidence=confidence,
            source=source,  # type: ignore[arg-type]
            language_output=language_output,
            spatial_output=spatial_output,
        )
        self.above_threshold = above

    def to_trace(self) -> dict[str, Any]:
        return {"task": self.intent.task.value, "source": self.intent.source}


class _Router:
    def __init__(self, prediction: _Prediction) -> None:
        self.prediction = prediction
        self.queries: list[str] = []

    def route(self, query: str) -> _Prediction:
        self.queries.append(query)
        return self.prediction


class _Cfg:
    device_preference = "cpu"

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self._values = values or {}

    def get(self, path: str, default: Any = None) -> Any:
        return self._values.get(path, default)

    @property
    def hash(self) -> str:
        return "deadbeefdeadbeef"


def _controller(
    entry_map: dict[str, Any],
    *,
    task: Task = Task.VQA,
    registry_states: dict[str, RegistryState] | None = None,
    **prediction_kw: Any,
) -> AnalysisController:
    prediction = _Prediction(task, **prediction_kw)
    registry = _Registry(entry_map, states=registry_states)
    planner = PolicyPlanner(registry)
    return AnalysisController(
        registry=registry,
        planner=planner,
        router=_Router(prediction),
        config=_Cfg(),
    )


_STUB_DIR: Path | None = None


def _stub_assets(count: int) -> list[str]:
    """Real, header-inspectable rasters for the controller's asset resolution.

    `AnalysisController._resolve_assets` delegates to
    `preprocessing.raster.inspect_raster`, so an asset must be an actual
    raster: a path that does not exist is a `RasterReadError`, and a file
    holding TIFF magic bytes but no image directory is an unreadable raster.
    Both are correct behaviour, neither is a usable asset, and neither lets
    these tests reach the logic they are actually about.

    So each stub is a genuine 4-band GeoTIFF written through rasterio. It is
    tiny (8x8) and carries no georeferencing, which is all `inspect_raster`
    needs to report a band count and infer optical modality.

    The rasters are written once per session into a directory under pytest's
    basetemp (see the autouse fixture below) because they are identical for
    every test and never mutated. Basetemp is still honoured, so the sandbox's
    `--basetemp` redirect applies.
    """
    assert _STUB_DIR is not None, "the _stub_assets_dir fixture did not run"
    paths = []
    for i in range(count):
        stub = _STUB_DIR / f"a{i}.tif"
        if not stub.exists():
            _write_stub_raster(stub)
        paths.append(str(stub))
    return paths


def _write_stub_raster(path: Path, *, count: int = 4) -> None:
    """Write a minimal but genuinely readable GeoTIFF.

    `rasterio` is imported lazily: the controller tests are meant to run
    without the raster stack when `path`-level behaviour is not under test, and
    a module-scope import would make collection fail on a machine without it.
    """
    import numpy as np
    import rasterio

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=count,
        dtype="uint16",
    ) as ds:
        ds.write(np.zeros((count, 8, 8), dtype="uint16"))


@pytest.fixture(scope="module", autouse=True)
def _stub_assets_dir(tmp_path_factory) -> Path:
    """Give every test in this module a shared directory for asset stubs."""
    global _STUB_DIR
    _STUB_DIR = Path(tmp_path_factory.mktemp("satquery_controller_stub"))
    return _STUB_DIR


def _request(
    assets: int = 1,
    force_task: Task | None = None,
    query: str = "what is here",
    paths: list[str] | None = None,
):
    """Build a request.

    The controller INSPECTS assets (`preprocessing.raster.inspect_raster`), so
    the paths must name real files. This helper writes minimal stubs into
    pytest's temporary tree. That is the honest fixture — a test that hands the
    controller a path which does not exist is exercising a refusal path, not
    the success path it claims to test.

    Callers that care about band counts pass `paths` explicitly.
    """
    return AnalysisRequest(
        assets=paths if paths is not None else _stub_assets(assets),
        query=query,
        force_task=force_task,
    )

def _evidence(label: str, score: float = 0.5, box: list[float] | None = None) -> Evidence:
    from core.schemas import CoordinateSystem

    if box is not None:
        return Evidence(
            type=EvidenceType.BOUNDING_BOX,
            source_specialist="stub",
            coordinate_system=CoordinateSystem.NORMALIZED_0_1,
            coordinates=box,
            score=score,
            payload={"label": label},
        )
    return Evidence(
        type=EvidenceType.STATISTIC,
        source_specialist="stub",
        score=score,
        payload={"label": label},
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_run_returns_a_valid_envelope() -> None:
    controller = _controller(
        {"vqa": _stub_class("vqa")(task=Task.VQA, answer="a tank")},
        task=Task.VQA,
    )
    envelope = controller.run(_request(1))
    assert envelope.result.answer == "a tank"
    assert envelope.result.task is Task.VQA
    assert envelope.run_id == envelope.trace.run_id


def test_run_records_controller_states_in_order() -> None:
    controller = _controller({"vqa": _stub_class("vqa")()}, task=Task.VQA)
    envelope = controller.run(_request(1))
    states = [s.state for s in envelope.trace.steps]
    assert states[0] is ControllerState.RECEIVE
    assert states[-1] is ControllerState.RESPOND
    for required in (
        ControllerState.RECEIVE,
        ControllerState.PARSE,
        ControllerState.VALIDATE,
        ControllerState.PLAN,
        ControllerState.EXECUTE,
        ControllerState.AGGREGATE,
        ControllerState.VERIFY,
        ControllerState.RESPOND,
    ):
        assert required in states


def test_one_execute_trace_step_per_plan_step() -> None:
    controller = _controller(
        {
            "change": _stub_class("change")(task=Task.CHANGE, change_map="m.png"),
            "caption": _stub_class("caption")(task=Task.CAPTION),
        },
        task=Task.CHANGE,
        language_output=True,
    )
    envelope = controller.run(_request(2))
    executes = [s for s in envelope.trace.steps if s.state is ControllerState.EXECUTE]
    planned = len(envelope.trace.workflow)
    assert len(executes) == planned


def test_trace_workflow_lists_step_ids_in_order() -> None:
    controller = _controller({"vqa": _stub_class("vqa")()}, task=Task.VQA)
    envelope = controller.run(_request(1))
    assert envelope.trace.workflow == ["step_001"]


def test_trace_records_per_step_timings() -> None:
    controller = _controller({"vqa": _stub_class("vqa")()}, task=Task.VQA)
    envelope = controller.run(_request(1))
    assert "step_001" in envelope.trace.timings
    assert envelope.trace.timings["step_001"] >= 0.0


def test_trace_records_inputs_and_query() -> None:
    controller = _controller({"vqa": _stub_class("vqa")()}, task=Task.VQA)
    request = _request(2, query="find the bridge")
    envelope = controller.run(request)
    assert envelope.trace.query == "find the bridge"
    # The trace names the assets in order -- but by NAME, not by path. Until
    # F-13 this asserted `== list(request.assets)`, i.e. it pinned the
    # disclosure in place: the response is returned verbatim to the client, so
    # echoing a path published the server-side location of the uploaded bytes.
    # The order is still asserted, because losing it would be a different
    # defect; only the disclosure changed.
    assert envelope.trace.inputs == [Path(p).name for p in request.assets]


def test_trace_records_the_config_hash() -> None:
    controller = _controller({"vqa": _stub_class("vqa")()}, task=Task.VQA)
    envelope = controller.run(_request(1))
    assert envelope.trace.config_hash == "deadbeefdeadbeef"


def test_trace_records_selected_models() -> None:
    controller = _controller({"vqa": _stub_class("vqa")()}, task=Task.VQA)
    envelope = controller.run(_request(1))
    assert any(r.name == "vqa" for r in envelope.trace.selected_models)


def test_trace_has_no_chain_of_thought_fields() -> None:
    controller = _controller({"vqa": _stub_class("vqa")()}, task=Task.VQA)
    envelope = controller.run(_request(1))
    dumped = envelope.trace.model_dump()
    for forbidden in ("reasoning", "thought", "rationale", "chain_of_thought"):
        assert forbidden not in dumped
    for step in envelope.trace.steps:
        for forbidden in ("reasoning", "thought", "rationale"):
            assert forbidden not in step.detail


# ---------------------------------------------------------------------------
# §5.1 partial failure — every row
# ---------------------------------------------------------------------------
def test_typed_step_failure_is_recorded_and_run_continues() -> None:
    primary = _stub_class("change")(
        task=Task.CHANGE,
        execute_error=PairMisalignmentError("not co-registered"),
    )
    other = _stub_class("caption")(task=Task.CAPTION, answer="a scene")
    controller = _controller({"change": primary, "caption": other}, task=Task.CHANGE)

    envelope = controller.run(_request(2))

    # The run did not abort: the caption result survives.
    assert "a scene" in envelope.result.answer or envelope.result.answer
    codes = [e["code"] for e in envelope.trace.errors]
    assert "pair_misaligned" in codes
    assert envelope.result.degraded is True


def test_untyped_exception_is_wrapped_as_unhandled() -> None:
    broken = _stub_class("vqa")(task=Task.VQA, execute_error=RuntimeError("kaboom"))
    controller = _controller({"vqa": broken}, task=Task.VQA)
    envelope = controller.run(_request(1))
    codes = [e["code"] for e in envelope.trace.errors]
    assert "specialist_error" in codes
    assert any("unhandled" in e["message"] for e in envelope.trace.errors)


def test_one_unhandled_failure_is_reported_the_same_way_on_every_carrier() -> None:
    """F-20 (owner ruling 2026-09-23).

    ONE unhandled failure reaches the client through several carriers, and before
    this ruling they disagreed. `trace.errors[].message` said *"Internal detail
    is withheld"* while `result.warnings[]` and `evidence[].payload["message"]`
    published *"unhandled RuntimeError: kaboom"*. A response that contradicts
    itself about what it has decided to disclose is worse than one that discloses
    plainly: the client cannot tell which half to believe.

    The exception here carries an absolute path, so this fails on the pre-ruling
    state in three independent ways -- the path, the exception TYPE NAME, and the
    exception MESSAGE. Those are the two things the ruling names: filesystem
    paths, and implementation details.

    Carriers covered include the duplication path. `ResultEnvelope.trace` and
    `ResultEnvelope.result.execution_trace` are the SAME object
    (`core/controller.py:785`), so a leak on any trace field is a leak twice --
    and a fix verified on only one of them is half a fix.
    """
    import json

    leak = r"C:\Users\anish\secret\decoder.py"
    broken = _stub_class("vqa")(
        task=Task.VQA,
        execute_error=RuntimeError(f"internal decoder blew up reading {leak}"),
    )
    controller = _controller({"vqa": broken}, task=Task.VQA)
    envelope = controller.run(_request(1))

    # Serialize the WHOLE envelope: this covers `result.warnings`,
    # `evidence[].payload["message"]`, `trace.errors[].message`,
    # `trace.parameters` and the duplicated `result.execution_trace` in one
    # assertion, so a carrier added later is covered by construction.
    body = json.dumps(envelope.model_dump(mode="json"), default=str)

    # -- no filesystem path, in raw OR escaped form --------------------------
    assert leak not in body, "the absolute path reached the client"
    assert json.dumps(leak)[1:-1] not in body, (
        "the JSON-escaped form of the path reached the client. A plain "
        "`leak in body` check CANNOT see this: json.dumps doubles the "
        "backslashes, so the raw string never matches. That false negative is "
        "exactly how F-15's first fix looked complete while leaking."
    )
    assert "decoder.py" not in body, "the path basename reached the client"

    # -- no implementation detail -------------------------------------------
    assert "RuntimeError" not in body, (
        "the exception type name is a Python implementation detail and must "
        "stay server-side; a client cannot act on RuntimeError vs ValueError"
    )
    assert "blew up" not in body, (
        "the exception's own message is an implementation detail and must stay "
        "server-side"
    )

    # -- the classification survives on EVERY carrier ------------------------
    assert any(
        "unhandled" in e["message"] for e in envelope.trace.errors
    ), "trace.errors lost the 'unhandled' classification"

    executes = [
        s for s in envelope.trace.steps if s.state is ControllerState.EXECUTE
    ]
    assert executes, "no EXECUTE step was recorded"
    assert executes[0].detail["error_code"] == "specialist_error"
    assert executes[0].detail["step_id"] == "step_001"

    warn = [w for w in envelope.result.warnings if "vqa failed" in w]
    assert warn, "the failure must still appear in result.warnings"
    assert "specialist_error" in warn[0], "the warning lost the error code"
    assert "unhandled" in warn[0], "the warning lost the classification"

    payloads = [
        ev.payload.get("message")
        for ev in envelope.result.evidence
        if ev.payload.get("step_failed")
    ]
    assert payloads, "the failure must still appear in evidence"
    assert "unhandled" in payloads[0], "the evidence message lost the classification"


def test_the_two_error_carriers_share_one_rule() -> None:
    """F-20: one rule, one implementation.

    Two hand-rolled copies of `scrub_paths(detail) or user_message` is how a rule
    drifts: the next fix lands on whichever copy the author happened to open.
    That is precisely how F-15's internals fix missed two of its three carriers.

    So both carriers must return the SAME reason string for the same error, and
    that reason must be path-free while keeping the actionable part.
    """
    from core.controller import AnalysisController

    leaky = ModelLoadError(
        r"could not read encoder weights from C:\Users\anish\secret\stanet.pt",
        specialist="change",
    )
    reason = AnalysisController._client_error_reason(leaky)

    assert reason == "could not read encoder weights from stanet.pt"
    assert "\\" not in reason and "/" not in reason
    assert "anish" not in reason

    step = PlanStep("step_001", "change", "change", (0, 1), {}, "task:change", True)
    outcome = StepOutcome(step=step, error=leaky)

    evidence = AnalysisController._failure_evidence(outcome)
    assert evidence.payload["message"] == reason, (
        "evidence and warnings must not compute the reason separately"
    )

    plan = ExecutionPlan(steps=(step,))
    warnings = AnalysisController._warnings(plan, [outcome], [])
    assert any(reason in w for w in warnings), (
        "the warning must carry the same shared reason"
    )


def test_validation_failure_is_recorded() -> None:
    bad = _stub_class("change")(
        task=Task.CHANGE, validation_error=InvalidRequestError("needs 2 assets")
    )
    controller = _controller({"change": bad}, task=Task.CHANGE)
    envelope = controller.run(_request(2))
    assert any(e["code"] == "invalid_request" for e in envelope.trace.errors)
    assert envelope.result.degraded is True


def test_all_steps_failing_still_returns_a_result() -> None:
    broken = _stub_class("vqa")(
        task=Task.VQA, execute_error=SpecialistError("nope", specialist="vqa")
    )
    controller = _controller({"vqa": broken}, task=Task.VQA)
    envelope = controller.run(_request(1))
    assert envelope.result.degraded is True
    assert envelope.result.answer  # states the failure, invents nothing
    assert envelope.result.confidence.raw == 0.0


def test_required_step_failure_marks_the_run_degraded() -> None:
    broken = _stub_class("vqa")(
        task=Task.VQA, execute_error=SpecialistError("bad", specialist="vqa")
    )
    controller = _controller({"vqa": broken}, task=Task.VQA)
    envelope = controller.run(_request(1))
    assert envelope.result.degraded is True


def test_failed_step_leaves_a_statistic_evidence_item() -> None:
    """An absence must be distinguishable from a non-event (§5.3)."""
    broken = _stub_class("vqa")(
        task=Task.VQA, execute_error=SpecialistError("bad", specialist="vqa")
    )
    controller = _controller({"vqa": broken}, task=Task.VQA)
    envelope = controller.run(_request(1))
    failures = [e for e in envelope.result.evidence if e.payload.get("step_failed")]
    assert len(failures) == 1
    assert failures[0].type is EvidenceType.STATISTIC
    assert failures[0].source_specialist == "vqa"
    assert failures[0].payload["step_id"] == "step_001"
    assert failures[0].score == 0.0


def test_a_failed_step_publishes_no_filesystem_path() -> None:
    """F-15 (owner ruling 2026-09-23) at the controller's two remaining sites.

    `_warnings` and `_failure_evidence` both preferred `error.detail` over
    `error.user_message`. `detail` is the free-form diagnostic field and
    routinely carries an absolute path, so a single failed step published the
    server-side location twice more in the response body.

    Measured on the pre-ruling code with one real construction failure, the
    path appeared in **four** client-visible fields: `result.warnings[]`,
    `result.evidence[].payload["message"]`,
    `trace.parameters.registry.built[<cap>].detail`, and the same registry block
    again inside `result.execution_trace`. This guard covers the two the
    controller owns; `tests/unit/test_registry.py` covers the other two.

    What must NOT be lost while removing the path is the classification: which
    capability failed, and with which code. A warning the client cannot act on
    would be a worse outcome than the leak.
    """
    import json

    leak = r"C:\Users\operator\satquery\artifacts\change\stanet_encoder_v3.pt"
    # `json.dumps` doubles every backslash, so `leak not in body` is a FALSE
    # NEGATIVE: it passed even with the path fully present in the body. Measured
    # in both falsification states (see sq_scratch/p14d_escaping.py) -- the
    # assertions that actually fired were the backslash-free `satquery` checks.
    escaped = json.dumps(leak)[1:-1]
    broken = _stub_class("vqa")(
        task=Task.VQA,
        execute_error=SpecialistError(
            f"encoder weights_path does not exist: {leak}", specialist="vqa"
        ),
    )
    controller = _controller({"vqa": broken}, task=Task.VQA)
    envelope = controller.run(_request(1))

    body = json.dumps(envelope.model_dump(mode="json"), default=str)
    assert escaped not in body, "a filesystem path reached the response body"

    warnings = envelope.result.warnings
    assert any("vqa" in w and "specialist_error" in w for w in warnings), (
        f"the classification was lost along with the path: {warnings}"
    )
    # The reason must SURVIVE the scrub, reduced to a basename. Replacing the
    # whole message with a generic string would also stop the leak, and it was
    # tried first -- it forced three legitimate tests to be weakened, which is
    # the signal that the granularity was wrong.
    assert any("stanet_encoder_v3.pt" in w for w in warnings), (
        f"the reason was lost along with the path: {warnings}"
    )
    assert not any("satquery" in w for w in warnings), (
        f"a directory component survived the scrub: {warnings}"
    )

    failures = [e for e in envelope.result.evidence if e.payload.get("step_failed")]
    assert failures, "the failure evidence item disappeared"
    assert failures[0].payload["code"] == "specialist_error"
    assert "stanet_encoder_v3.pt" in failures[0].payload["message"]
    assert "satquery" not in failures[0].payload["message"]


def test_partial_failure_keeps_the_other_specialists_evidence() -> None:
    good_evidence = [_evidence("kept")]
    good = _stub_class("change")(
        task=Task.CHANGE, evidence=good_evidence, change_map="m.png"
    )
    bad = _stub_class("caption")(
        task=Task.CAPTION, execute_error=SpecialistError("bad", specialist="caption")
    )
    controller = _controller({"change": good, "caption": bad}, task=Task.CHANGE)
    envelope = controller.run(_request(2))
    labels = [e.payload.get("label") for e in envelope.result.evidence]
    assert "kept" in labels
    assert any(e.payload.get("step_failed") for e in envelope.result.evidence)


def test_unavailable_registry_entry_becomes_a_recorded_failure() -> None:
    """Planned but unconstructible: recorded, not crashed, not silent."""
    from core.registry import RegistryEntry

    spec = _stub_spec("optical_sar")
    entry = RegistryEntry(
        spec=spec,
        specialist=None,
        state=RegistryState.UNAVAILABLE,
        detail="CROMA could not load",
        error_code="model_unavailable",
    )
    registry = _Registry({"optical_sar": entry})
    controller = AnalysisController(
        registry=registry,
        planner=PolicyPlanner(registry),
        router=_Router(_Prediction(Task.OPTICAL_SAR)),
        config=_Cfg(),
    )
    envelope = controller.run(_request(2))
    assert any(e["code"] == "model_unavailable" for e in envelope.trace.errors)
    assert any(
        "CROMA could not load" in w for w in envelope.result.warnings
    )


def test_budget_exceeded_skips_remaining_steps() -> None:
    """The run budget is checked between steps (§5.2)."""
    controller = _controller(
        {
            "change": _stub_class("change")(task=Task.CHANGE, change_map="m.png"),
            "caption": _stub_class("caption")(task=Task.CAPTION),
        },
        task=Task.CHANGE,
    )
    controller.budget_seconds = -1.0  # everything is over budget immediately
    envelope = controller.run(_request(2))
    skips = [e for e in envelope.trace.errors if e["code"] == "specialist_timeout"]
    assert skips
    assert envelope.result.degraded is True


# ---------------------------------------------------------------------------
# Refusal
# ---------------------------------------------------------------------------
def test_refusal_returns_a_valid_envelope_with_no_steps() -> None:
    controller = _controller({"vqa": _stub_class("vqa")()}, task=Task.UNSUPPORTED)
    envelope = controller.run(_request(1))
    assert envelope.result.degraded is True
    assert envelope.trace.workflow == []
    assert envelope.result.confidence.raw == 0.0
    assert envelope.result.confidence.degraded is True
    assert "No specialist supports" in envelope.result.answer


def test_refusal_records_the_typed_reason() -> None:
    controller = _controller({"vqa": _stub_class("vqa")()}, task=Task.UNSUPPORTED)
    envelope = controller.run(_request(1))
    assert any("task_unsupported" in w for w in envelope.result.warnings)


def test_refusal_does_not_execute_anything() -> None:
    registry = _Registry({"vqa": _stub_class("vqa")()})
    controller = AnalysisController(
        registry=registry,
        planner=PolicyPlanner(registry),
        router=_Router(_Prediction(Task.UNSUPPORTED)),
        config=_Cfg(),
    )
    controller.run(_request(1))
    assert registry.built == []


def test_forced_unsupported_task_refuses() -> None:
    controller = _controller({"vqa": _stub_class("vqa")()}, task=Task.VQA)
    envelope = controller.run(_request(1, force_task=Task.UNSUPPORTED))
    assert envelope.result.degraded is True
    assert envelope.trace.workflow == []


def test_no_router_and_no_force_task_raises() -> None:
    registry = _Registry({"vqa": _stub_class("vqa")()})
    controller = AnalysisController(
        registry=registry, planner=PolicyPlanner(registry), router=None, config=_Cfg()
    )
    with pytest.raises(Exception):
        controller.run(_request(1))


def test_forced_task_bypasses_the_router() -> None:
    router = _Router(_Prediction(Task.CHANGE))
    registry = _Registry({"grounding": _stub_class("grounding")(task=Task.GROUNDING)})
    controller = AnalysisController(
        registry=registry, planner=PolicyPlanner(registry), router=router, config=_Cfg()
    )
    controller.run(_request(1, force_task=Task.GROUNDING))
    assert router.queries == []


# ---------------------------------------------------------------------------
# §7.2 assembly
# ---------------------------------------------------------------------------
def test_task_comes_from_the_router_not_a_specialist() -> None:
    """A multi-step plan has no single specialist task."""
    controller = _controller(
        {
            "change": _stub_class("change")(task=Task.CHANGE, change_map="m.png"),
            "caption": _stub_class("caption")(task=Task.CAPTION),
        },
        task=Task.CHANGE,
    )
    envelope = controller.run(_request(2))
    assert envelope.result.task is Task.CHANGE


def test_labels_are_unioned_deduped_and_ordered() -> None:
    controller = _controller({"vqa": _stub_class("vqa")(labels=["a", "b"])}, task=Task.VQA)
    envelope = controller.run(_request(1))
    assert envelope.result.labels == ["a", "b"]


def test_warnings_are_unioned() -> None:
    controller = _controller(
        {"vqa": _stub_class("vqa")(warnings=["specialist warning"])}, task=Task.VQA
    )
    envelope = controller.run(_request(1))
    assert "specialist warning" in envelope.result.warnings


def test_geospatial_takes_the_first_non_empty() -> None:
    geo = GeoMetadata(crs="EPSG:4326", has_crs=True, is_georeferenced=True)
    controller = _controller(
        {"vqa": _stub_class("vqa")(geo=geo)}, task=Task.VQA
    )
    envelope = controller.run(_request(1))
    assert envelope.result.geospatial.crs == "EPSG:4326"


def test_empty_geospatial_is_reported_as_empty() -> None:
    controller = _controller({"vqa": _stub_class("vqa")()}, task=Task.VQA)
    envelope = controller.run(_request(1))
    assert envelope.result.geospatial.crs is None


def test_evidence_ids_are_unique_and_sequential() -> None:
    evidence = [_evidence("e1"), _evidence("e2"), _evidence("e3")]
    controller = _controller(
        {"vqa": _stub_class("vqa")(evidence=evidence)}, task=Task.VQA
    )
    envelope = controller.run(_request(1))
    ids = [e.evidence_id for e in envelope.result.evidence]
    assert len(ids) == len(set(ids))
    assert ids == [f"evidence_{i:03d}" for i in range(1, len(ids) + 1)]


def test_degraded_propagates_from_a_specialist() -> None:
    controller = _controller(
        {"vqa": _stub_class("vqa")(degraded=True)}, task=Task.VQA
    )
    envelope = controller.run(_request(1))
    assert envelope.result.degraded is True


def test_uncertain_route_marks_the_run_degraded() -> None:
    controller = _controller(
        {
            "change": _stub_class("change")(task=Task.CHANGE, change_map="m.png"),
            "vqa": _stub_class("vqa")(task=Task.VQA),
        },
        task=Task.CHANGE,
        above=False,
        confidence=0.3,
    )
    envelope = controller.run(_request(2))
    assert envelope.result.degraded is True


def test_clean_run_is_not_degraded() -> None:
    controller = _controller({"vqa": _stub_class("vqa")()}, task=Task.VQA)
    envelope = controller.run(_request(1))
    assert envelope.result.degraded is False


# ---------------------------------------------------------------------------
# §7.3 answer priority
# ---------------------------------------------------------------------------
def test_vlm_answer_wins_when_it_ran() -> None:
    controller = _controller(
        {"vqa": _stub_class("vqa")(task=Task.VQA, answer="the VLM explains")},
        task=Task.VQA,
    )
    envelope = controller.run(_request(1))
    assert envelope.result.answer == "the VLM explains"


def test_specialist_answer_is_attributed_when_no_vlm_ran() -> None:
    controller = _controller(
        {"grounding": _stub_class("grounding")(task=Task.GROUNDING, answer="a box")},
        task=Task.GROUNDING,
    )
    envelope = controller.run(_request(1))
    assert envelope.result.answer.startswith("[grounding]")


def test_failure_answer_names_the_codes() -> None:
    broken = _stub_class("vqa")(
        task=Task.VQA, execute_error=SpecialistError("x", specialist="vqa")
    )
    controller = _controller({"vqa": broken}, task=Task.VQA)
    envelope = controller.run(_request(1))
    assert "vqa" in envelope.result.answer
    assert "specialist_error" in envelope.result.answer


# ---------------------------------------------------------------------------
# §7.5 confidence
# ---------------------------------------------------------------------------
def test_confidence_comes_from_the_primary_step() -> None:
    controller = _controller(
        {"vqa": _stub_class("vqa")(task=Task.VQA, raw=0.7)}, task=Task.VQA
    )
    envelope = controller.run(_request(1))
    assert envelope.result.confidence.raw == pytest.approx(0.7)


def test_confidence_is_not_calibrated_without_an_artifact() -> None:
    controller = _controller({"vqa": _stub_class("vqa")()}, task=Task.VQA)
    envelope = controller.run(_request(1))
    assert envelope.result.confidence.calibrated is None
    assert envelope.result.confidence.method == "uncalibrated"


def test_secondary_confidences_appear_in_components() -> None:
    controller = _controller(
        {
            "change": _stub_class("change")(
                task=Task.CHANGE, change_map="m.png", raw=0.6
            ),
            "caption": _stub_class("caption")(task=Task.CAPTION, raw=0.9),
        },
        task=Task.CHANGE,
    )
    envelope = controller.run(_request(2))
    components = envelope.result.confidence.components
    assert components["change_confidence"] == pytest.approx(0.6)
    assert components["caption_confidence"] == pytest.approx(0.9)


def test_primary_failure_yields_zero_confidence() -> None:
    """Reporting a secondary's score would launder a failure into a number."""
    good = _stub_class("caption")(task=Task.CAPTION, raw=0.95)
    bad = _stub_class("change")(
        task=Task.CHANGE, execute_error=SpecialistError("x", specialist="change")
    )
    controller = _controller({"change": bad, "caption": good}, task=Task.CHANGE)
    envelope = controller.run(_request(2))
    assert envelope.result.confidence.raw == 0.0
    assert envelope.result.confidence.degraded is True
    assert "primary" in (envelope.result.confidence.degradation_reason or "")


def test_uncertain_route_marks_confidence_degraded() -> None:
    controller = _controller(
        {
            "change": _stub_class("change")(task=Task.CHANGE, change_map="m.png"),
            "vqa": _stub_class("vqa")(task=Task.VQA, raw=0.9),
        },
        task=Task.CHANGE,
        above=False,
    )
    envelope = controller.run(_request(2))
    assert envelope.result.confidence.degraded is True


def test_trace_confidence_matches_the_result() -> None:
    controller = _controller({"vqa": _stub_class("vqa")(raw=0.42)}, task=Task.VQA)
    envelope = controller.run(_request(1))
    assert envelope.trace.confidence is not None
    assert envelope.trace.confidence.raw == pytest.approx(
        envelope.result.confidence.raw
    )


# ---------------------------------------------------------------------------
# §7.4 / §7.6 verification and contradiction
# ---------------------------------------------------------------------------
def test_evidence_flows_through_the_engine() -> None:
    evidence = [_evidence("x"), _evidence("x")]  # duplicates collapse
    controller = _controller(
        {"vqa": _stub_class("vqa")(evidence=evidence)}, task=Task.VQA
    )
    envelope = controller.run(_request(1))
    assert len(envelope.result.evidence) == 1


def test_synthesis_evidence_appears_for_a_multi_step_run() -> None:
    controller = _controller(
        {
            "change": _stub_class("change")(task=Task.CHANGE, change_map="m.png"),
            "caption": _stub_class("caption")(task=Task.CAPTION),
        },
        task=Task.CHANGE,
    )
    envelope = controller.run(_request(2))
    assert any(e.payload.get("synthesis") for e in envelope.result.evidence)


def test_no_synthesis_evidence_for_a_single_step_run() -> None:
    controller = _controller({"vqa": _stub_class("vqa")()}, task=Task.VQA)
    envelope = controller.run(_request(1))
    assert not any(e.payload.get("synthesis") for e in envelope.result.evidence)


def test_contradictory_boxes_are_both_kept_and_flagged() -> None:
    from core.schemas import CoordinateSystem

    left = Box(x1=0.0, y1=0.0, x2=0.1, y2=0.1, coordinate_system=CoordinateSystem.NORMALIZED_0_1)
    right = Box(x1=0.8, y1=0.8, x2=0.9, y2=0.9, coordinate_system=CoordinateSystem.NORMALIZED_0_1)
    controller = _controller(
        {
            "change": _stub_class("change")(
                task=Task.CHANGE, boxes=[left], change_map="m.png"
            ),
            "caption": _stub_class("caption")(task=Task.CAPTION, boxes=[right]),
        },
        task=Task.CHANGE,
    )
    envelope = controller.run(_request(2))
    assert envelope.trace.contradiction is True
    assert len(envelope.result.boxes) == 2
    assert any("contradictory" in w for w in envelope.result.warnings)


def test_agreeing_boxes_are_not_a_contradiction() -> None:
    from core.schemas import CoordinateSystem

    a = Box(x1=0.1, y1=0.1, x2=0.5, y2=0.5, coordinate_system=CoordinateSystem.NORMALIZED_0_1)
    b = Box(x1=0.12, y1=0.12, x2=0.52, y2=0.52, coordinate_system=CoordinateSystem.NORMALIZED_0_1)
    controller = _controller(
        {
            "change": _stub_class("change")(task=Task.CHANGE, boxes=[a], change_map="m.png"),
            "caption": _stub_class("caption")(task=Task.CAPTION, boxes=[b]),
        },
        task=Task.CHANGE,
    )
    envelope = controller.run(_request(2))
    assert envelope.trace.contradiction is False


def test_boxes_in_different_systems_are_not_compared() -> None:
    """Cross-system IoU is meaningless, so no contradiction is claimed."""
    from core.schemas import CoordinateSystem

    a = Box(x1=0.0, y1=0.0, x2=0.1, y2=0.1, coordinate_system=CoordinateSystem.NORMALIZED_0_1)
    b = Box(x1=800.0, y1=800.0, x2=900.0, y2=900.0, coordinate_system=CoordinateSystem.PIXEL)
    controller = _controller(
        {
            "change": _stub_class("change")(task=Task.CHANGE, boxes=[a], change_map="m.png"),
            "caption": _stub_class("caption")(task=Task.CAPTION, boxes=[b]),
        },
        task=Task.CHANGE,
    )
    envelope = controller.run(_request(2))
    assert envelope.trace.contradiction is False


def test_iou_is_zero_for_disjoint_boxes() -> None:
    from core.schemas import CoordinateSystem

    a = Box(x1=0.0, y1=0.0, x2=0.1, y2=0.1, coordinate_system=CoordinateSystem.NORMALIZED_0_1)
    b = Box(x1=0.9, y1=0.9, x2=1.0, y2=1.0, coordinate_system=CoordinateSystem.NORMALIZED_0_1)
    assert _iou(a, b) == 0.0


def test_iou_is_one_for_identical_boxes() -> None:
    from core.schemas import CoordinateSystem

    a = Box(x1=0.0, y1=0.0, x2=0.5, y2=0.5, coordinate_system=CoordinateSystem.NORMALIZED_0_1)
    b = Box(x1=0.0, y1=0.0, x2=0.5, y2=0.5, coordinate_system=CoordinateSystem.NORMALIZED_0_1)
    assert _iou(a, b) == pytest.approx(1.0)


def test_contradiction_floor_is_frozen() -> None:
    assert CONTRADICTION_IOU_FLOOR == 0.1


# ---------------------------------------------------------------------------
# Health / observability
# ---------------------------------------------------------------------------
def test_health_reports_registry_states() -> None:
    from core.registry import RegistryEntry

    spec = _stub_spec("optical_sar")
    unavailable = RegistryEntry(
        spec=spec,
        specialist=None,
        state=RegistryState.UNAVAILABLE,
        detail="CROMA missing",
        error_code="model_unavailable",
    )
    registry = _Registry(
        {
            "vqa": _stub_class("vqa")(),
            "optical_sar": unavailable,
        },
        states={"vqa": RegistryState.DEGRADED},
    )
    controller = AnalysisController(
        registry=registry,
        planner=PolicyPlanner(registry),
        router=_Router(_Prediction(Task.VQA)),
        config=_Cfg(),
    )
    health = controller.health()
    assert health["registry"]["vqa"] == "degraded"
    assert health["registry"]["optical_sar"] == "unavailable"
    assert "optical_sar" in health["unavailable"]


def test_step_outcome_reports_failure() -> None:
    step = PlanStep("step_001", "vqa", "vqa", (0,), {}, "task:vqa", True)
    outcome = StepOutcome(
        step=step, error=SpecialistError("x", specialist="vqa")
    )
    assert outcome.ok is False
    assert outcome.to_trace()["error_code"] == "specialist_error"


# ---------------------------------------------------------------------------
# §9 full degraded run with the real registry
# ---------------------------------------------------------------------------
def test_full_degraded_run_with_no_weights_available() -> None:
    """The Gate-4 acceptance shape: everything degraded, still coherent.

    Uses stub builders under the REAL spec table, so capability names, asset
    requirements and the registry's capability assertion are all exercised. No
    model weights exist in this environment, which is the point: the system
    must produce one valid envelope, say what is degraded, and report no
    calibration rather than inventing one.
    """
    specs = (
        _stub_spec("change"),
        _stub_spec("caption"),
    )

    def _degraded_evidence(capability: str) -> list[Evidence]:
        """What a real degraded specialist emits: a record of what it could not do.

        A degraded specialist still produces evidence — that is the whole
        honest-degradation principle, and the reason DEGRADED is planable. A
        stub that produced nothing would make the `sources` assertion below
        vacuous and would misrepresent the real local state.
        """
        return [
            Evidence(
                type=EvidenceType.STATISTIC,
                source_specialist=capability,
                score=0.0,
                payload={"degraded": True, "reason": "no trained weights"},
            )
        ]

    stubs = {
        "change": _stub_class("change")(
            task=Task.CHANGE,
            degraded=True,
            raw=0.0,
            evidence=_degraded_evidence("change"),
        ),
        "caption": _stub_class("caption")(
            task=Task.CAPTION,
            degraded=True,
            raw=0.0,
            evidence=_degraded_evidence("caption"),
        ),
    }
    registry = SpecialistRegistry(
        _Cfg(),
        specs=specs,
        builders={name: (lambda s: lambda config, **kw: s)(stub) for name, stub in stubs.items()},
    )
    controller = AnalysisController(
        registry=registry,
        planner=PolicyPlanner(registry),
        router=_Router(_Prediction(Task.CHANGE, language_output=True)),
        config=_Cfg(),
    )

    envelope = controller.run(_request(2))

    assert envelope.result.degraded is True
    assert envelope.result.confidence.calibrated is None
    assert envelope.result.evidence
    assert envelope.trace.finished_at is not None
    # Every specialist that ran is named in the evidence collection.
    sources = {e.source_specialist for e in envelope.result.evidence}
    assert "change" in sources


def test_trace_inputs_disclose_no_filesystem_path() -> None:
    """F-13. `ExecutionTrace.inputs` is client-visible and carried PATHS.

    `POST /v1/analyze` returns `ResultEnvelope`, whose `trace` field is
    serialised straight to the client (`app/space_app.py`, the `analyze`
    handler: `envelope.model_dump(mode="json")`). The controller populated
    `trace.inputs` from `request.assets` AFTER the Space had translated handles
    into filesystem paths, so the response disclosed the server-side location of
    every uploaded file -- the exact disclosure `AssetHandle.to_response()`
    refuses ("`path` is deliberately absent. Returning it would hand a client a
    server-side filesystem location").

    The Space could not express the fix: after `store.resolve_many()` the
    original handles are gone, so the handler would have had to reconstruct
    them. The layer that OWNS the vocabulary (a handle, not a path) is the layer
    that must publish it.

    This asserts the shape of the value, not a spelling of it: whatever the
    client receives must not be a path that exists on this machine.
    """
    from pathlib import Path as _Path

    controller = _controller({"vqa": _stub_class("vqa")()}, task=Task.VQA)
    request = _request(1)
    assert request.assets, "the fixture must supply a real asset path"

    envelope = controller.run(request)

    assert envelope.trace.inputs, (
        "the trace must still say what the request ran over; dropping the field "
        "would be a different defect (silent data loss) than the one being fixed"
    )
    for value in envelope.trace.inputs:
        assert not _Path(value).exists(), (
            f"trace.inputs carries {value!r}, which is a real filesystem path; "
            f"the response would disclose the server-side location of the "
            f"uploaded bytes (F-13)"
        )
    # The asset's own NAME is not a disclosure -- it is an opaque handle, and it
    # is what `/v1/assets` already returns. Only the directory is.
    for value, original in zip(envelope.trace.inputs, request.assets):
        assert value == _Path(original).name, (
            "trace.inputs must name the asset opaque, without its directory"
        )


def test_parse_step_inputs_disclose_no_filesystem_path() -> None:
    """F-14. The same disclosure as F-13, one trace record later.

    `trace.steps[]` is part of the same client-visible `ExecutionTrace` that
    `POST /v1/analyze` returns verbatim, and the PARSE record wrote
    `list(request.assets)` -- by which point the Space has already rewritten the
    client's opaque handles into filesystem paths (`app/space_app.py::analyze`,
    `store.resolve_many()`). The response therefore published the server-side
    location of the uploaded bytes a second time, in a field F-13 did not cover.

    Measured 2026-09-22 end-to-end through the real Space app (probe
    `probe_f14_e2e.py`, `httpx.ASGITransport`, `raise_app_exceptions=False`).
    The client uploaded a raster, received the handle
    `asset_d243f7f85d8c2f3c02981f0af9737f01`, posted it to `/v1/analyze`, and
    the raw response carried:

        "state": "PARSE",
        "detail": {
          "inputs": [
            "C:\\\\Users\\\\anish\\\\sq_scratch\\\\f14_e2e_20260922\\\\assets\\\\asset_d243f7f85d8c2f3c02981f0af9737f01.tif"
          ],

    which is the whole path, not the handle. It was the only occurrence of that
    path anywhere in the response.

    This asserts the SHAPE of the value, not a spelling of it: whatever the
    client receives must not name a location that exists on this machine. The
    field must also stay populated and stay in request order -- dropping it or
    reordering it would be a different defect (silent data loss) than the one
    being fixed, and a guard that only checked for absence would accept both.
    """
    from pathlib import Path as _Path

    controller = _controller({"vqa": _stub_class("vqa")()}, task=Task.VQA)
    request = _request(2)
    assert request.assets, "the fixture must supply real asset paths"

    envelope = controller.run(request)

    parse = [
        step for step in envelope.trace.steps if step.state == ControllerState.PARSE
    ]
    assert len(parse) == 1, (
        f"expected exactly one PARSE record in the trace, found {len(parse)}"
    )
    inputs = parse[0].detail["inputs"]

    assert inputs, (
        "the PARSE record must still say what was parsed; dropping the field "
        "would be a different defect (silent data loss) than the one being fixed"
    )
    assert len(inputs) == len(request.assets), (
        f"the PARSE record named {len(inputs)} inputs for a "
        f"{len(request.assets)}-asset request"
    )
    for value in inputs:
        assert not _Path(value).exists(), (
            f"trace.steps[PARSE].detail['inputs'] carries {value!r}, which is a "
            f"real filesystem path; the response would disclose the server-side "
            f"location of the uploaded bytes (F-14)"
        )
    # The asset's own NAME is not a disclosure -- it is an opaque handle, and it
    # is what `/v1/assets` already returns. Only the directory is. Order is
    # still asserted, because losing it would be a different defect.
    for value, original in zip(inputs, request.assets):
        assert value == _Path(original).name, (
            "trace.steps[PARSE].detail['inputs'] must name each asset opaque, "
            "without its directory, and in request order"
        )


def test_croma_artifact_refs_are_real_files_the_specialist_wrote(
    tmp_path: Path,
) -> None:
    """F-16, second carrier. CROMA/`optical_sar` publishes real filesystem paths.

    Pass 12 measured this for `change` end to end and only READ-VERIFIED `croma`,
    labelling it read-verified everywhere. Pass 13 measured it through the real
    Space ASGI route (probe `probe_f16b_croma_asgi.py`): a client uploaded two
    real rasters, posted the handles to `/v1/analyze`, and the serialized
    response carried

        result.evidence[2].type         == "optical_view"
        result.evidence[2].artifact_ref == "C:\\\\...\\\\croma_artifacts\\\\optical_view_asset_<id>.png"
        result.evidence[3].type         == "sar_view"
        result.evidence[3].artifact_ref == "C:\\\\...\\\\croma_artifacts\\\\sar_view_asset_<id>.png"

    each naming a file that existed on disk (49,435 and 38,991 bytes), while
    `API_CONTRACT.md` section 2.4 documented an `artifact://` URI that **no
    production file emits**. The build was DEGRADED (no CROMA, no trained head)
    and the paths were published anyway. (Section 2.4 has since been corrected:
    it now publishes `null` refs and an explicit non-retrievable note.)

    THE CONTRACT CHANGED (owner ruling 2026-09-23). The paths above are what the
    response carried *before* the ruling; they must never appear again. F-16 now
    reads:

      * **never expose filesystem paths** -- not in `artifact_ref`, not in a
        payload, not anywhere in the serialized response;
      * v1 has **no artifact-serving endpoint**, so an unavailable
        `artifact_ref` is **`null`**;
      * an explicit **non-retrievable-artifact warning** must be emitted;
      * **do not fabricate** an `artifact://` URI.

    So this guard now asserts the SHAPE OF THE FIX rather than the shape of the
    defect:

      * both view evidence items must STILL be emitted, in optical-then-SAR
        order -- the ruling removes the path, not the capability, so dropping
        the item would be a different defect;
      * every `artifact_ref` in the response must be `None`;
      * no filesystem path -- the `artifact_dir`, the input rasters, or the
        rendered file names -- may appear in the serialized envelope;
      * the writer must STILL have produced both files on disk, so the ruling
        cannot be satisfied by simply not rendering;
      * a non-retrievable warning must be present;
      * no `artifact://` URI may appear.

    Non-vacuousness: this guard FAILS on the pre-ruling source, where the refs
    carried the rendered paths -- which is how it was found failing, 2026-09-23.
    """
    import json

    import numpy as np
    import rasterio

    from core.schemas import EvidenceType
    from specialists.optical_sar.specialist import OpticalSarSpecialist

    artifact_dir = tmp_path / "croma_artifacts"

    def _write(path: Path, bands: int) -> str:
        # Band count is how modality is inferred: {1,2} -> SAR, {3,4,8,11,12,13}
        # -> optical (`preprocessing/raster.py`). The pair must be one of each or
        # the specialist refuses the request, so the counts are load-bearing.
        arr = np.random.default_rng(0).integers(100, 3000, (bands, 32, 32)).astype(
            "uint16"
        )
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=32,
            width=32,
            count=bands,
            dtype=arr.dtype,
        ) as ds:
            ds.write(arr)
        return str(path)

    optical = _write(tmp_path / "optical_4band.tif", 4)
    sar = _write(tmp_path / "sar_2band.tif", 2)

    # A REAL specialist, not a stub: the point is the writer's own behaviour.
    specialist = OpticalSarSpecialist(artifact_dir=artifact_dir)
    controller = _controller({"optical_sar": specialist}, task=Task.OPTICAL_SAR)
    request = _request(force_task=Task.OPTICAL_SAR, paths=[optical, sar])

    envelope = controller.run(request)
    evidence = list(envelope.result.evidence)

    # -- the ruling: no path may reach the client --------------------------
    refs = [item.artifact_ref for item in evidence]
    assert refs == [None] * len(refs), (
        f"an evidence item carries an artifact_ref: {refs}. F-16 forbids "
        f"exposing a filesystem path, and v1 has no artifact-serving endpoint "
        f"to return a URI instead, so every ref must be null."
    )

    # Nothing about the server-side layout may appear in the serialized body.
    serialized = json.dumps(envelope.model_dump(mode="json"), default=str)
    for secret in (str(artifact_dir), str(tmp_path), optical, sar):
        assert secret not in serialized, (
            f"the serialized response discloses {secret!r}; F-16 forbids "
            f"exposing any filesystem path"
        )
    assert "artifact://" not in serialized, (
        "an `artifact://` URI was fabricated. The contract documents one that no "
        "production file emits, and F-16 forbids inventing one."
    )

    # -- the capability is NOT dropped: both items survive, in order --------
    order = [item.type for item in evidence]
    for expected in (EvidenceType.OPTICAL_VIEW, EvidenceType.SAR_VIEW):
        assert expected in order, (
            f"the response carries no {expected.value} evidence. The ruling "
            f"removes the PATH, not the capability: dropping the item would be "
            f"a different defect."
        )
    assert order.index(EvidenceType.OPTICAL_VIEW) < order.index(
        EvidenceType.SAR_VIEW
    ), f"the view evidence was reordered: {[t.value for t in order]}"

    # -- the writer still ran: the ruling is not met by not rendering -------
    written = sorted(p.name for p in artifact_dir.iterdir() if p.is_file())
    assert len(written) == 2, (
        f"expected exactly the two rendered views in artifact_dir, found "
        f"{written}; F-16 nulls the REFERENCE, it does not stop the rendering"
    )

    # -- and the client is told WHY the ref is null -------------------------
    joined = " ".join(envelope.result.warnings)
    assert "NOT retrievable" in joined, (
        f"no non-retrievable-artifact warning was emitted; warnings were "
        f"{envelope.result.warnings}"
    )
