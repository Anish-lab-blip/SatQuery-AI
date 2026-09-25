"""Gate-4 end-to-end test — the full control tier in its real local state.

`docs/PHASE15_CONTROLLER_DESIGN.md` §9 states the acceptance shape:

    a full `run()` with all four specialists in their real local
    (degraded/absent) state must return a valid `ResultEnvelope` with
    `degraded=True`, a populated `warnings` list naming each degradation, and
    `confidence.calibrated is None`.

This is deliberately the *unpleasant* configuration rather than a happy path.
Three of the four specialists have no trained weights on this machine, and CROMA
cannot load at all until `use_croma.py` is vendored
(`docs/PHASE14_OPTICAL_SAR_DECISIONS.md` item 3). The `change` head is the
exception: it is trained and benchmarked
(`artifacts/change/levir_change_v001/head.pt`) but is not wired into serving by
default — see `specialists/change/specialist.py`. The system's job in that state
is to produce a coherent, honest answer that says what is missing — not to
crash, and not to invent a number.

The four modules are wired through the REAL registry spec table, so capability
names, asset requirements, the capability assertion and the degrade/available
classification are all exercised. Only the model *builders* are substituted —
deliberately, so the test does not depend on which weights happen to exist on
this machine: it must exercise the degraded path whether or not a `change` head
is present. Loading SmolVLM, RemoteCLIP, STANet and CROMA directly would need
weights (and, for CROMA, a vendored `use_croma.py`) that a unit test cannot
assume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core.controller import AnalysisController
from core.planner import PolicyPlanner
from core.registry import RegistryState, SpecialistRegistry, SpecialistSpec
from core.schemas import (
    AnalysisRequest,
    ConfidenceBreakdown,
    ControllerState,
    Evidence,
    EvidenceType,
    GeoMetadata,
    SpecialistResult,
    Task,
)
from specialists.base import Specialist, SpecialistRequest


# ---------------------------------------------------------------------------
# Degraded stand-ins for the four real specialists
# ---------------------------------------------------------------------------
class _DegradedSpecialist(Specialist):
    """A specialist with no trained weights: it runs, and says so.

    This mirrors what each real builder actually returns in this environment —
    `build_change_specialist` with no checkpoint, `build_grounding_specialist`
    with no head, `build_optical_sar_specialist` with neither CROMA nor a
    fusion head. Each produces a normalised result, marked degraded, carrying a
    `STATISTIC` record of what it could not do.

    `has_head` is declared as a False-valued property because that is how the
    real specialists signal degradation to the registry
    (`specialists/change/specialist.py:173` `has_checkpoint`,
    `specialists/grounding/specialist.py:157` `has_head`,
    `specialists/optical_sar/specialist.py:171,175` `has_encoder`/`has_head`).
    A stub that omitted the flag would be reported AVAILABLE — correctly, since
    the registry does not invent a degradation a specialist never declared.
    """

    def __init__(self, capability: str, task: Task, reason: str) -> None:
        self.name = capability
        self.capabilities = (capability,)
        self.task = task
        self.reason = reason

    @property
    def has_head(self) -> bool:
        """False: no trained head is loaded. The registry reads this."""
        return False

    def validate_request(self, request: SpecialistRequest) -> None:
        return None

    def execute(self, request: SpecialistRequest) -> SpecialistResult:
        return SpecialistResult(
            task=self.task,
            answer="",
            evidence=[
                Evidence(
                    type=EvidenceType.STATISTIC,
                    source_specialist=self.name,
                    score=0.0,
                    payload={"degraded": True, "reason": self.reason},
                )
            ],
            confidence=ConfidenceBreakdown(
                raw=0.0,
                calibrated=None,
                method="uncalibrated",
                degraded=True,
                degradation_reason=self.reason,
            ),
            geospatial=GeoMetadata(),
            warnings=[self.reason],
            degraded=True,
        )

    def produce_evidence(self, result: SpecialistResult) -> list[Evidence]:
        return list(result.evidence)

    def estimate_confidence(self, result: SpecialistResult) -> ConfidenceBreakdown:
        return result.confidence


#: The four frozen capabilities, as the real registry declares them, paired with
#: the reason each is degraded on this machine.
_DEGRADED = {
    "vqa": (Task.VQA, "VLM weights absent; no explanation available"),
    "caption": (Task.CAPTION, "VLM weights absent; no caption available"),
    "grounding": (Task.GROUNDING, "no trained head; zero-shot fallback"),
    "change": (Task.CHANGE, "no trained checkpoint; detector is untrained"),
    "optical_sar": (Task.OPTICAL_SAR, "CROMA could not load; sensor-only"),
}


class _Cfg:
    """Minimal config double: the keys the controller actually reads."""

    device_preference = "cpu"

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self._values = values or {
            "evidence.max_items": 32,
            "agent.timeout_seconds": 120.0,
        }

    def get(self, path: str, default: Any = None) -> Any:
        return self._values.get(path, default)

    @property
    def hash(self) -> str:
        return "gate4cfg00000000"


class _Prediction:
    def __init__(self, task: Task, *, above: bool = True, source: str = "learned"):
        from core.schemas import Intent

        self.intent = Intent(task=task, confidence=0.9, source=source)  # type: ignore[arg-type]
        self.above_threshold = above

    def to_trace(self) -> dict[str, Any]:
        return {"task": self.intent.task.value}


class _Router:
    def __init__(self, prediction: _Prediction) -> None:
        self.prediction = prediction

    def route(self, query: str) -> _Prediction:
        return self.prediction


def _spec(capability: str) -> SpecialistSpec:
    return SpecialistSpec(
        name=capability,
        capabilities=(capability,),
        module="nonexistent",
        builder="build_nothing",
        requires_assets=2 if capability in ("change", "optical_sar") else 1,
    )


_STUB_DIR: Path | None = None


def _stub_assets(count: int) -> list[str]:
    """Real, header-inspectable rasters for the controller's asset resolution.

    `AnalysisController._resolve_assets` delegates to
    `preprocessing.raster.inspect_raster`, so an asset must be an actual
    raster: a missing path is a `RasterReadError`, and TIFF magic bytes with no
    image directory is an unreadable raster. Both are correct behaviour and
    neither lets these tests reach the assembly logic they are about.

    Each stub is therefore a genuine, tiny 4-band GeoTIFF.
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
    """Write a minimal but genuinely readable GeoTIFF."""
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
    _STUB_DIR = Path(tmp_path_factory.mktemp("satquery_gate4_assets"))
    return _STUB_DIR


def _build_registry(specs: tuple[SpecialistSpec, ...]) -> SpecialistRegistry:
    builders: dict[str, Any] = {}
    for spec in specs:
        inner = _DegradedSpecialist(spec.name, *_DEGRADED[spec.name])
        builders[spec.name] = (lambda s: lambda config, **kw: s)(inner)
    return SpecialistRegistry(_Cfg(), specs=specs, builders=builders)


def _controller(specs: tuple[SpecialistSpec, ...], task: Task) -> AnalysisController:
    registry = _build_registry(specs)
    return AnalysisController(
        registry=registry,
        planner=PolicyPlanner(registry),
        router=_Router(_Prediction(task)),
        config=_Cfg(),
    )


# ---------------------------------------------------------------------------
# The Gate-4 acceptance test
# ---------------------------------------------------------------------------
def test_gate4_full_degraded_run_returns_a_coherent_envelope() -> None:
    """A full multi-specialist run with everything degraded still holds up."""
    specs = (_spec("change"), _spec("caption"))
    controller = _controller(specs, Task.CHANGE)

    envelope = controller.run(
        AnalysisRequest(
            assets=_stub_assets(2),
            query="what changed between these two images?",
        )
    )

    # One valid envelope, not an exception.
    assert envelope.run_id
    assert envelope.result.task is Task.CHANGE
    assert envelope.trace.run_id == envelope.run_id

    # Degraded, and honest about it.
    assert envelope.result.degraded is True
    assert envelope.result.warnings, "degradation must be named, not implied"
    assert envelope.result.confidence.calibrated is None
    assert envelope.result.confidence.method == "uncalibrated"

    # Both specialists that ran are visible in the evidence.
    sources = {e.source_specialist for e in envelope.result.evidence}
    assert "change" in sources
    assert "caption" in sources

    # The trace records the whole lifecycle and finished.
    states = [s.state for s in envelope.trace.steps]
    assert ControllerState.RESPOND in states
    assert envelope.trace.finished_at is not None


def test_gate4_warnings_name_each_degradation() -> None:
    """Each degraded specialist's reason must reach the assembled warnings."""
    specs = (_spec("change"), _spec("caption"))
    controller = _controller(specs, Task.CHANGE)
    envelope = controller.run(
        AnalysisRequest(
            assets=_stub_assets(2),
            query="what changed, and describe it",
        )
    )
    joined = " ".join(envelope.result.warnings)
    assert "untrained" in joined
    assert "VLM weights absent" in joined


def test_gate4_all_four_specialists_degraded_single_task() -> None:
    """The four real capabilities, all DEGRADED, requested one at a time."""
    specs = tuple(_spec(c) for c in ("vqa", "caption", "grounding", "change", "optical_sar"))
    registry = _build_registry(specs)

    # Every capability classifies as DEGRADED, never UNAVAILABLE and never
    # silently dropped: a missing artifact is a degradation, not an absence.
    entries = {e.capability: e for e in registry.build_all()}
    for capability, entry in entries.items():
        assert entry.state is RegistryState.DEGRADED, capability
        assert entry.planable is True, capability
        assert entry.detail, f"{capability} must explain its degradation"


@pytest.mark.parametrize(
    "task,capability,assets",
    [
        (Task.VQA, "vqa", 1),
        (Task.CAPTION, "caption", 1),
        (Task.GROUNDING, "grounding", 1),
        (Task.CHANGE, "change", 2),
        (Task.OPTICAL_SAR, "optical_sar", 2),
    ],
)
def test_gate4_each_task_produces_an_honest_degraded_result(
    task: Task, capability: str, assets: int
) -> None:
    """Each task, in its degraded local state, yields a valid honest envelope."""
    controller = _controller((_spec(capability),), task)
    envelope = controller.run(
        AnalysisRequest(
            assets=_stub_assets(assets),
            query="a question",
        )
    )
    assert envelope.result.degraded is True
    assert envelope.result.confidence.calibrated is None
    assert envelope.result.confidence.raw == 0.0
    assert capability in {e.source_specialist for e in envelope.result.evidence}


def test_gate4_unavailable_specialist_is_recorded_not_crashed() -> None:
    """CROMA genuinely cannot load here; that must be a recorded fact."""
    from core.registry import RegistryEntry

    def _unloadable(config: Any, **kw: Any) -> Any:
        from core.errors import ModelUnavailableError

        raise ModelUnavailableError("use_croma.py is not vendored")

    spec = _spec("optical_sar")
    registry = SpecialistRegistry(
        _Cfg(), specs=(spec,), builders={"optical_sar": _unloadable}
    )
    controller = AnalysisController(
        registry=registry,
        planner=PolicyPlanner(registry),
        router=_Router(_Prediction(Task.OPTICAL_SAR)),
        config=_Cfg(),
    )

    envelope = controller.run(
        AnalysisRequest(assets=_stub_assets(2), query="compare")
    )

    # The run completed and reported the omission.
    assert envelope.result.degraded is True
    assert any(e["code"] == "model_unavailable" for e in envelope.trace.errors)
    assert any("vendored" in w for w in envelope.result.warnings)
    assert envelope.trace.finished_at is not None


def test_gate4_trace_is_serialisable() -> None:
    """The envelope must survive `model_dump` — the GUI and report both need it."""
    specs = (_spec("change"), _spec("caption"))
    controller = _controller(specs, Task.CHANGE)
    envelope = controller.run(
        AnalysisRequest(assets=_stub_assets(2), query="what changed?")
    )
    dumped = envelope.model_dump()
    assert dumped["trace"]["run_id"] == envelope.run_id
    assert dumped["result"]["degraded"] is True
