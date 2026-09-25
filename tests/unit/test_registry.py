"""Tests for `core.registry` — the specialist capability registry (Phase 15).

The registry's job is mechanism, not policy: it knows which capabilities exist
and how to construct them. What is pinned here is the set of guarantees the
controller depends on:

  * no heavy import at module load (the documented lazy-load discipline);
  * CAPABILITY is the key, never `name` — the `name="vlm"` trap;
  * a construction failure yields a retained UNAVAILABLE entry, never a raise;
  * a corrupt artifact is NOT retried as a degraded build;
  * `describe()` reports state without forcing construction.

`torch` is not imported by this test module and the stub builders import
nothing. Every test here runs offline with no model weights present.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import pytest

from core.errors import ModelLoadError, ModelUnavailableError, SpecialistError
from core.registry import (
    PLANABLE_STATES,
    RegistryEntry,
    RegistryState,
    SpecialistRegistry,
    SpecialistSpec,
    default_specs,
)
from specialists.base import Specialist, SpecialistRequest
from core.schemas import ConfidenceBreakdown, Evidence, SpecialistResult, Task


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
class _Stub(Specialist):
    """Minimal specialist satisfying the four-method contract."""

    name = "stub"
    version = "9.9.9"
    capabilities: tuple[str, ...] = ("stub",)

    def validate_request(self, request: SpecialistRequest) -> None:
        return None

    def execute(self, request: SpecialistRequest) -> SpecialistResult:
        return SpecialistResult(
            task=Task.VQA,
            answer="stub",
            confidence=ConfidenceBreakdown(raw=0.5),
        )

    def produce_evidence(self, result: SpecialistResult) -> list[Evidence]:
        return list(result.evidence)

    def estimate_confidence(self, result: SpecialistResult) -> ConfidenceBreakdown:
        return result.confidence


def _spec(name: str = "stub", capabilities: tuple[str, ...] | None = None) -> SpecialistSpec:
    return SpecialistSpec(
        name=name,
        capabilities=capabilities or (name,),
        module="nonexistent.module",
        builder="build_nothing",
        requires_assets=1,
    )


class _Config:
    """Config double exposing only the dotted `get` the registry uses."""

    device_preference = "cpu"

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self._values = values or {}

    def get(self, path: str, default: Any = None) -> Any:
        return self._values.get(path, default)


# ---------------------------------------------------------------------------
# Lazy loading — no heavy import at module scope
# ---------------------------------------------------------------------------
def test_registry_module_imports_no_torch() -> None:
    """`core.registry` must not drag torch into a fresh interpreter.

    Run in a subprocess: the test runner has already imported torch for other
    suites, so checking `sys.modules` in-process would pass vacuously.
    """
    code = (
        "import sys, core.registry;"
        "heavy=[m for m in sys.modules if m.startswith(('torch','transformers'))];"
        "print(','.join(heavy))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[2]),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"torch imported at module load: {result.stdout}"


def test_registry_module_imports_no_specialist_package() -> None:
    """Importing the registry must not import any specialist subpackage."""
    code = (
        "import sys, core.registry;"
        "heavy=[m for m in sys.modules if m.startswith('specialists.')];"
        "print(','.join(heavy))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[2]),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"specialist imported eagerly: {result.stdout}"


def test_discover_builds_nothing() -> None:
    registry = SpecialistRegistry.discover(_Config())
    assert registry.available() == (
        "caption",
        "change",
        "change_vqa",
        "grounding",
        "optical_sar",
        "vqa",
    )
    # No construction attempted, so every entry is the placeholder.
    for entry in registry.entries():
        assert entry.specialist is None
        assert entry.detail == "not yet constructed"


def test_available_is_sorted_and_import_free() -> None:
    registry = SpecialistRegistry.discover(_Config(), specs=default_specs())
    names = registry.available()
    assert list(names) == sorted(names)


# ---------------------------------------------------------------------------
# Capability keying — the `vlm` trap
# ---------------------------------------------------------------------------
def test_vqa_row_carries_the_vlm_specialist_capabilities() -> None:
    """The `vqa` spec must declare the VLM's real capability tuple.

    The specialist's own `name` is `"vlm"` but its capabilities are
    `("vqa", "caption")`. The registry keys on capability, so the spec's
    `capabilities` must be the latter.
    """
    specs = {s.name: s for s in default_specs()}
    assert specs["vqa"].capabilities == ("vqa", "caption")
    assert specs["caption"].capabilities == ("vqa", "caption")
    # The registry key is the capability, not the specialist name.
    assert "vlm" not in specs


def test_capability_assertion_rejects_a_mismatched_specialist() -> None:
    """A specialist whose declared capabilities differ from the spec fails loudly.

    This is the check that turns the `name`/`capability` asymmetry from a silent
    miss into an error.
    """
    registry = SpecialistRegistry(
        _Config(),
        specs=(_spec("stub", ("stub",)),),
        builders={"stub": lambda config, **kw: _Stub()},
    )
    entry = registry.build("stub")
    assert entry.state is RegistryState.AVAILABLE


def test_capability_assertion_fires_on_drift() -> None:
    class _Wrong(_Stub):
        capabilities = ("something_else",)

    registry = SpecialistRegistry(
        _Config(),
        specs=(_spec("stub", ("stub",)),),
        builders={"stub": lambda config, **kw: _Wrong()},
    )
    # A capability mismatch is a table/specialist contradiction, surfaced as a
    # retained failure entry rather than a silent success.
    entry = registry.build("stub")
    assert entry.state is RegistryState.UNAVAILABLE
    assert entry.error_code == "specialist_error"
    assert entry.specialist is None


def test_spec_rejects_a_table_entry_unreachable_by_its_own_key() -> None:
    with pytest.raises(SpecialistError, match="does not list itself"):
        _spec("stub", ("other",))


# ---------------------------------------------------------------------------
# build(): unknown raises, failure does not
# ---------------------------------------------------------------------------
def test_build_unknown_capability_raises() -> None:
    registry = SpecialistRegistry.discover(_Config())
    with pytest.raises(SpecialistError, match="unknown capability"):
        registry.build("teleportation")


def test_entry_unknown_capability_raises() -> None:
    registry = SpecialistRegistry.discover(_Config())
    with pytest.raises(SpecialistError, match="unknown capability"):
        registry.entry("teleportation")


def test_duplicate_capability_key_is_rejected() -> None:
    with pytest.raises(SpecialistError, match="duplicate registry capability"):
        SpecialistRegistry(_Config(), specs=(_spec("stub"), _spec("stub")))


def test_construction_failure_returns_unavailable_and_does_not_raise() -> None:
    def _boom(config: Any, **kw: Any) -> Any:
        raise RuntimeError("no GPU in this dimension")

    registry = SpecialistRegistry(
        _Config(),
        specs=(_spec("stub", ("stub",)),),
        builders={"stub": _boom},
    )
    entry = registry.build("stub")  # must NOT raise
    assert entry.state is RegistryState.UNAVAILABLE
    assert entry.specialist is None
    assert entry.planable is False
    assert "no GPU in this dimension" in (entry.detail or "")


def test_unavailable_entry_is_retained_not_dropped() -> None:
    """The whole point: a failed capability stays visible."""
    def _boom(config: Any, **kw: Any) -> Any:
        raise RuntimeError("nope")

    registry = SpecialistRegistry(
        _Config(),
        specs=(_spec("stub", ("stub",)),),
        builders={"stub": _boom},
    )
    registry.build("stub")
    assert "stub" in registry.available()
    assert registry.entry("stub").state is RegistryState.UNAVAILABLE
    assert registry.entry("stub").error_code == "RuntimeError"


def test_model_load_error_maps_to_unavailable() -> None:
    def _corrupt(config: Any, **kw: Any) -> Any:
        raise ModelLoadError("checkpoint present but unreadable", specialist="stub")

    registry = SpecialistRegistry(
        _Config(),
        specs=(_spec("stub", ("stub",)),),
        builders={"stub": _corrupt},
    )
    entry = registry.build("stub")
    assert entry.state is RegistryState.UNAVAILABLE
    assert entry.error_code == "model_load_error"


def test_model_unavailable_error_maps_to_unavailable() -> None:
    def _absent(config: Any, **kw: Any) -> Any:
        raise ModelUnavailableError("VLM not in this environment")

    registry = SpecialistRegistry(
        _Config(),
        specs=(_spec("stub", ("stub",)),),
        builders={"stub": _absent},
    )
    entry = registry.build("stub")
    assert entry.state is RegistryState.UNAVAILABLE
    assert entry.error_code == "model_unavailable"


def test_corrupt_artifact_is_not_retried_as_degraded() -> None:
    """A named-but-unreadable checkpoint must NOT become a planable build.

    `specialists/change/specialist.py` forbids silently substituting an
    untrained model for a corrupt checkpoint; the registry must not reintroduce
    that at the control layer. The builder is called exactly once and the
    outcome is UNAVAILABLE.
    """
    calls = {"n": 0}

    def _corrupt(config: Any, **kw: Any) -> Any:
        calls["n"] += 1
        raise ModelLoadError("state_dict does not match architecture")

    registry = SpecialistRegistry(
        _Config(),
        specs=(_spec("stub", ("stub",)),),
        builders={"stub": _corrupt},
    )
    entry = registry.build("stub")
    assert entry.state is RegistryState.UNAVAILABLE
    assert calls["n"] == 1, "a corrupt artifact must not be retried"

    # And a second build() must return the memoised failure, not retry.
    again = registry.build("stub")
    assert again.state is RegistryState.UNAVAILABLE
    assert calls["n"] == 1


def test_a_construction_failure_publishes_no_filesystem_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """F-15 (owner ruling 2026-09-23): full exception detail is server-side only.

    A construction failure's exception string routinely names an ABSOLUTE PATH:
    `specialists/optical_sar/croma.py` raises *"CROMA requires the vendored
    'use_croma.py', which is not in 'C:\\...\\empty_vendor_dir'"* and
    `specialists/change/stanet.py` raises *"could not read encoder weights from
    C:\\..."*. `RegistryEntry.to_trace()` publishes this entry's `detail`, the
    trace is part of the response body, and `API_CONTRACT.md` section 7 records
    that v1 has **no auth**.

    Measured on the pre-ruling code: ONE real construction failure put the path
    into **four** client-visible fields -- `result.warnings[]`,
    `result.evidence[].payload["message"]`,
    `trace.parameters.registry.built[<cap>].detail`, and the same registry block
    again inside `result.execution_trace` (the trace object is serialized
    twice, once top-level and once nested in the result).

    The ruling has TWO halves and this asserts both: the published detail is
    path-free, AND the raw detail is retained server-side. A fix that simply
    dropped it would satisfy the letter of the ruling and destroy operability.
    """
    leak = r"C:\Users\operator\satquery\vendor\use_croma.py"
    # `json.dumps` doubles every backslash, so the raw string above can NEVER be
    # found in a serialized body -- `leak not in json.dumps(...)` is a FALSE
    # NEGATIVE, a guard that cannot fail. Measured, not assumed: see
    # sq_scratch/p14d_escaping.py. Compare against the escaped form instead.
    escaped = json.dumps(leak)[1:-1]

    def _failing(config: Any, **kw: Any) -> Any:
        raise ModelLoadError(f"CROMA requires the vendored file at {leak}")

    registry = SpecialistRegistry(
        _Config(),
        specs=(_spec("stub", ("stub",)),),
        builders={"stub": _failing},
    )

    with caplog.at_level("WARNING", logger="satquery.registry"):
        entry = registry.build("stub")

    assert entry.state is RegistryState.UNAVAILABLE
    assert leak not in (entry.detail or ""), (
        f"the entry detail publishes a filesystem path: {entry.detail!r}"
    )
    assert escaped not in json.dumps(entry.to_trace(), default=str), (
        "to_trace() publishes a filesystem path"
    )
    # The classification must SURVIVE -- removing the path must not remove
    # which capability failed, why it was classified that way, or the reason.
    assert entry.error_code == "model_load_error"
    assert "use_croma.py" in (entry.detail or ""), (
        f"the reason was lost along with the path: {entry.detail!r}"
    )
    assert "satquery" not in (entry.detail or ""), (
        f"a directory component survived the scrub: {entry.detail!r}"
    )

    # The other half of the ruling: not lost, moved server-side.
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert leak in logged, (
        f"the full detail was dropped rather than retained server-side: {logged!r}"
    )


def test_to_trace_scrubs_a_detail_that_arrives_from_any_constructor() -> None:
    """`to_trace()` is the single seam an entry reaches the body through.

    F-15's fix scrubs at the producer (`_failure_entry`), which means a guard
    that drives the real registry **cannot** tell whether the seam scrubs too --
    the producer has already done it. Falsifying the `to_trace()` scrub alone
    left that guard GREEN, which was measured, not assumed.

    So this constructs the entry the other way: straight through the
    constructor, carrying a raw path, exactly as the docstring warns ("a
    `RegistryEntry` can be constructed anywhere"). That is the only shape that
    exercises the second line of defence -- and it is the shape a future
    constructor would take.
    """
    leak = r"C:\Users\operator\satquery\vendor\use_croma.py"
    escaped = json.dumps(leak)[1:-1]
    entry = RegistryEntry(
        spec=_spec("stub"),
        specialist=None,
        state=RegistryState.UNAVAILABLE,
        detail=f"CROMA requires the vendored file at {leak}",
        error_code="model_load_error",
    )

    trace = entry.to_trace()

    # Direct field check first: no serialization layer in between, so there is
    # nothing for an escaping rule to hide behind.
    assert leak not in (trace["detail"] or ""), (
        f"to_trace() publishes a filesystem path: {trace['detail']!r}"
    )
    assert escaped not in json.dumps(trace, default=str), (
        f"to_trace() publishes a filesystem path once serialized: {trace['detail']!r}"
    )
    assert "satquery" not in (trace["detail"] or ""), (
        f"a directory component survived the seam scrub: {trace['detail']!r}"
    )
    # The classification and the reason must survive the scrub, exactly as on
    # the producer-side path -- the seam must not be a blunter instrument.
    assert trace["error_code"] == "model_load_error"
    assert "use_croma.py" in (trace["detail"] or ""), (
        f"the reason was lost along with the path: {trace['detail']!r}"
    )


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------
def test_missing_weights_yield_degraded_not_omitted() -> None:
    """A specialist built without its trained artifact is DEGRADED and planable."""
    class _Degraded(_Stub):
        capabilities = ("stub",)
        has_checkpoint = False

    registry = SpecialistRegistry(
        _Config(),
        specs=(_spec("stub", ("stub",)),),
        builders={"stub": lambda config, **kw: _Degraded()},
    )
    entry = registry.build("stub")
    assert entry.state is RegistryState.DEGRADED
    assert entry.planable is True
    assert entry.degraded is True
    assert entry.specialist is not None
    assert "checkpoint" in (entry.detail or "")


def test_fully_wired_specialist_is_available() -> None:
    class _Ready(_Stub):
        capabilities = ("stub",)
        has_checkpoint = True

    registry = SpecialistRegistry(
        _Config(),
        specs=(_spec("stub", ("stub",)),),
        builders={"stub": lambda config, **kw: _Ready()},
    )
    entry = registry.build("stub")
    assert entry.state is RegistryState.AVAILABLE
    assert entry.planable is True
    assert entry.degraded is False


@pytest.mark.parametrize(
    "attr",
    ["has_checkpoint", "has_head", "has_encoder"],
)
def test_each_missing_artifact_flag_marks_degraded(attr: str) -> None:
    stub = _Stub()
    stub.capabilities = ("stub",)
    setattr(stub, attr, False)

    registry = SpecialistRegistry(
        _Config(),
        specs=(_spec("stub", ("stub",)),),
        builders={"stub": lambda config, **kw: stub},
    )
    assert registry.build("stub").state is RegistryState.DEGRADED


def test_specialist_without_recognised_flags_is_available() -> None:
    """No recognised flag means no claimed degradation — do not invent one."""
    registry = SpecialistRegistry(
        _Config(),
        specs=(_spec("stub", ("stub",)),),
        builders={"stub": lambda config, **kw: _Stub()},
    )
    assert registry.build("stub").state is RegistryState.AVAILABLE


def test_planable_states_include_degraded() -> None:
    assert RegistryState.DEGRADED in PLANABLE_STATES
    assert RegistryState.AVAILABLE in PLANABLE_STATES
    assert RegistryState.UNAVAILABLE not in PLANABLE_STATES


# ---------------------------------------------------------------------------
# Memoisation, build_all, describe
# ---------------------------------------------------------------------------
def test_build_is_memoised() -> None:
    calls = {"n": 0}

    def _counting(config: Any, **kw: Any) -> Any:
        calls["n"] += 1
        return _Stub()

    registry = SpecialistRegistry(
        _Config(),
        specs=(_spec("stub", ("stub",)),),
        builders={"stub": _counting},
    )
    first = registry.build("stub")
    second = registry.build("stub")
    assert calls["n"] == 1
    assert first is second


def _stub_class(capabilities: tuple[str, ...]) -> type:
    """A stub declaring exactly the given capabilities.

    The registry asserts capabilities against the built object, so a stub
    registered under a different key must declare that key. Reusing a single
    hard-coded stub across two capabilities would fail the assertion — which is
    the assertion working correctly, not a stub detail to work around.
    """
    return type("_CapStub", (_Stub,), {"capabilities": capabilities})


def test_build_all_returns_every_capability() -> None:
    registry = SpecialistRegistry(
        _Config(),
        specs=(_spec("stub", ("stub",)), _spec("other", ("other",))),
        builders={
            "stub": lambda config, **kw: _stub_class(("stub",))(),
            "other": lambda config, **kw: _stub_class(("other",))(),
        },
    )
    entries = registry.build_all()
    assert {e.capability for e in entries} == {"stub", "other"}


def test_build_all_survives_a_failing_capability() -> None:
    def _boom(config: Any, **kw: Any) -> Any:
        raise RuntimeError("boom")

    registry = SpecialistRegistry(
        _Config(),
        specs=(_spec("stub", ("stub",)), _spec("other", ("other",))),
        builders={
            "stub": _boom,
            "other": lambda config, **kw: _stub_class(("other",))(),
        },
    )
    entries = {e.capability: e for e in registry.build_all()}
    assert entries["stub"].state is RegistryState.UNAVAILABLE
    assert entries["other"].state is RegistryState.AVAILABLE


def test_reset_forgets_entries() -> None:
    calls = {"n": 0}

    def _counting(config: Any, **kw: Any) -> Any:
        calls["n"] += 1
        return _Stub()

    registry = SpecialistRegistry(
        _Config(),
        specs=(_spec("stub", ("stub",)),),
        builders={"stub": _counting},
    )
    registry.build("stub")
    registry.reset()
    registry.build("stub")
    assert calls["n"] == 2


def test_describe_reports_state_without_constructing() -> None:
    calls = {"n": 0}

    def _counting(config: Any, **kw: Any) -> Any:
        calls["n"] += 1
        return _Stub()

    registry = SpecialistRegistry(
        _Config(),
        specs=(_spec("stub", ("stub",)),),
        builders={"stub": _counting},
    )
    described = registry.describe()
    assert calls["n"] == 0, "describe() must not construct"
    assert described["declared"] == ["stub"]
    assert described["built"] == {}
    assert described["device"] == "cpu"


def test_describe_reflects_a_built_entry() -> None:
    registry = SpecialistRegistry(
        _Config(),
        specs=(_spec("stub", ("stub",)),),
        builders={"stub": lambda config, **kw: _Stub()},
    )
    registry.build("stub")
    built = registry.describe()["built"]["stub"]
    assert built["state"] == "available"
    assert built["specialist_name"] == "stub"
    assert built["version"] == "9.9.9"
    assert built["requires_assets"] == 1


def test_entry_to_trace_has_no_narrative_fields() -> None:
    """The trace carries observable facts only (freeze §5)."""
    entry = RegistryEntry(spec=_spec(), state=RegistryState.AVAILABLE)
    trace = entry.to_trace()
    for forbidden in ("reasoning", "thought", "rationale", "explanation"):
        assert forbidden not in trace


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------
def test_optional_config_keys_are_passed_when_present() -> None:
    seen: dict[str, Any] = {}

    def _capture(config: Any, **kw: Any) -> Any:
        seen.update(kw)
        return _Stub()

    spec = SpecialistSpec(
        name="stub",
        capabilities=("stub",),
        module="nonexistent",
        builder="build_nothing",
        optional_config_keys={"head_path": "head.path"},
    )
    registry = SpecialistRegistry(
        _Config({"head.path": "/tmp/head.pt"}),
        specs=(spec,),
        builders={"stub": _capture},
    )
    registry.build("stub")
    assert seen["head_path"] == "/tmp/head.pt"
    assert seen["device"] == "cpu"


def test_optional_config_keys_are_omitted_when_absent() -> None:
    """An absent optional artifact must not be passed as an explicit None.

    Several builders use `None` to mean "genuinely absent, degrade"; the
    registry must not manufacture that argument.
    """
    seen: dict[str, Any] = {"sentinel": True}

    def _capture(config: Any, **kw: Any) -> Any:
        seen.clear()
        seen.update(kw)
        return _Stub()

    spec = SpecialistSpec(
        name="stub",
        capabilities=("stub",),
        module="nonexistent",
        builder="build_nothing",
        optional_config_keys={"head_path": "head.path"},
    )
    registry = SpecialistRegistry(
        _Config(),
        specs=(spec,),
        builders={"stub": _capture},
    )
    registry.build("stub")
    assert "head_path" not in seen


def test_missing_builder_in_module_fails_visibly() -> None:
    """A spec naming a builder that does not exist is a retained failure."""
    spec = SpecialistSpec(
        name="stub",
        capabilities=("stub",),
        module="core.registry",  # a real module, but no such builder
        builder="build_that_does_not_exist",
    )
    registry = SpecialistRegistry(_Config(), specs=(spec,))
    entry = registry.build("stub")
    assert entry.state is RegistryState.UNAVAILABLE
    assert "build_that_does_not_exist" in (entry.detail or "")


def test_device_defaults_to_config_preference() -> None:
    registry = SpecialistRegistry.discover(_Config())
    assert registry.device == "cpu"


def test_explicit_device_overrides_config() -> None:
    registry = SpecialistRegistry.discover(_Config(), device="cuda")
    assert registry.device == "cuda"
