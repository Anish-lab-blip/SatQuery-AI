"""SatQuery AI — the analysis controller (Phase 15).

WHAT THIS IS FOR
----------------
`docs/ARCHITECTURE_FREEZE.md` section 5 gives the control tier two verbs that
live here: the controller *dispatches* and *assembles*. It does not decide which
specialists run — that is `core.planner`'s job alone — and it does not compute
anything a specialist computes.

    AnalysisRequest -> router.route() -> PolicyPlanner.plan() -> dispatch -> ResultEnvelope

The controller's own authority is narrow and worth stating exactly: it may
refuse to execute a plan that fails its own preconditions, and otherwise it
executes the plan it is given and assembles the result.

PARTIAL FAILURE NEVER ERASES EVIDENCE
-------------------------------------
A failed step never removes itself from the result. It becomes a recorded
failure with a typed code, and the run continues. Nothing aborts the run; there
is no early exit on step failure.

The reasoning is the same rule this codebase applies three times — C-1's mask
discipline, the grounding degenerate-box guard, and here: **an absence must be
distinguishable from a non-event.** If a specialist fails and simply does not
contribute evidence, a consumer cannot tell whether it ran and found nothing,
never ran, or crashed. So a failed step adds one `STATISTIC` evidence item
carrying its typed code. No schema change is needed: `STATISTIC` is already in
the frozen `EvidenceType` vocabulary, and the grounding and VQA specialists
already emit `STATISTIC` records rather than vanishing.

THE `sources` ASSERTION IS A REAL CHECK
---------------------------------------
`EvidenceEngine.aggregate` records `sources` BEFORE applying its item cap, so a
specialist whose evidence was capped away still appears. That makes "no
specialist ran and left no trace" *checkable* rather than merely intended:

    set(collection.sources) == {capability for each successful step}

A mismatch means a specialist ran and left nothing behind — the exact failure
this design exists to prevent. It is enforced in the `VERIFY` state as a real
assertion, not a comment.

CONFIDENCE IS NOT AVERAGED
--------------------------
Per-specialist confidences are uncalibrated and are not probabilities. Averaging
two uncalibrated hand-weighted sums produces a number that is not a measurement
of anything.

The aggregate confidence is therefore the confidence of the **primary step** —
the step whose capability matches the router's predicted task — passed through
the evidence engine's calibration, which degrades honestly to
`method="uncalibrated"`, `calibrated=None` when no artifact is fitted.

A run whose primary specialist failed has NO confidence: `raw=0.0`,
`degraded=True`. Reporting a secondary specialist's score as the run's
confidence would launder a failure into a number. Secondaries are not discarded
— their raw scores appear in `components`, namespaced by capability. That is
measurable and invents no combination rule.

CONFLICTS ARE REPRESENTED, NOT AVERAGED AWAY
--------------------------------------------
When two specialists make contradictory spatial claims, both are kept and
`trace.contradiction` is set. Selecting a winner requires a precedence weight,
and any such weight is a value judgment with no measurement behind it. It also
produces a third box that *neither specialist predicted* — a fabricated
coordinate.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from core.errors import (
    InvalidRequestError,
    SatQueryError,
    SpecialistError,
    SpecialistTimeoutError,
    UnsupportedQueryError,
    scrub_paths,
)
from core.planner import ExecutionPlan, PlanStep, PolicyPlanner
from core.registry import RegistryEntry, RegistryState, SpecialistRegistry
from core.schemas import (
    AnalysisRequest,
    AssetMetadata,
    Box,
    ConfidenceBreakdown,
    ControllerState,
    Evidence,
    EvidenceType,
    ExecutionTrace,
    GeoMetadata,
    Intent,
    Modality,
    ModelRef,
    ResultEnvelope,
    SpecialistResult,
    Task,
    TraceStep,
)

# NOTE: `specialists.base.SpecialistRequest` is imported lazily inside
# `_execute_one`, NOT at module scope. `specialists.base` imports
# `core.errors`, and `core/__init__.py` re-exports this module, so a
# module-level import here closes a cycle:
#
#     specialists.base -> core.errors -> core/__init__ -> core.controller
#                      -> specialists.base (partially initialised) -> ImportError
#
# Keeping the control tier free of module-scope specialist imports is also the
# documented lazy-load discipline, so the lazy import is correct on both counts.

#: IoU below which two same-system spatial claims are considered contradictory.
#: They name disjoint places. Frozen by design §7.6.
CONTRADICTION_IOU_FLOOR: float = 0.1

#: Fallback run budget in seconds when `agent.timeout_seconds` is unset.
DEFAULT_BUDGET_SECONDS: float = 120.0

#: F-20 (owner ruling 2026-09-23). The controller had no logger: an unhandled
#: specialist failure was wrapped into a client-visible error and the raw
#: internals were never written anywhere an operator could read them. The fix
#: moves the internals OFF every client-visible carrier, so they must land here
#: or be lost.
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step outcome
# ---------------------------------------------------------------------------
@dataclass
class StepOutcome:
    """What one plan step produced, including its failure."""

    step: PlanStep
    result: SpecialistResult | None = None
    registry_entry: RegistryEntry | None = None
    error: SatQueryError | None = None
    duration_ms: float = 0.0
    skipped_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.result is not None and self.error is None

    @property
    def capability(self) -> str:
        return self.step.capability

    def to_trace(self) -> dict[str, Any]:
        return {
            "step_id": self.step.step_id,
            "capability": self.step.capability,
            "ok": self.ok,
            "skipped_reason": self.skipped_reason,
            "error_code": self.error.code if self.error else None,
            "duration_ms": round(self.duration_ms, 3),
            "registry_state": (
                self.registry_entry.state.value if self.registry_entry else None
            ),
        }


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
class AnalysisController:
    """Executes a plan and assembles one `ResultEnvelope`."""

    def __init__(
        self,
        *,
        registry: SpecialistRegistry,
        planner: PolicyPlanner,
        router: Any | None = None,
        evidence: Any | None = None,
        config: Any | None = None,
        budget_seconds: float | None = None,
    ) -> None:
        """
        Args:
            registry: constructs specialists on demand.
            planner: the only component that chooses what runs.
            router: an `IntentRouter`. Optional so a caller can drive the
                controller with a forced task and no router at all.
            evidence: an `EvidenceEngine`. Constructed from config when omitted.
            config: the central config, for evidence/budget settings.
            budget_seconds: per-run wall-clock budget, checked between steps.
        """
        self.registry = registry
        self.planner = planner
        self.router = router
        self.config = config

        if evidence is None:
            from evidence.confidence import load_calibration
            from evidence.engine import EvidenceEngine

            # The calibration artifact is loaded HERE, at the one place the
            # deployed controller builds its evidence engine. Before 2026-09-22
            # this call passed no `calibration=`, so `EvidenceEngine` fell back
            # to its `None` default and every confidence the system emitted said
            # `method="uncalibrated"` -- even once a fitted artifact existed.
            # That was the single section 67 immutable-decision violation
            # ("calibrated confidence"), and it was a wiring gap, not a missing
            # capability: `load_calibration` already read the frozen config keys
            # and `TemperatureCalibration` already applied the mapping.
            #
            # `load_calibration` returns `None` -- never a fabricated default --
            # when the switch is off or the artifact is absent, so an
            # artifact-less deployment still degrades to the honest
            # pass-through rather than failing to construct.
            evidence = (
                EvidenceEngine.from_config(config, calibration=load_calibration(config))
                if config is not None
                else EvidenceEngine()
            )
        self.evidence = evidence

        if budget_seconds is None:
            budget_seconds = float(
                config.get("agent.timeout_seconds", DEFAULT_BUDGET_SECONDS)
                if config is not None
                else DEFAULT_BUDGET_SECONDS
            )
        self.budget_seconds = budget_seconds

    # -- public API --------------------------------------------------------

    def run(
        self,
        request: AnalysisRequest,
        *,
        assets: Sequence[AssetMetadata] | None = None,
        asset_modalities: Mapping[str, str] | None = None,
    ) -> ResultEnvelope:
        """Execute a request end to end and return one envelope.

        Args:
            request: the caller's request.
            assets: pre-resolved asset metadata, bypassing inspection. A caller
                that has already validated rasters should pass them rather than
                make the controller inspect the same files twice.
            asset_modalities: optional explicit modality per asset, keyed by
                basename or full path. Authoritative over band-count inference,
                which matters for files whose band count is ambiguous (a 1-band
                SAR and a 1-band panchromatic optical are indistinguishable by
                count alone).

        Never raises for a specialist-level failure: those are recorded in the
        result, the trace and the evidence. Raises only for a caller error the
        schema cannot catch.
        """
        started = time.perf_counter()
        run_id = request.run_id or self._new_run_id()
        trace = ExecutionTrace(
            run_id=run_id,
            query=request.query,
            # F-13. `inputs` is a CLIENT-VISIBLE field -- `ResultEnvelope` is
            # returned verbatim by `POST /v1/analyze`, `trace` included -- and
            # it was echoing `request.assets` after the caller had turned asset
            # *handles* into filesystem *paths*. The response therefore named the
            # server-side location of every uploaded byte, which is the exact
            # disclosure `AssetHandle.to_response()` refuses to make.
            #
            # The basename is what remains here, and it is not a disclosure: for
            # the upload path it is the opaque handle the client already holds,
            # and for a direct path it is the file's own name without its
            # directory. `_asset_label` is defined next to this call so that the
            # one place deriving the label and the one place deriving the
            # modality cannot drift apart.
            inputs=[_asset_label(a) for a in request.assets],
            config_hash=self.config.hash if self.config is not None else None,
        )
        self._record(trace, ControllerState.RECEIVE, {"asset_count": len(request.assets)})

        resolved = (
            list(assets)
            if assets is not None
            else self._resolve_assets(request, asset_modalities=asset_modalities)
        )
        trace.modalities = self._modalities(resolved)
        # F-14. The F-13 disclosure again, one record later. `trace.steps[]` is
        # part of the same client-visible `ExecutionTrace` (`POST /v1/analyze`
        # returns the envelope verbatim), and this PARSE detail was writing
        # `list(request.assets)` -- which by now are the PATHS the Space derived
        # from the client's handles -- so the response named the server-side
        # location of every uploaded byte in a second place. Measured
        # 2026-09-22 end-to-end through the real Space: the client sent the
        # handle `asset_d243f7f85d8c2f3c02981f0af9737f01` and received back
        # `C:\Users\anish\sq_scratch\...\assets\asset_d243...7f01.tif`.
        #
        # `_asset_label` is the ONE place the basename rule lives; it is reused
        # here rather than restated, because a second copy of an existing rule
        # is a second thing that can drift from it.
        self._record(
            trace,
            ControllerState.PARSE,
            {
                "inputs": [_asset_label(a) for a in request.assets],
                "modalities": [m.value for m in trace.modalities],
                "query_length": len(request.query),
            },
        )

        # -- VALIDATE ------------------------------------------------------
        prediction = self._route(request)
        trace.intent = prediction.intent if prediction is not None else None
        trace.task = prediction.intent.task if prediction is not None else request.force_task
        self._record(
            trace,
            ControllerState.VALIDATE,
            {
                "assets": len(request.assets),
                "force_task": request.force_task.value if request.force_task else None,
                "router": (
                    prediction.to_trace() if prediction is not None else None
                ),
            },
        )

        # -- PLAN ----------------------------------------------------------
        if prediction is None:
            # No router and no forced task: nothing can be planned. This is a
            # caller error, not a query the system declined.
            raise UnsupportedQueryError(
                "no router configured and no force_task supplied; there is no "
                "way to choose a specialist",
                recoverable=False,
            )

        plan = self.planner.plan(prediction, request, assets=resolved)
        trace.parameters = {
            **plan.to_trace(),
            "budget_seconds": self.budget_seconds,
            "registry": self.registry.describe(),
        }
        self._record(trace, ControllerState.PLAN, {"plan": plan.to_trace()})

        if plan.refused:
            return self._finish_refusal(request, trace, plan, started)

        # -- EXECUTE -------------------------------------------------------
        outcomes = self._execute(trace, plan, resolved, request, started)

        # F-19 (owner ruling 2026-09-23): re-snapshot the registry AFTER
        # execution. `registry.describe()` reports the registry's MEMOISED
        # entries, so the snapshot taken in PLAN above reports the PREVIOUS
        # request's builds -- while `trace.selected_models`, populated inside
        # `_execute`, reports THIS one. That made a single trace carry two
        # clocks: on a cold process the first request's trace said
        # `registry.built == {}` while its own run had built `optical_sar`.
        # Measured in pass 13 with two identical payloads on one controller:
        # request #1 body `built == []`, request #2 body `built ==
        # ['optical_sar']`.
        #
        # Re-snapshotting rather than moving the PLAN-time assignment keeps the
        # PLAN record's own context intact and leaves the early-return refusal
        # path (`_finish_refusal`, above) with a snapshot that is still correct
        # -- nothing is built on a refusal.
        trace.parameters["registry"] = self.registry.describe()

        # -- AGGREGATE -----------------------------------------------------
        result = self._aggregate(trace, plan, outcomes, prediction, request)
        self._record(
            trace,
            ControllerState.AGGREGATE,
            {
                "evidence": (
                    self.evidence.aggregate(
                        [o.result for o in outcomes if o.result is not None]
                    ).summary()
                ),
                "degraded": result.degraded,
            },
        )

        # -- VERIFY --------------------------------------------------------
        self._verify(trace, outcomes)
        self._detect_contradiction(trace, result, outcomes)
        self._record(trace, ControllerState.VERIFY, {"contradiction": trace.contradiction})

        # -- RESPOND -------------------------------------------------------
        result.execution_trace = trace
        trace.outputs = [e.evidence_id for e in result.evidence]
        trace.confidence = result.confidence
        self._record(trace, ControllerState.RESPOND, {"answer_length": len(result.answer)})
        trace.finished_at = self._now()

        return ResultEnvelope(run_id=run_id, result=result, trace=trace)

    def health(self) -> dict[str, Any]:
        """Registry state for the GUI's health panel. Constructs everything.

        .. deprecated:: 2026-09-22
            **NO LONGER A PUBLIC API PATH.** Retired by the owner ruling of
            2026-09-22 (H-1/H-2); do not expose this from an HTTP route.

        WHY IT WAS RETIRED
        ------------------
        Two independent reasons, and both are structural rather than cosmetic:

        1. **The shape is not the contract's.** `docs/API_CONTRACT.md` section
           2.1 fixes `GET /v1/health` to `HealthStatus` (`core/schemas.py:392`),
           which is `extra="forbid"` with the fields
           `status, schema_version, models, device, gpu_available`. This method
           returns `registry, degraded, unavailable, device` -- so constructing a
           `HealthStatus` from its output raises `extra_forbidden` on three keys
           and reports `status`/`gpu_available` missing. That is finding H-1, and
           it was asserted by
           `tests/integration/test_step7_backend_chain.py::TestH1ControllerHealthShape`.

        2. **The vocabulary is the registry's, not the contract's.** The values
           are `RegistryState` words (`available`/`degraded`/`unavailable`), and
           the contract's vocabulary
           (`loaded`/`absent`/`unavailable`/`not_requested`/`evicted`,
           `API_CONTRACT.md` section 2.3) overlaps on exactly one word with a
           *different meaning*. Serving these values would have relabelled every
           missing artifact a defect.

        THE REPLACEMENT
        ---------------
        `app.deployment.deployment_report()` is the single public source. It reads
        the registry's spec table and the filesystem and **constructs nothing**,
        which is the other half of why this method could not be the health
        source: its own docstring says it "Constructs everything", and
        requirement 4 of `docs/DEPLOYMENT_ARCHITECTURE.md` section 3.3 forbids
        loading a model for a metadata request. Calling this from `/v1/health`
        would spend the 5 GPU-minute daily budget on health probes.

        It is retained because it is a genuinely useful *operator diagnostic*
        (it is the only place that reports per-capability construction outcomes),
        and callers that already hold a constructed controller may use it. What
        it must not be is the shape a client sees.

        Returns:
            A dict in the REGISTRY vocabulary. Not a `HealthStatus`, deliberately
            -- translating belongs to `app.deployment`, in one place, where it is
            tested.
        """
        entries = self.registry.build_all()
        states = {e.capability: e.state.value for e in entries}
        return {
            "registry": states,
            "degraded": [c for c, s in states.items() if s == RegistryState.DEGRADED.value],
            "unavailable": [
                c for c, s in states.items() if s == RegistryState.UNAVAILABLE.value
            ],
            "device": self.registry.device,
        }

    # -- routing -----------------------------------------------------------

    def _route(self, request: AnalysisRequest) -> Any | None:
        """Produce a prediction: forced when asked, else routed.

        `force_task` bypasses the router entirely — recorded as
        `source="forced"`, which `Intent` already supports. The router's
        opinion is not consulted, and that fact is visible in the trace.
        """
        if request.force_task is not None:
            from router.classifier import RouterPrediction

            intent = Intent(
                task=request.force_task,
                confidence=1.0,
                source="forced",
            )
            return RouterPrediction(intent=intent, above_threshold=True)
        if self.router is None:
            return None
        return self.router.route(request.query)

    @staticmethod
    def _resolve_assets(
        request: AnalysisRequest,
        *,
        asset_modalities: Mapping[str, str] | None = None,
    ) -> list[AssetMetadata]:
        """Describe the request's assets by INSPECTING them.

        This delegates to `preprocessing.raster.inspect_raster`, which opens the
        raster header and reports band count, dtype, CRS, transform, bounds and
        resolution. It does NOT decode pixel data — a 12-band Sentinel-2 tile is
        ~100 MB and a four-specialist plan would decode it repeatedly.

        Why this is not shallow
        -----------------------
        An earlier version built `AssetMetadata(path=path)`, leaving `geo` and
        `modality` at their defaults. Every asset then reached every specialist
        with `band_count=None` and `modality=UNKNOWN`, which is indistinguishable
        from an unreadable file. The visible consequence: a valid 4-band optical
        + 2-band SAR pair was rejected by `OpticalSarSpecialist.validate_request`
        with "could not identify one optical and one SAR asset from modalities
        ['unknown', 'unknown']" — a user-visible refusal of a perfectly good
        request. The same loss starved every modality- and resolution-dependent
        branch in every specialist.

        A missing or unreadable asset raises `RasterReadError` (typed), which the
        controller surfaces with a user message. It does not degrade to an empty
        description, because "we could not read this" and "this had nothing to
        report" must stay distinguishable.

        Args:
            request: supplies the asset paths.
            asset_modalities: explicit modality overrides, keyed by basename or
                full path. Passed through to `inspect_raster(explicit_modality=)`,
                which treats a declared modality as authoritative over the
                band-count heuristic. A declaration that is not a valid
                `Modality` raises; silently ignoring it would make the override
                quietly ineffective, which is worse than a loud error.
        """
        overrides = _normalise_modality_overrides(asset_modalities)
        return [
            _inspect_asset(path, overrides.get(str(path)) or overrides.get(Path(path).name))
            for path in request.assets
        ]

    @staticmethod
    def _modalities(assets: Sequence[AssetMetadata]) -> list[Modality]:
        """Distinct modalities, in first-seen order."""
        seen: list[Modality] = []
        for asset in assets:
            modality = getattr(asset, "modality", Modality.UNKNOWN)
            if modality not in seen:
                seen.append(modality)
        return seen

    # -- execution ---------------------------------------------------------

    def _execute(
        self,
        trace: ExecutionTrace,
        plan: ExecutionPlan,
        assets: list[AssetMetadata],
        request: AnalysisRequest,
        started: float,
    ) -> list[StepOutcome]:
        """Run each step, recording outcomes. Never aborts on failure."""
        outcomes: list[StepOutcome] = []
        trace.workflow = [step.step_id for step in plan.steps]

        for step in plan.steps:
            # Budget is checked BETWEEN steps (§5.2). A true per-step timeout
            # needs a worker process or a signal handler, both of which
            # conflict with the single-process monolith constraint more than
            # they benefit. The honest position: v1 bounds total wall clock and
            # records what it skipped.
            elapsed = time.perf_counter() - started
            if elapsed > self.budget_seconds:
                outcome = StepOutcome(
                    step=step,
                    skipped_reason=f"budget_exceeded:{elapsed:.1f}s",
                )
                outcomes.append(outcome)
                trace.errors.append(
                    {
                        "step_id": step.step_id,
                        "code": SpecialistTimeoutError.code,
                        "message": f"skipped: run budget of "
                        f"{self.budget_seconds}s exceeded",
                    }
                )
                self._record(
                    trace, ControllerState.EXECUTE, outcome.to_trace()
                )
                continue

            outcome = self._execute_one(step, assets, request, trace)
            outcomes.append(outcome)
            self._record(trace, ControllerState.EXECUTE, outcome.to_trace())
            trace.timings[step.step_id] = round(outcome.duration_ms, 3)

            if outcome.error is not None:
                # F-15 (owner ruling 2026-09-23): the trace is CLIENT-FACING, so
                # it carries the operator-safe message only. It previously
                # preferred `error.detail` -- the technical text, which can name
                # filesystem paths and library internals -- over `user_message`.
                # The technical detail stays available on the `StepOutcome`
                # server-side, which is where a maintainer reads it.
                trace.errors.append(
                    {
                        "step_id": step.step_id,
                        "capability": step.capability,
                        "code": outcome.error.code,
                        "message": outcome.error.user_message
                        or "The step failed.",
                        "recoverable": outcome.error.recoverable,
                    }
                )

        trace.selected_models = self._selected_models(outcomes)
        return outcomes

    def _execute_one(
        self,
        step: PlanStep,
        assets: list[AssetMetadata],
        request: AnalysisRequest,
        trace: ExecutionTrace,
    ) -> StepOutcome:
        """Construct, validate and run one step. All failures are captured."""
        from specialists.base import SpecialistRequest

        began = time.perf_counter()

        entry = self.registry.build(step.capability)
        if entry.specialist is None:
            return StepOutcome(
                step=step,
                registry_entry=entry,
                error=self._unavailable_error(step, entry),
                duration_ms=(time.perf_counter() - began) * 1000.0,
            )

        specialist = entry.specialist
        specialist_request = SpecialistRequest(
            assets=list(assets),
            query=request.query,
            params=dict(step.params),
            run_id=trace.run_id,
        )

        try:
            specialist.validate_request(specialist_request)
            result = specialist.execute(specialist_request)
        except SatQueryError as exc:
            return StepOutcome(
                step=step,
                registry_entry=entry,
                error=exc,
                duration_ms=(time.perf_counter() - began) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001 - wrapped, then recorded
            # F-15 (owner ruling 2026-09-23): the exception's own text is
            # SERVER-SIDE only. It goes into `detail`, which stays on the
            # `StepOutcome` for a maintainer, and NOT into `user_message`, which
            # is what the client-facing trace shows.
            #
            # The CLASSIFICATION survives, though: the client still learns that
            # this was an UNHANDLED failure rather than a known typed error.
            # Losing that would hide the difference between "we predicted this
            # failure mode" and "we did not", which is exactly the distinction
            # `code: unhandled` exists to record.
            #
            # F-20 (owner ruling 2026-09-23) — F-15 WAS ONLY HALF IMPLEMENTED.
            # The paragraph above was true of the trace and false of everything
            # else: `_warnings` and `_failure_evidence` publish
            # `scrub_paths(error.detail)`, and `detail` was
            # `f"unhandled {type(exc).__name__}: {exc}"` -- so ONE unhandled
            # failure reached the client as "Internal detail is withheld" in
            # `trace.errors[].message` and as "unhandled RuntimeError: kaboom"
            # in `result.warnings[]` and `evidence[].payload["message"]`. The
            # response contradicted itself about what it had decided to disclose.
            #
            # The repair is at the PRODUCER, not at each carrier. `detail` is
            # the published field for typed errors -- F-15 established that, and
            # three tests pin path-free diagnostics that must survive -- so
            # making the unhandled case publishable is what makes all three
            # carriers agree by construction. Two internals leave it:
            #
            #   * `{exc}` -- the exception's own message, which can carry paths,
            #     library internals and internal identifiers;
            #   * `type(exc).__name__` -- a Python implementation detail. A
            #     client cannot act on "RuntimeError" versus "ValueError", and
            #     publishing it makes the trace's "internal detail is withheld"
            #     claim untrue.
            #
            # Both are preserved for the operator: `context` is never
            # serialized (`StepOutcome.to_trace()` publishes `error_code` only),
            # and the traceback goes to the log.
            _log.error(
                "unhandled failure in step %s (capability=%s): %s: %s",
                step.step_id,
                step.capability,
                type(exc).__name__,
                exc,
                exc_info=exc,
            )
            return StepOutcome(
                step=step,
                registry_entry=entry,
                error=SpecialistError(
                    "unhandled error",
                    user_message=(
                        "The step failed with an unhandled error. Internal "
                        "detail is withheld; see the server-side diagnostics."
                    ),
                    specialist=step.capability,
                    context={
                        "step_id": step.step_id,
                        "code": "unhandled",
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                ),
                duration_ms=(time.perf_counter() - began) * 1000.0,
            )

        return StepOutcome(
            step=step,
            result=result,
            registry_entry=entry,
            duration_ms=(time.perf_counter() - began) * 1000.0,
        )

    @staticmethod
    def _unavailable_error(step: PlanStep, entry: RegistryEntry) -> SpecialistError:
        """A planned-but-unconstructible capability, as a typed error."""
        from core.errors import ModelUnavailableError

        detail = f"{step.capability} was planned but could not be constructed"
        if entry.detail:
            detail = f"{detail}: {entry.detail}"
        return ModelUnavailableError(detail)

    def _selected_models(self, outcomes: Iterable[StepOutcome]) -> list[ModelRef]:
        """Model identities for the trace, from each specialist that ran.

        Built as `ModelRef` rather than raw dicts: `ExecutionTrace.selected_models`
        is typed `list[ModelRef]`, and pydantic would coerce a dict only by
        emitting a serialization warning. Constructing the type explicitly keeps
        the contract honest instead of relying on coercion.
        """
        refs: list[ModelRef] = []
        seen: set[str] = set()
        for outcome in outcomes:
            specialist = (
                outcome.registry_entry.specialist if outcome.registry_entry else None
            )
            if specialist is None:
                continue
            for ref in specialist.model_refs():
                key = f"{ref.get('name')}:{ref.get('role')}"
                if key in seen:
                    continue
                seen.add(key)
                refs.append(
                    ModelRef(
                        name=str(ref.get("name", "unknown")),
                        revision=ref.get("revision"),
                        role=ref.get("role"),
                    )
                )
        return refs

    # -- refusals ----------------------------------------------------------

    def _finish_refusal(
        self,
        request: AnalysisRequest,
        trace: ExecutionTrace,
        plan: ExecutionPlan,
        started: float,
    ) -> ResultEnvelope:
        """A refusal is a successful run with no steps, not an error."""
        refusal = plan.refusal
        message = refusal.user_message if refusal else "The request was refused."
        result = SpecialistResult(
            task=trace.task or Task.UNSUPPORTED,
            answer=message,
            confidence=ConfidenceBreakdown(
                raw=0.0,
                calibrated=None,
                method="uncalibrated",
                degraded=True,
                degradation_reason=(
                    f"refused:{refusal.reason}" if refusal else "refused"
                ),
            ),
            warnings=list(plan.notes),
            degraded=True,
        )
        trace.fallbacks.extend(plan.notes)
        self._record(
            trace,
            ControllerState.AGGREGATE,
            {"refused": True, "refusal": refusal.to_trace() if refusal else None},
        )
        self._record(trace, ControllerState.VERIFY, {"steps": 0})
        result.execution_trace = trace
        trace.confidence = result.confidence
        self._record(trace, ControllerState.RESPOND, {"refused": True})
        trace.finished_at = self._now()
        return ResultEnvelope(run_id=trace.run_id, result=result, trace=trace)

    # -- assembly ----------------------------------------------------------

    def _aggregate(
        self,
        trace: ExecutionTrace,
        plan: ExecutionPlan,
        outcomes: list[StepOutcome],
        prediction: Any,
        request: AnalysisRequest,
    ) -> SpecialistResult:
        """Merge N step outcomes into one `SpecialistResult` (§7.2)."""
        successful = [o for o in outcomes if o.ok]
        results = [o.result for o in successful if o.result is not None]

        # A failed step gets a STATISTIC record so its absence is visible (§5.3).
        failure_evidence = [self._failure_evidence(o) for o in outcomes if not o.ok]

        collection = self.evidence.aggregate(results) if results else None
        items: list[Evidence] = list(collection.items) if collection else []
        items.extend(self._synthesis_evidence(results, request))
        items.extend(failure_evidence)

        # Failure evidence is appended after aggregation and therefore carries
        # engine-assigned ids only by coincidence. Renumber the whole set so
        # every id remains unique and deterministic.
        items = self._renumber(items)

        task = trace.task or Task.UNSUPPORTED
        answer = self._answer(results, outcomes, plan)

        return SpecialistResult(
            task=task,
            answer=answer,
            labels=self._merge_list(results, "labels"),
            regions=self._concat(results, "regions"),
            boxes=self._concat(results, "boxes"),
            masks=self._merge_list(results, "masks"),
            change_map=self._first_non_none(results, "change_map"),
            evidence=items,
            confidence=self._confidence(trace, prediction, results),
            geospatial=self._first_geospatial(results),
            warnings=self._warnings(plan, outcomes, results),
            degraded=self._degraded(plan, outcomes, results),
        )

    # -- evidence ----------------------------------------------------------

    @staticmethod
    def _client_error_reason(error: SatQueryError) -> str:
        """The ONE client-facing reason string for a failed step (F-20).

        Three client-visible carriers report a failed step -- `result.warnings[]`,
        `evidence[].payload["message"]` and `trace.errors[].message` -- and before
        F-20 two of them published an internals string while the third promised
        the internals were withheld. The response contradicted itself.

        F-20's repair has two halves and both are needed:

          * the PRODUCER stopped putting internals into `detail`
            (`_execute_one`'s unhandled branch), so the published field is
            publishable; and
          * this function exists so the rule is written ONCE. Two copies of
            `scrub_paths(detail) or user_message` is how a rule drifts: the next
            fix lands on whichever copy the author happened to open.

        `detail` is preferred over `user_message` on purpose. F-15 established
        that a typed error's `detail` is the actionable diagnostic ("... has no
        builder 'build_x'", "no GPU in this dimension") while the base
        `user_message` is generic, and three tests pin exactly those
        diagnostics. The scrub keeps a path-free reason and reduces any absolute
        path to its basename.
        """
        return scrub_paths(error.detail) or error.user_message

    @staticmethod
    def _failure_evidence(outcome: StepOutcome) -> Evidence:
        """One `STATISTIC` item recording that a step did not produce a result."""
        code = outcome.error.code if outcome.error else "unknown"
        message = ""
        if outcome.error is not None:
            # F-15 / F-20 (owner rulings 2026-09-23): the client-facing `message`
            # comes from the single shared rule, not from a second copy of it.
            message = AnalysisController._client_error_reason(outcome.error)
        elif outcome.skipped_reason:
            message = outcome.skipped_reason
        return Evidence(
            type=EvidenceType.STATISTIC,
            source_specialist=outcome.step.capability,
            score=0.0,
            payload={
                "step_failed": True,
                "step_id": outcome.step.step_id,
                "code": code,
                "message": message,
                "skipped_reason": outcome.skipped_reason,
            },
        )

    @staticmethod
    def _synthesis_evidence(
        results: Sequence[SpecialistResult], request: AnalysisRequest
    ) -> list[Evidence]:
        """A `STATISTIC` recording the run itself.

        Two things a consumer must be able to see without inferring them: that
        several specialists were merged into one answer, and which query they
        were answering. Neither belongs to any single specialist, so neither
        can be attributed to one.
        """
        if not results:
            return []
        capabilities = [r.task.value for r in results]
        if len(results) < 2:
            return []
        return [
            Evidence(
                type=EvidenceType.STATISTIC,
                source_specialist="controller",
                score=None,
                payload={
                    "synthesis": True,
                    "contributors": capabilities,
                    "query": request.query,
                },
            )
        ]

    @staticmethod
    def _renumber(items: list[Evidence]) -> list[Evidence]:
        """Assign unique sequential ids across the final evidence list.

        `specialist_result.evidence` rejects duplicate ids, and the engine's
        ids stop being authoritative once controller-authored items are
        appended. Renumbering here keeps the collection citable.
        """
        return [
            item.model_copy(update={"evidence_id": f"evidence_{i:03d}"})
            for i, item in enumerate(items, start=1)
        ]

    # -- answer ------------------------------------------------------------

    @staticmethod
    def _answer(
        results: Sequence[SpecialistResult],
        outcomes: Sequence[StepOutcome],
        plan: ExecutionPlan,
    ) -> str:
        """A priority list, not a heuristic (design §7.3).

        1. The VLM explains, when it ran (freeze §5).
        2. Else the highest-confidence specialist answer, attributed.
        3. Else a plain statement of what failed, from the typed codes.
        4. Else the refusal message.

        No template composes a narrative from evidence: that is the VLM's job,
        and only when it actually ran.
        """
        for outcome in outcomes:
            if not outcome.ok or outcome.result is None:
                continue
            if outcome.capability in ("vqa", "caption") and outcome.result.answer:
                return outcome.result.answer

        answered = [r for r in results if r.answer]
        if answered:
            best = max(answered, key=lambda r: r.confidence.value)
            return f"[{best.task.value}] {best.answer}"

        failures = [o for o in outcomes if not o.ok]
        if failures:
            named = ", ".join(
                f"{o.step.capability} ({o.error.code if o.error else 'skipped'})"
                for o in failures
            )
            return f"No result could be produced: {named}."

        if plan.refusal is not None:
            return plan.refusal.user_message
        return "No specialist produced an answer."

    # -- merging -----------------------------------------------------------

    @staticmethod
    def _merge_list(results: Sequence[SpecialistResult], attr: str) -> list[Any]:
        """Union, order-preserving, deduped (design §7.2)."""
        out: list[Any] = []
        for result in results:
            for value in getattr(result, attr, []) or []:
                if value not in out:
                    out.append(value)
        return out

    @staticmethod
    def _concat(results: Sequence[SpecialistResult], attr: str) -> list[Any]:
        """Concatenation in plan-step order, duplicates preserved."""
        out: list[Any] = []
        for result in results:
            out.extend(getattr(result, attr, []) or [])
        return out

    @staticmethod
    def _first_non_none(results: Sequence[SpecialistResult], attr: str) -> Any:
        for result in results:
            value = getattr(result, attr, None)
            if value is not None:
                return value
        return None

    @staticmethod
    def _first_geospatial(results: Sequence[SpecialistResult]) -> GeoMetadata:
        """First non-empty geo metadata.

        Not merged: merging CRS or transform fields across assets would
        silently mix two coordinate reference systems into one field, which is
        a coordinate error the schema cannot catch (design §7.2).
        """
        for result in results:
            geo = result.geospatial
            if geo is not None and (geo.has_crs or geo.crs or geo.bounds):
                return geo
        return GeoMetadata()

    @staticmethod
    def _warnings(
        plan: ExecutionPlan,
        outcomes: Sequence[StepOutcome],
        results: Sequence[SpecialistResult],
    ) -> list[str]:
        """Union of plan notes, per-step warnings and controller warnings."""
        warnings: list[str] = list(plan.notes)
        for result in results:
            for warning in result.warnings:
                if warning not in warnings:
                    warnings.append(warning)
        for outcome in outcomes:
            if outcome.error is not None:
                # F-15 / F-20 (owner rulings 2026-09-23): the client-facing
                # warning carries the CLASSIFICATION (which capability, which
                # error code) and a reason from the single shared rule -- never
                # a raw `detail`, and never a second hand-rolled copy of the
                # scrub. `detail` is free-form and a specialist raising at
                # execution time can put an absolute path in it;
                # `_unavailable_error` is not the only producer.
                #
                # Scrubbed rather than replaced, so a path-free reason the
                # client can act on survives.
                message = (
                    f"{outcome.step.capability} failed "
                    f"({outcome.error.code}): "
                    f"{AnalysisController._client_error_reason(outcome.error)}"
                )
                if message not in warnings:
                    warnings.append(message)
            elif outcome.skipped_reason:
                message = (
                    f"{outcome.step.capability} skipped "
                    f"({outcome.skipped_reason})"
                )
                if message not in warnings:
                    warnings.append(message)
        return warnings

    @staticmethod
    def _degraded(
        plan: ExecutionPlan,
        outcomes: Sequence[StepOutcome],
        results: Sequence[SpecialistResult],
    ) -> bool:
        """Degraded when any step degraded, any step failed, or the route was uncertain."""
        if plan.uncertain:
            return True
        if any(not o.ok for o in outcomes):
            return True
        return any(r.degraded for r in results)

    # -- confidence --------------------------------------------------------

    def _confidence(
        self,
        trace: ExecutionTrace,
        prediction: Any,
        results: Sequence[SpecialistResult],
    ) -> ConfidenceBreakdown:
        """The primary step's confidence, calibrated; secondaries in components."""
        primary_task = trace.task
        primary = next(
            (r for r in results if primary_task is not None and r.task is primary_task),
            None,
        )

        components: dict[str, float] = {}
        for result in results:
            components[f"{result.task.value}_confidence"] = float(
                result.confidence.raw
            )

        if primary is None:
            # No primary means the run's own confidence cannot be reported.
            # Reporting a secondary's score here would launder a failure into a
            # number (design §7.5).
            return ConfidenceBreakdown(
                raw=0.0,
                calibrated=None,
                method="uncalibrated",
                components=components,
                degraded=True,
                degradation_reason="primary specialist produced no result",
            )

        breakdown = self.evidence.confidence_for(
            primary,
            extra_components=components,
            degraded=primary.degraded,
        )
        # The run is degraded if anything about it was, even when the primary
        # specialist itself reported a clean result.
        if not breakdown.degraded and prediction is not None:
            if not prediction.above_threshold:
                breakdown = breakdown.model_copy(
                    update={
                        "degraded": True,
                        "degradation_reason": "router confidence below threshold",
                    }
                )
        return breakdown

    # -- verification ------------------------------------------------------

    def _verify(
        self, trace: ExecutionTrace, outcomes: Sequence[StepOutcome]
    ) -> None:
        """Real checks, in the VERIFY state (design §7.4)."""
        successful = [o for o in outcomes if o.ok and o.result is not None]
        if not successful:
            return

        collected = self.evidence.aggregate([o.result for o in successful])
        expected = {o.capability for o in successful}
        observed = set(collected.sources)
        missing = expected - observed
        if missing:
            # A specialist ran and left no trace in the evidence. This is the
            # exact failure the sources bookkeeping exists to catch.
            trace.errors.append(
                {
                    "step_id": "*",
                    "code": "evidence_loss",
                    "message": f"specialists produced no evidence: {sorted(missing)}",
                }
            )

        for outcome in successful:
            if outcome.result is not None and not outcome.result.evidence:
                trace.fallbacks.append(
                    f"{outcome.capability} produced a result with no evidence items"
                )

    # -- contradictions ----------------------------------------------------

    @staticmethod
    def _detect_contradiction(
        trace: ExecutionTrace,
        result: SpecialistResult,
        outcomes: Sequence[StepOutcome],
    ) -> None:
        """Detect and record; never resolve (design §7.6).

        Two steps naming disjoint places in the same coordinate system are a
        contradiction. Both claims are kept; only the flag and a warning are
        added. Cross-system comparison is skipped because IoU across coordinate
        systems is meaningless.
        """
        claims: dict[str, list[Box]] = {}
        for outcome in outcomes:
            if not outcome.ok or outcome.result is None:
                continue
            for box in outcome.result.boxes:
                claims.setdefault(box.coordinate_system.value, []).append(box)

        for system, boxes in claims.items():
            if len(boxes) < 2:
                continue
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if _iou(boxes[i], boxes[j]) < CONTRADICTION_IOU_FLOOR:
                        trace.contradiction = True
                        message = (
                            "specialists produced contradictory spatial claims "
                            f"in {system}; both are retained in evidence"
                        )
                        if message not in result.warnings:
                            result.warnings.append(message)
                        return

    # -- trace helpers -----------------------------------------------------

    def _record(
        self,
        trace: ExecutionTrace,
        state: ControllerState,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Append one `TraceStep`. Observable facts only — no chain-of-thought."""
        trace.steps.append(
            TraceStep(state=state, detail=self._jsonable(detail or {}))
        )

    @staticmethod
    def _jsonable(value: Any) -> Any:
        """Make a detail dict serialisable without dropping information."""
        return _jsonable(value)

    @staticmethod
    def _now() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _new_run_id() -> str:
        import uuid

        return f"run_{uuid.uuid4().hex[:12]}"


def _normalise_modality_overrides(
    overrides: Mapping[str, str] | None,
) -> dict[str, str]:
    """Validate caller-supplied modality declarations, keyed exactly as given.

    `AssetMetadata.modality` defaults to `UNKNOWN`, and `inspect_raster` infers
    the modality from band count. That inference is genuinely ambiguous for a
    1-band scene: a single-band GeoTIFF is a perfectly ordinary SAR product and
    a perfectly ordinary panchromatic optical product, and the band count cannot
    tell them apart. Rather than guess, the caller may declare the modality.

    Values are normalised and validated here, at the boundary where they enter
    the system, so a typo fails immediately and loudly instead of silently
    degrading to "unknown" three layers down. A declaration that is ignored
    without complaint would be worse than a loud error: the caller would believe
    an override took effect when it did not.

    Keys are NOT normalised — they are looked up against the exact asset string
    and against its basename (see `_resolve_assets`), so a caller who passes
    either form gets a match.
    """
    if not overrides:
        return {}
    valid = {m.value for m in Modality}
    normalised: dict[str, str] = {}
    for key, value in overrides.items():
        text = str(value).strip().lower()
        if text not in valid:
            raise ValueError(
                f"asset_modalities[{key!r}] = {value!r} is not a valid modality; "
                f"expected one of {sorted(valid)}"
            )
        normalised[str(key)] = text
    return normalised


def _inspect_asset(path: str, explicit_modality: str | None) -> AssetMetadata:
    """Inspect one asset's header, keeping the raster import lazy.

    `preprocessing.raster.inspect_raster` reads the header only — it does not
    decode pixels — so this stays cheap, and the lazy import keeps the raster
    stack out of the control tier's module-scope imports.
    """
    from preprocessing.raster import inspect_raster

    return inspect_raster(path, explicit_modality=explicit_modality)


def _jsonable(value: Any) -> Any:
    """Recursively make a value JSON-serialisable, stringifying anything exotic.

    `ExecutionTrace.steps[].detail` is a free-form dict, and a plan carries
    tuples and enums that `json.dumps` would reject. Stringifying the leaves
    preserves the fact without inventing structure.
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes. Zero when disjoint."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _asset_label(asset: str) -> str:
    """The client-safe name of an asset, for `ExecutionTrace.inputs`. See F-13.

    `POST /v1/analyze` returns `ResultEnvelope` verbatim, `trace` included, so
    every field of the trace is a value WRITTEN to a client-visible field. When
    the caller is the Space, the request has already been rewritten from asset
    *handles* to filesystem *paths* (`app/space_app.py::analyze`), so echoing
    `request.assets` published the server-side location of each uploaded file.
    Measured on 2026-09-22 inside the sandbox basetemp, the response's
    `trace.inputs` was exactly:

        ["C:\\\\Users\\\\anish\\\\sq_scratch\\\\pt_f13_a\\\\satquery_controller_stub0\\\\a0.tif"]

    The directory is the disclosure. The name is not: on the upload path it is
    the opaque handle the client itself was just given by `/v1/assets`, and on a
    direct-path call it is the file's own name, which the caller supplied.

    `Path(...).name` is used rather than a split on the separator because it is
    the same call `_resolve_assets` already makes when looking a modality
    override up by basename -- so the label and the lookup cannot disagree about
    what an asset is called.

    A trailing separator yields `""`, which is not a path and discloses nothing;
    it is passed through rather than special-cased, because inventing a
    placeholder name would be the trace asserting something about an input it
    could not read. `_inspect_asset` refuses such a path before any result is
    built, so it is not reachable on a successful run.
    """
    return Path(asset).name


__all__ = [
    "CONTRADICTION_IOU_FLOOR",
    "DEFAULT_BUDGET_SECONDS",
    "AnalysisController",
    "StepOutcome",
]
