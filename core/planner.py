"""SatQuery AI — the deterministic policy planner (Phase 15).

WHAT THIS IS FOR
----------------
`docs/ARCHITECTURE_FREEZE.md` section 5 assigns each layer one verb:

    router *understands*; policy engine *decides*; specialists *compute*;
    VLM *explains*; evidence engine *proves*.

This module is the second verb. Its rule, restated because everything here
follows from it:

    **The planner is the only component permitted to choose what runs. Its
    input is a `RouterPrediction`; its output is an `ExecutionPlan`. It may
    select zero, one, or several steps, and it may refuse.**

The router's `Intent` is an *input to a decision*, never the decision.
`core/schemas.py:89` already says so: *"Output of the learned router. Advisory
only — the controller decides."*

WHY THE TWO ARE SEPARATE, CONCRETELY
------------------------------------
A design where `intent.task` selects a specialist in one step has collapsed
understand into decide. Three things break when that happens:

1. The router can never be overruled, downgraded, or widened. A
   confident-but-wrong prediction becomes unappealable.
2. A query needing two specialists can never get both. The router returns ONE
   task; *"what changed between these two images, and describe the scene"*
   routes to `change` alone, and the caption is lost. Section §3.5 is what
   recovers it.
3. There is no record of the decision. A plan step that carries a `reason`
   naming the rule that produced it is auditable; an `argmax` over a softmax is
   not a policy and cannot be inspected or tested.

THE THRESHOLD IS APPLIED ONCE, IN THE ROUTER
--------------------------------------------
`router/classifier.py:311-339` already consumes `router.confidence_threshold`
as a router-internal fallback trigger, and `route()`'s docstring is explicit
that *"the router never refuses to answer; `above_threshold` carries the
uncertainty."*

So the planner consumes `above_threshold` (a bool), never the numeric
threshold. Re-reading `router.confidence_threshold` here would be a second,
independent gate over the same quantity — two places bound to one knob, which
is how a threshold ends up meaning two different things.

PROVENANCE IS PART OF THE EVIDENCE
----------------------------------
`IntentRouter.adapter_source` returns `'trained'` or `'lexical_fallback'`, and
the router's own docstring warns that a caller who never passes `adapter_path`
gets fallback answers *"while believing the trained model is serving"*. The
distinguishability is not obvious from the confidence alone: on the spec
section 29 examples the fallback returns 0.850-0.920 against the trained
model's 0.780-1.000 — the fallback can be MORE confident.

A lexical fallback at 0.9 is not the same evidence as a learned model at 0.9:
one is a regex that matched, the other is a learned distribution. Treating them
identically would let a matched keyword outrank the model it fell back from. So
the planner applies a **provenance discount** — not a second numeric gate — to
its own reading of the confidence, and never edits `Intent.confidence` itself.
A measurement is the router's to report; the discount is the planner's reading.

THE PLAN IS ORDERED; IT IS NEITHER A DAG NOR A SET
--------------------------------------------------
Not a DAG (§3.2): no specialist consumes another's result, so the graph is a
provable flat fan-out converging on one evidence engine. A topological sort,
cycle detection and edge semantics would be complexity without a represented
relationship. The migration, if a chaining specialist ever arrives, is additive.

Not a set (§3.3): a set has no order, and the trace must be reproducible —
*"step_001 produced the grounding box"* requires step identity and a defined
ordering. Order is observable information.

REFUSAL IS A SUCCESSFUL RUN WITH NO STEPS
-----------------------------------------
Refusing is not an error. It returns a valid plan with `refused=True` and a
typed `PlanRefusal`, so the controller can assemble a normal envelope carrying
the reason. The refusal conditions are a closed list (§8) — no free-form logic,
because a refusal rule invented at the call site is a policy that cannot be
tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping

from core.errors import InvalidRequestError, UnsupportedQueryError
from core.schemas import AnalysisRequest, AssetMetadata, Task

if TYPE_CHECKING:  # pragma: no cover - typing only, never executed
    # Imported for annotation only. `router.classifier` imports torch at module
    # scope, and this module must stay importable and testable without torch —
    # the planner is pure, and design §9.3 leans on that ("the planner is pure
    # ... no models, no torch, no filesystem"). The only surface the planner
    # actually uses is `prediction.intent` and `prediction.above_threshold`,
    # so the dependency is structural and need not be a runtime import.
    from router.classifier import RouterPrediction

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

#: Multiplier the planner applies to its *reading* of a lexical-fallback
#: confidence. Frozen by design §2.4. It exists so a matched keyword cannot
#: outrank a learned model when the planner decides whether to widen a plan.
#: The router's own reported confidence is never altered.
LEXICAL_FALLBACK_DISCOUNT: float = 0.75

#: Above this effective confidence the planner treats the route as settled and
#: does not add the discretionary explanation step. Below it, and when the
#: request also asks for language output, a `vlm` step is added so the run
#: explains itself. Also frozen by §2.3.
SETTLED_CONFIDENCE: float = 0.60

#: Capability for each router task label. The label space is the router's
#: ontology; this is the planner's mapping from it to a specialist capability.
#: `UNSUPPORTED` is deliberately absent — it is a refusal, not a capability.
TASK_CAPABILITY: Mapping[Task, str] = {
    Task.VQA: "vqa",
    Task.CAPTION: "caption",
    Task.GROUNDING: "grounding",
    Task.CHANGE: "change",
    Task.OPTICAL_SAR: "optical_sar",
    Task.CHANGE_VQA: "change_vqa",
}

#: Asset count each capability requires. Mirrors the specialists' own
#: `validate_request`, which stays authoritative; this is the planner's cheap
#: precondition so it can refuse before construction is attempted.
CAPABILITY_ASSETS: Mapping[str, int] = {
    "vqa": 1,
    "caption": 1,
    "grounding": 1,
    "change": 2,
    "optical_sar": 2,
    # Two assets, like `change` — the difference is the output, not the input:
    # a short language answer rather than a change map.
    "change_vqa": 2,
}


# ---------------------------------------------------------------------------
# Plan representation
# ---------------------------------------------------------------------------
class PlanMode(str, Enum):
    """How the steps may be executed.

    `PARALLEL_SAFE` is a *declaration* that the steps share no mutable state,
    not a concurrency directive. v1 executes both modes sequentially — freeze
    section 5 forbids worker pools, queues and async frameworks. The marker
    records that a future `ThreadPoolExecutor` would be sound.
    """

    SEQUENTIAL = "sequential"
    PARALLEL_SAFE = "parallel_safe"


@dataclass(frozen=True)
class PlanStep:
    """One unit of work, with the rule that produced it.

    Attributes:
        step_id: stable, zero-padded, assigned by the planner in order.
        capability: the registry key to resolve, e.g. `"vqa"`.
        specialist: the registry key the capability resolves through. Kept
            distinct from `capability` because the VQA specialist answers to
            `"vqa"` AND `"caption"` while its own `name` is `"vlm"`.
        asset_indices: which request assets this step consumes.
        params: extra kwargs for the specialist, e.g. `{"task": "vqa"}`.
        reason: a machine-generated RULE IDENTIFIER from a closed set, not a
            narrative. A free-text field here would become a place to put
            chain-of-thought, and eventually would contain some (design §6).
        required: False means a failure degrades the run rather than
            invalidating it.
    """

    step_id: str
    capability: str
    specialist: str
    asset_indices: tuple[int, ...]
    params: Mapping[str, Any] = field(default_factory=dict)
    reason: str = "unspecified"
    required: bool = True

    def to_trace(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "capability": self.capability,
            "specialist": self.specialist,
            "asset_indices": list(self.asset_indices),
            "reason": self.reason,
            "required": self.required,
        }


@dataclass(frozen=True)
class PlanRefusal:
    """Why a plan has no steps. A refusal is not an error."""

    code: str
    reason: str
    user_message: str

    def to_trace(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "reason": self.reason,
            "user_message": self.user_message,
        }


@dataclass(frozen=True)
class ExecutionPlan:
    """An ordered set of steps, or a refusal. Never both."""

    steps: tuple[PlanStep, ...] = ()
    mode: PlanMode = PlanMode.SEQUENTIAL
    refused: bool = False
    refusal: PlanRefusal | None = None
    uncertain: bool = False
    router_source: str = "lexical_fallback"
    effective_confidence: float = 0.0
    notes: tuple[str, ...] = ()

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Distinct capabilities in plan order. Deterministic, deduped."""
        seen: list[str] = []
        for step in self.steps:
            if step.capability not in seen:
                seen.append(step.capability)
        return tuple(seen)

    def to_trace(self) -> dict[str, Any]:
        return {
            "steps": [s.to_trace() for s in self.steps],
            "mode": self.mode.value,
            "refused": self.refused,
            "refusal": self.refusal.to_trace() if self.refusal else None,
            "uncertain": self.uncertain,
            "router_source": self.router_source,
            "effective_confidence": round(self.effective_confidence, 4),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
class PolicyPlanner:
    """Turns a router prediction into a deterministic execution plan.

    Pure. Takes dataclasses, returns a dataclass. No models, no torch, no
    filesystem, no clock — every rule below is unit-testable with a hand-built
    `RouterPrediction`.

    Args:
        registry: the `SpecialistRegistry`, consulted for which capabilities
            exist. Only its declared capability list is read, so passing a
            registry built with `discover()` (which constructs nothing) is the
            intended usage.
        max_steps: cap on the number of planned steps, mirroring
            `agent.max_specialists`. Truncation is recorded in `notes`, never
            silent.
    """

    def __init__(self, registry: Any, *, max_steps: int | None = None) -> None:
        self.registry = registry
        self.max_steps = max_steps

    # -- public API --------------------------------------------------------

    def plan(
        self,
        prediction: "RouterPrediction",
        request: AnalysisRequest,
        *,
        assets: list[AssetMetadata] | None = None,
    ) -> ExecutionPlan:
        """The single decision point. See the module docstring.

        Args:
            prediction: the router's advisory claim.
            request: the original request, for `force_task` and asset count.
            assets: resolved asset metadata, when the caller has it. Not read
                for policy — only the COUNT matters here, and that comes from
                `request.assets`. Accepted so the controller can pass its
                already-resolved list without a second conversion.

        Returns:
            An `ExecutionPlan`. Refusal is expressed in the plan, not raised.
        """
        asset_count = len(request.assets)
        source = prediction.intent.source

        # --- force_task bypasses the router entirely (§8) -----------------
        # `Intent(source="forced")` already exists in the schema. When the
        # caller forces a task, the router's opinion is not consulted at all,
        # and that fact is recorded rather than hidden.
        forced = request.force_task
        task = forced if forced is not None else prediction.intent.task
        if forced is not None:
            source = "forced"

        # --- refusal conditions: a closed list (§8) -----------------------
        refusal = self._refusal_for(task, asset_count)
        if refusal is not None:
            return ExecutionPlan(
                steps=(),
                refused=True,
                refusal=refusal,
                uncertain=not prediction.above_threshold,
                router_source=source,
                effective_confidence=self._effective_confidence(prediction),
                notes=(f"refused:{refusal.reason}",),
            )

        # --- build the steps ----------------------------------------------
        steps: list[PlanStep] = []
        notes: list[str] = []

        primary = TASK_CAPABILITY[task]
        self._append(
            steps,
            capability=primary,
            indices=self._indices_for(primary, asset_count),
            reason=f"task:{task.value}",
            required=True,
            params=self._params_for(primary, task),
        )

        # --- §3.5 multi-step fan-out --------------------------------------
        steps.extend(self._widening_steps(prediction, task, asset_count, notes))

        # --- discretionary explanation on an uncertain route (§2.3) -------
        if (
            not prediction.above_threshold
            and prediction.intent.language_output
            and "vqa" not in {s.capability for s in steps}
            and self._capability_known("vqa")
        ):
            self._append(
                steps,
                capability="vqa",
                indices=(0,),
                reason="uncertain_route:explain",
                required=False,
            )
            notes.append("uncertain route: added an explanation step")

        # --- availability gate (§8, table row 3) --------------------------
        steps, avail_notes = self._drop_unavailable(steps)
        notes.extend(avail_notes)

        if not steps:
            return ExecutionPlan(
                steps=(),
                refused=True,
                refusal=self._refusal(
                    "model_unavailable",
                    "no_available_specialist",
                    "No specialist is available for this request in this "
                    "environment.",
                ),
                uncertain=not prediction.above_threshold,
                router_source=source,
                effective_confidence=self._effective_confidence(prediction),
                notes=tuple(notes),
            )

        # --- cap, recording any truncation ---------------------------------
        if self.max_steps is not None and len(steps) > self.max_steps:
            dropped = [s.capability for s in steps[self.max_steps:]]
            notes.append(
                f"plan truncated at {self.max_steps} steps; dropped {dropped}"
            )
            steps = steps[: self.max_steps]

        steps = self._renumber(steps)

        return ExecutionPlan(
            steps=tuple(steps),
            mode=self._mode_for(steps),
            refused=False,
            refusal=None,
            uncertain=not prediction.above_threshold,
            router_source=source,
            effective_confidence=self._effective_confidence(prediction),
            notes=tuple(notes),
        )

    # -- confidence --------------------------------------------------------

    @staticmethod
    def _effective_confidence(prediction: "RouterPrediction") -> float:
        """The planner's reading of the router's confidence.

        Applies the provenance discount for a lexical fallback (§2.4). The
        router's own `Intent.confidence` is never modified — this is a separate
        number used only for planning.
        """
        raw = float(prediction.intent.confidence)
        if prediction.intent.source == "lexical_fallback":
            return raw * LEXICAL_FALLBACK_DISCOUNT
        return raw

    # -- refusals ----------------------------------------------------------

    def _refusal_for(self, task: Task, asset_count: int) -> PlanRefusal | None:
        """The closed refusal list from §8. Order matters: query first."""
        if task is Task.UNSUPPORTED:
            return self._refusal(
                UnsupportedQueryError.code,
                "task_unsupported",
                UnsupportedQueryError.user_message,
            )
        if asset_count < 1:
            return self._refusal(
                InvalidRequestError.code,
                "zero_assets",
                InvalidRequestError.user_message,
            )
        return None

    @staticmethod
    def _refusal(code: str, reason: str, user_message: str) -> PlanRefusal:
        return PlanRefusal(code=code, reason=reason, user_message=user_message)

    # -- step construction -------------------------------------------------

    def _widening_steps(
        self,
        prediction: "RouterPrediction",
        task: Task,
        asset_count: int,
        notes: list[str],
    ) -> list[PlanStep]:
        """The §3.5 rules that turn ONE router task into SEVERAL steps.

        These are the reason the planner exists rather than a switch on
        `intent.task`. The router's binary heads describe *aspects* of the
        request that a single task label cannot carry.
        """
        extra: list[PlanStep] = []
        intent = prediction.intent

        # Availability is NOT gated here. The step is added whenever its rule
        # fires, and `_drop_unavailable` removes it with a recorded note if the
        # capability is unregistered. Gating here as well would create a second,
        # silent availability check whose omission leaves no trace — exactly the
        # "absence indistinguishable from a non-event" failure that §4.3 forbids.
        #
        # A change task on a two-asset request also satisfies a language-output
        # request, because the user asked two things: what changed, and (per
        # the language head) wants language about it.
        #
        # WHICH language output, though, depends on what the deployment has.
        # With a change-VQA capability registered (R-02), the request is
        # satisfied by an ANSWER to the change question — which is the specific
        # thing that was asked. Without it, the fallback is to caption the later
        # acquisition, which is the one a "what changed" answer describes.
        #
        # These are alternatives, not two things to do: captioning the post
        # image is not a second requirement, it is the best available substitute
        # when the answer capability is absent. Adding both would spend a step
        # on a strictly weaker output.
        if task is Task.CHANGE and asset_count >= 2 and intent.language_output:
            if self._capability_known("change_vqa"):
                if not self._capability_planned(extra, "change_vqa"):
                    self._append(
                        extra,
                        capability="change_vqa",
                        indices=self._indices_for("change_vqa", asset_count),
                        reason="task:change+language_output:answer",
                        required=False,
                    )
                    notes.append(
                        "change + language request: added a change-VQA step"
                    )
            elif not self._capability_planned(extra, "caption"):
                self._append(
                    extra,
                    capability="caption",
                    indices=(asset_count - 1,),
                    reason="task:change+language_output",
                    required=False,
                )
                notes.append("change + language request: added a caption step")

        # A request that wants BOTH coordinates and prose on one asset gets
        # grounding and vqa. The router can only return one task, so without
        # this rule the second aspect is silently lost.
        if intent.spatial_output and intent.language_output and asset_count >= 1:
            if task is not Task.GROUNDING:
                if not self._capability_planned(extra, "grounding"):
                    self._append(
                        extra,
                        capability="grounding",
                        indices=(0,),
                        reason="aspect:spatial_output",
                        required=False,
                    )
                    notes.append("spatial+language request: added a grounding step")

        return extra

    @staticmethod
    def _capability_planned(steps: list[PlanStep], capability: str) -> bool:
        return any(step.capability == capability for step in steps)

    @staticmethod
    def _params_for(capability: str, task: Task) -> dict[str, Any]:
        """Specialist kwargs a step must carry.

        The VLM specialist branches on `params["task"]` to decide between VQA
        and captioning, so a `vlm`-backed step without it is unrunnable. The
        mapping is by CAPABILITY, not by specialist, so `caption` passes
        `"task": "caption"` even though the same object serves both.
        """
        if capability in ("vqa", "caption"):
            return {"task": capability}
        return {}

    @staticmethod
    def _indices_for(capability: str, asset_count: int) -> tuple[int, ...]:
        """Which assets a capability consumes.

        A two-asset capability takes the first two in request order; a
        one-asset capability takes the first. Deliberately simple: the
        specialists validate their own asset count and pairing rules, and
        duplicating that logic here would give two places to disagree.
        """
        needed = CAPABILITY_ASSETS.get(capability, 1)
        return tuple(range(min(needed, asset_count)))

    @staticmethod
    def _append(
        steps: list[PlanStep],
        *,
        capability: str,
        indices: tuple[int, ...],
        reason: str,
        required: bool,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        steps.append(
            PlanStep(
                # Provisional id; `_renumber` assigns the canonical one after
                # widening, availability filtering and any truncation, so the
                # ids always match the final order.
                step_id=f"step_{len(steps) + 1:03d}",
                capability=capability,
                specialist=capability,
                asset_indices=indices,
                params=params or {},
                reason=reason,
                required=required,
            )
        )

    # -- availability ------------------------------------------------------

    def _capability_known(self, capability: str) -> bool:
        try:
            return capability in set(self.registry.available())
        except Exception:  # noqa: BLE001 - a broken registry is not a plan error
            return False

    def _drop_unavailable(
        self, steps: list[PlanStep]
    ) -> tuple[list[PlanStep], list[str]]:
        """Remove steps whose capability the registry does not declare.

        An unknown capability means the specialist is not registered at all,
        which is a fact about the deployment, not about the query. The
        omission is recorded in `notes` so "not planned because it cannot run"
        stays distinguishable from "not planned because the planner chose not
        to" (design §4.3).

        Note this filters only on *declaration*. Whether a declared capability
        can actually be CONSTRUCTED is discovered by the controller, which
        records an `UNAVAILABLE` registry entry — the planner must not attempt
        construction, which would defeat the lazy-load design.
        """
        known = {c for c in self.registry.available()} if self._registry_ok() else None
        if known is None:
            return steps, []

        kept: list[PlanStep] = []
        notes: list[str] = []
        for step in steps:
            if step.capability in known:
                kept.append(step)
            else:
                notes.append(
                    f"dropped step {step.step_id} ({step.capability}): "
                    f"capability not registered"
                )
        return kept, notes

    def _registry_ok(self) -> bool:
        try:
            self.registry.available()
        except Exception:  # noqa: BLE001
            return False
        return True

    # -- mode and numbering ------------------------------------------------

    @staticmethod
    def _mode_for(steps: list[PlanStep]) -> PlanMode:
        """`PARALLEL_SAFE` only when the asset partitions are demonstrably safe.

        Requires at least two steps, pairwise-disjoint asset indices, and no
        step declaring a resource constraint. With today's four specialists
        asset indices commonly overlap — a change plan and a caption plan can
        both read asset 1 — so this legitimately returns SEQUENTIAL in normal
        operation. That is correct, not a limitation (§3.4).
        """
        if len(steps) < 2:
            return PlanMode.SEQUENTIAL

        seen: set[int] = set()
        for step in steps:
            indices = set(step.asset_indices)
            if indices & seen:
                return PlanMode.SEQUENTIAL
            seen |= indices
        return PlanMode.PARALLEL_SAFE

    @staticmethod
    def _renumber(steps: list[PlanStep]) -> list[PlanStep]:
        """Assign canonical `step_NNN` ids matching the final order."""
        return [
            PlanStep(
                step_id=f"step_{index:03d}",
                capability=step.capability,
                specialist=step.specialist,
                asset_indices=step.asset_indices,
                params=step.params,
                reason=step.reason,
                required=step.required,
            )
            for index, step in enumerate(steps, start=1)
        ]


__all__ = [
    "CAPABILITY_ASSETS",
    "LEXICAL_FALLBACK_DISCOUNT",
    "SETTLED_CONFIDENCE",
    "TASK_CAPABILITY",
    "ExecutionPlan",
    "PlanMode",
    "PlanRefusal",
    "PlanStep",
    "PolicyPlanner",
]
