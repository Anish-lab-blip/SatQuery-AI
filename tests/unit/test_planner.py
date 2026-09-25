"""Tests for `core.planner` — the deterministic policy planner (Phase 15).

The planner is the only component permitted to choose what runs. What is pinned
here is that its decisions are a FUNCTION of the prediction and the request, with
every rule visible and every refusal typed:

  * the §2.3 router-state table, cell by cell;
  * the §2.4 provenance discount — and that it never edits `Intent.confidence`;
  * the §3.5 multi-step rules that recover what the router's single task loses;
  * the §8 closed refusal list;
  * plan ordering, step identity and mode;
  * the module stays importable without torch.

`RouterPrediction` is imported from `router.classifier` in this test module,
which does pull torch — that is fine for the tests. The *planner* must not, and
`test_planner_imports_without_torch` checks that in a subprocess where other
suites' imports cannot mask it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from core.planner import (
    CAPABILITY_ASSETS,
    LEXICAL_FALLBACK_DISCOUNT,
    TASK_CAPABILITY,
    ExecutionPlan,
    PlanMode,
    PlanRefusal,
    PlanStep,
    PolicyPlanner,
)
from core.schemas import AnalysisRequest, Intent, Modality, Task
from core.registry import SpecialistRegistry
from router.classifier import RouterPrediction
from router.label_space import TASK_CLASSES

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ALL_CAPABILITIES = ("vqa", "caption", "grounding", "change", "optical_sar")


class _Registry:
    """Registry double exposing only what the planner reads."""

    def __init__(self, capabilities: tuple[str, ...] = ALL_CAPABILITIES) -> None:
        self._caps = capabilities

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._caps))


class _BrokenRegistry:
    def available(self) -> tuple[str, ...]:
        raise RuntimeError("registry is on fire")


def _prediction(
    task: Task,
    *,
    confidence: float = 0.9,
    source: str = "learned",
    above_threshold: bool = True,
    modality: Modality = Modality.OPTICAL,
    temporal: bool = False,
    spatial_output: bool = False,
    language_output: bool = True,
) -> RouterPrediction:
    intent = Intent(
        task=task,
        modality=modality,
        temporal=temporal,
        spatial_output=spatial_output,
        language_output=language_output,
        confidence=confidence,
        source=source,  # type: ignore[arg-type]
    )
    return RouterPrediction(intent=intent, above_threshold=above_threshold)


def _request(assets: int = 1, force_task: Task | None = None) -> AnalysisRequest:
    return AnalysisRequest(
        assets=[f"/tmp/a{i}.tif" for i in range(assets)],
        query="what is here",
        force_task=force_task,
    )


def _planner(capabilities: tuple[str, ...] = ALL_CAPABILITIES, **kw: Any) -> PolicyPlanner:
    return PolicyPlanner(_Registry(capabilities), **kw)


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------
def test_planner_imports_without_torch() -> None:
    """The planner must import and run with no torch and no router package.

    Checked in a subprocess: the test runner has already imported torch for
    other suites, so an in-process check would pass vacuously.
    """
    code = (
        "import sys, core.planner;"
        "bad=[m for m in sys.modules if m.startswith(('torch','router'))];"
        "print(','.join(bad))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"planner pulled in {result.stdout}"


def test_planner_never_imports_a_specialist() -> None:
    """No plan path may touch the specialists package.

    Read from the planner SOURCE rather than `sys.modules`: other suites in this
    run legitimately import specialists, so an in-process `sys.modules` check
    would be vacuous. The subprocess purity test above covers runtime imports.
    """
    source = (REPO_ROOT / "core" / "planner.py").read_text(encoding="utf-8")
    code_lines = [
        line for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    assert not any("specialists" in line for line in code_lines), (
        f"planner imports a specialist: {code_lines}"
    )


# ---------------------------------------------------------------------------
# The §2.3 router-state table
# ---------------------------------------------------------------------------
def test_confident_known_task_plans_its_specialist() -> None:
    plan = _planner().plan(_prediction(Task.VQA), _request(1))
    assert plan.refused is False
    assert [s.capability for s in plan.steps] == ["vqa"]
    assert plan.uncertain is False


def test_confident_unsupported_task_refuses() -> None:
    plan = _planner().plan(
        _prediction(Task.UNSUPPORTED, language_output=False), _request(1)
    )
    assert plan.refused is True
    assert plan.steps == ()
    assert plan.refusal is not None
    assert plan.refusal.code == "unsupported_query"
    assert plan.refusal.reason == "task_unsupported"


def test_uncertain_known_task_is_planned_but_flagged() -> None:
    """Low confidence is a signal, not a hard gate (design §2.3)."""
    plan = _planner().plan(
        _prediction(Task.VQA, confidence=0.4, above_threshold=False), _request(1)
    )
    assert plan.refused is False
    assert plan.uncertain is True
    assert "vqa" in {s.capability for s in plan.steps}


def test_uncertain_route_adds_an_explanation_step() -> None:
    plan = _planner().plan(
        _prediction(Task.CHANGE, confidence=0.4, above_threshold=False, temporal=True),
        _request(2),
    )
    capabilities = [s.capability for s in plan.steps]
    assert "change" in capabilities
    assert "vqa" in capabilities
    explanation = next(s for s in plan.steps if s.capability == "vqa")
    assert explanation.required is False
    assert explanation.reason == "uncertain_route:explain"


def test_uncertain_route_does_not_duplicate_an_existing_vqa_step() -> None:
    plan = _planner().plan(
        _prediction(Task.VQA, confidence=0.4, above_threshold=False), _request(1)
    )
    assert [s.capability for s in plan.steps] == ["vqa"]


def test_settled_route_adds_no_explanation_step() -> None:
    plan = _planner().plan(_prediction(Task.VQA, confidence=0.95), _request(1))
    assert [s.capability for s in plan.steps] == ["vqa"]


def test_uncertain_unsupported_still_refuses() -> None:
    """UNSUPPORTED is a statement about the query, never about confidence."""
    plan = _planner().plan(
        _prediction(Task.UNSUPPORTED, confidence=0.2, above_threshold=False), _request(1)
    )
    assert plan.refused is True
    assert plan.refusal is not None
    assert plan.refusal.code == "unsupported_query"


def test_zero_assets_refuses_before_anything_else() -> None:
    """The zero-asset refusal is defensive: `AnalysisRequest` already forbids it.

    `AnalysisRequest.assets` is `Field(min_length=1)`, so a caller cannot
    construct a zero-asset request through the public schema. The planner's
    branch is therefore unreachable in normal operation — but the planner is
    also called with hand-built or adapted requests, and a refusal with a typed
    code is the correct response there. Reached via `model_construct`, which
    skips validation, because that is the only way to exercise it.
    """
    request = AnalysisRequest.model_construct(
        assets=[], query="what is here", force_task=None, run_id=None
    )
    plan = _planner().plan(_prediction(Task.VQA), request)
    assert plan.refused is True
    assert plan.refusal is not None
    assert plan.refusal.reason == "zero_assets"
    assert plan.refusal.code == "invalid_request"


def test_zero_asset_request_cannot_be_built_normally() -> None:
    """Confirms the schema guard that makes the branch above defensive."""
    with pytest.raises(Exception):
        AnalysisRequest(assets=[], query="what is here")


@pytest.mark.parametrize(
    "task,assets",
    [
        (Task.VQA, 1),
        (Task.CAPTION, 1),
        (Task.GROUNDING, 1),
        (Task.CHANGE, 2),
        (Task.OPTICAL_SAR, 2),
    ],
)
def test_every_known_task_plans_with_enough_assets(task: Task, assets: int) -> None:
    plan = _planner().plan(_prediction(task), _request(assets))
    assert plan.refused is False
    assert TASK_CAPABILITY[task] in {s.capability for s in plan.steps}


# ---------------------------------------------------------------------------
# §2.4 provenance discount
# ---------------------------------------------------------------------------
def test_lexical_fallback_is_discounted() -> None:
    plan = _planner().plan(
        _prediction(
            Task.VQA, confidence=0.90, source="lexical_fallback", above_threshold=False
        ),
        _request(1),
    )
    assert plan.effective_confidence == pytest.approx(0.90 * LEXICAL_FALLBACK_DISCOUNT)
    assert plan.router_source == "lexical_fallback"


def test_trained_source_is_not_discounted() -> None:
    plan = _planner().plan(
        _prediction(Task.VQA, confidence=0.90, source="learned"), _request(1)
    )
    assert plan.effective_confidence == pytest.approx(0.90)


def test_discount_never_mutates_the_router_measurement() -> None:
    """The discount is the planner's reading; `Intent.confidence` is untouched."""
    prediction = _prediction(
        Task.VQA, confidence=0.90, source="lexical_fallback", above_threshold=False
    )
    _planner().plan(prediction, _request(1))
    assert prediction.intent.confidence == 0.90


def test_forced_source_records_itself() -> None:
    plan = _planner().plan(
        _prediction(Task.VQA), _request(1, force_task=Task.GROUNDING)
    )
    assert plan.router_source == "forced"
    assert "grounding" in {s.capability for s in plan.steps}


def test_forced_task_bypasses_the_router_opinion() -> None:
    """`force_task` wins over the router's predicted task entirely."""
    plan = _planner().plan(
        _prediction(Task.CHANGE, temporal=True), _request(1, force_task=Task.CAPTION)
    )
    assert [s.capability for s in plan.steps] == ["caption"]


def test_forced_unsupported_refuses() -> None:
    plan = _planner().plan(
        _prediction(Task.VQA), _request(1, force_task=Task.UNSUPPORTED)
    )
    assert plan.refused is True
    assert plan.refusal is not None
    assert plan.refusal.reason == "task_unsupported"


# ---------------------------------------------------------------------------
# §3.5 multi-step fan-out
# ---------------------------------------------------------------------------
def test_change_with_language_output_adds_a_caption_step() -> None:
    """The router returns ONE task; the planner recovers the second request."""
    plan = _planner().plan(
        _prediction(Task.CHANGE, temporal=True, language_output=True), _request(2)
    )
    capabilities = [s.capability for s in plan.steps]
    assert capabilities == ["change", "caption"]
    caption = next(s for s in plan.steps if s.capability == "caption")
    assert caption.required is False
    assert caption.reason == "task:change+language_output"
    # The caption describes the later acquisition.
    assert caption.asset_indices == (1,)


def test_change_without_language_output_adds_nothing() -> None:
    plan = _planner().plan(
        _prediction(Task.CHANGE, temporal=True, language_output=False), _request(2)
    )
    assert [s.capability for s in plan.steps] == ["change"]


def test_spatial_and_language_on_one_asset_plans_grounding_and_vqa() -> None:
    plan = _planner().plan(
        _prediction(Task.VQA, spatial_output=True, language_output=True), _request(1)
    )
    capabilities = [s.capability for s in plan.steps]
    assert capabilities == ["vqa", "grounding"]


def test_spatial_and_language_when_task_is_already_grounding_adds_no_duplicate() -> None:
    plan = _planner().plan(
        _prediction(Task.GROUNDING, spatial_output=True, language_output=True),
        _request(1),
    )
    assert [s.capability for s in plan.steps] == ["grounding"]


def test_caption_params_carry_the_task_for_the_shared_vlm_specialist() -> None:
    """The VLM specialist branches on `params["task"]`; a step must carry it."""
    plan = _planner().plan(_prediction(Task.CAPTION), _request(1))
    step = plan.steps[0]
    assert step.params == {"task": "caption"}


def test_vqa_params_carry_the_task() -> None:
    plan = _planner().plan(_prediction(Task.VQA), _request(1))
    assert plan.steps[0].params == {"task": "vqa"}


def test_non_language_capability_carries_no_task_param() -> None:
    plan = _planner().plan(_prediction(Task.GROUNDING), _request(1))
    assert plan.steps[0].params == {}


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------
def test_unknown_capability_is_dropped_and_recorded() -> None:
    """`optical_sar` unregistered means the step is omitted, visibly."""
    plan = _planner(("vqa", "caption", "grounding", "change")).plan(
        _prediction(Task.OPTICAL_SAR, modality=Modality.OPTICAL_SAR), _request(2)
    )
    assert plan.refused is True
    assert plan.refusal is not None
    assert plan.refusal.reason == "no_available_specialist"
    assert plan.refusal.code == "model_unavailable"


def test_dropped_widening_step_leaves_the_primary_plan() -> None:
    plan = _planner(("change",)).plan(
        _prediction(Task.CHANGE, temporal=True, language_output=True), _request(2)
    )
    assert plan.refused is False
    assert [s.capability for s in plan.steps] == ["change"]
    assert any("caption" in n for n in plan.notes)


def test_broken_registry_does_not_raise() -> None:
    """A registry that cannot answer is not a planning error."""
    plan = PolicyPlanner(_BrokenRegistry()).plan(_prediction(Task.VQA), _request(1))
    assert plan.refused is False
    assert [s.capability for s in plan.steps] == ["vqa"]


# ---------------------------------------------------------------------------
# Plan shape: ordering, identity, mode
# ---------------------------------------------------------------------------
def test_step_ids_are_sequential_and_match_order() -> None:
    plan = _planner().plan(
        _prediction(Task.CHANGE, temporal=True, language_output=True), _request(2)
    )
    assert [s.step_id for s in plan.steps] == ["step_001", "step_002"]


def test_step_ids_are_renumbered_after_a_drop() -> None:
    """Ids follow the FINAL order, not the provisional construction order."""
    plan = _planner(("change", "vqa")).plan(
        _prediction(Task.CHANGE, temporal=True, language_output=True), _request(2)
    )
    # `caption` is dropped (unregistered), so `vqa` from the uncertain path and
    # any survivor must still be numbered from 001 with no gap.
    ids = [s.step_id for s in plan.steps]
    assert ids == [f"step_{i:03d}" for i in range(1, len(plan.steps) + 1)]


def test_plan_is_ordered_primary_first() -> None:
    plan = _planner().plan(
        _prediction(Task.CHANGE, temporal=True, language_output=True), _request(2)
    )
    assert plan.steps[0].capability == "change"


def test_capabilities_helper_dedupes_in_order() -> None:
    plan = _planner().plan(
        _prediction(Task.CHANGE, temporal=True, language_output=True), _request(2)
    )
    assert plan.capabilities == ("change", "caption")


def test_overlapping_assets_force_sequential_mode() -> None:
    """A change plan and a caption plan both read asset 1 — not parallel-safe."""
    plan = _planner().plan(
        _prediction(Task.CHANGE, temporal=True, language_output=True), _request(2)
    )
    assert plan.mode is PlanMode.SEQUENTIAL


def test_disjoint_assets_allow_parallel_safe() -> None:
    steps = [
        PlanStep("step_001", "change", "change", (0, 1), {}, "task:change", True),
        PlanStep("step_002", "caption", "caption", (2,), {}, "aspect", False),
    ]
    assert PolicyPlanner._mode_for(steps) is PlanMode.PARALLEL_SAFE


def test_single_step_is_always_sequential() -> None:
    steps = [PlanStep("step_001", "vqa", "vqa", (0,), {}, "task:vqa", True)]
    assert PolicyPlanner._mode_for(steps) is PlanMode.SEQUENTIAL


def test_empty_plan_has_no_mode_pressure() -> None:
    assert PolicyPlanner._mode_for([]) is PlanMode.SEQUENTIAL


def test_max_steps_truncates_and_records() -> None:
    plan = _planner(max_steps=1).plan(
        _prediction(Task.CHANGE, temporal=True, language_output=True), _request(2)
    )
    assert len(plan.steps) == 1
    assert any("truncated" in n for n in plan.notes)


def test_no_max_steps_means_no_truncation() -> None:
    plan = _planner().plan(
        _prediction(Task.CHANGE, temporal=True, language_output=True), _request(2)
    )
    assert not any("truncated" in n for n in plan.notes)


# ---------------------------------------------------------------------------
# Asset indices
# ---------------------------------------------------------------------------
def test_two_asset_capability_takes_the_first_two() -> None:
    plan = _planner().plan(_prediction(Task.CHANGE, temporal=True), _request(2))
    assert plan.steps[0].asset_indices == (0, 1)


def test_one_asset_capability_takes_the_first() -> None:
    plan = _planner().plan(_prediction(Task.GROUNDING), _request(3))
    assert plan.steps[0].asset_indices == (0,)


def test_capability_asset_counts_are_frozen() -> None:
    assert CAPABILITY_ASSETS == {
        "vqa": 1,
        "caption": 1,
        "grounding": 1,
        "change": 2,
        "optical_sar": 2,
        "change_vqa": 2,
    }


# ---------------------------------------------------------------------------
# Trace surface
# ---------------------------------------------------------------------------
def test_plan_trace_has_no_narrative_fields() -> None:
    plan = _planner().plan(_prediction(Task.VQA), _request(1))
    trace = plan.to_trace()
    for forbidden in ("reasoning", "thought", "rationale", "explanation"):
        assert forbidden not in trace


def test_step_reason_is_a_rule_identifier_not_prose() -> None:
    """`reason` names the rule that fired; it is not a narrative field."""
    plan = _planner().plan(
        _prediction(Task.CHANGE, temporal=True, language_output=True), _request(2)
    )
    reasons = {s.reason for s in plan.steps}
    assert reasons == {"task:change", "task:change+language_output"}
    # Rule identifiers are terse and machine-shaped: no sentence punctuation.
    for reason in reasons:
        assert " " not in reason


def test_refusal_trace_is_typed() -> None:
    plan = _planner().plan(_prediction(Task.UNSUPPORTED), _request(1))
    refusal = plan.to_trace()["refusal"]
    assert refusal["code"] == "unsupported_query"
    assert refusal["reason"] == "task_unsupported"
    assert refusal["user_message"]


def test_plan_trace_reports_uncertainty_and_source() -> None:
    plan = _planner().plan(
        _prediction(Task.VQA, confidence=0.3, above_threshold=False,
                    source="lexical_fallback"),
        _request(1),
    )
    trace = plan.to_trace()
    assert trace["uncertain"] is True
    assert trace["router_source"] == "lexical_fallback"


def test_plan_trace_records_every_step_field() -> None:
    plan = _planner().plan(_prediction(Task.VQA), _request(1))
    step = plan.to_trace()["steps"][0]
    assert set(step) == {
        "step_id",
        "capability",
        "specialist",
        "asset_indices",
        "reason",
        "required",
    }


# ---------------------------------------------------------------------------
# Integration with the real registry (no construction)
# ---------------------------------------------------------------------------
def test_real_registry_declaration_drives_availability() -> None:
    class _Cfg:
        device_preference = "cpu"

        def get(self, path: str, default: Any = None) -> Any:
            return default

    registry = SpecialistRegistry.discover(_Cfg())
    plan = PolicyPlanner(registry).plan(
        _prediction(Task.CHANGE, temporal=True, language_output=True), _request(2)
    )
    # R-02 changed this plan, and the change is the point of the assertion: with
    # a `change_vqa` capability declared, a change+language request is satisfied
    # by an ANSWER to the change question rather than by a caption of the later
    # acquisition. Captioning is the fallback for a deployment without it (see
    # `test_change_with_language_output_adds_a_caption_step`, which uses a
    # registry that does not declare `change_vqa`).
    assert [s.capability for s in plan.steps] == ["change", "change_vqa"]
    assert plan.steps[1].reason == "task:change+language_output:answer"
    assert plan.steps[1].asset_indices == (0, 1)
    # Nothing was constructed by planning.
    assert all(e.specialist is None for e in registry.entries())


def test_change_vqa_is_widened_by_the_planner_and_never_predicted_by_the_router() -> None:
    """R-02 / finding F7, stated in one place as the INTENDED contract.

    `change_vqa` is not a routing decision. The router's trained 6-class head
    cannot emit it (`router.label_space.TASK_CLASSES`), and the widening from
    `change` + `language_output` to `change_vqa` is the PLANNER's job, driven
    by the capability registry rather than by a label. Both halves are asserted
    here so a change to either is caught: adding a seventh router class, or
    removing the planner rule.
    """

    class _Cfg:
        device_preference = "cpu"

        def get(self, path: str, default: Any = None) -> Any:
            return default

    # Half 1 — the router's frozen space excludes it, so nothing can predict it.
    assert "change_vqa" not in TASK_CLASSES
    assert Task.CHANGE_VQA.value == "change_vqa"

    # Half 2 — the planner is what produces it, from `change` + language.
    registry = SpecialistRegistry.discover(_Cfg())
    plan = PolicyPlanner(registry).plan(
        _prediction(Task.CHANGE, temporal=True, language_output=True), _request(2)
    )
    assert [s.capability for s in plan.steps] == ["change", "change_vqa"]

    # And with no `change_vqa` capability registered, the caption fallback
    # returns — proving the widening is capability-driven, not task-driven.
    without = _planner(capabilities=("change", "caption", "vqa"))
    fallback = without.plan(
        _prediction(Task.CHANGE, temporal=True, language_output=True), _request(2)
    )
    assert [s.capability for s in fallback.steps] == ["change", "caption"]
