"""R-02 test area R — the change-VQA specialist and its planner/registry wiring.

    R  specialist contract, degraded semantics, skew refusal, planner gating

THE TWO BEHAVIOURS THIS AREA EXISTS TO PIN
------------------------------------------
1. **Degraded mode answers NOTHING.** `ChangeSpecialist` degrades to an untrained
   detector and still emits a change map, because a random map is visibly noise.
   This specialist must not copy that: an untrained 19-way classifier still
   emits a fluent, confident-looking `yes`, and a reader cannot tell it from a
   real answer. So the untrained case returns `answer=""`.

2. **A feature-spec mismatch refuses to answer.** `configs/base.yaml` carries no
   `change.checkpoint_path`, so serving builds an UNTRAINED STANet while training
   used the trained LEVIR checkpoint. Answering anyway would compute the answer
   from a representation the head was never fitted on -- the same failure as (1),
   so it gets the same treatment.

Both are tested by being driven, not by reading the code: a gate that has never
been shown to fire is not a gate.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from core.errors import InvalidRequestError
from core.planner import CAPABILITY_ASSETS, TASK_CAPABILITY, PolicyPlanner
from core.registry import SpecialistRegistry, default_specs
from core.schemas import AnalysisRequest, AssetMetadata, Intent, Task
from router.classifier import RouterPrediction
from specialists.base import SpecialistRequest
from specialists.change.vqa_specialist import (
    LOW_CONFIDENCE_THRESHOLD,
    TEMPORAL_ORDER_NOTE,
    ChangeVQASpecialist,
)
from training.change_vqa import model as M
from training.change_vqa import vocab as V
from training.change_vqa.features import (
    CHANGE_FEATURE_DIM,
    TEXT_FEATURE_DIM,
    feature_spec_hash,
)

TRAINED_SPEC = "c801326f85a185f8"
UNTRAINED_SPEC = "714efac5b0e6a6d1"

ALL_CAPABILITIES = ("vqa", "caption", "grounding", "change", "optical_sar")


@pytest.fixture
def scratch() -> Path:
    """A self-managed scratch directory; see the note in the dataset test module.

    `tmp_path` performs directory removals, which a sandbox delete guard has
    blocked in this repository.
    """
    return Path(tempfile.mkdtemp(prefix="sq_r02_integration_"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _Registry:
    """Registry double exposing only what the planner reads."""

    def __init__(self, capabilities: tuple[str, ...] = ALL_CAPABILITIES) -> None:
        self._caps = capabilities

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._caps))


def _prediction(
    task: Task,
    *,
    temporal: bool = False,
    language_output: bool = True,
) -> RouterPrediction:
    return RouterPrediction(
        intent=Intent(
            task=task,
            temporal=temporal,
            language_output=language_output,
            confidence=0.9,
            source="learned",
        ),
        above_threshold=True,
    )


def _request(assets: int = 2) -> AnalysisRequest:
    return AnalysisRequest(assets=[f"/tmp/a{i}.tif" for i in range(assets)], query="q")


def _planner(capabilities: tuple[str, ...] = ALL_CAPABILITIES) -> PolicyPlanner:
    return PolicyPlanner(_Registry(capabilities))


def _png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path)
    return path


class _StubFeatureExtractor:
    """Produces a fixed change feature and declares a spec hash."""

    def __init__(self, spec_hash: str, *, trained: bool = True) -> None:
        self.spec_hash = spec_hash
        self.trained = trained

    def extract(self, t1, t2):
        return np.zeros(CHANGE_FEATURE_DIM, dtype=np.float32), {"stub": True}


class _StubTextEncoder:
    def encode(self, texts):
        return np.zeros((len(list(texts)), TEXT_FEATURE_DIM), dtype=np.float32)


def _specialist_request(
    scratch: Path, query: str = "What is the largest change?"
) -> SpecialistRequest:
    """`SpecialistRequest` is frozen, so the query is supplied at construction."""
    return SpecialistRequest(
        assets=[
            AssetMetadata(path=str(_png(scratch / "im1" / "00001.png"))),
            AssetMetadata(path=str(_png(scratch / "im2" / "00001.png"))),
        ],
        query=query,
    )


def _trained_head() -> Any:
    head = M.build_change_vqa_head(
        change_feature_dim=CHANGE_FEATURE_DIM, text_feature_dim=TEXT_FEATURE_DIM
    )
    head.eval()
    head._satquery_trained = True  # type: ignore[attr-defined]
    return head


# ===========================================================================
# Area R - request validation
# ===========================================================================
def test_r_exactly_two_assets_are_required(scratch: Path) -> None:
    specialist = ChangeVQASpecialist()
    for count in (1, 3):
        request = SpecialistRequest(
            assets=[
                AssetMetadata(path=str(_png(scratch / f"a{i}.png")))
                for i in range(count)
            ],
            query="did buildings change?",
        )
        with pytest.raises(InvalidRequestError, match="exactly 2 assets"):
            specialist.validate_request(request)


def test_r_the_one_asset_error_explains_the_real_mistake(scratch: Path) -> None:
    """Treating a change question as ordinary VQA about a single image is the
    common caller error, so the message names it."""
    specialist = ChangeVQASpecialist()
    request = SpecialistRequest(
        assets=[AssetMetadata(path=str(_png(scratch / "a.png")))],
        query="did buildings change?",
    )
    with pytest.raises(InvalidRequestError) as excinfo:
        specialist.validate_request(request)
    assert "earlier one" in str(excinfo.value.user_message)


def test_r_a_missing_asset_path_is_rejected(scratch: Path) -> None:
    specialist = ChangeVQASpecialist()
    request = SpecialistRequest(
        assets=[
            AssetMetadata(path=str(_png(scratch / "a.png"))),
            AssetMetadata(path=str(scratch / "absent.png")),
        ],
        query="q",
    )
    with pytest.raises(InvalidRequestError, match="does not exist"):
        specialist.validate_request(request)


# ===========================================================================
# Area R - degraded mode produces no answer
# ===========================================================================
def test_r_without_a_head_the_specialist_reports_which_piece_is_missing() -> None:
    specialist = ChangeVQASpecialist(
        head=None,
        feature_extractor=_StubFeatureExtractor(TRAINED_SPEC),
        text_encoder=_StubTextEncoder(),
        head_path="/artifacts/change_vqa/head.pt",
    )
    assert specialist.has_head is False
    reason = specialist.unavailable_reason()
    assert reason is not None
    assert "trained change-VQA head" in reason
    assert "/artifacts/change_vqa/head.pt" in reason


def test_r_without_a_head_the_answer_is_empty_and_degraded(scratch: Path) -> None:
    """THE property of area R. An untrained head would emit a fluent `yes`."""
    specialist = ChangeVQASpecialist(
        head=None,
        feature_extractor=_StubFeatureExtractor(TRAINED_SPEC),
        text_encoder=_StubTextEncoder(),
    )
    result = specialist.execute(_specialist_request(scratch))

    assert result.task is Task.CHANGE_VQA
    assert result.answer == ""
    assert result.labels == []
    assert result.degraded is True
    assert result.confidence.method == "unavailable"
    assert result.confidence.calibrated is None
    assert result.evidence == []
    assert any("No answer was produced" in w for w in result.warnings)
    assert TEMPORAL_ORDER_NOTE in result.warnings


def test_r_an_untrained_head_object_still_counts_as_no_head() -> None:
    """`has_head` keys on the training flag, not on the object being non-None."""
    untrained = M.build_change_vqa_head(
        change_feature_dim=CHANGE_FEATURE_DIM, text_feature_dim=TEXT_FEATURE_DIM
    )
    assert not getattr(untrained, "_satquery_trained", False)
    specialist = ChangeVQASpecialist(
        head=untrained,
        feature_extractor=_StubFeatureExtractor(TRAINED_SPEC),
        text_encoder=_StubTextEncoder(),
    )
    assert specialist.has_head is False


def test_r_a_missing_feature_extractor_or_text_encoder_is_named() -> None:
    specialist = ChangeVQASpecialist(head=None)
    reason = specialist.unavailable_reason()
    assert reason is not None
    assert "change feature extractor" in reason
    assert "question text encoder" in reason


# ===========================================================================
# Area R - the train/serve skew refusal
# ===========================================================================
def test_r_a_matching_feature_spec_is_not_a_mismatch() -> None:
    specialist = ChangeVQASpecialist(
        head=_trained_head(),
        feature_extractor=_StubFeatureExtractor(TRAINED_SPEC),
        text_encoder=_StubTextEncoder(),
        head_metadata={"change_cache_spec": TRAINED_SPEC, "detector_trained": True},
    )
    assert specialist.feature_spec_mismatch() is None
    assert specialist.unavailable_reason() is None


def test_r_a_trained_head_on_an_untrained_representation_refuses_to_answer(
    scratch: Path,
) -> None:
    """The exact deployment skew: the head was fitted on trained-STANet features
    but serving would produce untrained ones."""
    specialist = ChangeVQASpecialist(
        head=_trained_head(),
        feature_extractor=_StubFeatureExtractor(UNTRAINED_SPEC, trained=False),
        text_encoder=_StubTextEncoder(),
        head_metadata={"change_cache_spec": TRAINED_SPEC, "detector_trained": True},
    )
    mismatch = specialist.feature_spec_mismatch()
    assert mismatch is not None
    # Both specs are named, so the reader can see which side is which.
    assert TRAINED_SPEC in mismatch
    assert UNTRAINED_SPEC in mismatch

    result = specialist.execute(_specialist_request(scratch))
    assert result.answer == ""
    assert result.degraded is True
    assert result.confidence.method == "unavailable"


def test_r_an_unknown_trained_spec_does_not_block_answering() -> None:
    """Absent metadata is not evidence of a mismatch; refusing on it would make
    the guard fire on every head written before the metadata existed."""
    specialist = ChangeVQASpecialist(
        head=_trained_head(),
        feature_extractor=_StubFeatureExtractor(TRAINED_SPEC),
        text_encoder=_StubTextEncoder(),
        head_metadata={},
    )
    assert specialist.feature_spec_mismatch() is None


# ===========================================================================
# Area R - a servable specialist answers with evidence and raw confidence
# ===========================================================================
def test_r_a_servable_specialist_answers_with_evidence(scratch: Path, monkeypatch) -> None:
    """Drives `execute` end to end with stubbed encoders, so the wiring between
    the specialist, `predict_answers` and the result schema is exercised without
    loading STANet or MiniLM."""
    monkeypatch.setattr(
        "preprocessing.imagery.load_image_array",
        lambda _p: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    specialist = ChangeVQASpecialist(
        head=_trained_head(),
        feature_extractor=_StubFeatureExtractor(TRAINED_SPEC),
        text_encoder=_StubTextEncoder(),
        head_metadata={"change_cache_spec": TRAINED_SPEC, "detector_trained": True},
    )
    result = specialist.execute(_specialist_request(scratch))

    assert result.task is Task.CHANGE_VQA
    assert result.answer in V.ANSWER_VOCABULARY
    assert result.degraded is False
    # Raw softmax, explicitly not calibrated.
    assert result.confidence.method == "uncalibrated"
    assert result.confidence.calibrated is None
    assert 0.0 <= result.confidence.value <= 1.0
    assert set(result.confidence.components) == {
        "top1_minus_top2",
        "top1_probability",
        "top2_probability",
        "n_answers_considered",
        "entropy_normalised",
    }

    kinds = [e.payload.get("kind") for e in result.evidence]
    assert kinds == ["answer_distribution", "class_wise_change_estimate"]
    estimate = result.evidence[1].payload
    assert estimate["class_order"] == list(V.CHANGE_CLASS_ORDER)
    assert len(estimate["class_change_magnitude"]) == V.N_CHANGE_CLASSES
    assert estimate["estimator_is_learned"] is True
    assert estimate["temporal_order"] == TEMPORAL_ORDER_NOTE


def test_r_the_largest_change_question_is_type_masked(scratch: Path, monkeypatch) -> None:
    """`largest_change` may only be answered with a class name."""
    monkeypatch.setattr(
        "preprocessing.imagery.load_image_array",
        lambda _p: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    specialist = ChangeVQASpecialist(
        head=_trained_head(),
        feature_extractor=_StubFeatureExtractor(TRAINED_SPEC),
        text_encoder=_StubTextEncoder(),
        head_metadata={"change_cache_spec": TRAINED_SPEC, "detector_trained": True},
    )
    result = specialist.execute(_specialist_request(scratch))
    assert result.answer in V.LEGAL_ANSWERS["largest_change"]
    assert result.evidence[0].payload["type_mask_applied"] is True


def test_r_an_unresolved_question_type_is_warned_about(scratch: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "preprocessing.imagery.load_image_array",
        lambda _p: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    specialist = ChangeVQASpecialist(
        head=_trained_head(),
        feature_extractor=_StubFeatureExtractor(TRAINED_SPEC),
        text_encoder=_StubTextEncoder(),
        head_metadata={"change_cache_spec": TRAINED_SPEC, "detector_trained": True},
    )
    request = _specialist_request(scratch, query="How many cars are parked here?")
    result = specialist.execute(request)
    assert any("could not be resolved" in w for w in result.warnings)


def test_r_an_untrained_detector_degrades_the_result_but_still_answers(
    scratch: Path, monkeypatch
) -> None:
    """A head served on an untrained detector is degraded even when the specs
    agree, because the change representation is not the trained one."""
    monkeypatch.setattr(
        "preprocessing.imagery.load_image_array",
        lambda _p: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    spec = feature_spec_hash(256, extractor="stanet:untrained")
    specialist = ChangeVQASpecialist(
        head=_trained_head(),
        feature_extractor=_StubFeatureExtractor(spec, trained=False),
        text_encoder=_StubTextEncoder(),
        head_metadata={"change_cache_spec": spec, "detector_trained": False},
    )
    result = specialist.execute(_specialist_request(scratch))
    assert result.degraded is True
    assert result.answer != ""
    assert any("UNTRAINED" in w for w in result.warnings)


def test_r_the_low_confidence_threshold_is_a_reporting_threshold_not_a_calibration() -> None:
    assert 0.0 < LOW_CONFIDENCE_THRESHOLD < 1.0


def test_r_describe_reports_the_skew_and_the_confidence_method() -> None:
    specialist = ChangeVQASpecialist(
        head=None,
        feature_extractor=_StubFeatureExtractor(UNTRAINED_SPEC, trained=False),
        text_encoder=_StubTextEncoder(),
        head_metadata={"change_cache_spec": TRAINED_SPEC},
    )
    described = specialist.describe()
    assert described["has_trained_head"] is False
    assert described["detector_trained"] is False
    assert described["confidence_method"] == "uncalibrated"
    assert described["trained_feature_spec"] == TRAINED_SPEC
    assert described["serving_feature_spec"] == UNTRAINED_SPEC
    assert described["feature_spec_mismatch"] is not None


def test_r_model_refs_name_the_three_components() -> None:
    specialist = ChangeVQASpecialist(
        head=None,
        feature_extractor=_StubFeatureExtractor(TRAINED_SPEC),
        text_encoder=_StubTextEncoder(),
        head_path="/x/head.pt",
    )
    names = {ref["name"] for ref in specialist.model_refs()}
    assert names == {"change_vqa_head", "stanet_change_detector", "question_encoder"}


# ===========================================================================
# Area R - planner gating
# ===========================================================================
def test_r_a_change_language_request_is_answered_when_the_capability_exists() -> None:
    plan = _planner(ALL_CAPABILITIES + ("change_vqa",)).plan(
        _prediction(Task.CHANGE, temporal=True), _request(2)
    )
    assert [s.capability for s in plan.steps] == ["change", "change_vqa"]
    assert plan.steps[1].reason == "task:change+language_output:answer"
    assert plan.steps[1].asset_indices == (0, 1)
    # The caption step is SUPERSEDED, not stacked: captioning is the fallback
    # for a deployment without an answer capability, not a second requirement.
    assert "caption" not in [s.capability for s in plan.steps]


def test_r_without_the_capability_the_existing_caption_fallback_is_unchanged() -> None:
    """The R-02 change must be invisible to a deployment that lacks it."""
    plan = _planner(ALL_CAPABILITIES).plan(
        _prediction(Task.CHANGE, temporal=True), _request(2)
    )
    assert [s.capability for s in plan.steps] == ["change", "caption"]
    assert plan.steps[1].reason == "task:change+language_output"


def test_r_a_change_request_without_language_output_plans_only_change() -> None:
    plan = _planner(ALL_CAPABILITIES + ("change_vqa",)).plan(
        _prediction(Task.CHANGE, temporal=True, language_output=False), _request(2)
    )
    assert [s.capability for s in plan.steps] == ["change"]


def test_r_the_change_vqa_step_is_optional_not_required() -> None:
    plan = _planner(ALL_CAPABILITIES + ("change_vqa",)).plan(
        _prediction(Task.CHANGE, temporal=True), _request(2)
    )
    assert plan.steps[1].required is False


def test_r_the_capability_mappings_are_declared() -> None:
    assert TASK_CAPABILITY[Task.CHANGE_VQA] == "change_vqa"
    assert CAPABILITY_ASSETS["change_vqa"] == 2


def test_r_change_vqa_is_not_keyed_as_vqa() -> None:
    """A `vqa`-keyed row would route a two-asset change question to the
    one-asset VLM and silently answer about only one acquisition."""
    specs = {spec.name: spec for spec in default_specs()}
    assert "change_vqa" in specs
    assert specs["change_vqa"].capabilities == ("change_vqa",)
    assert "vqa" not in specs["change_vqa"].capabilities
    assert specs["change_vqa"].requires_assets == 2


def test_r_the_registry_spec_points_at_the_real_module_and_builder() -> None:
    spec = {s.name: s for s in default_specs()}["change_vqa"]
    assert spec.module == "specialists.change.vqa_specialist"
    assert spec.builder == "build_change_vqa_specialist"
    assert spec.optional_config_keys["head_path"] == "change_vqa.head_path"
    # The detector checkpoint is shared with the change specialist, not a new one.
    assert spec.optional_config_keys["checkpoint_path"] == "change.checkpoint_path"


def test_r_the_real_registry_exposes_change_vqa_without_constructing_it() -> None:
    class _Cfg:
        device_preference = "cpu"

        def get(self, path: str, default: Any = None) -> Any:
            return default

    registry = SpecialistRegistry.discover(_Cfg())
    assert "change_vqa" in registry.available()
    plan = PolicyPlanner(registry).plan(
        _prediction(Task.CHANGE, temporal=True), _request(2)
    )
    assert [s.capability for s in plan.steps] == ["change", "change_vqa"]
    assert all(entry.specialist is None for entry in registry.entries())
