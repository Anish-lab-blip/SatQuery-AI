"""Regression tests for `app.serving` — the public serving composition root.

The point of these tests is the BEHAVIOUR, not the wiring: the trained change
head must reach the change specialist WITHOUT moving `Config.hash`, and the
serving path must degrade rather than crash when the artifact is absent.

Why the hash matters: `Config.hash` is a sha256 over the whole registry, and
`scripts/eval_change.py` refuses to score when it drifts (exit 3). Wiring the
head via config would invalidate the project's benchmark; the registry
`builders=` override is the supported path precisely because it is a call-site
argument that leaves the hash bit-identical.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from core.config import load_config
from core.registry import RegistryState, SpecialistRegistry, default_specs

from app import serving

pytest.importorskip("torch")

#: The hash the shipped head was trained under, and the artifact identity.
FROZEN_CONFIG_HASH = "78f1e3700da15aa1"
HEAD_SHA256 = "c5ef31277b67aa01a593aec0eac503eeaccc6d674349fda20ca44c9cc6f8e9fa"
HEAD_BYTES = 63_231_009
BASE_YAML_SHA256 = "88434f7f8f78e2b88d0d522e03ca86da25670a3f78b4f3c58c70b60044b80f89"

_REPO = Path(__file__).resolve().parents[2]
_BASE_YAML = _REPO / "configs" / "base.yaml"

requires_head = pytest.mark.skipif(
    not serving.CHANGE_CHECKPOINT.exists(),
    reason="trained change head absent from this checkout",
)


class _TrainedChangeStub:
    """The minimal object the registry classifies as AVAILABLE.

    Mirrors the real specialist's availability flag
    (`specialists/change/specialist.py:173` `has_checkpoint`) and the capability
    tuple the spec asserts against.
    """

    name = "change"
    version = "0.0.0-test"
    capabilities = ("change",)
    has_checkpoint = True


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1. The override reaches the change builder carrying the head
# ---------------------------------------------------------------------------
@requires_head
def test_serving_override_reaches_the_change_builder_with_the_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The serving override must reach the real builder carrying the head.

    A recording stub stands in for `build_change_specialist`; it must receive the
    `checkpoint_path` the override injects, pointing exactly at the shipped head.
    """
    recorded: dict[str, Any] = {}

    def _recording_builder(config: Any, **kw: Any) -> Any:
        recorded.update(kw)
        return _TrainedChangeStub()

    monkeypatch.setattr(
        "specialists.change.specialist.build_change_specialist", _recording_builder
    )

    registry = serving.build_serving_registry(load_config())
    entry = registry.build("change")

    # The load-bearing assertion: the injected artifact reaches the builder.
    assert recorded.get("checkpoint_path") == str(serving.CHANGE_CHECKPOINT)
    resolved = Path(recorded["checkpoint_path"]).resolve()
    assert resolved.as_posix().endswith("artifacts/change/levir_change_v001/head.pt")
    # The override was used, and the entry it produced is AVAILABLE.
    assert entry.specialist is not None and entry.specialist.has_checkpoint is True
    assert entry.state is RegistryState.AVAILABLE


# ---------------------------------------------------------------------------
# 2. The override is a call-site argument, not config — the hash must not move
# ---------------------------------------------------------------------------
def test_serving_does_not_move_the_config_hash() -> None:
    before = load_config().hash
    assert before == FROZEN_CONFIG_HASH

    registry = serving.build_serving_registry(load_config())
    controller = serving.build_serving_controller(load_config())

    assert load_config().hash == before == FROZEN_CONFIG_HASH
    assert registry.config.hash == FROZEN_CONFIG_HASH
    assert controller.config.hash == FROZEN_CONFIG_HASH


# ---------------------------------------------------------------------------
# 3. With the real checkpoint the capability is AVAILABLE
# ---------------------------------------------------------------------------
@requires_head
def test_real_checkpoint_yields_an_available_change_capability() -> None:
    registry = serving.build_serving_registry(load_config())
    entry = registry.build("change")

    assert entry.state is RegistryState.AVAILABLE
    assert entry.degraded is False
    assert entry.specialist is not None
    assert entry.specialist.has_checkpoint is True


# ---------------------------------------------------------------------------
# 4. Contrast — with NO override the capability is DEGRADED
# ---------------------------------------------------------------------------
def test_without_the_override_the_change_capability_is_degraded() -> None:
    """Pins the default the serving path deliberately departs from."""
    cfg = load_config()
    registry = SpecialistRegistry(cfg, specs=default_specs())  # no builders override
    entry = registry.build("change")

    assert entry.state is RegistryState.DEGRADED
    assert entry.specialist is not None
    assert entry.specialist.has_checkpoint is False


# ---------------------------------------------------------------------------
# 5. The checkpoint artifact is intact
# ---------------------------------------------------------------------------
@requires_head
def test_the_checkpoint_artifact_is_intact() -> None:
    head = serving.CHANGE_CHECKPOINT
    assert head.exists()
    assert head.stat().st_size == HEAD_BYTES
    assert _sha256(head) == HEAD_SHA256


# ---------------------------------------------------------------------------
# 6. A missing checkpoint degrades gracefully
# ---------------------------------------------------------------------------
def test_missing_checkpoint_degrades_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A serving path must run when the artifact is absent — degrade, not crash."""
    monkeypatch.setattr(
        serving, "CHANGE_CHECKPOINT", _REPO / "does" / "not" / "exist_head.pt"
    )

    registry = serving.build_serving_registry(load_config())  # must not raise
    entry = registry.build("change")                          # must not raise

    assert entry.state is RegistryState.DEGRADED
    assert entry.specialist is not None
    assert entry.specialist.has_checkpoint is False


# ---------------------------------------------------------------------------
# 7. Using app.serving leaves configs/base.yaml byte-identical
# ---------------------------------------------------------------------------
def test_base_yaml_is_untouched_by_using_app_serving() -> None:
    before = _sha256(_BASE_YAML)
    assert before == BASE_YAML_SHA256

    serving.build_serving_registry(load_config())
    serving.build_serving_controller(load_config())

    assert _sha256(_BASE_YAML) == before == BASE_YAML_SHA256


# ---------------------------------------------------------------------------
# 8. The public API is exported from the package
# ---------------------------------------------------------------------------
def test_public_api_is_exported() -> None:
    import app

    assert app.CHANGE_CHECKPOINT == serving.CHANGE_CHECKPOINT
    assert app.CHANGE_VQA_HEAD == serving.CHANGE_VQA_HEAD
    assert callable(app.build_serving_registry)
    assert callable(app.build_serving_controller)


# ---------------------------------------------------------------------------
# 9. R-02 / finding F2 — the change-VQA specialist is wired to the SAME
#    trained STANet the head's training features were built from.
#
#    Before this wiring, `change.checkpoint_path` being unset meant the serving
#    registry passed no `checkpoint_path`, so the specialist would build an
#    UNTRAINED STANet and answer from a representation the head was never fitted
#    on. `feature_spec_mismatch()` refused to answer on that skew (loud, not
#    silent) — but a deployment that can never answer is not a deployment.
# ---------------------------------------------------------------------------
class _ChangeVQAStub:
    """The minimal object the registry accepts for the `change_vqa` spec."""

    name = "change_vqa"
    version = "0.0.0-test"
    capabilities = ("change_vqa",)
    has_head = False


@requires_head
def test_serving_wires_change_vqa_to_the_same_trained_stanet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The load-bearing F2 assertion: one detector artifact, two specialists.

    A recording stub stands in for `build_change_vqa_specialist`; it must
    receive the SAME `checkpoint_path` the change detector receives, i.e.
    `app.serving.CHANGE_CHECKPOINT`.

    NOTE ON `head_path` (corrected 2026-09-22). This test previously asserted
    `recorded["head_path"] is None` and `entry.state is DEGRADED`, with the
    comment "the head is absent in this checkout". Both assertions encoded the
    *pre-promotion* state rather than the F2 guarantee: they passed only while
    the verified R-02 checkpoint was missing, so promoting the artifact turned
    them red. The intent here is the DETECTOR identity, which is what is
    asserted below. The absent-artifact behaviour those lines were reaching for
    is tested properly — by pointing the constants at missing files — in
    `test_absent_change_artifacts_are_passed_as_none_not_as_fake_paths`.
    """
    recorded: dict[str, Any] = {}

    def _recording_builder(config: Any, **kw: Any) -> Any:
        recorded.update(kw)
        return _ChangeVQAStub()

    monkeypatch.setattr(
        "specialists.change.vqa_specialist.build_change_vqa_specialist",
        _recording_builder,
    )

    registry = serving.build_serving_registry(load_config())
    entry = registry.build("change_vqa")

    # F2: the change-VQA path is handed the SAME detector as the change path.
    assert recorded.get("checkpoint_path") == str(serving.CHANGE_CHECKPOINT)
    assert Path(recorded["checkpoint_path"]).resolve().as_posix().endswith(
        "artifacts/change/levir_change_v001/head.pt"
    )
    # When the head is present the builder is handed its real path; when it is
    # absent the builder is told `None` — never a fabricated path. Either way
    # the value must be exactly one of those two things, which is the property
    # the old `is None` assertion was a special case of.
    if serving.CHANGE_VQA_HEAD.exists():
        assert recorded.get("head_path") == str(serving.CHANGE_VQA_HEAD)
    else:
        assert recorded.get("head_path") is None
    assert entry.specialist is not None


def test_absent_change_artifacts_are_passed_as_none_not_as_fake_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent must stay absent.

    `build_change_vqa_specialist` treats a MISSING artifact as a degradation and
    a CORRUPT one as a `ModelLoadError`. Handing it a path that does not exist
    would turn "absent" into "broken" and lose that distinction, so the wiring
    must resolve both paths to `None` when the files are not there.
    """
    recorded: dict[str, Any] = {}

    def _recording_builder(config: Any, **kw: Any) -> Any:
        recorded.update(kw)
        return _ChangeVQAStub()

    monkeypatch.setattr(
        "specialists.change.vqa_specialist.build_change_vqa_specialist",
        _recording_builder,
    )
    monkeypatch.setattr(
        serving, "CHANGE_CHECKPOINT", _REPO / "does" / "not" / "exist_head.pt"
    )
    monkeypatch.setattr(
        serving, "CHANGE_VQA_HEAD", _REPO / "does" / "not" / "exist_vqa_head.pt"
    )

    registry = serving.build_serving_registry(load_config())
    entry = registry.build("change_vqa")

    assert recorded == {"device": "cpu", "checkpoint_path": None, "head_path": None}
    assert entry.state is RegistryState.DEGRADED


def test_serving_and_preparation_share_one_stanet_checkpoint() -> None:
    """The drift guard: preparation and serving must name the SAME artifact.

    `scripts/prepare_change_vqa.py` builds the head's training features from
    `DEFAULT_CHANGE_CHECKPOINT`; `app.serving` builds the answer's features from
    `CHANGE_CHECKPOINT`. If those two ever point at different files the head is
    fitted on one representation and served on another — the exact skew finding
    F2 describes, and one no per-example metric reveals. Pinning the equality
    here makes any divergence a test failure rather than a quiet accuracy loss.
    """
    from scripts.prepare_change_vqa import DEFAULT_CHANGE_CHECKPOINT

    assert serving.CHANGE_CHECKPOINT.resolve() == DEFAULT_CHANGE_CHECKPOINT.resolve()


def test_the_change_vqa_head_path_matches_the_training_and_eval_defaults() -> None:
    """`app.serving.CHANGE_VQA_HEAD` must be where training writes and eval reads.

    `scripts/train_change_vqa.py` writes `<output_dir>/head.pt` with `output_dir`
    defaulting to `artifacts/change_vqa/run`, and `scripts/evaluate_change_vqa.py`
    reads `artifacts/change_vqa/run/head.pt`. Serving must look in the same
    place, or a returned Kaggle artifact would be invisible to the deployment.
    """
    from scripts.evaluate_change_vqa import DEFAULT_CHECKPOINT

    assert serving.CHANGE_VQA_HEAD.resolve() == DEFAULT_CHECKPOINT.resolve()
    assert serving.CHANGE_VQA_HEAD.as_posix().endswith(
        "artifacts/change_vqa/run/head.pt"
    )


def test_the_change_vqa_override_does_not_move_the_config_hash() -> None:
    """Wiring a second specialist through the seam is still a call-site argument."""
    before = load_config().hash
    assert before == FROZEN_CONFIG_HASH

    serving.build_serving_registry(load_config())
    serving.build_serving_controller(load_config())

    assert load_config().hash == FROZEN_CONFIG_HASH
    assert _sha256(_BASE_YAML) == BASE_YAML_SHA256
