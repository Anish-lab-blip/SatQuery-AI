"""STEP 1 — the promoted R-02 change-VQA head at its canonical serving path.

WHY THIS MODULE EXISTS
----------------------
Before this promotion the verified R-02 checkpoint existed only in the returned
Kaggle export. `app/serving.py` declared `CHANGE_VQA_HEAD` at
`artifacts/change_vqa/run/head.pt`, so serving *pointed* at a path that no
checkout contained -- the specialist would construct, report itself unavailable,
and answer nothing. These tests pin the promotion so the wiring is a fact rather
than an intention.

WHY THE LITERALS ARE REPEATED HERE
----------------------------------
`HEAD_SHA256` and `PARAMETER_COUNT` are also pinned in
`tests/unit/test_change_vqa_head.py` and in the promotion record. That
duplication is deliberate: this module's job is to assert that the artifact
*physically present at the serving path* is the verified one. Importing the
expected hash from the module under test would make the check vacuous.

THE TESTS ARE SKIPS, NOT FAILURES, WHEN THE ARTIFACT IS ABSENT
--------------------------------------------------------------
A fresh checkout of this repository does not contain the 5.8 MB checkpoint --
it is produced externally. Failing here would make the suite red for a
legitimate state, so the artifact-dependent tests skip with an explicit reason
while the *path contract* tests (which need no artifact) always run. That keeps
"the code is wired correctly" and "the artifact has been promoted" as two
separately observable facts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app import serving

pytest.importorskip("torch")

#: The verified R-02 head identity. Written down, not imported -- see the module
#: docstring. Matches `ARTIFACT_VERIFICATION.md` and the promotion record.
HEAD_SHA256 = "cfae5e43b97ca930f568dc5b8ae4f36b24e9ff717af226159802206ffd63a82a"
HEAD_BYTES = 5_822_809
PARAMETER_COUNT = 1_453_912

#: The frozen STANet detector whose features the head was fitted on. If this
#: drifts, the head is answering from a representation it never saw.
STANET_SHA256 = "c5ef31277b67aa01a593aec0eac503eeaccc6d674349fda20ca44c9cc6f8e9fa"

#: The config hash the head was trained under.
FROZEN_CONFIG_HASH = "78f1e3700da15aa1"

_REPO = Path(__file__).resolve().parents[2]
_PROMOTION = _REPO / "artifacts" / "change_vqa" / "run" / "PROMOTION.json"

requires_head = pytest.mark.skipif(
    not serving.CHANGE_VQA_HEAD.exists(),
    reason="R-02 change-VQA head not promoted into this checkout",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Path contract — always runs, artifact or not
# ---------------------------------------------------------------------------
def test_serving_declares_the_canonical_change_vqa_path() -> None:
    """The serving constant must name the path the trainer writes by default.

    If these drift the promotion lands somewhere serving never looks, and the
    failure mode is silent: the specialist degrades to "unavailable" and the
    only symptom is a missing answer.
    """
    assert serving.CHANGE_VQA_HEAD == (
        _REPO / "artifacts" / "change_vqa" / "run" / "head.pt"
    )


def test_trainer_and_serving_agree_on_the_head_path() -> None:
    """The wrapper's `--output-dir` default and `CHANGE_VQA_HEAD` are one path.

    `scripts/train_change_vqa.py:DEFAULT_OUT` is where an external run is told to
    write; `CHANGE_VQA_HEAD` is where serving reads. A mismatch means every
    correctly-trained head still has to be moved by hand.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_train_change_vqa", _REPO / "scripts" / "train_change_vqa.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert Path(module.DEFAULT_OUT) == serving.CHANGE_VQA_HEAD.parent
    assert Path(module.DEFAULT_PREPARED) == serving.CHANGE_VQA_HEAD.parent.parent


def test_evaluator_and_serving_agree_on_the_head_path() -> None:
    """Same argument, for the evaluator's `--checkpoint` default.

    An evaluation that scores a different file from the one serving loads is not
    evidence about the deployed system.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_evaluate_change_vqa", _REPO / "scripts" / "evaluate_change_vqa.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert Path(module.DEFAULT_CHECKPOINT) == serving.CHANGE_VQA_HEAD


# ---------------------------------------------------------------------------
# Promotion record
# ---------------------------------------------------------------------------
def test_promotion_record_exists_and_is_self_consistent() -> None:
    """The provenance record must exist and describe the promoted artifact."""
    assert _PROMOTION.exists(), "promotion record missing"
    payload = json.loads(_PROMOTION.read_text(encoding="utf-8"))

    assert payload["artifact"]["path"] == "artifacts/change_vqa/run/head.pt"
    assert payload["artifact"]["sha256"] == HEAD_SHA256
    assert payload["artifact"]["bytes"] == HEAD_BYTES
    assert payload["artifact"]["parameters"] == PARAMETER_COUNT
    assert payload["artifact"]["satquery_trained"] is True
    assert payload["artifact"]["non_finite_tensors"] == 0
    assert payload["artifact"]["weights_modified"] is False
    assert payload["identity"]["config_hash"] == FROZEN_CONFIG_HASH
    assert payload["frozen_dependency"]["sha256"] == STANET_SHA256
    assert payload["serving_wiring"]["code_change_required"] is False


# ---------------------------------------------------------------------------
# The artifact itself
# ---------------------------------------------------------------------------
@requires_head
def test_promoted_head_is_byte_identical_to_the_verified_export() -> None:
    """The promoted file's digest is the verified one, and the size agrees."""
    assert serving.CHANGE_VQA_HEAD.stat().st_size == HEAD_BYTES
    assert _sha256(serving.CHANGE_VQA_HEAD) == HEAD_SHA256


@requires_head
def test_promoted_head_strict_loads_with_the_expected_identity() -> None:
    """Strict load succeeds, parameters match, and the trained flag is set.

    A head that loads but carries `_satquery_trained=False` is an untrained
    initialisation with the right shapes -- it predicts at chance and looks
    perfectly healthy. The flag is the only thing that distinguishes them.
    """
    from training.change_vqa.model import load_change_vqa_head

    model = load_change_vqa_head(serving.CHANGE_VQA_HEAD, device="cpu")

    assert sum(p.numel() for p in model.parameters()) == PARAMETER_COUNT
    assert getattr(model, "_satquery_trained", False) is True
    assert model.training is False, "serving requires eval mode"


@requires_head
def test_promoted_head_has_no_non_finite_tensors() -> None:
    """No NaN/Inf in any parameter or buffer.

    A NaN checkpoint still loads and still predicts -- it just predicts roughly
    at chance. Counting non-finite tensors is the check that actually detects it.
    """
    import torch

    from training.change_vqa.model import load_change_vqa_head

    model = load_change_vqa_head(serving.CHANGE_VQA_HEAD, device="cpu")

    offenders = [
        name
        for name, tensor in list(model.named_parameters())
        + list(model.named_buffers())
        if not torch.isfinite(tensor).all()
    ]
    assert offenders == [], f"non-finite tensors: {offenders}"


@requires_head
def test_frozen_stanet_dependency_is_unchanged() -> None:
    """The detector behind the head's features is still the frozen one.

    This is the train/serve-skew guard: it is not enough for *a* checkpoint to be
    present at the change path, it must be the one the head was fitted against.
    """
    assert serving.CHANGE_CHECKPOINT.exists()
    assert _sha256(serving.CHANGE_CHECKPOINT) == STANET_SHA256


@requires_head
def test_promoted_metadata_matches_the_run_record() -> None:
    """`model_metadata.json` and `run_record.json` describe one artifact."""
    run_dir = serving.CHANGE_VQA_HEAD.parent
    meta = json.loads((run_dir / "model_metadata.json").read_text(encoding="utf-8"))
    record = json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))

    assert meta["config_hash"] == FROZEN_CONFIG_HASH
    assert record["config_hash"] == FROZEN_CONFIG_HASH
    assert record["model"]["checkpoint_sha256"] == HEAD_SHA256
    assert record["model"]["parameters"] == PARAMETER_COUNT
    assert meta["architecture"] == record["architecture"]


@requires_head
def test_promoted_head_is_discoverable_by_the_serving_registry() -> None:
    """End to end: the serving registry reports `change_vqa` as AVAILABLE.

    This is the assertion that the promotion actually changed behaviour. Before
    it, the path was declared but empty and the specialist could only ever
    report itself unavailable.
    """
    from core.config import load_config
    from core.registry import RegistryState, SpecialistRegistry

    config = load_config()
    registry = SpecialistRegistry.discover(
        config, device="cpu", builders={"change_vqa": serving._wired_change_vqa_builder}
    )
    entry = registry.entry("change_vqa")

    assert entry is not None
    assert entry.state is RegistryState.AVAILABLE, (
        f"change_vqa is {entry.state.value}; the head is present but the "
        f"specialist did not become available"
    )
