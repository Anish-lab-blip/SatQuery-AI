"""STEP 7 section I -- the R-02 head, verified through the real serving path.

WHY THIS IS A SEPARATE MODULE
-----------------------------
Each test here constructs the serving stack, which loads torch and the
real 1.45M-parameter head. Run inside the larger chain suite the module exceeded
this sandbox's per-command ceiling and was killed before reporting -- a
`SIGTERM`, not a failure. Splitting by cost keeps each file inside the ceiling
and, more importantly, keeps the *evidence class* separable: `pytest -m
real_inference` selects exactly this file and nothing else.

WHAT IS AND IS NOT CLAIMED
--------------------------
Everything below reads the REAL artifact from disk. Nothing is retrained and
nothing is mocked. The byte-level identity checks run first because a hash
mismatch would invalidate every later assertion.

Not claimed: that inference *served a request over the network*. There is no
reachable upstream in this environment (the sandbox routes egress through a
proxy that answers 502 -- see the chain suite's non-JSON test), so an end-to-end
`POST /v1/analyze` that reaches a live model was not performed. What WAS
performed is that the serving path loads the promoted artifact and the registry
resolves it as available. That is a real-inference result about loading, and it
is labelled as such rather than as a deployment test.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: The promoted R-02 artifact, as recorded in
#: `artifacts/change_vqa/run/PROMOTION.json`. Repeated here as literals
#: deliberately: a test that reads its expectation from the artifact it is
#: checking cannot detect a change in that artifact.
R02_SHA256 = "cfae5e43b97ca930f568dc5b8ae4f36b24e9ff717af226159802206ffd63a82a"
R02_BYTES = 5_822_809
R02_PARAMS = 1_453_912
STANET_SHA256_PREFIX = "c5ef3127"
STANET_BYTES = 63_231_009


# ===========================================================================
# I : R-02 serving-load verification  (real_inference)
# ===========================================================================
class TestI1R02ServingVerification:
    """Section I. The promoted artifact must load through the serving path.

    Every assertion here is about the REAL artifact on disk. Nothing is
    retrained, and nothing is mocked. The byte-level checks come first because
    a hash mismatch would invalidate everything after it.
    """

    @pytest.fixture(scope="class")
    def head_path(self) -> Path:
        path = REPO / "artifacts" / "change_vqa" / "run" / "head.pt"
        if not path.exists():
            pytest.skip(f"the promoted R-02 artifact is not present: {path}")
        return path

    @pytest.mark.real_inference
    def test_the_promoted_artifact_is_byte_identical_to_its_record(self, head_path: Path) -> None:
        """SHA256 and size must match `PROMOTION.json` exactly."""
        digest = hashlib.sha256(head_path.read_bytes()).hexdigest()
        assert digest == R02_SHA256, (
            f"the promoted R-02 artifact changed. expected {R02_SHA256}, got {digest}"
        )
        assert head_path.stat().st_size == R02_BYTES

    @pytest.mark.real_inference
    def test_the_artifact_parameter_count_is_unchanged(self, head_path: Path) -> None:
        """1,453,912 parameters -- counted from the tensors, not read from metadata."""
        torch = pytest.importorskip("torch", reason="torch is unavailable here")

        payload = torch.load(str(head_path), map_location="cpu", weights_only=False)
        state = payload["state_dict"]
        total = sum(v.numel() for v in state.values() if hasattr(v, "numel"))
        assert total == R02_PARAMS, f"parameter count drifted: {total}"

    @pytest.mark.real_inference
    def test_the_artifact_has_no_non_finite_tensors(self, head_path: Path) -> None:
        """NaN/Inf would make inference silently produce nonsense."""
        torch = pytest.importorskip("torch", reason="torch is unavailable here")

        state = torch.load(str(head_path), map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        bad = [k for k, v in state.items() if hasattr(v, "isfinite") and not bool(v.isfinite().all())]
        assert bad == [], f"non-finite tensors in the promoted artifact: {bad}"

    @pytest.mark.real_inference
    def test_the_stanet_checkpoint_is_unchanged(self) -> None:
        """The frozen Phase-9 change detector must stay byte-identical.

        Both the change specialist and the change-VQA feature builder depend on
        this file. A change here invalidates the change benchmark AND the R-02
        training features, so it is the highest-consequence invariant in the
        project.
        """
        path = REPO / "artifacts" / "change" / "levir_change_v001" / "head.pt"
        if not path.exists():
            pytest.skip("the frozen STANet checkpoint is not present")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest.startswith(STANET_SHA256_PREFIX), (
            f"the frozen STANet checkpoint changed: {digest[:16]}..."
        )
        assert path.stat().st_size == STANET_BYTES

    @pytest.mark.real_inference
    def test_the_loader_accepts_the_promoted_artifact(self, head_path: Path) -> None:
        """The REAL loader must load it, not just `torch.load`.

        This is the difference between "the file is intact" and "the serving
        path can use it". The loader asserts its own architecture and the
        `_satquery_trained` marker, so a silent architecture drift fails here.
        """
        pytest.importorskip("torch", reason="torch is unavailable here")
        try:
            from training.change_vqa.model import load_change_vqa_head
        except ImportError as exc:  # pragma: no cover
            pytest.skip(f"the change-VQA loader is unavailable: {exc}")

        model = load_change_vqa_head(str(head_path), device="cpu")
        assert model is not None
        assert getattr(model, "_satquery_trained", False) is True, (
            "the loaded head does not declare `_satquery_trained`; the serving "
            "path would treat it as untrained and degrade"
        )
        assert not model.training, "the head must be loaded in eval mode"

    @pytest.mark.real_inference
    def test_the_registry_resolves_the_change_vqa_capability(self) -> None:
        """With the artifact present, `change_vqa` must be buildable.

        This closes the loop the R-02 work opened: the head was promoted in
        STEP 1, and this asserts the registry actually resolves it rather than
        reporting DEGRADED.
        """
        pytest.importorskip("torch", reason="torch is unavailable here")
        from app.serving import build_serving_registry

        registry = build_serving_registry()
        entry = registry.build("change_vqa")
        assert entry is not None
        state = entry.state.value
        assert state in ("available", "degraded"), (
            f"change_vqa resolved to {state!r} with the artifact present; the "
            "promotion is not reaching the serving path"
        )
        assert state == "available", (
            "change_vqa is DEGRADED despite the promoted artifact being present "
            "and valid -- the promoted head is not being wired"
        )

    @pytest.mark.real_inference
    def test_the_serving_module_points_at_the_promoted_artifact(self) -> None:
        """The wiring constant must resolve to the artifact the test checked."""
        from app.serving import CHANGE_VQA_HEAD

        assert CHANGE_VQA_HEAD == REPO / "artifacts" / "change_vqa" / "run" / "head.pt"
        assert CHANGE_VQA_HEAD.exists()
