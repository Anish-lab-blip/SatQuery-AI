"""Phase 5 tests — SmolVLM contract, prompts, and the VQA specialist.

The model itself is never downloaded here. What IS tested is everything that
was learned by probing it (docs/PHASE5_VLM_CONTRACT.md):

    F5-1  the loader class is resolved by feature detection
    F5-2  the processor resolution pin is applied, and its loss is caught
    F5-3  prompts carry one <image> token per image
    F5-4  the dtype kwarg falls back on TypeError

A test that downloads 1 GB is a test nobody runs, so the loader is exercised
through its pure helpers and a stub model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from core.errors import InvalidRequestError, SpecialistError
from core.schemas import EvidenceType, Modality, Task
from specialists.base import Specialist, SpecialistRequest
from specialists.vqa import prompts
from specialists.vqa.inference import (
    VLMSpecialist,
    _looks_like_refusal,
    _load_image,
)
from specialists.vqa.model import (
    DTYPE_KWARG_CANDIDATES,
    LOADER_CANDIDATES,
    SmolVLM,
    resolve_dtype_kwarg,
    resolve_loader_class,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class StubVLM:
    """A stand-in with the same call surface as SmolVLM.

    `generate` records its arguments so tests can assert what the specialist
    asked for, and returns a fixed answer.
    """

    checkpoint = "stub-model"
    revision = "stub-rev"

    def __init__(self, answer: str = "A water body beside a road.") -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        image: Any,
        messages: list[dict[str, Any]],
        max_new_tokens: int = 128,
        do_sample: bool = False,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
        )
        return self.answer


@pytest.fixture()
def geotiff(tmp_path: Path) -> Path:
    """A small 3-band GeoTIFF the specialist can actually read.

    The pixel content is deliberately STRUCTURED, not uniform random noise.
    The F5-5 input-quality gate (preprocessing/quality.py) measures lag-1
    spatial autocorrelation and refuses uncorrelated input, because the VLM
    will hallucinate a fluent scene description for noise. A fixture built
    from `np.random.integers` therefore lands on the NOISE verdict and every
    execute() test would receive a refusal instead of a stub answer.
    """
    import rasterio
    from affine import Affine

    path = tmp_path / "scene.tif"
    transform = Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 2_500_000.0)

    # Smooth 2-D ramp (strong neighbour correlation) plus mild texture.
    yy, xx = np.mgrid[0:64, 0:64]
    ramp = (xx + yy) * 20.0
    texture = np.random.default_rng(0).integers(0, 120, (64, 64)).astype(np.float64)
    plane = ramp + texture

    data = np.stack(
        [plane, plane * 0.5 + 800.0, plane * 0.25 + 1600.0]
    ).astype("uint16")

    with rasterio.open(
        path, "w", driver="GTiff", width=64, height=64, count=3,
        dtype="uint16", crs="EPSG:32643", transform=transform,
    ) as ds:
        ds.write(data)
    return path


def _asset(path: Path) -> Any:
    from preprocessing.raster import inspect_raster

    return inspect_raster(path)


def _request(path: Path, task: str = "vqa", query: str = "What is visible?") -> SpecialistRequest:
    return SpecialistRequest(
        assets=[_asset(path)],
        query=query,
        params={"task": task},
    )


# ---------------------------------------------------------------------------
# F5-1 — loader class resolution
# ---------------------------------------------------------------------------


def test_loader_resolves_to_a_real_class() -> None:
    cls = resolve_loader_class()
    assert isinstance(cls, type)


def test_loader_is_resolved_by_feature_detection() -> None:
    """The chosen class must be one of the documented candidates."""
    cls = resolve_loader_class()
    assert cls.__name__ in LOADER_CANDIDATES


def test_vision2seq_is_still_listed_as_a_candidate() -> None:
    """F5-1: the absent class stays in the candidate list on purpose.

    Removing it would hide the finding. It is first-choice-absent, not
    forgotten.
    """
    assert "AutoModelForVision2Seq" in LOADER_CANDIDATES


# ---------------------------------------------------------------------------
# F5-4 — dtype kwarg resolution
# ---------------------------------------------------------------------------


def test_dtype_kwarg_is_one_of_the_known_spellings() -> None:
    assert resolve_dtype_kwarg(resolve_loader_class()) in DTYPE_KWARG_CANDIDATES


def test_dtype_kwarg_prefers_the_v5_spelling() -> None:
    """On this transformers version the resolution must be `dtype`, not
    `torch_dtype` (F5-4, verified live)."""
    assert resolve_dtype_kwarg(resolve_loader_class()) == "dtype"


# ---------------------------------------------------------------------------
# F5-2 — the processor pin, and the guard that protects it
# ---------------------------------------------------------------------------


def test_count_images_reads_a_single_image_batch() -> None:
    """The pinned shape: (batch, 1, C, H, W) -> 1 image."""
    inputs = {"pixel_values": torch.zeros(1, 1, 3, 512, 512)}
    assert SmolVLM._count_images(inputs) == 1


def test_count_images_reads_the_unpinned_shape() -> None:
    """The default shape: (batch, 17, C, H, W) -> 17 images.

    This is the exact measurement that motivates F5-2: 4x4 sub-images plus one
    overview.
    """
    inputs = {"pixel_values": torch.zeros(1, 17, 3, 512, 512)}
    assert SmolVLM._count_images(inputs) == 17


def test_count_images_handles_missing_pixel_values() -> None:
    assert SmolVLM._count_images({}) == 0


def test_generate_raises_when_the_pin_is_lost() -> None:
    """Prove the F5-2 guard fires.

    A processor that produced 17 images means the longest_edge pin did not
    apply. The specialist must refuse rather than silently pay ~17x the
    attention cost, which on a 5 GPU-minute/day deployment quota is the
    difference between working and not.
    """
    vlm = object.__new__(SmolVLM)
    vlm.checkpoint = "stub"
    vlm.revision = None
    vlm.device = "cpu"
    vlm.processor_longest_edge = 512
    vlm.do_image_splitting = True

    class ExplodingModel:
        def generate(self, **kwargs: Any) -> Any:
            raise AssertionError("generate should not be reached")

    class UnpinnedProcessor:
        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            return "<image> describe"

        def __call__(self, **kwargs: Any) -> dict[str, Any]:
            # The unpinned shape.
            return {"pixel_values": torch.zeros(1, 17, 3, 512, 512)}

        def batch_decode(self, *args: Any, **kwargs: Any) -> list[str]:
            return ["unused"]

    vlm.model = ExplodingModel()
    vlm.processor = UnpinnedProcessor()
    from specialists.vqa.model import VLMLoadInfo

    vlm.load_info = VLMLoadInfo(
        checkpoint="stub", revision=None, loader_class="x", dtype_kwarg="dtype",
        device="cpu", parameters=0, load_seconds=0.0,
        processor_longest_edge=512, max_images_seen=0,
    )

    with pytest.raises(Exception, match="17 images|longest_edge pin"):
        vlm.generate(object(), [{"role": "user", "content": []}])


# ---------------------------------------------------------------------------
# F5-3 — prompts
# ---------------------------------------------------------------------------


def test_prompt_version_is_declared() -> None:
    assert prompts.PROMPT_VERSION


def test_build_messages_emits_one_image_per_image() -> None:
    msgs = prompts.build_messages("vqa", "What is visible?", image_count=1)
    user_content = msgs[1]["content"]
    image_entries = [c for c in user_content if c["type"] == "image"]
    assert len(image_entries) == 1


def test_build_messages_emits_multiple_images_when_asked() -> None:
    """The change path needs two images in one prompt."""
    msgs = prompts.build_messages("vqa", "Compare.", image_count=2)
    user_content = msgs[1]["content"]
    assert len([c for c in user_content if c["type"] == "image"]) == 2


def test_build_messages_rejects_zero_images() -> None:
    with pytest.raises(ValueError):
        prompts.build_messages("vqa", "x", image_count=0)


def test_build_messages_has_a_system_preamble() -> None:
    msgs = prompts.build_messages("caption", "Describe this.")
    assert msgs[0]["role"] == "system"


def test_no_prompt_requests_coordinates() -> None:
    """C-5: the VLM never produces coordinates.

    The assertion checks for *requests*, not for the bare word. The system
    preamble legitimately mentions "coordinates" inside a prohibition
    ("do not speculate about locations, coordinates, measurements or dates"),
    which is the opposite of a request. A crude substring check would reject
    correct prompts and hide the real failure mode.
    """
    requesting_patterns = (
        "give me the coordinates",
        "return the coordinates",
        "output the bounding box",
        "provide the bounding box",
        "draw a box",
        "list the box coordinates",
    )
    for kind in ("vqa", "caption", "explain_grounding", "explain_change"):
        msgs = prompts.build_messages(kind, "test")
        text = " ".join(
            c.get("text", "")
            for m in msgs
            for c in m["content"]
            if isinstance(c, dict)
        ).lower()
        for pattern in requesting_patterns:
            assert pattern not in text, (
                f"prompt kind {kind!r} asks the VLM for spatial output: "
                f"{pattern!r}"
            )


def test_every_prompt_kind_is_present() -> None:
    described = prompts.describe_prompts()
    assert described["version"] == prompts.PROMPT_VERSION
    assert set(described["kinds"]) == {
        "vqa", "caption", "explain_grounding", "explain_change"
    }


# ---------------------------------------------------------------------------
# Answer normalisation and refusal detection
# ---------------------------------------------------------------------------


def test_refusal_markers_are_detected() -> None:
    assert _looks_like_refusal("Not enough information in the image.")
    assert _looks_like_refusal("I cannot determine this from the image.")
    assert not _looks_like_refusal("A large water body is visible.")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("  A   water  body ", "A water body"),
        ("Forest.", "Forest"),
        ("Urban area, farmland and water.", "Urban area, farmland and water."),
    ],
)
def test_answer_normalisation(raw: str, expected: str) -> None:
    assert VLMSpecialist._normalize_answer(raw) == expected


def test_empty_answer_becomes_an_explicit_refusal() -> None:
    assert "not enough information" in VLMSpecialist._normalize_answer("   ").lower()


def test_refusal_produces_degraded_confidence(geotiff: Path) -> None:
    specialist = VLMSpecialist(model=StubVLM("Not enough information in the image."))
    result = specialist.execute(_request(geotiff))
    assert result.confidence.degraded is True
    assert result.confidence.degradation_reason


def test_normal_answer_is_not_degraded(geotiff: Path) -> None:
    specialist = VLMSpecialist(model=StubVLM("A dense urban area."))
    result = specialist.execute(_request(geotiff))
    assert result.confidence.degraded is False


def test_confidence_components_are_measurable(geotiff: Path) -> None:
    """No LLM utterance. Every component must be a number with a provenance."""
    specialist = VLMSpecialist(model=StubVLM("Forest."))
    result = specialist.execute(_request(geotiff))
    assert "schema_valid" in result.confidence.components
    assert "declined" in result.confidence.components
    assert all(isinstance(v, float) for v in result.confidence.components.values())


# ---------------------------------------------------------------------------
# Specialist interface conformance
# ---------------------------------------------------------------------------


def test_vqa_specialist_implements_the_interface() -> None:
    assert issubclass(VLMSpecialist, Specialist)


def test_vqa_specialist_declares_both_capabilities() -> None:
    assert set(VLMSpecialist.capabilities) == {"vqa", "caption"}


def test_specialist_rejects_zero_assets() -> None:
    specialist = VLMSpecialist(model=StubVLM())
    with pytest.raises(InvalidRequestError):
        specialist.validate_request(
            SpecialistRequest(assets=[], query="x", params={"task": "vqa"})
        )


def test_specialist_rejects_two_assets(geotiff: Path) -> None:
    specialist = VLMSpecialist(model=StubVLM())
    asset = _asset(geotiff)
    with pytest.raises(InvalidRequestError):
        specialist.validate_request(
            SpecialistRequest(
                assets=[asset, asset], query="x", params={"task": "vqa"}
            )
        )


def test_specialist_rejects_an_unknown_task(geotiff: Path) -> None:
    specialist = VLMSpecialist(model=StubVLM())
    with pytest.raises(InvalidRequestError):
        specialist.validate_request(
            SpecialistRequest(
                assets=[_asset(geotiff)], query="x", params={"task": "grounding"}
            )
        )


def test_vqa_requires_a_question(geotiff: Path) -> None:
    specialist = VLMSpecialist(model=StubVLM())
    with pytest.raises(InvalidRequestError):
        specialist.validate_request(_request(geotiff, task="vqa", query="   "))


def test_caption_needs_no_question(geotiff: Path) -> None:
    specialist = VLMSpecialist(model=StubVLM())
    specialist.validate_request(_request(geotiff, task="caption", query=""))


def test_specialist_rejects_a_missing_file(tmp_path: Path) -> None:
    from core.schemas import AssetMetadata

    specialist = VLMSpecialist(model=StubVLM())
    with pytest.raises(InvalidRequestError):
        specialist.validate_request(
            SpecialistRequest(
                assets=[AssetMetadata(path=str(tmp_path / "nope.tif"))],
                query="x",
                params={"task": "vqa"},
            )
        )


# ---------------------------------------------------------------------------
# End-to-end with a stub model
# ---------------------------------------------------------------------------


def test_execute_returns_a_vqa_result(geotiff: Path) -> None:
    specialist = VLMSpecialist(model=StubVLM("A water body."))
    result = specialist.execute(_request(geotiff, task="vqa"))
    assert result.task is Task.VQA
    # Multi-word answers keep their punctuation; only single words are stripped.
    assert result.answer == "A water body."


def test_execute_returns_a_caption_result(geotiff: Path) -> None:
    specialist = VLMSpecialist(model=StubVLM("A coastal scene."))
    result = specialist.execute(_request(geotiff, task="caption", query=""))
    assert result.task is Task.CAPTION


def test_execute_preserves_geospatial_metadata(geotiff: Path) -> None:
    """The CRS must survive the VLM path. Silently dropping it would break
    every downstream geospatial evaluation."""
    specialist = VLMSpecialist(model=StubVLM("Urban."))
    result = specialist.execute(_request(geotiff))
    assert result.geospatial.has_crs is True
    assert "32643" in (result.geospatial.crs or "")


def test_execute_produces_evidence_attributed_to_this_specialist(geotiff: Path) -> None:
    specialist = VLMSpecialist(model=StubVLM("Forest."))
    result = specialist.execute(_request(geotiff))
    assert len(result.evidence) == 1
    assert result.evidence[0].source_specialist == "vlm"
    assert result.evidence[0].type is EvidenceType.STATISTIC


def test_evidence_carries_the_prompt_version(geotiff: Path) -> None:
    """An answer without its prompt version cannot be reproduced."""
    specialist = VLMSpecialist(model=StubVLM("Forest."))
    result = specialist.execute(_request(geotiff))
    assert result.evidence[0].payload["prompt_version"] == prompts.PROMPT_VERSION


def test_generate_is_greedy_by_default(geotiff: Path) -> None:
    """Deterministic decoding is a requirement, not a preference."""
    stub = StubVLM("Forest.")
    specialist = VLMSpecialist(model=stub)
    specialist.execute(_request(geotiff))
    assert stub.calls[0]["do_sample"] is False


def test_generate_respects_the_token_bound(geotiff: Path) -> None:
    stub = StubVLM("Forest.")
    specialist = VLMSpecialist(model=stub, max_new_tokens=42)
    specialist.execute(_request(geotiff))
    assert stub.calls[0]["max_new_tokens"] == 42


# ---------------------------------------------------------------------------
# Image loading preserves geometry
# ---------------------------------------------------------------------------


def test_load_image_returns_three_channel_rgb(geotiff: Path) -> None:
    image = _load_image(geotiff)
    assert image.mode == "RGB"
    assert image.size == (64, 64)


def test_load_image_is_deterministic(geotiff: Path) -> None:
    a = np.asarray(_load_image(geotiff))
    b = np.asarray(_load_image(geotiff))
    assert np.array_equal(a, b)