"""Integration tests: the F5-5 quality gate blocks the VLM before it runs.

The unit tests in `tests/unit/test_quality_gate.py` prove the gate classifies
correctly. These prove the *specialist* honours it -- specifically that a
refused input never reaches `model.generate`.

That distinction matters. A gate that classifies correctly but is consulted
after the model call would still allow the hallucination; the model would just
have already produced a fabricated answer. The load-bearing assertion is
`stub.calls == []`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core.schemas import Task
from specialists.base import SpecialistRequest
from specialists.vqa.inference import VLMSpecialist
from specialists.vqa.prompts import PROMPT_VERSION

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class RecordingStubVLM:
    """Same call surface as SmolVLM; records every generate() invocation."""

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
        self.calls.append({"image_shape": getattr(image, "size", None)})
        return self.answer


def _write_geotiff(path: Path, planes: np.ndarray) -> Path:
    import rasterio
    from affine import Affine

    count, height, width = planes.shape
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=count,
        dtype=str(planes.dtype), crs="EPSG:32643",
        transform=Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 2_500_000.0),
    ) as ds:
        ds.write(planes)
    return path


@pytest.fixture()
def structured_geotiff(tmp_path: Path) -> Path:
    """Strong neighbour correlation: passes the gate."""
    size = 128
    yy, xx = np.mgrid[0:size, 0:size]
    ramp = (xx + yy).astype(np.float64)
    texture = np.random.default_rng(0).integers(0, 40, (size, size)).astype(np.float64)
    plane = ramp + texture
    planes = np.stack([plane, plane * 0.5, plane * 0.25]).astype("uint16")
    return _write_geotiff(tmp_path / "structured.tif", planes)


@pytest.fixture()
def noise_geotiff(tmp_path: Path) -> Path:
    """The exact F5-5 input class: uniform random, uncorrelated."""
    planes = np.random.default_rng(0).integers(0, 4000, (3, 128, 128)).astype("uint16")
    return _write_geotiff(tmp_path / "noise.tif", planes)


@pytest.fixture()
def flat_geotiff(tmp_path: Path) -> Path:
    planes = np.full((3, 128, 128), 1200, dtype="uint16")
    return _write_geotiff(tmp_path / "flat.tif", planes)


def _request(path: Path, task: str = "vqa", query: str = "What is visible?") -> SpecialistRequest:
    from preprocessing.raster import inspect_raster

    return SpecialistRequest(
        assets=[inspect_raster(path)], query=query, params={"task": task}
    )


# ---------------------------------------------------------------------------
# The critical property: the model is never called on rejected input
# ---------------------------------------------------------------------------


def test_noise_input_never_reaches_the_model(noise_geotiff: Path) -> None:
    """The whole reason the gate exists.

    If this fails, the system can still produce a fluent fabricated scene
    description for data that is not imagery -- which is exactly what the slow
    probe measured.
    """
    stub = RecordingStubVLM()
    specialist = VLMSpecialist(model=stub)

    result = specialist.execute(_request(noise_geotiff))

    assert stub.calls == [], "the VLM was invoked despite a blocked input"
    assert result.degraded is True


def test_structured_input_does_reach_the_model(structured_geotiff: Path) -> None:
    """The gate must not be a blanket refusal."""
    stub = RecordingStubVLM()
    specialist = VLMSpecialist(model=stub)

    result = specialist.execute(_request(structured_geotiff))

    assert len(stub.calls) == 1
    assert result.degraded is False


# ---------------------------------------------------------------------------
# Refusal shape -- must still be a well-formed SpecialistResult
# ---------------------------------------------------------------------------


def test_refusal_is_a_valid_result_not_an_exception(noise_geotiff: Path) -> None:
    specialist = VLMSpecialist(model=RecordingStubVLM())
    result = specialist.execute(_request(noise_geotiff))

    assert result.task is Task.VQA
    assert result.answer
    assert isinstance(result.answer, str)


def test_refusal_says_why(noise_geotiff: Path) -> None:
    specialist = VLMSpecialist(model=RecordingStubVLM())
    result = specialist.execute(_request(noise_geotiff))

    assert "not enough information" in result.answer.lower()
    assert "structure" in result.answer.lower() or "autocorrelation" in result.answer.lower()


def test_refusal_confidence_is_zero(noise_geotiff: Path) -> None:
    """A fluent refusal must not carry a positive score."""
    specialist = VLMSpecialist(model=RecordingStubVLM())
    result = specialist.execute(_request(noise_geotiff))

    assert result.confidence.value == 0.0
    assert result.confidence.degraded is True
    assert result.confidence.degradation_reason


def test_refusal_confidence_records_the_measured_signal(noise_geotiff: Path) -> None:
    specialist = VLMSpecialist(model=RecordingStubVLM())
    result = specialist.execute(_request(noise_geotiff))

    components = result.confidence.components
    assert components["input_usable"] == 0.0
    assert "autocorrelation" in components


def test_refusal_evidence_records_the_quality_verdict(noise_geotiff: Path) -> None:
    specialist = VLMSpecialist(model=RecordingStubVLM())
    result = specialist.execute(_request(noise_geotiff))

    assert len(result.evidence) == 1
    payload = result.evidence[0].payload
    assert payload["refused"] is True
    assert payload["quality"]["verdict"] == "noise"
    assert payload["prompt_version"] == PROMPT_VERSION


def test_refusal_is_not_a_silent_empty_answer(noise_geotiff: Path) -> None:
    """`schema_valid` stays 1.0: the specialist DID produce a valid result.

    It declined to answer the question, which is different from failing.
    """
    specialist = VLMSpecialist(model=RecordingStubVLM())
    result = specialist.execute(_request(noise_geotiff))

    assert result.confidence.components["schema_valid"] == 1.0
    assert result.answer.strip()


# ---------------------------------------------------------------------------
# Flat input: degraded but allowed through
# ---------------------------------------------------------------------------


def test_flat_input_is_analysed_but_flagged(flat_geotiff: Path) -> None:
    """A blank tile is a real (if uninformative) input, not noise.

    It must not be hard-blocked -- but the result must say it was degraded.
    """
    stub = RecordingStubVLM()
    specialist = VLMSpecialist(model=stub)

    result = specialist.execute(_request(flat_geotiff))

    assert len(stub.calls) == 1, "flat input should still be analysed"
    assert result.degraded is True
    assert any("quality" in w for w in result.warnings)


def test_flat_input_confidence_is_penalised(flat_geotiff: Path) -> None:
    specialist = VLMSpecialist(model=RecordingStubVLM())
    result = specialist.execute(_request(flat_geotiff))

    assert result.confidence.degraded is True
    assert result.confidence.components["input_structured"] == 0.0


# ---------------------------------------------------------------------------
# The gate applies to caption too, not just vqa
# ---------------------------------------------------------------------------


def test_caption_on_noise_is_also_refused(noise_geotiff: Path) -> None:
    stub = RecordingStubVLM()
    specialist = VLMSpecialist(model=stub)

    result = specialist.execute(_request(noise_geotiff, task="caption", query=""))

    assert stub.calls == []
    assert result.task is Task.CAPTION
    assert result.degraded is True


# ---------------------------------------------------------------------------
# Confidence is multiplicative, not additive
# ---------------------------------------------------------------------------


def test_a_fluent_answer_to_unusable_input_cannot_score_well(
    noise_geotiff: Path,
) -> None:
    """The failure mode in one test.

    The stub returns a perfectly fluent, confident sentence. On blocked input
    the reported confidence must still be zero -- fluency must not be able to
    buy a score.
    """
    fluent = "A dense urban area with a river running along the eastern edge."
    specialist = VLMSpecialist(model=RecordingStubVLM(fluent))

    result = specialist.execute(_request(noise_geotiff))

    assert result.confidence.value == 0.0
    assert result.confidence.degraded is True


def test_confidence_is_unchanged_when_no_quality_is_supplied() -> None:
    """Backwards compatibility for callers that pass only an answer."""
    specialist = VLMSpecialist(model=RecordingStubVLM())
    breakdown = specialist._confidence_for("Forest.")

    assert breakdown.raw == pytest.approx(1.0)
    assert breakdown.degraded is False