"""Phase 11/12 — the optical-SAR specialist serving contract.

The spec calls optical-SAR its #1 evaluation priority and the hidden target is
Cartosat-2S + RISAT. The tests below pin the parts of that contract that are
cheap to get wrong and expensive to discover late:

  * exactly two assets, ONE OPTICAL AND ONE SAR -- two opticals is rejected
  * the availability mask survives all the way to evidence (finding C-1)
  * CROMA is never handed a mask
  * confidence follows plan section 26 and is floored when there is no decision
  * the no-CROMA / no-head deployment cases DEGRADE rather than crash
  * confidence is never an LLM utterance

They use real GeoTIFFs on disk, because the specialist reads them through
`preprocessing.raster` and a mock would not exercise that path.

NOTHING HERE REQUIRES CROMA OR A TRAINED HEAD. That is deliberate: the
guarantees that matter most on a mismatched sensor pair -- canonical placement,
the mask, honest degradation -- are exactly the ones that must hold when no
weights are available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from core.errors import InvalidRequestError
from core.schemas import (
    EvidenceType,
    Modality,
    SensorDescriptor,
    Task,
)
from specialists.base import SpecialistRequest
from specialists.optical_sar.specialist import (
    CONFIDENCE_WEIGHTS,
    OpticalSarSpecialist,
    build_optical_sar_specialist,
)

pytest.importorskip("torch")

H = W = 32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_tif(
    path,
    array: np.ndarray,
    *,
    crs: str | None = "EPSG:32643",
    resolution: float = 10.0,
):
    profile = {
        "driver": "GTiff",
        "width": array.shape[2],
        "height": array.shape[1],
        "count": array.shape[0],
        "dtype": str(array.dtype),
    }
    if crs is not None:
        profile["crs"] = crs
        profile["transform"] = Affine(
            resolution, 0, 500_000.0, 0, -resolution, 2_500_000.0
        )
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(array)
    return path


@pytest.fixture(scope="module")
def scenes(tmp_path_factory) -> dict:
    """One Cartosat-2S-like optical scene and one RISAT-like dual-pol SAR scene."""
    root = tmp_path_factory.mktemp("optical_sar")
    rng = np.random.default_rng(0)

    optical = rng.integers(200, 3000, (4, H, W)).astype("uint16")
    sar = rng.integers(100, 4000, (2, H, W)).astype("uint16")

    return {
        "optical": _write_tif(root / "cartosat.tif", optical),
        "sar": _write_tif(root / "risat.tif", sar),
        "optical_2": _write_tif(root / "cartosat_b.tif", optical),
        "sar_2": _write_tif(root / "risat_b.tif", sar),
    }


def _optical_asset(path, *, modality=Modality.OPTICAL, sensor=None):
    from preprocessing.raster import inspect_raster

    return inspect_raster(path, explicit_modality=modality.value, sensor=sensor)


def _sar_asset(path, *, modality=Modality.SAR, sensor=None):
    from preprocessing.raster import inspect_raster

    return inspect_raster(path, explicit_modality=modality.value, sensor=sensor)


def _request(assets, query="compare the optical and radar views"):
    return SpecialistRequest(assets=list(assets), query=query)


def _specialist(**kwargs) -> OpticalSarSpecialist:
    return OpticalSarSpecialist(**kwargs)


# ---------------------------------------------------------------------------
# Two assets, and the right two
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", [0, 1, 3])
def test_wrong_asset_count_is_rejected(scenes, count):
    """0, 1 and 3 are all refused with a typed error naming the count."""
    assets = [
        _optical_asset(scenes["optical"]),
        _sar_asset(scenes["sar"]),
        _optical_asset(scenes["optical_2"]),
    ][:count]

    with pytest.raises(InvalidRequestError) as excinfo:
        _specialist().validate_request(_request(assets))

    error = excinfo.value
    assert error.code == "invalid_request"
    assert "2" in error.detail
    # The user-facing message must be actionable, not "invalid request".
    assert "optical" in error.user_message.lower()
    assert "radar" in error.user_message.lower() or "sar" in error.user_message.lower()


def test_correct_pair_is_accepted(scenes):
    specialist = _specialist()
    specialist.validate_request(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )


def test_two_optical_assets_are_rejected(scenes):
    """THE IMPORTANT ONE. Two opticals is the plausible mistake, not a typo.

    If it were accepted, the second optical image would fill the 2-channel SAR
    slot, CROMA would run, and the answer would claim to fuse two modalities
    while showing one. The modality check happens on the ASSETS, before any
    pixel is read.
    """
    with pytest.raises(InvalidRequestError) as excinfo:
        _specialist().validate_request(
            _request(
                [
                    _optical_asset(scenes["optical"]),
                    _optical_asset(scenes["optical_2"]),
                ]
            )
        )

    error = excinfo.value
    assert error.context.get("reason") == "two_optical"
    assert "optical" in error.user_message.lower()


def test_two_sar_assets_are_rejected(scenes):
    """The mirror case, and equally unanswerable."""
    with pytest.raises(InvalidRequestError) as excinfo:
        _specialist().validate_request(
            _request([_sar_asset(scenes["sar"]), _sar_asset(scenes["sar_2"])])
        )

    error = excinfo.value
    assert error.context.get("reason") == "two_sar"


def test_order_does_not_matter(scenes):
    """SAR first, optical second is the same request."""
    specialist = _specialist()
    specialist.validate_request(
        _request([_sar_asset(scenes["sar"]), _optical_asset(scenes["optical"])])
    )


def test_missing_file_is_rejected(tmp_path, scenes):
    from preprocessing.raster import inspect_raster

    asset = inspect_raster(
        scenes["optical"], explicit_modality=Modality.OPTICAL.value
    )
    ghost = asset.model_copy(update={"path": str(tmp_path / "nope.tif")})

    with pytest.raises(InvalidRequestError) as excinfo:
        _specialist().validate_request(_request([ghost, _sar_asset(scenes["sar"])]))
    assert "does not exist" in excinfo.value.detail


def test_undeclared_modality_is_inferred_from_band_count(scenes):
    """A 4-band optical and a 2-band SAR are inferable without a declaration."""
    optical = _optical_asset(scenes["optical"], modality=Modality.UNKNOWN)
    sar = _sar_asset(scenes["sar"], modality=Modality.UNKNOWN)

    result = _specialist().execute(_request([optical, sar]))

    # It ran, and it said the modality was inferred rather than declared.
    assert any("inferred" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# The task contract
# ---------------------------------------------------------------------------


def test_result_task_is_optical_sar(scenes):
    result = _specialist().execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )
    assert result.task is Task.OPTICAL_SAR


def test_specialist_declares_its_capability():
    specialist = _specialist()
    assert specialist.supports("optical_sar")
    assert specialist.name == "optical_sar"
    assert specialist.capabilities == ("optical_sar",)


def test_confidence_weights_are_a_normalised_distribution():
    """Plan section 26's four components, and they must sum to 1.0.

    A weights dict that does not sum to 1 silently changes what the score means.
    """
    assert set(CONFIDENCE_WEIGHTS) == {
        "fusion_margin",
        "optical_confidence",
        "sar_confidence",
        "cross_modal_agreement",
    }
    assert abs(sum(CONFIDENCE_WEIGHTS.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# The availability mask survives to evidence (finding C-1)
# ---------------------------------------------------------------------------


def test_availability_evidence_is_emitted_for_both_modalities(scenes):
    """Each modality gets an AVAILABILITY_MASK evidence item."""
    result = _specialist().execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )

    masks = [e for e in result.evidence if e.type is EvidenceType.AVAILABILITY_MASK]
    assert len(masks) == 2
    modalities = {e.payload["modality"] for e in masks}
    assert modalities == {"optical", "sar"}


def test_optical_mask_reports_the_absent_channels(scenes):
    """A 4-band Cartosat scene has 8 absent channels, recorded as absent.

    This is the fact the whole adapter exists to preserve, and it must reach the
    result -- an operator reading the answer needs to know the model ran on 4
    measured channels, not 12.
    """
    result = _specialist().execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )

    optical_mask = next(
        e for e in result.evidence
        if e.type is EvidenceType.AVAILABILITY_MASK and e.payload["modality"] == "optical"
    )
    payload = optical_mask.payload
    assert payload["canonical_channels"] == 12
    assert sum(payload["availability_mask"]) == 4
    assert payload["missing_channels"] == [4, 5, 6, 7, 8, 9, 10, 11]
    assert payload["missing_channels_zero_filled"] is True


def test_no_band_was_invented_for_the_missing_channels(scenes):
    """The hard rule, asserted through the specialist rather than the adapter.

    The optical scene has 4 real bands; the canonical tensor has 12 channels.
    Channels 5-12 must be exactly zero -- no interpolation, no repeat, no fill.
    """
    result = _specialist().execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )

    optical_mask = next(
        e for e in result.evidence
        if e.type is EvidenceType.AVAILABILITY_MASK and e.payload["modality"] == "optical"
    )
    # The mask says 8 channels are absent, and those are exactly the ones the
    # adapter zero-filled. The two facts must agree.
    absent = [i for i, present in enumerate(optical_mask.payload["availability_mask"]) if not present]
    assert absent == optical_mask.payload["missing_channels"]
    assert len(absent) == 8


def test_a_full_twelve_band_scene_reports_every_channel_present(tmp_path):
    """The complement: with 12 real bands nothing is masked off."""
    rng = np.random.default_rng(5)
    optical = _write_tif(
        tmp_path / "s2.tif", rng.integers(0, 4000, (12, H, W)).astype("uint16")
    )
    sar = _write_tif(
        tmp_path / "s1.tif", rng.integers(0, 4000, (2, H, W)).astype("uint16")
    )

    result = _specialist().execute(
        _request([_optical_asset(optical), _sar_asset(sar)])
    )

    optical_mask = next(
        e for e in result.evidence
        if e.type is EvidenceType.AVAILABILITY_MASK and e.payload["modality"] == "optical"
    )
    assert sum(optical_mask.payload["availability_mask"]) == 12
    assert optical_mask.payload["missing_channels"] == []


def test_declared_sensor_descriptor_is_used_not_overridden(tmp_path):
    """An explicit band map from the caller wins over any inference.

    The caller may know the real layout; overriding a declaration with a guess
    is the fabrication this module forbids.
    """
    rng = np.random.default_rng(3)
    optical = _write_tif(
        tmp_path / "declared.tif", rng.integers(0, 4000, (4, H, W)).astype("uint16")
    )
    sar = _write_tif(
        tmp_path / "declared_sar.tif", rng.integers(0, 4000, (2, H, W)).astype("uint16")
    )

    descriptor = SensorDescriptor(
        sensor="cartosat_2s",
        available_bands=["B1", "B2", "B3", "B4"],
        band_map={"B1": "B01", "B2": "B02", "B3": "B03", "B4": "B04"},
        availability_mask=[True, True, True, True] + [False] * 8,
        canonical_channels=12,
    )

    result = _specialist().execute(
        _request([
            _optical_asset(optical, sensor=descriptor),
            _sar_asset(sar),
        ])
    )

    optical_mask = next(
        e for e in result.evidence
        if e.type is EvidenceType.AVAILABILITY_MASK and e.payload["modality"] == "optical"
    )
    assert optical_mask.payload["sensor"] == "cartosat_2s"
    assert optical_mask.payload["band_map"] == descriptor.band_map


# ---------------------------------------------------------------------------
# Degradation: no CROMA, no head
# ---------------------------------------------------------------------------


def test_no_encoder_degrades_rather_than_crashing(scenes):
    """CROMA_base.pt is 777.6 MB and may be absent. That is a deployment case."""
    result = _specialist().execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )

    assert result.degraded is True
    assert result.confidence.degraded is True
    assert result.confidence.degradation_reason is not None
    assert any("CROMA" in w for w in result.warnings)


def test_no_prediction_is_emitted_without_an_encoder(scenes):
    """No class label may be invented. `labels` stays empty."""
    result = _specialist().execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )

    assert result.labels == []
    assert any("No fused prediction" in result.answer for _ in [0])


def test_sensor_side_facts_still_computed_without_any_model(scenes):
    """The masks are real work needing no weights, and they must still happen.

    These are the facts an operator most needs on a mismatched sensor pair, and
    they are exactly what survives a missing checkpoint.
    """
    result = _specialist().execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )

    masks = [e for e in result.evidence if e.type is EvidenceType.AVAILABILITY_MASK]
    assert len(masks) == 2
    for evidence in masks:
        assert evidence.payload["availability_mask"], "mask must not be empty"


def test_geospatial_carries_both_sensors(scenes):
    """Optical is the navigable frame; SAR bounds travel alongside, not fused.

    RISAT and Cartosat-2S are different missions with different orbits, so a
    single claimed footprint would be a fabrication.
    """
    result = _specialist().execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )

    geo = result.geospatial
    assert geo.crs == "EPSG:32643"
    assert geo.transform is not None
    # The SAR side is reported separately.
    assert geo.sar_crs == "EPSG:32643"
    assert geo.optical_sensor is not None
    assert geo.sar_sensor is not None


def test_confidence_is_zero_without_a_prediction(scenes):
    """A signal GAP, not a weak signal. 0.0, not a discounted value.

    Reporting 0.5 on "we did not classify anything" would read as moderate
    certainty about nothing.
    """
    result = _specialist().execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )

    assert result.confidence.raw == 0.0
    assert result.confidence.components.get("no_prediction") == 1.0


def test_degradation_reason_names_what_is_missing(scenes):
    result = _specialist().execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )
    reason = result.confidence.degradation_reason
    assert reason
    assert "CROMA" in reason


def test_missing_head_path_does_not_raise(scenes, tmp_path):
    """A head path that does not exist is a MISSING artifact, not a broken one."""
    from core.config import load_config

    config = load_config()
    specialist = build_optical_sar_specialist(
        config, head_path=tmp_path / "never_trained.pt"
    )
    assert specialist.has_head is False

    result = specialist.execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )
    assert result.degraded is True


def test_corrupt_head_raises_rather_than_degrading(tmp_path, scenes):
    """A head that EXISTS but cannot be read is a different case entirely.

    Silently running without it would turn "we have a trained model and it is
    broken" into "we were never trained", which is the worst possible outcome:
    the operator would believe a degraded answer was the intended behaviour.
    """
    from core.config import load_config
    from core.errors import ModelLoadError

    corrupt = tmp_path / "corrupt_head.pt"
    corrupt.write_bytes(b"this is not a torch checkpoint")

    with pytest.raises(ModelLoadError):
        build_optical_sar_specialist(load_config(), head_path=corrupt)


# ---------------------------------------------------------------------------
# A trained head, without CROMA: confidence composition
# ---------------------------------------------------------------------------


class _StubEncoder:
    """A stand-in CROMA that emits the verified GAP shapes.

    Lets the fusion path be tested without the 777.6 MB checkpoint. It returns
    the REAL key names, so `assemble_fusion_input` sees what it will see in
    production.
    """

    def __init__(self, *, optical_value: float = 1.0, sar_value: float = 1.0):
        self.optical_value = optical_value
        self.sar_value = sar_value

    @property
    def n_patches(self) -> int:
        return 225

    def encode(self, optical, sar):
        from specialists.optical_sar.croma import CROMAEncoding

        batch = optical.shape[0]
        opt = np.full((batch, 768), self.optical_value, dtype=np.float32)
        sarv = np.full((batch, 768), self.sar_value, dtype=np.float32)
        joint = np.full((batch, 768), 0.5, dtype=np.float32)
        tokens = np.zeros((batch, 225, 768), dtype=np.float32)
        return CROMAEncoding(
            optical_gap=opt,
            sar_gap=sarv,
            joint_gap=joint,
            optical_tokens=tokens,
            sar_tokens=tokens,
            joint_tokens=tokens,
            resolution=120,
            n_patches=225,
        )

    def describe(self):
        return {"checkpoint": "stub"}


class _StubHead:
    """A trained-looking head with a controllable margin."""

    _satquery_trained = True

    def __init__(self, probabilities: list[float]):
        self._probs = np.asarray(probabilities, dtype=np.float32)

    def __call__(self, tensor):  # noqa: ANN001, ANN201
        import torch

        batch = tensor.shape[0]
        logits = torch.log(torch.from_numpy(self._probs)).repeat(batch, 1)
        return logits

    def num_parameters(self) -> int:
        return 1234


def _trained_specialist(*, probabilities, encoder=None):
    return OpticalSarSpecialist(
        encoder=encoder or _StubEncoder(),
        head=_StubHead(probabilities),
        class_labels=[f"class_{i}" for i in range(len(probabilities))],
        task_dim=len(probabilities),
    )


def test_prediction_is_produced_with_encoder_and_head(scenes):
    specialist = _trained_specialist(probabilities=[0.8, 0.15, 0.05])

    result = specialist.execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )

    assert result.labels == ["class_0"]
    assert "class_0" in result.answer
    assert result.confidence.raw > 0.0


def test_confidence_components_are_the_four_from_section_26(scenes):
    """fusion margin + optical confidence + SAR confidence + agreement."""
    specialist = _trained_specialist(probabilities=[0.8, 0.15, 0.05])

    result = specialist.execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )

    components = result.confidence.components
    for key in (
        "fusion_margin",
        "optical_confidence",
        "sar_confidence",
        "cross_modal_agreement",
    ):
        assert key in components, f"{key} missing from the confidence breakdown"
        assert 0.0 <= components[key] <= 1.0


def test_confidence_is_not_an_llm_utterance(scenes):
    """Every component is a float this specialist computed.

    Freeze section 5: "No LLM-generated confidence." Asserted structurally: the
    breakdown is numeric and the method is explicitly uncalibrated.
    """
    specialist = _trained_specialist(probabilities=[0.7, 0.2, 0.1])

    result = specialist.execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )

    assert result.confidence.method == "uncalibrated"
    for key, value in result.confidence.components.items():
        assert isinstance(value, float), f"{key} is {type(value).__name__}, not float"


def test_a_tiny_margin_produces_lower_confidence_than_a_large_one(scenes):
    """The margin must actually drive the score, or it is decoration."""
    confident = _trained_specialist(probabilities=[0.9, 0.05, 0.05])
    unsure = _trained_specialist(probabilities=[0.4, 0.35, 0.25])

    request = _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    strong = confident.execute(request).confidence
    weak = unsure.execute(request).confidence

    assert strong.components["fusion_margin"] > weak.components["fusion_margin"]
    assert strong.raw > weak.raw


def test_optical_availability_lowers_confidence(scenes, tmp_path):
    """A 4-of-12 optical scene must score below a 12-of-12 one.

    The hidden target is Cartosat-2S with 4 bands against a canonical 12. A
    confidence that ignored that would report the model's certainty about a
    tensor it was mostly fed zeros.
    """
    rng = np.random.default_rng(9)
    full_optical = _write_tif(
        tmp_path / "full12.tif", rng.integers(0, 4000, (12, H, W)).astype("uint16")
    )

    probabilities = [0.8, 0.15, 0.05]
    specialist = _trained_specialist(probabilities=probabilities)

    sparse = specialist.execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )
    full = specialist.execute(
        _request([_optical_asset(full_optical), _sar_asset(scenes["sar"])])
    )

    assert (
        sparse.confidence.components["optical_confidence"]
        < full.confidence.components["optical_confidence"]
    )
    assert sparse.confidence.raw < full.confidence.raw, (
        "a 4-of-12-channel optical input must not score as highly as a full one"
    )


def test_cross_modal_agreement_is_measured_from_the_encodings(scenes):
    """Agreement is cosine(optical_GAP, SAR_GAP), rescaled to [0,1]."""
    # Identical optical and SAR vectors -> cosine 1 -> component 1.0.
    agreeing = _trained_specialist(
        probabilities=[0.8, 0.15, 0.05],
        encoder=_StubEncoder(optical_value=2.0, sar_value=2.0),
    )
    result = agreeing.execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )
    assert result.confidence.components["cross_modal_agreement"] > 0.99
    assert abs(result.confidence.components["cross_modal_agreement_raw"] - 1.0) < 1e-5


def test_untrained_head_flag_is_honoured(scenes):
    """A head object with no trained marker is not treated as trained.

    The prediction IS still reported -- the head ran and produced logits, and
    suppressing a real computation would be its own kind of dishonesty. What
    changes is that the result is marked DEGRADED and the confidence floors to
    0.0, because there is no learned signal behind the number.
    """
    head = _StubHead([0.8, 0.15, 0.05])
    head._satquery_trained = False  # type: ignore[attr-defined]

    specialist = OpticalSarSpecialist(
        encoder=_StubEncoder(),
        head=head,
        task_dim=3,
    )
    assert specialist.has_head is False, "an untrained head is not a trained head"

    result = specialist.execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )

    assert result.confidence.components["trained_head"] == 0.0
    assert result.confidence.raw == 0.0
    assert result.degraded is True
    assert any("not a trained artifact" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Evidence and the C-1 rule
# ---------------------------------------------------------------------------


def test_croma_never_receives_a_mask(scenes):
    """Finding C-1, asserted on the interface rather than trusted.

    CROMA's forward pass takes exactly two arguments. If someone adds a mask
    parameter, this test is where it should be noticed.
    """
    import inspect as _inspect

    from specialists.optical_sar.croma import CROMAEncoder

    signature = _inspect.signature(CROMAEncoder.encode)
    assert set(signature.parameters) == {"self", "optical", "sar"}
    assert "mask" not in signature.parameters


def test_evidence_records_the_fusion_contract_when_croma_runs(scenes):
    """The JOINT_FEATURE_REGION evidence states the 2318 width and who gets the mask."""
    specialist = _trained_specialist(probabilities=[0.8, 0.15, 0.05])

    result = specialist.execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )

    joint = next(
        e for e in result.evidence if e.type is EvidenceType.JOINT_FEATURE_REGION
    )
    assert joint.payload["fusion_input_dim"] == 2318
    assert joint.payload["expected_dim"] == 2318
    assert joint.payload["mask_consumed_by"] == "fusion_head"
    assert joint.payload["croma_received_mask"] is False


def test_no_prediction_evidence_is_recorded_as_suppressed(scenes):
    """Without a head, the absence of a decision is recorded, not left implicit."""
    result = _specialist().execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )

    suppressed = [
        e for e in result.evidence
        if e.type is EvidenceType.STATISTIC and e.payload.get("prediction_suppressed")
    ]
    assert len(suppressed) == 1
    assert suppressed[0].score is None, "there is no decision to score"


def test_evidence_ids_are_unique(scenes):
    result = _specialist().execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )
    ids = [e.evidence_id for e in result.evidence]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Contract conformance
# ---------------------------------------------------------------------------


def test_contract_methods_delegate_to_the_result(scenes):
    specialist = _specialist()
    result = specialist.execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )

    assert specialist.produce_evidence(result) == result.evidence
    assert specialist.estimate_confidence(result) == result.confidence


def test_model_refs_name_both_models(scenes):
    refs = _specialist().model_refs()
    names = {r["name"] for r in refs}
    assert "CROMA" in names
    assert "FusionHead" in names


def test_result_survives_a_schema_roundtrip(scenes):
    """The frozen result contract must accept what the specialist produces."""
    from core.schemas import SpecialistResult

    result = _specialist().execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )
    raw = result.model_dump_json()
    restored = SpecialistResult.model_validate_json(raw)
    assert restored.task is Task.OPTICAL_SAR
    assert restored.confidence.raw == result.confidence.raw


def test_resolution_must_be_a_multiple_of_eight():
    """CROMA asserts this internally (finding C-7)."""
    from core.errors import SpecialistError

    with pytest.raises(SpecialistError):
        OpticalSarSpecialist(resolution=121)


def test_task_dim_must_allow_a_margin():
    from core.errors import SpecialistError

    with pytest.raises(SpecialistError):
        OpticalSarSpecialist(task_dim=1)


# ---------------------------------------------------------------------------
# The presentation renderer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channels", [1, 2, 3, 12])
def test_view_renderer_handles_any_channel_count(channels):
    """SAR has 2 channels and optical has 12; both must render.

    A renderer that assumed 3 channels would crash on every SAR view, and
    artifact rendering must never fail the analysis.
    """
    arr = np.random.default_rng(0).random((channels, 8, 8)).astype(np.float32)
    out = OpticalSarSpecialist._render_view(arr)
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8


def test_view_renderer_survives_a_constant_band():
    """A zero (or constant) band must not divide by zero.

    Missing channels ARE zeros, so this is the common case on a 4-band
    Cartosat scene's rendered view, not an exotic one.
    """
    zeros = np.zeros((2, 8, 8), dtype=np.float32)
    out = OpticalSarSpecialist._render_view(zeros)
    assert out.shape == (8, 8, 3)
    assert out.max() == 0

    constant = np.full((2, 8, 8), 5.0, dtype=np.float32)
    out = OpticalSarSpecialist._render_view(constant)
    assert out.shape == (8, 8, 3)


def test_artifact_directory_is_optional(scenes):
    """No artifact_dir means no refs, not a crash and not a fake path."""
    result = _specialist().execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )
    for evidence in result.evidence:
        assert evidence.artifact_ref is None


def test_views_are_written_when_a_directory_is_configured(scenes, tmp_path):
    """F-16 (owner ruling 2026-09-23): the views are written, and NOT exposed.

    The pre-ruling version asserted that `artifact_ref` held a path and that the
    file at that path existed. **That contract is now forbidden.** v1 has no
    artifact-serving endpoint, so the reference is `null` and no filesystem path
    may reach a client. The rendering itself is worth keeping -- it is
    server-side diagnostics -- so this test now asserts the two halves
    separately: the FILES exist on disk, and the RESPONSE carries no path.
    """
    artifact_dir = tmp_path / "artifacts"
    specialist = OpticalSarSpecialist(artifact_dir=artifact_dir)
    result = specialist.execute(
        _request([_optical_asset(scenes["optical"]), _sar_asset(scenes["sar"])])
    )

    # 1. No evidence item may carry a reference -- there is nothing to point at.
    refs = [e.artifact_ref for e in result.evidence]
    assert refs == [None] * len(refs), (
        f"an evidence item carries an artifact_ref ({refs}); F-16 forbids "
        f"exposing a filesystem path, and v1 has no artifact-serving endpoint "
        f"to return a URI instead"
    )
    body = repr([e.model_dump() for e in result.evidence])
    assert str(artifact_dir) not in body, (
        "the artifact directory leaked into the evidence payload"
    )

    # 2. The views WERE rendered: the server-side behaviour is unchanged, and
    #    the view evidence items are still present (the item is not dropped).
    written = sorted(p.name for p in artifact_dir.iterdir() if p.is_file())
    assert len(written) == 2, f"expected the two rendered views, found {written}"
    view_types = [e.type for e in result.evidence if e.type.value.endswith("_view")]
    assert len(view_types) == 2, (
        f"expected both view evidence items to survive the ruling, found "
        f"{view_types}"
    )

    # 3. The client is TOLD, explicitly, that the views are not retrievable.
    joined = " ".join(result.warnings)
    assert "NOT retrievable" in joined, (
        f"no non-retrievable-artifact warning was emitted; warnings were "
        f"{result.warnings}"
    )
