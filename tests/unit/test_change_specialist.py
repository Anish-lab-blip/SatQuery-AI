"""Phase 9 — the change specialist serving contract.

`specialists/change/` had a model, post-processing and metrics that all passed
their tests, and NO `Specialist` subclass. The controller therefore could not
dispatch to change detection at all, even though the architecture freeze
(section 2.4) makes bi-temporal change a first-class workflow.

These tests pin the parts of that contract that are easy to get wrong and
expensive to discover late:

  * the two-asset requirement (0/1/3 all rejected, with a typed error)
  * poor co-registration LOWERS CONFIDENCE AND SUPPRESSES SPATIAL CLAIMS,
    rather than silently proceeding
  * evidence is emitted, and a withheld claim is recorded rather than dropped
  * the no-checkpoint deployment case DEGRADES instead of crashing

The tests use real GeoTIFFs on disk, because the specialist reads them through
`preprocessing.imagery` and a mock would not exercise that path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import rasterio
from affine import Affine

from core.errors import (
    InvalidRequestError,
    ModelLoadError,
    TemporalPairError,
)
from core.schemas import EvidenceType, Task
from specialists.base import SpecialistRequest
from specialists.change.specialist import (
    ChangeSpecialist,
    build_change_specialist,
)

pytest.importorskip("torch")

H = W = 256


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_tif(
    path,
    array: np.ndarray,
    *,
    crs: str | None = "EPSG:32643",
    resolution: float = 10.0,
    x0: float = 500_000.0,
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
        profile["transform"] = Affine(resolution, 0, x0, 0, -resolution, 2_500_000.0)
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(array)
    return path


@pytest.fixture(scope="module")
def rasters(tmp_path_factory) -> dict:
    """A co-registered T1/T2 pair with a KNOWN change block.

    The block is a real intensity jump, so once a trained checkpoint exists
    this fixture would detect it. With an untrained model it only exercises the
    plumbing, which is what these tests are about.
    """
    root = tmp_path_factory.mktemp("change_specialist")
    rng = np.random.default_rng(0)

    t1 = rng.integers(500, 2500, (3, H, W)).astype("uint16")
    t2 = t1.copy()
    t2[:, 60:120, 60:120] = 3500  # the change

    # A mis-registered pair: the same scene translated by more than the
    # max-shift limit, which is what a broken georeferencing looks like.
    shifted = np.roll(t1, (12, 9), axis=(1, 2))

    return {
        "t1": _write_tif(root / "t1.tif", t1),
        "t2": _write_tif(root / "t2.tif", t2),
        "shifted": _write_tif(root / "shifted.tif", shifted),
        "plain_t1": _write_tif(root / "plain_t1.tif", t1, crs=None),
        "plain_t2": _write_tif(root / "plain_t2.tif", t2, crs=None),
    }


@pytest.fixture()
def specialist(rasters, tmp_path) -> ChangeSpecialist:
    """The no-checkpoint specialist -- the path that is actually deployed."""
    from core.config import load_config

    return build_change_specialist(
        load_config(),
        checkpoint_path=None,          # no trained artifact exists yet
        artifact_dir=tmp_path / "art",
        device="cpu",
    )


def _assets(rasters: dict, *keys: str):
    from preprocessing.raster import inspect_raster

    return [inspect_raster(rasters[k]) for k in keys]


def _request(rasters: dict, *keys: str, query: str = "what changed"):
    return SpecialistRequest(assets=_assets(rasters, *keys), query=query)


# ---------------------------------------------------------------------------
# 1. The two-asset requirement (spec failure matrix section 57)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "keys,label",
    [
        ((), "zero"),
        (("t1",), "one"),
        (("t1", "t2", "t1"), "three"),
    ],
)
def test_wrong_asset_count_is_rejected(specialist, rasters, keys, label) -> None:
    """0, 1 and 3 assets must all raise, and raise the TYPED error.

    A single image is the common mistake (the caller treated change as a
    single-image task); three is the other (a time series was attached).
    Neither is silently coerced to "use the first two", which would answer a
    question the caller did not ask.
    """
    request = SpecialistRequest(assets=_assets(rasters, *keys), query="q")

    with pytest.raises(InvalidRequestError) as excinfo:
        specialist.validate_request(request)

    # The user-facing reason must name the count: "invalid request" alone does
    # not tell an operator what to fix.
    assert "2" in excinfo.value.detail
    assert excinfo.value.code == "invalid_request"
    assert "two images" in excinfo.value.user_message.lower()


def test_two_assets_are_accepted(specialist, rasters) -> None:
    specialist.validate_request(_request(rasters, "t1", "t2"))


def test_missing_asset_path_is_rejected(specialist, rasters) -> None:
    """A path that does not exist is an InvalidRequest, not a crash later."""
    from core.schemas import AssetMetadata

    request = SpecialistRequest(
        assets=[
            _assets(rasters, "t1")[0],
            AssetMetadata(path=str(rasters["t1"]).replace("t1.tif", "nope.tif")),
        ],
        query="q",
    )
    with pytest.raises(InvalidRequestError):
        specialist.validate_request(request)


def test_identical_acquisitions_are_rejected(specialist, rasters) -> None:
    """The same file twice is not a temporal pair.

    Feeding T1 as both inputs would produce a map of pure noise presented as a
    change detection. It is caught before the model is ever called.
    """
    with pytest.raises(TemporalPairError) as excinfo:
        specialist.validate_request(_request(rasters, "t1", "t1"))

    assert excinfo.value.code == "temporal_pair_invalid"


# ---------------------------------------------------------------------------
# 2. Poor registration lowers confidence and SUPPRESSES spatial claims
# ---------------------------------------------------------------------------


def test_misregistered_pair_is_measured_unusable(specialist, rasters) -> None:
    """The shift is detected and the pair is declared unusable."""
    from preprocessing.imagery import load_image_array
    from specialists.change.postprocess import measure_registration

    a = load_image_array(rasters["t1"])
    b = load_image_array(rasters["shifted"])
    quality = measure_registration(a, b, max_shift_px=8)

    assert quality.is_usable is False
    assert quality.shift_magnitude > 8


def test_poor_registration_suppresses_regions_but_keeps_the_map(
    specialist, rasters
) -> None:
    """Spatial claims are WITHHELD; the change map itself is still computed.

    The spec is explicit: "Never convert poor registration into fake
    certainty." Regions are assertions about WHERE the ground changed, and an
    unregistered pair cannot support one. The map is retained because a caller
    may legitimately want to look at it -- what is forbidden is presenting it
    as a located detection.
    """
    result = specialist.execute(_request(rasters, "t1", "shifted"))

    assert result.regions == [], "regions were emitted for an unusable pair"
    assert result.degraded is True
    assert result.confidence.components.get("suppressed_by_registration") == 1.0
    assert result.confidence.raw == 0.0

    # The suppression is STATED, not silent.
    joined = " ".join(result.warnings).lower()
    assert "co-registration" in joined or "co-registered" in joined
    assert "suppress" in joined

    # The geospatial transform is withheld too: it would invite the caller to
    # place a region on a map we just said is not placeable.
    assert result.geospatial.is_georeferenced is False


def test_well_registered_pair_is_not_suppressed(specialist, rasters) -> None:
    """The negative case: a good pair must NOT be flagged as suppressed."""
    result = specialist.execute(_request(rasters, "t1", "t2"))

    assert "suppressed_by_registration" not in result.confidence.components
    assert result.confidence.components["registration_quality"] > 0.9


def test_registration_quality_drives_confidence_to_zero(specialist, rasters) -> None:
    """Two independent penalties, and the worse one wins.

    On the untrained path BOTH results floor at raw 0.0 -- the untrained signal
    gap is a floor in its own right -- so the raw score cannot distinguish
    them. What distinguishes them is the registration COMPONENT and the
    explicit suppression flag: a good pair scores ~1.0 there, a mis-registered
    one scores 0.0 and is marked suppressed. Asserting on `raw` alone would
    have been asserting a difference this configuration cannot produce.
    """
    good = specialist.execute(_request(rasters, "t1", "t2"))
    bad = specialist.execute(_request(rasters, "t1", "shifted"))

    # The registration signal itself must separate the two.
    assert good.confidence.components["registration_quality"] > 0.9
    assert bad.confidence.components["registration_quality"] == 0.0

    # And the mis-registered pair must be flagged, not merely scored low.
    assert "suppressed_by_registration" not in good.confidence.components
    assert bad.confidence.components["suppressed_by_registration"] == 1.0

    # Both floors are 0.0: a signal gap is not a weak signal.
    assert bad.confidence.raw == 0.0
    assert good.confidence.raw == 0.0


# ---------------------------------------------------------------------------
# 3. No-checkpoint degradation (the path that will actually be exercised)
# ---------------------------------------------------------------------------


def test_no_checkpoint_degrades_and_says_so(specialist, rasters) -> None:
    """A missing trained artifact is a DEPLOYMENT case, not a crash.

    Phase 8 set this precedent for the grounding head: the Space may ship
    without the checkpoint, and refusing to run would make it useless instead
    of merely less precise. What is NOT acceptable is presenting a randomly
    initialised change map as a detection.
    """
    assert specialist.has_checkpoint is False

    result = specialist.execute(_request(rasters, "t1", "t2"))

    assert result.degraded is True
    joined = " ".join(result.warnings).lower()
    assert "no trained change checkpoint" in joined
    assert "randomly initialised" in joined

    # The untrained state must be visible as NUMBERS, not only as a warning.
    assert result.confidence.components.get("trained_checkpoint") == 0.0
    assert result.confidence.raw == 0.0


def test_no_checkpoint_does_not_raise(specialist, rasters) -> None:
    """The whole point of degradation: `execute` returns a result."""
    result = specialist.execute(_request(rasters, "t1", "t2"))
    assert isinstance(result.task, Task)
    assert result.task is Task.CHANGE


def test_corrupt_checkpoint_raises_rather_than_silently_degrading(
    rasters, tmp_path
) -> None:
    """A NAMED checkpoint that cannot be read is a corrupt artifact.

    Silently falling back to an untrained model because a real checkpoint
    failed to load would be the worst outcome: the operator would believe a
    trained detection had run. The two cases must not be confused.
    """
    from core.config import load_config

    bad = tmp_path / "corrupt.pt"
    bad.write_bytes(b"this is not a checkpoint")

    with pytest.raises(ModelLoadError):
        build_change_specialist(load_config(), checkpoint_path=bad)


def test_checkpoint_is_reported_in_model_refs(rasters, tmp_path) -> None:
    """The trace must distinguish a trained run from an untrained one."""
    from core.config import load_config
    from specialists.change.stanet import (
        STANetStyleChangeDetector,
        save_change_model,
    )

    ckpt = tmp_path / "change.pt"
    save_change_model(ckpt, STANetStyleChangeDetector(width=64, pretrained=False))

    trained = build_change_specialist(load_config(), checkpoint_path=ckpt, device="cpu")
    assert trained.has_checkpoint is True
    refs = trained.model_refs()
    assert refs and "trained" in refs[0]["revision"].lower()

    untrained = build_change_specialist(load_config(), checkpoint_path=None, device="cpu")
    assert untrained.has_checkpoint is False
    assert "untrained" in untrained.model_refs()[0]["revision"].lower()


# ---------------------------------------------------------------------------
# 4. Evidence emission
# ---------------------------------------------------------------------------


def test_evidence_is_emitted(specialist, rasters) -> None:
    result = specialist.execute(_request(rasters, "t1", "t2"))

    assert result.evidence, "no evidence was produced"
    types = {e.type for e in result.evidence}
    assert EvidenceType.CHANGE_MAP in types
    assert EvidenceType.STATISTIC in types
    for item in result.evidence:
        assert item.source_specialist == "change"


def test_change_map_evidence_carries_the_threshold_and_counts(
    specialist, rasters
) -> None:
    result = specialist.execute(_request(rasters, "t1", "t2"))
    change_map = next(
        e for e in result.evidence if e.type is EvidenceType.CHANGE_MAP
    )
    assert change_map.payload["threshold"] == pytest.approx(specialist.threshold)
    assert "total_change_pixels" in change_map.payload
    assert "n_components_kept" in change_map.payload


def test_suppressed_pair_records_the_withheld_claim(specialist, rasters) -> None:
    """Dropping a claim silently would hide that a detection was discarded.

    The record is a STATISTIC, deliberately NOT a BOUNDING_BOX: emitting the
    box as spatial evidence would defeat the suppression it describes.
    """
    result = specialist.execute(_request(rasters, "t1", "shifted"))

    assert not any(
        e.type is EvidenceType.BOUNDING_BOX for e in result.evidence
    ), "a withheld region leaked into the evidence as spatial"

    stats = [e for e in result.evidence if e.type is EvidenceType.STATISTIC]
    suppression = [
        e for e in stats
        if e.payload.get("spatial_claims_suppressed") or
        e.payload.get("usable_for_spatial_claims") is False
    ]
    assert suppression, "the suppression was not recorded anywhere"


def test_suppressed_change_map_artifact_loses_its_georeferencing(
    specialist, rasters
) -> None:
    """The written map must not carry T1's CRS when the pair is suppressed.

    Copying the transform onto the artifact would produce a file that unlocks
    exactly the placed-on-a-map reading the suppression forbids: a downstream
    GIS tool would render it over the wrong ground without complaint. The
    pixels stay inspectable; the location claim does not.

    F-16 (owner ruling 2026-09-23): the map is written server-side but its path
    is NOT published. `result.change_map` is `None` and the CHANGE_MAP evidence
    carries no `artifact_ref`, so the file is located through the specialist's
    own configured directory -- where an operator would look for it anyway.

    The pre-ruling version of this test read the path out of `result.change_map`
    and opened it, which pinned the disclosure as the specification. The
    property under test was always about the FILE, not about the ref.
    """
    result = specialist.execute(_request(rasters, "t1", "shifted"))
    assert result.change_map is None, (
        f"a filesystem path was published as change_map ({result.change_map!r}); "
        "F-16 forbids exposing paths"
    )

    written = sorted(specialist.artifact_dir.glob("change_map_*.tif"))
    assert written, "no artifact was written to inspect"

    with rasterio.open(written[0]) as ds:
        assert ds.crs is None
        assert ds.transform.is_identity

    # The well-registered artifact DOES keep its georeferencing. It is written
    # to the same name (the stem comes from T1), so it must be inspected before
    # any later run overwrites it.
    good = specialist.execute(_request(rasters, "t1", "t2"))
    assert good.change_map is None
    again = sorted(specialist.artifact_dir.glob("change_map_*.tif"))
    assert again, "the well-registered run wrote no artifact"
    with rasterio.open(again[0]) as ds:
        assert ds.crs is not None


def test_the_change_carrier_never_publishes_a_path(specialist, rasters) -> None:
    """F-16's second carrier, guarded at the sibling site of the same ruling.

    `specialists/optical_sar` publishes the view refs; `change` publishes the
    change-map ref and `result.change_map`. The owner ruling covers both ("never
    expose filesystem paths"), but the optical-SAR fix alone left this one
    live -- measured, not inferred: with `change.artifact_dir` configured, the
    pre-ruling envelope carried the full server-side path in BOTH
    `result.change_map` and the CHANGE_MAP evidence's `artifact_ref`.

    This guard fails on that state. It also asserts the two things that must
    NOT be lost while removing the path: the map is still written, and the
    client is still given the change statistics plus an explicit warning that
    the artifact is not retrievable.
    """
    import json

    result = specialist.execute(_request(rasters, "t1", "t2"))
    body = json.dumps(result.model_dump(mode="json"), default=str)

    assert str(specialist.artifact_dir) not in body, (
        "the configured artifact directory leaked into the client-facing result"
    )
    assert result.change_map is None, (
        f"result.change_map carries {result.change_map!r}; F-16 forbids paths"
    )
    offending = [e.artifact_ref for e in result.evidence if e.artifact_ref is not None]
    assert not offending, (
        f"evidence items carry artifact refs ({offending}); F-16 forbids paths"
    )
    assert "artifact://" not in body, (
        "an artifact:// URI was fabricated; v1 has no artifact-serving endpoint"
    )

    # The control: removing the disclosure must not remove the work.
    assert sorted(specialist.artifact_dir.glob("change_map_*.tif")), (
        "the change map stopped being written server-side"
    )
    change_map_evidence = next(
        e for e in result.evidence if e.type is EvidenceType.CHANGE_MAP
    )
    assert change_map_evidence.payload["total_change_pixels"] is not None, (
        "the change statistics were lost along with the path"
    )
    assert any("NOT retrievable" in w for w in result.warnings), (
        f"no non-retrievable-artifact warning was emitted; warnings={result.warnings}"
    )


def test_produce_evidence_and_estimate_confidence_roundtrip(
    specialist, rasters
) -> None:
    """The two interrogation methods must return what `execute` built."""
    result = specialist.execute(_request(rasters, "t1", "t2"))
    assert specialist.produce_evidence(result) == result.evidence
    assert specialist.estimate_confidence(result) == result.confidence


# ---------------------------------------------------------------------------
# 5. Contract conformance
# ---------------------------------------------------------------------------


def test_is_a_specialist_with_the_change_capability(specialist) -> None:
    from specialists.base import Specialist

    assert isinstance(specialist, Specialist)
    assert specialist.capabilities == ("change",)
    assert specialist.supports("change")


def test_result_task_is_change(specialist, rasters) -> None:
    result = specialist.execute(_request(rasters, "t1", "t2"))
    assert result.task is Task.CHANGE


def test_no_crs_pair_warns_but_still_runs(specialist, rasters) -> None:
    """A missing CRS degrades to pixel coordinates; it does not abort."""
    result = specialist.execute(_request(rasters, "plain_t1", "plain_t2"))
    joined = " ".join(result.warnings).lower()
    assert "crs" in joined
    assert result.task is Task.CHANGE


def test_confidence_weights_sum_to_one() -> None:
    """The components are a convex combination before the gates are applied."""
    from specialists.change.specialist import CONFIDENCE_WEIGHTS

    assert sum(CONFIDENCE_WEIGHTS.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 6. The weighted-sum path itself (needs a detector that emits regions)
# ---------------------------------------------------------------------------


class _ConstantDetector:
    """A detector stub returning a FIXED probability map.

    The untrained real model emits almost nothing above 0.5 (its output bias is
    initialised to -2.0), so it never exercises the weighted-sum composition:
    every path floors to 0.0 first. This stub returns a map with a real,
    strong region so the composition -- weighted sum, then minimum against the
    registration gate -- is actually run.
    """

    def __init__(self, value: float = 0.95, block=(30, 90, 30, 90)) -> None:
        self.value = value
        self.block = block
        self._satquery_trained = True

    def eval(self) -> None:
        return None

    def num_parameters(self) -> int:
        return 0

    def __call__(self, t1, t2):  # noqa: ANN001
        import torch

        h, w = t1.shape[-2:]
        prob = torch.zeros(1, 1, h, w)
        r0, r1, c0, c1 = self.block
        prob[:, :, r0:r1, c0:c1] = self.value
        return type("Out", (), {"probabilities": prob, "logits": prob})()


def test_a_strong_region_produces_positive_confidence(rasters, tmp_path) -> None:
    """The composition must be reachable: a real region scores above zero."""
    spec = ChangeSpecialist(
        model=_ConstantDetector(),
        threshold=0.5,
        min_component_pixels=32,
        artifact_dir=tmp_path / "art",
    )
    result = spec.execute(_request(rasters, "t1", "t2"))

    assert spec.has_checkpoint is True
    assert result.regions, "the stub's strong block produced no region"
    assert result.confidence.raw > 0.0
    assert result.confidence.components["registration_quality"] > 0.9
    assert result.confidence.components["mean_change_probability"] > 0.5

    # The regions carry normalized boxes derived from the real image size.
    # They are schema `Region`s, which do NOT carry area_pixels -- the change
    # specific fields live on ChangeRegion and are surfaced in the evidence
    # payload instead, because SpecialistResult.regions is typed list[Region].
    for region in result.regions:
        assert 0.0 <= region.box.x1 <= region.box.x2 <= 1.0
        assert 0.0 <= region.box.y1 <= region.box.y2 <= 1.0

    boxes = [
        e for e in result.evidence if e.type is EvidenceType.BOUNDING_BOX
    ]
    assert boxes, "regions were emitted without bounding-box evidence"
    assert all(e.payload["area_pixels"] > 0 for e in boxes)


def test_alignment_is_a_gate_not_a_vote(rasters, tmp_path) -> None:
    """A bright, stable region must NOT survive a mis-registered pair.

    This is the load-bearing property of the composition. An unregistered pair
    produces edge-change everywhere -- which is exactly the bright, stable
    output that would win a naive weighted sum. Taking the minimum against the
    registration factor is what stops it.
    """
    spec = ChangeSpecialist(
        model=_ConstantDetector(),
        threshold=0.5,
        min_component_pixels=32,
    )
    good = spec.execute(_request(rasters, "t1", "t2"))
    bad = spec.execute(_request(rasters, "t1", "shifted"))

    # The SAME bright map in both cases; only the alignment differs.
    assert good.confidence.raw > 0.0
    assert bad.confidence.raw == 0.0
    assert bad.regions == []


def test_validate_request_is_the_single_source_of_pair_truth(
    specialist, rasters
) -> None:
    """`execute` must not validate differently from `validate_request`.

    They share `_assess_pair`, so registration is measured once and the
    validation verdict and the executed verdict cannot disagree.
    """
    request = _request(rasters, "t1", "shifted")
    specialist.validate_request(request)          # must not raise
    result = specialist.execute(request)
    assert result.confidence.components.get("suppressed_by_registration") == 1.0


# ---------------------------------------------------------------------------
# 7. Serving wiring — the registry `builders=` seam
# ---------------------------------------------------------------------------
#
# The trained head is deliberately NOT wired by config: `change.checkpoint_path`
# is absent from `base.yaml`, because populating it would move `Config.hash` and
# break the checkpoint's drift check (`scripts/eval_change.py` exits 3). The
# supported wiring path is the registry `builders=` override, which injects the
# checkpoint at the call site. These two tests pin that seam WITHOUT loading the
# 63 MB head -- the same reason the e2e test substitutes builders.

#: The real artifact the override is meant to inject. NEVER loaded here.
_CHANGE_HEAD = "artifacts/change/levir_change_v001/head.pt"

#: The config hash the shipped head was trained under. Recorded in
#: `artifacts/change/levir_change_v001/model_metadata.json`.
_FROZEN_CONFIG_HASH = "78f1e3700da15aa1"


class _TrainedChangeStub:
    """Stands in for the product of `build_change_specialist`.

    Only what the registry reads is present: `capabilities` (asserted against
    the spec after construction) and the `has_checkpoint` flag
    (`specialists/change/specialist.py:173`) that classifies availability.
    """

    name = "change"
    version = "0.0.0-test"
    capabilities = ("change",)
    has_checkpoint = True


def test_registry_builders_override_reaches_the_change_builder() -> None:
    """The `builders=` seam must reach the builder carrying the checkpoint.

    A recording stub stands in for the real builder and must receive the
    `checkpoint_path` the override injects -- proving the seam carries the
    artifact the config cannot supply. The registry's own `device` kwarg must
    arrive too, proving the registry (not importlib) invoked the override.
    """
    from core.config import load_config
    from core.registry import RegistryState, SpecialistRegistry, default_specs

    cfg = load_config()
    recorded: dict[str, Any] = {}

    def _stub_builder(config: Any, **kw: Any) -> Any:
        recorded.update(kw)
        return _TrainedChangeStub()

    def _wired(config: Any, **kw: Any) -> Any:
        # Exactly the real wiring: inject the checkpoint, then delegate.
        kw["checkpoint_path"] = _CHANGE_HEAD
        return _stub_builder(config, **kw)

    registry = SpecialistRegistry(
        cfg, specs=default_specs(), builders={"change": _wired}
    )
    entry = registry.build("change")

    # The load-bearing assertion: the injected artifact reaches the builder.
    assert recorded.get("checkpoint_path") == _CHANGE_HEAD
    # The registry invoked the override and passed its own kwargs.
    assert recorded.get("device") == registry.device
    assert entry.specialist is not None and entry.specialist.has_checkpoint is True
    assert entry.state is RegistryState.AVAILABLE
    assert entry.degraded is False
    # The override is a call-site argument, not config: the hash must not move.
    assert cfg.hash == _FROZEN_CONFIG_HASH


def test_without_the_override_the_change_entry_is_degraded() -> None:
    """The default (orphaned) state is DEGRADED, and it is deliberate.

    With no override the registry resolves the real builder through importlib,
    and `change.checkpoint_path` does not resolve (`_builder_kwargs`,
    `core/registry.py:449-453`, passes nothing), so the specialist builds a
    randomly-initialised detector and reports DEGRADED. Pinning this records the
    current default on purpose rather than leaving it accidental: if someone
    later wires the checkpoint via config, this test tells them the serving
    default changed.
    """
    from core.config import load_config
    from core.registry import RegistryState, SpecialistRegistry, default_specs

    cfg = load_config()
    assert cfg.get("change.checkpoint_path") is None  # the root cause

    registry = SpecialistRegistry(cfg, specs=default_specs())  # no override
    entry = registry.build("change")

    assert entry.state is RegistryState.DEGRADED
    assert entry.specialist is not None
    assert entry.specialist.has_checkpoint is False
