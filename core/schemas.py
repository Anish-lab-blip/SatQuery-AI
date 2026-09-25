"""SatQuery AI — canonical typed schemas.

This module is the binding contract between every component. Per
docs/ARCHITECTURE_FREEZE.md section 3, no specialist may invent its own result shape.

Findings baked in:
  C-1  channel-availability mask is a first-class fusion input, never a CROMA input
  C-5  grounding carries an explicit coordinate_system and a documented token floor
  C-8  deployment durations are declared, not defaulted
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ===========================================================================
# Enumerations
# ===========================================================================
class Task(str, Enum):
    VQA = "vqa"
    CAPTION = "caption"
    GROUNDING = "grounding"
    CHANGE = "change"
    OPTICAL_SAR = "optical_sar"
    #: R-02. Two temporally corresponding assets plus a change-oriented question
    #: in, a short answer out. Distinct from CHANGE, which is the change
    #: *detector* and returns a spatial change map with no language output, and
    #: distinct from VQA, which answers about ONE asset. The three are not
    #: interchangeable and the planner must not substitute one for another.
    CHANGE_VQA = "change_vqa"
    UNSUPPORTED = "unsupported"


class Modality(str, Enum):
    OPTICAL = "optical"
    SAR = "sar"
    OPTICAL_SAR = "optical_sar"
    UNKNOWN = "unknown"


class CoordinateSystem(str, Enum):
    """Never omit this. A bare box is meaningless without it. (C-5)"""

    NORMALIZED_0_1 = "normalized_0_1"
    PIXEL = "pixel"
    GEO = "geo"


class EvidenceType(str, Enum):
    IMAGE_CROP = "image_crop"
    TILE = "tile"
    BOUNDING_BOX = "bounding_box"
    MASK = "mask"
    CHANGE_MAP = "change_map"
    OPTICAL_VIEW = "optical_view"
    SAR_VIEW = "sar_view"
    JOINT_FEATURE_REGION = "joint_feature_region"
    STATISTIC = "statistic"
    GEOLOCATION = "geolocation"
    AVAILABILITY_MASK = "availability_mask"   # C-1: modality trust evidence


class ControllerState(str, Enum):
    RECEIVE = "RECEIVE"
    PARSE = "PARSE"
    VALIDATE = "VALIDATE"
    PLAN = "PLAN"
    PREPROCESS = "PREPROCESS"
    EXECUTE = "EXECUTE"
    AGGREGATE = "AGGREGATE"
    VERIFY = "VERIFY"
    RESPOND = "RESPOND"


# ===========================================================================
# Router
# ===========================================================================
class Intent(BaseModel):
    """Output of the learned router. Advisory only — the controller decides."""

    model_config = ConfigDict(extra="forbid")

    task: Task
    modality: Modality = Modality.UNKNOWN
    temporal: bool = False
    spatial_output: bool = False
    language_output: bool = True
    confidence: float = Field(ge=0.0, le=1.0)
    source: Literal["learned", "lexical_fallback", "forced"] = "learned"

    @model_validator(mode="after")
    def _consistency(self) -> "Intent":
        # A temporal task without spatial output is legal; the reverse is not implied.
        if self.task in (Task.CHANGE, Task.CHANGE_VQA) and not self.temporal:
            self.temporal = True
        if self.task is Task.OPTICAL_SAR and self.modality is Modality.UNKNOWN:
            self.modality = Modality.OPTICAL_SAR
        return self


# ===========================================================================
# Geospatial metadata
# ===========================================================================
class GeoMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    crs: str | None = None
    transform: list[float] | None = Field(
        default=None, description="Affine transform coefficients, row-major (6 values)."
    )
    bounds: list[float] | None = Field(default=None, description="[minx, miny, maxx, maxy]")
    width: int | None = None
    height: int | None = None
    band_count: int | None = None
    dtype: str | None = None
    nodata: float | None = None
    resolution: list[float] | None = None
    has_crs: bool = False
    is_georeferenced: bool = False


class SensorDescriptor(BaseModel):
    """Frozen sensor-adapter contract (plan section 18)."""

    model_config = ConfigDict(extra="forbid")

    sensor: str
    available_bands: list[str] = Field(default_factory=list)
    band_map: dict[str, str] = Field(default_factory=dict)
    normalization: str = "percentile"
    availability_mask: list[bool] = Field(default_factory=list)
    resolution: float | None = None
    canonical_channels: int = 0
    missing_channels_zero_filled: bool = True


class AssetMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(default_factory=lambda: _new_id("asset"))
    path: str
    modality: Modality = Modality.UNKNOWN
    sha256: str | None = None
    geo: GeoMetadata = Field(default_factory=GeoMetadata)
    sensor: SensorDescriptor | None = None
    acquisition_date: str | None = None
    scene_id: str | None = None
    dataset_id: str | None = None
    split: str | None = None


# ===========================================================================
# Spatial results  (C-5: coordinate_system is mandatory, not optional)
# ===========================================================================
class Box(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x1: float
    y1: float
    x2: float
    y2: float
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    label: str | None = None
    coordinate_system: CoordinateSystem = CoordinateSystem.NORMALIZED_0_1

    @model_validator(mode="after")
    def _ordered(self) -> "Box":
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("box coordinates must satisfy x2>=x1 and y2>=y1")
        return self


class Region(BaseModel):
    """A spatial result that may or may not be a strict rectangle."""

    model_config = ConfigDict(extra="forbid")

    region_id: str = Field(default_factory=lambda: _new_id("region"))
    box: Box | None = None
    mask_ref: str | None = None
    label: str | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    coordinate_system: CoordinateSystem = CoordinateSystem.NORMALIZED_0_1


class ChangeRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region_id: str = Field(default_factory=lambda: _new_id("chg"))
    box: Box
    area_pixels: int = Field(ge=0)
    mean_probability: float = Field(ge=0.0, le=1.0)
    stability: float | None = Field(default=None, ge=0.0, le=1.0)
    coordinate_system: CoordinateSystem = CoordinateSystem.NORMALIZED_0_1


# ===========================================================================
# Evidence
# ===========================================================================
class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(default_factory=lambda: _new_id("ev"))
    type: EvidenceType
    source_specialist: str
    coordinate_system: CoordinateSystem | None = None
    coordinates: list[float] | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    artifact_ref: str | None = Field(
        default=None,
        description=(
            "Reference to an externally retrievable artifact. NEVER a "
            "filesystem path (F-16, owner ruling 2026-09-23): v1 exposes no "
            "artifact-serving endpoint, so this is null unless a deployment "
            "supplies a client-fetchable reference. An artifact may still be "
            "written server-side where configured; being written is not the "
            "same as being retrievable."
        ),
    )
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _spatial_needs_crs(self) -> "Evidence":
        spatial = {
            EvidenceType.BOUNDING_BOX,
            EvidenceType.MASK,
            EvidenceType.CHANGE_MAP,
            EvidenceType.TILE,
            EvidenceType.IMAGE_CROP,
            EvidenceType.JOINT_FEATURE_REGION,
        }
        if self.type in spatial and self.coordinates and self.coordinate_system is None:
            raise ValueError(
                f"evidence type '{self.type.value}' carries coordinates "
                "but no coordinate_system"
            )
        return self


# ===========================================================================
# Confidence
# ===========================================================================
class ConfidenceBreakdown(BaseModel):
    """Never an LLM utterance. Always measurable signals (plan section 26)."""

    model_config = ConfigDict(extra="forbid")

    raw: float = Field(ge=0.0, le=1.0)
    calibrated: float | None = Field(default=None, ge=0.0, le=1.0)
    method: str = "uncalibrated"
    components: dict[str, float] = Field(default_factory=dict)
    degraded: bool = False
    degradation_reason: str | None = None

    @property
    def value(self) -> float:
        return self.calibrated if self.calibrated is not None else self.raw


# ===========================================================================
# Execution trace  (observable facts only — never chain-of-thought)
# ===========================================================================
class TraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: ControllerState
    started_at: str = Field(default_factory=_utcnow)
    duration_ms: float | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class ModelRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    revision: str | None = None
    role: str | None = None


class ExecutionTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: _new_id("run"))
    schema_version: str = SCHEMA_VERSION
    task: Task | None = None
    query: str | None = None
    inputs: list[str] = Field(default_factory=list)
    modalities: list[Modality] = Field(default_factory=list)
    intent: Intent | None = None
    validation: dict[str, Any] = Field(default_factory=dict)
    workflow: list[str] = Field(default_factory=list)
    steps: list[TraceStep] = Field(default_factory=list)
    selected_models: list[ModelRef] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)
    confidence: ConfidenceBreakdown | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    fallbacks: list[str] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    contradiction: bool = False
    config_hash: str | None = None
    started_at: str = Field(default_factory=_utcnow)
    finished_at: str | None = None


# ===========================================================================
# Specialist result  (the master contract)
# ===========================================================================
class SpecialistResult(BaseModel):
    """Every specialist returns exactly this. No exceptions."""

    model_config = ConfigDict(extra="forbid")

    task: Task
    answer: str = ""
    labels: list[str] = Field(default_factory=list)
    regions: list[Region] = Field(default_factory=list)
    boxes: list[Box] = Field(default_factory=list)
    masks: list[str] = Field(default_factory=list, description="artifact refs")
    change_map: str | None = Field(default=None, description="artifact ref")
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: ConfidenceBreakdown
    geospatial: GeoMetadata = Field(default_factory=GeoMetadata)
    execution_trace: ExecutionTrace | None = None
    schema_version: str = SCHEMA_VERSION
    warnings: list[str] = Field(default_factory=list)
    degraded: bool = False

    @field_validator("evidence")
    @classmethod
    def _unique_evidence_ids(cls, v: list[Evidence]) -> list[Evidence]:
        ids = [e.evidence_id for e in v]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence_id values must be unique within a result")
        return v

    @model_validator(mode="after")
    def _task_output_consistency(self) -> "SpecialistResult":
        if self.task is Task.GROUNDING and not (self.boxes or self.regions):
            # Grounding with no localisation is a degraded result, not a crash.
            if not self.degraded:
                self.warnings.append(
                    "grounding produced no boxes or regions; marked as degraded"
                )
                self.degraded = True
        # F-16c (owner ruling 2026-09-23) REMOVED a clause here. It read:
        #
        #     if self.task is Task.CHANGE and self.change_map is None and not self.regions:
        #         ... "change analysis produced no spatial output" ... degraded = True
        #
        # `change_map` was a PROXY for "a map was produced": before F-16 it held
        # a filesystem path whenever a file had been written. F-16 made the ref
        # permanently null -- a client cannot retrieve it in v1 -- so the proxy
        # died, the clause collapsed to `not regions`, and a SUCCESSFUL no-change
        # analysis began reporting `degraded: true`. A null, non-retrievable
        # artifact ref was manufacturing a degradation.
        #
        # That conflated two different things, and the conflation WAS the defect.
        # `degraded` means "the analysis could not be fully performed": no
        # trained detector, or co-registration too poor to support a spatial
        # claim. Both are set by the specialist itself
        # (`specialists/change/specialist.py:345` and `:365`) and neither is this
        # validator's to invent. "No change was detected" is a NORMAL, successful
        # outcome.
        #
        # Nothing is lost by removing it. The no-change fact was never this
        # clause's to carry: it is already reported three ways -- an empty
        # `regions`, the answer text ("No change detected above threshold ..."),
        # and the CHANGE_MAP evidence's `n_components_kept` /
        # `total_change_pixels` plus the STATISTIC evidence's `n_regions_emitted`.
        #
        # No replacement clause is added, deliberately. A narrower
        # "empty answer => degraded" rule was tried and backed out: it is not
        # what the ruling asked for, it invented a semantic the specialist
        # already owns, and it made a pre-existing, unrelated fixture
        # (`test_change_with_regions_is_not_degraded`, a result with regions and
        # no answer) fail. A fix that forces edits to tests it has nothing to do
        # with is signalling over-reach, not diligence. CHANGE is the one task
        # whose `degraded` flag is now set entirely by its specialist.
        if self.task is Task.CHANGE_VQA and not self.answer.strip():
            # A change-VQA result with no answer is degraded, not empty. Note
            # that an answer alone is NOT enough to be non-degraded: the
            # specialist sets `degraded` itself when it answered from an
            # untrained head, and this validator must not clear that.
            if not self.degraded:
                self.warnings.append(
                    "change-VQA produced no answer text; marked as degraded"
                )
                self.degraded = True
        return self


# ===========================================================================
# Request / response envelopes
# ===========================================================================
class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assets: list[str] = Field(min_length=1)
    query: str
    force_task: Task | None = None
    run_id: str | None = None


class ResultEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    result: SpecialistResult
    trace: ExecutionTrace
    schema_version: str = SCHEMA_VERSION


class HealthStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded", "error"] = "ok"
    schema_version: str = SCHEMA_VERSION
    models: dict[str, str] = Field(default_factory=dict)
    device: str | None = None
    gpu_available: bool = False


__all__ = [
    "SCHEMA_VERSION",
    "Task",
    "Modality",
    "CoordinateSystem",
    "EvidenceType",
    "ControllerState",
    "Intent",
    "GeoMetadata",
    "SensorDescriptor",
    "AssetMetadata",
    "Box",
    "Region",
    "ChangeRegion",
    "Evidence",
    "ConfidenceBreakdown",
    "TraceStep",
    "ModelRef",
    "ExecutionTrace",
    "SpecialistResult",
    "AnalysisRequest",
    "ResultEnvelope",
    "HealthStatus",
]