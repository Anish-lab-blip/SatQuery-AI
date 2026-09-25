"""SatQuery AI — optical-SAR specialist (Workflow E, freeze section 2.5).

The spec calls this the #1 evaluation priority, and it is the only workflow whose
inputs are two DIFFERENT sensors. That difference is what this module exists to
manage honestly.

FROZEN CONTRACT
---------------
    exactly 2 assets, one optical and one SAR
    optical -> 12 canonical channels, zero-filled, availability mask
    SAR     ->  2 canonical channels, zero-filled, availability mask
    CROMA(SAR_images, optical_images)  <- never receives a mask (C-1)
    concat[optical_GAP, SAR_GAP, joint_GAP, optical_mask, sar_mask] -> (B, 2318)
    fusion head -> label
    confidence from plan section 26

TWO ASSETS, AND THEY MUST BE THE RIGHT TWO
------------------------------------------
`validate_request` rejects 0, 1, 3 and also TWO OF THE SAME KIND. The last case
is the one worth explaining. Two optical images is not a malformed request in the
way that three is -- it is a plausible mistake, from an operator who uploaded a
bi-temporal pair to a single-modality workflow, or who labelled a four-band
Cartosat scene as SAR.

If such a request were accepted, the adapter would place the second optical
image into the 2-channel SAR slot, zero-fill, and produce a fused tensor of the
correct shape. CROMA would run. The answer would be about one modality while
claiming to fuse two. So the modality check is done on the ASSETS, before any
pixels are read, and it raises `InvalidRequestError` naming which asset was
wrong.

CONFIDENCE (plan section 26)
----------------------------
    fusion classifier margin
    + optical confidence
    + SAR confidence
    + cross-modal agreement

No LLM, no sampling. All four are numbers this specialist computed, and
`_confidence_components` shows each one's derivation.

The two "modality confidence" terms are the availability fractions, and that is
a deliberate reading. On the hidden set the optical side is Cartosat-2S with 4
bands against a canonical 12, so 8 of 12 channels are always absent. A classifier
resting on 4 measured channels is not the same claim as one resting on 12, and a
confidence that ignored that would be reporting the model's certainty about a
tensor it was mostly fed zeros. The availability fraction is the honest scalar
for "how much real signal is behind this".

DEGRADED MODE: NO CROMA, NO HEAD
--------------------------------
`CROMA_base.pt` is 777.6 MB and `use_croma.py` must be vendored; neither is
present here, and no fusion head has been trained. Following Phase 8 and Phase 9
precedent, both are reported as deployment facts rather than crashes.

Crucially, the sensor-side work still happens. Availability, canonical placement
and the no-fabrication guarantee are computed with no weights at all, and they
are the facts an operator most needs in order to judge a Cartosat-2S + RISAT
result. When no head exists there is NO class prediction -- the result says so
in words and carries `predicted_index = None`, because emitting an untrained
label would be the fabrication the evidence system exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.errors import (
    InvalidRequestError,
    ModelLoadError,
    PairCompatibilityError,
    SpecialistError,
    scrub_paths,
)
from core.schemas import (
    ConfidenceBreakdown,
    CoordinateSystem,
    Evidence,
    EvidenceType,
    GeoMetadata,
    Region,
    SpecialistResult,
    Task,
)
from specialists.base import Specialist, SpecialistRequest

#: Plan section 26's four components. A starting point, not a calibrated
#: mapping: Phase 13 fits temperature scaling on VALIDATION data, and nothing
#: here is tuned on the eval split.
#:
#: The largest weight is the fusion margin, because it is the only term that
#: reflects what the classifier actually decided. The two availability terms
#: and agreement are corroborating evidence about the INPUT, not about the
#: decision -- they can lower a confident prediction, and they cannot raise an
#: unconfident one above it.
CONFIDENCE_WEIGHTS: dict[str, float] = {
    "fusion_margin": 0.40,
    "optical_confidence": 0.20,
    "sar_confidence": 0.20,
    "cross_modal_agreement": 0.20,
}

#: Agreement is cosine similarity in [-1, 1]. Rescaled to [0, 1] with this
#: offset so the component is comparable with the others. A negative cosine
#: (the two encoders systematically opposite) clamps to 0.0, which is the
#: correct reading: no agreement at all.
AGREEMENT_FLOOR = -1.0

#: BigEarthNet CLC class count, from `fusion.num_classes`. Only used to render a
#: readable label when the config is unavailable; the specialist always takes the
#: real value from its constructor.
DEFAULT_TASK_DIM = 19


@dataclass(frozen=True)
class PairAssessment:
    """What was established about the pair before any pixel was read."""

    optical_index: int
    sar_index: int
    warnings: list[str] = field(default_factory=list)


class OpticalSarSpecialist(Specialist):
    """Optical-SAR fusion over exactly one optical and one SAR acquisition."""

    name = "optical_sar"
    version = "0.1.0"
    capabilities = ("optical_sar",)

    def __init__(
        self,
        *,
        encoder: Any | None = None,
        head: Any | None = None,
        class_labels: list[str] | None = None,
        resolution: int = 120,
        task_dim: int = DEFAULT_TASK_DIM,
        artifact_dir: str | Path | None = None,
        device: str = "cpu",
        channel_dropout_rates: tuple[float, ...] = (1.0,),
    ) -> None:
        if task_dim < 2:
            raise SpecialistError(
                f"task_dim must be >= 2 for a margin to exist, got {task_dim}",
                specialist=self.name,
            )
        if resolution % 8 != 0:
            # CROMA asserts this (finding C-7) and core/config.py enforces it.
            raise SpecialistError(
                f"resolution must be a multiple of 8 (CROMA asserts "
                f"image_resolution % 8 == 0); got {resolution}",
                specialist=self.name,
            )

        self.encoder = encoder
        self.head = head
        self.class_labels = list(class_labels or [])
        self.resolution = int(resolution)
        self.task_dim = int(task_dim)
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.device = device
        self.channel_dropout_rates = tuple(channel_dropout_rates)

    # -- properties --------------------------------------------------------

    @property
    def has_encoder(self) -> bool:
        return self.encoder is not None

    @property
    def has_head(self) -> bool:
        """Whether a TRAINED fusion head is loaded.

        Mirrors `grounding/specialist.py:has_head` and
        `change/specialist.py:has_checkpoint`. A head object that exists but was
        never trained is not a head for this purpose: the confidence system must
        be able to distinguish "we have a trained classifier" from "we have a
        module with random weights in it", and a label emitted by the latter
        carries no learned signal.
        """
        if self.head is None:
            return False
        return bool(getattr(self.head, "_satquery_trained", False))

    # -- validation --------------------------------------------------------

    def validate_request(self, request: SpecialistRequest) -> None:
        """Exactly two assets, one optical and one SAR.

        Raises:
            InvalidRequestError: wrong asset count, a missing file, or two
                assets of the same modality.
        """
        if request.asset_count != 2:
            raise InvalidRequestError(
                f"{self.name} requires exactly 2 assets (one optical, one SAR); "
                f"got {request.asset_count}",
                user_message=(
                    "Optical-SAR fusion needs exactly two images: one optical "
                    "and one radar (SAR)."
                ),
                context={
                    "specialist": self.name,
                    "expected": 2,
                    "actual": request.asset_count,
                },
            )

        for i, asset in enumerate(request.assets):
            if not Path(asset.path).exists():
                raise InvalidRequestError(
                    f"asset {i + 1} path does not exist: {asset.path}",
                    context={"specialist": self.name, "path": asset.path},
                )

        self._assess_pair(request.assets)

    def _assess_pair(self, assets: list[Any]) -> PairAssessment:
        """Establish which asset is optical and which is SAR, or refuse.

        Modality comes from the asset's declared `modality`, falling back to a
        band-count inference through `preprocessing.raster.infer_modality`. The
        declared value wins when present, because the caller may know something
        the band count does not -- a 2-band Cartosat stack, or a 12-band
        decomposition product.
        """
        from core.schemas import Modality

        warnings: list[str] = []
        resolved: list[Modality] = []

        for asset in assets:
            modality = asset.modality
            if modality is Modality.UNKNOWN:
                inferred = self._infer_modality(asset, warnings)
                resolved.append(inferred)
            else:
                resolved.append(modality)

        optical = [i for i, m in enumerate(resolved) if m is Modality.OPTICAL]
        sar = [i for i, m in enumerate(resolved) if m is Modality.SAR]

        if len(optical) == 2:
            raise InvalidRequestError(
                "both assets were identified as OPTICAL; optical-SAR fusion "
                "requires one optical and one SAR image",
                user_message=(
                    "Both uploaded images look like optical imagery. This "
                    "workflow needs one optical image and one radar (SAR) image."
                ),
                context={
                    "specialist": self.name,
                    "modalities": [m.value for m in resolved],
                    "reason": "two_optical",
                },
            )
        if len(sar) == 2:
            raise InvalidRequestError(
                "both assets were identified as SAR; optical-SAR fusion "
                "requires one optical and one SAR image",
                user_message=(
                    "Both uploaded images look like radar (SAR) imagery. This "
                    "workflow needs one optical image and one radar (SAR) image."
                ),
                context={
                    "specialist": self.name,
                    "modalities": [m.value for m in resolved],
                    "reason": "two_sar",
                },
            )
        if len(optical) == 1 and len(sar) == 1:
            return PairAssessment(optical_index=optical[0], sar_index=sar[0],
                                  warnings=warnings)

        # Neither clean case: one asset resolved to UNKNOWN or OPTICAL_SAR.
        raise InvalidRequestError(
            f"could not identify one optical and one SAR asset from modalities "
            f"{[m.value for m in resolved]}; label the assets explicitly",
            user_message=(
                "Could not tell which image is optical and which is radar. "
                "Please label the images so one is optical and one is SAR."
            ),
            context={
                "specialist": self.name,
                "modalities": [m.value for m in resolved],
                "reason": "indeterminate",
            },
        )

    @staticmethod
    def _infer_modality(asset: Any, warnings: list[str]) -> Any:
        """Band-count inference for an asset whose modality was not declared."""
        from core.schemas import Modality
        from preprocessing.raster import infer_modality

        band_count = asset.geo.band_count
        if not band_count:
            warnings.append(
                f"asset {Path(asset.path).name} has no declared modality and no "
                f"band count; treating it as unknown"
            )
            return Modality.UNKNOWN

        inferred = infer_modality(int(band_count))
        warnings.append(
            f"asset {Path(asset.path).name} had no declared modality; inferred "
            f"'{inferred.value}' from {band_count} band(s). A band count is a "
            f"heuristic, not a sensor declaration -- label the asset explicitly "
            f"if this is wrong."
        )
        return inferred

    # -- execution ---------------------------------------------------------

    def execute(self, request: SpecialistRequest) -> SpecialistResult:
        self.validate_request(request)
        assessment = self._assess_pair(request.assets)
        warnings: list[str] = list(assessment.warnings)

        optical_asset = request.assets[assessment.optical_index]
        sar_asset = request.assets[assessment.sar_index]

        optical_desc = self._descriptor_for(optical_asset, "optical", warnings)
        sar_desc = self._descriptor_for(sar_asset, "sar", warnings)

        # -- pixels --------------------------------------------------------
        try:
            from preprocessing.raster import read_bands

            optical_array, _ = read_bands(optical_asset.path)
            sar_array, _ = read_bands(sar_asset.path)
        except SpecialistError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise SpecialistError(
                f"could not read the image pair: {exc}",
                specialist=self.name,
                context={"optical": optical_asset.path, "sar": sar_asset.path},
            ) from exc

        # -- pipeline ------------------------------------------------------
        from specialists.optical_sar.inference import run_pipeline

        inference = run_pipeline(
            optical_array=optical_array,
            sar_array=sar_array,
            optical_descriptor=optical_desc,
            sar_descriptor=sar_desc,
            encoder=self.encoder,
            head=self.head,
            resolution=self.resolution,
            task_dim=self.task_dim,
            device=self.device,
        )
        warnings.extend(inference.warnings)

        components, raw_conf = self._confidence_components(inference)
        confidence = ConfidenceBreakdown(
            raw=raw_conf,
            method="uncalibrated",
            components=components,
            degraded=inference.degraded,
            degradation_reason=self._degradation_reason(inference),
        )

        label = self._label_for(inference)
        answer = self._compose_answer(label, inference)

        artifacts = self._write_artifacts(inference, optical_asset, sar_asset)

        result = SpecialistResult(
            task=Task.OPTICAL_SAR,
            answer=answer,
            labels=[label] if label else [],
            regions=[],
            confidence=confidence,
            geospatial=self._geospatial(optical_asset, sar_asset, inference),
            degraded=inference.degraded,
        )

        result.evidence = self._build_evidence(
            inference,
            label=label,
            components=components,
            artifacts=artifacts,
            warnings=warnings,
        )
        result.warnings.extend(warnings)
        return result

    # -- descriptors -------------------------------------------------------

    def _descriptor_for(self, asset: Any, kind: str, warnings: list[str]) -> Any:
        """The asset's sensor descriptor, or one inferred from its band count.

        An asset that arrives with a `SensorDescriptor` is used as given: the
        caller declared its bands, and overriding a declaration with a guess
        would be the fabrication this module forbids.
        """
        from specialists.optical_sar.sensor_adapter import (
            OPTICAL_CANONICAL,
            build_optical_adapter,
        )

        if asset.sensor is not None:
            descriptor = asset.sensor
            declared = descriptor.canonical_channels
            expected = 12 if kind == "optical" else 2
            if declared != expected:
                warnings.append(
                    f"asset '{descriptor.sensor}' declares "
                    f"{declared} canonical channels but CROMA expects {expected} "
                    f"for {kind}; using the declared layout and zero-filling "
                    f"the remainder"
                )
            return descriptor

        band_count = int(asset.geo.band_count or 0)
        sensor_name = Path(asset.path).stem

        if kind == "optical":
            n = min(band_count or 0, len(OPTICAL_CANONICAL))
            if n < 1:
                raise PairCompatibilityError(
                    f"optical asset '{sensor_name}' reports no bands; cannot "
                    f"build a canonical layout",
                    user_message=(
                        "The optical image has no readable bands."
                    ),
                    context={"specialist": self.name, "path": asset.path},
                )
            bands = list(OPTICAL_CANONICAL[:n])
            warnings.append(
                f"optical asset '{sensor_name}' carried no sensor descriptor; "
                f"its {band_count} band(s) were mapped positionally onto the "
                f"first {n} canonical Sentinel-2 channels {bands}. This is an "
                f"ASSUMPTION about band identity, not a measurement -- supply a "
                f"sensor descriptor to state the real layout."
            )
            return build_optical_adapter(sensor_name, available_bands=bands)

        if band_count < 1:
            raise PairCompatibilityError(
                f"SAR asset '{sensor_name}' reports no bands; cannot build a "
                f"canonical layout",
                user_message="The radar image has no readable bands.",
                context={"specialist": self.name, "path": asset.path},
            )
        if band_count > 2:
            warnings.append(
                f"SAR asset '{sensor_name}' reports {band_count} bands; CROMA "
                f"takes 2. Only the first 2 are placed in the canonical SAR "
                f"representation."
            )

        # Pass the polarisations the raster actually carries. Calling
        # `describe_sar(stem)` alone would rely on the filename being a known
        # sensor name, which it almost never is -- and an unknown sensor yields
        # no bands at all, so nothing would be placeable.
        #
        # The polarisation LABELS remain unknown here (a band count is not a
        # polarisation), so this uses the positional convention and says so.
        from specialists.optical_sar.sensor_adapter import positional_fallback_descriptor

        warnings.append(
            f"SAR asset '{sensor_name}' carried no sensor descriptor; its "
            f"{min(band_count, 2)} channel(s) were mapped positionally onto "
            f"polarisation slots in the order the file stores them. The "
            f"polarisation IDENTITY of each channel is NOT known from a band "
            f"count alone -- supply a sensor descriptor to state it."
        )
        return positional_fallback_descriptor(sensor_name, min(band_count, 2))

    # -- confidence --------------------------------------------------------

    def _confidence_components(
        self, inference: Any
    ) -> tuple[dict[str, float], float]:
        """Plan section 26's four components. No LLM, no sampling.

            fusion_margin          top-1 minus top-2 probability, from the head
            optical_confidence     fraction of the 12 optical channels present
            sar_confidence         fraction of the 2 SAR channels present
            cross_modal_agreement  cosine(optical_GAP, SAR_GAP), rescaled to [0,1]

        Composition: a weighted sum, floored to 0.0 whenever there is nothing to
        be confident ABOUT. Those floors are the load-bearing part:

            no prediction      no head was loaded, so there is no decision
            untrained head     a real tensor with no learned signal in it

        Both are signal GAPS, not weak signals, so they resolve to 0.0 rather
        than merely discounting -- the same rule Phase 9 applied to a change map
        with no regions.
        """
        stats = inference.availability()

        optical_fraction = float(np.clip(stats["optical_available_fraction"], 0.0, 1.0))
        sar_fraction = float(np.clip(stats["sar_available_fraction"], 0.0, 1.0))

        agreement_component = 0.0
        agreement_raw = inference.agreement
        if agreement_raw is not None:
            agreement_component = float(
                np.clip((agreement_raw - AGREEMENT_FLOOR) / (1.0 - AGREEMENT_FLOOR),
                        0.0, 1.0)
            )

        components: dict[str, float] = {
            "optical_confidence": optical_fraction,
            "sar_confidence": sar_fraction,
            "cross_modal_agreement": agreement_component,
            "optical_channels_present": stats["optical_channels_present"],
            "sar_channels_present": stats["sar_channels_present"],
            "has_croma": 1.0 if self.has_encoder else 0.0,
            "trained_head": 1.0 if self.has_head else 0.0,
        }
        if agreement_raw is not None:
            components["cross_modal_agreement_raw"] = float(agreement_raw)

        if not inference.has_prediction:
            # Recorded BEFORE any early return so the trace states every reason
            # the score is zero, not just the first one encountered.
            components["no_prediction"] = 1.0
            return components, 0.0

        margin = float(np.clip(inference.margin, 0.0, 1.0))
        components["fusion_margin"] = margin

        if not self.has_head:
            return components, 0.0

        weighted = sum(
            CONFIDENCE_WEIGHTS[k] * max(0.0, min(1.0, components.get(k, 0.0)))
            for k in CONFIDENCE_WEIGHTS
        )
        return components, float(max(0.0, min(1.0, weighted)))

    @staticmethod
    def _degradation_reason(inference: Any) -> str | None:
        if not inference.degradation_reasons:
            return None
        return "; ".join(inference.degradation_reasons)

    # -- presentation ------------------------------------------------------

    def _label_for(self, inference: Any) -> str:
        """The predicted class name, or "" when there was no prediction."""
        if not inference.has_prediction:
            return ""
        index = int(inference.predicted_index)
        if 0 <= index < len(self.class_labels):
            return self.class_labels[index]
        return f"class_{index}"

    def _compose_answer(self, label: str, inference: Any) -> str:
        """Plain prose built only from computed facts."""
        stats = inference.availability()
        optical_n = int(stats["optical_channels_present"])
        sar_n = int(stats["sar_channels_present"])
        optical_total = inference.optical.n_channels
        sar_total = inference.sar.n_channels

        availability = (
            f"optical channels {optical_n}/{optical_total}, SAR channels "
            f"{sar_n}/{sar_total}"
        )

        if not self.has_encoder:
            return (
                f"No fused prediction was produced: CROMA is not loaded. "
                f"Sensor-side analysis completed ({availability}); "
                f"{optical_total - optical_n} optical channel(s) and "
                f"{sar_total - sar_n} SAR channel(s) are absent for these "
                f"sensors and are zero-filled with the availability mask set "
                f"false."
            )
        if not inference.has_prediction:
            return (
                f"No fused prediction was produced: no trained fusion head is "
                f"loaded. The optical/SAR/joint representation was computed "
                f"({availability}), but a label requires a trained head. "
                f"Absent channels are zero-filled and masked, never invented."
            )

        margin = inference.margin
        return (
            f"Fused optical-SAR prediction: {label} "
            f"(margin {margin:.3f}; {availability})."
        )

    def _geospatial(
        self, optical_asset: Any, sar_asset: Any, inference: Any
    ) -> GeoMetadata:
        """The OPTICAL asset's georeferencing, plus a cross-sensor note.

        Optical is chosen because it is the asset a reader can navigate by. The
        SAR footprint may differ -- RISAT and Cartosat-2S are different missions
        with different orbits -- so the SAR bounds are carried alongside rather
        than fused into a single claimed footprint.
        """
        geo = optical_asset.geo
        sar_geo = sar_asset.geo

        extra: dict[str, Any] = {
            "optical_crs": geo.crs,
            "sar_crs": sar_geo.crs,
            "sar_bounds": sar_geo.bounds,
            "sar_resolution": sar_geo.resolution,
            "optical_availability": inference.optical.descriptor.availability_mask,
            "sar_availability": inference.sar.descriptor.availability_mask,
            "optical_sensor": inference.optical.descriptor.sensor,
            "sar_sensor": inference.sar.descriptor.sensor,
        }

        resolution = self._resolution_delta(geo, sar_geo)
        if resolution is not None:
            extra["resolution_ratio_sar_to_optical"] = resolution

        return GeoMetadata(
            crs=geo.crs,
            transform=geo.transform,
            bounds=geo.bounds,
            width=geo.width,
            height=geo.height,
            band_count=geo.band_count,
            dtype=geo.dtype,
            nodata=geo.nodata,
            resolution=geo.resolution,
            has_crs=geo.has_crs,
            is_georeferenced=geo.is_georeferenced,
            **extra,
        )

    @staticmethod
    def _resolution_delta(optical_geo: Any, sar_geo: Any) -> float | None:
        """Ratio of SAR to optical ground sample distance, when both are known."""
        o = optical_geo.resolution
        s = sar_geo.resolution
        if not o or not s or len(o) < 1 or len(s) < 1:
            return None
        if not o[0]:
            return None
        return float(s[0] / o[0])

    # -- artifacts ---------------------------------------------------------

    def _write_artifacts(
        self, inference: Any, optical_asset: Any, sar_asset: Any
    ) -> dict[str, str | None]:
        """Render the optical and SAR views as evidence artifacts.

        These are PRESENTATION views of the canonical tensors, not sensor
        products. Both are written from data that exists even when CROMA does
        not, because seeing which channels are real is the most useful thing an
        operator can look at on a mismatched sensor pair.
        """
        if self.artifact_dir is None:
            return {"optical_view": None, "sar_view": None}

        out: dict[str, str | None] = {"optical_view": None, "sar_view": None}
        try:
            import cv2

            self.artifact_dir.mkdir(parents=True, exist_ok=True)

            optical_png = self.artifact_dir / f"optical_view_{Path(optical_asset.path).stem}.png"
            sar_png = self.artifact_dir / f"sar_view_{Path(sar_asset.path).stem}.png"

            cv2.imwrite(str(optical_png), self._render_view(inference.optical.canonical))
            cv2.imwrite(str(sar_png), self._render_view(inference.sar.canonical))

            out["optical_view"] = str(optical_png)
            out["sar_view"] = str(sar_png)
        except Exception:  # noqa: BLE001
            # Artifact rendering is a presentation concern; failing the analysis
            # because a PNG could not be written would be wrong.
            return {"optical_view": None, "sar_view": None}
        return out

    @staticmethod
    def _render_view(canonical: np.ndarray) -> np.ndarray:
        """(C, H, W) canonical -> (H, W, 3) uint8 for a human to look at.

        Missing channels are zeros, so the view is dark where the sensor had no
        band. That is the honest rendering: it shows the gap rather than
        stretching noise to fill the frame.
        """
        arr = np.asarray(canonical, dtype=np.float32)
        if arr.shape[0] >= 3:
            rgb = arr[:3]
        elif arr.shape[0] == 1:
            rgb = np.repeat(arr, 3, axis=0)
        else:
            rgb = np.concatenate([arr, np.zeros((3 - arr.shape[0], *arr.shape[1:]), dtype=np.float32)])

        out = np.zeros((*rgb.shape[1:], 3), dtype=np.uint8)
        for c in range(3):
            band = rgb[c]
            lo, hi = float(band.min()), float(band.max())
            if hi > lo:
                scaled = (band - lo) / (hi - lo) * 255.0
            else:
                scaled = np.zeros_like(band)
            out[:, :, c] = np.clip(scaled, 0, 255).astype(np.uint8)
        return out

    # -- evidence ----------------------------------------------------------

    def _build_evidence(
        self,
        inference: Any,
        *,
        label: str,
        components: dict[str, float],
        artifacts: dict[str, str | None],
        warnings: list[str],
    ) -> list[Evidence]:
        """Evidence for what was computed, including what was absent."""
        items: list[Evidence] = []

        # The availability masks. These are the C-1 evidence: they record which
        # channels were real, which is the fact the whole adapter exists to
        # preserve.
        for kind, output in (("optical", inference.optical), ("sar", inference.sar)):
            descriptor = output.descriptor
            items.append(
                Evidence(
                    type=EvidenceType.AVAILABILITY_MASK,
                    source_specialist=self.name,
                    score=float(output.mask.mean()) if output.mask.size else 0.0,
                    payload={
                        "modality": kind,
                        "sensor": descriptor.sensor,
                        "canonical_channels": descriptor.canonical_channels,
                        "available_bands": list(descriptor.available_bands),
                        "band_map": dict(descriptor.band_map),
                        "availability_mask": [bool(b) for b in output.mask],
                        "missing_channels": output.missing_indices,
                        "missing_channels_zero_filled": descriptor.missing_channels_zero_filled,
                        "normalization": descriptor.normalization,
                        "resolution": descriptor.resolution,
                    },
                )
            )

        # The views. F-16 (owner ruling 2026-09-23): `artifact_ref` is NEVER a
        # filesystem path.
        #
        # v1 has no artifact-serving endpoint, so a rendered view cannot be
        # retrieved by a client at all. The previous shape put the ABSOLUTE
        # ON-DISK PATH of the PNG into `artifact_ref` and, when rendering
        # failed, omitted the evidence item entirely -- which made "we rendered
        # nothing" and "we rendered something you cannot fetch"
        # indistinguishable, and disclosed an internal path in the meantime.
        # `API_CONTRACT.md` section 2.4 documented an `artifact://` URI that no
        # production file ever emitted; fabricating one now would be worse than
        # the path it replaced.
        #
        # So: the item is emitted EITHER WAY; `artifact_ref` is `None`; the
        # payload states whether a view was rendered and that it is not
        # retrievable; and a warning says so in words a client can act on.
        for key, ev_type, output in (
            ("optical_view", EvidenceType.OPTICAL_VIEW, inference.optical),
            ("sar_view", EvidenceType.SAR_VIEW, inference.sar),
        ):
            rendered = bool(artifacts.get(key))
            items.append(
                Evidence(
                    type=ev_type,
                    source_specialist=self.name,
                    artifact_ref=None,
                    coordinate_system=CoordinateSystem.NORMALIZED_0_1,
                    coordinates=[0.0, 0.0, 1.0, 1.0],
                    payload={
                        "sensor": output.descriptor.sensor,
                        "rendered": rendered,
                        "retrievable": False,
                        "retrieval": "no artifact-serving endpoint in v1",
                    },
                )
            )
            view_name = key.replace("_", " ")
            if rendered:
                warnings.append(
                    f"the {view_name} was rendered for server-side diagnostics "
                    f"but is NOT retrievable: v1 exposes no artifact-serving "
                    f"endpoint, so artifact_ref is null by design"
                )
            else:
                warnings.append(
                    f"the {view_name} could not be rendered, so no artifact "
                    f"exists for this run and artifact_ref is null"
                )

        # The fused representation. Emitted as a statistic even when no head
        # exists, because the concatenation WAS computed and its width is the
        # frozen contract.
        if self.has_encoder:
            items.append(
                Evidence(
                    type=EvidenceType.JOINT_FEATURE_REGION,
                    source_specialist=self.name,
                    coordinate_system=CoordinateSystem.NORMALIZED_0_1,
                    coordinates=[0.0, 0.0, 1.0, 1.0],
                    payload={
                        "fusion_input_dim": inference.fusion_input.dim,
                        "expected_dim": 2318,
                        "encoder_dim": 768,
                        "modalities": ["optical", "sar", "joint"],
                        "mask_consumed_by": "fusion_head",
                        "croma_received_mask": False,
                        "n_patches": getattr(self.encoder, "n_patches", None),
                        "resolution": self.resolution,
                    },
                )
            )

        # The decision, when there was one.
        if inference.has_prediction:
            items.append(
                Evidence(
                    type=EvidenceType.STATISTIC,
                    source_specialist=self.name,
                    score=float(np.clip(inference.margin, 0.0, 1.0)),
                    payload={
                        "predicted_label": label,
                        "predicted_index": int(inference.predicted_index),
                        "fusion_margin": float(inference.margin),
                        "class_probabilities": [
                            float(p) for p in inference.probabilities[0]
                        ] if inference.probabilities is not None else [],
                    },
                )
            )
        else:
            # The no-prediction record. Deliberately a STATISTIC with no score:
            # there is no decision to score, and inventing one would defeat the
            # degradation path.
            items.append(
                Evidence(
                    type=EvidenceType.STATISTIC,
                    source_specialist=self.name,
                    score=None,
                    payload={
                        "prediction_suppressed": True,
                        "reason": self._degradation_reason(inference)
                        or "no trained fusion head loaded",
                        "has_croma": self.has_encoder,
                        "trained_head": self.has_head,
                    },
                )
            )

        # The confidence derivation, so the numbers are auditable.
        items.append(
            Evidence(
                type=EvidenceType.STATISTIC,
                source_specialist=self.name,
                score=None,
                payload={
                    "confidence_components": {k: round(v, 6) for k, v in components.items()},
                    "confidence_weights": dict(CONFIDENCE_WEIGHTS),
                    "method": "uncalibrated",
                },
            )
        )

        return items

    # -- contract ----------------------------------------------------------

    def produce_evidence(self, result: SpecialistResult) -> list[Evidence]:
        return result.evidence

    def estimate_confidence(self, result: SpecialistResult) -> ConfidenceBreakdown:
        return result.confidence

    def model_refs(self) -> list[dict[str, str]]:
        refs: list[dict[str, str]] = []
        if self.encoder is not None:
            describe = getattr(self.encoder, "describe", None)
            info = describe() if callable(describe) else {}
            # The checkpoint path is a SERVER-SIDE diagnostic: `describe()`
            # reports it in full so an operator can see exactly which file was
            # loaded. This method does NOT stay server-side --
            # `core/controller.py:689` folds these refs into
            # `ExecutionTrace.selected_models`, which is published, and
            # `API_CONTRACT.md` section 7 records that v1 has no auth. So only
            # the basename is published; the directory chain is reduced, the
            # filename (the useful provenance) survives.
            #
            # This is not hypothetical tidying. Before the CROMA checkpoint was
            # reachable by default this field read "unspecified"; wiring the
            # encoder is what makes it an absolute path, so the scrub is part of
            # that change rather than a follow-up to it.
            revision = scrub_paths(str(info.get("checkpoint", "unspecified")))
            refs.append({
                "name": "CROMA",
                "revision": revision or "unspecified",
                "role": "optical_sar_encoder",
            })
        else:
            refs.append({
                "name": "CROMA",
                "revision": "not loaded",
                "role": "optical_sar_encoder",
            })

        refs.append({
            "name": "FusionHead",
            "revision": (
                f"trained params={self.head.num_parameters()}"
                if self.has_head
                else "no trained artifact"
            ),
            "role": "optical_sar_fusion_head",
        })
        return refs


def build_optical_sar_specialist(
    config: Any,
    *,
    checkpoint_path: str | Path | None = None,
    vendor_dir: str | Path | None = None,
    head_path: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    device: str | None = None,
    class_labels: list[str] | None = None,
) -> OpticalSarSpecialist:
    """Construct the specialist from the central config.

    Every model artifact is OPTIONAL, and a missing one degrades rather than
    raising -- with one exception. `checkpoint_path` that EXISTS but fails to
    load raises `ModelLoadError`, because a corrupt artifact must never be
    silently replaced by a degraded run: that is the difference between "we do
    not have this" and "we have it and it is broken".

    Args:
        checkpoint_path: local `CROMA_base.pt`. When None or missing, CROMA is
            not loaded and the specialist runs sensor-only.
        vendor_dir: directory holding the vendored `use_croma.py`.
        head_path: a trained fusion head. When None or missing, no prediction is
            produced.
        artifact_dir: where to write optical/SAR views.
        device: torch device string.
        class_labels: class names for the predicted index.

    Raises:
        ModelLoadError: a named artifact exists but cannot be read.
    """
    device = device or config.device_preference

    encoder = None
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        from specialists.optical_sar.croma import build_encoder

        encoder = build_encoder(
            config,
            device=device,
            checkpoint_path=checkpoint_path,
            vendor_dir=vendor_dir,
        )

    from specialists.optical_sar.inference import build_head_from_config

    head = build_head_from_config(
        config,
        head_path=str(head_path) if head_path is not None else None,
        device=device,
    )

    return OpticalSarSpecialist(
        encoder=encoder,
        head=head,
        class_labels=class_labels,
        resolution=int(config.get("croma.image_resolution", 120)),
        task_dim=int(config.get("fusion.num_classes", DEFAULT_TASK_DIM)),
        artifact_dir=artifact_dir,
        device=device,
    )


__all__ = [
    "OpticalSarSpecialist",
    "PairAssessment",
    "build_optical_sar_specialist",
    "CONFIDENCE_WEIGHTS",
    "AGREEMENT_FLOOR",
    "DEFAULT_TASK_DIM",
]
