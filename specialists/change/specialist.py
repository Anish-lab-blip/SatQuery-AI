"""SatQuery AI — change-detection specialist (Workflow D).

Bi-temporal change detection over an EXACTLY-TWO-image pair. This is the module
that makes the Phase 9 model actually dispatchable; before it existed the
architecture, post-processing and metrics all passed their tests while the
controller had nothing to call, because `specialists/change/` had no
`Specialist` subclass.

FROZEN CONTRACT
---------------
    2 x GeoTIFF (T1, T2)        ->  (H, W, 3) uint8 each, percentile-stretched
    STANetStyleChangeDetector   ->  (B, 1, H, W) logits, sigmoid -> probabilities
    postprocess_change_map      ->  binary mask -> cleaned mask -> regions
    regions_to_schema           ->  ChangeRegion[] in normalized 0-1

TWO ASSETS, NOT ONE
-------------------
Change detection is meaningless on a single image, and undefined on three.
`validate_request` enforces exactly two, because a silent "just use the first
two" would produce a confident answer to a question the caller did not ask.

POOR REGISTRATION LOWERS CONFIDENCE AND SUPPRESSES SPATIAL CLAIMS
----------------------------------------------------------------
Architecture freeze section 2.4 and plan section 40: "Never convert poor
registration into artificial certainty." A change detector fed two
mis-registered acquisitions reports change along every edge in the scene. That
output is not wrong so much as meaningless, and it looks exactly like a
confident detection.

`measure_registration` is therefore run BEFORE the change map is trusted. If
the pair is not usable, the maps are still computed and reported -- the caller
may legitimately want to see them -- but the spatial region claims are
WITHHELD (no `ChangeRegion`s, no `change_map` artifact ref) and the confidence
is driven to the floor. The verdict says so, in words, in `warnings`.

DEGRADED MODE: NO CHECKPOINT WIRED BY DEFAULT
---------------------------------------------
A trained head now exists and is benchmarked
(`artifacts/change/levir_change_v001/head.pt`, test pooled IoU 0.8122), yet it
is deliberately NOT wired into serving by default. Populating
`change.checkpoint_path` in `configs/base.yaml` would move `Config.hash`, and
`scripts/eval_change.py` refuses to score on a hash drift (exit 3) -- so that
one edit would invalidate the project's own benchmark number. The wiring path
is therefore the registry `builders=` override (see `core/registry.py`), which
injects the checkpoint without touching the config. With no checkpoint wired,
per Phase 8's precedent (`grounding/specialist.py:has_head`) a missing artifact
is a *deployment* case, not a crash. The specialist runs the
randomly-initialised model, marks the result degraded, and says so plainly --
because a randomly-initialised change map is a map of noise, and presenting it
as a detection would be exactly the fabrication the evidence system exists to
prevent. The confidence components make the untrained state visible as
numbers, not only as a warning string.

WHY THE VLM IS NOT ASKED TO DESCRIBE THE CHANGE
-----------------------------------------------
The change map is produced by the detector. A language model may narrate it;
it may not produce it. Same rule as grounding finding C-5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.errors import (
    InvalidRequestError,
    PairCompatibilityError,
    SpecialistError,
    TemporalPairError,
)
from core.schemas import (
    ChangeRegion,
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
from specialists.change.postprocess import (
    PostprocessResult,
    RegistrationQuality,
    measure_registration,
    postprocess_change_map,
    regions_to_schema,
)

#: Weights for the measurable confidence components. A starting point, not a
#: calibrated mapping -- Phase 13 fits temperature scaling on VALIDATION data.
#: Nothing here is tuned on the eval split.
#:
#: `registration_quality` carries the largest single weight, and the final
#: score is a MINIMUM against it rather than only a weighted sum -- see
#: `_compose_confidence`. Alignment is a gate, not one vote among three.
CONFIDENCE_WEIGHTS: dict[str, float] = {
    "registration_quality": 0.50,
    "mean_change_probability": 0.30,
    "component_stability": 0.20,
}

#: Minimum probe-pixel intensity for a region to count as STABLE across
#: checks. A region whose pixels sit just above the threshold flips membership
#: on the tiniest perturbation; a region well above it does not. Stability is
#: therefore the mean headroom over the threshold, normalized.
#:
#: Measured on the untrained path: headroom is near-zero, so stability is
#: near-zero, which is the honest reading of a random model.
STABILITY_HEADROOM = 0.5

#: A pair is compared at this maximum translation before it is declared
#: unusable. Matches `measure_registration`'s default; declared here so the
#: specialist's policy is readable in one place.
DEFAULT_MAX_SHIFT_PX = 8

#: Below this many change pixels the map is treated as "no change asserted"
#: rather than as a faint detection. A handful of pixels is noise, and
#: reporting them as regions would be the degenerate-box failure of Phase 8 in
#: a different costume.
MIN_REGION_PIXELS = 32


@dataclass(frozen=True)
class PairAssessment:
    """Everything the specialist learned about the pair before trusting it."""

    registration: RegistrationQuality
    compatible: bool
    warnings: list[str]


class ChangeSpecialist(Specialist):
    """Bi-temporal change detection over exactly two acquisitions."""

    name = "change"
    version = "0.1.0"
    capabilities = ("change",)

    def __init__(
        self,
        model: Any | None = None,
        *,
        threshold: float = 0.5,
        min_component_pixels: int = MIN_REGION_PIXELS,
        max_shift_px: int = DEFAULT_MAX_SHIFT_PX,
        min_registration_response: float = 0.15,
        artifact_dir: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise SpecialistError(
                f"threshold must be in [0,1], got {threshold}", specialist=self.name
            )
        if min_component_pixels < 1:
            raise SpecialistError(
                f"min_component_pixels must be >= 1, got {min_component_pixels}",
                specialist=self.name,
            )

        self.model = model
        self.threshold = threshold
        self.min_component_pixels = min_component_pixels
        self.max_shift_px = max_shift_px
        self.min_registration_response = min_registration_response
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.device = device

        if self.model is not None:
            self.model.eval()

    # -- properties --------------------------------------------------------

    @property
    def has_checkpoint(self) -> bool:
        """Whether a TRAINED detector is loaded.

        `model` is never None in practice -- an untrained one is still a model.
        This distinguishes "a checkpoint was loaded" from "we built a random
        one and are being honest about it".
        """
        return bool(getattr(self.model, "_satquery_trained", False))

    # -- validation --------------------------------------------------------

    def validate_request(self, request: SpecialistRequest) -> None:
        """Exactly two assets, both present, and a compatible pair.

        Raises the most specific error available so the controller can map it
        to a user-facing message (plan section 57 failure matrix).
        """
        if request.asset_count != 2:
            # ONE asset is the common mistake: the caller treated this as a
            # single-image task. THREE is the other: they attached a time
            # series. Both are rejected with the same typed error, and the
            # reason names the count, because "invalid request" alone does not
            # tell an operator what to fix.
            raise InvalidRequestError(
                f"{self.name} requires exactly 2 assets (T1 and T2); "
                f"got {request.asset_count}",
                user_message=(
                    "Change detection needs exactly two images: an earlier one "
                    "and a later one."
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
                    f"asset T{i + 1} path does not exist: {asset.path}",
                    context={"specialist": self.name, "path": asset.path},
                )

        self._assess_pair(request.assets)

    def _assess_pair(self, assets: list[Any]) -> PairAssessment:
        """CRS compatibility, temporal distinctness, then registration.

        Split out so `execute` can reuse the SAME assessment the validator
        made, rather than measuring registration twice and reporting two
        numbers that could disagree.
        """
        t1, t2 = assets
        warnings: list[str] = []

        # -- temporal distinctness ----------------------------------------
        # The same bytes twice is not a change pair. `path` equality is the
        # cheap check; `sha256` when both carry it is the correct one.
        if t1.path == t2.path:
            raise TemporalPairError(
                f"T1 and T2 are the same file: {t1.path}",
                context={"specialist": self.name, "path": t1.path},
            )
        if t1.sha256 and t2.sha256 and t1.sha256 == t2.sha256:
            raise TemporalPairError(
                "T1 and T2 have identical sha256; a repeated acquisition is "
                "not a temporal pair",
                context={"specialist": self.name, "sha256": t1.sha256},
            )

        # -- CRS compatibility (freeze section 2.6) -----------------------
        from geospatial.crs import compare_crs

        compat = compare_crs(t1.geo.crs, t2.geo.crs)
        if not compat.compatible:
            # No CRS on either side is NOT fatal for pixel-domain change
            # detection -- the two rasters still tile to the same grid. It IS
            # fatal for any claim about ground coordinates, so the warning is
            # recorded and the geospatial block is withheld downstream.
            warnings.append(
                f"CRS comparison unavailable ({compat.reason}); change is "
                f"reported in normalized pixel coordinates only"
            )
        elif compat.requires_reprojection:
            warnings.append(
                f"T1 and T2 use different CRS ({compat.reason}); change is "
                f"valid only if the rasters already share a pixel grid"
            )

        # -- registration quality (freeze section 2.4) ---------------------
        reg = self._measure_registration_for(t1, t2, warnings)
        return PairAssessment(registration=reg, compatible=compat.compatible,
                              warnings=warnings)

    def _measure_registration_for(
        self, t1: Any, t2: Any, warnings: list[str]
    ) -> RegistrationQuality:
        """Measure alignment, degrading to a zero-quality verdict on failure.

        A failure to MEASURE is not the same as a measurement of failure, but
        both mean the pair cannot be trusted spatially, so both produce an
        unusable verdict. The distinction is recorded in the warning.
        """
        try:
            from preprocessing.imagery import load_image_array

            a = load_image_array(t1.path)
            b = load_image_array(t2.path)
            return measure_registration(
                a, b,
                max_shift_px=self.max_shift_px,
                min_response=self.min_registration_response,
            )
        except SpecialistError as exc:
            warnings.append(
                f"registration could not be measured ({exc.detail}); the pair "
                f"is treated as unusable for spatial claims"
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"registration could not be measured ({type(exc).__name__}: "
                f"{exc}); the pair is treated as unusable for spatial claims"
            )
        return RegistrationQuality(
            shift_x=0.0, shift_y=0.0, response=0.0,
            max_shift_px=self.max_shift_px, is_usable=False,
            reason="registration measurement failed",
        )

    # -- execution ---------------------------------------------------------

    def execute(self, request: SpecialistRequest) -> SpecialistResult:
        self.validate_request(request)
        t1, t2 = request.assets
        warnings: list[str] = []
        degraded = False

        assessment = self._assess_pair([t1, t2])
        reg = assessment.registration
        warnings.extend(assessment.warnings)

        if not reg.is_usable:
            warnings.append(
                "poor co-registration: " + reg.reason + ". Spatial change "
                "claims are suppressed; confidence reflects the measurement, "
                "not the change map."
            )

        # -- pixels --------------------------------------------------------
        try:
            from preprocessing.imagery import load_image_array

            a = load_image_array(t1.path)
            b = load_image_array(t2.path)
        except SpecialistError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise SpecialistError(
                f"could not prepare the image pair: {exc}",
                specialist=self.name,
                context={"t1": t1.path, "t2": t2.path},
            ) from exc

        # -- model ---------------------------------------------------------
        probabilities = self._predict(a, b, warnings)
        if not self.has_checkpoint:
            degraded = True
            warnings.append(
                "no trained change checkpoint is loaded; the detector is "
                "randomly initialised, so the change map carries NO learned "
                "signal. Treat every region as noise until a trained artifact "
                "is supplied."
            )

        # -- post-process --------------------------------------------------
        post = postprocess_change_map(
            probabilities,
            threshold=self.threshold,
            min_component_pixels=self.min_component_pixels,
        )

        # -- suppress spatial claims when alignment is bad -----------------
        # The maps are still returned (the caller may want to look at them),
        # but the REGION CLAIMS are withheld: a region is an assertion about
        # where something changed on the ground, and we cannot make it.
        suppress = not reg.is_usable
        if suppress and post.regions:
            warnings.append(
                f"withheld {len(post.regions)} region(s) because the pair is "
                f"not usable for spatial analysis"
            )
            regions: list[ChangeRegion] = []
        else:
            regions = regions_to_schema(
                post.regions, a.shape[1], a.shape[0]
            )

        artifacts = self._write_artifacts(
            probabilities, post, t1, t2, suppressed=suppress, warnings=warnings
        )

        components, raw_conf = self._confidence_components(
            probabilities, post, reg, suppressed=suppress
        )
        confidence = ConfidenceBreakdown(
            raw=raw_conf,
            method="uncalibrated",
            components=components,
            degraded=degraded or suppress,
            degradation_reason=self._degradation_reason(
                degraded=degraded, suppressed=suppress
            ),
        )

        change_px = post.total_change_pixels
        if suppress:
            answer = (
                f"Change detection could not be trusted: the two acquisitions "
                f"are not co-registered ({reg.shift_magnitude:.1f}px offset, "
                f"response {reg.response:.3f}). {change_px} candidate change "
                f"pixel(s) were found but are not attributable to real change."
            )
        elif not regions:
            answer = (
                f"No change detected above threshold {self.threshold:.2f}. "
                f"{change_px} raw change pixel(s) remained after filtering."
            )
        else:
            largest = max(r.area_pixels for r in regions)
            answer = (
                f"Detected change in {len(regions)} region(s), largest "
                f"{largest} px. {change_px} change pixel(s) total."
            )

        result = SpecialistResult(
            task=Task.CHANGE,
            answer=answer,
            labels=[request.query] if request.query.strip() else [],
            regions=[
                # Region is the general spatial shape; ChangeRegion carries the
                # change-specific fields. Both are emitted from the same data
                # so a consumer reading either gets the same answer.
                Region(
                    box=r.box,
                    label="change",
                    score=r.mean_probability,
                    coordinate_system=CoordinateSystem.NORMALIZED_0_1,
                )
                for r in regions
            ],
            change_map=artifacts["change_map"],
            confidence=confidence,
            geospatial=self._geospatial(t1, assessment),
            degraded=degraded or suppress,
        )

        result.evidence = self._build_evidence(
            probabilities, post, regions, reg,
            artifacts=artifacts,
            suppressed=suppress,
            n_withheld=len(post.regions) if suppress else 0,
        )
        result.warnings.extend(warnings)
        return result

    # -- inference ---------------------------------------------------------

    def _predict(self, a: np.ndarray, b: np.ndarray, warnings: list[str]) -> np.ndarray:
        """Run the detector and return a (H, W) probability map in [0, 1]."""
        import torch

        if self.model is None:
            raise SpecialistError(
                "change specialist has no model; construct it with a detector "
                "or use build_change_specialist()",
                specialist=self.name,
            )

        t1 = self._to_tensor(a)
        t2 = self._to_tensor(b)

        h, w = t1.shape[-2:]
        if h % 8 or w % 8:
            # The encoder's stride is 8. Rather than refusing a raster that is
            # one pixel off, reflect-pad up to the next multiple and crop the
            # result back. This is a PRESENTATION crop -- every downstream
            # coordinate is computed against the original (h, w).
            ph, pw = (-h) % 8, (-w) % 8
            t1 = torch.nn.functional.pad(t1, (0, pw, 0, ph), mode="reflect")
            t2 = torch.nn.functional.pad(t2, (0, pw, 0, ph), mode="reflect")
            warnings.append(
                f"padded {h}x{w} to {h + ph}x{w + pw} to satisfy the encoder's "
                f"stride-8 requirement; the change map was cropped back"
            )

        try:
            with torch.no_grad():
                out = self.model(t1.to(self.device), t2.to(self.device))
            prob = out.probabilities.detach().cpu().numpy()
        except SpecialistError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise SpecialistError(
                f"change model forward pass failed: {exc}",
                specialist=self.name,
                context={"input_shape": tuple(t1.shape)},
            ) from exc

        if prob.ndim == 4:
            prob = prob[0, 0]
        elif prob.ndim == 3:
            prob = prob[0]

        return np.asarray(prob[:h, :w], dtype=np.float64)

    @staticmethod
    def _to_tensor(array: np.ndarray) -> Any:
        """(H, W, 3) uint8 -> (1, 3, H, W) float32 in [0, 1]."""
        import torch

        arr = np.asarray(array, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[:, :, :3]
        # (H, W, C) -> (C, H, W)
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)

    # -- artifacts ---------------------------------------------------------

    def _write_artifacts(
        self,
        probabilities: np.ndarray,
        post: PostprocessResult,
        t1: Any,
        t2: Any,
        *,
        suppressed: bool,
        warnings: list[str],
    ) -> dict[str, str | None]:
        """Persist the change map server-side and return refs, never paths.

        F-16 (owner ruling 2026-09-23): **never expose filesystem paths**. v1 has
        no artifact-serving endpoint, so an unavailable ref is `null` and an
        explicit non-retrievable warning is emitted. A URI is NOT fabricated in
        its place -- `artifact://` appears in the older API_CONTRACT section 2.4
        example and no production file has ever emitted one, so inventing one
        here would trade a path disclosure for a broken promise.

        The map is still WRITTEN. It is the operator's diagnostic, and the
        suppression case below is a property of the file on disk. What changes is
        only that its location is a server-side fact, not a client-facing one.

        This is the sibling site of the same ruling: `specialists/optical_sar`
        carries the view refs and this carries the change-map ref. Both had to be
        brought into line; fixing one and not the other leaves the ruling half
        implemented, which is how the divergence would survive a green suite.

        When the pair is suppressed the map is written WITHOUT georeferencing.
        Copying T1's transform onto it would produce a file that unlocks
        exactly the placed-on-a-map reading the suppression exists to forbid:
        a downstream tool would happily render it over the wrong ground. The
        pixels are still available for inspection; the location claim is not.
        """
        if self.artifact_dir is None:
            return {"change_map": None}

        try:
            import rasterio  # noqa: F401  (presence check only)

            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(t1.path).stem
            out = self.artifact_dir / f"change_map_{stem}.tif"

            from preprocessing.raster import read_bands, write_raster

            _arr, profile = read_bands(t1.path)
            profile = {k: v for k, v in profile.items() if k != "count"}
            if suppressed:
                profile = {
                    k: v for k, v in profile.items()
                    if k not in {"crs", "transform", "bounds"}
                }
                profile["crs"] = None
                profile["transform"] = None
            write_raster(
                out,
                (probabilities * 255.0).astype(np.uint8),
                profile,
            )
        except Exception:  # noqa: BLE001
            # Artifact rendering is a presentation concern. Failing the whole
            # analysis because a raster could not be written would be wrong.
            warnings.append(
                "the change map could not be rendered server-side, so no "
                "artifact exists for this run and change_map is null"
            )
            return {"change_map": None}

        warnings.append(
            "the change map was rendered for server-side diagnostics but is "
            "NOT retrievable: v1 has no artifact-serving endpoint, so "
            "change_map is null by design"
        )
        return {"change_map": None}

    def _geospatial(self, t1: Any, assessment: PairAssessment) -> GeoMetadata:
        """T1's georeferencing, withheld when the pair cannot be trusted.

        Reported even when the CRS is missing so the caller can see what the
        raster actually carried. When alignment is unusable the transform is
        dropped, because a transform invites the caller to place a region on a
        map, and we have just said the regions are not placeable.
        """
        geo = t1.geo
        if not assessment.registration.is_usable:
            return GeoMetadata(
                crs=geo.crs,
                width=geo.width,
                height=geo.height,
                has_crs=geo.has_crs,
                is_georeferenced=False,
            )
        return geo

    # -- confidence --------------------------------------------------------

    def _confidence_components(
        self,
        probabilities: np.ndarray,
        post: PostprocessResult,
        reg: RegistrationQuality,
        *,
        suppressed: bool,
    ) -> tuple[dict[str, float], float]:
        """Measurable signals only. No LLM, no sampling (plan section 26).

        Three components, each traceable to a number this specialist computed:

            registration_quality     0 at a failed measurement, 1 when aligned
            mean_change_probability  mean probability over the KEPT regions
            component_stability      how far above the threshold those regions
                                     sit, normalized by STABILITY_HEADROOM

        Composition: a weighted sum, then a MINIMUM against the registration
        factor. The minimum is the load-bearing part. A weighted sum alone
        would let a bright, stable-looking change map carry a mis-registered
        pair to a respectable score -- and an unregistered pair produces
        edge-change everywhere, which is exactly the bright, stable output
        that would win the sum. Taking the minimum makes alignment a gate.
        """
        reg_factor = self._registration_factor(reg)

        if post.regions:
            probs = np.asarray(probabilities, dtype=np.float64)
            kept_ids = np.zeros(probs.shape, dtype=bool)
            # Re-derive membership from the cleaned mask so the statistic
            # describes the regions actually reported.
            for r in post.regions:
                c0, r0, c1, r1 = r.bbox_px
                kept_ids[r0:r1, c0:c1] = True

            in_region = probs[kept_ids] if kept_ids.any() else np.array([])
            mean_p = float(in_region.mean()) if in_region.size else 0.0
            headroom = (
                float(np.clip((in_region - self.threshold).mean() / STABILITY_HEADROOM,
                              0.0, 1.0))
                if in_region.size else 0.0
            )
        else:
            mean_p = 0.0
            headroom = 0.0

        components: dict[str, float] = {
            "registration_quality": reg_factor,
            "mean_change_probability": float(np.clip(mean_p, 0.0, 1.0)),
            "component_stability": float(np.clip(headroom, 0.0, 1.0)),
            "n_regions": float(len(post.regions)),
            "shift_magnitude_px": float(reg.shift_magnitude),
        }

        trained = self.has_checkpoint
        components["trained_checkpoint"] = 1.0 if trained else 0.0
        if suppressed:
            # Recorded BEFORE any early return, so a pair that is BOTH
            # mis-registered and untrained reports both facts. An earlier
            # ordering returned on the untrained floor first and left this
            # component unset, which made the trace say only half of what was
            # wrong.
            components["suppressed_by_registration"] = 1.0
        if not post.regions:
            components["no_regions_detected"] = 1.0

        # -- floors --------------------------------------------------------
        # Each is a SIGNAL GAP, not a weak signal, so each drives the score to
        # 0.0 rather than merely discounting it:
        #
        #   suppressed  mis-registered pair; change along every edge is
        #               expected, so a change map says nothing
        #   untrained   random weights; any region is noise wearing a box
        #   no regions  nothing was detected, so there is nothing to be
        #               confident ABOUT
        #
        # The third replaces an earlier behaviour where an empty result still
        # scored ~0.5, because a well-aligned pair contributes its full 0.50
        # registration weight while both signal terms are 0.0. A confidence of
        # 0.5 on "no change found" reads as "we are fairly sure", which
        # overstates it; the honest reading is the no-information case that
        # `_safe_div` in metrics/change.py also resolves to 0.0 rather than NaN.
        if suppressed or not trained or not post.regions:
            return components, 0.0

        weighted = sum(
            CONFIDENCE_WEIGHTS[k] * max(0.0, min(1.0, components.get(k, 0.0)))
            for k in CONFIDENCE_WEIGHTS
        )
        return components, float(max(0.0, min(1.0, min(weighted, reg_factor))))

    def _registration_factor(self, reg: RegistrationQuality) -> float:
        """Map a RegistrationQuality onto [0, 1].

        Reuses the quality object's own `confidence_factor`, which already
        combines the shift and response penalties conservatively. An unusable
        verdict is floored at 0.0 regardless of what the arithmetic says, so
        a pair that failed the gate can never contribute a positive factor.
        """
        if not reg.is_usable:
            return 0.0
        return float(np.clip(reg.confidence_factor(), 0.0, 1.0))

    @staticmethod
    def _degradation_reason(*, degraded: bool, suppressed: bool) -> str | None:
        reasons: list[str] = []
        if suppressed:
            reasons.append("pair not co-registered; spatial claims suppressed")
        if degraded:
            reasons.append("no trained change checkpoint loaded")
        return "; ".join(reasons) if reasons else None

    # -- evidence ----------------------------------------------------------

    def _build_evidence(
        self,
        probabilities: np.ndarray,
        post: PostprocessResult,
        regions: list[ChangeRegion],
        reg: RegistrationQuality,
        *,
        artifacts: dict[str, str | None],
        suppressed: bool,
        n_withheld: int,
    ) -> list[Evidence]:
        """Evidence derived from what the specialist actually computed."""
        items: list[Evidence] = []

        items.append(
            Evidence(
                type=EvidenceType.CHANGE_MAP,
                source_specialist=self.name,
                coordinate_system=CoordinateSystem.NORMALIZED_0_1,
                # F-16 (owner ruling 2026-09-23): always None. `artifacts` never
                # carries a path any more, so this is None on every path -- the
                # map was written, but its location is a server-side fact. The
                # statistics below are what the client is given instead.
                artifact_ref=artifacts.get("change_map"),
                score=None if suppressed else self._mean_probability(probabilities),
                payload={
                    "threshold": self.threshold,
                    "n_components_raw": post.n_components_raw,
                    "n_components_kept": post.n_components_kept,
                    "total_change_pixels": post.total_change_pixels,
                    "suppressed_by_registration": suppressed,
                },
            )
        )

        for i, region in enumerate(regions):
            items.append(
                Evidence(
                    type=EvidenceType.BOUNDING_BOX,
                    source_specialist=self.name,
                    coordinates=[
                        region.box.x1, region.box.y1,
                        region.box.x2, region.box.y2,
                    ],
                    coordinate_system=CoordinateSystem.NORMALIZED_0_1,
                    score=region.mean_probability,
                    payload={
                        "rank": i,
                        "area_pixels": region.area_pixels,
                        "label": "change",
                    },
                )
            )

        items.append(
            Evidence(
                type=EvidenceType.STATISTIC,
                source_specialist=self.name,
                score=None if suppressed else reg.confidence_factor(),
                payload={
                    "registration": reg.to_dict(),
                    "usable_for_spatial_claims": not suppressed,
                    "n_regions_emitted": len(regions),
                    "n_regions_withheld": n_withheld,
                },
            )
        )

        if suppressed and n_withheld:
            # The withheld-region record. Deliberately a STATISTIC, not a
            # BOUNDING_BOX: emitting the boxes would defeat the suppression.
            # It is an audit entry describing what was discarded and why.
            items.append(
                Evidence(
                    type=EvidenceType.STATISTIC,
                    source_specialist=self.name,
                    score=None,
                    payload={
                        "spatial_claims_suppressed": True,
                        "n_withheld": n_withheld,
                        "reason": (
                            "the two acquisitions are not co-registered; a "
                            "change region would assert a location the "
                            "measurement cannot support"
                        ),
                    },
                )
            )

        return items

    @staticmethod
    def _mean_probability(probabilities: np.ndarray) -> float | None:
        arr = np.asarray(probabilities, dtype=np.float64)
        if arr.size == 0:
            return None
        return float(np.clip(arr.mean(), 0.0, 1.0))

    # -- contract ----------------------------------------------------------

    def produce_evidence(self, result: SpecialistResult) -> list[Evidence]:
        return result.evidence

    def estimate_confidence(self, result: SpecialistResult) -> ConfidenceBreakdown:
        return result.confidence

    def model_refs(self) -> list[dict[str, str]]:
        refs: list[dict[str, str]] = []
        if self.model is not None:
            refs.append({
                "name": "STANetStyleChangeDetector",
                "revision": (
                    f"trained params={self.model.num_parameters()}"
                    if self.has_checkpoint
                    else "UNTRAINED (randomly initialised)"
                ),
                "role": "change_detector",
            })
        else:
            refs.append({
                "name": "STANetStyleChangeDetector",
                "revision": "not constructed",
                "role": "change_detector",
            })
        return refs


def build_change_specialist(
    config: Any,
    checkpoint_path: str | Path | None = None,
    *,
    artifact_dir: str | Path | None = None,
    device: str | None = None,
) -> ChangeSpecialist:
    """Construct the specialist from the central config.

    Args:
        checkpoint_path: a trained `.pt` written by `save_change_model`. When
            None, or when the file does not exist, the specialist builds a
            randomly-initialised detector and runs in DEGRADED mode, saying so
            in its result rather than failing.

    Raises:
        ModelLoadError: a checkpoint was named and exists but cannot be read.
            That is a corrupt artifact, not a missing one, and the two must
            not be confused -- silently running an untrained model because a
            real checkpoint failed to load would be the worst outcome.
    """
    from specialists.change.stanet import (
        STANetStyleChangeDetector,
        load_change_model,
    )

    device = device or config.device_preference

    model: Any
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        model = load_change_model(checkpoint_path, device=device)
        # load_change_model already calls eval(); mark provenance so
        # `has_checkpoint` can distinguish this from a random build.
        model._satquery_trained = True  # type: ignore[attr-defined]
    else:
        model = STANetStyleChangeDetector(
            width=int(config.get("change.decoder_width", 128)),
            pretrained=False,
            sa_mode=str(config.get("change.sa_mode", "PAM")),
            attention_budget_bytes=int(
                config.get("change.attention_budget_bytes", 256 * 1024 * 1024)
            ),
        )
        model._satquery_trained = False  # type: ignore[attr-defined]
        model.to(device)
        model.eval()

    return ChangeSpecialist(
        model=model,
        threshold=float(config.get("change.threshold", 0.5)),
        min_component_pixels=int(config.get("change.min_component_pixels", 32)),
        max_shift_px=int(config.get("change.max_shift_px", DEFAULT_MAX_SHIFT_PX)),
        artifact_dir=artifact_dir,
        device=device,
    )


__all__ = [
    "ChangeSpecialist",
    "PairAssessment",
    "build_change_specialist",
    "CONFIDENCE_WEIGHTS",
    "STABILITY_HEADROOM",
    "DEFAULT_MAX_SHIFT_PX",
    "MIN_REGION_PIXELS",
]
