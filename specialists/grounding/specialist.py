"""SatQuery AI — grounding specialist (Workflow C).

Turns a natural-language phrase plus one image into boxes, with evidence and a
measurable confidence. This is the module that makes the Phase 8 trained head
actually usable; before it existed, `head.pt` scored 0.2566 mean best IoU and
nothing in the system could call it.

FROZEN CONTRACT
---------------
    RemoteCLIP ViT-B/32 @ 224   ->  (49, 512) patch tokens, projected dim 512
    GroundingHead               ->  (1, 49, 5) = [tx, ty, tw, th, objectness]
    decode_cell_relative        ->  (1, 49, 4) normalized xyxy
    sigmoid(objectness)         ->  (49,) scores
    NMS                         ->  boxes

THE HEAD IS FROZEN AT INFERENCE
-------------------------------
`head.pt` is loaded in eval mode and never trained here. Training lives in
`training/grounding/train.py`; this module only runs the trained artifact.

DEGRADED MODES (plan section 37)
--------------------------------
Two, both explicit, neither fabricating output:

  no head checkpoint  -> fall back to the zero-shot decode, and SAY SO in
                         `warnings` and in the confidence components
  unusable image      -> refuse, do not call the model

The first is a real deployment case: the HF Space may ship without `head.pt`.

WHY THE VLM IS NOT ASKED FOR COORDINATES
----------------------------------------
Finding C-5 and plan section 14. The head produces the box; SmolVLM may only
describe what is inside it. Asking a language model to invent coordinates
produces confident numbers with no basis, which is the failure mode the whole
evidence system exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.code_revision import REPO_ROOT
from core.errors import InvalidRequestError, ModelUnavailableError, SpecialistError
from core.schemas import (
    Box,
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
from specialists.grounding.inference import (
    DEFAULT_DELTA,
    candidate_boxes,
    decode_candidates_from_features,
)
from preprocessing.imagery import load_pil_image

#: Weights for the measurable confidence components. These are a starting
#: point, not a calibrated mapping -- Phase 13 fits temperature scaling on
#: VALIDATION data. Nothing here is tuned on the eval split.
CONFIDENCE_WEIGHTS: dict[str, float] = {
    "max_objectness": 0.40,
    "top_mean_objectness": 0.35,
    "score_contrast": 0.25,
}

#: A candidate box covering more than this fraction of the normalized frame is
#: DEGENERATE: it says "somewhere in this image", which is what a failed
#: localisation looks like once it has been smoothed into a rectangle. The
#: result schema would still accept it, so the failure would be invisible to
#: any downstream consumer. It is dropped and recorded instead.
#:
#: Measured need (Phase 8 fallback E2E): the zero-shot decode returned a
#: full-frame box -- x1=0.000 y1=0.000 x2=1.000 y2=1.000, area 1.000 -- as its
#: top candidate for a query it could not actually locate.
#:
#: Applies to the ZERO-SHOT FALLBACK PATH ONLY. The trained-head decode is a
#: regressed box with its own learned prior; filtering it would change the
#: frozen Phase 8 benchmark numbers.
DEGENERATE_AREA_FRACTION = 0.9


def _box_area(box: list[float]) -> float:
    """Area of a normalized xyxy box. Zero for inverted or zero-width boxes."""
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    return width * height


@dataclass(frozen=True)
class DecodedBox:
    """One predicted box with the signal that produced it."""

    box: list[float]           # normalized xyxy
    score: float
    source: str                # "head" | "zero_shot"


class GroundingSpecialist(Specialist):
    """Text-guided grounding over one image."""

    name = "grounding"
    version = "0.1.0"
    capabilities = ("grounding",)

    def __init__(
        self,
        encoder: Any,
        head: Any | None = None,
        *,
        grid: int = 7,
        nms_iou: float = 0.50,
        max_candidates: int = 6,
        score_threshold: float = 0.30,
        delta: float = DEFAULT_DELTA,
        head_top_k: int = 5,
        resolution: int = 224,
        head_report: "HeadLoadReport | None" = None,
    ) -> None:
        if grid < 1:
            raise SpecialistError(f"grid must be >= 1, got {grid}")
        if not 0.0 <= nms_iou <= 1.0:
            raise SpecialistError(f"nms_iou must be in [0,1], got {nms_iou}")
        if max_candidates < 1:
            raise SpecialistError(
                f"max_candidates must be >= 1, got {max_candidates}"
            )

        self.encoder = encoder
        self.head = head
        self.grid = grid
        self.nms_iou = nms_iou
        #: Default 6, NOT the config's 20. The Phase 8 decision measured that a
        #: 20-box budget inflates mean best IoU relative to the 5.99-box
        #: baseline; 6 is the decode-matched setting. See
        #: docs/PHASE8_GROUNDING_HEAD_DECISION.md.
        self.max_candidates = max_candidates
        self.score_threshold = score_threshold
        self.delta = delta
        self.head_top_k = head_top_k
        self.resolution = resolution
        #: Why the head is or is not loaded. Carried so the result can say WHICH
        #: thing went wrong -- "no head configured" and "the head is on disk but
        #: will not load" need different actions, and a bare
        #: `degraded=True` collapses them.
        self.head_report = head_report or HeadLoadReport(
            requested_path=None,
            resolved_path=None,
            loaded=head is not None,
            source="injected" if head is not None else "none",
            reason=(
                None
                if head is not None
                else "a head object was not supplied and no path was resolved"
            ),
        )

        if self.head is not None:
            self.head.eval()

    # -- properties --------------------------------------------------------

    @property
    def has_head(self) -> bool:
        return self.head is not None

    # -- validation --------------------------------------------------------

    def validate_request(self, request: SpecialistRequest) -> None:
        if request.asset_count != 1:
            raise InvalidRequestError(
                f"grounding operates on exactly one image; got {request.asset_count}",
                context={"specialist": self.name, "assets": request.asset_count},
            )
        if not request.query.strip():
            raise InvalidRequestError(
                "grounding requires a non-empty phrase to locate",
                context={"specialist": self.name},
            )
        asset = request.assets[0]
        if not Path(asset.path).exists():
            raise InvalidRequestError(
                f"asset path does not exist: {asset.path}",
                context={"specialist": self.name, "path": asset.path},
            )

    # -- execution ---------------------------------------------------------

    def execute(self, request: SpecialistRequest) -> SpecialistResult:
        self.validate_request(request)
        asset = request.assets[0]

        import torch

        try:
            image = load_pil_image(asset.path)
        except SpecialistError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise SpecialistError(
                f"could not prepare image for grounding: {exc}",
                specialist=self.name,
                context={"path": asset.path},
            ) from exc

        try:
            encoded = self.encoder.encode_image(image)
            text = self.encoder.encode_text([request.query])[0]
        except Exception as exc:  # noqa: BLE001
            raise SpecialistError(
                f"encoder failed: {exc}",
                specialist=self.name,
                context={"path": asset.path},
            ) from exc

        warnings: list[str] = []
        degraded = False
        dropped_degenerate: list[DecodedBox] = []

        if self.has_head:
            decoded, score_stats = self._decode_with_head(encoded, text)
        else:
            decoded, score_stats = self._decode_zero_shot(encoded, text)
            warnings.append(
                "no trained grounding head is loaded; fell back to the "
                "zero-shot decode. Spatial precision is substantially lower."
            )
            degraded = True
            # Degenerate-box guard. The zero-shot decode can return a box that
            # spans almost the whole frame. Emitting it as a "region" answers
            # "somewhere in this image", which is not a localisation -- and the
            # result schema would still accept it. Drop it and say so.
            decoded, dropped_degenerate = self._drop_degenerate(decoded)
            if dropped_degenerate:
                warnings.append(
                    f"dropped {len(dropped_degenerate)} degenerate candidate(s) "
                    f"covering more than {DEGENERATE_AREA_FRACTION:.0%} of the "
                    f"frame; they carry no spatial information."
                )

        confidence = self._confidence_for(score_stats, used_head=self.has_head)

        boxes = [
            Box(
                x1=d.box[0], y1=d.box[1], x2=d.box[2], y2=d.box[3],
                score=d.score, label=request.query,
                coordinate_system=CoordinateSystem.NORMALIZED_0_1,
            )
            for d in decoded
        ]

        regions = [
            Region(
                box=b,
                label=request.query,
                score=b.score,
                coordinate_system=CoordinateSystem.NORMALIZED_0_1,
            )
            for b in boxes
        ]

        if boxes:
            answer = (
                f"Located {len(boxes)} candidate region(s) for "
                f"{request.query!r}. Highest objectness "
                f"{score_stats.get('max_objectness', 0.0):.2f}."
            )
        else:
            answer = (
                f"No confident region found for {request.query!r}. Highest "
                f"objectness {score_stats.get('max_objectness', 0.0):.2f}."
            )

        result = SpecialistResult(
            task=Task.GROUNDING,
            answer=answer,
            boxes=boxes,
            regions=regions,
            labels=[request.query],
            confidence=confidence,
            geospatial=asset.geo,
            degraded=degraded,
        )

        # Pixel and geo coordinates are DERIVED, never replacements. The box in
        # the result stays normalized; these are additional evidence so a
        # consumer with a CRS can place them on a map.
        geo_boxes = self._to_geo_boxes(boxes, asset.geo, warnings)

        result.evidence = self._build_evidence(
            decoded,
            score_stats,
            request.query,
            geo_boxes,
            asset.geo,
            dropped_degenerate=dropped_degenerate,
        )
        result.warnings.extend(warnings)
        return result

    # -- decoding ----------------------------------------------------------

    def _decode_with_head(
        self, encoded: Any, text: np.ndarray
    ) -> tuple[list[DecodedBox], dict[str, float]]:
        """Run the trained head and decode boxes."""
        import torch

        from specialists.grounding.head import decode_cell_relative, nms

        patches = torch.from_numpy(encoded.patch_tokens.astype(np.float32)).unsqueeze(0)
        text_t = torch.from_numpy(np.asarray(text, dtype=np.float32)).unsqueeze(0)

        with torch.no_grad():
            out = self.head(patches, text_t)
            coords = decode_cell_relative(out.raw[..., :4], self.grid, self.grid)[0]
            scores = torch.sigmoid(out.raw[..., 4])[0]

        stats = self._score_stats(scores)
        keep = scores >= self.score_threshold
        if int(keep.sum()) == 0:
            # Nothing cleared the gate. Return the single best cell rather than
            # an empty list: a grounding result with no box is not useful, and
            # the confidence already says how weak this is.
            peak = int(scores.argmax())
            return (
                [DecodedBox(box=[float(v) for v in coords[peak]],
                            score=float(scores[peak]), source="head")],
                stats,
            )

        selected_boxes = coords[keep]
        selected_scores = scores[keep]
        idx = nms(selected_boxes, selected_scores, self.nms_iou)[: self.max_candidates]

        decoded = [
            DecodedBox(
                box=[float(v) for v in selected_boxes[int(j)]],
                score=float(selected_scores[int(j)]),
                source="head",
            )
            for j in idx
        ]
        return decoded, stats

    def _decode_zero_shot(
        self, encoded: Any, text: np.ndarray
    ) -> tuple[list[DecodedBox], dict[str, float]]:
        """Fall back to the cosine-similarity decode.

        Uses the SAME `decode_candidates_from_features` the Phase 7 experiment
        and the Phase 8 eval used, so a fallback result is comparable to the
        published baseline rather than to some third decode.
        """
        candidates = decode_candidates_from_features(
            encoded, text, delta=self.delta, top_k=self.head_top_k
        )
        decoded = [
            DecodedBox(box=c.box, score=c.score, source="zero_shot")
            for c in candidates[: self.max_candidates]
        ]
        scores = np.asarray([d.score for d in decoded], dtype=np.float64)
        stats = {
            "max_objectness": float(scores.max()) if scores.size else 0.0,
            "top_mean_objectness": float(scores.mean()) if scores.size else 0.0,
            "score_contrast": (
                float(scores.max() - scores.min()) if scores.size > 1 else 0.0
            ),
        }
        return decoded, stats

    @staticmethod
    def _drop_degenerate(
        decoded: list[DecodedBox],
        *,
        max_area_fraction: float = DEGENERATE_AREA_FRACTION,
    ) -> tuple[list[DecodedBox], list[DecodedBox]]:
        """Split candidates into (kept, dropped-degenerate).

        A box covering more than ``max_area_fraction`` of the normalized frame
        carries no spatial information. It is the shape a *failed* localisation
        takes once it has been smoothed into a rectangle, and it is worse than
        returning nothing at all because the result schema accepts it -- so it
        looks like a successful localisation to anything downstream.

        Both halves are returned: the caller emits ``kept`` and records
        ``dropped`` in the evidence payload. Dropping without recording would
        hide the fact that the decode found nothing usable.

        Applied to the zero-shot fallback only. See DEGENERATE_AREA_FRACTION.
        """
        kept: list[DecodedBox] = []
        dropped: list[DecodedBox] = []
        for d in decoded:
            if _box_area(d.box) > max_area_fraction:
                dropped.append(d)
            else:
                kept.append(d)
        return kept, dropped

    @staticmethod
    def _score_stats(scores: Any) -> dict[str, float]:
        """Measurable objectness statistics. No LLM, no sampling."""
        s = np.asarray(scores.detach().cpu().numpy(), dtype=np.float64)
        if s.size == 0:
            return {"max_objectness": 0.0, "top_mean_objectness": 0.0,
                    "score_contrast": 0.0}
        top_n = min(3, s.size)
        top = np.sort(s)[::-1][:top_n]
        rest = np.sort(s)[:-top_n] if s.size > top_n else np.zeros(1)
        return {
            "max_objectness": float(s.max()),
            "top_mean_objectness": float(top.mean()),
            # How far the peaks stand above the field. A head that is confident
            # everywhere has low contrast and should not be trusted.
            "score_contrast": float(top.mean() - rest.mean()),
        }

    # -- coordinates -------------------------------------------------------

    def _to_geo_boxes(
        self, boxes: list[Box], geo: GeoMetadata, warnings: list[str]
    ) -> list[list[float]]:
        """Convert normalized boxes to geographic bounds when the CRS allows.

        Returns an empty list when the raster has no CRS or transform. That is a
        degradation, not a failure: the normalized boxes are still valid.
        """
        if not geo.has_crs or not geo.transform or not geo.width or not geo.height:
            warnings.append(
                "raster has no CRS/transform; boxes are reported in normalized "
                "coordinates only and cannot be placed on a map"
            )
            return []

        from geospatial.transform import affine_from_metadata, to_geo

        affine = affine_from_metadata(geo)
        if affine is None:
            warnings.append("raster transform could not be reconstructed")
            return []

        out: list[list[float]] = []
        for box in boxes:
            try:
                g = to_geo(box, width=geo.width, height=geo.height, transform=affine)
                out.append([g.x1, g.y1, g.x2, g.y2])
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"box could not be georeferenced: {exc}")
        return out

    # -- confidence --------------------------------------------------------

    def _confidence_for(
        self, stats: dict[str, float], *, used_head: bool
    ) -> ConfidenceBreakdown:
        """Confidence from measurable signals only (plan section 21)."""
        components = dict(stats)
        components["used_head"] = 1.0 if used_head else 0.0

        raw = sum(
            CONFIDENCE_WEIGHTS[k] * max(0.0, min(1.0, stats.get(k, 0.0)))
            for k in CONFIDENCE_WEIGHTS
        )
        raw = max(0.0, min(1.0, raw))

        return ConfidenceBreakdown(
            raw=raw,
            method="uncalibrated",
            components=components,
            degraded=not used_head,
            degradation_reason=(
                None if used_head else self._fallback_reason()
            ),
        )

    def _fallback_reason(self) -> str:
        """Why zero-shot is being used, naming the specific cause.

        A single string like "no trained head loaded" is true of four different
        situations with four different fixes, so the report's own reason is used
        when one is available.
        """
        detail = self.head_report.reason
        if detail:
            return f"zero-shot fallback: {detail}"
        return "zero-shot fallback: no trained head loaded"

    # -- evidence ----------------------------------------------------------

    def _build_evidence(
        self,
        decoded: list[DecodedBox],
        stats: dict[str, float],
        phrase: str,
        geo_boxes: list[list[float]],
        geo: GeoMetadata,
        *,
        dropped_degenerate: list[DecodedBox] | None = None,
    ) -> list[Evidence]:
        """Evidence derived from what the specialist actually computed."""
        items: list[Evidence] = []

        for i, d in enumerate(decoded):
            items.append(
                Evidence(
                    type=EvidenceType.BOUNDING_BOX,
                    source_specialist=self.name,
                    coordinates=d.box,
                    coordinate_system=CoordinateSystem.NORMALIZED_0_1,
                    score=d.score,
                    payload={
                        "rank": i,
                        "phrase": phrase,
                        "decode": d.source,
                        "grid": f"{self.grid}x{self.grid}",
                    },
                )
            )

        if geo_boxes:
            items.append(
                Evidence(
                    type=EvidenceType.GEOLOCATION,
                    source_specialist=self.name,
                    coordinates=geo_boxes[0],
                    coordinate_system=CoordinateSystem.GEO,
                    score=decoded[0].score if decoded else None,
                    payload={"crs": geo.crs, "n_boxes": len(geo_boxes)},
                )
            )

        items.append(
            Evidence(
                type=EvidenceType.STATISTIC,
                source_specialist=self.name,
                score=stats.get("max_objectness"),
                payload={
                    "phrase": phrase,
                    "n_candidates": len(decoded),
                    "decode": "head" if self.has_head else "zero_shot",
                    **{k: round(v, 4) for k, v in stats.items()},
                },
            )
        )

        # Dropped-candidate record. Deliberately NOT a BOUNDING_BOX entry: the
        # box was rejected, so emitting it as spatial evidence would defeat the
        # guard. It is a STATISTIC describing what the decode produced and why
        # it was discarded, which is what an audit needs.
        if dropped_degenerate:
            items.append(
                Evidence(
                    type=EvidenceType.STATISTIC,
                    source_specialist=self.name,
                    score=None,
                    payload={
                        "dropped_degenerate": True,
                        "n_dropped": len(dropped_degenerate),
                        "area_threshold": DEGENERATE_AREA_FRACTION,
                        "reason": (
                            "candidate covered more than "
                            f"{DEGENERATE_AREA_FRACTION:.0%} of the frame; "
                            "emitting it as a region would assert a "
                            "localisation the decode did not make"
                        ),
                        "dropped": [
                            {
                                "box": [round(v, 6) for v in d.box],
                                "score": round(d.score, 4),
                                "area": round(_box_area(d.box), 6),
                                "source": d.source,
                            }
                            for d in dropped_degenerate
                        ],
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
        refs = [
            {
                "name": getattr(self.encoder, "model_name", "RemoteCLIP"),
                "revision": str(getattr(self.encoder, "checkpoint_path", "unpinned")),
                "role": "grounding_encoder",
            }
        ]
        if self.has_head:
            refs.append({
                "name": "GroundingHead",
                "revision": f"params={self.head.num_parameters()}",
                "role": "grounding_head",
            })
        return refs


# ---------------------------------------------------------------------------
# Loading the trained head
# ---------------------------------------------------------------------------
#: The shipped trained grounding head.
#:
#: WHY THIS IS A CODE CONSTANT AND NOT A CONFIG KEY
#: ------------------------------------------------
#: `core/registry.py` already declares `grounding_head.head_path` as an optional
#: config key, and `build_grounding_specialist` already accepts `head_path`. Both
#: existed, and neither was ever set -- measured 2026-09-23: `head_path` appears
#: nowhere in `configs/base.yaml` or `core/config.py`, so the registry passed no
#: kwarg, the builder defaulted to None, and production silently ran the
#: zero-shot baseline while a 12.6 MB trained head sat on disk.
#:
#: The obvious fix -- set `grounding_head.head_path` in `base.yaml` -- was NOT
#: taken, deliberately. `base.yaml` is hashed into the frozen config identity
#: (`Config.hash = 78f1e3700da15aa1`), and that hash is a pinned invariant of
#: this project: reports, run manifests and published numbers cite it. Editing a
#: comment in `base.yaml` moves it. Wiring a default artifact path is not worth
#: invalidating a frozen identity, so the default lives here, in code, and the
#: config key keeps its original meaning: an OVERRIDE of this default.
#:
#: The consequence is that `head_path=None` no longer means "no head". It means
#: "use the shipped head if it is there". An explicit `head_path` still wins, and
#: a path that does not exist still degrades to zero-shot with a stated reason.
DEFAULT_HEAD_PATH: Path = (
    REPO_ROOT / "artifacts" / "grounding" / "remoteclip_grounding_v001" / "head.pt"
)


@dataclass
class HeadLoadReport:
    """What happened when the trained head was looked for, and why.

    Every field is populated from a measurement. `source` is the machine-readable
    version of the same fact, so a caller does not have to parse `reason`.
    """

    requested_path: str | None
    resolved_path: str | None
    loaded: bool
    #: "configured" | "shipped_default" | "injected" | "none"
    source: str
    reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_path": self.requested_path,
            "resolved_path": self.resolved_path,
            "loaded": self.loaded,
            "source": self.source,
            "reason": self.reason,
            "detail": dict(self.detail),
            "note": (
                "`loaded: false` means the zero-shot baseline is in use. The "
                "specialist reports itself degraded and this reason travels with "
                "the result, so a trained head is never claimed when one is not "
                "loaded."
            ),
        }


def load_grounding_head(
    path: str | Path | None,
    *,
    device: str = "cpu",
    feature_dim: int | None = None,
) -> tuple[Any | None, HeadLoadReport]:
    """Load and VALIDATE a trained head. Returns `(head_or_None, report)`.

    Never raises for a missing or broken checkpoint: the caller's fallback is the
    zero-shot baseline, and the report says which of the four situations occurred
    so the operator knows what to fix.

    The four outcomes are kept distinct on purpose:

      * path not supplied and the shipped default absent  -> "none"
      * path supplied but the file does not exist         -> "missing"
      * the file exists but will not load (corrupt, wrong
        torch version, a config dict the head rejects)    -> "invalid"
      * loaded and its architecture matches               -> "loaded"

    Args:
        path: the checkpoint. `None` falls back to `DEFAULT_HEAD_PATH`.
        device: torch device string.
        feature_dim: when given, the loaded head's `feature_dim` must equal it.
            This catches the silent shape drift `build_head` already guards at
            training time -- a head trained against a different encoder width
            would otherwise load and then fail deep inside the similarity step.
    """
    requested = None if path is None else str(path)
    resolved = Path(path) if path is not None else DEFAULT_HEAD_PATH
    source = "configured" if path is not None else "shipped_default"

    if not resolved.exists():
        reason = (
            f"the configured head_path {resolved} does not exist"
            if path is not None
            else f"no head_path was configured and the shipped head is absent at {resolved}"
        )
        return None, HeadLoadReport(
            requested_path=requested,
            resolved_path=str(resolved),
            loaded=False,
            source=source,
            reason=reason,
            detail={"exists": False},
        )

    try:
        from training.grounding.train import load_trained_head

        head = load_trained_head(resolved, device=device)
    except Exception as exc:  # noqa: BLE001
        # A corrupt checkpoint, an unexpected state_dict, a torch version that
        # cannot read it: all land here, and all degrade rather than crash. The
        # exception type and text are recorded, because "invalid" alone does not
        # tell an operator whether to re-download or re-train.
        return None, HeadLoadReport(
            requested_path=requested,
            resolved_path=str(resolved),
            loaded=False,
            source=source,
            reason=(
                f"the head at {resolved} exists but could not be loaded "
                f"({type(exc).__name__}: {exc})"
            ),
            detail={"exists": True, "error_type": type(exc).__name__},
        )

    detail: dict[str, Any] = {
        "exists": True,
        "feature_dim": getattr(head, "feature_dim", None),
        "hidden_dim": getattr(head, "hidden_dim", None),
        "n_parameters": (
            head.num_parameters() if hasattr(head, "num_parameters") else None
        ),
    }

    if feature_dim is not None and detail["feature_dim"] != feature_dim:
        return None, HeadLoadReport(
            requested_path=requested,
            resolved_path=str(resolved),
            loaded=False,
            source=source,
            reason=(
                f"the head at {resolved} was trained with feature_dim="
                f"{detail['feature_dim']} but this deployment expects "
                f"{feature_dim}; loading it would fail at the similarity step"
            ),
            detail=detail,
        )

    return head, HeadLoadReport(
        requested_path=requested,
        resolved_path=str(resolved),
        loaded=True,
        source=source,
        reason=None,
        detail=detail,
    )


def build_grounding_specialist(
    config: Any,
    checkpoint_path: str | Path | None = None,
    head_path: str | Path | None = None,
    device: str | None = None,
) -> GroundingSpecialist:
    """Construct the specialist from the central config.

    Args:
        checkpoint_path: local RemoteCLIP `.pt`; skips the Hub fetch.
        head_path: trained `head.pt`. **None now means "use the shipped head"**
            (`DEFAULT_HEAD_PATH`), not "no head" -- changed 2026-09-23, when it
            was measured that no config key or call site ever supplied this
            argument, so production ran the zero-shot baseline while a trained
            head sat on disk. An explicit path overrides the default. When the
            resolved path is missing, unreadable, or was trained against a
            different encoder width, the specialist runs in zero-shot fallback
            mode and the reason travels with every result.
        device: torch device string.

    Raises:
        ModelUnavailableError: the encoder itself cannot be loaded. Without it
            there is no grounding at all, so this is fatal rather than degraded.
    """
    from specialists.grounding.remoteclip import build_encoder

    device = device or config.device_preference
    resolution = int(config.get("grounding.image_size", 224))

    try:
        encoder = build_encoder(
            config, resolution=resolution, device=device,
            checkpoint_path=checkpoint_path,
        )
    except Exception as exc:  # noqa: BLE001
        raise ModelUnavailableError(
            f"RemoteCLIP encoder could not be loaded: {exc}",
            specialist="grounding",
        ) from exc

    # `head_path=None` means "use the shipped head if it is present", NOT "no
    # head". See DEFAULT_HEAD_PATH for why this default is a code constant rather
    # than a `base.yaml` entry. A head that is absent, corrupt or the wrong shape
    # still degrades to zero-shot, and the report says which.
    expected_feature_dim = int(config.get("grounding_head.feature_dim", 2048))
    head, report = load_grounding_head(
        head_path, device=device, feature_dim=expected_feature_dim
    )

    return GroundingSpecialist(
        encoder=encoder,
        head=head,
        grid=resolution // 32,
        nms_iou=float(config.get("grounding.nms_iou", 0.50)),
        # 6, not the config's max_candidates=20: see the class docstring.
        max_candidates=int(config.get("grounding.serving_max_candidates", 6)),
        score_threshold=float(config.get("grounding.confidence_threshold", 0.30)),
        delta=float(config.get("grounding.baseline_delta", DEFAULT_DELTA)),
        head_top_k=int(config.get("grounding.baseline_top_k", 5)),
        resolution=resolution,
        head_report=report,
    )


__all__ = [
    "GroundingSpecialist",
    "DecodedBox",
    "HeadLoadReport",
    "DEFAULT_HEAD_PATH",
    "load_grounding_head",
    "build_grounding_specialist",
    "CONFIDENCE_WEIGHTS",
    "DEGENERATE_AREA_FRACTION",
]