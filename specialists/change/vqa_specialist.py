"""SatQuery AI — the change-VQA specialist (R-02).

WHAT IT DOES
------------
Takes exactly two temporally corresponding acquisitions plus a change-oriented
question, and returns a short answer:

    T1 (pre) + T2 (post) + "what is the largest change?"
        -> class-wise change estimate
        -> "buildings"

It answers the eight CDVQA question types (`change_or_not`, `change_ratio`,
`change_ratio_types`, `change_to_what`, `increase_or_not`, `decrease_or_not`,
`largest_change`, `smallest_change`) over a closed 19-answer space.

THE DETECTOR IS FROZEN AND SHARED IN SPIRIT, NOT IN INSTANCE
------------------------------------------------------------
The STANet detector is loaded frozen and never updated. When the planner routes a
change+language request it plans BOTH a `change` step and a `change_vqa` step, so
the detector runs twice for one request. That is a real cost and it is stated
rather than hidden:

    * the model weights are loaded ONCE — the registry caches the specialist
      instance, so the second cost is a forward pass, not a 60 MB load;
    * the alternative (passing the change map from the `change` step into this
      one) needs the controller to hand artifacts between steps, which it does
      not do today. Inventing that here would be a second execution model.

`execute` therefore accepts an optional `change_map` path in `request.params` for
a future planner to populate. It is unused today and documented as such, so the
hook exists without pretending the wiring does.

DEGRADED MODE PRODUCES NO ANSWER, ON PURPOSE
--------------------------------------------
`ChangeSpecialist` degrades to an untrained detector and still emits a change
map. This specialist must not copy that, because the outputs are not comparable:
a random change map is visibly noise, whereas an untrained 19-way classifier
still emits a fluent, confident-looking `yes`. A user cannot tell the second from
a real answer, so the untrained case returns **no answer at all** — `answer=""`,
`degraded=True`, and a warning naming exactly which piece is missing. The
controller then falls through to another attributed specialist, which is the
correct behaviour for "this capability is not available yet".

That matters for the project's current state: R-02 is
`IMPLEMENTATION_READY_TRAINING_PENDING`, so with no trained head on disk this
specialist registers as AVAILABLE (it constructs) but answers nothing until the
trained artifact is dropped in. Registering it as UNAVAILABLE would be a
different claim, and the wrong one — the capability exists, its weights do not
yet.

CONFIDENCE IS UNCALIBRATED AND SAYS SO
--------------------------------------
`ConfidenceBreakdown.method` is `"uncalibrated"`. `raw` is the softmax
probability of the chosen answer; `components` carries the top1-top2 margin, the
number of distinct answers seen, and the estimator's mean magnitude. Nothing is
fitted — the R-03 calibration contract is a separate, open ruling — so
`calibrated` stays None and `value` therefore returns `raw`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core.errors import (
    InvalidRequestError,
    ModelLoadError,
    SpecialistError,
)
from core.schemas import (
    ConfidenceBreakdown,
    Evidence,
    EvidenceType,
    SpecialistResult,
    Task,
)
from specialists.base import Specialist, SpecialistRequest
from training.change_vqa.vocab import (
    CHANGE_CLASS_ORDER,
    INDEX_TO_ANSWER,
    N_ANSWERS,
    resolve_question_type,
)

#: T1 is the PRE acquisition, T2 the POST one. `label1=pre,label2=post` is
#: proven (agreement 1.0000 over all 2,968 scenes); the image-level leg
#: (`im1`=pre, `im2`=post) is SUPPORTED statistically but not proven. The
#: distinction travels in the evidence payload so a reader can see which leg the
#: result rests on.
TEMPORAL_ORDER_NOTE = (
    "T1=pre, T2=post. label1=pre/label2=post proven (agreement 1.0000 over "
    "2,968 scenes); im1=pre/im2=post supported statistically, not proven."
)

#: Above this raw softmax the specialist will not call the answer
#: low-confidence. It is a reporting threshold, not a calibration.
LOW_CONFIDENCE_THRESHOLD = 0.40


class ChangeVQASpecialist(Specialist):
    """Answers a change question about a pair of acquisitions."""

    name = "change_vqa"
    version = "0.1.0"
    capabilities = ("change_vqa",)

    def __init__(
        self,
        *,
        head: Any | None = None,
        feature_extractor: Any | None = None,
        text_encoder: Any | None = None,
        artifact_dir: str | Path | None = None,
        device: str = "cpu",
        apply_type_mask: bool = True,
        head_path: str | Path | None = None,
        head_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.head = head
        self.feature_extractor = feature_extractor
        self.text_encoder = text_encoder
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.device = device
        # The type mask restricts the answer to those legal for the question
        # type. On by default in SERVING because the type is known from the
        # question and an illegal answer is never right; the evaluation reports
        # masked and unmasked separately so the mask's contribution is visible.
        self.apply_type_mask = bool(apply_type_mask)
        self.head_path = str(head_path) if head_path else None
        #: The `model_metadata.json` written beside the head. Carries the feature
        #: spec the head was TRAINED on, which serving must match.
        self.head_metadata: dict[str, Any] = dict(head_metadata or {})

        if self.head is not None:
            self.head.eval()

    # -- properties --------------------------------------------------------

    @property
    def has_head(self) -> bool:
        """Whether a TRAINED reasoning head is loaded.

        `head` is None until a checkpoint is found. This is the property that
        decides whether an answer can be produced at all — see the module
        docstring on why an untrained head must not answer.
        """
        return self.head is not None and bool(
            getattr(self.head, "_satquery_trained", False)
        )

    @property
    def detector_trained(self) -> bool:
        return bool(
            self.feature_extractor is not None
            and getattr(self.feature_extractor, "trained", False)
        )

    def feature_spec_mismatch(self) -> str | None:
        """Whether the head was trained on a DIFFERENT representation than this
        deployment produces, or None when they agree.

        WHY THIS IS A REFUSAL AND NOT A WARNING
        ---------------------------------------
        A head is only meaningful for the feature distribution it was fitted on.
        Measured on this repository: `configs/base.yaml`'s `change:` section
        carries no `checkpoint_path` key, so the registry builds the change
        feature extractor with `checkpoint_path=None` and gets an UNTRAINED
        STANet — while `scripts/prepare_change_vqa.py` defaults to the trained
        LEVIR checkpoint. Training and serving would therefore consume different
        representations, and the head would still emit a fluent `yes`.

        That is the same failure the untrained-head guard exists to prevent, so
        it gets the same treatment: refuse to answer, and name both specs.

        The spec hash already encodes trained-vs-untrained
        (`feature_spec_hash(..., extractor="stanet:trained"|"stanet:untrained")`),
        so comparing it covers the detector-state mismatch too.
        """
        if self.feature_extractor is None:
            return None
        trained_spec = self.head_metadata.get("change_cache_spec")
        serving_spec = getattr(self.feature_extractor, "spec_hash", None)
        if not trained_spec or not serving_spec:
            return None
        if trained_spec == serving_spec:
            return None
        trained_detector = self.head_metadata.get("detector_trained")
        return (
            f"the head was trained on change-feature spec {trained_spec!r} "
            f"(detector_trained={trained_detector}) but this deployment produces "
            f"{serving_spec!r} (detector_trained={self.detector_trained}). "
            f"Answering anyway would compute the answer from a representation "
            f"the head was never fitted on"
        )

    def unavailable_reason(self) -> str | None:
        """Which piece is missing or mismatched, or None when servable."""
        missing = []
        if not self.has_head:
            missing.append(
                "a trained change-VQA head (expected at "
                f"{self.head_path or 'change_vqa.head_path'})"
            )
        if self.feature_extractor is None:
            missing.append("a change feature extractor")
        if self.text_encoder is None:
            missing.append("a question text encoder")
        mismatch = self.feature_spec_mismatch()
        if mismatch:
            missing.append(mismatch)
        if not missing:
            return None
        return "missing " + ", ".join(missing)

    # -- validation --------------------------------------------------------

    def validate_request(self, request: SpecialistRequest) -> None:
        """Exactly two assets, both present on disk.

        The one-asset mistake is the common one: the caller treated a change
        question as an ordinary VQA question about a single image, which would
        produce an answer about one acquisition rather than about what changed
        between two.
        """
        if request.asset_count != 2:
            raise InvalidRequestError(
                f"{self.name} requires exactly 2 assets (T1 pre, T2 post); "
                f"got {request.asset_count}",
                user_message=(
                    "A change question needs exactly two images: an earlier one "
                    "and a later one."
                ),
                context={
                    "specialist": self.name,
                    "expected": 2,
                    "actual": request.asset_count,
                },
            )
        for index, asset in enumerate(request.assets):
            if not Path(asset.path).exists():
                raise InvalidRequestError(
                    f"asset T{index + 1} path does not exist: {asset.path}",
                    context={"specialist": self.name, "path": asset.path},
                )

    # -- execution ---------------------------------------------------------

    def execute(self, request: SpecialistRequest) -> SpecialistResult:
        self.validate_request(request)

        reason = self.unavailable_reason()
        if reason is not None:
            return self._unavailable_result(request, reason)

        from preprocessing.imagery import load_image_array

        t1 = load_image_array(request.assets[0].path)
        t2 = load_image_array(request.assets[1].path)

        vector, facts = self.feature_extractor.extract(t1, t2)
        text_vector = self.text_encoder.encode([request.query])[0]

        from training.change_vqa.model import predict_answers

        qtype, resolved = resolve_question_type(request.query)
        from training.change_vqa.vocab import QUESTION_TYPE_TO_INDEX

        qtype_index = QUESTION_TYPE_TO_INDEX.get(qtype, 0)
        temporal_index = _temporal_index(request.query)

        predictions = predict_answers(
            self.head,
            change_features=np.asarray([vector], dtype=np.float32),
            text_features=np.asarray([text_vector], dtype=np.float32),
            qtype_indices=np.asarray([qtype_index], dtype=np.int64),
            temporal_indices=np.asarray([temporal_index], dtype=np.int64),
            qtypes=[qtype],
            apply_type_mask=self.apply_type_mask,
            device=self.device,
        )

        answer_index = int(predictions["answer_index"][0])
        answer = INDEX_TO_ANSWER[answer_index]
        confidence = float(predictions["confidence"][0])
        margin = float(predictions["margin"][0])

        warnings: list[str] = []
        degraded = False
        if not resolved:
            warnings.append(
                f"the question type could not be resolved from the question "
                f"text; the answer is unconstrained (treated as {qtype!r})"
            )
        if not self.detector_trained:
            degraded = True
            warnings.append(
                "the change feature extractor was built from an UNTRAINED "
                "STANet detector, so the change representation the answer rests "
                "on is not the trained one the plan specifies"
            )
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            warnings.append(
                f"low raw confidence ({confidence:.3f}); the head is "
                f"uncalibrated so this is not a probability of being correct"
            )

        return SpecialistResult(
            task=Task.CHANGE_VQA,
            answer=answer,
            labels=[answer],
            evidence=self._answer_evidence(
                predictions, answer_index, answer, confidence, margin, qtype
            ),
            confidence=self._confidence(predictions, confidence, margin),
            degraded=degraded,
            warnings=warnings,
            execution_trace=None,
        )

    def _unavailable_result(
        self, request: SpecialistRequest, reason: str
    ) -> SpecialistResult:
        """No trained head: say so, and answer nothing.

        `answer=""` is deliberate. An untrained 19-way classifier still emits a
        well-formed `yes`, and a reader cannot tell it from a real answer. The
        `SpecialistResult` validator marks an empty change-VQA answer as
        degraded, so this cannot be mistaken for a success downstream.
        """
        return SpecialistResult(
            task=Task.CHANGE_VQA,
            answer="",
            labels=[],
            evidence=[],
            confidence=ConfidenceBreakdown(
                raw=0.0,
                calibrated=None,
                method="unavailable",
                components={},
                degraded=True,
                degradation_reason=reason,
            ),
            degraded=True,
            warnings=[
                f"change-VQA cannot answer: {reason}. No answer was produced "
                f"rather than a plausible-looking untrained one.",
                TEMPORAL_ORDER_NOTE,
            ],
            execution_trace=None,
        )

    # -- evidence and confidence -------------------------------------------

    def _answer_evidence(
        self,
        predictions: dict[str, Any],
        answer_index: int,
        answer: str,
        confidence: float,
        margin: float,
        qtype: str,
    ) -> list[Evidence]:
        """Two pieces: the answer distribution, and the change estimate.

        The second is what makes the answer auditable. A reader can see WHICH
        class the model believed changed and by how much, rather than only the
        final word — which is the difference between an explainable answer and a
        lucky one.
        """
        probabilities = predictions["probabilities"][0]
        ranked = np.argsort(-probabilities)[:5]

        answer_evidence = Evidence(
            type=EvidenceType.STATISTIC,
            source_specialist=self.name,
            score=confidence,
            payload={
                "kind": "answer_distribution",
                "question_type": qtype,
                "answer": answer,
                "answer_index": answer_index,
                "n_answers": N_ANSWERS,
                "top_5": [
                    {
                        "answer": INDEX_TO_ANSWER[int(i)],
                        "probability": round(float(probabilities[int(i)]), 6),
                    }
                    for i in ranked
                ],
                "top1_minus_top2": round(margin, 6),
                "type_mask_applied": bool(
                    predictions.get("legal_mask_applied", False)
                ),
                "confidence_method": "uncalibrated",
            },
        )

        class_mag = predictions["class_mag"][0]
        class_delta = predictions["class_delta"][0]
        total_changed = float(predictions["total_changed"][0])
        estimate_evidence = Evidence(
            type=EvidenceType.STATISTIC,
            source_specialist=self.name,
            score=total_changed,
            payload={
                "kind": "class_wise_change_estimate",
                "class_order": list(CHANGE_CLASS_ORDER),
                "class_change_magnitude": [
                    round(float(v), 6) for v in class_mag
                ],
                "class_change_delta": [round(float(v), 6) for v in class_delta],
                "total_changed_fraction": round(total_changed, 6),
                "estimator_is_learned": True,
                "temporal_order": TEMPORAL_ORDER_NOTE,
            },
        )
        return [answer_evidence, estimate_evidence]

    def _confidence(
        self, predictions: dict[str, Any], confidence: float, margin: float
    ) -> ConfidenceBreakdown:
        """Raw softmax plus measured context. Nothing is fitted."""
        probabilities = predictions["probabilities"][0]
        top1, top2 = np.sort(probabilities)[-2:]
        return ConfidenceBreakdown(
            raw=float(np.clip(confidence, 0.0, 1.0)),
            calibrated=None,
            method="uncalibrated",
            components={
                "top1_minus_top2": round(float(margin), 6),
                "top1_probability": round(float(top1), 6),
                "top2_probability": round(float(top2), 6),
                "n_answers_considered": float(N_ANSWERS),
                "entropy_normalised": round(
                    float(
                        -np.sum(probabilities * np.log(probabilities + 1e-12))
                        / np.log(N_ANSWERS)
                    ),
                    6,
                ),
            },
            degraded=False,
            degradation_reason=None,
        )

    # -- contract ----------------------------------------------------------

    def produce_evidence(self, result: SpecialistResult) -> list[Evidence]:
        """Return the evidence already computed. Never invents a new artefact."""
        return list(result.evidence)

    def estimate_confidence(self, result: SpecialistResult) -> ConfidenceBreakdown:
        return result.confidence

    def model_refs(self) -> list[dict[str, str]]:
        refs = [
            {
                "name": "change_vqa_head",
                "revision": self.head_path or "not_loaded",
                "role": "reasoning head (trained)",
            }
        ]
        if self.feature_extractor is not None:
            refs.append(
                {
                    "name": "stanet_change_detector",
                    "revision": str(
                        getattr(self.feature_extractor, "spec_hash", "unknown")
                    ),
                    "role": "frozen change feature extractor (trained)"
                    if self.detector_trained
                    else "frozen change feature extractor (UNTRAINED)",
                }
            )
        if self.text_encoder is not None:
            refs.append(
                {
                    "name": "question_encoder",
                    "revision": "sentence-transformers/all-MiniLM-L6-v2",
                    "role": "frozen question encoder",
                }
            )
        return refs

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base.update(
            {
                "has_trained_head": self.has_head,
                "detector_trained": self.detector_trained,
                "unavailable_reason": self.unavailable_reason(),
                "feature_spec_mismatch": self.feature_spec_mismatch(),
                "trained_feature_spec": self.head_metadata.get(
                    "change_cache_spec", "unavailable"
                ),
                "serving_feature_spec": (
                    getattr(self.feature_extractor, "spec_hash", None)
                    if self.feature_extractor is not None else None
                ),
                "confidence_method": "uncalibrated",
            }
        )
        return base


def _temporal_index(question: str) -> int:
    from training.change_vqa.vocab import (
        TEMPORAL_REFERENCE_TO_INDEX,
        resolve_temporal_reference,
    )

    return TEMPORAL_REFERENCE_TO_INDEX.get(resolve_temporal_reference(question), 0)


def build_change_vqa_specialist(
    config: Any,
    checkpoint_path: str | Path | None = None,
    *,
    head_path: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    device: str | None = None,
) -> ChangeVQASpecialist:
    """Construct the specialist from the central config.

    Args:
        checkpoint_path: the frozen STANet detector (the same artifact
            `ChangeSpecialist` uses). Missing => an untrained extractor, recorded
            as such.
        head_path: the trained change-VQA head. Missing => the specialist
            constructs but cannot answer, and says so.

    Raises:
        ModelLoadError: a checkpoint was named and exists but cannot be read.
            That is a corrupt artifact, not a missing one. Degrading to an
            untrained model because a real checkpoint failed to load would turn
            a broken deployment into a silently-wrong one.
    """
    device = device or config.device_preference

    feature_extractor = None
    try:
        from training.change_vqa.features import build_change_feature_extractor

        feature_extractor = build_change_feature_extractor(
            config,
            checkpoint_path=checkpoint_path,
            device=device,
            image_size=int(config.get("change_vqa.image_size", 256)),
        )
    except ModelLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SpecialistError(
            f"could not build the change feature extractor: "
            f"{type(exc).__name__}: {exc}",
            specialist="change_vqa",
        ) from exc

    head = None
    head_metadata: dict[str, Any] = {}
    if head_path is not None and Path(head_path).exists():
        from training.change_vqa.model import load_change_vqa_head

        head = load_change_vqa_head(head_path, device=device)
        metadata_path = Path(head_path).parent / "model_metadata.json"
        if metadata_path.exists():
            import json as _json

            try:
                head_metadata = _json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
            except Exception as exc:  # noqa: BLE001
                raise SpecialistError(
                    f"the change-VQA head at {head_path} has a metadata file "
                    f"that cannot be parsed ({metadata_path}): "
                    f"{type(exc).__name__}: {exc}. Without it the head's trained "
                    f"feature spec cannot be checked against this deployment, so "
                    f"the head would be served without a skew check.",
                    specialist="change_vqa",
                ) from exc

    text_encoder = None
    try:
        from training.change_vqa.features import TextFeatureExtractor

        text_encoder = TextFeatureExtractor.build(config, device=device)
    except ModelLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SpecialistError(
            f"could not build the question text encoder: "
            f"{type(exc).__name__}: {exc}",
            specialist="change_vqa",
        ) from exc

    return ChangeVQASpecialist(
        head=head,
        feature_extractor=feature_extractor,
        text_encoder=text_encoder,
        artifact_dir=artifact_dir,
        device=device,
        apply_type_mask=bool(config.get("change_vqa.apply_type_mask", True)),
        head_path=head_path,
        head_metadata=head_metadata,
    )


__all__ = [
    "ChangeVQASpecialist",
    "LOW_CONFIDENCE_THRESHOLD",
    "TEMPORAL_ORDER_NOTE",
    "build_change_vqa_specialist",
]
