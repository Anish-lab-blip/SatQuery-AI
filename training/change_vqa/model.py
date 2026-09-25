"""SatQuery AI — the CDVQA reasoning head (R-02).

THE ARCHITECTURE, AND WHY IT IS THIS ONE
----------------------------------------
Measured facts that constrain the design (all re-measured 2026-09-21):

    * the answer space is CLOSED and has **19** members across 8 question types
    * answers are short (`yes`, `no`, `buildings`, `10_to_20`), never sentences
    * `label1`/`label2` ship for all 2,968 scenes, so class-wise change is
      COMPUTABLE supervision, not a guessed target
    * the budget is <= 3 h on T4x2 (plan section 46)

Those four facts rule out a generative decoder (19-way classification is
strictly easier and exactly matches the metric) and rule out training anything
large. What they leave is a small, two-stage reasoning head:

    STAGE 1  change representation -> class-wise change estimate
             "how much did each of the six classes change, and in which
              direction?"   (13 outputs, directly supervised by label1/label2)

    STAGE 2  class-wise estimate + question -> answer
             "given what changed and what was asked, which of the 19 answers?"

Stage 1 is not decoration and it is not an auxiliary loss bolted on. Its
outputs are CONCATENATED into stage 2's input, so the answer is computed FROM
the change estimate — the same way a person answers "what is the largest
change?" by first working out the per-class changes and then picking the
largest. The gradient flows through the estimate, so stage 1 is trained by the
answer loss as well as by its own.

This is why the head can answer `largest_change` and `smallest_change` at all.
A single pooled vector cannot support "which class changed most" — it has no
per-class structure to compare. The estimator supplies exactly that structure.

TWO-STAGE, NOT TWO-MODEL
------------------------
There is one trainable module and one loss. `class_mag`/`class_delta` are
predictions, not a second pipeline, and they are ALSO emitted as evidence, so
the answer arrives with its own audit trail attached.

CONFIDENCE IS RAW, AND SAYS SO
------------------------------
`confidence` is the softmax probability of the chosen answer, plus the top1-top2
margin. It is **not** calibrated: the R-03 fitting contract is a separate,
unresolved ruling and nothing here is fitted. `ConfidenceBreakdown.method`
therefore reads `"uncalibrated"` all the way to the result schema, and the
interface is built so a future temperature scaling can be applied to `logits`
without changing this module's shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from core.errors import ModelLoadError, SpecialistError
from training.change_vqa.vocab import (
    CHANGE_CLASS_ORDER,
    INDEX_TO_ANSWER,
    N_ANSWERS,
    N_CHANGE_CLASSES,
    N_QUESTION_TYPE_SLOTS,
    N_TEMPORAL_REFERENCE_SLOTS,
    UNKNOWN_QUESTION_TYPE_INDEX,
    legal_answer_mask,
)

#: Bumped when the module's parameter shapes change, so a checkpoint written
#: under an older definition fails to load rather than silently re-interpreting
#: its weights.
ARCHITECTURE_VERSION = "change_vqa_head_v1"

DEFAULT_TRUNK_DIM = 512
DEFAULT_TEXT_DIM = 256
DEFAULT_DROPOUT = 0.10
DEFAULT_QTYPE_EMBED_DIM = 32
DEFAULT_TEMPORAL_EMBED_DIM = 8

#: Stage-1 output width: 6 magnitudes + 6 signed deltas + 1 global fraction.
N_ESTIMATOR_OUTPUTS = 2 * N_CHANGE_CLASSES + 1


@dataclass
class ChangeVQAOutput:
    """Everything one forward pass produced, before any argmax."""

    answer_logits: Any            # (B, N_ANSWERS)
    class_mag: Any                # (B, 6) in [0, 1]   sigmoid
    class_delta: Any              # (B, 6) in [-1, 1]  tanh
    total_changed: Any            # (B,)   in [0, 1]   sigmoid
    legal_mask_applied: bool = False
    #: The pre-sigmoid logits behind `class_mag` / `total_changed`.
    #:
    #: These are NOT part of the published contract — `to_evidence()` emits the
    #: probabilities, `predict_answers()` returns them, and serving consumes
    #: them, so nothing downstream reads these two fields. They exist for
    #: exactly one reason: the LOSS must be computed from logits, because
    #: `BCE(sigmoid(z), t)` is numerically unusable at saturation. See
    #: `binary_cross_entropy_from_logits`.
    #:
    #: Defaulting to `None` keeps every hand-constructed output working; the
    #: loss falls back to the probability path when they are absent.
    mag_logits: Any = None
    total_logits: Any = None

    def to_evidence(self, index: int = 0) -> dict[str, Any]:
        """The class-wise change estimate for one row, as an evidence payload.

        This is what makes the answer auditable: the reader can see WHICH class
        the model believed changed and by how much, rather than only the final
        word. Rounded to 6 dp because a longer float would suggest a precision
        the estimator does not have.

        The `.detach()` calls are not cosmetic. Serving calls this under
        `torch.no_grad()` (`predict_answers`), but a caller may legitimately
        build evidence from a TRAINING forward pass, and `float()` on a
        grad-carrying tensor emits `UserWarning: Converting a tensor with
        requires_grad=True to a scalar` — a warning that trains a reader to
        ignore warnings. Detaching is exact: the float value is bit-identical
        whether or not the tensor carries a grad function, so this changes
        nothing observable except the absence of the warning. The `.tolist()`
        path below detaches implicitly, so only the scalar needed the explicit
        call; it is written explicitly here for all three for symmetry.
        """
        return {
            "class_order": list(CHANGE_CLASS_ORDER),
            "class_change_magnitude": [
                round(float(v), 6) for v in self.class_mag[index].detach().tolist()
            ],
            "class_change_delta": [
                round(float(v), 6) for v in self.class_delta[index].detach().tolist()
            ],
            "total_changed_fraction": round(
                float(self.total_changed[index].detach()), 6
            ),
            "estimator_is_learned": True,
        }


def build_change_vqa_head(
    *,
    change_feature_dim: int,
    text_feature_dim: int,
    trunk_dim: int = DEFAULT_TRUNK_DIM,
    text_dim: int = DEFAULT_TEXT_DIM,
    dropout: float = DEFAULT_DROPOUT,
    qtype_embed_dim: int = DEFAULT_QTYPE_EMBED_DIM,
    temporal_embed_dim: int = DEFAULT_TEMPORAL_EMBED_DIM,
) -> Any:
    """Construct the trainable head. Separate from the class so the class can be
    reconstructed from an embedded config without importing torch at module
    scope (the registry and the planner must stay torch-free)."""
    import torch
    import torch.nn as nn

    class _ChangeVQAHead(nn.Module):  # noqa: D401 - documented at module level
        def __init__(self) -> None:
            super().__init__()
            self.change_feature_dim = int(change_feature_dim)
            self.text_feature_dim = int(text_feature_dim)
            self.trunk_dim = int(trunk_dim)
            self.text_dim = int(text_dim)
            self.dropout_p = float(dropout)
            self.qtype_embed_dim = int(qtype_embed_dim)
            self.temporal_embed_dim = int(temporal_embed_dim)

            # --- stage 1: class-wise change estimation --------------------
            self.estimator = nn.Sequential(
                nn.Linear(self.change_feature_dim, 256),
                nn.GELU(),
                nn.Linear(256, N_ESTIMATOR_OUTPUTS),
            )

            # --- shared change trunk -------------------------------------
            self.change_trunk = nn.Sequential(
                nn.Linear(self.change_feature_dim, self.trunk_dim),
                nn.LayerNorm(self.trunk_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_p),
            )

            # --- question side -------------------------------------------
            self.qtype_embed = nn.Embedding(N_QUESTION_TYPE_SLOTS, self.qtype_embed_dim)
            self.temporal_embed = nn.Embedding(
                N_TEMPORAL_REFERENCE_SLOTS, self.temporal_embed_dim
            )
            question_in = self.text_feature_dim + self.qtype_embed_dim + self.temporal_embed_dim
            self.question_trunk = nn.Sequential(
                nn.Linear(question_in, self.text_dim),
                nn.LayerNorm(self.text_dim),
                nn.GELU(),
            )

            # --- stage 2: question-conditioned answer --------------------
            fused = self.trunk_dim + self.text_dim + N_ESTIMATOR_OUTPUTS
            self.answer_head = nn.Sequential(
                nn.Linear(fused, 512),
                nn.GELU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, N_ANSWERS),
            )

            self._init_weights()

        def _init_weights(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            # A near-uniform start on the answer logits. The class distribution
            # is heavily skewed (yes+no are 58% of Train), so a zero-biased head
            # would begin by predicting the majority class and take several
            # epochs to leave it; the estimator's bias is left at zero because
            # its targets are already balanced in [0, 1].
            final = self.answer_head[-1]
            nn.init.normal_(final.weight, std=0.01)
            with torch.no_grad():
                final.bias.fill_(0.0)

        # -- forward -------------------------------------------------------
        def forward(
            self,
            change_features: Any,
            text_features: Any,
            qtype_index: Any,
            temporal_index: Any,
            legal_mask: Any | None = None,
        ) -> ChangeVQAOutput:
            estimate = self.estimator(change_features)
            mag_raw = estimate[:, :N_CHANGE_CLASSES]
            delta_raw = estimate[:, N_CHANGE_CLASSES:2 * N_CHANGE_CLASSES]
            total_raw = estimate[:, 2 * N_CHANGE_CLASSES]

            class_mag = torch.sigmoid(mag_raw)
            class_delta = torch.tanh(delta_raw)
            total_changed = torch.sigmoid(total_raw)

            change = self.change_trunk(change_features)
            question = self.question_trunk(
                torch.cat(
                    [
                        text_features,
                        self.qtype_embed(qtype_index),
                        self.temporal_embed(temporal_index),
                    ],
                    dim=1,
                )
            )
            fused = torch.cat([change, question, estimate], dim=1)
            logits = self.answer_head(fused)

            applied = False
            if legal_mask is not None:
                logits, applied = _apply_legal_mask(logits, legal_mask)
            return ChangeVQAOutput(
                answer_logits=logits,
                class_mag=class_mag,
                class_delta=class_delta,
                total_changed=total_changed,
                legal_mask_applied=applied,
                # Handed to the loss so it can compute the two BCE terms
                # exactly. The probabilities above are unchanged and remain the
                # published output.
                mag_logits=mag_raw,
                total_logits=total_raw,
            )

        # -- identity ------------------------------------------------------
        def config_dict(self) -> dict[str, Any]:
            return {
                "architecture": ARCHITECTURE_VERSION,
                "change_feature_dim": self.change_feature_dim,
                "text_feature_dim": self.text_feature_dim,
                "trunk_dim": self.trunk_dim,
                "text_dim": self.text_dim,
                "dropout": self.dropout_p,
                "qtype_embed_dim": self.qtype_embed_dim,
                "temporal_embed_dim": self.temporal_embed_dim,
                "n_answers": N_ANSWERS,
                "n_change_classes": N_CHANGE_CLASSES,
                "n_estimator_outputs": N_ESTIMATOR_OUTPUTS,
                "n_question_type_slots": N_QUESTION_TYPE_SLOTS,
                "n_temporal_reference_slots": N_TEMPORAL_REFERENCE_SLOTS,
                "answer_vocabulary": list(INDEX_TO_ANSWER),
                "class_order": list(CHANGE_CLASS_ORDER),
            }

        def num_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters())

        def trainable_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    return _ChangeVQAHead()


def _apply_legal_mask(logits: Any, legal_mask: Any) -> tuple[Any, bool]:
    """Mask illegal answers to -inf, refusing to mask a row completely.

    A row whose every answer is illegal would argmax to index 0 — a confident
    wrong answer with no trace of why. That is caught here and the row is left
    unmasked, because an unmasked answer is a legitimate answer and a masked
    empty row is a silent corruption.
    """
    import torch

    mask = legal_mask.to(dtype=torch.bool)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0).expand(logits.shape[0], -1)
    if mask.shape != logits.shape:
        raise SpecialistError(
            f"legal mask shape {tuple(mask.shape)} does not match logits "
            f"{tuple(logits.shape)}",
            specialist="change_vqa",
        )
    usable = mask.any(dim=1)
    if not bool(usable.any()):
        return logits, False
    masked = logits.masked_fill(~mask & usable.unsqueeze(1), float("-inf"))
    # Rows with no legal answer keep their original logits.
    masked = torch.where(usable.unsqueeze(1), masked, logits)
    return masked, True


# ---------------------------------------------------------------------------
# Checkpoint I/O — strict, so an architecture drift fails loudly
# ---------------------------------------------------------------------------


def save_change_vqa_head(
    path: str | Path, model: Any, metadata: dict[str, Any] | None = None
) -> Path:
    import torch

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": model.config_dict()}, str(p)
    )
    if metadata is not None:
        (p.parent / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return p


def load_change_vqa_head(path: str | Path, device: str = "cpu") -> Any:
    """Load a trained head. Strict, mirroring `load_change_model`."""
    import torch

    p = Path(path)
    if not p.exists():
        raise ModelLoadError(
            f"change-VQA head not found: {p}", specialist="change_vqa"
        )
    try:
        payload = torch.load(str(p), map_location=device, weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(
            f"could not read the change-VQA head from {p}: {exc}",
            specialist="change_vqa",
        ) from exc

    config = payload.get("config")
    if not config:
        raise ModelLoadError(
            "the change-VQA checkpoint carries no embedded config; it cannot be "
            "reconstructed without guessing the architecture",
            specialist="change_vqa",
        )
    if config.get("architecture") != ARCHITECTURE_VERSION:
        raise ModelLoadError(
            f"the change-VQA checkpoint was written under architecture "
            f"{config.get('architecture')!r} but this build is "
            f"{ARCHITECTURE_VERSION!r}",
            specialist="change_vqa",
        )

    model = build_change_vqa_head(
        change_feature_dim=int(config["change_feature_dim"]),
        text_feature_dim=int(config["text_feature_dim"]),
        trunk_dim=int(config["trunk_dim"]),
        text_dim=int(config["text_dim"]),
        dropout=float(config["dropout"]),
        qtype_embed_dim=int(config["qtype_embed_dim"]),
        temporal_embed_dim=int(config["temporal_embed_dim"]),
    )
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(
            f"the change-VQA state_dict does not match the reconstructed "
            f"architecture: {exc}",
            specialist="change_vqa",
        ) from exc
    model.to(device)
    model.eval()
    model._satquery_trained = True  # type: ignore[attr-defined]
    return model


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


@dataclass
class ChangeVQALossBreakdown:
    total: Any
    answer: Any
    magnitude: Any
    delta: Any
    total_changed: Any
    weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        out = {
            "total": float(self.total.detach()),
            "answer_ce": float(self.answer.detach()),
            "class_magnitude_bce": float(self.magnitude.detach()),
            "class_delta_mse": float(self.delta.detach()),
            "total_changed_bce": float(self.total_changed.detach()),
        }
        out.update({f"weight_{k}": v for k, v in self.weights.items()})
        return out


#: Loss weights. The answer term dominates because the answer is the deliverable;
#: the estimator terms are strong enough to shape a usable per-class
#: representation (weight ~1.0 against a term whose natural scale is ~0.2) and
#: small enough not to pull the trunk away from the answer objective.
DEFAULT_LOSS_WEIGHTS: dict[str, float] = {
    "answer": 1.0,
    "magnitude": 1.0,
    "delta": 1.0,
    "total_changed": 0.5,
}


def binary_cross_entropy_from_logits(logits: Any, target: Any) -> Any:
    """BCE computed from raw logits — the numerically exact form. USE THIS.

    `BCEWithLogits(z, t)` and `BCE(sigmoid(z), t)` are the same function
    mathematically. They are not the same function numerically, and that
    difference is what killed the first real Kaggle run.

    MEASURED ON THIS REPOSITORY
    ---------------------------
    Drive the estimator to a saturated logit and compare the two forms:

        z = -30, t = 1   BCE(sigmoid(z)) backward ->  -0.0936   (should be -1.0)
                         BCEWithLogits(z) backward ->  -1.0
        z = +30, t = 0   BCE(sigmoid(z)) forward  ->   65.0     (true value 30.0)

    Two separate defects, both caused by taking the log of a probability:

    1. THE BACKWARD DIVIDES BY p(1 - p). `d/dp = (p - t) / (p * (1 - p))`. At
       `p = 1.0` exactly that is `-1 / 0`, which PyTorch clamps to a FINITE
       `-9.99999996e11`. Because the result is finite, `GradScaler` never flags
       it, never skips the step, and it reaches the optimizer intact. AdamW then
       squares it into the second-moment estimate: `v ~ (1e12)^2 = 1e24`, which
       suppresses every subsequent update to that parameter. That is the
       mechanism behind the observed epoch-1-to-epoch-2 collapse — val_acc
       0.4388 -> 0.2656 with mean confidence 0.9996.
    2. THE FORWARD IS WRONG, not merely unstable. PyTorch clamps each element at
       100, so a saturated element silently reports 65.0 where the true value is
       30.0. The loss stops measuring the quantity it claims to measure.

    Saturation is NOT an AMP problem. `sigmoid(17.0)` is exactly `1.0` in plain
    fp32. fp16 reaches it sooner (|z| >= 18 under autocast), but the fp32 path
    saturates as well, so `--no-amp` would only have moved the failure.

    `BCEWithLogits` uses the log-sum-exp identity: the forward is exact for any
    finite logit, and the backward is `sigmoid(z) - t`, bounded in [-1, 1]. There
    is no division by a probability anywhere in it, so it cannot saturate.

    The logits already exist in `forward()` before the sigmoid and are carried on
    `ChangeVQAOutput.mag_logits` / `.total_logits`. Using them changes NOTHING
    about what the model emits: `class_mag` is still `sigmoid(mag_raw)`, and
    `to_evidence()`, `predict_answers()` and serving are untouched. Only the
    internal loss arithmetic changes, from an explosive-and-wrong form to the
    exact one.
    """
    import torch
    import torch.nn.functional as F

    with torch.amp.autocast(device_type=logits.device.type, enabled=False):
        return F.binary_cross_entropy_with_logits(
            logits.float(), target.to(dtype=torch.float32)
        )


def binary_cross_entropy_probabilities(prediction: Any, target: Any) -> Any:
    """BCE on sigmoid probabilities. FALLBACK ONLY — prefer the logits form.

    Kept for a caller that holds only probabilities, and for a `ChangeVQAOutput`
    built by hand without logits. The training path does not use it:
    `change_vqa_loss` prefers `binary_cross_entropy_from_logits`.

    Two things this function has to do, and why:

    1. It must be *invoked* with autocast disabled. `aten::binary_cross_entropy`
       is registered as an outright ERROR under CUDA autocast, and that
       registration is unconditional — it fires whatever the operand dtypes are,
       so casting to fp32 is necessary but not sufficient. This was the very
       first Kaggle failure: `RuntimeError: ... are unsafe to autocast`.
    2. It must keep its operand away from 0 and 1. `d/dp = (p - t)/(p(1 - p))` is
       unbounded there; the clamp below bounds it at about `1/eps ~ 8.4e6`
       instead of the `1e12` an unclamped saturated operand produces. The clamp
       is a no-op for any probability inside `(eps, 1-eps)`, so it changes
       nothing in the interior — it only refuses to divide by zero.
    """
    import torch
    import torch.nn.functional as F

    eps = torch.finfo(torch.float32).eps
    with torch.amp.autocast(device_type=prediction.device.type, enabled=False):
        return F.binary_cross_entropy(
            prediction.float().clamp(min=eps, max=1.0 - eps),
            target.to(dtype=torch.float32),
        )


def change_vqa_loss(
    output: ChangeVQAOutput,
    *,
    answer_index: Any,
    class_mag: Any,
    class_delta: Any,
    total_changed: Any,
    weights: dict[str, float] | None = None,
) -> ChangeVQALossBreakdown:
    """Composite answer + estimator loss.

    The estimator targets come from `label1`/`label2`, so they are exact — there
    is no pseudo-labelling anywhere in this loss.

    `cross_entropy` and `mse_loss` are both autocast-safe, so they are called
    directly. The two BCE terms are not — they go through
    `binary_cross_entropy_from_logits`, which is autocast-safe *and* exact. The
    probability fallback below exists only for a hand-built `output` carrying no
    logits; `forward()` always populates them.
    """
    import torch
    import torch.nn.functional as F

    w = dict(DEFAULT_LOSS_WEIGHTS)
    if weights:
        w.update(weights)

    answer = F.cross_entropy(output.answer_logits, answer_index)

    if output.mag_logits is not None:
        magnitude = binary_cross_entropy_from_logits(output.mag_logits, class_mag)
    else:
        magnitude = binary_cross_entropy_probabilities(output.class_mag, class_mag)

    delta = F.mse_loss(output.class_delta, class_delta)

    if output.total_logits is not None:
        total = binary_cross_entropy_from_logits(output.total_logits, total_changed)
    else:
        total = binary_cross_entropy_probabilities(
            output.total_changed, total_changed
        )

    combined = (
        w["answer"] * answer
        + w["magnitude"] * magnitude
        + w["delta"] * delta
        + w["total_changed"] * total
    )
    return ChangeVQALossBreakdown(
        total=combined,
        answer=answer,
        magnitude=magnitude,
        delta=delta,
        total_changed=total,
        weights=w,
    )


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def predict_answers(
    model: Any,
    *,
    change_features: Any,
    text_features: Any,
    qtype_indices: Any,
    temporal_indices: Any,
    qtypes: Sequence[str] | None = None,
    apply_type_mask: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Deterministic inference. `model.eval()` + `no_grad` + argmax.

    Args:
        qtypes: native type strings, used ONLY to build the legal-answer mask.
        apply_type_mask: when True and a type is known, illegal answers are
            masked. Default False so the caller can report masked and unmasked
            numbers separately instead of quietly reporting the better one.

    Returns:
        dict with `answer_index`, `answer`, `confidence` (raw softmax),
        `margin` (top1 - top2), `probabilities` (the full `(B, 19)` softmax, so
        top-k metrics are computed rather than skipped), `class_mag`,
        `class_delta`, `total_changed`, `legal_mask_applied`.
    """
    import numpy as np
    import torch

    model.eval()
    with torch.no_grad():
        change = torch.as_tensor(change_features, dtype=torch.float32, device=device)
        text = torch.as_tensor(text_features, dtype=torch.float32, device=device)
        qidx = torch.as_tensor(qtype_indices, dtype=torch.long, device=device)
        tidx = torch.as_tensor(temporal_indices, dtype=torch.long, device=device)

        legal = None
        if apply_type_mask and qtypes is not None:
            rows = []
            for qtype in qtypes:
                mask = legal_answer_mask(qtype)
                rows.append(
                    mask
                    if mask is not None
                    else tuple(True for _ in range(N_ANSWERS))
                )
            legal = torch.as_tensor(np.asarray(rows), dtype=torch.bool, device=device)

        output = model(change, text, qidx, tidx, legal_mask=legal)
        logits = output.answer_logits
        probabilities = torch.softmax(logits, dim=1)
        top2 = probabilities.topk(k=2, dim=1)
        confidence = top2.values[:, 0]
        margin = top2.values[:, 0] - top2.values[:, 1]
        answer_index = probabilities.argmax(dim=1)

    return {
        "answer_index": answer_index.detach().cpu().numpy(),
        "answer": [INDEX_TO_ANSWER[int(i)] for i in answer_index.detach().cpu().numpy()],
        "confidence": confidence.detach().cpu().numpy(),
        "margin": margin.detach().cpu().numpy(),
        "probabilities": probabilities.detach().cpu().numpy(),
        "class_mag": output.class_mag.detach().cpu().numpy(),
        "class_delta": output.class_delta.detach().cpu().numpy(),
        "total_changed": output.total_changed.detach().cpu().numpy(),
        "legal_mask_applied": output.legal_mask_applied,
    }


__all__ = [
    "ARCHITECTURE_VERSION",
    "DEFAULT_TRUNK_DIM",
    "DEFAULT_TEXT_DIM",
    "DEFAULT_DROPOUT",
    "N_ESTIMATOR_OUTPUTS",
    "DEFAULT_LOSS_WEIGHTS",
    "ChangeVQAOutput",
    "ChangeVQALossBreakdown",
    "build_change_vqa_head",
    "save_change_vqa_head",
    "load_change_vqa_head",
    "binary_cross_entropy_from_logits",
    "binary_cross_entropy_probabilities",
    "change_vqa_loss",
    "predict_answers",
    "UNKNOWN_QUESTION_TYPE_INDEX",
]
