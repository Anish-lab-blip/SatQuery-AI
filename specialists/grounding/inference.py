"""SatQuery AI — zero-shot grounding baseline.

This is the BASELINE, not the product. Phase 8 trains a grounding head; this
module exists so the resolution experiment (scripts/exp_grounding_resolution.py)
has something to measure BEFORE the head is written, and so there is an ablation
floor to compare the head against later.

Method: patch-text cosine similarity, no learned parameters.

    image -> RemoteCLIP patch tokens (N, 512)
    phrase -> RemoteCLIP text embedding (512,)
    similarity = normalized_patches @ normalized_text   -> (N,)
    reshape to (grid_h, grid_w)
    -> box

Two box strategies, both reported because they measure different things:

  `argmax`     the single highest-scoring patch, as its token box. This is the
               pure localization floor: the best a patch-similarity model can do
               at this grid resolution with no smoothing.

  `threshold`  the bounding box of every patch scoring within `delta` of the
               peak. Usually larger and sometimes better on objects that span
               several patches, sometimes worse when the peak is a spurious
               spike. Reporting both makes the head's later job measurable.

Neither strategy is a claim about localization quality. The grid cell is the
floor, and the resolution experiment is what decides whether a finer grid buys
anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from specialists.grounding.remoteclip import EncodedImage, RemoteCLIPEncoder

#: Patch boxes within this cosine distance of the peak are included in the
#: `threshold` box. 0.02 is a starting value, not a tuned one -- it is an
#: ablation variable, and tuning it belongs on validation data.
DEFAULT_DELTA = 0.02


@dataclass(frozen=True)
class GroundingCandidate:
    """One box proposal with the signal that produced it."""

    box: list[float]            # normalized [x1, y1, x2, y2]
    score: float                # cosine similarity of the peak patch
    method: str                 # "argmax" | "threshold"
    n_patches: int              # patches contributing to this box

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": [round(v, 6) for v in self.box],
            "score": round(self.score, 4),
            "method": self.method,
            "n_patches": self.n_patches,
        }


def similarity_map(encoded: EncodedImage, text_embedding: np.ndarray) -> np.ndarray:
    """Cosine similarity between every patch token and one text embedding.

    Both sides are L2-normalized first. Without that the dot product is
    dominated by whichever vector happens to have the larger norm, which is a
    property of the features rather than of the match.
    """
    patches = encoded.patch_tokens.astype(np.float64)
    text = np.asarray(text_embedding, dtype=np.float64).reshape(-1)

    if patches.shape[1] != text.shape[0]:
        raise ValueError(
            f"dimension mismatch: patch tokens are {patches.shape[1]}-d, "
            f"text embedding is {text.shape[0]}-d"
        )

    patch_norms = np.linalg.norm(patches, axis=1, keepdims=True)
    patch_norms[patch_norms == 0] = 1.0
    text_norm = np.linalg.norm(text)
    if text_norm == 0:
        raise ValueError("text embedding has zero norm")

    normalized_patches = patches / patch_norms
    normalized_text = text / text_norm
    return normalized_patches @ normalized_text


def argmax_candidate(
    encoded: EncodedImage, similarity: np.ndarray
) -> GroundingCandidate:
    """The single best-scoring patch, as its token box.

    This is the localization floor: at 224 the box is 1/7 of the image in each
    dimension, at 448 it is 1/14.
    """
    boxes = encoded.token_boxes()
    peak = int(np.argmax(similarity))
    return GroundingCandidate(
        box=[float(v) for v in boxes[peak]],
        score=float(similarity[peak]),
        method="argmax",
        n_patches=1,
    )


def threshold_candidate(
    encoded: EncodedImage,
    similarity: np.ndarray,
    delta: float = DEFAULT_DELTA,
) -> GroundingCandidate | None:
    """Bounding box of every patch scoring within `delta` of the peak.

    Returns None when only one patch qualifies -- in that case the box would be
    identical to `argmax` and reporting it twice adds nothing.

    The box spans from the min to the max selected patch, so it is aligned to
    the token grid. That is deliberate: pretending to a finer boundary than the
    grid supports would be inventing precision the model does not have.
    """
    if delta < 0:
        raise ValueError(f"delta must be non-negative, got {delta}")

    boxes = encoded.token_boxes()
    peak = float(similarity.max())
    selected = np.flatnonzero(similarity >= peak - delta)

    if selected.size <= 1:
        return None

    grid_h, grid_w = encoded.grid
    rows, cols = np.divmod(selected, grid_w)
    _ = grid_h  # kept for symmetry; bounds are implied by the index range

    x1 = float(boxes[selected, 0].min())
    y1 = float(boxes[selected, 1].min())
    x2 = float(boxes[selected, 2].max())
    y2 = float(boxes[selected, 3].max())

    return GroundingCandidate(
        box=[x1, y1, x2, y2],
        score=peak,
        method="threshold",
        n_patches=int(selected.size),
    )


def decode_candidates_from_features(
    encoded: EncodedImage,
    text_embedding: np.ndarray,
    delta: float = DEFAULT_DELTA,
    top_k: int = 5,
) -> list[GroundingCandidate]:
    """Decode boxes from ALREADY-COMPUTED features. Returns candidates, best first.

    THE single implementation of the zero-shot decode. `ground_phrase` encodes
    an image then calls this; the evaluation script reads a feature cache then
    calls this. Because there is one decode, the two paths cannot disagree.

    WHY THIS FUNCTION EXISTS
    ------------------------
    The Phase 8 evaluation script originally built its own single-box baseline
    with `argmax_candidate`, while Phase 7 measured the baseline through
    `ground_phrase`. Same 16,159 records, same metric, same cached features --
    but a different DECODE:

        Phase 7 via ground_phrase  : mean best IoU 0.0972
        eval via argmax_candidate  : mean best IoU 0.0092

    A 10x gap. The eval printed `head beats zero-shot: True (+0.1123)` when the
    matched comparison is +0.0243. Both clear the 0.02 bar, but only the second
    is a claim about the head rather than about the decode.

    The duplication was the defect, so the duplication is what is removed:
    there is now exactly one place that turns similarity into boxes.

    Args:
        encoded: patch tokens plus grid, from the frozen encoder.
        text_embedding: (D,) text embedding in the same projected space.
        delta: patches within this cosine distance of the peak form the
            threshold box. A config value, not a tuned one.
        top_k: cap on patch-local maxima reported. Small by design: the
            zero-shot field has one strong peak and a tail of noise, and
            flooding the output would make precision meaningless.
    """
    similarity = similarity_map(encoded, text_embedding)

    candidates: list[GroundingCandidate] = []

    thresholded = threshold_candidate(encoded, similarity, delta)
    if thresholded is not None:
        candidates.append(thresholded)

    boxes = encoded.token_boxes()
    grid_h, grid_w = encoded.grid

    # Local maxima only: a patch must beat its 4-neighbours. Without this,
    # `top_k` returns k adjacent patches from a single blob, which looks like
    # k detections and is really one.
    order = np.argsort(similarity)[::-1]
    taken: list[int] = []
    for index in order:
        if len(taken) >= top_k:
            break
        row, col = divmod(int(index), grid_w)
        is_peak = True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < grid_h and 0 <= nc < grid_w:
                if similarity[nr * grid_w + nc] > similarity[index]:
                    is_peak = False
                    break
        if not is_peak:
            continue
        # Skip patches already covered by an accepted local maximum.
        if any(
            abs(row - divmod(t, grid_w)[0]) <= 1 and abs(col - divmod(t, grid_w)[1]) <= 1
            for t in taken
        ):
            continue
        taken.append(int(index))
        candidates.append(
            GroundingCandidate(
                box=[float(v) for v in boxes[int(index)]],
                score=float(similarity[int(index)]),
                method="argmax",
                n_patches=1,
            )
        )

    if not candidates:
        candidates.append(argmax_candidate(encoded, similarity))

    return candidates


def ground_phrase(
    encoder: RemoteCLIPEncoder,
    image: Any,
    phrase: str,
    delta: float = DEFAULT_DELTA,
    top_k: int = 5,
) -> list[GroundingCandidate]:
    """Ground one phrase in one image. Returns candidates, best first.

    Encodes, then delegates to `decode_candidates_from_features`. The decode
    itself is NOT duplicated here -- see that function for why.
    """
    encoded = encoder.encode_image(image)
    text = encoder.encode_text([phrase])[0]
    return decode_candidates_from_features(encoded, text, delta=delta, top_k=top_k)


def candidate_boxes(candidates: list[GroundingCandidate]) -> list[list[float]]:
    """Extract just the boxes, for the metrics module."""
    return [c.box for c in candidates]


__all__ = [
    "DEFAULT_DELTA",
    "GroundingCandidate",
    "similarity_map",
    "argmax_candidate",
    "threshold_candidate",
    "decode_candidates_from_features",
    "ground_phrase",
    "candidate_boxes",
]