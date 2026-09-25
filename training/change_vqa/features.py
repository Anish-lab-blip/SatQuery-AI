"""SatQuery AI — frozen change features for the CDVQA reasoning head (R-02).

WHY FROZEN FEATURES
-------------------
Plan section 46 budgets CDVQA at **<= 3 h on T4x2**. A head trained end to end
through the STANet encoder would spend that budget re-learning a change
representation the project already has a trained checkpoint for
(`artifacts/change/levir_change_v001/head.pt`, test pooled IoU 0.8122). The plan
also forbids retraining STANet unless evidence demands it (section 21 and the
R-02 brief both say so).

So the encoder, the difference fusion, the spatial attention, the decoder and
the change head are all run **frozen and cached**, and only a small head is
trained. Measured effect: the trainable parameter count falls from ~15.8 M to
~1.5 M and the training loop becomes a few minutes on a T4 rather than hours.

THE FEATURE IS THREE THINGS, NOT ONE
------------------------------------
`CHANGE_FEATURE_DIM = 1045`, and each part answers a different question the
CDVQA ontology asks:

    level_pooled   1024 = 4 levels x 128 ch x {mean, max}
                   "what does the change look like locally?"
    change_stats      5 = mean, std, frac>0.3, frac>0.5, frac>0.7 of the
                   trained change probability map
                   "how much of the scene changed?"  -> change_or_not,
                   change_ratio, change_ratio_types
    change_grid      16 = adaptive 4x4 pool of the same map
                   "WHERE did it change?"  -> largest_change, smallest_change

The spatial parts are not decoration. Without `change_grid` the head would see
a scene-global average and could not distinguish "buildings changed in one
corner" from "buildings changed everywhere", which is precisely the difference
between `largest_change` and `smallest_change`.

EQUIVALENCE IS PROVEN, NOT ASSUMED
----------------------------------
This module re-runs the detector's own submodules to capture the intermediate
fused levels that `STANetStyleChangeDetector.forward` does not return. That is
a second implementation of one forward pass, so it is a drift hazard — and it
is closed by a test that asserts the probability map produced here is
**bit-identical** to `STANetStyleChangeDetector.forward(...).probabilities`
(`tests/unit/test_change_vqa_features.py`). The frozen module is not modified
to expose the intermediates, because not touching it is the stronger guarantee.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from core.errors import ModelLoadError, SpecialistError

#: Bumped whenever the pooling, the resolution or the statistic set changes.
#: Recorded next to every cached tensor so a stale cache is detectable.
FEATURE_SPEC = "change_feat_v1"

#: Default working resolution. 256 is the change model's OWN training
#: resolution (`configs/base.yaml: change.tile_size: 256`; LEVIR-CD patches are
#: 256x256). Feeding the native 512 would push the frozen encoder outside the
#: distribution it was trained on, so 256 is the faithful default and 512 is
#: offered as an explicit, recorded alternative.
DEFAULT_IMAGE_SIZE = 256

#: Fraction-of-pixels thresholds summarised from the change map.
CHANGE_STAT_THRESHOLDS: tuple[float, ...] = (0.3, 0.5, 0.7)

#: Spatial grid the change map is pooled to.
CHANGE_GRID: int = 4

#: Dimensions. Derived, never typed: a literal here would drift silently from
#: the pooling code below and produce a shape error three files away.
N_LEVELS = 4
LEVEL_CHANNELS = 128
LEVEL_POOLED_DIM = N_LEVELS * LEVEL_CHANNELS * 2
CHANGE_STATS_DIM = 2 + len(CHANGE_STAT_THRESHOLDS)
CHANGE_GRID_DIM = CHANGE_GRID * CHANGE_GRID
CHANGE_FEATURE_DIM = LEVEL_POOLED_DIM + CHANGE_STATS_DIM + CHANGE_GRID_DIM

#: Frozen text-encoder identity. MiniLM is already a project dependency (the
#: intent router uses it) and is present in the local HF cache, so reusing it
#: costs no new download and no new licence surface.
TEXT_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_FEATURE_DIM = 384


class FeatureExtractionError(SpecialistError):
    """The frozen feature extractor could not produce a feature vector."""

    code = "change_vqa_features_error"


def feature_spec_hash(image_size: int, *, extractor: str) -> str:
    """Stable hash of the feature definition, for cache and artifact identity."""
    blob = json.dumps(
        {
            "spec": FEATURE_SPEC,
            "image_size": int(image_size),
            "extractor": extractor,
            "levels": N_LEVELS,
            "level_channels": LEVEL_CHANNELS,
            "thresholds": list(CHANGE_STAT_THRESHOLDS),
            "grid": CHANGE_GRID,
            "text_encoder": TEXT_ENCODER_NAME,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def text_spec_hash(*, encoder: str, revision: str | None = None) -> str:
    """Stable hash of the QUESTION-feature definition.

    A separate hash from `feature_spec_hash` on purpose: the change cache and the
    text cache are written by different steps and can legitimately be rebuilt
    independently. One combined hash would force a full re-extraction of both
    whenever either changed, and — worse — would let a stale text cache pass a
    check it should fail.

    `revision` is the pinned model revision when the caller knows it
    (`configs/base.yaml: router.revision`). It is recorded rather than required,
    because the router's own encoder is resolved by name in some paths and a
    None here means "not pinned at this call site", which is a weaker but
    truthful claim than inventing a revision string.
    """
    blob = json.dumps(
        {
            "spec": "change_vqa_text_v1",
            "encoder": encoder,
            "revision": revision or "unpinned",
            "dim": TEXT_FEATURE_DIM,
            "normalised": True,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Change features
# ---------------------------------------------------------------------------


@dataclass
class ChangeFeatureExtractor:
    """Runs a (frozen) STANet detector and returns a pooled change feature.

    Args:
        model: a constructed `STANetStyleChangeDetector`, typically loaded from
            the trained LEVIR checkpoint.
        device: torch device string.
        image_size: working resolution; the input is resized to this.
        trained: whether `model` carries trained weights. Recorded so a
            degraded feature set can never be mistaken for a trained one.
    """

    model: Any
    device: str = "cpu"
    image_size: int = DEFAULT_IMAGE_SIZE
    trained: bool = False

    def __post_init__(self) -> None:
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    # -- identity ----------------------------------------------------------

    @property
    def spec_hash(self) -> str:
        return feature_spec_hash(
            self.image_size,
            extractor="stanet:" + ("trained" if self.trained else "untrained"),
        )

    def config_dict(self) -> dict[str, Any]:
        return {
            "feature_spec": FEATURE_SPEC,
            "feature_spec_hash": self.spec_hash,
            "feature_dim": CHANGE_FEATURE_DIM,
            "image_size": self.image_size,
            "trained": self.trained,
            "detector_config": self.model.config_dict(),
            "level_pooled_dim": LEVEL_POOLED_DIM,
            "change_stats_dim": CHANGE_STATS_DIM,
            "change_grid_dim": CHANGE_GRID_DIM,
        }

    # -- internals ---------------------------------------------------------

    def _prepare(self, array: Any) -> Any:
        """(H, W, 3) uint8 -> (1, 3, S, S) float32 in [0, 1], resized to S."""
        import numpy as np
        import torch

        arr = np.asarray(array, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[:, :, :3]
        arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)
        if tensor.shape[-1] != self.image_size or tensor.shape[-2] != self.image_size:
            tensor = torch.nn.functional.interpolate(
                tensor,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return tensor.to(self.device)

    # -- the forward pass --------------------------------------------------

    def extract(
        self, image_t1: Any, image_t2: Any
    ) -> tuple[Any, dict[str, Any]]:
        """Return `(feature_vector[CHANGE_FEATURE_DIM], facts)`.

        `facts` carries the interpretable intermediates (per-class-agnostic
        change statistics and the pooled grid) so a caller can emit them as
        evidence without recomputing anything.
        """
        import numpy as np
        import torch

        t1 = self._prepare(image_t1)
        t2 = self._prepare(image_t2)
        if t1.shape != t2.shape:
            raise FeatureExtractionError(
                f"T1 and T2 prepared to different shapes: "
                f"{tuple(t1.shape)} vs {tuple(t2.shape)}"
            )

        model = self.model
        levels = []
        with torch.no_grad():
            f1 = model.encoder(t1)
            f2 = model.encoder(t2)
            for level, (a, b) in enumerate(zip(f1, f2)):
                fused = model.fuse[level](a, b)
                fused, _ran, _needed = model.attention[level](
                    fused, model.attention_budget_bytes
                )
                levels.append(fused)

            x = levels[3]
            x = model.dec3(x, levels[2])
            x = model.dec2(x, levels[1])
            x = model.dec1(x, levels[0])
            x = model.final_up(x)
            if x.shape[-2:] != t1.shape[-2:]:
                x = torch.nn.functional.interpolate(
                    x, size=t1.shape[-2:], mode="bilinear", align_corners=False
                )
            probability = torch.sigmoid(model.head(x))  # (1, 1, S, S)

            pooled: list[torch.Tensor] = []
            for level_features in levels:
                pooled.append(level_features.mean(dim=(2, 3)))
                pooled.append(level_features.amax(dim=(2, 3)))
            level_pooled = torch.cat(pooled, dim=1).reshape(-1)  # (1024,)

            flat = probability.reshape(-1)
            stats = torch.cat(
                [
                    flat.mean().reshape(1),
                    flat.std(unbiased=False).reshape(1),
                    *[
                        (flat > threshold).float().mean().reshape(1)
                        for threshold in CHANGE_STAT_THRESHOLDS
                    ],
                ]
            )  # (5,)

            grid = torch.nn.functional.adaptive_avg_pool2d(
                probability, (CHANGE_GRID, CHANGE_GRID)
            ).reshape(-1)  # (16,)

            vector = torch.cat([level_pooled, stats, grid], dim=0)

        vector_np = vector.detach().cpu().numpy().astype(np.float32)
        probability_np = (
            probability.detach().cpu().numpy().reshape(probability.shape[-2:])
        )
        if vector_np.shape[0] != CHANGE_FEATURE_DIM:
            raise FeatureExtractionError(
                f"feature vector is {vector_np.shape[0]}-d but "
                f"CHANGE_FEATURE_DIM is {CHANGE_FEATURE_DIM}; the pooling and "
                f"the declared dimension have drifted apart"
            )

        facts = {
            "change_probability_mean": float(probability_np.mean()),
            "change_probability_std": float(probability_np.std()),
            "change_fraction_gt_0_5": float((probability_np > 0.5).mean()),
            "change_probability_max": float(probability_np.max()),
            "change_grid": [
                float(v) for v in grid.detach().cpu().numpy().reshape(-1)
            ],
            "image_size": self.image_size,
            "trained_detector": self.trained,
        }
        return vector_np, facts

    def extract_probability_map(self, image_t1: Any, image_t2: Any) -> Any:
        """The change probability map alone, for evidence rendering.

        Reuses `extract` rather than duplicating the forward pass, so the map
        and the feature can never disagree.
        """
        _vector, _facts = self.extract(image_t1, image_t2)
        import numpy as np
        import torch

        t1 = self._prepare(image_t1)
        t2 = self._prepare(image_t2)
        model = self.model
        with torch.no_grad():
            levels = []
            f1 = model.encoder(t1)
            f2 = model.encoder(t2)
            for level, (a, b) in enumerate(zip(f1, f2)):
                fused = model.fuse[level](a, b)
                fused, _ran, _needed = model.attention[level](
                    fused, model.attention_budget_bytes
                )
                levels.append(fused)
            x = levels[3]
            x = model.dec3(x, levels[2])
            x = model.dec2(x, levels[1])
            x = model.dec1(x, levels[0])
            x = model.final_up(x)
            if x.shape[-2:] != t1.shape[-2:]:
                x = torch.nn.functional.interpolate(
                    x, size=t1.shape[-2:], mode="bilinear", align_corners=False
                )
            probability = torch.sigmoid(model.head(x))
        return np.asarray(
            probability.detach().cpu().numpy().reshape(probability.shape[-2:]),
            dtype=np.float64,
        )


def build_change_feature_extractor(
    config: Any,
    *,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> ChangeFeatureExtractor:
    """Construct the extractor from the central config plus an optional checkpoint.

    Mirrors `specialists/change/specialist.py:build_change_specialist`: a
    MISSING checkpoint degrades (untrained extractor, recorded as such), while a
    checkpoint that exists but cannot be read raises `ModelLoadError`. Silently
    running an untrained encoder because a real artifact failed to load would be
    the worst outcome, and it is the specific confusion the change specialist's
    own comment forbids.
    """
    from specialists.change.stanet import (
        STANetStyleChangeDetector,
        load_change_model,
    )

    device = device or config.device_preference
    trained = False

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        model = load_change_model(checkpoint_path, device=device)
        trained = True
    else:
        model = STANetStyleChangeDetector(
            width=int(config.get("change.decoder_width", 128)),
            pretrained=False,
            sa_mode=str(config.get("change.sa_mode", "PAM")),
            attention_budget_bytes=int(
                config.get("change.attention_budget_bytes", 256 * 1024 * 1024)
            ),
        )
        model.to(device)
        model.eval()

    return ChangeFeatureExtractor(
        model=model, device=device, image_size=image_size, trained=trained
    )


# ---------------------------------------------------------------------------
# Text features
# ---------------------------------------------------------------------------


@dataclass
class TextFeatureExtractor:
    """Frozen MiniLM sentence encoder for the question text.

    The router already depends on this model and the revision is pinned in
    `configs/base.yaml` (`router.revision`), so the text side of the reasoning
    head adds no new model to the deployment.
    """

    model: Any
    device: str = "cpu"
    batch_size: int = 256

    @classmethod
    def build(cls, config: Any, *, device: str | None = None) -> "TextFeatureExtractor":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ModelLoadError(
                f"sentence-transformers is required for change-VQA question "
                f"encoding: {exc}",
                specialist="change_vqa",
            ) from exc
        name = str(config.get("router.model", TEXT_ENCODER_NAME))
        device = device or config.device_preference
        model = SentenceTransformer(name, device=device)
        return cls(model=model, device=device)

    def encode(self, texts: Sequence[str]) -> Any:
        import numpy as np

        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        array = np.asarray(vectors, dtype=np.float32)
        if array.shape[-1] != TEXT_FEATURE_DIM:
            raise FeatureExtractionError(
                f"text encoder produced {array.shape[-1]}-d vectors but "
                f"TEXT_FEATURE_DIM is {TEXT_FEATURE_DIM}"
            )
        return array


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@dataclass
class ChangeFeatureCache:
    """A scene-keyed feature store, plus the spec it was built under.

    The spec hash travels WITH the tensors. A cache built at a different
    resolution or with a different extractor is detected on read and refused,
    because a silently-mismatched cache produces a model that trains on one
    representation and is served another — a failure that looks like a modelling
    problem and is not one.
    """

    scene_keys: tuple[str, ...]
    features: Any
    spec_hash: str
    extractor_config: dict[str, Any]
    _index: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._index = {key: i for i, key in enumerate(self.scene_keys)}

    def __len__(self) -> int:
        return len(self.scene_keys)

    def has(self, scene_key: str) -> bool:
        return scene_key in (self._index or {})

    def get(self, scene_key: str) -> Any:
        if not self.has(scene_key):
            raise FeatureExtractionError(
                f"no cached change feature for scene {scene_key!r}"
            )
        return self.features[self._index[scene_key]]

    def write(self, path: str | Path) -> Path:
        import numpy as np

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            scene_keys=np.asarray(self.scene_keys, dtype=object).astype(str),
            features=self.features.astype(np.float32),
            spec_hash=np.asarray(self.spec_hash),
            extractor_config=np.asarray(
                json.dumps(self.extractor_config, sort_keys=True)
            ),
        )
        return p

    @classmethod
    def read(cls, path: str | Path, *, expect_spec_hash: str | None = None) -> "ChangeFeatureCache":
        import numpy as np

        p = Path(path)
        if not p.exists():
            raise FeatureExtractionError(f"change feature cache not found: {p}")
        with np.load(p, allow_pickle=False) as payload:
            cache = cls(
                scene_keys=tuple(str(k) for k in payload["scene_keys"]),
                features=payload["features"].astype(np.float32),
                spec_hash=str(payload["spec_hash"]),
                extractor_config=json.loads(str(payload["extractor_config"])),
            )
        if expect_spec_hash is not None and cache.spec_hash != expect_spec_hash:
            raise FeatureExtractionError(
                f"change feature cache at {p} was built under spec "
                f"{cache.spec_hash!r} but the caller expects "
                f"{expect_spec_hash!r}; re-extract rather than training on a "
                f"representation that does not match serving"
            )
        return cache


@dataclass
class TextFeatureCache:
    """A question-id-keyed text feature store for one split."""

    question_ids: tuple[int, ...]
    features: Any
    spec_hash: str

    def __len__(self) -> int:
        return len(self.question_ids)

    def write(self, path: str | Path) -> Path:
        import numpy as np

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            question_ids=np.asarray(self.question_ids, dtype=np.int64),
            features=self.features.astype(np.float32),
            spec_hash=np.asarray(self.spec_hash),
        )
        return p

    @classmethod
    def read(cls, path: str | Path, *, expect_spec_hash: str | None = None) -> "TextFeatureCache":
        import numpy as np

        p = Path(path)
        if not p.exists():
            raise FeatureExtractionError(f"text feature cache not found: {p}")
        with np.load(p, allow_pickle=False) as payload:
            cache = cls(
                question_ids=tuple(int(q) for q in payload["question_ids"]),
                features=payload["features"].astype(np.float32),
                spec_hash=str(payload["spec_hash"]),
            )
        if expect_spec_hash is not None and cache.spec_hash != expect_spec_hash:
            raise FeatureExtractionError(
                f"text feature cache at {p} was built under spec "
                f"{cache.spec_hash!r}, caller expects {expect_spec_hash!r}"
            )
        return cache


def extract_change_features(
    extractor: ChangeFeatureExtractor,
    scene_items: Iterable[tuple[str, str, str]],
    *,
    loader: Any | None = None,
    on_progress: Any | None = None,
) -> ChangeFeatureCache:
    """Extract features for `(scene_key, t1_path, t2_path)` triples.

    A scene whose imagery cannot be read is SKIPPED and reported through
    `on_progress`, never silently dropped: a scene missing from the cache is a
    training sample the model never sees, and the caller must be able to count
    how many that was.
    """
    import numpy as np

    if loader is None:
        from preprocessing.imagery import load_image_array as loader  # type: ignore

    keys: list[str] = []
    vectors: list[Any] = []
    failures: list[dict[str, str]] = []

    for index, (scene_key, t1_path, t2_path) in enumerate(scene_items):
        try:
            t1 = loader(t1_path)
            t2 = loader(t2_path)
            vector, _facts = extractor.extract(t1, t2)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {"scene_key": scene_key, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        keys.append(scene_key)
        vectors.append(vector)
        if on_progress is not None:
            on_progress(index + 1, scene_key)

    features = (
        np.stack(vectors).astype(np.float32)
        if vectors
        else np.zeros((0, CHANGE_FEATURE_DIM), dtype=np.float32)
    )
    cache = ChangeFeatureCache(
        scene_keys=tuple(keys),
        features=features,
        spec_hash=extractor.spec_hash,
        extractor_config=extractor.config_dict(),
    )
    cache.failures = failures  # type: ignore[attr-defined]
    return cache


__all__ = [
    "FEATURE_SPEC",
    "DEFAULT_IMAGE_SIZE",
    "CHANGE_STAT_THRESHOLDS",
    "CHANGE_GRID",
    "CHANGE_FEATURE_DIM",
    "LEVEL_POOLED_DIM",
    "CHANGE_STATS_DIM",
    "CHANGE_GRID_DIM",
    "TEXT_ENCODER_NAME",
    "TEXT_FEATURE_DIM",
    "FeatureExtractionError",
    "ChangeFeatureExtractor",
    "TextFeatureExtractor",
    "ChangeFeatureCache",
    "TextFeatureCache",
    "feature_spec_hash",
    "text_spec_hash",
    "build_change_feature_extractor",
    "extract_change_features",
]
