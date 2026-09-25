"""SatQuery AI — router inference.

Ties the frozen encoder, the trained adapter, and the lexical fallback into a
single call that returns a validated `core.schemas.Intent`.

    query
      |
    encoder.encode       (frozen MiniLM, 384-d)
      |
    adapter              (5 heads, softmax / sigmoid)
      |
    confidence gate      (router.confidence_threshold)
      |
    lexical fallback     (only if below the gate)
      |
    Intent               (validated pydantic model)

The learned router is authoritative when it is confident. The fallback exists
only for the low-confidence case and for environments where the encoder cannot
load at all. Neither ever produces a task outside the frozen label space.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.errors import ModelLoadError, RoutingError
from core.schemas import Intent, Modality, Task
from router.adapter import IntentAdapter
from router.encoder import FrozenEncoder
from router.fallback import LexicalMatch, lexical_route
from router.label_space import (
    BINARY_HEADS,
    MODALITY_CLASSES,
    TASK_CLASSES,
)

#: Task label -> core.schemas.Task. They must stay in sync; asserted below.
_TASK_TO_SCHEMA: dict[str, Task] = {
    "vqa": Task.VQA,
    "caption": Task.CAPTION,
    "grounding": Task.GROUNDING,
    "change": Task.CHANGE,
    "optical_sar": Task.OPTICAL_SAR,
    "unsupported": Task.UNSUPPORTED,
}

_MODALITY_TO_SCHEMA: dict[str, Modality] = {
    "optical": Modality.OPTICAL,
    "sar": Modality.SAR,
    "optical_sar": Modality.OPTICAL_SAR,
    "unknown": Modality.UNKNOWN,
}


def _assert_schema_alignment() -> None:
    missing_tasks = [t for t in TASK_CLASSES if t not in _TASK_TO_SCHEMA]
    if missing_tasks:
        raise RoutingError(
            f"label_space TASK_CLASSES contains labels with no schema mapping: "
            f"{missing_tasks}"
        )
    missing_mods = [m for m in MODALITY_CLASSES if m not in _MODALITY_TO_SCHEMA]
    if missing_mods:
        raise RoutingError(
            f"label_space MODALITY_CLASSES contains labels with no schema mapping: "
            f"{missing_mods}"
        )


_assert_schema_alignment()


@dataclass
class RouterPrediction:
    """A router decision plus everything needed to explain it in a trace."""

    intent: Intent
    task_probs: dict[str, float] = field(default_factory=dict)
    modality_probs: dict[str, float] = field(default_factory=dict)
    binary_probs: dict[str, float] = field(default_factory=dict)
    used_fallback: bool = False
    fallback_rule: str | None = None
    matched_terms: tuple[str, ...] = ()
    above_threshold: bool = True

    def to_trace(self) -> dict[str, Any]:
        """Observable facts only. No chain-of-thought."""
        return {
            "task": self.intent.task.value,
            "modality": self.intent.modality.value,
            "temporal": self.intent.temporal,
            "spatial_output": self.intent.spatial_output,
            "language_output": self.intent.language_output,
            "confidence": round(self.intent.confidence, 4),
            "source": self.intent.source,
            "above_threshold": self.above_threshold,
            "used_fallback": self.used_fallback,
            "fallback_rule": self.fallback_rule,
        }


class IntentRouter:
    """Loads the adapter (and optionally the encoder) and routes queries."""

    def __init__(
        self,
        adapter: IntentAdapter | None = None,
        encoder: FrozenEncoder | None = None,
        confidence_threshold: float = 0.70,
        device: str = "cpu",
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise RoutingError(
                f"confidence_threshold must be in [0,1], got {confidence_threshold}"
            )
        self.adapter = adapter
        self.encoder = encoder
        self.confidence_threshold = confidence_threshold
        self.device = device

        if self.adapter is not None:
            self.adapter.eval()
            self.adapter.to(device)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config,
        adapter_path: str | Path | None = None,
        load_encoder: bool = True,
        device: str | None = None,
    ) -> "IntentRouter":
        """Build from the central config, optionally loading a trained adapter.

        IMPORTANT: `adapter_path` defaults to None, so the default router runs
        the LEXICAL FALLBACK, not the trained adapter. That default is
        deliberate -- it keeps tests fast and offline -- but it means a caller
        who never passes `adapter_path` gets fallback answers while believing
        the trained model is serving. The two are distinguishable but not
        obviously so: on the spec section 29 examples the fallback returns
        confidence 0.850-0.920 against the trained model's 0.780-1.000.

        Check `router.has_adapter` (and `router.adapter_source`) before
        trusting a routing decision as model-backed.
        """
        if device is None:
            configured = config.get("router.device", "auto")
            device = config.device_preference if configured == "auto" else configured

        threshold = float(config.get("router.confidence_threshold", 0.70))

        encoder: FrozenEncoder | None = None
        adapter: IntentAdapter | None = None

        if adapter_path is not None:
            adapter, metadata = load_adapter(adapter_path, device=device)
            # An adapter trained against a different input dimension cannot be
            # used with this encoder; fail loudly rather than producing garbage.
            expected_dim = int(config.get("router.embedding_dim", 384))
            if adapter.input_dim != expected_dim:
                raise ModelLoadError(
                    f"adapter input_dim={adapter.input_dim} does not match the "
                    f"configured embedding dim {expected_dim}",
                    specialist="router",
                )

        if load_encoder:
            from router.encoder import build_encoder

            encoder = build_encoder(config, device=device)
            if adapter is not None and adapter.input_dim != encoder.embedding_dim:
                raise ModelLoadError(
                    f"adapter input_dim={adapter.input_dim} but the loaded encoder "
                    f"produces {encoder.embedding_dim}-d embeddings",
                    specialist="router",
                )

        return cls(
            adapter=adapter,
            encoder=encoder,
            confidence_threshold=threshold,
            device=device,
        )

    # -- properties --------------------------------------------------------

    @property
    def has_adapter(self) -> bool:
        return self.adapter is not None

    @property
    def adapter_source(self) -> str:
        """Which path `route()` will take: 'trained' or 'lexical_fallback'.

        A caller can get a perfectly plausible routing decision from the
        fallback that did NOT come from the trained adapter -- see the WARNING
        in `from_config`. This makes the distinction checkable in one place
        instead of inferred from a confidence band.
        """
        return "trained" if self.has_adapter else "lexical_fallback"

    @property
    def has_encoder(self) -> bool:
        return self.encoder is not None

    # -- inference ---------------------------------------------------------

    def _predict_learned(self, query: str) -> RouterPrediction | None:
        """Run the encoder + adapter. Returns None when either is unavailable."""
        if self.adapter is None or self.encoder is None:
            return None

        embedding = self.encoder.encode_one(query, normalize=True)
        tensor = torch.from_numpy(embedding).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.adapter(tensor)
            task_probs = torch.softmax(out.task_logits, dim=-1)[0]
            modality_probs = torch.softmax(out.modality_logits, dim=-1)[0]
            temporal_p = torch.sigmoid(out.temporal_logit)[0]
            spatial_p = torch.sigmoid(out.spatial_logit)[0]
            language_p = torch.sigmoid(out.language_logit)[0]

        task_idx = int(torch.argmax(task_probs).item())
        modality_idx = int(torch.argmax(modality_probs).item())
        confidence = float(task_probs[task_idx].item())

        task_label = TASK_CLASSES[task_idx]
        modality_label = MODALITY_CLASSES[modality_idx]

        binary_probs = {
            "temporal": float(temporal_p.item()),
            "spatial_output": float(spatial_p.item()),
            "language_output": float(language_p.item()),
        }

        # Coherence repair. The heads are independent by construction, so a
        # confident task label with an incoherent binary head is possible. We
        # resolve in favour of the task label, because the task is what the
        # controller keys off — and we record that we did so.
        temporal = binary_probs["temporal"] >= 0.5
        spatial = binary_probs["spatial_output"] >= 0.5
        language = binary_probs["language_output"] >= 0.5

        if task_label in ("change",):
            temporal = True
        if task_label == "grounding":
            spatial = True
        if task_label == "unsupported":
            language = False
            spatial = False
            temporal = False
        if task_label == "optical_sar" and modality_label != "optical_sar":
            modality_label = "optical_sar"

        try:
            intent = Intent(
                task=_TASK_TO_SCHEMA[task_label],
                modality=_MODALITY_TO_SCHEMA[modality_label],
                temporal=temporal,
                spatial_output=spatial,
                language_output=language,
                confidence=confidence,
                source="learned",
            )
        except Exception as exc:  # noqa: BLE001 - pydantic validation
            raise RoutingError(
                f"learned router produced an invalid intent: {exc}",
                context={"query_length": len(query), "task": task_label},
            ) from exc

        return RouterPrediction(
            intent=intent,
            task_probs={name: float(task_probs[i].item())
                        for i, name in enumerate(TASK_CLASSES)},
            modality_probs={name: float(modality_probs[i].item())
                            for i, name in enumerate(MODALITY_CLASSES)},
            binary_probs=binary_probs,
            used_fallback=False,
            above_threshold=confidence >= self.confidence_threshold,
        )

    def _predict_fallback(self, query: str) -> RouterPrediction:
        match: LexicalMatch = lexical_route(query)
        intent = Intent(
            task=_TASK_TO_SCHEMA[match.task],
            modality=_MODALITY_TO_SCHEMA[match.modality],
            temporal=match.temporal,
            spatial_output=match.spatial_output,
            language_output=match.language_output,
            confidence=match.confidence,
            source="lexical_fallback",
        )
        return RouterPrediction(
            intent=intent,
            used_fallback=True,
            fallback_rule=match.rule,
            matched_terms=match.matched_terms,
            above_threshold=match.confidence >= self.confidence_threshold,
        )

    def route(self, query: str) -> RouterPrediction:
        """Route a query to an Intent.

        Order:
          1. learned router, if confident enough -> use it
          2. otherwise lexical fallback
          3. if the fallback also lands below threshold -> return it anyway,
             flagged, and let the controller decide. The router never refuses
             to answer; `above_threshold` carries the uncertainty.
        """
        if not isinstance(query, str) or not query.strip():
            raise RoutingError("query must be a non-empty string")

        learned: RouterPrediction | None = None
        if self.has_adapter and self.has_encoder:
            learned = self._predict_learned(query)
            if learned is not None and learned.above_threshold:
                return learned

        fallback = self._predict_fallback(query)

        if learned is None:
            return fallback

        # Both ran. Prefer whichever is more confident; the fallback wins ties
        # on interpretability (its rule and matched terms are inspectable).
        if fallback.intent.confidence >= learned.intent.confidence:
            return fallback
        return learned

    def route_batch(self, queries: list[str]) -> list[RouterPrediction]:
        return [self.route(q) for q in queries]


# ---------------------------------------------------------------------------
# Adapter persistence
# ---------------------------------------------------------------------------

ADAPTER_WEIGHTS = "adapter.pt"
ADAPTER_METADATA = "metadata.json"


def save_adapter(
    directory: str | Path,
    adapter: IntentAdapter,
    metadata: dict[str, Any],
) -> Path:
    """Write the adapter plus its metadata. Metadata is mandatory."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "config": adapter.config_dict(),
        },
        path / ADAPTER_WEIGHTS,
    )

    payload = dict(metadata)
    payload.setdefault("adapter_config", adapter.config_dict())
    payload.setdefault("num_parameters", adapter.num_parameters())
    (path / ADAPTER_METADATA).write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    return path


def load_adapter(
    directory: str | Path,
    device: str = "cpu",
) -> tuple[IntentAdapter, dict[str, Any]]:
    """Load a saved adapter and its metadata.

    Args:
        directory: the artifact DIRECTORY written by `save_adapter`, OR the
            weights file itself. Both are accepted: `save_adapter` writes
            `<dir>/adapter.pt`, so a caller who was handed the weight path --
            the obvious thing to try, since the name reads like a weight file --
            would otherwise have `.pt` appended a second time.

        device: torch device string.

    Raises:
        ModelLoadError: the weights are absent, unreadable, or carry no config.
    """
    path = Path(directory)

    # Accept the weights file directly. Without this, passing `.../adapter.pt`
    # produces `<...>/adapter.pt/adapter.pt`, whose doubled filename is the
    # tell but is easy to misread as a genuinely missing file.
    if path.is_file():
        if path.name != ADAPTER_WEIGHTS and path.suffix != ".pt":
            raise ModelLoadError(
                f"expected an adapter directory or a '{ADAPTER_WEIGHTS}' file, "
                f"got the file {path}",
                specialist="router",
            )
        weights = path
        meta_path = path.parent / ADAPTER_METADATA
    else:
        weights = path / ADAPTER_WEIGHTS
        meta_path = path / ADAPTER_METADATA

    if not weights.exists():
        # Name the specific confusion when a `.pt` path was given that exists
        # as a directory or not at all, rather than reporting a doubled path.
        if path.suffix == ".pt":
            raise ModelLoadError(
                f"no adapter weights at {weights}: {path} looks like a weights "
                f"file, not an adapter directory. Pass the file itself (it must "
                f"exist and end in .pt) or the directory containing "
                f"'{ADAPTER_WEIGHTS}'.",
                specialist="router",
                context={"given": str(path), "resolved": str(weights)},
            )
        raise ModelLoadError(
            f"adapter weights not found at {weights}", specialist="router"
        )

    try:
        payload = torch.load(weights, map_location=device, weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(
            f"could not read adapter weights: {exc}", specialist="router"
        ) from exc

    config = payload.get("config")
    if not config:
        raise ModelLoadError(
            "adapter checkpoint has no embedded config; it cannot be reconstructed "
            "without guessing the architecture",
            specialist="router",
        )

    adapter = IntentAdapter.from_config_dict(config)
    try:
        adapter.load_state_dict(payload["state_dict"], strict=True)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(
            f"adapter state_dict does not match the reconstructed architecture: {exc}",
            specialist="router",
        ) from exc

    adapter.to(device)
    adapter.eval()

    metadata: dict[str, Any] = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {"_warning": "metadata.json was not valid JSON"}

    return adapter, metadata


__all__ = [
    "RouterPrediction",
    "IntentRouter",
    "save_adapter",
    "load_adapter",
    "ADAPTER_WEIGHTS",
    "ADAPTER_METADATA",
]