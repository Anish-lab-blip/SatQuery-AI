"""SatQuery AI — frozen MiniLM sentence encoder.

Verified contract (Phase 4 probe, 2026-09-16, sentence-transformers 6.0.1):

    SentenceTransformer(
        model_name_or_path='sentence-transformers/all-MiniLM-L6-v2',
        revision='1110a243fdf4',      # pinned, verified reachable
        device='cpu',
    )
    .get_sentence_embedding_dimension()  -> 384
    .tokenizer.model_max_length          -> 256     (finding F4-1)
    .max_seq_length                      -> 256
    params                               -> 22,713,216
    encode(64 queries, CPU)              -> 0.118 s

The encoder is FROZEN. It exposes no trainable parameters and returns numpy
arrays with no gradient graph, which is what makes cached-embedding training
possible (finding F4-2).
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from core.errors import ModelLoadError

#: The tokenizer's own ceiling, verified by probe. Truncating above this is a
#: silent no-op, so the encoder refuses rather than pretending.
VERIFIED_TOKENIZER_MAX_LENGTH = 256

#: Verified embedding dimension for all-MiniLM-L6-v2.
VERIFIED_EMBEDDING_DIM = 384


class FrozenEncoder:
    """Thin wrapper over SentenceTransformer with the encoder held frozen.

    Deliberately narrow: it encodes text and reports its dimensions. It does
    not train, does not expose the underlying model, and does not permit
    gradient flow. Everything downstream treats it as a pure function.
    """

    def __init__(
        self,
        model_name: str,
        revision: str,
        max_length: int,
        device: str = "cpu",
        trust_remote_code: bool = False,
    ) -> None:
        if max_length < 1:
            raise ModelLoadError(f"max_length must be >= 1, got {max_length}")
        if max_length > VERIFIED_TOKENIZER_MAX_LENGTH:
            raise ModelLoadError(
                f"max_length={max_length} exceeds the MiniLM tokenizer ceiling of "
                f"{VERIFIED_TOKENIZER_MAX_LENGTH}; truncation would be a silent no-op"
            )

        self.model_name = model_name
        self.revision = revision
        self.max_length = max_length
        self.device = device
        self._load_seconds: float | None = None

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment issue
            raise ModelLoadError(
                f"sentence-transformers is not installed: {exc}", specialist="router"
            ) from exc

        started = time.time()
        try:
            self._model = SentenceTransformer(
                model_name,
                revision=revision,
                device=device,
                trust_remote_code=trust_remote_code,
            )
        except Exception as exc:  # noqa: BLE001 - hub errors are numerous
            raise ModelLoadError(
                f"could not load encoder '{model_name}' @ {revision}: {exc}",
                specialist="router",
                context={"model": model_name, "revision": revision},
            ) from exc
        self._load_seconds = time.time() - started

        # Freeze. This is the whole architectural premise.
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad_(False)

        # Apply the configured truncation for real (finding F4-1). Setting
        # max_seq_length rewrites the tokenizer's truncation limit, so the
        # config value is enforced rather than merely documented.
        self._model.max_seq_length = max_length

        tokenizer_limit = int(getattr(self._model.tokenizer, "model_max_length", 0))
        if tokenizer_limit and tokenizer_limit > 10**6:
            # Some tokenizers report a sentinel for "no limit".
            tokenizer_limit = VERIFIED_TOKENIZER_MAX_LENGTH
        self._tokenizer_max_length = tokenizer_limit

    # -- properties --------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    @property
    def tokenizer_max_length(self) -> int:
        return self._tokenizer_max_length

    @property
    def load_seconds(self) -> float | None:
        return self._load_seconds

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self._model.parameters())

    # -- inference ---------------------------------------------------------

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts to a (n, embedding_dim) float32 array.

        The returned array is detached: it carries no autograd history. That is
        intentional — the adapter consumes it as a fixed input feature.
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        cleaned = [t if isinstance(t, str) else str(t) for t in texts]
        try:
            embeddings = self._model.encode(
                cleaned,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"encoder inference failed: {exc}", specialist="router"
            ) from exc

        return np.asarray(embeddings, dtype=np.float32)

    def encode_one(self, text: str, normalize: bool = True) -> np.ndarray:
        return self.encode([text], batch_size=1, normalize=normalize)[0]

    def __repr__(self) -> str:
        return (
            f"<FrozenEncoder {self.model_name}@{self.revision} "
            f"dim={self.embedding_dim} max_len={self.max_length} "
            f"device={self.device}>"
        )


def build_encoder(config, device: str | None = None) -> FrozenEncoder:
    """Construct the encoder from the central configuration registry."""
    if device is None:
        configured = config.get("router.device", "auto")
        if configured == "auto":
            device = config.device_preference
        else:
            device = configured

    return FrozenEncoder(
        model_name=config.require("router.model"),
        revision=config.require("router.revision"),
        max_length=int(config.get("router.max_length", 128)),
        device=device,
    )


__all__ = [
    "FrozenEncoder",
    "build_encoder",
    "VERIFIED_EMBEDDING_DIM",
    "VERIFIED_TOKENIZER_MAX_LENGTH",
]