"""SatQuery AI router: frozen MiniLM encoder + multi-head intent adapter.

Public surface:

    from router import IntentRouter, build_corpus, lexical_route

    router = IntentRouter.from_config(config, adapter_path="artifacts/router/...")
    prediction = router.route("Show me the water body.")
    prediction.intent   # a validated core.schemas.Intent

Module map:

    label_space.py   the five-head output ontology
    encoder.py       frozen MiniLM wrapper (no gradients, cached embeddings)
    adapter.py       trainable multi-head classifier (nn.Module)
    dataset.py       curated + template corpus, group-tagged for leakage-safe splits
    fallback.py      deterministic lexical router
    classifier.py    IntentRouter — inference, persistence, confidence gate
    train.py         cached-embedding training with group-level split guard
"""

from router.adapter import AdapterOutput, IntentAdapter
from router.classifier import (
    ADAPTER_METADATA,
    ADAPTER_WEIGHTS,
    IntentRouter,
    RouterPrediction,
    load_adapter,
    save_adapter,
)
from router.dataset import (
    RouterCorpus,
    RouterExample,
    build_corpus,
    split_by_group,
    split_leakage_report,
)
from router.encoder import FrozenEncoder, build_encoder
from router.fallback import LexicalMatch, lexical_route
from router.label_space import (
    BINARY_HEADS,
    MODALITY_CLASSES,
    NUM_BINARY_HEADS,
    NUM_MODALITIES,
    NUM_TASKS,
    TASK_CLASSES,
)

__all__ = [
    # ontology
    "TASK_CLASSES",
    "MODALITY_CLASSES",
    "BINARY_HEADS",
    "NUM_TASKS",
    "NUM_MODALITIES",
    "NUM_BINARY_HEADS",
    # encoder
    "FrozenEncoder",
    "build_encoder",
    # adapter
    "IntentAdapter",
    "AdapterOutput",
    # data
    "RouterExample",
    "RouterCorpus",
    "build_corpus",
    "split_by_group",
    "split_leakage_report",
    # fallback
    "lexical_route",
    "LexicalMatch",
    # inference
    "IntentRouter",
    "RouterPrediction",
    "save_adapter",
    "load_adapter",
    "ADAPTER_WEIGHTS",
    "ADAPTER_METADATA",
]