"""SatQuery AI Phase 12: the optical-SAR fusion head and its frozen-feature trainer.

    train.py    the training loop over cached CROMA features, the mandatory
                channel-dropout wiring, the S1<->S2 pairing consumer, and the
                `--arm` normalisation mechanism

Import explicitly:

    from training.fusion.train import train_fusion_head

WHY THIS PACKAGE IS LIGHT AT PACKAGE LEVEL
------------------------------------------
`train.py` reaches for torch (the head is a torch module). A controller that
imported every training package at load would pay that cost on every request,
defeating the lazy-loading design (plan section 49). So the public names below
are re-exported LAZILY through a module `__getattr__`: `import training.fusion`
is cheap, and `from training.fusion import train_fusion_head` still works.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FusionTrainingError",
    "Arm",
    "ARMS",
    "ARM_CONTROL",
    "ARM_VARIANT",
    "resolve_arm",
    "apply_arm",
    "USE_8_BIT_ENV",
    "OPTICAL_DROPOUT_RATES",
    "SAR_DROPOUT_RATES",
    "RESULT_STATUS",
    "FusionFeature",
    "EpochRecord",
    "ValidationResult",
    "TrainingResult",
    "FusionFeatureDataset",
    "collate",
    "prepare_fusion_batch",
    "assert_split_disjoint",
    "evaluate_fusion_head",
    "train_fusion_head",
    "load_trained_fusion_head",
    "write_feature_cache",
    "read_feature_cache",
    "feature_cache_path",
    "CACHE_VERSION",
    "DEFAULT_FEATURE_CACHE_DIR",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve the public names from `training.fusion.train`."""
    if name in __all__:
        from training.fusion import train as _train

        return getattr(_train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
