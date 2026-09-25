"""SatQuery AI Phase 9: LEVIR-CD bi-temporal change detection.

    dataset.py   LEVIR-CD discovery, loading, scene-disjoint split, leakage guard
    train.py     the Siamese detector's training loop, validation, checkpointing

Import explicitly. `train.py` pulls in torch; a controller that imported every
specialist at package load would pay that for every request, defeating the
lazy-loading design. Only the dataset symbols are re-exported here, matching
`training/grounding/__init__.py`.

The detector itself, the post-processing and the metrics live elsewhere and are
not re-exported, because they already have owners:

    specialists/change/stanet.py        STANetStyleChangeDetector, change_loss
    specialists/change/postprocess.py   change map -> regions, registration gate
    evaluation/metrics/change.py        precision / recall / F1 / IoU / mIoU
"""

from training.change.dataset import (
    LEVIR_SPLITS,
    LEVIRCDError,
    LEVIRItem,
    assert_image_disjoint,
    discover_levir_dataset,
    load_levir_dataset,
    split_by_image,
)

__all__ = [
    "LEVIR_SPLITS",
    "LEVIRCDError",
    "LEVIRItem",
    "assert_image_disjoint",
    "discover_levir_dataset",
    "load_levir_dataset",
    "split_by_image",
]
