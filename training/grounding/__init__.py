"""SatQuery AI Phase 8: the learned grounding head over frozen RemoteCLIP.

    dataset.py   cached frozen features, image-disjoint split, feature extraction
    train.py     head training, validation on the benchmark metric, checkpointing

Import explicitly. `train.py` pulls in torch; a controller that imported every
specialist at package load would pay that for every request, defeating the
lazy-loading design.
"""

from training.grounding.dataset import (
    CACHE_VERSION,
    GroundingDataError,
    GroundingItem,
    assert_image_disjoint,
    attach_cached,
    build_items,
    cache_dir_for,
    extract_features,
    image_cache_path,
    split_by_image,
    text_cache_path,
)

__all__ = [
    "CACHE_VERSION",
    "GroundingDataError",
    "GroundingItem",
    "assert_image_disjoint",
    "attach_cached",
    "build_items",
    "cache_dir_for",
    "extract_features",
    "image_cache_path",
    "split_by_image",
    "text_cache_path",
]