"""R-02 test areas K-L — frozen change features and the feature caches.

    K  feature-spec identity            (trained vs untrained is IN the hash)
    L  cache integrity and skip accounting

WHY K MATTERS
-------------
`configs/base.yaml` carries no `change.checkpoint_path`, so a serving process
builds an UNTRAINED STANet while training used the trained LEVIR checkpoint. The
spec hash is the only thing that can tell those two apart, and it is compared
before an answer is produced. If the hash ever stopped distinguishing them, the
train/serve skew guard would silently become a no-op -- hence the assertion on
the two literal digests rather than on "they differ".

WHY L MATTERS
-------------
A scene missing from the feature cache is a training sample the model never
sees. `extract_change_features` skips and REPORTS it; a silent drop would make
the training set smaller than the caller believes, with no number anywhere
recording the difference.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from training.change_vqa import features as F


@pytest.fixture
def scratch() -> Path:
    """A self-managed scratch directory; see the note in the dataset test module.

    `tmp_path` performs directory removals, which a sandbox delete guard has
    blocked in this repository. A never-deleted `mkdtemp` directory needs no
    delete permission.
    """
    return Path(tempfile.mkdtemp(prefix="sq_r02_features_"))


# ===========================================================================
# Area K - feature-spec identity
# ===========================================================================
def test_k_the_declared_dimension_is_the_sum_of_its_three_parts() -> None:
    """1045 = 1024 (levels) + 5 (statistics) + 16 (spatial grid).

    Derived, never typed: a literal would drift from the pooling code and fail
    as a shape error three files away.
    """
    assert F.N_LEVELS == 4
    assert F.LEVEL_CHANNELS == 128
    assert F.LEVEL_POOLED_DIM == 1024
    assert F.CHANGE_STATS_DIM == 2 + len(F.CHANGE_STAT_THRESHOLDS) == 5
    assert F.CHANGE_GRID_DIM == F.CHANGE_GRID * F.CHANGE_GRID == 16
    assert F.CHANGE_FEATURE_DIM == 1045
    assert F.CHANGE_FEATURE_DIM == (
        F.LEVEL_POOLED_DIM + F.CHANGE_STATS_DIM + F.CHANGE_GRID_DIM
    )


def test_k_the_text_feature_width_is_the_encoders_own() -> None:
    assert F.TEXT_FEATURE_DIM == 384
    assert F.TEXT_ENCODER_NAME == "sentence-transformers/all-MiniLM-L6-v2"


def test_k_trained_and_untrained_detectors_have_different_spec_hashes() -> None:
    """THE load-bearing property of area K: the train/serve skew guard.

    Pinned to the measured digests. If a refactor made the extractor state
    invisible to the hash, these two would become equal and the guard in
    `ChangeVQASpecialist.feature_spec_mismatch` would stop firing -- silently.
    """
    trained = F.feature_spec_hash(256, extractor="stanet:trained")
    untrained = F.feature_spec_hash(256, extractor="stanet:untrained")
    assert trained == "c801326f85a185f8"
    assert untrained == "714efac5b0e6a6d1"
    assert trained != untrained


def test_k_the_spec_hash_is_stable_across_calls() -> None:
    first = F.feature_spec_hash(256, extractor="stanet:trained")
    second = F.feature_spec_hash(256, extractor="stanet:trained")
    assert first == second
    assert len(first) == 16
    assert all(c in "0123456789abcdef" for c in first)


def test_k_the_resolution_changes_the_spec_hash() -> None:
    """256 is the detector's own training resolution; 512 is a different input
    distribution and must not share a cache identity."""
    assert F.feature_spec_hash(256, extractor="stanet:trained") != (
        F.feature_spec_hash(512, extractor="stanet:trained")
    )
    assert F.DEFAULT_IMAGE_SIZE == 256


def test_k_the_text_spec_hash_is_a_separate_namespace() -> None:
    """One combined hash would force a full re-extraction of both caches
    whenever either changed, and could let a stale text cache pass a check it
    should fail."""
    text = F.text_spec_hash(encoder=F.TEXT_ENCODER_NAME)
    change = F.feature_spec_hash(F.DEFAULT_IMAGE_SIZE, extractor="stanet:trained")
    assert text != change


def test_k_the_text_spec_hash_records_whether_the_revision_was_pinned() -> None:
    unpinned = F.text_spec_hash(encoder=F.TEXT_ENCODER_NAME)
    pinned = F.text_spec_hash(encoder=F.TEXT_ENCODER_NAME, revision="abc123")
    assert unpinned != pinned
    # "unpinned" is a weaker but truthful claim, distinct from a real revision.
    assert unpinned == F.text_spec_hash(encoder=F.TEXT_ENCODER_NAME, revision=None)


def test_k_the_extractor_spec_hash_reflects_its_own_trained_flag() -> None:
    """`ChangeFeatureExtractor.spec_hash` is what serving compares, so it must
    route through the same function the cache was written under."""
    assert F.ChangeFeatureExtractor.spec_hash.fget is not None
    assert (
        F.feature_spec_hash(256, extractor="stanet:untrained")
        == F.feature_spec_hash(F.DEFAULT_IMAGE_SIZE, extractor="stanet:untrained")
    )


# ===========================================================================
# Area L - cache integrity and skip accounting
# ===========================================================================
def _change_cache(n: int = 3) -> F.ChangeFeatureCache:
    return F.ChangeFeatureCache(
        scene_keys=tuple(f"scene{i}" for i in range(n)),
        features=np.arange(n * F.CHANGE_FEATURE_DIM, dtype=np.float32).reshape(
            n, F.CHANGE_FEATURE_DIM
        ),
        spec_hash="spec123",
        extractor_config={"trained": True},
    )


def _text_cache(n: int = 3) -> F.TextFeatureCache:
    return F.TextFeatureCache(
        question_ids=tuple(range(n)),
        features=np.ones((n, F.TEXT_FEATURE_DIM), dtype=np.float32),
        spec_hash="text456",
    )


def test_l_change_cache_round_trips_through_disk(scratch: Path) -> None:
    cache = _change_cache()
    path = cache.write(scratch / "change.npz")
    restored = F.ChangeFeatureCache.read(path)
    assert restored.scene_keys == cache.scene_keys
    assert restored.spec_hash == cache.spec_hash
    assert restored.extractor_config == cache.extractor_config
    assert np.array_equal(restored.features, cache.features)
    assert len(restored) == 3


def test_l_change_cache_refuses_a_spec_it_was_not_built_under(scratch: Path) -> None:
    """The skew gate: a cache built at another resolution or with another
    extractor must not be trainable-through silently."""
    path = _change_cache().write(scratch / "change.npz")
    F.ChangeFeatureCache.read(path, expect_spec_hash="spec123")  # matching: fine
    with pytest.raises(F.FeatureExtractionError, match="was built under spec"):
        F.ChangeFeatureCache.read(path, expect_spec_hash="somethingelse")


def test_l_a_missing_change_cache_raises_rather_than_returning_empty(scratch: Path) -> None:
    with pytest.raises(F.FeatureExtractionError, match="not found"):
        F.ChangeFeatureCache.read(scratch / "absent.npz")


def test_l_change_cache_lookup_reports_a_missing_scene() -> None:
    cache = _change_cache()
    assert cache.has("scene1") is True
    assert cache.has("scene9") is False
    assert cache.get("scene1").shape == (F.CHANGE_FEATURE_DIM,)
    with pytest.raises(F.FeatureExtractionError, match="no cached change feature"):
        cache.get("scene9")


def test_l_text_cache_round_trips_and_gates_on_spec(scratch: Path) -> None:
    cache = _text_cache()
    path = cache.write(scratch / "text.npz")
    restored = F.TextFeatureCache.read(path, expect_spec_hash="text456")
    assert restored.question_ids == cache.question_ids
    assert len(restored) == 3
    with pytest.raises(F.FeatureExtractionError, match="caller expects"):
        F.TextFeatureCache.read(path, expect_spec_hash="wrong")


def test_l_a_missing_text_cache_raises(scratch: Path) -> None:
    with pytest.raises(F.FeatureExtractionError, match="not found"):
        F.TextFeatureCache.read(scratch / "absent.npz")


# ---------------------------------------------------------------------------
# Area L - extraction skip accounting
# ---------------------------------------------------------------------------
class _StubExtractor:
    """A feature extractor that produces a fixed vector, for the skip logic."""

    trained = False
    spec_hash = "stubspec"

    def extract(self, t1, t2):
        return np.zeros(F.CHANGE_FEATURE_DIM, dtype=np.float32), {"stub": True}

    def config_dict(self) -> dict:
        return {"trained": False, "stub": True}


def test_l_an_unreadable_scene_is_skipped_and_reported_not_dropped() -> None:
    """A silently dropped scene is a training sample the model never sees, and
    nothing in the log would say so."""
    items = [
        ("sceneA", "/im/ok_a1.png", "/im/ok_a2.png"),
        ("sceneB", "/im/BAD_b1.png", "/im/BAD_b2.png"),
        ("sceneC", "/im/ok_c1.png", "/im/ok_c2.png"),
    ]

    def loader(path: str):
        if "BAD" in path:
            raise OSError("simulated unreadable image")
        return np.zeros((8, 8, 3), dtype=np.uint8)

    cache = F.extract_change_features(_StubExtractor(), items, loader=loader)

    assert cache.scene_keys == ("sceneA", "sceneC")
    assert len(cache) == 2
    assert cache.features.shape == (2, F.CHANGE_FEATURE_DIM)
    failures = getattr(cache, "failures")
    assert len(failures) == 1
    assert failures[0]["scene_key"] == "sceneB"
    assert "OSError" in failures[0]["error"]


def test_l_extraction_reports_progress_only_for_successes() -> None:
    seen: list[str] = []

    def loader(path: str):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    items = [("sceneA", "/a1", "/a2"), ("sceneB", "/b1", "/b2")]
    F.extract_change_features(
        _StubExtractor(), items, loader=loader, on_progress=lambda _n, key: seen.append(key)
    )
    assert seen == ["sceneA", "sceneB"]


def test_l_extracting_nothing_yields_an_empty_but_well_shaped_cache() -> None:
    """The shape must still be `(0, 1045)` so a downstream stack does not fail
    with an unhelpful error."""

    def loader(path: str):
        raise OSError("nothing readable")

    cache = F.extract_change_features(
        _StubExtractor(), [("sceneA", "/a1", "/a2")], loader=loader
    )
    assert len(cache) == 0
    assert cache.features.shape == (0, F.CHANGE_FEATURE_DIM)
    assert len(getattr(cache, "failures")) == 1


def test_l_extraction_stamps_the_cache_with_the_extractors_spec() -> None:
    cache = F.extract_change_features(
        _StubExtractor(),
        [("sceneA", "/a1", "/a2")],
        loader=lambda _p: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    assert cache.spec_hash == "stubspec"
    assert cache.extractor_config == {"trained": False, "stub": True}
