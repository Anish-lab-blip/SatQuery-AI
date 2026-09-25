"""Tests for the frozen-feature cache.

The load-bearing test is `test_text_encoding_is_batched_not_per_phrase`.

A real run appeared to hang at "images 15600/15699" and was interrupted. The
image stage had actually COMPLETED -- 15,699 .npz files on disk, 11 minutes
before the interruption -- and the stage still running was text encoding,
which issued one forward pass per phrase and printed nothing. 11,589 phrases
took ~660 s (~57 ms each).

These tests pin both halves of the fix: the call shape (batched, not per
phrase) and the resumability (an existing cache entry is never re-encoded).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from training.grounding.dataset import (
    GroundingItem,
    attach_cached,
    extract_features,
    image_cache_path,
    text_cache_path,
)

GRID = 7
DIM = 512


class CountingEncoder:
    """Encoder stub that records the size of every text batch it receives.

    `text_calls` is the whole point: under the old implementation it would be
    a list of 1s, one entry per phrase.
    """

    def __init__(self, dim: int = DIM, grid: int = GRID) -> None:
        self.dim = dim
        self.grid = grid
        self.text_calls: list[int] = []
        self.image_calls = 0

    def encode_text(self, texts):
        self.text_calls.append(len(texts))
        return np.zeros((len(texts), self.dim), dtype=np.float32)

    def encode_image(self, image):
        self.image_calls += 1

        class _Encoded:
            patch_tokens = np.zeros((self.grid * self.grid, self.dim),
                                    dtype=np.float32)
            cls = np.zeros(self.dim, dtype=np.float32)

        return _Encoded()


class FailingTextEncoder(CountingEncoder):
    """Raises on the first batch only, to exercise the failure counter."""

    def __init__(self, *args, fail_first: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fail_first = fail_first
        self._calls = 0

    def encode_text(self, texts):
        self.text_calls.append(len(texts))
        self._calls += 1
        if self.fail_first and self._calls == 1:
            raise RuntimeError("simulated encoding failure")
        return np.zeros((len(texts), self.dim), dtype=np.float32)


def write_source_image(path: Path, size: int = 64) -> Path:
    """Write a real, decodable PNG so an item can actually be encoded.

    `extract_features` counts an item with no `source_path` as
    `images_missing_source` and never calls the encoder. A fixture that omits
    the path therefore tests the missing-source branch, not the encode branch
    -- which is exactly how `test_partial_cache_resumes_the_remainder` failed:
    it asserted 6 encodes and got 0, because all 6 items had no source.
    """
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), (120, 90, 60)).save(path)
    return path


def make_items(
    n_images: int,
    phrases_per_image: int = 1,
    source_root: Path | None = None,
) -> list[GroundingItem]:
    """Items for the cache tests.

    Args:
        source_root: when given, a real PNG is written per image and attached
            as `source_path`, so the encoder path is exercised. When None,
            items carry no source and land in `images_missing_source`.
    """
    items: list[GroundingItem] = []
    for i in range(n_images):
        key = f"img_{i:04d}"
        source = (
            write_source_image(source_root / f"{key}.png")
            if source_root is not None
            else None
        )
        for j in range(phrases_per_image):
            items.append(
                GroundingItem(
                    sample_id=f"{key}_ref_{j}",
                    image_key=key,
                    phrase=f"the object number {j}",
                    box=[0.2, 0.2, 0.6, 0.6],
                    source_path=source,
                )
            )
    return items


def write_image_cache(cache_dir: Path, key: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        image_cache_path(cache_dir, key),
        patches=np.zeros((GRID * GRID, DIM), dtype=np.float16),
        cls=np.zeros(DIM, dtype=np.float16),
    )


# ---------------------------------------------------------------------------
# The defect this file exists for
# ---------------------------------------------------------------------------

def test_text_encoding_is_batched_not_per_phrase(tmp_path: Path) -> None:
    """Every text call must carry many phrases, not one.

    Measured before the fix: 11,589 phrases -> 11,589 forward passes, ~57 ms
    each, ~660 s total, with no progress output.
    """
    items = make_items(n_images=1, phrases_per_image=1)
    # 600 distinct phrases, so batching is unambiguous.
    items = [
        GroundingItem(sample_id=f"s{i}", image_key="img_0000",
                      phrase=f"phrase {i}", box=[0.2, 0.2, 0.6, 0.6])
        for i in range(600)
    ]
    write_image_cache(tmp_path, "img_0000")

    enc = CountingEncoder()
    extract_features(items, enc, tmp_path, progress_every=0,
                     text_batch_size=256)

    assert enc.text_calls, "encoder was never called"
    assert max(enc.text_calls) > 1, (
        f"text encoding is still one phrase per call: {enc.text_calls[:5]}"
    )
    assert enc.text_calls == [256, 256, 88], (
        f"expected three batched calls, got {enc.text_calls}"
    )


def test_batch_size_is_honoured(tmp_path: Path) -> None:
    items = [
        GroundingItem(sample_id=f"s{i}", image_key="img_0000",
                      phrase=f"phrase {i}", box=[0.2, 0.2, 0.6, 0.6])
        for i in range(100)
    ]
    write_image_cache(tmp_path, "img_0000")
    enc = CountingEncoder()
    extract_features(items, enc, tmp_path, progress_every=0,
                     text_batch_size=32)
    assert enc.text_calls == [32, 32, 32, 4]


def test_every_phrase_ends_up_cached(tmp_path: Path) -> None:
    items = [
        GroundingItem(sample_id=f"s{i}", image_key="img_0000",
                      phrase=f"phrase {i}", box=[0.2, 0.2, 0.6, 0.6])
        for i in range(50)
    ]
    write_image_cache(tmp_path, "img_0000")
    report = extract_features(items, CountingEncoder(), tmp_path,
                              progress_every=0, text_batch_size=16)

    assert report["text_cache_misses"] == 50
    assert report["text_failed"] == 0
    for item in items:
        assert text_cache_path(tmp_path, item.phrase).exists(), item.phrase


# ---------------------------------------------------------------------------
# Resumability -- the interrupted run must not lose completed work
# ---------------------------------------------------------------------------

def test_existing_text_vectors_are_not_re_encoded(tmp_path: Path) -> None:
    """A rerun after an interruption reuses what is already on disk.

    The interrupted run left 11,589 valid vectors behind. Re-encoding them
    would waste the same 11 minutes again.
    """
    items = [
        GroundingItem(sample_id=f"s{i}", image_key="img_0000",
                      phrase=f"phrase {i}", box=[0.2, 0.2, 0.6, 0.6])
        for i in range(20)
    ]
    write_image_cache(tmp_path, "img_0000")

    first = CountingEncoder()
    report1 = extract_features(items, first, tmp_path, progress_every=0,
                               text_batch_size=8)
    assert report1["text_cache_misses"] == 20

    second = CountingEncoder()
    report2 = extract_features(items, second, tmp_path, progress_every=0,
                               text_batch_size=8)
    assert report2["text_cache_hits"] == 20
    assert report2["text_cache_misses"] == 0
    assert second.text_calls == [], "already-cached phrases were re-encoded"


def test_existing_image_entries_are_not_re_encoded(tmp_path: Path) -> None:
    items = make_items(n_images=5)
    for item in items:
        write_image_cache(tmp_path, item.image_key)

    enc = CountingEncoder()
    report = extract_features(items, enc, tmp_path, progress_every=0)
    assert report["image_cache_hits"] == 5
    assert report["image_cache_misses"] == 0
    assert enc.image_calls == 0


def test_partial_cache_resumes_the_remainder(tmp_path: Path) -> None:
    """Half the images present -> only the other half is encoded.

    Sources are real files here. The first version of this test used items
    with no `source_path`, so all 6 uncached images were counted as
    `missing_source` and `image_cache_misses` was 0 -- the assertion was
    checking a number the fixture could never produce.
    """
    items = make_items(n_images=10, source_root=tmp_path / "src")
    for item in items[:4]:
        write_image_cache(tmp_path, item.image_key)

    enc = CountingEncoder()
    report = extract_features(items, enc, tmp_path, progress_every=0)

    assert report["images_missing_source"] == 0
    assert report["image_cache_hits"] == 4
    assert report["image_cache_misses"] == 6
    assert enc.image_calls == 6


def test_completion_line_accounts_for_every_image(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """The image summary must name all four outcomes.

    A run printed "(4 cached, 0 encoded, 0 undecodable)" for 10 images and six
    vanished from the line. `missing_source` is the counter the handoff doc
    tells the operator to check, so it must be visible.
    """
    items = make_items(n_images=3, source_root=tmp_path / "src")
    write_image_cache(tmp_path, items[0].image_key)

    extract_features(items, CountingEncoder(), tmp_path, progress_every=0)
    out = capsys.readouterr().out

    assert "missing source" in out, f"summary omitted the counter:\n{out}"
    assert "1 cached" in out
    assert "2 encoded" in out


def test_missing_source_is_flagged_loudly(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A non-zero missing_source must not be a silent number in a list."""
    items = make_items(n_images=3)          # no source_root -> no sources
    extract_features(items, CountingEncoder(), tmp_path, progress_every=0)
    out = capsys.readouterr().out
    assert "had no source path" in out, out


def test_fully_sourced_run_reports_zero_missing_source(tmp_path: Path) -> None:
    """The negative case: a healthy corpus must report no missing sources."""
    items = make_items(n_images=4, source_root=tmp_path / "src")
    report = extract_features(items, CountingEncoder(), tmp_path,
                              progress_every=0)
    assert report["images_missing_source"] == 0
    assert report["image_cache_misses"] == 4


# ---------------------------------------------------------------------------
# Failure accounting
# ---------------------------------------------------------------------------

def test_failed_text_batch_is_counted_not_zero_filled(tmp_path: Path) -> None:
    """A chunk that will not encode is counted, and writes nothing.

    Zero vectors would train the head on blank text and the loss would look
    like it was converging.
    """
    items = [
        GroundingItem(sample_id=f"s{i}", image_key="img_0000",
                      phrase=f"phrase {i}", box=[0.2, 0.2, 0.6, 0.6])
        for i in range(64)
    ]
    write_image_cache(tmp_path, "img_0000")

    enc = FailingTextEncoder(fail_first=True)
    report = extract_features(items, enc, tmp_path, progress_every=0,
                              text_batch_size=16)

    assert report["text_failed"] == 16
    assert report["text_cache_misses"] == 48
    # The failed batch left no files behind.
    assert not text_cache_path(tmp_path, "phrase 0").exists()
    # A later batch did write.
    assert text_cache_path(tmp_path, "phrase 63").exists()


def test_missing_source_is_counted_separately_from_undecodable(tmp_path: Path) -> None:
    """Two different problems must not collapse into one counter."""
    items = [
        GroundingItem(sample_id="a", image_key="has_source",
                      phrase="p", box=[0.2, 0.2, 0.6, 0.6],
                      source_path=tmp_path / "does_not_exist.png"),
        GroundingItem(sample_id="b", image_key="no_source",
                      phrase="p", box=[0.2, 0.2, 0.6, 0.6]),
    ]
    report = extract_features(items, CountingEncoder(), tmp_path,
                              progress_every=0)
    assert report["images_undecodable"] == 1   # file named but unreadable
    assert report["images_missing_source"] == 1  # no path at all


# ---------------------------------------------------------------------------
# Phase attribution -- so a future stall is measurable, not guessed
# ---------------------------------------------------------------------------

def test_report_attributes_time_to_each_phase(tmp_path: Path) -> None:
    items = make_items(n_images=3)
    report = extract_features(items, CountingEncoder(), tmp_path,
                              progress_every=0)
    for key in ("image_seconds", "text_seconds", "seconds"):
        assert key in report, f"missing phase timing {key}"
        assert report[key] >= 0.0


# ---------------------------------------------------------------------------
# attach_cached
# ---------------------------------------------------------------------------

def test_attach_skips_an_unreadable_text_vector(tmp_path: Path) -> None:
    """A truncated .npy must drop one item, not abort the whole attach.

    A killed extraction can leave a partial file. Raising here would make an
    interrupted run look like a corrupt cache.
    """
    items = [
        GroundingItem(sample_id="ok", image_key="img_a", phrase="good",
                      box=[0.2, 0.2, 0.6, 0.6]),
        GroundingItem(sample_id="bad", image_key="img_b", phrase="truncated",
                      box=[0.2, 0.2, 0.6, 0.6]),
    ]
    write_image_cache(tmp_path, "img_a")
    write_image_cache(tmp_path, "img_b")
    text_cache_path(tmp_path, "good").parent.mkdir(parents=True, exist_ok=True)
    np.save(text_cache_path(tmp_path, "good"),
            np.zeros(DIM, dtype=np.float16))
    # Write a deliberately truncated vector file.
    bad = text_cache_path(tmp_path, "truncated")
    bad.write_bytes(b"\x93NUMPY" + b"\x00" * 20)

    kept = attach_cached(items, tmp_path, verbose=False)
    assert [i.sample_id for i in kept] == ["ok"]


def test_attach_keeps_items_with_both_entries(tmp_path: Path) -> None:
    items = [
        GroundingItem(sample_id="s", image_key="img_a", phrase="good",
                      box=[0.2, 0.2, 0.6, 0.6])
    ]
    write_image_cache(tmp_path, "img_a")
    text_cache_path(tmp_path, "good").parent.mkdir(parents=True, exist_ok=True)
    np.save(text_cache_path(tmp_path, "good"),
            np.zeros(DIM, dtype=np.float16))

    kept = attach_cached(items, tmp_path, verbose=False)
    assert len(kept) == 1
    assert kept[0].patches_path is not None
    assert kept[0].text is not None
    assert kept[0].text.shape == (DIM,)