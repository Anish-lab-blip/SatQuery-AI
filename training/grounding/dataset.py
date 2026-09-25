"""SatQuery AI — Phase 8 grounding dataset over frozen RemoteCLIP features.

The encoder is FROZEN, so a patch encoding is a pure function of (image,
resolution). That makes feature extraction cacheable exactly like the router's
sentence embeddings (finding F4-2), and it means GPU time is spent ONCE instead
of once per epoch.

    image  -> RemoteCLIP(224) -> (49, 512) patch tokens   -- cached
    phrase -> RemoteCLIP text -> (512,)    text embedding -- cached

Only the head trains. An epoch over cached features is a matrix operation.

LEAKAGE: SPLIT BY IMAGE, NOT BY RECORD
--------------------------------------
VRSBench train carries 142,390 conversations across 20,262 images; 36,313 are
`[refer]` turns. Several refers share one image. Splitting by RECORD would put
two referring expressions for the same photograph in different partitions —
the same failure mode as splitting tiles from one scene, and the model would be
evaluated on an image it trained on.

The split key is therefore the image filename. This mirrors
`evaluation.leakage.assign_splits_by_scene` deliberately: identical failure
mode, identical guard.

Verified disjointness (scripts/probe_grounding_head_contract.py):
    train images 20,262 | eval images 9,318 | INTERSECTION 0 | stems 0
so a head trained on train cannot have seen any eval image.

CACHE FORMAT
------------
One `.npz` per image:
    patches : (49, 512) float16      ~50 KB
    cls     : (512,)    float16

float16 halves the footprint (20,262 images -> ~1 GB) and the head trains in
fp16 on a T4 anyway. Text embeddings are cached separately, keyed by phrase,
because 36,313 refers share far fewer distinct strings than that.
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from core.errors import SatQueryError

#: Feature-cache layout version. Bump when the encoder, the resolution or the
#: set of stored fields changes, so a stale cache is a MISS rather than a silent
#: shape error in the middle of training.
CACHE_VERSION = "v1"


class GroundingDataError(SatQueryError):
    code = "grounding_data_error"
    user_message = "The grounding training data could not be prepared."


@dataclass
class GroundingItem:
    """One training example: frozen features plus a target box."""

    sample_id: str
    image_key: str              # the split key; never crosses partitions
    phrase: str
    box: list[float]            # normalized xyxy

    #: Absolute path to the source image. `extract_features` needs this to fill
    #: the cache. An item without it cannot be encoded -- and `build_items`
    #: populates it from the loader, which is the only place that knows it.
    source_path: Path | None = None

    patches_path: Path | None = None
    patches: np.ndarray | None = None
    text: np.ndarray | None = None

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """Materialize (patches, text). Reads the cache on first call."""
        if self.patches is None:
            if self.patches_path is None:
                raise GroundingDataError(
                    f"item {self.sample_id} has neither patches nor patches_path"
                )
            with np.load(self.patches_path) as data:
                self.patches = data["patches"].astype(np.float32)
        if self.text is None:
            raise GroundingDataError(
                f"item {self.sample_id} has no text embedding attached; "
                f"call attach_cached() first"
            )
        return self.patches, self.text


# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------

def cache_dir_for(root: Path, resolution: int, tag: str = "") -> Path:
    return Path(root) / f"remoteclip_{resolution}_{CACHE_VERSION}{tag}"


def image_cache_path(cache_dir: Path, image_key: str) -> Path:
    """Deterministic, filesystem-safe cache path for one image key."""
    digest = hashlib.sha256(image_key.encode()).hexdigest()[:16]
    return Path(cache_dir) / f"{digest}.npz"


def text_cache_path(cache_dir: Path, phrase: str) -> Path:
    digest = hashlib.sha256(phrase.encode()).hexdigest()[:16]
    return Path(cache_dir) / "text" / f"{digest}.npy"


# ---------------------------------------------------------------------------
# Building the item list
# ---------------------------------------------------------------------------

def build_items(
    data_root: str | Path,
    *,
    json_name: str = "VRSBench_train.json",
    limit: int | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> list[GroundingItem]:
    """Load referring samples and turn them into items with image split keys.

    Args:
        data_root: VRSBench root (holds the JSON and the image tree).
        json_name: which annotation file. Defaults to the TRAIN file -- that is
            the whole point of the parameter. The loader's default scan is
            shallowest-first, which alphabetically picks the EVAL file, so a
            caller that forgets to name it silently trains on eval.
        limit: cap the number of items, deterministically sampled.
        seed: controls the sampling.
        verbose: pass through to the loader.
    """
    from training.data.vrsbench import load_referring_samples

    samples = load_referring_samples(
        str(data_root),
        limit=None,                 # sample AFTER building, so the cap is per-item
        seed=seed,
        verbose=verbose,
        open_images=False,
        json_name=json_name,
    )

    items: list[GroundingItem] = []
    for sample in samples:
        if sample.image_path is None:
            continue
        items.append(
            GroundingItem(
                sample_id=sample.sample_id,
                image_key=sample.image_path.stem,
                phrase=sample.phrase,
                box=[float(v) for v in sample.box],
                source_path=sample.image_path,
            )
        )

    if limit is not None and len(items) > limit:
        rng = random.Random(seed)
        rng.shuffle(items)
        items = items[:limit]

    return items


def split_by_image(
    items: Sequence[GroundingItem],
    *,
    val_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[list[GroundingItem], list[GroundingItem]]:
    """Split by IMAGE KEY. No image appears in both partitions.

    Raises:
        GroundingDataError: bad ratio, or the split produced an empty partition.
    """
    if not 0.0 < val_fraction < 1.0:
        raise GroundingDataError(
            f"val_fraction must be in (0,1), got {val_fraction}"
        )

    by_image: dict[str, list[GroundingItem]] = {}
    for item in items:
        by_image.setdefault(item.image_key, []).append(item)

    keys = sorted(by_image)
    rng = random.Random(seed)
    shuffled = list(keys)
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_fraction))
    val_keys = set(shuffled[:n_val])
    train_keys = set(shuffled[n_val:])

    if not train_keys or not val_keys:
        raise GroundingDataError(
            f"split produced an empty partition: {len(train_keys)} train / "
            f"{len(val_keys)} val images from {len(keys)} total"
        )

    # Emit in sorted key order: the seed chooses WHICH image goes where, not
    # the order of the returned lists. Same rule as the router's group split.
    train = [i for k in keys if k in train_keys for i in by_image[k]]
    val = [i for k in keys if k in val_keys for i in by_image[k]]
    return train, val


def assert_image_disjoint(
    train: Sequence[GroundingItem], val: Sequence[GroundingItem]
) -> None:
    """Hard assertion that no image key crosses the partition boundary."""
    overlap = {i.image_key for i in train} & {i.image_key for i in val}
    if overlap:
        raise GroundingDataError(
            f"{len(overlap)} image(s) appear in both train and val, "
            f"e.g. {sorted(overlap)[:5]}"
        )


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    items: Sequence[GroundingItem],
    encoder,
    cache_dir: Path,
    *,
    progress_every: int = 200,
    text_batch_size: int = 256,
) -> dict[str, Any]:
    """Encode every image and phrase, caching both. Idempotent.

    A missing image and an undecodable image are counted SEPARATELY. The first
    means the corpus is incomplete (e.g. `Images_train.zip` not downloaded);
    the second means a file is corrupt. Merging them into one "failures" count
    would hide which problem you actually have.

    Nothing is zero-filled. A zero patch tensor teaches the head to predict
    boxes for blank input.
    """
    from PIL import Image

    cache_dir = Path(cache_dir)
    (cache_dir / "text").mkdir(parents=True, exist_ok=True)

    unique_images = sorted({i.image_key for i in items})

    # One pass to map image key -> source path. An earlier version searched
    # `items` inside the image loop, which is O(images x items): with 20,262
    # images and 36,313 items that is ~7e8 comparisons before any encoding.
    # It also looked for a `source_path` attribute the dataclass did not have,
    # so every image would have counted as a failure and the cache would have
    # stayed empty.
    key_to_source: dict[str, Path] = {}
    for item in items:
        if item.source_path is not None and item.image_key not in key_to_source:
            key_to_source[item.image_key] = item.source_path

    hits = misses = missing_source = undecodable = 0
    t0 = time.time()

    for n, key in enumerate(unique_images, start=1):
        out_path = image_cache_path(cache_dir, key)
        if out_path.exists():
            hits += 1
            continue

        src_path = key_to_source.get(key)
        if src_path is None:
            missing_source += 1
            continue

        try:
            image = Image.open(src_path).convert("RGB")
            encoded = encoder.encode_image(image)
            image.close()
        except Exception:  # noqa: BLE001
            undecodable += 1
            continue

        np.savez_compressed(
            out_path,
            patches=encoded.patch_tokens.astype(np.float16),
            cls=encoded.cls.astype(np.float16),
        )
        misses += 1
        if progress_every and n % progress_every == 0:
            rate = n / max(1e-6, time.time() - t0)
            print(f"    images {n}/{len(unique_images)}  {rate:.1f}/s")

    # Completion line for the image phase. The progress print above fires on
    # multiples of `progress_every`, so with 15,699 images the LAST line was
    # "15600/15699" and the final 99 images finished silently. A run was
    # interrupted at that point because it looked stalled. It was not.
    image_seconds = time.time() - t0
    # Every image must appear in this line. An earlier version printed only
    # cached / encoded / undecodable, so 6 items with no source path vanished
    # from the summary entirely -- and missing_source is precisely the signal
    # the handoff tells the operator to check ("non-zero means the extraction
    # path is wrong"). A counter nobody prints is not a diagnostic.
    print(
        f"    images {len(unique_images)}/{len(unique_images)}  "
        f"done in {image_seconds:.1f}s  "
        f"({hits} cached, {misses} encoded, "
        f"{missing_source} missing source, {undecodable} undecodable)"
    )
    if missing_source:
        print(
            f"    [!] {missing_source} image(s) had no source path. Those "
            f"items cannot be encoded and will be dropped at attach time."
        )

    # -- text --------------------------------------------------------------
    #
    # MEASURED FAILURE this replaces. The first real run appeared to stop at
    # "images 15600/15699". It had not stopped. All 15,699 images were on disk
    # 11 minutes BEFORE the run was interrupted; the loop that was actually
    # running was this one, and it had two defects:
    #
    #   * it encoded ONE phrase per forward pass. 11,589 phrases took ~660 s
    #     (~57 ms each). Batching collapses that to a handful of passes over
    #     the same 24,439 distinct strings.
    #   * it printed NOTHING. Eleven minutes of correct work with no output
    #     is indistinguishable from a hang, so the run was killed while it
    #     was working. The image loop printed progress; this one did not.
    #
    # Both are fixed here. The per-phrase cache FORMAT is unchanged, so the
    # 11,589 vectors already written stay valid -- a rerun skips them.
    phrases = sorted({i.phrase for i in items})
    text_hits = text_misses = text_failed = 0

    pending: list[tuple[str, Path]] = []
    for phrase in phrases:
        out_path = text_cache_path(cache_dir, phrase)
        if out_path.exists():
            text_hits += 1
            continue
        pending.append((phrase, out_path))

    text_started = time.time()
    if pending:
        print(
            f"    text: {len(pending)} phrase(s) to encode "
            f"in batches of {text_batch_size}"
        )

    since_print = 0
    for start in range(0, len(pending), text_batch_size):
        chunk = pending[start:start + text_batch_size]
        try:
            vectors = encoder.encode_text([phrase for phrase, _ in chunk])
        except Exception:  # noqa: BLE001
            # A whole chunk that will not encode is counted, not zero-filled.
            # Zero vectors would teach the head on blank text.
            text_failed += len(chunk)
            continue

        for (_phrase, out_path), vec in zip(chunk, vectors):
            np.save(out_path, vec.astype(np.float16))
            text_misses += 1

        since_print += len(chunk)
        if progress_every and since_print >= progress_every:
            since_print = 0
            done = start + len(chunk)
            elapsed = time.time() - text_started
            rate = done / max(1e-6, elapsed)
            remaining = (len(pending) - done) / rate if rate > 0 else 0.0
            print(
                f"    phrases {done}/{len(pending)}  {rate:.0f}/s  "
                f"eta {remaining / 60:.1f} min"
            )

    text_seconds = time.time() - text_started
    if pending:
        print(
            f"    text done in {text_seconds:.1f}s  "
            f"({text_misses} written, {text_failed} failed)"
        )

    return {
        "images_total": len(unique_images),
        "image_cache_hits": hits,
        "image_cache_misses": misses,
        "images_missing_source": missing_source,
        "images_undecodable": undecodable,
        "image_seconds": round(image_seconds, 1),
        "phrases_total": len(phrases),
        "text_cache_hits": text_hits,
        "text_cache_misses": text_misses,
        "text_failed": text_failed,
        "text_seconds": round(text_seconds, 1),
        "seconds": round(time.time() - t0, 1),
        "cache_dir": str(cache_dir),
    }


def attach_cached(
    items: Sequence[GroundingItem], cache_dir: Path, *, verbose: bool = True
) -> list[GroundingItem]:
    """Point each item at its cached patches and load its text embedding.

    Items whose cache entry is missing are DROPPED, never zero-filled. A zero
    patch tensor would train the head on blank input and the loss would look
    like it was converging.
    """
    kept: list[GroundingItem] = []
    dropped = 0
    corrupt = 0
    for item in items:
        p = image_cache_path(cache_dir, item.image_key)
        if not p.exists():
            dropped += 1
            continue
        t = text_cache_path(cache_dir, item.phrase)
        if not t.exists():
            dropped += 1
            continue
        # A vector file can exist and still be unreadable if a write was
        # interrupted. np.load raises on a truncated array, and an uncaught
        # raise here would abort the whole attach rather than skipping one
        # item -- so a killed extraction run would look like a corrupt cache.
        try:
            item.text = np.load(t).astype(np.float32)
        except Exception:  # noqa: BLE001
            corrupt += 1
            continue
        item.patches_path = p
        kept.append(item)

    if verbose:
        if dropped:
            print(f"  dropped {dropped} item(s) with missing cache entries")
        if corrupt:
            print(
                f"  dropped {corrupt} item(s) with unreadable text vectors "
                f"(interrupted write); re-run extraction to rebuild them"
            )
    return kept


__all__ = [
    "CACHE_VERSION",
    "GroundingDataError",
    "GroundingItem",
    "build_items",
    "split_by_image",
    "assert_image_disjoint",
    "extract_features",
    "attach_cached",
    "cache_dir_for",
    "image_cache_path",
    "text_cache_path",
]