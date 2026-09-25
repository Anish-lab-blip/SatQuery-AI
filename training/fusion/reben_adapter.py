#!/usr/bin/env python
"""reBEN v2 -> `PairedSample` adapter (Phase 12).

WHY THIS EXISTS
---------------
`pair_patches` in `training/data/bigearthnet.py` implements the BigEarthNet **v1**
co-located layout: one patch directory holding BOTH modalities, a per-patch
metadata JSON carrying the labels and the split, and bands that all share one
pixel size. The reBEN **v2** release that Phase 12 targets differs in all three
respects, so `pair_patches` cannot produce a single usable sample from it.
Measured against the shipped 28k selection and the real extracted tree:

  * **Separate trees, disjoint names.** The modalities live in different trees
    and the S1 directory name is NOT the S2 patch id
    (`S2B_MSIL2A_20170927T094019_N9999_R036_T35VLC_09_85` vs
    `S1A_IW_GRDH_1SDV_20170928T161150_35VLC_9_85`). The two name sets intersect
    in **0 of 28,000**. The released `s1_name` column of the metadata parquet is
    the join key, and nothing else is.
  * **No per-patch metadata.** The extracted tree holds band `.tif` files and
    nothing else, so `discover_patches(..., require_labels=True)` finds zero
    patches and the CLI exits 2. Labels and the split exist only in the parquet.
  * **Mixed native band resolutions.** Bands are stored at their native
    resolution -- 120x120 (10 m: B02/B03/B04/B08), 60x60 (20 m: B05/B06/B07/
    B8A/B11/B12) and 20x20 (60 m: B01/B09) -- inside the SAME patch. The loader
    requires one common size and raises on every real patch. All bands of a
    patch share the SAME geographic bounds, so resampling onto the 10 m grid is
    well defined and is what this module does.

This module is **ADDITIVE**. It does not modify `pair_patches`, `discover_patches`
or the extraction core; it produces `PairedSample`s and hands them to
`extract_fusion_features(samples=...)`, which already accepts them.

THE LEAKAGE BOUNDARY COMES FROM THE MANIFEST, NOT FROM A DIRECTORY NAME
-----------------------------------------------------------------------
`PairedSample.scene_id` defaults to `ben_<tile>`, which is too coarse here: 12 of
the 47 tiles in the shipped selection carry blocks in more than one split, so the
tile key would raise a spurious "the split leaks" abort and kill the run. The
manifest's `scene_id` (`ben_<tile>:<block ordinal>`) is passed through explicitly
via `PairedSample.scene_key`.

THE NO-SHIFT RULE IS PRESERVED
------------------------------
A canonical band with no file is written as zeros with its mask bit 0. It is
never skipped so that a later band slides into its slot.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

# The canonical channel orders and the band-token matcher are IMPORTED, never
# re-declared: `training/data/bigearthnet.py` is the single place where "some
# sensor's bands" becomes "CROMA's canonical channels". `_find_band_files` is
# private but importing it is deliberate -- a second copy of the token-matching
# rule could drift and silently mis-slot a band.
from training.data.bigearthnet import (  # noqa: PLC2701
    BigEarthNetError,
    CANONICAL_OPTICAL_BANDS,
    CANONICAL_SAR_BANDS,
    PairedSample,
    S1_BAND_TOKENS,
    S2_BAND_TOKENS,
    _find_band_files,
)

__all__ = [
    "RebenLayoutError",
    "RESAMPLING_MODES",
    "DEFAULT_TARGET_SHAPE",
    "S2_TREE",
    "S1_TREE",
    "METADATA_FILES",
    "load_selection_manifest",
    "load_metadata_index",
    "resolve_patch_dirs",
    "build_reben_samples",
]

S2_TREE = "BigEarthNet-S2"
S1_TREE = "BigEarthNet-S1"

#: Both released tables, in the order the preflight verifier reads them.
METADATA_FILES: tuple[str, ...] = (
    "metadata.parquet",
    "metadata_for_patches_with_snow_cloud_or_shadow.parquet",
)

RESAMPLING_MODES: tuple[str, ...] = ("bilinear", "nearest")

#: CROMA's input grid. `croma.image_resolution` is 120, and every band of a real
#: patch shares one geographic extent, so 120x120 is the 10 m native grid.
DEFAULT_TARGET_SHAPE: tuple[int, int] = (120, 120)

#: Accepted spellings of the split, kept in step with `extract.SPLIT_ALIASES`.
_SPLIT_ALIASES = {
    "training": "train",
    "valid": "val",
    "validation": "val",
    "testing": "test",
}


class RebenLayoutError(BigEarthNetError):
    """The reBEN v2 tree or its metadata table does not have the expected shape.

    A `BigEarthNetError` subclass on purpose: the extraction CLI already catches
    that type and exits 2 without encoding anything.
    """


# ---------------------------------------------------------------------------
# Manifest + metadata table
# ---------------------------------------------------------------------------


def load_selection_manifest(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a selection manifest (`.jsonl`): line 1 is the `_manifest` header.

    The header is returned too, because the hash, the label policy and the scene
    key are provenance that belongs in the run record, not in a comment.
    """
    src = Path(path)
    if not src.is_file():
        raise RebenLayoutError(f"selection manifest does not exist: {src}")
    lines = src.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RebenLayoutError(f"selection manifest is empty: {src}")
    try:
        header = json.loads(lines[0]).get("_manifest")
    except json.JSONDecodeError as exc:
        raise RebenLayoutError(f"selection manifest header is not JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise RebenLayoutError(
            f"selection manifest {src.name} has no '_manifest' header on line 1"
        )
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RebenLayoutError(f"manifest line {lineno} is not JSON: {exc}") from exc
    if not records:
        raise RebenLayoutError(f"selection manifest {src.name} has no records")
    return header, records


def load_metadata_index(parquet_dir: str | Path) -> dict[str, dict[str, Any]]:
    """`patch_id -> {s1_name, labels, split}` from both released parquet tables.

    This is the ONLY source of labels, of the split, and of the S1 join key: the
    extracted tree carries band files and no metadata at all.
    """
    base = Path(parquet_dir)
    index: dict[str, dict[str, Any]] = {}
    found = 0
    for name in METADATA_FILES:
        table = base / name
        if not table.is_file():
            continue
        found += 1
        try:
            import pyarrow.parquet as pq  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment issue
            raise RebenLayoutError(
                f"reading '{name}' needs pyarrow: {exc}"
            ) from exc
        columns = {"patch_id", "s1_name", "labels", "split"}
        try:
            data = pq.read_table(table).to_pydict()
        except Exception as exc:  # noqa: BLE001
            raise RebenLayoutError(f"could not read '{table}': {exc}") from exc
        missing = columns - set(data)
        if missing:
            raise RebenLayoutError(
                f"metadata table '{name}' is missing column(s) {sorted(missing)}; "
                f"it has {sorted(data)}"
            )
        for pid, s1n, labels, split in zip(
            data["patch_id"], data["s1_name"], data["labels"], data["split"]
        ):
            index[str(pid)] = {
                "s1_name": str(s1n),
                "labels": tuple(str(v) for v in (labels or ())),
                "split": _normalise_split(split),
            }
    if not found:
        raise RebenLayoutError(
            f"no metadata table under {base}; looked for {list(METADATA_FILES)}"
        )
    if not index:
        # A present-but-empty table is a defect. Returning `{}` would defer the
        # failure to a per-patch "not in either metadata table", once for every
        # patch in the run, and only after the whole pairing pass had started.
        raise RebenLayoutError(
            f"the metadata table(s) under {base} hold no rows, so no patch has "
            f"labels or a split"
        )
    return index


def _normalise_split(value: Any) -> str | None:
    text = str(value).strip().lower()
    text = _SPLIT_ALIASES.get(text, text)
    return text if text in {"train", "val", "test"} else None


# ---------------------------------------------------------------------------
# Directory resolution
# ---------------------------------------------------------------------------


def _granule_of_patch_id(patch_id: str) -> str:
    """`S2A_MSIL2A_<stamp>_N9999_R022_T33UUP_43_85` -> the granule directory."""
    return "_".join(patch_id.split("_")[:-2])


def _granule_of_s1_name(s1_name: str) -> str:
    """`S1A_IW_GRDH_1SDV_<stamp>_33UUP_68_74` -> the granule directory."""
    return "_".join(s1_name.split("_")[:-3])


def resolve_patch_dirs(
    reben_root: str | Path, patch_id: str, s1_name: str
) -> tuple[Path, Path]:
    """Return the (optical, SAR) patch directories for one manifest record.

    Raises naming the exact missing path: a missing directory means the
    extraction is incomplete, which is not something to paper over with a
    zero-filled modality.
    """
    base = Path(reben_root)
    optical = base / S2_TREE / _granule_of_patch_id(patch_id) / patch_id
    sar = base / S1_TREE / _granule_of_s1_name(s1_name) / s1_name
    if not optical.is_dir():
        raise RebenLayoutError(f"optical patch directory is missing: {optical}")
    if not sar.is_dir():
        raise RebenLayoutError(f"SAR patch directory is missing: {sar}")
    return optical, sar


# ---------------------------------------------------------------------------
# Band reading + resampling
# ---------------------------------------------------------------------------


def _load_resampled(path: Path, shape: tuple[int, int], resampling: str) -> np.ndarray:
    """Read one band as a 2-D float32 array on a `shape` grid.

    `rasterio` resamples through the band's OWN geotransform, which is why this
    is geographically correct without a hand-built transform: every band of a
    real patch shares one geographic extent, so the target grid is just that
    extent at a finer sampling.
    """
    try:
        import rasterio  # noqa: PLC0415
        from rasterio.enums import Resampling  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RebenLayoutError(
            f"reading band rasters needs rasterio: {exc}"
        ) from exc

    mode = {
        "bilinear": Resampling.bilinear,
        "nearest": Resampling.nearest,
    }[resampling]

    with rasterio.open(path) as src:
        band = src.read(1, out_shape=shape, resampling=mode)
    return np.asarray(band, dtype=np.float32)


BandReader = Callable[[Path, tuple[int, int], str], np.ndarray]


def _assemble(
    patch_dir: Path,
    tokens: Sequence[str],
    canonical: Sequence[str],
    shape: tuple[int, int],
    resampling: str,
    reader: BandReader,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Place files into canonical slots; never shift.

    Returns (stack, mask, present, missing, ignored).
    """
    files = _find_band_files(patch_dir, tuple(tokens))
    stack = np.zeros((len(canonical), shape[0], shape[1]), dtype=np.float32)
    mask = np.zeros((len(canonical),), dtype=np.float32)
    present: list[str] = []
    missing: list[str] = []
    for slot, token in enumerate(canonical):
        path = files.get(token)
        if path is None:
            missing.append(token)
            continue
        stack[slot] = reader(path, shape, resampling)
        mask[slot] = 1.0
        present.append(token)
    ignored = tuple(sorted(set(files) - set(canonical)))
    return stack, mask, tuple(present), tuple(missing), ignored


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


def build_reben_samples(
    *,
    manifest_path: str | Path,
    reben_root: str | Path,
    parquet_dir: str | Path | None = None,
    split: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    target_shape: tuple[int, int] = DEFAULT_TARGET_SHAPE,
    resampling: str = "bilinear",
    reader: BandReader | None = None,
) -> list[PairedSample]:
    """Build `PairedSample`s for a selection manifest against the real tree.

    Args:
        manifest_path: the `.jsonl` selection manifest.
        reben_root: the directory holding `BigEarthNet-S2` / `BigEarthNet-S1`.
        parquet_dir: where the two metadata tables live; defaults to the PARENT
            of `reben_root`, which is the shipped `data/bigearthnet_v2` layout.
        split: keep only this split. Applied BEFORE `limit`, so a bounded run
            yields samples from the partition it asked for.
        limit: cap on the number of samples returned.
        offset: skip this many records first, after sorting and after the split
            filter. Together with `limit` this gives a bounded WINDOW, which is
            what makes a chunked extraction possible: 28,000 materialised
            `PairedSample`s hold ~21 GiB of float32 arrays, more than this
            machine has, so the corpus must be encoded in slices that append to
            one cache through the pipeline's existing resume path.
        target_shape: the common (H, W) every band is resampled onto.
        resampling: one of `RESAMPLING_MODES`.
        reader: band reader, injectable for tests.

    Raises:
        RebenLayoutError: the manifest, the tables or a patch directory is
            unusable, or a record is missing a label / a split.
    """
    if resampling not in RESAMPLING_MODES:
        raise RebenLayoutError(
            f"unknown resampling mode {resampling!r}; expected one of "
            f"{list(RESAMPLING_MODES)}"
        )
    shape = (int(target_shape[0]), int(target_shape[1]))
    read = reader or _load_resampled

    _, records = load_selection_manifest(manifest_path)
    base = Path(reben_root)
    tables = Path(parquet_dir) if parquet_dir is not None else base.parent
    index = load_metadata_index(tables)

    wanted = [r for r in records if split is None or r.get("split") == split]
    wanted.sort(key=lambda r: str(r["sample_id"]))
    if offset:
        if offset < 0:
            raise RebenLayoutError(f"offset must not be negative: {offset}")
        wanted = wanted[offset:]
    if limit is not None:
        wanted = wanted[:limit]
    if not wanted:
        raise RebenLayoutError(
            f"no manifest record matches split {split!r} at offset {offset} "
            f"(of {len(records)} records)"
        )

    samples: list[PairedSample] = []
    for record in wanted:
        patch_id = str(record["sample_id"])
        meta = index.get(patch_id)
        if meta is None:
            raise RebenLayoutError(
                f"patch '{patch_id}' is not in either metadata table under {tables}"
            )
        labels = meta["labels"]
        if not labels:
            raise RebenLayoutError(f"patch '{patch_id}' has no labels")
        official_split = meta["split"] or _normalise_split(record.get("split"))
        if official_split is None:
            raise RebenLayoutError(f"patch '{patch_id}' has no usable split")

        optical_dir, sar_dir = resolve_patch_dirs(base, patch_id, meta["s1_name"])
        optical, optical_mask, o_present, o_missing, o_ignored = _assemble(
            optical_dir, S2_BAND_TOKENS, CANONICAL_OPTICAL_BANDS, shape, resampling, read
        )
        sar, sar_mask, s_present, s_missing, s_ignored = _assemble(
            sar_dir, S1_BAND_TOKENS, CANONICAL_SAR_BANDS, shape, resampling, read
        )

        samples.append(
            PairedSample(
                patch_id=patch_id,
                # The MGRS tile from the manifest (`source_scene`), NOT the
                # granule directory name the directory walk would produce.
                tile=str(record.get("source_scene") or ""),
                optical=optical,
                sar=sar,
                optical_mask=optical_mask,
                sar_mask=sar_mask,
                optical_present_bands=o_present,
                sar_present_bands=s_present,
                optical_missing_bands=o_missing,
                sar_missing_bands=s_missing,
                optical_ignored_bands=o_ignored,
                sar_ignored_bands=s_ignored,
                labels=labels,
                official_split=official_split,
                # The leakage boundary the selection actually guarantees.
                scene_key=str(record["scene_id"]),
            )
        )
    return samples
