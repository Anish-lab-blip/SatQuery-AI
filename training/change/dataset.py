"""SatQuery AI — LEVIR-CD dataset loader for change detection.

LEVIR-CD dataset structure (verified from official sources):

    LEVIR-CD/
    ├── train/
    │   ├── A/           # T1 images (time 1)
    │   ├── B/           # T2 images (time 2)
    │   └── label/       # Binary change masks (0=unchanged, 255=change)
    ├── val/
    │   ├── A/
    │   ├── B/
    │   └── label/
    └── test/
        ├── A/
        ├── B/
        └── label/

Standard splits (from official LEVIR-CD):
- train: 445 image pairs
- val: 64 image pairs
- test: 128 image pairs
Total: 637 pairs

Images: 1024x1024 PNG, 3-channel RGB
Labels: 0=unchanged, 255=change (binary masks)

References:
- https://github.com/justchenhao/LEVIR
- https://github.com/ChenHongruixuan/ChangeDetectionRepository
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from core.errors import SatQueryError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# LEVIR-CD scene counts live at the bottom of this file, annotated with the
# granularity distinction against `configs/base.yaml`. They were previously
# defined here as well, identically; two definitions of a public constant is a
# trap for the next reader even when the values agree.

# Image suffixes to look for
IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


class LEVIRCDError(Exception):
    """LEVIR-CD dataset error."""
    pass


@dataclass(frozen=True)
class LEVIRItem:
    """One LEVIR-CD sample (image pair + label)."""
    sample_id: str
    t1_path: str
    t2_path: str
    label_path: str
    image_key: str  # filename stem, used for grouping/splitting

    def __post_init__(self) -> None:
        # Verify files exist
        for p in (self.t1_path, self.t2_path, self.label_path):
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing file: {p}")


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------
#
# LEVIR-CD ships in two shapes in the wild, and they are NOT interchangeable.
# Both were established from real files, not from documentation.
#
# NESTED -- the official 1024px distribution and every fixture written before
#   2026-09-18:
#
#       <root>/<split>/{A,B,label}/<scene>.png
#
#   The split is the DIRECTORY. Each file is one whole scene, so the scene key
#   is the filename stem.
#
# FLAT -- `keykeylv/levir-cd-256`, the pre-tiled 256px dataset named by
#   `configs/base.yaml` (`change.levir_split: 7120/1024/2048` patches):
#
#       <root>/{A,B,label}/<split>_<scene>_<tile>.png
#       e.g. A/test_100_1.png, A/train_216_16.png
#
#   The split is a FILENAME PREFIX, there is no per-split directory, and each
#   scene is split into 16 tiles (4x4 of 256px, from a 1024px source). The scene
#   key must therefore DROP the trailing tile index -- otherwise every tile
#   becomes its own "scene" and the scene-disjoint guard becomes a no-op.
#
# That second point is the whole reason this distinction is written down. A
# loader that only knew NESTED returned zero items on FLAT (verified: 0 items,
# `discover_levir_dataset` -> `{}`), and a loader that treated each FLAT file as
# its own scene would pass `assert_image_disjoint` while leaking 15/16 of every
# scene across the split boundary.

LAYOUT_NESTED = "nested"
LAYOUT_FLAT = "flat"


def detect_levir_layout(root: str | Path) -> str | None:
    """Which of the two real LEVIR-CD shapes `root` has, or None for neither."""
    root = Path(root)
    for split in ("train", "val", "test"):
        split_dir = root / split
        if all((split_dir / sub).is_dir() for sub in ("A", "B", "label")):
            return LAYOUT_NESTED
    if all((root / sub).is_dir() for sub in ("A", "B", "label")):
        return LAYOUT_FLAT
    return None


def flat_stem_parts(stem: str) -> tuple[str, str] | None:
    """`test_100_1` -> `("test", "test_100")`; None when it does not parse.

    The scene key keeps the split prefix so a scene can never collide across
    splits, which also makes the key self-describing in a leak report.
    """
    parts = stem.split("_")
    if len(parts) < 3 or not parts[-1].isdigit():
        return None
    split = parts[0]
    if split not in ("train", "val", "test"):
        return None
    return split, "_".join(parts[:-1])


def _flat_scene_key(stem: str) -> str:
    """The scene a flat-layout tile belongs to.

    Tolerant on purpose: the split prefix is what `flat_stem_parts` validates,
    but a stem from `list/*.txt` may carry a different prefix spelling, and
    losing the scene grouping would silently make the leakage guard vacuous --
    the exact failure this whole section exists to prevent.
    """
    parsed = flat_stem_parts(stem)
    if parsed is not None:
        return parsed[1]
    parts = stem.split("_")
    return "_".join(parts[:-1]) if len(parts) >= 2 else stem


def read_split_list(root: str | Path, split: str) -> list[str] | None:
    """The dataset's OWN split declaration, `<root>/list/<split>.txt`.

    `keykeylv/levir-cd-256` ships `list/train.txt` (7120 lines), `list/val.txt`
    (1024) and `list/test.txt` (2048) -- which is exactly the frozen
    `change.levir_split: 7120/1024/2048` in `configs/base.yaml`. That is the
    dataset telling us its split, rather than us inferring it from a filename
    convention that merely happens to agree.

    Returns None when the file is absent, so callers can fall back rather than
    treat "no list" as "empty split".
    """
    path = Path(root) / "list" / f"{split}.txt"
    if not path.exists():
        return None
    names = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        name = line.strip()
        if not name:
            continue
        names.append(Path(name).stem if name.lower().endswith(".png") else name)
    return names


def verify_split_lists(root: str | Path) -> dict[str, dict[str, int]]:
    """Compare the shipped split lists against the filename-prefix split.

    The two MUST agree. They are independent derivations of the same fact, so a
    disagreement means either the prefix convention changed or the list is for a
    different revision -- and silently trusting either one would put tiles of one
    split into another. Reported, never auto-resolved.
    """
    root = Path(root)
    out: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        declared = read_split_list(root, split)
        if declared is None:
            continue
        derived = []
        a_dir = root / "A"
        if a_dir.is_dir():
            for path in a_dir.glob(f"{split}_*.png"):
                derived.append(path.stem)
        declared_set, derived_set = set(declared), set(derived)
        out[split] = {
            "declared": len(declared_set),
            "derived": len(derived_set),
            "agreement": len(declared_set & derived_set),
            "declared_only": len(declared_set - derived_set),
            "derived_only": len(derived_set - declared_set),
        }
    return out


def _find_image_files(dir_path: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    """Map filename stem -> Path for all images in a directory."""
    result = {}
    for suffix in suffixes:
        for path in dir_path.glob(f"*{suffix}"):
            if path.is_file():
                stem = path.stem
                if stem not in result:  # first match wins
                    result[stem] = path
    return result


def _resolve_triples(
    a_dir: Path, b_dir: Path, label_dir: Path
) -> list[tuple[str, Path, Path, Path]]:
    """Pair every A image with its B and label sibling.

    The sibling is matched on the SAME stem and the SAME suffix as the A file,
    not on a hardcoded `.png`. A layout that ships `.tif` used to discover zero
    pairs while looking perfectly healthy on disk.

    Iterates the stem->path map, never the directory Path: iterating the Path
    raised `AttributeError: 'WindowsPath' object has no attribute 'items'` on
    every real layout, so discovery could never have succeeded.
    """
    a_files = _find_image_files(a_dir, IMAGE_SUFFIXES)
    triples: list[tuple[str, Path, Path, Path]] = []
    for stem, t1_path in sorted(a_files.items()):
        suffix = t1_path.suffix
        t2_path = b_dir / f"{stem}{suffix}"
        label_path = label_dir / f"{stem}{suffix}"
        if not t2_path.exists():
            # Fall back to .png: LEVIR-CD labels are sometimes shipped as PNG
            # even when the imagery is not.
            t2_path = b_dir / f"{stem}.png"
            label_path = label_dir / f"{stem}.png"
        if t2_path.exists() and label_path.exists():
            triples.append((stem, t1_path, t2_path, label_path))
    return triples


def discover_levir_dataset(root: str | Path) -> dict[str, list[str]]:
    """Discover all valid samples in either real LEVIR-CD layout.

    Returns:
        Dict mapping split -> list of valid image stems (filename stems).
        Only includes samples where A, B, and label all exist.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"LEVIR-CD root not found: {root}")

    layout = detect_levir_layout(root)
    result: dict[str, list[str]] = {}

    if layout == LAYOUT_NESTED:
        for split in ("train", "val", "test"):
            split_dir = root / split
            if not all((split_dir / sub).is_dir() for sub in ("A", "B", "label")):
                continue
            triples = _resolve_triples(
                split_dir / "A", split_dir / "B", split_dir / "label"
            )
            if triples:
                print(f"  {split}: {len(triples)} valid pairs")
                result[split] = [stem for stem, _a, _b, _l in triples]

    elif layout == LAYOUT_FLAT:
        triples = _resolve_triples(root / "A", root / "B", root / "label")
        for stem, _a, _b, _l in triples:
            parsed = flat_stem_parts(stem)
            if parsed is None:
                continue
            split, _scene = parsed
            result.setdefault(split, []).append(stem)
        for split in sorted(result):
            print(f"  {split}: {len(result[split])} valid pairs")

    return {split: stems for split, stems in result.items() if stems}


def load_levir_dataset(
    root: str | Path,
    *,
    splits: tuple[str, ...] = ("train", "val", "test"),
    limit: int | None = None,
    seed: int = 42,
) -> list:
    """Load LEVIR-CD items from a dataset root in either real layout.

    Args:
        root: LEVIR-CD root. NESTED (`<root>/<split>/{A,B,label}/`) and FLAT
            (`<root>/{A,B,label}/<split>_<scene>_<tile>.png`) are both supported;
            the shape is detected, never assumed.
        splits: Which splits to load.
        limit: Optional cap on total items (after shuffling).
        seed: Random seed for shuffling.

    Returns:
        List of dicts with keys: t1_path, t2_path, label_path, split, sample_id,
        image_key, group_key. `image_key`/`group_key` carry the SCENE, which is
        what `split_by_image` and `assert_image_disjoint` group on.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"LEVIR-CD root not found: {root}")

    layout = detect_levir_layout(root)
    if layout is None:
        raise FileNotFoundError(
            f"no LEVIR-CD layout under {root}: expected either "
            f"<root>/<split>/{{A,B,label}}/ or <root>/{{A,B,label}}/"
        )

    # split -> list of (sample_id, scene_key, t1, t2, label)
    found: dict[str, list[tuple[str, str, Path, Path, Path]]] = {}

    if layout == LAYOUT_NESTED:
        for split in ("train", "val", "test"):
            split_dir = root / split
            if not all((split_dir / sub).is_dir() for sub in ("A", "B", "label")):
                continue
            triples = _resolve_triples(
                split_dir / "A", split_dir / "B", split_dir / "label"
            )
            for stem, t1, t2, label in triples:
                # Each file IS a scene here, so the scene key is the stem.
                found.setdefault(split, []).append(
                    (f"{split}_{stem}", stem, t1, t2, label)
                )
            if found.get(split):
                print(f"  {split}: {len(found[split])} valid pairs")

    else:  # LAYOUT_FLAT
        triples = _resolve_triples(root / "A", root / "B", root / "label")
        by_stem = {stem: (t1, t2, label) for stem, t1, t2, label in triples}

        for split in ("train", "val", "test"):
            declared = read_split_list(root, split)
            members: list[str] = []
            source = "filename prefix"
            missing = 0
            if declared is not None:
                # The dataset's own declaration wins over our inference. The two
                # agree on the shipped archive, but only one of them is the
                # dataset speaking about itself.
                source = f"list/{split}.txt"
                present = [s for s in declared if s in by_stem]
                missing = len(declared) - len(present)
                members = present
            else:
                for stem in by_stem:
                    parsed = flat_stem_parts(stem)
                    if parsed is not None and parsed[0] == split:
                        members.append(stem)

            for stem in members:
                t1, t2, label = by_stem[stem]
                scene_key = _flat_scene_key(stem)
                found.setdefault(split, []).append((stem, scene_key, t1, t2, label))

            if found.get(split):
                note = f"  [{missing} declared but absent]" if missing else ""
                print(f"  {split}: {len(found[split])} valid pairs from {source}{note}")

    items = []
    for split in splits:
        for sample_id, scene_key, t1, t2, label in found.get(split, []):
            items.append({
                "t1_path": str(t1),
                "t2_path": str(t2),
                "label_path": str(label),
                "split": split,
                "sample_id": sample_id,
                # The scene the sample belongs to. Without this, `split_by_image`
                # falls back to `sample_id`, every tile becomes its own scene, and
                # the scene-disjoint guard silently becomes vacuous.
                "image_key": scene_key,
                "group_key": scene_key,
                "layout": layout,
            })

    # Apply limit
    if limit is not None and len(items) > limit:
        random.Random(seed).shuffle(items)
        items = items[:limit]

    return items


# ---------------------------------------------------------------------------
# Splitting and leakage guards
# ---------------------------------------------------------------------------
#
# `scripts/train_change.py` imported `split_by_image` and `assert_image_disjoint`
# from this module, but only `training/grounding/dataset.py` ever defined them.
# That made `import scripts.train_change` fail outright, so no change-detection
# training could run. The grounding versions are typed to `GroundingItem` and
# raise `GroundingDataError`, so they are not directly reusable; these are the
# LEVIR equivalents, following the same rules.
#
# The rule that matters: split by SCENE, never by patch. LEVIR-CD ships
# neighbouring 256px crops of the same scene; a random patch split puts
# near-duplicate crops on both sides of the boundary and inflates validation
# scores without measuring generalisation.


def split_by_image(
    items: list,
    *,
    val_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[list, list]:
    """Split a list of LEVIR item dicts by scene, not by sample.

    Args:
        items: dicts as returned by `load_levir_dataset`, each carrying a
            `group_key` (falling back to `image_key`, then `sample_id`).
        val_fraction: fraction of distinct scenes held out for validation.
        seed: chooses WHICH scenes go where, never the order of the result.

    Returns:
        (train, val) with no scene present in both.

    Raises:
        LEVIRCDError: bad ratio, or a partition came out empty.
    """
    if not 0.0 < val_fraction < 1.0:
        raise LEVIRCDError(f"val_fraction must be in (0,1), got {val_fraction}")

    def key_of(item) -> str:
        if isinstance(item, dict):
            return str(
                item.get("group_key")
                or item.get("image_key")
                or item.get("sample_id")
            )
        return str(getattr(item, "image_key", getattr(item, "sample_id", "")))

    by_scene: dict[str, list] = {}
    for item in items:
        by_scene.setdefault(key_of(item), []).append(item)

    scenes = sorted(by_scene)
    shuffled = list(scenes)
    random.Random(seed).shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_fraction))
    val_keys = set(shuffled[:n_val])
    train_keys = set(shuffled[n_val:])

    if not train_keys or not val_keys:
        raise LEVIRCDError(
            f"split produced an empty partition: {len(train_keys)} train / "
            f"{len(val_keys)} val scenes from {len(scenes)} total"
        )

    # Emit in sorted scene order: the seed chooses WHICH scene is held out,
    # not the order of the returned lists. Same rule as the grounding and
    # router splits, so a re-run with the same seed is byte-identical.
    train = [i for k in scenes if k in train_keys for i in by_scene[k]]
    val = [i for k in scenes if k in val_keys for i in by_scene[k]]
    return train, val


def assert_image_disjoint(train: list, val: list) -> None:
    """Hard assertion that no scene crosses the partition boundary.

    Raises rather than warns: a leakage guard that only prints is a guard that
    gets ignored.
    """
    def keys_of(items):
        out = set()
        for item in items:
            if isinstance(item, dict):
                out.add(str(item.get("group_key") or item.get("image_key")
                           or item.get("sample_id")))
            else:
                out.add(str(getattr(item, "image_key",
                                    getattr(item, "sample_id", ""))))
        return out

    overlap = keys_of(train) & keys_of(val)
    if overlap:
        raise LEVIRCDError(
            f"{len(overlap)} scene(s) appear in both train and val, "
            f"e.g. {sorted(overlap)[:5]}"
        )


# Verified LEVIR-CD scene counts (train / val / test splits as shipped).
#
# Granularity note: this counts SCENE IMAGE PAIRS. `configs/base.yaml`
# (`change.levir_split`: 7120/1024/2048) and `docs/ARCHITECTURE_FREEZE.md`
# section 2.4 count 256px PATCHES after non-overlapping tiling. The two figures
# describe the same dataset at different granularities and are not in conflict;
# do not "fix" one to match the other.
LEVIR_SPLITS = {"train": 445, "val": 64, "test": 128}


__all__ = [
    "LAYOUT_FLAT",
    "LAYOUT_NESTED",
    "LEVIR_SPLITS",
    "LEVIRCDError",
    "LEVIRItem",
    "assert_image_disjoint",
    "detect_levir_layout",
    "discover_levir_dataset",
    "flat_stem_parts",
    "load_levir_dataset",
    "read_split_list",
    "split_by_image",
    "verify_split_lists",
]