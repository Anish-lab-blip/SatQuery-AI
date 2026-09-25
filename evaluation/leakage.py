"""SatQuery AI — data leakage prevention.

These rules are non-negotiable and are enforced in code, not by convention:

  1. Split by SCENE, never by sample/tile.
  2. No scene may appear in more than one split.
  3. Duplicate content (same sha256) is rejected.
  4. Public test sets are immutable and off-limits to training code.
  5. Hidden data must never influence thresholds, prompts, or routing.

Every function here either returns a clean result or raises. None of them "warn and
continue" — a leakage warning that scrolls past is indistinguishable from no check.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from core.errors import LeakageError
from evaluation.manifests import DatasetManifest, SampleRecord

# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass
class LeakageReport:
    """Result of a leakage audit. `clean` is the only thing that matters."""

    clean: bool
    scenes_across_splits: dict[str, list[str]] = field(default_factory=dict)
    duplicate_hashes: dict[str, list[str]] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)
    scene_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"leakage audit: {'CLEAN' if self.clean else 'VIOLATION'}",
            f"  splits      : {self.split_counts}",
            f"  scenes      : {self.scene_counts}",
        ]
        if self.scenes_across_splits:
            lines.append(f"  scenes in >1 split : {len(self.scenes_across_splits)}")
        if self.duplicate_hashes:
            lines.append(f"  duplicate hashes   : {len(self.duplicate_hashes)}")
        for w in self.warnings:
            lines.append(f"  warning     : {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def assign_splits_by_scene(
    records: Sequence[SampleRecord],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> list[SampleRecord]:
    """Assign splits at the SCENE level, deterministically.

    This is the function that prevents the classic remote-sensing leakage bug: slicing
    a 10 000 px scene into 256 px tiles and then randomly splitting tiles. Adjacent
    tiles are near-duplicates, so a random tile split leaks the test set into training.

    Implementation notes:
      * Scenes are shuffled with a seeded RNG so the assignment is reproducible.
      * Scene *order* is sorted first, making the shuffle independent of input order.
      * A scene's records all receive the same split — enforced, not hoped for.
    """
    if not records:
        return []
    if not (0.0 < train_ratio < 1.0):
        raise LeakageError(f"train_ratio must be in (0,1), got {train_ratio}")
    if not (0.0 <= val_ratio < 1.0):
        raise LeakageError(f"val_ratio must be in [0,1), got {val_ratio}")
    if train_ratio + val_ratio >= 1.0:
        raise LeakageError(
            f"train_ratio + val_ratio must be < 1, got "
            f"{train_ratio} + {val_ratio}"
        )

    by_scene: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        by_scene[record.scene_key].append(record)

    # Shuffle a stable, sorted list of scene keys so the assignment is reproducible
    # and independent of the order records arrived in.
    scene_keys = sorted(by_scene)
    rng = random.Random(seed)
    shuffled = list(scene_keys)
    rng.shuffle(shuffled)

    n_scenes = len(shuffled)
    n_train = max(1, int(n_scenes * train_ratio))
    n_val = int(n_scenes * val_ratio)

    assignment: dict[str, str] = {}
    for i, key in enumerate(shuffled):
        if i < n_train:
            assignment[key] = "train"
        elif i < n_train + n_val:
            assignment[key] = "val"
        else:
            assignment[key] = "test"

    # Emit in SORTED scene order, not shuffled order. The seed controls *which* scene
    # goes to which split; it must not control the order of the returned list, or
    # downstream code taking records[:N] would silently depend on the seed.
    out: list[SampleRecord] = []
    for key in scene_keys:
        for record in by_scene[key]:
            record.split = assignment[key]  # type: ignore[assignment]
            out.append(record)

    return out


# ---------------------------------------------------------------------------
# Auditing
# ---------------------------------------------------------------------------


def audit_manifest(
    manifest: DatasetManifest,
    *,
    require_all_splits: bool = True,
) -> LeakageReport:
    """Full leakage audit. Raises LeakageError when a hard rule is broken."""
    report = LeakageReport(clean=True)
    report.split_counts = manifest.split_counts()
    report.scene_counts = {
        "total": len(manifest.scenes()),
    }

    # --- rule 2: no scene in more than one split --------------------------
    scene_splits: dict[str, set[str]] = defaultdict(set)
    for record in manifest:
        scene_splits[record.scene_key].add(record.split)

    offenders = {
        scene: sorted(splits)
        for scene, splits in scene_splits.items()
        if len(splits) > 1
    }
    if offenders:
        report.clean = False
        report.scenes_across_splits = offenders

    # --- rule 3: duplicate content ----------------------------------------
    by_hash: dict[str, list[str]] = defaultdict(list)
    for record in manifest:
        if record.sha256:
            by_hash[record.sha256].append(record.sample_id)

    dupes = {h: ids for h, ids in by_hash.items() if len(ids) > 1}
    if dupes:
        report.clean = False
        report.duplicate_hashes = dupes

    # --- advisory: is the split actually scene-disjoint? ------------------
    if require_all_splits:
        present = {s for s, n in report.split_counts.items() if n > 0}
        if present != {"train", "val", "test"}:
            report.warnings.append(
                f"not all splits populated: {sorted(present)} "
                f"(train/val/test expected before benchmark evaluation)"
            )

    # --- advisory: scene_key fell all the way back to sample_id -----------
    fallback = sum(
        1 for r in manifest
        if r.scene_key == r.sample_id and r.scene_id is None and r.geographic_hash is None
    )
    if fallback:
        report.warnings.append(
            f"{fallback} samples have no scene_id, geographic_hash or source_scene; "
            f"scene-level isolation is vacuous for them"
        )

    if not report.clean:
        raise LeakageError(
            "manifest failed leakage audit:\n" + report.summary(),
            context={
                "scenes_across_splits": len(offenders),
                "duplicate_hashes": len(dupes),
            },
        )

    return report


def assert_no_scene_overlap(
    train: Iterable[SampleRecord],
    test: Iterable[SampleRecord],
) -> None:
    """Hard assertion that two collections share no scene. Raises on overlap."""
    train_scenes = {r.scene_key for r in train}
    test_scenes = {r.scene_key for r in test}
    overlap = train_scenes & test_scenes
    if overlap:
        raise LeakageError(
            f"{len(overlap)} scene(s) appear in both train and test, "
            f"e.g. {sorted(overlap)[:5]}",
            context={"overlap_count": len(overlap)},
        )


# ---------------------------------------------------------------------------
# The public-test firewall
# ---------------------------------------------------------------------------

#: Directory that holds immutable public test material. Training code must never read
#: from this tree. Import-time and call-time guards both check it.
PUBLIC_TEST_DIRNAME = "public_test"


class PublicTestFirewall:
    """Blocks training-time reads of the public test tree.

    Two layers of defence:
      * path inspection — any read whose resolved path is under the public test root
        is refused while the firewall is armed.
      * manifest inspection — a manifest that contains test-split records may not be
        handed to a training loop.

    The firewall is armed by default in training scripts and disarmed only by the
    evaluation runner, which is a separate entry point.
    """

    def __init__(self, public_test_root: str | Path) -> None:
        self.root = Path(public_test_root).resolve()
        self._armed = True

    def arm(self) -> None:
        self._armed = True

    def disarm(self) -> None:
        self._armed = False

    @property
    def armed(self) -> bool:
        return self._armed

    def check_read(self, path: str | Path) -> None:
        """Raise if reading `path` would violate the firewall."""
        if not self._armed:
            return
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            return  # outside the test tree — allowed
        raise LeakageError(
            f"training-time read of the public test tree was blocked: {resolved}",
            context={"public_test_root": str(self.root)},
        )

    def check_manifest(self, manifest: DatasetManifest) -> None:
        """Raise if a manifest exposes test-split material to training."""
        if not self._armed:
            return
        test_records = manifest.by_split("test")
        if test_records:
            raise LeakageError(
                f"manifest '{manifest.name}' contains {len(test_records)} test-split "
                f"record(s); it must not be used for training",
                context={"test_records": len(test_records)},
            )


# ---------------------------------------------------------------------------
# Hidden-set discipline
# ---------------------------------------------------------------------------


def assert_no_hidden_access(config: object) -> None:
    """Guard the hidden-evaluation contract.

    The config must explicitly declare that hidden data is inaccessible. If someone
    flips that flag to True, evaluation stops here rather than silently tuning on
    hidden material.
    """
    getter = getattr(config, "get", None)
    if getter is None:
        raise LeakageError("configuration object does not support .get()")

    if getter("evaluation.hidden_data_access") is not False:
        raise LeakageError(
            "evaluation.hidden_data_access must be explicitly False; "
            "hidden annotations are never available during development"
        )

    if getter("evaluation.official_aggregate_weights") is not None:
        raise LeakageError(
            "evaluation.official_aggregate_weights must remain null; "
            "inventing an aggregate formula is prohibited"
        )


# ---------------------------------------------------------------------------
# Helpers used by dataset preparation scripts
# ---------------------------------------------------------------------------


def content_hash_file(path: str | Path) -> str:
    """SHA-256 of file content, streamed. Delegates to the raster helper."""
    from preprocessing.raster import file_sha256

    return file_sha256(path)


def deduplicate(records: Iterable[SampleRecord]) -> tuple[list[SampleRecord], list[str]]:
    """Drop duplicate-content records, keeping the first occurrence.

    Returns (kept, dropped_sample_ids). Duplicates are a leakage vector *and* a
    silent train/test contamination path, so they are removed, not tolerated.
    """
    seen: dict[str, str] = {}
    kept: list[SampleRecord] = []
    dropped: list[str] = []

    for record in records:
        key = record.sha256
        if key is None:
            kept.append(record)
            continue
        if key in seen:
            dropped.append(record.sample_id)
            continue
        seen[key] = record.sample_id
        kept.append(record)

    return kept, dropped


__all__ = [
    "PUBLIC_TEST_DIRNAME",
    "LeakageReport",
    "assign_splits_by_scene",
    "audit_manifest",
    "assert_no_scene_overlap",
    "PublicTestFirewall",
    "assert_no_hidden_access",
    "content_hash_file",
    "deduplicate",
]