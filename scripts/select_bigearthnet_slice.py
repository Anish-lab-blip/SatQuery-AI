"""Select the Phase 12 BigEarthNet fusion slice — metadata only, no imagery.

Phase 12 needs 28,000 optical-SAR fusion samples drawn from BigEarthNet v2
(reBEN) with a leakage boundary that is honest about how the corpus is actually
partitioned. Two decisions were made by the owner and are implemented here
verbatim; this script does not re-open them:

1. **Label policy — single label.** A patch with exactly one CLC-19 label is
   eligible; a multi-label patch is SKIPPED and counted, never reduced and never
   mapped to class 0. The machinery is not reimplemented here: it is
   `training.fusion.extract.LABEL_POLICIES["skip_ambiguous"]`, imported, so the
   fusion feature cache and this manifest answer the multi-label question the
   same way by construction.

2. **Scene key — T2, the spatial block.** Not the MGRS tile. Measured below: the
   tile key leaves 52 of 54 tiles straddling partitions, which is why it cannot
   be the key. See `training.data.bigearthnet_blocks` for the full argument.

What this script does NOT do: it never reads, downloads or extracts a single
band file. The 117.7 GB of imagery is not on disk and the entire selection is
derived from the two shipped metadata parquet files.

Outputs (all under `artifacts/phase12_selection/`, nowhere else):

    selection_manifest_seed00.jsonl ... seed19.jsonl   one per recorded seed
    feasibility_report.json                            per-partition, per-seed

Usage:

    python scripts/select_bigearthnet_slice.py
    python scripts/select_bigearthnet_slice.py --seeds 0 1 2
    python scripts/select_bigearthnet_slice.py --no-write     # verify only

Exit codes:
    0  every seed reached every target and every check passed
    1  a check failed, or a seed could not reach a target
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config  # noqa: E402
from core.errors import SatQueryError  # noqa: E402
from evaluation.manifests import DatasetManifest, SampleRecord  # noqa: E402
from training.data.bigearthnet import CLC19_CLASSES  # noqa: E402
from training.data.bigearthnet_blocks import (  # noqa: E402
    BigEarthNetBlockError,
    Cell,
    block_cell_counts,
    count_blocks_by_split,
    parse_acquisition_date,
    parse_ground_cell,
    parse_s1_tile,
    parse_tile,
    reconstruct_blocks,
)
from training.fusion.extract import LABEL_POLICIES, policy_name  # noqa: E402

# The disjointness helper is imported rather than re-implemented: the whole
# point of the tile-key comparison is that the SAME function that rejects the
# official split on the tile key accepts it on the block key. A private name,
# deliberately -- it is the script-local helper `prepare_bigearthnet` uses.
from scripts.prepare_bigearthnet import _official_split_scene_disjoint  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "bigearthnet_v2"
OUT_DIR = REPO_ROOT / "artifacts" / "phase12_selection"

#: The full corpus is the union of these two files. The first EXCLUDES the
#: snow/cloud/shadow patches, so reading it alone silently drops 69,450 rows.
METADATA_FILES: tuple[str, ...] = (
    "metadata.parquet",
    "metadata_for_patches_with_snow_cloud_or_shadow.parquet",
)

#: Columns the selection needs. Read only those a given file actually has --
#: the companion file's schema is not guaranteed to match the primary one.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "patch_id",
    "labels",
    "split",
    "country",
    "s1_name",
)
#: Read when present, purely to DEMONSTRATE it carries no tile. Never a scene
#: source: `parse_tile` must fail on it for every row.
PROBE_COLUMNS: tuple[str, ...] = ("s2v1_name",)

#: The dataset's split vocabulary, verbatim from the parquet.
OFFICIAL_SPLITS: tuple[str, ...] = ("train", "validation", "test")

#: `SampleRecord.split` only accepts train|val|test. This mirrors
#: `training.data.bigearthnet._extract_split`, which does the same mapping.
MANIFEST_SPLIT: dict[str, str] = {"train": "train", "validation": "val", "test": "test"}

#: Exactly what Phase 12 asked for. Not scaled, not "close enough".
TARGETS: dict[str, int] = {"train": 20_000, "validation": 4_000, "test": 4_000}

#: The recorded seed set. Every one of these must reach every target; a seed
#: that cannot is reported loudly rather than dropped from the sweep.
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(20))

#: `LABEL_POLICIES` is keyed by name; the policy object itself is what runs.
POLICY_KEY = "skip_ambiguous"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetadataRow:
    """One patch's metadata. Imagery is deliberately absent -- see module docs."""

    patch_id: str
    labels: tuple[str, ...]
    official_split: str
    country: str
    s1_name: str


def _read_file(path: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Read one parquet with only the columns it actually declares.

    Returns (rows, columns_read, columns_absent).
    """
    schema = pq.ParquetFile(path).schema_arrow
    available = list(schema.names)
    wanted = list(REQUIRED_COLUMNS) + [c for c in PROBE_COLUMNS if c in available]
    missing_required = [c for c in REQUIRED_COLUMNS if c not in available]
    if missing_required:
        raise SatQueryError(
            f"{path.name} is missing required column(s) {missing_required}; "
            f"present: {available}"
        )
    columns = [c for c in wanted if c in available]
    absent = [c for c in wanted if c not in available]
    table = pq.read_table(path, columns=columns)
    return table.to_pylist(), columns, absent


def load_metadata(data_dir: Path) -> tuple[list[MetadataRow], dict[str, Any]]:
    """Read BOTH parquets, concatenate, and cross-check every derivable fact.

    The checks are the deliverable as much as the rows are: tile derivability,
    the patch_id/s1_name tile agreement, and the `s2v1_name` probe are all
    measured here rather than assumed from the brief.
    """
    rows: list[MetadataRow] = []
    per_file: dict[str, Any] = {}
    probe_parsed = 0
    probe_total = 0
    tile_mismatches: list[str] = []

    for name in METADATA_FILES:
        path = data_dir / name
        if not path.exists():
            raise SatQueryError(f"metadata parquet not found: {path}")
        payload, columns, absent = _read_file(path)
        per_file[name] = {
            "rows": len(payload),
            "columns_read": columns,
            "columns_absent": absent,
        }
        for raw in payload:
            patch_id = str(raw["patch_id"])
            s1_name = str(raw["s1_name"])
            labels = tuple(str(x) for x in (raw["labels"] or ()))

            # `parse_tile` raises when the tile is absent, so this line asserts
            # derivability for every row by construction.
            tile = parse_tile(patch_id)
            s1_tile = parse_s1_tile(s1_name)
            if tile != s1_tile:
                tile_mismatches.append(patch_id)

            if "s2v1_name" in raw and raw["s2v1_name"] is not None:
                probe_total += 1
                try:
                    parse_tile(str(raw["s2v1_name"]))
                except BigEarthNetBlockError:
                    pass
                else:
                    probe_parsed += 1

            rows.append(
                MetadataRow(
                    patch_id=patch_id,
                    labels=labels,
                    official_split=str(raw["split"]),
                    country=str(raw["country"]),
                    s1_name=s1_name,
                )
            )

    if tile_mismatches:
        raise SatQueryError(
            f"{len(tile_mismatches)} patch(es) disagree on their tile between "
            f"patch_id and s1_name, e.g. {tile_mismatches[:3]}; the tile is not "
            f"a reliable property of the row and the block key cannot be trusted"
        )

    facts = {
        "files": per_file,
        "rows_total": len(rows),
        "tile_derivable_rows": len(rows),
        "tile_mismatch_patch_id_vs_s1_name": len(tile_mismatches),
        "s2v1_name_rows": probe_total,
        "s2v1_name_rows_with_a_tile_token": probe_parsed,
    }
    return rows, facts


# ---------------------------------------------------------------------------
# Corpus derivation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EligiblePatch:
    """One single-label patch, placed in its block."""

    patch_id: str
    tile: str
    cell: Cell
    scene_id: str
    official_split: str
    label_index: int
    country: str
    acquisition_date: str


def build_corpus(
    rows: Sequence[MetadataRow],
) -> tuple[dict[Cell, str], dict[Cell, str], dict[str, Any]]:
    """Derive the cell->split map, the cell->scene map and the corpus facts."""
    split_vocabulary = sorted({row.official_split for row in rows})
    if tuple(split_vocabulary) != tuple(sorted(OFFICIAL_SPLITS)):
        raise SatQueryError(
            f"unexpected split vocabulary {split_vocabulary}; expected exactly "
            f"{sorted(OFFICIAL_SPLITS)}"
        )

    cell_splits: dict[Cell, str] = {}
    impure: list[Cell] = []
    for row in rows:
        cell: Cell = (parse_tile(row.patch_id),) + parse_ground_cell(row.patch_id)
        previous = cell_splits.get(cell)
        if previous is None:
            cell_splits[cell] = row.official_split
        elif previous != row.official_split:
            impure.append(cell)

    cell_to_scene = reconstruct_blocks(
        (tile, row, col, split) for (tile, row, col), split in cell_splits.items()
    )

    patches_per_cell = Counter()
    for row in rows:
        patches_per_cell[(parse_tile(row.patch_id),) + parse_ground_cell(row.patch_id)] += 1

    block_sizes = block_cell_counts(cell_to_scene)

    facts = {
        "tiles": len({tile for tile, _, _ in cell_splits}),
        "ground_cells": len(cell_splits),
        "impure_cells": len(impure),
        "patches_per_cell_histogram": {
            str(k): v for k, v in sorted(Counter(patches_per_cell.values()).items())
        },
        "blocks_total": len(block_sizes),
        "blocks_by_split": count_blocks_by_split(cell_to_scene, cell_splits),
        "block_size_cells_min": min(block_sizes.values()),
        "block_size_cells_max": max(block_sizes.values()),
        "patches_by_official_split": dict(
            sorted(Counter(row.official_split for row in rows).items())
        ),
    }
    return cell_splits, cell_to_scene, facts


def apply_label_policy(
    rows: Sequence[MetadataRow],
    cell_to_scene: Mapping[Cell, str],
) -> tuple[dict[str, list[EligiblePatch]], dict[str, Any]]:
    """Apply the SELECTED single-label policy through the shared machinery.

    `LABEL_POLICIES["skip_ambiguous"]` returns an index for an exactly-one-label
    patch and `None` otherwise. `None` means SKIP: the patch is counted and
    dropped. Nothing here reduces a multi-label patch to one of its labels, and
    nothing maps one to class 0.
    """
    policy = LABEL_POLICIES[POLICY_KEY]

    blocks_by_split: dict[str, list[EligiblePatch]] = {
        split: [] for split in OFFICIAL_SPLITS
    }
    skipped = Counter()
    eligible = Counter()

    for row in rows:
        index = policy(row.labels)
        if index is None:
            skipped[row.official_split] += 1
            continue
        cell: Cell = (parse_tile(row.patch_id),) + parse_ground_cell(row.patch_id)
        blocks_by_split[row.official_split].append(
            EligiblePatch(
                patch_id=row.patch_id,
                tile=cell[0],
                cell=cell,
                scene_id=cell_to_scene[cell],
                official_split=row.official_split,
                label_index=int(index),
                country=row.country,
                acquisition_date=parse_acquisition_date(row.patch_id),
            )
        )
        eligible[row.official_split] += 1

    facts = {
        "label_policy": policy_name(policy),
        "eligible_patches_by_split": {
            split: eligible[split] for split in OFFICIAL_SPLITS
        },
        "skipped_multi_or_zero_label_by_split": {
            split: skipped[split] for split in OFFICIAL_SPLITS
        },
    }
    return blocks_by_split, facts


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _order_blocks(scene_ids: Iterable[str], seed: int) -> list[str]:
    """A deterministic, seed-dependent permutation of scene ids.

    A SHA-256 of ``"<seed>:<scene_id>"`` is used as the sort key rather than
    `random.Random(seed).shuffle`. Both are deterministic, but the digest is
    deterministic across Python versions and implementations too: the selection
    is reproducible from the recorded seed alone, which is the property the
    seed sweep exists to demonstrate. The seed itself is never fed to a global
    RNG, so this script cannot perturb any other caller's random stream.
    """
    return sorted(
        scene_ids, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest()
    )


def select_for_seed(
    blocks_by_split: Mapping[str, Sequence[EligiblePatch]],
    seed: int,
    targets: Mapping[str, int],
) -> tuple[dict[str, list[EligiblePatch]], dict[str, Any]]:
    """Pick whole blocks until each partition hits its exact target.

    Blocks are taken WHOLE, so no block can straddle two partitions. The target
    is hit exactly: when the next whole block would overshoot, the block is
    TRIMMED -- its patches are taken in sorted `patch_id` order and the
    remainder is left unselected. Trimming is deterministic and is recorded
    (blocks trimmed, patches dropped) so the report says so rather than leaving
    a reader to infer it.
    """
    selection: dict[str, list[EligiblePatch]] = {}
    report: dict[str, Any] = {}

    for split in OFFICIAL_SPLITS:
        target = int(targets[split])
        by_scene: dict[str, list[EligiblePatch]] = defaultdict(list)
        for patch in blocks_by_split[split]:
            by_scene[patch.scene_id].append(patch)

        ordered = _order_blocks(by_scene, seed)
        chosen: list[EligiblePatch] = []
        trimmed: list[dict[str, Any]] = []
        remaining = target

        for scene_id in ordered:
            if remaining <= 0:
                break
            patches = sorted(by_scene[scene_id], key=lambda p: p.patch_id)
            take = patches[:remaining]
            chosen.extend(take)
            remaining -= len(take)
            if len(take) < len(patches):
                trimmed.append(
                    {
                        "scene_id": scene_id,
                        "taken": len(take),
                        "available": len(patches),
                        "dropped": len(patches) - len(take),
                    }
                )

        selection[split] = chosen
        report[split] = {
            "target": target,
            "selected": len(chosen),
            # Blocks that hold at least one eligible patch, which is fewer than
            # the corpus's block count: a block whose patches are all
            # multi-label cannot contribute to a single-label selection.
            "blocks_with_eligible_patches": len(by_scene),
            "blocks_used": len({p.scene_id for p in chosen}),
            "trimmed_blocks": trimmed,
            "patches_dropped_by_trim": sum(t["dropped"] for t in trimmed),
            "target_reached": len(chosen) == target,
        }

    return selection, report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def per_class_support(
    patches: Sequence[EligiblePatch],
) -> tuple[dict[str, int], list[str], float | None]:
    """Support for all 19 classes, in `CLC19_CLASSES` order, plus the zeros.

    The imbalance ratio is ``max support / min non-zero support`` over the 19
    classes. It is reported alongside `zero_support_classes` rather than folded
    into a single number: a ratio computed over non-zero classes only would
    hide the classes that are missing entirely, and those are the ones that
    matter.
    """
    counts = Counter(patch.label_index for patch in patches)
    support = {name: counts[i] for i, name in enumerate(CLC19_CLASSES)}
    zeros = [name for name in CLC19_CLASSES if support[name] == 0]
    non_zero = [v for v in support.values() if v > 0]
    ratio = round(max(non_zero) / min(non_zero), 4) if non_zero else None
    return support, zeros, ratio


def scene_split_map(records: Sequence[SampleRecord]) -> dict[str, set[str]]:
    """scene_key -> the set of partitions it appears in."""
    out: dict[str, set[str]] = defaultdict(set)
    for record in records:
        out[record.scene_key].add(record.split)
    return out


def to_records(patches: Sequence[EligiblePatch]) -> list[SampleRecord]:
    """Turn a selection into manifest records.

    Field mapping, stated explicitly because two of these choices are
    deliberate:

    * `scene_id` is the T2 block id; `source_scene` is the MGRS tile. The tile
      is provenance, NOT the leakage boundary -- `SampleRecord.scene_key`
      prefers `scene_id`, which is the point.
    * `acquisition_date` is written to the first-class
      `SampleRecord.acquisition_date` field rather than into `extra`, because
      that field exists for exactly this value. `extra` carries the fields the
      schema has no home for: the ground cell and the block id.
    * `sha256`, `path` and `geographic_hash` stay `None`. No imagery is on disk
      and no bounds are available, and inventing either would be a false
      provenance claim.
    """
    return [
        SampleRecord(
            dataset_id="bigearthnet",
            sample_id=patch.patch_id,
            split=MANIFEST_SPLIT[patch.official_split],  # type: ignore[arg-type]
            scene_id=patch.scene_id,
            source_scene=patch.tile,
            acquisition_date=patch.acquisition_date,
            sensor="sentinel-2",
            extra={
                "ground_cell": [patch.cell[1], patch.cell[2]],
                "block_id": patch.scene_id,
                "country": patch.country,
                "official_split": patch.official_split,
                "label_index": patch.label_index,
                "label": CLC19_CLASSES[patch.label_index],
            },
        )
        for patch in patches
    ]


def verify_seed(
    selection: Mapping[str, Sequence[EligiblePatch]],
    records: Sequence[SampleRecord],
    targets: Mapping[str, int],
) -> dict[str, Any]:
    """Every check the acceptance criteria name, measured on the selection."""
    by_scene = scene_split_map(records)
    straddling = {
        scene: sorted(splits) for scene, splits in by_scene.items() if len(splits) > 1
    }

    # (b) the official partition was honoured: every record's split is the
    # mapping of the patch's own official split, and no block spans two.
    # Matched by sample_id rather than by position, so a future reordering of
    # `to_records` cannot silently turn this check into a comparison of the
    # wrong pairs.
    official_by_sample = {
        patch.patch_id: patch.official_split
        for split in OFFICIAL_SPLITS
        for patch in selection[split]
    }
    mismatched = [
        record.sample_id
        for record in records
        if record.split != MANIFEST_SPLIT[official_by_sample[record.sample_id]]
    ]
    block_splits: dict[str, set[str]] = defaultdict(set)
    for split, patches in selection.items():
        for patch in patches:
            block_splits[patch.scene_id].add(patch.official_split)
    impure_blocks = sorted(s for s, v in block_splits.items() if len(v) > 1)

    # (c) targets.
    counts = {
        split: len(selection[split]) for split in OFFICIAL_SPLITS
    }
    targets_reached = all(counts[s] == int(targets[s]) for s in OFFICIAL_SPLITS)

    # (d) per-class support.
    support: dict[str, Any] = {}
    for split in OFFICIAL_SPLITS:
        counts_by_class, zeros, ratio = per_class_support(selection[split])
        support[split] = {
            "per_class_support": counts_by_class,
            "zero_support_classes": zeros,
            "classes_with_support": len(CLC19_CLASSES) - len(zeros),
            "imbalance_ratio": ratio,
        }

    return {
        "scene_disjoint": not straddling,
        "straddling_scene_count": len(straddling),
        "straddling_scenes_sample": dict(list(straddling.items())[:5]),
        "official_split_mismatches": len(mismatched),
        "impure_blocks": impure_blocks,
        "scenes_used": len(by_scene),
        "selected_by_split": counts,
        "targets_reached": targets_reached,
        "per_partition": support,
    }


def _flatten(selection: Mapping[str, Sequence[EligiblePatch]]) -> list[EligiblePatch]:
    """Selection order used by `to_records`: same split order, same rows."""
    return [patch for split in OFFICIAL_SPLITS for patch in selection[split]]


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def hr(title: str = "") -> None:
    print("-" * 72)
    if title:
        print(title)
        print("-" * 72)


def print_corpus(facts: Mapping[str, Any], label_facts: Mapping[str, Any]) -> None:
    hr("CORPUS (metadata only — no imagery read)")
    print(f"  rows total              : {facts['rows_total']:,}")
    for name, info in facts["files"].items():
        print(f"    {name:<52} {info['rows']:>9,} rows")
        if info["columns_absent"]:
            print(f"      columns absent        : {info['columns_absent']}")
    print(f"  tiles                   : {facts['tiles']:,}")
    print(f"  ground cells            : {facts['ground_cells']:,}")
    print(f"  impure cells            : {facts['impure_cells']:,}")
    print(
        f"  tile mismatch patch/s1  : "
        f"{facts['tile_mismatch_patch_id_vs_s1_name']:,}"
    )
    print(
        f"  s2v1_name with a tile   : {facts['s2v1_name_rows_with_a_tile_token']:,}"
        f"  (of {facts['s2v1_name_rows']:,} rows)"
    )
    print(f"  patches per cell        : {facts['patches_per_cell_histogram']}")
    print(f"  T2 blocks               : {facts['blocks_total']:,}")
    print(f"  T2 blocks by split      : {facts['blocks_by_split']}")
    print(
        f"  block size (cells)      : min {facts['block_size_cells_min']:,}  "
        f"max {facts['block_size_cells_max']:,}"
    )
    hr("LABEL POLICY")
    print(f"  policy                  : {label_facts['label_policy']}")
    print(
        f"  eligible by split       : "
        f"{label_facts['eligible_patches_by_split']}"
    )
    print(
        f"  skipped (not 1 label)   : "
        f"{label_facts['skipped_multi_or_zero_label_by_split']}"
    )


def print_seed(seed: int, report: Mapping[str, Any], verification: Mapping[str, Any],
               manifest_hash: str, path: Path | None) -> None:
    print(
        f"  seed {seed:>2}  "
        f"train {report['train']['selected']:,}/{report['train']['target']:,}  "
        f"val {report['validation']['selected']:,}/{report['validation']['target']:,}  "
        f"test {report['test']['selected']:,}/{report['test']['target']:,}  "
        f"| scenes {verification['scenes_used']:,}  "
        f"straddling {verification['straddling_scene_count']}  "
        f"| hash {manifest_hash[:16]}"
    )
    for split in OFFICIAL_SPLITS:
        part = report[split]
        support = verification["per_partition"][split]
        zeros = support["zero_support_classes"]
        print(
            f"          {split:<11} blocks {part['blocks_used']:>3}/"
            f"{part['blocks_with_eligible_patches']:<3} "
            f"trimmed {len(part['trimmed_blocks'])} "
            f"(dropped {part['patches_dropped_by_trim']}) "
            f"| classes {support['classes_with_support']}/19 "
            f"imbalance {support['imbalance_ratio']}"
        )
        print(f"          {'':<11} zero-support: {zeros if zeros else 'none'}")
    if path is not None:
        print(f"          {'':<11} -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select the Phase 12 BigEarthNet fusion slice (metadata only)"
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
        help="seeds to sweep; every one must reach every target",
    )
    parser.add_argument(
        "--no-write", action="store_true",
        help="run every check but write nothing",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    seeds = [int(s) for s in args.seeds]
    config = load_config()

    print("=" * 72)
    print("SATQUERY AI - PHASE 12 BIGEARTHNET SELECTION (T2 BLOCK, SINGLE LABEL)")
    print("=" * 72)
    print(f"data dir    : {data_dir}")
    print(f"out dir     : {out_dir}")
    print(f"scene key   : T2 spatial block (4-connected cells within (tile, split))")
    print(f"label policy: {POLICY_KEY}")
    print(f"seeds       : {seeds}")
    print(f"targets     : {TARGETS}")
    print(f"config hash : {config.hash}")
    print()

    rows, file_facts = load_metadata(data_dir)
    cell_splits, cell_to_scene, corpus_facts = build_corpus(rows)
    blocks_by_split, label_facts = apply_label_policy(rows, cell_to_scene)
    corpus_facts.update(file_facts)
    corpus_facts.update(label_facts)
    print_corpus(corpus_facts, label_facts)

    # -- feasibility, per seed ------------------------------------------
    hr("SELECTION")
    per_seed: list[dict[str, Any]] = []
    written: list[Path] = []
    all_targets = True
    all_disjoint = True

    for seed in seeds:
        selection, report = select_for_seed(blocks_by_split, seed, TARGETS)
        records = to_records(_flatten(selection))
        verification = verify_seed(selection, records, TARGETS)

        manifest = DatasetManifest.from_records(
            f"bigearthnet_reben_single_label_t2_seed{seed:02d}",
            records,
            notes={
                "seed": seed,
                "seeds_swept": seeds,
                "label_policy": label_facts["label_policy"],
                "label_policy_semantics": (
                    "a patch with exactly one CLC-19 label is eligible; a "
                    "multi-label patch is SKIPPED and counted, never reduced "
                    "and never mapped to class 0"
                ),
                "scene_key": "T2",
                "scene_key_definition": (
                    "4-connected components of the (row, col) grid computed "
                    "separately per (tile, official split); scene id "
                    "ben_<tile>:<tile-global ordinal>"
                ),
                "scene_key_purity": (
                    "blocks are partition-pure BY CONSTRUCTION (components are "
                    "computed inside each (tile, split)); the scene id carries "
                    "no split token, so the disjointness check below is a real "
                    "check on the selection rather than a tautology"
                ),
                "official_split_source": "metadata.parquet `split` column",
                "manifest_split_mapping": MANIFEST_SPLIT,
                "targets": TARGETS,
                "selection_order": "sha256('<seed>:<scene_id>') ascending",
                "trimming": (
                    "the final block of a partition is taken whole-block-first; "
                    "if it would overshoot, its patches are taken in sorted "
                    "patch_id order up to the target and the remainder is "
                    "unselected"
                ),
                "imagery": (
                    "none; the selection is derived from the two metadata "
                    "parquet files alone, no band file was read or downloaded"
                ),
                "config_hash": config.hash,
                "data_dir": str(data_dir),
                "metadata_files": list(METADATA_FILES),
                "field_mapping": {
                    "scene_id": "T2 block id (leakage boundary)",
                    "source_scene": "MGRS tile (provenance only)",
                    "acquisition_date": "SampleRecord.acquisition_date",
                    "extra": ["ground_cell", "block_id", "country",
                              "official_split", "label_index", "label"],
                },
            },
        )

        path: Path | None = None
        if not args.no_write:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = manifest.write(
                out_dir / f"selection_manifest_seed{seed:02d}.jsonl",
                overwrite=True,
            )
            written.append(path)

        print_seed(seed, report, verification, manifest.hash, path)

        all_targets = all_targets and verification["targets_reached"]
        all_disjoint = all_disjoint and verification["scene_disjoint"]
        per_seed.append(
            {
                "seed": seed,
                "manifest_name": manifest.name,
                "manifest_path": str(path) if path else None,
                "manifest_hash": manifest.hash,
                "partitions": report,
                "verification": verification,
            }
        )

    # -- the tile-key comparison ----------------------------------------
    #
    # The same helper that the preparation script uses to accept or reject the
    # official split, pointed at the TILE key. It must reject it: 52 of 54 tiles
    # appear in more than one partition. That rejection is what makes T2 a
    # genuinely different key rather than a restatement of the tile.
    hr("TILE KEY (T1) COMPARISON")
    tile_records = [
        SampleRecord(
            dataset_id="bigearthnet",
            sample_id=row.patch_id,
            split=MANIFEST_SPLIT[row.official_split],  # type: ignore[arg-type]
            scene_id=f"ben_{parse_tile(row.patch_id)}",
            source_scene=parse_tile(row.patch_id),
            sensor="sentinel-2",
        )
        for row in rows
    ]
    tile_disjoint, tile_offenders = _official_split_scene_disjoint(tile_records)
    print(f"  key                       : MGRS tile")
    print(f"  scene-disjoint            : {tile_disjoint}")
    print(f"  straddling tiles          : {len(tile_offenders)}")
    for scene, splits in list(tile_offenders.items())[:3]:
        print(f"    {scene} spans {splits}")
    print(
        "  -> the tile key CANNOT honour the official partition; the block key "
        "is not a restatement of it"
    )

    # -- observations ------------------------------------------------------
    #
    # The selection rule was specified (whole blocks, trim the last one), so the
    # consequences are reported rather than engineered away. The block sizes are
    # large relative to the partition budgets, which means a handful of blocks
    # supplies an entire partition and the class coverage is much thinner than
    # the eligible population would allow.
    coverage = {
        split: sorted(
            entry["verification"]["per_partition"][split]["classes_with_support"]
            for entry in per_seed
        )
        for split in OFFICIAL_SPLITS
    }
    blocks_used = {
        split: sorted(entry["partitions"][split]["blocks_used"] for entry in per_seed)
        for split in OFFICIAL_SPLITS
    }
    observations = [
        (
            "The whole-block rule means a partition is filled by very few, very "
            "large blocks. Blocks used per seed: "
            + "; ".join(
                f"{split} {blocks_used[split][0]}-{blocks_used[split][-1]}"
                f" of {per_seed[0]['partitions'][split]['blocks_with_eligible_patches']} "
                f"blocks holding eligible patches"
                for split in OFFICIAL_SPLITS
            )
            + "."
        ),
        (
            "Class coverage is consequently far thinner than the eligible "
            "population allows. Classes with non-zero support per seed: "
            + "; ".join(
                f"{split} {coverage[split][0]}-{coverage[split][-1]}/19"
                for split in OFFICIAL_SPLITS
            )
            + ". See `zero_support_classes_seen_across_seeds` for which classes."
        ),
        (
            "Some classes are unreachable under the single-label policy "
            "regardless of seed: validation has 3 classes with zero eligible "
            "patches corpus-wide and test has 2. Those cannot appear in any "
            "selection."
        ),
        (
            "This is a property of the specified rule, not a defect in it: "
            "whole-block selection maximises spatial separation and minimises "
            "the number of scenes, at the cost of class balance. A stratified "
            "or block-granularity-relaxed rule would trade the other way and "
            "would need to be specified separately."
        ),
    ]

    # -- verification summary -------------------------------------------
    hr("VERIFICATION")
    print(f"  (a) scene-disjoint across all seeds : {all_disjoint}")
    print(
        f"  (b) official partitions honoured    : "
        f"{all(v['verification']['official_split_mismatches'] == 0 and not v['verification']['impure_blocks'] for v in per_seed)}"
    )
    print(f"  (c) every seed reached every target : {all_targets}")
    worst_zero: dict[str, set[str]] = {split: set() for split in OFFICIAL_SPLITS}
    for entry in per_seed:
        for split in OFFICIAL_SPLITS:
            worst_zero[split].update(
                entry["verification"]["per_partition"][split]["zero_support_classes"]
            )
    for split in OFFICIAL_SPLITS:
        zeros = sorted(worst_zero[split])
        print(
            f"  (d) {split:<11} zero-support classes seen across seeds: "
            f"{len(zeros)}/19"
        )
        for name in zeros:
            print(f"        - {name}")
    print(f"  imagery downloaded                  : none")
    print(f"  config hash                         : {config.hash}")

    hr("OBSERVATIONS")
    for note in observations:
        print(f"  - {note}")

    if not all_targets:
        print()
        print("  [!] AT LEAST ONE SEED DID NOT REACH ITS TARGET — see per-seed rows above.")
    if not all_disjoint:
        print()
        print("  [!] AT LEAST ONE SEED IS NOT SCENE-DISJOINT — see per-seed rows above.")

    # -- write the report ------------------------------------------------
    report_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config.hash,
        "data_dir": str(data_dir),
        "metadata_files": list(METADATA_FILES),
        "imagery_read": False,
        "scene_key": {
            "name": "T2",
            "definition": (
                "4-connected components of the (row, col) patch grid, computed "
                "separately per (tile, official split) so no block spans a "
                "partition; scene id 'ben_<tile>:<tile-global ordinal>'"
            ),
            "purity": "by construction, not proved by the disjointness check",
        },
        "label_policy": {
            "name": label_facts["label_policy"],
            "selected_by": "owner decision; not substituted or optimised here",
            "semantics": (
                "exactly-one-label patches only; multi-label and zero-label "
                "patches are skipped and counted"
            ),
        },
        "seeds": seeds,
        "targets": TARGETS,
        "selection_order": "sha256('<seed>:<scene_id>') ascending",
        "corpus": corpus_facts,
        "tile_key_comparison": {
            "key": "MGRS tile",
            "scene_disjoint": tile_disjoint,
            "straddling_tiles": len(tile_offenders),
            "sample_offenders": dict(list(tile_offenders.items())[:5]),
        },
        "per_seed": per_seed,
        "all_seeds_reached_targets": all_targets,
        "all_seeds_scene_disjoint": all_disjoint,
        "zero_support_classes_seen_across_seeds": {
            split: sorted(worst_zero[split]) for split in OFFICIAL_SPLITS
        },
        "observations": observations,
    }

    if not args.no_write:
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "feasibility_report.json"
        report_path.write_text(
            json.dumps(report_payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        hr("WROTE")
        for path in written:
            print(f"  {path}")
        print(f"  {report_path}")
    else:
        print()
        print("NO-WRITE: every check ran, nothing was written")

    hr("=")
    if not (all_targets and all_disjoint):
        print("SELECTION FAILED")
        return 1
    print("SELECTION OK — every seed reached every target, scene-disjoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
