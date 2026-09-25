"""Prepare BigEarthNet for VLM adaptation.

Produces three files and prints one report:

    artifacts/data/bigearthnet_manifest.jsonl   leakage-audited manifest
    artifacts/data/bigearthnet_pairs.jsonl      instruction pairs for training
    artifacts/data/bigearthnet_report.json      structural report

    python scripts/prepare_bigearthnet.py --root /path/to/BigEarthNet-S2
    python scripts/prepare_bigearthnet.py --root ... --limit 2000   # smoke test
    python scripts/prepare_bigearthnet.py --root ... --dry-run      # no writes

Run --limit 2000 FIRST. It takes seconds and proves the tree is readable, the
tile identity is right, and the split is clean. Discovering a layout problem
after an hour of walking 118 GB is the thing this script exists to prevent.

Exit codes:
    0  corpus is usable
    1  corpus has problems (printed; see `warnings_for`)
    2  the tree could not be read at all
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config  # noqa: E402
from core.errors import SatQueryError  # noqa: E402
from evaluation.leakage import (  # noqa: E402
    assign_splits_by_scene,
    audit_manifest,
    deduplicate,
)
from evaluation.manifests import DatasetManifest, SampleRecord  # noqa: E402
from training.data.bigearthnet import (  # noqa: E402
    BigEarthNetPatch,
    build_instruction_pairs,
    discover_patches,
    summarize,
    warnings_for,
)

OUT_DIR = REPO_ROOT / "artifacts" / "data"


def hr(title: str = "") -> None:
    print("-" * 68)
    if title:
        print(title)
        print("-" * 68)


def _official_split_scene_disjoint(
    records: list[SampleRecord],
) -> tuple[bool, dict[str, list[str]]]:
    """Check that the metadata's split keeps every tile in one partition.

    Returns (is_disjoint, offenders) where offenders maps a scene_id to the
    partitions it spans. A tile appearing in two partitions means the split is
    patch-level, which leaks neighbouring ground across the boundary.
    """
    scene_splits: dict[str, set[str]] = {}
    for record in records:
        if record.split is not None:
            scene_splits.setdefault(record.scene_id or record.sample_id, set()).add(
                record.split
            )

    offenders = {
        scene: sorted(splits)
        for scene, splits in scene_splits.items()
        if len(splits) > 1
    }
    return not offenders, offenders


def report(summary: dict, problems: list[str]) -> None:
    hr("CORPUS")
    print(f"  patches                 : {summary['patches']:,}")
    print(f"  tiles                   : {summary['tiles']:,}")
    print(
        f"  patches per tile        : "
        f"min {summary['patches_per_tile_min']}  "
        f"median {summary['patches_per_tile_median']}  "
        f"max {summary['patches_per_tile_max']}  "
        f"mean {summary['patches_per_tile_mean']}"
    )
    print(f"  mean labels per patch   : {summary['mean_labels_per_patch']}")
    print(f"  labels present          : {summary['labels_present']}/19")
    print(f"  official splits         : {summary['official_splits']}")

    hr("BANDS")
    for count, n in sorted(summary["band_count_distribution"].items()):
        print(f"  {count:>2} bands : {n:,} patches")

    hr("TOP LABELS")
    for label, n in summary["top_labels"]:
        print(f"  {n:>7,}  {label}")

    hr("PROBLEMS")
    if problems:
        for p in problems:
            print(f"  [!] {p}")
    else:
        print("  none")
    hr()


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare BigEarthNet for training")
    ap.add_argument("--root", required=True, help="BigEarthNet-S2 root directory")
    ap.add_argument("--modality", default="s2", choices=["s2", "s1"])
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N patches; use for a smoke run")
    ap.add_argument("--negative-ratio", type=float, default=1.0,
                    help="absent-class questions per present class")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="report only; write nothing")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    config = load_config()
    seed = args.seed if args.seed is not None else config.seed
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR

    print("=" * 68)
    print("SATQUERY AI - BIGEARTHNET PREPARATION")
    print("=" * 68)
    print(f"root        : {args.root}")
    print(f"modality    : {args.modality.upper()}")
    print(f"limit       : {args.limit or 'none (full corpus)'}")
    print(f"seed        : {seed}")
    print(f"config hash : {config.hash}")
    print()

    # -- discovery ------------------------------------------------------
    hr("DISCOVERY")
    try:
        patches = discover_patches(args.root, args.modality, limit=args.limit)
    except SatQueryError as exc:
        print(f"  FAILED: {exc}")
        hr("=")
        return 2

    print(f"  found {len(patches):,} patches")

    # Warnings split into two classes, and the distinction is the whole point:
    #
    #   PROVENANCE  how the split was obtained. The official split may be
    #               absent or incomplete; a clean scene-level split gets
    #               derived from it and the corpus is usable. Recorded in the
    #               manifest notes and the report. Never blocks a run.
    #
    #   DEFECT      a substantive problem with the data itself -- vacuous
    #               tiling, missing labels, sparse labels. These mean a
    #               training run would produce a model whose validation score
    #               means less than it appears to, so they block.
    #
    # An earlier version treated an incomplete official split as a defect. It
    # then derived a clean split, wrote an audited-clean manifest, and exited 1
    # anyway -- refusing to train on a corpus it had just repaired. Provenance
    # is not a defect.
    #
    # Separately, a --limit run cannot have a complete split by construction,
    # so that provenance warning is already informational and stays so.
    PROVENANCE_MARKERS = (
        "no official split",
        "official split is incomplete",
    )

    summary = summarize(patches)
    all_warnings = warnings_for(summary)

    problems = [
        w for w in all_warnings
        if not any(marker in w for marker in PROVENANCE_MARKERS)
    ]
    informational = [
        w for w in all_warnings
        if any(marker in w for marker in PROVENANCE_MARKERS)
    ]

    # -- manifest --------------------------------------------------------
    records = [
        SampleRecord(
            dataset_id="bigearthnet",
            sample_id=p.patch_id,
            # Seeded with the official split when one exists so the
            # disjointness check can inspect it; overwritten below either way.
            split=p.official_split or "train",  # type: ignore[arg-type]
            scene_id=p.scene_id,
            source_scene=p.tile,
            sensor="sentinel-2" if p.modality == "s2" else "sentinel-1",
            path=str(p.directory),
        )
        for p in patches
    ]

    deduped, dropped = deduplicate(records)
    if dropped:
        print(f"  dropped {len(dropped)} duplicate-content records")
        by_id = {p.patch_id: p for p in patches}
        patches = [by_id[r.sample_id] for r in deduped]
        records = deduped
        summary = summarize(patches)
        problems = warnings_for(summary)

    # BigEarthNet v2 ships a geographical split. Honour it when it is complete
    # AND scene-disjoint; otherwise derive one at scene level. Never mix the
    # two -- a manifest that uses the official split for some patches and a
    # derived split for others has no defensible leakage boundary at all.
    #
    # The disjointness check is not decorative. An earlier version adopted the
    # official split on the strength of the metadata alone. A release (or a
    # re-export) whose split is patch-level would then be adopted, printed as
    # "using the OFFICIAL split", and written out -- with the only defence
    # being the audit that runs afterwards. Metadata is a claim; the check is
    # the evidence.
    official = summary.get("official_splits", {})
    official_complete = {"train", "val", "test"} <= set(official)
    official_disjoint, offenders = _official_split_scene_disjoint(records)

    if official_complete and official_disjoint:
        print("  using the OFFICIAL BigEarthNet split from metadata")
        print(f"    verified scene-disjoint across {len({r.scene_id for r in records})} tiles")
        for record, patch in zip(records, patches):
            record.split = patch.official_split  # type: ignore[assignment]
        split_source = "official"
    elif official_complete and not official_disjoint:
        print("  OFFICIAL split is NOT scene-disjoint; refusing to use it")
        for scene, splits in list(offenders.items())[:5]:
            print(f"    {scene} spans {splits}")
        if len(offenders) > 5:
            print(f"    ... and {len(offenders) - 5} more tiles")
        print("  deriving a scene-level split instead")
        problems.append(
            f"the official split puts {len(offenders)} tile(s) in more than one "
            f"partition; a derived scene-level split was used instead"
        )
        records = assign_splits_by_scene(records, seed=seed)
        split_source = "derived_after_rejecting_leaky_official"
    else:
        print("  no complete official split; deriving one at scene level")
        records = assign_splits_by_scene(records, seed=seed)
        split_source = "derived"

    manifest = DatasetManifest.from_records(
        f"bigearthnet_{args.modality}",
        records,
        notes={
            "seed": seed,
            "config_hash": config.hash,
            "limit": args.limit,
            "root": str(args.root),
            "split_source": split_source,
        },
    )

    hr("MANIFEST")
    print(f"  records : {len(manifest):,}")
    print(f"  hash    : {manifest.hash[:16]}")
    print(f"  splits  : {manifest.split_counts()}")

    # -- leakage audit ---------------------------------------------------
    hr("LEAKAGE AUDIT")
    try:
        audit = audit_manifest(manifest, require_all_splits=False)
        print(f"  clean   : {audit.clean}")
        print(f"  scenes  : {audit.scene_counts}")
        for w in audit.warnings:
            # The audit's own warnings are about the WRITTEN manifest, so they
            # are substantive. An empty partition here is a real problem: the
            # manifest exists and cannot serve its purpose.
            print(f"  [!] {w}")
            if w not in problems:
                problems.append(w)
    except SatQueryError as exc:
        print(f"  FAILED: {exc}")
        problems.append("leakage audit failed")

    # -- instruction pairs -----------------------------------------------
    hr("INSTRUCTION PAIRS")
    n_pairs = 0
    pairs_by_split: dict[str, int] = {}
    for record, patch in zip(records, patches):
        pairs = build_instruction_pairs(
            patch, negative_ratio=args.negative_ratio, seed=seed
        )
        n_pairs += len(pairs)
        pairs_by_split[record.split] = pairs_by_split.get(record.split, 0) + len(pairs)
    print(f"  generated : {n_pairs:,} pairs")
    print(f"  by split  : {pairs_by_split}")
    print(f"  per patch : {n_pairs / max(1, len(patches)):.2f}")

    report(summary, problems)

    if informational:
        hr("INFORMATIONAL (split provenance)")
        for item in informational:
            print(f"  [i] {item}")
        print()
        print("  These describe how the split was obtained, not a defect in")
        print("  the data. The written manifest is audited for leakage below.")
        if args.limit:
            print("  A --limit run cannot have a complete split by construction;")
            print("  re-run without --limit to evaluate split completeness.")
        hr()

    # -- write -----------------------------------------------------------
    if args.dry_run:
        print("DRY RUN: nothing written")
        hr("=")
        return 1 if problems else 0

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest.write(
        out_dir / f"bigearthnet_{args.modality}_manifest.jsonl", overwrite=True
    )

    pairs_path = out_dir / f"bigearthnet_{args.modality}_pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as fh:
        for record, patch in zip(records, patches):
            for pair in build_instruction_pairs(
                patch, negative_ratio=args.negative_ratio, seed=seed
            ):
                fh.write(
                    json.dumps(
                        {
                            "patch_id": patch.patch_id,
                            "scene_id": patch.scene_id,
                            "split": record.split,
                            "image_dir": str(patch.directory),
                            "question": pair["question"],
                            "answer": pair["answer"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    report_path = out_dir / f"bigearthnet_{args.modality}_report.json"
    report_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "root": str(args.root),
                "modality": args.modality,
                "limit": args.limit,
                "seed": seed,
                "config_hash": config.hash,
                "manifest_hash": manifest.hash,
                "summary": summary,
                "problems": problems,
                "pairs": n_pairs,
                "pairs_by_split": pairs_by_split,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )

    hr("WROTE")
    print(f"  {manifest_path}")
    print(f"  {pairs_path}   ({n_pairs:,} lines)")
    print(f"  {report_path}")
    hr("=")

    if problems:
        print("CORPUS HAS PROBLEMS - see the PROBLEMS section above.")
        print("Training on this corpus will produce a model whose validation")
        print("score means less than it appears to.")
        return 1

    if args.limit:
        print("CORPUS OK (TRUNCATED). This was a --limit smoke run.")
        print()
        print("It proves the tree is readable, the tile identity is right and")
        print("no two tiles share a split. It does NOT prove the full corpus is")
        print("healthy -- split completeness was not evaluated.")
        print()
        print("Next: re-run without --limit for the real preparation.")
        return 0

    print("CORPUS OK - ready for LoRA adaptation.")
    print()
    print("Next: upload the three files above to Kaggle, then run")
    print("      scripts/train_vlm_lora.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())