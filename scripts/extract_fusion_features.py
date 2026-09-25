"""Phase 12 entry point — extract the frozen CROMA feature cache.

A THIN CLI over `training/fusion/extract.py`, mirroring the Phase 8/9/12 split
between `scripts/train_grounding.py` / `scripts/train_change.py` /
`scripts/train_fusion.py` and their `training/` modules. Everything that computes
lives in the module; this file parses arguments, reports the environment, prints
the accounting, and returns an exit code.

This is the producer that `scripts/train_fusion.py` was waiting for: it turns
real paired BigEarthNet-S1+S2 patches into the `.npz` the trainer reads.

    # validate the corpus and the split; encode nothing, write nothing
    python scripts/extract_fusion_features.py \
        --optical-root /data/BigEarthNet-S2 --sar-root /data/BigEarthNet-S1 \
        --split train --out-cache artifacts/optical_sar/fusion_features/train.npz \
        --dry-run

    # a real extraction (needs CROMA_base.pt already on disk)
    python scripts/extract_fusion_features.py \
        --optical-root ... --sar-root ... --metadata metadata.parquet \
        --split train --out-cache .../train.npz --batch-size 8 --arm A \
        --croma-checkpoint ~/.cache/.../CROMA_base.pt

    # the leak check, before committing to an encoding pass
    python scripts/extract_fusion_features.py ... --split train \
        --out-cache .../train.npz --other-cache .../val.npz --dry-run

THE DRY RUN APPLIES THE LABEL POLICY
------------------------------------
`--dry-run` prints a PLAN that is produced by `plan_extraction` — the same
`select_samples` the real run calls — so "Planned: N" is the number of rows the
run will encode, not the number of candidates it started from. BigEarthNet v2.0
is MULTI-LABEL, so the difference is every ambiguous patch, and it is largest
exactly where a dry run is most useful. An ambiguous sample under
`require_single_label` fails in the dry run as it would mid-run.

EXIT CODES
----------
    0   extraction completed (or --dry-run validated a present corpus)
    2   a pre-flight refusal, before anything is encoded: the corpus, the
        metadata table or the other cache is missing or empty; a band file
        cannot be read; the split leaks; --out-cache cannot be written to; the
        existing cache was produced by a different run. NOT a crash.
    3   the run started and then failed: the extractor raised a typed error (a
        multi-label sample under `require_single_label` included, in --dry-run
        as well), or the destination failed underneath it (e.g. a
        PermissionError on write)

THIS SCRIPT NEVER DOWNLOADS. `croma.build_encoder` falls back to
`hf_hub_download` when no local checkpoint is given, so a real run requires
`--croma-checkpoint` pointing at a `CROMA_base.pt` already on disk. A missing
checkpoint is a deployment prerequisite and exits 2 with the path it wanted.

NO PERFORMANCE NUMBER IS A RESULT HERE. This script encodes features; it measures
nothing. The pre-registered 11.5 metric is computed nowhere in this repository
yet — see the module docstring of `training/fusion/train.py`.

The names below are re-exported at module scope on purpose, in the pattern
`tests/unit/test_fusion_training.py` pins for `scripts.train_fusion`. The
re-export is the contract.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.data.bigearthnet import BigEarthNetError, pair_patches  # noqa: E402
from training.fusion.extract import (  # noqa: E402
    DEFAULT_EXTRACT_BATCH_SIZE,
    LABEL_POLICIES,
    OFFICIAL_SPLITS,
    ExtractionError,
    ExtractionInputError,
    assert_scene_disjoint,
    build_extraction_encoder,
    extract_fusion_features,
    load_split_assignments,
    plan_extraction,
    verify_feature_cache,
)
from training.fusion.reben_adapter import (  # noqa: E402
    DEFAULT_TARGET_SHAPE,
    RESAMPLING_MODES,
    build_reben_samples,
)
from training.fusion.train import (  # noqa: E402
    ARM_CONTROL,
    ARMS,
    FusionTrainingError,
    apply_arm,
    environment_report,
    read_feature_cache,
    resolve_arm,
)

__all__ = [
    "extract_fusion_features",
    "verify_feature_cache",
    "load_split_assignments",
    "build_extraction_encoder",
    "build_reben_samples",
    "ExtractionError",
    "ExtractionInputError",
    "main",
]


def hr(title: str = "") -> None:
    print("-" * 70)
    if title:
        print(title)
        print("-" * 70)


def _missing_inputs(args) -> list[str]:
    """Every path the run needs that is not there, named exactly."""
    missing: list[str] = []
    if args.selection_manifest is not None:
        # reBEN v2 path: the manifest plus the two trees it points at.
        if not Path(args.selection_manifest).is_file():
            missing.append(
                f"selection manifest: {args.selection_manifest} (not a file)")
        if args.reben_root is None:
            missing.append(
                "reben root       : --reben-root is required with "
                "--selection-manifest")
        elif not Path(args.reben_root).is_dir():
            missing.append(f"reben root       : {args.reben_root} (not a directory)")
        if args.parquet_dir is not None and not Path(args.parquet_dir).is_dir():
            missing.append(f"parquet dir      : {args.parquet_dir} (not a directory)")
    else:
        if args.optical_root is None:
            missing.append(
                "optical root     : --optical-root is required unless "
                "--selection-manifest is given")
        elif not Path(args.optical_root).is_dir():
            missing.append(f"optical root     : {args.optical_root} (not a directory)")
        if args.sar_root is not None and not Path(args.sar_root).is_dir():
            missing.append(f"SAR root         : {args.sar_root} (not a directory)")
        if args.metadata is not None and not Path(args.metadata).is_file():
            missing.append(f"metadata table   : {args.metadata} (not a file)")
    if args.other_cache is not None and not Path(args.other_cache).is_file():
        missing.append(f"other cache      : {args.other_cache} (not a file)")
    return missing


def _unwritable_output(args) -> list[str]:
    """Why `--out-cache` cannot be written, named before anything is encoded.

    A destination problem discovered AFTER an hour of encoding is an hour lost,
    so the check runs with the input checks. The two deterministic failures are
    the target being a directory (a `FileExistsError`/`IsADirectoryError` at
    write time) and its parent being a regular file (`NotADirectoryError`).
    """
    problems: list[str] = []
    out = Path(args.out_cache)
    if out.is_dir():
        problems.append(f"out cache        : {args.out_cache} (is a directory)")
    parent = out.parent
    if parent.exists() and not parent.is_dir():
        problems.append(
            f"out cache parent : {parent} (exists and is not a directory)"
        )
    elif parent.is_dir() and not os.access(parent, os.W_OK):
        problems.append(f"out cache parent : {parent} (not writable)")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract the Phase 12 frozen CROMA feature cache"
    )
    ap.add_argument("--optical-root", default=None,
                    help="root of the Sentinel-2 (BigEarthNet-S2) tree; required "
                         "unless --selection-manifest is given")
    ap.add_argument("--sar-root", default=None,
                    help="root of the Sentinel-1 tree; omit when one tree holds "
                         "both modalities")
    ap.add_argument("--selection-manifest", default=None,
                    help="a Phase 12 selection manifest (.jsonl). When given, "
                         "samples are built by the reBEN v2 adapter instead of "
                         "pair_patches: labels/split come from the metadata "
                         "parquet, S1 is joined on its released `s1_name`, bands "
                         "are resampled onto a common grid, and the leakage "
                         "boundary is the manifest's block id. Required for a "
                         "reBEN v2 tree, which the v1 co-located pairing cannot "
                         "read")
    ap.add_argument("--reben-root", default=None,
                    help="directory holding BigEarthNet-S2 / BigEarthNet-S1; "
                         "used with --selection-manifest")
    ap.add_argument("--parquet-dir", default=None,
                    help="directory holding the two reBEN metadata tables; "
                         "defaults to the parent of --reben-root")
    ap.add_argument("--resampling", default="bilinear",
                    choices=sorted(RESAMPLING_MODES),
                    help="how native-resolution bands are resampled onto the "
                         "common grid (reBEN v2 stores 10/20/60 m inside one "
                         "patch); recorded in the run output")
    ap.add_argument("--metadata", default=None,
                    help="optional reBEN metadata table (.parquet/.csv/.tsv) "
                         "carrying the official patch_id -> split assignment")
    ap.add_argument("--split", required=True, choices=list(OFFICIAL_SPLITS),
                    help="the official partition to extract; every sample must "
                         "declare exactly this split")
    ap.add_argument("--out-cache", required=True,
                    help="destination .npz (a .json sidecar is written beside it)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the paired ids CONSIDERED, before the split filter "
                         "(with --selection-manifest: cap the WINDOW, applied "
                         "after the split filter)")
    ap.add_argument("--offset", type=int, default=0,
                    help="with --selection-manifest: skip this many records of "
                         "the requested split before --limit. Together they make "
                         "a bounded window, so a corpus too large to materialise "
                         "can be encoded in slices that append to one cache "
                         "through the resume path")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_EXTRACT_BATCH_SIZE)
    ap.add_argument("--arm", default=ARM_CONTROL, choices=sorted(ARMS),
                    help="normalisation arm A|B; drives the hash-exempt env "
                         "channel, never the config")
    ap.add_argument("--label-policy", default="require_single_label",
                    choices=sorted(LABEL_POLICIES),
                    help="how a label tuple becomes one index; neither choice "
                         "invents a class for a multi-label sample")
    ap.add_argument("--croma-checkpoint", default=None,
                    help="local CROMA_base.pt; REQUIRED for a real run (this "
                         "script never downloads)")
    ap.add_argument("--croma-vendor-dir", default=None,
                    help="directory holding the vendored use_croma.py")
    ap.add_argument("--other-cache", default=None,
                    help="the other split's cache; when given, the scene-level "
                         "leak check runs against it before encoding, and the "
                         "trainer's row-level (scene AND sample id) firewall "
                         "runs over the rows about to be written")
    ap.add_argument("--no-resume", action="store_true",
                    help="overwrite an existing cache instead of appending")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the corpus and the split; encode nothing, "
                         "write nothing")
    args = ap.parse_args()

    from core.config import load_config

    cfg = load_config(args.config)
    arm = resolve_arm(args.arm)
    # The arm is selected through the hash-exempt env channel, never by editing
    # configs/base.yaml: adding a key would move the frozen Config.hash off
    # 78f1e3700da15aa1. See training/fusion/train.py.
    apply_arm(arm, env=os.environ)
    policy = LABEL_POLICIES[args.label_policy]

    print("=" * 70)
    print("PHASE 12 — FROZEN CROMA FEATURE EXTRACTION")
    print("=" * 70)
    if args.selection_manifest is not None:
        print(f"corpus      : reBEN v2 (via --selection-manifest)")
        print(f"manifest    : {args.selection_manifest}")
        print(f"reben root  : {args.reben_root}")
        print(f"parquet dir : {args.parquet_dir or '(parent of reben root)'}")
        print(f"resampling  : {args.resampling} -> {DEFAULT_TARGET_SHAPE[0]}x"
              f"{DEFAULT_TARGET_SHAPE[1]}")
    else:
        print(f"corpus      : BigEarthNet v1 co-located (pair_patches)")
        print(f"optical root: {args.optical_root}")
        print(f"sar root    : {args.sar_root or '(same tree)'}")
        print(f"metadata    : {args.metadata or '(none: per-patch metadata only)'}")
    print(f"split       : {args.split}")
    print(f"out cache   : {args.out_cache}")
    print(f"limit       : {args.limit if args.limit is not None else 'none'}")
    if args.selection_manifest is not None:
        print(f"offset      : {args.offset}")
    print(f"batch size  : {args.batch_size}")
    print(f"arm         : {arm.name}  (use_8_bit={arm.use_8_bit})")
    print(f"label policy: {args.label_policy}")
    print(f"resume      : {not args.no_resume}")
    print(f"config hash : {cfg.hash}")
    print(f"dry run     : {args.dry_run}")
    print()

    hr("ENVIRONMENT")
    env = environment_report("cpu")
    for key in sorted(env):
        print(f"  {key:<22} {env[key]}")
    print()

    # -- input validation, before anything is read or built ------------------
    hr("INPUTS")
    missing = _missing_inputs(args)
    unwritable = _unwritable_output(args)
    if missing or unwritable:
        print("FAILED: the corpus or the destination is not usable.")
        for line in missing:
            print(f"  missing {line}")
        for line in unwritable:
            print(f"  cannot write {line}")
        print()
        print("  This producer reads local files only. It never downloads, and it")
        print("  will not guess at a layout: point --optical-root (and --sar-root)")
        print("  at an existing BigEarthNet tree, and --out-cache at a path whose")
        print("  parent directory exists and is writable.")
        return 2
    if args.selection_manifest is not None:
        print(f"  selection manifest : present")
        print(f"  reben root         : present")
    else:
        print(f"  optical root : present")
        if args.sar_root is not None:
            print(f"  SAR root     : present")
        if args.metadata is not None:
            print(f"  metadata     : present")
    print()

    if not args.dry_run:
        if args.croma_checkpoint is None:
            print("FAILED: --croma-checkpoint is required for a real run.")
            print()
            print("  CROMA_base.pt must already be on disk. This script never")
            print("  downloads, so it will not fetch the checkpoint for you;")
            print("  pass --croma-checkpoint <path to CROMA_base.pt>.")
            return 2
        if not Path(args.croma_checkpoint).is_file():
            print(f"FAILED: --croma-checkpoint does not exist: "
                  f"{args.croma_checkpoint}")
            print()
            print("  Nothing was fetched. Place CROMA_base.pt on disk and retry.")
            return 2

    # -- pairing -------------------------------------------------------------
    hr("PAIRED SAMPLES")
    try:
        if args.selection_manifest is not None:
            # reBEN v2: the v1 co-located `pair_patches` cannot read this corpus
            # (separate trees with disjoint names, no per-patch metadata, mixed
            # native band resolutions). The adapter joins on `s1_name`, takes
            # labels/split from the parquet, resamples onto a common grid and
            # carries the manifest's block id as the leakage boundary.
            with open(args.selection_manifest, encoding="utf-8") as fh:
                header = json.loads(fh.readline()).get("_manifest", {})
            notes = header.get("notes", {})
            print(f"  manifest name    : {header.get('name')}")
            print(f"  manifest hash    : {header.get('hash')}")
            print(f"  manifest records : {header.get('count')}")
            print(f"  label policy     : {notes.get('label_policy')}")
            print(f"  scene key        : {notes.get('scene_key')} "
                  f"({header.get('scenes')} blocks)")
            samples = build_reben_samples(
                manifest_path=args.selection_manifest,
                reben_root=args.reben_root,
                parquet_dir=args.parquet_dir,
                split=args.split,
                limit=args.limit,
                offset=args.offset,
                resampling=args.resampling,
            )
        else:
            if args.offset:
                print("FAILED: --offset needs --selection-manifest (the v1 "
                      "co-located pairing has no defined record order).")
                return 2
            samples = pair_patches(
                args.optical_root,
                args.sar_root,
                limit=args.limit,
                spatial_shape=None,
            )
    except BigEarthNetError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        print()
        print("  Nothing was written and nothing was downloaded.")
        return 2
    except OSError as exc:
        # A band file that cannot be read (rasterio's RasterioIOError and
        # Pillow's UnidentifiedImageError are both OSError subclasses) is a
        # corpus problem, caught BEFORE anything is encoded: exit 2, not 1.
        print(f"FAILED: a band file could not be read: {type(exc).__name__}: {exc}")
        print()
        print("  Nothing was encoded, nothing was written and nothing was")
        print("  downloaded. Repair the corpus or point --optical-root at it.")
        return 2

    if not samples:
        print("FAILED: no paired patches were found.")
        return 2

    declared: list[str | None] = [sample.official_split for sample in samples]
    metadata_columns = None
    conflicts = 0
    if args.metadata is not None:
        try:
            assignments = load_split_assignments(args.metadata)
        except ExtractionInputError as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")
            return 2
        resolved: list[str | None] = []
        for sample, declared_split in zip(samples, declared):
            assigned = assignments.get(sample.patch_id)
            if assigned is None:
                resolved.append(declared_split)
                continue
            if declared_split is not None and declared_split != assigned:
                conflicts += 1
            resolved.append(assigned)
        declared = resolved
        metadata_columns = len(assignments)

    by_split = Counter(value or "unspecified" for value in declared)
    print(f"  paired ids       : {len(samples):,}")
    for name in (*OFFICIAL_SPLITS, "unspecified"):
        if by_split.get(name):
            print(f"    {name:<13}: {by_split[name]:,}")
    if args.metadata is not None:
        print(f"  metadata rows    : {metadata_columns:,} usable assignments")
        print(f"  split conflicts  : {conflicts:,} (metadata wins; per-patch "
              f"metadata disagreed)")

    optical_present = [int(sample.optical_mask.sum()) for sample in samples]
    sar_present = [int(sample.sar_mask.sum()) for sample in samples]
    print(f"  optical channels : mean {sum(optical_present) / len(samples):.2f} "
          f"of 12 present; {sum(1 for v in optical_present if v == 0):,} sample(s) "
          f"with none")
    print(f"  SAR channels     : mean {sum(sar_present) / len(samples):.2f} "
          f"of 2 present; {sum(1 for v in sar_present if v == 0):,} sample(s) "
          f"with none")
    print()

    # -- the split firewall --------------------------------------------------
    hr("SPLIT FIREWALL")
    kept = [i for i, value in enumerate(declared) if value == args.split]
    if not kept:
        print(f"FAILED: no sample declares the {args.split!r} split.")
        print()
        print(f"  The corpus declares {dict(by_split)}. Nothing will be encoded:")
        print(f"  a cache labelled {args.split!r} must actually hold that split.")
        return 2

    kept_samples = [samples[i] for i in kept]
    kept_splits = [declared[i] for i in kept]
    kept_scenes = {sample.scene_id for sample in kept_samples}
    print(f"  requested split  : {args.split}")
    print(f"  candidates       : {len(kept_samples):,} of {len(samples):,} paired "
          f"(BEFORE the label policy; see the plan below)")
    print(f"  scenes           : {len(kept_scenes):,}")
    print(f"  homogeneous      : True (every kept sample declares {args.split!r})")

    other_features = None
    if args.other_cache is not None:
        try:
            other_features, _ = read_feature_cache(args.other_cache)
        except FusionTrainingError as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")
            return 2
        try:
            assert_scene_disjoint(
                kept_scenes,
                {feature.scene_id for feature in other_features},
                left_name=args.split,
                right_name=f"the other cache ({args.other_cache})",
            )
        except ExtractionInputError as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")
            print()
            print("  A scene in both partitions means the split leaks. Rebuild the")
            print("  split by scene before encoding anything.")
            return 2
        print(f"  vs other cache   : scene-disjoint "
              f"({len(other_features):,} rows in {args.other_cache})")
        print(f"                     the trainer's own row-level firewall "
              f"(scene AND sample ids) runs again over the rows about to be "
              f"written, before the write")
    print()

    # -- the plan ------------------------------------------------------------
    #
    # The dry run makes the SAME decision the run makes, not a similar one: the
    # label policy is applied here, over the same samples, by the same
    # `select_samples` the extractor calls. Counting candidates before the
    # policy over-reports by every ambiguous sample, and BigEarthNet v2.0 is
    # multi-label — so that over-report is largest exactly where the dry run
    # matters most. An ambiguous sample under `require_single_label` fails HERE,
    # as it would mid-run, rather than being quietly planned around.
    existing_features = None
    if not args.no_resume and Path(args.out_cache).exists():
        try:
            existing_features, _ = read_feature_cache(args.out_cache)
        except FusionTrainingError as exc:
            print(f"FAILED: the existing cache could not be read: {exc}")
            return 2

    try:
        plan = plan_extraction(
            kept_samples,
            label_policy=policy,
            split=args.split,
            declared_splits=kept_splits,
            existing_features=existing_features,
            cache_path=args.out_cache,
        )
    except ExtractionInputError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 2
    except ExtractionError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 3

    hr("PLAN (label policy applied)")
    print(f"  label policy     : {plan.label_policy}")
    print(f"  candidates       : {plan.n_requested:,} (declared {args.split!r})")
    print(f"  will encode      : {plan.n_selected:,}")
    print(f"  skipped ambiguous: "
          f"{plan.n_skipped_by_policy - plan.n_skipped_unlabelled:,} "
          f"(multi-label: the policy records no single class)")
    print(f"  skipped unlabelled: {plan.n_skipped_unlabelled:,} (no labels at all)")
    print(f"  already cached   : {plan.n_already_cached:,} row(s); "
          f"{plan.n_skipped_existing:,} candidate(s) skipped as already present")
    print(f"  rows after write : {plan.n_would_write:,} "
          f"({plan.n_already_cached:,} existing + {plan.n_selected:,} new)")
    if args.no_resume and Path(args.out_cache).exists():
        print("  resume is OFF    : the existing cache is REPLACED, not merged; "
              "the row count above is what will remain")
    print()

    if args.dry_run:
        print("DRY RUN — corpus read, split and label policy applied, nothing "
              "encoded, nothing written.")
        print(f"  Planned: {plan.n_selected:,} sample(s) across "
              f"{plan.n_scenes:,} scene(s), batch size {args.batch_size}.")
        print(f"  Would write: {args.out_cache} (+ .json sidecar)")
        print(f"  Would encode: {plan.n_selected:,} sample(s) into "
              f"3 x 768-d GAP vectors each (optical, SAR, joint)")
        print(f"  (the real run reports the same counts: n_encoded="
              f"{plan.n_selected:,}, n_total={plan.n_would_write:,})")
        return 0

    # -- the encoder ---------------------------------------------------------
    hr("ENCODER")
    try:
        encoder = build_extraction_encoder(
            cfg,
            checkpoint_path=args.croma_checkpoint,
            vendor_dir=args.croma_vendor_dir,
            device="cpu",
            use_8_bit=arm.use_8_bit,
        )
    except ExtractionInputError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001 - CROMA's own loaders raise typed errors
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 3

    print(f"  encoder          : CROMA (raw; this pipeline owns the stretch)")
    print(f"  resolution       : {encoder.resolution}")
    print(f"  encoder dim      : {encoder.encoder_dim}")
    print(f"  normalize_input  : {encoder.normalize_input}  (must be False)")
    print()

    # -- extraction ----------------------------------------------------------
    hr("EXTRACTION")
    try:
        result = extract_fusion_features(
            kept_samples,
            encoder=encoder,
            label_policy=policy,
            split=args.split,
            declared_splits=kept_splits,
            batch_size=args.batch_size,
            arm=arm.name,
            cache_path=args.out_cache,
            other_partition=other_features,
            other_partition_name=f"the other cache ({args.other_cache})",
            config=cfg,
            resume=not args.no_resume,
        )
    except ExtractionInputError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 2
    except FusionTrainingError as exc:
        # A typed trainer error raised mid-run (e.g. a cached feature that is
        # not 2318 wide): the run started, so this is 3, not a pre-flight 2.
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 3
    except ExtractionError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 3
    except OSError as exc:
        # The destination failed underneath the run (e.g. PermissionError from
        # the atomic commit). The run started; nothing usable was committed.
        print(f"FAILED: the cache could not be written: {type(exc).__name__}: {exc}")
        return 3

    for key, value in result.to_dict().items():
        print(f"  {key:<18} {value}")

    # The plan and the run go through the same `select_samples`, so these are
    # equal by construction. Printed anyway, and checked: a plan that disagrees
    # with the run is worse than no plan, and this is the one place both numbers
    # exist at once.
    agrees = (
        result.n_encoded == plan.n_selected
        and result.n_total == plan.n_would_write
        and result.n_skipped_by_policy == plan.n_skipped_by_policy
        and result.n_skipped_unlabelled == plan.n_skipped_unlabelled
    )
    print(f"  {'plan vs run':<18} n_encoded {plan.n_selected} = "
          f"{result.n_encoded}, n_total {plan.n_would_write} = {result.n_total}, "
          f"skipped_by_policy {plan.n_skipped_by_policy} = "
          f"{result.n_skipped_by_policy}, skipped_unlabelled "
          f"{plan.n_skipped_unlabelled} = {result.n_skipped_unlabelled}")
    print(f"  {'agrees':<18} {agrees}")
    print()

    if not agrees:
        print("FAILED: the run's accounting disagrees with the plan it was "
              "built from.")
        print("  The cache on disk is what the run wrote; the plan is what it "
              "predicted. One of them is wrong and neither is trustworthy until "
              "that is resolved.")
        return 3

    if not result.written:
        print("  nothing new to encode; the existing cache was left untouched.")
        print()
        hr("=")
        return 0

    # -- verify what was just written ---------------------------------------
    hr("VERIFY")
    report = verify_feature_cache(args.out_cache, config=cfg)
    for item in report["checks"]:
        mark = "ok  " if item["passed"] else "FAIL"
        print(f"  [{mark}] {item['check']}: {item['detail']}")
    print()
    print(f"  all checks passed: {report['all_checks_passed']}")
    print()

    hr("PROVENANCE")
    for key in (
        "config_hash", "arm", "use_8_bit", "croma_image_resolution",
        "encoder_dim", "croma_checkpoint_sha256", "label_policy", "split",
    ):
        print(f"  {key:<24} {result.metadata.get(key)}")
    print()

    if not report["all_checks_passed"]:
        print("FAILED: the cache just written did not pass its own integrity "
              "check.")
        print(f"  failed: {report['failed_checks']}")
        return 3

    hr("WROTE")
    print(f"  {args.out_cache}")
    print(f"  {Path(args.out_cache).with_suffix('.json')}")
    print()
    print(json.dumps(
        {"arm": arm.name, "config_hash": cfg.hash, "n": result.n_total},
        sort_keys=True,
    ))
    hr("=")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
