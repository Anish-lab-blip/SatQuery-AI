"""Phase 12 entry point — train the optical-SAR fusion head from cached features.

A THIN CLI over `training/fusion/train.py`, mirroring the Phase 8/9 split between
`scripts/train_grounding.py` / `scripts/train_change.py` and their `training/`
modules. Everything that computes lives in the module; this file parses
arguments, reports the environment, prints the accounting, and returns an exit
code.

    # validate the caches and the split; train nothing, write nothing
    python scripts/train_fusion.py --train-cache train.npz --val-cache val.npz --dry-run

    # a real run on CPU (the fusion head is CPU-only)
    python scripts/train_fusion.py --train-cache train.npz --val-cache val.npz \
        --arm A --epochs 20

EXIT CODES
----------
    0   training completed (or --dry-run validated present caches)
    2   a cache is missing, empty, or the split leaks -- NOT a crash
    3   the run started and the trainer raised a typed error

NO PERFORMANCE NUMBER IS A RESULT HERE. This loop has never seen the real paired
BigEarthNet-S1+S2 corpus. Every run record it writes carries `result_status` and
`pre_registered_metric_computed: false`; the pre-registered 11.5 metric is not
computed. See the module docstring of `training/fusion/train.py`.

The names below are re-exported at module scope on purpose.
`tests/unit/test_fusion_training.py` asserts `train_fusion_head`,
`evaluate_fusion_head`, `prepare_fusion_batch` and `FusionTrainingError` are
reachable as `scripts.train_fusion.*`. The re-export is the contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.fusion.train import (  # noqa: E402  (path set above)
    ARM_CONTROL,
    ARMS,
    FusionTrainingError,
    environment_report,
    evaluate_fusion_head,
    prepare_fusion_batch,
    read_feature_cache,
    resolve_arm,
    train_fusion_head,
)

OUT_DIR = REPO_ROOT / "artifacts" / "optical_sar" / "fusion_head_v001"

__all__ = [
    "train_fusion_head",
    "evaluate_fusion_head",
    "prepare_fusion_batch",
    "read_feature_cache",
    "resolve_arm",
    "environment_report",
    "FusionTrainingError",
    "main",
]


def hr(title: str = "") -> None:
    print("-" * 70)
    if title:
        print(title)
        print("-" * 70)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train the Phase 12 optical-SAR fusion head"
    )
    ap.add_argument("--train-cache", required=True,
                    help="feature-cache .npz for the training split")
    ap.add_argument("--val-cache", required=True,
                    help="feature-cache .npz for the held-out validation split")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--arm", default=ARM_CONTROL, choices=sorted(ARMS),
                    help="normalisation arm A|B; drives the hash-exempt env "
                         "channel, never the config")
    ap.add_argument("--device", default="cpu",
                    help="cpu only; the fusion head refuses anything else")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap OPTIMIZER steps, not epochs")
    ap.add_argument("--resume-from", default=None,
                    help="a checkpoint_last.pt to resume from")
    ap.add_argument("--dry-run", action="store_true",
                    help="load the caches and check the split; train nothing")
    args = ap.parse_args()

    from core.config import load_config

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("project.seed", 42))
    out_dir = Path(args.output_dir) if args.output_dir else OUT_DIR
    arm = resolve_arm(args.arm)

    print("=" * 70)
    print("PHASE 12 — OPTICAL-SAR FUSION-HEAD TRAINING")
    print("=" * 70)
    print(f"train cache : {args.train_cache}")
    print(f"val cache   : {args.val_cache}")
    print(f"output dir  : {out_dir}")
    print(f"device      : {args.device}")
    print(f"seed        : {seed}")
    print(f"arm         : {arm.name}  (use_8_bit={arm.use_8_bit})")
    print(f"config hash : {cfg.hash}")
    print(f"dry run     : {args.dry_run}")
    print()

    hr("ENVIRONMENT")
    env = environment_report(args.device)
    for key in sorted(env):
        print(f"  {key:<22} {env[key]}")
    print()

    hr("FEATURE CACHES")
    try:
        train_features, train_meta = read_feature_cache(args.train_cache)
        val_features, val_meta = read_feature_cache(args.val_cache)
    except FusionTrainingError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        print()
        print("  Each cache is a .npz written by training.fusion.write_feature_cache,")
        print("  with a JSON sidecar carrying the sample/scene ids.")
        return 2

    # Provenance is REQUIRED, not optional. `extract.py` records why the extraction
    # side stopped skipping absent keys: a forgotten entry became a silent PASS
    # (`RESUME_PROVENANCE_OPTIONAL_FIELDS` exists for the one field a caller may
    # legitimately omit). The extractor writes all of these on every cache, so a
    # missing one means this cache did not come from `extract_fusion_features.py`
    # and the arm it was built under cannot be established. Refuse, never infer.
    PROVENANCE_KEYS = ("arm", "use_8_bit", "config_hash", "label_policy",
                       "croma_image_resolution", "encoder_dim", "cache_version")
    for side, side_meta in (("train", train_meta), ("val", val_meta)):
        absent = [key for key in PROVENANCE_KEYS if key not in side_meta]
        if absent:
            print(f"FAILED: the {side} cache records no {', '.join(absent)}. Its "
                  f"provenance cannot be established, so the arm it was built "
                  f"under is unknown. Refusing to train on it.")
            return 2

    # The arm is BAKED INTO the cached features, not applied at train time, so a
    # cache built under one arm cannot be trained under the other. Without this the
    # run record would name an arm the features did not come from -- and nothing
    # downstream compares them.
    cache_arm = train_meta["arm"]
    cache_use_8_bit = train_meta["use_8_bit"]
    if str(cache_arm).strip().upper() != arm.name:
        print(f"FAILED: the cache records arm {cache_arm!r}, but --arm {arm.name} was "
              f"requested. The arm is baked into the features; rebuilding the cache "
              f"under arm {arm.name} is the only way to train it.")
        return 2
    if bool(cache_use_8_bit) != arm.use_8_bit:
        print(f"FAILED: the cache records use_8_bit={cache_use_8_bit!r}, but arm "
              f"{arm.name} means use_8_bit={arm.use_8_bit!r}.")
        return 2

    # The two splits must come from one extraction run. `split` is absent from
    # PROVENANCE_KEYS on purpose: it is SUPPOSED to differ between train and val.
    for key in PROVENANCE_KEYS:
        if train_meta[key] != val_meta[key]:
            print(f"FAILED: the train and val caches disagree on {key!r}: "
                  f"{train_meta[key]!r} vs {val_meta[key]!r}.")
            return 2

    print(f"  train samples    : {len(train_features):,}")
    print(f"  train scenes     : {len({f.scene_id for f in train_features}):,}")
    print(f"  val samples      : {len(val_features):,}")
    print(f"  val scenes       : {len({f.scene_id for f in val_features}):,}")
    print(f"  cache version    : {train_meta.get('cache_version')}")
    print(f"  cache arm        : {cache_arm!r} (use_8_bit={cache_use_8_bit!r})")
    print()

    # The split firewall: refuse to train on a leak.
    from training.fusion.train import assert_split_disjoint

    try:
        assert_split_disjoint(train_features, val_features)
    except FusionTrainingError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        print()
        print("  The train and val caches overlap by scene or sample id. Rebuild")
        print("  the split by scene; a leak here makes every validation number a lie.")
        return 2
    print("  scene-disjoint   : True")
    print()

    if args.dry_run:
        print("DRY RUN — caches loaded, split verified disjoint, nothing trained.")
        print("  No head was built; no files were written.")
        return 0

    hr("TRAINING")
    resume_from = Path(args.resume_from) if args.resume_from else None
    if resume_from is not None and not resume_from.exists():
        print(f"  --resume-from does not exist: {resume_from}")
        return 2

    try:
        result = train_fusion_head(
            train_features,
            val_features,
            cfg,
            output_dir=out_dir,
            arm=arm.name,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=seed,
            max_steps=args.max_steps,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            feature_cache=args.train_cache,
            resume_from=resume_from,
            verbose=True,
        )
    except FusionTrainingError as exc:
        print()
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 3

    print()
    print(result.summary())

    hr("WROTE")
    print(f"  {out_dir / 'head.pt'}")
    print(f"  {out_dir / 'training_metadata.json'}")
    print(f"  {out_dir / 'checkpoint_last.pt'}")
    print(f"  {out_dir / 'run_record.json'}")
    print()
    print(json.dumps({"arm": arm.name, "config_hash": cfg.hash}, sort_keys=True))
    hr("=")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
