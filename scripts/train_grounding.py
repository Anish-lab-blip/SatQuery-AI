"""Phase 8 entry point — build the feature cache, then train the grounding head.

Two separable stages, because they have wildly different costs:

    STAGE 1  EXTRACT   RemoteCLIP over every train image   ~20 ms/image on a T4
    STAGE 2  TRAIN     the 1.05M-param head on the cache   seconds per epoch

Splitting them means a training hyperparameter change does not re-encode 15,699
images, and a crashed training run does not lose the cache.

    # stage 1 only — measure the corpus before committing to training
    python scripts/train_grounding.py --data-root <root> --extract-only

    # both stages, real run
    python scripts/train_grounding.py --data-root <root> --checkpoint <ckpt.pt>

    # 2-batch forward+backward+checkpoint+reload, on CPU
    python scripts/train_grounding.py --data-root <root> --debug

Training protocol (project section 45): --seed, --output-dir, --resume,
--max-steps, --debug, --dry-run are all supported. --dry-run trains nothing.

THE BAR: the Phase 7 zero-shot baseline scored 0.0972 mean best IoU over 16,159
real eval records. This head must beat it by MIN_IMPROVEMENT_IOU to justify
existing. The comparison is printed whether or not it passes.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "artifacts" / "grounding" / "remoteclip_grounding_v001"


def hr(t: str = "") -> None:
    print("-" * 70)
    if t:
        print(t)
        print("-" * 70)


def environment_report(device: str) -> dict:
    facts: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "requested_device": device,
    }
    try:
        import torch

        facts["torch"] = torch.__version__
        facts["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            facts["gpu_name"] = props.name
            facts["gpu_total_vram_mb"] = round(props.total_memory / (1024 ** 2), 1)
            cap = f"{props.major}.{props.minor}"
            facts["compute_capability"] = cap
            facts["precision_note"] = (
                "T4 (SM 7.5): fp16 only, no bf16 tensor cores"
                if (props.major, props.minor) == (7, 5)
                else f"SM {cap}"
            )
    except Exception as exc:  # noqa: BLE001
        facts["torch"] = f"unavailable ({type(exc).__name__})"
    try:
        import transformers

        facts["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import open_clip

        facts["open_clip"] = open_clip.__version__
    except Exception:  # noqa: BLE001
        pass
    return facts


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the Phase 8 grounding head")
    ap.add_argument("--data-root", required=True,
                    help="VRSBench root holding VRSBench_train.json + the image tree")
    ap.add_argument("--checkpoint", default=None,
                    help="local RemoteCLIP .pt; skips the Hub fetch")
    ap.add_argument("--cache-dir", default=None,
                    help="feature cache dir; default is <output-dir>/cache")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap train items; use for a first real-data run")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--val-fraction", type=float, default=None)
    ap.add_argument("--extract-only", action="store_true",
                    help="build the feature cache and stop; do not train")
    ap.add_argument("--resume", action="store_true",
                    help="resume from <output-dir>/checkpoint_last.pt")
    ap.add_argument("--debug", action="store_true",
                    help="2 batches, forward+backward, checkpoint, reload")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate inputs and config; train nothing")
    args = ap.parse_args()

    from core.config import load_config
    from core.errors import SatQueryError

    cfg = load_config(args.config)
    device = args.device or cfg.device_preference
    seed = args.seed if args.seed is not None else int(cfg.get("project.seed", 42))
    out_dir = Path(args.output_dir) if args.output_dir else OUT_DIR
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    resolution = int(cfg.get("grounding.image_size", 224))

    print("=" * 70)
    print("PHASE 8 — GROUNDING HEAD TRAINING")
    print("=" * 70)
    print(f"data root   : {args.data_root}")
    print(f"output dir  : {out_dir}")
    print(f"cache dir   : {cache_dir}")
    print(f"checkpoint  : {args.checkpoint or 'hub fetch (pinned revision)'}")
    print(f"resolution  : {resolution}  (grid {resolution // 32}x{resolution // 32})")
    print(f"device      : {device}")
    print(f"seed        : {seed}")
    print(f"config hash : {cfg.hash}")
    print()

    hr("ENVIRONMENT")
    env = environment_report(device)
    for key in sorted(env):
        print(f"  {key:<22} {env[key]}")
    print()

    # -- resolve the checkpoint before any heavy work ----------------------
    checkpoint = args.checkpoint
    if checkpoint is None and not args.dry_run:
        from huggingface_hub import hf_hub_download

        try:
            checkpoint = hf_hub_download(
                cfg.get("grounding.checkpoint_repo"),
                cfg.get("grounding.checkpoint_file"),
                revision=cfg.get("grounding.checkpoint_revision"),
            )
            print(f"fetched checkpoint: {checkpoint}")
        except Exception as exc:  # noqa: BLE001
            print(f"CHECKPOINT FETCH FAILED: {type(exc).__name__}: {exc}")
            return 2
    elif checkpoint is not None:
        if not Path(checkpoint).exists():
            print(f"checkpoint does not exist: {checkpoint}")
            return 2

    # -- dataset -----------------------------------------------------------
    from training.grounding.dataset import (
        assert_image_disjoint,
        attach_cached,
        build_items,
        cache_dir_for,
        extract_features,
        split_by_image,
    )

    hr("DATASET (VRSBench train split)")
    try:
        items = build_items(
            args.data_root,
            json_name="VRSBench_train.json",
            limit=args.limit,
            seed=seed,
            verbose=True,
        )
    except SatQueryError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        print()
        print("  If this says 0 usable and every record was skipped for")
        print("  'image not found', Images_train.zip has not been extracted")
        print("  into the data root. Grounding head training needs those images.")
        return 2

    if not items:
        print("no usable training items")
        return 2

    print(f"  items           : {len(items):,}")
    print(f"  unique images   : {len({i.image_key for i in items}):,}")
    have_source = sum(1 for i in items if i.source_path is not None)
    print(f"  with source path: {have_source:,}")

    val_fraction = (args.val_fraction if args.val_fraction is not None
                    else float(cfg.get("grounding_training.val_fraction", 0.10)))
    train_items, val_items = split_by_image(items, val_fraction=val_fraction, seed=seed)
    assert_image_disjoint(train_items, val_items)   # raises, never warns
    print(f"  train           : {len(train_items):,} items "
          f"({len({i.image_key for i in train_items}):,} images)")
    print(f"  val             : {len(val_items):,} items "
          f"({len({i.image_key for i in val_items}):,} images)")
    print(f"  image-disjoint  : True")
    print()

    if args.dry_run:
        print("DRY RUN — inputs validated, nothing encoded or trained.")
        return 0

    # -- stage 1: extract --------------------------------------------------
    hr("STAGE 1 — FEATURE EXTRACTION (frozen encoder)")
    from specialists.grounding.remoteclip import build_encoder

    encoder = build_encoder(cfg, resolution=resolution, device=device,
                            checkpoint_path=checkpoint)
    print(f"  encoder : {encoder.describe()}")

    cache_root = cache_dir_for(cache_dir, resolution)
    report = extract_features(items, encoder, cache_root, progress_every=200)
    for key in ("images_total", "image_cache_hits", "image_cache_misses",
                "images_missing_source", "images_undecodable", "image_seconds",
                "phrases_total", "text_cache_hits", "text_cache_misses",
                "text_failed", "text_seconds", "seconds"):
        print(f"  {key:<22} {report[key]:,}" if isinstance(report[key], int)
              else f"  {key:<22} {report[key]}")

    if report["images_missing_source"] or report["images_undecodable"]:
        print()
        print("  [!] some images could not be encoded. A missing source means the")
        print("      image tree is incomplete; undecodable means a corrupt file.")
        print("      Neither is zero-filled — those items are dropped.")

    if args.extract_only:
        print()
        print("EXTRACT ONLY — cache built, training skipped.")
        return 0

    # -- attach + stage 2 --------------------------------------------------
    hr("STAGE 2 — TRAIN")
    train_items = attach_cached(train_items, cache_root)
    val_items = attach_cached(val_items, cache_root)
    print(f"  train items     : {len(train_items):,}")
    print(f"  val items       : {len(val_items):,}")
    if not train_items or not val_items:
        print("no cached items after attach; extraction must have failed")
        return 2
    print()

    from training.grounding.train import (
        MIN_IMPROVEMENT_IOU,
        ZERO_SHOT_BASELINE_IOU,
        train_grounding_head,
    )

    epochs = args.epochs if args.epochs is not None else int(cfg.get("grounding_training.epochs", 20))
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get("grounding_training.batch_size", 16))
    lr = args.learning_rate if args.learning_rate is not None else float(cfg.get("grounding_training.learning_rate", 1e-4))
    max_steps = 2 if args.debug else args.max_steps
    if args.debug:
        epochs = 1

    resume_from = None
    if args.resume:
        candidate = out_dir / "checkpoint_last.pt"
        if candidate.exists():
            resume_from = candidate
            print(f"  resuming from {candidate}")
        else:
            print(f"  --resume given but {candidate} does not exist; starting fresh")

    started = time.time()
    result = train_grounding_head(
        train_items, val_items, cfg,
        output_dir=out_dir,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        seed=seed,
        max_steps=max_steps,
        resume_from=resume_from,
        verbose=True,
    )
    wall = time.time() - started

    print()
    print(result.summary())

    hr("BASELINE COMPARISON")
    print(f"  Phase 7 zero-shot   : {ZERO_SHOT_BASELINE_IOU:.4f} mean best IoU "
          f"(16,159 real eval records)")
    print(f"  this head (val)     : {result.best_val_iou:.4f}")
    print(f"  improvement         : {result.best_val_iou - ZERO_SHOT_BASELINE_IOU:+.4f}")
    print(f"  required margin     : +{MIN_IMPROVEMENT_IOU:.2f}")
    print(f"  verdict             : "
          f"{'BEATS BASELINE' if result.beats_baseline else 'DOES NOT BEAT BASELINE'}")
    print()
    print("  NOTE: validation here is the TRAIN split's held-out images, not the")
    print("  16,159-record eval set. The eval set is scored separately by")
    print("  scripts/eval_grounding_head.py against the frozen artifact.")

    # -- write a compact run record ---------------------------------------
    run_record = {
        "run_id": f"grounding_head_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": cfg.hash,
        "seed": seed,
        "device": device,
        "environment": env,
        "data_root": str(args.data_root),
        "checkpoint": str(checkpoint),
        "cache": report,
        "train_items": len(train_items),
        "val_items": len(val_items),
        "hyperparameters": {
            "epochs": epochs, "batch_size": batch_size, "learning_rate": lr,
            "max_steps": max_steps, "val_fraction": val_fraction, "debug": args.debug,
        },
        "best_val_iou": result.best_val_iou,
        "baseline_mean_best_iou": ZERO_SHOT_BASELINE_IOU,
        "improvement": result.best_val_iou - ZERO_SHOT_BASELINE_IOU,
        "beats_baseline": result.beats_baseline,
        "wall_seconds": round(wall, 1),
        "artifact_dir": str(result.artifact_dir),
        "history": result.metadata.get("history", []),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = out_dir / "run_record.json"
    record_path.write_text(json.dumps(run_record, indent=2, sort_keys=True, default=str),
                           encoding="utf-8")

    # -- package for download ---------------------------------------------
    archive = shutil.make_archive(str(out_dir.parent / f"{out_dir.name}"), "zip", str(out_dir))
    hr("WROTE")
    print(f"  {out_dir / 'head.pt'}")
    print(f"  {out_dir / 'training_metadata.json'}")
    print(f"  {record_path}")
    print(f"  {archive}")
    hr("=")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())