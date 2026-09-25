"""Phase 9 entry point -- train the LEVIR-CD change detector.

A THIN CLI over `training/change/train.py`, mirroring the Phase 8 split between
`scripts/train_grounding.py` and `training/grounding/train.py`. Everything that
computes lives in the module; this file parses arguments, reports the
environment, prints the dataset accounting, and returns an exit code.

    # validate the data and the split; build nothing, train nothing
    python scripts/train_change.py --data-root <LEVIR-CD root> --dry-run

    # real run on CPU
    python scripts/train_change.py --data-root <LEVIR-CD root> --device cpu

    # a first real-data run that does not commit an hour
    python scripts/train_change.py --data-root <root> --limit 256 --epochs 3

EXIT CODES
----------
    0   training completed (or --dry-run validated a present dataset)
    2   the dataset is missing, empty, or cannot be split -- NOT a crash
    3   the run started and the trainer raised a typed error

`--dry-run` is the leakage check without the cost: it loads the items, performs
the scene-disjoint split, runs `assert_image_disjoint`, prints the accounting and
exits. It does NOT build the detector, so it cannot trigger a pretrained-weights
download, and it writes no files.

The three names below are re-exported at module scope on purpose.
`tests/unit/test_change_train_script_contract.py` asserts `change_loss`,
`train_change_head` and `evaluate` are reachable as `scripts.train_change.*`,
and that `change_loss` forwards its weights rather than discarding them. That
test must keep passing unchanged; the re-export is the contract, not a
convenience.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.change.train import (  # noqa: E402  (path set above)
    ChangeTrainingError,
    change_loss,
    evaluate,
    load_trained_change_model,
    train_change_head,
)

OUT_DIR = REPO_ROOT / "artifacts" / "change" / "levir_change_v001"

__all__ = [
    "change_loss",
    "train_change_head",
    "evaluate",
    "load_trained_change_model",
    "ChangeTrainingError",
    "main",
]


def hr(title: str = "") -> None:
    print("-" * 70)
    if title:
        print(title)
        print("-" * 70)


def environment_report(device: str) -> dict:
    """What ran, where. Recorded in the run record so a number has a context."""
    facts: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "requested_device": device,
    }
    try:
        import torch

        facts["torch"] = torch.__version__
        facts["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        facts["torch"] = f"unavailable ({type(exc).__name__})"
    try:
        import cv2

        facts["opencv"] = cv2.__version__
    except Exception:  # noqa: BLE001
        pass
    return facts


def _scene_key(item) -> str:
    if isinstance(item, dict):
        return str(
            item.get("group_key") or item.get("image_key") or item.get("sample_id")
        )
    for attr in ("group_key", "image_key", "sample_id"):
        value = getattr(item, attr, None)
        if value:
            return str(value)
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the Phase 9 change detector")
    ap.add_argument("--data-root", required=True,
                    help="LEVIR-CD root holding train/val/test A,B,label trees")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda; defaults to cpu (Phase 9 trains on CPU)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--val-fraction", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total items after shuffling; for a first run")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap OPTIMIZER steps, not epochs")
    ap.add_argument("--resume-from", default=None,
                    help="a checkpoint_last.pt to resume from")
    ap.add_argument("--dry-run", action="store_true",
                    help="load the dataset and check the split; train nothing")
    args = ap.parse_args()

    from core.config import load_config
    from core.errors import SatQueryError
    from training.change.dataset import (
        LEVIRCDError,
        assert_image_disjoint,
        load_levir_dataset,
        split_by_image,
    )

    cfg = load_config(args.config)
    device = args.device or cfg.device_preference
    seed = args.seed if args.seed is not None else int(cfg.get("project.seed", 42))
    out_dir = Path(args.output_dir) if args.output_dir else OUT_DIR
    tile_size = int(cfg.get("change.tile_size", 256))
    threshold = float(cfg.get("change.threshold", 0.50))
    val_fraction = (
        args.val_fraction if args.val_fraction is not None
        else float(cfg.get("change.val_fraction", 0.10))
    )

    print("=" * 70)
    print("PHASE 9 -- CHANGE-DETECTION TRAINING (LEVIR-CD)")
    print("=" * 70)
    print(f"data root   : {args.data_root}")
    print(f"output dir  : {out_dir}")
    print(f"tile size   : {tile_size}px  (encoder stride 8)")
    print(f"threshold   : {threshold:.2f}")
    print(f"device      : {device}")
    print(f"seed        : {seed}")
    print(f"val fraction: {val_fraction}")
    print(f"config hash : {cfg.hash}")
    print(f"dry run     : {args.dry_run}")
    print()

    hr("ENVIRONMENT")
    env = environment_report(device)
    for key in sorted(env):
        print(f"  {key:<22} {env[key]}")
    print()

    # -- dataset -----------------------------------------------------------
    #
    # THE TEST SPLIT IS NEVER LOADED HERE.
    #
    # This block used to call `load_levir_dataset(root)` with the default
    # `splits=("train", "val", "test")` and then hand ALL 637 scenes to
    # `split_by_image`, which re-partitioned train+val+test into a fresh
    # train/val pair. The frozen test split was therefore folded into the
    # training pool -- a public-test firewall violation, and invisible because
    # the only fixture ever used had a `train/` directory and nothing else.
    #
    # The dataset's own splits are authoritative: the frozen contract is
    # train=445 / val=64 / test=128 scenes. `val` is used when the dataset ships
    # one; `val_fraction` only applies to a root that ships train alone.
    hr("DATASET (LEVIR-CD)")
    try:
        pool = load_levir_dataset(
            args.data_root, splits=("train", "val"), limit=args.limit, seed=seed
        )
    except FileNotFoundError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        print()
        print("  The LEVIR-CD root does not exist or has no readable split.")
        print("  Accepted layouts:")
        print("    nested : <root>/<split>/{A,B,label}/<scene>.png")
        print("    flat   : <root>/{A,B,label}/<split>_<scene>_<tile>.png")
        return 2
    except SatQueryError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 2

    if not pool:
        print("no usable items")
        print()
        print("  No split directory held a matching A / B / label triple.")
        print("  LEVIR-CD ships A/, B/ and label/ per split; all three are needed.")
        return 2

    layout = pool[0].get("layout", "unknown")
    train_pool = [i for i in pool if i["split"] == "train"]
    val_pool = [i for i in pool if i["split"] == "val"]

    print(f"  layout          : {layout}")
    print(f"  items           : {len(pool):,}  (train+val only; test never loaded)")
    print(f"  train items     : {len(train_pool):,} "
          f"({len({_scene_key(i) for i in train_pool}):,} scenes)")
    print(f"  val items       : {len(val_pool):,} "
          f"({len({_scene_key(i) for i in val_pool}):,} scenes)")
    print()

    try:
        if val_pool:
            # The dataset's own val split wins. Re-splitting it would break the
            # frozen 445/64/128 contract and make the val number incomparable.
            train_items, val_items = train_pool, val_pool
            split_source = "the dataset's own val split"
        else:
            train_items, val_items = split_by_image(
                train_pool, val_fraction=val_fraction, seed=seed
            )
            split_source = f"carved from train (val_fraction={val_fraction})"
        assert_image_disjoint(train_items, val_items)   # raises, never warns
    except LEVIRCDError as exc:
        print(f"FAILED to split: {type(exc).__name__}: {exc}")
        print()
        print("  The scene-disjoint split needs at least two distinct scenes and")
        print("  a val_fraction strictly inside (0, 1).")
        return 2

    leaked = [i for i in train_items + val_items if i["split"] == "test"]
    if leaked:  # defensive: unreachable while splits=("train","val") is honoured
        print(f"FAILED: {len(leaked)} test-split items reached the training pool")
        return 2

    print(f"  train           : {len(train_items):,} items "
          f"({len({_scene_key(i) for i in train_items}):,} scenes)")
    print(f"  val             : {len(val_items):,} items "
          f"({len({_scene_key(i) for i in val_items}):,} scenes)")
    print(f"  val source      : {split_source}")
    print("  scene-disjoint  : True")
    print("  test split      : not loaded (firewall intact)")
    print()

    if args.dry_run:
        print("DRY RUN -- dataset loaded, split verified disjoint, nothing trained.")
        print("  No detector was built, so no pretrained weights were fetched.")
        print("  No files were written.")
        return 0

    # -- training ----------------------------------------------------------
    hr("TRAINING")
    epochs = args.epochs if args.epochs is not None else int(cfg.get("change.epochs", 20))
    batch_size = (
        args.batch_size if args.batch_size is not None
        else int(cfg.get("change.batch_size", 8))
    )
    lr = (
        args.learning_rate if args.learning_rate is not None
        else float(cfg.get("change.learning_rate", 1e-3))
    )

    resume_from = Path(args.resume_from) if args.resume_from else None
    if resume_from is not None and not resume_from.exists():
        print(f"  --resume-from does not exist: {resume_from}")
        return 2

    started = time.time()
    try:
        result = train_change_head(
            train_items,
            val_items,
            cfg,
            output_dir=out_dir,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            seed=seed,
            max_steps=args.max_steps,
            resume_from=resume_from,
            verbose=True,
        )
    except SatQueryError as exc:
        print()
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 3
    wall = time.time() - started

    print()
    print(result.summary())

    # -- run record --------------------------------------------------------
    run_record = {
        "run_id": f"change_head_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": cfg.hash,
        "seed": seed,
        "device": device,
        "environment": env,
        "data_root": str(args.data_root),
        "tile_size": tile_size,
        "threshold": threshold,
        "train_items": len(train_items),
        "val_items": len(val_items),
        "train_scenes": len({_scene_key(i) for i in train_items}),
        "val_scenes": len({_scene_key(i) for i in val_items}),
        "hyperparameters": {
            "epochs": epochs, "batch_size": batch_size, "learning_rate": lr,
            "max_steps": args.max_steps, "val_fraction": val_fraction,
        },
        "first_val_iou": result.first_val_iou,
        "best_val_iou": result.best_val_iou,
        "improved": result.improved,
        "wall_seconds": round(wall, 1),
        "artifact_dir": str(result.artifact_dir),
        "history": result.metadata.get("history", []),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = out_dir / "run_record.json"
    record_path.write_text(
        json.dumps(run_record, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    hr("WROTE")
    print(f"  {out_dir / 'head.pt'}")
    print(f"  {out_dir / 'model_metadata.json'}")
    print(f"  {out_dir / 'checkpoint_last.pt'}")
    print(f"  {record_path}")
    hr("=")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
