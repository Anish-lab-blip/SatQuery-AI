"""Phase 9 smoke test -- does the change detector actually LEARN?

Runs the full training path on a synthetic LEVIR-CD-shaped dataset with a KNOWN,
learnable signal, on CPU, in well under two minutes. No LEVIR-CD data, no
network, no downloads.

WHY A SYNTHETIC SIGNAL, AND WHY A MEASURED FLOOR
------------------------------------------------
A smoke test that only checks "the loop runs without crashing" passes on a model
that outputs a constant map. So this fixture plants a signal the detector can
find, and it measures the two floors the trained model must clear instead of
asserting them:

    untrained           a freshly initialised detector, evaluated for real.
                        `stanet.py` initialises the final logit bias to -2.0, so
                        the untrained map is almost all background and its IoU is
                        ~0. This is MEASURED here, not assumed.
    constant-output     the best a model with no spatial awareness can do. If it
                        predicts "change" everywhere its IoU is exactly the mean
                        change fraction of the val tiles; if it predicts
                        "background" everywhere its IoU is 0. The ceiling is
                        therefore the mean change fraction, printed below.

A trained run must clear `--min-iou` (default 0.50), which is far above both.
That is the difference between "the code executed" and "the head learned".

THE FIXTURE
-----------
Scenes, not tiles, own the background: every tile of a scene shares one texture,
exactly as neighbouring 256px crops of a real LEVIR-CD scene share a landscape.
Each tile then has a bright rectangular region present in T2 but not T1, and the
label is that rectangle. The split is by SCENE via `split_by_image`, so this also
exercises the leakage guard that training depends on.

Images are 80x80 and `change.tile_size` is 64, so the centre-crop path in
`training/change/train.py::_fit_to_tile` runs for real -- and the rectangle is
placed inside the central 64x64 window so the crop keeps the signal.

    python scripts/smoke_test_change.py
    python scripts/smoke_test_change.py --scenes 10 --epochs 15 --keep-temp

Exit 0 = the head learned. Non-zero = do not spend real data or GPU time on it.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

#: Synthetic image edge. Larger than the tile so the centre crop is exercised.
IMAGE_SIZE = 80

#: The tile the detector sees. Must be a multiple of 8 (encoder stride).
TILE_SIZE = 64

#: Edge of the changed rectangle. 18*18 = 324 px in a 64*64 tile is a 7.9%
#: change fraction -- inside LEVIR-CD's documented 5-15% band, so the class
#: imbalance the loss has to cope with is realistic rather than convenient.
RECT_SIZE = 18

#: The trained detector must clear this pooled validation IoU.
MIN_IOU = 0.50


def hr(title: str = "") -> None:
    print("-" * 70)
    if title:
        print(title)
        print("-" * 70)


def _scene_texture(rng: np.random.Generator, size: int) -> np.ndarray:
    """A smooth, non-trivial background so the encoder has something to encode."""
    yy, xx = np.mgrid[0:size, 0:size]
    base = (
        70.0
        + 30.0 * np.sin(xx / 7.0)
        + 30.0 * np.cos(yy / 9.0)
        + rng.normal(0.0, 4.0, (size, size))
    )
    return np.clip(base, 0.0, 255.0)


def build_synthetic_dataset(
    root: Path,
    *,
    scenes: int,
    tiles_per_scene: int,
    seed: int,
) -> None:
    """Write a LEVIR-CD-shaped tree: <root>/train/{A,B,label}/*.png.

    Every tile of a scene shares the scene's texture; the change rectangle is
    drawn at a per-tile random position inside the central crop window.
    """
    from PIL import Image

    for sub in ("A", "B", "label"):
        (root / "train" / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    lo = (IMAGE_SIZE - TILE_SIZE) // 2
    hi = IMAGE_SIZE - lo - RECT_SIZE
    if hi <= lo:
        raise SystemExit(
            f"RECT_SIZE {RECT_SIZE} does not fit the central {TILE_SIZE}px window "
            f"of an {IMAGE_SIZE}px image"
        )

    for scene_index in range(scenes):
        texture = _scene_texture(rng, IMAGE_SIZE)
        rgb = np.stack(
            [
                texture,
                np.clip(texture * 0.9 + 10.0, 0.0, 255.0),
                np.clip(texture * 0.7 + 30.0, 0.0, 255.0),
            ],
            axis=-1,
        ).astype(np.uint8)

        for tile_index in range(tiles_per_scene):
            stem = f"scene{scene_index:02d}__tile{tile_index:02d}"

            t1 = rgb.copy()
            t2 = rgb.copy()
            label = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

            # Per-tile jitter so the detector cannot memorise a fixed location.
            row = int(rng.integers(lo, hi + 1))
            col = int(rng.integers(lo, hi + 1))
            row1, col1 = row + RECT_SIZE, col + RECT_SIZE

            # The change: the region becomes much brighter in T2. Large and
            # unambiguous on purpose -- this fixture tests the PIPELINE, and a
            # signal at the edge of detectability would confound "broken" with
            # "needs more epochs".
            t2[row:row1, col:col1] = np.clip(
                235.0 + rng.normal(0.0, 3.0, (RECT_SIZE, RECT_SIZE, 3)), 0.0, 255.0
            ).astype(np.uint8)
            label[row:row1, col:col1] = 255

            Image.fromarray(t1).save(root / "train" / "A" / f"{stem}.png")
            Image.fromarray(t2).save(root / "train" / "B" / f"{stem}.png")
            Image.fromarray(label).save(root / "train" / "label" / f"{stem}.png")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 9 change-detector smoke test")
    ap.add_argument("--scenes", type=int, default=8)
    ap.add_argument("--tiles-per-scene", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=3e-3)
    ap.add_argument("--val-fraction", type=float, default=0.25)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-iou", type=float, default=MIN_IOU)
    ap.add_argument("--keep-temp", action="store_true",
                    help="keep the synthetic dataset and the trained artifact")
    args = ap.parse_args()

    import torch

    from core.config import load_config
    from specialists.change.stanet import build_change_model
    from training.change.dataset import (
        assert_image_disjoint,
        load_levir_dataset,
        split_by_image,
    )
    from training.change.train import evaluate, train_change_head

    print("=" * 70)
    print("PHASE 9 SMOKE TEST -- DOES THE CHANGE HEAD LEARN?")
    print("=" * 70)
    print(f"scenes          : {args.scenes} x {args.tiles_per_scene} tiles")
    print(f"image / tile    : {IMAGE_SIZE}px / {TILE_SIZE}px (centre crop)")
    print(f"change fraction : {(RECT_SIZE * RECT_SIZE) / (TILE_SIZE * TILE_SIZE):.4f}")
    print(f"epochs          : {args.epochs}")
    print(f"batch size      : {args.batch_size}")
    print(f"learning rate   : {args.learning_rate}")
    print(f"seed            : {args.seed}")
    print(f"device          : cpu")
    print()

    temp_dir = Path(tempfile.mkdtemp(prefix="change_smoke_"))
    exit_code = 1
    started = time.time()

    try:
        hr("FIXTURE")
        build_synthetic_dataset(
            temp_dir,
            scenes=args.scenes,
            tiles_per_scene=args.tiles_per_scene,
            seed=args.seed,
        )
        print(f"  wrote {args.scenes * args.tiles_per_scene} pairs to {temp_dir}")

        items = load_levir_dataset(str(temp_dir), splits=("train",))
        if not items:
            print("FAILED: the synthetic dataset loaded zero items")
            return 1

        # Scene grouping. The loader keys items by pair, and every pair here is
        # one tile, so the scene key is recovered from the stem prefix. Without
        # this the split would be by TILE, which is the leak the guard exists to
        # stop.
        for item in items:
            stem = item["sample_id"].split("_", 1)[1]
            item["group_key"] = stem.split("__", 1)[0]

        train_items, val_items = split_by_image(
            items, val_fraction=args.val_fraction, seed=args.seed
        )
        assert_image_disjoint(train_items, val_items)

        train_scenes = {i["group_key"] for i in train_items}
        val_scenes = {i["group_key"] for i in val_items}
        print(f"  train           : {len(train_items)} tiles "
              f"({len(train_scenes)} scenes)")
        print(f"  val             : {len(val_items)} tiles "
              f"({len(val_scenes)} scenes)")
        print(f"  scene-disjoint  : {not (train_scenes & val_scenes)}")
        print()

        # -- config: synthetic, offline, CPU --------------------------------
        cfg = load_config(
            overrides={
                "change": {
                    "pretrained": False,   # no download, no network
                    "tile_size": TILE_SIZE,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                }
            }
        )
        threshold = float(cfg.get("change.threshold", 0.50))

        # -- floor 1: the untrained detector, MEASURED ----------------------
        hr("FLOORS")
        # Seed identically to `train_change_head`, which reseeds from `seed`, so
        # this is the SAME initialisation training will start from.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        untrained = build_change_model(cfg, device="cpu")
        untrained_val = evaluate(
            untrained,
            val_items,
            threshold=threshold,
            device="cpu",
            batch_size=args.batch_size,
            tile_size=TILE_SIZE,
        )
        print(f"  untrained val IoU (measured) : {untrained_val.iou:.4f}")

        # -- floor 2: the constant-output ceiling, COMPUTED -----------------
        change_fraction = float(np.mean(untrained_val.per_image_change_fraction))
        print(f"  mean change fraction         : {change_fraction:.4f}")
        print(f"  constant-output ceiling      : {change_fraction:.4f} "
              f"(predict 'change' everywhere)")
        print(f"  required trained val IoU     : {args.min_iou:.4f}")
        print()

        # -- train ----------------------------------------------------------
        hr("TRAINING")
        result = train_change_head(
            train_items,
            val_items,
            cfg,
            output_dir=temp_dir / "artifacts",
            device="cpu",
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            max_steps=args.max_steps,
            verbose=True,
        )
        print()
        print(result.summary())

        # -- verdict --------------------------------------------------------
        hr("VERDICT")
        best = float(result.best_val_iou)
        cleared_untrained = best > untrained_val.iou
        cleared_constant = best > change_fraction
        cleared_threshold = best >= args.min_iou

        print(f"  untrained val IoU    : {untrained_val.iou:.4f}")
        print(f"  constant ceiling     : {change_fraction:.4f}")
        print(f"  best trained val IoU : {best:.4f}")
        print(f"  first-epoch val IoU  : {result.first_val_iou:.4f}")
        print(f"  beats untrained      : {cleared_untrained}")
        print(f"  beats constant       : {cleared_constant}")
        print(f"  clears {args.min_iou:.2f}           : {cleared_threshold}")
        print()

        if cleared_untrained and cleared_constant and cleared_threshold:
            print(f"SMOKE TEST: PASS -- the head learned the planted signal "
                  f"(val IoU {best:.4f} >= {args.min_iou:.2f})")
            exit_code = 0
        else:
            print("SMOKE TEST: FAIL -- the head did not learn the planted signal.")
            print("  A constant-output model would have scored "
                  f"{change_fraction:.4f}; this run scored {best:.4f}.")
            exit_code = 1

        print(f"  artifact : {result.artifact_dir}")
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print()
        print(f"SMOKE TEST: FAIL -- {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        elapsed = time.time() - started
        print()
        print(f"  wall clock : {elapsed:.1f}s")
        if args.keep_temp:
            print(f"  temp kept  : {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"  temp removed: {temp_dir}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
