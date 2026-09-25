"""Phase 8 smoke test — does the head actually LEARN?

Runs the full training path on a synthetic cache with a KNOWN, learnable
signal, on CPU, in under a minute. No VRSBench train images required.

WHY A SYNTHETIC SIGNAL
----------------------
`Images_train.zip` is not downloaded yet, so the real corpus cannot be used.
A smoke test that only checked "the loop runs without crashing" would pass on a
head that outputs constants. So this fixture plants a real signal:

    for each synthetic image, one cell (the one containing the box centre)
    carries a vector encoding that box; every other cell is noise.

A working head must find that cell and regress the box, so validation IoU must
rise well above its untrained value. If it does not, something in the pipeline
is broken and no amount of GPU time would have found it.

    python scripts/smoke_test_grounding_head.py
    python scripts/smoke_test_grounding_head.py --images 512 --epochs 25

Exit 0 = the head learns. Anything else = do not spend GPU time.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

GRID = 7
DIM = 512
PHRASE = "the target object"


def hr(t: str = "") -> None:
    print("-" * 70)
    if t:
        print(t)
        print("-" * 70)


def build_synthetic_cache(
    cache_dir: Path, n_images: int, seed: int = 42
) -> list[dict]:
    """Write a cache whose patches encode each image's box. Returns the boxes."""
    from training.grounding.dataset import image_cache_path, text_cache_path

    cache_dir = Path(cache_dir)
    (cache_dir / "text").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    boxes: list[dict] = []
    for i in range(n_images):
        cx = float(rng.uniform(0.20, 0.80))
        cy = float(rng.uniform(0.20, 0.80))
        w = float(rng.uniform(0.15, 0.40))
        h = float(rng.uniform(0.15, 0.40))
        box = [
            max(0.0, cx - w / 2),
            max(0.0, cy - h / 2),
            min(1.0, cx + w / 2),
            min(1.0, cy + h / 2),
        ]

        patches = rng.normal(0.0, 0.05, (GRID * GRID, DIM)).astype(np.float32)
        col = min(GRID - 1, int(cx * GRID))
        row = min(GRID - 1, int(cy * GRID))
        cell = row * GRID + col

        signal = np.zeros(DIM, dtype=np.float32)
        signal[0:4] = box
        signal[4] = 1.0
        patches[cell] += signal * 8.0

        key = f"syn_{i:05d}"
        np.savez_compressed(
            image_cache_path(cache_dir, key),
            patches=patches.astype(np.float16),
            cls=patches.mean(axis=0).astype(np.float16),
        )
        text = np.zeros(DIM, dtype=np.float32)
        text[5] = 1.0
        np.save(text_cache_path(cache_dir, PHRASE), text.astype(np.float16))

        boxes.append({"key": key, "box": box, "cell": cell})

    return boxes


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 8 grounding head smoke test")
    ap.add_argument("--images", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("PHASE 8 SMOKE TEST — GROUNDING HEAD LEARNS A KNOWN SIGNAL")
    print("=" * 70)

    failures: list[str] = []
    tmp = Path(tempfile.mkdtemp(prefix="p8_smoke_"))
    try:
        from core.config import load_config
        from training.grounding.dataset import (
            GroundingItem,
            assert_image_disjoint,
            attach_cached,
            split_by_image,
        )
        from training.grounding.train import (
            ZERO_SHOT_BASELINE_IOU,
            evaluate,
            load_trained_head,
            train_grounding_head,
        )

        cfg = load_config()
        cache_dir = tmp / "cache"

        hr("1. SYNTHETIC CACHE")
        boxes = build_synthetic_cache(cache_dir, args.images, seed=args.seed)
        print(f"  images written   : {len(boxes)}")
        print(f"  grid             : {GRID} x {GRID} = {GRID * GRID} cells")
        print(f"  cache            : {cache_dir}")

        items = [
            GroundingItem(
                sample_id=b["key"], image_key=b["key"], phrase=PHRASE, box=b["box"]
            )
            for b in boxes
        ]
        items = attach_cached(items, cache_dir, verbose=False)
        print(f"  items attached   : {len(items)}")
        if len(items) != len(boxes):
            failures.append(f"attach_cached kept {len(items)}/{len(boxes)}")

        hr("2. IMAGE-DISJOINT SPLIT")
        train_items, val_items = split_by_image(items, val_fraction=0.25, seed=args.seed)
        assert_image_disjoint(train_items, val_items)
        print(f"  train items      : {len(train_items)} "
              f"({len({i.image_key for i in train_items})} images)")
        print(f"  val items        : {len(val_items)} "
              f"({len({i.image_key for i in val_items})} images)")
        print(f"  disjoint         : True")

        hr("3. UNTRAINED BASELINE (random init)")
        from specialists.grounding.head import build_head

        head0 = build_head(cfg)
        before = evaluate(
            head0, val_items, grid=GRID, device="cpu", strategy="argmax",
            in_memory=True,
        )
        print(f"  mean best IoU    : {before.mean_best_iou:.4f}")
        print(f"  Recall@0.10      : {before.recall.get('0.10', 0.0):.4f}")
        print(f"  Recall@0.50      : {before.recall.get('0.50', 0.0):.4f}")

        hr("4. TRAIN")
        result = train_grounding_head(
            train_items,
            val_items,
            cfg,
            output_dir=tmp / "artifact",
            device="cpu",
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            in_memory=True,
            verbose=True,
        )

        hr("5. DID IT LEARN?")
        after = result.val_argmax
        gain = after.mean_best_iou - before.mean_best_iou
        print(f"  before (random)  : {before.mean_best_iou:.4f}")
        print(f"  after  (trained) : {after.mean_best_iou:.4f}")
        print(f"  gain             : {gain:+.4f}")
        print(f"  threshold        : +0.05 on this synthetic signal")
        if gain < 0.05:
            failures.append(
                f"head did not learn the synthetic signal (gain {gain:+.4f})"
            )
        else:
            print("  LEARNED")

        hr("6. CHECKPOINT ROUND-TRIP")
        ckpt = result.artifact_dir / "head.pt"
        if not ckpt.exists():
            failures.append("head.pt was not written")
        else:
            reloaded = load_trained_head(ckpt, device="cpu")
            again = evaluate(
                reloaded, val_items, grid=GRID, device="cpu", strategy="argmax",
                in_memory=True,
            )
            match = abs(again.mean_best_iou - after.mean_best_iou) < 1e-6
            print(f"  reloaded IoU     : {again.mean_best_iou:.4f}")
            print(f"  matches trained  : {match}")
            if not match:
                failures.append("reloaded head scores differently")

        hr("7. ARTIFACT METADATA")
        meta = result.metadata
        required = [
            "config_hash", "seed", "head_parameters", "train_items", "val_items",
            "history", "val_argmax", "best_val_iou", "baseline", "beats_baseline",
        ]
        missing = [k for k in required if k not in meta]
        print(f"  keys present     : {len(required) - len(missing)}/{len(required)}")
        if missing:
            failures.append(f"metadata missing {missing}")
        print(f"  head parameters  : {meta.get('head_parameters'):,}")
        print(f"  config hash      : {meta.get('config_hash')}")

        hr("8. BASELINE COMPARISON")
        print(f"  Phase 7 zero-shot (16,159 real eval records): "
              f"{ZERO_SHOT_BASELINE_IOU:.4f} mean best IoU")
        print(f"  this synthetic run                         : "
              f"{result.best_val_iou:.4f}")
        print()
        print("  These are NOT comparable. The synthetic signal is trivial by")
        print("  construction and the baseline is real imagery. What this proves")
        print("  is that the pipeline can LEARN — not how well it will score.")

        hr("RESULT")
        if failures:
            print("SMOKE TEST: FAILED")
            for f in failures:
                print(f"  - {f}")
            return 1
        print("SMOKE TEST: PASS — the head learns, checkpoints and reports")
        return 0
    finally:
        if args.keep:
            print(f"kept: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())