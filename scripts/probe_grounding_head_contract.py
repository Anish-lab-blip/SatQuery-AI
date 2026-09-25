"""Phase 8 contract probe — grounding head I/O, VRSBench train schema, leakage.

Answers six questions BEFORE any training code is written, because each one can
invalidate the head design:

  1. Do the VRSBench train and eval IMAGE SETS overlap? If they do, training on
     train poisons the eval metric and the whole Phase 7 decision is suspect.
  2. Does the train `[refer]` schema parse, and what fraction survives the
     out-of-range box filter? (Measured earlier: train GT is noisy, -73..196.)
  3. What is the train box-size distribution? It decides whether a 7x7 grid can
     even represent the target objects.
  4. What exactly does the FROZEN RemoteCLIP encoder emit at 224?
  5. Do the head's declared input/output shapes match that, on real imagery?
  6. Does the NMS + decode path behave on real score fields?

Writes artifacts/grounding/phase8_contract.json and prints a compact summary.

    python scripts/probe_grounding_head_contract.py --images 64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

VRS = REPO_ROOT / "training" / "data" / "vrsbench"
TRAIN_JSON = VRS / "VRSBench_train.json"
EVAL_JSON = VRS / "VRSBench_EVAL_referring.json"
OUT_DIR = REPO_ROOT / "artifacts" / "grounding"


def hr(t: str = "") -> None:
    print("-" * 70)
    if t:
        print(t)
        print("-" * 70)


# ---------------------------------------------------------------------------
# 1. leakage
# ---------------------------------------------------------------------------

def check_image_overlap() -> dict:
    """Do train and eval reference any of the same image files?"""
    hr("1. TRAIN/EVAL IMAGE OVERLAP  (leakage)")

    train = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    evalr = json.loads(EVAL_JSON.read_text(encoding="utf-8"))

    train_imgs = {str(r.get("image", "")).split("/")[-1] for r in train}
    eval_imgs = {str(r.get("image_id", "")) for r in evalr}
    train_imgs.discard("")
    eval_imgs.discard("")

    overlap = train_imgs & eval_imgs

    print(f"  train records        : {len(train):,}")
    print(f"  train unique images  : {len(train_imgs):,}")
    print(f"  eval records         : {len(evalr):,}")
    print(f"  eval unique images   : {len(eval_imgs):,}")
    print(f"  INTERSECTION         : {len(overlap):,}")
    if overlap:
        print(f"  [!] examples         : {sorted(overlap)[:5]}")
        print("  [!] TRAIN/EVAL LEAKAGE — training on train would contaminate eval")
    else:
        print("  clean: the two splits reference disjoint image sets")

    # Naming-convention sanity: a shared stem with different extensions would
    # slip past the exact-name check above.
    train_stems = {Path(n).stem for n in train_imgs}
    eval_stems = {Path(n).stem for n in eval_imgs}
    stem_overlap = train_stems & eval_stems
    print(f"  stem-only intersection: {len(stem_overlap):,}")
    if stem_overlap and not overlap:
        print("  [!] same stems, different extensions — treated as overlap")
    return {
        "train_records": len(train),
        "eval_records": len(evalr),
        "train_images": len(train_imgs),
        "eval_images": len(eval_imgs),
        "exact_overlap": len(overlap),
        "stem_overlap": len(stem_overlap),
        "leak_free": len(overlap) == 0 and len(stem_overlap) == 0,
    }


# ---------------------------------------------------------------------------
# 2 + 3. train schema and box distribution
# ---------------------------------------------------------------------------

def check_train_schema() -> dict:
    hr("2. TRAIN [refer] SCHEMA  (via the real loader)")

    from training.data.vrsbench import load_referring_samples

    # The loader scans shallowest-first and would pick the EVAL file
    # alphabetically. json_name pins the train file explicitly.
    t0 = time.time()
    try:
        samples = load_referring_samples(
            str(VRS), limit=None, verbose=True, open_images=False,
            json_name="VRSBench_train.json",
        )
    except TypeError as exc:
        print(f"  loader does not accept json_name yet: {exc}")
        print("  -> patch training/data/vrsbench.py before proceeding")
        return {"ok": False, "reason": "loader has no json_name parameter"}
    elapsed = time.time() - t0

    n = len(samples)
    print(f"  usable refer samples : {n:,}  ({elapsed:.1f}s, lazy images)")

    widths = np.asarray([s.box[2] - s.box[0] for s in samples], dtype=np.float64)
    heights = np.asarray([s.box[3] - s.box[1] for s in samples], dtype=np.float64)
    areas = widths * heights
    centres_x = np.asarray([(s.box[0] + s.box[2]) / 2 for s in samples])
    centres_y = np.asarray([(s.box[1] + s.box[3]) / 2 for s in samples])

    hr("3. TRAIN BOX DISTRIBUTION  (does a 7x7 grid represent these?)")
    for name, arr in (("width", widths), ("height", heights), ("area", areas)):
        print(f"  {name:<7} min {arr.min():.4f}  p25 {np.percentile(arr,25):.4f}  "
              f"med {np.median(arr):.4f}  p75 {np.percentile(arr,75):.4f}  "
              f"max {arr.max():.4f}")

    # A cell of a 7x7 grid is 1/7 = 0.1429 wide. How many targets are SMALLER
    # than one cell? Those cannot be localised by cell assignment alone.
    cell = 1.0 / 7.0
    smaller = int(np.sum((widths < cell) & (heights < cell)))
    print()
    print(f"  one grid cell        : {cell:.4f} of image width")
    print(f"  targets smaller than one cell : {smaller:,} "
          f"({100.0 * smaller / max(1, n):.1f}%)")
    print(f"  targets larger than half the image : "
          f"{int(np.sum(areas > 0.25)):,}")
    print()
    print(f"  centre x  min {centres_x.min():.3f} med {np.median(centres_x):.3f} "
          f"max {centres_x.max():.3f}")
    print(f"  centre y  min {centres_y.min():.3f} med {np.median(centres_y):.3f} "
          f"max {centres_y.max():.3f}")

    return {
        "ok": True,
        "usable": n,
        "seconds": round(elapsed, 1),
        "width": {"min": float(widths.min()), "median": float(np.median(widths)),
                  "max": float(widths.max())},
        "height": {"min": float(heights.min()), "median": float(np.median(heights)),
                   "max": float(heights.max())},
        "cell_size": cell,
        "targets_smaller_than_cell": smaller,
        "targets_smaller_than_cell_fraction": round(smaller / max(1, n), 4),
        "targets_larger_than_quarter_image": int(np.sum(areas > 0.25)),
    }


# ---------------------------------------------------------------------------
# 4 + 5. head I/O against the real encoder
# ---------------------------------------------------------------------------

def check_head_io(n_images: int, checkpoint: str | None) -> dict:
    hr("4. FROZEN ENCODER OUTPUT AT 224")

    from core.config import load_config
    from specialists.grounding.remoteclip import build_encoder
    from training.data.vrsbench import load_referring_samples

    cfg = load_config()
    resolution = int(cfg.get("grounding.image_size", 224))
    if resolution != 224:
        print(f"  [!] config grounding.image_size is {resolution}, expected 224")

    encoder = build_encoder(cfg, resolution=resolution, device="cpu",
                            checkpoint_path=checkpoint)
    print(f"  encoder : {encoder.describe()}")

    samples = load_referring_samples(
        str(VRS), limit=n_images, seed=0, verbose=False, open_images=False,
        json_name="VRSBench_EVAL_referring.json",
    )
    print(f"  probing {len(samples)} real images")

    from PIL import Image

    ok = 0
    shapes = set()
    dims = set()
    t0 = time.time()
    for s in samples:
        try:
            im = Image.open(s.image_path).convert("RGB")
            enc = encoder.encode_image(im)
            im.close()
        except Exception as exc:  # noqa: BLE001
            print(f"    skip {s.sample_id}: {type(exc).__name__}: {exc}")
            continue
        shapes.add(enc.patch_tokens.shape)
        dims.add(enc.dim)
        ok += 1
    elapsed = time.time() - t0

    print(f"  encoded              : {ok}/{len(samples)}")
    print(f"  patch token shapes   : {sorted(shapes)}")
    print(f"  projected dim        : {sorted(dims)}")
    print(f"  grid                 : {encoder.grid_size} x {encoder.grid_size}")
    print(f"  mean encode time     : {1000.0 * elapsed / max(1, ok):.1f} ms/image")

    hr("5. HEAD FORWARD PASS ON REAL FEATURES")

    import torch
    from specialists.grounding.head import GroundingHead, decode_cell_relative, nms

    head_cfg_dim = int(cfg.get("grounding_head.feature_dim", 2048))
    expected = 4 * encoder.embedding_dim
    print(f"  config feature_dim   : {head_cfg_dim}")
    print(f"  4 x encoder dim      : {expected}")
    print(f"  MATCH                : {head_cfg_dim == expected}")
    if head_cfg_dim != expected:
        print("  [!] head feature_dim and encoder disagree — fix before training")

    head = GroundingHead(
        feature_dim=int(cfg.get("grounding_head.feature_dim", 2048)),
        hidden_dim=int(cfg.get("grounding_head.hidden_dim", 512)),
        dropout=float(cfg.get("grounding_head.dropout", 0.10)),
    )
    head.eval()
    print(f"  head parameters      : {head.num_parameters():,}")

    sample = samples[0]
    im = Image.open(sample.image_path).convert("RGB")
    enc = encoder.encode_image(im)
    im.close()

    patches = torch.from_numpy(enc.patch_tokens).unsqueeze(0)          # (1, 49, 512)
    text = torch.from_numpy(encoder.encode_text([sample.phrase])[0]).unsqueeze(0)
    print(f"  patch input          : {tuple(patches.shape)}")
    print(f"  text input           : {tuple(text.shape)}")

    t0 = time.time()
    with torch.no_grad():
        out = head(patches, text)
    fwd_ms = (time.time() - t0) * 1000.0

    print(f"  raw output           : {tuple(out.raw.shape)}  (B, N, 5)")
    grid = encoder.grid_size
    boxes = decode_cell_relative(out.raw[..., :4], grid, grid)
    scores = torch.sigmoid(out.raw[..., 4])
    print(f"  decoded boxes        : {tuple(boxes.shape)}")
    print(f"  scores               : {tuple(scores.shape)}  "
          f"range [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  forward time         : {fwd_ms:.1f} ms")

    in_range = bool(((boxes >= -1e-6) & (boxes <= 1 + 1e-6)).all())
    ordered = bool((boxes[..., 2] >= boxes[..., 0]).all()
                   and (boxes[..., 3] >= boxes[..., 1]).all())
    print(f"  boxes within [0,1]   : {in_range}")
    print(f"  boxes well-ordered   : {ordered}")

    hr("6. NMS ON THE REAL SCORE FIELD")
    keep = nms(boxes[0], scores[0], iou_threshold=float(cfg.get("grounding.nms_iou", 0.5)))
    print(f"  candidates in        : {boxes.shape[1]}")
    print(f"  kept after NMS       : {int(keep.numel())}")
    print(f"  top score            : {float(scores[0].max()):.4f}")

    return {
        "resolution": resolution,
        "grid": grid,
        "patch_tokens_shape": list(next(iter(shapes))) if shapes else None,
        "projected_dim": int(next(iter(dims))) if dims else None,
        "feature_dim_config": head_cfg_dim,
        "feature_dim_expected": expected,
        "feature_dim_match": head_cfg_dim == expected,
        "head_parameters": head.num_parameters(),
        "forward_ms": round(fwd_ms, 3),
        "boxes_in_range": in_range,
        "boxes_ordered": ordered,
        "nms_kept": int(keep.numel()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 8 grounding head contract probe")
    ap.add_argument("--images", type=int, default=64)
    ap.add_argument("--checkpoint", default=None,
                    help="local RemoteCLIP .pt; skips the Hub fetch")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("=" * 70)
    print("PHASE 8 CONTRACT PROBE — GROUNDING HEAD")
    print("=" * 70)
    print()

    report: dict = {"created_at": datetime.now(timezone.utc).isoformat()}

    try:
        report["overlap"] = check_image_overlap()
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        report["overlap"] = {"ok": False, "error": str(exc)}
    print()

    try:
        report["train_schema"] = check_train_schema()
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        report["train_schema"] = {"ok": False, "error": str(exc)}
    print()

    try:
        report["head_io"] = check_head_io(args.images, args.checkpoint)
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        report["head_io"] = {"ok": False, "error": str(exc)}

    hr("SUMMARY")
    ov = report.get("overlap", {})
    print(f"  leakage-free            : {ov.get('leak_free')}")
    ts = report.get("train_schema", {})
    print(f"  train refer samples     : {ts.get('usable', 'n/a')}")
    print(f"  targets < 1 grid cell   : "
          f"{ts.get('targets_smaller_than_cell_fraction', 'n/a')}")
    hio = report.get("head_io", {})
    print(f"  head feature_dim match  : {hio.get('feature_dim_match')}")
    print(f"  head parameters         : {hio.get('head_parameters')}")
    print(f"  boxes in range          : {hio.get('boxes_in_range')}")
    hr("=")

    out = Path(args.out) if args.out else OUT_DIR / "phase8_contract.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str),
                   encoding="utf-8")
    print(f"wrote {out}")

    if not ov.get("leak_free", False):
        print("\nSTOP: train/eval image overlap must be resolved before training.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())