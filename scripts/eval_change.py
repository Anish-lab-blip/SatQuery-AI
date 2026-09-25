"""Phase 9 evaluation -- score a trained change detector on a LEVIR-CD split.

    python scripts/eval_change.py --data-root <LEVIR-CD root> \
        --checkpoint artifacts/change/levir_change_v001/head.pt --split test

WHAT IS REPORTED, AND WHY BOTH CONVENTIONS
------------------------------------------
`evaluation/metrics/change.py` computes POOLED metrics (sum the confusion
counts across tiles, then one precision/recall/F1) and MACRO metrics (per-tile,
then averaged). LEVIR-CD papers usually report pooled; some venues report macro.
They can disagree by several points, so both are printed and both are written to
the artifact rather than one being chosen and hoped for. An empty-change tile is
excluded from the macro average by that module and counted separately, because
scoring recall 0.0 there measures the dataset, not the model.

The mask that is scored is the RAW binarized one. `evaluation.metrics.change`
applies its own `>= threshold` rule; no morphological cleanup is applied. See
`training/change/train.py` for the full reason -- `postprocess_change_map`
documents that morphology is a presentation aid and that scoring on the cleaned
mask inflates the number relative to the literature.

CONFIG DRIFT IS REFUSED, NOT WARNED ABOUT
-----------------------------------------
The detector's architecture travels inside the checkpoint, but the run's
configuration does not: `specialists.change.stanet.save_change_model` writes the
`config_hash` to the `model_metadata.json` sidecar beside the `.pt`. If that
recorded hash differs from the hash of the configuration in force now, the
comparison is not reproducible, so this script exits 3 and scores nothing unless
`--allow-config-drift` is passed -- in which case the drift is recorded in the
artifact. A missing sidecar is reported as "not checked", never as agreement.

EXIT CODES
----------
    0   scored
    2   the dataset or the checkpoint is missing / empty
    3   config drift, refused without --allow-config-drift
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "artifacts" / "change"


def hr(title: str = "") -> None:
    print("-" * 70)
    if title:
        print(title)
        print("-" * 70)


def _environment_report(device: str) -> dict:
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
    return facts


def _quantiles(values: list[float]) -> dict[str, float]:
    """Compact description of the per-image change fraction."""
    if not values:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    ordered = sorted(values)

    def at(q: float) -> float:
        idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return float(ordered[idx])

    return {"min": at(0.0), "p50": at(0.5), "p90": at(0.9), "max": at(1.0)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate the Phase 9 change detector")
    ap.add_argument("--data-root", required=True,
                    help="LEVIR-CD root holding train/val/test A,B,label trees")
    ap.add_argument("--checkpoint", required=True,
                    help="a head.pt written by scripts/train_change.py")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap scored tiles; for a smoke run, not a result")
    ap.add_argument("--threshold", type=float, default=None,
                    help="probability threshold; default is change.threshold")
    ap.add_argument("--split", default="test",
                    choices=("train", "val", "test"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--allow-config-drift", action="store_true",
                    help="score anyway when the checkpoint's config hash differs; "
                         "the artifact then records that the run is not a "
                         "frozen-config evaluation")
    args = ap.parse_args()

    from core.config import load_config
    from core.errors import SatQueryError
    from evaluation.metrics.change import score_dataset  # noqa: F401  (contract)
    from training.change.dataset import load_levir_dataset
    from training.change.train import (
        checkpoint_metadata,
        evaluate,
        load_trained_change_model,
    )

    cfg = load_config()
    device = args.device or cfg.device_preference
    seed = args.seed if args.seed is not None else int(cfg.get("project.seed", 42))
    threshold = (
        args.threshold if args.threshold is not None
        else float(cfg.get("change.threshold", 0.50))
    )
    tile_size = int(cfg.get("change.tile_size", 256))
    out_dir = Path(args.output_dir) if args.output_dir else OUT_DIR

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"checkpoint does not exist: {ckpt_path}")
        return 2

    print("=" * 70)
    print(f"PHASE 9 EVALUATION -- CHANGE DETECTOR ON LEVIR-CD {args.split.upper()}")
    print("=" * 70)
    print(f"data root   : {args.data_root}")
    print(f"checkpoint  : {ckpt_path}")
    print(f"split       : {args.split}")
    print(f"tile size   : {tile_size}px")
    print(f"threshold   : {threshold:.2f}")
    print(f"device      : {device}")
    print(f"config hash : {cfg.hash}")
    print(f"limit       : {args.limit or 'ALL tiles'}")
    print()

    hr("ENVIRONMENT")
    env = _environment_report(device)
    for key in sorted(env):
        print(f"  {key:<22} {env[key]}")
    print()

    # -- checkpoint identity, before any data is loaded --------------------
    hr("CHECKPOINT")
    import torch

    try:
        raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        print(f"could not read checkpoint: {type(exc).__name__}: {exc}")
        return 2

    embedded_config = raw.get("config")
    if not embedded_config:
        print("checkpoint carries no embedded model config; refusing to guess")
        return 2

    sidecar = checkpoint_metadata(ckpt_path)
    ckpt_config_hash = sidecar.get("config_hash")
    drift_checked = "config_hash" in sidecar
    drift = bool(drift_checked and ckpt_config_hash not in (None, cfg.hash))

    print(f"  architecture      : {embedded_config}")
    print(f"  sidecar           : {ckpt_path.parent / 'model_metadata.json'}"
          f"{'' if drift_checked else '  (absent)'}")
    print(f"  checkpoint hash   : {ckpt_config_hash if drift_checked else 'NOT CHECKED'}")
    print(f"  current hash      : {cfg.hash}")
    print(f"  drift             : {drift}")
    print()

    if drift and not args.allow_config_drift:
        print("CONFIG DRIFT -- refusing to score.")
        print("  The head was trained under a different configuration, so this")
        print("  comparison is not reproducible. Re-train, or pass")
        print("  --allow-config-drift to proceed knowingly (the artifact will")
        print("  record that the result is not a frozen-config evaluation).")
        return 3
    if drift:
        print("  [!] config drift acknowledged; this is NOT a frozen-config run")
        print()

    # -- dataset -----------------------------------------------------------
    hr(f"EVAL SET ({args.split})")
    try:
        items = load_levir_dataset(
            args.data_root, splits=(args.split,), limit=args.limit, seed=seed
        )
    except FileNotFoundError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 2
    except SatQueryError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 2

    if not items:
        print(f"no items in the {args.split!r} split")
        print("  Expected layout: <root>/%s/{A,B,label}/*.png" % args.split)
        return 2

    # Scenes, not tiles. This printed `len({sample_id})`, which is the TILE count
    # dressed up as the scene count -- on the real flat layout that reads 2048
    # "scenes" for a 128-scene test split.
    n_scenes = len({i.get("group_key") or i.get("image_key") or i["sample_id"]
                    for i in items})
    print(f"  tiles            : {len(items):,}")
    print(f"  scenes           : {n_scenes:,}")
    print()

    # -- score -------------------------------------------------------------
    hr("SCORING")
    try:
        model = load_trained_change_model(ckpt_path, device=device)
    except SatQueryError as exc:
        print(f"FAILED to load the detector: {type(exc).__name__}: {exc}")
        return 2
    print(f"  detector : {model}")

    try:
        result = evaluate(
            model,
            items,
            threshold=threshold,
            device=device,
            batch_size=args.batch_size,
            tile_size=tile_size,
        )
    except SatQueryError as exc:
        print(f"FAILED to score: {type(exc).__name__}: {exc}")
        return 2

    print()
    print(result.summary())
    print()
    print(f"  pooled iou {result.pooled.iou:.4f}   macro iou {result.macro.iou:.4f}")
    fractions = result.per_image_change_fraction
    quantiles = _quantiles(fractions)
    print(f"  per-image change fraction (n={len(fractions)}): "
          f"min {quantiles['min']:.4f}  p50 {quantiles['p50']:.4f}  "
          f"p90 {quantiles['p90']:.4f}  max {quantiles['max']:.4f}")
    print()

    # -- artifact ----------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact": "change_eval",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": cfg.hash,
        "checkpoint": str(ckpt_path),
        "checkpoint_config_hash": ckpt_config_hash,
        "checkpoint_config_hash_checked": drift_checked,
        "checkpoint_embedded_config": embedded_config,
        "config_drift": drift,
        "config_drift_acknowledged": bool(drift and args.allow_config_drift),
        "split": args.split,
        "n": result.n,
        "n_images_with_change": result.n_images_with_change,
        "threshold": threshold,
        "tile_size": tile_size,
        "device": device,
        "environment": env,
        "metrics": result.to_dict(),
        "per_image_change_fraction": [round(f, 6) for f in fractions],
        "change_fraction_quantiles": {k: round(v, 6) for k, v in quantiles.items()},
    }
    if args.limit is not None:
        payload["smoke_run"] = True
        payload["smoke_run_note"] = (
            f"--limit {args.limit}: this is a smoke run, not a result"
        )

    out_path = out_dir / "eval_result.json"
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    hr("WROTE")
    print(f"  {out_path}")
    if args.limit is not None:
        print()
        print(f"  [!] --limit {args.limit}: this is a SMOKE run, not a result.")
    hr("=")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
