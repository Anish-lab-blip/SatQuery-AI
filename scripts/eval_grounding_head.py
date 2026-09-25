"""Phase 8 evaluation — score a trained grounding head on the FROZEN eval split.

This is the script that produces the benchmark number. `train_grounding.py`
validates on a held-out slice of the TRAIN split; that number is a training
signal, not a result. This one scores all 16,159 VRSBench eval records, which
is the same population the Phase 7 zero-shot baseline was measured on.

    # score a trained head
    python scripts/eval_grounding_head.py --checkpoint <artifact>/head.pt

    # same run also re-measures the zero-shot baseline on identical samples
    python scripts/eval_grounding_head.py --checkpoint <head.pt> --with-baseline

    # baseline only, no head (reproduces the Phase 7 number)
    python scripts/eval_grounding_head.py --with-baseline

TREAT THIS DATA AS FROZEN
-------------------------
VRSBench eval is evaluation data (project sections 34, 64). Nothing here is
tuned on it, and the script is deliberately not parameterised in ways that
would let it be: thresholds come from config, prompts are fixed, and the
decode strategies reported are the two declared in training. The point of a
benchmark is that it is run, not searched.

CONFIG-DRIFT GUARD
------------------
The head checkpoint records the config hash it was trained under. If the
current config differs, the comparison is not reproducible, so this script
REFUSES to run unless --allow-config-drift is passed explicitly. Scoring a
head under a different config silently is how a result stops meaning anything.

WHY --with-baseline MATTERS
---------------------------
Phase 7 measured zero-shot at mean best IoU 0.0972. Quoting that number beside
a head's score from a different run compares two different sample sets through
two different code paths. --with-baseline recomputes zero-shot HERE, on the
SAME items, from the SAME cached features, through the SAME metric call. Only
that comparison is apples-to-apples.
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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "artifacts" / "grounding" / "remoteclip_grounding_v001"

#: The Phase 7 measured zero-shot result, for reference. --with-baseline
#: recomputes this locally rather than trusting the constant.
PHASE7_BASELINE_IOU = 0.0972
PHASE7_BASELINE_RECALL_50 = 0.0234

RECALL_THRESHOLDS: tuple[float, ...] = (0.10, 0.25, 0.50)


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
    except Exception as exc:  # noqa: BLE001
        facts["torch"] = f"unavailable ({type(exc).__name__})"
    return facts


def main() -> int:
    ap = argparse.ArgumentParser(description="Score a grounding head on VRSBench eval")
    ap.add_argument("--data-root", default=None,
                    help="VRSBench root; defaults to training/data/vrsbench")
    ap.add_argument("--checkpoint", default=None,
                    help="trained head .pt; omit to run baseline only")
    ap.add_argument("--remoteclip", default=None,
                    help="local RemoteCLIP .pt; skips the Hub fetch")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--cache-dir", default=None,
                    help="eval feature cache; kept SEPARATE from the train cache")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap eval records; for a smoke run, not a result")
    ap.add_argument("--with-baseline", action="store_true",
                    help="also score zero-shot on the same items and features")
    ap.add_argument("--allow-config-drift", action="store_true",
                    help="proceed even if the current config hash differs from "
                         "the checkpoint's (the comparison is then NOT "
                         "reproducible and the artifact says so)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--head-top-k", type=int, default=None,
                    help="override grounding.max_candidates for the head decode. "
                         "Use to match the baseline's candidate count, so a win "
                         "cannot be attributed to emitting more boxes.")
    ap.add_argument("--head-score-threshold", type=float, default=None,
                    help="override grounding.confidence_threshold for the head decode")
    ap.add_argument("--tag", default="",
                    help="suffix for the output filenames, so two decode settings "
                         "can be scored without overwriting each other")
    args = ap.parse_args()

    from core.config import load_config

    cfg = load_config()
    device = args.device or cfg.device_preference
    data_root = Path(args.data_root) if args.data_root else (
        REPO_ROOT / "training" / "data" / "vrsbench"
    )
    out_dir = Path(args.output_dir) if args.output_dir else OUT_DIR
    resolution = int(cfg.get("grounding.image_size", 224))
    grid = resolution // 32
    nms_iou = float(cfg.get("grounding.nms_iou", 0.50))
    top_k = (args.head_top_k if args.head_top_k is not None
             else int(cfg.get("grounding.max_candidates", 20)))
    score_threshold = (args.head_score_threshold
                       if args.head_score_threshold is not None
                       else float(cfg.get("grounding.confidence_threshold", 0.40)))
    print(f"head decode : top_k={top_k} score_threshold={score_threshold} "
          f"nms_iou={nms_iou}")

    print("=" * 70)
    print("PHASE 8 EVALUATION — GROUNDING HEAD ON VRSBENCH EVAL")
    print("=" * 70)
    print(f"data root   : {data_root}")
    print(f"checkpoint  : {args.checkpoint or '(none — baseline only)'}")
    print(f"resolution  : {resolution}  (grid {grid}x{grid})")
    print(f"device      : {device}")
    print(f"config hash : {cfg.hash}")
    print(f"limit       : {args.limit or 'ALL eval records'}")
    print()

    hr("ENVIRONMENT")
    env = environment_report(device)
    for key in sorted(env):
        print(f"  {key:<22} {env[key]}")
    print()

    # -- eval items --------------------------------------------------------
    from training.data.vrsbench import load_referring_samples

    hr("EVAL SET (frozen)")
    try:
        samples = load_referring_samples(
            str(data_root),
            limit=args.limit,
            seed=int(cfg.get("project.seed", 42)),
            verbose=True,
            open_images=False,
            json_name="VRSBench_EVAL_referring.json",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED to load eval set: {type(exc).__name__}: {exc}")
        return 2

    if not samples:
        print("no eval samples loaded")
        return 2

    print(f"  eval records     : {len(samples):,}")
    print(f"  unique images    : {len({s.image_path.stem for s in samples}):,}")
    print()

    # -- cache eval features (separate from train) -------------------------
    from training.grounding.dataset import (
        GroundingItem,
        attach_cached,
        cache_dir_for,
        extract_features,
        image_cache_path,
    )

    items = [
        GroundingItem(
            sample_id=s.sample_id,
            image_key=s.image_path.stem,
            phrase=s.phrase,
            box=[float(v) for v in s.box],
            source_path=s.image_path,
        )
        for s in samples
        if s.image_path is not None
    ]
    if not items:
        print("no eval items have an image path")
        return 2

    cache_base = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    # A distinct tag: eval features must never be mistaken for train features.
    cache_root = cache_dir_for(cache_base, resolution, tag="_eval")

    hr("EVAL FEATURE CACHE")
    print(f"  cache dir : {cache_root}")
    need_extract = any(
        not image_cache_path(cache_root, i.image_key).exists() for i in items
    )
    if need_extract:
        from specialists.grounding.remoteclip import build_encoder

        remoteclip = args.remoteclip
        if remoteclip is None:
            from huggingface_hub import hf_hub_download

            remoteclip = hf_hub_download(
                cfg.get("grounding.checkpoint_repo"),
                cfg.get("grounding.checkpoint_file"),
                revision=cfg.get("grounding.checkpoint_revision"),
            )
        encoder = build_encoder(cfg, resolution=resolution, device=device,
                                checkpoint_path=remoteclip)
        print(f"  encoder   : {encoder.describe()}")
        report = extract_features(items, encoder, cache_root, progress_every=500)
        print(f"  extracted : {report['image_cache_misses']:,} new, "
              f"{report['image_cache_hits']:,} cached, "
              f"{report['images_missing_source']:,} missing source, "
              f"{report['images_undecodable']:,} undecodable "
              f"({report['seconds']}s)")
    else:
        print("  cache complete — no extraction needed")

    items = attach_cached(items, cache_root)
    print(f"  scoreable items  : {len(items):,}")
    if not items:
        print("no cached eval items; extraction must have failed")
        return 2
    print()

    from evaluation.metrics.grounding import score_dataset

    results: dict[str, dict] = {}

    # -- baseline (zero-shot from the SAME cached features) ----------------
    #
    # DECODE FIDELITY. This block previously built its own single-box baseline
    # with `argmax_candidate`, while Phase 7 measured the baseline through
    # `ground_phrase` (threshold box + up to 5 local maxima). Same 16,159
    # records, same metric, same features -- different decode:
    #
    #     Phase 7 via ground_phrase  : mean best IoU 0.0972
    #     eval via argmax_candidate  : mean best IoU 0.0092
    #
    # The eval then reported "head beats zero-shot: True (+0.1123)" when the
    # matched comparison is +0.0243. Both clear the 0.02 bar, but only the
    # second is a claim about the HEAD rather than about the decode.
    #
    # Both paths now call `decode_candidates_from_features`, so they cannot
    # diverge. `delta` and `top_k` come from config, and are recorded in the
    # artifact so the baseline is reproducible rather than implied.
    if args.with_baseline or args.checkpoint is None:
        hr("ZERO-SHOT BASELINE (same items, same features, same decode as Phase 7)")
        from specialists.grounding.inference import (
            DEFAULT_DELTA,
            candidate_boxes,
            decode_candidates_from_features,
        )
        from specialists.grounding.remoteclip import EncodedImage

        baseline_delta = float(cfg.get("grounding.baseline_delta", DEFAULT_DELTA))
        baseline_top_k = int(cfg.get("grounding.baseline_top_k", 5))
        print(f"  decode   : threshold delta={baseline_delta} + top_k={baseline_top_k} local maxima")
        print(f"  grid     : {grid}x{grid}")

        t0 = time.time()
        per_image: list[tuple[list[list[float]], list[list[float]]]] = []
        n_boxes: list[int] = []
        for item in items:
            patches, text = item.load()
            encoded = EncodedImage(
                cls=patches.mean(axis=0).astype(np.float32),
                patch_tokens=patches.astype(np.float32),
                grid=(grid, grid),
                resolution=resolution,
            )
            cands = decode_candidates_from_features(
                encoded, text, delta=baseline_delta, top_k=baseline_top_k
            )
            boxes = candidate_boxes(cands)
            n_boxes.append(len(boxes))
            per_image.append((boxes, [item.box]))
        elapsed = time.time() - t0
        mean_boxes = float(np.mean(n_boxes)) if n_boxes else 0.0
        print(f"  candidates/image : mean {mean_boxes:.2f}")

        rep = score_dataset(per_image, thresholds=RECALL_THRESHOLDS)
        # Key renamed from `zero_shot_argmax`. The decode is no longer argmax-only
        # -- it is the Phase 7 multi-candidate decode (threshold box + local
        # maxima). Leaving the old name would misdescribe the number that decides
        # whether the head earned its place.
        results["zero_shot_matched"] = {
            "strategy": "zero_shot_matched",
            "decode": {
                "kind": "threshold_box_plus_local_maxima",
                "delta": baseline_delta,
                "top_k": baseline_top_k,
                "mean_candidates_per_image": round(mean_boxes, 2),
            },
            "n": rep.n_images,
            "mean_best_iou": round(rep.mean_best_iou, 4),
            "mean_matched_iou": round(rep.mean_matched_iou, 4),
            "recall": {f"{k:.2f}": round(v, 4) for k, v in rep.recall.items()},
            "seconds": round(elapsed, 1),
        }
        print(rep.summary())
        print(f"  elapsed : {elapsed:.1f}s")
        print()

    # -- trained head ------------------------------------------------------
    if args.checkpoint is not None:
        import torch

        from training.grounding.train import evaluate, load_trained_head

        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"checkpoint does not exist: {ckpt_path}")
            return 2

        raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        ckpt_cfg_hash = raw.get("config_hash")
        print(f"checkpoint config hash : {ckpt_cfg_hash}")
        print(f"current  config hash   : {cfg.hash}")
        drift = ckpt_cfg_hash not in (None, cfg.hash)
        if drift and not args.allow_config_drift:
            print()
            print("CONFIG DRIFT — refusing to score.")
            print("  The head was trained under a different configuration, so this")
            print("  comparison would not be reproducible. Re-train, or pass")
            print("  --allow-config-drift to proceed knowingly (the artifact will")
            print("  record that the result is not a frozen-config evaluation).")
            return 3
        if drift:
            print("  [!] config drift acknowledged; result is NOT a frozen-config run")
        print()

        head = load_trained_head(ckpt_path, device=device)
        print(f"  head : {head}")

        hr("TRAINED HEAD")
        for strategy in ("argmax", "threshold"):
            t0 = time.time()
            res = evaluate(
                head, items, grid=grid, device=device, strategy=strategy,
                top_k=top_k, score_threshold=score_threshold, nms_iou=nms_iou,
                batch_size=args.batch_size, in_memory=False,
            )
            elapsed = time.time() - t0
            entry = res.to_dict()
            entry["seconds"] = round(elapsed, 1)
            results[f"head_{strategy}"] = entry
            print(f"  strategy {strategy:<10} IoU {res.mean_best_iou:.4f}  "
                  f"R@.10 {res.recall.get('0.10', 0.0):.4f}  "
                  f"R@.25 {res.recall.get('0.25', 0.0):.4f}  "
                  f"R@.50 {res.recall.get('0.50', 0.0):.4f}  "
                  f"({elapsed:.1f}s, {res.latency_ms_per_image:.1f} ms/img)")
        print()

    # -- comparison --------------------------------------------------------
    #
    # THREE numbers, and only ONE pair is decode-matched:
    #
    #   zero_shot_matched   multi-candidate (threshold box + local maxima)
    #   head_argmax         1 box per image
    #   head_threshold      NMS over scoring cells, multi-box
    #
    # `head_argmax` vs `zero_shot_matched` compares a 1-box decode against a
    # ~5-box decode, which flatters the head. `head_threshold` vs
    # `zero_shot_matched` is multi-box against multi-box. Both are printed; the
    # matched pair is labelled, because quoting the unmatched one would overstate
    # the result. (The earlier run printed only the unmatched comparison and a
    # `head beats zero-shot: True` line -- that is the number this replaces.)
    hr("COMPARISON")
    baseline = results.get("zero_shot_matched")
    head_keys = [k for k in ("head_argmax", "head_threshold") if k in results]

    if baseline is not None and head_keys:
        dec = baseline["decode"]
        print(f"  baseline decode : {dec['kind']}")
        print(f"                    delta={dec['delta']} top_k={dec['top_k']} "
              f"({dec['mean_candidates_per_image']:.2f} candidates/image)")
        print()
        print(f"  {'':<18}{'zero-shot':>12}" + "".join(
            f"{k.replace('head_', 'head '):>14}" for k in head_keys
        ))
        print(f"  {'mean best IoU':<18}{baseline['mean_best_iou']:>12.4f}" + "".join(
            f"{results[k]['mean_best_iou']:>14.4f}" for k in head_keys
        ))
        for thr in RECALL_THRESHOLDS:
            key = f"{thr:.2f}"
            print(f"  {'Recall@' + key:<18}{baseline['recall'][key]:>12.4f}" + "".join(
                f"{results[k]['recall'][key]:>14.4f}" for k in head_keys
            ))
        print()
        for k in head_keys:
            d = results[k]["mean_best_iou"] - baseline["mean_best_iou"]
            print(f"  {k:<15} vs zero-shot : {d:+.4f}  "
                  f"{'beats baseline' if d > 0 else 'does not beat baseline'}")
        print()
        print("  DECODE MATCHING -- which comparison is apples-to-apples")
        print("    head_argmax     1 box/image    vs multi-candidate  -> NOT matched")
        print("    head_threshold  NMS multi-box  vs multi-candidate  -> MATCHED")
        if "head_threshold" in results:
            d_matched = (results["head_threshold"]["mean_best_iou"]
                         - baseline["mean_best_iou"])
            print()
            print(f"  decode-matched delta (head threshold vs zero-shot) : {d_matched:+.4f}")
        print()
        print(f"  Phase 7 recorded baseline (cross-check) : {PHASE7_BASELINE_IOU:.4f}")
        print(f"  this run's zero-shot, same samples      : {baseline['mean_best_iou']:.4f}")
        gap = abs(baseline["mean_best_iou"] - PHASE7_BASELINE_IOU)
        print(f"  |difference|                            : {gap:.4f}")
        if gap > 0.01:
            print("  [!] this run's baseline differs from the Phase 7 record by more")
            print("      than 0.01; the decode or the sample set still differs.")
    elif head_keys:
        for k in head_keys:
            print(f"  {k:<15} mean best IoU : {results[k]['mean_best_iou']:.4f}")
        print(f"  Phase 7 baseline : {PHASE7_BASELINE_IOU:.4f}  "
              f"(different run -- pass --with-baseline for a matched comparison)")
    elif baseline is not None:
        print(f"  baseline only; no head scored "
              f"(mean best IoU {baseline['mean_best_iou']:.4f})")
    else:
        print("  nothing scored")
    print()

    # -- artifact ----------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": cfg.hash,
        "checkpoint_config_hash": (
            ckpt_cfg_hash if args.checkpoint is not None else None
        ),
        "config_drift": bool(
            args.checkpoint is not None
            and ckpt_cfg_hash not in (None, cfg.hash)
        ),
        "frozen_config_evaluation": bool(
            args.checkpoint is not None and ckpt_cfg_hash == cfg.hash
        ),
        "device": device,
        "environment": env,
        "resolution": resolution,
        "grid": grid,
        "n_eval_records": len(items),
        "eval_cache": str(cache_root),
        "limited_run": args.limit is not None,
        # Decode settings, recorded because they CHANGE THE NUMBER. The head's
        # threshold decode emits up to `head_top_k` boxes, and mean best IoU is
        # a max over predictions -- so more boxes raises it mechanically.
        # Measured on the same frozen head: top_k=20 -> 0.2838, top_k=6 -> 0.2566.
        # Without these fields the artifact cannot say which setting produced it.
        "head_decode": {
            "top_k": top_k,
            "score_threshold": score_threshold,
            "nms_iou": nms_iou,
            "config_default_top_k": int(cfg.get("grounding.max_candidates", 20)),
        },
        "results": results,
        "phase7_reference": {
            "mean_best_iou": PHASE7_BASELINE_IOU,
            "recall_at_0.50": PHASE7_BASELINE_RECALL_50,
            "source": "docs/PHASE7_RESOLUTION_DECISION.md",
        },
    }
    out_path = out_dir / f"eval_result{args.tag}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str),
                        encoding="utf-8")

    archive = out_dir.parent / f"{out_dir.name}_eval{args.tag}.zip"
    with __import__("zipfile").ZipFile(archive, "w",
                                       __import__("zipfile").ZIP_DEFLATED) as z:
        z.write(out_path, out_path.name)
        if args.checkpoint is not None:
            z.write(Path(args.checkpoint), Path(args.checkpoint).name)

    hr("WROTE")
    print(f"  {out_path}")
    print(f"  {archive}  ({archive.stat().st_size/1024:.0f} KB)")
    if args.limit is not None:
        print()
        print(f"  [!] --limit {args.limit}: this is a SMOKE run, not a result.")
    hr("=")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())