"""EXPERIMENT: does 448px buy localization over 224px?

The gated experiment required by finding C-5. One question, one measurement:

    At a FIXED zero-shot method, does doubling the RemoteCLIP input
    resolution improve grounding IoU / Recall@0.5 enough to justify 4x the
    tokens and 16x the attention cost?

Nothing here trains. It measures the ENCODER, so it can run before the
grounding head exists. The head's own resolution is a separate decision made
after this one.

    # validate the pipeline locally on a synthetic fixture (no quality claim)
    python scripts/exp_grounding_resolution.py --synthetic 24

    # a fast real check
    python scripts/exp_grounding_resolution.py --vrsbench /path --limit 200

    # the deciding run: every record, no sampling argument
    python scripts/exp_grounding_resolution.py --vrsbench /path --all

Reports, per resolution:
    mean best IoU, Recall@0.5, mean matched IoU
    a recall ladder at IoU {0.10, 0.25, 0.50}  (see the degeneracy note below)
    encoder latency per image (mean and p90)
    peak VRAM (GPU only; None on CPU)

Writes artifacts/grounding/resolution_experiment.json plus one JSONL per
resolution holding a line per sample. The JSONL is the source of truth: every
aggregate is recomputable from it, so the analysis can be redone without
re-running the encoder, and a run interrupted at sample 14,000 does not lose
the 13,999 that finished.

A KNOWN MEASURED DEGENERACY, reported rather than papered over:

    At n=12 the first real run returned Recall@0.5 = 0.0000 at BOTH
    resolutions, with mean best IoU 0.0596 (224) and 0.0912 (448).

    A token-grid argmax box is 1/7 of the image at 224 and 1/14 at 448. Most
    VRSBench objects are not that size, so IoU 0.5 is essentially unreachable
    for this method at either resolution and the pre-registered primary
    criterion cannot discriminate. The ladder is reported so the decision has
    a measurable axis alongside the pre-registered one; the verdict rule
    itself is UNCHANGED and still keys on Recall@0.5 and best IoU.

Exit codes:
    0  measurement completed (any verdict)
    1  measurement completed but no usable boxes were produced
    2  could not run (missing data, failed load)
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "artifacts" / "grounding"

#: The two resolutions under test. Both measured to load correctly by the
#: Phase 7 contract probe. Neither is a default.
RESOLUTIONS: tuple[int, ...] = (224, 448)

#: Below this many samples the verdict is reported INCONCLUSIVE rather than
#: guessed at. A difference smaller than sampling noise is not evidence.
MIN_SAMPLES_FOR_VERDICT = 30

#: Practical margin, chosen up front so it cannot be reinterpreted after the
#: fact. Large enough that noise on a modest fixture should not cross it.
IMPROVEMENT_MARGIN = 0.05

#: Recall is reported at every one of these. 0.5 is the pre-registered
#: criterion and is retained verbatim; 0.10 and 0.25 exist because the first
#: real run measured Recall@0.5 = 0.0 at both resolutions, which makes it
#: unable to discriminate. Adding a reporting ladder does not change the rule.
THRESHOLD_LADDER: tuple[float, ...] = (0.10, 0.25, 0.50)

#: Buckets for the per-sample best-IoU histogram, so a mean cannot hide a
#: bimodal distribution. Upper bound is exclusive except for the last bucket.
IOU_BUCKETS: tuple[float, ...] = (0.0, 0.10, 0.25, 0.50, 0.75, 1.01)


def hr(title: str = "") -> None:
    print("-" * 70)
    if title:
        print(title)
        print("-" * 70)


def environment_report(device: str) -> dict:
    """Print and return the environment facts a run must record."""
    facts: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        facts["torch"] = torch.__version__
        facts["cuda_available"] = bool(torch.cuda.is_available())
        facts["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            facts["gpu_name"] = props.name
            facts["gpu_total_vram_mb"] = round(props.total_memory / (1024 ** 2), 1)
            facts["gpu_count"] = torch.cuda.device_count()
            facts["compute_capability"] = f"{props.major}.{props.minor}"
            if props.major == 7 and props.minor == 5:
                facts["precision_note"] = "T4 (SM 7.5): fp16 only, no bf16 tensor cores"
    except Exception as exc:  # noqa: BLE001
        facts["torch"] = f"unavailable ({type(exc).__name__})"
    try:
        import open_clip

        facts["open_clip"] = open_clip.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import transformers

        facts["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        pass
    facts["requested_device"] = device
    return facts


@dataclass
class ReferringSample:
    """One (image, phrase, ground-truth box) triple in 0-1 coordinates."""

    sample_id: str
    image: object
    phrase: str
    box: list[float]
    image_path: Path | None = None


def synthetic_samples(n: int, seed: int = 42) -> list[ReferringSample]:
    """A controlled fixture: a coloured rectangle on a textured background.

    Pipeline validation ONLY. RemoteCLIP cannot ground "the red square" from a
    synthetic texture, so the IoU numbers here say nothing about real imagery.
    What this DOES prove is that images flow through, boxes come back in range,
    the metric computes, and neither resolution crashes.
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    samples: list[ReferringSample] = []

    for i in range(n):
        size = 512
        yy, xx = np.mgrid[0:size, 0:size]
        background = ((xx * 0.5 + yy * 0.5) % 256).astype(np.uint8)
        canvas = np.stack([background, background // 2, 255 - background], axis=-1)

        row = i % 4
        col = (i // 4) % 4
        cell = size // 4
        y0, x0 = row * cell + cell // 4, col * cell + cell // 4
        y1, x1 = y0 + cell // 2, x0 + cell // 2

        patch = canvas[y0:y1, x0:x1].copy()
        patch[:, :, 0] = 255
        patch[:, :, 1] = rng.integers(0, 60, patch[:, :, 1].shape)
        patch[:, :, 2] = 40
        canvas[y0:y1, x0:x1] = patch

        samples.append(
            ReferringSample(
                sample_id=f"synthetic_{i:04d}",
                image=Image.fromarray(canvas),
                phrase="a red square",
                box=[x0 / size, y0 / size, x1 / size, y1 / size],
            )
        )

    return samples


@dataclass
class ResolutionResult:
    resolution: int
    grid: tuple[int, int]
    n_samples: int
    n_with_boxes: int
    mean_best_iou: float
    recall_at_50: float
    mean_matched_iou: float
    latency_mean_ms: float
    latency_p90_ms: float
    peak_vram_mb: float | None
    n_patches: int
    recall: dict[str, float] = field(default_factory=dict)
    iou_histogram: dict[str, int] = field(default_factory=dict)
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    resumed: bool = False

    def to_dict(self) -> dict:
        return {
            "resolution": self.resolution,
            "grid": list(self.grid),
            "n_patches": self.n_patches,
            "n_samples": self.n_samples,
            "n_with_boxes": self.n_with_boxes,
            "failed": self.failed,
            "resumed": self.resumed,
            "mean_best_iou": round(self.mean_best_iou, 4),
            "recall_at_0.5": round(self.recall_at_50, 4),
            "recall": {k: round(v, 4) for k, v in self.recall.items()},
            "iou_histogram": dict(self.iou_histogram),
            "mean_matched_iou": round(self.mean_matched_iou, 4),
            "latency_mean_ms": round(self.latency_mean_ms, 1),
            "latency_p90_ms": round(self.latency_p90_ms, 1),
            "peak_vram_mb": (
                None if self.peak_vram_mb is None else round(self.peak_vram_mb, 1)
            ),
            "errors": self.errors[:5],
        }


def _empty(resolution: int, n: int, failed: int, errors: list[str]) -> ResolutionResult:
    return ResolutionResult(
        resolution=resolution, grid=(0, 0), n_samples=n, n_with_boxes=0,
        mean_best_iou=0.0, recall_at_50=0.0, mean_matched_iou=0.0,
        latency_mean_ms=0.0, latency_p90_ms=0.0, peak_vram_mb=None,
        n_patches=0, failed=failed, errors=errors,
    )


def _open_image(sample: ReferringSample):
    """Return (image, opened_here). Caller closes when opened_here is True.

    Holding every image open at once costs ~12.7 GB across the full eval
    split, so non-synthetic runs load one at a time.
    """
    if sample.image is not None:
        return sample.image, False
    if sample.image_path is None:
        raise RuntimeError(f"sample {sample.sample_id} has neither image nor image_path")
    from PIL import Image

    return Image.open(sample.image_path).convert("RGB"), True


def _aggregate(records: list[dict], resolution: int, n_patches: int,
               grid: tuple[int, int], peak_vram: float | None,
               failed: int, errors: list[str], resumed: bool) -> ResolutionResult:
    """Recompute every aggregate from per-sample records.

    Self-contained on purpose: a resumed run rebuilds the result from the JSONL
    alone, so a partial run does not need a second file kept in sync with it.
    """
    if not records:
        return _empty(resolution, 0, failed, errors)

    best = np.asarray([r["best_iou"] for r in records], dtype=np.float64)
    lat = np.asarray([r["latency_ms"] for r in records], dtype=np.float64)
    with_targets = [r for r in records if r["n_targets"] > 0]

    recall: dict[str, float] = {}
    for thr in THRESHOLD_LADDER:
        key = f"{thr:.2f}"
        if with_targets:
            recall[key] = float(np.mean([r["recall"][key] for r in with_targets]))
        else:
            recall[key] = 0.0

    histogram: dict[str, int] = {}
    for lo, hi in zip(IOU_BUCKETS[:-1], IOU_BUCKETS[1:]):
        label = f"{lo:.2f}-{hi:.2f}"
        histogram[label] = int(np.sum((best >= lo) & (best < hi)))

    matched = (
        float(np.mean([r["matched_iou"] for r in with_targets])) if with_targets else 0.0
    )

    return ResolutionResult(
        resolution=resolution,
        grid=grid,
        n_samples=len(records),
        n_with_boxes=sum(1 for r in records if r["n_boxes"] > 0),
        mean_best_iou=float(best.mean()),
        recall_at_50=recall.get("0.50", 0.0),
        mean_matched_iou=matched,
        latency_mean_ms=float(lat.mean()),
        latency_p90_ms=float(np.percentile(lat, 90)),
        peak_vram_mb=peak_vram,
        n_patches=n_patches,
        recall=recall,
        iou_histogram=histogram,
        failed=failed,
        errors=errors,
        resumed=resumed,
    )


def measure_resolution(
    resolution: int,
    samples: list[ReferringSample],
    delta: float,
    top_k: int,
    device: str,
    jsonl_path: Path,
    resume: bool = False,
    checkpoint_path: str | None = None,
) -> ResolutionResult:
    """Run the zero-shot baseline at one resolution, streaming per-sample rows."""
    from core.config import load_config
    from evaluation.metrics.grounding import score_one
    from specialists.grounding.inference import candidate_boxes, ground_phrase
    from specialists.grounding.remoteclip import build_encoder

    config = load_config()

    # -- resume ------------------------------------------------------------
    if resume and jsonl_path.exists():
        records = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(records) >= len(samples):
            print(f"  RESUMED: {len(records)} rows already in {jsonl_path.name}")
            return _aggregate(
                records, resolution, (resolution // 32) ** 2,
                (resolution // 32, resolution // 32), None, 0, [], True,
            )
        print(f"  resume requested but only {len(records)}/{len(samples)} rows exist; "
              f"re-running from scratch")
        jsonl_path.unlink(missing_ok=True)

    print(f"  loading encoder at {resolution}px ...")
    t0 = time.time()
    try:
        encoder = build_encoder(
            config,
            resolution=resolution,
            device=device,
            checkpoint_path=checkpoint_path,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  LOAD FAILED: {type(exc).__name__}: {exc}")
        return _empty(resolution, len(samples), len(samples), [f"load: {exc}"])

    print(f"  loaded in {time.time() - t0:.1f}s  {encoder.describe()}")

    if device == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats()

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    failed = 0
    errors: list[str] = []
    started = time.time()

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for index, sample in enumerate(samples):
            image, opened = _open_image(sample)
            try:
                t0 = time.time()
                candidates = ground_phrase(
                    encoder, image, sample.phrase, delta=delta, top_k=top_k
                )
                boxes = candidate_boxes(candidates)
                latency_ms = (time.time() - t0) * 1000.0
            except Exception as exc:  # noqa: BLE001
                failed += 1
                errors.append(f"{sample.sample_id}: {type(exc).__name__}: {exc}")
                continue
            finally:
                if opened:
                    image.close()

            score = score_one(boxes, [sample.box], thresholds=THRESHOLD_LADDER)
            record = {
                "sample_id": sample.sample_id,
                "n_preds": score.n_preds,
                "n_targets": score.n_targets,
                "n_boxes": len(boxes),
                "best_iou": round(score.best_iou, 6),
                "matched_iou": round(score.mean_matched_iou, 6),
                "recall": {f"{t:.2f}": round(score.recall[t], 6)
                           for t in THRESHOLD_LADDER},
                "latency_ms": round(latency_ms, 4),
            }
            records.append(record)
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            if (index + 1) % 500 == 0:
                fh.flush()
                elapsed = time.time() - started
                rate = (index + 1) / elapsed
                remaining = (len(samples) - index - 1) / rate if rate > 0 else 0
                print(f"    {index + 1}/{len(samples)}  "
                      f"{rate:.1f}/s  eta {remaining / 60:.1f} min")

    peak = None
    if device == "cuda":
        import torch

        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return _aggregate(
        records,
        resolution,
        encoder.grid_size ** 2,
        (encoder.grid_size, encoder.grid_size),
        peak,
        failed,
        errors,
        False,
    )


def verdict_for(coarse: ResolutionResult, fine: ResolutionResult) -> dict:
    """Decide whether the finer grid earned its cost.

       448 wins  if Recall@0.5 improves by >= IMPROVEMENT_MARGIN absolute
                 OR mean best IoU improves by >= IMPROVEMENT_MARGIN absolute
       224 wins  otherwise
       INCONCLUSIVE below MIN_SAMPLES_FOR_VERDICT samples

    UNCHANGED from the pre-registered rule. The threshold ladder added to the
    report does not enter this decision.
    """
    n = min(coarse.n_samples, fine.n_samples)
    recall_gain = fine.recall_at_50 - coarse.recall_at_50
    iou_gain = fine.mean_best_iou - coarse.mean_best_iou

    if n < MIN_SAMPLES_FOR_VERDICT:
        label = "INCONCLUSIVE"
        reason = (
            f"only {n} samples (need {MIN_SAMPLES_FOR_VERDICT}); a "
            f"{abs(recall_gain):.3f} recall difference is within what sampling "
            f"noise can produce at this n"
        )
    elif recall_gain >= IMPROVEMENT_MARGIN or iou_gain >= IMPROVEMENT_MARGIN:
        label = "448 WINS"
        reason = (
            f"recall {recall_gain:+.3f}, best IoU {iou_gain:+.3f} "
            f"(margin +{IMPROVEMENT_MARGIN})"
        )
    else:
        label = "224 WINS"
        reason = (
            f"recall {recall_gain:+.3f}, best IoU {iou_gain:+.3f} -- neither "
            f"reaches the +{IMPROVEMENT_MARGIN} margin, so the 4x tokens and "
            f"16x attention are not earned"
        )

    return {
        "verdict": label,
        "reason": reason,
        "recall_gain_448_over_224": round(recall_gain, 4),
        "best_iou_gain_448_over_224": round(iou_gain, 4),
        "latency_ratio_448_over_224": (
            round(fine.latency_mean_ms / coarse.latency_mean_ms, 2)
            if coarse.latency_mean_ms > 0 else None
        ),
        "n_samples": n,
        "verdict_rule": (
            f"448 if Recall@0.5 or mean best IoU improves by >= "
            f"{IMPROVEMENT_MARGIN} absolute; INCONCLUSIVE below "
            f"{MIN_SAMPLES_FOR_VERDICT} samples"
        ),
        "rule_changed_since_preregistration": False,
    }


def degeneracy_note(results: dict[int, ResolutionResult]) -> list[str]:
    """Flag metrics that cannot discriminate, rather than letting them decide."""
    notes: list[str] = []
    coarse, fine = results.get(224), results.get(448)
    if coarse is None or fine is None:
        return notes

    if coarse.recall_at_50 == 0.0 and fine.recall_at_50 == 0.0:
        notes.append(
            "Recall@0.5 is 0.0000 at BOTH resolutions. The pre-registered "
            "primary criterion cannot discriminate for this method, so the "
            "verdict rests on mean best IoU alone. See the recall ladder."
        )
    for label, r in (("224", coarse), ("448", fine)):
        ladder = {k: v for k, v in r.recall.items() if k != "0.50"}
        if ladder and all(v == 0.0 for v in ladder.values()):
            notes.append(
                f"Recall at every threshold below 0.50 is also 0.0 at {label}px; "
                f"the token grid is too coarse for this object population."
            )
    return notes


def main() -> int:
    ap = argparse.ArgumentParser(description="Grounding resolution experiment")
    ap.add_argument("--synthetic", type=int, default=None,
                    help="use N synthetic samples (pipeline validation only)")
    ap.add_argument("--vrsbench", type=str, default=None,
                    help="path to a VRSBench download (the real measurement)")
    ap.add_argument("--limit", type=int, default=200,
                    help="max samples to use (ignored with --all)")
    ap.add_argument("--all", action="store_true",
                    help="use every record in the eval split; no sampling")
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default=None,
                    help="suffix for the output filenames, to keep runs apart")
    ap.add_argument("--resume", action="store_true",
                    help="skip a resolution whose per-sample JSONL is complete")
    ap.add_argument("--checkpoint", default=None,
                    help="path to a local RemoteCLIP .pt; skips the Hub fetch. "
                         "Kaggle runs pass the dataset input path so the run "
                         "does not depend on Hub availability.")
    args = ap.parse_args()

    if args.synthetic is None and args.vrsbench is None:
        print("specify --synthetic N or --vrsbench PATH", file=sys.stderr)
        return 2

    from core.config import load_config
    from specialists.grounding.inference import DEFAULT_DELTA

    config = load_config()
    device = args.device or config.device_preference
    delta = args.delta if args.delta is not None else DEFAULT_DELTA
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    synthetic_mode = args.synthetic is not None
    tag = f"_{args.tag}" if args.tag else ""

    print("=" * 70)
    print("GROUNDING RESOLUTION EXPERIMENT")
    print("=" * 70)
    print(f"source      : {'synthetic' if synthetic_mode else args.vrsbench}")
    print(f"limit       : {'ALL' if args.all else args.limit}")
    print(f"delta       : {delta}")
    print(f"top_k       : {args.top_k}")
    print(f"device      : {device}")
    print(f"config hash : {config.hash}")
    print(f"thresholds  : {list(THRESHOLD_LADDER)}")
    print(f"checkpoint  : {args.checkpoint or 'hub fetch (pinned revision)'}")
    print()

    hr("ENVIRONMENT")
    env = environment_report(device)
    for key in sorted(env):
        print(f"  {key:<22} {env[key]}")
    print()

    # -- samples ---------------------------------------------------------
    if synthetic_mode:
        hr("SAMPLES (synthetic)")
        print("  NOTE: RemoteCLIP cannot ground synthetic shapes. The IoU below")
        print("  measures PIPELINE HEALTH, not localization quality. The verdict")
        print("  written by this run is marked unusable_as_evidence.")
        print()
        samples = synthetic_samples(args.synthetic, seed=args.seed)
    else:
        hr("SAMPLES (VRSBench)")
        try:
            from training.data.vrsbench import load_referring_samples

            samples = load_referring_samples(
                args.vrsbench,
                limit=None if args.all else args.limit,
                seed=args.seed,
                open_images=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            hr("=")
            return 2

    print(f"  loaded {len(samples)} samples")
    if not samples:
        print("  no samples -- nothing to measure")
        hr("=")
        return 2

    # -- measure ---------------------------------------------------------
    results: dict[int, ResolutionResult] = {}
    for resolution in RESOLUTIONS:
        hr(f"RESOLUTION {resolution}")
        jsonl_path = out_dir / f"per_sample_{resolution}{tag}.jsonl"
        results[resolution] = measure_resolution(
            resolution, samples, delta, args.top_k, device, jsonl_path,
            resume=args.resume, checkpoint_path=args.checkpoint,
        )
        r = results[resolution]
        print(f"  token grid     : {r.grid[0]} x {r.grid[1]} = {r.n_patches}")
        print(f"  with boxes     : {r.n_with_boxes}/{r.n_samples}")
        print(f"  mean best IoU  : {r.mean_best_iou:.4f}")
        for thr in THRESHOLD_LADDER:
            print(f"  Recall@{thr:<5}   : {r.recall.get(f'{thr:.2f}', 0.0):.4f}")
        print(f"  matched IoU    : {r.mean_matched_iou:.4f}")
        print(f"  latency        : mean {r.latency_mean_ms:.1f}ms  "
              f"p90 {r.latency_p90_ms:.1f}ms")
        print(f"  peak VRAM      : "
              f"{'CPU run' if r.peak_vram_mb is None else f'{r.peak_vram_mb:.1f} MB'}")
        print(f"  best-IoU spread: {r.iou_histogram}")
        if r.failed:
            print(f"  failed         : {r.failed}")
        print()

    # -- verdict ---------------------------------------------------------
    hr("VERDICT  (pre-registered rule, unchanged)")
    outcome = verdict_for(results[224], results[448])
    print(f"  {outcome['verdict']}")
    print(f"  {outcome['reason']}")
    print()
    print(f"  recall@0.5 gain 448/224 : {outcome['recall_gain_448_over_224']:+.4f}")
    print(f"  bestIoU    gain 448/224 : {outcome['best_iou_gain_448_over_224']:+.4f}")
    print(f"  latency          ratio  : {outcome['latency_ratio_448_over_224']}x")
    print()

    notes = degeneracy_note(results)
    hr("DEGENERACY NOTES")
    if notes:
        for note in notes:
            print(f"  [!] {note}")
    else:
        print("  none")
    hr()

    if synthetic_mode:
        print("  THIS WAS A SYNTHETIC RUN. The verdict above is NOT usable as")
        print("  evidence about grounding quality and must not be used to freeze")
        print("  the resolution. Re-run with --vrsbench.")
        hr()

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config.hash,
        "device": device,
        "environment": env,
        "source": "synthetic" if synthetic_mode else str(args.vrsbench),
        "synthetic": synthetic_mode,
        "usable_as_evidence": not synthetic_mode,
        "n_samples": len(samples),
        "sampled": not args.all,
        "checkpoint_source": args.checkpoint or "hub",
        "delta": delta,
        "top_k": args.top_k,
        "threshold_ladder": list(THRESHOLD_LADDER),
        "results": {str(k): v.to_dict() for k, v in results.items()},
        "verdict": outcome,
        "degeneracy_notes": notes,
        "per_sample_files": {
            str(k): f"per_sample_{k}{tag}.jsonl" for k in results
        },
    }
    out_path = out_dir / f"resolution_experiment{tag}.json"
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    hr("WROTE")
    print(f"  {out_path}")
    for k in results:
        print(f"  {out_dir / f'per_sample_{k}{tag}.jsonl'}")
    hr("=")

    if results[224].n_with_boxes == 0 and results[448].n_with_boxes == 0:
        print("PIPELINE PRODUCED NO BOXES -- investigate before trusting anything.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())