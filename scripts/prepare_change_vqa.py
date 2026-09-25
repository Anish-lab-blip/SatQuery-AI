"""R-02 preparation: scene targets, change features and question features.

THIN CLI, like `scripts/train_change.py`: everything that computes lives in
`training/change_vqa/`; this file parses arguments, reports the accounting and
returns an exit code.

WHAT IT PRODUCES
----------------
    <out>/scene_targets.jsonl      class-wise change targets from label1/label2
    <out>/scene_manifest.jsonl     one record per (native split, scene)
    <out>/change_features.npz      frozen STANet features, keyed by scene_key
    <out>/text_features_<split>.npz question features, keyed by question_id
    <out>/prepare_record.json      provenance for all of the above

THE RECORD IS CUMULATIVE, BECAUSE PREPARATION IS MULTI-PASS
-----------------------------------------------------------
`prepare_record.json` is documented as the provenance of every input tensor, but
a full preparation is several invocations with different `--skip-*` flags: the
Kaggle notebook builds targets, then change features, then question features, and
finally re-runs over all four splits with `--skip-text` to extend the targets.
Each pass used to overwrite the record, so the last pass — which skipped the
question features — produced a record with no `text_features` at all, and a
reviewer reading it would conclude the question features had never been built.

Sections a pass does not write are therefore carried forward from the previous
record. A single-artifact section (`targets`, `change_features`, `manifest`) keeps
the last pass that wrote it; `text_features` is a per-split mapping and is merged,
so all four splits survive. The record names which sections were carried, and
`written_this_pass` / `carried_forward` make the distinction explicit rather than
leaving a reader to guess which run produced a given field.

WHY THIS IS A SEPARATE, CACHED STEP
-----------------------------------
The plan budgets <= 3 h on T4x2 for change-VQA training. Extracting STANet
features for 2,968 scenes is 5,936 detector forwards; training the head over
65,967 questions is seconds. Caching the features is therefore not an
optimisation, it is what makes the budget reachable at all — and it is what lets
the head be retrained without paying for the backbone again.

Feature extraction is resumable: an existing cache is read and only the missing
scenes are extracted, so a run interrupted at scene 2,000 does not restart.

THE DETECTOR CHECKPOINT IS REQUIRED BY DEFAULT
----------------------------------------------
`--allow-untrained-detector` must be passed explicitly to build features from an
untrained STANet. Without it a missing checkpoint is a hard error. The reason is
that `build_change_feature_extractor` degrades gracefully — which is right for
serving and wrong here: a cache built from an untrained detector still trains, still
evaluates, and produces a number, and the only thing distinguishing it from the
real capability is whether someone remembered to check. The flag makes that a
decision rather than an accident, and the cache records the answer either way.

EXIT CODES
----------
    0   preparation completed (or --dry-run validated what it would do)
    2   the dataset is missing, empty, or fails the split-integrity check
    3   the run started and raised a typed error
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

from core.config import load_config  # noqa: E402  (path set above)
from core.errors import SatQueryError  # noqa: E402
from training.change_vqa.dataset import (  # noqa: E402
    EXPECTED_SPLIT_SIZES,
    ChangeVQAError,
    build_scene_manifest,
    compute_scene_targets_cached,
    load_change_vqa_records,
    verify_split_integrity,
    write_manifest,
    write_scene_targets,
)
from training.change_vqa.features import (  # noqa: E402
    CHANGE_FEATURE_DIM,
    DEFAULT_IMAGE_SIZE,
    TEXT_FEATURE_DIM,
    ChangeFeatureCache,
    TextFeatureCache,
    TextFeatureExtractor,
    build_change_feature_extractor,
    extract_change_features,
    text_spec_hash,
)

DEFAULT_OUT = REPO_ROOT / "artifacts" / "change_vqa"
DEFAULT_CHANGE_CHECKPOINT = (
    REPO_ROOT / "artifacts" / "change" / "levir_change_v001" / "head.pt"
)

EXIT_OK = 0
EXIT_DATA = 2
EXIT_ERROR = 3

__all__ = [
    "CHANGE_FEATURE_DIM",
    "TEXT_FEATURE_DIM",
    "ChangeFeatureCache",
    "TextFeatureCache",
    "build_change_feature_extractor",
    "extract_change_features",
    "build_parser",
    "main",
]

#: Sections of `prepare_record.json` that a pass may write, and how a later pass
#: that SKIPS them must treat the previous record's copy.
#:
#:   "replace"  a single artifact for the whole directory: the last pass that
#:              wrote it is the one that describes the file on disk
#:   "merge"    a per-split mapping: union the keys, this pass's entries winning
_PROVENANCE_SECTIONS: dict[str, str] = {
    "manifest": "replace",
    "targets": "replace",
    "change_features": "replace",
    "text_features": "merge",
}


def _carry_forward(record_path: Path, result: dict) -> dict:
    """Make `prepare_record.json` cumulative instead of last-pass-wins.

    Preparation is deliberately multi-pass, so a record written by one pass is
    not the provenance of the whole directory. Carrying forward the sections this
    pass skipped is what makes the documented claim — "provenance for all of the
    above" — true, and it is the difference between a reviewer seeing the text
    spec of every split and seeing none of them.
    """
    written = [name for name in _PROVENANCE_SECTIONS if name in result]
    carried: list[str] = []

    if record_path.exists():
        try:
            previous = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = None  # an unreadable record is not a reason to fail
        if isinstance(previous, dict):
            for name, mode in _PROVENANCE_SECTIONS.items():
                if name not in previous:
                    continue
                if mode == "replace":
                    if name not in result:
                        result[name] = previous[name]
                        carried.append(name)
                    continue
                old = previous[name]
                if not isinstance(old, dict):
                    continue
                fresh = result.get(name) or {}
                merged = dict(old)
                merged.update(fresh)
                if merged:
                    result[name] = merged
                # The section survives for the splits the previous pass built,
                # even though this pass rewrote it for its own splits.
                if any(key not in fresh for key in old):
                    carried.append(name)

    result["written_this_pass"] = written
    if carried:
        result["carried_forward"] = carried
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/prepare_change_vqa.py",
        description=(
            "Build the R-02 change-VQA inputs: label-derived scene targets, "
            "frozen STANet change features, and MiniLM question features."
        ),
    )
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data" / "cdvqa"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--splits", nargs="+", default=["Train", "Val"],
        help="native splits to prepare. Test/Test2 share scenes with each other "
             "and are only needed for evaluation.",
    )
    parser.add_argument(
        "--change-checkpoint", default=str(DEFAULT_CHANGE_CHECKPOINT),
        help="trained STANet checkpoint; the encoder is frozen and never updated",
    )
    parser.add_argument(
        "--allow-untrained-detector", action="store_true",
        help="permit a missing checkpoint to degrade to an untrained detector. "
             "The cache records this, and any result from it must be reported "
             "as untrained.",
    )
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--device", default=None,
                        help="default: config.device_preference")
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="prepare at most N scenes (smoke runs)")
    parser.add_argument("--limit-per-split", type=int, default=None,
                        help="prepare at most N scenes from EACH split. Use this "
                             "rather than --limit for a smoke run: --limit alone "
                             "takes only the first split's scenes, leaving no "
                             "validation scenes to select on.")
    parser.add_argument("--skip-targets", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--skip-manifest", action="store_true")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore an existing feature cache and re-extract")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true",
                        help="load and validate everything, extract nothing")
    return parser


def _resolve_device(config, requested: str | None) -> str:
    import torch

    if requested:
        device = requested
    else:
        device = config.device_preference
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ChangeVQAError(
            f"device {device!r} was requested but CUDA is unavailable; pass "
            f"--device cpu to run on CPU rather than silently switching"
        )
    return device


def _scenes_from_records(
    records, limit: int | None, limit_per_split: int | None = None
):
    """Unique (scene_key, t1_path, t2_path) triples, in first-seen order.

    Test and Test2 are the same scenes asked about twice, so keying on
    `scene_key` collapses them to one extraction — which is correct, because the
    features depend on the imagery, not on the questions.

    `limit_per_split` exists because `limit` alone takes the first N scenes of the
    *first* split, since Train is loaded first. A smoke run built that way has no
    validation scenes at all, so it cannot exercise selection, early stopping or
    checkpointing — the parts most likely to be wrong. Taking N from each split
    costs nothing and makes a small run a real run.
    """
    seen: dict[str, tuple[str, str, str]] = {}
    per_split: dict[str, int] = {}
    for record in records:
        if record.scene_key in seen:
            continue
        if limit_per_split is not None:
            taken = per_split.get(record.split, 0)
            if taken >= limit_per_split:
                continue
            per_split[record.split] = taken + 1
        seen[record.scene_key] = (
            record.scene_key, record.t1_path, record.t2_path
        )
        if limit is not None and len(seen) >= limit:
            break
    return list(seen.values())


def prepare(args: argparse.Namespace) -> dict:
    import numpy as np

    started = time.monotonic()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise ChangeVQAError(f"data root not found: {data_root}")

    config = load_config(args.config)
    device = _resolve_device(config, args.device)

    print(f"data root            : {data_root}")
    print(f"out dir              : {out_dir}")
    print(f"splits               : {args.splits}")
    print(f"device               : {device}")
    print(f"image size           : {args.image_size}")
    print(f"config hash          : {config.hash}")
    print()

    # -- records ---------------------------------------------------------
    records = load_change_vqa_records(
        data_root, splits=tuple(args.splits), require_images=True
    )
    if not records:
        raise ChangeVQAError(
            f"no change-VQA records resolved from {data_root} for "
            f"{args.splits}; the annotations or the imagery are missing"
        )
    integrity = verify_split_integrity(
        records,
        expect_full_split=(args.limit is None and args.limit_per_split is None),
        expected_splits=tuple(args.splits),
    )
    print("split integrity")
    for split, (exp_s, exp_q) in EXPECTED_SPLIT_SIZES.items():
        got_s = integrity["scenes_per_split"].get(split, 0)
        got_q = integrity["questions_per_split"].get(split, 0)
        if got_q or got_s:
            print(f"  {split:<7} scenes={got_s:>5} questions={got_q:>6} "
                  f"(corpus {exp_s}/{exp_q})")
    print(f"  is_clean             : {integrity['is_clean']}")
    if not integrity["is_clean"]:
        raise ChangeVQAError(
            "split integrity check failed, so the (scene, question, answer) "
            f"units are not trustworthy: {integrity['size_errors']} "
            f"{integrity['leakage']}"
        )

    scenes = _scenes_from_records(records, args.limit, args.limit_per_split)
    print(f"  scenes to prepare    : {len(scenes)}")
    selected_scene_keys = {triple[0] for triple in scenes}

    result: dict = {
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "splits": list(args.splits),
        "device": device,
        "image_size": args.image_size,
        "config_hash": config.hash,
        "n_records": len(records),
        "n_scenes": len(scenes),
        "integrity": integrity,
    }

    if args.dry_run:
        print("\n--dry-run: validated; nothing extracted, nothing written")
        result["dry_run"] = True
        return result

    # -- manifest --------------------------------------------------------
    if not args.skip_manifest:
        print("\nmanifest")
        manifest = build_scene_manifest(
            data_root, splits=tuple(args.splits), require_images=True
        )
        path = write_manifest(manifest, out_dir / "scene_manifest.jsonl")
        print(f"  {len(manifest.records)} records -> {path.name}")
        result["manifest"] = {
            "path": str(path), "records": len(manifest.records)
        }

    # -- targets ---------------------------------------------------------
    if not args.skip_targets:
        print("\ntargets (from label1/label2)")
        memo: dict = {}
        targets = []
        for index, (scene_key, _t1, _t2) in enumerate(scenes, start=1):
            file_name = f"{scene_key}.png"
            targets.append(
                compute_scene_targets_cached(data_root, file_name, memo)
            )
            if args.progress_every and index % args.progress_every == 0:
                print(f"  {index}/{len(scenes)}")
        path = write_scene_targets(targets, out_dir / "scene_targets.jsonl")
        mean_total = sum(t.total_changed for t in targets) / len(targets)
        print(f"  {len(targets)} scenes -> {path.name} "
              f"(mean total_changed {mean_total:.4f})")
        result["targets"] = {
            "path": str(path),
            "scenes": len(targets),
            "mean_total_changed": round(mean_total, 6),
        }

    # -- change features -------------------------------------------------
    if not args.skip_features:
        print("\nchange features (frozen STANet)")
        checkpoint = Path(args.change_checkpoint)
        if not checkpoint.exists() and not args.allow_untrained_detector:
            raise ChangeVQAError(
                f"change checkpoint not found: {checkpoint}. Refusing to build a "
                f"feature cache from an untrained detector, because a cache like "
                f"that still trains and still reports a number. Pass "
                f"--allow-untrained-detector if an untrained representation is "
                f"genuinely what you want."
            )
        extractor = build_change_feature_extractor(
            config,
            checkpoint_path=checkpoint if checkpoint.exists() else None,
            device=device,
            image_size=args.image_size,
        )
        print(f"  detector trained     : {extractor.trained}")
        print(f"  feature spec         : {extractor.spec_hash} "
              f"({CHANGE_FEATURE_DIM}-d)")

        cache_path = out_dir / "change_features.npz"
        existing: ChangeFeatureCache | None = None
        if cache_path.exists() and not args.no_resume:
            try:
                existing = ChangeFeatureCache.read(
                    cache_path, expect_spec_hash=extractor.spec_hash
                )
                print(f"  resuming             : {len(existing)} scenes cached")
            except Exception as exc:  # noqa: BLE001
                print(f"  existing cache unusable ({type(exc).__name__}), "
                      f"re-extracting")
                existing = None

        pending = [
            triple for triple in scenes
            if existing is None or not existing.has(triple[0])
        ]
        print(f"  to extract           : {len(pending)} scenes")

        if pending:
            def on_progress(done: int, scene_key: str) -> None:
                if args.progress_every and done % args.progress_every == 0:
                    print(f"  {done}/{len(pending)} ({scene_key})")

            fresh = extract_change_features(
                extractor, pending, on_progress=on_progress
            )
            failures = getattr(fresh, "failures", [])
            if failures:
                print(f"  WARNING: {len(failures)} scene(s) failed to extract")
                for failure in failures[:5]:
                    print(f"    {failure['scene_key']}: {failure['error']}")

            if existing is not None and len(existing):
                keys = list(existing.scene_keys) + list(fresh.scene_keys)
                features = np.concatenate(
                    [existing.features, fresh.features], axis=0
                ).astype(np.float32)
                merged = ChangeFeatureCache(
                    scene_keys=tuple(keys),
                    features=features,
                    spec_hash=extractor.spec_hash,
                    extractor_config=extractor.config_dict(),
                )
            else:
                merged = fresh
        else:
            merged = existing

        merged.write(cache_path)
        print(f"  {len(merged)} scenes -> {cache_path.name} "
              f"({features_bytes(merged)} bytes)")
        result["change_features"] = {
            "path": str(cache_path),
            "scenes": len(merged),
            "spec_hash": merged.spec_hash,
            "detector_trained": bool(extractor.trained),
            "checkpoint": str(checkpoint) if checkpoint.exists() else None,
            "extractor_config": extractor.config_dict(),
        }

    # -- text features ---------------------------------------------------
    if not args.skip_text:
        print("\ntext features (frozen MiniLM)")
        encoder = TextFeatureExtractor.build(config, device=device)
        spec = text_spec_hash(
            encoder=str(config.get("router.model", "")),
            revision=str(config.get("router.revision", "")) or None,
        )
        print(f"  encoder              : {config.get('router.model')}")
        print(f"  text spec            : {spec} ({TEXT_FEATURE_DIM}-d)")
        result["text_features"] = {}
        empty_text_splits: list[str] = []
        limited = args.limit is not None or args.limit_per_split is not None
        if limited:
            print(f"  restricted to the {len(selected_scene_keys)} selected scenes "
                  f"(a full prepare encodes every question)")
        for split in args.splits:
            rows = [r for r in records if r.split == split]
            if limited:
                rows = [r for r in rows if r.scene_key in selected_scene_keys]
            if not rows:
                # No rows means NO cache file is written. The evaluator treats a
                # requested split with no cache as an input error rather than
                # skipping it, so say so here, where the cause is visible,
                # instead of twenty minutes into an evaluation.
                print(f"  {split:<7} {'-':>6}           "
                      f"-> NO cache written (no questions resolved)")
                empty_text_splits.append(split)
                continue
            features = encoder.encode([r.question for r in rows])
            cache = TextFeatureCache(
                question_ids=tuple(r.question_id for r in rows),
                features=features,
                spec_hash=spec,
            )
            path = cache.write(out_dir / f"text_features_{split}.npz")
            print(f"  {split:<7} {len(rows):>6} questions -> {path.name}")
            result["text_features"][split] = {
                "path": str(path),
                "questions": len(rows),
                "spec_hash": spec,
            }

        if empty_text_splits:
            # Recorded, not just printed: this is the one input state that makes
            # the evaluator refuse to run, so it belongs in the provenance file
            # a reviewer reads.
            result["text_features_empty_splits"] = list(empty_text_splits)
            print()
            print(f"WARNING: no question-feature cache was written for "
                  f"{empty_text_splits} -- no questions resolved for them. "
                  f"`evaluate_change_vqa.py --splits {' '.join(empty_text_splits)}` "
                  f"will refuse to run rather than score fewer splits than "
                  f"requested. Check that the data root holds those splits' "
                  f"annotations.")

    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    result["python"] = sys.version.split()[0]
    result["platform"] = platform.platform()
    result["argv"] = sys.argv[1:]
    record_path = out_dir / "prepare_record.json"
    result = _carry_forward(record_path, result)
    record_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(f"\nprepare record       : {record_path}")
    print(f"elapsed              : {result['elapsed_seconds']:.1f}s")
    return result


def features_bytes(cache: ChangeFeatureCache) -> int:
    import numpy as np

    return int(cache.features.astype(np.float32).nbytes)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prepare(args)
    except (ChangeVQAError, SatQueryError) as exc:
        print(f"\nDATA/PREP ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_DATA
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
