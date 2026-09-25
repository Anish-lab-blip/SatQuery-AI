"""R-02 evaluation entry point: score a trained change-VQA head.

THIN CLI over `training/change_vqa/evaluate.py`. Loads a checkpoint, scores it on
one or more native splits, and writes a self-describing report plus qualitative
examples.

    # the held-out number
    python scripts/evaluate_change_vqa.py --splits Test Test2

    # what the type mask contributes, on the same checkpoint
    python scripts/evaluate_change_vqa.py --splits Test --apply-type-mask

MASKED AND UNMASKED ARE BOTH REPORTED
-------------------------------------
The head can optionally restrict its answer to those legal for the question type
(`change_or_not` can only be `yes`/`no`). That mask is a real gain, and reporting
only the masked number would hide how much of the accuracy the mask supplied
rather than the model. Every run therefore scores both and writes both, and the
summary prints the delta.

WHAT THIS REPORT IS NOT
-----------------------
It is not a calibrated confidence claim. `mean_confidence` is the raw softmax of
the chosen answer; nothing is fitted. The report says so in
`inference_config.confidence_method`.

And it is not a captioning score. The task classifies into a closed 19-answer
space, so BLEU / ROUGE-L / CIDEr / BERTScore are deliberately absent — the plan
lists them under CAPTIONING, R-16 governs them, and R-16 is open.

EXIT CODES
----------
    0   evaluation completed
    2   the checkpoint or the prepared inputs are missing — NOT a crash
    3   the run started and raised a typed error
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402  (path set above)

from core.config import load_config  # noqa: E402
from core.errors import SatQueryError  # noqa: E402
from training.change_vqa.dataset import (  # noqa: E402
    group_by_native_split,
    load_change_vqa_records,
    read_scene_targets,
    verify_split_integrity,
)
from training.change_vqa.evaluate import (  # noqa: E402
    evaluate_records,
    qualitative_examples,
)
from training.change_vqa.features import (  # noqa: E402
    ChangeFeatureCache,
    TextFeatureCache,
)
from training.change_vqa.model import (  # noqa: E402
    ARCHITECTURE_VERSION,
    load_change_vqa_head,
    predict_answers,
)

DEFAULT_PREPARED = REPO_ROOT / "artifacts" / "change_vqa"
DEFAULT_CHECKPOINT = REPO_ROOT / "artifacts" / "change_vqa" / "run" / "head.pt"

#: Splits a model must never have been fitted on. Reported as the held-out set.
HELD_OUT_SPLITS: tuple[str, ...] = ("Test", "Test2")

EXIT_OK = 0
EXIT_INPUTS = 2
EXIT_ERROR = 3

__all__ = ["HELD_OUT_SPLITS", "build_parser", "main", "score_split"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/evaluate_change_vqa.py",
        description="Score a trained change-VQA head on CDVQA splits.",
    )
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data" / "cdvqa"))
    parser.add_argument("--prepared-dir", default=str(DEFAULT_PREPARED))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--out-dir", default=None,
                        help="default: alongside the checkpoint")
    parser.add_argument("--splits", nargs="+", default=["Test", "Test2"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--config", default=None)
    parser.add_argument("--apply-type-mask", action="store_true",
                        help="restrict answers to those legal for the question "
                             "type. Both variants are always reported; this only "
                             "chooses which one is written as the primary report.")
    parser.add_argument("--n-success", type=int, default=8)
    parser.add_argument("--n-failure", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _assemble(records, change_cache, text_cache, targets):
    """Rows whose features and targets are all present, plus the skip counts."""
    text_position = {q: i for i, q in enumerate(text_cache.question_ids)}
    kept = []
    change_rows = []
    text_rows = []
    qtype_rows = []
    temporal_rows = []
    missing_change = missing_text = missing_targets = 0

    for record in records:
        if not change_cache.has(record.scene_key):
            missing_change += 1
            continue
        position = text_position.get(record.question_id)
        if position is None:
            missing_text += 1
            continue
        if record.file_name not in targets:
            missing_targets += 1
            continue
        kept.append(record)
        change_rows.append(change_cache.get(record.scene_key))
        text_rows.append(text_cache.features[position])
        qtype_rows.append(record.qtype_index)
        temporal_rows.append(_temporal_index(record.temporal_ref))

    return {
        "records": kept,
        "change": (
            np.stack(change_rows).astype(np.float32) if change_rows
            else np.zeros((0, 1), dtype=np.float32)
        ),
        "text": (
            np.stack(text_rows).astype(np.float32) if text_rows
            else np.zeros((0, 1), dtype=np.float32)
        ),
        "qtype": np.asarray(qtype_rows, dtype=np.int64),
        "temporal": np.asarray(temporal_rows, dtype=np.int64),
        "skipped": {
            "missing_change_feature": missing_change,
            "missing_text_feature": missing_text,
            "missing_targets": missing_targets,
        },
    }


def _explain_empty_split(split: str, records, assembled, targets) -> str:
    """Say *why* a split cannot be scored, not just that it cannot.

    The bare `missing_targets: 39686` form names the symptom. In the failure it
    was written for, the cause was one step earlier: the held-out targets had
    never been prepared, because `scene_targets.jsonl` was built for the
    training splits only and every later prepare call passed `--skip-targets`.
    A caller reading raw counts cannot see that, so the *dominant* skip reason is
    named explicitly and paired with the fix.
    """
    skipped = assembled["skipped"]
    n = len(records)
    if n == 0:
        return (
            f"split {split}: no records were resolved at all, so there is "
            f"nothing to score. Check that the data root holds this split's "
            f"annotations."
        )
    if skipped["missing_targets"] == n:
        return (
            f"split {split}: no record had its features and targets all present "
            f"({skipped}). Every one of this split's {n} records is missing a "
            f"target: scene_targets.jsonl holds {len(targets)} scene(s) and "
            f"covers none of them. Held-out targets are a separate preparation "
            f"step — build them with `--splits Train Val Test Test2` and WITHOUT "
            f"`--skip-targets`. Scoring needs the label-derived targets."
        )
    if skipped["missing_change_feature"] == n:
        return (
            f"split {split}: no record had its features and targets all present "
            f"({skipped}). Every one of this split's {n} records is missing a "
            f"change feature: that cache is per-scene, so prepare this split "
            f"with `--splits ...` including it and a `--change-checkpoint`."
        )
    if skipped["missing_text_feature"] == n:
        return (
            f"split {split}: no record had its features and targets all present "
            f"({skipped}). Every one of this split's {n} records is missing a "
            f"question feature: text features are per-question, so prepare this "
            f"split with `--splits ...` including it."
        )
    return (
        f"split {split}: no record had its features and targets all present "
        f"({skipped}); {n} record(s) were considered and every one was dropped, "
        f"for a mix of the reasons above."
    )


def _temporal_index(name: str) -> int:
    from training.change_vqa.vocab import TEMPORAL_REFERENCE_TO_INDEX

    return TEMPORAL_REFERENCE_TO_INDEX.get(name, 0)


def _predict(model, assembled, *, batch_size: int, device: str, mask: bool):
    """Batched inference. Returns the concatenated prediction dict."""
    n = len(assembled["records"])
    chunks = []
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        chunks.append(
            predict_answers(
                model,
                change_features=assembled["change"][start:stop],
                text_features=assembled["text"][start:stop],
                qtype_indices=assembled["qtype"][start:stop],
                temporal_indices=assembled["temporal"][start:stop],
                qtypes=[r.qtype for r in assembled["records"][start:stop]],
                apply_type_mask=mask,
                device=device,
            )
        )
    if not chunks:
        raise SatQueryError("no rows to predict")
    merged = {}
    for key in ("answer_index", "confidence", "margin", "class_mag",
                "class_delta", "total_changed", "probabilities"):
        merged[key] = np.concatenate([c[key] for c in chunks], axis=0)
    merged["answer"] = [a for c in chunks for a in c["answer"]]
    merged["legal_mask_applied"] = any(c["legal_mask_applied"] for c in chunks)
    return merged


def score_split(
    split: str,
    *,
    records,
    model,
    change_cache,
    text_cache,
    targets,
    batch_size: int,
    device: str,
    out_dir: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    config_hash: str,
    n_success: int,
    n_failure: int,
) -> dict:
    """Score one split, write its report and examples, return a summary."""
    assembled = _assemble(records, change_cache, text_cache, targets)
    if not assembled["records"]:
        raise SatQueryError(
            _explain_empty_split(split, records, assembled, targets)
        )

    scored = assembled["records"]
    target_list = [targets[r.file_name] for r in scored]
    n_skipped = sum(assembled["skipped"].values())

    variants: dict[str, dict] = {}
    for label, mask in (("unmasked", False), ("masked", True)):
        predictions = _predict(
            model, assembled, batch_size=batch_size, device=device, mask=mask
        )
        report = evaluate_records(
            scored,
            predictions,
            targets=target_list,
            split=split,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            config_hash=config_hash,
            inference_config={
                "batch_size": batch_size,
                "device": device,
                "apply_type_mask": mask,
                "decode": "argmax",
                "confidence_method": "uncalibrated",
                "note": (
                    "raw softmax; nothing is fitted. The R-03 calibration "
                    "contract is a separate, open ruling."
                ),
            },
            n_skipped=n_skipped,
        )
        # The report is written FIRST so its bytes exist to be hashed. See the
        # note on the two digests below.
        path = report.write(out_dir / f"eval_{split}_{label}.json")
        examples = qualitative_examples(
            scored,
            predictions,
            targets=target_list,
            n_success=n_success,
            n_failure=n_failure,
        )
        (out_dir / f"qualitative_{split}_{label}.json").write_text(
            json.dumps(examples, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        # TWO DIGESTS, TWO MEANINGS (provenance defect section 6a)
        # --------------------------------------------------------
        # `report_canonical_sha256` -- content digest of the payload as
        #     canonical compact JSON. Stable across reformatting; this is the
        #     value that was previously (and misleadingly) published as
        #     `report_sha256`.
        # `report_file_bytes_sha256` -- digest of the bytes actually on disk,
        #     i.e. what `sha256sum eval_<split>_<label>.json` prints. This is
        #     the one a reviewer can independently reproduce.
        # The legacy key is RETAINED with the canonical value so every recorded
        # consumer of the old field keeps reading the same number it always did.
        # Removing it would change the meaning of existing exports; renaming it
        # silently would change the meaning of existing readers.
        variants[label] = {
            "report_path": str(path),
            "report_canonical_sha256": report.canonical_sha256(),
            "report_file_bytes_sha256": report.file_bytes_sha256(path),
            "report_sha256": report.canonical_sha256(),
            "accuracy": report.accuracy,
            # §6c: the baselines are surfaced HERE too, not only inside the report
            # file. The summary is the artifact a reader opens first, and a
            # baseline they have to go hunting for is a baseline they will not
            # check. Summarised, not duplicated: the per-type detail and the gold
            # histogram stay in the report, which the summary points at.
            "baselines": {
                "global_majority": report.payload["baselines"]["global_majority"],
                "global_majority_answer": (
                    report.payload["baselines"].get("global_majority_answer")
                ),
                "per_type_majority": (
                    report.payload["baselines"]["per_type_majority"]
                ),
                "derivation": report.payload["baselines"]["derivation"],
                "caveat": report.payload["baselines"]["caveat"],
                "detail_in_report": True,
            },
            "macro_f1": report.payload["metrics"]["macro_f1"],
            "top_3_accuracy": report.payload["metrics"].get("top_3_accuracy"),
            "mean_confidence": report.payload["metrics"]["mean_confidence"],
            "per_type": report.payload["metrics"]["per_type"],
            "n_scored": report.payload["counts"]["n_scored"],
            "predictions": predictions,
        }

    return {
        "split": split,
        "held_out": split in HELD_OUT_SPLITS,
        "n_scored": variants["unmasked"]["n_scored"],
        "skipped": assembled["skipped"],
        "unmasked": {k: v for k, v in variants["unmasked"].items()
                     if k != "predictions"},
        "masked": {k: v for k, v in variants["masked"].items()
                   if k != "predictions"},
        "mask_gain": round(
            variants["masked"]["accuracy"] - variants["unmasked"]["accuracy"], 6
        ),
    }


def _sha256(path: Path) -> str:
    import hashlib

    if not path.exists():
        return "unavailable"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(path: Path) -> str:
    """SHA256 of a text file's BYTES, read by the caller that wrote it.

    Used for the summary itself: it is hashed after being written so the digest
    describes the file on disk, matching what a reviewer computes with
    `sha256sum`. `_sha256` already does exactly this; the name is kept separate
    because the summary note has to explain WHICH of the two hash conventions is
    in play at each key, and conflating them is provenance defect section 6a.
    """
    return _sha256(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = Path(args.checkpoint)
    prepared = Path(args.prepared_dir)
    out_dir = Path(args.out_dir) if args.out_dir else checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SatQuery AI — change-VQA evaluation (R-02)")
    print("=" * 72)
    print(f"checkpoint           : {checkpoint}")
    print(f"prepared inputs      : {prepared}")
    print(f"splits               : {args.splits}")
    print(f"held-out splits      : {list(HELD_OUT_SPLITS)}")
    print(f"device               : {args.device}")

    if not checkpoint.exists():
        print(f"\nMISSING CHECKPOINT: {checkpoint}")
        print("Train one first: python scripts/train_change_vqa.py")
        return EXIT_INPUTS

    change_cache_path = prepared / "change_features.npz"
    targets_path = prepared / "scene_targets.jsonl"
    # `--splits` NAMES the splits to score, so every one of them needs its
    # question-feature cache. This used to be checked inside the scoring loop,
    # where a missing cache was SKIPPED with a one-line notice: the run would
    # score the splits it had, write an `eval_summary.json` without the others,
    # and exit 0 -- a successful-looking artifact silently missing a held-out
    # split, which nothing downstream can detect. It is an input problem, so it
    # is reported here with the other missing inputs, before a model is loaded.
    text_cache_paths = {
        split: prepared / f"text_features_{split}.npz" for split in args.splits
    }
    for path in (change_cache_path, targets_path, *text_cache_paths.values()):
        if path.exists():
            continue
        print(f"\nMISSING PREPARED INPUT: {path}")
        if path in text_cache_paths.values():
            missing = sorted(s for s, p in text_cache_paths.items() if p == path)
            print(f"no question-feature cache for {missing}. Scoring needs one "
                  f"cache per split. A split with no resolvable records writes "
                  f"no cache at all, so also check that the data root holds "
                  f"those splits' annotations.")
            print("Run: python scripts/prepare_change_vqa.py "
                  f"--data-root <root> --out-dir {prepared} "
                  f"--splits {' '.join(args.splits)} "
                  "--skip-targets --skip-features --skip-manifest")
        else:
            print("Run: python scripts/prepare_change_vqa.py --splits Train Val "
                  "Test Test2")
        return EXIT_INPUTS

    try:
        config = load_config(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"config could not be loaded: {type(exc).__name__}: {exc}")
        config = None
    config_hash = config.hash if config is not None else "unavailable"

    metadata_path = checkpoint.parent / "model_metadata.json"
    metadata: dict = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        print(f"checkpoint state     : {metadata.get('state', 'unavailable')}")
        print(f"selected on          : {metadata.get('selected_on', 'unavailable')}")
        print(f"detector trained     : {metadata.get('detector_trained')}")

    if args.dry_run:
        print("\n--dry-run: inputs present; nothing scored, nothing written")
        return EXIT_OK

    try:
        model = load_change_vqa_head(checkpoint, device=args.device)
        change_cache = ChangeFeatureCache.read(change_cache_path)
        targets = read_scene_targets(targets_path)
        all_records = load_change_vqa_records(
            args.data_root, splits=tuple(args.splits), require_images=False
        )
        integrity = verify_split_integrity(
            all_records, expect_full_split=False,
            expected_splits=tuple(args.splits),
        )
        if not integrity["is_clean"]:
            raise SatQueryError(
                f"split integrity check failed: {integrity['size_errors']} "
                f"{integrity['leakage']}"
            )
        by_split = group_by_native_split(all_records)
    except SatQueryError as exc:
        print(f"\nINPUT ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_INPUTS
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(f"change cache         : {len(change_cache)} scenes, "
          f"spec {change_cache.spec_hash}")
    print(f"scene targets        : {len(targets)} scenes")
    print()

    summaries = []
    for split in args.splits:
        text_cache_path = text_cache_paths[split]
        text_cache = TextFeatureCache.read(text_cache_path)
        summary = score_split(
            split,
            records=by_split[split],
            model=model,
            change_cache=change_cache,
            text_cache=text_cache,
            targets=targets,
            batch_size=args.batch_size,
            device=args.device,
            out_dir=out_dir,
            checkpoint_path=checkpoint,
            checkpoint_sha256=_sha256(checkpoint),
            config_hash=config_hash,
            n_success=args.n_success,
            n_failure=args.n_failure,
        )
        summaries.append(summary)

        held = "" if summary["held_out"] else "  [NOT HELD OUT]"
        print(f"{split}{held}")
        print(f"  scored             : {summary['n_scored']}")
        print(f"  accuracy (unmasked): {summary['unmasked']['accuracy']:.4f}")
        print(f"  macro F1 (unmasked): {summary['unmasked']['macro_f1']:.4f}")
        print(f"  top-3 (unmasked)   : {summary['unmasked']['top_3_accuracy']}")
        print(f"  accuracy (masked)  : {summary['masked']['accuracy']:.4f}")
        print(f"  mask gain          : {summary['mask_gain']:+.4f}")
        print(f"  mean confidence    : {summary['unmasked']['mean_confidence']:.4f}")
        print(f"  report             : "
              f"{Path(summary['unmasked']['report_path']).name}")
        print()

    if not summaries:
        print("nothing was scored", file=sys.stderr)
        return EXIT_INPUTS

    combined = {
        "schema": "change_vqa_evaluation_v1",
        "architecture": ARCHITECTURE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_metadata": metadata,
        "config_hash": config_hash,
        "device": args.device,
        "splits": summaries,
        "note": (
            "Classification metrics only — the answer space is closed (19 "
            "answers), so BLEU/ROUGE-L/CIDEr/BERTScore are deliberately not "
            "reported (plan section 2.2 lists them under CAPTIONING; R-16 "
            "governs them and is open). Confidence is uncalibrated."
        ),
        # HASH CONVENTIONS (provenance defect section 6a)
        # ----------------------------------------------
        # Each report records BOTH digests under unambiguous names:
        #   report_canonical_sha256   -- content digest, stable across reformat
        #   report_file_bytes_sha256  -- byte digest, equals `sha256sum <file>`
        #   report_sha256             -- legacy key, still the canonical value
        # The older exports carried only the third, named as if it were the
        # second. Nothing recorded was retro-changed; nothing new is ambiguous.
        "hash_conventions": {
            "report_canonical_sha256": (
                "sha256 of canonical compact json.dumps(payload, sort_keys=True). "
                "Content digest: independent of indent/key order."
            ),
            "report_file_bytes_sha256": (
                "sha256 of the report's bytes on disk. Reproducible with "
                "`sha256sum <report_path>`."
            ),
            "report_sha256": (
                "LEGACY ALIAS retained for backward compatibility. Equal to "
                "report_canonical_sha256, NOT to the file bytes. Prefer the "
                "explicitly named key."
            ),
            "summary_file_bytes_sha256": (
                "sha256 of this eval_summary.json's own bytes. Recorded in a "
                "sidecar file (eval_summary.sha256) because a file cannot "
                "contain a hash of itself."
            ),
        },
        # METRIC PROVENANCE (provenance defect section 6c)
        # -----------------------------------------------
        # States, per metric, whether it is reproducible from this export alone.
        # `top_3_accuracy` and `mean_confidence` are NOT: a confusion matrix
        # cannot yield either, so they are recorded-only and must not be
        # presented as verifiable. `baselines` ARE derived, from gold labels.
        "metric_provenance": {
            "answer_derived": [
                "accuracy",
                "macro_f1",
                "per_type",
                "confusion_matrix",
                "n_scored",
            ],
            "recorded_only": {
                "top_3_accuracy": (
                    "Not reconstructible from confusion_matrix; recorded from "
                    "the scoring run. Treat as an unverifiable record."
                ),
                "mean_confidence": (
                    "Not reconstructible from confusion_matrix; recorded from "
                    "the scoring run. Raw softmax of the chosen answer; nothing "
                    "is fitted (see inference_config.confidence_method)."
                ),
            },
            "baseline_derived": [
                "baselines.global_majority",
                "baselines.per_type_majority",
            ],
            "baseline_derivation": (
                "Computed from gold labels only by "
                "training.change_vqa.evaluate.majority_baselines. Reproducible "
                "from this export without re-running the model."
            ),
            "calibration_note": (
                "A temperature-scaling artifact exists at "
                "artifacts/calibration_v001.json, but it is deliberately NOT "
                "applied here: applying it would break comparability with the "
                "already-published R-02 accuracy/macro-F1 figures. These "
                "numbers remain raw-softmax."
            ),
        },
        "argv": sys.argv[1:],
    }
    summary_path = out_dir / "eval_summary.json"
    summary_path.write_text(
        json.dumps(combined, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    # The summary cannot contain a hash of itself, so the byte digest is written
    # beside it. This is the value a reviewer reproduces with `sha256sum`.
    summary_digest = _text_sha256(summary_path)
    (out_dir / "eval_summary.sha256").write_text(
        f"{summary_digest}  eval_summary.json\n", encoding="utf-8"
    )
    print(f"summary              : {summary_path}")
    print(f"summary sha256       : {summary_digest}")

    non_held = [s["split"] for s in summaries if not s["held_out"]]
    if non_held:
        print()
        print(f"NOTE: {non_held} are NOT held out. Any accuracy reported for them "
              f"is a training-time number, not a generalisation claim.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
