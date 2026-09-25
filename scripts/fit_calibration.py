"""Fit the confidence calibration on VALIDATION data (Phase 14 / R-03).

WHAT THIS PRODUCES
------------------
`artifacts/calibration_v001.json` — the artifact the frozen config has always
named (`confidence.calibration_file: calibration_v001.json`) and that
`core/controller.py` never loaded. Producing it is what closes the single §67
immutable-decision violation ("calibrated confidence").

WHY VALIDATION, AND WHY THAT IS ENFORCED
----------------------------------------
The temperature is estimated from data the model did not train on, then applied
to data nobody tuned on. Fitting on Test or Test2 would make the reported
confidence a function of the answers it is used to score — a leak, not a
calibration. `--split` defaults to `Val` and both the fitter entry point and the
artifact writer REFUSE a held-out split, so the guard holds even under a typo.

REAL INFERENCE, NOT A FIXTURE
-----------------------------
This script loads the promoted R-02 head and runs `predict_answers` over the
validation split. The probabilities it fits against are produced by the same
code path the deployed controller uses.

USAGE
-----
    python scripts/fit_calibration.py --dry-run    # check inputs, exit
    python scripts/fit_calibration.py              # fit, write artifact

EXIT CODES
----------
    0   artifact written
    2   the checkpoint or the prepared inputs are missing — NOT a crash
    3   the fit started and raised a typed error
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from core.config import load_config  # noqa: E402
from core.errors import SatQueryError  # noqa: E402
from training.calibration.artifact import (  # noqa: E402
    DEFAULT_ARTIFACT_NAME,
    HELD_OUT_SPLITS,
    CalibrationFitError,
    build_artifact_payload,
    file_sha256,
    write_calibration_artifact,
)
from training.calibration.evaluate import (  # noqa: E402
    DEFAULT_N_BINS,
    compare_calibration,
    reliability_diagram,
)
from training.calibration.fitter import fit_temperature  # noqa: E402
from training.change_vqa.dataset import load_change_vqa_records  # noqa: E402
from training.change_vqa.features import (  # noqa: E402
    CHANGE_FEATURE_DIM,
    TEXT_FEATURE_DIM,
    ChangeFeatureCache,
    TextFeatureCache,
)
from training.change_vqa.model import (  # noqa: E402
    load_change_vqa_head,
    predict_answers,
)

DEFAULT_PREPARED = REPO_ROOT / "artifacts" / "change_vqa"
DEFAULT_CHECKPOINT = DEFAULT_PREPARED / "run" / "head.pt"
DEFAULT_OUT = REPO_ROOT / "artifacts" / DEFAULT_ARTIFACT_NAME
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "cdvqa"

CHANGE_CACHE_NAME = "change_features.npz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/fit_calibration.py",
        description="Fit temperature-scaling calibration on validation data.",
    )
    parser.add_argument(
        "--split",
        default="Val",
        help=f"split to fit on; MUST NOT be one of {sorted(HELD_OUT_SPLITS)}",
    )
    parser.add_argument("--prepared-dir", default=str(DEFAULT_PREPARED))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    parser.add_argument(
        "--apply-type-mask",
        action="store_true",
        help=(
            "fit against masked predictions. Default OFF: the unmasked "
            "distribution is what the deployed controller sees for a query "
            "whose type it is not certain of, so it is the honest one to "
            "calibrate."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check that the inputs exist and exit without fitting",
    )
    return parser


def _required_paths(args: argparse.Namespace) -> dict[str, Path]:
    prepared = Path(args.prepared_dir)
    return {
        "checkpoint": Path(args.checkpoint),
        "data_root": Path(args.data_root),
        "change_features": prepared / CHANGE_CACHE_NAME,
        "text_features": prepared / f"text_features_{args.split}.npz",
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print("=" * 72)
    print("SatQuery AI — confidence calibration fit (R-03 / Phase 14)")
    print("=" * 72)

    if args.split in HELD_OUT_SPLITS:
        print(f"REFUSED: --split {args.split} is a held-out benchmark split.")
        print("         Fitting here would leak; fit on Val instead.")
        return 3

    paths = _required_paths(args)
    missing = [f"{name}: {path}" for name, path in paths.items()
               if not path.exists()]
    if missing:
        print("MISSING INPUTS (nothing was fitted):")
        for item in missing:
            print(f"  - {item}")
        return 2

    prepared = Path(args.prepared_dir)
    checkpoint = Path(args.checkpoint)
    print(f"split                : {args.split}")
    print(f"data root            : {args.data_root}")
    print(f"prepared dir         : {prepared}")
    print(f"checkpoint           : {checkpoint}")
    print(f"output               : {args.out}")
    print(f"type mask            : {args.apply_type_mask}")

    if args.dry_run:
        print("\ndry run: all inputs present; nothing fitted, nothing written.")
        return 0

    try:
        config = load_config()
        checkpoint_sha = file_sha256(checkpoint)
        print(f"config hash          : {config.hash}")
        print(f"checkpoint sha256    : {checkpoint_sha}")

        # -- assemble rows: records + caches, the SAME join the evaluator uses --
        records = load_change_vqa_records(
            args.data_root, splits=(args.split,), require_images=False
        )
        if not records:
            print(f"no records resolved for split {args.split!r}")
            return 2

        change_cache = ChangeFeatureCache.read(prepared / CHANGE_CACHE_NAME)
        text_cache = TextFeatureCache.read(
            prepared / f"text_features_{args.split}.npz"
        )
        text_position = {int(q): i for i, q in enumerate(text_cache.question_ids)}

        change_rows, text_rows, qtype_rows, temporal_rows, gold = [], [], [], [], []
        skipped = {
            "missing_change_feature": 0,
            "missing_text_feature": 0,
        }
        from training.change_vqa.vocab import TEMPORAL_REFERENCE_TO_INDEX  # noqa

        for record in records:
            if not change_cache.has(record.scene_key):
                skipped["missing_change_feature"] += 1
                continue
            position = text_position.get(int(record.question_id))
            if position is None:
                skipped["missing_text_feature"] += 1
                continue
            change_rows.append(change_cache.get(record.scene_key))
            text_rows.append(text_cache.features[position])
            qtype_rows.append(record.qtype_index)
            temporal_rows.append(
                TEMPORAL_REFERENCE_TO_INDEX.get(record.temporal_ref, 0)
            )
            gold.append(int(record.answer_index))

        if not gold:
            print("no ALIGNED rows: the caches and the records disagree")
            return 3

        print(f"records              : {len(records)}")
        print(f"aligned rows         : {len(gold)}")
        print(f"skipped              : {skipped}")

        change_arr = np.stack(change_rows).astype(np.float32)
        text_arr = np.stack(text_rows).astype(np.float32)
        qtype_arr = np.asarray(qtype_rows, dtype=np.int64)
        temporal_arr = np.asarray(temporal_rows, dtype=np.int64)

        if change_arr.shape[1] != CHANGE_FEATURE_DIM:
            print(f"change dim {change_arr.shape[1]} != {CHANGE_FEATURE_DIM}")
            return 3
        if text_arr.shape[1] != TEXT_FEATURE_DIM:
            print(f"text dim {text_arr.shape[1]} != {TEXT_FEATURE_DIM}")
            return 3

        # -- REAL inference ---------------------------------------------------
        model = load_change_vqa_head(checkpoint, device=args.device)
        print("\nrunning inference on the promoted head...")
        chunks = []
        for start in range(0, len(gold), args.batch_size):
            stop = min(len(gold), start + args.batch_size)
            chunks.append(
                predict_answers(
                    model,
                    change_features=change_arr[start:stop],
                    text_features=text_arr[start:stop],
                    qtype_indices=qtype_arr[start:stop],
                    temporal_indices=temporal_arr[start:stop],
                    qtypes=None,
                    apply_type_mask=args.apply_type_mask,
                    device=args.device,
                )
            )
        probabilities = np.concatenate([c["probabilities"] for c in chunks], axis=0)
        print(f"probabilities        : {probabilities.shape}")

        # -- fit --------------------------------------------------------------
        fit = fit_temperature(probabilities, gold)
        print("\nFIT")
        for key, value in fit.describe().items():
            print(f"  {key:20}: {value}")

        from evidence.confidence import TemperatureCalibration  # noqa: PLC0415

        calibration = TemperatureCalibration(
            temperature=fit.temperature,
            fitted_on=args.split,
            artifact=Path(args.out).name,
            n_samples=fit.n_samples,
        )
        metrics = compare_calibration(
            probabilities, gold, calibration, n_bins=args.n_bins
        )
        reliability = reliability_diagram(probabilities, gold, n_bins=args.n_bins)

        print("\nMETRICS (fit-set: optimistic by construction)")
        for key, value in metrics.describe().items():
            print(f"  {key:20}: {value}")

        payload = build_artifact_payload(
            fit=fit,
            metrics=metrics,
            fitted_on=args.split,
            reliability=reliability,
            checkpoint_sha256=checkpoint_sha,
            checkpoint_path=str(checkpoint),
            config_hash=config.hash,
            dataset_id="cdvqa",
            feature_spec="change_feat_v1",
            n_classes=int(probabilities.shape[1]),
            extra={
                "scope": {
                    "specialist": "change_vqa",
                    "note": (
                        "This temperature calibrates the R-02 change-VQA head's "
                        "answer confidence. Other specialists emit their own "
                        "raw scores and are unaffected."
                    ),
                },
                "type_mask_applied": bool(args.apply_type_mask),
            },
        )
        path = write_calibration_artifact(payload, args.out)
        print(f"\nartifact written     : {path}")
        print(f"artifact sha256      : {file_sha256(path)}")
        if not fit.is_effective:
            print(
                "\nNOTE: the fitted temperature is the identity map. The "
                "consumer will report `uncalibrated` for it, honestly. That is "
                "the correct outcome when the raw scores are already calibrated."
            )
        return 0

    except CalibrationFitError as exc:
        print(f"\nREFUSED: {exc}")
        return 3
    except SatQueryError as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
