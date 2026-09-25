"""R-02 training entry point: train the CDVQA reasoning head.

THIN CLI over `training/change_vqa/train.py`, following the same split as
`scripts/train_change.py`. Everything that computes lives in the module; this
file resolves the default paths, reports the accounting and returns an exit code.

    # the real run, on the cached features
    python scripts/train_change_vqa.py --device cuda

    # a mechanics check that commits nothing
    python scripts/train_change_vqa.py --max-train-records 2000 --epochs 3

    # what would run, without running it
    python scripts/train_change_vqa.py --dry-run

TEST SPLITS ARE NOT READABLE HERE
---------------------------------
`training/change_vqa/train.py` loads only Train and Val, and there is no option
in either file that changes that. A model selected on the data it is later graded
on produces a number that means nothing, and an option is a thing someone
eventually passes.

EXIT CODES
----------
    0   training completed
    2   the prepared inputs are missing or unusable — NOT a crash
    3   the run started and the trainer raised a typed error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.errors import SatQueryError  # noqa: E402  (path set above)
from core.config import load_config  # noqa: E402
from training.change_vqa.train import (  # noqa: E402
    DEFAULT_TIME_LIMIT_SECONDS,
    FORBIDDEN_SPLITS,
    POST_TRAINING_STATE,
    READABLE_SPLITS,
    SEED,
    build_parser,
    environment_report,
    main as train_main,
)

DEFAULT_PREPARED = REPO_ROOT / "artifacts" / "change_vqa"
DEFAULT_OUT = REPO_ROOT / "artifacts" / "change_vqa" / "run"

EXIT_OK = 0
EXIT_INPUTS = 2
EXIT_ERROR = 3

#: Re-exported so a caller can assert the contract without reaching into the
#: module. Mirrors `scripts/train_change.py`'s re-export convention.
__all__ = [
    "READABLE_SPLITS",
    "FORBIDDEN_SPLITS",
    "POST_TRAINING_STATE",
    "build_parser",
    "check_inputs",
    "main",
]


def check_inputs(prepared: Path) -> list[str]:
    """Everything the trainer needs, named. Missing paths are returned, not raised."""
    required = {
        "change cache": prepared / "change_features.npz",
        "train text cache": prepared / "text_features_Train.npz",
        "val text cache": prepared / "text_features_Val.npz",
        "scene targets": prepared / "scene_targets.jsonl",
    }
    return [f"{label}: {path}" for label, path in required.items()
            if not path.exists()]


def build_wrapper_parser() -> argparse.ArgumentParser:
    """The trainer's parser, with the prepared-input paths derived from --prepared-dir.

    Built by mutating the trainer's own parser rather than duplicating it, so the
    two can never drift: a new flag on the trainer appears here automatically.

    The four prepared-input defaults are set to `None` here and filled in from
    `--prepared-dir` after parsing. Defaulting them to the canonical directory at
    parser-build time would silently ignore `--prepared-dir`, which is exactly the
    bug this shape avoids: an explicit default cannot tell "the user asked for the
    canonical directory" apart from "the user asked for something else".
    """
    parser = build_parser()
    for action in parser._actions:  # noqa: SLF001 - deliberate, see docstring
        if action.dest in (
            "change_cache", "text_cache_train", "text_cache_val", "targets"
        ):
            action.default = None
            action.required = False
        elif action.dest == "output_dir":
            action.default = str(DEFAULT_OUT)
    parser.prog = "python scripts/train_change_vqa.py"
    parser.add_argument(
        "--prepared-dir", default=str(DEFAULT_PREPARED),
        help="directory written by scripts/prepare_change_vqa.py",
    )
    parser.add_argument(
        "--config", default=None,
        help="config file; default configs/base.yaml",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="check the prepared inputs and exit without training",
    )
    return parser


def resolve_prepared_paths(args: argparse.Namespace, prepared: Path) -> None:
    """Fill any prepared-input path the user did not override."""
    defaults = {
        "change_cache": "change_features.npz",
        "text_cache_train": "text_features_Train.npz",
        "text_cache_val": "text_features_Val.npz",
        "targets": "scene_targets.jsonl",
    }
    for dest, filename in defaults.items():
        if getattr(args, dest, None) is None:
            setattr(args, dest, str(prepared / filename))


def main(argv: list[str] | None = None) -> int:
    parser = build_wrapper_parser()
    args, _unknown = parser.parse_known_args(argv)
    prepared = Path(args.prepared_dir)
    resolve_prepared_paths(args, prepared)

    print("=" * 72)
    print("SatQuery AI — change-VQA training (R-02)")
    print("=" * 72)
    print(f"prepared inputs      : {prepared}")
    print(f"output dir           : {args.output_dir}")
    print(f"readable splits      : {list(READABLE_SPLITS)}")
    print(f"forbidden splits     : {list(FORBIDDEN_SPLITS)} (not readable)")
    print(f"time limit           : {args.time_limit_seconds}s "
          f"(plan budget {DEFAULT_TIME_LIMIT_SECONDS}s)")
    print(f"seed                 : {args.seed}")
    print(f"resulting state      : {POST_TRAINING_STATE} "
          f"(training is not verification)")

    missing = check_inputs(prepared)
    if missing:
        print("\nMISSING PREPARED INPUTS:")
        for line in missing:
            print(f"  {line}")
        print("\nRun: python scripts/prepare_change_vqa.py")
        return EXIT_INPUTS

    try:
        config = load_config(args.config)
        print(f"config hash          : {config.hash}")
    except Exception as exc:  # noqa: BLE001
        print(f"config could not be loaded: {type(exc).__name__}: {exc}")
        config = None

    record_path = Path(args.output_dir) / "run_record.json"
    if args.dry_run:
        print("\n--dry-run: inputs present; nothing trained, nothing written")
        print(f"would write          : {record_path}")
        return EXIT_OK

    if args.config_hash is None and config is not None:
        args.config_hash = config.hash
    if getattr(args, "expect_change_spec", None) is None:
        args.expect_change_spec = None

    print()
    try:
        train_main(
            [
                "--data-root", args.data_root,
                "--change-cache", args.change_cache,
                "--text-cache-train", args.text_cache_train,
                "--text-cache-val", args.text_cache_val,
                "--targets", args.targets,
                "--output-dir", args.output_dir,
                "--device", args.device,
                "--time-limit-seconds", str(args.time_limit_seconds),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--lr", str(args.lr),
                "--weight-decay", str(args.weight_decay),
                "--warmup-ratio", str(args.warmup_ratio),
                "--grad-clip", str(args.grad_clip),
                "--patience", str(args.patience),
                "--min-delta", str(args.min_delta),
                "--trunk-dim", str(args.trunk_dim),
                "--text-dim", str(args.text_dim),
                "--dropout", str(args.dropout),
                "--seed", str(args.seed),
                "--config-hash", str(args.config_hash or "unavailable"),
                *(["--amp"] if args.amp else ["--no-amp"]),
                *(["--require-cuda"] if args.require_cuda else []),
                *(["--balance-by-type"] if args.balance_by_type else []),
                *(["--max-train-records", str(args.max_train_records)]
                  if args.max_train_records else []),
                *(["--max-val-records", str(args.max_val_records)]
                  if args.max_val_records else []),
                *(["--expect-change-spec", str(args.expect_change_spec)]
                  if args.expect_change_spec else []),
                *(["--expect-text-spec", str(args.expect_text_spec)]
                  if args.expect_text_spec else []),
            ]
        )
    except (SatQueryError,) as exc:
        print(f"\nTRAINING ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_INPUTS
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        print()
        print("=" * 72)
        print("SUMMARY")
        print("=" * 72)
        print(f"state                : {record['state']}")
        print(f"best epoch           : {record['selection']['best_epoch']}")
        print(f"best val accuracy    : {record['selection']['best_accuracy']}")
        print(f"stop reason          : {record['selection']['stop_reason']}")
        print(f"epochs completed     : {record['optimization']['epochs_completed']}")
        print(f"head parameters      : {record['model']['parameters']:,}")
        print(f"checkpoint           : {record['model']['checkpoint']}")
        print(f"checkpoint sha256    : {record['model']['checkpoint_sha256']}")
        print()
        print("NEXT: evaluate on the held-out split with")
        print("  python scripts/evaluate_change_vqa.py "
              "--splits Test Test2")
        print("R-02 is TRAINED_UNVERIFIED until that evaluation is returned "
              "and reviewed.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
