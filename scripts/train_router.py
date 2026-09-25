"""Train the SatQuery intent router.

    python scripts/train_router.py --smoke           # fast, stub encoder
    python scripts/train_router.py --stub            # fast, no model download
    python scripts/train_router.py                   # full run, real MiniLM

Runs entirely on CPU. Measured cost with the real encoder: ~40 s to load
MiniLM on a cold cache, ~0.1 s to embed the corpus, ~1 s to train 60 epochs.
There is no reason to spend Kaggle GPU quota on this.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config  # noqa: E402
from core.errors import SatQueryError  # noqa: E402
from router.dataset import build_corpus  # noqa: E402
from router.train import train_router  # noqa: E402


def _dry_run_report(corpus) -> None:
    """Print the corpus structure before committing to a training run."""
    print("CORPUS")
    print("-" * 62)
    print(f"  total examples : {len(corpus)}")
    print(f"  groups         : {len(corpus.groups())}")
    print(f"  by source      : {corpus.source_counts()}")
    print()
    print("  by task:")
    counts = corpus.task_counts()
    for task, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {task:<14} {n:>5}")

    print()
    print("  binary positives:")
    from router.label_space import BINARY_HEADS

    for head in BINARY_HEADS:
        pos = corpus.positives(head)
        print(f"    {head:<16} {pos:>5} / {len(corpus)}")

    print()
    print("  problems:", corpus.validate() or "none")

    deduped, conflicts = corpus.dedupe()
    print(f"  after dedupe   : {len(deduped)} ({len(corpus) - len(deduped)} removed)")
    print(f"  conflicts      : {conflicts or 'none'}")
    print("-" * 62)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the SatQuery intent router")
    parser.add_argument("--smoke", action="store_true",
                        help="3 epochs, stub encoder; validates the pipeline only")
    parser.add_argument("--stub", action="store_true",
                        help="use hash-based stub embeddings instead of MiniLM")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--artifact-dir", type=str, default=None)
    parser.add_argument("--template-repeats", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the corpus structure and exit without training")
    parser.add_argument("--json", action="store_true",
                        help="emit the metrics JSON instead of the text summary")
    args = parser.parse_args()

    config = load_config()
    cfg = config.get

    # -- corpus ------------------------------------------------------------
    corpus = build_corpus(n_template_repeats=args.template_repeats,
                          seed=args.seed or config.seed)

    if args.dry_run:
        _dry_run_report(corpus)
        return 0

    problems = corpus.validate()
    if problems:
        print("CORPUS VALIDATION FAILED", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 2

    # -- encoder -----------------------------------------------------------
    encoder = None
    if not args.stub:
        try:
            from router.encoder import build_encoder

            print("loading frozen encoder (first run downloads ~91 MB)...")
            encoder = build_encoder(config)
            print(f"  {encoder}")
            print(f"  load time: {encoder.load_seconds:.1f} s")
        except SatQueryError as exc:
            print(f"ENCODER UNAVAILABLE: {exc}", file=sys.stderr)
            print("  re-run with --stub to exercise the pipeline without the model",
                  file=sys.stderr)
            return 3

    # -- hyperparameters ---------------------------------------------------
    epochs = args.epochs if args.epochs is not None else int(cfg("router.training.epochs", 60))
    batch_size = args.batch_size if args.batch_size is not None else int(cfg("router.training.batch_size", 64))
    lr = args.learning_rate if args.learning_rate is not None else float(cfg("router.training.learning_rate", 1e-3))
    hidden = args.hidden_dim if args.hidden_dim is not None else int(cfg("router.hidden_dim", 128))
    seed = args.seed if args.seed is not None else config.seed
    artifact_dir = args.artifact_dir or str(REPO_ROOT / "artifacts" / "router" / "router_adapter_v001")

    if args.smoke:
        epochs = 3

    print()
    print(f"training: epochs={epochs} batch={batch_size} lr={lr} "
          f"hidden={hidden} seed={seed}")
    print(f"encoder : {'stub (NOT DEPLOYABLE)' if encoder is None else encoder.model_name}")
    print()

    # -- train -------------------------------------------------------------
    result = train_router(
        corpus=corpus,
        encoder=encoder,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        hidden_dim=hidden,
        dropout=float(cfg("router.dropout", 0.10)),
        task_loss_weight=float(cfg("router.training.task_loss_weight", 1.0)),
        modality_loss_weight=float(cfg("router.training.modality_loss_weight", 0.3)),
        binary_loss_weight=float(cfg("router.training.binary_loss_weight", 0.5)),
        seed=seed,
        device=config.device_preference,
        artifact_dir=artifact_dir,
        cache_dir=REPO_ROOT / "artifacts" / "router" / "cache",
        config_hash=config.hash,
        val_ratio=float(cfg("router.training.val_ratio", 0.15)),
        hard_negatives_to_test=bool(cfg("router.training.hard_negatives_to_test", True)),
        verbose=not args.json,
    )

    if args.json:
        print(json.dumps({
            "artifact_dir": str(result.artifact_dir),
            "corpus_hash": result.corpus_hash,
            "duration_seconds": round(result.duration_seconds, 2),
            "split": result.split_report,
            "metrics": {
                "train": result.train_metrics.to_dict(),
                "val": result.val_metrics.to_dict(),
                "test": result.test_metrics.to_dict(),
            },
        }, indent=2, sort_keys=True))
    else:
        print()
        print(result.summary())

    # -- verdict -----------------------------------------------------------
    if encoder is None and not args.smoke:
        print()
        print("NOTE: this run used the stub encoder. The artifact is marked")
        print("      encoder_type='stub' and must not be deployed or evaluated.")
        return 0

    # Gate 2 requires BOTH: every class measured, and accuracy >= 0.95.
    # A class absent from the test set cannot be claimed as passing.
    if result.unmeasured_test_tasks:
        print()
        print("GATE 2 INCONCLUSIVE: no test examples for "
              f"{', '.join(result.unmeasured_test_tasks)}")
        return 2

    return 0 if result.gate2_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())