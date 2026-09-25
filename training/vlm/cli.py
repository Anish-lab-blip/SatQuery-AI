"""SatQuery AI — Phase 6 command-line entry point.

Every subcommand is **config-driven**: the recipe comes from the frozen registry
(`configs/base.yaml`) through `VLMTrainingConfig.from_registry`, and any
subcommand flag overrides a single field. No source edit is ever required to run
a job -- which is what makes the Kaggle entrypoint a thin wrapper rather than a
fork.

    python -m training.vlm.cli prepare  --limit 4
    python -m training.vlm.cli baseline --limit 8
    python -m training.vlm.cli train    --max-steps 100 --output-dir artifacts/vlm/run1
    python -m training.vlm.cli evaluate --adapter artifacts/vlm/run1
    python -m training.vlm.cli smoke
    python -m training.vlm.cli package

Every subcommand prints a single JSON object to stdout, so a caller (or a
notebook) can parse the result without scraping prose.

`smoke` runs a tiny end-to-end CPU path and labels its output `is_smoke: true`;
its artifact export is refused unless `--allow-smoke` is passed, so a 1-step
probe cannot masquerade as a Phase 6 result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Flags that map 1:1 onto VLMTrainingConfig fields.
_FIELD_FLAGS: tuple[tuple[str, str, type], ...] = (
    ("--corpus-root", "corpus_root", str),
    ("--metadata-parquet", "metadata_parquet", str),
    ("--output-dir", "output_dir", str),
    ("--base-model", "base_model", str),
    ("--seed", "seed", int),
    ("--batch-size", "batch_size", int),
    ("--gradient-accumulation", "gradient_accumulation", int),
    ("--learning-rate", "learning_rate", float),
    ("--weight-decay", "weight_decay", float),
    ("--epochs", "epochs", int),
    ("--warmup-ratio", "warmup_ratio", float),
    ("--max-steps", "max_steps", int),
    ("--max-wall-seconds", "max_wall_seconds", int),
    ("--max-seq-length", "max_seq_length", int),
    ("--save-every-steps", "save_every_steps", int),
    ("--precision", "precision", str),
    ("--lora-rank", "lora_rank", int),
    ("--lora-alpha", "lora_alpha", int),
    ("--lora-dropout", "lora_dropout", float),
    ("--negative-ratio", "negative_ratio", float),
    ("--resume-from", "resume_from", str),
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser with all Phase 6 subcommands."""
    parser = argparse.ArgumentParser(
        prog="training.vlm.cli",
        description="SatQuery AI Phase 6 — BigEarthNet -> SmolVLM LoRA adaptation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        for flag, _field, kind in _FIELD_FLAGS:
            kwargs: dict[str, Any] = {"default": None}
            if kind is int:
                kwargs["type"] = int
            elif kind is float:
                kwargs["type"] = float
            p.add_argument(flag, **kwargs)
        p.add_argument(
            "--limit", type=int, default=None,
            help="cap the number of patches discovered (smoke runs)",
        )
        p.add_argument("--device", default="cpu")
        p.add_argument("--dry-run", action="store_true")

    for name in ("prepare", "baseline", "train", "evaluate", "smoke"):
        add_common(sub.add_parser(name, help=f"run the {name} step"))

    train_parser = sub.choices["train"]
    train_parser.add_argument(
        "--allow-smoke", action="store_true",
        help="permit exporting an artifact for a CPU-sized (<=20 step) run",
    )

    evaluate_parser = sub.choices["evaluate"]
    evaluate_parser.add_argument("--adapter", default=None)
    evaluate_parser.add_argument("--max-new-tokens", type=int, default=8)

    smoke_parser = sub.choices["smoke"]
    smoke_parser.add_argument("--allow-smoke", action="store_true")

    pkg = sub.add_parser("package", help="package the repo as a Kaggle code dataset")
    pkg.add_argument("--out", default=None)
    return parser


def _config_from_args(args: argparse.Namespace) -> Any:
    """Build the config from the registry plus the CLI overrides."""
    from training.vlm.config import VLMTrainingConfig

    overrides: dict[str, Any] = {}
    for flag, field, _kind in _FIELD_FLAGS:
        value = getattr(args, flag.lstrip("-").replace("-", "_"), None)
        if value is not None:
            overrides[field] = value
    return VLMTrainingConfig.from_registry(**overrides)


def _build_corpus(cfg: Any, limit: int | None, progress: Any = None) -> Any:
    from training.vlm.dataset import build_corpus

    return build_corpus(cfg, limit=limit, progress=progress)


def _load_model(cfg: Any, device: str) -> tuple[Any, Any]:
    """Load the base model and the pinned processor via the serving loader.

    Reuses `specialists.vqa.model.SmolVLM` so the F5-2 resolution pin is the one
    actually enforced, rather than a second copy that could drift.
    """
    from specialists.vqa.model import SmolVLM

    vlm = SmolVLM(
        checkpoint=cfg.base_model,
        revision=cfg.revision,
        processor_longest_edge=cfg.processor_longest_edge,
        device=device,
        do_image_splitting=cfg.do_image_splitting,
    )
    return vlm.model, vlm.processor


def _corpus_summary(corpus: Any, cfg: Any) -> dict[str, Any]:
    return {
        "split_counts": corpus.split_counts(),
        "scene_counts": corpus.scene_counts(),
        "class_counts": corpus.class_counts,
        "render": corpus.render,
        "families": list(corpus.families),
        "split_info": getattr(corpus, "split_info", {}),
        "warnings": list(corpus.warnings),
        "coverage": corpus.coverage.to_dict() if corpus.coverage is not None else None,
        "n_samples": len(corpus),
        "seed": cfg.seed,
    }


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------
def cmd_prepare(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _config_from_args(args)
    corpus = _build_corpus(cfg, args.limit)
    summary = _corpus_summary(corpus, cfg)
    return {"command": "prepare", "config_hash": cfg.config_hash, **summary}


def cmd_baseline(args: argparse.Namespace) -> dict[str, Any]:
    from training.vlm.evaluate import evaluate_split

    cfg = _config_from_args(args)
    corpus = _build_corpus(cfg, args.limit)
    model, processor = _load_model(cfg, args.device)
    val = evaluate_split(model, processor, corpus.by_split("val"), device=args.device)
    test = evaluate_split(model, processor, corpus.by_split("test"), device=args.device)
    return {
        "command": "baseline",
        "config_hash": cfg.config_hash,
        "val": val.to_dict(),
        "test": test.to_dict(),
    }


def cmd_train(args: argparse.Namespace) -> dict[str, Any]:
    from training.vlm.artifact import export_adapter
    from training.vlm.lora import attach_lora, describe_frozen
    from training.vlm.run_manifest import build_run_manifest
    from training.vlm.trainer import latest_checkpoint, resume_from_checkpoint, train

    cfg = _config_from_args(args)
    corpus = _build_corpus(cfg, args.limit)
    model, processor = _load_model(cfg, args.device)

    model, lora_report = attach_lora(model, cfg)
    frozen_report = describe_frozen(model)

    if cfg.resume_from:
        checkpoint = (
            latest_checkpoint(cfg.output_dir)
            if cfg.resume_from == "latest"
            else cfg.resume_from
        )
        if checkpoint:
            resume_from_checkpoint(model, checkpoint)

    state = train(cfg, corpus, processor, model, device=args.device)

    if args.dry_run:
        return {
            "command": "train",
            "dry_run": True,
            "state": state.to_dict(),
            "lora": lora_report.to_dict(),
        }

    from datetime import datetime, timezone

    from training.vlm.formatting import describe_prompt_contract

    manifest = build_run_manifest(
        config=cfg,
        corpus_summary=_corpus_summary(corpus, cfg),
        lora_report=lora_report,
        frozen_report=frozen_report,
        prompt_contract=describe_prompt_contract(processor),
        state=state,
        started_at=datetime.now(timezone.utc).isoformat(),
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    artifact_dir = export_adapter(
        model, processor, cfg.output_dir,
        run_manifest=manifest,
        is_smoke=cfg.is_cpu_sized,
        allow_smoke=bool(getattr(args, "allow_smoke", False)),
    )
    # export_adapter already wrote run_manifest.json AND the checksum manifest
    # over it; rewriting the manifest here would invalidate that checksum.
    return {
        "command": "train",
        "config_hash": cfg.config_hash,
        "artifact_dir": str(artifact_dir),
        "state": state.to_dict(),
        "lora": lora_report.to_dict(),
        "frozen": frozen_report,
        "manifest_hash": manifest["manifest_hash"],
    }


def cmd_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from training.vlm.evaluate import evaluate_split
    from training.vlm.lora import load_adapter

    cfg = _config_from_args(args)
    corpus = _build_corpus(cfg, args.limit)
    model, processor = _load_model(cfg, args.device)
    if args.adapter:
        model = load_adapter(model, args.adapter)
    val = evaluate_split(
        model, processor, corpus.by_split("val"),
        device=args.device, max_new_tokens=args.max_new_tokens,
    )
    test = evaluate_split(
        model, processor, corpus.by_split("test"),
        device=args.device, max_new_tokens=args.max_new_tokens,
    )
    return {
        "command": "evaluate",
        "adapter": args.adapter,
        "val": val.to_dict(),
        "test": test.to_dict(),
    }


def cmd_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """A tiny end-to-end CPU path, labelled unmistakably as a smoke test."""
    from training.vlm.artifact import export_adapter, verify_artifact
    from training.vlm.evaluate import evaluate_split
    from training.vlm.lora import attach_lora, describe_frozen
    from training.vlm.trainer import train

    overrides = {
        "max_steps": args.max_steps if args.max_steps is not None else 2,
        "batch_size": args.batch_size if args.batch_size is not None else 1,
        "gradient_accumulation": (
            args.gradient_accumulation
            if args.gradient_accumulation is not None
            else 1
        ),
        "output_dir": args.output_dir or "artifacts/vlm/smoke",
        "max_wall_seconds": args.max_wall_seconds or 600,
    }
    from training.vlm.config import VLMTrainingConfig

    cfg = VLMTrainingConfig.from_registry(**overrides)
    limit = args.limit if args.limit is not None else 4
    corpus = _build_corpus(cfg, limit)
    model, processor = _load_model(cfg, args.device)
    model, lora_report = attach_lora(model, cfg)
    frozen_report = describe_frozen(model)

    state = train(cfg, corpus, processor, model, device=args.device)

    val = evaluate_split(
        model, processor, corpus.by_split("val")[:4], device=args.device
    )

    from training.vlm.formatting import describe_prompt_contract
    from training.vlm.run_manifest import build_run_manifest

    manifest = build_run_manifest(
        config=cfg,
        corpus_summary=_corpus_summary(corpus, cfg),
        lora_report=lora_report,
        frozen_report=frozen_report,
        prompt_contract=describe_prompt_contract(processor),
        state=state,
    )
    artifact_dir = export_adapter(
        model, processor, cfg.output_dir,
        run_manifest=manifest,
        is_smoke=True,
        allow_smoke=bool(args.allow_smoke),
    ) if args.allow_smoke else None
    verification = (
        verify_artifact(artifact_dir) if artifact_dir is not None else None
    )

    return {
        "command": "smoke",
        "is_smoke": True,
        "note": (
            "SMOKE TEST — a tiny CPU probe. This is NOT a Phase 6 result; the "
            "adapter trained here learned from a handful of patches for a few "
            "steps and must not be reported as an adaptation."
        ),
        "state": state.to_dict(),
        "lora": lora_report.to_dict(),
        "frozen": frozen_report,
        "val_metrics": val.to_dict(),
        "artifact_dir": str(artifact_dir) if artifact_dir else None,
        "artifact_verification": verification,
    }


def cmd_package(args: argparse.Namespace) -> dict[str, Any]:
    """Package the repo as a Kaggle code dataset, reusing the existing script."""
    import subprocess

    repo_root = Path(__file__).resolve().parent.parent.parent
    script = repo_root / "scripts" / "package_kaggle_code.py"
    cmd = [sys.executable, str(script)]
    if args.out:
        cmd += ["--out", args.out]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "command": "package",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


_COMMANDS = {
    "prepare": cmd_prepare,
    "baseline": cmd_baseline,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "smoke": cmd_smoke,
    "package": cmd_package,
}


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = _COMMANDS[args.command]
    try:
        result = handler(args)
    except Exception as exc:  # noqa: BLE001 - the CLI reports any failure as JSON
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
