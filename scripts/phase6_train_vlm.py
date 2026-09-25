"""SatQuery AI — Phase 6 Kaggle entrypoint (single executable job).

Runs the entire Phase 6 workstream in one process, end to end:

    1. resolve the config from the frozen registry + CLI flags + an optional JSON
    2. build the BigEarthNet -> SmolVLM instruction corpus (scene-disjoint split)
    3. evaluate the UN-ADAPTED base model on val and test  (the baseline)
    4. attach a language-model-scoped LoRA and train
    5. evaluate the ADAPTED model on val and test
    6. apply the predeclared acceptance rule (contract item V)
    7. export the adapter + run manifest + checksum manifest

    python scripts/phase6_train_vlm.py \
        --corpus-root /kaggle/input/bigearthnet/BigEarthNet-S2 \
        --metadata-parquet /kaggle/input/bigearthnet/metadata.parquet \
        --output-dir /kaggle/working/phase6_adapter \
        --max-wall-seconds 27000

**No source file is edited to run this.** Every knob is a flag or a JSON file.
The ≤8 h T4x2 budget is enforced by `--max-wall-seconds` (default 27000 s, 7.5 h),
which is applied to the WHOLE job: training stops against it, and the four
evaluations are bounded by the same absolute deadline. A run cut short during
evaluation returns INCONCLUSIVE rather than accepting a prefix -- an accuracy
measured on the first N samples is a property of those N samples, not of the
split.

`--eval-limit` (default 1000 per split) bounds how many samples each evaluation
scores. The subset is a pure function of `(split, limit, seed)`, so the baseline
and the adapted model are always scored on the SAME samples -- contract item U
requires that the two be compared on one frozen evaluation set.

The script prints one JSON summary to stdout so a notebook cell can parse it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

#: Default wall budget: 7.5 h, under the 8 h Kaggle limit, leaving room for the
#: two evaluations that run after training.
DEFAULT_MAX_WALL_SECONDS = 27000

#: Default samples scored per split, for each of the four evaluations.
DEFAULT_EVAL_LIMIT = 1000

#: CLI flag -> VLMTrainingConfig field.
_FIELD_FLAGS: tuple[tuple[str, str], ...] = (
    ("--corpus-root", "corpus_root"),
    ("--metadata-parquet", "metadata_parquet"),
    ("--output-dir", "output_dir"),
    ("--base-model", "base_model"),
    ("--batch-size", "batch_size"),
    ("--gradient-accumulation", "gradient_accumulation"),
    ("--learning-rate", "learning_rate"),
    ("--epochs", "epochs"),
    ("--lora-rank", "lora_rank"),
    ("--lora-alpha", "lora_alpha"),
    ("--lora-dropout", "lora_dropout"),
    ("--max-steps", "max_steps"),
    ("--max-wall-seconds", "max_wall_seconds"),
    ("--max-seq-length", "max_seq_length"),
    ("--save-every-steps", "save_every_steps"),
    ("--precision", "precision"),
    ("--seed", "seed"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 BigEarthNet -> SmolVLM LoRA job")
    parser.add_argument(
        "--config-json", default=None,
        help="a JSON file of VLMTrainingConfig overrides (CLI flags win)",
    )
    for flag, _field in _FIELD_FLAGS:
        parser.add_argument(flag, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap discovered patches (smoke runs only)")
    parser.add_argument("--eval-limit", type=int, default=DEFAULT_EVAL_LIMIT,
                        help=(
                            "max samples scored per split, per evaluation "
                            f"(default {DEFAULT_EVAL_LIMIT}; 0 or negative means "
                            "no limit). Seeded from --seed so the baseline and "
                            "the adapted model see the same samples."
                        ))
    parser.add_argument("--device", default=None,
                        help="cpu or cuda; auto-detected when omitted")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="skip the base-model baseline evaluation")
    parser.add_argument(
        "--allow-smoke-export", action="store_true",
        help=(
            "permit exporting an artifact from a CPU-sized run. Without this, a "
            "CPU run trains and evaluates but REFUSES to export, because a probe "
            "must not produce something that reads as a Phase 6 result. When set, "
            "the manifest records is_smoke=true."
        ))
    parser.add_argument("--dry-run", action="store_true",
                        help="build config + corpus and print the plan; do not train")
    return parser.parse_args(argv)


def _resolve_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Merge the JSON file and the CLI flags (CLI wins)."""
    overrides: dict[str, object] = {}
    if args.config_json:
        payload = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        overrides.update(payload)
    for flag, field in _FIELD_FLAGS:
        value = getattr(args, flag.lstrip("-").replace("-", "_"), None)
        if value is not None:
            overrides[field] = _coerce(field, value)
    if "max_wall_seconds" not in overrides:
        overrides["max_wall_seconds"] = DEFAULT_MAX_WALL_SECONDS
    return overrides


def _coerce(field: str, value: object) -> object:
    """Coerce a CLI string to the field's type (argparse gives strings for JSON)."""
    int_fields = {
        "batch_size", "gradient_accumulation", "epochs", "lora_rank", "lora_alpha",
        "max_steps", "max_wall_seconds", "max_seq_length", "save_every_steps", "seed",
    }
    float_fields = {"learning_rate", "lora_dropout", "warmup_ratio", "weight_decay"}
    if field in int_fields:
        return int(value)  # type: ignore[arg-type]
    if field in float_fields:
        return float(value)  # type: ignore[arg-type]
    return value


def _detect_device(requested: str | None) -> str:
    if requested:
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def _eval_subset(
    samples: list[object], *, limit: int | None, seed: int, split: str
) -> list[object]:
    """A deterministic, seeded evaluation subset for one split.

    Contract item U requires the baseline and the adapted model to be scored on
    the SAME frozen evaluation set. So this is a pure function of
    `(split, limit, seed)` and is computed **once**, before the baseline runs,
    then handed to both models. Re-drawing it per model would silently compare
    two different question sets and move the V1 delta.

    `random.Random(str)` seeds via SHA-512 (CPython `version=2`), so the choice
    does not depend on `PYTHONHASHSEED` and is reproducible across machines.

    Args:
        samples: the split's samples.
        limit: max samples to score; `None` or `<= 0` means no limit.
        seed: the run seed.
        split: split name, so each split draws an independent subset.

    Returns:
        The subset, in ascending original order.
    """
    items = list(samples)
    if limit is None or limit <= 0 or len(items) <= limit:
        return items
    rng = random.Random(f"{seed}:{split}")
    picked = sorted(rng.sample(range(len(items)), limit))
    return [items[i] for i in picked]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = _detect_device(args.device)

    from training.vlm.config import VLMTrainingConfig

    cfg = VLMTrainingConfig.from_registry(**_resolve_overrides(args))
    started_at = datetime.now(timezone.utc).isoformat()
    # One absolute deadline for the WHOLE job. Evaluation is bounded by the same
    # clock as training, so a run that runs out of budget while scoring cannot
    # report a result for a prefix of a split as if it were the split.
    job_started = time.monotonic()
    job_deadline = (
        job_started + cfg.max_wall_seconds if cfg.max_wall_seconds else None
    )

    if args.dry_run:
        print(json.dumps({
            "command": "phase6",
            "dry_run": True,
            "device": device,
            "eval_limit": args.eval_limit,
            "max_wall_seconds": cfg.max_wall_seconds,
            "config": cfg.to_dict(),
        }, indent=2, default=str))
        return 0

    from training.vlm.artifact import export_adapter
    from training.vlm.dataset import build_corpus
    from training.vlm.evaluate import (
        decide_acceptance,
        describe_acceptance_rule,
        evaluate_split,
    )
    from training.vlm.formatting import describe_prompt_contract
    from training.vlm.lora import attach_lora, describe_frozen
    from training.vlm.run_manifest import (
        build_run_manifest,
        manifest_hash,
        write_run_manifest,
    )
    from training.vlm.trainer import train

    # 2. corpus ------------------------------------------------------------
    corpus = build_corpus(cfg, limit=args.limit)
    corpus_summary = {
        "split_counts": corpus.split_counts(),
        "scene_counts": corpus.scene_counts(),
        "class_counts": corpus.class_counts,
        "render": corpus.render,
        "families": list(corpus.families),
        "split_info": getattr(corpus, "split_info", {}),
        "warnings": list(corpus.warnings),
        "coverage": corpus.coverage.to_dict() if corpus.coverage else None,
        "n_samples": len(corpus),
        "seed": cfg.seed,
    }

    # load the base model + pinned processor through the serving loader
    from specialists.vqa.model import SmolVLM

    vlm = SmolVLM(
        checkpoint=cfg.base_model,
        revision=cfg.revision,
        processor_longest_edge=cfg.processor_longest_edge,
        device=device,
        do_image_splitting=cfg.do_image_splitting,
    )
    model, processor = vlm.model, vlm.processor

    # 3. baseline (un-adapted) --------------------------------------------
    # The evaluation subsets are drawn ONCE, here, and reused for the adapted
    # model below. Item U requires one frozen evaluation set; re-drawing per
    # model would compare two different question sets and move the V1 delta.
    eval_sets = {
        split: _eval_subset(
            corpus.by_split(split),
            limit=args.eval_limit,
            seed=cfg.seed,
            split=split,
        )
        for split in ("val", "test")
    }
    baseline_val = baseline_test = None
    if not args.skip_baseline:
        baseline_val = evaluate_split(
            model, processor, eval_sets["val"], device=device, deadline=job_deadline
        )
        baseline_test = evaluate_split(
            model, processor, eval_sets["test"], device=device, deadline=job_deadline
        )

    # 4. train -------------------------------------------------------------
    model, lora_report = attach_lora(model, cfg)
    frozen_report = describe_frozen(model)
    state = train(cfg, corpus, processor, model, device=device)

    # 5. adapted evaluation -------------------------------------------------
    adapted_val = evaluate_split(
        model, processor, eval_sets["val"], device=device, deadline=job_deadline
    )
    adapted_test = evaluate_split(
        model, processor, eval_sets["test"], device=device, deadline=job_deadline
    )

    evaluation_budget = {
        "requested_limit_per_split": args.eval_limit,
        "seed": cfg.seed,
        "selection": (
            "seeded subset drawn once and reused for baseline and adapted "
            "(contract item U); reproducible from (split, limit, seed)"
        ),
        "wall_budget_seconds": cfg.max_wall_seconds,
        "available_per_split": {s: len(corpus.by_split(s)) for s in ("val", "test")},
        "selected_per_split": {s: len(eval_sets[s]) for s in ("val", "test")},
        "n_evaluated": {
            "baseline_val": baseline_val.n if baseline_val else None,
            "baseline_test": baseline_test.n if baseline_test else None,
            "adapted_val": adapted_val.n,
            "adapted_test": adapted_test.n,
        },
        "truncated": {
            "baseline_val": bool(baseline_val.truncated) if baseline_val else None,
            "baseline_test": bool(baseline_test.truncated) if baseline_test else None,
            "adapted_val": bool(adapted_val.truncated),
            "adapted_test": bool(adapted_test.truncated),
        },
    }

    # 6. artifact + verification (V4) --------------------------------------
    from training.vlm.artifact import refresh_artifact_checksums, verify_artifact

    manifest = build_run_manifest(
        config=cfg,
        corpus_summary=corpus_summary,
        lora_report=lora_report,
        frozen_report=frozen_report,
        prompt_contract=describe_prompt_contract(processor),
        state=state,
        metrics=adapted_val.to_dict(),
        baseline=(baseline_val.to_dict() if baseline_val else None),
        metrics_test=adapted_test.to_dict(),
        baseline_test=(baseline_test.to_dict() if baseline_test else None),
        evaluation_budget=evaluation_budget,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    artifact_dir = export_adapter(
        model, processor, cfg.output_dir,
        run_manifest=manifest,
        is_smoke=cfg.is_cpu_sized,
        allow_smoke=args.allow_smoke_export,
    )
    # `export_adapter` writes its OWN manifest copy with `is_smoke`,
    # `artifact_dir` and `adapter_sha256` filled in from the bytes it actually
    # wrote (`artifact.py` copies the dict it is handed). Re-read that file
    # rather than rewriting our pre-export dict: doing the latter silently DROPS
    # the smoke label and the adapter identity from the artifact's own record.
    # Measured 2026-09-23 -- the rewrite left `is_smoke: null` and
    # `adapter_sha256: null` in the shipped run_manifest.json.
    manifest = json.loads(
        (Path(artifact_dir) / "run_manifest.json").read_text(encoding="utf-8")
    )

    # V4 reload check. It must be handed the UNWRAPPED base model: the adapter's
    # saved `target_modules` regex is anchored `^model\.text_model\.`, and a PEFT
    # wrapper prefixes its module names (`base_model.model.model.text_model.…`),
    # so reloading onto the wrapper raises NoMatchingPeftModuleError and the
    # check reports a false negative. Measured 2026-09-23: reloading onto
    # `get_base_model()` succeeds, onto the wrapper fails.
    verification = verify_artifact(
        artifact_dir, base_model=model.get_base_model()
    )

    # 7. the predeclared acceptance decision -------------------------------
    if baseline_val is None:
        decision = {
            "status": "INCONCLUSIVE",
            "reasons": ["baseline evaluation was skipped (--skip-baseline)"],
        }
    else:
        decision_obj = decide_acceptance(
            baseline_val, adapted_val,
            baseline_test=baseline_test, adapted_test=adapted_test,
            run_completed=state.run_completed,
            artifact_ok=bool(verification.get("ok")) and lora_report.trainable_params > 0,
        )
        decision = decision_obj.to_dict()

    # The decision is only knowable after the artifact exists and verifies, so it
    # is added to the manifest now and the checksum manifest is regenerated over
    # the rewrite -- otherwise the checksum would silently stop matching.
    manifest["decision"] = decision
    manifest["manifest_hash"] = manifest_hash(manifest)
    write_run_manifest(manifest, Path(artifact_dir) / "run_manifest.json")
    refresh_artifact_checksums(artifact_dir)

    summary = {
        "command": "phase6",
        "device": device,
        "config_hash": cfg.config_hash,
        "artifact_dir": str(artifact_dir),
        "manifest_hash": manifest["manifest_hash"],
        "trainable_params": lora_report.trainable_params,
        "lora": lora_report.to_dict(),
        "frozen": frozen_report,
        "training_state": state.to_dict(),
        "baseline_val": baseline_val.to_dict() if baseline_val else None,
        "adapted_val": adapted_val.to_dict(),
        "baseline_test": baseline_test.to_dict() if baseline_test else None,
        "adapted_test": adapted_test.to_dict(),
        "decision": decision,
        "acceptance_rule": describe_acceptance_rule(),
        "evaluation_budget": evaluation_budget,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
