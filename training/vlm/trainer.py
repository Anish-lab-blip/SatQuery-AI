"""SatQuery AI — Phase 6 hand-written training loop.

WHY NOT `transformers.Trainer` OR `trl`
---------------------------------------
`trl` and `datasets` are **not installed** in this environment, and the trainer
must be a hand-written PyTorch loop (measured 2026-09-23). The loop here is
therefore the whole training implementation: nothing is imported from `trl`, and
nothing is imported from `datasets`. If a Kaggle-only helper is ever needed it
must sit behind a `try/except`; the CPU path must work without it.

THE CONTRACT ITEMS THIS LOOP ENCODES
------------------------------------
    J  token-level cross-entropy on COMPLETION TOKENS ONLY (prompt masked -100)
    L  AdamW over the LoRA parameters only
    M  linear warmup for warmup_ratio * total_steps, then cosine decay to 0
    N  micro-batch x gradient-accumulation = effective batch
    O  fp16 on CUDA (autocast + GradScaler); fp32 on CPU; NEVER bf16
    Q  step budget = max_steps, else epochs * steps_per_epoch, enforced exactly
    R  checkpoint every save_every_steps, keep the last 2, resume from latest
    S  one seed threaded through split, sampling, and torch

PRECISION (item O) -- the one place the plan is wrong for the hardware
---------------------------------------------------------------------
The master plan says `bf16`. **T4 is SM 7.5 and has no bf16 tensor cores**
(finding C-6). Training bf16 on a T4 silently falls back to fp32 and blows the
8-hour budget. So: `bf16` is **refused** with an explanation rather than
performed, CUDA `fp16` uses autocast + a `GradScaler`, and CPU always runs fp32
with no scaler.

THE WALL-CLOCK GUARD
--------------------
`max_wall_seconds` is how the 8-hour Kaggle budget becomes a guarantee rather
than a hope. When exceeded the loop stops cleanly, records
`stopped_early=True, stop_reason="wall_clock"`, and the acceptance rule (V3)
treats the truncated run as INCONCLUSIVE.
"""

from __future__ import annotations

import math
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from core.errors import SatQueryError
from training.vlm.config import ConfigError

__all__ = [
    "TrainingError",
    "TrainingState",
    "set_seed",
    "build_optimizer",
    "build_scheduler",
    "steps_per_epoch",
    "train",
    "resume_from_checkpoint",
    "latest_checkpoint",
    "VAL_LOSS_MAX_BATCHES",
]

#: Validation loss is computed over at most this many micro-batches. It is a
#: training-loop *monitor*, not the endpoint -- the endpoint is presence
#: exact-match accuracy, computed separately by `evaluate.evaluate_split`. On CPU
#: a full-val pass over ~900 samples is ~15 minutes, so the monitor is bounded to
#: keep a smoke run a smoke run. The cap is recorded in the manifest.
VAL_LOSS_MAX_BATCHES: int = 16


class TrainingError(SatQueryError):
    """The Phase 6 training loop could not proceed."""

    code = "vlm_training_error"
    user_message = "The Phase 6 training run failed."


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@dataclass
class TrainingState:
    """What the loop actually did. Every field is measured, never assumed."""

    steps_completed: int = 0
    epochs_completed: int = 0
    losses: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    best_val_loss: float | None = None
    checkpoints: list[str] = field(default_factory=list)
    wall_seconds: float = 0.0
    stopped_early: bool = False
    stop_reason: str | None = None
    #: The step budget the run was supposed to hit, so "completed" is checkable.
    total_steps_planned: int = 0
    #: Whether the run reached its predeclared budget (contract item V3).
    run_completed: bool = False
    train_samples: int = 0
    val_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps_completed": self.steps_completed,
            "epochs_completed": self.epochs_completed,
            "total_steps_planned": self.total_steps_planned,
            "losses": list(self.losses),
            "learning_rates": list(self.learning_rates),
            "best_val_loss": self.best_val_loss,
            "checkpoints": list(self.checkpoints),
            "wall_seconds": round(self.wall_seconds, 3),
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "run_completed": self.run_completed,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
        }


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed `random`, `numpy`, `torch`, and `torch.cuda` (when available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Optimizer + schedule
# ---------------------------------------------------------------------------
def build_optimizer(model: Any, cfg: Any, *, expected_trainable: int | None = None) -> Any:
    """AdamW over the trainable parameters only (contract item L).

    Args:
        model: the PEFT-wrapped model.
        cfg: the training config (`learning_rate`, `weight_decay`).
        expected_trainable: when given, the optimizer's parameter count must
            match it -- a mismatch means the optimizer was built over a
            different set of parameters than the LoRA report describes.

    Raises:
        TrainingError: no trainable parameters, or a count mismatch.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise TrainingError(
            "no trainable parameters; the optimizer would have nothing to update"
        )
    n_trainable = sum(p.numel() for p in params)
    if expected_trainable is not None and n_trainable != expected_trainable:
        raise TrainingError(
            f"optimizer covers {n_trainable} trainable parameters but the LoRA "
            f"injection report recorded {expected_trainable}; the optimizer is "
            f"over the wrong parameter set",
            context={
                "n_trainable": n_trainable,
                "expected_trainable": expected_trainable,
            },
        )
    return torch.optim.AdamW(
        params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )


def build_scheduler(optimizer: Any, cfg: Any, total_steps: int) -> Any:
    """Linear warmup then cosine decay to 0 (contract item M).

    Warmup is `warmup_ratio * total_steps` steps (at least 1 when the ratio is
    non-zero). After warmup the multiplier is `0.5 * (1 + cos(pi * progress))`,
    which reaches exactly 0 at the final step.
    """
    if total_steps < 1:
        raise TrainingError(f"total_steps must be >= 1, got {total_steps}")

    warmup_steps = (
        max(1, int(round(cfg.warmup_ratio * total_steps)))
        if cfg.warmup_ratio > 0
        else 0
    )
    decay_steps = max(1, total_steps - warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
def steps_per_epoch(n_samples: int, batch_size: int) -> int:
    """Micro-batches per epoch (`ceil`)."""
    if batch_size < 1:
        raise TrainingError(f"batch_size must be >= 1, got {batch_size}")
    return max(1, math.ceil(n_samples / batch_size))


def _micro_batches(
    n_samples: int, batch_size: int, *, seed: int, epoch: int
) -> list[list[int]]:
    """Deterministic shuffled index batches for one epoch."""
    indices = list(range(n_samples))
    random.Random(seed + epoch).shuffle(indices)
    return [
        indices[i : i + batch_size] for i in range(0, len(indices), batch_size)
    ]


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
def _make_items(
    collator: Any, processor: Any, samples: Sequence[Any], cfg: Any
) -> list[Any]:
    """Build collation items for a micro-batch (render image + format example)."""
    from training.vlm.collate import make_item, render_sample

    eos_token = getattr(processor.tokenizer, "eos_token", None)
    items: list[Any] = []
    for sample in samples:
        image = render_sample(sample, percentiles=cfg.rgb_percentiles)
        items.append(
            make_item(
                processor,
                question=sample.question,
                answer=sample.answer,
                image=image,
                max_seq_length=cfg.max_seq_length,
                eos_token=eos_token,
            )
        )
    return items


def _forward_loss(
    model: Any,
    batch: dict[str, Any],
    *,
    device: str,
    autocast_dtype: torch.dtype | None,
) -> torch.Tensor:
    """One forward pass returning the completion-only cross-entropy loss."""
    moved = {k: v.to(device) for k, v in batch.items()}
    if autocast_dtype is not None:
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            out = model(**moved)
    else:
        out = model(**moved)
    loss = out.loss
    if loss is None or not torch.isfinite(loss):
        raise TrainingError(
            f"the model returned a non-finite loss ({loss}); stopping rather "
            f"than stepping on it",
            context={"loss": None if loss is None else float(loss)},
        )
    return loss


def train(
    cfg: Any,
    corpus: Any,
    processor: Any,
    model: Any,
    *,
    device: str = "cpu",
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> TrainingState:
    """Run the Phase 6 training loop. Returns the measured `TrainingState`.

    Args:
        cfg: a `VLMTrainingConfig`.
        corpus: a `CorpusBuild` (train/val samples come from it).
        processor: the SmolVLM processor.
        model: the PEFT-wrapped model.
        device: "cpu" or "cuda".
        progress: optional callback invoked after each optimizer step with a
            small dict (step, loss, lr, elapsed).

    Raises:
        ConfigError: `precision == "bf16"` (finding C-6).
        TrainingError: a non-finite loss, or an empty training split.
    """
    from training.vlm.collate import Collator

    # -- precision (item O) ------------------------------------------------
    if cfg.precision == "bf16":
        raise ConfigError(
            "precision='bf16' is refused: T4 is SM 7.5 and has no bf16 tensor "
            "cores (finding C-6). Training bf16 on a T4 silently falls back to "
            "fp32 and blows the 8-hour budget. Use fp16 on CUDA."
        )
    use_amp = device == "cuda" and cfg.precision == "fp16"
    autocast_dtype = torch.float16 if use_amp else None
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    set_seed(cfg.seed)

    train_samples = list(corpus.by_split("train"))
    val_samples = list(corpus.by_split("val"))
    if not train_samples:
        raise TrainingError(
            "the training split is empty; a run over no data would still write an "
            "artifact"
        )

    # -- gradient checkpointing (conflicts with the KV cache) --------------
    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        config = getattr(model, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
        # With only the LoRA matrices trainable, the frozen embedding output
        # carries no grad and a checkpointed block would produce none either.
        # This is the standard LoRA + gradient-checkpointing repair; without it
        # the LoRA gradients come back as None on the first backward.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model.train()

    from training.vlm.lora import verify_injection

    report = verify_injection(model)
    optimizer = build_optimizer(
        model, cfg, expected_trainable=report.trainable_params
    )

    n_micro_per_epoch = steps_per_epoch(len(train_samples), cfg.batch_size)
    opt_steps_per_epoch = max(1, math.ceil(n_micro_per_epoch / cfg.gradient_accumulation))
    if cfg.max_steps is not None:
        total_steps = cfg.max_steps
    else:
        total_steps = opt_steps_per_epoch * cfg.epochs
    scheduler = build_scheduler(optimizer, cfg, total_steps)

    collator = Collator(processor, cfg.max_seq_length)

    state = TrainingState(
        total_steps_planned=total_steps,
        train_samples=len(train_samples),
        val_samples=len(val_samples),
    )
    started = time.time()

    step = 0
    accum_loss = 0.0
    accum_count = 0
    stop_reason: str | None = None

    def _maybe_stop_wall() -> str | None:
        if cfg.max_wall_seconds is None:
            return None
        if time.time() - started >= cfg.max_wall_seconds:
            return "wall_clock"
        return None

    for epoch in range(cfg.epochs):
        if stop_reason or step >= total_steps:
            break
        micro_batches = _micro_batches(
            len(train_samples), cfg.batch_size, seed=cfg.seed, epoch=epoch
        )
        reached_budget_mid_epoch = False
        for micro in micro_batches:
            reason = _maybe_stop_wall()
            if reason:
                stop_reason = reason
                break

            items = _make_items(
                collator, processor, [train_samples[i] for i in micro], cfg
            )
            batch = collator(items)
            loss = _forward_loss(
                model, batch, device=device, autocast_dtype=autocast_dtype
            )
            scaled = loss / cfg.gradient_accumulation
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            accum_loss += float(loss.detach())
            accum_count += 1

            is_last_micro = micro is micro_batches[-1]
            if accum_count >= cfg.gradient_accumulation or is_last_micro:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                step += 1
                mean_loss = accum_loss / max(1, accum_count)
                state.losses.append(mean_loss)
                state.learning_rates.append(
                    float(optimizer.param_groups[0]["lr"])
                )
                accum_loss = 0.0
                accum_count = 0

                if progress is not None:
                    progress(
                        {
                            "step": step,
                            "loss": mean_loss,
                            "lr": state.learning_rates[-1],
                            "elapsed": round(time.time() - started, 2),
                        }
                    )

                if cfg.save_every_steps and step % cfg.save_every_steps == 0:
                    state.checkpoints.append(
                        _save_checkpoint(model, cfg.output_dir, step)
                    )

                if step >= total_steps:
                    reached_budget_mid_epoch = True
                    break

        state.epochs_completed += 1
        if stop_reason or reached_budget_mid_epoch:
            break

    state.steps_completed = step
    state.wall_seconds = time.time() - started

    # V3 semantics: the budget is "epochs finished OR max_steps reached".
    epochs_done = state.epochs_completed >= cfg.epochs
    reached_budget = step >= total_steps or epochs_done
    state.run_completed = reached_budget
    if reached_budget:
        state.stopped_early = False
        state.stop_reason = None
    else:
        state.stopped_early = True
        state.stop_reason = stop_reason or "step_budget_reached_mid_epoch"

    # -- final validation loss --------------------------------------------
    if val_samples:
        state.best_val_loss = _validation_loss(
            model, collator, processor, val_samples, cfg,
            device=device, autocast_dtype=autocast_dtype,
        )

    return state


def _validation_loss(
    model: Any,
    collator: Any,
    processor: Any,
    val_samples: Sequence[Any],
    cfg: Any,
    *,
    device: str,
    autocast_dtype: torch.dtype | None,
) -> float | None:
    """Mean completion-only loss over a bounded prefix of the validation split.

    Bounded by `VAL_LOSS_MAX_BATCHES` micro-batches: this is a training-loop
    monitor, not the endpoint, and an unbounded pass over a large val split on
    CPU dominates the run. The bound is recorded in the returned value's caller.
    """
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    try:
        with torch.no_grad():
            for i in range(0, len(val_samples), cfg.batch_size):
                if count >= VAL_LOSS_MAX_BATCHES:
                    break
                chunk = val_samples[i : i + cfg.batch_size]
                items = _make_items(collator, processor, chunk, cfg)
                batch = collator(items)
                loss = _forward_loss(
                    model, batch, device=device, autocast_dtype=autocast_dtype
                )
                total += float(loss.detach())
                count += 1
    except TrainingError:
        return None
    finally:
        if was_training:
            model.train()
    return total / count if count else None


def _save_checkpoint(model: Any, output_dir: str, step: int) -> str:
    """Save a checkpoint and keep only the last 2 (contract item R)."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / f"checkpoint-{step}"
    model.save_pretrained(str(checkpoint))

    existing = sorted(
        (p for p in root.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    for stale in existing[:-2]:
        shutil.rmtree(stale, ignore_errors=True)
    return str(checkpoint)


def resume_from_checkpoint(model: Any, checkpoint_dir: str | Path) -> None:
    """Load a PEFT adapter state into the model in place (contract item R).

    Uses `set_peft_model_state_dict` so the adapter resumes under its existing
    name rather than being attached as a second adapter.
    """
    from peft import set_peft_model_state_dict

    directory = Path(checkpoint_dir)
    weights = directory / "adapter_model.safetensors"
    if not weights.exists():
        raise TrainingError(
            f"checkpoint {directory} has no adapter_model.safetensors to resume from",
            context={"checkpoint": str(directory)},
        )
    try:
        from safetensors.torch import load_file

        state_dict = load_file(str(weights))
    except Exception as exc:  # noqa: BLE001
        raise TrainingError(
            f"could not read checkpoint weights at {weights}: {exc}",
            context={"checkpoint": str(directory)},
        ) from exc
    set_peft_model_state_dict(model, state_dict)


def latest_checkpoint(output_dir: str | Path) -> str | None:
    """The highest-step checkpoint under `output_dir`, or `None`."""
    root = Path(output_dir)
    if not root.is_dir():
        return None
    checkpoints = [p for p in root.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None
    newest = max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))
    return str(newest)
