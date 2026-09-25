"""SatQuery AI — Phase 6 VLM adaptation configuration.

WHERE THE VALUES COME FROM
--------------------------
The recipe is **already declared** in the frozen configuration registry
(`configs/base.yaml` section `training:`), and finding C-6 resolved the one place
it disagreed with the master plan:

    lora_rank: 16                  plan section 43: rank 16
    lora_alpha: 32                 plan section 43: alpha 32
    lora_dropout: 0.05             plan section 43: dropout 0.05
    vlm_learning_rate: 0.0002      plan section 43: lr 2e-4
    vlm_batch_size: 2              plan section 43: effective_batch_size 16
    vlm_gradient_accumulation: 8     -> 2 x 8 = 16, which MATCHES the plan
    vlm_epochs: 1                  plan section 43: epochs 1
    warmup_ratio: 0.05             plan section 43: warmup_ratio 0.05
    weight_decay: 0.01             plan section 43: weight_decay 0.01
    precision: fp16                plan says bf16 -- CORRECTED, see below
    save_every_steps: 500

`precision` is the single divergence, and the repository is authoritative:
**T4 is SM 7.5 and has no bf16 tensor cores** (finding C-6,
`docs/PHASE0_CONTRACT_VALIDATION.md:291`, `docs/ARCHITECTURE_FREEZE.md:159`). The
plan's `bf16` was written before the P100 retirement made T4x2 the target
(plan section 45). Training bf16 on a T4 silently falls back to fp32 and blows
the 8-hour budget. The correction is recorded in every run manifest so the
deviation from the plan is auditable rather than invisible.

WHY THIS FILE EXISTS AT ALL, GIVEN THE VALUES ARE IN `base.yaml`
--------------------------------------------------------------
Three reasons, none of which is "to have a config object":

  1. **The keys are declared but read by no code.** Measured 2026-09-23: the only
     `training.*` key any module reads is `training.precision`
     (`core/config.py:108`). The other ten are inert. This module is what makes
     them load-bearing.
  2. **The registry is frozen.** `Config.hash` is a sha256 over the whole YAML and
     run manifests cite it, so a new knob MUST NOT be added to `base.yaml`. Every
     Phase 6-specific setting that the registry does not already carry is a
     dataclass field here with a code default, overridable from the command line.
  3. **The registry is not a training contract.** It carries no target modules, no
     max sequence length, no loss masking rule. Those are contract decisions and
     belong where they can be documented and validated.

NO KEY IS ADDED TO `configs/base.yaml` BY THIS MODULE. Adding one would move
`Config.hash` and invalidate every manifest that cites it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

__all__ = [
    "PRECISION_CHOICES",
    "LORA_TARGET_MODULES",
    "VLMTrainingConfig",
    "ConfigError",
]


class ConfigError(ValueError):
    """A Phase 6 training setting is invalid or contradictory."""


#: Allowed precision values, matching `core/config.py`'s validation.
PRECISION_CHOICES: tuple[str, ...] = ("fp16", "bf16", "fp32")

#: LoRA target modules -- contract item G.
#:
#: SmolVLM's language model is a Llama-style decoder, so these are the standard
#: attention and MLP projections. They are chosen as a SET rather than discovered
#: at runtime because:
#:
#:   * a runtime `findall` that silently matches nothing produces a LoRA with
#:     ZERO trainable parameters, which trains happily and learns nothing -- the
#:     worst failure available here, and one this module refuses by measuring the
#:     injected parameter count instead of trusting the call;
#:   * the vision tower is deliberately ABSENT from this list. Plan section 43
#:     freezes the vision backbone; adding `q_proj` blindly across the whole model
#:     would adapt the image encoder too, which is exactly what the plan forbids.
LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class VLMTrainingConfig:
    """The Phase 6 training recipe, resolved and validated.

    Built with `from_registry()` so the defaults come from `base.yaml` rather than
    being re-typed here -- a second copy of `lora_rank: 16` in Python would be a
    value that can drift from the hashed one.
    """

    # -- from the frozen registry -----------------------------------------
    base_model: str = "HuggingFaceTB/SmolVLM-500M-Instruct"
    revision: str | None = None
    processor_longest_edge: int = 512
    do_image_splitting: bool = True

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation: int = 8
    epochs: int = 1
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    precision: str = "fp16"
    gradient_checkpointing: bool = True
    save_every_steps: int = 500

    # -- Phase 6 contract settings (code defaults; NOT in the registry) ----
    #: Contract item G. See `LORA_TARGET_MODULES` for why the vision tower is
    #: absent.
    lora_target_modules: tuple[str, ...] = LORA_TARGET_MODULES
    #: Contract item P. Bounds activation memory for a 500M model at 512 px.
    max_seq_length: int = 512
    #: Contract item S.
    seed: int = 42
    #: Contract item F. Scene-disjoint by Sentinel tile.
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    #: Contract item Q. A hard cap that makes the 8-hour budget enforceable
    #: rather than hoped for. `None` means "use `epochs`".
    max_steps: int | None = None
    #: Contract item Q / the 8-hour Kaggle budget. When the loop exceeds this many
    #: seconds it stops cleanly and records `stopped_early` with
    #: `stop_reason="wall_clock"`, which the acceptance rule treats as
    #: INCONCLUSIVE (V3). This is how the budget becomes a guarantee rather than a
    #: hope. `None` disables the guard.
    max_wall_seconds: int | None = None
    #: Resume a run from a checkpoint directory (contract item R). `None` starts
    #: fresh; the CLI's `--resume-from latest` resolves to the newest checkpoint.
    resume_from: str | None = None
    #: Instruction families to generate. Restricted by default -- see
    #: `training.vlm.dataset` for why multi-label enumeration is disabled on this
    #: corpus.
    instruction_families: tuple[str, ...] = ("presence",)
    #: How many absent-class questions per present class.
    negative_ratio: float = 1.0
    #: RGB percentile stretch, matching `specialists/optical_sar/radiometry.py`.
    rgb_percentiles: tuple[float, float] = (2.0, 98.0)

    # -- data locations ----------------------------------------------------
    corpus_root: str = "data/bigearthnet_v2/reben/BigEarthNet-S2"
    metadata_parquet: str = "data/bigearthnet_v2/metadata.parquet"
    output_dir: str = "artifacts/vlm/bigearthnet_lora_v001"

    #: Populated by `from_registry` so a run manifest can cite the exact hash the
    #: recipe was read from.
    config_hash: str | None = None
    #: Recorded when the plan and the registry disagree, so the deviation is
    #: visible in the artifact rather than buried in a commit message.
    plan_deviations: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    # -- validation --------------------------------------------------------
    def validate(self) -> None:
        """Raise `ConfigError` on any contradictory or unusable setting.

        Validation is explicit rather than relying on dataclass types because the
        values arrive from YAML and a CLI, where `"16"` and `16` are both possible
        and only one of them works.
        """
        if self.precision not in PRECISION_CHOICES:
            raise ConfigError(
                f"precision must be one of {list(PRECISION_CHOICES)}, "
                f"got {self.precision!r}"
            )
        if self.lora_rank < 1:
            raise ConfigError(f"lora_rank must be >= 1, got {self.lora_rank}")
        if self.lora_alpha < 1:
            raise ConfigError(f"lora_alpha must be >= 1, got {self.lora_alpha}")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ConfigError(
                f"lora_dropout must be in [0,1), got {self.lora_dropout}"
            )
        if self.learning_rate <= 0:
            raise ConfigError(
                f"learning_rate must be > 0, got {self.learning_rate}"
            )
        if self.batch_size < 1:
            raise ConfigError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.gradient_accumulation < 1:
            raise ConfigError(
                f"gradient_accumulation must be >= 1, got "
                f"{self.gradient_accumulation}"
            )
        if self.epochs < 1 and self.max_steps is None:
            raise ConfigError(
                "epochs must be >= 1 unless max_steps is set; a run with neither "
                "would train for no steps and still write an artifact"
            )
        if self.max_steps is not None and self.max_steps < 1:
            raise ConfigError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.max_wall_seconds is not None and self.max_wall_seconds < 1:
            raise ConfigError(
                f"max_wall_seconds must be >= 1, got {self.max_wall_seconds}"
            )
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ConfigError(
                f"warmup_ratio must be in [0,1), got {self.warmup_ratio}"
            )
        if self.weight_decay < 0:
            raise ConfigError(
                f"weight_decay must be >= 0, got {self.weight_decay}"
            )
        if self.max_seq_length < 1:
            raise ConfigError(
                f"max_seq_length must be >= 1, got {self.max_seq_length}"
            )
        if not self.lora_target_modules:
            raise ConfigError(
                "lora_target_modules is empty; a LoRA with no target modules has "
                "zero trainable parameters and would train without learning"
            )
        if self.negative_ratio < 0:
            raise ConfigError(
                f"negative_ratio must be >= 0, got {self.negative_ratio}"
            )
        if not (0.0 < self.train_ratio < 1.0):
            raise ConfigError(
                f"train_ratio must be in (0,1), got {self.train_ratio}"
            )
        if not 0.0 <= self.val_ratio < 1.0:
            raise ConfigError(
                f"val_ratio must be in [0,1), got {self.val_ratio}"
            )
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ConfigError(
                f"train_ratio + val_ratio must be < 1, got "
                f"{self.train_ratio} + {self.val_ratio}"
            )
        lo, hi = self.rgb_percentiles
        if not 0.0 <= lo < hi <= 100.0:
            raise ConfigError(
                f"rgb_percentiles must satisfy 0 <= lo < hi <= 100, got "
                f"({lo}, {hi})"
            )
        unknown = set(self.instruction_families) - {"presence", "dominance", "multi"}
        if unknown:
            raise ConfigError(
                f"unknown instruction families {sorted(unknown)}; expected a "
                f"subset of presence/dominance/multi"
            )

    # -- derived -----------------------------------------------------------
    @property
    def effective_batch_size(self) -> int:
        """Micro-batch x accumulation. The plan's section 43 figure is 16."""
        return self.batch_size * self.gradient_accumulation

    @property
    def is_cpu_sized(self) -> bool:
        """Whether the settings look like a smoke test rather than a real run.

        Used to refuse the label "trained" for a run that was never a training
        run. A 1-step CPU probe must not produce an artifact that reads as a
        Phase 6 result.
        """
        return self.max_steps is not None and self.max_steps <= 20

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the run manifest. Tuples become lists for JSON."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = list(value) if isinstance(value, tuple) else value
        out["effective_batch_size"] = self.effective_batch_size
        return out

    # -- construction ------------------------------------------------------
    @classmethod
    def from_registry(cls, config: Any = None, **overrides: Any) -> "VLMTrainingConfig":
        """Build from `configs/base.yaml`, then apply explicit overrides.

        The registry supplies the recipe; `overrides` (from the CLI) win. Reading
        the registry rather than re-typing the values is what keeps a single
        source of truth for the hashed settings.

        Args:
            config: a loaded `core.config.Config`. Loaded if omitted.
            **overrides: field names to override. `None` values are ignored so a
                CLI can pass every flag unconditionally.
        """
        if config is None:
            from core.config import load_config

            config = load_config()

        def g(key: str, default: Any) -> Any:
            value = config.get(f"training.{key}", None)
            return default if value is None else value

        known = {f.name for f in fields(cls)}
        seed_kwargs: dict[str, Any] = {
            "base_model": config.require("vlm.checkpoint"),
            "revision": config.get("vlm.revision"),
            "processor_longest_edge": int(
                config.get("vlm.processor_longest_edge", 512)
            ),
            "do_image_splitting": bool(
                config.get("vlm.do_image_splitting", True)
            ),
            "lora_rank": int(g("lora_rank", 16)),
            "lora_alpha": int(g("lora_alpha", 32)),
            "lora_dropout": float(g("lora_dropout", 0.05)),
            "learning_rate": float(g("vlm_learning_rate", 2e-4)),
            "batch_size": int(g("vlm_batch_size", 2)),
            "gradient_accumulation": int(g("vlm_gradient_accumulation", 8)),
            "epochs": int(g("vlm_epochs", 1)),
            "warmup_ratio": float(g("warmup_ratio", 0.05)),
            "weight_decay": float(g("weight_decay", 0.01)),
            "precision": str(g("precision", "fp16")),
            "gradient_checkpointing": bool(g("gradient_checkpointing", True)),
            "save_every_steps": int(g("save_every_steps", 500)),
            "config_hash": getattr(config, "hash", None),
        }

        for key, value in overrides.items():
            if value is None:
                continue
            if key not in known:
                raise ConfigError(
                    f"unknown configuration field {key!r}; known fields are "
                    f"{sorted(known)}"
                )
            seed_kwargs[key] = value

        # The deviation record is not an override surface -- it is a fact about
        # this run, written by the module that knows the plan disagrees.
        seed_kwargs.setdefault(
            "plan_deviations",
            {
                "precision": (
                    "plan section 43 specifies bf16; fp16 is used because T4 is "
                    "SM 7.5 and has no bf16 tensor cores (finding C-6). "
                    "Recorded so the deviation is auditable."
                )
            },
        )
        return cls(**seed_kwargs)
