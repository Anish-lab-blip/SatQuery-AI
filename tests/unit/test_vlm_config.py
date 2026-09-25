"""Tests for the Phase 6 training recipe (`training.vlm.config`).

The recipe is not re-typed here. `from_registry()` reads `configs/base.yaml`,
whose hash `78f1e3700da15aa1` is a frozen invariant cited by run manifests, so
the values are asserted against the registry-loaded config rather than against
literals the test itself supplies. A second copy of `lora_rank: 16` in Python
would be a value that can drift from the hashed one -- which is the whole reason
the module exists.

Every `validate()` branch gets its own test: a validator with an untested branch
is a validator you cannot trust.
"""

from __future__ import annotations

import json
from dataclasses import fields

import pytest

from training.vlm.config import (
    LORA_TARGET_MODULES,
    PRECISION_CHOICES,
    ConfigError,
    VLMTrainingConfig,
)

#: The frozen registry hash. Cited by run manifests (contract item X).
FROZEN_CONFIG_HASH = "78f1e3700da15aa1"


@pytest.fixture()
def cfg() -> VLMTrainingConfig:
    return VLMTrainingConfig.from_registry()


# ---------------------------------------------------------------------------
# from_registry -- the values come from the hashed registry
# ---------------------------------------------------------------------------
def test_from_registry_reads_the_frozen_recipe(cfg: VLMTrainingConfig) -> None:
    assert cfg.lora_rank == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.05
    assert cfg.learning_rate == 2e-4
    assert cfg.batch_size == 2
    assert cfg.gradient_accumulation == 8
    assert cfg.epochs == 1
    assert cfg.warmup_ratio == 0.05
    assert cfg.weight_decay == 0.01
    assert cfg.precision == "fp16"


def test_config_hash_is_the_frozen_invariant(cfg: VLMTrainingConfig) -> None:
    assert cfg.config_hash == FROZEN_CONFIG_HASH


def test_effective_batch_size_is_the_plans_sixteen(cfg: VLMTrainingConfig) -> None:
    """The plan's section 43 figure. This equality is the whole reason the
    micro-batch/accumulation pair is acceptable."""
    assert cfg.effective_batch_size == 16
    assert cfg.effective_batch_size == cfg.batch_size * cfg.gradient_accumulation


def test_plan_deviations_records_the_bf16_to_fp16_divergence(
    cfg: VLMTrainingConfig,
) -> None:
    deviation = cfg.plan_deviations["precision"]
    assert "bf16" in deviation
    assert "C-6" in deviation


def test_target_modules_are_the_contract_set(cfg: VLMTrainingConfig) -> None:
    assert cfg.lora_target_modules == LORA_TARGET_MODULES
    assert cfg.lora_target_modules == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def test_default_instruction_family_is_presence_only(cfg: VLMTrainingConfig) -> None:
    """The local corpus is single-label, so the multi-label family is disabled."""
    assert cfg.instruction_families == ("presence",)


# ---------------------------------------------------------------------------
# validate() -- one test per rejection branch
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("precision", ["fp64", "float16", "BF16", ""])
def test_validate_rejects_bad_precision(precision: str) -> None:
    with pytest.raises(ConfigError, match="precision"):
        VLMTrainingConfig(precision=precision)


def test_precision_choices_match_the_registry_validation() -> None:
    assert PRECISION_CHOICES == ("fp16", "bf16", "fp32")


@pytest.mark.parametrize("rank", [0, -1])
def test_validate_rejects_non_positive_lora_rank(rank: int) -> None:
    with pytest.raises(ConfigError, match="lora_rank"):
        VLMTrainingConfig(lora_rank=rank)


@pytest.mark.parametrize("dropout", [-0.01, 1.0, 1.5])
def test_validate_rejects_lora_dropout_outside_unit_interval(dropout: float) -> None:
    with pytest.raises(ConfigError, match="lora_dropout"):
        VLMTrainingConfig(lora_dropout=dropout)


@pytest.mark.parametrize("lr", [0.0, -1e-4])
def test_validate_rejects_non_positive_learning_rate(lr: float) -> None:
    with pytest.raises(ConfigError, match="learning_rate"):
        VLMTrainingConfig(learning_rate=lr)


def test_validate_rejects_epochs_below_one_without_max_steps() -> None:
    with pytest.raises(ConfigError, match="epochs"):
        VLMTrainingConfig(epochs=0, max_steps=None)


def test_validate_allows_epochs_below_one_when_max_steps_is_set() -> None:
    """The rejection only applies when there is no step cap to fall back on."""
    capped = VLMTrainingConfig(epochs=0, max_steps=10)
    assert capped.max_steps == 10


@pytest.mark.parametrize(
    "train_ratio,val_ratio", [(0.9, 0.1), (0.95, 0.05), (0.6, 0.5)]
)
def test_validate_rejects_train_plus_val_ratio_at_or_above_one(
    train_ratio: float, val_ratio: float
) -> None:
    with pytest.raises(ConfigError, match="train_ratio"):
        VLMTrainingConfig(train_ratio=train_ratio, val_ratio=val_ratio)


@pytest.mark.parametrize(
    "percentiles",
    [(98.0, 2.0), (-1.0, 98.0), (2.0, 101.0), (5.0, 5.0)],
)
def test_validate_rejects_bad_rgb_percentiles(
    percentiles: tuple[float, float],
) -> None:
    with pytest.raises(ConfigError, match="rgb_percentiles"):
        VLMTrainingConfig(rgb_percentiles=percentiles)


def test_validate_rejects_an_unknown_instruction_family() -> None:
    with pytest.raises(ConfigError, match="instruction famil"):
        VLMTrainingConfig(instruction_families=("presence", "describe"))


def test_validate_rejects_empty_lora_target_modules() -> None:
    with pytest.raises(ConfigError, match="lora_target_modules"):
        VLMTrainingConfig(lora_target_modules=())


def test_from_registry_rejects_an_unknown_override() -> None:
    with pytest.raises(ConfigError, match="unknown configuration field"):
        VLMTrainingConfig.from_registry(not_a_field=1)


# ---------------------------------------------------------------------------
# to_dict -- the run-manifest serialisation
# ---------------------------------------------------------------------------
def test_to_dict_converts_tuples_to_lists_and_round_trips(
    cfg: VLMTrainingConfig,
) -> None:
    d = cfg.to_dict()

    assert isinstance(d["lora_target_modules"], list)
    assert d["rgb_percentiles"] == [2.0, 98.0]
    assert d["instruction_families"] == ["presence"]
    assert d["effective_batch_size"] == 16
    # JSON-serialisable, which is what the run manifest requires.
    json.dumps(d)

    # Rebuilding from the field values and re-serialising reproduces the dict
    # exactly (the tuple->list conversion is stable).
    field_names = {f.name for f in fields(VLMTrainingConfig)}
    rebuilt = VLMTrainingConfig(**{k: v for k, v in d.items() if k in field_names})
    assert rebuilt.to_dict() == d
