"""Tests for Phase 6 LoRA injection (`training.vlm.lora`).

The highest-value file in the Phase 6 suite, because the failure it guards is
SILENT. The SigLIP vision tower's attention projections are named `q_proj`,
`k_proj`, `v_proj` -- exactly like the language model's -- so the textbook
`target_modules=["q_proj","k_proj","v_proj"]` injects LoRA into the image encoder
that plan section 43 requires to be frozen, and PEFT reports success. The run
trains, the loss falls, and the image encoder has drifted.

A tiny `Idefics3ForConditionalGeneration` is built from small configs so every
assertion here is structural and fast. No hardcoded trainable-parameter count is
asserted: the property that matters is *where* the trainable parameters are.
"""

from __future__ import annotations

from typing import Any

import pytest

from training.vlm.config import VLMTrainingConfig
from training.vlm.lora import (
    LANGUAGE_MODEL_TARGET_REGEX,
    LoRAError,
    adapter_is_reloadable,
    attach_lora,
    build_lora_config,
    canonical_module_name,
    describe_frozen,
    load_adapter,
    save_adapter,
    verify_injection,
)


def tiny_idefics3() -> Any:
    """A minimal `Idefics3ForConditionalGeneration`, fast to build and enough to
    reproduce the vision/language projection-name collision."""
    from transformers import (
        Idefics3Config,
        Idefics3ForConditionalGeneration,
        Idefics3VisionConfig,
    )
    from transformers.models.llama.configuration_llama import LlamaConfig

    vision = Idefics3VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_channels=3,
        image_size=32,
        patch_size=16,
    )
    text = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=512,
    )
    return Idefics3ForConditionalGeneration(
        Idefics3Config(vision_config=vision, text_config=text)
    )


@pytest.fixture()
def cfg() -> VLMTrainingConfig:
    return VLMTrainingConfig.from_registry()


# ---------------------------------------------------------------------------
# The guard -- every trainable parameter is in the language model
# ---------------------------------------------------------------------------
def test_attach_lora_scopes_every_trainable_parameter_to_the_text_model(
    cfg: VLMTrainingConfig,
) -> None:
    model, report = attach_lora(tiny_idefics3(), cfg)

    assert report.all_trainable_in_text_model is True
    assert report.trainable_params > 0
    assert report.trainable_subtrees == {"model.text_model": report.trainable_params}

    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all(".text_model." in name for name in trainable)
    assert not any(".vision_model." in name for name in trainable)
    assert not any(".connector." in name for name in trainable)


def test_naive_target_modules_really_do_hit_the_vision_tower() -> None:
    """THE POINT OF THIS FILE.

    Demonstrates the hazard is real, so the scoped regex is demonstrably
    necessary rather than decorative. The textbook target list injects LoRA into
    `model.vision_model`, and PEFT reports success.

    If a future `transformers` release stops sharing the projection names, this
    test fails with a plain explanation rather than a silent pass.
    """
    from peft import LoraConfig, get_peft_model

    naive = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj"],
    )
    model = get_peft_model(tiny_idefics3(), naive)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    vision = [name for name in trainable if ".vision_model." in name]

    assert vision, (
        "the naive target list did not reach the vision tower on this "
        "transformers version; the hazard is not reproduced here, so the scoped "
        "regex is not demonstrably necessary"
    )
    # It also reaches the language model -- so it is the scoping, not the target
    # names, that keeps the vision tower frozen.
    assert len(vision) < len(trainable)


def test_build_lora_config_uses_the_scoped_regex(cfg: VLMTrainingConfig) -> None:
    lora_config = build_lora_config(cfg)
    assert lora_config.target_modules == LANGUAGE_MODEL_TARGET_REGEX
    assert lora_config.bias == "none"
    assert lora_config.r == cfg.lora_rank
    assert lora_config.lora_alpha == cfg.lora_alpha


# ---------------------------------------------------------------------------
# verify_injection -- both silent failures are refused
# ---------------------------------------------------------------------------
def test_verify_injection_refuses_zero_trainable_parameters() -> None:
    """A LoRA that matched nothing trains happily and learns nothing."""
    model = tiny_idefics3()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(LoRAError) as exc:
        verify_injection(model)
    assert "ZERO trainable" in str(exc.value)


def test_verify_injection_refuses_a_trainable_parameter_outside_the_text_model(
    cfg: VLMTrainingConfig,
) -> None:
    model, _ = attach_lora(tiny_idefics3(), cfg)
    offender = None
    for name, parameter in model.named_parameters():
        if ".vision_model." in name:
            parameter.requires_grad_(True)
            offender = name
            break
    assert offender is not None
    with pytest.raises(LoRAError) as exc:
        verify_injection(model)
    assert "outside the language model" in str(exc.value)


def test_describe_frozen_proves_the_vision_tower_is_frozen(
    cfg: VLMTrainingConfig,
) -> None:
    """Positive evidence the image encoder is intact and untouched: a non-zero
    frozen count under `model.vision_model`."""
    model, _ = attach_lora(tiny_idefics3(), cfg)
    frozen = describe_frozen(model)
    assert frozen["model.vision_model"] > 0
    assert frozen["model.text_model"] > 0


# ---------------------------------------------------------------------------
# canonical_module_name -- the PEFT-wrapper correction
# ---------------------------------------------------------------------------
def test_canonical_module_name_strips_the_peft_wrapper() -> None:
    """The contract's literal `startswith("model.text_model.")` guard was wrong
    once PEFT wrapped the model; the canonicaliser reduces the wrapped name to
    the unwrapped form the contract speaks."""
    assert (
        canonical_module_name(
            "base_model.model.model.text_model.layers.0.self_attn.q_proj"
        )
        == "model.text_model.layers.0.self_attn.q_proj"
    )
    # An already-canonical name is returned unchanged.
    assert (
        canonical_module_name("model.text_model.layers.0.self_attn.q_proj")
        == "model.text_model.layers.0.self_attn.q_proj"
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def test_adapter_round_trips_through_disk(
    cfg: VLMTrainingConfig, tmp_path
) -> None:
    from peft import PeftModel

    model, _ = attach_lora(tiny_idefics3(), cfg)
    out = save_adapter(model, tmp_path / "adapter")
    assert (out / "adapter_config.json").exists()

    reloaded = load_adapter(tiny_idefics3(), out)
    assert isinstance(reloaded, PeftModel)


def test_load_adapter_refuses_a_missing_directory() -> None:
    with pytest.raises(LoRAError):
        load_adapter(tiny_idefics3(), "does/not/exist")


def test_adapter_is_reloadable_returns_false_for_a_bogus_path() -> None:
    """Returns a bool, never raises -- artifact verification (V4) depends on it."""
    assert adapter_is_reloadable(tiny_idefics3(), "does/not/exist") is False
