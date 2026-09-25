"""SatQuery AI — Phase 6 BigEarthNet -> SmolVLM LoRA adaptation.

This package implements the Phase 6 workstream described in
`docs/PHASE6_AUDIT_AND_CONTRACT.md`. The public surface is re-exported here, but
**lazily**: importing `training.vlm.config` must not drag torch, transformers, or
peft into the process, because config is a pure dataclass module and the
acceptance rule's constants must be readable without a model installed.

PEP 562 module `__getattr__` provides that: a name is imported from its defining
submodule only when it is first accessed.

    from training.vlm import VLMTrainingConfig        # cheap: config only
    from training.vlm import attach_lora              # pulls peft on demand
"""

from __future__ import annotations

from typing import Any

__all__ = [
    # config
    "ConfigError",
    "VLMTrainingConfig",
    "LORA_TARGET_MODULES",
    "PRECISION_CHOICES",
    # dataset
    "CorpusBuild",
    "InstructionSample",
    "PatchSample",
    "RGB_BANDS",
    "VLMDataError",
    "build_corpus",
    "describe_render",
    "discover_labelled_patches",
    "render_rgb",
    # formatting
    "FormattedExample",
    "IGNORE_INDEX",
    "build_training_messages",
    "describe_prompt_contract",
    "format_example",
    # collate
    "CollationError",
    "CollationItem",
    "Collator",
    "as_patch_sample",
    "make_item",
    "render_sample",
    # lora
    "LANGUAGE_MODEL_TARGET_REGEX",
    "LoRAError",
    "LoRAInjectionReport",
    "adapter_is_reloadable",
    "attach_lora",
    "build_lora_config",
    "describe_frozen",
    "language_model_regex",
    "load_adapter",
    "save_adapter",
    "verify_injection",
    # trainer
    "TrainingError",
    "TrainingState",
    "build_optimizer",
    "build_scheduler",
    "set_seed",
    "train",
    # evaluate
    "ACCEPTANCE_RULE_VERSION",
    "LEGACY_RULE_VERSIONS",
    "AcceptanceDecision",
    "MetricSet",
    "decide_acceptance",
    "describe_acceptance_rule",
    "evaluate_split",
    "normalise_answer",
    "predict",
    # artifact
    "ArtifactError",
    "export_adapter",
    "hash_tree",
    "refresh_artifact_checksums",
    "sha256_file",
    "verify_artifact",
    # run manifest
    "build_run_manifest",
    "environment_versions",
    "manifest_hash",
    "write_run_manifest",
]

#: name -> submodule that defines it. Imported on first access (PEP 562).
_LAZY: dict[str, str] = {
    "ConfigError": "config",
    "VLMTrainingConfig": "config",
    "LORA_TARGET_MODULES": "config",
    "PRECISION_CHOICES": "config",
    "CorpusBuild": "dataset",
    "InstructionSample": "dataset",
    "PatchSample": "dataset",
    "RGB_BANDS": "dataset",
    "VLMDataError": "dataset",
    "build_corpus": "dataset",
    "describe_render": "dataset",
    "discover_labelled_patches": "dataset",
    "render_rgb": "dataset",
    "FormattedExample": "formatting",
    "IGNORE_INDEX": "formatting",
    "build_training_messages": "formatting",
    "describe_prompt_contract": "formatting",
    "format_example": "formatting",
    "CollationError": "collate",
    "CollationItem": "collate",
    "Collator": "collate",
    "as_patch_sample": "collate",
    "make_item": "collate",
    "render_sample": "collate",
    "LANGUAGE_MODEL_TARGET_REGEX": "lora",
    "LoRAError": "lora",
    "LoRAInjectionReport": "lora",
    "adapter_is_reloadable": "lora",
    "attach_lora": "lora",
    "build_lora_config": "lora",
    "describe_frozen": "lora",
    "language_model_regex": "lora",
    "load_adapter": "lora",
    "save_adapter": "lora",
    "verify_injection": "lora",
    "TrainingError": "trainer",
    "TrainingState": "trainer",
    "build_optimizer": "trainer",
    "build_scheduler": "trainer",
    "set_seed": "trainer",
    "train": "trainer",
    "ACCEPTANCE_RULE_VERSION": "evaluate",
    "LEGACY_RULE_VERSIONS": "evaluate",
    "AcceptanceDecision": "evaluate",
    "MetricSet": "evaluate",
    "decide_acceptance": "evaluate",
    "describe_acceptance_rule": "evaluate",
    "evaluate_split": "evaluate",
    "normalise_answer": "evaluate",
    "predict": "evaluate",
    "ArtifactError": "artifact",
    "export_adapter": "artifact",
    "hash_tree": "artifact",
    "refresh_artifact_checksums": "artifact",
    "sha256_file": "artifact",
    "verify_artifact": "artifact",
    "build_run_manifest": "run_manifest",
    "environment_versions": "run_manifest",
    "manifest_hash": "run_manifest",
    "write_run_manifest": "run_manifest",
}


def __getattr__(name: str) -> Any:
    """Import a public name from its defining submodule on first access."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}; known names are "
            f"{sorted(__all__)}"
        )
    from importlib import import_module

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
