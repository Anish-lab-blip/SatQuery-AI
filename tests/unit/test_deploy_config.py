"""Tests for the Phase 18 deploy packaging manifest and its validator.

These tests prove the manifest is INERT — it is never merged into the config
registry and cannot move `Config.hash` — and that the validator rejects each
failure mode. Negative cases use in-memory dicts via `validate_documents`, so the
real config files are never mutated.
"""

from __future__ import annotations

import subprocess
import sys

import yaml
import pytest

from core.config import get_config, load_config
from scripts.validate_deploy_config import (
    BASE_PATH,
    DEPLOY_PATH,
    REGISTRY_MARKER,
    validate,
    validate_documents,
)

#: The project's existing frozen-hash constant (the config-hash authority). We
#: import it rather than re-declare it, so the two cannot drift.
from tests.test_config import FROZEN_CONFIG_HASH

GPU_DURATION_KEYS = (
    "gpu_duration_vqa",
    "gpu_duration_grounding",
    "gpu_duration_change",
    "gpu_duration_optical_sar",
)


# ---------------------------------------------------------------------------
# Real files
# ---------------------------------------------------------------------------
def test_deploy_manifest_exists_and_declares_the_non_registry_marker() -> None:
    assert DEPLOY_PATH.exists(), f"missing deploy manifest: {DEPLOY_PATH}"
    doc = yaml.safe_load(DEPLOY_PATH.read_text(encoding="utf-8"))
    assert doc[REGISTRY_MARKER] is False


def test_validator_reports_no_errors_on_the_real_files() -> None:
    assert validate() == []


def test_validator_cli_exits_zero_as_a_standalone_script() -> None:
    """The validator must run exactly as documented: as a plain script.

    This drives the real entry point in a child interpreter (the same one running
    the suite) to prove the module's own `sys.path` bootstrap makes `core`
    importable WITHOUT pytest's `pythonpath = .` shortcut. Exit 0 on success.
    """
    repo_root = DEPLOY_PATH.parent.parent  # configs/deploy.yaml -> repo root
    script = repo_root / "scripts" / "validate_deploy_config.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout
    assert "ModuleNotFoundError" not in (result.stdout + result.stderr)


def test_parsing_deploy_yaml_does_not_change_the_registry_hash() -> None:
    """Parsing `deploy.yaml` must not touch the registry — proven non-vacuously.

    `load_config()` reads exactly one path (`configs/base.yaml`) and never globs
    `configs/*.yaml`, so `deploy.yaml` is structurally outside the loader. The
    earlier version of this test only compared `load_config().hash` to itself,
    which holds even after a merge and therefore protected nothing. The real
    checks are (1) the live registry's top-level key set EQUALS `base.yaml`'s —
    a key merged in from anywhere would break it — and (2) the
    `FROZEN_CONFIG_HASH` guard (`test_config_hash_regression_guard`). A key-set
    comparison alone cannot detect a merge into `base.yaml`; the hash guard is
    what closes that gap.
    """
    # Re-parse the deploy manifest exactly as a caller might.
    with DEPLOY_PATH.open("r", encoding="utf-8") as fh:
        yaml.safe_load(fh)

    base_doc = yaml.safe_load(BASE_PATH.read_text(encoding="utf-8"))
    registry_doc = get_config().as_dict()

    assert set(registry_doc) == set(base_doc)
    assert load_config().hash == FROZEN_CONFIG_HASH


def test_registry_key_set_equality_bites_on_a_merged_key() -> None:
    """Negative control: the key-set equality above must reject a merged key.

    In-memory only — the real config files are never touched. A registry dict
    carrying one extra top-level key fails the exact comparison the test above
    relies on, so that assertion is not vacuous.
    """
    base_doc = yaml.safe_load(BASE_PATH.read_text(encoding="utf-8"))
    registry_with_merge = dict(base_doc)
    registry_with_merge["leaked_merge_key"] = {"merged": True}

    assert set(registry_with_merge) != set(base_doc)


def test_registry_has_no_key_unique_to_deploy_yaml() -> None:
    deploy_doc = yaml.safe_load(DEPLOY_PATH.read_text(encoding="utf-8"))
    registry_doc = get_config().as_dict()
    deploy_only = set(deploy_doc) - set(registry_doc)
    # The only deploy-only key is the marker, and it is NOT in the registry.
    assert deploy_only == {REGISTRY_MARKER}
    for key in deploy_only:
        assert key not in registry_doc


def test_deploy_deployment_block_equals_base_deployment_block() -> None:
    deploy_doc = yaml.safe_load(DEPLOY_PATH.read_text(encoding="utf-8"))
    base_doc = yaml.safe_load(BASE_PATH.read_text(encoding="utf-8"))
    assert deploy_doc["deployment"] == base_doc["deployment"]


# ---------------------------------------------------------------------------
# C-8 invariants — asserted individually so a failure names the exact rule
# ---------------------------------------------------------------------------
def _deploy_block() -> dict:
    return yaml.safe_load(DEPLOY_PATH.read_text(encoding="utf-8"))["deployment"]


def test_c8_torch_compile_is_false() -> None:
    assert _deploy_block()["torch_compile"] is False


def test_c8_cpu_mode_is_required() -> None:
    assert _deploy_block()["cpu_mode_required"] is True


def test_c8_lazy_load_is_enabled() -> None:
    assert _deploy_block()["lazy_load"] is True


def test_c8_single_model_cache() -> None:
    assert _deploy_block()["cache_max_models"] == 1


@pytest.mark.parametrize("key", GPU_DURATION_KEYS)
def test_c8_gpu_durations_are_positive_ints(key: str) -> None:
    value = _deploy_block()[key]
    assert isinstance(value, int) and not isinstance(value, bool)
    assert value > 0


def test_deploy_platform_is_huggingface_spaces() -> None:
    assert _deploy_block()["platform"] == "huggingface-spaces"


def test_deploy_sdk_is_gradio() -> None:
    assert _deploy_block()["sdk"] == "gradio"


def test_config_hash_regression_guard() -> None:
    """`base.yaml` is HASH-LOAD-BEARING for the project's benchmark number.

    `Config.hash` hashes the WHOLE registry, and the recorded benchmark hash
    `78f1e3700da15aa1` is asserted by `tests/test_config.py` and by the change-head
    eval drift check (`scripts/eval_change.py`). Any intentional change to
    `base.yaml` must update this expectation (and the recorded hash) in the SAME
    commit; otherwise a well-meaning config edit silently invalidates the
    benchmark. The deploy manifest must never cause such a move.
    """
    assert load_config().hash == FROZEN_CONFIG_HASH


# ---------------------------------------------------------------------------
# Validator failure modes — in-memory dicts only
# ---------------------------------------------------------------------------
def _good_deploy_doc() -> dict:
    return {
        REGISTRY_MARKER: False,
        "deployment": {
            "platform": "huggingface-spaces",
            "sdk": "gradio",
            "zerogpu": True,
            "torch_compile": False,
            "gpu_duration_vqa": 20,
            "gpu_duration_grounding": 45,
            "gpu_duration_change": 30,
            "gpu_duration_optical_sar": 45,
            "cpu_mode_required": True,
            "lazy_load": True,
            "cache_max_models": 1,
        },
    }


def _good_registry_doc() -> dict:
    return {
        "project": {"name": "satquery-ai"},
        "deployment": dict(_good_deploy_doc()["deployment"]),
    }


def test_good_pair_validates_clean() -> None:
    assert validate_documents(_good_deploy_doc(), _good_registry_doc()) == []


def test_missing_registry_marker_is_rejected() -> None:
    deploy = _good_deploy_doc()
    del deploy[REGISTRY_MARKER]
    errors = validate_documents(deploy, _good_registry_doc())
    assert any(REGISTRY_MARKER in message for message in errors)


def test_true_registry_marker_is_rejected() -> None:
    deploy = _good_deploy_doc()
    deploy[REGISTRY_MARKER] = True
    errors = validate_documents(deploy, _good_registry_doc())
    assert any(REGISTRY_MARKER in message for message in errors)


def test_key_missing_from_deploy_is_rejected() -> None:
    deploy = _good_deploy_doc()
    del deploy["deployment"]["lazy_load"]
    errors = validate_documents(deploy, _good_registry_doc())
    assert any("lazy_load" in message for message in errors)


def test_extra_key_in_deploy_is_rejected() -> None:
    deploy = _good_deploy_doc()
    deploy["deployment"]["brand_new_key"] = 1
    errors = validate_documents(deploy, _good_registry_doc())
    assert any("brand_new_key" in message for message in errors)


def test_value_mismatch_is_rejected_and_names_both_values() -> None:
    deploy = _good_deploy_doc()
    deploy["deployment"]["sdk"] = "streamlit"
    errors = validate_documents(deploy, _good_registry_doc())
    message = next((m for m in errors if "sdk" in m), "")
    assert message, errors
    assert "'streamlit'" in message and "'gradio'" in message


def test_torch_compile_true_is_rejected() -> None:
    deploy = _good_deploy_doc()
    deploy["deployment"]["torch_compile"] = True
    errors = validate_documents(deploy, _good_registry_doc())
    assert any("torch_compile" in message for message in errors)


def test_cpu_mode_false_is_rejected() -> None:
    deploy = _good_deploy_doc()
    deploy["deployment"]["cpu_mode_required"] = False
    errors = validate_documents(deploy, _good_registry_doc())
    assert any("cpu_mode_required" in message for message in errors)


def test_bad_cache_size_is_rejected() -> None:
    deploy = _good_deploy_doc()
    deploy["deployment"]["cache_max_models"] = 2
    errors = validate_documents(deploy, _good_registry_doc())
    assert any("cache_max_models" in message for message in errors)


def test_non_positive_gpu_duration_is_rejected() -> None:
    deploy = _good_deploy_doc()
    deploy["deployment"]["gpu_duration_vqa"] = 0
    errors = validate_documents(deploy, _good_registry_doc())
    assert any("gpu_duration_vqa" in message for message in errors)


def test_merged_registry_is_rejected() -> None:
    # If the marker key ever appeared in the registry, the file was merged.
    registry = _good_registry_doc()
    registry[REGISTRY_MARKER] = False
    errors = validate_documents(_good_deploy_doc(), registry)
    assert any(REGISTRY_MARKER in message for message in errors)
