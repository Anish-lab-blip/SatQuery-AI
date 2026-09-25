"""SatQuery AI — validate the Phase 18 deploy packaging manifest.

`configs/deploy.yaml` is a Hugging Face Spaces packaging manifest that is
**deliberately outside the configuration registry**: `core/config.py` reads only
`configs/base.yaml`, and `Config.hash` hashes the whole registry, so merging the
deploy file would move the recorded benchmark hash. This script proves the file
is inert and internally consistent:

  a. `configs/deploy.yaml` exists, parses, and declares `registry: false`.
  b. Its `deployment:` block equals `configs/base.yaml`'s `deployment:` block,
     key for key and value for value (either direction of a mismatch is named).
  c. The deploy-only marker is absent from the live registry
     (`core.config.get_config()`), and `deploy.yaml` contributes at least one
     top-level key the registry lacks. This is a NECESSARY condition for the
     file being outside the registry — NOT a proof against a merge: the
     registry is built FROM `configs/base.yaml`, so a key merged into
     `base.yaml` would appear on both sides and compare equal. The real
     guarantee against a merge is the `Config.hash` regression guard,
     `tests/unit/test_deploy_config.py::test_config_hash_regression_guard`,
     which pins the recorded hash (`FROZEN_CONFIG_HASH`).
  d. The C-8 deployment invariants hold.

Deterministic, dependency-free beyond `yaml` (already a core dependency), and
safe to import: `validate_documents` is pure and takes in-memory dicts, so tests
can exercise every failure mode without touching the real files.

Exit code 0 when every check passes; 1 (with a readable, key-naming message) when
any check fails. This script deploys nothing and reads nothing from the network.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
# Make the repo importable when run as `python scripts/validate_deploy_config.py`
# (pytest already puts the root on sys.path via `pythonpath = .`; the CLI does
# not, matching the pattern in the sibling `scripts/train_fusion.py`).
sys.path.insert(0, str(REPO_ROOT))

DEPLOY_PATH = REPO_ROOT / "configs" / "deploy.yaml"
BASE_PATH = REPO_ROOT / "configs" / "base.yaml"

#: The top-level marker that declares the file is not a registry member.
REGISTRY_MARKER = "registry"
REGISTRY_MARKER_VALUE = False

#: The deployment keys the C-8 / quota contract fixes, and their required values.
REQUIRED_BOOL: dict[str, bool] = {
    "torch_compile": False,
    "cpu_mode_required": True,
    "lazy_load": True,
}
REQUIRED_INT: dict[str, int] = {"cache_max_models": 1}
REQUIRED_STR: dict[str, str] = {
    "platform": "huggingface-spaces",
    "sdk": "gradio",
}
#: Every `gpu_duration_*` value must be a positive int (declared quota in seconds).
GPU_DURATION_PREFIX = "gpu_duration_"

__all__ = [
    "DEPLOY_PATH",
    "BASE_PATH",
    "REGISTRY_MARKER",
    "validate_documents",
    "validate",
    "main",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return data


def _check_marker(deploy_doc: dict[str, Any]) -> list[str]:
    """Check (a): the non-registry marker is present and false."""
    errors: list[str] = []
    if REGISTRY_MARKER not in deploy_doc:
        errors.append(
            f"deploy.yaml is missing the non-registry marker "
            f"{REGISTRY_MARKER!r}: {REGISTRY_MARKER_VALUE!r}"
        )
    elif deploy_doc[REGISTRY_MARKER] is not REGISTRY_MARKER_VALUE:
        errors.append(
            f"deploy.yaml marker {REGISTRY_MARKER!r} must be "
            f"{REGISTRY_MARKER_VALUE!r}, got {deploy_doc[REGISTRY_MARKER]!r}"
        )
    return errors


def _check_deployment_blocks_equal(
    deploy_doc: dict[str, Any], registry_doc: dict[str, Any]
) -> list[str]:
    """Check (b): the two `deployment:` blocks are equal, key for key."""
    errors: list[str] = []
    deploy_block = deploy_doc.get("deployment")
    registry_block = registry_doc.get("deployment")
    if not isinstance(deploy_block, dict):
        return ["deploy.yaml has no 'deployment:' mapping"]
    if not isinstance(registry_block, dict):
        return ["base.yaml has no 'deployment:' mapping"]

    missing_in_deploy = sorted(set(registry_block) - set(deploy_block))
    extra_in_deploy = sorted(set(deploy_block) - set(registry_block))
    for key in missing_in_deploy:
        errors.append(
            f"deployment key {key!r} is in base.yaml ({registry_block[key]!r}) "
            f"but missing from deploy.yaml"
        )
    for key in extra_in_deploy:
        errors.append(
            f"deployment key {key!r} is in deploy.yaml ({deploy_block[key]!r}) "
            f"but missing from base.yaml"
        )
    for key in sorted(set(deploy_block) & set(registry_block)):
        if deploy_block[key] != registry_block[key]:
            errors.append(
                f"deployment key {key!r} differs: "
                f"deploy.yaml={deploy_block[key]!r} base.yaml={registry_block[key]!r}"
            )
    return errors


def _check_not_merged(
    deploy_doc: dict[str, Any], registry_doc: dict[str, Any]
) -> list[str]:
    """Check (c): the deploy-only marker is absent from the registry, and the
    deploy file contributes at least one top-level key the registry lacks.

    This is a NECESSARY condition, not a merge proof: the registry is built from
    `configs/base.yaml`, so a key merged there would appear on BOTH sides and
    compare equal. The real guarantee is the `Config.hash` regression guard in
    `tests/unit/test_deploy_config.py::test_config_hash_regression_guard`.
    """
    errors: list[str] = []
    deploy_only = sorted(set(deploy_doc) - set(registry_doc))
    if not deploy_only:
        errors.append(
            "deploy.yaml contributes no top-level key absent from the registry; "
            "the check is vacuous (expected at least the "
            f"{REGISTRY_MARKER!r} marker)"
        )
    if REGISTRY_MARKER in registry_doc:
        errors.append(
            f"the registry contains the deploy-only marker {REGISTRY_MARKER!r}; "
            f"deploy.yaml appears to have been merged"
        )
    return errors


def _check_c8(deploy_doc: dict[str, Any]) -> list[str]:
    """Check (d): the C-8 / quota deployment invariants."""
    errors: list[str] = []
    block = deploy_doc.get("deployment")
    if not isinstance(block, dict):
        return ["deploy.yaml has no 'deployment:' mapping"]
    for key, expected in REQUIRED_BOOL.items():
        actual = block.get(key)
        if actual is not expected:
            errors.append(
                f"deployment.{key} must be {expected!r}, got {actual!r}"
            )
    for key, expected in REQUIRED_INT.items():
        actual = block.get(key)
        if actual != expected or isinstance(actual, bool):
            errors.append(
                f"deployment.{key} must be the int {expected!r}, got {actual!r}"
            )
    for key, expected in REQUIRED_STR.items():
        actual = block.get(key)
        if actual != expected:
            errors.append(
                f"deployment.{key} must be {expected!r}, got {actual!r}"
            )
    for key in sorted(k for k in block if k.startswith(GPU_DURATION_PREFIX)):
        value = block[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(
                f"deployment.{key} must be a positive int (seconds), got {value!r}"
            )
    return errors


def validate_documents(
    deploy_doc: dict[str, Any], registry_doc: dict[str, Any]
) -> list[str]:
    """Validate an in-memory deploy doc against a registry doc.

    Args:
        deploy_doc: parsed `configs/deploy.yaml`.
        registry_doc: the registry dict (`Config.as_dict()` of the real config).

    Returns:
        A list of failure messages; empty means every check passed. Pure — it
        mutates nothing and touches no file, so tests can drive every failure
        mode with synthetic dicts.
    """
    errors: list[str] = []
    errors += _check_marker(deploy_doc)
    errors += _check_deployment_blocks_equal(deploy_doc, registry_doc)
    errors += _check_not_merged(deploy_doc, registry_doc)
    errors += _check_c8(deploy_doc)
    return errors


def validate() -> list[str]:
    """Run every check against the real files and the live registry."""
    if not DEPLOY_PATH.exists():
        return [f"deploy manifest not found: {DEPLOY_PATH}"]
    try:
        deploy_doc = _load_yaml(DEPLOY_PATH)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        return [f"deploy manifest could not be parsed: {exc}"]

    from core.config import get_config

    registry_doc = get_config().as_dict()
    return validate_documents(deploy_doc, registry_doc)


def main() -> int:
    print("=" * 70)
    print("PHASE 18 — DEPLOY PACKAGING MANIFEST VALIDATION")
    print("=" * 70)
    print(f"deploy manifest : {DEPLOY_PATH}")
    print(f"registry file   : {BASE_PATH}")
    print()

    errors = validate()
    if errors:
        print(f"FAILED — {len(errors)} check(s):")
        for message in errors:
            print(f"  - {message}")
        return 1

    print("PASS — deploy.yaml is inert (not in the registry) and C-8-consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
