"""Tests for the Phase 6 adapter artifact (`training.vlm.artifact`).

A verifier that cannot fail is not a verifier, so the load-bearing test here
writes an artifact, mutates one file, and asserts `verify_artifact` refuses it.
The other load-bearing test asserts the smoke-run refusal: a 1-step CPU probe
must not produce a directory that reads as a Phase 6 result.

Dummy model/processor objects stand in for the PEFT model, so no model is
loaded -- the artifact contract is about the bytes written and the manifest, not
about the weights.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from training.vlm.artifact import (
    CHECKSUM_FILENAME,
    MANIFEST_FILENAME,
    ArtifactError,
    export_adapter,
    hash_tree,
    sha256_file,
    verify_artifact,
)
from training.vlm.config import VLMTrainingConfig
from training.vlm.evaluate import ACCEPTANCE_RULE_VERSION
from training.vlm.run_manifest import build_run_manifest, manifest_hash

FROZEN_CONFIG_HASH = "78f1e3700da15aa1"


class _DummyModel:
    """Stand-in for a PEFT model: writes the two adapter files."""

    def save_pretrained(self, out_dir: str | Path) -> None:
        directory = Path(out_dir)
        (directory / "adapter_model.safetensors").write_bytes(b"adapter-weights")
        (directory / "adapter_config.json").write_text(
            '{"r": 16, "lora_alpha": 32}', encoding="utf-8"
        )


class _DummyProcessor:
    """Stand-in for the processor: writes its preprocessing config."""

    def save_pretrained(self, out_dir: str | Path) -> None:
        (Path(out_dir) / "processor_config.json").write_text(
            '{"size": {"longest_edge": 512}}', encoding="utf-8"
        )


def _export(tmp_path: Path, **kwargs: Any) -> Path:
    return export_adapter(
        _DummyModel(),
        _DummyProcessor(),
        tmp_path / "artifact",
        run_manifest={"phase": 6},
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"payload")
    assert sha256_file(path) == hashlib.sha256(b"payload").hexdigest()


def test_hash_tree_is_deterministic_and_excludes_the_checksum_file(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "tree"
    directory.mkdir()
    (directory / "a.txt").write_text("alpha", encoding="utf-8")
    (directory / "b.txt").write_text("beta", encoding="utf-8")
    (directory / CHECKSUM_FILENAME).write_text("{}", encoding="utf-8")

    first = hash_tree(directory)
    second = hash_tree(directory)
    assert first == second
    assert set(first) == {"a.txt", "b.txt"}
    assert CHECKSUM_FILENAME not in first


def test_hash_tree_refuses_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError):
        hash_tree(tmp_path / "nope")


# ---------------------------------------------------------------------------
# Export + verify
# ---------------------------------------------------------------------------
def test_export_then_verify_succeeds(tmp_path: Path) -> None:
    out = _export(tmp_path)
    report = verify_artifact(out)
    assert report["ok"] is True
    assert report["n_files"] >= 3
    assert (out / CHECKSUM_FILENAME).exists()
    assert (out / MANIFEST_FILENAME).exists()


def test_verify_artifact_detects_tampering(tmp_path: Path) -> None:
    """A verifier that cannot fail is not a verifier."""
    out = _export(tmp_path)
    (out / "adapter_config.json").write_text('{"r": 999}', encoding="utf-8")
    with pytest.raises(ArtifactError) as exc:
        verify_artifact(out)
    assert "digest mismatch" in str(exc.value)


def test_verify_artifact_detects_a_missing_file(tmp_path: Path) -> None:
    out = _export(tmp_path)
    (out / "adapter_model.safetensors").unlink()
    with pytest.raises(ArtifactError):
        verify_artifact(out)


def test_verify_artifact_requires_the_checksum_manifest(tmp_path: Path) -> None:
    out = _export(tmp_path)
    (out / CHECKSUM_FILENAME).unlink()
    with pytest.raises(ArtifactError):
        verify_artifact(out)


# ---------------------------------------------------------------------------
# The smoke-run refusal
# ---------------------------------------------------------------------------
def test_export_refuses_a_smoke_run_without_allow_smoke(tmp_path: Path) -> None:
    cfg = VLMTrainingConfig.from_registry(max_steps=1)
    assert cfg.is_cpu_sized is True
    with pytest.raises(ArtifactError):
        _export(tmp_path, is_smoke=cfg.is_cpu_sized)


def test_export_allows_a_smoke_run_when_explicitly_opted_in(tmp_path: Path) -> None:
    cfg = VLMTrainingConfig.from_registry(max_steps=1)
    out = _export(tmp_path, is_smoke=cfg.is_cpu_sized, allow_smoke=True)
    manifest = json.loads((out / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["is_smoke"] is True


# ---------------------------------------------------------------------------
# The run manifest -- contract item X
# ---------------------------------------------------------------------------
def _manifest() -> dict[str, Any]:
    return build_run_manifest(
        config=VLMTrainingConfig.from_registry(),
        corpus_summary={"n_samples": 10},
        lora_report={"trainable_params": 17408, "trainable_fraction": 0.02},
        frozen_report={"model.vision_model": 1000, "model.text_model": 500},
        prompt_contract={"version": "v001"},
        state={"steps": 1},
        metrics={"n": 10, "exact_match": 0.9},
        baseline={"n": 10, "exact_match": 0.5},
        metrics_test={"n": 10, "exact_match": 0.88},
        baseline_test={"n": 10, "exact_match": 0.5},
    )


def test_run_manifest_carries_the_contract_item_x_fields() -> None:
    manifest = _manifest()
    for key in (
        "base_model",
        "base_model_revision",
        "config_hash",
        "prompt_version",
        "seed",
        "trainable_params",
        "trainable_fraction",
        "metrics",
        "baseline",
        "metrics_test",
        "baseline_test",
        "acceptance_rule_version",
        "manifest_hash",
        "environment",
        "plan_deviations",
        "effective_batch_size",
    ):
        assert key in manifest, key

    assert manifest["config_hash"] == FROZEN_CONFIG_HASH
    # The val and test blocks are recorded separately, so the decision split (val)
    # and the confirmation split (test) cannot be confused.
    assert manifest["metrics"]["exact_match"] == 0.9
    assert manifest["metrics_test"]["exact_match"] == 0.88
    assert manifest["baseline_test"]["exact_match"] == 0.5
    # Bumped v001 -> v002 on 2026-09-24 (the V2 resolution fix). The manifest
    # records the rule version so a threshold change is auditable, not silent;
    # v001 stays replayable via `rule_version="v001"` (test_vlm_evaluate.py).
    assert manifest["acceptance_rule_version"] == ACCEPTANCE_RULE_VERSION == "v002"
    assert manifest["prompt_version"] == "v001"
    assert manifest["environment"]["python"]
    # The self-digest recomputes from the content.
    assert manifest["manifest_hash"] == manifest_hash(manifest)


def test_exported_manifest_records_the_adapter_sha256(tmp_path: Path) -> None:
    """`adapter_sha256` is filled in after the weights exist, so the recorded
    digest is of the bytes actually written."""
    manifest = _manifest()
    out = export_adapter(
        _DummyModel(), _DummyProcessor(), tmp_path / "art", run_manifest=manifest
    )
    written = json.loads((out / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert written["adapter_sha256"]
    assert written["manifest_hash"] == manifest["manifest_hash"]
    assert written["is_smoke"] is False
