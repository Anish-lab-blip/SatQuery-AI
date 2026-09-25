"""SatQuery AI — Phase 6 run manifest (contract item X).

A run manifest is the difference between "we trained an adapter" and "we can
show what the adapter is". It records, in one JSON object, every input that
determined the artifact:

    base model id + revision      which weights the adapter sits on
    adapter SHA256                which adapter bytes were produced
    code revision + snapshot      which CODE produced them (contract item X)
    config hash                   the frozen registry hash the recipe came from
    manifest hash                 a self-digest, so the record cannot drift
    prompt version                the serving prompt set it was trained against
    seed                          the single seed threaded through everything
    trainable-param count         measured, not asserted
    evaluation budget             eval limit, realised n_evaluated, truncation
    environment versions          so a re-run can be reproduced or refuted
    wall time                     how long it took
    acceptance rule version       the predeclared decision rule (item V)

Values come from **real imports and real measurements**. Nothing is hardcoded:
a version string typed into this file would be a value that can drift from the
environment that actually ran.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "environment_versions",
    "code_provenance",
    "build_run_manifest",
    "write_run_manifest",
    "manifest_hash",
]


def _module_version(name: str) -> str:
    """Import `name` and return its `__version__`, or 'unavailable'."""
    try:
        module = __import__(name)
    except Exception:  # noqa: BLE001 - a missing optional dep is not fatal
        return "unavailable"
    return str(getattr(module, "__version__", "unavailable"))


def environment_versions() -> dict[str, Any]:
    """Python, platform, library versions, and CUDA availability + device name.

    CUDA facts are read from `torch.cuda` when torch is importable. When it is
    not, `cuda_available` is `False` and the device name is `None` -- the honest
    answer, never a guess.
    """
    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _module_version("torch"),
        "transformers": _module_version("transformers"),
        "peft": _module_version("peft"),
        "accelerate": _module_version("accelerate"),
        "numpy": _module_version("numpy"),
        "cuda_available": False,
        "cuda_device_name": None,
        "cuda_device_count": 0,
    }
    try:
        import torch

        out["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            out["cuda_device_count"] = int(torch.cuda.device_count())
            out["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - torch absent is a valid environment
        pass
    return out


def code_provenance(root: str | Path | None = None) -> dict[str, Any]:
    """Which code produced this run (contract item X).

    Delegates to `core.code_revision.revision_block`, which already records the
    git revision, the content snapshot digest and the uploaded-dataset identity,
    and already refuses to invent an identifier it cannot observe. Re-deriving any
    of that here would create a second, drifting definition of "the code that
    ran" -- exactly the failure this field exists to prevent.

    Never raises. A manifest is written at the END of a long run, so a revision
    lookup that fails must degrade to an explicit `available: False` rather than
    destroy the record of a run that already happened.

    Args:
        root: the code root. Defaults to `core.code_revision.REPO_ROOT`, which on
            Kaggle resolves to the attached code dataset.

    Returns:
        The revision block, or `{"available": False, ...}` when unavailable.
    """
    try:
        from core.code_revision import REPO_ROOT, revision_block
    except Exception as exc:  # noqa: BLE001 - a manifest must still be writable
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "note": (
                "the revision helpers could not be imported; the code identity "
                "for this run is unknown and is NOT recorded as a placeholder"
            ),
        }
    target = Path(root) if root is not None else REPO_ROOT
    try:
        return revision_block(target)
    except Exception as exc:  # noqa: BLE001 - see above
        return {
            "available": False,
            "code_root": str(target),
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_run_manifest(
    *,
    config: Any,
    corpus_summary: dict[str, Any],
    lora_report: Any,
    frozen_report: dict[str, int],
    prompt_contract: dict[str, Any],
    state: Any,
    metrics: dict[str, Any] | None = None,
    baseline: dict[str, Any] | None = None,
    metrics_test: dict[str, Any] | None = None,
    baseline_test: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
    evaluation_budget: dict[str, Any] | None = None,
    code_root: str | Path | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> dict[str, Any]:
    """Assemble the full run manifest (contract item X).

    Args:
        config: the `VLMTrainingConfig` used.
        corpus_summary: split/scene counts, render contract, manifest hashes.
        lora_report: a `LoRAInjectionReport` (or its `to_dict()`).
        frozen_report: `describe_frozen` output -- proves the vision tower froze.
        prompt_contract: `describe_prompt_contract` output.
        state: the `TrainingState` (or its `to_dict()`).
        metrics: adapted-model metrics on the val split, when evaluated.
        baseline: baseline-model metrics on the val split, when evaluated.
        metrics_test: adapted-model metrics on the **test** split, when
            evaluated. Recorded separately from `metrics` so a reader can tell
            the decision split (val) from the confirmation split (test); a
            REJECTED run must still persist its held-out delta.
        baseline_test: baseline-model metrics on the **test** split, when
            evaluated.
        decision: the acceptance decision, when taken.
        evaluation_budget: the eval-limit contract -- requested limit per split,
            realised `n_evaluated`, and whether any split was truncated.
        code_root: code root for the revision block. Defaults to the repo root.
        started_at / finished_at: ISO timestamps.

    Returns:
        A JSON-serialisable dict.
    """
    from training.vlm.evaluate import describe_acceptance_rule

    lora_dict = (
        lora_report.to_dict() if hasattr(lora_report, "to_dict") else dict(lora_report)
    )
    state_dict = state.to_dict() if hasattr(state, "to_dict") else dict(state)

    manifest: dict[str, Any] = {
        "phase": 6,
        "artifact_kind": "bigearthnet_smolvlm_lora",
        "base_model": getattr(config, "base_model", None),
        "base_model_revision": getattr(config, "revision", None),
        "config_hash": getattr(config, "config_hash", None),
        "code_revision": code_provenance(code_root),
        "prompt_version": prompt_contract.get("version"),
        "prompt_contract": prompt_contract,
        "seed": getattr(config, "seed", None),
        "precision": getattr(config, "precision", None),
        "effective_batch_size": getattr(config, "effective_batch_size", None),
        "max_seq_length": getattr(config, "max_seq_length", None),
        "processor_longest_edge": getattr(config, "processor_longest_edge", None),
        "do_image_splitting": getattr(config, "do_image_splitting", None),
        "trainable_params": lora_dict.get("trainable_params"),
        "trainable_fraction": lora_dict.get("trainable_fraction"),
        "lora": lora_dict,
        "frozen_params": frozen_report,
        "corpus": corpus_summary,
        "training_state": state_dict,
        "metrics": metrics,
        "baseline": baseline,
        "metrics_test": metrics_test,
        "baseline_test": baseline_test,
        "decision": decision,
        "evaluation_budget": evaluation_budget,
        "acceptance_rule_version": describe_acceptance_rule()["version"],
        "plan_deviations": getattr(config, "plan_deviations", {}),
        "environment": environment_versions(),
        "config": (
            config.to_dict() if hasattr(config, "to_dict") else None
        ),
        "started_at": started_at,
        "finished_at": finished_at,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest["manifest_hash"] = manifest_hash(manifest)
    return manifest


def manifest_hash(manifest: dict[str, Any]) -> str:
    """Stable SHA256 over canonical JSON.

    `manifest_hash` itself is excluded so the digest is of the content, not of
    the digest. Keys are sorted and non-JSON types are stringified, so the hash
    is reproducible across machines.
    """
    payload = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def write_run_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    """Write a manifest to `path`. Returns the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return target
