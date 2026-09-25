"""SatQuery AI — Phase 6 adapter artifact export and verification.

WHAT AN ARTIFACT IS (contract items W and X)
--------------------------------------------
A Phase 6 artifact is a directory that answers, without trusting anyone's
memory, three questions:

  1. **What was trained?**  The PEFT adapter weights (`adapter_model.safetensors`
     + `adapter_config.json`).
  2. **What was it trained against?**  The processor/tokenizer files, so the
     image preprocessing at load time is byte-identical to the preprocessing at
     train time. Shipping the adapter without the processor is how an adapter
     silently acquires a preprocessing mismatch that nothing in the logs shows.
  3. **Can it be reproduced?**  `run_manifest.json` (config hash, prompt version,
     seed, environment, trainable-param count, wall time) and
     `ARTIFACT_SHA256SUMS.json` (a SHA256 of every other file in the directory).

`verify_artifact` re-checks the checksum manifest and, when a base model is
given, that the adapter actually reloads onto it. A mismatch raises rather than
warning, because an artifact whose hashes do not match is not evidence.

THE SMOKE-TEST REFUSAL
----------------------
A 1-step CPU probe (`cfg.is_cpu_sized`) must never produce a directory that
reads as a Phase 6 result. `export_adapter` refuses a smoke run unless
`allow_smoke=True` is passed explicitly, and when it is, the manifest records
`is_smoke: true` so the label is impossible to lose.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from core.errors import SatQueryError

__all__ = [
    "CHECKSUM_FILENAME",
    "MANIFEST_FILENAME",
    "ArtifactError",
    "sha256_file",
    "hash_tree",
    "export_adapter",
    "refresh_artifact_checksums",
    "verify_artifact",
]

#: The checksum manifest's own name. Excluded from `hash_tree` -- a file cannot
#: contain its own digest.
CHECKSUM_FILENAME = "ARTIFACT_SHA256SUMS.json"

#: The run manifest's name inside the artifact directory.
MANIFEST_FILENAME = "run_manifest.json"


class ArtifactError(SatQueryError):
    """A Phase 6 artifact could not be written or failed verification."""

    code = "artifact_error"
    user_message = "The Phase 6 adapter artifact is missing or corrupt."


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    """SHA256 of a file's bytes, streamed so a large weight file is fine."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(directory: str | Path) -> dict[str, str]:
    """Map every file under `directory` to its SHA256, keyed by relative POSIX path.

    Sorted by path so the result is deterministic and a re-hash of an unchanged
    tree produces the same dict. The checksum file itself is excluded: including
    it would make the digest depend on the digest.
    """
    root = Path(directory)
    if not root.exists():
        raise ArtifactError(
            f"cannot hash a directory that does not exist: {root}",
            context={"directory": str(root)},
        )

    out: dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel == CHECKSUM_FILENAME:
            continue
        out[rel] = sha256_file(path)
    return out


def _tree_hash(hashes: dict[str, str]) -> str:
    """A single digest over a `{relpath: sha256}` map, canonical and sorted."""
    blob = json.dumps(hashes, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_adapter(
    model: Any,
    processor: Any,
    out_dir: str | Path,
    *,
    run_manifest: dict[str, Any],
    is_smoke: bool = False,
    allow_smoke: bool = False,
) -> Path:
    """Write a complete, self-describing adapter artifact. Returns its directory.

    Writes the PEFT adapter, the processor/tokenizer, the run manifest, and the
    checksum manifest. The run manifest's `adapter_sha256` and `manifest_hash`
    are filled in here, after the weights exist, so the recorded digests are of
    the bytes actually written.

    Args:
        model: the PEFT-wrapped model to save.
        processor: the processor/tokenizer whose preprocessing the adapter
            assumes.
        out_dir: destination directory (created if absent).
        run_manifest: the manifest dict from `run_manifest.build_run_manifest`.
        is_smoke: whether this run was a CPU-sized smoke test. Refused unless
            `allow_smoke` is also set.
        allow_smoke: explicit opt-in to exporting a smoke artifact.

    Raises:
        ArtifactError: the run is a smoke test and `allow_smoke` is not set.
    """
    if is_smoke and not allow_smoke:
        raise ArtifactError(
            "refusing to export an artifact for a CPU-sized smoke run "
            "(cfg.is_cpu_sized): a 1-step probe must not produce something that "
            "reads as a Phase 6 result. Pass allow_smoke=True only for a "
            "deliberately-labelled probe.",
            context={"is_smoke": True},
        )

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    # 1. adapter weights + adapter_config.json
    model.save_pretrained(str(target))
    # 2. processor / tokenizer -- the preprocessing contract travels with the
    #    weights so a load cannot silently use a different resolution.
    processor.save_pretrained(str(target))

    # 3. run manifest, with the adapter digest filled in from the written bytes.
    manifest = dict(run_manifest)
    manifest["is_smoke"] = bool(is_smoke)
    manifest["artifact_dir"] = str(target)
    weights = {
        rel: sha
        for rel, sha in hash_tree(target).items()
        if rel.startswith("adapter_model") or rel == "adapter_config.json"
    }
    manifest["adapter_sha256"] = _tree_hash(weights) if weights else None
    (target / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    # 4. checksum manifest over everything else.
    hashes = hash_tree(target)
    (target / CHECKSUM_FILENAME).write_text(
        json.dumps(hashes, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def refresh_artifact_checksums(out_dir: str | Path) -> Path:
    """Regenerate `ARTIFACT_SHA256SUMS.json` over the directory's current content.

    Call this after any deliberate rewrite of a file inside the artifact (e.g.
    adding the acceptance decision to `run_manifest.json` once it is known).
    Without it, a later rewrite would silently invalidate the checksum manifest
    and `verify_artifact` would then reject an otherwise-intact artifact.
    """
    root = Path(out_dir)
    if not root.is_dir():
        raise ArtifactError(
            f"cannot refresh checksums for a missing directory: {root}",
            context={"artifact_dir": str(root)},
        )
    checksum_path = root / CHECKSUM_FILENAME
    if checksum_path.exists():
        checksum_path.unlink()
    hashes = hash_tree(root)
    checksum_path.write_text(
        json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8"
    )
    return checksum_path


def verify_artifact(
    out_dir: str | Path,
    *,
    base_model: Any = None,
) -> dict[str, Any]:
    """Verify an artifact's integrity and, optionally, that it reloads.

    Args:
        out_dir: the artifact directory.
        base_model: when given, the adapter is reloaded onto it as a reload
            check (contract item V4).

    Returns:
        A report dict with `ok`, `n_files`, `checked`, `reload_ok`, `errors`.

    Raises:
        ArtifactError: a file in the checksum manifest is missing or its digest
            does not match, or the adapter fails to reload.
    """
    root = Path(out_dir)
    if not root.is_dir():
        raise ArtifactError(
            f"artifact directory does not exist: {root}",
            context={"artifact_dir": str(root)},
        )

    checksum_path = root / CHECKSUM_FILENAME
    if not checksum_path.exists():
        raise ArtifactError(
            f"artifact is missing its checksum manifest {CHECKSUM_FILENAME}",
            context={"artifact_dir": str(root)},
        )

    try:
        recorded: dict[str, str] = json.loads(checksum_path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ArtifactError(
            f"could not read {CHECKSUM_FILENAME}: {exc}",
            context={"artifact_dir": str(root)},
        ) from exc

    errors: list[str] = []
    for rel, expected in recorded.items():
        path = root / rel
        if not path.exists():
            errors.append(f"missing file: {rel}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            errors.append(
                f"digest mismatch for {rel}: recorded {expected[:12]}, "
                f"actual {actual[:12]}"
            )

    if errors:
        raise ArtifactError(
            f"artifact failed integrity verification: {len(errors)} problem(s); "
            f"first: {errors[0]}",
            context={"artifact_dir": str(root), "errors": errors[:10]},
        )

    reload_ok: bool | None = None
    unwrapped_from_peft = False
    if base_model is not None:
        from training.vlm.lora import adapter_is_reloadable

        # A PEFT wrapper is the wrong receiver for the reload check: the adapter's
        # saved `target_modules` regex is anchored `^model\.text_model\.`, and the
        # wrapper prefixes its module names (`base_model.model.model.text_model.…`),
        # so `PeftModel.from_pretrained` raises `NoMatchingPeftModuleError` and this
        # gate reports a FALSE NEGATIVE -- aborting a run whose artifact is fine.
        # Measured 2026-09-23: reloading onto `get_base_model()` succeeds, onto the
        # wrapper fails. Unwrap here rather than trusting every caller, because a
        # false negative in a verification gate is a silent failure of the gate.
        unwrap = getattr(base_model, "get_base_model", None)
        if callable(unwrap):
            base_model = unwrap()
            unwrapped_from_peft = True

        reload_ok = adapter_is_reloadable(base_model, root)
        if not reload_ok:
            raise ArtifactError(
                "artifact hashes are intact but the adapter does not reload onto "
                "the supplied base model",
                context={"artifact_dir": str(root)},
            )

    return {
        "ok": True,
        "artifact_dir": str(root),
        "n_files": len(recorded),
        "checked": sorted(recorded),
        "reload_ok": reload_ok,
        "unwrapped_from_peft": unwrapped_from_peft,
        "errors": [],
    }
