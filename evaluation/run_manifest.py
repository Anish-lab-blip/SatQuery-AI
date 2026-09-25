"""Populate a `RunManifest` from the running process, honestly.

WHY THIS EXISTS (STEP 4)
------------------------
`RunManifest` exists to make an evaluation run reproducible, and it was being
constructed with four fields left at their defaults:

    code_revision  = None          -> nothing identified the code
    environment    = {}            -> nothing identified the runtime
    prompt_versions= {}            -> nothing identified the prompts
    started_at     = None          -> nothing identified when it ran

A manifest with those empty is a manifest in name only: it cannot answer "which
code, on which machine, with which prompts" for the run it claims to describe.
This module fills them from measurements taken at call time.

THE RULE THIS MODULE FOLLOWS
----------------------------
**Never fabricate. Record the absence.** Every field here is either a value that
was genuinely measured, or an explicit statement that it was not available. There
is no default that looks like data:

* an unreadable git commit becomes a `code_revision` string saying so, plus the
  content snapshot digest that IS available;
* a missing prompt module is reported as `"unavailable"`, not as an empty string
  that reads like "no prompts were used";
* an environment value that cannot be probed is recorded as `"unavailable"`.

The one thing this module must never do is emit a plausible-looking placeholder.
A placeholder is worse than a gap, because a gap is visibly a gap.
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.code_revision import REPO_ROOT, code_snapshot_digest, code_revision
from evaluation.manifests import RunManifest

__all__ = [
    "environment_block",
    "prompt_versions",
    "code_revision_identifier",
    "populate_run_manifest",
]

#: Modules whose PROMPT_VERSION identifies the wording a specialist was given.
#: Enumerated rather than globbed: a specialist that stops exporting a version
#: should show up as MISSING, not silently drop out of the manifest.
_PROMPT_MODULES: tuple[tuple[str, str], ...] = (
    ("vqa", "specialists.vqa.prompts"),
    ("optical_sar", "specialists.optical_sar.prompts"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment_block() -> dict[str, str]:
    """Measured runtime facts, as a flat `str -> str` map.

    Values are stringified because `RunManifest.environment` is
    `dict[str, str]`: a nested structure would serialise inconsistently between
    runs and would break the manifest hash for non-substantive reasons.

    Anything that cannot be measured says `"unavailable"`. Nothing is guessed.
    """
    env: dict[str, str] = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unavailable",
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "repo_root": str(REPO_ROOT),
    }

    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = str(bool(torch.cuda.is_available()))
        env["cuda_version"] = torch.version.cuda or "unavailable"
        env["cudnn_version"] = str(torch.backends.cudnn.version() or "unavailable")
        env["gpu_count"] = str(torch.cuda.device_count())
        env["gpu_names"] = ",".join(
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ) or "none"
    except Exception as exc:  # noqa: BLE001
        # torch may be absent or broken in a minimal environment. Either way the
        # manifest should say so rather than omit the keys, because an omitted key
        # is indistinguishable from "this run needed no torch".
        reason = f"unavailable ({type(exc).__name__})"
        for key in ("torch", "cuda_available", "cuda_version", "cudnn_version",
                    "gpu_count", "gpu_names"):
            env[key] = reason

    try:
        import numpy

        env["numpy"] = numpy.__version__
    except Exception as exc:  # noqa: BLE001
        env["numpy"] = f"unavailable ({type(exc).__name__})"

    return env


def prompt_versions() -> dict[str, str]:
    """The `PROMPT_VERSION` each specialist would use, by specialist name.

    A prompt change is a behaviour change that no config hash captures, so it
    belongs in the manifest. A specialist whose version cannot be read is
    recorded as `"unavailable"` rather than skipped -- a missing entry and an
    absent specialist must not look the same.
    """
    import importlib

    versions: dict[str, str] = {}
    for name, module_name in _PROMPT_MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "PROMPT_VERSION", None)
            versions[name] = str(version) if version is not None else "unavailable"
        except Exception as exc:  # noqa: BLE001
            versions[name] = f"unavailable ({type(exc).__name__})"
    return versions


def code_revision_identifier(root: str | Path = REPO_ROOT) -> str:
    """A single string identifying the code, for `RunManifest.code_revision`.

    `RunManifest.code_revision` is `str | None`. This fills it with git metadata
    when git is genuinely usable, and with the content snapshot digest when it is
    not. The string is self-describing either way, so a reader never has to guess
    which kind of identifier they are looking at.

    A dirty working tree is named explicitly. It matters: a commit hash from a
    dirty tree does NOT identify the bytes that ran, and a manifest that omits
    that fact overstates what it proves.
    """
    root = Path(root)
    git = code_revision(root)
    snapshot = code_snapshot_digest(root)
    snap_part = (
        f"snapshot:sha256:{snapshot['sha256']}"
        if snapshot.get("available")
        else "snapshot:unavailable"
    )

    if git.get("available"):
        commit = git["commit"]
        state = "dirty" if git.get("dirty") else "clean"
        return f"git:{commit} ({state}) {snap_part}"

    reason = git.get("reason") or "reason not reported"
    return f"git:unavailable ({reason}) {snap_part}"


def populate_run_manifest(
    manifest: RunManifest,
    *,
    root: str | Path = REPO_ROOT,
    started_at: str | None = None,
    ended_at: str | None = None,
    extra_prompt_versions: dict[str, str] | None = None,
    extra_environment: dict[str, str] | None = None,
) -> RunManifest:
    """Fill the run-identifying fields of `manifest` IN PLACE, from measurement.

    Fields that the caller has already set are left alone. That matters for a
    resume: a manifest describing an earlier run must not silently acquire the
    current process's start time.

    `started_at`/`ended_at`: when supplied they are recorded verbatim; when not,
    `started_at` is taken as now and `ended_at` is left as it was. A caller that
    wants a duration must supply both -- this function will not invent an end
    time for a run that has not ended.

    Returns the same object, mutated, so it can be used inline.
    """
    now = _utc_now()

    if not manifest.code_revision:
        manifest.code_revision = code_revision_identifier(root)

    if not manifest.environment:
        manifest.environment = environment_block()

    if not manifest.prompt_versions:
        manifest.prompt_versions = prompt_versions()

    if extra_prompt_versions:
        # Caller-supplied versions win, but gaps are still filled: an unknown
        # version for one specialist should not blank the others.
        for key, value in extra_prompt_versions.items():
            manifest.prompt_versions[key] = value

    if extra_environment:
        manifest.environment.update(extra_environment)

    if started_at is not None:
        manifest.started_at = started_at
    elif not manifest.started_at:
        manifest.started_at = now

    # `ended_at` is deliberately not defaulted. RunManifest has no such field
    # today; when it gains one, an unset end time must stay unset rather than
    # becoming "now", which would claim the run took zero seconds.
    if ended_at is not None:
        setattr(manifest, "ended_at", ended_at)

    return manifest


def manifest_completeness(manifest: RunManifest) -> dict[str, Any]:
    """What the manifest can and cannot identify, as a checker-friendly summary.

    Emitted alongside a manifest so a reader can tell an empty field that was
    never populated from one that was populated and happens to be short. It is a
    REPORT, not a gate: this module does not decide whether a manifest is good
    enough, because no quality threshold is defined anywhere in the plan.
    """
    def state(value: Any) -> str:
        if value is None:
            return "absent"
        if isinstance(value, (dict, str)) and not value:
            return "empty"
        return "present"

    return {
        "run_id": manifest.run_id,
        "code_revision": state(manifest.code_revision),
        "environment": state(manifest.environment),
        "prompt_versions": state(manifest.prompt_versions),
        "started_at": state(manifest.started_at),
        "model_revisions": state(manifest.model_revisions),
        "thresholds": state(manifest.thresholds),
        "environment_keys": sorted(manifest.environment),
        "prompt_version_values": dict(manifest.prompt_versions),
        "note": (
            "States whether each run-identifying field is present. 'absent' means "
            "it was never populated; this module never fills a field with a "
            "placeholder, so a populated field is always a measurement."
        ),
    }
