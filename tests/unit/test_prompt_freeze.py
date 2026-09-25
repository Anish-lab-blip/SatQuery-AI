"""Tests for the frozen prompt-set record (`evaluation/prompt_freeze.json`).

WHY THIS EXISTS
---------------
Plan §72 Gate 5 requires "all prompts frozen" and §79 VLM lists "frozen prompt
set", but the plan names no artifact for it (audit: §79 line 4338, §72 line 3840,
freeze doc §5 `docs/ARCHITECTURE_FREEZE.md:178`, §55/§63 — none name a file,
path, or schema). The owner approved a committed freeze record at
`evaluation/prompt_freeze.json`: `evaluation/` holds the repo's manifest/hash
machinery and is NOT gitignored, whereas `artifacts/` is (`.gitignore:20`), so a
record there would not be durable evidence.

The record freezes the prompt modules' SOURCE BYTES. The freeze principle says
"Prompts are versioned FILES", so the file is the unit being frozen; source bytes
cover both the prompt constants and the rendering logic that produces the final
text. Hashing "the constants only" would require enumerating which constants
count — a choice the plan does not specify, i.e. an invention.

These tests prove the record is LOAD-BEARING:
  1. the recorded module set EQUALS the live discovered set — adding a prompt
     module breaks the freeze loudly;
  2. each recorded `version` equals the module's live `PROMPT_VERSION`;
  3. each recorded `sha256` equals the live `content_hash_file` value;
  4. the recorded `set_sha256` recomputes from the recorded modules;
  5. the outer `hash` is the self-hash of the `prompt_freeze` payload;
  6. the committed document equals what the builder computes from the live tree;
  7. NEGATIVE CONTROL: perturbing one module's bytes IN MEMORY changes the set
     hash — a hash that cannot change proves nothing.

The prompt files are only ever READ here. Freezing must not become editing.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

from evaluation.leakage import content_hash_file

REPO_ROOT = Path(__file__).resolve().parents[2]
FREEZE_PATH = REPO_ROOT / "evaluation" / "prompt_freeze.json"
SPECIALISTS_DIR = REPO_ROOT / "specialists"

#: The recorded hashing method, mirrored verbatim in the artifact.
METHOD = "sha256 over sorted (module_path, file_sha256) pairs"

#: Outer self-hash length, mirroring `RunManifest.hash` (evaluation/manifests.py:308).
SELF_HASH_CHARS = 16


# ---------------------------------------------------------------------------
# Discovery + canonical hashing — the single source of truth for the record
# ---------------------------------------------------------------------------
def _discovered_module_paths() -> list[Path]:
    """Every specialist prompt module, by DISCOVERY (never a hardcoded list).

    `specialists/*/prompts.py` means a newly added prompt module cannot silently
    escape the freeze.
    """
    return sorted(SPECIALISTS_DIR.glob("*/prompts.py"))


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _live_version(path: Path) -> str:
    """The module's live `PROMPT_VERSION`, read by importing the module."""
    module_name = ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)
    return importlib.import_module(module_name).PROMPT_VERSION


def _module_entries() -> list[dict[str, str]]:
    return [
        {
            "path": _rel(path),
            "version": _live_version(path),
            "sha256": content_hash_file(path),
        }
        for path in _discovered_module_paths()
    ]


def _set_sha256(modules: list[dict[str, str]]) -> str:
    """sha256 of the canonical (sorted-key) JSON of the module entries."""
    blob = json.dumps(modules, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _self_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:SELF_HASH_CHARS]


def build_freeze_document() -> dict:
    """The freeze record computed from the LIVE prompt modules.

    Pure: reads the modules, writes nothing. The committed artifact must equal
    this; regenerate with
    `json.dumps(build_freeze_document(), indent=2, sort_keys=True)`.
    """
    modules = _module_entries()
    payload = {
        "method": METHOD,
        "modules": modules,
        "set_sha256": _set_sha256(modules),
    }
    return {"hash": _self_hash(payload), "prompt_freeze": payload}


def _committed() -> dict:
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The record matches the live prompt set
# ---------------------------------------------------------------------------
def test_recorded_module_set_equals_live_discovered_set() -> None:
    recorded = {entry["path"] for entry in _committed()["prompt_freeze"]["modules"]}
    live = {_rel(path) for path in _discovered_module_paths()}
    assert recorded == live, (
        "a prompt module was added or removed without re-freezing: "
        f"recorded={sorted(recorded)} live={sorted(live)}"
    )


def test_recorded_versions_match_live_prompt_versions() -> None:
    for entry in _committed()["prompt_freeze"]["modules"]:
        live = _live_version(REPO_ROOT / entry["path"])
        assert entry["version"] == live, entry["path"]


def test_recorded_sha256_values_match_live_content_hash_file() -> None:
    for entry in _committed()["prompt_freeze"]["modules"]:
        live = content_hash_file(REPO_ROOT / entry["path"])
        assert entry["sha256"] == live, entry["path"]


# ---------------------------------------------------------------------------
# The record's own hashes recompute
# ---------------------------------------------------------------------------
def test_recorded_set_sha256_matches_recomputed() -> None:
    payload = _committed()["prompt_freeze"]
    assert payload["set_sha256"] == _set_sha256(payload["modules"])


def test_outer_hash_is_the_self_hash_of_the_payload() -> None:
    document = _committed()
    assert document["hash"] == _self_hash(document["prompt_freeze"])


def test_method_string_matches_the_documented_algorithm() -> None:
    assert _committed()["prompt_freeze"]["method"] == METHOD


def test_committed_document_equals_the_builder() -> None:
    # Belt and braces: the committed record must be exactly what the live tree
    # produces. Catches an edit to the artifact or a change to any prompt module.
    assert _committed() == build_freeze_document()


# ---------------------------------------------------------------------------
# Negative control — the freeze must bite
# ---------------------------------------------------------------------------
def test_set_hash_is_sensitive_to_content_perturbation() -> None:
    """A hash that cannot change proves nothing.

    Perturb ONE module's content IN MEMORY ONLY (hash the file bytes plus a
    marker) and show both the per-file hash and the set hash change. Nothing is
    written to the prompt files.
    """
    modules = _committed()["prompt_freeze"]["modules"]
    baseline = _set_sha256(modules)

    first = modules[0]
    raw = (REPO_ROOT / first["path"]).read_bytes()
    perturbed_file_sha = hashlib.sha256(raw + b"\n# perturbed in memory only\n").hexdigest()
    assert perturbed_file_sha != first["sha256"], "the perturbation did not change the file hash"

    mutated = [dict(entry) for entry in modules]
    mutated[0]["sha256"] = perturbed_file_sha
    assert _set_sha256(mutated) != baseline, (
        "the set hash is not sensitive to module content; the freeze is vacuous"
    )
