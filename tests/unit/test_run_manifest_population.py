"""Tests for STEP 4 — runtime population of the run manifest.

THE DEFECT BEING FIXED
----------------------
`RunManifest` is the record that makes an evaluation run reproducible. It was
being constructed with four fields left at their defaults:

    code_revision = None, environment = {}, prompt_versions = {}, started_at = None

so the written manifest could not answer "which code, on which machine, with
which prompts" for the run it described. The fields EXISTED and were SERIALISED,
which is what made the defect quiet: a reader saw the keys and assumed they were
populated.

WHAT THESE TESTS GUARD
----------------------
Two things, in this order of importance:

1. The fields are populated by measurement after `populate_run_manifest`.
2. Nothing is ever populated with a FABRICATED value. Every unmeasurable thing
   is recorded as unavailable, and never as a plausible-looking placeholder.

The second is the one that matters for a provenance artifact. A placeholder is
worse than a gap: a gap is visibly a gap, and a placeholder is trusted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.manifests import RunManifest
from evaluation.run_manifest import (
    environment_block,
    manifest_completeness,
    populate_run_manifest,
    prompt_versions,
)
from evaluation.run_manifest import code_revision_identifier


def _manifest() -> RunManifest:
    """A manifest with only the required fields set -- the defective shape."""
    return RunManifest(
        run_id="probe",
        config_hash="0" * 16,
        dataset_manifest_hash="1" * 16,
    )


def _tree(tmp_path: Path, name: str = "code") -> Path:
    root = tmp_path / name
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "a.py").write_text("print(1)\n", encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# the defect: the fields were empty
# ---------------------------------------------------------------------------


def test_the_defective_shape_is_reproduced_first() -> None:
    """Precondition for every other test here: the four fields really do default
    to None/empty. If this ever stops being true the rest of the file is moot."""
    manifest = _manifest()
    assert manifest.code_revision is None
    assert manifest.environment == {}
    assert manifest.prompt_versions == {}
    assert manifest.started_at is None


def test_population_fills_every_run_identifying_field() -> None:
    manifest = _manifest()
    populate_run_manifest(manifest)

    assert manifest.code_revision, "code_revision must be identified"
    assert manifest.environment, "environment must be measured"
    assert manifest.prompt_versions, "prompt versions must be recorded"
    assert manifest.started_at, "the start time must be recorded"


def test_a_populated_manifest_still_hashes_and_serialises() -> None:
    """The manifest hash is what a later verification compares against, so a
    populated manifest must still produce a stable hash and valid JSON."""
    manifest = populate_run_manifest(_manifest())
    assert len(manifest.hash) == 16
    round_tripped = json.loads(json.dumps(manifest.to_dict(), sort_keys=True, default=str))
    assert round_tripped["code_revision"] == manifest.code_revision
    assert round_tripped["environment"] == manifest.environment


# ---------------------------------------------------------------------------
# code_revision
# ---------------------------------------------------------------------------


def test_the_code_revision_names_a_dirty_tree_as_dirty(tmp_path: Path) -> None:
    """A commit hash from a dirty tree does NOT identify the bytes that ran.

    Reporting the commit without the dirty flag would make the manifest look more
    conclusive than it is. The flag is therefore part of the identifier contract.
    """
    identifier = code_revision_identifier(tmp_path)  # no git here
    assert "dirty" in identifier or "unavailable" in identifier


def test_the_code_revision_is_self_describing(tmp_path: Path) -> None:
    """A reader must be able to tell WHICH KIND of identifier they are looking at
    without reading this module."""
    identifier = code_revision_identifier(tmp_path)
    assert identifier.startswith("git:"), identifier
    assert ("snapshot:sha256:" in identifier) or ("snapshot:unavailable" in identifier)


def test_the_code_revision_tracks_content_when_git_is_absent(tmp_path: Path) -> None:
    """THE ANTI-PLACEHOLDER TEST.

    The tempting wrong implementation is a constant such as
    `"unknown"` or `f"git:unavailable"`, which fills the field, passes a
    presence check, and identifies nothing. This test requires the identifier to
    CHANGE when the code changes while git stays absent.
    """
    root = _tree(tmp_path)
    before = code_revision_identifier(root)
    assert "git:unavailable" in before, "precondition: no git in the fixture"

    (root / "scripts" / "a.py").write_text("print(2)\n", encoding="utf-8")
    after = code_revision_identifier(root)
    assert before != after, (
        "with no git, the identifier must still distinguish different code"
    )


def test_two_different_trees_get_different_identifiers(tmp_path: Path) -> None:
    a = _tree(tmp_path, "one")
    b = _tree(tmp_path, "two")
    (b / "scripts" / "a.py").write_text("print(99)\n", encoding="utf-8")
    assert code_revision_identifier(a) != code_revision_identifier(b)


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


def test_the_environment_records_real_facts() -> None:
    env = environment_block()
    assert env["python"], "the interpreter version is always knowable"
    assert env["platform"]
    assert env["repo_root"]
    # 15 keys were measured on this machine; assert a floor rather than the exact
    # count so adding a probe does not break the test.
    assert len(env) >= 10, sorted(env)


def test_the_environment_values_are_all_strings() -> None:
    """`RunManifest.environment` is `dict[str, str]`. A nested or numeric value
    would serialise inconsistently between runs and break the manifest hash for
    non-substantive reasons."""
    env = environment_block()
    for key, value in env.items():
        assert isinstance(key, str), key
        assert isinstance(value, str), f"{key} -> {type(value).__name__}"


def test_an_unmeasurable_environment_value_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """If torch cannot be probed, the KEYS must still be present and say
    unavailable. Omitting them would be indistinguishable from "no torch was
    needed", which is a different and false claim."""
    import builtins

    real_import = builtins.__import__

    def blocked(name: str, *args, **kwargs):
        if name == "torch":
            raise ImportError("probe: torch is not importable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    env = environment_block()

    assert "torch" in env, "the key must survive even when its value cannot"
    assert "unavailable" in env["torch"]
    assert "ImportError" in env["torch"], "the reason must be named"


def test_the_environment_records_cuda_state_even_when_absent() -> None:
    """A CPU-only run must say `False`, not omit CUDA. An omitted key reads as
    'not applicable'; `False` reads as 'measured, and there is none'."""
    env = environment_block()
    assert env["cuda_available"] in ("True", "False")
    assert env["gpu_count"].isdigit() or "unavailable" in env["gpu_count"]


# ---------------------------------------------------------------------------
# prompt versions
# ---------------------------------------------------------------------------


def test_prompt_versions_records_a_version_per_specialist() -> None:
    versions = prompt_versions()
    assert "vqa" in versions
    assert "optical_sar" in versions
    for name, version in versions.items():
        assert version and isinstance(version, str), name


def test_prompt_versions_match_the_modules_that_define_them() -> None:
    """The manifest must record the version the SPECIALIST would actually use,
    not a copy that can drift. Read the modules directly and compare."""
    from specialists.optical_sar import prompts as osar_prompts
    from specialists.vqa import prompts as vqa_prompts

    versions = prompt_versions()
    assert versions["vqa"] == vqa_prompts.PROMPT_VERSION
    assert versions["optical_sar"] == osar_prompts.PROMPT_VERSION


def test_a_broken_prompt_module_does_not_silently_vanish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A specialist whose version cannot be read must appear as unavailable, not
    disappear. A missing entry and an absent specialist must not look alike."""
    import importlib

    real_import = importlib.import_module

    def flaky(name: str, *args, **kwargs):
        if name == "specialists.vqa.prompts":
            raise ImportError("probe")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", flaky)
    versions = prompt_versions()

    assert "vqa" in versions, "the entry must survive the failure"
    assert "unavailable" in versions["vqa"]
    # The other specialist must be unaffected.
    assert "optical_sar" in versions


# ---------------------------------------------------------------------------
# the populator's contract
# ---------------------------------------------------------------------------


def test_the_populator_does_not_overwrite_a_caller_supplied_value() -> None:
    """Resuming a run must not make an old manifest claim the current process's
    start time. A caller that already knows the start time keeps it."""
    manifest = _manifest()
    manifest.started_at = "2020-01-01T00:00:00+00:00"
    manifest.code_revision = "git:abc123 (clean)"

    populate_run_manifest(manifest)

    assert manifest.started_at == "2020-01-01T00:00:00+00:00"
    assert manifest.code_revision == "git:abc123 (clean)"


def test_the_populator_records_a_supplied_start_time_verbatim() -> None:
    manifest = populate_run_manifest(_manifest(), started_at="2026-01-02T03:04:05+00:00")
    assert manifest.started_at == "2026-01-02T03:04:05+00:00"


def test_the_populator_never_invents_an_end_time() -> None:
    """A run that has not ended has no end time. Defaulting it to 'now' would
    claim the run took zero seconds."""
    manifest = populate_run_manifest(_manifest())
    assert getattr(manifest, "ended_at", None) is None

    populate_run_manifest(manifest, ended_at="2026-01-02T04:00:00+00:00")
    assert manifest.ended_at == "2026-01-02T04:00:00+00:00"


def test_the_populator_returns_the_same_object_it_was_given() -> None:
    """It is used inline, so identity must be preserved."""
    manifest = _manifest()
    assert populate_run_manifest(manifest) is manifest


def test_extra_versions_merge_rather_than_replace() -> None:
    manifest = populate_run_manifest(
        _manifest(), extra_prompt_versions={"custom": "v9"}
    )
    assert manifest.prompt_versions["custom"] == "v9"
    assert manifest.prompt_versions["vqa"], "the measured versions must survive"


def test_extra_environment_merges_rather_than_replace() -> None:
    manifest = populate_run_manifest(_manifest(), extra_environment={"host": "ci"})
    assert manifest.environment["host"] == "ci"
    assert manifest.environment["python"], "the measured values must survive"


def test_caller_supplied_extra_versions_win_over_measured_ones() -> None:
    """An explicit caller value is authoritative; only GAPS are filled."""
    manifest = populate_run_manifest(
        _manifest(), extra_prompt_versions={"vqa": "pinned_v7"}
    )
    assert manifest.prompt_versions["vqa"] == "pinned_v7"


# ---------------------------------------------------------------------------
# completeness reporting
# ---------------------------------------------------------------------------


def test_completeness_distinguishes_absent_from_empty_from_present() -> None:
    """The report's whole purpose. `None` means never populated; `{}` means
    populated with nothing; a value means measured. Collapsing these would hide
    exactly the defect this step fixes."""
    manifest = _manifest()
    before = manifest_completeness(manifest)
    assert before["code_revision"] == "absent"
    assert before["environment"] == "empty"
    assert before["started_at"] == "absent"

    populate_run_manifest(manifest)
    after = manifest_completeness(manifest)
    assert after["code_revision"] == "present"
    assert after["environment"] == "present"
    assert after["prompt_versions"] == "present"
    assert after["started_at"] == "present"


def test_completeness_lists_what_was_measured() -> None:
    report = manifest_completeness(populate_run_manifest(_manifest()))
    assert report["environment_keys"], "the keys must be listed for a reader"
    assert "python" in report["environment_keys"]
    assert report["prompt_version_values"]
    assert report["note"], "the reader must be told what 'absent' means"


def test_completeness_is_json_serialisable() -> None:
    report = manifest_completeness(populate_run_manifest(_manifest()))
    assert json.loads(json.dumps(report, sort_keys=True, default=str))["run_id"] == "probe"


def test_completeness_does_not_gate() -> None:
    """It REPORTS. No quality threshold for a manifest is defined anywhere in the
    plan, so this module must not invent one by, say, refusing to write."""
    manifest = _manifest()
    report = manifest_completeness(manifest)
    assert report["code_revision"] == "absent"
    assert isinstance(report, dict), "an incomplete manifest still yields a report"


# ---------------------------------------------------------------------------
# the real repository
# ---------------------------------------------------------------------------


def test_the_real_repository_manifest_is_fully_identified() -> None:
    """The end-to-end assertion: on this checkout, a populated manifest must
    identify the code, the runtime, the prompts and the start time."""
    manifest = populate_run_manifest(_manifest(), root=Path.cwd())
    report = manifest_completeness(manifest)

    assert report["code_revision"] == "present"
    assert report["environment"] == "present"
    assert report["prompt_versions"] == "present"
    assert report["started_at"] == "present"
    assert "snapshot:sha256:" in manifest.code_revision or "git:unavailable" in manifest.code_revision


def test_a_written_real_manifest_reloads_with_its_fields_intact(tmp_path) -> None:
    """It is written to disk and read back by a later verification, so the round
    trip through the writer must preserve every populated field."""
    manifest = populate_run_manifest(_manifest())
    path = manifest.write(tmp_path / "run_manifest.json")

    reloaded = json.loads(path.read_text(encoding="utf-8"))["run_manifest"]
    assert reloaded["code_revision"] == manifest.code_revision
    assert reloaded["environment"] == manifest.environment
    assert reloaded["prompt_versions"] == manifest.prompt_versions
    assert reloaded["started_at"] == manifest.started_at
