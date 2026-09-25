"""Tests for the BigEarthNet preparation CLI.

The script is what the user runs against 118 GB of imagery, so its failure
modes have to be caught here rather than discovered after an hour of I/O.

Two properties matter:

  * a corpus with problems exits non-zero, so a shell `&&` chain stops
  * `--dry-run` writes nothing, so a smoke run cannot clobber a real manifest
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "prepare_bigearthnet.py"


def make_patch(
    tile_dir: Path,
    patch_name: str,
    labels: list[str],
    split: str | None = None,
) -> Path:
    patch_dir = tile_dir / patch_name
    patch_dir.mkdir(parents=True, exist_ok=True)
    for token in ("B02", "B03", "B04"):
        (patch_dir / f"{patch_name}_{token}.tif").write_bytes(b"stub")
    metadata: dict[str, object] = {"labels": labels}
    if split:
        metadata["split"] = split
    (patch_dir / f"{patch_name}_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return patch_dir


@pytest.fixture()
def ben_tree(tmp_path: Path) -> Path:
    """Six tiles, four patches each, with a scene-disjoint official split.

    The split is assigned per TILE, not per patch, because that is what a real
    geographical split looks like. An earlier version of this fixture gave one
    tile's patches to train, val and test -- a patch-level split, which is the
    leakage this whole subsystem exists to prevent. The fixture was modelling
    a leaky dataset.

        tiles 0-3 -> train   (16 patches)
        tile  4   -> val     ( 4 patches)
        tile  5   -> test    ( 4 patches)
    """
    root = tmp_path / "BigEarthNet-S2"
    root.mkdir()
    for t in range(6):
        tile = root / f"S2A_TILE_{t}"
        split = "train" if t < 4 else ("val" if t == 4 else "test")
        for i in range(4):
            make_patch(
                tile,
                f"patch_{t}_{i}",
                ["Arable land", "Pastures"],
                split=split,
            )
    return root


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=300,
    )


def test_script_exists() -> None:
    assert SCRIPT.exists()


def test_full_run_succeeds(ben_tree: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = run(["--root", str(ben_tree), "--out-dir", str(out)])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CORPUS OK" in result.stdout


def test_full_run_writes_three_files(ben_tree: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    run(["--root", str(ben_tree), "--out-dir", str(out)])

    manifest = out / "bigearthnet_s2_manifest.jsonl"
    pairs = out / "bigearthnet_s2_pairs.jsonl"
    report = out / "bigearthnet_s2_report.json"
    assert manifest.exists()
    assert pairs.exists()
    assert report.exists()


def test_manifest_round_trips_and_audits_clean(
    ben_tree: Path, tmp_path: Path
) -> None:
    from evaluation.leakage import audit_manifest
    from evaluation.manifests import DatasetManifest

    out = tmp_path / "out"
    run(["--root", str(ben_tree), "--out-dir", str(out)])

    manifest = DatasetManifest.read(out / "bigearthnet_s2_manifest.jsonl")
    assert len(manifest) == 24
    assert audit_manifest(manifest).clean is True


def test_official_split_is_used_when_complete(
    ben_tree: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    result = run(["--root", str(ben_tree), "--out-dir", str(out)])
    report = json.loads((out / "bigearthnet_s2_report.json").read_text())

    assert result.returncode == 0
    assert report["summary"]["official_splits"] == {
        "train": 16, "val": 4, "test": 4
    }
    assert "using the OFFICIAL" in result.stdout


def test_a_patch_level_official_split_is_rejected(tmp_path: Path) -> None:
    """Metadata is a claim; disjointness is the evidence.

    A release whose metadata splits one tile's patches across partitions must
    not be adopted just because the metadata says `split`. The script has to
    check, refuse, and derive a scene-level split instead.
    """
    root = tmp_path / "root"
    for t in range(4):
        tile = root / f"TILE_{t}"
        for i in range(4):
            # Patch-level split: one tile spans train/val/test.
            split = "train" if i < 2 else ("val" if i == 2 else "test")
            make_patch(tile, f"patch_{t}_{i}", ["Arable land"], split=split)

    out = tmp_path / "out"
    result = run(["--root", str(root), "--out-dir", str(out)])

    assert "NOT scene-disjoint" in result.stdout, (
        "the leaky official split was adopted without a check"
    )
    assert "deriving a scene-level split" in result.stdout
    assert result.returncode == 1, "a rejected split should be reported as a problem"

    # And the written manifest must actually be clean.
    from evaluation.leakage import audit_manifest
    from evaluation.manifests import DatasetManifest

    manifest = DatasetManifest.read(out / "bigearthnet_s2_manifest.jsonl")
    assert audit_manifest(manifest).clean is True


def test_derived_split_is_recorded_in_the_manifest_notes(tmp_path: Path) -> None:
    """How the split was obtained must be auditable after the fact."""
    root = tmp_path / "root"
    for t in range(6):
        tile = root / f"TILE_{t}"
        for i in range(3):
            make_patch(tile, f"p_{t}_{i}", ["Arable land"])  # no split metadata

    out = tmp_path / "out"
    result = run(["--root", str(root), "--out-dir", str(out)])

    from evaluation.manifests import DatasetManifest

    manifest = DatasetManifest.read(out / "bigearthnet_s2_manifest.jsonl")
    assert manifest.notes.get("split_source") == "derived"
    # Missing official metadata is provenance, not a defect.
    assert result.returncode == 0, result.stdout
    assert "INFORMATIONAL (split provenance)" in result.stdout


def test_substantive_defects_still_block(tmp_path: Path) -> None:
    """Provenance is informational; a real data defect still exits non-zero.

    Guards against the fix being too permissive -- moving provenance to
    informational must not silence vacuous tiling.
    """
    root = tmp_path / "root"
    for i in range(4):
        tile = root / f"TILE_{i}"
        make_patch(tile, f"patch_{i}", ["Arable land"], split="train")

    out = tmp_path / "out"
    result = run(["--root", str(root), "--out-dir", str(out)])
    assert result.returncode == 1, "vacuous tiling stopped blocking"
    assert "one patch" in result.stdout


def test_pairs_contain_no_count_questions(ben_tree: Path, tmp_path: Path) -> None:
    """Plan section 44, checked on the actual emitted file."""
    out = tmp_path / "out"
    run(["--root", str(ben_tree), "--out-dir", str(out)])

    lines = (out / "bigearthnet_s2_pairs.jsonl").read_text().splitlines()
    assert lines
    for line in lines:
        question = json.loads(line)["question"].lower()
        assert "how many" not in question
        assert "where" not in question


def test_pairs_carry_image_dir_and_split(ben_tree: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    run(["--root", str(ben_tree), "--out-dir", str(out)])

    first = json.loads(
        (out / "bigearthnet_s2_pairs.jsonl").read_text().splitlines()[0]
    )
    for key in ("patch_id", "scene_id", "split", "image_dir", "question", "answer"):
        assert key in first


def test_dry_run_writes_nothing(ben_tree: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = run(["--root", str(ben_tree), "--out-dir", str(out), "--dry-run"])

    assert result.returncode == 0
    assert "DRY RUN" in result.stdout
    assert not out.exists(), "dry run created files"


def test_limit_stops_early(ben_tree: Path, tmp_path: Path) -> None:
    """A --limit smoke run must exit 0.

    It cannot have a complete split by construction, so the
    split-completeness warning is informational rather than a problem. An
    earlier version returned 1 here, which made the smoke-run flag fail on
    every invocation -- a guard that fires on the tool used to validate the
    guard is a broken guard.
    """
    out = tmp_path / "out"
    result = run(
        ["--root", str(ben_tree), "--out-dir", str(out), "--limit", "5"]
    )
    assert result.returncode == 0, result.stdout
    assert "found 5 patches" in result.stdout
    assert "TRUNCATED" in result.stdout


def test_limit_run_reports_split_completeness_as_informational(
    ben_tree: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    result = run(
        ["--root", str(ben_tree), "--out-dir", str(out), "--limit", "5"]
    )
    assert "INFORMATIONAL" in result.stdout
    # Case-insensitive: this asserts the guidance is present, not how it is
    # capitalised. An earlier version asserted "Re-run" against a lowercase
    # message and failed on typography rather than behaviour.
    assert "re-run without --limit" in result.stdout.lower()


def test_limit_run_still_catches_real_leakage(tmp_path: Path) -> None:
    """Truncation must not disable the checks it does not affect.

    Leakage is not a truncation artefact. If the tiles that WERE discovered
    span partitions, that must still be reported as a problem on a --limit run.

    The fixture needs a complete official split (train/val/test all present),
    or the disjointness check is never reached: an incomplete split is
    discarded wholesale and a clean derived split is built instead, which
    repairs the leak without reporting it. That is a different code path, and
    it is covered by `test_incomplete_official_split_is_discarded`.
    """
    root = tmp_path / "root"
    for t in range(6):
        tile = root / f"TILE_{t}"
        for i in range(3):
            # Patch-level leak WITH all three partitions present.
            split = ("train", "val", "test")[i]
            make_patch(tile, f"p_{t}_{i}", ["Arable land"], split=split)

    out = tmp_path / "out"
    result = run(
        ["--root", str(root), "--out-dir", str(out), "--limit", "6"]
    )
    assert "NOT scene-disjoint" in result.stdout, (
        "truncation suppressed leakage detection"
    )
    assert result.returncode == 1


def test_incomplete_official_split_is_discarded_and_repaired(
    tmp_path: Path,
) -> None:
    """An incomplete official split is not used at all.

    Discovered by a broken test. A dataset whose metadata carries only
    `train` and `test` (no `val`) cannot produce a three-way manifest from
    that metadata, so the script discards it entirely and derives a scene-level
    split. Where that metadata was ALSO patch-level, the derived split repairs
    the leak -- the manifest comes out clean and the run exits 0.

    That is a defensible outcome, but it must not be silent: the discarded
    metadata has to be visible in the output and recorded in the manifest.
    """
    root = tmp_path / "root"
    for t in range(4):
        tile = root / f"TILE_{t}"
        for i in range(3):
            split = "train" if i < 2 else "test"  # leaky AND incomplete
            make_patch(tile, f"p_{t}_{i}", ["Arable land"], split=split)

    out = tmp_path / "out"
    result = run(["--root", str(root), "--out-dir", str(out)])

    assert "no complete official split" in result.stdout
    assert "deriving one at scene level" in result.stdout
    # Provenance is reported, not blocking. The corpus was repaired, so the
    # run must succeed -- refusing to train on a corpus that was just fixed
    # would be the wrong answer.
    assert result.returncode == 0, result.stdout

    from evaluation.leakage import audit_manifest
    from evaluation.manifests import DatasetManifest

    manifest = DatasetManifest.read(out / "bigearthnet_s2_manifest.jsonl")
    assert manifest.notes.get("split_source") == "derived"
    assert audit_manifest(manifest).clean is True, (
        "the derived split did not repair the patch-level leak"
    )


def test_missing_root_exits_2(tmp_path: Path) -> None:
    result = run(["--root", str(tmp_path / "nope"), "--out-dir", str(tmp_path / "o")])
    assert result.returncode == 2


def test_vacuous_split_exits_nonzero(tmp_path: Path) -> None:
    """Every tile holding one patch makes scene splitting meaningless."""
    root = tmp_path / "root"
    for i in range(4):
        tile = root / f"TILE_{i}"
        make_patch(tile, f"patch_{i}", ["Arable land"], split="train")

    out = tmp_path / "out"
    result = run(["--root", str(root), "--out-dir", str(out)])

    assert result.returncode == 1, "a vacuous corpus was reported as usable"
    assert "one patch" in result.stdout


def test_report_records_the_manifest_hash(ben_tree: Path, tmp_path: Path) -> None:
    """The hash is what ties a trained artifact back to its corpus."""
    out = tmp_path / "out"
    run(["--root", str(ben_tree), "--out-dir", str(out)])
    report = json.loads((out / "bigearthnet_s2_report.json").read_text())
    assert len(report["manifest_hash"]) == 64
    assert report["config_hash"]
    assert report["seed"] == 42


def test_rerun_overwrites_cleanly(ben_tree: Path, tmp_path: Path) -> None:
    """Preparation is idempotent; a second run must not fail on existing files."""
    out = tmp_path / "out"
    first = run(["--root", str(ben_tree), "--out-dir", str(out)])
    second = run(["--root", str(ben_tree), "--out-dir", str(out)])

    assert first.returncode == 0
    assert second.returncode == 0