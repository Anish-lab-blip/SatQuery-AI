"""Tests for scripts/check_cdvqa_second_overlap.py.

The Phase 10 blocker is a question about the outside world -- does SECOND ship
CDVQA's filenames? -- so the script that answers it must be trustworthy before
it is ever pointed at a 4 GB download. These tests pin the decision logic
against synthetic archives, including the cases where the honest answer is
"ambiguous, stop".

No real dataset is needed: the fixtures build a tiny zip and a tiny annotation
set on tmp_path.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.check_cdvqa_second_overlap import (
    EXIT_BAD_INPUT,
    EXIT_MATCH,
    EXIT_NO_OVERLAP,
    EXIT_PARTIAL,
    CDVQAOverlapError,
    analyse,
    cdvqa_unique_filenames,
    main,
    png_basenames_from_zip,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def _write_annotations(root: Path, per_split: dict[str, list[str]]) -> Path:
    ann = root / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    for split, names in per_split.items():
        rows = [{"file_name": n, "id": i} for i, n in enumerate(names)]
        (ann / f"{split}_images.json").write_text(
            json.dumps({"images": rows}), encoding="utf-8"
        )
    return root


def _write_zip(path: Path, members: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for m in members:
            z.writestr(m, b"\x89PNG\r\n\x1a\n")
    return path


@pytest.fixture()
def cdvqa_root(tmp_path: Path) -> Path:
    return _write_annotations(
        tmp_path / "cdvqa",
        {"Train": ["00003.png", "00011.png"], "Val": ["00013.png"]},
    )


# --------------------------------------------------------------------------
# png_basenames_from_zip
# --------------------------------------------------------------------------
def test_zip_basenames_are_collected(tmp_path: Path):
    z = _write_zip(tmp_path / "a.zip", ["00003.png", "00011.png"])
    assert png_basenames_from_zip(z) == {"00003.png", "00011.png"}


def test_nested_members_are_reduced_to_basenames(tmp_path: Path):
    """SECOND may nest per split; CDVQA's file_name is a bare filename.

    Comparing full member paths would report a false 'no overlap' on a
    perfectly good download.
    """
    z = _write_zip(
        tmp_path / "a.zip",
        ["SECOND/train/00003.png", "SECOND/val/deep/00011.png"],
    )
    assert png_basenames_from_zip(z) == {"00003.png", "00011.png"}


def test_non_png_members_are_ignored(tmp_path: Path):
    z = _write_zip(
        tmp_path / "a.zip",
        ["00003.png", "labels/00003_label.png.tif", "readme.txt", "x.jpg"],
    )
    assert png_basenames_from_zip(z) == {"00003.png"}


def test_directory_entries_are_ignored(tmp_path: Path):
    with zipfile.ZipFile(tmp_path / "a.zip", "w") as z:
        z.writestr("images/", b"")
        z.writestr("images/00003.png", b"\x89PNG\r\n\x1a\n")
    assert png_basenames_from_zip(tmp_path / "a.zip") == {"00003.png"}


def test_missing_zip_raises(tmp_path: Path):
    with pytest.raises(CDVQAOverlapError, match="not found"):
        png_basenames_from_zip(tmp_path / "nope.zip")


def test_corrupt_zip_raises(tmp_path: Path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"this is not a zip")
    with pytest.raises(CDVQAOverlapError, match="neither a readable zip nor a RAR"):
        png_basenames_from_zip(bad)


def test_corrupt_archive_error_names_the_magic_bytes(tmp_path: Path):
    """A wrong-format archive must say *what* it actually is.

    This is the guard for the real defect found on 2026-09-18: SECOND's public
    release is named `.zip` but is a RAR5 archive, and the old reader reported
    only "not a readable zip" -- which sends a reader looking for a corrupt
    download rather than a wrong assumption about the format.
    """
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"this is not a zip")
    with pytest.raises(CDVQAOverlapError) as exc:
        png_basenames_from_zip(bad)
    # The hex of "this is " -- evidence of what was actually on disk.
    assert "7468697320697320" in str(exc.value)


def test_rar_magic_is_detected_and_routed_to_the_rar_reader(tmp_path: Path, monkeypatch):
    """A RAR archive must not be handed to `zipfile`.

    The real SECOND download is RAR5 (`Rar!\\x1a\\x07\\x01\\x00`). Routing is
    asserted directly so the dispatch is pinned independently of whether a
    libarchive `tar` happens to be installed on the test machine.
    """
    import scripts.check_cdvqa_second_overlap as mod

    called: dict[str, object] = {}

    def _fake_rar(path):
        called["path"] = path
        return {"00003.png"}

    monkeypatch.setattr(mod, "png_basenames_from_rar", _fake_rar)
    archive = tmp_path / "second.zip"
    archive.write_bytes(b"Rar!\x1a\x07\x01\x00" + b"\x00" * 32)

    assert mod.png_basenames_from_zip(archive) == {"00003.png"}
    assert called["path"] == archive


# --------------------------------------------------------------------------
# cdvqa_unique_filenames
# --------------------------------------------------------------------------
def test_annotations_are_read_per_split(cdvqa_root: Path):
    got = cdvqa_unique_filenames(cdvqa_root)
    assert got == {"Train": {"00003.png", "00011.png"}, "Val": {"00013.png"}}


def test_duplicate_image_rows_collapse(tmp_path: Path):
    """`images` has one row per question, not per image -- 16x duplication."""
    root = _write_annotations(
        tmp_path / "cdvqa", {"Train": ["00003.png"] * 16}
    )
    assert cdvqa_unique_filenames(root)["Train"] == {"00003.png"}


def test_missing_annotations_dir_raises(tmp_path: Path):
    with pytest.raises(CDVQAOverlapError, match="annotations directory not found"):
        cdvqa_unique_filenames(tmp_path / "empty")


def test_no_split_files_raises(tmp_path: Path):
    root = tmp_path / "cdvqa"
    (root / "annotations").mkdir(parents=True)
    with pytest.raises(CDVQAOverlapError, match="no <Split>_images.json"):
        cdvqa_unique_filenames(root)


def test_malformed_json_raises(tmp_path: Path):
    root = tmp_path / "cdvqa"
    ann = root / "annotations"
    ann.mkdir(parents=True)
    (ann / "Train_images.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(CDVQAOverlapError, match="cannot read"):
        cdvqa_unique_filenames(root)


# --------------------------------------------------------------------------
# analyse -- the decision logic
# --------------------------------------------------------------------------
def test_exact_match_is_match():
    res = analyse({"00003.png", "00011.png"}, {"Train": {"00003.png", "00011.png"}})
    assert res["verdict"] == "MATCH"
    assert res["exit_code"] == EXIT_MATCH
    assert res["intersection"] == 2
    assert res["cdvqa_not_in_second"] == 0


def test_superset_still_matches():
    """SECOND may carry more images than CDVQA used; that is fine."""
    res = analyse(
        {"00003.png", "00011.png", "99999.png"}, {"Train": {"00003.png", "00011.png"}}
    )
    assert res["verdict"] == "MATCH"
    assert res["second_not_in_cdvqa"] == 1


def test_disjoint_is_no_overlap():
    res = analyse({"a.png", "b.png"}, {"Train": {"00003.png"}})
    assert res["verdict"] == "NO_OVERLAP"
    assert res["exit_code"] == EXIT_NO_OVERLAP
    assert res["intersection"] == 0


def test_subset_is_partial():
    res = analyse({"00003.png", "other.png"}, {"Train": {"00003.png", "00011.png"}})
    assert res["verdict"] == "PARTIAL"
    assert res["exit_code"] == EXIT_PARTIAL
    assert res["cdvqa_not_in_second"] == 1


def test_empty_second_is_no_overlap():
    res = analyse(set(), {"Train": {"00003.png"}})
    assert res["verdict"] == "NO_OVERLAP"


def test_expected_count_mismatch_is_flagged():
    """A changed annotation set must not silently pass as a clean comparison."""
    res = analyse({"00003.png"}, {"Train": {"00003.png"}})
    assert res["cdvqa_unique_count"] == 1
    assert res["cdvqa_count_matches_expectation"] is False


# --------------------------------------------------------------------------
# main -- end to end, exit codes
# --------------------------------------------------------------------------
def test_main_match_exits_zero(tmp_path: Path, cdvqa_root: Path, capsys):
    z = _write_zip(tmp_path / "s.zip", ["00003.png", "00011.png", "00013.png"])
    rc = main(["--second-zip", str(z), "--cdvqa-root", str(cdvqa_root)])
    out = capsys.readouterr().out
    assert rc == EXIT_MATCH
    assert "VERDICT: MATCH" in out
    assert "UNBLOCKED" in out


def test_main_no_overlap_exits_two(tmp_path: Path, cdvqa_root: Path, capsys):
    z = _write_zip(tmp_path / "s.zip", ["aaa.png", "bbb.png"])
    rc = main(["--second-zip", str(z), "--cdvqa-root", str(cdvqa_root)])
    out = capsys.readouterr().out
    assert rc == EXIT_NO_OVERLAP
    assert "NO OVERLAP" in out
    assert "Do NOT infer a mapping" in out
    assert "remains BLOCKED" in out


def test_main_partial_exits_three(tmp_path: Path, cdvqa_root: Path, capsys):
    z = _write_zip(tmp_path / "s.zip", ["00003.png", "zzz.png"])
    rc = main(["--second-zip", str(z), "--cdvqa-root", str(cdvqa_root)])
    assert rc == EXIT_PARTIAL
    assert "PARTIAL" in capsys.readouterr().out


def test_main_missing_zip_exits_four(tmp_path: Path, cdvqa_root: Path, capsys):
    rc = main(["--second-zip", str(tmp_path / "nope.zip"),
               "--cdvqa-root", str(cdvqa_root)])
    assert rc == EXIT_BAD_INPUT
    assert "FAILED to inspect" in capsys.readouterr().err


def test_main_writes_json(tmp_path: Path, cdvqa_root: Path):
    z = _write_zip(tmp_path / "s.zip", ["00003.png", "00011.png", "00013.png"])
    out_json = tmp_path / "res.json"
    rc = main(["--second-zip", str(z), "--cdvqa-root", str(cdvqa_root),
               "--json", str(out_json)])
    assert rc == EXIT_MATCH
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["verdict"] == "MATCH"
    assert payload["intersection"] == 3
    assert payload["cdvqa_root"] == str(cdvqa_root)


def test_main_never_extracts_the_archive(tmp_path: Path, cdvqa_root: Path):
    """The whole point is to inspect a 4 GB download before unpacking it."""
    z = _write_zip(tmp_path / "s.zip", ["nested/dir/00003.png"])
    before = {p.name for p in tmp_path.iterdir()}
    main(["--second-zip", str(z), "--cdvqa-root", str(cdvqa_root)])
    after = {p.name for p in tmp_path.iterdir()}
    assert before == after, "the archive must not be extracted"
    assert not (tmp_path / "nested").exists()
