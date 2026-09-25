"""Tests for `scripts/diagnose_feature_cache.py`'s exit-code contract.

The script had a silent false-pass defect: it globbed `--cache` as if it were
always a DIRECTORY. Pointed at a consolidated cache FILE (`train.npz`), the glob
returned nothing, the tool printed `image entries (.npz) : 0`, and `main()`
returned 0 unconditionally -- a clean pass for a 170 MB cache it never opened.

These tests exercise both layouts through `main()`, assert the EXIT CODE and the
REPORTED OUTPUT, and would fail against the pre-fix code:

  * `test_a_consolidated_npz_file_is_opened_and_counted` fails pre-fix on the
    stdout assertion -- the old code printed `image entries (.npz) : 0` and
    never the file's real row count.
  * the empty-directory, missing-path and corrupt-file tests fail pre-fix on the
    exit code -- the old `main()` returned 0 for all three.
  * the corrupt-DIRECTORY tests fail pre-fix because the directory verdict was
    `bool(npz)`: it printed `CORRUPT` for truncated files and still exited 0, and
    a small directory was sampled twice (`first 3 + last 3` overlap).

No real artifact, no CROMA, no network. Every fixture is built in `tmp_path`
with `numpy.savez`, so the suite is self-contained and never touches the 170 MB
files under `artifacts/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import scripts.diagnose_feature_cache as diag

#: The row-oriented arrays of a real consolidated fusion-feature cache, and the
#: widths they carry (3x768 GAP, 12+2 masks, one label per row).
CONSOLIDATED_ARRAYS = {
    "optical_gap": 768,
    "sar_gap": 768,
    "joint_gap": 768,
    "optical_mask": 12,
    "sar_mask": 2,
}


def _run(monkeypatch, cache: Path) -> int:
    """Invoke `main()` with `--cache` set to `cache`, as the CLI would."""
    monkeypatch.setattr(
        sys, "argv", ["diagnose_feature_cache.py", "--cache", str(cache)]
    )
    return diag.main()


def _write_consolidated(path: Path, n: int = 7) -> Path:
    """A consolidated `.npz` shaped like the real `train.npz`, but tiny."""
    arrays = {
        key: np.zeros((n, width), dtype=np.float32)
        for key, width in CONSOLIDATED_ARRAYS.items()
    }
    arrays["label_index"] = np.zeros((n,), dtype=np.int64)
    np.savez(path, **arrays)
    return path


# ---------------------------------------------------------------------------
# File mode — the regression this fix exists for
# ---------------------------------------------------------------------------


def test_a_consolidated_npz_file_is_opened_and_counted(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """A consolidated `.npz` must be OPENED, not globbed as an empty dir.

    Pre-fix this test fails on the stdout assertions: the old code treated the
    file as a directory, found no `*.npz` inside it, and printed
    `image entries (.npz) : 0` for a cache holding 7 real rows.
    """
    cache = _write_consolidated(tmp_path / "train.npz", n=7)

    code = _run(monkeypatch, cache)
    out = capsys.readouterr().out

    assert code == 0
    # The load-bearing assertion: the REAL row count, from the opened file.
    assert "image entries        : 7" in out
    # The exact false-pass line the old code printed must be gone.
    assert "image entries (.npz) : 0" not in out
    # The arrays were actually read, by name and shape.
    assert "optical_gap" in out
    assert "label_index" in out
    assert "(7, 768)" in out


def test_a_consolidated_file_with_rows_still_passes(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """Control: the zero-row fix must not over-tighten a healthy file."""
    cache = _write_consolidated(tmp_path / "train.npz", n=7)

    code = _run(monkeypatch, cache)
    out = capsys.readouterr().out

    assert code == 0
    assert "image entries        : 7" in out


def test_a_consolidated_file_with_zero_rows_fails(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """A file that holds arrays but no ROWS inspected nothing, so it fails.

    This mirrors the no-arrays branch: the file opened, but zero entries were
    examined, which is the "NOTHING was inspected" case the docstring names.
    """
    cache = tmp_path / "train.npz"
    np.savez(cache, features=np.zeros((0, 2318), dtype=np.float32))

    code = _run(monkeypatch, cache)
    out = capsys.readouterr().out

    assert code != 0
    # The report still shows the count that explains the verdict.
    assert "image entries        : 0" in out
    assert "no rows" in out
    assert "NOTHING WAS INSPECTED" in out


def test_a_scalar_only_npz_does_not_print_a_nonsense_warning(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """A rank-0 scalar has no first axis, so there is nothing to "disagree"."""
    cache = tmp_path / "scalar.npz"
    np.savez(cache, scalar=np.float32(1.0))

    code = _run(monkeypatch, cache)
    out = capsys.readouterr().out

    assert code != 0
    assert "disagree on their first axis: []" not in out
    assert "WARNING" not in out


def test_a_file_that_does_not_load_fails_with_a_message(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """An unloadable `.npz` is a failure, and the file is named in the output."""
    bad = tmp_path / "broken.npz"
    bad.write_bytes(b"this is not an npz archive at all")

    code = _run(monkeypatch, bad)
    out = capsys.readouterr().out

    assert code != 0
    assert "CORRUPT broken.npz" in out
    assert "NOTHING WAS INSPECTED" in out


# ---------------------------------------------------------------------------
# Directory mode — the existing layout must keep working
# ---------------------------------------------------------------------------


def test_a_populated_directory_is_counted(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """The control: the per-sample layout still reports and exits 0."""
    cache = tmp_path / "cache"
    cache.mkdir()
    for i in range(4):
        np.savez(cache / f"{i:05d}.npz", patches=np.zeros((1, 768), dtype=np.float32))

    code = _run(monkeypatch, cache)
    out = capsys.readouterr().out

    assert code == 0
    assert "image entries (.npz) : 4" in out
    assert "OK      00000.npz" in out


def test_the_directory_report_names_its_sample_coverage(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """The report must state the sampling bound, not just imply it.

    With 20 entries the loadable check samples 6; a truncation at index 10 is
    outside that sample, so the reader has to be told how much was checked.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    for i in range(20):
        np.savez(cache / f"{i:05d}.npz", patches=np.zeros((1, 768), dtype=np.float32))

    code = _run(monkeypatch, cache)
    out = capsys.readouterr().out

    assert code == 0
    assert "load-checked" in out
    assert "6 of 20 entries" in out


def test_a_directory_with_no_npz_entries_fails(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """The silent-pass case: an empty directory must NOT report a clean pass."""
    empty = tmp_path / "empty_cache"
    empty.mkdir()

    code = _run(monkeypatch, empty)
    out = capsys.readouterr().out

    assert code != 0
    assert "image entries (.npz) : 0" in out
    assert "NOTHING WAS INSPECTED" in out


# ---------------------------------------------------------------------------
# Directory mode — a sampled truncation is a finding, not a pass
# ---------------------------------------------------------------------------


def test_a_directory_whose_npz_are_all_corrupt_fails(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """Three unloadable `.npz`: the tool prints CORRUPT and must exit non-zero."""
    cache = tmp_path / "cache"
    cache.mkdir()
    for i in range(3):
        (cache / f"{i:05d}.npz").write_bytes(b"this is not an npz archive")

    code = _run(monkeypatch, cache)
    out = capsys.readouterr().out

    assert code != 0
    assert "CORRUPT 00000.npz" in out
    assert "3 sampled .npz failed to load" in out
    assert "NOTHING WAS INSPECTED" in out


def test_a_directory_with_one_corrupt_npz_among_valid_ones_fails(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """ANY sampled corruption fails, even when other files in the dir are valid.

    The directory is populated (5 entries), so presence alone would read as a
    pass; the truncated first file is what must fail the verdict.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    for i in range(5):
        path = cache / f"{i:05d}.npz"
        if i == 0:
            path.write_bytes(b"an interrupted np.savez left this partial")
        else:
            np.savez(path, patches=np.zeros((1, 768), dtype=np.float32))

    code = _run(monkeypatch, cache)
    out = capsys.readouterr().out

    assert code != 0
    assert "image entries (.npz) : 5" in out
    assert "CORRUPT 00000.npz" in out
    assert "1 sampled .npz failed to load" in out


def test_a_small_directory_is_not_sampled_twice(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """A 3-file directory must not print each file twice in the sample block.

    The sample is `first 3 + last 3`; for 3 files those slices are identical, so
    an undeduplicated sample reports a 3-entry cache as if it held 6.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    for i in range(3):
        np.savez(cache / f"{i:05d}.npz", patches=np.zeros((1, 768), dtype=np.float32))

    code = _run(monkeypatch, cache)
    out = capsys.readouterr().out

    assert code == 0
    # Scope the count to the loadable-check block: filenames legitimately appear
    # again in the NEWEST WRITES block, which would mask a double-sampled check.
    block = out.split("LOADABLE CHECK", 1)[1].split("NEWEST WRITES", 1)[0]
    for i in range(3):
        assert block.count(f"{i:05d}.npz") == 1
    assert block.count("OK      ") == 3


# ---------------------------------------------------------------------------
# Missing path
# ---------------------------------------------------------------------------


def test_a_missing_path_fails(monkeypatch, tmp_path: Path, capsys) -> None:
    """A path that does not exist is a failure, not a clean pass."""
    code = _run(monkeypatch, tmp_path / "does_not_exist")

    out = capsys.readouterr().out

    assert code != 0
    assert "does not exist" in out
    assert "NOTHING WAS INSPECTED" in out
