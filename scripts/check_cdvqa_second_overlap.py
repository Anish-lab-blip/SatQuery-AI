"""Decide whether SECOND's public release provides CDVQA's imagery.

Phase 10 is blocked because the CDVQA repository ships annotations only -- zero
PNGs. The paper says the imagery is drawn from the 2,968 publicly available
SECOND pairs, and our own measurement of the annotations agrees exactly (2,968
unique ``file_name`` values). But CDVQA's ``file_name`` is a *sparse global id*
(``00003.png`` .. ``24649.png``, 21,679 gaps), not a compact ``0..2967`` index,
so whether SECOND ships the same names is an open question.

This script answers that question by measurement and nothing else.

**It never guesses a mapping.** It reports four counts and a verdict. If the
names do not line up, the correct response is content matching or contacting the
authors -- not inferring an order.

Usage
-----
    python scripts/check_cdvqa_second_overlap.py --second-zip data/cdvqa/second.zip
    python scripts/check_cdvqa_second_overlap.py --second-zip second.zip \
        --cdvqa-root data/cdvqa --json out.json

The archive is opened, never extracted -- safe to run on a multi-GB download
before deciding whether to unpack it. Both **zip and RAR** are accepted: SECOND's
public release is named `.zip` but is actually a **RAR5** archive, detected by
magic bytes. Trusting the extension produced a false "not a readable zip" failure
on a perfectly good download.

Exit codes
----------
    0  names match: SECOND supplies exactly the CDVQA imagery -> Phase 10 unblocked
    2  no overlap: naming differs -> needs content matching or the authors
    3  partial overlap: ambiguous -> inspect before acting
    4  bad input: archive or annotations missing/unreadable
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# The union of unique file_name across Train/Val/Test, measured from the shipped
# annotations. Used as a cross-check, not as an assumption.
EXPECTED_CDVQA_UNIQUE = 2968

ANNOTATION_SPLITS = ("Train", "Val", "Test", "Test2")

EXIT_MATCH = 0
EXIT_NO_OVERLAP = 2
EXIT_PARTIAL = 3
EXIT_BAD_INPUT = 4


class CDVQAOverlapError(Exception):
    """Raised when the inputs cannot be inspected at all."""


def _read_magic(path: Path, n: int = 8) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


def _libarchive_tar() -> str | None:
    """Locate a libarchive-backed `tar` that can read RAR.

    GNU tar cannot read RAR, but libarchive's bsdtar can -- and Windows ships
    one at `System32\\tar.exe`. Found by capability, not by assuming a name.
    """
    import shutil

    candidates: list[str] = []
    if sys.platform == "win32":
        candidates.append(str(Path(os.environ.get("SystemRoot", r"C:\Windows"))
                              / "System32" / "tar.exe"))
    for name in ("bsdtar", "tar"):
        found = shutil.which(name)
        if found:
            candidates.append(found)

    for cand in candidates:
        if not Path(cand).exists():
            continue
        try:
            out = subprocess.run(
                [cand, "--version"], capture_output=True, text=True, timeout=20
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if "bsdtar" in out.stdout or "libarchive" in out.stdout:
            return cand
    return None


def png_basenames_from_rar(rar_path: Path) -> set[str]:
    """PNG basenames in a RAR, via libarchive. Read-only; never extracts."""
    tar_bin = _libarchive_tar()
    if tar_bin is None:
        raise CDVQAOverlapError(
            f"{rar_path} is a RAR archive, but no libarchive-backed 'tar' is "
            "available to read it. Install 7-Zip/WinRAR, or extract the archive "
            "and point --second-zip at a zip instead."
        )
    try:
        proc = subprocess.run(
            [tar_bin, "-tf", str(rar_path)],
            capture_output=True, text=True, timeout=1800,
        )
    except subprocess.SubprocessError as exc:
        raise CDVQAOverlapError(f"could not list {rar_path}: {exc}") from exc
    if proc.returncode != 0:
        raise CDVQAOverlapError(
            f"listing {rar_path} failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()[:300]}"
        )
    return {
        Path(line.strip()).name
        for line in proc.stdout.splitlines()
        if line.strip().lower().endswith(".png")
    }


def png_basenames_from_zip(zip_path: Path) -> set[str]:
    """Return every PNG basename in the archive, at any depth.

    Basenames, not full member paths: SECOND may nest its images per split, and
    CDVQA's ``file_name`` is a bare filename. Comparing full paths would produce
    a false 'no overlap' on a perfectly good download.

    Handles **both** zip and RAR. SECOND's public release is named `.zip` but is
    in fact a **RAR5** archive -- detected by magic bytes, because trusting the
    extension produced a false "not a readable zip" failure on a good download.
    """
    if not zip_path.is_file():
        raise CDVQAOverlapError(f"archive not found: {zip_path}")

    magic = _read_magic(zip_path)
    if magic.startswith(b"Rar!\x1a\x07"):
        return png_basenames_from_rar(zip_path)

    try:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
    except zipfile.BadZipFile as exc:
        raise CDVQAOverlapError(
            f"{zip_path} is neither a readable zip nor a RAR archive "
            f"(magic {magic.hex()}): {exc}"
        ) from exc

    return {
        Path(n).name
        for n in names
        if n.lower().endswith(".png") and not n.endswith("/")
    }


def cdvqa_unique_filenames(cdvqa_root: Path) -> dict[str, set[str]]:
    """Return {split: {file_name}} for every annotation split found on disk."""
    ann = cdvqa_root / "annotations"
    if not ann.is_dir():
        raise CDVQAOverlapError(f"annotations directory not found: {ann}")

    per_split: dict[str, set[str]] = {}
    for split in ANNOTATION_SPLITS:
        f = ann / f"{split}_images.json"
        if not f.is_file():
            continue
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise CDVQAOverlapError(f"cannot read {f}: {exc}") from exc
        rows = payload.get("images")
        if not isinstance(rows, list):
            raise CDVQAOverlapError(f"{f} has no 'images' list")
        per_split[split] = {r["file_name"] for r in rows}

    if not per_split:
        raise CDVQAOverlapError(
            f"no <Split>_images.json found under {ann} "
            f"(looked for {', '.join(ANNOTATION_SPLITS)})"
        )
    return per_split


def analyse(second: set[str], per_split: dict[str, set[str]]) -> dict:
    """Compute the overlap facts. Pure function, no I/O."""
    cdvqa = set().union(*per_split.values())
    inter = second & cdvqa

    if not cdvqa:
        raise CDVQAOverlapError("CDVQA annotation set is empty")

    if inter == cdvqa and cdvqa <= second:
        verdict, code = "MATCH", EXIT_MATCH
    elif not inter:
        verdict, code = "NO_OVERLAP", EXIT_NO_OVERLAP
    else:
        verdict, code = "PARTIAL", EXIT_PARTIAL

    return {
        "second_png_count": len(second),
        "cdvqa_unique_count": len(cdvqa),
        "cdvqa_expected_unique": EXPECTED_CDVQA_UNIQUE,
        "cdvqa_count_matches_expectation": len(cdvqa) == EXPECTED_CDVQA_UNIQUE,
        "intersection": len(inter),
        "cdvqa_not_in_second": len(cdvqa - second),
        "second_not_in_cdvqa": len(second - cdvqa),
        "per_split_unique": {k: len(v) for k, v in sorted(per_split.items())},
        "verdict": verdict,
        "exit_code": code,
    }


def _report(res: dict, second_zip: Path) -> None:
    print(f"archive            : {second_zip}")
    print(f"SECOND PNG files   : {res['second_png_count']:,}")
    print(f"CDVQA unique names : {res['cdvqa_unique_count']:,}"
          f"  (expected {res['cdvqa_expected_unique']:,})")
    print(f"per-split unique   : "
          + ", ".join(f"{k}={v:,}" for k, v in res["per_split_unique"].items()))
    print()
    print(f"intersection       : {res['intersection']:,}")
    print(f"CDVQA not in SECOND: {res['cdvqa_not_in_second']:,}")
    print(f"SECOND not in CDVQA: {res['second_not_in_cdvqa']:,}")
    print()

    if res["verdict"] == "MATCH":
        print("VERDICT: MATCH -- SECOND supplies every CDVQA image name.")
        print("Phase 10 is UNBLOCKED. Extract to data/cdvqa/images/ and re-run")
        print("  python -c \"from training.data.cdvqa import *; ...\"")
        print("or simply re-run the adapter tests with require_images=True.")
    elif res["verdict"] == "NO_OVERLAP":
        print("VERDICT: NO OVERLAP -- the naming conventions differ.")
        print("Do NOT infer a mapping from ordering or counts. Two legitimate")
        print("options:")
        print("  (a) derive the mapping by image content matching against the")
        print("      CDVQA question/answer semantics, and validate it;")
        print("  (b) request the image release from the CDVQA authors")
        print("      (https://github.com/YZHJessica/CDVQA).")
        print("Phase 10 remains BLOCKED.")
    else:
        print("VERDICT: PARTIAL -- ambiguous, do not proceed automatically.")
        print("Inspect which names matched and which did not before acting.")

    if not res["cdvqa_count_matches_expectation"]:
        print()
        print(f"NOTE: CDVQA unique count {res['cdvqa_unique_count']:,} differs from")
        print(f"the expected {res['cdvqa_expected_unique']:,}. The annotation set")
        print("may have changed -- re-check before trusting this comparison.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--second-zip", type=Path, required=True,
                    help="SECOND archive; inspected, never extracted")
    ap.add_argument("--cdvqa-root", type=Path, default=Path("data/cdvqa"),
                    help="CDVQA root containing annotations/ (default: data/cdvqa)")
    ap.add_argument("--json", type=Path, default=None,
                    help="optional path to write the result as JSON")
    args = ap.parse_args(argv)

    try:
        second = png_basenames_from_zip(args.second_zip)
        per_split = cdvqa_unique_filenames(args.cdvqa_root)
        res = analyse(second, per_split)
    except CDVQAOverlapError as exc:
        print(f"FAILED to inspect: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT

    _report(res, args.second_zip)

    if args.json is not None:
        # The output directory is the caller's to choose and may not exist yet;
        # failing here would discard a completed measurement over a mkdir.
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {**res, "second_zip": str(args.second_zip),
                 "cdvqa_root": str(args.cdvqa_root)},
                indent=2, sort_keys=True,
            ),
            encoding="utf-8",
        )
        print()
        print(f"result written to {args.json}")

    return res["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
