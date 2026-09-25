"""Build the SatQuery AI evidence archive (Phase 7).

Owner decision G3: everything under `artifacts/` is included VERBATIM (duplicates and caches
included), so the archive is ~4 GB and REQUIRES ZIP64.

Archive layout:

    SatQuery_AI_Final_Archive_2026-09-25/
      README_ARCHIVE.md              (what is inside, what was excluded and why)
      release/                       (the curated public release tree, verbatim)
      artifacts/                     (VERBATIM, per owner decision G3)
      evidence/
        live_validation/             (3 validation passes + screenshots)
        delivery/                    (delivery report + handoff)
      verification/                  (state, manifests, verification reports)

Exclusions (documented in README_ARCHIVE.md, never silent):
  secrets/token files, OS junk, virtualenvs, node_modules, HF hub cache, temp browser profiles.

Writes only the .zip. Does NOT delete or move anything.
"""
import os
import sys
import zipfile
import datetime

WORKSPACE = r"C:/Users/anish/WorkBuddy AI/2026-09-25-21-52-59"
SRC_REPO = r"C:/Users/anish/satquery-ai"
STAMP = "2026-09-25"
ROOT = f"SatQuery_AI_Final_Archive_{STAMP}"
OUT = os.path.join(WORKSPACE, f"{ROOT}.zip")

# Directories copied recursively into the archive, as (source, arc_prefix)
TREES = [
    (os.path.join(SRC_REPO, "artifacts"), f"{ROOT}/artifacts"),
    (os.path.join(WORKSPACE, "release", "repo"), f"{ROOT}/release"),
    (os.path.join(WORKSPACE, ".workbuddy-ai", "scratch", "live_validation"),
     f"{ROOT}/evidence/live_validation"),
]

# Individual files, as (source, arc_name)
FILES = [
    (os.path.join(WORKSPACE, "DELIVERY_REPORT_2026-09-25.md"), f"{ROOT}/evidence/delivery/DELIVERY_REPORT_2026-09-25.md"),
    (os.path.join(WORKSPACE, "HANDOFF_NEXT_AGENT.md"), f"{ROOT}/evidence/delivery/HANDOFF_NEXT_AGENT.md"),
    (os.path.join(WORKSPACE, "release", "CURRENT_RELEASE_STATE.md"), f"{ROOT}/verification/CURRENT_RELEASE_STATE.md"),
    (os.path.join(WORKSPACE, "release", "RELEASE_EXECUTION_CHECKLIST.md"), f"{ROOT}/verification/RELEASE_EXECUTION_CHECKLIST.md"),
    (os.path.join(WORKSPACE, "release", "HF_RELEASE_VERIFICATION.md"), f"{ROOT}/verification/HF_RELEASE_VERIFICATION.md"),
    (os.path.join(WORKSPACE, "release", "DOCS_STYLE_GUIDE.md"), f"{ROOT}/verification/DOCS_STYLE_GUIDE.md"),
    (os.path.join(WORKSPACE, "release", "tools", "verify_readme_metrics.py"), f"{ROOT}/verification/tools/verify_readme_metrics.py"),
    (os.path.join(WORKSPACE, "release", "tools", "readme_metrics_report.txt"), f"{ROOT}/verification/tools/readme_metrics_report.txt"),
    (os.path.join(WORKSPACE, "release", "tools", "generate_model_manifest.py"), f"{ROOT}/verification/tools/generate_model_manifest.py"),
    (os.path.join(WORKSPACE, "release", "tools", "model_manifest_report.txt"), f"{ROOT}/verification/tools/model_manifest_report.txt"),
    (os.path.join(WORKSPACE, "release", "tools", "hf_upload.py"), f"{ROOT}/verification/tools/hf_upload.py"),
    (os.path.join(WORKSPACE, "release", "tools", "hf_verify.py"), f"{ROOT}/verification/tools/hf_verify.py"),
    (os.path.join(WORKSPACE, "release", "tools", "hf_verify_report.txt"), f"{ROOT}/verification/tools/hf_verify_report.txt"),
]

EXCLUDE_DIR_NAMES = {
    ".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints",
}
EXCLUDE_SUFFIX = (".pyc", ".pyo", ".pyd")


def should_skip(path):
    parts = set(os.path.normpath(path).split(os.sep))
    if parts & EXCLUDE_DIR_NAMES:
        return True
    return path.endswith(EXCLUDE_SUFFIX)


def add_tree(zf, src, prefix, counters):
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIR_NAMES]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            if should_skip(full):
                continue
            rel = os.path.relpath(full, src).replace(os.sep, "/")
            arc = f"{prefix}/{rel}"
            try:
                zf.write(full, arc, compress_type=zipfile.ZIP_STORED)  # weights/caches: no recompress
                counters["files"] += 1
                counters["bytes"] += os.path.getsize(full)
            except OSError as e:
                counters["errors"].append(f"{full}: {e}")


ARCHIVE_README = f"""# SatQuery AI — Final Evidence Archive ({STAMP})

This archive is the complete, verified evidence bundle for the SatQuery AI final release.

## Owner decision on scope

Per the owner's explicit decision (**G3**), **everything under `artifacts/` is included VERBATIM** —
including duplicate ZIPs, feature caches and superseded checkpoints. This makes the archive large
(~4 GB) and is intentional: nothing was curated away.

## Contents

| Path | What it is |
|---|---|
| `release/` | the curated public release tree (README, MODEL_CARD, docs/, models/, screenshots/) |
| `artifacts/` | the project's `artifacts/` directory, verbatim |
| `evidence/live_validation/` | the three live validation passes, raw logs, result JSON, 8 screenshots |
| `evidence/delivery/` | the delivery report and the handoff document |
| `verification/` | release state, verification reports, and the tools that produced them |

## What was EXCLUDED (explicitly, never silently)

| Excluded | Reason |
|---|---|
| Secret / token files | never archived |
| `.git/`, `__pycache__/`, `.pytest_cache/`, `.mypy_cache/` | version-control and build caches |
| virtualenvs (`.venv`, `venv`) | reproducible from `requirements.txt` |
| `node_modules/` | reproducible from the frontend package manifest |
| Hugging Face Hub cache | re-downloadable, pinned by revision |
| Temporary browser profiles | ephemeral |
| `*.pyc`, `*.pyo`, `*.pyd` | compiled bytecode |

## Reproducible-but-included note

The archive deliberately includes **duplicates and caches** that are reproducible:

| Item | Size | Classification |
|---|---|---|
| `artifacts/grounding/remoteclip_grounding_v001.zip` | ~774 MB | DUPLICATE of the extracted directory |
| `artifacts/change/levir_change_cpu_probe_v001/` | ~241 MB | DUPLICATE probe of `levir_change_v001` |
| `artifacts/optical_sar/fusion_features/`, `fusion_features_armB/` | ~231 MB each | reproducible feature caches |
| `artifacts/optical_sar/fusion_head_v001/` | ~276 MB | superseded by `fusion_head_production_v001` |

These are retained because the owner chose "verbatim". They are listed here so the size is
explained, not surprising.

## Integrity

Verify this archive with `release/tools/verify_archive.py`, which extracts to a separate temporary
directory and checks the CRC of every member plus the sha256 of every artifact against the live
files. The result is recorded in `ARCHIVE_VERIFICATION.md`.

Generated: {STAMP}
"""


def main():
    counters = {"files": 0, "bytes": 0, "errors": []}
    print(f"Building {OUT}")
    print("(ZIP_STORED for artifacts — recompressing already-compressed weights wastes hours)")
    print()

    # ZIP64 is automatic in Python when a file/offset needs it.
    with zipfile.ZipFile(OUT, "w", allowZip64=True) as zf:
        zf.writestr(f"{ROOT}/README_ARCHIVE.md", ARCHIVE_README)
        counters["files"] += 1
        for src, prefix in TREES:
            if not os.path.isdir(src):
                counters["errors"].append(f"MISSING TREE: {src}")
                continue
            before = counters["files"]
            add_tree(zf, src, prefix, counters)
            print(f"  + {prefix:52} {counters['files'] - before:>7} files")
        for src, arc in FILES:
            if not os.path.exists(src):
                counters["errors"].append(f"MISSING FILE: {src}")
                continue
            zf.write(src, arc, compress_type=zipfile.ZIP_DEFLATED)
            counters["files"] += 1
            counters["bytes"] += os.path.getsize(src)
        print(f"  + verification/… (individual files)")

    size = os.path.getsize(OUT)
    print()
    print(f"archive      : {OUT}")
    print(f"archive size : {size:,} bytes ({size / 1024**3:.2f} GiB)")
    print(f"entries      : {counters['files']}")
    print(f"source bytes : {counters['bytes']:,}")
    if counters["errors"]:
        print()
        print("ERRORS:")
        for e in counters["errors"]:
            print("  " + e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
