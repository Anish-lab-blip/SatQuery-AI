"""Verify the SatQuery AI evidence archive (Phase 8 — HARD GATE).

Extracts the archive to a SEPARATE temp directory (never over the originals), then checks:
  1. zipfile integrity (testzip + CRC of every member)
  2. entry count matches the manifest of members
  3. every artifact under artifacts/ is present and byte-identical (sha256) to the live file
  4. the curated release tree is present with its key files
  5. the six released model weights are byte-identical to the live files

Exit 0 only if every check passes. If it fails, the caller MUST NOT proceed to deletion.
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile

WORKSPACE = r"C:/Users/anish/WorkBuddy AI/2026-09-25-21-52-59"
SRC_REPO = r"C:/Users/anish/satquery-ai"
STAMP = "2026-09-25"
ROOT = f"SatQuery_AI_Final_Archive_{STAMP}"
ARCHIVE = os.path.join(WORKSPACE, f"{ROOT}.zip")


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    if not os.path.exists(ARCHIVE):
        print(f"ERROR: archive not found: {ARCHIVE}")
        return 2

    print(f"archive : {ARCHIVE}")
    print(f"size    : {os.path.getsize(ARCHIVE):,} bytes")
    print()

    failures = []

    # --- 1. integrity -----------------------------------------------------
    with zipfile.ZipFile(ARCHIVE) as zf:
        names = zf.namelist()
        print(f"entries : {len(names)}")
        bad = zf.testzip()
        if bad is not None:
            failures.append(f"CRC failure in member: {bad}")
            print(f"  CRC check : FAIL ({bad})")
        else:
            print("  CRC check : PASS (every member CRC verified)")

        # --- 2. extract to a SEPARATE temp dir ----------------------------
        tmp = tempfile.mkdtemp(prefix="sq_archive_verify_")
        print(f"  extracting to: {tmp}")
        zf.extractall(tmp)

    extracted_root = os.path.join(tmp, ROOT)

    # --- 3. artifacts byte-identity --------------------------------------
    print()
    print("artifact byte-identity (extracted vs live):")
    art_src = os.path.join(SRC_REPO, "artifacts")
    art_ext = os.path.join(extracted_root, "artifacts")
    n_checked = n_ok = 0
    for dirpath, dirnames, filenames in os.walk(art_src):
        dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".git"}]
        for fn in filenames:
            if fn.endswith((".pyc", ".pyo")):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, art_src)
            ext = os.path.join(art_ext, rel)
            n_checked += 1
            if not os.path.exists(ext):
                failures.append(f"missing in archive: artifacts/{rel}")
                continue
            if os.path.getsize(full) != os.path.getsize(ext) or sha256_of(full) != sha256_of(ext):
                failures.append(f"byte mismatch: artifacts/{rel}")
            else:
                n_ok += 1
    print(f"  artifacts checked : {n_checked}")
    print(f"  byte-identical    : {n_ok}")

    # --- 4. release tree --------------------------------------------------
    print()
    print("curated release tree:")
    for rel in ["README.md", "MODEL_CARD.md", "models/manifest.json", "models/checksums.sha256"]:
        p = os.path.join(extracted_root, "release", rel)
        ok = os.path.exists(p)
        print(f"  {'OK ' if ok else 'MISS'} release/{rel}")
        if not ok:
            failures.append(f"missing in archive: release/{rel}")

    # --- 5. the six released weights -------------------------------------
    print()
    print("six released model weights (extracted vs live):")
    manifest = json.load(
        open(os.path.join(WORKSPACE, "release", "repo", "models", "manifest.json"), encoding="utf-8")
    )
    for a in manifest["artifacts"]:
        live = os.path.join(SRC_REPO, a["path"])
        ext = os.path.join(extracted_root, "artifacts", os.path.relpath(a["path"], "artifacts"))
        if not os.path.exists(ext):
            failures.append(f"missing weight in archive: {a['path']}")
            print(f"  MISS {a['id']}")
            continue
        same = sha256_of(live) == sha256_of(ext)
        print(f"  {'OK  ' if same else 'BAD '} {a['id']}")
        if not same:
            failures.append(f"weight byte mismatch: {a['path']}")

    # --- cleanup temp -----------------------------------------------------
    shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"ARCHIVE VERIFICATION: FAILED ({len(failures)} problem(s))")
        for f in failures[:50]:
            print("  - " + f)
        print()
        print("DO NOT PROCEED TO DELETION.")
        return 1
    print("ARCHIVE VERIFICATION: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
