#!/usr/bin/env python
"""Phase 12 extraction FINAL INTEGRITY verifier (read-only).

Independent, on-disk verification that the extraction produced exactly what it
claimed. Deliberately does NOT trust the extractor's own report: it re-walks the
output tree and checks structure, then cross-checks totals against the reports.

Per-tree structural rules
-------------------------
  BigEarthNet-S2   : 28,000 patch dirs, each with exactly 12 band files whose
                     suffixes are {B01..B09, B11, B12, B8A}  (no B10)
  BigEarthNet-S1   : 28,000 patch dirs, each with exactly 2 files {VH, VV}
  Reference_Maps   : 28,000 patch dirs, each with exactly 1 file ending
                     _reference_map.tif

Also asserts: no zero-byte files, and per-tree file/byte totals agree with the
corresponding <stage>_report.json.

Deletes nothing. Exit codes: 0 = verified, 1 = verification failed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BANDS = {"B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"}

TREES = {
    "s2":  ("BigEarthNet-S2", 12),
    "s1":  ("BigEarthNet-S1", 2),
    "ref": ("Reference_Maps", 1),
}


def scan_tree(root: Path, kind: str) -> dict:
    """Walk one tree and aggregate per-patch leaves."""
    entry = {
        "tree": root.name,
        "root": str(root),
        "present": root.exists(),
        "files": 0,
        "bytes": 0,
        "zero_byte": 0,
        "patch_dirs": 0,
        "bad_patch_dirs": 0,
        "bad_patch_sample": [],
    }
    if not root.exists():
        return entry

    per_patch: dict[str, list[str]] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            entry["files"] += 1
            entry["bytes"] += size
            if size == 0:
                entry["zero_byte"] += 1
            rel = os.path.relpath(full, root).replace("\\", "/")
            parts = rel.split("/")
            if len(parts) >= 3:
                per_patch.setdefault("/".join(parts[:2]), []).append(parts[-1])

    entry["patch_dirs"] = len(per_patch)
    for pdir, leaves in per_patch.items():
        bad = False
        if kind == "s2":
            got = {lf.rsplit(".tif", 1)[0].rsplit("_", 1)[-1] for lf in leaves}
            if len(leaves) != 12 or got != BANDS:
                bad = True
        elif kind == "s1":
            got = {lf.rsplit(".tif", 1)[0].rsplit("_", 1)[-1] for lf in leaves}
            if len(leaves) != 2 or got != {"VH", "VV"}:
                bad = True
        elif kind == "ref":
            if len(leaves) != 1 or not leaves[0].endswith("_reference_map.tif"):
                bad = True
        if bad:
            entry["bad_patch_dirs"] += 1
            if len(entry["bad_patch_sample"]) < 5:
                entry["bad_patch_sample"].append({"patch": pdir, "leaves": sorted(leaves)})
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 12 integrity verifier")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--log-root", default=None)
    ap.add_argument("--stages", default="s2,s1,ref")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--expect-s2", type=int, default=336_000)
    ap.add_argument("--expect-s1", type=int, default=56_000)
    ap.add_argument("--expect-ref", type=int, default=28_000)
    ap.add_argument("--expect-patch-dirs", type=int, default=28_000)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root else repo / "data" / "bigearthnet_v2" / "reben"
    log_root = Path(args.log_root).resolve() if args.log_root else None

    expected_files = {"s2": args.expect_s2, "s1": args.expect_s1, "ref": args.expect_ref}
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    res: dict = {"ok": True, "failures": [], "out_root": str(out_root), "trees": {}}

    def fail(msg: str) -> None:
        res["failures"].append(msg)
        res["ok"] = False

    for s in stages:
        if s not in TREES:
            continue
        tree_name, _ = TREES[s]
        entry = scan_tree(out_root / tree_name, s)
        entry["expected_files"] = expected_files[s]

        if not entry["present"]:
            fail(f"tree missing: {tree_name}")
            entry["ok"] = False
            res["trees"][s] = entry
            continue

        if entry["files"] != expected_files[s]:
            fail(f"{tree_name} has {entry['files']} files, expected {expected_files[s]}")
        if entry["patch_dirs"] != args.expect_patch_dirs:
            fail(f"{tree_name} has {entry['patch_dirs']} patch dirs, expected {args.expect_patch_dirs}")
        if entry["bad_patch_dirs"]:
            fail(f"{tree_name} has {entry['bad_patch_dirs']} structurally bad patch dirs")
        if entry["zero_byte"]:
            fail(f"{tree_name} has {entry['zero_byte']} zero-byte files")

        entry["ok"] = not (entry["files"] != expected_files[s]
                           or entry["patch_dirs"] != args.expect_patch_dirs
                           or entry["bad_patch_dirs"] or entry["zero_byte"])

        # cross-check against the extractor's own report
        if log_root is not None:
            rp = log_root / f"{s}_report.json"
            if rp.exists():
                rep = json.loads(rp.read_text(encoding="utf-8"))
                if entry["files"] != rep.get("files_extracted"):
                    fail(f"{tree_name} on-disk files {entry['files']} != report "
                         f"files_extracted {rep.get('files_extracted')}")
                if entry["bytes"] != rep.get("bytes_written"):
                    fail(f"{tree_name} on-disk bytes {entry['bytes']} != report "
                         f"bytes_written {rep.get('bytes_written')}")
                entry["report_files_extracted"] = rep.get("files_extracted")
                entry["report_bytes_written"] = rep.get("bytes_written")
            else:
                fail(f"report not found for stage {s}: {rp}")

        res["trees"][s] = entry

    Path(args.out_json).write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("INTEGRITY_OK" if res["ok"] else "INTEGRITY_FAILED")
    for f in res["failures"]:
        print("  FAIL: " + f)
    for s, e in res["trees"].items():
        print(f"  {e['tree']:<18} files={e['files']:,} bytes={e['bytes']:,} "
              f"patch_dirs={e['patch_dirs']:,} bad={e['bad_patch_dirs']} zero={e['zero_byte']}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
