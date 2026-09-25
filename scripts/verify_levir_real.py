"""Verify a real LEVIR-CD tree against the frozen Phase 9 contract.

Answers, by MEASUREMENT, the questions that a synthetic fixture cannot:

  * which layout is this, and is it the one `configs/base.yaml` names?
  * are A / B / label actually paired, tile for tile?
  * what are the real image dimensions, mode and dtype?
  * what is the real label encoding, and is it the 0/255 the loader assumes?
  * do the split counts match `change.levir_split` (7120 / 1024 / 2048)?
  * do the shipped `list/*.txt` splits agree with the filename prefixes?
  * are the three splits SCENE-disjoint -- i.e. is there any leak?
  * does every sampled tile actually differ between T1 and T2?

    python scripts/verify_levir_real.py --data-root data/levir

Exit 0 = every check passed. Exit 2 = the tree is missing or unreadable.
Exit 3 = a check FAILED; the report names which.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def hr(title: str = "") -> None:
    print("-" * 72)
    if title:
        print(title)
        print("-" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify a real LEVIR-CD tree")
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data" / "levir"))
    ap.add_argument("--sample", type=int, default=200,
                    help="tiles to open and measure")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="write the report JSON here")
    args = ap.parse_args()

    from core.config import load_config
    from training.change.dataset import (
        detect_levir_layout,
        load_levir_dataset,
        read_split_list,
        verify_split_lists,
    )

    root = Path(args.data_root)
    cfg = load_config()
    frozen = dict(cfg.get("change.levir_split", {}) or {})
    tile_size = int(cfg.get("change.tile_size", 256))
    checks: list[tuple[str, bool, str]] = []
    report: dict = {"data_root": str(root), "config_hash": cfg.hash}

    print("=" * 72)
    print("LEVIR-CD REAL-DATA VERIFICATION")
    print("=" * 72)
    print(f"data root   : {root}")
    print(f"config hash : {cfg.hash}")
    print(f"frozen split: {frozen}   (patches)")
    print(f"tile size   : {tile_size}px")
    print()

    if not root.exists():
        print(f"FAILED: {root} does not exist")
        return 2

    layout = detect_levir_layout(root)
    print(f"layout      : {layout}")
    if layout is None:
        print("FAILED: no LEVIR-CD layout detected")
        return 2
    checks.append(("layout detected", True, str(layout)))
    report["layout"] = layout

    # -- 1. directory accounting -------------------------------------------
    hr("1. DIRECTORY ACCOUNTING")
    dir_counts = {}
    for sub in ("A", "B", "label"):
        d = root / sub
        n = sum(1 for _ in d.glob("*.png")) if d.is_dir() else 0
        dir_counts[sub] = n
        print(f"  {sub:<8} {n:>7} png")
    report["dir_counts"] = dir_counts

    paired = dir_counts["A"] == dir_counts["B"] == dir_counts["label"]
    checks.append(("A/B/label counts agree", paired,
                   f"{dir_counts['A']} / {dir_counts['B']} / {dir_counts['label']}"))
    if not paired:
        print("  !! A, B and label do not hold the same number of files")

    # -- 2. split counts vs the frozen contract ----------------------------
    hr("2. SPLIT COUNTS vs change.levir_split")
    per_split = {}
    for split in ("train", "val", "test"):
        items = load_levir_dataset(str(root), splits=(split,))
        scenes = {i["group_key"] for i in items}
        per_split[split] = {"tiles": len(items), "scenes": len(scenes)}
        want = frozen.get(split)
        mark = "OK" if want in (None, len(items)) else "MISMATCH"
        print(f"  {split:<6} tiles={len(items):>6}  scenes={len(scenes):>5}  "
              f"frozen={want}  {mark}")
        checks.append((f"{split} tile count matches frozen",
                       want in (None, len(items)), f"{len(items)} vs {want}"))
    report["splits"] = per_split

    # -- 3. shipped split lists vs filename prefixes -----------------------
    hr("3. SHIPPED list/*.txt vs FILENAME PREFIX")
    agreement = verify_split_lists(root)
    report["split_list_agreement"] = agreement
    if not agreement:
        print("  (no list/ directory -- nothing to compare)")
    for split, stats in sorted(agreement.items()):
        ok = stats["declared_only"] == 0 and stats["derived_only"] == 0
        print(f"  {split:<6} declared={stats['declared']:>6} "
              f"derived={stats['derived']:>6} "
              f"agree={stats['agreement']:>6} "
              f"declared_only={stats['declared_only']} "
              f"derived_only={stats['derived_only']}  "
              f"{'OK' if ok else 'DISAGREE'}")
        checks.append((f"{split} list agrees with prefix", ok,
                       f"{stats['declared_only']}/{stats['derived_only']}"))

    # -- 4. scene disjointness across splits (the leakage control) ---------
    hr("4. SCENE DISJOINTNESS (leakage control)")
    scene_sets = {}
    for split in ("train", "val", "test"):
        items = load_levir_dataset(str(root), splits=(split,))
        scene_sets[split] = {i["group_key"] for i in items}
    overlap_tv = scene_sets["train"] & scene_sets["val"]
    overlap_tt = scene_sets["train"] & scene_sets["test"]
    overlap_vt = scene_sets["val"] & scene_sets["test"]
    for name, ov in (("train/val", overlap_tv), ("train/test", overlap_tt),
                     ("val/test", overlap_vt)):
        ok = not ov
        print(f"  {name:<11} shared scenes: {len(ov):>4}  {'OK' if ok else 'LEAK'}")
        checks.append((f"{name} scene-disjoint", ok, str(sorted(ov)[:3])))
    report["scene_overlap"] = {
        "train_val": len(overlap_tv), "train_test": len(overlap_tt),
        "val_test": len(overlap_vt),
    }

    # -- 5. tile integrity on a real sample --------------------------------
    hr(f"5. TILE INTEGRITY (sample of {args.sample})")
    import numpy as np
    from PIL import Image

    all_items = []
    for split in ("train", "val", "test"):
        all_items.extend(load_levir_dataset(str(root), splits=(split,)))
    rng = random.Random(args.seed)
    sample = rng.sample(all_items, min(args.sample, len(all_items)))

    dims: Counter = Counter()
    modes: Counter = Counter()
    label_values: Counter = Counter()
    identical = 0
    missing = 0
    label_size_mismatch = 0

    for item in sample:
        for key in ("t1_path", "t2_path", "label_path"):
            if not Path(item[key]).exists():
                missing += 1
        with Image.open(item["t1_path"]) as im1:
            dims[im1.size] += 1
            modes[im1.mode] += 1
            a = np.asarray(im1.convert("RGB"))
        with Image.open(item["t2_path"]) as im2:
            b = np.asarray(im2.convert("RGB"))
        with Image.open(item["label_path"]) as iml:
            lab = np.asarray(iml.convert("L"))
        if lab.shape != a.shape[:2]:
            label_size_mismatch += 1
        for value, count in zip(*np.unique(lab, return_counts=True)):
            label_values[int(value)] += int(count)
        if np.array_equal(a, b):
            identical += 1

    print(f"  image sizes      : {dict(dims)}")
    print(f"  image modes      : {dict(modes)}")
    print(f"  label values     : {dict(sorted(label_values.items()))}")
    print(f"  missing siblings : {missing}")
    print(f"  label size != A  : {label_size_mismatch}")
    print(f"  T1 == T2 tiles   : {identical} / {len(sample)}")

    checks.append(("every sampled tile has A/B/label", missing == 0, str(missing)))
    checks.append((f"images are {tile_size}x{tile_size}",
                   set(dims) == {(tile_size, tile_size)}, str(dict(dims))))
    checks.append(("label matches image size", label_size_mismatch == 0,
                   str(label_size_mismatch)))
    checks.append(("T1 and T2 are not identical", identical == 0, str(identical)))

    # The labels are NOT strictly {0, 255}. Measured on the real archive: a
    # handful of pixels sit at 156 -- a resampling/compression artefact of the
    # 1024 -> 256 tiling. `training.change.train.LABEL_CHANGE_LEVEL` is 128
    # precisely so these binarize to "change" rather than being lost, so they are
    # handled, not ignored. The check therefore asks the question that matters:
    # can the intermediate values move a metric? Reported either way, so a
    # growing intermediate population is visible rather than silent.
    total_px = sum(label_values.values()) or 1
    intermediate = {v: c for v, c in label_values.items() if v not in (0, 255)}
    inter_frac = sum(intermediate.values()) / total_px
    checks.append((
        "labels binary 0/255 (intermediate negligible)",
        inter_frac < 1e-4,
        f"intermediate={dict(sorted(intermediate.items()))} "
        f"= {inter_frac:.2e} of {total_px:,} px",
    ))
    # And: every intermediate value must binarize the same way the 255 class does,
    # i.e. sit at or above LABEL_CHANGE_LEVEL. A value below it would silently
    # become "no change" and would be a genuine defect.
    from training.change.train import LABEL_CHANGE_LEVEL

    below = {v: c for v, c in intermediate.items() if v < LABEL_CHANGE_LEVEL}
    checks.append((
        f"intermediate labels >= LABEL_CHANGE_LEVEL ({LABEL_CHANGE_LEVEL})",
        not below,
        f"below-threshold={dict(sorted(below.items()))}",
    ))

    report["sample"] = {
        "n": len(sample),
        "sizes": {f"{w}x{h}": c for (w, h), c in dims.items()},
        "modes": dict(modes),
        "label_values": {str(k): v for k, v in sorted(label_values.items())},
        "missing_siblings": missing,
        "label_size_mismatch": label_size_mismatch,
        "identical_pairs": identical,
    }

    # -- verdict -----------------------------------------------------------
    hr("VERDICT")
    failed = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<38} {detail}")
    print()
    report["checks"] = [{"name": n, "passed": o, "detail": d} for n, o, d in checks]
    report["failed"] = [c[0] for c in failed]

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        print(f"  report written: {args.out}")

    print()
    if failed:
        print(f"VERIFICATION: FAILED ({len(failed)} check(s))")
        return 3
    print("VERIFICATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
