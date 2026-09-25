#!/usr/bin/env python
"""Phase 12 extraction PREFLIGHT verifier (read-only).

Gate that must pass BEFORE any byte is downloaded. Fails loudly on any mismatch.

Checks
------
1. manifest identity : declared hash reproduced with the documented recipe
                       (evaluation/manifests.py:192-204), record count, splits,
                       T2 scene-block count and purity.
2. composition       : how many of the selected patches come from the clean
                       parquet vs the snow/cloud/shadow parquet.
3. path-set equality : s2_dirs.txt / s1_dirs.txt / ref_dirs.txt must reproduce
                       the manifest EXACTLY (set equality, both directions).
                       This is deliberately NOT tautological: the expected paths
                       are rebuilt from the manifest + parquets, not from the
                       dirs files themselves.
4. plan.json         : expected counts agree with the dirs files.
5. environment       : interpreter version and zstandard import.

Writes a JSON report and prints a human summary. Deletes nothing.
Exit codes: 0 = all checks pass, 1 = at least one check failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def build(repo: Path) -> dict:
    res: dict = {"ok": True, "failures": []}

    def fail(msg: str) -> None:
        res["failures"].append(msg)
        res["ok"] = False

    # ---------------------------------------------------------------- manifest
    mf = repo / "artifacts" / "phase12_selection" / "selection_manifest_seed10.jsonl"
    if not mf.exists():
        fail(f"manifest missing: {mf}")
        return res

    lines = mf.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])["_manifest"]
    recs = [json.loads(l) for l in lines[1:] if l.strip()]

    # Documented recipe. Sorting by (dataset_id, sample_id) makes the hash
    # independent of file order.
    canon = sorted(recs, key=lambda d: (d["dataset_id"], d["sample_id"]))
    blob = json.dumps(canon, sort_keys=True, default=str).encode()
    recomputed = hashlib.sha256(blob).hexdigest()

    sp = Counter(r["split"] for r in recs)
    scene_map: dict[str, set] = {}
    for r in recs:
        scene_map.setdefault(r["scene_id"], set()).add(r["split"])
    impure = [k for k, v in scene_map.items() if len(v) > 1]

    res["manifest"] = {
        "path": str(mf),
        "name": header.get("name"),
        "declared_hash": header["hash"],
        "recomputed_hash": recomputed,
        "hash_ok": recomputed == header["hash"],
        "count": len(recs),
        "declared_count": header.get("count"),
        "scenes": len(scene_map),
        "declared_scenes": header.get("scenes"),
        "impure_blocks": len(impure),
        "splits": dict(sp),
        "label_policy": header.get("notes", {}).get("label_policy"),
        "scene_key": header.get("notes", {}).get("scene_key"),
        "config_hash": header.get("notes", {}).get("config_hash"),
    }
    if not res["manifest"]["hash_ok"]:
        fail("manifest hash does not reproduce")
    if len(recs) != 28_000:
        fail(f"manifest record count {len(recs)} != 28,000")
    if header.get("count") != 28_000:
        fail(f"manifest declared count {header.get('count')} != 28,000")
    if len(scene_map) != header.get("scenes"):
        fail(f"scene count {len(scene_map)} != declared {header.get('scenes')}")
    if impure:
        fail(f"{len(impure)} scene_id blocks straddle splits")
    for k, v in (("train", 20_000), ("val", 4_000), ("test", 4_000)):
        if sp.get(k) != v:
            fail(f"split {k} = {sp.get(k)}, want {v}")
    if header.get("notes", {}).get("config_hash") != "78f1e3700da15aa1":
        fail("manifest config_hash is not the frozen 78f1e3700da15aa1")

    # --------------------------------------------------------------- parquets
    clean_pq = repo / "data" / "bigearthnet_v2" / "metadata.parquet"
    dirty_pq = repo / "data" / "bigearthnet_v2" / "metadata_for_patches_with_snow_cloud_or_shadow.parquet"
    for p in (clean_pq, dirty_pq):
        if not p.exists():
            fail(f"missing parquet: {p}")
    if res["failures"]:
        return res

    import pyarrow.parquet as pq

    ct = pq.read_table(clean_pq, columns=["patch_id", "s1_name"]).to_pydict()
    dt = pq.read_table(dirty_pq, columns=["patch_id", "s1_name"]).to_pydict()
    clean = set(ct["patch_id"])
    dirty = set(dt["patch_id"])
    selected = {r["sample_id"] for r in recs}

    res["composition"] = {
        "selected_from_clean": len(selected & clean),
        "selected_from_snow_cloud": len(selected & dirty),
        "selected_in_neither": len(selected - clean - dirty),
        "parquet_clean_rows": len(clean),
        "parquet_snow_cloud_rows": len(dirty),
        "parquet_overlap": len(clean & dirty),
        "parquet_union": len(clean | dirty),
    }
    if len(selected & clean) + len(selected & dirty) != len(selected):
        fail("some selected patches are in neither parquet")
    if clean & dirty:
        fail("the two metadata parquets overlap")
    if len(clean | dirty) != 549_488:
        fail(f"parquet union {len(clean | dirty)} != 549,488")

    s1map: dict[str, str] = {}
    s1map.update(zip(ct["patch_id"], ct["s1_name"]))
    s1map.update(zip(dt["patch_id"], dt["s1_name"]))

    # ------------------------------------------------------ dirs-file equality
    def s2_dir(pid: str) -> str:
        return "BigEarthNet-S2/" + "_".join(pid.split("_")[:-2]) + "/" + pid

    def ref_dir(pid: str) -> str:
        return "Reference_Maps/" + "_".join(pid.split("_")[:-2]) + "/" + pid

    def s1_dir(name: str) -> str:
        return "BigEarthNet-S1/" + "_".join(name.split("_")[:-3]) + "/" + name

    def read_dirs(name: str) -> set[str]:
        p = repo / ".scratch" / "p12_extract" / name
        if not p.exists():
            fail(f"missing dirs file: {p}")
            return set()
        return {l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}

    a2 = read_dirs("s2_dirs.txt")
    a1 = read_dirs("s1_dirs.txt")
    ar = read_dirs("ref_dirs.txt")

    e2 = {s2_dir(p) for p in selected}
    er = {ref_dir(p) for p in selected}
    e1 = {s1_dir(s1map[p]) for p in selected if p in s1map}

    res["dirs"] = {
        "s2_lines": len(a2), "s1_lines": len(a1), "ref_lines": len(ar),
        "s2_set_equal": a2 == e2, "s1_set_equal": a1 == e1, "ref_set_equal": ar == er,
        "s2_symmetric_diff": len(a2 ^ e2), "s1_symmetric_diff": len(a1 ^ e1),
        "ref_symmetric_diff": len(ar ^ er),
    }
    for label, got in (("s2_dirs.txt", a2), ("s1_dirs.txt", a1), ("ref_dirs.txt", ar)):
        if len(got) != 28_000:
            fail(f"{label} has {len(got)} lines, want 28,000")
    if a2 != e2:
        fail("s2_dirs.txt does not reproduce the manifest")
    if a1 != e1:
        fail("s1_dirs.txt does not reproduce the manifest")
    if ar != er:
        fail("ref_dirs.txt does not reproduce the manifest")

    # ---------------------------------------------------------------- plan.json
    plan_p = repo / ".scratch" / "p12_extract" / "plan.json"
    if not plan_p.exists():
        fail(f"missing plan.json: {plan_p}")
    else:
        plan = json.loads(plan_p.read_text(encoding="utf-8"))
        res["plan"] = plan
        if plan.get("records") != 28_000:
            fail("plan.json records != 28,000")
        if plan.get("expected_s2_files") != 336_000:
            fail("plan.json expected_s2_files != 336,000")
        if plan.get("expected_s1_files") != 56_000:
            fail("plan.json expected_s1_files != 56,000")

    # --------------------------------------------------------------- environment
    res["env"] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "zstandard": None,
        "libzstd": None,
    }
    try:
        import zstandard  # noqa: PLC0415
        res["env"]["zstandard"] = zstandard.__version__
        res["env"]["libzstd"] = ".".join(str(x) for x in zstandard.ZSTD_VERSION)
    except Exception as exc:  # pragma: no cover - environment dependent
        fail(f"cannot import zstandard: {exc!r}")

    extractor = repo / ".scratch" / "p12_stream_extract.py"
    res["extractor"] = {"path": str(extractor), "exists": extractor.exists()}
    if not extractor.exists():
        fail(f"extractor missing: {extractor}")

    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 12 preflight verifier")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    res = build(Path(args.repo).resolve())
    Path(args.out_json).write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("PREFLIGHT_OK" if res["ok"] else "PREFLIGHT_FAILED")
    for f in res["failures"]:
        print("  FAIL: " + f)
    if res.get("composition"):
        c = res["composition"]
        print(f"  composition: clean={c['selected_from_clean']} "
              f"snow_cloud={c['selected_from_snow_cloud']}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
