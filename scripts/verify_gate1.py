"""Gate 1 verification — the checks that must pass before ANY training runs.

Run:
    python scripts/verify_gate1.py

Exit code 0 means training may start. Exit code 1 means it may not.
This script prints a human-readable report AND writes a machine-readable
gate1_report.json so the result can be pasted back without transcription errors.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _utc_now() -> str:
    """The run's start time, as an ISO-8601 UTC string.

    Recorded rather than derived: a caller that later wants a duration needs a
    real start, and `datetime.now(timezone.utc)` at the moment the manifest is
    built is exactly that. Left naive nowhere, so the value is unambiguous about
    its zone.
    """
    return datetime.now(timezone.utc).isoformat()

from core.config import load_config  # noqa: E402
from core.errors import LeakageError  # noqa: E402
from evaluation.leakage import (  # noqa: E402
    PublicTestFirewall,
    assert_no_hidden_access,
    assert_no_scene_overlap,
    assign_splits_by_scene,
    audit_manifest,
)
from evaluation.manifests import (  # noqa: E402
    DatasetManifest,
    RunManifest,
    SampleRecord,
    geographic_hash,
)
from evaluation.run_manifest import (  # noqa: E402
    manifest_completeness,
    populate_run_manifest,
)

OUT_PATH = REPO_ROOT / "artifacts" / "gate1_report.json"


def synthetic_scene(prefix: str, n_scenes: int, tiles_per_scene: int) -> list[SampleRecord]:
    """Build a dataset shaped like a real tiled product: scenes own adjacent tiles."""
    records: list[SampleRecord] = []
    for s in range(n_scenes):
        for t in range(tiles_per_scene):
            records.append(
                SampleRecord(
                    dataset_id="gate1_synthetic",
                    sample_id=f"{prefix}_scene{s:03d}_tile{t:02d}",
                    split="train",
                    scene_id=f"{prefix}_scene{s:03d}",
                    source_scene=f"{prefix}_SRC_{s:03d}",
                    acquisition_date="2026-02-01",
                    sensor="sentinel-2",
                    sha256=f"{prefix}{s:04d}{t:04d}".ljust(64, "a"),
                )
            )
    return records


def main() -> int:
    results: dict[str, object] = {}
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {name}")
        if detail:
            print(f"         {detail}")
        results[name] = {"ok": ok, "detail": detail}
        if not ok:
            failures.append(name)

    print("=" * 66)
    print("SATQUERY AI - GATE 1 VERIFICATION")
    print("=" * 66)
    print()

    # -- 1. configuration is reproducible ---------------------------------
    print("1. REPRODUCIBLE ENVIRONMENT")
    cfg = load_config()
    check("config loads and hashes", bool(cfg.hash), f"config hash = {cfg.hash}")
    check("seed is fixed", isinstance(cfg.seed, int), f"seed = {cfg.seed}")
    check("public test is marked immutable",
          cfg.get("evaluation.immutable_public_test") is True,
          "evaluation.immutable_public_test = True")
    print()

    # -- 2. manifest builds and round-trips -------------------------------
    print("2. DATASET MANIFEST")
    records = assign_splits_by_scene(synthetic_scene("a", 20, 5), seed=cfg.seed)
    manifest = DatasetManifest.from_records("gate1_synthetic", records)
    manifest_path = manifest.write(
        REPO_ROOT / "artifacts" / "gate1_manifest.jsonl", overwrite=True
    )
    restored = DatasetManifest.read(manifest_path)

    check("manifest builds", len(manifest) == 100, f"{len(manifest)} records")
    check("manifest round-trips byte-stable", restored.hash == manifest.hash,
          f"manifest hash = {manifest.hash[:16]}...")
    counts = manifest.split_counts()
    check("all three splits populated", all(v > 0 for v in counts.values()),
          f"splits = {counts}")
    check("scene count is honest", len(manifest.scenes()) == 20,
          f"{len(manifest.scenes())} unique scenes")
    print()

    # -- 3. leakage audit -------------------------------------------------
    print("3. SCENE-LEVEL LEAKAGE ISOLATION")
    report = audit_manifest(restored)
    check("audit reports clean", report.clean,
          f"scenes across splits = {len(report.scenes_across_splits)}")

    train = restored.by_split("train")
    test = restored.by_split("test")
    val = restored.by_split("val")
    assert_no_scene_overlap(train, test)
    check("no scene shared between train and test", True,
          f"train={len(train)} val={len(val)} test={len(test)} records")
    assert_no_scene_overlap(train, val)
    check("no scene shared between train and val", True, "")

    # Verify every scene is fully contained in one split
    scene_splits: dict[str, set[str]] = {}
    for r in restored:
        scene_splits.setdefault(r.scene_key, set()).add(r.split)
    split_sizes = {len(s) for s in scene_splits.values()}
    check("every scene belongs to exactly one split", split_sizes == {1},
          f"distinct split-counts across scenes = {sorted(split_sizes)}")
    print()

    # -- 4. the guard actually catches the classic bug --------------------
    print("4. ADVERSARIAL: RANDOM TILE SPLIT MUST BE REJECTED")
    bad = synthetic_scene("b", n_scenes=1, tiles_per_scene=8)  # one scene, 8 tiles
    for i, r in enumerate(bad):
        r.split = "test" if i % 2 else "train"  # type: ignore[assignment]
    bad_manifest = DatasetManifest.from_records("bad_random_tile_split", bad)
    try:
        audit_manifest(bad_manifest, require_all_splits=False)
        check("random tile split is rejected", False, "guard did NOT fire")
    except LeakageError as exc:
        first = str(exc).splitlines()[0]
        check("random tile split is rejected", True, f"{first}")

    # Same for 1-px-offset neighbours identified only by geographic hash
    geo_records = []
    for i in range(6):
        base = 500_000.0 + i * 100.0  # adjacent 640 m tiles, 100 m apart
        geo_records.append(
            SampleRecord(
                dataset_id="geo", sample_id=f"g{i}", split="test" if i % 2 else "train",
                geographic_hash=geographic_hash([base, 2_500_000.0, base + 640.0, 2_500_640.0]),
            )
        )
    try:
        audit_manifest(DatasetManifest.from_records("geo", geo_records),
                       require_all_splits=False)
        check("geo-hash neighbours cannot be split across train/test", False,
              "guard did NOT fire")
    except LeakageError:
        check("geo-hash neighbours cannot be split across train/test", True,
              "proximity-based leakage detected via geographic_hash")
    print()

    # -- 5. public test firewall ------------------------------------------
    print("5. PUBLIC TEST FIREWALL")
    fw = PublicTestFirewall(REPO_ROOT / "evaluation" / "public_test")
    blocked = False
    try:
        fw.check_read(REPO_ROOT / "evaluation" / "public_test" / "any.tif")
    except LeakageError:
        blocked = True
    check("training-time read of public_test is blocked", blocked, "")

    allowed = True
    try:
        fw.check_read(REPO_ROOT / "demo" / "case.tif")
    except LeakageError:
        allowed = False
    check("reads outside public_test are allowed", allowed, "")

    manifest_blocked = False
    try:
        fw.check_manifest(restored)  # contains test-split records
    except LeakageError:
        manifest_blocked = True
    check("manifest with test records cannot feed training", manifest_blocked, "")
    print()

    # -- 6. hidden-set discipline -----------------------------------------
    print("6. HIDDEN-SET DISCIPLINE")
    try:
        assert_no_hidden_access(cfg)
        check("hidden_data_access is False", True, "")
        check("no invented aggregate weights",
              cfg.get("evaluation.official_aggregate_weights") is None, "")
    except LeakageError as exc:
        check("hidden-set discipline", False, str(exc))
    print()

    # -- 7. run manifest --------------------------------------------------
    print("7. RUN MANIFEST (reproducibility record)")
    rm = RunManifest(
        run_id="gate1_verification",
        config_hash=cfg.hash,
        dataset_manifest_hash=manifest.hash,
        model_revisions={"vlm": cfg.get("vlm.revision"),
                         "router": cfg.get("router.revision")},
        thresholds={"router": cfg.get("router.confidence_threshold"),
                    "change": cfg.get("change.threshold"),
                    "grounding": cfg.get("grounding.confidence_threshold")},
        seed=cfg.seed,
    )
    # STEP 4: the four run-identifying fields used to be left at their defaults
    # (None / empty), so the manifest could not answer "which code, on which
    # machine, with which prompts". Populate them from measurement. Nothing here
    # is fabricated: an unavailable value is recorded AS unavailable.
    started_at = _utc_now()
    populate_run_manifest(rm, started_at=started_at)
    completeness = manifest_completeness(rm)

    rm_path = rm.write(REPO_ROOT / "artifacts" / "gate1_run_manifest.json")
    check("run manifest written", rm_path.exists(), f"run manifest hash = {rm.hash}")
    check("code revision identified", completeness["code_revision"] == "present",
          f"code_revision = {rm.code_revision}")
    check("environment populated", completeness["environment"] == "present",
          f"{len(rm.environment)} key(s)")
    check("prompt versions populated", completeness["prompt_versions"] == "present",
          f"prompt_versions = {rm.prompt_versions}")
    check("start time recorded", completeness["started_at"] == "present",
          f"started_at = {rm.started_at}")
    print(f"  code_revision : {rm.code_revision}")
    print(f"  started_at    : {rm.started_at}")
    print(f"  prompts       : {rm.prompt_versions}")
    print()

    # -- verdict ----------------------------------------------------------
    print("=" * 66)
    if failures:
        print(f"GATE 1: BLOCKED - {len(failures)} check(s) failed")
        for f in failures:
            print(f"  - {f}")
    else:
        print("GATE 1: PASS - training may proceed")
    print("=" * 66)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "gate": 1,
                "verdict": "BLOCKED" if failures else "PASS",
                "failures": failures,
                "config_hash": cfg.hash,
                "manifest_hash": manifest.hash,
                "run_manifest_hash": rm.hash,
                "checks": results,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\nreport written to: {OUT_PATH}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())