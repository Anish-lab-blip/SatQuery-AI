"""Phase 3 tests — Gate 1: dataset manifests and scene-level leakage isolation.

No training may start until this file is green. These tests deliberately include the
classic remote-sensing failure mode (adjacent tiles split randomly) so that the guard
is proven to catch it, not merely asserted to exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.errors import LeakageError
from evaluation.leakage import (
    PublicTestFirewall,
    assert_no_hidden_access,
    assert_no_scene_overlap,
    assign_splits_by_scene,
    audit_manifest,
    deduplicate,
)
from evaluation.manifests import (
    DatasetManifest,
    ManifestError,
    RunManifest,
    SampleRecord,
    geographic_hash,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_records(
    n_scenes: int = 10, tiles_per_scene: int = 4, dataset_id: str = "synthetic"
) -> list[SampleRecord]:
    """Build records where each scene owns several adjacent tiles."""
    out: list[SampleRecord] = []
    for s in range(n_scenes):
        for t in range(tiles_per_scene):
            out.append(
                SampleRecord(
                    dataset_id=dataset_id,
                    sample_id=f"scene{s:03d}_tile{t:02d}",
                    split="train",  # placeholder; assign_splits_by_scene overwrites
                    scene_id=f"scene{s:03d}",
                    source_scene=f"SRC_{s:03d}",
                    acquisition_date="2026-01-15",
                    sensor="sentinel-2",
                    sha256=f"{s:04d}{t:04d}".ljust(64, "0"),
                )
            )
    return out


def georeferenced_records(n: int = 6) -> list[SampleRecord]:
    """Records whose scene identity must be recovered from bounds, not scene_id."""
    out: list[SampleRecord] = []
    for i in range(n):
        # 500 m apart within a scene, scenes far apart
        base_x = (i // 3) * 500_000.0
        out.append(
            SampleRecord(
                dataset_id="geo",
                sample_id=f"tile_{i}",
                split="train",
                geographic_hash=geographic_hash(
                    [base_x + (i % 3) * 500.0, 2_500_000.0, base_x + 640.0, 2_500_640.0]
                ),
            )
        )
    return out


# ---------------------------------------------------------------------------
# geographic_hash — the anti-leakage primitive
# ---------------------------------------------------------------------------


def test_geographic_hash_groups_nearby_tiles() -> None:
    a = geographic_hash([500_000.0, 2_500_000.0, 500_640.0, 2_500_640.0])
    b = geographic_hash([500_100.0, 2_500_100.0, 500_740.0, 2_500_740.0])
    assert a == b, "tiles within the same kilometre bucket must share a hash"


def test_geographic_hash_separates_distant_tiles() -> None:
    a = geographic_hash([500_000.0, 2_500_000.0, 500_640.0, 2_500_640.0])
    b = geographic_hash([600_000.0, 2_600_000.0, 600_640.0, 2_600_640.0])
    assert a != b


def test_geographic_hash_none_when_bounds_absent() -> None:
    assert geographic_hash(None) is None


def test_geographic_hash_rejects_bad_bucket() -> None:
    with pytest.raises(ManifestError):
        geographic_hash([0, 0, 1, 1], bucket=0)


# ---------------------------------------------------------------------------
# SampleRecord
# ---------------------------------------------------------------------------


def test_sample_record_requires_identity() -> None:
    with pytest.raises(ManifestError):
        SampleRecord(dataset_id="", sample_id="x", split="train")
    with pytest.raises(ManifestError):
        SampleRecord(dataset_id="d", sample_id="", split="train")


def test_sample_record_rejects_bad_split() -> None:
    with pytest.raises(ManifestError):
        SampleRecord(dataset_id="d", sample_id="s", split="validation")  # type: ignore[arg-type]


def test_scene_key_prefers_explicit_scene_id() -> None:
    r = SampleRecord(dataset_id="d", sample_id="s", split="train",
                     scene_id="sc", geographic_hash="gh", source_scene="src")
    assert r.scene_key == "sc"


def test_scene_key_falls_back_to_geo_hash_then_source() -> None:
    r = SampleRecord(dataset_id="d", sample_id="s", split="train",
                     geographic_hash="gh", source_scene="src")
    assert r.scene_key == "gh"

    r2 = SampleRecord(dataset_id="d", sample_id="s", split="train",
                      source_scene="src")
    assert r2.scene_key == "src"


def test_scene_key_last_resort_is_sample_id() -> None:
    r = SampleRecord(dataset_id="d", sample_id="s", split="train")
    assert r.scene_key == "s"


def test_sample_record_roundtrips_unknown_fields_into_extra() -> None:
    payload = {
        "dataset_id": "d", "sample_id": "s", "split": "train",
        "benchmark_label": "harbor",
    }
    r = SampleRecord.from_dict(dict(payload))
    assert r.extra.get("benchmark_label") == "harbor"
    assert r.to_dict()["dataset_id"] == "d"


# ---------------------------------------------------------------------------
# Scene-level splitting
# ---------------------------------------------------------------------------


def test_splits_are_assigned_per_scene_not_per_sample() -> None:
    records = make_records(n_scenes=10, tiles_per_scene=4)
    assigned = assign_splits_by_scene(records)

    scene_splits: dict[str, set[str]] = {}
    for r in assigned:
        scene_splits.setdefault(r.scene_key, set()).add(r.split)

    for scene, splits in scene_splits.items():
        assert len(splits) == 1, f"scene {scene} leaked across splits {splits}"


def test_splitting_is_deterministic_for_a_seed() -> None:
    a = assign_splits_by_scene(make_records(), seed=42)
    b = assign_splits_by_scene(make_records(), seed=42)
    assert [r.split for r in a] == [r.split for r in b]


def test_splitting_differs_across_seeds() -> None:
    """Different seeds must partition SCENES differently.

    Note: comparing flattened [r.split] lists is insensitive to the seed, because
    the label pattern is always n_train*T x 'train', then 'val', then 'test'. The
    meaningful comparison is the per-scene assignment.
    """
    def assignments(seed: int) -> dict[str, str]:
        return {
            r.scene_key: r.split
            for r in assign_splits_by_scene(make_records(n_scenes=30), seed=seed)
        }

    a = assignments(1)
    b = assignments(999)
    assert a != b, "scene-to-split assignment must depend on the seed"
    # but the split sizes must still match the requested ratios
    for mapping in (a, b):
        counts: dict[str, int] = {}
        for split in mapping.values():
            counts[split] = counts.get(split, 0) + 1
        assert counts == {"train": 24, "val": 3, "test": 3}


def test_split_assignment_is_seed_stable() -> None:
    """The same seed must always produce the same per-scene assignment."""
    def assignments(seed: int) -> dict[str, str]:
        return {
            r.scene_key: r.split
            for r in assign_splits_by_scene(make_records(n_scenes=30), seed=seed)
        }

    assert assignments(7) == assignments(7)


def test_output_order_is_independent_of_seed() -> None:
    """The seed picks splits; it must not reorder the returned records."""
    a = [r.sample_id for r in assign_splits_by_scene(make_records(n_scenes=20), seed=1)]
    b = [r.sample_id for r in assign_splits_by_scene(make_records(n_scenes=20), seed=999)]
    assert a == b


def test_splitting_is_independent_of_input_order() -> None:
    base = make_records(n_scenes=20)
    shuffled = list(reversed(base))
    a = {r.sample_id: r.split for r in assign_splits_by_scene(base, seed=7)}
    b = {r.sample_id: r.split for r in assign_splits_by_scene(shuffled, seed=7)}
    assert a == b, "scene assignment must not depend on record order"


def test_all_three_splits_are_populated() -> None:
    counts = {}
    for r in assign_splits_by_scene(make_records(n_scenes=20)):
        counts[r.split] = counts.get(r.split, 0) + 1
    assert set(counts) == {"train", "val", "test"}


def test_splitting_rejects_bad_ratios() -> None:
    records = make_records()
    with pytest.raises(LeakageError):
        assign_splits_by_scene(records, train_ratio=0.0)
    with pytest.raises(LeakageError):
        assign_splits_by_scene(records, train_ratio=0.7, val_ratio=0.5)


def test_splitting_empty_input_is_empty() -> None:
    assert assign_splits_by_scene([]) == []


def test_geographic_hash_recovers_scenes_without_scene_ids() -> None:
    """Two tiles 500 m apart must land in the same split even with no scene_id."""
    assigned = assign_splits_by_scene(georeferenced_records(n=6), seed=3)
    by_hash: dict[str, set[str]] = {}
    for r in assigned:
        by_hash.setdefault(r.scene_key, set()).add(r.split)
    for key, splits in by_hash.items():
        assert len(splits) == 1, f"geo-hash scene {key} leaked across {splits}"


# ---------------------------------------------------------------------------
# The classic failure mode — prove the guard catches it
# ---------------------------------------------------------------------------


def test_random_tile_split_is_detected_as_leakage() -> None:
    """Simulate the bug this whole module exists to prevent.

    Take one scene, slice it into 8 tiles, and split the TILES randomly. The audit
    must reject it.
    """
    records = make_records(n_scenes=1, tiles_per_scene=8)
    for i, r in enumerate(records):
        r.split = "test" if i % 2 else "train"  # type: ignore[assignment]

    manifest = DatasetManifest.from_records("bad_split", records)
    with pytest.raises(LeakageError):
        audit_manifest(manifest, require_all_splits=False)


def test_scene_disjoint_split_passes_audit() -> None:
    manifest = DatasetManifest.from_records(
        "good_split", assign_splits_by_scene(make_records(n_scenes=12))
    )
    report = audit_manifest(manifest)
    assert report.clean is True
    assert not report.scenes_across_splits


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def test_duplicate_hashes_are_detected() -> None:
    records = make_records(n_scenes=2, tiles_per_scene=2)
    records[1].sha256 = records[0].sha256  # same content, different sample_id
    # Force them into one scene so only the duplicate rule can fire
    for r in records:
        r.scene_id = "one"
        r.split = "train"  # type: ignore[assignment]

    manifest = DatasetManifest.from_records("dupes", records)
    with pytest.raises(LeakageError):
        audit_manifest(manifest, require_all_splits=False)


def test_deduplicate_removes_repeats() -> None:
    a = SampleRecord(dataset_id="d", sample_id="a", split="train", sha256="x" * 64)
    b = SampleRecord(dataset_id="d", sample_id="b", split="train", sha256="x" * 64)
    c = SampleRecord(dataset_id="d", sample_id="c", split="train", sha256="y" * 64)

    kept, dropped = deduplicate([a, b, c])
    assert [r.sample_id for r in kept] == ["a", "c"]
    assert dropped == ["b"]


def test_deduplicate_keeps_records_without_hashes() -> None:
    a = SampleRecord(dataset_id="d", sample_id="a", split="train")
    kept, dropped = deduplicate([a])
    assert len(kept) == 1 and dropped == []


# ---------------------------------------------------------------------------
# Overlap assertion
# ---------------------------------------------------------------------------


def test_assert_no_scene_overlap_catches_shared_scene() -> None:
    train = make_records(n_scenes=2)
    test = make_records(n_scenes=2)
    with pytest.raises(LeakageError):
        assert_no_scene_overlap(train, test)


def test_assert_no_scene_overlap_passes_for_disjoint() -> None:
    train = make_records(n_scenes=2)
    test = [r for r in make_records(n_scenes=4) if r.scene_id == "scene003"]
    assert_no_scene_overlap(train, test)


# ---------------------------------------------------------------------------
# Manifest container
# ---------------------------------------------------------------------------


def test_manifest_hash_is_order_independent() -> None:
    records = assign_splits_by_scene(make_records(n_scenes=8))
    a = DatasetManifest.from_records("m", records)
    b = DatasetManifest.from_records("m", list(reversed(records)))
    assert a.hash == b.hash


def test_manifest_hash_changes_when_content_changes() -> None:
    records = assign_splits_by_scene(make_records(n_scenes=8))
    a = DatasetManifest.from_records("m", records)
    records[0].sensor = "risat"
    b = DatasetManifest.from_records("m", records)
    assert a.hash != b.hash


def test_manifest_owns_its_records() -> None:
    """Mutating the caller's records must not retroactively change a manifest hash."""
    records = assign_splits_by_scene(make_records(n_scenes=4))
    manifest = DatasetManifest.from_records("m", records)
    before = manifest.hash

    records[0].sensor = "tampered_after_construction"
    records[0].sha256 = "f" * 64
    records.clear()

    assert manifest.hash == before, "manifest must not share state with its caller"
    assert len(manifest) == len(assign_splits_by_scene(make_records(n_scenes=4)))


def test_manifest_hash_is_stable_across_repeated_reads() -> None:
    manifest = DatasetManifest.from_records("m", make_records(n_scenes=5))
    assert manifest.hash == manifest.hash


def test_manifest_hash_is_sha256_length() -> None:
    m = DatasetManifest.from_records("m", make_records())
    assert len(m.hash) == 64


def test_manifest_counts_and_queries() -> None:
    m = DatasetManifest.from_records("m", assign_splits_by_scene(make_records(n_scenes=10)))
    assert len(m) == 40
    assert sum(m.split_counts().values()) == 40
    assert m.dataset_ids() == {"synthetic"}
    assert len(m.by_split("train")) > 0


def test_manifest_jsonl_roundtrip(tmp_path: Path) -> None:
    original = DatasetManifest.from_records(
        "roundtrip", assign_splits_by_scene(make_records(n_scenes=9))
    )
    path = original.write(tmp_path / "manifest.jsonl")
    restored = DatasetManifest.read(path)

    assert restored.hash == original.hash
    assert len(restored) == len(original)
    assert restored.name == "roundtrip"


def test_manifest_write_refuses_to_clobber(tmp_path: Path) -> None:
    m = DatasetManifest.from_records("m", make_records(n_scenes=3))
    path = m.write(tmp_path / "m.jsonl")
    with pytest.raises(ManifestError):
        m.write(path)
    m.write(path, overwrite=True)  # explicit rebuild is allowed


def test_manifest_read_detects_corruption(tmp_path: Path) -> None:
    m = DatasetManifest.from_records("m", assign_splits_by_scene(make_records(n_scenes=6)))
    path = m.write(tmp_path / "m.jsonl")

    # Tamper: flip a sensor value, leaving the header hash stale.
    lines = path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[1])
    payload["sensor"] = "tampered"
    lines[1] = json.dumps(payload, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="hash mismatch"):
        DatasetManifest.read(path)


def test_manifest_read_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ManifestError):
        DatasetManifest.read(tmp_path / "nope.jsonl")


# ---------------------------------------------------------------------------
# Public-test firewall
# ---------------------------------------------------------------------------


def test_firewall_blocks_reads_under_test_root(tmp_path: Path) -> None:
    root = tmp_path / "evaluation" / "public_test"
    root.mkdir(parents=True)
    sample = root / "case_001.tif"
    sample.write_bytes(b"x")

    fw = PublicTestFirewall(root)
    assert fw.armed
    with pytest.raises(LeakageError):
        fw.check_read(sample)


def test_firewall_allows_reads_outside_test_root(tmp_path: Path) -> None:
    root = tmp_path / "evaluation" / "public_test"
    root.mkdir(parents=True)
    other = tmp_path / "demo" / "case.tif"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"x")

    PublicTestFirewall(root).check_read(other)  # must not raise


def test_firewall_can_be_disarmed_for_the_evaluator(tmp_path: Path) -> None:
    root = tmp_path / "public_test"
    root.mkdir()
    sample = root / "c.tif"
    sample.write_bytes(b"x")

    fw = PublicTestFirewall(root)
    fw.disarm()
    fw.check_read(sample)  # evaluation mode
    fw.arm()
    with pytest.raises(LeakageError):
        fw.check_read(sample)


def test_firewall_blocks_manifests_containing_test_records(tmp_path: Path) -> None:
    records = assign_splits_by_scene(make_records(n_scenes=12))
    manifest = DatasetManifest.from_records("m", records)
    assert manifest.by_split("test"), "fixture should produce test records"

    fw = PublicTestFirewall(tmp_path / "public_test")
    with pytest.raises(LeakageError):
        fw.check_manifest(manifest)


def test_firewall_allows_train_only_manifest(tmp_path: Path) -> None:
    records = [r for r in assign_splits_by_scene(make_records(n_scenes=10))
               if r.split == "train"]
    manifest = DatasetManifest.from_records("train_only", records)
    fw = PublicTestFirewall(tmp_path / "public_test")
    fw.check_manifest(manifest)  # must not raise


# ---------------------------------------------------------------------------
# Hidden-set discipline
# ---------------------------------------------------------------------------


def test_hidden_access_guard_accepts_compliant_config() -> None:
    from core.config import load_config

    assert_no_hidden_access(load_config())  # must not raise


def test_hidden_access_guard_rejects_enabled_hidden_data() -> None:
    from core.config import load_config

    cfg = load_config(overrides={"evaluation": {"hidden_data_access": True}})
    with pytest.raises(LeakageError):
        assert_no_hidden_access(cfg)


def test_hidden_access_guard_rejects_invented_weights() -> None:
    from core.config import load_config

    cfg = load_config(
        overrides={"evaluation": {"official_aggregate_weights": {"vqa": 0.5}}}
    )
    with pytest.raises(LeakageError):
        assert_no_hidden_access(cfg)


# ---------------------------------------------------------------------------
# RunManifest — reproducibility record
# ---------------------------------------------------------------------------


def test_run_manifest_hash_is_stable_and_content_sensitive() -> None:
    a = RunManifest(run_id="r1", config_hash="abc", dataset_manifest_hash="def")
    b = RunManifest(run_id="r1", config_hash="abc", dataset_manifest_hash="def")
    c = RunManifest(run_id="r1", config_hash="abc", dataset_manifest_hash="CHANGED")

    assert a.hash == b.hash
    assert a.hash != c.hash


def test_run_manifest_writes_json(tmp_path: Path) -> None:
    rm = RunManifest(
        run_id="run_test",
        config_hash="cfg123",
        dataset_manifest_hash="ds456",
        model_revisions={"vlm": "main", "grounding": "main"},
        thresholds={"router": 0.7, "change": 0.5},
        seed=42,
    )
    path = rm.write(tmp_path / "run_manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["run_manifest"]["run_id"] == "run_test"
    assert payload["run_manifest"]["thresholds"]["change"] == 0.5
    assert payload["hash"] == rm.hash


# ---------------------------------------------------------------------------
# Gate 1 — end-to-end
# ---------------------------------------------------------------------------


def test_gate1_end_to_end(tmp_path: Path) -> None:
    """The exact checks Gate 1 requires before any training run."""
    from core.config import load_config

    # 1. environment/config is reproducible
    cfg = load_config()
    assert cfg.hash

    # 2. manifest builds and round-trips
    records = assign_splits_by_scene(make_records(n_scenes=20, tiles_per_scene=5))
    manifest = DatasetManifest.from_records("gate1", records)
    path = manifest.write(tmp_path / "gate1.jsonl")
    restored = DatasetManifest.read(path)
    assert restored.hash == manifest.hash

    # 3. scene-level leakage audit passes
    report = audit_manifest(restored)
    assert report.clean

    # 4. no scene crosses the train/test boundary
    assert_no_scene_overlap(restored.by_split("train"), restored.by_split("test"))

    # 5. hidden-data discipline is intact
    assert_no_hidden_access(cfg)

    # 6. a run manifest can be frozen
    rm = RunManifest(
        run_id="gate1",
        config_hash=cfg.hash,
        dataset_manifest_hash=manifest.hash,
        seed=cfg.seed,
    )
    assert rm.write(tmp_path / "run_manifest.json").exists()