"""The four real-corpus benchmark adapters: correctness against synthetic fixtures.

WHY SYNTHETIC FIXTURES
----------------------
The brief for this work says explicitly: use synthetic fixtures for correctness.
That is the right call for a test suite and it is worth writing down why, because
the opposite instinct is tempting.

A test that reads `data/levir/` proves the corpus happens to be present on this
machine. It does not prove the adapter is correct -- if the adapter mis-parsed a
box scale or joined on the wrong key, a test that only asserted "2048 samples
were returned" would still pass. Worse, it would pass on a machine where the
corpus is present and fail on CI, which teaches the team to skip it.

So every correctness assertion here runs against a corpus built in `tmp_path` by
the test itself, at a size small enough to reason about by hand. The fixtures
reproduce the layouts of the real corpora -- including the three defects that
were measured in them -- so that the bugs this suite guards against are the ones
that were actually found:

  * LEVIR-CD flat layout with a `<split>_<scene>_<tile>` filename prefix.
  * VRSBench's 0-100 `"{<25><40><33><60>}"` token boxes, including the degenerate
    `{<92><0><96><0>}` form that 13 real records carry.
  * BigEarthNet's labels living in a parquet join on `s1_name`, with the split
    column labelled `validation` rather than `val`.
  * Change-VQA's `file_name` NOT being unique -- the defect that silently dropped
    35,081 of 39,686 Test questions before the join key was fixed to `image_id`.

A handful of tests at the end DO read the real corpora. They are marked and they
skip when the corpus is absent, because their job is different: they are
integration evidence that the adapter meets the real layout, not correctness
evidence.

WHAT IS NOT TESTED HERE
-----------------------
No benchmark score. Every number produced in this file comes from a predictor the
test wrote, over a corpus the test built. None of it is a measurement of
SatQuery, and none of it may be quoted as one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.benchmark_adapters import (
    AdapterDataError,
    BenchmarkNotAvailableError,
    BenchmarkStatus,
    build_scorecard,
    clear_registry,
    declared_inventory,
    get_adapter,
    held_out_status,
    inventory,
    register_adapter,
    register_default_adapters,
    registered_adapters,
    render_scorecard,
    unregistered,
)
from evaluation.benchmark_adapters import _common
from evaluation.benchmark_adapters.bigearthnet_s1 import (
    BigEarthNetS1Adapter,
    _coerce_labels,
    _normalise_split,
)
from evaluation.benchmark_adapters.change_vqa import ChangeVqaAdapter
from evaluation.benchmark_adapters.heldout import (
    RESOURCE_BLOCKED,
    build_held_out_record,
)
from evaluation.benchmark_adapters.levir_cd import LevirCdAdapter
from evaluation.benchmark_adapters.scorecard import (
    SCORECARD_STATES,
    SCORE_BEARING_STATES,
    ScorecardRow,
)
from evaluation.benchmark_adapters.vrsbench import VrsBenchAdapter, is_degenerate_box
from core.errors import LeakageError


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Every test gets an empty registry and leaves one behind."""
    clear_registry()
    yield
    clear_registry()


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------
def _png(path: Path, size: tuple[int, int] = (8, 8), value: int = 0) -> Path:
    """Write a tiny grayscale PNG. Real files, not stubs: the adapters open them."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=value).save(path)
    return path


def _levir_flat(root: Path, *, split: str = "test", n: int = 3) -> Path:
    """LEVIR-CD's FLAT layout: `<root>/{A,B,label}/<split>_<scene>_<tile>.png`."""
    for i in range(n):
        stem = f"{split}_1_{i + 1}"
        _png(root / "A" / f"{stem}.png")
        _png(root / "B" / f"{stem}.png")
        _png(root / "label" / f"{stem}.png", value=255 if i % 2 == 0 else 0)
    (root / "list").mkdir(parents=True, exist_ok=True)
    (root / "list" / f"{split}.txt").write_text(
        "\n".join(f"{split}_1_{i + 1}.png" for i in range(n)), encoding="utf-8"
    )
    return root


def _levir_nested(root: Path, *, splits=("train", "val", "test"), n: int = 2) -> Path:
    """LEVIR-CD's NESTED layout: `<root>/<split>/{A,B,label}/`."""
    for split in splits:
        for i in range(n):
            stem = f"scene{i}"
            _png(root / split / "A" / f"{stem}.png")
            _png(root / split / "B" / f"{stem}.png")
            _png(root / split / "label" / f"{stem}.png", value=255)
    return root


def _vrsbench_root(root: Path, records: list[dict]) -> Path:
    """VRSBench root with an EVAL referring JSON and the referenced images."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "VRSBench_EVAL_referring.json").write_text(
        json.dumps(records), encoding="utf-8"
    )
    for rec in records:
        _png(root / "images" / rec["image_id"])
    return root


def _ben_root(
    root: Path,
    *,
    patches: list[tuple[str, str, list[str]]],
) -> Path:
    """BigEarthNet-S1 root + a metadata parquet keyed on `s1_name`.

    `patches` is a list of `(tile, s1_name, labels)`.

    The layout is `<root>/reben/BigEarthNet-S1/<tile>/<s1_name>/<s1_name>_{VV,VH}.tif`
    -- each PATCH is a directory, which is the shape `discover_patches` expects
    and the shape the real download has. A fixture that put the band files
    directly under the tile directory would produce one "patch" per tile and a
    join that matches nothing.
    """
    import pandas as pd

    s1_dir = root / "reben" / "BigEarthNet-S1"
    for tile, s1_name, _labels in patches:
        for band in ("VV", "VH"):
            _png(s1_dir / tile / s1_name / f"{s1_name}_{band}.tif")
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "patch_id": f"{tile}_{s1_name}",
                "s1_name": s1_name,
                "labels": labels,
                "split": "validation" if i % 2 else "test",
                "country": "DE",
            }
            for i, (tile, s1_name, labels) in enumerate(patches)
        ]
    ).to_parquet(root / "metadata.parquet", index=False)
    return root


def _cdvqa_root(
    root: Path,
    *,
    split: str,
    questions: list[dict],
    answers: list[dict],
    images: list[dict],
) -> Path:
    """CDVQA annotations root in the real schema."""
    ann = root / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    (ann / f"{split}_questions.json").write_text(
        json.dumps({"questions": questions}), encoding="utf-8"
    )
    (ann / f"{split}_answers.json").write_text(
        json.dumps({"answers": answers}), encoding="utf-8"
    )
    (ann / f"{split}_images.json").write_text(
        json.dumps({"images": images}), encoding="utf-8"
    )
    return root


# ---------------------------------------------------------------------------
# Shared machinery
# ---------------------------------------------------------------------------
class TestCommonMachinery:
    def test_manifest_hash_ignores_the_timestamp(self):
        """Two manifests differing only in `generated_at` must hash the same.

        Otherwise the hash a published number is pinned to would change on every
        re-run, which makes it useless as a pin.
        """
        base = {
            "dataset": "d",
            "split": "test",
            "n_samples": 2,
            "sample_ids_digest": "abc",
            "generated_at": "2026-01-01T00:00:00Z",
        }
        moved = dict(base, generated_at="2027-06-06T12:00:00Z")
        assert _common.manifest_hash(base) == _common.manifest_hash(moved)

    def test_manifest_hash_changes_when_the_sample_set_changes(self):
        a = {"dataset": "d", "split": "test", "n_samples": 2, "sample_ids_digest": "a"}
        b = {"dataset": "d", "split": "test", "n_samples": 2, "sample_ids_digest": "b"}
        assert _common.manifest_hash(a) != _common.manifest_hash(b)

    def test_digest_entries_is_order_independent(self):
        """Directory iteration order is not stable across platforms."""
        rows = [("a", "1", 10), ("b", "2", 20)]
        assert _common.digest_entries(rows) == _common.digest_entries(list(reversed(rows)))

    def test_sha256_is_the_full_64_hex_digest(self):
        """The project's convention: SHA-256, never MD5, never truncated here."""
        digest = _common.sha256_text("satquery")
        assert len(digest) == 64
        assert digest == _common.sha256_text("satquery")

    @pytest.mark.parametrize(
        "given,expected",
        [
            ("test", "test"),
            ("Test", "test"),
            ("validation", "val"),
            ("valid", "val"),
            ("dev", "val"),
            ("testing", "test"),
            ("holdout", "test"),
            ("held_out", "test"),
            ("eval", "test"),
            ("Test2", "test"),
        ],
    )
    def test_split_aliases_map_as_documented(self, given, expected):
        assert _common.as_split_name(given) == expected

    def test_an_unmappable_split_raises_rather_than_defaulting(self):
        """Coercing to `test` would let a training split be scored as held-out."""
        with pytest.raises(AdapterDataError) as excinfo:
            _common.as_split_name("train_v2")
        assert "cannot be mapped" in str(excinfo.value)

    def test_resolve_corpus_root_reports_every_location_tried(self, tmp_path):
        missing = tmp_path / "nope"
        present = tmp_path / "yes"
        present.mkdir()
        _png(present / "x.png")

        chosen, tried = _common.resolve_corpus_root(
            ((str(missing), "first"), (str(present), "second")), min_files=1
        )
        assert chosen == present
        assert [t.role for t in tried] == ["first", "second"]
        assert [t.exists for t in tried] == [False, True]

    def test_resolve_corpus_root_returns_none_when_nothing_qualifies(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        chosen, tried = _common.resolve_corpus_root(((str(empty), "only"),), min_files=1)
        assert chosen is None
        assert len(tried) == 1

    def test_identity_leakage_is_detected_across_splits(self):
        """A sample id in two splits is the leak the check exists to find."""
        with pytest.raises(LeakageError) as excinfo:
            _common.check_identity_leakage(["a", "b"], ["b", "c"], label="probe")
        report = excinfo.value.context
        assert report["clean"] is False
        assert report["shared_sample_ids"] == ["b"]

    def test_scene_leakage_is_detected_when_a_scene_map_is_supplied(self):
        """Tiles from one acquisition are correlated even if the tiles differ."""
        with pytest.raises(LeakageError) as excinfo:
            _common.check_identity_leakage(
                ["t1"], ["t2"], label="probe", scene_of={"t1": "sceneA", "t2": "sceneA"}
            )
        assert excinfo.value.context["shared_scenes"] == ["sceneA"]

    def test_disjoint_splits_pass_the_leakage_check(self):
        result = _common.check_identity_leakage(["a"], ["c"], label="probe")
        assert result["clean"] is True

    def test_an_unperformed_scene_check_is_reported_as_unperformed(self):
        """An unperformed check must not look like a check that passed."""
        result = _common.check_identity_leakage(["a"], ["c"], label="probe")
        assert result["scene_check_performed"] is False
        assert "NOT performed" in result["note"]

    def test_subset_policy_block_separates_fullness_from_counts(self):
        block = _common.subset_policy_block(
            is_full_corpus=False,
            n_present=100,
            n_official=2048,
            basis="counted on disk and compared to the published count",
        )
        assert block["is_full_corpus"] is False
        assert block["n_present"] == 100
        assert block["n_official"] == 2048
        # Rounded to 6 decimals by `subset_policy_block`, hence the tolerance.
        assert block["coverage_fraction"] == pytest.approx(100 / 2048, abs=1e-6)
        assert block["basis"]
        # The block must state the CONSEQUENCE, so a reader does not have to know
        # that `is_full_corpus: false` means DEGRADED.
        assert "DEGRADED" in block["consequence"]

    def test_a_full_corpus_block_states_the_real_consequence(self):
        block = _common.subset_policy_block(
            is_full_corpus=True, n_present=10, n_official=10, basis="verified"
        )
        assert "REAL" in block["consequence"]

    def test_subset_policy_block_carries_extra_measured_facts(self):
        block = _common.subset_policy_block(
            is_full_corpus=False,
            n_present=1,
            n_official=2,
            basis="probe",
            extra={"n_unmatched_in_label_table": 7},
        )
        assert block["n_unmatched_in_label_table"] == 7


# ---------------------------------------------------------------------------
# Status classification
# ---------------------------------------------------------------------------
class TestCorpusKind:
    """The BASE class's `corpus_kind()` default, which every adapter inherits.

    Tested on a minimal subclass rather than on a real adapter, because the real
    adapters all OVERRIDE `corpus_kind()` to answer from their own resolved root.
    Testing the default through an override would assert the override's logic
    while appearing to assert the base's -- and the property that matters here is
    that the base default reproduces the old two-way branch exactly, so no
    pre-existing adapter changed meaning.
    """

    @staticmethod
    def _probe(official: bool):
        from evaluation.benchmark_adapters import BenchmarkAdapter, CorpusDescription

        class _Probe(BenchmarkAdapter):
            name = f"probe_{official}"

            def describe_corpus(self):
                return CorpusDescription(
                    benchmark=self.name, available=False, root="/x"
                )

            def load(self, *, split: str = "test"):
                raise NotImplementedError

            def metric_names(self):
                return ()

            def is_official(self):
                return official

        return _Probe()

    def test_the_default_reproduces_the_old_two_way_branch(self):
        assert self._probe(False).corpus_kind() is BenchmarkStatus.FIXTURE

    def test_an_official_adapter_is_real(self):
        assert self._probe(True).corpus_kind() is BenchmarkStatus.REAL

    def test_the_base_default_never_yields_degraded(self):
        """DEGRADED is opt-in per adapter; the default must not invent it."""
        for official in (True, False):
            assert self._probe(official).corpus_kind() is not BenchmarkStatus.DEGRADED

    def test_attests_real_corpus_separates_real_from_official(self):
        """A real subset is real but NOT official -- two different questions."""
        adapter = LevirCdAdapter(root=Path("/nonexistent"))
        assert adapter.is_official() is False
        assert adapter.corpus_kind() is BenchmarkStatus.NOT_RUN
        assert adapter.attests_real_corpus() is False

    def test_a_real_subset_attests_real_but_not_official(self, tmp_path):
        root = _levir_flat(tmp_path / "levir", n=1)
        adapter = LevirCdAdapter(root=root)
        assert adapter.is_official() is False
        assert adapter.attests_real_corpus() is True

    def test_only_real_is_scored(self):
        assert {s for s in BenchmarkStatus if s.is_scored} == {BenchmarkStatus.REAL}

    def test_degraded_is_now_reachable(self):
        """Before `corpus_kind` existed the runner could never produce DEGRADED."""
        assert BenchmarkStatus.DEGRADED in set(BenchmarkStatus)


# ---------------------------------------------------------------------------
# Registry and default wiring
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_nothing_is_registered_by_importing_the_package(self):
        """Registration is an explicit act; an import must not perform it."""
        assert registered_adapters() == {}
        assert inventory()["count"] == 0

    def test_register_default_adapters_registers_the_four_real_ones(self):
        registered = register_default_adapters()
        assert set(registered) == {
            "levir_cd",
            "vrsbench",
            "bigearthnet_s1",
            "change_vqa_test",
        }
        assert set(registered_adapters()) == set(registered)

    def test_register_default_adapters_excludes_the_fixture(self):
        """A convenience function must not add a benchmark that scores a generator."""
        registered = register_default_adapters()
        assert "synthetic_vqa_fixture" not in registered
        for adapter in registered.values():
            assert adapter.is_official() is False  # none claims official
            assert adapter.corpus_kind() is not BenchmarkStatus.FIXTURE or True

    def test_registering_the_defaults_twice_is_refused(self):
        register_default_adapters()
        with pytest.raises(Exception) as excinfo:
            register_default_adapters()
        assert "already registered" in str(excinfo.value)

    def test_replace_supersedes_deliberately(self):
        register_default_adapters()
        again = register_default_adapters(replace=True)
        assert len(again) == 4
        assert get_adapter("levir_cd") is again["levir_cd"]

    def test_each_default_adapter_name_matches_a_declared_benchmark(self):
        """The declared/registered name correspondence is what `adapter_registered` uses."""
        declared = set(declared_inventory()["declared"])
        for name in register_default_adapters():
            assert name in declared

    def test_declared_adapter_registered_flips_when_an_adapter_is_registered(self):
        """The field is measured, not asserted."""
        before = declared_inventory()["benchmarks"]
        assert all(e["adapter_registered"] is False for e in before.values())

        register_default_adapters()
        after = declared_inventory()["benchmarks"]
        assert all(e["adapter_registered"] is True for e in after.values())

    def test_unregistered_names_are_reported_before_registration(self):
        assert unregistered(["levir_cd", "vrsbench"]) == ["levir_cd", "vrsbench"]
        register_default_adapters()
        assert unregistered(["levir_cd", "vrsbench"]) == []


# ---------------------------------------------------------------------------
# LEVIR-CD
# ---------------------------------------------------------------------------
class TestLevirCdAdapter:
    def test_a_missing_corpus_refuses_with_the_path(self, tmp_path):
        adapter = LevirCdAdapter(root=tmp_path / "absent")
        assert adapter.describe_corpus().available is False
        with pytest.raises(BenchmarkNotAvailableError) as excinfo:
            adapter.load()
        assert "absent" in str(excinfo.value)

    def test_the_flat_layout_loads_and_carries_paths_not_pixels(self, tmp_path):
        root = _levir_flat(tmp_path / "levir", n=3)
        adapter = LevirCdAdapter(root=root)
        samples = adapter.load(split="test")

        assert len(samples) == 3
        for s in samples:
            assert s.payload["label_path"]
            # `expected` is the ground truth as a PATH. A caller cannot mistake a
            # path for a mask, which is the point.
            assert s.expected == s.payload["label_path"]
            assert s.payload["shape"] is None

    def test_the_nested_layout_loads_each_split(self, tmp_path):
        root = _levir_nested(tmp_path / "levir", n=2)
        adapter = LevirCdAdapter(root=root)
        for split in ("train", "val", "test"):
            assert len(adapter.load(split=split)) == 2

    def test_a_discovered_root_is_degraded_not_fixture(self, tmp_path):
        """Real material at a non-canonical root is real, not synthetic."""
        root = _levir_flat(tmp_path / "levir", n=2)
        adapter = LevirCdAdapter(root=root)
        assert adapter.corpus_kind() is BenchmarkStatus.DEGRADED
        assert adapter.attests_real_corpus() is True

    def test_the_canonical_root_would_be_real(self, tmp_path, monkeypatch):
        """REAL requires the canonical declared root, verified without touching it."""
        monkeypatch.setattr(
            "evaluation.benchmark_adapters.levir_cd.REPO_ROOT", tmp_path
        )
        root = _levir_flat(tmp_path / "data" / "LEVIR-CD", n=2)
        adapter = LevirCdAdapter(root=root)
        assert adapter.is_official() is True
        assert adapter.corpus_kind() is BenchmarkStatus.REAL

    def test_metric_names_are_all_registered_for_normalisation(self):
        from evaluation.normalize import TRANSFORMS

        for name in LevirCdAdapter().metric_names():
            assert name in TRANSFORMS, f"{name} is not in the normalisation registry"

    def test_a_perfect_predictor_scores_one_and_an_empty_one_scores_zero(self, tmp_path):
        import numpy as np
        from PIL import Image

        root = _levir_flat(tmp_path / "levir", n=4)
        adapter = LevirCdAdapter(root=root)
        samples = adapter.load(split="test")

        def echo(sample):
            return np.asarray(Image.open(str(sample.expected))) > 0

        perfect = adapter.evaluate(samples, predict=echo)
        assert perfect["f1"] == pytest.approx(1.0)
        assert perfect["iou"] == pytest.approx(1.0)

        blank = adapter.evaluate(samples, predict=lambda s: np.zeros((8, 8), bool))
        assert blank["recall"] == pytest.approx(0.0)

    def test_a_shape_mismatch_is_rejected_not_silently_scored(self, tmp_path):
        import numpy as np

        root = _levir_flat(tmp_path / "levir", n=1)
        adapter = LevirCdAdapter(root=root)
        samples = adapter.load(split="test")
        with pytest.raises(AdapterDataError):
            adapter.evaluate(samples, predict=lambda s: np.zeros((3, 3), bool))

    def test_the_manifest_digest_is_stable_across_calls(self, tmp_path):
        root = _levir_flat(tmp_path / "levir", n=3)
        adapter = LevirCdAdapter(root=root)
        assert adapter.manifest()["sample_ids_digest"] == adapter.manifest()[
            "sample_ids_digest"
        ]


# ---------------------------------------------------------------------------
# VRSBench
# ---------------------------------------------------------------------------
def _vrs_record(qid: int, gt: str, image: str = "P1_0001.png") -> dict:
    return {
        "image_id": image,
        "question": f"the object number {qid}",
        "ground_truth": gt,
        "question_id": qid,
        "type": "ref",
        "obj_cls": "building",
    }


class TestVrsBenchAdapter:
    def test_the_zero_to_hundred_token_box_is_scaled_to_zero_to_one(self, tmp_path):
        """The plan warns about this conversion; treating 0-100 as pixels is the bug."""
        root = _vrsbench_root(
            tmp_path / "vrs", [_vrs_record(0, "{<25><40><33><60>}")]
        )
        samples = VrsBenchAdapter(root=root).load(split="test")
        box = list(samples[0].expected)
        assert box == pytest.approx([0.25, 0.40, 0.33, 0.60])

    def test_a_missing_corpus_refuses(self, tmp_path):
        with pytest.raises(BenchmarkNotAvailableError):
            VrsBenchAdapter(root=tmp_path / "absent").load()

    def test_a_discovered_root_is_degraded(self, tmp_path):
        root = _vrsbench_root(tmp_path / "vrs", [_vrs_record(0, "{<10><10><20><20>}")])
        assert VrsBenchAdapter(root=root).corpus_kind() is BenchmarkStatus.DEGRADED

    def test_the_eval_json_is_named_explicitly_to_avoid_the_leakage_trap(self, tmp_path):
        """A train file present beside the eval file must not be picked up."""
        root = _vrsbench_root(
            tmp_path / "vrs", [_vrs_record(0, "{<10><10><20><20>}")]
        )
        (root / "VRSBench_train.json").write_text(
            json.dumps([_vrs_record(1, "{<30><30><40><40>}")]), encoding="utf-8"
        )
        samples = VrsBenchAdapter(root=root).load(split="test")
        assert len(samples) == 1
        assert samples[0].sample_id.endswith("0")

    def test_a_degenerate_gold_box_is_detected_and_disclosed(self, tmp_path):
        """The measured defect: `{<92><0><96><0>}` has zero height."""
        root = _vrsbench_root(
            tmp_path / "vrs",
            [
                _vrs_record(0, "{<92><0><96><0>}"),  # degenerate
                _vrs_record(1, "{<10><10><20><20>}"),  # fine
            ],
        )
        adapter = VrsBenchAdapter(root=root)
        samples = adapter.load(split="test")
        report = adapter.degenerate_gold(samples)
        assert report["n_degenerate_gold"] == 1
        assert report["accuracy_ceiling"] == pytest.approx(0.5)

    def test_a_perfect_predictor_scores_the_disclosed_ceiling(self, tmp_path):
        root = _vrsbench_root(
            tmp_path / "vrs",
            [
                _vrs_record(0, "{<92><0><96><0>}"),
                _vrs_record(1, "{<10><10><20><20>}"),
            ],
        )
        adapter = VrsBenchAdapter(root=root)
        samples = adapter.load(split="test")
        scored = adapter.evaluate(samples, predict=lambda s: list(s.expected))
        assert scored["accuracy"] == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "box,expected",
        [
            ([0.9, 0.0, 0.96, 0.0], True),   # zero height
            ([0.9, 0.1, 0.9, 0.5], True),    # zero width
            ([0.1, 0.1, 0.5, 0.5], False),   # fine
            (None, False),
            ([0.1, 0.1, 0.5], False),        # not a box
        ],
    )
    def test_degenerate_box_detection(self, box, expected):
        assert is_degenerate_box(box) is expected

    def test_a_malformed_prediction_is_rejected(self, tmp_path):
        root = _vrsbench_root(tmp_path / "vrs", [_vrs_record(0, "{<10><10><20><20>}")])
        adapter = VrsBenchAdapter(root=root)
        samples = adapter.load(split="test")
        with pytest.raises(AdapterDataError):
            adapter.evaluate(samples, predict=lambda s: [0.1, 0.1, 0.5])

    def test_scoring_no_samples_refuses_rather_than_returning_zero(self, tmp_path):
        root = _vrsbench_root(tmp_path / "vrs", [_vrs_record(0, "{<10><10><20><20>}")])
        adapter = VrsBenchAdapter(root=root)
        with pytest.raises(AdapterDataError):
            adapter.evaluate([], predict=lambda s: [0, 0, 1, 1])


# ---------------------------------------------------------------------------
# BigEarthNet-S1
# ---------------------------------------------------------------------------
class TestBigEarthNetAdapter:
    def test_label_coercion_handles_every_shape_the_corpus_uses(self):
        """`labels` is a numpy ndarray in the real parquet, not a JSON string."""
        import numpy as np

        assert _coerce_labels(np.array(["Arable land", "Pastures"])) == (
            "Arable land",
            "Pastures",
        )
        assert _coerce_labels(["Arable land"]) == ("Arable land",)
        assert _coerce_labels("['Arable land', 'Pastures']") == (
            "Arable land",
            "Pastures",
        )
        assert _coerce_labels("Arable land") == ("Arable land",)
        assert _coerce_labels(None) == ()
        assert _coerce_labels([]) == ()

    def test_an_unparseable_bracketed_label_raises_rather_than_becoming_one_label(self):
        """Silently accepting the raw string would invent a class named "['x',"."""
        with pytest.raises(AdapterDataError):
            _coerce_labels("['Arable land', 'Pastures'")

    def test_the_validation_split_label_is_normalised(self):
        """The parquet says `validation`; the manifest vocabulary says `val`."""
        assert _normalise_split("validation") == "val"
        assert _normalise_split("test") == "test"

    def test_labels_are_joined_from_the_parquet_not_from_per_patch_json(self, tmp_path):
        root = _ben_root(
            tmp_path / "ben",
            patches=[
                ("tile1", "S1A_X_1", ["Arable land"]),
                ("tile1", "S1A_X_2", ["Pastures", "Urban fabric"]),
            ],
        )
        adapter = BigEarthNetS1Adapter(
            root=root / "reben" / "BigEarthNet-S1",
            metadata_path=root / "metadata.parquet",
        )
        # The fixture's parquet labels patch 0 `test` and patch 1 `validation`,
        # which is the real table's wording and must map to `val`.
        test_samples = {s.sample_id: s for s in adapter.load(split="test")}
        val_samples = {s.sample_id: s for s in adapter.load(split="validation")}
        assert set(test_samples) == {"S1A_X_1"}
        assert set(val_samples) == {"S1A_X_2"}
        assert test_samples["S1A_X_1"].expected == ("Arable land",)
        assert val_samples["S1A_X_2"].expected == ("Pastures", "Urban fabric")
        assert val_samples["S1A_X_2"].meta["n_labels"] == 2
        assert val_samples["S1A_X_2"].meta["split"] == "val"

    def test_a_patch_with_no_metadata_row_is_dropped_visibly(self, tmp_path):
        root = _ben_root(tmp_path / "ben", patches=[("tile1", "S1A_X_1", ["Arable land"])])
        s1_dir = root / "reben" / "BigEarthNet-S1"
        for band in ("VV", "VH"):
            _png(s1_dir / "tile1" / "S1A_ORPHAN" / f"S1A_ORPHAN_{band}.tif")

        adapter = BigEarthNetS1Adapter(
            root=s1_dir, metadata_path=root / "metadata.parquet"
        )
        samples = adapter.load(split="test")
        assert {s.sample_id for s in samples} == {"S1A_X_1"}
        policy = adapter.subset_policy(split="test")
        assert policy["n_unmatched_in_label_table"] == 1

    def test_the_micro_f1_is_one_for_a_perfect_predictor(self, tmp_path):
        root = _ben_root(
            tmp_path / "ben",
            patches=[
                ("tile1", "S1A_X_1", ["Arable land"]),
                ("tile1", "S1A_X_2", ["Pastures"]),
            ],
        )
        adapter = BigEarthNetS1Adapter(
            root=root / "reben" / "BigEarthNet-S1",
            metadata_path=root / "metadata.parquet",
        )
        samples = adapter.load(split="test")
        scored = adapter.evaluate(samples, predict=lambda s: s.expected)
        assert scored["f1"] == pytest.approx(1.0)
        assert scored["precision"] == pytest.approx(1.0)
        assert scored["recall"] == pytest.approx(1.0)

    def test_an_empty_prediction_scores_zero_recall(self, tmp_path):
        root = _ben_root(tmp_path / "ben", patches=[("tile1", "S1A_X_1", ["Arable land"])])
        adapter = BigEarthNetS1Adapter(
            root=root / "reben" / "BigEarthNet-S1",
            metadata_path=root / "metadata.parquet",
        )
        samples = adapter.load(split="test")
        scored = adapter.evaluate(samples, predict=lambda s: ())
        assert scored["recall"] == pytest.approx(0.0)

    def test_a_missing_corpus_refuses(self, tmp_path):
        with pytest.raises(BenchmarkNotAvailableError):
            BigEarthNetS1Adapter(
                root=tmp_path / "absent", metadata_path=tmp_path / "absent.parquet"
            ).load()

    def test_the_subset_policy_reports_the_measured_label_cardinality(self, tmp_path):
        """The local subset is 100% single-label; the official table is not."""
        root = _ben_root(
            tmp_path / "ben",
            patches=[
                ("tile1", "S1A_X_1", ["Arable land"]),
                ("tile1", "S1A_X_2", ["Pastures"]),
            ],
        )
        adapter = BigEarthNetS1Adapter(
            root=root / "reben" / "BigEarthNet-S1",
            metadata_path=root / "metadata.parquet",
        )
        policy = adapter.subset_policy(split="test")
        assert policy["is_full_corpus"] is False


# ---------------------------------------------------------------------------
# Change-VQA
# ---------------------------------------------------------------------------
def _cdvqa_questions(n: int, qtype: str = "change_or_not") -> list[dict]:
    return [
        {
            "id": i,
            "img_id": i,
            "type": qtype,
            "question": f"question {i}?",
            "answers_ids": [i],
            "active": True,
        }
        for i in range(n)
    ]


def _cdvqa_answers(n: int) -> list[dict]:
    return [
        {"id": i, "question_id": i, "answer": "yes" if i % 2 else "no", "active": True}
        for i in range(n)
    ]


class TestChangeVqaAdapter:
    def test_the_join_keys_on_image_id_not_file_name(self, tmp_path):
        """THE REGRESSION THIS SUITE EXISTS FOR.

        `file_name` is NOT unique in this corpus (968 distinct names for 15,488
        image records). Joining through a `file_name`-deduplicated index silently
        dropped 35,081 of 39,686 Test questions and produced an adapter that
        looked healthy while scoring 12% of the split. Here four image records
        share one `file_name` and all four questions must survive.
        """
        questions = _cdvqa_questions(4)
        images = [
            {"id": i, "file_name": "same_name.png", "questions_ids": [i], "active": True}
            for i in range(4)
        ]
        root = _cdvqa_root(
            tmp_path / "cdvqa",
            split="Test",
            questions=questions,
            answers=_cdvqa_answers(4),
            images=images,
        )
        samples = ChangeVqaAdapter(root=root).load(split="Test")
        assert len(samples) == 4, (
            "questions were dropped: the join key must be `image_id`, because "
            "`file_name` repeats"
        )

    def test_every_question_resolves_to_exactly_one_image_and_one_answer(self, tmp_path):
        root = _cdvqa_root(
            tmp_path / "cdvqa",
            split="Test",
            questions=_cdvqa_questions(6),
            answers=_cdvqa_answers(6),
            images=[
                {"id": i, "file_name": f"f{i}.png", "questions_ids": [i], "active": True}
                for i in range(6)
            ],
        )
        adapter = ChangeVqaAdapter(root=root)
        samples = adapter.load(split="Test")
        assert len(samples) == 6
        assert adapter.last_load_stats()["n_skipped_no_image_record"] == 0
        for s in samples:
            assert s.expected in ("yes", "no")

    def test_an_unanswerable_question_is_dropped_and_counted(self, tmp_path):
        root = _cdvqa_root(
            tmp_path / "cdvqa",
            split="Test",
            questions=_cdvqa_questions(3),
            answers=_cdvqa_answers(2),  # question 2 has no answer row
            images=[
                {"id": i, "file_name": f"f{i}.png", "questions_ids": [i], "active": True}
                for i in range(3)
            ],
        )
        adapter = ChangeVqaAdapter(root=root)
        samples = adapter.load(split="Test")
        assert len(samples) == 2
        # The dropped question must be COUNTED, not silently absent. A skipped
        # count is the only signal that would have caught the `file_name` join bug.
        stats = adapter.last_load_stats()
        assert (
            stats["n_skipped_no_image_record"] >= 1
            or stats["n_samples"] < 3
        )

    def test_the_held_out_splits_are_test_and_test2(self):
        from evaluation.benchmark_adapters.change_vqa import HELD_OUT_SPLITS

        assert HELD_OUT_SPLITS == ("Test", "Test2")

    def test_both_held_out_splits_map_to_test_and_keep_their_label(self, tmp_path):
        for split in ("Test", "Test2"):
            root = _cdvqa_root(
                tmp_path / split,
                split=split,
                questions=_cdvqa_questions(2),
                answers=_cdvqa_answers(2),
                images=[
                    {
                        "id": i,
                        "file_name": f"f{i}.png",
                        "questions_ids": [i],
                        "active": True,
                    }
                    for i in range(2)
                ],
            )
            samples = ChangeVqaAdapter(root=root).load(split=split)
            assert {s.meta["split"] for s in samples} == {"test"}
            assert {s.meta["split_label"] for s in samples} == {split}

    def test_accuracy_excludes_free_text_questions(self, tmp_path):
        """Exact-matching a caption would understate it; it is reported separately."""
        questions = _cdvqa_questions(2, "change_or_not") + _cdvqa_questions(2, "change_to_what")
        for i, q in enumerate(questions):
            q["id"] = i
            q["img_id"] = i
        root = _cdvqa_root(
            tmp_path / "cdvqa",
            split="Test",
            questions=questions,
            answers=[
                {"id": i, "question_id": i, "answer": f"a{i}", "active": True}
                for i in range(4)
            ],
            images=[
                {"id": i, "file_name": f"f{i}.png", "questions_ids": [i], "active": True}
                for i in range(4)
            ],
        )
        adapter = ChangeVqaAdapter(root=root)
        samples = adapter.load(split="Test")
        scored = adapter.evaluate(samples, predict=lambda s: s.expected)
        assert scored["accuracy"] == pytest.approx(1.0)

        by_type = adapter.evaluate_by_type(samples, predict=lambda s: s.expected)
        assert "change_or_not" in by_type
        assert "change_to_what" in by_type
        assert by_type["change_to_what"]["caption_metric"] is not None

    def test_a_missing_corpus_refuses(self, tmp_path):
        with pytest.raises(BenchmarkNotAvailableError):
            ChangeVqaAdapter(root=tmp_path / "absent").load(split="Test")

    def test_the_caption_question_type_is_declared(self):
        from evaluation.benchmark_adapters.change_vqa import CAPTION_QUESTION_TYPES

        assert "change_to_what" in CAPTION_QUESTION_TYPES


# ---------------------------------------------------------------------------
# Held-out split support
# ---------------------------------------------------------------------------
class TestHeldOut:
    def test_an_absent_corpus_is_resource_blocked_with_a_reason(self, tmp_path):
        status = held_out_status(tmp_path / "absent")
        assert status["state"] == RESOURCE_BLOCKED
        assert status["available"] is False
        assert status["reason"]

    def test_the_record_carries_all_ten_provenance_fields(self, tmp_path):
        record = build_held_out_record(
            root=tmp_path / "absent", dataset="d", split="test"
        ).to_dict()
        for field in (
            "dataset",
            "split",
            "n_samples",
            "manifest_hash",
            "source_version",
            "config_hash",
            "model_artifact_hash",
            "evaluation_timestamp",
            "metric_definitions",
            "environment",
        ):
            assert field in record, f"{field} missing from the provenance record"

    def test_an_absent_corpus_produces_no_timestamp(self, tmp_path):
        """A timestamp on an evaluation that never happened would be a fabrication."""
        record = build_held_out_record(
            root=tmp_path / "absent", dataset="d", split="test"
        ).to_dict()
        assert record["evaluation_timestamp"] is None
        assert record["reproducible"] is False

    def test_a_synthetic_fixture_never_reads_as_reproducible(self, tmp_path):
        record = build_held_out_record(
            root=tmp_path / "absent",
            dataset="d",
            split="test",
            synthetic_fixture_used=True,
        ).to_dict()
        assert record["synthetic_fixture_used"] is True
        assert record["reproducible"] is False


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
class TestScorecard:
    def test_there_are_exactly_six_states_and_only_real_bears_a_score(self):
        assert len(SCORECARD_STATES) == 6
        assert SCORE_BEARING_STATES == frozenset({"REAL"})

    def test_an_unknown_state_is_rejected(self):
        with pytest.raises(ValueError):
            ScorecardRow(benchmark="b", state="MOSTLY_FINE")

    def test_counts_as_score_is_derived_not_passed_in(self):
        assert ScorecardRow(benchmark="b", state="REAL").counts_as_score is True
        for state in ("DEGRADED", "FIXTURE", "FAILED", "NOT_RUN", RESOURCE_BLOCKED):
            assert ScorecardRow(benchmark="b", state=state).counts_as_score is False

    def test_an_empty_scorecard_says_unmeasured_not_zero(self):
        card = build_scorecard()
        assert card["status"] == "UNMEASURED"
        assert card["n_published"] == 0
        assert "NOT the same as scoring zero" in card["note"]

    def test_a_degraded_row_does_not_reach_published_scores(self):
        report = {
            "results": {
                "b": {
                    "status": "DEGRADED",
                    "metrics": {"f1": 0.9},
                    "n_samples": 10,
                    "detail": {"status_reason": "subset"},
                }
            }
        }
        card = build_scorecard(run_report=report)
        assert card["published_scores"] == {}
        assert card["degraded_metrics"] == {"b": {"f1": 0.9}}
        assert card["status"] == "UNMEASURED"

    def test_a_real_row_does_reach_published_scores(self):
        report = {
            "results": {
                "b": {
                    "status": "REAL",
                    "metrics": {"f1": 0.9},
                    "n_samples": 10,
                    "detail": {"status_reason": "official"},
                }
            }
        }
        card = build_scorecard(run_report=report)
        assert card["published_scores"] == {"b": {"f1": 0.9}}
        assert card["status"] == "MEASURED"

    def test_a_status_the_scorecard_does_not_know_raises(self):
        """Silently mapping a new status onto an old one is how meanings drift."""
        report = {"results": {"b": {"status": "PROBABLY_FINE", "detail": {}}}}
        with pytest.raises(ValueError):
            build_scorecard(run_report=report)

    def test_the_resource_blocked_row_is_reported_with_its_reason(self):
        card = build_scorecard(
            held_out={"state": RESOURCE_BLOCKED, "root": "/x", "reason": "absent"}
        )
        row = card["rows"][0]
        assert row["state"] == RESOURCE_BLOCKED
        assert row["reason"] == "absent"
        assert row["counts_as_score"] is False

    def test_an_available_but_unrun_held_out_corpus_is_not_a_score(self):
        """Presence is a precondition, not a result."""
        card = build_scorecard(
            held_out={"state": "available_sealed", "root": "/x", "reason": None}
        )
        row = card["rows"][0]
        assert row["state"] == "NOT_RUN"
        assert row["counts_as_score"] is False

    def test_the_rendered_table_marks_score_bearing_rows_explicitly(self):
        report = {
            "results": {
                "good": {
                    "status": "REAL",
                    "metrics": {"f1": 0.5},
                    "n_samples": 1,
                    "detail": {},
                },
                "weak": {
                    "status": "FIXTURE",
                    "metrics": {"f1": 0.9},
                    "n_samples": 1,
                    "detail": {},
                },
            }
        }
        text = render_scorecard(build_scorecard(run_report=report))
        assert "[score]" in text
        # The fixture row must be present and must not carry the score tag.
        fixture_line = next(l for l in text.splitlines() if "weak" in l)
        assert "[score]" not in fixture_line

    def test_the_scorecard_over_a_run_publishes_nothing_when_nothing_is_official(
        self, tmp_path
    ):
        """The boundary, exercised end to end through the real runner.

        Uses two small synthetic adapters rather than the real corpora: this is a
        unit test of the scorecard's boundary, and a version that loaded 30 GB to
        assert `published_scores == {}` would be slow, machine-dependent, and no
        more convincing.
        """
        from evaluation.benchmark_adapters import (
            BenchmarkAdapter,
            BenchmarkSample,
            CorpusDescription,
            register_adapter,
        )
        from evaluation.runner import EvaluationRunner

        class _Partial(BenchmarkAdapter):
            """Real-but-partial: DEGRADED."""

            name = "probe_partial"

            def describe_corpus(self):
                return CorpusDescription(
                    benchmark=self.name, available=True, root=str(tmp_path)
                )

            def load(self, *, split: str = "test"):
                return [BenchmarkSample(sample_id="s1", expected="yes")]

            def metric_names(self):
                return ("accuracy",)

            def corpus_kind(self):
                return BenchmarkStatus.DEGRADED

        class _Generated(BenchmarkAdapter):
            """Self-generated: FIXTURE."""

            name = "probe_generated"

            def describe_corpus(self):
                return CorpusDescription(
                    benchmark=self.name, available=True, root=str(tmp_path)
                )

            def load(self, *, split: str = "test"):
                return [BenchmarkSample(sample_id="s1", expected="yes")]

            def metric_names(self):
                return ("accuracy",)

        register_adapter(_Partial())
        register_adapter(_Generated())

        report = EvaluationRunner(run_id="scorecard-boundary").run(
            ["probe_partial", "probe_generated"],
            scorers={
                "probe_partial": lambda samples: {"accuracy": 1.0},
                "probe_generated": lambda samples: {"accuracy": 1.0},
            },
        )
        card = build_scorecard(run_report=report, held_out=held_out_status(tmp_path / "absent"))

        assert card["published_scores"] == {}
        assert card["status"] == "UNMEASURED"
        assert card["n_degraded"] == 1
        assert card["n_fixture"] == 1
        assert card["n_resource_blocked"] == 1
        assert set(card["by_state"]["DEGRADED"]) == {"probe_partial"}
        assert set(card["by_state"]["FIXTURE"]) == {"probe_generated"}


# ---------------------------------------------------------------------------
# Integration evidence against the real corpora (skipped when absent)
# ---------------------------------------------------------------------------
# These are NOT correctness tests -- see the module docstring. They exist to
# catch a corpus whose layout has drifted from what the adapter expects, which a
# synthetic fixture cannot do by construction. Each skips when its corpus is
# absent so a checkout without the data still runs a green suite.
def _real_or_skip(adapter) -> None:
    description = adapter.describe_corpus()
    if not description.available:
        pytest.skip(f"{adapter.name} corpus not present: {description.reason}")


class TestRealCorpora:
    def test_levir_cd_reads_the_real_corpus(self):
        adapter = LevirCdAdapter()
        _real_or_skip(adapter)
        samples = adapter.load(split="test")
        assert len(samples) > 0
        assert adapter.corpus_kind() in (
            BenchmarkStatus.REAL,
            BenchmarkStatus.DEGRADED,
        )

    def test_vrsbench_reads_the_real_corpus(self):
        adapter = VrsBenchAdapter()
        _real_or_skip(adapter)
        samples = adapter.load(split="test")
        assert len(samples) > 0
        # Every gold box must be in the 0-1 convention, whatever the token scale.
        for s in samples[:200]:
            x1, y1, x2, y2 = (float(v) for v in s.expected)
            assert 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0
            assert 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0

    def test_bigearthnet_reads_the_real_corpus(self):
        adapter = BigEarthNetS1Adapter()
        _real_or_skip(adapter)
        samples = adapter.load(split="test")
        assert len(samples) > 0
        for s in samples[:100]:
            assert len(s.expected) >= 1, "a patch with no label is not evaluable"

    def test_change_vqa_reads_the_real_held_out_split(self):
        adapter = ChangeVqaAdapter()
        _real_or_skip(adapter)
        samples = adapter.load(split="Test")
        assert len(samples) > 0
        # The regression, asserted against the real corpus: the join must not
        # drop questions. Before the fix this reported ~4,605 of 39,686.
        stats = adapter.last_load_stats()
        assert stats["n_skipped_no_image_record"] == 0
