"""Tests for the pre-registered 11.5 metric tool.

These tests pin the tool's REFUSALS, which are the part that matters. The metric
itself is a number in an artifact; what must be tested is that the tool cannot
silently produce a number that is not the pre-registered metric:

  * a cache built under a different `config_hash` is refused (a different
    experiment's numbers would not be the pre-registered metric);
  * a missing head or cache is refused rather than reported as a score;
  * an empty split is refused rather than scored as 0.0.

Two of these are load-bearing in a way worth stating: if the `config_hash` guard
were removed, `test_wrong_config_hash_is_refused` would fail by returning a
number instead of raising. If the empty-split guard were removed, the same test
would fail by reporting `accuracy = 0.0`, which reads like a bad model rather
than a broken input -- the exact confusion the guard exists to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.eval_fusion_115 import (  # noqa: E402
    EXPECTED_CONFIG_HASH,
    NUM_CLASSES,
    TOOL_NAME,
    Fusion115Error,
    evaluate_115,
    main,
)


def _write_cache(
    directory: Path,
    split: str,
    *,
    n: int = 8,
    config_hash: str = EXPECTED_CONFIG_HASH,
    arm: str = "A",
    labels: list[int] | None = None,
) -> Path:
    """Write a minimal but structurally real feature cache + sidecar.

    The widths are the frozen ones, not arbitrary: three 768-wide GAPs plus a
    12-wide optical mask and a 2-wide SAR mask total 2318, which is what
    `expected_fusion_dim()` enforces. Getting these wrong raises inside
    `read_feature_cache`, so the fixture has to be right for the test to mean
    anything.
    """
    from specialists.optical_sar.fusion_head import expected_fusion_dim

    rng = np.random.default_rng(0)
    assert 3 * 768 + 12 + 2 == expected_fusion_dim(), "fixture drift"
    label_array = (
        np.zeros((n,), dtype=np.int64)
        if labels is None
        else np.asarray(labels, dtype=np.int64)
    )
    assert label_array.shape == (n,), "label list must have length n"
    np.savez(
        directory / f"{split}.npz",
        optical_gap=rng.standard_normal((n, 768)).astype(np.float32),
        sar_gap=rng.standard_normal((n, 768)).astype(np.float32),
        joint_gap=rng.standard_normal((n, 768)).astype(np.float32),
        optical_mask=np.ones((n, 12), dtype=np.float32),
        sar_mask=np.ones((n, 2), dtype=np.float32),
        label_index=label_array,
    )
    (directory / f"{split}.json").write_text(
        json.dumps(
            {
                "n": n,
                "sample_ids": [f"s{i}" for i in range(n)],
                "scene_ids": [f"scene{i % 2}" for i in range(n)],
                "metadata": {"arm": arm, "config_hash": config_hash},
            }
        ),
        encoding="utf-8",
    )
    return directory / f"{split}.npz"


def _write_head(path: Path) -> Path:
    """Write a real, buildable fusion head checkpoint."""
    import torch

    from specialists.optical_sar.fusion_head import build_fusion_head

    head = build_fusion_head(device="cpu", input_dim=2318, hidden_dim=512,
                            task_dim=NUM_CLASSES, dropout=0.0)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "head_config": {
                "input_dim": 2318,
                "hidden_dim": 512,
                "task_dim": NUM_CLASSES,
                "dropout": 0.0,
            },
        },
        path,
    )
    return path


def test_a_well_formed_input_produces_the_metric(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_cache(cache, "test")
    head = _write_head(tmp_path / "head.pt")

    report = evaluate_115(head, cache, "test")

    assert report["tool"] == TOOL_NAME
    assert report["metric"] == "pre_registered_11.5"
    assert report["split"] == "test"
    assert report["n_scored"] == 8
    assert report["num_classes"] == NUM_CLASSES
    assert 0.0 <= report["accuracy"] <= 1.0
    assert 0.0 <= report["macro_f1"] <= 1.0
    # The tool must never present itself as deciding anything.
    assert report["is_deciding_statistic"] is False
    # It must be reproducible from the artifact: the head is hashed, not trusted.
    assert len(report["head_sha256"]) == 64
    assert report["head_bytes"] == head.stat().st_size


def test_wrong_config_hash_is_refused(tmp_path: Path) -> None:
    """A cache from a different registry is a different experiment."""
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_cache(cache, "test", config_hash="deadbeefdeadbeef")
    head = _write_head(tmp_path / "head.pt")

    with pytest.raises(Fusion115Error, match="config_hash"):
        evaluate_115(head, cache, "test")


def test_missing_head_is_refused(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_cache(cache, "test")

    with pytest.raises(Fusion115Error, match="head does not exist"):
        evaluate_115(tmp_path / "absent.pt", cache, "test")


def test_missing_cache_is_refused(tmp_path: Path) -> None:
    head = _write_head(tmp_path / "head.pt")

    with pytest.raises(Fusion115Error, match="feature cache does not exist"):
        evaluate_115(head, tmp_path / "no-such-dir", "test")


def test_empty_split_is_refused_not_scored_as_zero(tmp_path: Path) -> None:
    """An empty split must raise, never report `accuracy = 0.0`.

    Reporting 0.0 would be indistinguishable from a model that predicts nothing
    correctly -- a broken input masquerading as a bad result.

    Falsification note (measured): removing THIS tool's guard does not produce a
    wrong number, because `read_feature_cache`/`evaluate_fusion_head` reject an
    empty split underneath. So this test pins the *error message and layer*, not
    the only line of defence -- `Fusion115Error` with "empty" is what the caller
    receives, and `FusionTrainingError` is what they would get instead. Recorded
    because "the test failed when I removed the guard" and "the guard is the only
    thing preventing a wrong number" are different claims.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_cache(cache, "test", n=0)
    head = _write_head(tmp_path / "head.pt")

    with pytest.raises(Fusion115Error, match="empty"):
        evaluate_115(head, cache, "test")


def test_cli_returns_2_on_refusal_and_prints_no_metric(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The exit code and the absence of a printed metric are the contract."""
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_cache(cache, "test", config_hash="deadbeefdeadbeef")
    head = _write_head(tmp_path / "head.pt")

    code = main(["--head", str(head), "--cache-dir", str(cache), "--split", "test"])

    assert code == 2
    captured = capsys.readouterr()
    assert "ACCURACY" not in captured.out
    assert "ERROR" in captured.err


def test_cli_writes_the_report_and_returns_0(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_cache(cache, "test")
    head = _write_head(tmp_path / "head.pt")
    out = tmp_path / "report.json"

    code = main(
        [
            "--head", str(head),
            "--cache-dir", str(cache),
            "--split", "test",
            "--report", str(out),
        ]
    )

    assert code == 0
    assert out.exists()
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["n_scored"] == 8
    assert written["cache_config_hash"] == EXPECTED_CONFIG_HASH


def test_the_tool_never_reads_the_validation_split(tmp_path: Path) -> None:
    """Scoring `test` must not require a `val` cache to exist at all.

    This is the separation the trainer's `PLUMBING_ONLY` flag records: the
    held-out split is evaluated in isolation, by a step that has no val access.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_cache(cache, "test")
    # Deliberately no val.npz / train.npz.
    head = _write_head(tmp_path / "head.pt")

    report = evaluate_115(head, cache, "test")

    assert report["split"] == "test"
    assert not (cache / "val.npz").exists()


def test_the_tool_reports_class_presence_alongside_macro_f1(
    tmp_path: Path,
) -> None:
    """The macro-F1 denominator must be stated, not left to the reader.

    `macro_f1` is a mean over ALL `num_classes` slots. A class with no examples
    scores 0.0 and lowers the mean. That is the pre-registered definition
    ("macro-F1 over the 19-class label space"), but a reader who expects a
    present-classes mean will compute a larger number from the same per-class
    scores and may conclude the recorded figure is wrong. On the frozen test
    split the two are 0.434161 (over 19) and 0.589218 (over the 14 present).

    This test covers the reporting half: the tool must name the denominator and
    list which classes were present. The arithmetic half is pinned separately in
    `test_macro_f1_averages_over_all_slots_not_present_ones`, because a randomly
    initialised head cannot produce a discriminating case end-to-end.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    # Only 4 distinct classes appear, so 15 slots are absent.
    _write_cache(cache, "test", n=8, labels=[0, 0, 0, 0, 1, 1, 2, 3])
    head = _write_head(tmp_path / "head.pt")

    report = evaluate_115(head, cache, "test")

    assert report["num_classes"] == NUM_CLASSES
    assert report["macro_f1_denominator"] == (
        "all 19 classes (absent classes contribute 0.0)"
    )
    assert report["classes_present"] == (0, 1, 2, 3)
    assert report["classes_absent"] == (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                        15, 16, 17, 18)
    # One term per slot, absent slots included.
    per_class = list(report["_per_class_f1"])
    assert len(per_class) == NUM_CLASSES
    for c in report["classes_absent"]:
        assert per_class[c] == 0.0
    # The scalar must be the mean of those 19 terms.
    assert report["macro_f1"] == pytest.approx(sum(per_class) / NUM_CLASSES)


def test_macro_f1_averages_over_all_slots_not_present_ones() -> None:
    """Pin the arithmetic on controlled predictions.

    An end-to-end run uses a randomly initialised head, which predicts a single
    class for every row and yields all-zero per-class F1 — the two definitions
    then coincide and no assertion can separate them. Driving `_macro_f1`
    directly with hand-built predictions is the only way to make the test
    genuinely falsifiable, so this is where the arithmetic is pinned.
    """
    from training.fusion.train import _macro_f1, _per_class_f1

    # Perfect on classes 0 and 1; classes 2 and 3 never predicted and never
    # labelled. Over 19 slots the mean is (1 + 1) / 19.
    preds = np.asarray([0, 0, 1, 1], dtype=np.int64)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    per_class = _per_class_f1(preds, labels, 19)
    over_all = sum(per_class) / 19
    over_present = sum(per_class[c] for c in (0, 1)) / 2

    assert over_present == pytest.approx(1.0)
    assert over_all == pytest.approx(2.0 / 19)
    assert _macro_f1(preds, labels, 19) == pytest.approx(2.0 / 19)
    # The two must differ, or this case cannot distinguish them.
    assert over_all != pytest.approx(over_present)
    # And the function must follow `num_classes`, not the present set.
    assert _macro_f1(preds, labels, 2) == pytest.approx(1.0)


def test_a_zero_scored_class_is_a_real_miss_not_an_absent_class() -> None:
    """A `0.0` in the per-class vector has two possible causes. Separate them.

    This guards a documentation error made and corrected in
    `docs/PHASE12_115_METRIC_COMPUTED.md` §3.5. The per-class vector contains
    seven `0.0` entries; an earlier revision attributed *all* of them to classes
    absent from the split and concluded the low macro-F1 was "a statement about
    the evaluation population, not broad head weakness". That was wrong: two of
    the seven fell on classes that WERE present, i.e. genuine total misses.

    The distinction is decidable from the data alone, which is the point: a class
    that is absent never appears as a label, so it cannot be present. Any `0.0`
    whose class DOES appear as a label is a real miss.

    Without this test the confusion is invisible — both causes produce the same
    `0.0`, and the flattering reading passes every existing assertion.
    """
    from training.fusion.train import _per_class_f1

    # Classes 0..3 exist in the label space. Rows 0,1 carry label 0 and are
    # predicted 0; rows 2,3 carry label 1 and are predicted 2 (a miss). So:
    #   class 0 -> tp=2 fp=0 fn=0 -> F1 = 1.0   (never wrongly predicted)
    #   class 1 -> tp=0 fp=0 fn=2 -> F1 = 0.0   PRESENT, never predicted: REAL miss
    #   class 2 -> tp=0 fp=2 fn=0 -> F1 = 0.0   predicted but never labelled
    #   class 3 -> tp=0 fp=0 fn=0 -> F1 = 0.0   absent: definitional zero
    preds = np.asarray([0, 0, 2, 2], dtype=np.int64)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    per_class = _per_class_f1(preds, labels, 4)
    present = sorted({int(v) for v in labels})
    absent = [c for c in range(4) if c not in set(present)]

    assert per_class[0] == pytest.approx(1.0)
    assert per_class[1] == pytest.approx(0.0)
    assert per_class[2] == pytest.approx(0.0)
    assert per_class[3] == pytest.approx(0.0)

    assert present == [0, 1]
    assert absent == [2, 3]

    zeros = [c for c in range(4) if per_class[c] == 0.0]
    definitional = sorted(set(zeros) & set(absent))
    real_misses = sorted(set(zeros) - set(absent))

    # The whole point: the zero-set is NOT just the absent set.
    assert definitional == [2, 3]
    assert real_misses == [1]
    assert set(zeros) != set(absent)

    # And a real miss must not be explained away as populational: class 1 has
    # two samples, so its 0.0 describes predictions the head actually made.
    assert int(np.sum(labels == 1)) == 2
