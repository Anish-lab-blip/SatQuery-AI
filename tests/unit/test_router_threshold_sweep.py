"""Phase 4/13 — contracts for the validation-only ROUTER threshold sweep.

`router/classifier.py` gates the learned router behind a confidence threshold
(`IntentRouter.confidence_threshold`, default `0.70`), and
`docs/PHASE4_ROUTER_REPORT.md:163` records that value as **uncalibrated**. This
suite pins the three things that make a sweep of it legitimate:

  1. the CLI refuses to select on anything but the validation split — test is a
     one-shot benchmark and train is the fitting set, so a threshold chosen on
     either is not a held-out selection at all;
  2. the sweep is a PURE function over (confidence, correctness) pairs, so the
     trade-off it reports can be checked without loading the encoder;
  3. selection follows a stated rule, including the lower-threshold tie-break
     (the most permissive gate that is no less accurate).

The encoder is deliberately NOT loaded here: the split firewall and the
threshold validation both fire before it, and the sweep arithmetic is pure. The
`corpus-limited` caveat (val n=86 vs the plan's >= 500) is recorded in the
artifact, not asserted here — see `scripts/sweep_router_threshold.py`.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _module():
    return importlib.import_module("scripts.sweep_router_threshold")


# ---------------------------------------------------------------------------
# CLI — the split firewall
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("forbidden", ["test", "train"])
def test_cli_refuses_forbidden_split_and_writes_nothing(
    tmp_path, monkeypatch, forbidden
) -> None:
    """`--split test` and `--split train` exit 4 and leave the output dir alone.

    A threshold selected on test spends the one-shot benchmark; one selected on
    train measures memorisation. Both are refused with no override, and — the
    part that is easy to get wrong — no artifact is written on the way out. The
    firewall runs BEFORE the config or the encoder is touched, so this test needs
    no model.
    """
    module = _module()

    out_dir = tmp_path / "sweep_out"
    out_dir.mkdir()
    before = sorted(p.name for p in out_dir.iterdir())

    monkeypatch.setattr(sys, "argv", [
        "sweep_router_threshold.py",
        "--output-dir", str(out_dir),
        "--split", forbidden,
    ])

    rc = module.main()

    assert rc == 4
    assert sorted(p.name for p in out_dir.iterdir()) == before
    assert not (out_dir / module.ARTIFACT_NAME).exists()


def test_cli_accepts_only_val_as_its_default_split() -> None:
    module = _module()
    assert module.ALLOWED_SPLIT == "val"


def test_artifact_name_is_the_documented_one() -> None:
    module = _module()
    assert module.ARTIFACT_NAME == "threshold_sweep_val.json"


# ---------------------------------------------------------------------------
# CLI — threshold input validation, before any model is loaded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ["1.5", "-0.1", "0.5,1.5", "abc", ""])
def test_cli_rejects_bad_thresholds_without_scoring(
    tmp_path, monkeypatch, spec
) -> None:
    """Out-of-range or malformed thresholds exit 2 with no artifact.

    `_build_thresholds` runs before the corpus or the encoder is loaded, so a
    typo costs nothing and cannot leave a half-written artifact behind.
    """
    module = _module()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    monkeypatch.setattr(sys, "argv", [
        "sweep_router_threshold.py",
        "--output-dir", str(out_dir),
        "--split", "val",
        f"--thresholds={spec}",
    ])

    rc = module.main()

    assert rc == 2
    assert list(out_dir.iterdir()) == [], "a rejected run must write nothing"


def test_cli_empty_thresholds_is_not_the_default_grid(
    tmp_path, monkeypatch, capsys
) -> None:
    """`--thresholds ""` must be an error, not a silent full-grid sweep."""
    module = _module()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    monkeypatch.setattr(sys, "argv", [
        "sweep_router_threshold.py",
        "--output-dir", str(out_dir),
        "--split", "val",
        "--thresholds=",
    ])

    rc = module.main()

    assert rc == 2
    assert "parsed to nothing" in capsys.readouterr().out
    assert not (out_dir / module.ARTIFACT_NAME).exists()


# ---------------------------------------------------------------------------
# _build_thresholds — pure grid construction
# ---------------------------------------------------------------------------


def test_build_thresholds_default_grid_is_well_formed() -> None:
    module = _module()
    args = type("A", (), {
        "thresholds": None, "start": 0.50, "stop": 0.99, "step": 0.01,
    })()
    grid = module._build_thresholds(args)
    assert grid[0] == 0.50
    assert grid[-1] == 0.99
    assert len(grid) == 50
    assert grid == sorted(grid)
    assert all(0.0 <= t <= 1.0 for t in grid)


def test_build_thresholds_explicit_list_rounds_to_6dp() -> None:
    module = _module()
    args = type("A", (), {"thresholds": "0.05, 0.5 ,0.95", "start": 0, "stop": 1, "step": 1})()
    assert module._build_thresholds(args) == [0.05, 0.5, 0.95]


def test_build_thresholds_rejects_step_le_zero() -> None:
    module = _module()
    args = type("A", (), {"thresholds": None, "start": 0.0, "stop": 1.0, "step": 0.0})()
    with pytest.raises(ValueError):
        module._build_thresholds(args)


# ---------------------------------------------------------------------------
# sweep_confidence — the pure trade-off function
# ---------------------------------------------------------------------------


def _handmade() -> tuple[list[float], list[bool]]:
    """Confidences and correctness where the threshold genuinely matters.

    Six queries: three confident-and-correct, one confident-and-WRONG, two
    low-confidence. Moving the threshold across 0.4/0.8 changes both coverage
    and served-set accuracy, so the curve is not degenerate.
    """
    confidences = [0.9, 0.85, 0.8, 0.75, 0.5, 0.3]
    correct = [True, True, True, False, True, False]
    return confidences, correct


def test_sweep_returns_one_row_per_threshold_in_order() -> None:
    module = _module()
    conf, ok = _handmade()
    rows, _sel = module.sweep_confidence(conf, ok, [0.3, 0.5, 0.9])

    assert [r["threshold"] for r in rows] == [0.3, 0.5, 0.9]
    for row in rows:
        assert set(row) == {
            "threshold", "n_covered", "coverage",
            "covered_task_accuracy", "fallback_rate",
        }


def test_coverage_is_monotonic_non_increasing_in_threshold() -> None:
    """Raising the gate can only ever serve fewer or equal queries."""
    module = _module()
    conf, ok = _handmade()
    rows, _ = module.sweep_confidence(conf, ok, [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95])
    coverages = [r["coverage"] for r in rows]
    assert coverages == sorted(coverages, reverse=True)
    assert rows[-1]["n_covered"] <= rows[0]["n_covered"]


def test_served_accuracy_is_the_accuracy_among_covered_queries() -> None:
    module = _module()
    conf, ok = _handmade()
    rows, _ = module.sweep_confidence(conf, ok, [0.7])
    row = rows[0]
    # threshold 0.7 covers 0.9/0.85/0.8/0.75 -> correct among them = 3/4
    assert row["n_covered"] == 4
    assert row["covered_task_accuracy"] == pytest.approx(0.75)
    assert row["fallback_rate"] == pytest.approx(2 / 6)


def test_threshold_above_every_confidence_covers_nothing() -> None:
    module = _module()
    conf, ok = _handmade()
    rows, _ = module.sweep_confidence(conf, ok, [0.999])
    assert rows[0]["n_covered"] == 0
    assert rows[0]["coverage"] == 0.0
    assert rows[0]["covered_task_accuracy"] is None
    assert rows[0]["fallback_rate"] == 1.0


# ---------------------------------------------------------------------------
# Selection rule and tie-break
# ---------------------------------------------------------------------------


def test_covered_accuracy_selects_the_highest_served_accuracy() -> None:
    module = _module()
    conf, ok = _handmade()
    _rows, selected = module.sweep_confidence(
        conf, ok, [0.3, 0.5, 0.7, 0.8, 0.9], select_by="covered_accuracy"
    )
    # 0.8 covers 0.9/0.85/0.8 -> 3/3 = 1.0, the max; it is the lowest such.
    assert selected["threshold"] == 0.8
    assert selected["covered_task_accuracy"] == 1.0


def test_covered_accuracy_tie_break_prefers_the_lower_threshold() -> None:
    """On an exact accuracy tie the LOWER threshold wins — serve more.

    Both 0.8 and 0.9 give served accuracy 1.0 here, so the tie-break is the only
    thing that can decide, and it must pick 0.8 (more coverage, same accuracy).
    """
    module = _module()
    conf, ok = _handmade()
    rows, selected = module.sweep_confidence(
        conf, ok, [0.8, 0.9], select_by="covered_accuracy"
    )
    accs = {r["threshold"]: r["covered_task_accuracy"] for r in rows}
    assert accs[0.8] == accs[0.9] == 1.0  # exact tie on accuracy
    assert selected["threshold"] == 0.8

    # Order-independent: the same grid reversed still picks the lower threshold.
    _rows2, selected2 = module.sweep_confidence(
        conf, ok, [0.9, 0.8], select_by="covered_accuracy"
    )
    assert selected2["threshold"] == 0.8


def test_coverage_at_full_accuracy_selects_lowest_perfect_threshold() -> None:
    module = _module()
    conf, ok = _handmade()
    _rows, selected = module.sweep_confidence(
        conf, ok, [0.3, 0.5, 0.7, 0.8, 0.9],
        select_by="coverage_at_full_accuracy",
    )
    assert selected["threshold"] == 0.8
    assert selected["covered_task_accuracy"] == 1.0


def test_coverage_at_full_accuracy_falls_back_to_highest_when_none_perfect() -> None:
    module = _module()
    # 0.9 correct, 0.85 WRONG: no threshold gives served accuracy 1.0 with >=1 cover.
    conf = [0.9, 0.85]
    ok = [True, False]
    _rows, selected = module.sweep_confidence(
        conf, ok, [0.5, 0.8, 0.9], select_by="coverage_at_full_accuracy"
    )
    assert selected["threshold"] == 0.9  # the highest threshold


def test_unknown_select_by_raises() -> None:
    module = _module()
    conf, ok = _handmade()
    with pytest.raises(ValueError):
        module.sweep_confidence(conf, ok, [0.5], select_by="macro_f1")


def test_empty_val_set_raises() -> None:
    module = _module()
    with pytest.raises(ValueError):
        module.sweep_confidence([], [], [0.5])


def test_length_mismatch_raises() -> None:
    module = _module()
    with pytest.raises(ValueError):
        module.sweep_confidence([0.5, 0.6], [True], [0.5])


# ---------------------------------------------------------------------------
# _split_contamination — `test_split_touched` is MEASURED, not hardcoded
#
# QA flagged the original artifact for carrying `"test_split_touched": False`
# as a literal: a claim no change could falsify. The value is now DERIVED from
# the texts actually handed to the encoder. These tests pin the derivation.
# ---------------------------------------------------------------------------


def test_split_contamination_is_false_when_only_val_is_scored() -> None:
    module = _module()
    touched, n_test, n_val = module._split_contamination(
        ["a", "b"], val_texts={"a", "b"}, test_texts={"c", "d"}
    )
    assert touched is False
    assert n_test == 0
    assert n_val == 2


def test_split_contamination_flips_true_when_a_test_text_is_scored() -> None:
    """The load-bearing proof: the measurement is falsifiable.

    If a test example ever reaches the encoder, the measurement MUST report it —
    which is exactly what a hardcoded `False` could never do.
    """
    module = _module()
    touched, n_test, n_val = module._split_contamination(
        ["a", "b", "c"], val_texts={"a", "b"}, test_texts={"c", "d"}
    )
    assert touched is True
    assert n_test == 1
    assert n_val == 2


def test_script_derives_test_split_touched_rather_than_hardcoding_it() -> None:
    """Regression guard for the QA-flagged defect.

    The script must call the measurement and must NOT contain the hardcoded
    literal that started this. Reverting the fix makes this test fail.
    """
    module = _module()
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "_split_contamination(" in source
    assert '"test_split_touched": False' not in source
