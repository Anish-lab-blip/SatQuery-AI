"""Tests for the predeclared seed-variance pass (Gate F D-03 / D-05).

These tests exercise the RULE, not the plumbing:

  * a non-zero spread is reported as the floor;
  * an all-identical (spread == 0) seed set reports `1 / N_val` as the floor;
  * fewer than the predeclared minimum of runs is refused.

The `spread == 0` branch is load-bearing. Flipping it in
`scripts.fusion_seed_variance.compute_floor` makes the identical-set test fail
(it would report `0.0` instead of `1 / N_val`) and the non-zero test fail (it
would report the resolution instead of the spread).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.fusion_seed_variance import (  # noqa: E402
    DECIDING_METRIC_KEY,
    D05_TARGET_SEEDS,
    MIN_RUNS,
    VAL_SAMPLES_KEY,
    FusionSeedVarianceError,
    build_report,
    compute_floor,
    main,
    read_run_record,
)

#: The validation-split size the run records carry; the resolution is 1/N_val.
N_VAL = 4000

#: 1 / 4000, the validation-accuracy resolution for the above.
RESOLUTION_4000 = 0.00025


def _run_record(
    path: Path, *, seed: int, accuracy: float, n_val: int = N_VAL
) -> Path:
    """Write a minimal `run_record.json`-shaped file and return its path."""
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                DECIDING_METRIC_KEY: accuracy,
                VAL_SAMPLES_KEY: n_val,
                "arm": "A",
            }
        ),
        encoding="utf-8",
    )
    return path


def _argv(paths: list[Path], report: Path) -> list[str]:
    argv: list[str] = []
    for path in paths:
        argv += ["--run", str(path)]
    argv += ["--report", str(report)]
    return argv


# ---------------------------------------------------------------------------
# The rule: compute_floor
# ---------------------------------------------------------------------------


def test_nonzero_spread_reports_the_spread_as_the_floor() -> None:
    spread, resolution, floor = compute_floor([0.90, 0.92, 0.91], N_VAL)

    assert spread == pytest.approx(0.02)
    assert resolution == pytest.approx(RESOLUTION_4000)
    assert floor == pytest.approx(0.02)
    # The floor is the spread, NOT the resolution, when the seeds differ.
    assert floor != resolution


def test_identical_accuracies_report_the_resolution_as_the_floor() -> None:
    spread, resolution, floor = compute_floor([0.875, 0.875, 0.875], N_VAL)

    assert spread == 0.0
    assert resolution == pytest.approx(RESOLUTION_4000)
    # spread == 0 -> the floor is the validation-accuracy resolution 1/N_val.
    assert floor == resolution


def test_resolution_is_taken_from_n_val_not_hardcoded() -> None:
    _, resolution_a, _ = compute_floor([0.5, 0.5, 0.5], 4000)
    _, resolution_b, _ = compute_floor([0.5, 0.5, 0.5], 2000)

    assert resolution_a == pytest.approx(0.00025)
    assert resolution_b == pytest.approx(0.0005)


def test_an_empty_seed_set_is_refused() -> None:
    with pytest.raises(FusionSeedVarianceError):
        compute_floor([], N_VAL)


# ---------------------------------------------------------------------------
# The refusal: fewer than MIN_RUNS
# ---------------------------------------------------------------------------


def test_fewer_than_three_runs_is_refused() -> None:
    runs = [
        {"seed": i, DECIDING_METRIC_KEY: 0.9, VAL_SAMPLES_KEY: N_VAL}
        for i in range(MIN_RUNS - 1)
    ]
    with pytest.raises(FusionSeedVarianceError, match="at least"):
        build_report(runs)


def test_record_without_the_deciding_metric_is_refused(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"seed": 0, VAL_SAMPLES_KEY: N_VAL}), encoding="utf-8"
    )
    with pytest.raises(FusionSeedVarianceError, match=DECIDING_METRIC_KEY):
        read_run_record(bad)


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


def test_cli_reports_the_floor_and_exits_zero(tmp_path: Path) -> None:
    accuracies = [0.90, 0.92, 0.91, 0.93, 0.94]
    paths = [
        _run_record(tmp_path / f"r{i}.json", seed=i, accuracy=acc)
        for i, acc in enumerate(accuracies)
    ]
    report_path = tmp_path / "report.json"

    rc = main(_argv(paths, report_path))
    assert rc == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["n_runs"] == D05_TARGET_SEEDS
    assert report["d05_target_met"] is True
    assert report["seeds"] == [0, 1, 2, 3, 4]
    assert report["n_val"] == N_VAL
    assert report["spread"] == pytest.approx(max(accuracies) - min(accuracies))
    assert report["floor"] == pytest.approx(report["spread"])
    # It reports the floor only; it never declares a winner.
    assert "winner" not in report


def test_cli_identical_seeds_report_the_resolution(tmp_path: Path) -> None:
    paths = [
        _run_record(tmp_path / f"r{i}.json", seed=i, accuracy=0.875)
        for i in range(D05_TARGET_SEEDS)
    ]
    report_path = tmp_path / "report.json"

    rc = main(_argv(paths, report_path))
    assert rc == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["spread"] == 0.0
    assert report["resolution"] == pytest.approx(RESOLUTION_4000)
    assert report["floor"] == pytest.approx(RESOLUTION_4000)


def test_cli_refuses_fewer_than_three_runs(tmp_path: Path) -> None:
    paths = [
        _run_record(tmp_path / f"r{i}.json", seed=i, accuracy=0.9)
        for i in range(MIN_RUNS - 1)
    ]
    report_path = tmp_path / "report.json"

    rc = main(_argv(paths, report_path))
    assert rc == 2
    # A refused pass writes no report.
    assert not report_path.exists()
