"""Phase 9 — contracts for the validation-only threshold sweep.

The shipped change detector was scored at a fixed, untuned 0.50
(`evaluation/metrics/change.py` calls its own default "a config value, not a
tuned one"). Selecting a better threshold requires forwarding a split ONCE and
scoring the same probability maps many times. These tests pin the three things
that make that legitimate:

  1. the sweep and the single-shot evaluation share ONE code path
     (`score_dataset`), so a swept number and a shipped number cannot drift;
  2. selection follows a stated rule, including the higher-threshold tie-break;
  3. the CLI refuses to select on anything but the validation split — test is a
     one-shot benchmark and train is the fitting set, so a threshold chosen on
     either is not a held-out selection at all.

`evaluate` is also re-checked here: it was refactored to call
`collect_change_predictions`, and a pure extraction must not move its numbers.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics.change import score_dataset  # noqa: E402
from training.change.train import (  # noqa: E402
    ChangeTrainingError,
    ThresholdSweep,
    collect_change_predictions,
    evaluate,
    sweep_thresholds,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _varied_pairs() -> list[tuple[np.ndarray, np.ndarray]]:
    """Probability maps whose metrics genuinely move with the threshold.

    The probabilities are a noisy function of the truth, so 0.2 / 0.4 / 0.6 /
    0.8 produce different masks and therefore different scores. A dataset where
    every threshold scored the same would make a selection test meaningless.
    """
    rng = np.random.default_rng(7)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(4):
        truth = rng.random((32, 32)) > 0.7
        prob = np.clip(truth * 0.55 + rng.random((32, 32)) * 0.55, 0.0, 1.0)
        pairs.append((truth, prob.astype(np.float32)))
    return pairs


def _make_items(tmp_path: Path, n: int = 2) -> list[dict]:
    """Write `n` tiny A/B/label PNG triples and return the item dicts."""
    from PIL import Image

    a_dir, b_dir, l_dir = tmp_path / "A", tmp_path / "B", tmp_path / "label"
    for d in (a_dir, b_dir, l_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(3)
    items: list[dict] = []
    for i in range(n):
        t1 = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        t2 = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        lab = np.zeros((32, 32), dtype=np.uint8)
        lab[8:24, 8:24] = 255
        a = a_dir / f"val_{i}_0.png"
        b = b_dir / f"val_{i}_0.png"
        l = l_dir / f"val_{i}_0.png"
        Image.fromarray(t1).save(a)
        Image.fromarray(t2).save(b)
        Image.fromarray(lab).save(l)
        items.append({
            "t1_path": str(a), "t2_path": str(b), "label_path": str(l),
            "split": "val", "sample_id": f"val_{i}_0",
            "image_key": f"val_{i}", "group_key": f"val_{i}",
        })
    return items


# ---------------------------------------------------------------------------
# sweep_thresholds — shape and ordering
# ---------------------------------------------------------------------------


def test_sweep_returns_one_row_per_threshold_in_order() -> None:
    pairs = _varied_pairs()
    thresholds = [0.1, 0.5, 0.9]

    sweep = sweep_thresholds(pairs, thresholds)

    assert isinstance(sweep, ThresholdSweep)
    assert [row["threshold"] for row in sweep.rows] == [0.1, 0.5, 0.9]
    assert sweep.thresholds == [0.1, 0.5, 0.9]
    assert sweep.n_pairs == len(pairs)
    # every row carries both conventions and the shared tile-with-change count
    for row in sweep.rows:
        assert set(row) == {"threshold", "pooled", "macro", "n_images_with_change"}
        assert set(row["pooled"]) >= {"precision", "recall", "f1", "iou", "miou"}
        assert set(row["macro"]) >= {"precision", "recall", "f1", "iou", "miou"}


def test_sweep_to_dict_round_trips_the_curve() -> None:
    sweep = sweep_thresholds(_varied_pairs(), [0.3, 0.6])
    payload = sweep.to_dict()

    assert payload["select_by"] == "macro_iou"
    assert payload["thresholds"] == [0.3, 0.6]
    assert [r["threshold"] for r in payload["rows"]] == [0.3, 0.6]
    assert payload["selected"] == sweep.selected
    assert payload["n_pairs"] == sweep.n_pairs


# ---------------------------------------------------------------------------
# The test that proves one code path
# ---------------------------------------------------------------------------


def test_sweep_matches_single_shot_score() -> None:
    """`sweep_thresholds(pairs, [t]).rows[0]` == `score_dataset(pairs, t)`.

    This is the cross-path equivalence the whole design rests on: if the sweep
    ever re-implemented the metric arithmetic, or scored a different mask, this
    would diverge. It is the reason a swept threshold can be trusted to mean the
    same thing the shipped evaluation means.
    """
    pairs = _varied_pairs()
    for t in (0.2, 0.5, 0.8):
        report = score_dataset(pairs, threshold=t)
        sweep = sweep_thresholds(pairs, [t])
        assert sweep.rows[0]["pooled"] == report.pooled.to_dict()
        assert sweep.rows[0]["macro"] == report.macro.to_dict()
        assert sweep.rows[0]["n_images_with_change"] == report.n_images_with_change


# ---------------------------------------------------------------------------
# Selection rule and tie-break
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "select_by,reader",
    [
        ("macro_iou", lambda rep: rep.macro.iou),
        ("pooled_iou", lambda rep: rep.pooled.iou),
        ("f1", lambda rep: rep.pooled.f1),
    ],
)
def test_select_by_reads_the_named_metric(select_by, reader) -> None:
    pairs = _varied_pairs()
    thresholds = [0.2, 0.4, 0.6, 0.8]
    scores = [reader(score_dataset(pairs, threshold=t)) for t in thresholds]
    # Guard the premise: a tie here would make "argmax" ambiguous and the test
    # would be asserting the tie-break, not the metric read.
    assert len({round(s, 12) for s in scores}) == len(thresholds)

    sweep = sweep_thresholds(pairs, thresholds, select_by=select_by)

    expected = thresholds[int(np.argmax(scores))]
    assert sweep.selected["threshold"] == expected
    assert sweep.select_by == select_by


def test_tie_break_prefers_the_higher_threshold() -> None:
    """On an exact tie the HIGHER threshold wins — the conservative choice.

    Probabilities of exactly 0.9 / 0.1 binarize identically at 0.3 and 0.7, so
    the two thresholds score exactly the same and the rule is the only thing
    that can decide. A higher threshold asserts change on fewer pixels and so
    makes fewer false positives, which is the direction that costs least on a
    corpus where only ~5% of pixels change.
    """
    truth = np.array([[1, 0], [0, 1]], dtype=bool)
    prob = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    pairs = [(truth, prob)]

    up = sweep_thresholds(pairs, [0.3, 0.7], select_by="macro_iou")
    assert up.rows[0]["macro"]["iou"] == up.rows[1]["macro"]["iou"]  # exact tie
    assert up.selected["threshold"] == 0.7

    # The rule is order-independent: the same pair given low-first still picks high.
    down = sweep_thresholds(pairs, [0.7, 0.3], select_by="macro_iou")
    assert down.selected["threshold"] == 0.7


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------


def test_unknown_select_by_raises() -> None:
    with pytest.raises(ChangeTrainingError):
        sweep_thresholds(_varied_pairs(), [0.5], select_by="macro_f1")


def test_empty_thresholds_raises() -> None:
    with pytest.raises(ChangeTrainingError):
        sweep_thresholds(_varied_pairs(), [])


# ---------------------------------------------------------------------------
# collect_change_predictions — one pair per tile, in dataset order
# ---------------------------------------------------------------------------


def test_collect_predictions_returns_one_pair_per_item(tmp_path) -> None:
    items = _make_items(tmp_path, n=3)
    torch.manual_seed(0)
    from specialists.change.stanet import STANetStyleChangeDetector

    model = STANetStyleChangeDetector(width=64, pretrained=False)

    pairs = collect_change_predictions(
        model, items, device="cpu", batch_size=2, tile_size=32
    )

    assert len(pairs) == len(items)
    for truth, prob in pairs:
        assert truth.shape == (32, 32)
        assert prob.shape == (32, 32)
        assert prob.dtype == np.float32
        assert float(prob.min()) >= 0.0 and float(prob.max()) <= 1.0


# ---------------------------------------------------------------------------
# Regression — evaluate() is a pure extraction over collect_change_predictions
# ---------------------------------------------------------------------------


def test_evaluate_matches_collect_plus_score(tmp_path) -> None:
    """`evaluate` must be exactly `collect_change_predictions` + `score_dataset`.

    The refactor lifted the forward loop out of `evaluate`; the numbers must not
    have moved. Comparing `evaluate` against the extracted pieces at the same
    threshold is the strongest available check that the extraction was pure.
    """
    items = _make_items(tmp_path, n=2)
    torch.manual_seed(0)
    from specialists.change.stanet import STANetStyleChangeDetector

    model = STANetStyleChangeDetector(width=64, pretrained=False)

    result = evaluate(
        model, items, threshold=0.5, device="cpu", batch_size=2, tile_size=32
    )
    pairs = collect_change_predictions(
        model, items, device="cpu", batch_size=2, tile_size=32
    )
    report = score_dataset(pairs, threshold=0.5)

    assert result.pooled.to_dict() == report.pooled.to_dict()
    assert result.macro.to_dict() == report.macro.to_dict()
    assert result.n == report.n_images
    assert result.n == len(items)
    assert result.threshold == 0.5
    assert result.seconds >= 0.0
    assert result.per_image_change_fraction == list(report.per_image_change_fraction)


def test_evaluate_is_deterministic_on_a_fixed_model(tmp_path) -> None:
    items = _make_items(tmp_path, n=2)
    torch.manual_seed(0)
    from specialists.change.stanet import STANetStyleChangeDetector

    model = STANetStyleChangeDetector(width=64, pretrained=False)

    first = evaluate(model, items, threshold=0.5, device="cpu", batch_size=2, tile_size=32)
    second = evaluate(model, items, threshold=0.5, device="cpu", batch_size=2, tile_size=32)

    assert first.pooled.to_dict() == second.pooled.to_dict()
    assert first.macro.to_dict() == second.macro.to_dict()


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
    part that is easy to get wrong — no artifact is written on the way out.
    """
    module = importlib.import_module("scripts.sweep_change_threshold")

    out_dir = tmp_path / "sweep_out"
    out_dir.mkdir()
    before = sorted(p.name for p in out_dir.iterdir())

    monkeypatch.setattr(sys, "argv", [
        "sweep_change_threshold.py",
        "--data-root", str(tmp_path / "data"),
        "--checkpoint", str(tmp_path / "head.pt"),
        "--output-dir", str(out_dir),
        "--split", forbidden,
    ])

    rc = module.main()

    assert rc == 4
    assert sorted(p.name for p in out_dir.iterdir()) == before
    assert not (out_dir / "threshold_sweep_val.json").exists()


def test_cli_accepts_only_val_as_its_default_split() -> None:
    module = importlib.import_module("scripts.sweep_change_threshold")
    assert module.ALLOWED_SPLIT == "val"


# ---------------------------------------------------------------------------
# CLI — threshold input validation (defects 1 & 2)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sweep_cli_env(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny, self-contained LEVIR root + checkpoint so `main()` can really run.

    Built here rather than borrowed from `artifacts/` so these tests do not
    depend on the shipped head, and so a valid input set is available to prove
    that an exit 2 is caused by the THRESHOLD and nothing else.
    """
    from PIL import Image

    from core.config import load_config
    from specialists.change.stanet import (
        STANetStyleChangeDetector,
        save_change_model,
    )

    root = tmp_path / "data"
    for sub in ("A", "B", "label"):
        (root / "val" / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    for i in range(2):
        t1 = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        t2 = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        lab = np.zeros((32, 32), dtype=np.uint8)
        lab[8:24, 8:24] = 255
        stem = f"scene{i}"
        Image.fromarray(t1).save(root / "val" / "A" / f"{stem}.png")
        Image.fromarray(t2).save(root / "val" / "B" / f"{stem}.png")
        Image.fromarray(lab).save(root / "val" / "label" / f"{stem}.png")

    torch.manual_seed(0)
    model = STANetStyleChangeDetector(width=64, pretrained=False)
    ckpt = tmp_path / "ckpt" / "head.pt"
    # A matching sidecar hash keeps drift False, so the sweep proceeds.
    save_change_model(
        ckpt, model, metadata={"config_hash": load_config().hash, "artifact": "test"}
    )
    return root, ckpt


@pytest.mark.parametrize("spec", ["1.5", "-0.1", "0.5,1.5", "abc", ""])
def test_cli_rejects_bad_thresholds_without_scoring(
    tmp_path, monkeypatch, sweep_cli_env, spec
) -> None:
    """Out-of-range or malformed thresholds exit 2 with no artifact.

    `1.5` and `-0.1` used to propagate `ValueError` up from `binarize` and die
    with a traceback AFTER the full forward pass; `""` used to fall through to
    the default 19-point grid and write an artifact. Both now fail in
    `_build_thresholds`, before the model or the dataset is touched. The data
    root and checkpoint here are VALID, so exit 2 can only come from the
    threshold validation.
    """
    root, ckpt = sweep_cli_env
    module = importlib.import_module("scripts.sweep_change_threshold")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    monkeypatch.setattr(sys, "argv", [
        "sweep_change_threshold.py",
        "--data-root", str(root),
        "--checkpoint", str(ckpt),
        "--output-dir", str(out_dir),
        "--split", "val",
        f"--thresholds={spec}",
    ])

    rc = module.main()

    assert rc == 2
    assert list(out_dir.iterdir()) == [], "a rejected run must write nothing"


def test_cli_empty_thresholds_is_not_the_default_grid(
    tmp_path, monkeypatch, sweep_cli_env, capsys
) -> None:
    """`--thresholds ""` must be an error, not a silent 19-point sweep."""
    root, ckpt = sweep_cli_env
    module = importlib.import_module("scripts.sweep_change_threshold")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    monkeypatch.setattr(sys, "argv", [
        "sweep_change_threshold.py",
        "--data-root", str(root),
        "--checkpoint", str(ckpt),
        "--output-dir", str(out_dir),
        "--split", "val",
        "--thresholds=",
    ])

    rc = module.main()

    assert rc == 2
    assert "parsed to nothing" in capsys.readouterr().out
    assert not (out_dir / "threshold_sweep_val.json").exists()


# ---------------------------------------------------------------------------
# CLI — overwrite policy (defect 3)
# ---------------------------------------------------------------------------


def test_cli_preserves_artifact_without_force_and_overwrites_with_it(
    tmp_path, monkeypatch, sweep_cli_env
) -> None:
    """Default refuses to clobber; `--force` replaces and records the fact."""
    root, ckpt = sweep_cli_env
    module = importlib.import_module("scripts.sweep_change_threshold")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    base = [
        "sweep_change_threshold.py",
        "--data-root", str(root),
        "--checkpoint", str(ckpt),
        "--output-dir", str(out_dir),
        "--split", "val",
        "--limit", "1",
        "--device", "cpu",
        "--thresholds", "0.5",
    ]
    out_path = out_dir / "threshold_sweep_val.json"

    monkeypatch.setattr(sys, "argv", base)
    assert module.main() == 0
    first_bytes = out_path.read_bytes()
    assert json.loads(first_bytes)["forced_overwrite"] is False

    # without --force: exit 2, artifact byte-for-byte unchanged
    monkeypatch.setattr(sys, "argv", base)
    assert module.main() == 2
    assert out_path.read_bytes() == first_bytes

    # with --force: replaced, and the replacement is recorded
    monkeypatch.setattr(sys, "argv", base + ["--force"])
    assert module.main() == 0
    replaced = json.loads(out_path.read_bytes())
    assert replaced["forced_overwrite"] is True
    assert replaced["artifact"] == "change_threshold_sweep"


# ---------------------------------------------------------------------------
# CLI — baseline naming (defect 4)
# ---------------------------------------------------------------------------


def test_baseline_threshold_names_the_real_grid_point(
    tmp_path, monkeypatch, sweep_cli_env
) -> None:
    """On a grid without 0.50 the baseline is the NEAREST point, and is labelled so.

    `|0.7 - 0.5| < |0.3 - 0.5|`, so the nearest grid point to the shipped 0.50
    is 0.70. The artifact must say `baseline_threshold: 0.7` rather than imply
    it compared against 0.50, and the old misleading keys must be gone.
    """
    root, ckpt = sweep_cli_env
    module = importlib.import_module("scripts.sweep_change_threshold")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    monkeypatch.setattr(sys, "argv", [
        "sweep_change_threshold.py",
        "--data-root", str(root),
        "--checkpoint", str(ckpt),
        "--output-dir", str(out_dir),
        "--split", "val",
        "--limit", "1",
        "--device", "cpu",
        "--thresholds", "0.3,0.7",
    ])

    assert module.main() == 0
    payload = json.loads((out_dir / "threshold_sweep_val.json").read_text())

    assert payload["baseline_threshold"] == 0.7
    assert payload["baseline"]["threshold"] == 0.7
    assert "delta_vs_baseline" in payload
    assert "baseline_at_0.50" not in payload
    assert "delta_vs_0.50" not in payload


# ---------------------------------------------------------------------------
# CLI — a scoring-path ValueError is a BUG, not bad input
# ---------------------------------------------------------------------------


def test_scoring_valueerror_is_not_mislabelled_as_bad_input(
    tmp_path, monkeypatch, sweep_cli_env, capsys
) -> None:
    """A `ValueError` from the scoring path must surface, not be called bad input.

    An over-broad `except ValueError` around the scoring call swallowed internal
    failures, printed `FAILED to score (invalid threshold): ...`, exited 2, and
    discarded the traceback. `_build_thresholds` rejects out-of-range input
    before any compute, and `ChangeTrainingError` is a `SatQueryError`, so a
    `ValueError` here can only be a defect in the model/metric code. This pins
    that it propagates as an exception and is never attributed to the threshold.
    """
    root, ckpt = sweep_cli_env
    module = importlib.import_module("scripts.sweep_change_threshold")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def _boom(*_args, **_kwargs):
        raise ValueError("internal bug: unexpected tensor shape in forward")

    # `main()` imports the name at call time, so patching the source module is
    # enough to make the scoring path raise.
    monkeypatch.setattr("training.change.train.collect_change_predictions", _boom)

    monkeypatch.setattr(sys, "argv", [
        "sweep_change_threshold.py",
        "--data-root", str(root),
        "--checkpoint", str(ckpt),
        "--output-dir", str(out_dir),
        "--split", "val",
        "--limit", "1",
        "--device", "cpu",
        "--thresholds", "0.5",   # VALID input: the failure is not the threshold
    ])

    # The defect must NOT be converted into a clean exit 2.
    with pytest.raises(ValueError):
        module.main()

    # The assertion that actually pins the defect: the failure is not
    # misattributed to the threshold.
    assert "invalid threshold" not in capsys.readouterr().out
    assert not (out_dir / "threshold_sweep_val.json").exists()
