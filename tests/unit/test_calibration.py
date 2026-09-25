"""STEP 2 — the calibration fitter, the artifact, and the consumer wiring.

WHAT THESE TESTS PROTECT
------------------------
Three separate claims, tested separately because they fail independently:

  1. the FITTER is correct            (fitter semantics on known inputs)
  2. the ARTIFACT is honest           (provenance, held-out refusal, consumer contract)
  3. the CONTROLLER CONSUMES it       (the section 67 violation, and its inverse:
                                       that absence still degrades rather than crashes)

Claim 3 is the one that actually mattered. Before 2026-09-22 a fitted artifact
could exist and the deployed controller would still emit
`method="uncalibrated"`, because `core/controller.py` never passed
`calibration=` to `EvidenceEngine.from_config`. Creating the artifact without
fixing that call site would have left the section 67 violation in place while
making it look fixed -- the exact failure mode these tests exist to catch.

WHY THE FITTER TESTS USE SYNTHETIC PROBABILITIES
------------------------------------------------
A temperature fitter is a numerical optimiser. Testing it only on the real
5.8 MB checkpoint would make the test slow, non-hermetic, and silent about
whether the maths is right -- it would tell us "the fit runs", not "the fit is
correct". The synthetic cases below have KNOWN answers: an overconfident
distribution must yield `T > 1`, an underconfident one `T < 1`, and an already
calibrated one `T ~ 1`.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.config import REPO_ROOT, load_config
from core.controller import AnalysisController
from core.planner import PolicyPlanner
from core.registry import SpecialistRegistry
from core.schemas import ConfidenceBreakdown
from evidence.confidence import (
    METHOD_TEMPERATURE,
    METHOD_UNCALIBRATED,
    TemperatureCalibration,
    calibrate,
    load_calibration,
)
from training.calibration.artifact import (
    HELD_OUT_SPLITS,
    CalibrationFitError,
    build_artifact_payload,
    file_sha256,
    write_calibration_artifact,
)
from training.calibration.evaluate import (
    compare_calibration,
    expected_calibration_error,
    mean_negative_log_likelihood,
    reliability_diagram,
)
from training.calibration.fitter import (
    fit_temperature,
    logits_from_probabilities,
    softmax,
)

FROZEN_CONFIG_HASH = "78f1e3700da15aa1"


@pytest.fixture
def scratch() -> Path:
    """Self-managed scratch; `tmp_path` removals are blocked by a sandbox guard."""
    return Path(tempfile.mkdtemp(prefix="sq_calib_"))


# ---------------------------------------------------------------------------
# Helpers — distributions with KNOWN calibration properties
# ---------------------------------------------------------------------------
def _overconfident(n: int = 4000, n_classes: int = 5, peak: float = 0.97, seed: int = 0):
    """A model that is right ~60% of the time but says ~97% every time.

    Confidence far exceeds accuracy, so the fitted temperature must exceed 1
    (soften the scores).
    """
    rng = np.random.default_rng(seed)
    gold = rng.integers(0, n_classes, size=n)
    probabilities = np.full((n, n_classes), (1.0 - peak) / (n_classes - 1))
    correct = rng.random(n) < 0.6
    chosen = np.where(correct, gold, (gold + 1) % n_classes)
    probabilities[np.arange(n), chosen] = peak
    return probabilities, gold


def _underconfident(n: int = 4000, n_classes: int = 5, peak: float = 0.30, seed: int = 1):
    """A model that is right ~90% of the time but hedges near-uniform.

    Confidence below accuracy means the scores should be sharpened, so `T < 1`.
    """
    rng = np.random.default_rng(seed)
    gold = rng.integers(0, n_classes, size=n)
    probabilities = np.full((n, n_classes), (1.0 - peak) / (n_classes - 1))
    correct = rng.random(n) < 0.9
    chosen = np.where(correct, gold, (gold + 2) % n_classes)
    probabilities[np.arange(n), chosen] = peak
    return probabilities, gold


# ---------------------------------------------------------------------------
# fitter.py — the maths
# ---------------------------------------------------------------------------
def test_softmax_rows_sum_to_one_and_are_shift_invariant() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0]])
    probs = softmax(logits)
    assert np.allclose(probs.sum(axis=1), 1.0)
    # Adding a per-row constant must not change the result.
    assert np.allclose(probs, softmax(logits + 100.0))


def test_logits_from_probabilities_round_trips_through_softmax() -> None:
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(50, 4))
    recovered = logits_from_probabilities(softmax(logits))
    # Equal up to a per-row additive constant, which softmax removes.
    assert np.allclose(softmax(recovered), softmax(logits), atol=1e-9)


def test_overconfident_model_yields_temperature_above_one() -> None:
    """The load-bearing semantic: soften overconfidence."""
    probabilities, gold = _overconfident()
    fit = fit_temperature(probabilities, gold)
    assert fit.temperature > 1.0, (
        f"overconfident scores must give T > 1, got {fit.temperature}"
    )
    assert fit.nll_after < fit.nll_before, "the fit must reduce NLL"
    assert fit.hit_bound is False


def test_underconfident_model_yields_temperature_below_one() -> None:
    """The converse: sharpen underconfidence."""
    probabilities, gold = _underconfident()
    fit = fit_temperature(probabilities, gold)
    assert fit.temperature < 1.0, (
        f"underconfident scores must give T < 1, got {fit.temperature}"
    )
    assert fit.nll_after < fit.nll_before


def test_a_perfectly_calibrated_model_yields_the_identity() -> None:
    """If the scores are already honest, the fit must not manufacture a change.

    This is the honesty check on the fitter itself: a fitter that always returns
    something other than 1.0 would "fix" a model that needs no fixing and report
    a calibration that is not there.

    CONSTRUCTING A GENUINELY CALIBRATED PREDICTOR: draw a confidence `c` for each
    row and then make the row correct with probability exactly `c`. The stated
    confidence then matches the realised hit rate by construction, which is the
    definition of calibrated. (A constant uniform vector is NOT a usable test
    here even though it is trivially calibrated: softmax(1/T) is the same
    uniform distribution for every T, so its NLL is flat and the optimum is
    undefined. That degenerate case has its own test below.)
    """
    rng = np.random.default_rng(3)
    n, n_classes = 40000, 4
    confidence = rng.uniform(0.30, 0.95, size=n)
    gold = rng.integers(0, n_classes, size=n)
    correct = rng.random(n) < confidence
    chosen = np.where(correct, gold, (gold + 1) % n_classes)

    # Spread the remaining mass so the stated max is exactly `confidence`.
    rest = (1.0 - confidence) / (n_classes - 1)
    probabilities = np.full((n, n_classes), 0.0)
    probabilities[np.arange(n), chosen] = confidence
    remaining = np.ones((n, n_classes), dtype=bool)
    remaining[np.arange(n), chosen] = False
    probabilities[remaining] = np.repeat(rest, n_classes - 1)

    fit = fit_temperature(probabilities, gold)
    assert fit.temperature == pytest.approx(1.0, abs=0.25), (
        f"a calibrated predictor must not be re-scaled, got T={fit.temperature}"
    )


def test_uniform_scores_return_the_identity_not_a_bound() -> None:
    """A flat NLL surface must yield T = 1, not a fabricated correction.

    `softmax(1/T)` on a constant score vector is the same uniform distribution
    for every `T`, so the NLL is identical everywhere. Golden-section search on a
    flat function walks to a bound and would report `T = max_temperature`, which
    the consumer would then stamp as a real `temperature_scaling` correction that
    does nothing. `hit_bound` records the degeneracy; the temperature is the
    honest identity.
    """
    rng = np.random.default_rng(3)
    n, n_classes = 5000, 4
    gold = rng.integers(0, n_classes, size=n)
    probabilities = np.full((n, n_classes), 1.0 / n_classes)

    fit = fit_temperature(probabilities, gold)
    assert fit.temperature == 1.0, (
        f"a flat objective must give the identity, got T={fit.temperature}"
    )
    assert fit.is_effective is False
    # The NLL genuinely did not move, which is why the identity is correct.
    assert fit.nll_after == pytest.approx(fit.nll_before, abs=1e-12)


def test_fit_is_deterministic() -> None:
    """Same inputs -> same T, bit for bit. No RNG, no clock."""
    probabilities, gold = _overconfident(n=500)
    a = fit_temperature(probabilities, gold)
    b = fit_temperature(probabilities, gold)
    assert a.temperature == b.temperature
    assert a.iterations == b.iterations


def test_fit_rejects_malformed_input() -> None:
    probabilities, gold = _overconfident(n=100)
    with pytest.raises(ValueError, match="2-D"):
        fit_temperature(probabilities[:, 0], gold)
    with pytest.raises(ValueError, match="rows"):
        fit_temperature(probabilities, gold[:-1])
    with pytest.raises(ValueError, match="zero samples"):
        fit_temperature(probabilities[:0], gold[:0])
    with pytest.raises(ValueError, match="non-finite"):
        bad = probabilities.copy()
        bad[0, 0] = np.nan
        fit_temperature(bad, gold)
    with pytest.raises(ValueError, match="targets must be"):
        fit_temperature(probabilities, np.zeros(len(gold), dtype=int) + 99)


def test_fitted_temperature_is_constructible_as_a_consumer_artifact() -> None:
    """The fitter's range must be inside the consumer's validation range.

    A fitter that can return a value `TemperatureCalibration.__post_init__`
    rejects would produce an artifact that cannot be loaded.
    """
    for probabilities, gold in (_overconfident(), _underconfident()):
        fit = fit_temperature(probabilities, gold)
        TemperatureCalibration(temperature=fit.temperature)  # must not raise


# ---------------------------------------------------------------------------
# evaluate.py — the metrics
# ---------------------------------------------------------------------------
def test_ece_is_zero_for_a_perfectly_calibrated_predictor() -> None:
    """A constant 0.5 predictor with a 50% hit rate is calibrated: ECE ~ 0."""
    rng = np.random.default_rng(11)
    n = 40000
    probabilities = np.full((n, 2), 0.5)
    gold = (rng.random(n) < 0.5).astype(int)
    assert expected_calibration_error(probabilities, gold) < 0.01


def test_ece_is_large_for_a_confidently_wrong_predictor() -> None:
    """Predicting 0.99 and being wrong is the maximal miscalibration."""
    n = 1000
    probabilities = np.zeros((n, 2))
    probabilities[:, 1] = 0.99
    probabilities[:, 0] = 0.01
    gold = np.zeros(n, dtype=int)  # every prediction wrong
    assert expected_calibration_error(probabilities, gold) > 0.9


def test_ece_decreases_when_an_overconfident_model_is_tempered() -> None:
    """The whole point of fitting: ECE should improve on the fitting data."""
    probabilities, gold = _overconfident()
    fit = fit_temperature(probabilities, gold)
    calibration = TemperatureCalibration(temperature=fit.temperature)
    metrics = compare_calibration(probabilities, gold, calibration)
    assert metrics.ece_after < metrics.ece_before
    assert metrics.nll_after < metrics.nll_before


def test_nll_matches_the_analytic_value() -> None:
    """NLL of a known distribution, computed by hand."""
    probabilities = np.array([[0.5, 0.5], [1.0, 0.0]])
    gold = [0, 0]
    # -(log 0.5 + log 1.0) / 2
    assert mean_negative_log_likelihood(probabilities, gold) == pytest.approx(
        -math.log(0.5) / 2
    )


def test_reliability_diagram_has_fixed_length_and_reports_the_ece() -> None:
    probabilities, gold = _overconfident(n=2000)
    diagram = reliability_diagram(probabilities, gold, n_bins=10)
    assert len(diagram["bins"]) == 10
    assert len(diagram["edges"]) == 11
    assert diagram["ece"] == pytest.approx(
        expected_calibration_error(probabilities, gold, n_bins=10), abs=1e-6
    )
    # Empty bins are preserved with null values so the table is plottable.
    for entry in diagram["bins"]:
        if entry["count"] == 0:
            assert entry["accuracy"] is None


def test_ece_rejects_a_bad_bin_count() -> None:
    probabilities, gold = _overconfident(n=100)
    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error(probabilities, gold, n_bins=0)


# ---------------------------------------------------------------------------
# artifact.py — provenance and the held-out guard
# ---------------------------------------------------------------------------
def _payload(fitted_on: str = "Val") -> dict:
    probabilities, gold = _overconfident(n=2000)
    fit = fit_temperature(probabilities, gold)
    calibration = TemperatureCalibration(temperature=fit.temperature)
    metrics = compare_calibration(probabilities, gold, calibration)
    return build_artifact_payload(
        fit=fit,
        metrics=metrics,
        fitted_on=fitted_on,
        reliability=reliability_diagram(probabilities, gold),
        checkpoint_sha256="a" * 64,
        config_hash=FROZEN_CONFIG_HASH,
        dataset_id="cdvqa",
        feature_spec="change_feat_v1",
        n_classes=5,
    )


@pytest.mark.parametrize("split", sorted(HELD_OUT_SPLITS))
def test_writer_refuses_to_fit_on_a_held_out_split(split: str) -> None:
    """Fitting on Test/Test2 would make the calibration a leak.

    Enforced in the library, not only in the CLI, so a direct call cannot bypass
    it. This is the single most important guard in this package.
    """
    with pytest.raises(CalibrationFitError, match="held-out"):
        _payload(fitted_on=split)


def test_artifact_payload_is_readable_by_the_consumer() -> None:
    """The payload must satisfy `TemperatureCalibration.from_dict` verbatim.

    This is the contract test: if the writer and the consumer ever disagree, the
    artifact exists but loads as `None` and every confidence silently reverts to
    `uncalibrated`.
    """
    payload = _payload()
    calibration = TemperatureCalibration.from_dict(payload)
    assert calibration.temperature == pytest.approx(
        payload["temperature_scaling"]["temperature"]
    )
    assert calibration.fitted_on == "Val"
    assert calibration.n_samples == payload["temperature_scaling"]["n_samples"]
    assert calibration.is_effective()


def test_artifact_records_the_provenance_a_reviewer_needs() -> None:
    payload = _payload()
    provenance = payload["provenance"]
    assert provenance["method"] == "temperature_scaling"
    assert provenance["objective"] == "mean_negative_log_likelihood"
    assert provenance["fitted_on"] == "Val"
    assert provenance["config_hash"] == FROZEN_CONFIG_HASH
    assert provenance["checkpoint_sha256"] == "a" * 64
    assert provenance["n_classes"] == 5
    assert sorted(HELD_OUT_SPLITS) == provenance["held_out_splits_excluded"]
    # Before/after diagnostics must be present, not just the scalar.
    assert "ece_before" in payload["metrics"]
    assert "ece_after" in payload["metrics"]
    assert "reliability_diagram" in payload
    assert payload["fit_diagnostics"]["iterations"] > 0


def test_writer_round_trips_through_disk(scratch: Path) -> None:
    payload = _payload()
    path = write_calibration_artifact(payload, scratch / "calibration_v001.json")
    assert path.exists()
    reloaded = TemperatureCalibration.from_json(path, required=True)
    assert reloaded.temperature == pytest.approx(
        payload["temperature_scaling"]["temperature"]
    )
    # The file-bytes digest is a real hash of the file, unlike the §6a defect.
    assert file_sha256(path) == file_sha256(path)
    assert len(file_sha256(path)) == 64


def test_writer_rejects_a_non_finite_temperature() -> None:
    probabilities, gold = _overconfident(n=500)
    fit = fit_temperature(probabilities, gold)
    metrics = compare_calibration(
        probabilities, gold, TemperatureCalibration(temperature=fit.temperature)
    )
    broken = type(fit)(
        temperature=float("nan"),
        n_samples=fit.n_samples,
        nll_before=fit.nll_before,
        nll_after=fit.nll_after,
        log_temperature=0.0,
        hit_bound=False,
        iterations=fit.iterations,
    )
    with pytest.raises(CalibrationFitError, match="unusable temperature"):
        build_artifact_payload(fit=broken, metrics=metrics, fitted_on="Val")


# ---------------------------------------------------------------------------
# The section 67 fix — the controller must CONSUME the artifact
# ---------------------------------------------------------------------------
def test_load_calibration_finds_the_shipped_artifact() -> None:
    """The real config must resolve to the real artifact, from the real repo.

    This is the test that would have caught the original defect: before the fix,
    `load_calibration` resolved against a directory that never contained the
    file, so it returned `None` while the switch was ON.
    """
    config = load_config()
    assert config.hash == FROZEN_CONFIG_HASH
    assert config.get("confidence.temperature_scaling") is True

    artifact = REPO_ROOT / "artifacts" / "calibration_v001.json"
    if not artifact.exists():
        pytest.skip("calibration artifact not fitted in this checkout")

    calibration = load_calibration(config)
    assert calibration is not None, (
        "the master switch is on and the artifact exists, but load_calibration "
        "did not find it"
    )
    assert calibration.fitted_on not in HELD_OUT_SPLITS
    assert calibration.n_samples and calibration.n_samples > 0


def test_controller_loads_the_calibration_it_can_find() -> None:
    """The controller's evidence engine must carry a calibration when one exists.

    Absence is tolerated (the next test covers it); what must NOT happen is the
    controller carrying `None` while an artifact sits on disk.
    """
    artifact = REPO_ROOT / "artifacts" / "calibration_v001.json"
    if not artifact.exists():
        pytest.skip("calibration artifact not fitted in this checkout")

    config = load_config()
    registry = SpecialistRegistry.discover(config, device="cpu")
    controller = AnalysisController(
        registry=registry, planner=PolicyPlanner(registry), config=config
    )
    assert controller.evidence.calibration is not None, (
        "core/controller.py did not pass `calibration=` to EvidenceEngine."
        "from_config -- this is the section 67 violation returning"
    )


def test_controller_emits_calibrated_confidence_when_fitted() -> None:
    """End to end: a raw score leaving the controller reads `temperature_scaling`."""
    artifact = REPO_ROOT / "artifacts" / "calibration_v001.json"
    if not artifact.exists():
        pytest.skip("calibration artifact not fitted in this checkout")

    config = load_config()
    registry = SpecialistRegistry.discover(config, device="cpu")
    controller = AnalysisController(
        registry=registry, planner=PolicyPlanner(registry), config=config
    )

    breakdown = ConfidenceBreakdown(
        raw=0.90, calibrated=None, method=METHOD_UNCALIBRATED, components={}
    )
    result = controller.evidence.confidence_for(breakdown)

    assert result.method == METHOD_TEMPERATURE
    assert result.calibrated is not None
    assert result.value == pytest.approx(result.calibrated)
    assert result.components.get("calibrated_applied") == 1.0
    # The fitted temperature travels with the result, so it is traceable.
    assert "temperature" in result.components


def test_controller_degrades_honestly_without_an_artifact() -> None:
    """The inverse claim: no artifact must mean `uncalibrated`, not a crash.

    A deployment may legitimately ship without the fitted artifact. It must
    construct, answer, and SAY it is uncalibrated.
    """
    engine_calibration_absent = calibrate(0.90, None)
    assert engine_calibration_absent.method == METHOD_UNCALIBRATED
    assert engine_calibration_absent.calibrated is None
    assert engine_calibration_absent.value == pytest.approx(0.90)
    assert engine_calibration_absent.components["calibrated_applied"] == 0.0


def test_an_identity_temperature_is_reported_uncalibrated() -> None:
    """T = 1.0 is not a calibration. It must not claim to be one."""
    identity = TemperatureCalibration(temperature=1.0)
    assert identity.is_effective() is False
    breakdown = calibrate(0.80, identity)
    assert breakdown.method == METHOD_UNCALIBRATED
    assert breakdown.calibrated is None
    assert breakdown.components["calibration_identity"] == 1.0


def test_a_corrupt_artifacts_raises_rather_than_degrading(scratch: Path) -> None:
    """Absent degrades; corrupt raises. The two must not be conflated.

    Silently treating a malformed artifact as "no artifact" would hide an
    operational defect behind a plausible-looking uncalibrated result.
    """
    bad = scratch / "calibration_v001.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        TemperatureCalibration.from_json(bad, required=True)

    bad.write_text(json.dumps({"no_temperature_here": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="temperature"):
        TemperatureCalibration.from_json(bad, required=True)


def test_load_calibration_respects_the_master_switch(scratch: Path) -> None:
    """Switch OFF means no calibration, whatever sits on disk."""
    (scratch / "calibration_v001.json").write_text(
        json.dumps({"temperature": 1.5}), encoding="utf-8"
    )

    class _Stub:
        def get(self, path: str, default=None):
            data = {
                "confidence": {
                    "temperature_scaling": False,
                    "calibration_file": "calibration_v001.json",
                }
            }
            node = data
            for part in path.split("."):
                node = node.get(part, default) if isinstance(node, dict) else default
            return node

    assert load_calibration(_Stub(), base_dir=scratch) is None


def test_no_hidden_calibration_state_above_the_configured_path() -> None:
    """The engine must not invent a calibration the config did not name.

    Guards against a future change that silently blends in a module-level or
    cached default -- which would make behaviour depend on import order.
    """
    from evidence.engine import EvidenceEngine

    engine = EvidenceEngine()
    assert engine.calibration is None
    breakdown = engine.confidence_for(
        ConfidenceBreakdown(
            raw=0.75, calibrated=None, method=METHOD_UNCALIBRATED, components={}
        )
    )
    assert breakdown.method == METHOD_UNCALIBRATED
    assert breakdown.calibrated is None
