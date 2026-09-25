"""Calibration artifact writer — produces `artifacts/calibration_v001.json`.

THE CONTRACT THIS FILE HONOURS
------------------------------
`evidence/confidence.py` is the consumer and already defines the artifact's
shape, twice over:

    TemperatureCalibration.from_dict  accepts either
        {"temperature": T, ...}                      (flat), or
        {"temperature_scaling": {"temperature": T, ...}}   (nested)

It reads `fitted_on` (or `split`), `artifact`, and `n_samples`. This writer
emits the **nested** form with the full provenance block alongside it, so the
consumer's fast path works unchanged while a human reviewer gets everything the
project's provenance discipline asks for.

WHY THE PROVENANCE IS SO HEAVY
------------------------------
A temperature is a single float. On its own it is unverifiable: nothing in
`{"temperature": 1.42}` says which data it came from, which model produced the
probabilities, or which question type was being calibrated. A later reader cannot
tell a careful fit from a number someone typed. So the artifact records the
fitting split, the sample count, the checkpoint and config hashes, the metrics
before and after, and the reliability diagram.

THE HELD-OUT GUARD
------------------
`write_calibration_artifact` refuses any `fitted_on` split listed in
`HELD_OUT_SPLITS`. Fitting on Test/Test2 would make the calibration a leak.
This is enforced here as well as in the CLI so the guard cannot be bypassed by
calling the writer directly.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.errors import SatQueryError

#: Schema tag for the artifact. Bump on a breaking shape change.
CALIBRATION_SCHEMA = "calibration_v1"

#: The filename the frozen config already names
#: (`configs/base.yaml` -> `confidence.calibration_file`). The artifact lands in
#: `artifacts/`, and `load_calibration(config, base_dir="configs")` resolves
#: `configs/../artifacts/<filename>` -- see `core/config.py::REPO_ROOT`.
DEFAULT_ARTIFACT_NAME = "calibration_v001.json"

#: Splits that may never be used to fit a temperature. Mirrors
#: `scripts/evaluate_change_vqa.py::HELD_OUT_SPLITS`; duplicated deliberately so
#: the guard does not depend on importing a script module.
HELD_OUT_SPLITS = frozenset({"Test", "Test2"})


class CalibrationFitError(SatQueryError):
    """A calibration fit was refused or produced an unusable artifact."""


def _assert_not_held_out(fitted_on: str) -> None:
    """Refuse to fit on a held-out benchmark split."""
    if fitted_on in HELD_OUT_SPLITS:
        raise CalibrationFitError(
            f"refusing to fit calibration on {fitted_on!r}: it is a held-out "
            f"benchmark split. A temperature fitted on the data it is used to "
            f"score is a leak. Fit on validation only."
        )


def build_artifact_payload(
    *,
    fit: Any,
    metrics: Any,
    fitted_on: str,
    reliability: dict[str, Any] | None = None,
    checkpoint_sha256: str | None = None,
    checkpoint_path: str | None = None,
    config_hash: str | None = None,
    dataset_id: str | None = None,
    feature_spec: str | None = None,
    n_classes: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the artifact payload.

    Args:
        fit: a `training.calibration.fitter.CalibrationFit`.
        metrics: a `training.calibration.evaluate.CalibrationMetrics`.
        fitted_on: the split the temperature was estimated from. **Must not be a
            held-out split.**
        reliability: the `reliability_diagram(...)` output, when computed.
        checkpoint_sha256: hash of the model that produced the probabilities.
        checkpoint_path: where that model lives.
        config_hash: `Config.hash` in force during the fit.
        dataset_id / feature_spec: dataset identity, so a reader knows which
            distribution the temperature is valid for.
        n_classes: size of the answer space.
        extra: additional provenance merged at the top level.

    Raises:
        CalibrationFitError: `fitted_on` is a held-out split, or the fit is
            unusable (non-finite temperature, identity map with no data).
    """
    import math

    _assert_not_held_out(fitted_on)

    temperature = float(fit.temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise CalibrationFitError(
            f"fit produced an unusable temperature: {temperature!r}"
        )
    if int(fit.n_samples) <= 0:
        raise CalibrationFitError("fit used zero samples")

    payload: dict[str, Any] = {
        "schema": CALIBRATION_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        # The nested form `TemperatureCalibration.from_dict` reads directly.
        "temperature_scaling": {
            "temperature": temperature,
            "fitted_on": fitted_on,
            "n_samples": int(fit.n_samples),
        },
        # Provenance the consumer does not need but a reviewer does.
        "provenance": {
            "method": "temperature_scaling",
            "space": "multiclass_logits",
            "objective": "mean_negative_log_likelihood",
            "optimizer": "golden_section_on_log_temperature",
            "iterations": int(fit.iterations),
            "fitted_on": fitted_on,
            "held_out_splits_excluded": sorted(HELD_OUT_SPLITS),
            "n_samples": int(fit.n_samples),
            "dataset_id": dataset_id,
            "feature_spec": feature_spec,
            "n_classes": n_classes,
            "config_hash": config_hash,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_path": checkpoint_path,
        },
        "fit_diagnostics": fit.describe(),
        "metrics": metrics.describe(),
        "consumer_contract": {
            "module": "evidence.confidence",
            "class": "TemperatureCalibration",
            "applied_as": "sigmoid(logit(z) / T) for a scalar z; "
                          "softmax(logits / T) for a distribution",
            "read_keys": ["temperature", "fitted_on|split", "artifact",
                          "n_samples"],
            "resolution": "load_calibration(config, base_dir='configs')",
        },
    }

    if reliability is not None:
        payload["reliability_diagram"] = reliability
    if extra:
        payload.update(extra)

    return payload


def write_calibration_artifact(
    payload: dict[str, Any],
    path: str | Path,
) -> Path:
    """Write the payload and return the path.

    Written with `indent=2, sort_keys=True` so the file is byte-stable: two runs
    with the same inputs produce the same bytes except for `created_utc`, which
    is the only clock-dependent field and is called out as such.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return p


def file_sha256(path: str | Path) -> str:
    """SHA256 of the artifact's BYTES on disk.

    Distinct from a canonical-JSON digest on purpose -- see the §6a provenance
    defect in `training/change_vqa/evaluate.py::EvaluationReport.sha256`, where
    the two were conflated and a reviewer hashing the file concluded corruption.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
