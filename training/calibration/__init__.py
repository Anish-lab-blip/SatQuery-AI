"""SatQuery AI — confidence calibration fitting (Phase 14 / R-03).

WHAT THIS PACKAGE DOES
----------------------
`evidence/confidence.py` is the *consumer* of a fitted temperature: it maps a raw
specialist score through `sigmoid(logit(z) / T)`. Its own docstring says the
artifact loader "reads whatever `training/calibration/` produces". Until now that
directory held only a `.gitkeep`, so no artifact was ever produced, no
`artifacts/calibration_v001.json` existed, and every confidence the system
emitted said `method="uncalibrated"`. That is the single §67 immutable-decision
violation ("calibrated confidence").

This package is the missing producer. It fits `T` on VALIDATION data only and
writes a provenance-carrying artifact at the path the frozen config already
names (`confidence.calibration_file: calibration_v001.json`).

THE ONE RULE THAT MATTERS
-------------------------
**Fit on validation. Never on Test or Test2.**

The whole point of temperature scaling is that `T` is estimated from data the
model did not train on, and then honestly applied to data nobody tuned on. Fitting
on a held-out benchmark split would convert a calibration artifact into a leak:
the reported confidence would be fitted against the very answers it is used to
score. `fitter.py` therefore takes an explicit `fitted_on` split name and
**refuses** any split in `HELD_OUT_SPLITS`, rather than trusting the caller.

WHAT IS *NOT* HERE
------------------
No quality threshold. The plan does not specify one, so none is invented:
`evaluate.py` computes ECE and NLL and reports them; deciding whether a given
value is "good enough" is a maintainer ruling, exactly like the accuracy /
macro-F1 question in R-02.

No second calibration method. Temperature scaling is what the frozen config
(`confidence.temperature_scaling`) and the frozen math in
`evidence/confidence.py` describe. Adding a competing method would create two
possible answers to "what is this confidence", which is the ambiguity this
project's provenance discipline exists to prevent.
"""

from __future__ import annotations

from training.calibration.fitter import (
    CalibrationFit,
    fit_temperature,
    logits_from_probabilities,
    softmax,
)
from training.calibration.evaluate import (
    CalibrationMetrics,
    expected_calibration_error,
    mean_negative_log_likelihood,
    reliability_diagram,
)
from training.calibration.artifact import (
    CALIBRATION_SCHEMA,
    DEFAULT_ARTIFACT_NAME,
    build_artifact_payload,
    write_calibration_artifact,
)

__all__ = [
    "CALIBRATION_SCHEMA",
    "DEFAULT_ARTIFACT_NAME",
    "CalibrationFit",
    "CalibrationMetrics",
    "build_artifact_payload",
    "expected_calibration_error",
    "fit_temperature",
    "logits_from_probabilities",
    "mean_negative_log_likelihood",
    "reliability_diagram",
    "softmax",
    "write_calibration_artifact",
]
