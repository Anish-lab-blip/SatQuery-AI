"""SatQuery AI — confidence calibration (Phase 13).

WHERE THIS SITS
---------------
`docs/ARCHITECTURE_FREEZE.md` section 1 places confidence calibration directly
downstream of the evidence engine. Specialists produce a *raw* reliability score
from signals they can actually point at (`max_objectness`, input-quality
autocorrelation, `used_head`, ...). That raw score is not a probability: it is a
hand-weighted sum, and a hand-weighted sum is not calibrated by construction.
This module is the one place that maps raw -> calibrated.

THE HONESTY RULE (the reason this module exists at all)
-------------------------------------------------------
A calibration that claims to be fitted when it is not is a FALSE CLAIM OF
RELIABILITY -- strictly worse than reporting nothing, because a downstream
consumer will trust it. So:

    no fitted artifact  ->  pass the raw score through unchanged,
                            set `method="uncalibrated"`,
                            leave `calibrated=None`

`calibrated=None` is what makes it honest: `ConfidenceBreakdown.value` then
returns `raw`, and any consumer that wants to distinguish "we calibrated this"
from "we did not" can read `method` or `calibrated is None`. We never fill in a
plausible-looking number to make a schema field look complete.

Note the existing specialists already do the right thing on their own: both
`VLMSpecialist._confidence_for` and `GroundingSpecialist._confidence_for` emit
`method="uncalibrated", calibrated=None`. This module generalises that behaviour
and adds the fitted path.

NO LLM PATH EXISTS HERE
-----------------------
There is deliberately no function that turns text into a number. Confidence is
a measurement (freeze section 5: "No LLM-generated confidence"). The only inputs
accepted are a float the specialist computed and a calibration artifact fitted
on validation data.

TEMPERATURE SCALING AND ITS DERIVATION
--------------------------------------
The frozen config key is `confidence.temperature_scaling` (`configs/base.yaml`).
The standard formulation maps a logit through `sigmoid(logit / T)`. Our raw
scores are already in (0, 1), so they are read as probabilities and converted to
log-odds first:

    logit(z)   = log(z / (1 - z))
    calibrated = sigmoid(logit(z) / T)

Two consequences worth stating because they are the failure modes:

  T = 1.0  ->  identity. A "fitted" T of exactly 1.0 carries no information, so
               it is reported as uncalibrated rather than silently pretending to
               have done something. (`_is_effective` enforces this.)
  z = 0.0  ->  log-odds diverge. Both endpoints are clamped at the boundary
               values (0.0 and 1.0) so a hard zero cannot become a NaN.

A fitter is NOT included: Phase 13 fits T on validation data, and inventing one
here would be exactly the kind of unfounded number this module refuses to emit.
`TemperatureCalibration` is the *consumer* of a fitted `T`; the artifact loader
reads whatever `training/calibration/` produces.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.schemas import ConfidenceBreakdown

#: Method string reported when a fitted artifact was applied.
METHOD_TEMPERATURE = "temperature_scaling"

#: Method string reported whenever no mapping was applied. Matches the literal
#: the existing specialists already emit, so trace consumers see one vocabulary.
METHOD_UNCALIBRATED = "uncalibrated"

#: Scores are clamped to this closed interval before the log-odds transform.
#: `logit(0)` and `logit(1)` are infinite; clamping at the boundary keeps the
#: mapping finite and keeps a hard 0.0 / 1.0 from becoming a NaN.
_EPS = 1e-6

#: Temperature values are validated into this range by `__post_init__`. A
#: temperature of zero or below is not a calibration; it is a division error.
_MIN_TEMPERATURE = 1e-3
_MAX_TEMPERATURE = 1e3

#: How close to 1.0 a fitted temperature must land before it is treated as
#: "does nothing". Bit-for-bit 1.0 is the honest test; this tolerance absorbs
#: float round-trip through JSON without accepting a real transform.
_IDENTITY_TOLERANCE = 1e-6


def _sigmoid(x: float) -> float:
    """Numerically stable logistic function.

    The naive form overflows for large negative `x`; the branch below keeps
    `exp`'s argument non-positive so the result is finite for every input.
    """
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    """Log-odds of a probability, with the endpoints clamped to stay finite."""
    clamped = min(1.0 - _EPS, max(_EPS, p))
    return math.log(clamped / (1.0 - clamped))


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


def _is_effective(temperature: float) -> bool:
    """Does this temperature actually transform anything?

    A temperature within `_IDENTITY_TOLERANCE` of 1.0 is the identity map. It is
    treated as "not fitted" so the honest report wins: an artifact that carries
    T=1.0 has learned nothing, and reporting `method="temperature_scaling"` for
    it would assert a correction that was never made.
    """
    return abs(temperature - 1.0) > _IDENTITY_TOLERANCE


@dataclass(frozen=True)
class TemperatureCalibration:
    """A fitted temperature-scaling artifact.

    Attributes:
        temperature: the fitted scalar. `T > 1` softens (pulls scores toward the
            middle), `T < 1` sharpens. Validated on construction.
        fitted_on: free-text provenance, e.g. `"valid"` or a split hash. Carried
            into the breakdown components so a result can be traced back to the
            artifact that shaped it.
        artifact: path/identifier of the source file, for the same reason.
        n_samples: how many validation samples the fit used, when known.

    Raises:
        ValueError: if `temperature` is non-finite or outside
            `[_MIN_TEMPERATURE, _MAX_TEMPERATURE]`.
    """

    temperature: float
    fitted_on: str | None = None
    artifact: str | None = None
    n_samples: int | None = None

    def __post_init__(self) -> None:
        t = float(self.temperature)
        if not math.isfinite(t) or not _MIN_TEMPERATURE <= t <= _MAX_TEMPERATURE:
            raise ValueError(
                f"temperature must be finite and in "
                f"[{_MIN_TEMPERATURE}, {_MAX_TEMPERATURE}], got {self.temperature!r}"
            )
        object.__setattr__(self, "temperature", t)

    def is_effective(self) -> bool:
        """True when applying this calibration changes any score."""
        return _is_effective(self.temperature)

    def apply(self, raw: float) -> float:
        """Map a raw score in [0, 1] through the fitted temperature."""
        return _clamp01(_sigmoid(_logit(float(raw)) / self.temperature))

    @property
    def components(self) -> dict[str, float]:
        """The provenance block merged into a `ConfidenceBreakdown`."""
        components: dict[str, float] = {"temperature": self.temperature}
        if self.n_samples is not None:
            components["calibration_samples"] = float(self.n_samples)
        return components

    def describe(self) -> dict[str, Any]:
        return {
            "method": METHOD_TEMPERATURE,
            "temperature": self.temperature,
            "effective": self.is_effective(),
            "fitted_on": self.fitted_on,
            "artifact": self.artifact,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        source: str | Path | None = None,
    ) -> "TemperatureCalibration":
        """Build from a calibration artifact's parsed JSON.

        Accepts either the flat form (`{"temperature": 1.4, ...}`) or a nested
        `{"temperature_scaling": {...}}` form, because the artifact format is
        owned by `training/calibration/` and this loader should not be the thing
        that decides it.
        """
        body: Any = payload
        if isinstance(payload.get("temperature_scaling"), dict):
            body = payload["temperature_scaling"]
        elif isinstance(payload.get("calibration"), dict):
            body = payload["calibration"]

        if not isinstance(body, dict) or "temperature" not in body:
            raise ValueError(
                "calibration artifact has no 'temperature' field "
                f"(source: {source or '<dict>'})"
            )
        return cls(
            temperature=float(body["temperature"]),
            fitted_on=body.get("fitted_on") or body.get("split"),
            artifact=str(source) if source is not None else body.get("artifact"),
            n_samples=(
                int(body["n_samples"]) if body.get("n_samples") is not None else None
            ),
        )

    @classmethod
    def from_json(
        cls, path: str | Path, *, required: bool = False
    ) -> "TemperatureCalibration | None":
        """Load a calibration artifact from disk.

        Args:
            path: the artifact path. Relative paths are resolved against the
                current working directory, matching how the rest of the repo
                resolves config-relative filenames.
            required: when False (the default and the honest one), a missing or
                unreadable file returns `None` so the caller degrades to the
                uncalibrated pass-through instead of failing a run. When True, a
                missing file raises -- use it where calibration is a hard
                prerequisite.

        Raises:
            ValueError: the file exists and is malformed (never swallowed: a
                corrupt artifact silently treated as "no artifact" would hide an
                operational defect).
            FileNotFoundError: `required=True` and the file is absent.
        """
        p = Path(path)
        if not p.exists():
            if required:
                raise FileNotFoundError(f"calibration artifact not found: {p}")
            return None
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"calibration artifact is not valid JSON: {p}") from exc
        return cls.from_dict(payload, source=p)


def calibrate(
    raw: float,
    calibration: TemperatureCalibration | None,
    *,
    extra_components: dict[str, float] | None = None,
    degraded: bool = False,
    degradation_reason: str | None = None,
) -> ConfidenceBreakdown:
    """Map a raw specialist score through the fitted calibration, honestly.

    This is the single entry point used by `EvidenceEngine.confidence_for`.

    Args:
        raw: the specialist's raw reliability score, in [0, 1].
        calibration: a fitted artifact, or `None`. `None` -- or an artifact
            whose temperature is the identity -- produces the pass-through.
        extra_components: additional measured signals to merge into
            `ConfidenceBreakdown.components`.
        degraded: forwarded to the breakdown.
        degradation_reason: forwarded to the breakdown.

    Returns:
        A `ConfidenceBreakdown` where:

        * fitted and effective -> `calibrated` holds the mapped value,
          `method="temperature_scaling"`;
        * otherwise          -> `calibrated=None`, `method="uncalibrated"`, and
          `value` returns `raw` unchanged.

    Raises:
        ValueError: `raw` is not a finite number. Refusing is deliberate: a
            non-finite score is a bug upstream, and clamping it to 0.5 would
            invent a measurement.
    """
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"raw confidence must be finite, got {raw!r}")
    clamped = _clamp01(value)

    components: dict[str, float] = dict(extra_components or {})

    if calibration is None or not calibration.is_effective():
        # The honest path. `calibrated` stays None; `value` falls back to raw.
        components["calibrated_applied"] = 0.0
        if calibration is not None:
            # A real artifact was supplied but it is the identity map. Say so
            # rather than dropping the information on the floor.
            components["calibration_identity"] = 1.0
            components.update(calibration.components)
        return ConfidenceBreakdown(
            raw=clamped,
            calibrated=None,
            method=METHOD_UNCALIBRATED,
            components=components,
            degraded=degraded,
            degradation_reason=degradation_reason,
        )

    components["calibrated_applied"] = 1.0
    components.update(calibration.components)

    return ConfidenceBreakdown(
        raw=clamped,
        calibrated=calibration.apply(clamped),
        method=METHOD_TEMPERATURE,
        components=components,
        degraded=degraded,
        degradation_reason=degradation_reason,
    )


def calibrate_result(
    breakdown: ConfidenceBreakdown,
    calibration: TemperatureCalibration | None,
) -> ConfidenceBreakdown:
    """Calibrate an existing breakdown, preserving its degradation semantics.

    Used when a specialist has already produced its own
    `ConfidenceBreakdown` (both shipped specialists do) and the aggregate step
    wants to attach a calibration without re-deriving the components.

    The specialist's own components and degradation reason are carried through
    untouched: this function adds the calibration, it does not re-judge the
    specialist's measurement.
    """
    return calibrate(
        breakdown.raw,
        calibration,
        extra_components=dict(breakdown.components),
        degraded=breakdown.degraded,
        degradation_reason=breakdown.degradation_reason,
    )


def load_calibration(
    config: Any,
    *,
    base_dir: str | Path | None = None,
) -> TemperatureCalibration | None:
    """Resolve the calibration artifact named by the central config.

    Reads the frozen keys:

        confidence.temperature_scaling   (bool)  master switch
        confidence.calibration_file      (str)   artifact filename

    Returns `None` -- never a fabricated default -- when the switch is off, when
    no filename is configured, or when the artifact is absent. Every one of
    those is a legitimate deployment state (an HF Space may ship without the
    fitted artifact), and each degrades to the pass-through.

    WHERE THE FILE IS LOOKED FOR
    ----------------------------
    `base_dir` is resolved in this order (added 2026-09-22):

    1. an explicit `base_dir` argument, when the caller passes one;
    2. `configs/` under the repository root, **but only if the artifact is
       actually there** -- it never is by default, so this branch exists only to
       keep an existing caller that parks the artifact beside the config working;
    3. `<repo root>/artifacts/` -- the artifact's real home, and the location
       `scripts/fit_calibration.py` writes to.

    Step 3 is the important one and it is why this function changed. The frozen
    config names a *bare filename* (`calibration_v001.json`), while the docstring
    examples in this repository's own docs pass `base_dir="."` AND
    `base_dir="configs"` -- two different directories, neither of which is
    `artifacts/`. A deployment following either example would silently resolve
    to a nonexistent path, degrade to `method="uncalibrated"`, and report a
    fitted artifact as never having been fitted. That is precisely the
    silent-failure shape this project's provenance discipline exists to prevent,
    so the search now anchors to the repository root using the same `REPO_ROOT`
    that `app/serving.py` uses for the change checkpoint.

    The search is ordered and deterministic: the first existing candidate wins,
    and no candidate existing means `None` (honest degradation, not an error).

    Raises:
        ValueError: the located file is malformed. A corrupt artifact is never
            swallowed -- treating it as "no artifact" would hide an operational
            defect.
    """
    if not bool(config.get("confidence.temperature_scaling", False)):
        return None

    filename = config.get("confidence.calibration_file")
    if not filename:
        return None

    name = str(filename)
    candidates: list[Path] = []

    if base_dir is not None:
        # An explicit base_dir is honoured first, but a REPO-anchored candidate
        # is still tried: several call sites in this repo pass "." meaning "the
        # repo", which only works when CWD happens to be the repo root.
        candidates.append(Path(base_dir) / name)

    from core.config import REPO_ROOT  # local import: keeps this module cheap

    if base_dir is not None:
        candidates.append(Path(REPO_ROOT) / name)
    candidates.append(Path(REPO_ROOT) / "artifacts" / name)
    candidates.append(Path(REPO_ROOT) / "configs" / name)

    for candidate in candidates:
        if candidate.exists():
            return TemperatureCalibration.from_json(candidate, required=True)

    # Nothing found: the master switch is on but no artifact ships. Degrade.
    return None


__all__ = [
    "METHOD_TEMPERATURE",
    "METHOD_UNCALIBRATED",
    "TemperatureCalibration",
    "calibrate",
    "calibrate_result",
    "load_calibration",
]
