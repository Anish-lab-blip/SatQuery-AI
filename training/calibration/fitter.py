"""Temperature-scaling fitter — estimates `T` on validation data.

THE MATH, AND WHY IT IS WRITTEN OUT
-----------------------------------
`evidence/confidence.py` applies its temperature in PROBABILITY space:

    calibrated = sigmoid( logit(z) / T ),   logit(z) = log(z / (1 - z))

That is binary-temperature scaling. For a multi-class softmax the standard
formulation temperatures the LOGITS before the softmax:

    p_i(T) = softmax(logits / T)_i

These are the same operation only in the two-class case. This module therefore
fits `T` by **numerically minimising the multi-class NLL on the validation
logits**, and does NOT reuse the binary closed-form identity -- deriving `T` one
way and applying it another is how a "fitted" number ends up meaning nothing.

`T` is recovered from probabilities by `logits = log(p)`, which is exact when the
probabilities came from a softmax produced by this model's `predict_answers`
(they do: `probabilities` is the full `(B, 19)` softmax). Up to the additive
constant that softmax removes, `log(p)` differs from the true logits by a
per-row shift, and `softmax(z / T)` is invariant to that shift -- so optimising
on `log(p) / T` recovers the same `T` as optimising on `z / T`.

WHY SCIPY IS NOT USED
---------------------
Not a dependency of this project, and adding one for a one-dimensional convex
minimisation would be a worse trade than 30 lines of golden-section search. NLL
in `T` for a softmax is convex in `log T` (it is a log-partition function of a
scaled exponential family), so a bracketed golden-section search on `log T`
converges to the global optimum. The search is deterministic: fixed bounds, fixed
iteration count, no RNG, no clock. Two runs on the same inputs produce the same
`T` bit for bit, which is what the reproducibility discipline requires.

BOUNDS
------
`log T` is searched over `[log(1e-3), log(1e3)]`, matching the validation range
`TemperatureCalibration.__post_init__` enforces. A fit that lands on a bound is
reported as `hit_bound=True` -- an honest signal that the data wanted to go
further, rather than a silent clamp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

#: Search bounds on `T`. The upper/lower limits mirror
#: `evidence/confidence.py::_MIN_TEMPERATURE` / `_MAX_TEMPERATURE` so a fitted
#: value can always be constructed into a `TemperatureCalibration`.
MIN_TEMPERATURE = 1e-3
MAX_TEMPERATURE = 1e3

#: Golden-section iterations. The interval shrinks by ~0.618 per step; 200 steps
#: drives a 13.8-natural-log range (log 1e3 - log 1e-3) below 1e-40, i.e. far
#: past float64 resolution. 200 costs microseconds and makes the result
#: independent of any tolerance the caller might pick.
_ITERATIONS = 200

#: NLL improvement below which the objective is treated as flat, so the fit
#: returns the identity rather than a bound. See the degeneracy note in
#: `fit_temperature`. 1e-12 is far below any real signal (the R-02 fit on
#: 16,441 validation rows moved NLL by 1.1e-4) and far above float64 noise on
#: a mean over thousands of terms.
_DEGENERATE_TOLERANCE = 1e-12


def softmax(logits: Any) -> Any:
    """Numerically stable softmax along the last axis.

    Subtracting the row max keeps `exp` in a safe range; the result is identical
    to the naive form (softmax is shift-invariant) without the overflow.
    """
    import numpy as np

    z = np.asarray(logits, dtype=np.float64)
    shifted = z - np.max(z, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def logits_from_probabilities(probabilities: Any, *, eps: float = 1e-12) -> Any:
    """Recover logits from a softmax output via `log(p)`.

    Clamped at `eps` so a hard zero (which a softmax can produce when a logit
    leads by a wide margin in float32) cannot become `-inf` and poison the NLL.

    The recovered values differ from the model's true logits by a per-row
    additive constant. That is harmless here: `softmax(z / T)` is invariant to
    it, so the fitted `T` is the same either way.
    """
    import numpy as np

    p = np.asarray(probabilities, dtype=np.float64)
    return np.log(np.clip(p, eps, 1.0))


def _nll(logits: Any, targets: Any, log_temperature: float) -> float:
    """Mean negative log-likelihood of `targets` under `softmax(logits / T)`."""
    import numpy as np

    scaled = logits / math.exp(log_temperature)
    # log-softmax, computed stably: log p_i = scaled_i - logsumexp(scaled)
    row_max = np.max(scaled, axis=-1, keepdims=True)
    shifted = scaled - row_max
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1)) + row_max[:, 0]
    log_prob = scaled[np.arange(len(targets)), targets] - log_sum_exp
    return float(-np.mean(log_prob))


@dataclass(frozen=True)
class CalibrationFit:
    """The outcome of a fit, with the diagnostics needed to judge it.

    Attributes:
        temperature: the fitted scalar.
        n_samples: validation rows the fit used.
        nll_before: mean NLL at `T = 1` (the uncalibrated baseline).
        nll_after: mean NLL at the fitted `T`.
        log_temperature: the parameter actually optimised.
        hit_bound: True when the optimum sits on a search bound -- the data
            wanted to move further than the allowed range. Reported, not hidden.
        iterations: golden-section steps taken (fixed, for reproducibility).
    """

    temperature: float
    n_samples: int
    nll_before: float
    nll_after: float
    log_temperature: float
    hit_bound: bool
    iterations: int

    @property
    def improvement(self) -> float:
        """NLL reduction from calibrating. Negative means the fit made it worse."""
        return self.nll_before - self.nll_after

    @property
    def is_effective(self) -> bool:
        """True when the fitted temperature is not the identity map.

        Mirrors `evidence.confidence._is_effective` so the fitter and the
        consumer agree on what "fitted" means. A `T` within 1e-6 of 1.0 carries
        no information, and the consumer would report it `uncalibrated` anyway.
        """
        return abs(self.temperature - 1.0) > 1e-6

    def describe(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "n_samples": self.n_samples,
            "nll_before": self.nll_before,
            "nll_after": self.nll_after,
            "nll_improvement": self.improvement,
            "log_temperature": self.log_temperature,
            "hit_bound": self.hit_bound,
            "iterations": self.iterations,
            "effective": self.is_effective,
        }


def fit_temperature(
    probabilities: Any,
    targets: Sequence[int],
    *,
    min_temperature: float = MIN_TEMPERATURE,
    max_temperature: float = MAX_TEMPERATURE,
) -> CalibrationFit:
    """Fit a temperature by minimising multi-class NLL on validation data.

    Args:
        probabilities: `(N, C)` softmax outputs from `predict_answers`.
        targets: `N` gold class indices.
        min_temperature: lower search bound.
        max_temperature: upper search bound.

    Returns:
        A `CalibrationFit`. `nll_before` is computed at `T = 1` so the caller can
        report what the fit actually bought.

    Raises:
        ValueError: on empty inputs, a shape mismatch, an out-of-range target, or
            non-finite probabilities. Refusing is deliberate -- a fit silently
            performed on malformed data is an invented number.
    """
    import numpy as np

    probs = np.asarray(probabilities, dtype=np.float64)
    gold = np.asarray(list(targets), dtype=np.int64)

    if probs.ndim != 2:
        raise ValueError(f"probabilities must be 2-D (N, C), got shape {probs.shape}")
    if len(gold) == 0:
        raise ValueError("cannot fit a temperature on zero samples")
    if len(gold) != probs.shape[0]:
        raise ValueError(
            f"probabilities has {probs.shape[0]} rows but {len(gold)} targets"
        )
    if not np.isfinite(probs).all():
        raise ValueError("probabilities contain non-finite values")
    if gold.min() < 0 or gold.max() >= probs.shape[1]:
        raise ValueError(
            f"targets must be in [0, {probs.shape[1]}), "
            f"got range [{gold.min()}, {gold.max()}]"
        )

    logits = logits_from_probabilities(probs)
    nll_before = _nll(logits, gold, 0.0)

    lo = math.log(min_temperature)
    hi = math.log(max_temperature)
    # Golden-section search on the convex NLL in log T.
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - inv_phi * (hi - lo)
    d = lo + inv_phi * (hi - lo)
    fc = _nll(logits, gold, c)
    fd = _nll(logits, gold, d)
    for _ in range(_ITERATIONS):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - inv_phi * (hi - lo)
            fc = _nll(logits, gold, c)
        else:
            lo, c, fc = c, d, fd
            d = lo + inv_phi * (hi - lo)
            fd = _nll(logits, gold, d)

    log_t = 0.5 * (lo + hi)
    temperature = math.exp(log_t)
    # Clamp to the constructible range: the search bounds are inclusive but a
    # float round-trip through exp can land a hair outside.
    temperature = min(max_temperature, max(min_temperature, temperature))
    nll_after = _nll(logits, gold, math.log(temperature))

    # DEGENERATE OBJECTIVE
    # -------------------
    # A flat NLL surface has no interior optimum, and golden-section search on a
    # flat function walks to whichever bound it happens to favour. This is not
    # hypothetical: a constant (uniform) score vector is EXACTLY shift-invariant
    # under temperature -- softmax(1/T) is the same uniform distribution for
    # every T -- so its NLL is identical everywhere and the search returns
    # `max_temperature` (1000) for data that wanted no change at all.
    #
    # Reporting `T = 1000` there would be a fabricated correction: the consumer
    # would stamp `method="temperature_scaling"` on a mapping that does nothing.
    # When the objective is flat to within numerical noise the honest answer is
    # the identity, so that is what is returned. `hit_bound` still records what
    # the search did, so a caller can see the degeneracy.
    if abs(nll_after - nll_before) <= _DEGENERATE_TOLERANCE:
        temperature = 1.0
        log_t = 0.0
        nll_after = nll_before

    tol = 1e-9
    return CalibrationFit(
        temperature=temperature,
        n_samples=int(len(gold)),
        nll_before=nll_before,
        nll_after=nll_after,
        log_temperature=log_t,
        hit_bound=(
            abs(log_t - math.log(min_temperature)) < tol
            or abs(log_t - math.log(max_temperature)) < tol
        ),
        iterations=_ITERATIONS,
    )
