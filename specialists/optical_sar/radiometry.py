"""SatQuery AI — CROMA encoder-input radiometry (DEV-2 ruling).

WHY THIS MODULE EXISTS
----------------------
`configs/base.yaml` declares two radiometric transforms:

    optical: {normalization: percentile, lower_percentile: 2, upper_percentile: 98}
    sar:     {representation: db, clip_min_db: -30, clip_max_db: 5}

Neither is read by any code. CROMA's released inference wrapper
(`vendor/use_croma.py`, sha256 `a38567beed29eb08`) applies **no** input scaling
either -- verified first-hand: the module defines no `normalize`, and a grep for
`normal|255|mean|std|percentile|clip|uint8` returns only internal `nn.LayerNorm`
layers and `mean(dim=1)` GAP calls.

So the encoder was receiving whatever dynamic range the caller happened to hand
it. That is a silent distribution shift away from the `[0, 1]`-ish input the
frozen pretrained weights were fit under. `docs/PHASE14_CROMA_NORMALISATION_CHANGE.md`
records the ruling that closes it; this module implements it.

WHAT IS IMPLEMENTED HERE, AND WHAT IS NOT
-----------------------------------------
Implemented: the **encoder-input** stage -- per-channel `mean +/- 2*std` ->
`[0, 1]`, the transform the CROMA authors' own README instructs users to apply
(`https://github.com/antofuller/CROMA` README; corroborated by the authors'
instruction for their released benchmark tensors: *"convert tensors to floats and
divide by 255"*).

NOT implemented: the percentile / dB **conditioning** stage that `base.yaml`
names. The ruling (`PHASE14_CROMA_NORMALISATION_CHANGE.md` section 2.2) states the
two are ordered stages, not alternatives, and that both must run. The second is
specified here; the first remains unspecified and unsourced -- no source examined
in `docs/CROMA_NORMALISATION_UPSTREAM_EVIDENCE.md` uses or endorses a percentile
stretch or a dB clip. Implementing a guess for it is exactly the fabrication the
DEV-2 ruling exists to prevent, so it is left visibly absent rather than silently
approximated. **Consequence: the two `optical.*` and three `sar.*` config keys are
still read by no code.** That is recorded, not fixed.

WHERE THIS SITS
---------------
    GeoTIFF -> band map + zero-fill + availability mask   (sensor_adapter)
            -> [percentile / dB conditioning]             (NOT IMPLEMENTED)
            -> per-channel mean +/- 2*std -> [0, 1]       <- THIS MODULE
            -> CROMA.encode

THE TRANSFORM
-------------
Verbatim from the CROMA README, per channel `c`:

    min_value = x[:, c].mean() - 2 * x[:, c].std()
    max_value = x[:, c].mean() + 2 * x[:, c].std()
    img = (x[:, c] - min_value) / (max_value - min_value) * 255.0
    img = clip(img, 0, 255).to(uint8)
    # before the forward pass:
    x = x.float() / 255

DELIBERATE DEVIATION 1 -- per sample, not per batch
---------------------------------------------------
The README computes `x[:, c].mean()` over the **whole batch**, so at `N > 1` the
window depends on what else is in the batch. That makes a single image's encoding
a function of its neighbours, which is unacceptable for a serving system whose
whole point is reproducibility: the same query must not produce a different
answer because another request was batched alongside it.

This module computes the window **per sample**, over that sample's channel only.
`docs/PHASE14_CROMA_NORMALISATION_CHANGE.md` section 3.1.1 already specifies the
`(C, H, W)` tensor -- i.e. one sample -- so this is the ruling's own reading, not
a reinterpretation. At `N = 1` the two are identical; the README's own example
runs at `N = 1`.

DELIBERATE DEVIATION 2 -- unbiased standard deviation
----------------------------------------------------
Upstream is torch, whose `.std()` defaults to the **unbiased** estimator
(`ddof=1`). NumPy's `.std()` defaults to `ddof=0`. This module uses `ddof=1` to
match the reference implementation rather than the more convenient NumPy
default. At 120x120 the two differ by a factor of ~1.00003, so the choice is
numerically immaterial -- it is made for fidelity, and stated so nobody has to
guess which convention a number came from.

THE ZERO-CHANNEL RULE (the load-bearing part)
---------------------------------------------
An unavailable channel is **all zeros** -- that is the sensor adapter's
convention, and the same rule the fusion head's dropout obeys. Such a channel has
`std = 0`, so `max_value - min_value == 0` and the stretch divides by zero.

Unavailable channels are therefore **skipped and left at exactly zero**. They are
never normalised and never given a fabricated scale. This is the C-1 discipline
applied to radiometry: the same rule that forbids inventing a band forbids
inventing a dynamic range for a band that does not exist.

The degenerate-window test is the mechanism, and it needs no mask. That is
deliberate: `CROMAEncoder.encode` has a **pinned** signature of exactly
`(self, optical, sar)` -- `tests/unit/test_optical_sar_croma.py` asserts the
absence of any mask parameter, because freeze finding C-1 says CROMA never
receives one. Inferring unavailability from the data keeps that guard intact
while still honouring the rule, and it is robust to a caller that forgets to pass
a mask.

A channel that is present but *constant* (a flat band) also has a degenerate
window. It is skipped by the same rule, because no defensible scale exists for it
either, and it is recorded with a **different status** (`degenerate`) so the trace
distinguishes "the sensor did not measure this" from "the sensor measured a flat
band". Nothing is invented in either case.

WHAT THIS MODULE CANNOT CLAIM
-----------------------------
That the transform **matches** CROMA's pretraining input distribution. The author
states the pretraining dataloader "followed SatMAE's data preprocessing" and that
it "was specific to the hardware I used" -- and it was **never released**
(`CROMA_NORMALISATION_UPSTREAM_EVIDENCE.md` sections 2.6 and 5.3). The exact
pretraining distribution is therefore not recoverable from public sources.

This is the transform the authors' own code instructs users to apply, adopted as
the best-supported default. It is not a verified match, and the gated experiment
in `PHASE14_CROMA_NORMALISATION_CHANGE.md` section 4 is the only mechanism that
could ever turn it into one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Identifier recorded on the descriptor and in every trace. Names the *shape*
#: of the transform, not a claim about its provenance.
TRANSFORM_NAME = "per_channel_mean_pm_2std"

#: The window half-width. Upstream hardcodes `2`; named here so the gated
#: experiment's variant arms can vary it without a code edit going unnoticed.
WINDOW_SIGMAS = 2.0

#: `use_8_bit` per `PHASE14_CROMA_NORMALISATION_CHANGE.md` section 2.2
#: ("Decision on `use_8_bit`: ENABLED"). The uint8 round-trip guarantees the
#: `[0, 1]` range the frozen encoder requires; the float path's clip is applied
#: to a stretch that is unbounded by construction (`mean +/- 2*std` is not a
#: min/max clamp).
DEFAULT_USE_8_BIT = True

#: Hash-exempt override, in the pattern this repo already blessed for
#: `SATQUERY_DEVICE` (`core/config.py:86-91`). Needed because adding
#: `croma.use_8_bit` to `configs/base.yaml` would move `Config.hash` away from
#: the frozen `78f1e3700da15aa1` and detach the Phase 9 benchmark -- the exact
#: trap recorded in `docs/ARCHITECTURE_CHANGE_CHANGE_SERVING_WIRING.md` section
#: 4.1. See `resolve_use_8_bit` for the full resolution order.
ENV_USE_8_BIT = "SATQUERY_CROMA_USE_8_BIT"

#: Channel status values. `unavailable` is the masked-off / zero-fill case;
#: `degenerate` is a present-but-constant band; `normalised` is the real case.
STATUS_NORMALISED = "normalised"
STATUS_UNAVAILABLE = "unavailable"
STATUS_DEGENERATE = "degenerate"

#: Below this many finite pixels a standard deviation is not meaningful.
MIN_FINITE_PIXELS = 2

#: Modality labels, so a trace says which input a window belongs to.
MODALITY_OPTICAL = "optical"
MODALITY_SAR = "sar"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelWindow:
    """The window actually applied to one channel of one sample.

    `lower` / `upper` are `None` for a skipped channel -- deliberately, because a
    skipped channel has no window, and recording `0.0` there would read as a real
    measurement of a zero-width range.
    """

    sample: int
    channel: int
    status: str
    lower: float | None
    upper: float | None
    finite_pixels: int

    @property
    def skipped(self) -> bool:
        return self.status != STATUS_NORMALISED

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample": self.sample,
            "channel": self.channel,
            "status": self.status,
            "lower": self.lower,
            "upper": self.upper,
            "finite_pixels": self.finite_pixels,
        }


@dataclass(frozen=True)
class RadiometryReport:
    """What the encoder actually received, per modality.

    Ruling section 3.1.4 requires the applied transform, `use_8_bit` and the
    per-channel `(min, max)` window to reach the trace. This is that record.
    """

    modality: str
    transform: str
    use_8_bit: bool
    n_samples: int
    n_channels: int
    windows: tuple[ChannelWindow, ...]

    @property
    def skipped(self) -> tuple[ChannelWindow, ...]:
        return tuple(w for w in self.windows if w.skipped)

    @property
    def normalised_channels(self) -> tuple[int, ...]:
        """Channels normalised in **every** sample of the batch.

        A channel normalised in only some samples is not listed: the point of
        this property is to answer "did this channel carry real signal?", and
        "sometimes" is not an answer to that.
        """
        by_channel: dict[int, set[bool]] = {}
        for w in self.windows:
            by_channel.setdefault(w.channel, set()).add(not w.skipped)
        return tuple(
            c for c in sorted(by_channel) if by_channel[c] == {True}
        )

    @property
    def unavailable_channels(self) -> tuple[int, ...]:
        by_channel: dict[int, set[bool]] = {}
        for w in self.windows:
            by_channel.setdefault(w.channel, set()).add(
                w.status == STATUS_UNAVAILABLE
            )
        return tuple(c for c in sorted(by_channel) if by_channel[c] == {True})

    @property
    def degenerate_channels(self) -> tuple[int, ...]:
        by_channel: dict[int, set[bool]] = {}
        for w in self.windows:
            by_channel.setdefault(w.channel, set()).add(
                w.status == STATUS_DEGENERATE
            )
        return tuple(c for c in sorted(by_channel) if by_channel[c] == {True})

    def to_dict(self, *, max_windows: int = 64) -> dict[str, Any]:
        """Bounded summary. Never silently truncates -- sets `truncated`."""
        windows = [w.to_dict() for w in self.windows[:max_windows]]
        return {
            "modality": self.modality,
            "transform": self.transform,
            "use_8_bit": self.use_8_bit,
            "n_samples": self.n_samples,
            "n_channels": self.n_channels,
            "normalised_channels": list(self.normalised_channels),
            "unavailable_channels": list(self.unavailable_channels),
            "degenerate_channels": list(self.degenerate_channels),
            "windows": windows,
            "windows_total": len(self.windows),
            "truncated": len(self.windows) > len(windows),
        }


@dataclass(frozen=True)
class RadiometrySummary:
    """Both modalities, plus the settings that produced them."""

    use_8_bit: bool
    transform: str
    optical: RadiometryReport
    sar: RadiometryReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform": self.transform,
            "use_8_bit": self.use_8_bit,
            "optical": self.optical.to_dict(),
            "sar": self.sar.to_dict(),
        }


# ---------------------------------------------------------------------------
# Configuration resolution
# ---------------------------------------------------------------------------


def resolve_use_8_bit(config: Any | None = None) -> tuple[bool, str]:
    """Resolve `use_8_bit`, and say where the value came from.

    Resolution order, most specific first:

    1. `SATQUERY_CROMA_USE_8_BIT` environment variable, when set.
    2. `croma.use_8_bit` in the config, **when that key exists and is a bool**.
    3. `DEFAULT_USE_8_BIT` (`True`, per the ruling).

    Step 2 is written so it starts working the moment the key is added, and is a
    no-op until then -- `Config.get` returns its default for a missing key
    (`core/config.py:57-64`).

    WHY THIS IS NOT JUST `configs/base.yaml`
    ----------------------------------------
    `Config.hash` is a sha256 over the whole registry with no exclusion
    mechanism (`core/config.py:76-80`). Adding one key moves it. The shipped
    Phase 9 checkpoint records `78f1e3700da15aa1`, and `scripts/eval_change.py`
    refuses to score on drift (exit 3), so writing the key into `base.yaml` today
    would detach the project's benchmark number from its config.

    That is a real cost and it is the user's decision to take, not a silent one
    to make in a normalisation patch. Until it is taken, this resolution order
    gives the ruling's behaviour (`use_8_bit` enabled, inspectable, reversible)
    through a hash-exempt channel -- the same pattern already used for
    `SATQUERY_DEVICE`. The value is recorded in every trace either way, so it is
    never invisible.

    Returns:
        `(value, source)` where `source` is `"env"`, `"config"` or `"default"`.
    """
    env = os.environ.get(ENV_USE_8_BIT)
    if env is not None and env.strip():
        return env.strip().lower() in {"1", "true", "yes", "on"}, "env"

    if config is not None:
        value = config.get("croma.use_8_bit", None)
        if isinstance(value, bool):
            return value, "config"

    return DEFAULT_USE_8_BIT, "default"


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------


def _channel_window(
    values: np.ndarray,
) -> tuple[float, float, int] | None:
    """`(lower, upper, n_finite)` for one channel, or None if undefined.

    `values` is 2-D `(H, W)`. Returns None when the window cannot be computed
    without inventing a scale -- no finite pixels, or a zero-width window.
    """
    finite = np.isfinite(values)
    n_finite = int(finite.sum())
    if n_finite < MIN_FINITE_PIXELS:
        return None

    # float64 for the statistics: a 120x120 float32 channel can accumulate
    # enough error in `mean` to matter once it is divided by a narrow window.
    sample = values[finite].astype(np.float64, copy=False)
    mean = float(sample.mean())
    # ddof=1 to match torch's default, which is what upstream uses. See the
    # module docstring, "DELIBERATE DEVIATION 2".
    std = float(sample.std(ddof=1))

    lower = mean - WINDOW_SIGMAS * std
    upper = mean + WINDOW_SIGMAS * std
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None
    if upper - lower <= 0.0:
        # Zero-width window: every finite pixel is identical. Dividing by this
        # would be division by zero; substituting a scale would fabricate one.
        return None
    return lower, upper, n_finite


def normalise_for_croma(
    array: np.ndarray,
    *,
    use_8_bit: bool = DEFAULT_USE_8_BIT,
    modality: str = MODALITY_OPTICAL,
) -> tuple[np.ndarray, RadiometryReport]:
    """Apply the CROMA encoder-input stretch. Returns `(array, report)`.

    Args:
        array: `(B, C, H, W)` or `(C, H, W)` float. Missing channels are expected
            to be exactly zero, per the sensor adapter's convention.
        use_8_bit: the uint8 round-trip. `True` guarantees the output lies in
            `{0/255, ..., 1}`; `False` clips to `[0, 1]` directly.
        modality: label for the report (`"optical"` or `"sar"`).

    Returns:
        A new array of the same shape and dtype (`float32`), plus the report.
        Skipped channels are **exactly zero** in the output.

    Raises:
        ValueError: `array` is not 3-D or 4-D.

    The function is pure and deterministic: no sampling, no learned statistics,
    no RNG, and no dependence on batch composition. Same input -> same output.
    """
    arr = np.asarray(array)
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 4:
        raise ValueError(
            f"radiometry expects (B, C, H, W) or (C, H, W); got {arr.shape}"
        )

    source = arr.astype(np.float32, copy=False)
    out = np.zeros_like(source, dtype=np.float32)
    n_samples, n_channels = int(source.shape[0]), int(source.shape[1])
    windows: list[ChannelWindow] = []

    for b in range(n_samples):
        for c in range(n_channels):
            channel = source[b, c]

            # An all-zero channel is the sensor adapter's unavailability
            # convention. Checked first so the status says *why* it was skipped:
            # "the sensor did not measure this" is not the same fact as "the
            # sensor measured a flat band", and the trace must not merge them.
            if not np.any(channel):
                windows.append(
                    ChannelWindow(
                        sample=b,
                        channel=c,
                        status=STATUS_UNAVAILABLE,
                        lower=None,
                        upper=None,
                        finite_pixels=0,
                    )
                )
                continue

            window = _channel_window(channel)
            if window is None:
                windows.append(
                    ChannelWindow(
                        sample=b,
                        channel=c,
                        status=STATUS_DEGENERATE,
                        lower=None,
                        upper=None,
                        finite_pixels=int(np.isfinite(channel).sum()),
                    )
                )
                continue

            lower, upper, n_finite = window

            # A non-finite pixel is a defective measurement, not a measurement
            # of zero, and it has no place in a linear stretch. It is pinned to
            # 0.0 -- the same value the sensor adapter uses for "nothing was
            # measured here" -- rather than left to propagate NaN into the
            # frozen encoder, where one NaN would poison the whole forward pass
            # (and, on the uint8 path, raise a cast warning even when the clip
            # happens to bound it). `finite_pixels` in the report keeps the gap
            # detectable: fewer finite pixels than the channel's area means the
            # stretch was computed over a subset.
            finite = np.isfinite(channel)
            stretched = np.zeros(channel.shape, dtype=np.float64)
            stretched[finite] = (
                channel[finite].astype(np.float64) - lower
            ) / (upper - lower)

            if use_8_bit:
                # Upstream's exact path: scale to 0..255, clip, quantise to
                # uint8, then divide by 255 before the forward pass. The
                # quantisation is a bounded, uniform error (~1/256 of full
                # scale) and it is what *guarantees* the range invariant.
                quantised = np.clip(stretched * 255.0, 0.0, 255.0).astype(
                    np.uint8
                )
                out[b, c] = quantised.astype(np.float32) / 255.0
            else:
                out[b, c] = np.clip(stretched, 0.0, 1.0).astype(np.float32)

            windows.append(
                ChannelWindow(
                    sample=b,
                    channel=c,
                    status=STATUS_NORMALISED,
                    lower=lower,
                    upper=upper,
                    finite_pixels=n_finite,
                )
            )

    report = RadiometryReport(
        modality=modality,
        transform=TRANSFORM_NAME,
        use_8_bit=bool(use_8_bit),
        n_samples=n_samples,
        n_channels=n_channels,
        windows=tuple(windows),
    )
    return out, report


def warnings_for(report: RadiometryReport) -> list[str]:
    """Operator-facing notes for anything skipped for a reason other than absence.

    A masked-off channel is the expected case and produces no warning -- Cartosat-2S
    always has 8 of 12 optical channels absent, and warning about it every request
    would be noise. A *present* channel that could not be stretched is a data
    defect and must be visible.
    """
    notes: list[str] = []
    degenerate = report.degenerate_channels
    if degenerate:
        notes.append(
            f"{report.modality}: channel(s) {list(degenerate)} are present but "
            f"have a zero-width dynamic range (every finite pixel identical), so "
            f"no per-channel mean +/- 2*std window exists. They were left at "
            f"zero rather than given a fabricated scale."
        )
    return notes


def describe_transform(use_8_bit: bool, *, source: str | None = None) -> dict[str, Any]:
    """The transform's identity, for the trace and the encoder descriptor."""
    described: dict[str, Any] = {
        "transform": TRANSFORM_NAME,
        "window_sigmas": WINDOW_SIGMAS,
        "use_8_bit": bool(use_8_bit),
        "range": "[0, 1]",
        "zero_channel_policy": "skipped and left at zero; no scale is invented",
        "per_sample": True,
        "std_ddof": 1,
        "source": "CROMA README normalize() -- the authors' instructed transform",
    }
    if source is not None:
        described["use_8_bit_source"] = source
    return described


def summarise(
    optical_report: RadiometryReport, sar_report: RadiometryReport
) -> RadiometrySummary:
    """Bundle both modalities under one `use_8_bit`.

    They must agree: the ruling requires the identical stretch on both, and the
    vendored `use_croma.py` uses one `ViT` class for both modalities with no
    divergence (`vendor/use_croma.py:54,68`). Asserted rather than assumed.
    """
    if optical_report.use_8_bit != sar_report.use_8_bit:
        raise ValueError(
            "optical and SAR radiometry disagree on use_8_bit "
            f"({optical_report.use_8_bit} vs {sar_report.use_8_bit}); upstream "
            "applies one transform to both"
        )
    return RadiometrySummary(
        use_8_bit=optical_report.use_8_bit,
        transform=TRANSFORM_NAME,
        optical=optical_report,
        sar=sar_report,
    )


__all__ = [
    "ChannelWindow",
    "DEFAULT_USE_8_BIT",
    "ENV_USE_8_BIT",
    "MIN_FINITE_PIXELS",
    "MODALITY_OPTICAL",
    "MODALITY_SAR",
    "RadiometryReport",
    "RadiometrySummary",
    "STATUS_DEGENERATE",
    "STATUS_NORMALISED",
    "STATUS_UNAVAILABLE",
    "TRANSFORM_NAME",
    "WINDOW_SIGMAS",
    "describe_transform",
    "normalise_for_croma",
    "resolve_use_8_bit",
    "summarise",
    "warnings_for",
]
