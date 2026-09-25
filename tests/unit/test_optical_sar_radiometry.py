"""CROMA encoder-input radiometry — the DEV-2 ruling, as executable guards.

Authority: `docs/PHASE14_CROMA_NORMALISATION_CHANGE.md` section 3.1, which
specifies five testable requirements. Each has a section below:

    3.1.1  per-channel, over one sample's (C, H, W)
    3.1.2  a zero channel is skipped and left at zero -- no invented scale
    3.1.3  deterministic
    3.1.4  the applied transform, use_8_bit and the per-channel window are recorded
    3.1.5  applied identically to SAR and optical

Section 7 is the part that matters most, and the reason this file is not merely a
unit test of a helper: it proves the stage is **reached by the encoder path**. A
normaliser that is correct and never called would pass every test in sections 1-6
while changing nothing about what the frozen encoder receives -- which is the
failure mode this repo keeps finding.

WHY THE PER-SAMPLE GUARD IS A REAL REGRESSION GUARD
---------------------------------------------------
`test_sample_zero_is_unaffected_by_its_batch_neighbours` fails against a
*correct-looking* per-batch implementation. That matters: the repo's rule is that
a test passing under both the intended and the defective implementation is not a
guard. The README's own example computes the window over the whole batch, so a
faithful-to-README implementation would be per-batch and would fail this test --
deliberately, because batch-dependent encoding is unacceptable for a serving
system (see the module docstring's "DELIBERATE DEVIATION 1").

Numerical expectations here are computed from the same array the code sees, never
from a hand-written constant, so a change in the window definition cannot be
absorbed by a stale expected value.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from specialists.optical_sar import radiometry
from specialists.optical_sar.radiometry import (
    DEFAULT_USE_8_BIT,
    ENV_USE_8_BIT,
    MODALITY_OPTICAL,
    MODALITY_SAR,
    STATUS_DEGENERATE,
    STATUS_NORMALISED,
    STATUS_UNAVAILABLE,
    TRANSFORM_NAME,
    normalise_for_croma,
    resolve_use_8_bit,
    summarise,
    warnings_for,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gradient(channel: int, h: int = 8, w: int = 8):
    """A deterministic channel with a real dynamic range.

    Each channel index gets its own scale, so a window shared across channels
    would be visibly different from per-channel windows.
    """
    lo = float(channel) * 10.0
    return np.linspace(lo, lo + 4.0, h * w, dtype=np.float32).reshape(h, w)


def _sample(n_channels: int, h: int = 8, w: int = 8, *, available: int | None = None):
    """`(1, C, H, W)`: `available` channels carry signal, the rest are exactly 0.

    This mirrors the sensor adapter's convention exactly -- Cartosat-2S fills 4
    of 12 slots and leaves the other 8 at zero with the mask false.
    """
    if available is None:
        available = n_channels
    arr = np.zeros((1, n_channels, h, w), dtype=np.float32)
    for c in range(available):
        arr[0, c] = _gradient(c, h, w)
    return arr


def _expected_window(channel: np.ndarray) -> tuple[float, float]:
    """`mean +/- 2*std` (ddof=1) over the finite pixels -- computed independently."""
    values = channel.astype(np.float64)
    finite = values[np.isfinite(values)]
    mean = float(finite.mean())
    std = float(finite.std(ddof=1))
    return mean - 2.0 * std, mean + 2.0 * std


# ---------------------------------------------------------------------------
# 1. The transform's identity
# ---------------------------------------------------------------------------


def test_transform_name_is_recorded_not_implied():
    assert TRANSFORM_NAME == "per_channel_mean_pm_2std"
    described = radiometry.describe_transform(True)
    assert described["transform"] == TRANSFORM_NAME
    assert described["window_sigmas"] == 2.0
    assert described["use_8_bit"] is True
    assert described["range"] == "[0, 1]"
    # The zero-channel rule must be stated in the record, not just implemented.
    assert "zero" in described["zero_channel_policy"]
    # The std convention is part of the transform's identity: torch's default is
    # ddof=1 and numpy's is ddof=0. A reader must not have to guess.
    assert described["std_ddof"] == 1


def test_use_8_bit_defaults_to_enabled_per_the_ruling():
    """Section 2.2: "Decision on `use_8_bit`: ENABLED"."""
    assert DEFAULT_USE_8_BIT is True


# ---------------------------------------------------------------------------
# 2. Requirement 3.1.1 -- per channel, within a sample
# ---------------------------------------------------------------------------


def test_each_channel_gets_its_own_window():
    """Two channels with different scales must get different windows.

    A per-scene stretch (one window across all bands) would give them the same
    one. That is the transform `preprocessing/imagery.py` implements, and
    section 1.1 of the ruling disqualifies it for exactly this reason.
    """
    arr = np.zeros((1, 2, 8, 8), dtype=np.float32)
    arr[0, 0] = np.linspace(0.0, 4.0, 64, dtype=np.float32).reshape(8, 8)
    arr[0, 1] = np.linspace(1000.0, 4000.0, 64, dtype=np.float32).reshape(8, 8)

    _, report = normalise_for_croma(arr)

    by_channel = {w.channel: w for w in report.windows}
    assert by_channel[0].status == STATUS_NORMALISED
    assert by_channel[1].status == STATUS_NORMALISED
    # Windows differ by orders of magnitude -- they cannot have been shared.
    assert by_channel[0].upper < 100.0
    assert by_channel[1].lower > 100.0
    assert report.normalised_channels == (0, 1)


def test_a_channels_window_is_unaffected_by_another_channels_values():
    """A channel's window is a function of *that channel's* values alone.

    `test_each_channel_gets_its_own_window` proves two channels get different
    windows -- which catches a single window shared across the whole scene, but
    not a window that pools statistics across channels. This holds channel 0
    fixed and changes only channel 1, so any leakage of channel 1's values into
    channel 0's window (or output) is visible. That is the property a global /
    cross-channel statistic violates even when the two windows still differ.
    """
    base = _sample(2)  # channels 0 and 1 both carry signal
    changed = base.copy()
    changed[0, 1] = _gradient(1) * 1000.0 + 12_345.0  # only channel 1 moves

    out_base, report_base = normalise_for_croma(base)
    out_changed, report_changed = normalise_for_croma(changed)

    window_base = {w.channel: w for w in report_base.windows}[0]
    window_changed = {w.channel: w for w in report_changed.windows}[0]

    # Channel 0's window must be byte-identical: channel 1 is irrelevant to it.
    assert window_base.lower == window_changed.lower
    assert window_base.upper == window_changed.upper
    assert window_base.finite_pixels == window_changed.finite_pixels
    # ...and so must channel 0's normalised output.
    assert np.array_equal(out_base[0, 0], out_changed[0, 0])


def test_window_is_mean_plus_minus_two_std():
    """The window is `mean +/- 2*std` (ddof=1), not a min/max clamp."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    arr = values.reshape(1, 1, 2, 4)

    _, report = normalise_for_croma(arr)
    window = report.windows[0]
    expected_lower, expected_upper = _expected_window(arr[0, 0])

    assert window.lower == pytest.approx(expected_lower, rel=1e-6)
    assert window.upper == pytest.approx(expected_upper, rel=1e-6)
    # Not the data range: mean +/- 2*std of this set is wider than [1, 8].
    assert window.lower < float(values.min())
    assert window.upper > float(values.max())


def test_the_stretch_is_mean_pm_2std_and_not_a_percentile():
    """A discriminator between this transform and the one `base.yaml` names.

    `mean +/- 2*std` is **not** outlier-robust: one extreme pixel inflates the
    standard deviation and compresses everything else toward the window centre.
    A 2-98 percentile stretch would not do that -- it would map the bulk of the
    channel across most of `[0, 1]`.

    This is pinned because the two transforms are easy to confuse (section 1.1 of
    the ruling exists precisely because a percentile stretch is already present
    elsewhere in the repo), and because a future "cleanup" that substituted one
    for the other would otherwise pass every other test in this file.
    """
    arr = np.zeros((1, 1, 8, 8), dtype=np.float32)
    body = np.linspace(10.0, 20.0, 63, dtype=np.float32)
    arr[0, 0].flat[:63] = body
    arr[0, 0].flat[63] = 1_000_000.0  # the outlier

    out, _ = normalise_for_croma(arr, use_8_bit=False)
    body_out = out[0, 0].flat[:63]

    # Compressed to a sliver by the outlier -- the signature of mean +/- 2*std.
    assert float(body_out.max() - body_out.min()) < 0.01
    # But not collapsed to a constant, and still inside the range.
    assert float(body_out.max() - body_out.min()) > 0.0
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


# ---------------------------------------------------------------------------
# 3. Requirement 3.1.2 -- the zero-channel rule (the load-bearing one)
# ---------------------------------------------------------------------------


def test_unavailable_channels_are_left_at_exactly_zero():
    """Section 3.1.2. The Cartosat-2S case: 4 of 12 optical channels present."""
    arr = _sample(12, available=4)
    out, report = normalise_for_croma(arr)

    assert report.unavailable_channels == tuple(range(4, 12))
    assert report.normalised_channels == (0, 1, 2, 3)
    # Exactly zero, not "close to zero" and not a fabricated scale.
    for c in range(4, 12):
        assert np.all(out[0, c] == 0.0), f"channel {c} was not left at zero"


def test_an_all_zero_channel_does_not_divide_by_zero():
    """`std == 0` on an all-zero channel; the ruling's stated hazard."""
    arr = np.zeros((1, 1, 8, 8), dtype=np.float32)
    out, report = normalise_for_croma(arr)

    assert np.all(np.isfinite(out))
    assert np.all(out == 0.0)
    assert report.windows[0].status == STATUS_UNAVAILABLE
    assert report.windows[0].lower is None
    assert report.windows[0].upper is None


def test_an_unavailable_channel_gets_no_window_recorded():
    """`lower`/`upper` are None, never 0.0.

    Recording 0.0 would read as a real measurement of a zero-width range, which
    is a fabricated fact rather than an absent one.
    """
    _, report = normalise_for_croma(_sample(2, available=1))
    unavailable = [w for w in report.windows if w.status == STATUS_UNAVAILABLE]
    assert unavailable
    for w in unavailable:
        assert w.lower is None and w.upper is None
        assert w.finite_pixels == 0


def test_a_constant_nonzero_channel_is_skipped_and_labelled_differently():
    """A flat band has a zero-width window too, but it is NOT the same fact.

    "The sensor did not measure this" and "the sensor measured a flat band" must
    not merge in the trace, so the status differs even though the outcome (left
    at zero, no invented scale) is the same.
    """
    arr = np.zeros((1, 2, 8, 8), dtype=np.float32)
    arr[0, 0] = _gradient(0)   # real
    arr[0, 1] = 500.0          # present, but perfectly flat

    out, report = normalise_for_croma(arr)
    by_channel = {w.channel: w for w in report.windows}

    assert by_channel[0].status == STATUS_NORMALISED
    assert by_channel[1].status == STATUS_DEGENERATE
    assert np.all(out[0, 1] == 0.0)
    assert report.degenerate_channels == (1,)
    # ...and it is distinguishable from unavailability, not merged into it.
    assert report.unavailable_channels == ()


def test_a_degenerate_channel_is_reported_to_the_operator():
    """A present-but-unstretchable band is a data defect, so it must be visible.

    A masked-off channel is the expected case and must stay silent -- warning on
    it every request would be noise for every Cartosat-2S input.
    """
    _, expected_quiet = normalise_for_croma(_sample(12, available=4))
    assert warnings_for(expected_quiet) == []

    arr = np.zeros((1, 1, 8, 8), dtype=np.float32)
    arr[0, 0] = 7.5
    _, report = normalise_for_croma(arr)
    notes = warnings_for(report)
    assert len(notes) == 1
    assert "channel(s) [0]" in notes[0]
    assert "fabricated" in notes[0]


def test_non_finite_pixels_are_excluded_from_the_window():
    """Requirement 3.1.1 says the window is over *finite* pixels only."""
    arr = np.zeros((1, 1, 4, 4), dtype=np.float32)
    arr[0, 0] = np.linspace(1.0, 16.0, 16, dtype=np.float32).reshape(4, 4)
    arr[0, 0, 0, 0] = np.nan
    arr[0, 0, 0, 1] = np.inf

    out, report = normalise_for_croma(arr)
    window = report.windows[0]
    expected_lower, expected_upper = _expected_window(arr[0, 0])

    assert window.status == STATUS_NORMALISED
    assert window.finite_pixels == 14
    assert window.lower == pytest.approx(expected_lower, rel=1e-6)
    assert window.upper == pytest.approx(expected_upper, rel=1e-6)
    # NaN/inf must not leak into the output...
    assert np.all(np.isfinite(out))

    # ...and "not leaking" is not enough to pin the behaviour: a naive
    # clip-then-cast would also produce finite values, because NaN compares
    # false against both clip bounds and lands on whichever bound is applied
    # last. The actual contract is that a non-finite pixel is *pinned to 0.0*
    # (the sensor adapter's "nothing measured here" value), and that the finite
    # pixels around it are stretched exactly as if the gap were not there.
    assert out[0, 0, 0, 0] == 0.0, "NaN pixel must be pinned to exactly 0.0"
    assert out[0, 0, 0, 1] == 0.0, "inf pixel must be pinned to exactly 0.0"

    finite = np.isfinite(arr[0, 0])
    expected = np.zeros((4, 4), dtype=np.float64)
    expected[finite] = (
        arr[0, 0][finite].astype(np.float64) - expected_lower
    ) / (expected_upper - expected_lower)
    expected = np.clip(expected * 255.0, 0.0, 255.0).astype(np.uint8) / 255.0
    assert np.array_equal(out[0, 0], expected.astype(np.float32))


def test_a_channel_with_fewer_than_two_finite_pixels_is_skipped():
    """One pixel has no standard deviation at ddof=1."""
    arr = np.full((1, 1, 4, 4), np.nan, dtype=np.float32)
    arr[0, 0, 0, 0] = 3.0

    out, report = normalise_for_croma(arr)
    assert report.windows[0].status == STATUS_DEGENERATE
    assert report.windows[0].finite_pixels == 1
    assert np.all(out == 0.0)


# ---------------------------------------------------------------------------
# 4. Requirement 3.1.3 -- deterministic, and bounded
# ---------------------------------------------------------------------------


def test_the_transform_is_deterministic():
    arr = _sample(12, available=4)
    first, _ = normalise_for_croma(arr)
    second, _ = normalise_for_croma(arr.copy())
    assert np.array_equal(first, second)


def test_the_caller_owned_input_is_not_mutated():
    """Purity: the transform returns a new array and leaves the input intact.

    `source = arr.astype(np.float32, copy=False)` aliases a float32 input, so a
    write through `source` would corrupt the caller's buffer. The contract is a
    fresh output array; the caller's array must be byte-identical afterwards.
    This is what makes `test_the_transform_is_deterministic` meaningful -- if the
    first call rewrote the input, the second call would be reading a different
    array, and the two calls would only agree by accident.
    """
    arr = _sample(12, available=4)
    before = arr.copy()
    out, _ = normalise_for_croma(arr)

    assert np.array_equal(arr, before), "caller-owned input was modified in place"
    assert out is not arr, "output must not alias the caller's array"


def test_output_is_bounded_to_zero_one_with_8_bit():
    """Requirement 3.1.3's range invariant, and section 2.2's reason for it."""
    arr = np.linspace(0.0, 10_000.0, 64, dtype=np.float32).reshape(1, 1, 8, 8)

    out, report = normalise_for_croma(arr, use_8_bit=True)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert report.use_8_bit is True
    # The uint8 round-trip means every value is exactly k/255.
    scaled = out[0, 0] * 255.0
    assert np.allclose(scaled, np.round(scaled), atol=1e-5)


def test_output_is_bounded_to_zero_one_without_8_bit():
    arr = np.linspace(0.0, 10_000.0, 64, dtype=np.float32).reshape(1, 1, 8, 8)

    out, report = normalise_for_croma(arr, use_8_bit=False)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert report.use_8_bit is False


def test_the_two_use_8_bit_paths_differ_but_agree_closely():
    """The quantisation is real and bounded (~1/255), not a no-op.

    If `use_8_bit` made no difference, the ruling's gated experiment arm C --
    which exists solely to isolate this decision -- would be measuring nothing.
    """
    arr = _sample(4, available=4)
    eight, _ = normalise_for_croma(arr, use_8_bit=True)
    full, _ = normalise_for_croma(arr, use_8_bit=False)

    assert not np.array_equal(eight, full)
    assert np.max(np.abs(eight - full)) <= 1.0 / 255.0 + 1e-6


def test_bad_rank_raises():
    with pytest.raises(ValueError):
        normalise_for_croma(np.zeros((8, 8), dtype=np.float32))


def test_channel_shape_is_preserved():
    arr = _sample(12, available=12, h=12, w=10)
    out, report = normalise_for_croma(arr)
    assert out.shape == arr.shape
    assert out.dtype == np.float32
    assert report.n_samples == 1
    assert report.n_channels == 12


# ---------------------------------------------------------------------------
# 5. Requirement 3.1.5 -- SAR is treated identically
# ---------------------------------------------------------------------------


def test_sar_uses_the_same_function_and_records_its_modality():
    """Upstream uses one `ViT` class for both modalities and one `normalize`.

    A separate SAR path would be an invention; section 3.1.5 of the ruling asks
    for the difference to be recorded if one exists. It does not exist.
    """
    sar = _sample(2, available=2)
    out, report = normalise_for_croma(sar, modality=MODALITY_SAR)

    assert report.modality == MODALITY_SAR
    assert report.normalised_channels == (0, 1)
    assert out.max() <= 1.0

    # Identical inputs through the identical code path give identical results,
    # modulo the modality label.
    optical_labelled, optical_report = normalise_for_croma(
        sar, modality=MODALITY_OPTICAL
    )
    assert np.array_equal(out, optical_labelled)
    assert optical_report.modality == MODALITY_OPTICAL


def test_summarise_refuses_a_modality_divergence():
    """A disagreement between the two modalities is a bug, not a preference."""
    _, optical = normalise_for_croma(
        _sample(2), use_8_bit=True, modality=MODALITY_OPTICAL
    )
    _, sar = normalise_for_croma(_sample(2), use_8_bit=False, modality=MODALITY_SAR)
    with pytest.raises(ValueError):
        summarise(optical, sar)


def test_summarise_carries_both_modalities():
    _, optical = normalise_for_croma(
        _sample(12, available=4), modality=MODALITY_OPTICAL
    )
    _, sar = normalise_for_croma(_sample(2, available=2), modality=MODALITY_SAR)
    summary = summarise(optical, sar)

    assert summary.use_8_bit is True
    assert summary.transform == TRANSFORM_NAME
    payload = summary.to_dict()
    assert payload["optical"]["n_channels"] == 12
    assert payload["sar"]["n_channels"] == 2


# ---------------------------------------------------------------------------
# 6. `use_8_bit` resolution -- the config-hash-safe path
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Minimal `Config.get` stand-in, so these tests need no YAML."""

    def __init__(self, value=None):
        self._value = value

    def get(self, path, default=None):
        if path == "croma.use_8_bit" and self._value is not None:
            return self._value
        return default


def test_resolution_order_env_beats_config(monkeypatch):
    cfg = _FakeConfig(False)
    assert resolve_use_8_bit(cfg) == (False, "config")

    monkeypatch.setenv(ENV_USE_8_BIT, "true")
    assert resolve_use_8_bit(cfg) == (True, "env")

    monkeypatch.setenv(ENV_USE_8_BIT, "0")
    assert resolve_use_8_bit(cfg) == (False, "env")


def test_an_absent_config_key_falls_through_to_the_default():
    """The config path must be a no-op until the key is deliberately added.

    This is what makes the resolution hash-safe: `Config.get` returns its default
    for a missing key, so nothing changes until someone decides to move the
    frozen config hash.
    """
    assert resolve_use_8_bit(_FakeConfig()) == (DEFAULT_USE_8_BIT, "default")
    assert resolve_use_8_bit(None) == (DEFAULT_USE_8_BIT, "default")


def test_a_non_bool_config_value_is_not_honoured():
    """`"yes"` is not a bool here; coercing it silently would hide a typo."""
    assert resolve_use_8_bit(_FakeConfig("yes")) == (DEFAULT_USE_8_BIT, "default")


def test_the_frozen_config_hash_has_not_moved():
    """THE LOAD-BEARING GUARD for this change.

    Adding `croma.use_8_bit` to `configs/base.yaml` would move `Config.hash` and
    detach the Phase 9 benchmark from its config (exit 3 in
    `scripts/eval_change.py`). The normalisation is wired through a hash-exempt
    channel precisely so this assertion keeps holding.
    """
    from core.config import load_config

    assert load_config().hash == "78f1e3700da15aa1"


# ---------------------------------------------------------------------------
# 7. The stage is REACHED by the encoder -- not merely defined
# ---------------------------------------------------------------------------


class _CapturingCROMA:
    """Records the exact tensors and keyword names the forward pass received."""

    def __init__(self, dim: int = 768, n_patches: int = 225):
        self.dim = dim
        self.n_patches = n_patches
        self.received: dict = {}

    def eval(self):
        return self

    def parameters(self):
        import torch

        return iter([torch.zeros(1, requires_grad=False)])

    def named_parameters(self):
        import torch

        return iter([("w", torch.zeros(1, requires_grad=False))])

    def to(self, device):
        return self

    def __call__(self, **kwargs):
        """Keyword-only, and the names are captured rather than assumed.

        An earlier draft hardcoded the captured names, which made the C-1
        assertion in `test_the_encoder_still_sends_exactly_two_kwargs`
        unfalsifiable. Capturing `set(kwargs)` is what makes it a real check.
        """
        import torch

        self.received = {
            "SAR_images": kwargs["SAR_images"].clone(),
            "optical_images": kwargs["optical_images"].clone(),
            "kwargs": set(kwargs),
        }
        batch = kwargs["optical_images"].shape[0]
        return {
            "optical_GAP": torch.ones(batch, self.dim),
            "SAR_GAP": torch.ones(batch, self.dim),
            "joint_GAP": torch.ones(batch, self.dim),
        }


def _encoder(**kwargs):
    pytest.importorskip("torch")
    from specialists.optical_sar.croma import CROMAEncoder

    return CROMAEncoder(_CapturingCROMA(), resolution=120, device="cpu", **kwargs)


def _optical_with_4_of_12(scale: float = 10_000.0) -> np.ndarray:
    arr = np.zeros((1, 12, 120, 120), dtype=np.float32)
    for c in range(4):
        arr[0, c] = np.linspace(
            0.0, scale, 120 * 120, dtype=np.float32
        ).reshape(120, 120)
    return arr


def test_the_encoder_actually_normalises_what_it_sends():
    """If this fails, the normaliser is correct and unused.

    Input values run into the tens of thousands, so an un-normalised path is
    unmistakable: the encoder would receive raw values far outside `[0, 1]`.
    """
    encoder = _encoder()
    sar = np.zeros((1, 2, 120, 120), dtype=np.float32)
    sar[0, :] = np.linspace(0.0, 5_000.0, 120 * 120, dtype=np.float32).reshape(
        120, 120
    )

    out = encoder.encode(_optical_with_4_of_12(), sar)

    received = encoder.model.received
    assert float(received["optical_images"].max()) <= 1.0
    assert float(received["SAR_images"].max()) <= 1.0
    assert float(received["optical_images"].min()) >= 0.0

    # ...and the record of what was applied came back with the encoding.
    assert out.radiometry is not None
    assert out.radiometry.use_8_bit is True
    assert out.radiometry.optical.normalised_channels == (0, 1, 2, 3)
    assert out.radiometry.optical.unavailable_channels == tuple(range(4, 12))


def test_the_encoder_still_sends_exactly_two_kwargs():
    """Finding C-1 survives the normalisation: no mask reaches CROMA.

    The normalisation is a caller-side stage. If wiring it had threaded a mask
    into the forward call, this is where it would show.
    """
    encoder = _encoder()
    encoder.encode(
        np.zeros((1, 12, 120, 120), dtype=np.float32),
        np.zeros((1, 2, 120, 120), dtype=np.float32),
    )
    assert encoder.model.received["kwargs"] == {"SAR_images", "optical_images"}


def test_encode_signature_is_unchanged():
    """The pinned C-1 guard in test_optical_sar_croma.py must keep holding."""
    from specialists.optical_sar.croma import CROMAEncoder

    params = set(inspect.signature(CROMAEncoder.encode).parameters)
    assert params == {"self", "optical", "sar"}


def test_unavailable_channels_reach_the_model_as_exactly_zero():
    """The zero-fill convention must survive the normalisation unchanged.

    If the stretch were applied to a zero channel, `(0 - lower)/(upper - lower)`
    would be a large positive number, not zero -- and CROMA would be fed a
    fabricated measurement for a band the sensor never took.
    """
    encoder = _encoder()
    encoder.encode(
        _optical_with_4_of_12(scale=800.0),
        np.zeros((1, 2, 120, 120), dtype=np.float32),
    )

    sent = encoder.model.received["optical_images"].numpy()
    for c in range(4, 12):
        assert np.all(sent[0, c] == 0.0), f"channel {c} was fabricated"


def test_sample_zero_is_unaffected_by_its_batch_neighbours():
    """THE PER-SAMPLE GUARD. Fails against a faithful-to-README per-batch impl.

    The README computes `x[:, c].mean()` over the whole batch, so with a
    per-batch window, sample 0's encoding would change depending on what was
    batched with it. For a serving system that is unacceptable: the same query
    must not return a different answer because another request shared the batch.

    Sample 1 here has a wildly different scale, so a per-batch implementation
    would give sample 0 a visibly different window.
    """
    optical_0 = np.zeros((1, 12, 120, 120), dtype=np.float32)
    optical_0[0, 0] = np.linspace(
        0.0, 10.0, 120 * 120, dtype=np.float32
    ).reshape(120, 120)
    optical_1 = np.zeros((1, 12, 120, 120), dtype=np.float32)
    optical_1[0, 0] = np.linspace(
        0.0, 100_000.0, 120 * 120, dtype=np.float32
    ).reshape(120, 120)

    sar_pair = np.zeros((2, 2, 120, 120), dtype=np.float32)
    for b in range(2):
        sar_pair[b] = np.linspace(
            0.0, 1.0, 120 * 120, dtype=np.float32
        ).reshape(120, 120)

    alone = _encoder()
    alone.encode(optical_0, np.zeros((1, 2, 120, 120), dtype=np.float32))
    sample_alone = alone.model.received["optical_images"].numpy()[0, 0]

    batched = _encoder()
    batched.encode(np.concatenate([optical_0, optical_1], axis=0), sar_pair)
    sample_batched = batched.model.received["optical_images"].numpy()[0, 0]

    assert np.array_equal(sample_alone, sample_batched), (
        "sample 0's encoding changed when a differently-scaled sample was added "
        "to the batch: the window is being computed per batch, not per sample"
    )


def test_normalisation_can_be_disabled_for_the_gated_experiment():
    """Ruling section 4 arm B needs a no-stretch control.

    Disabling it must be visible in the description, so a trace can never show a
    normalisation that did not run.
    """
    encoder = _encoder(normalize_input=False)
    assert encoder.describe()["input_normalisation"]["applied"] is False

    optical = np.zeros((1, 12, 120, 120), dtype=np.float32)
    optical[0, 0] = 10_000.0
    out = encoder.encode(optical, np.zeros((1, 2, 120, 120), dtype=np.float32))

    assert out.radiometry is None
    # Un-normalised: the raw value reached the model.
    assert float(encoder.model.received["optical_images"].max()) == 10_000.0


def test_the_description_records_the_transform_and_its_source():
    """Requirement 3.1.4: the applied transform must be inspectable."""
    described = _encoder().describe()["input_normalisation"]

    assert described["applied"] is True
    assert described["transform"] == TRANSFORM_NAME
    assert described["use_8_bit"] is True
    assert described["use_8_bit_source"] in {"default", "env", "config", "explicit"}
    assert described["std_ddof"] == 1


def test_the_encoder_honours_the_environment_override(monkeypatch):
    monkeypatch.setenv(ENV_USE_8_BIT, "false")
    encoder = _encoder()
    assert encoder.use_8_bit is False
    assert encoder.use_8_bit_source == "env"
