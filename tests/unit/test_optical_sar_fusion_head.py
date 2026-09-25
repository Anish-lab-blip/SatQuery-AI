"""Optical-SAR fusion head — the frozen concatenation and channel dropout.

Freeze section 2.5 fixes two things this file pins:

    concat(optical_GAP, SAR_GAP, joint_GAP, optical_mask, sar_mask) -> (B, 2318)
    fusion head: LayerNorm -> Linear(2318, W) -> GELU -> Dropout -> Linear(W, task_dim)

and one rule that is easy to skip because nothing breaks when you do:

    "Channel/band dropout during fusion-head training is mandatory; it is what
     teaches the head to trust the availability mask."

Mandatory is the operative word. A training transform that silently stopped
running would produce a head that still trains, still converges, and still
reports a number -- it would just have learned to ignore the mask. So the
dropout function is tested directly rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.errors import SpecialistError

pytest.importorskip("torch")

from specialists.optical_sar.fusion_head import (  # noqa: E402
    CROMA_GAP_KEYS,
    assemble_fusion_input,
    build_fusion_head,
    channel_dropout,
    expected_fusion_dim,
    mask_availability_stats,
)


def _croma_output(batch: int = 1, dim: int = 768) -> dict:
    rng = np.random.default_rng(0)
    return {
        "optical_GAP": rng.standard_normal((batch, dim)).astype(np.float32),
        "SAR_GAP": rng.standard_normal((batch, dim)).astype(np.float32),
        "joint_GAP": rng.standard_normal((batch, dim)).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# The frozen width
# ---------------------------------------------------------------------------


def test_expected_fusion_dim_is_2318():
    """3 * 768 + 12 + 2. The number the config asserts and the head is built to."""
    assert expected_fusion_dim() == 2318


def test_croma_gap_keys_are_the_verified_ones():
    """Verified against github.com/antofuller/CROMA README."""
    assert CROMA_GAP_KEYS == ("optical_GAP", "SAR_GAP", "joint_GAP")


def test_assembled_tensor_is_2318_wide():
    """THE DIMENSION TEST. (B, 2318) or the fusion head cannot be built."""
    out = assemble_fusion_input(
        _croma_output(),
        optical_mask=np.ones((1, 12), dtype=np.float32),
        sar_mask=np.ones((1, 2), dtype=np.float32),
    )
    assert out.dim == 2318
    assert out.batch_size == 1


def test_concatenation_order_is_frozen():
    """A permutation gives the right SHAPE and the wrong meaning.

    Each block is filled with a distinct constant so the layout can be checked
    by value. This is the assertion a shape-only test would miss entirely.
    """
    batch, dim = 1, 768
    croma = {
        "optical_GAP": np.full((batch, dim), 1.0, np.float32),
        "SAR_GAP": np.full((batch, dim), 2.0, np.float32),
        "joint_GAP": np.full((batch, dim), 3.0, np.float32),
    }
    optical_mask = np.full((batch, 12), 4.0, np.float32)
    sar_mask = np.full((batch, 2), 5.0, np.float32)

    out = assemble_fusion_input(
        croma, optical_mask=optical_mask, sar_mask=sar_mask
    )

    assert np.all(out.tensor[:, 0:768] == 1.0), "optical_GAP must come first"
    assert np.all(out.tensor[:, 768:1536] == 2.0), "SAR_GAP second"
    assert np.all(out.tensor[:, 1536:2304] == 3.0), "joint_GAP third"
    assert np.all(out.tensor[:, 2304:2316] == 4.0), "optical_mask fourth"
    assert np.all(out.tensor[:, 2316:2318] == 5.0), "sar_mask last"


def test_missing_croma_key_raises_with_the_real_key_names():
    """A missing key must name the expected keys, not fail on a shape later."""
    incomplete = _croma_output()
    del incomplete["joint_GAP"]
    with pytest.raises(SpecialistError) as excinfo:
        assemble_fusion_input(
            incomplete,
            optical_mask=np.ones((1, 12)),
            sar_mask=np.ones((1, 2)),
        )
    assert "joint_GAP" in str(excinfo.value.detail)


def test_wrong_mask_width_raises():
    with pytest.raises(SpecialistError):
        assemble_fusion_input(
            _croma_output(),
            optical_mask=np.ones((1, 11)),   # 11, not 12
            sar_mask=np.ones((1, 2)),
        )


def test_batch_mismatch_between_gaps_raises():
    croma = _croma_output(batch=2)
    croma["joint_GAP"] = np.zeros((1, 768), dtype=np.float32)
    with pytest.raises(SpecialistError):
        assemble_fusion_input(
            croma,
            optical_mask=np.ones((2, 12)),
            sar_mask=np.ones((2, 2)),
        )


# ---------------------------------------------------------------------------
# The head itself
# ---------------------------------------------------------------------------


def test_head_is_built_to_2318_and_refuses_anything_else():
    """A wrong input_dim is refused at construction, not at the first forward."""
    head = build_fusion_head(input_dim=2318, hidden_dim=512, task_dim=19, dropout=0.2)
    assert head is not None

    from core.errors import ModelLoadError

    with pytest.raises(ModelLoadError):
        build_fusion_head(input_dim=2317, hidden_dim=512, task_dim=19, dropout=0.2)


def test_head_forward_accepts_the_assembled_tensor():
    """End-to-end shape: assembly output feeds the head without a reshape."""
    import torch

    head = build_fusion_head(input_dim=2318, hidden_dim=64, task_dim=19, dropout=0.0)
    out = assemble_fusion_input(
        _croma_output(),
        optical_mask=np.ones((1, 12), dtype=np.float32),
        sar_mask=np.ones((1, 2), dtype=np.float32),
    )
    with torch.no_grad():
        logits = head(torch.from_numpy(out.tensor))
    assert logits.shape == (1, 19)


def test_head_output_is_logits_not_probabilities():
    """No softmax in the head; a loss function adds its own.

    Two softmaxes in a row is a real and quiet bug: the probabilities compress
    toward the middle and training just looks slow.
    """
    import torch

    head = build_fusion_head(input_dim=2318, hidden_dim=64, task_dim=19, dropout=0.0)
    out = assemble_fusion_input(
        _croma_output(),
        optical_mask=np.ones((1, 12), dtype=np.float32),
        sar_mask=np.ones((1, 2), dtype=np.float32),
    )
    with torch.no_grad():
        logits = head(torch.from_numpy(out.tensor))
    # Logits are unbounded; a softmax output would sum to 1.
    assert not torch.isclose(logits.sum(), torch.tensor(1.0))


def test_head_rejects_a_non_cpu_device():
    """Deployment is CPU-capable (freeze section 4); a device string that would
    silently fall back is refused instead."""
    with pytest.raises(SpecialistError):
        build_fusion_head(
            input_dim=2318, hidden_dim=64, task_dim=19, dropout=0.2, device="cuda:0"
        )


# ---------------------------------------------------------------------------
# Channel dropout — mandatory, so tested
# ---------------------------------------------------------------------------


def test_channel_dropout_zeroes_the_dropped_channels():
    """Dropped channels are zeroed, not renormalised.

    Renormalising would scale the surviving channels up, fabricating a magnitude
    the real missing-channel case does not have. The head must see the same
    scale it will see at inference on a 4-band sensor.
    """
    rng = np.random.default_rng(1)
    features = np.ones((8, 12), dtype=np.float32)
    mask = np.ones((8, 12), dtype=np.float32)

    dropped, dropped_mask = channel_dropout(
        features, mask, keep_probabilities=(0.5,), rng=rng
    )

    assert dropped_mask.shape == mask.shape
    # Every position the mask now calls absent is exactly zero.
    absent = dropped_mask == 0.0
    assert np.all(dropped[absent] == 0.0)
    # Every position still present kept its original value -- no rescaling.
    present = dropped_mask > 0.0
    assert np.all(dropped[present] == 1.0)


def test_channel_dropout_updates_the_mask_to_match():
    """THE POINT of the exercise.

    Dropping features while leaving the mask saying "present" would teach the
    head that the availability mask is unreliable -- the exact opposite of what
    freeze section 2.5 says the dropout is for.
    """
    rng = np.random.default_rng(2)
    features = np.ones((64, 12), dtype=np.float32)
    mask = np.ones((64, 12), dtype=np.float32)

    dropped, dropped_mask = channel_dropout(
        features, mask, keep_probabilities=(0.4,), rng=rng
    )

    # Wherever the mask says absent, the feature is zero. Mask and data agree.
    assert np.all((dropped == 0.0) == (dropped_mask == 0.0))
    # And the drop actually happened at roughly the requested rate.
    available_fraction = dropped_mask.mean()
    assert 0.25 < available_fraction < 0.55, (
        f"keeping 40% of channels should leave ~40% available, got "
        f"{available_fraction:.3f}"
    )


def test_channel_dropout_never_resurrects_an_absent_channel():
    """A channel the SENSOR did not measure must stay absent.

    This is the no-fabrication rule reaching into training. A Cartosat scene has
    channels 5-12 permanently unavailable; dropout may make MORE channels
    unavailable, and must never make an unavailable one available.
    """
    rng = np.random.default_rng(3)
    features = np.ones((32, 12), dtype=np.float32)
    # Cartosat-2S: 4 real channels, 8 masked off.
    mask = np.zeros((32, 12), dtype=np.float32)
    mask[:, :4] = 1.0

    _, dropped_mask = channel_dropout(
        features, mask, keep_probabilities=(1.0,), rng=rng
    )

    # With keep_probability 1.0 nothing is dropped...
    assert np.all(dropped_mask == mask)
    # ...and channels 4-11 are still absent. They can never be switched on.
    assert np.all(dropped_mask[:, 4:] == 0.0)


def test_channel_dropout_is_seeded_and_reproducible():
    """Same seed, same drop. A training transform must not be a source of
    run-to-run variance that nobody can reproduce."""
    features = np.ones((16, 12), dtype=np.float32)
    mask = np.ones((16, 12), dtype=np.float32)

    a, _ = channel_dropout(
        features, mask, keep_probabilities=(0.5,), rng=np.random.default_rng(7)
    )
    b, _ = channel_dropout(
        features, mask, keep_probabilities=(0.5,), rng=np.random.default_rng(7)
    )
    assert np.array_equal(a, b)


def test_channel_dropout_supports_the_plan_schedule():
    """Plan section 20's two schedules: optical 100/80/60/40, SAR 100/50."""
    rng = np.random.default_rng(11)
    optical_features = np.ones((200, 12), dtype=np.float32)
    optical_mask = np.ones((200, 12), dtype=np.float32)

    _, dropped = channel_dropout(
        optical_features,
        optical_mask,
        keep_probabilities=(1.0, 0.8, 0.6, 0.4),
        rng=rng,
    )
    mean_available = dropped.mean()
    # Averaging the four rates gives 0.7. Allow a generous band for sampling.
    assert 0.6 < mean_available < 0.8, f"got {mean_available:.3f}"


def test_channel_dropout_rejects_out_of_range_probabilities():
    with pytest.raises(SpecialistError):
        channel_dropout(
            np.ones((4, 4), dtype=np.float32),
            np.ones((4, 4), dtype=np.float32),
            keep_probabilities=(1.5,),
            rng=np.random.default_rng(0),
        )


def test_channel_dropout_rejects_shape_mismatch():
    with pytest.raises(SpecialistError):
        channel_dropout(
            np.ones((4, 12), dtype=np.float32),
            np.ones((4, 2), dtype=np.float32),
            keep_probabilities=(0.5,),
            rng=np.random.default_rng(0),
        )


def test_empty_schedule_is_a_no_op():
    """No schedule means no dropout, which must not crash or alter the input."""
    features = np.ones((4, 6), dtype=np.float32)
    mask = np.ones((4, 6), dtype=np.float32)
    out, out_mask = channel_dropout(
        features, mask, keep_probabilities=(), rng=np.random.default_rng(0)
    )
    assert np.array_equal(out, features)
    assert np.array_equal(out_mask, mask)


# ---------------------------------------------------------------------------
# Availability statistics — feeds plan section 26
# ---------------------------------------------------------------------------


def test_availability_stats_report_channel_fractions():
    stats = mask_availability_stats(
        np.array([[1.0] * 4 + [0.0] * 8]),   # Cartosat: 4 of 12
        np.array([[1.0, 1.0]]),
    )
    assert stats["optical_channels_present"] == 4.0
    assert stats["sar_channels_present"] == 2.0
    # float32 input, so the fraction carries float32 precision. 1e-9 would be a
    # float64 expectation and would fail on a correct value.
    assert abs(stats["optical_available_fraction"] - 4 / 12) < 1e-6
    assert stats["sar_available_fraction"] == 1.0


def test_availability_stats_handle_an_empty_mask():
    stats = mask_availability_stats(np.zeros((0, 0)), np.zeros((0, 0)))
    assert stats["optical_available_fraction"] == 0.0
    assert stats["sar_available_fraction"] == 0.0
