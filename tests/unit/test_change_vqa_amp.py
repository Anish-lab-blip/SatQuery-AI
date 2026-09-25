"""R-02 — the mixed-precision training contract.

WHY THIS FILE EXISTS
--------------------
The first real Kaggle run died twice on mixed precision, in two different ways,
and both were invisible from a CPU-only host:

  1. `aten::binary_cross_entropy` is an ERROR under CUDA autocast. The fix lives
     in `training/change_vqa/model.py` and is pinned in
     `tests/unit/test_change_vqa_head.py` (area Q).

  2. The AMP dtype was chosen with `torch.cuda.is_bf16_supported()`, which
     returns True on Turing (T4, cc 7.5) because it counts *emulated* bf16. The
     plan's hardware row for R-02 is T4x2, so the trainer would have selected
     emulated bf16 — slower than fp16 on that card, since emulation does not use
     the tensor cores — and paired it with a `GradScaler`, which bf16 does not
     need at all.

WHAT CAN AND CANNOT BE TESTED HERE
----------------------------------
This host has no CUDA. `torch.cuda.is_bf16_supported()` and
`torch.cuda.get_device_capability()` therefore cannot be exercised against real
silicon, and `torch.amp.autocast("cuda")` is a no-op that warns and disables
itself. So these tests pin the DECISION FUNCTION — that a cc<8.0 device maps to
fp16+scaler and a cc>=8.0 device maps to bf16+no-scaler — by driving the
capability probe through `monkeypatch`. They do not, and do not claim to,
execute a real bf16 kernel.

The end-to-end proof that the training step runs is a separate, real run:
`tests/unit/test_change_vqa_smoke.py` drives forward/backward/step/validate/save
/reload on CPU. Full CUDA AMP behaviour is NOT locally exercised and is reported
as such.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from training.change_vqa import train as T


# ---------------------------------------------------------------------------
# The capability probe
# ---------------------------------------------------------------------------


def test_the_probe_reports_no_native_bfloat16_on_a_non_cuda_host() -> None:
    """A probe must answer, not raise — it is called on the training path."""
    assert torch.cuda.is_available() is False
    assert T.native_bfloat16_supported() is False


def test_the_probe_does_not_raise_when_the_device_query_fails(
    monkeypatch: Any,
) -> None:
    """A driver failure must degrade to 'not supported', never kill the run."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("no CUDA driver")

    monkeypatch.setattr(torch.cuda, "get_device_capability", boom)
    assert T.native_bfloat16_supported() is False


@pytest.mark.parametrize(
    "capability, expected",
    [((7, 5), False), ((8, 0), True), ((8, 6), True), ((9, 0), True)],
)
def test_the_probe_tracks_compute_capability(
    monkeypatch: Any, capability: tuple[int, int], expected: bool
) -> None:
    """T4 is (7, 5) — this is the row that decides the real Kaggle run."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda index=0: capability
    )
    assert T.native_bfloat16_supported() is expected


# ---------------------------------------------------------------------------
# The dtype / scaler pairing
# ---------------------------------------------------------------------------


def test_amp_disabled_selects_plain_float32_and_no_scaler() -> None:
    assert T.select_amp(False) == (torch.float32, False)


def test_a_t4_selects_float16_with_a_grad_scaler(monkeypatch: Any) -> None:
    """Turing: fp16 uses the tensor cores; emulated bf16 would not."""
    monkeypatch.setattr(T, "native_bfloat16_supported", lambda index=0: False)
    assert T.select_amp(True) == (torch.float16, True)


def test_native_bfloat16_never_gets_a_grad_scaler(monkeypatch: Any) -> None:
    """The regression guard for the original defect.

    `GradScaler` exists to keep fp16 gradients out of the subnormal range.
    bfloat16 has fp32's exponent range, so a scaler there is a no-op that can
    still silently skip an optimizer step — which is worse than a no-op, because
    the run record would report mixed precision over a stalled update.
    """
    monkeypatch.setattr(T, "native_bfloat16_supported", lambda index=0: True)
    dtype, use_scaler = T.select_amp(True)
    assert dtype is torch.bfloat16
    assert use_scaler is False


def test_the_scaler_is_only_ever_paired_with_float16(monkeypatch: Any) -> None:
    """Stated as an invariant, so it survives any future dtype being added."""
    for native in (False, True):
        monkeypatch.setattr(T, "native_bfloat16_supported", lambda i=0, n=native: n)
        dtype, use_scaler = T.select_amp(True)
        if dtype is torch.bfloat16:
            assert use_scaler is False
        else:
            assert dtype is torch.float16
            assert use_scaler is True


# ---------------------------------------------------------------------------
# The reason string that ends up in the run record
# ---------------------------------------------------------------------------


def test_the_reason_names_the_dtype_and_the_scaler_decision(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(T, "native_bfloat16_supported", lambda index=0: False)
    enabled, reason = T.resolve_amp(True, "cuda")
    assert enabled is True
    assert "float16" in reason
    assert "GradScaler" in reason

    monkeypatch.setattr(T, "native_bfloat16_supported", lambda index=0: True)
    enabled, reason = T.resolve_amp(True, "cuda")
    assert enabled is True
    assert "bfloat16" in reason
    assert "no GradScaler" in reason


def test_amp_on_a_cpu_host_is_refused_with_a_reason() -> None:
    """`--amp` on a CPU host is a request that cannot be honoured. Saying so is
    the point: a record claiming AMP over an fp32 run misdescribes it."""
    enabled, reason = T.resolve_amp(True, "cpu")
    assert enabled is False
    assert "unavailable" in reason
    assert "fp32" in reason


def test_amp_can_always_be_switched_off_explicitly() -> None:
    enabled, reason = T.resolve_amp(False, "cuda")
    assert enabled is False
    assert reason == "disabled by --no-amp"


# ---------------------------------------------------------------------------
# Non-finite guards
# ---------------------------------------------------------------------------
# A non-finite loss does not raise on its own: it propagates through
# `optimizer.step()` into every weight, and the run then saves a checkpoint of
# NaNs that still loads, still predicts, and still reports an accuracy near
# chance. These guards are the only thing standing between a saturated sigmoid
# and a plausible-looking artifact.


def _breakdown(value: float) -> Any:
    from training.change_vqa.model import ChangeVQALossBreakdown

    term = torch.tensor(value)
    return ChangeVQALossBreakdown(
        total=term,
        answer=term,
        magnitude=term,
        delta=term,
        total_changed=term,
        weights=dict(T.DEFAULT_LOSS_WEIGHTS),
    )


def test_a_finite_loss_passes_the_guard() -> None:
    assert T.assert_finite_step(_breakdown(1.25), epoch=1, batch_index=0, n_batches=4) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_loss_stops_before_the_optimizer_step(value: float) -> None:
    from core.errors import SpecialistError

    with pytest.raises(SpecialistError):
        T.assert_finite_step(_breakdown(value), epoch=3, batch_index=4, n_batches=10)


def test_the_non_finite_message_locates_the_step_and_suggests_no_amp() -> None:
    """The message has to be actionable: which step, and what to try first."""
    from core.errors import SpecialistError

    with pytest.raises(SpecialistError) as excinfo:
        T.assert_finite_step(_breakdown(float("nan")), epoch=3, batch_index=4, n_batches=10)
    message = str(excinfo.value)
    assert "epoch 3" in message
    assert "batch 5/10" in message
    assert "--no-amp" in message
    # The weights block is noise in a failure message; the terms are not.
    assert "answer_ce" in message
    assert "weight_answer" not in message


def test_a_finite_model_passes_the_parameter_sweep() -> None:
    model = torch.nn.Linear(4, 2)
    assert T.assert_finite_parameters(model, epoch=1) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_non_finite_parameter_stops_the_epoch(value: float) -> None:
    from core.errors import SpecialistError

    model = torch.nn.Linear(4, 2)
    with torch.no_grad():
        model.weight[0, 0] = value
    with pytest.raises(SpecialistError) as excinfo:
        T.assert_finite_parameters(model, epoch=7)
    assert "epoch 7" in str(excinfo.value)
    assert "weight" in str(excinfo.value)
