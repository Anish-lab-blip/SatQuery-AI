"""CROMA wrapper — the interface verified against the real model.

Every fact asserted here was read from the official repository during this pass:

    https://github.com/antofuller/CROMA  README.md
    https://huggingface.co/antofuller/CROMA  revision 0dd28e3d633b

The reason these assertions exist rather than the facts merely being documented:
the freeze declares a contract for a model nobody in this repository has run.
If the pinned revision ever changes shape, the first symptom would otherwise be a
silently wrong number much later. Two of the checks below (the mask-parameter
absence and the output key names) are the load-bearing ones, because both are
architectural rules rather than conveniences.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.errors import ModelLoadError

pytest.importorskip("torch")

from specialists.optical_sar.croma import (  # noqa: E402
    CROMA_REPO_URL,
    REQUIRED_VENDOR_FILE,
    VERIFIED_ENCODER_DIM,
    VERIFIED_MODALITY,
    VERIFIED_OPTICAL_CHANNELS,
    VERIFIED_PATCHES_AT_120,
    VERIFIED_SAR_CHANNELS,
    VERIFIED_SIZE,
    CROMAEncoder,
    load_vendored_pretrained_croma,
)


class _StubCROMA:
    """Emits the REAL output keys with the REAL shapes.

    Verified against the README's documented return value, so the wrapper's
    parsing is exercised against what CROMA actually returns.
    """

    def __init__(self, *, dim: int = 768, n_patches: int = 225):
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

    def to(self, device):  # noqa: ANN001, ANN201
        return self

    def __call__(self, *, SAR_images, optical_images):  # noqa: N803
        import torch

        self.received = {
            "SAR_images": SAR_images,
            "optical_images": optical_images,
            "kwargs": {"SAR_images", "optical_images"},
        }
        batch = optical_images.shape[0]
        return {
            "optical_GAP": torch.ones(batch, self.dim),
            "SAR_GAP": torch.ones(batch, self.dim) * 2,
            "joint_GAP": torch.ones(batch, self.dim) * 3,
            "optical_encodings": torch.ones(batch, self.n_patches, self.dim),
            "SAR_encodings": torch.ones(batch, self.n_patches, self.dim),
            "joint_encodings": torch.ones(batch, self.n_patches, self.dim),
        }


def _encoder(**kwargs) -> CROMAEncoder:
    return CROMAEncoder(_StubCROMA(**kwargs), resolution=120, device="cpu")


# ---------------------------------------------------------------------------
# The verified constants
# ---------------------------------------------------------------------------


def test_verified_constants_match_the_freeze():
    """Freeze section 2.5's numbers, restated as assertions."""
    assert VERIFIED_SIZE == "base"
    assert VERIFIED_MODALITY == "both"
    assert VERIFIED_ENCODER_DIM == 768
    assert VERIFIED_OPTICAL_CHANNELS == 12
    assert VERIFIED_SAR_CHANNELS == 2
    assert VERIFIED_PATCHES_AT_120 == 225


def test_225_patches_at_120_pixels():
    """120 % 8 == 0 -> 15 x 15 = 225. Finding C-7."""
    assert _encoder().n_patches == 225
    assert VERIFIED_PATCHES_AT_120 == (120 // 8) ** 2


def test_resolution_must_be_a_multiple_of_eight():
    """CROMA asserts this internally; refusing early gives a better message."""
    with pytest.raises(ModelLoadError):
        CROMAEncoder(_StubCROMA(), resolution=121)


# ---------------------------------------------------------------------------
# Finding C-1: CROMA never receives a mask
# ---------------------------------------------------------------------------


def test_encode_takes_no_mask_parameter():
    """THE C-1 ASSERTION.

    CROMA is a masked autoencoder. Handing it an availability mask invites it to
    reconstruct the missing channels, which is the fabrication the sensor adapter
    exists to prevent. The mask is consumed by the fusion head instead.
    """
    import inspect

    params = set(inspect.signature(CROMAEncoder.encode).parameters)
    assert params == {"self", "optical", "sar"}
    for forbidden in ("mask", "availability_mask", "optical_mask", "sar_mask"):
        assert forbidden not in params


def test_forward_call_passes_only_the_two_image_tensors():
    """Verified at the call site, not just the signature."""
    stub = _StubCROMA()
    encoder = CROMAEncoder(stub, resolution=120, device="cpu")

    encoder.encode(
        np.zeros((1, 12, 120, 120), dtype=np.float32),
        np.zeros((1, 2, 120, 120), dtype=np.float32),
    )

    assert stub.received["kwargs"] == {"SAR_images", "optical_images"}


def test_channel_widths_are_enforced():
    """A 4-band optical tensor cannot be passed straight in.

    This is the guard that stops a raw sensor's bands reaching pretrained weights
    without going through the adapter.
    """
    encoder = _encoder()

    with pytest.raises(ModelLoadError) as excinfo:
        encoder.encode(
            np.zeros((1, 4, 120, 120), dtype=np.float32),   # Cartosat, not 12
            np.zeros((1, 2, 120, 120), dtype=np.float32),
        )
    assert "12" in str(excinfo.value.detail)

    with pytest.raises(ModelLoadError):
        encoder.encode(
            np.zeros((1, 12, 120, 120), dtype=np.float32),
            np.zeros((1, 3, 120, 120), dtype=np.float32),   # 3, not 2
        )


def test_batch_mismatch_between_modalities_is_rejected():
    encoder = _encoder()
    with pytest.raises(ModelLoadError):
        encoder.encode(
            np.zeros((2, 12, 120, 120), dtype=np.float32),
            np.zeros((1, 2, 120, 120), dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def test_encodings_expose_the_three_gap_vectors():
    encoder = _encoder()
    out = encoder.encode(
        np.zeros((1, 12, 120, 120), dtype=np.float32),
        np.zeros((1, 2, 120, 120), dtype=np.float32),
    )

    assert out.optical_gap.shape == (1, 768)
    assert out.sar_gap.shape == (1, 768)
    assert out.joint_gap.shape == (1, 768)
    assert out.n_patches == 225
    assert out.dim == 768


def test_forward_dict_uses_the_verified_key_names():
    """`assemble_fusion_input` expects these exact keys."""
    encoder = _encoder()
    out = encoder.encode(
        np.zeros((1, 12, 120, 120), dtype=np.float32),
        np.zeros((1, 2, 120, 120), dtype=np.float32),
    )

    as_dict = out.as_forward_dict()
    assert set(as_dict) == {"optical_GAP", "SAR_GAP", "joint_GAP"}


def test_patch_tokens_are_carried_through():
    """Kept because evidence may want to show where a representation came from."""
    encoder = _encoder()
    out = encoder.encode(
        np.zeros((1, 12, 120, 120), dtype=np.float32),
        np.zeros((1, 2, 120, 120), dtype=np.float32),
    )

    assert out.optical_tokens.shape == (1, 225, 768)


def test_a_missing_output_key_raises_with_the_real_key_names():
    """A CROMA build that renamed a key must fail loudly, now."""
    class _Renamed(_StubCROMA):
        def __call__(self, *, SAR_images, optical_images):  # noqa: N803
            out = super().__call__(
                SAR_images=SAR_images, optical_images=optical_images
            )
            out["SAR_GAP_vector"] = out.pop("SAR_GAP")
            return out

    encoder = CROMAEncoder(_Renamed(), resolution=120, device="cpu")
    with pytest.raises(ModelLoadError) as excinfo:
        encoder.encode(
            np.zeros((1, 12, 120, 120), dtype=np.float32),
            np.zeros((1, 2, 120, 120), dtype=np.float32),
        )
    assert "SAR_GAP" in str(excinfo.value.detail)


def test_a_non_dict_output_raises():
    class _ListOutput(_StubCROMA):
        def __call__(self, *, SAR_images, optical_images):  # noqa: N803
            return [1, 2, 3]

    encoder = CROMAEncoder(_ListOutput(), resolution=120, device="cpu")
    with pytest.raises(ModelLoadError):
        encoder.encode(
            np.zeros((1, 12, 120, 120), dtype=np.float32),
            np.zeros((1, 2, 120, 120), dtype=np.float32),
        )


def test_transposed_patch_tokens_are_normalised():
    """Some builds return (B, dim, n_patches). Transpose rather than fail."""
    class _Transposed(_StubCROMA):
        def __call__(self, *, SAR_images, optical_images):  # noqa: N803
            out = super().__call__(
                SAR_images=SAR_images, optical_images=optical_images
            )
            out["optical_encodings"] = out["optical_encodings"].permute(0, 2, 1)
            return out

    encoder = CROMAEncoder(_Transposed(), resolution=120, device="cpu")
    out = encoder.encode(
        np.zeros((1, 12, 120, 120), dtype=np.float32),
        np.zeros((1, 2, 120, 120), dtype=np.float32),
    )
    assert out.optical_tokens.shape == (1, 225, 768)


# ---------------------------------------------------------------------------
# Frozen encoder
# ---------------------------------------------------------------------------


def test_encoder_is_frozen():
    """CROMA's weights are never trained.

    The trainable parts are the fusion head (Phase 12) and, separately, the
    grounding projection. Unfreezing CROMA would be an architecture change.
    """
    encoder = _encoder()
    trainable = [
        name
        for name, param in encoder.model.named_parameters()
        if param.requires_grad
    ]
    assert trainable == []


def test_a_trainable_parameter_is_refused_at_construction():
    class _Unfrozen(_StubCROMA):
        def named_parameters(self):
            import torch

            return iter([("w", torch.zeros(1, requires_grad=True))])

    with pytest.raises(ModelLoadError):
        CROMAEncoder(_Unfrozen(), resolution=120, device="cpu")


# ---------------------------------------------------------------------------
# The vendored-file requirement (a recorded deviation)
# ---------------------------------------------------------------------------


def test_missing_vendor_file_says_exactly_what_to_do(tmp_path):
    """CROMA is not on PyPI. `use_croma.py` must be vendored, and the error must
    say so rather than failing obscurely.

    Reported to team-lead as DEV-1: the freeze does not name the constructor, and
    the real one lives in a file that must be downloaded. This test pins the
    message so the requirement cannot be quietly lost.
    """
    with pytest.raises(ModelLoadError) as excinfo:
        load_vendored_pretrained_croma(tmp_path)

    detail = str(excinfo.value.detail)
    assert REQUIRED_VENDOR_FILE in detail
    assert CROMA_REPO_URL in detail


def test_description_states_the_mask_rule():
    """The describe() output is what the trace records. It must say CROMA gets
    no mask, so the C-1 decision is visible in every run."""
    description = _encoder().describe()
    assert description["receives_mask"] is False
    assert description["modality"] == "both"
    assert description["size"] == "base"
    assert description["frozen"] is True
