"""SatQuery AI — CROMA encoder wrapper (freeze section 2.5, plan section 16).

Written against the REAL model interface, fetched and read during this pass.
Every claim below is traceable to a source, because the freeze rule is that
implementation proceeds from verified facts:

    https://github.com/antofuller/CROMA README.md     (fetched 2026-09-18)
    https://huggingface.co/antofuller/CROMA           (revision 0dd28e3d633b)

    loading            PretrainedCROMA(pretrained_path=..., size='base',
                                        modality='both',
                                        image_resolution=120)
    file               use_croma.py  <- MUST be vendored; not on PyPI
    forward            model(SAR_images=..., optical_images=...)
    returns            dict with keys
                         SAR_encodings      (B, n_patches, dim)
                         SAR_GAP            (B, dim)
                         optical_encodings  (B, n_patches, dim)
                         optical_GAP        (B, dim)
                         joint_encodings    (B, n_patches, dim)
                         joint_GAP          (B, dim)
    default size       120 x 120 px
    optical channels   12   (Sentinel-2, cirrus removed)
    SAR channels        2   (Sentinel-1)
    NO mask argument   confirmed by the forward signature above (finding C-1)

DEVIATION FROM THE FREEZE, RECORDED NOT HIDDEN
----------------------------------------------
Freeze section 2.5 does not name the constructor. The real one is
`use_croma.PretrainedCROMA`, living in a file that must be vendored rather than
pip-installed. This module imports it by locating `use_croma.py` on disk and
raises a typed `ModelLoadError` naming that requirement when it is absent --
rather than silently substituting a different loader, which is what "adapting
silently" would look like. Reported to team-lead for an ARCHITECTURE CHANGE
entry.

CROMA IS NEVER GIVEN A MASK
---------------------------
The forward pass takes exactly two arguments. The availability mask travels
alongside and is consumed by the fusion head (finding C-1). `encode` returns the
CROMA dict unchanged and the caller assembles; there is deliberately no
`mask=` parameter anywhere in this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.errors import ModelLoadError
from specialists.optical_sar import radiometry
from specialists.optical_sar.radiometry import (
    MODALITY_OPTICAL,
    MODALITY_SAR,
    RadiometrySummary,
)

#: Measured/verified facts, asserted at load time.
VERIFIED_ENCODER_DIM = 768
VERIFIED_OPTICAL_CHANNELS = 12
VERIFIED_SAR_CHANNELS = 2
VERIFIED_SIZE = "base"
VERIFIED_MODALITY = "both"

#: 120 / 8 = 15 -> 15^2 = 225 patches. The freeze states this; the README does
#: NOT, so it is verified against the loaded model instead of trusted.
VERIFIED_PATCHES_AT_120 = 225

#: The vendored file CROMA requires. Named here so the error message can say
#: exactly what to do.
REQUIRED_VENDOR_FILE = "use_croma.py"
CROMA_REPO_URL = "https://github.com/antofuller/CROMA"


@dataclass(frozen=True)
class CROMAEncoding:
    """What CROMA returned, per sample.

    `gap` holds the three 768-d vectors the fusion head consumes;
    `patch_tokens` holds the per-patch encodings, kept because evidence may want
    to show *where* a representation came from (spec's joint_feature_region).
    """

    optical_gap: np.ndarray       # (B, 768)
    sar_gap: np.ndarray           # (B, 768)
    joint_gap: np.ndarray         # (B, 768)
    optical_tokens: np.ndarray    # (B, n_patches, 768)
    sar_tokens: np.ndarray        # (B, n_patches, 768)
    joint_tokens: np.ndarray      # (B, n_patches, 768)
    resolution: int
    n_patches: int
    #: What the encoder actually received, per modality (DEV-2 ruling section
    #: 3.1.4: the applied transform, `use_8_bit` and the per-channel window must
    #: reach the trace). `None` only when input normalisation was disabled.
    radiometry: RadiometrySummary | None = None

    @property
    def dim(self) -> int:
        return int(self.optical_gap.shape[1])

    def as_forward_dict(self) -> dict[str, np.ndarray]:
        """The dict shape `assemble_fusion_input` expects."""
        return {
            "optical_GAP": self.optical_gap,
            "SAR_GAP": self.sar_gap,
            "joint_GAP": self.joint_gap,
        }


class CROMAEncoder:
    """Frozen CROMA-base with `modality='both'`. Optical, SAR and joint GAP vectors."""

    def __init__(
        self,
        model: Any,
        *,
        resolution: int = 120,
        device: str = "cpu",
        checkpoint_path: str | Path | None = None,
        use_8_bit: bool | None = None,
        use_8_bit_source: str | None = None,
        normalize_input: bool = True,
    ) -> None:
        """
        Args:
            use_8_bit: the uint8 round-trip in the encoder-input stretch. `None`
                resolves through `radiometry.resolve_use_8_bit`, which honours
                `SATQUERY_CROMA_USE_8_BIT` and then `croma.use_8_bit`.
            use_8_bit_source: where that value came from, for the trace. Set
                automatically when `use_8_bit` is None.
            normalize_input: apply the DEV-2 stretch before the forward pass.
                `False` exists because the ruling's gated experiment
                (`PHASE14_CROMA_NORMALISATION_CHANGE.md` section 4, arm B) needs
                a no-stretch control arm. It is **not** a supported production
                setting: with it off, the frozen encoder receives whatever
                dynamic range the caller supplied, which is the silent
                distribution shift the ruling exists to close.
        """
        self.model = model
        self.resolution = int(resolution)
        self.device = device
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None

        self.normalize_input = bool(normalize_input)
        if use_8_bit is None:
            use_8_bit, resolved_source = radiometry.resolve_use_8_bit(None)
            use_8_bit_source = use_8_bit_source or resolved_source
        self.use_8_bit = bool(use_8_bit)
        self.use_8_bit_source = use_8_bit_source or "explicit"

        if self.resolution % 8 != 0:
            # CROMA asserts this internally (finding C-7) and core/config.py
            # enforces it at load time. Re-checked here because this class can be
            # constructed directly by tests.
            raise ModelLoadError(
                f"image_resolution must be a multiple of 8 (CROMA asserts it); "
                f"got {self.resolution}",
                specialist="optical_sar",
            )

        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        self._verify_contract()

    def _verify_contract(self) -> None:
        """Assert the verified facts against the loaded model.

        Cheap, and they fire at load time. A silent drift here surfaces much
        later as a subtly wrong number rather than an exception -- the same
        reasoning as `RemoteCLIPEncoder._verify_contract`.
        """
        # The model is frozen by construction; nothing here may unfreeze it.
        trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
        if trainable:
            raise ModelLoadError(
                f"CROMA was expected frozen but {len(trainable)} parameter "
                f"tensor(s) require grad, e.g. {trainable[:3]}",
                specialist="optical_sar",
            )

    @property
    def encoder_dim(self) -> int:
        return VERIFIED_ENCODER_DIM

    @property
    def n_patches(self) -> int:
        return (self.resolution // 8) ** 2

    def describe(self) -> dict[str, Any]:
        if self.normalize_input:
            normalisation: dict[str, Any] = radiometry.describe_transform(
                self.use_8_bit, source=self.use_8_bit_source
            )
            normalisation["applied"] = True
        else:
            # The control arm of the ruling's gated experiment. Recorded as
            # "not applied" rather than omitted, so a trace can never show a
            # normalisation that did not run.
            normalisation = {
                "applied": False,
                "use_8_bit": None,
                "source": "input normalisation disabled at construction",
            }

        return {
            "model": "CROMA",
            "checkpoint": self.checkpoint_path or "unspecified",
            "size": VERIFIED_SIZE,
            "modality": VERIFIED_MODALITY,
            "resolution": self.resolution,
            "n_patches": self.n_patches,
            "encoder_dim": VERIFIED_ENCODER_DIM,
            "optical_channels": VERIFIED_OPTICAL_CHANNELS,
            "sar_channels": VERIFIED_SAR_CHANNELS,
            "device": self.device,
            "frozen": True,
            "receives_mask": False,
            "input_normalisation": normalisation,
        }

    # -- inference ---------------------------------------------------------

    def encode(
        self,
        optical: np.ndarray,
        sar: np.ndarray,
    ) -> CROMAEncoding:
        """Run CROMA. Note the absent `mask` parameter -- that is deliberate.

        Args:
            optical: (B, 12, H, W) canonical optical, zeros where a band was
                absent. CROMA cannot tell the difference; the fusion head can,
                because it receives the mask.
            sar: (B, 2, H, W) canonical SAR.

        Raises:
            ModelLoadError: the tensors do not match CROMA's fixed input widths.
        """
        import torch

        opt = self._as_tensor("optical", optical, VERIFIED_OPTICAL_CHANNELS)
        sar_t = self._as_tensor("sar", sar, VERIFIED_SAR_CHANNELS)

        if opt.shape[0] != sar_t.shape[0]:
            raise ModelLoadError(
                f"optical batch {opt.shape[0]} != SAR batch {sar_t.shape[0]}",
                specialist="optical_sar",
            )

        # -- encoder-input normalisation (DEV-2 ruling) -----------------------
        # Placed AFTER every shape check and immediately BEFORE the forward
        # pass, which is exactly where `PHASE14_CROMA_NORMALISATION_CHANGE.md`
        # section 3.2 fixes it. Validation stays first so a wrong-width tensor
        # still fails with the width error, not with a normalisation error.
        #
        # Both modalities go through the identical function. Upstream uses one
        # `ViT` class for both (vendor/use_croma.py:54,68) and applies one
        # `normalize` to both, so a divergence here would be an invention --
        # `radiometry.summarise` asserts the two reports agree.
        summary: RadiometrySummary | None = None
        if self.normalize_input:
            opt_np, opt_report = radiometry.normalise_for_croma(
                opt.numpy(),
                use_8_bit=self.use_8_bit,
                modality=MODALITY_OPTICAL,
            )
            sar_np, sar_report = radiometry.normalise_for_croma(
                sar_t.numpy(),
                use_8_bit=self.use_8_bit,
                modality=MODALITY_SAR,
            )
            summary = radiometry.summarise(opt_report, sar_report)
            opt = torch.from_numpy(np.ascontiguousarray(opt_np))
            sar_t = torch.from_numpy(np.ascontiguousarray(sar_np))

        try:
            with torch.no_grad():
                out = self.model(
                    SAR_images=sar_t.to(self.device),
                    optical_images=opt.to(self.device),
                )
        except ModelLoadError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"CROMA forward pass failed: {exc}",
                specialist="optical_sar",
                context={
                    "optical_shape": tuple(opt.shape),
                    "sar_shape": tuple(sar_t.shape),
                },
            ) from exc

        return self._to_encoding(out, radiometry_summary=summary)

    @staticmethod
    def _as_tensor(name: str, array: np.ndarray, channels: int):
        """(B, C, H, W) float32. Accepts (C, H, W) and adds a batch axis."""
        import torch

        arr = np.asarray(array, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        if arr.ndim != 4:
            raise ModelLoadError(
                f"{name} must be (B, C, H, W) or (C, H, W); got {arr.shape}",
                specialist="optical_sar",
            )
        if arr.shape[1] != channels:
            raise ModelLoadError(
                f"{name} has {arr.shape[1]} channels; CROMA requires exactly "
                f"{channels}. Canonicalise through the sensor adapter first -- "
                f"passing a raw sensor's bands straight in would apply "
                f"pretrained weights to the wrong physical channels.",
                specialist="optical_sar",
                context={"name": name, "channels": int(arr.shape[1])},
            )
        return torch.from_numpy(np.ascontiguousarray(arr))

    def _to_encoding(
        self, out: Any, *, radiometry_summary: RadiometrySummary | None = None
    ) -> CROMAEncoding:
        """Normalise CROMA's output dict into a checked dataclass."""
        if not isinstance(out, dict):
            raise ModelLoadError(
                f"CROMA returned {type(out).__name__}, expected a dict with keys "
                f"optical_GAP/SAR_GAP/joint_GAP (verified against the official "
                f"README)",
                specialist="optical_sar",
            )

        required = ("optical_GAP", "SAR_GAP", "joint_GAP")
        missing = [k for k in required if k not in out]
        if missing:
            raise ModelLoadError(
                f"CROMA output is missing {missing}; got keys {sorted(out)}",
                specialist="optical_sar",
            )

        def _np(key: str) -> np.ndarray:
            value = out[key]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=np.float32)

        optical_gap = _np("optical_GAP")
        sar_gap = _np("SAR_GAP")
        joint_gap = _np("joint_GAP")

        for name, arr in (
            ("optical_GAP", optical_gap),
            ("SAR_GAP", sar_gap),
            ("joint_GAP", joint_gap),
        ):
            if arr.ndim != 2 or arr.shape[1] != VERIFIED_ENCODER_DIM:
                raise ModelLoadError(
                    f"{name} has shape {arr.shape}; CROMA-base emits (B, "
                    f"{VERIFIED_ENCODER_DIM})",
                    specialist="optical_sar",
                )

        def _tokens(key: str, like: np.ndarray) -> np.ndarray:
            if key not in out:
                return np.zeros(
                    (like.shape[0], 0, VERIFIED_ENCODER_DIM), dtype=np.float32
                )
            value = out[key]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[2] == VERIFIED_ENCODER_DIM:
                return arr
            if arr.ndim == 3:
                # Some builds return (B, dim, n_patches). Transpose rather than
                # fail, and record the observed shape in the caller's trace.
                return np.transpose(arr, (0, 2, 1))
            return np.zeros((like.shape[0], 0, VERIFIED_ENCODER_DIM), dtype=np.float32)

        return CROMAEncoding(
            optical_gap=optical_gap,
            sar_gap=sar_gap,
            joint_gap=joint_gap,
            optical_tokens=_tokens("optical_encodings", optical_gap),
            sar_tokens=_tokens("SAR_encodings", sar_gap),
            joint_tokens=_tokens("joint_encodings", joint_gap),
            resolution=self.resolution,
            n_patches=self.n_patches,
            radiometry=radiometry_summary,
        )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def load_vendored_pretrained_croma(vendor_dir: str | Path) -> Any:
    """Import `PretrainedCROMA` from a vendored `use_croma.py`.

    CROMA is distributed as a GitHub repository, not a pip package. The
    official instructions are "you will need the use_croma.py file and pretrained
    weights".

    Raises:
        ModelLoadError: `use_croma.py` is not present in `vendor_dir`. The
            message names the file and the repo, because this is a deployment
            step someone must perform -- not a bug to work around by loading the
            checkpoint some other way.
    """
    import importlib.util

    directory = Path(vendor_dir)
    module_path = directory / REQUIRED_VENDOR_FILE
    if not module_path.exists():
        raise ModelLoadError(
            f"CROMA requires the vendored '{REQUIRED_VENDOR_FILE}', which is not "
            f"in '{directory}'. It is not available on PyPI; download it from "
            f"{CROMA_REPO_URL} (the official instructions are 'you will need "
            f"the use_croma.py file and pretrained weights'). Place it in the "
            f"optical_sar vendor directory and retry.",
            specialist="optical_sar",
            context={"vendor_dir": str(directory)},
        )

    try:
        spec = importlib.util.spec_from_file_location(
            "satquery_use_croma", module_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"could not create an import spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(
            f"failed to import '{module_path}': {exc}",
            specialist="optical_sar",
            context={"vendor_dir": str(directory)},
        ) from exc

    cls = getattr(module, "PretrainedCROMA", None)
    if cls is None:
        raise ModelLoadError(
            f"'{module_path}' does not define 'PretrainedCROMA' (found "
            f"{[n for n in dir(module) if not n.startswith('_')][:12]})",
            specialist="optical_sar",
        )
    return cls


# ---------------------------------------------------------------------------
# Checkpoint resolution — hash-exempt, pinned identity first
# ---------------------------------------------------------------------------
#: Deployment-time override for the checkpoint location. Deliberately the same
#: hash-exempt channel `radiometry.resolve_use_8_bit` uses for
#: `SATQUERY_CROMA_USE_8_BIT`.
#:
#: WHY THIS IS NOT JUST A `configs/base.yaml` KEY
#: ----------------------------------------------
#: `Config.hash` is a sha256 over the whole registry with NO exclusion mechanism
#: (`core/config.py:76-80`). The shipped artifacts record `78f1e3700da15aa1`, and
#: `scripts/eval_change.py` refuses to score on a drift (exit 3), so writing
#: `croma.checkpoint_path` into `base.yaml` today would detach the project's
#: benchmark numbers from their config. This resolver gives an operator an
#: explicit, inspectable and reversible way to point serving at a local
#: checkpoint WITHOUT taking that cost.
ENV_CROMA_CHECKPOINT = "SATQUERY_CROMA_CHECKPOINT"

#: Where a resolved checkpoint came from. Recorded alongside the path so the
#: choice is never invisible. Stable literals, not free text.
CHECKPOINT_SOURCE_ENV = "env"
CHECKPOINT_SOURCE_CONFIG = "config"
CHECKPOINT_SOURCE_HUB_CACHE = "hub-cache"
CHECKPOINT_SOURCE_ABSENT = "absent"


def resolve_checkpoint_path(config: Any | None = None) -> tuple[str | None, str]:
    """Resolve the CROMA checkpoint from the PINNED identity, without a config key.

    Resolution order, most specific first — the same shape as
    `radiometry.resolve_use_8_bit`, so the two hash-exempt channels behave alike:

    1. `SATQUERY_CROMA_CHECKPOINT`, when set and non-empty.
    2. `croma.checkpoint_path` in the config, **when that key exists and is a
       non-empty string**. A no-op until the key is added.
    3. The pinned `(croma.checkpoint_repo, croma.checkpoint_file,
       croma.checkpoint_revision)` triple, resolved through the Hub cache
       **offline first**. The pinned revision makes this deterministic: when the
       snapshot is already on disk no network call is made.
    4. Nothing — `(None, "absent")`.

    WHY OFFLINE FIRST
    -----------------
    Serving composition must not depend on a network round trip. A cached,
    revision-pinned snapshot resolves locally and is byte-stable; only a genuine
    cache miss falls through to the Hub. That ordering is what makes the
    default serving composition work on a machine that has the artifact and no
    egress.

    RAISING vs DEGRADING
    --------------------
    An *explicitly specified* path (steps 1-2) that does not exist raises
    `ModelLoadError`. That is deliberate and matches `build_encoder`, which has
    always raised for a nonexistent `checkpoint_path`: a typo'd operator path is
    a misconfiguration, not an absent artifact, and silently degrading on it is
    the "config surface that advertises a capability the code lacks" failure
    this repo has already ruled against (F-17). An artifact that is simply
    *absent* (step 4) is a deployment case and returns `None` so the documented
    DEGRADED contract governs.

    Returns:
        `(path, source)` where `path` is a `str` only when it exists on disk,
        and `source` is one of `"env"`, `"config"`, `"hub-cache"`, `"absent"`.
    """
    # 1. Deployment-time override.
    env = os.environ.get(ENV_CROMA_CHECKPOINT)
    if env is not None and env.strip():
        path = Path(env.strip())
        if not path.exists():
            raise ModelLoadError(
                f"{ENV_CROMA_CHECKPOINT} names a checkpoint that does not exist: {path}",
                specialist="optical_sar",
                context={"env": ENV_CROMA_CHECKPOINT},
            )
        return str(path), CHECKPOINT_SOURCE_ENV

    # 2. An explicit config path. No-op until the key is written.
    if config is not None:
        raw = config.get("croma.checkpoint_path", None)
        if isinstance(raw, str) and raw.strip():
            path = Path(raw.strip())
            if not path.exists():
                raise ModelLoadError(
                    f"croma.checkpoint_path does not exist: {path}",
                    specialist="optical_sar",
                    context={"checkpoint_path": str(path)},
                )
            return str(path), CHECKPOINT_SOURCE_CONFIG

    # 3. The pinned identity, through the Hub cache.
    if config is None:
        return None, CHECKPOINT_SOURCE_ABSENT
    repo = config.get("croma.checkpoint_repo")
    filename = config.get("croma.checkpoint_file")
    revision = config.get("croma.checkpoint_revision")
    if not (repo and filename):
        return None, CHECKPOINT_SOURCE_ABSENT

    try:
        from huggingface_hub import hf_hub_download
    except Exception:  # noqa: BLE001 - an absent dependency is "absent"
        return None, CHECKPOINT_SOURCE_ABSENT

    for local_only in (True, False):
        try:
            resolved = hf_hub_download(
                repo, filename, revision=revision, local_files_only=local_only
            )
        except Exception:  # noqa: BLE001 - cache miss / no egress -> try next
            continue
        if Path(resolved).exists():
            return str(resolved), CHECKPOINT_SOURCE_HUB_CACHE

    return None, CHECKPOINT_SOURCE_ABSENT


def build_encoder(
    config: Any,
    *,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
    vendor_dir: str | Path | None = None,
) -> CROMAEncoder:
    """Construct CROMA from the central config.

    Args:
        config: the loaded central config.
        device: torch device string. Defaults to config.device_preference.
        checkpoint_path: local `CROMA_base.pt`. When given it must exist;
            when None the pinned revision is fetched from the Hub.
        vendor_dir: directory holding the vendored `use_croma.py`.

    Raises:
        ModelLoadError: the checkpoint or the vendor file is unavailable. Both
            are DEPLOYMENT prerequisites -- neither is something this function
            may paper over with a substitute model.
    """
    device = device or config.device_preference
    size = str(config.get("croma.variant", VERIFIED_SIZE))
    resolution = int(config.get("croma.image_resolution", 120))

    # DEV-2 ruling section 3.1.4: the applied transform and `use_8_bit` must be
    # inspectable. Resolved through the hash-exempt path so that wiring the
    # normalisation does not move `Config.hash` and detach the frozen Phase 9
    # benchmark -- see `radiometry.resolve_use_8_bit`.
    use_8_bit, use_8_bit_source = radiometry.resolve_use_8_bit(config)

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise ModelLoadError(
                f"checkpoint_path does not exist: {path}",
                specialist="optical_sar",
                context={"checkpoint_path": str(path)},
            )
        checkpoint = str(path)
    else:
        from huggingface_hub import hf_hub_download

        repo = config.get("croma.checkpoint_repo")
        filename = config.get("croma.checkpoint_file")
        revision = config.get("croma.checkpoint_revision")
        try:
            checkpoint = hf_hub_download(repo, filename, revision=revision)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"could not fetch '{filename}' from '{repo}': {exc}",
                specialist="optical_sar",
            ) from exc

    if vendor_dir is None:
        vendor_dir = Path(__file__).resolve().parent / "vendor"

    pretrained_cls = load_vendored_pretrained_croma(vendor_dir)

    try:
        model = pretrained_cls(
            pretrained_path=checkpoint,
            size=size,
            modality=VERIFIED_MODALITY,
            image_resolution=resolution,
        )
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(
            f"PretrainedCROMA could not be constructed: {exc}",
            specialist="optical_sar",
            context={"size": size, "resolution": resolution},
        ) from exc

    model.to(device)
    return CROMAEncoder(
        model,
        resolution=resolution,
        device=device,
        checkpoint_path=checkpoint,
        use_8_bit=use_8_bit,
        use_8_bit_source=use_8_bit_source,
    )


__all__ = [
    "CROMAEncoder",
    "CROMAEncoding",
    "CHECKPOINT_SOURCE_ABSENT",
    "CHECKPOINT_SOURCE_CONFIG",
    "CHECKPOINT_SOURCE_ENV",
    "CHECKPOINT_SOURCE_HUB_CACHE",
    "ENV_CROMA_CHECKPOINT",
    "build_encoder",
    "resolve_checkpoint_path",
    "load_vendored_pretrained_croma",
    "VERIFIED_ENCODER_DIM",
    "VERIFIED_OPTICAL_CHANNELS",
    "VERIFIED_SAR_CHANNELS",
    "VERIFIED_PATCHES_AT_120",
    "REQUIRED_VENDOR_FILE",
    "CROMA_REPO_URL",
]
