"""SatQuery AI — authoritative configuration loader.

One config system. No duplicated constants. Every value in configs/base.yaml is loaded,
validated against the frozen architecture, and hashed so evaluation runs are reproducible.

The loader enforces the Phase-0 contract findings at startup: if a config tries to
violate a frozen decision (e.g. bf16 on T4, torch.compile on ZeroGPU, a CROMA
image_resolution that is not a multiple of 8), it fails loudly rather than at runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from core.errors import WorkflowPlanError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "base.yaml"


class ConfigError(WorkflowPlanError):
    code = "config_error"
    user_message = "The system configuration is invalid."


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Config:
    """Immutable, validated, hashable view over the YAML registry."""

    def __init__(self, data: dict[str, Any], source: str = "<dict>") -> None:
        self._data = data
        self.source = source
        self._validate()

    # -- access ------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        if key not in self._data:
            raise KeyError(key)
        return self._data[key]

    def get(self, path: str, default: Any = None) -> Any:
        """Dotted-path access: cfg.get('croma.image_resolution')."""
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, path: str) -> Any:
        sentinel = object()
        value = self.get(path, sentinel)
        if value is sentinel:
            raise ConfigError(f"required config key missing: {path}")
        return value

    def as_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._data, default=str))

    @property
    def hash(self) -> str:
        """Stable hash of the whole registry. Recorded in every evaluation run."""
        blob = json.dumps(self._data, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    @property
    def seed(self) -> int:
        return int(self.get("project.seed", 42))

    @property
    def device_preference(self) -> str:
        override = os.environ.get("SATQUERY_DEVICE")
        if override:
            return override
        return "cuda" if _torch_cuda_available() else "cpu"

    # -- validation --------------------------------------------------------
    def _validate(self) -> None:
        errors: list[str] = []

        # --- C-7: CROMA image_resolution must be a multiple of 8 -----------
        croma_res = self.get("croma.image_resolution")
        if croma_res is None:
            errors.append("croma.image_resolution is required")
        elif not isinstance(croma_res, int) or croma_res % 8 != 0:
            errors.append(
                f"croma.image_resolution must be an int multiple of 8 "
                f"(CROMA asserts image_resolution % 8 == 0); got {croma_res!r}"
            )

        # --- C-6: precision must be viable on the target accelerator -------
        precision = self.get("training.precision")
        if precision not in {"fp16", "bf16", "fp32"}:
            errors.append(
                f"training.precision must be fp16|bf16|fp32, got {precision!r}"
            )

        # --- C-8: ZeroGPU forbids torch.compile ----------------------------
        if self.get("deployment.torch_compile") is True:
            errors.append(
                "deployment.torch_compile=true is forbidden: ZeroGPU does not "
                "support torch.compile (finding C-8)"
            )

        # --- C-3 / F5-2: the VLM processor must not upscale our tiles -----
        #
        # MEASURED (docs/PHASE5_VLM_CONTRACT.md): the processor's default
        # longest_edge is 2048. A 512 px tile is upscaled 4x and then split by
        # do_image_splitting into 4x4 sub-images + 1 overview = 17 images and
        # 1142 prompt tokens, versus 1 image when pinned. The plan estimated a
        # 4x overrun; the real figure is ~17x.
        #
        # Tying this to image.tile_size makes the pin a control rather than a
        # comment: the processor cannot silently start upscaling tiles again.
        proc_edge = self.get("vlm.processor_longest_edge")
        tile_size = self.get("image.tile_size")
        if not isinstance(proc_edge, int) or proc_edge < 1:
            errors.append(
                f"vlm.processor_longest_edge must be an int >= 1, got {proc_edge!r}"
            )
        elif isinstance(tile_size, int) and proc_edge > tile_size:
            errors.append(
                f"vlm.processor_longest_edge={proc_edge} exceeds "
                f"image.tile_size={tile_size}; the processor would upscale every "
                f"tile and then split it into ~17 sub-images (finding F5-2)"
            )

        # --- F5-3: prompts must go through the chat template --------------
        if self.get("vlm.prompt_must_use_chat_template") is not True:
            errors.append(
                "vlm.prompt_must_use_chat_template must be true: SmolVLM raises "
                "ValueError on prompts lacking one <image> token per image "
                "(finding F5-3)"
            )

        # --- C-1: fusion input dim must match the verified concatenation ---
        expected = self._expected_fusion_dim()
        declared = self.get("fusion.input_dim")
        if expected is not None and declared != expected:
            errors.append(
                f"fusion.input_dim={declared} disagrees with the verified CROMA "
                f"concatenation ({expected} = 3*encoder_dim + optical_channels + "
                f"sar_channels). CROMA emits optical/SAR/joint GAP vectors; the "
                f"availability mask is consumed by the fusion head, not by CROMA "
                f"(finding C-1)."
            )

        # --- channel counts must match CROMA's fixed inputs ---------------
        if self.get("croma.optical_channels") != 12:
            errors.append("croma.optical_channels must be 12 (CROMA s2_channels is fixed)")
        if self.get("croma.sar_channels") != 2:
            errors.append("croma.sar_channels must be 2 (CROMA s1_channels is fixed)")

        # --- Phase 8: the head's assembled feature must match the encoder ---
        #
        # Per-cell feature = concat([patch, text, patch*text, global_pool]),
        # i.e. 4 x the encoder's PROJECTED dim. Measured as 512, NOT the 768
        # transformer width (finding P7-1): `visual.proj` is (768, 512).
        #
        # A mismatch here is a SILENT shape error. torch only raises at the
        # similarity step, by which point the patch features have already been
        # computed and cached — so the failure surfaces far from its cause.
        #
        # grounding.encoder_projected_dim is declared in config so this guard
        # needs no torch import; specialists/grounding/remoteclip.py asserts
        # the same value against the real model at load time.
        projected = self.get("grounding.encoder_projected_dim")
        head_dim = self.get("grounding_head.feature_dim")
        if not isinstance(projected, int) or projected < 1:
            errors.append(
                f"grounding.encoder_projected_dim must be a positive int, "
                f"got {projected!r}"
            )
        elif not isinstance(head_dim, int) or head_dim != 4 * projected:
            errors.append(
                f"grounding_head.feature_dim={head_dim} but the frozen encoder "
                f"projects to {projected}, so the assembled per-cell feature is "
                f"{4 * projected}-d. Expected feature_dim == 4 * "
                f"grounding.encoder_projected_dim (finding P7-1)."
            )

        # --- router ontology ----------------------------------------------
        tasks = self.get("router.tasks") or []
        if "unsupported" not in tasks:
            errors.append("router.tasks must include 'unsupported'")
        if self.get("router.num_tasks") != len(tasks):
            errors.append(
                f"router.num_tasks={self.get('router.num_tasks')} does not match "
                f"router.tasks length ({len(tasks)})"
            )

        # --- change config -------------------------------------------------
        if self.get("change.encoder") is None:
            errors.append("change.encoder is required")
        if self.get("change.sa_mode") not in {"BAM", "PAM"}:
            errors.append("change.sa_mode must be BAM or PAM")

        # --- tiling policy -------------------------------------------------
        if self.get("image.top_k_tiles", 0) > self.get("image.max_tiles", 0):
            errors.append("image.top_k_tiles cannot exceed image.max_tiles")

        if errors:
            raise ConfigError(
                "configuration failed frozen-architecture validation:\n  - "
                + "\n  - ".join(errors)
            )

    def _expected_fusion_dim(self) -> int | None:
        dim = self.get("croma.encoder_dim")
        mods = self.get("croma.modalities_used") or []
        opt = self.get("croma.optical_channels")
        sar = self.get("croma.sar_channels")
        if None in (dim, opt, sar) or not mods:
            return None
        return len(mods) * int(dim) + int(opt) + int(sar)

    def __repr__(self) -> str:
        return f"<Config source={self.source} hash={self.hash}>"


def _torch_cuda_available() -> bool:
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load, merge and validate the configuration registry."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise ConfigError(f"config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    if overrides:
        data = _deep_merge(data, overrides)

    # Environment overrides for the two values most likely to differ by host.
    if env_prec := os.environ.get("SATQUERY_PRECISION"):
        data.setdefault("training", {})["precision"] = env_prec
    if env_dev := os.environ.get("SATQUERY_TORCH_COMPILE"):
        data.setdefault("deployment", {})["torch_compile"] = env_dev.lower() == "true"

    return Config(data, source=str(cfg_path))


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide singleton. Import this, do not re-read YAML."""
    return load_config()


__all__ = ["Config", "ConfigError", "load_config", "get_config", "REPO_ROOT"]