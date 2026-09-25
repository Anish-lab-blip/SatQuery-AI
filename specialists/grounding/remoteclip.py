"""SatQuery AI — RemoteCLIP encoder wrapper for grounding.

Written against measured facts, not the model card
(docs/PHASE7_GROUNDING_CONTRACT.md):

    load route        pretrained=<local path>          works, open-clip-torch 3.3.0
    parameters        151,277,313
    patch size        32
    transformer width 768   (positional_embedding is (N+1, 768))
    PROJECTED dim     512   <- this is what matches the text tower
    text dim          512
    tokens @224       7x7  = 49   (+1 CLS)
    tokens @448       14x14 = 196 (+1 CLS), pos-embed resized to (197, 768)

The 768-vs-512 distinction is the one that bites. `visual.positional_embedding`
is 768 wide and `visual.proj` is (768, 512); the embeddings the text tower can
be compared against are the PROJECTED ones. Using 768 anywhere here would be a
shape error that torch would only surface at the similarity computation, by
which point the patch features have already been computed and cached.

This module is deliberately narrow. It encodes images and text, and it exposes
per-patch tokens at a chosen resolution. It does NOT decide boxes — that is the
grounding head's job (Phase 8). Zero-shot patch-text similarity lives in
`inference.py` and is a baseline, not the product.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.errors import ModelLoadError

#: Measured, not assumed. Asserted against at load time.
VERIFIED_PATCH_SIZE = 32
VERIFIED_TRANSFORMER_WIDTH = 768
VERIFIED_PROJECTED_DIM = 512
VERIFIED_PARAMETERS = 151_277_313

#: Supported input resolutions. Both were measured to work; which one to USE is
#: an empirical question answered by the resolution experiment, not a default
#: chosen here. Neither is marked preferred.
SUPPORTED_RESOLUTIONS: tuple[int, ...] = (224, 448)


@dataclass(frozen=True)
class EncodedImage:
    """Per-image encoding at one resolution.

    `patch_tokens` is (grid_h * grid_w, projected_dim) -- row-major, so token
    index `r * grid_w + c` is the patch at row r, column c.
    """

    cls: np.ndarray                 # (projected_dim,)
    patch_tokens: np.ndarray        # (n_patches, projected_dim)
    grid: tuple[int, int]           # (grid_h, grid_w)
    resolution: int

    @property
    def n_patches(self) -> int:
        return int(self.patch_tokens.shape[0])

    @property
    def dim(self) -> int:
        return int(self.patch_tokens.shape[1])

    def patch_index(self, row: int, col: int) -> int:
        """Row-major index of the patch at (row, col). Bounds-checked."""
        grid_h, grid_w = self.grid
        if not (0 <= row < grid_h and 0 <= col < grid_w):
            raise IndexError(f"patch ({row},{col}) outside grid {self.grid}")
        return row * grid_w + col

    def token_boxes(self) -> np.ndarray:
        """Normalized [x1,y1,x2,y2] for every patch token.

        This is the localization floor made explicit: each token covers
        1/grid_w by 1/grid_h of the image. At 224 that is 1/7 (~0.143) of the
        width; at 448 it is 1/14 (~0.071).
        """
        grid_h, grid_w = self.grid
        boxes = np.zeros((grid_h * grid_w, 4), dtype=np.float64)
        for r in range(grid_h):
            for c in range(grid_w):
                i = self.row_major(r, c, grid_h, grid_w)
                boxes[i] = [
                    c / grid_w,
                    r / grid_h,
                    (c + 1) / grid_w,
                    (r + 1) / grid_h,
                ]
        return boxes

    @staticmethod
    def row_major(r: int, c: int, grid_h: int, grid_w: int) -> int:
        return r * grid_w + c


class RemoteCLIPEncoder:
    """Frozen RemoteCLIP ViT-B/32. Image and text encoding at a chosen resolution."""

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = "ViT-B-32",
        resolution: int = 224,
        device: str = "cpu",
    ) -> None:
        if resolution not in SUPPORTED_RESOLUTIONS:
            raise ModelLoadError(
                f"resolution {resolution} is not in the verified set "
                f"{list(SUPPORTED_RESOLUTIONS)}. Both were measured to load "
                f"correctly; an unmeasured resolution must not be assumed to.",
                specialist="grounding",
            )

        self.model_name = model_name
        self.resolution = resolution
        self.device = device
        self.checkpoint_path = checkpoint_path

        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover
            raise ModelLoadError(
                f"open-clip-torch is not installed: {exc}", specialist="grounding"
            ) from exc

        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=str(checkpoint_path),
                force_image_size=resolution if resolution != 224 else None,
            )
            tokenizer = open_clip.get_tokenizer(model_name)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"could not load RemoteCLIP '{model_name}' from "
                f"'{checkpoint_path}' at {resolution}px: {exc}",
                specialist="grounding",
                context={"resolution": resolution},
            ) from exc

        model.eval()
        model.to(device)
        for param in model.parameters():
            param.requires_grad_(False)

        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        self._verify_contract()

    def _verify_contract(self) -> None:
        """Assert the measured facts still hold for this loaded model.

        These are cheap checks and they fire at load time. A silent shape drift
        here would surface much later, as a similarity score that is subtly
        wrong rather than an exception.
        """
        visual = self.model.visual

        patch_size = getattr(visual, "patch_size", None)
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]
        if patch_size != VERIFIED_PATCH_SIZE:
            raise ModelLoadError(
                f"patch_size is {patch_size}, expected {VERIFIED_PATCH_SIZE}",
                specialist="grounding",
            )

        pos = getattr(visual, "positional_embedding", None)
        if pos is None or pos.shape[-1] != VERIFIED_TRANSFORMER_WIDTH:
            raise ModelLoadError(
                f"transformer width is {getattr(pos, 'shape', None)}, expected "
                f"{VERIFIED_TRANSFORMER_WIDTH}",
                specialist="grounding",
            )

        expected_positions = (self.resolution // VERIFIED_PATCH_SIZE) ** 2 + 1
        if pos.shape[0] != expected_positions:
            raise ModelLoadError(
                f"positional_embedding has {pos.shape[0]} entries; at "
                f"{self.resolution}px with patch {VERIFIED_PATCH_SIZE} the grid "
                f"should need {expected_positions} (1 CLS + "
                f"{(self.resolution // VERIFIED_PATCH_SIZE) ** 2} patches)",
                specialist="grounding",
            )

        proj = getattr(visual, "proj", None)
        if proj is not None and proj.shape[1] != VERIFIED_PROJECTED_DIM:
            raise ModelLoadError(
                f"visual.proj maps to {proj.shape[1]}, expected "
                f"{VERIFIED_PROJECTED_DIM}",
                specialist="grounding",
            )

    @property
    def grid_size(self) -> int:
        """Tokens per side at this resolution."""
        return self.resolution // VERIFIED_PATCH_SIZE

    @property
    def embedding_dim(self) -> int:
        return VERIFIED_PROJECTED_DIM

    # -- image -------------------------------------------------------------

    @torch.no_grad()
    def encode_image(self, image: Any) -> EncodedImage:
        """Encode one PIL image to CLS + per-patch projected tokens.

        Replicates open_clip's vision forward up to `ln_post` and returns the
        per-patch tokens BEFORE the final pooling, which `visual.forward` does
        not expose. Verified to reproduce the same CLS vector.
        """
        visual = self.model.visual
        x = self.preprocess(image).unsqueeze(0).to(self.device)

        patches = visual.conv1(x)
        grid_h, grid_w = int(patches.shape[-2]), int(patches.shape[-1])
        patches = patches.reshape(patches.shape[0], patches.shape[1], -1)
        patches = patches.permute(0, 2, 1)

        cls = visual.class_embedding.to(patches.dtype).to(self.device)
        cls = cls + torch.zeros(
            patches.shape[0], 1, patches.shape[-1],
            dtype=patches.dtype, device=self.device,
        )
        tokens = torch.cat([cls, patches], dim=1)
        tokens = tokens + visual.positional_embedding.to(tokens.dtype)
        tokens = visual.ln_pre(tokens)
        tokens = tokens.permute(1, 0, 2)
        tokens = visual.transformer(tokens)
        tokens = tokens.permute(1, 0, 2)
        tokens = visual.ln_post(tokens)

        cls_out = tokens[:, 0, :]
        patch_out = tokens[:, 1:, :]
        if visual.proj is not None:
            cls_out = cls_out @ visual.proj
            patch_out = patch_out @ visual.proj

        return EncodedImage(
            cls=cls_out[0].float().cpu().numpy(),
            patch_tokens=patch_out[0].float().cpu().numpy(),
            grid=(grid_h, grid_w),
            resolution=self.resolution,
        )

    # -- text --------------------------------------------------------------

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Encode prompts to normalized (n, projected_dim) features.

        Tokenization is made explicit because `open_clip.get_tokenizer` returns
        a tensor for some backends and a list for others, and `encode_text`
        rejects a list. That mismatch cost one probe run its 448 measurement.
        """
        if not texts:
            return np.zeros((0, VERIFIED_PROJECTED_DIM), dtype=np.float32)

        tokens = self.tokenizer(texts)
        if not hasattr(tokens, "shape"):
            tokens = torch.as_tensor(tokens)
        tokens = tokens.to(self.device)

        features = self.model.encode_text(tokens)
        return features.float().cpu().numpy()

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "checkpoint": str(self.checkpoint_path),
            "resolution": self.resolution,
            "grid": (self.grid_size, self.grid_size),
            "n_patches": self.grid_size ** 2,
            "embedding_dim": VERIFIED_PROJECTED_DIM,
            "device": self.device,
            "frozen": True,
        }


def build_encoder(
    config,
    resolution: int | None = None,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> RemoteCLIPEncoder:
    """Construct the encoder from the central config.

    Args:
        resolution: defaults to the configured value. The resolution
            experiment overrides it explicitly and passes both values; nothing
            here decides which resolution is correct.
        device: torch device string. Falls back to config.device_preference.
        checkpoint_path: when given, load this local ``.pt`` instead of
            fetching from the Hub. Kaggle runs stage the 605 MB checkpoint as a
            dataset input, which removes the run's dependence on Hub
            availability and pins the exact bytes being measured. When None,
            the pinned revision from config is fetched.

    Raises:
        ModelLoadError: the local path does not exist, or the Hub fetch failed.
    """
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise ModelLoadError(
                f"checkpoint_path does not exist: {path}",
                specialist="grounding",
                context={"checkpoint_path": str(path)},
            )
        checkpoint = str(path)
    else:
        from huggingface_hub import hf_hub_download

        repo = config.get("grounding.checkpoint_repo")
        filename = config.get("grounding.checkpoint_file")
        revision = config.get("grounding.checkpoint_revision")

        try:
            checkpoint = hf_hub_download(repo, filename, revision=revision)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"could not fetch '{filename}' from '{repo}': {exc}",
                specialist="grounding",
            ) from exc

    return RemoteCLIPEncoder(
        checkpoint_path=checkpoint,
        model_name=config.get("grounding.model_name", "ViT-B-32"),
        resolution=(
            resolution
            if resolution is not None
            else int(config.get("grounding.image_size", 224))
        ),
        device=device or config.device_preference,
    )


__all__ = [
    "RemoteCLIPEncoder",
    "EncodedImage",
    "build_encoder",
    "SUPPORTED_RESOLUTIONS",
    "VERIFIED_PATCH_SIZE",
    "VERIFIED_TRANSFORMER_WIDTH",
    "VERIFIED_PROJECTED_DIM",
    "VERIFIED_PARAMETERS",
]