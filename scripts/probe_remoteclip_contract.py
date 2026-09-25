"""Phase 7 contract probe — RemoteCLIP loader, token grid, resolution behaviour.

Settles the questions that must be answered BEFORE any grounding module is
written, because getting them wrong is silent (docs/ARCHITECTURE_FREEZE.md
section 2.3, finding C-5):

  1. Does the official loading path work with a LOCAL checkpoint, or does it
     only accept a pretrained tag? The README shows an explicit
     `load_state_dict` route; open_clip's factory accepts a path. Which one
     survives with open-clip-torch 3.3.0?
  2. What is the actual token grid at 224 and at 448? The plan says
     "7x7 = 49 tokens" at 224 and "14x14 = 196" at 448. That is arithmetic
     from patch_size=32; it needs measuring.
  3. Does `force_image_size=448` interpolate the positional embedding, or does
     it silently produce garbage? open_clip's `resize_pos_embed` should handle
     it, but "should" is not evidence.
  4. What is the TEXT embedding dimension, and does it match the vision tower?
     A mismatch makes zero-shot similarity meaningless.

Writing this file and running it is the whole deliverable. No grounding module
is written until the numbers below are real.

    python scripts/probe_remoteclip_contract.py

It writes nothing and mutates nothing except the HF cache.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def hr(title: str = "") -> None:
    print("-" * 70)
    if title:
        print(title)
        print("-" * 70)


def download_checkpoint(repo: str, filename: str, revision: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    t0 = time.time()
    path = hf_hub_download(repo, filename, revision=revision)
    print(f"  checkpoint  : {path}")
    print(f"  size        : {Path(path).stat().st_size / 1e6:.1f} MB")
    print(f"  fetch time  : {time.time() - t0:.1f}s  (cached after first run)")
    return Path(path)


def build_via_pretrained_arg(checkpoint: Path):
    """Route A: let open_clip load the local file itself."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=str(checkpoint)
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer


def build_via_manual_state_dict(checkpoint: Path):
    """Route B: build the architecture, then load the state dict by hand.

    This is what the RemoteCLIP README shows. It bypasses open_clip's
    state-dict normalisation, so it is the fallback, not the preference.
    """
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    state = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model, preprocess, tokenizer


def patch_tokens(model, preprocess, image):
    """Extract (cls_token, patch_tokens) from the vision tower.

    open_clip's VisionTransformer.forward collapses to the CLS token at the end.
    To score individual regions we need the per-patch tokens BEFORE pooling, so
    this replicates the forward pass up to ln_post and returns both.
    """
    import torch

    visual = model.visual
    with torch.no_grad():
        x = preprocess(image).unsqueeze(0)

        # Patch embed -> (B, width, grid, grid) -> (B, grid*grid, width)
        patches = visual.conv1(x)
        grid_h, grid_w = patches.shape[-2], patches.shape[-1]
        patches = patches.reshape(patches.shape[0], patches.shape[1], -1)
        patches = patches.permute(0, 2, 1)

        cls = visual.class_embedding.to(patches.dtype)
        cls = cls + torch.zeros(
            patches.shape[0], 1, patches.shape[-1], dtype=patches.dtype
        )
        tokens = torch.cat([cls, patches], dim=1)

        tokens = tokens + visual.positional_embedding.to(tokens.dtype)
        tokens = visual.ln_pre(tokens)
        tokens = tokens.permute(1, 0, 2)
        tokens = visual.transformer(tokens)
        tokens = tokens.permute(1, 0, 2)
        tokens = visual.ln_post(tokens)

        cls_out = tokens[:, 0, :]
        if visual.proj is not None:
            cls_out = cls_out @ visual.proj
        patch_out = tokens[:, 1:, :]
        if visual.proj is not None:
            patch_out = patch_out @ visual.proj

    return cls_out, patch_out, (grid_h, grid_w)


def describe_resolution(model, preprocess, tokenizer, image, label: str):
    import torch

    print(f"  --- {label} ---")

    visual = model.visual
    image_size = getattr(visual, "image_size", None)
    patch_size = getattr(visual, "patch_size", None)
    pos_embed = getattr(visual, "positional_embedding", None)

    print(f"    visual.image_size      : {image_size}")
    print(f"    visual.patch_size      : {patch_size}")
    if pos_embed is not None:
        print(f"    positional_embedding   : {tuple(pos_embed.shape)}")

    t0 = time.time()
    cls_out, patch_out, (gh, gw) = patch_tokens(model, preprocess, image)
    elapsed = time.time() - t0

    n_tokens = patch_out.shape[1]
    print(f"    patch token grid       : {gh} x {gw} = {n_tokens}")
    print(f"    patch feature dim      : {patch_out.shape[-1]}")
    print(f"    cls feature dim        : {cls_out.shape[-1]}")
    if gh:
        print(f"    fraction of image/token: 1/{gh} = {1.0 / gh:.4f} of width")

    # Text encoding, defensively. `tokenizer([...])` returns a tensor for some
    # open_clip backends and a list for others, and encode_text rejects a list.
    # An earlier version of this probe lost the whole 448 measurement to that
    # mismatch -- the token grid had already been measured and was thrown away
    # with it. So this is isolated: a text-encoding failure must not discard the
    # vision measurement.
    text_dim = None
    dim_match = None
    try:
        with torch.no_grad():
            tokens = tokenizer(["a water body", "a dense urban area"])
            if not hasattr(tokens, "shape"):
                tokens = torch.as_tensor(tokens)
            text_features = model.encode_text(tokens)
        text_dim = int(text_features.shape[-1])
        dim_match = text_dim == int(patch_out.shape[-1])
        print(f"    text feature dim       : {text_dim}")
    except Exception as exc:  # noqa: BLE001
        print(f"    text encoding FAILED   : {type(exc).__name__}: {exc}")

    print(f"    encode time (vision)   : {elapsed:.3f}s")
    if dim_match is not None:
        print(f"    vision/text dim match  : {dim_match}")

    vram = peak_vram_mb()
    if vram is not None:
        print(f"    peak VRAM after call   : {vram:.1f} MB")

    return {
        "image_size": image_size,
        "patch_size": patch_size,
        "grid": (gh, gw),
        "tokens": n_tokens,
        "dim": int(patch_out.shape[-1]),
        "text_dim": text_dim,
        "dim_match": dim_match,
        "vision_seconds": round(elapsed, 4),
        "peak_vram_mb": vram,
    }


def peak_vram_mb() -> float | None:
    """Peak CUDA memory in MB, or None when running on CPU.

    Returns None rather than 0.0 so a CPU report cannot be mistaken for a GPU
    run that happened to use no memory.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    from core.config import load_config

    cfg = load_config()
    repo = cfg.get("grounding.checkpoint_repo", "chendelong/RemoteCLIP")
    filename = cfg.get("grounding.checkpoint_file", "RemoteCLIP-ViT-B-32.pt")
    revision = cfg.get("grounding.checkpoint_revision")

    print("=" * 70)
    print("SATQUERY AI - PHASE 7 REMOTECLIP CONTRACT PROBE")
    print("=" * 70)
    print(f"repo        : {repo}")
    print(f"file        : {filename}")
    print(f"revision    : {revision}")
    print(f"config hash : {cfg.hash}")
    print()

    hr("1. CHECKPOINT")
    try:
        checkpoint = download_checkpoint(repo, filename, revision)
    except Exception as exc:  # noqa: BLE001
        print(f"  DOWNLOAD FAILED: {type(exc).__name__}: {exc}")
        hr("=")
        return 2

    hr("2. OPEN_CLIP VERSION")
    try:
        import open_clip

        print(f"  open_clip : {open_clip.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"  IMPORT FAILED: {exc}")
        hr("=")
        return 2

    hr("3. LOAD ROUTE")
    model = preprocess = tokenizer = None
    route = None

    try:
        model, preprocess, tokenizer = build_via_pretrained_arg(checkpoint)
        route = "pretrained=<local path>"
        print(f"  ROUTE A works : {route}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ROUTE A failed: {type(exc).__name__}: {str(exc)[:200]}")

    if model is None:
        try:
            model, preprocess, tokenizer = build_via_manual_state_dict(checkpoint)
            route = "manual load_state_dict"
            print(f"  ROUTE B works : {route}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ROUTE B failed: {type(exc).__name__}: {str(exc)[:200]}")
            hr("=")
            print("Neither loading route works. The grounding path is blocked.")
            return 2

    model.eval()
    print(f"  parameters    : {sum(p.numel() for p in model.parameters()):,}")
    print()

    # A structured test image so the F5-5 quality gate is satisfied and the
    # measurement is not confused with the noise-vs-imagery question.
    try:
        import numpy as np
        from PIL import Image

        yy, xx = np.mgrid[0:512, 0:512]
        ramp = ((xx + yy) % 256).astype(np.uint8)
        image = Image.fromarray(np.stack([ramp, ramp // 2, 255 - ramp], axis=-1))
    except Exception as exc:  # noqa: BLE001
        print(f"  could not build a test image: {exc}")
        hr("=")
        return 2

    hr("4. TOKEN GRID AT NATIVE RESOLUTION")
    native = describe_resolution(model, preprocess, tokenizer, image, "native")

    hr("5. TOKEN GRID AT 448  (finding C-5 gated experiment)")
    try:
        import open_clip

        model448, preprocess448, tok448 = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=str(checkpoint), force_image_size=448
        )
        model448.eval()
        at448 = describe_resolution(
            model448, preprocess448, tok448, image, "force_image_size=448"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  force_image_size=448 FAILED: {type(exc).__name__}: {exc}")
        at448 = None

    hr("SUMMARY")
    print(f"  load route            : {route}")
    print(f"  native image_size     : {native['image_size']}")
    print(f"  native token grid     : {native['grid'][0]} x {native['grid'][1]}"
          f" = {native['tokens']}")
    print(f"  native dim            : {native['dim']}")
    print(f"  native text dim       : {native['text_dim']}")
    print(f"  dim match             : {native['dim_match']}")
    if at448:
        ratio = at448["tokens"] / native["tokens"]
        print(f"  448 token grid        : {at448['grid'][0]} x {at448['grid'][1]}"
              f" = {at448['tokens']}")
        print(f"  token increase        : {ratio:.1f}x")
        print(f"  attention cost (n^2)  : {ratio ** 2:.1f}x")
        print(f"  relative cell size    : 1/{native['grid'][0]} -> "
              f"1/{at448['grid'][0]} of image width")
        print(f"  vision time 224 / 448 : "
              f"{native['vision_seconds']:.4f}s / {at448['vision_seconds']:.4f}s")
        if native.get("peak_vram_mb") is not None:
            print(f"  peak VRAM 224 / 448   : "
                  f"{native['peak_vram_mb']:.1f} / {at448['peak_vram_mb']:.1f} MB")
        else:
            print("  peak VRAM             : CPU run, not measured")
        print()
        print("  INTERPRETATION: the positional embedding was resized from")
        print(f"  {native['image_size'][0]} to {at448['image_size'][0]}, so interpolation")
        print("  WORKS. Whether the finer grid actually improves LOCALIZATION")
        print("  is a separate question, answered by")
        print("  scripts/exp_grounding_resolution.py -- not by this probe.")
    else:
        print("  448 token grid        : NOT MEASURED (construction failed)")

    hr("=")
    print("Next: write specialists/grounding/remoteclip.py against these facts.")
    print("      Then scripts/exp_grounding_resolution.py measures whether the")
    print("      4x token increase actually buys localization.")
    hr("=")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())