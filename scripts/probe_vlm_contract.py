"""Phase 5 contract probe — SmolVLM loader and processor.

Answers three questions that MUST be settled before any dependent file is written
(docs/ARCHITECTURE_FREEZE.md section 2.2, findings C-2 and C-3):

  C-2  which loader class is current, and which dtype kwarg does this
       transformers version actually accept?
  C-3  what does the processor do to a 512 px tile by default, and how do we
       pin it so our tiling policy is not silently undone?

This script is deliberately split into a FAST api-surface pass and a SLOW load
pass so a contract contradiction surfaces before a 1 GB download.

    python scripts/probe_vlm_contract.py --api-only
    python scripts/probe_vlm_contract.py

It writes nothing and mutates nothing. It prints facts.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def hr(title: str = "") -> None:
    print("-" * 68)
    if title:
        print(title)
        print("-" * 68)


def probe_api_surface() -> dict[str, object]:
    """Fast pass: what does this transformers version actually expose?"""
    hr("1. TRANSFORMERS API SURFACE")

    import transformers
    from transformers import AutoProcessor

    print(f"transformers version : {transformers.__version__}")

    findings: dict[str, object] = {"transformers": transformers.__version__}

    # -- C-2: which loader classes exist? ---------------------------------
    candidates = [
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForMultimodalLM",
        "AutoModel",
    ]
    print()
    print("  loader classes:")
    for name in candidates:
        present = hasattr(transformers, name)
        findings[f"has_{name}"] = present
        marker = "present" if present else "ABSENT "
        print(f"    [{marker}] transformers.{name}")

    # -- C-2: which dtype kwarg does from_pretrained accept? --------------
    #
    # The Auto* classes accept **kwargs, so their own signatures reveal
    # nothing. The real signature lives on PreTrainedModel.from_pretrained.
    print()
    print("  from_pretrained dtype kwargs:")
    try:
        from transformers import PreTrainedModel

        base_params = list(
            inspect.signature(PreTrainedModel.from_pretrained).parameters
        )
    except Exception as exc:  # noqa: BLE001
        base_params = []
        print(f"    (could not introspect PreTrainedModel: {exc})")

    has_dtype = "dtype" in base_params
    has_torch_dtype = "torch_dtype" in base_params
    findings["base_accepts_dtype"] = has_dtype
    findings["base_accepts_torch_dtype"] = has_torch_dtype
    print("    PreTrainedModel.from_pretrained:")
    print(f"      dtype=       {'YES' if has_dtype else 'no'}")
    print(f"      torch_dtype= {'YES' if has_torch_dtype else 'no'}")
    if base_params:
        print(f"      (named params: {len(base_params)})")

    # -- C-2: is AutoModelForVision2Seq deprecated? -----------------------
    cls = getattr(transformers, "AutoModelForVision2Seq", None)
    if cls is not None:
        doc = (cls.__doc__ or "") + " " + (getattr(cls, "__init__", None).__doc__ or "")
        deprecated = "deprecat" in doc.lower()
        findings["vision2seq_deprecated_in_doc"] = deprecated
        print()
        print(f"  AutoModelForVision2Seq deprecation mentioned in docstring: {deprecated}")

    # -- C-3: processor construction surface ------------------------------
    print()
    print("  AutoProcessor.from_pretrained params (first 20):")
    try:
        sig = inspect.signature(AutoProcessor.from_pretrained)
        for i, p in enumerate(list(sig.parameters)[:20]):
            print(f"    {i:>2}. {p}")
    except (TypeError, ValueError) as exc:
        print(f"    (signature unavailable: {exc})")

    hr()
    return findings


def probe_processor_contract(repo: str, revision: str | None) -> dict[str, object]:
    """Fast pass: what does the processor do to a 512 px tile?"""
    hr("2. PROCESSOR / RESOLUTION CONTRACT  (finding C-3)")

    from transformers import AutoProcessor

    findings: dict[str, object] = {}

    kwargs: dict[str, object] = {}
    if revision:
        kwargs["revision"] = revision

    t0 = time.time()
    print(f"  loading processor: {repo}")
    try:
        proc = AutoProcessor.from_pretrained(repo, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"  PROCESSOR LOAD FAILED: {type(exc).__name__}: {exc}")
        return {"processor_load_failed": str(exc)}
    print(f"  load time: {time.time() - t0:.1f}s")

    ip = getattr(proc, "image_processor", None)
    if ip is not None:
        print()
        print("  image_processor attributes of interest:")
        for attr in ("size", "longest_edge", "shortest_edge", "do_resize",
                     "do_image_splitting", "max_image_size", "image_mean",
                     "image_std", "resample"):
            if hasattr(ip, attr):
                value = getattr(ip, attr)
                findings[f"ip_{attr}"] = value
                print(f"    {attr:<22} = {value}")

    # -- what does a 512x512 tile become? ---------------------------------
    try:
        import numpy as np
        from PIL import Image

        tile = Image.fromarray(
            (np.random.default_rng(0).random((512, 512, 3)) * 255).astype("uint8")
        )

        print()
        print("  encoding a 512x512 RGB tile:")
        # SmolVLM requires the prompt to carry one <image> token per image.
        # Raw text raises: "The total number of <image> tokens in the prompts
        # should be the same as the number of images passed." The chat
        # template inserts them; hand-written prompts do not. (finding F5-4)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            }
        ]
        prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
        out = proc(images=tile, text=prompt, return_tensors="pt")
        findings["prompt_char_len"] = len(prompt)
        print(f"    prompt length          {len(prompt)} chars")
        for key, value in out.items():
            shape = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
            print(f"    {key:<22} {shape}")
            findings[f"encoded_{key}"] = str(shape)

        # How many image tokens did we get? (the real cost driver)
        if "pixel_values" in out:
            pv = out["pixel_values"]
            print(f"    pixel_values dtype     {pv.dtype}")
            print(f"    pixel_values shape     {tuple(pv.shape)}")

    except Exception as exc:  # noqa: BLE001
        print(f"  encoding probe failed: {type(exc).__name__}: {exc}")
        findings["encoding_failed"] = str(exc)

    # -- can we pin longest_edge explicitly? ------------------------------
    print()
    print("  attempting explicit size={'longest_edge': 512} ...")
    try:
        proc2 = AutoProcessor.from_pretrained(
            repo, size={"longest_edge": 512}, **kwargs
        )
        ip2 = getattr(proc2, "image_processor", None)
        pinned = getattr(ip2, "size", None) if ip2 else None
        findings["pinned_size"] = pinned
        print(f"    result: size = {pinned}")

        import numpy as np
        from PIL import Image

        tile = Image.fromarray(
            (np.random.default_rng(0).random((512, 512, 3)) * 255).astype("uint8")
        )
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "x"},
                ],
            }
        ]
        p = proc2.apply_chat_template(msgs, add_generation_prompt=True)
        out2 = proc2(images=tile, text=p, return_tensors="pt")
        pv2 = out2.get("pixel_values")
        if pv2 is not None:
            findings["pinned_pixel_values_shape"] = str(tuple(pv2.shape))
            print(f"    pixel_values shape = {tuple(pv2.shape)}")
    except Exception as exc:  # noqa: BLE001
        print(f"    explicit-size construction FAILED: {type(exc).__name__}: {exc}")
        findings["pinned_size_failed"] = str(exc)

    hr()
    return findings


def probe_model_load(repo: str, revision: str | None) -> dict[str, object]:
    """Slow pass: actually load the model and run one inference."""
    hr("3. MODEL LOAD + INFERENCE SMOKE TEST")

    import torch
    import transformers
    from transformers import AutoProcessor

    findings: dict[str, object] = {}

    # Resolve loader class by feature detection (finding C-2).
    loader_name = None
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        if hasattr(transformers, name):
            loader_name = name
            break
    if loader_name is None:
        print("  NO USABLE LOADER CLASS FOUND")
        return {"no_loader": True}

    loader = getattr(transformers, loader_name)
    print(f"  loader class : transformers.{loader_name}")

    # Resolve dtype kwarg by signature detection (finding C-2).
    #
    # Auto* classes take **kwargs, so introspection must target
    # PreTrainedModel. In transformers v5 the kwarg is `dtype`; v4 used
    # `torch_dtype`. Resolve it, then fall back at call time if the guess
    # was wrong.
    dtype_kwarg: str | None = None
    try:
        from transformers import PreTrainedModel

        base_params = list(
            inspect.signature(PreTrainedModel.from_pretrained).parameters
        )
        if "dtype" in base_params:
            dtype_kwarg = "dtype"
        elif "torch_dtype" in base_params:
            dtype_kwarg = "torch_dtype"
    except Exception:  # noqa: BLE001
        pass
    if dtype_kwarg is None:
        dtype_kwarg = "dtype"  # transformers v5 default
    print(f"  dtype kwarg  : {dtype_kwarg}")
    findings["loader"] = loader_name
    findings["dtype_kwarg"] = dtype_kwarg

    kwargs: dict[str, object] = {}
    if revision:
        kwargs["revision"] = revision
    kwargs[dtype_kwarg] = torch.float32  # CPU-safe

    t0 = time.time()
    print(f"  loading model: {repo} (this may download ~1 GB) ...")
    try:
        model = loader.from_pretrained(repo, **kwargs)
    except TypeError as exc:
        # The dtype kwarg guess was wrong; try the other spelling once.
        other = "torch_dtype" if dtype_kwarg == "dtype" else "dtype"
        print(f"  {dtype_kwarg}= rejected ({exc}); retrying with {other}=")
        kwargs.pop(dtype_kwarg, None)
        kwargs[other] = torch.float32
        findings["dtype_kwarg"] = other
        try:
            model = loader.from_pretrained(repo, **kwargs)
        except Exception as exc2:  # noqa: BLE001
            print(f"  MODEL LOAD FAILED: {type(exc2).__name__}: {exc2}")
            return {**findings, "model_load_failed": str(exc2)}
    except Exception as exc:  # noqa: BLE001
        print(f"  MODEL LOAD FAILED: {type(exc).__name__}: {exc}")
        return {**findings, "model_load_failed": str(exc)}
    load_s = time.time() - t0
    model.eval()
    findings["load_seconds"] = round(load_s, 1)
    print(f"  load time    : {load_s:.1f}s")

    n_params = sum(p.numel() for p in model.parameters())
    findings["parameters"] = n_params
    print(f"  parameters   : {n_params:,}")

    proc_kwargs: dict[str, object] = {"size": {"longest_edge": 512}}
    if revision:
        proc_kwargs["revision"] = revision
    try:
        processor = AutoProcessor.from_pretrained(repo, **proc_kwargs)
    except Exception:  # noqa: BLE001
        processor = AutoProcessor.from_pretrained(repo, **kwargs)

    import numpy as np
    from PIL import Image

    tile = Image.fromarray(
        (np.random.default_rng(0).random((512, 512, 3)) * 255).astype("uint8")
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "Describe this image in one sentence."}],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=tile, text=prompt, return_tensors="pt")
    findings["inference_input_shapes"] = {
        k: str(tuple(v.shape)) for k, v in inputs.items() if hasattr(v, "shape")
    }

    t0 = time.time()
    try:
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"  GENERATE FAILED: {type(exc).__name__}: {exc}")
        return {**findings, "generate_failed": str(exc)}
    gen_s = time.time() - t0
    findings["generate_seconds"] = round(gen_s, 1)
    print(f"  generate time: {gen_s:.1f}s  (CPU)")

    try:
        text = processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()
    except Exception as exc:  # noqa: BLE001
        text = f"<decode failed: {exc}>"
    findings["sample_output"] = text
    print(f"  output       : {text!r}")

    hr()
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5 SmolVLM contract probe")
    ap.add_argument("--api-only", action="store_true",
                    help="fast pass only; no model download")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    try:
        from core.config import load_config

        cfg = load_config()
        repo = args.repo or cfg.get("vlm.checkpoint")
        revision = args.revision or cfg.get("vlm.revision")
    except Exception:  # noqa: BLE001
        repo = args.repo or "HuggingFaceTB/SmolVLM-500M-Instruct"
        revision = args.revision

    print("=" * 68)
    print("SATQUERY AI - PHASE 5 VLM CONTRACT PROBE")
    print("=" * 68)
    print(f"repo     : {repo}")
    print(f"revision : {revision}")
    print()

    findings = probe_api_surface()
    findings.update(probe_processor_contract(repo, revision))

    if not args.api_only:
        findings.update(probe_model_load(repo, revision))

    hr("SUMMARY")
    for key in sorted(findings):
        print(f"  {key:<38} {findings[key]}")
    hr("=")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())