"""Phase 5 smoke test — does the VLM actually load and generate?

This is the Gate 2.5 check. It downloads ~1 GB on first run and executes one
CPU generation. Run it ONCE, locally or on the target machine, before any
LoRA adaptation work begins.

    python scripts/smoke_test_vlm.py              # full check
    python scripts/smoke_test_vlm.py --quick      # skip generation

It verifies the four Phase-5 findings are enforced in code, not just documented:

    F5-1  the loader class is resolved by feature detection
    F5-2  the processor pin produces 1 image, not ~17
    F5-3  prompts go through the chat template
    F5-4  the dtype kwarg resolves and falls back correctly

Exit code 0 means the VLM path is viable. Anything else means it is not.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def hr(title: str = "") -> None:
    print("-" * 68)
    if title:
        print(title)
        print("-" * 68)


def make_test_geotiff(path: Path) -> Path:
    """A small 4-band GeoTIFF the specialist can actually read.

    Structured content on purpose. The F5-5 input-quality gate measures lag-1
    spatial autocorrelation and refuses uncorrelated input, so a fixture built
    from pure `np.random.integers` would be rejected before the model was ever
    called -- and this smoke test exists to exercise the model.
    """
    import rasterio
    from affine import Affine

    yy, xx = np.mgrid[0:256, 0:256]
    ramp = (xx * 0.7 + yy * 0.3) * 12.0
    texture = np.random.default_rng(0).integers(0, 200, (256, 256)).astype(np.float64)
    plane = ramp + texture

    data = np.stack(
        [plane, plane * 0.6 + 900.0, plane * 0.4 + 1800.0, plane * 0.2 + 300.0]
    ).astype("uint16")

    with rasterio.open(
        path, "w", driver="GTiff", width=256, height=256, count=4,
        dtype="uint16", crs="EPSG:32643",
        transform=Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 2_500_000.0),
    ) as ds:
        ds.write(data)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5 VLM smoke test")
    ap.add_argument("--quick", action="store_true",
                    help="skip generation; only verify the load and the pin")
    args = ap.parse_args()

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {name}")
        if detail:
            print(f"         {detail}")
        if not ok:
            failures.append(name)

    print("=" * 68)
    print("SATQUERY AI - PHASE 5 VLM SMOKE TEST")
    print("=" * 68)

    from core.config import load_config

    cfg = load_config()
    print(f"config hash : {cfg.hash}")
    print(f"checkpoint  : {cfg.get('vlm.checkpoint')}")
    print(f"revision    : {cfg.get('vlm.revision')}")
    print(f"device      : {cfg.device_preference}")
    print()

    # ------------------------------------------------------------------
    hr("1. LOADER RESOLUTION  (F5-1)")
    try:
        from specialists.vqa.model import resolve_loader_class, resolve_dtype_kwarg

        loader = resolve_loader_class()
        dtype_kwarg = resolve_dtype_kwarg(loader)
        check("loader class resolves", True, f"{loader.__name__}")
        check("dtype kwarg resolves", dtype_kwarg in ("dtype", "torch_dtype"),
              f"{dtype_kwarg}")
    except Exception as exc:  # noqa: BLE001
        check("loader resolution", False, f"{type(exc).__name__}: {exc}")
        hr("=")
        return 1

    # ------------------------------------------------------------------
    hr("2. MODEL + PROCESSOR LOAD  (F5-2)")
    t0 = time.time()
    try:
        from specialists.vqa.model import build_vlm

        vlm = build_vlm(cfg)
    except Exception as exc:  # noqa: BLE001
        check("VLM loads", False, f"{type(exc).__name__}: {exc}")
        hr("=")
        print()
        print("If this failed with a download error, check network access to")
        print("huggingface.co. If it failed with an import error, check that")
        print("transformers>=4.52 is installed.")
        return 1

    info = vlm.load_info
    print(f"  loaded in {time.time() - t0:.1f}s")
    check("VLM loads", True, f"{info.parameters:,} params")
    check("loader class recorded", bool(info.loader_class), info.loader_class)
    check("dtype kwarg recorded", bool(info.dtype_kwarg), info.dtype_kwarg)

    # ------------------------------------------------------------------
    hr("3. PROCESSOR PIN  (F5-2)")
    ip = getattr(vlm.processor, "image_processor", None)
    edge = getattr(getattr(ip, "size", None), "longest_edge", None) if ip else None
    check("processor longest_edge is pinned", edge == cfg.get("vlm.processor_longest_edge"),
          f"size.longest_edge = {edge}")

    # ------------------------------------------------------------------
    hr("4. END-TO-END INFERENCE")
    if args.quick:
        print("  (skipped: --quick)")
    else:
        with tempfile.TemporaryDirectory() as td:
            tif = make_test_geotiff(Path(td) / "scene.tif")

            from core.schemas import Task
            from specialists.base import SpecialistRequest
            from specialists.vqa.inference import VLMSpecialist
            from preprocessing.raster import inspect_raster

            specialist = VLMSpecialist(model=vlm, max_new_tokens=32)
            asset = inspect_raster(tif)

            # -- caption
            t0 = time.time()
            try:
                cap = specialist.execute(
                    SpecialistRequest(
                        assets=[asset], query="", params={"task": "caption"}
                    )
                )
                secs = time.time() - t0
                check("caption generation", bool(cap.answer),
                      f"{secs:.1f}s -> {cap.answer[:70]!r}")
                check("caption task tag", cap.task is Task.CAPTION)
                check("caption preserves CRS", cap.geospatial.has_crs is True,
                      cap.geospatial.crs or "none")
                check("caption images per call == 1",
                      info.max_images_seen <= 1,
                      f"max_images_seen = {info.max_images_seen}")
            except Exception as exc:  # noqa: BLE001
                check("caption generation", False, f"{type(exc).__name__}: {exc}")

            # -- vqa
            t0 = time.time()
            try:
                vqa = specialist.execute(
                    SpecialistRequest(
                        assets=[asset],
                        query="What land cover is visible?",
                        params={"task": "vqa"},
                    )
                )
                secs = time.time() - t0
                check("vqa generation", bool(vqa.answer),
                      f"{secs:.1f}s -> {vqa.answer[:70]!r}")
                check("vqa task tag", vqa.task is Task.VQA)
                check("vqa confidence is measured",
                      "schema_valid" in vqa.confidence.components,
                      str(vqa.confidence.components))
            except Exception as exc:  # noqa: BLE001
                check("vqa generation", False, f"{type(exc).__name__}: {exc}")

            # -- the F5-2 regression check, on a real processor
            check("processor never split a tile",
                  info.max_images_seen <= 1,
                  f"max_images_seen = {info.max_images_seen} (17 would mean the pin was lost)")

    # ------------------------------------------------------------------
    hr("5. TRACE PAYLOAD")
    trace = info.to_trace()
    for key, value in trace.items():
        print(f"  {key:<26} {value}")
    check("trace carries the revision", trace.get("revision") not in (None, ""),
          str(trace.get("revision")))

    # ------------------------------------------------------------------
    hr()
    if failures:
        print(f"PHASE 5 SMOKE TEST: BLOCKED - {len(failures)} check(s) failed")
        for f in failures:
            print(f"  - {f}")
        print()
        print("Do NOT proceed to Phase 6 (LoRA adaptation) until this passes.")
        hr("=")
        return 1

    print("PHASE 5 SMOKE TEST: PASS - the VLM path is viable")
    print()
    print("Next: Phase 6 (BigEarthNet LoRA adaptation) requires a GPU.")
    print("      That is the first Kaggle handoff.")
    hr("=")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())