"""Grounding specialist end-to-end verification, both decode modes.

Builds a real GeoTIFF, then runs the specialist twice:

    WITH trained head        head.pt loaded, regressed boxes, degraded=False
    WITHOUT head (fallback)  zero-shot decode, degraded=True

Reports, per mode: task, box/region/evidence counts, degraded flag, confidence
value and components, warnings, the first box and its area, its
coordinate_system, and a JSON round-trip length.

WHY THIS IS A SCRIPT AND NOT A ONE-LINER
----------------------------------------
It has to be re-runnable unchanged after every edit to the grounding path, and
its output has to be comparable between runs. The degenerate-box guard was
found by reading exactly this output: fallback mode previously reported a first
box of x1=0.000 y1=0.000 x2=1.000 y2=1.000, area 1.000.

    python scripts/verify_grounding_e2e.py
    python scripts/verify_grounding_e2e.py --head path/to/head.pt
    python scripts/verify_grounding_e2e.py --device cuda

Exit codes:
    0  both modes ran and produced schema-valid results
    1  one or both modes failed
    2  the fixture or the checkpoint could not be prepared
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

#: The checkpoint the Phase 7/8 runs used. Pinned by revision in config; this is
#: the resolved cache path, used to avoid a Hub fetch during verification.
DEFAULT_CHECKPOINT = (
    Path.home()
    / ".cache/huggingface/hub/models--chendelong--RemoteCLIP"
    / "snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-B-32.pt"
)

DEFAULT_HEAD = (
    REPO_ROOT / "artifacts" / "grounding" / "remoteclip_grounding_v001" / "head.pt"
)

PHRASE = "the water body"


def hr(title: str = "") -> None:
    print("-" * 70)
    if title:
        print(title)
        print("-" * 70)


def build_fixture(directory: Path) -> Path:
    """A georeferenced 3-band GeoTIFF with real spatial structure.

    Structured rather than noise: `preprocessing.quality` refuses uncorrelated
    input, so a noise fixture would exercise the wrong branch.
    """
    import rasterio
    from affine import Affine

    size = 512
    yy, xx = np.mgrid[0:size, 0:size]
    plane = (xx * 0.7 + yy * 0.3) * 12.0
    texture = np.random.default_rng(0).integers(0, 200, (size, size)).astype(np.float64)
    pl = plane + texture
    data = np.stack([pl, pl * 0.6 + 900.0, pl * 0.4 + 1800.0]).astype("uint16")

    path = directory / "scene.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=size, height=size, count=3,
        dtype="uint16", crs="EPSG:32643",
        transform=Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 2_500_000.0),
    ) as ds:
        ds.write(data)
    return path


def run_mode(
    label: str,
    head_path: Path | None,
    tif: Path,
    checkpoint: Path,
    device: str,
) -> tuple[bool, dict]:
    """Run one mode. Returns (ok, facts)."""
    from core.config import load_config
    from core.schemas import Task
    from specialists.base import SpecialistRequest
    from specialists.grounding.specialist import build_grounding_specialist
    from preprocessing.raster import inspect_raster

    hr(label)
    facts: dict = {"label": label}

    cfg = load_config()
    try:
        specialist = build_grounding_specialist(
            cfg,
            checkpoint_path=str(checkpoint),
            head_path=str(head_path) if head_path is not None else None,
            device=device,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  CONSTRUCTION FAILED: {type(exc).__name__}: {exc}")
        return False, {**facts, "error": f"{type(exc).__name__}: {exc}"}

    asset = inspect_raster(tif, compute_hash=False)
    facts["has_head"] = specialist.has_head
    facts["asset_crs"] = asset.geo.crs
    print(f"  has_head  : {specialist.has_head}")
    print(f"  asset     : crs={asset.geo.crs} {asset.geo.width}x{asset.geo.height}")

    request = SpecialistRequest(assets=[asset], query=PHRASE)
    specialist.validate_request(request)

    try:
        result = specialist.execute(request)
    except Exception as exc:  # noqa: BLE001
        print(f"  EXECUTE FAILED: {type(exc).__name__}: {exc}")
        return False, {**facts, "error": f"{type(exc).__name__}: {exc}"}

    payload = json.loads(result.model_dump_json())

    facts.update(
        {
            "task": result.task.value,
            "n_boxes": len(result.boxes),
            "n_regions": len(result.regions),
            "n_evidence": len(result.evidence),
            "degraded": result.degraded,
            "confidence": round(result.confidence.value, 4),
            "components": sorted(result.confidence.components),
            "warnings": list(result.warnings),
            "json_keys": len(payload),
            "json_chars": len(result.model_dump_json()),
        }
    )

    print(f"  task      : {result.task.value}")
    print(f"  boxes     : {len(result.boxes)}")
    print(f"  regions   : {len(result.regions)}")
    print(f"  evidence  : {len(result.evidence)}")
    print(f"  degraded  : {result.degraded}")
    print(f"  conf      : {result.confidence.value:.4f}  "
          f"comps={sorted(result.confidence.components)}")
    for w in result.warnings:
        print(f"  warning   : {w}")

    if result.boxes:
        b = result.boxes[0]
        area = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
        facts["first_box"] = [round(b.x1, 3), round(b.y1, 3),
                              round(b.x2, 3), round(b.y2, 3)]
        facts["first_box_area"] = round(area, 4)
        facts["first_box_crs"] = b.coordinate_system.value
        print(f"  box[0]    : x1={b.x1:.3f} y1={b.y1:.3f} x2={b.x2:.3f} "
              f"y2={b.y2:.3f} score={b.score:.3f} area={area:.4f}")
        print(f"  crs field : {b.coordinate_system.value}")
    else:
        facts["first_box"] = None
        print("  box[0]    : (none)")

    # Degenerate-box audit: only the fallback path may drop candidates.
    drops = [e for e in result.evidence if e.payload.get("dropped_degenerate")]
    facts["n_drop_records"] = len(drops)
    facts["n_dropped_candidates"] = drops[0].payload["n_dropped"] if drops else 0
    if drops:
        print(f"  dropped   : {drops[0].payload['n_dropped']} candidate(s) "
              f"over {drops[0].payload['area_threshold']:.0%} of frame")

    print(f"  json ok   : {len(payload)} top-level keys, "
          f"{len(result.model_dump_json())} chars")

    # -- assertions ------------------------------------------------------
    ok = True
    if result.task is not Task.GROUNDING:
        print(f"  ASSERT FAILED: task is {result.task.value}")
        ok = False
    if not payload.get("evidence"):
        print("  ASSERT FAILED: no evidence produced")
        ok = False

    # A surviving box must not be degenerate on either path.
    for b in result.boxes:
        a = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
        if a > 0.9:
            print(f"  ASSERT FAILED: emitted a {a:.3f}-area box")
            ok = False

    if specialist.has_head and drops:
        print("  ASSERT FAILED: the head path recorded a drop it must not perform")
        ok = False
    if not specialist.has_head and not result.degraded:
        print("  ASSERT FAILED: fallback mode did not set degraded=True")
        ok = False

    return ok, facts


def main() -> int:
    ap = argparse.ArgumentParser(description="Grounding E2E verification")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--head", default=str(DEFAULT_HEAD))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default=None, help="write the facts to this path")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")

    print("=" * 70)
    print("GROUNDING SPECIALIST -- END-TO-END VERIFICATION")
    print("=" * 70)

    checkpoint = Path(args.checkpoint)
    head = Path(args.head)
    if not checkpoint.exists():
        print(f"checkpoint not found: {checkpoint}")
        return 2
    print(f"checkpoint : {checkpoint}")
    print(f"head       : {head}  {'FOUND' if head.exists() else 'MISSING'}")
    print(f"device     : {args.device}")
    print()

    tmp = Path(tempfile.mkdtemp(prefix="grounding_e2e_"))
    try:
        tif = build_fixture(tmp)
        results: list[tuple[str, bool, dict]] = []

        ok_a, facts_a = run_mode(
            "MODE A -- WITH trained head",
            head if head.exists() else None,
            tif, checkpoint, args.device,
        )
        results.append(("with_head", ok_a, facts_a))
        print()

        ok_b, facts_b = run_mode(
            "MODE B -- WITHOUT head (zero-shot fallback)",
            None, tif, checkpoint, args.device,
        )
        results.append(("fallback", ok_b, facts_b))
        print()

        hr("SUMMARY")
        print(f"  {'mode':<12}{'boxes':>7}{'regions':>9}{'evid':>6}"
              f"{'degraded':>10}{'conf':>9}{'dropped':>9}{'ok':>5}")
        for name, ok, f in results:
            print(f"  {name:<12}{f.get('n_boxes', -1):>7}"
                  f"{f.get('n_regions', -1):>9}{f.get('n_evidence', -1):>6}"
                  f"{str(f.get('degraded')):>10}"
                  f"{f.get('confidence', float('nan')):>9.4f}"
                  f"{f.get('n_dropped_candidates', -1):>9}"
                  f"{str(ok):>5}")
        print()

        if args.json:
            Path(args.json).write_text(
                json.dumps({n: f for n, _ok, f in results},
                           indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
            print(f"  facts written to {args.json}")

        all_ok = all(ok for _n, ok, _f in results)
        hr("=")
        print("GROUNDING E2E: PASS" if all_ok else "GROUNDING E2E: FAILED")
        return 0 if all_ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())