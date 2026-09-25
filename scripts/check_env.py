"""SatQuery AI — environment and contract preflight.

Reports everything the engineering team needs to decide the next step, in one run,
so a failure costs one round-trip instead of five.

A preflight MUST survive a package that installs but fails to load. Windows Application
Control, missing CUDA runtime DLLs, and broken wheels all raise OSError/RuntimeError at
import time — not ImportError. Every probe below is therefore wrapped defensively: a
package that cannot be imported is reported, never fatal.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import platform
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# (import name, distribution name, required?)
PACKAGES: list[tuple[str, str, bool]] = [
    ("numpy", "numpy", True),
    ("yaml", "pyyaml", True),
    ("pydantic", "pydantic", True),
    ("rasterio", "rasterio", True),
    ("pyproj", "pyproj", True),
    ("cv2", "opencv-python-headless", True),
    ("torch", "torch", False),
    ("torchvision", "torchvision", False),
    ("transformers", "transformers", False),
    ("open_clip", "open-clip-torch", False),
    ("sentence_transformers", "sentence-transformers", False),
    ("peft", "peft", False),
    ("gradio", "gradio", False),
    ("reportlab", "reportlab", False),
    ("huggingface_hub", "huggingface_hub", False),
]


def line(char: str = "-", n: int = 62) -> None:
    print(char * n)


def _probe(import_name: str) -> tuple[bool, str]:
    """Attempt an import. Returns (ok, detail).

    Catches everything: a broken native dependency is a finding, not a crash.
    """
    try:
        importlib.import_module(import_name)
        return True, ""
    except ImportError as exc:
        return False, f"ImportError: {exc}"
    except OSError as exc:
        # e.g. WinError 4551 Application Control blocking a .dll
        return False, f"OSError: {exc}"
    except Exception as exc:  # noqa: BLE001 - a preflight must never abort
        return False, f"{type(exc).__name__}: {exc}"


def report_python() -> None:
    line("=")
    print("SATQUERY AI - ENVIRONMENT PREFLIGHT")
    line("=")
    print(f"python           : {sys.version.split()[0]}")
    print(f"executable       : {sys.executable}")
    print(f"platform         : {platform.platform()}")
    print(f"machine          : {platform.machine()}")
    print(f"repo root        : {REPO_ROOT}")
    line()


def report_packages(verbose: bool = False) -> tuple[list[str], list[tuple[str, str]]]:
    """Returns (missing_required, broken)."""
    print("PACKAGES")
    line()
    missing_required: list[str] = []
    broken: list[tuple[str, str]] = []

    for import_name, dist_name, required in PACKAGES:
        try:
            version = md.version(dist_name)
        except md.PackageNotFoundError:
            version = "?"

        ok, detail = _probe(import_name)
        if ok:
            print(f"  [OK      ] {dist_name:<28} {version}")
            continue

        # Installed but unimportable is a distinct, more serious state.
        installed = version != "?"
        if installed:
            print(f"  [BROKEN  ] {dist_name:<28} {version}")
            broken.append((dist_name, detail))
        else:
            tag = "MISSING-REQ " if required else "missing-opt "
            print(f"  [{tag}] {dist_name}")
            if required:
                missing_required.append(dist_name)

        if verbose and detail:
            print(f"             {detail}")

    line()
    return missing_required, broken


def report_gpu() -> str:
    print("ACCELERATOR")
    line()
    ok, detail = _probe("torch")
    if not ok:
        print("  torch          : unavailable")
        if detail:
            print(f"                   {detail.splitlines()[0]}")
        print("  gpu            : none (CPU mode)")
        line()
        return "unavailable"

    import torch  # safe: probe succeeded

    print(f"  torch          : {torch.__version__}")
    print(f"  cuda available : {torch.cuda.is_available()}")
    print(f"  cuda version   : {torch.version.cuda}")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024**3)
            cap = f"{props.major}.{props.minor}"
            note = ""
            if props.major == 7 and props.minor == 5:
                note = "  <- T4: fp16 ONLY, no bf16 tensor cores (finding C-6)"
            elif props.major >= 8:
                note = "  <- Ampere+: bf16 viable"
            print(f"  gpu[{i}]        : {props.name}  {total_gb:.1f} GB  sm_{cap}{note}")
    else:
        print("  gpu            : none (CPU mode)")
    line()
    return device


def report_contracts() -> int:
    print("FROZEN CONTRACT CHECKS")
    line()
    failures = 0
    try:
        from core.config import load_config

        cfg = load_config()

        checks: list[tuple[str, bool, str]] = []

        res = cfg.get("croma.image_resolution")
        checks.append(("C-7", isinstance(res, int) and res % 8 == 0,
                       f"croma.image_resolution={res} (multiple of 8)"))

        prec = cfg.get("training.precision")
        checks.append(("C-6", prec in {"fp16", "bf16", "fp32"},
                       f"training.precision={prec}"))

        edge = cfg.get("vlm.processor_longest_edge")
        tile = cfg.get("image.tile_size")
        checks.append(("F5-2", isinstance(edge, int) and 0 < edge <= tile,
                       f"vlm.processor_longest_edge={edge} <= tile_size={tile} "
                       f"(1 image, not ~17)"))

        expect = (len(cfg.get("croma.modalities_used", [])) * cfg.get("croma.encoder_dim")
                  + cfg.get("croma.optical_channels") + cfg.get("croma.sar_channels"))
        declared = cfg.get("fusion.input_dim")
        checks.append(("C-1", expect == declared,
                       f"fusion.input_dim={declared} (expected {expect})"))

        tc = cfg.get("deployment.torch_compile")
        checks.append(("C-8", tc is False, f"deployment.torch_compile={tc} (must be false)"))

        for tag, ok, label in checks:
            print(f"  [{'OK   ' if ok else 'FAIL '}] {tag} {label}")
            failures += 0 if ok else 1

        print(f"\n  config source    : {cfg.source}")
        print(f"  config hash      : {cfg.hash}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL ] contract validation raised: {type(exc).__name__}: {exc}")
        if "--trace" in sys.argv:
            traceback.print_exc()
        failures += 1

    line()
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="SatQuery preflight")
    parser.add_argument("--check-contracts", action="store_true",
                        help="also validate the frozen config invariants")
    parser.add_argument("--verbose", action="store_true",
                        help="print full import-error detail for broken packages")
    args = parser.parse_args()

    report_python()
    missing, broken = report_packages(verbose=args.verbose)
    report_gpu()

    contract_failures = report_contracts() if args.check_contracts else 0

    print("SUMMARY")
    line()
    print(f"  missing REQUIRED packages : {', '.join(missing) if missing else 'none'}")
    print(f"  installed but BROKEN      : "
          f"{', '.join(name for name, _ in broken) if broken else 'none'}")
    if args.check_contracts:
        print(f"  frozen contract failures  : {contract_failures}")

    core_ready = not missing
    print(f"  core (schema/config/geo)  : {'READY' if core_ready else 'NOT READY'}")
    if broken:
        print("  note                      : broken packages do not block core "
              "development; they are needed for model phases")
    line("=")

    # Only hard-fail on required-package or contract problems. A broken optional
    # package (e.g. torch under an Application Control policy) is reported, not fatal.
    return 1 if (missing or contract_failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())