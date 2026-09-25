"""Rehearse the Kaggle notebook's cells against a local fixture.

Purpose: prove that v2's discovery + copy + argument construction works on the
EXTRACTED layout the Kaggle web UI produces, before the user spends GPU time.

WHAT IS SUBSTITUTED (and only this):
    Path('/kaggle/input')   -> the fixture input root
    Path('/kaggle/working') -> the fixture working root
and, for the CPU rehearsal only:
    '--all'           -> '--limit', '<N>'
    '--device','cuda' -> '--device', 'cpu'
    '--tag','full'    -> '--tag','rehearsal'

The discovery, copy, assert, and package logic runs BYTE-IDENTICAL.

HOW CELLS RUN: all cells are exec'd in ONE process sharing ONE namespace, which
is what a Jupyter kernel does. An earlier version spawned a separate `python -c`
per cell; cell 1's `INPUT_ROOT` was gone by cell 2 and the run died with
`NameError`. That was a harness bug, not a notebook bug -- but it is exactly the
kind of false failure that would send someone debugging the wrong file.

Fixture layout mirrors the reported Kaggle reality:
    <input>/code-dataset/satquery-code/      extracted repo tree (read-only)
    <input>/data-dataset/
        VRSBench_EVAL_referring.json         sits beside Images_val/
        RemoteCLIP-ViT-B-32.pt               hardlinked from the HF cache
        Images_val/Images_val/*.png          double-nested, as uploaded

    python scripts/rehearse_kaggle_notebook.py --images 24 --limit 6

Exit 0 = every cell ran and the artifact contract held.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys
import tempfile
import traceback
import zipfile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "notebooks" / "kaggle_grounding_resolution.ipynb"
CODE_ZIP = pathlib.Path(os.environ["LOCALAPPDATA"]) / "Temp" / "satquery-code.zip"
VRS = REPO_ROOT / "training" / "data" / "vrsbench"
HF_CKPT = (
    pathlib.Path.home()
    / ".cache/huggingface/hub/models--chendelong--RemoteCLIP"
    / "snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-B-32.pt"
)


#: Modules the notebook's later cells need. Cell 5 runs check_env.py and cell 6
#: runs the experiment, both as `sys.executable` subprocesses -- so whatever
#: interpreter runs THIS script is the one those cells get.
REQUIRED_IMPORTS: tuple[str, ...] = (
    "yaml", "pydantic", "rasterio", "pyproj", "cv2", "torch", "open_clip",
)


def hr(t: str = "") -> None:
    print("-" * 70)
    if t:
        print(t)
        print("-" * 70)


def check_interpreter() -> list[str]:
    """Report which required modules THIS interpreter can import.

    WHY THIS EXISTS: the rehearsal execs the notebook's cells in this process,
    and cells 5 and 6 spawn `sys.executable` as a subprocess. If this script was
    launched with an interpreter that lacks the project's dependencies, cell 5
    fails with `frozen contract check failed` -- which reads like a config
    problem when the real cause is the wrong python. Measured once:

        missing REQUIRED packages : rasterio, pyproj, opencv-python-headless
        frozen contract failures  : 0          <- config was fine all along
        core (schema/config/geo)  : NOT READY

    On Kaggle this never happens: cell 4 installs into the kernel interpreter,
    and the kernel interpreter is what cells 5 and 6 then spawn. Locally it
    means running this script with the venv python.
    """
    import importlib

    missing: list[str] = []
    for name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    return missing


def build_fixture(root: pathlib.Path, n_images: int) -> tuple[pathlib.Path, pathlib.Path]:
    inp = root / "input"
    work = root / "working"
    work.mkdir(parents=True)

    # --- code dataset: extracted tree, made read-only ---------------------
    code_dir = inp / "code-dataset" / "satquery-code"
    code_dir.mkdir(parents=True)
    with zipfile.ZipFile(CODE_ZIP) as z:
        z.extractall(code_dir)

    # --- data dataset ------------------------------------------------------
    data_dir = inp / "data-dataset"
    data_dir.mkdir(parents=True)
    shutil.copy2(
        VRS / "VRSBench_EVAL_referring.json",
        data_dir / "VRSBench_EVAL_referring.json",
    )

    if not HF_CKPT.exists():
        raise SystemExit(f"checkpoint not cached: {HF_CKPT}")
    try:
        os.link(HF_CKPT, data_dir / "RemoteCLIP-ViT-B-32.pt")
    except OSError:
        shutil.copy2(HF_CKPT, data_dir / "RemoteCLIP-ViT-B-32.pt")

    inner = data_dir / "Images_val" / "Images_val"
    inner.mkdir(parents=True)
    with zipfile.ZipFile(VRS / "Images_val.zip") as z:
        entries = [n for n in z.namelist() if n.lower().endswith(".png")][:n_images]
        for name in entries:
            z.extract(name, data_dir)

    for p in code_dir.rglob("*"):
        if p.is_file():
            try:
                os.chmod(p, 0o444)
            except OSError:
                pass

    print(f"  code source : {code_dir}")
    print(f"    files     : {sum(1 for p in code_dir.rglob('*') if p.is_file())} (chmod 444)")
    print(f"  data root   : {data_dir}")
    print(f"    json      : {(data_dir / 'VRSBench_EVAL_referring.json').stat().st_size/1e6:.1f} MB")
    print(f"    ckpt      : {(data_dir / 'RemoteCLIP-ViT-B-32.pt').stat().st_size/1e6:.1f} MB")
    print(f"    images    : {n_images} at Images_val/Images_val/")
    return inp, work


def load_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def rewrite(
    src: str,
    inp: pathlib.Path,
    work: pathlib.Path,
    limit: int,
    min_images: int,
) -> str:
    """Substitute ONLY the roots, the run-scope args, and the fixture image floor.

    The notebook's own guard is `assert n_img > 1000` -- a real protection
    against launching a ~78-minute run against an empty or half-attached data
    tree. The rehearsal fixture extracts only a handful of images, so the floor
    is lowered for the rehearsal ONLY. The notebook keeps 1000.
    """
    out = src.replace("Path('/kaggle/input')", f"Path(r'{inp.as_posix()}')")
    out = out.replace("Path('/kaggle/working')", f"Path(r'{work.as_posix()}')")

    # Cell 4 runs `pip install`. On Kaggle that is REQUIRED and works. Locally
    # the venv already has every package, and the local runtime python has no
    # pip module at all -- so the cell is skipped here. This is the one cell
    # whose behaviour genuinely differs between the two environments; it is
    # inert (installs packages) and verifies nothing about the pipeline.
    if "'-m', 'pip', 'install'" in out:
        out = (
            "print('REHEARSAL: skipped pip install "
            "(local venv already has the deps; Kaggle runs this cell verbatim)')\n"
        )
        return out

    if "n_img > 1000" in out:
        out = out.replace("n_img > 1000", f"n_img > {min_images}")
        out = out.replace("expected ~9,350", f"expected >{min_images} (rehearsal fixture)")
    if "exp_grounding_resolution.py" in out:
        out = out.replace("'--all',", f"'--limit', '{limit}',")
        out = out.replace("'--device', 'cuda',", "'--device', 'cpu',")
        out = out.replace("'--tag', 'full',", "'--tag', 'rehearsal',")
    if "analyze_grounding_resolution.py" in out:
        out = out.replace("'--tag', '_full'", "'--tag', '_rehearsal'")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=24)
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("KAGGLE NOTEBOOK v2 REHEARSAL (extracted-input layout)")
    print("=" * 70)
    print(f"interpreter: {sys.executable}")

    missing_modules = check_interpreter()
    if missing_modules:
        print()
        print("WRONG INTERPRETER -- refusing to run")
        print(f"  cannot import: {', '.join(missing_modules)}")
        print()
        print("  The notebook's cells 5 and 6 spawn sys.executable, so this")
        print("  script must run under the interpreter that has the project's")
        print("  dependencies. Re-run as:")
        print()
        print("      .venv/Scripts/python.exe scripts/rehearse_kaggle_notebook.py")
        return 2
    print("required modules: all importable")

    root = pathlib.Path(tempfile.mkdtemp(prefix="nb2_rehearsal_"))
    failures: list[str] = []
    try:
        hr("FIXTURE")
        inp, work = build_fixture(root, args.images)

        cells = load_cells()
        print(f"\n  notebook code cells: {len(cells)}")

        # ONE namespace, ONE process -- a Jupyter kernel does not isolate cells.
        ns: dict = {"__name__": "__main__"}
        for i, src in enumerate(cells, start=1):
            label = src.strip().splitlines()[0][:58] if src.strip() else "(empty)"
            hr(f"CELL {i}  {label}")
            code = rewrite(src, inp, work, args.limit, min_images=args.images // 2)
            try:
                exec(compile(code, f"<cell {i}>", "exec"), ns)
            except SystemExit as exc:  # a cell called exit()
                failures.append(f"cell {i} raised SystemExit({exc.code})")
                print(f"  !! cell {i} exited with {exc.code}")
                break
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                failures.append(f"cell {i} raised")
                print(f"  !! cell {i} FAILED")
                break

        hr("ARTIFACTS")
        art = work / "satquery-ai" / "artifacts" / "grounding"
        if art.exists():
            for p in sorted(art.iterdir()):
                print(f"  {p.name:40s} {p.stat().st_size/1024:10.1f} KB")
            per_sample = sorted(art.glob("per_sample_*_rehearsal.jsonl"))
            if len(per_sample) == 2:
                for p in per_sample:
                    n = len(p.read_text(encoding="utf-8").splitlines())
                    print(f"  {p.name:40s} {n} lines")
            else:
                failures.append(
                    f"expected 2 per_sample rehearsal JSONLs, found {len(per_sample)}"
                )
        else:
            failures.append("artifacts dir missing")
            print("  MISSING:", art)

        packed = work / "grounding_resolution_full.zip"
        if packed.exists():
            print(f"  {packed.name:40s} {packed.stat().st_size/1024:10.1f} KB")
        else:
            failures.append("packed artifact zip missing")

        hr("RESULT")
        if failures:
            print("REHEARSAL: FAILED")
            for f in failures:
                print("  -", f)
            return 1
        print("REHEARSAL: PASS")
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print("kept:", root)


if __name__ == "__main__":
    raise SystemExit(main())