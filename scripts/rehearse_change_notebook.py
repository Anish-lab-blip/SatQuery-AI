"""Rehearse `notebooks/kaggle_change_training.ipynb` locally, before GPU time.

Purpose: prove that the notebook's discovery -> copy -> preflight -> train ->
eval -> package chain works end to end, on a dataset tree in the REAL flat
layout, so a Kaggle T4 session is not the first thing that ever runs it.

WHAT IS SUBSTITUTED (and only this):
    Path('/kaggle/input')   -> the fixture input root
    Path('/kaggle/working') -> the fixture working root
    the pip-install cell    -> skipped (the local venv already has the deps)
    RUN_EPOCHS/RUN_BATCH/RUN_DEVICE -> small CPU values

The discovery, copy, assert, preflight, train, eval and package logic runs
BYTE-IDENTICAL. All cells exec in ONE process sharing ONE namespace, as a
Jupyter kernel does -- an earlier grounding rehearsal ran each cell as a separate
process, so cell 1's variables were gone by cell 2 and it died with NameError.
That was a harness bug that looked like a notebook bug.

THE FIXTURE USES REAL LEVIR-CD TILES when the dataset is present, sampled at
scene granularity (all 16 tiles of a scene, never a partial scene). Pass
`--synthetic` to build a small flat tree instead, which is only useful for
smoke-checking the harness itself.

    python scripts/rehearse_change_notebook.py --scenes 2 --epochs 1

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
NOTEBOOK = REPO_ROOT / "notebooks" / "kaggle_change_training.ipynb"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "levir"

#: Directories that must not be copied into the fixture "code" dataset.
#: `.pytest_*` matters: those basetemp dirs hold miniature LEVIR trees, and
#: copying them in makes the notebook's root discovery find a fake dataset
#: inside the CODE input before it ever looks at the data input. That happened,
#: and it read as "images under data root: 0".
COPY_EXCLUDE = {
    ".venv", ".git", ".git.damaged.bak", "data", "artifacts",
    ".pytest_cache", ".scratch", "notebooks", "__pycache__",
}


def _excluded(rel: pathlib.Path) -> bool:
    return any(
        part in COPY_EXCLUDE or part.startswith(".pytest") for part in rel.parts
    )

REQUIRED_IMPORTS: tuple[str, ...] = (
    "yaml", "pydantic", "rasterio", "pyproj", "cv2", "torch",
)


def hr(t: str = "") -> None:
    print("-" * 70)
    if t:
        print(t)
        print("-" * 70)


def check_interpreter() -> list[str]:
    """The notebook's later cells spawn `sys.executable`, so THIS interpreter
    must be the one with the project's dependencies."""
    import importlib

    missing: list[str] = []
    for name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    return missing


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _find_levir_source(root: pathlib.Path) -> pathlib.Path | None:
    """The extracted flat/nested LEVIR root under `root`, or None."""
    if not root.exists():
        return None
    sys.path.insert(0, str(REPO_ROOT))
    from training.change.dataset import detect_levir_layout

    for candidate in (root, root / "levir-cd-256", root / "LEVIR-CD"):
        if candidate.is_dir() and detect_levir_layout(candidate):
            return candidate
    for candidate in sorted(p for p in root.rglob("*") if p.is_dir()):
        if detect_levir_layout(candidate):
            return candidate
    return None


def build_real_data_fixture(src: pathlib.Path, dest: pathlib.Path, n_scenes: int) -> dict:
    """Copy whole scenes (all 16 tiles) from the real dataset into a flat tree."""
    from collections import defaultdict

    triples: dict[str, dict[str, pathlib.Path]] = defaultdict(dict)
    a_dir = src / "A"
    if a_dir.is_dir():  # flat
        for a in sorted(a_dir.iterdir()):
            stem = a.stem
            parts = stem.split("_")
            if len(parts) < 3:
                continue
            split, scene = parts[0], "_".join(parts[:-1])
            triples[split].setdefault(scene, {})[stem] = a
    else:  # nested
        for split in ("train", "val", "test"):
            for a in sorted((src / split / "A").glob("*.png")) if (src / split / "A").is_dir() else []:
                triples[split].setdefault(a.stem, {})[a.stem] = a

    chosen: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        scenes = sorted(triples.get(split, {}))
        # Keep only scenes with a complete tile set, so the fixture is not
        # accidentally a partial scene -- which would make tiles/scene assertions
        # fail for a reason that has nothing to do with the notebook.
        sizes = {s: len(triples[split][s]) for s in scenes}
        modal = max(set(sizes.values()), key=list(sizes.values()).count) if sizes else 0
        complete = [s for s in scenes if sizes[s] == modal]
        chosen[split] = complete[:n_scenes]

    for sub in ("A", "B", "label"):
        (dest / sub).mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for split, scenes in chosen.items():
        n = 0
        for scene in scenes:
            for stem in sorted(triples[split][scene]):
                suffix = triples[split][scene][stem].suffix
                for sub in ("A", "B", "label"):
                    sibling = (
                        src / sub / f"{stem}{suffix}"
                        if (src / sub).is_dir()
                        else src / split / sub / f"{stem}{suffix}"
                    )
                    if sibling.exists():
                        shutil.copy2(sibling, dest / sub / sibling.name)
                        n += 1
        counts[split] = n // 3
    return counts


def build_synthetic_data_fixture(dest: pathlib.Path, n_scenes: int, tiles: int = 16) -> dict:
    """A flat tree with the real naming, for smoke-checking the harness itself."""
    import numpy as np
    from PIL import Image

    for sub in ("A", "B", "label"):
        (dest / sub).mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        n = 0
        for s in range(n_scenes):
            scene = 100 + s
            for tile in range(1, tiles + 1):
                name = f"{split}_{scene}_{tile}.png"
                base = (scene * 3 + tile) % 200
                Image.fromarray(
                    np.full((256, 256, 3), base, dtype="uint8")
                ).save(dest / "A" / name)
                Image.fromarray(
                    np.full((256, 256, 3), (base + 30) % 256, dtype="uint8")
                ).save(dest / "B" / name)
                Image.fromarray(
                    np.full((256, 256), 255 if tile % 2 else 0, dtype="uint8"), mode="L"
                ).save(dest / "label" / name)
                n += 1
        counts[split] = n
    return counts


def build_fixture(root: pathlib.Path, n_scenes: int, synthetic: bool, data_root: pathlib.Path):
    inp = root / "input"
    work = root / "working"
    work.mkdir(parents=True)

    # --- code dataset: the repo tree, made read-only ----------------------
    code_dir = inp / "code-dataset" / "satquery-code"
    code_dir.mkdir(parents=True)
    copied = 0
    for path in REPO_ROOT.rglob("*"):
        rel = path.relative_to(REPO_ROOT)
        if _excluded(rel):
            continue
        target = code_dir / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file() and path.suffix in (".py", ".yaml", ".yml", ".json", ".md", ".txt", ".ipynb"):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1
    print(f"  code source : {code_dir}")
    print(f"    files     : {copied}")

    # --- data dataset ------------------------------------------------------
    data_dir = inp / "data-dataset" / "levir-cd-256"
    data_dir.mkdir(parents=True)
    if synthetic:
        counts = build_synthetic_data_fixture(data_dir, n_scenes)
        source = "SYNTHETIC flat tree"
    else:
        src = _find_levir_source(data_root)
        if src is None:
            raise SystemExit(
                f"no LEVIR-CD dataset found under {data_root}. "
                f"Pass --synthetic to rehearse the harness without real data."
            )
        counts = build_real_data_fixture(src, data_dir, n_scenes)
        source = f"REAL tiles from {src}"
    print(f"  data root   : {data_dir}")
    print(f"    source    : {source}")
    for split in ("train", "val", "test"):
        print(f"    {split:6} tiles={counts.get(split, 0)}")

    for p in code_dir.rglob("*"):
        if p.is_file():
            try:
                os.chmod(p, 0o444)
            except OSError:
                pass
    return inp, work, counts


# ---------------------------------------------------------------------------
# Notebook execution
# ---------------------------------------------------------------------------


def load_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def rewrite(src: str, inp: pathlib.Path, work: pathlib.Path, epochs: int,
            batch: int, device: str) -> str:
    out = src.replace("Path('/kaggle/input')", f"Path(r'{inp.as_posix()}')")
    out = out.replace("Path('/kaggle/working')", f"Path(r'{work.as_posix()}')")

    # The pip cell is REQUIRED on Kaggle and inert locally.
    if "'-m', 'pip', 'install'" in out:
        return (
            "print('REHEARSAL: skipped pip install "
            "(local venv already has the deps; Kaggle runs this cell verbatim)')\n"
        )

    # The RUN SCOPE block: the notebook's own comment says the rehearsal
    # rewrites exactly these three lines.
    out = out.replace("RUN_EPOCHS = 20", f"RUN_EPOCHS = {epochs}")
    out = out.replace("RUN_BATCH = 8", f"RUN_BATCH = {batch}")
    out = out.replace("RUN_DEVICE = 'cuda'", f"RUN_DEVICE = '{device}'")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=2,
                    help="scenes per split in the fixture")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("KAGGLE CHANGE-TRAINING NOTEBOOK REHEARSAL")
    print("=" * 70)
    print(f"interpreter: {sys.executable}")

    missing = check_interpreter()
    if missing:
        print()
        print("WRONG INTERPRETER -- refusing to run")
        print(f"  cannot import: {', '.join(missing)}")
        print("  Re-run as: .venv/Scripts/python.exe scripts/rehearse_change_notebook.py")
        return 2
    print("required modules: all importable")

    root = pathlib.Path(tempfile.mkdtemp(prefix="nb_change_rehearsal_"))
    failures: list[str] = []
    try:
        hr("FIXTURE")
        inp, work, counts = build_fixture(
            root, args.scenes, args.synthetic, pathlib.Path(args.data_root)
        )

        cells = load_cells()
        print(f"\n  notebook code cells: {len(cells)}")

        ns: dict = {"__name__": "__main__"}
        for i, src in enumerate(cells, start=1):
            label = src.strip().splitlines()[0][:58] if src.strip() else "(empty)"
            hr(f"CELL {i}  {label}")
            code = rewrite(src, inp, work, args.epochs, args.batch, args.device)
            try:
                exec(compile(code, f"<cell {i}>", "exec"), ns)
            except SystemExit as exc:
                failures.append(f"cell {i} raised SystemExit({exc.code})")
                print(f"  !! cell {i} exited with {exc.code}")
                break
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                failures.append(f"cell {i} raised")
                print(f"  !! cell {i} FAILED")
                break

        hr("ARTIFACTS")
        change_dir = work / "satquery-ai" / "artifacts" / "change"
        ckpt = change_dir / "levir_change_v001" / "head.pt"
        if ckpt.exists():
            print(f"  {ckpt.relative_to(work)}  {ckpt.stat().st_size/1024:,.0f} KB")
            for extra in ("model_metadata.json", "checkpoint_last.pt", "run_record.json"):
                p = ckpt.parent / extra
                print(f"    {extra:26s} {'yes' if p.exists() else 'MISSING'}")
        else:
            failures.append("no head.pt produced")
            print("  MISSING:", ckpt)

        eval_result = change_dir / "eval_test" / "eval_result.json"
        if eval_result.exists():
            payload = json.loads(eval_result.read_text(encoding="utf-8"))
            print(f"  eval_result.json: split={payload['split']} n={payload['n']} "
                  f"threshold={payload['threshold']} drift={payload['config_drift']}")
            print(f"    pooled IoU     : {payload['metrics']['pooled']['iou']:.4f}")
            print(f"    macro IoU      : {payload['metrics']['macro']['iou']:.4f}")
        else:
            failures.append("no eval_result.json produced")
            print("  MISSING:", eval_result)

        packed = work / "change_levir_v001.zip"
        if packed.exists():
            print(f"  {packed.name}  {packed.stat().st_size/1024:,.0f} KB")
        else:
            failures.append("packed artifact zip missing")

        hr("RESULT")
        if failures:
            print("REHEARSAL: FAILED")
            for f in failures:
                print("  -", f)
            return 1
        print("REHEARSAL: PASS")
        print("  every notebook cell executed, and the artifact contract held")
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print("kept:", root)


if __name__ == "__main__":
    raise SystemExit(main())
