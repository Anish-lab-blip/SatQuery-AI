"""Package the repo as a Kaggle code dataset.

Kaggle notebooks cannot read the local filesystem, so the code tree has to be
uploaded as a dataset. This builds the zip to upload.

    python scripts/package_kaggle_code.py
    python scripts/package_kaggle_code.py --out D:/satquery-code.zip

WHAT IS INCLUDED: the Python packages, configs, docs and scripts the notebook
needs to run `train_change.py` and `eval_change.py`.

WHAT IS EXCLUDED, and why it matters: `.venv` (gigabytes), `.git` and
`.git.damaged.bak`, the repository-**root** `data` (the LEVIR-CD tree -- that is
a SEPARATE dataset), `artifacts` (large trained checkpoints), and every
`.pytest*` basetemp directory. Those four names match only the first path
component: `training/data/` is Python source and ships (see D4 below).

The `.pytest*` exclusion is not cosmetic. Those directories contain miniature
LEVIR-CD trees, and the notebook discovers its data root by searching
`/kaggle/input` for a directory with an A/B/label layout. Shipping fixture trees
inside the code dataset makes the notebook find a fake dataset first -- which is
exactly what happened during the first rehearsal, and it reported
`images under data root: 0`.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.code_revision import code_snapshot_digest  # noqa: E402  (path set above)

#: Directory names excluded at **any depth**. These are never Python source
#: anywhere in the tree, so a recursive name match is both safe and desirable:
#: `.venv312` and `.venv312b` sit beside the real `.venv` and each is gigabytes.
#: (They are dot-directories and so are already caught by the dot rule below;
#: they are listed anyway so the intent is readable.)
EXCLUDE_DIRS = {
    ".venv", ".venv312", ".venv312b", ".git", ".git.damaged.bak", "__pycache__",
    ".pytest_cache", ".scratch", ".mypy_cache", ".ruff_cache",
}

#: Directory names excluded **only as the first path component** (directly under
#: the repository root).
#:
#: D4 (measured 2026-09-23). These four names used to live in `EXCLUDE_DIRS`,
#: whose components were matched at EVERY depth. So `training/data/bigearthnet.py`
#: had `parts[:-1] == ("training", "data")`, matched `"data"`, and was dropped --
#: and `iter_source_files` pruned the directory in place, so the walk never even
#: descended. The zip shipped 347 files with NO `training/data/` entry
#: (`unzip -l <archive> | grep -c 'training/data/'` -> 0) and the Kaggle notebook
#: died at cell 8 with `ModuleNotFoundError: No module named 'training.data'`.
#: The `required` list below printed "all present" and exited 0 throughout,
#: because every one of its entries lived under `training/vlm/` -- the third time
#: a hand-maintained sample failed to cover what the docs said it covered.
#:
#: The intent was always the repository-ROOT `data/` (the LEVIR-CD tree, shipped
#: as a SEPARATE Kaggle dataset), so the fix scopes these names to `parts[0]`
#: rather than deleting `"data"` outright. Deleting it would have let the whole
#: LEVIR-CD tree -- and the VRSBench payload, see below -- ride along.
#:
#: `frontend_pending_cleanup` (measured 2026-09-24). A fully-untracked local
#: staging tree (`git ls-files frontend_pending_cleanup` -> 0 entries) holding a
#: copy of the frontend prototype, its `.npm-cache-local` / `.tools` /
#: `.workbuddy-ai` droppings, and an 840 MB `_copy.tar`. It is not part of the
#: upload set, but no rule above matched it, so it shipped to Kaggle: **429 of
#: the archive's 792 entries** (162 `.md`, 260 `.json`, 6 `.txt`, 1 `.py`) came
#: from it. Excluded as a repository-root tree, beside `data`/`artifacts`/`logs`/
#: `reports` -- deliberately NOT via a broad untracked-path purge, because the
#: git index is partial by design and many top-level dirs that are ALSO untracked
#: (`app/`, `core/`, `gateway/`, `geospatial/`, `notebooks/`, `preprocessing/`)
#: are real source that must ship. The directory is left on disk; deleting it is
#: an owner decision, not this script's.
ROOT_ONLY_DIRS = {"data", "artifacts", "logs", "reports", "frontend_pending_cleanup"}

#: Data payloads excluded by **path prefix** (D4b, measured 2026-09-23). Scoping
#: `data` to the repository root made `training/data/` visible -- and with it
#: `training/data/vrsbench/VRSBench_train.json` (64,923,606 B) and
#: `VRSBench_EVAL_referring.json` (10,277,680 B). `.json` IS an include suffix,
#: so those two files alone would have grown the archive from 2.1 MB to ~77 MB.
#: (The 29,614 `.png` files and 2 `.zip` files there are filtered by suffix and
#: would not have ridden along; the JSONs are the trap.)
#:
#: A prefix rule, not a bare directory name: `vrsbench` is also a *filename*, and
#: the module `training/data/vrsbench.py` must still ship.
EXCLUDE_DATA_PAYLOADS = {"training/data/vrsbench"}

EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pt", ".zip", ".log", ".sha256"}
INCLUDE_SUFFIXES = {
    ".py", ".yaml", ".yml", ".json", ".md", ".txt", ".ipynb", ".cfg", ".toml", ".ini",
}

#: Suffixes of THIS script's own outputs. `.zip` is already excluded above, but
#: the sidecars are not: `<archive>.zip.provenance.json` ends in `.json`, which
#: `INCLUDE_SUFFIXES` accepts, and it is not dot-prefixed. So a build run with
#: `--out` pointing inside the repo swept the PREVIOUS build's provenance file
#: into the archive -- shipping a record that names a different archive's hash
#: (`00029223d7085c56...`) and making the build non-reproducible, since identical
#: source then produced different archives depending on whether a sidecar was
#: already lying around. Measured 2026-09-23: 346 files before, 347 after.
PACKAGING_OUTPUT_SUFFIXES = (".zip.sha256", ".zip.provenance.json")

#: Path-component prefixes RUNBOOK_CHANGE_VQA_KAGGLE.md §1 excludes. These are
#: test and run droppings whose names carry a random or timestamped suffix
#: (`router_a53fnqz_`, `chg_20260921T...`), so an exact-name match cannot catch
#: them.
EXCLUDE_PREFIXES = ("router_", "chg_")


def _is_excluded_data_payload(posix: str) -> bool:
    """True for a path under an `EXCLUDE_DATA_PAYLOADS` prefix.

    Prefix rather than bare name on purpose: `vrsbench` is also a *filename*
    (`training/data/vrsbench.py`) and that module must ship, while the
    `training/data/vrsbench/` payload directory must not.
    """
    return any(posix == p or posix.startswith(p + "/") for p in EXCLUDE_DATA_PAYLOADS)


def excluded(rel: Path) -> bool:
    """True when `rel` is not part of the code that actually runs on Kaggle.

    Directory components are matched against `EXCLUDE_DIRS` **at any depth**, and
    any dot-prefixed directory is skipped outright. That second rule is
    load-bearing rather than cosmetic: this tree carries `.venv312` and
    `.venv312b` beside the real `.venv`, so matching only the literal name
    `.venv` let the archive swallow two entire virtualenvs -- 7,114 entries /
    29.4 MB of site-packages instead of the ~700 files of actual source. It
    mirrors the walk in `core.code_revision.code_snapshot_files`, which skips
    dot-directories for the same reason.

    `ROOT_ONLY_DIRS` is different: `data`, `artifacts`, `logs` and `reports`
    match **only as the first path component**. They name repository-root trees
    that are not part of the upload set, and matching them by name at every depth
    is D4 -- it silently hid `training/data/` and broke `import training.data` on
    Kaggle. `EXCLUDE_DATA_PAYLOADS` then removes the VRSBench JSON payload that
    becomes visible once `data` is root-scoped (D4b).

    A dot-prefixed *file* is skipped only at the repository root, so scratch
    droppings such as `.full.txt` or `.scratch_bad.py` cannot ride along while a
    legitimately dotted filename deeper in the tree still can.
    """
    parts = rel.parts
    if len(parts) > 1 and parts[0] in ROOT_ONLY_DIRS:
        return True
    if _is_excluded_data_payload(rel.as_posix()):
        return True
    for part in parts[:-1]:  # directories only
        if part in EXCLUDE_DIRS or part.startswith("."):
            return True
        if part.startswith(EXCLUDE_PREFIXES):
            return True
    name = parts[-1]
    if len(parts) == 1 and (name.startswith(".") or name.startswith(EXCLUDE_PREFIXES)):
        return True
    if name.endswith(PACKAGING_OUTPUT_SUFFIXES):
        return True
    return False


def iter_source_files(root: Path) -> Iterator[Path]:
    """Yield repo-relative files that could belong in the archive.

    Walks with `os.walk` and prunes excluded directories **in place**, rather
    than filtering the output of a full `rglob`. That is not a micro-
    optimisation: this tree carries `.venv`, `.venv312` and `.venv312b`, so
    filter-after-walk materialises **1,088,312 paths** and spends **103.5 s**
    discarding 1,088,000 of them (measured 2026-09-23). Pruning visits ~1,100
    paths and finishes in well under a second, and the archive is byte-identical
    because the surviving files are the same set in the same order.

    Pruning also makes the exclusion structural rather than a post-hoc check:
    the walk cannot descend into an excluded directory, so a rule cannot be
    bypassed by a path shape `excluded()` did not anticipate. `excluded()` is
    still applied to every candidate that comes back, as defence in depth --
    these are the two halves of the same rule and they are meant to agree.

    They agree by construction here: the prune predicate below mirrors
    `excluded()` exactly. `EXCLUDE_DIRS` matches by name at any depth, but
    `ROOT_ONLY_DIRS` is pruned **only at the top level** (so a nested `data/`
    directory is walked into, not skipped) and `EXCLUDE_DATA_PAYLOADS` is pruned
    by the child's *relative path*, not its bare name.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        at_root = rel_dir == Path(".")
        kept = []
        for d in sorted(dirnames):
            if d.startswith(".") or d.startswith(EXCLUDE_PREFIXES) or d in EXCLUDE_DIRS:
                continue
            if at_root and d in ROOT_ONLY_DIRS:
                continue
            if _is_excluded_data_payload((rel_dir / d).as_posix()):
                continue
            kept.append(d)
        dirnames[:] = kept
        for name in sorted(filenames):
            yield (Path(dirpath) / name).relative_to(root)


def _module_candidates(path_no_ext: str) -> tuple[str, str]:
    """The two archive paths a dotted module may live at."""
    return f"{path_no_ext}.py", f"{path_no_ext}/__init__.py"


def _relative_base(entry: str, level: int) -> str | None:
    """Archive-relative directory a `level`-deep relative import resolves against.

    `entry` is the importing file's archive path (e.g.
    `training/vlm/dataset.py`). Level 1 means "this file's own package", level 2
    the parent, and so on. Returns None when the import reaches above the
    archive root (which no shipped module legitimately does).
    """
    parts = entry.split("/")[:-1]  # the importing file's package directory
    drop = level - 1
    if drop > len(parts):
        return None
    base = parts[: len(parts) - drop]
    return "/".join(base) if base else "."


def _missing_import_targets(
    node: ast.AST, entry: str, tops: set[str], names: set[str], repo_root: Path
) -> list[str]:
    """Archive paths an intra-repo import needs but which are absent from `names`.

    Only intra-repo packages count: `tops` is derived from the archive itself, so
    stdlib and third-party names (`numpy`, `rasterio`, `PIL`, `pandas`) are not in
    it and are ignored -- including the `import rasterio` / `from PIL import Image`
    calls that sit inside function bodies of the `training/data/` modules.

    Relative imports are resolved against the importer's package directory rather
    than being treated as a package name; a relative target is only reported when
    it genuinely exists in the source tree, so an attribute import such as
    `from . import CONSTANT` cannot raise a false alarm.
    """
    out: list[str] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.split(".")[0] not in tops:
                continue
            mod, pkg = _module_candidates(alias.name.replace(".", "/"))
            if mod not in names and pkg not in names:
                out.append(f"{mod} (or {pkg})")
    elif isinstance(node, ast.ImportFrom):
        if node.level == 0:
            if not node.module or node.module.split(".")[0] not in tops:
                return out
            mod, pkg = _module_candidates(node.module.replace(".", "/"))
            if mod not in names and pkg not in names:
                out.append(f"{mod} (or {pkg})")
            return out
        base = _relative_base(entry, node.level)
        if base is None:
            return out
        if node.module:
            target = f"{base}/{node.module.replace('.', '/')}"
            mod, pkg = _module_candidates(target)
            if mod not in names and pkg not in names:
                if (repo_root / mod).exists() or (repo_root / pkg).exists():
                    out.append(f"{mod} (or {pkg})")
            return out
        for alias in node.names:
            target = f"{base}/{alias.name}"
            mod, pkg = _module_candidates(target)
            if mod in names or pkg in names:
                continue
            if (repo_root / mod).exists() or (repo_root / pkg).exists():
                out.append(f"{mod} (or {pkg})")
    return out


def assert_import_closure(out: Path, repo_root: Path) -> list[str]:
    """Verify every intra-repo import in the archive resolves inside it.

    This is the durable fix for D4/D4c. Adding six names to `required` repairs
    *this* instance of the missing-module bug but leaves the class of bug alive,
    because `required` is a hand-maintained sample and was wrong three times. The
    assertion below is instead derived from the code: it parses the `.py` files
    the archive actually ships, extracts their intra-repo imports with `ast`, and
    checks that each resolves to a module or package path present in
    `z.namelist()`. Anything missing is a packaging bug and is reported here, on
    the build machine, rather than on Kaggle.

    Returns a sorted list of human-readable missing targets (empty when the
    closure is complete).

    Limitation, recorded so nobody over-trusts this: it only reads imports
    written in files the archive actually ships, so an archive that omits an
    entire package -- with no shipped file left to import it -- returns `[]`
    trivially and looks complete. A wholly-absent package is invisible here; the
    hand-maintained `required` list above is the backstop for exactly that case,
    which is why both are kept. An assertion that does not cover what it claims
    is worse than none, and this file has now learned that three times.
    """
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        tops = {n.split("/", 1)[0] for n in names if "/" in n}
        missing: dict[str, str] = {}  # missing target -> first importer seen
        for entry in sorted(names):
            if not entry.endswith(".py"):
                continue
            try:
                tree = ast.parse(z.read(entry).decode("utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue  # not our problem to lint; packaging is the job here
            for node in ast.walk(tree):
                for target in _missing_import_targets(node, entry, tops, names, repo_root):
                    missing.setdefault(target, entry)
    return [f"{target} <- imported by {missing[target]}" for target in sorted(missing)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(os.environ.get("TEMP", ".")) / "satquery-code.zip"))
    args = ap.parse_args()
    out = Path(args.out)

    files: list[Path] = []
    # The archive and its sidecars, expressed relative to the repo so the check
    # below is a set lookup on a path we already have. Resolving every candidate
    # instead would cost a syscall per file -- and there are ~1.1 M of them
    # before pruning, so it is the difference between a fast build and a slow
    # one. `excluded()`'s suffix rule covers the documented `*.zip` naming; this
    # covers every other name an `--out` might use.
    try:
        out_rel = out.resolve().relative_to(REPO_ROOT)
    except ValueError:
        out_rel = None  # outside the repo, so it cannot be packaged anyway
    own_outputs: set[Path] = set()
    if out_rel is not None:
        own_outputs = {
            out_rel,
            out_rel.with_suffix(out_rel.suffix + ".sha256"),
            out_rel.with_suffix(out_rel.suffix + ".provenance.json"),
        }

    for rel in iter_source_files(REPO_ROOT):
        if excluded(rel) or rel in own_outputs:
            continue
        if rel.suffix in EXCLUDE_SUFFIXES or rel.suffix not in INCLUDE_SUFFIXES:
            continue
        files.append(rel)

    # Sorted by Path (parts-wise), which is exactly what the previous
    # `sorted(REPO_ROOT.rglob("*"))` produced. Sorting the strings instead would
    # order `a.py` before `a/b.py` where Path orders them the other way, which
    # would reorder the zip entries and change the archive digest for no reason.
    files.sort()

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in files:
            z.write(REPO_ROOT / rel, rel.as_posix())

    archive_sha256 = _sha256(out)

    # Assert the files the notebook's own cell 3 asserts, so a broken package
    # fails here rather than on Kaggle.
    #
    # The Phase 6 group was missing until 2026-09-23. The list held only the
    # change-VQA entrypoints, while RUNBOOK_PHASE6_VLM_KAGGLE.md §1 told the
    # reader that this assertion covers `scripts/phase6_train_vlm.py` and the
    # `training/vlm/` modules. It did not: a package missing the entire Phase 6
    # pipeline printed "required: all present" and exited 0, and the failure
    # would have surfaced on Kaggle after the upload. An assertion that does not
    # cover what the docs say it covers is worse than none, because it is
    # trusted.
    #
    # It failed a third time (D4c): every entry lived under `training/vlm/`, so a
    # package missing `training/data/` still passed. Six entries were added -- but
    # the list is still a hand-maintained sample, which is why the closure check
    # below now derives the assertion from the code (D4d).
    required = [
        # change-VQA (R-02)
        "scripts/train_change.py", "scripts/eval_change.py",
        "training/change/train.py", "training/change/dataset.py",
        "specialists/change/stanet.py",
        # Phase 6 (R-01)
        "scripts/phase6_train_vlm.py",
        "training/vlm/__init__.py", "training/vlm/artifact.py",
        "training/vlm/cli.py", "training/vlm/collate.py",
        "training/vlm/config.py", "training/vlm/dataset.py",
        "training/vlm/evaluate.py", "training/vlm/formatting.py",
        "training/vlm/lora.py", "training/vlm/run_manifest.py",
        "training/vlm/trainer.py",
        # `training.data` (D4c). All six are required by the import closure:
        # `training/vlm/dataset.py:526` imports `training.data.bigearthnet`, :528
        # `training.data.bigearthnet_blocks` and :286/:668
        # `training.data.bigearthnet_labels`; and `training/data/__init__.py:16`
        # eagerly imports `training.data.bigearthnet` while :26 eagerly imports
        # `training.data.cdvqa` -- so `from training.data.bigearthnet import
        # CLC19_CLASSES` executes `__init__.py` and fails without `cdvqa.py` too.
        # `vrsbench.py` is not imported by `__init__.py`; it is listed for
        # consistency. These six were absent from the archive entirely (D4).
        "training/data/__init__.py", "training/data/bigearthnet.py",
        "training/data/bigearthnet_blocks.py", "training/data/bigearthnet_labels.py",
        "training/data/cdvqa.py", "training/data/vrsbench.py",
        # shared
        "configs/base.yaml", "core/config.py", "scripts/check_env.py",
    ]
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
    missing = [r for r in required if r not in names]

    # D4d -- the durable fix. `required` above is a hand-maintained sample, and
    # that sample failed three times; this derives the assertion from the code
    # instead. It fails the build (exit 1) with the missing paths named, so a
    # broken package cannot be uploaded.
    missing_imports = assert_import_closure(out, REPO_ROOT)

    print(f"packaged {len(files)} files -> {out}")
    print(f"  size    : {out.stat().st_size / 1e6:.1f} MB")
    print(f"  sha256  : {archive_sha256}")
    print(f"  required: {'all present' if not missing else 'MISSING ' + str(missing)}")
    if missing_imports:
        print(f"  imports : MISSING {len(missing_imports)} intra-repo module(s):")
        for target in missing_imports[:20]:
            print(f"            {target}")
    else:
        print("  imports : closure ok")

    # Provenance defect section 6b: the code that produced an export needs an
    # identity a reviewer can pin, and a Kaggle code dataset has no .git. The
    # archive digest is that identity -- record it next to the archive so it can
    # be handed to the notebook (SATQUERY_CODE_SHA256) and written into the
    # export's hashes.json. This is the same digest a reviewer gets from
    # `sha256sum satquery-code.zip`, so it is independently reproducible.
    sidecar = out.with_suffix(out.suffix + ".sha256")
    sidecar.write_text(f"{archive_sha256}  {out.name}\n", encoding="utf-8")

    # The snapshot digest covers the same tree by content, so it survives a
    # re-zip. Both are recorded because they answer different questions: the
    # archive digest identifies THE UPLOAD, the snapshot digest identifies THE
    # CODE.
    snapshot = code_snapshot_digest(REPO_ROOT)
    print(f"  snapshot: {snapshot['files']} files, {snapshot['bytes']} B, "
          f"{snapshot['sha256']}")
    provenance = {
        "archive": {
            "name": out.name,
            "bytes": out.stat().st_size,
            "sha256": archive_sha256,
        },
        "code_snapshot": snapshot,
        "required_files": {r: (r in names) for r in required},
        "import_closure": {
            "ok": not missing_imports,
            "missing": missing_imports,
        },
        "included": sorted(rel.as_posix() for rel in files),
        "note": (
            "Hand `archive.sha256` to the notebook as SATQUERY_CODE_SHA256 so "
            "the export can record which upload produced it. It is the value "
            "`sha256sum <archive>` prints."
        ),
    }
    sidecar.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"  sidecar : {sidecar.name}, {sidecar.stem}.provenance.json")
    if missing or missing_imports:
        return 1
    print("\nUpload this zip as a Kaggle dataset (type: file), then attach it.")
    return 0


def _sha256(path: Path) -> str:
    """Hash a file, or say it is unavailable. Never invent a digest."""
    import hashlib

    if not path.exists():
        return "unavailable"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
