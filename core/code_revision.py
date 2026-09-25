"""Identify the code revision that produced an artifact.

WHY THIS EXISTS (provenance defect §6b)
---------------------------------------
`hashes.json` pinned the CHECKPOINT byte-exactly (`run.checkpoint_sha256`,
`cfae5e43…`) while the CODE was identified only as a path
(`/code/code_root = /kaggle/input/…`). A path is not a revision: the same path
holding different bytes is indistinguishable. The `revision` block tried to help
but returned

    {"git_available": false, "git_commit": null, "git_dirty": null}

because a Kaggle code dataset has no `.git`. So the export proved which weights
ran and not which code produced them.

There is also a second, quieter problem with relying on a commit hash here: the
repository's working tree is **dirty** during these runs, so even a real
`git_commit` would not identify the uploaded bytes.

WHAT IS PROVIDED INSTEAD
------------------------
Three identifiers, recorded together, none of them pretending to be the others:

1. `git`            -- the commit and dirty flag, when a `.git` is actually
                       present. `available: false` otherwise, stated plainly.
2. `snapshot`       -- a content digest over the code files that determine
                       behaviour. Reproducible: same tree, same digest; one byte
                       changed, different digest.
3. `uploaded_dataset` -- the Kaggle dataset identity, when Kaggle exposes one.

A missing identifier is recorded as MISSING. `snapshot.sha256` is `None` rather
than a plausible-looking value when the digest cannot honestly be taken, because
a fabricated revision identifier is worse than an absent one -- the absent one is
visibly absent, the fabricated one is trusted.

WHAT THE SNAPSHOT DIGEST COVERS, AND WHY
----------------------------------------
Source, config and documentation (`.py .yaml .yml .json .ipynb .cfg .toml .ini
.md`). Behaviour is determined by those. Large binaries are excluded because the
two that matter are pinned separately by sha256 (the frozen STANet detector in
`hashes.json.inputs`, the trained head in `hashes.json.run`); including them would
make this digest slow without making it more specific. Dependency directories
(`.venv`, `node_modules`, `__pycache__`) and data/artifact trees are excluded for
the same reason.

The digest is `sha256` over `relpath\\0sha256(contents)\\n` for every covered file
in sorted-relative-path order. Sorting is not cosmetic: filesystem iteration
order is not stable across platforms, and a digest that depended on it would not
be reproducible.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "CODE_SNAPSHOT_ALGORITHM",
    "code_snapshot_digest",
    "code_snapshot_files",
    "code_revision",
    "uploaded_dataset_identity",
    "revision_block",
    "REPO_ROOT",
]

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Suffixes whose contents change behaviour. Ordered for a stable listing.
SNAPSHOT_SUFFIXES: tuple[str, ...] = (
    ".py", ".yaml", ".yml", ".cfg", ".toml", ".ini", ".json", ".ipynb", ".md",
)

#: Directories that are never part of "the code that ran".
SNAPSHOT_SKIP_DIRS: frozenset[str] = frozenset({
    "__pycache__", ".git", ".venv", ".venv-wsl", "node_modules",
    ".ipynb_checkpoints", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".git.damaged.bak",
    # Inputs and outputs, pinned separately or regenerable.
    "data", "artifacts",
})

SNAPSHOT_MAX_DEPTH = 12

#: The exact algorithm, recorded alongside the digest so a reviewer can
#: reimplement it without reading this file.
CODE_SNAPSHOT_ALGORITHM = (
    "sha256 over, for each covered file in sorted-relative-path order: "
    "utf8(relpath) + 0x00 + hex(sha256(contents)) + 0x0a"
)


def code_snapshot_files(
    root: str | Path,
    *,
    suffixes: Iterable[str] = SNAPSHOT_SUFFIXES,
    skip_dirs: Iterable[str] = SNAPSHOT_SKIP_DIRS,
    max_depth: int = SNAPSHOT_MAX_DEPTH,
) -> list[tuple[str, Path]]:
    """Covered code files under `root`, as `(relative posix path, path)`.

    Sorted by relative path so the caller's digest cannot depend on directory
    iteration order. Depth-bounded and skip-listed so pointing this at an
    unexpected directory cannot turn it into a filesystem-wide walk.
    """
    root = Path(root)
    wanted = {s.lower() for s in suffixes}
    skip = set(skip_dirs)
    found: list[tuple[str, Path]] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > max_depth:
            continue
        try:
            entries = sorted(current.iterdir(), key=lambda q: q.name)
        except OSError:
            continue
        for entry in entries:
            try:
                is_dir = entry.is_dir()
            except OSError:
                continue
            if is_dir:
                if entry.name in skip or entry.name.startswith("."):
                    continue
                stack.append((entry, depth + 1))
            elif entry.suffix.lower() in wanted:
                try:
                    found.append((entry.relative_to(root).as_posix(), entry))
                except ValueError:
                    continue
    return sorted(found)


def code_snapshot_digest(
    root: str | Path,
    *,
    suffixes: Iterable[str] = SNAPSHOT_SUFFIXES,
    skip_dirs: Iterable[str] = SNAPSHOT_SKIP_DIRS,
    max_depth: int = SNAPSHOT_MAX_DEPTH,
) -> dict[str, Any]:
    """Content digest over the code tree, plus what it covered.

    Returns `available`, `algorithm`, `files`, `bytes` and `sha256`.

    `available` is False and `sha256` is None when the digest cannot be taken --
    an empty tree, or a file that cannot be read. In the unreadable case the
    offending path is named, because silently dropping one file would produce a
    digest that looks complete while covering less, which is exactly the class
    of defect this module exists to remove.
    """
    base: dict[str, Any] = {
        "algorithm": CODE_SNAPSHOT_ALGORITHM,
        "suffixes": sorted(SNAPSHOT_SUFFIXES if suffixes is SNAPSHOT_SUFFIXES
                           else tuple(suffixes)),
    }
    files = code_snapshot_files(
        root, suffixes=suffixes, skip_dirs=skip_dirs, max_depth=max_depth
    )
    if not files:
        return {
            **base,
            "available": False,
            "files": 0,
            "bytes": 0,
            "sha256": None,
            "note": "no covered code files were reachable under this root",
        }

    digest = hashlib.sha256()
    total = 0
    for rel, path in files:
        try:
            blob = path.read_bytes()
        except OSError as exc:
            return {
                **base,
                "available": False,
                "files": len(files),
                "bytes": total,
                "sha256": None,
                "note": f"unreadable file {rel}: {type(exc).__name__}",
            }
        total += len(blob)
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(blob).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return {
        **base,
        "available": True,
        "files": len(files),
        "bytes": total,
        "sha256": digest.hexdigest(),
    }


def code_revision(root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """The git revision, when a `.git` is genuinely present.

    Returns `available: False` -- not a guessed value -- when git is absent, when
    the directory is not a git work tree, or when the command fails. `dirty` is
    reported because a dirty tree means the commit hash does NOT identify the
    files on disk, and a caller that hides that is overstating its provenance.
    """
    root = Path(root)
    out: dict[str, Any] = {
        "available": False,
        "commit": None,
        "dirty": None,
        "reason": None,
    }
    git = shutil.which("git")
    if git is None:
        out["reason"] = "git executable not found on PATH"
        return out
    if not (root / ".git").exists():
        out["reason"] = f"no .git directory under {root}"
        return out
    try:
        head = subprocess.run(
            [git, "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
        if head.returncode != 0:
            out["reason"] = f"git rev-parse exited {head.returncode}"
            return out
        status = subprocess.run(
            [git, "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=120,
        )
        out["available"] = True
        out["commit"] = head.stdout.strip() or None
        out["dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError) as exc:  # noqa: BLE001
        out["reason"] = f"{type(exc).__name__}: {exc}"
    return out


def uploaded_dataset_identity(root: str | Path) -> dict[str, Any]:
    """The Kaggle dataset identity, when Kaggle exposes metadata next to it.

    Kaggle mounts an attached dataset as a plain directory and does not put the
    slug in a guaranteed location, so this reads the metadata file when present
    and reports `available: False` when not. Off Kaggle it always reports
    unavailable. Nil returned rather than guessed.
    """
    root = Path(root)
    for candidate in (
        root.parent / "dataset-metadata.json",
        root / "dataset-metadata.json",
    ):
        try:
            if not candidate.is_file():
                continue
            meta = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        slug = meta.get("id") or meta.get("slug")
        if slug:
            return {"available": True, "slug": slug, "source": str(candidate)}
    return {"available": False, "slug": None, "source": None}


def revision_block(
    root: str | Path = REPO_ROOT,
    *,
    include_snapshot: bool = True,
) -> dict[str, Any]:
    """The full revision record: git, snapshot digest, uploaded-dataset identity.

    This is the value to embed in a manifest or export. It always carries a
    `note` saying which identifiers are present and which are not, so a reader
    cannot mistake an absent identifier for a present one.
    """
    root = Path(root)
    git = code_revision(root)
    block: dict[str, Any] = {
        "code_root": str(root),
        "git": git,
        "identifier_kinds": [
            "git.commit (only when a .git is present)",
            "snapshot.sha256 (content digest over source/config files)",
            "uploaded_dataset.slug (Kaggle dataset id, when exposed)",
        ],
    }
    if include_snapshot:
        block["snapshot"] = code_snapshot_digest(root)
    block["uploaded_dataset"] = uploaded_dataset_identity(root)

    distinct = [
        k for k in ("git", "snapshot", "uploaded_dataset")
        if (block.get(k) or {}).get("available") in (True,)
    ]
    block["note"] = (
        f"{len(distinct)} of 3 revision identifiers available "
        f"({', '.join(distinct) if distinct else 'none'}). An identifier that is "
        "unavailable is recorded as unavailable, never as a placeholder value. "
        "The trained checkpoint is pinned separately by sha256."
    )
    return block
