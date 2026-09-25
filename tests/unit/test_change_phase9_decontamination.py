"""Phase 9 -- pins that stop the change-detection module from being re-contaminated.

WHAT WENT WRONG, AND WHY IMPORTING WAS NOT ENOUGH
-------------------------------------------------
An earlier autonomous run pasted Phase 4 (router) and Phase 7/8 (grounding) code
into `scripts/train_change.py`, `scripts/eval_change.py` and
`scripts/smoke_test_change.py`. The result was subtle and expensive:

  * the 1,006-test suite stayed GREEN, because nothing imported those scripts;
  * all three files IMPORTED cleanly, so nothing looked broken;
  * none of them could EXECUTE. The undefined names -- `TASK_TO_INDEX`,
    `BINARY_HEADS`, `similarity_map`, `argmax_candidate`, `image_cache_path`,
    `save_adapter`, `ZERO_SHOT_BASELINE_IOU` and ~60 more -- only surfaced when
    the code path actually ran.

An import is a syntax-and-module-level check. It says nothing about the inside
of a function body. So this file adds three independent guards:

  1. a FORBIDDEN-IDENTIFIER scan -- the grounding and router vocabulary must not
     appear at all, and the LEVIR-CD change-detection vocabulary must;
  2. a REAL undefined-name check over the symbol tables of every change module,
     which is the class of defect that caused the outage. The checker is itself
     self-tested below, so it cannot silently rot into a no-op;
  3. a subprocess run of `--dry-run`, which proves the CLI is EXECUTABLE and not
     merely importable, and that it fails cleanly rather than with a traceback.

`tests/unit/test_change_train_script_contract.py` is the pre-existing contract
and is left untouched.
"""

from __future__ import annotations

import builtins
import re
import subprocess
import symtable
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Every module that must stay change-detection specific.
CHANGE_MODULES: tuple[str, ...] = (
    "scripts/train_change.py",
    "scripts/eval_change.py",
    "scripts/smoke_test_change.py",
    "training/change/train.py",
    "training/change/dataset.py",
    "training/change/__init__.py",
)

#: The three scripts the task calls out by name.
CHANGE_SCRIPTS: tuple[str, ...] = (
    "scripts/train_change.py",
    "scripts/eval_change.py",
    "scripts/smoke_test_change.py",
)

#: Vocabulary from Phase 4 (router), Phase 7/8 (grounding) and the RemoteCLIP
#: feature cache. None of it belongs in a change-detection module. The list is
#: the measured undefined-name set from the contaminated revision plus the
#: imports that pulled the wrong subsystem in.
FORBIDDEN_IDENTIFIERS: tuple[str, ...] = (
    # Phase 4 router
    "TASK_TO_INDEX",
    "MODALITY_TO_INDEX",
    "BINARY_HEADS",
    "save_adapter",
    "RouterAdapter",
    # Phase 7/8 grounding
    "ZERO_SHOT_BASELINE_IOU",
    "ZERO_SHOT_BASELINE_RECALL_50",
    "PHASE7_BASELINE_IOU",
    "PHASE7_BASELINE_RECALL_50",
    "RECALL_THRESHOLDS",
    "similarity_map",
    "argmax_candidate",
    "EncodedImage",
    "load_trained_head",
    "nms",
    # RemoteCLIP feature cache
    "RemoteCLIP",
    "remoteclip",
    "build_encoder",
    "image_cache_path",
    "extract_features",
    "attach_cached",
    "hf_hub_download",
    # the benchmark this phase does not use
    "VRSBench",
    "vrsbench",
)

#: Change-detection vocabulary each module must carry instead. Per-module so a
#: script cannot satisfy the check by mentioning one symbol in a docstring.
REQUIRED_IDENTIFIERS: dict[str, tuple[str, ...]] = {
    "scripts/train_change.py": (
        "change_loss",
        "train_change_head",
        "evaluate",
        "load_levir_dataset",
        "split_by_image",
        "assert_image_disjoint",
        "LEVIR",
    ),
    "scripts/eval_change.py": (
        "load_levir_dataset",
        "load_trained_change_model",
        "score_dataset",
        "change",
        "LEVIR",
    ),
    "scripts/smoke_test_change.py": (
        "train_change_head",
        "split_by_image",
        "assert_image_disjoint",
        "load_levir_dataset",
        "LEVIR",
    ),
    "training/change/train.py": (
        "change_loss",
        "train_change_head",
        "evaluate",
        "load_trained_change_model",
        "assert_image_disjoint",
        "LEVIR",
    ),
}


def _source(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _mentions(source: str, identifier: str) -> bool:
    """Whole-word match, so `nms` cannot be satisfied by a longer word."""
    return re.search(
        r"(?<![A-Za-z0-9_])" + re.escape(identifier) + r"(?![A-Za-z0-9_])", source
    ) is not None


# ---------------------------------------------------------------------------
# 1. Vocabulary: the wrong subsystem must be gone, the right one present
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", CHANGE_MODULES)
def test_change_modules_contain_no_router_or_grounding_identifiers(module: str) -> None:
    """The contamination must not come back, in code OR in prose."""
    source = _source(module)
    present = [t for t in FORBIDDEN_IDENTIFIERS if _mentions(source, t)]
    assert not present, (
        f"{module} references {present}, which belong to the Phase 4 router / "
        f"Phase 7-8 grounding subsystem. This is the contamination that made "
        f"the change scripts importable but not executable."
    )


@pytest.mark.parametrize("module", sorted(REQUIRED_IDENTIFIERS))
def test_change_modules_reference_change_detection_identifiers(module: str) -> None:
    """Absence of the wrong code is not presence of the right code."""
    source = _source(module)
    missing = [t for t in REQUIRED_IDENTIFIERS[module] if not _mentions(source, t)]
    assert not missing, f"{module} does not reference {missing}"


@pytest.mark.parametrize("module", CHANGE_SCRIPTS)
def test_change_scripts_import_only_change_subsystems(module: str) -> None:
    """No import may reach into the grounding or router packages."""
    source = _source(module)
    for banned in (
        "training.grounding",
        "specialists.grounding",
        "specialists.change.remoteclip",
        "router.",
        "import router",
    ):
        assert banned not in source, f"{module} imports {banned!r}"


# ---------------------------------------------------------------------------
# 2. The real defect class: names referenced but never bound
# ---------------------------------------------------------------------------

#: Names that are legitimately global without a binding in the file.
_ALLOWED_UNBOUND: frozenset[str] = frozenset(
    {
        "__file__", "__name__", "__doc__", "__package__", "__spec__",
        "__loader__", "__builtins__", "__annotations__", "__all__", "__path__",
        "__class__", "self", "cls",
    }
)


def _unbound_module_names(source: str, filename: str = "<source>") -> list[str]:
    """Names referenced from a global scope but bound nowhere in the module.

    This is the static check whose absence let the contaminated revision ship.
    It walks every scope in the module's symbol table and reports names that are
    GLOBAL, referenced, and neither assigned nor imported there.

    Three deliberate exclusions, each verified by the self-test below:

      * builtins, and the module's own dunders;
      * names bound anywhere at module scope (a function may legitimately use a
        module-level import);
      * FREE variables. A closure over a variable defined in an enclosing
        function is not an undefined name, and flagging it would make this
        guard noisy enough that someone would delete it.
    """
    top = symtable.symtable(source, filename, "exec")
    allowed = set(dir(builtins)) | set(_ALLOWED_UNBOUND)
    allowed |= {
        symbol.get_name()
        for symbol in top.get_symbols()
        if symbol.is_assigned() or symbol.is_imported()
    }

    found: set[str] = set()

    def walk(table: symtable.SymbolTable) -> None:
        for symbol in table.get_symbols():
            if not symbol.is_referenced():
                continue
            if symbol.is_assigned() or symbol.is_imported():
                continue
            if not symbol.is_global():
                continue
            if symbol.get_name() in allowed:
                continue
            found.add(symbol.get_name())
        for child in table.get_children():
            walk(child)

    walk(top)
    return sorted(found)


def test_unbound_name_checker_catches_the_contamination_pattern() -> None:
    """Self-test: the guard must actually detect what it claims to detect.

    Without this, a bug in `_unbound_module_names` would turn every check below
    into a silent pass -- which is exactly how the original contamination
    survived a green suite.
    """
    contaminated = (
        "def _to_tensors(examples, embeddings, device):\n"
        "    return [TASK_TO_INDEX[e.task] for e in examples]\n"
    )
    assert _unbound_module_names(contaminated) == ["TASK_TO_INDEX"]

    assert _unbound_module_names(
        "def f():\n    return totally_undefined_thing\n"
    ) == ["totally_undefined_thing"]


def test_unbound_name_checker_does_not_flag_closures_or_imports() -> None:
    """Self-test: no false positives on the two shapes this codebase uses."""
    closure = (
        "import math\n"
        "def outer(values):\n"
        "    ordered = sorted(values)\n"
        "    def at(q):\n"
        "        return ordered[q]\n"
        "    return at(math.floor(0.5))\n"
    )
    assert _unbound_module_names(closure) == []


@pytest.mark.parametrize("module", CHANGE_MODULES)
def test_change_modules_have_no_unbound_module_level_names(module: str) -> None:
    """No change module may reference a name it never binds.

    This is the assertion that would have failed on the contaminated revision
    while the import test passed.
    """
    unbound = _unbound_module_names(_source(module), module)
    assert not unbound, (
        f"{module} references {unbound} without binding them. These would raise "
        f"NameError only at runtime -- the exact failure mode that left three "
        f"change scripts importable but not executable."
    )


# ---------------------------------------------------------------------------
# 3. The training module's public surface
# ---------------------------------------------------------------------------


def test_change_train_module_imports_and_exposes_entry_points() -> None:
    """`training/change/train.py` did not exist; it must exist and be complete."""
    from training.change import train as train_module

    for name in (
        "change_loss",
        "train_change_head",
        "evaluate",
        "load_trained_change_model",
    ):
        assert hasattr(train_module, name), f"training/change/train.py lost {name!r}"

    for name in ("ChangeTrainingError", "ChangePairDataset", "TrainingResult"):
        assert hasattr(train_module, name), f"training/change/train.py lost {name!r}"


def test_change_package_is_importable() -> None:
    """`training/change/__init__.py` did not exist either."""
    import training.change as package

    for name in ("load_levir_dataset", "split_by_image", "assert_image_disjoint"):
        assert hasattr(package, name), f"training.change lost {name!r}"


# ---------------------------------------------------------------------------
# 4. The CLI must EXECUTE, not merely import
# ---------------------------------------------------------------------------


def _run_train_change(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "train_change.py"), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_train_change_dry_run_fails_cleanly_on_a_missing_data_root(tmp_path: Path) -> None:
    """Non-zero exit, and NOT a traceback.

    A missing dataset is an operator error with a readable remedy, not a crash.
    """
    result = _run_train_change(
        "--data-root", str(tmp_path / "definitely-absent"), "--dry-run"
    )
    combined = result.stdout + result.stderr

    assert result.returncode != 0, "a missing data root must not exit 0"
    assert "Traceback" not in combined, (
        "a missing data root produced a traceback instead of a readable message:\n"
        + combined
    )
    assert "does not exist" in combined or "not found" in combined


def test_train_change_dry_run_validates_a_real_dataset_and_writes_nothing(
    tmp_path: Path,
) -> None:
    """With data present: exit 0, verify the split, and write no files.

    `--output-dir` points into `tmp_path`, so "wrote nothing" is checkable
    directly rather than inferred.
    """
    from PIL import Image

    root = tmp_path / "levir"
    for sub in ("A", "B", "label"):
        (root / "train" / sub).mkdir(parents=True, exist_ok=True)

    for index in range(4):
        stem = f"scene{index:02d}"
        base = (10 + index * 7) % 255
        t1 = Image.fromarray(np.full((16, 16, 3), base, dtype="uint8"))
        t2 = Image.fromarray(np.full((16, 16, 3), base + 40, dtype="uint8"))
        label = Image.fromarray(np.full((16, 16), 255, dtype="uint8"))
        t1.save(root / "train" / "A" / f"{stem}.png")
        t2.save(root / "train" / "B" / f"{stem}.png")
        label.save(root / "train" / "label" / f"{stem}.png")

    out_dir = tmp_path / "must-not-be-created"
    result = _run_train_change(
        "--data-root", str(root), "--output-dir", str(out_dir), "--dry-run"
    )
    combined = result.stdout + result.stderr

    assert "Traceback" not in combined, combined
    assert result.returncode == 0, combined
    assert "scene-disjoint" in combined
    assert not out_dir.exists(), "--dry-run must not write an output directory"
