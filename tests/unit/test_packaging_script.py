"""Regression tests for `scripts/package_kaggle_code.py`.

This script had **no test at all**, and it produced three separate defects that
all reached the point of being found by hand:

1. It shipped two entire virtualenvs (`.venv312`, `.venv312b`) plus root-level
   scratch files, because `excluded()` matched only the literal name `.venv`.
   Caught by comparing the archive (7,114 files) against `code_snapshot_files`
   (409).
2. It packaged its **own provenance sidecar**. `.json` is an included suffix and
   `<archive>.zip.provenance.json` is not dot-prefixed, so a second build swept
   the first build's record into the archive. The file count moved 346 -> 347 and
   the digest changed without any source change. Caught by comparing build 1
   against build 2.
3. Its `required` assertion covered **no Phase 6 file**, while
   `RUNBOOK_PHASE6_VLM_KAGGLE.md` §1 tells the reader that it does.

A fourth problem, found while fixing the second: the script filtered the output of
a full `REPO_ROOT.rglob("*")` instead of pruning as it walked. The tree carries
`.venv`, `.venv312` and `.venv312b`, so the walk materialised **1,088,312 paths**
and spent **103.5 s** to discard all but ~350 of them. Replacing it with a pruned
`os.walk` took the build from **139 s to 3 s** and produced a **byte-identical**
archive (347 files, same digest) -- which is the point: the optimisation is only
correct because the archive did not change.

5. **D4**: `EXCLUDE_DIRS` held `"data"` and matched it at *every* depth, so
   `training/data/*.py` was excluded and the Kaggle notebook died at cell 8 with
   `ModuleNotFoundError: No module named 'training.data'`. The archive shipped
   347 files with **zero** `training/data/` entries, while `required` printed
   "all present" -- a third time a hand-maintained sample missed what the docs
   said it covered. Fixing it made `training/data/vrsbench/` visible, whose two
   `.json` payloads are **75.2 MB** (D4b), and the durable fix asserts the
   archive's **import closure** rather than a hand-listed sample (D4c/D4d).

The load-bearing test is `test_rebuild_with_sidecar_present_is_identical`.
Reproducibility is the one property a provenance sidecar exists to establish, so
defect 2 made the artefact undermine its own purpose -- and it is invisible to
any test that builds once, because the first build is what leaves the sidecar
behind. The bug lives in the interaction between a build and the previous
build's droppings, which is why nothing caught it.

That test had to be rewritten before it was worth anything. Its first version
built into `tmp_path`, which is outside the repo, so the sidecar could not be
packaged whatever the code did -- and it duly **passed against the unfixed
script**. Every test in this file was then checked by reverting the fix and
confirming it fails; the six that survived that check are the ones that pin
something. A test that has never been seen to fail is a guess.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.package_kaggle_code import (
    EXCLUDE_DIRS,
    INCLUDE_SUFFIXES,
    PACKAGING_OUTPUT_SUFFIXES,
    excluded,
    iter_source_files,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The Phase 6 pipeline. `RUNBOOK_PHASE6_VLM_KAGGLE.md` §1 states the archive
#: must contain these and that the script asserts it.
PHASE6_REQUIRED = [
    "scripts/phase6_train_vlm.py",
    "training/vlm/__init__.py",
    "training/vlm/artifact.py",
    "training/vlm/cli.py",
    "training/vlm/collate.py",
    "training/vlm/config.py",
    "training/vlm/dataset.py",
    "training/vlm/evaluate.py",
    "training/vlm/formatting.py",
    "training/vlm/lora.py",
    "training/vlm/run_manifest.py",
    "training/vlm/trainer.py",
]

#: The `training.data` package (D4). All six are part of the import closure that
#: `training/vlm/dataset.py` and `training/data/__init__.py` pull in; none of them
#: reached the archive before the fix.
TRAINING_DATA_REQUIRED = [
    "training/data/__init__.py",
    "training/data/bigearthnet.py",
    "training/data/bigearthnet_blocks.py",
    "training/data/bigearthnet_labels.py",
    "training/data/cdvqa.py",
    "training/data/vrsbench.py",
]


def _build(
    out: Path, monkeypatch: pytest.MonkeyPatch, *, require_ok: bool = True
) -> dict[str, object]:
    """Run the packaging script exactly as the runbook invokes it."""
    monkeypatch.setattr(sys, "argv", ["package_kaggle_code.py", "--out", str(out)])
    rc = main()
    if require_ok:
        assert rc == 0, "packaging script reported a failure"

    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    return {
        "rc": rc,
        "names": names,
        "files": len(names),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }


@pytest.fixture
def build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build the archive once into a temporary directory."""

    def _run() -> dict[str, object]:
        return _build(tmp_path / "satquery-code.zip", monkeypatch)

    return _run


# --------------------------------------------------------------------------
# Defect 2 -- the script packaged its own provenance sidecar.
# --------------------------------------------------------------------------


def test_rebuild_with_sidecar_present_is_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second build must be byte-identical even though the first left a sidecar.

    This is the scenario that triggered the bug: build 1 writes
    `satquery-code.zip.provenance.json`, build 2 sees it in the tree. Before the
    fix build 2 absorbed it and produced a different digest.

    The output has to live **inside** the walked tree or the test is vacuous:
    when `--out` is outside it, the sidecar cannot be packaged no matter what
    the code does. That is not hypothetical -- this test was first written
    building into `tmp_path`, and it passed against the *unfixed* script.
    Patching `REPO_ROOT` to a miniature tree puts the output where the bug
    actually lived, so the assertion can fail.
    """
    module = sys.modules["scripts.package_kaggle_code"]
    _mini_tree(tmp_path)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    out = tmp_path / "satquery-code.zip"

    first = _build(out, monkeypatch, require_ok=False)

    # Precondition: the build must actually write the sidecar the bug was about.
    # Without this the test would pass vacuously if the sidecar stopped being
    # written at all.
    sidecar = out.with_suffix(".zip.provenance.json")
    assert sidecar.exists(), "precondition: build 1 must leave a provenance sidecar"

    second = _build(out, monkeypatch, require_ok=False)

    assert second["files"] == first["files"], (
        f"rebuild changed the file count {first['files']} -> {second['files']}: "
        "the archive is packaging something produced by a previous build"
    )
    assert second["sha256"] == first["sha256"], (
        "rebuild produced a different digest for identical source"
    )


def test_archive_contains_no_entry_named_after_itself(
    build, tmp_path: Path
) -> None:
    """No archive, sidecar or provenance record may appear inside the archive."""
    result = build()
    self_referential = [
        n for n in result["names"] if n.endswith(PACKAGING_OUTPUT_SUFFIXES) or n.endswith(".zip")
    ]
    assert self_referential == [], f"archive packaged its own output: {self_referential}"


@pytest.mark.parametrize(
    "rel",
    [
        "satquery-code.zip.sha256",
        "satquery-code.zip.provenance.json",
        "some/nested/satquery-code.zip.provenance.json",
        "anything.zip.provenance.json",
        "anything.zip.sha256",
    ],
)
def test_excluded_rejects_own_outputs_by_name(rel: str) -> None:
    assert excluded(Path(rel)) is True


# --------------------------------------------------------------------------
# Defect 1 -- the script shipped virtualenvs and scratch droppings.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel",
    [
        ".venv/lib/python3.11/site-packages/pip/__init__.py",
        ".venv312/Lib/site-packages/torch/__init__.py",
        ".venv312b/Lib/site-packages/numpy/__init__.py",
        ".scratch_bad.py",
        ".full.txt",
        "router_a53fnqz_/run.json",
        "chg_20260921T000000/config.yaml",
        "logs/train.log",
        "reports/summary.md",
        "data/bigearthnet_v2/metadata.parquet",
        "artifacts/vlm/run_manifest.json",
        ".pytest_cache/v/cache/nodeids",
    ],
)
def test_excluded_rejects_droppings(rel: str) -> None:
    assert excluded(Path(rel)) is True


@pytest.mark.parametrize(
    "rel",
    [
        "scripts/phase6_train_vlm.py",
        "scripts/package_kaggle_code.py",
        "training/vlm/artifact.py",
        "configs/base.yaml",
        "core/code_revision.py",
        "notebooks/kaggle_phase6_vlm_lora.ipynb",
        "docs/PHASE6_AUDIT_AND_CONTRACT.md",
        "RUNBOOK_PHASE6_VLM_KAGGLE.md",
    ],
)
def test_legitimate_source_is_not_excluded(rel: str) -> None:
    """Guard against over-exclusion: a fix that drops real source is worse.

    Every defect in this file was an *inclusion* bug, so the exclusions were
    widened twice. This is the counterweight.
    """
    assert excluded(Path(rel)) is False


# --------------------------------------------------------------------------
# Defect 3 -- the required-file assertion covered no Phase 6 file.
# --------------------------------------------------------------------------


def test_required_assertion_covers_the_phase6_pipeline(build) -> None:
    """The archive must contain the Phase 6 pipeline, and the script must assert it.

    Reading the sidecar tests the assertion list `main()` actually used, rather
    than a copy of it in the test.
    """
    result = build()
    assert not any(n.startswith("/") or ".." in n for n in result["names"])

    for rel in PHASE6_REQUIRED:
        assert rel in result["names"], f"{rel} missing from the archive"


def test_provenance_records_the_phase6_files_as_required(
    build, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`required_files` must list the Phase 6 pipeline, all present.

    Before this was fixed the key set held only the change-VQA entrypoints, so a
    package missing `training/vlm/` entirely still reported "all present" and
    exited 0.
    """
    out = tmp_path / "satquery-code.zip"
    _build(out, monkeypatch)

    provenance = json.loads(
        out.with_suffix(".zip.provenance.json").read_text(encoding="utf-8")
    )
    asserted = provenance["required_files"]

    missing_from_assertion = [r for r in PHASE6_REQUIRED if r not in asserted]
    assert missing_from_assertion == [], (
        f"required_files does not assert these Phase 6 files: {missing_from_assertion}"
    )
    assert all(asserted[r] for r in PHASE6_REQUIRED), (
        "a Phase 6 file is asserted but was reported absent"
    )


def test_required_assertion_fails_loudly_when_a_file_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verifier that cannot fail is not a verifier.

    Simulate the exact failure the assertion exists to catch: the Phase 6
    entrypoint is absent from the tree. The script must exit non-zero rather
    than print "all present".
    """
    real_excluded = sys.modules["scripts.package_kaggle_code"].excluded

    def fake_excluded(rel: Path) -> bool:
        if rel.as_posix() == "scripts/phase6_train_vlm.py":
            return True
        return real_excluded(rel)

    monkeypatch.setattr(
        sys.modules["scripts.package_kaggle_code"], "excluded", fake_excluded
    )

    out = tmp_path / "satquery-code.zip"
    monkeypatch.setattr(sys, "argv", ["package_kaggle_code.py", "--out", str(out)])
    assert main() == 1, "a package missing the Phase 6 entrypoint must fail locally"


# --------------------------------------------------------------------------
# Structural sanity, so the archive stays uploadable.
# --------------------------------------------------------------------------


def test_archive_has_no_dot_directories(build) -> None:
    """No dot-prefixed *directory* component may appear in the archive.

    The assertion used to scan every path *segment*, so it rejected a dot-prefixed
    *file* at depth >= 2 as well -- `frontend/.impeccable.md` -- and failed on the
    real archive. That was an over-broad assertion, not a packaging bug: the
    contract is "zero dot-directories", and `excluded()` deliberately rejects dot
    **directories** at any depth but dot **files** only at the repository root, so
    a legitimately dotted filename deeper in the tree ships by design
    (`excluded('frontend/.impeccable.md') is False`). The rule is narrowed to the
    directory components (`n.split("/")[:-1]`) so the test checks what its name
    and `docs/PHASE6_AUDIT_AND_CONTRACT.md` claim, without widening the production
    exclusion to satisfy a mis-written test.
    """
    result = build()
    leaked = [
        n
        for n in result["names"]
        if any(p.startswith(".") for p in n.split("/")[:-1])  # directory parts only
    ]
    assert leaked == [], f"dot-directory entries leaked into the archive: {leaked[:5]}"


def test_every_entry_has_an_included_suffix(build) -> None:
    result = build()
    bad = [n for n in result["names"] if Path(n).suffix not in INCLUDE_SUFFIXES]
    assert bad == [], f"entries with a non-included suffix: {bad[:5]}"


def test_exclude_dirs_are_actually_honoured(build) -> None:
    """`EXCLUDE_DIRS` must be load-bearing, not decorative."""
    assert EXCLUDE_DIRS, "EXCLUDE_DIRS is empty"
    result = build()
    offenders = [
        n
        for n in result["names"]
        if any(part in EXCLUDE_DIRS for part in n.split("/")[:-1])
    ]
    assert offenders == [], f"excluded directories leaked: {offenders[:5]}"


# --------------------------------------------------------------------------
# The walk prunes rather than filtering after the fact (the 139 s -> 3 s fix).
# --------------------------------------------------------------------------


def _mini_tree(root: Path) -> None:
    """A small tree carrying one of every shape the exclusion rules care about."""
    (root / "training" / "vlm").mkdir(parents=True)
    (root / "training" / "vlm" / "lora.py").write_text("x = 1\n", encoding="utf-8")
    (root / "configs").mkdir()
    (root / "configs" / "base.yaml").write_text("a: 1\n", encoding="utf-8")

    for rel in (
        ".venv312/Lib/site-packages/torch/__init__.py",
        ".venv312b/Lib/site-packages/numpy/__init__.py",
        ".venv/lib/python3.11/site-packages/pip/__init__.py",
        "logs/train.log",
        "reports/summary.md",
        "data/bigearthnet_v2/metadata.parquet",
        "artifacts/vlm/run_manifest.json",
        ".pytest_cache/v/cache/nodeids",
        "router_a53fnqz_/run.json",
        "chg_20260921T000000/config.yaml",
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("dropping\n", encoding="utf-8")


def test_walk_prunes_excluded_directories(tmp_path: Path) -> None:
    """Excluded directories are never descended into."""
    _mini_tree(tmp_path)
    got = {p.as_posix() for p in iter_source_files(tmp_path)}

    assert "training/vlm/lora.py" in got
    assert "configs/base.yaml" in got

    for prefix in (".venv", ".venv312", ".venv312b", "logs", "reports", "data", "artifacts"):
        leaked = [p for p in got if p.startswith(f"{prefix}/")]
        assert leaked == [], f"{prefix}/ was descended into: {leaked[:3]}"


def test_walk_and_excluded_agree(tmp_path: Path) -> None:
    """The walk's pruning and `excluded()` are two halves of one rule.

    They are applied independently -- pruning decides what is visited, and
    `excluded()` re-checks every candidate -- so if they ever disagree, one of
    them is wrong. This pins them together.
    """
    _mini_tree(tmp_path)
    disagreements = [p for p in iter_source_files(tmp_path) if excluded(p)]
    assert disagreements == [], (
        f"the walk yielded paths that `excluded()` rejects: {disagreements}"
    )


def test_own_output_is_excluded_even_when_out_lives_in_the_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An `--out` named something other than `*.zip` must still not package itself.

    `excluded()` only recognises the documented `*.zip.provenance.json` naming.
    This uses `weird.bin`, whose sidecar `weird.bin.provenance.json` ends in
    `.json` -- an included suffix, and not matched by the name rule -- so the
    only thing that can exclude it is the resolved-output check in `main()`.
    """
    module = sys.modules["scripts.package_kaggle_code"]
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "phase6_train_vlm.py").write_text("x = 1\n", encoding="utf-8")

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    out = tmp_path / "weird.bin"
    out.write_bytes(b"a stale archive from an earlier run")
    sidecar = tmp_path / "weird.bin.provenance.json"
    sidecar.write_text('{"archive": {"sha256": "stale"}}', encoding="utf-8")

    # Precondition: the name rule alone does NOT exclude this sidecar, so the
    # test is actually exercising the resolved-output check.
    assert excluded(Path("weird.bin.provenance.json")) is False

    monkeypatch.setattr(sys, "argv", ["package_kaggle_code.py", "--out", str(out)])
    main()

    packaged = zipfile.ZipFile(out).namelist()
    assert "weird.bin.provenance.json" not in packaged, (
        "the archive packaged its own provenance sidecar"
    )
    assert "weird.bin" not in packaged, "the archive packaged itself"


# --------------------------------------------------------------------------
# D4 -- `data` was matched at every depth, so `training/data/` was never shipped.
# --------------------------------------------------------------------------


def _mini_data_tree(root: Path) -> None:
    """A tree carrying both a repository-root `data/` and a `training/data/` package."""
    for rel in (
        "data/levir/train/a.png",
        "training/data/__init__.py",
        "training/data/bigearthnet.py",
        "training/data/vrsbench.py",
        "training/data/vrsbench/VRSBench_train.json",
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x = 1\n", encoding="utf-8")


def test_training_data_modules_are_packaged_but_root_data_is_not(build) -> None:
    """D4: `training/data/` is source and ships; the repository-root `data/` does not.

    Before the fix, `EXCLUDE_DIRS` contained `"data"` and matched it at every
    depth, so every `training/data/*.py` was dropped and the notebook died at
    cell 8 with `ModuleNotFoundError: No module named 'training.data'`. Against
    the unfixed script this fails on the first loop: the modules are absent.
    """
    result = build()
    names = set(result["names"])

    for rel in TRAINING_DATA_REQUIRED:
        assert rel in names, f"{rel} missing: `data` is still excluded recursively"

    leaked = [n for n in names if n == "data" or n.startswith("data/")]
    assert leaked == [], f"the repository-root data/ tree leaked into the archive: {leaked[:5]}"


def test_vrsbench_payload_is_excluded_but_the_module_is_not() -> None:
    """D4b: the payload rule is a *path prefix*, not a bare directory name.

    `training/data/vrsbench/` is the 75.2 MB corpus; `training/data/vrsbench.py`
    is a module that must ship. A bare-name rule cannot tell them apart. Against
    the unfixed script the module was excluded anyway (via `data`), so the
    `is False` assertion below is what fails on revert.
    """
    for rel in (
        "training/data/vrsbench/VRSBench_train.json",
        "training/data/vrsbench/VRSBench_EVAL_referring.json",
        "training/data/vrsbench/Images_train/x.png",
    ):
        assert excluded(Path(rel)) is True, f"{rel} would ship the VRSBench payload"

    assert excluded(Path("training/data/vrsbench.py")) is False, (
        "the vrsbench *module* must ship; the rule must not match on bare name"
    )


def test_payload_prefix_rule_is_component_aware() -> None:
    """D4b: the payload rule matches a path *component*, not a raw string prefix.

    `training/data/vrsbench` names a directory. A bare `str.startswith` would
    also swallow every sibling whose name merely begins with `vrsbench`
    (`vrsbench_extra.py`, `vrsbenchx/`), silently dropping real modules. This is
    the regression guard: it fails if the rule is ever loosened to a raw string
    prefix.
    """
    assert excluded(Path("training/data/vrsbench/VRSBench_train.json")) is True

    for rel in (
        "training/data/vrsbench_extra.py",
        "training/data/vrsbench_extra/mod.py",
        "training/data/vrsbenchx/y.py",
    ):
        assert excluded(Path(rel)) is False, (
            f"{rel} was excluded: the payload rule matched a raw string prefix, "
            "not a path component"
        )


def test_walk_keeps_vrsbench_prefixed_siblings(tmp_path: Path) -> None:
    """The prune half must be component-aware too, or the two halves disagree."""
    for rel in (
        "training/data/vrsbench/VRSBench_train.json",
        "training/data/vrsbench_extra.py",
        "training/data/vrsbench_extra/mod.py",
        "training/data/vrsbenchx/y.py",
    ):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x = 1\n", encoding="utf-8")

    got = {p.as_posix() for p in iter_source_files(tmp_path)}
    assert not any(p.startswith("training/data/vrsbench/") for p in got), (
        "the walk descended into the VRSBench payload"
    )
    for rel in (
        "training/data/vrsbench_extra.py",
        "training/data/vrsbench_extra/mod.py",
        "training/data/vrsbenchx/y.py",
    ):
        assert rel in got, f"the walk pruned the sibling {rel}"
    assert [p for p in iter_source_files(tmp_path) if excluded(p)] == []


def test_walk_keeps_training_data_modules_and_drops_the_vrsbench_payload(
    tmp_path: Path,
) -> None:
    """The prune half of the rule must agree with `excluded()` on D4/D4b."""
    _mini_data_tree(tmp_path)
    got = {p.as_posix() for p in iter_source_files(tmp_path)}

    assert "training/data/bigearthnet.py" in got, "the walk pruned `training/data/`"
    assert "training/data/vrsbench.py" in got, "the walk pruned the vrsbench module"
    assert not any(p.startswith("training/data/vrsbench/") for p in got), (
        "the walk descended into the VRSBench payload"
    )
    assert not any(p.startswith("data/") for p in got), (
        "the walk descended into the repository-root data/ tree"
    )


def test_import_closure_fails_on_an_archive_missing_a_module(tmp_path: Path) -> None:
    """D4d: the assertion must fail the build when an intra-repo module is absent.

    `required` is a hand-maintained sample and was wrong three times; this check
    derives the assertion from the code, so it must be able to fail. Against the
    unfixed script `assert_import_closure` does not exist and this test fails on
    the `is not None` assertion.
    """
    module = sys.modules["scripts.package_kaggle_code"]
    closure = getattr(module, "assert_import_closure", None)
    assert closure is not None, (
        "D4d: `assert_import_closure` must exist -- a hand-listed sample is not "
        "a closure assertion"
    )

    archive = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("pkg/__init__.py", "")
        z.writestr("pkg/consumer.py", "from pkg.missing import thing\n")

    missing = closure(archive, tmp_path)
    assert any("pkg/missing.py" in m for m in missing), (
        f"a missing intra-repo module was not reported: {missing}"
    )


def test_import_closure_passes_on_the_real_build(
    build, tmp_path: Path
) -> None:
    """The real archive's intra-repo imports must all resolve inside it."""
    module = sys.modules["scripts.package_kaggle_code"]
    closure = getattr(module, "assert_import_closure", None)
    assert closure is not None, "D4d: `assert_import_closure` must exist"

    result = build()
    archive = tmp_path / "satquery-code.zip"
    assert closure(archive, module.REPO_ROOT) == [], (
        "the real archive has an open import closure"
    )
    assert "training/data/bigearthnet.py" in result["names"]


def test_nested_data_directory_is_not_excluded() -> None:
    """D4: the rule is root-scoped, so a nested `data/` is ordinary source.

    A fix that deleted `"data"` outright would also pass this, which is why
    `test_training_data_modules_are_packaged_but_root_data_is_not` checks the
    other direction.
    """
    assert excluded(Path("pkg/sub/data/module.py")) is False
    assert excluded(Path("app/data/loader.py")) is False


def test_walk_descends_into_a_nested_data_directory(tmp_path: Path) -> None:
    """The prune half must not skip a nested `data/` either."""
    path = tmp_path / "pkg" / "data" / "loader.py"
    path.parent.mkdir(parents=True)
    path.write_text("x = 1\n", encoding="utf-8")

    got = {p.as_posix() for p in iter_source_files(tmp_path)}
    assert "pkg/data/loader.py" in got, "a nested data/ directory was pruned"


def test_walk_and_excluded_agree_on_a_nested_data_path(tmp_path: Path) -> None:
    """The two halves of the rule must agree on a nested `data` path (D4)."""
    path = tmp_path / "pkg" / "data" / "loader.py"
    path.parent.mkdir(parents=True)
    path.write_text("x = 1\n", encoding="utf-8")

    yielded = {p.as_posix() for p in iter_source_files(tmp_path)}
    assert "pkg/data/loader.py" in yielded
    assert excluded(Path("pkg/data/loader.py")) is False, (
        "the walk yields it but `excluded()` rejects it -- the halves disagree"
    )
    disagreements = [p for p in iter_source_files(tmp_path) if excluded(p)]
    assert disagreements == []

