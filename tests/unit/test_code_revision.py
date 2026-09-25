"""Tests for `core.code_revision` — provenance defect section 6b.

The defect: `hashes.json` pinned the trained checkpoint by sha256 but identified
the CODE only as a path, and the Kaggle revision probe returned
`{"git_available": false, "git_commit": null}` because a Kaggle code dataset has
no `.git`. So the export proved which weights ran and not which code produced
them.

These tests are written so that a plausible-looking but WRONG implementation
fails. In particular they distinguish this module from the easy alternative:

    def revision(root): return {"git_commit": None, "git_available": False}

which produces a structurally valid block that carries no information at all.
`test_the_block_carries_an_identifier_that_actually_varies_with_content` exists
precisely to catch that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.code_revision import (
    CODE_SNAPSHOT_ALGORITHM,
    REPO_ROOT,
    code_revision,
    code_snapshot_digest,
    code_snapshot_files,
    revision_block,
    uploaded_dataset_identity,
)


# ---------------------------------------------------------------------------
# fixture trees
# ---------------------------------------------------------------------------


def _tree(tmp_path: Path, name: str = "code") -> Path:
    """A minimal code tree with one file of each interesting kind."""
    root = tmp_path / name
    (root / "scripts").mkdir(parents=True)
    (root / "training" / "inner").mkdir(parents=True)
    (root / "configs").mkdir(parents=True)
    (root / "scripts" / "a.py").write_text("print(1)\n", encoding="utf-8")
    (root / "training" / "inner" / "b.py").write_text("print(2)\n", encoding="utf-8")
    (root / "configs" / "base.yaml").write_text("key: value\n", encoding="utf-8")
    (root / "README.md").write_text("# readme\n", encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# what the digest covers
# ---------------------------------------------------------------------------


def test_digest_covers_source_and_config(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    rels = [rel for rel, _ in code_snapshot_files(root)]
    assert "scripts/a.py" in rels
    assert "training/inner/b.py" in rels
    assert "configs/base.yaml" in rels
    assert "README.md" in rels


def test_digest_excludes_binaries_and_caches(tmp_path: Path) -> None:
    """A digest that swept in .venv or __pycache__ would be enormous and would
    change for reasons unrelated to the code."""
    root = _tree(tmp_path)
    (root / "weights.pt").write_bytes(b"\x00\x01\x02\x03")
    (root / "blob.bin").write_bytes(b"\x00" * 32)
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "ghost.py").write_text("x = 1\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "pkg.js").write_text("1\n", encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "note.md").write_text("not code\n", encoding="utf-8")
    (root / "artifacts").mkdir()
    (root / "artifacts" / "meta.json").write_text("{}\n", encoding="utf-8")

    rels = [rel for rel, _ in code_snapshot_files(root)]
    for excluded in (
        "weights.pt", "blob.bin", "__pycache__/ghost.py", ".git/HEAD",
        "node_modules/pkg.js", "data/note.md", "artifacts/meta.json",
    ):
        assert excluded not in rels, f"{excluded} should not be covered"


def test_the_file_list_is_sorted_so_directory_order_cannot_matter(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    rels = [rel for rel, _ in code_snapshot_files(root)]
    assert rels == sorted(rels)


# ---------------------------------------------------------------------------
# the digest's contract
# ---------------------------------------------------------------------------


def test_the_digest_is_reproducible(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    first = code_snapshot_digest(root)
    second = code_snapshot_digest(root)
    assert first["sha256"] == second["sha256"]
    assert first["available"] is True
    assert len(first["sha256"]) == 64


def test_the_digest_moves_when_a_byte_changes(tmp_path: Path) -> None:
    """The whole point. An identifier that does not track content identifies
    nothing."""
    root = _tree(tmp_path)
    before = code_snapshot_digest(root)["sha256"]
    (root / "scripts" / "a.py").write_text("print(11)\n", encoding="utf-8")
    after = code_snapshot_digest(root)["sha256"]
    assert before != after


def test_the_digest_moves_when_a_file_is_renamed(tmp_path: Path) -> None:
    """Same bytes, different path, different code. A digest over contents only
    would miss a rename that changes which module gets imported."""
    root = _tree(tmp_path)
    before = code_snapshot_digest(root)["sha256"]
    (root / "scripts" / "a.py").rename(root / "scripts" / "renamed.py")
    after = code_snapshot_digest(root)["sha256"]
    assert before != after


def test_the_digest_moves_when_a_file_is_added(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    before = code_snapshot_digest(root)["sha256"]
    (root / "scripts" / "new.py").write_text("x = 1\n", encoding="utf-8")
    after = code_snapshot_digest(root)["sha256"]
    assert before != after


def test_the_digest_moves_when_a_file_is_removed(tmp_path: Path) -> None:
    """Removal is tested by NARROWING WHAT IS COVERED, not by deleting a file.

    WHY NOT `unlink()`: this sandbox intercepts destructive path operations and
    aborts the interpreter with SystemExit before the assertion can run -- the
    traceback points at the sandbox shim, not at any code here. A test that only
    passes where deletion is permitted would be green locally and red in CI for
    the wrong reason.

    The property under test is that the digest is a function of the SET of
    covered files. Excluding one file by suffix exercises exactly that, through
    the same code path, deterministically on every platform.
    """
    root = _tree(tmp_path)
    full = code_snapshot_digest(root)["sha256"]

    # Drop every .md: `README.md` leaves the covered set, nothing else does.
    narrowed = code_snapshot_digest(root, suffixes=(".py", ".yaml", ".yml"))["sha256"]

    assert full != narrowed
    assert "README.md" not in {
        rel for rel, _ in code_snapshot_files(root, suffixes=(".py", ".yaml", ".yml"))
    }
    assert "README.md" in {rel for rel, _ in code_snapshot_files(root)}


def test_two_trees_with_different_roots_are_distinguishable(tmp_path: Path) -> None:
    """The digest is over RELATIVE paths, so it is a property of the code and not
    of where the code happens to be mounted. Two identical trees under different
    names must agree -- otherwise the digest would change when Kaggle renames a
    dataset."""
    a = _tree(tmp_path, "one")
    b = _tree(tmp_path, "two")
    assert code_snapshot_digest(a)["sha256"] == code_snapshot_digest(b)["sha256"]

    (b / "configs" / "base.yaml").write_text("key: other\n", encoding="utf-8")
    assert code_snapshot_digest(a)["sha256"] != code_snapshot_digest(b)["sha256"]


def test_the_digest_records_what_it_covered(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    result = code_snapshot_digest(root)
    assert result["available"] is True
    assert result["files"] == len(code_snapshot_files(root))
    assert result["bytes"] > 0
    assert result["algorithm"] == CODE_SNAPSHOT_ALGORITHM
    # The algorithm string must name the separator convention, or a reviewer
    # cannot reimplement it.
    assert "relpath" in CODE_SNAPSHOT_ALGORITHM
    assert "sha256" in CODE_SNAPSHOT_ALGORITHM


# ---------------------------------------------------------------------------
# honesty: unavailable is None, never a placeholder
# ---------------------------------------------------------------------------


def test_an_empty_tree_yields_none_not_a_plausible_digest(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = code_snapshot_digest(empty)
    assert result["available"] is False
    assert result["sha256"] is None
    assert result["files"] == 0
    assert "note" in result and result["note"]


def test_an_unreadable_file_names_itself_and_refuses_to_claim_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silently skipping an unreadable file would produce a digest that looks
    complete while covering less -- the same class of defect as 6b itself."""
    root = _tree(tmp_path)
    (root / "scripts" / "locked.py").write_text("secret = 1\n", encoding="utf-8")

    real_read_bytes = Path.read_bytes

    def flaky(self: Path) -> bytes:
        if self.name == "locked.py":
            raise PermissionError("probe")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", flaky)
    result = code_snapshot_digest(root)
    assert result["available"] is False
    assert result["sha256"] is None
    assert "locked.py" in result["note"]


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------


def test_an_absent_git_is_reported_as_absent_not_as_an_empty_commit(tmp_path: Path) -> None:
    """This is the exact Kaggle case that produced `git_available: false`."""
    root = _tree(tmp_path)
    result = code_revision(root)
    assert result["available"] is False
    assert result["commit"] is None
    assert result["reason"], "a caller must be able to learn WHY it is absent"


def test_a_real_git_repository_yields_a_commit_and_a_dirty_flag() -> None:
    """Run against the actual repository. Skipped if this checkout has no .git,
    which is legitimate: the assertion is about the code path, not the checkout.
    """
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("this checkout has no .git directory")
    result = code_revision(REPO_ROOT)
    if not result["available"]:
        pytest.skip(f"git unavailable here: {result['reason']}")
    assert result["commit"] and len(result["commit"]) >= 7
    assert isinstance(result["dirty"], bool)


def test_the_dirty_flag_is_present_because_it_decides_whether_the_commit_identifies_the_files(
    tmp_path: Path,
) -> None:
    """A commit hash from a dirty tree does not identify the bytes that ran.
    A revision block that reports the commit but not the dirty flag overstates
    its provenance, so the field must exist even when git is unavailable."""
    result = code_revision(_tree(tmp_path))
    assert "dirty" in result
    assert result["dirty"] is None  # unknown, and visibly so


# ---------------------------------------------------------------------------
# Kaggle dataset identity
# ---------------------------------------------------------------------------


def test_dataset_identity_is_unavailable_off_kaggle(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    result = uploaded_dataset_identity(root)
    assert result["available"] is False
    assert result["slug"] is None


def test_dataset_identity_is_read_when_kaggle_exposes_it(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root.parent / "dataset-metadata.json").write_text(
        json.dumps({"id": "creatorballs/satquery-ai-code"}), encoding="utf-8"
    )
    result = uploaded_dataset_identity(root)
    assert result["available"] is True
    assert result["slug"] == "creatorballs/satquery-ai-code"
    assert result["source"]


def test_dataset_identity_accepts_the_slug_key_too(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root.parent / "dataset-metadata.json").write_text(
        json.dumps({"slug": "someone/some-code"}), encoding="utf-8"
    )
    assert uploaded_dataset_identity(root)["slug"] == "someone/some-code"


def test_a_malformed_metadata_file_is_not_treated_as_identity(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root.parent / "dataset-metadata.json").write_text("not json at all", encoding="utf-8")
    result = uploaded_dataset_identity(root)
    assert result["available"] is False


# ---------------------------------------------------------------------------
# the composite block
# ---------------------------------------------------------------------------


def test_the_block_carries_an_identifier_that_actually_varies_with_content(
    tmp_path: Path,
) -> None:
    """THE ANTI-STUB TEST.

    The tempting wrong implementation is

        return {"code_root": str(root),
                "revision": {"git_commit": None, "git_available": False}}

    which is structurally identical to the defective block being fixed and would
    pass any test that only checked for the presence of keys. This test requires
    the block to contain at least one identifier that changes when the code
    changes while git stays absent.
    """
    root = _tree(tmp_path)
    assert code_revision(root)["available"] is False, "precondition: no git"

    before = revision_block(root)
    assert before["snapshot"]["available"] is True
    assert before["snapshot"]["sha256"] is not None

    (root / "scripts" / "a.py").write_text("print(999)\n", encoding="utf-8")
    after = revision_block(root)
    assert after["snapshot"]["sha256"] != before["snapshot"]["sha256"], (
        "the revision block must identify the code even when git cannot"
    )


def test_the_block_states_how_many_identifiers_were_available(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    block = revision_block(root)
    assert "of 3 revision identifiers available" in block["note"]
    assert block["identifier_kinds"], "the kinds must be enumerated for a reader"


def test_the_block_does_not_claim_a_git_commit_it_does_not_have(tmp_path: Path) -> None:
    block = revision_block(_tree(tmp_path))
    assert block["git"]["available"] is False
    assert block["git"]["commit"] is None
    # The snapshot is present, but it must not be presented AS a commit.
    assert "commit" not in json.dumps(block["snapshot"])


def test_the_block_never_claims_more_identifiers_than_it_has(tmp_path: Path) -> None:
    """With no git and no Kaggle metadata, exactly one identifier (the snapshot)
    is available, and the note must say one -- not three."""
    block = revision_block(_tree(tmp_path))
    available = [
        k for k in ("git", "snapshot", "uploaded_dataset")
        if block[k].get("available") is True
    ]
    assert available == ["snapshot"]
    assert "1 of 3" in block["note"]


def test_the_block_is_json_serialisable(tmp_path: Path) -> None:
    """It is embedded in run_record.json and hashes.json, so it must survive
    `json.dumps(..., default=str)` without leaving non-serialisable objects."""
    block = revision_block(_tree(tmp_path))
    round_tripped = json.loads(json.dumps(block, sort_keys=True, default=str))
    assert round_tripped["snapshot"]["sha256"] == block["snapshot"]["sha256"]


def test_the_snapshot_can_be_skipped_for_a_cheap_block(tmp_path: Path) -> None:
    """A fast caller may want the git fact without hashing the tree. The option
    must actually omit the digest rather than returning an empty one."""
    block = revision_block(_tree(tmp_path), include_snapshot=False)
    assert "snapshot" not in block
    assert "note" in block


# ---------------------------------------------------------------------------
# against the real repository
# ---------------------------------------------------------------------------


def test_the_real_repository_snapshot_is_reproducible() -> None:
    first = code_snapshot_digest(REPO_ROOT)
    second = code_snapshot_digest(REPO_ROOT)
    assert first["sha256"] == second["sha256"]
    if first["available"]:
        assert first["files"] > 50, (
            "the project has far more than 50 covered files; a tiny count means "
            "the walk is not reaching the tree"
        )


def test_the_real_repository_snapshot_covers_the_files_the_notebook_asserts() -> None:
    """These are the CODE_MARKERS the Kaggle notebook uses to identify the
    project root. If the digest did not cover them it would not identify the code
    the notebook actually runs."""
    covered = {rel for rel, _ in code_snapshot_files(REPO_ROOT)}
    for marker in (
        "scripts/prepare_change_vqa.py",
        "training/change_vqa/model.py",
        "specialists/change/vqa_specialist.py",
        "configs/base.yaml",
    ):
        assert marker in covered, f"{marker} must be part of the code identity"
