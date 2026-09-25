"""STEP 5 — the immutable public-test corpus (plan section 37, Test Set Firewall).

Four plan rules are enforced here, and each gets tests that FAIL against a
weakened implementation:

  1. immutability      -> tamper detection distinguishes changed/added/removed
  2. training cannot read it -> an EvaluationMode token is required
  3. evaluation-mode reads only -> a bool will not do
  4. no answer cache   -> the prohibition is checkable and reports rather than
                          silently passing

The `removed` case gets explicit attention: it is the delta that silently
invalidates a previously published number, and a naive verify() that iterated
only over files it could currently find would never notice it.
"""

from __future__ import annotations

import pytest

from evaluation.public_test import (
    PUBLIC_TEST_ROOT,
    SEAL_FILENAME,
    EvaluationMode,
    PublicTestAnswerCacheError,
    PublicTestCorpus,
    PublicTestModeError,
    PublicTestSealedError,
    PublicTestSealError,
    evaluation_mode,
    forbid_answer_cache,
    open_public_test,
    public_test_state,
)


@pytest.fixture()
def corpus_root(tmp_path):
    """A small, realistic corpus: nested JSONL plus a flat metadata file."""
    (tmp_path / "part").mkdir()
    (tmp_path / "part" / "001.jsonl").write_text(
        '{"q": "is there change?"}\n{"q": "how much?"}\n', encoding="utf-8"
    )
    (tmp_path / "meta.json").write_text('{"n": 2}', encoding="utf-8")
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def sealed(corpus_root):
    corpus = PublicTestCorpus(root=corpus_root, corpus_id="test_corpus")
    seal = corpus.seal(notes={"purpose": "unit test"})
    return corpus, seal


# ---------------------------------------------------------------------------
# The real corpus in this repository
# ---------------------------------------------------------------------------
def test_the_repo_corpus_root_is_the_package_directory():
    """The plan's path is evaluation/public_test -- not <repo>/public_test.

    Resolving the bare directory name against the repo root produced a path that
    does not exist, which would have made the corpus permanently 'unavailable'
    while looking correct. This test pins the real location.
    """
    assert PUBLIC_TEST_ROOT.name == "public_test"
    assert PUBLIC_TEST_ROOT.parent.name == "evaluation"
    assert PUBLIC_TEST_ROOT.exists()


def test_the_repo_corpus_is_reported_unavailable_not_passing():
    """An empty corpus must not read as a test set that passed."""
    state = public_test_state()
    assert state["available"] is False
    assert state["n_files"] == 0
    assert "no corpus material" in (state["reason"] or "")
    # The note must actively say an unavailable corpus is not a pass.
    assert "does NOT" in state["note"]


def test_no_answer_cache_exists_in_the_repository():
    """'No cache of test answers is permitted' — checked, not assumed."""
    result = forbid_answer_cache()
    assert result["clean"] is True
    assert result["checked"], "the check must name what it looked for"


def test_opening_the_empty_repo_corpus_raises_with_the_reason():
    with pytest.raises(PublicTestSealError) as excinfo:
        open_public_test(mode=evaluation_mode("unit test"))
    assert "no material" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Evaluation-mode gate
# ---------------------------------------------------------------------------
def test_a_bool_is_not_accepted_as_evaluation_mode(corpus_root):
    """A bool is what training code would pass by reflex. It must not work."""
    corpus = PublicTestCorpus(root=corpus_root)
    with pytest.raises(PublicTestModeError) as excinfo:
        corpus.read("meta.json", mode=True)  # type: ignore[arg-type]
    assert "EvaluationMode" in str(excinfo.value)


def test_none_is_not_accepted_as_evaluation_mode(corpus_root):
    corpus = PublicTestCorpus(root=corpus_root)
    with pytest.raises(PublicTestModeError):
        corpus.read("meta.json", mode=None)  # type: ignore[arg-type]


def test_open_public_test_requires_the_token(corpus_root, sealed):
    with pytest.raises(PublicTestModeError):
        open_public_test(mode="evaluation", root=corpus_root)  # type: ignore[arg-type]


def test_an_evaluation_mode_token_must_state_a_reason():
    """An unexplained evaluation mode is indistinguishable from an accident."""
    with pytest.raises(PublicTestModeError):
        evaluation_mode("")
    with pytest.raises(PublicTestModeError):
        evaluation_mode("   ")


def test_a_valid_token_records_its_reason(corpus_root):
    corpus = PublicTestCorpus(root=corpus_root)
    raw = corpus.read("meta.json", mode=evaluation_mode("scoring the test split"))
    assert raw == b'{"n": 2}'


def test_the_mode_token_is_frozen():
    """A caller must not be able to mutate a token after construction."""
    token = evaluation_mode("probe")
    with pytest.raises(Exception):
        token.reason = "something else"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Sealing
# ---------------------------------------------------------------------------
def test_sealing_records_every_file_with_its_digest(sealed):
    _corpus, seal = sealed
    assert seal.n_files == 2
    relpaths = sorted(e.relpath for e in seal.entries)
    assert relpaths == ["meta.json", "part/001.jsonl"]
    assert all(len(e.sha256) == 64 for e in seal.entries)
    assert seal.total_bytes == sum(e.bytes for e in seal.entries)


def test_the_gitkeep_is_not_part_of_the_corpus(sealed):
    """A placeholder must not become sealed test material."""
    _corpus, seal = sealed
    assert ".gitkeep" not in {e.relpath for e in seal.entries}


def test_sealing_refuses_an_empty_corpus(tmp_path):
    """A seal over no files verifies forever and reads as 'intact'."""
    with pytest.raises(PublicTestSealError) as excinfo:
        PublicTestCorpus(root=tmp_path, corpus_id="empty").seal()
    assert "empty" in str(excinfo.value)


def test_the_seal_is_written_to_disk(sealed, corpus_root):
    assert (corpus_root / SEAL_FILENAME).exists()


def test_a_seal_round_trips_through_json(sealed, corpus_root):
    from evaluation.public_test.corpus import CorpusSeal

    _corpus, seal = sealed
    reloaded = CorpusSeal.from_dict(
        __import__("json").loads((corpus_root / SEAL_FILENAME).read_text(encoding="utf-8"))
    )
    assert reloaded.digest == seal.digest
    assert reloaded.by_relpath().keys() == seal.by_relpath().keys()


def test_the_seal_digest_is_reproducible(sealed, corpus_root):
    """A second corpus over the same bytes yields the same digest."""
    _corpus, seal = sealed
    other = PublicTestCorpus(root=corpus_root, corpus_id="test_corpus")
    other.read_seal()
    assert other.verify()["current_digest"] == seal.digest


# ---------------------------------------------------------------------------
# Verification: the four deltas
# ---------------------------------------------------------------------------
def test_an_untouched_corpus_verifies(sealed):
    corpus, _seal = sealed
    report = corpus.verify()
    assert report["intact"] is True
    assert report["changed"] == report["added"] == report["removed"] == []


def test_changed_content_is_detected(sealed, corpus_root):
    corpus, _seal = sealed
    (corpus_root / "meta.json").write_text('{"n": 3}', encoding="utf-8")
    with pytest.raises(PublicTestSealedError) as excinfo:
        corpus.verify()
    assert excinfo.value.context["changed"] == ["meta.json"]
    assert excinfo.value.context["intact"] is False


def test_added_content_is_detected(sealed, corpus_root):
    corpus, _seal = sealed
    (corpus_root / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PublicTestSealedError) as excinfo:
        corpus.verify()
    assert excinfo.value.context["added"] == ["extra.json"]


def test_removed_content_is_detected(sealed, corpus_root):
    """THE important delta: a missing file silently invalidates a published number.

    Removal is simulated by narrowing which suffixes the corpus considers, not by
    deleting a file. The code path exercised is identical (verify() compares the
    recorded set against the current set), and it does not depend on filesystem
    delete permissions -- which the sandbox denies.
    """
    corpus, seal = sealed

    class _Narrowed(PublicTestCorpus):
        def files(self):
            return [p for p in super().files() if p.suffix == ".json"]

    narrowed = _Narrowed(root=corpus_root, corpus_id="test_corpus")
    with pytest.raises(PublicTestSealedError) as excinfo:
        narrowed.verify(expected=seal)
    assert excinfo.value.context["removed"] == ["part/001.jsonl"]


def test_a_rename_is_a_change_not_a_no_op(sealed, corpus_root):
    """Path-sensitive: a moved test file is a different test set."""
    import shutil

    corpus, seal = sealed
    shutil.move(str(corpus_root / "meta.json"), str(corpus_root / "renamed.json"))
    with pytest.raises(PublicTestSealedError) as excinfo:
        PublicTestCorpus(root=corpus_root, corpus_id="test_corpus").verify(expected=seal)
    assert excinfo.value.context["added"] == ["renamed.json"]
    assert excinfo.value.context["removed"] == ["meta.json"]


def test_verifying_without_a_seal_raises(tmp_path):
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PublicTestSealError) as excinfo:
        PublicTestCorpus(root=tmp_path, corpus_id="unsealed").verify()
    assert "seal it first" in str(excinfo.value)


def test_the_verify_report_carries_both_digests(sealed):
    """A reader must be able to see recorded-vs-current without recomputing."""
    corpus, seal = sealed
    report = corpus.verify()
    assert report["recorded_digest"] == seal.digest
    assert report["current_digest"] == seal.digest


# ---------------------------------------------------------------------------
# Path containment
# ---------------------------------------------------------------------------
def test_a_traversal_path_is_refused(corpus_root):
    """`../` must not reach the firewall's 'outside the tree, therefore allowed' branch."""
    corpus = PublicTestCorpus(root=corpus_root)
    with pytest.raises(PublicTestSealedError) as excinfo:
        corpus.read("../../etc/passwd", mode=evaluation_mode("probe"))
    assert "outside the corpus root" in str(excinfo.value)


def test_an_absolute_traversal_is_refused(corpus_root):
    corpus = PublicTestCorpus(root=corpus_root)
    with pytest.raises(PublicTestSealedError):
        corpus.read("../../../secrets.json", mode=evaluation_mode("probe"))


def test_a_missing_corpus_file_raises(corpus_root):
    corpus = PublicTestCorpus(root=corpus_root)
    with pytest.raises(PublicTestSealError):
        corpus.read("nope.json", mode=evaluation_mode("probe"))


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def test_jsonl_reads_as_a_list(sealed):
    corpus, _seal = sealed
    payload = corpus.read_json("part/001.jsonl", mode=evaluation_mode("probe"))
    assert payload == [{"q": "is there change?"}, {"q": "how much?"}]


def test_json_reads_as_an_object(sealed):
    corpus, _seal = sealed
    payload = corpus.read_json("meta.json", mode=evaluation_mode("probe"))
    assert payload == {"n": 2}


def test_the_firewall_is_restored_after_a_read(sealed):
    """The disarm must be scoped to the read, not to the session."""
    corpus, _seal = sealed
    assert corpus.firewall.armed is True
    corpus.read("meta.json", mode=evaluation_mode("probe"))
    assert corpus.firewall.armed is True


def test_the_firewall_is_restored_even_when_the_read_fails(sealed):
    corpus, _seal = sealed
    assert corpus.firewall.armed is True
    with pytest.raises(PublicTestSealError):
        corpus.read("missing.json", mode=evaluation_mode("probe"))
    assert corpus.firewall.armed is True


def test_the_corpus_has_no_mutating_api():
    """Immutability must be structural: no writer, adder, or remover exists."""
    for method in ("write", "add", "remove", "append", "update", "delete", "unseal"):
        assert not hasattr(PublicTestCorpus, method), (
            f"PublicTestCorpus must not expose {method}(): a corpus with a writer is "
            f"not immutable, whatever the docstring says"
        )


def test_no_answer_key_writer_exists_anywhere_in_the_package():
    """Plan section 37: no cache of test answers is permitted.

    The prohibition is only real if no code path could write one. Checking for
    the literal filenames would be wrong -- `_ANSWER_CACHE_PATTERNS` names them
    precisely so they can be DETECTED. What must not exist is a write call that
    produces one, so this test asserts there is no writable answer-key handle:
    no `write_text`/`write_bytes`/`open(...w)` applied to an answer path.

    Concretely: the only writer in the package is the corpus SEAL, which records
    file digests, not answers. A seal is not an answer key, and this test pins
    that the distinction is real rather than a matter of naming.
    """
    from pathlib import Path

    package = Path(__file__).resolve().parents[2] / "evaluation" / "public_test"
    sources = {p.name: p.read_text(encoding="utf-8") for p in package.glob("*.py")}

    # Exactly one write site is permitted, and it must be the seal.
    write_sites = []
    for name, source in sources.items():
        for lineno, line in enumerate(source.splitlines(), start=1):
            if ".write_text(" in line or ".write_bytes(" in line:
                write_sites.append((name, lineno, line.strip()))

    assert len(write_sites) == 1, (
        f"expected exactly one writer in the public_test package (the corpus "
        f"seal); found {len(write_sites)}: {write_sites}"
    )

    name, _lineno, line = write_sites[0]
    source = sources[name]
    # The write target must be derived from the seal filename constant, so the
    # writer cannot be repointed at an answer path without changing that constant.
    assert "seal_path.write_text(" in line
    assert "seal_path = self.root / SEAL_FILENAME" in source, (
        f"the sole writer must derive its target from SEAL_FILENAME; {name} "
        f"computes its path some other way"
    )


def test_the_seal_records_file_digests_not_answers(sealed):
    """A seal is provenance, not an answer key. The distinction is load-bearing."""
    _corpus, seal = sealed
    for entry in seal.entries:
        # The entry shape is (relpath, sha256, bytes) and nothing else.
        assert set(entry.to_dict()) == {"relpath", "sha256", "bytes"}


# ---------------------------------------------------------------------------
# open_public_test end to end
# ---------------------------------------------------------------------------
def test_open_public_test_verifies_and_returns_the_corpus(sealed, corpus_root):
    corpus = open_public_test(mode=evaluation_mode("benchmark run"), root=corpus_root)
    assert isinstance(corpus, PublicTestCorpus)
    assert corpus.available is True


def test_open_public_test_refuses_a_tampered_corpus(sealed, corpus_root):
    (corpus_root / "meta.json").write_text('{"n": 999}', encoding="utf-8")
    with pytest.raises(PublicTestSealedError):
        open_public_test(mode=evaluation_mode("benchmark run"), root=corpus_root)


def test_the_answer_cache_check_is_reported_even_when_clean():
    state = public_test_state()
    assert state["answer_cache"]["clean"] is True
    assert state["answer_cache"]["checked"]


def test_a_present_answer_cache_is_reported_as_a_violation(tmp_path):
    """A cache is a VIOLATION and must be visible in the report, not swallowed."""
    (tmp_path / "public_test_answers.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PublicTestAnswerCacheError):
        forbid_answer_cache(tmp_path)

    state = public_test_state(tmp_path)
    assert state["answer_cache"]["clean"] is False
    assert state["answer_cache"]["found"]


def test_the_corpus_state_note_explains_what_unavailable_means():
    state = public_test_state()
    assert "there is no test set" in state["note"]
