"""STEP 5 — benchmark adapters: contract, registry, and the non-invention guard.

The tests that matter here are the ANTI-STUB ones: they assert that an adapter
which has no dataset refuses rather than returning synthetic samples. A test
suite that only checked the happy path would pass even if `load()` silently
generated fake data, which is precisely the failure mode this package exists to
prevent.
"""

from __future__ import annotations

import pytest

from evaluation.benchmark_adapters import (
    DECLARED_BENCHMARKS,
    AdapterError,
    AdapterNotRegisteredError,
    BenchmarkAdapter,
    BenchmarkNotAvailableError,
    BenchmarkSample,
    BenchmarkStatus,
    DeclaredBenchmark,
    SyntheticVQAAdapter,
    clear_registry,
    declared_benchmark_names,
    declared_inventory,
    describe_declared,
    get_adapter,
    inventory,
    predict_from_payload,
    register_adapter,
    registered_adapters,
    unregister_adapter,
    unregistered,
)


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Every test gets an empty registry and leaves one behind."""
    clear_registry()
    yield
    clear_registry()


# ---------------------------------------------------------------------------
# The anti-stub guard: an adapter with no corpus must REFUSE
# ---------------------------------------------------------------------------
class _NoCorpusAdapter(BenchmarkAdapter):
    """An adapter whose benchmark is genuinely absent.

    This is the shape a real adapter takes before its dataset is supplied. It
    must refuse, and it must refuse with the path in the message.
    """

    name = "absent_benchmark"

    def describe_corpus(self):
        from evaluation.benchmark_adapters import CorpusDescription

        return CorpusDescription(
            benchmark=self.name,
            available=False,
            root="/nowhere/absent_benchmark",
            reason="corpus root does not exist: /nowhere/absent_benchmark",
        )

    def load(self, *, split: str = "test"):
        # NO synthetic fallback. This is the line the whole package defends.
        raise BenchmarkNotAvailableError(
            f"corpus absent for {self.name}", context=self.describe_corpus().to_dict()
        )

    def metric_names(self):
        return ("accuracy",)


def test_an_adapter_without_a_corpus_refuses_rather_than_inventing_samples():
    """THE central test: absence must produce an error, not a sample list."""
    adapter = _NoCorpusAdapter()
    with pytest.raises(BenchmarkNotAvailableError) as excinfo:
        adapter.require_available()

    # The path must be IN the error, or the message is not actionable.
    assert "/nowhere/absent_benchmark" in str(excinfo.value)
    assert excinfo.value.context["available"] is False


def test_the_refusal_carries_a_machine_readable_corpus_description():
    adapter = _NoCorpusAdapter()
    description = adapter.describe_corpus()
    assert description.available is False
    assert description.reason is not None
    payload = description.to_dict()
    assert payload["root"] == "/nowhere/absent_benchmark"
    assert payload["n_files"] == 0


def test_the_runner_status_for_an_absent_corpus_is_not_run_not_zero():
    """An absent corpus yields NOT_RUN. It must never yield a 0.0 score."""
    from evaluation.runner import EvaluationRunner

    register_adapter(_NoCorpusAdapter())
    report = EvaluationRunner(run_id="t-absent").run(["absent_benchmark"])

    assert report["results"]["absent_benchmark"]["status"] == "NOT_RUN"
    assert report["results"]["absent_benchmark"]["metrics"] == {}
    assert report["real_metrics"] == {}
    assert report["summary"]["n_real"] == 0


def test_abstract_adapters_cannot_be_instantiated():
    """The base class provides no working implementation on purpose."""
    with pytest.raises(TypeError):
        BenchmarkAdapter()  # type: ignore[abstract]


def test_declaring_a_benchmark_without_naming_it_is_refused():
    """An unnamed adapter would register under '' and collide."""
    with pytest.raises(AdapterError):

        class _Unnamed(BenchmarkAdapter):
            def describe_corpus(self):
                raise NotImplementedError

            def load(self, *, split: str = "test"):
                raise NotImplementedError

            def metric_names(self):
                return ()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_the_registry_is_empty_by_default():
    """Nothing is registered out of the box: no dataset exists to be wrapped."""
    assert registered_adapters() == {}
    assert inventory()["registered"] == []
    assert inventory()["count"] == 0


def test_an_empty_registry_does_not_read_as_a_pass():
    """The inventory must say plainly that an empty registry proves nothing."""
    payload = inventory()
    assert "does NOT" in payload["note"]
    assert "passed" in payload["note"]


def test_registering_twice_is_refused_unless_deliberate():
    register_adapter(SyntheticVQAAdapter(n_samples=4))
    with pytest.raises(AdapterError) as excinfo:
        register_adapter(SyntheticVQAAdapter(n_samples=8))
    assert "already registered" in str(excinfo.value)

    register_adapter(SyntheticVQAAdapter(n_samples=8), replace=True)
    assert get_adapter("synthetic_vqa_fixture").n_samples == 8


def test_a_missing_adapter_raises_rather_than_returning_none():
    with pytest.raises(AdapterNotRegisteredError) as excinfo:
        get_adapter("does_not_exist")
    # The message must list what IS there, so a typo is diagnosable.
    assert "registered" in str(excinfo.value)


def test_unregistering_something_absent_raises():
    with pytest.raises(AdapterNotRegisteredError):
        unregister_adapter("never_registered")


def test_registered_adapters_returns_a_copy():
    """A caller must not be able to mutate the registry through the accessor."""
    register_adapter(SyntheticVQAAdapter(n_samples=4))
    snapshot = registered_adapters()
    snapshot.clear()
    assert registered_adapters() != {}


def test_unregistered_reports_which_names_have_no_adapter():
    assert unregistered(["levir_cd", "vrsbench"]) == ["levir_cd", "vrsbench"]
    register_adapter(SyntheticVQAAdapter(n_samples=4))
    assert unregistered(["synthetic_vqa_fixture", "levir_cd"]) == ["levir_cd"]


# ---------------------------------------------------------------------------
# Declared benchmarks
# ---------------------------------------------------------------------------
def test_the_declared_benchmarks_are_the_ones_the_plan_names():
    names = declared_benchmark_names()
    assert "levir_cd" in names
    assert "vrsbench" in names
    assert "bigearthnet_s1" in names
    assert len(names) == len(set(names)), "declared names must be unique"


def test_a_declaration_cannot_emit_a_score():
    """Declarations are data. There is no evaluation method on them."""
    benchmark = DECLARED_BENCHMARKS[0]
    for method in ("load", "score", "evaluate", "predict"):
        assert not hasattr(benchmark, method), (
            f"DeclaredBenchmark must not expose {method}(): a declaration that can "
            f"be evaluated is a fabricated benchmark waiting to happen"
        )


def test_declared_benchmarks_report_themselves_as_unregistered():
    payload = declared_inventory()
    for entry in payload["benchmarks"].values():
        assert entry["adapter_registered"] is False


def test_citations_are_marked_unverified_here():
    """A copied citation must not look like a checked one."""
    payload = declared_inventory()
    for entry in payload["benchmarks"].values():
        assert entry["citation_verified_here"] is False
        assert entry["citation"]


def test_declared_availability_is_measured_not_assumed():
    payload = declared_inventory()
    for name, entry in payload["benchmarks"].items():
        corpus = entry["corpus"]
        assert corpus["available"] is bool(corpus["n_files"])
        assert name in payload["declared"]


def test_an_absent_declared_benchmark_reports_the_path_and_a_reason():
    absent = [
        e for e in declared_inventory()["benchmarks"].values()
        if not e["corpus"]["available"]
    ]
    assert absent, "at least one declared benchmark has no corpus in this repo"
    for entry in absent:
        assert entry["corpus"]["reason"]
        assert entry["corpus"]["root"]


def test_describe_declared_counts_files_when_the_root_exists(tmp_path):
    """A populated root is reported as present, with real counts."""
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "c.json").write_text("{}", encoding="utf-8")

    benchmark = DeclaredBenchmark(
        name="fixture_declared",
        root=str(tmp_path),
        task="probe",
        expected_metrics=("accuracy",),
        citation="N/A",
        plan_reference="test",
    )
    description = describe_declared(benchmark)
    assert description.available is True
    assert description.n_files == 2
    assert description.bytes == 4


def test_an_empty_existing_root_is_reported_unavailable_with_a_reason(tmp_path):
    benchmark = DeclaredBenchmark(
        name="empty_declared",
        root=str(tmp_path),
        task="probe",
        expected_metrics=("accuracy",),
        citation="N/A",
        plan_reference="test",
    )
    description = describe_declared(benchmark)
    assert description.available is False
    assert "empty" in (description.reason or "")


# ---------------------------------------------------------------------------
# The fixture adapter — honest by construction
# ---------------------------------------------------------------------------
def test_the_fixture_adapter_declares_itself_non_official():
    """This is what keeps its scores out of real_metrics."""
    assert SyntheticVQAAdapter().is_official() is False


def test_the_fixture_adapter_is_available_but_flagged():
    """It IS available (it generates), and says so as generated material."""
    adapter = SyntheticVQAAdapter(n_samples=4)
    description = adapter.describe_corpus()
    assert description.available is True
    assert "generated" in (description.note or "").lower()


def test_the_fixture_is_deterministic():
    first = SyntheticVQAAdapter(n_samples=12, seed=99).load()
    second = SyntheticVQAAdapter(n_samples=12, seed=99).load()
    assert [s.sample_id for s in first] == [s.sample_id for s in second]
    assert [s.payload for s in first] == [s.payload for s in second]
    assert [s.expected for s in first] == [s.expected for s in second]


def test_a_different_seed_produces_different_samples():
    a = SyntheticVQAAdapter(n_samples=16, seed=1).load()
    b = SyntheticVQAAdapter(n_samples=16, seed=2).load()
    assert [s.payload for s in a] != [s.payload for s in b]


def test_the_fixture_hits_its_target_accuracy_approximately():
    """The fixture's construction must actually work, or it tests nothing."""
    samples = SyntheticVQAAdapter(n_samples=400, seed=5, accuracy=0.8).load()
    correct = sum(1 for s in samples if s.meta["correct_by_construction"])
    assert 0.72 <= correct / len(samples) <= 0.88


def test_the_fixture_scores_through_the_real_metric_helper():
    from evaluation.metrics.vqa import answer_accuracy

    samples = SyntheticVQAAdapter(n_samples=64, seed=3, accuracy=0.75).load()
    preds = [predict_from_payload(s) for s in samples]
    golds = [s.expected for s in samples]
    accuracy = answer_accuracy(preds, golds)
    assert 0.6 <= accuracy <= 0.9


def test_the_fixture_rejects_impossible_parameters():
    with pytest.raises(BenchmarkNotAvailableError):
        SyntheticVQAAdapter(n_samples=0)
    with pytest.raises(BenchmarkNotAvailableError):
        SyntheticVQAAdapter(accuracy=1.5)


def test_a_malformed_fixture_payload_is_rejected_not_guessed():
    sample = BenchmarkSample(
        sample_id="bad", payload={"logits": [1.0], "answers": ["yes", "no"]}
    )
    with pytest.raises(ValueError):
        predict_from_payload(sample)


def test_a_sample_without_an_id_is_rejected():
    with pytest.raises(AdapterError):
        BenchmarkSample(sample_id="")


def test_the_fixture_adapter_serialises_its_disclaimer():
    payload = SyntheticVQAAdapter(n_samples=4).to_dict()
    assert payload["fixture"]["n_samples"] == 4
    assert "never" in payload["fixture"]["disclaimer"].lower()
    assert payload["official"] is False


# ---------------------------------------------------------------------------
# Status vocabulary
# ---------------------------------------------------------------------------
def test_only_real_is_scored():
    """The word 'real' is the only one that may back a benchmark claim."""
    scored = {s for s in BenchmarkStatus if s.is_scored}
    assert scored == {BenchmarkStatus.REAL}


def test_the_status_enum_serialises_as_its_name():
    """A report must carry the readable word, not an ordinal."""
    assert BenchmarkStatus.NOT_RUN.value == "NOT_RUN"
    assert str(BenchmarkStatus.FIXTURE) == "BenchmarkStatus.FIXTURE"
    assert BenchmarkStatus.FIXTURE == "FIXTURE"


def test_every_status_has_a_legend_entry_in_the_report():
    from evaluation.runner import EvaluationRunner, _STATUS_LEGEND

    register_adapter(SyntheticVQAAdapter(n_samples=4))
    report = EvaluationRunner(run_id="t-legend").run(["synthetic_vqa_fixture"])
    legend = report["metric_status_legend"]
    assert set(legend) == {s.value for s in BenchmarkStatus}
    assert set(legend) == {s.value for s in _STATUS_LEGEND}
    assert "never quote" in legend["FIXTURE"].lower()
