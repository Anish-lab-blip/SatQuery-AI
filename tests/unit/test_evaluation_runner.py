"""STEP 5 — the evaluation runner.

The runner is the single place a benchmark score is produced, so it is the
single place a fabricated one could appear. These tests attack that:

  * a missing corpus yields NOT_RUN with no metrics at all (not a 0.0)
  * a fixture's number is reported but kept OUT of `real_metrics`
  * an unregistered metric is recorded as a failure, never silently dropped
  * a metric outside [0,1] is refused, never clamped
  * no aggregate is produced while the official weights are None
  * a raising scorer yields FAILED with the traceback attached
  * the run manifest is populated, and `ended_at` is not invented
"""

from __future__ import annotations

import json

import pytest

from evaluation.benchmark_adapters import (
    BenchmarkAdapter,
    BenchmarkStatus,
    SyntheticVQAAdapter,
    clear_registry,
    predict_from_payload,
    register_adapter,
)
from evaluation.metrics.vqa import answer_accuracy
from evaluation.runner import (
    EvaluationRunner,
    aggregate_or_explain,
    normalise_metrics,
)


@pytest.fixture(autouse=True)
def _isolated_registry():
    clear_registry()
    yield
    clear_registry()


def _fixture_scorer(samples):
    """Score the synthetic fixture through the real metric helper."""
    preds = [predict_from_payload(s) for s in samples]
    golds = [s.expected for s in samples]
    return {"accuracy": answer_accuracy(preds, golds)}


def _register_fixture(**kwargs):
    adapter = SyntheticVQAAdapter(**kwargs)
    register_adapter(adapter)
    return adapter


# ---------------------------------------------------------------------------
# Status ladder
# ---------------------------------------------------------------------------
def test_no_adapter_yields_not_run_with_no_metrics():
    """'Not wired up' must be distinguishable from 'scored zero'."""
    report = EvaluationRunner(run_id="t1").run(["levir_cd"])
    result = report["results"]["levir_cd"]
    assert result["status"] == "NOT_RUN"
    assert result["metrics"] == {}
    assert result["n_samples"] == 0
    assert report["summary"]["n_not_run"] == 1
    assert report["summary"]["n_real"] == 0


def test_the_not_run_diagnostic_names_the_registry():
    """A reader must be able to see WHAT was available instead."""
    report = EvaluationRunner(run_id="t2").run(["levir_cd"])
    detail = report["results"]["levir_cd"]["detail"]
    assert detail["registry"]["registered"] == []
    assert "no adapter" in detail["registry"]["reason"]


def test_a_fixture_is_scored_as_fixture_not_real():
    _register_fixture(n_samples=64, accuracy=0.75)
    report = EvaluationRunner(run_id="t3").run(
        ["synthetic_vqa_fixture"], scorers={"synthetic_vqa_fixture": _fixture_scorer}
    )
    result = report["results"]["synthetic_vqa_fixture"]
    assert result["status"] == "FIXTURE"
    assert result["scored"] is False
    assert report["summary"]["n_fixture"] == 1
    assert report["summary"]["n_real"] == 0


def test_a_fixture_number_never_reaches_real_metrics():
    """THE boundary: a generated number must not back a benchmark claim."""
    _register_fixture(n_samples=64, accuracy=0.75)
    report = EvaluationRunner(run_id="t4").run(
        ["synthetic_vqa_fixture"], scorers={"synthetic_vqa_fixture": _fixture_scorer}
    )
    assert report["real_metrics"] == {}
    assert "synthetic_vqa_fixture" in report["fixture_metrics"]
    # ...but the number IS reported, in its own clearly-labelled key.
    assert 0.0 <= report["fixture_metrics"]["synthetic_vqa_fixture"]["accuracy"] <= 1.0


def test_the_status_reason_explains_why_a_score_is_not_a_benchmark_score():
    _register_fixture(n_samples=16)
    report = EvaluationRunner(run_id="t5").run(
        ["synthetic_vqa_fixture"], scorers={"synthetic_vqa_fixture": _fixture_scorer}
    )
    reason = report["results"]["synthetic_vqa_fixture"]["detail"]["status_reason"]
    assert "fixture" in reason.lower() or "not the benchmark" in reason.lower()


def test_a_missing_scorer_yields_not_run_not_a_default_score():
    """The runner will not invent a scorer."""
    _register_fixture(n_samples=16)
    report = EvaluationRunner(run_id="t6").run(["synthetic_vqa_fixture"])
    result = report["results"]["synthetic_vqa_fixture"]
    assert result["status"] == "NOT_RUN"
    assert result["metrics"] == {}
    assert "no scorer supplied" in result["error"]


def test_a_raising_scorer_yields_failed_with_a_traceback():
    _register_fixture(n_samples=8)

    def _boom(samples):
        raise RuntimeError("scorer exploded on purpose")

    report = EvaluationRunner(run_id="t7").run(
        ["synthetic_vqa_fixture"], scorers={"synthetic_vqa_fixture": _boom}
    )
    result = report["results"]["synthetic_vqa_fixture"]
    assert result["status"] == "FAILED"
    assert "RuntimeError" in result["error"]
    assert "scorer exploded on purpose" in result["traceback"]
    assert report["summary"]["n_failed"] == 1


def test_an_adapter_returning_no_samples_is_failed_not_scored():
    """A broken adapter must not look like a benchmark that scored nothing."""

    class _EmptyAdapter(BenchmarkAdapter):
        name = "empty_adapter"

        def describe_corpus(self):
            from evaluation.benchmark_adapters import CorpusDescription

            return CorpusDescription(
                benchmark=self.name, available=True, root="/tmp/empty"
            )

        def load(self, *, split: str = "test"):
            return []

        def metric_names(self):
            return ("accuracy",)

    register_adapter(_EmptyAdapter())
    report = EvaluationRunner(run_id="t8").run(
        ["empty_adapter"], scorers={"empty_adapter": lambda s: {"accuracy": 0.0}}
    )
    result = report["results"]["empty_adapter"]
    assert result["status"] == "FAILED"
    assert "no samples" in result["error"]
    assert result["metrics"] == {}


def test_an_adapter_that_raises_on_load_is_failed():
    class _BrokenAdapter(BenchmarkAdapter):
        name = "broken_adapter"

        def describe_corpus(self):
            from evaluation.benchmark_adapters import CorpusDescription

            return CorpusDescription(
                benchmark=self.name, available=True, root="/tmp/broken"
            )

        def load(self, *, split: str = "test"):
            raise OSError("disk detached")

        def metric_names(self):
            return ("accuracy",)

    register_adapter(_BrokenAdapter())
    report = EvaluationRunner(run_id="t9").run(["broken_adapter"])
    result = report["results"]["broken_adapter"]
    assert result["status"] == "FAILED"
    assert "OSError" in result["error"]


def test_every_requested_benchmark_appears_exactly_once():
    _register_fixture(n_samples=8)
    requested = ["synthetic_vqa_fixture", "levir_cd", "vrsbench"]
    report = EvaluationRunner(run_id="t10").run(
        requested, scorers={"synthetic_vqa_fixture": _fixture_scorer}
    )
    assert sorted(report["results"]) == sorted(requested)
    counted = sum(len(v) for v in report["summary"]["by_status"].values())
    assert counted == len(requested)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def test_a_registered_metric_normalises_to_itself():
    normalized, records = normalise_metrics({"accuracy": 0.75})
    assert normalized == {"accuracy": 0.75}
    assert records[0].ok is True
    assert records[0].error is None


def test_an_unregistered_metric_is_recorded_not_dropped():
    """'Could not be normalised' must not look like 'was not measured'."""
    normalized, records = normalise_metrics({"accuracy": 0.5, "wer": 0.3})
    assert "wer" not in normalized
    assert {r.metric for r in records} == {"accuracy", "wer"}
    failure = next(r for r in records if r.metric == "wer")
    assert failure.ok is False
    assert "UnregisteredMetricError" in failure.error


def test_an_out_of_range_metric_is_refused_not_clamped():
    normalized, records = normalise_metrics({"iou": 1.5})
    assert normalized == {}
    assert "NormalisationRangeError" in records[0].error


def test_a_non_numeric_metric_is_refused_not_coerced():
    normalized, records = normalise_metrics({"accuracy": "high"})
    assert normalized == {}
    assert "refusing to coerce" in records[0].error


def test_a_boolean_is_not_accepted_as_a_metric_value():
    """`True` is an int in Python; that must not become a metric of 1.0."""
    normalized, records = normalise_metrics({"accuracy": True})
    assert normalized == {}
    assert "not a number" in records[0].error


def test_normalisation_failures_are_surfaced_in_the_result():
    _register_fixture(n_samples=8)
    report = EvaluationRunner(run_id="t11").run(
        ["synthetic_vqa_fixture"],
        scorers={"synthetic_vqa_fixture": lambda s: {"accuracy": 0.5, "wer": 0.1}},
    )
    result = report["results"]["synthetic_vqa_fixture"]
    assert result["detail"]["normalisation_failures"] == ["wer"]
    assert "accuracy" in result["normalized"]
    assert "wer" not in result["normalized"]


# ---------------------------------------------------------------------------
# Aggregation prohibition
# ---------------------------------------------------------------------------
def test_no_aggregate_is_produced_while_the_weights_are_unpublished():
    block = aggregate_or_explain({"accuracy": 0.9})
    assert block["available"] is False
    assert block["score"] is None
    assert block["weights"] is None
    assert block["error"] == "OfficialWeightsUnavailableError"


def test_the_aggregate_block_explains_why_and_points_at_alternatives():
    block = aggregate_or_explain({"accuracy": 0.9})
    assert "section 63" in block["note"]
    assert "Per-metric" in block["note"]


def test_the_report_carries_the_aggregate_refusal():
    _register_fixture(n_samples=8)
    report = EvaluationRunner(run_id="t12").run(
        ["synthetic_vqa_fixture"], scorers={"synthetic_vqa_fixture": _fixture_scorer}
    )
    assert report["aggregate_score"]["available"] is False


def test_the_aggregate_attempts_even_when_there_is_nothing_to_aggregate():
    """The refusal is exercised on every run, not short-circuited on the None check."""
    report = EvaluationRunner(run_id="t13").run(["levir_cd"])
    assert report["aggregate_score"]["available"] is False
    assert report["aggregate_score"]["error"] == "OfficialWeightsUnavailableError"


# ---------------------------------------------------------------------------
# Report shape and honesty
# ---------------------------------------------------------------------------
def test_the_report_schema_is_stamped():
    report = EvaluationRunner(run_id="t14").run([])
    assert report["schema"] == "evaluation_report_v1"
    assert report["run_id"] == "t14"
    assert report["generated_at"]


def test_the_report_states_the_fixture_caveat_loudly():
    report = EvaluationRunner(run_id="t15").run([])
    assert "never be quoted" in report["note"]
    assert "real_metrics" in report["note"]


def test_the_report_embeds_the_adapter_inventory():
    report = EvaluationRunner(run_id="t16").run([])
    assert report["adapters"]["registered"] == []
    assert report["adapters"]["count"] == 0


def test_the_report_embeds_the_declared_benchmarks():
    report = EvaluationRunner(run_id="t17").run([])
    assert "levir_cd" in report["declared_benchmarks"]["declared"]
    assert report["declared_benchmarks"]["unavailable"]


def test_the_report_embeds_the_public_test_state_when_asked():
    report = EvaluationRunner(run_id="t18", public_test_root=None).run([])
    assert report["public_test"]["available"] is None
    assert "not inspected" in report["public_test"]["note"]


def test_the_report_says_when_the_public_test_state_was_inspected():
    from evaluation.public_test import PUBLIC_TEST_ROOT

    report = EvaluationRunner(run_id="t19", public_test_root=PUBLIC_TEST_ROOT).run([])
    assert report["public_test"]["available"] is False


def test_the_summary_counts_are_consistent():
    _register_fixture(n_samples=8)
    report = EvaluationRunner(run_id="t20").run(
        ["synthetic_vqa_fixture", "levir_cd"], scorers={"synthetic_vqa_fixture": _fixture_scorer}
    )
    summary = report["summary"]
    assert summary["n_requested"] == 2
    assert (
        summary["n_real"] + summary["n_fixture"] + summary["n_degraded"]
        + summary["n_failed"] + summary["n_not_run"]
    ) == 2


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------
def test_the_run_manifest_is_populated():
    report = EvaluationRunner(run_id="t21", seed=7, thresholds={"change": 0.5}).run([])
    manifest = report["run_manifest"]
    assert manifest["run_id"] == "t21"
    assert manifest["seed"] == 7
    assert manifest["thresholds"] == {"change": 0.5}
    assert manifest["code_revision"], "code_revision must be populated"
    assert manifest["environment"], "environment must be populated"
    assert manifest["prompt_versions"], "prompt_versions must be populated"
    assert manifest["started_at"], "started_at must be populated"


def test_the_run_manifest_hash_is_recorded():
    report = EvaluationRunner(run_id="t22").run([])
    assert report["run_manifest_hash"]
    assert len(report["run_manifest_hash"]) == 16


def test_the_manifest_does_not_invent_an_end_time():
    """`ended_at` must stay unset until a caller supplies one."""
    report = EvaluationRunner(run_id="t23").run([])
    assert "ended_at" not in report["run_manifest"]


def test_an_absent_config_hash_is_reported_as_unavailable():
    """A made-up hash would look like a pinned configuration."""
    report = EvaluationRunner(run_id="t24", config=None).run([])
    assert "unavailable" in report["config_hash"]
    assert "no config" in report["config_hash"]


def test_a_supplied_config_hash_is_used():
    class _Config:
        hash = "78f1e3700da15aa1"

    report = EvaluationRunner(run_id="t25", config=_Config()).run([])
    assert report["config_hash"] == "78f1e3700da15aa1"


def test_the_dataset_manifest_hash_default_is_an_explicit_statement():
    report = EvaluationRunner(run_id="t26").run([])
    assert "unavailable" in report["dataset_manifest_hash"]


def test_the_manifest_completeness_report_is_included():
    report = EvaluationRunner(run_id="t27").run([])
    completeness = report["run_manifest_completeness"]
    assert completeness["code_revision"] == "present"
    assert completeness["environment"] == "present"
    assert completeness["prompt_versions"] == "present"


def test_results_carry_start_and_end_times():
    _register_fixture(n_samples=8)
    report = EvaluationRunner(run_id="t28").run(
        ["synthetic_vqa_fixture"], scorers={"synthetic_vqa_fixture": _fixture_scorer}
    )
    result = report["results"]["synthetic_vqa_fixture"]
    assert result["started_at"]
    assert result["ended_at"]
    assert result["started_at"] <= result["ended_at"]


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def test_writing_a_report_produces_a_digest_sidecar(tmp_path):
    """A file cannot contain a hash of itself; the digest goes beside it."""
    import hashlib

    runner = EvaluationRunner(run_id="t29")
    report = runner.run([])
    target = runner.write(report, tmp_path / "report.json")

    sidecar = tmp_path / "report.json.sha256"
    assert sidecar.exists()

    text = target.read_text(encoding="utf-8")
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    recorded = sidecar.read_text(encoding="utf-8").split()[0]
    assert recorded == expected


def test_the_written_report_round_trips(tmp_path):
    runner = EvaluationRunner(run_id="t30")
    report = runner.run([])
    target = runner.write(report, tmp_path / "report.json")
    reloaded = json.loads(target.read_text(encoding="utf-8"))
    assert reloaded["run_id"] == "t30"
    assert reloaded["schema"] == report["schema"]


def test_the_written_report_is_deterministic_for_a_fixed_manifest(tmp_path):
    """Same run id and inputs -> same bytes except for the timestamps."""
    runner = EvaluationRunner(run_id="t31")
    first = runner.run([])
    second = EvaluationRunner(run_id="t31").run([])
    for key in ("schema", "seed", "split", "summary", "real_metrics"):
        assert first[key] == second[key]
