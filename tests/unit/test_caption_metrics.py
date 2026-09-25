"""Caption metrics (ruling R-16): BLEU, ROUGE-L, CIDEr, BERTScore.

WHAT THIS SUITE IS FOR
----------------------
R-16 asks for four metrics built on established implementations, with pinned
versions, isolated optional dependencies, and explicit degradation instead of a
fabricated score. Each of those is a mechanism, and each mechanism gets a test:

  * **real execution** -- BLEU and ROUGE-L are computed for real, and the
    assertions are on values that a wrong implementation would not produce
    (identical input must score BLEU 100 / ROUGE-L 1.0).
  * **missing dependency** -- the availability probe is stubbed to report a
    missing package, and the metric must then report `available: False` with a
    reason and an install hint, and `score_captions` must RAISE rather than
    substitute a different metric.
  * **model weights** -- the HuggingFace cache is redirected to a `tmp_path` and
    populated with a valid entry, an empty one, and a malformed one, so the
    model requirement's verdict is tested by construction rather than by whatever
    happens to be cached on the machine running the suite.
  * **malformed input** -- mismatched lengths and unknown metric names raise.
  * **empty input** -- empty predictions with empty references is a degenerate
    case that must not divide by zero or invent a value.
  * **determinism** -- the same input must produce the same numbers, because a
    metric that drifts cannot be pinned in a report.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No assertion that a particular metric is available on the machine running the
suite. Availability is an environment fact, not a property of the code, and a
test that asserted it would be a test that fails on someone else's laptop. The
two metrics that ARE available here (BLEU, ROUGE-L) are tested for real; the two
that are not (CIDEr needs a JVM, BERTScore needs weights) are tested through the
degradation path, which is the behaviour R-16 actually specifies.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.metrics import caption
from evaluation.metrics.caption import (
    CAPTION_METRICS,
    PINNED_VERSIONS,
    CaptionMetricError,
    CaptionMetricUnavailableError,
    caption_capability,
    caption_metric_info,
    caption_metric_info_or_none,
    caption_metrics_summary,
    score_captions,
)

REFS = ["a large building near the river", "the red roof of a house"]
PREDS = ["a large building near a river", "the roof of a red house"]


def _available(metric: str) -> bool:
    return caption_metric_info(metric, probe_network=False)["available"]


needs_bleu = pytest.mark.skipif(not _available("bleu"), reason="sacrebleu not installed")
needs_rouge = pytest.mark.skipif(
    not _available("rouge_l"), reason="rouge_score not installed"
)


# ---------------------------------------------------------------------------
# Import discipline
# ---------------------------------------------------------------------------
class TestImportDiscipline:
    def test_the_module_imports_no_third_party_package_at_import_time(self):
        """R-16 requires the module to be importable without any of the four.

        Checked by reading the module's top-level imports rather than by trusting
        a comment: a top-level `import torch` would poison the model-free paths
        the API's CPU-only contract depends on.
        """
        import ast

        source = Path(caption.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        top_level: set[str] = set()
        for node in tree.body:  # only module-level statements
            if isinstance(node, ast.Import):
                top_level.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level.add(node.module.split(".")[0])

        third_party = {"sacrebleu", "rouge_score", "pycocoevalcap", "bert_score", "torch"}
        assert not (top_level & third_party), (
            f"third-party imports at module level: {sorted(top_level & third_party)}; "
            f"they must be inside the functions that use them"
        )

    def test_the_capability_probe_does_not_import_the_metric_packages(self):
        """`find_spec` locates a module without executing it, which is the point."""
        import importlib.util

        # If the probe had imported bert_score, torch would be in sys.modules.
        import sys

        before = set(sys.modules)
        caption_capability(probe_network=False)
        after = set(sys.modules)
        newly = {m for m in (after - before) if m.split(".")[0] in ("torch", "bert_score")}
        assert not newly, f"the availability probe imported {newly}"
        assert importlib.util.find_spec is not None


# ---------------------------------------------------------------------------
# Availability reporting
# ---------------------------------------------------------------------------
class TestAvailability:
    def test_each_of_the_four_is_reported_separately(self):
        """The whole point of R-16: one flag would destroy the distinction."""
        capability = caption_capability(probe_network=False)
        assert set(capability) == set(CAPTION_METRICS)
        for name, info in capability.items():
            assert isinstance(info["available"], bool)
            assert info["implementation"]
            assert info["entry_point"]

    def test_the_capability_report_names_the_implementation_and_version(self):
        info = caption_metric_info("bleu", probe_network=False)
        assert info["implementation"] == "sacrebleu"
        assert info["pinned"] == "sacrebleu==2.4.3"
        assert info["pinned_version"] == "2.4.3"

    def test_an_unavailable_metric_gives_a_reason_and_an_install_hint(self):
        """A bare `False` is not actionable; the message must name the remedy."""
        for name in CAPTION_METRICS:
            info = caption_metric_info(name, probe_network=False)
            if not info["available"]:
                assert info["reason"], f"{name} is unavailable with no reason"
                assert info["install_hint"]
                assert info["missing_extra_requirements"]

    def test_the_version_pin_comparison_is_exact_not_a_substring(self):
        """`"1.2" in "pkg==1.20"` would be True; equality is not."""
        assert caption._pin_version("sacrebleu==2.4.3") == "2.4.3"
        assert caption._pin_version(None) is None
        assert caption._pin_version("no-pin-here") is None

    def test_a_pinned_version_mismatch_would_be_flagged(self, monkeypatch):
        monkeypatch.setattr(caption, "_installed_version", lambda dist: "9.9.9")
        info = caption_metric_info("bleu", probe_network=False)
        assert info["version_matches_pin"] is False

    def test_an_unknown_metric_name_raises(self):
        with pytest.raises(CaptionMetricError):
            caption_metric_info("wer")

    def test_the_or_none_variant_returns_none_instead_of_raising(self):
        assert caption_metric_info_or_none("wer") is None
        assert caption_metric_info_or_none("bleu", probe_network=False) is not None

    def test_the_summary_lists_available_and_unavailable_separately(self):
        summary = caption_metrics_summary()
        assert set(summary["available"]) | set(summary["unavailable"]) == set(
            CAPTION_METRICS
        )
        assert summary["n_available"] + len(summary["unavailable"]) == summary["n_total"]
        assert "R-16" in summary["ruling"]


# ---------------------------------------------------------------------------
# Declared requirements are VERIFIED
# ---------------------------------------------------------------------------
class TestRequirementVerification:
    def test_an_unrecognised_requirement_counts_as_unsatisfied(self):
        """Unverified must not be treated as fine -- that is how a report overstates."""
        reason = caption._check_extra_requirement("some-future-requirement")
        assert reason is not None
        assert "unrecognised" in reason

    def test_java_is_checked(self, monkeypatch):
        monkeypatch.setattr(caption, "_has_java", lambda: False)
        assert "java" in (caption._check_extra_requirement("java") or "")
        monkeypatch.setattr(caption, "_has_java", lambda: True)
        assert caption._check_extra_requirement("java") is None

    def test_torch_is_checked_by_importability(self, monkeypatch):
        monkeypatch.setattr(caption, "_importable", lambda m: (False, "stub"))
        assert caption._check_extra_requirement("torch") is not None
        monkeypatch.setattr(caption, "_importable", lambda m: (True, None))
        assert caption._check_extra_requirement("torch") is None

    def test_a_valid_cache_entry_satisfies_the_model_requirement(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(caption, "_hf_hub_dir", lambda: tmp_path / "hub")
        snap = tmp_path / "hub" / "models--acme--model" / "snapshots" / "rev1"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text(json.dumps({"model_type": "x"}), encoding="utf-8")
        assert caption._cached_model_valid("acme/model") is True
        assert caption._check_extra_requirement(
            "cached-model-weights", model_repo="acme/model"
        ) is None

    def test_a_zero_byte_cache_entry_is_a_defect_and_blocks_the_requirement(
        self, tmp_path, monkeypatch
    ):
        """The measured real case: a 0-byte config.json shadows any download."""
        monkeypatch.setattr(caption, "_hf_hub_dir", lambda: tmp_path / "hub")
        snap = tmp_path / "hub" / "models--acme--model" / "snapshots" / "rev1"
        snap.mkdir(parents=True)
        (snap / "config.json").write_bytes(b"")

        assert caption._cached_model_valid("acme/model") is False
        defect = caption._cached_model_defect("acme/model")
        assert defect is not None and "0 bytes" in defect
        reason = caption._check_extra_requirement(
            "cached-model-weights", model_repo="acme/model"
        )
        assert reason is not None and "blocked" in reason

    def test_a_malformed_cache_entry_is_a_defect(self, tmp_path, monkeypatch):
        monkeypatch.setattr(caption, "_hf_hub_dir", lambda: tmp_path / "hub")
        snap = tmp_path / "hub" / "models--acme--model" / "snapshots" / "rev1"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("<html>not json</html>", encoding="utf-8")
        assert caption._cached_model_valid("acme/model") is False
        assert "not valid JSON" in (caption._cached_model_defect("acme/model") or "")

    def test_an_absent_cache_entry_is_not_a_defect_just_an_absence(
        self, tmp_path, monkeypatch
    ):
        """`not cached` and `cached but broken` need different actions."""
        monkeypatch.setattr(caption, "_hf_hub_dir", lambda: tmp_path / "hub")
        assert caption._cached_model_valid("acme/model") is False
        assert caption._cached_model_defect("acme/model") is None
        reason = caption._check_extra_requirement(
            "cached-model-weights", model_repo="acme/model"
        )
        assert reason is not None
        assert "not present" in reason

    def test_the_legacy_requirement_name_is_still_recognised(self, tmp_path, monkeypatch):
        """A spec carrying the old name must not read as an unrecognised requirement."""
        monkeypatch.setattr(caption, "_hf_hub_dir", lambda: tmp_path / "hub")
        snap = tmp_path / "hub" / "models--acme--model" / "snapshots" / "rev1"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}", encoding="utf-8")
        assert (
            caption._check_extra_requirement(
                "network-or-cached-model", model_repo="acme/model"
            )
            is None
        )

    def test_the_network_probe_does_not_decide_availability(
        self, tmp_path, monkeypatch
    ):
        """A reachable hub is NOT evidence the metric can run -- measured."""
        monkeypatch.setattr(caption, "_hf_hub_dir", lambda: tmp_path / "hub")
        monkeypatch.setattr(caption, "_network_reachable", lambda *a, **k: True)
        monkeypatch.setattr(caption, "_hub_serves_config", lambda *a, **k: True)
        # Hub is reachable and serving, but nothing is cached.
        assert (
            caption._check_extra_requirement(
                "cached-model-weights", model_repo="acme/model"
            )
            is not None
        )
        diagnostics = caption.model_obtainability_diagnostics("acme/model")
        assert diagnostics["hub_reachable"] is True
        assert diagnostics["cached_valid"] is False

    def test_diagnostics_say_they_do_not_decide_availability(self, tmp_path, monkeypatch):
        monkeypatch.setattr(caption, "_hf_hub_dir", lambda: tmp_path / "hub")
        diagnostics = caption.model_obtainability_diagnostics(
            "acme/model", probe_network=False
        )
        assert "Diagnostics only" in diagnostics["note"]


# ---------------------------------------------------------------------------
# Missing dependency -> explicit degradation, never a fake score
# ---------------------------------------------------------------------------
class TestMissingDependency:
    @pytest.mark.parametrize("metric", list(CAPTION_METRICS))
    def test_a_missing_package_is_reported_not_substituted(self, metric, monkeypatch):
        monkeypatch.setattr(caption, "_importable", lambda m: (False, "stub: absent"))
        monkeypatch.setattr(caption, "_installed_version", lambda d: None)
        info = caption_metric_info(metric, probe_network=False)
        assert info["available"] is False
        assert "not installed" in info["reason"]
        assert info["install_hint"]

    @pytest.mark.parametrize("metric", list(CAPTION_METRICS))
    def test_scoring_a_missing_metric_raises_rather_than_falling_back(
        self, metric, monkeypatch
    ):
        monkeypatch.setattr(caption, "_importable", lambda m: (False, "stub: absent"))
        monkeypatch.setattr(caption, "_installed_version", lambda d: None)
        with pytest.raises(CaptionMetricUnavailableError) as excinfo:
            score_captions(PREDS, REFS, metrics=(metric,))
        message = str(excinfo.value)
        assert "No substitute metric is returned" in message

    def test_a_metric_that_is_available_is_unaffected_by_anothers_absence(
        self, monkeypatch
    ):
        """Isolation: stubbing one metric's package must not disable the others."""
        real_importable = caption._importable

        monkeypatch.setattr(
            caption,
            "_importable",
            lambda m: (False, "stub") if m == "sacrebleu" else real_importable(m),
        )
        assert caption_metric_info("bleu", probe_network=False)["available"] is False
        assert caption_metric_info("rouge_l", probe_network=False)["available"] is True


# ---------------------------------------------------------------------------
# Real execution
# ---------------------------------------------------------------------------
class TestRealExecution:
    @needs_bleu
    def test_bleu_scores_identical_input_at_one_hundred(self):
        """A wrong implementation would not produce exactly 100 here."""
        out = score_captions(REFS, REFS, metrics=("bleu",))
        assert out["bleu"]["value"] == pytest.approx(100.0)
        assert out["bleu"]["implementation"] == "sacrebleu"

    @needs_rouge
    def test_rouge_l_scores_identical_input_at_one(self):
        out = score_captions(REFS, REFS, metrics=("rouge_l",))
        assert out["rouge_l"]["value"] == pytest.approx(1.0)

    @needs_bleu
    def test_bleu_is_strictly_between_zero_and_one_hundred_for_a_near_match(self):
        out = score_captions(PREDS, REFS, metrics=("bleu",))
        assert 0.0 < out["bleu"]["value"] < 100.0

    @needs_rouge
    def test_rouge_l_is_strictly_between_zero_and_one_for_a_near_match(self):
        out = score_captions(PREDS, REFS, metrics=("rouge_l",))
        assert 0.0 < out["rouge_l"]["value"] < 1.0

    @needs_bleu
    def test_a_worse_prediction_scores_no_better(self):
        """Monotonicity: a real metric must order a worse caption below a better one."""
        better = score_captions(REFS, REFS, metrics=("rouge_l",))["rouge_l"]["value"]
        worse = score_captions(
            ["an unrelated sentence entirely", "another unrelated sentence"],
            REFS,
            metrics=("rouge_l",),
        )["rouge_l"]["value"]
        assert worse < better

    @needs_bleu
    def test_the_score_records_the_implementation_that_produced_it(self):
        """A number without its implementation is not reproducible."""
        out = score_captions(PREDS, REFS, metrics=("bleu",))["bleu"]
        assert out["implementation"] == "sacrebleu"
        assert out["version"] == "2.4.3"
        assert out["n"] == len(PREDS)
        assert out["detail"]["tokenizer"]
        assert out["detail"]["smoothing"]

    @needs_bleu
    @needs_rouge
    def test_two_metrics_are_returned_independently(self):
        out = score_captions(PREDS, REFS, metrics=("bleu", "rouge_l"))
        assert set(out) == {"bleu", "rouge_l"}
        assert out["bleu"]["value"] != out["rouge_l"]["value"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
class TestDeterminism:
    @needs_bleu
    @needs_rouge
    def test_the_same_input_produces_the_same_numbers(self):
        a = score_captions(PREDS, REFS, metrics=("bleu", "rouge_l"))
        b = score_captions(PREDS, REFS, metrics=("bleu", "rouge_l"))
        assert a == b

    @needs_bleu
    def test_the_metric_dict_is_json_serialisable(self):
        """It travels into a report, so it must survive a JSON round-trip."""
        out = score_captions(PREDS, REFS, metrics=("bleu",))
        assert json.loads(json.dumps(out)) == out


# ---------------------------------------------------------------------------
# Malformed and degenerate input
# ---------------------------------------------------------------------------
class TestBadInput:
    def test_mismatched_lengths_raise(self):
        with pytest.raises(CaptionMetricError) as excinfo:
            score_captions(["a"], ["a", "b"], metrics=("bleu",))
        assert "aligned" in str(excinfo.value)

    def test_an_unknown_metric_name_raises(self):
        with pytest.raises(CaptionMetricError) as excinfo:
            score_captions(PREDS, REFS, metrics=("wer",))
        assert "unknown caption metric" in str(excinfo.value)

    def test_an_unknown_name_is_rejected_before_anything_is_computed(self):
        """A typo must not silently score the valid subset of the request."""
        with pytest.raises(CaptionMetricError):
            score_captions(PREDS, REFS, metrics=("bleu", "nope"))

    def test_an_empty_corpus_is_refused_rather_than_scored(self):
        """A metric over zero pairs is undefined, not zero.

        Before this was fixed the four behaved differently on empty input:
        sacrebleu raised a bare `IndexError` from inside the library and ROUGE-L
        returned a fabricated 0.0. Neither is acceptable, so the boundary refuses
        once, for all four metrics, with a diagnosable message.
        """
        with pytest.raises(CaptionMetricError) as excinfo:
            score_captions([], [], metrics=("bleu",))
        message = str(excinfo.value)
        assert "empty corpus" in message
        assert "not a score of zero" in message

    def test_an_empty_corpus_is_refused_for_every_metric(self):
        for metric in CAPTION_METRICS:
            with pytest.raises(CaptionMetricError):
                score_captions([], [], metrics=(metric,))

    @needs_rouge
    def test_an_empty_prediction_string_is_accepted_as_a_value_not_a_crash(self):
        """One empty caption is a bad caption, not an empty corpus."""
        out = score_captions([""], ["a building"], metrics=("rouge_l",))
        assert out["rouge_l"]["value"] == pytest.approx(0.0)

    def test_an_unavailable_metric_raises_even_when_other_metrics_are_requested(self):
        """Partial success would let the caller believe all four were computed."""
        if _available("cider"):
            pytest.skip("cider is available here, so this path cannot be exercised")
        with pytest.raises(CaptionMetricUnavailableError):
            score_captions(PREDS, REFS, metrics=("bleu", "cider"))


# ---------------------------------------------------------------------------
# BERTScore's checkpoint constraint
# ---------------------------------------------------------------------------
class TestBertScoreCheckpoint:
    def test_an_unsupported_checkpoint_gives_a_named_error_not_a_keyerror(
        self, monkeypatch
    ):
        """bert-score resolves checkpoints through its own table.

        A perfectly valid HuggingFace repo that is not in that table raises a bare
        `KeyError` from inside the library. That is translated into a named error
        carrying the constraint, so the caller is not left reading bert-score's
        source to find out why.
        """
        monkeypatch.setattr(caption, "_importable", lambda m: (True, None))
        monkeypatch.setattr(caption, "_installed_version", lambda d: "0.3.13")
        monkeypatch.setattr(
            caption,
            "_check_extra_requirement",
            lambda req, **kw: None,  # pretend the weights are fine
        )
        with pytest.raises(CaptionMetricError) as excinfo:
            score_captions(
                PREDS, REFS, metrics=("bertscore",), model_type="not/a/bert-score/model"
            )
        message = str(excinfo.value)
        assert "model2layers" in message
        assert "does not support" in message

    def test_the_gate_checks_the_checkpoint_the_caller_asked_for(
        self, tmp_path, monkeypatch
    ):
        """The gate must not refuse a healthy override because the default is broken.

        Measured: with the declared default (`roberta-large`) poisoned, a request
        naming a different checkpoint was refused, because the gate tested the
        default instead of the request.
        """
        monkeypatch.setattr(caption, "_hf_hub_dir", lambda: tmp_path / "hub")
        snap = tmp_path / "hub" / "models--good--model" / "snapshots" / "r"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}", encoding="utf-8")

        poisoned = tmp_path / "hub" / "models--bad--model" / "snapshots" / "r"
        poisoned.mkdir(parents=True)
        (poisoned / "config.json").write_bytes(b"")

        assert caption_metric_info("bertscore", probe_network=False)["available"] is False
        info = caption_metric_info(
            "bertscore", probe_network=False, model_override="good/model"
        )
        assert info["model_repo"] == "good/model"
        assert info["declared_model_repo"] == "roberta-large"
        assert info["model_repo_overridden"] is True
