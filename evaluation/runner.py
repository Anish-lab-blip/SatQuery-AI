"""The evaluation runner: manifests -> adapters -> metrics -> normalise -> report.

WHAT THIS RUNNER IS FOR
-----------------------
It is the single place where a benchmark score is produced, so it is the single
place where a fabricated score could be produced. Its design therefore
concentrates on making fabrication structurally impossible rather than on
producing numbers:

  1. **Every benchmark gets a status**, from `BenchmarkStatus`. There is no code
     path that emits a score without one.
  2. **A score's status bounds what it may be used for.** Only `REAL` scores are
     collected into `report["real_metrics"]`; `FIXTURE` and `DEGRADED` scores are
     reported in their own keys. A caller who wants a benchmark number has to
     ask for `real_metrics`, which is empty whenever no real corpus was present.
  3. **No aggregate is invented.** `aggregate_score` is populated only if the
     plan's `official_weights` cease to be `None`. While they are `None`, the
     key holds an explanation of why no aggregate exists -- and the call to
     `normalize.aggregate()` is still made and its refusal recorded, so the
     prohibition is exercised rather than merely documented.
  4. **Normalisation is not optional and not silent.** Each metric goes through
     `normalize.normalize()`; an unregistered metric raises, and the raise is
     captured as a normalisation failure for that metric rather than being
     dropped. A metric that could not be normalised is visible, not missing.
  5. **The run manifest is a precondition, not a nice-to-have.** A run without a
     populated manifest is reported as such.

WHAT IT WILL NOT DO
-------------------
It will not invent a benchmark, a dataset, a threshold, or an aggregate. It will
not turn a missing corpus into an empty-but-green result. And it will not decide
whether a number is "good" -- that is a judgement, and the plan's official
metrics are the only thing that could make it, which is why their absence is
reported as a first-class fact.

USAGE
-----
    from evaluation.runner import EvaluationRunner

    runner = EvaluationRunner(config=config, seed=42)
    report = runner.run(["levir_cd"], scorers={"levir_cd": my_scorer})
    print(report["summary"]["line"])
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from core.code_revision import REPO_ROOT
from evaluation.benchmark_adapters import (
    AdapterNotRegisteredError,
    BenchmarkAdapter,
    BenchmarkNotAvailableError,
    BenchmarkStatus,
    declared_inventory,
    get_adapter,
    inventory as adapter_inventory,
    registered_adapters,
)
from evaluation.manifests import RunManifest
from evaluation.normalize import (
    NormalisationError,
    OfficialWeightsUnavailableError,
    aggregate,
    normalize,
    official_weights,
)
from evaluation.public_test import public_test_state
from evaluation.run_manifest import manifest_completeness, populate_run_manifest

__all__ = [
    "BenchmarkResult",
    "EvaluationRunner",
    "MetricNormalisation",
    "normalise_metrics",
    "aggregate_or_explain",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
@dataclass
class MetricNormalisation:
    """The fate of one raw metric on its way through the section 63 registry."""

    metric: str
    raw: float
    normalized: float | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.normalized is not None and self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "raw": self.raw,
            "normalized": self.normalized,
            "error": self.error,
        }


def normalise_metrics(raw: Mapping[str, Any]) -> tuple[dict[str, float], list[MetricNormalisation]]:
    """Normalise each raw metric, recording failures instead of dropping them.

    Returns `(normalized, records)`. A metric that fails normalisation is ABSENT
    from `normalized` and PRESENT in `records` with its error, so a consumer
    cannot mistake "could not be normalised" for "was not measured".

    Non-numeric values raise rather than being coerced: a metric whose value is
    a string is a bug in the adapter, and `float("abc")` failing deep inside a
    report writer is a worse way to find out.
    """
    normalized: dict[str, float] = {}
    records: list[MetricNormalisation] = []

    for metric in sorted(raw):
        value = raw[metric]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            records.append(
                MetricNormalisation(
                    metric=metric,
                    raw=value,  # type: ignore[arg-type]
                    normalized=None,
                    error=(
                        f"metric value is {type(value).__name__}, not a number; "
                        f"refusing to coerce"
                    ),
                )
            )
            continue

        try:
            normalized[metric] = normalize(metric, float(value))
            records.append(
                MetricNormalisation(metric=metric, raw=float(value),
                                    normalized=normalized[metric])
            )
        except NormalisationError as exc:
            records.append(
                MetricNormalisation(
                    metric=metric,
                    raw=float(value),
                    normalized=None,
                    error=f"{type(exc).__name__}: {exc.user_message}",
                )
            )

    return normalized, records


def aggregate_or_explain(metric_scores: Mapping[str, float]) -> dict[str, Any]:
    """Attempt the official aggregate, and RECORD its refusal.

    While `official_weights is None` this always fails. Recording the failure
    rather than short-circuiting on the `None` check means the prohibition is
    actually exercised on every run: if someone later assigns weights without
    implementing a formula, this call still raises, and the report says so. A
    bare `if official_weights is None` would have hidden that.
    """
    try:
        score = aggregate(metric_scores)
        return {
            "available": True,
            "score": score,
            "weights": official_weights,
            "note": "the official aggregate was computed from published weights",
        }
    except OfficialWeightsUnavailableError as exc:
        return {
            "available": False,
            "score": None,
            "weights": official_weights,
            "error": f"{type(exc).__name__}",
            "reason": exc.user_message,
            "note": (
                "No official aggregate exists. Plan section 63 prohibits inventing "
                "an aggregate formula, so no single score is reported. Per-metric "
                "normalised values remain available above."
            ),
        }


# ---------------------------------------------------------------------------
# Per-benchmark result
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    """The outcome of one benchmark attempt.

    `status` is the load-bearing field. `metrics` is populated for FIXTURE,
    DEGRADED and REAL alike, because the numbers are real arithmetic; what
    differs is what they describe, and that is what `status` records.
    """

    benchmark: str
    status: BenchmarkStatus
    n_samples: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    normalized: dict[str, float] = field(default_factory=dict)
    normalisation: list[MetricNormalisation] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None
    started_at: str | None = None
    ended_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "status": self.status.value,
            "scored": self.status.is_scored,
            "n_samples": self.n_samples,
            "metrics": dict(self.metrics),
            "normalized": dict(self.normalized),
            "normalisation": [n.to_dict() for n in self.normalisation],
            "detail": dict(self.detail),
            "error": self.error,
            "traceback": self.traceback,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------
#: A scorer turns loaded samples into raw metrics. The runner knows nothing about
#: any benchmark's task; the scorer is where task knowledge lives, which keeps
#: the runner free of benchmark-specific assumptions it would otherwise be
#: tempted to guess at.
Scorer = Callable[[Sequence[Any]], Mapping[str, Any]]


class EvaluationRunner:
    """Runs declared benchmarks and reports exactly what happened.

    Args:
        config: the resolved config object. Used for the config hash and for the
            hidden-data assertions. Optional so the runner can be exercised in a
            unit test without a full config.
        seed: recorded in the run manifest.
        thresholds: recorded in the run manifest.
        model_revisions: recorded in the run manifest.
        run_id: recorded in the run manifest. Defaults to a UTC timestamp so two
            runs never collide, which matters because the manifest hash is what
            a published number is pinned to.
    """

    def __init__(
        self,
        *,
        config: Any = None,
        seed: int = 42,
        thresholds: Mapping[str, float] | None = None,
        model_revisions: Mapping[str, str] | None = None,
        run_id: str | None = None,
        root: str | Path = REPO_ROOT,
        public_test_root: str | Path | None = None,
    ) -> None:
        self.config = config
        self.seed = seed
        self.thresholds = dict(thresholds or {})
        self.model_revisions = dict(model_revisions or {})
        self.run_id = run_id or f"eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        self.root = Path(root)
        self.public_test_root = public_test_root
        self.started_at = _utc_now()

    # -- manifest ----------------------------------------------------------
    def config_hash(self) -> str:
        """The config hash, or an explicit statement that it is unavailable.

        Never a placeholder. A made-up hash in a run manifest is worse than an
        absent one, because it looks like a pinned configuration.
        """
        if self.config is None:
            return "unavailable (no config supplied)"
        candidate = getattr(self.config, "hash", None)
        if candidate:
            return str(candidate)
        getter = getattr(self.config, "get", None)
        if callable(getter):
            value = getter("config_hash", None)
            if value:
                return str(value)
        return "unavailable (config exposes no hash)"

    def build_run_manifest(self, *, dataset_manifest_hash: str) -> RunManifest:
        """Assemble a fully populated run manifest for this run."""
        manifest = RunManifest(
            run_id=self.run_id,
            config_hash=self.config_hash(),
            dataset_manifest_hash=dataset_manifest_hash,
            model_revisions=dict(self.model_revisions),
            prompt_versions={},
            thresholds=dict(self.thresholds),
            seed=self.seed,
            code_revision=None,
            environment={},
            started_at=None,
        )
        return populate_run_manifest(
            manifest,
            root=self.root,
            started_at=self.started_at,
            # `ended_at` deliberately not passed: the run has not ended yet, and
            # this runner will not invent an end time for it.
        )

    # -- running ----------------------------------------------------------
    def run_one(
        self,
        benchmark: str,
        *,
        scorer: Scorer | None = None,
        split: str = "test",
    ) -> BenchmarkResult:
        """Run one benchmark and classify the outcome.

        The status ladder, in order:
          1. no adapter registered            -> NOT_RUN
          2. corpus unavailable               -> NOT_RUN
          3. load() or scorer() raised        -> FAILED
          4. adapter's own verdict on its material, via `corpus_kind()`:
               REAL / DEGRADED / FIXTURE

        Step 4 was a two-way branch on `is_official()` when this docstring was
        first written:

            step 4: adapter reports non-official -> FIXTURE
            step 5: adapter reports official     -> REAL

        That made `DEGRADED` unreachable, so a real-but-partial corpus was
        labelled FIXTURE -- i.e. called synthetic. `corpus_kind()` now answers
        the question directly, and its base implementation reproduces the old
        two-way result exactly, so no adapter changed meaning when this landed.
        `is_official()` still matters: it is the adapter's attested claim, and
        the burden of proof sits with the adapter.
        """
        result = BenchmarkResult(
            benchmark=benchmark,
            status=BenchmarkStatus.NOT_RUN,
            started_at=_utc_now(),
        )

        # --- 1. adapter present? ------------------------------------------
        try:
            adapter: BenchmarkAdapter = get_adapter(benchmark)
        except AdapterNotRegisteredError as exc:
            result.error = exc.user_message
            result.detail["registry"] = {
                "registered": sorted(registered_adapters()),
                "reason": "no adapter is wired up for this benchmark",
            }
            result.ended_at = _utc_now()
            return result

        result.detail["adapter"] = {
            "class": type(adapter).__name__,
            "official": adapter.is_official(),
            "metrics": list(adapter.metric_names()),
        }

        # --- 2. corpus present? -------------------------------------------
        try:
            samples = adapter.require_available(split=split)
        except BenchmarkNotAvailableError as exc:
            result.error = exc.user_message
            result.detail["corpus"] = dict(exc.context or {})
            result.ended_at = _utc_now()
            return result
        except Exception as exc:  # noqa: BLE001
            result.status = BenchmarkStatus.FAILED
            result.error = f"{type(exc).__name__}: {exc}"
            result.traceback = traceback.format_exc()
            result.ended_at = _utc_now()
            return result

        # --- 3. score ------------------------------------------------------
        if not samples:
            # An adapter that claims availability but yields no samples has a
            # defect. Treating it as a zero-sample success would look like a
            # benchmark that scored nothing rather than an adapter that is broken.
            result.status = BenchmarkStatus.FAILED
            result.error = (
                f"adapter {type(adapter).__name__} reported an available corpus but "
                f"returned no samples for split={split!r}"
            )
            result.ended_at = _utc_now()
            return result

        if scorer is None:
            result.error = (
                f"no scorer supplied for {benchmark!r}; the runner will not invent "
                f"one. Pass scorer=<callable taking loaded samples>"
            )
            result.ended_at = _utc_now()
            return result

        try:
            raw = dict(scorer(samples))
        except Exception as exc:  # noqa: BLE001
            result.status = BenchmarkStatus.FAILED
            result.error = f"{type(exc).__name__}: {exc}"
            result.traceback = traceback.format_exc()
            result.ended_at = _utc_now()
            return result

        # --- 4/5. classify -------------------------------------------------
        # `corpus_kind()` replaces the earlier two-way `is_official()` branch,
        # which could never yield DEGRADED. Its default reproduces the old
        # behaviour exactly, so only adapters that opt in to DEGRADED change.
        result.status = adapter.corpus_kind()
        result.n_samples = len(samples)
        result.metrics = raw

        normalized, records = normalise_metrics(raw)
        result.normalized = normalized
        result.normalisation = records
        result.detail["normalisation_failures"] = [
            r.metric for r in records if not r.ok
        ]
        result.detail["status_reason"] = _STATUS_REASON[result.status]
        result.ended_at = _utc_now()
        return result

    def run(
        self,
        benchmarks: Sequence[str],
        *,
        scorers: Mapping[str, Scorer] | None = None,
        split: str = "test",
        dataset_manifest_hash: str = "unavailable (no dataset manifest supplied)",
    ) -> dict[str, Any]:
        """Run every named benchmark and assemble the report.

        `scorers` is keyed by benchmark name. A benchmark with no scorer is still
        run as far as the adapter check, so its status distinguishes "not wired
        up" from "scored".
        """
        scorers = dict(scorers or {})
        results = [
            self.run_one(name, scorer=scorers.get(name), split=split)
            for name in benchmarks
        ]

        manifest = self.build_run_manifest(dataset_manifest_hash=dataset_manifest_hash)

        by_status: dict[str, list[str]] = {s.value: [] for s in BenchmarkStatus}
        for r in results:
            by_status[r.status.value].append(r.benchmark)

        # Only REAL results contribute to `real_metrics`. This is the single
        # boundary that keeps a fixture number out of a benchmark claim.
        real_metrics: dict[str, dict[str, float]] = {
            r.benchmark: dict(r.normalized)
            for r in results
            if r.status.is_scored
        }
        fixture_metrics: dict[str, dict[str, Any]] = {
            r.benchmark: dict(r.metrics)
            for r in results
            if r.status is BenchmarkStatus.FIXTURE
        }
        # A DEGRADED result is real arithmetic over a real-but-partial corpus. It
        # is reported separately from `real_metrics` so the published-benchmark
        # boundary stays exactly as it was, while the subset numbers are still
        # available and unmistakably labelled.
        degraded_metrics: dict[str, dict[str, Any]] = {
            r.benchmark: dict(r.metrics)
            for r in results
            if r.status is BenchmarkStatus.DEGRADED
        }

        # The aggregate is attempted over REAL metrics only. With no real
        # benchmark present there is nothing to aggregate, which is reported
        # rather than papered over.
        flat_real = {
            f"{bench}.{metric}": value
            for bench, metrics in real_metrics.items()
            for metric, value in metrics.items()
        }
        aggregate_block = aggregate_or_explain(flat_real)

        report: dict[str, Any] = {
            "schema": "evaluation_report_v1",
            "run_id": self.run_id,
            "generated_at": _utc_now(),
            "runner_started_at": self.started_at,
            "seed": self.seed,
            "thresholds": dict(self.thresholds),
            "split": split,
            "run_manifest": manifest.to_dict(),
            "run_manifest_hash": manifest.hash,
            "run_manifest_completeness": manifest_completeness(manifest),
            "config_hash": manifest.config_hash,
            "dataset_manifest_hash": manifest.dataset_manifest_hash,
            "adapters": adapter_inventory(),
            "declared_benchmarks": declared_inventory(),
            "public_test": (
                public_test_state(self.public_test_root)
                if self.public_test_root is not None
                else {
                    "available": None,
                    "note": "public-test state not inspected for this run",
                }
            ),
            "results": {r.benchmark: r.to_dict() for r in results},
            "summary": {
                "requested": list(benchmarks),
                "by_status": by_status,
                "n_requested": len(benchmarks),
                "n_real": len(by_status[BenchmarkStatus.REAL.value]),
                "n_fixture": len(by_status[BenchmarkStatus.FIXTURE.value]),
                "n_degraded": len(by_status[BenchmarkStatus.DEGRADED.value]),
                "n_failed": len(by_status[BenchmarkStatus.FAILED.value]),
                "n_not_run": len(by_status[BenchmarkStatus.NOT_RUN.value]),
            },
            "real_metrics": real_metrics,
            "fixture_metrics": fixture_metrics,
            "degraded_metrics": degraded_metrics,
            "aggregate_score": aggregate_block,
            "metric_status_legend": {
                s.value: _STATUS_LEGEND[s] for s in BenchmarkStatus
            },
            "note": (
                "`real_metrics` is empty unless a benchmark ran against its own "
                "full corpus. `degraded_metrics` holds real numbers produced "
                "against a disclosed subset or proxy, and describes that subset "
                "rather than the benchmark. `fixture_metrics` holds numbers "
                "produced against generated material and must never be quoted as "
                "benchmark scores. No aggregate exists while the official weights "
                "are unpublished."
            ),
        }
        return report

    # -- writing -----------------------------------------------------------
    def write(self, report: Mapping[str, Any], path: str | Path) -> Path:
        """Write the report as JSON, plus a sidecar digest of the file itself.

        A file cannot contain a hash of itself, so the digest goes beside it.
        This mirrors the convention established for `eval_summary.json` in the
        provenance work, so a reviewer finds one convention rather than two.
        """
        import hashlib
        import json

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(report, indent=2, sort_keys=True, default=str)
        target.write_text(text, encoding="utf-8")

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        sidecar = target.with_suffix(target.suffix + ".sha256")
        sidecar.write_text(f"{digest}  {target.name}\n", encoding="utf-8")
        return target


#: Why each status was assigned, in the adapter's own terms. Kept beside the
#: legend so a reader sees both "what this status means" and "why it was given".
_STATUS_REASON: dict[BenchmarkStatus, str] = {
    BenchmarkStatus.REAL: (
        "adapter attests its corpus is the benchmark's own"
    ),
    BenchmarkStatus.DEGRADED: (
        "adapter attests the material is real but is a disclosed subset or proxy "
        "of the benchmark, not the full corpus"
    ),
    BenchmarkStatus.FIXTURE: (
        "adapter generates or proxies its material; the score describes the "
        "fixture, not the benchmark"
    ),
}

#: What each status means, emitted in every report so a reader never has to find
#: this module to interpret a number.
_STATUS_LEGEND: dict[BenchmarkStatus, str] = {
    BenchmarkStatus.NOT_RUN: (
        "No execution took place: no adapter is registered for this benchmark, or "
        "its corpus is absent. No number is reported."
    ),
    BenchmarkStatus.FAILED: (
        "Execution was attempted and raised. The traceback is attached. Any "
        "partial value is not reported."
    ),
    BenchmarkStatus.DEGRADED: (
        "Execution completed against a corpus the adapter does not attest as the "
        "benchmark's own (a subset or proxy). The numbers describe that corpus."
    ),
    BenchmarkStatus.FIXTURE: (
        "Execution completed against generated material. The arithmetic is real; "
        "the subject is not the benchmark. Never quote as a benchmark score."
    ),
    BenchmarkStatus.REAL: (
        "Execution completed against the benchmark's own corpus, as attested by "
        "the adapter. This is the only status that yields a benchmark score."
    ),
}
