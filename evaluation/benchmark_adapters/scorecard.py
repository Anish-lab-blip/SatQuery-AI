"""The benchmark scorecard: one row per benchmark, exactly one state each.

WHY THIS EXISTS
---------------
`evaluation.runner` produces a report per run, and `base.BenchmarkStatus` gives
each *attempt* a status. What was missing is the project-level view a reader
actually needs: for every benchmark the plan names, what is the single honest
word for where it stands, and is there a number behind it?

The six states
--------------
Five come from `BenchmarkStatus` and describe an *attempt*. The sixth comes from
`heldout` and describes a *resource condition* -- the corpus does not exist, so
no attempt was possible and none should be implied. Without it, "we have not run
this because the data is not here" and "we have not run this yet" would look the
same, and they need different actions.

    REAL              ran against the benchmark's own full corpus. The only
                      state that may back a published score.
    DEGRADED          ran, against a real but partial/proxy corpus. The number is
                      real arithmetic over disclosed material; it describes that
                      material, not the benchmark.
    FIXTURE           ran, against self-generated material. Proves the machinery.
                      Never a benchmark score.
    NOT_RUN           not attempted: no adapter registered, or the caller skipped.
    FAILED            attempted and raised.
    RESOURCE_BLOCKED  not attemptable: the immutable corpus does not exist here.

THE ONE RULE THIS MODULE ENFORCES
---------------------------------
`published_scores` contains REAL rows and nothing else. There is no code path
that puts a DEGRADED or FIXTURE number into it, and no averaging across states --
a mean of a real subset and a synthetic fixture would be a number describing
neither. When there are no REAL rows the scorecard says the benchmarks are
UNMEASURED, which is not the same as scoring zero, and the wording is checked by
a test because that distinction is the whole point of the scorecard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from evaluation.benchmark_adapters.heldout import RESOURCE_BLOCKED

__all__ = [
    "SCORECARD_STATES",
    "STATE_MEANINGS",
    "SCORE_BEARING_STATES",
    "ScorecardRow",
    "build_scorecard",
    "render_scorecard",
]


#: The six states, in the order a reader should meet them: best evidence first,
#: then progressively weaker claims, then the two kinds of absence.
SCORECARD_STATES: tuple[str, ...] = (
    "REAL",
    "DEGRADED",
    "FIXTURE",
    "FAILED",
    "NOT_RUN",
    RESOURCE_BLOCKED,
)

#: What each state asserts. Kept as data so the wording travels into the report
#: instead of being retyped by whoever writes the summary.
STATE_MEANINGS: dict[str, str] = {
    "REAL": (
        "ran against the benchmark's own corpus; the only state that may back a "
        "published score"
    ),
    "DEGRADED": (
        "ran against a real but partial or proxy corpus; the number describes "
        "that material, not the benchmark"
    ),
    "FIXTURE": (
        "ran against self-generated material; proves the machinery and is never "
        "a benchmark score"
    ),
    "FAILED": "execution was attempted and raised",
    "NOT_RUN": "not attempted: no adapter registered, or deliberately skipped",
    RESOURCE_BLOCKED: (
        "not attemptable here: the immutable evaluation corpus does not exist"
    ),
}

#: Which states may carry a number a reader is allowed to quote. Exactly one.
SCORE_BEARING_STATES: frozenset[str] = frozenset({"REAL"})


@dataclass
class ScorecardRow:
    """One benchmark's standing.

    `metrics` is populated for any state that ran (REAL, DEGRADED, FIXTURE) and
    empty otherwise. `counts_as_score` is derived from the state, never passed in,
    so a caller cannot label a fixture row as score-bearing.
    """

    benchmark: str
    state: str
    metrics: dict[str, Any] = field(default_factory=dict)
    n_samples: int | None = None
    reason: str | None = None
    corpus_root: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state not in SCORECARD_STATES:
            raise ValueError(
                f"unknown scorecard state {self.state!r}; expected one of "
                f"{list(SCORECARD_STATES)}"
            )

    @property
    def counts_as_score(self) -> bool:
        return self.state in SCORE_BEARING_STATES

    @property
    def ran(self) -> bool:
        """Whether any execution happened, whatever the material was."""
        return self.state in ("REAL", "DEGRADED", "FIXTURE")

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "state": self.state,
            "meaning": STATE_MEANINGS[self.state],
            "counts_as_score": self.counts_as_score,
            "ran": self.ran,
            "n_samples": self.n_samples,
            "metrics": dict(self.metrics),
            "reason": self.reason,
            "corpus_root": self.corpus_root,
            "evidence": dict(self.evidence),
        }


def _rows_from_run_report(report: Mapping[str, Any]) -> list[ScorecardRow]:
    """Turn one runner report into rows. One row per result, state preserved.

    The reason is taken in priority order from the keys the runner actually sets:
    `detail.status_reason` for a completed attempt, `detail.corpus.reason` for a
    corpus that was absent, and `error` for anything that raised. Reading a key
    the runner does not set would produce a row with no explanation, which is the
    failure this function is here to avoid.
    """
    rows: list[ScorecardRow] = []
    for name, result in sorted((report.get("results") or {}).items()):
        status = str(result.get("status"))
        if status not in SCORECARD_STATES:
            # A status this scorecard does not know about must not be silently
            # mapped onto one it does -- that is how a new status quietly
            # acquires the meaning of an old one.
            raise ValueError(
                f"runner reported status {status!r} for {name!r}, which is not a "
                f"scorecard state ({list(SCORECARD_STATES)}); the scorecard and "
                f"the runner have diverged"
            )
        detail = result.get("detail") or {}
        corpus = detail.get("corpus") or {}
        adapter_block = detail.get("adapter") or {}
        reason = (
            detail.get("status_reason")
            or corpus.get("reason")
            or result.get("error")
            or None
        )
        rows.append(
            ScorecardRow(
                benchmark=name,
                state=status,
                metrics=dict(result.get("metrics") or {}),
                n_samples=result.get("n_samples"),
                reason=reason,
                corpus_root=corpus.get("root"),
                evidence={
                    "official": adapter_block.get("official"),
                    "status_reason": detail.get("status_reason"),
                    "normalisation_failures": detail.get("normalisation_failures"),
                    "error": result.get("error"),
                    "scored": result.get("scored"),
                },
            )
        )
    return rows


def _row_from_held_out(payload: Mapping[str, Any]) -> ScorecardRow:
    """Turn the held-out status into a row.

    Only `resource_blocked` becomes a RESOURCE_BLOCKED row. An available-but-
    unsealed or sealed corpus is NOT a scorecard state: it is a precondition that
    has been met, and claiming REAL from it would assert a score that no run
    produced. Such a row is reported as NOT_RUN with the precondition recorded in
    `evidence`, so "the data is here and nothing has been run against it" is
    distinguishable from both a score and a missing corpus.
    """
    state = str(payload.get("state"))
    if state == RESOURCE_BLOCKED:
        return ScorecardRow(
            benchmark="held_out_public_split",
            state=RESOURCE_BLOCKED,
            reason=payload.get("reason"),
            corpus_root=payload.get("root"),
            evidence={"held_out_state": state},
        )
    return ScorecardRow(
        benchmark="held_out_public_split",
        state="NOT_RUN",
        reason=(
            "the held-out corpus is present and "
            f"{state}, but no evaluation has been run against it"
        ),
        corpus_root=payload.get("root"),
        evidence={"held_out_state": state},
    )


def build_scorecard(
    *,
    run_report: Mapping[str, Any] | None = None,
    held_out: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the project-level scorecard.

    Args:
        run_report: an `EvaluationRunner.run()` report, or None when no run was
            performed. None produces no attempt rows -- which is different from a
            report full of NOT_RUN, and the output says which it is.
        held_out: a `heldout.held_out_status()` payload, or None to omit the row.

    Returns:
        A report-ready dict. `published_scores` holds REAL rows only. When there
        are none, `status` is `"UNMEASURED"` and `note` says so in words, because
        an empty scorecard and a scorecard of zeros are not the same claim.
    """
    rows: list[ScorecardRow] = []
    if run_report is not None:
        rows.extend(_rows_from_run_report(run_report))
    if held_out is not None:
        rows.append(_row_from_held_out(held_out))

    by_state: dict[str, list[str]] = {s: [] for s in SCORECARD_STATES}
    for row in rows:
        by_state[row.state].append(row.benchmark)

    published = {
        row.benchmark: dict(row.metrics) for row in rows if row.counts_as_score
    }
    degraded = {
        row.benchmark: dict(row.metrics) for row in rows if row.state == "DEGRADED"
    }
    fixture = {
        row.benchmark: dict(row.metrics) for row in rows if row.state == "FIXTURE"
    }

    if published:
        status = "MEASURED"
        note = (
            f"{len(published)} benchmark(s) ran against their own full corpus. "
            f"Only these may be quoted as benchmark scores."
        )
    else:
        status = "UNMEASURED"
        note = (
            "No benchmark ran against its own full corpus, so no benchmark score "
            "exists. This is NOT the same as scoring zero: an unmeasured "
            "benchmark has no value, and reporting 0.0 for it would invent one. "
            "Rows in DEGRADED or FIXTURE describe the material that was actually "
            "read, and are not substitutes."
        )

    return {
        "schema": "benchmark_scorecard_v1",
        "status": status,
        "n_rows": len(rows),
        "n_published": len(published),
        "n_degraded": len(degraded),
        "n_fixture": len(fixture),
        "n_failed": len(by_state["FAILED"]),
        "n_not_run": len(by_state["NOT_RUN"]),
        "n_resource_blocked": len(by_state[RESOURCE_BLOCKED]),
        "by_state": by_state,
        "rows": [r.to_dict() for r in rows],
        # The only boundary that matters: REAL and nothing else.
        "published_scores": published,
        "degraded_metrics": degraded,
        "fixture_metrics": fixture,
        "states": {
            s: {"meaning": STATE_MEANINGS[s], "counts_as_score": s in SCORE_BEARING_STATES}
            for s in SCORECARD_STATES
        },
        "run_performed": run_report is not None,
        "note": note,
    }


def render_scorecard(scorecard: Mapping[str, Any]) -> str:
    """A plain-text table of the scorecard, for a terminal or a log.

    Deliberately terse: one line per benchmark, state first. A reader who reads
    only this must still come away knowing which rows carry a quotable number,
    so `counts_as_score` is marked with an explicit `[score]` tag rather than
    left to the reader to derive from the state name.
    """
    lines = [
        f"benchmark scorecard -- {scorecard.get('status')} "
        f"({scorecard.get('n_published')} published, "
        f"{scorecard.get('n_degraded')} degraded, "
        f"{scorecard.get('n_fixture')} fixture, "
        f"{scorecard.get('n_resource_blocked')} resource-blocked)",
        "",
    ]
    if not scorecard.get("rows"):
        lines.append("  (no benchmarks were attempted or inspected)")
        return "\n".join(lines)

    width = max(len(str(r["benchmark"])) for r in scorecard["rows"])
    for row in scorecard["rows"]:
        tag = "[score]" if row["counts_as_score"] else "       "
        metrics = (
            ", ".join(f"{k}={v}" for k, v in sorted(row["metrics"].items()))
            or "-"
        )
        lines.append(
            f"  {tag} {str(row['benchmark']).ljust(width)}  "
            f"{row['state'].ljust(17)} {metrics}"
        )
        if row.get("reason") and not row["counts_as_score"]:
            lines.append(f"          why: {row['reason']}")
    lines.append("")
    lines.append(f"  {scorecard.get('note')}")
    return "\n".join(lines)
