"""Declared benchmarks and their (absent) corpora.

CORRECTION 2026-09-23 — "absent" means absent AT THE DECLARED ROOT, which is a
canonical TARGET location, not absent from the repository. All FOUR of the
corpora are on disk under other paths (`data/levir/`,
`training/data/vrsbench/`, `data/bigearthnet_v2/reben/BigEarthNet-S1`,
`data/cdvqa/annotations/`), and all four predate this file — so the earlier
per-benchmark notes asserting that the repository held no such corpus were false
when written, not merely superseded. What is genuinely missing is the SCORING
half, not the data. The notes below were corrected accordingly; the declared
roots were deliberately left alone, because pinning the target location is their
purpose.

SECOND CORRECTION 2026-09-23 (same day, separate finding) — the first correction
above initially said "three of the four", because it was written before the
Change-VQA held-out splits were located. The `change_vqa_test` note originally
claimed the held-out question/answer splits "live in the Kaggle export, not in
the repository". That claim is FALSE: `data/cdvqa/annotations/` holds
`Test_questions.json` / `Test_answers.json` and the `Test2` pair, and the join is
exact — every one of the 39,686 Test question ids resolves to exactly one image
record and one answer row. The note has been rewritten to record what was
measured. This is logged as a separate correction rather than folded into the
first because the two findings have different causes: the first was a
declared-root-vs-actual-root confusion, the second was a corpus that was looked
for in one place and assumed absent when it was in another.

A third, narrower correction: `adapter_registered` in `to_dict()` was a hard-coded
`False`. That was accurate when no adapter existed and would have become a lie
the moment one was registered, so it is now computed against the live registry.
`citation_verified_here` stays hard-coded `False` and that is correct — this
session still has not re-fetched any citation.

WHY DECLARE BENCHMARKS WE CANNOT RUN
------------------------------------
The plan's checklist names "benchmark adapters" as a required evaluation
deliverable. Declaring the benchmarks explicitly -- with their real roots and
real citations -- has three concrete benefits over leaving the directory empty:

  1. The gap becomes measurable. A report can say "LEVIR-CD: NOT_RUN,
     /data/LEVIR-CD not present" instead of "no benchmark was evaluated", which
     is the difference between a gap and a shrug.
  2. The path and citation are pinned now, so the person who eventually supplies
     the data does not have to re-derive which benchmark was meant and where it
     should go.
  3. Nothing here can produce a number. `DeclaredBenchmark` is data only. It has
     no `load()`, so it cannot be accidentally evaluated.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No synthetic sample generator. No "demo split". No `load()` that falls back to
generated data when the corpus is missing. Each of those would let a benchmark
row appear with a real-looking score describing material that is not the
benchmark. The declaration stops at the point where fabrication would begin.

The `citations` are quoted from the authoritative plan, which records them as the
plan's own sources; `citation_verified_here` is False on every entry because this
session did not independently re-fetch them. Recording that distinction matters:
a citation that has been checked and one that has been copied must not look the
same in a provenance report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.code_revision import REPO_ROOT
from evaluation.benchmark_adapters.base import CorpusDescription, registered_adapters

__all__ = [
    "DeclaredBenchmark",
    "DECLARED_BENCHMARKS",
    "declared_benchmark_names",
    "describe_declared",
    "declared_inventory",
]


@dataclass(frozen=True)
class DeclaredBenchmark:
    """A benchmark the plan names, with where its material would live.

    Data only. Intentionally has no evaluation method: a declaration can report
    absence but can never emit a score.
    """

    name: str
    #: Path at which the corpus would be expected. Relative paths resolve against
    #: the repository root so a checkout in a different location still reports a
    #: sensible path.
    root: str
    task: str
    #: The metric keys a real adapter for this benchmark would be expected to
    #: produce. These are the names that must exist in the plan section 63
    #: normalisation registry before any such adapter could report a value.
    expected_metrics: tuple[str, ...]
    citation: str
    plan_reference: str
    note: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_root(self) -> Path:
        path = Path(self.root)
        return path if path.is_absolute() else (REPO_ROOT / path)

    def to_dict(self) -> dict[str, Any]:
        root = self.resolved_root
        return {
            "name": self.name,
            "root": str(root),
            "root_exists": root.exists(),
            "task": self.task,
            "expected_metrics": list(self.expected_metrics),
            "citation": self.citation,
            # Kept separate so a copied citation is never mistaken for a checked one.
            "citation_verified_here": False,
            "plan_reference": self.plan_reference,
            "note": self.note,
            # MEASURED, not asserted (changed 2026-09-23). This was the constant
            # `False`, which was true only for as long as no adapter existed. A
            # declaration whose "is it wired up?" field cannot change is a field
            # that will eventually be wrong, so it now reads the live registry.
            # A declared name matches an adapter name by construction here: the
            # four adapter `name` attributes are exactly `levir_cd`, `vrsbench`,
            # `bigearthnet_s1` and `change_vqa_test`.
            "adapter_registered": self.name in registered_adapters(),
        }


#: Benchmarks the plan names for evaluation. Each entry is a statement of intent
#: plus a CANONICAL TARGET path. That path does not currently exist, and that is
#: the point: it is where the corpus belongs, not where it happens to be today.
#: See the module docstring -- the corpora themselves are on disk under other
#: roots, so `root_exists: false` on these entries must never be read as "the
#: benchmark is missing". The per-adapter `describe_corpus()` is what reports
#: availability; these declarations report only the target location.
DECLARED_BENCHMARKS: tuple[DeclaredBenchmark, ...] = (
    DeclaredBenchmark(
        name="levir_cd",
        root="data/LEVIR-CD",
        task="binary change detection (bitemporal optical pairs)",
        expected_metrics=("f1", "iou", "precision", "recall"),
        citation="LEVIR-CD (Zou et al.), as recorded in the implementation plan",
        plan_reference="plan section 13 / section 65 (benchmark table)",
        note=(
            "The plan names LEVIR-CD as the change-detection benchmark. The "
            "corpus IS on disk, but not at this declared root: it is at "
            "`data/levir/` (`keykeylv/levir-cd-256` -- flat A/B/label/edge trees "
            "with the split as a filename prefix, plus `list/`), and it predates "
            "this file. This root is the canonical TARGET location, so "
            "`root_exists: false` here is by design and is NOT a claim that the "
            "corpus is missing. The Phase-9 STANet artifact was trained and "
            "evaluated before this work order and its numbers are 'recorded-only' "
            "in the R-02 export, so this adapter would produce a comparable figure "
            "only once a scorer reads the corpus where it actually lives."
        ),
    ),
    DeclaredBenchmark(
        name="vrsbench",
        root="data/VRSBench",
        task="visual grounding / referring expressions",
        expected_metrics=("accuracy", "iou"),
        citation="VRSBench (lx709), as recorded in the implementation plan",
        plan_reference="plan sections 13-14 (grounding strategy)",
        note=(
            "The plan records that VRSBench annotation boxes are normalised to "
            "0-100 and warns that an evaluator must convert to the internal 0-1 "
            "convention explicitly rather than treating the numbers as pixels. "
            "`evaluation.metrics.grounding.benchmark_to_normalized` implements that "
            "conversion. The corpus IS on disk, but not at this declared root: it "
            "is at `training/data/vrsbench/` (train/val image archives plus "
            "`VRSBench_EVAL_referring.json`), and it predates this file. This root "
            "is the canonical TARGET location, so `root_exists: false` here is by "
            "design and is NOT a claim that the corpus is missing."
        ),
        extras={"box_scale": 100.0},
    ),
    DeclaredBenchmark(
        name="bigearthnet_s1",
        root="data/BigEarthNet-S1",
        task="SAR classification / single-SAR view",
        expected_metrics=("f1", "precision", "recall"),
        citation="BigEarthNet-S1, as recorded in the implementation plan",
        plan_reference="plan section 13 (single-SAR row of the modality table)",
        note=(
            "Named for the SAR view. The plan requires a sensor adapter that "
            "inspects the actually-available polarisation channels rather than "
            "assuming a fixed pair. The corpus IS on disk, but not at this "
            "declared root: it is at `data/bigearthnet_v2/reben/BigEarthNet-S1`, "
            "and it predates this file. This root is the canonical TARGET "
            "location, so `root_exists: false` here is by design and is NOT a "
            "claim that the corpus is missing."
        ),
    ),
    DeclaredBenchmark(
        name="change_vqa_test",
        root="artifacts/change_vqa/run",
        task="change-VQA (the R-02 held-out pair)",
        expected_metrics=("accuracy",),
        citation="N/A -- first-party held-out split produced by this project's R-02 run",
        plan_reference="plan section 2.2 (Change-VQA answer accuracy)",
        note=(
            "ORIGINAL (retained for the record, now known to be false): 'The one "
            "'benchmark' whose material IS partially present: the promoted R-02 "
            "head. But the held-out question/answer splits live in the Kaggle "
            "export, not in the repository, so a runner here still cannot score "
            "it.'  "
            "CORRECTION 2026-09-23: the second sentence is wrong. The held-out "
            "splits ARE in the repository, at `data/cdvqa/annotations/`: "
            "`Test_questions.json` + `Test_answers.json` (39,686 questions) and "
            "the `Test2` pair (31,036 questions), plus the Train/Val pair. The "
            "join is exact -- every question id resolves to exactly one image "
            "record and one answer row, 0 dropped -- so a runner HERE can score "
            "the Test/Test2 splits. What remains true is the first sentence and "
            "the last: the promoted R-02 head is present but its published "
            "figures stay recorded-only, because neither top-3 accuracy nor mean "
            "confidence is reconstructible from a confusion matrix. Note also "
            "that `file_name` is NOT unique in this corpus (968 distinct names "
            "for 15,488 image records), so any join keyed on it silently drops "
            "~88% of the questions; the adapter keys on `image_id` instead."
        ),
        extras={"held_out_splits": ["Test", "Test2"]},
    ),
)


def declared_benchmark_names() -> tuple[str, ...]:
    """The declared benchmark names, in declaration order."""
    return tuple(b.name for b in DECLARED_BENCHMARKS)


def describe_declared(benchmark: DeclaredBenchmark) -> CorpusDescription:
    """Measure one declared benchmark's corpus presence on disk.

    Counts files and bytes when the root exists, so a partially-populated corpus
    is visible as "present but small" rather than as a bare yes/no. Absence is
    reported with the path, because "not available" without a location is not
    actionable.
    """
    root = benchmark.resolved_root
    if not root.exists():
        return CorpusDescription(
            benchmark=benchmark.name,
            available=False,
            root=str(root),
            reason=f"corpus root does not exist: {root}",
            note=benchmark.note,
        )

    n_files = 0
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            n_files += 1
            try:
                total += path.stat().st_size
            except OSError:
                # A file that vanished mid-scan or is unreadable contributes its
                # existence but not a size. Recording 0 bytes for it is honest;
                # inventing a size would not be.
                pass

    return CorpusDescription(
        benchmark=benchmark.name,
        available=n_files > 0,
        root=str(root),
        n_files=n_files,
        bytes=total,
        kinds={},
        reason=None if n_files > 0 else f"corpus root exists but is empty: {root}",
        note=benchmark.note,
    )


def declared_inventory() -> dict[str, Any]:
    """Every declared benchmark, with present/absent measured.

    The headline `available` count is the figure a report should quote, because
    it is derived from the filesystem rather than from intent.
    """
    entries = {}
    for benchmark in DECLARED_BENCHMARKS:
        payload = benchmark.to_dict()
        payload["corpus"] = describe_declared(benchmark).to_dict()
        entries[benchmark.name] = payload

    available = sorted(n for n, e in entries.items() if e["corpus"]["available"])
    return {
        "declared": list(declared_benchmark_names()),
        "count": len(DECLARED_BENCHMARKS),
        "available": available,
        "unavailable": sorted(set(entries) - set(available)),
        "benchmarks": entries,
        "note": (
            "Declared benchmarks are declarations, not adapters. A declared "
            "benchmark with no registered adapter cannot be evaluated and will be "
            "reported NOT_RUN."
        ),
    }
