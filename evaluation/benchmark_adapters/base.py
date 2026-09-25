"""Benchmark adapter contract (plan section 65, checklist line "benchmark adapters").

WHY THIS FILE IS MOSTLY ERRORS
------------------------------
The plan names several external benchmarks the system should eventually be
measured against. **None of their datasets are in this repository.** There is no
LEVIR-CD tree, no VRSBench tree, no BigEarthNet tree, no benchmark annotation
file of any kind under `data/`.

That creates a specific and dangerous temptation: write an adapter that loads
"samples" from a synthetic generator, run it, and report a number. The number
would be real, arithmetic would be correct, and it would be completely
meaningless -- it would describe a generator, not a benchmark. The plan's
checklist line "benchmark adapters" would then read as satisfied while the
system remained unmeasured.

This module removes that failure mode **structurally** rather than by policy:

  * `BenchmarkAdapter.load()` is the ONLY way to obtain benchmark samples, and
    its base implementation raises `BenchmarkNotAvailableError`. An adapter that
    has no real dataset does not "return nothing" -- it refuses, loudly.
  * `load()` is abstract, so an adapter subclass MUST state where its samples
    come from. The concrete implementation must point at a real corpus root.
  * Registration is explicit and per-adapter. Nothing is registered by default,
    because nothing can be satisfied by default.
  * `require_available()` is what the runner calls. An unavailable benchmark
    becomes `BenchmarkStatus.NOT_RUN` in the report -- a visible gap rather than
    a fabricated row.

A caller therefore cannot accidentally evaluate a benchmark that does not exist.
If they want a number, they must first supply the dataset and WITNESS its
presence via `describe_corpus()`, which reports what is actually on disk.

STATUS VOCABULARY
-----------------
The runner reports exactly one status per benchmark, from `BenchmarkStatus`:

  NOT_RUN   nothing was executed (no dataset, no adapter, or the caller skipped it)
  FAILED    execution was attempted and raised
  DEGRADED  execution completed against a corpus the adapter itself flags as
            not the official one (e.g. a subset, a proxy, a partial download)
  FIXTURE   execution completed against synthetic/self-generated material. The
            number describes the fixture and MUST NOT be read as a benchmark score
  REAL      execution completed against a corpus the adapter verifies as genuine

`FIXTURE` exists so that self-testing is possible without pretending. The status
travels with the result into the report, so a fixture number can never be quoted
as if it were a real one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from core.errors import SatQueryError

__all__ = [
    "BenchmarkStatus",
    "BenchmarkNotAvailableError",
    "AdapterNotRegisteredError",
    "BenchmarkSource",
    "CorpusDescription",
    "BenchmarkSample",
    "BenchmarkAdapter",
    "register_adapter",
    "get_adapter",
    "registered_adapters",
    "unregister_adapter",
    "clear_registry",
]


# ---------------------------------------------------------------------------
# Status vocabulary
# ---------------------------------------------------------------------------
class BenchmarkStatus(str, Enum):
    """The outcome of one benchmark attempt. Exactly one applies.

    Being a `str` enum keeps the value JSON-serialisable as-is, so a report
    written to disk carries the human-readable word rather than an integer.
    """

    NOT_RUN = "NOT_RUN"
    FAILED = "FAILED"
    DEGRADED = "DEGRADED"
    FIXTURE = "FIXTURE"
    REAL = "REAL"

    @property
    def is_scored(self) -> bool:
        """Whether a status carries metrics that describe a real benchmark.

        Only `REAL` does. `DEGRADED` describes a corpus that is not the official
        one, and `FIXTURE` describes self-generated material; both are useful for
        engineering, neither may be reported as a benchmark score.
        """
        return self is BenchmarkStatus.REAL


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class AdapterError(SatQueryError):
    """An adapter could not be used as asked."""

    code = "adapter_error"
    user_message = "A benchmark adapter could not be used."


class BenchmarkNotAvailableError(AdapterError):
    """The benchmark's corpus is not present, so it cannot be evaluated.

    This is the non-invention guard. It is raised instead of returning an empty
    or synthetic sample list, because an empty list looks like "the benchmark has
    no samples" and a synthetic list looks like data.
    """

    code = "benchmark_not_available"
    user_message = "The benchmark dataset is not available in this environment."


class AdapterNotRegisteredError(AdapterError):
    """No adapter is registered under the requested name."""

    code = "adapter_not_registered"
    user_message = "No benchmark adapter is registered under that name."


# ---------------------------------------------------------------------------
# Corpus description
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchmarkSource:
    """Where a benchmark's material is supposed to come from.

    Kept as data, not as prose, so a report can state the provenance of a claim
    without a human retyping it. `root` is the path that must exist for the
    benchmark to be evaluable at all.
    """

    name: str
    root: Path
    citation: str
    note: str = ""

    def exists(self) -> bool:
        return self.root.exists()


@dataclass
class CorpusDescription:
    """What is ACTUALLY on disk for a benchmark, measured at call time."""

    benchmark: str
    available: bool
    root: str
    n_files: int = 0
    bytes: int = 0
    kinds: dict[str, str] = field(default_factory=dict)
    reason: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "available": self.available,
            "root": self.root,
            "n_files": self.n_files,
            "bytes": self.bytes,
            "kinds": dict(self.kinds),
            "reason": self.reason,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# One benchmark item
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkSample:
    """One evaluable item, in the runner's internal shape.

    `payload` holds whatever the adapter's task needs (an image path, a pair of
    masks, a referring expression...). Keeping it an opaque dict means the runner
    never has to know how any particular benchmark is laid out; the adapter that
    produced the sample is the adapter that knows how to score it.

    `expected` is the gold annotation, present only when the corpus supplies one.
    A missing gold value (`None`) is recorded as missing rather than filled in.
    """

    sample_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    expected: Any = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise AdapterError("a benchmark sample must carry a sample_id")


# ---------------------------------------------------------------------------
# The adapter contract
# ---------------------------------------------------------------------------
class BenchmarkAdapter(ABC):
    """Base class for a benchmark adapter.

    Subclasses MUST implement `describe_corpus()` and `load()`, and MUST declare
    `metric_names()` so the runner knows which metric keys its results will
    carry. Nothing else is required, and nothing else is provided by default --
    there is deliberately no baseline implementation that "works" without a
    dataset.
    """

    #: The benchmark's registered name. Subclasses override.
    name: str = ""

    #: Where the benchmark's material lives, and how to cite it. Subclasses
    #: override with a real citation; the base value claims nothing.
    source: BenchmarkSource | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # A subclass that forgets to name itself would be registered under the
        # empty string and silently collide with the next such subclass.
        if cls.name == "" and not getattr(cls, "__abstractmethods__", None):
            raise AdapterError(
                f"{cls.__name__} does not set a `name`; an unnamed adapter would "
                f"be registered under '' and collide with any other unnamed adapter"
            )

    # -- corpus ------------------------------------------------------------
    @abstractmethod
    def describe_corpus(self) -> CorpusDescription:
        """Report what is actually present for this benchmark, measured.

        Must not raise for a missing corpus -- absence is the thing being
        reported. Must not claim `available=True` without having checked.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, *, split: str = "test") -> list[BenchmarkSample]:
        """Return the benchmark's samples for `split`.

        The base class does not provide an implementation, and the recommended
        implementation raises `BenchmarkNotAvailableError` when
        `describe_corpus().available` is False. Never return a synthetic list
        from here: a quiet fallback in exactly this method is how a fabricated
        benchmark score gets produced.
        """
        raise NotImplementedError

    # -- task shape --------------------------------------------------------
    @abstractmethod
    def metric_names(self) -> tuple[str, ...]:
        """The metric keys this adapter's scores will carry.

        Declared up front so the runner can normalise them through the plan
        section 63 registry. A metric name that is not in that registry will
        raise at normalisation time, which is the intended behaviour: an
        unregistered metric is a contract decision, not a guess.
        """
        raise NotImplementedError

    # -- status ------------------------------------------------------------
    def is_official(self) -> bool:
        """Whether the corpus this adapter reads is the benchmark's own corpus.

        Defaults to False. An adapter returns True only when it can substantiate
        it (e.g. a verified manifest digest). The runner maps False to
        `DEGRADED` rather than `REAL`, so the burden of proof sits here.
        """
        return False

    def corpus_kind(self) -> "BenchmarkStatus":
        """Which status this adapter's material warrants: REAL, DEGRADED or FIXTURE.

        Added 2026-09-23. Before this method existed the runner classified by a
        two-way branch on `is_official()`:

            BenchmarkStatus.REAL if adapter.is_official() else BenchmarkStatus.FIXTURE

        which made `DEGRADED` **unreachable** even though `BenchmarkStatus`
        documents it and `_STATUS_LEGEND` explains it as "a subset or proxy". A
        real-but-partial corpus -- and all FOUR of the declared benchmarks ship
        as exactly that here -- would therefore have been labelled `FIXTURE`,
        which claims the material is synthetic. That is a worse error than the
        one the status vocabulary was built to prevent.

        The default preserves the old two-way behaviour exactly, so no existing
        adapter changes meaning. An adapter reading a real subset overrides this
        to return `DEGRADED`.
        """
        return (
            BenchmarkStatus.REAL
            if self.is_official()
            else BenchmarkStatus.FIXTURE
        )

    def attests_real_corpus(self) -> bool:
        """Whether this adapter's material is real (REAL or DEGRADED, not FIXTURE).

        Distinct from `is_official()`: a real subset is real but not official.
        Used by reports that need "did this describe actual benchmark material?"
        separately from "was it the whole benchmark?".
        """
        return self.corpus_kind() in (BenchmarkStatus.REAL, BenchmarkStatus.DEGRADED)

    def require_available(self, *, split: str = "test") -> list[BenchmarkSample]:
        """Load, or raise `BenchmarkNotAvailableError` with the reason.

        This is the method the runner calls. It converts an unavailable corpus
        into an explicit error carrying the description, so the report can record
        WHY the benchmark was not run instead of an unexplained blank.
        """
        description = self.describe_corpus()
        if not description.available:
            raise BenchmarkNotAvailableError(
                f"benchmark {self.name!r} is not available: "
                f"{description.reason or 'corpus not present'} "
                f"(root={description.root})",
                context=description.to_dict(),
            )
        return self.load(split=split)

    # -- reporting ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        description = self.describe_corpus()
        return {
            "name": self.name,
            "class": type(self).__name__,
            "available": description.available,
            "official": self.is_official(),
            "metrics": list(self.metric_names()),
            "source": (
                {
                    "citation": self.source.citation,
                    "root": str(self.source.root),
                    "exists": self.source.exists(),
                    "note": self.source.note,
                }
                if self.source is not None
                else None
            ),
            "corpus": description.to_dict(),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#: name -> adapter INSTANCE. Deliberately empty. Nothing is registered by
#: default because no benchmark dataset is present in this repository, and a
#: pre-registered adapter would have to fabricate its corpus to exist.
_REGISTRY: dict[str, BenchmarkAdapter] = {}


def register_adapter(adapter: BenchmarkAdapter, *, replace: bool = False) -> None:
    """Register an adapter instance under its own `name`.

    Duplicate registration is an error unless `replace=True`: two adapters
    claiming one benchmark name means the report cannot say which one produced
    its numbers.
    """
    name = adapter.name
    if not name:
        raise AdapterError(
            f"{type(adapter).__name__} has an empty `name` and cannot be registered"
        )
    if name in _REGISTRY and not replace:
        existing = type(_REGISTRY[name]).__name__
        raise AdapterError(
            f"an adapter named {name!r} is already registered ({existing}); pass "
            f"replace=True only when deliberately superseding it"
        )
    _REGISTRY[name] = adapter


def unregister_adapter(name: str) -> None:
    """Remove an adapter. Raises if it was not registered."""
    if name not in _REGISTRY:
        raise AdapterNotRegisteredError(
            f"no adapter registered as {name!r}; registered: {sorted(_REGISTRY)}",
            context={"registered": sorted(_REGISTRY)},
        )
    del _REGISTRY[name]


def clear_registry() -> None:
    """Empty the registry. Used by tests to isolate their own fixtures."""
    _REGISTRY.clear()


def get_adapter(name: str) -> BenchmarkAdapter:
    """Look up an adapter by name.

    Raises `AdapterNotRegisteredError` rather than returning None, so a typo in
    a benchmark name surfaces as an error instead of as a benchmark that quietly
    did not run.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise AdapterNotRegisteredError(
            f"no adapter registered as {name!r}; registered: {sorted(_REGISTRY)}",
            context={"registered": sorted(_REGISTRY)},
        ) from None


def registered_adapters() -> dict[str, BenchmarkAdapter]:
    """A copy of the registry, so callers cannot mutate it by accident."""
    return dict(_REGISTRY)


def inventory() -> dict[str, Any]:
    """What is registered and whether each can actually run.

    This is the honest answer to "which benchmarks are wired up?", and its
    usefulness is precisely that it can say "none of them are". Called by the
    runner so every report carries the boundary of what was measured.
    """
    items: dict[str, Any] = {}
    for name, adapter in sorted(_REGISTRY.items()):
        try:
            items[name] = adapter.to_dict()
        except Exception as exc:  # noqa: BLE001
            items[name] = {
                "name": name,
                "class": type(adapter).__name__,
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return {
        "registered": sorted(_REGISTRY),
        "count": len(_REGISTRY),
        "adapters": items,
        "note": (
            "An empty registry means no benchmark adapter is wired up. It does NOT "
            "mean every benchmark passed. Dataset availability is reported per "
            "adapter under `corpus`."
        ),
    }


def unregistered(names: Iterable[str]) -> list[str]:
    """Which of `names` have no registered adapter, sorted.

    Lets a report say "these benchmarks were requested and could not even be
    attempted" separately from "these were attempted and failed".
    """
    return sorted(n for n in names if n not in _REGISTRY)
