"""Benchmark adapters (plan section 65; checklist line "benchmark adapters").

The contract lives in `base.py`, the declarations in `declared.py`, the four real
adapters in `levir_cd.py` / `vrsbench.py` / `bigearthnet_s1.py` / `change_vqa.py`,
the held-out split support in `heldout.py`, and the one synthetic adapter in
`fixture_vqa.py`.

WHAT IS TRUE RIGHT NOW (revised 2026-09-23)
-------------------------------------------
The earlier version of this docstring said:

    * The registry is **empty**. No published benchmark dataset is present in
      this repository, and this package will not invent one.
    * Four benchmarks are **declared** ... Declarations can report absence; they
      cannot produce a score.

The first bullet's *conclusion* still holds -- the registry is still empty at
import, because registration is an explicit act -- but its *premise* was measured
and found false. Three of the four corpora are on disk, and they predate this
package:

    levir_cd        data/levir/                         2,048 test tiles
    vrsbench        training/data/vrsbench/             16,159 eval records
    bigearthnet_s1  data/bigearthnet_v2/reben/...       28,000 S1 patches
    change_vqa_test data/cdvqa/annotations/             39,686 Test questions

So the gap was never the data. It was the **scoring half**: nothing read those
corpora, and `DEGRADED` was unreachable in the runner even though `base.py`
documented it. Both are now fixed -- four adapters exist and `corpus_kind()`
makes `DEGRADED` a real, selectable status.

WHAT IS STILL TRUE AND STILL DELIBERATE
---------------------------------------
* **The registry is empty at import.** Nothing is registered as a side effect of
  importing this module. A pre-registered adapter would appear in every report's
  inventory as a benchmark that "ran"; the one adapter that could be registered
  without a corpus is the synthetic fixture, and registering *that* is exactly
  the misreading this package exists to prevent.
* **`load()` still refuses rather than fabricating.** Every adapter's `load()`
  raises `BenchmarkNotAvailableError` when its corpus is absent. There is no
  fallback generator anywhere in this package.
* **A declaration still cannot emit a score.** `DeclaredBenchmark` remains data
  with no `load()`. Its `adapter_registered` field is now *measured* against the
  live registry instead of hard-coded `False` -- a constant that would have
  become a lie the moment an adapter was registered.

HOW TO GET A RUN
----------------
Registration is explicit, so ask for it:

    from evaluation.benchmark_adapters import register_default_adapters
    register_default_adapters()          # four real adapters, no fixture

Each registered adapter then reports its own honest status through
`corpus_kind()`: `REAL` at its canonical root, `DEGRADED` where only a real
subset is present, `FIXTURE` never (these four read real corpora).

To exercise the runner's machinery without any corpus at all, register the
fixture explicitly and accept that its status is FIXTURE:

    from evaluation.benchmark_adapters import SyntheticVQAAdapter, register_adapter
    register_adapter(SyntheticVQAAdapter(n_samples=16))   # status will be FIXTURE
"""

from evaluation.benchmark_adapters._common import (
    AdapterDataError,
    CorpusLocation,
    as_split_name,
    check_identity_leakage,
    digest_entries,
    environment_block,
    manifest_hash,
    provenance_block,
    resolve_corpus_root,
    sample_manifest,
    sha256_file,
    sha256_text,
    subset_policy_block,
    utc_now,
)
from evaluation.benchmark_adapters.base import (
    AdapterError,
    AdapterNotRegisteredError,
    BenchmarkAdapter,
    BenchmarkNotAvailableError,
    BenchmarkSample,
    BenchmarkSource,
    BenchmarkStatus,
    CorpusDescription,
    clear_registry,
    get_adapter,
    inventory,
    register_adapter,
    registered_adapters,
    unregister_adapter,
    unregistered,
)
from evaluation.benchmark_adapters.bigearthnet_s1 import BigEarthNetS1Adapter
from evaluation.benchmark_adapters.change_vqa import ChangeVqaAdapter
from evaluation.benchmark_adapters.declared import (
    DECLARED_BENCHMARKS,
    DeclaredBenchmark,
    declared_benchmark_names,
    declared_inventory,
    describe_declared,
)
from evaluation.benchmark_adapters.fixture_vqa import (
    SyntheticVQAAdapter,
    predict_from_payload,
)
from evaluation.benchmark_adapters.heldout import (
    RESOURCE_BLOCKED,
    HeldOutRecord,
    HeldOutState,
    build_held_out_record,
    held_out_status,
    held_out_summary,
)
from evaluation.benchmark_adapters.levir_cd import LevirCdAdapter
from evaluation.benchmark_adapters.scorecard import (
    SCORECARD_STATES,
    SCORE_BEARING_STATES,
    STATE_MEANINGS,
    ScorecardRow,
    build_scorecard,
    render_scorecard,
)
from evaluation.benchmark_adapters.vrsbench import VrsBenchAdapter

__all__ = [
    # contract
    "BenchmarkAdapter",
    "BenchmarkStatus",
    "BenchmarkSample",
    "BenchmarkSource",
    "CorpusDescription",
    "AdapterError",
    "BenchmarkNotAvailableError",
    "AdapterNotRegisteredError",
    "AdapterDataError",
    # registry
    "register_adapter",
    "unregister_adapter",
    "get_adapter",
    "registered_adapters",
    "clear_registry",
    "inventory",
    "unregistered",
    "register_default_adapters",
    "DEFAULT_ADAPTER_FACTORIES",
    # declarations
    "DeclaredBenchmark",
    "DECLARED_BENCHMARKS",
    "declared_benchmark_names",
    "describe_declared",
    "declared_inventory",
    # the four real adapters
    "LevirCdAdapter",
    "VrsBenchAdapter",
    "BigEarthNetS1Adapter",
    "ChangeVqaAdapter",
    # held-out split support
    "HeldOutState",
    "HeldOutRecord",
    "RESOURCE_BLOCKED",
    "held_out_status",
    "build_held_out_record",
    "held_out_summary",
    # the project-level scorecard
    "SCORECARD_STATES",
    "STATE_MEANINGS",
    "SCORE_BEARING_STATES",
    "ScorecardRow",
    "build_scorecard",
    "render_scorecard",
    # shared adapter machinery
    "CorpusLocation",
    "utc_now",
    "sha256_file",
    "sha256_text",
    "digest_entries",
    "sample_manifest",
    "manifest_hash",
    "environment_block",
    "provenance_block",
    "resolve_corpus_root",
    "check_identity_leakage",
    "subset_policy_block",
    "as_split_name",
    # the synthetic adapter, which must be registered by hand
    "SyntheticVQAAdapter",
    "predict_from_payload",
]


# ---------------------------------------------------------------------------
# The default adapter set
# ---------------------------------------------------------------------------
#: Factories for the four adapters that read REAL corpora. Held as zero-argument
#: callables rather than instances so that registering them cannot share mutable
#: per-instance caches (each adapter memoises its resolved root and, for
#: BigEarthNet, its patch list) between two independent runs.
#:
#: The synthetic fixture is deliberately absent from this tuple. Including it
#: would mean `register_default_adapters()` quietly added a benchmark whose
#: scores describe a generator -- the exact confusion the FIXTURE status exists
#: to prevent, reintroduced by a convenience function.
DEFAULT_ADAPTER_FACTORIES: tuple[tuple[str, object], ...] = (
    ("levir_cd", LevirCdAdapter),
    ("vrsbench", VrsBenchAdapter),
    ("bigearthnet_s1", BigEarthNetS1Adapter),
    ("change_vqa_test", ChangeVqaAdapter),
)


def register_default_adapters(*, replace: bool = False) -> dict[str, BenchmarkAdapter]:
    """Register the four real-corpus adapters. Returns what was registered.

    This is the explicit act that `__init__`'s contract requires. It is a
    function rather than an import-time side effect for two reasons:

      1. A report that lists a registered adapter is making a claim about what
         was wired up. That claim should be traceable to a line of caller code,
         not to the fact that a module happened to be imported.
      2. Registering at import would make the empty-registry invariant
         untestable, and that invariant is the one that keeps "no benchmark was
         run" from being confused with "every benchmark passed".

    `replace=True` is required to supersede an existing registration, matching
    `register_adapter`'s rule: two adapters under one benchmark name means the
    report cannot say which one produced its numbers.
    """
    registered: dict[str, BenchmarkAdapter] = {}
    for name, factory in DEFAULT_ADAPTER_FACTORIES:
        adapter = factory()  # type: ignore[operator]
        register_adapter(adapter, replace=replace)
        registered[name] = adapter
    return registered
