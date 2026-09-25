"""Caption metrics — BLEU, ROUGE-L, CIDEr, BERTScore (plan section 2.2, ruling R-16).

WHY THIS MODULE IS SHAPED LIKE THIS
-----------------------------------
Plan `:257-285` (section 2.2, "Evaluation dimensions", under **Captioning**):

    Locally additionally calculate:
    BLEU / ROUGE-L / CIDEr / BERTScore

Ruling **R-16** governs how they are built. This module implements the ruling's
requirements as mechanisms rather than as good intentions:

  1. **Established implementations, not re-derivations.** Every metric delegates
     to a published third-party package. Nothing here re-implements BLEU's
     clipping or CIDEr's TF-IDF from scratch, because a mathematically
     equivalent private implementation is a different measurement that happens
     to share a name -- and it would be the number people quote.
  2. **Pinned versions.** `PINNED_VERSIONS` records the version each metric is
     expected to run against, and `caption_capability()` reports the version
     actually installed, so a mismatch is visible rather than silent.
  3. **Isolated optional dependencies.** No third-party import happens at module
     import time. `evaluation.metrics.vqa`'s docstring records why: BERTScore
     pulls torch, and a top-level import would poison the model-free paths that
     the API's CPU-only contract depends on. Every import is inside a function.
  4. **Importable without the dependencies.** This module imports cleanly in an
     environment where none of the four is installed -- which is the situation on
     this machine, and the situation the test suite exercises.
  5. **Explicit degradation, never a fake score.** A metric whose dependency is
     missing reports `available: False` with the reason and the package to
     install. `score_captions()` **raises** for a requested-but-unavailable
     metric rather than substituting another one. There is no fallback chain,
     because a silent fallback is how "BLEU" ends up meaning "token F1".
  6. **Never fabricate.** There is no code path here that produces a number
     without having executed a real implementation.
  7. **Declared requirements are VERIFIED, not listed** (added 2026-09-23). Point
     5 originally covered only the missing-package case, and the availability
     probe checked exactly one declared requirement (`java`). bert-score declares
     `("torch", "transformers", "network-or-cached-model")` and none of the three
     was checked, so it reported `available: True` on the strength of
     `find_spec("bert_score")` alone -- an importability claim being read as a
     computability claim. `_check_extra_requirement()` now verifies every
     declared requirement and reports each verdict in `requirement_checks`; an
     unrecognised requirement counts as UNVERIFIED, which is treated as
     unsatisfied rather than as fine.

STATE OF THE FOUR ON THIS MACHINE (measured 2026-09-23)
------------------------------------------------------
All four packages are now installed (`sacrebleu==2.4.3`, `rouge_score==0.1.2`,
`pycocoevalcap==1.2`, `bert-score==0.3.13`), so the "importable without the
dependencies" path described in point 4 is no longer the machine's situation --
it remains a supported and tested state, not the current one. Measured outcomes:

    bleu       AVAILABLE. Real corpus BLEU produced: 100.0 on identical input,
               41.14 on a near-match.
    rouge_l    AVAILABLE. Real ROUGE-L F produced: 1.0 on identical input,
               0.833 on a near-match.
    cider      UNAVAILABLE. pycocoevalcap is installed, but there is no `java` on
               PATH, so its PTB tokenizer cannot run.
    bertscore  UNAVAILABLE. The package and its `torch`/`transformers` stack are
               installed and the hub serves the checkpoint, but the local
               HuggingFace cache holds a corrupt `models--roberta-large` entry
               whose `config.json` is 0 bytes. `transformers` prefers a local
               snapshot over a download, so the poisoned entry makes the
               checkpoint unobtainable even though the network is fine. This is
               an ENVIRONMENT defect with a one-line remedy, reported in full by
               `caption_capability()['bertscore']['reason']`; it is not a defect
               in this module. BERTScore itself is exercisable by naming a
               supported checkpoint that is not poisoned, via
               `score_captions(..., model_type=...)`.

No caption score is published anywhere by this change. Producing one requires a
corpus of predictions and references, which is a separate act.

THE FOUR HAVE FOUR DIFFERENT BLOCKERS (this is the point of R-16)
----------------------------------------------------------------
R-16 was raised precisely because "the four have three different blockers". The
capability report keeps them separate rather than collapsing them into one
"captioning: unavailable", because the remedies differ:

    ROUGE-L    buildable now; `rouge_score` is pure Python
    BLEU       needs a pinned variant chosen (sacrebleu vs nltk differ)
    CIDEr      `pycocoevalcap` CIDEr needs a Java runtime for its PTB tokenizer
    BERTScore  pulls torch + transformers; must not reach a model-free path

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not decide whether the official caption definitions are settled -- the
plan itself records them as `UNVERIFIED` (`:4231-4237`), and R-16 is open. So
every score this module produces is labelled with the implementation and version
that produced it, and nothing here presents a caption number as an official
benchmark figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from core.errors import SatQueryError

__all__ = [
    "CAPTION_METRICS",
    "PINNED_VERSIONS",
    "CaptionMetricUnavailableError",
    "CaptionMetricError",
    "caption_capability",
    "score_captions",
    "caption_metric_info",
    "CaptionScore",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class CaptionMetricError(SatQueryError):
    """A caption metric could not be computed."""

    code = "caption_metric_error"
    user_message = "A caption metric could not be computed."


class CaptionMetricUnavailableError(CaptionMetricError):
    """A requested caption metric's implementation is not installed.

    Raised instead of substituting a different metric. A caller who asks for
    BLEU and silently receives token F1 has a number that is wrong in a way no
    report will show.
    """

    code = "caption_metric_unavailable"
    user_message = "The requested caption metric is not available in this environment."


# ---------------------------------------------------------------------------
# The four metrics, and what each is built on
# ---------------------------------------------------------------------------
CAPTION_METRICS: tuple[str, ...] = ("bleu", "rouge_l", "cider", "bertscore")

#: The version each metric is expected to run against. Pinned here so the
#: expectation is explicit and checkable; `caption_capability()` reports the
#: installed version beside it, and flags a mismatch.
PINNED_VERSIONS: dict[str, str] = {
    "bleu": "sacrebleu==2.4.3",
    "rouge_l": "rouge_score==0.1.2",
    "cider": "pycocoevalcap==1.2",
    "bertscore": "bert-score==0.3.13",
}


@dataclass(frozen=True)
class MetricSpec:
    """One caption metric and the established implementation behind it.

    Attributes:
        name: the metric key callers use.
        implementation: the package that computes it, named in every report.
        module: the importable module name, used for the availability probe.
        distribution: the pip distribution, used in the install hint.
        entry_point: the callable this module invokes, for inspection.
        note: why this implementation, and anything a user must know.
        extra_requirements: non-Python requirements (e.g. a JVM).
    """

    name: str
    implementation: str
    module: str
    distribution: str
    entry_point: str
    note: str = ""
    extra_requirements: tuple[str, ...] = ()
    #: The HuggingFace repo whose weights this metric needs, when it needs any.
    #: Declared so that "is the model obtainable?" is answerable for a NAMED
    #: artifact instead of for "a model" in the abstract -- the earlier version
    #: of this module declared `network-or-cached-model` and could not say which
    #: model, which is why the requirement went unverified.
    model_repo: str | None = None


#: The registry. Each entry names a real, published implementation.
_SPECS: dict[str, MetricSpec] = {
    "bleu": MetricSpec(
        name="bleu",
        implementation="sacrebleu",
        module="sacrebleu",
        distribution="sacrebleu",
        entry_point="sacrebleu.corpus_bleu",
        note=(
            "sacrebleu is the de-facto reproducible BLEU: it pins the "
            "tokenizer, the smoothing method and the n-gram order, and reports "
            "its own signature. Plain BLEU has several mutually incomparable "
            "variants, which is why R-16 asks for a PINNED variant rather than "
            "'BLEU' in the abstract. Reported as corpus BLEU (4-gram, exp "
            "smoothing, 13a tokenizer) -- the signature travels with the score."
        ),
    ),
    "rouge_l": MetricSpec(
        name="rouge_l",
        implementation="rouge_score",
        module="rouge_score",
        distribution="rouge_score",
        entry_point="rouge_score.rouge_scorer.RougeScorer",
        note=(
            "Google Research's rouge_score. ROUGE-L is the longest-common-"
            "subsequence F-measure; the package reports precision, recall and "
            "F separately, and this module returns the F-measure. No extra "
            "runtime requirement, which is why R-16 records it as buildable now."
        ),
    ),
    "cider": MetricSpec(
        name="cider",
        implementation="pycocoevalcap",
        module="pycocoevalcap.cider.cider",
        distribution="pycocoevalcap",
        entry_point="pycocoevalcap.cider.cider.Cider",
        note=(
            "pycocoevalcap's CIDEr, the reference implementation from the "
            "original CIDEr paper's authors. CIDEr is corpus-level: it needs "
            "the whole candidate/reference set together, because the TF-IDF "
            "weights are computed across the corpus. It is therefore NOT "
            "computable per-sentence and must not be averaged over sentences."
        ),
        extra_requirements=("java",),
    ),
    "bertscore": MetricSpec(
        name="bertscore",
        implementation="bert-score",
        module="bert_score",
        distribution="bert-score",
        entry_point="bert_score.score",
        note=(
            "bert-score (Zhang et al.). This is the metric that forces the "
            "lazy-import discipline: it pulls torch and transformers, so a "
            "top-level import here would drag a model stack into every "
            "model-free evaluation path. It also DOWNLOADS a model on first "
            "use, so it is not offline-safe by default; this module passes "
            "lang= and reports that the underlying model is a separate "
            "artifact."
        ),
        extra_requirements=("torch", "transformers", "cached-model-weights"),
        # bert-score's default for lang="en". Named here so the capability report
        # can check THIS checkpoint rather than "some model, somewhere".
        model_repo="roberta-large",
    ),
}


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------
def _installed_version(distribution: str) -> str | None:
    """The installed version of a distribution, or None if it is absent.

    Uses `importlib.metadata` rather than importing the module: the point is to
    answer "is this installed?" without paying the import cost (and, for
    bert_score, without loading torch).
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(distribution)
        except PackageNotFoundError:
            return None
    except Exception:  # noqa: BLE001
        return None


def _importable(module: str) -> tuple[bool, str | None]:
    """Whether `module` can be imported, without importing it.

    `importlib.util.find_spec` locates the module without executing it, which is
    what keeps this probe from loading torch. Returns `(ok, error_reason)`.
    """
    import importlib.util

    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError, ModuleNotFoundError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if spec is None:
        return False, "module not found on sys.path"
    return True, None


def _has_java() -> bool:
    """Whether a Java runtime is on PATH. CIDEr's tokenizer needs one."""
    import shutil

    return shutil.which("java") is not None


#: Memoised network reachability, per host, per process.
#:
#: Memoised for a specific reason: the probe is a TCP connect with a timeout, and
#: it sits on the path of `caption_capability()`, which a report may call many
#: times. Without the memo, asking "what can be computed?" would cost a socket
#: timeout on every call. Within one process the answer is effectively constant,
#: so caching it costs nothing in truthfulness and buys a lot in latency.
_NETWORK_PROBE: dict[str, bool] = {}


def _reset_probe_cache() -> None:
    """Forget the memoised network probe. For tests, and for a caller that has
    just changed the network state and wants a fresh answer."""
    _NETWORK_PROBE.clear()


def _network_reachable(host: str = "huggingface.co", *, timeout: float = 2.0) -> bool:
    """Whether a TCP connection to `host:443` can be opened right now.

    Deliberately a plain connect rather than an HTTP request: the question being
    answered is only "could a model download start?", and a connect is the
    cheapest honest way to ask it. A `False` here does NOT mean the metric is
    broken -- it means the weights cannot be fetched from the network, which
    matters only when they are also not cached.
    """
    if host in _NETWORK_PROBE:
        return _NETWORK_PROBE[host]
    import socket

    try:
        with socket.create_connection((host, 443), timeout=timeout):
            ok = True
    except OSError:
        ok = False
    _NETWORK_PROBE[host] = ok
    return ok


def _hf_hub_dir() -> "Path":
    """The HuggingFace hub cache directory, honouring the standard env vars."""
    import os
    from pathlib import Path

    home = Path(os.environ.get("HF_HOME") or (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "huggingface"
    ))
    return home / "hub"


def _cached_model_valid(repo_id: str) -> bool:
    """Whether `repo_id` is cached AND its `config.json` is non-empty valid JSON.

    The validity check is not belt-and-braces; it was added because a real cache
    entry on this machine was found to hold a **0-byte** `config.json`. A
    presence-only check would have reported "cached" for a checkpoint that cannot
    be loaded, and `available: True` would then have been a claim the very next
    call to the metric would falsify.

    Any snapshot with a parseable config counts. Snapshots are content-addressed,
    so a valid one is a valid one regardless of which revision it names.
    """
    import json
    from pathlib import Path

    slug = "models--" + repo_id.replace("/", "--")
    snapshots = _hf_hub_dir() / slug / "snapshots"
    if not snapshots.is_dir():
        return False
    try:
        candidates = [p for p in snapshots.iterdir() if p.is_dir()]
    except OSError:
        return False
    for snap in candidates:
        config = snap / "config.json"
        try:
            if config.stat().st_size <= 0:
                continue
            json.loads(config.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        return True
    return False


#: Memoised "does the hub serve this repo's config?" answers, per repo.
_HUB_CONFIG_PROBE: dict[str, bool] = {}


def _cached_model_defect(repo_id: str) -> str | None:
    """A description of a cache entry that EXISTS but is unusable, else None.

    Why this is separate from `_cached_model_valid`: "not cached" and "cached but
    broken" need different actions, and only the second one is a defect. This
    distinction was added after finding a real `models--roberta-large` snapshot on
    this machine holding a **0-byte** `config.json`.

    That case matters more than it looks. `transformers` prefers a local snapshot
    over a fresh download, so a poisoned cache entry makes a checkpoint
    unobtainable *even when the hub is serving it perfectly*. A check that only
    asked "is the hub up?" would have reported the weights as obtainable, and the
    very next call to the metric would have failed -- the exact overstatement this
    module is meant to prevent.
    """
    import json

    slug = "models--" + repo_id.replace("/", "--")
    snapshots = _hf_hub_dir() / slug / "snapshots"
    if not snapshots.is_dir():
        return None
    try:
        snaps = [p for p in snapshots.iterdir() if p.is_dir()]
    except OSError:
        return None
    if not snaps:
        return f"cache entry {slug!r} exists but holds no snapshot"

    problems: list[str] = []
    for snap in snaps:
        config = snap / "config.json"
        try:
            size = config.stat().st_size
        except OSError:
            problems.append(f"{snap.name}: config.json missing")
            continue
        if size <= 0:
            problems.append(f"{snap.name}: config.json is {size} bytes (empty)")
            continue
        try:
            json.loads(config.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"{snap.name}: config.json is not valid JSON ({exc})")
    if not problems:
        return None
    return (
        f"corrupt HuggingFace cache entry under {snapshots}: "
        + "; ".join(problems)
        + f". Remove {snapshots.parent} so the checkpoint can be re-downloaded."
    )


def _hub_serves_config(repo_id: str, *, timeout: float = 6.0) -> bool:
    """Whether the hub can actually serve `repo_id`'s `config.json` right now.

    Fetches the ~1 KB config rather than merely opening a socket, because a TCP
    connect to huggingface.co is NOT evidence that a checkpoint can be
    downloaded. On this machine the socket connects and the download still
    produces an empty file, so the weaker probe would have reported the weights
    as obtainable when they are not. Reading the config is the smallest request
    that actually distinguishes the two cases.

    Returns False on any failure -- DNS, TLS, HTTP status, timeout, or a body
    that is not JSON. A failure here is not fatal to the metric; it only means
    the weights must already be cached.
    """
    if repo_id in _HUB_CONFIG_PROBE:
        return _HUB_CONFIG_PROBE[repo_id]
    import json
    import urllib.error
    import urllib.request

    url = f"https://huggingface.co/{repo_id}/resolve/main/config.json"
    ok = False
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            body = response.read(65536)
        json.loads(body.decode("utf-8"))
        ok = True
    except (urllib.error.URLError, OSError, ValueError, UnicodeDecodeError):
        ok = False
    _HUB_CONFIG_PROBE[repo_id] = ok
    return ok


def model_obtainability_diagnostics(
    repo_id: str, *, probe_network: bool = True
) -> dict[str, Any]:
    """Facts about whether `repo_id`'s weights are usable HERE. Decides nothing.

    Kept deliberately separate from the availability verdict. The network facts
    are useful to a human diagnosing a blocked metric, but they must not decide
    `available`, because they were measured to be unreliable:

        hub reachable (TCP)            True
        hub serves a valid config.json True   (fetched via urllib, parsed OK)
        huggingface_hub writes         0-byte config.json   <-- the metric fails

    All three were observed for `roberta-large` and `distilbert-base-uncased` on
    2026-09-23. A verdict built on the first two would have said "available" and
    been wrong. So `available` is decided by the cache alone, and these facts are
    reported alongside it for whoever has to fix the environment.

    Args:
        repo_id: the HuggingFace repo, e.g. "roberta-large".
        probe_network: whether to perform the (memoised) network probes. False
            makes this a pure filesystem check.
    """
    defect = _cached_model_defect(repo_id)
    return {
        "repo_id": repo_id,
        "cached_valid": _cached_model_valid(repo_id),
        "cache_defect": defect,
        "hub_reachable": _network_reachable() if probe_network else None,
        "hub_serves_config": (
            _hub_serves_config(repo_id) if probe_network else None
        ),
        "note": (
            "Diagnostics only; `available` is decided by `cached_valid` alone. A "
            "reachable hub is NOT evidence that huggingface_hub can download the "
            "checkpoint -- measured to produce 0-byte config files here."
        ),
    }


def _check_extra_requirement(
    req: str,
    *,
    probe_network: bool = True,
    model_repo: str | None = None,
) -> str | None:
    """Verify ONE declared extra requirement. `None` when satisfied.

    Returns a human-readable reason when NOT satisfied, so the caller can put it
    straight into `missing_extra_requirements` and the reason string.

    This function exists because the previous version of this module declared
    `extra_requirements=("torch", "transformers", "network-or-cached-model")` for
    bert-score and then checked only `"java"`. The declared model requirement was
    therefore never verified, so bert-score reported `available: True` on the
    strength of `find_spec("bert_score")` alone -- a claim about importability
    being read as a claim about computability. That is the specific gap this
    closes.

    An UNRECOGNISED requirement is reported as missing rather than ignored. A
    requirement nobody knows how to check is an unverified requirement, and
    silently treating unverified as satisfied is how a capability report starts
    overstating what the environment can do.
    """
    if req == "java":
        return None if _has_java() else "java (required by the PTB tokenizer)"

    if req in ("torch", "transformers"):
        ok, err = _importable(req)
        return None if ok else f"{req} is not importable ({err})"

    if req in ("cached-model-weights", "network-or-cached-model"):
        # NOTE ON THE NAME. The requirement was originally declared as
        # `network-or-cached-model`, and the check accepted a reachable hub as
        # satisfying it. That was measured to be wrong on 2026-09-23: a TCP
        # connect succeeds, `urllib` fetches a perfectly valid config.json, and
        # `huggingface_hub` still writes a 0-BYTE config.json -- observed for
        # both `roberta-large` and `distilbert-base-uncased` on the same day.
        # So "the hub is reachable" is not evidence that the metric can run, and
        # a probe that treated it as evidence would have reported `available:
        # True` immediately before the metric failed.
        #
        # The requirement is therefore now satisfied ONLY by weights that are
        # already present and valid in the local cache. A capability probe must
        # not download: fetching weights is a multi-GB side effect, and a probe
        # that downloads cannot be called from a report.
        #
        # The old name is still accepted so a spec carrying it is not silently
        # treated as an unrecognised requirement.
        if model_repo is None:
            hub = _hf_hub_dir()
            cached = any(
                _cached_model_valid(p.name[len("models--"):].replace("--", "/"))
                for p in (hub.iterdir() if hub.is_dir() else [])
                if p.name.startswith("models--")
            )
            if cached:
                return None
            return (
                "cached-model-weights (no valid model in the HuggingFace cache; a "
                "capability probe does not download weights)"
            )

        if _cached_model_valid(model_repo):
            return None
        defect = _cached_model_defect(model_repo)
        if defect is not None:
            # A broken cache entry shadows any download, so the requirement is
            # NOT satisfied even when the hub would happily serve the file.
            return f"model weights for {model_repo!r} are blocked: {defect}"
        return (
            f"model weights for {model_repo!r} are not present in the local "
            f"HuggingFace cache, and a capability probe does not download them. "
            f"Fetch them first, then re-run this check."
        )

    return f"{req} (unrecognised requirement; not verified)"


def caption_metric_info(
    metric: str,
    *,
    probe_network: bool = True,
    model_override: str | None = None,
) -> dict[str, Any]:
    """Full availability record for one caption metric.

    Args:
        metric: one of `CAPTION_METRICS`.
        probe_network: when False, a requirement that needs a network check is
            treated as unsatisfied rather than probed. Callers who want a fast,
            conservative answer pass False; the default probes (memoised, so at
            most one connect per host per process).
        model_override: check THIS checkpoint instead of the spec's declared one.
            `score_captions(model_type=...)` uses it so the gate tests the model
            the caller actually asked for. Without it the gate checked the
            declared default and refused a request for a different, perfectly
            available checkpoint -- measured: `model_type='distilbert-base-uncased'`
            was rejected because `roberta-large` was poisoned.

    Raises:
        CaptionMetricError: `metric` is not one of the four.

    WHAT `available` MEANS (made explicit 2026-09-23)
    ------------------------------------------------
    `available` is True only when the implementation is importable AND every
    requirement the spec declares has been positively verified. The per-requirement
    verdicts are returned in `requirement_checks`, so a consumer can see which
    requirement was decisive instead of having to infer it from a single boolean.
    Before this change only `java` was checked, so bert-score's model requirement
    was declared and never verified.
    """
    spec = _SPECS.get(str(metric).strip().lower())
    if spec is None:
        raise CaptionMetricError(
            f"unknown caption metric {metric!r}; known metrics are "
            f"{list(CAPTION_METRICS)}"
        )

    importable, import_error = _importable(spec.module)
    installed = _installed_version(spec.distribution)

    # The checkpoint actually being asked about: the caller's override if given,
    # otherwise the spec's declared default.
    effective_model = model_override or spec.model_repo

    # Each declared requirement is verified, not merely listed. `None` means
    # satisfied; a string is the reason it is not.
    requirement_checks: dict[str, str | None] = {
        req: _check_extra_requirement(
            req, probe_network=probe_network, model_repo=effective_model
        )
        for req in spec.extra_requirements
    }
    missing_extra: list[str] = [
        reason for reason in requirement_checks.values() if reason is not None
    ]

    available = importable and not missing_extra

    reasons: list[str] = []
    if not importable:
        reasons.append(
            f"python package {spec.distribution!r} is not installed "
            f"({import_error})"
        )
    reasons.extend(missing_extra)

    model_diag = (
        model_obtainability_diagnostics(effective_model, probe_network=probe_network)
        if effective_model
        else None
    )
    pinned = PINNED_VERSIONS.get(spec.name)
    pinned_version = _pin_version(pinned)
    return {
        "metric": spec.name,
        "available": available,
        # What the boolean above actually asserts, stated so it cannot be
        # over-read as "a score has been produced" or as "the metric will run on
        # this machine's hardware".
        "availability_scope": (
            "implementation importable AND every declared requirement verified; "
            "says nothing about whether a score has been computed"
        ),
        "implementation": spec.implementation,
        "entry_point": spec.entry_point,
        "module": spec.module,
        "distribution": spec.distribution,
        "installed_version": installed,
        "pinned": pinned,
        "pinned_version": pinned_version,
        # Exact version equality. Was a substring test (`installed in pinned`),
        # which would have accepted "1.2" against a pin of "==1.20".
        "version_matches_pin": (
            None if installed is None else installed == pinned_version
        ),
        "extra_requirements": list(spec.extra_requirements),
        "missing_extra_requirements": missing_extra,
        # The named artifact this metric needs, when it needs one. None for the
        # three metrics that are pure computation. `model_repo` is the checkpoint
        # that was actually CHECKED; `declared_model_repo` is the spec's default,
        # and they differ when the caller passed a `model_type` override.
        "model_repo": effective_model,
        "declared_model_repo": spec.model_repo,
        "model_repo_overridden": bool(
            model_override and model_override != spec.model_repo
        ),
        # Network/cache facts for a human fixing the environment. NOT the basis of
        # `available` -- see `model_obtainability_diagnostics`.
        "model_diagnostics": model_diag,
        # Per-requirement verdicts: None = verified satisfied, str = why not.
        "requirement_checks": dict(requirement_checks),
        "reason": None if available else "; ".join(reasons),
        "install_hint": (
            None
            if available
            else f"pip install {pinned or spec.distribution}"
        ),
        "note": spec.note,
    }


def _pin_version(pinned: str | None) -> str | None:
    """The bare version from a `name==version` pin, or None.

    Kept separate from the comparison so the pin format lives in one place: the
    `PINNED_VERSIONS` table is written as `distribution==version` because that is
    what a reader needs to copy into a requirements file, but comparing against
    that string directly would compare a version to a distribution spec.
    """
    if not pinned or "==" not in pinned:
        return None
    return pinned.split("==", 1)[1].strip() or None


def caption_capability(*, probe_network: bool = True) -> dict[str, dict[str, Any]]:
    """Availability of all four caption metrics, each reported separately.

    Returns a dict keyed by metric name. Deliberately NOT a single boolean: R-16
    exists because the four have different blockers, and a single flag would
    destroy exactly the distinction the ruling was raised to capture.

    `probe_network=False` gives a fast, conservative answer that treats an
    unverifiable model requirement as unsatisfied. The default probes once per
    host per process (memoised), so repeated calls are cheap.
    """
    return {
        name: caption_metric_info(name, probe_network=probe_network)
        for name in CAPTION_METRICS
    }


def caption_metric_info_or_none(
    metric: str,
    *,
    probe_network: bool = True,
    model_override: str | None = None,
) -> dict[str, Any] | None:
    """`caption_metric_info` that returns None instead of raising on a bad name."""
    try:
        return caption_metric_info(
            metric, probe_network=probe_network, model_override=model_override
        )
    except CaptionMetricError:
        return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@dataclass
class CaptionScore:
    """One metric's result, carrying the implementation that produced it."""

    metric: str
    value: float | None
    implementation: str
    version: str | None
    n: int
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value,
            "implementation": self.implementation,
            "version": self.version,
            "n": self.n,
            "detail": dict(self.detail),
        }


def _score_bleu(preds: Sequence[str], refs: Sequence[str]) -> tuple[float, dict[str, Any]]:
    """Corpus BLEU via sacrebleu, with its own signature recorded."""
    import sacrebleu

    result = sacrebleu.corpus_bleu(list(preds), [list(refs)])
    return float(result.score), {
        "signature": str(getattr(result, "format", lambda *a, **k: "")()),
        "score_scale": "0-100 (sacrebleu's native scale)",
        "tokenizer": "13a (sacrebleu default)",
        "smoothing": "exp (sacrebleu default)",
    }


def _score_rouge_l(preds: Sequence[str], refs: Sequence[str]) -> tuple[float, dict[str, Any]]:
    """Mean ROUGE-L F-measure via rouge_score."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(preds, refs)
    ]
    return (sum(scores) / len(scores) if scores else 0.0), {
        "aggregation": "mean of per-item rougeL F-measure",
        "use_stemmer": True,
        "score_scale": "0-1",
    }


def _score_cider(preds: Sequence[str], refs: Sequence[str]) -> tuple[float, dict[str, Any]]:
    """Corpus CIDEr via pycocoevalcap. Corpus-level, not per-sentence."""
    from pycocoevalcap.cider.cider import Cider

    gts = {i: [ref] for i, ref in enumerate(refs)}
    res = {i: [pred] for i, pred in enumerate(preds)}
    score, per_item = Cider().compute_score(gts, res)
    return float(score), {
        "aggregation": "corpus-level (CIDEr computes TF-IDF across the corpus)",
        "per_item_mean": float(per_item.mean()) if hasattr(per_item, "mean") else None,
        "score_scale": "0-10 typical (unbounded above)",
    }


def _score_bertscore(
    preds: Sequence[str],
    refs: Sequence[str],
    *,
    lang: str = "en",
    model_type: str | None = None,
) -> tuple[float, dict[str, Any]]:
    """Mean BERTScore F1 via bert-score.

    `lang` is required by bert-score and is passed explicitly rather than left to
    the library's default, so the model actually used is a declared choice.
    `model_type` overrides the checkpoint that `lang` would otherwise select.

    The override exists for a concrete reason: bert-score's default for `lang="en"`
    is `roberta-large`, and when that checkpoint cannot be obtained the metric is
    unusable even though the implementation is fine. Naming the checkpoint lets a
    caller point at one that IS available, and lets the capability report state
    which artifact the score depends on instead of "some model".

    `num_layers` and `batch_size` are recorded in the detail because BERTScore's
    value depends on which layer's embeddings are used; leaving them implicit
    would make two BERTScore numbers incomparable without saying so.
    """
    from bert_score import score as bert_score

    if model_type:
        # bert-score resolves the checkpoint through its own `model2layers` table,
        # so an unsupported name raises a bare `KeyError: '<name>'` from deep
        # inside the library. Measured here: `model_type='sentence-transformers/
        # all-MiniLM-L6-v2'` -- a perfectly valid HuggingFace repo -- raises
        # exactly that, because bert-score does not know which of its layers to
        # use for it. Translating it into a named error with the constraint
        # spelled out is the difference between a five-second fix and a search
        # through bert-score's source.
        from bert_score.utils import model2layers

        if model_type not in model2layers:
            supported = sorted(model2layers)
            raise CaptionMetricError(
                f"bert-score does not support model_type={model_type!r}: it "
                f"resolves checkpoints through its own model2layers table, which "
                f"holds {len(supported)} entries and not this one. A valid "
                f"HuggingFace repo is NOT sufficient. Examples it does support: "
                f"{supported[:6]}. Omit model_type to use bert-score's default "
                f"for lang={lang!r}."
            )

    kwargs: dict[str, Any] = {"lang": lang, "verbose": False}
    if model_type:
        kwargs["model_type"] = model_type
    precision, recall, f1 = bert_score(list(preds), list(refs), **kwargs)
    return float(f1.mean()), {
        "aggregation": "mean of per-item BERTScore F1",
        "lang": lang,
        "model_type": model_type or f"bert-score default for lang={lang!r}",
        "score_scale": "0-1",
        "num_layers": None,
        "batch_size": None,
        "note": (
            "The underlying checkpoint is a separate artifact from the bert-score "
            "package: the package can be installed while the weights are not "
            "obtainable. The score describes whatever checkpoint ran, which is why "
            "it is recorded here."
        ),
    }


_SCORERS: dict[str, Callable[..., tuple[float, dict[str, Any]]]] = {
    "bleu": _score_bleu,
    "rouge_l": _score_rouge_l,
    "cider": _score_cider,
    "bertscore": _score_bertscore,
}


def score_captions(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    metrics: Sequence[str] = CAPTION_METRICS,
    model_type: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute the requested caption metrics.

    Args:
        predictions: candidate captions.
        references: the matching references, same length and order.
        metrics: which metrics to compute. Defaults to all four.
        model_type: BERTScore checkpoint override. Ignored by the other three
            metrics, which have no model. When None, bert-score's default for
            `lang` applies.

    Returns:
        `{metric: CaptionScore.to_dict()}`. A metric that is not available is
        **absent** from the result, and its reason is available from
        `caption_capability()`. Callers who need to know must ask; this function
        does not silently substitute.

    Raises:
        CaptionMetricUnavailableError: a REQUESTED metric's implementation is not
            installed. Raised rather than skipped, because a caller who asked for
            BLEU and received only ROUGE-L has a silently different measurement.
        CaptionMetricError: `predictions` and `references` differ in length, or
            an unknown metric name was requested.
    """
    if len(predictions) != len(references):
        raise CaptionMetricError(
            f"predictions and references must be aligned: "
            f"{len(predictions)} vs {len(references)}"
        )

    if len(predictions) == 0:
        # FIXED 2026-09-23. Before this, an empty corpus reached the scorers and
        # the four behaved differently: sacrebleu raised a bare
        # `IndexError: list index out of range` from deep inside the library,
        # while `_score_rouge_l` returned a 0.0 that no computation had produced.
        # Neither is acceptable -- one is a crash the caller cannot diagnose, the
        # other is a fabricated number wearing a metric's name.
        #
        # A metric over zero pairs is undefined, so this refuses. It does NOT
        # return 0.0, because 0.0 is a legitimate score for a bad caption set and
        # an empty set is a different situation that must not share its value.
        raise CaptionMetricError(
            "cannot compute caption metrics over an empty corpus: 0 predictions "
            "and 0 references were supplied. An empty corpus is not a score of "
            "zero; it is a call with nothing to measure."
        )

    unknown = [m for m in metrics if str(m).strip().lower() not in _SPECS]
    if unknown:
        raise CaptionMetricError(
            f"unknown caption metric(s) {unknown}; known metrics are "
            f"{list(CAPTION_METRICS)}"
        )

    out: dict[str, dict[str, Any]] = {}
    for raw in metrics:
        name = str(raw).strip().lower()
        # The gate must test the checkpoint the caller actually asked for, not the
        # spec's declared default -- otherwise a request for a different, healthy
        # checkpoint is refused on account of the default being unavailable.
        info = caption_metric_info(
            name, model_override=model_type if name == "bertscore" else None
        )
        if not info["available"]:
            raise CaptionMetricUnavailableError(
                f"caption metric {name!r} is not available: {info['reason']}. "
                f"Install with: {info['install_hint']}. No substitute metric is "
                f"returned, because a substituted number would be reported under "
                f"the requested metric's name.",
                context=info,
            )

        # Only BERTScore takes a checkpoint; the other three are pure computation
        # and would reject an unexpected keyword.
        kwargs: dict[str, Any] = (
            {"model_type": model_type} if name == "bertscore" else {}
        )
        value, detail = _SCORERS[name](list(predictions), list(references), **kwargs)
        out[name] = CaptionScore(
            metric=name,
            value=float(value),
            implementation=info["implementation"],
            version=info["installed_version"],
            n=len(predictions),
            detail=detail,
        ).to_dict()

    return out


def caption_metrics_summary() -> dict[str, Any]:
    """A report-ready summary of what can and cannot be computed.

    The shape a documentation or scorecard consumer wants: the four metrics, the
    implementation and version for each, and an explicit statement that no
    caption number has been produced.
    """
    capability = caption_capability()
    return {
        "metrics": capability,
        "available": sorted(n for n, i in capability.items() if i["available"]),
        "unavailable": sorted(n for n, i in capability.items() if not i["available"]),
        "n_available": sum(1 for i in capability.values() if i["available"]),
        "n_total": len(CAPTION_METRICS),
        "ruling": "R-16 (owner rulings packet 2026-09-20, section 17)",
        "note": (
            "Each metric is reported separately because the four have different "
            "blockers. No caption score is produced while a metric's dependency "
            "is absent; the module raises rather than substituting."
        ),
    }
