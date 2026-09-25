"""Held-out / public-test split support (plan section 37, "Test Set Firewall").

THE ONE RULE THIS MODULE ENFORCES
---------------------------------
A held-out benchmark result may only be generated when the **actual immutable
evaluation corpus exists**. Everything else here is bookkeeping in service of
that rule:

  * if the corpus is absent, the outcome is `resource_blocked` -- a stated,
    machine-readable condition, not a zero and not a shrug;
  * if it is present, the result record carries the ten things a reader needs to
    reproduce it: dataset, split, sample count, manifest hash, source/version,
    preprocessing/config hash, model artifact hash, evaluation timestamp, metric
    definitions, and environment;
  * a synthetic fixture can exercise the *adapter*, and this module says so in
    the record -- it can never stand in for the corpus.

WHY THIS WRAPS RATHER THAN REIMPLEMENTS
---------------------------------------
`evaluation.public_test.corpus` already implements the plan's four firewall
rules as mechanisms: immutability via a seal, no-import via a bound root,
evaluation-mode via an unforgeable token, and no-answer-cache via a prohibition
that is checkable. Re-implementing any of that here would create a second
definition of "the test set is intact", which is precisely the guarantee that
must have exactly one definition.

This module's job is the *record*: turning "the corpus is available and sealed"
into the provenance block a published number is pinned to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.code_revision import REPO_ROOT
from evaluation.benchmark_adapters._common import (
    environment_block,
    provenance_block,
    sha256_file,
    utc_now,
)

__all__ = [
    "HeldOutState",
    "HeldOutRecord",
    "held_out_status",
    "build_held_out_record",
    "RESOURCE_BLOCKED",
]

#: The status string used when the corpus is genuinely unavailable. Spelled once
#: so every consumer compares against the same token.
RESOURCE_BLOCKED = "resource_blocked"


class HeldOutState(str):
    """String constants for held-out availability. Kept as plain strings so the
    value serialises directly into a report without an enum wrapper."""


#: The states a held-out evaluation can be in. Distinct on purpose: a missing
#: corpus and a present-but-unsealed corpus need different actions.
STATES: tuple[str, ...] = (
    "available_sealed",      # present and verified against its seal
    "available_unsealed",    # present but no seal yet -- not yet reproducible
    RESOURCE_BLOCKED,        # corpus absent: adapter may be complete, data is not
    "not_inspected",         # the caller did not ask
)


@dataclass
class HeldOutRecord:
    """The ten-field provenance record a held-out result must carry.

    Every field is populated or explicitly `None`/`unavailable ...`. Nothing is
    defaulted to a plausible value: a fabricated config hash in a held-out record
    is worse than an absent one, because it looks like a pinned configuration.
    """

    state: str
    dataset: str
    split: str
    root: str
    available: bool
    n_samples: int | None = None
    manifest_hash: str | None = None
    source_version: str | None = None
    config_hash: str | None = None
    model_artifact_hash: str | None = None
    evaluation_timestamp: str | None = None
    metric_definitions: dict[str, str] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    seal_digest: str | None = None
    reason: str | None = None
    synthetic_fixture_used: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "dataset": self.dataset,
            "split": self.split,
            "root": self.root,
            "available": self.available,
            "n_samples": self.n_samples,
            "manifest_hash": self.manifest_hash,
            "source_version": self.source_version,
            "config_hash": self.config_hash,
            "model_artifact_hash": self.model_artifact_hash,
            "evaluation_timestamp": self.evaluation_timestamp,
            "metric_definitions": dict(self.metric_definitions),
            "environment": dict(self.environment),
            "seal_digest": self.seal_digest,
            "reason": self.reason,
            "synthetic_fixture_used": self.synthetic_fixture_used,
            "note": self.note,
            "reproducible": bool(
                self.available
                and self.manifest_hash
                and self.config_hash
                and self.seal_digest
            ),
        }


def held_out_status(root: str | Path | None = None) -> dict[str, Any]:
    """Measure the held-out corpus's availability, without evaluating anything.

    Returns a dict rather than raising, because "is the test set there?" is a
    question a report must be able to answer with "no".
    """
    from evaluation.public_test.corpus import PUBLIC_TEST_ROOT, PublicTestCorpus

    target = Path(root) if root is not None else PUBLIC_TEST_ROOT
    try:
        corpus = PublicTestCorpus(root=target)
        description = corpus.describe()
    except Exception as exc:  # noqa: BLE001 -- a report must not die here
        return {
            "state": RESOURCE_BLOCKED,
            "root": str(target),
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "note": "the public-test corpus could not be inspected",
        }

    if not description.get("available"):
        return {
            "state": RESOURCE_BLOCKED,
            "root": description.get("root", str(target)),
            "available": False,
            "n_files": description.get("n_files", 0),
            "reason": description.get("reason")
            or "no corpus material is present",
            "note": (
                "An empty public-test corpus is reported as unavailable, not as "
                "a passing test set. Adapters may be complete while this stays "
                "blocked; that is a resource condition, not an implementation "
                "failure."
            ),
        }

    seal_present = (target / "CORPUS_SEAL.json").exists()
    return {
        "state": "available_sealed" if seal_present else "available_unsealed",
        "root": description.get("root", str(target)),
        "available": True,
        "n_files": description.get("n_files", 0),
        "bytes": description.get("bytes", 0),
        "seal_present": seal_present,
        "reason": None,
        "note": (
            "sealed: the corpus is pinned and verifiable"
            if seal_present
            else "no seal present, so a result from this corpus is not yet "
            "reproducible -- seal it before publishing a number"
        ),
    }


def build_held_out_record(
    *,
    dataset: str,
    split: str = "test",
    root: str | Path | None = None,
    n_samples: int | None = None,
    manifest: dict[str, Any] | None = None,
    config_hash: str | None = None,
    model_artifacts: dict[str, str | Path] | None = None,
    metric_definitions: dict[str, str] | None = None,
    synthetic_fixture_used: bool = False,
) -> HeldOutRecord:
    """Assemble the provenance record for a held-out evaluation.

    Args:
        dataset: the benchmark identity.
        split: the split label.
        root: the corpus root; defaults to the public-test root.
        n_samples: how many samples the evaluation covers, when known.
        manifest: a manifest from `_common.sample_manifest`, when one was built.
        config_hash: the config hash in force, or None.
        model_artifacts: `{name: path}` for every model artifact involved; each is
            hashed here so the record pins the exact weights.
        metric_definitions: metric name -> definition.
        synthetic_fixture_used: set True when the run used generated material. The
            record then says so and `reproducible` stays False, because a fixture
            cannot stand in for the corpus.

    Returns:
        A `HeldOutRecord`. When the corpus is absent the state is
        `resource_blocked` and `evaluation_timestamp` is **None** -- a timestamp
        on an evaluation that never happened would be a fabrication.
    """
    status = held_out_status(root)

    artifact_hashes: dict[str, str] = {}
    for name, path in (model_artifacts or {}).items():
        candidate = Path(path)
        if candidate.is_file():
            artifact_hashes[name] = sha256_file(candidate)
        else:
            artifact_hashes[name] = f"unavailable (not found: {candidate})"

    manifest_hash = None
    if manifest is not None:
        from evaluation.benchmark_adapters._common import manifest_hash as _mh

        manifest_hash = _mh(manifest)

    available = bool(status.get("available"))
    ran = available and not synthetic_fixture_used

    return HeldOutRecord(
        state=status.get("state", RESOURCE_BLOCKED),
        dataset=dataset,
        split=split,
        root=str(status.get("root", root or "")),
        available=available,
        n_samples=n_samples,
        manifest_hash=manifest_hash,
        source_version=(
            f"public-test corpus seal {status.get('seal_digest')}"
            if status.get("seal_digest")
            else None
        ),
        config_hash=config_hash,
        model_artifact_hash=(
            next(iter(artifact_hashes.values())) if len(artifact_hashes) == 1 else None
        ),
        evaluation_timestamp=utc_now() if ran else None,
        metric_definitions=dict(metric_definitions or {}),
        environment=environment_block() if ran else {},
        seal_digest=status.get("seal_digest"),
        reason=status.get("reason"),
        synthetic_fixture_used=synthetic_fixture_used,
        note=(
            "corpus present; record is reproducible"
            if ran and not synthetic_fixture_used
            else (
                "synthetic fixture used: the adapter is exercised, no benchmark "
                "result is claimed"
                if synthetic_fixture_used
                else "corpus unavailable: the adapter may be complete, but no "
                "held-out evaluation can be run or reported"
            )
        ),
    )


def held_out_summary(
    *,
    dataset: str = "public_test",
    split: str = "test",
    root: str | Path | None = None,
    config_hash: str | None = None,
    model_artifacts: dict[str, str | Path] | None = None,
    metric_definitions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The held-out block for a scorecard, without needing a manifest."""
    record = build_held_out_record(
        dataset=dataset,
        split=split,
        root=root,
        config_hash=config_hash,
        model_artifacts=model_artifacts,
        metric_definitions=metric_definitions,
    )
    return record.to_dict()
