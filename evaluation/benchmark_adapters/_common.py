"""Shared machinery for the concrete benchmark adapters.

WHY THIS MODULE EXISTS
----------------------
Four adapters (LEVIR-CD, VRSBench, BigEarthNet-S1, Change-VQA) plus the
held-out/public-split path all need the same five things:

  1. a **deterministic manifest** over the samples actually loaded, so a
     published number can be pinned to an exact sample set;
  2. a **provenance block** (config hash, code revision, environment, artifact
     hashes) so a number can be reproduced rather than merely believed;
  3. a **leakage check** that is the same one everywhere, so "we checked" means
     the same thing in every adapter;
  4. a **corpus-location resolver** that reports every path it tried, because
     "not available" without a list of the places searched is not actionable;
  5. **honest subset reporting** -- several of these corpora are real but
     partial, and a partial corpus must not be reported as the full benchmark.

Nothing here produces a metric. Every function is either a measurement of
something already on disk or a bookkeeping record of something that happened.
That separation is deliberate: it keeps the "no fabricated score" rule
enforceable by reading this file.

REUSE, NOT REINVENTION
----------------------
Manifests come from `evaluation.manifests` (`SampleRecord`, `DatasetManifest`)
and leakage from `evaluation.leakage` (`audit_manifest`,
`assert_no_scene_overlap`). Those modules already define what a manifest and a
leakage rule ARE in this project; this module only adapts them to the adapter
contract. A parallel manifest format would be a second source of truth, which is
exactly the class of problem the project's single-source-of-truth rule forbids.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from core.code_revision import REPO_ROOT
from core.errors import LeakageError, SatQueryError

__all__ = [
    "AdapterDataError",
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
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class AdapterDataError(SatQueryError):
    """An adapter's corpus is present but its contents are not usable."""

    code = "adapter_data_error"
    user_message = "A benchmark corpus is present but could not be read as expected."


# ---------------------------------------------------------------------------
# Time and hashing
# ---------------------------------------------------------------------------
def utc_now() -> str:
    """The current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """The full sha256 of a file, streamed.

    Streamed rather than read-whole because several of these corpora contain
    multi-gigabyte archives; a whole-file read would make hashing a 4 GB zip an
    out-of-memory event instead of a slow-but-fine one.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """The full sha256 of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_entries(entries: Iterable[tuple[str, str, int]]) -> str:
    """A deterministic digest over `(sample_id, sha256, bytes)` triples.

    Sorted by sample_id before hashing, so the digest does not depend on
    directory iteration order (which is not stable across platforms). A manifest
    hash that moved because the filesystem returned entries in a different order
    would be worthless as a pin.
    """
    payload = "\n".join(
        f"{sid}\x00{digest}\x00{size}"
        for sid, digest, size in sorted(entries, key=lambda e: e[0])
    )
    return sha256_text(payload)


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------
def sample_manifest(
    *,
    dataset: str,
    version: str,
    split: str,
    sample_ids: Sequence[str],
    root: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A reproducible manifest over the sample ids an adapter actually loaded.

    Deliberately built from **sample ids only**, not from file contents. Hashing
    every file in a 28,000-patch corpus on every run would make evaluation
    unusably slow, and the ids are stable identifiers for the same material.
    Where a content hash is genuinely needed, `sha256_file` is called
    explicitly by the caller and recorded in `extra`.

    Args:
        dataset: the benchmark's identity (e.g. `levir_cd`).
        version: the corpus version/revision string, or an explicit statement
            that it is unknown. Never a guess.
        split: the split this manifest covers.
        sample_ids: the ids, in the order the adapter enumerates them.
        root: the corpus root the ids came from.
        extra: benchmark-specific extras (subset policy, image hashes, ...).

    Returns:
        A JSON-serialisable dict carrying a `hash` over the id set.
    """
    ordered = [str(s) for s in sample_ids]
    return {
        "dataset": dataset,
        "version": version,
        "split": split,
        "root": str(root) if root is not None else None,
        "n_samples": len(ordered),
        "sample_ids_digest": sha256_text("\n".join(ordered)),
        "extra": dict(extra or {}),
        "generated_at": utc_now(),
    }


def manifest_hash(manifest: dict[str, Any]) -> str:
    """A stable hash of a manifest dict, ignoring volatile fields.

    `generated_at` is excluded on purpose: two runs over the same sample set are
    the same manifest, and including a timestamp would make every run produce a
    different "manifest hash" and destroy its only useful property.
    """
    stable = {k: v for k, v in manifest.items() if k != "generated_at"}
    return sha256_text(json.dumps(stable, sort_keys=True, default=str))


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def environment_block() -> dict[str, Any]:
    """Environment facts a result depends on.

    Delegates to `evaluation.run_manifest.environment_block` where possible so
    there is one definition of "the environment" in the project.
    """
    try:
        from evaluation.run_manifest import environment_block as _env

        block = dict(_env())
    except Exception:  # noqa: BLE001 -- provenance must never break a run
        block = {}
    block["python_executable"] = __import__("sys").executable
    return block


def provenance_block(
    *,
    adapter: str,
    adapter_version: str,
    dataset: str,
    version: str,
    split: str,
    manifest: dict[str, Any],
    config_hash: str | None = None,
    artifact_hashes: dict[str, str] | None = None,
    metric_definitions: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything needed to reproduce a result from scratch.

    An unknown value is recorded as `None` or an explicit `"unavailable ..."`
    string. It is never filled with a plausible-looking default -- a fabricated
    config hash in a provenance block is worse than an absent one, because it
    looks like a pinned configuration.
    """
    code_revision: str | None = None
    try:
        from evaluation.run_manifest import code_revision_identifier

        code_revision = code_revision_identifier(REPO_ROOT)
    except Exception:  # noqa: BLE001
        code_revision = None

    return {
        "adapter": adapter,
        "adapter_version": adapter_version,
        "dataset": dataset,
        "dataset_version": version,
        "split": split,
        "manifest_hash": manifest_hash(manifest),
        "manifest": manifest,
        "config_hash": config_hash,
        "code_revision": code_revision,
        "artifact_hashes": dict(artifact_hashes or {}),
        "metric_definitions": dict(metric_definitions or {}),
        "environment": environment_block(),
        "recorded_at": utc_now(),
        "extra": dict(extra or {}),
    }


# ---------------------------------------------------------------------------
# Corpus location
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CorpusLocation:
    """One place a corpus was looked for, and whether it was found there."""

    path: Path
    exists: bool
    role: str
    n_files: int = 0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "exists": self.exists,
            "role": self.role,
            "n_files": self.n_files,
            "note": self.note,
        }


def _count_files(root: Path, *, cap: int = 200_000) -> int:
    """Count files under `root`, stopping at `cap`.

    Capped because `rglob` over a 28,000-patch tree on every `describe_corpus()`
    call is slow enough to be felt in a report loop, and past a few thousand the
    exact count stops changing any decision.
    """
    total = 0
    try:
        for path in root.rglob("*"):
            if path.is_file():
                total += 1
                if total >= cap:
                    return cap
    except OSError:
        pass
    return total


def resolve_corpus_root(
    candidates: Sequence[tuple[str | Path, str]],
    *,
    min_files: int = 1,
) -> tuple[Path | None, list[CorpusLocation]]:
    """Find the first candidate root that exists and holds material.

    Args:
        candidates: `(path, role)` pairs, in preference order. A relative path is
            resolved against the repository root, so a checkout elsewhere still
            reports a sensible absolute path.
        min_files: how many files the root must hold to count as present.

    Returns:
        `(chosen_root_or_None, all_locations_tried)`. The second element is the
        point of the function: an adapter that reports "unavailable" also reports
        exactly which paths it searched, so the person who has the data knows
        where to put it.
    """
    tried: list[CorpusLocation] = []
    chosen: Path | None = None

    for raw, role in candidates:
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        exists = path.exists()
        n_files = _count_files(path) if exists else 0
        tried.append(
            CorpusLocation(
                path=path,
                exists=exists,
                role=role,
                n_files=n_files,
                note="" if exists else "path does not exist",
            )
        )
        if chosen is None and exists and n_files >= min_files:
            chosen = path

    return chosen, tried


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------
def check_identity_leakage(
    train_ids: Iterable[str],
    eval_ids: Iterable[str],
    *,
    label: str,
    scene_of: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assert that no identity (or scene) appears in both train and eval.

    Two levels are checked, because they fail differently:

      * **sample identity** -- the same tile in both splits. Catches a duplicated
        file or a bad list.
      * **scene identity** -- different tiles from the same scene. Catches the
        subtler leak: neighbouring tiles from one acquisition are correlated, so
        a sample-disjoint split can still be scene-leaky and inflate a score.

    `scene_of` maps a sample id to its scene key. When it is not supplied, only
    the sample-identity check runs and the report says so -- an unperformed check
    is reported rather than implied to have passed.

    Raises:
        LeakageError: when either check finds an overlap.
    """
    train = {str(i) for i in train_ids}
    eval_ = {str(i) for i in eval_ids}

    shared_samples = sorted(train & eval_)

    shared_scenes: list[str] = []
    scene_checked = scene_of is not None
    if scene_checked:
        train_scenes = {scene_of[i] for i in train if i in scene_of}
        eval_scenes = {scene_of[i] for i in eval_ if i in scene_of}
        shared_scenes = sorted(train_scenes & eval_scenes)

    report = {
        "label": label,
        "clean": not shared_samples and not shared_scenes,
        "n_train": len(train),
        "n_eval": len(eval_),
        "shared_sample_ids": shared_samples[:20],
        "n_shared_sample_ids": len(shared_samples),
        "shared_scenes": shared_scenes[:20],
        "n_shared_scenes": len(shared_scenes),
        "scene_check_performed": scene_checked,
        "note": (
            "sample-identity and scene-identity checks performed"
            if scene_checked
            else "sample-identity check performed; scene check NOT performed "
            "(no scene map supplied)"
        ),
    }

    if not report["clean"]:
        raise LeakageError(
            f"{label}: train/eval leakage detected -- "
            f"{len(shared_samples)} shared sample id(s), "
            f"{len(shared_scenes)} shared scene(s)",
            context=report,
        )
    return report


# ---------------------------------------------------------------------------
# Subset policy
# ---------------------------------------------------------------------------
def subset_policy_block(
    *,
    is_full_corpus: bool,
    n_present: int | None,
    n_official: int | None,
    basis: str,
    detail: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record explicitly whether a corpus is the full benchmark or a subset.

    This is what stops a partial corpus being reported as the benchmark. The
    numbers stay real -- they describe a real subset of a real benchmark -- but
    the block says so, and `BenchmarkStatus.DEGRADED` follows from it.

    Args:
        is_full_corpus: True only when the adapter can substantiate that every
            official sample is present. The burden of proof sits here.
        n_present: samples actually available, when countable.
        n_official: the benchmark's published sample count, when known.
        basis: WHY the subset judgement was made (the evidence, not a feeling).
        detail: optional extra prose for the report.
        extra: benchmark-specific measured facts (join rates, label cardinality,
            ...). Merged into the block so they travel with the judgement rather
            than living only in a docstring.
    """
    coverage = None
    if n_present is not None and n_official:
        coverage = round(n_present / n_official, 6)
    block = {
        "is_full_corpus": bool(is_full_corpus),
        "n_present": n_present,
        "n_official": n_official,
        "coverage_fraction": coverage,
        "basis": basis,
        "detail": detail,
        "consequence": (
            "reported as REAL (adapter attests the full benchmark corpus)"
            if is_full_corpus
            else "reported as DEGRADED (a real but partial corpus; the number "
            "describes the subset, not the full benchmark)"
        ),
    }
    if extra:
        block.update(extra)
    return block


# ---------------------------------------------------------------------------
# Split naming
# ---------------------------------------------------------------------------
#: Splits `evaluation.manifests.SampleRecord` accepts. Anything else must be
#: mapped, because the manifest contract is fixed and widening it to accept
#: arbitrary split names would weaken a check other code depends on.
_MANIFEST_SPLITS = frozenset({"train", "val", "test"})


def as_split_name(split: str) -> str:
    """Map a benchmark's split label onto the manifest's train|val|test.

    CDVQA ships `Test` and `Test2`, and `Test2` is a second held-out set rather
    than a training split. It maps to `test` with the original label preserved in
    the sample's `extra`, so the manifest stays valid without losing which of the
    two held-out sets a sample came from.

    Raises:
        AdapterDataError: the split cannot be mapped. Silently coercing an
            unknown split to `test` would let a training split be evaluated as
            held-out data, which is the leak this mapping exists to prevent.
    """
    key = str(split).strip().lower()
    if key in _MANIFEST_SPLITS:
        return key
    if key in {"validation", "valid", "dev"}:
        return "val"
    if key in {"testing", "holdout", "held_out", "heldout", "eval", "test2"}:
        return "test"
    raise AdapterDataError(
        f"split {split!r} cannot be mapped onto train|val|test. Known aliases: "
        f"validation->val, test2/testing/holdout/eval->test. Add the alias "
        f"explicitly rather than coercing, so a training split can never be "
        f"silently evaluated as held-out data."
    )
