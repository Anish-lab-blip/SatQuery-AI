"""SatQuery AI — Phase 12 frozen-feature extraction (the missing producer).

WHY THIS MODULE EXISTS
----------------------
`training/fusion/train.py` trains a head over cached CROMA features, and
`write_feature_cache` writes that cache — but until this module existed nothing
outside a test ever *produced* one. `FusionFeature(...)` was constructed only
inside `tests/unit/test_fusion_training.py`, so the trainer had no upstream: a
real paired BigEarthNet-S1+S2 corpus could not reach it. This module is that
link.

    paired patches -> normalise -> resize -> CROMA (frozen) -> FusionFeature
                   -> write_feature_cache -> train_fusion_head

WHAT IT PRODUCES, AND WHAT IT IS NOT
------------------------------------
A cache of frozen features plus a JSON sidecar carrying the provenance needed
to tell WHICH arm produced it (`config_hash`, `use_8_bit`, the CROMA checkpoint
digest, the resolution, the label policy). No number this module produces is a
result; it moves data, it does not measure anything.

BOTH DEPENDENCIES ARE INJECTED
------------------------------
`encoder` is anything with `.encode(optical_batch, sar_batch)` returning an
object with `.as_forward_dict()` and `.dim` — the `CROMAEncoder` contract. The
band `loader` is `training.data.bigearthnet.BandLoader`. Both are parameters, so
the whole path runs against fakes with zero real data and zero CROMA.

`torch.no_grad()` IS STRUCTURAL, NOT A HOPE
-------------------------------------------
Every encode goes through `_encode_batch`, which is the only place in the module
that calls `.encode`. CROMA is frozen: building a gradient graph here would be
pure waste and a standing invitation to unfreeze the encoder by accident.

THE ENCODER MUST NOT NORMALISE A SECOND TIME
--------------------------------------------
This pipeline applies the DEV-2 encoder-input stretch itself
(`radiometry.normalise_for_croma`), because the arm's `use_8_bit` has to be
applied at exactly one place. `CROMAEncoder` applies the same stretch inside
`encode` when `normalize_input=True` (its default). Passing such an encoder here
would stretch twice — a silent distribution shift that produces a perfectly
well-shaped cache full of wrong numbers. `_assert_raw_encoder` refuses it by
name rather than trusting the caller; `build_extraction_encoder` constructs the
raw encoder this module expects.

THE MULTI-LABEL QUESTION IS AN OPEN DECISION, NOT A CONVENTION
--------------------------------------------------------------
BigEarthNet v2.0 is MULTI-LABEL: a patch carries several of the 19 CLC classes.
The frozen head is a single-label 19-class softmax with cross-entropy. Something
must collapse the label tuple to one index, and there is no defensible default —
"take the first" silently discards information and quietly biases the corpus.

So the collapse is an EXPLICIT parameter, `label_policy: Sequence[str] -> int |
None`. With no policy supplied:

  * exactly one label   -> its index (unambiguous; nothing was chosen)
  * more than one label -> `AmbiguousLabelError` naming the sample and the
                           labels. Never a silent pick.
  * zero labels         -> `None` -> the sample is SKIPPED and COUNTED. It is
                           never mapped to class 0: "we do not know" and "arable
                           land" are different facts and must not be merged.

The policy's name is recorded in the cache metadata, so a later reader can tell
which convention (if any) a cache was built under.

RESUMABILITY IS CONDITIONAL ON THE PROVENANCE MATCHING
------------------------------------------------------
Re-running with an existing cache loads it, skips the `sample_id`s already
present and appends the rest. Existing rows are preserved verbatim — the merge
is `existing + new`, never a replacement. The count of skipped rows is reported.

But the merged cache keeps ONE provenance record, so appending rows from a
different arm, resolution or config would produce a cache whose metadata
describes only half its rows. `check_resume_provenance` refuses that and names
the field and both values; `resume=False` rebuilds from scratch instead.

A DRY RUN MAKES THE SAME DECISION, NOT A SIMILAR ONE
----------------------------------------------------
`plan_extraction` answers "what would this run do" by calling the SAME
`select_samples` the run calls: the split firewall, the resume skip and the
label policy are applied once, in one place. So `plan.n_selected` is
`result.n_encoded` and `plan.n_would_write` is `result.n_total`.

That matters because BigEarthNet v2.0 is multi-label: on the real corpus the
number of ambiguous samples can be large, and a plan that counted candidates
before the policy would over-report by exactly that many — most wrong at the
moment it is most needed. A dry run also RAISES what the run would raise: an
ambiguous sample under `require_single_label` is an error in both.

THE WRITE IS ATOMIC-ISH, AND THAT IS DELIBERATE
-----------------------------------------------
`write_feature_cache` writes both files under temporary names and renames them
into place, so no reader ever sees a half-written `.npz`. The two renames cannot
be one operation; see that function's docstring for the residual window and why
it is detectable rather than silent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

from core.errors import ModelLoadError, SatQueryError
from specialists.optical_sar.fusion_head import DEFAULT_TASK_DIM, expected_fusion_dim
from specialists.optical_sar.inference import resize_to_canonical
from specialists.optical_sar.radiometry import (
    MODALITY_OPTICAL,
    MODALITY_SAR,
    describe_transform,
    normalise_for_croma,
    resolve_use_8_bit,
    summarise,
)
from training.data.bigearthnet import (
    CLC19_CLASSES,
    BandLoader,
    BigEarthNetError,
    PairedSample,
    pair_patches,
)
from training.fusion.train import (
    ARMS,
    CACHE_VERSION,
    Arm,
    FusionFeature,
    FusionTrainingError,
    assert_split_disjoint,
    read_feature_cache,
    resolve_arm,
    write_feature_cache,
)

# ---------------------------------------------------------------------------
# The frozen contract, declared ONCE
# ---------------------------------------------------------------------------
#
# The widths are not re-invented here: they are the inputs to
# `expected_fusion_dim`, which is the frozen concatenation's own definition, and
# the result is asserted against 2318 at import. A drift in either direction
# becomes an ImportError rather than a cache that trains to a meaningless number.

FROZEN_ENCODER_DIM = 768
FROZEN_OPTICAL_CHANNELS = 12
FROZEN_SAR_CHANNELS = 2
FROZEN_NUM_CLASSES = DEFAULT_TASK_DIM
FROZEN_FUSION_DIM = expected_fusion_dim(
    encoder_dim=FROZEN_ENCODER_DIM,
    optical_channels=FROZEN_OPTICAL_CHANNELS,
    sar_channels=FROZEN_SAR_CHANNELS,
)

if FROZEN_FUSION_DIM != 2318:  # pragma: no cover - a drift guard, not a branch
    raise ImportError(
        f"the frozen fusion width recomputed to {FROZEN_FUSION_DIM}, not 2318; "
        f"the constants above no longer describe the frozen concatenation "
        f"(3*768 + 12 + 2). Fix them before any cache is written."
    )

#: Layout version of the extraction metadata. Distinct from `CACHE_VERSION`,
#: which versions the array layout: this one versions the provenance record.
EXTRACTION_METADATA_VERSION = 1

#: Samples encoded per forward pass. Small on purpose: 120x120x12 optical plus
#: 120x120x2 SAR per sample, and the whole corpus never fits in memory at once.
DEFAULT_EXTRACT_BATCH_SIZE = 8

#: The three official reBEN partitions. Nothing else is a split.
OFFICIAL_SPLITS: tuple[str, ...] = ("train", "val", "test")

#: Column names a reBEN metadata table might use. Tried in order; an
#: unrecognised schema is an error naming the columns found, never a guess.
PATCH_ID_COLUMNS: tuple[str, ...] = ("patch_id", "patch", "name", "id")
SPLIT_COLUMNS: tuple[str, ...] = ("split", "official_split", "reben_split", "ben_split")

#: Spellings the reBEN releases use for the same partition.
SPLIT_ALIASES: dict[str, str] = {
    "training": "train",
    "valid": "val",
    "validation": "val",
    "testing": "test",
}


class ExtractionError(SatQueryError):
    code = "fusion_extraction_error"
    user_message = "Frozen-feature extraction could not proceed."


class ExtractionInputError(ExtractionError):
    """Missing, empty or leaky input — a refusal, not a crash."""

    code = "fusion_extraction_input"
    user_message = "The extraction input is missing, empty or leaks across splits."


class AmbiguousLabelError(ExtractionError):
    """A multi-label sample reached the single-label head with no policy.

    Raised rather than resolved. Picking a class here would silently rewrite the
    corpus's labels, and every downstream number would inherit the choice
    without recording it.
    """

    code = "fusion_ambiguous_label"
    user_message = (
        "A patch carries several land-cover labels and no label policy says how "
        "to reduce them to one."
    )


class CacheProvenanceError(ExtractionInputError):
    """An existing cache was produced by a DIFFERENT run than this one.

    Resuming merges the old rows into the new cache verbatim. If the two halves
    came from different arms, resolutions or configs, the merged cache is a
    blend of two experiments with one provenance record on top — and every
    number trained on it inherits a description of only half its rows. So a
    mismatch is refused, and the error names the field and both values.
    """

    code = "fusion_cache_provenance"
    user_message = (
        "The existing feature cache was produced under different provenance and "
        "cannot be resumed into."
    )


# ---------------------------------------------------------------------------
# The label policy
# ---------------------------------------------------------------------------

#: `Sequence[str] -> int | None`. `None` means "this sample has no usable
#: label": skip it and count it.
LabelPolicy = Callable[[Sequence[str]], "int | None"]


def _callable_fingerprint(policy: Callable[..., Any]) -> str:
    """A short, stable digest that distinguishes two anonymous callables.

    `type(policy).__name__` is not an identity: every lambda in the process is
    `"function"`, so a cache built under `lambda labels: 0` and one built under
    `lambda labels: min(labels, key=...)` record the SAME provenance string and
    the record cannot answer the question it exists to answer.

    The digest is taken over the code object's own fields (`co_code`,
    `co_consts`, `co_names`, `co_varnames`, `co_argcount`) — everything that
    makes the function's BEHAVIOUR — and deliberately NOT over `co_filename` or
    `co_firstlineno`, which would make the identity move when the same policy is
    written on a different line or in a different checkout.
    """
    code = getattr(policy, "__code__", None)
    if code is not None:
        material = "|".join(
            (
                code.co_code.hex(),
                repr(code.co_consts),
                repr(code.co_names),
                repr(code.co_varnames),
                str(code.co_argcount),
            )
        )
    else:  # a callable object: its type is the stable part
        kind = type(policy)
        module = getattr(kind, "__module__", "?")
        qualname = getattr(kind, "__qualname__", kind.__name__)
        material = f"{module}.{qualname}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
    return f"{type(policy).__name__}:{digest}"


def policy_name(policy: LabelPolicy | None) -> str:
    """A stable, distinct name for a policy, recorded in the cache metadata.

    A named function is its own `__name__`. A lambda has no useful one, so it is
    reported as `<type>:<digest of its code>` — two different lambdas get two
    different names, and the same lambda gets the same name on every run.
    """
    if policy is None:
        return "none"
    explicit = getattr(policy, "policy_name", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    name = getattr(policy, "__name__", None)
    if isinstance(name, str) and name and name != "<lambda>":
        return name
    return _callable_fingerprint(policy)


def label_index(label: str, vocabulary: Sequence[str] = CLC19_CLASSES) -> int:
    """Index of one CLC class. An unknown class is an error, not a drop.

    `BigEarthNetPatch.label_indices` drops unknown labels so an hour-long
    preparation walk is not killed by a future release adding a class. That is
    the right call for a *manifest*. It is the wrong call here: a feature cache
    is the training input, and silently discarding a label changes the task.
    """
    index = {str(name): i for i, name in enumerate(vocabulary)}
    key = str(label)
    if key not in index:
        raise ExtractionError(
            f"label {key!r} is not one of the {len(vocabulary)} known classes; "
            f"the frozen head has {FROZEN_NUM_CLASSES} outputs and a cache must "
            f"not invent a mapping for a class it does not have"
        )
    return index[key]


def require_single_label(labels: Sequence[str]) -> int | None:
    """Built-in policy: only an unambiguous one-label sample has an index.

    Zero labels -> `None` (skip). More than one -> `AmbiguousLabelError`. This is
    the policy to use when the corpus has not been reduced to single labels and
    the operator wants the ambiguity surfaced rather than resolved.
    """
    items = tuple(str(x) for x in labels)
    if not items:
        return None
    if len(items) > 1:
        raise AmbiguousLabelError(
            f"a sample carries {len(items)} labels {list(items)}; a single-label "
            f"softmax head cannot represent that and no policy was given to "
            f"reduce it. Pass an explicit label_policy."
        )
    return label_index(items[0])


def skip_ambiguous(labels: Sequence[str]) -> int | None:
    """Built-in policy: unambiguous single labels only.

    A multi-label sample returns `None`, so it is SKIPPED and counted — not
    reduced to one of its labels. This is not a convention for multi-label
    samples; it is a decision to exclude them, and the cache metadata records
    `label_policy: "skip_ambiguous"` so a reader knows the corpus is a
    single-label subset rather than the whole one.
    """
    items = tuple(str(x) for x in labels)
    if len(items) != 1:
        return None
    return label_index(items[0])


#: The policies the CLI can select by name. Neither invents a class for an
#: ambiguous sample: one refuses it, the other excludes it.
LABEL_POLICIES: dict[str, LabelPolicy] = {
    "require_single_label": require_single_label,
    "skip_ambiguous": skip_ambiguous,
}


def resolve_label(
    labels: Sequence[str],
    policy: LabelPolicy | None,
    *,
    vocabulary: Sequence[str] = CLC19_CLASSES,
    sample_id: str = "",
) -> int | None:
    """Collapse a label tuple to one index, or `None` to skip the sample.

    This is the only place the multi-label question is answered, and with
    `policy is None` it is answered by refusing to answer.
    """
    items = tuple(str(x) for x in labels)

    if policy is None:
        if not items:
            return None  # zero labels: skipped and counted, never class 0
        if len(items) > 1:
            raise AmbiguousLabelError(
                f"sample {sample_id!r} carries {len(items)} labels {list(items)}. "
                f"BigEarthNet is a multi-label corpus and the frozen head is a "
                f"single-label softmax; reducing {len(items)} labels to one is a "
                f"decision, not a detail. Pass label_policy=<callable> to state "
                f"which convention this cache is built under."
            )
        return label_index(items[0], vocabulary)

    value = policy(items)
    if value is None:
        return None
    index = int(value)
    if not 0 <= index < len(vocabulary):
        raise ExtractionError(
            f"the label policy {policy_name(policy)!r} returned {index} for "
            f"sample {sample_id!r}, outside [0, {len(vocabulary)}); the head has "
            f"{FROZEN_NUM_CLASSES} outputs and a cache row must be one of them"
        )
    return index


# ---------------------------------------------------------------------------
# The encode seam
# ---------------------------------------------------------------------------


def _encode_batch(encoder: Any, optical: np.ndarray, sar: np.ndarray) -> Any:
    """The ONLY place this module calls `.encode`. `no_grad` is structural.

    CROMA is frozen (freeze section 2.5). A gradient graph built here would be
    pure waste, and leaving the choice to each call site is how a frozen encoder
    quietly acquires `requires_grad`.
    """
    with torch.no_grad():
        return encoder.encode(optical, sar)


def _assert_raw_encoder(encoder: Any) -> None:
    """Refuse an encoder that normalises, because this pipeline already did.

    `CROMAEncoder.normalize_input` defaults to True. An encoder built that way
    applies the identical DEV-2 stretch inside `encode`, so this pipeline's
    stretch would be applied on top of it. The result is not an exception
    anywhere — it is a well-shaped cache of subtly wrong numbers, which is the
    worst outcome available in this module.
    """
    if bool(getattr(encoder, "normalize_input", False)):
        raise ExtractionError(
            "the encoder applies its own encoder-input normalisation "
            "(normalize_input=True) and this pipeline applies the DEV-2 stretch "
            "itself; the stretch would run TWICE and silently shift the input "
            "distribution. Build the encoder with normalize_input=False "
            "(see build_extraction_encoder) so the arm is applied exactly once."
        )


def _prepare_inputs(
    sample: PairedSample,
    *,
    use_8_bit: bool,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """`PairedSample` -> normalised, resized `(C, R, R)` tensors.

    Order is fixed by `docs/PHASE14_CROMA_NORMALISATION_CHANGE.md` section 3.2:
    the stretch is an encoder-INPUT transform, so it runs before the resize that
    produces the encoder's actual input. Both modalities go through the same
    function — upstream uses one `ViT` for both, and `summarise` asserts the two
    reports agree on `use_8_bit`.
    """
    optical = np.asarray(sample.optical, dtype=np.float32)
    sar = np.asarray(sample.sar, dtype=np.float32)
    if optical.ndim != 3 or sar.ndim != 3:
        raise ExtractionError(
            f"paired sample {sample.patch_id!r} is not (C, H, W): optical "
            f"{optical.shape}, SAR {sar.shape}"
        )
    if optical.shape[0] != FROZEN_OPTICAL_CHANNELS:
        raise ExtractionError(
            f"paired sample {sample.patch_id!r} has {optical.shape[0]} optical "
            f"channels; the frozen encoder takes {FROZEN_OPTICAL_CHANNELS}"
        )
    if sar.shape[0] != FROZEN_SAR_CHANNELS:
        raise ExtractionError(
            f"paired sample {sample.patch_id!r} has {sar.shape[0]} SAR channels; "
            f"the frozen encoder takes {FROZEN_SAR_CHANNELS}"
        )

    optical_norm, optical_report = normalise_for_croma(
        optical, use_8_bit=use_8_bit, modality=MODALITY_OPTICAL
    )
    sar_norm, sar_report = normalise_for_croma(
        sar, use_8_bit=use_8_bit, modality=MODALITY_SAR
    )
    report = summarise(optical_report, sar_report)

    # `normalise_for_croma` promotes (C, H, W) to (1, C, H, W) internally and
    # returns that shape; `resize_to_canonical` takes (C, H, W) per sample.
    return (
        resize_to_canonical(optical_norm[0], resolution),
        resize_to_canonical(sar_norm[0], resolution),
        report,
    )


# ---------------------------------------------------------------------------
# Split firewall
# ---------------------------------------------------------------------------


def _identity(item: Any) -> str:
    """The per-sample key of a `FusionFeature` or a `PairedSample`."""
    for attr in ("sample_id", "patch_id"):
        value = getattr(item, attr, None)
        if value:
            return str(value)
    raise ExtractionError(
        f"{type(item).__name__} carries neither sample_id nor patch_id, so it "
        f"cannot be checked against a split"
    )


def assert_split_homogeneous(
    features: Sequence[Any],
    splits: Sequence[str | None],
    *,
    requested: str,
) -> None:
    """Every feature must declare exactly the requested official split.

    Accepts anything with `sample_id` (`FusionFeature`) or `patch_id`
    (`PairedSample`). A `None` split is a failure, not a wildcard: "the release
    did not say" is not the same fact as "this is the train split", and treating
    it as one is how a test partition ends up inside a training cache.
    """
    if requested not in OFFICIAL_SPLITS:
        raise ExtractionInputError(
            f"{requested!r} is not an official split; expected one of "
            f"{list(OFFICIAL_SPLITS)}"
        )
    if len(splits) != len(features):
        raise ExtractionError(
            f"{len(features)} features but {len(splits)} declared splits"
        )

    offenders = [
        f"{_identity(item)}={declared!r}"
        for item, declared in zip(features, splits)
        if declared != requested
    ]
    if offenders:
        raise ExtractionInputError(
            f"{len(offenders)} sample(s) do not belong to the requested split "
            f"{requested!r}: {offenders[:5]}. A cache must be one homogeneous "
            f"partition; mixing partitions here is a leak, not a convenience."
        )


def assert_train_val_disjoint(
    train: Sequence[FusionFeature], val: Sequence[FusionFeature]
) -> None:
    """Reuse the trainer's own firewall rather than restating its rules.

    Raises whatever `assert_split_disjoint` raises (`FusionTrainingError`) — one
    implementation of the rule, not two that can drift apart.
    """
    assert_split_disjoint(list(train), list(val))


def assert_scene_disjoint(
    left: Iterable[str],
    right: Iterable[str],
    *,
    left_name: str = "train",
    right_name: str = "val",
) -> None:
    """Scene-level disjointness for ids that are not features yet.

    The extraction CLI holds `PairedSample`s, not `FusionFeature`s, and the
    leak has to be caught BEFORE an hour of encoding — so the same rule is
    available for plain scene ids.
    """
    overlap = {str(s) for s in left} & {str(s) for s in right}
    if overlap:
        raise ExtractionInputError(
            f"{len(overlap)} scene(s) appear in both {left_name} and "
            f"{right_name}, e.g. {sorted(overlap)[:5]}; the split leaks and every "
            f"number computed on it would be a lie"
        )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractionResult:
    """What one extraction pass did, in counts a reviewer can check."""

    cache_path: Path | None
    written: bool
    features: tuple[FusionFeature, ...]
    n_requested: int
    n_encoded: int
    n_total: int
    n_skipped_existing: int
    n_skipped_unlabelled: int
    n_skipped_by_policy: int
    n_scenes: int
    batch_size: int
    encoder_dim: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_path": str(self.cache_path) if self.cache_path else None,
            "written": self.written,
            "n_requested": self.n_requested,
            "n_encoded": self.n_encoded,
            "n_total": self.n_total,
            "n_skipped_existing": self.n_skipped_existing,
            "n_skipped_unlabelled": self.n_skipped_unlabelled,
            "n_skipped_by_policy": self.n_skipped_by_policy,
            "n_scenes": self.n_scenes,
            "batch_size": self.batch_size,
            "encoder_dim": self.encoder_dim,
        }


def _sha256_file(path: Path) -> str | None:
    """Digest of a file, or None when it cannot be read. Never raises."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


#: The provenance fields a resume must agree on, as (metadata key, label).
#: `encoder_dim` is only comparable when the encoder declares its width, so a
#: caller omits it from `expected` when it is unknown.
RESUME_PROVENANCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("cache_version", "cache layout version"),
    ("extraction_metadata_version", "extraction metadata version"),
    ("config_hash", "config hash"),
    ("arm", "normalisation arm"),
    ("use_8_bit", "use_8_bit"),
    ("croma_image_resolution", "CROMA input resolution"),
    ("encoder_dim", "encoder width"),
    ("split", "official split"),
    # `split` says WHICH PARTITION the rows came from; `label_policy` says WHICH
    # ROWS OF IT were kept. Both decide the row population, so both belong here.
    # Without this, a cache built under one selection rule can be resumed under
    # another and the merged cache keeps a single provenance record that is true
    # for only half its rows -- the same failure mode as the arm-merge defect,
    # on the one field this project's open decision actually turns on.
    ("label_policy", "label policy"),
)


#: Fields in `RESUME_PROVENANCE_FIELDS` that a caller may legitimately leave out
#: of `expected`, because this run cannot state them yet.
#:
#: Every OTHER field is REQUIRED. `check_resume_provenance` used to skip any key
#: absent from `expected`, which turned a forgotten entry into a silent PASS --
#: and that is not hypothetical: `label_policy` sat in the field list above and
#: in every cache's metadata while the caller's `expected` dict never carried
#: it, so the comparison never ran and a fully green suite showed nothing. The
#: caller is now told, loudly, when it forgets a required field.
RESUME_PROVENANCE_OPTIONAL_FIELDS: frozenset[str] = frozenset({"encoder_dim"})


def check_resume_provenance(
    existing: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    path: str | Path,
) -> None:
    """Refuse to merge into a cache that was produced by a different run.

    The merged cache keeps ONE provenance record (`existing + new` rows), so if
    the two halves disagree the record describes only half the rows. Every
    comparable field is checked and ALL mismatches are reported, not just the
    first: fixing them one restart at a time is how a run gets abandoned.

    Raises:
        CacheProvenanceError: the existing metadata is absent, or any comparable
            field disagrees. The message names the field and both values.
    """
    if not existing:
        raise CacheProvenanceError(
            f"the existing cache at {path} carries no provenance metadata, so "
            f"what produced its rows cannot be established and appending to it "
            f"would blend an unknown run into this one. Re-run with "
            f"resume=False (or --no-resume) to rebuild it."
        )

    omitted = [
        key
        for key, _ in RESUME_PROVENANCE_FIELDS
        if key not in expected and key not in RESUME_PROVENANCE_OPTIONAL_FIELDS
    ]
    if omitted:
        raise ValueError(
            f"check_resume_provenance was given no value for {omitted!r}, which "
            f"is not in RESUME_PROVENANCE_OPTIONAL_FIELDS. A field the caller "
            f"forgets to state would otherwise be skipped and the resume would "
            f"be ALLOWED on a comparison that never happened."
        )

    mismatches: list[str] = []
    for key, label in RESUME_PROVENANCE_FIELDS:
        if key not in expected:
            continue
        want = expected[key]
        if key not in existing:
            mismatches.append(
                f"  {label} ({key}): existing cache records nothing, this run "
                f"{want!r}"
            )
        elif existing[key] != want:
            mismatches.append(
                f"  {label} ({key}): existing cache {existing[key]!r}, this run "
                f"{want!r}"
            )
    if mismatches:
        raise CacheProvenanceError(
            f"refusing to resume into {path}: it was produced by a different "
            f"run and the merged cache would carry one provenance record over "
            f"rows from two experiments.\n"
            + "\n".join(mismatches)
            + "\nRe-run with resume=False (or --no-resume) to rebuild it, or "
            "point --out-cache at a fresh path."
        )


@dataclass(frozen=True)
class SampleSelection:
    """Which samples will be encoded, and why the rest will not.

    ONE implementation, used by BOTH the extractor and `plan_extraction`, so a
    dry run cannot report a different count from the run it is planning. A plan
    that disagrees with the run is worse than no plan: it is trusted.
    """

    to_encode: tuple[tuple[PairedSample, int], ...]
    n_skipped_existing: int
    n_skipped_by_policy: int
    n_skipped_unlabelled: int

    @property
    def n_selected(self) -> int:
        return len(self.to_encode)

    @property
    def scene_ids(self) -> tuple[str, ...]:
        return tuple(sample.scene_id for sample, _ in self.to_encode)


def select_samples(
    samples: Sequence[PairedSample],
    *,
    label_policy: LabelPolicy | None,
    vocabulary: Sequence[str] = CLC19_CLASSES,
    existing_ids: Iterable[str] = (),
) -> SampleSelection:
    """Apply the label policy to every candidate, once, for both paths.

    The multi-label decision is made HERE and nowhere else: a dry run that
    planned around it and a run that applied it would report two different
    corpora, and BigEarthNet v2.0 is multi-label, so on the real corpus those
    two numbers are far apart.

    Raises:
        ExtractionError: a duplicate sample id (a cache with a repeated id
            cannot be resumed or split honestly).
        AmbiguousLabelError: a multi-label sample and no policy — in a dry run
            exactly as in a real one.
    """
    already = {str(value) for value in existing_ids}
    seen: set[str] = set()
    to_encode: list[tuple[PairedSample, int]] = []
    skipped_existing = 0
    skipped_by_policy = 0
    skipped_unlabelled = 0

    for sample in samples:
        sample_id = str(sample.patch_id)
        if sample_id in seen:
            raise ExtractionError(
                f"duplicate sample id {sample_id!r} in the input; a cache with a "
                f"repeated id cannot be resumed, split or evaluated honestly"
            )
        seen.add(sample_id)

        if sample_id in already:
            skipped_existing += 1
            continue

        label = resolve_label(
            sample.labels,
            label_policy,
            vocabulary=vocabulary,
            sample_id=sample_id,
        )
        if label is None:
            skipped_by_policy += 1
            if not sample.labels:
                skipped_unlabelled += 1
            continue

        to_encode.append((sample, label))

    return SampleSelection(
        to_encode=tuple(to_encode),
        n_skipped_existing=skipped_existing,
        n_skipped_by_policy=skipped_by_policy,
        n_skipped_unlabelled=skipped_unlabelled,
    )


@dataclass(frozen=True)
class ExtractionPlan:
    """What a run WOULD do, in the same counts the run itself reports.

    `n_selected` is the post-policy count — the number of samples that will
    actually be encoded — and `n_would_write` is the number of rows the cache
    will hold afterwards. Neither is the pre-policy candidate count.
    """

    n_requested: int
    n_selected: int
    n_skipped_existing: int
    n_skipped_by_policy: int
    n_skipped_unlabelled: int
    n_already_cached: int
    n_would_write: int
    n_scenes: int
    scene_ids: tuple[str, ...]
    label_policy: str
    split: str | None
    cache_path: Path | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_requested": self.n_requested,
            "n_selected": self.n_selected,
            "n_skipped_existing": self.n_skipped_existing,
            "n_skipped_by_policy": self.n_skipped_by_policy,
            "n_skipped_unlabelled": self.n_skipped_unlabelled,
            "n_already_cached": self.n_already_cached,
            "n_would_write": self.n_would_write,
            "n_scenes": self.n_scenes,
            "label_policy": self.label_policy,
            "split": self.split,
            "cache_path": str(self.cache_path) if self.cache_path else None,
        }


def plan_extraction(
    samples: Sequence[PairedSample],
    *,
    label_policy: LabelPolicy | None,
    vocabulary: Sequence[str] = CLC19_CLASSES,
    existing_features: Sequence[FusionFeature] | None = None,
    split: str | None = None,
    declared_splits: Sequence[str | None] | None = None,
    cache_path: str | Path | None = None,
) -> ExtractionPlan:
    """Account for a run without encoding anything — the dry run's one source.

    Everything that decides WHICH samples are encoded is applied here: the split
    firewall, the resume skip, and the label policy. Only the encode, the
    degenerate-channel aggregate and the write are left to the run, so
    `plan.n_selected == result.n_encoded` and
    `plan.n_would_write == result.n_total` hold by construction rather than by
    agreement between two implementations.

    Raises:
        ExtractionInputError: a mixed/undeclared split.
        AmbiguousLabelError: a multi-label sample under a policy that refuses
            one — a dry run must surface this, not plan around it.
        ExtractionError: a duplicate sample id.
    """
    items = list(samples)
    declared = (
        list(declared_splits)
        if declared_splits is not None
        else [sample.official_split for sample in items]
    )
    if split is not None:
        assert_split_homogeneous(items, declared, requested=split)

    existing = list(existing_features or ())
    selection = select_samples(
        items,
        label_policy=label_policy,
        vocabulary=vocabulary,
        existing_ids=(feature.sample_id for feature in existing),
    )

    return ExtractionPlan(
        n_requested=len(items),
        n_selected=selection.n_selected,
        n_skipped_existing=selection.n_skipped_existing,
        n_skipped_by_policy=selection.n_skipped_by_policy,
        n_skipped_unlabelled=selection.n_skipped_unlabelled,
        n_already_cached=len(existing),
        n_would_write=len(existing) + selection.n_selected,
        n_scenes=len(set(selection.scene_ids)),
        scene_ids=selection.scene_ids,
        label_policy=policy_name(label_policy),
        split=split,
        cache_path=Path(cache_path) if cache_path is not None else None,
    )


def extract_fusion_features(
    samples: Sequence[PairedSample] | None = None,
    *,
    encoder: Any,
    loader: BandLoader | None = None,
    optical_root: str | Path | None = None,
    sar_root: str | Path | None = None,
    spatial_shape: tuple[int, int] | None = None,
    limit: int | None = None,
    label_policy: LabelPolicy | None = None,
    vocabulary: Sequence[str] = CLC19_CLASSES,
    batch_size: int = DEFAULT_EXTRACT_BATCH_SIZE,
    use_8_bit: bool | None = None,
    arm: str | Arm | None = None,
    resolution: int | None = None,
    split: str | None = None,
    declared_splits: Sequence[str | None] | None = None,
    cache_path: str | Path | None = None,
    other_partition: Sequence[FusionFeature] | None = None,
    other_partition_name: str = "the other partition",
    config: Any | None = None,
    resume: bool = True,
) -> ExtractionResult:
    """Turn paired patches into cached frozen features.

    Args:
        samples: paired samples to encode. Mutually exclusive with
            `optical_root` — supplying both would make "what am I encoding"
            ambiguous.
        encoder: anything with `.encode(optical, sar)` returning an object with
            `.as_forward_dict()` and `.dim`. MUST NOT normalise (see
            `_assert_raw_encoder`).
        loader: band reader, used only when building samples from `optical_root`.
        optical_root / sar_root / spatial_shape / limit: passed to
            `pair_patches` when samples are not supplied.
        label_policy: `Sequence[str] -> int | None`. See the module docstring:
            with None, a multi-label sample raises rather than being resolved.
        batch_size: samples per forward pass. The corpus is never fully
            materialised as tensors.
        use_8_bit / arm: the normalisation arm. `arm` is the pre-registered
            name ("A"/"B"); `use_8_bit` overrides it for tests and they must
            agree when both are given.
        resolution: CROMA input size; defaults to `croma.image_resolution`.
        split: the official partition being extracted. When given, every sample
            must declare exactly that split.
        declared_splits: the split each sample declares, when it is not on the
            sample itself (e.g. read from a reBEN metadata table).
        cache_path: where to write. None encodes without writing.
        other_partition: the other split's cached features. When given, the
            trainer's own leak firewall (`assert_split_disjoint`, via
            `assert_train_val_disjoint`) runs over the rows about to be written
            BEFORE the write, so a leaky cache is never produced. The
            scene-level check on plain ids is `assert_scene_disjoint`, which
            callers run before encoding.
        other_partition_name: how the other partition is named in the error.
        resume: True (default) loads an existing cache and skips its ids;
            False overwrites from scratch. A resume whose provenance disagrees
            with this run is refused (see `check_resume_provenance`).

    Returns:
        An `ExtractionResult`. `features` is every row the cache holds after the
        merge, so a caller can run the firewall over the real content.
        `n_encoded` and `n_total` are the same counts `plan_extraction` predicts
        for the same inputs, because both go through `select_samples`.

    Raises:
        ExtractionInputError: no/ambiguous input, an empty result, an unreadable
            existing cache, or a split that is not homogeneous.
        AmbiguousLabelError: a multi-label sample with no policy.
        ExtractionError: a typed failure during the run.
    """
    from core.config import load_config

    cfg = config if config is not None else load_config()
    _assert_raw_encoder(encoder)

    if batch_size < 1:
        raise ExtractionError(f"batch_size must be >= 1, got {batch_size}")

    # -- the arm ------------------------------------------------------------
    if arm is None:
        resolved_arm = None
    else:
        resolved_arm = resolve_arm(arm.name if isinstance(arm, Arm) else arm)
    if (
        use_8_bit is not None
        and resolved_arm is not None
        and bool(use_8_bit) != resolved_arm.use_8_bit
    ):
        raise ExtractionError(
            f"use_8_bit={use_8_bit} contradicts arm {resolved_arm.name} "
            f"(use_8_bit={resolved_arm.use_8_bit}); the cache would record an arm "
            f"it was not built with"
        )
    if use_8_bit is None:
        if resolved_arm is not None:
            use_8_bit, use_8_bit_source = resolved_arm.use_8_bit, f"arm {resolved_arm.name}"
        else:
            use_8_bit, use_8_bit_source = resolve_use_8_bit(cfg)
    else:
        use_8_bit = bool(use_8_bit)
        use_8_bit_source = "explicit"
    arm_name = resolved_arm.name if resolved_arm is not None else None

    resolved_resolution = int(
        resolution if resolution is not None else cfg.get("croma.image_resolution", 120)
    )

    # The encoder carries its own input size. If it disagrees with the size this
    # run will feed it, the two are describing different experiments and one of
    # them is stale; the cache would record `resolved_resolution` while holding
    # features the encoder produced at another size.
    declared_resolution = getattr(encoder, "resolution", None)
    if declared_resolution is not None and int(declared_resolution) != resolved_resolution:
        raise ExtractionError(
            f"the encoder was built for {int(declared_resolution)}px but this run "
            f"resizes to {resolved_resolution}px (config "
            f"croma.image_resolution={cfg.get('croma.image_resolution')!r}); the "
            f"cache would record one resolution and hold features from another"
        )

    # -- the source ---------------------------------------------------------
    if samples is None and optical_root is None:
        raise ExtractionInputError(
            "nothing to extract: pass either `samples` or `optical_root`"
        )
    if samples is not None and optical_root is not None:
        raise ExtractionInputError(
            "both `samples` and `optical_root` were given; the source would be "
            "ambiguous"
        )
    if samples is not None and loader is not None:
        raise ExtractionInputError(
            "a band loader was supplied together with explicit samples; the "
            "loader would be silently ignored"
        )

    source_info: dict[str, Any]
    if samples is None:
        try:
            items = list(
                pair_patches(
                    optical_root,
                    sar_root,
                    limit=limit,
                    loader=loader,
                    spatial_shape=spatial_shape,
                )
            )
        except BigEarthNetError as exc:
            raise ExtractionInputError(str(exc)) from exc
        source_info = {
            "kind": "roots",
            "optical_root": str(optical_root),
            "sar_root": str(sar_root) if sar_root is not None else None,
            "limit": limit,
        }
    else:
        items = list(samples)
        source_info = {"kind": "samples"}

    if not items:
        raise ExtractionInputError("no paired samples to extract from")

    declared = (
        list(declared_splits)
        if declared_splits is not None
        else [sample.official_split for sample in items]
    )
    if split is not None:
        assert_split_homogeneous(items, declared, requested=split)

    # -- resumability -------------------------------------------------------
    existing_features: list[FusionFeature] = []
    target = Path(cache_path) if cache_path is not None else None
    if target is not None and resume and target.exists():
        try:
            existing_features, existing_metadata = read_feature_cache(target)
        except FusionTrainingError as exc:
            raise ExtractionInputError(
                f"the existing cache at {target} could not be read: {exc}"
            ) from exc

        # Appending to a cache produced by another run would put ONE provenance
        # record on top of rows from two experiments. Every field this run can
        # already state is compared before a single sample is encoded.
        expected_provenance: dict[str, Any] = {
            "cache_version": CACHE_VERSION,
            "extraction_metadata_version": EXTRACTION_METADATA_VERSION,
            "config_hash": cfg.hash,
            "arm": arm_name,
            "use_8_bit": bool(use_8_bit),
            "croma_image_resolution": resolved_resolution,
            "split": split,
            # `split` names the partition; this names WHICH ROWS of it survived
            # selection. Two runs over the same split under different policies
            # produce different row populations, so a merged cache would keep
            # one policy label over rows chosen by two rules.
            "label_policy": policy_name(label_policy),
        }
        declared_dim = getattr(encoder, "encoder_dim", None)
        if isinstance(declared_dim, int) and not isinstance(declared_dim, bool):
            expected_provenance["encoder_dim"] = int(declared_dim)
        check_resume_provenance(existing_metadata, expected_provenance, path=target)
    existing_ids = {feature.sample_id for feature in existing_features}

    # -- which samples are encoded (the SAME decision a dry run makes) -------
    selection = select_samples(
        items,
        label_policy=label_policy,
        vocabulary=vocabulary,
        existing_ids=existing_ids,
    )
    skipped_existing = selection.n_skipped_existing
    skipped_unlabelled = selection.n_skipped_unlabelled
    skipped_by_policy = selection.n_skipped_by_policy

    # -- encode, in batches -------------------------------------------------
    features: list[FusionFeature] = []
    pending: list[tuple[PairedSample, int, np.ndarray, np.ndarray]] = []
    degenerate_samples = 0
    encoder_dim: int | None = None
    n_encoded = 0

    def flush() -> None:
        nonlocal encoder_dim, n_encoded
        if not pending:
            return
        optical_batch = np.stack([item[2] for item in pending]).astype(np.float32)
        sar_batch = np.stack([item[3] for item in pending]).astype(np.float32)

        # What the encoder is actually handed, checked here rather than trusted:
        # a resize that silently did nothing (or resized to the wrong size) is
        # invisible downstream — the GAP vectors still have 768 dimensions, the
        # cache still passes every shape check, and only the numbers are wrong.
        expected_spatial = (resolved_resolution, resolved_resolution)
        for name, batch in (("optical", optical_batch), ("SAR", sar_batch)):
            if batch.ndim != 4 or tuple(batch.shape[2:]) != expected_spatial:
                raise ExtractionError(
                    f"the {name} batch handed to encode is {tuple(batch.shape)}; "
                    f"expected (B, C, {resolved_resolution}, "
                    f"{resolved_resolution}) per sample. The stretch/resize step "
                    f"did not produce the encoder's input size."
                )

        encoding = _encode_batch(encoder, optical_batch, sar_batch)

        dim = int(encoding.dim)
        if encoder_dim is None:
            encoder_dim = dim
        elif encoder_dim != dim:
            raise ExtractionError(
                f"the encoder changed width mid-run: {encoder_dim} then {dim}; a "
                f"cache with two widths cannot be read back"
            )

        forward = encoding.as_forward_dict()
        optical_gap = np.asarray(forward["optical_GAP"], dtype=np.float32)
        sar_gap = np.asarray(forward["SAR_GAP"], dtype=np.float32)
        joint_gap = np.asarray(forward["joint_GAP"], dtype=np.float32)
        for name, array in (
            ("optical_GAP", optical_gap),
            ("SAR_GAP", sar_gap),
            ("joint_GAP", joint_gap),
        ):
            if array.shape[0] != len(pending):
                raise ExtractionError(
                    f"the encoder returned {array.shape[0]} rows of {name} for a "
                    f"batch of {len(pending)} samples"
                )

        for index, (sample, label, _, _) in enumerate(pending):
            features.append(
                FusionFeature(
                    sample_id=sample.patch_id,
                    scene_id=sample.scene_id,
                    optical_gap=optical_gap[index],
                    sar_gap=sar_gap[index],
                    joint_gap=joint_gap[index],
                    optical_mask=np.asarray(sample.optical_mask, dtype=np.float32),
                    sar_mask=np.asarray(sample.sar_mask, dtype=np.float32),
                    label_index=int(label),
                )
            )
        n_encoded += len(pending)
        pending.clear()

    for sample, label in selection.to_encode:
        optical, sar, report = _prepare_inputs(
            sample, use_8_bit=use_8_bit, resolution=resolved_resolution
        )
        # The per-sample window trace is O(corpus) and is NOT stored; the
        # transform's identity and `use_8_bit` are, and they are what identifies
        # the arm. One aggregate is kept because "the sensor measured a flat
        # band" is a data defect that no mask records.
        if report.optical.degenerate_channels or report.sar.degenerate_channels:
            degenerate_samples += 1
        pending.append((sample, label, optical, sar))
        if len(pending) >= batch_size:
            flush()
    flush()

    # -- the leak firewall, on the artifact that would be written -----------
    #
    # `assert_scene_disjoint` (called by the CLI before any encoding) compares
    # plain scene ids and fails fast. This one runs the TRAINER's own rule over
    # the rows that are about to be written, and it also catches the leak the
    # scene check cannot see: the same patch id in two partitions. It runs
    # BEFORE the write, so a leaky cache is never produced in the first place.
    if other_partition is not None:
        try:
            assert_train_val_disjoint(
                list(existing_features) + features, list(other_partition)
            )
        except FusionTrainingError as exc:
            raise ExtractionInputError(
                f"the rows about to be written to {cache_path} leak into "
                f"{other_partition_name}: {exc}"
            ) from exc

    # -- provenance and write ----------------------------------------------
    to_write = list(existing_features) + features
    checkpoint_path = getattr(encoder, "checkpoint_path", None)
    checkpoint_sha = None
    if checkpoint_path:
        candidate = Path(str(checkpoint_path))
        if candidate.exists():
            checkpoint_sha = _sha256_file(candidate)

    if encoder_dim is None:
        fallback = getattr(encoder, "encoder_dim", None)
        encoder_dim = int(fallback) if isinstance(fallback, int) else None

    metadata: dict[str, Any] = {
        "cache_version": CACHE_VERSION,
        "extraction_metadata_version": EXTRACTION_METADATA_VERSION,
        "config_hash": cfg.hash,
        "arm": arm_name,
        "use_8_bit": bool(use_8_bit),
        "use_8_bit_source": use_8_bit_source,
        "croma_image_resolution": resolved_resolution,
        "encoder_dim": encoder_dim,
        "croma_checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "croma_checkpoint_sha256": checkpoint_sha,
        "normalisation": describe_transform(bool(use_8_bit)),
        "label_policy": policy_name(label_policy),
        "vocabulary_size": len(vocabulary),
        "num_classes": FROZEN_NUM_CLASSES,
        "split": split,
        "batch_size": int(batch_size),
        "n_samples": len(to_write),
        "n_encoded_this_run": len(features),
        "n_skipped_existing": skipped_existing,
        "n_skipped_unlabelled": skipped_unlabelled,
        "n_skipped_by_policy": skipped_by_policy,
        "n_scenes": len({feature.scene_id for feature in to_write}),
        "n_samples_with_degenerate_channels": degenerate_samples,
        "source": source_info,
        "deterministic": True,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }

    written = False
    if target is not None:
        if features:
            write_feature_cache(target, to_write, metadata=metadata)
            written = True
        elif not existing_features:
            raise ExtractionInputError(
                f"no sample produced a feature ({len(items)} considered, "
                f"{skipped_by_policy} skipped for having no usable label); "
                f"refusing to write an empty cache"
            )

    result_features = to_write if target is not None else features
    return ExtractionResult(
        cache_path=target,
        written=written,
        features=tuple(result_features),
        n_requested=len(items),
        n_encoded=len(features),
        n_total=len(to_write),
        n_skipped_existing=skipped_existing,
        n_skipped_unlabelled=skipped_unlabelled,
        n_skipped_by_policy=skipped_by_policy,
        n_scenes=len({feature.scene_id for feature in result_features}),
        batch_size=int(batch_size),
        encoder_dim=encoder_dim,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------------------------


def verify_feature_cache(
    path: str | Path, *, config: Any | None = None
) -> dict[str, Any]:
    """Check a feature cache against the frozen contract. Never raises.

    Returns a structured report — `all_checks_passed` plus a list of
    `{check, passed, detail}` — mirroring `artifacts/optical_sar/croma_forward.json`
    so a failure reads the same way as every other verification in this repo.

    Two families of check are run, and the second is the one that is easy to
    forget:

      * SHAPE — the frozen contract. Three 768-d GAP blocks, a 12-d optical mask
        and a 2-d SAR mask, so 2318 wide; finite; labels in range; the sidecar's
        ids agreeing with the array rows.
      * PROVENANCE — WHICH run produced these numbers: the extraction metadata
        version, the config hash, the arm and its `use_8_bit`, the CROMA input
        resolution and the encoder width. A cache can satisfy every shape check
        and still be features from another arm at another resolution, and a
        trainer that read it would report a number belonging to a different
        experiment. These are compared against the live config (pass `config` to
        compare against an explicit one instead of re-loading it).

    A corrupt cache is reported, not repaired and not deleted: an artifact that
    failed a check is evidence about the run that produced it, and the reader's
    job is to decide, not this function's.
    """
    src = Path(path)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    def report(**extra: Any) -> dict[str, Any]:
        verdict = bool(checks) and all(item["passed"] for item in checks)
        payload: dict[str, Any] = {
            "path": str(src),
            "all_checks_passed": verdict,
            "verdict": verdict,
            "failed_checks": [c["check"] for c in checks if not c["passed"]],
            "checks": checks,
        }
        payload.update(extra)
        return payload

    if not src.exists():
        check("npz_exists", False, f"{src} does not exist")
        return report(cache_version=None, n=0)

    check("npz_exists", True, str(src))

    sidecar_path = src.with_suffix(".json")
    if not sidecar_path.exists():
        check(
            "sidecar_exists",
            False,
            f"{sidecar_path} is missing (an interrupted write); the ids cannot "
            f"be recovered from the arrays alone",
        )
        return report(cache_version=None, n=0)
    check("sidecar_exists", True, str(sidecar_path))

    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a verifier reports, never raises
        check("sidecar_readable", False, f"{type(exc).__name__}: {exc}")
        return report(cache_version=None, n=0)
    check("sidecar_readable", True, f"{len(sidecar)} key(s)")

    version = sidecar.get("cache_version")
    check(
        "cache_version_matches",
        version == CACHE_VERSION,
        f"sidecar {version!r}, expected {CACHE_VERSION!r}",
    )

    try:
        with np.load(str(src)) as data:
            arrays = {key: data[key] for key in data.files}
    except Exception as exc:  # noqa: BLE001
        check("npz_readable", False, f"{type(exc).__name__}: {exc}")
        return report(cache_version=version, n=0)
    check("npz_readable", True, f"keys {sorted(arrays)}")

    required = (
        "optical_gap",
        "sar_gap",
        "joint_gap",
        "optical_mask",
        "sar_mask",
        "label_index",
    )
    missing = [key for key in required if key not in arrays]
    if missing:
        check(
            "required_arrays_present",
            False,
            f"missing {missing}; present {sorted(arrays)}",
        )
        return report(cache_version=version, n=0)
    check("required_arrays_present", True, ", ".join(required))

    def width(array: np.ndarray) -> int | None:
        return int(array.shape[1]) if array.ndim == 2 else None

    optical_gap = arrays["optical_gap"]
    sar_gap = arrays["sar_gap"]
    joint_gap = arrays["joint_gap"]
    optical_mask = arrays["optical_mask"]
    sar_mask = arrays["sar_mask"]
    labels = arrays["label_index"]

    gap_widths = {name: width(array) for name, array in (
        ("optical_GAP", optical_gap), ("SAR_GAP", sar_gap), ("joint_GAP", joint_gap),
    )}
    check(
        "gap_vectors_are_768_d",
        all(value == FROZEN_ENCODER_DIM for value in gap_widths.values()),
        f"{gap_widths}, expected {FROZEN_ENCODER_DIM}",
    )
    optical_mask_width = width(optical_mask)
    check(
        "optical_mask_is_12_d",
        optical_mask_width == FROZEN_OPTICAL_CHANNELS,
        f"{optical_mask_width}, expected {FROZEN_OPTICAL_CHANNELS}",
    )
    sar_mask_width = width(sar_mask)
    check(
        "sar_mask_is_2_d",
        sar_mask_width == FROZEN_SAR_CHANNELS,
        f"{sar_mask_width}, expected {FROZEN_SAR_CHANNELS}",
    )

    widths = [*gap_widths.values(), optical_mask_width, sar_mask_width]
    total = sum(widths) if all(value is not None for value in widths) else None
    check(
        "total_width_is_2318",
        total == FROZEN_FUSION_DIM,
        f"3*{gap_widths.get('optical_GAP')} + {optical_mask_width} + "
        f"{sar_mask_width} = {total}, expected {FROZEN_FUSION_DIM}",
    )

    non_finite = [
        name
        for name, array in (
            ("optical_gap", optical_gap),
            ("sar_gap", sar_gap),
            ("joint_gap", joint_gap),
            ("optical_mask", optical_mask),
            ("sar_mask", sar_mask),
        )
        if not bool(np.all(np.isfinite(array)))
    ]
    check(
        "no_nan_or_inf",
        not non_finite,
        "all finite" if not non_finite else f"non-finite values in {non_finite}",
    )

    label_ok = bool(labels.size) and bool(
        np.all((labels >= 0) & (labels < FROZEN_NUM_CLASSES))
    )
    check(
        "label_index_in_range",
        label_ok,
        f"range [{int(labels.min())}, {int(labels.max())}], expected "
        f"[0, {FROZEN_NUM_CLASSES})"
        if labels.size
        else "no labels",
    )

    sample_ids = [str(value) for value in sidecar.get("sample_ids") or []]
    scene_ids = [str(value) for value in sidecar.get("scene_ids") or []]

    check(
        "sample_ids_unique",
        len(sample_ids) == len(set(sample_ids)),
        f"{len(sample_ids)} ids, {len(set(sample_ids))} distinct",
    )
    check(
        "scene_ids_present",
        len(scene_ids) == len(sample_ids) and all(value.strip() for value in scene_ids),
        f"{len(scene_ids)} scene id(s) for {len(sample_ids)} sample(s); "
        f"scene id is the leakage boundary and may not be empty",
    )

    rows = int(labels.shape[0]) if labels.ndim else 0
    check(
        "row_counts_agree",
        len(sample_ids) == len(scene_ids) == rows,
        f"sidecar sample_ids={len(sample_ids)}, scene_ids={len(scene_ids)}, "
        f"array rows={rows}",
    )
    check(
        "sidecar_n_matches_rows",
        sidecar.get("n") == rows,
        f"sidecar n={sidecar.get('n')!r}, array rows={rows}",
    )

    # -- provenance: WHICH run produced these numbers ------------------------
    metadata = dict(sidecar.get("metadata") or {})
    check(
        "provenance_present",
        bool(metadata),
        f"{len(metadata)} metadata key(s)"
        if metadata
        else "the sidecar carries no metadata, so the arm, the resolution and "
        "the config behind these rows cannot be established",
    )
    check(
        "extraction_metadata_version_matches",
        metadata.get("extraction_metadata_version") == EXTRACTION_METADATA_VERSION,
        f"metadata records {metadata.get('extraction_metadata_version')!r}, "
        f"this reader expects {EXTRACTION_METADATA_VERSION!r}",
    )

    expected_hash: Any = None
    expected_resolution: Any = None
    config_error: str | None = None
    if config is not None:
        cfg = config
    else:
        try:
            from core.config import load_config

            cfg = load_config()
        except Exception as exc:  # noqa: BLE001 - a verifier reports, never raises
            cfg = None
            config_error = f"the config could not be loaded: {type(exc).__name__}: {exc}"
    if cfg is not None:
        expected_hash = getattr(cfg, "hash", None)
        try:
            expected_resolution = int(cfg.get("croma.image_resolution", 120))
        except Exception as exc:  # noqa: BLE001
            config_error = f"croma.image_resolution is unusable: {exc}"

    check(
        "config_hash_matches",
        expected_hash is not None and metadata.get("config_hash") == expected_hash,
        config_error
        or f"metadata {metadata.get('config_hash')!r}, current config "
        f"{expected_hash!r}",
    )

    arm_name = metadata.get("arm")
    use_8_bit = metadata.get("use_8_bit")
    if arm_name is None:
        arm_ok = isinstance(use_8_bit, bool)
        arm_detail = (
            f"no arm recorded; use_8_bit={use_8_bit!r} (must be a bool so the "
            f"stretch is known)"
        )
    elif arm_name not in ARMS:
        arm_ok = False
        arm_detail = f"arm {arm_name!r} is not one of {sorted(ARMS)}"
    else:
        arm_ok = isinstance(use_8_bit, bool) and ARMS[arm_name].use_8_bit == use_8_bit
        arm_detail = (
            f"arm {arm_name!r} implies use_8_bit="
            f"{ARMS[arm_name].use_8_bit}, metadata records {use_8_bit!r}"
        )
    check("arm_use_8_bit_consistent", arm_ok, arm_detail)

    check(
        "croma_image_resolution_matches",
        expected_resolution is not None
        and metadata.get("croma_image_resolution") == expected_resolution,
        config_error
        or f"metadata {metadata.get('croma_image_resolution')!r}, current config "
        f"{expected_resolution!r}",
    )
    check(
        "encoder_dim_matches",
        metadata.get("encoder_dim") == FROZEN_ENCODER_DIM,
        f"metadata {metadata.get('encoder_dim')!r}, the frozen encoder width is "
        f"{FROZEN_ENCODER_DIM}",
    )
    check(
        "label_policy_recorded",
        isinstance(metadata.get("label_policy"), str)
        and bool(metadata.get("label_policy")),
        f"{metadata.get('label_policy')!r}; a multi-label corpus reduced to one "
        f"class without a recorded policy is an unrecorded decision",
    )

    return report(cache_version=version, n=rows, metadata=metadata)


# ---------------------------------------------------------------------------
# reBEN metadata tables
# ---------------------------------------------------------------------------


def _normalise_split(value: Any) -> str | None:
    text = str(value).strip().lower()
    text = SPLIT_ALIASES.get(text, text)
    return text if text in OFFICIAL_SPLITS else None


def load_split_assignments(path: str | Path) -> dict[str, str]:
    """Read `patch_id -> official split` from a reBEN metadata table.

    `.parquet` is the released form; `.csv`/`.tsv` are accepted because a parquet
    reader (pyarrow) is not a dependency of this project and a metadata table
    exported as CSV is the same table.

    The column names are DISCOVERED from a declared list and the discovered
    names are the caller's to print. A table whose schema is not recognised
    raises naming the columns it does have — guessing which column is the split
    is how a test partition gets relabelled as training.

    Raises:
        ExtractionInputError: the file or the schema is unusable.
    """
    src = Path(path)
    if not src.exists():
        raise ExtractionInputError(f"metadata table does not exist: {src}")

    suffix = src.suffix.lower()
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment issue
        raise ExtractionInputError(
            f"reading '{src.name}' needs pandas: {exc}"
        ) from exc

    if suffix == ".parquet":
        try:
            frame = pd.read_parquet(src)
        except Exception as exc:  # noqa: BLE001
            raise ExtractionInputError(
                f"could not read the parquet metadata '{src}': {exc}. A parquet "
                f"reader (pyarrow) must be installed; export the table as CSV if "
                f"it is not."
            ) from exc
    elif suffix in {".csv", ".tsv"}:
        try:
            frame = pd.read_csv(src, sep="\t" if suffix == ".tsv" else ",")
        except Exception as exc:  # noqa: BLE001
            raise ExtractionInputError(
                f"could not read the metadata table '{src}': {exc}"
            ) from exc
    else:
        raise ExtractionInputError(
            f"unsupported metadata table format {suffix!r}; expected .parquet, "
            f".csv or .tsv"
        )

    columns = {str(name).strip().lower(): name for name in frame.columns}
    id_column = next((columns[key] for key in PATCH_ID_COLUMNS if key in columns), None)
    split_column = next(
        (columns[key] for key in SPLIT_COLUMNS if key in columns), None
    )
    if id_column is None or split_column is None:
        raise ExtractionInputError(
            f"the metadata table '{src.name}' has columns {list(frame.columns)}; "
            f"expected a patch-id column from {list(PATCH_ID_COLUMNS)} and a "
            f"split column from {list(SPLIT_COLUMNS)}"
        )

    assignments: dict[str, str] = {}
    for patch_id, value in zip(
        frame[id_column].astype(str), frame[split_column].astype(str)
    ):
        split = _normalise_split(value)
        if split is not None:
            assignments[str(patch_id)] = split

    if not assignments:
        raise ExtractionInputError(
            f"no usable split values in '{src.name}' column {split_column!r}; "
            f"expected some of {list(OFFICIAL_SPLITS)}"
        )
    return assignments


# ---------------------------------------------------------------------------
# The real encoder, built raw
# ---------------------------------------------------------------------------


def build_extraction_encoder(
    config: Any,
    *,
    checkpoint_path: str | Path,
    vendor_dir: str | Path | None = None,
    device: str = "cpu",
    use_8_bit: bool | None = None,
) -> Any:
    """Build CROMA in RAW mode, because this pipeline owns the stretch.

    `CROMAEncoder(..., normalize_input=False)` — see `_assert_raw_encoder` for
    why the default would be a silent double normalisation.

    This function NEVER downloads. `croma.build_encoder` falls back to
    `hf_hub_download` when no checkpoint is given; a frozen-feature producer
    that reaches for the network mid-run is not reproducible, so a missing local
    checkpoint is refused instead.

    Raises:
        ExtractionInputError: the checkpoint or the vendored `use_croma.py` is
            absent — a deployment prerequisite, not something to fetch.
    """
    from specialists.optical_sar.croma import (
        VERIFIED_MODALITY,
        VERIFIED_SIZE,
        CROMAEncoder,
        load_vendored_pretrained_croma,
    )

    path = Path(checkpoint_path)
    if not path.exists():
        raise ExtractionInputError(
            f"CROMA checkpoint does not exist: {path}. This producer never "
            f"downloads; place CROMA_base.pt on disk and pass its path."
        )

    if vendor_dir is None:
        vendor_dir = (
            Path(__file__).resolve().parents[2] / "specialists" / "optical_sar" / "vendor"
        )

    size = str(config.get("croma.variant", VERIFIED_SIZE))
    resolution = int(config.get("croma.image_resolution", 120))

    try:
        pretrained = load_vendored_pretrained_croma(vendor_dir)
    except ModelLoadError as exc:
        raise ExtractionInputError(str(exc)) from exc

    model = pretrained(
        pretrained_path=str(path),
        size=size,
        modality=VERIFIED_MODALITY,
        image_resolution=resolution,
    )
    model.to(device)

    return CROMAEncoder(
        model,
        resolution=resolution,
        device=device,
        checkpoint_path=str(path),
        use_8_bit=use_8_bit,
        use_8_bit_source="extraction-arm",
        normalize_input=False,
    )


__all__ = [
    "FROZEN_ENCODER_DIM",
    "FROZEN_OPTICAL_CHANNELS",
    "FROZEN_SAR_CHANNELS",
    "FROZEN_FUSION_DIM",
    "FROZEN_NUM_CLASSES",
    "EXTRACTION_METADATA_VERSION",
    "DEFAULT_EXTRACT_BATCH_SIZE",
    "OFFICIAL_SPLITS",
    "PATCH_ID_COLUMNS",
    "SPLIT_COLUMNS",
    "SPLIT_ALIASES",
    "ExtractionError",
    "ExtractionInputError",
    "AmbiguousLabelError",
    "CacheProvenanceError",
    "LabelPolicy",
    "LABEL_POLICIES",
    "RESUME_PROVENANCE_FIELDS",
    "policy_name",
    "check_resume_provenance",
    "label_index",
    "require_single_label",
    "skip_ambiguous",
    "resolve_label",
    "assert_split_homogeneous",
    "assert_train_val_disjoint",
    "assert_scene_disjoint",
    "ExtractionResult",
    "SampleSelection",
    "select_samples",
    "ExtractionPlan",
    "plan_extraction",
    "extract_fusion_features",
    "verify_feature_cache",
    "load_split_assignments",
    "build_extraction_encoder",
]
