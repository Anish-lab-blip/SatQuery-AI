"""SatQuery AI — Phase 6 BigEarthNet -> SmolVLM instruction dataset.

WHAT THIS BUILDS
----------------
One training sample is: **an RGB image rendered from a BigEarthNet-S2 patch**, a
**question**, and a **short answer**. The questions are generated from the patch's
labels; the image is the patch.

Three pieces of existing infrastructure are reused rather than re-implemented,
because a second copy of any of them would be a second thing to keep in sync:

  * **patches**      `training.data.bigearthnet.discover_patches`
  * **labels**       `training.data.bigearthnet_labels.BigEarthNetLabelIndex`
                     (the per-patch `metadata.json` path does not exist locally)
  * **scene keys**   `training.data.bigearthnet_blocks.reconstruct_blocks` (T2)
  * **splits**       the release's own partition, checked with
                     `evaluation.leakage.assert_no_scene_overlap`

WHY THE SPLIT IS THE OFFICIAL PARTITION KEYED BY T2 BLOCKS
----------------------------------------------------------
An earlier revision of this module discarded the official partition and
re-partitioned by Sentinel tile. **That was wrong, and the reason it was wrong is
worth recording.**

The tile-level argument was: the release's split is not scene-disjoint, because
**52 of 54 tiles straddle two or more partitions** (only 2 are partition-pure;
`docs/PHASE12_LABEL_POLICY_DECISION.md` section 6.2, `:544`). That measurement is
correct. The conclusion drawn from it was not. It conflated the **scene key** with
the **partition**:

  * The partition is drawn on the ground location `(tile, row, col)`, and that unit
    is **pure** -- measured, 0 impure cells. A tile straddling partitions is not a
    leak; it is simply a tile containing more than one scene.
  * The tile is therefore the wrong *key*, not the partition the wrong *split*.

The right key is the **T2 spatial block**: the 4-connected components of the
`(row, col)` grid, computed **separately per `(tile, split)`**, so a component can
never cross a partition boundary. `training.data.bigearthnet_blocks` implements
this, is unit-tested (`tests/unit/test_bigearthnet_blocks.py`), and its docstring
records that T2 is *the one Phase 12 selected*. Its scene id is
``ben_<tile>:<k>``, deliberately **without** a split token: if a block ever did
straddle two partitions its id would appear in both, and the disjointness check
would fire. An id that encoded the split would make the check pass by
construction of the *string*.

So this module now **keeps the release's partition** and keys it by T2 block. That
is both the repository's settled decision and the better one: the evaluation set is
the dataset's own held-out geography, and the leakage check is meaningful instead
of vacuous.

Two consequences, stated rather than hidden:

  1. Blocks are reconstructed from **the patches in this corpus**, not from the
     full 480,038-patch release. Restricting the cell set can only *split* a block
     into more components; it can never merge cells across a partition. The
     grouping is therefore conservative, and the leakage guarantee is unaffected.
  2. `train_ratio` / `val_ratio` no longer *generate* the split -- the release does.
     They are retained as a **sanity check** on it (`SPLIT_RATIO_TOLERANCE`), so the
     registry's ratios stay load-bearing rather than becoming two more declared-but-
     unread keys.

WHY THE INSTRUCTION FAMILIES ARE RESTRICTED
-------------------------------------------
Measured 2026-09-23 on the local corpus: **every one of the 24,732 manifest-matched
patches carries exactly ONE label**, against a corpus-wide mean of 2.949 labels
and a maximum of 11. The 28k is the *single-label subset* selected for the Phase 12
classification metric, not a representative BigEarthNet sample.

The master plan's section 44 example answer is
`"Arable land, mixed forest and urban fabric."` -- a multi-label enumeration this
corpus cannot produce. Training the multi-label template here would teach the model
that *a scene has exactly one land-cover class*, which is a property of the subset
presented as a fact about remote sensing. With 56.9% of the subset labelled
"Marine waters" it would also acquire a strong water prior.

So `instruction_families` defaults to **`("presence",)`**, which is fully
well-posed on this corpus: "Is <label> present?" -> "Yes." / "No.". The other
families are **available but not enabled**, and enabling one without a multi-label
corpus is refused by `_check_family_is_well_posed` rather than quietly producing
degenerate data.

THE RGB RENDERING, AND WHY THOSE THREE BANDS
--------------------------------------------
SmolVLM is a *vision*-language model: it needs a visual image. BigEarthNet-S2 ships
12 single-band reflectance TIFFs. Bands **B04 (red), B03 (green), B02 (blue)** are
the true-colour triple -- the only choice that makes the model look at something
resembling a satellite image. B05-B12 have no RGB analogue, and inventing a
false-colour mapping would train on imagery that does not look like the deployment
imagery.

Normalisation is a **per-band 2nd-98th percentile stretch**, which is the
convention already used in `specialists/optical_sar/radiometry.py`. It is recorded
in the run manifest so the transform is reproducible: reflectance TIFFs are uint16
and would otherwise render almost black.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from core.errors import SatQueryError

__all__ = [
    "RGB_BANDS",
    "SPLIT_RATIO_TOLERANCE",
    "FAMILY_NEEDS_MULTI_LABEL",
    "FAMILIES_WITH_GENERATORS",
    "VLMDataError",
    "PatchSample",
    "InstructionSample",
    "CorpusBuild",
    "discover_labelled_patches",
    "render_rgb",
    "build_corpus",
    "describe_render",
]

#: How far the release's actual split proportions may drift from the registry's
#: `train_ratio` / `val_ratio` before the corpus build warns. The registry no
#: longer generates the split (the release does), so this is the check that keeps
#: those two keys load-bearing instead of inert.
SPLIT_RATIO_TOLERANCE: float = 0.15

#: True-colour triple. Order is (red, green, blue) -- the order a PIL image wants.
RGB_BANDS: tuple[str, str, str] = ("B04", "B03", "B02")

#: The families the plan's section 44 names, and whether each is well-posed on a
#: single-label corpus. `dominance` is additionally ill-posed on ANY BigEarthNet
#: release because the annotation carries no per-class area -- `build_instruction_pairs`
#: already handles that by declining to name a dominant class.
FAMILY_NEEDS_MULTI_LABEL: dict[str, bool] = {
    "presence": False,
    "dominance": False,
    "multi": True,
}

#: Families that actually have a generator implemented in this module. A family
#: can be well-posed (see `FAMILY_NEEDS_MULTI_LABEL`) and still have no generator;
#: requesting such a family would fall through the generation loop, emit zero
#: samples, and surface as the generic "generated nothing" error -- which blames
#: the data for what is a configuration error. `_check_families_have_generators`
#: refuses it before discovery.
FAMILIES_WITH_GENERATORS: tuple[str, ...] = ("presence",)


class VLMDataError(SatQueryError):
    code = "vlm_data_error"
    user_message = "The VLM instruction corpus could not be built."


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PatchSample:
    """One usable patch: its files, its labels, and its leakage scene."""

    patch_id: str
    #: The T2 spatial-block id, ``ben_<tile>:<k>`` -- the leakage key. NOT the
    #: Sentinel tile: a tile straddles partitions, a block cannot.
    scene_id: str
    tile: str
    directory: Path
    labels: tuple[str, ...]
    band_paths: dict[str, Path]
    #: The release's own partition, normalised to train/val/test.
    official_split: str | None
    #: The split this patch is used in. Under the T2 policy this EQUALS
    #: `official_split`; it stays a separate field so the assignment is explicit
    #: in every record instead of implied by a convention.
    split: str = "train"

    def band(self, token: str) -> Path | None:
        return self.band_paths.get(token)

    def has_rgb(self) -> bool:
        return all(self.band(t) is not None for t in RGB_BANDS)

    @property
    def scene_key(self) -> str:
        """The leakage key `evaluation.leakage` reads.

        `assert_no_scene_overlap` and the manifest audit take `record.scene_key`,
        not `record.scene_id`. `evaluation.manifests.SampleRecord` defines that as
        "the explicit `scene_id` if present, else a documented fallback chain";
        here the T2 block id is always present, so this mirrors it directly rather
        than duplicating the fallback chain.
        """
        return self.scene_id


@dataclass(frozen=True)
class InstructionSample:
    """One (image, question, answer) triple -- the unit the trainer consumes."""

    sample_id: str
    patch_id: str
    scene_id: str
    split: str
    question: str
    answer: str
    family: str
    directory: Path
    band_paths: dict[str, Path]

    def to_manifest_record(self) -> Any:
        """Convert to the repository's `SampleRecord` so the manifest, the hash,
        and the leakage audit are the existing ones rather than new ones."""
        from evaluation.manifests import SampleRecord

        return SampleRecord(
            dataset_id="bigearthnet_v2_s2_vlm",
            sample_id=self.sample_id,
            split=self.split,  # type: ignore[arg-type]
            scene_id=self.scene_id,
            sensor="s2",
            path=str(self.directory),
            extra={
                "patch_id": self.patch_id,
                "family": self.family,
                "question": self.question,
                "answer": self.answer,
            },
        )


@dataclass
class CorpusBuild:
    """Everything a run manifest needs to describe the corpus it trained on."""

    samples: list[InstructionSample] = field(default_factory=list)
    patches: list[PatchSample] = field(default_factory=list)
    coverage: Any = None
    class_counts: dict[str, int] = field(default_factory=dict)
    render: dict[str, Any] = field(default_factory=dict)
    families: tuple[str, ...] = ()
    #: How the split was made, and what it actually came out as: the leakage key
    #: in use, the release's proportions, and the T2 block counts. Recorded in the
    #: run manifest so a reader can see the partition without re-deriving it.
    split_info: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def by_split(self, split: str) -> list[InstructionSample]:
        return [s for s in self.samples if s.split == split]

    def split_counts(self) -> dict[str, int]:
        out = {"train": 0, "val": 0, "test": 0}
        for s in self.samples:
            out[s.split] = out.get(s.split, 0) + 1
        return out

    def scene_counts(self) -> dict[str, int]:
        out = {"train": 0, "val": 0, "test": 0}
        for split in out:
            out[split] = len({s.scene_id for s in self.samples if s.split == split})
        return out

    def __len__(self) -> int:
        return len(self.samples)


# ---------------------------------------------------------------------------
# Discovery + labelling
# ---------------------------------------------------------------------------
def discover_labelled_patches(
    *,
    corpus_root: str | Path,
    metadata_parquet: str | Path,
    limit: int | None = None,
) -> tuple[list[PatchSample], Any, list[str]]:
    """Discover S2 patches and attach labels from the release manifest.

    Returns `(patches, coverage, warnings)`.

    `discover_patches` is called with `require_labels=False` **deliberately**. The
    default `True` asks the per-patch `metadata.json` path for labels, and the local
    corpus has no such file, so `True` returns nothing and raises a message that
    blames band files. Asking it to skip that filter and joining labels from the
    parquet instead is the whole point of this function.
    """
    from training.data.bigearthnet import discover_patches
    from training.data.bigearthnet_labels import BigEarthNetLabelIndex

    index = BigEarthNetLabelIndex.from_parquet(metadata_parquet)
    raw = discover_patches(corpus_root, "s2", limit=limit, require_labels=False)

    coverage = index.coverage(p.patch_id for p in raw)

    patches: list[PatchSample] = []
    for p in raw:
        labels = index.labels_for(p.patch_id)
        if labels is None or not labels:
            # No manifest row, or a row with no labels. Excluded, never defaulted.
            continue
        if not all(t in p.band_paths for t in RGB_BANDS):
            # A patch without the true-colour triple cannot be rendered. Counted
            # by the coverage warning rather than silently dropped.
            continue
        patches.append(
            PatchSample(
                patch_id=p.patch_id,
                scene_id=p.scene_id,
                tile=p.tile,
                directory=p.directory,
                labels=labels,
                band_paths=dict(p.band_paths),
                official_split=index.split_for(p.patch_id),
            )
        )

    warnings = list(coverage.warnings())
    n_missing_rgb = coverage.n_matched - len(patches)
    if n_missing_rgb > 0:
        warnings.append(
            f"{n_missing_rgb} labelled patches are missing one or more of "
            f"{list(RGB_BANDS)} and cannot be rendered as true colour; excluded"
        )
    if not patches:
        raise VLMDataError(
            f"no usable S2 patches under {corpus_root}: discovered "
            f"{coverage.n_requested}, labelled {coverage.n_matched}, renderable "
            f"with {list(RGB_BANDS)} {len(patches)}. See the warnings for which "
            f"filter removed them.",
            context={"coverage": coverage.to_dict()},
        )
    return patches, coverage, warnings


# ---------------------------------------------------------------------------
# RGB rendering
# ---------------------------------------------------------------------------
def render_rgb(
    patch: PatchSample,
    *,
    percentiles: tuple[float, float] = (2.0, 98.0),
    size: int | None = None,
) -> Any:
    """Render a patch's true-colour image as a PIL `Image`.

    Percentile stretch per band, then uint8. The stretch is computed on the patch
    itself rather than from corpus-wide statistics: a global stretch computed over
    the training split and applied at inference would be a preprocessing
    dependency the serving path does not have, and the adapter must load against
    the same preprocessing the base model sees.

    A constant band (lo == hi, e.g. all-zero fill) is stretched to zeros rather
    than divided by zero -- the failure would otherwise be a NaN image, which the
    processor silently turns into a plausible tensor.
    """
    from training.data.bigearthnet import _load_band_default

    channels: list[np.ndarray] = []
    for token in RGB_BANDS:
        path = patch.band(token)
        if path is None:
            raise VLMDataError(
                f"patch {patch.patch_id} has no band {token}; "
                f"renderable bands are {sorted(patch.band_paths)}",
                context={"patch_id": patch.patch_id},
            )
        channels.append(np.asarray(_load_band_default(path), dtype=np.float64))

    lo_p, hi_p = percentiles
    stretched: list[np.ndarray] = []
    for arr in channels:
        lo = float(np.percentile(arr, lo_p))
        hi = float(np.percentile(arr, hi_p))
        if hi - lo < 1e-9:
            stretched.append(np.zeros_like(arr))
            continue
        scaled = (arr - lo) / (hi - lo)
        stretched.append(np.clip(scaled, 0.0, 1.0))

    rgb = np.stack(stretched, axis=-1)
    out = (rgb * 255.0).round().astype(np.uint8)

    from PIL import Image

    image = Image.fromarray(out, mode="RGB")
    if size is not None and size > 0:
        image = image.resize((size, size), Image.BILINEAR)
    return image


def describe_render(
    *, percentiles: tuple[float, float], size: int | None = None
) -> dict[str, Any]:
    """The rendering contract, for the run manifest.

    Recorded so the image transform is reproducible and so a future change to it
    is visible in an artifact rather than only in a diff.
    """
    return {
        "bands": list(RGB_BANDS),
        "band_order": "RGB",
        "normalisation": "per-band percentile stretch",
        "percentiles": list(percentiles),
        "output_dtype": "uint8",
        "resize": size,
        "note": (
            "per-patch stretch, not corpus-global, so inference needs no "
            "training-set statistics"
        ),
    }


# ---------------------------------------------------------------------------
# Instruction generation
# ---------------------------------------------------------------------------
def _check_families_have_generators(families: Sequence[str]) -> None:
    """Refuse, before discovery, any requested family with no generator.

    `dominance` is well-posed on a single-label corpus (`FAMILY_NEEDS_MULTI_LABEL`
    is `False`), so `_check_family_is_well_posed` passes it -- but no generator
    emits it, so the generation loop would produce zero samples and the failure
    would surface as the generic "no instruction samples were generated" error,
    blaming the data for a configuration mistake. Naming the un-generatable
    families and listing the ones that do exist is the whole point: it is a
    caller error, and it is reported as one.
    """
    requested = list(dict.fromkeys(families))
    unsupported = [f for f in requested if f not in FAMILIES_WITH_GENERATORS]
    if unsupported:
        raise VLMDataError(
            f"instruction families {unsupported} have no generator implemented; "
            f"generators exist for {list(FAMILIES_WITH_GENERATORS)}. Requesting a "
            f"family without a generator produces no samples and would be reported "
            f"as a data problem rather than a configuration one.",
            context={
                "requested": requested,
                "without_generator": unsupported,
                "with_generator": list(FAMILIES_WITH_GENERATORS),
            },
        )


def _check_family_is_well_posed(
    families: Sequence[str], *, n_single_label: int, n_matched: int
) -> list[str]:
    """Refuse to generate a family the corpus cannot support honestly."""
    warnings: list[str] = []
    all_single = n_matched > 0 and n_single_label == n_matched
    for family in families:
        needs_multi = FAMILY_NEEDS_MULTI_LABEL.get(family)
        if needs_multi is None:
            raise VLMDataError(
                f"unknown instruction family {family!r}; expected one of "
                f"{sorted(FAMILY_NEEDS_MULTI_LABEL)}"
            )
        if needs_multi and all_single:
            raise VLMDataError(
                f"instruction family {family!r} needs multi-label patches, but all "
                f"{n_matched} labelled patches in this corpus carry exactly ONE "
                f"label. Generating it would teach the model that a scene has a "
                f"single land-cover class, which is a property of this subset "
                f"(the single-label selection made for the Phase 12 metric), not "
                f"of remote sensing. Use presence/absence, or supply a "
                f"multi-label BigEarthNet sample.",
                context={
                    "family": family,
                    "n_matched": n_matched,
                    "n_single_label": n_single_label,
                },
            )
        if needs_multi:
            warnings.append(
                f"instruction family {family!r} is enabled on a partly "
                f"multi-label corpus ({n_matched - n_single_label} multi-label "
                f"patches); enumeration answers will be short for the rest"
            )
    return warnings


def _presence_pairs(
    labels: tuple[str, ...],
    *,
    vocabulary: Sequence[str],
    negative_ratio: float,
    rng: Any,
) -> list[tuple[str, str, str]]:
    """(question, answer, family) triples for presence/absence.

    Balanced by `negative_ratio`: at 1.0 each present class contributes one
    negative, so a model cannot reach 50% by always answering "yes". That matters
    more here than usual -- with one present label and 18 absent classes the naive
    generator would produce a corpus that is ~5% positive.
    """
    present = [l for l in labels if l in vocabulary]
    absent = [l for l in vocabulary if l not in set(present)]

    out: list[tuple[str, str, str]] = []
    for label in present:
        out.append((f"Is {label} present in this image?", "Yes.", "presence"))

    n_negative = min(len(absent), max(1, round(len(present) * negative_ratio)))
    if n_negative:
        for label in rng.sample(absent, n_negative):
            out.append(
                (f"Is {label} present in this image?", "No.", "presence")
            )
    return out


def build_corpus(
    config: Any,
    *,
    limit: int | None = None,
    progress: Any = None,
) -> CorpusBuild:
    """Build the full instruction corpus: discover, split, generate.

    Deterministic given `config.seed`: the split shuffle, the negative-class
    sampling, and the family choice are all seeded from it, so a re-run produces
    an identical corpus and therefore an identical manifest hash.

    Refuses a requested family that has no generator **before** touching the
    data, so a configuration error is reported as one instead of as a data
    problem (see `_check_families_have_generators`).
    """
    import random

    from evaluation.leakage import assert_no_scene_overlap
    from training.data.bigearthnet import CLC19_CLASSES
    from training.data.bigearthnet_blocks import (
        count_blocks_by_split,
        parse_ground_cell,
        parse_tile,
        reconstruct_blocks,
    )

    # Configuration is checked before discovery: an un-generatable family is a
    # caller error, and discovering patches first would only make it slower and
    # easier to misattribute.
    _check_families_have_generators(config.instruction_families)

    patches, coverage, warnings = discover_labelled_patches(
        corpus_root=config.corpus_root,
        metadata_parquet=config.metadata_parquet,
        limit=limit,
    )

    # -- scene keys: T2 spatial blocks --------------------------------------
    # One scene key per PATCH, not per instruction pair: the split is a property
    # of the imagery, and assigning it per pair could put two questions about the
    # same image on opposite sides of the boundary.
    missing_split = [p.patch_id for p in patches if p.official_split is None]
    if missing_split:
        raise VLMDataError(
            f"{len(missing_split)} patches carry no split in the release manifest "
            f"(e.g. {missing_split[:3]}). The T2 policy uses the release's own "
            f"partition, so a patch without one cannot be placed without inventing "
            f"a split -- which is the thing the tile-based policy was discarded for.",
            context={"n_missing": len(missing_split)},
        )

    cell_of: dict[str, tuple[str, int, int]] = {}
    cells: list[tuple[str, int, int, str]] = []
    for p in patches:
        row, col = parse_ground_cell(p.patch_id)
        # `p.tile` is the reBEN *acquisition folder* (`S2A_MSIL2A_<date>_..._T33UUP`),
        # NOT the MGRS tile. Using it here would give every seasonal acquisition of
        # one ground location its own namespace, so the same `(row, col)` would
        # receive a different scene id per acquisition -- and `reconstruct_blocks`
        # deliberately omits the split from the id precisely so that a straddling
        # block would be caught by the disjointness check. Different ids per
        # acquisition would make that check pass on a corpus that leaked.
        # `parse_tile` is what yields the MGRS tile (`33UUP`), which is the
        # namespace the block ids in the Phase 12 record are built in.
        tile = str(parse_tile(p.patch_id))
        cell = (tile, int(row), int(col))
        cell_of[p.patch_id] = cell
        cells.append((cell[0], cell[1], cell[2], str(p.official_split)))

    # Components are built per `(tile, split)`, so a block can never straddle the
    # partition -- that is what makes the block a valid scene key for a split the
    # release drew on ground locations.
    cell_to_scene = reconstruct_blocks(cells)
    cell_splits = {(t, r, c): s for t, r, c, s in cells}
    blocks_by_split = count_blocks_by_split(cell_to_scene, cell_splits)

    patches = [
        PatchSample(
            patch_id=p.patch_id,
            scene_id=cell_to_scene[cell_of[p.patch_id]],
            tile=p.tile,
            directory=p.directory,
            labels=p.labels,
            band_paths=p.band_paths,
            official_split=p.official_split,
            split=str(p.official_split),
        )
        for p in patches
    ]

    # Enforced, not hoped for. A block appearing in two splits means the model is
    # evaluated on imagery it trained on.
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        assert_no_scene_overlap(
            [p for p in patches if p.split == left],
            [p for p in patches if p.split == right],
        )

    # -- the registry's ratios, now a sanity check rather than a generator ---
    n_total = len(patches)
    actual_ratios = {
        split: sum(1 for p in patches if p.split == split) / n_total
        for split in ("train", "val", "test")
    }
    for split, expected in (("train", config.train_ratio), ("val", config.val_ratio)):
        if abs(actual_ratios[split] - float(expected)) > SPLIT_RATIO_TOLERANCE:
            warnings.append(
                f"the release's {split} proportion is {actual_ratios[split]:.3f}, "
                f"differing from the registry's {float(expected):.3f} by more than "
                f"SPLIT_RATIO_TOLERANCE={SPLIT_RATIO_TOLERANCE}. The release's own "
                f"partition is used regardless; this is reported so the divergence "
                f"is visible in the run manifest."
            )

    # -- family admissibility, decided before generating anything -----------
    warnings += _check_family_is_well_posed(
        config.instruction_families,
        n_single_label=coverage.n_single_label,
        n_matched=coverage.n_matched,
    )

    # -- generate -----------------------------------------------------------
    rng = random.Random(config.seed)
    samples: list[InstructionSample] = []
    for patch in patches:
        if "presence" in config.instruction_families:
            triples = _presence_pairs(
                patch.labels,
                vocabulary=CLC19_CLASSES,
                negative_ratio=config.negative_ratio,
                rng=rng,
            )
        else:
            triples = []
        for i, (question, answer, family) in enumerate(triples):
            samples.append(
                InstructionSample(
                    sample_id=f"{patch.patch_id}#{family}{i}",
                    patch_id=patch.patch_id,
                    scene_id=patch.scene_id,
                    split=patch.split,
                    question=question,
                    answer=answer,
                    family=family,
                    directory=patch.directory,
                    band_paths=patch.band_paths,
                )
            )
        if progress is not None and len(samples) and len(samples) % 5000 < 2:
            progress(len(samples))

    if not samples:
        raise VLMDataError(
            f"no instruction samples were generated from {len(patches)} patches "
            f"with families {list(config.instruction_families)}; a training run "
            f"over an empty corpus would produce an adapter that learned nothing",
            context={"n_patches": len(patches)},
        )

    from training.data.bigearthnet_labels import BigEarthNetLabelIndex

    index = BigEarthNetLabelIndex.from_parquet(config.metadata_parquet)

    return CorpusBuild(
        samples=samples,
        patches=patches,
        coverage=coverage,
        class_counts=index.class_counts(p.patch_id for p in patches),
        render=describe_render(percentiles=config.rgb_percentiles),
        families=tuple(config.instruction_families),
        split_info={
            "policy": "release_partition_keyed_by_T2_blocks",
            "scene_key": "ben_<tile>:<k>",
            "scene_key_module": "training.data.bigearthnet_blocks.reconstruct_blocks",
            "leakage_check": "evaluation.leakage.assert_no_scene_overlap",
            "ratios": actual_ratios,
            "blocks_by_split": blocks_by_split,
            "patches_by_split": {
                split: sum(1 for p in patches if p.split == split)
                for split in ("train", "val", "test")
            },
            "registry_ratios": {
                "train_ratio": config.train_ratio,
                "val_ratio": config.val_ratio,
                "tolerance": SPLIT_RATIO_TOLERANCE,
            },
            "note": (
                "blocks are reconstructed from this corpus's patches, not the full "
                "release; that can only split a block further, never merge across "
                "a partition, so the grouping is conservative"
            ),
        },
        warnings=warnings,
    )
