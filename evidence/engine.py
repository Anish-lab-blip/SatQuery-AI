"""SatQuery AI — the evidence engine (Phase 13).

WHAT IT IS FOR
--------------
`docs/ARCHITECTURE_FREEZE.md` section 5 gives each layer one verb:

    router *understands*; policy engine *decides*; specialists *compute*;
    VLM *explains*; evidence engine *proves*.

This module is the "proves" step. Specialists each emit `Evidence` for what
*they* computed (see `GroundingSpecialist._build_evidence`). This engine does
not re-derive any of that. Its job is aggregation:

    collect across specialists -> order -> deduplicate -> renumber -> bound

WHY AGGREGATION NEEDS ITS OWN MODULE
------------------------------------
Three problems appear only once more than one specialist is in play, and none of
them is solved inside a single specialist:

1. **Stable identity.** `Evidence.evidence_id` defaults to a random uuid, which
   is fine within one result but useless when the controller merges four
   specialists' evidence into the trace of a single run. Nothing can be cited.
   The engine renumbers to `evidence_001`, `evidence_002`, ... so a downstream
   artefact (report, UI, audit) can reference one item deterministically.

2. **Order.** `Evidence` items have no ordering field. Without a defined sort the
   collection's order follows whatever order the specialists happened to finish
   in, so the same inputs produce a different JSON on a different run. Pure
   functions must not do that, and this whole system's reproducibility argument
   rests on it.

3. **Duplication.** The VQA specialist emits a `STATISTIC` carrying its answer;
   that same answer also travels in `SpecialistResult.answer`. Two specialists
   that both georeference the same asset emit the same `GEOLOCATION`. Neither is
   wrong, and neither knows about the other, so dedup has to happen here.

THE PURITY CONTRACT
-------------------
`aggregate` is pure and deterministic:

    * source results are never mutated -- `model_copy` is used to rebuild the
      collection rather than editing `Evidence.evidence_id` in place;
    * ids are assigned from the sorted position, not from input order;
    * dedup is order-insensitive by construction (the key is built from the
      sorted specialist list);
    * no clock, no RNG, no I/O.

Same inputs -> byte-identical output. `tests/unit/test_evidence_engine.py`
pins this.

IDENTITY IS DEFINED, NOT ASSUMED
--------------------------------
`Evidence.evidence_id` is a uuid default, so it cannot be part of an identity
key -- two structurally identical items from two runs would never dedup. The key
is therefore the *content* of the observation:

    (type, source_specialist, coordinate_system, rounded coordinates,
     rounded score)

`payload` and `artifact_ref` are deliberately EXCLUDED. Two items that agree on
the same geolocation, one carrying a `crs` in its payload and one not, have made
the same claim about the world; deduplicating them is correct, and the surviving
item's payload is merged with the discarded one's so the `crs` is not lost.
Conversely two items that share a type and score but disagree on coordinates are
two different claims and are both kept -- which matters, because suppressing a
spatial disagreement would be exactly the silent contradiction the freeze
forbids.

DIFFERENT SPECIALISTS AGREEING
------------------------------
`source_specialist` is a single string, not a set (schema is frozen). So two
specialists making the same claim are two DIFFERENT items by the key above, and
both survive. That is the correct conservative behaviour: collapsing them would
require the engine to pick a winner, and the engine has no basis for that. What
it does instead is keep both and record the disagreement-free agreement in the
collection's `sources` and in each item's `payload` via `_record_agreement`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.schemas import (
    Box,
    ConfidenceBreakdown,
    CoordinateSystem,
    Evidence,
    EvidenceType,
    GeoMetadata,
    Region,
    SpecialistResult,
)

from evidence.confidence import (
    TemperatureCalibration,
    calibrate,
    calibrate_result,
)

#: Prefix for engine-assigned evidence ids. Sequential, zero-padded to three
#: digits so lexical sort matches numeric sort up to 999 items -- well past the
#: `evidence.max_items` bound of 32.
ID_PREFIX = "evidence"

#: Default cap on the aggregated collection, mirroring `evidence.max_items` in
#: `configs/base.yaml`. The controller passes the configured value; this is the
#: fallback so the engine is usable without a config.
DEFAULT_MAX_ITEMS = 32

#: Rounding applied before an evidence item is turned into a dedup key. Six
#: decimals on normalized coordinates is ~1e-6 of the frame, far below any
#: meaningful spatial difference, but enough to absorb float noise from two
#: specialists rounding the same value two different ways.
_KEY_PRECISION = 6


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), _KEY_PRECISION)


def _identity_key(item: Evidence) -> tuple[Any, ...]:
    """The content identity of one evidence observation.

    Built from the claim, not from the container: see the module docstring for
    why `payload`, `artifact_ref` and `evidence_id` are excluded.
    """
    coords = (
        tuple(_round(c) for c in item.coordinates)
        if item.coordinates is not None
        else None
    )
    return (
        item.type.value,
        item.source_specialist,
        item.coordinate_system.value if item.coordinate_system else None,
        coords,
        _round(item.score),
    )


def _claim_key(item: Evidence) -> tuple[Any, ...]:
    """Identity of the *claim*, ignoring which specialist made it.

    Used only to detect corroboration. Deliberately distinct from
    `_identity_key`: that one answers "is this the same item", this one answers
    "are these two specialists saying the same thing about the world".
    """
    coords = (
        tuple(_round(c) for c in item.coordinates)
        if item.coordinates is not None
        else None
    )
    return (
        item.type.value,
        item.coordinate_system.value if item.coordinate_system else None,
        coords,
        _round(item.score),
    )


def _sort_key(item: Evidence) -> tuple[Any, ...]:
    """Total order over evidence, independent of specialist completion order.

    Ordering rationale, in priority order:

    1. `type` -- groups like with like, so a reader sees all the boxes together.
    2. specialist name -- stable and meaningful, unlike the uuid.
    3. score DESCENDING -- within a type, the strongest claim leads. Reversed via
       `-score` so the tuple stays uniformly ascending.
    4. coordinates -- the tie-breaker that makes the order total. Without it two
       items of the same type, specialist and score would sort by Python's
       stable-sort insertion order, which reintroduces input-order dependence.

    Items with no score sort AFTER items with a score: a missing score is not
    evidence of strength.
    """
    if item.score is None:
        score_key: tuple[int, float] = (1, 0.0)
    else:
        score_key = (0, -float(item.score))
    coords = (
        tuple(_round(c) or 0.0 for c in item.coordinates)
        if item.coordinates is not None
        else ()
    )
    return (item.type.value, item.source_specialist, score_key, coords)


@dataclass(frozen=True)
class EvidenceCollection:
    """The aggregated, ordered, deduplicated, renumbered evidence set.

    Attributes:
        items: the canonical evidence, ids `evidence_001`... in sort order.
        sources: sorted specialist names that contributed anything, BEFORE the
            cap was applied. Provenance is a property of the run, not of the
            surviving items.
        dropped_duplicates: how many items were merged into an existing claim.
            Non-zero is normal and healthy -- it means two specialists agreed.
        dropped_over_limit: how many items the cap discarded. Non-zero is a
            warning: it means a specialist's evidence did not survive. Recorded
            rather than silently truncated.
        total_before_limit: the deduplicated count prior to capping.
        truncated: True when `dropped_over_limit > 0`. A boolean summary so a
            trace can branch on it without recomputing.
    """

    items: list[Evidence] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    dropped_duplicates: int = 0
    dropped_over_limit: int = 0
    total_before_limit: int = 0
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def ids(self) -> list[str]:
        return [item.evidence_id for item in self.items]

    def by_type(self, evidence_type: EvidenceType) -> list[Evidence]:
        return [item for item in self.items if item.type is evidence_type]

    def by_specialist(self, name: str) -> list[Evidence]:
        return [item for item in self.items if item.source_specialist == name]

    #: Payload key under which `_record_agreement` stashes the specialists that
    #: made the identical claim. A single string field cannot hold a set, so the
    #: agreement is recorded in the payload -- which is exactly what the payload
    #: is for: observations that support the evidence but are not the claim.
    AGREEMENT_KEY = "corroborated_by"

    def summary(self) -> dict[str, Any]:
        """Observable trace facts. No chain-of-thought, no interpretation."""
        return {
            "returned": len(self.items),
            "total_before_limit": self.total_before_limit,
            "dropped_duplicates": self.dropped_duplicates,
            "dropped_over_limit": self.dropped_over_limit,
            "truncated": self.truncated,
            "sources": list(self.sources),
            "types": sorted({item.type.value for item in self.items}),
        }


class EvidenceEngine:
    """Aggregates specialist evidence into one canonical, cited collection.

    Stateless and deterministic. Construct once, call as often as needed.

    Args:
        max_items: cap on the returned collection. Must be >= 1; a cap of zero
            would silently discard every piece of evidence, which is a policy
            decision this engine has no business making on its own.
        calibration: an optional fitted temperature-scaling artifact, applied by
            `confidence_for`. When `None` the engine degrades honestly -- see
            `evidence.confidence.calibrate`.
    """

    def __init__(
        self,
        *,
        max_items: int = DEFAULT_MAX_ITEMS,
        calibration: TemperatureCalibration | None = None,
    ) -> None:
        if max_items < 1:
            raise ValueError(f"max_items must be >= 1, got {max_items}")
        self.max_items = max_items
        self.calibration = calibration

    # -- primary entry point ----------------------------------------------

    def aggregate(
        self,
        results: SpecialistResult | Sequence[SpecialistResult] | None = None,
        *,
        evidence: Iterable[Evidence] | None = None,
    ) -> EvidenceCollection:
        """Merge evidence from one or more specialist results.

        Accepts either form because the controller has results and a test has a
        bare evidence list; forcing callers to wrap one in the other would add
        a fake `SpecialistResult` to every unit test.

        Args:
            results: one result or a sequence of them. Specialists state their
                evidence twice -- in `SpecialistResult.evidence` and via
                `Specialist.produce_evidence`, which currently delegates to the
                same list -- so only `result.evidence` is read. Reading both
                would double-count every item and inflate the duplicate count.
            evidence: a raw evidence iterable, used INSTEAD of `results`.

        Returns:
            An `EvidenceCollection`. Never raises on empty input.

        Raises:
            ValueError: neither argument produced any evidence source.
        """
        if results is None and evidence is None:
            raise ValueError("aggregate() requires results= or evidence=")
        if results is not None and evidence is not None:
            raise ValueError("aggregate() accepts results= or evidence=, not both")

        collected: list[Evidence] = []
        if evidence is not None:
            collected.extend(evidence)
        elif isinstance(results, SpecialistResult):
            collected.extend(results.evidence)
        else:
            for result in results:  # type: ignore[union-attr]
                collected.extend(result.evidence)

        return self._canonicalise(collected)

    # -- pipeline ---------------------------------------------------------

    def _canonicalise(self, raw: Iterable[Evidence]) -> EvidenceCollection:
        """Dedup -> sort -> renumber -> cap. The whole pipeline, in one place."""
        items = list(raw)
        sources = sorted({item.source_specialist for item in items})

        deduped, dropped_duplicates = self._deduplicate(items)
        ordered = sorted(deduped, key=_sort_key)
        # Two specialists can make the same claim, and `deduplicate` keeps both
        # (different `source_specialist` -> different key). Record the agreement
        # now that ordering is fixed, so the annotation is part of the pure
        # pipeline rather than a post-hoc edit a caller might forget.
        annotated = self._record_agreement(ordered)

        total = len(annotated)
        capped = annotated[: self.max_items]
        dropped_over_limit = total - len(capped)

        return EvidenceCollection(
            items=[
                self._renumber(item, index) for index, item in enumerate(capped, start=1)
            ],
            sources=sources,
            dropped_duplicates=dropped_duplicates,
            dropped_over_limit=dropped_over_limit,
            total_before_limit=total,
            truncated=dropped_over_limit > 0,
        )

    @staticmethod
    def _deduplicate(items: list[Evidence]) -> tuple[list[Evidence], int]:
        """Collapse byte-identical restatements of the same claim.

        Within one specialist, the same observation can appear twice (a
        specialist that both emits a box and re-emits it as a region, say). The
        key includes `source_specialist`, so only same-specialist repeats are
        merged here; two specialists agreeing is handled by `_record_agreement`,
        which preserves the second specialist's identity rather than discarding
        it.

        Payloads are merged on collision: a discarded duplicate that carried a
        `crs` or a `prompt_version` the survivor lacked still contributes it.
        """
        seen: dict[tuple[Any, ...], Evidence] = {}
        merged: list[Evidence] = []
        dropped = 0

        for item in items:
            key = _identity_key(item)
            existing = seen.get(key)
            if existing is None:
                seen[key] = item
                merged.append(item)
                continue

            dropped += 1
            # Survivor's own keys win; the loser only fills gaps. Mutating the
            # survivor in place would be fine here because `merged` holds the
            # same object, but rebuilding keeps the "never mutate a source
            # result" contract true even if the caller kept a reference.
            payload = {**item.payload, **existing.payload}
            if payload != existing.payload:
                updated = existing.model_copy(update={"payload": payload})
                seen[key] = updated
                merged[merged.index(existing)] = updated

        return merged, dropped

    @staticmethod
    def _record_agreement(items: list[Evidence]) -> list[Evidence]:
        """Annotate items that several specialists claim identically.

        The claim itself ignores `source_specialist` -- two specialists reporting
        the same geolocation agree about the world. This pass finds those groups
        and stamps the *other* specialists' names into each member's payload, so
        a consumer can tell "these agree" from "these are two unrelated items
        that happen to share a type".

        Nothing is removed: dropping a member would throw away a specialist's
        attribution, and the freeze does not permit silently discarding a
        specialist's evidence.
        """
        groups: dict[tuple[Any, ...], list[int]] = {}
        for index, item in enumerate(items):
            claim = _claim_key(item)
            groups.setdefault(claim, []).append(index)

        out = list(items)
        for indexes in groups.values():
            if len(indexes) < 2:
                continue
            names = sorted({items[i].source_specialist for i in indexes})
            if len(names) < 2:
                # Same specialist restating itself across the collection. The
                # identity key would normally have merged these, so reaching
                # here means the payloads differed; it is not corroboration.
                continue
            for i in indexes:
                item = out[i]
                others = [n for n in names if n != item.source_specialist]
                payload = {**item.payload, EvidenceCollection.AGREEMENT_KEY: others}
                out[i] = item.model_copy(update={"payload": payload})
        return out

    @staticmethod
    def _renumber(item: Evidence, index: int) -> Evidence:
        """Give one item its canonical id, leaving the source untouched."""
        return item.model_copy(update={"evidence_id": f"{ID_PREFIX}_{index:03d}"})

    # -- canonical vocabulary ---------------------------------------------

    @staticmethod
    def evidence_type_for(result: SpecialistResult) -> EvidenceType:
        """The canonical evidence type for a specialist's primary output.

        This is the ONE place that knows the mapping from specialist task to
        evidence vocabulary, so no specialist needs to branch on it. Note it
        returns the *primary* type; a specialist's own `produce_evidence` emits
        its real per-artefact types, and those are preserved by `aggregate`.

        The members are exactly those of `core.schemas.EvidenceType`. The
        freeze's section 3 prose list omits `availability_mask`; that member is
        real (C-1: modality trust evidence) and is included here because a
        missing member would otherwise force a wrong fallback.
        """
        from core.schemas import Task

        mapping = {
            Task.GROUNDING: EvidenceType.BOUNDING_BOX,
            Task.CHANGE: EvidenceType.CHANGE_MAP,
            Task.OPTICAL_SAR: EvidenceType.JOINT_FEATURE_REGION,
            Task.VQA: EvidenceType.STATISTIC,
            Task.CAPTION: EvidenceType.STATISTIC,
            Task.UNSUPPORTED: EvidenceType.STATISTIC,
        }
        return mapping.get(result.task, EvidenceType.STATISTIC)

    # -- artifact-derived evidence ----------------------------------------

    def evidence_from_box(
        self,
        box: Box,
        *,
        source_specialist: str,
        payload: dict[str, Any] | None = None,
        asset_ref: str | None = None,
    ) -> Evidence:
        """Wrap a result `Box` as canonical `BOUNDING_BOX` evidence.

        Convenience for the controller when it needs evidence for a box that a
        specialist emitted but did not itself describe. The `coordinate_system`
        is copied from the box, never assumed -- `core.schemas` requires it and
        `Evidence` rejects spatial coordinates without one.
        """
        return Evidence(
            type=EvidenceType.BOUNDING_BOX,
            source_specialist=source_specialist,
            coordinate_system=box.coordinate_system,
            coordinates=[box.x1, box.y1, box.x2, box.y2],
            score=box.score,
            artifact_ref=asset_ref,
            payload={"label": box.label, **(payload or {})},
        )

    def evidence_from_region(
        self,
        region: Region,
        *,
        source_specialist: str,
        payload: dict[str, Any] | None = None,
        asset_ref: str | None = None,
    ) -> Evidence:
        """Wrap a result `Region` as evidence.

        A region carrying a mask reference becomes `MASK` evidence; one with only
        a box becomes `BOUNDING_BOX` evidence. A region with neither is reported
        as a `STATISTIC` rather than a spatial claim, because there is no
        geometry to prove.
        """
        body: dict[str, Any] = {
            "region_id": region.region_id,
            "label": region.label,
            **(payload or {}),
        }
        if region.mask_ref:
            body["mask_ref"] = region.mask_ref
            return Evidence(
                type=EvidenceType.MASK,
                source_specialist=source_specialist,
                coordinate_system=region.coordinate_system,
                coordinates=(
                    [region.box.x1, region.box.y1, region.box.x2, region.box.y2]
                    if region.box is not None
                    else None
                ),
                score=region.score if region.score is not None else None,
                artifact_ref=asset_ref or region.mask_ref,
                payload=body,
            )
        if region.box is not None:
            return Evidence(
                type=EvidenceType.BOUNDING_BOX,
                source_specialist=source_specialist,
                coordinate_system=region.box.coordinate_system,
                coordinates=[
                    region.box.x1, region.box.y1, region.box.x2, region.box.y2,
                ],
                score=region.score if region.score is not None else region.box.score,
                artifact_ref=asset_ref,
                payload=body,
            )
        return Evidence(
            type=EvidenceType.STATISTIC,
            source_specialist=source_specialist,
            score=region.score,
            artifact_ref=asset_ref,
            payload=body,
        )

    def evidence_from_geospatial(
        self,
        geo: GeoMetadata,
        *,
        source_specialist: str,
        bounds: list[float] | None = None,
        score: float | None = None,
    ) -> Evidence | None:
        """Record that a raster is georeferenced and where it sits.

        Returns `None` when there is no CRS to report. Emitting a `GEOLOCATION`
        item without a CRS would assert a placement the data does not support,
        which is the same failure mode the grounding degenerate-box guard exists
        to prevent.
        """
        if not geo.has_crs and not geo.crs:
            return None
        return Evidence(
            type=EvidenceType.GEOLOCATION,
            source_specialist=source_specialist,
            coordinate_system=CoordinateSystem.GEO if bounds else None,
            coordinates=list(bounds) if bounds else None,
            score=score,
            payload={
                "crs": geo.crs,
                "bounds": list(geo.bounds) if geo.bounds else None,
                "width": geo.width,
                "height": geo.height,
                "is_georeferenced": geo.is_georeferenced,
            },
        )

    # -- calibration -------------------------------------------------------

    def confidence_for(
        self,
        result: SpecialistResult | ConfidenceBreakdown,
        *,
        extra_components: dict[str, float] | None = None,
        degraded: bool | None = None,
        degradation_reason: str | None = None,
    ) -> ConfidenceBreakdown:
        """Calibrate a specialist's confidence through this engine's artifact.

        Accepts either a full result or the breakdown alone, so a caller that
        already has the pieces does not rebuild a `SpecialistResult` to use it.

        The specialist's own degradation verdict WINS unless explicitly
        overridden: it knows things the engine does not (a zero-shot fallback, a
        failed input-quality gate), and overwriting `degraded=False` here would
        launder a degraded result into a confident-looking one.

        When no calibration artifact is configured this is a pass-through that
        reports `method="uncalibrated"` -- never a manufactured fitted mapping.
        """
        if isinstance(result, ConfidenceBreakdown):
            breakdown = result
            components = dict(breakdown.components)
            components.update(extra_components or {})
            final_degraded = breakdown.degraded if degraded is None else degraded
            final_reason = (
                breakdown.degradation_reason
                if degradation_reason is None
                else degradation_reason
            )
            return calibrate(
                breakdown.raw,
                self.calibration,
                extra_components=components,
                degraded=final_degraded,
                degradation_reason=final_reason,
            )

        components = dict(result.confidence.components)
        components.update(extra_components or {})
        final_degraded = result.confidence.degraded if degraded is None else degraded
        if final_degraded and not result.confidence.degraded:
            # The engine is marking a result degraded that the specialist did
            # not. Source that to the result's own warnings when it has one, so
            # the reason is a fact rather than a restatement of the boolean.
            final_degraded = True
        final_reason = (
            result.confidence.degradation_reason
            if degradation_reason is None
            else degradation_reason
        )
        if final_reason is None and final_degraded:
            final_reason = result.warnings[0] if result.warnings else "aggregate degraded"

        return calibrate(
            result.confidence.raw,
            self.calibration,
            extra_components=components,
            degraded=final_degraded,
            degradation_reason=final_reason,
        )

    def calibrate_result(
        self, result: SpecialistResult
    ) -> ConfidenceBreakdown:
        """Calibrate a result's existing breakdown, preserving its provenance."""
        return calibrate_result(result.confidence, self.calibration)

    # -- config ------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        calibration: TemperatureCalibration | None = None,
        base_dir: str | None = None,
    ) -> "EvidenceEngine":
        """Build from the central config, honouring `evidence.max_items`.

        `calibration` is explicit rather than auto-loaded: whether a fitted
        artifact exists is an operational fact the caller may know better than
        the config file does, and silently loading one would make the engine's
        behaviour depend on filesystem state.
        """
        return cls(
            max_items=int(config.get("evidence.max_items", DEFAULT_MAX_ITEMS)),
            calibration=calibration,
        )


def aggregate_evidence(
    results: SpecialistResult | Sequence[SpecialistResult],
    *,
    max_items: int = DEFAULT_MAX_ITEMS,
    calibration: TemperatureCalibration | None = None,
) -> EvidenceCollection:
    """Module-level shorthand for the common one-shot aggregation."""
    return EvidenceEngine(max_items=max_items, calibration=calibration).aggregate(
        results
    )


def evidence_digest(collection: EvidenceCollection | Sequence[Evidence]) -> str:
    """Stable digest of an evidence collection, for tests and traces.

    Built from `_identity_key` -- the content of each claim, in order -- and NOT
    from the assigned ids or from payloads. So it changes when the claims change
    and does not change when a specialist restates the same claim under a new
    uuid. That is what makes it useful as a reproducibility assertion:

        aggregate(inputs_a) is reproducible iff digest(a) == digest(b)

    against the same input, even across processes where the uuid defaults differ.

    The corroboration annotation in the payload IS excluded, because it is a
    derived observation about the collection, not part of the claim.
    """
    items = (
        collection.items
        if isinstance(collection, EvidenceCollection)
        else list(collection)
    )
    hasher = hashlib.sha256()
    for item in items:
        hasher.update(repr(_identity_key(item)).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


__all__ = [
    "DEFAULT_MAX_ITEMS",
    "ID_PREFIX",
    "EvidenceCollection",
    "EvidenceEngine",
    "aggregate_evidence",
    "evidence_digest",
]
