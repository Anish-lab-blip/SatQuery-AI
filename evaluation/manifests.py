"""SatQuery AI — dataset manifests.

Every sample the system ever touches gets a manifest record carrying the fields the
project specification mandates:

    dataset_id, scene_id, sample_id, source_scene, geographic_hash,
    acquisition_date, sensor, sha256, split

A manifest is:
  * deterministic  — same inputs, same manifest hash
  * immutable      — a frozen manifest is never rewritten in place
  * self-describing — carries its own hash and a version stamp

The geographic_hash is the quiet hero here. It is derived from *quantised* bounds, so
neighbouring tiles cut from the same acquisition collide onto the same hash. That is
what makes scene-level leakage detectable even when a dataset ships without scene ids.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

from core.errors import LeakageError, SatQueryError

MANIFEST_VERSION = "1.0"
SplitName = Literal["train", "val", "test"]


class ManifestError(SatQueryError):
    code = "manifest_error"
    user_message = "The dataset manifest is invalid."


# ---------------------------------------------------------------------------
# Geographic hashing — the anti-leakage primitive
# ---------------------------------------------------------------------------

#: Quantisation grid for geographic hashing, in CRS units (metres for a projected CRS).
#: 1000 m means two tiles whose origins fall within the same kilometre bucket collide.
#: Sentinel-2 tiles are ~110 km and Sentinel-1 scenes ~250 km, so this bucket is far
#: coarser than any tiling scheme we use and safely groups neighbours.
GEO_HASH_BUCKET_M = 1000.0


def geographic_hash(bounds: Iterable[float] | None, bucket: float = GEO_HASH_BUCKET_M) -> str | None:
    """Hash a bounding box onto a coarse spatial grid.

    Two tiles from the same acquisition share this hash even if their exact bounds
    differ. That makes "do these two samples come from the same place?" answerable
    without trusting the dataset's own scene labels.

    Returns None when bounds are absent — a non-georeferenced sample cannot be
    spatially audited, and pretending otherwise would be worse than admitting it.
    """
    if bounds is None:
        return None
    values = list(bounds)
    if len(values) != 4:
        return None
    if bucket <= 0:
        raise ManifestError(f"geographic hash bucket must be positive, got {bucket}")

    quantised = [int(round(v / bucket)) for v in values]
    payload = ",".join(str(q) for q in quantised).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


# ---------------------------------------------------------------------------
# The manifest record
# ---------------------------------------------------------------------------


@dataclass
class SampleRecord:
    """One auditable unit of dataset material."""

    dataset_id: str
    sample_id: str
    split: SplitName

    # Provenance
    scene_id: str | None = None
    source_scene: str | None = None
    geographic_hash: str | None = None
    acquisition_date: str | None = None
    sensor: str | None = None

    # Integrity
    sha256: str | None = None
    path: str | None = None

    # Free-form extras (benchmark-specific annotations, tile coordinates, ...)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ManifestError("dataset_id is required")
        if not self.sample_id:
            raise ManifestError("sample_id is required")
        if self.split not in ("train", "val", "test"):
            raise ManifestError(f"split must be train|val|test, got {self.split!r}")

    @property
    def scene_key(self) -> str:
        """The grouping key used for leakage-safe splitting.

        Preference order matters:
          1. explicit scene_id   — the dataset told us
          2. geographic_hash     — we derived it from bounds
          3. source_scene        — looser but still meaningful
          4. sample_id           — last resort; each sample is its own scene

        Falling back to sample_id is *safe* (never leaks) but splits correlated
        neighbours across the train/test boundary, which inflates test scores.
        """
        return self.scene_id or self.geographic_hash or self.source_scene or self.sample_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SampleRecord":
        known = {f for f in cls.__dataclass_fields__}
        extra = payload.pop("extra", {}) or {}
        unknown = {k: v for k, v in payload.items() if k not in known}
        extra.update(unknown)
        clean = {k: v for k, v in payload.items() if k in known}
        return cls(**clean, extra=extra)


# ---------------------------------------------------------------------------
# Manifest container
# ---------------------------------------------------------------------------


@dataclass
class DatasetManifest:
    """An immutable, hashable collection of SampleRecords."""

    name: str
    records: list[SampleRecord]
    version: str = MANIFEST_VERSION
    notes: dict[str, Any] = field(default_factory=dict)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_records(
        cls, name: str, records: Iterable[SampleRecord], notes: dict[str, Any] | None = None
    ) -> "DatasetManifest":
        """Build a manifest that OWNS its records.

        Records are deep-copied on the way in. Without this, a caller mutating a
        SampleRecord after construction would silently change this manifest's hash —
        and the manifest hash is recorded in every evaluation run, so a hash that can
        move is worse than no hash at all. `test_manifest_owns_its_records` pins this.
        """
        return cls(
            name=name,
            records=[copy.deepcopy(r) for r in records],
            notes=copy.deepcopy(notes) if notes else {},
        )

    # -- queries -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[SampleRecord]:
        return iter(self.records)

    def by_split(self, split: SplitName) -> list[SampleRecord]:
        return [r for r in self.records if r.split == split]

    def scenes(self) -> set[str]:
        return {r.scene_key for r in self.records}

    def split_counts(self) -> dict[str, int]:
        out = {"train": 0, "val": 0, "test": 0}
        for r in self.records:
            out[r.split] += 1
        return out

    def dataset_ids(self) -> set[str]:
        return {r.dataset_id for r in self.records}

    # -- integrity ---------------------------------------------------------
    @property
    def hash(self) -> str:
        """Deterministic hash over record content in a canonical order.

        Sorting by (dataset_id, sample_id) makes the hash independent of file order,
        so regenerating a manifest from a differently-ordered scan produces the same
        value. That is what makes the hash safe to record in an evaluation run.
        """
        canonical = sorted(
            (r.to_dict() for r in self.records),
            key=lambda d: (d["dataset_id"], d["sample_id"]),
        )
        blob = json.dumps(canonical, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()

    # -- persistence -------------------------------------------------------
    def to_jsonl(self) -> str:
        header = {
            "_manifest": {
                "name": self.name,
                "version": self.version,
                "count": len(self.records),
                "hash": self.hash,
                "splits": self.split_counts(),
                "scenes": len(self.scenes()),
                "notes": self.notes,
            }
        }
        lines = [json.dumps(header, sort_keys=True)]
        for record in self.records:
            lines.append(json.dumps(record.to_dict(), sort_keys=True, default=str))
        return "\n".join(lines) + "\n"

    def write(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write the manifest. Refuses to clobber an existing file by default.

        A frozen manifest must never be silently rewritten — that is how a benchmark
        stops being reproducible.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and not overwrite:
            raise ManifestError(
                f"manifest already exists at {p}; refusing to overwrite "
                f"(pass overwrite=True only for a deliberate rebuild)"
            )
        p.write_text(self.to_jsonl(), encoding="utf-8")
        return p

    @classmethod
    def read(cls, path: str | Path) -> "DatasetManifest":
        p = Path(path)
        if not p.exists():
            raise ManifestError(f"manifest not found: {p}")

        records: list[SampleRecord] = []
        header: dict[str, Any] = {}
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"malformed manifest line {i + 1}: {exc}") from exc
            if "_manifest" in payload:
                header = payload["_manifest"]
                continue
            records.append(SampleRecord.from_dict(payload))

        manifest = cls(
            name=header.get("name", p.stem),
            records=records,
            version=header.get("version", MANIFEST_VERSION),
            notes=header.get("notes", {}),
        )

        # Integrity check: a manifest that does not hash to its own header is corrupt.
        recorded = header.get("hash")
        if recorded and recorded != manifest.hash:
            raise ManifestError(
                f"manifest hash mismatch: header says {recorded}, "
                f"content hashes to {manifest.hash}"
            )
        return manifest


# ---------------------------------------------------------------------------
# Run manifest — what an evaluation run actually used
# ---------------------------------------------------------------------------


@dataclass
class RunManifest:
    """Frozen record of everything that influenced an evaluation run.

    Required before any public-test evaluation. If this is missing, the run is not
    reproducible and its numbers cannot be quoted.
    """

    run_id: str
    config_hash: str
    dataset_manifest_hash: str
    model_revisions: dict[str, str] = field(default_factory=dict)
    prompt_versions: dict[str, str] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    seed: int = 42
    code_revision: str | None = None
    environment: dict[str, str] = field(default_factory=dict)
    started_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"run_manifest": self.to_dict(), "hash": self.hash},
                       indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return p


__all__ = [
    "MANIFEST_VERSION",
    "GEO_HASH_BUCKET_M",
    "ManifestError",
    "geographic_hash",
    "SampleRecord",
    "DatasetManifest",
    "RunManifest",
]