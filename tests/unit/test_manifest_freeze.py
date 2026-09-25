"""Tests for the frozen dataset-manifest record (`evaluation/manifest_freeze.json`).

WHY THIS EXISTS
---------------
Plan §72 Gate 2 requires "dataset manifests frozen" (plan:3806). Before this
record, the ONLY frozen manifest was the synthetic 100-record
`artifacts/gate1_manifest.jsonl`; the REAL manifests — the twenty Phase 12
selection manifests (`artifacts/phase12_selection/selection_manifest_seed00..19.jsonl`,
28,001 lines / 28,000 records each) — were not frozen anywhere. This record
freezes them, following the P1-11 precedent `evaluation/prompt_freeze.json`:
`evaluation/` holds the repo's manifest/hash machinery and is NOT gitignored,
whereas `artifacts/` is, so a record there would not be durable evidence.

REFRAMING — WHAT IS, AND IS NOT, PROVABLE
-----------------------------------------
The selection manifests carry a **richer header** than `DatasetManifest.to_jsonl()`
emits (it adds `field_mapping`, `imagery`, `data_dir`, ... to `notes`). That
header only survives a round trip because `read()` stores it in `notes` and
`to_jsonl()` re-emits it.

Direct measurement (2026-09-21) showed that `read(p) -> to_jsonl() -> write`
reproduces the source bytes **exactly** for all twenty manifests — but **only on
this Windows checkout**, where the sources are CRLF and `Path.write_text`
translates the LF that `to_jsonl()` emits back to CRLF. A POSIX writer emits LF
and would **not** reproduce the bytes. Byte-identity is therefore a
**platform-dependent observation, not a portable property**, and this record
does **NOT** assert or rely on byte-stability.

What IS asserted, and what is proven below, is the portable **hash-stability**:

    * `DatasetManifest.read(p).hash == the header's own recorded hash`
      (`read()` already enforces this as an integrity check, and we assert it
      again here so the property is pinned, not merely relied on), and
    * `read(p) -> to_jsonl() -> read() -> .hash` is identical to `read(p).hash`,
      independent of header shape and of newline convention (proven explicitly
      for LF vs CRLF).

Hashing is delegated to `evaluation.manifests.DatasetManifest.hash` — the
canonical, order-independent hash — and is never reimplemented here.

WHAT THE RECORD STORES
----------------------
Per manifest: `filename`, repo-relative `path`, `bytes`, `raw_sha256`,
`dataset_manifest_hash` (`DatasetManifest.hash`), `record_count`,
`split_counts`, `scene_count`. Plus a document-level `method`, `generated_at`
and this `reframing` note, and an outer `hash` that is the self-hash of the
payload.

`evaluation/manifest_freeze.json` is a read-only artifact. It introduces no
config key and does not change `Config.hash`.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evaluation.manifests import DatasetManifest, ManifestError

REPO_ROOT = Path(__file__).resolve().parents[2]
FREEZE_PATH = REPO_ROOT / "evaluation" / "manifest_freeze.json"
SELECTION_DIR = REPO_ROOT / "artifacts" / "phase12_selection"

#: The recorded hashing method, mirrored verbatim in the artifact.
METHOD = (
    "raw_sha256 = sha256 of the file bytes; dataset_manifest_hash = "
    "DatasetManifest.hash (sha256 over records sorted by (dataset_id, "
    "sample_id), order-independent); both recorded, never reimplemented"
)

#: The reframing note, mirrored verbatim in the artifact.
#:
#: MEASUREMENT NOTE (2026-09-21): the initial premise was that the manifests'
#: richer headers made a byte-exact re-serialisation impossible. Direct
#: measurement showed that `read(p) -> to_jsonl() -> write` DOES reproduce the
#: source bytes EXACTLY for all twenty manifests -- but ONLY on this Windows
#: checkout, where the sources are CRLF and `Path.write_text` translates the
#: LF that `to_jsonl()` emits back to CRLF. A POSIX writer emits LF and would
#: NOT reproduce the bytes. Byte-identity is therefore a PLATFORM-DEPENDENT
#: observation, not a portable property, and it is deliberately NOT asserted.
#: Hash-stability is the portable property, and it is what this record pins.
REFRAMING = (
    "The selection manifests carry a richer header than DatasetManifest.to_jsonl() "
    "emits (field_mapping, imagery, data_dir, ...), which only survives a round "
    "trip because read() stores them in `notes` and to_jsonl() re-emits them. A "
    "byte-exact re-serialisation is therefore NOT a portable property: it also "
    "depends on the writing platform's newline translation (the sources are CRLF; "
    "a POSIX writer emits LF, which would NOT match). Byte-stability is NOT "
    "asserted and NOT relied upon. The asserted, portable property is "
    "HASH-STABILITY: read(p).hash == header.hash, and "
    "read(p) -> to_jsonl() -> read() -> .hash is identical -- independent of "
    "header shape and of newline convention."
)

#: Outer self-hash length, mirroring `RunManifest.hash` (evaluation/manifests.py).
SELF_HASH_CHARS = 16


# ---------------------------------------------------------------------------
# Discovery + per-manifest record — the single source of truth
# ---------------------------------------------------------------------------


def _discovered_manifest_paths() -> list[Path]:
    """Every Phase 12 selection manifest, by DISCOVERY (never a hardcoded list)."""
    return sorted(SELECTION_DIR.glob("selection_manifest_seed*.jsonl"))


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_record_count(path: Path) -> int:
    """Non-blank lines minus the single `_manifest` header line."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return len(lines) - 1  # exactly one header line


def _entry(path: Path) -> dict:
    """The frozen facts for one manifest. Hashing is delegated, not redone."""
    manifest = DatasetManifest.read(path)
    return {
        "filename": path.name,
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "raw_sha256": _raw_sha256(path),
        "dataset_manifest_hash": manifest.hash,
        "record_count": len(manifest),
        "split_counts": manifest.split_counts(),
        "scene_count": len(manifest.scenes()),
    }


def _live_manifest_entries() -> list[dict]:
    return [_entry(p) for p in _discovered_manifest_paths()]


def _self_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:SELF_HASH_CHARS]


def build_freeze_document(generated_at: str | None = None) -> dict:
    """The freeze record computed from the LIVE tree. Pure: reads, writes nothing.

    `generated_at` is injectable so the committed document can be compared
    exactly (the timestamp is not a hashed fact, but passing it in makes the
    comparison a strict equality).
    """
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "method": METHOD,
        "generated_at": generated_at,
        "reframing": REFRAMING,
        "manifests": _live_manifest_entries(),
    }
    return {"hash": _self_hash(payload), "manifest_freeze": payload}


def _committed() -> dict:
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Cheap structural checks
# ---------------------------------------------------------------------------


def test_freeze_record_exists_and_is_committed() -> None:
    assert FREEZE_PATH.exists(), f"missing committed freeze record: {FREEZE_PATH}"


def test_recorded_manifest_set_equals_live_discovered_set() -> None:
    recorded = {entry["path"] for entry in _committed()["manifest_freeze"]["manifests"]}
    live = {p.relative_to(REPO_ROOT).as_posix() for p in _discovered_manifest_paths()}
    assert recorded == live, (
        "a selection manifest was added or removed without re-freezing: "
        f"recorded={sorted(recorded)} live={sorted(live)}"
    )


def test_method_and_reframing_are_recorded() -> None:
    payload = _committed()["manifest_freeze"]
    assert payload["method"] == METHOD
    assert payload["reframing"] == REFRAMING
    # The portable claim is hash-stability; byte-stability is explicitly NOT asserted.
    assert "HASH-STABILITY" in payload["reframing"]
    assert "Byte-stability is NOT" in payload["reframing"]


def test_outer_hash_is_the_self_hash_of_the_payload() -> None:
    document = _committed()
    assert document["hash"] == _self_hash(document["manifest_freeze"])


def test_each_recorded_raw_sha256_and_size_match_disk() -> None:
    """Cheap and strong: pins the exact bytes of all twenty manifests."""
    for entry in _committed()["manifest_freeze"]["manifests"]:
        path = REPO_ROOT / entry["path"]
        assert path.exists(), entry["path"]
        assert path.stat().st_size == entry["bytes"], entry["path"]
        assert _raw_sha256(path) == entry["raw_sha256"], entry["path"]


def test_each_recorded_record_count_matches_the_raw_line_count() -> None:
    for entry in _committed()["manifest_freeze"]["manifests"]:
        path = REPO_ROOT / entry["path"]
        assert _raw_record_count(path) == entry["record_count"], entry["path"]
        assert entry["record_count"] == 28000, entry["path"]


# ---------------------------------------------------------------------------
# The load-bearing test: the committed record equals the live tree
# ---------------------------------------------------------------------------


def test_committed_document_equals_the_builder() -> None:
    """The strongest single assertion: re-derive everything and compare.

    This reads all twenty manifests through `DatasetManifest.read` and recomputes
    `DatasetManifest.hash`, so it proves `dataset_manifest_hash`, `split_counts`
    and `scene_count` for every manifest against the live files — not merely the
    cheap raw-byte facts above. It then re-reads each manifest's OWN header and
    asserts `read(p).hash == header.hash`, pinning the integrity property that
    `read()` enforces internally.
    """
    committed = _committed()
    rebuilt = build_freeze_document(
        generated_at=committed["manifest_freeze"]["generated_at"]
    )
    assert committed == rebuilt

    for entry in committed["manifest_freeze"]["manifests"]:
        path = REPO_ROOT / entry["path"]
        header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])["_manifest"]
        assert header["hash"] == entry["dataset_manifest_hash"], entry["path"]


def test_hash_is_stable_across_a_reserialisation_round_trip(tmp_path) -> None:
    """read -> to_jsonl -> read must reproduce the SAME hash (seed00).

    This is the hash-stability claim, and ONLY that. Byte-identity is
    deliberately NOT asserted: it happens to hold on this Windows checkout (CRLF
    sources + text-mode newline translation) but is platform-dependent and would
    fail under a POSIX writer. The hash, which is over canonical record content,
    is the portable invariant.
    """
    entry = _committed()["manifest_freeze"]["manifests"][0]
    source = REPO_ROOT / entry["path"]

    first = DatasetManifest.read(source)
    reserialised = tmp_path / "round_trip.jsonl"
    reserialised.write_text(first.to_jsonl(), encoding="utf-8")

    second = DatasetManifest.read(reserialised)

    assert first.hash == second.hash == entry["dataset_manifest_hash"]


def test_hash_is_invariant_to_newline_convention(tmp_path) -> None:
    """The hash does not depend on LF vs CRLF — the property that IS portable.

    `to_jsonl()` emits LF; the frozen sources are CRLF. Writing the same record
    set with either convention and reading it back must give the SAME hash. This
    is why hash-stability is asserted and byte-stability is not.
    """
    entry = _committed()["manifest_freeze"]["manifests"][0]
    source = REPO_ROOT / entry["path"]
    manifest = DatasetManifest.read(source)

    lf_text = manifest.to_jsonl()
    crlf_text = lf_text.replace("\n", "\r\n")

    lf_file = tmp_path / "lf.jsonl"
    crlf_file = tmp_path / "crlf.jsonl"
    lf_file.write_bytes(lf_text.encode("utf-8"))
    crlf_file.write_bytes(crlf_text.encode("utf-8"))

    assert lf_file.read_bytes() != crlf_file.read_bytes()  # conventions differ
    assert DatasetManifest.read(lf_file).hash == manifest.hash
    assert DatasetManifest.read(crlf_file).hash == manifest.hash
    assert manifest.hash == entry["dataset_manifest_hash"]


# ---------------------------------------------------------------------------
# Negative control — the integrity guard must bite
# ---------------------------------------------------------------------------


def test_a_flipped_byte_in_a_record_raises_a_hash_mismatch(tmp_path) -> None:
    """Tampering with one record must make `read()` refuse the manifest.

    A scratch copy of seed00 has one character of one record changed (still
    valid JSON, so the failure is the HASH check and not a parse error), and
    `read()` must raise the hash-mismatch `ManifestError`. Nothing is written
    back to the real manifest.
    """
    source = _discovered_manifest_paths()[0]
    lines = source.read_text(encoding="utf-8").splitlines()

    # Tamper the FIRST record (index 1; index 0 is the header).
    record = json.loads(lines[1])
    original = record["sample_id"]
    record["sample_id"] = original + "_tampered"
    lines[1] = json.dumps(record, sort_keys=True)

    scratch = tmp_path / "tampered.jsonl"
    scratch.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ManifestError) as excinfo:
        DatasetManifest.read(scratch)
    assert "hash mismatch" in str(excinfo.value)

    # And the real manifest is untouched.
    assert _raw_sha256(source) == _committed()["manifest_freeze"]["manifests"][0]["raw_sha256"]
