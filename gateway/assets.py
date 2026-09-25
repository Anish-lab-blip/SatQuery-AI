"""SatQuery AI — the ephemeral asset store (the `POST /v1/assets` backend).

WHY THIS EXISTS
---------------
`docs/API_CONTRACT.md` section 2.5 recorded the gap honestly: the plan fixes the
surface at three endpoints, yet `AnalysisRequest.assets` is `list[str]` --
asset *handles*, not bytes. Those two facts cannot both hold without a fourth
endpoint, and the contract recorded it as "NOT IN THE PLAN", an open decision,
with Option A (out-of-band upload) recommended over Option B (inline multipart).

**The owner ruling of 2026-09-22 resolved it: BUILD, Option A, with opaque
ephemeral handles.** This module is that endpoint's storage layer.

THE DESIGN, AND THE THREE OBLIGATIONS THE CONTRACT NAMED
--------------------------------------------------------
`API_CONTRACT.md` section 2.5 says that whichever option is chosen, the frontend
needs *"a per-file size limit, a content-type allowlist, and an idempotency story
for retries"*. Each is answered here, and each answer is a decision worth naming:

1. **Per-file size limit.** `GatewayConfig.max_file_bytes` (default 4 MiB),
   enforced *while reading*, not after: the check is against bytes accumulated so
   far, so a client cannot make the process buffer an unbounded body by lying
   about `Content-Length`. The gateway checks `Content-Length` before reading
   (`gateway/policy.py` step 3 of the admit ladder) and this is the second line --
   the body is already buffered by then, so the gateway's check is the one that
   saves memory and this one is the one that saves the disk.

2. **Content-type allowlist.** `GatewayConfig.allowed_content_types`. An absent
   or unrecognised type is refused rather than defaulted, because guessing an
   image format from a filename extension is how a `.tif` that is really a PDF
   reaches a raster reader.

3. **Idempotency.** There is **no** idempotency key, and that is a deliberate
   answer rather than an omission. A retried upload is a *new* handle, not the
   same one, because the two requests are indistinguishable at the server: no
   plan-specified key exists, and inventing one would be inventing contract. The
   cost is a leaked handle, which :attr:`AssetStore.capacity` bounds and
   :meth:`AssetStore.sweep` reclaims. The contract must record this.

WHY THE HANDLE IS OPAQUE, AND WHY THAT IS THE WHOLE POINT
---------------------------------------------------------
The handle is `asset_<32 hex chars>` from `secrets.token_hex(16)` -- 128 bits,
not derivable from the filename, the size, or the content. Two consequences:

  * **A client cannot enumerate the store.** `asset_0`, `asset_1` or a hash of
    the content would let a caller who guessed one handle reach another user's
    upload. There is no auth in v1 (`API_CONTRACT.md` section 7), so the handle
    IS the capability: unguessable is not a nicety, it is the only access
    control the endpoint has.
  * **A handle does not leak a server path.** The stored filename is the handle
    plus a sanitised suffix taken from the *content type*, never from the
    client-supplied name -- so a name like `../../etc/passwd` cannot become a
    path. The original name is not stored at all, because it is not needed and
    storing it would be storing attacker-controlled text for no reason.

WHY EPHEMERAL, AND WHAT "EPHEMERAL" IS BOUND TO
-----------------------------------------------
`configs/deploy.yaml` sets `cache_max_models: 1`: at most one model is resident,
and analyses are serialized. An upload is therefore consumed within one analysis,
immediately or never. So the TTL is short (default 15 minutes) and the capacity
is small (default 32 files) -- sizing either for "a library of uploads" would
reserve disk on a Space that has none to spare.

**Expiry is checked on read, not by a background thread.** A timer would be a
second source of truth about liveness, and a Space that sleeps between requests
cannot be relied on to run it. `get()` compares the recorded deadline against
`time.monotonic()`, so a lapsed handle is refused even if nothing ever swept it.
`sweep()` is then a *housekeeping* call -- it reclaims disk, it does not define
correctness.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not decode, validate, or inspect the image. A handle means "these bytes
were accepted under this content type at this time"; whether they are a readable
GeoTIFF is the raster reader's question, and answering it twice would be a second
place to disagree (`core/planner.py::_indices_for`'s warning, applied here).
"""

from __future__ import annotations

import os
import secrets
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "AssetHandle",
    "AssetStore",
    "AssetStoreFullError",
    "AssetTooLargeError",
    "UnknownAssetError",
    "UnsupportedContentTypeError",
    "read_body_bounded",
]


async def read_body_bounded(request: Any, limit: int) -> tuple[bytes, bool]:
    """Read a request body, refusing it the moment it exceeds `limit`.

    Returns `(body, False)` on success and `(b"", True)` when the cap was
    exceeded.

    WHY THIS LIVES HERE, IN THE SHARED STORE MODULE (F-9)
    ----------------------------------------------------
    This function was `_read_body_bounded` in `gateway/app.py`, added by the F-6
    fix for the gateway's own `await request.body()`. The **Space had the same
    line** (`app/space_app.py`, `POST /v1/assets`) and therefore the same defect
    -- measured 2026-09-22: with the cap at 1 MiB, a **64 MiB** body produced a
    peak allocation of **128 MiB**, and a 16 MiB body 32 MiB, tracking body size
    **linearly with no ceiling**. It answered `413` eventually, but only after
    buffering everything. The gateway's fix had protected one caller of two.

    Fixing the Space by copying the helper would have created two copies of a
    security control -- and two copies drift, which is the F-7 finding restated.
    So the function moved down to the module **both layers already import**, and
    both now call this one implementation. `gateway/assets.py` is dependency-free
    (no fastapi, no starlette, no httpx, no torch), which is why it can serve the
    Space's requirement 1: importing this module must not pull the web stack.

    WHY NOT `await request.body()`
    ------------------------------
    `request.body()` accumulates the WHOLE body before returning, so a cap
    checked afterwards is not a cap -- it is a post-mortem. Starlette exposes a
    streaming `request.stream()`, and this reads from it and stops at the limit,
    so memory is bounded by `limit + one chunk` rather than by whatever the
    client chose to send.

    The `Content-Length` header is deliberately NOT consulted here, not even as a
    fast path. The F-6 measurement is the reason: a declared length drew `413`
    with 0.2 MiB peak, but an **omitted** one drew `502` with a 13.9 MiB peak, so
    anything keyed on that header holds only for clients that tell the truth.
    A client that lies low is caught by the accumulation check; a client that
    omits the header is measured like any other.
    """
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            # Stop reading immediately. Doing so lets the server close the
            # connection without the client delivering the rest of the body,
            # which is the point of a streaming cap.
            return b"", True
        chunks.append(chunk)
    return b"".join(chunks), False


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class AssetStoreError(Exception):
    """Base class, so a caller can catch the family without enumerating it.

    Deliberately NOT a `core.errors.SatQueryError`: `core/errors.py` is the
    taxonomy the contract publishes (`API_CONTRACT.md` section 5.2, 23 codes),
    and none of those codes means "this handle is unknown". Adding one would move
    the taxonomy, which the contract forbids the gateway from doing
    (`docs/DEPLOYMENT_ARCHITECTURE.md` section 2.3). The route therefore maps
    these onto *existing* codes, and says which -- see `gateway/app.py`.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class AssetTooLargeError(AssetStoreError):
    """The file exceeds `max_file_bytes`."""


class UnsupportedContentTypeError(AssetStoreError):
    """The declared content type is absent or not on the allowlist."""


class AssetStoreFullError(AssetStoreError):
    """The store is at capacity and every slot is live."""


class UnknownAssetError(AssetStoreError):
    """No live handle matches. Covers *expired* as well as *never issued*.

    The two are one error on purpose: a client cannot act on the difference
    (both mean "upload again"), and distinguishing them would report whether a
    handle had ever existed -- a small information leak about other clients'
    uploads, which matters precisely because there is no auth.
    """


# ---------------------------------------------------------------------------
# Handle
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AssetHandle:
    """One accepted upload.

    Attributes:
        asset_id: the opaque handle. The client's only reference.
        path: where the bytes live. Server-side; never serialised to a client.
        content_type: the accepted type, after normalisation.
        bytes: size on disk, measured at write time.
        created_at: `time.monotonic()` at acceptance.
        expires_at: monotonic deadline. Monotonic rather than wall-clock so a
            system clock adjustment cannot extend or truncate a TTL.
        expires_at_iso: the wall-clock rendering the contract returns. Derived
            once, at acceptance, so it is stable across calls.
    """

    asset_id: str
    path: Path
    content_type: str
    bytes: int
    created_at: float
    expires_at: float
    expires_at_iso: str

    def to_response(self) -> dict[str, Any]:
        """The `POST /v1/assets` response body.

        `path` is deliberately absent. Returning it would hand a client a
        server-side filesystem location -- an information disclosure, and an
        invitation to construct a path directly instead of via a handle.
        """
        return {
            "asset_id": self.asset_id,
            "content_type": self.content_type,
            "bytes": self.bytes,
            "expires_at": self.expires_at_iso,
        }


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
@dataclass
class AssetStore:
    """A bounded, TTL'd, opaque-handle store of uploaded bytes.

    Not thread-safe, and that is a stated property rather than an oversight:
    `docs/ARCHITECTURE_FREEZE.md` section 5 puts v1 execution in a single
    process with sequential plans, and `cache_max_models: 1` means requests are
    serialized anyway (a lock would be complexity without a represented hazard --
    the same reasoning `SpecialistRegistry` records for its own lack of one).

    Args:
        root: directory to hold the bytes. Created if absent.
        max_file_bytes: per-file cap. Enforced during the read.
        max_files: capacity. A new upload when full sweeps expired handles
            first, then refuses -- it does not evict a LIVE handle, because
            evicting one would invalidate a handle a client may be about to use,
            turning a working flow into an intermittent failure.
        ttl_seconds: handle lifetime.
        allowed_content_types: the allowlist.
    """

    root: Path
    max_file_bytes: int
    max_files: int = 32
    ttl_seconds: float = 900.0
    allowed_content_types: tuple[str, ...] = ()

    _handles: dict[str, AssetHandle] = field(default_factory=dict, init=False)
    # F-11 (owner ruling 2026-09-23): `_sweeps` and `_refusals` were counters
    # incremented on the request path whose only reader was `stats()`, and
    # `stats()` had no production caller -- the document told operators to use
    # an instrument nothing held. A measurement nobody reads is a cost paid on
    # every request for nothing, so the counters and `stats()` are REMOVED
    # rather than left wired "for later". `sweep()` itself stays: it reclaims
    # disk and is a control, not a diagnostic.

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be positive")
        if self.max_files <= 0:
            raise ValueError("max_files must be positive")
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")

    # -- writing -----------------------------------------------------------

    def put(
        self,
        data: bytes,
        *,
        content_type: str | None,
        now: float | None = None,
    ) -> AssetHandle:
        """Accept bytes and return a handle.

        Args:
            data: the file's bytes, already read by the caller. Reading is the
                caller's job because the size cap must be enforced while reading
                (see the module docstring) and only the caller knows whether the
                body is a stream or a buffer.
            content_type: the client-declared type, or None.
            now: monotonic timestamp, injectable so a test can advance time
                without sleeping.

        Raises:
            AssetTooLargeError: `len(data)` exceeds the cap.
            UnsupportedContentTypeError: the type is absent or not allowed.
            AssetStoreFullError: capacity is reached and all slots are live.
        """
        if len(data) > self.max_file_bytes:
            raise AssetTooLargeError(
                f"{len(data)} bytes exceeds the {self.max_file_bytes} byte "
                f"per-file limit"
            )
        if not data:
            # A zero-byte upload cannot be a raster. Refused here rather than
            # downstream so the failure names the upload, not the reader.
            raise AssetTooLargeError("the uploaded file is empty")

        normalised = _normalise_content_type(content_type)
        allowed = tuple(t.lower() for t in self.allowed_content_types)
        if allowed and normalised not in allowed:
            raise UnsupportedContentTypeError(
                f"content-type {content_type!r} is not accepted; allowed: "
                f"{', '.join(allowed)}"
            )

        stamp = time.monotonic() if now is None else now
        self.sweep(now=stamp)
        if len(self._handles) >= self.max_files:
            raise AssetStoreFullError(
                f"the store holds {len(self._handles)} live handles, at its "
                f"limit of {self.max_files}; retry after a handle expires or "
                f"reduce the upload rate"
            )

        asset_id = _new_handle()
        suffix = _suffix_for(normalised)
        path = self.root / f"{asset_id}{suffix}"
        # Write via a temporary file and `os.replace`, so a reader can never
        # observe a partially written upload. `os.replace` is atomic on both
        # POSIX and Windows when source and destination share a volume.
        fd, tmp_name = tempfile.mkstemp(dir=str(self.root), suffix=".part")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp_name, path)
        except BaseException:
            # Leave nothing behind on failure, and do not mask the original
            # exception if the cleanup itself fails.
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

        expires_at = stamp + self.ttl_seconds
        record = AssetHandle(
            asset_id=asset_id,
            path=path,
            content_type=normalised,
            bytes=len(data),
            created_at=stamp,
            expires_at=expires_at,
            expires_at_iso=_iso_for(expires_at - stamp),
        )
        self._handles[asset_id] = record
        return record

    # -- reading -----------------------------------------------------------

    def get(self, asset_id: str, *, now: float | None = None) -> AssetHandle:
        """Resolve a handle, or raise `UnknownAssetError`.

        Expiry is evaluated here rather than trusted to a sweeper: a handle past
        its deadline is unknown even if nothing has run `sweep()`. That is what
        makes the TTL a guarantee instead of a housekeeping hope.
        """
        stamp = time.monotonic() if now is None else now
        record = self._handles.get(asset_id)
        if record is None:
            raise UnknownAssetError(f"unknown or expired asset handle {asset_id!r}")
        if stamp >= record.expires_at:
            self._forget(asset_id)
            raise UnknownAssetError(f"unknown or expired asset handle {asset_id!r}")
        if not record.path.exists():
            # The record survived but the bytes did not: an operator cleared the
            # directory, or a container restarted with a fresh volume. Reported
            # as unknown -- the handle is not usable, which is what the client
            # needs to know -- and the stale record is dropped so the slot is
            # reclaimed rather than held by a phantom.
            self._forget(asset_id)
            raise UnknownAssetError(
                f"asset handle {asset_id!r} resolves to a file that is no longer "
                f"present"
            )
        return record

    def resolve_many(
        self, asset_ids: list[str], *, now: float | None = None
    ) -> tuple[AssetHandle, ...]:
        """Resolve every handle, raising on the first failure.

        All-or-nothing: a partial resolution would let an analysis start with
        one of a required pair missing, which the specialists would then reject
        with a pairing error that names the wrong cause. Failing here names the
        real one.
        """
        return tuple(self.get(asset_id, now=now) for asset_id in asset_ids)

    # -- lifetime ----------------------------------------------------------

    def sweep(self, *, now: float | None = None) -> int:
        """Drop expired handles and delete their bytes. Returns the count.

        Disk is reclaimed, not just the record: a store that forgot handles but
        kept their files would leak disk monotonically, and a Space's disk is
        small enough that this would be the first thing to fail.
        """
        stamp = time.monotonic() if now is None else now
        expired = [
            asset_id
            for asset_id, record in self._handles.items()
            if stamp >= record.expires_at
        ]
        for asset_id in expired:
            self._forget(asset_id)
        return len(expired)

    def _forget(self, asset_id: str) -> None:
        record = self._handles.pop(asset_id, None)
        if record is None:
            return
        try:
            record.path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            # A file that cannot be deleted must not break a metadata read. The
            # record is already gone, so the handle is refused from now on,
            # which is the property that matters.
            pass

    def clear(self) -> None:
        """Drop every handle and delete every file. For tests and shutdown."""
        for asset_id in tuple(self._handles):
            self._forget(asset_id)

    # -- observability -----------------------------------------------------
    #
    # F-11 (owner ruling 2026-09-23): `stats()` lived here. It reported
    # `live`/`capacity`/`ttl_seconds`/`max_file_bytes`/`sweeps`/
    # `capacity_refusals` for an operator, but **no production module ever
    # called it** -- `docs/DEPLOYMENT_ARCHITECTURE.md` section 5 pointed an
    # operator at an instrument the deployment did not expose. It has been
    # removed together with the two counters that fed it, so the surface and
    # the document can no longer disagree. Anything that needs live counts can
    # read `len(store)`; capacity and TTL are construction-time constants on
    # the dataclass.

    def __len__(self) -> int:
        return len(self._handles)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _new_handle() -> str:
    """An opaque, unguessable handle.

    `secrets`, not `random`: the handle is this endpoint's only access control
    (`API_CONTRACT.md` section 7 -- there is no auth in v1). 128 bits from
    `token_hex(16)` is not brute-forceable, and the prefix keeps a handle
    recognisable in a log without making it derivable.
    """
    return f"asset_{secrets.token_hex(16)}"


#: Content type -> stored suffix. Derived from the TYPE, never from the
#: client-supplied filename, so a hostile name cannot influence a path.
_SUFFIXES: Mapping[str, str] = {
    "image/tiff": ".tif",
    "image/geotiff": ".tif",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "application/octet-stream": ".bin",
}


def _normalise_content_type(content_type: str | None) -> str:
    """Lower-case the type and strip parameters, or '' when absent.

    `image/tiff; charset=binary` is a legitimate header and the parameter is not
    part of the type. `None` becomes `''` so the allowlist check refuses it --
    defaulting an absent type to `application/octet-stream` would make the
    allowlist unenforceable for exactly the clients that omit the header.
    """
    if not content_type:
        return ""
    return content_type.split(";", 1)[0].strip().lower()


def _suffix_for(content_type: str) -> str:
    return _SUFFIXES.get(content_type, ".bin")


def _iso_for(ttl_seconds: float) -> str:
    """Render the wall-clock expiry, `Z`-suffixed per the contract's convention.

    Computed from a monotonic TTL converted to wall-clock *now*, so the string
    the client sees matches the deadline the store enforces, without the store
    ever depending on the wall clock for correctness.
    """
    from datetime import datetime, timedelta, timezone

    moment = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")
