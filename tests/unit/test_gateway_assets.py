"""The ephemeral asset store: opaque handles, three obligations, real lifetimes.

WHY THIS FILE EXISTS
--------------------
`API_CONTRACT.md` section 2.5 recorded `POST /v1/assets` as "NOT IN THE PLAN"
with two options. The owner ruling of 2026-09-22 chose **Option A** -- out-of-band
upload with opaque ephemeral handles -- and that section names the three things
the frontend would need, which are therefore the three things tested here:

    "a per-file size limit, a content-type allowlist, and an idempotency story
     for retries"

WHAT IS ASSERTED, AND WHY EACH MATTERS
--------------------------------------
  * The handle is **opaque and unguessable**. There is no auth in v1
    (`API_CONTRACT.md` section 7), so the handle IS the access control. A
    sequential or content-derived handle would let one client reach another's
    upload. Tested by inspection (no filename, no index, no content in it) and by
    minting many and asserting uniqueness.
  * A handle does **not** leak a path. `AssetHandle.to_response()` must not
    contain the stored location -- that would disclose a server path and invite a
    client to bypass the handle entirely.
  * The **size cap fires on the bytes**, not on a declared length.
  * The **content-type allowlist refuses an absent type** rather than defaulting,
    because defaulting is how a `.tif` that is really a PDF reaches a reader.
  * **Expiry is enforced on read**, not only by a sweeper, so a lapsed handle is
    refused even if nothing swept it.
  * **A live handle is never evicted** to make room. Evicting one would
    invalidate a handle a client is about to use.
  * The **stored name derives from the content type**, never from the client's
    filename, so `../../etc/passwd` cannot become a path.

Time is injected (`now=`) rather than slept through, so these tests assert real
TTL behaviour without waiting.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re

import pytest

from gateway.assets import (
    AssetStore,
    AssetStoreFullError,
    AssetTooLargeError,
    UnknownAssetError,
    UnsupportedContentTypeError,
)

pytestmark = pytest.mark.unit

ALLOWED = ("image/tiff", "image/geotiff", "image/png", "image/jpeg")


@pytest.fixture()
def store(tmp_path_factory) -> AssetStore:
    """A store rooted outside the repo.

    `tmp_path_factory` rather than `tmp_path`: the repo's convention
    (`MEMORY.md`) is that new tests must not use pytest's `tmp_path` cleanup
    machinery in this sandbox, which deletes directories the shim blocks. The
    factory's basetemp is passed as a fresh path by the invocation, so it is
    created once and left in place.
    """
    root = tmp_path_factory.mktemp("assets")
    return AssetStore(
        root=root,
        max_file_bytes=1024,
        max_files=4,
        ttl_seconds=60.0,
        allowed_content_types=ALLOWED,
    )


# ---------------------------------------------------------------------------
# The handle: opacity and non-disclosure
# ---------------------------------------------------------------------------


class TestHandleOpacity:
    """The handle is the only access control, so it must be unguessable."""

    def test_handles_are_unique_across_many_uploads(self, tmp_path_factory) -> None:
        """Uniqueness is a property of the generator, not of capacity.

        Deliberately built on a *wide* store rather than the shared fixture: the
        fixture caps capacity at 4 so the capacity tests can fill it, and a
        50-upload loop against it would (correctly) be refused at the fifth.
        Asserting uniqueness therefore needs a store whose limit does not
        interfere -- otherwise this test would pass or fail for a reason that
        has nothing to do with the handle.
        """
        wide = AssetStore(
            root=tmp_path_factory.mktemp("wide"),
            max_file_bytes=1024,
            max_files=64,
            ttl_seconds=60.0,
            allowed_content_types=ALLOWED,
        )
        handles = {
            wide.put(b"x", content_type="image/png").asset_id for _ in range(50)
        }
        assert len(handles) == 50, "handles collided; two clients could collide"

    def test_unique_handles_survive_sequential_retries(
        self, tmp_path_factory
    ) -> None:
        """A retry is a *new* upload, so it must mint a new handle.

        The idempotency story for `POST /v1/assets` is "a retry mints a new
        handle and the previous one lapses" (there is no request key in the
        contract). This asserts the minting half of that story: two identical
        payloads do not collapse onto one handle, which is what would happen if
        the handle were a content hash.
        """
        wide = AssetStore(
            root=tmp_path_factory.mktemp("retry"),
            max_file_bytes=1024,
            max_files=8,
            ttl_seconds=60.0,
            allowed_content_types=ALLOWED,
        )
        first = wide.put(b"identical", content_type="image/png")
        second = wide.put(b"identical", content_type="image/png")
        assert first.asset_id != second.asset_id
        assert first.path != second.path

    def test_a_handle_does_not_contain_the_filename(self, store: AssetStore) -> None:
        """A content-derived or name-derived handle would be guessable."""
        handle = store.put(b"payload", content_type="image/png")
        assert "png" not in handle.asset_id.replace("image", ""), (
            "the handle embeds the content type; a handle must carry no "
            "information about the upload"
        )
        assert "/" not in handle.asset_id and "\\" not in handle.asset_id

    def test_a_handle_is_long_enough_to_be_unguessable(self, store: AssetStore) -> None:
        """128 bits from `secrets.token_hex(16)`. A short handle is attackable."""
        handle = store.put(b"x", content_type="image/png").asset_id
        hex_part = handle.removeprefix("asset_")
        assert len(hex_part) == 32, (
            f"the handle carries {len(hex_part) * 4} bits of entropy; this "
            f"endpoint has no auth, so the handle must be unguessable"
        )

    def test_the_response_does_not_disclose_a_server_path(
        self, store: AssetStore
    ) -> None:
        """The response is what a client receives. A path in it is a leak."""
        handle = store.put(b"x", content_type="image/png")
        response = handle.to_response()
        assert set(response) == {"asset_id", "content_type", "bytes", "expires_at"}, (
            f"the upload response has unexpected fields {sorted(response)}; a "
            f"server-side path must never be among them"
        )
        body = repr(response)
        assert str(store.root) not in body, "the response discloses the store root"
        assert str(handle.path) not in body, "the response discloses the file path"

    def test_the_stored_name_ignores_a_hostile_content_type(
        self, store: AssetStore
    ) -> None:
        """A path cannot be built from a type that is not on the allowlist --
        the request is refused before any name is derived."""
        with pytest.raises(UnsupportedContentTypeError):
            store.put(b"x", content_type="../../etc/passwd")
        # And nothing was written into the root.
        assert list(store.root.iterdir()) == []

    def test_a_stored_file_stays_inside_the_root(self, store: AssetStore) -> None:
        handle = store.put(b"x", content_type="image/png")
        assert handle.path.parent == store.root
        assert handle.path.name.startswith(handle.asset_id)


# ---------------------------------------------------------------------------
# Obligation 1: the per-file size limit
# ---------------------------------------------------------------------------


class TestSizeLimit:
    """Enforced on the BYTES, not on a declared length."""

    def test_a_file_at_the_limit_is_accepted(self, store: AssetStore) -> None:
        handle = store.put(b"a" * 1024, content_type="image/png")
        assert handle.bytes == 1024

    def test_a_file_over_the_limit_is_refused(self, store: AssetStore) -> None:
        with pytest.raises(AssetTooLargeError) as exc:
            store.put(b"a" * 1025, content_type="image/png")
        assert "1024" in str(exc.value)

    def test_an_empty_file_is_refused(self, store: AssetStore) -> None:
        """Zero bytes cannot be a raster; refusing names the upload, not the
        reader."""
        with pytest.raises(AssetTooLargeError):
            store.put(b"", content_type="image/png")

    def test_a_refused_upload_writes_nothing(self, store: AssetStore) -> None:
        """A partial write left behind would consume a slot and disk."""
        with pytest.raises(AssetTooLargeError):
            store.put(b"a" * 2048, content_type="image/png")
        assert len(store) == 0
        assert list(store.root.iterdir()) == []


# ---------------------------------------------------------------------------
# Obligation 2: the content-type allowlist
# ---------------------------------------------------------------------------


class TestContentTypeAllowlist:
    def test_an_allowed_type_is_accepted(self, store: AssetStore) -> None:
        for content_type in ALLOWED:
            handle = store.put(b"x", content_type=content_type)
            assert handle.content_type == content_type

    def test_an_absent_content_type_is_refused_not_defaulted(
        self, store: AssetStore
    ) -> None:
        """Defaulting to `application/octet-stream` would make the allowlist
        unenforceable for exactly the clients that omit the header."""
        with pytest.raises(UnsupportedContentTypeError):
            store.put(b"x", content_type=None)
        with pytest.raises(UnsupportedContentTypeError):
            store.put(b"x", content_type="")

    def test_a_disallowed_type_is_refused(self, store: AssetStore) -> None:
        for content_type in ("text/plain", "application/pdf", "image/webp"):
            with pytest.raises(UnsupportedContentTypeError):
                store.put(b"x", content_type=content_type)

    def test_a_type_parameter_is_stripped_before_the_check(
        self, store: AssetStore
    ) -> None:
        """`image/png; charset=binary` is a legitimate header."""
        handle = store.put(b"x", content_type="image/png; charset=binary")
        assert handle.content_type == "image/png"

    def test_the_type_is_case_insensitive(self, store: AssetStore) -> None:
        handle = store.put(b"x", content_type="IMAGE/PNG")
        assert handle.content_type == "image/png"

    def test_an_empty_allowlist_accepts_any_type(self, tmp_path_factory) -> None:
        """An operator who configures no allowlist gets no restriction -- and
        that is a decision the store makes explicitly, not by accident."""
        store = AssetStore(
            root=tmp_path_factory.mktemp("open"),
            max_file_bytes=64,
            allowed_content_types=(),
        )
        assert store.put(b"x", content_type="text/plain").content_type == "text/plain"


# ---------------------------------------------------------------------------
# Lifetime: expiry is enforced on read
# ---------------------------------------------------------------------------


class TestExpiry:
    def test_a_fresh_handle_resolves(self, store: AssetStore) -> None:
        handle = store.put(b"x", content_type="image/png", now=0.0)
        assert store.get(handle.asset_id, now=1.0).asset_id == handle.asset_id

    def test_an_expired_handle_is_refused(self, store: AssetStore) -> None:
        handle = store.put(b"x", content_type="image/png", now=0.0)
        with pytest.raises(UnknownAssetError):
            store.get(handle.asset_id, now=61.0)

    def test_expiry_is_enforced_without_a_sweep(self, store: AssetStore) -> None:
        """The guarantee, not a housekeeping hope.

        A Space sleeps between requests, so a background sweeper cannot be relied
        on. `get()` compares the deadline itself.
        """
        handle = store.put(b"x", content_type="image/png", now=0.0)
        # No sweep() call between the two.
        with pytest.raises(UnknownAssetError):
            store.get(handle.asset_id, now=1_000.0)

    def test_the_boundary_is_inclusive_of_the_deadline(self, store: AssetStore) -> None:
        """At exactly `expires_at` the handle is gone, not alive."""
        handle = store.put(b"x", content_type="image/png", now=0.0)
        with pytest.raises(UnknownAssetError):
            store.get(handle.asset_id, now=60.0)

    def test_sweeping_deletes_the_bytes_too(self, store: AssetStore) -> None:
        """Forgetting the record but keeping the file leaks disk monotonically."""
        handle = store.put(b"x", content_type="image/png", now=0.0)
        assert handle.path.exists()
        removed = store.sweep(now=61.0)
        assert removed == 1
        assert not handle.path.exists()
        assert len(store) == 0

    def test_sweeping_spares_live_handles(self, store: AssetStore) -> None:
        live = store.put(b"x", content_type="image/png", now=0.0)
        store.put(b"y", content_type="image/png", now=0.0)
        assert store.sweep(now=30.0) == 0
        assert live.path.exists()

    def test_a_handle_whose_file_vanished_is_refused(self, store: AssetStore) -> None:
        """An operator cleared the directory or a container restarted.

        The handle is unusable, so it must be refused -- and the stale record
        must be dropped so the slot is reclaimed.
        """
        handle = store.put(b"x", content_type="image/png")
        handle.path.unlink()
        with pytest.raises(UnknownAssetError):
            store.get(handle.asset_id)
        assert len(store) == 0

    def test_expiry_is_reported_as_an_iso_timestamp(self, store: AssetStore) -> None:
        """The contract's timestamp convention: ISO 8601, `Z`-suffixed."""
        handle = store.put(b"x", content_type="image/png")
        assert handle.expires_at_iso.endswith("Z")
        assert "T" in handle.expires_at_iso


# ---------------------------------------------------------------------------
# Capacity: bounded, and it never evicts a live handle
# ---------------------------------------------------------------------------


class TestCapacity:
    def test_the_store_refuses_when_full(self, store: AssetStore) -> None:
        """`max_files=4` in the fixture."""
        for _ in range(4):
            store.put(b"x", content_type="image/png")
        with pytest.raises(AssetStoreFullError):
            store.put(b"x", content_type="image/png")

    def test_a_full_store_reclaims_expired_slots_before_refusing(
        self, store: AssetStore
    ) -> None:
        """The refusal is only reached when every slot is LIVE."""
        for _ in range(4):
            store.put(b"x", content_type="image/png", now=0.0)
        # All four have expired by now; a new upload must succeed.
        handle = store.put(b"y", content_type="image/png", now=61.0)
        assert handle.asset_id
        assert len(store) == 1

    def test_a_live_handle_is_never_evicted_to_make_room(
        self, store: AssetStore
    ) -> None:
        """Evicting a live handle would invalidate one a client is about to use,
        turning a working flow into an intermittent failure.

        The first four handles were created at now=0; the fifth upload at now=1
        finds all four still live, so it must be REFUSED even though a slot would
        free up at now=60.
        """
        live = [store.put(b"x", content_type="image/png", now=0.0) for _ in range(4)]
        with pytest.raises(AssetStoreFullError):
            store.put(b"y", content_type="image/png", now=1.0)
        for handle in live:
            assert store.get(handle.asset_id, now=1.0).asset_id == handle.asset_id

    def test_the_unreached_operator_diagnostic_is_gone(self, store: AssetStore) -> None:
        """F-11 (owner ruling 2026-09-23): `stats()` was REMOVED.

        This guard replaces the pre-ruling test that asserted `stats()` returned
        six values without disclosing a path. It fails on the pre-ruling state,
        where `stats()` existed -- so it is not vacuous.

        Why removal rather than wiring: `stats()` had no production caller, and
        its two counters (`_sweeps`, `_refusals`) were incremented on the
        request path to feed a reader that did not exist. The owner ruling is
        to remove the unused per-request computation, not to add a consumer.
        """
        assert not hasattr(store, "stats"), (
            "AssetStore.stats() is back. F-11 removed it because nothing "
            "called it; if it returns it needs a real production caller and a "
            "document that points at it."
        )
        assert not hasattr(store, "_refusals"), (
            "the per-request `_refusals` counter is back; it existed only to "
            "feed `stats()`"
        )
        assert not hasattr(store, "_sweeps"), (
            "the `_sweeps` counter is back; it existed only to feed `stats()`"
        )

        # Removing it took nothing away: every fact it reported is still
        # reachable from where it actually lives, and none of it is a path.
        store.put(b"x", content_type="image/png")
        assert len(store) == 1
        assert store.max_files == 4
        assert store.max_file_bytes == 1024
        assert store.ttl_seconds == 60.0


# ---------------------------------------------------------------------------
# Obligation 3: the retry story, stated as behaviour
# ---------------------------------------------------------------------------


class TestRetrySemantics:
    """There is no idempotency key. The contract must record that; the behaviour
    must be that a retry is a NEW handle, not a replaced one."""

    def test_a_retried_upload_yields_a_new_handle(self, store: AssetStore) -> None:
        """The two requests are indistinguishable at the server, and no
        plan-specified key exists. So a retry is a new handle -- and the cost is
        a leaked handle, which capacity bounds and sweeping reclaims.
        """
        first = store.put(b"payload", content_type="image/png")
        second = store.put(b"payload", content_type="image/png")
        assert first.asset_id != second.asset_id, (
            "the second upload reused the first handle; that is idempotent "
            "behaviour, which this endpoint does not implement and must not "
            "appear to"
        )
        # Both are live and independently retrievable.
        assert store.get(first.asset_id).bytes == 7
        assert store.get(second.asset_id).bytes == 7

    def test_a_leaked_handle_is_reclaimed_by_capacity_or_expiry(
        self, store: AssetStore
    ) -> None:
        """The bounded cost of the retry story, asserted rather than assumed."""
        for _ in range(4):
            store.put(b"x", content_type="image/png", now=0.0)
        assert len(store) == 4
        # Expiry reclaims all of them, so a retry loop cannot fill the store
        # permanently.
        assert store.sweep(now=61.0) == 4
        assert len(store) == 0


# ---------------------------------------------------------------------------
# Resolving a pair
# ---------------------------------------------------------------------------


class TestResolveMany:
    def test_both_handles_resolve(self, store: AssetStore) -> None:
        a = store.put(b"a", content_type="image/png")
        b = store.put(b"b", content_type="image/png")
        resolved = store.resolve_many([a.asset_id, b.asset_id])
        assert [h.asset_id for h in resolved] == [a.asset_id, b.asset_id]

    def test_one_bad_handle_fails_the_whole_pair(self, store: AssetStore) -> None:
        """All-or-nothing.

        A partial resolution would let an analysis start with one of a required
        pair missing, and the specialist would then reject it with a *pairing*
        error that names the wrong cause. Failing here names the real one.
        """
        good = store.put(b"a", content_type="image/png")
        with pytest.raises(UnknownAssetError):
            store.resolve_many([good.asset_id, "asset_does_not_exist"])

    def test_resolution_preserves_request_order(self, store: AssetStore) -> None:
        """Order is load-bearing: a two-asset capability takes assets in request
        order (`core/planner.py::_indices_for`), so a temporal pair must not be
        silently reversed."""
        a = store.put(b"a", content_type="image/png")
        b = store.put(b"b", content_type="image/png")
        assert [h.asset_id for h in store.resolve_many([b.asset_id, a.asset_id])] == [
            b.asset_id,
            a.asset_id,
        ]


class TestConstruction:
    def test_an_invalid_capacity_is_refused(self, tmp_path_factory) -> None:
        with pytest.raises(ValueError):
            AssetStore(
                root=tmp_path_factory.mktemp("bad"),
                max_file_bytes=0,
            )

    def test_an_invalid_ttl_is_refused(self, tmp_path_factory) -> None:
        with pytest.raises(ValueError):
            AssetStore(
                root=tmp_path_factory.mktemp("bad2"),
                max_file_bytes=64,
                ttl_seconds=0,
            )

    def test_clear_removes_records_and_files(self, store: AssetStore) -> None:
        handles = [store.put(b"x", content_type="image/png") for _ in range(3)]
        store.clear()
        assert len(store) == 0
        for handle in handles:
            assert not handle.path.exists()


# ---------------------------------------------------------------------------
# F-11 — the operator surface, derived from source rather than assumed
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRODUCTION_DIRS = ("app", "gateway", "core")


def _production_python_files():
    for d in _PRODUCTION_DIRS:
        for path in (_REPO_ROOT / d).rglob("*.py"):
            if any(part in {"__pycache__", ".venv"} for part in path.parts):
                continue
            yield path


def _attribute_uses(name: str) -> list[str]:
    """Every `<something>.<name>` reference in production source, as `file:line`."""
    hits: list[str] = []
    for path in _production_python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == name:
                hits.append(f"{path.relative_to(_REPO_ROOT).as_posix()}:{node.lineno}")
    return sorted(hits)


class TestTheOperatorDiagnosticSurface:
    """F-11, **CLOSED by owner ruling 2026-09-23**.

    The pre-ruling guards asserted the GAP: `AssetStore.stats()` existed, was
    documented as "Counts for the operator", returned six values including
    `capacity_refusals`, and **nothing read it**. The owner ruled that the
    unused per-request computation be REMOVED rather than given a consumer --
    adding a metrics route is a contract change and the endpoint surface is
    fixed at four.

    So these guards assert the CLOSED state, not the open one. They are
    **moved, not deleted**, which is what the pre-ruling docstring instructed
    ("Do not delete the guard; move it"). Each fails on the pre-ruling source,
    so none of them is vacuous.
    """

    def test_the_diagnostic_has_been_removed(self) -> None:
        """`stats()` and the two counters that fed it are gone from the class.

        Fails pre-ruling: `AssetStore.stats` existed.
        """
        from gateway.assets import AssetStore

        assert not hasattr(AssetStore, "stats"), (
            "AssetStore.stats() is back. F-11 was closed by REMOVING it: it had "
            "no production caller, and its counters were paid for on every "
            "request. Re-opening it needs a real consumer and a document that "
            "points at one."
        )
        assert not hasattr(AssetStore, "_refusals"), (
            "the per-request `_refusals` counter is back; its only reader was "
            "the removed `stats()`"
        )
        assert not hasattr(AssetStore, "_sweeps"), (
            "the `_sweeps` counter is back; its only reader was the removed "
            "`stats()`"
        )

    def test_the_refusal_counter_is_gone_from_the_source(self) -> None:
        """The value that distinguished the two `503` causes is not merely
        unreferenced -- it is **not defined**.

        That is the difference between the pre-ruling gap ("computed, unread")
        and the ruling's remedy ("not computed at all").

        NOTE ON THE INSTRUMENT (found while writing this guard): a naive
        `"capacity_refusals" not in source` check **fails on a comment** that
        merely names the removed value -- which is how the first version of this
        guard tripped on the very comment recording the removal. A guard that
        documentation can fail is measuring the wrong thing. So this parses the
        source and looks for the name as a **string literal** (`ast.Constant`),
        which is how the dict key was written, and ignores comments entirely.
        """
        import ast

        source = (_REPO_ROOT / "gateway" / "assets.py").read_text(encoding="utf-8")
        literals = [
            node.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant) and node.value == "capacity_refusals"
        ]
        assert literals == [], (
            "`capacity_refusals` is back in gateway/assets.py as a string literal "
            "(it was the key of the removed `stats()` dict). The owner ruling "
            "removed it, so the code and the document must agree."
        )
        # ...and the counter attribute itself must not be assigned anywhere.
        attrs = [
            node.attr
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Attribute)
            and node.attr in {"_refusals", "_sweeps"}
        ]
        assert attrs == [], (
            f"a removed counter attribute is referenced again: {attrs}"
        )
        assert _attribute_uses("capacity_refusals") == []

    def test_the_documentation_records_the_closure(self) -> None:
        """The document must record the RULING, not an open gap.

        The pre-ruling §5 recovery row warned that the two `503` causes cannot
        be told apart and that the instrument was absent. **That warning must
        survive** -- the ambiguity is real and is NOT fixed by removing the
        counter -- but the section must no longer describe `stats()` as an
        existing method that nothing reaches.
        """
        doc = (_REPO_ROOT / "docs" / "DEPLOYMENT_ARCHITECTURE.md").read_text(
            encoding="utf-8"
        )
        assert "does not expose" in doc, (
            "the §5 recovery row must keep warning that the two `503` causes "
            "cannot be told apart; removing the counter did not remove the "
            "ambiguity"
        )
        assert "F-11" in doc, "§5 must name the finding it records"
        assert "removed" in doc.lower(), (
            "§5 must record that F-11 was closed by REMOVAL (owner ruling "
            "2026-09-23), not left describing an open gap"
        )


class TestSectionNumbersAreUnique:
    """F-11b -- a defect I introduced myself while correcting F-11.

    Inserting the saturation note above reused the number `5.1`, which the rate
    limiter note already held. The result was two `### 5.1` headings and a table
    cell pointing at the wrong one -- the same documentation-drift class this
    audit exists to find, created by the correction rather than found by it.

    A duplicated heading number is not cosmetic: a cross-reference like
    "see §5.1" silently resolves to whichever section the reader happens to
    reach first. This guard makes the next such insert fail loudly.
    """

    def test_each_deployment_subsection_number_appears_once(self) -> None:
        doc = (_REPO_ROOT / "docs" / "DEPLOYMENT_ARCHITECTURE.md").read_text(
            encoding="utf-8"
        )
        numbers = re.findall(r"^### (5\.\d+) ", doc, flags=re.MULTILINE)
        assert numbers, "§5 has no numbered subsections; the notes were renumbered away"
        duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
        assert duplicates == [], (
            f"duplicated §5 subsection number(s): {duplicates}. A `see §x.y` "
            f"cross-reference now resolves ambiguously."
        )

    def test_the_rate_limiter_pointer_names_its_own_section(self) -> None:
        """The §5 table points at the rate-limiter note by number.

        Falsified the honest way: pointing this cell back at `§5.1` fails, since
        the rate limiter is now `§5.2`.
        """
        doc = (_REPO_ROOT / "docs" / "DEPLOYMENT_ARCHITECTURE.md").read_text(
            encoding="utf-8"
        )
        row = next(
            (ln for ln in doc.splitlines() if "Per-IP rate limit bypassed" in ln), None
        )
        assert row is not None, "the rate-limit bypass row was removed"
        assert re.search(r"see §5\.\d+", row), (
            "the rate-limit row no longer points at its explanatory section"
        )
        # The number in the pointer must name a real heading.
        number = re.search(r"see §(5\.\d+)", row).group(1)
        heading = re.search(
            rf"^### {re.escape(number)} (.+)$", doc, flags=re.MULTILINE
        )
        assert heading is not None, f"the pointer names §{number}, which does not exist"
        assert "rate limiter" in heading.group(1).lower(), (
            f"§{number} is titled {heading.group(1)!r}, not the rate-limiter section"
        )
