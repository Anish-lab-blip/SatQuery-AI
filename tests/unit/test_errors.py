"""Tests for `core.errors` — the taxonomy, and the F-15 path scrubber.

`scrub_paths` exists because the owner ruling F-15 (2026-09-23) requires
client-facing exception messages to be sanitized while full detail is retained
server-side. Exception messages in this repo routinely embed an ABSOLUTE path,
and those strings reach fields an unauthenticated caller receives
(`API_CONTRACT.md` section 7: there is no auth in v1).

The scrub is a BASENAME reduction, not a deletion. That distinction is the
whole design: a blunt replacement also stops the leak, but it discards
path-free reasons the client can act on, and three existing tests pinning those
reasons had to be weakened before the granularity was corrected. This module
pins the granularity directly.

The URL case is the load-bearing one. One of the very messages being scrubbed
contains `https://github.com/antofuller/CROMA`, and a scrubber that mangles it
would be a worse defect than the disclosure it repairs.
"""

from __future__ import annotations

import pytest

from core.errors import (
    ModelLoadError,
    SatQueryError,
    SpecialistError,
    scrub_paths,
)

# ---------------------------------------------------------------------------
# The scrubber
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        # Windows, backslashes.
        (
            r"could not read encoder weights from C:\Users\anish\artifacts\head.pt",
            "could not read encoder weights from head.pt",
        ),
        # Windows, forward slashes (the Space builds these on some paths).
        (
            "checkpoint_path does not exist: C:/srv/satquery/artifacts/change/x.pt",
            "checkpoint_path does not exist: x.pt",
        ),
        # UNC share.
        (
            r"cannot read \\fileserver\share\data\t1.tif",
            "cannot read t1.tif",
        ),
        # POSIX absolute (the Linux Space).
        (
            "cannot read /home/operator/satquery/data/t1.tif",
            "cannot read t1.tif",
        ),
        # Path inside quotes, which is how croma.py writes it.
        (
            "which is not in '/srv/satquery/vendor/optical_sar'.",
            "which is not in 'optical_sar'.",
        ),
    ],
)
def test_an_absolute_path_is_reduced_to_its_basename(raw: str, expected: str) -> None:
    assert scrub_paths(raw) == expected


def test_a_url_is_left_intact() -> None:
    """The load-bearing negative.

    `croma.py`'s own missing-vendor message contains this URL. Mangling it
    would break the one piece of the message that tells an operator where to
    get the file -- a worse outcome than the path disclosure being fixed.
    """
    message = (
        "It is not available on PyPI; download it from "
        "https://github.com/antofuller/CROMA and retry."
    )
    assert scrub_paths(message) == message


@pytest.mark.parametrize(
    "message",
    [
        "and/or",
        "either/or both",
        "see artifacts/change/head.pt",  # relative: not a location disclosure
        "the ratio 3/4 of the pixels",
        "no GPU in this dimension",
        "has no builder 'build_that_does_not_exist'",
    ],
)
def test_prose_and_relative_paths_are_untouched(message: str) -> None:
    """A scrubber that mangles ordinary text is worse than no scrubber.

    Relative paths are deliberately out of scope: a rule broad enough to catch
    `artifacts/change/head.pt` also catches `and/or` and the segments of a URL.
    """
    assert scrub_paths(message) == message


@pytest.mark.parametrize("value", [None, ""])
def test_empty_input_passes_through(value: str | None) -> None:
    assert scrub_paths(value) == value


def test_the_real_croma_message_keeps_its_reason_and_loses_its_location() -> None:
    """End to end on the message that was actually measured leaking.

    Measured before the fix, this string reached FOUR client-visible fields
    with the absolute vendor directory intact. After it, the reason survives
    and the location does not.
    """
    leak = r"C:\Users\anish\AppData\Local\Temp\p14c_x\empty_vendor_dir"
    raw = (
        "CROMA requires the vendored 'use_croma.py', which is not in "
        f"'{leak}'. It is not available on PyPI; download it from "
        "https://github.com/antofuller/CROMA (the official instructions are "
        "'you will need the use_croma.py file and pretrained weights'). Place "
        "it in the optical_sar vendor directory and retry."
    )
    scrubbed = scrub_paths(raw) or ""

    assert leak not in scrubbed, "the absolute location survived"
    assert "C:\\Users" not in scrubbed, "a directory component survived"
    # ... and everything an operator can act on is still there.
    assert "use_croma.py" in scrubbed
    assert "empty_vendor_dir" in scrubbed
    assert "https://github.com/antofuller/CROMA" in scrubbed


# ---------------------------------------------------------------------------
# The taxonomy is unchanged by the above
# ---------------------------------------------------------------------------
def test_the_scrubber_does_not_touch_the_taxonomy() -> None:
    """`scrub_paths` is a function, not a hook: error classes are untouched."""
    error = ModelLoadError(r"cannot read C:\a\b\c.pt", specialist="change")
    assert error.detail == r"cannot read C:\a\b\c.pt", (
        "the error object must retain the full detail -- the SERVER-SIDE copy; "
        "scrubbing happens at the client-facing write sites"
    )
    assert error.user_message == ModelLoadError.user_message
    assert isinstance(error, SpecialistError)
    assert isinstance(error, SatQueryError)
    assert error.code == "model_load_error"
