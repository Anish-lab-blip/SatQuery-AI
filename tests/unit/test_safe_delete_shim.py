"""The Windows safe-delete shim: verbatim prefixes, and the shell's return code.

WHY THIS FILE EXISTS
--------------------
Two defects in the WorkBuddy safe-delete shim (`cli/vendor/shim/sitecustomize.py`)
produced phantom test failures in this repo. Both are recorded with measurements
in `docs/STEP8_FINAL_CONFORMANCE_AUDIT.md` section 31; this file is the guard that
keeps them fixed.

DEFECT 1 -- a verbatim temp path was not recognised as a temp path
------------------------------------------------------------------
`_path_for_compare` returned `normcase(realpath(abspath(path)))`, which preserves
the Windows verbatim prefix `\\\\?\\` verbatim. `os.path.relpath` cannot relate
`\\\\?\\c:\\...` to `c:\\...`, so `_is_under_root` returned False and the OS-temp
exemption in `_should_bypass_safe_delete` did not apply.

Measured, before the fix::

    plain    temp subdir  -> bypass True
    verbatim temp subdir  -> bypass False      <-- the bug
    non-temp subdir       -> bypass False

That is why routine pytest `garbage-*` collection -- which walks
`\\\\?\\C:\\Users\\...\\Temp\\pytest-of-anish\\garbage-*` -- reached the bulk
guard, tripped `confirmRequired` at 69 entries against a threshold of 50, and
latched a rejection that then blocked *every* delete in the conversation.

DEFECT 2 -- a successful delete was reported as a failure
---------------------------------------------------------
`_platform_trash` raised whenever `SHFileOperationW` returned non-zero. Measured
on this host with `FOF_ALLOWUNDO|FOF_NOCONFIRMATION|FOF_NOERRORUI|FOF_SILENT`,
calling shell32 directly (the shim not involved)::

    series                                   non-temp rc      temp rc
    4 regions x 3 interleaved rounds         2, 2, 2, ...     0, 0, 0
    6 processes x 20 deletes                  2 (120/120)      0 (120/120)

The target is removed in **every** case, and the user's Recycle Bin is populated
and active (1,300+ `$I` / `$R` entries), so the return code is not a reliable
"could not delete" signal.

The code is also **intermittent**: across pytest invocations the same non-temp
delete sometimes returned 0, and in one run a *temp* delete returned non-zero.
The precise Windows-internal trigger is NOT established and is not claimed here.
What is established, and what the fix rests on, is that the code does not track
whether the delete succeeded.

That is why the guard for this defect stubs the shell: a test that depends on the
real return code would sometimes pass against the broken shim. The stub tests
fail on the original shim on every run.

WHAT IS ASSERTED
----------------
  * A verbatim-prefixed path inside the OS temp root compares equal to its plain
    form and is exempted from the guard.
  * A verbatim-prefixed path *outside* the temp root is still guarded -- the fix
    normalises the prefix, it does not widen the exemption.
  * Deleting through a verbatim temp path succeeds end to end.
  * A non-zero shell code on a delete whose target is genuinely gone does not
    raise (stubbed shell, stubbed `lexists` -- deterministic, deletes nothing).
  * A non-zero shell code on a delete whose target SURVIVES still raises
    (stubbed shell). Fail-closed is preserved; only the false positive is gone.
  * Against the real shell, a non-temp delete does not raise (behavioural
    confirmation, not the guard -- see the test's own docstring).

A RULE FOR ANYONE EXTENDING THIS FILE
-------------------------------------
**Do not perform a real delete outside the OS temp root.** The shim wraps
`os.unlink`, `os.remove`, `os.rmdir`, `shutil.rmtree` and `Path.unlink`, so an
"innocent" `os.unlink` in a fixture reaches the bulk guard. An earlier draft of
the defect-2 test did exactly that, tripped `confirmRequired` at the threshold
and latched a rejection that blocked every subsequent delete in the session --
the very failure this file exists to prevent. Stub the shell and patch
`os.path.lexists` instead; that is what the tests here do, and it is why the
guard's state file is byte-identical before and after a run of this module.

These tests exercise the shim that is actually loaded into this interpreter, so
they skip cleanly when the shim is absent or disabled (a non-Windows host, a bare
Python, CI without WorkBuddy). They never touch the bulk guard's state, and they
never delete anything the test did not create.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

#: The Windows verbatim / extended-length prefix.
VERBATIM = "\\\\?\\"

sitecustomize = pytest.importorskip(
    "sitecustomize",
    reason="the WorkBuddy safe-delete shim is not installed in this interpreter",
)

needs_shim_helpers = pytest.mark.skipif(
    not hasattr(sitecustomize, "_should_bypass_safe_delete"),
    reason="this `sitecustomize` is not the safe-delete shim",
)

windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="the verbatim prefix is a Windows concept"
)

#: A path that is definitely NOT under the OS temp root, and is never created.
NON_TEMP = os.path.join(os.path.abspath(os.sep), "satquery-verbatim-probe", "x.txt")


def _temp_probe(relative: str) -> str:
    """A path under the OS temp root. Never created; the predicates are pure."""
    return os.path.join(tempfile.gettempdir(), "satquery-verbatim-probe", relative)


# ---------------------------------------------------------------------------
# Defect 1 -- the verbatim prefix must not defeat the temp exemption
# ---------------------------------------------------------------------------


@windows_only
@needs_shim_helpers
def test_a_verbatim_temp_path_compares_equal_to_its_plain_form() -> None:
    """The two forms name the same location, so they must normalise the same.

    This is the exact assertion the defect broke: `realpath` preserved the
    prefix, so the two strings differed and every downstream comparison failed.
    """
    plain = _temp_probe("garbage-abc")
    verbatim = VERBATIM + plain

    assert sitecustomize._path_for_compare(plain) == (
        sitecustomize._path_for_compare(verbatim)
    ), "a verbatim path did not normalise to its plain form"


@windows_only
@needs_shim_helpers
def test_a_verbatim_temp_path_is_recognised_as_a_temp_path() -> None:
    """Defect 1, at the predicate that actually decides the code path.

    `_safe_path_unlink` consults `_should_bypass_safe_delete` and returns the
    real unlink when it is True. Returning False here is what sent a temp path
    down the trash route and into the bulk guard.
    """
    plain = _temp_probe("garbage-abc")
    verbatim = VERBATIM + plain

    assert sitecustomize._is_under_os_tmp_dir(plain) is True, (
        "the plain temp path lost its exemption -- the predicate itself changed"
    )
    assert sitecustomize._is_under_os_tmp_dir(verbatim) is True, (
        "the verbatim temp path is still not recognised as a temp path"
    )
    assert sitecustomize._should_bypass_safe_delete(verbatim) is True, (
        "a temp path in verbatim form still reaches the bulk guard"
    )


@windows_only
@needs_shim_helpers
def test_a_verbatim_non_temp_path_is_still_guarded() -> None:
    """The fix normalises a prefix; it must not widen the exemption.

    A verbatim path outside the temp root has to keep going through the guard.
    If this test ever fails, the exemption has been broadened and the guard can
    be bypassed by writing `\\\\?\\` in front of any path.
    """
    verbatim = VERBATIM + NON_TEMP

    assert sitecustomize._is_under_os_tmp_dir(NON_TEMP) is False
    assert sitecustomize._should_bypass_safe_delete(verbatim) is False, (
        "the verbatim prefix became a way to bypass the guard"
    )


@windows_only
@needs_shim_helpers
def test_a_verbatim_unc_path_normalises_too() -> None:
    """`\\\\?\\UNC\\server\\share` is the UNC spelling of the same prefix."""
    verbatim = "\\\\?\\UNC\\server\\share\\x.txt"
    plain = "\\\\server\\share\\x.txt"

    assert sitecustomize._path_for_compare(verbatim) == (
        sitecustomize._path_for_compare(plain)
    )


@windows_only
@needs_shim_helpers
def test_deleting_through_a_verbatim_temp_path_succeeds(tmp_path_factory) -> None:
    """Defect 1, end to end: the delete happens and nothing is raised.

    The file is created and removed by this test, inside a directory this test
    owns, under the OS temp root -- where the exemption applies, so the guard is
    never consulted and no state is touched.
    """
    root = Path(tempfile.gettempdir()) / "satquery-verbatim-probe"
    root.mkdir(parents=True, exist_ok=True)
    target = root / "victim.txt"
    target.write_bytes(b"x")

    verbatim = Path(VERBATIM + str(target))
    verbatim.unlink()

    assert not target.exists()


# ---------------------------------------------------------------------------
# Defect 2 -- a non-zero return code on a delete that worked
# ---------------------------------------------------------------------------


@windows_only
@needs_shim_helpers
def test_a_nonzero_code_with_the_target_gone_is_not_a_failure(
    tmp_path_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defect 2, pinned deterministically -- and WITHOUT deleting anything.

    The shell is stubbed to return a non-zero code, and the target is reported
    as already gone, which is exactly the observed combination. Nothing may be
    raised, because the delete did happen.

    Two deliberate choices, both learned the hard way:

      * The shell is stubbed. The real shell returns a non-zero code for most
        non-temp deletes on this host but was also observed returning 0 in some
        invocations, so a test that depends on the real return code would
        sometimes pass against the broken shim and prove nothing. This one fails
        on the original shim every time.

      * The target is NOT actually deleted. An earlier draft of this test let
        the stub call `os.unlink`, which the shim wraps -- so the test performed
        a real non-temp delete, reached the bulk guard, tripped
        `confirmRequired` at the threshold and latched a rejection that then
        blocked every subsequent delete in the session. A regression test for
        the delete guard must not itself delete anything outside the temp root.
        Patching `os.path.lexists` exercises the same branch with no side effect.
    """
    base = tmp_path_factory.mktemp("shell-delete-nonzero")
    target = base / "victim.txt"
    target.write_bytes(b"x")

    calls = {"n": 0}

    class _ShellReportsFailure:
        @staticmethod
        def SHFileOperationW(_struct_ref: object) -> int:
            calls["n"] += 1
            return 2  # ERROR_FILE_NOT_FOUND, on a delete that succeeded

    real_lexists = os.path.lexists

    def _lexists(path: object) -> bool:
        # The target "is gone"; everything else is answered truthfully.
        if os.path.abspath(str(path)) == os.path.abspath(str(target)):
            return False
        return real_lexists(path)

    monkeypatch.setattr(sitecustomize, "_shell32", _ShellReportsFailure)
    monkeypatch.setattr(os.path, "lexists", _lexists)

    # Must not raise.
    sitecustomize._platform_trash(str(target))

    assert calls["n"] == 1, "the shell stub was never reached"
    assert target.exists(), "this test must not delete anything"


@windows_only
@needs_shim_helpers
def test_a_delete_whose_target_survives_still_raises(
    tmp_path_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail-closed is preserved: only the false positive was removed.

    The shell is stubbed to report failure *without* deleting, and the target is
    truthfully still present. That is the fact the shim exists to protect, so
    the error must still be raised. Without this test, "don't raise on a
    non-zero code" could be mistaken for "never raise".

    Deletes nothing: the stub does not remove the file.
    """
    base = tmp_path_factory.mktemp("shell-delete-failclosed")
    target = base / "victim.txt"
    target.write_bytes(b"x")

    class _FailingShell:
        @staticmethod
        def SHFileOperationW(_struct_ref: object) -> int:
            return 2  # ERROR_FILE_NOT_FOUND, and nothing is deleted

    monkeypatch.setattr(sitecustomize, "_shell32", _FailingShell)

    with pytest.raises(OSError) as caught:
        sitecustomize._platform_trash(str(target))

    assert "SHFileOperationW" in str(caught.value)
    assert target.exists(), "the stub was supposed to leave the target in place"


@windows_only
@needs_shim_helpers
def test_a_real_shell_delete_of_a_non_temp_file_does_not_raise(
    tmp_path_factory,
) -> None:
    """End-to-end against the real shell -- a behavioural confirmation.

    This is NOT the guard for defect 2; the two stub tests above are, because
    the real shell's return code is intermittent (it was measured non-zero for
    120 of 120 non-temp deletes in one series, and observed as 0 in some
    invocations). What this test does establish is that the shipped code path
    works against the actual shell with the actual Windows behaviour, which is
    the thing the suite depends on.

    Skips when the basetemp is itself under the OS temp root, because the shell
    returns 0 there and the non-temp path is not exercised.
    """
    base = tmp_path_factory.mktemp("shell-delete")
    if sitecustomize._is_under_os_tmp_dir(str(base)):
        pytest.skip(
            "basetemp is under the OS temp root, where the shell returns 0; "
            "run with a non-temp --basetemp to exercise this"
        )

    target = base / "victim.txt"
    target.write_bytes(b"x")
    assert target.exists()

    # Must not raise, whatever the shell returns.
    sitecustomize._platform_trash(str(target))

    assert not target.exists(), "the delete did not actually happen"
