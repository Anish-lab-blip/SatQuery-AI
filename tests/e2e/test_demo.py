"""End-to-end tests for the Phase-19 demonstration driver (`demo.run_demo`).

Two kinds of claim are checked here, and they need different machinery:

  1. The demo produces exactly one well-formed JSON report per fixture query,
     with the honesty rule holding (`calibrated is null`, `method`
     `"uncalibrated"`). This is checked IN-PROCESS — it is a statement about the
     reports the demo wrote.

  2. The demo constructs no model. This is checked in a FRESH INTERPRETER, via
     `tests/e2e/_demo_probe.py`, and the check is PROVEN able to fail by a
     negative control.

WHY THE MODEL-FREE CHECK IS NOT IN-PROCESS
------------------------------------------
An in-process "did the demo import a model stack?" check is unsound, in both
directions:

  * pytest imports every test module during COLLECTION, and at least six of them
    do module-scope `import torch` (plus `tests/unit/test_planner.py` imports
    `router.classifier`). `torch` is therefore already in `sys.modules` before
    any test body runs, so a "newly imported" diff is blind to a real import —
    it cannot TRUE-negative.
  * a blocked or FAILED import never enters `sys.modules`, so the same diff
    cannot catch that case either.

The corollary is the point the earlier framing got wrong: a preload-blind check
is not "immune to false negatives" — it is unable to be a negative at all in a
shared process. The check is only meaningful in isolation, and the subprocess
probe is what guarantees that isolation.

Note also that a builder which RAISES is not a usable probe: `build()`
(`core/registry.py:404-405`) catches broadly and turns the failure into a
retained UNAVAILABLE entry, so `main` still returns 0 and nothing observable
changes. The negative control therefore makes a builder *import* a module.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from core.config import load_config

from demo import run_demo
from reports import generator as report_generator

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROBE_MODULE = "tests.e2e._demo_probe"


def _run_probe(out_dir: Path, mode: str) -> tuple[int, dict]:
    """Run the demo in a fresh interpreter; return (returncode, json summary).

    `sys.executable` is the interpreter running the tests, so the child has the
    same environment (notably `torch`, which IS installed in `.venv` — making
    the clean-mode negative a real one).
    """
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_REPO_ROOT) + (os.pathsep + existing if existing else "")
    completed = subprocess.run(
        [sys.executable, "-m", _PROBE_MODULE, str(out_dir), mode],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    summary: dict = {}
    for line in reversed(completed.stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{"):
            summary = json.loads(stripped)
            break
    return completed.returncode, summary


# ---------------------------------------------------------------------------
# 1. Model-free — in a fresh interpreter, with a negative control
# ---------------------------------------------------------------------------
def test_demo_constructs_no_model_in_a_fresh_interpreter(tmp_path) -> None:
    out_dir = tmp_path / "reports"

    code, summary = _run_probe(out_dir, "clean")

    assert code == 0, summary
    assert summary.get("mode") == "clean"
    assert summary.get("demo_rc") == 0
    assert summary.get("forbidden") == []
    # The child wrote one report per fixture query.
    for fixture in run_demo.FIXTURE_QUERIES:
        assert (out_dir / f"{fixture.name}.json").exists(), fixture.name


def test_the_no_model_check_can_actually_fail(tmp_path) -> None:
    """Negative control: a builder that imports torch must trip the probe.

    Without this, the check above could be a tautology. The forbidden-mode child
    replaces the `vqa` stub builder with one that does `import torch`; `torch` is
    installed, so the import succeeds and the probe must report it and exit
    non-zero. The `"torch" in forbidden` assertion also guards against the child
    failing for an unrelated reason (a crash yields an empty summary).
    """
    out_dir = tmp_path / "reports_forbidden"

    code, summary = _run_probe(out_dir, "forbidden")

    assert code != 0, "the probe must fail when a builder imports torch"
    assert summary.get("mode") == "forbidden"
    assert "torch" in summary.get("forbidden", [])


# ---------------------------------------------------------------------------
# 2. Reports — in-process, on the actual written files
# ---------------------------------------------------------------------------
def test_demo_writes_one_well_formed_report_per_fixture(tmp_path) -> None:
    exit_code = run_demo.main(["--out-dir", str(tmp_path)])

    assert exit_code == 0
    assert run_demo.FIXTURE_QUERIES, "the demo must have fixtures"
    assert sorted(p.name for p in tmp_path.glob("*.json")) == sorted(
        f"{fixture.name}.json" for fixture in run_demo.FIXTURE_QUERIES
    )

    for fixture in run_demo.FIXTURE_QUERIES:
        path = tmp_path / f"{fixture.name}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert set(data) == report_generator.REPORT_FIELDS
        assert data["task"] == fixture.task.value
        assert data["answer"]
        assert data["trace_states"]
        # The honesty rule, restated after calibration was actually wired.
        #
        # This used to assert `calibrated is None` and `method == "uncalibrated"`
        # with the comment "no fitted artifact exists, so no calibration is
        # claimed". That was true when written and is now stale: a real
        # temperature-scaling artifact exists at artifacts/calibration_v001.json
        # and the controller loads it, so the demo emits a calibrated value.
        #
        # The property that actually matters is not "uncalibrated" -- it is that
        # the report never claims a calibration it did not apply, and never
        # presents a raw score as calibrated. So: `method` must be a known
        # method, and `calibrated` must be present exactly when the method says a
        # correction was applied. That holds whether or not the artifact exists,
        # which is what makes it a better test than the one it replaces.
        confidence = data["confidence"]
        assert confidence["method"] in ("uncalibrated", "temperature_scaling")
        if confidence["method"] == "uncalibrated":
            assert confidence["calibrated"] is None
        else:
            assert confidence["calibrated"] is not None
            assert confidence["raw"] is not None


def test_demo_controller_is_built_from_stub_specialists() -> None:
    """The registry the demo drives resolves only the fixed, in-memory stubs."""
    controller = run_demo._build_demo_controller(load_config())

    vqa = controller.registry.build("vqa")
    grounding = controller.registry.build("grounding")

    assert isinstance(vqa.specialist, run_demo._StubVqaSpecialist)
    assert isinstance(grounding.specialist, run_demo._StubGroundingSpecialist)


def test_demo_report_is_deterministic_across_two_runs(tmp_path) -> None:
    """Two runs over the same fixtures produce byte-identical reports."""
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert run_demo.main(["--out-dir", str(first)]) == 0
    assert run_demo.main(["--out-dir", str(second)]) == 0

    for fixture in run_demo.FIXTURE_QUERIES:
        a = (first / f"{fixture.name}.json").read_text(encoding="utf-8")
        b = (second / f"{fixture.name}.json").read_text(encoding="utf-8")
        assert a == b, fixture.name


def test_demo_default_out_dir_is_declared() -> None:
    assert run_demo.DEFAULT_OUT_DIR == "artifacts/demo_reports"
