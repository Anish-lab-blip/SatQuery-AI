"""Subprocess probe: run the demo in a FRESH interpreter and report which
top-level modules it newly imports.

WHY A SUBPROCESS, AND NOT AN IN-PROCESS GUARD
---------------------------------------------
An in-process check cannot answer "does the demo import a model stack?":

  * pytest imports every test module during COLLECTION, and several of them do
    module-scope `import torch` (e.g. `tests/routing/test_router.py`,
    `tests/model/test_vlm_contract.py`, `tests/unit/test_change.py`,
    `tests/unit/test_grounding_nms.py`) and `tests/unit/test_planner.py` imports
    `router.classifier`. So by the time any test body runs, `torch` is already
    in `sys.modules`, and a "newly imported" diff is blind to a real import.
  * a blocked or FAILED import never enters `sys.modules` at all, so a diff
    misses that case too.

A fresh interpreter has an empty baseline, so the diff is meaningful. This is
the only construction immune to both problems. The parent test
(`tests/e2e/test_demo.py`) asserts this probe's exit code and parses its JSON.

FAILURE SEMANTICS (why the negative control imports, rather than raises)
------------------------------------------------------------------------
A builder that RAISES is not a reliable probe. `SpecialistRegistry.build()`
catches broadly (`core/registry.py:404-405`) and converts any construction
failure into a retained UNAVAILABLE entry, so the run continues and `main`
still returns 0. A swallowed exception therefore leaves no observable trace in
`sys.modules` or in the return code. So the `forbidden` mode makes a builder
*import* a module (which succeeds and lands in `sys.modules`) instead.

USAGE
-----
    python -m tests.e2e._demo_probe <out_dir> <mode>

mode:
    clean      -> run the demo unchanged; exit 0 unless a forbidden module was
                  newly imported.
    forbidden  -> replace the `vqa` stub builder with one that does
                  `import torch`, then run the demo; exit non-zero because
                  torch is then newly imported. This is the negative control
                  that proves the check CAN fail.

Output: a single JSON object on stdout (the demo's own stdout is redirected so
the stream stays parseable).
"""

from __future__ import annotations

import contextlib
import io
import json
import sys

#: Top-level modules whose appearance means a model (or the model stack) was
#: pulled in. Kept as top-level names so a `torch.foo` submodule is caught too.
FORBIDDEN_TOP_LEVEL = ("torch", "transformers", "timm", "torchvision", "gradio", "croma")


def _top_level() -> set[str]:
    """The set of top-level module names currently imported."""
    return {name.split(".")[0] for name in sys.modules}


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(json.dumps({"error": "usage: _demo_probe <out_dir> [clean|forbidden]"}))
        return 2
    out_dir = argv[1]
    mode = argv[2] if len(argv) > 2 else "clean"

    before = _top_level()

    import demo.run_demo as run_demo

    if mode == "forbidden":
        original = run_demo._STUB_BUILDERS["vqa"]

        def _forbidden_builder(config, **kwargs):  # noqa: ANN001
            import torch  # noqa: F401 - the whole point: pull in the model stack
            return original(config, **kwargs)

        run_demo._STUB_BUILDERS["vqa"] = _forbidden_builder

    # The demo prints one summary line per query; keep stdout clean for our JSON.
    with contextlib.redirect_stdout(io.StringIO()):
        demo_rc = run_demo.main(["--out-dir", out_dir])

    new_top_level = sorted(_top_level() - before)
    forbidden = sorted(set(new_top_level) & set(FORBIDDEN_TOP_LEVEL))

    print(
        json.dumps(
            {
                "mode": mode,
                "demo_rc": demo_rc,
                "new_top_level": new_top_level,
                "forbidden": forbidden,
            }
        )
    )
    return 1 if forbidden else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
