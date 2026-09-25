"""The Space entrypoint must be cheap, honest, and hash-preserving.

WHY THIS EXISTS
---------------
`app/space_app.py` is the inference service that will run on the Hugging Face
Space. Its constraints come from `configs/deploy.yaml` and
`docs/DEPLOYMENT_ARCHITECTURE.md` section 3.3, and three of them have teeth:

* **`lazy_load: true` + requirement 4** -- a metadata request must not load a
  model. If importing this module built the controller, every ZeroGPU cold start
  would pay a full model load just to answer `/v1/health`, and the 5 GPU-minute
  daily budget would be spent on health checks.
* **5 GPU-minutes/day** -- the durations must be the frozen ones. A guessed
  duration either under-reserves (ZeroGPU kills a legitimate call) or
  over-reserves (the quota runs out). Neither is detectable from a log.
* **`Config.hash`** -- the entrypoint must not move it. Adding a
  `gpu_duration_change_vqa` key would be the obvious way to handle `change_vqa`,
  and it would invalidate the project's own benchmark number.

WHAT IS NOT ASSERTED
--------------------
`build_space_app()` and the ZeroGPU decoration cannot run here: FastAPI is
blocked by a WDAC policy (`orjson`, `WinError 4551`) and `spaces` is not
installed. These tests cover the module's import behaviour, its metadata path and
its inputs. The ASGI app and the `@spaces.GPU` decoration are **specified and
reviewed, not executed** -- recorded in `docs/PHASE19_FINAL_HARDENING.md`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _scratch_asset_dir(name: str) -> Path:
    """A private directory to point `SATQUERY_ASSET_DIR` at.

    WHY NOT `tmp_path`
    ------------------
    This sandbox intercepts directory removal and enforces a per-turn
    bulk-delete cap. pytest's `tmp_path` fixture garbage-collects during the
    run, so a suite with several `tmp_path` tests exhausts the turn's budget and
    later tests die in FIXTURE SETUP with `SystemExit: 1` from the shim:

        [safe-delete][SAFE_DELETE_BULK_REJECTED] {"count":97,"threshold":50}

    That turns an assertion into an ERROR for reasons unrelated to the code, and
    the failure moves as tests are added -- the worst kind of flake. This helper
    makes a directory the test owns and never deletes, so the endpoint's
    behaviour is what is measured. A per-PID name avoids collisions across runs.
    """
    import os

    path = REPO / ".scratch_space_app_assets" / f"{name}-{os.getpid()}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_the_entrypoint_imports_without_torch() -> None:
    """Requirement 1: importing must not pull the model stack.

    Asserted in a subprocess because the test process itself has already imported
    torch (via `core.config` or the servers) -- an in-process check would always
    see it and could never fail.

    CORRECTED 2026-09-22. This test previously also asserted
    `fastapi_present == "False"` ("the metadata path must not depend on the web
    framework"). That assertion is now **false and must stay false**: the G-1
    defect proved that a route annotation is resolved by `eval(annotation,
    func.__globals__)`, so `Request`, `Response` and `JSONResponse` have to be
    bound at *module* scope. Importing them inside `build_space_app` made every
    POST return `422 {"detail":[{"loc":["query","request"]}]}` **silently**, with
    no handler ever running.

    The original intent -- "a metadata request must not pay for the model stack"
    -- is preserved exactly by the torch half, which is the half that costs
    seconds on a ZeroGPU cold start. FastAPI is a small pure-Python import that
    the Space needs anyway to *serve* anything at all, so requiring its absence
    was never a real requirement; it was an over-reading of "imports cheaply".
    The real requirement (requirement 4, "never load a model for a metadata
    request") is tested separately by `test_describe_deployment_loads_no_model`.
    """
    script = (
        "import sys; from app import space_app; "
        "print('torch' in sys.modules, 'fastapi' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        check=True,
    )
    torch_present, fastapi_present = result.stdout.split()
    assert torch_present == "False", (
        "importing app.space_app pulled in torch; every ZeroGPU cold start would "
        "pay a model stack import just to answer /v1/health"
    )
    # fastapi is expected. Asserted rather than ignored so that a *future*
    # regression which swaps it for something heavier is visible.
    assert fastapi_present == "True", (
        "app.space_app no longer imports fastapi; if the web framework was "
        "removed from the module scope, check the route annotations first -- "
        "see the G-1 note in this docstring"
    )


def test_describe_deployment_loads_no_model() -> None:
    """Requirement 4: metadata must come from the filesystem, not from weights."""
    from app import space_app

    before = set(sys.modules)
    described = space_app.describe_deployment()
    newly_imported = set(sys.modules) - before

    # A model load would pull torch (or a specialist module) in.
    assert "torch" not in newly_imported, (
        "describe_deployment() imported torch; a metadata request would consume "
        "GPU quota"
    )
    assert described["capabilities"], "the description has no capabilities"


def test_describe_deployment_preserves_the_config_hash() -> None:
    """Requirement 5: the entrypoint must not move `Config.hash`."""
    from app import space_app
    from core.config import get_config

    described = space_app.describe_deployment()
    assert described["config_hash"] == "78f1e3700da15aa1"
    assert described["config_hash"] == get_config().hash


def test_the_gpu_durations_are_the_frozen_ones() -> None:
    """A guessed duration is undetectable from a log; pin it to the manifest."""
    import yaml

    from app import space_app

    deploy = yaml.safe_load((REPO / "configs" / "deploy.yaml").read_text("utf-8"))
    block = deploy["deployment"]

    for task, duration in space_app.GPU_DURATIONS.items():
        if task == "change_vqa":
            # No key of its own, deliberately: adding one moves Config.hash.
            assert f"gpu_duration_{task}" not in block, (
                "configs/deploy.yaml gained a gpu_duration_change_vqa key; "
                "Config.hash has moved and the frozen benchmark is invalid"
            )
            assert duration == block["gpu_duration_change"], (
                "change_vqa must reuse the change budget"
            )
            continue
        expected = block.get(f"gpu_duration_{task}")
        if expected is None:
            # vqa/caption share a key, which is fine; assert they agree.
            assert duration == block["gpu_duration_vqa"]
            continue
        assert duration == expected, (
            f"app.space_app.GPU_DURATIONS[{task!r}]={duration} disagrees with "
            f"configs/deploy.yaml ({expected})"
        )


def test_an_undeclared_task_is_a_refusal_not_a_default() -> None:
    """Guessing a duration would reserve the wrong slice of the daily budget."""
    from app import space_app

    with pytest.raises(KeyError, match="gpu_duration"):
        space_app.decorate_gpu("a_task_that_does_not_exist")


def test_the_gpu_decoration_is_an_identity_when_spaces_is_absent() -> None:
    """`cpu_mode_required: true` means a CPU run must work, not crash.

    Without `spaces` installed the decorator must return the function unchanged,
    so the CPU path is the same code path.
    """
    from app import space_app

    if space_app._spaces_module() is not None:
        pytest.skip("the `spaces` package is installed; the CPU fallback is moot")

    @space_app.decorate_gpu("change")
    def sample() -> str:
        return "ran"

    assert sample() == "ran"


def test_the_controller_is_not_built_at_import() -> None:
    """Requirement 1 and `lazy_load: true`: the controller is deferred."""
    import subprocess

    script = (
        "import sys; from app import space_app; "
        "print(space_app._CONTROLLER is None)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        check=True,
    )
    assert result.stdout.strip() == "True", (
        "the controller was built at import time; `lazy_load: true` and "
        "requirement 4 forbid that"
    )


def _code_only(source: str) -> str:
    """The module's source with docstrings removed.

    Needed because several prohibitions below are stated IN the docstrings
    ("this file deliberately does not build a Gradio interface"). Searching the
    raw source finds the prohibition itself and reports a violation. This helper
    is the fix for a mistake made twice in this repository's doc tests: a search
    for a forbidden token must run over the code, not over the prose that
    forbids it.
    """
    import ast

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                node.body = body[1:]
    return ast.unparse(tree)


def test_the_entrypoint_does_not_import_the_gradio_gui() -> None:
    """Phase 16 is the frontend agent's; this file must not compete with it.

    `docs/DEPLOYMENT_ARCHITECTURE.md` section 3.1 resolves the ownership
    question: the Phase-18 blocker applies to a Gradio GUI, so this entrypoint
    serves JSON only.
    """
    source = (REPO / "app" / "space_app.py").read_text(encoding="utf-8")
    code = _code_only(source)
    for forbidden in ("gr.Blocks", "gr.Interface", "import gradio", "from gradio"):
        assert forbidden not in code, (
            f"the entrypoint's CODE contains {forbidden!r}; it must serve JSON "
            f"only so it cannot compete with the frontend agent's Phase-16 GUI"
        )


def test_the_entrypoint_delegates_to_the_serving_composition_root() -> None:
    """Requirement 2: reimplementing the wiring would duplicate the F2 fix.

    `app/serving.py` wires the change and change-VQA heads through the registry's
    `builders=` override -- the seam that keeps `Config.hash` unchanged. A second
    implementation would drift from it.
    """
    source = (REPO / "app" / "space_app.py").read_text(encoding="utf-8")
    code = _code_only(source)
    assert "build_serving_controller" in code, (
        "the entrypoint does not use the existing composition root"
    )
    assert "SpecialistRegistry" not in code, (
        "the entrypoint builds its own registry instead of delegating"
    )


def test_describe_deployment_does_not_use_the_constructing_health_method() -> None:
    """`AnalysisController.health()` constructs everything; a probe must not.

    Its own docstring says so. Using it from `/v1/health` would load every model
    on every probe.
    """
    source = (REPO / "app" / "space_app.py").read_text(encoding="utf-8")
    code = _code_only(source)
    assert ".health()" not in code, (
        "the entrypoint calls AnalysisController.health(), which its docstring "
        "documents as 'Constructs everything'"
    )


def test_describe_deployment_returns_internal_registry_vocabulary() -> None:
    """`describe_deployment()` reports the *internal* view, not the wire shape.

    RENAMED AND REWRITTEN 2026-09-22. This test previously asserted a bare
    `available: bool` plus a reason. The owner ruling made the registry
    authoritative and put a single adapter (`app.deployment`) between it and the
    wire, so there are now two deliberately different shapes:

      * `deployment_report().capabilities_block()` -- the **wire** shape, which
        is contract section 2.2 (`task`, `available`, `reason`, `requires_pair`,
        `max_assets`). That is what a client sees.
      * `deployment_report().as_internal()` -- the **internal** shape, which
        carries the registry word (`registry_state`), the translated word
        (`contract_state`) and the artifact evidence (`artifacts`, `detail`). It
        is never served.

    `space_app.describe_deployment()` delegates to the second, because its job
    is to describe *this deployment to an operator*, not to speak to a client.
    The wire-level assertions live in `test_deployment_adapter.py`; this one
    pins the entrypoint's accessor to the internal vocabulary so the two cannot
    be silently swapped -- serving `as_internal()` would leak the registry
    vocabulary the ruling said must stay internal.
    """
    from app import space_app

    described = space_app.describe_deployment()
    assert described["capabilities"], "the description has no capabilities"

    for cap in described["capabilities"]:
        assert "capability" in cap, (
            f"capability {cap!r} has no `capability` key; the internal view is "
            f"keyed that way, not by `task`"
        )
        assert "registry_state" in cap and "contract_state" in cap, (
            f"capability {cap['capability']!r} does not carry both vocabularies; "
            f"the adapter exists precisely to hold them apart"
        )
        assert cap["registry_state"] in {"available", "degraded", "unavailable"}
        assert cap["contract_state"] in {
            "loaded",
            "absent",
            "unavailable",
            "not_requested",
            "evicted",
        }


def test_the_internal_description_carries_artifact_evidence_when_absent() -> None:
    """`as_internal()` names the *missing artifact*, not just the capability.

    This is the whole point of the split introduced by the ruling. The wire
    payload says `available: false` and a reason; the internal view additionally
    records which artifact was looked for and whether it was found, so an
    operator can tell "the weights are not on this host" apart from "the config
    path is unset". The measured reality on this machine is that `optical_sar`
    is absent because `croma.checkpoint_path` is unset -- a deployment fact, and
    the internal view must say so rather than blaming the registry.
    """
    from app import space_app

    described = space_app.describe_deployment()
    by_capability = {cap["capability"]: cap for cap in described["capabilities"]}

    assert "optical_sar" in by_capability, (
        "optical_sar is missing from the description; the registry resolves it "
        "even when its artifacts are not shipped"
    )

    absent = [c for c in by_capability.values() if c["contract_state"] == "absent"]
    for cap in absent:
        assert cap["reason"], (
            f"{cap['capability']!r} is absent with no reason; an absent "
            f"capability that explains nothing is not diagnosable"
        )
        assert cap["artifacts"] is not None, (
            f"{cap['capability']!r} is absent but carries no artifact evidence, "
            f"so an operator cannot see what was looked for"
        )


# ---------------------------------------------------------------------------
# STEP 8 follow-up -- asset sizing was hardcoded; it is now deployment state.
# The docs recorded the resulting ceiling as a known scaling limit. Raising it
# no longer requires editing code, so the limit is a configuration choice.
# ---------------------------------------------------------------------------
def test_the_asset_sizing_defaults_match_the_documented_values() -> None:
    """The defaults are load-bearing: the contract prints them as code constants.

    If a default moves, `API_CONTRACT.md` §2.5.1 and `DEPLOYMENT_ARCHITECTURE.md`
    §4 both become wrong. Pinning them here makes that a test failure rather than
    a documentation drift.
    """
    from app import space_app as space_app_module

    assert space_app_module._asset_max_files() == 32
    assert space_app_module._asset_ttl_seconds() == 900.0


def test_the_asset_sizing_is_environment_configurable(monkeypatch) -> None:
    """An operator must be able to raise the ceiling without editing code."""
    from app import space_app as space_app_module

    monkeypatch.setenv("SATQUERY_ASSET_MAX_FILES", "64")
    monkeypatch.setenv("SATQUERY_ASSET_TTL_S", "1800")
    assert space_app_module._asset_max_files() == 64
    assert space_app_module._asset_ttl_seconds() == 1800.0


def test_a_malformed_per_file_cap_is_refused_not_silently_defaulted(monkeypatch) -> None:
    """F-7. A bad `SATQUERY_MAX_FILE_BYTES` must not become a *different* cap.

    This function previously swallowed a parse failure and returned the 4 MiB
    default. Measured 2026-09-22, that meant the two layers did the *opposite*
    thing for the same value:

        SATQUERY_MAX_FILE_BYTES='abc'
            gateway -> raised at startup
            Space   -> silently used 4 MiB

    A silent default is the worse outcome of the two, and not because it is
    unsafe: it means an operator who typed a malformed cap keeps accepting
    uploads against a limit nobody chose, while the layer that *did* fail has
    already told them the value is bad. The Space must therefore refuse rather
    than paper over the value the gateway rejected -- and the message must name
    the variable, because this surfaces in a Space build or start log where the
    operator has no other context.

    Note this is deliberately NOT the same disposition as
    `SATQUERY_ASSET_MAX_FILES`/`_TTL_S` above. Those fall back, because a
    non-positive value there would *disable a safety property* (making every
    handle dead on arrival) rather than merely disagree with another layer; here
    the failure is a cross-layer disagreement, so failing closed is correct.
    """
    from app import space_app as space_app_module

    for bad in ("abc", "4e6", "4 MiB"):
        monkeypatch.setenv("SATQUERY_MAX_FILE_BYTES", bad)
        with pytest.raises(ValueError, match="SATQUERY_MAX_FILE_BYTES"):
            space_app_module._asset_max_file_bytes()


def test_both_layers_agree_on_the_per_file_cap_for_every_input(monkeypatch) -> None:
    """F-7's real invariant: one variable, and the layers cannot disagree.

    `app/space_app.py::get_asset_store`'s docstring claims the Space's
    `max_file_bytes` "mirrors `GatewayConfig`'s per-file cap so the two layers
    cannot disagree about what 'too large' means", and `_asset_max_file_bytes`
    says "A single source keeps the two layers agreeing".

    Reading one variable is necessary but NOT sufficient, and this test is what
    makes the claim true rather than aspirational: it drives BOTH parsers over
    the same inputs and requires that they either agree on the integer or BOTH
    refuse. Measured before the fix, four inputs diverged ('0', '-1', 'abc',
    '4e6') — each one a deployment where the two layers would act on different
    caps. This test failed on all four.
    """
    from app import space_app as space_app_module
    from gateway.policy import config_from_env

    base = {
        "SATQUERY_SPACE_URL": "https://s.hf.space",
        "SATQUERY_ALLOWED_ORIGINS": "https://f.example.com",
    }

    def gateway_cap(value):
        try:
            env = dict(base)
            if value is not None:
                env["SATQUERY_MAX_FILE_BYTES"] = value
            return ("ok", config_from_env(env).max_file_bytes)
        except ValueError:
            return ("refused", None)

    def space_cap(value, monkeypatch_):
        monkeypatch_.delenv("SATQUERY_MAX_FILE_BYTES", raising=False)
        if value is not None:
            monkeypatch_.setenv("SATQUERY_MAX_FILE_BYTES", value)
        try:
            return ("ok", space_app_module._asset_max_file_bytes())
        except ValueError:
            return ("refused", None)

    for value in (None, "1024", "4194304", "  1024  ", "0", "-1", "abc", "4e6"):
        g = gateway_cap(value)
        s = space_cap(value, monkeypatch)
        assert g == s, (
            f"SATQUERY_MAX_FILE_BYTES={value!r} produced gateway={g} but "
            f"space={s}. Reading one variable is not the same as agreeing on its "
            f"value; whichever layer is right, the other is now acting on a cap "
            f"the operator did not set."
        )


def test_a_bad_asset_sizing_value_falls_back_rather_than_disabling_expiry(
    monkeypatch,
) -> None:
    """A malformed or non-positive value must NOT become a zero TTL.

    A zero TTL would make every handle dead on arrival -- a silent breakage that
    looks like a server fault rather than a misconfiguration. Falling back to the
    default is the only safe reading of "this value is not usable".
    """
    from app import space_app as space_app_module

    for bad in ("abc", "-5", "0"):
        monkeypatch.setenv("SATQUERY_ASSET_MAX_FILES", bad)
        monkeypatch.setenv("SATQUERY_ASSET_TTL_S", bad)
        assert space_app_module._asset_max_files() == 32, f"max_files fell over on {bad!r}"
        assert space_app_module._asset_ttl_seconds() == 900.0, f"ttl fell over on {bad!r}"


def test_the_asset_store_is_built_with_the_resolved_values(
    monkeypatch,
) -> None:
    """The helpers must actually reach the store, not merely exist.

    Uses `_scratch_asset_dir` rather than a fixed path: a hardcoded `/tmp/...`
    leaves a real directory behind on Windows (where it resolves to
    `C:/tmp/...`), which is a filesystem side effect from a unit test. The
    helper also avoids the `tmp_path` delete-quota problem documented on it.
    """
    from app import space_app as space_app_module

    monkeypatch.setenv("SATQUERY_ASSET_MAX_FILES", "48")
    monkeypatch.setenv("SATQUERY_ASSET_TTL_S", "1200")
    monkeypatch.setenv("SATQUERY_ASSET_ENABLED", "1")
    monkeypatch.setenv("SATQUERY_ASSET_DIR", str(_scratch_asset_dir("store-values")))
    # Reset the process cache so the store is rebuilt from the new environment.
    monkeypatch.setattr(space_app_module, "_ASSET_STORE", None)
    store = space_app_module.get_asset_store()
    assert store.max_files == 48
    assert store.ttl_seconds == 1200.0


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# F-9 — the Space must not buffer a body it has already decided to refuse
# ---------------------------------------------------------------------------
class TestTheSpaceRefusesBeforeItBuffers:
    """`POST /v1/assets` used `await request.body()`, then passed the bytes to
    `store.put()`, which applies `max_file_bytes`.

    That is the F-6 defect on the other layer: the cap ran **after** the buffer,
    so it could reject the upload but could not save the memory.

    ## Why these guards measure allocation and not the status code

    The first version of this class asserted only `status == 413`. It was
    **vacuous**: reverting the fix to `await request.body()` still returned
    `413`, so the guards passed in both worlds. The status code is produced by
    `store.put()`'s own cap either way -- the fixed and unfixed paths differ
    only in *when* they refuse, which a status code cannot express.

    `413` is therefore necessary but not sufficient. What separates the two
    worlds is the peak allocation while the body streams in, so that is what
    these guards assert. Measured 2026-09-22 with the cap at 1 MiB
    (probe `probe_f9_space_body.py`, `tracemalloc` at ASGI level):

        body      peak (unbounded read)   peak (bounded read)
        0.5 MiB         1.0 MiB                1.0 MiB
        4 MiB           8.0 MiB                1.3 MiB
        16 MiB         32.0 MiB                1.3 MiB
        64 MiB        128.0 MiB                1.3 MiB   <- flat, no ceiling

    Allocation tracked body size **linearly with no ceiling** before the fix.
    The assertion is deliberately one-sided -- `peak < body_len` -- so it pins
    the defect (the body was held in full) without pinning an implementation
    detail or a specific allocator number, which would be brittle.

    The ASGI driver is required because `httpx` will not send a body without
    `Content-Length`, and the omitted-length case is exactly what mattered for
    F-6. Skipped rather than failed when the web stack is absent: this
    environment can supply FastAPI (measured 2026-09-22, fastapi 0.141.1) but
    the project has run under a WDAC policy that blocked `orjson.pyd`, so the
    guard must not turn a deployment difference into a red test.
    """

    _CAP = 1 * 1024 * 1024  # 1 MiB, small enough to exercise quickly

    @staticmethod
    async def _post(app, body_len: int, *, declare_length: bool):
        """Drive `POST /v1/assets` at ASGI level.

        Returns `(status, peak_bytes)`. `peak_bytes` is the peak traced
        allocation attributable to this one request, so it measures what the
        handler held rather than what the process happened to have lying
        around.
        """
        import tracemalloc

        chunk = 256 * 1024
        sent = 0
        statuses: list[int] = []

        async def receive() -> dict:
            nonlocal sent
            remaining = body_len - sent
            if remaining <= 0:
                return {"type": "http.disconnect"}
            n = min(chunk, remaining)
            sent += n
            return {
                "type": "http.request",
                "body": b"x" * n,
                "more_body": sent < body_len,
            }

        async def send(message: dict) -> None:
            if message["type"] == "http.response.start":
                statuses.append(message["status"])

        headers = [(b"content-type", b"application/octet-stream")]
        if declare_length:
            headers.append((b"content-length", str(body_len).encode()))
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/assets",
            "raw_path": b"/v1/assets",
            "query_string": b"",
            "root_path": "",
            "headers": headers,
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        }

        tracemalloc.start()
        try:
            await app(scope, receive, send)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return (statuses[0] if statuses else 0), peak

    @pytest.fixture()
    def space_app(self, monkeypatch, request):
        pytest.importorskip("fastapi", reason="the web stack is unavailable here")
        from app import space_app as space_app_module

        monkeypatch.setenv("SATQUERY_ASSET_ENABLED", "1")
        monkeypatch.setenv(
            "SATQUERY_ASSET_DIR",
            str(_scratch_asset_dir(f"buffers-{request.node.name}")),
        )
        monkeypatch.setenv("SATQUERY_MAX_FILE_BYTES", str(self._CAP))
        monkeypatch.setattr(space_app_module, "_ASSET_STORE", None)
        return space_app_module.build_space_app()

    def test_an_oversized_body_is_not_held_in_full(self, space_app) -> None:
        """F-9's actual defect: the handler held the whole body before refusing.

        This is the guard that fails on the unbounded read and passes on the
        bounded one. `peak < body_len` is the whole claim: the handler must
        never simultaneously hold more bytes than the cap it is enforcing.
        """
        import asyncio

        body_len = 4 * self._CAP
        status, peak = asyncio.run(self._post(space_app, body_len, declare_length=False))

        assert status == 413, (
            f"an oversized undeclared-length body got {status}; the Space must "
            f"refuse it with the contract's oversized_image envelope"
        )
        assert peak < body_len, (
            f"peak traced allocation was {peak} bytes for a {body_len}-byte "
            f"body: the Space buffered the request in full before refusing it. "
            f"The cap must be applied while reading, not by store.put() after "
            f"the bytes are already in memory."
        )

    def test_the_peak_does_not_grow_with_the_body(self, space_app) -> None:
        """The shape of the defect is unbounded growth, so pin the shape.

        A single size could be satisfied by a lucky allocator coincidence. Two
        sizes far apart cannot: if the reader is unbounded, the 16x larger body
        produces a proportionally larger peak; if it is bounded, both peaks sit
        in the same small band.
        """
        import asyncio

        small = 4 * self._CAP
        large = 16 * self._CAP
        s_status, s_peak = asyncio.run(self._post(space_app, small, declare_length=False))
        l_status, l_peak = asyncio.run(self._post(space_app, large, declare_length=False))

        assert (s_status, l_status) == (413, 413), (
            f"both bodies must be refused, got {s_status} and {l_status}"
        )
        assert l_peak < large, (
            f"a {large}-byte body peaked at {l_peak} bytes: still buffering in "
            f"full"
        )
        # The larger body must not push the peak up in proportion to itself.
        assert l_peak < (large // 2), (
            f"peaks were {s_peak} (for {small} bytes) and {l_peak} (for {large} "
            f"bytes); a fourfold body increase moved the peak by more than a "
            f"small band, which is the signature of an unbounded read"
        )

    def test_a_body_within_the_cap_is_still_accepted(self, space_app) -> None:
        """The negative control.

        Without it, a guard that refused *every* upload -- or a streaming reader
        that truncated at the wrong byte count -- would leave the tests above
        green. Pins the boundary inclusively: exactly the cap is accepted.
        """
        import asyncio

        status, _ = asyncio.run(
            self._post(space_app, self._CAP, declare_length=False)
        )
        assert status == 201, (
            f"a body of exactly SATQUERY_MAX_FILE_BYTES got {status}; the cap is "
            f"inclusive, and a streaming reader that rejects it is off by one"
        )

    def test_the_refusal_does_not_depend_on_content_length(self, space_app) -> None:
        """Both declared and omitted lengths must be refused identically.

        F-6's measurement is the reason this matters: a declared length was
        handled correctly while an **omitted** one was not, so a guard that only
        covers the declared case tests the client's honesty rather than the
        server's behaviour. The peak is asserted too, because the bounded reader
        must not trust `Content-Length` to decide when to stop.
        """
        import asyncio

        body_len = 4 * self._CAP
        declared, declared_peak = asyncio.run(
            self._post(space_app, body_len, declare_length=True)
        )
        omitted, omitted_peak = asyncio.run(
            self._post(space_app, body_len, declare_length=False)
        )
        assert declared == omitted == 413, (
            f"declared-length={declared}, omitted-length={omitted}; the Space "
            f"must not treat a declared size differently from an absent one, "
            f"because that is a guard on the client's claim"
        )
        assert max(declared_peak, omitted_peak) < body_len, (
            f"peaks were {declared_peak} (declared) and {omitted_peak} "
            f"(omitted) for a {body_len}-byte body; both paths must refuse "
            f"while reading"
        )


# ---------------------------------------------------------------------------
# The upload endpoint's TWO states: enabled with a handle, disabled with a
# refusal naming the switch. Both are deployment configuration, so both are
# asserted as configuration.
# ---------------------------------------------------------------------------
class TestTheUploadEndpointReportsItsConfigurationHonestly:
    """`POST /v1/assets` is the one endpoint that writes to disk, so whether it
    is ON is deployment state a client must be able to rely on.

    WHY THESE TWO TOGETHER, AND WHY THE FIRST ONE MUST ASSERT AN ID
    ---------------------------------------------------------------
    `_asset_store_available()` is a conjunction: it requires BOTH
    `SATQUERY_ASSET_ENABLED` and `SATQUERY_ASSET_DIR`. A test that only proved
    the disabled case would pass against a handler that refuses
    *unconditionally* -- i.e. against a deployment where the fourth endpoint is
    dead, which is exactly the state the operator instruction "enable asset
    registration/upload" exists to change. So the enabled case must assert the
    handler actually reached the store and minted a handle.

    And the assertion is on the ID's SHAPE, not merely its presence. "Non-empty
    string" is satisfied by an error message, a path, or a constant. The handle
    is `asset_<32 hex>` from `secrets.token_hex(16)` -- 128 bits of CSPRNG
    output, and it is the endpoint's only access control -- so the shape is the
    security property and is asserted as such.

    The filesystem-disclosure check is the other half of "suitable for the
    deployment": `AssetHandle.to_response()` deliberately omits `path`, and a
    client must not be able to learn where its bytes landed, because that is
    both an information disclosure and an invitation to bypass the handle.
    """

    _BODY = b"\x89PNG\r\n\x1a\n" + b"stand-in for an uploaded raster" * 8

    @staticmethod
    def _post_asset(app, *, content_type: str):
        """One `POST /v1/assets` at ASGI level, returning `(status, body_dict)`.

        Reads the response body rather than only the status, because the whole
        claim here is about what the envelope CONTAINS -- a 201 with no handle
        is not "upload works".
        """
        import asyncio
        import json

        async def _drive():
            received: list[dict] = []
            sent_body = {"chunks": []}

            async def receive() -> dict:
                if sent_body["chunks"]:
                    return {"type": "http.disconnect"}
                sent_body["chunks"].append(True)
                return {
                    "type": "http.request",
                    "body": TestTheUploadEndpointReportsItsConfigurationHonestly._BODY,
                    "more_body": False,
                }

            async def send(message: dict) -> None:
                if message["type"] == "http.response.start":
                    received.append({"status": message["status"]})
                elif message["type"] == "http.response.body":
                    received[-1]["body"] = message.get("body", b"")

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/assets",
                "raw_path": b"/v1/assets",
                "query_string": b"",
                "root_path": "",
                "headers": [
                    (b"content-type", content_type.encode()),
                    (
                        b"content-length",
                        str(
                            len(
                                TestTheUploadEndpointReportsItsConfigurationHonestly._BODY
                            )
                        ).encode(),
                    ),
                ],
                "client": ("127.0.0.1", 12345),
                "server": ("testserver", 80),
            }
            await app(scope, receive, send)
            status = received[0]["status"]
            raw = received[0].get("body", b"")
            try:
                parsed = json.loads(raw) if raw else {}
            except ValueError:
                parsed = {"_unparsed": raw.decode("utf-8", "replace")}
            return status, parsed

        return asyncio.run(_drive())

    def test_an_enabled_upload_returns_a_valid_asset_id(self, monkeypatch) -> None:
        """Enabled: 201, and the body carries a well-formed opaque handle.

        This is the positive control for the whole endpoint. Note it also pins
        that the store was actually reached -- a handler that returned 201 with
        an empty body, or a handle it never stored, would be reported as a
        working upload while `POST /v1/analyze` could not resolve it.
        """
        import re

        pytest.importorskip("fastapi", reason="the web stack is unavailable here")
        from app import space_app as space_app_module

        scratch = _scratch_asset_dir("enabled")
        monkeypatch.setenv("SATQUERY_ASSET_ENABLED", "1")
        monkeypatch.setenv("SATQUERY_ASSET_DIR", str(scratch))
        monkeypatch.setattr(space_app_module, "_ASSET_STORE", None)
        app = space_app_module.build_space_app()

        status, body = self._post_asset(app, content_type="application/octet-stream")

        assert status == 201, (
            f"an enabled deployment answered {status} for a valid upload; "
            f"body was {body!r}. SATQUERY_ASSET_ENABLED=1 with a writable "
            f"SATQUERY_ASSET_DIR must make POST /v1/assets accept."
        )
        asset_id = body.get("asset_id")
        assert isinstance(asset_id, str) and re.fullmatch(
            r"asset_[0-9a-f]{32}", asset_id
        ), (
            f"asset_id was {asset_id!r}, which is not the contract's opaque "
            f"handle shape (asset_ + 32 hex chars, from secrets.token_hex(16)). "
            f"The handle is this endpoint's only access control."
        )
        assert body.get("bytes") == len(self._BODY), (
            f"the envelope reported {body.get('bytes')} bytes for a "
            f"{len(self._BODY)}-byte upload"
        )
        assert body.get("content_type") == "application/octet-stream"
        assert "expires_at" in body

        # And the handle must actually resolve to stored bytes -- an ID the
        # store cannot find is not a usable upload.
        store = space_app_module.get_asset_store()
        resolved = store.get(asset_id)
        assert resolved is not None, (
            "the returned asset_id does not resolve in the store; a client "
            "could not use it in POST /v1/analyze"
        )
        assert resolved.path.exists(), (
            "the handle resolved to a path with no bytes behind it"
        )

    def test_the_upload_response_never_discloses_a_filesystem_path(self, monkeypatch) -> None:
        """The envelope must not name a server-side location.

        `to_response()` omits `path` deliberately (`gateway/assets.py:219-225`).
        A deployment is "suitable" only if enabling uploads does not turn the
        endpoint into a path oracle: the client's correct next move is to pass
        the handle to `/v1/analyze`, never to construct a directory itself.
        """
        import os

        pytest.importorskip("fastapi", reason="the web stack is unavailable here")
        from app import space_app as space_app_module

        scratch = _scratch_asset_dir("no-disclosure")
        monkeypatch.setenv("SATQUERY_ASSET_ENABLED", "1")
        monkeypatch.setenv("SATQUERY_ASSET_DIR", str(scratch))
        monkeypatch.setattr(space_app_module, "_ASSET_STORE", None)
        app = space_app_module.build_space_app()

        _status, body = self._post_asset(app, content_type="application/octet-stream")

        assert "path" not in body, (
            f"the upload response carried a 'path' key ({body.get('path')!r}); "
            f"that discloses a server-side filesystem location and invites a "
            f"client to bypass the handle"
        )
        rendered = repr(body)
        assert str(scratch) not in rendered, (
            f"the response body contains the configured SATQUERY_ASSET_DIR "
            f"({scratch}); the response must not reveal where bytes are stored"
        )
        assert os.sep + "assets" not in rendered.replace("\\", os.sep), (
            f"the response body looks like it contains a filesystem path: "
            f"{rendered}"
        )

    def test_a_disabled_upload_is_refused_and_names_the_switch(self, monkeypatch) -> None:
        """Disabled: refused, with the reason and the env var that fixes it.

        The refusal is asserted on its CONTENT, not just its status. A 4xx with
        a generic body would leave an operator with no way to learn that the
        endpoint exists and is simply switched off -- which is the difference
        between a configured deployment and a broken one.
        """
        pytest.importorskip("fastapi", reason="the web stack is unavailable here")
        from app import space_app as space_app_module

        # Disabled means the conjunction fails. Clear BOTH, so this cannot pass
        # merely because the sandbox happens to have no directory configured.
        monkeypatch.delenv("SATQUERY_ASSET_ENABLED", raising=False)
        monkeypatch.delenv("SATQUERY_ASSET_DIR", raising=False)
        monkeypatch.setattr(space_app_module, "_ASSET_STORE", None)
        app = space_app_module.build_space_app()

        status, body = self._post_asset(app, content_type="application/octet-stream")

        assert status in (409, 503), (
            f"a disabled deployment answered {status} for an upload; it must "
            f"refuse rather than accept bytes it has not been told to keep. "
            f"Body was {body!r}"
        )
        assert status != 201, "a disabled deployment accepted an upload"
        rendered = repr(body)
        assert "SATQUERY_ASSET_ENABLED" in rendered, (
            f"the refusal does not name the switch that enables uploads; body "
            f"was {body!r}. An operator cannot act on 'not available' alone."
        )
        assert "asset" in rendered.lower(), (
            f"the refusal does not mention what was refused: {body!r}"
        )

    def test_enablement_requires_both_variables(self, monkeypatch) -> None:
        """The conjunction, asserted one variable at a time.

        `_asset_store_available()` requires BOTH. If a single variable sufficed,
        a deployment that set only the flag would start writing uploaded bytes
        to `tempfile.gettempdir() / "satquery-assets"` -- an unconfigured
        location nobody chose. That is the failure this pins.
        """
        pytest.importorskip("fastapi", reason="the web stack is unavailable here")
        from app import space_app as space_app_module

        # Flag only: still refuses.
        monkeypatch.setenv("SATQUERY_ASSET_ENABLED", "1")
        monkeypatch.delenv("SATQUERY_ASSET_DIR", raising=False)
        assert space_app_module._asset_store_available() is False, (
            "the flag alone enabled uploads; the directory is the part that "
            "says WHERE, and without it bytes land in an unchosen temp path"
        )

        # Directory only: still refuses.
        monkeypatch.delenv("SATQUERY_ASSET_ENABLED", raising=False)
        monkeypatch.setenv("SATQUERY_ASSET_DIR", str(_scratch_asset_dir("dir-only")))
        assert space_app_module._asset_store_available() is False, (
            "the directory alone enabled uploads; enabling a disk-writing "
            "endpoint must be an explicit operator action, not a side effect "
            "of a directory existing"
        )

        # Both: enabled.
        monkeypatch.setenv("SATQUERY_ASSET_ENABLED", "1")
        assert space_app_module._asset_store_available() is True

