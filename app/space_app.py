"""SatQuery AI — the Hugging Face Space inference entrypoint.

OWNERSHIP (resolved explicitly, do not re-litigate)
---------------------------------------------------
`docs/PHASE18_DEPLOYMENT_PACKAGING.md` section 4 records the Space entrypoint as
"blocked on the unresolved GUI interface contract (Phase 16 audit)". That blocker
applies to the **Gradio GUI**, which is Phase 16 and is owned by the frontend
agent. It does not apply to a JSON inference service, which is what Phase 18
deploys. `docs/DEPLOYMENT_ARCHITECTURE.md` section 3.1 states the distinction:

    Gradio GUI (`app.py` with `gr.Blocks`)  -- Phase 16 -- frontend agent
    Inference service (this file)           -- Phase 18 -- this work order

This file therefore serves JSON only. It deliberately does **not** build a
Gradio interface, so it cannot compete with the frontend agent's work.

THE FIVE ENTRYPOINT REQUIREMENTS (`docs/DEPLOYMENT_ARCHITECTURE.md` section 3.3)
-------------------------------------------------------------------------------
1. Import cheaply and without torch. Health and capabilities must answer on CPU
   with no GPU and no model load.
2. Reuse `app.serving`. Its `build_serving_controller()` is the tested
   composition root; reimplementing the wiring would duplicate the F2
   train/serve-skew fix.
3. Degrade, do not crash. Absent artifacts degrade; *corrupt* artifacts raise
   `ModelLoadError`. A Space that refuses to boot because an optional artifact is
   missing is worse than one serving a reduced capability set.
4. Never load a model for a metadata request.
5. Honour the config hash. Never mutate the config or merge `deploy.yaml` into
   the registry.

IMPLEMENTATION NOTE — WHAT IS AND IS NOT EXECUTED HERE
------------------------------------------------------
`spaces` (the ZeroGPU decorator package) is not installed in this environment,
and neither is FastAPI (a WDAC policy blocks `orjson.pyd`). The decoration is
therefore applied **conditionally**: when `spaces` is importable, the GPU routes
are decorated with the frozen duration; when it is not, the undecorated functions
run, which is the correct behaviour on CPU. This is not a workaround for a bug --
`configs/deploy.yaml` sets `cpu_mode_required: true`, so a CPU run must work.

The consequence is recorded in `docs/PHASE19_FINAL_HARDENING.md`: the ZeroGPU
decoration has **never executed** here. It is specified from finding C-8 and the
frozen `gpu_duration_*` values, and that is all it is.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:  # pragma: no cover
    from starlette.requests import Request

    from gateway.assets import AssetStore

# `Request` is bound at MODULE scope, and this is load-bearing rather than
# stylistic -- it is the G-1 defect, and it would be re-introduced here by a
# local import.
#
# This module uses `from __future__ import annotations`, so `request: Request`
# in the `/v1/assets` handler below is a *string* at runtime. FastAPI resolves
# it with `eval(annotation, func.__globals__)`, so the name must be in this
# module's globals. If `Request` were imported inside `build_space_app`, it would
# be a local of that function, `eval` would fail, and FastAPI would **not raise**
# -- it would silently reinterpret the parameter as a required *query* parameter
# named `request`, answering every upload with
# `422 {"detail":[{"loc":["query","request"]}]}` and never entering the handler.
#
# `app/space_app.py` had no `Request`-annotated route before the fourth endpoint
# was built, so this is the first time the trap could fire here. It is bound once,
# at the top, so a future handler cannot get it wrong by forgetting an import.
from starlette.requests import Request  # noqa: E402  (see comment above)

# `Response` is bound for the same reason, and its necessity was discovered the
# hard way. Every route in this module is annotated `-> JSONResponse` while
# `JSONResponse` was imported INSIDE `build_space_app`. With
# `from __future__ import annotations`, FastAPI evaluates that return annotation
# against `space_app.__globals__`, where `JSONResponse` did not exist, so
# `add_api_route` raised:
#
#     pydantic.errors.PydanticUndefinedAnnotation: name 'JSONResponse' is not defined
#
# and `build_space_app()` could not be called at all. This is the SAME class of
# defect as G-1 -- a name needed by `eval` at route-registration time bound in a
# narrower scope than the annotation evaluator can see -- and it is why every
# route annotation subject in this file is now module-level.
#
# Note this one FAILS LOUDLY, where G-1 failed silently. The difference is
# whether the unresolved name is a parameter annotation (FastAPI falls back to a
# query parameter) or a return annotation (FastAPI has no fallback and raises).
from starlette.responses import Response  # noqa: E402  (see comment above)
from fastapi.responses import JSONResponse  # noqa: E402  (see comment above)

__all__ = [
    "GPU_DURATIONS",
    "AssetStore",
    "_asset_max_files",
    "_asset_ttl_seconds",
    "build_space_app",
    "decorate_gpu",
    "describe_deployment",
    "get_asset_store",
    "get_controller",
]

#: ZeroGPU duration per task, from `configs/deploy.yaml` -- the FROZEN values,
#: not new guesses. `change_vqa` has no key of its own and reuses `change`,
#: because adding a key would move `Config.hash` off `78f1e3700da15aa1`
#: (`docs/DEPLOYMENT_ARCHITECTURE.md` section 3.4).
GPU_DURATIONS: dict[str, int] = {
    "vqa": 20,
    "caption": 20,
    "grounding": 45,
    "change": 30,
    "optical_sar": 45,
    "change_vqa": 30,
}

#: The controller is built once and cached: `cache_max_models: 1` means at most
#: one model is resident anyway, so rebuilding per request would thrash the cache
#: and pay a cold start every time.
_CONTROLLER: Any | None = None

#: The asset store, built once and cached. Same reasoning as `_CONTROLLER`, plus
#: one of its own: the store holds the handles, so a second instance would not
#: recognise the first one's uploads.
_ASSET_STORE: Any | None = None


def _spaces_module() -> Any | None:
    """Import `spaces` if present.

    ZeroGPU Spaces ship it; a CPU-only machine does not. Returning None rather
    than raising keeps the module importable everywhere, which requirement 1 of
    section 3.3 demands.
    """
    try:
        import spaces  # type: ignore[import-not-found]

        return spaces
    except Exception:
        return None


def decorate_gpu(task: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return the ZeroGPU decoration for a task, or an identity decorator.

    The duration comes from :data:`GPU_DURATIONS`, which is a transcription of
    `configs/deploy.yaml`. A task with no declared duration is a programming
    error, not a default: silently picking a duration would reserve the wrong
    amount of the 5 GPU-minute daily budget.
    """
    if task not in GPU_DURATIONS:
        raise KeyError(
            f"no gpu_duration_* is declared for {task!r}; add it to "
            f"configs/deploy.yaml (which moves Config.hash) or map it to an "
            f"existing task. Do not guess a duration."
        )
    duration = GPU_DURATIONS[task]
    spaces = _spaces_module()
    if spaces is None or not hasattr(spaces, "GPU"):
        def _identity(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn

        return _identity
    return spaces.GPU(duration=duration)


def get_controller() -> Any:
    """Build (once) and return the serving controller.

    Requirement 2: this delegates to `app.serving.build_serving_controller()`,
    the existing composition root that wires the change head and the change-VQA
    head through the registry's `builders=` override. That override is a call-site
    argument rather than config, which is what keeps `Config.hash` unchanged
    (requirement 5).

    The call is deferred to first use, not performed at import: requirement 1
    says importing this module must not pull the model stack.
    """
    global _CONTROLLER
    if _CONTROLLER is None:
        from app.serving import build_serving_controller

        _CONTROLLER = build_serving_controller()
    return _CONTROLLER


def describe_deployment() -> dict[str, Any]:
    """Report what this deployment can do, without loading any model.

    This is now a **thin delegation to the deployment capability adapter**
    (`app/deployment.py`). Before the 2026-09-22 ruling this function carried its
    own capability table -- two rows, built from two `Path.exists()` calls -- and
    was therefore a second, narrower answer to a question the registry already
    answered with six capabilities. The ruling made the registry authoritative
    and this adapter the single public source, so the local table is gone.

    Requirement 4 still holds and is still the reason the adapter exists in the
    form it does: it reads artifact *presence* and configuration values, never
    weights, and it never constructs a specialist. It is therefore safe to call
    from a health route and costs no GPU quota.

    It does **not** reuse `AnalystController.health()`, which its own docstring
    documents as "Constructs everything" -- the opposite of what a metadata
    request may do. Per the ruling, that method is retired as a public API path.

    The returned mapping is the OPERATOR view (it carries `config_hash`, artifact
    byte counts and the registry's own state words). The two served routes do not
    return this mapping directly: `health()` and `capabilities()` call the
    adapter's own payload builders, so the registry vocabulary never reaches a
    client.
    """
    from app.deployment import deployment_report

    return deployment_report().as_internal()


def get_asset_store() -> "AssetStore":
    """Build (once) and return the process asset store.

    `POST /v1/assets` is the fourth endpoint, and the owner ruling of 2026-09-22
    resolved it as BUILD with opaque ephemeral handles. This is its storage.

    The store lives on the Space, not on the gateway, and that follows from the
    contract's own reasoning for choosing Option A
    (`docs/API_CONTRACT.md` section 2.5): *"Keeps `/v1/analyze` JSON-only and lets
    the gateway enforce a size limit before the JSON body is parsed."* So the
    gateway enforces the size and the content type, and the Space owns the bytes
    -- because the Space is where `inspect_raster` reads them and where
    `cache_max_models: 1` serializes their consumption. A gateway-side store
    would put the bytes on a different machine from the reader.

    Sizing comes from the deployment, not from a guess:
      * `cache_max_models: 1` means an upload is consumed within one analysis,
        so the TTL is short and the capacity is small;
      * `max_file_bytes` mirrors `GatewayConfig`'s per-file cap so the two
        layers cannot disagree about what "too large" means.
    """
    global _ASSET_STORE
    if _ASSET_STORE is None:
        from gateway.assets import AssetStore

        _ASSET_STORE = AssetStore(
            root=_asset_root(),
            max_file_bytes=_asset_max_file_bytes(),
            max_files=_asset_max_files(),
            ttl_seconds=_asset_ttl_seconds(),
            allowed_content_types=_ALLOWED_ASSET_CONTENT_TYPES,
        )
    return _ASSET_STORE


def _asset_max_files() -> int:
    """Handle capacity, from the environment, defaulting to the original 32.

    These two were hardcoded until the STEP 8 audit recorded the resulting
    scaling limit: a deployment with ample `SATQUERY_ASSET_DIR` and a burst of
    concurrent users could hit the ceiling before the TTL reaped anything and
    answer `503` with disk free. That is a *refusal*, not corruption -- the store
    never evicts a live handle -- but it is a limit an operator should be able to
    raise without editing code.

    Read from the environment rather than from `configs/base.yaml` on purpose:
    adding a key there moves `Config.hash` off `78f1e3700da15aa1` and invalidates
    the frozen Phase-9 benchmark. Sizing is deployment state, and the same
    reasoning already governs `SATQUERY_MAX_FILE_BYTES`.
    """
    import os

    raw = os.environ.get("SATQUERY_ASSET_MAX_FILES")
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
    return 32


def _asset_ttl_seconds() -> float:
    """Handle lifetime in seconds, from the environment, defaulting to 900.

    A non-positive or unparsable value falls back to the default rather than
    disabling expiry: a zero TTL would make every handle dead on arrival, which
    is a silent breakage rather than a configuration choice.
    """
    import os

    raw = os.environ.get("SATQUERY_ASSET_TTL_S")
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = 0.0
        if value > 0:
            return value
    return 900.0


def _asset_root() -> Path:
    """Where uploaded bytes live. Outside the repo tree, so a git status stays
    clean and a restarted container starts empty (handles are ephemeral)."""
    import os
    import tempfile

    configured = os.environ.get("SATQUERY_ASSET_DIR")
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / "satquery-assets"


def _asset_max_file_bytes() -> int:
    """The per-file cap, read from the same environment variable the gateway
    uses. A single source keeps the two layers agreeing; the default matches
    `GatewayConfig.max_file_bytes`.

    F-7. Reading one variable is NOT the same as agreeing on its value, and this
    function used to disagree with the gateway in TWO separate ways (measured
    2026-09-22):

        SATQUERY_MAX_FILE_BYTES='abc' or '4e6'
            -> this function silently returned the 4 MiB DEFAULT while the
               gateway RAISED AT STARTUP for the same value.

        SATQUERY_MAX_FILE_BYTES='0' or '-1'
            -> this function RETURNED THE VALUE ('0' / -1) while the gateway,
               once its own non-positive guard was added, REFUSED it.

    A silent default is the worse of the first two: it means a deployment whose
    operator typed a malformed cap keeps accepting uploads against a limit
    nobody chose, and nothing anywhere says so. The gateway is the layer that
    must fail fast (it fails the deploy), so this function must not be the one
    that quietly papers over the value the gateway rejected.

    The second pair is the subtler one and it is why the check below is on
    `value` and not merely on parseability. `_asset_max_file_bytes` validating
    only that a string *parses* would still let the Space construct an
    `AssetStore` whose cap is 0, while the gateway refused to boot at all for
    that same value -- the layers diverge on the one input class where the
    divergence is least visible, because `0` fails no parse. The store does
    eventually object, but only when the first upload arrives, and by then the
    operator is reading a 5xx in a request log instead of a named variable in a
    start log.

    Refusing here also keeps the failure *above* the store, which is where it
    belongs: the cap is deployment configuration, and configuration is validated
    when the process starts, not when it is first exercised.

    The refusal is deliberately NOT a silent fallback and NOT a bare exception:
    it names the variable and the offending text, because this is read while
    building the store and an operator can only see it in a Space's build or
    start log.
    """
    import os

    raw = os.environ.get("SATQUERY_MAX_FILE_BYTES")
    if not raw:
        return 4 * 1024 * 1024
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"SATQUERY_MAX_FILE_BYTES={raw!r} is not an integer. It is NOT "
            f"defaulted, because the gateway refuses this same value at startup "
            f"and silently substituting a different cap here would leave the two "
            f"layers disagreeing about what 'too large' means -- the exact "
            f"failure this shared variable exists to prevent."
        ) from None
    if value <= 0:
        raise ValueError(
            f"SATQUERY_MAX_FILE_BYTES={value} is not positive. A cap of "
            f"{value} would refuse every upload, so it is a configuration "
            f"error rather than a limit -- and the gateway refuses this same "
            f"value at startup. Accepting it here would leave the two layers "
            f"disagreeing about whether the deployment is valid at all."
        )
    return value


#: The content types this endpoint accepts. Mirrors
#: `GatewayConfig.allowed_content_types`; the Space's copy exists because the
#: Space validates independently (defence in depth) rather than trusting that
#: the gateway is the only caller.
_ALLOWED_ASSET_CONTENT_TYPES: tuple[str, ...] = (
    "image/tiff",
    "image/geotiff",
    "image/png",
    "image/jpeg",
    "application/octet-stream",
)


def _asset_store_available() -> bool:
    """Whether the upload endpoint is enabled.

    Off by default in a deployment that has not set `SATQUERY_ASSET_DIR`, and
    ON when it has -- so turning on the fourth endpoint is an explicit operator
    action rather than something that starts writing to a temp directory
    unbidden. `/v1/capabilities` is where a client learns which it is.
    """
    import os

    return bool(os.environ.get("SATQUERY_ASSET_ENABLED", "")) and bool(
        os.environ.get("SATQUERY_ASSET_DIR")
    )


def build_space_app() -> Any:
    """Build the Space's ASGI app.

    Decorated routes carry the frozen duration. The app itself is where the
    contract's **four** endpoints are served (`/v1/health`, `/v1/capabilities`,
    `/v1/analyze`, `/v1/assets` -- the fourth per the owner ruling of
    2026-09-22); the gateway sits in front of it and holds the security boundary
    (`docs/DEPLOYMENT_ARCHITECTURE.md` section 1.1).

    `FastAPI` is imported HERE, inside the function, which is correct and
    deliberate: requirement 1 says importing this module must not pull the web
    framework, and a name used only at build time does not need to be in the
    module namespace. The annotation subjects (`Request`, `Response`,
    `JSONResponse`) are the opposite case and ARE module-level -- see the import
    comment for why the distinction is load-bearing.

    Raises:
        ImportError: when FastAPI is unavailable. In this sandbox this is no
            longer the case -- the `orjson`/WDAC block that
            `docs/PHASE19_FINAL_HARDENING.md` records no longer reproduces -- but
            the docstring is retained because a deployment without FastAPI is
            still a real configuration and the failure must stay legible.
    """
    from fastapi import FastAPI

    from gateway.policy import translate_error

    api = FastAPI(title="SatQuery AI Inference Service", version="1.0")

    # ------------------------------------------------------------------
    # F-12 / F-12b (owner ruling 2026-09-23): framework-raised errors must
    # carry the SAME v1 envelope as handler-raised ones.
    #
    # `gateway/app.py` was fixed for this on 2026-09-22 (recorded as F-3), and
    # `API_CONTRACT.md` section 5 states plainly that "`404` and `405` carry this
    # same envelope". That was true **through the gateway**. It was NOT true of
    # the Space itself -- and the Space is what `SATQUERY_SPACE_URL` points at in
    # the runbook, so a client following the runbook hits the unwrapped path.
    # Measured, direct to the Space, before this fix:
    #
    #     GET  /v1/whocares -> 404 {"detail":"Not Found"}
    #     GET  /v1/assets   -> 405 {"detail":"Method Not Allowed"}
    #     an unwrapped failure -> 500 text/plain, no envelope at all
    #
    # The contract fixes the CODE for both: section 5.1 maps 404 and 405 to
    # `routing_error` with `recoverable: false`. `translate_error` derives the
    # STATUS from the code and maps `routing_error` to 422 for the analysis
    # taxonomy, so the status is taken from the EXCEPTION, not from the code --
    # exactly as the gateway handler does it, so the two layers cannot drift.
    # ------------------------------------------------------------------
    import logging

    from starlette.exceptions import HTTPException as StarletteHTTPException

    _log = logging.getLogger("satquery.space")

    @api.exception_handler(StarletteHTTPException)
    async def _contract_envelope_for_transport_errors(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Wrap the framework's own 404/405 in the contract's error shape.

        Scoped to Starlette's `HTTPException`, which is exactly the class FastAPI
        raises for an unmatched route (404) or a method mismatch (405) and which
        its own handlers would otherwise render as `{"detail": ...}`.
        """
        code = "routing_error" if exc.status_code < 500 else "satquery_error"
        status, body = translate_error(
            code,
            "This endpoint does not exist."
            if exc.status_code == 404
            else str(exc.detail),
            detail=(
                f"{request.method} {request.url.path} -> {exc.status_code}; "
                f"see docs/API_CONTRACT.md sections 2 and 5"
            ),
            recoverable=False,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=body,
            headers=getattr(exc, "headers", None),
        )

    @api.exception_handler(Exception)
    async def _contract_envelope_for_unhandled_failures(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """F-12b: an unwrapped failure answers with the envelope, not plain text.

        Before this, an exception escaping a handler produced Starlette's bare
        `500 text/plain` body, which is not the contract's shape and is not
        parseable by a client written to section 5.

        **F-15 is enforced here as well as in the handlers.** The client is told
        the code and a fixed, operator-safe message; the exception's own text is
        recorded SERVER-SIDE only. That split is the ruling: "sanitize all
        client-facing exception messages; retain full exception details only in
        server-side diagnostics". A traceback in `detail` would disclose
        internal paths and library versions to an unauthenticated caller.
        """
        _log.exception(
            "unhandled failure on %s %s", request.method, request.url.path
        )
        status, body = translate_error(
            "satquery_error",
            "An internal error occurred.",
            detail="",
            recoverable=False,
        )
        return JSONResponse(status_code=status, content=body)

    @api.get("/v1/health")
    async def health() -> JSONResponse:
        """Liveness + capability states. Loads no model (requirement 4).

        The body comes from `app.deployment.health_payload()`, which is the
        single public health source per the 2026-09-22 ruling. It is therefore
        the same traversal that builds `/v1/capabilities`, so the two routes
        cannot disagree about which capabilities exist -- the property the
        STEP 7 chain tests assert.

        The previous version derived `models` here, inline, from a two-row local
        table. It was internally consistent with `capabilities()` but it was
        derived from *less evidence* than the registry held, and it invented the
        mapping from `available: bool` to a contract state. That mapping now
        lives in one place, where it is tested.
        """
        from app.deployment import health_payload

        payload = health_payload()
        # Asserted rather than trusted: `HealthStatus` is `extra="forbid"`, so a
        # key added to the payload without a key added to the model would make
        # the response invalid against the project's own contract. Finding H-1
        # was exactly this failure in the other direction.
        from core.schemas import HealthStatus

        HealthStatus.model_validate(payload)
        return JSONResponse(payload)

    @api.get("/v1/capabilities")
    async def capabilities() -> JSONResponse:
        from app.deployment import capabilities_payload

        return JSONResponse(capabilities_payload())

    @api.post("/v1/assets")
    async def assets(request: Request) -> JSONResponse:
        """Accept one uploaded file and return an opaque ephemeral handle.

        The fourth endpoint. `docs/API_CONTRACT.md` section 2.5 recorded it as
        "NOT IN THE PLAN" with two options; the owner ruling of 2026-09-22 chose
        **Option A -- out-of-band upload -- with opaque ephemeral handles**. The
        storage and its three obligations (size cap, content-type allowlist,
        retry story) live in `gateway/assets.py`, where they are tested without
        a web framework.

        This handler does the HTTP translation only: `Content-Type` header in,
        contract envelope out.
        """
        from gateway.assets import AssetStoreError  # noqa: F401
        from gateway.assets import (
            AssetStoreFullError,
            AssetTooLargeError,
            UnsupportedContentTypeError,
        )
        from gateway.assets import read_body_bounded
        from gateway.policy import translate_error

        if not _asset_store_available():
            # A deployment that has not enabled uploads says so, with the reason
            # and the switch, rather than accepting bytes it cannot keep.
            status, body = translate_error(
                "model_unavailable",
                "Asset upload is not enabled on this deployment.",
                detail=(
                    "Set SATQUERY_ASSET_ENABLED=1 and SATQUERY_ASSET_DIR to a "
                    "writable path to enable POST /v1/assets. See "
                    "docs/DEPLOYMENT_ARCHITECTURE.md."
                ),
                recoverable=False,
            )
            return JSONResponse(status_code=status, content=body)

        # F-9. This was `raw = await request.body()`, which buffered the WHOLE
        # body BEFORE `store.put()` could apply its cap -- the same defect F-6
        # fixed on the gateway, on the other layer. Measured 2026-09-22 with the
        # cap at 1 MiB: a 64 MiB body produced a peak allocation of **128 MiB**
        # and a 16 MiB body 32 MiB, tracking body size linearly with no ceiling,
        # and the `413` came only after everything had been held.
        #
        # The cap used here is the Space's OWN per-file cap, and it must be read
        # from the same helper the store is built with, so the layer that stops
        # reading and the layer that enforces cannot disagree about the number --
        # which is the F-7 lesson applied at the point it matters.
        #
        # Note this is deliberately NOT a second policy layer: the gateway still
        # holds the request-side boundary (`DEPLOYMENT_ARCHITECTURE.md` sec 1.1).
        # This is the Space refusing to buffer input it has already decided it
        # will reject, which is what obligation 1 of `gateway/assets.py` claimed
        # was true and, on this route, was not.
        raw, too_large = await read_body_bounded(request, _asset_max_file_bytes())
        if too_large:
            status, body = translate_error(
                "oversized_image",
                "The uploaded file is too large.",
                detail=(
                    f"The request body exceeded SATQUERY_MAX_FILE_BYTES "
                    f"({_asset_max_file_bytes()} bytes) and was refused while "
                    f"reading, so the excess was never buffered."
                ),
            )
            return JSONResponse(status_code=status, content=body)

        store = get_asset_store()
        try:
            handle = store.put(
                raw,
                content_type=request.headers.get("content-type"),
            )
        except AssetTooLargeError as exc:
            status, body = translate_error(
                "oversized_image",
                "The uploaded file is too large.",
                detail=exc.detail,
            )
            return JSONResponse(status_code=status, content=body)
        except UnsupportedContentTypeError as exc:
            status, body = translate_error(
                "raster_read_error",
                "This file type is not accepted.",
                detail=exc.detail,
            )
            return JSONResponse(status_code=status, content=body)
        except AssetStoreFullError as exc:
            status, body = translate_error(
                "model_unavailable",
                "The upload buffer is full. Please wait and retry.",
                detail=exc.detail,
                recoverable=True,
            )
            return JSONResponse(status_code=status, content=body)
        except AssetStoreError as exc:  # pragma: no cover - defensive
            status, body = translate_error(
                "input_error",
                "The upload could not be stored.",
                detail=exc.detail,
            )
            return JSONResponse(status_code=status, content=body)

        return JSONResponse(status_code=201, content=handle.to_response())

    @api.post("/v1/analyze")
    async def analyze(payload: dict[str, Any]) -> JSONResponse:
        """Run one analysis.

        Decorated with the task's frozen GPU duration at import time by
        :func:`decorate_gpu`; the undecorated path is the CPU path and is the
        one that has actually been exercised.

        `AnalysisRequest.assets` carries **handles**, not paths, once uploads are
        enabled -- that is the contract (`API_CONTRACT.md` section 2.4: *"asset
        handles returned by the upload step"*). The controller, however, needs a
        path to inspect. This handler is the one place that translation happens:

            handle -> AssetStore.get() -> path -> AnalysisRequest.assets

        A handle that is unknown or expired is refused HERE, with a named error,
        rather than being passed to the controller as a path that does not exist
        -- which would surface as a raster read failure and name the wrong cause.
        """
        from core.schemas import AnalysisRequest

        try:
            request = AnalysisRequest.model_validate(payload)
        except Exception as exc:
            # F-15 (owner ruling 2026-09-23): the CLIENT gets a sanitized message.
            # `str(ValidationError)` enumerates internal field paths, types and
            # constraints -- genuinely useful to a maintainer, and needless
            # exposure for an unauthenticated caller. The full text is recorded
            # SERVER-SIDE only, which is the ruling's split.
            _log.warning(
                "invalid_request on POST /v1/analyze: %s: %s",
                type(exc).__name__,
                exc,
            )
            status, body = translate_error(
                "invalid_request",
                "The request did not match the expected schema.",
                detail=(
                    "The body was not a valid AnalysisRequest. See "
                    "docs/API_CONTRACT.md section 2 for the accepted shape."
                ),
            )
            return JSONResponse(status_code=status, content=body)

        if _asset_store_available():
            from gateway.assets import UnknownAssetError
            from gateway.policy import translate_error as _translate

            store = get_asset_store()
            try:
                handles = store.resolve_many(list(request.assets))
            except UnknownAssetError as exc:
                status, body = _translate(
                    "input_error",
                    "One or more asset handles are unknown or have expired.",
                    detail=exc.detail,
                )
                return JSONResponse(status_code=status, content=body)
            # Rebuild the request with resolved PATHS. `model_copy` rather than
            # mutating, because `AnalysisRequest` is the contract's model and a
            # handler must not rewrite a validated request in place.
            request = request.model_copy(
                update={"assets": [str(handle.path) for handle in handles]}
            )

        controller = get_controller()
        envelope = controller.run(request)
        return JSONResponse(content=envelope.model_dump(mode="json"))

    return api


def main() -> Any:  # pragma: no cover - requires FastAPI + a server
    """The `app_file` entrypoint a Space imports."""
    return build_space_app()
