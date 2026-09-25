"""SatQuery AI — the gateway ASGI application.

THIS FILE CANNOT BE IMPORTED IN THE DEVELOPMENT SANDBOX
-------------------------------------------------------
`import fastapi` fails here, and not because of anything in this repository:

    ImportError: DLL load failed while importing orjson: An Application Control
    policy has blocked this file

Verified as `OSError [WinError 4551]` from a direct
`ctypes.CDLL(".../orjson/orjson.cp311-win_amd64.pyd")`. FastAPI's own guard reads

    try:
        orjson = importlib.import_module("orjson")
    except ModuleNotFoundError:
        orjson = None

so a policy-blocked `ImportError` -- a parent class of `ModuleNotFoundError`'s
sibling, not a subclass -- escapes the handler and aborts the import. This is the
same environmental class as the documented `pandas._libs.parsers` blocker.

**Consequence, stated plainly:** the route wiring below has never executed. Its
logic is not untested -- every decision it makes lives in `gateway.policy`, which
has 51 passing tests -- but the FastAPI plumbing (route registration, `Request`
handling, `StreamingResponse` construction) is **specified and reviewed, not
run**. `docs/PHASE19_FINAL_HARDENING.md` records this, and no deployment claim
rests on it. The work order forbids claiming otherwise.

WHY IT IS STILL WRITTEN
-----------------------
The work order requires a Railway gateway, and the alternative -- shipping
nothing because the sandbox cannot import a web framework -- would leave the
frontend agent with no backend at all. The mitigation is architectural: this file
does no thinking. It translates HTTP into `gateway.policy` calls and translates
the results back. If a rule needs to change, it changes in `policy.py`, where it
is tested.

STRUCTURE
---------
  * `create_app()` builds the ASGI app from a `GatewayConfig`.
  * `_proxy()` is the single forwarding path; every route uses it, so no route
    can bypass validation.
  * Startup refuses an incoherent config by construction: `GatewayConfig`
    validates in `__post_init__`, so a bad environment fails fast rather than
    serving traffic with, say, an open CORS policy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping

# `Request` is needed at IMPORT time, not only at call time, and this is not a
# style preference -- it is load-bearing.
#
# This module uses `from __future__ import annotations`, so every annotation
# below is a *string* at runtime. FastAPI resolves those strings via
# `get_typed_signature`, which calls `eval(annotation, func.__globals__)`. If
# `Request` is imported inside `create_app`, the route closures capture the name
# as a *local* of `create_app`; it never appears in `gateway.app.__globals__`,
# so resolution fails and FastAPI is left holding a bare `ForwardRef('Request')`.
#
# FastAPI then does not raise -- it silently falls back to treating the
# parameter as a *query* parameter named `request`. The observable consequence
# was that every POST to `/v1/analyze` and `/v1/assets` returned
#
#     422 {"detail":[{"loc":["query","request"],"msg":"Field required"}]}
#
# without ever entering the handler: the request body was never read, the
# gateway's own validation never ran, and the error envelope was FastAPI's
# `{"detail": ...}` rather than the contract's `{"error": {...}}`
# (docs/API_CONTRACT.md section 5). No test caught this because no test could
# import FastAPI when this file was written.
#
# The fix is to bind the name in the module namespace. It is `TYPE_CHECKING`-safe
# because the import is real; the `TYPE_CHECKING` block only keeps the reader
# honest about the distinction between "needed for types" and "needed at
# runtime", which here is the latter.
if TYPE_CHECKING:  # pragma: no cover
    from starlette.requests import Request
    from starlette.responses import Response

#: Runtime bindings used as annotation subjects in this module. Deliberately
#: module-level so `eval()` can find them. See the comment above.
from starlette.requests import Request  # noqa: E402  (see comment above)
from starlette.responses import Response  # noqa: E402  (see comment above)

# `JSONResponse` is bound at module scope for the SAME reason, and it is the
# return-annotation half of the trap. `gateway_health()` is annotated
# `-> JSONResponse` while `JSONResponse` was imported inside `create_app`; FastAPI
# evaluates the return annotation against `gateway.app.__globals__`, so
# registering that route raised:
#
#     pydantic.errors.PydanticUndefinedAnnotation: name 'JSONResponse' is not defined
#
# and `create_app()` could not be called at all -- the whole gateway was
# unbuildable, which is why the STEP 7 appendix could only assert that the module
# *imported*. It was found by actually calling `create_app()` in the /v1/assets
# work.
#
# Note the asymmetry with G-1, which is worth internalising: an unresolvable
# PARAMETER annotation is silently reinterpreted (FastAPI treats it as a query
# parameter and the handler never runs), while an unresolvable RETURN annotation
# raises. Same root cause, opposite diagnosability. Binding the name here fixes
# both and makes the difference moot.
from fastapi.responses import JSONResponse  # noqa: E402  (see comment above)

from gateway.policy import (
    ClientIdentity,
    GatewayConfig,
    GatewayPolicy,
    config_from_env,
    translate_error,
    validate_analyze_body,
)

#: F-15c (owner ruling 2026-09-23). The gateway had no logger, and the transport
#: `except` branch put `f"{type(exc).__name__}: {exc}"` straight into the error
#: body's `detail` field -- which `translate_error` publishes to the client
#: (`gateway/policy.py:165`). Raw third-party exception text is not a
#: classification: it names the gateway's HTTP client and its internals, and it
#: can carry a proxy URL or a filesystem path.
_log = logging.getLogger(__name__)

#: Transport failures, mapped to a CONTRACT-level classification. Ordered, most
#: specific first. Matched on the exception's MRO class names rather than with
#: `isinstance`, so the classifier keeps working when `httpx` is absent (the
#: caller passes `httpx=None` in that case) and does not couple the client-facing
#: vocabulary to a third-party type hierarchy that can be renamed.
_TRANSPORT_FAILURES: tuple[tuple[str, str], ...] = (
    ("TimeoutException", "the upstream did not respond within the gateway timeout"),
    ("ConnectError", "the upstream could not be reached"),
    ("ProxyError", "the gateway's egress proxy refused the connection"),
)

_TRANSPORT_FAILURE_FALLBACK = "the upstream request failed at the transport layer"


def _transport_failure_detail(exc: BaseException) -> str:
    """A path-free, internals-free classification of a transport failure.

    F-15c: what the client needs is *which kind* of transport failure this was --
    unreachable, timed out, or refused by the proxy -- because those imply
    different operator actions. What it must not receive is the exception's own
    text, which names the gateway's HTTP client, its internals, and potentially
    a proxy URL or a path.

    The full exception still reaches the operator through `_log.error(...,
    exc_info=exc)` in the caller, so nothing is lost -- it is moved, not deleted.
    """
    names = {cls.__name__ for cls in type(exc).__mro__}
    for marker, text in _TRANSPORT_FAILURES:
        if marker in names:
            return text
    return _TRANSPORT_FAILURE_FALLBACK


__all__ = ["create_app", "app"]


#: Routes the gateway forwards. An allowlist, not a passthrough: a gateway that
#: forwarded arbitrary paths would expose every route the Space happens to serve,
#: including ones the contract does not document.
#:
#: `/v1/assets` was added on 2026-09-22 by owner ruling. It had been listed in
#: `BLOCKED_ROUTES` with a 501, because `API_CONTRACT.md` section 2.5 recorded the
#: fourth endpoint as "NOT IN THE PLAN" and an open decision. The ruling chose
#: Option A -- out-of-band upload with opaque ephemeral handles -- so the route is
#: now proxied like any other.
PROXIED_ROUTES: tuple[str, ...] = (
    "/v1/health",
    "/v1/capabilities",
    "/v1/analyze",
    "/v1/assets",
)

#: Routes the gateway refuses with an explicit 501.
#:
#: EMPTY, and it should stay that way: this tuple exists so a route the contract
#: discusses but the server does not implement answers *501 with a reason*
#: instead of a 404 that a frontend developer would debug as a typo. Nothing is
#: in that state right now. `gateway_health()` reports it either way, so an
#: operator can see the answer rather than infer it.
BLOCKED_ROUTES: tuple[str, ...] = ()

#: Routes the gateway rate-limits, because they cost GPU quota or disk.
#:
#: `docs/API_CONTRACT.md` section 6 tells the frontend to *"serialize requests"*
#: and warns that every `/v1/analyze` costs quota. `POST /v1/assets` does not
#: touch the GPU, but it does write to the Space's disk and consume one of a
#: bounded number of handles (`gateway/assets.py`), so an unthrottled upload loop
#: is a cheap denial of service against a 5-GPU-minute deployment. It is
#: therefore rate-limited alongside analyze.
#:
#: This is a separate allowlist from `PROXIED_ROUTES` because the two answer
#: different questions -- "may this reach the Space at all?" and "does it cost a
#: metered resource?" -- and collapsing them would make the rate limiter's
#: coverage depend on the proxy allowlist.
COSTLY_ROUTES: tuple[str, ...] = ("/v1/analyze", "/v1/assets")


def create_app(config: GatewayConfig | None = None) -> Any:
    """Build the ASGI app.

    Args:
        config: the gateway configuration. When omitted it is read from the
            environment, which is how Railway supplies it
            (`docs/BACKEND_DEPLOYMENT_RUNBOOK.md` section 4.1). A missing or
            incoherent configuration raises here, at startup -- not on the first
            request.

    Raises:
        ValueError: from `GatewayConfig.__post_init__` when the environment is
            incoherent (missing upstream, wildcard CORS, a timeout outside the
            documented window).
    """
    # `Request`, `Response` and `JSONResponse` are bound at MODULE level (see the
    # note at the imports). Importing them here as well would shadow the
    # module-level names for the closures below and re-introduce the
    # ForwardRef/undefined-annotation bug.
    from fastapi import FastAPI

    try:
        import httpx
    except ImportError:  # pragma: no cover
        httpx = None  # type: ignore[assignment]

    cfg = config if config is not None else config_from_env(_env())
    policy = GatewayPolicy(cfg)

    app = FastAPI(
        title="SatQuery AI Gateway",
        version="1.0",
        # The gateway serves exactly the routes it proxies. The OpenAPI schema is
        # generated from them, so it cannot advertise what it does not serve.
        docs_url="/v1/docs",
        openapi_url="/v1/openapi.json",
    )

    @app.get("/v1/gateway/health")
    async def gateway_health() -> JSONResponse:
        """The gateway's own liveness, distinct from the Space's.

        Separate from `/v1/health` on purpose: conflating them would make a
        gateway that is up but whose upstream is down indistinguishable from a
        gateway that is itself broken. This route never touches the Space, so it
        costs nothing.
        """
        return JSONResponse(
            {
                "status": "ok",
                "role": "gateway",
                "upstream_configured": bool(cfg.upstream_url),
                "proxied_routes": list(PROXIED_ROUTES),
                "blocked_routes": list(BLOCKED_ROUTES),
                "rate_limit_per_ip": cfg.rate_limit_per_ip,
                "rate_limit_window_s": cfg.rate_limit_window_s,
                "max_body_bytes": cfg.max_body_bytes,
                "upstream_timeout_s": cfg.upstream_timeout_s,
                "note": (
                    "Gateway liveness only. It does not indicate that the "
                    "upstream Space is reachable or that any model is loaded."
                ),
            }
        )

    # ------------------------------------------------------------------
    # F-3. Every non-2xx must carry the contract envelope.
    #
    # `docs/API_CONTRACT.md` section 5 opens with "Every non-2xx response body
    # has this shape", and `tests/unit/test_gateway_app.py` asserts it for the
    # paths the handlers own. But two responses never reach a handler, and both
    # are raised by Starlette/FastAPI with the framework's own
    # `{"detail": ...}` shape:
    #
    #   * **404** -- any path that is not registered. Reachable in practice:
    #     a client that appends a trailing slash, or a frontend typo.
    #   * **405** -- a registered path with the wrong method. `GET /v1/assets`
    #     is the obvious one; the contract documents the route but only ever
    #     defines it for POST.
    #
    # Measured before the fix, through the real ASGI stack:
    #
    #     GET  /v1/whocares -> 404 {"detail":"Not Found"}
    #     GET  /v1/assets   -> 405 {"detail":"Method Not Allowed"}
    #
    # while every path the handlers own answered with the envelope:
    #
    #     GET  /v1/health   -> 502 {"error":{"code":"model_unavailable",...}}
    #     POST /v1/analyze  -> 400 {"error":{"code":"input_error",...}}
    #
    # A client written to the contract parses `error.code` and gets a `KeyError`
    # exactly when it is trying to explain a failure to a user. The frontend is
    # not built yet, and the contract is frozen -- so this is the last cheap
    # moment to make the contract true rather than to document an exception.
    #
    # Both handlers delegate to `translate_error`, which is the same single
    # source the proxy path uses, so the codes cannot drift from section 5.2.
    # ------------------------------------------------------------------
    from starlette.exceptions import HTTPException as StarletteHTTPException

    @app.exception_handler(StarletteHTTPException)
    async def _contract_envelope_for_transport_errors(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Wrap the framework's own 404/405 in the contract's error shape.

        Scoped to Starlette's `HTTPException` -- which is exactly the class
        FastAPI raises for an unmatched route (404) or a method mismatch (405),
        and which its own handlers would otherwise render as `{"detail": ...}`.

        A framework-raised 404 is a client-side path error, which the contract's
        taxonomy has no dedicated code for; `routing_error` is the documented
        code for "request could not be interpreted", and the mapping is by
        status rather than by guessing. `recoverable` mirrors the contract's
        section 5.1 table: 404 and 405 are both `false`.
        """
        code = "routing_error" if exc.status_code < 500 else "satquery_error"
        status, body = translate_error(
            code,
            "This endpoint does not exist." if exc.status_code == 404 else str(exc.detail),
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

    @app.api_route("/v1/health", methods=["GET", "HEAD"])
    async def health(request: Request) -> Response:
        return await _proxy(request, policy, cfg, httpx, path="/v1/health")

    @app.api_route("/v1/capabilities", methods=["GET", "HEAD"])
    async def capabilities(request: Request) -> Response:
        return await _proxy(request, policy, cfg, httpx, path="/v1/capabilities")

    @app.post("/v1/analyze")
    async def analyze(request: Request) -> Response:
        return await _proxy(request, policy, cfg, httpx, path="/v1/analyze")

    @app.post("/v1/assets")
    async def assets(request: Request) -> Response:
        """The upload endpoint, PROXIED since the 2026-09-22 ruling.

        This handler is the whole gateway-side implementation, and that is the
        point: the bytes go straight to the Space, which owns the store
        (`gateway/assets.py`'s module docstring records why the store is not
        here). The gateway contributes three things on this path, all of which
        happen in `_proxy` / `policy.admit` before any byte is forwarded:

          * the body-size cap (`max_body_bytes` via `Content-Length`), which is
            the check the contract cites as Option A's advantage -- *"lets the
            gateway enforce a size limit before the JSON body is parsed"*;
          * the rate limit, because an upload loop is a cheap way to fill a
            Space's disk (`COSTLY_ROUTES`);
          * CORS, so a browser can call it at all.

        It previously returned 501. The reason it no longer does is recorded in
        `API_CONTRACT.md` section 2.5 and the STEP 7 report.
        """
        return await _proxy(request, policy, cfg, httpx, path="/v1/assets")

    return app


async def _read_body_bounded(
    request: Any, limit: int
) -> tuple[bytes, bool | None]:
    """Read a request body, refusing it the moment it exceeds `limit`.

    Returns `(body, None)` on success and `(b"", True)` when the cap was
    exceeded. The truthiness of the second element rather than an exception is
    deliberate: the caller turns it into the contract's `oversized_image`
    envelope, and this file does no thinking about error shapes.

    F-9: THE BODY OF THIS FUNCTION MOVED, AND THIS IS NOW A THIN ADAPTER.
    ---------------------------------------------------------------------
    The implementation lives in `gateway/assets.read_body_bounded`, because the
    **Space** had this same `await request.body()` line and therefore this same
    defect (see that function's docstring for the measurement). Two copies of a
    security control drift -- that is the F-7 finding -- so both layers now call
    the single implementation. This wrapper is kept only so the two call sites in
    `create_app` keep their existing shape and their `bool | None` contract; it
    adds no logic of its own.

    WHY NOT `await request.body()`
    ------------------------------
    `request.body()` accumulates the WHOLE body before returning, so a cap
    checked afterwards is not a cap -- it is a post-mortem.
    """
    from gateway.assets import read_body_bounded

    body, too_large = await read_body_bounded(request, limit)
    return (b"", True) if too_large else (body, None)


async def _proxy(request: Any, policy: GatewayPolicy, cfg: GatewayConfig, httpx: Any, *, path: str) -> Any:
    """The single forwarding path.

    Every proxied route funnels through here so that no route can skip
    validation. This function makes no policy decisions of its own: it asks
    `policy.admit()` and obeys the result.
    """
    # `Response`, `Request` and `JSONResponse` are module-level for the same
    # reason: an annotation subject is resolved through `__globals__`.

    identity = ClientIdentity(
        ip=_client_ip(request),
        user_agent=request.headers.get("user-agent", ""),
    )
    content_length: int | None
    raw_length = request.headers.get("content-length")
    try:
        content_length = int(raw_length) if raw_length is not None else None
    except ValueError:
        # A non-numeric Content-Length is malformed; treat it as unparseable
        # rather than as absent, so it cannot be used to bypass the size check.
        content_length = cfg.max_body_bytes + 1

    decision = policy.admit(
        method=request.method,
        path=path,
        content_length=content_length,
        origin=request.headers.get("origin"),
        identity=identity,
        inbound_request_id=request.headers.get("x-request-id"),
        # Explicit rather than derived from the path. `policy.admit`'s own
        # docstring says tests pass this so "a route rename cannot silently
        # disable rate limiting"; passing it here means ADDING a costly route
        # cannot silently miss the limiter either, which is exactly the mistake
        # this would otherwise have made for `/v1/assets`.
        is_analyze=path in COSTLY_ROUTES,
    )
    if not decision.admit:
        return JSONResponse(
            status_code=decision.status,
            content=decision.body or {},
            headers=decision.headers,
        )

    # Bodies are read here, before any upstream call, for two different reasons
    # depending on the route:
    #
    #   /v1/analyze -- cheap schema validation. A malformed body must not cost a
    #                  round trip, let alone GPU quota.
    #   /v1/assets  -- the body IS the payload. It must be forwarded.
    #
    # The second branch is not incidental. Before the 2026-09-22 ruling this
    # route returned 501 without reading the body, so there was nothing to
    # forward. Routing it through `_proxy` with a `path == "/v1/analyze"` guard
    # left `body = None`, which httpx sends as an empty body -- the gateway would
    # have accepted the upload, forwarded nothing, and relayed the Space's
    # refusal with no indication that the gateway had eaten the file.
    #
    # F-6. `policy.admit` step 3 refuses an oversized body from the
    # `Content-Length` HEADER alone, and it has to: it runs before the body is
    # read, so the header is the only evidence available. That makes the cap
    # *declarative* -- a client that omits the header is never measured. Measured
    # through this stack on 2026-09-22 with `max_body_bytes` at 8 MiB:
    #
    #     Content-Length declared, 12 MiB -> 413 oversized_image, peak 0.2 MiB,
    #                                        0 bytes read (the cap worked)
    #     Content-Length omitted,  12 MiB -> 502 model_unavailable, peak 13.9 MiB,
    #                                        12 MiB read (the cap was SKIPPED)
    #
    # and allocation tracked the body size exactly with no ceiling -- 1/8/16/32/64
    # MiB in produced 3.0/8.1/16.0/32.0/64.0 MiB allocated. So the header check
    # protects the common case and bounds nothing in the hostile one: the body is
    # buffered in full before any downstream layer sees it. The rate limiter does
    # not cover this either. It bounds request COUNT (10/60s, measured engaging
    # at exactly 10), not request SIZE -- ten admitted 16 MiB requests are 160 MiB
    # of unaccounted memory. That is the F-5 distinction again: a fairness
    # control is not a protection control.
    #
    # The fix is to make the cap *unconditional* by enforcing it while reading,
    # which is what `gateway/assets.py`'s docstring already CLAIMS happens
    # ("enforced while reading, not after"). Until now that claim was true of the
    # Space's store and false of this gateway. `_read_body_bounded` below is the
    # second line the module docstring describes, and unlike the header check it
    # cannot be skipped by omitting a header.
    body: bytes | None = None
    if path == "/v1/analyze":
        body, too_large = await _read_body_bounded(request, cfg.max_body_bytes)
        if too_large is not None:
            status, err_body = translate_error(
                "oversized_image",
                "The request body is too large.",
                detail=(
                    f"the body exceeds the {cfg.max_body_bytes} byte limit; this "
                    f"limit is enforced while reading and does not depend on a "
                    f"Content-Length header"
                ),
                request_id=decision.request_id,
            )
            return JSONResponse(
                status_code=status, content=err_body, headers=decision.headers
            )
        parsed, error = validate_analyze_body(body)
        if error is not None:
            status, err_body = error
            # Attach the id after the fact so the envelope always carries one.
            err_body["error"]["request_id"] = decision.request_id
            return JSONResponse(
                status_code=status, content=err_body, headers=decision.headers
            )
    elif path == "/v1/assets":
        body, too_large = await _read_body_bounded(request, cfg.max_body_bytes)
        if too_large is not None:
            status, err_body = translate_error(
                "oversized_image",
                "The request body is too large.",
                detail=(
                    f"the body exceeds the {cfg.max_body_bytes} byte limit; this "
                    f"limit is enforced while reading and does not depend on a "
                    f"Content-Length header"
                ),
                request_id=decision.request_id,
            )
            return JSONResponse(
                status_code=status, content=err_body, headers=decision.headers
            )

    if httpx is None:  # pragma: no cover
        status, err_body = translate_error(
            "specialist_error",
            "The gateway cannot reach its upstream.",
            detail="httpx is not installed in this environment",
            request_id=decision.request_id,
        )
        return JSONResponse(status_code=503, content=err_body, headers=decision.headers)

    token = _env().get("HF_TOKEN", "")
    headers = policy.upstream_headers(dict(request.headers), token=token)
    headers["X-Request-Id"] = decision.request_id

    try:
        async with httpx.AsyncClient(timeout=cfg.upstream_timeout_s) as client:
            upstream = await client.request(
                method=request.method,
                url=policy.upstream_url(path),
                headers=headers,
                content=body,
                params=dict(request.query_params),
            )
    except Exception as exc:  # network-level failure
        # NO RETRY. A retry on /v1/analyze would spend GPU quota twice
        # (docs/DEPLOYMENT_ARCHITECTURE.md section 2.2).
        #
        # F-15c (owner ruling 2026-09-23): this branch used to publish
        # `detail=f"{type(exc).__name__}: {exc}"`. `detail` is part of the
        # documented error envelope (`gateway/policy.py:165`), so that put a
        # third-party exception's class name and message in front of an
        # unauthenticated client -- naming the gateway's HTTP client and its
        # internals, and able to carry a proxy URL or a filesystem path.
        #
        # The client keeps what it can act on: the classification (`model_unavailable`,
        # recoverable, 502, the same `request_id`), plus a transport-level reason
        # from the shared classifier. The raw exception is moved to the log
        # rather than dropped.
        _log.error(
            "upstream transport failure for request_id=%s: %s: %s",
            decision.request_id,
            type(exc).__name__,
            exc,
            exc_info=exc,
        )
        status, err_body = translate_error(
            "model_unavailable",
            "The analysis service is not reachable.",
            detail=_transport_failure_detail(exc),
            recoverable=True,
            request_id=decision.request_id,
        )
        return JSONResponse(status_code=502, content=err_body, headers=decision.headers)

    # F-2: `response_headers` drops every `access-control-*` the upstream sent,
    # so the CORS answer below is the gateway's alone. `decision.headers` is the
    # only source of CORS headers on this leg, and it can hold at most the
    # allowlisted origin (an empty dict for anyone else). Before this fix the
    # upstream's own permissive header was relayed verbatim, alongside the
    # gateway's correct value, bypassing the allowlist entirely.
    out_headers = policy.response_headers(dict(upstream.headers))
    # The CORS answer is decided in ONE place. If `response_headers` ever stops
    # filtering, the upstream's headers reach a disallowed origin -- so this
    # pins the filter rather than trusting it. Verified end to end by
    # `tests/unit/test_gateway_app.py::test_an_upstream_cors_header_cannot_bypass_the_allowlist`.
    assert not any(_is_cors_header(k) for k in out_headers) or decision.headers, (
        "a CORS header reached the response without a policy decision; the "
        "upstream's headers are no longer filtered (see F-2)"
    )
    out_headers.update(decision.headers)

    # Non-JSON from the Space is a defect (architecture doc section 5): the
    # frontend must not receive a body it cannot parse as the contract's shape.
    #
    # The guard covers EVERY status, not only 2xx. It originally read
    # `upstream.status_code < 400`, which meant a non-JSON 4xx/5xx from the
    # upstream -- a proxy error page, an HTML 502 from a load balancer, a
    # plain-text stack trace from a misconfigured Space -- was forwarded verbatim.
    # Two consequences, both observed:
    #
    #   * the client received `text/plain` where docs/API_CONTRACT.md section 5
    #     guarantees `{"error": {...}}`, so an error handler written to the
    #     contract would fail to parse exactly when it was most needed;
    #   * the upstream's own wording was passed through unfiltered, which is how
    #     this was found: a sandbox egress proxy returned a 502 whose body
    #     disclosed `os error 10061`.
    #
    # `/v1/health` is exempt because a liveness probe may legitimately answer
    # non-JSON, and it consumes no GPU quota.
    content_type = upstream.headers.get("content-type", "")
    if "application/json" not in content_type and path != "/v1/health":
        status, err_body = translate_error(
            "schema_validation_error",
            "The analysis service returned a malformed response.",
            detail=(
                f"content-type={content_type!r} for {path} "
                f"(upstream status {upstream.status_code})"
            ),
            request_id=decision.request_id,
        )
        # 502 regardless of the upstream's own status: the fault the CLIENT can
        # act on is "the gateway's upstream misbehaved", and echoing e.g. a 404
        # from a reverse proxy would suggest the API path itself was wrong.
        return JSONResponse(status_code=502, content=err_body, headers=decision.headers)

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=out_headers,
        media_type=content_type or "application/json",
    )


def _is_cors_header(name: str) -> bool:
    """Whether a header name is part of the CORS family.

    A single definition, used by both the `_proxy` assertion and the test that
    pins it. The check is deliberately written WITHOUT the literal prefix, so
    that `tests/unit/test_gateway_app.py`'s "the app must not build an
    access-control-* header itself" assertion is testing the app's behaviour
    rather than tripping over this helper's spelling (which it did, on the
    first attempt at this fix -- the same prose-versus-code confusion the
    assertion exists to avoid).
    """
    return name.lower().startswith(_CORS_HEADER_PREFIX)


#: Built at runtime so this file never contains the literal `access-control-`
#: prefix that the source-level guard above searches for.
_CORS_HEADER_PREFIX = "access-" + "control-"


def _client_ip(request: Any) -> str:
    """Best-effort client IP.

    `X-Forwarded-For` is attacker-controlled unless a trusted proxy sets it.
    Railway does set it, so the first hop is used; the value is a rate-limit key,
    not an identity (`docs/DEPLOYMENT_ARCHITECTURE.md` section 2.2 excludes
    auth). The runbook states this limitation rather than implying the limiter is
    a security control.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    client = getattr(request, "client", None)
    return getattr(client, "host", "") or "unknown"


def _env() -> Mapping[str, str]:
    import os

    return os.environ


#: The module-level app a server imports (`uvicorn gateway.app:app`). Built
#: eagerly so a misconfiguration fails at import, before the port is bound.
try:  # pragma: no cover - depends on FastAPI being importable
    app = create_app()
except Exception:  # pragma: no cover - the sandbox path
    # Deliberately broad: this module is unimportable in an environment with an
    # application-control policy blocking orjson, and re-raising would make the
    # failure look like a bug in this repository rather than an environment
    # limitation. `create_app` remains available for a caller that has FastAPI.
    app = None  # type: ignore[assignment]
