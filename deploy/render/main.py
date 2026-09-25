"""SatQuery AI — Render orchestrator (the API gateway boundary).

A thin, **stateless** proxy that sits between the Cloudflare Pages frontend and
the GitHub Codespace inference server. It holds no model, no state, no database,
and performs **no auth** (per plan §73/§74).

Topology
--------
    Cloudflare Pages (frontend)
        -> Render (this service)            [PORT, supplied by Render]
            -> GitHub Codespace (inference) [CODESPACE_PORT, public forwarded URL]

Wake flow
---------
If the Codespace is stopped, this service starts it via the GitHub Codespaces
REST API, polls its ``/v1/health`` until it answers 200, then proxies the
request. The client simply **waits** (blocking, wake-then-proxy) and receives
the result; the frontend independently shows "Waking inference engine..." on slow
responses. We optionally tag the response with ``X-SatQuery-State: waking`` so
the frontend can confirm the delay was a cold start.

Proxied routes (never answered locally)
---------------------------------------
    POST /api/infer        -> POST {codespace}/v1/analyze
    GET  /api/capabilities-> GET  {codespace}/v1/capabilities
    POST /api/assets       -> POST {codespace}/v1/assets

There is deliberately **no second copy** of the capability table here; the
gateway proxies ``/v1/capabilities`` and nothing else decides that question.

Error contract
--------------
Upstream failures are wrapped in the v1 envelope
``{"error": {"code", "message", "detail", "recoverable"}}`` (see
``docs/DEPLOYMENT_ARCHITECTURE.md`` §2.3). A connection error to the Codespace
maps to ``502`` with ``recoverable: true``; a wake that times out maps to ``504``
with ``recoverable: true``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:  # Imported as a package (`deploy.render.main`) on Render.
    from .codespaces import (
        CodespaceAuthError,
        CodespaceError,
        CodespaceNotFoundError,
        forwarded_url,
        get_codespace,
        start_codespace,
    )
except ImportError:  # pragma: no cover - local/dev fallback
    from codespaces import (  # type: ignore
        CodespaceAuthError,
        CodespaceError,
        CodespaceNotFoundError,
        forwarded_url,
        get_codespace,
        start_codespace,
    )

__all__ = ["WakeTimeout", "create_app", "app"]

_log = logging.getLogger("satquery.orchestrator")

# Polling knobs for the wake loop.
_WAKE_POLL_INTERVAL_S = 2.0
_WAKE_HEALTH_TIMEOUT_S = 10.0

# A shared client for the inference proxy. Its timeout is the upstream timeout;
# the shorter per-poll health timeout uses its own client.
_HTTP: httpx.AsyncClient | None = None


# ---------------------------------------------------------------------------
# Configuration (read from env; see docs/DEPLOYMENT_ARCHITECTURE.md §4)
# ---------------------------------------------------------------------------


def _github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise OrchestratorConfigError(
            "GITHUB_TOKEN is not set; the orchestrator cannot wake the Codespace."
        )
    return token


def _codespace_name() -> str:
    name = os.environ.get("CODESPACE_NAME", "").strip()
    if not name:
        raise OrchestratorConfigError(
            "CODESPACE_NAME is not set; the orchestrator does not know which "
            "Codespace to target."
        )
    return name


def _codespace_port() -> int:
    return int(os.environ.get("CODESPACE_PORT", "8000"))


def _wake_timeout_s() -> float:
    return float(os.environ.get("SATQUERY_WAKE_TIMEOUT_S", "120"))


def _upstream_timeout_s() -> float:
    return float(os.environ.get("SATQUERY_UPSTREAM_TIMEOUT_S", "90"))


#: Local development-server origins, ADDED to the operator's allowlist so the
#: frontend can be driven from a dev server without deploying to Cloudflare on
#: every JavaScript change.
#:
#: The ports are the ones a static-file dev server actually uses here:
#: `python -m http.server` and `npx serve` default to 8000/3000, Vite to 5173,
#: Create-React-App to 3000, Live Server to 5500, and 8080 is the common
#: alternative when 8000 is taken (it usually is -- the Space itself uses 8000).
#:
#: Both `localhost` and `127.0.0.1` are listed because they are **distinct
#: origins to the browser**: `http://localhost:8080` and `http://127.0.0.1:8080`
#: do not match each other, and a developer who pastes either one must not get a
#: silent CORS failure that looks like a broken API.
#:
#: This list is deliberately EXPLICIT, never a wildcard or a suffix match. It
#: cannot be used to reach the deployment from an arbitrary host: only a browser
#: running on the developer's own machine can send `Origin: http://localhost:*`.
#: Production traffic from `https://satquery.pages.dev` is unaffected.
_DEV_ORIGINS: tuple[str, ...] = tuple(
    f"http://{host}:{port}"
    for host in ("localhost", "127.0.0.1")
    for port in ("3000", "5500", "5173", "8000", "8080")
)

#: The production frontend origin. Listed here rather than only in the
#: environment so that a deployment which forgets `SATQUERY_ALLOWED_ORIGINS`
#: still serves the real frontend -- an empty allowlist would otherwise take the
#: live site down, which is a worse failure than the one this guards.
_PRODUCTION_ORIGINS: tuple[str, ...] = ("https://satquery.pages.dev",)


def _dev_origins_enabled() -> bool:
    """Whether the development origins are currently part of the allowlist.

    Derived from the same switch `_allowed_origins` reads, rather than by
    re-deriving it from the assembled list: inferring it from the output made
    the health payload depend on set arithmetic that could drift from the
    switch. One source, one answer.
    """
    return os.environ.get("SATQUERY_ALLOW_DEV_ORIGINS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "",
    )


def _allowed_origins(include_dev: bool | None = None) -> list[str]:
    """The CORS allowlist: operator origins + production + (dev) localhost.

    Order of assembly, and why:

    1. `SATQUERY_ALLOWED_ORIGINS` -- the operator's comma-separated list. Still
       the authoritative source for any additional deployment origin.
    2. `_PRODUCTION_ORIGINS` -- always present, so production cannot be removed
       by an environment typo.
    3. `_DEV_ORIGINS` -- added unless explicitly disabled.

    Args:
        include_dev: Force the development origins on or off. Defaults to the
            value of `SATQUERY_ALLOW_DEV_ORIGINS` (default enabled). Set it to
            `0` for a deployment that must expose only production origins; the
            production origin and the operator list are never affected by this
            switch.

    Returns:
        A de-duplicated list preserving first-seen order.

    Raises:
        ValueError: if a wildcard is requested. Checked here as well as in
            `GatewayConfig.__post_init__` so this function cannot be the way a
            `*` reaches `CORSMiddleware`, which does not run that validator.
    """
    raw = os.environ.get("SATQUERY_ALLOWED_ORIGINS", "")
    origins: list[str] = [o.strip() for o in raw.split(",") if o.strip()]

    if include_dev is None:
        include_dev = _dev_origins_enabled()

    origins.extend(_PRODUCTION_ORIGINS)
    if include_dev:
        origins.extend(_DEV_ORIGINS)

    if "*" in origins:
        raise ValueError(
            "SATQUERY_ALLOWED_ORIGINS must not contain '*'. A wildcard exposes "
            "the deployment to any origin (docs/DEPLOYMENT_ARCHITECTURE.md 2.1)."
        )

    seen: set[str] = set()
    deduped: list[str] = []
    for origin in origins:
        if origin not in seen:
            seen.add(origin)
            deduped.append(origin)
    return deduped


async def _http() -> httpx.AsyncClient:
    global _HTTP
    if _HTTP is None or _HTTP.is_closed:
        _HTTP = httpx.AsyncClient(timeout=_upstream_timeout_s())
    return _HTTP


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OrchestratorError(Exception):
    """Base class for errors the orchestrator surfaces as an envelope."""

    code: str = "satquery_error"
    message: str = "An internal error occurred."
    detail: str = ""
    status: int = 500
    recoverable: bool = False

    def __init__(self, detail: str = "", *, message: str | None = None) -> None:
        self.detail = detail or ""
        if message is not None:
            self.message = message
        super().__init__(self.message)


class WakeTimeout(OrchestratorError):
    """The Codespace did not reach `available` within the wake timeout."""

    code = "wake_timeout"
    message = "The inference engine did not start in time. Please retry shortly."
    status = 504
    recoverable = True


class OrchestratorConfigError(OrchestratorError):
    """The orchestrator is misconfigured (missing GITHUB_TOKEN / CODESPACE_NAME)."""

    code = "orchestrator_config_error"
    message = "The orchestrator is misconfigured."
    status = 500
    recoverable = False


class OrchestratorUpstreamError(OrchestratorError):
    """A control-plane call to GitHub or a connection to the Codespace failed."""

    code = "upstream_unreachable"
    message = "The inference engine is unreachable."
    status = 502
    recoverable = True


def _envelope(
    code: str,
    message: str,
    detail: str = "",
    *,
    status: int = 500,
    recoverable: bool = False,
) -> JSONResponse:
    """Build a v1 error envelope response."""
    body = {
        "error": {
            "code": code,
            "message": message,
            "detail": detail or None,
            "recoverable": bool(recoverable),
        }
    }
    return JSONResponse(status_code=status, content=body)


# ---------------------------------------------------------------------------
# Wake flow
# ---------------------------------------------------------------------------


async def ensure_codespace_up() -> tuple[str, bool]:
    """Ensure the Codespace is running; return ``(base_url, woke)``.

    Steps:
      1. ``GET`` the Codespace via the GitHub API.
      2. If ``state != "available"``, ``POST .../start``.
      3. Poll ``GET {base}/v1/health`` until 200 or until
         ``SATQUERY_WAKE_TIMEOUT_S`` elapses.

    Returns:
        A tuple of the public forwarded-port base URL (no trailing slash) and a
        boolean that is True when a wake/start was required.

    Raises:
        OrchestratorConfigError: missing token / codespace name.
        OrchestratorUpstreamError: GitHub API auth/transport failure (502,
            recoverable).
        WakeTimeout: the engine never became ready (504, recoverable).
    """
    token = _github_token()
    name = _codespace_name()
    port = _codespace_port()

    try:
        cs = await get_codespace(token, name)
    except CodespaceAuthError as exc:
        raise OrchestratorUpstreamError(str(exc)) from exc
    except CodespaceNotFoundError as exc:
        raise OrchestratorConfigError(str(exc)) from exc
    except CodespaceError as exc:
        raise OrchestratorUpstreamError(str(exc)) from exc

    base = forwarded_url(cs, port)
    state = (cs.get("state") or "").lower()
    woke = state != "available"
    if woke:
        try:
            await start_codespace(token, name)
        except CodespaceError as exc:
            raise OrchestratorUpstreamError(str(exc)) from exc

    deadline = time.monotonic() + _wake_timeout_s()
    last_err: str | None = None
    while time.monotonic() < deadline:
        try:
            async with httpx.AsyncClient(timeout=_WAKE_HEALTH_TIMEOUT_S) as client:
                health = await client.get(f"{base}/v1/health")
            if health.status_code == 200:
                return base, woke
            last_err = f"health status {health.status_code}"
        except httpx.HTTPError as exc:
            last_err = str(exc)
        await asyncio.sleep(_WAKE_POLL_INTERVAL_S)

    raise WakeTimeout(
        f"Timed out after {_wake_timeout_s():.0f}s waiting for /v1/health. "
        f"Last probe error: {last_err or 'unknown'}."
    )


# ---------------------------------------------------------------------------
# Proxy helper
# ---------------------------------------------------------------------------


async def _proxy(
    method: str,
    url: str,
    *,
    json: Any = None,
    content: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Proxy a JSON request to the Codespace and return its body verbatim.

    Connection/transport errors and non-JSON upstream bodies are translated into
    the v1 envelope (``502``, ``recoverable: true``); the upstream status is
    otherwise passed through unchanged.
    """
    client = await _http()
    try:
        resp = await client.request(
            method, url, json=json, content=content, headers=headers
        )
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.TimeoutException) as exc:
        _log.warning("upstream connection error to %s: %s", url, exc)
        return _envelope(
            "upstream_unreachable",
            "The inference engine is unreachable.",
            str(exc),
            status=502,
            recoverable=True,
        )
    except httpx.HTTPError as exc:
        _log.warning("upstream request error to %s: %s", url, exc)
        return _envelope(
            "upstream_error",
            "The inference engine request failed.",
            str(exc),
            status=502,
            recoverable=True,
        )

    try:
        data = resp.json()
    except Exception:
        _log.error(
            "non-JSON upstream response from %s (status %s)", url, resp.status_code
        )
        return _envelope(
            "schema_validation_error",
            "The inference engine returned a non-JSON response.",
            f"upstream status {resp.status_code}",
            status=502,
            recoverable=True,
        )

    return JSONResponse(content=data, status_code=resp.status_code)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(title="SatQuery AI Orchestrator", version="1.0")

    # CORS fix: CORSMiddleware answers OPTIONS preflight itself, which resolves
    # the earlier 405 on preflight. allow_credentials=False per the contract
    # (no auth, no cookies); the origin list is assembled by `_allowed_origins`
    # from the operator env var plus the production and development origins.
    #
    # `allow_origins` stays an explicit list. The dev origins are enumerated
    # host:port pairs, NOT a regex or a wildcard, so allowing localhost for
    # development cannot admit an arbitrary remote site.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )
    app.add_exception_handler(OrchestratorError, _handle_orchestrator_error)

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        """Orchestrator liveness. Reports its own configuration; never answers
        for the Codespace (that is /api/capabilities)."""
        return {
            "status": "ok",
            "service": "satquery-orchestrator",
            "config": {
                "codespace_name": os.environ.get("CODESPACE_NAME", ""),
                "codespace_port": _codespace_port(),
                "has_github_token": bool(os.environ.get("GITHUB_TOKEN")),
                # Reports the EFFECTIVE list, so an operator can confirm from
                # outside what the service will actually accept -- not just what
                # they set. The dev entries are visible here, which is how a
                # production deployment proves it turned them off.
                "allowed_origins": _allowed_origins(),
                "production_origins": list(_PRODUCTION_ORIGINS),
                "dev_origins_enabled": _dev_origins_enabled(),
                "wake_timeout_s": _wake_timeout_s(),
                "upstream_timeout_s": _upstream_timeout_s(),
                "device": os.environ.get("SATQUERY_DEVICE", ""),
            },
        }

    @app.post("/api/infer")
    async def infer(request: Request) -> JSONResponse:
        """Main path: wake (if needed) then proxy ``POST /v1/analyze``.

        Blocking wake-then-proxy — the client waits and receives the result. The
        ``X-SatQuery-State`` header tells the frontend whether a cold start
        occurred.
        """
        try:
            body = await request.json()
        except Exception:
            return _envelope(
                "invalid_request",
                "The request body was not valid JSON.",
                status=400,
                recoverable=False,
            )

        base, woke = await ensure_codespace_up()
        out = await _proxy("POST", f"{base}/v1/analyze", json=body)
        out.headers["X-SatQuery-State"] = "waking" if woke else "ready"
        return out

    @app.get("/api/capabilities")
    async def capabilities() -> JSONResponse:
        """Proxy ``GET /v1/capabilities`` — no local capability table."""
        base, _ = await ensure_codespace_up()
        return await _proxy("GET", f"{base}/v1/capabilities")

    @app.post("/api/assets")
    async def assets(request: Request) -> JSONResponse:
        """Proxy ``POST /v1/assets`` verbatim (multipart/raw bytes)."""
        base, _ = await ensure_codespace_up()
        raw = await request.body()
        content_type = request.headers.get("content-type", "application/octet-stream")
        return await _proxy(
            "POST",
            f"{base}/v1/assets",
            content=raw,
            headers={"content-type": content_type},
        )

    return app


async def _handle_orchestrator_error(request: Request, exc: OrchestratorError) -> JSONResponse:
    return _envelope(
        exc.code,
        exc.message,
        exc.detail,
        status=exc.status,
        recoverable=exc.recoverable,
    )


# The ASGI object uvicorn imports: `uvicorn deploy.render.main:app`.
app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run("deploy.render.main:app", host="0.0.0.0", port=port)
