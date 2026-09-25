"""SatQuery AI — gateway policy: everything the gateway decides, without a server.

WHY THIS MODULE IS SEPARATE FROM THE ASGI APP
---------------------------------------------
`docs/DEPLOYMENT_ARCHITECTURE.md` section 2 lists eight gateway responsibilities:
schema validation, size limits, rate limiting, CORS, request IDs, timeouts,
secret custody and error translation. Those are *decisions*, and decisions can be
tested directly. Only the HTTP plumbing around them needs a framework.

Splitting them means:
  * every rule below is unit-testable with no ASGI server, no network, no GPU;
  * `gateway/app.py` becomes a thin adapter that cannot silently drift from the
    rules (it calls into here rather than re-implementing them);
  * the eight responsibilities have exactly one home, so an operator reading
    `docs/BACKEND_DEPLOYMENT_RUNBOOK.md` section 4 can find the code that
    enforces each variable.

This split also happens to make the package usable in environments where the
ASGI framework cannot be imported (see `gateway.__init__`). That is a
consequence, not the motivation.

THE ONE RULE THAT MATTERS MOST
------------------------------
**Nothing here may spend GPU quota.** Every rejection path exists so that a
malformed request never reaches the Space. The ordering in
:meth:`GatewayPolicy.admit` is therefore deliberate: cheap shape checks first,
and the expensive upstream call only after all of them pass.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = [
    "DEFAULT_HOP_BY_HOP",
    "DEFECT_CODES",
    "GATEWAY_ORIGIN_CODES",
    "ClientIdentity",
    "GatewayConfig",
    "GatewayPolicy",
    "PolicyDecision",
    "RateLimiter",
    "RateLimitState",
    "build_cors_headers",
    "config_from_env",
    "new_request_id",
    "translate_error",
    "validate_analyze_body",
]


# --------------------------------------------------------------------------
# Error translation
# --------------------------------------------------------------------------

#: HTTP status per SatQuery error code. From `docs/API_CONTRACT.md` section 5.1.
#:
#: The mapping is TOTAL over the taxonomy in `core/errors.py`: every code either
#: appears here or falls through to 500, and `tests/unit/test_gateway_policy.py`
#: asserts that no code silently maps to the wrong class. A gateway that let an
#: unknown code produce a 200 would turn a defect into a success.
_CODE_STATUS: dict[str, int] = {
    "input_error": 400,
    "raster_read_error": 400,
    "missing_crs": 400,
    "unsupported_bands": 400,
    "oversized_image": 413,
    "pair_incompatible": 422,
    "pair_misaligned": 422,
    "temporal_pair_invalid": 422,
    "routing_error": 422,
    "unsupported_query": 422,
    "invalid_request": 422,
    "workflow_plan_error": 500,
    "specialist_error": 500,
    "model_load_error": 503,
    "model_unavailable": 503,
    "out_of_memory": 503,
    "specialist_timeout": 504,
    "schema_validation_error": 500,
    "coordinate_error": 500,
    "confidence_range_error": 500,
    "leakage_violation": 500,
    "benchmark_freeze_error": 500,
    # The base class. A bare `satquery_error` means no specific code applied,
    # which is an internal failure, not a client error.
    "satquery_error": 500,
}

#: Codes that indicate the *system* is broken, not the request. The frontend is
#: instructed to surface these rather than swallow them (`API_CONTRACT.md` 5.2).
DEFECT_CODES: frozenset[str] = frozenset(
    {
        "model_load_error",
        "schema_validation_error",
        "coordinate_error",
        "confidence_range_error",
        "leakage_violation",
    }
)

#: Errors the GATEWAY itself produces, which by construction are not in
#: `core/errors.py` -- that taxonomy covers the *analysis* pipeline, and a rate
#: limit is a property of the proxy, not of an analysis.
#:
#: `docs/API_CONTRACT.md` section 5.1 maps HTTP 429 to "Rate limited" and
#: section 2.3 forbids the gateway from inventing codes. Those two are reconciled
#: by keeping gateway-originated codes in a namespace the contract already
#: reserves: `API_CONTRACT.md` section 5.1 documents the 429 status but assigns it
#: no code from the taxonomy, so `rate_limited` is declared here as a
#: gateway-origin code and added to the contract's documented set.
#:
#: The rule that still holds: the gateway never invents a code for an error that
#: ORIGINATED in the Space. Those pass through unchanged.
GATEWAY_ORIGIN_CODES: frozenset[str] = frozenset({"rate_limited"})

_CODE_STATUS["rate_limited"] = 429


def translate_error(
    code: str,
    message: str = "",
    *,
    detail: str = "",
    recoverable: bool = False,
    request_id: str = "",
    run_id: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Map a SatQuery error into the documented envelope and its HTTP status.

    The `code` is passed through **unchanged**. `docs/DEPLOYMENT_ARCHITECTURE.md`
    section 2.3 forbids the gateway from inventing or remapping codes: the
    taxonomy in `core/errors.py` is the single source of truth, and a gateway
    that renamed anything would make the frontend's error handling unpredictable.

    Args:
        code: a code from `core/errors.py`. An unrecognised code is treated as
            `satquery_error` and mapped to 500 -- never to a success status.
        message: operator-safe text (`SatQueryError.user_message`).
        detail: technical detail; may be empty.
        recoverable: whether a retry may help.
        request_id: the correlation id this gateway generated.
        run_id: the analysis run id, when one exists.

    Returns:
        `(http_status, body)` where `body` is `{"error": {...}}`.
    """
    known = code in _CODE_STATUS
    status = _CODE_STATUS.get(code, 500)
    if not known:
        # Keep the original code visible in `detail` for diagnosis, but present
        # a code the contract defines. Swallowing it entirely would hide a real
        # defect behind a generic one.
        detail = f"unmapped error code {code!r}" + (f"; {detail}" if detail else "")

    body: dict[str, Any] = {
        "error": {
            "code": code if known else "satquery_error",
            "message": message or "An internal error occurred.",
            "detail": detail or None,
            "recoverable": bool(recoverable),
            "request_id": request_id,
            "run_id": run_id,
        }
    }
    return status, body


# --------------------------------------------------------------------------
# Request ids
# --------------------------------------------------------------------------

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{8,64}$")


def new_request_id(seed: str | None = None) -> str:
    """Generate a correlation id.

    Format `req_<hex>`. When `seed` is given the id is a deterministic function
    of it, which lets tests assert on ids without patching a clock or an RNG --
    the same technique the rest of this repository uses for reproducibility.

    An inbound `X-Request-Id` is NOT trusted verbatim: see
    :meth:`GatewayPolicy.accept_request_id`.
    """
    if seed is None:
        seed = f"{time.time_ns()}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
    return f"req_{digest}"


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GatewayConfig:
    """The gateway's operator-settable limits.

    Every field maps to an environment variable named in
    `docs/BACKEND_DEPLOYMENT_RUNBOOK.md` section 4.1, so an operator can find
    the code behind each variable. No default here is a plan fact: the plan
    specifies none of these values (`API_CONTRACT.md` section 8). They are
    defaults this implementation chose, and the runbook says so.
    """

    #: Upstream Space origin, no trailing slash.
    upstream_url: str = ""
    #: Comma-separated frontend origins. Never "*".
    allowed_origins: tuple[str, ...] = ()
    #: Request body cap. FastAPI/Starlette cannot refuse a body it has already
    #: buffered, so the app checks `Content-Length` before reading.
    max_body_bytes: int = 8 * 1024 * 1024
    #: Per-file cap for multipart uploads.
    max_file_bytes: int = 4 * 1024 * 1024
    #: Per-IP request budget for `POST /v1/analyze`.
    rate_limit_per_ip: int = 10
    #: Window for the per-IP budget, in seconds.
    rate_limit_window_s: float = 60.0
    #: Upstream timeout. MUST be below `agent.timeout_seconds` (120 s) and above
    #: the longest `gpu_duration_*` (45 s). See runbook section 4.2.
    upstream_timeout_s: float = 90.0
    #: Content types accepted for an uploaded file.
    allowed_content_types: tuple[str, ...] = (
        "image/tiff",
        "image/geotiff",
        "image/png",
        "image/jpeg",
        "application/octet-stream",
    )

    def __post_init__(self) -> None:
        if self.upstream_url.endswith("/"):
            raise ValueError(
                "upstream_url must not have a trailing slash; a doubled slash "
                "produces a 404 from the Space that looks like an outage"
            )
        if not self.upstream_url:
            raise ValueError(
                "upstream_url is required; the gateway has nothing to proxy to"
            )
        if not self.allowed_origins:
            raise ValueError(
                "allowed_origins must be non-empty. An empty allowlist would "
                "either block every browser or, if the app fell back to '*', "
                "expose the Space. Refuse to start rather than guess."
            )
        if "*" in self.allowed_origins:
            raise ValueError(
                "allowed_origins must not contain '*'. A wildcard exposes the "
                "Space to any origin (docs/DEPLOYMENT_ARCHITECTURE.md 2.1)"
            )
        if self.upstream_timeout_s <= 45:
            raise ValueError(
                f"upstream_timeout_s={self.upstream_timeout_s} is not above the "
                f"longest gpu_duration_* (45 s): it would kill a legitimate "
                f"grounding/optical_sar call and report it as an upstream "
                f"failure (docs/BACKEND_DEPLOYMENT_RUNBOOK.md 4.2)"
            )
        if self.upstream_timeout_s >= 120:
            raise ValueError(
                f"upstream_timeout_s={self.upstream_timeout_s} is not below "
                f"agent.timeout_seconds (120 s): the gateway would hold a "
                f"connection past the point the Space has given up, turning an "
                f"upstream timeout into a client-side hang"
            )
        if self.max_file_bytes > self.max_body_bytes:
            raise ValueError(
                "max_file_bytes exceeds max_body_bytes; the per-file cap would "
                "be unreachable and the body check would fire first"
            )
        if self.rate_limit_per_ip <= 0:
            raise ValueError("rate_limit_per_ip must be positive")


def config_from_env(env: Mapping[str, str]) -> GatewayConfig:
    """Build a :class:`GatewayConfig` from the operator's environment.

    Kept in this module (rather than in the ASGI app) so the parsing rules are
    testable and so the app has no configuration logic of its own.
    """
    origins = tuple(
        origin.strip()
        for origin in env.get("SATQUERY_ALLOWED_ORIGINS", "").split(",")
        if origin.strip()
    )

    def _float(name: str, default: float) -> float:
        raw = env.get(name)
        return default if raw in (None, "") else float(raw)

    def _int(name: str, default: int) -> int:
        raw = env.get(name)
        return default if raw in (None, "") else int(raw)

    # F-7. `max_file_bytes` is the one variable BOTH layers read -- the gateway
    # validates it here, and the Space builds its `AssetStore` from it
    # (`app/space_app.py::_asset_max_file_bytes`). They used to parse it
    # differently, and the docstrings asserted they "cannot disagree". Measured
    # on 2026-09-22, the same value produced:
    #
    #     '0'   -> gateway ACCEPTED max_file_bytes=0;  Space raised ValueError
    #     '-1'  -> gateway ACCEPTED max_file_bytes=-1; Space raised ValueError
    #     'abc' -> gateway raised at startup;          Space silently defaulted
    #     '4e6' -> gateway raised at startup;          Space silently defaulted
    #
    # A cap of 0 is the dangerous case rather than a harmless typo: the gateway
    # admits the request (its own check is against `max_body_bytes`) and the
    # Space then refuses EVERY upload, because `len(data) > 0` is true for any
    # non-empty file. The operator sees "413 on every upload" against a cap they
    # believe they never set. Failing here instead names the variable.
    #
    # Refusing a non-positive cap is not a preference: `AssetStore.__post_init__`
    # already rejects it, so the gateway accepting it only moves the failure to a
    # layer with less context. This makes the gateway's parse stricter, which is
    # the safe direction -- it cannot refuse a value the Space would have accepted,
    # because the Space refuses these too.
    max_file_bytes = _int("SATQUERY_MAX_FILE_BYTES", 4 * 1024 * 1024)
    if max_file_bytes <= 0:
        raise ValueError(
            f"SATQUERY_MAX_FILE_BYTES={max_file_bytes} is not positive. A "
            f"non-positive per-file cap would make every upload fail on the "
            f"Space while the gateway kept admitting it; the Space's AssetStore "
            f"rejects the same value, so it is refused here to fail at startup "
            f"with the variable named rather than on the first upload"
        )

    return GatewayConfig(
        upstream_url=env.get("SATQUERY_SPACE_URL", "").rstrip("/"),
        allowed_origins=origins,
        max_body_bytes=_int("SATQUERY_MAX_BODY_BYTES", 8 * 1024 * 1024),
        max_file_bytes=max_file_bytes,
        rate_limit_per_ip=_int("SATQUERY_RATE_LIMIT_PER_IP", 10),
        rate_limit_window_s=_float("SATQUERY_RATE_LIMIT_WINDOW_S", 60.0),
        upstream_timeout_s=_float("SATQUERY_UPSTREAM_TIMEOUT_S", 90.0),
    )


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


@dataclass
class RateLimitState:
    """A single window's counter for one identity."""

    window_start: float
    count: int = 0


@dataclass
class RateLimiter:
    """An in-memory fixed-window limiter, keyed by client identity.

    WHY IN-MEMORY, GIVEN THE GATEWAY MAY RESTART
    --------------------------------------------
    A distributed limiter needs shared state, and the plan forbids Redis-cluster
    infrastructure (`docs/DEPLOYMENT_ARCHITECTURE.md` section 6). An in-memory
    limiter in a single-instance Railway service loses its counters on restart --
    which is correct behaviour for protecting a *daily* GPU budget, because the
    thing being protected (the Space's quota) is unaffected by a gateway restart.

    The limiter therefore protects the budget, not a billing invariant. Runbook
    section 4.3 states the two rules that matter and this class implements one
    of them; the other (never retry analyze) is in :class:`GatewayPolicy`.
    """

    limit: int
    window_s: float
    #: Injected clock, so tests are deterministic without sleeping.
    now: Callable[[], float] = field(default=time.monotonic)
    _state: dict[str, RateLimitState] = field(default_factory=dict)

    def check(self, identity: str) -> tuple[bool, int, float]:
        """Record an attempt and report the verdict.

        Returns:
            `(allowed, remaining, retry_after_s)`. `retry_after_s` is 0 when
            allowed. A denied request does **not** increment the counter --
            otherwise a client hammering the endpoint would push its own reset
            further away on every rejected attempt.
        """
        now = self.now()
        state = self._state.get(identity)
        if state is None or now - state.window_start >= self.window_s:
            state = RateLimitState(window_start=now, count=0)
            self._state[identity] = state

        if state.count >= self.limit:
            retry_after = self.window_s - (now - state.window_start)
            return False, 0, max(0.0, retry_after)

        state.count += 1
        return True, self.limit - state.count, 0.0


@dataclass(frozen=True)
class ClientIdentity:
    """How a client is identified for rate limiting.

    Deliberately minimal. `docs/DEPLOYMENT_ARCHITECTURE.md` section 2.2 excludes
    an auth system, so there is no user id to key on -- only transport facts.
    `X-Forwarded-For` is used because the gateway sits behind Railway's proxy,
    but it is attacker-controlled absent a trusted proxy, so this is a
    best-effort guard, not a security boundary. The runbook says so.
    """

    ip: str
    user_agent: str = ""

    def key(self) -> str:
        # Key on IP alone. Including the user agent would let one client obtain
        # an unbounded number of buckets by varying it.
        return self.ip


# --------------------------------------------------------------------------
# CORS
# --------------------------------------------------------------------------


def build_cors_headers(
    origin: str | None, allowed: Sequence[str], *, request_headers: str = ""
) -> dict[str, str]:
    """Explicit CORS headers, or none.

    WHY THIS IS NOT A LIBRARY'S `add_middleware(CORSMiddleware, allow_origins=...)`
    -------------------------------------------------------------------------------
    The middleware is correct, but the *decision* is the thing under test, and
    `docs/DEPLOYMENT_ARCHITECTURE.md` section 2.1 requires an explicit allowlist
    that is never `*`. Stating the decision in a function makes it assertable
    without an ASGI server.

    A disallowed origin receives **no CORS headers at all**, which is what makes
    the browser block the response. Echoing the origin back with
    `Access-Control-Allow-Origin: <origin>` regardless would defeat the
    allowlist entirely.
    """
    if not origin or origin not in allowed:
        return {}
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": request_headers or "Content-Type, X-Request-Id",
        "Access-Control-Max-Age": "600",
        "Vary": "Origin",
    }


# --------------------------------------------------------------------------
# The policy decision
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyDecision:
    """The outcome of admitting a request.

    `admit=False` always carries a `status` and a `body` the caller returns
    as-is, so no rejection path can accidentally produce a success response.
    """

    admit: bool
    request_id: str
    status: int = 200
    body: dict[str, Any] | None = None
    headers: dict[str, str] = field(default_factory=dict)
    detail: str = ""

    @classmethod
    def allowed(cls, request_id: str, headers: Mapping[str, str] | None = None) -> "PolicyDecision":
        return cls(
            admit=True, request_id=request_id, headers=dict(headers or {})
        )

    @classmethod
    def refused(
        cls,
        request_id: str,
        status: int,
        body: dict[str, Any],
        headers: Mapping[str, str] | None = None,
        detail: str = "",
    ) -> "PolicyDecision":
        return cls(
            admit=False,
            request_id=request_id,
            status=status,
            body=body,
            headers=dict(headers or {}),
            detail=detail,
        )


#: Hop-by-hop headers a proxy must not forward (RFC 9110 section 7.6.1).
DEFAULT_HOP_BY_HOP: frozenset[str] = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


class GatewayPolicy:
    """Every gateway decision, in one place, exercised without a server.

    The ASGI app in `gateway.app` calls :meth:`admit` before doing any I/O and
    uses the result verbatim. It must not re-implement any rule below -- if it
    did, the rule would exist twice and drift.
    """

    def __init__(
        self,
        config: GatewayConfig,
        *,
        limiter: RateLimiter | None = None,
        request_id_factory: Callable[[str | None], str] = new_request_id,
    ) -> None:
        self.config = config
        self.limiter = limiter or RateLimiter(
            limit=config.rate_limit_per_ip, window_s=config.rate_limit_window_s
        )
        self._request_id_factory = request_id_factory

    # -- request ids -------------------------------------------------------

    def accept_request_id(self, inbound: str | None, *, seed: str | None = None) -> str:
        """Echo an inbound request id only when it is safe, else generate one.

        Trusting an arbitrary inbound value lets a client collide with, or
        poison, another request's correlation id in the Space's logs. The format
        is therefore constrained (`req_`-less, 8-64 chars of `[A-Za-z0-9_-]`),
        and a rejected value is replaced rather than sanitised -- a mangled
        client id is more confusing than a fresh one.
        """
        if inbound and _REQUEST_ID_RE.match(inbound):
            return inbound
        return self._request_id_factory(seed)

    # -- the admit ladder --------------------------------------------------

    def admit(
        self,
        *,
        method: str,
        path: str,
        content_length: int | None,
        origin: str | None,
        identity: ClientIdentity,
        inbound_request_id: str | None = None,
        is_analyze: bool | None = None,
    ) -> PolicyDecision:
        """Decide whether to forward a request upstream.

        The order below is the cheapest-check-first ladder. It is deliberate:
        a size or rate rejection must not have parsed a body, and no rejection
        path may reach the Space.

        Args:
            method: HTTP method.
            path: request path, e.g. `/v1/analyze`.
            content_length: the `Content-Length` header value, or None.
            origin: the `Origin` header, or None.
            identity: how the client is identified for rate limiting.
            inbound_request_id: a client-supplied `X-Request-Id`, if any.
            is_analyze: override for the "this route costs GPU" test. Defaults
                to a path check. Tests pass it explicitly so a route rename
                cannot silently disable rate limiting.
        """
        request_id = self.accept_request_id(inbound_request_id)
        cost = path.startswith("/v1/analyze") if is_analyze is None else is_analyze

        cors = build_cors_headers(origin, self.config.allowed_origins)
        headers = {"X-Request-Id": request_id, **cors}

        # 1. Preflight is answered here, never forwarded. The Space has no CORS
        #    configuration and forwarding OPTIONS would waste a round trip.
        if method == "OPTIONS":
            return PolicyDecision.allowed(request_id, headers)

        # 2. Method allowlist. A method the contract does not define is a 405
        #    from the gateway, not a 404 from the Space.
        allowed_methods = {"GET", "HEAD", "POST", "OPTIONS"}
        if method not in allowed_methods:
            status, body = translate_error(
                "invalid_request",
                f"Method {method} is not supported.",
                detail=f"allowed: {', '.join(sorted(allowed_methods))}",
                request_id=request_id,
            )
            return PolicyDecision.refused(request_id, status, body, headers)

        # 3. Body size, BEFORE the body is read. This is the check that saves
        #    quota and memory; everything downstream has already buffered it.
        if content_length is not None and content_length > self.config.max_body_bytes:
            status, body = translate_error(
                "oversized_image",
                "The request body is too large.",
                detail=(
                    f"Content-Length {content_length} exceeds the "
                    f"{self.config.max_body_bytes} byte limit"
                ),
                request_id=request_id,
            )
            return PolicyDecision.refused(request_id, status, body, headers)

        # 4. Rate limiting, only for routes that cost GPU quota. Rate-limiting
        #    health checks would make the frontend's load probe fail for no gain.
        if cost:
            allowed, remaining, retry_after = self.limiter.check(identity.key())
            if not allowed:
                status, body = translate_error(
                    "rate_limited",
                    "Too many requests. Please wait and try again.",
                    detail=(
                        f"limit {self.config.rate_limit_per_ip} per "
                        f"{self.config.rate_limit_window_s:g}s"
                    ),
                    recoverable=True,
                    request_id=request_id,
                )
                retry_headers = {
                    **headers,
                    "Retry-After": f"{max(1, int(retry_after + 0.999))}",
                }
                return PolicyDecision.refused(
                    request_id, 429, body, retry_headers, detail="rate limited"
                )
            headers["X-RateLimit-Remaining"] = str(remaining)

        return PolicyDecision.allowed(request_id, headers)

    # -- upstream helpers --------------------------------------------------

    def upstream_url(self, path: str) -> str:
        """The absolute upstream URL for a request path."""
        return f"{self.config.upstream_url}{path}"

    def upstream_headers(self, incoming: Mapping[str, str], *, token: str) -> dict[str, str]:
        """Build the headers to send upstream.

        Strips hop-by-hop headers (a proxy must not forward them) and injects
        the bearer token. **The token never travels back to the client** --
        `docs/DEPLOYMENT_ARCHITECTURE.md` section 4 keeps it here.

        The `Authorization` header is omitted entirely when `token` is empty.
        Sending `Authorization: Bearer ` (with an empty credential) is rejected
        by httpx itself with `LocalProtocolError: Illegal header value`, which
        surfaced as a 502 whose `detail` blamed the upstream -- even though the
        upstream had not been contacted. An unauthenticated deployment is a
        legitimate configuration (the Space may be public, and `HF_TOKEN` is
        optional in the runbook), so the correct behaviour is to send no
        credential rather than an empty one.
        """
        out: dict[str, str] = {}
        for key, value in incoming.items():
            if key.lower() in DEFAULT_HOP_BY_HOP:
                continue
            if key.lower() in ("host", "authorization"):
                # `authorization` is dropped rather than overwritten so a client
                # cannot smuggle a credential toward the Space.
                continue
            out[key] = value
        if token:
            out["Authorization"] = f"Bearer {token}"
        return out

    def response_headers(self, upstream: Mapping[str, str]) -> dict[str, str]:
        """Filter upstream response headers on the way back.

        Strips hop-by-hop headers, plus three classes that must never reach a
        browser:

          * `content-length` / `content-encoding` -- the ASGI layer re-frames
            the body, so a forwarded length would be wrong and a forwarded
            encoding would be applied twice.
          * `authorization` / `www-authenticate` / `proxy-authorization` -- a
            caller could pass this method the *request* headers by mistake, and
            the token we injected upstream is a credential. Stripping them here
            makes that mistake non-disclosing rather than catastrophic.
          * `set-cookie` -- the gateway is stateless and has no cookie to set;
            forwarding one from the Space would create a session the
            architecture does not have (`docs/DEPLOYMENT_ARCHITECTURE.md` §2.2
            excludes persistence).
          * **every CORS header** (`access-control-*`). This is the fix for
            F-2, and it is a security rule, not hygiene. Until 2026-09-22 the
            CORS decision was made only on the *request* leg
            (`build_cors_headers`, via `admit()`) and the *response* leg
            rebuilt the upstream's headers wholesale -- so a CORS header the
            Space emitted on its own was relayed to the browser verbatim, on
            top of the gateway's allowlist answer.

            The bypass was measured, not theorised. With the allowlist set to
            `["https://app.example.com"]` and the Space answering with
            `Access-Control-Allow-Origin: *`:

              * `GET /v1/health` from `https://evil.example.net` returned
                **two** values for `access-control-allow-origin`, `*` and the
                allowlisted origin;
              * `POST /v1/analyze` from the same disallowed origin returned
                `200` with `Access-Control-Allow-Origin: *` and
                `Access-Control-Allow-Credentials: true`.

            Both are the failure `docs/DEPLOYMENT_ARCHITECTURE.md` §2.1 exists
            to prevent: the allowlist is bypassed by a header the gateway
            never inspected. The `*` case is worse than a permissive echo,
            because a duplicated `ACAO` is not a value a browser can match to
            an allowlisted origin -- it makes the gateway's own correct
            headers unreliable for allowed callers while the `*` still
            admits everyone.

            Stripping on the response leg means the browser sees *only* what
            `build_cors_headers` decided. The upstream is not trusted to make
            a CORS decision on the gateway's behalf, which is the same
            principle already applied to `authorization` and `set-cookie`.
        """
        blocked = frozenset(
            {
                "content-length",
                "content-encoding",
                "authorization",
                "www-authenticate",
                "proxy-authorization",
                "set-cookie",
                "cookie",
            }
        )
        return {
            key: value
            for key, value in upstream.items()
            if key.lower() not in DEFAULT_HOP_BY_HOP
            and key.lower() not in blocked
            # F-2. A prefix rule, not an enum: RFC 6648 discourages new `Access-`
            # headers, but the CORS family has grown (`-Allow-Credentials`,
            # `-Expose-Headers`, `-Max-Age`, `-Allow-Methods`, `-Allow-Headers`)
            # and an enum would silently miss whichever is added next. Nothing
            # the Space may legitimately return starts with this prefix, because
            # the CORS answer is the gateway's to give.
            and not key.lower().startswith("access-control-")
        }


# --------------------------------------------------------------------------
# Body validation (no framework needed)
# --------------------------------------------------------------------------


def validate_analyze_body(raw: bytes) -> tuple[dict[str, Any] | None, tuple[int, dict[str, Any]] | None]:
    """Parse and minimally validate a JSON `POST /v1/analyze` body.

    Returns `(parsed, None)` on success or `(None, (status, error_body))` on
    failure. The schema's own validation is authoritative (Pydantic,
    `extra="forbid"`); this function's job is to answer cheaply, before
    importing the serving stack, so that a malformed body never costs a model
    load.

    Deliberately NOT a re-implementation of `AnalysisRequest`: it checks only
    what the contract's documented error envelope can carry. Two obligations it
    DOES take on, because forwarding either would cost a round trip (and
    possibly GPU quota) for a request the Space will certainly reject:

      * every field the contract marks required is present and well-typed;
      * no field outside the model's own field set is present, because
        `AnalysisRequest` uses `extra="forbid"`.

    The unknown-field check is derived from `AnalysisRequest.model_fields`
    rather than from a hard-coded list, so adding a field to the schema does not
    require editing the gateway. `tests/unit/test_gateway_policy.py`
    cross-checks this function's verdicts against the real model in BOTH
    directions, which is what caught the originally-missing unknown-field rule.
    """
    known_fields = _analyze_request_fields()

    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, (400, translate_error("input_error", "The request body is not valid JSON.", detail=str(exc))[1])

    if not isinstance(parsed, dict):
        return None, (
            422,
            translate_error(
                "invalid_request",
                "The request body must be a JSON object.",
                detail=f"got {type(parsed).__name__}",
            )[1],
        )

    # `extra="forbid"` on the model means an unknown key is a 422, not a
    # silently ignored field. Checking here keeps the rejection cheap.
    unknown = sorted(set(parsed) - known_fields)
    if unknown:
        return None, (
            422,
            translate_error(
                "invalid_request",
                "The request body contains unknown fields.",
                detail=f"unknown: {', '.join(unknown)}; allowed: {', '.join(sorted(known_fields))}",
            )[1],
        )

    assets = parsed.get("assets")
    if not isinstance(assets, list) or not assets:
        return None, (
            422,
            translate_error(
                "invalid_request",
                "`assets` must be a non-empty array of asset handles.",
                detail=f"assets={assets!r}",
            )[1],
        )
    if not all(isinstance(item, str) and item for item in assets):
        return None, (
            422,
            translate_error(
                "invalid_request",
                "Every entry in `assets` must be a non-empty string handle.",
            )[1],
        )
    if "query" not in parsed or not isinstance(parsed["query"], str):
        return None, (
            422,
            translate_error(
                "invalid_request",
                "`query` is required and must be a string.",
                detail=f"query={parsed.get('query')!r}",
            )[1],
        )

    # An unsupported enum value is a 422, not a forward. `docs/API_CONTRACT.md`
    # section 5.1 maps "wrong enum value" to 422 alongside the unknown-field
    # case, and section 2.4 documents `force_task` as one of the `Task` values.
    #
    # This check was MISSING and the omission was not harmless: a body with
    # `force_task: "nonsense"` was forwarded to the Space, whose schema rejected
    # it -- so the client received a 502/upstream error for a defect entirely
    # local to the request. That both mis-states the fault and spends a GPU-quota
    # round trip on a body the Space cannot accept.
    #
    # The accepted set is derived from the `Task` enum, not hard-coded, for the
    # same reason as the field set above: adding a task must not require editing
    # the gateway. An absent or null `force_task` is legal (the field is
    # optional, `core/schemas.py`).
    force_task = parsed.get("force_task")
    if force_task is not None:
        if not isinstance(force_task, str):
            return None, (
                422,
                translate_error(
                    "invalid_request",
                    "`force_task` must be a string or null.",
                    detail=f"force_task={force_task!r}",
                )[1],
            )
        allowed = _task_values()
        if allowed is not None and force_task not in allowed:
            return None, (
                422,
                translate_error(
                    "invalid_request",
                    "`force_task` is not a recognised task.",
                    detail=(
                        f"force_task={force_task!r}; "
                        f"allowed: {', '.join(sorted(allowed))}"
                    ),
                )[1],
            )

    run_id = parsed.get("run_id")
    if run_id is not None and not isinstance(run_id, str):
        return None, (
            422,
            translate_error(
                "invalid_request",
                "`run_id` must be a string or null.",
                detail=f"run_id={run_id!r}",
            )[1],
        )

    return parsed, None


def _task_values() -> frozenset[str] | None:
    """The values `Task` accepts, derived from the enum itself.

    Returns `None` when the schema module cannot be imported, and the caller
    then SKIPS the enum check rather than guessing.

    That asymmetry is deliberate. An unknown-field check needs a field set, and
    the documented four field names are a short, stable list that has not changed
    since the contract was written. The `Task` values are a different case: they
    are the router's vocabulary and the schema is explicit that it may grow
    (`core/schemas.py`). A stale copy here would silently reject a newly-added
    task at the gateway, before the Space could accept it -- a gate that fails
    closed on valid input, which is worse than no gate. The correct degradation
    is to forward, since the Space's own Pydantic model is authoritative anyway.

    The cross-check test asserts the primary path is taken, so this fallback
    cannot quietly become the normal path.
    """
    try:
        from core.schemas import Task

        return frozenset(member.value for member in Task)
    except Exception:  # pragma: no cover - only in a broken install
        return None


def _analyze_request_fields() -> frozenset[str]:
    """The field names `AnalysisRequest` accepts.

    Imported lazily so that importing `gateway.policy` does not pull the schema
    module (which imports Pydantic) into a process that only needs the size
    check. Falls back to the documented set if the import is unavailable, so a
    gateway can still refuse an obviously mistyped request -- but the fallback is
    a last resort, and the cross-check test asserts the primary path is used.
    """
    try:
        from core.schemas import AnalysisRequest

        return frozenset(AnalysisRequest.model_fields)
    except Exception:  # pragma: no cover - only in a broken install
        return frozenset({"assets", "query", "force_task", "run_id"})
