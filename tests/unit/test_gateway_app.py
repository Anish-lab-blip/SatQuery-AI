"""The gateway ASGI layer: its contract, its allowlist, and real HTTP execution.

SUPERSEDED PREMISE -- READ THIS FIRST
-------------------------------------
An earlier revision of this file asserted that the ASGI layer **could not run at
all** here, because a WDAC policy blocked `orjson.pyd` (`OSError [WinError
4551]`) and that `ImportError` escaped FastAPI's `except ModuleNotFoundError`
guard. That was true when it was written. It is **no longer true**: in the
current environment `import fastapi` succeeds (0.141.1), the whole web stack
imports, and `create_app()` builds a working application.

The premise did not merely go stale -- it was hiding four real defects, because
no test could reach the code they lived in. Those defects are recorded in
`docs/STEP7_BACKEND_CHAIN_REPORT.md` and fixed in the same change that rewrote
this docstring. The lesson is worth stating plainly: a documented blocker is a
place where evidence stops, and "documented" is not the same as "verified".

WHAT THIS FILE NOW DOES
-----------------------
1. Asserts the module still imports and does not drag the model stack in
   (unchanged, and still load-bearing).
2. Asserts the route allowlist is exactly the contract's (unchanged).
3. **Drives real HTTP requests through the real ASGI application** using
   httpx's `ASGITransport`. This is not a mock: Starlette's routing, dependency
   resolution, `Request` parsing, JSONResponse serialisation and the CORS
   middleware all execute. What it does NOT do is bind a socket or cross the
   network, so it is labelled `integration` rather than `smoke`.

WHY THE NEW TESTS ARE REGRESSION GUARDS, NOT DECORATION
-------------------------------------------------------
Each one pins a defect that was live in this repository and invisible:
  * `Request` did not resolve, so every POST was answered by FastAPI's body
    validation with `{"detail": ...}` and never reached a handler.
  * A bad `force_task` was forwarded upstream instead of refused locally.
  * An empty `HF_TOKEN` produced `Authorization: Bearer ` and a
    `LocalProtocolError` that was reported as an upstream failure.
  * A non-JSON upstream error body was passed through verbatim, contract be
    damned -- which leaked an internal `os error 10061` to the client.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import gateway.app as app_module
from gateway.policy import GatewayConfig

REPO = Path(__file__).resolve().parents[2]


def _code_only(source: str) -> str:
    """The source with every docstring removed.

    Several assertions in this file ask "is this behaviour still live?" by
    searching the module text. That question is about *code*, but these modules
    deliberately document the behaviour they retired -- so a raw substring check
    reports the explanation as if it were the implementation. Stripping
    docstrings first makes the search mean what it says.

    Kept as a local copy rather than imported from `test_deployment_adapter`:
    that module imports the adapter, which imports the registry, and pulling
    that chain into every gateway test run costs more than ten duplicated lines.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)

#: A routable-looking origin for the upstream. Nothing listens on port 9
#: ("discard"), which is the point: the upstream legs of these tests are
#: *expected* to fail, and the assertion is about how the gateway reports that.
UPSTREAM = "http://127.0.0.1:9"
ORIGIN = "https://frontend.example.com"


def _app():
    """Build the real ASGI app, or skip honestly if the stack is unavailable.

    The skip is not silent: it carries the reason, so a run in a
    policy-blocked environment reports "the web stack could not be imported"
    rather than the misleading "no tests for this area".
    """
    pytest.importorskip("fastapi", reason="the web stack is unavailable here")
    from gateway.app import create_app

    return create_app(
        GatewayConfig(upstream_url=UPSTREAM, allowed_origins=(ORIGIN,))
    )


def _client(app):
    """An httpx client wired directly to the ASGI app -- no socket, no network."""
    import httpx

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gw.test"
    )


def _run(coro_fn):
    import anyio

    return anyio.run(coro_fn)


def _raw_post(app, path, *, body=b"", headers=(), content_length=None, ip="203.0.113.7"):
    """POST through the ASGI app with FULL control over the headers.

    `httpx` will not send a body without a `Content-Length`, which is precisely
    the case F-6 is about, so this bypasses it and drives the ASGI interface
    directly. `content_length=None` omits the header entirely -- the hostile
    case. Returns `(status_code, parsed_json_or_raw_bytes)`.
    """
    import anyio

    hdrs = [
        (b"host", b"gw.test"),
        (b"origin", ORIGIN.encode()),
        (b"content-type", b"image/tiff"),
        *[(k.encode(), v.encode()) for k, v in headers],
    ]
    if content_length is not None:
        hdrs.append((b"content-length", str(content_length).encode()))

    delivered = {"i": 0}
    chunk = 1024 * 1024

    async def receive():
        remaining = len(body) - delivered["i"]
        if remaining <= 0:
            return {"type": "http.request", "body": b"", "more_body": False}
        take = min(chunk, remaining)
        piece = body[delivered["i"] : delivered["i"] + take]
        delivered["i"] += take
        return {
            "type": "http.request",
            "body": piece,
            "more_body": delivered["i"] < len(body),
        }

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": hdrs,
        "client": (ip, 51000),
        "server": ("gw.test", 80),
        "scheme": "http",
    }
    out = {"status": None, "chunks": []}

    async def send(message):
        if message["type"] == "http.response.start":
            out["status"] = message["status"]
        elif message["type"] == "http.response.body":
            out["chunks"].append(message.get("body", b""))

    async def go():
        await app(scope, receive, send)

    anyio.run(go)

    raw = b"".join(out["chunks"])
    try:
        import json as _json

        return out["status"], _json.loads(raw)
    except Exception:
        return out["status"], raw


def _raw_peak_post(app, path, *, body=b"", headers=(), content_length=None, ip="203.0.113.7"):
    """`_raw_post`, plus the peak traced allocation for that single request.

    F-9 for the gateway. `_raw_post` returns a status code, and a status code
    cannot distinguish "refused while reading" from "refused after buffering" --
    both are 413. The peak allocation is the observable that separates them, so
    a caller can assert the handler never held the body in full.

    This is a separate helper rather than a change to `_raw_post` so the
    existing F-6 callers keep their exact return shape and their measurements
    stay comparable with the ones already recorded in the report.
    """
    import tracemalloc

    tracemalloc.start()
    try:
        status, parsed = _raw_post(
            app,
            path,
            body=body,
            headers=headers,
            content_length=content_length,
            ip=ip,
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return status, parsed, peak


# ===========================================================================
# 1. Import discipline (no web stack, no torch)
# ===========================================================================
def test_the_package_imports_without_the_web_stack() -> None:
    """`gateway` and `gateway.policy` must not require FastAPI or torch.

    This is what makes the policy layer testable in an environment where the ASGI
    framework is unavailable -- and, in production, what keeps the gateway's
    validation independent of the model stack.

    Note the scope: this asserts `gateway.policy` is separable. It does NOT
    claim `gateway.app` is unimportable; see the ASGI section below, which now
    imports it for real.
    """
    import subprocess
    import sys

    script = (
        "import sys; import gateway, gateway.policy; "
        "print('fastapi' in sys.modules, 'torch' in sys.modules, "
        "'starlette' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        check=True,
    )
    fastapi_present, torch_present, starlette_present = result.stdout.split()
    assert fastapi_present == "False", "importing gateway pulled in FastAPI"
    assert torch_present == "False", "importing gateway pulled in torch"
    assert starlette_present == "False", "importing gateway pulled in starlette"


def test_the_app_module_imports_and_degrades_when_fastapi_is_unavailable() -> None:
    """A blocked framework must not look like a bug in this repository.

    `gateway.app` is imported successfully; `create_app` remains callable; only
    the eagerly-built module-level `app` may be None. An operator can therefore
    tell "this environment cannot import FastAPI" from "this code is broken".
    """
    assert callable(app_module.create_app), "create_app must remain importable"
    assert app_module.app is None or hasattr(app_module.app, "routes")


# ===========================================================================
# 2. The route allowlist
# ===========================================================================
def test_the_proxied_route_allowlist_is_exactly_the_contract() -> None:
    """The gateway forwards the contract's four paths and nothing else.

    An allowlist rather than a passthrough: a gateway that forwarded arbitrary
    paths would expose every route the Space serves, including ones the contract
    does not document and the frontend was never told about.

    Four now, not three: the owner ruling of 2026-09-22 chose Option A for
    `POST /v1/assets` (out-of-band upload with opaque ephemeral handles), so the
    upload path moved out of the blocked set and into the proxied one.
    """
    assert app_module.PROXIED_ROUTES == (
        "/v1/health",
        "/v1/capabilities",
        "/v1/analyze",
        "/v1/assets",
    )
    assert app_module.BLOCKED_ROUTES == (), (
        "a route is still blocked, but the ruling closed the only open decision"
    )

    from core.schemas import SCHEMA_VERSION

    assert SCHEMA_VERSION  # the schema is importable; the paths are the contract's

    contract = (REPO / "docs" / "API_CONTRACT.md").read_text(encoding="utf-8")
    for path in app_module.PROXIED_ROUTES:
        assert path in contract, (
            f"the gateway proxies {path}, which the contract does not document"
        )


def test_the_upload_route_is_proxied_and_is_not_free() -> None:
    """`/v1/assets` is proxied, and it is treated as a costly route.

    The ruling did not make upload free. It is the one endpoint that accepts a
    request *body* of unbounded size and writes to disk, so it belongs in
    `COSTLY_ROUTES` alongside `/v1/analyze` -- the set that gets the read
    timeout, the body limit and the upload-specific error translation. Proxying
    it without that would hand an unauthenticated caller a disk-filling primitive
    (`API_CONTRACT.md` section 7: there is no auth in v1).
    """
    assert "/v1/assets" in app_module.PROXIED_ROUTES
    assert "/v1/assets" not in app_module.BLOCKED_ROUTES
    assert "/v1/assets" in app_module.COSTLY_ROUTES

    source = (REPO / "gateway" / "app.py").read_text(encoding="utf-8")
    # Docstrings are stripped before the search: the module deliberately
    # *documents* the retired 501 so a reader can find the history, and a raw
    # substring check would flag that explanation as a live code path. The
    # question is whether any handler still returns 501, not whether the word
    # appears.
    assert "501" not in _code_only(source), (
        "the gateway still returns 501 somewhere; nothing is blocked any more"
    )
    assert "docs/API_CONTRACT.md" in source or "API_CONTRACT.md" in source, (
        "the retired decision is no longer cited anywhere in the gateway, so a "
        "reader cannot find why the route changed"
    )


def test_create_app_refuses_an_incoherent_config() -> None:
    """A bad environment fails at startup, not on the first request.

    `GatewayConfig.__post_init__` validates, and `create_app` builds one eagerly.
    A gateway that started with, say, a wildcard CORS policy would serve traffic
    before anyone noticed.
    """
    with pytest.raises(ValueError):
        GatewayConfig(upstream_url="https://s.hf.space", allowed_origins=("*",))
    with pytest.raises(ValueError):
        GatewayConfig(upstream_url="", allowed_origins=("https://a.dev",))


def test_the_app_source_does_not_reimplement_policy_rules() -> None:
    """The app must call `policy`, not restate its rules.

    If a rule existed in both places it would drift, and the tested copy would be
    the one that is wrong. This asserts the app defers to `admit()` and does not
    carry its own origin/size/rate logic.

    The CORS assertion was rewritten on 2026-09-22. It previously read

        assert "Access-Control-Allow-Origin" not in source

    and the problem is that this is not a test of the app at all: the string
    occurred nowhere in `gateway/app.py`, so the assertion was a hardcoded
    `True` that could only ever fail by accident -- which is exactly what
    happened when a comment explaining the CORS fix (F-2) mentioned the header
    by name. It was sensitive to prose and insensitive to the defect it named:
    at the time it was "passing", `_proxy` was relaying the upstream's CORS
    headers to every origin, which is the duplication this file's name suggests
    it guards against.

    What is asserted now is the property that actually matters and is actually
    checkable here: the CORS *decision* comes from `policy`, and the app never
    constructs a CORS value of its own. Whether the upstream's headers are
    filtered is asserted at the response level, end to end, by
    `test_an_upstream_cors_header_cannot_bypass_the_allowlist`.
    """
    source = (REPO / "gateway" / "app.py").read_text(encoding="utf-8")
    assert "policy.admit(" in source, "the app does not call policy.admit()"
    assert "validate_analyze_body(" in source, (
        "the app does not use the shared body validator"
    )
    # These would indicate duplicated policy.
    assert "RateLimiter(" not in source, "the app constructs its own rate limiter"
    assert "policy.response_headers(" in source, (
        "the app no longer filters upstream response headers through policy"
    )
    # The app may *mention* CORS headers (its comments explain the F-2 fix), but
    # it must never assign one. The distinction is the point: prose is allowed,
    # a second decision is not.
    code = _code_only(source)
    assert "Access-Control-Allow-Origin" not in code, (
        "the app assigns a CORS header instead of taking it from policy"
    )
    assert "access-control-" not in code.lower(), (
        "the app builds an access-control-* header itself instead of using policy"
    )


# ===========================================================================
# 3. REAL ASGI EXECUTION  (integration)
# ===========================================================================
pytestmark_asgi = pytest.mark.integration


@pytest.mark.integration
def test_the_real_route_table_is_registered() -> None:
    """The app registers exactly the contract's routes.

    This is the assertion the old docstring said was impossible. It is the
    weakest useful statement about the ASGI layer: the routes exist.
    """
    app = _app()
    table = {}
    for route in app.routes:
        path = getattr(route, "path", None)
        if path:
            table[path] = set(getattr(route, "methods", None) or [])

    for path in app_module.PROXIED_ROUTES:
        assert path in table, f"{path} is proxied by policy but not registered"
    assert "POST" in table["/v1/analyze"]
    assert "GET" in table["/v1/health"]
    assert "GET" in table["/v1/capabilities"]
    # The upload route is registered so it can return 501 rather than 404.
    assert "/v1/assets" in table


@pytest.mark.integration
def test_request_is_injected_and_not_read_as_a_query_parameter() -> None:
    """REGRESSION GUARD: `request: Request` must bind the ASGI request.

    The defect this pins: `gateway/app.py` uses `from __future__ import
    annotations`, so `request: Request` is the *string* `'Request'`. FastAPI
    resolves annotations via `eval(annotation, func.__globals__)`. `Request` was
    imported inside `create_app`, so it was never in the module globals,
    resolution failed, and FastAPI **silently** reinterpreted the parameter as a
    required *query* parameter named `request`.

    The symptom was that every POST answered `422 {"detail":[{"loc":["query",
    "request"],"msg":"Field required"}]}` -- wrong status, wrong shape, and the
    handler never ran. Nothing failed loudly.

    Asserting the symptom would be brittle. This asserts the cause: the route's
    dependency tree must have no query parameters, and the type must resolve to a
    real class rather than a `ForwardRef`.
    """
    app = _app()
    route = next(r for r in app.routes if getattr(r, "path", None) == "/v1/analyze")
    dependant = route.dependant

    assert list(dependant.query_params) == [], (
        "the handler has query parameters; `Request` did not resolve and FastAPI "
        "fell back to treating it as a query string argument"
    )

    # Positive control: the annotation must resolve to starlette's Request class.
    from starlette.requests import Request as StarletteRequest

    assert app_module.Request is StarletteRequest, (
        "gateway.app must bind Request at module scope, or FastAPI cannot resolve "
        "the string annotation"
    )


@pytest.mark.integration
def test_a_malformed_analyze_body_is_refused_with_the_contract_error_envelope() -> None:
    """A bad body returns `{"error": {...}}`, not FastAPI's `{"detail": ...}`.

    `docs/API_CONTRACT.md` section 5 fixes the error shape. FastAPI's default
    validation envelope has a completely different shape, so a client written to
    the contract would fail to parse it -- exactly when it most needs to.

    This test would have failed on all four inputs before the fix, with `422`
    and a `detail` key.
    """
    app = _app()

    async def body():
        async with _client(app) as client:
            out = {}
            for label, payload in [
                ("empty_assets", {"assets": [], "query": "x"}),
                ("unknown_field", {"assets": ["a"], "query": "x", "bogus": 1}),
                ("missing_query", {"assets": ["a"]}),
                ("non_object", [1, 2, 3]),
            ]:
                r = await client.post(
                    "/v1/analyze",
                    content=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json", "Origin": ORIGIN},
                )
                out[label] = r
            r = await client.post(
                "/v1/analyze",
                content=b"{not json",
                headers={"Content-Type": "application/json", "Origin": ORIGIN},
            )
            out["malformed_json"] = r
            return out

    responses = _run(body)

    for label, r in responses.items():
        assert r.status_code in (400, 422), f"{label}: unexpected {r.status_code}"
        parsed = r.json()
        assert "error" in parsed, (
            f"{label}: response is not the contract envelope; got keys {sorted(parsed)}"
        )
        assert "detail" not in parsed, (
            f"{label}: FastAPI's raw validation envelope leaked to the client"
        )
        err = parsed["error"]
        for field in ("code", "message", "recoverable", "request_id"):
            assert field in err, f"{label}: error envelope missing {field!r}"


@pytest.mark.integration
def test_an_unsupported_enum_value_is_refused_locally_not_forwarded() -> None:
    """REGRESSION GUARD: a bad `force_task` is a local 422, not an upstream trip.

    Before the fix, `force_task: "nonsense"` was forwarded. The Space rejected it
    and the gateway reported the failure as an upstream problem -- mis-stating
    the fault and spending a round trip (potentially GPU quota) on a body that
    could never be accepted.

    Note the third case: a bad enum must NOT become a 502. That distinction is
    the whole point -- the client's request is wrong, not the service.
    """
    app = _app()

    async def body():
        async with _client(app) as client:
            out = {}
            for label, payload in [
                ("unknown_task", {"assets": ["a"], "query": "x", "force_task": "nonsense"}),
                ("non_string_task", {"assets": ["a"], "query": "x", "force_task": 5}),
                ("non_string_run_id", {"assets": ["a"], "query": "x", "run_id": 7}),
            ]:
                r = await client.post(
                    "/v1/analyze",
                    content=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json", "Origin": ORIGIN},
                )
                out[label] = r
            return out

    for label, r in _run(body).items():
        assert r.status_code == 422, f"{label}: expected a local 422, got {r.status_code}"
        assert r.json()["error"]["code"] == "invalid_request"


@pytest.mark.integration
def test_a_legitimate_request_reaches_the_upstream_leg() -> None:
    """A well-formed body is NOT refused locally; it is forwarded.

    The complement of the enum test: the gateway must not become so eager to
    refuse that it blocks valid traffic. The upstream here is unreachable, so
    the correct outcome is an upstream-fault error -- not a validation error.
    """
    app = _app()

    async def body():
        async with _client(app) as client:
            return await client.post(
                "/v1/analyze",
                content=json.dumps(
                    {"assets": ["a", "b"], "query": "how did it change?", "force_task": None}
                ).encode(),
                headers={"Content-Type": "application/json", "Origin": ORIGIN},
            )

    r = _run(body)
    assert r.status_code != 422, (
        "a well-formed request was refused locally; the validator is over-strict"
    )
    assert r.status_code not in (400,), "a well-formed request was called malformed"


@pytest.mark.integration
def test_a_non_json_upstream_error_never_reaches_the_client_verbatim() -> None:
    """REGRESSION GUARD: a non-JSON upstream body is translated, not relayed.

    The defect: the guard read `upstream.status_code < 400`, so a non-JSON 4xx/5xx
    was passed through with its original content-type and content. That breaks
    the contract's error shape AND relays whatever the upstream said.

    Found by accident and worth recording exactly how: this sandbox routes
    egress through a proxy which answers `502` with the plain text
    `upstream connect failed: ... (os error 10061)`. The gateway handed that
    straight to the "client", disclosing an internal errno. The fix covers all
    statuses.

    Asserted here: for a non-JSON upstream response, the client receives
    `application/json` and a contract envelope, and the upstream's text does not
    appear.
    """
    app = _app()

    async def body():
        async with _client(app) as client:
            return await client.post(
                "/v1/analyze",
                content=json.dumps({"assets": ["a"], "query": "x"}).encode(),
                headers={"Content-Type": "application/json", "Origin": ORIGIN},
            )

    r = _run(body)
    assert "application/json" in r.headers.get("content-type", ""), (
        "a non-JSON upstream response reached the client with a non-JSON "
        "content-type, so a contract-conformant client cannot parse the error"
    )
    parsed = r.json()
    assert "error" in parsed, "the translated upstream failure is not the contract envelope"
    # The upstream's own wording must not be relayed.
    assert "os error" not in r.text.lower(), (
        "the upstream's internal errno text was relayed to the client"
    )


@pytest.mark.integration
def test_an_oversized_body_is_refused_even_when_content_length_is_omitted() -> None:
    """F-6. The body-size cap must not depend on a client-supplied header.

    `policy.admit` step 3 refuses an oversized body from the `Content-Length`
    HEADER, because it runs before the body is read and the header is the only
    evidence available. That makes the cap declarative: a client that omits the
    header was never measured.

    Measured through this stack before the fix, cap 8 MiB, body 12 MiB:
    `Content-Length` omitted returned 502 `model_unavailable` (the request
    reached the upstream call) with a peak allocation of 13.9 MiB -- i.e. the
    whole body was buffered past the cap. Allocation then tracked body size
    exactly with no ceiling (1/8/16/32/64 MiB in -> 3.0/8.1/16.0/32.0/64.0 MiB
    allocated), so the header check bounds nothing in the hostile case.

    This test pins the FIX rather than the symptom: both cases must produce the
    contract's `oversized_image` envelope. The distinction the assertions draw
    is that the omitted-header case must be refused *by the same code* -- if a
    future edit reinstates `await request.body()`, the omitted case reverts to
    502 and this test fails.
    """
    cap = GatewayConfig(upstream_url=UPSTREAM, allowed_origins=(ORIGIN,)).max_body_bytes
    body = b"x" * (cap + (2 * 1024 * 1024))

    app = _app()
    declared, declared_json = _raw_post(
        app, "/v1/assets", body=body, content_length=len(body)
    )
    omitted, omitted_json = _raw_post(app, "/v1/assets", body=body)

    assert declared == 413, (
        f"a declared-oversized body was not refused: got HTTP {declared}. The "
        f"header path is the one that already worked; if this fails, the policy "
        f"ladder itself has moved."
    )
    assert omitted == 413, (
        f"an oversized body with NO Content-Length got HTTP {omitted}, not 413. "
        f"The cap is being skipped when the header is absent -- the body is "
        f"buffered in full before any check runs. Do not fix this by reordering "
        f"`policy.admit`; it cannot see a body it has not read. Fix it by "
        f"enforcing the cap while reading (`_read_body_bounded` in gateway/app.py)."
    )
    for label, parsed in (("declared", declared_json), ("omitted", omitted_json)):
        assert isinstance(parsed, dict) and parsed.get("error", {}).get("code") == (
            "oversized_image"
        ), (
            f"the {label} case did not return the contract's oversized_image "
            f"envelope: {parsed!r}"
        )
    assert "enforced while reading" in omitted_json["error"]["detail"], (
        "the omitted case was refused by something other than the streaming cap; "
        "the detail should name the enforcement point, because a client that "
        "omitted the header needs to be told the limit does not depend on it"
    )


def test_a_body_within_the_cap_is_still_forwarded_when_content_length_is_omitted() -> None:
    """The streaming cap must not refuse a legitimate body.

    A negative control for the test above: if `_read_body_bounded` were
    off-by-one, or compared the running total wrongly, every upload would become
    a 413 and the previous test would still pass. This asserts the other side of
    the boundary, and pins it at the EXACT edge -- `cap` bytes must pass,
    `cap + 1` must not.
    """
    cfg = GatewayConfig(upstream_url=UPSTREAM, allowed_origins=(ORIGIN,))
    cap = cfg.max_body_bytes
    app = _app()

    at_cap, _ = _raw_post(app, "/v1/assets", body=b"x" * cap)
    over_cap, _ = _raw_post(app, "/v1/assets", body=b"x" * (cap + 1))

    assert at_cap != 413, (
        f"a body of exactly {cap} bytes (the configured cap) was refused; the "
        f"boundary is exclusive when it should be inclusive"
    )
    assert over_cap == 413, (
        f"a body of {cap + 1} bytes (one over the cap) was NOT refused; the "
        f"streaming cap is off by at least one byte"
    )


def test_the_upload_route_forwards_its_body_and_reports_an_upstream_failure() -> None:
    """`/v1/assets` proxies the body and returns the contract envelope, not 501.

    This replaces `test_the_blocked_upload_route_answers_501_with_the_envelope`.
    That test asserted the pre-ruling truth; it is not relaxed here, it is
    inverted, because the ruling chose Option A. The upstream is unreachable in
    this sandbox, so the observable outcome is a *translated upstream failure* --
    which is still the right thing to assert: it proves the gateway read the
    body, issued a POST rather than falling through to 501, and wrapped the
    failure in the contract envelope rather than relaying transport text.
    """
    app = _app()

    async def body():
        async with _client(app) as client:
            return await client.post(
                "/v1/assets",
                content=b"\\x89PNG\\r\\n",
                headers={"Content-Type": "image/png", "Origin": ORIGIN},
            )

    r = _run(body)
    assert r.status_code != 501, (
        "the upload route is still blocked; the ruling moved it into the "
        "proxied allowlist"
    )
    assert "application/json" in r.headers.get("content-type", ""), (
        "an upload failure did not come back as JSON, so a contract-conformant "
        "client cannot parse it"
    )
    parsed = r.json()
    assert "error" in parsed, "the upload failure is not the contract envelope"
    assert "os error" not in r.text.lower(), (
        "the upstream's internal errno text was relayed to the client"
    )


@pytest.mark.integration
def test_the_gateway_health_route_never_touches_the_upstream() -> None:
    """The gateway's own liveness probe is answered locally.

    It exists precisely so that "the gateway is down" and "the upstream is down"
    are distinguishable. If it proxied, it could not distinguish them.
    """
    app = _app()

    async def body():
        async with _client(app) as client:
            return await client.get("/v1/gateway/health")

    r = _run(body)
    assert r.status_code == 200
    parsed = r.json()
    assert parsed["role"] == "gateway"
    assert parsed["upstream_configured"] is True
    assert set(parsed["proxied_routes"]) == set(app_module.PROXIED_ROUTES)
    # The response must not claim the upstream is healthy.
    assert "note" in parsed


# ===========================================================================
# 5. The error envelope holds for framework-level responses too  (F-3)
# ===========================================================================
@pytest.mark.integration
def test_every_non_2xx_carries_the_contract_envelope_including_404_and_405() -> None:
    """REGRESSION GUARD: "every non-2xx" has to include the ones no handler owns.

    `docs/API_CONTRACT.md` section 5 says "**Every** non-2xx response body has
    this shape". The existing envelope test
    (`test_a_malformed_analyze_body_is_refused_with_the_contract_error_envelope`)
    covers the paths the *handlers* own, because those are the paths it can reach
    by sending a body. Two responses never reach a handler, and both are raised
    by Starlette with its own `{"detail": ...}` body:

      * **404** -- a path that is not registered (a trailing slash, a typo);
      * **405** -- a registered path with the wrong method, e.g. `GET /v1/assets`,
        which the contract documents while defining only `POST`.

    Measured before the fix: `GET /v1/whocares` -> `404 {"detail":"Not Found"}`
    and `GET /v1/assets` -> `405 {"detail":"Method Not Allowed"}`, while
    `GET /v1/health` correctly returned `{"error": {"code": ...}}`. A client
    written to the contract would raise `KeyError: 'error'` precisely when it
    was trying to render a failure.

    The 404/405 cases are asserted together with a 400 on the same client, so
    the test cannot pass by the whole response path being broken.
    """
    app = _app()

    async def body():
        async with _client(app) as client:
            out = {}
            out["not_found"] = await client.get(
                "/v1/whocares", headers={"Origin": ORIGIN}
            )
            out["wrong_method"] = await client.get(
                "/v1/assets", headers={"Origin": ORIGIN}
            )
            out["handler_owned"] = await client.post(
                "/v1/analyze",
                content=b"{not json",
                headers={"Content-Type": "application/json", "Origin": ORIGIN},
            )
            return out

    responses = _run(body)

    assert responses["not_found"].status_code == 404, (
        "the unregistered-path case is not a 404 any more, so this test is no "
        "longer exercising the branch it was written for"
    )
    assert responses["wrong_method"].status_code == 405, (
        "GET /v1/assets is no longer a 405; the method-mismatch branch is untested"
    )

    for label, r in responses.items():
        parsed = r.json()
        assert "error" in parsed, (
            f"{label}: a non-2xx is not the contract envelope; got keys "
            f"{sorted(parsed)} -- FastAPI's raw `detail` shape leaked"
        )
        assert "detail" not in parsed, (
            f"{label}: FastAPI's raw validation envelope leaked to the client"
        )
        err = parsed["error"]
        for field in ("code", "message", "recoverable"):
            assert field in err, f"{label}: error envelope missing {field!r}"

    # The codes must come from the documented taxonomy, not be invented here.
    contract = (REPO / "docs" / "API_CONTRACT.md").read_text(encoding="utf-8")
    for label in ("not_found", "wrong_method"):
        code = responses[label].json()["error"]["code"]
        assert f"`{code}`" in contract, (
            f"{label}: code {code!r} is not in the contract's section 5.2 taxonomy"
        )


# ===========================================================================
# 6. Trailing-slash behaviour is documented, so it must be pinned
# ===========================================================================
@pytest.mark.integration
def test_a_trailing_slash_redirects_to_the_gateways_own_host() -> None:
    """REGRESSION GUARD: the contract now documents this, so it must be true.

    `docs/API_CONTRACT.md` §5.1 gained a note on 2026-09-22 stating that a
    trailing slash answers `307` and that the `Location` is built from the
    **gateway's own host** rather than the client's request URL. That second
    half is the part worth pinning: it was measured, not assumed, and it is the
    kind of statement that becomes false silently if Starlette's
    `redirect_slashes` behaviour or the app's mount point ever changes.

    The first draft of that contract note claimed a trailing slash was a `404`.
    The probe written to confirm it showed `307`. Recording the correction here
    as a test is deliberate: the contract is the artefact a client author reads,
    and an unverified sentence in it is worse than no sentence.

    `follow_redirects=False` is essential -- httpx follows by default, which
    would hide the status this test exists to assert.
    """
    app = _app()

    async def body():
        async with _client(app) as client:
            return await client.get("/v1/analyze/", follow_redirects=False)

    r = _run(body)

    assert r.status_code == 307, (
        f"a trailing slash no longer 307s (got {r.status_code}); the contract's "
        "§5.1 note is now wrong"
    )
    location = r.headers.get("location", "")
    assert location.endswith("/v1/analyze"), (
        f"the redirect does not target the unslashed path: {location!r}"
    )
    # The host is the gateway's, taken from the client's base_url -- NOT derived
    # from an Origin/Referer the caller controls. Asserted so a future change to
    # an absolute-from-request-URL redirect cannot pass unnoticed.
    assert "gw.test" in location, (
        "the Location is no longer built from the request's own host; the "
        f"contract note in §5.1 needs revisiting (got {location!r})"
    )


# ===========================================================================
# 4. CORS on the RESPONSE leg  (F-2)
# ===========================================================================
@pytest.mark.integration
def test_an_upstream_cors_header_cannot_bypass_the_allowlist() -> None:
    """REGRESSION GUARD: the Space may not make the CORS decision.

    The defect (F-2, found 2026-09-22): `build_cors_headers` answered the CORS
    question on the *request* leg and was tested thoroughly -- but `_proxy`
    rebuilt the response headers from the upstream map, and
    `policy.response_headers` filtered eight header classes that did not
    include any `access-control-*`. So a CORS header the Space emitted was
    relayed verbatim, on top of whatever the allowlist had decided.

    Measured before the fix, with the allowlist `("https://frontend.example.com",)`
    and an upstream answering `Access-Control-Allow-Origin: *`:

      * the allowlisted origin received the header **twice** -- `['*',
        'https://frontend.example.com']`. A duplicated `ACAO` is not a value a
        browser can match, so the gateway's own correct answer became
        unreliable for legitimate callers;
      * a **disallowed** origin received `['*']` and a `200` -- a complete
        cross-origin read of the API from an origin the operator never
        allowed.

    This is the same shape as the trailing-slash guard (D-12): a rule that
    exists, is tested, and is bypassed by the other layer. The assertion below
    therefore checks the *end-to-end response*, not `build_cors_headers`, which
    was already correct and already covered.

    The upstream here is a real ASGI app built in-process that deliberately
    emits a permissive CORS header. It is not a mock of the gateway's own
    behaviour -- it stands in for the Space, which is the component the gateway
    must not trust.
    """
    pytest.importorskip("fastapi", reason="the web stack is unavailable here")
    import anyio
    import httpx
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def permissive(request):
        """A Space that sets its own CORS headers -- the thing the gateway must
        ignore, because only the gateway knows the operator's allowlist."""
        out = JSONResponse({"schema_version": "1.0", "status": "ok"})
        out.headers["access-control-allow-origin"] = "*"
        out.headers["access-control-allow-credentials"] = "true"
        return out

    upstream = Starlette(
        routes=[
            Route("/v1/health", permissive, methods=["GET", "HEAD"]),
            Route("/v1/analyze", permissive, methods=["POST"]),
        ]
    )
    from gateway.app import create_app

    app = create_app(
        GatewayConfig(upstream_url="http://up.test", allowed_origins=(ORIGIN,))
    )

    origin_header = "access-control-allow-origin"
    credentials_header = "access-control-allow-credentials"

    async def body():
        real_client = httpx.AsyncClient

        def client_for_upstream(**kwargs):
            kwargs.pop("timeout", None)
            return real_client(
                transport=httpx.ASGITransport(app=upstream), base_url="http://up.test"
            )

        # The gateway builds its own client inside `_proxy`; redirect that one
        # construction at the in-process Space. Scoped to this coroutine and
        # restored in `finally`, so no other test can observe the patch.
        httpx.AsyncClient = client_for_upstream
        try:
            async with real_client(
                transport=httpx.ASGITransport(app=app), base_url="http://gw.test"
            ) as client:
                allowed = await client.get("/v1/health", headers={"Origin": ORIGIN})
                denied = await client.post(
                    "/v1/analyze",
                    content=json.dumps({"assets": ["h1"], "query": "what changed"}).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "Origin": "https://evil.example.net",
                    },
                )
                return allowed, denied
        finally:
            httpx.AsyncClient = real_client

    allowed, denied = anyio.run(body)

    # 1. The allowlisted origin gets exactly the gateway's decision -- one value,
    #    and not the upstream's `*`.
    assert allowed.headers.get_list(origin_header) == [ORIGIN], (
        "the upstream's CORS header was relayed alongside the gateway's; a "
        f"duplicated ACAO breaks the gateway's own answer (got "
        f"{allowed.headers.get_list(origin_header)})"
    )

    # 2. A disallowed origin gets nothing, on the route that costs GPU quota.
    assert denied.headers.get_list(origin_header) == [], (
        "a disallowed origin received a CORS header from the upstream, so the "
        "browser will hand it the response -- the allowlist is bypassed"
    )
    assert denied.headers.get_list(credentials_header) == [], (
        "the upstream's Access-Control-Allow-Credentials reached a disallowed "
        "origin, which is a credentialed cross-origin read"
    )

    # 3. Positive control: the request really did reach the Space, so (2) is not
    #    passing because the upstream leg never ran.
    assert denied.status_code == 200, (
        "the disallowed response was not a success, so this test would pass "
        f"even if the CORS fix were absent (got {denied.status_code})"
    )
    assert denied.json().get("status") == "ok", (
        "the upstream body did not reach the client, so the response path under "
        "test did not execute"
    )



@pytest.mark.integration
def test_the_oversized_body_is_not_held_in_full_while_being_refused() -> None:
    """F-9 on the gateway: the cap must save memory, not just produce a 413.

    `test_an_oversized_body_is_refused_even_when_content_length_is_omitted`
    asserts the envelope, and it is a real guard -- but like the Space's first
    version it is a *status* guard, and a status cannot tell "refused while
    reading" from "refused after buffering". Both are 413. This test asserts the
    allocation instead, which is the property the fix was actually for.

    Measured on this stack, cap `GatewayConfig().max_body_bytes` (8 MiB) with
    `Content-Length` OMITTED, before and after the fix -- the numbers the probe
    `probe_f9_gateway_body.py` prints:

        body        peak (unbounded)   peak (bounded)   peak (declared-hdr)
        16 MiB          32.0 MiB          9.0 MiB            0.0 MiB
        64 MiB         128.0 MiB          9.0 MiB            0.0 MiB

    The omitted-length column is the one that matters: 9 MiB flat across a 4x
    body increase (the residual is the ASGI server's own read buffer, not the
    body). The declared column is 0.0 because `policy.admit` refuses on the
    header before a single byte is read.

    The claim is one-sided on purpose -- `peak < body_len` -- so it pins the
    defect (the whole body was held) without pinning an allocator number.
    """
    cfg = GatewayConfig(upstream_url=UPSTREAM, allowed_origins=(ORIGIN,))
    cap = cfg.max_body_bytes
    body_len = cap + (8 * 1024 * 1024)
    app = _app()

    status, parsed, peak = _raw_peak_post(app, "/v1/assets", body=b"x" * body_len)

    assert status == 413, (
        f"an oversized undeclared-length body got HTTP {status}, not 413; the "
        f"envelope guard above should have caught this, so if this fires the "
        f"two guards disagree about the same request"
    )
    assert parsed.get("error", {}).get("code") == "oversized_image", (
        f"the refusal was not the contract envelope: {parsed!r}"
    )
    assert peak < body_len, (
        f"peak traced allocation was {peak} bytes for a {body_len}-byte body "
        f"(cap {cap}): the gateway buffered the body in full before refusing "
        f"it. The cap must be enforced while reading."
    )


# ---------------------------------------------------------------------------
# F-15c — the transport branch must not publish raw exception text
# ---------------------------------------------------------------------------
def test_a_transport_failure_publishes_a_classification_not_the_exception() -> None:
    """F-15c (owner ruling 2026-09-23).

    `_proxy`'s transport `except` branch used to publish
    `f"{type(exc).__name__}: {exc}"` as the error body's `detail` -- and `detail`
    is part of the documented envelope (`gateway/policy.py:165`), so that reached
    an unauthenticated client.

    The realistic exception in this sandbox is path-free (an egress proxy answers
    every outbound request with a plain-text 502, so the transport branch is
    rarely even reached), which is why the leak was never *observed*. That is a
    statement about this environment, not about the code -- so the guard drives
    the branch directly and hands it a deliberately hostile exception that names
    its own class, a filesystem path and a proxy URL.

    All three must be absent from the body, while the classification,
    `recoverable` and `request_id` survive intact.
    """
    import json

    pytest.importorskip("fastapi", reason="the web stack is unavailable here")
    import anyio
    import httpx

    app = _app()

    class _UpstreamTransportError(Exception):
        """Shaped like a transport failure, but hostile in its text."""

    leak = r"C:/Users/anish/secret/egress.conf"
    proxy = "http://proxy.internal.example:3128"

    async def body():
        real_client = httpx.AsyncClient

        class _FailingClient:
            def __init__(self, **kwargs: Any) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc_info: Any) -> bool:
                return False

            async def request(self, *args: Any, **kwargs: Any):
                raise _UpstreamTransportError(
                    f"ConnectError reading {leak} via {proxy}: handshake failed"
                )

        # `_proxy` builds its own client; redirect that one construction.
        # Scoped to this coroutine and restored in `finally`.
        httpx.AsyncClient = _FailingClient
        try:
            async with real_client(
                transport=httpx.ASGITransport(app=app), base_url="http://gw.test"
            ) as client:
                return await client.post(
                    "/v1/analyze",
                    content=json.dumps(
                        {"assets": ["a", "b"], "query": "what changed"}
                    ).encode(),
                    headers={"Content-Type": "application/json", "Origin": ORIGIN},
                )
        finally:
            httpx.AsyncClient = real_client

    r = anyio.run(body)

    assert r.status_code == 502, (
        f"expected the transport branch's 502, got {r.status_code}: {r.text[:300]}"
    )

    err = r.json()["error"]

    # -- the classification and the request-id semantics SURVIVE -------------
    assert err["code"] == "model_unavailable"
    assert err["recoverable"] is True
    assert err["request_id"], "the correlation id was lost"
    assert err["message"], "the operator-safe message was lost"
    assert err["detail"], "the transport failure lost its classification"

    # -- no internals -------------------------------------------------------
    text = r.text
    assert "UpstreamTransportError" not in text, (
        "the exception's class name reached the client"
    )
    assert "handshake failed" not in text, (
        "the exception's own message reached the client"
    )
    assert leak not in text and "egress.conf" not in text, (
        "a filesystem path reached the client"
    )
    assert proxy not in text, "the egress proxy URL reached the client"


def test_the_transport_classifier_is_contract_level_not_a_third_party_name() -> None:
    """F-15c: the classification names the FAILURE, not the library.

    A client can act on "the upstream could not be reached" versus "the upstream
    timed out" -- different operator actions. It cannot act on `ConnectError`
    versus `ReadTimeout`, which are the gateway's HTTP client's own vocabulary.
    """
    pytest.importorskip("httpx")
    import httpx

    from gateway.app import _transport_failure_detail

    assert _transport_failure_detail(httpx.ConnectError("x")) == (
        "the upstream could not be reached"
    )
    assert _transport_failure_detail(httpx.ReadTimeout("x")) == (
        "the upstream did not respond within the gateway timeout"
    )
    assert _transport_failure_detail(httpx.ProxyError("x")) == (
        "the gateway's egress proxy refused the connection"
    )

    class _Novel(Exception):
        """A failure the classifier has never seen must still classify."""

    assert _transport_failure_detail(_Novel("boom")) == (
        "the upstream request failed at the transport layer"
    )

    for exc in (
        httpx.ConnectError("x"),
        httpx.ReadTimeout("x"),
        httpx.ProxyError("x"),
        _Novel("boom"),
    ):
        detail = _transport_failure_detail(exc)
        assert type(exc).__name__ not in detail, (
            f"the classification leaked the class name {type(exc).__name__!r}"
        )
        assert "httpx" not in detail, "the classification names the HTTP client"
