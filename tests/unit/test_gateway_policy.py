"""The gateway's decisions must be correct, cheap, and never reach the Space on a rejection.

WHY THIS EXISTS
---------------
`docs/DEPLOYMENT_ARCHITECTURE.md` section 1.1 gives three reasons the gateway
exists, and the strongest is quota: ZeroGPU grants 5 GPU-minutes/day, so **a
request the gateway can reject on shape must never cost GPU time**. That makes
the gateway's rejection paths safety-critical in a way a normal proxy's are not.

Two classes of mistake are pinned here.

1. **A rejection that does not reject.** If `admit()` returns `admit=True` for an
   oversized body or a rate-limited client, the request walks to the Space and
   spends quota. Every refusal path is asserted, including the HTTP status,
   because a refusal with a 200 status would be forwarded as a success.
2. **A rule that exists in two places.** `validate_analyze_body` answers cheaply
   before the serving stack is imported. If it disagreed with the authoritative
   `AnalysisRequest` model, the gateway would forward a body the Space rejects
   (or refuse a body the Space would accept). A cross-check asserts they agree.

WHAT IS NOT ASSERTED
--------------------
The ASGI application in `gateway/app.py` cannot be imported in this environment:
a WDAC policy blocks `orjson.pyd` with `WinError 4551`, which escapes FastAPI's
own `except ModuleNotFoundError` guard. That is recorded in
`docs/PHASE19_FINAL_HARDENING.md`. Consequently these tests exercise the
**policy layer**, which is where all eight gateway responsibilities live -- not
the HTTP plumbing. No test here claims a request was served over HTTP.
"""

from __future__ import annotations

import json
from typing import Iterator

import pytest

from core.errors import (
    InputError,
    ModelLoadError,
    ModelUnavailableError,
    OversizedImageError,
    PairMisalignmentError,
    SatQueryError,
    SpecialistTimeoutError,
)
from gateway.policy import (
    DEFAULT_HOP_BY_HOP,
    GATEWAY_ORIGIN_CODES,
    ClientIdentity,
    GatewayConfig,
    GatewayPolicy,
    RateLimiter,
    build_cors_headers,
    config_from_env,
    new_request_id,
    translate_error,
    validate_analyze_body,
)


def _config(**overrides) -> GatewayConfig:
    base = dict(
        upstream_url="https://satquery-test.hf.space",
        allowed_origins=("https://satquery.pages.dev",),
        upstream_timeout_s=90.0,
    )
    base.update(overrides)
    return GatewayConfig(**base)


def _policy(**overrides) -> GatewayPolicy:
    return GatewayPolicy(_config(**overrides))


# ==========================================================================
# 1. Configuration refuses incoherent limits
# ==========================================================================


def test_a_trailing_slash_on_the_upstream_is_refused() -> None:
    """A doubled slash produces a 404 that reads like an outage."""
    with pytest.raises(ValueError, match="trailing slash"):
        _config(upstream_url="https://space.hf.space/")


def test_a_wildcard_origin_is_refused() -> None:
    """`*` would expose the Space to every origin (architecture doc 2.1)."""
    with pytest.raises(ValueError, match="must not contain"):
        _config(allowed_origins=("https://ok.pages.dev", "*"))


def test_an_empty_origin_allowlist_is_refused() -> None:
    """Falling back to `*` on an empty allowlist would be a silent exposure."""
    with pytest.raises(ValueError, match="non-empty"):
        _config(allowed_origins=())


def test_a_timeout_below_the_longest_gpu_call_is_refused() -> None:
    """The runbook's lower bound, enforced in code rather than in prose."""
    with pytest.raises(ValueError, match="45"):
        _config(upstream_timeout_s=30.0)


def test_a_timeout_at_or_above_the_agent_budget_is_refused() -> None:
    """Above 120 s the gateway outlives the Space's own giving-up point."""
    with pytest.raises(ValueError, match="120"):
        _config(upstream_timeout_s=120.0)


def test_a_per_file_cap_above_the_body_cap_is_refused() -> None:
    """An unreachable per-file cap is a config bug, not a limit."""
    with pytest.raises(ValueError, match="unreachable"):
        _config(max_body_bytes=1024, max_file_bytes=4096)


# ==========================================================================
# 2. Request-id handling
# ==========================================================================


def test_a_generated_request_id_is_deterministic_for_a_seed() -> None:
    assert new_request_id("abc") == new_request_id("abc")
    assert new_request_id("abc") != new_request_id("abd")


def test_a_well_formed_inbound_request_id_is_echoed() -> None:
    policy = _policy()
    assert policy.accept_request_id("req_abc12345") == "req_abc12345"


@pytest.mark.parametrize(
    "hostile",
    [
        "",
        "short",
        "has spaces and that is not allowed",
        "a" * 200,
        "<script>alert(1)</script>",
        "req_ok\ntrailing",
    ],
)
def test_a_malformed_inbound_request_id_is_replaced_not_sanitised(hostile: str) -> None:
    """A mangled echo is more confusing than a fresh id, so it is replaced."""
    policy = _policy()
    out = policy.accept_request_id(hostile)
    assert out != hostile
    assert out.startswith("req_")


# ==========================================================================
# 3. The admit ladder -- every refusal refuses
# ==========================================================================


def test_a_normal_request_is_admitted() -> None:
    decision = _policy().admit(
        method="POST",
        path="/v1/analyze",
        content_length=100,
        origin="https://satquery.pages.dev",
        identity=ClientIdentity(ip="203.0.113.7"),
    )
    assert decision.admit is True
    assert decision.status == 200
    assert decision.body is None


def test_an_oversized_body_is_refused_before_the_body_is_read() -> None:
    """The check exists to avoid buffering; it must fire on Content-Length."""
    decision = _policy(max_body_bytes=1024, max_file_bytes=512).admit(
        method="POST",
        path="/v1/analyze",
        content_length=2048,
        origin=None,
        identity=ClientIdentity(ip="203.0.113.7"),
    )
    assert decision.admit is False
    assert decision.status == 413
    assert decision.body is not None
    assert decision.body["error"]["code"] == "oversized_image"


def test_a_refusal_never_carries_a_success_status() -> None:
    """A 200 refusal forwarded upstream would be treated as an answer."""
    decision = _policy(max_body_bytes=10, max_file_bytes=10).admit(
        method="POST",
        path="/v1/analyze",
        content_length=99,
        origin=None,
        identity=ClientIdentity(ip="203.0.113.7"),
    )
    assert decision.admit is False
    assert decision.status >= 400


def test_a_rate_limited_client_is_refused_with_a_retry_after() -> None:
    policy = _policy(rate_limit_per_ip=2, rate_limit_window_s=60.0)
    identity = ClientIdentity(ip="203.0.113.7")
    for _ in range(2):
        assert policy.admit(
            method="POST",
            path="/v1/analyze",
            content_length=10,
            origin=None,
            identity=identity,
        ).admit

    third = policy.admit(
        method="POST",
        path="/v1/analyze",
        content_length=10,
        origin=None,
        identity=identity,
    )
    assert third.admit is False
    assert third.status == 429
    assert "Retry-After" in third.headers


def test_a_rejected_attempt_does_not_extend_the_window() -> None:
    """Otherwise a hammering client pushes its own reset out on every try."""
    clock = [0.0]
    limiter = RateLimiter(limit=1, window_s=10.0, now=lambda: clock[0])
    assert limiter.check("a")[0] is True
    for _ in range(10):
        assert limiter.check("a")[0] is False
    clock[0] = 10.0
    # The window opened at t=0 and is 10 s long, so t=10 starts a fresh one.
    assert limiter.check("a")[0] is True


def test_health_checks_do_not_consume_the_rate_limit_budget() -> None:
    """Polling health is how a frontend shows liveness; it must stay cheap."""
    policy = _policy(rate_limit_per_ip=1)
    identity = ClientIdentity(ip="203.0.113.7")
    for _ in range(5):
        decision = policy.admit(
            method="GET",
            path="/v1/health",
            content_length=None,
            origin=None,
            identity=identity,
        )
        assert decision.admit is True


def test_a_preflight_is_answered_and_never_forwarded() -> None:
    decision = _policy().admit(
        method="OPTIONS",
        path="/v1/analyze",
        content_length=None,
        origin="https://satquery.pages.dev",
        identity=ClientIdentity(ip="203.0.113.7"),
    )
    assert decision.admit is True
    assert decision.headers["Access-Control-Allow-Origin"] == "https://satquery.pages.dev"


def test_an_unsupported_method_is_refused_by_the_gateway() -> None:
    decision = _policy().admit(
        method="DELETE",
        path="/v1/analyze",
        content_length=None,
        origin=None,
        identity=ClientIdentity(ip="203.0.113.7"),
    )
    assert decision.admit is False
    assert decision.status == 422


def test_the_analyze_route_is_identified_by_path_by_default() -> None:
    """A route rename must not silently disable rate limiting."""
    policy = _policy(rate_limit_per_ip=1)
    identity = ClientIdentity(ip="203.0.113.7")
    assert policy.admit(
        method="POST",
        path="/v1/analyze",
        content_length=10,
        origin=None,
        identity=identity,
    ).admit
    assert not policy.admit(
        method="POST",
        path="/v1/analyze",
        content_length=10,
        origin=None,
        identity=identity,
    ).admit


# ==========================================================================
# 4. CORS is an allowlist, not an echo
# ==========================================================================


def test_a_disallowed_origin_receives_no_cors_headers() -> None:
    """Echoing the origin back regardless would defeat the allowlist."""
    assert build_cors_headers("https://evil.example", ("https://ok.pages.dev",)) == {}


def test_a_missing_origin_receives_no_cors_headers() -> None:
    assert build_cors_headers(None, ("https://ok.pages.dev",)) == {}


def test_an_allowed_origin_is_echoed_with_vary() -> None:
    headers = build_cors_headers("https://ok.pages.dev", ("https://ok.pages.dev",))
    assert headers["Access-Control-Allow-Origin"] == "https://ok.pages.dev"
    assert headers["Vary"] == "Origin"


# ==========================================================================
# 5. Error translation preserves codes
# ==========================================================================


@pytest.mark.parametrize(
    ("exc", "expected_status"),
    [
        (InputError("x"), 400),
        (OversizedImageError("x"), 413),
        (PairMisalignmentError("x"), 422),
        (ModelUnavailableError("x"), 503),
        (ModelLoadError("x"), 503),
        (SpecialistTimeoutError("x"), 504),
    ],
)
def test_a_real_error_class_maps_to_the_contract_status(
    exc: SatQueryError, expected_status: int
) -> None:
    """The mapping must match `docs/API_CONTRACT.md` section 5.1 exactly.

    Constructed from the REAL error classes rather than from string literals, so
    a class rename or a code change makes this fail rather than silently leaving
    the gateway mapping a status the contract does not document.
    """
    status, body = translate_error(exc.code, exc.user_message)
    assert status == expected_status, (
        f"{type(exc).__name__} (code {exc.code!r}) maps to {status}, but "
        f"API_CONTRACT.md section 5.1 documents {expected_status}"
    )
    # The code must survive verbatim -- section 2.3 forbids remapping.
    assert body["error"]["code"] == exc.code
    # And the operator-safe message must be what the error class declares.
    assert body["error"]["message"] == exc.user_message


def test_the_code_is_passed_through_unchanged() -> None:
    status, body = translate_error("pair_misaligned", "not co-registered", detail="RMSE 4.2px")
    assert body["error"]["code"] == "pair_misaligned"
    assert body["error"]["message"] == "not co-registered"
    assert body["error"]["detail"] == "RMSE 4.2px"
    assert status == 422


def test_an_unknown_code_becomes_a_base_error_not_a_success() -> None:
    """A gateway that let an unknown code through as 200 would hide a defect."""
    status, body = translate_error("invented_code", "something")
    assert status == 500
    assert body["error"]["code"] == "satquery_error"
    assert "invented_code" in body["error"]["detail"]


def test_every_real_taxonomy_code_is_in_the_mapping_table() -> None:
    """No code may fall through to the generic 500 by omission.

    The table is hand-written; this test makes a missing entry a failure rather
    than a silent degradation to `satquery_error`.
    """
    import inspect

    from core import errors as errors_module
    from gateway import policy as policy_module

    real_codes = {
        obj.code
        for obj in vars(errors_module).values()
        if inspect.isclass(obj)
        and issubclass(obj, SatQueryError)
        and obj is not SatQueryError
    }
    # `satquery_error` itself is the base and IS in the table.
    real_codes.add(SatQueryError.code)

    missing = sorted(real_codes - set(policy_module._CODE_STATUS))
    assert not missing, (
        f"these real error codes have no HTTP status mapping and would all "
        f"collapse to 500: {missing}"
    )


def test_the_gateway_origin_codes_do_not_collide_with_the_taxonomy() -> None:
    """A gateway code that shadowed a taxonomy code would misreport an error."""
    import inspect

    from core import errors as errors_module

    taxonomy = {
        obj.code
        for obj in vars(errors_module).values()
        if inspect.isclass(obj) and issubclass(obj, SatQueryError)
    }
    assert not (GATEWAY_ORIGIN_CODES & taxonomy), (
        f"gateway-origin codes {sorted(GATEWAY_ORIGIN_CODES & taxonomy)} shadow "
        f"real taxonomy codes; a Space error would be reported as a gateway one"
    )


# ==========================================================================
# 6. The cheap validator agrees with the authoritative schema
# ==========================================================================


def test_a_valid_body_passes_the_cheap_validator() -> None:
    raw = json.dumps(
        {"assets": ["a0", "a1"], "query": "what changed?", "force_task": "change_vqa"}
    ).encode()
    parsed, error = validate_analyze_body(raw)
    assert error is None
    assert parsed is not None
    assert parsed["query"] == "what changed?"


@pytest.mark.parametrize(
    ("raw", "expected_status"),
    [
        (b"not json at all", 400),
        (b"[1,2,3]", 422),
        (json.dumps({"query": "q"}).encode(), 422),
        (json.dumps({"assets": [], "query": "q"}).encode(), 422),
        (json.dumps({"assets": [1], "query": "q"}).encode(), 422),
        (json.dumps({"assets": ["a"]}).encode(), 422),
        (json.dumps({"assets": ["a"], "query": 5}).encode(), 422),
    ],
)
def test_a_malformed_body_is_refused(raw: bytes, expected_status: int) -> None:
    parsed, error = validate_analyze_body(raw)
    assert parsed is None
    assert error is not None
    assert error[0] == expected_status


def test_the_cheap_validator_agrees_with_the_real_model_on_valid_bodies() -> None:
    """If these disagree, the gateway forwards bodies the Space will reject.

    That failure looks like a server bug to the frontend and costs a round trip
    (and, if the Space had already started, GPU quota). The two must agree.
    """
    from core.schemas import AnalysisRequest

    bodies = [
        {"assets": ["a"], "query": ""},
        {"assets": ["a0", "a1"], "query": "q", "force_task": "change"},
        {"assets": ["a"], "query": "q", "run_id": "abc"},
        {"assets": ["a"], "query": "q", "force_task": None, "run_id": None},
    ]
    for payload in bodies:
        parsed, error = validate_analyze_body(json.dumps(payload).encode())
        assert error is None, f"the cheap validator refused a valid body: {payload}"

        # And the authoritative model must accept it too.
        model = AnalysisRequest.model_validate(payload)
        assert model.assets == payload["assets"]


def test_the_cheap_validator_refuses_what_the_real_model_refuses() -> None:
    """The converse direction: nothing the schema rejects may be forwarded."""
    from pydantic import ValidationError

    from core.schemas import AnalysisRequest

    invalid = [
        {"assets": [], "query": "q"},
        {"query": "q"},
        {"assets": ["a"]},
        {"assets": ["a"], "query": "q", "unknown_field": 1},
    ]
    for payload in invalid:
        with pytest.raises(ValidationError):
            AnalysisRequest.model_validate(payload)
        parsed, error = validate_analyze_body(json.dumps(payload).encode())
        assert error is not None, (
            f"the cheap validator forwarded a body the schema rejects: {payload}"
        )


# ==========================================================================
# 7. Upstream header handling
# ==========================================================================


def test_hop_by_hop_headers_are_not_forwarded() -> None:
    policy = _policy()
    out = policy.upstream_headers(
        {"Connection": "keep-alive", "Transfer-Encoding": "chunked", "X-Custom": "ok"},
        token="secret",
    )
    assert "Connection" not in out
    assert "Transfer-Encoding" not in out
    assert out["X-Custom"] == "ok"
    assert DEFAULT_HOP_BY_HOP  # the constant is non-empty


def test_the_token_is_injected_and_a_client_authorization_is_dropped() -> None:
    """A client must not be able to smuggle a credential toward the Space."""
    policy = _policy()
    out = policy.upstream_headers({"Authorization": "Bearer attacker"}, token="real")
    assert out["Authorization"] == "Bearer real"


def test_the_host_header_is_not_forwarded() -> None:
    policy = _policy()
    out = policy.upstream_headers({"Host": "evil.example"}, token="t")
    assert "Host" not in out


# ==========================================================================
# 8. Environment parsing
# ==========================================================================


def test_config_from_env_reads_the_documented_variables() -> None:
    config = config_from_env(
        {
            "SATQUERY_SPACE_URL": "https://space.hf.space/",
            "SATQUERY_ALLOWED_ORIGINS": "https://a.pages.dev, https://b.pages.dev",
            "SATQUERY_MAX_BODY_BYTES": "2048",
            "SATQUERY_MAX_FILE_BYTES": "1024",
            "SATQUERY_RATE_LIMIT_PER_IP": "3",
            "SATQUERY_UPSTREAM_TIMEOUT_S": "80",
        }
    )
    assert config.upstream_url == "https://space.hf.space"  # slash stripped
    assert config.allowed_origins == ("https://a.pages.dev", "https://b.pages.dev")
    assert config.max_body_bytes == 2048
    assert config.rate_limit_per_ip == 3
    assert config.upstream_timeout_s == 80.0


def test_config_from_env_refuses_an_empty_origin_list() -> None:
    """An unset variable must fail loudly, not default to open CORS."""
    with pytest.raises(ValueError, match="non-empty"):
        config_from_env({"SATQUERY_SPACE_URL": "https://s.hf.space"})


def test_a_non_positive_per_file_cap_is_refused_at_startup() -> None:
    """F-7. `SATQUERY_MAX_FILE_BYTES` is the one variable BOTH layers read.

    The gateway used to accept a non-positive value because `_int` only parsed
    and `GatewayConfig.__post_init__` never checked this field's sign. Measured
    2026-09-22, `SATQUERY_MAX_FILE_BYTES=0` produced:

        gateway -> ACCEPTED max_file_bytes=0
        Space   -> ValueError: max_file_bytes must be positive

    That is the dangerous ordering, not a harmless typo. The gateway's own size
    check is against `max_body_bytes`, so it keeps admitting the request; the
    Space then refuses EVERY upload, because `len(data) > 0` is true for any
    non-empty file. An operator sees "413 on every upload" against a cap they
    believe they never set, and nothing names the variable.

    `AssetStore.__post_init__` already rejects these values, so refusing them
    here only moves the failure to the layer that can name it. Note the
    assertion is on the VARIABLE NAME, not merely on `ValueError`: this must fail
    as a configuration error, not incidentally via the `max_file_bytes >
    max_body_bytes` cross-check, which would blame the wrong variable.
    """
    for bad in ("0", "-1"):
        with pytest.raises(ValueError, match="SATQUERY_MAX_FILE_BYTES"):
            config_from_env(
                {
                    "SATQUERY_SPACE_URL": "https://s.hf.space",
                    "SATQUERY_ALLOWED_ORIGINS": "https://f.example.com",
                    "SATQUERY_MAX_FILE_BYTES": bad,
                }
            )


def test_a_positive_per_file_cap_still_parses() -> None:
    """The negative control for the test above.

    Without this, a `_int`-level change that refused *every* value for this
    variable would leave the refusal test green while breaking every deployment.
    """
    config = config_from_env(
        {
            "SATQUERY_SPACE_URL": "https://s.hf.space",
            "SATQUERY_ALLOWED_ORIGINS": "https://f.example.com",
            "SATQUERY_MAX_FILE_BYTES": "1024",
        }
    )
    assert config.max_file_bytes == 1024


def test_the_trailing_slash_guard_is_narrowed_by_the_env_layer() -> None:
    """Two layers treat a trailing slash differently. Pin BOTH, and the reason.

    `GatewayConfig.__post_init__` *refuses* a trailing slash, and
    `test_a_trailing_slash_on_the_upstream_is_refused` proves that — but it
    constructs the class **directly**, which is not how a deployment is
    configured. `config_from_env` applies `.rstrip("/")` first, so via the
    operator path the value is silently normalised and that guard is
    **unreachable**. `docs/PHASE19_FINAL_HARDENING.md` lists the trailing slash
    among states that are "Refused", which is true of the class and not of the
    env path.

    The behaviour is still correct — normalising produces the same URL the guard
    would have demanded, so the doubled-slash 404 it exists to prevent cannot
    occur. What matters is that nobody "fixes" the env layer into refusing the
    normalisable form: that would break deployments that work today, for no
    safety gain. Pinning both halves makes the asymmetry deliberate.
    """
    # Layer 2 (env): normalises, does not refuse.
    config = config_from_env(
        {
            "SATQUERY_SPACE_URL": "https://space.hf.space/",
            "SATQUERY_ALLOWED_ORIGINS": "https://a.pages.dev",
        }
    )
    assert config.upstream_url == "https://space.hf.space"

    # Layer 1 (class): refuses the same input outright.
    with pytest.raises(ValueError, match="trailing slash"):
        GatewayConfig(
            upstream_url="https://space.hf.space/",
            allowed_origins=("https://a.pages.dev",),
        )

    # The end result an operator cares about is identical either way, which is
    # why the normalisation is sufficient and the guard need not be widened.
    assert GatewayPolicy(config).upstream_url("/v1/health") == (
        "https://space.hf.space/v1/health"
    )
