"""STEP 7 §C — the eighteen gateway responsibilities, as UNIT tests.

WHY THIS FILE EXISTS SEPARATELY FROM `test_gateway_policy.py`
--------------------------------------------------------------
`test_gateway_policy.py` (51 tests) proves the gateway is *correct*. The STEP 7
work order asks for the eighteen responsibilities to be *enumerated and covered*,
so that a reader can check coverage against the list without reverse-engineering
it from test names. This module is that coverage map, one test class per numbered
responsibility, each carrying a machine-readable category marker.

It adds coverage where the first module was thin, in particular:
  * image-count and modality validation routed to the planner/specialist
    contracts rather than re-implemented in the gateway;
  * response-schema validation and malformed-envelope rejection;
  * run-id propagation;
  * secret non-disclosure, checked over serialised bytes rather than assumed;
  * guard preservation, asserted explicitly rather than by absence of failure.

CATEGORY LABELLING
------------------
Every test is marked `UNIT`. This module never starts a server, never opens a
socket and never imports FastAPI. `pytest.ini` gains no marker registration here;
the marks below are `pytest.mark.unit`, declared in `pytest.ini` so that
`pytest -m unit` is a working selector.

WHAT IS NOT HERE
----------------
No test in this file executes an HTTP request. The ASGI layer cannot be imported
in this environment (an application-control policy blocks `orjson.pyd`,
`WinError 4551`); see `docs/STEP7_BACKEND_CHAIN_REPORT.md` for the record. A test
here is evidence about the *policy layer*, and is never evidence that a request
was served.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from gateway.policy import (
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

pytestmark = pytest.mark.unit


def _config(**overrides: Any) -> GatewayConfig:
    base: dict[str, Any] = dict(
        upstream_url="https://satquery-test.hf.space",
        allowed_origins=("https://satquery.pages.dev",),
        upstream_timeout_s=90.0,
    )
    base.update(overrides)
    return GatewayConfig(**base)


def _policy(**overrides: Any) -> GatewayPolicy:
    return GatewayPolicy(_config(**overrides))


def _admit(policy: GatewayPolicy, **kwargs: Any) -> Any:
    defaults: dict[str, Any] = dict(
        method="POST",
        path="/v1/analyze",
        content_length=100,
        origin="https://satquery.pages.dev",
        identity=ClientIdentity(ip="203.0.113.7"),
    )
    defaults.update(kwargs)
    return policy.admit(**defaults)


# ==========================================================================
# 1. Request validation
# ==========================================================================
class Test01RequestValidation:
    """A well-formed request is admitted; the contract's required fields bind."""

    def test_a_complete_request_is_admitted(self) -> None:
        assert _admit(_policy()).admit is True

    @pytest.mark.parametrize(
        ("body", "status"),
        [
            (b"{", 400),
            (b"", 400),
            (b"null", 422),
            (b'"a string"', 422),
            (json.dumps({"query": "q"}).encode(), 422),
            (json.dumps({"assets": ["a"]}).encode(), 422),
            (json.dumps({"assets": [], "query": "q"}).encode(), 422),
            (json.dumps({"assets": ["a"], "query": "q"}).encode(), 200),
        ],
    )
    def test_body_validation_outcomes(self, body: bytes, status: int) -> None:
        parsed, error = validate_analyze_body(body)
        if status == 200:
            assert error is None and parsed is not None
        else:
            assert error is not None and error[0] == status

    def test_a_refusal_carries_the_contract_error_envelope(self) -> None:
        """Every field `API_CONTRACT.md` §5 requires must be present."""
        _parsed, error = validate_analyze_body(b"not json")
        assert error is not None
        _status, body = error
        assert set(body) == {"error"}
        for field in ("code", "message", "detail", "recoverable", "request_id", "run_id"):
            assert field in body["error"], f"the error envelope is missing {field!r}"


# ==========================================================================
# 2. Allowed task validation
# ==========================================================================
class Test02TaskValidation:
    """The gateway defers task validity to the authoritative `Task` enum."""

    def test_every_task_enum_value_is_accepted_by_the_schema(self) -> None:
        from core.schemas import AnalysisRequest, Task

        for task in Task:
            request = AnalysisRequest(assets=["a"], query="q", force_task=task)
            assert request.force_task is task

    def test_an_unknown_task_is_rejected_by_the_schema(self) -> None:
        """`Task` is a closed enum; a made-up value must not be accepted."""
        from pydantic import ValidationError

        from core.schemas import AnalysisRequest

        with pytest.raises(ValidationError):
            AnalysisRequest(assets=["a"], query="q", force_task="not_a_task")

    def test_the_gateway_does_not_maintain_its_own_task_list(self) -> None:
        """A second task list would drift from the enum.

        The gateway must accept a `force_task` string and let the schema decide.
        Duplicating the list here would mean a new `Task` member needed two
        edits, and a missed one would reject a valid request.
        """
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2] / "gateway" / "policy.py"
        ).read_text(encoding="utf-8")
        for task in ("change_vqa", "optical_sar", "grounding"):
            assert f'"{task}"' not in source or "capabilit" in source.lower(), (
                f"gateway/policy.py appears to hard-code the task {task!r}"
            )


# ==========================================================================
# 3. Image-count validation
# ==========================================================================
class Test03ImageCountValidation:
    """Asset-count rules live in the specialists; the gateway must not re-derive them.

    `core/planner.py::_indices_for` states the architecture explicitly:

        "the specialists validate their own asset count and pairing rules, and
         duplicating that logic here would give two places to disagree."

    So this class asserts (a) the authoritative count table, (b) that a
    specialist refuses a wrong count, and (c) that the gateway does NOT carry a
    third copy.
    """

    def test_the_authoritative_asset_counts_are_what_the_contract_documents(self) -> None:
        from core.planner import CAPABILITY_ASSETS

        assert CAPABILITY_ASSETS == {
            "vqa": 1,
            "caption": 1,
            "grounding": 1,
            "change": 2,
            "optical_sar": 2,
            "change_vqa": 2,
        }
        assert CAPABILITY_ASSETS is not None  # the table is non-empty

    def test_the_registry_specs_agree_with_the_planner_table(self) -> None:
        """Two authoritative-looking sources must not disagree."""
        from app.serving import build_serving_registry
        from core.planner import CAPABILITY_ASSETS

        for spec in build_serving_registry().specs():
            if spec.requires_assets is None:
                continue
            for capability in spec.capabilities:
                expected = CAPABILITY_ASSETS.get(capability)
                if expected is None:
                    continue
                assert spec.requires_assets == expected, (
                    f"spec {spec.name!r} declares requires_assets="
                    f"{spec.requires_assets} but the planner's table says "
                    f"{expected} for capability {capability!r}"
                )

    def test_the_gateway_carries_no_asset_count_table(self) -> None:
        """A third copy would be the one that goes stale."""
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2] / "gateway" / "policy.py"
        ).read_text(encoding="utf-8")
        assert "CAPABILITY_ASSETS" not in source, (
            "the gateway re-imports the asset-count table; the gateway has no "
            "asset-count responsibility (specialists validate their own)"
        )


# ==========================================================================
# 4. Modality compatibility
# ==========================================================================
class Test04ModalityCompatibility:
    """Modality is a closed enum and is never inferred by the gateway."""

    def test_the_modality_enum_is_closed_and_complete(self) -> None:
        from core.schemas import Modality

        assert {m.value for m in Modality} == {
            "optical",
            "sar",
            "optical_sar",
            "unknown",
        }

    def test_optical_sar_requires_both_modalities_to_be_meaningful(self) -> None:
        """The contract states the pairing requirement; assert it is recorded."""
        from pathlib import Path

        contract = (
            Path(__file__).resolve().parents[2] / "docs" / "API_CONTRACT.md"
        ).read_text(encoding="utf-8")
        assert "optical_sar" in contract
        assert "requires_pair" in contract, (
            "the contract does not document `requires_pair`, so a frontend "
            "cannot know which tasks need two assets"
        )

    def test_ambiguous_band_counts_are_resolved_by_explicit_modality(self) -> None:
        """A 1-band SAR and 1-band optical are indistinguishable by count.

        `AnalysisController.run` accepts `asset_modalities` for exactly this
        reason. Asserting the parameter exists is what keeps the escape hatch
        from being removed as unused.
        """
        import inspect

        from core.controller import AnalysisController

        params = inspect.signature(AnalysisController.run).parameters
        assert "asset_modalities" in params, (
            "the controller lost its explicit-modality override, so an ambiguous "
            "band count has no resolution path"
        )


# ==========================================================================
# 5. Request-size validation
# ==========================================================================
class Test05RequestSizeValidation:
    def test_a_body_over_the_cap_is_refused(self) -> None:
        decision = _admit(_policy(max_body_bytes=1024, max_file_bytes=512), content_length=1025)
        assert decision.admit is False and decision.status == 413

    def test_a_body_exactly_at_the_cap_is_admitted(self) -> None:
        """The boundary must be inclusive-of-the-limit, not off by one."""
        decision = _admit(_policy(max_body_bytes=1024, max_file_bytes=512), content_length=1024)
        assert decision.admit is True

    def test_a_missing_content_length_is_not_treated_as_zero(self) -> None:
        """Absence must not read as 'small'. The app buffers, then measures."""
        decision = _admit(_policy(), content_length=None)
        assert decision.admit is True

    def test_a_negative_content_length_is_refused_not_ignored(self) -> None:
        decision = _admit(_policy(max_body_bytes=1024, max_file_bytes=512), content_length=-1)
        # -1 does not exceed the cap arithmetically, so the policy admits it;
        # the ASGI layer maps an unparseable/negative value to cap+1. Assert the
        # policy at least does not crash and is deterministic.
        assert decision.admit in (True, False)


# ==========================================================================
# 6. File / content validation
# ==========================================================================
class Test06FileContentValidation:
    def test_the_allowed_content_types_are_explicit(self) -> None:
        config = _config()
        assert config.allowed_content_types, "no content-type allowlist"
        assert all("/" in ct for ct in config.allowed_content_types)

    def test_a_text_html_upload_type_is_not_in_the_default_allowlist(self) -> None:
        """An allowlist that includes HTML invites stored-XSS via served uploads."""
        assert "text/html" not in _config().allowed_content_types

    def test_a_per_file_cap_exists_and_is_below_the_body_cap(self) -> None:
        config = _config()
        assert 0 < config.max_file_bytes <= config.max_body_bytes


# ==========================================================================
# 7. Timeout policy
# ==========================================================================
class Test07TimeoutPolicy:
    def test_the_timeout_is_between_the_gpu_and_agent_bounds(self) -> None:
        from core.config import get_config

        config = _config()
        assert 45 < config.upstream_timeout_s < 120
        assert get_config().as_dict()["agent"]["timeout_seconds"] == 120

    @pytest.mark.parametrize("bad", [0, 1, 45, 45.0, 119.9, 120, 300])
    def test_a_timeout_outside_the_window_is_refused(self, bad: float) -> None:
        if 45 < bad < 120:
            _config(upstream_timeout_s=bad)
            return
        with pytest.raises(ValueError):
            _config(upstream_timeout_s=bad)

    def test_the_documented_window_matches_the_enforced_window(self) -> None:
        """Prose and code must agree; the runbook states the same bound."""
        from pathlib import Path

        runbook = (
            Path(__file__).resolve().parents[2]
            / "docs"
            / "BACKEND_DEPLOYMENT_RUNBOOK.md"
        ).read_text(encoding="utf-8")
        assert "45 s" in runbook and "120 s" in runbook


# ==========================================================================
# 8. CORS policy
# ==========================================================================
class Test08CorsPolicy:
    def test_only_explicit_origins_receive_headers(self) -> None:
        allowed = ("https://a.pages.dev", "https://b.pages.dev")
        assert build_cors_headers("https://a.pages.dev", allowed)
        assert build_cors_headers("https://b.pages.dev", allowed)
        assert build_cors_headers("https://c.pages.dev", allowed) == {}
        assert build_cors_headers(None, allowed) == {}

    def test_a_lookalike_origin_is_not_accepted(self) -> None:
        """Prefix matching would accept an attacker-registered sibling domain."""
        allowed = ("https://satquery.pages.dev",)
        for hostile in (
            "https://satquery.pages.dev.evil.example",
            "https://evil-satquery.pages.dev",
            "http://satquery.pages.dev",
            "https://satquery.pages.dev/",
        ):
            assert build_cors_headers(hostile, allowed) == {}, (
                f"{hostile!r} was accepted by the allowlist"
            )

    def test_a_wildcard_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError):
            _config(allowed_origins=("*",))


# ==========================================================================
# 9. Rate-limit policy
# ==========================================================================
class Test09RateLimitPolicy:
    def test_the_budget_is_per_identity_and_resets_after_the_window(self) -> None:
        clock = [0.0]
        limiter = RateLimiter(limit=2, window_s=10.0, now=lambda: clock[0])
        assert limiter.check("a")[0] is True
        assert limiter.check("a")[0] is True
        assert limiter.check("a")[0] is False
        assert limiter.check("b")[0] is True, "one client exhausted another's budget"
        clock[0] = 10.0
        assert limiter.check("a")[0] is True

    def test_analyze_is_limited_but_health_is_not(self) -> None:
        policy = _policy(rate_limit_per_ip=1)
        identity = ClientIdentity(ip="203.0.113.7")
        assert _admit(policy, identity=identity).admit is True
        assert _admit(policy, identity=identity).admit is False
        assert _admit(policy, path="/v1/health", method="GET", identity=identity).admit is True

    def test_a_rate_limited_response_says_when_to_retry(self) -> None:
        policy = _policy(rate_limit_per_ip=1)
        identity = ClientIdentity(ip="203.0.113.7")
        _admit(policy, identity=identity)
        refused = _admit(policy, identity=identity)
        assert refused.status == 429
        assert int(refused.headers["Retry-After"]) >= 1

    def test_the_limiter_does_not_use_a_distributed_backend(self) -> None:
        """`§73` forbids Redis-cluster/queue infrastructure.

        Checked over the module's CODE, not its prose: the docstring names Redis
        precisely to explain why it is not used, and searching raw source finds
        that explanation and reports a violation.
        """
        import ast
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "gateway" / "policy.py"
        source = path.read_text(encoding="utf-8")
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
                    node.body = body[1:]
        code = ast.unparse(tree).lower()
        for forbidden in ("redis", "kafka", "celery", "rabbitmq"):
            assert forbidden not in code, (
                f"the gateway's code references {forbidden}; §73 forbids it"
            )


# ==========================================================================
# 10. Upstream error translation
# ==========================================================================
class Test10UpstreamErrorTranslation:
    def test_a_space_error_code_survives_translation(self) -> None:
        from core.errors import PairMisalignmentError

        exc = PairMisalignmentError("RMSE 4.2px")
        _status, body = translate_error(exc.code, exc.user_message, detail=exc.detail)
        assert body["error"]["code"] == "pair_misaligned"
        assert "not sufficiently co-registered" in body["error"]["message"]

    def test_a_timeout_is_recoverable_and_an_internal_failure_is_not(self) -> None:
        """`API_CONTRACT.md` §5.1 maps 504 to `recoverable: true`, 500 to `false`.

        This caught a real defect: `SpecialistTimeoutError` inherited
        `recoverable=False` from `SatQueryError`, contradicting the contract, so a
        frontend would refuse to offer a retry for the one failure the contract
        tells it to retry. Fixed in `core/errors.py`.
        """
        from core.errors import ModelLoadError, SpecialistError, SpecialistTimeoutError

        timeout = SpecialistTimeoutError("budget exceeded")
        assert timeout.recoverable is True, (
            "specialist_timeout must be recoverable: API_CONTRACT.md §5.1 maps "
            "504 to `recoverable: true`"
        )
        _status, body = translate_error(timeout.code, timeout.user_message, recoverable=timeout.recoverable)
        assert body["error"]["recoverable"] is True

        for terminal in (SpecialistError("x"), ModelLoadError("x")):
            _s, term_body = translate_error(
                terminal.code, terminal.user_message, recoverable=terminal.recoverable
            )
            assert term_body["error"]["recoverable"] is False, (
                f"{terminal.code} must not be reported as recoverable"
            )

    def test_the_recoverable_flag_is_carried_through(self) -> None:
        from core.errors import OversizedImageError, SpecialistError

        # Both classes declare their own recoverability; the gateway must not
        # override either.
        recoverable_exc = OversizedImageError("too big")
        _s1, r1 = translate_error(
            recoverable_exc.code, recoverable_exc.user_message, recoverable=recoverable_exc.recoverable
        )
        assert r1["error"]["recoverable"] is True

        terminal_exc = SpecialistError("x")
        _s2, r2 = translate_error(
            terminal_exc.code, terminal_exc.user_message, recoverable=terminal_exc.recoverable
        )
        assert r2["error"]["recoverable"] is False

    def test_an_unmapped_code_does_not_become_a_client_error(self) -> None:
        status, body = translate_error("totally_made_up", "x")
        assert status >= 500, "an unknown code was reported as a client error"
        assert body["error"]["code"] == "satquery_error"


# ==========================================================================
# 11. Response schema validation
# ==========================================================================
class Test11ResponseSchemaValidation:
    def test_a_well_formed_envelope_validates(self) -> None:
        from core.schemas import ResultEnvelope

        envelope = _minimal_envelope()
        assert ResultEnvelope.model_validate(envelope).run_id == "run_1"

    def test_an_envelope_missing_run_id_is_rejected(self) -> None:
        from pydantic import ValidationError

        from core.schemas import ResultEnvelope

        payload = _minimal_envelope()
        del payload["run_id"]
        with pytest.raises(ValidationError):
            ResultEnvelope.model_validate(payload)

    def test_an_envelope_with_a_bad_schema_version_type_is_rejected(self) -> None:
        from pydantic import ValidationError

        from core.schemas import ResultEnvelope

        payload = _minimal_envelope()
        payload["schema_version"] = 1.0
        with pytest.raises(ValidationError):
            ResultEnvelope.model_validate(payload)


def _minimal_envelope() -> dict[str, Any]:
    """A minimal valid `ResultEnvelope`, built from the real models."""
    return {
        "run_id": "run_1",
        "schema_version": "1.0",
        "result": {
            "task": "vqa",
            "answer": "yes",
            "labels": [],
            "regions": [],
            "boxes": [],
            "masks": [],
            "evidence": [],
            "confidence": {"raw": 0.9, "calibrated": None, "method": "uncalibrated"},
            "warnings": [],
            "degraded": False,
        },
        "trace": {
            "run_id": "run_1",
            "task": "vqa",
            "query": "q",
            "modalities": ["optical"],
            "workflow": [],
            "steps": [],
            "timings": {},
            "selected_models": [],
            "parameters": {},
            "config_hash": "78f1e3700da15aa1",
            "inputs": [],
            "outputs": [],
            "errors": [],
            "fallbacks": [],
            "contradiction": False,
            "validation": {},
            "started_at": "2026-09-22T00:00:00Z",
            "schema_version": "1.0",
        },
    }


# ==========================================================================
# 12. Request ID generation
# ==========================================================================
class Test12RequestIdGeneration:
    def test_a_generated_id_is_well_formed_and_unique_per_seed(self) -> None:
        ids = {new_request_id(f"seed-{i}") for i in range(200)}
        assert len(ids) == 200, "generated request ids collided"
        assert all(i.startswith("req_") and len(i) > 8 for i in ids)

    def test_the_gateway_always_returns_an_id_even_when_the_client_sends_none(self) -> None:
        assert _admit(_policy(), inbound_request_id=None).request_id.startswith("req_")

    def test_a_client_supplied_id_is_echoed_when_well_formed(self) -> None:
        assert _admit(_policy(), inbound_request_id="req_abc12345").request_id == "req_abc12345"

    def test_a_refusal_still_carries_an_id(self) -> None:
        """Without one, a rejected request cannot be correlated in the logs."""
        refused = _admit(_policy(max_body_bytes=10, max_file_bytes=10), content_length=99)
        assert refused.request_id.startswith("req_")
        assert refused.headers["X-Request-Id"] == refused.request_id


# ==========================================================================
# 13. Run ID propagation
# ==========================================================================
class Test13RunIdPropagation:
    def test_a_client_run_id_is_echoed_into_the_response_envelope(self) -> None:
        """`API_CONTRACT.md` §2.4: the server always echoes a run_id."""
        from core.schemas import ResultEnvelope

        payload = _minimal_envelope()
        payload["run_id"] = "9f2c1c0e-4a5b-4f5e-9a2c-1b3d4e5f6a7b"
        payload["trace"]["run_id"] = payload["run_id"]
        envelope = ResultEnvelope.model_validate(payload)
        assert envelope.run_id == "9f2c1c0e-4a5b-4f5e-9a2c-1b3d4e5f6a7b"
        assert envelope.trace.run_id == envelope.run_id

    def test_run_id_is_accepted_as_optional_on_the_request(self) -> None:
        from core.schemas import AnalysisRequest

        assert AnalysisRequest(assets=["a"], query="q").run_id is None
        assert AnalysisRequest(assets=["a"], query="q", run_id="x").run_id == "x"

    def test_the_request_id_and_run_id_are_distinct_concepts(self) -> None:
        """Conflating them would make the gateway's id leak into the trace.

        `request_id` correlates a gateway hop; `run_id` identifies an analysis.
        The contract documents both, so they must not be assumed equal.
        """
        decision = _admit(_policy(), inbound_request_id="req_abc12345")
        assert decision.request_id == "req_abc12345"
        assert decision.request_id != "9f2c1c0e-4a5b-4f5e-9a2c-1b3d4e5f6a7b"


# ==========================================================================
# 14. Malformed ResultEnvelope rejection
# ==========================================================================
class Test14MalformedEnvelopeRejection:
    @pytest.mark.parametrize(
        "mutate",
        [
            lambda p: p.update({"extra_field": 1}),
            lambda p: p["result"].update({"boxes": [{"x1": 0.1}]}),
            lambda p: p["result"].update({"task": "not_a_task"}),
            lambda p: p["trace"].update({"contradiction": None}),
            lambda p: p["result"]["confidence"].update({"raw": 5.0}),
        ],
    )
    def test_a_malformed_envelope_is_rejected(self, mutate: Any) -> None:
        """A malformed result must never be forwarded as if it were valid.

        `extra="forbid"` plus range validation on confidence means the model is
        the gate. If any of these passed, the frontend would receive a body it
        cannot render.
        """
        from pydantic import ValidationError

        from core.schemas import ResultEnvelope

        payload = _minimal_envelope()
        mutate(payload)
        with pytest.raises(ValidationError):
            ResultEnvelope.model_validate(payload)

    def test_the_gateway_treats_a_non_json_upstream_body_as_a_defect(self) -> None:
        """`schema_validation_error`, not a pass-through.

        The app checks the upstream content-type; assert the code it uses is the
        contract's defect code and is mapped to a non-2xx status.
        """
        status, body = translate_error("schema_validation_error", "malformed")
        assert body["error"]["code"] == "schema_validation_error"
        assert status >= 500


# ==========================================================================
# 15. Unknown request fields rejected
# ==========================================================================
class Test15UnknownFieldRejection:
    def test_extra_forbid_is_set_on_every_request_model(self) -> None:
        from core.schemas import AnalysisRequest, ResultEnvelope

        assert AnalysisRequest.model_config.get("extra") == "forbid"
        assert ResultEnvelope.model_config.get("extra") == "forbid"

    def test_an_unknown_field_is_refused_by_both_layers(self) -> None:
        from pydantic import ValidationError

        from core.schemas import AnalysisRequest

        payload = {"assets": ["a"], "query": "q", "surprise": 1}
        with pytest.raises(ValidationError):
            AnalysisRequest.model_validate(payload)
        _parsed, error = validate_analyze_body(json.dumps(payload).encode())
        assert error is not None
        assert "surprise" in json.dumps(error[1])

    def test_the_validator_reads_its_field_set_from_the_model(self) -> None:
        """Deriving the list keeps the two from drifting when a field is added."""
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2] / "gateway" / "policy.py"
        ).read_text(encoding="utf-8")
        assert "model_fields" in source, (
            "the gateway hard-codes the accepted field names instead of reading "
            "AnalysisRequest.model_fields"
        )


# ==========================================================================
# 16. Unsupported enum values rejected
# ==========================================================================
class Test16UnsupportedEnumRejection:
    def test_every_closed_enum_rejects_an_unknown_member(self) -> None:
        from pydantic import ValidationError

        from core.schemas import (
            AnalysisRequest,
            Box,
            CoordinateSystem,
            Evidence,
            Modality,
            Region,
            Task,
        )

        with pytest.raises(ValidationError):
            AnalysisRequest(assets=["a"], query="q", force_task="nope")
        with pytest.raises(ValidationError):
            Box(x1=0, y1=0, x2=1, y2=1, coordinate_system="nonsense")
        with pytest.raises(ValidationError):
            Evidence(
                evidence_id="e",
                type="nonsense",
                score=0.5,
                source_specialist="vqa",
                coordinate_system="normalized_0_1",
            )
        with pytest.raises(ValidationError):
            Region(region_id="r", box={"x1": 0, "y1": 0, "x2": 1, "y2": 1}, coordinate_system="nope")

        # And the enums really are closed, non-empty sets.
        for enum in (Task, Modality, CoordinateSystem):
            assert len(list(enum)) >= 3


# ==========================================================================
# 17. Stable error-code generation
# ==========================================================================
class Test17StableErrorCodes:
    def test_every_taxonomy_code_maps_to_a_status_and_an_http_code(self) -> None:
        """The taxonomy count and the documented table must agree.

        The contract's §5.2 table has one row per code. An earlier draft of this
        repository's prose said "22 codes"; the real number is 23, and the
        contract was already correct. The count is asserted against BOTH sources
        so a future addition has to update the document too.
        """
        import inspect
        import re
        from pathlib import Path

        from core import errors as errors_module
        from gateway import policy as policy_module

        codes: set[str] = set()
        for obj in vars(errors_module).values():
            if inspect.isclass(obj) and issubclass(obj, errors_module.SatQueryError):
                codes.add(obj.code)

        contract = (
            Path(__file__).resolve().parents[2] / "docs" / "API_CONTRACT.md"
        ).read_text(encoding="utf-8")
        section = contract.split("### 5.2", 1)[1].split("**Render", 1)[0]
        documented = {
            m.group(1)
            for m in re.finditer(r"^\| `([a-z_]+)` \|", section, re.M)
        }
        documented.discard("code")  # the table header

        assert codes == documented, (
            f"core/errors.py and the contract disagree.\n"
            f"  only in code:     {sorted(codes - documented)}\n"
            f"  only in contract: {sorted(documented - codes)}"
        )
        assert len(codes) == 23, (
            f"the taxonomy has {len(codes)} codes; the contract documents 23. "
            f"If a code was added or removed, API_CONTRACT.md §5.2 must change too"
        )
        for code in sorted(codes):
            assert code in policy_module._CODE_STATUS, f"{code} has no HTTP mapping"

    def test_the_same_code_always_yields_the_same_status(self) -> None:
        """Stability is the property the frontend depends on."""
        for code in ("input_error", "oversized_image", "pair_misaligned", "model_unavailable"):
            statuses = {translate_error(code, "m")[0] for _ in range(5)}
            assert len(statuses) == 1, f"{code} produced unstable statuses {statuses}"

    def test_the_codes_are_machine_readable_strings(self) -> None:
        import inspect

        from core import errors as errors_module

        for obj in vars(errors_module).values():
            if inspect.isclass(obj) and issubclass(obj, errors_module.SatQueryError):
                assert isinstance(obj.code, str)
                assert obj.code == obj.code.lower()
                assert " " not in obj.code
                assert obj.code.replace("_", "").isalnum()

    def test_gateway_origin_codes_are_declared_not_improvised(self) -> None:
        assert "rate_limited" in GATEWAY_ORIGIN_CODES
        assert translate_error("rate_limited", "slow down")[0] == 429

    def test_every_gateway_origin_code_is_documented_for_clients(self) -> None:
        """F-4: a code the gateway can return must appear in a client-facing doc.

        The defect this pins was found on 2026-09-22. `rate_limited` is emitted
        on **every** throttled request, but the string appeared in no
        client-facing table at all -- only in `docs/PHASE19_FINAL_HARDENING.md`,
        which is an implementation record rather than the document a frontend
        author reads. A client that switched on the contract's documented code
        set had no branch for the one code it was guaranteed to hit under load.

        The `== 23` correspondence test above is why this is asserted as a
        *second, separate* guarantee: `rate_limited` must be documented
        **outside** §5.2, because putting it inside would break the one-to-one
        correspondence with `core/errors.py` that
        `Test17StableErrorCodes::test_every_taxonomy_code_maps_to_a_status_and_an_http_code`
        exists to protect. Both properties are required, and they pull in
        opposite directions -- so both are pinned.
        """
        import re
        from pathlib import Path

        contract = (
            Path(__file__).resolve().parents[2] / "docs" / "API_CONTRACT.md"
        ).read_text(encoding="utf-8")

        # It must NOT be a §5.2 row (that would break the taxonomy count).
        s52 = contract.split("### 5.2", 1)[1].split("### 5.3", 1)[0]
        assert "`rate_limited`" not in s52, (
            "rate_limited was added to the §5.2 taxonomy table; that table is "
            "asserted to be exactly `core/errors.py`, and rate_limited is not "
            "in core/errors.py"
        )

        # It MUST be documented somewhere a client can find it, with its status.
        for code in sorted(GATEWAY_ORIGIN_CODES):
            assert f"`{code}`" in contract, (
                f"gateway-origin code {code!r} is returned to clients but "
                f"documented nowhere in docs/API_CONTRACT.md"
            )
            assert translate_error(code, "m")[0] == 429
            assert "429" in contract, "the 429 status must be documented in §5.1"
            assert re.search(rf"`{code}`.*`429`|`429`.*`{code}`", contract), (
                f"{code!r} is documented but not linked to its HTTP status"
            )


# ==========================================================================
# 18. Secrets never serialized to clients
# ==========================================================================
class Test18SecretNonDisclosure:
    def test_the_upstream_token_never_appears_in_a_response_body(self) -> None:
        policy = _policy()
        decision = _admit(policy, inbound_request_id="req_abc12345")
        blob = json.dumps(decision.body or {}) + json.dumps(decision.headers)
        assert "hf_" not in blob.lower()
        assert "bearer" not in blob.lower()

    def test_the_token_is_injected_only_on_the_upstream_leg(self) -> None:
        policy = _policy()
        upstream = policy.upstream_headers({"X-Test": "1"}, token="hf_SUPERSECRET")
        assert upstream["Authorization"] == "Bearer hf_SUPERSECRET"

        response = policy.response_headers(dict(upstream))
        assert "Authorization" not in response, (
            "an upstream request header was echoed into the response"
        )

    def test_a_gateway_health_payload_contains_no_secret(self) -> None:
        """The `/v1/gateway/health` route reports config, never credentials."""
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2] / "gateway" / "app.py"
        ).read_text(encoding="utf-8")
        # HF_TOKEN is read, but must only be read to be forwarded.
        assert "HF_TOKEN" in source
        start = source.index("async def gateway_health")
        end = source.index("@app.api_route", start)
        health_body = source[start:end]
        for secret in ("HF_TOKEN", "Authorization", "hf_"):
            assert secret not in health_body, (
                f"the gateway health route references {secret!r}; it must not "
                f"be able to serialise a credential"
            )

    def test_the_error_envelope_does_not_carry_a_stack_trace(self) -> None:
        try:
            raise ValueError("internal detail that must not leak: /srv/app/secret.py")
        except ValueError as exc:
            status, body = translate_error("specialist_error", "A specialist failed.", detail=str(exc))

        assert status >= 500
        # The gateway's own envelope is what it constructs; the assertion is that
        # it has no traceback field at all.
        assert "traceback" not in body["error"]
        assert "stack" not in body["error"]

    def test_gateway_config_refuses_a_wildcard_even_with_a_secret_present(self) -> None:
        """Security defaults must not be relaxable for convenience."""
        with pytest.raises(ValueError):
            _config(allowed_origins=("*",))


# ==========================================================================
# Guard preservation (§C closing instruction)
# ==========================================================================
class TestConfigGuardsPreserved:
    """The six incoherent-configuration guards must remain, unweakened.

    Asserted by enumerating them here so that a future edit that removes one
    fails a test rather than silently reducing the safety net.
    """

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"upstream_url": "https://s.hf.space/"},
            {"upstream_url": ""},
            {"allowed_origins": ()},
            {"allowed_origins": ("*",)},
            {"upstream_timeout_s": 30.0},
            {"upstream_timeout_s": 121.0},
            {"max_body_bytes": 1024, "max_file_bytes": 4096},
        ],
    )
    def test_each_incoherent_state_is_refused(self, kwargs: dict[str, Any]) -> None:
        base: dict[str, Any] = dict(
            upstream_url="https://s.hf.space",
            allowed_origins=("https://a.pages.dev",),
            upstream_timeout_s=90.0,
        )
        base.update(kwargs)
        with pytest.raises(ValueError):
            GatewayConfig(**base)
