"""The API contract must not drift from the schemas it documents.

WHY THIS EXISTS
---------------
`docs/API_CONTRACT.md` is the artifact the frontend agent builds against. Its
JSON examples are the only place a consumer sees the wire format without reading
`core/schemas.py`.

A documentation example that looks plausible but does not validate is worse than
no example: the frontend builds against it, the server rejects the request with
422, and the cause looks like a frontend bug. Writing this contract produced
exactly that failure four times over -- `Box` fields were nested instead of flat,
`Evidence` used `summary`/`value`/`source` instead of
`coordinates`/`score`/`source_specialist`, `CoordinateSystem` values were
shorthand rather than the real `normalized_0_1`/`pixel`/`geo`, and
`ExecutionTrace.inputs`/`outputs` were dicts rather than lists.

Each was caught by validating the examples against the real models, which is what
these tests do permanently.

WHAT IS AND IS NOT ASSERTED
---------------------------
These tests validate the EXAMPLES, not the prose. They cannot detect a wrong
sentence (e.g. a mis-stated latency budget). What they guarantee is that a
frontend developer who copies an example verbatim gets a request the server
accepts -- which is the failure mode that actually costs time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from core.schemas import (
    SCHEMA_VERSION,
    AnalysisRequest,
    CoordinateSystem,
    HealthStatus,
    ResultEnvelope,
    Task,
)

CONTRACT = Path(__file__).resolve().parents[2] / "docs" / "API_CONTRACT.md"


@pytest.fixture(scope="module")
def contract_text() -> str:
    assert CONTRACT.exists(), f"contract document missing: {CONTRACT}"
    return CONTRACT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def json_blocks(contract_text: str) -> list[dict]:
    """Every fenced ```json block in the contract, parsed."""
    blocks = re.findall(r"```json\n(.*?)```", contract_text, re.S)
    parsed = []
    for index, raw in enumerate(blocks):
        try:
            parsed.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"contract json block #{index} is not valid JSON: {exc}"
            ) from exc
    return parsed


def _find(blocks: list[dict], predicate) -> dict:
    for block in blocks:
        if predicate(block):
            return block
    raise AssertionError("no example matching the predicate was found in the contract")


# ---------------------------------------------------------------------------
# The examples must validate
# ---------------------------------------------------------------------------
def test_every_json_block_in_the_contract_parses(json_blocks):
    """A malformed example is unusable; this fails loudly rather than silently."""
    assert json_blocks, "the contract contains no json examples"


def test_the_result_envelope_example_validates(json_blocks):
    example = _find(
        json_blocks,
        lambda b: isinstance(b, dict) and {"run_id", "result", "trace"} <= set(b),
    )
    envelope = ResultEnvelope.model_validate(example)
    assert envelope.schema_version == SCHEMA_VERSION
    assert envelope.result.confidence.method in ("uncalibrated", "temperature_scaling")


def test_no_contract_example_fabricates_an_artifact_uri(json_blocks):
    """F-16 (owner ruling 2026-09-23): v1 has no artifact-serving endpoint.

    The published examples must not promise a URI the service cannot emit, and
    must not carry a filesystem path in `change_map` / `artifact_ref`.

    This walks the PARSED examples rather than the document text on purpose.
    The prose in section 2.4 deliberately quotes the old
    `artifact://run/9f2c.../change_map.png` example when explaining why it was
    removed, so a text search would be tripped by documentation *about* the
    removal -- which is measuring the wrong thing. A guard must fail on the
    defect, not on a description of the defect.
    """

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key, value
                yield from walk(value)
        elif isinstance(node, list):
            for item in node:
                yield from walk(item)

    for block in json_blocks:
        for key, value in walk(block):
            if isinstance(value, str):
                assert not value.startswith("artifact://"), (
                    f"a contract example fabricates an artifact URI ({value!r}); "
                    "v1 has no artifact-serving endpoint"
                )
            if key in ("artifact_ref", "change_map"):
                assert value is None, (
                    f"a contract example carries {key}={value!r}; F-16 forbids "
                    "exposing filesystem paths or fabricating refs"
                )


def test_the_health_example_validates(json_blocks):
    example = _find(
        json_blocks, lambda b: isinstance(b, dict) and {"status", "models"} <= set(b)
    )
    health = HealthStatus.model_validate(example)
    assert health.status in ("ok", "degraded", "error")


def test_the_request_example_validates(json_blocks):
    example = _find(
        json_blocks,
        lambda b: isinstance(b, dict) and {"assets", "query"} <= set(b) and "run_id" in b,
    )
    request = AnalysisRequest.model_validate(example)
    assert request.assets
    assert request.force_task is not None


# ---------------------------------------------------------------------------
# The documented enums must be the real enums
# ---------------------------------------------------------------------------
def test_the_documented_task_values_are_the_real_ones(contract_text: str):
    """A documented task value the enum does not have is a 422 waiting to happen."""
    documented = set(re.findall(r"^\| `([a-z_]+)` \| .* \| \d+ \|$", contract_text, re.M))
    missing = {t.value for t in Task} - documented
    assert not missing, (
        f"Task values absent from the contract's task table: {sorted(missing)}. "
        f"Every enum member must be documented, or a consumer cannot know it exists."
    )


def test_the_documented_coordinate_system_values_are_the_real_ones(contract_text: str):
    """The shorthand `normalized`/`geographic` are NOT the enum values.

    This is a specific, real defect found while writing the contract: the
    documented names were `normalized` and `geographic` while the enum defines
    `normalized_0_1` and `geo`. A frontend reading the documented names would send
    values the server rejects.
    """
    for coordinate_system in CoordinateSystem:
        assert f"`{coordinate_system.value}`" in contract_text, (
            f"CoordinateSystem.{coordinate_system.value} is not documented"
        )

    for wrong in ("`normalized`", "`geographic`"):
        # Guard against the shorthand reappearing as a *value* entry. It may
        # legitimately appear in prose, so only the table-cell form is checked.
        assert f"| {wrong} |" not in contract_text, (
            f"{wrong} appears as a table value but is not a CoordinateSystem member"
        )


def test_the_documented_confidence_fields_match_the_model(contract_text: str):
    """`value` is a property, not a field. Documenting it as one would mislead."""
    from core.schemas import ConfidenceBreakdown

    for field in ConfidenceBreakdown.model_fields:
        assert f"`{field}`" in contract_text, (
            f"ConfidenceBreakdown.{field} is not documented"
        )
    assert "NOT a JSON field" in contract_text, (
        "the contract must warn that `confidence.value` is not serialised"
    )


# ---------------------------------------------------------------------------
# The error taxonomy must be complete
# ---------------------------------------------------------------------------
def test_every_error_code_is_documented(contract_text: str):
    """An undocumented code forces the frontend to guess at user-facing copy."""
    from core.errors import SatQueryError
    import core.errors as errors_module

    codes = set()
    for name in errors_module.__all__:
        obj = getattr(errors_module, name)
        if isinstance(obj, type) and issubclass(obj, SatQueryError):
            codes.add(obj.code)

    undocumented = sorted(c for c in codes if f"`{c}`" not in contract_text)
    assert not undocumented, (
        f"error codes missing from the contract: {undocumented}"
    )


# ---------------------------------------------------------------------------
# The gaps must be stated, not hidden
# ---------------------------------------------------------------------------
def test_the_upload_gap_is_declared_and_its_resolution_is_recorded(contract_text: str):
    """`assets: list[str]` + a 3-endpoint surface cannot both be true without a 4th.

    INVERTED 2026-09-22. This previously required the literal string
    `"NOT IN THE PLAN"` -- i.e. it required the document to say the gap was
    *unresolved*. The owner ruling chose Option A, so the gap is closed; the
    requirement now is that the document records **both** the gap and what
    closed it, rather than quietly deleting the history.

    An unasserted correction is how a resolved decision gets un-resolved: if
    someone later removes the reasoning and keeps only "here is the endpoint",
    the next reader cannot tell whether the design was chosen or drifted into.
    """
    assert "/v1/assets" in contract_text
    assert "Option A" in contract_text, (
        "the contract no longer records which upload design was chosen"
    )
    assert "IMPLEMENTED" in contract_text, (
        "the contract does not state that /v1/assets is implemented; it may "
        "still be describing the endpoint as an open decision"
    )
    assert "NOT IN THE PLAN" in contract_text, (
        "the contract no longer preserves the original 'not in the plan' "
        "finding; the decision record must keep the gap it closed"
    )


def test_the_authentication_boundary_is_recorded(contract_text: str):
    """Absent auth is a plan decision (section 74), not an oversight."""
    assert "There is no authentication in v1" in contract_text
    assert "section 74" in contract_text
    assert "Do not build a login screen" in contract_text


def test_the_endpoint_surface_is_exactly_four_and_documented(contract_text: str):
    """Three plan endpoints plus one added by ruling, each in the table.

    INVERTED 2026-09-22. The original asserted `== {health, capabilities,
    analyze}` with the message *"a fourth must be flagged, not slipped in"*.
    That test did its job: it is the reason the fourth endpoint could not be
    added silently. The ruling then added it deliberately, so the assertion
    moves to the new invariant -- four endpoints, each documented in the core
    table, and `/v1/assets` still carrying its decision record.

    Kept as an exact-set assertion rather than a subset check: a fifth endpoint
    appearing without a decision is still exactly the failure this guards.
    """
    documented = set(re.findall(r"\| `(GET|POST)` \| `(/v1/[a-z]+)` \|", contract_text))
    paths = {path for _method, path in documented}
    assert paths == {
        "/v1/health",
        "/v1/capabilities",
        "/v1/analyze",
        "/v1/assets",
    }, (
        f"the core endpoint table must list exactly the served surface "
        f"(3 from the plan + /v1/assets by ruling); got {sorted(paths)}"
    )


def test_the_schema_version_in_the_contract_matches_the_code(contract_text: str):
    """A version mismatch between doc and code is a silent contract break."""
    assert f'"{SCHEMA_VERSION}"' in contract_text


def test_the_frozen_config_hash_is_recorded(contract_text: str):
    """The frontend may encounter it in a trace; it must be the real value."""
    assert "78f1e3700da15aa1" in contract_text


def test_the_zerogpu_constraints_are_documented(contract_text: str):
    """These determine whether the UI can parallelise. Omitting them misleads."""
    assert "5 GPU-minutes/day" in contract_text
    assert "cache_max_models" in contract_text
    assert "ZeroGPU does not support it" in contract_text or "torch.compile" in contract_text


def test_the_calibration_caveat_is_recorded(contract_text: str):
    """`temperature_scaling` is not inherently 'more accurate'. Measured fact."""
    assert "0.9772731820958189" in contract_text
    assert "ECE" in contract_text


def test_the_coordinate_convention_warning_is_present(contract_text: str):
    """The VRSBench 0-100 vs 0-1 trap is a documented real-world bug source."""
    assert "0–100" in contract_text or "0-100" in contract_text


# ---------------------------------------------------------------------------
# STEP 8 conformance audit -- four claims the contract made that the code
# contradicted. Each was found by EXECUTING the adapter and comparing, not by
# reading. They are pinned here so the correction cannot silently revert.
# ---------------------------------------------------------------------------
def test_the_translation_table_does_not_claim_unreachable_states(contract_text: str):
    """C-4. The published table must not present `loaded` as a live translation.

    Why this is not pedantry: the pre-audit table read
    `available -> loaded` as its FIRST row, i.e. it described the commonest
    registry state as producing a state the adapter cannot emit. `app/deployment.py`
    assigns `state` at exactly four sites and never assigns `"loaded"`, because
    requirement 4 forbids constructing a model to answer a metadata request, so
    "a model is resident" is unobservable. A frontend that branched on `loaded`
    would have a dead branch for every capability.

    The check is deliberately about the CLAIM: the vocabulary section may name
    `loaded` (it is a closed vocabulary a client must tolerate), but it must say
    it is never emitted.
    """
    import app.deployment as dep

    # The code fact, read from the module rather than trusted from the doc.
    import ast

    tree = ast.parse(open(dep.__file__, encoding="utf-8").read())
    assigned: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "state":
                    if isinstance(node.value, ast.Constant):
                        assigned.add(str(node.value.value))
    assert "loaded" not in assigned, (
        f"the adapter now assigns state='loaded'; the contract's story changes. "
        f"assigned: {sorted(assigned)}"
    )
    # ...and the contract must not present the unreachable pair as the mapping.
    assert "| `available` | `loaded` |" not in contract_text, (
        "the contract again claims registry `available` emits `loaded`, which "
        "no code path produces"
    )
    assert "never emitted" in contract_text, (
        "the contract must state plainly which states are never emitted"
    )


def test_the_contract_explains_that_available_coexists_with_not_requested(
    contract_text: str,
):
    """C-4b. The measured healthy state is `available: true` + `not_requested`.

    A reader who sees both will assume a bug. The contract must pre-empt it or
    the frontend will mislabel a working deployment.
    """
    assert "not_requested" in contract_text
    # The two fields answer different questions; the doc must say so.
    assert "can this deployment serve" in contract_text.lower() or (
        "has anything loaded it yet" in contract_text.lower()
    ), "the contract must explain that available and not_requested coexist"


def test_every_field_in_the_capabilities_response_is_declared(contract_text: str):
    """C-3. An undeclared response field breaks a strict reader.

    `torch_compile` was returned by `/v1/capabilities` and documented in NO
    document. Because section 1.1 now warns that unknown fields are REJECTED on
    read and write, an undeclared field in a RESPONSE is a real integration
    hazard, not cosmetics.
    """
    import app.deployment as dep

    body = dep.capabilities_payload()
    # Every top-level key and every deployment key must appear in the contract.
    for key in body:
        assert f"`{key}`" in contract_text or key in contract_text, (
            f"the contract does not document the response field {key!r}"
        )
    for key in body["deployment"]:
        assert key in contract_text, (
            f"the contract does not document deployment.{key}; it is served to "
            f"every client"
        )


def test_the_content_type_allowlist_is_enumerated_not_described(contract_text: str):
    """C-5. "A closed list" is not actionable; the five members must be named.

    The list lived only in code. TIFF is the type the geospatial specialists
    need, and a frontend author reading "a closed list" plus an `image/png`
    example would reasonably conclude PNG/JPEG suffice.
    """
    import app.space_app as space_app_mod

    from gateway.policy import GatewayConfig

    # `GatewayConfig()` cannot be built bare -- `__post_init__` refuses an empty
    # upstream and an empty origin list (the documented fail-at-startup
    # behaviour, section 4.1). Placeholders are enough: the allowlist is a
    # class-level default independent of both.
    allowlist = GatewayConfig(
        upstream_url="https://example.invalid",
        allowed_origins=("https://frontend.invalid",),
    ).allowed_content_types
    assert allowlist, "GatewayConfig must declare an allowlist"
    for media_type in allowlist:
        assert media_type in contract_text, (
            f"the accepted content type {media_type!r} is not documented; a "
            f"client cannot discover it"
        )
    # And the Space's independent copy must not have drifted. It lives in
    # `app/space_app.py` as `_ALLOWED_ASSET_CONTENT_TYPES` -- a deliberate
    # second copy so the Space validates rather than trusting the gateway.
    space_copy = space_app_mod._ALLOWED_ASSET_CONTENT_TYPES
    assert set(space_copy) == set(allowlist), (
        f"the Space's allowlist drifted from the gateway's: "
        f"{set(space_copy) ^ set(allowlist)}"
    )


def test_the_extra_forbid_rule_and_its_single_exception_are_both_documented(
    contract_text: str,
):
    """C-8. The contract claimed EVERY contract-facing model forbids unknown
    fields. `GeoMetadata` allows them, and is reachable in every `/v1/analyze`
    response.

    The claim was stronger than the code, in the one direction a client author
    would act on: they would validate strictly everywhere and break on a
    `geospatial` key the server is entitled to add. The exception was known
    (recorded in `STEP7_BACKEND_CHAIN_REPORT.md`) but had never reached the
    frontend-facing contract.

    This test derives the exception set from the models, so a SECOND permissive
    model fails here rather than silently widening the open surface.
    """
    import inspect

    from pydantic import BaseModel

    import core.schemas as S

    models = {
        name: obj
        for name, obj in vars(S).items()
        if inspect.isclass(obj)
        and issubclass(obj, BaseModel)
        and obj is not BaseModel
        and obj.__module__ == S.__name__
    }
    permissive = sorted(
        name for name, cls in models.items()
        if cls.model_config.get("extra", "ignore") == "allow"
    )
    assert permissive == ["GeoMetadata"], (
        f"the set of models allowing unknown fields changed: {permissive}. The "
        f"contract documents exactly one exception; add the new one there or "
        f"reconsider the schema."
    )
    # ...and the contract must name it, and must not claim an unqualified rule.
    assert "GeoMetadata" in contract_text, (
        "the contract does not name the single permissive model"
    )
    assert 'extra="allow"' in contract_text, (
        "the contract does not state the exception's actual setting"
    )
    assert "geospatial" in contract_text, (
        "the contract does not point at the reachable sub-object"
    )


# ---------------------------------------------------------------------------
# The integration suite's SCOPE — a claim in the contract/report that must stay
# true. `docs/ITEM5_INTEGRATION_SUITE_SCOPE.md` states that no test in
# `tests/integration/` dials a network address, which is why running that suite
# completes without a deployment and why its 26 green tests are not evidence
# about the deployment. That is a property of the FILES, so it is checked
# against the files rather than asserted in prose.
# ---------------------------------------------------------------------------
_INTEGRATION_DIR = Path(__file__).resolve().parents[1] / "integration"

# Hosts that are structurally unreachable rather than merely unused: RFC 2606
# reserves `.test`, so a request to one cannot leave the machine.
_RESERVED_HOST_SUFFIXES = (".test", ".invalid", ".example", ".localhost")

# Things that would mean a real socket is being opened.
_NETWORK_CALL = re.compile(
    r"\b(?:requests\.(?:get|post|put|delete|head|request|Session)"
    r"|urlopen|urlretrieve|socket\.(?:socket|create_connection)"
    r"|http\.client\.HTTP)"
)


def _integration_sources() -> list[Path]:
    return sorted(_INTEGRATION_DIR.glob("*.py"))


def test_the_integration_suite_exists_and_is_what_this_module_describes() -> None:
    """Guard the guard: if the suite moves, the checks below silently stop applying."""
    sources = _integration_sources()
    assert sources, (
        f"no integration sources found under {_INTEGRATION_DIR}; the scope claims "
        f"below would pass vacuously"
    )
    assert any(p.name == "test_step7_backend_chain.py" for p in sources), (
        f"the backend-chain module is no longer present: {[p.name for p in sources]}"
    )


def test_no_integration_test_opens_a_network_socket() -> None:
    """The reason `tests/integration/` completes with no deployment running.

    If this ever fails, the suite has grown a genuine network dependency and
    `docs/ITEM5_INTEGRATION_SUITE_SCOPE.md` section 2 is false -- which would
    change what a green integration run means. Fail loudly rather than let the
    document drift away from the code.
    """
    offenders: list[str] = []
    for path in _integration_sources():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if _NETWORK_CALL.search(line):
                offenders.append(f"{path.name}:{lineno}: {stripped}")
    assert not offenders, (
        "the integration suite has acquired a real network call, so it no longer "
        "runs without a deployment:\n  " + "\n  ".join(offenders)
    )


def test_every_url_the_integration_suite_uses_is_structurally_unreachable() -> None:
    """Every URL must be a reserved-TLD `base_url` for `ASGITransport`.

    A real hostname here would mean the suite depends on something outside this
    machine, and the 26-passing number would no longer be reproducible offline.
    """
    url = re.compile(r"https?://([a-zA-Z0-9._-]+)")
    seen: dict[str, list[str]] = {}
    for path in _integration_sources():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for host in url.findall(line):
                seen.setdefault(host, []).append(f"{path.name}:{lineno}")

    assert seen, "no URLs found at all; this test would be vacuous"
    reachable = sorted(
        h for h in seen if not h.endswith(_RESERVED_HOST_SUFFIXES)
    )
    assert not reachable, (
        f"the integration suite references hosts that are not reserved-TLD "
        f"placeholders: { {h: seen[h] for h in reachable} }. These would need DNS "
        f"and a live service, so the suite could no longer be run offline."
    )


def test_the_scope_document_states_both_halves_of_the_item_5_ruling() -> None:
    """The document is the deliverable here, so pin its load-bearing claims.

    Two claims must survive editing: that the suite does NOT need a deployment,
    and that item 5 is NOT thereby PASS. Dropping either turns a precise
    partial result into a misleading one.
    """
    doc = (
        Path(__file__).resolve().parents[2] / "docs" / "ITEM5_INTEGRATION_SUITE_SCOPE.md"
    )
    assert doc.exists(), f"{doc} is missing; item 5's scope is undocumented"
    text = doc.read_text(encoding="utf-8")

    assert "26 passed" in text, (
        "the document does not record the measured suite result"
    )
    assert "PARTIALLY SATISFIED" in text, (
        "the document must not present item 5 as either PASS or wholly BLOCKED"
    )
    assert "ASGITransport" in text, (
        "the document does not name the mechanism that makes a deployment unnecessary"
    )
    # ...and it must say what the live run would ADD, not only what it would not.
    assert "cache_max_models=1" in text, (
        "the document does not connect item 6 to the live-upstream gap"
    )
    # ...including the gateway gap, which is the non-obvious part.
    assert "no integration coverage" in text, (
        "the document does not record that the gateway has no integration coverage"
    )
