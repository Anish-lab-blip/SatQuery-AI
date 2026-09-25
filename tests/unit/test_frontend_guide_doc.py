"""The contract documents must stay consistent with each other and the schemas.

`docs/API_CONTRACT.md` and `docs/FRONTEND_INTEGRATION.md` are the frontend
agent's primary inputs. They exist so that another engineer can build a client
WITHOUT reading `core/schemas.py`.

That only works if they are accurate. Writing them produced, in order:
`CoordinateSystem` documented as `normalized`/`geographic` (real: `normalized_0_1`
/ `geo`), `Box` geometry documented as nested (real: flat), `Evidence` fields
named `summary`/`value`/`source` (real: `coordinates`/`score`/`source_specialist`),
`ExecutionTrace.inputs`/`outputs` as objects (real: lists), and `Task.unsupported`
missing from both tables.

Every one of those was found by comparing the prose against the model, not by
proof-reading. These tests make that comparison permanent.

WHAT IS NOT ASSERTED
--------------------
Prose can still be wrong in ways these tests cannot see -- a mis-stated latency,
a wrong reason for a decision. What is asserted is that every value a frontend
must send or read is the real value, which is the class of error that produces a
422 and a wasted debugging session.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.schemas import Box, CoordinateSystem, Evidence, EvidenceType, Region, Task

DOCS = Path(__file__).resolve().parents[2] / "docs"
API = DOCS / "API_CONTRACT.md"
FRONTEND = DOCS / "FRONTEND_INTEGRATION.md"


@pytest.fixture(scope="module")
def api_text() -> str:
    assert API.exists(), f"missing {API}"
    return API.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontend_text() -> str:
    assert FRONTEND.exists(), f"missing {FRONTEND}"
    return FRONTEND.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
def test_every_task_is_documented_in_both_documents(api_text, frontend_text):
    for task in Task:
        assert f"`{task.value}`" in api_text, f"Task.{task.value} missing from API contract"
        assert f"`{task.value}`" in frontend_text, (
            f"Task.{task.value} missing from the frontend guide; a value the client "
            f"may receive but cannot find documented is a crash waiting to happen"
        )


def test_every_coordinate_system_is_documented_in_both(api_text, frontend_text):
    for coordinate_system in CoordinateSystem:
        assert f"`{coordinate_system.value}`" in api_text
        assert f"`{coordinate_system.value}`" in frontend_text


def test_every_evidence_type_is_documented_in_both(api_text, frontend_text):
    for evidence_type in EvidenceType:
        assert f"`{evidence_type.value}`" in api_text, (
            f"EvidenceType.{evidence_type.value} missing from the API contract"
        )
        assert evidence_type.value in frontend_text, (
            f"EvidenceType.{evidence_type.value} missing from the frontend guide"
        )


# ---------------------------------------------------------------------------
# Geometry — the shape mismatch trap
# ---------------------------------------------------------------------------
def test_the_geometry_shape_claims_match_the_models(api_text, frontend_text):
    """`Box` is flat; `Region` nests a `box`. Documenting them as alike breaks clients."""
    assert {"x1", "y1", "x2", "y2"} <= set(Box.model_fields)
    assert "box" in Region.model_fields
    assert "box" not in Box.model_fields, (
        "Box gained a nested `box`; both documents claim it is flat and must be updated"
    )

    for name, text in (("API contract", api_text), ("frontend guide", frontend_text)):
        assert "flat" in text.lower(), (
            f"the {name} must state that Box geometry is flat; an implementer who "
            f"assumes a nested object writes a renderer that silently draws nothing"
        )


def test_the_evidence_fields_documented_are_the_real_field_names(api_text):
    """`summary`/`value`/`source` are plausible and wrong; the real names differ.

    Checked against the JSON EXAMPLE rather than the whole document, because the
    document legitimately says things like "`score` — not `value`". A frontend
    copies the example, so the example is where a wrong name does damage.
    """
    import json
    import re

    for field in Evidence.model_fields:
        assert f"`{field}`" in api_text, f"Evidence.{field} is undocumented"

    envelope = None
    for raw in re.findall(r"```json\n(.*?)```", api_text, re.S):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "result" in obj and "evidence" in obj.get("result", {}):
            envelope = obj
            break

    assert envelope is not None, "no ResultEnvelope example with evidence was found"
    for item in envelope["result"]["evidence"]:
        assert set(item) <= set(Evidence.model_fields), (
            f"the example evidence item carries fields the model does not define: "
            f"{sorted(set(item) - set(Evidence.model_fields))}. A frontend copying "
            f"this example would send a body the server rejects."
        )
        for wrong in ("summary", "value", "source"):
            assert wrong not in item, (
                f"the example evidence item uses `{wrong}`, which is not an Evidence "
                f"field"
            )


# ---------------------------------------------------------------------------
# Cross-document consistency
# ---------------------------------------------------------------------------
def test_the_frontend_guide_delegates_the_contract_to_the_api_document(frontend_text):
    """Two drifting copies of the same contract is worse than one."""
    assert "API_CONTRACT.md" in frontend_text


def test_both_documents_agree_there_are_four_endpoints(api_text, frontend_text):
    """All four served paths must appear in both documents.

    UPDATED 2026-09-22: `/v1/assets` joined the surface by owner ruling. A
    frontend guide that omitted it would leave the upload flow undocumented in
    the one document the frontend agent reads.
    """
    for path in ("/v1/health", "/v1/capabilities", "/v1/analyze", "/v1/assets"):
        assert path in api_text, f"{path} is missing from the API contract"
        assert path in frontend_text, f"{path} is missing from the frontend guide"


def test_both_documents_declare_the_upload_decision_and_how_to_use_it(
    frontend_text, api_text
):
    """INVERTED 2026-09-22: the upload gap is closed, so the guide must teach it.

    This previously required `"BLOCKED" in frontend_text` -- the guide had to
    tell the frontend agent *not* to build uploads, because the endpoint was
    undefined. The ruling defined it, so the requirement flips: the guide must
    now contain an actual upload procedure, and must still state the three
    constraints the ruling attached (size limit, content-type allowlist,
    retry semantics).
    """
    assert "/v1/assets" in api_text
    assert "upload" in frontend_text.lower()

    assert "BLOCKED" not in frontend_text, (
        "the frontend guide still tells the agent upload is blocked; the "
        "endpoint is implemented and served"
    )
    # The guide must be actionable, not merely no-longer-blocked.
    for required in ("FormData", "asset_id", "expires_at"):
        assert required in frontend_text, (
            f"the upload section does not mention {required!r}; a guide that "
            f"unblocks a feature without showing how to use it is worse than "
            f"one that blocks it"
        )
    # The three constraints the ruling named.
    assert "413" in frontend_text and "415" in frontend_text, (
        "the upload section does not document the size and type refusals"
    )
    assert "retry" in frontend_text.lower(), (
        "the upload section does not document the retry/idempotency story"
    )


def test_both_documents_forbid_a_login_screen(api_text, frontend_text):
    """Auth is excluded by plan section 74; a login screen would be unbuildable."""
    assert "Do not build a login screen" in api_text
    assert "Do not build a login screen" in frontend_text


def test_both_documents_require_no_secrets_in_the_browser(api_text, frontend_text):
    assert "HF token" in api_text
    assert "No secrets in the browser" in frontend_text


def test_the_frontend_guide_states_the_confidence_property_trap(frontend_text):
    """Reading `confidence.value` yields undefined; the guide must say so."""
    assert "c.value" in frontend_text or "confidence.value" in frontend_text
    assert "calibrated" in frontend_text and "raw" in frontend_text


def test_the_frontend_guide_forbids_auto_retry(frontend_text):
    """An auto-retry loop can exhaust a 5 GPU-minute daily budget."""
    assert "Never automatically retry" in frontend_text or "never automatically retry" in frontend_text.lower()


def test_the_frontend_guide_requires_serialized_analyses(frontend_text):
    """`cache_max_models: 1` means concurrent requests evict each other."""
    assert "erializ" in frontend_text
    assert "5 GPU-min" in frontend_text or "5 GPU-minutes/day" in frontend_text


def test_the_frontend_guide_lists_the_defect_codes(frontend_text):
    """These must be captured, not explained away."""
    for code in (
        "schema_validation_error",
        "coordinate_error",
        "confidence_range_error",
        "model_load_error",
    ):
        assert code in frontend_text, f"defect code {code} not listed for the frontend"


def test_the_frontend_guide_states_that_nothing_is_deployed(frontend_text):
    """The frontend must not build against an imagined deployment."""
    assert "not deployed" in frontend_text or "not deployed" in frontend_text.lower()
    assert "No live URL exists yet" in frontend_text


# ---------------------------------------------------------------------------
# STEP 8 conformance audit -- C-4b / C-5, frontend-facing half
# ---------------------------------------------------------------------------
def test_the_guide_warns_that_not_requested_is_not_a_fault(frontend_text):
    """C-4b. The measured healthy state pairs `available: true` with
    `not_requested`. A frontend that reads the latter as a failure disables
    every task on a perfectly working deployment, so the guide must say so.
    """
    assert "not_requested" in frontend_text, (
        "the guide never mentions the state that every capability reports by "
        "default; a frontend author cannot interpret /v1/capabilities without it"
    )
    lowered = frontend_text.lower()
    assert "do not treat" in lowered or "not a fault" in lowered or (
        "not an error" in lowered
    ), "the guide must state plainly that not_requested is not a failure"


def test_the_guide_enumerates_the_accepted_upload_types(frontend_text):
    """C-5. TIFF is required for the geospatial tasks; naming it is not optional.

    `application/octet-stream` matters too: browsers often fail to label a
    `.tif`, and the server refuses an unlabelled part rather than guessing.
    """
    assert "image/tiff" in frontend_text, (
        "the guide does not name TIFF, which the change and optical/SAR tasks "
        "require"
    )
    assert "application/octet-stream" in frontend_text, (
        "the guide does not name the generic fallback type; a browser that "
        "omits the type would be refused with 415 and no explanation"
    )
    assert "415" in frontend_text, "the guide does not mention the refusal code"
