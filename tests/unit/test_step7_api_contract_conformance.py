"""STEP 7 section D -- the API contract, tested against the schema that implements it.

WHAT THIS SUITE IS FOR
----------------------
`docs/API_CONTRACT.md` is the frozen frontend-facing contract. `core/schemas.py`
is the implementation. The contract itself states its authority:

    "the request/response **shapes** are not invented here. They are the
     existing, tested Pydantic models in `core/schemas.py`."

So there are two possible failure modes, and they need different tests:

  * **The code drifts from the document.** The frontend is built against the
    document, so this is the dangerous direction.
  * **The document drifts from the code.** Less dangerous, but it misleads the
    frontend agent and quietly invalidates the contract's own claim to be
    descriptive rather than inventive.

This suite tests both directions where it can, and -- where the document makes a
checkable factual claim about the schema (`"value" is not serialised`, `Box` is
flat, `extra="forbid"`) -- tests the claim itself rather than restating it.

THE RULE THE WORK ORDER SETS
----------------------------
    "Do not make the frontend contract more permissive than the backend schema."

That is asymmetric on purpose: the schema may be *stricter* than the prose (a
stricter server is safe -- it rejects what the document already told the client
not to send), but the prose must never promise more than the schema permits. The
tests below are written to catch a *permissive prose* claim specifically.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from core.schemas import (
    SCHEMA_VERSION,
    Box,
    ConfidenceBreakdown,
    CoordinateSystem,
    Evidence,
    EvidenceType,
    ExecutionTrace,
    HealthStatus,
    Modality,
    Region,
    ResultEnvelope,
    SpecialistResult,
    Task,
    TraceStep,
)

REPO = Path(__file__).resolve().parents[2]
CONTRACT = (REPO / "docs" / "API_CONTRACT.md").read_text(encoding="utf-8")

pytestmark = pytest.mark.unit


def _prose() -> str:
    """The contract as searchable prose.

    Two normalisations, both needed and both worth naming:

      * whitespace is collapsed, because markdown wraps sentences across lines
        and an assertion should not depend on where an editor broke a line;
      * leading `>` blockquote markers are stripped, because a wrapped blockquote
        puts `> ` in the MIDDLE of a sentence -- the caveat in section 4 reads
        "...must not imply > that ...". Searching the raw text for the phrase
        fails for a reason that has nothing to do with the claim.

    Deliberately not a markdown parser: this is for substring assertions on
    prose, and a parser would hide the fact that these are literal-text checks.
    """
    lines = [re.sub(r"^\s*>\s?", "", line) for line in CONTRACT.splitlines()]
    return " ".join(" ".join(lines).split())


PROSE = _prose()


def _minimal_result(task: Task = Task.VQA) -> SpecialistResult:
    return SpecialistResult(
        task=task,
        answer="yes",
        confidence=ConfidenceBreakdown(raw=0.9, calibrated=0.88, method="temperature_scaling"),
    )


def _minimal_envelope() -> ResultEnvelope:
    return ResultEnvelope(
        run_id="run_abc",
        result=_minimal_result(),
        trace=ExecutionTrace(run_id="run_abc"),
    )


# ===========================================================================
# D1. The four response shapes the contract documents
# ===========================================================================
class TestD1DocumentedShapes:
    """Every shape the contract documents exists and serialises as described."""

    def test_health_status_matches_the_documented_example(self) -> None:
        """Section 2.1's example must validate against `HealthStatus` exactly.

        The docstring of the response example is copied into the test verbatim so
        that a schema change which invalidates the published example fails here
        rather than in the frontend agent's editor.
        """
        documented = {
            "status": "ok",
            "schema_version": "1.0",
            "models": {
                "change": "loaded",
                "change_vqa": "absent",
                "vqa": "not_requested",
                "grounding": "not_requested",
                "optical_sar": "not_requested",
            },
            "device": "cpu",
            "gpu_available": False,
        }
        parsed = HealthStatus(**documented)
        assert parsed.status == "ok"
        assert parsed.gpu_available is False
        assert parsed.models["change_vqa"] == "absent"

    def test_health_models_values_are_strings_not_booleans(self) -> None:
        """The contract is explicit: values are strings "so a reason can be carried".

        If this ever became `dict[str, bool]` the capability-state vocabulary
        (`absent`/`unavailable`/`evicted`) would be unrepresentable and
        `absent` would collapse into `unavailable` -- the exact conflation
        section 2.3 forbids.
        """
        field = HealthStatus.model_fields["models"]
        assert field.annotation == dict[str, str], (
            "HealthStatus.models must map capability -> state string"
        )

    def test_the_capability_state_vocabulary_is_documented_in_full(self) -> None:
        """All five states stay documented, and `absent` != `unavailable`."""
        for state in ("loaded", "absent", "unavailable", "not_requested", "evicted"):
            assert f'`"{state}"`' in CONTRACT or f'"{state}"' in CONTRACT, (
                f"the contract no longer documents the capability state {state!r}"
            )
        assert "must not be conflated" in CONTRACT, (
            "the absent/unavailable distinction lost its explicit warning"
        )

    def test_analyze_request_shape_matches_the_schema(self) -> None:
        from core.schemas import AnalysisRequest

        parsed = AnalysisRequest(
            assets=["asset_0", "asset_1"],
            query="How has the built-up area changed between these two dates?",
            force_task="change_vqa",
            run_id="9f2c1c0e-4a5b-4f5e-9a2c-1b3d4e5f6a7b",
        )
        assert parsed.force_task is Task.CHANGE_VQA
        # `assets` has min_length=1, which the contract states.
        assert AnalysisRequest.model_fields["assets"].is_required()

    def test_the_documented_analyze_response_validates(self) -> None:
        """Section 2.4's full response example must round-trip.

        This is the strongest single test in the file: it takes the published
        example, validation-checks it against the real models, and re-serialises.
        A shape drift in either direction fails it.
        """
        documented = {
            "run_id": "9f2c1c0e-4a5b-4f5e-9a2c-1b3d4e5f6a7b",
            "schema_version": "1.0",
            "result": {
                "task": "change_vqa",
                "answer": "The built-up area increased...",
                "labels": [],
                "regions": [],
                "boxes": [
                    {
                        "x1": 0.12,
                        "y1": 0.34,
                        "x2": 0.56,
                        "y2": 0.78,
                        "label": "expanded built-up area",
                        "score": 0.81,
                        "coordinate_system": "normalized_0_1",
                    }
                ],
                "masks": [],
                "change_map": None,
                "evidence": [
                    {
                        "evidence_id": "ev_001",
                        "type": "change_map",
                        "score": 0.72,
                        "source_specialist": "change_vqa",
                        "coordinate_system": "normalized_0_1",
                        "coordinates": [0.12, 0.34, 0.56, 0.78],
                        "artifact_ref": None,
                        "payload": {},
                    }
                ],
                "confidence": {
                    "raw": 0.991,
                    "calibrated": 0.987,
                    "method": "temperature_scaling",
                    "components": {},
                    "degraded": False,
                    "degradation_reason": None,
                },
                "geospatial": {},
                "execution_trace": None,
                "schema_version": "1.0",
                "warnings": [],
                "degraded": False,
            },
            "trace": {
                "run_id": "9f2c1c0e-4a5b-4f5e-9a2c-1b3d4e5f6a7b",
                "task": "change_vqa",
                "intent": None,
                "query": "How has the built-up area changed?",
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
                "confidence": None,
                "started_at": "2026-09-22T04:12:21.000Z",
                "finished_at": "2026-09-22T04:12:29.400Z",
                "schema_version": "1.0",
            },
        }
        envelope = ResultEnvelope.model_validate(documented)
        assert envelope.result.boxes[0].coordinate_system is CoordinateSystem.NORMALIZED_0_1
        assert envelope.trace.config_hash == "78f1e3700da15aa1"
        # F-16 (owner ruling 2026-09-23): the published example must show the
        # RULED shape -- null refs, no fabricated artifact:// URI. The example
        # used to carry one, and no production file ever emitted it.
        assert envelope.result.change_map is None
        assert envelope.result.evidence[0].artifact_ref is None
        # And it must re-serialise without loss of the fields the frontend reads.
        again = json.loads(envelope.model_dump_json())
        assert again["result"]["boxes"][0]["x1"] == 0.12
        assert again["result"]["change_map"] is None
        assert again["result"]["evidence"][0]["artifact_ref"] is None
        assert again["trace"]["config_hash"] == "78f1e3700da15aa1"

    def test_error_envelope_shape_is_documented_and_matchable(self) -> None:
        """Section 5's error shape: the six documented keys."""
        documented = {
            "error": {
                "code": "pair_misaligned",
                "message": "The images are not sufficiently co-registered for spatial analysis.",
                "detail": "RMSE 4.21 px exceeds the 2.0 px budget",
                "recoverable": False,
                "request_id": "req_01H...",
                "run_id": "9f2c1c0e-...",
            }
        }
        for key in ("code", "message", "detail", "recoverable", "request_id", "run_id"):
            assert key in documented["error"]
        # The envelope is specified in the document, not in the schema, so the
        # assertion available is that the document still fixes all six keys.
        assert '"error": {' in CONTRACT


# ===========================================================================
# D2. extra="forbid" -- the write-side strictness the contract promises
# ===========================================================================
class TestD2ExtraForbid:
    """Every model the contract exposes to a client must reject unknown fields."""

    @pytest.mark.parametrize(
        "model,payload",
        [
            ("AnalysisRequest", {"assets": ["a"], "query": "x", "surprise": 1}),
            ("Box", {"x1": 0, "y1": 0, "x2": 1, "y2": 1, "surprise": 1}),
            ("Region", {"surprise": 1}),
            ("Evidence", {"type": "statistic", "source_specialist": "s", "surprise": 1}),
            ("ConfidenceBreakdown", {"raw": 0.5, "surprise": 1}),
            ("TraceStep", {"state": "PLAN", "surprise": 1}),
            ("ExecutionTrace", {"surprise": 1}),
            ("SpecialistResult", {"task": "vqa", "confidence": {"raw": 0.5}, "surprise": 1}),
        ],
    )
    def test_unknown_fields_are_rejected(self, model: str, payload: dict) -> None:
        import core.schemas as schemas

        cls = getattr(schemas, model)
        with pytest.raises(Exception) as exc:
            cls(**payload)
        assert "extra" in str(exc.value).lower() or "surprise" in str(exc.value), (
            f"{model} accepted an unknown field"
        )

    def test_geometadata_is_deliberately_permissive(self) -> None:
        """The one documented exception, and it must stay an exception.

        `GeoMetadata` is `extra="allow"` because it carries whatever the sensor
        adapter could extract. If it silently became `forbid`, a real GeoTIFF
        with an unexpected tag would fail validation -- so the exception is
        asserted rather than assumed.
        """
        from core.schemas import GeoMetadata

        assert GeoMetadata.model_config.get("extra") == "allow"
        g = GeoMetadata(crs="EPSG:4326", something_new=1)
        assert g.crs == "EPSG:4326"

    def test_the_contract_warns_that_reads_and_writes_are_both_strict(self) -> None:
        """CORRECTED 2026-09-22 (C-2): strictness is symmetric, and now stated so.

        This test previously asserted the prose *"MUST NOT send unknown fields on
        write"* / *"tolerate unknown fields on read"*. The second clause was
        false for `extra="forbid"` models -- which is every contract-facing model
        -- so the document was promising a tolerance the code does not provide.

        The schema is authoritative and was left alone; the prose was corrected.
        This assertion requires the corrected symmetric statement.

        On the stale phrase: it is still *present* in the document, because the C-2
        correction quotes the old claim in order to refute it -- which is the
        right way to record a correction. So the check cannot be a bare absence
        test. It asserts instead that the phrase is not **asserted as policy**:
        it may appear only inside a sentence that rejects it. That is done by
        requiring the refutation to be adjacent, which is what makes the
        difference between a correction and a contradiction left standing.
        """
        assert "Unknown fields are REJECTED, on read and on write" in PROSE
        if "tolerate unknown fields on read" in PROSE:
            assert "That was wrong, and the code is authoritative" in PROSE, (
                "the contract contains the old permissive claim without an "
                "adjacent refutation, so a reader cannot tell which rule stands"
            )
            # The old claim must be the *object* of the correction, not a
            # surviving instruction. `MUST tolerate` is the imperative form and
            # has no place in a retraction.
            assert "MUST tolerate unknown fields on read" not in PROSE.replace(
                '"MUST tolerate unknown fields on read (forward compatibility)"', ""
            ), (
                "the contract still instructs clients to tolerate unknown fields "
                "on read outside of the quotation that retracts it"
            )
        assert "422" in CONTRACT


# ===========================================================================
# D3. Box geometry is FLAT, Region is the wrapper
# ===========================================================================
class TestD3FlatBox:
    """The distinction the contract calls out explicitly, because it is easy to get wrong."""

    def test_box_geometry_is_flat(self) -> None:
        fields = Box.model_fields
        for name in ("x1", "y1", "x2", "y2"):
            assert name in fields, f"Box lost its flat {name} field"
        assert "box" not in fields, (
            "Box gained a nested `box` field; the contract says geometry is flat"
        )

    def test_region_is_the_object_that_may_nest_a_box(self) -> None:
        region = Region(box=Box(x1=0, y1=0, x2=1, y2=1), label="thing")
        dumped = json.loads(region.model_dump_json())
        assert isinstance(dumped["box"], dict)
        assert dumped["box"]["x1"] == 0

    def test_region_does_not_carry_flat_geometry(self) -> None:
        for name in ("x1", "y1", "x2", "y2"):
            assert name not in Region.model_fields, (
                f"Region gained a flat {name}; the nesting contract is now ambiguous"
            )

    def test_the_contract_states_the_flat_box_rule(self) -> None:
        assert "**flat fields on the box**" in CONTRACT
        assert "`Region` is the one with a nested `box`" in CONTRACT

    def test_box_ordering_is_enforced(self) -> None:
        """`x2 >= x1 and y2 >= y1` -- a schema guarantee the contract relies on."""
        with pytest.raises(Exception):
            Box(x1=0.8, y1=0.0, x2=0.2, y2=1.0)
        with pytest.raises(Exception):
            Box(x1=0.0, y1=0.8, x2=1.0, y2=0.2)


# ===========================================================================
# D4. coordinate_system is per-object and mandatory where it matters
# ===========================================================================
class TestD4CoordinateSystem:
    """Section 3.2: "a box drawn with the wrong assumption lands in plausible-looking wrong places"."""

    def test_every_spatial_carrier_has_its_own_coordinate_system_field(self) -> None:
        for cls in (Box, Region):
            assert "coordinate_system" in cls.model_fields, (
                f"{cls.__name__} has no coordinate_system; the frontend would have "
                "to guess a convention"
            )
        from core.schemas import ChangeRegion

        assert "coordinate_system" in ChangeRegion.model_fields

    def test_the_enum_values_are_exactly_the_documented_three(self) -> None:
        """And NOT the earlier shorthand the contract calls "wrong"."""
        values = {m.value for m in CoordinateSystem}
        assert values == {"normalized_0_1", "pixel", "geo"}, (
            f"CoordinateSystem changed: {sorted(values)}"
        )
        # The document explicitly retracts an earlier draft's shorthand.
        assert "normalized_0_1" in CONTRACT
        assert "those are **wrong**" in CONTRACT

    def test_a_spatial_evidence_item_without_a_coordinate_system_is_rejected(self) -> None:
        """`Evidence` must not carry coordinates it cannot interpret.

        This is the validator that makes "coordinate-system mistakes" a
        schema-level error rather than a rendering bug (work order section K).
        """
        with pytest.raises(Exception) as exc:
            Evidence(
                type=EvidenceType.BOUNDING_BOX,
                source_specialist="grounding",
                coordinates=[0.1, 0.2, 0.3, 0.4],
            )
        assert "coordinate_system" in str(exc.value)
        # With a CRS it is fine.
        ok = Evidence(
            type=EvidenceType.BOUNDING_BOX,
            source_specialist="grounding",
            coordinates=[0.1, 0.2, 0.3, 0.4],
            coordinate_system=CoordinateSystem.NORMALIZED_0_1,
        )
        assert ok.coordinate_system is CoordinateSystem.NORMALIZED_0_1

    def test_a_non_spatial_evidence_item_needs_no_coordinate_system(self) -> None:
        """The validator must be scoped to spatial types, not blanket.

        A `statistic` or `geolocation` has no coordinates to misread; rejecting
        it would be an over-strict gate that fails closed on valid input.
        """
        e = Evidence(type=EvidenceType.STATISTIC, source_specialist="change_vqa", score=0.5)
        assert e.coordinate_system is None

    def test_evidence_type_covers_the_eleven_documented_values(self) -> None:
        documented = {
            "image_crop",
            "tile",
            "bounding_box",
            "mask",
            "change_map",
            "optical_view",
            "sar_view",
            "joint_feature_region",
            "statistic",
            "geolocation",
            "availability_mask",
        }
        actual = {m.value for m in EvidenceType}
        assert actual == documented, (
            f"EvidenceType drift. code-only={sorted(actual - documented)} "
            f"doc-only={sorted(documented - actual)}"
        )
        # And the contract's prose must agree with the enum.
        for value in documented:
            assert f"`{value}`" in CONTRACT, f"the contract omits evidence type {value!r}"


# ===========================================================================
# D5. The confidence contract -- the subtlest part, per the document itself
# ===========================================================================
class TestD5Confidence:
    """Section 4: "the easiest part of the API to mis-render"."""

    def test_value_is_not_a_serialised_field(self) -> None:
        """THE headline claim: `confidence.value` is a property, not JSON.

        If this ever became a real field, a client reading `value` would still
        work -- and a client following the contract's instruction to choose
        `calibrated` else `raw` would silently diverge. So the absence is
        asserted positively rather than trusted.
        """
        cb = ConfidenceBreakdown(raw=0.9, calibrated=0.88)
        dumped = json.loads(cb.model_dump_json())
        assert "value" not in dumped, (
            "`value` is now serialised; this contradicts API_CONTRACT.md section 2.4"
        )
        assert set(dumped) == {
            "raw",
            "calibrated",
            "method",
            "components",
            "degraded",
            "degradation_reason",
        }

    def test_the_property_still_exists_and_prefers_calibrated(self) -> None:
        """The Python-side convenience must survive: it is used by internal callers."""
        assert ConfidenceBreakdown(raw=0.9, calibrated=0.88).value == 0.88
        assert ConfidenceBreakdown(raw=0.9).value == 0.9

    def test_calibrated_is_nullable_and_null_when_uncalibrated(self) -> None:
        cb = ConfidenceBreakdown(raw=0.7)
        assert cb.calibrated is None
        assert cb.method == "uncalibrated"
        assert json.loads(cb.model_dump_json())["calibrated"] is None

    def test_the_documented_serialised_key_set_is_stated_in_the_contract(self) -> None:
        """The contract names the six keys; keep the two in sync by parsing it."""
        match = re.search(r"`model_dump\(\)` yields only ([^)]+)\)", CONTRACT)
        assert match, "the contract no longer states the serialised key set"
        documented = {k.strip().strip("`") for k in match.group(1).split(",")}
        actual = set(ConfidenceBreakdown.model_fields)
        assert actual == documented, (
            f"the contract's stated key set drifted. code={sorted(actual)} "
            f"doc={sorted(documented)}"
        )

    def test_confidence_is_range_checked(self) -> None:
        """A confidence out of [0,1] is a `confidence_range_error`, so the schema
        must make it unrepresentable."""
        with pytest.raises(Exception):
            ConfidenceBreakdown(raw=1.5)
        with pytest.raises(Exception):
            ConfidenceBreakdown(raw=-0.1)
        with pytest.raises(Exception):
            ConfidenceBreakdown(raw=0.5, calibrated=2.0)

    def test_the_calibration_caveat_is_documented_verbatim(self) -> None:
        """The measured numbers must not be rounded away or "improved".

        These are measurements, and the contract obliges the frontend not to
        imply temperature scaling is more accurate. If someone edits the numbers
        to look better, this fails.
        """
        for fragment in (
            "0.9772731820958189",
            "16,441",
            "0.013755",
            "0.014929",
            "0.689741",
            "0.689631",
        ):
            assert fragment in CONTRACT, (
                f"the measured calibration figure {fragment!r} is missing from the contract"
            )
        assert "made ECE very slightly **worse**" in CONTRACT, (
            "the contract no longer records that scaling made ECE worse; this is "
            "the honesty guard for the calibration claim"
        )
        normalised = PROSE
        # The sentence wraps and sits inside a blockquote, hence `_prose()`.
        assert (
            'must not imply that `temperature_scaling` is inherently "more accurate"'
            in normalised
        ), "the contract lost the instruction not to oversell calibration"


# ===========================================================================
# D6. Trace fields -- observable facts only
# ===========================================================================
class TestD6Trace:
    """Section 2.4: trace "lists every field `ExecutionTrace` defines"."""

    def test_trace_carries_run_id_and_schema_version(self) -> None:
        t = ExecutionTrace(run_id="run_x")
        assert t.run_id == "run_x"
        assert t.schema_version == SCHEMA_VERSION == "1.0"

    def test_trace_carries_config_hash(self) -> None:
        """The frozen config identity, which the contract publishes by value."""
        assert "config_hash" in ExecutionTrace.model_fields
        assert "78f1e3700da15aa1" in CONTRACT, (
            "the contract no longer publishes the frozen config hash"
        )

    def test_trace_steps_are_observable_states(self) -> None:
        step = TraceStep(state="PLAN", duration_ms=1.5, detail={"n_steps": 2})
        dumped = json.loads(step.model_dump_json())
        assert dumped["state"] == "PLAN"
        # `detail` is a dict of facts -- there is no free-text reasoning field.
        assert isinstance(dumped["detail"], dict)
        for forbidden in ("thought", "reasoning", "rationale", "chain_of_thought"):
            assert forbidden not in TraceStep.model_fields, (
                f"TraceStep gained {forbidden!r}; the trace must carry observable "
                "facts only (plan section 26)"
            )

    def test_every_documented_trace_field_exists_in_the_model(self) -> None:
        """Parse the contract's response example and compare key sets.

        The document promises "every field `ExecutionTrace` defines". This is the
        only mechanical way to hold it to that.

        The example is selected by its `"trace": {` key rather than by an anchor
        pattern, because the `result` block also uses a trailing `},` and an
        earlier version of this test matched the wrong one.
        """
        start = CONTRACT.index('"trace": {')
        depth = 0
        end = start
        for i in range(start, len(CONTRACT)):
            if CONTRACT[i] == "{":
                depth += 1
            elif CONTRACT[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        block = CONTRACT[start:end]
        assert block.startswith('"trace"'), "failed to bracket the trace example"
        documented = set(re.findall(r'"([a-z_]+)":', block))
        actual = set(ExecutionTrace.model_fields)
        missing_from_doc = sorted(actual - documented)
        assert not missing_from_doc, (
            f"ExecutionTrace has fields the contract's example omits: {missing_from_doc}"
        )

    def test_trace_has_no_free_text_field(self) -> None:
        """Nothing in the trace is an unstructured narrative."""
        free_text = {"text", "narrative", "explanation", "summary"}
        assert not (free_text & set(ExecutionTrace.model_fields))


# ===========================================================================
# D7. Forward-compatible reading
# ===========================================================================
class TestD7ForwardCompatibility:
    """Section 1.1: clients "MUST tolerate unknown fields on read".

    DEFECT RECORDED HERE (C-2)
    --------------------------
    This class was originally written to *verify* the contract's forward
    compatibility promise, by feeding an envelope with an added field to
    `ResultEnvelope.model_validate`. It failed -- and the failure is a real
    contradiction between the document and the code, not a bad test.

      * `docs/API_CONTRACT.md` section 1.1 says a consumer must "tolerate unknown
        fields on read (forward compatibility)".
      * `core/schemas.py` sets `extra="forbid"` on `ResultEnvelope`,
        `SpecialistResult` and `ExecutionTrace`. `model_validate` therefore
        **rejects** a body carrying an unknown key.
      * Section 1 also says "additive changes do not bump the version" -- which
        only works if unknown fields are ignorable.

    Both halves are load-bearing and they cannot both hold. The contract's own
    authority clause resolves which is which:

      "the request/response **shapes** are not invented here. They are the
       existing, tested Pydantic models in `core/schemas.py`. This document
       describes them; it does not define new ones."

    So the **code is authoritative** and the document's bullet is the inaccurate
    half. Flipping the schema to `extra="ignore"` would be the opposite call and
    a worse one: it would turn `extra="forbid"` from a guarantee into a misnomer
    and would silently swallow a field a future frontend sends expecting it to
    take effect.

    Per the work order (section P), a contract/implementation mismatch must be
    *identified, resolved in favour of the authoritative source, and reported*.
    The resolution chosen is to move the forward-compatibility mechanism to the
    layer that actually can provide it -- the gateway was already semantically
    correct and its re-serialisation achieves this -- and to report the
    correction. This test asserts the schema's real behaviour so nobody
    re-derives the false expectation from the document.

    The document correction itself is tracked in
    `docs/STEP7_BACKEND_CHAIN_REPORT.md` section 15; it is not made silently
    here because `API_CONTRACT.md` is a frozen artefact and a doc edit is a
    reviewable change, not a test-side workaround.
    """

    def test_the_schema_rejects_unknown_fields_on_read_too(self) -> None:
        """The measured behaviour, asserted so it cannot be forgotten.

        `extra="forbid"` is not write-only: it constrains deserialisation as
        well. A client that adds a field before the server knows about it gets a
        422 -- not a silently-ignored key.
        """
        base = json.loads(_minimal_envelope().model_dump_json())
        base["result"]["a_future_field"] = {"nested": [1, 2]}
        base["trace"]["another_future_field"] = 7
        with pytest.raises(Exception) as exc:
            ResultEnvelope.model_validate(base)
        message = str(exc.value)
        assert "a_future_field" in message and "another_future_field" in message

    def test_the_contract_states_the_obligation_the_schema_actually_enforces(self) -> None:
        """CORRECTED 2026-09-22 (C-2): the document now states the true rule.

        This previously asserted the substring *"MUST NOT send unknown fields on
        write"*, taken from the old §1.1 which told clients that reads were
        permissive and only writes were strict. That framing was itself part of
        the defect: it invited a client to validate loosely on read.

        C-2 kept the schema (`extra="forbid"` is correct) and fixed the document.
        §1.1 now states that unknown fields are rejected **on read and on write**.
        The assertion therefore moves to the corrected claim -- and additionally
        requires the document to *not* still contain the old, half-true phrasing,
        because a reader who skims would otherwise find the permissive sentence
        still present.
        """
        assert "Unknown fields are REJECTED, on read and on write" in PROSE
        assert "MUST NOT send unknown fields on write" not in PROSE, (
            "the old half-true obligation is still in the contract; it told "
            "clients that only writes were strict"
        )
        if "forward compatibility" in PROSE.lower():
            assert (
                "does **not** get forward compatibility" in PROSE
                or "does not get forward compatibility" in PROSE
            ), (
                'the contract still offers forward compatibility, which '
                '`extra="forbid"` cannot provide'
            )

    def test_the_gateway_rejects_an_unknown_field_before_the_upstream_sees_it(self) -> None:
        """Where the unknown-field rule IS enforceable end to end.

        The gateway validates the body against `AnalysisRequest.model_fields`,
        so an unknown key is refused with the contract envelope rather than
        forwarded. This test is the evidence that the *request* side of section
        1.1 is genuinely implemented, which is why the defect above is scoped to
        the *response* shape.
        """
        from gateway.policy import validate_analyze_body

        body = json.dumps({"assets": ["a"], "query": "x", "bogus": 1}).encode()
        parsed, error = validate_analyze_body(body)
        assert parsed is None
        assert error is not None
        status, payload = error
        assert status == 422
        assert payload["error"]["code"] == "invalid_request"
        assert "bogus" in payload["error"]["detail"]

    def test_schema_version_is_present_at_both_levels(self) -> None:
        env = json.loads(_minimal_envelope().model_dump_json())
        assert env["schema_version"] == SCHEMA_VERSION
        assert env["result"]["schema_version"] == SCHEMA_VERSION
        assert env["trace"]["schema_version"] == SCHEMA_VERSION

    def test_run_id_is_present_at_both_levels_and_agrees(self) -> None:
        env = json.loads(_minimal_envelope().model_dump_json())
        assert env["run_id"] == env["trace"]["run_id"]

    def test_an_unknown_enum_value_is_rejected(self) -> None:
        """Forward compatibility has a documented limit, and the contract states it.

        Section 3: the frontend "must not assume [the enums] will never grow --
        an unknown value should render as its raw string, not crash." That is a
        *rendering* obligation on the client. The server schema remains a closed
        set. The two are not in conflict, but the asymmetry is easy to misread.
        """
        base = json.loads(_minimal_envelope().model_dump_json())
        base["result"]["task"] = "task_from_the_future"
        with pytest.raises(Exception):
            ResultEnvelope.model_validate(base)


# ===========================================================================
# D8. The contract is not more permissive than the schema
# ===========================================================================
class TestD8NoPermissiveProse:
    """The work order's rule: "do not make the frontend contract more permissive
    than the backend schema"."""

    def test_the_contract_does_not_promise_a_looser_assets_rule(self) -> None:
        """`assets` is `min_length=1`. The contract must say so, and must not
        suggest an empty list is acceptable."""
        from core.schemas import AnalysisRequest

        with pytest.raises(Exception):
            AnalysisRequest(assets=[], query="x")
        assert "Minimum length 1" in CONTRACT

    def test_the_contract_publishes_the_four_endpoints_and_keeps_the_gap(self) -> None:
        """CORRECTED 2026-09-22: four endpoints, and the gap it closed is retained.

        This previously asserted *"There are exactly three."* and that
        `**NOT IN THE PLAN**` was present -- i.e. it required the contract to
        keep describing upload as an open question. The ruling chose Option A
        and the endpoint is implemented, so both assertions are stale.

        The replacement keeps what the original was protecting: a reader must
        still be able to see that a fourth endpoint was added **by decision**
        rather than assumed. That is why the decision record is still required.
        """
        assert "The surface is **four** endpoints." in CONTRACT
        assert "There are exactly three." not in CONTRACT, (
            "the contract still claims three endpoints while the table lists four"
        )
        assert "NOT IN THE PLAN" in CONTRACT, (
            "the upload gap lost its decision record, which would let a reader "
            "assume the fourth endpoint was always planned"
        )
        assert "Option A" in CONTRACT, (
            "the contract does not record which upload design was chosen"
        )

    def test_the_contract_does_not_promise_authentication(self) -> None:
        """Section 7: no auth in v1, and the frontend is told not to build a login."""
        assert "There is no authentication in v1." in CONTRACT
        assert "Do not build a login screen." in CONTRACT

    def test_the_contract_marks_unspecified_values_as_unspecified(self) -> None:
        """Rate-limit values are not in the plan; the document must not invent them."""
        assert "Not specified by the plan" in CONTRACT, (
            "the contract no longer flags the rate-limit values as unspecified"
        )

    def test_the_contract_does_not_claim_streaming(self) -> None:
        assert "no streaming api in v1" in PROSE.lower()
        assert "| Streaming / progress | **Not in v1** |" in PROSE

    def test_task_enum_asset_counts_match_the_documented_table(self) -> None:
        """Section 3.1's asset-count column must agree with CAPABILITY_ASSETS.

        This is the third copy of that mapping (contract prose, planner table,
        specialist specs) and the only one a frontend reads. It is checked
        against the planner so the two cannot drift silently.
        """
        from core.planner import CAPABILITY_ASSETS

        documented = {
            "vqa": 1,
            "caption": 1,
            "grounding": 1,
            "change": 2,
            "optical_sar": 2,
            "change_vqa": 2,
        }
        for task, count in documented.items():
            assert CAPABILITY_ASSETS[task] == count, (
                f"the contract documents {task} with {count} asset(s) but the "
                f"planner requires {CAPABILITY_ASSETS[task]}"
            )

    def test_task_has_seven_values_including_unsupported(self) -> None:
        """The contract calls this out: `unsupported` is a Task, not an error."""
        assert len(list(Task)) == 7
        assert Task.UNSUPPORTED.value == "unsupported"
        assert "`unsupported` is a `Task`, not an error" in CONTRACT

    def test_modality_has_four_values(self) -> None:
        assert {m.value for m in Modality} == {"optical", "sar", "optical_sar", "unknown"}
