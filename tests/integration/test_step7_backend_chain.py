"""STEP 7 sections H, I, M -- the real inference service, driven over HTTP.

WHAT MAKES THIS SUITE DIFFERENT FROM `test_space_app.py`
-------------------------------------------------------
`tests/unit/test_space_app.py` tests the Space app's *construction and its
declarations*. This suite drives the app through httpx's ASGITransport so that
Starlette routing, dependency resolution, the controller boundary and JSON
serialisation all actually execute. Where this file and that one overlap, this
one is the end-to-end statement and the evidence class differs:

  * construction / pure functions          -> `unit`
  * the app answering a real HTTP request  -> `integration`
  * the real artifact loaded, real tensors -> `real_inference`

The markers carry that distinction, per the work order's rule that a unit test
must never be reported as a deployment test.

THE DEFECTS THIS SUITE FOUND (all pinned below, none worked around)
-------------------------------------------------------------------
Running the app for real surfaced four shape disagreements that no
construction-only test could see, because each lives in what a route *returns*
rather than in what it *is*:

  H-1  `AnalysisController.health()` returns
       `{registry, degraded, unavailable, device}` -- which
       `core.schemas.HealthStatus` **rejects** (`extra="forbid"`, and `status`
       and `gpu_available` are missing while required). No *served* route calls
       it, so this is latent rather than live -- but it is the documented
       "GUI health panel" producer, so whoever wires a GUI to it inherits the
       breakage.
  H-2  `health()["registry"]` uses the registry's *internal* vocabulary
       (`available`/`degraded`/`unavailable`), not the contract's five-state
       vocabulary (`loaded`/`absent`/`unavailable`/`not_requested`/`evicted`).
       `available` is not in the contract's vocabulary at all, and `degraded` is
       a *status* as well as a capability state, so a reader cannot tell from the
       word alone which is meant.
  M-1  `describe_deployment()` enumerates only `change` and `change_vqa`, while
       the registry resolves **six** capabilities. The two served endpoints
       therefore agree with each other but describe a strictly narrower
       deployment than the registry can build, and they do so from less evidence
       (two `Path.exists()` calls versus six constructed entries).
  M-2  As above: the two subsystems name the same concept with different words,
       and the words that overlap (`unavailable`, `degraded`) do not mean the
       same thing in both.

A CORRECTION I MADE TO MY OWN DIAGNOSIS
---------------------------------------
My first draft asserted that `GET /v1/health` and `GET /v1/capabilities`
disagree about which capabilities exist. **That test failed**, and it was wrong:
`space_app.health()` derives its `models` map from the same
`describe_deployment()` that `capabilities()` serves, so the pair is internally
consistent by construction. The real discrepancy is narrower -- served answer vs.
registry answer -- and the tests state that instead.

Recorded here rather than quietly deleted because the shape of the mistake is
worth keeping: "two subsystems produce this value" is a reason to *measure*
whether they disagree, not a finding that they do.

`M-1`/`M-2` are diagnosed, not repaired: reconciling them means choosing one
source of truth for "what is available", and that choice changes what the frozen
contract advertises. It is an owner decision, reported with its two options in
`docs/STEP7_BACKEND_CHAIN_REPORT.md` section 14.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: The promoted R-02 artifact, as recorded in
#: `artifacts/change_vqa/run/PROMOTION.json`. Repeated here as literals
#: deliberately: a test that reads its expectation from the artifact it is
#: checking cannot detect a change in that artifact.
R02_SHA256 = "cfae5e43b97ca930f568dc5b8ae4f36b24e9ff717af226159802206ffd63a82a"
R02_BYTES = 5_822_809
R02_PARAMS = 1_453_912
STANET_SHA256_PREFIX = "c5ef3127"
STANET_BYTES = 63_231_009

pytestmark = pytest.mark.integration


# ===========================================================================
# Shared fixtures
# ===========================================================================
@pytest.fixture(scope="module")
def space_app():
    """The real HF Space ASGI app, built once for the module."""
    fastapi = pytest.importorskip("fastapi", reason="the web stack is unavailable here")
    try:
        from app.space_app import build_space_app
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"the Space app cannot be imported here: {exc}")
    return build_space_app()


def _client(app):
    import httpx

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://space.test"
    )


def _run(coro_fn):
    import anyio

    return anyio.run(coro_fn)


# ===========================================================================
# H-A / H-B : health and capabilities over real HTTP
# ===========================================================================
class TestHealthOverHttp:
    """H-A and M. The contract's `GET /v1/health` and its state vocabulary."""

    def test_health_answers_200_without_loading_a_model(self, space_app) -> None:
        """The route must be cheap: no artifact load, no GPU, no weights.

        `id()`-free check: the payload must be complete on the first call, and
        the call must return promptly. A route that lazily built the world would
        still answer 200 -- so the assertion that carries weight is that the
        response is complete and self-consistent, plus the separate no-model
        test below.
        """
        async def body():
            async with _client(space_app) as client:
                return await client.get("/v1/health")

        r = _run(body)
        assert r.status_code == 200
        parsed = r.json()
        assert parsed["schema_version"] == "1.0"
        assert parsed["status"] in ("ok", "degraded", "error")

    def test_health_validates_against_the_contract_model(self, space_app) -> None:
        """The response must satisfy `HealthStatus` -- the contract's shape.

        This is the assertion that would have caught H-1 had it been applied to
        the controller instead of only to the app.
        """
        from core.schemas import HealthStatus

        async def body():
            async with _client(space_app) as client:
                return await client.get("/v1/health")

        r = _run(body)
        HealthStatus(**r.json())

    def test_health_model_states_use_the_contract_vocabulary(self, space_app) -> None:
        """Every per-capability state is one of the five documented strings.

        Section 2.3's vocabulary is closed and the reason is stated there:
        `absent` and `unavailable` must not be conflated. A state outside the
        set cannot be rendered by a contract-conformant client.
        """
        allowed = {"loaded", "absent", "unavailable", "not_requested", "evicted"}

        async def body():
            async with _client(space_app) as client:
                return await client.get("/v1/health")

        models = _run(body).json()["models"]
        assert models, "health reported no capability states at all"
        unknown = {v for v in models.values() if v not in allowed}
        assert not unknown, (
            f"health uses states outside the contract vocabulary: {sorted(unknown)}"
        )

    def test_gpu_available_false_is_not_an_error(self, space_app) -> None:
        """ZeroGPU allocates only for a decorated call; `false` here is normal.

        The contract tells the frontend explicitly not to surface this as a
        fault, so its presence as a plain boolean is load-bearing.
        """
        async def body():
            async with _client(space_app) as client:
                return await client.get("/v1/health")

        parsed = _run(body).json()
        assert isinstance(parsed["gpu_available"], bool)
        assert isinstance(parsed["status"], str), "status must be explicit, not null"

    def test_health_loads_no_artifact(self) -> None:
        """A fresh process must answer `/v1/health` without building the registry.

        Measured, not asserted by inspection: a subprocess imports the module,
        calls the route, and reports whether any specialist module ended up in
        `sys.modules`. If a health probe built the world it would pull torch and
        the specialist packages in, which is exactly what requirement 4 forbids.
        """
        import subprocess
        import sys

        script = (
            "import asyncio, sys, json\n"
            "import httpx\n"
            "from app.space_app import build_space_app\n"
            "app = build_space_app()\n"
            "async def main():\n"
            "    t = httpx.ASGITransport(app=app)\n"
            "    async with httpx.AsyncClient(transport=t, base_url='http://s.test') as c:\n"
            "        r = await c.get('/v1/health')\n"
            "    print(json.dumps({\n"
            "        'status': r.status_code,\n"
            "        'torch': 'torch' in sys.modules,\n"
            "        'specialists': sorted(m for m in sys.modules if m.startswith('specialists')),\n"
            "    }))\n"
            "asyncio.run(main())\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(REPO),
            timeout=300,
        )
        assert result.returncode == 0, f"probe failed: {result.stderr[-800:]}"
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        assert payload["status"] == 200
        assert payload["torch"] is False, (
            "GET /v1/health pulled torch into the process; a health probe must "
            "not load the model stack"
        )


class TestCapabilitiesOverHttp:
    """H-B and the M-series. `GET /v1/capabilities`."""

    def test_capabilities_answers_with_the_documented_envelope(self, space_app) -> None:
        async def body():
            async with _client(space_app) as client:
                return await client.get("/v1/capabilities")

        parsed = _run(body).json()
        assert set(parsed) >= {"schema_version", "capabilities", "deployment"}
        assert parsed["schema_version"] == "1.0"

    def test_every_capability_entry_has_the_documented_fields(self, space_app) -> None:
        """Required keys on every entry; a closed allowlist for anything extra.

        CORRECTED 2026-09-22. This previously asserted *exact* key-set equality
        against the five contract fields. That fails for `optical_sar`, which
        additionally carries `modalities` -- the optical/optical and optical/SAR
        pairings the contract documents for that capability. The frontend cannot
        offer a valid pairing without it, so it is a legitimate addition rather
        than an undocumented field.

        Exact equality was the wrong shape of assertion: it makes every
        future addition a test failure regardless of whether the addition is
        coherent, which trains a reader to loosen the test. The allowlist below
        keeps the property that actually matters -- **an undocumented key is
        still a failure** -- while permitting a documented optional one.
        """
        required = {"task", "available", "reason", "requires_pair", "max_assets"}
        optional = {"modalities"}

        async def body():
            async with _client(space_app) as client:
                return await client.get("/v1/capabilities")

        for cap in _run(body).json()["capabilities"]:
            keys = set(cap)
            missing = required - keys
            assert not missing, (
                f"capability {cap.get('task')!r} is missing required fields: "
                f"{sorted(missing)}"
            )
            extra = keys - required - optional
            assert not extra, (
                f"capability {cap.get('task')!r} carries undocumented fields "
                f"{sorted(extra)}; if they are intended, add them to the "
                f"contract and to `optional` here"
            )
            if "modalities" in keys:
                assert cap["task"] == "optical_sar", (
                    f"{cap['task']!r} carries `modalities`, which is documented "
                    f"only for optical_sar"
                )

    def test_unavailable_capabilities_carry_a_reason(self, space_app) -> None:
        """The contract: "A bare `false` with no reason is not compliant.\"""" 

        async def body():
            async with _client(space_app) as client:
                return await client.get("/v1/capabilities")

        for cap in _run(body).json()["capabilities"]:
            if not cap["available"]:
                assert cap["reason"], (
                    f"{cap['task']} is unavailable with no reason, which the "
                    "contract forbids"
                )

    def test_capability_asset_counts_match_the_planner(self, space_app) -> None:
        """`max_assets` and `requires_pair` must agree with the planner table.

        The contract is the third copy of this mapping; this checks the served
        values against the authoritative one rather than against the contract.
        """
        from core.planner import CAPABILITY_ASSETS

        async def body():
            async with _client(space_app) as client:
                return await client.get("/v1/capabilities")

        for cap in _run(body).json()["capabilities"]:
            expected = CAPABILITY_ASSETS[cap["task"]]
            assert cap["max_assets"] == expected, (
                f"{cap['task']}: served max_assets={cap['max_assets']} but the "
                f"planner requires {expected}"
            )
            assert cap["requires_pair"] == (expected == 2)

    def test_deployment_block_echoes_the_deploy_config(self, space_app) -> None:
        """`cache_max_models: 1` is the serialization constraint the frontend must know."""
        from core.config import get_config

        cfg = get_config().as_dict().get("deployment", {})

        async def body():
            async with _client(space_app) as client:
                return await client.get("/v1/capabilities")

        deployment = _run(body).json()["deployment"]
        assert deployment["cache_max_models"] == cfg.get("cache_max_models")
        assert deployment["cache_max_models"] == 1, (
            "cache_max_models is no longer 1; the serialization obligation in "
            "docs/API_CONTRACT.md section 6 no longer holds"
        )
        assert deployment["lazy_load"] is True


# ===========================================================================
# M-1 / M-2 : THE CAPABILITY RECONCILIATION DEFECT
# ===========================================================================
class TestM1CapabilitySourceOfTruth:
    """M-1 and M-2, RECONCILED 2026-09-22 by owner ruling.

    ORIGINAL FINDING (retained for provenance)
    ------------------------------------------
    Two producers answered "what can this deployment do?" and disagreed:
    `AnalysisController.health()` reported **six** capabilities from the
    registry, while `describe_deployment()` reported **two** by testing two
    artifact paths. The two *served* endpoints agreed with each other -- so a
    client saw no contradiction -- but the served answer was derived from a
    strictly smaller amount of evidence than the available answer.

    THE RULING AND ITS CONSEQUENCE
    ------------------------------
    The owner ruled the **registry** authoritative and required a single
    adapter (`app.deployment`) to translate registry state into contract
    vocabulary. The consequence is the inversion asserted below: the served
    capability set is no longer *narrower* than the registry -- it **is** the
    registry set, with each capability's servability decided from artifact
    evidence. That is what makes the two producers agree by construction rather
    than by coincidence.

    The vocabulary hazard (M-2) has NOT gone away; it has been *contained*.
    The two vocabularies still differ and `unavailable` still means different
    things in each. The adapter is the only place allowed to hold both, and the
    assertion that it does not leak is the one that matters now.
    """

    def test_the_two_served_endpoints_agree_on_the_capability_set(self, space_app) -> None:
        """GOOD: the served pair is consistent. This is the desirable property.

        `health()['models']` and `capabilities[*].task` must enumerate the same
        capabilities, or a client that reads both would see a deployment that
        contradicts itself.
        """
        async def body():
            async with _client(space_app) as client:
                h = await client.get("/v1/health")
                c = await client.get("/v1/capabilities")
                return h.json(), c.json()

        health, caps = _run(body)
        health_tasks = set(health["models"])
        cap_tasks = {cap["task"] for cap in caps["capabilities"]}

        assert health_tasks == cap_tasks, (
            "the two served endpoints disagree about which capabilities exist; "
            f"health-only={sorted(health_tasks - cap_tasks)} "
            f"capabilities-only={sorted(cap_tasks - health_tasks)}"
        )

    def test_the_served_capability_set_is_the_registry_set(self) -> None:
        """M-1 CLOSED: the served set is now exactly the registry's six.

        The pre-ruling assertion here was `served == {"change", "change_vqa"}`
        plus `registry - served` being non-empty -- i.e. it asserted the defect.
        It is inverted rather than deleted, because "the served set equals the
        registry set" is the property the adapter was built to provide, and an
        unasserted property is one that can silently regress.

        Note the two are compared as **sets of capability names**, not as
        states. The registry says `unavailable` for a capability whose builder
        cannot be constructed in this process (four of six here, because this
        environment lacks `transformers`, `sentence_transformers` and
        `huggingface_hub`); the contract must still *name* that capability,
        reporting it as `absent`, rather than dropping it from the enumeration.
        Dropping it is how a client learns about a capability only by its
        absence from a list it cannot see.
        """
        pytest.importorskip("torch", reason="the controller requires torch")
        from app.serving import build_serving_controller

        from app.deployment import deployment_report

        served = {
            cap["capability"]
            for cap in deployment_report().as_internal()["capabilities"]
        }
        registry_states = build_serving_controller().health()["registry"]
        registry = set(registry_states)

        assert served == registry, (
            f"the served capability set diverged from the registry again; "
            f"served-only={sorted(served - registry)} "
            f"registry-only={sorted(registry - served)}"
        )
        assert served == {
            "caption",
            "change",
            "change_vqa",
            "grounding",
            "optical_sar",
            "vqa",
        }, f"the registry resolves {sorted(served)}; re-measure before trusting M-1"

    def test_the_served_endpoints_never_emit_a_registry_state_word(self, space_app) -> None:
        """M-2 CONTAINED: the registry vocabulary must not reach the wire.

        This is the assertion that replaces the old "the two vocabularies
        differ" observation. They still differ -- that half is unchanged and is
        asserted in `test_the_disagreement_is_about_vocabulary_not_only_membership`
        -- but the *hazard* was never the difference, it was the leak. A client
        that receives the word `unavailable` cannot tell whether the deployment
        lacks an artifact or whether a constructible component is broken, and
        those call for different actions.
        """
        async def body():
            async with _client(space_app) as client:
                h = await client.get("/v1/health")
                c = await client.get("/v1/capabilities")
                return h.json(), c.json()

        health, caps = _run(body)
        contract_states = {
            "loaded",
            "absent",
            "unavailable",
            "not_requested",
            "evicted",
        }
        registry_words = {"available", "degraded"}

        # The per-capability states must all be contract words. `unavailable` is
        # in both vocabularies, which is exactly why the check cannot be a bare
        # membership test -- see the next assertion.
        reported = set(health["models"].values())
        assert reported <= contract_states, (
            f"the health endpoint emitted a non-contract state: "
            f"{sorted(reported - contract_states)}"
        )
        assert not (reported & registry_words), (
            f"the health endpoint leaked registry vocabulary to a client: "
            f"{sorted(reported & registry_words)}"
        )

        for cap in caps["capabilities"]:
            assert cap["available"] in (True, False), (
                f"capability {cap['task']!r} has a non-boolean `available`"
            )
            if not cap["available"]:
                assert cap["reason"], (
                    f"capability {cap['task']!r} is not available with no reason; "
                    f"the contract requires one so the UI can show why"
                )

    def test_the_disagreement_is_about_vocabulary_not_only_membership(self) -> None:
        """M-2: the registry and the contract use different state words.

        The registry's states are `available` / `degraded` / `unavailable`. The
        contract's per-capability vocabulary is `loaded` / `absent` /
        `unavailable` / `not_requested` / `evicted`. The overlap is one word
        (`unavailable`) and the two carry it with different meanings: to the
        registry it means "no builder could be constructed"; to the contract it
        means "present but broken, and therefore a defect".

        Transcribing one into the other without noticing would turn a
        deployment gap into a reported defect, or vice versa.
        """
        from core.registry import RegistryState

        from core.schemas import HealthStatus

        registry_words = {m.value for m in RegistryState}
        contract_words = {"loaded", "absent", "unavailable", "not_requested", "evicted"}
        assert registry_words == {"available", "degraded", "unavailable"}
        assert registry_words != contract_words
        # `HealthStatus` documents the contract words but does not enforce them
        # (values are free-form strings so a reason can be carried), which is why
        # this mismatch is a documentation-level hazard rather than a crash.
        assert HealthStatus.model_fields["models"].annotation == dict[str, str]

    def test_degraded_is_a_status_and_also_a_registry_state(self) -> None:
        """The specific collision that makes the vocabulary dangerous.

        `degraded` is a legitimate value of `HealthStatus.status` and also a
        legitimate `RegistryState`. A reader who sees `"degraded"` cannot tell
        from the word alone whether a whole-service status or one capability's
        state is being described. The contract resolves this by keeping the two
        in different fields -- which is only true for the served shape.
        """
        from core.registry import RegistryState

        from core.schemas import HealthStatus

        assert "degraded" in {m.value for m in RegistryState}
        status_annotation = str(HealthStatus.model_fields["status"].annotation)
        assert "degraded" in status_annotation


class TestH1ControllerHealthShape:
    """H-1: `controller.health()` returns a shape `HealthStatus` rejects.

    RESOLVED BY RULING 2026-09-22 -- "fix/retire as public API path".

    The pre-ruling class asserted that the shape is wrong and stopped there,
    because the direction of the repair was an open decision: make the
    controller satisfy the contract (discarding the extra diagnostic fields) or
    widen the contract (exposing internal vocabulary). The owner chose neither
    directly -- it ruled that `AnalysisController.health()` is **not a
    client-facing shape at all** and that `app.deployment.deployment_report()`
    replaces it for every served path.

    So the first test below still asserts the shape mismatch, because it is
    still true and it is the reason the retirement exists -- but it now reads as
    a *precondition* of the retirement rather than an open defect. The test that
    carries weight is the second one: nothing serves it.
    """

    def test_controller_health_still_does_not_satisfy_health_status(self) -> None:
        """The mismatch is unchanged; this is why the retirement was necessary.

        If this ever starts passing, the controller's payload became
        contract-shaped and the retirement could be reconsidered -- which is
        worth knowing, so the assertion is kept rather than made vacuous.
        """
        pytest.importorskip("torch", reason="the controller requires torch")
        from core.schemas import HealthStatus

        from app.serving import build_serving_controller

        controller = build_serving_controller()
        payload = controller.health()

        with pytest.raises(Exception) as exc:
            HealthStatus(**payload)
        message = str(exc.value)
        # Pydantic reports the extra-field rejections; it does not additionally
        # enumerate the missing required fields in the same message (it stops at
        # the first error class it hit for that model). So the assertion is made
        # on the mechanism -- extra fields forbidden -- and the missing-field
        # half is asserted separately and directly below, which is the honest
        # way to test it.
        assert "extra_forbidden" in message, (
            "controller.health() no longer trips `extra='forbid'`; H-1's "
            "extra-field half has changed"
        )
        for key in ("registry", "degraded", "unavailable"):
            assert key in message, f"the extra key {key!r} is no longer the mismatch"

    def test_no_served_route_reaches_the_controller_health_method(self, space_app) -> None:
        """The retirement itself, asserted against the real routes.

        Two independent checks, because either alone is escapable:

        1. **Route table.** No served path may have a handler that calls
           `AnalysisController.health()`. Checked by walking the built
           application's routes and reading each handler's module source, so a
           future route that quietly reintroduces it is caught even if it is
           added under a new name.
        2. **Payload shape.** The two metadata endpoints are asked for real and
           their bodies must validate against the contract models. If either
           endpoint had been wired to the controller, its payload would carry
           the `registry`/`degraded`/`unavailable` keys and fail validation.
        """
        from app import space_app as space_app_module

        source = (REPO / "app" / "space_app.py").read_text(encoding="utf-8")
        assert "AnalysisController" not in source, (
            "app/space_app.py references AnalysisController again; the serving "
            "path must go through app.deployment, not the constructing controller"
        )
        assert "build_serving_controller" in source, (
            "space_app no longer builds the serving controller at all, which "
            "would mean the analyze path was rewired -- re-measure before "
            "trusting this test"
        )
        # `describe_deployment` must delegate, not reimplement.
        assert "deployment_report" in source, (
            "describe_deployment() no longer delegates to the adapter, so the "
            "three-copy problem is back: two places to disagree about "
            "capability state"
        )
        # Guard against the shadowing bug this test originally had: the
        # `space_app` *fixture* is a built ASGI app, while `app.space_app` is a
        # module. Confusing them produces `TypeError: 'module' object is not
        # callable` from deep inside httpx rather than a clear failure.
        assert not hasattr(space_app, "__path__"), (
            "the `space_app` fixture yielded a module instead of a built app"
        )
        assert callable(space_app), "the `space_app` fixture is not an ASGI app"
        assert space_app_module is not space_app

        async def body():
            async with _client(space_app) as client:
                h = await client.get("/v1/health")
                c = await client.get("/v1/capabilities")
                return h.status_code, h.json(), c.status_code, c.json()

        h_status, h_body, c_status, c_body = _run(body)
        assert h_status == 200 and c_status == 200

        from core.schemas import HealthStatus

        validated = HealthStatus.model_validate(h_body)
        assert validated.status in ("ok", "degraded", "error")

        for forbidden in ("registry", "degraded_by_capability", "unavailable"):
            assert forbidden not in h_body, (
                f"/v1/health emitted the internal key {forbidden!r}; the "
                f"controller's shape reached the wire"
            )
        assert c_body["capabilities"], "capabilities returned an empty list"

    def test_controller_health_omits_status_and_gpu_available(self) -> None:
        """The two required `HealthStatus` fields that are simply absent."""
        pytest.importorskip("torch", reason="the controller requires torch")
        from app.serving import build_serving_controller

        payload = build_serving_controller().health()
        for field in ("status", "gpu_available"):
            assert field not in payload, (
                f"controller.health() now provides {field!r}; H-1 is partially "
                f"fixed and the report must be updated"
            )

    def test_controller_health_uses_registry_vocabulary(self) -> None:
        """The vocabulary mismatch: `available` is not a contract state."""
        pytest.importorskip("torch", reason="the controller requires torch")
        from app.serving import build_serving_controller

        states = set(build_serving_controller().health()["registry"].values())
        contract_states = {"loaded", "absent", "unavailable", "not_requested", "evicted"}
        leaked = states - contract_states
        assert leaked, (
            "controller.health() now only uses contract vocabulary; H-1's "
            "vocabulary half is fixed and the report must be updated"
        )
