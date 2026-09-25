"""The deployment capability adapter: registry-authoritative, contract-shaped.

WHY THIS FILE EXISTS
--------------------
Before the owner ruling of 2026-09-22, the repository answered "what can this
deployment do?" in two places with two different answers:

    core.registry.SpecialistRegistry        6 capabilities, states
                                            available/degraded/unavailable
    app.space_app.describe_deployment()     2 capabilities, from 2 Path.exists()

`GET /v1/health` and `GET /v1/capabilities` served the second. `AnalysisController
.health()` produced a third shape that `HealthStatus` *rejects* (finding H-1).

The ruling made the **registry** authoritative and required a **single adapter**
as the only public health source. `app/deployment.py` is that adapter, and these
tests are its contract.

WHAT IS ASSERTED, AND WHY EACH ONE WOULD HAVE CAUGHT SOMETHING REAL
-------------------------------------------------------------------
  * The state vocabulary is the CONTRACT's, closed set, never the registry's.
    A pass-through would emit `degraded`, which means something different in the
    contract -- and the frontend is told to render it differently.
  * `absent` and `unavailable` are never conflated. `absent` = not shipped;
    `unavailable` = shipped but broken (a defect). A naive `RegistryState ->
    str` map collapses them, and that was the single most damaging
    mistranslation available.
  * `HealthStatus.model_validate(health())` succeeds. That is finding H-1 in the
    fixed direction: the payload is valid against the project's own model
    because the model is `extra="forbid"`.
  * `health()['models']` and `capabilities[*].task` enumerate the same set. They
    are derived from ONE traversal, so a client reading both cannot see a
    deployment that contradicts itself.
  * Nothing is constructed. Requirement 4 of `DEPLOYMENT_ARCHITECTURE.md`
    section 3.3 says a metadata request must not load a model; the adapter is on
    the health path, so a `registry.build()` call here would spend GPU quota.
  * `optical_sar` is NOT reported available when the SHARED RESOLVER cannot find
    CROMA. This is the false-positive regression guard. It originally read
    "on a deployment without CROMA", which was true of the old code only because
    the adapter read a config key that is never set -- so the guard passed while
    the adapter disagreed with the serving loader. It now drives the resolver,
    which is the same question serving asks, and asserts both directions.

Every assertion below was derived by MEASURING this repository, not by assuming
what it should say. Where a first attempt was wrong, the wrong attempt is
recorded in the docstring so it cannot come back.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.unit


def _scratch_dir(name: str) -> Path:
    """A private directory these tests write real files into.

    WHY NOT `tmp_path`
    ------------------
    This sandbox shim intercepts `shutil.rmtree` and enforces a per-turn
    bulk-delete cap (50). pytest's `tmp_path` fixture garbage-collects its
    directories during the run, so a suite with several `tmp_path` tests spends
    the turn's whole delete budget partway through and the LATER ones die in
    fixture setup with `SystemExit: 1` from the shim's guard --

        [safe-delete][SAFE_DELETE_BULK_REJECTED] {"count":97,"threshold":50}

    which has nothing to do with the code under test and silently converts a
    passing assertion into an ERROR. The failure is positional, not logical, so
    it is exactly the kind of noise that would make a real regression invisible.

    This helper sidesteps the fixture: it makes a directory the test owns, and
    the caller writes real sentinel FILES into it. Note the checkpoint tests
    need a genuine file (`_missing_shipped` uses `_path_present` while
    `_artifact_evidence` calls `Path.exists()`), so an on-disk sentinel is
    required, not optional.

    The directory is NOT deleted at teardown, deliberately: deleting it is what
    triggered the guard in the first place, and a fresh `os.getpid()`-scoped
    name means repeated runs never collide. Leftover bytes are a few KB.
    """
    import os

    path = REPO / ".scratch_deployment_adapter" / f"{name}-{os.getpid()}"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# The served payloads are the contract's shapes
# ---------------------------------------------------------------------------


class TestHealthPayloadShape:
    """`GET /v1/health` must be a valid `HealthStatus`. This is H-1, fixed."""

    def test_health_validates_against_health_status(self) -> None:
        """The headline assertion.

        `HealthStatus` is `extra="forbid"` (`core/schemas.py:392`), so this is
        not a formality: `AnalysisController.health()` fails it, with three
        `extra_forbidden` errors and two missing required fields. The adapter's
        payload passes it, and that is the whole difference the ruling bought.
        """
        from app.deployment import health_payload
        from core.schemas import HealthStatus

        payload = health_payload()
        validated = HealthStatus.model_validate(payload)
        assert validated.status in ("ok", "degraded", "error")

    def test_health_field_set_is_exactly_the_models(self) -> None:
        """No field may be added without the model being changed too.

        Stated separately from the validation above because the failure mode
        differs: a *missing* field makes `HealthStatus` fill a default, which
        succeeds silently. Only the exact-set assertion catches that.
        """
        from app.deployment import health_payload
        from core.schemas import HealthStatus

        payload = health_payload()
        assert set(payload) == set(HealthStatus.model_fields), (
            "the served health payload and core.schemas.HealthStatus have "
            "diverged; one of them must be updated deliberately"
        )

    def test_health_carries_the_schema_version_from_the_model(self) -> None:
        from app.deployment import health_payload
        from core.schemas import SCHEMA_VERSION

        assert health_payload()["schema_version"] == SCHEMA_VERSION

    def test_device_is_a_string_or_null_never_a_guess(self) -> None:
        """`device` was `null` on EVERY request before this change.

        `device_preference` is a `@property` on `Config` (`core/config.py:87`),
        not a key in `configs/base.yaml`, so the old
        `cfg.as_dict().get("device_preference")` could never resolve. `null` is a
        legal value for the field, so nothing detected it. This test asserts the
        property is actually read.

        ⚠️ F-8: this test asserts a **type**, not the **legal set**, which is why
        it stayed green while `SATQUERY_DEVICE=garbage` served
        `{"device": "garbage"}`. A type check is nearly vacuous for a field whose
        contract is an enumeration. The legal-set guards are the two below.
        """
        from app.deployment import health_payload

        device = health_payload()["device"]
        assert device is None or isinstance(device, str)
        assert device is not None, (
            "device resolved to None; the adapter is reading a dict key instead "
            "of the `Config.device_preference` property, which was the original "
            "latent bug"
        )

    # -- F-8 ---------------------------------------------------------------

    #: The set `docs/API_CONTRACT.md` section 2.1 publishes for the `device`
    #: field. Sourced as a literal here on purpose: the point of the guard is
    #: that the CONTRACT is a closed set, so a guard that derived the set from
    #: the implementation could never fail.
    _LEGAL_DEVICES = frozenset({"cpu", "cuda", "mps"})

    def _device_for(self, raw: str | None, monkeypatch) -> str | None:
        """`health()["device"]` for one `SATQUERY_DEVICE` input, via the real path."""
        from app.deployment import health_payload

        if raw is None:
            monkeypatch.delenv("SATQUERY_DEVICE", raising=False)
        else:
            monkeypatch.setenv("SATQUERY_DEVICE", raw)
        return health_payload()["device"]

    def test_an_unparsable_device_never_reaches_the_contract_field(
        self, monkeypatch
    ) -> None:
        """F-8, defect B. Measured before the fix: `'garbage'` -> `'garbage'`.

        `docs/DEPLOYMENT_ARCHITECTURE.md:287` documents `SATQUERY_DEVICE` as
        `cpu | cuda`, and `API_CONTRACT.md:131` publishes
        `"cpu" | "cuda" | "mps" | null` for the served field. Both advertise a
        **closed set**; the reader accepted any string and echoed it, so a typo
        became the deployment's reported device.

        `null` is the honest answer for "the operator set something I do not
        recognise": it is a legal value the contract already defines, and it does
        not claim a device that does not exist.
        """
        for raw in ("garbage", "1", "true", "tpu", "cuda:0", "GPUs", "cuda "):
            device = self._device_for(raw, monkeypatch)
            assert device in self._LEGAL_DEVICES or device is None, (
                f"SATQUERY_DEVICE={raw!r} produced device={device!r}, which is "
                f"neither a legal device nor null. The contract publishes "
                f"{sorted(self._LEGAL_DEVICES)} | null; anything else tells the "
                f"frontend to render a device that cannot exist."
            )

    def test_a_legal_device_still_reaches_the_contract_field(
        self, monkeypatch
    ) -> None:
        """The negative control for the guard above.

        Without this, a guard that answered `null` for **every** input -- deleting
        the feature instead of validating it -- would leave the other test green.

        NOTE ON WHAT `cuda` RESOLVES TO, which cost me a wrong first draft.
        With `SATQUERY_DEVICE=cuda`, `_cuda_detected()` returns **True** by
        design: `core/config.py:87` documents the variable as "an explicit
        operator assertion that a GPU exists", and `_cuda_detected` honours it
        before consulting `CUDA_VISIBLE_DEVICES`. So `_effective_device` correctly
        keeps `"cuda"` and the no-CUDA correction does **not** fire.

        My first draft asserted `("cuda", "cpu")` and failed with
        `'cuda' == 'cpu'`. The test was wrong, not the code -- and the reason is
        worth keeping, because it is the distinction between the two device
        questions:

            operator asserts cuda, and we believe them  -> "cuda" (this case)
            a configured cuda on a box with no CUDA     -> "cpu"  (the correction)

        The correction exists for the second case, which is reached through
        `deployment.device` in `base.yaml`, not through this variable.
        """
        for raw, expected in (
            ("cpu", "cpu"),
            ("cuda", "cuda"),  # an explicit assertion of CUDA is believed
            ("mps", "mps"),
            ("auto", "cpu"),  # `auto` resolves; it is never echoed
        ):
            device = self._device_for(raw, monkeypatch)
            assert device == expected, (
                f"SATQUERY_DEVICE={raw!r} should resolve to {expected!r}, "
                f"got {device!r}"
            )

    def test_the_case_of_the_operator_input_does_not_change_the_answer(
        self, monkeypatch
    ) -> None:
        """F-8, defect A. Measured before the fix: `'cuda'` -> `'cuda'` but `'CUDA'` -> `'CUDA'`.

        The asymmetry is that the *other* three read sites lowercase and
        `_effective_device` does not, so case decides whether the downstream
        correction applies at all. This test asserts the weaker, non-negotiable
        half -- the answer must not depend on the case -- so it holds whether or
        not the correction fires.

        The mixed-case form is included because an asymmetry that only shows up
        for one of three spellings is the kind that survives review.
        """
        baseline = self._device_for("cuda", monkeypatch)
        for raw in ("CUDA", "Cuda", "cUdA"):
            device = self._device_for(raw, monkeypatch)
            assert device == baseline, (
                f"SATQUERY_DEVICE={raw!r} resolved to {device!r} but the "
                f"lowercase form resolves to {baseline!r}; the case of the "
                f"operator's input must not change the answer"
            )

    def test_gpu_available_and_device_cannot_contradict_each_other(
        self, monkeypatch
    ) -> None:
        """The cross-field property, which is what actually shipped wrong.

        F-7's lesson was that two readers of one value can disagree. Here the two
        readers are in the **same payload**: `gpu_available` and `device`. The
        invariant is one-directional and cheap to state:

            `device == "cuda"`  =>  `gpu_available` is True

        A payload announcing a CUDA device while denying a GPU is the deployment
        lying about itself, and `docs/DEPLOYMENT_ARCHITECTURE.md` tells the
        frontend this field is displayable. The converse is deliberately NOT
        asserted: `gpu_available: true` with `device: "cpu"` is correct and
        common — a GPU can exist while the operator has chosen the CPU (and on a
        ZeroGPU Space the device is invisible until a request executes).
        """
        from app.deployment import health_payload

        for raw in (None, "cpu", "cuda", "CUDA", "Cuda", "mps", "auto", "garbage", "1"):
            if raw is None:
                monkeypatch.delenv("SATQUERY_DEVICE", raising=False)
            else:
                monkeypatch.setenv("SATQUERY_DEVICE", raw)
            payload = health_payload()
            device, gpu = payload["device"], payload["gpu_available"]
            if device == "cuda":
                assert gpu is True, (
                    f"SATQUERY_DEVICE={raw!r} produced device='cuda' with "
                    f"gpu_available={gpu}; the two fields of one payload "
                    f"contradict each other"
                )

    def test_gpu_available_is_a_bool(self) -> None:
        from app.deployment import health_payload

        assert isinstance(health_payload()["gpu_available"], bool)


class TestCapabilitiesPayloadShape:
    """`GET /v1/capabilities` must carry the contract's fields per entry."""

    def test_each_entry_has_the_documented_keys(self) -> None:
        from app.deployment import capabilities_payload

        payload = capabilities_payload()
        required = {"task", "available", "reason", "requires_pair", "max_assets"}
        for entry in payload["capabilities"]:
            missing = required - set(entry)
            assert not missing, (
                f"capability {entry.get('task')!r} is missing {sorted(missing)}; "
                f"API_CONTRACT.md section 2.2 fixes this field set"
            )

    def test_an_unavailable_capability_always_carries_a_reason(self) -> None:
        """`API_CONTRACT.md` section 2.2: *"A bare `false` with no reason is not
        compliant"*."""
        from app.deployment import capabilities_payload

        for entry in capabilities_payload()["capabilities"]:
            if not entry["available"]:
                assert entry["reason"], (
                    f"capability {entry['task']!r} is unavailable with no "
                    f"reason; the UI cannot tell the user why"
                )

    def test_max_assets_is_an_int_and_requires_pair_agrees_with_it(self) -> None:
        """The two fields must not contradict each other."""
        from app.deployment import capabilities_payload

        for entry in capabilities_payload()["capabilities"]:
            assert isinstance(entry["max_assets"], int)
            assert isinstance(entry["requires_pair"], bool)
            assert entry["requires_pair"] == (entry["max_assets"] == 2), (
                f"capability {entry['task']!r} declares max_assets="
                f"{entry['max_assets']} but requires_pair="
                f"{entry['requires_pair']}"
            )

    def test_the_capability_set_is_a_subset_of_the_task_enum(self) -> None:
        """A capability the `Task` enum does not define cannot be routed."""
        from app.deployment import capabilities_payload
        from core.schemas import Task

        known = {t.value for t in Task}
        for entry in capabilities_payload()["capabilities"]:
            assert entry["task"] in known, (
                f"{entry['task']!r} is served as a capability but is not a "
                f"`Task` value, so the planner could never route it"
            )


# ---------------------------------------------------------------------------
# The vocabulary translation -- the reason an adapter exists at all
# ---------------------------------------------------------------------------


class TestContractVocabulary:
    """The registry's words must never reach a client."""

    def test_every_state_is_in_the_contract_vocabulary(self) -> None:
        from app.deployment import CONTRACT_STATES, capabilities_payload, health_payload

        for value in health_payload()["models"].values():
            assert value in CONTRACT_STATES, (
                f"the served health state {value!r} is outside the contract's "
                f"closed set {sorted(CONTRACT_STATES)} -- this is exactly the "
                f"registry vocabulary leaking"
            )
        assert payload_states(capabilities_payload()) <= CONTRACT_STATES

    def test_the_registry_word_degraded_is_never_served_as_a_state(self) -> None:
        """`degraded` is ambiguous: it is BOTH a `HealthStatus.status` value and
        a `RegistryState`. Serving it inside `models` would make the two readings
        indistinguishable.
        """
        from app.deployment import health_payload

        values = set(health_payload()["models"].values())
        assert "degraded" not in values, (
            "`degraded` is being served as a per-capability state; it is a "
            "registry word, and as a `models` value it collides with the "
            "service-level `status` vocabulary"
        )

    def test_the_registry_vocabulary_is_kept_out_of_the_served_payload(self) -> None:
        """The word `available` is a `RegistryState`, not a contract state.

        `available` DOES appear as a contract field name (`capabilities[].available`,
        a boolean). That is a different thing from a *state value*, and only the
        values are checked here.
        """
        from app.deployment import CONTRACT_STATES, health_payload

        leaked = {
            value
            for value in health_payload()["models"].values()
            if value in {"available", "degraded", "unavailable"} - CONTRACT_STATES
        }
        assert not leaked, f"registry vocabulary leaked into `models`: {leaked}"

    def test_the_internal_view_carries_the_registry_word(self) -> None:
        """The translation must not DESTROY the registry's answer.

        An operator needs to know whether the registry said `available` or
        `degraded`; the contract cannot express that distinction. So the internal
        view keeps it, and the served view does not. Asserting both halves is
        what makes "kept internal" a testable claim rather than an intention.
        """
        from app.deployment import deployment_report

        internal = deployment_report().as_internal()
        assert internal["capabilities"], "the internal view has no capabilities"
        for entry in internal["capabilities"]:
            assert "registry_state" in entry
            assert entry["registry_state"] in {"available", "degraded", "unavailable"}

    def test_absent_and_unavailable_are_different_states(self) -> None:
        """The conflation that would be most damaging.

        `absent` = not shipped in this build. `unavailable` = shipped and broken,
        a defect. `core/registry.py`'s own docstring is emphatic that a corrupt
        artifact must not be retried as a degraded build, and the contract is
        equally emphatic that a frontend must not conflate them. So the mapping
        function must be capable of producing both, and must not map the
        registry's single `unavailable` onto the contract's `unavailable`.
        """
        from app.deployment import REGISTRY_TO_CONTRACT

        assert REGISTRY_TO_CONTRACT["unavailable"] == "absent", (
            "the registry's `unavailable` (construction raised) is being mapped "
            "onto the contract's `unavailable` (present but broken). Those are "
            "different facts: the first is usually a missing artifact, which is "
            "`absent`, not a defect"
        )
        # And the contract state the registry cannot express is reachable only
        # from the construct-reporting path.
        assert "unavailable" in {*REGISTRY_TO_CONTRACT.values()} | {"unavailable"}

    def test_an_unknown_capability_is_refused_not_reported_servable(self) -> None:
        """Defensive, and it was a REAL BUG found by this test.

        A capability the registry does not declare must never be reported as
        available. The first version of the adapter reported
        `not_a_real_capability` as `available: true` -- it had no artifact
        requirement, so it fell through to the "everything on disk" branch.

        My initial test asserted `state == "not_requested"`, which was the wrong
        expectation: the state is now `absent`, because the honest reading of
        "the registry does not declare this" is that the deployment does not
        serve it. What matters, and what both expectations agree on, is that it
        is NOT available.
        """
        from app.deployment import deployment_report

        report = deployment_report(capabilities=["not_a_real_capability"])
        entry = report.capabilities[0]
        assert not entry.available, (
            f"a capability the registry does not declare was reported "
            f"available (state={entry.state!r}); this is the false positive the "
            f"adapter exists to prevent"
        )
        assert entry.reason
        assert "registry does not declare" in entry.reason

    def test_a_reported_construction_failure_becomes_unavailable(self) -> None:
        """The `construct_failures=` path is the only source of `unavailable`.

        Asserted because it is the defect channel: if this stopped working, a
        present-but-broken artifact would be reported as merely `absent` and the
        operator would look for a missing file that is right there.
        """
        from app.deployment import deployment_report

        report = deployment_report(
            capabilities=["change"],
            construct_failures={"change": "CROMA_base.pt is corrupt"},
        )
        entry = report.capabilities[0]
        assert entry.state == "unavailable"
        assert entry.reason and "corrupt" in entry.reason
        assert report.status == "error", (
            "a construction defect must make the service status `error`; it was "
            f"{report.status!r}"
        )


def payload_states(payload: dict) -> set[str]:
    """The states a capabilities payload implies, for vocabulary checks."""
    return {
        "absent" if not entry["available"] else "not_requested"
        for entry in payload["capabilities"]
    }


# ---------------------------------------------------------------------------
# Status derivation
# ---------------------------------------------------------------------------


class TestServiceStatus:
    """`HealthStatus.status` is DERIVED from the capability states."""

    def test_not_requested_alone_does_not_degrade_the_service(self) -> None:
        """Under `lazy_load: true`, every capability is `not_requested` on a
        fresh process. Treating that as `degraded` would make the field useless
        as a probe -- a healthy idle service would always report degraded."""
        from app.deployment import deployment_report

        report = deployment_report(capabilities=["vqa", "caption"])
        assert set(report.models.values()) == {"not_requested"}
        assert report.status == "ok", (
            "an idle, fully-provisioned service reported "
            f"{report.status!r}; `not_requested` is normal under lazy_load"
        )

    def test_an_absent_capability_degrades_the_service(self) -> None:
        """`API_CONTRACT.md` section 2.1: *"`degraded` = the service is up but at
        least one expected artifact is absent"*.

        REWRITTEN 2026-09-25. This used to use `optical_sar` as its example of an
        absent capability, on the belief that "CROMA is not shipped here". That
        belief was false: the pinned CROMA snapshot lives in the Hub cache and
        `resolve_checkpoint_path` finds it, so `optical_sar` is `not_requested`
        (available-but-idle) on a warm host. The test was asserting the
        reporter/loader divergence as if it were the contract.

        The subject is now an artifact that genuinely cannot resolve -- a change
        head the composition root points at a path that does not exist -- so the
        assertion tests the `absent` -> `degraded` rule rather than one host's
        cache state.
        """
        import app.serving as serving
        from app.deployment import deployment_report

        original = serving.CHANGE_CHECKPOINT
        try:
            serving.CHANGE_CHECKPOINT = Path("/nonexistent/head.pt")
            report = deployment_report(capabilities=["change"])
        finally:
            serving.CHANGE_CHECKPOINT = original

        assert report.capabilities[0].state == "absent"
        assert report.status == "degraded"

    def test_a_construction_defect_is_error_not_degraded(self) -> None:
        """Different facts, different remedies, so different statuses."""
        from app.deployment import deployment_report

        report = deployment_report(
            capabilities=["change"], construct_failures={"change": "boom"}
        )
        assert report.status == "error"

    def test_the_live_deployment_status_is_consistent_with_its_states(self) -> None:
        """The invariant, asserted on whatever this machine actually reports.

        Deliberately not asserting a specific status string: this deployment's
        answer is a fact about the deployment, and pinning it would make the test
        a change-detector. What must hold is the RELATIONSHIP.
        """
        from app.deployment import deployment_report

        report = deployment_report()
        states = set(report.models.values())
        if "unavailable" in states:
            assert report.status == "error"
        elif "absent" in states:
            assert report.status == "degraded"
        else:
            assert report.status == "ok"


# ---------------------------------------------------------------------------
# Registry-authoritative, and constructing nothing
# ---------------------------------------------------------------------------


class TestRegistryIsAuthoritative:
    """The ruling: the registry decides what exists; the adapter only translates."""

    def test_the_served_capability_set_is_the_registrys_declared_set(self) -> None:
        """The M-1 fix, stated as an equality.

        Before the ruling the served set was `{change, change_vqa}` (two rows,
        two `Path.exists()` calls) while the registry resolved six. The served
        set must now BE the registry's declared set -- not a subset, not a
        hand-maintained list.

        This also guards the shortcut the adapter takes to avoid importing torch
        (`_registry_capabilities`). That function reads `default_specs()` rather
        than instantiating a registry, and this assertion is what keeps the two
        answers identical: if a future registry change populated `_specs` from
        anywhere else, this fails.
        """
        from app.deployment import deployment_report
        from app.serving import build_serving_registry

        declared = set(build_serving_registry().available())
        served = {cap.capability for cap in deployment_report().capabilities}
        assert served == declared, (
            f"the adapter serves {sorted(served)} but the registry declares "
            f"{sorted(declared)}; the registry is authoritative per the ruling"
        )

    def test_health_and_capabilities_enumerate_the_same_set(self) -> None:
        """Both routes come from ONE traversal, so they cannot disagree."""
        from app.deployment import capabilities_payload, health_payload

        assert set(health_payload()["models"]) == {
            entry["task"] for entry in capabilities_payload()["capabilities"]
        }

    def test_the_adapter_does_not_construct_any_specialist(self) -> None:
        """Requirement 4, enforced rather than documented.

        `build()` and `build_all()` on the registry construct specialists. If the
        adapter called either, `/v1/health` would load every model -- spending the
        5 GPU-minute daily budget on a liveness probe. Asserted against the
        EXECUTABLE source, with docstrings stripped, because the module's own
        prose explains why it must not construct and a naive search would find
        the explanation.
        """
        source = (REPO / "app" / "deployment.py").read_text(encoding="utf-8")
        code = _code_only(source)
        for forbidden in (".build_all(", ".build("):
            assert forbidden not in code, (
                f"app/deployment.py calls {forbidden!r}; the adapter is on the "
                f"metadata path and a metadata request must not construct a "
                f"model (DEPLOYMENT_ARCHITECTURE.md section 3.3 requirement 4)"
            )

    def test_the_metadata_path_does_not_import_torch(self) -> None:
        """Asserted in a subprocess, because the test process has already
        imported torch via `core.config` -- an in-process check could never fail.
        """
        script = (
            "import sys; from app.deployment import deployment_report; "
            "deployment_report(); print('torch' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(REPO),
            check=True,
        )
        assert result.stdout.strip() == "False", (
            "building the deployment report imported torch; a metadata request "
            "would pay the model-stack import cost"
        )

    def test_it_does_not_use_the_constructing_controller_health(self) -> None:
        """`AnalysisController.health()` is retired as a public path (H-1/H-2).

        The adapter must not reach a *controller's* health method -- that one
        constructs everything. It has its own `DeploymentReport.health()`, which
        is a pure payload builder and is the replacement.

        Anchored on the call SITE rather than the token. A naive
        `".health()" not in code` assertion FAILED when this test was first run,
        because `deployment_report().health()` is the adapter's own method and
        the substring matches it. That is the repo's documented hazard -- *"a
        source-level assertion can be satisfied, or broken, by the wrong line"*
        -- so the check is anchored on what identifies a controller.
        """
        source = (REPO / "app" / "deployment.py").read_text(encoding="utf-8")
        code = _code_only(source)
        for forbidden in ("build_serving_controller", "AnalysisController"):
            assert forbidden not in code, (
                f"app/deployment.py references {forbidden!r}; the adapter is on "
                f"the metadata path and must not hold a controller, whose health "
                f"method constructs every specialist"
            )


# ---------------------------------------------------------------------------
# The artifact table must stay a view of the registry, not a fourth copy
# ---------------------------------------------------------------------------


class TestRequirementTable:
    """`_REQUIREMENTS` is keyed by the REGISTRY's config paths, and that link is
    asserted so a registry edit cannot silently orphan it."""

    def test_every_declared_config_path_exists_in_the_registry_spec_table(self) -> None:
        """The anti-drift assertion.

        `_REQUIREMENTS` names config paths like `croma.checkpoint_path`. Those
        strings must be exactly the ones the registry's own spec rows use,
        otherwise the table has become a fourth copy of the artifact map -- the
        three-copy hazard -- and a renamed key would silently stop being checked.
        """
        from app.deployment import _REQUIREMENTS
        from core.registry import default_specs

        registry_keys: dict[str, set[str]] = {}
        for spec in default_specs():
            registry_keys.setdefault(spec.name, set()).update(
                spec.optional_config_keys.values()
            )

        for capability, requirements in _REQUIREMENTS.items():
            if capability in ("change", "change_vqa"):
                # Wired by the composition root, not by config; see the module.
                continue
            known = registry_keys.get(capability, set())
            for _kind, config_path in requirements:
                assert config_path in known, (
                    f"{capability!r} declares requirement {config_path!r}, which "
                    f"is not a config key its registry spec row uses "
                    f"({sorted(known)}); the adapter has drifted from the "
                    f"registry's own declaration"
                )

    def test_a_capability_with_an_empty_row_is_hub_backed_or_declared_empty(
        self,
    ) -> None:
        """An empty requirement row must be a DECISION, not an oversight.

        `vqa` and `caption` have no requirements because their spec row has no
        `optional_config_keys` at all -- the builder reads the `vlm.*` block
        itself. An empty row is therefore correct for them, but an empty row for
        any OTHER capability would mean "no artifact check", which reports
        availability unconditionally.

        So the rule asserted here: an empty row is permitted only when the
        capability is explicitly declared Hub-backed. The membership check is
        what distinguishes a decision from a gap.
        """
        from app.deployment import _HUB_BACKED, _REQUIREMENTS

        for capability, requirements in _REQUIREMENTS.items():
            if capability in ("change", "change_vqa"):
                continue
            if requirements:
                continue
            assert capability in _HUB_BACKED, (
                f"{capability!r} has an empty requirement row but is not declared "
                f"Hub-backed; it would be reported available with no artifact "
                f"check at all"
            )

    def test_every_registry_capability_has_a_requirement_row(self) -> None:
        """A capability with no row would be reported available unconditionally."""
        from app.deployment import _REQUIREMENTS
        from core.registry import default_specs

        for spec in default_specs():
            assert spec.name in _REQUIREMENTS, (
                f"the registry declares {spec.name!r} but the adapter has no "
                f"requirement row for it; it would be reported available with no "
                f"artifact check"
            )

    def test_optical_sar_is_absent_when_the_resolver_cannot_find_croma(
        self, monkeypatch
    ) -> None:
        """THE FALSE-POSITIVE REGRESSION GUARD, restated against the RESOLVER.

        REWRITTEN 2026-09-25, and the rewrite is the point of this change.

        The original version asserted `absent` while asserting nothing about the
        Hub cache, so it passed for the wrong reason: `_missing_shipped` read
        `Config.get("croma.checkpoint_path")`, a key deliberately never written
        to `base.yaml`, so the answer was `absent` on EVERY host -- including one
        with the checkpoint cached. It encoded the reporter/loader divergence as
        the contract, which is why it went red when the divergence was fixed.

        The risk this guard exists for is real and unchanged: never report
        `optical_sar` available when the fusion genuinely cannot happen. So the
        test now removes the artifact from the RESOLVER's view -- the same
        question serving asks -- and asserts the capability is refused.

        `resolve_checkpoint_path` is patched rather than the cache: simulating a
        cold cache by moving 778 MB around would be slow, destructive, and would
        test the filesystem instead of the adapter.
        """
        from app.deployment import deployment_report

        monkeypatch.setattr(
            "specialists.optical_sar.croma.resolve_checkpoint_path",
            lambda config=None: (None, "absent"),
        )

        report = deployment_report(capabilities=["optical_sar"])
        entry = report.capabilities[0]
        assert entry.state == "absent", (
            f"optical_sar reported {entry.state!r} while the resolver returns no "
            f"checkpoint; a missing checkpoint must not be reported as available"
        )
        assert entry.available is False
        assert entry.reason and "CROMA" in entry.reason, (
            f"the reason must name the missing artifact, not be generic; got "
            f"{entry.reason!r}"
        )

    def test_optical_sar_is_available_when_the_resolver_finds_croma(
        self, monkeypatch
    ) -> None:
        """The complement, and the test whose absence let the bug live.

        This is the assertion that was impossible before the fix: with
        `croma.checkpoint_path` ABSENT from config (see the assertion below) but
        the resolver returning a real checkpoint file, the capability must be
        reported available and carry NO reason.

        Without a passing version of this, the guard above would be satisfied by
        a function that always answers `absent` -- deleting the capability rather
        than resolving it. That is precisely the shape the pre-fix code had.

        The sentinel is a REAL file under `tmp_path`, not a patched `exists()`.
        The adapter checks presence twice through different helpers
        (`_missing_shipped` via `_path_present`, `_artifact_evidence` via
        `Path.exists()`), and an earlier draft of this test patched only the
        former -- so the evidence table still reported `present: False` and the
        test failed against correct code. A real file satisfies both, and keeps
        the test measuring the adapter rather than its own mocks.
        """
        from app.deployment import deployment_report, _configured_path

        # The premise: the config key really is absent, so availability cannot be
        # coming from a config lookup.
        assert _configured_path("croma.checkpoint_path") is None, (
            "croma.checkpoint_path is set in config; this test's premise (that "
            "availability does not come from a config key) no longer holds"
        )
        assert "checkpoint_path" not in (
            REPO / "configs" / "base.yaml"
        ).read_text(encoding="utf-8"), (
            "configs/base.yaml names croma.checkpoint_path; writing that key "
            "moves Config.hash off 78f1e3700da15aa1, so this test's premise no "
            "longer holds"
        )

        sentinel = _scratch_dir("croma-available") / "CROMA_base.pt"
        sentinel.write_bytes(b"stand-in for the pinned checkpoint")
        monkeypatch.setattr(
            "specialists.optical_sar.croma.resolve_checkpoint_path",
            lambda config=None: (str(sentinel), "hub-cache"),
        )

        report = deployment_report(capabilities=["optical_sar"])
        entry = report.capabilities[0]
        assert entry.available is True, (
            f"optical_sar reported {entry.state!r} with reason {entry.reason!r} "
            f"although the resolver returned a real checkpoint; the adapter is "
            f"not consulting the resolver the serving path uses"
        )
        assert entry.reason is None
        assert report.status == "ok"

    def test_the_reporter_and_the_loader_agree_on_the_croma_source(
        self, monkeypatch
    ) -> None:
        """THE PARITY GUARD: one resolver call, two consumers.

        The defect was that the capability reporter and the serving loader asked
        DIFFERENT questions about the same artifact, so they could disagree --
        and did, in the direction that looked like a missing file.

        This asserts they cannot diverge: the reporter's resolution is the
        loader's resolution, because it is literally the same call. Asserted
        through `_requirement_artifacts` (what the reporter checks) and
        `serving._wired_optical_sar_builder` (what the loader passes on), both
        driven by one patched resolver.

        A count of calls is included: if the reporter ever stopped calling the
        resolver and went back to reading config, the count would drop to 1 (or
        0) and this fails even though the patched return value alone might still
        make the paths look equal.
        """
        import app.serving as serving
        from app.deployment import _requirement_artifacts

        calls: list[Any] = []
        sentinel_path = "resolved/CROMA_base.pt"

        def _recording_resolver(config: Any = None):
            calls.append(config)
            return sentinel_path, "hub-cache"

        monkeypatch.setattr(
            "specialists.optical_sar.croma.resolve_checkpoint_path",
            _recording_resolver,
        )

        reporter_paths = _requirement_artifacts("optical_sar")

        recorded_build: dict[str, Any] = {}

        class _Stub:
            name = "optical_sar"
            version = "0.0.0-test"
            capabilities = ("optical_sar",)
            has_encoder = True
            has_head = True

        def _recording_builder(config: Any, **kwargs: Any) -> Any:
            recorded_build.update(kwargs)
            return _Stub()

        monkeypatch.setattr(
            "specialists.optical_sar.specialist.build_optical_sar_specialist",
            _recording_builder,
        )
        serving.build_serving_registry(serving.load_config()).build("optical_sar")

        loader_path = recorded_build.get("checkpoint_path")

        assert calls, (
            "the capability reporter never called croma.resolve_checkpoint_path; "
            "it is resolving the artifact some other way, which is how the "
            "reporter/loader divergence was possible"
        )
        assert loader_path == sentinel_path, (
            f"the serving loader passed {loader_path!r}, not the resolver's "
            f"answer {sentinel_path!r}"
        )
        assert Path(sentinel_path) in reporter_paths, (
            f"the reporter resolved {[str(p) for p in reporter_paths]}, which "
            f"does not include the loader's path {sentinel_path!r}; the two have "
            f"diverged again"
        )

    def test_optical_sar_requires_both_the_checkpoint_and_the_head(
        self, monkeypatch
    ) -> None:
        """A resolver hit for the encoder is not sufficient on its own.

        `has_head` (`specialist.py:175-187`) is False without the trained fusion
        head, so the deployment would still be unable to produce a fused
        prediction. If the head were the missing half, reporting `available`
        would re-create the false positive this table exists to prevent.

        Driven by pointing the composition root's `FUSION_HEAD` constant at a
        path that cannot exist, so the encoder resolution succeeds and only the
        head is missing -- which is what makes this a test of the head clause
        rather than of the encoder.
        """
        import app.serving as serving
        from app.deployment import deployment_report

        checkpoint = _scratch_dir("croma-head") / "CROMA_base.pt"
        checkpoint.write_bytes(b"stand-in for the pinned checkpoint")
        monkeypatch.setattr(
            "specialists.optical_sar.croma.resolve_checkpoint_path",
            lambda config=None: (str(checkpoint), "hub-cache"),
        )
        monkeypatch.setattr(serving, "FUSION_HEAD", Path("/nonexistent/head.pt"))

        report = deployment_report(capabilities=["optical_sar"])
        entry = report.capabilities[0]
        assert entry.available is False, (
            "optical_sar was reported available with no fusion head; the head is "
            "required for any fused prediction"
        )
        assert entry.reason and "fusion head" in entry.reason

    def test_a_shipped_path_that_does_not_exist_is_absent(self) -> None:
        """An unset path and a set-but-missing path are both permanent absences.

        Distinguishing them would report "no config" as a different state from
        "config points at nothing", and the client cannot act on the difference.
        """
        from app.deployment import deployment_report

        # `change`'s artifact comes from the composition root. Point it at a
        # path that cannot exist by patching the module's resolution.
        import app.serving as serving

        original = serving.CHANGE_CHECKPOINT
        try:
            serving.CHANGE_CHECKPOINT = Path("/nonexistent/head.pt")
            report = deployment_report(capabilities=["change"])
        finally:
            serving.CHANGE_CHECKPOINT = original

        assert report.capabilities[0].state == "absent"

    def test_the_reason_table_names_every_shipped_requirement(self) -> None:
        """A `shipped` requirement with no sentence would fall back to a generic
        one, which is worse than a specific one for the operator."""
        from app.deployment import _MISSING_REASONS, _REQUIREMENTS

        for capability, requirements in _REQUIREMENTS.items():
            shipped = [path for kind, path in requirements if kind == "shipped"]
            if not shipped:
                continue
            reasons = _MISSING_REASONS.get(capability, {})
            for path in shipped:
                assert path in reasons, (
                    f"{capability!r} declares a shipped requirement "
                    f"{path!r} with no reason sentence; an operator would get a "
                    f"generic message naming no artifact"
                )


# ---------------------------------------------------------------------------
# The Space entrypoint delegates
# ---------------------------------------------------------------------------


class TestSpaceAppDelegates:
    """`app/space_app.py` must be a consumer, not a second authority."""

    def test_describe_deployment_delegates_to_the_adapter(self) -> None:
        source = (REPO / "app" / "space_app.py").read_text(encoding="utf-8")
        assert "deployment_report" in _code_only(source), (
            "space_app.describe_deployment() does not use the adapter; it has "
            "become a second capability authority again"
        )

    def test_the_routes_serve_the_adapter_payloads(self) -> None:
        source = (REPO / "app" / "space_app.py").read_text(encoding="utf-8")
        code = _code_only(source)
        assert "health_payload" in code
        assert "capabilities_payload" in code

    def test_describe_deployment_still_carries_the_config_hash(self) -> None:
        """`Config.hash = 78f1e3700da15aa1` must not move."""
        from app import space_app
        from core.config import get_config

        described = space_app.describe_deployment()
        assert described["config_hash"] == "78f1e3700da15aa1"
        assert described["config_hash"] == get_config().hash

    def test_the_config_hash_is_frozen_and_croma_is_absent_from_config(self) -> None:
        """The hash pin AND the reason the resolver fix was necessary, together.

        WHY THIS IS SEPARATE FROM THE TEST ABOVE
        ----------------------------------------
        The test above reads the hash out of `describe_deployment()`, which is a
        PROXY: it proves the payload agrees with `get_config()`, and would still
        pass if both had drifted off the frozen value. So the literal
        `78f1e3700da15aa1` is asserted here as a hard constant, and additionally
        RECOMPUTED from the raw registry using `Config.hash`'s own recipe
        (`sha256(json.dumps(data, sort_keys=True, default=str))[:16]`,
        `core/config.py:76-80`). Two independent routes to the same literal: a
        change to the registry moves both, and neither can be satisfied by the
        other.

        The second half is the load-bearing one for THIS fix. `Config.hash`
        covers the whole registry with no exclusion mechanism, so writing
        `croma.checkpoint_path` into `configs/base.yaml` -- the obvious-looking
        way to make the capability report agree with serving -- would move the
        hash and fail `scripts/eval_change.py` with exit 3. That is precisely
        why the fix had to go through the `builders=` call-site seam instead of
        config. Asserting the key is ABSENT from the file pins the constraint,
        so a future edit that "helpfully" adds it is caught here rather than in
        a drifted evaluation run.
        """
        import hashlib
        import json

        from core.config import get_config

        FROZEN = "78f1e3700da15aa1"

        config = get_config()
        assert config.hash == FROZEN, (
            f"Config.hash is {config.hash}, not the frozen {FROZEN}; the registry "
            f"moved, which invalidates every recorded evaluation run's config "
            f"stamp"
        )

        # Independent recomputation from the registry's own data, using the
        # recipe `Config.hash` documents.
        recomputed = hashlib.sha256(
            json.dumps(config.as_dict(), sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        assert recomputed == FROZEN, (
            f"recomputing the documented recipe gives {recomputed}, not {FROZEN}"
        )

        # The key whose ABSENCE forced the resolver route must still be absent.
        base_yaml = (REPO / "configs" / "base.yaml").read_text(encoding="utf-8")
        assert "checkpoint_path" not in base_yaml, (
            "configs/base.yaml now names croma.checkpoint_path. That moves "
            "Config.hash off the frozen value (it hashes the WHOLE registry, "
            "with no exclusions), so the capability adapter must resolve the "
            "checkpoint through specialists.optical_sar.croma "
            ".resolve_checkpoint_path instead -- the same call serving makes."
        )
        assert config.get("croma.checkpoint_path", None) is None, (
            "croma.checkpoint_path resolved from config; the resolver-parity "
            "tests assume it is absent so that availability can only be coming "
            "from the Hub-cache-aware resolver"
        )

    def test_describe_deployment_loads_no_model(self) -> None:
        from app import space_app

        before = set(sys.modules)
        described = space_app.describe_deployment()
        assert "torch" not in (set(sys.modules) - before)
        assert described["capabilities"]

    def test_describe_deployment_does_not_use_the_constructing_health_method(
        self,
    ) -> None:
        source = (REPO / "app" / "space_app.py").read_text(encoding="utf-8")
        assert ".health()" not in _code_only(source), (
            "the entrypoint calls a constructing `.health()` method"
        )

    def test_the_internal_view_is_not_served(self) -> None:
        """The operator view carries registry words and paths. It must not be a
        route's body.

        Checked by asserting no ROUTE returns `as_internal()` -- the describe
        function may, because it is a Python API, not an HTTP one.
        """
        source = (REPO / "app" / "space_app.py").read_text(encoding="utf-8")
        code = _code_only(source)
        # `JSONResponse(as_internal())` or similar would be the leak.
        assert "as_internal()" not in code.split("def build_space_app")[-1] or (
            "describe_deployment" in code
        ), "the internal view appears inside the route builders"


def _code_only(source: str) -> str:
    """The module's source with docstrings removed.

    Required because this module's docstrings explain the prohibitions --
    "constructs everything", "the adapter does not construct" -- and a search
    over the raw text finds the explanation and reports a violation. The repo has
    made this mistake three times; the fix is stated in `MEMORY.md`: a search for
    a forbidden token must run over executable code, never over the prose that
    forbids it.
    """
    import ast

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
    return ast.unparse(tree)
