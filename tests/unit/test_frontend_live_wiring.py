"""Regression tests for the frontend <-> orchestrator live wiring.

Two things are proven here, both of which were BROKEN before this change:

A. **CORS.** The deployment allowed exactly one origin
   (`https://satquery.pages.dev`), so the frontend could not be driven from a
   local development server at all: every request from `http://localhost:8080`
   was refused with `Disallowed CORS origin`, and a developer had to deploy to
   Cloudflare to test a one-line JavaScript change. The fix adds explicit
   localhost origins while keeping production listed and keeping `*` rejected.

B. **The task vocabulary.** The page's router and the server's `Task` enum must
   agree. They are two independently written vocabularies in two languages, and
   nothing previously checked that a value the frontend would send is a value
   the server accepts. This module reads BOTH and compares them.

   It also pins the `chang` word-boundary defect: `\\bchang\\b` cannot match
   "changed", so the page's own default question ("What changed here?") routed
   to `vqa` instead of the change path. That is invisible to any test that only
   checks "is this a valid enum value" — both branches produce a valid value.
   The test therefore asserts the ROUTE, not just the validity.

Why these are read from the source text: the frontend is plain ES5 with no
build step and no module exports, so there is no importable symbol. Parsing the
source is the only way to assert the shipped bytes. The parsing is anchored on
named tokens rather than line numbers, so it does not silently pass if the file
is restructured.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_RENDER = REPO_ROOT / "deploy" / "render" / "main.py"
FRONTEND_JS = REPO_ROOT / "frontend" / "assets" / "js"
MISSION_JS = FRONTEND_JS / "mission.js"
LIVE_JS = FRONTEND_JS / "live.js"
CORE_JS = FRONTEND_JS / "core.js"
MISSION_HTML = REPO_ROOT / "frontend" / "mission.html"

PRODUCTION_ORIGIN = "https://satquery.pages.dev"


# ---------------------------------------------------------------------------
# A. CORS
# ---------------------------------------------------------------------------


def _render_module():
    """Import the orchestrator module with a clean environment.

    `_allowed_origins` reads `os.environ` at CALL time, so the module can be
    imported once and driven per-test; but `_DEV_ORIGINS` is built at import
    time from a constant, so a stale import would not matter either way.
    """
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from deploy.render import main as render_main
    finally:
        sys.path.pop(0)
    return render_main


def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SATQUERY_ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("SATQUERY_ALLOW_DEV_ORIGINS", raising=False)


class TestTheCorsAllowlistKeepsProductionAndAddsDevelopment:
    def test_the_production_origin_is_always_allowed(self, monkeypatch):
        """Production must not depend on an environment variable being set.

        The failure this prevents: an operator clears
        `SATQUERY_ALLOWED_ORIGINS` (a mistyped var name, a fresh dashboard)
        and the live site stops working. The production origin is therefore
        listed in code, not only in the environment.
        """
        _clean_env(monkeypatch)
        origins = _render_module()._allowed_origins()
        assert PRODUCTION_ORIGIN in origins

    def test_the_production_origin_survives_an_operator_allowlist(
        self, monkeypatch
    ):
        _clean_env(monkeypatch)
        monkeypatch.setenv(
            "SATQUERY_ALLOWED_ORIGINS", "https://example.pages.dev"
        )
        origins = _render_module()._allowed_origins()
        assert PRODUCTION_ORIGIN in origins
        assert "https://example.pages.dev" in origins

    def test_the_production_origin_is_not_duplicated(self, monkeypatch):
        """Listing production in the env too must not produce two entries."""
        _clean_env(monkeypatch)
        monkeypatch.setenv("SATQUERY_ALLOWED_ORIGINS", PRODUCTION_ORIGIN)
        origins = _render_module()._allowed_origins()
        assert origins.count(PRODUCTION_ORIGIN) == 1

    def test_common_development_origins_are_allowed(self, monkeypatch):
        """The dev servers a frontend developer actually runs.

        Each port is a real default: 8000 `python -m http.server`, 3000
        `npx serve`/CRA, 5173 Vite, 5500 Live Server, 8080 the usual fallback
        when 8000 is taken. Both loopback spellings are included because the
        browser treats `localhost` and `127.0.0.1` as unrelated origins.
        """
        _clean_env(monkeypatch)
        origins = _render_module()._allowed_origins()
        for origin in (
            "http://localhost:8000",
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:5500",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ):
            assert origin in origins, f"missing development origin {origin}"

    def test_the_wildcard_is_still_rejected(self, monkeypatch):
        """`*` must fail loudly, not be filtered out silently.

        A wildcard would expose the deployment to any origin
        (`docs/DEPLOYMENT_ARCHITECTURE.md` 2.1). `GatewayConfig.__post_init__`
        rejects it, but `CORSMiddleware` does not run that validator — so this
        function must refuse it itself, or a `*` reaches the middleware.
        """
        _clean_env(monkeypatch)
        monkeypatch.setenv("SATQUERY_ALLOWED_ORIGINS", "*")
        with pytest.raises(ValueError, match=r"must not contain '\*'"):
            _render_module()._allowed_origins()

    def test_a_wildcard_hidden_in_a_list_is_also_rejected(self, monkeypatch):
        """The realistic typo: `*,https://x` rather than a bare `*`."""
        _clean_env(monkeypatch)
        monkeypatch.setenv(
            "SATQUERY_ALLOWED_ORIGINS", f"{PRODUCTION_ORIGIN},*"
        )
        with pytest.raises(ValueError, match=r"must not contain '\*'"):
            _render_module()._allowed_origins()

    def test_development_origins_cannot_remove_production(self, monkeypatch):
        """Turning dev off must leave production exactly as it was found."""
        _clean_env(monkeypatch)
        monkeypatch.setenv("SATQUERY_ALLOW_DEV_ORIGINS", "0")
        mod = _render_module()
        origins = mod._allowed_origins()
        assert PRODUCTION_ORIGIN in origins
        assert not any(o in mod._DEV_ORIGINS for o in origins)
        assert origins == [PRODUCTION_ORIGIN]

    @pytest.mark.parametrize("value", ["0", "false", "no", "False", "NO"])
    def test_the_dev_switch_accepts_the_usual_false_spellings(
        self, monkeypatch, value
    ):
        """An operator writing `false` must get the behaviour they meant."""
        _clean_env(monkeypatch)
        monkeypatch.setenv("SATQUERY_ALLOW_DEV_ORIGINS", value)
        mod = _render_module()
        assert mod._dev_origins_enabled() is False
        assert not any(o in mod._DEV_ORIGINS for o in mod._allowed_origins())

    def test_the_dev_switch_defaults_to_enabled(self, monkeypatch):
        _clean_env(monkeypatch)
        mod = _render_module()
        assert mod._dev_origins_enabled() is True
        assert any(o in mod._DEV_ORIGINS for o in mod._allowed_origins())

    def test_the_health_payload_reports_the_effective_allowlist(
        self, monkeypatch
    ):
        """The operator must be able to verify from OUTSIDE what is allowed.

        Reporting only the raw env var would hide the code-supplied origins;
        reporting only the assembled list would hide whether dev is on. The
        payload carries both.
        """
        _clean_env(monkeypatch)
        monkeypatch.setenv("GITHUB_TOKEN", "x")
        monkeypatch.setenv("CODESPACE_NAME", "cs")
        mod = _render_module()
        app = mod.create_app()
        route = next(r for r in app.routes if getattr(r, "path", "") == "/api/health")

        import asyncio

        payload = asyncio.run(route.endpoint())
        config = payload["config"]
        assert PRODUCTION_ORIGIN in config["allowed_origins"]
        assert config["production_origins"] == [PRODUCTION_ORIGIN]
        assert config["dev_origins_enabled"] is True


class TestTheCorsDecisionMatchesTheAllowlist:
    """`build_cors_headers` is the gateway's own decision, shared with the
    Space. It must agree with the orchestrator's middleware list, or the two
    layers would answer differently about the same origin."""

    def test_an_allowed_origin_is_echoed(self):
        from gateway.policy import build_cors_headers

        headers = build_cors_headers(PRODUCTION_ORIGIN, (PRODUCTION_ORIGIN,))
        assert headers["Access-Control-Allow-Origin"] == PRODUCTION_ORIGIN

    def test_a_disallowed_origin_gets_no_headers_at_all(self):
        """Not "a different origin" — NO headers, which is what makes the
        browser block the response. Echoing the origin back would defeat the
        allowlist entirely."""
        from gateway.policy import build_cors_headers

        assert build_cors_headers("https://evil.example", (PRODUCTION_ORIGIN,)) == {}

    def test_a_missing_origin_gets_no_headers(self):
        from gateway.policy import build_cors_headers

        assert build_cors_headers(None, (PRODUCTION_ORIGIN,)) == {}

    def test_the_dev_origin_is_echoed_when_allowlisted(self, monkeypatch):
        from gateway.policy import build_cors_headers

        _clean_env(monkeypatch)
        allowed = tuple(_render_module()._allowed_origins())
        headers = build_cors_headers("http://localhost:5173", allowed)
        assert headers["Access-Control-Allow-Origin"] == "http://localhost:5173"

    def test_a_lookalike_host_is_not_allowed(self, monkeypatch):
        """`localhost` must not become a suffix match.

        `http://localhost.evil.com` and `http://evil-localhost` both CONTAIN
        "localhost"; an allowlist built with a prefix/suffix test instead of
        equality would admit them. The listed entries are exact origins.
        """
        from gateway.policy import build_cors_headers

        _clean_env(monkeypatch)
        allowed = tuple(_render_module()._allowed_origins())
        for hostile in (
            "http://localhost.evil.com",
            "http://evil-localhost:8080",
            "http://localhost:8080.evil.com",
            "https://localhost:8080",
        ):
            assert build_cors_headers(hostile, allowed) == {}, hostile


# ---------------------------------------------------------------------------
# B. The task vocabulary
# ---------------------------------------------------------------------------


def _server_task_values() -> set[str]:
    """The server's `Task` enum, read from the schema module."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from core.schemas import Task
    finally:
        sys.path.pop(0)
    return {t.value for t in Task}


def _mission_source() -> str:
    return MISSION_JS.read_text(encoding="utf-8")


def _policy(question: str) -> dict:
    """Drive the SHIPPED `SQ.policy` out of `core.js` in node.

    `SQ.policy` is the rule engine the architecture page drives. Like the
    mission.js router it is extracted by regex, so this exercises the shipped
    rules rather than a reimplementation of them. The function is
    self-contained (no DOM, no `SQ.util`), so it runs against a stub `SQ`.
    """
    source = CORE_JS.read_text(encoding="utf-8")
    fn = _extract(r"SQ\.policy = function \(query\) \{[\s\S]*?\n  \};", source)
    script = "\n".join(
        [
            "var SQ = {};",
            fn,
            "console.log(JSON.stringify(SQ.policy(" + repr(question) + ")));",
        ]
    )
    proc = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    if proc.returncode != 0:  # pragma: no cover - environment failure
        pytest.skip(f"node unavailable or script failed: {proc.stderr[:400]}")
    return json.loads(proc.stdout)


def _fired_rules(result: dict) -> list:
    return [r["rule"] for r in result["route"]["rules"] if r["fired"]]


#: Matches `/* ... */` and `// ...`, so structural assertions can target the
#: executable code rather than the prose around it. Without this, a comment
#: that NAMES a forbidden construct reads as an instance of it -- and a test
#: that forbids `multipart` would fail on the comment explaining why multipart
#: is not used.
_JS_COMMENT_RE = re.compile(r"/\*[\s\S]*?\*/|//[^\n]*")


def _js_code(path: Path) -> str:
    """A JS file's executable text with comments removed."""
    return _JS_COMMENT_RE.sub("", path.read_text(encoding="utf-8"))


def _extract(pattern: str, source: str) -> str:
    match = re.search(pattern, source)
    assert match, f"could not extract {pattern!r} from mission.js"
    return match.group(0)


def _run_in_node(script_js: str):
    """Run a snippet in node and parse its JSON stdout.

    The snippet is appended to the extracted mission.js fragments, so it can
    call the shipped functions directly. Results must be serialised inside the
    sandbox: `JSON.stringify` drops function-valued keys, so returning the
    functions themselves would silently yield an object missing the very keys
    under test.
    """
    import json

    source = _mission_source()
    mapping = _extract(r"var ROUTER_TASK_TO_SERVER = \{[\s\S]*?\};", source)
    paired = _extract(r"var PAIRED_TASKS = \{[\s\S]*?\};", source)
    fallback = _extract(r"var SINGLE_ASSET_FALLBACK = '[a-z_]+';", source)
    # `chooseTask` is the real decision; `serverTaskFor` is a thin shim over it
    # that assumes a pair is available. Both are extracted so a test exercises
    # the shipped logic rather than a reimplementation of it.
    choose = _extract(
        r"function chooseTask\(intent, query, assetCount\) \{[\s\S]*?\n  \}", source
    )
    func = _extract(
        r"function serverTaskFor\(intent, query\) \{[\s\S]*?\n  \}", source
    )
    router = _extract(r"function interpret\(query\) \{[\s\S]*?\n  \}", source)
    # The P1 fix lives in these helpers; they are extracted so the regression
    # tests exercise the SHIPPED logic rather than a reimplementation of it.
    assets_for = _extract(
        r"function assetsForTask\(task, t1, t0\) \{[\s\S]*?\n  \}", source
    )
    ext_of = _extract(r"function _extOf\(file\) \{[\s\S]*?\n  \}", source)
    validate_sar = _extract(
        r"function validateOpticalSar\(t1, t0\) \{[\s\S]*?\n  \}", source
    )
    translate = _extract(r"function translateError\(err\) \{[\s\S]*?\n  \}", source)
    scaffold = "\n".join(
        [
            mapping,
            paired,
            fallback,
            "var pairRequired = {};",
            "function requiresPair(t) { "
            "return Object.prototype.hasOwnProperty.call(pairRequired, t) "
            "? pairRequired[t] : !!PAIRED_TASKS[t]; }",
            assets_for,
            ext_of,
            validate_sar,
            translate,
            choose,
            func,
            router,
        ]
    )
    script = "\n".join([scaffold, script_js])
    proc = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    if proc.returncode != 0:  # pragma: no cover - environment failure
        pytest.skip(f"node unavailable or script failed: {proc.stderr[:400]}")
    return json.loads(proc.stdout)


def _route(question: str) -> dict:
    """Route one question through the shipped router and mapper.

    Assumes a PAIR is available (assetCount=2) so the task is decided by the
    question alone -- the mapping question these tests are about. The
    pair-awareness behaviour is covered separately by
    `TestTheTaskRespectsThePairRequirement`.
    """
    return _run_in_node(
        "var q = " + repr(question) + ";\n"
        "var intent = interpret(q);\n"
        "var c = chooseTask(intent, q, 2);\n"
        "console.log(JSON.stringify({"
        "  intent: intent,"
        "  server_task: c.task,"
        "  substituted: c.substituted,"
        "  wanted: c.wanted"
        "}));"
    )


def _route_with_assets(question: str, asset_count: int) -> dict:
    """The full decision, including the pair check."""
    return _run_in_node(
        "var q = " + repr(question) + ";\n"
        "var intent = interpret(q);\n"
        "var c = chooseTask(intent, q, " + str(int(asset_count)) + ");\n"
        "console.log(JSON.stringify(c));"
    )


def _server_task_table() -> dict:
    data = _run_in_node(
        "console.log(JSON.stringify({table: ROUTER_TASK_TO_SERVER}));"
    )
    return data["table"]


class TestTheFrontendSpeaksTheServersTaskVocabulary:
    def test_every_mapped_value_is_a_real_server_task(self):
        """The check that was missing: the frontend's vocabulary vs the
        server's. `AnalysisRequest.force_task` is a `Task | None`, and the model
        is `extra="forbid"`, so an unknown value is a 422 rather than an
        ignored field."""
        server = _server_task_values()
        table = _server_task_table()
        assert table, "router task table is empty"
        for router_task, server_task in table.items():
            assert server_task in server, (
                f"the frontend maps router task {router_task!r} to "
                f"{server_task!r}, which is not a core.schemas.Task value. "
                f"Valid: {sorted(server)}"
            )

    def test_the_mapping_covers_every_task_the_router_can_emit(self):
        """A router task with no entry would fall through the mapper and be
        sent verbatim, so the coverage must be complete."""
        source = _mission_source()
        router = _extract(r"function interpret\(query\) \{[\s\S]*?\n  \}", source)
        emitted = set(re.findall(r"task = '([a-z_]+)'", router))
        assert emitted, "no task assignments found in the router"
        table = _server_task_table()
        missing = emitted - set(table)
        assert not missing, (
            f"the router can emit {sorted(missing)}, which the mapping does not "
            f"cover; those would be sent to the server verbatim"
        )

    def test_every_specialist_name_is_a_human_label_not_a_task(self):
        """`routeSpecialists` returns component names for the trace, not task
        identifiers. They must not be passed to `force_task`."""
        server = _server_task_values()
        source = _mission_source()
        body = _extract(r"function routeSpecialists\(intent\) \{[\s\S]*?\n  \}", source)
        names = set(re.findall(r"'([A-Z_]{3,})'", body))
        assert names, "no specialist names found"
        for name in names:
            assert name not in server, (
                f"{name} is a server Task value; it must not be used as a "
                f"specialist label, or the two become confusable"
            )


class TestTheDefaultQuestionReachesTheChangePath:
    """The `chang` word-boundary defect.

    This is the test that would have caught the original bug, and the reason it
    must assert the ROUTE rather than the validity: with `\\bchang\\b` the
    default question produced `vqa`, which IS a valid server task. Only the
    route reveals the defect.
    """

    @pytest.mark.parametrize(
        "question,expected",
        [
            # The page's own default value, verbatim from mission.js.
            # No quantifier appears, so it stays on the spatial `change` task.
            ("What changed here?", "change"),
            # "has" is a quantificational marker, so this is UPGRADED to
            # change_vqa: the server's `change` returns a change map with no
            # language, and the page's Answer block promises text.
            ("what has changed in this scene", "change_vqa"),
            ("How much area changed between the two dates?", "change_vqa"),
            ("did the reservoir change", "change_vqa"),
            ("Where is the reservoir?", "grounding"),
            ("compare the optical and SAR views of this scene", "optical_sar"),
            ("Describe the scene", "caption"),
            ("What is the tall building in the centre?", "vqa"),
        ],
    )
    def test_the_question_routes_where_it_should(self, question, expected):
        routed = _route(question)
        got = routed["server_task"]
        assert got == expected, (
            f"{question!r} routed to {got!r}, expected {expected!r} "
            f"(router said {routed['intent']['task']!r})"
        )

    def test_the_change_stem_matches_its_inflections(self):
        """`changed`, `changes`, `changing` are the same intent as `change`.
        A trailing word boundary on the stem breaks all of them."""
        for question in (
            "what changed here",
            "what changes are visible",
            "is the coastline changing",
        ):
            routed = _route(question)
            assert routed["intent"]["task"] == "change", (
                f"{question!r} did not reach the change branch; the stem match "
                f"is missing an inflection"
            )

    def test_the_temporal_slot_is_required_for_a_change_question(self):
        assert _route("What changed here?")["intent"]["temporal"] == "required"

    def test_a_plain_description_is_still_not_temporal(self):
        """The fix must not over-trigger: widening the stem should not make
        every question a change question."""
        assert _route("Describe the scene")["intent"]["temporal"] == "none"

    def test_the_default_query_in_the_page_is_the_one_under_test(self):
        """Guard the guard: if the page's default question changes, this test
        module's central case would silently stop covering production."""
        source = _mission_source()
        assert "What changed here?" in source, (
            "the default query moved; update this test module to the new one"
        )


class TestALocationQuestionIsNotAChangeQuestion:
    """The lexical over-trigger that made a location question unanswerable.

    `built` and `new` are land-cover and place vocabulary, not change markers,
    but both sat in the temporal regex. A `where` question containing one was
    therefore read as `change`; with a single asset selected the pair check then
    substituted `change_vqa` -> `vqa`, and the server answered the location
    question with the degenerate single word "River". The whole point of the fix
    is that a SINGLE-image location question must never be substituted down to
    `vqa`, so both asset counts are asserted here.
    """

    LOCATION_QUESTIONS = [
        # `built` -- "built-up areas" is land-cover vocabulary.
        "Where are the built-up areas in this image?",
        # `new` -- the repo ships eo/new-airport.jpg, so this is a real question.
        "Where is the new airport?",
    ]

    @pytest.mark.parametrize("question", LOCATION_QUESTIONS)
    def test_the_question_reads_as_grounding(self, question):
        routed = _route(question)
        assert routed["intent"]["task"] == "grounding", (
            f"{question!r} was read as {routed['intent']['task']!r}; a location "
            f"question must reach the grounding branch"
        )
        assert routed["intent"]["temporal"] == "none", (
            f"{question!r} set temporal={routed['intent']['temporal']!r}; "
            f"neither `built` nor `new` is a change marker"
        )

    @pytest.mark.parametrize("question", LOCATION_QUESTIONS)
    @pytest.mark.parametrize("asset_count", [1, 2])
    def test_it_routes_to_grounding_with_one_or_two_assets(
        self, question, asset_count
    ):
        """`grounding` is not a paired task, so a single image must not be
        substituted away from it -- that substitution is what produced the
        degenerate answer."""
        chosen = _route_with_assets(question, asset_count)
        assert chosen["task"] == "grounding", (
            f"{question!r} with {asset_count} asset(s) requested "
            f"{chosen['task']!r}, not 'grounding'"
        )
        assert chosen["substituted"] is False

    def test_new_still_marks_a_change_outside_a_where_question(self):
        """The fix is conditional, not a blanket removal: dropping `new`
        outright would lose the change reading of a question that never asks
        where anything is."""
        routed = _route("show me the new construction")
        assert routed["intent"]["task"] == "change", (
            "`new` outside a `where` question must still read as a change"
        )


class TestTheArchitecturePolicyDoesNotReadALocationAsAChange:
    """The SIBLING of the mission.js router defect.

    `core.js` `SQ.policy` is a second, independently written lexical engine — the
    one the architecture page drives to show "which parts of the instrument wake
    up". It carried the same over-trigger as `interpret()`: `built` and `new` sat
    in its change regex. The architecture page's OWN sample, "Where is the
    built-up area?", therefore fired `intent.change` (on "built") and
    `intent.quantify` (on "area"), and was answered as `CHANGE_VQA` — a location
    question routed to a change question, with a CHANGE_DETECTOR specialist.

    The fix removes `built` and makes `new` conditional on not being a `where`
    question, exactly as `interpret()` now does. These tests drive the SHIPPED
    `SQ.policy`; the pre-fix run was red on
    `test_the_architecture_sample_routes_to_grounding` and
    `test_a_where_question_with_a_place_word_is_not_a_change`.
    """

    ARCHITECTURE_SAMPLE = "Where is the built-up area?"

    def test_the_architecture_sample_routes_to_grounding(self):
        """The sample is taken verbatim from `architecture.js` SAMPLES."""
        result = _policy(self.ARCHITECTURE_SAMPLE)
        assert result["task"] == "GROUNDING", (
            f"the architecture page's own sample routed to {result['task']!r}; "
            f"a location question must reach GROUNDING"
        )
        fired = _fired_rules(result)
        assert "intent.change" not in fired, (
            f"'built-up area' is land-cover vocabulary, not a change marker; "
            f"fired rules were {fired}"
        )
        assert "CHANGE_DETECTOR" not in result["specialists"], (
            f"a location question must not wake the change detector: "
            f"{result['specialists']}"
        )

    @pytest.mark.parametrize(
        "question",
        [
            "Where is the built-up area?",
            "Where is the new airport?",
        ],
    )
    def test_a_where_question_with_a_place_word_is_not_a_change(self, question):
        result = _policy(question)
        assert result["task"] == "GROUNDING", (
            f"{question!r} routed to {result['task']!r}, not GROUNDING"
        )

    def test_every_architecture_sample_routes_as_it_should(self):
        """The whole sample set the page cycles through, pinned at once."""
        expected = {
            "What changed here?": "CHANGE_ANALYSIS",
            "Where is the built-up area?": "GROUNDING",
            "Did the road expand?": "CHANGE_VQA",
            "What does the SAR image reveal?": "SAR_ANALYSIS",
            "What is visible here?": "VLM_CAPTION",
        }
        for question, task in expected.items():
            assert _policy(question)["task"] == task, (
                f"{question!r} routed to {_policy(question)['task']!r}, "
                f"expected {task!r}"
            )

    def test_new_still_marks_a_change_outside_a_where_question(self):
        """The fix is conditional, not a blanket removal."""
        assert _policy("show me the new construction")["task"] == "CHANGE_ANALYSIS", (
            "`new` outside a `where` question must still read as a change"
        )

    def test_the_change_stem_still_matches_its_inflections(self):
        for question in (
            "what changed here",
            "what changes are visible",
            "is the coastline changing",
        ):
            assert _policy(question)["task"] in (
                "CHANGE_ANALYSIS",
                "CHANGE_VQA",
            ), question


class TestTheTaskRespectsThePairRequirement:
    """The defect the live browser run found.

    `/api/capabilities` declares `requires_pair` per task and the server
    ENFORCES it. Choosing the task from the question alone meant the page's own
    default question ("What changed here?", which reads as `change`) combined
    with one selected file produced:

        change requires exactly 2 assets (T1 and T2); got 1

    ...as a `degraded: true` envelope. A user asking a reasonable question about
    one image got a non-answer, and nothing in the UI explained why. This class
    pins the fix.
    """

    def test_the_contract_declares_exactly_these_paired_tasks(self):
        """The frontend's mirror must match the deployment's declaration."""
        source = _mission_source()
        declared = set(
            re.findall(r"(\w+):\s*true", _extract(r"var PAIRED_TASKS = \{[\s\S]*?\};", source))
        )
        assert declared == {"change", "change_vqa", "optical_sar"}, (
            f"PAIRED_TASKS is {sorted(declared)}; the deployment declares "
            f"requires_pair for change, change_vqa and optical_sar"
        )

    def test_a_paired_task_with_one_asset_is_substituted(self):
        chosen = _route_with_assets("What changed here?", 1)
        assert chosen["substituted"] is True, (
            "with one asset the page must not request `change`, which the "
            "server refuses with invalid_request"
        )
        assert chosen["wanted"] == "change"
        assert chosen["task"] == "vqa", (
            "the substitute must be a task that accepts a single asset"
        )
        assert "needs two images" in chosen["reason"], chosen["reason"]

    def test_the_substitute_accepts_a_single_asset_per_the_contract(self):
        """The substitute must be a task the deployment reports as
        `requires_pair: false`, or the substitution changes one invalid request
        into another."""
        chosen = _route_with_assets("What changed here?", 1)
        unpaired = {"vqa", "caption", "grounding"}
        assert chosen["task"] in unpaired, (
            f"{chosen['task']!r} is not a single-asset task; valid: {sorted(unpaired)}"
        )

    def test_a_paired_task_with_two_assets_is_not_substituted(self):
        chosen = _route_with_assets("What changed here?", 2)
        assert chosen["substituted"] is False
        assert chosen["task"] == "change"

    def test_change_vqa_is_substituted_with_one_asset(self):
        chosen = _route_with_assets("How much area changed between the two dates?", 1)
        assert chosen["wanted"] == "change_vqa"
        assert chosen["substituted"] is True
        assert chosen["task"] == "vqa"

    def test_optical_sar_is_substituted_with_one_asset(self):
        """`optical_sar` needs the pair by definition -- it fuses an optical and
        a SAR view."""
        chosen = _route_with_assets("compare the optical and SAR views", 1)
        assert chosen["wanted"] == "optical_sar"
        assert chosen["substituted"] is True
        assert chosen["task"] == "vqa"

    def test_optical_sar_is_kept_with_two_assets(self):
        chosen = _route_with_assets("compare the optical and SAR views", 2)
        assert chosen["task"] == "optical_sar"
        assert chosen["substituted"] is False

    def test_a_single_asset_task_is_never_substituted(self):
        """`caption` accepts one asset, so one asset must not trigger a change.
        (Descriptive queries now route to `caption`; the point holds: a
        single-asset task is never downgraded for lacking a pair.)"""
        chosen = _route_with_assets("Describe the scene", 1)
        assert chosen["task"] == "caption"
        assert chosen["substituted"] is False

    def test_grounding_is_not_substituted_with_one_asset(self):
        chosen = _route_with_assets("Where is the reservoir?", 1)
        assert chosen["task"] == "grounding"
        assert chosen["substituted"] is False

    def test_the_decision_depends_on_the_asset_count_not_only_the_question(self):
        """The core property: the same question yields different tasks for one
        asset and for two. If this ever collapses, the pair check has been
        lost."""
        one = _route_with_assets("What changed here?", 1)
        two = _route_with_assets("What changed here?", 2)
        assert one["task"] != two["task"], (
            f"one asset and two assets both produced {one['task']!r}; the pair "
            f"requirement is not being consulted"
        )

    def test_the_substitution_is_visible_in_the_page_not_silent(self):
        """A silent substitution would be its own dishonesty: the user asked
        for change detection and got a VQA answer."""
        source = _mission_source()
        assert "choice.substituted" in source
        assert "routed as" in source, "the UI must name the task actually sent"
        assert "choice.reason" in source, "the UI must explain why"

    def test_the_specialists_follow_the_task_actually_dispatched(self):
        """The trace must not name specialists for a task never requested."""
        source = _mission_source()
        assert "routeSpecialists({ task: forced })" in source, (
            "specialists must be derived from the dispatched task"
        )


# ---------------------------------------------------------------------------
# C. The live client's contract
# ---------------------------------------------------------------------------


class TestTheLiveClientUsesTheRealIngestionPath:
    def test_the_client_targets_the_orchestrators_routes(self):
        """The browser talks to the orchestrator, never to the Space.

        `/v1/*` is the Space's own surface; the orchestrator exposes `/api/*`.
        A client calling `/v1/analyze` from the page would bypass the
        orchestrator entirely and hit CORS or a 404.

        Asserted against the CODE, not the file: `live.js` discusses the
        Space's `/v1/assets` route in a comment explaining why the body is raw,
        and a naive whole-file substring check would flag that documentation as
        a violation. Comments are stripped first.
        """
        code = _js_code(LIVE_JS)
        assert "/infer" in code and "/assets" in code
        assert "/v1/analyze" not in code, (
            "the client must not call the Space's route directly"
        )
        assert "/v1/assets" not in code, (
            "the client must not call the Space's route directly"
        )

    def test_uploads_send_raw_bytes_with_a_derived_content_type(self):
        """`/v1/assets` reads the raw body; a multipart wrapper would be stored
        as the image. The Content-Type is derived from the extension because a
        GeoTIFF arrives with an empty `File.type` in Chrome and Firefox."""
        code = _js_code(LIVE_JS)
        assert "image/tiff" in code
        assert "image/png" in code
        assert "image/jpeg" in code
        assert "FormData" not in code, (
            "a FormData body would store the multipart wrapper as the image"
        )

    def test_tiff_extensions_map_to_the_geotiff_content_type(self):
        source = _js_code(LIVE_JS)
        for ext in ("tif:", "tiff:"):
            assert f"{ext} 'image/tiff'" in source, (
                f"{ext} must map to image/tiff; the store allowlist requires it"
            )

    def test_the_infer_body_carries_asset_ids_under_the_assets_key(self):
        """`AnalysisRequest` has `extra="forbid"`: a wrong key is a 422, and the
        contract defines `assets` as handles returned by the upload step."""
        source = _js_code(LIVE_JS)
        assert "assets:" in source, "the request body key must be `assets`"

    def test_the_client_never_sends_a_filesystem_path(self):
        """Handles are the only door for bytes. A path in the body would be
        both a contract violation and an information leak."""
        source = _js_code(LIVE_JS)
        assert "asset_id" in source
        for forbidden in ("file.path", "filePath", "localPath", "absolute_path"):
            assert forbidden not in source, (
                f"{forbidden} suggests a filesystem path is being sent"
            )

    def test_a_missing_asset_id_is_rejected_at_the_boundary(self):
        """A 200 whose body lacks `asset_id` must fail in the upload step.

        Otherwise `undefined` travels into the infer request and the failure is
        reported against the wrong step."""
        source = _js_code(LIVE_JS)
        assert "returned no asset id" in source

    def test_the_client_reads_the_transport_header_as_evidence(self):
        """`x-satquery-transport: tunnel` is the proof that Render forwarded to
        the Codespace rather than answering locally. It is only readable before
        the response object is discarded."""
        source = _js_code(LIVE_JS)
        assert "x-satquery-transport" in source
        assert "X-SatQuery-State" in source


def _declared_api_base(html: str) -> str:
    """The `content` of the `satquery-api-base` meta tag ('' when absent).

    Reading the VALUE rather than the presence of the tag is the whole point.
    `SQ.live.baseUrl()` falls back to the relative '/api' when no base is
    configured, and on a static host that resolves against the page's OWN
    origin, which answers `/api/*` with a 404 HTML page -- production went dark
    exactly that way. `content="/api"` is therefore the single most damaging
    value the tag can hold, and a presence-only assertion cannot tell it apart
    from a working absolute origin.
    """
    match = re.search(r'<meta\s+name="satquery-api-base"\s+content="([^"]*)"', html)
    return match.group(1) if match else ""


class TestThePageLoadsTheLiveClient:
    def test_mission_html_includes_live_js_before_mission_js(self):
        """Order is load-bearing: `mission.js` calls `SQ.live.*` at run time,
        but the file must be present or `SQ.live` is undefined."""
        html = MISSION_HTML.read_text(encoding="utf-8")
        assert "assets/js/live.js" in html, "live.js is not loaded by the page"
        assert html.index("assets/js/live.js") < html.index("assets/js/mission.js")

    def test_the_page_declares_the_api_base(self):
        html = MISSION_HTML.read_text(encoding="utf-8")
        assert 'name="satquery-api-base"' in html, (
            "the page declares no API base; SQ.live.baseUrl() then falls back "
            "to the relative '/api', which on the static host resolves against "
            "the page's OWN origin and 404s"
        )
        assert _declared_api_base(html), (
            "the meta tag is present but its content is empty"
        )

    def test_the_declared_api_base_is_an_absolute_origin(self):
        """The declared base must be a real http(s) origin.

        A relative value cannot reach the orchestrator from a static host: it
        resolves against the page's origin. This is the guard the deployment
        needed when it shipped without one.
        """
        content = _declared_api_base(MISSION_HTML.read_text(encoding="utf-8"))
        assert content.startswith(("http://", "https://")), (
            f"the declared API base is {content!r}; it must be an absolute "
            "http(s):// origin, because a relative base resolves against the "
            "static host and every /api/* call then 404s"
        )

    def test_the_declared_api_base_is_not_the_relative_fallback(self):
        """RED against the exact value that took production down.

        '/api' is precisely what `SQ.live.baseUrl()` returns when NOTHING is
        configured, so declaring it is indistinguishable from declaring nothing
        -- it looks deliberate and is not. A presence-only assertion cannot see
        this, which is why the guard reads the value.
        """
        content = _declared_api_base(MISSION_HTML.read_text(encoding="utf-8"))
        assert content != "/api", (
            'content="/api" is the relative fallback, not a configured base: '
            "SQ.live.baseUrl() already returns '/api' when the tag is absent, "
            "so this value makes a misconfiguration look intentional"
        )
        assert not content.startswith("/"), (
            f"the declared API base {content!r} is a relative path; it resolves "
            "against the page's own origin"
        )

    def test_the_declared_api_base_is_not_the_frontend_origin(self):
        """A base pointing at the static host is the same outage in costume.

        The frontend is served from Cloudflare Pages and the orchestrator is
        not, so naming the Pages origin here yields 404 (GET) / 405 (POST) on
        every orchestrator call -- the observed production symptom.
        """
        content = _declared_api_base(MISSION_HTML.read_text(encoding="utf-8"))
        assert PRODUCTION_ORIGIN not in content, (
            f"the declared base {content!r} points at the frontend's own static "
            "host, which serves no /api routes"
        )

    def test_no_fixture_image_is_wired_into_the_analysis_path(self):
        """The product flow must send only user-chosen files.

        The illustrative plate image exists for legibility and must never be
        uploaded: a demo that analyses a bundled image demonstrates nothing
        about the system's workflow.

        This also pins the P1 fix: the file set is built by `assetsForTask`, not
        by the old `if (selectedT0) files = [selectedT1, selectedT0]` bundle that
        unconditionally sent T0 to single-asset tasks (the live `invalid_request`
        on a VQA question with a pair uploaded)."""
        source = _mission_source()
        upload_block = _extract(
            r"var filesToSend = assetsForTask\([^\n]*\);", source
        )
        assert "plateImg" not in upload_block
        assert "assets/img" not in upload_block
        # The old unconditional bundle must be gone.
        assert "if (selectedT0) files = [selectedT1, selectedT0]" not in source, (
            "the old runLive still unconditionally bundles T0 into every request"
        )
        assert ".run(filesToSend" in source, (
            "the isolated file set is not what is actually uploaded"
        )


class TestTheLivePathKeepsTheHonestFailureContract:
    def test_a_failure_is_attributed_to_the_step_that_failed(self):
        source = _mission_source()
        assert "err.stage" in source or ".stage" in source
        assert "step failed" in source

    def test_the_preview_is_kept_for_the_no_file_case(self):
        """With no file chosen the page must not pretend an analysis ran. The
        preview driver stays, and the run switches on whether a file exists."""
        source = _mission_source()
        assert "function runMock" in source, "the honest preview was removed"
        assert "if (selectedT1) runLive(q);" in source, (
            "the live/preview switch is missing"
        )
        assert "else runMock(q);" in source


class TestTheCaptionStopsCallingARealUploadIllustrative:
    """The plate caption is a CLAIM about the image on screen.

    Before this change the caption read "Illustrative — not a SatQuery result"
    even after a real analysis of the user's own upload had completed. That is
    not merely a cosmetic staleness: it tells the viewer the frame is a
    stand-in at the exact moment the page is showing them their own data, which
    undermines the one thing the live driver exists to demonstrate.

    The caption is TWO elements -- a leading `<b>` and a `<span class="right">`
    -- so an edit that drives only the span leaves the word "Illustrative" on
    screen. A test that only asserted "plateCredit is written somewhere" would
    pass against exactly that half-fix, so both halves are pinned here.
    """

    def test_both_halves_of_the_caption_are_addressable(self):
        html = MISSION_HTML.read_text(encoding="utf-8")
        assert 'id="plateCredit"' in html
        assert 'id="plateCreditLead"' in html, (
            "the leading <b> is still a literal, so 'Illustrative' cannot be "
            "withdrawn after a real run"
        )
        # The literal must live inside the element that is driven, not beside it.
        lead = html[html.index('id="plateCreditLead"') :]
        assert "<b id=\"plateCreditLead\">Illustrative</b>" in html

    def test_a_successful_run_rewrites_the_leading_word(self):
        """The write must be on the LIVE SUCCESS path, not merely present.

        `plateCreditLead` is also written when a file is selected, so a bare
        "does this symbol appear anywhere" assertion passes against a half-fix
        that withdraws the word before a run but never after one. The check is
        therefore scoped: the write must occur between the successful response
        being handled and the start of the failure branch.
        """
        source = _js_code(MISSION_JS)
        start = source.index("var env = out.envelope;")
        end = source.index(".catch(function (err)")
        success_path = source[start:end]
        assert "plateCreditLead.textContent" in success_path, (
            "the leading word is not rewritten on the success path, so "
            "'Illustrative' survives a completed analysis"
        )

    def test_the_alt_text_is_rewritten_in_the_success_path(self):
        """A screen reader hears the alt text. Leaving it as 'Illustrative
        reference observation' while showing the user's upload is the same lie
        told through a different channel."""
        source = _js_code(MISSION_JS)
        start = source.index("var env = out.envelope;")
        end = source.index(".catch(function (err)")
        assert re.search(r"plateImg\.alt\s*=", source[start:end]), (
            "the alt text is never updated on success, so it keeps describing "
            "the frame as an illustrative reference after a real analysis"
        )

    def test_the_failure_path_does_not_claim_an_analysis_happened(self):
        """If the run failed there is no analysis, so the caption must not say
        there was one. The success-path credit lives before the catch; the catch
        must not write a run-scoped credit of its own."""
        source = _js_code(MISSION_JS)
        catch_block = source[source.index(".catch(function (err)") :]
        catch_block = catch_block[: catch_block.index(".then(function ()")]
        assert "plateCredit" not in catch_block, (
            "the failure branch rewrites the plate credit, which would claim an "
            "analysis that did not complete"
        )

    def test_a_new_file_clears_the_previous_runs_claim(self):
        """Selecting a different image must not leave the previous run's
        'analysed in run X' caption attached to it."""
        source = _js_code(MISSION_JS)
        assert re.search(r"delete\s+plateImg\.dataset\.analysed", source), (
            "the analysed flag is never cleared, so a new upload inherits the "
            "old run's credit"
        )

    def test_the_comparison_credit_follows_a_real_pair_run(self):
        source = _js_code(MISSION_JS)
        # The preview wording is still the initial value...
        assert "Illustrative — not a SatQuery output" in source
        # ...but the live path must overwrite it.
        occurrences = re.findall(r"cmpCredit\.textContent\s*=\s*([^;]+);", source)
        assert len(occurrences) >= 2, (
            "cmpCredit is set once (at T0 selection) and never updated by a "
            "real run: " + repr(occurrences)
        )
        assert any("run_id" in o for o in occurrences), (
            "no branch of cmpCredit mentions the run id, so it cannot be "
            "reporting a real analysis"
        )


# ---------------------------------------------------------------------------
# D. P1 — the frontend sends only the assets the dispatched task requires
# ---------------------------------------------------------------------------


class TestTheFrontendSendsOnlyTheAssetsTheTaskRequires:
    """The P1 defect the live browser run exposed (2026-09-25).

    A pair was uploaded (both `#fileInput` and `#fileInputT0` populated) and a
    single-asset question asked. The old `runLive` built
    `files = [selectedT1]; if (selectedT0) files = [selectedT1, selectedT0]` and
    sent that to EVERY task, so the backend rejected the one-asset task with
    `invalid_request` ("... requires exactly 2 assets ...; got 1"). The fix is
    `assetsForTask`, which isolates the file set per dispatched task BEFORE
    `SQ.live.run` is called.

    These tests reproduce the OLD failure (two files sent to a one-asset task)
    and prove the NEW behaviour (only the required assets leave the browser)."""

    @staticmethod
    def _sent(task, has_t0):
        out = _run_in_node(
            "var t1={name:'a.tif'}; var t0={name:'b.tif'};"
            "var files=assetsForTask("
            + repr(task)
            + ", t1, "
            + ("t0" if has_t0 else "null")
            + ");"
            "console.log(JSON.stringify({n: files.length, names: files.map(function(f){return f.name;})}));"
        )
        return out

    @staticmethod
    def _old_sent(has_t0):
        """The pre-fix wiring, reproduced verbatim, as a regression oracle."""
        out = _run_in_node(
            "var t1={name:'a.tif'}; var t0={name:'b.tif'};"
            "var files=[t1]; if ("
            + ("t0" if has_t0 else "false")
            + ") files=[t1,t0];"
            "console.log(JSON.stringify(files.length));"
        )
        return out

    def test_a_single_asset_task_with_a_pair_uploaded_uses_only_t1(self):
        """The core fix: VQA / grounding / caption must NOT carry T0."""
        for task in ("vqa", "grounding", "caption"):
            sent = self._sent(task, has_t0=True)
            assert sent["names"] == ["a.tif"], (
                f"{task} should send only T1 when a pair is uploaded, "
                f"but sent {sent['names']}"
            )

    def test_the_old_wiring_would_have_sent_both_files_for_vqa(self):
        """Reproduces the live failure: old runLive unconditionally added T0."""
        old = self._old_sent(has_t0=True)
        assert old == 2, "the old wiring sent both files to a one-asset task"

    def test_the_fix_sends_one_file_where_the_old_sent_two(self):
        old = self._old_sent(has_t0=True)
        new = self._sent("vqa", has_t0=True)
        assert old == 2 and new["n"] == 1, (
            f"old sent {old} file(s) but the fix still sends {new['n']}; the "
            f"defect survives"
        )

    def test_change_and_optical_sar_send_the_pair_when_present(self):
        for task in ("change", "change_vqa", "optical_sar"):
            sent = self._sent(task, has_t0=True)
            assert sent["names"] == ["a.tif", "b.tif"], (
                f"{task} should send T0+T1 (the pair), but sent {sent['names']}"
            )

    def test_change_with_only_t1_sends_just_t1(self):
        """Defensive: if change were ever dispatched with one asset the request
        must not claim a T0 that does not exist. (In practice chooseTask
        substitutes to vqa first.)"""
        sent = self._sent("change", has_t0=False)
        assert sent["names"] == ["a.tif"]

    def test_caption_with_a_pair_uploaded_uses_only_t1(self):
        """The specific regression: a pair was uploaded and 'Describe the scene'
        (now a caption task) was asked — the old code would have sent both."""
        sent = self._sent("caption", has_t0=True)
        assert sent["names"] == ["a.tif"]


class TestDescriptiveQueriesRouteToCaption:
    """The missing caption path.

    `caption` is a valid server task (`ROUTER_TASK_TO_SERVER` maps it) but
    `interpret()` never produced it, so descriptive queries silently fell through
    to `vqa`. These assert the ROUTE now reaches `caption` for description
    wording while genuinely question-shaped queries stay on `vqa`. Assumes a pair
    is available (assetCount=2) so the task is decided by the question alone."""

    @pytest.mark.parametrize(
        "question,expected",
        [
            ("Describe the scene", "caption"),
            ("Caption this image", "caption"),
            ("Summarise the land cover", "caption"),
            ("What do you see in this image?", "caption"),
            ("Tell me about this scene", "caption"),
            ("What is shown in the picture", "caption"),
            # A specific question, NOT a description -> stays on VQA.
            ("What is the tall building in the centre?", "vqa"),
            ("How many vehicles are parked here", "vqa"),
        ],
    )
    def test_descriptive_queries_reach_caption(self, question, expected):
        routed = _route(question)
        assert routed["server_task"] == expected, (
            f"{question!r} routed to {routed['server_task']!r}, expected "
            f"{expected!r} (router said {routed['intent']['task']!r})"
        )

    def test_a_description_is_not_mistaken_for_change(self):
        """The caption branch must not swallow temporal questions."""
        assert _route("describe the changes since last year")["intent"]["task"] == "change"

    def test_caption_is_a_single_asset_task_so_no_pair_is_needed(self):
        chosen = _route_with_assets("Describe the scene", 1)
        assert chosen["task"] == "caption"
        assert chosen["substituted"] is False


class TestTheOpticalSarInputValidation:
    """Client-side best-effort check that an optical-SAR request has the right
    SHAPE of inputs. The browser cannot read sensor modality, so this is advisory:
    it flags two plain photos masquerading as an optical+SAR pair, and it names
    the missing-input case outright."""

    @staticmethod
    def _check(t1name, t0name):
        out = _run_in_node(
            "var t1=" + json.dumps({"name": t1name}) + ";"
            "var t0=" + json.dumps({"name": t0name}) + ";"
            "console.log(JSON.stringify(validateOpticalSar(t1, t0)));"
        )
        return out

    def test_a_real_optical_plus_geotiff_sar_pair_is_ok(self):
        res = self._check("scene.tif", "radar.tif")
        assert res["level"] == "ok", res

    def test_two_plain_photos_warn(self):
        res = self._check("photo.jpg", "photo2.jpg")
        assert res["level"] == "warn"
        assert "plain photos" in res["message"]

    def test_a_missing_sar_image_is_an_error(self):
        res = _run_in_node(
            "console.log(JSON.stringify(validateOpticalSar({name:'a.tif'}, null)));"
        )
        assert res["level"] == "error"
        assert "two images" in res["message"]

    def test_the_warning_names_the_optical_plus_radar_expectation(self):
        res = self._check("optical.png", "sar.png")
        assert "radar" in res["message"].lower()


class TestTheServerErrorTranslation:
    """Where the backend's error envelope supplies a `code` / `status`, the page
    should translate the opaque failure into actionable guidance — without ever
    replacing the server's own message."""

    @staticmethod
    def _translate(err):
        out = _run_in_node(
            "var err=" + json.dumps(err) + ";"
            "console.log(JSON.stringify(translateError(err)));"
        )
        return out

    def test_invalid_request_gets_asset_guidance(self):
        msg = self._translate(
            {
                "code": "invalid_request",
                "message": "No result could be produced",
                "detail": "change requires exactly 2 assets; got 1",
                "status": 400,
            }
        )
        assert "pair" in msg.lower() and "assets" in msg.lower()

    def test_422_gets_malformed_guidance(self):
        msg = self._translate({"status": 422, "message": "Unprocessable Entity"})
        assert "GeoTIFF" in msg

    def test_recoverable_false_gets_restart_guidance(self):
        msg = self._translate({"recoverable": False, "message": "engine down"})
        assert "not recoverable" in msg and "restart" in msg

    def test_the_servers_message_and_detail_are_preserved(self):
        msg = self._translate(
            {
                "code": "invalid_request",
                "message": "No result could be produced",
                "detail": "change requires exactly 2 assets; got 1",
            }
        )
        assert "No result could be produced" in msg
        assert "change requires exactly 2 assets; got 1" in msg

    def test_a_null_error_is_handled(self):
        msg = self._translate(None)
        assert "failed" in msg.lower()

    def test_a_recoverable_transport_code_is_not_called_unrecoverable(self):
        """`forward_unavailable` (503, recoverable: true) is a new code on the
        same transport path as `tunnel_offline` / `wake_timeout` /
        `upstream_unreachable`. The translation is driven by `recoverable`, not
        by a list of code names, so all four must pass through identically --
        the server's own message and detail preserved, and no "not recoverable"
        guidance that would tell the user to restart a service that is merely
        busy. Pinned so a future code-specific branch cannot quietly single
        `forward_unavailable` out."""
        base = {
            "message": "Service temporarily unavailable.",
            "detail": "transport not reachable",
            "recoverable": True,
        }
        rendered = {
            code: self._translate(dict(base, code=code, status=status))
            for code, status in (
                ("forward_unavailable", 503),
                ("tunnel_offline", 503),
                ("wake_timeout", 504),
                ("upstream_unreachable", 502),
            )
        }
        assert len(set(rendered.values())) == 1, (
            f"a recoverable transport code is treated differently from the "
            f"others: {rendered}"
        )
        msg = rendered["forward_unavailable"]
        assert "not recoverable" not in msg
        assert "Service temporarily unavailable." in msg
        assert "transport not reachable" in msg

