"""The deployment runbook must not assert facts the repository contradicts.

WHY THIS EXISTS
---------------
`docs/BACKEND_DEPLOYMENT_RUNBOOK.md` is the document an operator follows when
deploying. Unlike the API contract, its claims are not JSON examples -- they are
version numbers, byte sizes, path names, capability names and environment
variable names. A runbook is worse than useless if a copied command fails or a
quoted number was never measured: the operator concludes the *system* is broken
rather than the *document*.

Two failure modes are pinned here.

1. **Invented measurements.** The runbook quotes command output. If it quotes
   a byte size, a capability list, a function name or an exit code that the
   repository does not produce, the quote is fabricated. Where a fact is cheap
   to verify locally, this module verifies it.
2. **Invented API surface.** The runbook calls into the codebase
   (`build_serving_registry`, `available()`, `CHANGE_CHECKPOINT`,
   `CHANGE_VQA_HEAD`). A runbook that names a function the code does not export
   is a runbook whose commands raise `AttributeError` on the operator's first
   copy-paste.

WHAT IS AND IS NOT ASSERTED
---------------------------
The absence of a deployment cannot be tested here, so these tests assert the
runbook's *local* claims and, separately, assert that the document itself labels
its unimplemented sections as unimplemented. The second class matters more than
it looks: a runbook that describes a live deployment without having performed
one is the exact overclaim the work order forbids.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

RUNBOOK = (
    Path(__file__).resolve().parents[2] / "docs" / "BACKEND_DEPLOYMENT_RUNBOOK.md"
)
ARCHITECTURE = (
    Path(__file__).resolve().parents[2] / "docs" / "DEPLOYMENT_ARCHITECTURE.md"
)
CONTRACT = Path(__file__).resolve().parents[2] / "docs" / "API_CONTRACT.md"


@pytest.fixture(scope="module")
def runbook_text() -> str:
    assert RUNBOOK.exists(), f"runbook missing: {RUNBOOK}"
    return RUNBOOK.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def architecture_text() -> str:
    assert ARCHITECTURE.exists(), f"architecture doc missing: {ARCHITECTURE}"
    return ARCHITECTURE.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# 1. The local claims must be true
# --------------------------------------------------------------------------


def test_the_runbook_names_only_public_serving_entrypoints(runbook_text: str) -> None:
    """Every `app.serving` symbol the runbook names must actually be exported."""
    from app import serving

    referenced = set(re.findall(r"\b(CHANGE_[A-Z_]+)\b", runbook_text))
    assert referenced, "the runbook references no app.serving artifact constants"

    for name in sorted(referenced):
        assert hasattr(serving, name), (
            f"the runbook names app.serving.{name}, which the module does not "
            f"export; a copied command would raise AttributeError"
        )

    # The two builders are referenced in prose; assert them too.
    for name in ("build_serving_registry", "build_serving_controller"):
        assert hasattr(serving, name), (
            f"the runbook names app.serving.{name}, which does not exist"
        )


def test_the_runbook_quoted_artifact_sizes_match_the_real_files(
    runbook_text: str,
) -> None:
    """Any byte size the runbook quotes must equal the file on disk.

    The sizes are quoted in the payload table. If a head is ever regenerated the
    quote must be updated with it -- a stale size reads as a corrupted upload.
    """
    from app.serving import CHANGE_CHECKPOINT, CHANGE_VQA_HEAD

    for path, label in (
        (CHANGE_CHECKPOINT, "change head"),
        (CHANGE_VQA_HEAD, "change_vqa head"),
    ):
        if not path.exists():
            # Absent in a fresh checkout; the runbook documents that case, so
            # there is no size to check.
            continue
        size = path.stat().st_size
        assert f"{size:,}" in runbook_text or str(size) in runbook_text, (
            f"the runbook does not quote the real size of the {label} "
            f"({size} bytes); if it quotes one, it will be wrong"
        )


def test_the_runbook_does_not_quote_a_size_that_disagrees_with_disk(
    runbook_text: str,
) -> None:
    """A quoted size for a head must be the real one, not a remembered one.

    Guards the payload table specifically: it is the place an operator compares
    against `ls`, so a wrong number there produces a false corruption alarm.

    Every artifact the runbook sizes by hand is checked, not just the change
    head -- an earlier version of this test collected only one real size and
    therefore flagged the correct change-VQA size as a mismatch.
    """
    from app.serving import CHANGE_CHECKPOINT, CHANGE_VQA_HEAD

    present = {
        f"{path.stat().st_size:,}"
        for path in (CHANGE_CHECKPOINT, CHANGE_VQA_HEAD)
        if path.exists()
    }
    if not present:
        pytest.skip("no head artifacts are present in this checkout")

    quoted = re.findall(r"([\d]{1,3}(?:,\d{3})+)\s*B\)", runbook_text)
    assert quoted, "the runbook quotes no artifact sizes at all"
    for value in quoted:
        assert value in present, (
            f"the runbook quotes {value} B for an artifact, which no present "
            f"artifact has (real: {sorted(present)})"
        )


def test_the_runbook_capability_list_matches_the_registry(
    runbook_text: str,
) -> None:
    """The six capability names the runbook lists must be the registry's six."""
    from app.serving import build_serving_registry

    declared = set(build_serving_registry().available())

    # The runbook lists them inside a quoted `available(): (...)` transcript.
    match = re.search(r"available\(\):\s*\(([^)]*)\)", runbook_text)
    assert match, "the runbook no longer contains the available() transcript"
    quoted = {item.strip().strip("'\"") for item in match.group(1).split(",")}
    quoted.discard("")

    assert quoted == declared, (
        f"the runbook's capability transcript does not match the registry: "
        f"runbook={sorted(quoted)} registry={sorted(declared)}"
    )


def test_the_runbook_environment_variables_are_the_architecture_ones(
    runbook_text: str, architecture_text: str
) -> None:
    """Every gateway variable in the runbook must appear in the architecture doc.

    The architecture document is the source of the gateway's configuration
    surface. A runbook that introduces a variable the design does not define is
    either a design change smuggled into an operations document, or a typo.
    """
    # Names introduced BY the architecture doc, plus the ones the runbook may
    # legitimately add as operator tunables (timeouts/limits the plan leaves
    # unspecified -- API_CONTRACT.md sections 6 and 8).
    allowed_new = {
        "SATQUERY_UPSTREAM_TIMEOUT_S",
        "SATQUERY_MAX_BODY_BYTES",
        "SATQUERY_MAX_FILE_BYTES",
        "SATQUERY_RATE_LIMIT_PER_IP",
    }
    runbook_vars = set(re.findall(r"\b(SATQUERY_[A-Z_]+|HF_TOKEN|PORT)\b", runbook_text))
    assert runbook_vars, "the runbook defines no gateway environment variables"

    for name in sorted(runbook_vars):
        if name in allowed_new:
            continue
        assert name in architecture_text, (
            f"the runbook defines {name}, which does not appear in "
            f"docs/DEPLOYMENT_ARCHITECTURE.md; the gateway surface would have "
            f"drifted from its design"
        )


def test_the_runbook_declares_the_undecided_items_as_undecided(
    runbook_text: str,
) -> None:
    """The two open decisions must be named as open, not silently chosen.

    The SDK choice and the upload shape are the maintainer's calls. A runbook
    that picked one would be a competing implementation of an unmade decision.
    """
    lowered = runbook_text.lower()
    assert "sdk" in lowered, "the runbook does not mention the SDK decision"
    assert "upload" in lowered or "assets" in lowered, (
        "the runbook does not mention the upload-shape decision"
    )
    # And it must say they block the deploy step.
    assert re.search(r"\bopen\b", lowered), (
        "the runbook does not mark any decision as open"
    )
    assert "do not proceed" in lowered or "do not start" in lowered, (
        "the runbook does not state that the open decisions block deployment"
    )


def test_the_runbook_does_not_claim_a_deployment_happened(
    runbook_text: str,
) -> None:
    """The central honesty check: nothing here has been deployed.

    The work order forbids claiming deployment success unless it was actually
    deployed and tested. This test pins that the document says so, in its own
    words, and does not rely on a reader inferring it.
    """
    lowered = runbook_text.lower()
    assert "no deployment has been performed" in lowered, (
        "the runbook does not state that no deployment has been performed"
    )
    assert "nothing is deployed" in lowered or "nothing described here has been executed" in lowered, (
        "the runbook does not state up-front that nothing is deployed"
    )


def test_the_runbook_marks_unimplemented_sections_as_blocked(
    runbook_text: str,
) -> None:
    """Post-deploy verification cannot run without a deployment; it must say so."""
    assert "BLOCKED" in runbook_text, (
        "the runbook does not use the BLOCKED marker for unrunnable steps"
    )
    # The verification section in particular.
    section = runbook_text.split("## 5. Verification after a deploy", 1)
    assert len(section) == 2, "the runbook no longer has a section 5"
    assert "BLOCKED" in section[1][:400], (
        "section 5 (post-deploy verification) does not open by declaring itself "
        "blocked; a reader could mistake it for something that has been run"
    )


def test_the_runbook_distinguishes_verified_from_designed(
    runbook_text: str,
) -> None:
    """A VERIFIED marker must appear, and DESIGN sections must be labelled."""
    assert "VERIFIED" in runbook_text, (
        "the runbook quotes no measured output; every claim would be unevidenced"
    )
    assert "DESIGN" in runbook_text, (
        "the runbook does not label its design-only sections"
    )


# --------------------------------------------------------------------------
# 2. Cross-document consistency
# --------------------------------------------------------------------------


def test_the_runbook_and_architecture_agree_on_the_config_hash(
    runbook_text: str, architecture_text: str
) -> None:
    """A drifting hash mention is how a config edit gets normalised."""
    from core.config import get_config

    real = get_config().hash
    assert real == "78f1e3700da15aa1", (
        f"the live config hash moved to {real}; the frozen benchmark is invalid "
        f"and both documents must be revisited"
    )
    assert real in runbook_text, "the runbook does not state the frozen hash"
    assert real in architecture_text, "the architecture doc does not state the hash"


def test_the_runbook_and_contract_agree_on_the_zero_gpu_budget(
    runbook_text: str,
) -> None:
    """The 5-minute daily budget is the reason the gateway exists."""
    assert "5 GPU-minutes/day" in runbook_text, (
        "the runbook does not state the ZeroGPU daily budget, which is the "
        "constraint the gateway's rate limiting protects"
    )
    # Read the durations from the deploy manifest itself: it is the single
    # source of truth, and importing a helper would just restate it.
    import yaml

    deploy = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "configs" / "deploy.yaml").read_text(
            encoding="utf-8"
        )
    )
    durations = {
        key: value
        for key, value in deploy["deployment"].items()
        if key.startswith("gpu_duration_")
    }
    assert durations, "the deploy manifest declares no gpu_duration_* values"
    for value in durations.values():
        assert f"{value} s" in runbook_text, (
            f"the runbook does not quote the frozen GPU duration {value}s, so an "
            f"operator cannot check the timeout ordering against the manifest"
        )


def test_the_runbook_timeout_ordering_is_stated(
    runbook_text: str,
) -> None:
    """The gateway timeout must be documented as shorter than the Space budget."""
    assert "shorter" in runbook_text.lower(), (
        "the runbook does not state the gateway-vs-upstream timeout relationship"
    )
    assert "SATQUERY_UPSTREAM_TIMEOUT_S" in runbook_text


def test_the_runbook_timeout_bounds_match_the_frozen_config(
    runbook_text: str,
) -> None:
    """The two numbers bounding the gateway timeout must be the real ones.

    Section 4.2 gives an operator an explicit window (above the longest GPU
    call, below the agent timeout). Those bounds are quoted from the config; if
    either moves, the quoted window becomes advice that would cause the failure
    it exists to prevent.
    """
    import yaml

    from core.config import get_config

    repo = Path(__file__).resolve().parents[2]

    agent_timeout = get_config().as_dict().get("agent", {}).get("timeout_seconds")
    assert agent_timeout, "the agent timeout is not configured"
    assert f"{agent_timeout} s" in runbook_text or f"{agent_timeout}s" in runbook_text, (
        f"the runbook does not quote the real agent timeout ({agent_timeout}s)"
    )

    deploy = yaml.safe_load((repo / "configs" / "deploy.yaml").read_text("utf-8"))
    durations = {
        key: value
        for key, value in deploy["deployment"].items()
        if key.startswith("gpu_duration_")
    }
    longest = max(durations.values())
    assert f"{longest} s" in runbook_text, (
        f"the runbook does not quote the longest GPU duration ({longest}s)"
    )

    # The window must be coherent: the runbook must not advise a value that
    # contradicts the numbers it just quoted.
    assert longest < agent_timeout, (
        f"the frozen config is incoherent: longest gpu duration {longest}s is "
        f"not below the agent timeout {agent_timeout}s"
    )


def test_the_runbook_forbids_editing_the_config_registry(
    runbook_text: str,
) -> None:
    """The one irreversible mistake must be called out where an operator looks."""
    assert "configs/base.yaml" in runbook_text, (
        "the runbook does not name configs/base.yaml, the file whose edit breaks "
        "the frozen benchmark"
    )
    assert "Never edit" in runbook_text or "never edit" in runbook_text, (
        "the runbook does not use an explicit prohibition on editing the config"
    )


def test_the_runbook_never_promises_a_retry_of_analyze(
    runbook_text: str,
) -> None:
    """A retry spends GPU quota twice; the gateway must never do it."""
    assert "never retry" in runbook_text.lower(), (
        "the runbook does not forbid automatic retries of POST /v1/analyze"
    )


def test_the_runbook_marks_this_sections_status_honestly(
    runbook_text: str,
) -> None:
    """Section 8 must exist and must not describe unbuilt parts as done."""
    section = runbook_text.split("## 8. Status of every procedure", 1)
    assert len(section) == 2, "the runbook no longer has a section 8 status table"
    body = section[1]
    assert "BLOCKED" in body, "section 8 does not mark the blocked procedures"
    assert "DESIGN" in body, "section 8 does not mark the designed-only procedures"
    assert "VERIFIED" in body, "section 8 does not mark the verified procedures"


def test_the_runbook_cross_references_the_contract(runbook_text: str) -> None:
    """The runbook must point at the contract the frontend builds against."""
    assert "docs/API_CONTRACT.md" in runbook_text, (
        "the runbook does not reference the API contract"
    )


# ---------------------------------------------------------------------------
# STEP 8 conformance audit -- C-6
# ---------------------------------------------------------------------------
def test_every_gateway_environment_variable_is_documented(runbook_text: str) -> None:
    """C-6. A settable knob that no document names is undiscoverable.

    Found by diffing the `SATQUERY_*` names the gateway READS against those the
    runbook MENTIONS. `SATQUERY_RATE_LIMIT_WINDOW_S` was read by
    `gateway/policy.py` and documented nowhere: an operator could set the count
    half of the rate limit and never learn the window existed.

    The check is deliberately mechanical -- it derives the set from the source
    rather than listing variables, so a variable added to the gateway without a
    runbook row fails here instead of on someone's deployment.
    """
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    read_names: set[str] = set()
    for rel in ("gateway/policy.py", "gateway/app.py", "gateway/assets.py"):
        source = (repo / rel).read_text(encoding="utf-8")
        read_names |= set(re.findall(r'SATQUERY_[A-Z_]+', source))

    assert read_names, "no SATQUERY_* variables found in gateway source"

    undocumented = sorted(n for n in read_names if n not in runbook_text)
    assert not undocumented, (
        f"the gateway reads these variables but the runbook never names them: "
        f"{undocumented}. An operator cannot set what they cannot find."
    )


# --------------------------------------------------------------------------
# Deriving a security limitation from the source (F-5)
# --------------------------------------------------------------------------


def test_the_runbook_discloses_the_rate_limiters_derivation(
    runbook_text: str, architecture_text: str
) -> None:
    """F-5: the limiter's key is derived from a client header, and that must be stated.

    The defect this pins was found on 2026-09-22 by probing the *request* leg --
    the symmetric question to the CORS finding (F-2), which was a header leaking
    back to the client. `gateway/app.py::_client_ip` keys the rate limiter on the
    first hop of `X-Forwarded-For`, which the client supplies. Measured: `5/8`
    requests throttled without the header, **`0/8`** when a fresh value is sent
    per request -- the limiter does not hold at all.

    This is not asserted to be a *code* defect, and the test does not demand a
    code change. It asserts a **documentation** property: an operator who reads
    the runbook or the architecture doc must be able to learn that
    `SATQUERY_RATE_LIMIT_PER_IP` is a fairness control rather than a protection
    control for the GPU quota. Before this, neither document said so, while §5 of
    the architecture doc enumerated twelve other failure modes.

    The check is derived from the source rather than from a hard-coded claim: it
    reads which header `_client_ip` prefers, so if the derivation is ever changed
    the assertion still describes what the code does.
    """
    repo = Path(__file__).resolve().parents[2]
    app_source = (repo / "gateway" / "app.py").read_text(encoding="utf-8")

    # Derive the header the limiter keys on, rather than assuming it.
    match = re.search(
        r"def _client_ip.*?\n(.*?)\n\ndef ", app_source, re.S
    )
    assert match, "`_client_ip` is gone; the limiter's key derivation needs re-reading"
    body = match.group(1)
    keyed_on = re.findall(r'headers\.get\("([a-z-]+)"\)', body)
    assert "x-forwarded-for" in keyed_on, (
        "the limiter no longer keys on X-Forwarded-For; this test (and the F-5 "
        f"documentation) describe a derivation that changed. Now keys on: {keyed_on}"
    )

    for name, text in (
        ("BACKEND_DEPLOYMENT_RUNBOOK.md", runbook_text),
        ("DEPLOYMENT_ARCHITECTURE.md", architecture_text),
    ):
        lowered = text.lower()
        assert "x-forwarded-for" in lowered, (
            f"{name} documents the rate limit but never names X-Forwarded-For, "
            f"which is what the limiter keys on"
        )
        assert "protection" in lowered or "fairness" in lowered, (
            f"{name} does not say whether the limiter is a protection control; an "
            f"operator will assume it is"
        )
        assert "0/8" in text or "0 / 8" in text, (
            f"{name} states the limitation qualitatively but not the measurement; "
            f"the measured bypass is what makes it credible"
        )
