"""SatQuery AI — the deployment capability adapter.

WHAT THIS MODULE IS, AND WHY IT EXISTS
--------------------------------------
The repository contained **two independent answers** to the question "what can
this deployment do?", and only one of them was ever served:

    core.registry.SpecialistRegistry           6 capabilities
       states: available | degraded | unavailable       <- REGISTRY vocabulary

    app.space_app.describe_deployment()        2 capabilities
       derived from two `Path.exists()` calls     <- AD-HOC, narrow

`GET /v1/health` and `GET /v1/capabilities` both read the second one. So the
served answer was derived from *strictly less evidence* than the registry could
already provide, and `AnalysisController.health()` -- which reads the registry --
was a third, unserved shape that the contract's own `HealthStatus` model
**rejects** (finding H-1).

That is the three-copy problem (`MEMORY.md`) applied to capability state: two
tables that must agree, with no mechanism forcing them to.

THE RULING THIS ENCODES (owner decision, 2026-09-22)
----------------------------------------------------
    Capability authority  ->  the REGISTRY
    Public health source  ->  this ADAPTER, and only this adapter
    H-1/H-2               ->  fix or retire as a public API path

So: one adapter, registry-authoritative, and it is the single public source for
`/v1/health` and `/v1/capabilities`. `AnalysisController.health()` is no longer
a public API path (see its docstring); the registry's own vocabulary never
crosses this boundary.

THE TWO VOCABULARIES, AND WHY THE TRANSLATION IS NOT COSMETIC
-------------------------------------------------------------
    registry   `RegistryState`  ∈ {available, degraded, unavailable}
    contract   (`API_CONTRACT.md` section 2.3)
                                ∈ {loaded, absent, unavailable,
                                   not_requested, evicted}

The word `unavailable` appears in BOTH, and it means something different in
each:

    registry   construction raised -- "no builder could be constructed"
    contract   present but broken -- "this build ships it and it is broken"

Those must not be conflated: absent means "this build does not ship it";
unavailable means "this build ships it and it is broken, which is a defect".
A naive pass-through of the registry's three states would silently relabel every
*missing artifact* as a *defect*, and the frontend is instructed
(`API_CONTRACT.md` section 2.3) to treat the two very differently. `degraded` is
worse still: it is also a whole-service `HealthStatus.status` value, so the bare
word is ambiguous between "this capability is running on fallbacks" and "the
service as a whole is degraded".

THE TRANSLATION TABLE
---------------------
    RegistryState.AVAILABLE    -> "loaded"        present and constructible
    RegistryState.DEGRADED     -> "loaded"        present, runs on fallbacks;
                                                  the reason is carried in
                                                  `capabilities[].reason`, NOT
                                                  by inventing a state the
                                                  contract does not define
    RegistryState.UNAVAILABLE  -> "absent"        when the artifacts are simply
                                                  not in this deployment
                               -> "unavailable"   when they are present but
                                                  could not be constructed --
                                                  i.e. a defect
    never constructed at all   -> "not_requested" the honest state under
                                                  `lazy_load: true`

`evicted` is **produced by the running model cache**, not by this adapter: it is
a statement about what has happened in this process's lifetime, and no static
inspection of the filesystem can observe it. This adapter therefore never emits
it, and says so rather than guessing.

WHY THE ADAPTER DOES NOT CONSTRUCT ANYTHING
-------------------------------------------
Requirement 4 of `docs/DEPLOYMENT_ARCHITECTURE.md` section 3.3: *"Never load a
model for a metadata request."* A ZeroGPU cold start that loads every model just
to answer `/v1/health` spends the 5 GPU-minute daily budget on health checks.

So this adapter reads the registry's **spec table** and the **filesystem**, and
never calls `build()` / `build_all()`.

TWO KINDS OF UNAVAILABILITY, AND WHY THEY MUST NOT BE COLLAPSED
---------------------------------------------------------------
Measuring this deployment's constructed registry states produced a result that
forced a design decision:

    caption      unavailable  "transformers is not installed: No module named 'transformers'"
    change       available    None
    change_vqa   unavailable  "sentence-transformers is required ... No module named 'sentence_transformers'"
    grounding    unavailable  "RemoteCLIP encoder could not be loaded: No module named 'huggingface_hub'"
    optical_sar  degraded     "no trained head; running on fallback"
    vqa          unavailable  "transformers is not installed: ..."

Four of those six are caused by **a missing Python package in this sandbox's
venv**. The deployment installs those packages (`requirements.txt`); a probe run
from a developer's narrow venv does not have them. That is an *environment*
fact, and it is not a fact about the deployment. `optical_sar`'s `degraded`, by
contrast, is a genuine deployment fact: it names a **missing trained artifact**.

Collapsing the two would produce a health payload that says "this deployment
cannot do VQA" when the truth is "the process that ran the probe had no
`transformers` installed". So the adapter separates:

    ARTIFACT availability   what this DEPLOYMENT ships -- the served answer
    BUILD availability      whether THIS PROCESS could construct it -- the
                            operator's answer, obtainable only by constructing

The served payload is built from artifact presence alone. `construct_failures=`
exists for a caller that *has* already paid for construction (the controller's
diagnostic path) and wants the defect reported; the metadata path never passes
it. This is why the served `models` map reads `not_requested` / `absent`, and
never claims to reflect a build that has not happened.

DEGRADATION IS A RUNTIME STATE, NOT A PHASE COMPLETION
------------------------------------------------------
A degraded `optical_sar` is represented as `available: false` with a reason, which
is accurate and does not claim that Phase 12 completed.

CORRECTION (2026-09-25): this paragraph previously asserted that `optical_sar`
resolves `degraded` *because the CROMA checkpoint and the fusion head are absent*.
**That attribution was wrong.** Both artifacts exist and are wired:

  * `artifacts/optical_sar/fusion_head_production_v001/head.pt` — 14,427,457 bytes,
    sha256 `785815729a3a39fc34dc41894efaf00d8739365d970a3f830a326e68ae888dab`,
    the Phase-12 production head designated by ruling R-14 (2026-09-21).
  * `CROMA_base.pt` — resolved by `croma.resolve_checkpoint_path` from the pinned
    Hub identity (`antofuller/CROMA`, revision `0dd28e3d633b`), offline first.

`app/serving.py::_wired_optical_sar_builder` injects both through the hash-exempt
channel, so `available` now depends on whether the artifacts are reachable *in the
running deployment* — not on whether they were ever produced. On a host whose Hub
cache lacks the pinned revision (and with no `SATQUERY_CROMA_CHECKPOINT` set), the
resolver returns `None`, the encoder is not built, and the DEGRADED contract below
still governs. That is a deployment fact and is reported as one.

CORRECTION 2 (2026-09-25): `_missing_shipped` used to read
`Config.get("croma.checkpoint_path")` -- a key that is DELIBERATELY never written
to `configs/base.yaml` -- so it reported `absent` on every host regardless of the
Hub cache, and disagreed with the loader above *by construction*. It now resolves
through the SAME `croma.resolve_checkpoint_path` the serving path uses, via
`_optical_sar_artifacts`. See that function for the reporter/loader divergence.

The ruling's actual prohibition — **do not retrain CROMA** (plan section 73) — is
unaffected and still stands. Nothing here retrains anything; the wiring only points
at the existing, hashed artifact.

So the honest representation remains a runtime state, and Phase 12 completion is
still a separate question this adapter does not answer either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "CapabilityReport",
    "CONTRACT_STATES",
    "DeploymentReport",
    "REGISTRY_TO_CONTRACT",
    "deployment_report",
    "health_payload",
    "capabilities_payload",
]


#: The contract's closed vocabulary (`API_CONTRACT.md` section 2.3). Listed as
#: data so a test can assert the adapter never emits anything outside it -- the
#: whole point of an adapter is that its output range is checked.
CONTRACT_STATES: frozenset[str] = frozenset(
    {"loaded", "absent", "unavailable", "not_requested", "evicted"}
)

#: Registry state -> contract state, for the registry's *own* words.
#:
#: `UNAVAILABLE` is deliberately mapped to `"absent"` here and refined to
#: `"unavailable"` only when the artifacts are demonstrably present. Mapping it
#: straight to `"unavailable"` would be the single most damaging mistranslation
#: available in this module: it would label every missing artifact a defect.
REGISTRY_TO_CONTRACT: Mapping[str, str] = {
    "available": "loaded",
    "degraded": "loaded",
    "unavailable": "absent",
}


# ---------------------------------------------------------------------------
# What a capability needs, and where it lives
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CapabilityRequirement:
    """One capability's declared artifact requirement.

    Read from the registry's spec table and `core.planner.CAPABILITY_ASSETS`
    (the two authoritative sources for asset counts, which
    `tests/unit/test_planner.py` already asserts agree). This dataclass carries
    **no new numbers** -- it is a view, not a third copy.
    """

    capability: str
    requires_assets: int | None
    artifacts: tuple[Path, ...]


#: What each capability needs, declared as (kind, config_path) pairs.
#:
#: WHY THIS IS A TABLE AND NOT DERIVED FROM `optional_config_keys` ALONE
#: --------------------------------------------------------------------
#: The registry's spec rows list every artifact config key a builder accepts, but
#: a spec row cannot distinguish a *required* artifact from an *optional* one --
#: `build_grounding_specialist` treats a missing `head_path` as a graceful
#: fallback to zero-shot, while `build_optical_sar_specialist` treats a missing
#: CROMA checkpoint as "sensor-only, and no prediction" (`specialists/
#: optical_sar/specialist.py:900-912`). Both are `optional_config_keys` in the
#: registry's terms, because "optional" there means "the registry may omit the
#: kwarg", not "the capability survives without it".
#:
#: So the requirement lives here, keyed by the SAME config paths the registry
#: uses -- not by a second set of hard-coded filesystem paths. If a spec row
#: renames a config key, this table must be updated with it, and
#: `test_the_adapter_artifact_table_matches_the_registry` fails until it is.
#:
#: KIND is what the requirement means when unset or missing:
#:   "shipped"   the artifact must be on disk in this deployment. An unset path
#:               or a missing file is `absent` -- a permanent statement.
#:   "hub"       the model is fetched from the Hugging Face Hub. An unset path
#:               is `not_requested` with a Hub-named reason, never `absent`.
_REQUIREMENTS: Mapping[str, tuple[tuple[str, str], ...]] = {
    # `change` is wired from the composition root, not config; see
    # `_requirement_artifacts`.
    "change": (),
    # Optional head, but a REFUSAL when the detector is missing. The detector is
    # `change.checkpoint_path`, which is unset in base.yaml and supplied at the
    # call site; absence is handled by the composition-root branch.
    "change_vqa": (),
    # RemoteCLIP is Hub-fetched; `grounding.checkpoint_path` is a local override
    # and `grounding_head.head_path` is a genuinely optional head.
    "grounding": (("hub", "grounding.checkpoint_path"),),
    # CROMA is a 777.6 MB checkpoint with no Hub download path in this codebase
    # (`specialists/optical_sar/specialist.py:52`), and without it the specialist
    # answers nothing. So an unset `croma.checkpoint_path` genuinely means this
    # deployment cannot fuse.
    #
    # CORRECTION (2026-09-25): the parenthetical above is stale. CROMA DOES have a
    # Hub path — `croma.resolve_checkpoint_path` step 3 resolves the pinned
    # (repo, file, revision) triple through the Hub cache, offline first — and
    # `app/serving.py::_wired_optical_sar_builder` supplies the result, so
    # `croma.checkpoint_path` in `configs/base.yaml` is deliberately NEVER set
    # (setting it would move `Config.hash`, which `scripts/eval_change.py` treats
    # as unfitness to score, exit 3).
    #
    # The two `shipped` entries below are NOT read as config keys. Both are
    # resolved by `_optical_sar_artifacts` through the SAME channel serving uses:
    # `fusion.head_path` from the `app.serving.FUSION_HEAD` constant, and
    # `croma.checkpoint_path` from `croma.resolve_checkpoint_path`. The labels are
    # kept because `_MISSING_REASONS` is keyed by them, and because they are the
    # config paths the registry spec row names — so this table stays a view of the
    # registry rather than a fourth copy (see
    # `test_every_declared_config_path_exists_in_the_registry_spec_table`).
    "optical_sar": (
        ("shipped", "croma.checkpoint_path"),
        ("shipped", "fusion.head_path"),
    ),
    # SmolVLM is Hub-fetched. `vqa` and `caption` are two capabilities of ONE
    # specialist (`specialists/vqa/inference.py`: `name="vlm"`,
    # `capabilities=("vqa","caption")`), and its builder reads the `vlm.*` block
    # from the config itself -- so the registry spec row carries NO
    # `optional_config_keys` at all (`core/registry.py:174-183` says why: a
    # `config_keys` entry would pass an unsupported kwarg and fail construction).
    #
    # An earlier version of this table declared `vlm.model_id`, which is not a
    # key the registry uses. `test_every_declared_config_path_exists_in_the_
    # registry_spec_table` failed and this line was corrected. The empty tuple is
    # the honest declaration: this adapter cannot name a local artifact for these
    # capabilities, so it reports `not_requested` and says the model is fetched
    # remotely (`_HUB_REASONS`) rather than inventing a key to check.
    "vqa": (),
    "caption": (),
}

#: Reasons for a `shipped` requirement that is not satisfied. Keyed by
#: capability then config path, so the sentence names the exact missing artifact.
#:
#: `optical_sar` has two, and the distinction is load-bearing: without CROMA the
#: specialist is sensor-only, and without the fusion head it produces no
#: prediction at all. Reporting the same sentence for both would hide which one
#: an operator needs to supply.
_MISSING_REASONS: Mapping[str, Mapping[str, str]] = {
    "optical_sar": {
        "croma.checkpoint_path": (
            "the CROMA backbone checkpoint (CROMA_base.pt) is not present in "
            "this deployment; without it optical/SAR fusion degrades to "
            "sensor-only"
        ),
        "fusion.head_path": (
            "the trained optical/SAR fusion head is not present in this "
            "deployment; without it no fused prediction is produced"
        ),
    },
    "change": {
        "change.checkpoint_path": (
            "the trained change checkpoint is not present in this deployment"
        ),
    },
    "change_vqa": {
        "change.checkpoint_path": (
            "the change detector backing the change-VQA head is absent"
        ),
    },
}

#: Reasons for a `hub` requirement when no local path is configured.
_HUB_REASONS: Mapping[str, str] = {
    "grounding": (
        "the RemoteCLIP encoder is fetched from the Hugging Face Hub on first "
        "use and no local checkpoint_path is configured in this deployment"
    ),
    "vqa": (
        "the SmolVLM weights are fetched from the Hugging Face Hub on first use "
        "and no local checkpoint_path is configured in this deployment"
    ),
    "caption": (
        "the SmolVLM weights are fetched from the Hugging Face Hub on first use "
        "and no local checkpoint_path is configured in this deployment"
    ),
}


def _optical_sar_artifacts() -> tuple[tuple[str, Path | None], ...]:
    """The `optical_sar` artifacts, each resolved the way SERVING resolves it.

    WHY THIS FUNCTION EXISTS -- THE REPORTER/LOADER DIVERGENCE
    ---------------------------------------------------------
    Before this function, `optical_sar` was checked by `_configured_path` alone,
    which reads `Config.get("croma.checkpoint_path")`. That key is DELIBERATELY
    never written to `configs/base.yaml` -- writing it would move `Config.hash`
    off `78f1e3700da15aa1`, and `scripts/eval_change.py` treats a moved hash as
    "unfit to score" (exit 3). So the check returned `None` on EVERY host,
    warm or cold, restarted or not, and `optical_sar` could never be reported
    available -- even with the pinned checkpoint sitting in the Hub cache.

    Meanwhile `app/serving.py::_wired_optical_sar_builder` (line ~192) resolved
    the SAME artifact through `croma.resolve_checkpoint_path`, which is
    Hub-cache aware: env -> config -> pinned (repo, file, revision) triple,
    offline first. The two disagreed by construction, and the disagreement
    surfaced as "the artifact is missing" -- a false negative that sent an
    operator looking for a file that was right there.

    The fix is to make the reporter ask the LOADER's question. This is the same
    special-case shape `_requirement_artifacts` already applies to `change` and
    `change_vqa`, whose artifacts are also composition-root wired rather than
    config keys -- `optical_sar` was simply the one capability that needed the
    treatment and never got it.

    Returns `(config_path_label, resolved_path_or_None)` pairs, so the caller
    can both report presence AND name the requirement in `_MISSING_REASONS`'
    vocabulary. The labels are the SAME config-path strings the registry spec
    row uses (`core/registry.py:254-257`), so the requirement table stays a view
    of the registry rather than a fourth copy.

    RESOLUTION ORDER, mirroring serving exactly
    -------------------------------------------
    1. The fusion head from `app.serving.FUSION_HEAD` -- a composition-root
       CONSTANT, never a config key. `grep -rn '"fusion.head_path"'` finds it
       written in exactly one place in the tree (the requirement table itself),
       so a config lookup could never succeed here either.
    2. CROMA from `croma.resolve_checkpoint_path` -- the resolver whose answer
       the serving path actually uses.

    WHY THE IMPORTS ARE INSIDE THE FUNCTION
    ---------------------------------------
    `specialists.optical_sar.croma` imports NumPy at module level
    (`croma.py:52`), and `croma.resolve_checkpoint_path` lazily imports
    `huggingface_hub`. This adapter is on the METADATA path, where
    requirement 1 of `DEPLOYMENT_ARCHITECTURE.md` section 3.3 says a metadata
    request must not pay the model-stack import cost, and
    `test_the_metadata_path_does_not_import_torch` enforces the torch half in a
    subprocess. Deferring the imports keeps that property: a deployment with no
    NumPy still serves `/v1/capabilities`, reporting `optical_sar` absent --
    which is TRUE there.

    `app.serving` is safe to import at function scope: `app/__init__.py` already
    re-exports `FUSION_HEAD` from it and it pulls no torch.

    DEGRADING VERSUS RAISING
    ------------------------
    `resolve_checkpoint_path` RAISES `ModelLoadError` for an explicitly
    specified path that does not exist (env or config), because a typo'd
    operator path is a misconfiguration rather than an absent artifact. That
    distinction is preserved: `_resolve_croma_checkpoint` catches only
    `ImportError`, so a bad operator path propagates while a missing optional
    dependency becomes `None`. See that function for the full reasoning.
    """
    from app.serving import FUSION_HEAD

    checkpoint = _resolve_croma_checkpoint()
    return (
        ("croma.checkpoint_path", Path(checkpoint) if checkpoint else None),
        ("fusion.head_path", FUSION_HEAD if _path_present(FUSION_HEAD) else None),
    )


def _resolve_croma_checkpoint() -> str | None:
    """The CROMA checkpoint serving would use, or None when unresolvable here.

    Delegates to `croma.resolve_checkpoint_path` -- the SAME resolver
    `app/serving.py` uses, which is the entire point of this function. See
    `_optical_sar_artifacts` for the divergence it closes.

    TWO DEGRADATIONS, DELIBERATELY DIFFERENT
    ----------------------------------------
    `ImportError` (and only that class) is caught and mapped to `None`: a
    metadata route on a host without NumPy or `huggingface_hub` cannot resolve
    the checkpoint, and "cannot resolve here" IS an absence. This is the case
    the docstring's requirement-1 reasoning is about, and it must not turn
    `/v1/capabilities` into a 500.

    `ModelLoadError` is NOT caught. `resolve_checkpoint_path` raises it when an
    EXPLICITLY specified path (env `SATQUERY_CROMA_CHECKPOINT` or
    `croma.checkpoint_path`) does not exist. That is a misconfiguration, not an
    absence, and silently reporting `absent` would send the operator hunting for
    a missing file when the real fault is a bad path. Letting it propagate keeps
    the two facts distinct -- the same reasoning `build_encoder`'s contract
    already uses.
    """
    try:
        from specialists.optical_sar.croma import resolve_checkpoint_path
    except ImportError:
        # Optional dependency absent on this host: NumPy (module level,
        # `croma.py:52`) or a transitively unavailable import. "Cannot resolve
        # here" is an absence; a metadata route must not 500 over it.
        return None

    checkpoint, _source = resolve_checkpoint_path(_maybe_config())
    return checkpoint


def _maybe_config() -> Any | None:
    """The process config, or None when it cannot be read.

    `resolve_checkpoint_path` treats `None` as "no config, skip steps 1-2 and
    report absent", which is the right degradation for a metadata route: an
    unreadable config is reported upstream by whatever needed it, and a
    capability probe must not 500 because of it.
    """
    try:
        from core.config import get_config

        return get_config()
    except Exception:  # noqa: BLE001 - an unreadable config is "absent" here
        return None


def _requirement_artifacts(capability: str) -> tuple[Path, ...]:
    """The `shipped` artifacts this capability needs, as resolved paths.

    Only `shipped` requirements appear here. A `hub` requirement is by
    definition not a local path, so requiring it on disk would report a
    perfectly capable deployment as `absent` -- the false negative that
    `_HUB_REASONS` exists to avoid.

    `change` and `change_vqa` are special-cased: their artifacts are wired by
    the serving composition root, NOT by config (`app/serving.py:16-28`, because
    a new config key would move `Config.hash`). The composition root's exported
    constants are authoritative there, and `CHANGE_VQA_HEAD` is paired with
    `CHANGE_CHECKPOINT` because the head was fitted on that detector's features
    (finding F2) -- a change answer resting on an untrained detector would be a
    silent correctness failure.

    `optical_sar` is special-cased for the SAME reason, and shares its
    resolution with serving through `_optical_sar_artifacts` -- see that
    function for the reporter/loader divergence this closes.
    """
    if capability == "change":
        from app.serving import CHANGE_CHECKPOINT

        return (CHANGE_CHECKPOINT,)
    if capability == "change_vqa":
        from app.serving import CHANGE_CHECKPOINT, CHANGE_VQA_HEAD

        return (CHANGE_VQA_HEAD, CHANGE_CHECKPOINT)
    if capability == "optical_sar":
        return tuple(p for _label, p in _optical_sar_artifacts() if p is not None)

    paths: list[Path] = []
    for kind, config_path in _REQUIREMENTS.get(capability, ()):
        if kind != "shipped":
            continue
        resolved = _configured_path(config_path)
        if resolved is not None:
            paths.append(resolved)
    return tuple(paths)


def _missing_shipped(capability: str) -> tuple[str, ...]:
    """`shipped` requirements that this deployment does NOT satisfy.

    A `shipped` requirement is unsatisfied when it cannot be resolved OR when
    the file it names does not exist. Both are permanent for this revision, which
    is what the contract's `absent` means. This is the check that stops the
    adapter reporting `optical_sar` as available on a machine with no CROMA
    checkpoint -- the false positive that would tell a frontend to enable a
    button whose handler can only refuse.

    `optical_sar` resolves through `_optical_sar_artifacts`, i.e. the same
    Hub-cache-aware resolver `app/serving.py` uses. Reading a config key here
    instead would report `absent` on every host regardless of the cache, which
    is the divergence this function used to have. Note the check is on the
    RESOLVED path, so a warm cache means "satisfied" and a cold one means
    "absent" -- the redeployment state is what is reported, not a static
    property of the source tree.
    """
    if capability == "optical_sar":
        return tuple(
            label for label, path in _optical_sar_artifacts() if path is None
        )

    missing: list[str] = []
    for kind, config_path in _REQUIREMENTS.get(capability, ()):
        if kind != "shipped":
            continue
        resolved = _configured_path(config_path)
        if resolved is None:
            missing.append(config_path)
        elif not _path_present(resolved):
            missing.append(config_path)
    return tuple(missing)


def _path_present(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _hub_unconfigured(capability: str) -> bool:
    """Whether this capability's weights are remote and nothing local is set.

    True means "the model is fetched remotely and this deployment names no local
    copy", which is reported as `not_requested` with a Hub-named reason.

    TWO WAYS TO BE TRUE, and both occur in the real spec table:

      * the row declares at least one `hub` requirement and no `shipped` one, and
        no path is configured (`grounding`);
      * the row declares NO requirements at all (`vqa`, `caption` -- their
        builder reads the `vlm.*` block itself, so the registry spec row carries
        no `optional_config_keys`, see `_REQUIREMENTS`).

    The second case is why this function cannot simply test "every requirement is
    an unconfigured hub". An empty requirement set is vacuously all-`hub`, and an
    earlier version of this logic required `any(kind == "hub")` -- which would
    have sent `vqa` down the "everything is on disk" branch and reported a
    Hub-fetched model as locally present. The `_HUB_BACKED` set is what
    discriminates: a capability with no requirements and no Hub association
    would be a genuine gap in the table, and `test_every_registry_capability_
    has_a_requirement_row` plus this membership check catch it.
    """
    if capability not in _HUB_BACKED:
        return False
    requirements = _REQUIREMENTS.get(capability, ())
    if any(kind == "shipped" for kind, _ in requirements):
        return False
    return all(_configured_path(path) is None for _, path in requirements)


def _configured_path(config_path: str) -> Path | None:
    """Resolve one artifact path from config, or None when unset.

    `None` means "the config does not name a path", which is a real and common
    state: every artifact path in `configs/base.yaml` is deliberately unset so
    that the trained artifacts can be wired at the call site without moving
    `Config.hash`. A caller that wants presence-based reporting must therefore
    also consult :data:`_HUB_BACKED`.
    """
    try:
        from core.config import get_config

        value = get_config().get(config_path)
    except Exception:  # noqa: BLE001 - an unreadable config is reported upstream
        return None
    if not value:
        return None
    return Path(str(value))


#: Capabilities whose model is fetched from the Hugging Face Hub rather than
#: shipped as a local artifact. Retained as a declaration of *which* capabilities
#: can legitimately be unavailable-remote; the per-capability reason sentence
#: lives in `_HUB_REASONS`, and the decision of whether a capability is
#: Hub-backed at all is derived from `_REQUIREMENTS`.
_HUB_BACKED: frozenset[str] = frozenset({"grounding", "vqa", "caption"})


#: The registry's declared capability set is authoritative. This is the
#: *shape* of the served list for capabilities the adapter knows about; it is
#: not a capability list, and it must never be used as one. Everything below is
#: derived from `registry.available()`.
@dataclass(frozen=True)
class CapabilityReport:
    """One row of `GET /v1/capabilities`, plus the state for `GET /v1/health`.

    `state` is the CONTRACT state; `registry_state` is the registry's own word,
    retained for the operator-facing `detail` and for tests. Exposing
    `registry_state` in the *served* payload would leak the internal vocabulary
    the ruling keeps internal -- so it appears only in `as_internal()`.
    """

    capability: str
    state: str
    registry_state: str
    available: bool
    reason: str | None
    requires_pair: bool
    max_assets: int
    artifacts: tuple[dict[str, Any], ...]
    detail: str | None = None

    def as_capability(self) -> dict[str, Any]:
        """The `GET /v1/capabilities` entry.

        Field names and types follow `API_CONTRACT.md` section 2.2 exactly. The
        contract lists `modalities`; it is omitted here rather than guessed --
        see :func:`_modalities_for`.
        """
        entry: dict[str, Any] = {
            "task": self.capability,
            "available": self.available,
            "reason": self.reason,
            "requires_pair": self.requires_pair,
            "max_assets": self.max_assets,
        }
        modalities = _modalities_for(self.capability)
        if modalities is not None:
            entry["modalities"] = modalities
        return entry

    def as_internal(self) -> dict[str, Any]:
        """Registry vocabulary and artifact evidence. NOT part of the contract.

        This is the operator's view: it is what makes a degradation diagnosable.
        It is deliberately never served by an HTTP route, because the ruling
        keeps the registry's vocabulary internal.
        """
        return {
            "capability": self.capability,
            "registry_state": self.registry_state,
            "contract_state": self.state,
            "reason": self.reason,
            "artifacts": list(self.artifacts),
            "detail": self.detail,
        }


#: Modalities per capability, or None when the capability's modality is not a
#: fixed property of the capability but of the request's assets.
#:
#: `vqa`/`caption`/`grounding` accept any single raster, and `change`/`change_vqa`
#: accept two acquisitions that may be optical, SAR or mixed. Only `optical_sar`
#: has a modality it is *defined* by. Reporting a guess for the rest would be an
#: invented contract field -- worse than omitting an optional one.
_MODALITIES: Mapping[str, tuple[str, ...] | None] = {
    "optical_sar": ("optical", "sar"),
}


def _modalities_for(capability: str) -> list[str] | None:
    found = _MODALITIES.get(capability)
    return list(found) if found is not None else None


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DeploymentReport:
    """Everything both routes need, derived once from the registry."""

    config_hash: str
    device: str | None
    platform: str | None
    zerogpu: Any
    lazy_load: Any
    cache_max_models: Any
    torch_compile: Any
    gpu_durations: Mapping[str, int]
    capabilities: tuple[CapabilityReport, ...]
    device_is_cuda: bool

    # -- derived views -----------------------------------------------------

    @property
    def models(self) -> dict[str, str]:
        """The `HealthStatus.models` map: capability -> CONTRACT state."""
        return {cap.capability: cap.state for cap in self.capabilities}

    @property
    def status(self) -> str:
        """`HealthStatus.status`, derived rather than asserted.

        The rule the contract states (`API_CONTRACT.md` section 2.1):
        `degraded` = *"the service is up but at least one expected artifact is
        absent"*. Two things follow, and both matter:

          * a capability that is merely `not_requested` does **not** degrade the
            service -- under `lazy_load: true` that is every capability on a
            fresh process, and reporting `degraded` for a healthy idle service
            would make the field useless as a probe;
          * `unavailable` (present but broken) is a defect. It is reported as
            `error`, not `degraded`, because the two have different remedies and
            `API_CONTRACT.md` section 5.2 already renders `model_load_error` as
            *"Defect -- surface it"*.
        """
        states = {cap.state for cap in self.capabilities}
        if "unavailable" in states:
            return "error"
        if "absent" in states:
            return "degraded"
        return "ok"

    @property
    def gpu_available(self) -> bool:
        """Whether a CUDA device was *detected*, not whether one is allocated.

        `API_CONTRACT.md` section 2.1: *"`false` is normal on ZeroGPU Spaces
        until a request is executing"*, and the frontend is told not to surface
        it as a fault. So this is a device probe, and it must not be read as
        "the GPU works right now".

        DELIBERATELY DOES NOT IMPORT TORCH, and that is a correction.

        The first version probed `torch.cuda.is_available()`, which meant the
        health payload pulled the entire torch stack on every request. That
        breaks requirement 1 of `docs/DEPLOYMENT_ARCHITECTURE.md` section 3.3 --
        *"Import cheaply and without torch"* -- and it made
        `test_the_metadata_path_does_not_import_torch` fail, which is how it was
        found. Importing torch to answer a liveness probe on a ZeroGPU Space
        would also pay that cost inside the metered cold start.

        So the probe is environmental instead. It is also the MORE accurate
        answer for this field: what the contract asks is whether a GPU device is
        *present*, which on a ZeroGPU Space is a property of the platform and
        shows up as `CUDA_VISIBLE_DEVICES` / `SATQUERY_DEVICE`, not as a
        torch-visible device before any allocation exists.

        `SATQUERY_DEVICE=cuda` is honoured as an explicit operator assertion that
        a GPU exists, because that variable is the deployment's own declaration
        (`core/config.py:87-91` reads it first).
        """
        import os

        if os.environ.get("SATQUERY_DEVICE", "").strip().lower() == "cuda":
            return True
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible and visible != "-1":
            return True
        # `nvidia-smi` is not probed: shelling out on a health route would be a
        # per-request subprocess, and a Space's GPU is invisible until ZeroGPU
        # allocates it, so a negative answer from here would be misleading.
        return False

    @property
    def all_available(self) -> bool:
        return all(cap.available for cap in self.capabilities)

    # -- payloads ----------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """The `GET /v1/health` payload. Exactly `HealthStatus`'s field set.

        `HealthStatus` is `extra="forbid"` (`core/schemas.py:392`), so the field
        set here is not a preference -- an extra key makes the payload invalid
        against the project's own model. That is finding H-1, and a test asserts
        `HealthStatus.model_validate(self.health())` succeeds.
        """
        return {
            "status": self.status,
            "schema_version": _schema_version(),
            "models": self.models,
            "device": self.device,
            "gpu_available": self.gpu_available,
        }

    def capabilities_block(self) -> dict[str, Any]:
        return {
            "schema_version": _schema_version(),
            "capabilities": [cap.as_capability() for cap in self.capabilities],
            "deployment": {
                "platform": self.platform,
                "zerogpu": self.zerogpu,
                "lazy_load": self.lazy_load,
                "cache_max_models": self.cache_max_models,
                "torch_compile": self.torch_compile,
            },
        }

    def as_internal(self) -> dict[str, Any]:
        """The operator view. Never served over HTTP."""
        return {
            "schema_version": _schema_version(),
            "config_hash": self.config_hash,
            "status": self.status,
            "device": self.device,
            "device_is_cuda": self.device_is_cuda,
            "deployment": self.capabilities_block()["deployment"],
            "gpu_durations": dict(self.gpu_durations),
            "capabilities": [cap.as_internal() for cap in self.capabilities],
            "note": (
                "Capability state is derived from the registry's declared spec "
                "table and from artifact PRESENCE on the filesystem; no model is "
                "loaded, so a metadata request costs no GPU quota. "
                "`not_requested` means nothing has attempted the load yet: this "
                "adapter cannot distinguish 'never asked' from 'asked and "
                "evicted' without constructing, and `evicted` is therefore never "
                "emitted here. `absent` (not shipped) and `unavailable` (shipped "
                "but broken) are kept distinct; the latter is a defect."
            ),
        }


def _schema_version() -> str:
    from core.schemas import SCHEMA_VERSION

    return SCHEMA_VERSION


# ---------------------------------------------------------------------------
# The adapter entry point
# ---------------------------------------------------------------------------
def _registry_capabilities() -> tuple[str, ...]:
    """The authoritative capability set, read WITHOUT importing torch.

    The capability SET is a static property of the spec table, so the right way
    to read it is `default_specs()` -- a table of strings. It is tempting to call
    `build_serving_registry().available()` instead, and the first version of this
    function did exactly that. It works, but it drags torch in:

        app.serving.build_serving_registry()
          -> SpecialistRegistry.__init__
            -> `device or config.device_preference`
              -> core.config._torch_cuda_available()
                -> `import torch`            # core/config.py:239

    `Config.device_preference` probes CUDA by importing torch
    (`core/config.py:87-91`), so ANY code path that instantiates a registry
    imports torch. On the metadata path that violates requirement 1 of
    `docs/DEPLOYMENT_ARCHITECTURE.md` section 3.3 -- *"Import cheaply and without
    torch"* -- and it was caught by
    `test_the_metadata_path_does_not_import_torch`.

    There is no behaviour lost by avoiding the registry here. `available()`
    returns `tuple(sorted(self._specs))`, and `_specs` is populated from exactly
    `default_specs()` (`core/registry.py:328-329`). So this returns the same
    answer without the import -- and a test asserts the two agree, so the
    shortcut cannot drift from the real registry.
    """
    from core.registry import default_specs

    return tuple(sorted(spec.name for spec in default_specs()))


def _asset_count(capability: str) -> int:
    """Asset count from the planner's authoritative table.

    `core.planner.CAPABILITY_ASSETS` and `SpecialistSpec.requires_assets` are the
    two sources `MEMORY.md` names as authoritative and which must agree; the
    planner's table is the one the *plan* is built from, so it is the one this
    adapter reads. Adding a third copy here is precisely the hazard
    `_indices_for`'s docstring warns about.
    """
    try:
        from core.planner import CAPABILITY_ASSETS

        return int(CAPABILITY_ASSETS[capability])
    except Exception:  # noqa: BLE001 - fall back to the registry's own number
        from core.registry import default_specs

        for spec in default_specs():
            if spec.name == capability and spec.requires_assets is not None:
                return int(spec.requires_assets)
        return 1


def _artifact_evidence(paths: Iterable[Path]) -> tuple[tuple[dict[str, Any], ...], list[str]]:
    """Inspect artifact presence. Returns `(evidence, absent_paths)`.

    Presence only -- `stat().st_size` is read, never a byte of the file. No
    `hashlib`, no `torch.load`: hashing a 63 MB checkpoint on a health route
    would be a self-inflicted timeout, and hashing is not this adapter's job
    (`scripts/verify_artifacts.py` owns that).
    """
    evidence: list[dict[str, Any]] = []
    absent: list[str] = []
    for path in paths:
        try:
            present = path.exists()
            size = path.stat().st_size if present else None
        except OSError:
            # An unreadable path is not a present artifact. Reported as absent
            # rather than raising: a metadata route must not 500 because a
            # mount is misbehaving.
            present, size = False, None
        evidence.append({"name": path.name, "present": present, "bytes": size})
        if not present:
            absent.append(path.name)
    return tuple(evidence), absent


def _report_for(
    capability: str,
    *,
    declared: bool,
    construct_failure: str | None = None,
) -> CapabilityReport:
    """Translate ONE capability. The whole adapter is this function, looped.

    `declared` says whether this capability is one the REGISTRY declares, as
    opposed to one a caller passed in. This distinction is load-bearing and was
    initially missing: a capability the registry does not know about, with no
    artifact requirement, fell through to the "everything is on disk" branch and
    was reported `available: true`. That is the failure this module exists to
    prevent -- claiming a deployment can serve something it cannot -- so an
    undeclared capability is now refused before the artifact logic runs.

    `construct_failure` is supplied by a caller that has already attempted
    construction (the controller's diagnostic path). The metadata path never
    passes it, because it never constructs.
    """
    if not declared:
        # Not in the registry -> not a capability. Reason is explicit so a
        # client reading `/v1/capabilities` is told why the entry exists at all
        # (only reachable when a caller passes an explicit capability list).
        return CapabilityReport(
            capability=capability,
            state="absent",
            registry_state="unavailable",
            available=False,
            reason=(
                "the registry does not declare this capability in this "
                "deployment"
            ),
            requires_pair=False,
            max_assets=_asset_count(capability),
            artifacts=(),
            detail=None,
        )

    paths = _requirement_artifacts(capability)
    evidence, absent_files = _artifact_evidence(paths)
    requires_assets = _asset_count(capability)

    # `shipped` requirements this deployment does not satisfy: an unset config
    # path, or a path naming a file that is not there. Both are permanent for
    # this revision, which is what the contract's `absent` means.
    #
    # `absent_files` covers the composition-root-wired capabilities (change,
    # change_vqa): their paths come from `app.serving` rather than from config,
    # so they never appear in `_REQUIREMENTS` and `_missing_shipped` cannot see
    # them. Both routes feed the same `absent` verdict.
    missing = _missing_shipped(capability) or absent_files

    reason: str | None = None

    if construct_failure is not None:
        # A caller that HAS attempted construction reports the defect here.
        # `unavailable` = present but could not be loaded: a defect, which the
        # contract keeps distinct from `absent`.
        state = "unavailable"
        reason = f"could not be loaded in this process: {construct_failure}"
    elif missing:
        # THE DEPLOYMENT'S OWN ANSWER: an artifact it requires is not here.
        # The sentence names the specific artifact, because "optical_sar cannot
        # fuse" and "optical_sar has no fusion head" need different remedies.
        reasons = _MISSING_REASONS.get(capability, {})
        named = [reasons[p] for p in missing if p in reasons]
        if named:
            reason = "; ".join(named)
        else:
            reason = (
                f"required artifact(s) not present in this deployment: "
                f"{', '.join(missing)}"
            )
        state = "absent"
    elif _hub_unconfigured(capability):
        # No local artifact is declared AND no path is configured, and the
        # capability is Hub-backed. NOT `absent` -- that would claim a permanent
        # absence for a capability the deployment can serve with network and a
        # warm cache. See `_HUB_REASONS`.
        state = "not_requested"
        reason = _HUB_REASONS.get(capability)
    else:
        # Every shipped artifact the capability declares is on disk. Whether it
        # has actually been constructed is NOT observable without constructing,
        # and requirement 4 forbids constructing. `not_requested` is the honest
        # answer under `lazy_load: true`.
        state = "not_requested"

    # `available` means "this deployment can serve this capability": no missing
    # shipped artifact, and no construction defect reported. A `not_requested`
    # capability is available-but-idle, which is the normal state of a lazy
    # deployment, and the reason field explains the Hub dependency.
    available = state == "not_requested" and construct_failure is None
    return CapabilityReport(
        capability=capability,
        state=state,
        registry_state=(
            "available" if available else
            "degraded" if state == "loaded" else
            "unavailable"
        ),
        available=available,
        reason=reason,
        requires_pair=requires_assets == 2,
        max_assets=requires_assets,
        artifacts=evidence,
        detail=None,
    )


def deployment_report(
    *,
    capabilities: Iterable[str] | None = None,
    construct_failures: Mapping[str, str] | None = None,
) -> DeploymentReport:
    """Build the deployment report. Constructs nothing.

    Args:
        capabilities: override the capability set. Defaults to the serving
            registry's declared set -- the authoritative answer per the ruling.
            Tests pass an explicit set to exercise states without a filesystem.
        construct_failures: capability -> error string, for a caller that HAS
            attempted construction and wants the defect surfaced. The metadata
            path never populates this.

    Returns:
        A :class:`DeploymentReport` whose `health()` and `capabilities_block()`
        are the two served payloads, derived from one traversal so they cannot
        disagree.

    Raises:
        Nothing on the metadata path. A missing artifact degrades; a broken
        config propagates, because a service that cannot read its own config is
        not degraded, it is misconfigured.
    """
    from app.space_app import GPU_DURATIONS
    from core.config import get_config

    cfg = get_config()
    deployment: Mapping[str, Any] = cfg.as_dict().get("deployment", {}) or {}
    requested = tuple(capabilities) if capabilities is not None else _registry_capabilities()
    # `registry_declares` is the REGISTRY's set, always. It must not be confused
    # with `requested`: a caller may pass a capability the registry does not
    # declare (a test, or a future caller probing a hypothetical), and such a
    # capability must be refused rather than reported servable. Conflating the
    # two was a real bug -- `known = set(declared)` made every explicitly-passed
    # capability "declared", so `not_a_real_capability` came back
    # `available: true`.
    registry_declares = set(_registry_capabilities())

    failures = dict(construct_failures or {})
    reports = tuple(
        _report_for(
            cap,
            declared=cap in registry_declares,
            construct_failure=failures.get(cap),
        )
        for cap in sorted(set(requested))
    )

    return DeploymentReport(
        config_hash=cfg.hash,
        # The device the process will actually use. Read from the `Config`
        # PROPERTY, not from a dict key -- see `_effective_device` for the
        # latent bug that reading a dict key caused.
        device=_effective_device(cfg),
        platform=deployment.get("platform"),
        zerogpu=deployment.get("zerogpu"),
        lazy_load=deployment.get("lazy_load"),
        cache_max_models=deployment.get("cache_max_models"),
        torch_compile=deployment.get("torch_compile"),
        gpu_durations=dict(GPU_DURATIONS),
        capabilities=reports,
        device_is_cuda=_cuda_detected(),
    )


def _effective_device(cfg: Any) -> str | None:
    """The device this process will actually use, or None if undeterminable.

    THIS FIXES A LATENT BUG, and it is worth naming precisely.

    `device_preference` is a **`@property` on `Config`** (`core/config.py:87-91`),
    not a key in `configs/base.yaml`. It reads `SATQUERY_DEVICE` from the
    environment and otherwise probes `torch.cuda.is_available()`. The previous
    implementation in `app/space_app.py` read
    `cfg.as_dict().get("device_preference")` -- a mapping lookup for a name that
    a mapping can never contain -- so `GET /v1/health` served `"device": null`
    on every request, in every deployment, and nothing could detect it because
    `null` is a legal value for the field.

    WHICH SOURCE, AND WHY THE PROPERTY IS THE WRONG ONE HERE
    --------------------------------------------------------
    The property is the runtime truth, but reading it **imports torch**
    (`core/config.py:239`), which breaks requirement 1 on the metadata path --
    the same reason `_registry_capabilities` reads the spec table instead of
    instantiating a registry. So this reads the sources the property itself
    reads, in the same order, without triggering its probe:

        1. `SATQUERY_DEVICE`               -- the property's own first source
        2. `deployment.device` (base.yaml) -- the configured preference
        3. "cpu"                            -- the property's fallback

    and then applies the config-ahead-of-hardware correction:

        a configured cuda on a machine with no CUDA  ->  "cpu"

    The correction matters because the frontend is told to display this field:
    reporting `"cuda"` from a process that cannot use CUDA would be a false
    statement about the deployment.

    `base.yaml:55` records `device: auto  # auto | cpu | cuda`, so `auto` is
    resolved to the concrete device rather than echoed — `"auto"` is not one of
    the values `API_CONTRACT.md` section 2.1 lists for this field
    (`"cpu" | "cuda" | "mps" | null`).

    F-8 — NORMALISED, AND WHY THIS FUNCTION IS THE ONE THAT VALIDATES
    -----------------------------------------------------------------
    Measured before this fix (probe `probe_f8_split.py`), `SATQUERY_DEVICE` had
    **four read sites** and only three of them normalised:

        core/config.py:88      Config.device_preference        raw, no strip, no lower
        app/deployment.py:570  gpu_available                   .strip().lower()
        app/deployment.py:946  this function                   .strip(), no lower   <-- the odd one
        app/deployment.py:970  _cuda_detected                  .strip().lower()

    Two defects followed from that, both measured:

      A. **Case changed the answer.** `'cuda'` -> `'cpu'` but `'CUDA'` -> `'CUDA'`,
         because the correction below compared case-sensitively while
         `gpu_available` did not. One health payload could therefore announce
         `gpu_available: true` alongside `device: "CUDA"` — a GPU is claimed and
         the device name is not a device.
      B. **An unparsable value was echoed.** `'garbage'` -> `device: "garbage"`.
         `DEPLOYMENT_ARCHITECTURE.md:287` and `API_CONTRACT.md:131` both publish a
         closed set; the reader accepted any string, so a typo became the
         deployment's reported device.

    This function now normalises (casefold + strip) and **validates**, returning
    `None` for anything outside the contract's set. `None` is deliberate over
    raising or over a silent `"cpu"`:

      * `None` is already a legal value for the field (`"cpu" | "cuda" | "mps" |
        null`), so it travels the existing contract rather than inventing one;
      * it is honest — "the operator set something I do not recognise" is not
        "this deployment uses the CPU";
      * defaulting to `"cpu"` would mean a typo silently changes which device the
        process is believed to use, which is the `_asset_max_file_bytes` mistake
        from F-7 in a different variable. Do not repeat it.

    Note what is *not* changed here. `Config.device_preference` (`core/config.py:87`)
    still returns the raw override, because it is a general-purpose property whose
    other callers may legitimately want the operator's literal text, and narrowing
    it would be a wider change than this defect warrants. This function is the
    **served** path, and it is the one the contract constrains, so it is the one
    that validates. `gpu_available` and `_cuda_detected` already handle case, so
    they needed no change.
    """
    import os

    override = os.environ.get("SATQUERY_DEVICE", "").strip()
    if override:
        preference = override
    else:
        raw = cfg.as_dict().get("deployment", {}) or {}
        candidate = raw.get("device")
        preference = str(candidate).strip() if candidate else "auto"

    # F-8: normalise before comparing, so the case of the operator's input cannot
    # decide which branch runs.
    folded = preference.lower()

    if folded == "auto":
        return "cuda" if _cuda_detected() else "cpu"
    if folded == "cuda" and not _cuda_detected():
        return "cpu"
    if folded in _LEGAL_DEVICES:
        return folded
    # F-8: an unrecognised value is not a device. `None` is the contract's own
    # way of saying "unknown", and is preferred over echoing the raw text or
    # silently claiming the CPU.
    return None


#: The set `docs/API_CONTRACT.md` section 2.1 publishes for the `device` field.
#: Kept as a literal, not derived from the return paths above: the guard's whole
#: value is that it encodes the CONTRACT's set, so deriving it from the
#: implementation would make it unable to fail.
_LEGAL_DEVICES: frozenset[str] = frozenset({"cpu", "cuda", "mps"})


def _cuda_detected() -> bool:
    """Whether a CUDA device is present, WITHOUT importing torch.

    Same reasoning as :attr:`DeploymentReport.gpu_available`: a metadata request
    must not pull the model stack (requirement 1), and importing torch to answer
    "is there a GPU?" costs more than the answer is worth on a health route.
    """
    import os

    if os.environ.get("SATQUERY_DEVICE", "").strip().lower() == "cuda":
        return True
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return bool(visible) and visible != "-1"


# ---------------------------------------------------------------------------
# Thin payload helpers (what the routes call)
# ---------------------------------------------------------------------------
def health_payload() -> dict[str, Any]:
    """The `GET /v1/health` body. This is the ONLY public health source."""
    return deployment_report().health()


def capabilities_payload() -> dict[str, Any]:
    """The `GET /v1/capabilities` body."""
    return deployment_report().capabilities_block()
