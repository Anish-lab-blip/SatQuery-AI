"""SatQuery AI — the specialist registry (Phase 15).

WHAT THIS IS FOR
----------------
`docs/ARCHITECTURE_FREEZE.md` section 6 puts the control tier in three files.
This is the mechanism layer: it knows *which* specialists exist and *how* to
construct them, and nothing about policy. It never decides what runs — that is
`core.planner`'s job alone (section 5: "policy engine *decides*").

    SpecialistRegistry.discover()      -> a spec table; imports nothing
    SpecialistRegistry.build(cap)      -> construct one, memoised
    SpecialistRegistry.build_all()     -> construct everything (health panel)
    SpecialistRegistry.describe()      -> observable state for the trace

WHY A SPEC TABLE RATHER THAN IMPORTS
------------------------------------
This module holds **no** module-level import of any specialist. The builders
live behind `importlib.import_module`, called inside `build()`.

The reason is documented in `specialists/__init__.py`: importing
`specialists.vqa.inference` pulls in the VLM loader; importing
`specialists.optical_sar.*` pulls in CROMA. A registry that imported all four at
package load would pay every model's import cost on every request, defeating
the lazy-loading design (plan section 49). Three earlier defect clusters in this
repo came from import-time failures, so the discipline is not theoretical.

A caller who only wants to ask "which capabilities exist?" therefore pays
nothing: `discover()` reads a table of strings.

CAPABILITY IS THE KEY, NOT THE NAME
-----------------------------------
This is the trap this module exists to encode. The VQA specialist declares:

    name = "vlm"                      # specialists/vqa/inference.py:114
    capabilities = ("vqa", "caption") # specialists/vqa/inference.py:116

Every other specialist's `name` equals its single capability
(`grounding`/`("grounding",)`, `change`/`("change",)`,
`optical_sar`/`("optical_sar",)`).

So a registry keyed on `name` would make `vqa` permanently unfindable while
every other specialist kept working — a bug that hides. The registry keys on
**capability** and asserts the capability tuple against the constructed object,
so a future mismatch fails loudly at construction instead of silently
returning nothing.

THREE STATES, NOT TWO
---------------------
A specialist whose weights are missing is not the same as one that cannot be
built at all, and neither is the same as one that is fully wired:

    AVAILABLE    constructed, artifacts present
    DEGRADED     constructed, running on fallback/untrained weights -- planable
    UNAVAILABLE  construction raised -- entry RETAINED, never silently dropped

`build()` returns an `UNAVAILABLE` entry rather than raising (except for an
unknown capability) so the controller can say "optical_sar was planned but
could not be constructed: CROMA could not load" instead of either crashing or
pretending the step never existed. This is the honest-degradation principle
applied to the control layer: an absence must be distinguishable from a
non-event.

A MISSING ARTIFACT IS NOT A CORRUPT ONE
---------------------------------------
The four builders already encode this distinction and the registry must
preserve it rather than reintroduce the bug at the control layer:

    [[ the change builder treats a *missing* checkpoint as degraded but a
       *corrupt* one as fatal — specialists/change/specialist.py:834-838:
       "silently running an untrained model because a real checkpoint failed
       to load would be the worst outcome." ]]

So a `ModelLoadError` / `ModelUnavailableError` raised during construction maps
to `UNAVAILABLE` (a defect, surfaced loudly), NOT retried as `DEGRADED`. The
registry is not the place to helpfully re-add a fallback the builder refused.
"""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from core.errors import (
    ModelLoadError,
    ModelUnavailableError,
    SpecialistError,
    scrub_paths,
)

_log = logging.getLogger("satquery.registry")

# ---------------------------------------------------------------------------
# Registry states
# ---------------------------------------------------------------------------


class RegistryState(str, Enum):
    """Whether a capability can be planned, and why not when it cannot."""

    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


#: States the planner is permitted to plan. DEGRADED is included deliberately:
#: a degraded specialist still runs and returns an honest, degraded result, so a
#: local install with no trained weights exercises real code paths (design §9).
PLANABLE_STATES: frozenset[RegistryState] = frozenset(
    {RegistryState.AVAILABLE, RegistryState.DEGRADED}
)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpecialistSpec:
    """One capability's declaration. A row in a table, not an import.

    Attributes:
        name: registry key. This is the *capability* string, never the
            specialist's own `name` attribute — see the module docstring on the
            `vlm`/`vqa` trap.
        capabilities: the capability tuple the built object must declare. This
            is ASSERTED against the real object after construction, so a drift
            between this table and the specialist fails loudly.
        module: dotted module path containing the builder.
        builder: builder function name inside `module`.
        requires_assets: exact asset count this capability needs, or None for
            "any". Used by the planner's precondition checks and by the
            registry's own detail reporting.
        config_keys: extra keyword arguments, mapped as
            `{builder_kwarg: dotted.config.key}`. Keeps the four heterogeneous
            builder signatures out of the registry's control flow.
        optional_config_keys: as `config_keys`, but a missing key contributes
            nothing rather than passing `None`. Used for artifacts that have no
            sensible config default (e.g. a trained head path).
        failure_states: registry state to use when construction raises, keyed by
            exception class name. Anything unlisted falls back to UNAVAILABLE —
            the conservative choice, because a construction failure we do not
            understand must not be silently downgraded to a planable state.
    """

    name: str
    capabilities: tuple[str, ...]
    module: str
    builder: str
    requires_assets: int | None = None
    config_keys: Mapping[str, str] = field(default_factory=dict)
    optional_config_keys: Mapping[str, str] = field(default_factory=dict)
    failure_states: Mapping[str, RegistryState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in self.capabilities:
            raise SpecialistError(
                f"registry spec {self.name!r} does not list itself among its "
                f"capabilities {self.capabilities!r}; the registry keys on "
                f"capability, so this spec would be unreachable",
                specialist="registry",
            )


# ---------------------------------------------------------------------------
# The spec table
# ---------------------------------------------------------------------------
def default_specs() -> tuple[SpecialistSpec, ...]:
    """The four frozen specialists, as data.

    `requires_assets` mirrors each specialist's own `validate_request`, which
    remains authoritative — this is the planner's cheap precondition, not a
    replacement for the specialist's check.

    Note the VQA row: `name="vqa"` (the key) against a specialist whose own
    `name` is `"vlm"`. That asymmetry is real and is asserted after build.
    """
    return (
        # `build_vqa_specialist(config, device)` reads `vlm.*` from the config
        # itself, so it takes no extra keyword arguments. A `config_keys` entry
        # here would pass an unsupported kwarg and fail construction.
        SpecialistSpec(
            name="vqa",
            capabilities=("vqa", "caption"),
            module="specialists.vqa.inference",
            builder="build_vqa_specialist",
            requires_assets=None,
        ),
        SpecialistSpec(
            name="caption",
            capabilities=("vqa", "caption"),
            module="specialists.vqa.inference",
            builder="build_vqa_specialist",
            requires_assets=None,
        ),
        SpecialistSpec(
            name="grounding",
            capabilities=("grounding",),
            module="specialists.grounding.specialist",
            builder="build_grounding_specialist",
            requires_assets=1,
            optional_config_keys={
                "checkpoint_path": "grounding.checkpoint_path",
                "head_path": "grounding_head.head_path",
            },
        ),
        SpecialistSpec(
            name="change",
            capabilities=("change",),
            module="specialists.change.specialist",
            builder="build_change_specialist",
            requires_assets=2,
            optional_config_keys={
                "checkpoint_path": "change.checkpoint_path",
                "artifact_dir": "change.artifact_dir",
            },
        ),
        # R-02. `requires_assets=2` like `change` — the same two temporal
        # acquisitions, a different output: a short answer rather than a change
        # map. The capability key is `change_vqa`, which is deliberately NOT
        # `vqa`: a `vqa`-keyed row would make the planner route a two-asset
        # change question to the one-asset VLM and silently answer about only
        # one of the two acquisitions.
        #
        # F-17 (owner ruling 2026-09-23): `artifact_dir` was declared here as
        # `change_vqa.artifact_dir` and passed to the builder, but no code ever
        # read it -- `ChangeVqaSpecialist` stores the value and never consults
        # it. A config surface that advertises a capability the code lacks is
        # worse than an absent one: an operator sets the key, sees no error,
        # and receives no warning. The declaration is therefore REMOVED rather
        # than left as a silently-accepted no-op. `change` keeps its own
        # `artifact_dir` because that specialist genuinely writes a change map.
        SpecialistSpec(
            name="change_vqa",
            capabilities=("change_vqa",),
            module="specialists.change.vqa_specialist",
            builder="build_change_vqa_specialist",
            requires_assets=2,
            optional_config_keys={
                "checkpoint_path": "change.checkpoint_path",
                "head_path": "change_vqa.head_path",
            },
        ),
        SpecialistSpec(
            name="optical_sar",
            capabilities=("optical_sar",),
            module="specialists.optical_sar.specialist",
            builder="build_optical_sar_specialist",
            requires_assets=2,
            optional_config_keys={
                "checkpoint_path": "croma.checkpoint_path",
                "vendor_dir": "croma.vendor_dir",
                "head_path": "fusion.head_path",
                "artifact_dir": "croma.artifact_dir",
            },
        ),
    )


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegistryEntry:
    """The outcome of resolving one capability.

    Retained even when construction failed. A dropped entry would make
    "could not be built" indistinguishable from "was never asked for", which is
    the failure mode this whole design exists to prevent (design §4.3).
    """

    spec: SpecialistSpec
    specialist: Any | None = None
    state: RegistryState = RegistryState.AVAILABLE
    detail: str | None = None
    error_code: str | None = None
    build_ms: float | None = None

    @property
    def capability(self) -> str:
        return self.spec.name

    @property
    def planable(self) -> bool:
        return self.state in PLANABLE_STATES

    @property
    def degraded(self) -> bool:
        return self.state is RegistryState.DEGRADED

    def to_trace(self) -> dict[str, Any]:
        """Observable facts only. No chain-of-thought.

        F-15 (owner ruling 2026-09-23): `detail` is scrubbed here as a SECOND
        line of defence, not the only one. `_failure_entry` already scrubs what
        it stores, but a `RegistryEntry` can be constructed anywhere, and this
        method is the single seam through which an entry reaches the response
        body -- so the scrub belongs here too. Fixing only the producer would
        leave every other constructor a way to reintroduce the disclosure.
        """
        return {
            "capability": self.spec.name,
            "state": self.state.value,
            "specialist_name": getattr(self.specialist, "name", None),
            "version": getattr(self.specialist, "version", None),
            "detail": scrub_paths(self.detail),
            "error_code": self.error_code,
            "requires_assets": self.spec.requires_assets,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class SpecialistRegistry:
    """Discovers specialist capabilities lazily and constructs them on demand.

    Construction is memoised: the second `build("vqa")` returns the cached
    entry rather than loading the VLM again.

    Thread-safety note: this is a single-process monolith (freeze §5), and v1
    executes plans sequentially, so no lock is taken. Adding one without
    concurrent execution would be complexity without a represented hazard.
    """

    def __init__(
        self,
        config: Any,
        *,
        device: str | None = None,
        specs: tuple[SpecialistSpec, ...] | None = None,
        builders: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Args:
            config: the central `core.config.Config`.
            device: torch device string; defaults to `config.device_preference`.
            specs: override the spec table (used by tests).
            builders: override builders by capability name, bypassing importlib
                entirely. This is how tests inject stubs without touching the
                filesystem or torch.
        """
        self.config = config
        self.device = device or config.device_preference
        self._specs: dict[str, SpecialistSpec] = {}
        self._entries: dict[str, RegistryEntry] = {}
        self._overrides: Mapping[str, Any] = builders or {}

        for spec in specs if specs is not None else default_specs():
            self._register(spec)

    # -- discovery ---------------------------------------------------------

    @classmethod
    def discover(
        cls,
        config: Any,
        *,
        device: str | None = None,
        specs: tuple[SpecialistSpec, ...] | None = None,
        builders: Mapping[str, Any] | None = None,
    ) -> "SpecialistRegistry":
        """Register every spec. Builds nothing and imports no specialist."""
        return cls(config, device=device, specs=specs, builders=builders)

    def _register(self, spec: SpecialistSpec) -> None:
        # Two specs sharing the key "vqa"/"caption" both resolve to the VLM
        # builder. That is intentional: they are two capabilities of one
        # specialist, and each gets its own entry so the planner can ask for
        # either by capability. Duplicate KEYS, however, are a table defect.
        if spec.name in self._specs:
            raise SpecialistError(
                f"duplicate registry capability {spec.name!r}",
                specialist="registry",
            )
        self._specs[spec.name] = spec

    def available(self) -> tuple[str, ...]:
        """Declared capabilities, sorted. Import-free."""
        return tuple(sorted(self._specs))

    def specs(self) -> tuple[SpecialistSpec, ...]:
        return tuple(self._specs[name] for name in self.available())

    def entry(self, capability: str) -> RegistryEntry:
        """The entry for a capability, WITHOUT constructing it.

        Returns a lazily-described placeholder when nothing has been built yet,
        so a caller can inspect the plan-relevant facts (asset count, module)
        without paying for construction.
        """
        spec = self._specs.get(capability)
        if spec is None:
            raise SpecialistError(
                f"unknown capability {capability!r}; known: "
                f"{list(self.available())}",
                specialist="registry",
            )
        return self._entries.get(
            capability,
            RegistryEntry(
                spec=spec,
                specialist=None,
                state=RegistryState.AVAILABLE,
                detail="not yet constructed",
            ),
        )

    def entries(self) -> tuple[RegistryEntry, ...]:
        """All known entries, by capability, without constructing anything."""
        return tuple(self.entry(name) for name in self.available())

    # -- construction ------------------------------------------------------

    def build(self, capability: str) -> RegistryEntry:
        """Construct the specialist for a capability, memoised.

        Raises:
            SpecialistError: ONLY for an unknown capability. A construction
                *failure* is returned as an `UNAVAILABLE` entry so the caller
                can record it in the trace rather than losing the fact.
        """
        if capability not in self._specs:
            raise SpecialistError(
                f"unknown capability {capability!r}; known: "
                f"{list(self.available())}",
                specialist="registry",
            )
        cached = self._entries.get(capability)
        if cached is not None:
            return cached

        spec = self._specs[capability]
        started = time.perf_counter()
        try:
            specialist = self._construct(spec)
            # Validation lives INSIDE the try deliberately. A capability
            # mismatch is a construction failure like any other, and the
            # contract is that only an unknown capability raises from build().
            # Letting it escape would crash the plan instead of recording an
            # omission, which is the behaviour this module exists to prevent.
            entry = self._success_entry(spec, specialist, started)
        except Exception as exc:  # noqa: BLE001 - mapped to an entry below
            entry = self._failure_entry(spec, exc, started)

        self._entries[capability] = entry
        return entry

    def build_all(self) -> tuple[RegistryEntry, ...]:
        """Construct everything. Used by the GUI health panel and smoke tests."""
        return tuple(self.build(name) for name in self.available())

    def reset(self) -> None:
        """Forget every constructed entry. Does not unload torch models."""
        self._entries.clear()

    # -- internals ---------------------------------------------------------

    def _construct(self, spec: SpecialistSpec) -> Any:
        """Import the builder and call it. The only place importlib is used."""
        builder = self._overrides.get(spec.name)
        if builder is None:
            module = importlib.import_module(spec.module)
            builder = getattr(module, spec.builder, None)
            if builder is None:
                raise SpecialistError(
                    f"{spec.module!r} has no builder {spec.builder!r}",
                    specialist="registry",
                )

        kwargs = self._builder_kwargs(spec)
        return builder(self.config, **kwargs)

    def _builder_kwargs(self, spec: SpecialistSpec) -> dict[str, Any]:
        """Assemble keyword arguments from the config mapping.

        Only keys that actually resolve are passed. Passing an explicit `None`
        for an absent optional key would be indistinguishable to a builder from
        a deliberately-null one, and several builders use `None` to mean
        "artifact genuinely absent, degrade" — a distinction that must not be
        manufactured by the registry.
        """
        kwargs: dict[str, Any] = {"device": self.device}
        for kwarg, config_path in spec.config_keys.items():
            value = self.config.get(config_path)
            if value is not None:
                kwargs[kwarg] = value
        for kwarg, config_path in spec.optional_config_keys.items():
            value = self.config.get(config_path)
            if value is not None:
                kwargs[kwarg] = value
        return kwargs

    def _success_entry(
        self, spec: SpecialistSpec, specialist: Any, started: float
    ) -> RegistryEntry:
        """Validate the built object and record its state.

        The capability assertion is the load-bearing check. It is what turns
        the `name="vlm"` / `capability="vqa"` asymmetry from a silent
        unfindable specialist into a loud failure, and it catches the broader
        class of defect this repo has hit repeatedly: a declared interface that
        drifted from the real one.
        """
        declared = tuple(getattr(specialist, "capabilities", ()))
        if declared != spec.capabilities:
            raise SpecialistError(
                f"capability mismatch for {spec.name!r}: the registry declares "
                f"{spec.capabilities!r} but {type(specialist).__name__} "
                f"declares {declared!r}",
                specialist="registry",
            )

        elapsed = (time.perf_counter() - started) * 1000.0
        degraded, reason = self._degradation_of(specialist)
        return RegistryEntry(
            spec=spec,
            specialist=specialist,
            state=RegistryState.DEGRADED if degraded else RegistryState.AVAILABLE,
            detail=reason,
            build_ms=round(elapsed, 3),
        )

    @staticmethod
    def _degradation_of(specialist: Any) -> tuple[bool, str | None]:
        """Ask a constructed specialist whether it is running on fallbacks.

        Deliberately duck-typed and conservative: a specialist that does not
        expose any recognised flag is reported AVAILABLE, because inventing a
        degradation the specialist did not declare would be worse than
        reporting a clean build. Each specialist's own result still carries its
        authoritative `degraded` flag at execution time.
        """
        for attr, reason in (
            ("has_checkpoint", "no trained checkpoint; running untrained"),
            ("has_head", "no trained head; running on fallback"),
            ("has_encoder", "no encoder; running on fallback"),
        ):
            value = getattr(specialist, attr, None)
            if value is False:
                return True, reason
        # An explicitly-absent model the specialist reports as None.
        if getattr(specialist, "model", "missing") is None:
            return True, "no model loaded"
        return False, None

    def _failure_entry(
        self, spec: SpecialistSpec, exc: Exception, started: float
    ) -> RegistryEntry:
        """Map a construction failure to a retained, non-planable entry.

        A corrupt artifact must NOT be retried as a degraded build: that is
        exactly the outcome `specialists/change/specialist.py:834-838` forbids.
        The entry is returned, not raised, so the controller can record why the
        capability was omitted.

        F-15 (owner ruling 2026-09-23) — **the exception string is a SERVER-SIDE
        diagnostic and must not be published.** It routinely names absolute
        paths: `croma.py` raises *"CROMA requires the vendored 'use_croma.py',
        which is not in 'C:\\...\\empty_vendor_dir'"* and `stanet.py` raises
        *"could not read encoder weights from C:\\..."*. `to_trace()` publishes
        this entry's `detail` to the client, and `API_CONTRACT.md` §7 records
        that there is **no auth in v1** — so a raw `str(exc)` here is a
        filesystem disclosure to an unauthenticated caller.

        Measured before this fix: a real construction failure put the vendor
        directory path into **four** client-visible fields — `result.warnings[]`,
        `result.evidence[].payload["message"]`, `trace.parameters.registry.
        built[<cap>].detail`, and the same registry block again inside
        `result.execution_trace` (the trace is the same object, serialized
        twice).

        So: log the full detail here, where an operator can read it, and publish
        a path-scrubbed copy. The scrub is a **basename reduction**, not a
        replacement, so a path-free reason the client can act on survives --
        see `core.errors.scrub_paths`.
        """
        state = spec.failure_states.get(
            type(exc).__name__, RegistryState.UNAVAILABLE
        )
        code = getattr(exc, "code", type(exc).__name__)
        raw_detail = getattr(exc, "detail", None) or str(exc) or type(exc).__name__

        _log.warning(
            "specialist construction failed: capability=%s code=%s detail=%s",
            spec.name,
            code,
            raw_detail,
            exc_info=exc,
        )

        # F-15: what is published is the SCRUBBED detail. The raw string above
        # is the server-side diagnostic; this is the client-facing one.
        detail = scrub_paths(raw_detail)
        elapsed = (time.perf_counter() - started) * 1000.0

        if isinstance(exc, (ModelLoadError, ModelUnavailableError)):
            # Both are already the right classification. Kept explicit so the
            # intent is readable rather than implied by the default.
            state = RegistryState.UNAVAILABLE

        return RegistryEntry(
            spec=spec,
            specialist=None,
            state=state,
            detail=detail,
            error_code=str(code),
            build_ms=round(elapsed, 3),
        )

    # -- observability -----------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Registry state for the trace. Constructs nothing.

        Reports both the declared capability set and any entry that has already
        been built, so a trace recorded mid-run reflects what was actually
        attempted without forcing construction of the rest.
        """
        return {
            "declared": list(self.available()),
            "built": {
                name: entry.to_trace() for name, entry in sorted(self._entries.items())
            },
            "device": self.device,
        }


__all__ = [
    "PLANABLE_STATES",
    "RegistryEntry",
    "RegistryState",
    "SpecialistRegistry",
    "SpecialistSpec",
    "default_specs",
]
