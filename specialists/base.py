"""SatQuery AI — the common specialist interface.

Every specialist implements exactly this. The controller must not care whether
a specialist is a local Python object, a model, or a future remote worker
(docs/ARCHITECTURE_FREEZE.md section 2).

The interface is four methods, and the split between them is deliberate:

    validate_request   -> can this specialist serve this request at all?
    execute            -> do the work, return a SpecialistResult
    produce_evidence   -> what observable artefacts support the result?
    estimate_confidence-> what measurable signals support the score?

`execute` returns a fully-formed `SpecialistResult`; the other three exist so
the controller and the evidence engine can interrogate a specialist without
running it.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from core.errors import SpecialistError
from core.schemas import (
    AssetMetadata,
    ConfidenceBreakdown,
    Evidence,
    SpecialistResult,
)


@dataclass(frozen=True)
class SpecialistRequest:
    """Everything a specialist is given. No specialist reads global state."""

    assets: list[AssetMetadata]
    query: str
    params: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None

    @property
    def asset_count(self) -> int:
        return len(self.assets)


class Specialist(abc.ABC):
    """Abstract base for every SatQuery specialist."""

    #: Stable machine-readable name, used in traces and evidence sources.
    name: str = "specialist"

    #: Semantic version. Bump when behaviour changes, not when code moves.
    version: str = "0.0.0"

    #: Capability strings this specialist can serve, e.g. {"vqa", "caption"}.
    capabilities: tuple[str, ...] = ()

    # -- contract ----------------------------------------------------------

    @abc.abstractmethod
    def validate_request(self, request: SpecialistRequest) -> None:
        """Raise a typed `SatQueryError` if this request cannot be served.

        Returning normally means "I can attempt this". It is not a promise of
        success — only that the inputs are structurally acceptable.

        Implementations must raise the *most specific* error available
        (`InvalidRequestError`, `PairMisalignmentError`, ...), never a bare
        Exception, because the controller maps error codes to user messages.
        """

    @abc.abstractmethod
    def execute(self, request: SpecialistRequest) -> SpecialistResult:
        """Perform the work and return a normalised result.

        Raises:
            SpecialistError subclasses on failure. Never returns None.
        """

    @abc.abstractmethod
    def produce_evidence(self, result: SpecialistResult) -> list[Evidence]:
        """Extract the observable evidence supporting a result.

        Called by the evidence engine. Must not invent anything the specialist
        did not actually compute.
        """

    @abc.abstractmethod
    def estimate_confidence(self, result: SpecialistResult) -> ConfidenceBreakdown:
        """Return the measurable confidence breakdown for a result.

        Never an LLM utterance. Every component must be a number this
        specialist can point at.
        """

    # -- helpers -----------------------------------------------------------

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def require_assets(self, request: SpecialistRequest, count: int) -> None:
        """Assert an exact asset count, raising a typed error otherwise."""
        from core.errors import InvalidRequestError

        if request.asset_count != count:
            raise InvalidRequestError(
                f"{self.name} requires exactly {count} asset(s), "
                f"got {request.asset_count}",
                context={"specialist": self.name, "expected": count,
                         "actual": request.asset_count},
            )

    def require_capability(self, capability: str) -> None:
        if not self.supports(capability):
            raise SpecialistError(
                f"{self.name} does not declare capability {capability!r}",
                specialist=self.name,
            )

    def model_refs(self) -> list[dict[str, str]]:
        """Model identities for the execution trace. Override where relevant."""
        return [{"name": self.name, "revision": self.version, "role": "specialist"}]

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": list(self.capabilities),
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} v{self.version}>"


__all__ = ["Specialist", "SpecialistRequest"]