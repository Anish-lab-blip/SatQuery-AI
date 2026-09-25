"""SatQuery AI — error taxonomy.

Every failure the system can produce is one of these. The controller maps each to a
user-facing message, a trace entry, and a fallback decision (see
docs/ARCHITECTURE_FREEZE.md section 5 and the plan's Failure Matrix).

Never raise a bare Exception from specialist or controller code.
"""

from __future__ import annotations

import re
from typing import Any


# --------------------------------------------------------------------------
# Path scrubbing (F-15)
# --------------------------------------------------------------------------
#: Filesystem-path shapes that must never be published. Each pattern matches an
#: **absolute** path; the replacement keeps only its final component, so
#: "cannot read C:\\a\\b\\weights.pt" becomes "cannot read weights.pt" -- still
#: diagnostic, no longer a location disclosure.
#:
#: Relative paths are deliberately NOT matched. A rule broad enough to catch
#: `artifacts/change/head.pt` also catches `and/or` and the path segments of a
#: URL, and a scrubber that mangles ordinary prose is a worse defect than the
#: disclosure it fixes. The measured leaks are all absolute.
_WINDOWS_DRIVE_PATH = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/](?:[^\\/\s\"'<>|:*?]+[\\/])*[^\\/\s\"'<>|:*?]*"
)
_UNC_PATH = re.compile(r"\\\\[^\\/\s\"'<>|:*?]+(?:\\[^\\/\s\"'<>|:*?]+)+")
#: The lookbehind refuses to start a match immediately after `:` or `/`, which
#: is what keeps `https://github.com/antofuller/CROMA` intact.
_POSIX_PATH = re.compile(
    r"(?<![:\w/])/(?:[^/\s\"'<>|:*?]+/)*[^/\s\"'<>|:*?]+"
)


def scrub_paths(text: str | None) -> str | None:
    """Reduce every absolute filesystem path in `text` to its final component.

    WHY THIS EXISTS
    ---------------
    F-15 (owner ruling 2026-09-23): *sanitize all client-facing exception
    messages; retain full exception details only in server-side diagnostics.*

    Exception messages in this repo routinely embed an absolute path:
    `specialists/optical_sar/croma.py` raises *"CROMA requires the vendored
    'use_croma.py', which is not in 'C:\\\\...\\\\empty_vendor_dir'"* and
    `specialists/change/stanet.py` raises *"could not read encoder weights from
    C:\\\\..."*. Those strings reach client-visible fields, and
    `API_CONTRACT.md` section 7 records that v1 has **no auth**.

    The repair is a BASENAME reduction rather than a deletion, and that is the
    idiom `core/controller.py::_asset_label` already uses for F-13/F-14: one
    rule, one implementation, applied at every client-facing write site.

    A blunt replacement of the whole message with a generic string would also
    stop the leak, and it was tried first -- but it discarded path-free
    diagnostics the client can legitimately act on (*"... has no builder
    'build_x'"*, *"no GPU in this dimension"*), and three existing tests that
    pin exactly those diagnostics failed. **A fix that forces legitimate tests
    to be weakened is aimed at the wrong granularity.**

    URLs are left intact on purpose: `https://github.com/antofuller/CROMA`
    appears inside one of the very messages this scrubs, and mangling it would
    be a worse defect than the one being repaired.
    """
    if not text:
        return text

    def _basename(match: re.Match[str]) -> str:
        raw = match.group(0)
        tail = re.split(r"[\\/]", raw.rstrip("\\/"))[-1]
        return tail or raw

    for pattern in (_UNC_PATH, _WINDOWS_DRIVE_PATH, _POSIX_PATH):
        text = pattern.sub(_basename, text)
    return text


class SatQueryError(Exception):
    """Base class for every SatQuery failure.

    Attributes:
        code:        stable machine-readable identifier, used in traces.
        user_message: text safe to show the operator.
        detail:      technical detail for the execution trace (never chain-of-thought).
        recoverable: whether the controller may continue with a fallback.
    """

    code: str = "satquery_error"
    user_message: str = "An internal error occurred."

    def __init__(
        self,
        detail: str = "",
        *,
        user_message: str | None = None,
        recoverable: bool = False,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(detail or self.user_message)
        self.detail = detail
        self.recoverable = recoverable
        self.context = context or {}
        if user_message is not None:
            self.user_message = user_message

    def to_trace(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "detail": self.detail,
            "recoverable": self.recoverable,
            "context": self.context,
        }


# --------------------------------------------------------------------------
# Input / raster
# --------------------------------------------------------------------------
class InputError(SatQueryError):
    code = "input_error"
    user_message = "The uploaded input could not be read."


class RasterReadError(InputError):
    code = "raster_read_error"
    user_message = "The uploaded file is not a readable TIFF/GeoTIFF."


class MissingCRSError(InputError):
    code = "missing_crs"
    user_message = "The image has no coordinate reference system."
    # Degraded, not fatal: non-geospatial analysis may still be possible.

    def __init__(self, detail: str = "", **kw: Any) -> None:
        kw.setdefault("recoverable", True)
        super().__init__(detail, **kw)


class UnsupportedBandsError(InputError):
    code = "unsupported_bands"
    user_message = "The image band layout is not supported."


class OversizedImageError(InputError):
    code = "oversized_image"
    user_message = "The image exceeds the configured pixel budget."
    # Recoverable via downscale.

    def __init__(self, detail: str = "", **kw: Any) -> None:
        kw.setdefault("recoverable", True)
        super().__init__(detail, **kw)


# --------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------
class PairCompatibilityError(SatQueryError):
    code = "pair_incompatible"
    user_message = "The two images are not sufficiently compatible."


class PairMisalignmentError(PairCompatibilityError):
    code = "pair_misaligned"
    user_message = "The images are not sufficiently co-registered for spatial analysis."


class TemporalPairError(PairCompatibilityError):
    code = "temporal_pair_invalid"
    user_message = "Temporal analysis requires two distinct, compatible acquisitions."


# --------------------------------------------------------------------------
# Routing / planning
# --------------------------------------------------------------------------
class RoutingError(SatQueryError):
    code = "routing_error"
    user_message = "The request could not be interpreted."


class UnsupportedQueryError(RoutingError):
    code = "unsupported_query"
    user_message = "No specialist supports this request."


class InvalidRequestError(SatQueryError):
    code = "invalid_request"
    user_message = "The uploaded inputs do not support the requested task."


class WorkflowPlanError(SatQueryError):
    code = "workflow_plan_error"
    user_message = "The requested workflow could not be planned."


# --------------------------------------------------------------------------
# Specialists
# --------------------------------------------------------------------------
class SpecialistError(SatQueryError):
    code = "specialist_error"
    user_message = "A specialist failed to complete."

    def __init__(self, detail: str = "", *, specialist: str = "", **kw: Any) -> None:
        self.specialist = specialist
        kw.setdefault("context", {})
        kw["context"].setdefault("specialist", specialist)
        super().__init__(detail, **kw)


class ModelLoadError(SpecialistError):
    code = "model_load_error"
    user_message = "A required model could not be loaded."


class ModelUnavailableError(SpecialistError):
    code = "model_unavailable"
    user_message = "A required model is not available in this environment."
    # Recoverable: the controller degrades the workflow.

    def __init__(self, detail: str = "", **kw: Any) -> None:
        kw.setdefault("recoverable", True)
        super().__init__(detail, **kw)


class OutOfMemoryError(SpecialistError):
    code = "out_of_memory"
    user_message = "Ran out of memory; retrying at reduced resolution."
    # Recoverable: retry at lower resolution.

    def __init__(self, detail: str = "", **kw: Any) -> None:
        kw.setdefault("recoverable", True)
        super().__init__(detail, **kw)


class SpecialistTimeoutError(SpecialistError):
    code = "specialist_timeout"
    user_message = "Processing timed out."
    #: Recoverable, per `docs/API_CONTRACT.md` section 5.1, which maps 504 with
    #: `recoverable: true`. Two independent reasons:
    #:
    #:   1. `docs/API_CONTRACT.md` is the frozen frontend-facing contract. A
    #:      frontend that reads `recoverable: false` will not offer a retry for
    #:      the one failure the contract explicitly tells it to retry.
    #:   2. The plan's Failure Matrix (§57) lists Timeout with the recovery
    #:      "abort specialist" and the fallback "partial result" -- i.e. the
    #:      controller continues rather than failing the request. A terminal
    #:      `recoverable=False` contradicts that.
    #:
    #: Inheriting `False` from `SatQueryError` was the defect this default
    #: corrects. Note the controller currently only reuses `.code` for its
    #: budget-skip trace entry (`core/controller.py:464`), so nothing in the
    #: pipeline constructed this class and the wrong default was never
    #: observable from the inside -- only from a client.
    def __init__(self, detail: str = "", **kw: Any) -> None:
        kw.setdefault("recoverable", True)
        super().__init__(detail, **kw)


# --------------------------------------------------------------------------
# Output integrity
# --------------------------------------------------------------------------
class SchemaValidationError(SatQueryError):
    code = "schema_validation_error"
    user_message = "The system produced a malformed result."


class CoordinateError(SchemaValidationError):
    code = "coordinate_error"
    user_message = "A spatial result carried invalid coordinates."


class ConfidenceRangeError(SchemaValidationError):
    code = "confidence_range_error"
    user_message = "A confidence value fell outside the valid range."


# --------------------------------------------------------------------------
# Leakage / evaluation
# --------------------------------------------------------------------------
class LeakageError(SatQueryError):
    code = "leakage_violation"
    user_message = "A data isolation rule was violated."


class BenchmarkFreezeError(SatQueryError):
    code = "benchmark_freeze_error"
    user_message = "Evaluation cannot proceed: the benchmark is not frozen."


__all__ = [
    "SatQueryError",
    "InputError",
    "RasterReadError",
    "MissingCRSError",
    "UnsupportedBandsError",
    "OversizedImageError",
    "PairCompatibilityError",
    "PairMisalignmentError",
    "TemporalPairError",
    "RoutingError",
    "UnsupportedQueryError",
    "InvalidRequestError",
    "WorkflowPlanError",
    "SpecialistError",
    "ModelLoadError",
    "ModelUnavailableError",
    "OutOfMemoryError",
    "SpecialistTimeoutError",
    "SchemaValidationError",
    "CoordinateError",
    "ConfidenceRangeError",
    "LeakageError",
    "BenchmarkFreezeError",
]