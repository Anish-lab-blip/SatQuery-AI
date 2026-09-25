"""SatQuery AI — CRS handling.

Comparison and compatibility checks between coordinate reference systems. Kept
separate from transform.py so that "can these two rasters be compared at all?" is
answerable without touching pixel arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyproj import CRS
from pyproj.exceptions import CRSError as PyProjCRSError

from core.errors import MissingCRSError


@dataclass(frozen=True)
class CRSCompatibility:
    """Result of comparing two coordinate reference systems."""

    compatible: bool
    identical: bool
    same_units: bool
    left: str | None
    right: str | None
    reason: str = ""

    @property
    def requires_reprojection(self) -> bool:
        return self.compatible and not self.identical


def parse_crs(value: str | None) -> CRS | None:
    """Parse a CRS string. Returns None if absent; raises if malformed."""
    if value is None or value == "":
        return None
    try:
        return CRS.from_user_input(value)
    except PyProjCRSError as exc:
        raise MissingCRSError(f"unparseable CRS '{value}': {exc}") from exc


def describe_crs(value: str | None) -> dict[str, object]:
    """Human-readable summary for the execution trace."""
    crs = parse_crs(value)
    if crs is None:
        return {"present": False}
    return {
        "present": True,
        "srs": crs.srs,
        "name": crs.name,
        "is_geographic": crs.is_geographic,
        "is_projected": crs.is_projected,
        "axis_units": crs.axis_info[0].unit_name if crs.axis_info else None,
    }


def compare_crs(left: str | None, right: str | None) -> CRSCompatibility:
    """Determine whether two CRS values are compatible for joint analysis.

    Identical CRS is always compatible. Different but reprojectable CRS values are
    compatible with a reprojection requirement. A missing CRS on either side is NOT
    compatible for spatial comparison — but it is not fatal for non-spatial tasks.
    """
    if left is None or right is None:
        return CRSCompatibility(
            compatible=False,
            identical=False,
            same_units=False,
            left=left,
            right=right,
            reason="one or both rasters lack a CRS; spatial comparison is unsafe",
        )

    crs_l = parse_crs(left)
    crs_r = parse_crs(right)
    assert crs_l is not None and crs_r is not None

    identical = crs_l.equals(crs_r)

    units_l = crs_l.axis_info[0].unit_name if crs_l.axis_info else "unknown"
    units_r = crs_r.axis_info[0].unit_name if crs_r.axis_info else "unknown"
    same_units = units_l == units_r

    # Both projected or both geographic is the safe case. Mixing a projected CRS
    # with a geographic one is legal to reproject but signals a pipeline mistake.
    same_kind = crs_l.is_projected == crs_r.is_projected

    if identical:
        return CRSCompatibility(True, True, same_units, left, right, "identical CRS")
    if same_kind:
        return CRSCompatibility(
            True, False, same_units, left, right,
            f"reprojection required ({crs_l.name} -> {crs_r.name})",
        )
    return CRSCompatibility(
        True, False, same_units, left, right,
        f"mixed projected/geographic CRS ({crs_l.name} vs {crs_r.name}); "
        "reprojection required and should be verified",
    )


def is_metric(crs_value: str | None) -> bool:
    """True when the CRS measures in linear units (metres), so area is meaningful."""
    crs = parse_crs(crs_value)
    if crs is None:
        return False
    if not crs.is_projected:
        return False
    if not crs.axis_info:
        return False
    return crs.axis_info[0].unit_name in {"metre", "meter", "m"}


__all__ = [
    "CRSCompatibility",
    "parse_crs",
    "describe_crs",
    "compare_crs",
    "is_metric",
]