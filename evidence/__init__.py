"""SatQuery AI — evidence engine and confidence calibration (Phase 13).

`docs/ARCHITECTURE_FREEZE.md` section 6 places this package between the
specialists and the result normaliser. Its two modules do two distinct jobs:

    engine.py       aggregate, order, deduplicate and cite specialist evidence
    confidence.py   map a raw specialist score through a fitted calibration,
                    or degrade honestly when none exists

Typical use from the controller:

    from evidence import EvidenceEngine, load_calibration

    engine = EvidenceEngine.from_config(
        config, calibration=load_calibration(config, base_dir=".")
    )
    collection = engine.aggregate([vqa_result, grounding_result])
    trace.confidence = engine.confidence_for(grounding_result)

The public surface is re-exported here so callers never import a submodule path
directly; that keeps a future split of these two modules from rippling through
the codebase.
"""

from __future__ import annotations

from evidence.confidence import (
    METHOD_TEMPERATURE,
    METHOD_UNCALIBRATED,
    TemperatureCalibration,
    calibrate,
    calibrate_result,
    load_calibration,
)
from evidence.engine import (
    DEFAULT_MAX_ITEMS,
    ID_PREFIX,
    EvidenceCollection,
    EvidenceEngine,
    aggregate_evidence,
    evidence_digest,
)

__all__ = [
    # engine
    "EvidenceEngine",
    "EvidenceCollection",
    "aggregate_evidence",
    "evidence_digest",
    "DEFAULT_MAX_ITEMS",
    "ID_PREFIX",
    # confidence
    "TemperatureCalibration",
    "calibrate",
    "calibrate_result",
    "load_calibration",
    "METHOD_TEMPERATURE",
    "METHOD_UNCALIBRATED",
]
