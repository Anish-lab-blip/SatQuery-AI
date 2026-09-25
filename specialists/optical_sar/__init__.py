"""SatQuery AI optical-SAR specialist (Workflow E).

CROMA-base optical/SAR/joint encoders, a sensor adapter that maps arbitrary
sensor bands onto CROMA's canonical channels, and the frozen fusion head.

    croma.py          the encoder — `PretrainedCROMA(size='base',
                      modality='both', image_resolution=120)`
    sensor_adapter.py band-map abstraction; canonical channels + availability
                      mask; the two hard rules about not inventing bands
    radiometry.py     the CROMA encoder-input stretch (DEV-2 ruling):
                      per-channel mean +/- 2*std -> [0, 1], unavailable channels
                      skipped and left at zero
    fusion_head.py    concat(GAPs, masks) -> (B, 2318); LayerNorm -> Linear ->
                      GELU -> Dropout -> Linear
    inference.py      the pipeline, and the degradation paths
    specialist.py     the `Specialist` subclass the controller dispatches to
    prompts.py        explanation templates (frozen, versioned)
    vendor/           upstream `use_croma.py` (see vendor/PROVENANCE.md)

Import explicitly:

    from specialists.optical_sar.specialist import (
        OpticalSarSpecialist, build_optical_sar_specialist,
    )
    from specialists.optical_sar.sensor_adapter import adapt_optical, adapt_sar
    from specialists.optical_sar.radiometry import normalise_for_croma

WHY THIS PACKAGE IS LIGHT AT PACKAGE LEVEL
------------------------------------------
`croma.py`, `fusion_head.py` and `inference.py` all reach for torch, and
`croma.py` additionally requires a vendored `use_croma.py` that may not exist in
a given deployment. A controller that imported every specialist at package load
would pay that cost, and take that dependency, on every request — defeating the
lazy-loading design in the plan's section 49.

So `specialist.py` is NOT re-exported here. Import it by its full path.

WHAT *IS* RE-EXPORTED
---------------------
The sensor adapter and the radiometry stage. Both are pure NumPy and pure logic:
no torch, no checkpoint, no network. That means the canonical-channel contract,
the availability mask, and the encoder-input transform — the facts the hidden
Cartosat-2S + RISAT evaluation depends on — are importable and testable in an
environment where none of the models exist. That is deliberate, not incidental:
the guarantees that matter most here are the ones that survive without the
weights.
"""

from specialists.optical_sar.radiometry import (
    ENV_USE_8_BIT,
    RadiometryReport,
    RadiometrySummary,
    normalise_for_croma,
    resolve_use_8_bit,
)
from specialists.optical_sar.sensor_adapter import (
    CARTOSAT_2S_BANDS,
    OPTICAL_CANONICAL,
    SAR_CANONICAL,
    SensorAdapterOutput,
    adapt_optical,
    adapt_sar,
    build_optical_adapter,
    canonical_to_sensor,
    describe_sar,
    positional_fallback_descriptor,
)

__all__ = [
    "OPTICAL_CANONICAL",
    "SAR_CANONICAL",
    "CARTOSAT_2S_BANDS",
    "SensorAdapterOutput",
    "build_optical_adapter",
    "describe_sar",
    "adapt_optical",
    "adapt_sar",
    "positional_fallback_descriptor",
    "canonical_to_sensor",
    "ENV_USE_8_BIT",
    "RadiometryReport",
    "RadiometrySummary",
    "normalise_for_croma",
    "resolve_use_8_bit",
]
