"""SatQuery AI change-detection specialist (Workflow D).

STANet-style Siamese change detection over a bi-temporal image pair, plus
deterministic post-processing and registration measurement.

    stanet.py       the model — shared ResNet18 encoder, spatial-temporal
                    attention, feature-difference aggregation decoder
    postprocess.py  probability map -> regions; phase-correlation alignment
    specialist.py   the `Specialist` subclass the controller dispatches to

Import explicitly:

    from specialists.change.stanet import STANetStyleChangeDetector
    from specialists.change.postprocess import postprocess_change_map
    from specialists.change.specialist import (
        ChangeSpecialist, build_change_specialist,
    )

Deliberately light at package level. `stanet.py` pulls in torch; a controller
that imported every specialist at package load would pay that cost for every
request, defeating the lazy-loading design. `specialist.py` is therefore NOT
re-exported here -- it imports torch transitively through the detector, and
re-exporting it would pull that cost back in for every importer of this
package. Import it by its full path.
"""

from specialists.change.postprocess import (
    PostprocessResult,
    RegionStats,
    RegistrationQuality,
    connected_regions,
    measure_registration,
    morphological_cleanup,
    postprocess_change_map,
    regions_to_schema,
    to_grayscale_float,
)

__all__ = [
    "PostprocessResult",
    "RegionStats",
    "RegistrationQuality",
    "to_grayscale_float",
    "measure_registration",
    "morphological_cleanup",
    "connected_regions",
    "postprocess_change_map",
    "regions_to_schema",
]