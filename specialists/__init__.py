"""SatQuery AI specialists.

Every specialist implements the `Specialist` interface from `specialists.base`
and returns the common `core.schemas.SpecialistResult`. The controller does not
care whether a specialist is a local object, a model, or a future remote worker.

Deliberately light: this package exports the interface only. Concrete
specialists live in their own subpackages and are imported explicitly, because
importing `specialists.vqa.inference` pulls in the VLM loader, and importing
`specialists.optical_sar.*` pulls in CROMA. A controller that imported all of
them at package load would pay every model's import cost for every request,
which defeats the lazy-loading design (plan section 49).

Import a specialist directly:

    from specialists.vqa.inference import VLMSpecialist, build_vqa_specialist

Not from the package root.
"""

from specialists.base import Specialist, SpecialistRequest

__all__ = ["Specialist", "SpecialistRequest"]