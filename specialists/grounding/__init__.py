"""SatQuery AI grounding specialist.

RemoteCLIP ViT-B/32 provides the image-text alignment; a learned head
(Phase 8) provides the boxes. This package is deliberately split so the
encoder can be measured before the head exists:

    remoteclip.py   frozen encoder, contract asserted at load time
    inference.py    zero-shot patch-text baseline (ablation floor, not product)

Import explicitly:

    from specialists.grounding.remoteclip import build_encoder
    from specialists.grounding.inference import ground_phrase

Deliberately light at package level. Importing `remoteclip` pulls in torch and
open_clip; a controller that imported every specialist at package load would
pay that cost for every request, defeating the lazy-loading design.
"""

from specialists.grounding.remoteclip import (
    SUPPORTED_RESOLUTIONS,
    VERIFIED_PROJECTED_DIM,
    EncodedImage,
    RemoteCLIPEncoder,
)

__all__ = [
    "RemoteCLIPEncoder",
    "EncodedImage",
    "SUPPORTED_RESOLUTIONS",
    "VERIFIED_PROJECTED_DIM",
]