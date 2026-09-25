"""SatQuery AI — the public application package.

`app.serving` is the composition root that wires a deployable controller. It is
deliberately thin: it applies the registry `builders=` override that routes the
trained change head to the change specialist, the same trained STANet to the
R-02 change-VQA specialist (`change_vqa`) so an answer and the head's training
features rest on one detector artifact, and the verified CROMA encoder plus the
Phase 12 fusion head to `optical_sar`. All policy, mechanism and computation
stay in `core/` and `specialists/`.
"""

from app.serving import (
    CHANGE_CHECKPOINT,
    CHANGE_VQA_HEAD,
    FUSION_HEAD,
    build_serving_controller,
    build_serving_registry,
)

__all__ = [
    "CHANGE_CHECKPOINT",
    "CHANGE_VQA_HEAD",
    "FUSION_HEAD",
    "build_serving_controller",
    "build_serving_registry",
]
