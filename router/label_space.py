"""SatQuery AI — intent router label space.

Single source of truth for the router's output ontology. Both the dataset
generator and the training script import from here, so a class added in one
place cannot silently desynchronise the other.

Five heads (plan section 10):

    task            6 classes   what workflow to run
    modality        4 classes   what sensors are required
    temporal        binary      does it need two acquisitions
    spatial_output  binary      must the answer contain coordinates
    language_output binary      must the answer contain prose

`task` and `modality` use softmax cross-entropy. The three binary heads use a
single logit with BCEWithLogitsLoss — a 2-way softmax would waste a parameter
and make the loss harder to weight.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Task head
# ---------------------------------------------------------------------------

TASK_CLASSES: tuple[str, ...] = (
    "vqa",
    "caption",
    "grounding",
    "change",
    "optical_sar",
    "unsupported",
)
TASK_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(TASK_CLASSES)}
NUM_TASKS: int = len(TASK_CLASSES)

#: Tasks that inherently require two acquisitions.
TEMPORAL_TASKS: frozenset[str] = frozenset({"change"})

#: Tasks that inherently require two modalities.
DUAL_MODALITY_TASKS: frozenset[str] = frozenset({"optical_sar"})

#: Tasks whose primary output is spatial rather than textual.
SPATIAL_TASKS: frozenset[str] = frozenset({"grounding"})

# ---------------------------------------------------------------------------
# Modality head
# ---------------------------------------------------------------------------

MODALITY_CLASSES: tuple[str, ...] = (
    "optical",
    "sar",
    "optical_sar",
    "unknown",
)
MODALITY_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(MODALITY_CLASSES)}
NUM_MODALITIES: int = len(MODALITY_CLASSES)

# ---------------------------------------------------------------------------
# Binary heads
# ---------------------------------------------------------------------------

BINARY_HEADS: tuple[str, ...] = ("temporal", "spatial_output", "language_output")
NUM_BINARY_HEADS: int = len(BINARY_HEADS)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def task_index(name: str) -> int:
    """Index of a task class. Raises KeyError on an unknown label."""
    return TASK_TO_INDEX[name]


def modality_index(name: str) -> int:
    """Index of a modality class. Raises KeyError on an unknown label."""
    return MODALITY_TO_INDEX[name]


def is_valid_task(name: str) -> bool:
    return name in TASK_TO_INDEX


def is_valid_modality(name: str) -> bool:
    return name in MODALITY_TO_INDEX


def default_attributes(task: str) -> dict[str, bool]:
    """Per-task attribute defaults, used to sanity-check generated examples.

    These are *defaults*, not invariants: `change` with `spatial_output=False`
    is a perfectly legal request ("what changed?"). They exist so the dataset
    generator and the fallback agree on what a task usually implies.
    """
    return {
        "temporal": task in TEMPORAL_TASKS,
        "spatial_output": task in SPATIAL_TASKS,
        "language_output": task != "unsupported",
    }


def default_modality(task: str) -> str:
    if task in DUAL_MODALITY_TASKS:
        return "optical_sar"
    if task == "unsupported":
        return "unknown"
    return "unknown"


__all__ = [
    "TASK_CLASSES",
    "TASK_TO_INDEX",
    "NUM_TASKS",
    "TEMPORAL_TASKS",
    "DUAL_MODALITY_TASKS",
    "SPATIAL_TASKS",
    "MODALITY_CLASSES",
    "MODALITY_TO_INDEX",
    "NUM_MODALITIES",
    "BINARY_HEADS",
    "NUM_BINARY_HEADS",
    "task_index",
    "modality_index",
    "is_valid_task",
    "is_valid_modality",
    "default_attributes",
    "default_modality",
]