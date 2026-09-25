"""SatQuery AI core: configuration, schemas, errors, controller, planner, registry.

The control tier (freeze section 6) is `core/registry.py`, `core/planner.py`
and `core/controller.py`. All three are re-exported here, and all three are
torch-free at import: the registry holds a table of strings and constructs
through `importlib`, the planner is pure, and the controller imports the
evidence engine lazily. Adding an eager specialist import here would break that
and pay every model's import cost on every request.
"""

from core.config import Config, get_config, load_config
from core.controller import AnalysisController, StepOutcome
from core.errors import SatQueryError
from core.planner import (
    ExecutionPlan,
    PlanMode,
    PlanRefusal,
    PlanStep,
    PolicyPlanner,
)
from core.registry import (
    RegistryEntry,
    RegistryState,
    SpecialistRegistry,
    SpecialistSpec,
)

__all__ = [
    "AnalysisController",
    "Config",
    "ExecutionPlan",
    "PlanMode",
    "PlanRefusal",
    "PlanStep",
    "PolicyPlanner",
    "RegistryEntry",
    "RegistryState",
    "SatQueryError",
    "SpecialistRegistry",
    "SpecialistSpec",
    "StepOutcome",
    "get_config",
    "load_config",
]
