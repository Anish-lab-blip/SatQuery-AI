"""SatQuery AI — Phase 6 LoRA injection, scoped to the language model.

THE HAZARD THIS MODULE EXISTS TO CLOSE (measured 2026-09-23)
------------------------------------------------------------
`AutoModelForImageTextToText` loads `Idefics3ForConditionalGeneration`
(507,482,304 parameters, top-level children `['model', 'lm_head']`). Enumerating
every submodule and grouping by name suffix gives:

    q_proj     44 hits   32 under model.text_model  +  12 under model.vision_model
    k_proj     44 hits   32 under model.text_model  +  12 under model.vision_model
    v_proj     44 hits   32 under model.text_model  +  12 under model.vision_model
    o_proj     32 hits   model.text_model only
    out_proj   12 hits   model.vision_model only (SigLIP attention output)
    gate_proj  32 hits   model.text_model only
    up_proj    32 hits   model.text_model only
    down_proj  32 hits   model.text_model only

So `q_proj`/`k_proj`/`v_proj` are **not unique to the language model**. The
SigLIP vision tower has 12 attention blocks using the same three names. Passing
the conventional `target_modules=["q_proj","k_proj","v_proj"]` would inject LoRA
into the vision encoder that plan section 43 requires to be **frozen**, and PEFT
would report success. The failure is silent: the run trains, the loss falls, and
the image encoder has drifted.

Two defences, both mandatory (contract items G and I):

  1. **Scope injection by regex.** `LANGUAGE_MODEL_TARGET_REGEX` is an anchored
     full-match pattern over the module key, and PEFT 0.21 matches a string
     `target_modules` with `re.fullmatch` (verified live against
     `peft.tuners.tuners_utils.match_target_against_key`). The vision tower
     cannot match it.
  2. **Verify injection by measurement.** `verify_injection` walks
     `named_parameters()` and raises if the trainable count is zero, or if any
     trainable parameter is outside the language-model subtree. A LoRA that
     matched nothing trains happily and learns nothing, which is the worst
     failure available here.

THE PEFT NAME-PREFIX CORRECTION (measured, and it changes the guard)
--------------------------------------------------------------------
The contract phrased the guard as "every trainable parameter name must begin
with `model.text_model.`". Measured, that literal check is **wrong once PEFT has
wrapped the model**: `get_peft_model` returns a `PeftModelForCausalLM` whose
parameter names carry a wrapper chain, e.g.

    base_model.model.model.text_model.layers.0.self_attn.q_proj.lora_A.default.weight

There are two extra `model.` segments relative to the unwrapped model
(`base_model` -> `LoraModel` -> `Idefics3ForConditionalGeneration` ->
`Idefics3Model` -> `text_model`). A literal `startswith("model.text_model.")`
would therefore reject **every** correctly-injected LoRA.

The guard is instead expressed against the stable `.text_model.` / `.vision_model.`
**subtree markers**, which do not move when PEFT changes its wrapper depth, and
`_canonical_module_name` reduces a wrapped name to the unwrapped form the
contract speaks (`model.text_model...`) so the report is legible. This is a
measured correction, not a relaxation: the substantive requirement -- nothing
outside the language model is trainable -- is enforced strictly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from core.errors import SatQueryError

__all__ = [
    "LANGUAGE_MODEL_TARGET_REGEX",
    "TEXT_MODEL_MARKER",
    "VISION_MODEL_MARKER",
    "CONNECTOR_MARKER",
    "LoRAError",
    "LoRAInjectionReport",
    "language_model_regex",
    "build_lora_config",
    "attach_lora",
    "verify_injection",
    "describe_frozen",
    "save_adapter",
    "load_adapter",
    "adapter_is_reloadable",
    "canonical_module_name",
]

#: The default projection set (contract item G), mirroring
#: `training.vlm.config.LORA_TARGET_MODULES`.
_DEFAULT_PROJECTIONS: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

#: Anchored full-match regex over the module key (contract item G).
#:
#: Anchored with `^...$` because PEFT applies `re.fullmatch`, and a suffix list
#: without the `model.text_model.` scope is exactly the vision-tower bug. The
#: projection set is the Llama-style attention + MLP block of the decoder.
#:
#: This is the canonical form for the default projection set. `build_lora_config`
#: derives the equivalent regex from `cfg.lora_target_modules` (which defaults to
#: the same set), so that config field is load-bearing rather than inert -- the
#: audit's row 3/4 complaint about declared-but-unread keys.
LANGUAGE_MODEL_TARGET_REGEX: str = (
    r"^model\.text_model\..*\.(q_proj|k_proj|v_proj|o_proj|"
    r"gate_proj|up_proj|down_proj)$"
)

#: Subtree markers. The guard keys off these rather than a fixed prefix depth so
#: it survives a PEFT release that changes its wrapper chain.
TEXT_MODEL_MARKER: str = ".text_model."
VISION_MODEL_MARKER: str = ".vision_model."
CONNECTOR_MARKER: str = ".connector."


def language_model_regex(target_modules: Sequence[str]) -> str:
    """Build the anchored language-model-scoped regex for a projection set.

    For the default set this is byte-identical to `LANGUAGE_MODEL_TARGET_REGEX`.
    Escaping each name means a projection containing a regex metacharacter cannot
    silently widen the match.
    """
    if not target_modules:
        raise LoRAError(
            "the LoRA projection set is empty; a regex built from it would match "
            "no module and produce a LoRA that trains without learning"
        )
    alternatives = "|".join(re.escape(name) for name in target_modules)
    return rf"^model\.text_model\..*\.({alternatives})$"


class LoRAError(SatQueryError):
    """A LoRA adapter could not be injected, verified, or reloaded."""

    code = "lora_error"
    user_message = "The Phase 6 LoRA adapter could not be applied."


# ---------------------------------------------------------------------------
# Canonical naming — undo PEFT's wrapper so names match the contract
# ---------------------------------------------------------------------------
def canonical_module_name(name: str) -> str:
    """Reduce a (possibly PEFT-wrapped) module name to the unwrapped form.

    `base_model.model.model.text_model.layers.0.self_attn.q_proj` becomes
    `model.text_model.layers.0.self_attn.q_proj`, which is the name the same
    module has on the unwrapped model and the name the contract's regex is
    written against. The first subtree marker is the anchor; everything before
    it is PEFT's wrapper and is dropped.
    """
    for marker in (
        "model.text_model.",
        "model.vision_model.",
        "model.connector.",
        "lm_head",
    ):
        index = name.find(marker)
        if index != -1:
            return name[index:]
    return name


def _subtree_of(name: str) -> str:
    """Classify a parameter/module name into its architectural subtree."""
    if VISION_MODEL_MARKER in name:
        return "model.vision_model"
    if CONNECTOR_MARKER in name:
        return "model.connector"
    if TEXT_MODEL_MARKER in name:
        return "model.text_model"
    return "other"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def build_lora_config(cfg: Any) -> Any:
    """Build the PEFT `LoraConfig` for the Phase 6 recipe (contract item G).

    `bias="none"` and `task_type="CAUSAL_LM"` are fixed: the former keeps the
    trainable-parameter count to the LoRA matrices alone (contract item H), the
    latter selects the decoder-only adapter wrapper for this VLM's language
    tower. `target_modules` is the **regex string**, not a suffix list -- see the
    module docstring for why a suffix list is the bug.
    """
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=language_model_regex(cfg.lora_target_modules),
    )


# ---------------------------------------------------------------------------
# The injection report
# ---------------------------------------------------------------------------
@dataclass
class LoRAInjectionReport:
    """What injection actually did, measured rather than asserted.

    Every field is read back from the live model. Nothing here is copied from
    the config: a config that says `r=16` and an injection that matched nothing
    are exactly the divergence this record exists to expose.
    """

    n_target_modules: int
    target_module_names: list[str]
    trainable_params: int
    total_params: int
    trainable_fraction: float
    trainable_subtrees: dict[str, int] = field(default_factory=dict)
    all_trainable_in_text_model: bool = False
    #: The distinct wrapper prefixes observed on trainable names, recorded so a
    #: PEFT release that changes its wrapper chain is visible in the manifest.
    observed_prefixes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_target_modules": self.n_target_modules,
            "target_module_names": list(self.target_module_names),
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "trainable_fraction": round(self.trainable_fraction, 8),
            "trainable_subtrees": dict(self.trainable_subtrees),
            "all_trainable_in_text_model": self.all_trainable_in_text_model,
            "observed_prefixes": list(self.observed_prefixes),
        }


def _trainable_prefix(name: str) -> str:
    """The wrapper prefix before the first architectural marker.

    `base_model.model.model.text_model.…` -> `base_model.model.model.`. Used to
    record the observed PEFT wrapper chain, not to drive the guard.
    """
    canonical = canonical_module_name(name)
    if canonical == name:
        return ""
    return name[: len(name) - len(canonical)]


# ---------------------------------------------------------------------------
# Injection + verification
# ---------------------------------------------------------------------------
def attach_lora(model: Any, cfg: Any) -> tuple[Any, LoRAInjectionReport]:
    """Wrap `model` with a language-model-scoped LoRA and verify the result.

    Returns `(peft_model, report)`. The report is produced by measuring the
    wrapped model, so a successful return is evidence that the injection landed
    in the language model and nowhere else.

    Raises:
        LoRAError: `get_peft_model` rejected the target modules. Measured
            2026-09-23: PEFT 0.21 raises `NoMatchingPeftModuleError` when the
            target regex matches no module at all, so a total miss surfaces here
            rather than in `verify_injection` -- both are converted to `LoRAError`
            so the failure mode is uniform. `verify_injection`'s zero-trainable
            guard remains the backstop for a partial match that yields none.
    """
    from peft import get_peft_model

    try:
        peft_model = get_peft_model(model, build_lora_config(cfg))
    except Exception as exc:  # noqa: BLE001 - PEFT raises a wide range
        raise LoRAError(
            f"LoRA injection failed: {type(exc).__name__}: {exc}. The target "
            f"regex is {LANGUAGE_MODEL_TARGET_REGEX!r}; a regex that matches no "
            f"module produces a LoRA that would train without learning.",
            context={"target_regex": LANGUAGE_MODEL_TARGET_REGEX},
        ) from exc
    report = verify_injection(peft_model)
    return peft_model, report


def verify_injection(model: Any) -> LoRAInjectionReport:
    """Measure an injected model and refuse the two silent failures.

    Raises:
        LoRAError: if no parameter is trainable (a LoRA that matched nothing
            trains happily and learns nothing), or if any trainable parameter
            lies outside the language-model subtree (the vision-tower guard of
            contract item I).
    """
    trainable = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for _, p in trainable)

    if trainable_params == 0:
        regex = _target_regex(model)
        raise LoRAError(
            "LoRA injection produced ZERO trainable parameters. A LoRA that "
            f"matched no module (target regex {regex!r}) "
            "trains without learning and is the worst available failure; the "
            "run is refused rather than allowed to write a meaningless adapter.",
            context={"target_regex": regex},
        )

    # The vision-tower guard (contract item I). Expressed against the subtree
    # marker, not a fixed prefix depth -- see the module docstring.
    offenders = [
        name for name, _ in trainable if TEXT_MODEL_MARKER not in name
    ]
    if offenders:
        raise LoRAError(
            f"{len(offenders)} trainable parameter(s) fall outside the language "
            f"model; plan section 43 freezes the vision encoder and the "
            f"connector/projector. First offender: {offenders[0]!r}",
            context={
                "n_offenders": len(offenders),
                "example": offenders[0],
            },
        )

    subtrees: dict[str, int] = {}
    for name, param in trainable:
        key = _subtree_of(name)
        subtrees[key] = subtrees.get(key, 0) + param.numel()

    prefixes = sorted({_trainable_prefix(n) for n, _ in trainable if _trainable_prefix(n)})

    target_names = sorted(
        canonical_module_name(name)
        for name, _ in model.named_modules()
        if _matches_target(canonical_module_name(name), _target_regex(model))
    )

    return LoRAInjectionReport(
        n_target_modules=len(target_names),
        target_module_names=target_names,
        trainable_params=trainable_params,
        total_params=total_params,
        trainable_fraction=(
            trainable_params / total_params if total_params else 0.0
        ),
        trainable_subtrees=subtrees,
        all_trainable_in_text_model=not offenders,
        observed_prefixes=prefixes,
    )


def _target_regex(model: Any) -> str:
    """The target regex actually stored on the injected model, if readable.

    Reading it from `peft_config` means the report reflects the regex that was
    really used, rather than a second copy that could drift from it. Falls back
    to the canonical constant for an un-wrapped model.
    """
    peft_config = getattr(model, "peft_config", None)
    if isinstance(peft_config, dict):
        for config in peft_config.values():
            target = getattr(config, "target_modules", None)
            if isinstance(target, str):
                return target
    return LANGUAGE_MODEL_TARGET_REGEX


def _matches_target(canonical_name: str, regex: str) -> bool:
    """Whether a canonical module name matches the language-model regex."""
    return re.fullmatch(regex, canonical_name) is not None


def describe_frozen(model: Any) -> dict[str, int]:
    """Frozen (non-trainable) parameter counts grouped by architectural subtree.

    Lets the run manifest prove the vision encoder and the connector were frozen
    rather than merely assert it: a non-zero frozen count under
    `model.vision_model` is the positive evidence that the image encoder is
    intact and untouched by the adapter.
    """
    out: dict[str, int] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            continue
        key = _subtree_of(name)
        out[key] = out.get(key, 0) + param.numel()
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_adapter(model: Any, out_dir: str | Path) -> Path:
    """Write the PEFT adapter to `out_dir`. Returns the directory."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target))
    return target


def load_adapter(base_model: Any, adapter_dir: str | Path) -> Any:
    """Reload an adapter onto `base_model` with `PeftModel.from_pretrained`.

    Raises:
        LoRAError: if the adapter cannot be loaded. A silent `None` return would
            make an unadapted model look adapted.
    """
    from peft import PeftModel

    directory = Path(adapter_dir)
    if not directory.exists():
        raise LoRAError(
            f"adapter directory does not exist: {directory}",
            context={"adapter_dir": str(directory)},
        )
    try:
        return PeftModel.from_pretrained(base_model, str(directory))
    except Exception as exc:  # noqa: BLE001 - PEFT raises a wide range
        raise LoRAError(
            f"could not reload the adapter at {directory}: {exc}",
            context={"adapter_dir": str(directory)},
        ) from exc


def adapter_is_reloadable(base_model: Any, adapter_dir: str | Path) -> bool:
    """Whether the adapter reloads cleanly. Never raises; returns a bool.

    Used by artifact verification (contract item V4): a non-reloadable adapter
    makes the run INCONCLUSIVE rather than accepted.
    """
    try:
        load_adapter(base_model, adapter_dir)
        return True
    except LoRAError:
        return False
