"""SatQuery AI — the public serving entry point.

WHAT THIS IS FOR
----------------
`docs/ARCHITECTURE_FREEZE.md` section 5 puts the control tier in
`core.controller.AnalysisController`, driven by a
`core.registry.SpecialistRegistry`. This module is the thin, public composition
root that wires a *deployable* controller: it resolves the trained change head
and routes it to the change specialist — WITHOUT editing `configs/base.yaml`.

It does the same for the R-02 change-VQA specialist (`change_vqa`), so that the
STANet detector backing a change ANSWER is the same trained artifact that
`scripts/prepare_change_vqa.py` used to build the head's training features. See
`_wired_change_vqa_builder` for the train/serve-skew finding this closes.

It also does the same for `optical_sar`, resolving the CROMA encoder from its
PINNED identity rather than from a config key. That capability was DEGRADED by
default for a purely structural reason: the specialist builds CROMA only when
handed a `checkpoint_path` that exists, `croma.checkpoint_path` is not in
`configs/base.yaml` (and must not be, or `Config.hash` moves), so no path ever
arrived and the encoder was never loaded — while the artifact sat on disk the
whole time. See `_wired_optical_sar_builder`.

WHY THE CHECKPOINT IS NOT WIRED VIA CONFIG
------------------------------------------
`Config.hash` (`core/config.py:79-80`) is a sha256 over the WHOLE registry, so
adding one key moves the hash. The shipped head records `78f1e3700da15aa1` in
`artifacts/change/levir_change_v001/model_metadata.json`, and
`scripts/eval_change.py` refuses to score on a hash drift (exit 3). Editing the
config to point at the head would therefore invalidate the project's own
benchmark number.

The supported wiring path is the registry's `builders=` override
(`core/registry.py:420-433`): the override is keyed by the spec name — `"change"`
(`core/registry.py:204-213`) — and is a CALL-SITE argument, not config, so the
hash is untouched. That override is applied here.

DEGRADE, DO NOT CRASH
---------------------
A serving path must run even when the artifact is absent. When the checkpoint
does not exist this module applies NO override, so the registry resolves the
real builder with no `checkpoint_path` and the documented DEGRADED contract
applies (`specialists/change/specialist.py`). Absent is a deployment case;
*corrupt* is a defect, and the real builder still surfaces it as
`ModelLoadError` — the two are deliberately not conflated.

The `change_vqa` override is applied unconditionally, because its builder's
contract is finer-grained than the change specialist's: a missing head and a
missing detector are both *named* refusals (`ChangeVQASpecialist.has_head`,
`unavailable_reason`), so wiring it can never turn "absent" into a crash. Both
paths are passed as `None` when the file does not exist, which the builder reads
as "artifact genuinely absent" rather than "path I was told about is broken".
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.config import REPO_ROOT, load_config
from core.controller import AnalysisController
from core.planner import PolicyPlanner
from core.registry import SpecialistRegistry

_log = logging.getLogger(__name__)

#: The trained, benchmarked change head (test pooled IoU 0.8122). The single
#: source of truth for where serving looks for it; tests monkeypatch this to
#: simulate an absent artifact. It is also the detector whose features
#: `scripts/prepare_change_vqa.py` builds, so training and serving share it.
#:
#: The two literals are kept equal by
#: `tests/unit/test_app_serving.py::test_serving_and_preparation_share_one_stanet_checkpoint`,
#: which fails if either side drifts.
CHANGE_CHECKPOINT: Path = (
    REPO_ROOT / "artifacts" / "change" / "levir_change_v001" / "head.pt"
)

#: The R-02 change-VQA reasoning head. Written by
#: `scripts/train_change_vqa.py` (`--output-dir`, default
#: `artifacts/change_vqa/run`) and read by `scripts/evaluate_change_vqa.py`
#: (`DEFAULT_CHECKPOINT`, the same path). It does NOT exist in a fresh checkout:
#: it is produced by the external Kaggle run and returned to the maintainer —
#: see `docs/R02_KAGGLE_TRAINING_GUIDE.md`. Absent => the specialist constructs,
#: reports itself unavailable, and answers nothing.
CHANGE_VQA_HEAD: Path = (
    REPO_ROOT / "artifacts" / "change_vqa" / "run" / "head.pt"
)

#: The verified production optical-SAR fusion head (Phase 12). Its identity is
#: pre-registered, not inferred: sha256
#: `785815729a3a39fc34dc41894efaf00d8739365d970a3f830a326e68ae888dab`,
#: 14,427,457 bytes, 1,201,711 parameters, and the checkpoint self-identifies
#: with the embedded `config_hash` `78f1e3700da15aa1` and `arm='A'`.
#:
#: It is a REPO-LOCAL artifact, not a config key, so wiring it here leaves
#: `Config.hash` bit-identical — the same reason the change head is wired
#: through the `builders=` seam rather than through `configs/base.yaml`.
FUSION_HEAD: Path = (
    REPO_ROOT / "artifacts" / "optical_sar" / "fusion_head_production_v001" / "head.pt"
)


def _wired_change_builder(config: Any, **kwargs: Any) -> Any:
    """The documented seam: inject the checkpoint, then delegate.

    The registry calls a builder as `builder(self.config, **kwargs)` and only
    passes config keys that resolve (`_builder_kwargs`, `core/registry.py:435-453`).
    `change.checkpoint_path` is unset, so nothing arrives and the real builder
    would degrade — this override supplies the missing argument.

    The real builder is imported lazily so importing `app.serving` stays cheap
    and does not pull the model stack (torch) into a process that never serves a
    change query.
    """
    from specialists.change.specialist import build_change_specialist

    kwargs["checkpoint_path"] = str(CHANGE_CHECKPOINT)
    return build_change_specialist(config, **kwargs)


def _wired_change_vqa_builder(config: Any, **kwargs: Any) -> Any:
    """The same seam, for the R-02 change-VQA specialist (finding F2).

    WHY THIS EXISTS — TRAIN/SERVE SKEW
    ----------------------------------
    `scripts/prepare_change_vqa.py` builds its change features from the TRAINED
    STANet (`DEFAULT_CHANGE_CHECKPOINT`). Serving had no equivalent wiring:
    `change.checkpoint_path` is unset in `configs/base.yaml`, so the registry
    passes no `checkpoint_path` and the specialist would construct an UNTRAINED
    STANet and answer from a representation the head was never fitted on.

    `ChangeVQASpecialist.feature_spec_mismatch()` already refuses to answer on
    that skew, so the failure was loud rather than silent — but a deployment
    that can never answer is still not a deployment. This override supplies the
    SAME checkpoint `_wired_change_builder` uses, so the detector backing a
    change answer and the detector behind the head's training features are one
    artifact by construction; the spec check stays armed as the second line of
    defence, not the only one.

    Absent files are passed as `None` on purpose: the builder's contract is that
    a missing artifact degrades while a CORRUPT one raises `ModelLoadError`.
    Handing it a path that does not exist would conflate the two.
    """
    from specialists.change.vqa_specialist import build_change_vqa_specialist

    kwargs["checkpoint_path"] = (
        str(CHANGE_CHECKPOINT) if CHANGE_CHECKPOINT.exists() else None
    )
    kwargs["head_path"] = str(CHANGE_VQA_HEAD) if CHANGE_VQA_HEAD.exists() else None
    return build_change_vqa_specialist(config, **kwargs)


def _wired_optical_sar_builder(config: Any, **kwargs: Any) -> Any:
    """The same seam, for the optical-SAR specialist (Workflow E).

    WHY THIS EXISTS — THE ENCODER WAS UNREACHABLE BY DEFAULT
    -------------------------------------------------------
    `specialists/optical_sar/specialist.py:946` builds CROMA only when it is
    handed a `checkpoint_path` that exists:

        if checkpoint_path is not None and Path(checkpoint_path).exists():

    `croma.checkpoint_path` is not in `configs/base.yaml` — and must not be, or
    `Config.hash` moves — so the registry passed no `checkpoint_path`, the gate
    was False, and the default serving composition ran with `encoder=None`. The
    registry then correctly reported DEGRADED ("no encoder; running on
    fallback"): a deployment that could never answer an optical/SAR question.
    The checkpoint was present on disk the whole time; nothing asked for it.

    WHAT THIS DOES
    --------------
    Resolves the checkpoint from the PINNED identity through the hash-exempt
    channel (`croma.resolve_checkpoint_path`: env -> config -> pinned Hub
    cache, offline first), then hands it to the real builder. The resolution is
    recorded as `source` so "which checkpoint did this process actually use" is
    answerable from the trace rather than from an assumption.

    Absent is still absent: the resolver returns `None` when no artifact exists,
    and the builder reads that as "genuinely absent" and degrades. It is a
    *specified but missing* path that raises, matching `build_encoder`'s
    existing contract — the two cases are deliberately not conflated.

    The fusion head is wired the same way, for the same reason: `has_head`
    (`specialist.py:175-187`) is False without it, so the capability would stay
    DEGRADED even with the encoder loaded. Absent head => `None` => degrade.
    """
    from specialists.optical_sar.croma import resolve_checkpoint_path
    from specialists.optical_sar.specialist import build_optical_sar_specialist

    checkpoint, source = resolve_checkpoint_path(config)
    kwargs["checkpoint_path"] = checkpoint
    kwargs["head_path"] = str(FUSION_HEAD) if FUSION_HEAD.exists() else None

    # "Which channel answered" is a composition-time fact, so it is LOGGED
    # rather than attached to the returned specialist. An unread attribute on a
    # production object is how a contract quietly grows a second, undocumented
    # shape; and it must not be published either, because the trace reaches the
    # client and v1 has no auth. The path itself is already carried by the
    # encoder and reduced to a basename at the client seam
    # (`specialist.py::model_refs`).
    _log.info(
        "optical_sar: CROMA checkpoint resolved via %s (%s)",
        source,
        "present" if checkpoint else "absent",
    )
    return build_optical_sar_specialist(config, **kwargs)


def build_serving_registry(
    config: Any | None = None,
    *,
    device: str | None = None,
) -> SpecialistRegistry:
    """Build the serving registry, wiring the change head when it is present.

    Args:
        config: the central `core.config.Config`. Loaded unchanged when omitted;
            never mutated.
        device: torch device string; defaults to `config.device_preference`.

    Returns:
        A `SpecialistRegistry` that constructs nothing yet (`discover()` reads a
        spec table). The `change` capability is wired to the trained head when
        `CHANGE_CHECKPOINT` exists; otherwise no override is applied and the
        DEGRADED contract governs. `change_vqa` is always wired through the same
        seam, so it shares that checkpoint and never silently falls back to an
        untrained detector.
    """
    cfg = load_config() if config is None else config
    builders: dict[str, Any] = {
        "change_vqa": _wired_change_vqa_builder,
        # Registered UNCONDITIONALLY, unlike `change` below. The resolver decides
        # at build time whether an artifact exists, so gating registration on a
        # path this module does not know yet would be circular. Wiring it cannot
        # turn "absent" into a crash: the resolver returns None, the builder
        # degrades, and a construction failure is retained as an UNAVAILABLE
        # entry by `SpecialistRegistry.build` rather than escaping.
        "optical_sar": _wired_optical_sar_builder,
    }
    if CHANGE_CHECKPOINT.exists():
        builders["change"] = _wired_change_builder
    return SpecialistRegistry.discover(cfg, device=device, builders=builders)


def build_serving_controller(
    config: Any | None = None,
    *,
    device: str | None = None,
) -> AnalysisController:
    """Build the serving controller over the serving registry.

    Constructed with `registry=`, `planner=` and `config=` only. No router is
    attached, so a caller drives it with `AnalysisRequest(..., force_task=...)`;
    a natural-language router can be supplied by the caller's own composition if
    the router weights are available.
    """
    cfg = load_config() if config is None else config
    registry = build_serving_registry(cfg, device=device)
    return AnalysisController(
        registry=registry,
        planner=PolicyPlanner(registry),
        config=cfg,
    )


__all__ = [
    "CHANGE_CHECKPOINT",
    "CHANGE_VQA_HEAD",
    "FUSION_HEAD",
    "build_serving_controller",
    "build_serving_registry",
]
