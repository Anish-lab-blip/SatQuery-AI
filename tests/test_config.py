"""Phase 1 tests — configuration registry and frozen contract guards.

Every Phase-0 finding that the loader claims to enforce must actually reject a bad
config. A guard that only exists in prose is not a guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import Config, ConfigError, load_config
from core.errors import WorkflowPlanError

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------
def test_default_config_loads() -> None:
    cfg = load_config()
    assert cfg.get("project.name") == "satquery-ai"
    assert cfg.get("croma.encoder_dim") == 768


def test_config_hash_is_stable() -> None:
    a = load_config()
    b = load_config()
    assert a.hash == b.hash
    assert len(a.hash) == 16


def test_hash_changes_when_config_changes() -> None:
    base = load_config()
    tweaked = load_config(overrides={"fusion": {"hidden_dim": 1024}})
    assert base.hash != tweaked.hash


#: The config hash the shipped change head was trained under. Recorded in
#: `artifacts/change/levir_change_v001/model_metadata.json`.
FROZEN_CONFIG_HASH = "78f1e3700da15aa1"


def test_config_hash_matches_the_shipped_checkpoint_record() -> None:
    """The hash is a frozen CONTRACT, not merely a stability property.

    `test_config_hash_is_stable` only checks `a.hash == b.hash`; it never pins
    the VALUE. But the value is load-bearing: `scripts/eval_change.py` reads
    `config_hash` from `artifacts/change/levir_change_v001/model_metadata.json`
    and REFUSES TO SCORE when the current hash drifts (exit 3). A well-meaning
    config edit therefore silently invalidates the project's benchmark.

    The specific temptation this guards: populating `change.checkpoint_path` in
    `base.yaml` to wire the trained change head into serving. That edit moves
    the hash away from the value above and breaks the drift check.

    Do NOT quote a single "drifted value" -- it does not exist. `Config.hash`
    hashes the whole `_data` registry, so the moved hash depends on the exact
    path STRING written in, and the same logical change yields many values
    (measured, `.scratch/probe_hash_path_rendering.py`):

        backslashes / `Path` object   -> f4487e1a2cf13733
        forward slashes               -> f571a5f85372ff54
        relative                      -> b1d8637d6ab733d9
        bare filename                 -> 4fcc6a4a4f681c87
        directory with trailing slash -> f5a6ec3db8217218

    Only the UNMODIFIED hash above is stable and quotable. The SUPPORTED wiring
    path is the registry `builders=` override (see `core/registry.py`), which
    injects the checkpoint at the call site and leaves this hash bit-identical.
    """
    assert load_config().hash == FROZEN_CONFIG_HASH, (
        "config hash drifted from the value the shipped change head was trained "
        f"under ({FROZEN_CONFIG_HASH!r}). If the change was deliberate, do NOT "
        "just update this constant: re-record the config_hash in "
        "artifacts/change/levir_change_v001/model_metadata.json and re-run "
        "scripts/eval_change.py -- the old benchmark number is only reproducible "
        "under the config it was measured with."
    )


def test_dotted_access_and_require() -> None:
    cfg = load_config()
    assert cfg.get("change.tile_size") == 256
    assert cfg.require("project.seed") == 42
    with pytest.raises(ConfigError):
        cfg.require("does.not.exist")


def test_seed_is_int() -> None:
    assert load_config().seed == 42


# --------------------------------------------------------------------------
# C-1 — the mask goes to the fusion head, and the dim must match CROMA's output
# --------------------------------------------------------------------------
def test_fusion_dim_matches_verified_croma_concatenation() -> None:
    cfg = load_config()
    expected = (len(cfg.get("croma.modalities_used")) * cfg.get("croma.encoder_dim")
                + cfg.get("croma.optical_channels")
                + cfg.get("croma.sar_channels"))
    assert expected == 2318
    assert cfg.get("fusion.input_dim") == expected


def test_fusion_dim_mismatch_is_rejected() -> None:
    with pytest.raises(WorkflowPlanError):
        load_config(overrides={"fusion": {"input_dim": 999}})


def test_croma_channel_counts_are_pinned() -> None:
    with pytest.raises(WorkflowPlanError):
        load_config(overrides={"croma": {"optical_channels": 8}})
    with pytest.raises(WorkflowPlanError):
        load_config(overrides={"croma": {"sar_channels": 3}})


# --------------------------------------------------------------------------
# C-7 — CROMA asserts image_resolution % 8 == 0
# --------------------------------------------------------------------------
def test_croma_resolution_must_be_multiple_of_8() -> None:
    with pytest.raises(WorkflowPlanError):
        load_config(overrides={"croma": {"image_resolution": 121}})


def test_croma_native_resolution_is_valid() -> None:
    cfg = load_config()
    assert cfg.get("croma.image_resolution") == 120
    assert cfg.get("croma.image_resolution") % 8 == 0


# --------------------------------------------------------------------------
# C-3 — the VLM processor must not upscale our 512 px tiles
# --------------------------------------------------------------------------
def test_processor_longest_edge_must_be_positive_int() -> None:
    with pytest.raises(WorkflowPlanError):
        load_config(overrides={"vlm": {"processor_longest_edge": 0}})


def test_processor_must_not_upscale_tiles() -> None:
    """Finding F5-2, measured: the processor default is 2048.

    A 512 px tile is upscaled 4x and then split by do_image_splitting into
    4x4 sub-images + 1 overview = 17 images and 1142 prompt tokens, versus 1
    image when pinned. The plan estimated a 4x overrun; the real figure is ~17x.
    """
    cfg = load_config()
    assert cfg.get("vlm.processor_longest_edge") <= cfg.get("image.tile_size")


def test_processor_upscaling_is_rejected() -> None:
    """Prove the guard fires: an edge above the tile size is refused at load."""
    with pytest.raises(WorkflowPlanError):
        load_config(overrides={"vlm": {"processor_longest_edge": 4096}})


def test_processor_pin_is_tied_to_tile_size() -> None:
    """The pin is a control, not a coincidence: it equals the tile size."""
    cfg = load_config()
    assert cfg.get("vlm.processor_longest_edge") == cfg.get("image.tile_size")


def test_chat_template_is_required() -> None:
    """Finding F5-3: SmolVLM raises ValueError without one <image> token per
    image, so the chat-template path must not be disableable."""
    with pytest.raises(WorkflowPlanError):
        load_config(overrides={"vlm": {"prompt_must_use_chat_template": False}})


# --------------------------------------------------------------------------
# C-6 / C-8 — accelerator and deployment guards
# --------------------------------------------------------------------------
def test_precision_must_be_known() -> None:
    with pytest.raises(WorkflowPlanError):
        load_config(overrides={"training": {"precision": "tf32"}})


def test_precision_defaults_to_fp16_for_t4() -> None:
    # T4 is SM 7.5: bf16 tensor cores do not exist there.
    assert load_config().get("training.precision") == "fp16"


def test_torch_compile_is_forbidden_on_zerogpu() -> None:
    with pytest.raises(WorkflowPlanError):
        load_config(overrides={"deployment": {"torch_compile": True}})


def test_cpu_mode_is_required() -> None:
    assert load_config().get("deployment.cpu_mode_required") is True


# --------------------------------------------------------------------------
# Router ontology
# --------------------------------------------------------------------------
def test_router_tasks_include_unsupported() -> None:
    assert "unsupported" in load_config().get("router.tasks")


def test_router_num_tasks_matches_list() -> None:
    cfg = load_config()
    assert cfg.get("router.num_tasks") == len(cfg.get("router.tasks"))


def test_router_missing_unsupported_is_rejected() -> None:
    with pytest.raises(WorkflowPlanError):
        load_config(overrides={"router": {"tasks": ["vqa", "caption"]}})


# --------------------------------------------------------------------------
# Tiling policy
# --------------------------------------------------------------------------
def test_top_k_cannot_exceed_max_tiles() -> None:
    with pytest.raises(WorkflowPlanError):
        load_config(overrides={"image": {"top_k_tiles": 999}})


def test_tile_overlap_smaller_than_tile_size() -> None:
    cfg = load_config()
    assert cfg.get("image.tile_overlap") < cfg.get("image.tile_size")
    assert cfg.get("change.tile_overlap") < cfg.get("change.tile_size")


# --------------------------------------------------------------------------
# Change detection — verified upstream hyperparameters
# --------------------------------------------------------------------------
def test_change_hyperparameters_match_verified_stanet() -> None:
    cfg = load_config()
    assert cfg.get("change.learning_rate") == 0.001
    assert cfg.get("change.batch_size") == 8
    assert cfg.get("change.tile_size") == 256
    assert cfg.get("change.bce_weight") + cfg.get("change.dice_weight") == pytest.approx(1.0)


def test_change_sa_mode_is_valid() -> None:
    assert load_config().get("change.sa_mode") in {"BAM", "PAM"}


def test_change_sa_mode_rejects_junk() -> None:
    with pytest.raises(WorkflowPlanError):
        load_config(overrides={"change": {"sa_mode": "XYZ"}})


# --------------------------------------------------------------------------
# Grounding — C-5
# --------------------------------------------------------------------------
def test_grounding_resolution_is_frozen_at_224() -> None:
    """Phase 7 decided 224 over 448 on 16,159 records.

    Evidence: docs/PHASE7_RESOLUTION_DECISION.md. 448 was worse on mean best
    IoU (-0.0147), worse at every recall threshold, and 1.59x the latency,
    with a paired 95% CI of [-0.0160, -0.0134]. The config asserts the freeze
    rather than leaving a flag that implies the question is still open.
    """
    cfg = load_config()
    assert cfg.get("grounding.image_size") == 224
    assert cfg.get("grounding.resolution_frozen") is True


def test_resolution_experiment_flag_is_gone() -> None:
    """`allow_resolution_experiment` said "not yet decided". It is decided.

    A stale flag is worse than no flag: it invites someone to re-run a settled
    experiment, or to read the config and conclude the resolution is unfrozen.
    """
    assert load_config().get("grounding.allow_resolution_experiment") is None


# --------------------------------------------------------------------------
# Grounding head — Phase 8 feature contract
# --------------------------------------------------------------------------
def test_head_feature_dim_matches_the_frozen_encoder() -> None:
    """Per-cell feature = 4 x projected_dim. Encoder projects to 512, so 2048.

    A mismatch here is a SILENT shape error at the similarity step, after the
    patch features have already been computed and cached.
    """
    from specialists.grounding.remoteclip import VERIFIED_PROJECTED_DIM

    cfg = load_config()
    assert cfg.get("grounding_head.feature_dim") == 4 * VERIFIED_PROJECTED_DIM


def test_head_feature_dim_mismatch_is_rejected() -> None:
    """Prove the guard fires, rather than trusting the comment that says it does."""
    with pytest.raises(WorkflowPlanError):
        load_config(overrides={"grounding_head": {"feature_dim": 768}})


def test_head_positive_confidence_weight_is_set() -> None:
    """Objectness sees 1 positive cell out of 49.

    Unweighted mean BCE is minimised by predicting "no object" everywhere —
    the classic single-stage-detector collapse. This weight is what stops it.
    """
    assert load_config().get("grounding_head.positive_confidence_weight") > 1.0


def test_grounding_training_exposes_run_protocol_keys() -> None:
    """Section 45: every training script supports seed/resume/max-steps."""
    cfg = load_config()
    for key in ("epochs", "batch_size", "learning_rate", "weight_decay",
                "grad_clip", "val_fraction", "warmup_ratio", "save_every_steps"):
        assert cfg.get(f"grounding_training.{key}") is not None, \
            f"missing grounding_training.{key}"


def test_vrsbench_box_scale_is_declared() -> None:
    # VRSBench normalises boxes to 0-100; we store 0-1.
    assert load_config().get("grounding.benchmark_box_scale") == 100.0


# --------------------------------------------------------------------------
# Evaluation isolation
# --------------------------------------------------------------------------
def test_hidden_data_access_is_disabled() -> None:
    assert load_config().get("evaluation.hidden_data_access") is False


def test_no_official_aggregate_weights_are_invented() -> None:
    assert load_config().get("evaluation.official_aggregate_weights") is None


def test_leakage_split_key_is_scene_level() -> None:
    assert load_config().get("evaluation.leakage_split_key") == "scene_id"


# --------------------------------------------------------------------------
# Config object behaviour
# --------------------------------------------------------------------------
def test_as_dict_roundtrips_to_json_safe_types() -> None:
    import json

    payload = load_config().as_dict()
    json.dumps(payload)  # must not raise
    assert payload["project"]["name"] == "satquery-ai"


def test_config_rejects_missing_file() -> None:
    with pytest.raises(ConfigError):
        load_config(REPO_ROOT / "configs" / "does_not_exist.yaml")


def test_unknown_key_in_override_is_kept_not_crashing() -> None:
    # overrides may add keys; the registry is permissive, the guards are specific
    cfg = load_config(overrides={"project": {"extra": "x"}})
    assert cfg.get("project.extra") == "x"