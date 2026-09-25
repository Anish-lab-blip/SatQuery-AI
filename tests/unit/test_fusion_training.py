"""Tests for the Phase 12 fusion-head training loop (fixtures only).

No real data, no network, no GPU. Every feature is synthetic. These tests prove
the PLUMBING: the loop runs, the frozen contract is enforced, channel dropout's
updated mask is fed forward, the split firewall holds, and the arm switches the
run record without moving `config_hash`.

They do NOT produce a result. The pre-registered 11.5 metric requires the real
paired corpus, >=3 seeds and a paired design; see the module docstring of
`training/fusion/train.py`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from training.fusion.train import (  # noqa: E402
    ARM_CONTROL,
    ARM_VARIANT,
    OPTICAL_DROPOUT_RATES,
    RESULT_STATUS,
    SAR_DROPOUT_RATES,
    USE_8_BIT_ENV,
    FusionFeature,
    FusionTrainingError,
    assert_split_disjoint,
    evaluate_fusion_head,
    prepare_fusion_batch,
    read_feature_cache,
    resolve_arm,
    train_fusion_head,
    write_feature_cache,
)

#: Column offsets of the frozen concatenation: 3*768 GAP, then the 12+2 masks.
MASK_START = 2304
SAR_MASK_START = 2316
FROZEN_CONFIG_HASH = "78f1e3700da15aa1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_features(
    n: int,
    *,
    scene: str,
    seed: int = 0,
    num_classes: int = 19,
    sample_offset: int = 0,
) -> list[FusionFeature]:
    rng = np.random.default_rng(seed)
    return [
        FusionFeature(
            sample_id=f"{scene}_s{sample_offset + i}",
            scene_id=scene,
            optical_gap=rng.standard_normal(768).astype(np.float32),
            sar_gap=rng.standard_normal(768).astype(np.float32),
            joint_gap=rng.standard_normal(768).astype(np.float32),
            optical_mask=np.ones(12, dtype=np.float32),
            sar_mask=np.ones(2, dtype=np.float32),
            label_index=int((sample_offset + i) % num_classes),
        )
        for i in range(n)
    ]


@pytest.fixture(autouse=True)
def _restore_use_8_bit():
    """The trainer sets an env var; restore it so tests cannot bleed into each other."""
    prior = os.environ.get(USE_8_BIT_ENV)
    yield
    if prior is None:
        os.environ.pop(USE_8_BIT_ENV, None)
    else:
        os.environ[USE_8_BIT_ENV] = prior


@pytest.fixture()
def train_features() -> list[FusionFeature]:
    return make_features(24, scene="train_tile", seed=1)


@pytest.fixture()
def val_features() -> list[FusionFeature]:
    return make_features(12, scene="val_tile", seed=2)


# ---------------------------------------------------------------------------
# The loop runs and emits a run record
# ---------------------------------------------------------------------------


def test_loop_runs_end_to_end_and_emits_a_run_record(
    tmp_path: Path, train_features, val_features
) -> None:
    out = tmp_path / "run"
    result = train_fusion_head(
        train_features,
        val_features,
        output_dir=out,
        epochs=2,
        batch_size=8,
        seed=3,
        verbose=False,
    )
    assert (out / "run_record.json").exists()
    assert (out / "training_metadata.json").exists()
    assert (out / "head.pt").exists()
    assert (out / "checkpoint_last.pt").exists()
    assert len(result.history) == 2

    record = json.loads((out / "run_record.json").read_text(encoding="utf-8"))
    assert len(record["history"]) == 2
    assert record["input_dim"] == 2318
    assert record["task_dim"] == 19


def test_run_record_flags_plumbing_only_and_metric_not_computed(
    tmp_path: Path, train_features, val_features
) -> None:
    out = tmp_path / "run"
    train_fusion_head(
        train_features, val_features, output_dir=out, epochs=1, batch_size=8,
        seed=3, verbose=False,
    )
    record = json.loads((out / "run_record.json").read_text(encoding="utf-8"))
    assert record["pre_registered_metric_computed"] is False
    assert "PLUMBING_ONLY" in record["result_status"]
    assert "NOT a result" in record["result_status"]


# ---------------------------------------------------------------------------
# The frozen contract
# ---------------------------------------------------------------------------


def test_input_dim_mismatch_is_rejected(
    tmp_path: Path, train_features, val_features
) -> None:
    """A `fusion.input_dim` that disagrees with 2318 must stop the run.

    The config is mutated AFTER construction so the trainer's own check -- not
    `Config._validate` -- is what fires.
    """
    from core.config import load_config

    cfg = load_config()
    cfg._data["fusion"]["input_dim"] = 2317  # noqa: SLF001 - deliberate bypass
    with pytest.raises(FusionTrainingError, match="input_dim"):
        train_fusion_head(
            train_features, val_features, cfg, output_dir=tmp_path / "x",
            epochs=1, verbose=False,
        )


def test_a_feature_of_the_wrong_width_is_rejected() -> None:
    """A mask whose width breaks the 2318 total is refused at construction."""
    with pytest.raises(FusionTrainingError, match="frozen fusion input"):
        FusionFeature(
            sample_id="s",
            scene_id="t",
            optical_gap=np.zeros(768, np.float32),
            sar_gap=np.zeros(768, np.float32),
            joint_gap=np.zeros(768, np.float32),
            optical_mask=np.zeros(11, np.float32),  # 11, not 12
            sar_mask=np.zeros(2, np.float32),
            label_index=0,
        )


# ---------------------------------------------------------------------------
# Channel dropout -- the mandatory part
# ---------------------------------------------------------------------------


def test_updated_mask_is_fed_forward() -> None:
    """If the pre-dropout mask were fed, this fails.

    `keep_probabilities=(0.0,)` drops every channel. The head input's mask region
    must therefore be all zeros. A version that discarded the updated mask would
    leave the base mask (all ones) in the tensor.
    """
    gaps = np.zeros((4, 2304), dtype=np.float32)
    optical_mask = np.ones((4, 12), dtype=np.float32)
    sar_mask = np.ones((4, 2), dtype=np.float32)

    dropped = prepare_fusion_batch(
        gaps, optical_mask, sar_mask,
        rng=np.random.default_rng(0),
        optical_rates=(0.0,), sar_rates=(0.0,),
    )
    assert np.all(dropped[:, MASK_START:SAR_MASK_START] == 0.0), (
        "the updated optical mask was discarded"
    )
    assert np.all(dropped[:, SAR_MASK_START:] == 0.0), (
        "the updated SAR mask was discarded"
    )

    kept = prepare_fusion_batch(
        gaps, optical_mask, sar_mask,
        rng=np.random.default_rng(0),
        optical_rates=(1.0,), sar_rates=(1.0,),
    )
    assert np.all(kept[:, MASK_START:SAR_MASK_START] == 1.0)


def test_dropout_cannot_resurrect_an_absent_band() -> None:
    """A channel the sensor never measured stays absent even at p=1.0."""
    gaps = np.zeros((4, 2304), dtype=np.float32)
    optical_mask = np.zeros((4, 12), dtype=np.float32)
    optical_mask[:, :4] = 1.0  # Cartosat-2S: 4 real channels, 8 masked off
    sar_mask = np.ones((4, 2), dtype=np.float32)

    out = prepare_fusion_batch(
        gaps, optical_mask, sar_mask,
        rng=np.random.default_rng(1),
        optical_rates=(1.0,), sar_rates=(1.0,),
    )
    assert np.all(out[:, MASK_START + 4 : SAR_MASK_START] == 0.0), (
        "dropout resurrected a band the sensor never measured"
    )


def test_evaluation_does_not_apply_dropout(train_features, val_features) -> None:
    """Validation must be deterministic: two evaluations agree exactly."""
    from specialists.optical_sar.fusion_head import build_fusion_head

    head = build_fusion_head(input_dim=2318, hidden_dim=32, task_dim=19, dropout=0.0)
    first = evaluate_fusion_head(head, val_features, batch_size=8, num_classes=19)
    second = evaluate_fusion_head(head, val_features, batch_size=8, num_classes=19)
    assert first.to_dict() == second.to_dict()


# ---------------------------------------------------------------------------
# The split firewall
# ---------------------------------------------------------------------------


def test_split_firewall_raises_on_a_shared_scene(train_features) -> None:
    val = make_features(4, scene=train_features[0].scene_id, seed=9)
    with pytest.raises(FusionTrainingError, match="scene"):
        assert_split_disjoint(train_features, val)


def test_training_refuses_a_leaky_split(tmp_path: Path, train_features) -> None:
    val = make_features(4, scene=train_features[0].scene_id, seed=9)
    with pytest.raises(FusionTrainingError):
        train_fusion_head(
            train_features, val, output_dir=tmp_path / "x", epochs=1, verbose=False
        )


# ---------------------------------------------------------------------------
# The arm mechanism -- no config edit
# ---------------------------------------------------------------------------


def test_arm_changes_the_run_record_but_not_config_hash(
    tmp_path: Path, train_features, val_features
) -> None:
    a_dir = tmp_path / "arm_a"
    b_dir = tmp_path / "arm_b"
    train_fusion_head(
        train_features, val_features, output_dir=a_dir, arm=ARM_CONTROL,
        epochs=1, batch_size=8, seed=5, verbose=False,
    )
    train_fusion_head(
        train_features, val_features, output_dir=b_dir, arm=ARM_VARIANT,
        epochs=1, batch_size=8, seed=5, verbose=False,
    )

    rec_a = json.loads((a_dir / "run_record.json").read_text(encoding="utf-8"))
    rec_b = json.loads((b_dir / "run_record.json").read_text(encoding="utf-8"))

    assert rec_a["config_hash"] == rec_b["config_hash"] == FROZEN_CONFIG_HASH
    assert rec_a["arm"] == "A" and rec_b["arm"] == "B"
    assert rec_a["use_8_bit"] is True and rec_b["use_8_bit"] is False


def test_the_arm_drives_the_hash_exempt_env_channel(
    tmp_path: Path, train_features, val_features
) -> None:
    train_fusion_head(
        train_features, val_features, output_dir=tmp_path / "run", arm="B",
        epochs=1, batch_size=8, seed=5, verbose=False,
    )
    assert os.environ.get(USE_8_BIT_ENV) == "false"


def test_resolve_arm_rejects_an_unknown_name() -> None:
    assert resolve_arm("a").name == "A"
    with pytest.raises(FusionTrainingError, match="unknown arm"):
        resolve_arm("C")


def test_dropout_schedules_are_the_plan_section_20_ones() -> None:
    assert OPTICAL_DROPOUT_RATES == (1.0, 0.8, 0.6, 0.4)
    assert SAR_DROPOUT_RATES == (1.0, 0.5)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def _stable_history(history):
    """Per-epoch fields that must be reproducible (wall-clock `seconds` excluded)."""
    return [
        {k: v for k, v in record.to_dict().items() if k != "seconds"}
        for record in history
    ]


def test_same_seed_gives_the_same_result(
    tmp_path: Path, train_features, val_features
) -> None:
    first = train_fusion_head(
        train_features, val_features, output_dir=tmp_path / "r1",
        epochs=2, batch_size=8, seed=11, verbose=False,
    )
    second = train_fusion_head(
        train_features, val_features, output_dir=tmp_path / "r2",
        epochs=2, batch_size=8, seed=11, verbose=False,
    )
    assert _stable_history(first.history) == _stable_history(second.history)
    assert first.best_val_accuracy == second.best_val_accuracy


# ---------------------------------------------------------------------------
# Feature cache round-trip
# ---------------------------------------------------------------------------


def test_feature_dataset_serves_the_frozen_blocks(val_features) -> None:
    from training.fusion.train import FusionFeatureDataset, collate

    dataset = FusionFeatureDataset(val_features)
    assert len(dataset) == len(val_features)
    gaps, o_mask, s_mask, label = dataset[0]
    assert tuple(gaps.shape) == (2304,)  # 3 * 768
    assert tuple(o_mask.shape) == (12,)
    assert tuple(s_mask.shape) == (2,)
    assert int(label) == val_features[0].label_index

    batch = collate([dataset[0], dataset[1]])
    assert tuple(batch[0].shape) == (2, 2304)
    assert tuple(batch[1].shape) == (2, 12)
    assert tuple(batch[2].shape) == (2, 2)


def test_feature_cache_round_trip(tmp_path: Path, train_features) -> None:
    path = write_feature_cache(
        tmp_path / "cache.npz",
        train_features,
        metadata={"per_channel_windows": {"B04": [0.0, 1.0]}},
    )
    loaded, meta = read_feature_cache(path)
    assert len(loaded) == len(train_features)
    assert loaded[0].sample_id == train_features[0].sample_id
    assert loaded[0].scene_id == train_features[0].scene_id
    assert loaded[0].label_index == train_features[0].label_index
    assert np.allclose(loaded[0].optical_gap, train_features[0].optical_gap)
    assert meta["per_channel_windows"] == {"B04": [0.0, 1.0]}


# ---------------------------------------------------------------------------
# The CLI re-export contract
# ---------------------------------------------------------------------------


def test_script_reexports_the_loop() -> None:
    from scripts import train_fusion

    for name in (
        "train_fusion_head",
        "evaluate_fusion_head",
        "prepare_fusion_batch",
        "read_feature_cache",
        "resolve_arm",
        "FusionTrainingError",
    ):
        assert hasattr(train_fusion, name), f"scripts.train_fusion.{name} missing"


def test_package_lazily_reexports_the_loop() -> None:
    import training.fusion as fusion

    assert fusion.train_fusion_head is train_fusion_head
    assert fusion.FusionTrainingError is FusionTrainingError


# ---------------------------------------------------------------------------
# Cache/arm provenance -- the guard that makes a mismatch loud
#
# The arm is BAKED INTO the cached features, not applied at train time, so a cache
# built under one arm must not be trainable under the other. These tests pin the
# guard's behaviour in BOTH directions: a mismatch must refuse, and a matching pair
# must still be accepted. Every cache lives in `tmp_path`; the real Phase 12 output
# path is never touched.
# ---------------------------------------------------------------------------


def _provenance_metadata(split: str, **overrides) -> dict:
    """The metadata the real extractor writes onto every cache sidecar."""
    metadata = {
        "cache_version": "v1",
        "config_hash": FROZEN_CONFIG_HASH,
        "arm": ARM_CONTROL,
        "use_8_bit": True,
        "croma_image_resolution": 120,
        "encoder_dim": 768,
        "label_policy": "require_single_label",
        "split": split,
    }
    metadata.update(overrides)
    return metadata


def _run_train_cli(monkeypatch, argv: list[str]) -> int:
    import sys

    import scripts.train_fusion as cli

    monkeypatch.setattr(sys, "argv", ["train_fusion.py", *argv])
    return cli.main()


def _cli_argv(tmp_path: Path, train: Path, val: Path, arm: str) -> list[str]:
    return [
        "--train-cache", str(train),
        "--val-cache", str(val),
        "--arm", arm,
        "--output-dir", str(tmp_path / "cli_out"),
        "--dry-run",
    ]


def test_cli_accepts_a_cache_whose_arm_matches(
    monkeypatch, tmp_path, train_features, val_features, capsys
) -> None:
    train = write_feature_cache(
        tmp_path / "train.npz", train_features, metadata=_provenance_metadata("train")
    )
    val = write_feature_cache(
        tmp_path / "val.npz", val_features, metadata=_provenance_metadata("val")
    )

    rc = _run_train_cli(monkeypatch, _cli_argv(tmp_path, train, val, ARM_CONTROL))

    assert rc == 0
    assert "cache arm" in capsys.readouterr().out


def test_cli_refuses_an_arm_mismatch(
    monkeypatch, tmp_path, train_features, val_features, capsys
) -> None:
    train = write_feature_cache(
        tmp_path / "train.npz", train_features, metadata=_provenance_metadata("train")
    )
    val = write_feature_cache(
        tmp_path / "val.npz", val_features, metadata=_provenance_metadata("val")
    )

    rc = _run_train_cli(monkeypatch, _cli_argv(tmp_path, train, val, ARM_VARIANT))

    out = capsys.readouterr().out
    assert rc == 2
    assert "records arm" in out
    assert "baked into the features" in out


def test_cli_refuses_a_use_8_bit_mismatch(
    monkeypatch, tmp_path, train_features, val_features, capsys
) -> None:
    # Both sidecars say use_8_bit=False while arm A means True.
    train = write_feature_cache(
        tmp_path / "train.npz", train_features,
        metadata=_provenance_metadata("train", use_8_bit=False),
    )
    val = write_feature_cache(
        tmp_path / "val.npz", val_features,
        metadata=_provenance_metadata("val", use_8_bit=False),
    )

    rc = _run_train_cli(monkeypatch, _cli_argv(tmp_path, train, val, ARM_CONTROL))

    out = capsys.readouterr().out
    assert rc == 2
    assert "use_8_bit" in out


def test_cli_refuses_a_cache_that_omits_provenance(
    monkeypatch, tmp_path, train_features, val_features, capsys
) -> None:
    """Absence is a refusal, not a pass -- the guard must fail CLOSED."""
    train_meta = _provenance_metadata("train")
    val_meta = _provenance_metadata("val")
    del train_meta["arm"], train_meta["use_8_bit"]
    del val_meta["arm"], val_meta["use_8_bit"]
    train = write_feature_cache(tmp_path / "train.npz", train_features, metadata=train_meta)
    val = write_feature_cache(tmp_path / "val.npz", val_features, metadata=val_meta)

    rc = _run_train_cli(monkeypatch, _cli_argv(tmp_path, train, val, ARM_VARIANT))

    out = capsys.readouterr().out
    assert rc == 2
    assert "records no" in out


def test_cli_refuses_when_the_two_splits_disagree(
    monkeypatch, tmp_path, train_features, val_features, capsys
) -> None:
    train = write_feature_cache(
        tmp_path / "train.npz", train_features, metadata=_provenance_metadata("train")
    )
    val = write_feature_cache(
        tmp_path / "val.npz", val_features,
        metadata=_provenance_metadata("val", arm=ARM_VARIANT, use_8_bit=False),
    )

    rc = _run_train_cli(monkeypatch, _cli_argv(tmp_path, train, val, ARM_CONTROL))

    out = capsys.readouterr().out
    assert rc == 2
    assert "disagree" in out


# ---------------------------------------------------------------------------
# Resume provenance -- a checkpoint carries the run that wrote it
# ---------------------------------------------------------------------------


def _make_checkpoint(tmp_path: Path, train_features, val_features, *, arm: str) -> Path:
    out = tmp_path / "ckpt_src"
    train_fusion_head(
        train_features, val_features, output_dir=out, arm=arm,
        epochs=1, batch_size=8, max_steps=1, seed=3, verbose=False,
    )
    return out / "checkpoint_last.pt"


def _resume(tmp_path: Path, train_features, val_features, ckpt: Path, **kwargs):
    return train_fusion_head(
        train_features, val_features, output_dir=tmp_path / "resumed",
        epochs=2, batch_size=8, max_steps=1, seed=3, resume_from=ckpt,
        verbose=False, **kwargs,
    )


def test_resume_accepts_a_matching_checkpoint(tmp_path, train_features, val_features) -> None:
    ckpt = _make_checkpoint(tmp_path, train_features, val_features, arm=ARM_CONTROL)

    _resume(tmp_path, train_features, val_features, ckpt, arm=ARM_CONTROL)

    record = json.loads((tmp_path / "resumed" / "run_record.json").read_text("utf-8"))
    assert record["arm"] == ARM_CONTROL
    assert record["use_8_bit"] is True


def test_resume_refuses_a_different_arm(tmp_path, train_features, val_features) -> None:
    ckpt = _make_checkpoint(tmp_path, train_features, val_features, arm=ARM_CONTROL)

    with pytest.raises(FusionTrainingError, match="arm:"):
        _resume(tmp_path, train_features, val_features, ckpt, arm=ARM_VARIANT)


def test_resume_refuses_a_changed_dropout(tmp_path, train_features, val_features) -> None:
    """`dropout` has no parameters, so strict=True cannot catch this -- only the guard can."""
    ckpt = _make_checkpoint(tmp_path, train_features, val_features, arm=ARM_CONTROL)

    with pytest.raises(FusionTrainingError, match=r"head_config\[dropout\]"):
        _resume(tmp_path, train_features, val_features, ckpt, arm=ARM_CONTROL, dropout=0.5)


def test_resume_refuses_a_changed_config_hash(tmp_path, train_features, val_features) -> None:
    import torch

    ckpt = _make_checkpoint(tmp_path, train_features, val_features, arm=ARM_CONTROL)
    data = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    data["config_hash"] = "deadbeefdeadbeef"
    tampered = tmp_path / "tampered.pt"
    torch.save(data, str(tampered))

    with pytest.raises(FusionTrainingError, match="config_hash"):
        _resume(tmp_path, train_features, val_features, tampered, arm=ARM_CONTROL)


def test_resume_refuses_a_checkpoint_without_provenance(
    tmp_path, train_features, val_features
) -> None:
    """Absence is a refusal: a checkpoint the trainer did not write is not resumable."""
    import torch

    ckpt = _make_checkpoint(tmp_path, train_features, val_features, arm=ARM_CONTROL)
    data = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    for key in ("config_hash", "arm", "use_8_bit", "head_config"):
        data.pop(key, None)
    stripped = tmp_path / "stripped.pt"
    torch.save(data, str(stripped))

    with pytest.raises(FusionTrainingError, match="cannot be identified"):
        _resume(tmp_path, train_features, val_features, stripped, arm=ARM_CONTROL)


# ---------------------------------------------------------------------------
# The checkpoint contract
# ---------------------------------------------------------------------------
#
# A 2026-09-20 audit established that a resume restores the head parameters and
# the optimizer state, but NOT the stochastic streams: `torch.manual_seed`,
# `np.random.seed`, `random.seed`, `shuffle_rng` and `dropout_rng` are all
# re-seeded from the same seed on entry to `train_fusion_head`. A resumed run
# therefore replays the ORIGINAL epoch-0 shuffle and dropout streams instead of
# continuing the uninterrupted trajectory, and it does NOT reproduce an
# uninterrupted run bit-for-bit. `best_val_accuracy` and `history` also reset,
# so a resumed run's record describes only the resumed segment.
#
# The test below pins the checkpoint's key set as an EXACT equality. It asserts
# what is present; it does NOT assert that the absence of RNG state is
# desirable. Its purpose is to make any future change to the checkpoint format
# deliberate -- adding RNG state must fail this test and be acknowledged rather
# than slipping in unnoticed.

CHECKPOINT_KEYS = frozenset({
    "state_dict",
    "optimizer",
    "epoch",
    "config_hash",
    "arm",
    "use_8_bit",
    "head_config",
})


def test_checkpoint_carries_exactly_the_documented_state(
    tmp_path, train_features, val_features
) -> None:
    import torch

    ckpt = _make_checkpoint(tmp_path, train_features, val_features, arm=ARM_CONTROL)
    data = torch.load(str(ckpt), map_location="cpu", weights_only=False)

    # Exact equality, with the delta spelled out. A bare `assert set(data) == ...`
    # fails with an empty message, which hides WHICH key appeared or vanished.
    unexpected = set(data) - CHECKPOINT_KEYS
    missing = CHECKPOINT_KEYS - set(data)
    assert set(data) == CHECKPOINT_KEYS, (
        "the checkpoint key set changed; a change to the checkpoint format must "
        f"be deliberate. unexpected={sorted(unexpected)} missing={sorted(missing)}"
    )

    # The provenance the resume guard depends on.
    assert data["config_hash"] == FROZEN_CONFIG_HASH
    assert data["arm"] == ARM_CONTROL
    assert data["use_8_bit"] is True
    assert set(data["head_config"]) >= {"input_dim", "hidden_dim", "task_dim", "dropout"}

    # `_save_checkpoint` is called with `epoch + 1`, so the stored epoch is
    # 1-based and a resume continues at the right place.
    assert data["epoch"] == 1

    # State the checkpoint deliberately does NOT carry. Listed in one place so a
    # reader sees the gap directly rather than having to infer it from an
    # absence.
    for absent in (
        "rng_state",
        "torch_rng_state",
        "numpy_rng_state",
        "global_step",
        "best_val_accuracy",
        "history",
        "scheduler",
    ):
        assert absent not in data


# ---------------------------------------------------------------------------
# `final_val` describes the BEST state, not the last epoch
# ---------------------------------------------------------------------------


def test_final_val_is_the_best_state_not_the_last_epoch(
    tmp_path, train_features, val_features
) -> None:
    """`final_val` is evaluated only AFTER the best state is restored.

    `train_fusion_head` restores the best-by-validation-accuracy state into the
    head (`head.load_state_dict(best_state)`, `:1005-1006`) and only then calls
    `evaluate_fusion_head` (`:1008-1010`). That function is `@torch.no_grad()`,
    calls `head.eval()`, and passes `apply_dropout=False`, so it is
    deterministic. The reported `final_val` therefore describes the BEST
    checkpoint, not the final epoch -- despite the field name.

    For accuracy the two are the same number by construction. For macro-F1 they
    are not: `final_val.macro_f1` is the best epoch's value, while
    `history[-1].val_macro_f1` is the last epoch's.

    A 2026-09-20 mutation test showed the ORIGINAL inputs (`seed=3, epochs=3`)
    produced `history=[0.0, 0.0, 0.0]` -- a single distinct value. With
    `best == last`, this test PASSED even after the best-state restore was
    deleted: the assertion was vacuous. The inputs are now a discriminating
    configuration (`seed=4, epochs=3` -> `[0.1667, 0.1667, 0.0833]`), and the
    precondition below is asserted explicitly so that a future fixture change
    fails LOUDLY instead of silently making this test vacuous again.
    """
    out = tmp_path / "run"
    result = train_fusion_head(
        train_features,
        val_features,
        output_dir=out,
        epochs=3,
        batch_size=8,
        seed=4,
        verbose=False,
    )

    accs = [record.val_accuracy for record in result.history]

    # PRECONDITION. Without this the test cannot distinguish "final_val is the
    # best state" from "final_val is the last epoch", and every assertion below
    # would hold regardless of the code under test.
    assert accs[-1] < result.best_val_accuracy, (
        "fixture stopped discriminating: the last epoch must be WORSE than the "
        "best epoch, otherwise the assertions below pass vacuously. "
        f"history={accs}, best={result.best_val_accuracy}. Choose a seed where "
        "validation accuracy peaks before the last epoch."
    )

    # The claim under test.
    assert result.final_val.accuracy == result.best_val_accuracy, (
        "final_val must describe the BEST state, not the last epoch. "
        f"final_val.accuracy={result.final_val.accuracy}, "
        f"best_val_accuracy={result.best_val_accuracy}, "
        f"last_epoch_val_accuracy={accs[-1]}"
    )
    assert result.final_val.accuracy > accs[-1]

    # The stricter claim, which accuracy alone cannot express: `final_val`
    # carries the macro-F1 of the BEST epoch, not of the last one. Both numbers
    # come from the same deterministic function on the same parameters, so this
    # is an exact identity rather than a tolerance.
    # `best_val_accuracy` is updated on a STRICT improvement (`:976`), so the
    # best epoch is the FIRST epoch attaining the maximum.
    best_epoch = accs.index(result.best_val_accuracy)
    assert best_epoch < len(accs) - 1  # implied by the precondition above
    assert result.final_val.macro_f1 == pytest.approx(
        result.history[best_epoch].val_macro_f1
    )

    # The persisted record must agree with the returned object.
    record = json.loads((out / "run_record.json").read_text(encoding="utf-8"))
    assert record["final_val"]["accuracy"] == record["best_val_accuracy"]

    # The persisted record must agree with the returned object.
    record = json.loads((out / "run_record.json").read_text(encoding="utf-8"))
    assert record["final_val"]["accuracy"] == record["best_val_accuracy"]


# ---------------------------------------------------------------------------
# RESULT_STATUS must not assert something that is no longer true
# ---------------------------------------------------------------------------
def test_result_status_does_not_deny_a_metric_that_is_actually_computed() -> None:
    """The flag's trailing clause was TRUE once and became FALSE. Pin the fix.

    `RESULT_STATUS` is stamped into every run record this module writes
    (`training/fusion/train.py:1094` and `:1154`). Its original tail read "the
    pre-registered 11.5 metric is not computed", which was accurate while no
    production head existed and became FALSE on 2026-09-23 when the frozen head
    was recovered and the metric WAS computed and independently reproduced.

    The danger is specific: a false claim in a CONSTANT propagates into every
    FUTURE record, so unlike a stale prose line it cannot be caught by reading a
    document. This guard reads the constant and fails if the denial returns.

    Note what this does NOT do: it does not rewrite the ten historical
    `run_record.json` files. Those are preserved byte-for-byte and clarified
    beside rather than edited (docs/PHASE12_R08_TRUTHFUL_BACKFILL.md).
    """
    lowered = RESULT_STATUS.lower()

    # -- the flag must keep doing its job -----------------------------------
    assert "PLUMBING_ONLY" in RESULT_STATUS
    assert "NOT a result" in RESULT_STATUS

    # -- and must not deny a metric that exists -----------------------------
    for denial in (
        "is not computed",
        "not computed",
        "metric is not",
    ):
        assert denial.lower() not in lowered, (
            f"RESULT_STATUS again claims {denial!r}; the pre-registered metric "
            f"IS computed (accuracy 0.931000). A constant is stamped into every "
            f"future run record, so this must not regress."
        )


def test_result_status_cites_a_record_that_exists_and_says_what_is_claimed() -> None:
    """The citation in the flag must resolve, and must agree with the flag.

    A status string that points at a file is a claim about the filesystem. This
    project's rule is that the disk wins over the written record, so the disk is
    what gets checked.
    """
    repo_root = Path(__file__).resolve().parents[2]
    cited = (
        repo_root
        / "artifacts"
        / "optical_sar"
        / "fusion_head_production_v001"
        / "pre_registered_115_metric.json"
    )
    assert cited.exists(), (
        f"RESULT_STATUS cites {cited.name}, which does not exist"
    )

    record = json.loads(cited.read_text(encoding="utf-8"))

    # The flag says the metric IS computed, and separately. Both must hold.
    # Field names read off the record itself, not assumed: the config hash is
    # spelled `cache_config_hash` here.
    assert record["accuracy"] == pytest.approx(0.931)
    assert record["macro_f1"] == pytest.approx(0.434161)
    assert record["loss"] == pytest.approx(0.254592)
    assert record["n_scored"] == 4000
    assert record["cache_arm"] == "A"
    assert record["cache_config_hash"] == FROZEN_CONFIG_HASH
    assert record["split"] == "test"
    assert record["num_classes"] == 19
    assert record["is_deciding_statistic"] is False

    # The record is a claim ABOUT AN ARTIFACT, so check the artifact too.
    head = repo_root / "artifacts" / "optical_sar" / "fusion_head_production_v001" / "head.pt"
    assert head.exists(), "the record names a head that is not on disk"
    assert head.stat().st_size == record["head_bytes"] == 14_427_457

    # The flag names the record by basename; that name must be in the constant.
    assert cited.name in RESULT_STATUS
