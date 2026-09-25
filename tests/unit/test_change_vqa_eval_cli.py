"""R-02 — the evaluation CLI's empty-split diagnosis.

WHY THIS FILE EXISTS
--------------------
The second real Kaggle run reached evaluation and died with:

    SatQueryError: split Test: no record had its features and targets all
    present ({'missing_change_feature': 0, 'missing_text_feature': 0,
    'missing_targets': 39686})

Every number in that message is correct and none of them is the cause. The cause
was one preparation step earlier: `scene_targets.jsonl` had been built for
`--splits Train Val`, and every later prepare call passed `--skip-targets`, so
the held-out targets the evaluator needs had never been written. From three skip
counts alone, "the targets for this split do not exist" is indistinguishable from
"the labels were read and every one was unusable" — and the two have opposite
fixes. The run had to be repeated to find that out.

`_explain_empty_split` now names the dominant reason and the fix. These tests
drive it through `score_split` with real records and real caches, so what is
proven is the message a caller actually receives, not a string constant.

`score_split` raises BEFORE it touches the model, so `model=None` is passed
deliberately: a test that needed a working model to reach the error would be
testing the wrong thing. The one test that needs a scorable assembly calls
`_assemble` directly rather than fabricating a model.

WHAT IS DELIBERATELY NOT TESTED HERE
------------------------------------
The quality of a scored report. That is area Q, in `test_change_vqa_head.py`.
This file covers only the refusal path.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.errors import SatQueryError
from scripts.evaluate_change_vqa import (
    _assemble,
    _explain_empty_split,
    score_split,
)
from training.change_vqa.dataset import ChangeVQARecord, SceneTargets
from training.change_vqa.features import (
    CHANGE_FEATURE_DIM,
    TEXT_FEATURE_DIM,
    ChangeFeatureCache,
    TextFeatureCache,
)
from training.change_vqa.vocab import N_CHANGE_CLASSES

SPLIT = "Test"


@pytest.fixture()
def scratch() -> Path:
    # `tempfile.mkdtemp`, never pytest's `tmp_path`: this sandbox blocks the
    # directory removals `tmp_path` performs at teardown. The directory is left
    # in place on purpose.
    return Path(tempfile.mkdtemp(prefix="r02_eval_cli_"))


# ---------------------------------------------------------------------------
# The smallest real objects that can reach the guard
# ---------------------------------------------------------------------------


def _record(scene_key: str, question_id: int, split: str = SPLIT) -> ChangeVQARecord:
    return ChangeVQARecord(
        split=split,
        question_id=question_id,
        file_name=f"{scene_key}.png",
        scene_key=scene_key,
        qtype="change_or_not",
        qtype_index=0,
        resolved_qtype="change_or_not",
        qtype_resolved=True,
        temporal_ref="none",
        change_class=None,
        question="did anything change?",
        answer="yes",
        answer_index=0,
        t1_path=f"{scene_key}_t1.png",
        t2_path=f"{scene_key}_t2.png",
    )


def _target(scene_key: str) -> SceneTargets:
    zeros = tuple(0.0 for _ in range(N_CHANGE_CLASSES))
    return SceneTargets(
        file_name=f"{scene_key}.png",
        scene_key=scene_key,
        class_mag=zeros,
        class_delta=zeros,
        class_ratio=zeros,
        total_changed=0.0,
        n_pixels=4,
        n_labeled_pixels=4,
        label1_sha256="0" * 64,
        label2_sha256="0" * 64,
    )


def _targets(*scene_keys: str) -> dict[str, SceneTargets]:
    """A `file_name -> SceneTargets` map, exactly as `read_scene_targets` builds
    it. Keyed on the FILE NAME, which is what `_assemble` looks up."""
    return {f"{key}.png": _target(key) for key in scene_keys}


def _change_cache(*scene_keys: str) -> ChangeFeatureCache:
    return ChangeFeatureCache(
        scene_keys=tuple(scene_keys),
        features=np.zeros((len(scene_keys), CHANGE_FEATURE_DIM), dtype=np.float32),
        spec_hash="synthetic-spec",
        extractor_config={"synthetic": True},
    )


def _text_cache(*question_ids: int) -> TextFeatureCache:
    return TextFeatureCache(
        question_ids=tuple(question_ids),
        features=np.zeros((len(question_ids), TEXT_FEATURE_DIM), dtype=np.float32),
        spec_hash="synthetic-spec",
    )


def _drive(records, *, change_cache, text_cache, targets, split: str = SPLIT) -> dict:
    """`score_split` with everything it needs except a model, which it must not
    reach. The other arguments are only read after the guard."""
    return score_split(
        split,
        records=records,
        model=None,
        change_cache=change_cache,
        text_cache=text_cache,
        targets=targets,
        batch_size=8,
        device="cpu",
        out_dir=Path("unused"),
        checkpoint_path=Path("unused.pt"),
        checkpoint_sha256="unused",
        config_hash="unused",
        n_success=1,
        n_failure=1,
    )


# ---------------------------------------------------------------------------
# The regression: the exact Kaggle state
# ---------------------------------------------------------------------------


def test_the_kaggle_failure_now_names_the_missing_targets_step() -> None:
    """The reproduction. Features for Test exist, the questions resolve, and
    `scene_targets.jsonl` was never extended past Train/Val.

    Before the fix this raised the bare three-count message, which is what sent
    the run back to Kaggle a second time.
    """
    scene_keys = [f"test_scene_{i}" for i in range(4)]
    records = [_record(key, 100 + i) for i, key in enumerate(scene_keys)]

    with pytest.raises(SatQueryError) as excinfo:
        _drive(
            records,
            change_cache=_change_cache(*scene_keys),
            text_cache=_text_cache(*[100 + i for i in range(4)]),
            targets={},  # <- the Kaggle state: held-out targets never built
        )

    message = str(excinfo.value)
    assert "missing_targets" in message
    assert "--skip-targets" in message, message
    assert "Train Val Test Test2" in message, message
    assert "scene_targets.jsonl" in message, message
    # The count that was reported must survive, so the two runs are comparable.
    assert "4" in message


def test_the_message_says_how_many_scenes_the_targets_file_holds() -> None:
    """`missing_targets == n` with a NON-empty targets file is the real shape:
    the file existed and held 2,000 Train/Val scenes, and covered none of Test.
    Saying how many it holds is what distinguishes that from an empty file."""
    records = [_record("test_scene_0", 1)]

    with pytest.raises(SatQueryError) as excinfo:
        _drive(
            records,
            change_cache=_change_cache("test_scene_0"),
            text_cache=_text_cache(1),
            targets=_targets("train_scene_0", "val_scene_0"),
        )

    message = str(excinfo.value)
    assert "holds 2 scene(s)" in message, message
    assert "covers none of them" in message, message


def test_the_error_is_typed_so_the_cli_reports_it_as_an_input_problem() -> None:
    """`main()` maps `SatQueryError` to exit 2 (missing inputs) and every other
    exception to exit 3 (the run started and broke). A bare `RuntimeError` here
    would tell an operator the pipeline crashed rather than that a step was
    skipped."""
    with pytest.raises(SatQueryError):
        _drive(
            [_record("test_scene_0", 1)],
            change_cache=_change_cache("test_scene_0"),
            text_cache=_text_cache(1),
            targets={},
        )


# ---------------------------------------------------------------------------
# The other two skip reasons, which must not be blamed on targets
# ---------------------------------------------------------------------------


def test_missing_change_features_are_diagnosed_separately() -> None:
    records = [_record("test_scene_0", 1)]

    with pytest.raises(SatQueryError) as excinfo:
        _drive(
            records,
            change_cache=_change_cache("some_other_scene"),  # per-scene cache
            text_cache=_text_cache(1),
            targets=_targets("test_scene_0"),
        )

    message = str(excinfo.value)
    assert "change feature" in message, message
    assert "--change-checkpoint" in message, message
    assert "--skip-targets" not in message, message


def test_missing_text_features_are_diagnosed_separately() -> None:
    records = [_record("test_scene_0", 1)]

    with pytest.raises(SatQueryError) as excinfo:
        _drive(
            records,
            change_cache=_change_cache("test_scene_0"),
            text_cache=_text_cache(999),  # per-question cache
            targets=_targets("test_scene_0"),
        )

    message = str(excinfo.value)
    assert "question feature" in message, message
    assert "--skip-targets" not in message, message


def test_a_split_with_no_records_at_all_says_so() -> None:
    """No records is a different failure — the annotations were not found — and
    must not be reported as a missing cache."""
    with pytest.raises(SatQueryError) as excinfo:
        _drive(
            [],
            change_cache=_change_cache(),
            text_cache=_text_cache(),
            targets={},
        )

    message = str(excinfo.value)
    assert "no records were resolved at all" in message, message
    assert "missing_targets" not in message, message


def test_a_mixed_shortfall_does_not_claim_a_single_cause() -> None:
    """Two records, one missing a feature and one missing a target: neither
    reason accounts for the whole split, so the message must not pick one."""
    records = [
        _record("test_scene_0", 1),
        _record("test_scene_1", 2),
    ]

    with pytest.raises(SatQueryError) as excinfo:
        _drive(
            records,
            change_cache=_change_cache("test_scene_0"),  # scene_1 absent
            text_cache=_text_cache(1, 2),
            targets={},  # both targets absent, but scene_1 fails earlier
        )

    message = str(excinfo.value)
    assert "a mix of the reasons above" in message, message
    assert "Every one of this split's" not in message, message


# ---------------------------------------------------------------------------
# The guard must not fire when it should not
# ---------------------------------------------------------------------------


def test_a_complete_assembly_keeps_every_record_and_skips_nothing() -> None:
    """The control. A guard that refuses a scorable split is worse than the
    cryptic message it replaced, so full coverage is asserted directly."""
    scene_keys = [f"test_scene_{i}" for i in range(3)]
    records = [_record(key, 10 + i) for i, key in enumerate(scene_keys)]

    assembled = _assemble(
        records,
        _change_cache(*scene_keys),
        _text_cache(*[10 + i for i in range(3)]),
        _targets(*scene_keys),
    )

    assert len(assembled["records"]) == 3
    assert assembled["skipped"] == {
        "missing_change_feature": 0,
        "missing_text_feature": 0,
        "missing_targets": 0,
    }


def test_a_partial_shortfall_still_scores_what_it_can() -> None:
    """Half the records scorable is a scorable split. The refusal is only for
    the case where NOTHING survives."""
    records = [_record(f"test_scene_{i}", i) for i in range(4)]

    assembled = _assemble(
        records,
        _change_cache("test_scene_0", "test_scene_1"),
        _text_cache(0, 1, 2, 3),
        _targets("test_scene_0", "test_scene_1"),
    )

    assert len(assembled["records"]) == 2
    assert assembled["skipped"]["missing_change_feature"] == 2
    assert assembled["skipped"]["missing_targets"] == 0


# ---------------------------------------------------------------------------
# The causal statement: the same records, one preparation step apart
# ---------------------------------------------------------------------------


def test_train_val_only_targets_reproduce_the_kaggle_symptom_and_held_out_targets_fix_it() -> (
    None
):
    """The whole defect, in one test.

    Identical records, identical caches. The only variable is whether the targets
    file was extended past the training splits — and that alone flips the split
    from "no record could be scored" to "every record scored". Nothing about the
    features, the questions or the model changed, which is exactly what the
    original message failed to convey.
    """
    scene_keys = [f"test_scene_{i}" for i in range(3)]
    records = [_record(key, 200 + i) for i, key in enumerate(scene_keys)]
    change_cache = _change_cache(*scene_keys)
    text_cache = _text_cache(*[200 + i for i in range(3)])

    # The state on Kaggle: targets built for Train/Val only.
    train_val_only = _assemble(
        records, change_cache, text_cache, _targets("train_scene_0", "val_scene_0")
    )
    assert train_val_only["records"] == []
    assert train_val_only["skipped"]["missing_targets"] == 3
    assert train_val_only["skipped"]["missing_change_feature"] == 0
    assert train_val_only["skipped"]["missing_text_feature"] == 0

    explanation = _explain_empty_split(
        SPLIT, records, train_val_only, _targets("train_scene_0", "val_scene_0")
    )
    assert "--skip-targets" in explanation

    # The state after the fix: the same file, extended to all four splits.
    extended = _assemble(
        records, change_cache, text_cache, _targets(*scene_keys)
    )
    assert len(extended["records"]) == 3
    assert sum(extended["skipped"].values()) == 0


# ---------------------------------------------------------------------------
# A REQUESTED split with no cache is an input problem, not a skip
# ---------------------------------------------------------------------------
#
# Found while closing K4. The scoring loop used to `continue` past a split whose
# `text_features_<split>.npz` was absent, print one line, and still exit 0. So a
# run with `Test` prepared and `Test2` not would score `Test`, write an
# `eval_summary.json` containing only `Test`, and report success -- a
# successful-looking artifact silently missing a held-out split.
#
# That is worse than the K4 crash. K4 stopped the run and left a traceback; this
# left a plausible file and no signal, and the missing split is exactly the kind
# of thing a reviewer skims past.
#
# `prepare` can produce the state legitimately: a split that resolves no
# questions writes no cache file at all. It now warns and records the split, and
# the evaluator now refuses.


def _prepared_dir(root: Path, *, with_text_for: tuple[str, ...]) -> Path:
    """A prepared directory with the two shared artifacts and a
    question-feature cache for exactly `with_text_for`.

    Real caches written by the real writers, so this test does not depend on the
    check happening before the caches are read.
    """
    from training.change_vqa.dataset import write_scene_targets

    prepared = root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    ChangeFeatureCache(
        scene_keys=("test_scene_0",),
        features=np.zeros((1, CHANGE_FEATURE_DIM), dtype=np.float32),
        spec_hash="synthetic-spec",
        extractor_config={"synthetic": True},
    ).write(prepared / "change_features.npz")
    write_scene_targets([_target("test_scene_0")], prepared / "scene_targets.jsonl")
    for split in with_text_for:
        _text_cache(1).write(prepared / f"text_features_{split}.npz")
    return prepared


def _run_cli(scratch: Path, prepared: Path, *splits: str) -> int:
    from scripts.evaluate_change_vqa import main

    checkpoint = scratch / "head.pt"
    checkpoint.write_bytes(b"")  # exists; never loaded, the input check runs first
    return main([
        "--data-root", str(scratch / "data"),
        "--prepared-dir", str(prepared),
        "--checkpoint", str(checkpoint),
        "--splits", *splits,
    ])


def test_a_requested_split_without_a_text_cache_is_refused_as_an_input_problem(
    scratch: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.evaluate_change_vqa import EXIT_INPUTS

    prepared = _prepared_dir(scratch, with_text_for=("Test",))

    rc = _run_cli(scratch, prepared, "Test", "Test2")

    out = capsys.readouterr().out
    assert rc == EXIT_INPUTS, (
        "a requested split with no cache is missing input, which is exit 2 -- "
        "not a crash (3) and not success (0)"
    )
    assert "text_features_Test2.npz" in out, out
    assert "Test2" in out, out
    assert "MISSING PREPARED INPUT" in out, out
    # And it must not have blamed the split that IS prepared.
    assert "text_features_Test.npz" not in out, out


def test_the_refusal_is_fail_fast_before_any_model_is_loaded(
    scratch: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The missing cache is caught before a checkpoint is read.

    This test was rewritten after a mutation harness showed the first version was
    **vacuous**. It used to assert `rc != 0` and "no `eval_summary.json` was
    written" — but the fixture's checkpoint is an empty file, so the pre-fix
    behaviour also died, in the loader, and also wrote no summary. Both states
    satisfied both assertions, so the test pinned nothing.

    The property that actually separates the two states is **what was reported**,
    because the exit code does not: an unreadable checkpoint is an input problem
    too, so both states exit 2. Pre-fix, the run passed the input stage and the
    output named a model-load failure. Post-fix, the output names the missing
    cache and never reaches the loader. So the discriminator is that the model
    was never consulted — which is also the property worth having: a prepare
    mistake should cost nothing, not a full model load.
    """
    from scripts.evaluate_change_vqa import EXIT_INPUTS

    prepared = _prepared_dir(scratch, with_text_for=("Test",))

    rc = _run_cli(scratch, prepared, "Test", "Test2")

    captured = capsys.readouterr()
    everything = captured.out + captured.err
    assert rc == EXIT_INPUTS, (
        "a requested split with no cache is missing input, which is exit 2"
    )
    assert "MISSING PREPARED INPUT" in everything, everything
    assert "text_features_Test2.npz" in everything, everything
    assert "ModelLoadError" not in everything, (
        "the input check must run BEFORE the checkpoint is loaded; reaching the "
        "loader means the missing cache was not caught, which is the pre-fix "
        f"behaviour; output was:\n{everything}"
    )
    assert not (scratch / "eval_summary.json").exists(), (
        "no summary may be written for a run that could not score every "
        "requested split"
    )


def test_all_requested_caches_present_gets_past_the_input_check(
    scratch: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The control. A guard that refuses a properly prepared directory is worse
    than the silent skip it replaced.

    The checkpoint is an empty file, so the run must fail *later* — trying to
    read the head. That is the assertion: the output names the checkpoint
    failure, which is only reachable if every text-cache check passed.

    Note the exit code is 2 here as well, and correctly so: an unreadable
    checkpoint is also an input problem (`ModelLoadError` is a `SatQueryError`).
    So the exit code alone cannot distinguish "refused the split" from "refused
    the checkpoint" — the message can, and that is what is asserted.
    """
    from scripts.evaluate_change_vqa import EXIT_INPUTS

    prepared = _prepared_dir(scratch, with_text_for=("Test", "Test2"))

    rc = _run_cli(scratch, prepared, "Test", "Test2")

    captured = capsys.readouterr()
    everything = captured.out + captured.err
    assert "MISSING PREPARED INPUT" not in everything, everything
    assert "ModelLoadError" in everything, (
        "the run must have reached the checkpoint, which it can only do if "
        f"every requested split's cache was found; output was:\n{everything}"
    )
    assert rc == EXIT_INPUTS, "an unreadable checkpoint is an input problem"
    assert rc != 0
