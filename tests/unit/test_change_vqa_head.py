"""R-02 test areas M-Q — batching, the reasoning head, and the metrics.

    M  deterministic batching with explicit skip accounting
    N  head architecture and parameter count
    O  legal-answer masking at inference
    P  strict checkpoint I/O
    Q  composite loss, inference contract, and the metric definitions

WHY N AND P ARE PINNED TO LITERALS
----------------------------------
`1453912` and `ARCHITECTURE_VERSION` are the two facts that decide whether a
checkpoint on disk belongs to this build. A shape check catches a changed
tensor; only the version string and the parameter count catch a head that was
rebuilt with a different but shape-compatible wiring -- which is the failure
that produces a plausible, wrong answer.

WHY M TESTS THE GLOBAL RNG
--------------------------
The documented claim is that the shuffle uses a PRIVATE generator so two loops
over the same data produce the same order "regardless of what else the process
has drawn from `random`". That is only true if the loop does not touch the
global state, so the test measures the global state before and after rather than
just comparing two shuffles.
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from core.errors import ModelLoadError, SpecialistError
from training.change_vqa import collate as C
from training.change_vqa import dataset as D
from training.change_vqa import evaluate as E
from training.change_vqa import model as M
from training.change_vqa import vocab as V
from training.change_vqa.features import CHANGE_FEATURE_DIM, TEXT_FEATURE_DIM


@pytest.fixture
def scratch() -> Path:
    """A self-managed scratch directory; see the note in the dataset test module.

    `tmp_path` performs directory removals, which a sandbox delete guard has
    blocked in this repository.
    """
    return Path(tempfile.mkdtemp(prefix="sq_r02_head_"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _record(question_id: int, *, scene_key: str | None = None, **overrides: Any):
    scene = scene_key or f"scene{question_id}"
    base: dict[str, Any] = {
        "split": "Train",
        "question_id": question_id,
        "file_name": f"{scene}.png",
        "scene_key": scene,
        "qtype": "change_or_not",
        "qtype_index": V.QUESTION_TYPE_TO_INDEX["change_or_not"],
        "resolved_qtype": "change_or_not",
        "qtype_resolved": True,
        "temporal_ref": "unspecified",
        "change_class": None,
        "question": "Have the areas of buildings changed?",
        "answer": "yes",
        "answer_index": V.ANSWER_TO_INDEX["yes"],
        "t1_path": f"/d/im1/{scene}.png",
        "t2_path": f"/d/im2/{scene}.png",
    }
    base.update(overrides)
    return D.ChangeVQARecord(**base)


def _change_cache(scene_keys: list[str]) -> Any:
    return __import__(
        "training.change_vqa.features", fromlist=["ChangeFeatureCache"]
    ).ChangeFeatureCache(
        scene_keys=tuple(scene_keys),
        features=np.zeros((len(scene_keys), CHANGE_FEATURE_DIM), dtype=np.float32),
        spec_hash="s",
        extractor_config={"trained": True},
    )


def _text_cache(question_ids: list[int]) -> Any:
    from training.change_vqa.features import TextFeatureCache

    return TextFeatureCache(
        question_ids=tuple(question_ids),
        features=np.zeros((len(question_ids), TEXT_FEATURE_DIM), dtype=np.float32),
        spec_hash="t",
    )


def _targets(file_names: list[str]) -> dict[str, D.SceneTargets]:
    return {
        name: D.SceneTargets(
            file_name=name,
            scene_key=Path(name).stem,
            class_mag=(0.1,) * V.N_CHANGE_CLASSES,
            class_delta=(0.0,) * V.N_CHANGE_CLASSES,
            class_ratio=(0.2,) * V.N_CHANGE_CLASSES,
            total_changed=0.25,
            n_pixels=65536,
            n_labeled_pixels=131072,
            label1_sha256="a" * 64,
            label2_sha256="b" * 64,
        )
        for name in file_names
    }


def _predictions(n: int, *, seed: int = 0) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    logits = generator.normal(size=(n, V.N_ANSWERS))
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    top2 = np.sort(probabilities, axis=1)[:, -2:]
    return {
        "answer_index": probabilities.argmax(axis=1),
        "answer": [V.INDEX_TO_ANSWER[int(i)] for i in probabilities.argmax(axis=1)],
        "confidence": top2[:, 1],
        "margin": top2[:, 1] - top2[:, 0],
        "probabilities": probabilities,
        "class_mag": np.full((n, V.N_CHANGE_CLASSES), 0.2, dtype=np.float32),
        "class_delta": np.zeros((n, V.N_CHANGE_CLASSES), dtype=np.float32),
        "total_changed": np.full(n, 0.3, dtype=np.float32),
        "legal_mask_applied": False,
    }


# ===========================================================================
# Area M - batching
# ===========================================================================
def test_m_a_fully_available_batch_assembles_with_the_declared_shapes() -> None:
    records = [_record(i) for i in range(4)]
    batch = C.build_batch(
        records,
        change_cache=_change_cache([f"scene{i}" for i in range(4)]),
        text_cache=_text_cache([0, 1, 2, 3]),
        targets=_targets([f"scene{i}.png" for i in range(4)]),
    )
    assert len(batch) == 4
    assert batch.n_skipped == 0
    assert batch.change_features.shape == (4, CHANGE_FEATURE_DIM)
    assert batch.text_features.shape == (4, TEXT_FEATURE_DIM)
    assert batch.answer_index.shape == (4,)
    assert batch.class_mag.shape == (4, V.N_CHANGE_CLASSES)
    assert batch.total_changed.shape == (4,)
    assert batch.question_ids == [0, 1, 2, 3]


def test_m_a_missing_change_feature_is_counted_not_silently_dropped() -> None:
    records = [_record(i) for i in range(3)]
    batch = C.build_batch(
        records,
        change_cache=_change_cache(["scene0", "scene2"]),  # scene1 absent
        text_cache=_text_cache([0, 1, 2]),
        targets=_targets([f"scene{i}.png" for i in range(3)]),
    )
    assert batch.question_ids == [0, 2]
    assert batch.n_skipped_missing_change_feature == 1
    assert batch.skip_summary()["missing_change_feature"] == 1


def test_m_a_missing_text_feature_is_counted_separately() -> None:
    records = [_record(i) for i in range(3)]
    batch = C.build_batch(
        records,
        change_cache=_change_cache([f"scene{i}" for i in range(3)]),
        text_cache=_text_cache([0, 2]),  # question 1 absent
        targets=_targets([f"scene{i}.png" for i in range(3)]),
    )
    assert batch.question_ids == [0, 2]
    assert batch.n_skipped_missing_text_feature == 1
    assert batch.n_skipped_missing_change_feature == 0


def test_m_missing_scene_targets_are_counted() -> None:
    records = [_record(i) for i in range(3)]
    batch = C.build_batch(
        records,
        change_cache=_change_cache([f"scene{i}" for i in range(3)]),
        text_cache=_text_cache([0, 1, 2]),
        targets=_targets(["scene0.png", "scene2.png"]),
    )
    assert batch.question_ids == [0, 2]
    assert batch.n_skipped_missing_targets == 1


def test_m_a_batch_that_assembles_nothing_raises() -> None:
    """An empty batch would let a training loop run to completion while
    training on nothing."""
    with pytest.raises(SpecialistError, match="none of 2 record"):
        C.build_batch(
            [_record(0), _record(1)],
            change_cache=_change_cache([]),
            text_cache=_text_cache([]),
            targets={},
        )


def test_m_shuffling_is_reproducible_from_its_seed() -> None:
    records = [_record(i) for i in range(10)]
    caches = dict(
        change_cache=_change_cache([f"scene{i}" for i in range(10)]),
        text_cache=_text_cache(list(range(10))),
        targets=_targets([f"scene{i}.png" for i in range(10)]),
    )
    first = [
        b.question_ids
        for b in C.iter_batches(records, batch_size=3, shuffle=True, seed=7, **caches)
    ]
    second = [
        b.question_ids
        for b in C.iter_batches(records, batch_size=3, shuffle=True, seed=7, **caches)
    ]
    assert first == second
    assert sorted(i for chunk in first for i in chunk) == list(range(10))


def test_m_different_seeds_produce_different_orders() -> None:
    records = [_record(i) for i in range(20)]
    caches = dict(
        change_cache=_change_cache([f"scene{i}" for i in range(20)]),
        text_cache=_text_cache(list(range(20))),
        targets=_targets([f"scene{i}.png" for i in range(20)]),
    )
    a = [b.question_ids for b in C.iter_batches(records, batch_size=20, shuffle=True, seed=1, **caches)]
    b = [b.question_ids for b in C.iter_batches(records, batch_size=20, shuffle=True, seed=2, **caches)]
    assert a != b


def test_m_batching_does_not_disturb_the_global_random_state() -> None:
    """Measured before and after, because comparing two shuffles would pass even
    if the loop reseeded the global generator."""
    records = [_record(i) for i in range(8)]
    caches = dict(
        change_cache=_change_cache([f"scene{i}" for i in range(8)]),
        text_cache=_text_cache(list(range(8))),
        targets=_targets([f"scene{i}.png" for i in range(8)]),
    )
    random.seed(1234)
    control = [random.random() for _ in range(5)]

    random.seed(1234)
    list(C.iter_batches(records, batch_size=3, shuffle=True, seed=42, **caches))
    after = [random.random() for _ in range(5)]

    assert after == control


def test_m_a_batch_size_below_one_is_rejected() -> None:
    with pytest.raises(SpecialistError, match="batch_size must be >= 1"):
        list(
            C.iter_batches(
                [_record(0)],
                change_cache=_change_cache(["scene0"]),
                text_cache=_text_cache([0]),
                targets=_targets(["scene0.png"]),
                batch_size=0,
            )
        )


def test_m_skip_summary_totals_across_batches() -> None:
    """Each chunk must still assemble at least one record.

    A chunk that assembles NOTHING raises by design (`build_batch`), so the skip
    totals can only be observed on chunks that are partially usable. The cache
    here holds scenes 0 and 2, so with `batch_size=2` the chunks are (0,1) and
    (2,3) -- each keeps one record and drops one.
    """
    records = [_record(i) for i in range(4)]
    batches = list(
        C.iter_batches(
            records,
            change_cache=_change_cache(["scene0", "scene2"]),
            text_cache=_text_cache([0, 1, 2, 3]),
            targets=_targets([f"scene{i}.png" for i in range(4)]),
            batch_size=2,
        )
    )
    assert len(batches) == 2
    assert [b.question_ids for b in batches] == [[0], [2]]
    totals = C.summarise_skips(batches)
    assert totals["missing_change_feature"] == 2
    assert totals["missing_text_feature"] == 0


# ===========================================================================
# Area N - head architecture
# ===========================================================================
@pytest.fixture(scope="module")
def head() -> Any:
    return M.build_change_vqa_head(
        change_feature_dim=CHANGE_FEATURE_DIM, text_feature_dim=TEXT_FEATURE_DIM
    )


def test_n_the_head_parameter_count_is_the_measured_one(head: Any) -> None:
    assert head.num_parameters() == 1453912
    assert head.trainable_parameters() == 1453912


def test_n_the_head_is_an_order_of_magnitude_smaller_than_the_detector(head: Any) -> None:
    """1.45 M against the frozen detector's 15.8 M -- the whole reason the 3 h
    budget holds without retraining STANet."""
    assert head.num_parameters() < 2_000_000
    assert head.num_parameters() * 10 < 15_779_969


def test_n_the_stage_one_estimator_has_thirteen_outputs(head: Any) -> None:
    """6 magnitudes + 6 signed deltas + 1 global fraction."""
    assert M.N_ESTIMATOR_OUTPUTS == 13
    assert M.N_ESTIMATOR_OUTPUTS == 2 * V.N_CHANGE_CLASSES + 1


def test_n_a_forward_pass_produces_the_declared_shapes(head: Any) -> None:
    out = head(
        torch.zeros(5, CHANGE_FEATURE_DIM),
        torch.zeros(5, TEXT_FEATURE_DIM),
        torch.zeros(5, dtype=torch.long),
        torch.zeros(5, dtype=torch.long),
    )
    assert tuple(out.answer_logits.shape) == (5, V.N_ANSWERS)
    assert tuple(out.class_mag.shape) == (5, V.N_CHANGE_CLASSES)
    assert tuple(out.class_delta.shape) == (5, V.N_CHANGE_CLASSES)
    assert tuple(out.total_changed.shape) == (5,)
    assert out.legal_mask_applied is False


def test_n_the_estimator_outputs_are_bounded_by_their_activations(head: Any) -> None:
    """sigmoid for magnitudes, tanh for signed deltas: a class cannot have
    changed by more than the whole scene."""
    out = head(
        torch.randn(8, CHANGE_FEATURE_DIM) * 5,
        torch.randn(8, TEXT_FEATURE_DIM),
        torch.randint(0, 8, (8,)),
        torch.randint(0, 3, (8,)),
    )
    assert bool((out.class_mag >= 0).all() and (out.class_mag <= 1).all())
    assert bool((out.class_delta >= -1).all() and (out.class_delta <= 1).all())
    assert bool((out.total_changed >= 0).all() and (out.total_changed <= 1).all())


def test_n_the_config_dict_carries_everything_needed_to_rebuild(head: Any) -> None:
    config = head.config_dict()
    assert config["architecture"] == M.ARCHITECTURE_VERSION
    assert config["change_feature_dim"] == CHANGE_FEATURE_DIM
    assert config["text_feature_dim"] == TEXT_FEATURE_DIM
    assert config["n_answers"] == 19
    assert config["n_change_classes"] == 6
    assert config["n_estimator_outputs"] == 13
    assert config["n_question_type_slots"] == V.N_QUESTION_TYPE_SLOTS
    assert config["n_temporal_reference_slots"] == V.N_TEMPORAL_REFERENCE_SLOTS
    assert config["answer_vocabulary"] == list(V.ANSWER_VOCABULARY)
    assert config["class_order"] == list(V.CHANGE_CLASS_ORDER)


def test_n_the_estimator_evidence_payload_is_auditable(head: Any) -> None:
    # `no_grad` mirrors serving (`predict_answers` runs under it). Without it,
    # `ChangeVQAOutput.to_evidence` converts a grad-carrying tensor to a scalar
    # and PyTorch emits a UserWarning -- see finding F6 in the status document.
    with torch.no_grad():
        out = head(
            torch.zeros(2, CHANGE_FEATURE_DIM),
            torch.zeros(2, TEXT_FEATURE_DIM),
            torch.zeros(2, dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
        )
        payload = out.to_evidence(index=1)
    assert payload["class_order"] == list(V.CHANGE_CLASS_ORDER)
    assert len(payload["class_change_magnitude"]) == V.N_CHANGE_CLASSES
    assert len(payload["class_change_delta"]) == V.N_CHANGE_CLASSES
    assert payload["estimator_is_learned"] is True
    assert 0.0 <= payload["total_changed_fraction"] <= 1.0


def test_n_to_evidence_is_warning_free_on_a_grad_carrying_tensor(head: Any) -> None:
    """F6 regression: `to_evidence` must not warn, and must not change a value.

    Serving never saw this: `predict_answers` calls `to_evidence` under
    `torch.no_grad()`. But a caller may legitimately build evidence from a
    TRAINING forward pass, where the tensors carry a grad function, and `float()`
    on such a tensor emits `UserWarning: Converting a tensor with
    requires_grad=True to a scalar` — a warning that teaches a reader to ignore
    warnings. The fix is `.detach()`, which removes the warning and is
    numerically exact; this test proves BOTH halves, so a future edit cannot
    silence the warning by rounding the number instead.
    """
    import warnings

    out = head(
        torch.zeros(2, CHANGE_FEATURE_DIM),
        torch.zeros(2, TEXT_FEATURE_DIM),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
    )
    # The precondition: this forward really does carry a grad function, so the
    # test is not vacuously warning-free.
    assert out.total_changed.requires_grad is True
    assert out.class_mag.requires_grad is True

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a failure
        from_grad = out.to_evidence(index=0)

    with torch.no_grad():
        detached = out.to_evidence(index=0)

    # Numerically identical, so the fix changed nothing observable.
    assert from_grad == detached
    assert isinstance(from_grad["total_changed_fraction"], float)
    assert isinstance(from_grad["class_change_magnitude"][0], float)


# ===========================================================================
# Area O - legal-answer masking
# ===========================================================================
def test_o_a_mask_restricts_the_argmax_to_legal_answers() -> None:
    logits = torch.randn(4, V.N_ANSWERS)
    mask = torch.zeros(V.N_ANSWERS, dtype=torch.bool)
    mask[V.ANSWER_TO_INDEX["yes"]] = True
    masked, applied = M._apply_legal_mask(logits, mask)
    assert applied is True
    assert masked.argmax(dim=1).tolist() == [V.ANSWER_TO_INDEX["yes"]] * 4


def test_o_an_all_false_mask_leaves_the_logits_untouched() -> None:
    """THE property of area O.

    Masking every answer would make the argmax collapse to index 0 -- a
    confident wrong answer with no trace of why. The guard leaves the row alone
    instead, and reports that it did not mask.
    """
    logits = torch.randn(3, V.N_ANSWERS)
    masked, applied = M._apply_legal_mask(logits, torch.zeros(3, V.N_ANSWERS, dtype=torch.bool))
    assert applied is False
    assert torch.equal(masked, logits)


def test_o_a_partially_empty_mask_only_masks_the_rows_that_have_a_legal_answer() -> None:
    logits = torch.zeros(2, V.N_ANSWERS)
    logits[0, 5] = 10.0  # row 0 would pick an illegal answer
    logits[1, 7] = 10.0  # row 1 has no legal answer at all
    mask = torch.zeros(2, V.N_ANSWERS, dtype=torch.bool)
    mask[0, 3] = True  # row 0: only index 3 is legal
    masked, applied = M._apply_legal_mask(logits, mask)
    assert applied is True
    assert masked.argmax(dim=1).tolist() == [3, 7]


def test_o_a_mask_of_the_wrong_width_is_rejected() -> None:
    with pytest.raises(SpecialistError, match="does not match logits"):
        M._apply_legal_mask(torch.randn(2, V.N_ANSWERS), torch.ones(3, 4, dtype=torch.bool))


# ===========================================================================
# Area P - checkpoint I/O
# ===========================================================================
def _fresh_head() -> Any:
    return M.build_change_vqa_head(
        change_feature_dim=CHANGE_FEATURE_DIM, text_feature_dim=TEXT_FEATURE_DIM
    )


def test_p_a_head_round_trips_and_reproduces_its_predictions(scratch: Path) -> None:
    original = _fresh_head()
    original.eval()
    path = M.save_change_vqa_head(
        scratch / "head.pt",
        original,
        metadata={"change_cache_spec": "c801326f85a185f8", "detector_trained": True},
    )
    assert (scratch / "model_metadata.json").exists()

    restored = M.load_change_vqa_head(path)
    features = torch.zeros(2, CHANGE_FEATURE_DIM)
    text = torch.zeros(2, TEXT_FEATURE_DIM)
    qidx = torch.zeros(2, dtype=torch.long)
    tidx = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        assert torch.allclose(
            original(features, text, qidx, tidx).answer_logits,
            restored(features, text, qidx, tidx).answer_logits,
        )


def test_p_a_loaded_head_is_marked_trained(scratch: Path) -> None:
    """`has_head` keys on this flag, so it is what makes the specialist willing
    to answer at all."""
    path = M.save_change_vqa_head(scratch / "head.pt", _fresh_head())
    restored = M.load_change_vqa_head(path)
    assert getattr(restored, "_satquery_trained", False) is True
    assert not getattr(_fresh_head(), "_satquery_trained", False)


def test_p_the_metadata_lands_beside_the_checkpoint(scratch: Path) -> None:
    metadata = {"change_cache_spec": "abc", "detector_trained": False, "seed": 42}
    M.save_change_vqa_head(scratch / "head.pt", _fresh_head(), metadata=metadata)
    written = json.loads((scratch / "model_metadata.json").read_text("utf-8"))
    assert written == metadata


def test_p_a_missing_checkpoint_raises(scratch: Path) -> None:
    with pytest.raises(ModelLoadError, match="not found"):
        M.load_change_vqa_head(scratch / "absent.pt")


def test_p_a_checkpoint_without_an_embedded_config_is_refused(scratch: Path) -> None:
    """Reconstructing the architecture by guessing is exactly what must not
    happen."""
    path = scratch / "head.pt"
    torch.save({"state_dict": _fresh_head().state_dict()}, path)
    with pytest.raises(ModelLoadError, match="no embedded config"):
        M.load_change_vqa_head(path)


def test_p_an_architecture_version_mismatch_is_refused(scratch: Path) -> None:
    head = _fresh_head()
    config = head.config_dict()
    config["architecture"] = "change_vqa_head_v0"
    path = scratch / "head.pt"
    torch.save({"state_dict": head.state_dict(), "config": config}, path)
    with pytest.raises(ModelLoadError, match="was written under architecture"):
        M.load_change_vqa_head(path)


def test_p_a_state_dict_from_another_width_is_refused(scratch: Path) -> None:
    small = M.build_change_vqa_head(change_feature_dim=64, text_feature_dim=TEXT_FEATURE_DIM)
    config = _fresh_head().config_dict()  # claims width 1045
    path = scratch / "head.pt"
    torch.save({"state_dict": small.state_dict(), "config": config}, path)
    with pytest.raises(ModelLoadError, match="does not match the reconstructed"):
        M.load_change_vqa_head(path)


# ===========================================================================
# Area Q - loss, inference, and metrics
# ===========================================================================
def test_q_the_composite_loss_has_four_terms_and_reports_its_weights(head: Any) -> None:
    n = 4
    out = head(
        torch.randn(n, CHANGE_FEATURE_DIM),
        torch.randn(n, TEXT_FEATURE_DIM),
        torch.zeros(n, dtype=torch.long),
        torch.zeros(n, dtype=torch.long),
    )
    breakdown = M.change_vqa_loss(
        out,
        answer_index=torch.zeros(n, dtype=torch.long),
        class_mag=torch.full((n, V.N_CHANGE_CLASSES), 0.2),
        class_delta=torch.zeros(n, V.N_CHANGE_CLASSES),
        total_changed=torch.full((n,), 0.3),
    )
    payload = breakdown.to_dict()
    for key in ("total", "answer_ce", "class_magnitude_bce", "class_delta_mse", "total_changed_bce"):
        assert key in payload
    assert breakdown.weights == M.DEFAULT_LOSS_WEIGHTS
    assert payload["weight_answer"] == 1.0
    assert payload["weight_total_changed"] == 0.5


def test_q_the_total_loss_is_the_weighted_sum_of_its_terms(head: Any) -> None:
    n = 3
    out = head(
        torch.randn(n, CHANGE_FEATURE_DIM),
        torch.randn(n, TEXT_FEATURE_DIM),
        torch.zeros(n, dtype=torch.long),
        torch.zeros(n, dtype=torch.long),
    )
    breakdown = M.change_vqa_loss(
        out,
        answer_index=torch.zeros(n, dtype=torch.long),
        class_mag=torch.full((n, V.N_CHANGE_CLASSES), 0.2),
        class_delta=torch.zeros(n, V.N_CHANGE_CLASSES),
        total_changed=torch.full((n,), 0.3),
    )
    # `.detach()` on each term: these tensors carry a grad function because the
    # loss is built for backprop, and `float()` on a grad-carrying tensor emits
    # `UserWarning: Converting a tensor with requires_grad=True to a scalar`.
    # Detaching is exact (the float is bit-identical), and it keeps the suite
    # warning-clean so a real warning is not lost in the noise.
    expected = (
        breakdown.weights["answer"] * float(breakdown.answer.detach())
        + breakdown.weights["magnitude"] * float(breakdown.magnitude.detach())
        + breakdown.weights["delta"] * float(breakdown.delta.detach())
        + breakdown.weights["total_changed"] * float(breakdown.total_changed.detach())
    )
    assert float(breakdown.total.detach()) == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# The mixed-precision failure that stopped the Kaggle run
# ---------------------------------------------------------------------------
# `aten::binary_cross_entropy` is registered as an ERROR under CUDA autocast:
#
#   RuntimeError: torch.nn.functional.binary_cross_entropy and
#   torch.nn.BCELoss are unsafe to autocast.
#
# The registration is unconditional, so it fires whatever the operand dtypes
# are -- casting to fp32 alone does NOT avoid it; the op must be invoked with
# autocast disabled. These tests pin the mechanism: that the helper really does
# run with autocast off and fp32 operands, and that it does not leak its own
# autocast state back to the caller.
#
# THIS FIX WAS NECESSARY BUT NOT SUFFICIENT. It made the BCE call autocast-safe;
# it did not make it numerically safe, and the next real run died on a NaN loss
# instead. The saturated-sigmoid defect, and the logits-based loss that fixes it,
# are pinned in section S below. Section Q is kept because the op-level
# registration is still real and the helper is still the fallback path.
#
# HONEST SCOPE: the CUDA guard itself cannot be triggered on a CPU-only host.
# `torch.amp.autocast("cuda")` there is a no-op that warns and disables itself,
# so what is proven here is the *mechanism* -- that no BCE call is made while
# autocast is active -- not the CUDA RuntimeError itself. The trainer's AMP
# dtype policy is covered separately in `tests/unit/test_change_vqa_train.py`.


def test_q_the_bce_helper_is_exactly_plain_bce_on_the_same_probabilities() -> None:
    """The fix must not change the number, only how the op is invoked."""
    torch.manual_seed(0)
    prediction = torch.sigmoid(torch.randn(32, V.N_CHANGE_CLASSES))
    target = torch.rand(32, V.N_CHANGE_CLASSES)
    expected = torch.nn.functional.binary_cross_entropy(prediction, target)
    actual = M.binary_cross_entropy_probabilities(prediction, target)
    assert float(actual.detach()) == float(expected.detach())


def test_q_the_bce_helper_disables_autocast_and_casts_to_float32(
    monkeypatch: Any,
) -> None:
    """The mechanism: inside an ACTIVE autocast block the call still lands fp32."""
    seen: list[tuple[bool, str, str]] = []
    real = torch.nn.functional.binary_cross_entropy

    def spy(prediction: Any, target: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(
            (
                torch.is_autocast_enabled(prediction.device.type),
                str(prediction.dtype),
                str(target.dtype),
            )
        )
        return real(prediction, target, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "binary_cross_entropy", spy)

    before = torch.is_autocast_enabled("cpu")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu") is True
        prediction = torch.sigmoid(torch.randn(8, V.N_CHANGE_CLASSES)).to(
            torch.bfloat16
        )
        target = torch.rand(8, V.N_CHANGE_CLASSES)
        value = M.binary_cross_entropy_probabilities(prediction, target)

    assert seen == [(False, "torch.float32", "torch.float32")]
    assert torch.isfinite(value).all()
    # The helper must not leak its own autocast state to the caller.
    assert torch.is_autocast_enabled("cpu") is before


def test_q_the_composite_loss_runs_inside_an_active_autocast_block(
    head: Any,
) -> None:
    """The exact context the trainer uses. Before the fix this raised on CUDA."""
    n = 8
    torch.manual_seed(0)
    change = torch.randn(n, CHANGE_FEATURE_DIM)
    text = torch.randn(n, TEXT_FEATURE_DIM)
    qtype = torch.zeros(n, dtype=torch.long)
    temporal = torch.zeros(n, dtype=torch.long)
    answer = torch.zeros(n, dtype=torch.long)
    mag = torch.full((n, V.N_CHANGE_CLASSES), 0.2)
    delta = torch.zeros(n, V.N_CHANGE_CLASSES)
    total = torch.full((n,), 0.3)

    def step() -> Any:
        out = head(change, text, qtype, temporal)
        return M.change_vqa_loss(
            out,
            answer_index=answer,
            class_mag=mag,
            class_delta=delta,
            total_changed=total,
        )

    reference = float(step().total.detach())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        under_autocast = step()
    assert torch.isfinite(under_autocast.total)
    # The tolerance is loose ON PURPOSE and the reason is worth stating: the two
    # BCE terms are now computed in fp32, but the answer cross-entropy still runs
    # on bfloat16 logits (it is autocast-safe and it is where the FLOPs are), so
    # the totals are not bit-identical. Measured spread on this suite is ~0.2%;
    # the assertion is here to catch a *broken* loss, not to pin a float.
    assert float(under_autocast.total.detach()) == pytest.approx(
        reference, rel=0.01
    )


def test_q_the_estimator_still_emits_probabilities_so_the_contract_holds(
    head: Any,
) -> None:
    """Pins the published contract: the estimator's outputs are probabilities.

    `class_mag`/`total_changed` are `sigmoid` outputs and are published as the
    answer's evidence, so they cannot be replaced by logits without training
    against one quantity and serving another. If someone ever changes the
    activations, this test fails and forces the serving path to be reconsidered.

    The loss no longer *reads* these values -- it reads the logits behind them,
    because BCE-on-probabilities is numerically unusable at saturation (section
    R). The emitted values are what this test guards, and they are unchanged.
    """
    n = 16
    out = head(
        torch.randn(n, CHANGE_FEATURE_DIM),
        torch.randn(n, TEXT_FEATURE_DIM),
        torch.zeros(n, dtype=torch.long),
        torch.zeros(n, dtype=torch.long),
    )
    # `.detach()` throughout: these carry a grad function, and `float()` on a
    # grad-carrying tensor warns. The suite is kept warning-clean on purpose so
    # that a real warning is not lost in the noise.
    assert float(out.class_mag.detach().min()) >= 0.0
    assert float(out.class_mag.detach().max()) <= 1.0
    assert float(out.total_changed.detach().min()) >= 0.0
    assert float(out.total_changed.detach().max()) <= 1.0
    assert float(out.class_delta.detach().min()) >= -1.0
    assert float(out.class_delta.detach().max()) <= 1.0

    target = torch.full((n, V.N_CHANGE_CLASSES), 0.2)
    breakdown = M.change_vqa_loss(
        out,
        answer_index=torch.zeros(n, dtype=torch.long),
        class_mag=target,
        class_delta=torch.zeros(n, V.N_CHANGE_CLASSES),
        total_changed=torch.full((n,), 0.3),
    )
    # The loss agrees with the probability form here because the two forms are
    # the SAME function away from saturation. Section R covers the saturated
    # regime, where the probability form diverges and the logits form does not.
    expected = torch.nn.functional.binary_cross_entropy(out.class_mag, target)
    assert float(breakdown.magnitude.detach()) == pytest.approx(
        float(expected.detach()), rel=1e-6
    )


def test_q_predict_answers_returns_a_normalised_distribution(head: Any) -> None:
    """`probabilities` is what makes top-3 computable rather than silently
    skipped."""
    n = 6
    predictions = M.predict_answers(
        head,
        change_features=np.zeros((n, CHANGE_FEATURE_DIM), dtype=np.float32),
        text_features=np.zeros((n, TEXT_FEATURE_DIM), dtype=np.float32),
        qtype_indices=np.zeros(n, dtype=np.int64),
        temporal_indices=np.zeros(n, dtype=np.int64),
        qtypes=["change_or_not"] * n,
    )
    assert predictions["probabilities"].shape == (n, V.N_ANSWERS)
    assert np.allclose(predictions["probabilities"].sum(axis=1), 1.0, atol=1e-5)
    # confidence is the top-1 probability and margin is top1 - top2.
    assert np.allclose(predictions["confidence"], predictions["probabilities"].max(axis=1))
    top2 = np.sort(predictions["probabilities"], axis=1)[:, -2:]
    assert np.allclose(predictions["margin"], top2[:, 1] - top2[:, 0])


def test_q_inference_with_a_type_mask_never_returns_an_illegal_answer(head: Any) -> None:
    n = 5
    predictions = M.predict_answers(
        head,
        change_features=np.zeros((n, CHANGE_FEATURE_DIM), dtype=np.float32),
        text_features=np.zeros((n, TEXT_FEATURE_DIM), dtype=np.float32),
        qtype_indices=np.zeros(n, dtype=np.int64),
        temporal_indices=np.zeros(n, dtype=np.int64),
        qtypes=["change_or_not"] * n,
        apply_type_mask=True,
    )
    legal = V.LEGAL_ANSWERS["change_or_not"]
    assert all(answer in legal for answer in predictions["answer"])
    assert predictions["legal_mask_applied"] is True


def test_q_answer_accuracy_is_exact_match() -> None:
    assert E.answer_accuracy([0, 1, 2], [0, 1, 2]) == 1.0
    assert E.answer_accuracy([0, 1, 2], [0, 1, 3]) == pytest.approx(2 / 3)
    assert E.answer_accuracy([0], []) == 0.0


def test_q_top_k_accuracy_distinguishes_a_near_miss_from_a_wild_one() -> None:
    """Several ratio bins are adjacent, so a rank-2 gold is not the same error
    as a gold outside the top 3."""
    probabilities = np.zeros((1, V.N_ANSWERS))
    probabilities[0, 5] = 0.6
    probabilities[0, 7] = 0.3
    probabilities[0, 9] = 0.1
    assert E.top_k_accuracy(probabilities, [7], k=1) == 0.0
    assert E.top_k_accuracy(probabilities, [7], k=2) == 1.0
    assert E.top_k_accuracy(probabilities, [9], k=3) == 1.0
    assert E.top_k_accuracy(probabilities, [9], k=2) == 0.0


def test_q_top_k_accuracy_rejects_a_wrongly_shaped_distribution() -> None:
    with pytest.raises(ValueError, match="must be"):
        E.top_k_accuracy(np.zeros((2, 5)), [0, 1], k=3)


def test_q_macro_f1_is_one_for_a_perfect_prediction() -> None:
    assert E.macro_f1([0, 1, 2], [0, 1, 2]) == 1.0


def test_q_macro_f1_averages_over_observed_labels_not_all_nineteen() -> None:
    """Averaging over all 19 would count an answer that never appears as a
    zero-F1 class, making the score depend on the split's vocabulary."""
    # Two labels, one perfectly predicted, one never predicted at all.
    score = E.macro_f1([0, 0], [0, 1])
    assert 0.0 < score < 1.0
    assert E.macro_f1([], []) == 0.0


def test_q_confusion_matrix_is_rows_gold_columns_predicted() -> None:
    matrix = E.confusion_matrix([1, 1, 2], [1, 2, 2], normalise=False)
    assert matrix.shape == (V.N_ANSWERS, V.N_ANSWERS)
    assert matrix[1, 1] == 1
    assert matrix[2, 1] == 1
    assert matrix[2, 2] == 1
    assert matrix.sum() == 3


def test_q_a_normalised_confusion_row_sums_to_one_for_a_present_gold() -> None:
    matrix = E.confusion_matrix([1, 1, 2], [1, 2, 2], normalise=True)
    assert matrix[1].sum() == pytest.approx(1.0)
    assert matrix[2].sum() == pytest.approx(1.0)
    # An answer that never occurs has no row mass and must not divide by zero.
    assert matrix[5].sum() == 0.0


def test_q_per_type_accuracy_reports_none_for_a_type_that_was_never_asked() -> None:
    """Printing 0.0 for an unasked type would read as total failure."""
    records = [_record(0), _record(1)]
    report = E.per_type_accuracy(records, [records[0].answer_index] * 2)
    assert report["change_or_not"]["support"] == 2
    assert report["change_or_not"]["accuracy"] == 1.0
    for qtype in V.QUESTION_TYPE_ORDER:
        if qtype == "change_or_not":
            continue
        assert report[qtype]["support"] == 0
        assert report[qtype]["accuracy"] is None


def test_q_estimator_errors_are_zero_against_matching_targets() -> None:
    targets = [_targets(["scene0.png"])["scene0.png"]]
    mag = np.asarray([targets[0].class_mag], dtype=np.float64)
    delta = np.asarray([targets[0].class_delta], dtype=np.float64)
    total = np.asarray([targets[0].total_changed], dtype=np.float64)
    errors = E.estimator_errors(mag, delta, total, targets)
    assert errors["class_magnitude_mae"] == 0.0
    assert errors["class_delta_mae"] == 0.0
    assert errors["total_changed_mae"] == 0.0
    assert set(errors["class_magnitude_mae_per_class"]) == set(V.CHANGE_CLASS_ORDER)


# ---------------------------------------------------------------------------
# Area Q - the report refuses captioning metrics
# ---------------------------------------------------------------------------
def _report(n: int = 6, *, seed: int = 3):
    records = [_record(i) for i in range(n)]
    return E.evaluate_records(
        records,
        _predictions(n, seed=seed),
        targets=[],
        split="Val",
        checkpoint_path=None,
        checkpoint_sha256=None,
        config_hash="78f1e3700da15aa1",
        inference_config={"batch_size": n, "apply_type_mask": False},
        n_failed=1,
        n_skipped=2,
    )


def test_q_the_report_carries_provenance() -> None:
    payload = _report().to_dict()
    assert payload["metric_implementation"] == E.METRIC_IMPLEMENTATION
    assert payload["dataset"]["dataset_id"] == "cdvqa"
    assert payload["dataset"]["split"] == "Val"
    assert payload["dataset"]["preprocessing_version"] == D.PREPROCESSING_VERSION
    assert payload["config_hash"] == "78f1e3700da15aa1"
    assert payload["checkpoint"]["sha256"] == "unavailable"
    assert payload["inference_config"]["apply_type_mask"] is False


def test_q_the_report_counts_failures_and_skips_rather_than_dropping_them() -> None:
    payload = _report().to_dict()
    assert payload["counts"] == {"n_scored": 6, "n_failed": 1, "n_skipped": 2}


def test_q_the_report_contains_no_captioning_metric() -> None:
    """BLEU/ROUGE-L/CIDEr/BERTScore are CAPTIONING metrics (plan section 2.2);
    R-16 governs them and is open. Reporting one here would describe a task the
    system does not perform."""
    metrics = _report().to_dict()["metrics"]
    forbidden = {"bleu", "rouge", "rouge_l", "cider", "bertscore", "meteor"}
    assert not (set(metrics) & forbidden)
    assert "answer_accuracy" in metrics
    assert "macro_f1" in metrics
    assert "top_3_accuracy" in metrics
    assert "per_type" in metrics
    assert "captioning" in _report().to_dict()["note"].lower()


def test_q_the_report_accuracy_property_reads_the_primary_metric() -> None:
    report = _report()
    assert report.accuracy == report.to_dict()["metrics"]["answer_accuracy"]
    assert 0.0 <= report.accuracy <= 1.0


def test_q_the_report_hashes_its_own_payload_stably() -> None:
    report = _report()
    assert report.sha256() == report.sha256()
    assert len(report.sha256()) == 64


def test_q_the_report_omits_the_estimator_block_when_no_targets_are_given() -> None:
    assert "estimator" not in _report().to_dict()["metrics"]


# ---------------------------------------------------------------------------
# Area Q - qualitative examples
# ---------------------------------------------------------------------------
def test_q_qualitative_examples_include_failures_and_list_them_first() -> None:
    """A qualitative section that shows only correct predictions is a marketing
    document."""
    records = [_record(i) for i in range(8)]
    predictions = _predictions(8, seed=11)
    # Force a known split of correct/incorrect.
    gold = [records[i].answer_index for i in range(8)]
    predictions["answer_index"] = np.asarray(gold, dtype=np.int64).copy()
    predictions["answer_index"][0] = (gold[0] + 1) % V.N_ANSWERS
    predictions["answer_index"][1] = (gold[1] + 2) % V.N_ANSWERS

    payload = E.qualitative_examples(
        records, predictions, targets=[], n_success=5, n_failure=5
    )
    assert payload["n_correct"] == 6
    assert payload["n_incorrect"] == 2
    assert len(payload["failures"]) == 2
    assert len(payload["successes"]) == 5
    keys = list(payload)
    assert keys.index("failures") < keys.index("successes")


def test_q_every_qualitative_entry_declares_its_confidence_uncalibrated() -> None:
    records = [_record(i) for i in range(4)]
    payload = E.qualitative_examples(
        records, _predictions(4), targets=[], n_success=4, n_failure=4
    )
    entries = payload["failures"] + payload["successes"]
    assert entries
    for entry in entries:
        assert entry["confidence_is_calibrated"] is False
        assert entry["t1_path"] and entry["t2_path"]
        assert entry["question"] and entry["gold_answer"]
        assert set(entry["class_change_magnitude"]) == set(V.CHANGE_CLASS_ORDER)


def test_q_qualitative_examples_attach_the_label_derived_targets_when_given() -> None:
    records = [_record(i) for i in range(2)]
    targets = [_targets([f"scene{i}.png" for i in range(2)])[f"scene{i}.png"] for i in range(2)]
    payload = E.qualitative_examples(
        records, _predictions(2), targets=targets, n_success=2, n_failure=2
    )
    entries = payload["failures"] + payload["successes"]
    assert entries
    for entry in entries:
        assert entry["label_derived_targets"]["total_changed"] == 0.25
        assert entry["label_derived_targets"]["label1_sha256"] == "a" * 64


# ---------------------------------------------------------------------------
# Section S -- the saturated-BCE failure that stopped the second Kaggle run
# ---------------------------------------------------------------------------
# Deliberately NOT called "Area R". The brief's areas run A-R, and Area R is the
# integration area (tests/unit/test_change_vqa_integration.py, whose tests are
# `test_r_*`). This is a post-brief section added after the first external run,
# so it takes the next free letter and the `test_s_*` prefix: two different
# things called "R" in one suite is exactly the ambiguity F5 already warns about.
#
# The first fix made the BCE call *autocast-safe*. It did not make it
# *numerically safe*, and the real run then died at epoch 3 batch 61 with:
#
#   SpecialistError: epoch 3 batch 61/258: the loss is nan, which is not finite.
#   Components: {'total': nan, 'answer_ce': nan,
#                'class_magnitude_bce': 23.5, 'class_delta_mse': 0.98,
#                'total_changed_bce': 21.4}
#
# The SHAPE of that message is the diagnosis: the two BCE terms are finite and
# large, and only `answer_ce` is nan. Section S pins the chain that produces it.
#
#   `sigmoid(17.0)` is exactly 1.0 in fp32, and fp16 saturates at |z| >= 18.
#   BCE on a saturated probability has `d/dp = (p - t)/(p(1 - p))`, which PyTorch
#   clamps to a FINITE -9.99999996e11. Finite means GradScaler neither flags nor
#   skips it. On the way back through the `.float()` node that value is cast to
#   fp16, where it becomes -inf; the sigmoid's own backward then multiplies it by
#   `sigmoid'(z) = 0`, and `inf * 0` is nan. The nan lands on the estimator's
#   weights, so the NEXT forward pass yields a nan `answer_logits` -- which is
#   exactly `answer_ce: nan` beside finite BCE terms.
#
# Computing the loss from the LOGITS removes the chain: the backward becomes
# `sigmoid(z) - t`, bounded in [-1, 1], with no division by a probability
# anywhere. The emitted probabilities are unchanged, so the published contract,
# `to_evidence()`, `predict_answers()` and serving are all untouched.
#
# This is NOT an AMP-only bug. fp32 saturates at the same logits; it merely
# multiplies the 1e12 by 0 instead of by inf, so `--no-amp` would have hidden the
# crash while leaving the loss silently WRONG -- a clamped 100.0 per saturated
# element where the true value is 30.0.


def _saturated_fp16_head(batch: int = 8) -> tuple[Any, Any, dict[str, Any]]:
    """A head driven into fp16 sigmoid saturation, with targets opposing it.

    The feature dimensionality is the real one; only the magnitude is pushed.
    That is not contrived: real cached change features reach an absolute max of
    40.0 and a row L2 norm of 382.0, so a logit magnitude past 18 is reachable
    in a real run -- which is why this happened on Kaggle and not only here.
    """
    torch.manual_seed(0)
    head = M.build_change_vqa_head(
        change_feature_dim=CHANGE_FEATURE_DIM, text_feature_dim=TEXT_FEATURE_DIM
    ).half()
    change = torch.full((batch, CHANGE_FEATURE_DIM), 500.0, dtype=torch.float16)
    text = torch.zeros((batch, TEXT_FEATURE_DIM), dtype=torch.float16)
    qtype = torch.zeros(batch, dtype=torch.long)
    temporal = torch.zeros(batch, dtype=torch.long)
    output = head(change, text, qtype, temporal)
    targets: dict[str, Any] = {
        "answer_index": torch.zeros(batch, dtype=torch.long),
        "class_mag": torch.zeros(batch, V.N_CHANGE_CLASSES),
        "class_delta": torch.zeros(batch, V.N_CHANGE_CLASSES),
        "total_changed": torch.zeros(batch),
    }
    return head, output, targets


def _nonfinite_parameter_grads(module: Any) -> int:
    return sum(
        1
        for parameter in module.parameters()
        if parameter.grad is not None
        and not bool(torch.isfinite(parameter.grad).all())
    )


def test_s_the_fp16_saturation_this_reproduces_is_reachable() -> None:
    """Guard the premise: the logits really do saturate, in fp16, at this scale."""
    _head, output, _targets = _saturated_fp16_head()
    # `.detach()` because these tensors carry a grad_fn; `float()` on one warns.
    assert float(output.mag_logits.detach().abs().max()) > 18.0
    # fp16 sigmoid is exactly 0 or 1 out here, which is what starts the chain.
    assert set(output.class_mag.detach().flatten().tolist()) <= {0.0, 1.0}


def test_s_the_pre_fix_probability_loss_does_poison_the_gradients() -> None:
    """The defect, reproduced through the ops the pre-fix loss used.

    This is the failing half of the regression pair. It reconstructs the old
    loss exactly -- unclamped BCE on the emitted probabilities -- and asserts
    that it damages the gradients. If a future change made the probability form
    safe, this test would fail and the fix could be reconsidered.
    """
    import torch.nn.functional as F

    head, output, targets = _saturated_fp16_head()
    pre_fix = (
        F.cross_entropy(output.answer_logits, targets["answer_index"])
        + F.binary_cross_entropy(output.class_mag.float(), targets["class_mag"])
        + F.mse_loss(output.class_delta, targets["class_delta"])
        + 0.5
        * F.binary_cross_entropy(
            output.total_changed.float(), targets["total_changed"]
        )
    )
    pre_fix.backward()
    assert _nonfinite_parameter_grads(head) > 0, (
        "the pre-fix probability loss was expected to produce non-finite "
        "gradients at fp16 saturation; it did not, so this regression pair no "
        "longer demonstrates anything"
    )


def test_s_the_fixed_loss_keeps_every_gradient_finite_at_the_same_point() -> None:
    """The passing half: the same point through `change_vqa_loss` is clean."""
    head, output, targets = _saturated_fp16_head()
    loss = M.change_vqa_loss(output, **targets)
    assert bool(torch.isfinite(loss.total))
    loss.total.backward()
    assert _nonfinite_parameter_grads(head) == 0


def test_s_the_fixed_loss_is_exact_where_the_probability_form_is_clamped() -> None:
    """The loss must MEASURE the error rather than saturate at 100 per element."""
    import torch.nn.functional as F

    _head, output, targets = _saturated_fp16_head()
    fixed = float(M.change_vqa_loss(output, **targets).magnitude.detach())
    exact = float(
        F.binary_cross_entropy_with_logits(
            output.mag_logits.detach().float(), targets["class_mag"]
        )
    )
    clamped = float(
        F.binary_cross_entropy(
            output.class_mag.detach().float(), targets["class_mag"]
        )
    )
    assert fixed == pytest.approx(exact, rel=1e-5)
    # The probability form badly under-reports; that is the second defect.
    assert clamped < exact * 0.5


def test_s_the_fix_is_the_same_function_away_from_saturation() -> None:
    """Equivalence in the interior: this is not a loss redesign."""
    torch.manual_seed(3)
    logits = torch.randn(64, V.N_CHANGE_CLASSES) * 2.0
    target = torch.rand(64, V.N_CHANGE_CLASSES)
    via_logits = M.binary_cross_entropy_from_logits(logits, target)
    via_probabilities = M.binary_cross_entropy_probabilities(
        torch.sigmoid(logits), target
    )
    assert float(via_logits) == pytest.approx(
        float(via_probabilities), rel=1e-5, abs=1e-7
    )


def test_s_the_logits_are_an_addition_not_a_replacement() -> None:
    """The published contract is untouched: forward still emits sigmoid values."""
    torch.manual_seed(0)
    head = M.build_change_vqa_head(
        change_feature_dim=CHANGE_FEATURE_DIM, text_feature_dim=TEXT_FEATURE_DIM
    )
    change = torch.randn(4, CHANGE_FEATURE_DIM)
    text = torch.randn(4, TEXT_FEATURE_DIM)
    zero = torch.zeros(4, dtype=torch.long)
    output = head(change, text, zero, zero)

    assert output.mag_logits is not None
    assert output.total_logits is not None
    assert torch.allclose(output.class_mag, torch.sigmoid(output.mag_logits))
    assert torch.allclose(
        output.total_changed, torch.sigmoid(output.total_logits)
    )

    # The audit payload still carries probabilities, never logits.
    evidence = output.to_evidence(0)
    assert all(0.0 <= v <= 1.0 for v in evidence["class_change_magnitude"])
    assert 0.0 <= evidence["total_changed_fraction"] <= 1.0


def test_s_the_probability_fallback_still_serves_a_hand_built_output() -> None:
    """An output constructed without logits must keep working, on the fallback."""
    import torch.nn.functional as F

    torch.manual_seed(5)
    logits = torch.randn(16, V.N_CHANGE_CLASSES)
    target = torch.rand(16, V.N_CHANGE_CLASSES)
    output = M.ChangeVQAOutput(
        answer_logits=torch.zeros(16, V.N_ANSWERS),
        class_mag=torch.sigmoid(logits),
        class_delta=torch.zeros(16, V.N_CHANGE_CLASSES),
        total_changed=torch.zeros(16),
    )
    assert output.mag_logits is None and output.total_logits is None

    loss = M.change_vqa_loss(
        output,
        answer_index=torch.zeros(16, dtype=torch.long),
        class_mag=target,
        class_delta=torch.zeros(16, V.N_CHANGE_CLASSES),
        total_changed=torch.zeros(16),
    )
    assert bool(torch.isfinite(loss.total))
    expected = float(F.binary_cross_entropy(torch.sigmoid(logits), target))
    assert float(loss.magnitude) == pytest.approx(expected, rel=1e-5)
