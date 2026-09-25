"""Phase 4 tests — the intent router.

Three things are pinned here, in descending order of importance:

1. The router can never emit a task outside the frozen ontology, and its labels
   stay aligned with `core.schemas.Task`. A router that produced "describe"
   instead of "caption" would break the controller silently.
2. The split is by GROUP. The test that matters is not "does the good split
   pass" but "does a deliberately leaky split get caught" — the same adversarial
   discipline Gate 1 applies to scene-level leakage.
3. The deterministic fallback agrees with the label space on the curated hard
   negatives. If `"describe the water body"` and `"show me the water body"`
   land on the same task, the fallback is wrong.

Training tests use `encoder=None`, which selects the hash-based stub embedding.
That keeps the whole file offline and deterministic; a test that needs to
download a 91 MB model is a test nobody runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core.schemas import Intent, Modality, Task
from router.adapter import IntentAdapter
from router.classifier import IntentRouter, load_adapter, save_adapter
from router.dataset import (
    HARD_NEGATIVE_PREFIX,
    TEMPLATES,
    RouterCorpus,
    RouterExample,
    build_corpus,
    split_by_group,
    split_leakage_report,
)
from router.encoder import VERIFIED_EMBEDDING_DIM, VERIFIED_TOKENIZER_MAX_LENGTH
from router.fallback import is_available, lexical_route, self_check
from router.label_space import (
    BINARY_HEADS,
    MODALITY_CLASSES,
    NUM_BINARY_HEADS,
    NUM_MODALITIES,
    NUM_TASKS,
    TASK_CLASSES,
    TASK_TO_INDEX,
    is_valid_task,
)
from router.train import evaluate_split, train_router

# ---------------------------------------------------------------------------
# 1. Label space <-> schema alignment
# ---------------------------------------------------------------------------


#: The router's label space, pinned. It is SIX classes because the shipped task
#: head was trained on six: `NUM_TASKS` sizes the head's final layer, so adding
#: a class reshapes that layer and invalidates the checkpoint
#: (`artifacts/router/…`). It is therefore frozen, and any schema task the
#: router cannot emit is listed in `PLANNER_DERIVED_TASKS` below instead.
FROZEN_ROUTER_TASK_CLASSES: tuple[str, ...] = (
    "vqa",
    "caption",
    "grounding",
    "change",
    "optical_sar",
    "unsupported",
)

#: Schema tasks the router deliberately does NOT predict.
#:
#: `change_vqa` (added by R-02) is not a routing decision. A change question is
#: routed to `change`; the PLANNER then widens `change` + `language_output` to
#: `change_vqa` when a `change_vqa` capability is registered. The router is
#: never asked to tell `change` from `change_vqa`, so the two can share one
#: frozen class without ambiguity — and resizing the head to "fix" the
#: asymmetry would break the trained checkpoint for no gain.
PLANNER_DERIVED_TASKS: frozenset[str] = frozenset({"change_vqa"})


def test_task_classes_are_frozen_at_the_trained_six_class_space() -> None:
    """The router head is trained; its label space must not silently resize.

    This replaces an earlier assertion that `set(TASK_CLASSES)` equaled every
    `Task` value. That equality held only while the schema and the router
    ontology grew together; R-02 added `Task.CHANGE_VQA` to the schema without
    (and deliberately not) adding a seventh router class, so the equality is no
    longer the contract. The contract is now stated in three parts below: the
    router's space is pinned, it is a subset of the schema, and the difference
    is enumerated by name.
    """
    assert TASK_CLASSES == FROZEN_ROUTER_TASK_CLASSES
    assert NUM_TASKS == 6


def test_every_router_class_is_a_schema_task() -> None:
    """One direction of the alignment, and the one that must never break: the
    controller keys off `core.schemas.Task`, so a router class the schema does
    not define would route to a workflow that does not exist."""
    schema_values = {t.value for t in Task}
    assert set(TASK_CLASSES) <= schema_values


def test_schema_tasks_the_router_does_not_predict_are_enumerated() -> None:
    """The other direction, stated explicitly rather than left implicit.

    The schema is allowed to be larger than the router's frozen space. What is
    NOT allowed is for the difference to be accidental: every such task must
    appear in `PLANNER_DERIVED_TASKS`, so growing the gap requires editing a
    named constant and cannot happen by drift.
    """
    schema_values = {t.value for t in Task}
    assert schema_values - set(TASK_CLASSES) == PLANNER_DERIVED_TASKS


def test_change_vqa_is_planner_derived_not_router_predicted() -> None:
    """R-02 regression (finding F7): `Task.CHANGE_VQA` exists in the schema but
    is NOT a router class, and must not become one.

    The intended behaviour, end to end:

    * the router can never emit `change_vqa` — it is absent from both
      `TASK_CLASSES` and `TASK_TO_INDEX`, and `is_valid_task` rejects it;
    * a change question is routed to `change`;
    * the PLANNER widens `change` + `language_output` to `change_vqa` when the
      capability is registered — pinned in
      `tests/unit/test_planner.py::test_real_registry_declaration_drives_availability`
      and, at the specialist level, in
      `tests/unit/test_change_vqa_integration.py`.

    If a future change adds `change_vqa` to the label space, this test fails —
    which is the point, because that edit would resize the trained task head.
    """
    assert Task.CHANGE_VQA.value == "change_vqa"
    assert "change_vqa" not in TASK_CLASSES
    assert "change_vqa" not in TASK_TO_INDEX
    assert is_valid_task("change_vqa") is False
    assert NUM_TASKS == 6


def test_modality_classes_match_schema_modality_values() -> None:
    assert set(MODALITY_CLASSES) == {m.value for m in Modality}


def test_num_tasks_and_modalities_match_their_lists() -> None:
    assert NUM_TASKS == len(TASK_CLASSES)
    assert NUM_MODALITIES == len(MODALITY_CLASSES)
    assert NUM_BINARY_HEADS == len(BINARY_HEADS)


def test_task_indices_are_contiguous_and_unique() -> None:
    assert sorted(TASK_TO_INDEX.values()) == list(range(NUM_TASKS))


def test_unsupported_is_in_the_ontology() -> None:
    """The router must be able to refuse. Without this class the only options
    are 'guess' and 'crash'."""
    assert "unsupported" in TASK_CLASSES


# ---------------------------------------------------------------------------
# 2. Deterministic lexical fallback
# ---------------------------------------------------------------------------


def test_fallback_self_check_passes() -> None:
    failures = self_check()
    assert failures == [], (
        "fallback disagrees with the label space:\n  " + "\n  ".join(failures)
    )


def test_fallback_is_always_available() -> None:
    """No model, no download, no network. This is the whole point of it."""
    assert is_available() is True


@pytest.mark.parametrize(
    "query,expected_task",
    [
        ("Describe this image.", "caption"),
        ("Write a caption for this scene.", "caption"),
        ("What land cover is visible?", "vqa"),
        ("How many buildings are there?", "vqa"),
        ("Show me the water body.", "grounding"),
        ("Locate the buildings.", "grounding"),
        ("Where is the road?", "grounding"),
        ("What changed between these images?", "change"),
        ("Compare the optical and radar images.", "optical_sar"),
        ("Book me a flight to Delhi.", "unsupported"),
    ],
)
def test_fallback_routes_canonical_queries(query: str, expected_task: str) -> None:
    assert lexical_route(query).task == expected_task


def test_fallback_hard_negative_describe_vs_show() -> None:
    """One token apart, different task. This is the pair the router exists for."""
    assert lexical_route("Describe the water body.").task == "caption"
    assert lexical_route("Show me the water body.").task == "grounding"


def test_fallback_hard_negative_what_vs_where() -> None:
    what = lexical_route("What changed?")
    where = lexical_route("Where did the change happen?")

    assert what.task == where.task == "change"
    assert what.spatial_output is False
    assert where.spatial_output is True


def test_fallback_hard_negative_temporal_vs_modality() -> None:
    temporal = lexical_route("Compare these two images.")
    modality = lexical_route("Compare the optical and radar images.")

    assert temporal.task == "change"
    assert temporal.temporal is True
    assert modality.task == "optical_sar"
    assert modality.modality == "optical_sar"


def test_fallback_never_invents_capability() -> None:
    """Nonsense must land on `unsupported`, not on a real workflow."""
    for query in ("asdfghjkl", "12345", "???", "xyzzy plugh"):
        match = lexical_route(query)
        assert match.task == "unsupported", f"{query!r} invented task {match.task!r}"
        assert match.spatial_output is False
        assert match.temporal is False
        assert match.language_output is False


def test_fallback_empty_query_is_unsupported_not_a_crash() -> None:
    for query in ("", "   ", "\n\t"):
        assert lexical_route(query).task == "unsupported"


def test_fallback_produces_valid_ontology_labels() -> None:
    probes = [
        "describe this", "how many", "show me the", "what changed",
        "optical and radar", "hello world", "is there water",
    ]
    for query in probes:
        match = lexical_route(query)
        assert match.task in TASK_CLASSES
        assert match.modality in MODALITY_CLASSES
        assert 0.0 <= match.confidence <= 1.0


# ---------------------------------------------------------------------------
# 3. Corpus
# ---------------------------------------------------------------------------


def test_corpus_builds_and_validates() -> None:
    problems = build_corpus().validate()
    assert problems == [], "corpus validation failed:\n  " + "\n  ".join(problems)


def test_corpus_covers_every_task_class() -> None:
    counts = build_corpus().task_counts()
    for task in TASK_CLASSES:
        assert counts[task] > 0, f"task {task!r} has no training examples"


def test_corpus_meets_the_plan_volume_floor() -> None:
    """Plan section 10 asks for 2,000-3,000 examples. The curated + template
    corpus is deliberately smaller and denser; this pins the floor so a future
    edit cannot silently shrink it below a workable size."""
    corpus = build_corpus()
    assert len(corpus) >= 300, f"corpus shrank to {len(corpus)} examples"
    assert len(corpus.groups()) >= 30, "too few groups to split safely"


def test_corpus_dedupes_idempotently_and_without_conflicts() -> None:
    corpus = build_corpus()
    deduped, conflicts = corpus.dedupe()
    assert conflicts == [], f"same text carries two labels: {conflicts}"

    # Curated and template phrasing deliberately overlap ("Show me the water
    # body." appears in both), so dedupe may drop a few. It must be stable.
    again, conflicts2 = deduped.dedupe()
    assert conflicts2 == []
    assert len(again) == len(deduped), "dedupe is not idempotent"


def test_corpus_rejects_a_group_spanning_tasks_without_the_hn_prefix() -> None:
    bad = RouterCorpus([
        RouterExample("a", "vqa", group="g1"),
        RouterExample("b", "caption", group="g1"),
    ])
    assert any("spans multiple tasks" in p for p in bad.validate())


def test_corpus_allows_hard_negative_groups_to_span_tasks() -> None:
    """The hard-negative families exist precisely to pair different tasks."""
    ok = RouterCorpus([
        RouterExample("Describe the water body.", "caption",
                      group=f"{HARD_NEGATIVE_PREFIX}pair"),
        RouterExample("Show me the water body.", "grounding",
                      group=f"{HARD_NEGATIVE_PREFIX}pair"),
    ])
    assert [p for p in ok.validate() if "spans multiple tasks" in p] == []


def test_corpus_rejects_unsupported_examples_that_claim_capability() -> None:
    bad = RouterCorpus([
        RouterExample("do a thing", "unsupported", spatial_output=True, group="g"),
    ])
    assert any("claims a capability" in p for p in bad.validate())


def test_corpus_rejects_change_example_that_is_not_temporal() -> None:
    bad = RouterCorpus([
        RouterExample("what changed", "change", temporal=False, group="g"),
    ])
    assert any("not marked temporal" in p for p in bad.validate())


def test_corpus_rejects_grounding_example_that_is_not_spatial() -> None:
    bad = RouterCorpus([
        RouterExample("show me", "grounding", spatial_output=False, group="g"),
    ])
    assert any("not marked spatial_output" in p for p in bad.validate())


def test_corpus_rejects_optical_sar_with_wrong_modality() -> None:
    bad = RouterCorpus([
        RouterExample("compare", "optical_sar", modality="optical", group="g"),
    ])
    assert any("modality" in p for p in bad.validate())


def test_corpus_rejects_empty_input() -> None:
    assert RouterCorpus([]).validate() == ["corpus is empty"]


def test_example_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError):
        RouterExample("x", "not_a_task", group="g")
    with pytest.raises(ValueError):
        RouterExample("x", "vqa", modality="gamma_ray", group="g")


def test_example_requires_a_group_and_non_empty_text() -> None:
    with pytest.raises(ValueError):
        RouterExample("x", "vqa", group="")
    with pytest.raises(ValueError):
        RouterExample("   ", "vqa", group="g")


def test_example_label_tuple_shape() -> None:
    e = RouterExample("show me the road", "grounding", "optical",
                      spatial_output=True, group="g")
    task_idx, modality_idx, binaries = e.label_tuple()
    assert task_idx == TASK_TO_INDEX["grounding"]
    assert isinstance(modality_idx, int)
    assert len(binaries) == len(BINARY_HEADS)
    assert binaries[1] == 1.0  # spatial_output


# ---------------------------------------------------------------------------
# 4. Group-level split — the leakage guard
# ---------------------------------------------------------------------------


def test_split_never_puts_a_group_in_two_places() -> None:
    report = split_leakage_report(split_by_group(build_corpus(), seed=42))
    assert report["clean"] is True
    assert report["groups_across_splits"] == {}


def test_split_populates_all_three_partitions() -> None:
    split = split_by_group(build_corpus(), seed=42)
    assert len(split["train"]) > 0
    assert len(split["val"]) > 0
    assert len(split["test"]) > 0


def test_split_gives_every_task_class_training_examples() -> None:
    """A task head cannot learn a class it never sees. This is the check that
    stops a random group split from silently producing a broken router."""
    split = split_by_group(build_corpus(), seed=42)
    train_tasks = {e.task for e in split["train"]}
    missing = [t for t in TASK_CLASSES if t not in train_tasks]
    assert missing == [], f"no training examples for {missing}"


def test_split_places_hard_negatives_in_test() -> None:
    """Hard negatives are the cases we most want measured, so they are held out
    of training rather than scored on after being memorised."""
    split = split_by_group(build_corpus(), seed=42, hard_negatives_to_test=True)
    test_groups = {e.group for e in split["test"]}
    assert any(g.startswith(HARD_NEGATIVE_PREFIX) for g in test_groups)

    train_groups = {e.group for e in split["train"]}
    assert test_groups & train_groups == set()


def test_split_is_deterministic_for_a_seed() -> None:
    a = split_by_group(build_corpus(), seed=7)
    b = split_by_group(build_corpus(), seed=7)
    for name in ("train", "val", "test"):
        assert [e.text for e in a[name]] == [e.text for e in b[name]]


def test_split_differs_across_seeds() -> None:
    def assignment(seed: int) -> dict[str, str]:
        split = split_by_group(build_corpus(), seed=seed)
        return {e.group: name for name, ex in split.items() for e in ex}

    assert assignment(1) != assignment(999)


def test_split_output_is_in_sorted_group_order() -> None:
    """The seed chooses which group goes where; it must not control list order,
    or downstream code taking records[:N] would silently depend on the seed."""
    for seed in (1, 999):
        split = split_by_group(build_corpus(), seed=seed)
        for name in ("train", "val", "test"):
            groups = [e.group for e in split[name]]
            assert groups == sorted(groups), f"{name} (seed={seed}) is not sorted"


def test_split_rejects_bad_ratios() -> None:
    corpus = build_corpus()
    with pytest.raises(ValueError):
        split_by_group(corpus, train_ratio=0.0)
    with pytest.raises(ValueError):
        split_by_group(corpus, train_ratio=0.7, val_ratio=0.5)


def test_split_rejects_an_empty_corpus() -> None:
    with pytest.raises(ValueError):
        split_by_group(RouterCorpus([]))


def test_split_assigns_every_group_exactly_once() -> None:
    corpus = build_corpus()
    split = split_by_group(corpus, seed=5)
    seen = [e.group for ex in split.values() for e in ex]
    assert sorted(seen) == sorted(e.group for e in corpus)


def test_split_gives_every_task_class_test_examples() -> None:
    """A class with no test examples cannot be evaluated.

    This is the defect the stratified splitter exists to prevent: `vqa` had 72
    training examples and ZERO test examples, so the acceptance gate reported a
    task accuracy over a model that had never been asked about vqa at all.
    """
    split = split_by_group(build_corpus(), seed=42)
    test_tasks = {e.task for e in split["test"]}
    missing = [t for t in TASK_CLASSES if t not in test_tasks]
    assert missing == [], f"no test examples for {missing}"


def test_split_gives_every_task_class_validation_examples() -> None:
    split = split_by_group(build_corpus(), seed=42)
    val_tasks = {e.task for e in split["val"]}
    missing = [t for t in TASK_CLASSES if t not in val_tasks]
    assert missing == [], f"no validation examples for {missing}"


def test_every_task_reaches_every_split_across_seeds() -> None:
    """Stratification must be seed-robust, not lucky at seed=42."""
    for seed in (1, 7, 42, 99, 1234):
        split = split_by_group(build_corpus(), seed=seed)
        for part in ("train", "val", "test"):
            present = {e.task for e in split[part]}
            missing = [t for t in TASK_CLASSES if t not in present]
            assert missing == [], f"seed={seed} {part} has no examples for {missing}"


def test_stratification_does_not_weaken_leakage_safety() -> None:
    """Stratifying changes WHICH split a group lands in, never whether a group
    is split. Group atomicity must survive every seed."""
    for seed in (1, 7, 42, 99, 1234):
        report = split_leakage_report(split_by_group(build_corpus(), seed=seed))
        assert report["clean"] is True, f"leakage introduced at seed={seed}"
        assert report["groups_across_splits"] == {}


def test_split_ratio_is_approximately_respected() -> None:
    """Stratification rebalances, but should not wildly distort the ratios."""
    split = split_by_group(build_corpus(), seed=42)
    total = sum(len(v) for v in split.values())
    train_frac = len(split["train"]) / total
    assert 0.55 <= train_frac <= 0.85, f"train fraction drifted to {train_frac:.2f}"


def test_no_template_uses_another_tasks_characteristic_vocabulary() -> None:
    """A template that looks like task A but is labelled task B is label noise.

    Concrete case this exists for: "Which part of the image contains the X?"
    was labelled `grounding` while sharing its interrogative ("which") and verb
    ("contains") with the VQA template "Can you tell if this contains X?". The
    router learned the contradiction and misclassified every instance of it.

    This is a heuristic, not a proof — it flags the pattern for human review
    rather than claiming the corpus is clean.
    """
    import re

    # (template regex, task that legitimately owns the vocabulary)
    VQA_FRAMES = (
        r"\bwhich\b.*\bcontains\b",
        r"\bcan you tell\b",
        r"\bhow many\b",
        r"\bis there\b.*\bpresent\b",
    )

    offenders: list[str] = []
    for group, task, template, *_ in TEMPLATES:
        if task == "vqa":
            continue
        lowered = template.lower()
        for pattern in VQA_FRAMES:
            if re.search(pattern, lowered):
                offenders.append(f"{group} ({task}): {template!r}")

    assert offenders == [], (
        "grounding/caption/change templates using VQA-shaped phrasing:\n  "
        + "\n  ".join(offenders)
    )


def test_unsupported_is_not_overweighted() -> None:
    """Template count buys group trainability; subject count only buys volume.

    Unsupported grew to 30% of the corpus when its subject list was expanded to
    twenty, which forced the inverse-frequency class weight down to 0.55 and
    destabilised the neighbouring classes. Ten templates x ten subjects keeps
    the class proportionate.
    """
    corpus = build_corpus()
    counts = corpus.task_counts()
    total = sum(counts.values())
    share = counts["unsupported"] / total
    assert share <= 0.20, f"unsupported is {share:.1%} of the corpus"


def test_no_class_dominates_the_corpus() -> None:
    """No single task may exceed a third of the training signal.

    This is the general form of the unsupported-overweighting bug: expanding one
    class's vocabulary is easy and silently skews the task head.
    """
    corpus = build_corpus()
    counts = corpus.task_counts()
    total = sum(counts.values())
    for task, n in counts.items():
        share = n / total
        assert share <= 0.35, f"{task} is {share:.1%} of the corpus"


def test_leakage_report_catches_a_deliberately_leaky_split() -> None:
    """The adversarial case: one corpus, arbitrarily split by EXAMPLE.

    Restored after it was accidentally overwritten. A guard that is deleted
    without a failing test is worse than one that was never written, because the
    suite keeps reporting green.
    """
    corpus = build_corpus()
    half = len(corpus) // 2
    leaky = {"train": list(corpus.examples[:half]),
             "val": [],
             "test": list(corpus.examples[half:])}

    report = split_leakage_report(leaky)
    assert report["clean"] is False
    assert report["groups_across_splits"], "the guard did not fire"


def test_every_guard_in_the_suite_still_exists() -> None:
    """Meta-test: the controls this project relies on are still present.

    Written after `test_leakage_report_catches_a_deliberately_leaky_split` was
    silently replaced by an unrelated edit and the suite still passed. Deleting
    a control must break the build.

    Guards are checked in the module that owns them. An earlier version of this
    test listed `test_fusion_dim_mismatch_is_rejected` here, where it does not
    live — the test failed on its first run, which is the correct outcome for a
    meta-test that was making a false claim.
    """
    import importlib
    import sys

    # module name -> guard tests that module must define
    REQUIRED: dict[str, tuple[str, ...]] = {
        "tests.routing.test_router": (
            "test_leakage_report_catches_a_deliberately_leaky_split",
            "test_split_gives_every_task_class_test_examples",
            "test_split_gives_every_task_class_validation_examples",
            "test_stratification_does_not_weaken_leakage_safety",
            "test_no_template_uses_another_tasks_characteristic_vocabulary",
            "test_no_class_dominates_the_corpus",
            "test_fallback_never_invents_capability",
            "test_router_prediction_trace_has_no_reasoning",
        ),
        "tests.test_config": (
            "test_fusion_dim_mismatch_is_rejected",
            "test_torch_compile_is_forbidden_on_zerogpu",
            "test_croma_resolution_must_be_multiple_of_8",
            "test_head_feature_dim_mismatch_is_rejected",
        ),
        "tests.leakage.test_leakage": (
            "test_random_tile_split_is_detected_as_leakage",
            "test_firewall_blocks_reads_under_test_root",
            "test_hidden_access_guard_rejects_enabled_hidden_data",
        ),
        "tests.unit.test_vrsbench_loader": (
            "test_the_removed_heuristic_would_have_diverged",
            "test_a_genuinely_inverted_box_is_rejected",
            "test_unparseable_schema_raises_with_diagnostics",
        ),
    }

    missing: list[str] = []
    for module_name, guards in REQUIRED.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            module = sys.modules.get(module_name)
        if module is None:
            missing.append(f"{module_name} (module itself is missing)")
            continue
        for guard in guards:
            if not hasattr(module, guard):
                missing.append(f"{module_name}::{guard}")

    assert missing == [], (
        "guard tests this project depends on have been removed:\n  "
        + "\n  ".join(missing)
    )


# ---------------------------------------------------------------------------
# 5. Adapter
# ---------------------------------------------------------------------------


def test_adapter_forward_shapes() -> None:
    out = IntentAdapter(input_dim=384, hidden_dim=128)(torch.randn(8, 384))
    assert out.task_logits.shape == (8, NUM_TASKS)
    assert out.modality_logits.shape == (8, NUM_MODALITIES)
    assert out.temporal_logit.shape == (8,)
    assert out.spatial_logit.shape == (8,)
    assert out.language_logit.shape == (8,)


def test_adapter_rejects_wrong_input_dim() -> None:
    adapter = IntentAdapter(input_dim=384, hidden_dim=64)
    with pytest.raises(ValueError, match="input_dim"):
        adapter(torch.randn(4, 256))


def test_adapter_rejects_non_2d_input() -> None:
    adapter = IntentAdapter(input_dim=384, hidden_dim=64)
    with pytest.raises(ValueError, match="2-D"):
        adapter(torch.randn(4, 8, 384))


def test_adapter_rejects_invalid_hyperparameters() -> None:
    with pytest.raises(ValueError):
        IntentAdapter(input_dim=0, hidden_dim=64)
    with pytest.raises(ValueError):
        IntentAdapter(input_dim=384, hidden_dim=0)
    with pytest.raises(ValueError):
        IntentAdapter(input_dim=384, hidden_dim=64, dropout=1.0)


def test_adapter_parameter_count_is_small() -> None:
    """The premise of Phase 4 is that this is cheap. If it stops being cheap,
    the frozen-encoder design has been abandoned by accident."""
    assert IntentAdapter(input_dim=384, hidden_dim=128).num_parameters() < 100_000


def test_adapter_config_roundtrips() -> None:
    a = IntentAdapter(input_dim=384, hidden_dim=96, dropout=0.2)
    b = IntentAdapter.from_config_dict(a.config_dict())
    assert (b.input_dim, b.hidden_dim, b.dropout_p) == (384, 96, 0.2)


def test_binary_logit_lookup() -> None:
    out = IntentAdapter(input_dim=384, hidden_dim=32)(torch.randn(2, 384))
    assert out.binary_logit("temporal") is out.temporal_logit
    assert out.binary_logit("spatial_output") is out.spatial_logit
    assert out.binary_logit("language_output") is out.language_logit
    with pytest.raises(KeyError):
        out.binary_logit("nonsense")


def test_adapter_save_load_roundtrip(tmp_path: Path) -> None:
    adapter = IntentAdapter(input_dim=384, hidden_dim=64, dropout=0.1)
    adapter.eval()

    save_adapter(tmp_path / "adapter", adapter, {"seed": 42, "config_hash": "abc"})
    loaded, metadata = load_adapter(tmp_path / "adapter")

    assert metadata["seed"] == 42
    assert metadata["adapter_config"]["hidden_dim"] == 64
    assert loaded.num_parameters() == adapter.num_parameters()

    x = torch.randn(3, 384)
    with torch.no_grad():
        assert torch.allclose(adapter(x).task_logits, loaded(x).task_logits)


def test_load_adapter_rejects_a_missing_directory(tmp_path: Path) -> None:
    from core.errors import ModelLoadError

    with pytest.raises(ModelLoadError):
        load_adapter(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# 5b. load_adapter path handling (directory vs. weights file)
# ---------------------------------------------------------------------------


def test_load_adapter_accepts_the_weights_file_directly(tmp_path: Path) -> None:
    """Passing `<dir>/adapter.pt` must work, not append `.pt` a second time.

    `save_adapter` writes a directory, but the weight path is the obvious
    thing to try because the name reads like a weight file. Previously that
    produced `adapter.pt/adapter.pt` and a confusing "not found".
    """
    adapter = IntentAdapter(input_dim=384, hidden_dim=64, dropout=0.1)
    save_adapter(tmp_path / "adapter", adapter, {"seed": 7})

    loaded_from_dir, _ = load_adapter(tmp_path / "adapter")
    loaded_from_file, meta = load_adapter(tmp_path / "adapter" / "adapter.pt")

    assert loaded_from_file.num_parameters() == loaded_from_dir.num_parameters()
    # Metadata is a sibling of the weights, so it is still found.
    assert meta["seed"] == 7


def test_load_adapter_error_names_the_real_problem(tmp_path: Path) -> None:
    """A `.pt` path that is not a file must not report a doubled path.

    The doubled filename was the tell, but it read as a genuinely missing
    file. The message must say what was actually wrong.
    """
    from core.errors import ModelLoadError

    phony = tmp_path / "router_adapter_v001.pt"   # does not exist
    with pytest.raises(ModelLoadError) as excinfo:
        load_adapter(phony)

    detail = excinfo.value.detail
    assert "adapter.pt/adapter.pt" not in detail
    assert "weights file" in detail
    assert str(phony) in detail


def test_load_adapter_rejects_a_non_pt_file(tmp_path: Path) -> None:
    """A file that is neither the weights name nor a `.pt` is a wrong argument."""
    from core.errors import ModelLoadError

    junk = tmp_path / "notes.txt"
    junk.write_text("not an adapter")
    with pytest.raises(ModelLoadError):
        load_adapter(junk)


def test_from_config_defaults_to_the_lexical_fallback() -> None:
    """The documented default: no adapter_path means NO trained adapter.

    The default is deliberate (it keeps tests offline and fast), but a caller
    who never passes `adapter_path` gets fallback answers while believing the
    trained model is serving. `adapter_source` makes that checkable.
    """
    from core.config import load_config

    router = IntentRouter.from_config(load_config(), load_encoder=False)

    assert router.has_adapter is False
    assert router.adapter_source == "lexical_fallback"


def test_adapter_source_reports_trained_when_an_adapter_is_present(
    tmp_path: Path,
) -> None:
    adapter = IntentAdapter(input_dim=384, hidden_dim=64, dropout=0.1)
    router = IntentRouter(adapter=adapter, encoder=None)
    assert router.adapter_source == "trained"


# ---------------------------------------------------------------------------
# 6. Training (stub encoder — no model download)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """One training run shared by the training tests. Small and fast."""
    out = tmp_path_factory.mktemp("router_train")
    return train_router(
        encoder=None,          # stub embeddings: deterministic, offline
        epochs=25,
        batch_size=32,
        hidden_dim=64,
        seed=42,
        device="cpu",
        artifact_dir=out / "adapter",
        cache_dir=out / "cache",
        config_hash="test",
        verbose=False,
    )


def test_training_produces_an_artifact(trained) -> None:
    assert trained.artifact_dir.exists()
    assert (trained.artifact_dir / "adapter.pt").exists()
    assert (trained.artifact_dir / "metadata.json").exists()


def test_training_metadata_records_provenance(trained) -> None:
    meta = trained.metadata
    for key in ("artifact", "created_at", "seed", "encoder", "adapter",
                "corpus", "split", "hyperparameters", "metrics", "history"):
        assert key in meta, f"metadata is missing {key!r}"
    assert meta["encoder_type"] == "stub"
    assert meta["seed"] == 42
    assert meta["config_hash"] == "test"


def test_training_metadata_is_json_serialisable(trained) -> None:
    json.dumps(trained.metadata)  # must not raise


def test_training_split_report_is_clean(trained) -> None:
    assert trained.split_report["clean"] is True


def test_training_history_has_one_entry_per_epoch(trained) -> None:
    assert len(trained.history) == 25
    assert all("loss" in row for row in trained.history)


def test_training_loss_decreases(trained) -> None:
    first = trained.history[0]["loss"]
    last = trained.history[-1]["loss"]
    assert last < first, f"loss went from {first:.4f} to {last:.4f}"


def test_stub_training_learns_the_training_set(trained) -> None:
    """With a stub encoder we assert only that the optimiser works — train
    accuracy well above the 1/6 chance floor. Generalisation is NOT asserted,
    because a bag-of-hashes has no semantics."""
    assert trained.train_metrics.task_accuracy > 0.5


def test_training_metrics_cover_all_splits(trained) -> None:
    for m in (trained.train_metrics, trained.val_metrics, trained.test_metrics):
        assert m.n > 0
        assert 0.0 <= m.task_accuracy <= 1.0
        assert set(m.binary_accuracy) == set(BINARY_HEADS)
        assert set(m.task_support) == set(TASK_CLASSES)

        # A recall may only be reported for a class that actually had examples.
        assert set(m.per_task_recall) <= set(TASK_CLASSES)
        for task, _ in m.per_task_recall.items():
            assert m.task_support[task] > 0, (
                f"{task} recall reported with zero support — a vacuous zero"
            )


def test_training_summary_is_renderable(trained) -> None:
    text = trained.summary()
    assert "ROUTER TRAINING COMPLETE" in text
    assert "GATE 2" in text


def test_evaluate_split_on_empty_input() -> None:
    adapter = IntentAdapter(input_dim=384, hidden_dim=32)
    metrics = evaluate_split(
        adapter, [], np.zeros((0, 384), dtype=np.float32), "empty", "cpu"
    )
    assert metrics.n == 0
    assert metrics.task_accuracy == 0.0


# ---------------------------------------------------------------------------
# 7. IntentRouter
# ---------------------------------------------------------------------------


def test_router_without_adapter_falls_back() -> None:
    router = IntentRouter(adapter=None, encoder=None, confidence_threshold=0.7)
    assert router.has_adapter is False

    pred = router.route("Show me the water body.")
    assert pred.used_fallback is True
    assert pred.intent.task is Task.GROUNDING
    assert pred.intent.spatial_output is True
    assert pred.intent.source == "lexical_fallback"


def test_router_returns_a_validated_intent() -> None:
    router = IntentRouter(confidence_threshold=0.5)
    for query in ("Describe this image.", "What changed?",
                  "Compare optical and radar."):
        pred = router.route(query)
        assert isinstance(pred.intent, Intent)
        assert pred.intent.task.value in TASK_CLASSES
        assert 0.0 <= pred.intent.confidence <= 1.0


def test_router_rejects_empty_query() -> None:
    from core.errors import RoutingError

    router = IntentRouter(confidence_threshold=0.5)
    with pytest.raises(RoutingError):
        router.route("")
    with pytest.raises(RoutingError):
        router.route("   ")


def test_router_rejects_out_of_range_threshold() -> None:
    from core.errors import RoutingError

    with pytest.raises(RoutingError):
        IntentRouter(confidence_threshold=1.5)
    with pytest.raises(RoutingError):
        IntentRouter(confidence_threshold=-0.1)


def test_router_prediction_trace_has_no_reasoning() -> None:
    """Observable facts only. No chain-of-thought may be representable."""
    trace = IntentRouter(confidence_threshold=0.5).route("What changed?").to_trace()
    forbidden = {"thoughts", "reasoning", "chain_of_thought", "scratchpad", "cot"}
    assert not (set(trace) & forbidden)
    assert "task" in trace and "confidence" in trace


def test_router_batch_returns_one_prediction_per_query() -> None:
    queries = ["Describe this.", "Show me the river.", "What changed?"]
    preds = IntentRouter(confidence_threshold=0.5).route_batch(queries)
    assert [p.intent.task for p in preds] == [
        Task.CAPTION, Task.GROUNDING, Task.CHANGE
    ]


def test_router_confidence_gate_flags_low_confidence() -> None:
    """A nonsense query must be flagged, not presented as a confident answer."""
    pred = IntentRouter(confidence_threshold=0.9).route("zzzz qqqq wwww")
    assert pred.intent.task is Task.UNSUPPORTED
    assert pred.above_threshold is False


def test_router_learned_path_is_taken_when_confident(tmp_path: Path) -> None:
    """Wire a freshly-trained stub adapter in and confirm the learned path is
    actually exercised, not silently skipped."""
    result = train_router(
        encoder=None, epochs=20, hidden_dim=64, seed=3,
        device="cpu", artifact_dir=tmp_path / "a", verbose=False,
    )
    adapter, _ = load_adapter(result.artifact_dir)

    class _StubEncoder:
        """Same embedding the training run used, so the adapter sees its own
        feature distribution."""

        model_name = "stub"
        revision = "none"
        max_length = 0
        embedding_dim = VERIFIED_EMBEDDING_DIM
        num_parameters = 0

        def encode(self, texts, batch_size=64, normalize=True):
            from router.train import _stub_embeddings

            corpus = RouterCorpus([RouterExample(t, "vqa", group="g") for t in texts])
            return _stub_embeddings(corpus)

        def encode_one(self, text, normalize=True):
            return self.encode([text])[0]

    router = IntentRouter(adapter=adapter, encoder=_StubEncoder(),
                          confidence_threshold=0.0)
    pred = router.route("Describe this image.")
    assert pred.used_fallback is False
    assert pred.intent.source == "learned"


# ---------------------------------------------------------------------------
# 8. Encoder contract guards (no model download)
# ---------------------------------------------------------------------------


def test_verified_constants_match_the_phase4_probe() -> None:
    assert VERIFIED_EMBEDDING_DIM == 384
    assert VERIFIED_TOKENIZER_MAX_LENGTH == 256


def test_encoder_rejects_max_length_above_the_tokenizer_ceiling() -> None:
    """Truncating above the ceiling is a silent no-op, so it must be an error.
    This is finding F4-1 encoded as a control rather than a comment."""
    from core.errors import ModelLoadError
    from router.encoder import FrozenEncoder

    with pytest.raises(ModelLoadError, match="ceiling"):
        FrozenEncoder("x", "y", max_length=512)


def test_encoder_rejects_zero_max_length() -> None:
    from core.errors import ModelLoadError
    from router.encoder import FrozenEncoder

    with pytest.raises(ModelLoadError):
        FrozenEncoder("x", "y", max_length=0)


def test_config_router_max_length_is_inside_the_ceiling() -> None:
    from core.config import load_config

    configured = load_config().get("router.max_length")
    assert 0 < configured <= VERIFIED_TOKENIZER_MAX_LENGTH


def test_config_embedding_dim_matches_verified() -> None:
    from core.config import load_config

    assert load_config().get("router.embedding_dim") == VERIFIED_EMBEDDING_DIM


def test_config_revision_is_pinned_not_a_branch() -> None:
    """Reproducibility requires a commit sha, not a moving branch name."""
    from core.config import load_config

    revision = load_config().get("router.revision")
    assert revision not in (None, "", "main", "master")


def test_config_exposes_router_training_keys() -> None:
    from core.config import load_config

    cfg = load_config()
    for key in ("epochs", "batch_size", "learning_rate", "val_ratio",
                "task_loss_weight", "binary_loss_weight",
                "hard_negatives_to_test"):
        assert cfg.get(f"router.training.{key}") is not None, \
            f"missing router.training.{key}"