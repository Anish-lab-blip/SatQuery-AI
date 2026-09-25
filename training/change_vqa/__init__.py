"""SatQuery AI — change-VQA / CDVQA reasoning (R-02).

WHAT THIS PACKAGE IS
--------------------
The reasoning layer that turns a pair of temporally corresponding remote-sensing
images plus a change-oriented question into an answer. It is the piece R-02 was
always about; the pre-existing CDVQA code in `training.data.cdvqa` is an
annotation-only adapter with no imagery, no tensors and no reasoning path.

    image pair (T1, T2)  ->  frozen STANet detector  ->  change feature
                         ->  class-wise change estimate
                         ->  question-conditioned 19-way answer
                         ->  answer + evidence + provenance

THE DESIGN IN ONE PARAGRAPH
---------------------------
Measured constraints (2026-09-21) pick the architecture: the answer space is
closed and has 19 members, answers are short (`yes`, `buildings`, `10_to_20`),
`label1`/`label2` ship for every scene so class-wise change is computable
supervision, and the plan budgets <= 3 h on T4x2. Those rule out a generative
decoder — 19-way classification is strictly easier and is exactly what the metric
scores — and rule out training anything large. What is left is a two-stage head
(`model.py`): a class-wise change estimator, whose outputs are CONCATENATED into
the answer head so the answer is computed *from* the change estimate rather than
alongside it. That is what lets it answer `largest_change` / `smallest_change`,
which a single pooled vector cannot express.

MODULES
-------
    vocab.py      the 19-answer space, 8 question types, 6 change classes, and
                  the question -> (type, temporal reference, class) resolver
    dataset.py    scene targets from label1/label2, typed records, the scene
                  manifest, and the split-integrity gate
    features.py   frozen STANet feature extraction, MiniLM question features,
                  and the two caches
    model.py      the two-stage head, its loss, and checkpoint I/O
    collate.py    deterministic batching that skips and COUNTS what it drops
    prompts.py    canonical question phrasings and the explanation prompt
    evaluate.py   metrics, the provenance-rich report, qualitative examples
    train.py      the time-budgeted training loop

THE DETECTOR IS FROZEN, AND THAT IS LOAD-BEARING
------------------------------------------------
`features.py` reads from a trained STANet checkpoint and never updates it. Only
the head in `model.py` is trained. That is why the 3 h budget is reachable at
all: there is no 15.8 M-parameter backbone in the optimizer and no image is
decoded during training. A cache built with an UNTRAINED detector is permitted
but is recorded as such everywhere it travels, because a head trained on an
untrained representation is not the capability the plan describes.

NOTHING HERE IS CALIBRATED
--------------------------
Confidence is raw softmax plus a top1-top2 margin, labelled `"uncalibrated"` all
the way to the result schema. The R-03 calibration contract is a separate, open
ruling. Fitting a temperature in this package would be a fabricated claim.

IMPORT-SAFETY
-------------
No module here imports torch at module scope; torch is imported inside the
functions that need it. So importing this package stays cheap and works in a
torch-free context, which is what lets the specialist layer and the tooling
import it without dragging in a 2 GB dependency.

    >>> from training.change_vqa import N_ANSWERS, ANSWER_VOCABULARY
    >>> len(ANSWER_VOCABULARY) == N_ANSWERS == 19
    True
"""

from __future__ import annotations

from training.change_vqa.collate import (
    ChangeVQABatch,
    build_batch,
    iter_batches,
    summarise_skips,
)
from training.change_vqa.dataset import (
    DATASET_ID,
    EXPECTED_MANIFEST_RECORDS,
    EXPECTED_SPLIT_SIZES,
    EXPECTED_TOTAL_QUESTIONS,
    EXPECTED_TOTAL_SCENES,
    PREPROCESSING_VERSION,
    SPLIT_MAP,
    ChangeVQAError,
    ChangeVQARecord,
    SceneTargets,
    assert_no_leakage,
    assert_split_integrity,
    build_scene_manifest,
    by_native_split,
    compute_scene_targets,
    dataset_statistics,
    decode_label_map,
    group_by_native_split,
    leakage_report,
    load_change_vqa_records,
    read_manifest,
    read_scene_targets,
    verify_split_integrity,
    write_manifest,
    write_scene_targets,
)
from training.change_vqa.evaluate import (
    METRIC_IMPLEMENTATION,
    EvaluationReport,
    answer_accuracy,
    confusion_matrix,
    estimator_errors,
    evaluate_records,
    macro_f1,
    per_type_accuracy,
    qualitative_examples,
    top_k_accuracy,
)
from training.change_vqa.features import (
    CHANGE_FEATURE_DIM,
    FEATURE_SPEC,
    TEXT_ENCODER_NAME,
    TEXT_FEATURE_DIM,
    ChangeFeatureCache,
    ChangeFeatureExtractor,
    FeatureExtractionError,
    TextFeatureCache,
    TextFeatureExtractor,
    build_change_feature_extractor,
    extract_change_features,
    feature_spec_hash,
    text_spec_hash,
)
from training.change_vqa.model import (
    ARCHITECTURE_VERSION,
    DEFAULT_LOSS_WEIGHTS,
    N_ESTIMATOR_OUTPUTS,
    ChangeVQALossBreakdown,
    ChangeVQAOutput,
    build_change_vqa_head,
    change_vqa_loss,
    load_change_vqa_head,
    predict_answers,
    save_change_vqa_head,
)
from training.change_vqa.prompts import (
    EXPLANATION_SYSTEM,
    build_explanation_prompt,
    canonical_query,
    validate_templates,
)
from training.change_vqa.vocab import (
    ANSWER_VOCABULARY,
    ANSWER_TO_INDEX,
    CHANGE_CLASS_ORDER,
    INDEX_TO_ANSWER,
    LEGAL_ANSWERS,
    N_ANSWERS,
    N_CHANGE_CLASSES,
    N_QUESTION_TYPE_SLOTS,
    N_TEMPORAL_REFERENCE_SLOTS,
    QUESTION_TYPE_ORDER,
    TEMPORAL_REFERENCE_ORDER,
    answers_outside_vocabulary,
    legal_answer_mask,
    ratio_bin_for,
    resolve_change_class,
    resolve_question_type,
    resolve_temporal_reference,
    validate_vocabulary,
)

__all__ = [
    # vocab
    "ANSWER_VOCABULARY",
    "ANSWER_TO_INDEX",
    "INDEX_TO_ANSWER",
    "N_ANSWERS",
    "CHANGE_CLASS_ORDER",
    "N_CHANGE_CLASSES",
    "QUESTION_TYPE_ORDER",
    "N_QUESTION_TYPE_SLOTS",
    "TEMPORAL_REFERENCE_ORDER",
    "N_TEMPORAL_REFERENCE_SLOTS",
    "LEGAL_ANSWERS",
    "legal_answer_mask",
    "ratio_bin_for",
    "resolve_change_class",
    "resolve_question_type",
    "resolve_temporal_reference",
    "validate_vocabulary",
    "answers_outside_vocabulary",
    # dataset
    "DATASET_ID",
    "PREPROCESSING_VERSION",
    "SPLIT_MAP",
    "ChangeVQAError",
    "ChangeVQARecord",
    "SceneTargets",
    "decode_label_map",
    "compute_scene_targets",
    "load_change_vqa_records",
    "group_by_native_split",
    "build_scene_manifest",
    "by_native_split",
    "leakage_report",
    "assert_no_leakage",
    "EXPECTED_SPLIT_SIZES",
    "EXPECTED_TOTAL_QUESTIONS",
    "EXPECTED_TOTAL_SCENES",
    "EXPECTED_MANIFEST_RECORDS",
    "verify_split_integrity",
    "assert_split_integrity",
    "dataset_statistics",
    "write_manifest",
    "read_manifest",
    "write_scene_targets",
    "read_scene_targets",
    # features
    "FEATURE_SPEC",
    "CHANGE_FEATURE_DIM",
    "TEXT_FEATURE_DIM",
    "TEXT_ENCODER_NAME",
    "FeatureExtractionError",
    "ChangeFeatureExtractor",
    "TextFeatureExtractor",
    "ChangeFeatureCache",
    "TextFeatureCache",
    "feature_spec_hash",
    "text_spec_hash",
    "build_change_feature_extractor",
    "extract_change_features",
    # model
    "ARCHITECTURE_VERSION",
    "N_ESTIMATOR_OUTPUTS",
    "DEFAULT_LOSS_WEIGHTS",
    "ChangeVQAOutput",
    "ChangeVQALossBreakdown",
    "build_change_vqa_head",
    "save_change_vqa_head",
    "load_change_vqa_head",
    "change_vqa_loss",
    "predict_answers",
    # collate
    "ChangeVQABatch",
    "build_batch",
    "iter_batches",
    "summarise_skips",
    # prompts
    "EXPLANATION_SYSTEM",
    "build_explanation_prompt",
    "canonical_query",
    "validate_templates",
    # evaluate
    "METRIC_IMPLEMENTATION",
    "EvaluationReport",
    "evaluate_records",
    "answer_accuracy",
    "top_k_accuracy",
    "macro_f1",
    "confusion_matrix",
    "per_type_accuracy",
    "estimator_errors",
    "qualitative_examples",
]
