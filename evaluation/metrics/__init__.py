"""SatQuery AI evaluation metrics.

Re-exports the Tier-1 text metrics from `evaluation.metrics.vqa` — the plan's
§2.2 metrics for VQA ("exact match / normalized exact match / F1 where
appropriate", `:258-266`) and Change-VQA ("answer accuracy", `:308-312`).

`change.py` and `grounding.py` are NOT re-exported here, for a concrete reason:
both publish the public names `score_one` and `score_dataset`, so a flat
re-export would silently shadow one module with the other. They also pull NumPy,
whereas `vqa.py` is pure stdlib, so re-exporting them would make this package
heavier for no gain. Import them as submodules, where the qualifier is explicit:

    from evaluation.metrics import change, grounding
    from evaluation.metrics.change import score_dataset as change_scores
    from evaluation.metrics.grounding import score_dataset as grounding_scores

`caption` (R-16, added 2026-09-23) is also imported as a SUBMODULE, not
re-exported flat -- even though it has neither of the two problems above (its
public names collide with nothing here, and it imports no third-party package at
module import time, so it is importable in an environment where none of the four
caption implementations is installed). The reason is that this package's public
surface is deliberately NARROW and is pinned as such:

    tests/unit/test_vqa_metrics.py::test_package_reexports_the_vqa_public_names
        assert set(metrics.__all__) == set(vqa.__all__)

That test asserts *exact* equality, so adding caption's names here would fail it.
The pin is treated as a deliberate design decision rather than an accident, and
is therefore not changed unilaterally -- the same rule applied to the F-21
capability guard (`docs/OWNER_DECISIONS_2026-09-23.md` D-7). Nothing is lost:
the submodule form is the documented interface and reads cleanly.

    from evaluation.metrics import caption
    caption.score_captions(predictions, references)
    caption.caption_capability()      # which of the four can actually run here
"""

from evaluation.metrics.vqa import (
    ARTICLES,
    CONTRACTIONS,
    NORMALIZATION_RULES,
    NUMBER_WORDS,
    PUNCTUATION,
    answer_accuracy,
    exact_match,
    normalized_exact_match,
    normalize_answer,
    token_f1,
)

__all__ = [
    "ARTICLES",
    "CONTRACTIONS",
    "NORMALIZATION_RULES",
    "NUMBER_WORDS",
    "PUNCTUATION",
    "answer_accuracy",
    "exact_match",
    "normalized_exact_match",
    "normalize_answer",
    "token_f1",
]
