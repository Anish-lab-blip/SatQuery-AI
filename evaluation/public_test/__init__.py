"""Immutable public-test corpus (plan section 37, "Test Set Firewall").

The mechanism lives in `corpus.py`. This module re-exports it and adds the one
thing a caller needs before anything else: the current state of the corpus.

WHAT IS TRUE RIGHT NOW
----------------------
`evaluation/public_test/` holds no corpus material -- only a `.gitkeep`. So
`public_test_state()["available"]` is False, and any attempt to open the corpus
for evaluation raises with that reason attached.

That is the correct outcome, not a defect. An empty corpus must not present as a
passing test set, and the previous behaviour of an empty directory was ambiguous
in exactly that way.

    from evaluation.public_test import public_test_state, open_public_test, evaluation_mode
    public_test_state()["available"]          # -> False, with a reason
    corpus = open_public_test(mode=evaluation_mode("benchmark run"))  # -> raises
"""

from __future__ import annotations

from typing import Any

from evaluation.leakage import PUBLIC_TEST_DIRNAME
from evaluation.public_test.corpus import (
    PUBLIC_TEST_ROOT,
    PUBLIC_TEST_SUFFIXES,
    SEAL_FILENAME,
    CorpusEntry,
    CorpusSeal,
    EvaluationMode,
    PublicTestAnswerCacheError,
    PublicTestCorpus,
    PublicTestError,
    PublicTestModeError,
    PublicTestSealedError,
    PublicTestSealError,
    evaluation_mode,
    forbid_answer_cache,
    open_public_test,
)

__all__ = [
    "PUBLIC_TEST_ROOT",
    "PUBLIC_TEST_DIRNAME",
    "PUBLIC_TEST_SUFFIXES",
    "SEAL_FILENAME",
    "EvaluationMode",
    "evaluation_mode",
    "CorpusEntry",
    "CorpusSeal",
    "PublicTestCorpus",
    "PublicTestError",
    "PublicTestSealedError",
    "PublicTestSealError",
    "PublicTestModeError",
    "PublicTestAnswerCacheError",
    "forbid_answer_cache",
    "open_public_test",
    "public_test_state",
]


def public_test_state(root: str | Any = PUBLIC_TEST_ROOT) -> dict[str, Any]:
    """The corpus's current state, plus the answer-cache check, without raising.

    This is the safe call: it reports rather than enforces, so a report can state
    the corpus boundary even when the corpus is empty. Enforcement happens in
    `open_public_test`, where failing loudly is the correct behaviour.
    """
    corpus = PublicTestCorpus(root=root)
    state = corpus.describe()

    try:
        state["answer_cache"] = forbid_answer_cache(corpus.root)
    except PublicTestAnswerCacheError as exc:
        # A cache is a VIOLATION, and must be visible in a report rather than
        # converted into an exception the caller might swallow.
        state["answer_cache"] = {
            "clean": False,
            "violation": exc.user_message,
            "found": (exc.context or {}).get("found", []),
        }

    state["seal_on_disk"] = (corpus.root / SEAL_FILENAME).exists()
    state["note"] = (
        "available=False means the corpus holds no material. It does NOT mean the "
        "test set passed: there is no test set. No answer cache exists by design; "
        "'no cache' is a property of this module, not a claim about the corpus."
    )
    return state
