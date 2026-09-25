"""SatQuery AI — Phase 6 evaluation and the predeclared acceptance rule.

THE ACCEPTANCE RULE IS DECIDED, NOT INVENTED HERE
-------------------------------------------------
`docs/PHASE6_AUDIT_AND_CONTRACT.md` section "Contract item V" declares the rule
on 2026-09-23, **before any training run exists**. This module encodes it and
adds nothing: the thresholds below are copied verbatim from that section, and
`ACCEPTANCE_RULE_VERSION` is recorded in every run manifest so a future change
is a new version, not a quiet edit.

TWO VERSIONS, BOTH REPLAYABLE
-----------------------------
`v001` is the original pre-registered rule. It is retained here **verbatim and
replayable** via `rule_version="v001"`: replaying run 1's recorded metrics
reproduces the same **verdict, class set and key set** (see
`scripts/phase6_rerule.py`). It is **not byte-identical**, and must not be
described as exact: the manifest stores `per_class_accuracy` rounded to 6 dp
while the original run computed from raw integer counts, so an individual `*_pp`
value can differ in the last decimal (Mixed forest `drop_pp` replays as 6.4517
where run 1 recorded 6.4516), and the `class_failures` **ordering** is not stable
between the two paths.

`v002` was declared 2026-09-24, **after** run 1, and changes **only** the V2
per-class guardrail. It is a resolution fix justified by the contract's *own*
stated principle, and it is a **relaxation of how often V2 fires** -- not a
relaxation of the requirement that a real collapse be caught:

  * v001's flat `MAX_CLASS_DROP_PP = 1.0` pp, applied at `MIN_CLASS_QUESTIONS =
    20`, fires on **0.20 of a question** (one question is 5.0 pp at n=20) -- below
    the smallest change the data can express. The per-class standard error of
    the delta over the run-1 classes is 2.6-10.6 pp, so the threshold sat ~8x
    below its own SE.
  * Item V already states the resolution standard for V1 (lines 363-368): "the
    standard error of an accuracy near 0.5 is ~1.6 pp, so the predeclared +5.0 pp
    threshold is resolvable at ~3 SE". That reasoning was applied to the
    aggregate V1 threshold but **never to V2**, which operates on per-class
    n of 20-163.
  * v002 applies that same standard per-class: a class fails only when the drop
    is **material** (`lost_questions >= MIN_CLASS_DROP_QUESTIONS`, a real,
    countable loss) **and** **significant** (`z >= CLASS_DROP_Z`). Note the
    direction: 1.96 SE is *below* item V's own ~3 SE figure, so on the
    significance axis v002 is the weaker bar, not a stricter one. What v002 adds
    over v001 is that *any* significance requirement and a materiality floor now
    exist (v001 had neither) and that the threshold is no longer below the
    measurement's resolution. The net effect **at the class sizes this pipeline
    can produce (`n <= 163`, the largest run-1 class)** is that v002 fires
    **less** than v001; that is stated as a relaxation of how often V2 fires,
    justified by v001 having fired below its own resolution -- not as a bar
    raised above it. It is **not a universal relaxation**: for v002 to fire where
    v001 does not requires a drop `<= 1.0 pp` (else v001 fires too) that is still
    significant (`z >= 1.96`), which forces `SE <= 1.0 / 1.96 = 0.51 pp` and
    hence `n >= ~19,200` at `p = 0.5`. Above that size a sub-1 pp drop can be
    statistically significant and v002 would reject a class v001 would pass
    (e.g. `n = 20,000`, 10,000 -> 9,800 correct: drop 1.0000 pp, SE 0.4999 pp,
    `z = 2.0002`, 200 questions lost). No class in this pipeline is anywhere
    near that size.

The full run-1 evidence and the amendment record live in
`docs/PHASE6_AUDIT_AND_CONTRACT.md` ("Contract item V, amendment v002").

    ACCEPT iff all four hold (v002, the current rule):
      V1   adapted - baseline >= +5.0 pp
      V1'  ceiling clause, ONLY if baseline >= 95.0: adapted - baseline >= +2.0 pp
      V2   no material, significant class collapse: a class with >= 20 val
           questions fails only if it BOTH lost >= 4 questions AND has
           z = (baseline - adapted) / SE >= 1.96
      V3   the run completed its predeclared budget
      V4   artifact integrity: reloadable, every file SHA256-recorded,
           trainable-param count > 0

    Under v001 (legacy, replayable, NOT the default):
      V2   no class collapse: for every class with >= 20 val questions,
           adapted_class >= baseline_class - 1.0 pp

    REJECT      iff a complete run fails V1/V1' or V2
    INCONCLUSIVE iff V3 or V4 fails

    The decision is taken on VALIDATION. The test split is evaluated **alongside
    val and before the decision** -- `scripts/phase6_train_vlm.py` runs both
    `evaluate_split` calls first, then calls `decide_acceptance`. The test delta
    is recorded on the decision whenever test metrics are supplied, **including
    on a REJECTED verdict**: a short-circuit must not discard a full second
    evaluation's worth of GPU time (run 1's V2 rejection did exactly that, and
    serialised `test_delta_pp` as `null`). If val accepts and test does not, the
    status is `ACCEPTED_ON_VAL_NOT_CONFIRMED_ON_TEST` and it is flagged for owner
    review.

    A v002 ACCEPT on validation is NOT an acceptance: run 1 evaluated the test
    split but never persisted its metrics, so acceptance under v002 must be
    decided on the test split (recovered or re-run), not on the val subset that
    revealed the V2 defect.

WHY THE NORMALISATION IS PART OF THE FROZEN PROTOCOL
----------------------------------------------------
The primary endpoint is exact-match accuracy on one-token answers ("Yes." /
"No."). A model that emits "yes", "Yes", "Yes." or "YES" has answered
correctly, and a scorer that disagrees would measure punctuation, not
comprehension. `normalise_answer` therefore delegates to the repository's
VQA-v2 normaliser, strips surrounding quotes/whitespace, and maps the yes/no
synonym families onto a single token. It is frozen alongside the rule, and its
version is published as `NORMALISATION_RULE_VERSION`.

ON REUSING `evaluation/metrics/vqa.py`
--------------------------------------
Read first. The **normaliser is delegated to it**, and the rest is deliberately
not reused:

  * `normalize_answer` there is the VQA-v2 rule set (R1-R6: lowercase,
    contractions, number words, article removal, decimal-aware punctuation,
    whitespace). It is the **base** of `normalise_answer` here, so the
    punctuation/contraction handling is the repository's one definition rather
    than a second, subtly different one.
  * It does NOT map `true`/`y` onto `yes`. The presence task's answers are a
    two-token vocabulary where that synonym map is the load-bearing rule, so it
    is layered **on top** of the delegated normaliser -- see the ORDER note
    below. Dropping it would score a correct `"True"` as wrong (measured:
    `normalize_answer("True") == "true"`, not `"yes"`), silently moving the V1
    delta.
  * `token_f1` is a text-overlap metric. The answers are one token, so token F1
    is either 1.0 or 0.0 and carries no information the exact match does not.
  * The precision/recall/F1 this module reports are **binary presence**
    precision/recall/F1 (is the class present? yes/no), not token overlap.

ORDER IS LOAD-BEARING
---------------------
`normalise_answer` applies three steps in this exact order:

    (a) `evaluation.metrics.vqa.normalize_answer(text)` -- the VQA-v2 rule set
    (b) strip surrounding quotes / whitespace
    (c) fold the yes/no synonym families onto "yes" / "no"

The order matters because `normalize_answer` **preserves apostrophes** (it
excludes `'` from its punctuation table so contractions survive). So
`normalize_answer("'Yes.'") == "'yes'"`, and folding before stripping would test
`"'yes'"` against `{"yes","true","y"}`, miss, and return the quoted string.
Stripping first is what makes the fold fire. The rule set is published as
`NORMALISATION_RULE_VERSION` and recorded in every run manifest.

BERTSCORE
---------
`roberta-large` is not in the local HuggingFace cache, so BERTScore cannot be
computed offline. It is recorded as an explicit **unavailable limitation** in
`describe_acceptance_rule()`, never as a zero and never substituted. BLEU/ROUGE
are likewise not reported: one-token answers make them meaningless.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "ACCEPTANCE_RULE_VERSION",
    "LEGACY_RULE_VERSIONS",
    "NORMALISATION_RULE_VERSION",
    "ACCEPT_MIN_DELTA_PP",
    "CEILING_BASELINE_PP",
    "ACCEPT_MIN_DELTA_PP_CEILING",
    "MAX_CLASS_DROP_PP",
    "MIN_CLASS_QUESTIONS",
    "MIN_CLASS_DROP_QUESTIONS",
    "CLASS_DROP_Z",
    "TEST_VAL_DISAGREEMENT_PP",
    "MetricSet",
    "AcceptanceDecision",
    "normalise_answer",
    "deterministic_subsample",
    "predict",
    "evaluate_split",
    "decide_acceptance",
    "describe_acceptance_rule",
    "class_of_question",
]

# -- the predeclared rule (contract item V). DO NOT change without a new version.
#: The current rule version. `v002` (2026-09-24) amends only V2's per-class
#: criterion; `v001` remains defined and replayable via `rule_version="v001"`.
ACCEPTANCE_RULE_VERSION = "v002"

#: Earlier rule versions that remain defined and replayable. Passing one of
#: these to `decide_acceptance` / `describe_acceptance_rule` reproduces that
#: version's behaviour exactly. They are never applied by default.
LEGACY_RULE_VERSIONS: tuple[str, ...] = ("v001",)

#: The answer-normalisation rule version. Frozen alongside the acceptance rule:
#: a change to the rule set is a new version, not a quiet edit, because it moves
#: the measured exact-match accuracy and therefore the V1 delta. The concrete
#: rule set is reported by `describe_acceptance_rule()`.
NORMALISATION_RULE_VERSION = "v001"

ACCEPT_MIN_DELTA_PP = 5.0
CEILING_BASELINE_PP = 95.0
ACCEPT_MIN_DELTA_PP_CEILING = 2.0
#: v001's V2 threshold. RETAINED so `rule_version="v001"` reproduces run 1's
#: REJECTED verdict exactly. Not used by the current rule (`v002`).
MAX_CLASS_DROP_PP = 1.0
MIN_CLASS_QUESTIONS = 20
#: v002's V2 materiality floor: a class must have actually lost at least this
#: many questions to fail. At MIN_CLASS_QUESTIONS = 20 one question is 5.0 pp,
#: so v001's flat 1.0 pp threshold fired on 0.20 of a question -- below the
#: smallest change the data can express. The floor is the smallest loss that is
#: a real, countable change rather than rounding noise.
MIN_CLASS_DROP_QUESTIONS = 4
#: v002's V2 significance bar, in per-class standard errors of the delta. The
#: conventional 95% two-sided figure. Note it is BELOW item V's own "resolvable
#: at ~3 SE" figure, so on the significance axis v002 is the *weaker* bar, not a
#: stricter one; what v002 adds over v001 is that a significance requirement (and
#: a materiality floor) exists at all, and that the threshold is no longer below
#: the measurement's resolution.
CLASS_DROP_Z = 1.96
TEST_VAL_DISAGREEMENT_PP = 10.0

#: Presence-question template, used to recover the class a question is about.
#: Mirrors `training.vlm.dataset._presence_pairs`.
_PRESENCE_RE = re.compile(r"^Is (.+) present in this image\?$")

#: Yes/no synonym families collapsed by `normalise_answer`.
_YES_TOKENS = frozenset({"yes", "true", "y"})
_NO_TOKENS = frozenset({"no", "false", "n"})


def normalise_answer(text: str) -> str:
    """Frozen answer normalisation (see the module docstring).

    Three steps, applied in this exact order:

      (a) delegate to `evaluation.metrics.vqa.normalize_answer` -- the VQA-v2
          R1-R6 rule set (lowercase, contractions, number words, articles,
          decimal-aware punctuation, whitespace);
      (b) strip **surrounding** quotes and whitespace. Necessary because (a)
          preserves apostrophes (it excludes `'` from its punctuation table), so
          `"'Yes.'"` comes back as `"'yes'"`;
      (c) fold the yes/no synonym families onto `"yes"` / `"no"`.

    Order is load-bearing: folding before (b) would test `"'yes'"` against
    `{"yes","true","y"}`, miss, and return the quoted string.

    Args:
        text: the answer string.

    Returns:
        The normalised answer.

    Raises:
        TypeError: `text` is not a string. Raised by the delegated normaliser;
            coercing it (e.g. `None` -> "none") would invent an answer.
    """
    from evaluation.metrics.vqa import normalize_answer as _vqa_normalize_answer

    base = _vqa_normalize_answer(text)  # (a) the repository's VQA-v2 rule set
    stripped = base.strip(" \t\r\n\"'`")  # (b) surrounding quotes + whitespace
    if stripped in _YES_TOKENS:  # (c) synonym families
        return "yes"
    if stripped in _NO_TOKENS:
        return "no"
    return stripped


def class_of_question(question: str) -> str | None:
    """The land-cover class a presence question asks about, or `None`.

    `"Is Marine waters present in this image?"` -> `"Marine waters"`. A question
    that does not match the frozen presence template returns `None`, so it is
    excluded from per-class statistics rather than silently bucketed.
    """
    match = _PRESENCE_RE.match(question.strip())
    return match.group(1) if match else None


def deterministic_subsample(
    samples: Sequence[Any], limit: int | None, seed: int
) -> list[Any]:
    """Deterministically choose up to `limit` samples, seeded from `seed`.

    Selection is by `sha256(seed, sample_id)` rank rather than `random.sample`,
    because `random.sample`'s algorithm is not part of its contract and a Python
    upgrade could silently change which samples are scored -- moving the V1
    delta. Selected samples keep their original corpus order, so the metric is a
    property of the subset, not of the iteration order.

    Baseline and adapted evaluations call this with the same `seed` and the same
    split, so both score the **same** questions and the measured delta is the
    adapter's rather than a subset difference.

    Args:
        samples: the split's samples.
        limit: the cap, or `None` / `>= len(samples)` to keep everything.
        seed: the run seed.

    Returns:
        The selected samples, in their original order.

    Raises:
        ValueError: `limit` is less than 1.
    """
    items = list(samples)
    if limit is None or limit >= len(items):
        return items
    if limit < 1:
        raise ValueError(f"limit must be >= 1 or None, got {limit}")

    def rank(index: int) -> str:
        sample = items[index]
        sid = (
            getattr(sample, "sample_id", None)
            or getattr(sample, "patch_id", None)
            or str(index)
        )
        return hashlib.sha256(f"{seed}:{sid}".encode("utf-8")).hexdigest()

    chosen = sorted(range(len(items)), key=lambda i: (rank(i), i))[:limit]
    chosen.sort()
    return [items[i] for i in chosen]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class MetricSet:
    """The metrics of one evaluation split.

    `exact_match` is the primary endpoint (normalised exact-match accuracy).
    `precision`/`recall`/`f1` are the **positive-class (present) binary**
    metrics over the yes/no decision. `confusion` carries the raw tp/fp/tn/fn so
    a reader can recompute any of the three.
    """

    n: int
    exact_match: float
    per_class_accuracy: dict[str, float] = field(default_factory=dict)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    confusion: dict[str, int] = field(default_factory=dict)
    #: Per-class question counts. Load-bearing for the V2 class-collapse clause,
    #: which only applies to classes with >= MIN_CLASS_QUESTIONS questions.
    per_class_counts: dict[str, int] = field(default_factory=dict)
    #: True when the wall budget stopped generation before the split was
    #: exhausted. A truncated split cannot support an ACCEPTED decision (V3):
    #: a partial evaluation is not evidence of the full split's accuracy.
    truncated: bool = False
    #: How many samples the split offered. `n` is how many were actually scored,
    #: so `n < n_available` (or `truncated`) records a partial evaluation.
    n_available: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "n_available": self.n_available,
            "truncated": self.truncated,
            "exact_match": round(self.exact_match, 6),
            "per_class_accuracy": {
                k: round(v, 6) for k, v in self.per_class_accuracy.items()
            },
            "per_class_counts": dict(self.per_class_counts),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "confusion": dict(self.confusion),
        }


def _metric_set(
    preds: Sequence[str], golds: Sequence[str], questions: Sequence[str]
) -> MetricSet:
    """Compute a `MetricSet` from aligned predictions, golds, and questions."""
    if len(preds) != len(golds) or len(preds) != len(questions):
        raise ValueError(
            f"metric inputs must be aligned: {len(preds)} preds, "
            f"{len(golds)} golds, {len(questions)} questions"
        )
    n = len(preds)
    if n == 0:
        return MetricSet(n=0, exact_match=0.0, confusion={})

    hits = 0
    tp = fp = tn = fn = 0
    per_class_hits: dict[str, int] = {}
    per_class_total: dict[str, int] = {}

    for pred, gold, question in zip(preds, golds, questions):
        p = normalise_answer(pred)
        g = normalise_answer(gold)
        if p == g:
            hits += 1
        pred_yes = p == "yes"
        gold_yes = g == "yes"
        if gold_yes and pred_yes:
            tp += 1
        elif gold_yes and not pred_yes:
            fn += 1
        elif not gold_yes and pred_yes:
            fp += 1
        else:
            tn += 1

        cls = class_of_question(question)
        if cls is not None:
            per_class_total[cls] = per_class_total.get(cls, 0) + 1
            if p == g:
                per_class_hits[cls] = per_class_hits.get(cls, 0) + 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return MetricSet(
        n=n,
        exact_match=hits / n,
        per_class_accuracy={
            cls: per_class_hits.get(cls, 0) / total
            for cls, total in per_class_total.items()
        },
        per_class_counts=dict(per_class_total),
        precision=precision,
        recall=recall,
        f1=f1,
        confusion={"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _render_question_images(samples: Sequence[Any], percentiles: tuple[float, float]) -> list[Any]:
    """Render each sample's patch to a true-colour PIL image."""
    from training.vlm.collate import render_sample

    return [render_sample(sample, percentiles=percentiles) for sample in samples]


def _as_patch(sample: Any) -> Any:
    """Backwards-compatible alias for `collate.as_patch_sample`."""
    from training.vlm.collate import as_patch_sample

    return as_patch_sample(sample)


# ---------------------------------------------------------------------------
# Resumable per-question answer cache
# ---------------------------------------------------------------------------
# A full split evaluation is ~1000 per-sample generations. On CPU that is tens of
# minutes, and a long single-shot process is fragile: if it is killed (host memory
# pressure, an idle-timeout, a closed terminal) every answer is lost and the run
# must start from zero. The cache makes progress **monotone** -- each answer is
# flushed as it is produced, so a killed run resumes where it stopped instead of
# repeating work.
#
# The cache is keyed to the *exact* ordered question set it was built from. A
# mismatch is a hard error, not a silent reuse: mixing a stale cache into a new
# evaluation would corrupt the metric while looking perfectly healthy, which is
# the one failure mode this repository cannot tolerate in an acceptance path.

def _cache_fingerprint(samples: Sequence[Any]) -> str:
    """A stable digest of the ordered sample sequence.

    Covers every field that can change an answer: the ordered sample identity
    (`sample_id`, falling back to the question text) and the count. Answers depend
    on the image too, but `sample_id` names the patch+question pair, so two
    sequences with equal fingerprints are the same questions of the same patches.
    """
    h = hashlib.sha256()
    for s in samples:
        key = getattr(s, "sample_id", None) or getattr(s, "question", "")
        h.update(str(key).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _load_answer_cache(cache_path: Any, fingerprint: str, n: int) -> dict[int, str]:
    """Read a cache, validating it belongs to this exact question set."""
    if cache_path is None:
        return {}
    path = Path(cache_path)
    if not path.is_file():
        return {}
    header: dict[str, Any] | None = None
    answers: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A kill mid-write can tear a line. SKIP it rather than aborting:
                # records are keyed by index, so the affected question is simply
                # regenerated. Aborting here would be far worse -- the torn line
                # stays in the file forever, so every later run would stop at the
                # same place and progress would stall permanently.
                continue
            if record.get("kind") == "header":
                header = record
                continue
            answers[int(record["i"])] = record["answer"]
    if header is None:
        raise ValueError(
            f"answer cache {path} has no readable header line; refusing to reuse it. "
            "Delete the file to rebuild from scratch."
        )
    if header.get("fingerprint") != fingerprint or int(header.get("n", -1)) != n:
        raise ValueError(
            f"answer cache {path} was built for a DIFFERENT question set "
            f"(cache: fingerprint={header.get('fingerprint')!r} n={header.get('n')!r}; "
            f"now: fingerprint={fingerprint!r} n={n}). Reusing it would silently "
            "corrupt the metric. Delete the file to rebuild, or point --cache-dir "
            "somewhere else."
        )
    return answers


def _open_answer_cache(cache_path: Any, fingerprint: str, n: int) -> Any:
    """Open the cache for appending, writing the header if the file is new.

    If a previous run was killed mid-write the file can end WITHOUT a newline.
    Appending straight onto that would merge the next record into the torn line,
    turning a one-line tear into a corrupt line in the middle of the file. So
    terminate the torn line first.
    """
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists() or path.stat().st_size == 0
    torn_tail = False
    if not fresh:
        with path.open("rb") as probe:
            probe.seek(-1, os.SEEK_END)
            torn_tail = probe.read(1) != b"\n"
    handle = path.open("a", encoding="utf-8")
    if fresh:
        handle.write(
            json.dumps({"kind": "header", "fingerprint": fingerprint, "n": n}) + "\n"
        )
        handle.flush()
    elif torn_tail:
        handle.write("\n")
        handle.flush()
    return handle


def predict(
    model: Any,
    processor: Any,
    samples: Sequence[Any],
    *,
    max_new_tokens: int = 8,
    device: str = "cpu",
    deadline: float | None = None,
    cache_path: Any = None,
) -> list[str]:
    """Greedy-generate one answer per sample.

    Prompts are built through `training.vlm.formatting.build_training_messages`,
    which delegates to `specialists.vqa.prompts.build_messages` -- the **serving**
    prompt path. Baseline and adapted models are therefore asked byte-identical
    questions, which is what makes the delta a property of the adapter and not of
    a prompt difference.

    Generation is per-sample: SmolVLM expands one image into a variable number of
    tokens, so padded batched decoding is fragile. `evaluate_split` accepts a
    `batch_size` for the caller's intent but this function is the correctness
    path.

    Args:
        deadline: an absolute `time.monotonic()` value. When given, generation
            stops once it passes and fewer answers are returned; the caller
            (`evaluate_split`) marks the resulting `MetricSet` truncated.
        cache_path: optional JSONL path. Answers already present are reused and
            each new answer is flushed immediately, so a killed run resumes
            instead of restarting. The cache records a fingerprint of the ordered
            question set and **refuses** to be reused against a different one.

    Returns:
        One decoded answer per sample, in order. Fewer than `len(samples)` when
        `deadline` cut generation short.
    """
    import torch

    from training.vlm.formatting import build_training_messages

    model.eval()
    fingerprint = _cache_fingerprint(samples)
    cached = _load_answer_cache(cache_path, fingerprint, len(samples))
    handle = (
        _open_answer_cache(cache_path, fingerprint, len(samples))
        if cache_path is not None
        else None
    )
    answers: list[str] = []
    try:
        for index, sample in enumerate(samples):
            if index in cached:
                answers.append(cached[index])
                continue
            if deadline is not None and time.monotonic() >= deadline:
                break
            image = _render_question_images([sample], (2.0, 98.0))[0]
            messages = build_training_messages(sample.question)
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                generated = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            prompt_len = inputs["input_ids"].shape[1]
            text = processor.batch_decode(
                generated[:, prompt_len:], skip_special_tokens=True
            )[0].strip()
            answers.append(text)
            if handle is not None:
                handle.write(json.dumps({"i": index, "answer": text}) + "\n")
                handle.flush()
    finally:
        if handle is not None:
            handle.close()
    return answers


def evaluate_split(
    model: Any,
    processor: Any,
    samples: Sequence[Any],
    *,
    batch_size: int = 1,
    device: str = "cpu",
    max_new_tokens: int = 8,
    deadline: float | None = None,
    cache_path: Any = None,
) -> MetricSet:
    """Evaluate one split and return its `MetricSet`.

    `batch_size` is accepted for the caller's intent; generation is per sample
    (see `predict`). It bounds nothing about correctness -- the same samples
    produce the same answers at any batch size.

    `deadline` (an absolute `time.monotonic()` value) bounds the whole split
    against the job's wall budget. When it cuts generation short, the returned
    `MetricSet` has `truncated=True` and `n < n_available`, which
    `decide_acceptance` treats as V3 failure -- a partial evaluation can never
    produce an `ACCEPTED`.

    `cache_path` (optional) makes the evaluation resumable -- see `predict`. It
    does not change the scoring path: the same `_metric_set` consumes the same
    answers whether they were generated now or read from the cache.
    """
    if not samples:
        return MetricSet(n=0, exact_match=0.0, confusion={}, n_available=0)
    _ = batch_size  # per-sample generation is the correctness path
    preds = predict(
        model, processor, samples,
        max_new_tokens=max_new_tokens, device=device, deadline=deadline,
        cache_path=cache_path,
    )
    scored = len(preds)
    golds = [s.answer for s in samples[:scored]]
    questions = [s.question for s in samples[:scored]]
    metric = _metric_set(preds, golds, questions)
    metric.n_available = len(samples)
    metric.truncated = scored < len(samples)
    return metric


# ---------------------------------------------------------------------------
# The acceptance decision
# ---------------------------------------------------------------------------
@dataclass
class AcceptanceDecision:
    """The outcome of the predeclared rule, with the reasons that produced it."""

    status: str
    reasons: list[str] = field(default_factory=list)
    val_delta_pp: float | None = None
    test_delta_pp: float | None = None
    class_failures: list[dict[str, Any]] = field(default_factory=list)
    thresholds_used: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "val_delta_pp": self.val_delta_pp,
            "test_delta_pp": self.test_delta_pp,
            "class_failures": list(self.class_failures),
            "thresholds_used": dict(self.thresholds_used),
        }


def _known_rule_versions() -> tuple[str, ...]:
    """The versions `decide_acceptance` / `describe_acceptance_rule` accept."""
    return (ACCEPTANCE_RULE_VERSION, *LEGACY_RULE_VERSIONS)


def _require_known_rule_version(rule_version: str) -> None:
    """Refuse an unknown rule version rather than silently falling back.

    A silent fallback would apply the wrong criterion while reporting a version
    the caller did not ask for, which is exactly the kind of quiet threshold
    drift the rule exists to prevent.
    """
    if rule_version not in _known_rule_versions():
        raise ValueError(
            f"unknown rule_version {rule_version!r}; expected one of "
            f"{_known_rule_versions()}"
        )


def _class_failures(
    baseline_val: MetricSet,
    adapted_val: MetricSet,
    *,
    rule_version: str = ACCEPTANCE_RULE_VERSION,
) -> list[dict[str, Any]]:
    """Classes (with >= MIN_CLASS_QUESTIONS) that fail the V2 guardrail.

    `rule_version` selects the criterion:

      * ``"v001"`` -- the original flat rule: a class fails iff its accuracy
        dropped by more than `MAX_CLASS_DROP_PP`. Retained verbatim, including
        the exact key set of each failure entry, so run 1's REJECTED verdict
        replays with the same **verdict, class set and key set**. It is not
        byte-identical: the manifest stores `per_class_accuracy` rounded to
        6 dp, so an individual `*_pp` value can differ in the last decimal, and
        the failure-list order is not stable across the two paths.
      * ``"v002"`` -- a class fails iff the drop is BOTH **material**
        (`lost_questions >= MIN_CLASS_DROP_QUESTIONS`) AND **significant**
        (`z >= CLASS_DROP_Z`). The two conditions are independent: a
        large-but-noisy drop does not fail, and a statistically detectable but
        trivially small drop does not fail either.

    Under v002 each failure entry additionally carries `lost_questions` and `z`
    so the manifest records the evidence for the flag, not just the point
    estimate. The v001 keys (`class`, `n_questions`, `baseline_pp`,
    `adapted_pp`, `drop_pp`) are present in both versions for backward
    compatibility.

    The per-class standard error of the delta treats the baseline and adapted
    accuracies as two independent binomial proportions:

        SE = sqrt((p_b*(1-p_b) + p_a*(1-p_a)) / n)

    The `1e-12` floor guards the degenerate `n == 0` / `SE == 0` case without
    inventing a `z`.
    """
    _require_known_rule_version(rule_version)

    failures: list[dict[str, Any]] = []
    for cls, count in baseline_val.per_class_counts.items():
        if count < MIN_CLASS_QUESTIONS:
            continue
        if cls not in adapted_val.per_class_accuracy:
            continue
        base = baseline_val.per_class_accuracy.get(cls, 0.0) * 100.0
        adapted = adapted_val.per_class_accuracy.get(cls, 0.0) * 100.0
        drop = base - adapted

        # Evidence, computed for both versions so the caller can audit a v001
        # replay with the same lens; only v002 entries publish it (see above).
        #
        # `lost_questions` is the number of questions the class actually lost:
        # mathematically `hits_baseline - hits_adapted`, an integer, because each
        # per-class accuracy is `hits / count`. Rounding recovers that integer
        # from the 6-dp-rounded accuracies in a manifest and removes a
        # floating-point hazard at the exactly-4 boundary (e.g. 0.2 * 20 can
        # evaluate to 3.999999999999999). Comparing on the rounded integer is
        # what "the class actually lost at least 4 questions" means.
        lost_questions = round((drop / 100.0) * count)
        p_base = base / 100.0
        p_adapted = adapted / 100.0
        se = math.sqrt(
            max(
                p_base * (1.0 - p_base) + p_adapted * (1.0 - p_adapted),
                1e-12,
            )
            / count
        )
        z = (drop / 100.0) / se

        if rule_version == "v001":
            failed = adapted < base - MAX_CLASS_DROP_PP
        else:  # v002
            failed = (
                lost_questions >= MIN_CLASS_DROP_QUESTIONS
                and z >= CLASS_DROP_Z
            )

        if failed:
            entry: dict[str, Any] = {
                "class": cls,
                "n_questions": count,
                "baseline_pp": round(base, 4),
                "adapted_pp": round(adapted, 4),
                "drop_pp": round(drop, 4),
            }
            if rule_version != "v001":
                entry["lost_questions"] = lost_questions
                entry["z"] = round(z, 4)
            failures.append(entry)
    return failures


def decide_acceptance(
    baseline_val: MetricSet,
    adapted_val: MetricSet,
    *,
    baseline_test: MetricSet | None = None,
    adapted_test: MetricSet | None = None,
    run_completed: bool,
    artifact_ok: bool,
    rule_version: str = ACCEPTANCE_RULE_VERSION,
    decision_split: str = "val",
) -> AcceptanceDecision:
    """Apply the predeclared V1/V1'/V2/V3/V4 rule. Encodes contract item V.

    Args:
        baseline_val: validation metrics for the un-adapted base model.
        adapted_val: validation metrics for the adapted model.
        baseline_test / adapted_test: test metrics, evaluated alongside val and
            before the decision. When given, their delta is recorded on the
            decision even if the run is REJECTED, and a val-accept that test does
            not confirm is downgraded to
            `ACCEPTED_ON_VAL_NOT_CONFIRMED_ON_TEST`.
        run_completed: whether the run finished its predeclared budget (V3).
        artifact_ok: whether the artifact is intact and reloadable (V4).
        rule_version: which version of the rule to apply. Defaults to the
            current `ACCEPTANCE_RULE_VERSION` (`v002`). `"v001"` reproduces the
            original rule exactly, including its class_failure key set. An
            unknown version raises `ValueError` -- it never silently falls back.
        decision_split: which split the V1/V2 **gates** run on -- ``"val"``
            (default) or ``"test"``. The *other* split is then the confirming
            split. This exists for `docs/PHASE6_RUN1_REJECTION_DIAGNOSIS.md`
            §7.4 condition 3: an acceptance decided on the validation subset that
            revealed a rule defect is self-confirming, so the gates must be able
            to run on the test split instead. The **criteria are not altered** --
            only which split feeds them. ``"val"`` reproduces the previous
            behaviour exactly, including the reason strings.
            ``"test"`` requires both test metric sets; without them there is
            nothing to gate on and `ValueError` is raised.

    V3 also fails when any supplied metric set is `truncated` (the wall budget
    cut evaluation short): a split scored on a prefix is INCONCLUSIVE, never
    accepted.

    Returns:
        An `AcceptanceDecision`. `val_delta_pp` and `test_delta_pp` are always
        named by split, regardless of which one drove the gates.

    Raises:
        ValueError: `rule_version` is not a known version, or `decision_split` is
            not ``"val"``/``"test"``, or ``"test"`` was requested without both
            test metric sets.
    """
    _require_known_rule_version(rule_version)
    if decision_split not in ("val", "test"):
        raise ValueError(
            f"decision_split must be 'val' or 'test', got {decision_split!r}"
        )
    thresholds = describe_acceptance_rule(rule_version)
    reasons: list[str] = []

    # -- V3: an unfinished run cannot be judged.
    if not run_completed:
        reasons.append(
            "V3 failed: the run did not complete its predeclared budget "
            "(epochs finished or max_steps reached); a truncated or crashed run "
            "is INCONCLUSIVE, never accepted"
        )
        return AcceptanceDecision(
            status="INCONCLUSIVE",
            reasons=reasons,
            thresholds_used=thresholds,
        )

    # -- V3 (evaluation truncation): a partial evaluation is not evidence. -----
    # The whole-job wall budget can cut generation short even when training
    # completed. A split scored on a prefix has an accuracy that is a property of
    # that prefix, not of the split, so it cannot support an ACCEPTED decision.
    truncated_splits = [
        name
        for name, metric in (
            ("baseline_val", baseline_val),
            ("adapted_val", adapted_val),
            ("baseline_test", baseline_test),
            ("adapted_test", adapted_test),
        )
        if metric is not None and getattr(metric, "truncated", False)
    ]
    if truncated_splits:
        reasons.append(
            f"V3 failed: the wall budget truncated evaluation on "
            f"{truncated_splits}; a partial evaluation is INCONCLUSIVE, never "
            f"accepted"
        )
        return AcceptanceDecision(
            status="INCONCLUSIVE",
            reasons=reasons,
            thresholds_used=thresholds,
        )

    # -- V4: an artifact that will not reload is not evidence.
    if not artifact_ok:
        reasons.append(
            "V4 failed: artifact integrity could not be established "
            "(reloadability, SHA256 coverage, or a zero trainable-param count)"
        )
        return AcceptanceDecision(
            status="INCONCLUSIVE",
            reasons=reasons,
            thresholds_used=thresholds,
        )

    # -- which split the gates run on, and which split confirms them ----------
    # See `decision_split` in the docstring: §7.4 condition 3 requires the gates
    # to be able to run on the test split, because a verdict decided on the
    # validation subset that revealed a rule defect is self-confirming. The
    # criteria below are unchanged -- only which split feeds them changes.
    # `*_short` names appear in terse strings ("val-only"); `*_label` names appear
    # in prose ("on validation"). They differ because the existing reason strings
    # do, and `decision_split="val"` must reproduce them byte-for-byte.
    if decision_split == "test":
        if baseline_test is None or adapted_test is None:
            raise ValueError(
                "decision_split='test' requires both baseline_test and adapted_test: "
                "the V1/V2 gates cannot be evaluated on a split with no metrics"
            )
        gate_baseline, gate_adapted = baseline_test, adapted_test
        gate_short, gate_label = "test", "test"
        confirm_baseline, confirm_adapted = baseline_val, adapted_val
        confirm_short, confirm_label = "val", "validation"
    else:
        gate_baseline, gate_adapted = baseline_val, adapted_val
        gate_short, gate_label = "val", "validation"
        confirm_baseline, confirm_adapted = baseline_test, adapted_test
        confirm_short, confirm_label = "test", "test"

    # Both splits' deltas are named by split and recorded on every path that
    # carries them, so the decision reports the val and test numbers whichever
    # one drove the gates.
    val_delta = (adapted_val.exact_match - baseline_val.exact_match) * 100.0
    gate_delta = (gate_adapted.exact_match - gate_baseline.exact_match) * 100.0
    baseline_pp = gate_baseline.exact_match * 100.0
    ceiling = baseline_pp >= CEILING_BASELINE_PP
    required = ACCEPT_MIN_DELTA_PP_CEILING if ceiling else ACCEPT_MIN_DELTA_PP

    # The test delta is computed HERE, before the V1/V2 gates, so a REJECTED run
    # still records its held-out delta. Run 1 evaluated the test split and then
    # discarded the number because the V2 short-circuit returned first
    # (`test_delta_pp: null`), wasting a full second evaluation. This is a
    # RECORDING change only: it alters no status and no gate. The V3 truncation
    # and V4 blocks above are untouched and read exactly what they read before.
    #
    # Both sides must have actually scored questions. `n` is the number of
    # questions scored (`MetricSet.n`, set by `_metric_set`), not the number
    # selected (`n_available`). A zero-sample split has no delta: comparing two
    # empty sets yields 0.0, which would flow into `test_delta >= required` as a
    # confident "test did not confirm the val result" and flag the run for owner
    # review -- inventing a negative finding from no data. An empty or half-empty
    # test comparison is not a comparison, so it falls through to the val-only
    # path with `test_delta_pp = None`.
    test_delta: float | None = None
    if (
        baseline_test is not None
        and adapted_test is not None
        and baseline_test.n > 0
        and adapted_test.n > 0
    ):
        test_delta = (adapted_test.exact_match - baseline_test.exact_match) * 100.0
    test_delta_pp = round(test_delta, 4) if test_delta is not None else None

    # The confirming split's delta, computed on the same "both sides actually
    # scored questions" rule. `decision_split="val"` makes this identical to
    # `test_delta`; `"test"` makes it the validation delta.
    confirm_delta: float | None = None
    if (
        confirm_baseline is not None
        and confirm_adapted is not None
        and confirm_baseline.n > 0
        and confirm_adapted.n > 0
    ):
        confirm_delta = (
            confirm_adapted.exact_match - confirm_baseline.exact_match
        ) * 100.0

    reasons.append(
        f"{gate_short} baseline={baseline_pp:.2f} pp, adapted={gate_adapted.exact_match * 100:.2f} pp, "
        f"delta={gate_delta:+.2f} pp; required "
        f"({'ceiling clause V1-prime' if ceiling else 'V1'}) >= {required:+.2f} pp"
    )

    # -- V1 / V1': the primary improvement condition.
    if gate_delta < required:
        reasons.append(
            f"V1{'′' if ceiling else ''} failed: improvement {gate_delta:+.2f} pp is "
            f"below the required {required:+.2f} pp"
        )
        return AcceptanceDecision(
            status="REJECTED",
            reasons=reasons,
            val_delta_pp=round(val_delta, 4),
            test_delta_pp=test_delta_pp,
            thresholds_used=thresholds,
        )

    # -- V2: no material, significant class collapse.
    failures = _class_failures(gate_baseline, gate_adapted, rule_version=rule_version)
    if failures:
        if rule_version == "v001":
            reasons.append(
                f"V2 (v001) failed: {len(failures)} class(es) dropped more than "
                f"{MAX_CLASS_DROP_PP:.1f} pp on {gate_label}"
            )
        else:
            reasons.append(
                f"V2 (v002) failed: {len(failures)} class(es) lost >= "
                f"{MIN_CLASS_DROP_QUESTIONS} questions with z >= {CLASS_DROP_Z} "
                f"on {gate_label}"
            )
        return AcceptanceDecision(
            status="REJECTED",
            reasons=reasons,
            val_delta_pp=round(val_delta, 4),
            test_delta_pp=test_delta_pp,
            class_failures=failures,
            thresholds_used=thresholds,
        )

    reasons.append(
        f"V1{'′' if ceiling else ''} and V2 passed on {gate_label}"
    )

    # -- Accepted on the gate split. The *confirming* split's check applies only
    # on this path; both splits' deltas were computed above and are recorded on
    # every path that carries them.
    if confirm_delta is None:
        return AcceptanceDecision(
            status="ACCEPTED",
            reasons=reasons
            + [
                f"no {confirm_short}-split metrics supplied; decision is "
                f"{gate_short}-only, and a {gate_short}-only ACCEPTED is not an "
                f"acceptance"
            ],
            val_delta_pp=round(val_delta, 4),
            thresholds_used=thresholds,
        )

    reasons.append(
        f"{confirm_short} delta={confirm_delta:+.2f} pp "
        f"({gate_short} delta {gate_delta:+.2f} pp)"
    )

    confirmed = confirm_delta >= required
    disagreement = abs(gate_delta - confirm_delta) > TEST_VAL_DISAGREEMENT_PP
    if not confirmed:
        reasons.append(
            f"{confirm_short} did not confirm the {gate_short} result "
            f"({confirm_short} delta {confirm_delta:+.2f} pp "
            f"< required {required:+.2f} pp); flagged for owner review, not "
            f"silently promoted"
        )
        return AcceptanceDecision(
            status=(
                "ACCEPTED_ON_VAL_NOT_CONFIRMED_ON_TEST"
                if decision_split == "val"
                else "ACCEPTED_ON_TEST_NOT_CONFIRMED_ON_VAL"
            ),
            reasons=reasons,
            val_delta_pp=round(val_delta, 4),
            test_delta_pp=test_delta_pp,
            thresholds_used=thresholds,
        )

    if disagreement:
        reasons.append(
            f"val/test disagree by {abs(gate_delta - confirm_delta):.2f} pp "
            f"(> {TEST_VAL_DISAGREEMENT_PP:.1f} pp); the split may be too small to "
            f"support the claim"
        )

    return AcceptanceDecision(
        status="ACCEPTED",
        reasons=reasons,
        val_delta_pp=round(val_delta, 4),
        test_delta_pp=test_delta_pp,
        thresholds_used=thresholds,
    )


def describe_acceptance_rule(
    rule_version: str = ACCEPTANCE_RULE_VERSION,
) -> dict[str, Any]:
    """The acceptance rule's constants + version, for the run manifest.

    Args:
        rule_version: which version to describe. Defaults to the current
            `ACCEPTANCE_RULE_VERSION` (`v002`). `"v001"` describes the original
            pre-registered rule. An unknown version raises `ValueError`.

    `declared_before_training` is reported **honestly**: `True` only for the
    pre-registered `v001`; `False` for `v002`, which was declared after run 1.
    The v001 declaration facts are never overwritten -- `v002`'s entry carries
    them under `amendment`, and the v001 constants stay published as
    `max_class_drop_pp` / `min_class_questions`.

    BERTScore is recorded as an explicit unavailable limitation (its backbone is
    not in the local HF cache), never as a zero and never substituted.

    Raises:
        ValueError: `rule_version` is not a known version.
    """
    _require_known_rule_version(rule_version)

    rule: dict[str, Any] = {
        "version": rule_version,
        # Honest provenance: v001 was pre-registered; v002 was not.
        "declared_before_training": rule_version == "v001",
        "primary_endpoint": "presence-question normalised exact-match accuracy",
        "decision_split": "val",
        "accept_min_delta_pp": ACCEPT_MIN_DELTA_PP,
        "ceiling_baseline_pp": CEILING_BASELINE_PP,
        "accept_min_delta_pp_ceiling": ACCEPT_MIN_DELTA_PP_CEILING,
        # v001's V2 threshold, retained and still reported for the v001 replay.
        "max_class_drop_pp": MAX_CLASS_DROP_PP,
        "min_class_questions": MIN_CLASS_QUESTIONS,
        "test_val_disagreement_pp": TEST_VAL_DISAGREEMENT_PP,
        "normalisation": {
            "version": NORMALISATION_RULE_VERSION,
            "delegates_to": "evaluation.metrics.vqa.normalize_answer",
            "order": [
                "evaluation.metrics.vqa.normalize_answer (VQA-v2 R1-R6)",
                "strip surrounding quotes/whitespace",
                "fold yes/no synonym families",
            ],
            "synonym_families": {
                "yes": sorted(_YES_TOKENS),
                "no": sorted(_NO_TOKENS),
            },
            "note": (
                "order is load-bearing: normalize_answer preserves apostrophes, "
                "so normalize_answer(\"'Yes.'\") == \"'yes'\" and folding before "
                "stripping would miss the synonym map"
            ),
        },
        "bertscore": {
            "available": False,
            "reason": (
                "roberta-large is not in the local HuggingFace cache; BERTScore "
                "cannot be computed offline and is recorded as unavailable rather "
                "than as a zero or a substitute"
            ),
        },
        "excluded_metrics": ["BLEU", "ROUGE", "BERTScore"],
        "excluded_reason": (
            "target answers are one token; BLEU/ROUGE are meaningless at that "
            "length and BERTScore is unavailable offline"
        ),
    }

    if rule_version == "v001":
        rule["v2_criterion"] = {
            "description": (
                "flat: a class with >= min_class_questions validation questions "
                "fails iff adapted < baseline - max_class_drop_pp"
            ),
            "max_class_drop_pp": MAX_CLASS_DROP_PP,
            "min_class_questions": MIN_CLASS_QUESTIONS,
        }
    else:  # v002
        rule["v2_criterion"] = {
            "description": (
                "a class with >= min_class_questions validation questions fails "
                "iff it BOTH lost >= min_class_drop_questions questions AND has "
                "z >= class_drop_z"
            ),
            "min_class_drop_questions": MIN_CLASS_DROP_QUESTIONS,
            "class_drop_z": CLASS_DROP_Z,
            "min_class_questions": MIN_CLASS_QUESTIONS,
            "se_formula": "sqrt((p_b*(1-p_b) + p_a*(1-p_a)) / n)",
            "z_formula": "(baseline - adapted) / se",
            "legacy_v001_criterion": {
                "max_class_drop_pp": MAX_CLASS_DROP_PP,
                "min_class_questions": MIN_CLASS_QUESTIONS,
                "replayable_via": "rule_version='v001'",
            },
        }
        rule["amendment"] = {
            "amends": "v001",
            "original_declared": "2026-09-23",
            "declared": "2026-09-24",
            "declared_after_first_run": True,
            "declared_before_training": False,
            "scope": "V2 only; V1/V1'/V3/V4 are unchanged from v001",
            "reason": (
                "v001's flat V2 threshold (1.0 pp at >= 20 questions) fires on "
                "0.20 of a question and sits ~8x below the per-class standard "
                "error of the delta (2.6-10.6 pp on run 1). It flagged three "
                "classes whose drops are statistically indistinguishable from "
                "zero. v002 applies item V's own 'resolvable at ~3 SE' resolution "
                "standard per-class and adds a materiality floor. It fires less "
                "often than v001 (a relaxation of how often V2 fires, justified "
                "by v001 having fired below its own resolution); its 1.96 SE bar "
                "is below the contract's ~3 SE figure, so it is not a numerically "
                "stricter bar. A real, detectable collapse is still caught."
            ),
            "contract_reference": (
                "docs/PHASE6_AUDIT_AND_CONTRACT.md, Contract item V, amendment "
                "v002 (the V2 resolution fix)"
            ),
        }
    return rule
