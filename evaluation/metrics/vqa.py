"""VQA and Change-VQA text metrics — exact match, normalized exact match,
token F1, and answer accuracy.

TIER 1 — PURE STDLIB, NO DEPENDENCIES
-------------------------------------
Plan §2.2 ("Evaluation dimensions") names the metrics for two families:

    VQA        (`:258-266`) : exact match; normalized exact match; "F1 where appropriate"
    Change VQA (`:308-312`) : answer accuracy; semantic match if the benchmark specifies it

This module implements the half that needs no model and no new dependency. The
dependency-heavy half (BLEU / ROUGE-L / CIDEr / BERTScore for captioning, and
"semantic match" for Change-VQA) is deliberately NOT attempted here: BERTScore
pulls torch, which would poison the model-free import paths. Only the standard
library is imported (`re`, `collections`).

WHAT "EXACT MATCH" MEANS HERE
-----------------------------
`exact_match` is RAW string equality: case- and punctuation-sensitive, with no
normalization. `normalized_exact_match` applies `normalize_answer` to both sides
first. Both return a float in {0.0, 1.0} (never a bool), so they compose with
`answer_accuracy` and any averaging without a cast.

THE NORMALIZATION RULE SET (a convention choice, NOT plan-mandated)
-------------------------------------------------------------------
The plan says "normalized exact match" without defining the rules. The rule set
below is the VQA-v2 convention, applied in this order, one rule per line:

    R1  lowercase the text
    R2  fix common contractions      ("dont" -> "don't")
    R3  map number words to digits   ("two"  -> "2")
    R4  remove articles              ("a", "an", "the")
    R5  strip punctuation            (each punctuation character becomes a space;
                                      a period or comma BETWEEN TWO DIGITS is
                                      preserved, so "3.14" and "1,000" survive)
    R6  collapse whitespace and strip

R1-R6 are published as `NORMALIZATION_RULES`, and the tables behind R2-R5 are
published as `CONTRACTIONS`, `NUMBER_WORDS`, `ARTICLES` and `PUNCTUATION`, so
the rules are inspectable rather than buried inline. Each rule is pinned by its
own test (`tests/unit/test_vqa_metrics.py`), so an edit that silently drops a
rule fails a test.

NOTE FOR THE OWNER — THIS IS A JUDGEMENT CALL
---------------------------------------------
This rule set is a CONVENTION CHOICE. The plan does not mandate it, and the
exact membership of the contraction and punctuation tables is a judgement call
(the contraction table is a curated common subset of the VQA-v2 map). It is
implemented because *some* documented, deterministic rule set is required for
"normalized exact match" to mean anything, and VQA-v2 is the standard one. It
should be ratified — or replaced — before these numbers are treated as a
benchmark figure.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

# ---------------------------------------------------------------------------
# The published rule set
# ---------------------------------------------------------------------------
#: The normalization rules, in application order. One entry per rule; see the
#: module docstring. Exposed so the pipeline is inspectable at a glance.
NORMALIZATION_RULES: tuple[str, ...] = (
    "R1 lowercase the text",
    "R2 fix common contractions",
    "R3 map number words to digits",
    "R4 remove articles",
    "R5 strip punctuation (digit-internal '.' and ',' preserved)",
    "R6 collapse whitespace and strip",
)

#: Articles dropped by R4.
ARTICLES: frozenset[str] = frozenset({"a", "an", "the"})

#: Number words mapped to digits by R3 (the VQA-v2 `manualMap`).
NUMBER_WORDS: dict[str, str] = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

#: Contractions repaired by R2, keyed by their apostrophe-less form — the form
#: that actually needs repair. A curated common subset of the VQA-v2 contraction
#: map; the full map contains many rare forms that add noise without value here.
CONTRACTIONS: dict[str, str] = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "shant": "shan't",
    "shes": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "somebodys": "somebody's",
    "someones": "someone's",
    "thats": "that's",
    "thered": "there'd",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whod": "who'd",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

#: Punctuation replaced by a space in R5 (the VQA-v2 set). The apostrophe is
#: deliberately EXCLUDED so contractions survive; a period is handled separately
#: so a decimal point is not destroyed.
PUNCTUATION: tuple[str, ...] = (
    ";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_",
    "-", ">", "<", "@", "`", ",", "?", "!",
)

# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
#: A word-ish token: letters/digits, optionally carrying an internal apostrophe
#: (so "don't" stays one token). R2-R4 operate on these tokens.
_TOKEN_RE = re.compile(r"[a-z0-9']+")

#: A period or comma with a digit on BOTH sides — a decimal point or a thousands
#: separator. Preserved through R5 rather than destroyed.
_DIGIT_PERIOD_RE = re.compile(r"(?<=\d)\.(?=\d)")
_DIGIT_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")

#: Whitespace runs, collapsed by R6.
_WHITESPACE_RE = re.compile(r"\s+")

#: Sentinels used to carry digit-internal separators through R5 untouched. They
#: are non-printing and cannot occur in the input, so they cannot collide.
_PERIOD_SENTINEL = "\x00"
_COMMA_SENTINEL = "\x01"


def _map_token(match: re.Match[str]) -> str:
    """Apply R2 (contractions), R3 (number words) and R4 (articles) to a token."""
    word = match.group(0)
    word = CONTRACTIONS.get(word, word)  # R2
    word = NUMBER_WORDS.get(word, word)  # R3
    if word in ARTICLES:  # R4
        return ""
    return word


def _strip_punctuation(text: str) -> str:
    """Apply R5: punctuation becomes a space, but numbers survive intact.

    A period or comma between two digits is first replaced by a sentinel, so the
    punctuation pass cannot touch it; the sentinel is restored afterwards. This
    is what keeps "3.14" and "1,000" from degrading into "3 14" / "1 000".
    """
    text = _DIGIT_PERIOD_RE.sub(_PERIOD_SENTINEL, text)
    text = _DIGIT_COMMA_RE.sub(_COMMA_SENTINEL, text)
    for char in PUNCTUATION:
        text = text.replace(char, " ")
    # Remaining periods are not digit-internal; VQA-v2 removes them outright.
    text = text.replace(".", "")
    return text.replace(_PERIOD_SENTINEL, ".").replace(_COMMA_SENTINEL, ",")


def normalize_answer(text: str) -> str:
    """Apply the R1-R6 rule set and return the normalized answer.

    See the module docstring for the rules. The function is pure and
    idempotent: `normalize_answer(normalize_answer(x)) == normalize_answer(x)`.

    Args:
        text: the answer string to normalize.

    Returns:
        The normalized string (possibly empty, e.g. for an answer that was only
        an article).

    Raises:
        TypeError: `text` is not a string. A non-string here is a caller bug,
            and silently coercing it (e.g. a `None` becoming "none" -> "0")
            would invent an answer.
    """
    if not isinstance(text, str):
        raise TypeError(f"normalize_answer() requires a str, got {type(text).__name__}")

    lowered = text.lower()  # R1
    mapped = _TOKEN_RE.sub(_map_token, lowered)  # R2, R3, R4
    depunctuated = _strip_punctuation(mapped)  # R5
    return _WHITESPACE_RE.sub(" ", depunctuated).strip()  # R6


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def exact_match(pred: str, gold: str) -> float:
    """Raw string equality: 1.0 if `pred == gold`, else 0.0.

    Case- and punctuation-sensitive by design — this is the metric the plan
    names *before* normalization, so folding case here would make it identical
    to `normalized_exact_match` and destroy the distinction between the two.
    """
    return 1.0 if pred == gold else 0.0


def normalized_exact_match(pred: str, gold: str) -> float:
    """1.0 if `normalize_answer(pred) == normalize_answer(gold)`, else 0.0."""
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    """Token-multiset F1 between a prediction and a gold answer.

    Tokens are the whitespace-separated words of `normalize_answer(...)`, and the
    overlap is the MULTISET intersection, so "cat cat" against "cat" scores 2/3,
    not 1.0. A set intersection would ignore repetition and inflate the score.

    Degenerate cases (documented, and pinned by tests):

        * both sides empty           -> 1.0 (they agree: both said nothing)
        * exactly one side empty     -> 0.0 (nothing to match)
        * disjoint tokens            -> 0.0
    """
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def answer_accuracy(preds: Sequence[str], golds: Sequence[str]) -> float:
    """Change-VQA answer accuracy: the fraction of normalized-exact matches.

    `preds[i]` is scored against `golds[i]`. This is the plan's Change-VQA
    "answer accuracy" (`:308-312`); "semantic match" is deliberately not
    attempted (it is benchmark-dependent and the plan leaves it optional).

    Args:
        preds: predicted answers.
        golds: the matching gold answers, same length and order as `preds`.

    Returns:
        A float in [0.0, 1.0]. An empty input returns 0.0 — the no-information
        value, matching `evaluation.metrics.change`'s guarded-division choice.

    Raises:
        ValueError: `preds` and `golds` differ in length. A mismatch means the
            caller has mis-aligned questions and answers, and scoring the
            overlap silently would hide that.
    """
    if len(preds) != len(golds):
        raise ValueError(
            f"answer_accuracy() needs aligned sequences: "
            f"{len(preds)} predictions vs {len(golds)} golds"
        )
    if not preds:
        return 0.0
    hits = sum(
        normalized_exact_match(pred, gold) for pred, gold in zip(preds, golds)
    )
    return hits / len(preds)


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
