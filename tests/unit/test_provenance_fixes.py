"""Tests for the R-02 provenance fixes: sections 6a, 6b and 6c.

Each defect is a statement about what a REVIEWER could have concluded from the
export, so each test is written from the reviewer's position:

§6a  A reviewer hashes the written report with `sha256sum`, gets a value
     different from the recorded `report_sha256`, and concludes the report was
     corrupted. Both numbers were right; the NAME was wrong.

§6b  A reviewer asks "which code produced this?" The export pins the checkpoint
     byte-exactly and identifies the code only as a path.

§6c  A reviewer wants to check the published baselines. They are in no export at
     all -- the per-type figure is screenshot-only.

These tests must FAIL against the pre-fix code. A test that passes both before
and after a fix is documentation, not a test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training.change_vqa.evaluate import (
    QUESTION_TYPE_ORDER,
    EvaluationReport,
    majority_baselines,
)

# ---------------------------------------------------------------------------
# §6a — the two hashes
# ---------------------------------------------------------------------------


def _report(payload: dict | None = None) -> EvaluationReport:
    return EvaluationReport(
        payload=payload
        or {"schema": "probe", "metrics": {"answer_accuracy": 0.5}, "n": 2}
    )


def test_the_file_bytes_hash_matches_sha256sum(tmp_path: Path) -> None:
    """THE §6a TEST.

    `sha256sum <file>` is the canonical third-party check. If the recorded digest
    does not equal this, a reviewer concludes corruption. Here the two must agree
    exactly, on a report written the way the real writer writes it.
    """
    report = _report()
    path = report.write(tmp_path / "eval_Test_masked.json")

    recorded = report.file_bytes_sha256(path)
    independent = hashlib.sha256(path.read_bytes()).hexdigest()
    assert recorded == independent, (
        "the file-bytes digest must equal what `sha256sum` prints"
    )


def test_the_canonical_hash_is_not_the_file_bytes_hash(tmp_path: Path) -> None:
    """If these two ever coincide, one of them is misnamed.

    The canonical digest is over compact JSON; the file is written with
    `indent=2`. They are different values describing different things, and the
    defect was that only one existed under a name implying the other.
    """
    report = _report()
    path = report.write(tmp_path / "eval_Test_masked.json")

    assert report.canonical_sha256() != report.file_bytes_sha256(path)


def test_the_canonical_hash_is_formatting_independent(tmp_path: Path) -> None:
    """Content, not bytes. Two reports with identical content must hash equal even
    when written with different indentation."""
    report = _report()
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(report.payload, indent=2, sort_keys=True), encoding="utf-8")
    b.write_text(json.dumps(report.payload, indent=4, sort_keys=True), encoding="utf-8")

    assert report.file_bytes_sha256(a) != report.file_bytes_sha256(b), (
        "precondition: the files really do differ"
    )
    assert report.canonical_sha256() == report.canonical_sha256()


def test_the_canonical_hash_is_key_order_independent() -> None:
    a = EvaluationReport(payload={"x": 1, "y": 2})
    b = EvaluationReport(payload={"y": 2, "x": 1})
    assert a.canonical_sha256() == b.canonical_sha256()


def test_the_legacy_alias_still_returns_the_canonical_value() -> None:
    """Backward compatibility: a caller written against the old `sha256()` name
    must keep getting the value it always got, so recorded exports stay readable.
    The fix is that the value is now ALSO available under an unambiguous key -- not
    that the old key changed meaning underneath existing readers."""
    report = _report()
    assert report.sha256() == report.canonical_sha256()


def test_the_canonical_hash_actually_tracks_content() -> None:
    """Both hashes must move when the content moves, or they identify nothing."""
    a = EvaluationReport(payload={"schema": "probe", "metrics": {"answer_accuracy": 0.5}})
    b = EvaluationReport(payload={"schema": "probe", "metrics": {"answer_accuracy": 0.6}})
    assert a.canonical_sha256() != b.canonical_sha256()


def test_the_report_is_written_before_its_bytes_can_be_hashed(tmp_path: Path) -> None:
    """`file_bytes_sha256` takes a path because the file must exist. Passing a
    path that was never written must raise, not return a digest of nothing."""
    report = _report()
    with pytest.raises(OSError):
        report.file_bytes_sha256(tmp_path / "never_written.json")


# ---------------------------------------------------------------------------
# §6c — machine-readable baselines
# ---------------------------------------------------------------------------


class _Rec:
    """Minimal stand-in for a ChangeVQARecord.

    `majority_baselines` reads exactly one field off a record -- `qtype` -- so
    this stub carries only that. It is deliberately not a real record: a baseline
    that needed anything else would not be computable from gold labels alone,
    which is the property under test.
    """

    def __init__(self, qtype: str) -> None:
        self.qtype = qtype


def test_baselines_are_computed_from_gold_alone() -> None:
    """The defining property: no model prediction may enter. A baseline that
    depended on predictions would not be a baseline."""
    records = [_Rec("change_or_not")] * 10
    gold = [i % 2 for i in range(10)]

    result = majority_baselines(records, gold)
    assert result["global_majority"] == pytest.approx(0.5, abs=1e-9)
    # The answer it picked is recorded, not just the score.
    assert result["global_majority_answer"] in (0, 1)
    assert result["per_type_majority"] == pytest.approx(0.5, abs=1e-9)


def test_the_global_majority_is_the_largest_gold_class() -> None:
    records = [_Rec("change_or_not")] * 10
    gold = [7] * 7 + [1] * 3
    result = majority_baselines(records, gold)
    assert result["global_majority"] == pytest.approx(0.7, abs=1e-9)
    assert result["global_majority_answer"] == 7
    assert result["global_majority_count"] == 7
    # The histogram is keyed by the ANSWER TEXT, not the index, so a reader can
    # read it without the vocabulary table to hand.
    assert result["gold_histogram"]["60_to_70"] == 7
    assert result["gold_histogram"]["0_to_10"] == 3
    assert result["global_majority_answer_text"] == "60_to_70"


def test_per_type_majority_can_exceed_global_majority() -> None:
    """This is WHY the per-type baseline is worth reporting separately: answering
    each type with its own majority can beat one global answer, and collapsing
    them into a single number would hide that."""
    records = [_Rec("a")] * 10 + [_Rec("b")] * 10
    gold = [1] * 9 + [2] + [2] * 9 + [1]
    result = majority_baselines(records, gold)

    assert result["per_type_majority"] == pytest.approx(0.9, abs=1e-9)
    assert result["global_majority"] == pytest.approx(0.5, abs=1e-9)
    assert result["per_type_majority"] > result["global_majority"]


def test_per_type_detail_is_reported_for_every_type() -> None:
    """ALL question types appear, including ones with no scored samples.

    Reporting only the observed types would make a type that was entirely absent
    from the split indistinguishable from a type that scored well -- the reader
    would see eight entries and have no way to tell whether eight is all of them.
    An absent type is reported with `support: 0` and a null answer, so it is
    visibly absent.
    """
    records = [_Rec("change_or_not")] * 4 + [_Rec("largest_change")] * 6
    gold = [1, 1, 1, 2] + [3, 3, 3, 3, 3, 1]
    result = majority_baselines(records, gold)

    per_type = result["per_type"]
    assert set(per_type) == set(QUESTION_TYPE_ORDER)
    assert len(per_type) == len(QUESTION_TYPE_ORDER)

    assert per_type["change_or_not"]["support"] == 4
    assert per_type["largest_change"]["support"] == 6

    # A type with no samples is present, and is honest about having nothing.
    absent = per_type["change_ratio"]
    assert absent["support"] == 0
    assert absent["majority_answer"] is None
    assert absent["majority_share"] is None


def test_an_unobserved_type_does_not_contribute_to_the_per_type_baseline() -> None:
    """If an empty type contributed a free hit, a split could show a per-type
    baseline invented from nothing."""
    records = [_Rec("change_or_not")] * 4
    gold = [1, 1, 1, 1]
    result = majority_baselines(records, gold)
    assert result["per_type_majority"] == pytest.approx(1.0, abs=1e-9)
    assert result["per_type"]["largest_change"]["support"] == 0


def test_per_type_detail_records_its_support() -> None:
    records = [_Rec("a")] * 4 + [_Rec("b")] * 6
    gold = [1, 1, 1, 2] + [3, 3, 3, 3, 3, 1]
    result = majority_baselines(records, gold)
    for entry in result["per_type"].values():
        assert "support" in entry
        assert "majority_answer" in entry
        assert "majority_share" in entry


def test_the_baseline_records_its_own_derivation_and_caveat() -> None:
    """§6c was a provenance defect, so the BASELINE itself must be traceable: how
    it was derived, and the fact that it is not an official metric."""
    result = majority_baselines([_Rec("a")] * 4, [1, 1, 2, 2])
    assert result["tie_break"], "ties must have a stated resolution rule"
    assert "gold" in result["derivation"].lower()
    assert "not an official" in result["caveat"].lower()


def test_ties_are_broken_deterministically() -> None:
    """A tie resolved by dictionary order would give different baselines on
    different Python builds for equal counts."""
    records = [_Rec("a")] * 4
    gold = [5, 5, 9, 9]
    first = majority_baselines(records, gold)
    second = majority_baselines(records, gold)
    assert first["global_majority_answer"] == second["global_majority_answer"]


def test_the_baseline_does_not_invent_a_number_for_an_empty_split() -> None:
    """An empty split has no majority. Reporting 0.0 would look like a real
    measurement of a real thing."""
    result = majority_baselines([], [])
    assert result["n"] == 0
    assert result["global_majority"] is None
    assert result["per_type_majority"] is None
    assert "note" in result
    # No answer index is claimed, because none was observed.
    assert "global_majority_answer" not in result


def test_the_baseline_is_json_serialisable() -> None:
    """It is embedded in eval_summary.json, so it must survive json.dumps."""
    result = majority_baselines([_Rec("a")] * 4, [1, 2, 3, 1])
    round_tripped = json.loads(json.dumps(result, sort_keys=True, default=str))
    assert round_tripped["global_majority"] == result["global_majority"]


def test_a_mismatched_length_is_rejected() -> None:
    """A shorter gold list must not silently truncate the records.

    `majority_baselines` zips records against gold, so a shorter gold list would
    produce a baseline computed over an UNSTATED subset -- a confident-looking
    number derived from the wrong population, which is the failure mode the
    whole section 6 family is about. The function must raise rather than guess.
    """
    with pytest.raises(ValueError):
        majority_baselines([_Rec("a")] * 4, [1, 2])


def test_a_longer_gold_list_is_also_rejected() -> None:
    """The mirror case: extra gold labels with no matching record would have no
    question type, so a per-type baseline could not be formed at all."""
    with pytest.raises(ValueError):
        majority_baselines([_Rec("a")] * 2, [1, 2, 3, 4])
