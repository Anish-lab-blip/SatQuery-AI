"""Tests for the reBEN label analyser.

The analyser produces the single number the Phase 12 label decision turns on: the
single-label fraction *f*. Option (b) reaches the full 20,000/4,000/4,000 slice
iff *f* >= 7.28%. A wrong *f* therefore silently opens or closes a scientific
option, which is why the analyser refuses to guess an encoding it cannot identify
rather than defaulting to "1 label per cell".

The tests below pin four classes of failure:

  1. the encoding is DETECTED correctly (sequence / one-hot / repr / separated /
     index), because the same cell content means different things under each;
  2. values that carry no label (None, NaN) count as ZERO, not one;
  3. genuinely ambiguous encodings are REFUSED (exit 4), not resolved;
  4. the printed *f* and the threshold verdict are correct end to end.

`test_the_naive_counter_disagrees_with_every_pinned_case` is the non-vacuity
guard: it reimplements the original "one cell = one label" counter and asserts it
gets each pinned case WRONG. If someone reverts the analyser to that counter, the
pinned cases above fail first.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "analyze_reben_labels.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_reben_labels", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_reben_labels"] = module
    spec.loader.exec_module(module)
    return module


ar = _load_module()

ONEHOT_2 = [1, 0, 1] + [0] * 16  # 19 wide, two set bits
ONEHOT_1 = [1, 0, 0] + [0] * 16  # 19 wide, one set bit


# ---------------------------------------------------------------------------
# 1. encoding detection
# ---------------------------------------------------------------------------

def test_sequence_encoding_is_detected():
    enc, why = ar.detect_encoding([["Arable land"], ["Arable land", "Pastures"]])
    assert enc == "sequence"
    assert "sequence" in why


def test_onehot_encoding_is_detected_by_full_width_and_binary_content():
    enc, why = ar.detect_encoding([ONEHOT_1, ONEHOT_2])
    assert enc == "onehot"
    assert "19-wide" in why


def test_a_narrow_binary_list_is_a_sequence_not_a_onehot():
    """Width matters: a 3-element 0/1 list is a list of ids, not a one-hot row."""
    enc, _ = ar.detect_encoding([[1, 0, 1], [0, 1, 0]])
    assert enc == "sequence"


def test_sequence_repr_encoding_is_detected():
    enc, _ = ar.detect_encoding(["['Arable land', 'Pastures']", "['Pastures']"])
    assert enc == "sequence_repr"


def test_separated_string_encoding_is_detected_with_its_separator():
    enc, why = ar.detect_encoding(["Arable land, Pastures", "Pastures"])
    assert enc == "string_separated"
    assert "','" in why


def test_class_index_encoding_is_detected():
    enc, _ = ar.detect_encoding([3, 7, 12, 0])
    assert enc == "index"


# ---------------------------------------------------------------------------
# 2. counting
# ---------------------------------------------------------------------------

def test_sequence_counts_its_elements():
    assert ar.count_labels(["a", "b", "c"], "sequence") == 3
    assert ar.count_labels(["a"], "sequence") == 1
    assert ar.count_labels([], "sequence") == 0


def test_onehot_counts_set_bits_not_the_vector_width():
    """The original bug: len() of a 19-wide row returned 19 for every patch."""
    assert ar.count_labels(ONEHOT_2, "onehot") == 2
    assert ar.count_labels(ONEHOT_1, "onehot") == 1
    assert ar.count_labels([0] * 19, "onehot") == 0


def test_missing_values_count_as_zero_labels():
    """The original bug: str(nan) is truthy, so NaN counted as ONE label."""
    assert ar.count_labels(None, "sequence") == 0
    assert ar.count_labels(float("nan"), "sequence") == 0
    assert ar.count_labels(None, "onehot") == 0
    assert ar.count_labels(float("nan"), "string_separated", ",") == 0


def test_separated_string_counts_non_empty_parts():
    assert ar.count_labels("Arable land, Pastures", "string_separated", ",") == 2
    assert ar.count_labels("Pastures", "string_separated", ",") == 1
    assert ar.count_labels("A, , B", "string_separated", ",") == 2


def test_sequence_repr_counts_the_parsed_elements():
    assert ar.count_labels("['a', 'b']", "sequence_repr") == 2
    assert ar.count_labels("('a',)", "sequence_repr") == 1


def test_index_encoding_counts_exactly_one():
    assert ar.count_labels(7, "index") == 1


# ---------------------------------------------------------------------------
# 3. refusals
# ---------------------------------------------------------------------------

def test_a_bare_string_is_refused_rather_than_counted_as_one():
    """'Arable land Pastures' is either two classes or one class with a space."""
    with pytest.raises(ar.LabelEncodingError) as exc:
        ar.detect_encoding(["Arable land Pastures", "Pastures"])
    assert "ambiguous" in str(exc.value).lower()


def test_a_bool_column_is_refused():
    with pytest.raises(ar.LabelEncodingError):
        ar.detect_encoding([True, False, True])


def test_a_zero_one_integer_column_is_refused_as_ambiguous():
    with pytest.raises(ar.LabelEncodingError) as exc:
        ar.detect_encoding([0, 1, 1, 0])
    assert "ambiguous" in str(exc.value).lower()


def test_out_of_range_integers_are_refused():
    with pytest.raises(ar.LabelEncodingError):
        ar.detect_encoding([0, 1, 99])


def test_mixed_types_are_refused():
    with pytest.raises(ar.LabelEncodingError):
        ar.detect_encoding([["a"], "b"])


def test_an_all_missing_column_is_refused():
    with pytest.raises(ar.LabelEncodingError):
        ar.detect_encoding([None, float("nan")])


def test_separated_encoding_without_a_separator_is_refused():
    with pytest.raises(ar.LabelEncodingError):
        ar.count_labels("a,b", "string_separated", None)


def test_an_unknown_encoding_name_is_refused():
    with pytest.raises(ar.LabelEncodingError):
        ar.count_labels(["a"], "telepathy")


def test_a_value_that_contradicts_the_declared_encoding_is_refused():
    """A str where a sequence was declared must not be silently coerced."""
    with pytest.raises(ar.LabelEncodingError):
        ar.count_labels("a,b", "sequence")
    with pytest.raises(ar.LabelEncodingError):
        ar.count_labels(7, "string_separated", ",")


# ---------------------------------------------------------------------------
# 4. main() end to end
# ---------------------------------------------------------------------------

CLC = list(ar.CLC19_LABELS)


def _frame(
    single: int,
    multi: int,
    zero: int,
    split: str = "train",
    cloudy: int = 0,
    snowy: int = 0,
) -> pd.DataFrame:
    """A synthetic frame using the REAL reBEN class names.

    The analyser validates that sampled label values are members of the CLC19
    vocabulary, so placeholder names like 'class_0' would be (correctly) refused.
    """
    rows = []
    for i in range(single):
        rows.append({"split": split, "labels": [CLC[i % 19]]})
    for i in range(multi):
        rows.append({"split": split, "labels": [CLC[i % 19], CLC[(i + 1) % 19]]})
    for _ in range(zero):
        rows.append({"split": split, "labels": []})
    df = pd.DataFrame(rows)
    df["contains_cloud_or_shadow"] = [True] * cloudy + [False] * (len(df) - cloudy)
    df["contains_seasonal_snow"] = [True] * snowy + [False] * (len(df) - snowy)
    return df


def _run(monkeypatch, tmp_path, df, argv, capsys, companion=None):
    path = tmp_path / "metadata.parquet"
    path.write_bytes(b"stub")  # existence only; load_table is monkeypatched
    tables = [df] if companion is None else [df, companion]

    def fake_load(p):
        return tables.pop(0)

    monkeypatch.setattr(ar, "load_table", fake_load)
    args = ["--metadata", str(path)]
    if companion is not None:
        comp = tmp_path / "snow_cloud.parquet"
        comp.write_bytes(b"stub")
        args += ["--metadata-snow-cloud", str(comp)]
    code = ar.main(args + list(argv))
    return code, capsys.readouterr().out


def test_main_reports_the_exact_single_label_fraction(monkeypatch, tmp_path, capsys):
    df = _frame(single=100, multi=900, zero=0)
    code, out = _run(monkeypatch, tmp_path, df, [], capsys)
    assert code == 0
    assert "f (single-label fraction) = 0.100000" in out
    assert "single-label (exactly 1) :       100" in out


def test_main_says_above_threshold_when_f_is_high(monkeypatch, tmp_path, capsys):
    df = _frame(single=100, multi=900, zero=0)  # f = 0.10 >= 0.0728
    _, out = _run(monkeypatch, tmp_path, df, [], capsys)
    assert "ABOVE the 7.28% threshold" in out


def test_main_says_below_threshold_when_f_is_low(monkeypatch, tmp_path, capsys):
    df = _frame(single=50, multi=950, zero=0)  # f = 0.05 < 0.0728
    _, out = _run(monkeypatch, tmp_path, df, [], capsys)
    assert "BELOW the 7.28% threshold" in out


def test_main_counts_zero_label_patches_separately(monkeypatch, tmp_path, capsys):
    df = _frame(single=100, multi=100, zero=300)
    _, out = _run(monkeypatch, tmp_path, df, [], capsys)
    assert "no label at all          :       300" in out


def test_main_exits_4_when_the_encoding_is_ambiguous(monkeypatch, tmp_path, capsys):
    df = pd.DataFrame(
        {"split": ["train"] * 4, "labels": ["Arable land", "Pastures", "Marine waters", "Inland waters"]}
    )
    code, out = _run(monkeypatch, tmp_path, df, [], capsys)
    assert code == 4
    assert "REFUSING TO GUESS" in out
    assert "Nothing was counted" in out


def test_main_honours_an_explicit_encoding_override(monkeypatch, tmp_path, capsys):
    """The same ambiguous column is countable once the encoding is stated."""
    df = pd.DataFrame(
        {
            "split": ["train"] * 4,
            "labels": ["Arable land Pastures", "Pastures", "Inland waters", "Marine waters"],
        }
    )
    code, out = _run(
        monkeypatch,
        tmp_path,
        df,
        ["--label-encoding", "string_separated", "--label-separator", " "],
        capsys,
    )
    assert code == 0
    assert "encoding : string_separated" in out


def test_main_refuses_separated_encoding_without_a_separator(
    monkeypatch, tmp_path, capsys
):
    df = pd.DataFrame({"split": ["train"], "labels": ["a,b"]})
    code, out = _run(
        monkeypatch, tmp_path, df, ["--label-encoding", "string_separated"], capsys
    )
    assert code == 4
    assert "requires --label-separator" in out


def test_main_flags_metadata_below_the_official_totals(
    monkeypatch, tmp_path, capsys
):
    """A published total the metadata does not reach is flagged as a MISMATCH.

    The comparison runs against the CLEAN subset (the population the published
    table describes), so a 100-assignment train split is far below 688,603.
    """
    df = _frame(single=100, multi=0, zero=0)  # 100 assignments vs 688,603
    _, out = _run(monkeypatch, tmp_path, df, [], capsys)
    assert "MISMATCH" in out
    assert "688,603" in out


def test_main_warns_that_index_encoding_is_a_tautology(
    monkeypatch, tmp_path, capsys
):
    df = pd.DataFrame({"split": ["train"] * 5, "labels": [3, 7, 12, 0, 5]})
    code, out = _run(monkeypatch, tmp_path, df, [], capsys)
    assert code == 0
    assert "tautology, not a measurement" in out


def test_main_exits_2_when_the_file_is_missing(tmp_path, capsys):
    missing = tmp_path / "nope.parquet"
    code = ar.main(["--metadata", str(missing)])
    assert code == 2
    assert "does not exist" in capsys.readouterr().out


def test_main_exits_3_when_the_columns_cannot_be_identified(
    monkeypatch, tmp_path, capsys
):
    df = pd.DataFrame({"alpha": [1, 2], "beta": [3, 4]})
    code, out = _run(monkeypatch, tmp_path, df, [], capsys)
    assert code == 3
    assert "Could not identify" in out


def test_main_accepts_explicit_column_names(monkeypatch, tmp_path, capsys):
    df = pd.DataFrame(
        {"my_split": ["train"], "my_labels": [[CLC[0], CLC[1]]]}
    )
    code, out = _run(
        monkeypatch,
        tmp_path,
        df,
        ["--split-column", "my_split", "--label-column", "my_labels"],
        capsys,
    )
    assert code == 0
    assert "label column : my_labels" in out


# ---------------------------------------------------------------------------
# 4b. schema awareness: the real parquet layout
# ---------------------------------------------------------------------------
#
# Established from ConfigILM's own loader (configilm/extra/BENv2_utils.py and
# BENv2_DataSet.py), which reads:
#     metadata["labels"]  -> iterable of class-name strings
#     metadata["split"]   -> "train" / "val" / "test"
#     metadata["patch_id"], metadata["contains_cloud_or_shadow"],
#     metadata["contains_seasonal_snow"]
# and which CONCATENATES metadata_for_patches_with_snow_cloud_or_shadow.parquet
# because metadata.parquet excludes the snow/cloud/shadow patches.

def test_analyser_refuses_a_label_column_that_is_not_the_clc19_vocabulary(
    monkeypatch, tmp_path, capsys
):
    """Counting a plausible-looking but WRONG column would be worse than failing."""
    df = pd.DataFrame(
        {
            "split": ["train"] * 3,
            "labels": [["not_a_class"], ["also_not"], ["nope"]],
        }
    )
    code, out = _run(monkeypatch, tmp_path, df, [], capsys)
    assert code == 3
    assert "not one sampled value is a reBEN class name" in out


def test_analyser_reports_how_many_sampled_values_are_recognised(
    monkeypatch, tmp_path, capsys
):
    df = _frame(single=10, multi=10, zero=0)
    _, out = _run(monkeypatch, tmp_path, df, [], capsys)
    assert "recognised as a CLC19 class" in out


def test_warns_loudly_when_no_companion_file_is_given(
    monkeypatch, tmp_path, capsys
):
    """metadata.parquet excludes snow/cloud patches; silence here would mislead."""
    df = _frame(single=10, multi=10, zero=0)
    _, out = _run(monkeypatch, tmp_path, df, [], capsys)
    assert "No companion file given" in out
    assert "CLEAN SUBSET ONLY" in out


def test_companion_file_is_concatenated_into_the_counts(
    monkeypatch, tmp_path, capsys
):
    primary = _frame(single=10, multi=10, zero=0)
    companion = _frame(single=30, multi=0, zero=0)
    code, out = _run(monkeypatch, tmp_path, primary, [], capsys, companion=companion)
    assert code == 0
    assert "combined : rows 50" in out
    assert "single-label (exactly 1) :        40" in out
    assert "No companion file given" not in out


def test_reports_the_clean_subset_fraction_separately(
    monkeypatch, tmp_path, capsys
):
    df = _frame(single=40, multi=60, zero=0, cloudy=50)
    _, out = _run(monkeypatch, tmp_path, df, [], capsys)
    assert "restricted to the CLEAN rows" in out
    assert "f on the clean subset" in out


def test_reports_the_snow_cloud_flag_breakdown(monkeypatch, tmp_path, capsys):
    df = _frame(single=40, multi=60, zero=0, cloudy=50, snowy=25)
    _, out = _run(monkeypatch, tmp_path, df, [], capsys)
    assert "contains_cloud_or_shadow" in out
    assert "contains_seasonal_snow" in out
    assert "clean (neither flag)" in out


def test_clean_subset_can_be_suppressed(monkeypatch, tmp_path, capsys):
    df = _frame(single=40, multi=60, zero=0, cloudy=50)
    _, out = _run(monkeypatch, tmp_path, df, ["--no-clean-filter"], capsys)
    assert "restricted to the CLEAN rows" not in out


def test_the_canonical_clc19_vocabulary_is_complete_and_matches_the_pdf_table():
    """The 19 names must match the ones in the recovered official table."""
    assert len(ar.CLC19_LABELS) == 19
    assert len(set(ar.CLC19_LABELS)) == 19
    for name in (
        "Urban fabric",
        "Industrial or commercial units",
        "Arable land",
        "Beaches, dunes, sands",
        "Coastal wetlands",
        "Marine waters",
    ):
        assert name in ar.CLC19_LABELS
    # the longest name is the one most likely to be mangled by a bad extraction
    assert (
        "Land principally occupied by agriculture, with significant areas "
        "of natural vegetation" in ar.CLC19_LABELS
    )


# ---------------------------------------------------------------------------
# 4c. THE CLASS-INDEX ORDER — the silent-mis-mapping risk
# ---------------------------------------------------------------------------
#
# A 19-class softmax has no idea what its outputs mean; the meaning comes entirely
# from the ORDER of the vocabulary. Two orders are in circulation:
#
#   * the ORIGINAL CLC order (the order the classes appear in the official
#     BigEarthNet documents) — what this repo uses;
#   * the LEXICOGRAPHIC order — what ConfigILM's `ben_19_labels_to_multi_hot`
#     uses BY DEFAULT (`lex_sorted=True`), and therefore what a great deal of
#     published reBEN baseline code produces.
#
# They are not the same order: 18 of the 19 classes sit at a different index.
# Nothing in the repo previously pinned the order, so a reordering of the
# vocabulary would have changed what every class index MEANS while every
# existing test still passed.

#: The order, written out literally. If this test fails, a class index has changed
#: meaning and every previously-computed result is invalid until re-derived.
EXPECTED_CLC19_ORDER = (
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of "
    "natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
)


def test_clc19_order_is_pinned_exactly():
    """Pin the ORDER, not just the membership.

    A set-equality test would pass under any permutation. This one does not.
    """
    assert tuple(ar.CLC19_LABELS) == EXPECTED_CLC19_ORDER


def test_analyser_vocabulary_matches_the_repo_canonical_constant():
    """The analyser keeps its own copy; drift between copies must fail here."""
    from training.data.bigearthnet import CLC19_CLASSES

    assert tuple(ar.CLC19_LABELS) == tuple(CLC19_CLASSES)


def test_extractor_vocabulary_matches_the_repo_canonical_constant():
    """The extractor maps label names to head indices, so its order is load-bearing."""
    from training.data.bigearthnet import CLC19_CLASSES
    from training.fusion.extract import CLC19_CLASSES as EXTRACT_VOCAB

    assert tuple(EXTRACT_VOCAB) == tuple(CLC19_CLASSES)


def test_label_index_maps_names_to_the_documented_indices():
    """Pin concrete name -> index pairs, the thing the head actually emits."""
    from training.fusion.extract import label_index

    assert label_index("Urban fabric") == 0
    assert label_index("Arable land") == 2
    assert label_index("Complex cultivation patterns") == 5
    assert label_index("Marine waters") == 18
    # the long name must survive its line-continuation in the source constant
    assert label_index(
        "Land principally occupied by agriculture, with significant areas "
        "of natural vegetation"
    ) == 6


def test_an_unknown_class_name_is_refused_not_defaulted():
    from training.fusion.extract import label_index

    with pytest.raises(Exception):
        label_index("Not a CLC class")


def test_lexicographic_order_is_a_different_order_and_18_of_19_classes_move():
    """The guard that stops the two orders being treated as interchangeable.

    If someone "fixes" the vocabulary to match ConfigILM's default, this test
    still passes — but `test_clc19_order_is_pinned_exactly` fails, which is the
    point: the change cannot happen silently.
    """
    repo_order = tuple(ar.CLC19_LABELS)
    lex_order = tuple(sorted(repo_order))

    assert repo_order != lex_order, (
        "the vocabulary is now lexicographic; if that was deliberate, the "
        "documented class order and any cross-reference to published reBEN "
        "results must be updated together"
    )

    moved = [i for i, name in enumerate(repo_order) if lex_order.index(name) != i]
    assert len(moved) == 18, (
        f"expected 18 of 19 classes to move under lexicographic ordering, "
        f"got {len(moved)}"
    )

    # the one class that happens to coincide, asserted explicitly so a future
    # reordering cannot quietly change the count without failing here
    coincide = [name for i, name in enumerate(repo_order) if lex_order.index(name) == i]
    assert coincide == ["Complex cultivation patterns"]


# ---------------------------------------------------------------------------
# 5. non-vacuity: the original counter must get every pinned case WRONG
# ---------------------------------------------------------------------------

def _naive_counter(value) -> int:
    """The analyser's original rule: a cell is one label unless it is a sequence.

    This is the implementation the tests above are written against. It is kept
    here ONLY so that the non-vacuity test can prove it fails.
    """
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        return len(value)
    if hasattr(value, "__len__") and not isinstance(value, str):
        return len(value)
    return 1 if str(value).strip() else 0


def test_the_naive_counter_disagrees_with_every_discriminating_case():
    """Non-vacuity guard.

    If the analyser is reverted to the naive counter, this test fails, and so do
    the counting tests above. That is the point: the guards are load-bearing.

    The cases are split into two groups on purpose:

      * DISCRIMINATING - the naive counter gets these WRONG, so they are the ones
        that actually distinguish the two implementations;
      * BENIGN - the naive counter gets these RIGHT (`[]` is 0 labels under both
        rules), so they prove the test is not merely asserting "naive is always
        wrong", which would be a vacuous guard in the opposite direction.
    """
    discriminating = [
        ("one-hot 19-wide, 2 set bits", ONEHOT_2, 2),
        ("one-hot 19-wide, 0 set bits", [0] * 19, 0),
        ("NaN", float("nan"), 0),
        ("separated string of 2", "Arable land, Pastures", 2),
        ("sequence repr of 2", "['a', 'b']", 2),
    ]
    benign = [
        ("empty list", [], 0),
        ("list of 2", ["a", "b"], 2),
        ("None", None, 0),
    ]

    for name, value, truth in discriminating:
        assert _naive_counter(value) != truth, (
            f"{name}: the naive counter was RIGHT, so this case does not "
            f"discriminate and the guard is weaker than it claims"
        )
        enc, sep = _encoding_for(value)
        assert ar.count_labels(value, enc, sep) == truth, (
            f"{name}: the analyser disagrees with the truth"
        )

    for name, value, truth in benign:
        assert _naive_counter(value) == truth, (
            f"{name}: expected the naive counter to agree here; if it does not, "
            f"move this case into the discriminating group"
        )
        enc, sep = _encoding_for(value)
        assert ar.count_labels(value, enc, sep) == truth

    assert len(discriminating) >= 5, (
        "too few discriminating cases: the guard would not catch a revert"
    )


def _encoding_for(value) -> tuple[str, str | None]:
    """The (encoding, separator) the analyser should infer for a pinned value."""
    if isinstance(value, str):
        if value.strip().startswith("["):
            return "sequence_repr", None
        return "string_separated", ","
    items = value if isinstance(value, list) else []
    if len(items) == 19:
        return "onehot", None
    return "sequence", None


def test_the_naive_counter_would_have_wrongly_closed_option_b():
    """A one-hot column makes the naive counter report 19 labels per patch, so
    `n_labels == 1` is never true and the single-label fraction reads as 0 --
    silently closing option (b)."""
    df = pd.DataFrame({"labels": [ONEHOT_2, ONEHOT_1, ONEHOT_1]})
    naive_single = sum(1 for v in df["labels"] if _naive_counter(v) == 1)
    real_single = sum(1 for v in df["labels"] if ar.count_labels(v, "onehot") == 1)
    assert naive_single == 0
    assert real_single == 2


def test_the_naive_counter_would_have_wrongly_opened_option_b():
    """A separated-string column makes the naive counter report 1 label for a
    two-class patch, inflating the single-label fraction."""
    value = "Arable land, Pastures"
    assert _naive_counter(value) == 1
    assert ar.count_labels(value, "string_separated", ",") == 2


# ---------------------------------------------------------------------------
# 6. The published-table cross-check (three real defects, found against the
#    actual reBEN metadata)
#
#    All three were invisible in a green suite because nothing pinned the
#    split vocabulary, the population the published table describes, or the
#    fact that the vocabulary scan was reading a head() sample.
# ---------------------------------------------------------------------------


def test_pdf_totals_keys_match_the_split_names_the_metadata_uses() -> None:
    """DEFECT 1 — the table was keyed on `val`; the metadata says `validation`.

    `check`-style lookups then matched NO rows, so the cross-check compared a
    real published total against a sum of ZERO and printed a delta of -368,377
    on data that agrees exactly. The key must be the value that is in the file.
    """
    assert set(ar.PDF_TOTALS) == {"train", "validation", "test"}
    assert "val" not in ar.PDF_TOTALS


def test_the_published_totals_are_the_clean_subset_total_not_the_corpus_total() -> None:
    """DEFECT 2 — the published table describes the snow/cloud-FREE subset.

    Its caption says so. The consequence is arithmetic: 1,415,857 is the clean
    subset's assignment total over 480,038 patches (mean 2.9495). The full
    corpus is 1,623,173 over 549,488 patches (mean 2.9540). Dividing the CLEAN
    total by the FULL patch count gives 2.5767 -- the figure this project
    previously recorded, and one that describes no population that exists.
    """
    assert sum(ar.PDF_TOTALS.values()) == 1_415_857
    assert ar.CORPUS_PATCHES == 549_488
    assert sum(ar.PDF_TOTALS.values()) != 1_623_173
    # The population mismatch, stated as arithmetic so it cannot be re-derived.
    assert round(1_415_857 / 549_488, 4) == 2.5767
    assert round(1_415_857 / 480_038, 4) == 2.9495
    assert round(1_623_173 / 549_488, 4) == 2.9540
    assert "snow" in ar.PDF_TOTALS_POPULATION.lower()


def test_section_5_compares_against_the_clean_subset_not_the_combined_frame(
    monkeypatch, tmp_path, capsys
) -> None:
    """The comparison must use the population the published table describes.

    Fixture: 3 clean single-label train patches plus 2 flagged multi-label train
    patches. Clean assignments = 3; combined assignments = 3 + 4 = 7. The
    published total is set to the CLEAN value, so a correct comparison gives
    delta 0 and the control below shows the wrong frame would not.
    """
    primary = _frame(single=3, multi=0, zero=0, split="train")
    companion = _frame(single=0, multi=2, zero=0, split="train", cloudy=2)
    monkeypatch.setattr(
        ar, "PDF_TOTALS", {"train": 3, "validation": 0, "test": 0}
    )
    code, out = _run(monkeypatch, tmp_path, primary, [], capsys, companion=companion)
    assert code == 0
    assert "clean subset" in out
    assert "MISMATCH" not in out


def test_section_5_would_flag_a_mismatch_against_the_combined_frame(
    monkeypatch, tmp_path, capsys
) -> None:
    """The control for the test above: same fixture, published total set to the
    COMBINED value (7). If the analyser compared against the combined frame it
    would agree, so a MISMATCH here proves it is really using the clean subset."""
    primary = _frame(single=3, multi=0, zero=0, split="train")
    companion = _frame(single=0, multi=2, zero=0, split="train", cloudy=2)
    monkeypatch.setattr(
        ar, "PDF_TOTALS", {"train": 7, "validation": 0, "test": 0}
    )
    code, out = _run(monkeypatch, tmp_path, primary, [], capsys, companion=companion)
    assert code == 0
    assert "MISMATCH" in out


def test_section_5_reports_a_split_the_published_table_does_not_name(
    monkeypatch, tmp_path, capsys
) -> None:
    """DEFECT 1's general form: a split value the published table does not name
    is what let a real total be compared against a sum of zero. It must be
    reported, and a split the data lacks must be shown as not compared rather
    than as a deficit."""
    # The data says `validation`; the published table says `val`. This is the
    # exact historical pairing, reproduced deliberately.
    primary = _frame(single=2, multi=0, zero=0, split="validation")
    companion = _frame(single=2, multi=0, zero=0, split="validation", cloudy=2)
    monkeypatch.setattr(ar, "PDF_TOTALS", {"train": 0, "val": 2, "test": 0})

    code, out = _run(monkeypatch, tmp_path, primary, [], capsys, companion=companion)

    assert code == 0
    assert "SPLIT NAME MISMATCH" in out
    assert "validation" in out
    # Every published split is absent from the data, so nothing is compared and
    # no delta is reported -- in particular not a delta against zero.
    assert "not compared" in out
    assert "<-- MISMATCH" not in out


def test_vocabulary_scan_reads_the_whole_column_not_a_head_sample(
    monkeypatch, tmp_path, capsys
) -> None:
    """DEFECT 3 — the scan used head(20_000) on a parquet ordered by tile.

    The first 20,000 rows are a geographically clustered block, not a sample:
    on the real corpus the scan reported "18 of 19" for a corpus that contains
    all 19. Here the 19th class appears ONLY after row 20,000.
    """
    rows = [{"split": "train", "labels": [CLC[i % 18]]} for i in range(25_000)]
    rows.append({"split": "train", "labels": [CLC[18]]})
    df = pd.DataFrame(rows)
    df["contains_cloud_or_shadow"] = False
    df["contains_seasonal_snow"] = False

    # The fixture really does discriminate: a head(20_000) sample sees only 18.
    head_classes = {CLC[i % 18] for i in range(20_000)}
    assert len(head_classes) == 18

    code, out = _run(monkeypatch, tmp_path, df, [], capsys)
    assert code == 0
    assert "19 of 19" in out
