"""Establish the reBEN label situation from the official metadata table.

Run this the moment `metadata.parquet` is on disk. It answers, in one pass, the
questions the Phase 12 label decision turns on:

  1. the exact official patch count in each split (train / val / test)
  2. the distribution of labels per patch
  3. the exact SINGLE-LABEL fraction  <- the number option (b) lives or dies on
  4. the per-split single-label counts, so a single-label slice can be sized
  5. a cross-check of the totals against the official class-distribution document

Requires `pyarrow` (or `fastparquet`). If neither is present it says so and exits
non-zero rather than guessing — a wrong number here would silently mis-set the
whole experiment.

THE LABEL ENCODING IS DETECTED, NOT ASSUMED
-------------------------------------------
"Number of labels on a patch" is not a property of the value alone; it depends on
how the column encodes labels. The same cell content means different things under
different encodings:

    ['Arable land', 'Pastures']      -> 2 labels   (sequence)
    "Arable land, Pastures"          -> 2 labels   (string, separated)
    "Arable land Pastures"           -> ???        (space-separated, OR one label
                                                    whose name contains a space)
    [1,0,1,0,...]  (19 wide)         -> 2 labels   (one-hot)
    5                                -> ???        (bitmask? class index?)
    nan                              -> 0 labels   (missing)

A counter that treats every value as "1 label" inflates the single-label fraction
and would wrongly open option (b); one that returns the width of a one-hot vector
returns 19 for every patch, making the single-label fraction zero and wrongly
closing option (b). Both are silent.

So this script detects the encoding, prints the evidence for its detection, and
REFUSES (exit 4) when the encoding is ambiguous instead of picking one. Pass
--label-encoding to state it explicitly.

Read-only. Writes nothing.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

#: Label-assignment totals from the official split document, Table 1.
#: https://bigearth.net/static/documents/BigEarthNet_v2_Split.pdf
#: Summed over the 19 classes, transcribed from the table.
#:
#: THE POPULATION MATTERS. The table's own caption reads "the number of image
#: pairs of each land use land cover (LULC) class that is NOT COVERED BY SNOW OR
#: CLOUDS in training, validation, and test sets for the reBEN dataset". These
#: are therefore totals for the SNOW/CLOUD-FREE SUBSET (480,038 patches), NOT
#: for the full 549,488-patch corpus. Verified exactly: all 19 per-class values
#: and all three per-split totals reproduce the clean subset bit for bit.
#:
#: Dividing these by the FULL patch count is a population mismatch and gives
#: 2.5767 labels/patch, a figure that describes no population that exists.
#: The correct means are 2.9495 (clean subset) and 2.9540 (full corpus).
#:
#: Keys are the split values AS THEY APPEAR IN THE METADATA. The metadata says
#: `validation`; an earlier revision of this constant said `val`, which silently
#: matched no rows and printed a delta of -368,377 against a sum of zero.
PDF_TOTALS = {"train": 688_603, "validation": 368_377, "test": 358_877}
#: The population `PDF_TOTALS` describes. The cross-check below is only valid
#: when it compares against THIS subset.
PDF_TOTALS_POPULATION = "snow/cloud-free subset (480,038 patches)"
CORPUS_PATCHES = 549_488
NUM_CLASSES = 19

#: The 19 CLC2018 v2020_u1 class names, verbatim from ConfigILM's
#: `configilm/extra/BENv2_utils.py` (`NEW_LABELS_ORIGINAL_ORDER`). Used to check
#: that the column we counted really holds reBEN labels, so that a wrong column or
#: a wrong vocabulary is caught rather than silently counted.
CLC19_LABELS = (
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
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

#: Columns the official loader uses, so we can report what it would keep.
PATCH_ID_COLUMNS = ("patch_id", "s1_name")
SNOW_CLOUD_FLAGS = ("contains_cloud_or_shadow", "contains_seasonal_snow")

SPLIT_CANDIDATES = ("split", "official_split", "reben_split", "ben_split")
LABEL_CANDIDATES = ("labels", "BEN-19", "BEN_19", "labels_19", "label")

#: Encodings this script can count without guessing.
ENCODINGS = (
    "auto",
    "sequence",        # list / tuple / set / ndarray of class names or ids
    "onehot",          # fixed-width 0/1 vector, one slot per class
    "sequence_repr",   # a string holding the repr of a sequence
    "string_separated",  # one string, labels split on a separator
    "index",           # a single class index -> exactly one label, by construction
)

EXIT_NO_ENGINE = 2
EXIT_BAD_COLUMNS = 3
EXIT_AMBIGUOUS_ENCODING = 4


class LabelEncodingError(Exception):
    """The label column's encoding cannot be determined without guessing."""


# --------------------------------------------------------------------------
# value helpers
# --------------------------------------------------------------------------

def _as_list(value):
    """Return a plain list for anything sequence-like, else None."""
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        out = tolist()
        return out if isinstance(out, list) else [out]
    return None


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        import math

        return isinstance(value, float) and math.isnan(value)
    except Exception:
        return False


def _is_boolish(value) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return value in (0, 1)
    if isinstance(value, float):
        return value in (0.0, 1.0)
    return False


def _looks_like_sequence_repr(text: str) -> bool:
    stripped = text.strip()
    if not (stripped.startswith("[") or stripped.startswith("(")):
        return False
    try:
        parsed = ast.literal_eval(stripped)
    except Exception:
        return False
    return isinstance(parsed, (list, tuple, set))


# --------------------------------------------------------------------------
# encoding detection
# --------------------------------------------------------------------------

def detect_encoding(values) -> tuple[str, str]:
    """Return (encoding, evidence). Raises LabelEncodingError when ambiguous."""
    sample = [v for v in values if not _is_missing(v)]
    if not sample:
        raise LabelEncodingError(
            "every label cell is missing (None/NaN); there is nothing to count"
        )

    seqs = [_as_list(v) for v in sample]
    if all(s is not None for s in seqs):
        widths = {len(s) for s in seqs}
        flat = [x for s in seqs for x in s]
        boolish = all(_is_boolish(x) for x in flat)
        if widths == {NUM_CLASSES} and boolish:
            return (
                "onehot",
                f"every value is a {NUM_CLASSES}-wide 0/1 vector "
                f"(n={len(sample)} sampled)",
            )
        return (
            "sequence",
            f"every value is a sequence of {sorted(widths)} element(s) "
            f"(n={len(sample)} sampled)",
        )

    if all(isinstance(v, str) for v in sample):
        if all(_looks_like_sequence_repr(v) for v in sample):
            return "sequence_repr", "every value is the repr of a list/tuple"
        seps = [sep for sep in (",", ";", "|") if any(sep in v for v in sample)]
        if seps:
            return (
                "string_separated",
                f"every value is a string containing {seps!r}",
            )
        raise LabelEncodingError(
            "every value is a bare string with no separator. This is genuinely "
            "ambiguous: 'Arable land Pastures' is either two class names separated "
            "by a space, or one class name that contains a space. Counting it as 1 "
            "would inflate the single-label fraction and could wrongly open "
            "option (b). Pass --label-encoding string_separated --label-separator ' ' "
            "if the corpus uses space separation, or --label-encoding sequence if it "
            "does not."
        )

    if all(isinstance(v, bool) for v in sample):
        raise LabelEncodingError(
            "every value is a bare bool; a single boolean column cannot carry a "
            "19-class label. Inspect the parquet header."
        )

    if all(isinstance(v, int) and not isinstance(v, bool) for v in sample):
        distinct = sorted(set(sample))
        if set(distinct) <= {0, 1}:
            raise LabelEncodingError(
                "every value is an integer in {0, 1}. This is ambiguous: it could "
                "be a bitmask, or a per-class flag column. Pass --label-encoding "
                "index if it is a class index, or inspect the parquet header."
            )
        if set(distinct) <= set(range(NUM_CLASSES)):
            return (
                "index",
                f"every value is an integer in [0,{NUM_CLASSES}); "
                f"{len(distinct)} distinct value(s)",
            )
        raise LabelEncodingError(
            f"integer values outside [0,{NUM_CLASSES}) were found "
            f"(min={min(distinct)}, max={max(distinct)}); not a class index"
        )

    kinds = Counter(type(v).__name__ for v in sample)
    raise LabelEncodingError(f"mixed or unrecognised value types: {dict(kinds)}")


def count_labels(value, encoding: str, separator: str | None = None) -> int:
    """Count labels on one cell, under a KNOWN encoding."""
    if _is_missing(value):
        return 0

    if encoding == "index":
        return 1

    if encoding in ("sequence", "onehot"):
        items = _as_list(value)
        if items is None:
            raise LabelEncodingError(
                f"encoding is {encoding!r} but a value is {type(value).__name__}"
            )
        if encoding == "onehot":
            return sum(1 for x in items if x)
        return len(items)

    if encoding == "sequence_repr":
        if not isinstance(value, str):
            raise LabelEncodingError("encoding is 'sequence_repr' but value is not str")
        parsed = ast.literal_eval(value.strip())
        return len(parsed)

    if encoding == "string_separated":
        if not isinstance(value, str):
            raise LabelEncodingError(
                "encoding is 'string_separated' but value is not str"
            )
        if not separator:
            raise LabelEncodingError("'string_separated' needs --label-separator")
        return sum(1 for part in value.split(separator) if part.strip())

    raise LabelEncodingError(f"unknown encoding {encoding!r}")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_table(path: Path):
    try:
        import pyarrow.parquet as pq

        return pq.read_table(str(path)).to_pandas()
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover - engine present but file bad
        print(f"ERROR: pyarrow present but failed to read the file: {exc}")
        raise SystemExit(EXIT_NO_ENGINE)

    try:
        import pandas as pd

        return pd.read_parquet(str(path))
    except Exception as exc:
        print("ERROR: no parquet engine available.")
        print(f"  pyarrow: missing   fastparquet: missing   ({type(exc).__name__})")
        print("  Install one, e.g.:  .venv/Scripts/python.exe -m pip install pyarrow")
        raise SystemExit(EXIT_NO_ENGINE)


def pick_column(df, candidates, kind: str):
    for name in candidates:
        if name in df.columns:
            return name
    return None


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="path to metadata.parquet")
    ap.add_argument(
        "--metadata-snow-cloud",
        default=None,
        help="path to metadata_for_patches_with_snow_cloud_or_shadow.parquet. "
        "metadata.parquet EXCLUDES the snow/cloud/shadow patches, so without this "
        "the counts describe the clean subset only, not the full corpus.",
    )
    ap.add_argument(
        "--no-clean-filter",
        action="store_true",
        help="do not also report the subset the official loader would keep "
        "(rows with neither contains_cloud_or_shadow nor contains_seasonal_snow)",
    )
    ap.add_argument(
        "--label-encoding",
        choices=ENCODINGS,
        default="auto",
        help="how the label column encodes labels; 'auto' detects and refuses "
        "to guess when ambiguous",
    )
    ap.add_argument(
        "--label-separator",
        default=None,
        help="separator for --label-encoding string_separated",
    )
    ap.add_argument(
        "--label-column", default=None, help="override the label column name"
    )
    ap.add_argument(
        "--split-column", default=None, help="override the split column name"
    )
    args = ap.parse_args(argv)

    import pandas as pd  # local: the pure helpers stay importable without it

    path = Path(args.metadata)
    if not path.exists():
        print(f"ERROR: {path} does not exist")
        return EXIT_NO_ENGINE

    primary = load_table(path)
    companion = None
    companion_path = None
    if args.metadata_snow_cloud:
        companion_path = Path(args.metadata_snow_cloud)
        if not companion_path.exists():
            print(f"ERROR: {companion_path} does not exist")
            return EXIT_NO_ENGINE
        companion = load_table(companion_path)

    print("=" * 78)
    print("reBEN OFFICIAL METADATA — label situation")
    print("=" * 78)

    # ---- 0. source --------------------------------------------------------
    print("=" * 78)
    print("SOURCE")
    print("=" * 78)
    print(f"  primary  : {path.name}  rows {len(primary):,}")
    if companion is not None:
        print(f"  companion: {companion_path.name}  rows {len(companion):,}")
        df = pd.concat([primary, companion], ignore_index=True)
        print(f"  combined : rows {len(df):,}")
    else:
        df = primary
        print()
        print("  !! No companion file given. `metadata.parquet` EXCLUDES the patches")
        print("     covered by snow, cloud or shadow; those live in")
        print("     metadata_for_patches_with_snow_cloud_or_shadow.parquet.")
        print("     Everything below therefore describes the CLEAN SUBSET ONLY,")
        print(f"     not the {CORPUS_PATCHES:,}-patch corpus.")
    print()
    print(f"  columns : {list(primary.columns)}")
    print()

    # flag breakdown, so the reader can see what the official loader would keep
    flags_present = [f for f in SNOW_CLOUD_FLAGS if f in df.columns]
    clean_mask = None
    if flags_present:
        clean_mask = pd.Series(True, index=df.index)
        for flag in flags_present:
            clean_mask &= ~df[flag].astype(bool)
        print(f"  snow/cloud flags present: {flags_present}")
        for flag in flags_present:
            print(f"    {flag:<28} True: {int(df[flag].astype(bool).sum()):>9,}")
        print(f"    clean (neither flag)          : {int(clean_mask.sum()):>9,}")
        print()
        print("  The official dataloader keeps only the clean rows unless told")
        print("  otherwise (`include_cloudy` / `include_snowy`).")
    else:
        print("  snow/cloud flag columns not present in this file.")
    print()
    print(f"  label column candidates present: "
          f"{[c for c in LABEL_CANDIDATES if c in df.columns]}")
    print(f"  split column candidates present: "
          f"{[c for c in SPLIT_CANDIDATES if c in df.columns]}")
    print()

    split_col = args.split_column or pick_column(df, SPLIT_CANDIDATES, "split")
    label_col = args.label_column or pick_column(df, LABEL_CANDIDATES, "label")
    if split_col is None or label_col is None:
        print(
            f"Could not identify the split/label columns "
            f"(split={split_col!r}, label={label_col!r}). "
            f"Inspect the header above and pass --split-column / --label-column."
        )
        return EXIT_BAD_COLUMNS

    print(f"split column : {split_col}")
    print(f"label column : {label_col}")
    print()

    # ---- encoding ---------------------------------------------------------
    encoding = args.label_encoding
    evidence = "stated explicitly on the command line"
    if encoding == "auto":
        try:
            encoding, evidence = detect_encoding(df[label_col].tolist())
        except LabelEncodingError as exc:
            print("=" * 78)
            print("REFUSING TO GUESS — label encoding is ambiguous")
            print("=" * 78)
            print(f"  {exc}")
            print()
            print("  Nothing was counted. A wrong count here silently mis-sets the")
            print("  experiment, so this script does not pick an encoding for you.")
            print()
            return EXIT_AMBIGUOUS_ENCODING

    print("=" * 78)
    print("LABEL ENCODING")
    print("=" * 78)
    print(f"  encoding : {encoding}")
    print(f"  basis    : {evidence}")
    if encoding == "string_separated":
        if not args.label_separator:
            print()
            print("  REFUSING: 'string_separated' requires --label-separator.")
            return EXIT_AMBIGUOUS_ENCODING
        print(f"  separator: {args.label_separator!r}")
    if encoding == "index":
        print(
            "  NOTE: an 'index' encoding means every patch carries exactly one\n"
            "        class BY CONSTRUCTION. If the corpus is multi-label, this is\n"
            "        the wrong encoding and the single-label fraction below is a\n"
            "        tautology, not a measurement."
        )

    # vocabulary check: a plausible count of the WRONG column would be worse than
    # no count, so confirm the values really look like reBEN class names.
    if encoding in ("sequence", "onehot", "sequence_repr", "string_separated"):
        # Scan the WHOLE column, not a head() sample. The parquet is ordered by
        # tile, so the first 20,000 rows are a geographically clustered block,
        # not a random sample: scanning them reported "18 of 19 classes" on a
        # corpus that in fact contains all 19. A vocabulary check that can miss
        # a class is worse than none, because it looks like a finding.
        seen: set[str] = set()
        for value in df[label_col]:
            if _is_missing(value):
                continue
            if encoding == "onehot":
                continue
            items = _as_list(value)
            if items is None and encoding == "sequence_repr":
                try:
                    items = list(ast.literal_eval(str(value).strip()))
                except Exception:
                    items = None
            if items is None and encoding == "string_separated":
                items = [p for p in str(value).split(args.label_separator) if p.strip()]
            if items is None:
                continue
            seen.update(str(x) for x in items)
        if seen:
            known = {x for x in seen if x in CLC19_LABELS}
            unknown = sorted(seen - known)
            missing_classes = [x for x in CLC19_LABELS if x not in seen]
            print(f"  distinct label values found   : {len(seen)}")
            print(f"  recognised as a CLC19 class   : {len(known)} of {NUM_CLASSES}")
            if missing_classes:
                print(f"  CLC19 classes NOT present     : {len(missing_classes)}")
                for name in missing_classes[:5]:
                    print(f"      absent: {name}")
            if unknown:
                shown = unknown[:5]
                print(f"  NOT recognised ({len(unknown)})       : {shown}"
                      f"{' ...' if len(unknown) > 5 else ''}")
                if not known:
                    print()
                    print("  REFUSING: not one sampled value is a reBEN class name.")
                    print("  This is almost certainly the wrong column.")
                    print()
                    return EXIT_BAD_COLUMNS
    print()

    df = df.copy()
    df["_n"] = [
        count_labels(v, encoding, args.label_separator) for v in df[label_col]
    ]

    # ---- 1. counts --------------------------------------------------------
    print("=" * 78)
    print("1. EXACT PATCH COUNTS PER SPLIT")
    print("=" * 78)
    counts = df[split_col].value_counts().sort_index()
    for s, n in counts.items():
        print(f"  {s:<8} {n:>9,}")
    print(f"  {'TOTAL':<8} {len(df):>9,}")
    print()

    # ---- 2. distribution --------------------------------------------------
    print("=" * 78)
    print("2. LABELS PER PATCH — distribution")
    print("=" * 78)
    hist = Counter(df["_n"].tolist())
    for k in sorted(hist):
        print(f"  {k:>3} label(s) : {hist[k]:>9,}  ({hist[k] / len(df):6.2%})")
    print(f"\n  mean labels per patch  : {df['_n'].mean():.4f}")
    print(f"  total label assignments: {int(df['_n'].sum()):,}")
    print()

    # ---- 3. the deciding number ------------------------------------------
    print("=" * 78)
    print("3. THE DECIDING NUMBER — single-label fraction")
    print("=" * 78)
    single = int((df["_n"] == 1).sum())
    multi = int((df["_n"] > 1).sum())
    zero = int((df["_n"] == 0).sum())
    f = single / len(df)
    print(f"  single-label (exactly 1) : {single:>9,}  ({f:6.4%})")
    print(f"  multi-label  (>= 2)      : {multi:>9,}  ({multi / len(df):6.4%})")
    print(f"  no label at all          : {zero:>9,}  ({zero / len(df):6.4%})")
    print()
    print(f"  f (single-label fraction) = {f:.6f}")
    print()
    print("  Option (b) reaches a 20,000 / 4,000 / 4,000 slice iff f >= 7.28%.")
    if f >= 0.0728:
        print("  -> f is ABOVE the 7.28% threshold.")
    elif f > 0:
        print("  -> f is BELOW the 7.28% threshold.")
        print(f"     Single-label train patches available: ~{int(274_744 * f):,}")
    else:
        print("  -> f is 0: option (b) has no population at all.")
    print()

    # the population the official loader actually uses, when we can tell
    if clean_mask is not None and not args.no_clean_filter:
        sub = df[clean_mask]
        if len(sub):
            c_single = int((sub["_n"] == 1).sum())
            c_f = c_single / len(sub)
            print("  --- restricted to the CLEAN rows the official loader keeps ---")
            print(f"      (neither contains_cloud_or_shadow nor contains_seasonal_snow)")
            print(f"      patches                  : {len(sub):>9,}")
            print(f"      single-label             : {c_single:>9,}  ({c_f:6.4%})")
            print(f"      f on the clean subset    = {c_f:.6f}")
            print()
            print("      f differs between the full corpus and the clean subset. State")
            print("      which population a result is reported on; they are not the same.")
            print()

    # ---- 4. per split -----------------------------------------------------
    print("=" * 78)
    print("4. SINGLE-LABEL AVAILABILITY PER SPLIT")
    print("=" * 78)
    for s in sorted(counts.index):
        sub = df[df[split_col] == s]
        n1 = int((sub["_n"] == 1).sum())
        print(
            f"  {s:<8} total {len(sub):>9,}   single-label {n1:>9,}  "
            f"({n1 / max(1, len(sub)):6.4%})"
        )
    print()

    # ---- 5. cross-check ---------------------------------------------------
    # The PDF's caption fixes the population: "not covered by snow or clouds".
    # Comparing it against the combined frame was a population mismatch that
    # produced three non-zero deltas on data that in fact agrees exactly.
    print("=" * 78)
    print("5. CROSS-CHECK against the official split document (Table 1)")
    print("=" * 78)
    print(f"  the published table describes : {PDF_TOTALS_POPULATION}")
    observed = sorted(str(value) for value in df[split_col].dropna().unique())
    print(f"  split values in this data     : {observed}")

    # A split value the published table does not name is EXACTLY the condition
    # that made an earlier revision compare a real published total against a
    # sum of zero. Report it; do not refuse, because a partial corpus is a
    # legitimate input and refusing on it would break honest partial runs.
    stray = [value for value in observed if value not in PDF_TOTALS]
    if stray:
        print()
        print(f"  SPLIT NAME MISMATCH: this data uses {stray}, which the published")
        print("  table does not name. A lookup keyed on the table's spelling matches")
        print("  no rows and reports the published total as a deficit against zero.")

    if clean_mask is None or args.no_clean_filter:
        print()
        print("  this run did not build the clean subset, so the published totals")
        print("  CANNOT be compared against it. Supply --metadata-snow-cloud (and")
        print("  drop --no-clean-filter) to make the comparison meaningful.")
        print("=" * 78)
    else:
        cmp_df = df[clean_mask]
        print(f"  this run compares against     : clean subset, "
              f"{len(cmp_df):,} patches")
        print()
        print(
            f"  {'split':<12}{'labels (clean)':>18}{'labels (published)':>20}"
            f"{'delta':>12}"
        )
        for name in sorted(PDF_TOTALS):
            want = PDF_TOTALS[name]
            if name not in observed:
                print(f"  {name:<12}{'not present':>18}{want:>20,}"
                      f"{'not compared':>14}")
                continue
            got = int(cmp_df.loc[cmp_df[split_col] == name, "_n"].sum())
            delta = got - want
            flag = "" if delta == 0 else "  <-- MISMATCH"
            print(f"  {name:<12}{got:>18,}{want:>20,}{delta:>12,}{flag}")
        print()
        print(f"  metadata rows {len(df):,} vs published corpus {CORPUS_PATCHES:,}")
        print()
        print("  Every published total above is a LABEL-ASSIGNMENT count over the")
        print("  snow/cloud-free subset. A mean computed from it describes THAT")
        print("  subset; dividing it by the full-corpus patch count mixes two")
        print("  populations and describes neither.")
        print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
